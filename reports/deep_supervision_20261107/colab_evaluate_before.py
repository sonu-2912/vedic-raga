"""Validation-only calibration, full test evaluation and CPU deployment bundle."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report

from backend.datasets import FER2013
from backend.models import FER_CLASSES
from scripts.colab_common import (check_manifest, device_for, digest, fixed_split,
                                 metrics, model_from, predictions, save_checkpoint, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--baseline-report', default='reports/emotion_evaluation.json')
    p.add_argument('--baseline-manifest', default='reports/emotion_evaluation_manifest.json')
    a = p.parse_args(); torch.set_num_threads(a.threads)
    run = Path(a.run_dir); device = device_for(a.device)
    if (run / 'bundle_manifest.json').exists():
        previous = json.loads((run / 'bundle_manifest.json').read_text())
        if previous['source_best_sha256'] != digest(run / 'best.pt'):
            raise ValueError('Best checkpoint changed after test evaluation. Do not repeatedly tune on test; use a fresh independent test protocol.')
        for relative, sha in previous['files'].items():
            if digest(run / 'deployment' / relative) != sha:
                raise ValueError('Existing evaluated artifact changed; restore from the saved deployment bundle.')
        print('Already evaluated this selected checkpoint. Reusing saved results without reading test again.')
        print(run / 'RagaMoodSync_Trained_Model.zip'); return
    manifest = json.loads((run / 'fer_training_manifest.json').read_text()); check_manifest(manifest)
    ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if ck.get('synthetic'):
        raise ValueError('Smoke-test checkpoints cannot be evaluated/deployed as trained research models')
    if digest(a.data) != ck['colab_settings']['dataset_sha256']:
        raise ValueError('Dataset changed since training')
    export = run / 'deployment'; reports = export / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    model = model_from(ck, device)
    validation = fixed_split(a.data, manifest, 'val')
    raw, truth = predictions(model, validation, device, a.batch_size)
    flip, _ = predictions(model, validation, device, a.batch_size, flip=True)
    raw_metrics=metrics(raw,truth);flip_metrics=metrics(flip,truth)
    raw_f1 = raw_metrics['classification_report']['macro avg']['f1-score']
    flip_f1 = flip_metrics['classification_report']['macro avg']['f1-score']
    raw_disgust=raw_metrics['classification_report']['disgust']['f1-score'];flip_disgust=flip_metrics['classification_report']['disgust']['f1-score']
    use_flip = flip_f1 > raw_f1 + .002 and flip_disgust>=raw_disgust and flip_metrics['accuracy']>=raw_metrics['accuracy']
    chosen = flip if use_flip else raw
    log_temp = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_temp], lr=.1, max_iter=60, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(chosen / log_temp.exp().clamp(.05, 10), truth)
        loss.backward(); return loss
    opt.step(closure)
    temperature = float(log_temp.exp().clamp(.05, 10).detach())
    calibration = {'split': 'fixed validation only; reused for checkpoint selection', 'n': len(validation),
                   'single_macro_f1': raw_f1, 'flip_macro_f1': flip_f1,
                   'horizontal_flip_tta': use_flip, 'temperature': temperature,
                   'flip_selection_rule': 'mean logits only when validation macro F1 improves >0.002, accuracy and disgust F1 do not decrease',
                   'single_disgust_f1':raw_disgust,'flip_disgust_f1':flip_disgust,
                   'nll_before': float(torch.nn.functional.cross_entropy(chosen, truth)),
                   'nll_after': float(torch.nn.functional.cross_entropy(chosen / temperature, truth)),
                   'accuracy_unchanged_by_temperature': metrics(chosen, truth)['accuracy'],
                   'limitations': 'No independent calibration cohort or webcam-domain validation. Temperature does not fix wrong labels.'}
    ck['inference'] = {'horizontal_flip_tta': use_flip, 'temperature': temperature}
    ck['calibration'] = calibration
    save_checkpoint(export / 'models/fusdense.pt', ck)
    sha = digest(export / 'models/fusdense.pt')
    write_json(reports / 'calibration.json', {**calibration, 'checkpoint_sha256': sha})
    val_report = metrics(chosen, truth, temperature)
    write_json(reports / 'validation_evaluation.json', {**val_report, 'split': 'val', 'checkpoint_sha256': sha})
    classes = val_report['classification_report']
    policy = {'checkpoint_sha256': sha, 'split': 'fixed validation only', 'n': len(validation),
              'minimum_confidence': .7, 'minimum_margin': .2, 'minimum_class_f1': .6,
              'minimum_class_recall': .6, 'class_report': classes,
              'eligible_classes': [c for c in FER_CLASSES if classes[c]['f1-score'] >= .6 and classes[c]['recall'] >= .6],
              'limitation': 'Operational abstention only; full unfiltered test metrics are reported separately.'}
    write_json(reports / 'expression_policy.json', policy)
    # Only now open test pixels. Exclude every train/val pixel hash, then deduplicate test.
    ds = FER2013(a.data, 'test')
    seen = {r['pixel_sha256'] for split in ['train', 'val'] for r in manifest[split]}
    kept = []; removed = 0
    for row, fingerprint in zip(ds.rows, ds.fingerprints()):
        if fingerprint in seen: removed += 1; continue
        seen.add(fingerprint); kept.append(row)
    ds.rows = kept; ds.cache_images()
    logits, labels = predictions(model, ds, device, a.batch_size, flip=use_flip)
    result = metrics(logits, labels, temperature)
    probabilities = (logits / temperature).softmax(1).numpy(); confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels.numpy(); bins = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else confidence <= hi)
        n = int(mask.sum())
        bins.append({'lower': float(lo), 'upper': float(hi), 'n': n,
                     'accuracy': float(correct[mask].mean()) if n else None,
                     'confidence': float(confidence[mask].mean()) if n else None})
    result.update({'split': 'test', 'dataset': 'FER2013 Kaggle v1 combined test folder',
                   'checkpoint_sha256': sha, 'checkpoint_epoch': ck['epoch'], 'pilot': False,
                   'synthetic': False, 'architecture': ck['architecture'], 'inference': ck['inference'],
                   'duplicate_pixels_removed': removed, 'reliability_bins': bins,
                   'expected_calibration_error_10_bins': sum(b['n'] / len(ds) * abs(b['accuracy'] - b['confidence']) for b in bins if b['n']),
                   'limitation': 'Expanded historically inspected test folder, not a fresh or identity-disjoint benchmark. All seven classes and all retained images included.'})
    records = [{'file': str(f), 'label': y, 'pixel_sha256': h} for (f, y), h in zip(ds.rows, ds.fingerprints())]
    write_json(reports / 'emotion_evaluation.json', result)
    write_json(reports / 'emotion_evaluation_manifest.json', {'records': records})
    write_json(reports / 'fer_training_manifest.json', manifest)
    shutil.copy2(run / 'training_history.json', reports / 'training_history.json')
    shutil.copy2(run / 'settings.json', reports / 'colab_settings.json')
    if (run / 'colab_environment.txt').exists():
        shutil.copy2(run / 'colab_environment.txt', reports / 'colab_environment.txt')
    baseline = json.loads(Path(a.baseline_report).read_text())
    baseline_manifest = json.loads(Path(a.baseline_manifest).read_text())
    if baseline_manifest['records'] != records or baseline['predictions']['truth'] != result['predictions']['truth']:
        raise ValueError('Baseline test records differ. Refusing a misleading same-set comparison.')
    comparisons = {}
    for name, before, after in [
        ('accuracy', baseline['accuracy'], result['accuracy']),
        ('macro_f1', baseline['classification_report']['macro avg']['f1-score'], result['classification_report']['macro avg']['f1-score']),
        ('disgust_recall', baseline['classification_report']['disgust']['recall'], result['classification_report']['disgust']['recall']),
        ('disgust_f1', baseline['classification_report']['disgust']['f1-score'], result['classification_report']['disgust']['f1-score'])]:
        comparisons[name] = {'baseline': before, 'colab': after, 'difference': after - before}
    recommended = comparisons['accuracy']['difference'] > 0 and comparisons['macro_f1']['difference'] > 0 and comparisons['disgust_recall']['difference'] >= 0 and comparisons['disgust_f1']['difference'] > 0
    comparison = {'n': len(ds), 'checkpoint_sha256':sha, 'metrics': comparisons, 'recommended_for_integration': recommended,
                  'rule': 'Accuracy, macro F1 and disgust F1 increase, disgust recall does not decrease; an engineering deployment check, not significance evidence',
                  'limitation': 'Descriptive comparison; do not tune further using this test result. GPU execution itself is not an architectural innovation.'}
    write_json(reports / 'colab_comparison.json', comparison)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.asarray(result['confusion_matrix']), display_labels=FER_CLASSES).plot(ax=ax, cmap='Greens', colorbar=False)
    ax.set_title(f'Unfiltered FER2013 test (n={len(ds)})'); fig.tight_layout()
    fig.savefig(reports / 'emotion_evaluation.png', dpi=160); plt.close(fig)
    history = json.loads((reports / 'training_history.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for key in ['loss', 'val_loss']: axes[0].plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    for key in ['accuracy', 'val_accuracy']: axes[1].plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    axes[2].plot([r['epoch'] for r in history], [r['val_macro_f1'] for r in history], label='val macro F1')
    for ax in axes: ax.legend(); ax.set_xlabel('Epoch'); ax.grid(alpha=.15)
    fig.suptitle(f'Recorded training history; current continuation on {device.type.upper()}; objectives differ across phases')
    fig.tight_layout(); fig.savefig(reports / 'training_curves.png', dpi=160); plt.close(fig)
    test_classes = result['classification_report']
    rows = '\n'.join(f"| {c} | {test_classes[c]['precision']:.4f} | {test_classes[c]['recall']:.4f} | {test_classes[c]['f1-score']:.4f} |" for c in FER_CLASSES)
    text = f"""# Actual {device.type.upper()} continuation emotion-model results

Selected epoch: {ck['epoch']}. Fixed training: {len(manifest['train'])}; validation: {len(manifest['val'])}.
Unfiltered test: {len(ds)} faces. Accuracy: {result['accuracy']:.6f}; macro F1: {test_classes['macro avg']['f1-score']:.6f}.
Checkpoint SHA-256: {sha}. Deployment recommendation: {recommended}.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
{rows}

All results are measured from the selected checkpoint. Calibration and flip selection use validation only. Test remains historically inspected, not fresh or identity-disjoint. This run does not evaluate EEG, HRV, weather, raga-response or therapeutic efficacy. No guaranteed accuracy target is asserted. See colab_comparison.json for the complete same-set comparison, including regressions.
"""
    (reports / 'COLAB_EXPERIMENTS.md').write_text(text, encoding='utf-8')
    bundle = {'schema': 'ragamoodsync-colab-v1', 'source_best_sha256': digest(run / 'best.pt'),
              'checkpoint_sha256': sha, 'recommended_for_integration': recommended,
              'files': {str(f.relative_to(export)).replace('\\', '/'): digest(f) for f in sorted(export.rglob('*')) if f.is_file()}}
    write_json(run / 'bundle_manifest.json', bundle)
    write_json(export / 'bundle_manifest.json', bundle)
    with zipfile.ZipFile(run / 'RagaMoodSync_Trained_Model.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(export.rglob('*')):
            if file.is_file(): archive.write(file, file.relative_to(export))
    print(json.dumps(comparison, indent=2))
    print(run / 'RagaMoodSync_Trained_Model.zip')


if __name__ == '__main__':
    main()
