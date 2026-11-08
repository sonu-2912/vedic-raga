"""Report a completed, predeclared full-network experiment without inventing results."""
import json
from pathlib import Path
import torch
from scripts.colab_common import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/consistency_20261107'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    done = read(RUN / 'completion.json')
    protocol = read(RUN / 'protocol.json')
    history = read(RUN / 'training_history.json')
    epochs = [r for r in history if r.get('phase') == 'full_network_consistency']
    initial = read(RUN / 'initial_validation.json')
    best = read(RUN / 'best_validation.json') if done['validation_improved'] else initial
    baseline = torch.load(RUN / 'baseline.pt', map_location='cpu', weights_only=True)
    selected = torch.load(RUN / 'best.pt', map_location='cpu', weights_only=True)
    changed = {}
    for name, tensor in selected['state_dict'].items():
        old = baseline['state_dict'][name]
        if not torch.equal(tensor, old):
            group = name.split('.')[0]
            item = changed.setdefault(group, {'changed_tensors':0,'changed_elements':0,'max_absolute_change':0.})
            item['changed_tensors'] += 1
            item['changed_elements'] += int((tensor != old).sum())
            item['max_absolute_change'] = max(item['max_absolute_change'], float((tensor.float()-old.float()).abs().max()))
    evaluation = RUN/'deployment/reports/emotion_evaluation.json'
    comparison = read(RUN/'deployment/reports/colab_comparison.json') if evaluation.exists() else None
    active_sha = digest(ROOT/'models/fusdense.pt')
    deployed = bool(comparison and comparison['recommended_for_integration'] and active_sha==comparison['checkpoint_sha256'])
    summary = {'protocol': protocol, 'completion': done, 'baseline_sha256':digest(RUN/'baseline.pt'),
               'selected_sha256':digest(RUN/'best.pt'),'active_sha256':active_sha,'deployed':deployed,
               'selected_candidate':selected.get('candidate_type'),'selected_epoch':selected.get('epoch'),
               'changed_tensor_groups':changed,'validation_initial_raw':initial,'validation_selected_raw':best,
               'test_comparison':comparison,'synthetic':False,
               'limitation':'Previously inspected test and repeatedly reused validation; no fresh independent benchmark or clinical outcome.'}
    rows = '\n'.join(f"| {r['epoch']} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} | {r['val_disgust_recall']:.4f} | {r['val_disgust_f1']:.4f} |" for r in epochs)
    if comparison:
        old_test = read(RUN/'baseline_test.json')
        new_test = read(evaluation)
        for name in ['precision', 'recall']:
            before = old_test['classification_report']['macro avg'][name]
            after = new_test['classification_report']['macro avg'][name]
            comparison['metrics']['macro_'+name] = {'baseline':before,'colab':after,'difference':after-before}
    if comparison:
        before=old_test['classification_report']['disgust']['precision']
        after=new_test['classification_report']['disgust']['precision']
        comparison['metrics']['disgust_precision']={'baseline':before,'colab':after,'difference':after-before}
    metrics = ['accuracy','macro_precision','macro_recall','macro_f1','disgust_precision','disgust_recall','disgust_f1']
    test = ('\n'.join(f"| {k.replace('_',' ')} | {comparison['metrics'][k]['baseline']:.6f} | {comparison['metrics'][k]['colab']:.6f} | {comparison['metrics'][k]['difference']:+.6f} |" for k in metrics)
            if comparison else 'No new test metrics: no qualifying validation candidate was evaluated.')
    write_json(RUN/'experiment_summary.json', summary)
    write_json(ROOT/'reports/latest_emotion_experiment.json', {
        'method':'full_network_consistency','completed_epochs':len(epochs),
        'candidate_checkpoint_sha256':comparison['checkpoint_sha256'] if comparison else None,
        'test_comparison':comparison, 'synthetic':False,
        'limitation':'All seven classes and all retained test errors included. Historical test, repeatedly reused validation; no independent webcam, physiological fusion or clinical benefit demonstrated.'})
    text = f'''# Full-network consistency experiment: actual results

Completed {len(epochs)} additional full emotion-path epochs on the existing fixed 23,503 training / 3,970 validation faces. No new dataset, sensor reading or fabricated label is used. Selected candidate: {selected.get('candidate_type','baseline')}, index {selected.get('epoch')}. Deployed: **{deployed}**.

## What changed

All FusDenseNet emotion-path parameters were eligible for gradient updates, including the stem, dense fusion, attention and classifier. Batch-normalization running statistics were frozen. Reconstruction-only parameters receive no reconstruction objective in this experiment. This continues the existing v4 architecture rather than claiming an architectural improvement in isolation.

Training uses horizontal flips, mild bilinear affine augmentation and brightness/contrast jitter, weighted cross entropy on **every training face**, dropout 0.45, a smaller backbone learning rate (1e-5) than classifier (5e-5), cosine decay and gradient clipping. A 0.15-weight KL term (T=2) uses the preceding model's predictions on training faces only, and only where that teacher matches the training label. An incorrect teacher never suppresses the cross-entropy term. Cached teachers are indexed explicitly through shuffled batches. Exponential moving average weights provide one validation-selected candidate alongside the student each epoch.

The immutable protocol specifies maximum 8 epochs, a 90-minute budget checked only at completed epoch boundaries (so one long or interrupted epoch can overshoot it), patience 4, and selection score 0.4 accuracy + 0.6 macro F1 with accuracy, macro F1, disgust recall and disgust F1 floors relative to raw baseline validation. Flip averaging and temperature are then chosen/fitted on validation only. No threshold filters the reported classification test set. All wrong predictions and all seven classes remain.

## Validation log, before flip averaging

| Epoch | Accuracy | Macro F1 | Disgust recall | Disgust F1 |
|---|---:|---:|---:|---:|
{rows}

The log includes rejected epochs. Its epoch seconds are elapsed wall time, potentially including pauses, not active CPU compute. The budget was checked after the fifth completed epoch; do not interpret this as a strict 90-minute wall-time cap. Raw-validation comparisons must not be confused with the preceding deployed model's 57.41% flip-averaged validation score. Training diagnostics measured 74.18% unaugmented training accuracy against 57.41% validation, consistent with overfitting, not evidence that simply training longer will reach 80%.

## One frozen candidate, full same-set test

| Metric | Before | Candidate | Difference |
|---|---:|---:|---:|
{test}

Precision trade-off: disgust true positives increased from19 to29 of79, while false positives increased from15 to31. Macro precision and disgust precision both decreased. Improved recall/F1 does not establish dependable disgust recognition.

Deployment requires increased full-test accuracy, macro F1 and disgust F1 with no decrease in disgust recall. If this gate fails, the previous model remains active and the rejected candidate's results remain available. This is a descriptive engineering gate, not statistical significance. The 6,561-face combined Kaggle test is historically inspected, not fresh or identity-disjoint. Repeated validation selection risks overfitting its 50 disgust examples. No 80% accuracy, webcam-domain reliability or clinical claim is made.

## Real physiology and recommendation limits

Existing official GamER fixed-game, exactly-30-second, participant-held-out EEG/ECG experiments remain negative for primary fusion improvement: EEG/fusion macro F1 0.4583 in QWOP and 0.6061 in CBAAD. No new paired facial/raga/weather response data has become available. DREAMER's official Zenodo record 546113 has restricted access and exposes no downloadable files; manual custodian access and dataset terms are required. No substitutes or simulated recordings are scored as real. DENS remains incomplete due to source-dimension mismatches. Local feedback is insufficient for a defensible held-out recommendation benefit; no recommendation precision/recall is invented.

Distillation background: Hinton, Vinyals and Dean (2015), https://doi.org/10.48550/arXiv.1503.02531 . The teacher-correct mask and specific training recipe are project choices.

Primary sources: https://doi.org/10.1007/s41870-025-02531-y ; https://doi.org/10.5281/zenodo.8191877 ; https://doi.org/10.1109/JBHI.2017.2688239 ; https://zenodo.org/records/546113 .

## Reproduce

The completed run is frozen after evaluation. Re-running its evaluator verifies/reuses the saved bundle; it does not retrain against the test result. For a new prospective run, prepare a new run directory and record protocol/split hashes before training.

```powershell
.\\.venv\\Scripts\\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/consistency_20261107 --device cpu --threads 4 --baseline-report reports/consistency_20261107/baseline_test.json --baseline-manifest reports/consistency_20261107/baseline_test_manifest.json
.\\.venv\\Scripts\\python.exe -m scripts.report_consistency
```

The tensor-change audit and exact hashes are in reports/consistency_20261107/experiment_summary.json. This proves weights changed, not that each individual change caused better generalization.
'''
    (ROOT/'CONSISTENCY_RESULTS.md').write_text(text,encoding='utf-8')
    print(json.dumps({'completed_epochs':len(epochs),'deployed':deployed,'changed_groups':list(changed),'test_comparison':comparison},indent=2))

if __name__=='__main__':
    main()
