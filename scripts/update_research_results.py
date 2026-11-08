"""Rebuild reports from matching real checkpoint/evaluation and saved test evidence."""
import json,hashlib,math,zipfile
from datetime import datetime
from xml.etree import ElementTree as ET
import torch
import numpy as np
from backend.catalog import ROOT
from backend.models import FusDenseNet,FER_CLASSES

def read(name):return json.loads((ROOT/'reports'/name).read_text(encoding='utf-8'))
def main():
    ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True)
    real=read('emotion_evaluation.json');train=read('fer_training_manifest.json');history=read('training_history.json')
    assert real['checkpoint_sha256']==hashlib.sha256((ROOT/'models/fusdense.pt').read_bytes()).hexdigest(),'Checkpoint changed; rerun final evaluation'
    assert not real['synthetic'] and not ck['synthetic']
    model=FusDenseNet(**ck['architecture']);params=sum(p.numel() for p in model.parameters())
    macro=real['classification_report']['macro avg'];weighted=real['classification_report']['weighted avg']
    inference=ck.get('inference',{});calibration=ck.get('calibration',{})
    from .render_training_curves import main as render_curves
    render_curves()
    baseline_path=ROOT/'reports/baseline_re_evaluation.json'
    baseline=read('baseline_re_evaluation.json') if baseline_path.exists() else None
    hog=read('hog_baseline.json') if (ROOT/'reports/hog_baseline.json').exists() else None
    if hog:
        assert hog['test_files']==[r['file'] for r in read('emotion_evaluation_manifest.json')['records']]
        hog_text=f"Fixed HOG + linear SVM comparator: **{100*hog['accuracy']:.2f}% test accuracy**, macro F1 **{hog['classification_report']['macro avg']['f1-score']:.6f}**, on the exact same {hog['n']:,} test files. Descriptor: 900 dimensions (48x48 window, 16x16 blocks, 8x8 stride/cells, 9 bins), training-fitted StandardScaler, LinearSVC C=0.1, class_weight=balanced, max_iter=5000. Hyperparameters were fixed, with no validation/test search. This classical comparator is not deployed or relabeled as FusDenseNet. Evidence: reports/hog_baseline.json/png. HOG method source: [Dalal & Triggs, CVPR 2005](https://doi.org/10.1109/CVPR.2005.177); that source studies human detection and does not supply emotion accuracy claims."
    else:hog_text='Classical HOG + linear SVM comparator pending.'
    if baseline:
        assert baseline['n']==real['n'] and baseline['predictions']['truth']==real['predictions']['truth']
        assert read('baseline_re_evaluation_manifest.json')['records']==read('emotion_evaluation_manifest.json')['records'],'Baseline and revised evaluated images must match exactly'
        comparison={'protocol':'Same complete retained test images, excluded against full revised training/validation pixel hashes. Checkpoints differ in data budget, architecture and training recipe; not a controlled architecture ablation.','n':real['n'],'baseline':{'accuracy':baseline['accuracy'],'macro_f1':baseline['classification_report']['macro avg']['f1-score']},'revised':{'accuracy':real['accuracy'],'macro_f1':macro['f1-score']},'accuracy_gain_percentage_points':100*(real['accuracy']-baseline['accuracy']),'macro_f1_gain':macro['f1-score']-baseline['classification_report']['macro avg']['f1-score']}
        comparison['baseline']['checkpoint_sha256']=baseline['checkpoint_sha256'];comparison['revised']['checkpoint_sha256']=real['checkpoint_sha256']
        (ROOT/'reports/model_comparison.json').write_text(json.dumps(comparison,indent=2))
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(11,4));x=np.arange(7)
        for offset,r,label,color in [(-.2,baseline,'Original 2-epoch pilot','#b0957a'),(.2,real,'Revised full-training split','#244f41')]:
            ax[0].bar(x+offset,[r['classification_report'][c]['recall'] for c in FER_CLASSES],.4,label=label,color=color)
        ax[0].set_xticks(x,FER_CLASSES,rotation=30);ax[0].set_ylabel('Recall');ax[0].set_ylim(0,1);ax[0].legend(fontsize=8)
        ax[1].bar(['Pilot accuracy','Revised accuracy','Pilot macro F1','Revised macro F1'],[baseline['accuracy'],real['accuracy'],comparison['baseline']['macro_f1'],macro['f1-score']],color=['#b0957a','#244f41']*2)
        ax[1].tick_params(axis='x',rotation=28);ax[1].set_ylim(0,1);fig.suptitle(f'Actual same-test-set comparison (n={real["n"]}); combined training changes')
        fig.tight_layout();fig.savefig(ROOT/'reports/model_comparison.png',dpi=160);plt.close(fig)
    suite=ET.parse(ROOT/'reports/pytest.xml').getroot().find('testsuite');summary=read('verification_summary.json')
    source_checks=[]
    with zipfile.ZipFile(ROOT/'raag-ratnakar-main.zip') as archive:
        for member in archive.infolist():
            if not member.is_dir():source_checks.append(archive.read(member.filename)==(ROOT/'external'/member.filename).read_bytes())
    summary.update({'original_repository_files':len(source_checks),'original_repository_unchanged':all(source_checks),'reference_pdf_identical':(ROOT/'vedic raga 1.pdf').read_bytes()==(ROOT/'references/RagaMoodSync_Paper.pdf').read_bytes()})
    summary.update({'tests':int(suite.attrib['tests']),'passed':int(suite.attrib['tests'])-int(suite.attrib['failures'])-int(suite.attrib['errors'])-int(suite.attrib.get('skipped',0)),'failures':int(suite.attrib['failures']),'errors':int(suite.attrib['errors']),'test_seconds':float(suite.attrib['time']),'real_fer2013_metrics':{'n':real['n'],'accuracy':real['accuracy'],'macro_precision':macro['precision'],'macro_recall':macro['recall'],'macro_f1':macro['f1-score'],'pilot':ck['pilot'],'parameters':params},'training':{'epochs_completed':len(history),'selected_epoch':ck['epoch'],'train_count':train['train_count'],'val_count':train['val_count'],'seconds':sum(r['seconds'] for r in history),'timing_note':'Epoch 5 includes an overnight interruption; summed wall time is not active CPU training duration. Continuation records aggregate process CPU seconds separately.','resume_provenance':ck.get('resume_provenance')},'updated':datetime.now().isoformat()})
    (ROOT/'reports/verification_summary.json').write_text(json.dumps(summary,indent=2))
    classrows='\n'.join(f"| {c} | {real['classification_report'][c]['precision']:.4f} | {real['classification_report'][c]['recall']:.4f} | {real['classification_report'][c]['f1-score']:.4f} | {int(real['classification_report'][c]['support'])} |" for c in FER_CLASSES)
    comparisons=f"Same-set original pilot: accuracy {baseline['accuracy']:.6f}, macro F1 {baseline['classification_report']['macro avg']['f1-score']:.6f}. Revised accuracy gain: {100*(real['accuracy']-baseline['accuracy']):.2f} percentage points. Both checkpoints were evaluated on exactly the same retained test examples. This comparison changes architecture, optimizer, data and training duration together; it is not causal evidence for cross-attention or pooling alone." if baseline else 'Same-set baseline comparison not yet generated.'
    report=f'''# Experiments and actual results

Updated {datetime.now():%d %B %Y %H:%M}, local Asia/Kolkata time. Python 3.11, PyTorch 2.5.1+cpu, Windows; no CUDA or physical sensors. All numbers below come from persisted outputs.

## Original paper claims

[Pingle & Ragha (2025)](https://doi.org/10.1007/s41870-025-02531-y) Table 2 reports 99.72% accuracy and Table 5 reports 99.82%. These inconsistent paper values are not project results. Architecture dimensions, exact training protocol and original custom raga ratings are not fully recoverable. This project is paper-informed, not an exact reproduction.

## Revised FusDenseNet: actual real FER2013 experiment

FER2013 Kaggle version 1: 28,709 training-folder and 7,178 combined test-folder images. The seeded stratified 15% validation partition comes only from training. All available training-partition images were considered, with no subset limit. Exact pixel hashes remove {train['duplicates_removed']['train']} duplicate training images and {train['duplicates_removed']['val']} duplicate/overlapping validation images, leaving **{train['train_count']:,} training / {train['val_count']:,} validation faces**. Test exclusions use the complete final training/validation manifest plus within-test duplicates: {real['duplicate_pixels_removed']} removed, **{real['n']:,} evaluated**. No test data selects the checkpoint or hyperparameters.

The Kaggle test folder combines the original FER test partitions and some was previously evaluated in the pilot. This is an expanded benchmark evaluation, not a claim of a fresh unseen dataset, PrivateTest-only or subject-disjoint evaluation. Identity overlap and near duplicates remain unresolved. Manifest files contain exact filenames and hashes.

Architecture v{ck['architecture'].get('version',1)} has **{params:,} trainable parameters**. It retains three dual-branch fusion-dense stages, dilated convolutions, cross-dense links, four-head cross-attention, normalized gates, concatenation and reconstruction. The revision adds fusion BatchNorm/ReLU, [-1,1] input scaling, global mean/max/RMS plus 2x2 spatial pooling and a 588→128→7 classifier. These are documented implementation choices, not recovered paper specifications.

Seed 42; AdamW, weight decay 0.0001, cosine LR decay, batch {ck['training_args']['batch_size']}, {ck['training_args']['threads']} CPU threads. The first phase used initial LR 0.002 and ended after epoch 5 across an overnight interruption. The continuation warm-started the best weights and explicitly restarted AdamW/cosine at LR {ck['training_args']['learning_rate']}; this is not exact optimizer-state continuation. Resume provenance is stored in the checkpoint. Loss: inverse-square-root frequency-weighted CE with label smoothing 0.03 + 0.05 reconstruction MSE. Horizontal flip, ±15° rotation and crop scale 0.85–1.0 apply only to training. No vertical flip, no default denoising, no external ImageNet or supplied Navrasa weights. Gradient norm clips at 5. Completed **{len(history)} epochs**. Per-epoch wall times are retained, but epoch 5 includes an overnight interruption and must not be treated as active compute. Continuation rows additionally record aggregate process CPU seconds across threads. **Best validation macro F1** selected epoch **{ck['epoch']}**, validation macro F1 **{ck['validation_macro_f1']:.6f}**. Validation CE is unweighted and excludes reconstruction, so its loss curve has a different objective from training.

| Metric | Actual revised test value |
|---|---:|
| Accuracy | {real['accuracy']:.6f} |
| Majority-class accuracy baseline | {real['majority_class_baseline_accuracy']:.6f} |
| Balanced accuracy / macro recall | {macro['recall']:.6f} |
| Macro precision | {macro['precision']:.6f} |
| Macro F1 | {macro['f1-score']:.6f} |
| Weighted F1 | {weighted['f1-score']:.6f} |
| Evaluation inference seconds | {real['seconds']:.3f} |
| Negative log likelihood | {real['negative_log_likelihood']:.6f} |
| Multiclass Brier score | {real['brier_score']:.6f} |
| Expected calibration error (10 bins) | {real['expected_calibration_error_10_bins']:.6f} |

Inference recipe: horizontal-flip averaging **{inference.get('horizontal_flip_tta',False)}**, temperature **{inference.get('temperature',1.):.6f}**. Two-view mean logits are selected only if validation macro F1 exceeds single-view F1 by more than 0.002. Temperature minimizes validation NLL and does not change the selected class. Calibration uses the same validation partition used for checkpoint selection, not an independent calibration set. No test samples fit temperature or select inference policy. Validation calibration report: `reports/calibration.json`. It does not establish reliable confidence on webcam images.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{classrows}

{comparisons}

{hog_text}

Historical pilot result remains 25.82% accuracy / 0.1032 macro F1 on its different 1,375-image subset. Its checkpoint and reports are preserved in `reports/baseline_pilot/`. Do not directly subtract accuracies from different test sets. Revised evidence: `reports/emotion_evaluation.json`, confusion PNG, `training_history.json`, `training_curves.png`, `emotion_evaluation_manifest.json`, `fer_training_manifest.json`, `model_comparison.json/png`. Checkpoint SHA-256: `{real['checkpoint_sha256']}`. The API displays metrics only when the fingerprint matches. Confusion/error rates remain material; webcam images differ from FER training crops. An expression classifier cannot establish internal mood, clinical benefit or fairness.

## Software and integration verification

**{summary['passed']}/{summary['tests']} tests pass**, {summary['failures']} failures and {summary['errors']} errors; {summary['test_seconds']:.2f} seconds. Tests cover v1/v2 forward/backward and attention gradients, automatic checkpoint refresh, matching metric provenance, FER split handling, metadata counts, SQLite preference updates and isolation, multimodal missingness and numerical signal features. Frontend production build is verified separately. The API integration script uses a public NASA portrait without a ground-truth emotion label, with synthetic feedback in an isolated temporary database; it cannot measure facial accuracy or listener improvement.

Preservation recheck: all {summary['original_repository_files']} extracted original repository files byte-identical: {summary['original_repository_unchanged']}. Reference PDF copy identical to original: {summary['reference_pdf_identical']}.

## Other experiments and pending research

Earlier engineering-only results and signal/feedback examples are preserved verbatim in `reports/baseline_pilot/EXPERIMENTS.md` and their JSON outputs. Artificial textures are synthetic, not FER2013. EEG/HRV sample mode is simulated and is not a sensor reading. FDCF synthetic training tests its interface, not recommendation quality. Source metadata has 72 ragas, 528 link entries and 516 unique URLs; fields absent from the source remain missing.

Real interaction-trained FDCF, valid ranking metrics and IKS efficacy ablation remain pending adequate consented user data. Controlled equal-budget attention/dilation/pooling retraining ablations and multi-seed uncertainty analysis remain pending. The new overall comparison does not isolate architecture effects. DEAP/WESAD are not synchronized Hindustani raga-response datasets. No clinical, sensor, patentability or novelty result is claimed.
'''
    (ROOT/'EXPERIMENTS.md').write_text(report,encoding='utf-8')
    old=(ROOT/'reports/baseline_pilot/MODEL_CARDS.md').read_text(encoding='utf-8');rest=old[old.index('## Supplied Navrasa baseline'):]
    card=f'''# Model cards

## Project FusDenseNet version 2

Locally trained from random initialization; **{params:,} parameters**, paper-informed mechanisms with disclosed dimensions. This is neither the supplied Navrasa checkpoint nor an original paper author checkpoint. See PAPER_IMPLEMENTATION_MAPPING.md for equations and explicit assumptions.

Input: face-cropped 48x48 grayscale [0,1], internally scaled to [-1,1]. EXIF orientation is corrected on upload; a frontal Haar detector crops the largest detected face. No image is persisted. Output order: {', '.join(FER_CLASSES)}. Validation-selected horizontal-flip averaging: {inference.get('horizontal_flip_tta',False)}; validation-fitted temperature: {inference.get('temperature',1.):.6f}. Default denoising is none. Temperature fitting does not change class predictions or establish calibration for webcam inputs, and outputs do not measure a person's internal mood. Low confidence and manual correction remain explicit.

Training: full seeded training partition, exact-pixel deduplicated {train['train_count']:,} train / {train['val_count']:,} validation, {len(history)} epochs, best validation macro F1 checkpoint epoch {ck['epoch']}. AdamW/cosine, batch {ck['training_args']['batch_size']}, weighted smoothed CE + 0.05 reconstruction MSE. No FER test or private user image was used for training. Actual evaluation: {real['n']:,} retained combined-test faces, accuracy {real['accuracy']:.6f}, macro precision {macro['precision']:.6f}, recall {macro['recall']:.6f}, F1 {macro['f1-score']:.6f}. Full per-class errors and training curves are in EXPERIMENTS.md and reports. Some test-folder examples were previously evaluated in the historical pilot; no untouched-new-dataset claim is made.

Checkpoint: `models/fusdense.pt`, SHA-256 `{real['checkpoint_sha256']}`. Old v1 pilot is retained under `reports/baseline_pilot/`. New weights reload automatically when the file changes in the updated API process. A backend already running older source must restart once to load the revised implementation.

Limitations: no identity-disjoint audit, near-duplicate/fairness study, independent/domain-specific calibration, clinical validation or demonstrated robustness to live webcam domain shifts. Temperature fitting reuses the checkpoint-selection validation set. User confirmation remains required in the listener flow. Rights: FER2013 source advertises ODbL/DbCL; dataset is local and not redistributed. No useful-performance guarantee is made for an individual expression.

'''+rest
    (ROOT/'MODEL_CARDS.md').write_text(card,encoding='utf-8')
    if hog:
        with (ROOT/'MODEL_CARDS.md').open('a',encoding='utf-8') as output:
            output.write(f"\n## Classical comparator: HOG + linear SVM\n\nFixed 900-dimensional HOG descriptor, training-fitted scaling and LinearSVC C=0.1 with balanced class weights. Trained on {hog['train_n']:,} deduplicated FER training faces. Actual same-test results: {hog['n']:,} faces, accuracy {hog['accuracy']:.6f}, macro F1 {hog['classification_report']['macro avg']['f1-score']:.6f}. Hyperparameters were fixed in advance. This comparator is never served as FusDenseNet or used to select its checkpoint. No subject-disjoint or webcam robustness claim. Reproduce with `python -m scripts.hog_baseline --data data/private/fer2013-v1.zip`.\n")
    status_path=ROOT/'PROJECT_STATUS.md';status_history=ROOT/'reports/baseline_pilot/PROJECT_STATUS_history.md'
    if not status_history.exists():status_history.write_bytes(status_path.read_bytes())
    status=f'''# RagaMoodSync project status

Measured update: {datetime.now():%d %B %Y %H:%M}, Asia/Kolkata. Prior progress history is preserved under reports/baseline_pilot/PROJECT_STATUS_history.md.

- Completed: paper-informed FusDenseNet v2, {params:,} parameters; dense fusion, cross-attention, cross-dense links, spatial/global pooling and reconstruction. Original paper author weights and exact architecture recovery are not claimed.
- Actual revised training: {train['train_count']:,} train / {train['val_count']:,} validation images after exact-pixel deduplication; {len(history)} completed epochs across two phases. Best validation macro F1 selected epoch {ck['epoch']} ({ck['validation_macro_f1']:.6f}). Continuation restarted AdamW explicitly. Epoch 5 includes an overnight wall-clock interruption.
- Actual final test: {real['n']:,} retained FER2013 combined-test faces; accuracy {100*real['accuracy']:.2f}%, macro precision {macro['precision']:.4f}, recall {macro['recall']:.4f}, F1 {macro['f1-score']:.4f}. Checksum-matched evidence is in reports/emotion_evaluation.json. Some test-folder examples appeared in the old pilot evaluation; no fresh-dataset or identity-disjoint claim.
- Completed: validation-only inference selection and temperature fit; final calibration metrics reported separately. Webcam-domain reliability remains unverified.
- Completed: customer-facing music interface, FusDenseNet default, photo mood confirmation, recommendations, raga details/links, SQLite feedback and history. Separate research page displays live architecture, measured metrics, epoch history and plots.
- Verification: {summary['passed']}/{summary['tests']} automated tests passed; {summary['failures']} failures, {summary['errors']} errors. Production frontend builds under normal workspace permissions using Vite native config loading.
- Preserved: all {summary['original_repository_files']} original extracted files byte-identical: {summary['original_repository_unchanged']}; PDF copy identical: {summary['reference_pdf_identical']}. Catalog: 72 ragas, 528 link entries, 516 unique URLs. Original source lacks a license file; attribution and rights notes remain.
- Completed: old pilot same-set reevaluation and fixed HOG/linear-SVM comparator; synthetic engineering tests remain separate. This is a combined training/data/architecture comparison, not a controlled architectural ablation.
- Completed: offline EEG/HRV and clearly synthetic sample mode, IKS provenance/missing-field handling and a trainable FDCF module. No sensors or real synchronized raga-response study are claimed.
- Reports updated: EXPERIMENTS.md, MODEL_CARDS.md, paper mapping, dataset access and literature comparator.
- Presentation build/visual review follows automatically; inspect reports/finalization_status.json and reports/slides_build/validation-fer-full.json for its actual state.

Remaining research: controlled equal-budget/multi-seed ablations; stronger/subject-disjoint FER protocol and webcam study; real interaction-trained FDCF and valid recommendation metrics; expert-reviewed missing raga metadata; synchronized multimodal data, calibrated sensors and clinical evidence. Group details in the presentation remain editable placeholders.

Checkpoint SHA-256: `{real['checkpoint_sha256']}`.
'''
    status_path.write_text(status,encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
