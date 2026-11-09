"""Assemble measured experiment facts from saved evidence; never invent metrics."""
import json,sys,platform,zipfile,hashlib
from pathlib import Path
from xml.etree import ElementTree as ET
from backend.catalog import ROOT

def load(name):return json.loads((ROOT/'reports'/name).read_text(encoding='utf-8'))
def main():
    s=load('synthetic_emotion_smoke.json');e=load('end_to_end.json');sig=load('sample_signal_results.json');f=load('fdcf_synthetic_training.json')
    real=load('emotion_evaluation.json');train=load('fer_training_manifest.json');curves=load('training_history.json');macro=real['classification_report']['macro avg']
    suite=ET.parse(ROOT/'reports/pytest.xml').getroot().find('testsuite')
    passed=int(suite.attrib['tests'])-int(suite.attrib['failures'])-int(suite.attrib['errors'])-int(suite.attrib.get('skipped',0))
    integrity=[]
    with zipfile.ZipFile(ROOT/'raag-ratnakar-main.zip') as z:
        for member in z.infolist():
            if member.is_dir():continue
            original=z.read(member.filename);actual=(ROOT/'external'/member.filename).read_bytes()
            integrity.append(original==actual)
    summary={'tests':int(suite.attrib['tests']),'passed':passed,'failures':int(suite.attrib['failures']),'errors':int(suite.attrib['errors']),'test_seconds':float(suite.attrib['time']),'original_repository_files':len(integrity),'original_repository_unchanged':all(integrity),'reference_pdf_identical':(ROOT/'vedic raga 1.pdf').read_bytes()==(ROOT/'references/RagaMoodSync_Paper.pdf').read_bytes(),'python':sys.version.split()[0],'platform':platform.platform(),'synthetic_emotion':{k:s[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','parameters','seconds']},'real_fer2013_metrics':{'n':real['n'],'accuracy':real['accuracy'],'macro_precision':macro['precision'],'macro_recall':macro['recall'],'macro_f1':macro['f1-score'],'pilot':True},'clinical_results':None}
    (ROOT/'reports/verification_summary.json').write_text(json.dumps(summary,indent=2))
    report=f'''# Experiments and measured results

All measurements below come from saved project outputs. These are separate from original paper claims. Test execution date: 2 November 2026, Asia/Kolkata. Runtime: Python {summary['python']}, PyTorch 2.5.1+cpu, Windows, two Torch CPU threads. No CUDA or physical sensors.

## 1. Original paper results (not reproduced)

Pingle & Ragha [DOI](https://doi.org/10.1007/s41870-025-02531-y), Table 2: accuracy 0.9972, precision 0.9896, recall 0.990, F1 0.9901. Table 5 reports accuracy 0.9982, precision 0.9905, recall 0.9912, F1 0.9921 for its full model. Those inconsistent values are retained as paper-reported evidence. Neither is a project result. See PAPER_ANALYSIS.md for protocol and specification gaps.

## 2. Actual software verification

- Automated tests: **{passed}/{summary['tests']} passed**, {summary['failures']} failures, {summary['errors']} errors, {summary['test_seconds']:.2f} seconds. Evidence: `reports/pytest.xml`.
- Tests cover catalog counts and missing fields, like/dislike score changes, profile isolation, persistence, link validation, invalid requests/images, multimodal optionality, known HRV numerical values, synthetic EEG bands, model output dimensions, cross-attention/FDCF gradients, history erasure, ZIP split/limit/image correctness and official CSV Usage splits.
- Upstream deprecation warnings from Starlette/AnyIO and Matplotlib/Pyparsing remain. They do not indicate a failed test.
- Frontend `npm run build` passed. Vite dev server and FastAPI returned HTTP 200. `uv pip check` reported all installed packages compatible. Evidence: `reports/live_health.json`, `requirements-lock.txt`.
- All {len(integrity)} original repository files are byte-identical to the supplied ZIP. Reference PDF copy is byte-identical to the original. Verified catalog: 72 ragas, 528 link entries, 516 unique URLs.
- Browser verified manual sad mood, six visible recommendations, synthetic-profile like feedback, score change 0.711 to 0.768, and optional synthetic EEG/HRV rendering. Screenshots: `reports/application_home.jpg`, `application_recommendations.jpg`, `application_signals.jpg`. No real participant feedback was collected.

## 3. Actual one-photo integration test, with synthetic feedback

`python -m scripts.verify_end_to_end` processed a NASA/scikit-image public-domain portrait (one unlabeled image), detected a frontal face at {e['prediction']['face_box']}, ran supplied EfficientNet-B2 weights, recommended ragas, saved one synthetic feedback event and one listening-link event in an isolated temporary database. Elapsed pipeline time: {e['elapsed_seconds']:.3f} seconds including model initialization in this measured run, not a latency distribution.

Mapped emotion output: {e['prediction']['emotion']} at {e['prediction']['confidence']:.4f}; native Rasa winner: {e['prediction']['native_label']}. The difference is expected under many-to-one probability aggregation, which is an unvalidated project mapping. Confidence is low and not calibrated. There is **no ground-truth emotion**, so no facial accuracy can be calculated.

First recommendation {e['first_raga']} changed score from {e['score_before']:.6f} to {e['score_after']:.6f} after a synthetic like and +1 mood report. This verifies adaptation mechanics, not listener improvement. `reports/end_to_end.json` contains the full actual API outputs.

## 4. Actual FER2013 subset pilot

The public Kaggle metadata API became accessible after the initial HTML page did not expose license metadata. It advertised version 1, 56,510,189 bytes of contained data and an Open Database/Database Contents license. After license and disk review, the public download succeeded without credentials. The 63,252,113-byte ZIP contains 35,887 images (28,709 train, 7,178 combined test). Its SHA-256 is recorded in `reports/fer2013_archive_audit.json`. No dataset is republished.

Protocol fixed before test evaluation: seeded 15% stratified validation partition from the archive train folder, then seeded class-proportional limits of 2,100 train, 700 validation and 1,400 test. Pixel hashes removed {train['duplicates_removed']['train']} duplicate training images and {train['duplicates_removed']['val']} validation images, leaving **{train['train_count']} train / {train['val_count']} validation**. Excluding train/validation pixel hashes and within-test duplicates removed **{real['duplicate_pixels_removed']}** selected test images, leaving **{real['n']} held-out images**. Identity disjointness and near-duplicate removal are not established. File labels and pixel hashes are recorded in the train/test manifests.

Two epochs, Adam learning rate 0.001, batch 32, seed 42, augmentation on training only, CE + 0.1 reconstruction MSE. Best validation checkpoint: epoch {real['checkpoint_epoch']}. Training took {sum(r['seconds'] for r in curves):.2f} measured seconds, evaluation {real['seconds']:.2f} seconds. No test image selected the checkpoint. This is the project architecture, not supplied paper weights.

| Real held-out pilot metric | Measured value |
|---|---:|
| Accuracy | {real['accuracy']:.6f} |
| Macro precision | {macro['precision']:.6f} |
| Macro recall | {macro['recall']:.6f} |
| Macro F1 | {macro['f1-score']:.6f} |

The majority-class baseline on this selected test set is {max(v['support'] for k,v in real['classification_report'].items() if k in ['angry','disgust','fear','happy','sad','surprise','neutral'])/real['n']:.6f}. The pilot predicts only happy/sad/neutral and assigns zero recall to four classes. Its low macro F1 demonstrates substantial underfitting and imbalance sensitivity. This does not establish a useful validated emotion recognizer, superiority to the supplied Navrasa baseline, or reproduction of the paper's 99.72% result. It is a bounded, actual real-data experiment.

Artifacts: `models/fusdense.pt`, `reports/emotion_evaluation.json`, `reports/emotion_evaluation.png`, `reports/training_history.json`, `reports/training_curves.png`, `reports/fer_training_manifest.json`, `reports/fer_test_manifest.json`. Checkpoint SHA-256: `{real['checkpoint_sha256']}`. Classification API responses attach this evaluation only when its checkpoint checksum matches. The default remains the separately labeled pretrained Navrasa baseline, with manual mood correction available for both models.

## 5. Synthetic FusDenseNet engineering experiment

Three fusion stages with cross-dense connections, four-head attention, dilated dense layers and mean/max/RMS pooling: **{s['parameters']:,} parameters**. Seed 42. Twenty-one generated artificial textures for optimization, fourteen separately noised artificial textures for checking. Seven texture classes T0–T6, not facial-emotion categories. Sixteen Adam steps with CE + 0.1 reconstruction MSE, completed in {s['seconds']:.3f} seconds.

| Metric | Measured synthetic value |
|---|---:|
| Accuracy | {s['accuracy']:.6f} |
| Macro precision | {s['macro_precision']:.6f} |
| Macro recall | {s['macro_recall']:.6f} |
| Macro F1 | {s['macro_f1']:.6f} |
| Initial training objective | {s['curves'][0]['loss']:.6f} |
| Final training objective | {s['curves'][-1]['loss']:.6f} |

The final checker predicts one texture class for all fourteen examples. This deliberately bounded smoke run verifies optimization, graph connectivity and metric/plot generation, but **fails to demonstrate useful generalization**. It is not tuned into a favorable research result. Training-mode loss decreases while eval-mode performance stays low, consistent with an extremely small optimization budget and unstable small-sample BatchNorm statistics. That explanation is an engineering hypothesis, not an experimentally isolated cause.

Raw results: `reports/synthetic_emotion_smoke.json`; confusion matrix: `reports/synthetic_confusion_matrix.png`; curves: `reports/synthetic_training_curves.png`. The PowerPoint loss series rounds raw losses to six decimals for editable Excel chart compatibility. No synthetic-trained facial checkpoint is saved for serving.

## 6. Synthetic signal processing

One simulated EEG channel, 16 seconds at 128 Hz, generated with dominant 10 Hz activity plus a 6 Hz component and noise. MNE 1–35 Hz IIR filtering and SciPy Welch PSD produced relative theta {sig['eeg']['bands']['theta']['relative'][0]:.6f}, alpha {sig['eeg']['bands']['alpha']['relative'][0]:.6f}, beta {sig['eeg']['bands']['beta']['relative'][0]:.6f}. These bands do not cover the entire denominator and are not expected to sum to one.

Ninety simulated RR intervals produced RMSSD **{sig['hrv']['rmssd_ms']:.4f} ms**, SDNN **{sig['hrv']['sdnn_ms']:.4f} ms**, mean heart rate {sig['hrv']['mean_hr_bpm']:.4f} bpm. The known alternating 800/900 ms RR unit test independently verifies RMSSD = 100 ms and sample SDNN. Raw results are `reports/sample_signal_results.json`. No actual sensor measurements, autonomic diagnosis or emotional validity are claimed.

## 7. Synthetic FDCF training interface test

`python -m scripts.verify_fdcf_training` created {f['observations']} labeled synthetic interaction events for {f['users']} synthetic users in an isolated database, ran {f['epochs']} optimization epochs and produced training BCE {f['training_bce']:.6f}. It verifies the train/save path only. Artifact `reports/fdcf_synthetic_smoke.pt` is marked synthetic and the serving loader rejects synthetic checkpoints. `reports/fdcf_synthetic_training.json` records the disclosure. There is no held-out recommendation-quality result.

## 8. Recommendations and IKS sensitivity

`reports/sample_recommendations.json` records sad, happy and neutral examples before/after synthetic feedback. `reports/iks_sensitivity.json` compares IKS on/off rankings for identical cold-start input. The lists differ because metadata contributes to scoring, while disabled IKS yields ties resolved alphabetically. This is **ranking sensitivity**, not an IKS benefit or relevance-metric ablation.

The actual recommendation evaluation reports pending when insufficient real interaction data exists. No hit-rate, NDCG, precision-at-k or recall-at-k is fabricated. After collecting real feedback, the provided heuristic evaluator uses rolling chronological history, a four-event warmup and held-out positive events. It disables any globally trained FDCF model to avoid future-label leakage. Global FDCF training loss must not be presented as recommendation accuracy.

## 9. Pending research experiments

- Full-budget FER2013 training and a strict official/private/subject-disjoint benchmark beyond the small combined-folder pilot.
- Fixed-budget cross-attention, dilation and denoising ablations using identical splits.
- Native Navrasa held-out validation. Serialized validation metadata is upstream-reported only; the separate NFEC dataset is absent.
- Prospective user-level recommendation study, chronological FDCF evaluation and genuine IKS quality ablation.
- Expert-reviewed metadata and performance-level annotations. Audio/tempo representation learning.
- Artifact-screened physiological validation with participant-disjoint data, synchronized raga-response acquisition and learned multimodal fusion.
- Any clinical evaluation, causal mood-improvement claim or real sensor experiment.

Commands and dataset access procedures are in README.md and DATASETS.md. The application remains runnable while these research experiments are pending.
'''
    (ROOT/'EXPERIMENTS.md').write_text(report,encoding='utf-8')
    waveform=load('synthetic_waveform_results.json') if (ROOT/'reports/synthetic_waveform_results.json').exists() else None
    if waveform:
        with (ROOT/'EXPERIMENTS.md').open('a',encoding='utf-8') as out:
            out.write(f'''\n## 10. Final extension and browser checks (3 November 2026)\n\nNeuroKit2 waveform processing was also executed on generated 45-second signals (seed 42), beyond the RR-only sample. ECG at 250 Hz yielded {waveform['ecg']['beats']} detected beats, RMSSD {waveform['ecg']['rmssd_ms']:.4f} ms and SDNN {waveform['ecg']['sdnn_ms']:.4f} ms. PPG at 64 Hz yielded {waveform['ppg']['beats']} detected beats, RMSSD {waveform['ppg']['rmssd_ms']:.4f} ms and SDNN {waveform['ppg']['sdnn_ms']:.4f} ms. These are simulation outputs, not sensor or clinical results. Raw evidence: `reports/synthetic_waveform_results.json`.\n\nThe browser also completed the actual upload → FusDenseNet pilot → happy expression output (26.0% uncalibrated confidence) → raga recommendation path using the public-domain demo portrait. Screenshot: `reports/application_fusdense.jpg`. The API attached the checksum-matched held-out pilot report. Private webcam capture was not used. Only the generated `synthetic-ui-demo` database rows were cleaned up for handoff.\n''')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
