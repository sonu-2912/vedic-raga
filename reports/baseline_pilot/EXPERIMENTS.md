# Experiments and measured results

All measurements below come from saved project outputs. These are separate from original paper claims. Test execution date: 2 November 2026, Asia/Kolkata. Runtime: Python 3.11.18, PyTorch 2.5.1+cpu, Windows, two Torch CPU threads. No CUDA or physical sensors.

## 1. Original paper results (not reproduced)

Pingle & Ragha [DOI](https://doi.org/10.1007/s41870-025-02531-y), Table 2: accuracy 0.9972, precision 0.9896, recall 0.990, F1 0.9901. Table 5 reports accuracy 0.9982, precision 0.9905, recall 0.9912, F1 0.9921 for its full model. Those inconsistent values are retained as paper-reported evidence. Neither is a project result. See PAPER_ANALYSIS.md for protocol and specification gaps.

## 2. Actual software verification

- Automated tests: **18/18 passed**, 0 failures, 0 errors, 29.69 seconds. Evidence: `reports/pytest.xml`.
- Tests cover catalog counts and missing fields, like/dislike score changes, profile isolation, persistence, link validation, invalid requests/images, multimodal optionality, known HRV numerical values, synthetic EEG bands, model output dimensions, cross-attention/FDCF gradients, history erasure, ZIP split/limit/image correctness and official CSV Usage splits.
- Upstream deprecation warnings from Starlette/AnyIO and Matplotlib/Pyparsing remain. They do not indicate a failed test.
- Frontend `npm run build` passed. Vite dev server and FastAPI returned HTTP 200. `uv pip check` reported all installed packages compatible. Evidence: `reports/live_health.json`, `requirements-lock.txt`.
- All 55 original repository files are byte-identical to the supplied ZIP. Reference PDF copy is byte-identical to the original. Verified catalog: 72 ragas, 528 link entries, 516 unique URLs.
- Browser verified manual sad mood, six visible recommendations, synthetic-profile like feedback, score change 0.711 to 0.768, and optional synthetic EEG/HRV rendering. Screenshots: `reports/application_home.jpg`, `application_recommendations.jpg`, `application_signals.jpg`. No real participant feedback was collected.

## 3. Actual one-photo integration test, with synthetic feedback

`python -m scripts.verify_end_to_end` processed a NASA/scikit-image public-domain portrait (one unlabeled image), detected a frontal face at [177, 66, 95, 95], ran supplied EfficientNet-B2 weights, recommended ragas, saved one synthetic feedback event and one listening-link event in an isolated temporary database. Elapsed pipeline time: 1.793 seconds including model initialization in this measured run, not a latency distribution.

Mapped emotion output: happy at 0.3297; native Rasa winner: bhibhatsa. The difference is expected under many-to-one probability aggregation, which is an unvalidated project mapping. Confidence is low and not calibrated. There is **no ground-truth emotion**, so no facial accuracy can be calculated.

First recommendation Dhani changed score from 0.711250 to 0.780221 after a synthetic like and +1 mood report. This verifies adaptation mechanics, not listener improvement. `reports/end_to_end.json` contains the full actual API outputs.

## 4. Actual FER2013 subset pilot

The public Kaggle metadata API became accessible after the initial HTML page did not expose license metadata. It advertised version 1, 56,510,189 bytes of contained data and an Open Database/Database Contents license. After license and disk review, the public download succeeded without credentials. The 63,252,113-byte ZIP contains 35,887 images (28,709 train, 7,178 combined test). Its SHA-256 is recorded in `reports/fer2013_archive_audit.json`. No dataset is republished.

Protocol fixed before test evaluation: seeded 15% stratified validation partition from the archive train folder, then seeded class-proportional limits of 2,100 train, 700 validation and 1,400 test. Pixel hashes removed 9 duplicate training images and 3 validation images, leaving **2091 train / 697 validation**. Excluding train/validation pixel hashes and within-test duplicates removed **25** selected test images, leaving **1375 held-out images**. Identity disjointness and near-duplicate removal are not established. File labels and pixel hashes are recorded in the train/test manifests.

Two epochs, Adam learning rate 0.001, batch 32, seed 42, augmentation on training only, CE + 0.1 reconstruction MSE. Best validation checkpoint: epoch 2. Training took 94.72 measured seconds, evaluation 9.25 seconds. No test image selected the checkpoint. This is the project architecture, not supplied paper weights.

| Real held-out pilot metric | Measured value |
|---|---:|
| Accuracy | 0.258182 |
| Macro precision | 0.105469 |
| Macro recall | 0.158097 |
| Macro F1 | 0.103229 |

The majority-class baseline on this selected test set is 0.250182. The pilot predicts only happy/sad/neutral and assigns zero recall to four classes. Its low macro F1 demonstrates substantial underfitting and imbalance sensitivity. This does not establish a useful validated emotion recognizer, superiority to the supplied Navrasa baseline, or reproduction of the paper's 99.72% result. It is a bounded, actual real-data experiment.

Artifacts: `models/fusdense.pt`, `reports/emotion_evaluation.json`, `reports/emotion_evaluation.png`, `reports/training_history.json`, `reports/training_curves.png`, `reports/fer_training_manifest.json`, `reports/fer_test_manifest.json`. Checkpoint SHA-256: `2295f39d4de8a901457d986aa23d6aa06faf1c416c87ebd138a236ee6618bb86`. Classification API responses attach this evaluation only when its checkpoint checksum matches. The default remains the separately labeled pretrained Navrasa baseline, with manual mood correction available for both models.

## 5. Synthetic FusDenseNet engineering experiment

Three fusion stages with cross-dense connections, four-head attention, dilated dense layers and mean/max/RMS pooling: **163,744 parameters**. Seed 42. Twenty-one generated artificial textures for optimization, fourteen separately noised artificial textures for checking. Seven texture classes T0–T6, not facial-emotion categories. Sixteen Adam steps with CE + 0.1 reconstruction MSE, completed in 12.371 seconds.

| Metric | Measured synthetic value |
|---|---:|
| Accuracy | 0.142857 |
| Macro precision | 0.020408 |
| Macro recall | 0.142857 |
| Macro F1 | 0.035714 |
| Initial training objective | 1.955514 |
| Final training objective | 0.398977 |

The final checker predicts one texture class for all fourteen examples. This deliberately bounded smoke run verifies optimization, graph connectivity and metric/plot generation, but **fails to demonstrate useful generalization**. It is not tuned into a favorable research result. Training-mode loss decreases while eval-mode performance stays low, consistent with an extremely small optimization budget and unstable small-sample BatchNorm statistics. That explanation is an engineering hypothesis, not an experimentally isolated cause.

Raw results: `reports/synthetic_emotion_smoke.json`; confusion matrix: `reports/synthetic_confusion_matrix.png`; curves: `reports/synthetic_training_curves.png`. The PowerPoint loss series rounds raw losses to six decimals for editable Excel chart compatibility. No synthetic-trained facial checkpoint is saved for serving.

## 6. Synthetic signal processing

One simulated EEG channel, 16 seconds at 128 Hz, generated with dominant 10 Hz activity plus a 6 Hz component and noise. MNE 1–35 Hz IIR filtering and SciPy Welch PSD produced relative theta 0.099674, alpha 0.873446, beta 0.018944. These bands do not cover the entire denominator and are not expected to sum to one.

Ninety simulated RR intervals produced RMSSD **11.7125 ms**, SDNN **22.1226 ms**, mean heart rate 70.4631 bpm. The known alternating 800/900 ms RR unit test independently verifies RMSSD = 100 ms and sample SDNN. Raw results are `reports/sample_signal_results.json`. No actual sensor measurements, autonomic diagnosis or emotional validity are claimed.

## 7. Synthetic FDCF training interface test

`python -m scripts.verify_fdcf_training` created 16 labeled synthetic interaction events for 2 synthetic users in an isolated database, ran 8 optimization epochs and produced training BCE 0.607901. It verifies the train/save path only. Artifact `reports/fdcf_synthetic_smoke.pt` is marked synthetic and the serving loader rejects synthetic checkpoints. `reports/fdcf_synthetic_training.json` records the disclosure. There is no held-out recommendation-quality result.

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

## 10. Final extension and browser checks (3 November 2026)

NeuroKit2 waveform processing was also executed on generated 45-second signals (seed 42), beyond the RR-only sample. ECG at 250 Hz yielded 52 detected beats, RMSSD 8.6533 ms and SDNN 9.6085 ms. PPG at 64 Hz yielded 51 detected beats, RMSSD 58.2072 ms and SDNN 69.0458 ms. These are simulation outputs, not sensor or clinical results. Raw evidence: `reports/synthetic_waveform_results.json`.

The browser also completed the actual upload → FusDenseNet pilot → happy expression output (26.0% uncalibrated confidence) → raga recommendation path using the public-domain demo portrait. Screenshot: `reports/application_fusdense.jpg`. The API attached the checksum-matched held-out pilot report. Private webcam capture was not used. Only the generated `synthetic-ui-demo` database rows were cleaned up for handoff.
