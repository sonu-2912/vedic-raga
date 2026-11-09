## Latest measured update: further training and controlled physiology

Active FusDenseNet v3: accuracy **52.58%**, macro F1 **0.4658**, disgust recall **22.78%**, disgust F1 **0.2571** on all 6,561 retained test faces. Previous baseline was 52.55%. This remains a limited historically inspected facial benchmark, not reliable webcam emotion recognition.

Overall accuracy changed by only two test faces. Disgust precision fell from 0.3333 to 0.2951, while recall and F1 improved. Version 3 adds 768 normalized early spatial stem features to the original 588 fusion features, followed by a 1,356-to-128-to-7 MLP. The selected improvement uses classifier-only refinement; further full-network trials were rejected on validation.

Game-constant, fixed-30s real EEG/ECG evaluation shows **no fusion F1 gain**. DENS participant/movie-held-out evaluation is incomplete because only 2/12 source headers match referenced signal dimensions. No DENS performance is invented. Full current results, regressions and commands: [FURTHER_RESULTS.md](FURTHER_RESULTS.md). Earlier numerical sections below are historical and may refer to previous checkpoints or confounded pooled tasks.

<!-- actual-real-data-update -->
## Latest measured update — 3 November 2026

Active classifier: **52.55% unfiltered test accuracy**, macro F1 **0.4492**, disgust recall **12.66%**. New experiments, phase counts, all-class metrics, actual real-recording GamER fusion and its regressions/confounds are documented in [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md). Older figures below describe earlier checkpoints and stages. No result here establishes reliable webcam emotion, facial-plus-physiological fusion or raga/weather benefit. Current automated verification: 37/37 tests passed.

<!-- actual-real-data-update -->
# Experiments and actual results

Updated 03 November 2026 12:53, local Asia/Kolkata time. Python 3.11, PyTorch 2.5.1+cpu, Windows; no CUDA or physical sensors. All numbers below come from persisted outputs.

## Original paper claims

[Pingle & Ragha (2025)](https://doi.org/10.1007/s41870-025-02531-y) Table 2 reports 99.72% accuracy and Table 5 reports 99.82%. These inconsistent paper values are not project results. Architecture dimensions, exact training protocol and original custom raga ratings are not fully recoverable. This project is paper-informed, not an exact reproduction.

## Revised FusDenseNet: actual real FER2013 experiment

FER2013 Kaggle version 1: 28,709 training-folder and 7,178 combined test-folder images. The seeded stratified 15% validation partition comes only from training. All available training-partition images were considered, with no subset limit. Exact pixel hashes remove 899 duplicate training images and 337 duplicate/overlapping validation images, leaving **23,503 training / 3,970 validation faces**. Test exclusions use the complete final training/validation manifest plus within-test duplicates: 617 removed, **6,561 evaluated**. No test data selects the checkpoint or hyperparameters.

The Kaggle test folder combines the original FER test partitions and some was previously evaluated in the pilot. This is an expanded benchmark evaluation, not a claim of a fresh unseen dataset, PrivateTest-only or subject-disjoint evaluation. Identity overlap and near duplicates remain unresolved. Manifest files contain exact filenames and hashes.

Architecture v2 has **223,776 trainable parameters**. It retains three dual-branch fusion-dense stages, dilated convolutions, cross-dense links, four-head cross-attention, normalized gates, concatenation and reconstruction. The revision adds fusion BatchNorm/ReLU, [-1,1] input scaling, global mean/max/RMS plus 2x2 spatial pooling and a 588→128→7 classifier. These are documented implementation choices, not recovered paper specifications.

Seed 42; AdamW, weight decay 0.0001, cosine LR decay, batch 64, 8 CPU threads. The first phase used initial LR 0.002 and ended after epoch 5 across an overnight interruption. The continuation warm-started the best weights and explicitly restarted AdamW/cosine at LR 0.001; this is not exact optimizer-state continuation. Resume provenance is stored in the checkpoint. Loss: inverse-square-root frequency-weighted CE with label smoothing 0.03 + 0.05 reconstruction MSE. Horizontal flip, ±15° rotation and crop scale 0.85–1.0 apply only to training. No vertical flip, no default denoising, no external ImageNet or supplied Navrasa weights. Gradient norm clips at 5. Completed **17 epochs**. Per-epoch wall times are retained, but epoch 5 includes an overnight interruption and must not be treated as active compute. Continuation rows additionally record aggregate process CPU seconds across threads. **Best validation macro F1** selected epoch **16**, validation macro F1 **0.398849**. Validation CE is unweighted and excludes reconstruction, so its loss curve has a different objective from training.

| Metric | Actual revised test value |
|---|---:|
| Accuracy | 0.501296 |
| Majority-class accuracy baseline | 0.260783 |
| Balanced accuracy / macro recall | 0.411529 |
| Macro precision | 0.484062 |
| Macro F1 | 0.395982 |
| Weighted F1 | 0.475317 |
| Evaluation inference seconds | 24.882 |
| Negative log likelihood | 1.275916 |
| Multiclass Brier score | 0.608768 |
| Expected calibration error (10 bins) | 0.016374 |

Inference recipe: horizontal-flip averaging **False**, temperature **0.743999**. Two-view mean logits are selected only if validation macro F1 exceeds single-view F1 by more than 0.002. Temperature minimizes validation NLL and does not change the selected class. Calibration uses the same validation partition used for checkpoint selection, not an independent calibration set. No test samples fit temperature or select inference policy. Validation calibration report: `reports/calibration.json`. It does not establish reliable confidence on webcam images.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| angry | 0.3584 | 0.4245 | 0.3887 | 888 |
| disgust | 0.6667 | 0.0253 | 0.0488 | 79 |
| fear | 0.2724 | 0.0824 | 0.1266 | 922 |
| happy | 0.7137 | 0.8217 | 0.7639 | 1711 |
| sad | 0.3415 | 0.3239 | 0.3325 | 1198 |
| surprise | 0.5857 | 0.6351 | 0.6094 | 581 |
| neutral | 0.4500 | 0.5677 | 0.5021 | 1182 |

Same-set original pilot: accuracy 0.265203, macro F1 0.104785. Revised accuracy gain: 23.61 percentage points. Both checkpoints were evaluated on exactly the same retained test examples. This comparison changes architecture, optimizer, data and training duration together; it is not causal evidence for cross-attention or pooling alone.

Fixed HOG + linear SVM comparator: **41.47% test accuracy**, macro F1 **0.347015**, on the exact same 6,561 test files. Descriptor: 900 dimensions (48x48 window, 16x16 blocks, 8x8 stride/cells, 9 bins), training-fitted StandardScaler, LinearSVC C=0.1, class_weight=balanced, max_iter=5000. Hyperparameters were fixed, with no validation/test search. This classical comparator is not deployed or relabeled as FusDenseNet. Evidence: reports/hog_baseline.json/png. HOG method source: [Dalal & Triggs, CVPR 2005](https://doi.org/10.1109/CVPR.2005.177); that source studies human detection and does not supply emotion accuracy claims.

Historical pilot result remains 25.82% accuracy / 0.1032 macro F1 on its different 1,375-image subset. Its checkpoint and reports are preserved in `reports/baseline_pilot/`. Do not directly subtract accuracies from different test sets. Revised evidence: `reports/emotion_evaluation.json`, confusion PNG, `training_history.json`, `training_curves.png`, `emotion_evaluation_manifest.json`, `fer_training_manifest.json`, `model_comparison.json/png`. Checkpoint SHA-256: `04e2a0e288b071a576422ba5adbf02d8fad604e4f22747064eab6aa7f7e7f53e`. The API displays metrics only when the fingerprint matches. Confusion/error rates remain material; webcam images differ from FER training crops. An expression classifier cannot establish internal mood, clinical benefit or fairness.

## Software and integration verification

**29/29 tests pass**, 0 failures and 0 errors; 34.79 seconds. Tests cover v1/v2 forward/backward and attention gradients, automatic checkpoint refresh, matching metric provenance, FER split handling, metadata counts, SQLite preference updates and isolation, multimodal missingness and numerical signal features. Frontend production build is verified separately. The API integration script uses a public NASA portrait without a ground-truth emotion label, with synthetic feedback in an isolated temporary database; it cannot measure facial accuracy or listener improvement.

Preservation recheck: all 55 extracted original repository files byte-identical: True. Reference PDF copy identical to original: True.

## Other experiments and pending research

Earlier engineering-only results and signal/feedback examples are preserved verbatim in `reports/baseline_pilot/EXPERIMENTS.md` and their JSON outputs. Artificial textures are synthetic, not FER2013. EEG/HRV sample mode is simulated and is not a sensor reading. FDCF synthetic training tests its interface, not recommendation quality. Source metadata has 72 ragas, 528 link entries and 516 unique URLs; fields absent from the source remain missing.

Real interaction-trained FDCF, valid ranking metrics and IKS efficacy ablation remain pending adequate consented user data. Controlled equal-budget attention/dilation/pooling retraining ablations and multi-seed uncertainty analysis remain pending. The new overall comparison does not isolate architecture effects. DEAP/WESAD are not synchronized Hindustani raga-response datasets. No clinical, sensor, patentability or novelty result is claimed.

## Multimodal correction — 3 November 2026

The full facial classifier remains 50.13% accurate on the retained test set; disgust recognition is weak. Conservative validation-based abstention replaces unsupported mood suggestions. Weather, EEG/HRV recordings, reference safeguards, listening goals and visible source contributions now reach the listener's recommendation path. This is experimental rule-based context integration, **not a validated improvement in physiological emotion recognition, recommendation quality or health**. Actual synthetic ablations and limitations are documented in [MULTIMODAL_STATUS.md](MULTIMODAL_STATUS.md). Current tests: 29/29 passed.

## Updated presentation and interface verification

Current deck: `output/RagaMoodSync_Research_Multimodal.pptx` (also copied to `output/RagaMoodSync_Research.pptx`). Exactly 13 editable slides, all rendered slides visually reviewed; package/layout/import/native-chart checks passed with zero findings. Previous FER-only deck is preserved as `output/RagaMoodSync_Research_Updated.pptx`. Native PowerPoint execution is not claimed. Desktop context explanations and 390px mobile layout were checked with isolated test data; no production feedback was written.

## Final real-data deliverables

Installed improved checkpoint: 52.55% unfiltered test accuracy, macro F1 0.4492; macro precision declined and disgust remains weak. Real GamER paired EEG/ECG experiments, offset sensitivity and game-context controls completed. Final verification: 37/37 tests passed; live model/evidence APIs and source checksums independently audited; production frontend build passed. Current deck: `output/RagaMoodSync_Research_Real_Data_Final.pptx` (13 editable slides), also copied to `output/RagaMoodSync_Research.pptx`. All 13 rendered slides reviewed; structural/layout/import/native-chart checks passed with zero findings. Native PowerPoint execution is not claimed. No aligned facial-physiology or raga/weather outcome validation is claimed.
