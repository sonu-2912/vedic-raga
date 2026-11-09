<!-- actual-real-data-update -->
## Latest measured update — 3 November 2026

Active classifier: **52.55% unfiltered test accuracy**, macro F1 **0.4492**, disgust recall **12.66%**. New experiments, phase counts, all-class metrics, actual real-recording GamER fusion and its regressions/confounds are documented in [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md). Older figures below describe earlier checkpoints and stages. No result here establishes reliable webcam emotion, facial-plus-physiological fusion or raga/weather benefit. Current automated verification: 37/37 tests passed.

<!-- actual-real-data-update -->
# Model cards

## Project FusDenseNet version 2

Locally trained from random initialization; **223,776 parameters**, paper-informed mechanisms with disclosed dimensions. This is neither the supplied Navrasa checkpoint nor an original paper author checkpoint. See PAPER_IMPLEMENTATION_MAPPING.md for equations and explicit assumptions.

Input: face-cropped 48x48 grayscale [0,1], internally scaled to [-1,1]. EXIF orientation is corrected on upload; a frontal Haar detector crops the largest detected face. No image is persisted. Output order: angry, disgust, fear, happy, sad, surprise, neutral. Validation-selected horizontal-flip averaging: False; validation-fitted temperature: 0.743999. Default denoising is none. Temperature fitting does not change class predictions or establish calibration for webcam inputs, and outputs do not measure a person's internal mood. Low confidence and manual correction remain explicit.

Training: full seeded training partition, exact-pixel deduplicated 23,503 train / 3,970 validation, 17 epochs, best validation macro F1 checkpoint epoch 16. AdamW/cosine, batch 64, weighted smoothed CE + 0.05 reconstruction MSE. No FER test or private user image was used for training. Actual evaluation: 6,561 retained combined-test faces, accuracy 0.501296, macro precision 0.484062, recall 0.411529, F1 0.395982. Full per-class errors and training curves are in EXPERIMENTS.md and reports. Some test-folder examples were previously evaluated in the historical pilot; no untouched-new-dataset claim is made.

Checkpoint: `models/fusdense.pt`, SHA-256 `04e2a0e288b071a576422ba5adbf02d8fad604e4f22747064eab6aa7f7e7f53e`. Old v1 pilot is retained under `reports/baseline_pilot/`. New weights reload automatically when the file changes in the updated API process. A backend already running older source must restart once to load the revised implementation.

Limitations: no identity-disjoint audit, near-duplicate/fairness study, independent/domain-specific calibration, clinical validation or demonstrated robustness to live webcam domain shifts. Temperature fitting reuses the checkpoint-selection validation set. User confirmation remains required in the listener flow. Rights: FER2013 source advertises ODbL/DbCL; dataset is local and not redistributed. No useful-performance guarantee is made for an individual expression.

## Supplied Navrasa baseline

File: original `external/raag-ratnakar-main/best_navarasa_model.pth`, preserved byte-for-byte. EfficientNet-B2 with 1408→512→9 classifier and dropout, reconstructed strictly from supplied source. Resize 256, center crop 224, ImageNet RGB normalization. CPU inference and safe `weights_only=True` loading verified.

Checkpoint-native labels are nine aesthetic Rasa categories, not the seven FER classes. Native probabilities and the native top class remain visible in the API. A disclosed many-to-one heuristic aggregates these into seven operational emotion probabilities. This mapping is uncalibrated and is not benchmarked as FER performance. Do not compare the two models' probabilities as equally calibrated measurements.

The checkpoint stores val_acc 93.4250% and epoch 28, while its accompanying document reports other upstream test results. These are **upstream-reported metadata**, not project-reproduced measurements. No NFEC images/subject splits were supplied. The repository contains no license file; redistribution rights remain unresolved. See REPOSITORY_AUDIT.md and THIRD_PARTY_NOTICES.md.

## FDCF

`backend/models.py` provides learned user/music/emotion embeddings, focus softmax, dynamic short/long preferences, emotion–music elementwise product and MLP matching. Dimensions, preference windows and loss are project choices documented in the paper mapping. `scripts/train_fdcf.py` trains from observed SQLite explicit feedback only. No real interaction-trained checkpoint exists initially. Cold-start IKS and feedback scoring is explicitly labeled as a heuristic, not FDCF.

`reports/fdcf_synthetic_smoke.pt` is a synthetic training-interface artifact marked `synthetic=True`; the serving loader rejects it. Its training BCE is not recommendation quality. Real recommendation evaluation, calibrated ranking and prospective mood-improvement studies remain pending.

## Classical comparator: HOG + linear SVM

Fixed 900-dimensional HOG descriptor, training-fitted scaling and LinearSVC C=0.1 with balanced class weights. Trained on 23,503 deduplicated FER training faces. Actual same-test results: 6,561 faces, accuracy 0.414723, macro F1 0.347015. Hyperparameters were fixed in advance. This comparator is never served as FusDenseNet or used to select its checkpoint. No subject-disjoint or webcam robustness claim. Reproduce with `python -m scripts.hog_baseline --data data/private/fer2013-v1.zip`.

## Multimodal correction — 3 November 2026

The full facial classifier remains 50.13% accurate on the retained test set; disgust recognition is weak. Conservative validation-based abstention replaces unsupported mood suggestions. Weather, EEG/HRV recordings, reference safeguards, listening goals and visible source contributions now reach the listener's recommendation path. This is experimental rule-based context integration, **not a validated improvement in physiological emotion recognition, recommendation quality or health**. Actual synthetic ablations and limitations are documented in [MULTIMODAL_STATUS.md](MULTIMODAL_STATUS.md). Current tests: 29/29 passed.
