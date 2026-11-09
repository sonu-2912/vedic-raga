# Model cards

## Project FusDenseNet CPU pilot

File: `models/fusdense.pt`. Paper-informed architecture: three repeated fusion-dense stages with cross-dense connections, four-head cross-attention, original/smoothed visible views, dilation, concatenation, mean/max/RMS pooling and a seven-class head. 163,744 parameters. This is not an exact recovered architecture or original author checkpoint. Details and deviations: PAPER_IMPLEMENTATION_MAPPING.md.

Training data: public Kaggle msambare/fer2013 version 1, acquired after license/size review. Seed 42, stratified limited subsets, exact grayscale-pixel duplicate removal, 2,091 training and 697 validation images, two CPU epochs, batch 32, Adam 0.001, CE + 0.1 MSE. Best validation checkpoint epoch 2. Inputs: grayscale 48x48, values [0,1], no inference denoising. Labels in output order: angry, disgust, fear, happy, sad, surprise, neutral.

Actual held-out **subset**: 1,375 images from Kaggle's combined test folder after excluding selected train/validation pixel hashes and within-test duplicate pixels. Accuracy 0.258182, macro precision 0.105469, macro recall 0.158097, macro F1 0.103229. Four classes have zero recall. This is a weak, underfit pilot and should not be presented as a useful validated recognizer or evidence for the paper's reported accuracy. No identity-disjoint protocol, near-duplicate audit, fairness study or probability calibration was established. It remains an explicitly selectable research model; manual mood correction is available.

SHA-256: `2295f39d4de8a901457d986aa23d6aa06faf1c416c87ebd138a236ee6618bb86`. Evaluation only attaches to API inference when this fingerprint matches. Raw metrics, confusion matrix, training curves and split manifests are under reports. The source dataset's advertised ODbL/DbCL terms are recorded in DATASETS.md. No clinical or mood-improvement validity is inferred.

## Supplied Navrasa baseline

File: original `external/raag-ratnakar-main/best_navarasa_model.pth`, preserved byte-for-byte. EfficientNet-B2 with 1408→512→9 classifier and dropout, reconstructed strictly from supplied source. Resize 256, center crop 224, ImageNet RGB normalization. CPU inference and safe `weights_only=True` loading verified.

Checkpoint-native labels are nine aesthetic Rasa categories, not the seven FER classes. Native probabilities and the native top class remain visible in the API. A disclosed many-to-one heuristic aggregates these into seven operational emotion probabilities. This mapping is uncalibrated and is not benchmarked as FER performance. Do not compare the two models' probabilities as equally calibrated measurements.

The checkpoint stores val_acc 93.4250% and epoch 28, while its accompanying document reports other upstream test results. These are **upstream-reported metadata**, not project-reproduced measurements. No NFEC images/subject splits were supplied. The repository contains no license file; redistribution rights remain unresolved. See REPOSITORY_AUDIT.md and THIRD_PARTY_NOTICES.md.

## FDCF

`backend/models.py` provides learned user/music/emotion embeddings, focus softmax, dynamic short/long preferences, emotion–music elementwise product and MLP matching. Dimensions, preference windows and loss are project choices documented in the paper mapping. `scripts/train_fdcf.py` trains from observed SQLite explicit feedback only. No real interaction-trained checkpoint exists initially. Cold-start IKS and feedback scoring is explicitly labeled as a heuristic, not FDCF.

`reports/fdcf_synthetic_smoke.pt` is a synthetic training-interface artifact marked `synthetic=True`; the serving loader rejects it. Its training BCE is not recommendation quality. Real recommendation evaluation, calibrated ranking and prospective mood-improvement studies remain pending.
