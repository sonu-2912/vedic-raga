# TRAIN-only pure FusDenseNet BatchNorm check

2026-11-09T01:19:24.666229+05:30

Three fixed candidates: current pure baseline, final compact student and final compact averaged network. Each receives one pass over all23503unaugmented TRAIN images. Labels are ignored and every learned parameter remains bit-identical; only native BatchNorm running buffers change. No validation/test image contributes to these statistics.

This uses the installed PyTorch2.5.1 update_bn implementation, inspected locally, consistent with [official PyTorch documentation](https://docs.pytorch.org/docs/2.15/optim.html#taking-care-of-batch-normalization). The drift audit on1024TRAINfaces found modest average mean shifts0.0514storedSD for the original and0.0578for the fine-tuned model; this does not establish a major BatchNorm defect or cause of overfitting.

All validation results below use3970real faces, including50disgust examples. High disgust recall with collapsed precision is rejected.

| Model/stage | Accuracy | Macro F1 | Disgust P/R/F1 | Happy recall | Surprise recall | Neutral recall |
|---|---:|---:|---|---:|---:|---:|
| Current PURE validation | 58.41% | 0.5397 | 0.3673/36.00%/0.3636 | 78.59% | 67.43% | 53.00% |
| deployed_pure_baseline | 58.16% | 0.5300 | 0.3478/32.00%/0.3333 | 82.03% | 62.29% | 53.42% |
| final_compact_student | 57.08% | 0.5070 | 0.1576/52.00%/0.2419 | 82.79% | 64.00% | 56.21% |
| final_compact_EMA | 57.43% | 0.5120 | 0.1667/48.00%/0.2474 | 82.41% | 64.57% | 55.51% |

Two normalization contract tests passed in8.53seconds. All original raw/EMA/source models and buffer audits are retained; no rejected model replaces the app.

## Test and deployment decision

None of three TRAIN-only BN-recalibrated pure models passed every unchanged class and aggregate validation gate. App/PPT unchanged.

New test evaluated: False. Eligible: False. All prospective accuracy/macroF1, macroP/R, disgustP/R/F1 and historical happy/surprise/neutral recovery floors remain unchanged. No alternate model is tested after a test failure.

No new test result is claimed. The deployed pure result remains58.34% accuracy,0.5554macroF1,46.84%disgust recall. No app or PPT replacement is justified by this phase.

Protected app/model/PPT files all unchanged before any integration: True.

All classes, faces and errors remain included. No uncertainty-based filtering, post-test thresholds, external encoder or probability mixture is used. TRAIN/validation/test cohorts are fixed. This is exploratory engineering with repeated validation use and historical test inspection, not clinical or webcam-domain confirmation. EEG/HRV/weather/raga-efficacy results are unchanged. No user photograph or real sensor was used.
