# Full-network consistency experiment: actual results

Completed 5 additional full emotion-path epochs on the existing fixed 23,503 training / 3,970 validation faces. No new dataset, sensor reading or fabricated label is used. Selected candidate: student, index 124. Deployed: **True**.

## What changed

All FusDenseNet emotion-path parameters were eligible for gradient updates, including the stem, dense fusion, attention and classifier. Batch-normalization running statistics were frozen. Reconstruction-only parameters receive no reconstruction objective in this experiment. This continues the existing v4 architecture rather than claiming an architectural improvement in isolation.

Training uses horizontal flips, mild bilinear affine augmentation and brightness/contrast jitter, weighted cross entropy on **every training face**, dropout 0.45, a smaller backbone learning rate (1e-5) than classifier (5e-5), cosine decay and gradient clipping. A 0.15-weight KL term (T=2) uses the preceding model's predictions on training faces only, and only where that teacher matches the training label. An incorrect teacher never suppresses the cross-entropy term. Cached teachers are indexed explicitly through shuffled batches. Exponential moving average weights provide one validation-selected candidate alongside the student each epoch.

The immutable protocol specifies maximum 8 epochs, a 90-minute budget checked only at completed epoch boundaries (so one long or interrupted epoch can overshoot it), patience 4, and selection score 0.4 accuracy + 0.6 macro F1 with accuracy, macro F1, disgust recall and disgust F1 floors relative to raw baseline validation. Flip averaging and temperature are then chosen/fitted on validation only. No threshold filters the reported classification test set. All wrong predictions and all seven classes remain.

## Validation log, before flip averaging

| Epoch | Accuracy | Macro F1 | Disgust recall | Disgust F1 |
|---|---:|---:|---:|---:|
| 121 | 0.5715 | 0.5197 | 0.2600 | 0.3291 |
| 122 | 0.5713 | 0.5177 | 0.2800 | 0.3111 |
| 123 | 0.5741 | 0.5174 | 0.2600 | 0.2989 |
| 124 | 0.5804 | 0.5225 | 0.2800 | 0.3043 |
| 125 | 0.5796 | 0.5213 | 0.2800 | 0.2979 |

The log includes rejected epochs. Its epoch seconds are elapsed wall time, potentially including pauses, not active CPU compute. The budget was checked after the fifth completed epoch; do not interpret this as a strict 90-minute wall-time cap. Raw-validation comparisons must not be confused with the preceding deployed model's 57.41% flip-averaged validation score. Training diagnostics measured 74.18% unaugmented training accuracy against 57.41% validation, consistent with overfitting, not evidence that simply training longer will reach 80%.

## One frozen candidate, full same-set test

| Metric | Before | Candidate | Difference |
|---|---:|---:|---:|
| accuracy | 0.561347 | 0.574150 | +0.012803 |
| macro precision | 0.546594 | 0.539191 | -0.007404 |
| macro recall | 0.498146 | 0.532511 | +0.034365 |
| macro f1 | 0.510547 | 0.532040 | +0.021494 |
| disgust precision | 0.558824 | 0.483333 | -0.075490 |
| disgust recall | 0.240506 | 0.367089 | +0.126582 |
| disgust f1 | 0.336283 | 0.417266 | +0.080983 |

Precision trade-off: disgust true positives increased from19 to29 of79, while false positives increased from15 to31. Macro precision and disgust precision both decreased. Improved recall/F1 does not establish dependable disgust recognition.

Deployment requires increased full-test accuracy, macro F1 and disgust F1 with no decrease in disgust recall. If this gate fails, the previous model remains active and the rejected candidate's results remain available. This is a descriptive engineering gate, not statistical significance. The 6,561-face combined Kaggle test is historically inspected, not fresh or identity-disjoint. Repeated validation selection risks overfitting its 50 disgust examples. No 80% accuracy, webcam-domain reliability or clinical claim is made.

## Real physiology and recommendation limits

Existing official GamER fixed-game, exactly-30-second, participant-held-out EEG/ECG experiments remain negative for primary fusion improvement: EEG/fusion macro F1 0.4583 in QWOP and 0.6061 in CBAAD. No new paired facial/raga/weather response data has become available. DREAMER's official Zenodo record 546113 has restricted access and exposes no downloadable files; manual custodian access and dataset terms are required. No substitutes or simulated recordings are scored as real. DENS remains incomplete due to source-dimension mismatches. Local feedback is insufficient for a defensible held-out recommendation benefit; no recommendation precision/recall is invented.

Distillation background: Hinton, Vinyals and Dean (2015), https://doi.org/10.48550/arXiv.1503.02531 . The teacher-correct mask and specific training recipe are project choices.

Primary sources: https://doi.org/10.1007/s41870-025-02531-y ; https://doi.org/10.5281/zenodo.8191877 ; https://doi.org/10.1109/JBHI.2017.2688239 ; https://zenodo.org/records/546113 .

## Reproduce

The completed run is frozen after evaluation. Re-running its evaluator verifies/reuses the saved bundle; it does not retrain against the test result. For a new prospective run, prepare a new run directory and record protocol/split hashes before training.

```powershell
.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/consistency_20261107 --device cpu --threads 4 --baseline-report reports/consistency_20261107/baseline_test.json --baseline-manifest reports/consistency_20261107/baseline_test_manifest.json
.\.venv\Scripts\python.exe -m scripts.report_consistency
```

The tensor-change audit and exact hashes are in reports/consistency_20261107/experiment_summary.json. This proves weights changed, not that each individual change caused better generalization.
