This file describes the independent six-epoch coverage trial, which did not qualify. A subsequent native-classifier orientation repair passed its frozen full-test gate. Latest active results: [PURE_ORIENTATION_REPAIR_RESULTS.md](PURE_ORIENTATION_REPAIR_RESULTS.md). App/PPT protection statements below refer to the earlier coverage-stage decision.

# Pure FusDenseNet training repair: measured findings

Status: finished. Completed new full-network epochs: **6**. Candidate tested: **False**. Passed deployment gate: **False**. Improved checkpoint installed: **False**. No hybrid, external encoder, teacher, inference ensemble, class threshold or logit offset is used.

## Why the earlier changes were not enough

1. The old sampling design repeatedly resampled natural faces within an epoch. In a measured epoch it drew 23,552 samples but visited only **14,720/23,503 distinct faces (62.63%)**. This is a coverage limitation, not a PyTorch error. Repeating the same 330 disgust faces did not create new identities.
2. Its normalized disgust CE multiplier was **0.5792**. Consequently the expected CE class mass was **2.45%**, despite a **4.22%** sampling prior. That compensating weight weakened the intended minority emphasis.
3. The verified baseline was brittle to the chosen augmentation. On the same 330 TRAIN disgust faces, raw recall was **95.45% (315/330)**, mirrored recall **35.45% (117/330)**, affine recall **70.30%**, and old photometric-only recall **92.42%**. These perturbations demonstrate orientation/spatial fragility, not held-out accuracy or proof that augmentation itself caused every failure.
4. On six training-only batches from the failed final v4 state, R-Drop's backbone gradient opposed the disgust CE direction in **5/6** cases and SupCon in **3/6**. Their gradient norms were smaller than CE; they are not established as the sole or dominant cause. The results show that adding auxiliary objectives was not uniformly helping the target.
5. The earlier baseline audit found 83.01% raw training versus58.04% validation accuracy, and95.45% versus28% disgust recall. Only330 unique training disgust faces and50 validation disgust faces are available. More native parameters did not remove this generalization problem. There is no evidence of a disconnected attention graph; earlier gradient contracts passed.

FER+ independently studies noisy facial-expression labels using multiple annotators: [Barsoum et al. (2016)](https://doi.org/10.48550/arXiv.1608.01041). Its [official annotation repository](https://github.com/microsoft/FERPlus) requires correspondence to the original FER2013 CSV row order. That mapping has not been verified for this workspace's JPEG archive, so no FER+ annotations are applied and no test labels are changed. This is context, not evidence that a particular project label is wrong.

## Repair implemented

The same1,828,896-parameter pure-v4 FusDenseNet path is fine-tuned end to end. Every training face appears in a shuffled natural pass each epoch. Each batch adds4disgust,2angry and2sad TRAIN-only exposures from cycling shuffled pools; angry/sad were among the original TRAIN disgust confusions. Every64-index batch is unique internally. Padding and actual unique/class counts are logged separately. These are repeated training exposures, not new images or people.

The objective is unit-weight supervised cross entropy, without compensating importance factors, label smoothing, SupCon or R-Drop. The classifier retains its original0.35dropout and old BatchNorm references; weights/affine parameters still learn. Half the examples are unmodified; half receive mild brightness/contrast jitter. No horizontal flip or affine augmentation is used in this repair. This preserves the available feature signal during fine-tuning; it does **not** establish orientation robustness. Backbone LR2e-5 and classifier LR3e-5 are lower than the failed recipe. Student and averaged-weight single-network checkpoints are selected on validation, never mixed at inference.

This combined repair is not an isolated equal-budget ablation. It addresses measured training weaknesses but cannot establish which individual change caused any gain. All source hashes, audit results, protocols and completed epoch states are preserved in reports/pure_repair_20261108/.

## Prospective selection and deployment rules

At most12epochs, minimum6before patience4,105-minute wall budget at completed-epoch boundaries. A conditional budget allocation declared after4completed validation epochs caps the run after at least6complete epochs ONLY if no checkpoint qualified; then the separate orientation-classifier repair can use remaining CPU time. The original protocol, adaptive allocation and discarded partial-next-epoch status are preserved. Validation accuracy, macroF1 and disgustF1 must exceed the earlier pure baseline, and disgust recall must gain at least8percentagepoints (28%to36%minimum). After all training, at most ONE qualifying checkpoint can be frozen for the full same-set test.

App/PPT changes require strictly better test accuracy,macroF1 and disgustF1, plus **at least5percentagepoints disgust recall gain** over36.71%. With79test disgust faces, this requires at least4additional correct disgust classifications. The same6,561retained test faces, all7classes and all errors are included. Single-view inference and positive validation-only temperature cannot change the predicted class. No alternate test candidate is allowed after the result. This is an engineering gate, not significance evidence. Historical test reuse and repeated validation prevent a fresh confirmatory claim.

## Every completed repair epoch

TRAIN agreement uses the augmented, oversampled distribution with dropout; validation uses raw natural-frequency images without dropout. They are not directly comparable generalization-gap estimates.

| New epoch | Validation accuracy | Macro F1 | Disgust recall | Disgust F1 | Unique train faces | Disgust draws |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.5796 | 0.5207 | 0.2600 | 0.2708 | 23503 | 2010 |
| 2 | 0.5756 | 0.5200 | 0.2800 | 0.2828 | 23503 | 2010 |
| 3 | 0.5773 | 0.5227 | 0.2800 | 0.3011 | 23503 | 2012 |
| 4 | 0.5796 | 0.5247 | 0.2800 | 0.3011 | 23503 | 2010 |
| 5 | 0.5763 | 0.5239 | 0.2800 | 0.3111 | 23503 | 2010 |
| 6 | 0.5756 | 0.5225 | 0.2800 | 0.3077 | 23503 | 2010 |

## Best observed validation diagnostic

Best by the prespecified validation objective; it is not automatically eligible or deployed.

| Metric | Earlier pure validation | New diagnostic | Difference |
|---|---:|---:|---:|
| accuracy | 0.580353 | 0.576322 | -0.004030 |
| macro precision | 0.524452 | 0.534995 | +0.010543 |
| macro recall | 0.525654 | 0.517669 | -0.007985 |
| macro f1 | 0.522483 | 0.523908 | +0.001425 |
| disgust precision | 0.333333 | 0.350000 | +0.016667 |
| disgust recall | 0.280000 | 0.280000 | +0.000000 |
| disgust f1 | 0.304348 | 0.311111 | +0.006763 |

## Full test result

No new candidate test result. The earlier pure test scores remain historical references; app/PPT must remain unchanged.

| Metric | Earlier pure test | New pure test | Difference |
|---|---:|---:|---:|


Current deployment decision: {"eligible": false, "test_evaluated": false, "installed": false, "reason": "No pure checkpoint passed the predeclared validation accuracy/F1/recall gate. App and PPT unchanged.", "protected_artifacts_unchanged": true, "new_epochs": 6}

The app/PPT protection checks before installation are recorded individually in repair_summary.json. Real EEG/HRV results and their negative fusion finding are unchanged. No new physiological, webcam-domain, weather/raga-response, therapeutic,80%accuracy or exact-author-reproduction claim is made. Primary architecture authority: https://doi.org/10.1007/s41870-025-02531-y .
