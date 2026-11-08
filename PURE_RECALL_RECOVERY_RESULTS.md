# Pure FusDenseNet class-recall recovery experiment

Completed 8 November 2026. This report records a new experiment separately from the deployed model.

Deployment gate passed: **False**. New test evaluated: **False**. App/model/PPT protected hashes unchanged: **True**.

No validation checkpoint recovered all requested classes while retaining disgust and aggregate metrics. App/PPT unchanged.

## Actual validation results

Every row below uses all 3,970 retained validation faces, including all 50 disgust examples. Classifier rows are the best feasibility score within each recipe, not test results. Feasibility is a selection penalty, not a model-quality metric.

| Stage | Completed epochs | Accuracy | Macro F1 | Disgust P/R/F1 | Happy recall | Surprise recall | Neutral recall | Meets all gates |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Deployed pure baseline | â€” | 58.41% | 0.5397 | 0.3673/36.00%/0.3636 | 78.59% | 67.43% | 53.00% | baseline |
| common1.3_retain0.2 | 18 | 58.54% | 0.5400 | 0.3390/40.00%/0.3670 | 80.50% | 68.86% | 52.86% | False |
| common1.3_retain1.0 | 12 | 58.66% | 0.5357 | 0.2903/36.00%/0.3214 | 80.31% | 70.29% | 53.42% | False |
| common1.6_retain0.2 | 12 | 58.74% | 0.5375 | 0.3396/36.00%/0.3495 | 81.07% | 70.00% | 57.46% | False |
| common1.6_retain1.0 | 12 | 58.84% | 0.5381 | 0.3273/36.00%/0.3429 | 81.07% | 68.57% | 56.62% | False |
| Full network epoch 1 (EMA) | 1 | 58.87% | 0.5355 | 0.3404/32.00%/0.3299 | 81.64% | 69.43% | 57.74% | False |
| Full network epoch 2 (EMA) | 1 | 58.79% | 0.5358 | 0.3478/32.00%/0.3333 | 81.55% | 70.00% | 57.18% | False |
| Full network epoch 3 (EMA) | 1 | 59.07% | 0.5383 | 0.3404/32.00%/0.3299 | 81.45% | 70.29% | 57.74% | False |

## What changed and what failed

The corrected run completed 54 native-classifier epochs across four independent recipes, then 3 full-network epochs. These are different workloads and are not counted as 57 full-network epochs.

The previous recovery attempt accidentally reduced total happy/surprise/neutral class cost when downweighting their mirrored views. The corrected factors are 2/(1+lambda) for raw and 2lambda/(1+lambda) for mirrored views; their sum is two. The earlier interrupted run completed 21 classifier-only epochs, with its partial next epoch discarded; it is preserved in reports/pure_recall_recovery_20261108 and produced no test candidate. These 21 epochs are separate from the 54 corrected classifier epochs. This implementation error is verified algebraically and by a unit test, but is not claimed to explain every generalization failure.

Four native-head recipes increase the happy/surprise/neutral TRAIN loss cost and retain correct TRAIN disgust outputs from the same current pure classifier. The full-network stage also protects correct non-disgust examples through balanced disgust-vs-rest retention. Its deterministic retention branch bypasses classifier dropout; ordinary classification loss still uses dropout. This is TRAIN-only self-reference, not an external teacher or inference hybrid. All full-network emotion-path parameters can update; BatchNorm running statistics remain frozen. The reference is discarded for inference.

Raising common-class costs can recover those classes by shifting decision boundaries while harming disgust precision or other classes. Every requested class and all disgust precision/recall/F1 metrics are checked separately. A higher accuracy alone does not qualify.

TRAIN has 23,503 source images and only 330 disgust images, versus 6,039 happy images. Mirroring creates views, not new subjects. Validation has only 50 disgust cases. The spatial/attention backbone and small imbalanced data remain potential constraints; no isolated ablation here proves a single cause. Historical validation reuse and previously inspected test results limit independent confirmation.

## Test and deployment policy

One eligible candidate may be frozen after validation. The full 6,561-face test must improve accuracy and macro F1, preserve macro precision/recall and disgust precision/recall/F1, and restore happy/surprise/neutral recall to at least 81.59%/73.84%/58.54%. No threshold, alternate test candidate, external encoder or inference ensemble is allowed.

Measured generalization audit of the deployed pure checkpoint: original TRAIN accuracy 92.08%, mirrored TRAIN accuracy 76.07%, versus validation 58.41%. Disgust recall: original TRAIN 100.00%, mirrored TRAIN 96.67%, validation 36.00%. This is evidence of a large generalization gap, not proof of a particular single cause. Scores use cached TRAIN logits from the same verified checkpoint.

No new test accuracy, F1 or disgust recall is claimed: no validation candidate qualified. The deployed actual test metrics remain 58.34% accuracy, 0.5554 macro F1, and 46.84% disgust recall (37/79).

## Reproducibility and verification

Executed source snapshots, file hashes, protocols, every validation report, histories and decision JSON are retained in the run directory. Current checkpoint and PPT are not overwritten by this reporting script.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.report_recall_recovery --run-dir reports/pure_recall_recovery_v2_20261108
```

The completed run is immutable; training scripts reject completed/frozen run directories. A new future run requires a separate directory and prospectively declared protocol.

EEG/HRV results are unchanged. No new real physiological-fusion, weather-response, clinical benefit or raga-efficacy result is claimed. No user photograph was used for training or evaluation.

A direct checkpoint comparison after 1 full epoch(s) verified changes in 113 emotion-path parameter tensors, including 12 attention tensors. This verifies feature-learning updates occurred; it does not establish that those updates improved held-out performance. Detailed relative changes are saved in full_parameter_update_audit.json.

Test suite: {'tests': 92.0, 'failures': 0.0, 'errors': 0.0, 'skipped': 0.0, 'time': 133.712}.

## Conditional native-classifier feature-mixup experiment

Completed 79 classifier-only epochs across four independent recipes. No validation checkpoint passed all requirements; no new test candidate was evaluated. This phase ran only after the earlier round produced no qualifying candidate and used no new test. It reused the deployed pure backbone; failed full-network candidates are not its ancestors.

Training interpolates two TRAIN feature vectors and class targets, using 75% mixed-target CE, 25% unmixed CE and correct positive/negative TRAIN disgust-boundary retention from the same pure classifier. The reference is discarded for inference. No external encoder, inference probability ensemble or class offset is used. This is classifier-only regularization, not new backbone feature learning.

This frozen-feature adaptation is inspired by [Verma et al., Manifold Mixup, ICML 2019](https://proceedings.mlr.press/v97/verma19a.html), not an exact reproduction of joint hidden-representation training. Interpolated TRAIN features create no new subjects. Validation/test images are real, unchanged and unfiltered.

| Best feasibility validation checkpoint per recipe | Epochs | Accuracy | Macro F1 | Disgust P/R/F1 | Happy recall | Surprise recall | Neutral recall | Meets gates |
|---|---:|---:|---:|---|---:|---:|---:|---|
| mix0.2_common1.1 | 20 | 58.41% | 0.5394 | 0.3830/36.00%/0.3711 | 80.11% | 68.57% | 50.91% | False |
| mix0.2_common1.3 | 20 | 58.29% | 0.5378 | 0.3654/38.00%/0.3725 | 80.69% | 70.86% | 53.28% | False |
| mix0.4_common1.1 | 19 | 58.06% | 0.5353 | 0.3585/38.00%/0.3689 | 79.83% | 70.29% | 52.02% | False |
| mix0.4_common1.3 | 20 | 58.29% | 0.5349 | 0.3673/36.00%/0.3636 | 80.40% | 70.86% | 55.79% | False |

## Latest deployed PURE FusDenseNet test results

Accuracy **58.34%**, macro F1 **0.5554**, all 6561 retained test faces. These are unchanged existing results, not a newly evaluated candidate.

| Class | Recall | Support |
|---|---:|---:|
| angry | 48.65% | 888 |
| disgust | 46.84% | 79 |
| fear | 41.11% | 922 |
| happy | 78.61% | 1711 |
| sad | 46.16% | 1198 |
| surprise | 69.54% | 581 |
| neutral | 57.36% | 1182 |

Final software tests: {'tests': 94.0, 'failures': 0.0, 'errors': 0.0, 'skipped': 0.0, 'time': 129.343}. All seven protected model/app/report/PPT hashes remain unchanged. App and PPT were not updated because no candidate met the requested improvement conditions.

All failed recipe histories, student/EMA reports, exact source snapshots and decision JSON are retained. Historical validation/test reuse limits independent confirmation. No physiological, weather, therapeutic or sensor improvement is claimed.
