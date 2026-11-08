# Pure FusDenseNet capacity-control results

Experiment started8November2026; report generated 2026-11-09T00:22:53.617775+05:30. These are actual measured outcomes, not original-paper results.

Eligible for deployment: **False**. New full-test candidate evaluated: **False**. All protected model/app/PPT hashes unchanged before any integration: **True**.

No compact pure FusDenseNet validation checkpoint recovered all requested classes while retaining disgust and aggregate metrics. App/PPT unchanged.

## Architecture and training changes

The deployed 1,828,896-parameter model has 1,681,415 classifier parameters (91.94%). Its original TRAIN accuracy is92.08%, versus58.41% validation accuracy; TRAIN disgust recall100% versus36% validation. This proves a large generalization gap, not a specific single cause. Previous reweighting/mixup trials did not yield a qualifying replacement.

This experiment keeps the native stem, three fusion-dense stages, cross-dense concatenations, dilated convolutions, cross-attention and final mean/max/RMS/2x2 pooling. A fixed differentiable spatial transform coarsens the two fine4x4 shortcut grids into2x2 mean/max regions and normalizes each concatenated shortcut. Feature width3276 becomes1260. Smaller randomly initialized128/256 classifiers have309,792/472,096 total parameters, respectively, reducing total capacity83.06%/74.19%. These are explicit project choices, not recovered author settings.

Four independent native heads use normalized original/mirrored TRAIN features,10% feature dropout,50% hidden dropout,label smoothing.025,AdamW decay.1 and varying common-class loss costs. The TRAIN-only standard-deviation floor is.05; 172 of1260features fall below it. A smaller train/validation gap with lower validation accuracy is underfitting, not an improvement.

The next phase fine-tunes the full native feature path with mild TRAIN-only affine/brightness/contrast augmentation and complementary mirror views. BatchNorm running statistics stay frozen. The training reference is the original pure classifier on correctly labelled TRAIN disgust-vs-rest positive/negative examples, discarded at inference. No external encoder, hybrid inference, probability ensemble, class offset or held-out target in training is used.

The pooling/capacity/optimizer/augmentation changes form one exploratory engineering intervention, not an isolated equal-compute causal ablation. New heads are independently initialized; rejected recipes are not ancestors of selected weights. Head and full-network epochs have different cost.

Completed: 279 classifier epochs across four recipes, plus 6 full-network epochs. All23,503natural training sources are seen in each full epoch. Two orientations are views of existing TRAIN images, not new subjects.

## Actual full-validation results

All3,970retained validation faces and all50disgust examples are included. Per-recipe rows are the highest feasibility checkpoint, not cherry-picked test results. Feasibility penalizes unmet gates and is not an accuracy/F1 metric.

| Stage | Epochs | Accuracy | Macro F1 | Disgust P/R/F1 | Happy recall | Surprise recall | Neutral recall | Meets all gates |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Deployed pure baseline | - | 58.41% | 0.5397 | 0.3673/36.00%/0.3636 | 78.59% | 67.43% | 53.00% | baseline |
| coarse128_common1.05 | 87 | 57.20% | 0.5158 | 0.2754/38.00%/0.3193 | 80.59% | 70.57% | 56.62% | False |
| coarse128_common1.2 | 1 | 48.61% | 0.4067 | 0.0648/50.00%/0.1147 | 77.82% | 69.43% | 56.35% | False |
| coarse256_common1.05 | 91 | 58.31% | 0.5360 | 0.3774/40.00%/0.3883 | 81.45% | 70.29% | 56.35% | False |
| coarse256_common1.2 | 100 | 58.41% | 0.5348 | 0.3800/38.00%/0.3800 | 80.78% | 72.29% | 60.95% | False |
| Full epoch1 (student) | 1 | 58.11% | 0.5291 | 0.3333/40.00%/0.3636 | 79.83% | 70.86% | 67.92% | False |
| Full epoch2 (student) | 1 | 57.78% | 0.5281 | 0.3600/36.00%/0.3600 | 80.02% | 70.86% | 60.95% | False |
| Full epoch3 (EMA) | 1 | 58.46% | 0.5341 | 0.3226/40.00%/0.3571 | 80.59% | 72.00% | 61.23% | False |
| Full epoch4 (student) | 1 | 58.29% | 0.5299 | 0.3276/38.00%/0.3519 | 79.64% | 72.00% | 65.69% | False |
| Full epoch5 (EMA) | 1 | 58.26% | 0.5322 | 0.3279/40.00%/0.3604 | 80.21% | 72.57% | 61.23% | False |
| Full epoch6 (EMA) | 1 | 58.29% | 0.5309 | 0.3125/40.00%/0.3509 | 80.11% | 72.86% | 62.06% | False |

Timing limitation: long wall-clock intervals of716s and1655s occurred during otherwise4-6s head epochs; the pause cause is unconfirmed. The128/common1.2 recipe therefore stopped after one completed epoch and is undertrained, not a comparable converged fit. Its output is preserved and not hidden. Before full-network training/test access, the timing-only amendment switched the full budget to process CPU seconds divided by4threads and enforced at least3completed epochs. The6epoch maximum, training losses and all validation/test gates stayed unchanged. Both source versions and the declaration are retained in timing_amendment/.

## Frozen-candidate policy and actual test outcome

Same6561testfaces. Strict accuracy/macroF1 gains over current58.3448%/.555367. No reduction in current macroP/R or disgustP/R/F1 (retain37/79disgust correct). Happy/surprise/neutral recall recover at least81.5897%/73.8382%/58.5448%. No alternate testcandidate; if any condition fails, app/PPT unchanged.

No alternate candidate is tested after a test failure. Every test class, face and error remains included. Positive validation temperature calibration changes confidence, not argmax decisions.

No validation checkpoint qualified, so no new test accuracy or recall is claimed. The existing deployed full-test result remains58.34% accuracy,0.5554macroF1,46.84%disgust recall (37/79). The app and PPT remain unchanged.

## Verification and limits

Four new contract tests passed (18.68seconds): native-vs-cached feature equivalence, retained attention gradients, checkpoint roundtrip and detached training references. The raw-image preflight used64validation images; maximum feature discrepancy2.384e-6. It read no test images. No user photograph or sensor data was used.

Validation has been repeatedly reused and the historical test previously inspected. This is not fresh subject-disjoint, webcam-domain or clinical confirmation. EEG/HRV and weather efficacy results are unchanged. Failure to meet the gate is reported; no80% result or improvement from parameter reduction alone is claimed.

Protocols, executed source hashes, normalization audit, preflight, every student/EMA validation report, histories and deployment decision remain in reports/pure_capacity_20261108/. Experimental architecture code is isolated in scripts/compact_fusdense.py; the live backend is modified only after a successful full-test gate.

Rebuild this report after completion:

```powershell
.\.venv\Scripts\python.exe -m scripts.report_compact_capacity
```

## Existing-model feature-group sensitivity

A fixed deployed model was evaluated on all3970validation faces after replacing one group by its TRAIN mean. This is an input sensitivity check, not a retrained/causal ablation or a test result. No test data was read.

| Replaced group | Validation accuracy | Macro F1 | Disgust recall |
|---|---:|---:|---:|
| unmodified | 58.41% | 0.5397 | 36.00% |
| replace_fusion_attention_features_by_train_mean | 30.68% | 0.2665 | 10.00% |
| replace_early_spatial_features_by_train_mean | 56.60% | 0.5053 | 24.00% |
| replace_dense_spatial_features_by_train_mean | 51.66% | 0.4608 | 20.00% |
