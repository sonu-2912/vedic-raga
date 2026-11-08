# Compact native residual FusDenseNet results

2026-11-09T01:19:24.666229+05:30

Completed4additional full-network epochs. This phase ran only after the earlier compact6-epoch phase and all three BN checks produced no eligible/tested candidate. No earlier new test was opened.

Version11 keeps the original native FusDenseNet stem, three fusion-dense stages, cross-dense connections, attention and global/coarse pooling. It adds zero-output native residual dense refinements in each fusion branch, using two dilated dense growth layers, GroupNorm and local1x1feature fusion. The compact1260-to256-to7classifier is retained. This is a disclosed project extension, not an exact recovered author topology. No external encoder, teacher from another architecture or probability ensemble is used.

Total parameters589,456;117,360new feature parameters. Initial predictions matched the final compact source on64real validation faces with maximum logit discrepancy0.0. Three new tests passed in21.81seconds: exact initial-function preservation, gradients reaching new residual filters, and portable checkpoint loading.

The normalization component follows [Wu and He, Group Normalization (2018)](https://arxiv.org/abs/1803.08494); their published results are not project results.

New feature modules use LR1e-4, retained native feature parameters1e-5, classifier3e-5; mild TRAIN image augmentation, dropout.5,smoothing.025 and same-current-pure TRAIN disgust-boundary retention. TRAIN class-cost multipliers: happy1.25,surprise1.25,neutral1.0. This addresses excessive neutral allocation seen in the compact phase through supervised training costs, not inference offsets. The joint architecture/loss/optimizer change is not a causal ablation.

Earlier279classifier trial epochs were independent recipes, including one timing-limited one-epoch recipe. The accepted source lineage contains only its selected classifier warm-up, the6compact full-network passes and these additional native refinement passes. Full/head epochs are not equivalent compute. No failed BN-recalibrated weights initialize this phase.

All validation rows use the same3970real faces. Student and averaged models are separately evaluated; each is a single pure network, not an inference mixture.

| Model/stage | Accuracy | Macro F1 | Disgust P/R/F1 | Happy recall | Surprise recall | Neutral recall |
|---|---:|---:|---|---:|---:|---:|
| Current PURE validation | 58.41% | 0.5397 | 0.3673/36.00%/0.3636 | 78.59% | 67.43% | 53.00% |
| Native refinement epoch1 student | 58.44% | 0.5329 | 0.3393/38.00%/0.3585 | 81.26% | 74.00% | 56.21% |
| Native refinement epoch1 EMA | 57.73% | 0.5250 | 0.3175/40.00%/0.3540 | 81.55% | 73.71% | 55.79% |
| Native refinement epoch2 student | 57.98% | 0.5275 | 0.3333/38.00%/0.3551 | 81.36% | 73.71% | 56.76% |
| Native refinement epoch2 EMA | 57.93% | 0.5272 | 0.3125/40.00%/0.3509 | 81.36% | 73.14% | 56.90% |
| Native refinement epoch3 student | 58.16% | 0.5273 | 0.2817/40.00%/0.3306 | 79.73% | 73.43% | 60.39% |
| Native refinement epoch3 EMA | 57.96% | 0.5281 | 0.3279/40.00%/0.3604 | 81.55% | 73.71% | 56.90% |
| Native refinement epoch4 student | 57.76% | 0.5240 | 0.2985/40.00%/0.3419 | 80.98% | 73.71% | 57.60% |
| Native refinement epoch4 EMA | 58.01% | 0.5279 | 0.3175/40.00%/0.3540 | 81.36% | 73.71% | 57.32% |

## Test and deployment decision

No native residual compact validation checkpoint recovered all requested classes while retaining disgust and aggregate metrics. App/PPT unchanged.

New test evaluated: False. Eligible: False. All prospective accuracy/macroF1, macroP/R, disgustP/R/F1 and historical happy/surprise/neutral recovery floors remain unchanged. No alternate model is tested after a test failure.

No new test result is claimed. The deployed pure result remains58.34% accuracy,0.5554macroF1,46.84%disgust recall. No app or PPT replacement is justified by this phase.

Protected app/model/PPT files all unchanged before any integration: True.

All classes, faces and errors remain included. No uncertainty-based filtering, post-test thresholds, external encoder or probability mixture is used. TRAIN/validation/test cohorts are fixed. This is exploratory engineering with repeated validation use and historical test inspection, not clinical or webcam-domain confirmation. EEG/HRV/weather/raga-efficacy results are unchanged. No user photograph or real sensor was used.

## Fixed-checkpoint overfitting audit

All23503original TRAIN faces, eval mode, no augmentation/dropout: baseline TRAINaccuracy 92.08%, validation 58.41%, gap 33.66%; candidate TRAINaccuracy 72.52%, validation 58.01%, gap 14.51%. These are comparable fixed-checkpoint measurements, not epoch-running augmented training accuracy. A smaller gap alone is not a successful improvement.
