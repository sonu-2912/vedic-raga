# Audit-driven spatial generalization: actual results

Completed 6 full-network v5epochs and 58 classifier-only epochs across three head recipes; validation-qualified candidate: True; new candidate deployed: True. Active architecture version **7**; v6/v7 denote hybrid variants with a frozen ImageNet DenseNet121 branch; v7 averages probabilities from an independently trained head. Neither is a pure FusDenseNet result. Active full test accuracy **59.26%**, macro F1 **0.5530**, disgust recall **43.04%**. Active validation accuracy **60.28%**. No80% claim or reliable webcam recognition result.

## What held the previous model back

Measured raw TRAIN accuracy83.0107% versus validation58.0353%; disgust TRAIN recall95.4545% versus validation28.00%. These are different data splits, not a training improvement. TRAIN disgust330faces, happy6039; validation disgust50, so one example changes its recall by2percentage points. The classifier holds1,681,415of1,828,896parameters (91.94%). This combination is consistent with overfitting on limited minority-class diversity. Exact duplicate hashes were already excluded. No label-order bug was found: folder labels map explicitly to angry/disgust/fear/happy/sad/surprise/neutral. Attention gradients are nonzero at every stage; fused and spatial feature-zeroing diagnostics both reduce validation accuracy. Those diagnostics do not prove causal feature importance or isolate architecture.

Recomputing BN statistics with2,048training-only images reduced validation accuracy to57.5819% and disgust recall24%; rejected. Max mean activation shift was0.0738standard deviations, providing no evidence that stale BN references are the primary bottleneck. These are reused-validation diagnostics, not held-out gains.

## First experiment: v5 spatial refinement

v5 retains all three fusion-dense stages, cross-dense concatenation, cross-attention and multi-pooling. It adds two60-channel3x3convolutions as a residual refinement of first dense24x24features before spatial pooling (65,040additional parameters). Last convolution is zero-initialized, and real validation initialization logits matched v4 exactly (max absolute difference0). This is a disclosed project extension; exact author reproduction is not claimed.

Full-network optimization; old BN statistics frozen and new branch BN learned; classifier dropout.5; backbone/head LR5e-5, new branch3e-4; AdamW decay.003/.02/.003; mild affine, photometric augmentation and15%small-patch erasing. No own-teacher distillation or foreign teacher. No generated face images. Sampling prior q=.7p+.3/7 is derived only from TRAIN counts, yielding about1,238disgust draws/epoch versus330under natural sampling. Draws repeat real training images with random augmentation; they are not new people. Internal output bias starts at log(q/p); exported bias adds log(p/q). This matches initialization exactly and accounts for the sampling shift without a hand-set disgust threshold. The prior-shift assumption is approximate after finite training, not a performance guarantee.

Seed808; at most8epochs,80minutes checked at epoch boundaries, patience6. Validation selection requires both accuracy and disgust recall strictly improve and macro/disgust F1 not decrease; score.5accuracy+.3macroF1+.2disgustF1; student/EMA. At most one frozen qualifying candidate can be tested. The prospectively saved recall-priority gate requires test accuracy/macroF1/disgustF1 improve and disgust recall not decrease; all precision regressions are still reported. This differs from the preceding experiment's all-seven-floor gate and was declared BEFORE test access. No alternative is tested after a candidate fails.

Primary architecture authority: https://doi.org/10.1007/s41870-025-02531-y . Class-prior adjustment background: Menon et al., ICLR2021, https://doi.org/10.48550/arXiv.2007.07314 . The sampling-mixture/bias-folding recipe and residual refinement are project choices, not a claim to reproduce that paper's protocol.

## Complete new validation log

Training accuracy in machine logs is measured on augmented, oversampled q-distributed examples with dropout. Validation uses natural-prior-corrected student/EMA logits. They measure different conditions; their gap is not an ordinary generalization estimate.

| Epoch | Validation accuracy | Macro F1 | Disgust recall | Disgust F1 | Actual disgust draws |
|---|---:|---:|---:|---:|---:|
| 126 | 0.5673 | 0.5143 | 0.2200 | 0.2973 | 1305 |
| 127 | 0.5700 | 0.5082 | 0.1800 | 0.2687 | 1250 |
| 128 | 0.5788 | 0.5189 | 0.2200 | 0.2857 | 1259 |
| 129 | 0.5809 | 0.5164 | 0.1800 | 0.2535 | 1210 |
| 130 | 0.5829 | 0.5193 | 0.2000 | 0.2703 | 1250 |
| 131 | 0.5856 | 0.5223 | 0.2000 | 0.2740 | 1189 |

## Separate hybrid transfer experiment

This is **FusDenseNet v4 plus a frozen ImageNet DenseNet121 residual feature branch**, a disclosed architecture extension. It retains every FusDenseNet feature/classifier path and adds1024-channel features through a LayerNorm/1024->128->7 residual head. ONLY the new residual head was trained for8epochs; both encoders remained frozen. These epochs are not full-network training. Zero-initialized residual outputs preserve initial FusDenseNet predictions. No FER-pretrained teacher, FERPlus annotations, generated face data or altered evaluation labels were used. The ImageNet source was not trained with FER expression labels; complete upstream image overlap cannot be ruled out.

Official Torchvision ImageNet1K_V1 URL:https://download.pytorch.org/models/densenet121-a639ec97.pth; SHA:a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395; bytes:32342954. Native74.434% is ImageNet top1, never a FER/project result. Torchvision BSD3Clause notice preserved; no ImageNet images downloaded or redistributed. The224-pixel extraction attempt was abandoned before any feature-cache/model result dueCPUcost and remains preserved. The actual probe uses resize128/center112 with official ImageNet RGB normalization, a disclosedCPUadaptation differing from native224-pixel preprocessing. Exact cache/source SHA checks bind each frozen feature/label to its training/validation manifest. Selected composite inference is checked against cached computation before test. No sensor/body data contributes to this facial classification experiment.

AdamW LR.001/decay.01, CE power.25/smoothing.02, dropout.5, cosine20maximumepochs, patience8. A new residual head can learn complementary visual features while the original emotion model is retained; this is a hypothesis tested here, not guaranteed generalization or a causal controlled ablation. Added after v5epoch2validation and the negative translation screen, before any new test access; validation is repeatedly reused. Both recipes finish before one highest-scoring qualified candidate is frozen and tested. No alternative is tested after failure. Selection:{'run': 'reports/direct_dense_head_long_20261108', 'score': 0.5402229861791384, 'validation_accuracy': 0.6027707808564232, 'validation_macro_f1': 0.5497485075755532, 'validation_disgust_recall': 0.34}.

| Classifier-only epoch | Validation accuracy | Macro F1 | Disgust recall | Disgust F1 |
|---|---:|---:|---:|---:|
| 1 | 0.5796 | 0.5213 | 0.2200 | 0.2716 |
| 2 | 0.5859 | 0.5249 | 0.2200 | 0.2750 |
| 3 | 0.5854 | 0.5261 | 0.2200 | 0.2857 |
| 4 | 0.5899 | 0.5278 | 0.2200 | 0.2683 |
| 5 | 0.5882 | 0.5279 | 0.2200 | 0.2857 |
| 6 | 0.5912 | 0.5308 | 0.2200 | 0.2750 |
| 7 | 0.5884 | 0.5316 | 0.2200 | 0.2973 |
| 8 | 0.5940 | 0.5333 | 0.2200 | 0.2785 |

The separate five-view2-pixel translation screen measured validation accuracy58.06%, macroF10.5220, disgust recall28.00%; qualified:False. It was rejected and not used for test selection or deployment. It is not a new training result.

## Direct-label hybrid classifier and early-stopping correction

The residual-head recipe improved aggregate validation accuracy but did not improve disgust recognition. A third recipe trained the SAME new head directly on FER TRAIN labels instead of adding already-confident FusDenseNet training logits to its training loss. Source backbone weights and feature caches are unchanged. The old model recognizes95.45%oftraining disgust examples; under residual-only correction this leaves little error signal for a new classifier to learn those expressions. This is a motivating optimization diagnosis, not a controlled causal proof.

The direct head uses inverse-square-root frequency CE weights, no label smoothing, dropout.5, AdamW.001/decay.01. Inference combines the old calibrated probabilities (validation-fitted baseline temperature recorded in protocol) with new-head probabilities at one of3declared weights .25/.5/.75. No class-specific thresholds or excluded predictions. The core retains at least25%weight. This is **v7probability-ensemble hybrid**, NOTpureFusDenseNet and NOTthev6residual-logit recipe. It uses no new dataset images, synthetic faces or EEG/HRV labels.

The first direct-head trial stopped at8epochs with no qualified candidate, despite improving validation objectives. Its original protocol/logs/negative result are preserved. A separate run corrects the early-stopping rule: patience12tracks unconstrained validation objective progress, while acceptance still requires BOTH accuracy and disgust recall improve plus macro/disgustF1 floors. Maximum60epochs; actual completed42. Both feature extractors remain frozen; these are classifier-only epochs. Three probability weights and many epochs on repeatedly reused validation mean substantial selection bias; no fresh-confirmatory claim. All these choices were made before any new test access. At most ONE candidate across completed recipes is tested, with no alternate after failure.

| Direct-head epoch | Combined validation accuracy | Macro F1 | Disgust recall | New-head probability weight |
|---|---:|---:|---:|---:|
| 1 | 0.5849 | 0.5230 | 0.2400 | 0.25 |
| 2 | 0.5866 | 0.5237 | 0.2400 | 0.25 |
| 3 | 0.5892 | 0.5276 | 0.2400 | 0.25 |
| 4 | 0.5942 | 0.5340 | 0.2600 | 0.50 |
| 5 | 0.5924 | 0.5290 | 0.2400 | 0.25 |
| 6 | 0.5907 | 0.5277 | 0.2400 | 0.25 |
| 7 | 0.5927 | 0.5292 | 0.2400 | 0.25 |
| 8 | 0.5917 | 0.5284 | 0.2400 | 0.25 |
| 9 | 0.5909 | 0.5277 | 0.2400 | 0.25 |
| 10 | 0.5932 | 0.5304 | 0.2400 | 0.25 |
| 11 | 0.5897 | 0.5300 | 0.2800 | 0.50 |
| 12 | 0.5897 | 0.5230 | 0.2200 | 0.50 |
| 13 | 0.5952 | 0.5390 | 0.2800 | 0.50 |
| 14 | 0.5947 | 0.5335 | 0.2600 | 0.50 |
| 15 | 0.5960 | 0.5364 | 0.2800 | 0.50 |
| 16 | 0.5927 | 0.5304 | 0.2800 | 0.50 |
| 17 | 0.5909 | 0.5236 | 0.2200 | 0.50 |
| 18 | 0.5922 | 0.5280 | 0.2400 | 0.50 |
| 19 | 0.5942 | 0.5362 | 0.2800 | 0.50 |
| 20 | 0.5987 | 0.5395 | 0.3200 | 0.50 |
| 21 | 0.5922 | 0.5327 | 0.2800 | 0.50 |
| 22 | 0.5922 | 0.5319 | 0.3000 | 0.50 |
| 23 | 0.5967 | 0.5423 | 0.3600 | 0.50 |
| 24 | 0.5957 | 0.5345 | 0.3000 | 0.50 |
| 25 | 0.5962 | 0.5380 | 0.2800 | 0.50 |
| 26 | 0.6038 | 0.5483 | 0.3000 | 0.50 |
| 27 | 0.5990 | 0.5396 | 0.2600 | 0.50 |
| 28 | 0.6008 | 0.5434 | 0.3200 | 0.50 |
| 29 | 0.5982 | 0.5404 | 0.3000 | 0.50 |
| 30 | 0.6028 | 0.5497 | 0.3400 | 0.50 |
| 31 | 0.6045 | 0.5464 | 0.2800 | 0.50 |
| 32 | 0.6040 | 0.5479 | 0.3200 | 0.50 |
| 33 | 0.6003 | 0.5402 | 0.3000 | 0.50 |
| 34 | 0.6048 | 0.5445 | 0.2800 | 0.50 |
| 35 | 0.5977 | 0.5360 | 0.2600 | 0.50 |
| 36 | 0.6015 | 0.5414 | 0.2800 | 0.50 |
| 37 | 0.6015 | 0.5406 | 0.2800 | 0.50 |
| 38 | 0.6023 | 0.5444 | 0.3000 | 0.50 |
| 39 | 0.6020 | 0.5448 | 0.2800 | 0.50 |
| 40 | 0.6033 | 0.5422 | 0.2600 | 0.50 |
| 41 | 0.6015 | 0.5438 | 0.3000 | 0.50 |
| 42 | 0.6048 | 0.5479 | 0.3000 | 0.50 |

Standalone new-head validation metrics are saved separately, not treated as combined-model metrics. Training accuracy is the standalone head with dropout; combined validation accuracy uses BOTHencoders. Curves have different definitions and are not an ordinary generalization-gap estimate.

## Validation comparison

Frozen qualified candidate; N=3970, not test.

| Metric | Baseline | Candidate/diagnostic | Difference |
|---|---:|---:|---:|
| accuracy | 0.580353 | 0.602771 | +0.022418 |
| macro precision | 0.524452 | 0.554623 | +0.030170 |
| macro recall | 0.525654 | 0.551439 | +0.025785 |
| macro f1 | 0.522483 | 0.549749 | +0.027265 |
| disgust precision | 0.333333 | 0.404762 | +0.071429 |
| disgust recall | 0.280000 | 0.340000 | +0.060000 |
| disgust f1 | 0.304348 | 0.369565 | +0.065217 |

## Full test comparison

N=6561. All seven classes and all retained errors included.

| Metric | Previous model | New candidate | Difference |
|---|---:|---:|---:|
| accuracy | 0.574150 | 0.592593 | +0.018442 |
| macro f1 | 0.532040 | 0.553048 | +0.021008 |
| disgust recall | 0.367089 | 0.430380 | +0.063291 |
| disgust f1 | 0.417266 | 0.482270 | +0.065003 |
| macro precision | 0.539191 | 0.562781 | +0.023591 |
| macro recall | 0.532511 | 0.552622 | +0.020111 |
| disgust precision | 0.483333 | 0.548387 | +0.065054 |

The validation cohort is repeatedly reused and the combined Kaggle test is historically inspected. These are exploratory descriptive results, not fresh/identity-disjoint confirmation or significance evidence. FER2013 low-resolution expressions do not establish internal mood or webcam-domain accuracy. Real GamER fixed-game/30-second participant-held-out physiology results remain unchanged and show no primary fusion F1 gain. No new physiological, weather, raga-outcome or therapeutic efficacy is claimed.

Exact artifacts: reports/bottleneck_audit_20261108/audit.json and reports/spatial_generalization_20261108/. All original resources and prior failures preserved.
