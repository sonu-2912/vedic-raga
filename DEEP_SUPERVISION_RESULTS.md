# Further balanced training experiments: actual results

Completed 5 auxiliary-supervised and 8 conservative full emotion-path epochs, plus 20 separate frozen-backbone classifier epochs. Validation-qualified replacement: **True**. Selected recipe: **reports/consistency_long_20261107**. New candidate deployed: **False**. Active test accuracy **57.42%**, macro F1 **0.5320**; active validation accuracy **58.04%**. No80% result is claimed. Active source SHA `8fc1713541e49bac8fc6ec4f10a5d5ba4ca9ff3860e227caeda3b7b67870a618`.

## Method and data

Same verified23,503training/3,970validation faces. All candidates retain FusDenseNet v4 mechanisms. Active inference: 1,828,896 parameters,3276→512→7 classifier. The full-feature recipe keeps the512-unit head; the separate capacity recipe changes only its width to256, a disclosed project dimension choice. All three fusion stages, cross-attention and multi-pooling remain. A training-only LayerNorm/588→128→7 companion supervises only the late fused feature slice, so its gradients reach dense fusion/attention without using the spatial shortcut. The companion is absent from the saved inference state. In the auxiliary recipe only, training masks the early spatial shortcut in20%of examples with inverted scaling; evaluation never masks it. MixUp Beta(.2,.2) applies to half of batches and both labels receive weighted CE. Weight power.25, label smoothing.02, main+ .25companion loss. Early LR1e-5, late2e-4, main classifier1e-4, companion1e-3; two-epoch warmup then cosine decay, frozen BN running references, EMA/student validation. No external teacher or unverified FER+ labels.

This is a disclosed project training extension, not exact author reproduction or a controlled causal architecture ablation. Basis: Lee et al. (2015), Deeply-Supervised Nets, https://proceedings.mlr.press/v38/lee15a.html . MixUp: Zhang et al., https://arxiv.org/abs/1710.09412 . Primary authority remains https://doi.org/10.1007/s41870-025-02531-y .

## Separate capacity-control recipe

A second exploratory validation-only recipe was declared after two deep-supervision epochs, before any new test access. It freezes the baseline feature extractor and ranks512hidden neurons by class-balanced mean training activation multiplied by output-weight contribution. It retains256units, folds the mean discarded contribution into the output bias, and retrains that smaller MLP on training-normalized features. AdamW.001, weight decay.003, dropout.5, CE weight power.25/smoothing.02 plus.1correct-teacher KL T2. Neither normalization, neuron ranking nor teacher targets use validation/test labels. The final normalization is algebraically folded into the first Linear layer; full validation predictions must remain identical after conversion. This is classifier-only training, not additional full-network epochs. The initial full512-unit baseline defines the same seven validation floors.

Source-feature weights are copied unchanged for a pruned candidate. This is not a causal capacity ablation: regularization and optimization also differ. All four candidate runs and negative results are retained. Only the highest-scoring qualifying validation candidate is frozen for one test evaluation; no alternate candidate may be tested afterward. Validation is exploratory and repeatedly reused, not independent.

## Final fast classifier optimizer comparison

A fourth and final exploratory recipe, declared during the continuation before test access, keeps the full512-unit classifier and frozen baseline features. It uses sharpness-aware optimization (rho.05) with AdamW LR.0001/wd.003, dropout.45, weighted CE power.25/smoothing.02 and.1correct-own-teacher KL T2. Both gradient passes reuse the same dropout masks. Original parameters are restored exactly before the optimizer update, including on second-pass failure. Normalization uses training features only and is folded back with validation-logit/argmax checks. It is classifier-only training, not extra full-network epochs or an architectural change. Basis: Foret et al. (2021), https://doi.org/10.48550/arXiv.2010.01412 . This project recipe is not exact original SAM/FusDenseNet reproduction.

## Longer continuation and rejected foreign teacher

After the negative first recipes, a third exploratory run continued the previously successful own-teacher method from the same deployed baseline. Backbone LR2e-5, classifier5e-5, CE class-weight power.25/smoothing.02, dropout.45, mild affine/brightness/contrast augmentation and.2correct-teacher KL T2. Teacher inputs are only source/manifest-SHA-bound TRAIN features from the same baseline; raw teacher inference matches its deployed single-view mode. All training faces still receive CE. No MixUp, companion, shortcut dropout or external teacher contributes to this continuation. Student/EMA selection uses the same seven floors and.35accuracy+.65macroF1. Maximum12epochs, patience8,90-minute budget at completed-epoch boundaries.

An official Apache2-licensed OpenCV MobileFaceNet teacher was separately screened on all3,970fixed validation faces using replicated grayscale/112×112 inputs without invented landmarks. It measured43.4509% accuracy, below the declared screen threshold, and was **rejected and never used to train FusDenseNet**. Its published88.27% belongs to RAF-DB, not this project. The foreign checkpoint's complete training-image manifest is unavailable, so upstream overlap is unknown. This adapted diagnostic is not a clean independent benchmark or a FusDenseNet result. Original weights/notice/SHA were preserved locally; no foreign pixels or weights are published with the project. Exact screen: reports/teacher_screen_20261107/.

## Selection safeguards

The auxiliary full-network run uses maximum12epochs, patience5 and a90-minute budget checked after completed epochs (a long/interrupted epoch can overshoot). Score.35accuracy+.65macro F1, only if accuracy, macro precision/recall/F1 and disgust precision/recall/F1 all meet the initial raw-validation floors. Student and EMA are compared each epoch. Flip averaging/calibration use validation only; strict flip selection also prevents seven-metric regression versus raw. The separate evaluation protocol was saved before test access. Only one frozen qualifying candidate can be tested; strict integration requires positive accuracy/macro F1/disgust F1 and none of the seven declared metrics decreasing. The previous evaluated model remains active if no candidate qualifies or any strict test check fails.

## Complete new validation log

| Epoch | Accuracy | Macro precision | Macro F1 | Disgust precision | Disgust recall | Disgust F1 |
|---|---:|---:|---:|---:|---:|---:|
| 126 | 0.5494 | 0.5440 | 0.4995 | 0.4138 | 0.2400 | 0.3038 |
| 127 | 0.5587 | 0.5242 | 0.4933 | 0.3333 | 0.1600 | 0.2162 |
| 128 | 0.5529 | 0.5262 | 0.4908 | 0.4000 | 0.1600 | 0.2286 |
| 129 | 0.5670 | 0.5387 | 0.4997 | 0.4118 | 0.1400 | 0.2090 |
| 130 | 0.5733 | 0.5320 | 0.5066 | 0.2558 | 0.2200 | 0.2366 |

Rows include rejected epochs. Raw student/EMA reports for every epoch are retained. Training agreement is lambda-weighted agreement with mixed labels under augmentation/dropout; it is not ordinary facial classification accuracy. Elapsed epoch seconds can include pauses and are not CPU-time measurements. Best observed diagnostic: validation_03_SAM.json; it is not necessarily eligible. Selected inference checkpoint index: 133.

## Complete classifier-only validation log

| Recipe | Epoch | Accuracy | Macro precision | Macro F1 | Disgust precision | Disgust recall | Disgust F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pruned256 | 126 | 0.5683 | 0.5319 | 0.5144 | 0.3939 | 0.2600 | 0.3133 |
| Pruned256 | 127 | 0.5632 | 0.5369 | 0.5187 | 0.4828 | 0.2800 | 0.3544 |
| Pruned256 | 128 | 0.5683 | 0.5364 | 0.5083 | 0.4737 | 0.1800 | 0.2609 |
| Pruned256 | 129 | 0.5673 | 0.5319 | 0.5225 | 0.4324 | 0.3200 | 0.3678 |
| Pruned256 | 130 | 0.5660 | 0.5377 | 0.5165 | 0.4444 | 0.2400 | 0.3117 |
| Pruned256 | 131 | 0.5587 | 0.5156 | 0.4970 | 0.3333 | 0.1800 | 0.2338 |
| Pruned256 | 132 | 0.5685 | 0.5336 | 0.5188 | 0.4516 | 0.2800 | 0.3457 |
| Pruned256 | 133 | 0.5705 | 0.5330 | 0.5157 | 0.4138 | 0.2400 | 0.3038 |
| Pruned256 | 134 | 0.5655 | 0.5367 | 0.5102 | 0.4783 | 0.2200 | 0.3014 |
| Pruned256 | 135 | 0.5695 | 0.5423 | 0.5233 | 0.4828 | 0.2800 | 0.3544 |
| SAM512 | 126 | 0.5804 | 0.5478 | 0.5296 | 0.4643 | 0.2600 | 0.3333 |
| SAM512 | 127 | 0.5798 | 0.5451 | 0.5265 | 0.4444 | 0.2400 | 0.3117 |
| SAM512 | 128 | 0.5816 | 0.5582 | 0.5333 | 0.5200 | 0.2600 | 0.3467 |
| SAM512 | 129 | 0.5783 | 0.5428 | 0.5276 | 0.4333 | 0.2600 | 0.3250 |
| SAM512 | 130 | 0.5798 | 0.5497 | 0.5302 | 0.4643 | 0.2600 | 0.3333 |
| SAM512 | 131 | 0.5791 | 0.5514 | 0.5299 | 0.4815 | 0.2600 | 0.3377 |
| SAM512 | 132 | 0.5786 | 0.5471 | 0.5282 | 0.4643 | 0.2600 | 0.3333 |
| SAM512 | 133 | 0.5773 | 0.5457 | 0.5278 | 0.4643 | 0.2600 | 0.3333 |
| SAM512 | 134 | 0.5781 | 0.5508 | 0.5300 | 0.4815 | 0.2600 | 0.3377 |
| SAM512 | 135 | 0.5768 | 0.5414 | 0.5240 | 0.4286 | 0.2400 | 0.3077 |

## Complete conservative full-network validation log

| Epoch | Accuracy | Macro precision | Macro F1 | Disgust recall | Disgust F1 |
|---|---:|---:|---:|---:|---:|
| 126 | 0.5801 | 0.5321 | 0.5246 | 0.2600 | 0.3133 |
| 127 | 0.5773 | 0.5275 | 0.5200 | 0.2400 | 0.2963 |
| 128 | 0.5758 | 0.5291 | 0.5208 | 0.2600 | 0.3171 |
| 129 | 0.5814 | 0.5335 | 0.5256 | 0.2600 | 0.3133 |
| 130 | 0.5821 | 0.5388 | 0.5278 | 0.2600 | 0.3250 |
| 131 | 0.5786 | 0.5373 | 0.5255 | 0.2600 | 0.3291 |
| 132 | 0.5841 | 0.5485 | 0.5307 | 0.2600 | 0.3377 |
| 133 | 0.5809 | 0.5399 | 0.5285 | 0.2800 | 0.3457 |

## Validation comparison (qualifying candidate)

| Metric | Initial | Candidate/diagnostic | Difference |
|---|---:|---:|---:|
| accuracy | 0.580353 | 0.580856 | +0.000504 |
| macro precision | 0.524452 | 0.539872 | +0.015420 |
| macro recall | 0.525654 | 0.525989 | +0.000335 |
| macro f1 | 0.522483 | 0.528514 | +0.006030 |
| disgust precision | 0.333333 | 0.451613 | +0.118280 |
| disgust recall | 0.280000 | 0.280000 | +0.000000 |
| disgust f1 | 0.304348 | 0.345679 | +0.041331 |

## Full same-set test comparison

| Metric | Before | Candidate | Difference |
|---|---:|---:|---:|
| accuracy | 0.574150 | 0.572931 | -0.001219 |
| macro f1 | 0.532040 | 0.533686 | +0.001645 |
| disgust recall | 0.367089 | 0.354430 | -0.012658 |
| disgust f1 | 0.417266 | 0.427481 | +0.010215 |
| macro precision | 0.539191 | 0.545808 | +0.006617 |
| macro recall | 0.532511 | 0.530794 | -0.001717 |
| disgust precision | 0.483333 | 0.538462 | +0.055128 |

Every test face and all7classes/errors are retained. The6,561-face combined Kaggle test was historically inspected and validation repeatedly reused. It is not a fresh identity-disjoint or webcam benchmark. No claim follows that every user's disgust or smile will be recognized correctly. A weight-change audit proves training occurred, not improved generalization by itself.

## Other metrics remain separate

The real GamER fixed-game/fixed30s/participant-held-out EEG/ECG results still show no primary fusion F1 gain. No new synchronized face/physiology/raga/weather dataset was acquired. DENS dimension issues and DREAMER restricted access remain unresolved. No simulated signal is reported as real, and no recommendation-quality or therapeutic metric is invented.

## Reproduce

The completed evaluator is locked after test access and verifies/reuses its saved bundle. New prospective training requires a separate protocol and run directory. Original datasets, weights, reports and source resources remain preserved.

```powershell
.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/balanced_selected_20261107 --device cpu --threads 4 --strict-all-metrics --baseline-report reports/deep_supervision_20261107/baseline_test.json --baseline-manifest reports/deep_supervision_20261107/baseline_test_manifest.json
.\.venv\Scripts\python.exe -m scripts.report_deep_supervision
```

If no candidate qualified, do not run the evaluator to create a baseline-only test result; the report records that test was intentionally not accessed. Exact protocol, per-epoch reports, auxiliary optimizer state for recovery, tensor changes and manifest hashes are under reports/deep_supervision_20261107/. DATASETS.md contains access and licence information.
