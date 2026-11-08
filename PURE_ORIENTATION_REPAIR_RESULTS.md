# Pure FusDenseNet orientation repair: actual results

Independent full-network coverage stage: **6 completed epochs**, no qualifying candidate and no test evaluation. The subsequent native-classifier stage completed **72 classifier-only epochs** across two independent widths,512 and256. These are different compute budgets and are not equivalent full-network epochs. The accepted candidate's backbone, if any, comes directly from the earlier pure checkpoint, not from the failed coverage run.

Passed full test gate: **True**. New checkpoint installed: **True**. Candidate actually tested: **True**. Selected classifier: **head512**. Inference parameters: **1828896**. No external encoder, teacher logits/distillation, probability mixture, inference ensemble, class threshold or post-hoc class-logit offset is used.

## What the failure analysis established

The earlier sampler visited only62.63% of distinct training images in one measured epoch. Its disgust CE multiplier0.579 weakened the intended minority emphasis. However, fixing coverage and loss alone did not improve the held-out target: training loss fell while validation disgust recall remained around28%. Those fixes were not sufficient for generalization.

The baseline's TRAIN disgust recall was315/330(95.45%) on original views but117/330(35.45%) on their mirrors. The classifier contained1,681,415of1,828,896parameters and depended on orientation-sensitive spatial features. This motivated teaching its own classifier both views before changing its backbone further. These diagnostics motivate the repair; they do not establish an isolated causal attribution. Detailed six-batch gradient and augmentation audits: PURE_REPAIR_ANALYSIS.md and reports/pure_repair_20261108/.

## Training performed

Both raw and horizontally mirrored TRAIN representations were computed by the **same original pure FusDenseNet**, source SHA8fc1713541e49bac8fc6ec4f10a5d5ba4ca9ff3860e227caeda3b7b67870a618. They are not features from DenseNet121, Navrasa or another encoder. There are47,006augmented training views from23,503source images, including660disgust views from330source disgust images; these are not new people. Exact mirror-pixel overlaps with held-out raw validation/test input hashes were both zero. This is not an identity/near-duplicate audit.

The native MLP learned using sqrt-inverse-TRAIN-frequency weighted cross entropy, dropout0.5, AdamW and validation selection. No auxiliary teacher, RDrop or SupCon loss was used. One512-unit head retained its dimensions; the other used TRAIN-only activation ranking to prune to256units. Mean/std were fitted to TRAIN views only and folded algebraically into the existing Linear layer before export. Full-validation argmax equivalence of normalized versus folded inference was checked. Every non-classifier tensor remains bit-identical to the earlier pure model; this is a classifier improvement, not a claim that new fusion-block features were learned.

Selection required improved validation accuracy,macroF1,disgustF1 and at least8percentagepoints disgustrecall gain. Both head trials finished before freezing at most one candidate. A fresh full-network single-view validation preflight confirmed the selected cache-based result before any new test inference. If the earlier full-network stage had produced a test candidate, this entire fallback stage would have been skipped, regardless of its test result.

## Selected validation on all3,970faces

| Metric | Earlier pure | Selected pure | Difference |
|---|---:|---:|---:|
| accuracy | 0.580353 | 0.584131 | +0.003778 |
| macro precision | 0.524452 | 0.540370 | +0.015917 |
| macro recall | 0.525654 | 0.540175 | +0.014521 |
| macro f1 | 0.522483 | 0.539698 | +0.017214 |
| disgust precision | 0.333333 | 0.367347 | +0.034014 |
| disgust recall | 0.280000 | 0.360000 | +0.080000 |
| disgust f1 | 0.304348 | 0.363636 | +0.059289 |

## Full same-set test on all6,561faces

All7classes,79disgust faces and every error are included. App/PPT changes require strictly positive accuracy,macroF1 and disgustF1 gains plus at least5percentagepoints disgust recall gain (at least4additional correct disgust cases outof79). No alternate test candidate, test-label relabelling, abstention filter or favorable subset was used. Validation-only positive temperature calibration cannot change the predicted class.

| Metric | Earlier pure test | New pure test | Difference |
|---|---:|---:|---:|
| accuracy | 0.574150 | 0.583448 | +0.009297 |
| macro precision | 0.539191 | 0.557472 | +0.018281 |
| macro recall | 0.532511 | 0.554650 | +0.022140 |
| macro f1 | 0.532040 | 0.555367 | +0.023327 |
| disgust precision | 0.483333 | 0.506849 | +0.023516 |
| disgust recall | 0.367089 | 0.468354 | +0.101266 |
| disgust f1 | 0.417266 | 0.486842 | +0.069576 |

Deployment decision: {"eligible": true, "test_evaluated": true, "installed": true, "reason": "Passed full same-set test accuracy/macroF1/disgustF1 and at least5percentagepoint disgustrecall gain", "minimum_disgust_recall_gain": 0.05, "metrics": {"accuracy": {"baseline": 0.574150281969212, "colab": 0.5834476451760402, "difference": 0.009297363206828169}, "macro_f1": {"baseline": 0.5320404110907176, "colab": 0.555367099062993, "difference": 0.02332668797227533}, "disgust_recall": {"baseline": 0.3670886075949367, "colab": 0.46835443037974683, "difference": 0.10126582278481011}, "disgust_f1": {"baseline": 0.4172661870503597, "colab": 0.4868421052631579, "difference": 0.06957591821279818}, "macro_precision": {"baseline": 0.5391905789239385, "colab": 0.5574717626824482, "difference": 0.018281183758509734}, "macro_recall": {"baseline": 0.5325106108129906, "colab": 0.5546504859140958, "difference": 0.022139875101105266}, "disgust_precision": {"baseline": 0.48333333333333334, "colab": 0.5068493150684932, "difference": 0.02351598173515984}}, "checkpoint_sha256": "f0082aff0511c40d8dff89fc03bbea965b416db55a0f771701efaa38f490c548", "protected_artifacts_unchanged": true, "new_head_epochs": 72, "limitation": "Historically inspected test and reused validation, not fresh identity-disjoint confirmation or significance evidence.", "source_tests_passed": 85, "source_test_seconds": 118.56}

## Original/mirrored TRAIN diagnostic

These examples were used in fitting and are **not** held-out accuracy. Measured correct counts/recalls: {"original": {"n_train_disgust": 330, "baseline_correct": 315, "candidate_correct": 330, "baseline_recall": 0.9545454382896423, "candidate_recall": 1.0, "scope": "TRAIN examples used for fitting; diagnostic only, not held-out evidence."}, "mirrored": {"n_train_disgust": 330, "baseline_correct": 117, "candidate_correct": 319, "baseline_recall": 0.3545454442501068, "candidate_recall": 0.9666666388511658, "scope": "TRAIN examples used for fitting; diagnostic only, not held-out evidence."}}.

## Every completed classifier-only epoch

| Recipe / epoch index | Validation accuracy | Macro F1 | Disgust recall | Disgust F1 |
|---|---:|---:|---:|---:|
| head_refinement_orientation512 / 126 | 0.5746 | 0.5168 | 0.2600 | 0.2626 |
| head_refinement_orientation512 / 127 | 0.5776 | 0.5238 | 0.3200 | 0.2936 |
| head_refinement_orientation512 / 128 | 0.5804 | 0.5221 | 0.2600 | 0.2600 |
| head_refinement_orientation512 / 129 | 0.5778 | 0.5178 | 0.2800 | 0.2545 |
| head_refinement_orientation512 / 130 | 0.5824 | 0.5280 | 0.3000 | 0.2913 |
| head_refinement_orientation512 / 131 | 0.5831 | 0.5302 | 0.3200 | 0.2963 |
| head_refinement_orientation512 / 132 | 0.5793 | 0.5240 | 0.2800 | 0.2828 |
| head_refinement_orientation512 / 133 | 0.5821 | 0.5288 | 0.3200 | 0.2991 |
| head_refinement_orientation512 / 134 | 0.5824 | 0.5382 | 0.3400 | 0.3542 |
| head_refinement_orientation512 / 135 | 0.5781 | 0.5275 | 0.3200 | 0.3107 |
| head_refinement_orientation512 / 136 | 0.5811 | 0.5323 | 0.3200 | 0.3168 |
| head_refinement_orientation512 / 137 | 0.5829 | 0.5310 | 0.3200 | 0.3077 |
| head_refinement_orientation512 / 138 | 0.5844 | 0.5333 | 0.3000 | 0.3191 |
| head_refinement_orientation512 / 139 | 0.5786 | 0.5259 | 0.2800 | 0.2979 |
| head_refinement_orientation512 / 140 | 0.5877 | 0.5380 | 0.3400 | 0.3366 |
| head_refinement_orientation512 / 141 | 0.5839 | 0.5336 | 0.3200 | 0.3232 |
| head_refinement_orientation512 / 142 | 0.5834 | 0.5317 | 0.3000 | 0.3226 |
| head_refinement_orientation512 / 143 | 0.5841 | 0.5397 | 0.3600 | 0.3636 |
| head_refinement_orientation512 / 144 | 0.5836 | 0.5330 | 0.3000 | 0.3261 |
| head_refinement_orientation512 / 145 | 0.5884 | 0.5385 | 0.3000 | 0.3297 |
| head_refinement_orientation512 / 146 | 0.5846 | 0.5387 | 0.3400 | 0.3617 |
| head_refinement_orientation512 / 147 | 0.5806 | 0.5278 | 0.2800 | 0.3043 |
| head_refinement_orientation512 / 148 | 0.5824 | 0.5295 | 0.2800 | 0.3043 |
| head_refinement_orientation512 / 149 | 0.5816 | 0.5311 | 0.3000 | 0.3297 |
| head_refinement_orientation512 / 150 | 0.5821 | 0.5288 | 0.2800 | 0.3077 |
| head_refinement_orientation512 / 151 | 0.5798 | 0.5361 | 0.3600 | 0.3711 |
| head_refinement_orientation512 / 152 | 0.5841 | 0.5301 | 0.2800 | 0.3011 |
| head_refinement_orientation512 / 153 | 0.5821 | 0.5281 | 0.2600 | 0.2989 |
| head_refinement_orientation256 / 126 | 0.5728 | 0.5126 | 0.2400 | 0.2637 |
| head_refinement_orientation256 / 127 | 0.5756 | 0.5207 | 0.2800 | 0.2947 |
| head_refinement_orientation256 / 128 | 0.5746 | 0.5223 | 0.3200 | 0.2963 |
| head_refinement_orientation256 / 129 | 0.5783 | 0.5286 | 0.3200 | 0.3333 |
| head_refinement_orientation256 / 130 | 0.5786 | 0.5264 | 0.3400 | 0.3148 |
| head_refinement_orientation256 / 131 | 0.5834 | 0.5339 | 0.3600 | 0.3303 |
| head_refinement_orientation256 / 132 | 0.5793 | 0.5283 | 0.3000 | 0.3261 |
| head_refinement_orientation256 / 133 | 0.5829 | 0.5327 | 0.3400 | 0.3269 |
| head_refinement_orientation256 / 134 | 0.5804 | 0.5302 | 0.3200 | 0.3137 |
| head_refinement_orientation256 / 135 | 0.5841 | 0.5335 | 0.3000 | 0.3261 |
| head_refinement_orientation256 / 136 | 0.5851 | 0.5365 | 0.3200 | 0.3404 |
| head_refinement_orientation256 / 137 | 0.5821 | 0.5347 | 0.3400 | 0.3400 |
| head_refinement_orientation256 / 138 | 0.5866 | 0.5351 | 0.3000 | 0.3191 |
| head_refinement_orientation256 / 139 | 0.5894 | 0.5408 | 0.3200 | 0.3478 |
| head_refinement_orientation256 / 140 | 0.5884 | 0.5380 | 0.3400 | 0.3333 |
| head_refinement_orientation256 / 141 | 0.5834 | 0.5342 | 0.3400 | 0.3333 |
| head_refinement_orientation256 / 142 | 0.5861 | 0.5348 | 0.3000 | 0.3191 |
| head_refinement_orientation256 / 143 | 0.5874 | 0.5347 | 0.2600 | 0.3059 |
| head_refinement_orientation256 / 144 | 0.5839 | 0.5317 | 0.3000 | 0.3093 |
| head_refinement_orientation256 / 145 | 0.5849 | 0.5361 | 0.3400 | 0.3400 |
| head_refinement_orientation256 / 146 | 0.5859 | 0.5344 | 0.3200 | 0.3137 |
| head_refinement_orientation256 / 147 | 0.5856 | 0.5349 | 0.3000 | 0.3158 |
| head_refinement_orientation256 / 148 | 0.5861 | 0.5380 | 0.3400 | 0.3366 |
| head_refinement_orientation256 / 149 | 0.5919 | 0.5429 | 0.3200 | 0.3441 |
| head_refinement_orientation256 / 150 | 0.5922 | 0.5436 | 0.3200 | 0.3333 |
| head_refinement_orientation256 / 151 | 0.5935 | 0.5403 | 0.2800 | 0.3111 |
| head_refinement_orientation256 / 152 | 0.5907 | 0.5420 | 0.3400 | 0.3505 |
| head_refinement_orientation256 / 153 | 0.5884 | 0.5380 | 0.3000 | 0.3261 |
| head_refinement_orientation256 / 154 | 0.5884 | 0.5397 | 0.3000 | 0.3409 |
| head_refinement_orientation256 / 155 | 0.5846 | 0.5326 | 0.2800 | 0.3111 |
| head_refinement_orientation256 / 156 | 0.5882 | 0.5373 | 0.2800 | 0.3146 |
| head_refinement_orientation256 / 157 | 0.5909 | 0.5411 | 0.3000 | 0.3371 |
| head_refinement_orientation256 / 158 | 0.5924 | 0.5398 | 0.3000 | 0.3158 |
| head_refinement_orientation256 / 159 | 0.5919 | 0.5444 | 0.3200 | 0.3516 |
| head_refinement_orientation256 / 160 | 0.5884 | 0.5384 | 0.3000 | 0.3333 |
| head_refinement_orientation256 / 161 | 0.5884 | 0.5383 | 0.3000 | 0.3297 |
| head_refinement_orientation256 / 162 | 0.5894 | 0.5398 | 0.3000 | 0.3371 |
| head_refinement_orientation256 / 163 | 0.5909 | 0.5395 | 0.3000 | 0.3191 |
| head_refinement_orientation256 / 164 | 0.5904 | 0.5387 | 0.3000 | 0.3158 |
| head_refinement_orientation256 / 165 | 0.5884 | 0.5372 | 0.3000 | 0.3191 |
| head_refinement_orientation256 / 166 | 0.5909 | 0.5401 | 0.3000 | 0.3191 |
| head_refinement_orientation256 / 167 | 0.5912 | 0.5398 | 0.3000 | 0.3226 |
| head_refinement_orientation256 / 168 | 0.5929 | 0.5431 | 0.3000 | 0.3333 |
| head_refinement_orientation256 / 169 | 0.5914 | 0.5418 | 0.3000 | 0.3297 |

## Boundaries

The repeatedly reused validation and historically inspected test do not provide fresh, identity-disjoint or statistical-significance confirmation. Architecture widths, pruning, normalization and data changes are project choices, not an isolated causal/equal-compute ablation or exact author reproduction. No80%accuracy, reliable webcam behavior, clinical, weather/raga-response or physiological-fusion improvement is claimed. EEG/HRV evaluation remains unchanged and negative for primary fusion gain. App/model/PPT hashes stayed unchanged until a positive deployment gate; the individual protection checks and changed native-classifier tensor names are retained in repair_summary.json. All earlier resources and failures remain preserved.

Primary architecture source: [Pingle and Ragha(2025)](https://doi.org/10.1007/s41870-025-02531-y). FER+ label-quality context is documented separately, not used to relabel this dataset: [Barsoum et al.(2016)](https://doi.org/10.48550/arXiv.1608.01041).


Descriptive paired-image bootstrap (4,000resamples; not identity-aware/confirmatory): accuracy-difference95%interval [0.0009144947416552355, 0.01783264746227709]; disgust-recall-difference95%interval [0.02531645569620253, 0.17721518987341772]. Intervals were not used to choose or retune the candidate. Full report: reports/pure_orientation_20261108/paired_uncertainty.json.


## Per-class recall tradeoffs on the same full test

All seven tracked aggregate/disgust metrics improved, but not every individual class improved. Happy, surprise and neutral recall declined. No alternate candidate is selected after observing these results.

| Class | Earlier pure recall | Active pure recall | Difference |
|---|---:|---:|---:|
| angry | 45.16% | 48.65% | +3.49 pp |
| disgust | 36.71% | 46.84% | +10.13 pp |
| fear | 36.77% | 41.11% | +4.34 pp |
| happy | 81.59% | 78.61% | -2.98 pp |
| sad | 40.15% | 46.16% | +6.01 pp |
| surprise | 73.84% | 69.54% | -4.30 pp |
| neutral | 58.54% | 57.36% | -1.18 pp |
