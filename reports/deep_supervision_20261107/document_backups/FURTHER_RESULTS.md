## Latest completed full-network experiment — 7 November 2026

Active model: **57.42% full-test accuracy**, macro precision **0.5392**, recall **0.5325**, F1 **0.5320**; disgust recall **36.71%**, F1 **0.4173**. Inference validation: **58.04%**. All 6,561 retained test faces and all seven classes remain included.

The new augmentation/distillation/averaged-weight experiment completed 5 full emotion-path epochs. Its candidate was deployed after passing the fixed integration gate. Results and regressions: [CONSISTENCY_RESULTS.md](CONSISTENCY_RESULTS.md). No 80% result is claimed. Validation is repeatedly reused and the combined FER test was historically inspected; this is not fresh webcam-domain evidence. Real fixed-game EEG/HRV fusion still shows no primary gain, and DREAMER official access is restricted. Earlier numerical sections below are historical.

## Current audited result — 7 November 2026

Validation **57.41%**, full unfiltered test **56.13%**, macro F1 **0.5105**. The target80% validation accuracy was not achieved. Active version4 keeps dense fusion/attention and adds an early dense spatial feature path. New test and class results,63 additional classifier-only training epochs, repairs and limitations are in [AUDIT_RESULTS.md](AUDIT_RESULTS.md). **41 tests passed.** Earlier numbered results below are historical.

# Further actual results: FusDenseNet and controlled physiology

Generated 2026-11-03T23:55:14. This document supersedes earlier active-model summaries. Original paper results, historical experiments and simulated signal fixtures remain separate.

## Facial model: unfiltered same-set evaluation

Overall accuracy rose by only **2 correctly classified faces out of 6,561** (0.0305 percentage points). This is essentially unchanged accuracy, not a substantial accuracy improvement. Disgust recall improved from 10/79 to 18/79, while disgust precision fell from 0.3333 to 0.2951. Its higher F1 does not establish reliable disgust detection.

All 6,561 retained test faces and all seven classes are included, including wrong predictions. This is a historically inspected combined Kaggle test folder, not a fresh, identity-disjoint or PrivateTest-only evaluation. Fixed training 23,503, validation 3,970; exact pixel duplicates excluded. Repeated validation selection can overfit, particularly its 50 disgust examples.

| Metric | Previous 52.55% checkpoint | New checkpoint | Difference |
|---|---:|---:|---:|
| accuracy | 0.525530 | 0.525834 | +0.000305 |
| macro f1 | 0.449166 | 0.465782 | +0.016616 |
| disgust recall | 0.126582 | 0.227848 | +0.101266 |
| disgust f1 | 0.183486 | 0.257143 | +0.073657 |
| macro precision | 0.471820 | 0.475751 | +0.003931 |
| macro recall | 0.448536 | 0.463744 | +0.015208 |

| Expression | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| angry | 0.4578 | 0.4088 | 0.4319 | 888 |
| disgust | 0.2951 | 0.2278 | 0.2571 | 79 |
| fear | 0.3304 | 0.2050 | 0.2530 | 922 |
| happy | 0.7651 | 0.7826 | 0.7738 | 1711 |
| sad | 0.3652 | 0.4725 | 0.4119 | 1198 |
| surprise | 0.6331 | 0.6386 | 0.6358 | 581 |
| neutral | 0.4836 | 0.5110 | 0.4969 | 1182 |

Deployed SHA-256 `0906fe709300259b5e6c342eee79e9d6fab703e3dc3fa4a746b6422473aaf5df`. Architecture version 3, 322,080 parameters. FusDenseNet retains all three fusion-dense stages, cross-attention, cross-dense links, dilation and global/spatial pooling. The disclosed project v3 adds 768 normalized 4×4 stem mean/max spatial features before the 8×8 bottleneck, concatenated with the original 588 features, followed by 1,356→128→7 classification. This adds 98,304 weights and is not an exact recovered author architecture or controlled equal-budget ablation. Initialization preserved v2 logits before training (see spatial_variant_initialization.json).

MLP refinement uses only fixed training features. Its feature normalization is folded into the existing first Linear layer. Training and validation phases share the same fixed pixel manifests. Selected index 78 is an index across phases, not that many full-network epochs. The recorded source-run history contains 19 full-network and 69 MLP epochs, including historical epochs and later rejected checkpoints. These counts are logs, not the number of updates in the selected weights. Other experiments are separate, not added to the selected model's training history. Their completed new work is:

| Experiment | Additional completed epochs | Best validation macro F1 (including start) | Scope |
|---|---:|---:|---|
| further_fusdense_run | 2 | 0.456915 | Full network |
| frozen_bn_fusdense_run | 2 | 0.456915 | Full network |
| spatial_head_run | 39 | 0.473989 | MLP only |
| spatial_flip_head_run | 10 | 0.473989 | MLP only |
| spatial_disgust_head_run | 10 | 0.473989 | MLP only |
| spatial_full_run | 5 | 0.473989 | Full network |

Partial interrupted epochs are excluded. Weighted sampling, frozen BN, flip features and stronger rare-class weighting are recorded even when rejected. Selection uses validation accuracy/macro F1/disgust F1 floors and .75 macro F1 + .25 disgust F1, before any new test access. Deployment requires improved unfiltered test accuracy, macro F1 and disgust F1, without lower disgust recall. This descriptive gate does not establish significance. Temperature fitting and flip-TTA selection use validation only, and do not exclude difficult faces. Strong disgust or webcam reliability remains unestablished. No approval was needed for authorized training.

## Real EEG/ECG fusion with game identity held constant

Official GamER CC BY 4.0 recordings, DOI https://doi.org/10.5281/zenodo.8191877. Separate fixed-game tasks, centered exactly 30-second labelled windows, leave-one-participant-out folds, fixed balanced logistic regression C=1 and fold-fitted StandardScaler. EEG theta/alpha/beta/delta/gamma relative powers, entropy and Hjorth features; real ECG-derived RR mean, RMSSD, SDNN, pNN50, median and IQR. EEG and ECG share the source interval. No simulated signal contributes to these results.

| Constant game | Windows / people | EEG macro F1 | Cardiac macro F1 | Fusion macro F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| QWOP | 13 / 12 | 0.4583 | 0.3158 | 0.4583 | 0.7692 |
| CBAAD | 13 / 13 | 0.6061 | 0.5752 | 0.6061 | 0.6923 |

**Neither primary task shows a fusion macro-F1 gain.** Fusion accuracy is below majority accuracy. Participant-bootstrap paired intervals include zero. Game identity and window duration are constant within each task, so neither cue distinguishes labels. This controls those two particular confounds, not every artifact, movement or annotation error. Primary offset subtraction is fixed before evaluation; alternate addition is a sensitivity analysis, never selected for a better score. Exact raw-window recomputation, source annotation IDs, fold isolation, constant controls and independently recomputed confusion matrices passed `reports/gamer_fixed_context_audit.json`.

The older pooled Angry/Sad result is retained as historical and confounded (game identity alone classified all labels). It is not evidence of a physiology-specific emotion improvement. Thirty-second RR statistics are exploratory, not conventional five-minute clinical HRV. Longer-window within-game cohorts are insufficient. No facial/EEG alignment, disgust physiology, real raga responses, weather benefit or therapeutic efficacy is demonstrated.

## Independent game-free dataset attempt

DENS https://doi.org/10.18112/openneuro.ds003751.v1.0.2 and https://github.com/OpenNeuroDatasets/ds003751 contain EEG/ECG and movie-related self-ratings. A protocol fixed the first 12 released participants, 60-second windows and crossed participant/movie held-out folds. Only 2 official headers matched their annex-referenced float32 signal dimensions. The other 10 were excluded before signal interpretation. This cohort cannot support the planned benchmark, so **no DENS accuracy or fusion result is claimed**. Headers were fully MD5 checked; no partial signal is misrepresented as full-FDT checksum verified. See dens_download_plan.json, dens_access_outcome.json and dens_protocol.json. The modular evaluator is implemented but remains unvalidated on an adequate DENS cohort.

Dataset metadata says CC0 while its README/LICENSE specifies ODbL/DbCL. Preserve the source licence, attribution and more protective conditions pending clarification. No full signals or third-party assets are bundled into the PPT. No replacement subjects were chosen after observing scores.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/further_selected_run --device cpu --threads 2 --baseline-report reports/further_baseline/emotion_evaluation.json --baseline-manifest reports/further_baseline/emotion_evaluation_manifest.json
.\.venv\Scripts\python.exe -m scripts.gamer_fixed_context
.\.venv\Scripts\python.exe -m scripts.verify_fixed_context
.\.venv\Scripts\python.exe -m pytest -q
```

The frozen FER export is hash-verified and reused on repeat evaluation, preventing training/tuning of that run after test access. Exact metrics, confusion matrices and complete saved predictions remain in JSON. See DATASETS.md for access and licence instructions. The installed model can still misclassify a user's face; raw predictions and all class probabilities remain visible.
