# Project audit and measured training results — 7 November 2026

Validation accuracy **57.41%** on 3,970 faces. Full unfiltered test accuracy **56.13%** on 6,561 faces; macro F1 **0.5105**. Previous test accuracy 52.58%. The requested80% validation result was not achieved and is not reported. A later training epoch reached83.98% training accuracy while validation remained near56%, demonstrating overfitting; training accuracy must not be represented as validation.

## Repairs and verification

The project was relocated to `C:/Users/bindu/OneDrive - iitr.ac.in/VEDIC RAGA`. Its Python3.11 environment still referenced the removed path; pyvenv.cfg was repaired with a preserved backup. The first audit test attempt hit an inaccessible old Windows temp folder; using a fresh isolated test directory resolved it. Final result: **41 passed**,15 existing dependency deprecation warnings. Pinned dependency consistency passed. Production React build passed. Optional unavailable Navrasa weights now give a clear error instead of a missing-file crash. Core FusDenseNet and the existing native seven-class labels remain.

## Actual training, architecture and evaluation

Two version4 candidate experiments completed:31 MLP-only epochs with128 hidden units, and32 with512 hidden units. Both used all23,503 fixed training faces and3,970 fixed validation faces. Backbone features were frozen. All original dense fusion, cross-attention, cross-dense concatenation and pooling paths remain. A new path retains1,920 mean/max spatial features from the first dense blocks before8×8 attention pooling. It concatenates them with the retained1,356 features. The selected model has **1,828,896 parameters**, **3276→512→7** classification. These are disclosed project choices, not an exact recovered author model or a controlled equal-budget causal ablation.

Zero-initializing the added classifier path preserved the starting model's predictions within floating-point tolerance before training. Normalization used training features only and was folded into the Linear layer. Variant selection used validation only, with no lower initial macro F1/disgust F1. Only the chosen wider candidate was evaluated on the complete test corpus. The test corpus was historically inspected and is neither fresh nor identity-disjoint. Exact pixel duplicates were excluded using the preserved manifest; all retained errors and all seven labels remain in the metrics.

| Expression | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| angry | 0.4696 | 0.4347 | 0.4515 | 888 |
| disgust | 0.5588 | 0.2405 | 0.3363 | 79 |
| fear | 0.3948 | 0.3991 | 0.3970 | 922 |
| happy | 0.7779 | 0.8229 | 0.7998 | 1711 |
| sad | 0.4661 | 0.3506 | 0.4002 | 1198 |
| surprise | 0.6929 | 0.6368 | 0.6637 | 581 |
| neutral | 0.4660 | 0.6024 | 0.5255 | 1182 |

Disgust recall remains weak despite improved F1. There is no claim of reliable webcam or clinical performance. Checkpoint SHA-256 `9a733a87fc5ec0db544b74504d60ca9ea3d3c94c3648d85e61e5397ca10d43ed`. All candidate logs, negative results, selection and the frozen evaluated bundle remain under reports/audit_20261107/. Continued training of the evaluated run is locked. Metadata and test predictions were not fabricated.

## Real physiology and remaining gaps

Earlier real GamER fixed-game, fixed30-second EEG/ECG tests with held-out participants found no primary fusion F1 gain. That negative result remains unchanged. Game and duration controls were audited. Other artifacts and annotation timing uncertainty remain. The proposed DENS participant/movie-held-out benchmark remains incomplete because only2/12 predeclared source headers matched referenced signal dimensions. No DENS score is invented. Face-plus-physiology, weather outcome benefit, real raga-response efficacy and interaction-trained FDCF ranking quality remain unvalidated. Explicit sample mode remains synthetic.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/audit_20261107/selected_run --device cpu --threads 2 --baseline-report reports/audit_20261107/baseline/emotion_evaluation.json --baseline-manifest reports/audit_20261107/baseline/emotion_evaluation_manifest.json
.\.venv\Scripts\python.exe -m pytest -q --basetemp reports/audit_20261107/test_tmp_repeat
```

Repeated evaluation verifies/reuses the frozen export; it does not retune against test. Dataset instructions are in DATASETS.md. GitHub publication succeeded using the existing local sonu-2912 Git credential. Private repository: https://github.com/sonu-2912/RagaMoodSync . Verified remote main commit `5bc6d4432cb9fbf2a24a34b716d7ae7765709429`. The Codex connector account differed, so the normal local Git authentication route was used without exposing credentials. A separate source export excludes original restricted resources, datasets and private feedback.
