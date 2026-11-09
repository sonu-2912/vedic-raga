## Latest measured update: further training and controlled physiology

Active FusDenseNet v3: accuracy **52.58%**, macro F1 **0.4658**, disgust recall **22.78%**, disgust F1 **0.2571** on all 6,561 retained test faces. Previous baseline was 52.55%. This remains a limited historically inspected facial benchmark, not reliable webcam emotion recognition.

Overall accuracy changed by only two test faces. Disgust precision fell from 0.3333 to 0.2951, while recall and F1 improved. Version 3 adds 768 normalized early spatial stem features to the original 588 fusion features, followed by a 1,356-to-128-to-7 MLP. The selected improvement uses classifier-only refinement; further full-network trials were rejected on validation.

Game-constant, fixed-30s real EEG/ECG evaluation shows **no fusion F1 gain**. DENS participant/movie-held-out evaluation is incomplete because only 2/12 source headers match referenced signal dimensions. No DENS performance is invented. Full current results, regressions and commands: [FURTHER_RESULTS.md](FURTHER_RESULTS.md). Earlier numerical sections below are historical and may refer to previous checkpoints or confounded pooled tasks.

# Actual model improvement and real physiological fusion

Updated 03 November 2026 22:10, Asia/Kolkata.

## Unfiltered facial model results

All **6,561 retained FER2013 test images and all seven classes** were scored. No abstention, thresholding or removal of wrong predictions is used to calculate these results. The selected checkpoint is deployed: **True**. Active application accuracy is 52.55%; the candidate results below are kept separate if deployment failed.

| Metric | Previous checkpoint | Selected candidate | Difference |
|---|---:|---:|---:|
| accuracy | 50.13% | 52.55% | 2.42% |
| macro f1 | 0.3960 | 0.4492 | 0.0532 |
| disgust recall | 2.53% | 12.66% | 10.13% |
| disgust f1 | 0.0488 | 0.1835 | 0.1347 |
| macro precision | 0.4841 | 0.4718 | -0.0122 |
| macro recall | 0.4115 | 0.4485 | 0.0370 |

Candidate macro precision 0.4718; macro recall 0.4485. Selected checkpoint validation accuracy 53.58%, macro F1 0.4595. Validation and test are different measurements.

| Test class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| angry | 0.4252 | 0.4032 | 0.4139 | 888 |
| disgust | 0.3333 | 0.1266 | 0.1835 | 79 |
| fear | 0.3139 | 0.1638 | 0.2153 | 922 |
| happy | 0.7586 | 0.7989 | 0.7783 | 1711 |
| sad | 0.3619 | 0.4341 | 0.3947 | 1198 |
| surprise | 0.6223 | 0.6523 | 0.6370 | 581 |
| neutral | 0.4875 | 0.5609 | 0.5216 | 1182 |

### What was trained

FusDenseNet remains the existing paper-informed 223,776-parameter architecture, with three fusion-dense stages, cross-attention and spatial/global pooling. No unrelated network is relabelled as FusDenseNet. Windows **CPU**, not CUDA/Colab hardware, executed all new work.

Full-network continuation completed 3 additional epochs using AdamW LR 0.0003, class-weight power 0.85, label smoothing 0.02, reconstruction weight 0.05 and fixed data augmentation. It warm-started epoch-16 weights, with explicit optimizer restart. Head-only experiments completed 34 and 30 epochs. They train the existing MLP on frozen full-training-split features with training-fitted normalization. The affine normalization is folded into its existing first Linear layer; dimensions and parameter count do not change. No external pretrained emotion weights are substituted. Head epochs are much cheaper than full-network epochs and must not be presented as equal compute.

Selected experiment: **head_refinement**, source `C:\Users\bindu\OneDrive\Desktop\VEDIC RAGA\reports\head_refinement_continued_run\best.pt`, epoch index 39. Checkpoint selection maximizes validation macro F1 among candidates at least matching the initial validation accuracy. The selected candidate is tested once after selection. Deployment additionally requires higher test accuracy and macro F1 without lower disgust recall, as an engineering gate; it is not statistical evidence. Validation reuse across recipes can overfit that split. The combined FER test folder was historically inspected; it is not a newly acquired or identity-disjoint benchmark. Webcam generalization remains unmeasured. Better aggregate scores do not establish reliable disgust recognition.

Checkpoint SHA-256: `dbc4a9f9293fb4e1b28ae249958964cfe12e1b3b17a18f7e75bea77925b7a4ba`. Exact unchanged train/validation pixel manifests and same-test-record comparison are in `reports/selected_improvement_run`. Earlier checkpoints/reports are preserved by the importer under `reports/colab_import_backups` if deployed. The reusable scripts retain `colab` names for compatibility; no GPU execution is implied.

## Real physiological fusion: GamER

Source: [Papadopoulos et al., GamER v1](https://doi.org/10.5281/zenodo.8191877), CC BY 4.0, official Zenodo release. The 919,686,572-byte ZIP and both source CSV files were acquired from the publisher and verified against its MD5 checksums after checking access, licence, size and free disk space. Source EEG and ECG are sampled together in each HDF5 recording. No synthetic arrays or fabricated participant responses enter this evaluation.

Whole annotated intervals of at least 15 seconds are used; both signals must pass quality gates. 278 accepted intervals overall; 289 exclusions/invalid annotations are listed in `reports/gamer_features.json`. Each ablation uses exactly the same accepted cohort. Source labels are a psychologist's annotations based on self-reports and observation, not direct measurements of internal mood. Features: relative EEG theta/alpha/beta/delta/gamma, spectral entropy and Hjorth measures; ECG-derived mean/median RR, RMSSD, SDNN, pNN50 and RR IQR. Many intervals are ultra-short and do not support conventional clinical HRV interpretation. No frequency-domain HRV is claimed.

StandardScaler and fixed balanced LogisticRegression C=1 are fitted on training participants only. Leave-one-participant-out predictions are recorded for every retained interval. No participant occurs in both training and test within a fold. There is no hyperparameter search on outer-fold scores. An exploratory game-ID-only control was added before scoring to expose task/context confounding; it is not a physiological model.

### anger vs sadness

161 real labelled intervals, 39 participants; 154 intervals shorter than 60 seconds. Class counts: {"Angry": 93, "Sad": 68}.

| Method | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| EEG | 68.94% | 0.6812 | 0.6758 | 0.6775 |
| Cardiac | 63.35% | 0.6274 | 0.6294 | 0.6278 |
| Fusion | 74.53% | 0.7391 | 0.7400 | 0.7396 |
| Majority | 57.76% | 0.2888 | 0.5000 | 0.3661 |
| Game ID control | 100.00% | 1.0000 | 1.0000 | 1.0000 |

Fusion minus EEG macro F1: **+0.0621**. Approximate paired participant-bootstrap 95% interval: **[-0.0404, +0.1700]**. Bootstrap resamples participants in existing outer-fold predictions, not refitted studies; it is not multiplicity-adjusted significance evidence.

Offset-sign sensitivity (addition instead of subtraction), independently processed: 148 intervals / 39 participants. EEG F1 0.6679; cardiac F1 0.5876; fusion F1 0.6900; game-only F1 1.0000. The favorable convention is not selected after scoring. Different accepted boundary/quality cohorts mean the two signs are not a controlled paired comparison.

### five class

240 real labelled intervals, 40 participants; 229 intervals shorter than 60 seconds. Class counts: {"Angry": 93, "Sad": 68, "Bored": 30, "Stressed": 25, "Happy": 24}.

| Method | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| EEG | 30.00% | 0.2620 | 0.2605 | 0.2529 |
| Cardiac | 25.00% | 0.2493 | 0.2931 | 0.2428 |
| Fusion | 32.92% | 0.2926 | 0.3015 | 0.2861 |
| Majority | 38.75% | 0.0775 | 0.2000 | 0.1117 |
| Game ID control | 61.67% | 0.4855 | 0.5489 | 0.4910 |

Fusion minus EEG macro F1: **+0.0332**. Approximate paired participant-bootstrap 95% interval: **[-0.0513, +0.1168]**. Bootstrap resamples participants in existing outer-fold predictions, not refitted studies; it is not multiplicity-adjusted significance evidence.

Offset-sign sensitivity (addition instead of subtraction), independently processed: 227 intervals / 40 participants. EEG F1 0.2498; cardiac F1 0.2076; fusion F1 0.2523; game-only F1 0.4917. The favorable convention is not selected after scoring. Different accepted boundary/quality cohorts mean the two signs are not a controlled paired comparison.

### Interpretation of the measured results

The primary anger-versus-sadness fusion score is numerically higher than EEG alone, but the participant-bootstrap interval for the F1 difference includes zero. Game ID alone classifies the two labels perfectly in this accepted cohort, demonstrating complete task/label confounding. This experiment cannot establish emotion-specific physiology. In the five-class task, fusion accuracy is below the majority baseline even though macro F1 is higher. These weaknesses and the macro-precision regression in the facial model must not be hidden behind the favorable accuracy or F1 figures. No learned physiology classifier is deployed as a validated mood detector.

### Limits of this physiological evidence

GamER does not contain disgust labels, FER2013 faces, local weather or Hindustani raga responses. Game identity is correlated with source emotion labels; features may encode task, movement or recording artifacts. A game-only score can be high for this reason. A positive fusion difference on these tasks does not establish emotion-specific physiological generalization, a facial-fusion improvement or mood benefit. No classifier trained on these source labels silently replaces the listener's seven FER classes. EEG/HRV files can still add explicit context in the listener; weather uses disclosed experimental Ritu preferences. No clinical or weather-outcome validation has been performed.

Video annotation time minus VIDEO_OFFSET is an explicit alignment assumption. The release does not independently confirm that sign and the original videos are absent. Both sign conventions are published; alignment needs author confirmation before stronger claims. Single-lead EEG does not support ICA, and there is no independent manual artifact adjudication or clinical beat annotation reference. The demonstrated result is a reproducible, limited real-recording benchmark, not a validated physiological emotion product. DREAMER and Emognition access restrictions were respected; no gated mirrors were used.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m scripts.gamer_experiment
.\.venv\Scripts\python.exe -m scripts.gamer_experiment --sensitivity
.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/selected_improvement_run --device cpu --threads 2
.\.venv\Scripts\python.exe -m pytest -q tests --junitxml=reports/pytest.xml
```

The selected FER run is locked after evaluation. Repeating its evaluation command reuses checksum-verified recorded outputs rather than repeatedly reading test labels. Real physiology evaluation reads the verified private source ZIP and source CSVs; no credentials are required for the open GamER release. Generated reports retain all accepted interval predictions and participant-disjoint fold manifests for audit, locally.

Verification: 37 tests, 0 failures, 0 errors. Production frontend build is checked separately.

## Final real-data deliverables

Installed improved checkpoint: 52.55% unfiltered test accuracy, macro F1 0.4492; macro precision declined and disgust remains weak. Real GamER paired EEG/ECG experiments, offset sensitivity and game-context controls completed. Final verification: 37/37 tests passed; live model/evidence APIs and source checksums independently audited; production frontend build passed. Current deck: `output/RagaMoodSync_Research_Real_Data_Final.pptx` (13 editable slides), also copied to `output/RagaMoodSync_Research.pptx`. All 13 rendered slides reviewed; structural/layout/import/native-chart checks passed with zero findings. Native PowerPoint execution is not claimed. No aligned facial-physiology or raga/weather outcome validation is claimed.
