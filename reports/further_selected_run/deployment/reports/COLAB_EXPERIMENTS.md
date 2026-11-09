# Actual CPU continuation emotion-model results

Selected epoch: 78. Fixed training: 23503; validation: 3970.
Unfiltered test: 6561 faces. Accuracy: 0.525834; macro F1: 0.465782.
Checkpoint SHA-256: 0906fe709300259b5e6c342eee79e9d6fab703e3dc3fa4a746b6422473aaf5df. Deployment recommendation: True.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| angry | 0.4578 | 0.4088 | 0.4319 |
| disgust | 0.2951 | 0.2278 | 0.2571 |
| fear | 0.3304 | 0.2050 | 0.2530 |
| happy | 0.7651 | 0.7826 | 0.7738 |
| sad | 0.3652 | 0.4725 | 0.4119 |
| surprise | 0.6331 | 0.6386 | 0.6358 |
| neutral | 0.4836 | 0.5110 | 0.4969 |

All results are measured from the selected checkpoint. Calibration and flip selection use validation only. Test remains historically inspected, not fresh or identity-disjoint. This run does not evaluate EEG, HRV, weather, raga-response or therapeutic efficacy. No guaranteed accuracy target is asserted. See colab_comparison.json for the complete same-set comparison, including regressions.
