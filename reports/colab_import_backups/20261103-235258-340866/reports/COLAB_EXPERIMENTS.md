# Actual CPU continuation emotion-model results

Selected epoch: 39. Fixed training: 23503; validation: 3970.
Unfiltered test: 6561 faces. Accuracy: 0.525530; macro F1: 0.449166.
Checkpoint SHA-256: dbc4a9f9293fb4e1b28ae249958964cfe12e1b3b17a18f7e75bea77925b7a4ba. Deployment recommendation: True.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| angry | 0.4252 | 0.4032 | 0.4139 |
| disgust | 0.3333 | 0.1266 | 0.1835 |
| fear | 0.3139 | 0.1638 | 0.2153 |
| happy | 0.7586 | 0.7989 | 0.7783 |
| sad | 0.3619 | 0.4341 | 0.3947 |
| surprise | 0.6223 | 0.6523 | 0.6370 |
| neutral | 0.4875 | 0.5609 | 0.5216 |

All results are measured from the selected checkpoint. Calibration and flip selection use validation only. Test remains historically inspected, not fresh or identity-disjoint. This run does not evaluate EEG, HRV, weather, raga-response or therapeutic efficacy. No guaranteed accuracy target is asserted. See colab_comparison.json for the complete same-set comparison, including regressions.
