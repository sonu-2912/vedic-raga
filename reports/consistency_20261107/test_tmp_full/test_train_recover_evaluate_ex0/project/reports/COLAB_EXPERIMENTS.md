# Actual CPU continuation emotion-model results

Selected epoch: 0. Fixed training: 59; validation: 11.
Unfiltered test: 14 faces. Accuracy: 0.142857; macro F1: 0.035714.
Checkpoint SHA-256: fc7e503b60ca16fc05aab5ab647d2f1ce1601fcb76d316ebaf7357f7104dea08. Deployment recommendation: False.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| angry | 0.0000 | 0.0000 | 0.0000 |
| disgust | 0.0000 | 0.0000 | 0.0000 |
| fear | 0.1429 | 1.0000 | 0.2500 |
| happy | 0.0000 | 0.0000 | 0.0000 |
| sad | 0.0000 | 0.0000 | 0.0000 |
| surprise | 0.0000 | 0.0000 | 0.0000 |
| neutral | 0.0000 | 0.0000 | 0.0000 |

All results are measured from the selected checkpoint. Calibration and flip selection use validation only. Test remains historically inspected, not fresh or identity-disjoint. This run does not evaluate EEG, HRV, weather, raga-response or therapeutic efficacy. No guaranteed accuracy target is asserted. See colab_comparison.json for the complete same-set comparison, including regressions.
