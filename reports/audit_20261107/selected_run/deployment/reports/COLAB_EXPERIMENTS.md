# Actual CPU continuation emotion-model results

Selected epoch: 110. Fixed training: 23503; validation: 3970.
Unfiltered test: 6561 faces. Accuracy: 0.561347; macro F1: 0.510547.
Checkpoint SHA-256: 9a733a87fc5ec0db544b74504d60ca9ea3d3c94c3648d85e61e5397ca10d43ed. Deployment recommendation: True.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| angry | 0.4696 | 0.4347 | 0.4515 |
| disgust | 0.5588 | 0.2405 | 0.3363 |
| fear | 0.3948 | 0.3991 | 0.3970 |
| happy | 0.7779 | 0.8229 | 0.7998 |
| sad | 0.4661 | 0.3506 | 0.4002 |
| surprise | 0.6929 | 0.6368 | 0.6637 |
| neutral | 0.4660 | 0.6024 | 0.5255 |

All results are measured from the selected checkpoint. Calibration and flip selection use validation only. Test remains historically inspected, not fresh or identity-disjoint. This run does not evaluate EEG, HRV, weather, raga-response or therapeutic efficacy. No guaranteed accuracy target is asserted. See colab_comparison.json for the complete same-set comparison, including regressions.
