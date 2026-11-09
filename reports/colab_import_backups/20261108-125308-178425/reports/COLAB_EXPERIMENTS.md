# Actual CPU continuation emotion-model results

Selected epoch: 124. Fixed training: 23503; validation: 3970.
Unfiltered test: 6561 faces. Accuracy: 0.574150; macro F1: 0.532040.
Checkpoint SHA-256: 8fc1713541e49bac8fc6ec4f10a5d5ba4ca9ff3860e227caeda3b7b67870a618. Deployment recommendation: True.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| angry | 0.5025 | 0.4516 | 0.4757 |
| disgust | 0.4833 | 0.3671 | 0.4173 |
| fear | 0.4472 | 0.3677 | 0.4036 |
| happy | 0.7747 | 0.8159 | 0.7948 |
| sad | 0.4674 | 0.4015 | 0.4320 |
| surprise | 0.6076 | 0.7384 | 0.6667 |
| neutral | 0.4915 | 0.5854 | 0.5344 |

All results are measured from the selected checkpoint. Calibration and flip selection use validation only. Test remains historically inspected, not fresh or identity-disjoint. This run does not evaluate EEG, HRV, weather, raga-response or therapeutic efficacy. No guaranteed accuracy target is asserted. See colab_comparison.json for the complete same-set comparison, including regressions.
