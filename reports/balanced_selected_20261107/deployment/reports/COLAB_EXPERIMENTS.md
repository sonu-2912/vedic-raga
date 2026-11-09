# Actual CPU continuation emotion-model results

Selected epoch: 133. Fixed training: 23503; validation: 3970.
Unfiltered test: 6561 faces. Accuracy: 0.572931; macro F1: 0.533686.
Checkpoint SHA-256: 286d783f9f8692148bbcd765e20a817f7b82c421f0d6bd14c22b00bbcb9acb80. Deployment recommendation: False.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| angry | 0.4760 | 0.4910 | 0.4834 |
| disgust | 0.5385 | 0.3544 | 0.4275 |
| fear | 0.4365 | 0.3839 | 0.4085 |
| happy | 0.7809 | 0.8083 | 0.7944 |
| sad | 0.4701 | 0.3806 | 0.4207 |
| surprise | 0.6169 | 0.7177 | 0.6635 |
| neutral | 0.5018 | 0.5795 | 0.5379 |

All results are measured from the selected checkpoint. Calibration and flip selection use validation only. Test remains historically inspected, not fresh or identity-disjoint. This run does not evaluate EEG, HRV, weather, raga-response or therapeutic efficacy. No guaranteed accuracy target is asserted. See colab_comparison.json for the complete same-set comparison, including regressions.
