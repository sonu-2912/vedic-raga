# Actual CPU continuation emotion-model results

Selected epoch: 155. Fixed training: 23503; validation: 3970.
Unfiltered test: 6561 faces. Accuracy: 0.592593; macro F1: 0.553048.
Checkpoint SHA-256: a4ed3e54d3b863e6381a8de3ec16428020e58c6d006e48cdfbf72a03ee88c249. Deployment recommendation: True.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| angry | 0.5198 | 0.4741 | 0.4959 |
| disgust | 0.5484 | 0.4304 | 0.4823 |
| fear | 0.4570 | 0.3286 | 0.3823 |
| happy | 0.7886 | 0.8328 | 0.8101 |
| sad | 0.4934 | 0.4349 | 0.4623 |
| surprise | 0.6165 | 0.7194 | 0.6640 |
| neutral | 0.5158 | 0.6481 | 0.5744 |

All results are measured from the selected checkpoint. Calibration and flip selection use validation only. Test remains historically inspected, not fresh or identity-disjoint. This run does not evaluate EEG, HRV, weather, raga-response or therapeutic efficacy. No guaranteed accuracy target is asserted. See colab_comparison.json for the complete same-set comparison, including regressions.
