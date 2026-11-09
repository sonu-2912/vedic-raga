# Actual CPU continuation emotion-model results

Selected epoch: 143. Fixed training: 23503; validation: 3970.
Unfiltered test: 6561 faces. Accuracy: 0.583448; macro F1: 0.555367.
Checkpoint SHA-256: f0082aff0511c40d8dff89fc03bbea965b416db55a0f771701efaa38f490c548. Deployment recommendation: True.

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| angry | 0.4909 | 0.4865 | 0.4887 |
| disgust | 0.5068 | 0.4684 | 0.4868 |
| fear | 0.4639 | 0.4111 | 0.4359 |
| happy | 0.8166 | 0.7861 | 0.8011 |
| sad | 0.4559 | 0.4616 | 0.4587 |
| surprise | 0.6506 | 0.6954 | 0.6722 |
| neutral | 0.5176 | 0.5736 | 0.5441 |

All results are measured from the selected checkpoint. Calibration and flip selection use validation only. Test remains historically inspected, not fresh or identity-disjoint. This run does not evaluate EEG, HRV, weather, raga-response or therapeutic efficacy. No guaranteed accuracy target is asserted. See colab_comparison.json for the complete same-set comparison, including regressions.
