# RagaMoodSync overfitting study: actual results

Completed: 2026-11-09T01:19:30.315065+05:30.

**The requested model-performance improvement was not achieved.** None of the new candidates passed every validation requirement. No new full-test candidate was evaluated, and the deployed model, application and PowerPoint remain unchanged.

The study completed 299 classifier-only trial epochs and 10 full-network epochs. These are different workloads. The classifier count includes one timing-limited, one-epoch recipe, which is preserved and identified in the detailed report. The training set contains 23,503 source faces, including 330 disgust and 6,039 happy faces. Augmentation creates views, not new subjects.

## Best verified deployed pure model

On all **6,561 retained test faces**, accuracy remains **58.34%** and macro F1 remains **0.5554**. Disgust precision is 0.5068, recall is 46.84% (37/79), and F1 is 0.4868. These are previously measured results, not a new test evaluation from this study.

| Class | Test recall | Support |
|---|---:|---:|
| angry | 48.65% | 888 |
| disgust | 46.84% | 79 |
| fear | 41.11% | 922 |
| happy | 78.61% | 1711 |
| sad | 46.16% | 1198 |
| surprise | 69.54% | 581 |
| neutral | 57.36% | 1182 |

## Comparable fixed-checkpoint overfitting measurements

Each diagnostic uses all 23,503 original training faces and all 3,970 validation faces in evaluation mode. There is no augmentation, dropout, confidence filtering or label change. The fine-spatial candidate uses verified cached features because its entire backbone is bit-identical to the verified source. Training scores are in-sample results.

| Diagnostic model | Parameters | Training accuracy | Validation accuracy | Gap |
|---|---:|---:|---:|---:|
| Deployed model | 1,828,896 | 92.08% | 58.41% | 33.66 pp |
| Final compact residual model, rejected | 589,456 | 72.52% | 58.01% | 14.51 pp |
| Fine-spatial 384-unit diagnostic, rejected | 1,408,544 | 93.19% | 58.41% | 34.78 pp |

The aggressive model reduced the gap by reducing fitting ability, without improving validation quality. The milder classifier still overfits. Capacity reduction alone did not solve generalization. These trials jointly changed architecture and training settings, so they do not establish a single cause. Label noise and identity effects have not been independently measured.

## Actual validation quality

These rows are diagnostic checkpoints, not eligible replacements. All 3,970 validation faces, including 50 disgust examples, remain included.

| Model | Accuracy | Macro F1 | Disgust precision / recall / F1 | Happy recall | Surprise recall | Neutral recall |
|---|---:|---:|---|---:|---:|---:|
| Deployed model | 58.41% | 0.5397 | 0.3673 / 36.00% / 0.3636 | 78.59% | 67.43% | 53.00% |
| Final native residual model | 58.01% | 0.5279 | 0.3175 / 40.00% / 0.3540 | 81.36% | 73.71% | 57.32% |
| Fine-spatial diagnostic | 58.41% | 0.5387 | 0.3673 / 36.00% / 0.3636 | 79.64% | 69.71% | 51.05% |

## Changes and findings

- **Compact native classifier:** 4×4 spatial features were pooled to 2×2 regions, reducing input width from 3,276 to 1,260. Four head recipes preceded six full-network passes. They did not meet the required aggregate and class metrics. See [compact results](PURE_CAPACITY_RESULTS.md).
- **Native residual feature learning:** six residual dense refinements added 117,360 feature parameters. Initial predictions matched the source exactly on 64 real validation faces. After four full-network passes, 155 emotion-path tensors had changed, including all 42 refinement tensors. Parameter updates occurred, but the required validation improvement did not. See [native residual results](PURE_RESIDUAL_CAPACITY_RESULTS.md).
- **BatchNorm check:** three fixed pure-model candidates received TRAIN-only population-statistic updates. Learned parameters stayed unchanged. One candidate reached 52% validation disgust recall with only 15.76% precision. All were rejected. The modest measured statistic drift did not support a major BatchNorm defect. See [normalization results](PURE_BATCHNORM_RESULTS.md).
- **Checkpoint rollback:** four validation checks restored earlier native feature parameters, with or without restoring the classifier. The resulting models lost compatibility between their co-adapted components. Examples produced 78% disgust recall with 4.18% precision, or 80% precision with 16% recall. None qualified. These are validation class metrics, not high overall accuracy.
- **Milder capacity control:** the verified backbone and all 3,276 fine spatial features were preserved while the classifier was pruned from 512 to 384 units. TRAIN-only neuron ranking and omitted-activation mean compensation preceded 20 classifier epochs. The fixed training accuracy of 93.19% versus 58.41% validation shows that overfitting remains.
- **Timing correction:** long wall-clock pauses ended one head recipe early. Before later full-network training, the budget was changed to process CPU time divided by four threads, with a minimum number of completed epochs. Both source versions and the declaration are retained. No validation or test requirement changed.

## Verification and limits

**104 software tests passed**, with zero failures or errors and 15 dependency warnings, in 129.71 seconds. All 55 original repository files and all eight protected model/app/report/PPT files remain byte-identical. The unchanged PowerPoint contains exactly 13 slides. Live health and research endpoints match the deployed checkpoint.

Every class and error remains included in the metrics. There is no external encoder, prediction mixture, test-label fitting, post-test threshold or fabricated result. Training references come from the same verified pure classifier and are discarded for inference. No new test was opened because no validation candidate qualified.

Validation has been repeatedly reused and the historical test previously inspected. These results are not fresh subject-disjoint or webcam-domain confirmation. EEG, HRV, weather and raga-efficacy results are unchanged. No new sensor reading, synchronized physiological-fusion benefit, clinical benefit or 80% accuracy is claimed.

Run the software checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Protocols, executed source snapshots, every validation report, histories, checkpoints and explicit non-deployment decisions are preserved under the five experiment directories listed in [the final audit](reports/capacity_study_final_audit.json).
