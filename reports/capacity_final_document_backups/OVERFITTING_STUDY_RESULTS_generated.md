# RagaMoodSync overfitting study: actual final results

2026-11-09T01:19:30.315065+05:30

**Outcome: the requested performance improvement was not achieved.** No candidate met every validation requirement, no new full-test candidate was evaluated, and the app/model/PPT remain unchanged. Failed candidates were not installed.

The study completed **299classifier-only trial epochs** and **10full-network epochs**. They are different workloads, not309full-network epochs. The four compact-head recipes include a one-epoch timing-limited run, explicitly retained and reported. TRAIN has23,503source faces, including330disgust faces versus6,039happy faces; augmented views are not new subjects.

## Best verified deployed PURE model

All6561retained test faces: **accuracy58.34%; macroF10.5554**. Disgust precision0.5068, recall46.84%(37/79),F10.4868. These are unchanged previously measured results, not a new test result from this study.

| Class | Actual deployed test recall | Support |
|---|---:|---:|
| angry | 48.65% | 888 |
| disgust | 46.84% | 79 |
| fear | 41.11% | 922 |
| happy | 78.61% | 1711 |
| sad | 46.16% | 1198 |
| surprise | 69.54% | 581 |
| neutral | 57.36% | 1182 |

## Fixed-checkpoint overfitting measurements

All23,503original TRAIN images and3,970validation images; eval mode, no dropout, augmentation, confidence filtering or label changes. Fine-model cached features are valid because its entire backbone is bit-identical to the verified source. These are in-sample TRAIN scores, not new generalization evidence.

| Diagnostic model | Parameters | TRAIN accuracy | Validation accuracy | TRAIN minus validation |
|---|---:|---:|---:|---:|
| Verified deployed model |1,828,896|92.08%|58.41%|33.66%|
| Final compact residual model, rejected |589,456|72.52%|58.01%|14.51%|
| Fine-spatial384head diagnostic, rejected |1,408,544|93.19%|58.41%|34.78%|

The aggressive model reduced the gap by reducing fitting ability, without improving validation quality. The milder classifier still overfits. Neither observation establishes a successful fix. Parameter count alone is not the sole explanation; the current feature/training/data combination still lacks the required generalization. These joint trials are not causal ablations, and label-noise or identity effects are not established here.

## What actually changed and what failed

- Native spatial capacity control:3276to1260features and128/256heads;4independent head recipes followed by6full-network passes. Fine spatial information/capacity was reduced, but aggregate/class requirements were not met. Details:PURE_CAPACITY_RESULTS.md.
- Native residual-dense feature learning: six zero-output residual refinements, two dilated growth layers each, GroupNorm,117,360new feature parameters,4additional full-network passes. Real64face initial-function check had0logit error. 155emotion-path parameter tensors changed, including42new refinement tensors. This proves learning updates, not a performance improvement. Details:PURE_RESIDUAL_CAPACITY_RESULTS.md.
- TRAIN-only BatchNorm recalibration:3fixed candidates, parameters unchanged. It did not improve quality. One compact candidate had52%validation disgust recall but only15.76%precision, and was rejected. Modest mean-statistic drift did not support a major BN defect. Details:PURE_BATCHNORM_RESULTS.md.
- Native parameter rollback:4predeclared validation checks within one network. Restoring older feature parameters broke co-adapted representations; examples had78%disgust recall with4.18%precision, or80%precision with16%recall. None qualified. These are validation metrics, not high overall accuracy.
- Milder native classifier control:512to384units, all3276fine spatial/fusion features retained, class-balanced TRAIN neuron ranking and omitted-activation mean compensation;20classifier-only epochs. Starts from the verified active pure model, not failed full-network weights. Fixed TRAINaccuracy93.19%versus58.41%validation shows overfitting remains.
- Timing fix: very long wall-clock pauses prematurely cut one classifier recipe. The subsequent full phases used process CPU/4with minimum completed epochs and fixed maximum passes. Source before/after and declarations are preserved; no metric gate changed.

## Integrity and limitations

Every class and error remains included. No external encoder, probability mixture, test-label fitting, post-test threshold or fabricated metric was used. Same-current-pure TRAIN references were used only during training; they are discarded for inference. The test budget was never used because no validation candidate qualified. The fixed test was historically inspected and validation repeatedly reused; there is no fresh subject-disjoint or webcam-domain confirmation.

Final software suite: **104passed**,0failures,0errors,15dependency warnings; pytest reported129.71seconds (XML129.711). Original55repository files remain byte-identical. All8protected files and the13-slide PowerPoint are unchanged. Live health/research endpoints match the deployed checkpoint SHA.

EEG/HRV/weather/raga-efficacy results are unchanged. This study makes no new sensor, synchronized physiological-fusion, clinical-benefit or80%accuracy claim.

Re-run software checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Completed experiment directories preserve protocols, exact executed sources, every validation report, histories, checkpoints and explicit non-deployment decisions. Their completed/frozen data are not overwritten for further training.
