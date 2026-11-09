# Pure training repair in progress — 8 November 2026

The previous pure-model failures were audited on training images only. Observed: old sampler covered 62.63% of distinct training faces in a measured epoch; its disgust CE importance was0.579, reducing CE class mass to2.45% despite4.22%sampling. Baseline training-disgust recall95.45% fell to35.45% under horizontalflip,70.30%under affine and92.42%under old photometric augmentation. RDrop gradients opposed the disgust CE direction in5of6audited batches; SupCon in3of6. These are diagnostics, not causal proof.

A new pure-v4 repair retains the complete original feature/classifier path, covers every natural training face each epoch, adds cycling disgust/angry/sad training exposures, uses unit supervised CE and gentler photometric-only augmentation, and reduces learning rates. Three new coverage/recovery/selection tests passed. Protocol and protected-artifact hashes are saved under reports/pure_repair_20261108/. No new evaluation result is claimed.

App model and PPT remain unchanged. They will be updated ONLY if one frozen pure candidate passes actual full-test gains, including at least5percentagepoints disgust recall gain. No hybrid or fabricated results. The current verified pure baseline remains57.42%testaccuracy,0.5320macroF1,36.71%disgustrecall.

Prior completed state follows:

# Project status — 8 November 2026

The bounded pure-network training experiments are finished: 5 regularized v4 epochs plus 4 native residual-dense v8 epochs. No new checkpoint met the combined validation requirements, particularly disgust recall. No new candidate was tested or deployed. This is a negative experiment, not a claimed improvement.

| Evidence | Accuracy | Macro F1 | Disgust recall |
|---|---:|---:|---:|
| Best new native validation objective, 3,970 faces | 58.36% | 0.5276 | 24.00% |
| Highest new native validation accuracy, 3,970 faces | 58.72% | 0.5252 | 22.00% |
| Active earlier pure baseline test, 6,561 faces | 57.42% | 0.5320 | 36.71% |

To honor the user's explicit pure-model requirement, the app now uses the earlier independently evaluated pure v4 checkpoint, SHA 8fc1713541e49bac8fc6ec4f10a5d5ba4ca9ff3860e227caeda3b7b67870a618. The former hybrid's 59.26% is retained only as a separately labelled historical reference. It supplied no weights, features or teacher predictions to these experiments.

Changes: native residual-dense refinements inside all six fusion branches, GroupNorm, class-paired training, training-only supervised contrastive projection and classifier R-Drop. Native v8 has 1,946,256 inference parameters; the 93,080-parameter training projection is removed. Its initial logits exactly matched pure v4, and feature-gradient contracts passed. Full protocols, failed epochs, source hashes, trained-weight changes and metrics: PURE_FUSDENSENET_RESULTS.md and reports/pure_fusdense_20261108/experiment_summary.json.

Validation and the historically inspected test are not fresh identity-disjoint evidence. Baseline training/validation gaps and only 330 training disgust images remain material limitations. No 80% accuracy, reliable webcam recognition, clinical, weather/raga-outcome or joint face/physiology efficacy is established. Real controlled EEG/HRV results remain unchanged and show no primary fusion F1 gain.

Verification: 78 source tests passed in 127.31 seconds, 15 dependency warnings. Production frontend built; live health, emotion, recommendation and real-physiology API checks passed. Every test error is retained and confusion/class metrics were independently recomputed. All 55 original upstream files remain byte-identical. No user photo was used for training/testing.

The editable presentation contains exactly 13 slides, with actual training curves, explicitly historical baseline test confusion, negative native validation diagnostics, real controlled physiology and complete references. All 13 rendered slides reviewed; structural/layout checks have zero findings. GitHub publication complete and remote main verified: c6096fb29d1ccb6c6d12f9cb4b97215bdd1dff74. Filtered export passed 78 tests in 126.09 seconds and production frontend build. Working tree clean. Archive: output/github/RagaMoodSync_GitHub_Ready_Pure.zip. Receipt: reports/pure_publication_receipt.json. Native PowerPoint opening is not verified. Final deck: output/RagaMoodSync_Research_Pure_FusDenseNet_Reviewed_v2.pptx. Earlier resources and decks are preserved.
