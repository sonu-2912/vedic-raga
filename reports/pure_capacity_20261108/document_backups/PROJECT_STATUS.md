# Completed pure FusDenseNet recall-recovery trials - 8 November 2026

Training finished: 54 corrected classifier recovery epochs, 3 full-network epochs, and 79 separate feature-mixup classifier epochs. An earlier 21-epoch weighting-error run was interrupted and retained. No new validation candidate met every happy/surprise/neutral recovery and disgust/aggregate retention condition. No new test was evaluated, and app/model/PPT hashes remain unchanged.

Active actual test: accuracy 58.34%, macro F1 0.5554, disgust recall 46.84% (37/79). This round did not improve the deployed model. The full-network final EMA validation reached 59.07% accuracy but macro F1 0.5383 and disgust recall 32%, so it was rejected.

Measured deployed-model overfitting: original TRAIN accuracy 92.08% versus validation 58.41%; TRAIN disgust recall 100% versus validation 36%. These gaps are measured, not a proven single-cause diagnosis. Full training changed 113 emotion-path tensors, including 12 attention tensors; changes did not establish a qualifying improvement. All validation faces/classes/errors are retained.

Latest software suite: 94 passed, 15 dependency warnings, 129.35 seconds. Separate report, actual curves and every failure: PURE_RECALL_RECOVERY_RESULTS.md and reports/pure_recall_recovery_v2_20261108/, reports/pure_recall_mixup_20261108/. EEG/HRV results remain unchanged.

Earlier completed records follow.

# Class recall recovery in progress — 8 November 2026

The current pure model/app/PPT remain unchanged at58.34% accuracy,0.5554macroF1,46.84% disgustrecall. New training must recover happy/surprise/neutral recall to at least earlier pure levels while preserving current disgust and aggregate metrics. A stricter gate is declared before any new test; at mostone frozen candidate after all training. App/PPT update only if all full-test goals pass.

Native classifier training corrects the prior cost/augmentation tradeoff: more cost on true happy/surprise/neutral TRAINlabels, lower mirror loss for those classes, and TRAIN-only same-pure-classifier retention protecting correctly labelled disgust. No external model, inference mixture or class threshold. A bounded full-network fine-tuning stage follows. Full/head epochs remain separately counted. Protocol, protected hashes and source snapshots:reports/pure_recall_recovery_v2_20261108/. The earlier unnormalized view-weight run was interrupted and preserved; no test was read. Corrected classifier recipes completed 54 epochs, with none meeting all validation gates. Full-network fine-tuning is running, protecting the TRAIN disgust-vs-rest boundary on correct positive and negative references. Seven new recovery contract tests pass (20.85 seconds). No app/model/PPT update has been made.

Earlier completed state follows:

# Project status — verified pure improvement, 8 November 2026

The improved pure FusDenseNet is active in the app. One frozen candidate passed the full same-set test gate, with no alternate test search.

| All 6,561 test faces | Earlier pure | Active pure |
|---|---:|---:|
| Accuracy |57.42%|58.34%|
| Macro precision |0.5392|0.5575|
| Macro recall |0.5325|0.5547|
| Macro F1 |0.5320|0.5554|
| Disgust precision |0.4833|0.5068|
| Disgust recall |36.71%|46.84%|
| Disgust F1 |0.4173|0.4868|

There are 61 more correct classifications, including 8 additional disgust cases (29→37/79). Disgust false positives increased31→36 while precision improved. All seven tracked metrics improved. Active source checkpoint SHA:f0082aff0511c40d8dff89fc03bbea965b416db55a0f771701efaa38f490c548. Validation:58.41% accuracy,0.5397 macro F1,36% disgust recall on3,970faces.

Failure audit: old sampler covered62.63% of distinct training images in a measured epoch; the0.579disgust loss multiplier weakened its oversampling. TRAIN disgust recall fell95.45%→35.45% under mirroring. Six full-network coverage/loss repair epochs did not qualify and were preserved. The predeclared conditional budget allocation then enabled native-classifier orientation repair. Its two independent512/256heads completed28+44=72classifier-only epochs, not72full-network epochs. The accepted512head learns raw/mirrored TRAIN features from the same original pure backbone. Every non-classifier tensor is bit-identical to the earlier pure model. No deeper feature-learning, hybrid, external encoder, teacher, inference ensemble, class threshold or post-test offset claim.

TRAIN-only normalization is folded into the native Linear layer. Training uses47,006augmented views from23,503source images, not new people. Exact mirrored TRAIN overlaps with held-out raw inputs were zero. No test labels were changed. Deployment required positive accuracy,macroF1,disgustF1 and at least5percentagepoints disgust recall gain. App/PPT hashes stayed unchanged until that real gate passed.

Verification:85source tests passed in118.56seconds;15dependency warnings. Frontend built. Live health, facial inference, recommendation and controlled-physiology API checks passed with the active SHA. Confusion/class metrics independently recomputed, all errors retained, all55original upstream files unchanged. No user photograph was used. Native PowerPoint opening is not verified.

Final editable13-slide deck:output/RagaMoodSync_Research_Pure_Orientation_Repair_Reviewed_v3.pptx. All13slides reviewed individually at full size; zero package/layout findings. Default alias:output/RagaMoodSync_Research.pptx. Historical decks and resources preserved. Full findings:PURE_REPAIR_ANALYSIS.md and PURE_ORIENTATION_REPAIR_RESULTS.md.

Accuracy remains modest. Validation was repeatedly reused and the test historically inspected, not fresh identity-disjoint confirmation. Descriptive paired-image bootstrap is not identity-aware or confirmatory. No80%accuracy, reliablewebcam/clinical performance or new physiology/weather/raga efficacy claim. Real EEG/HRV controlled fusion results remain unchanged and negative for primary gain. GitHub publication complete. Remote main verified at 6d4a2cedec7ee763900485ed4407fbaf2f83688d; working tree clean. Filtered export85tests passed in114.82seconds and production frontend built. Versioned archive:output/github/RagaMoodSync_GitHub_Ready_Orientation.zip. Receipt:reports/orientation_publication_receipt.json.

Class-level tradeoffs: happy, surprise and neutral recall declined. All per-class recalls and differences are reported in PURE_ORIENTATION_REPAIR_RESULTS.md and the app research view. No claim that every expression improved.
