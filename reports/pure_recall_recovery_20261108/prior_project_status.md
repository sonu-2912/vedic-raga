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
