# Pure orientation repair audit — 8 November 2026

The active pure model achieved58.34% accuracy,0.5554macro F1 and46.84% disgust recall on all6,561same retained test faces, improving all seven declared metrics. Its native classifier changed; every backbone tensor remains bit-identical. No hybrid or external teacher/encoder.

85source tests passed in118.56seconds,15dependency warnings. Production frontend and live emotion/recommendation/health/controlled-physiology checks passed. All confusion/class metrics recomputed,55upstream originals unchanged. No user photo used. One frozen candidate tested, no alternate test search. Data and test limitations remain explicit.

Exactly13editable slides; all individually reviewed at full size. Package/layout checks have zero findings. Native PowerPoint opening was not verified. Final deck:output/RagaMoodSync_Research_Pure_Orientation_Repair_Reviewed_v3.pptx. Visual receipt:reports/slides_orientation_repair_v2/visual_review.json. Filtered export85tests passed in114.82seconds and production frontend built; GitHub main verified at 6d4a2cedec7ee763900485ed4407fbaf2f83688d. Earlier audit below is historical.

# Pure FusDenseNet audit — 8 November 2026

Pure experiments completed9full-network epochs across v4/v8recipes. Newpuredeployed:False. ActiveisPure:True. Active accuracy57.42%, macroF10.5320, disgustrecall36.71%. Formerhybrid59.26%is separately labelled reference-only. No new gain is claimed if no candidate qualified/passed.

78tests passed, 127.31seconds. Production frontend and liveAPI verification recorded in runlogs. Raw retained-prediction metrics/confusion recomputed, originals55unchanged. No userphoto used. Full pure-state/initialization/gradient contracts tested; nativev8initial logits equalv4. Budget-paused v4run actualcounts and native-stage limits are explicit. Earlier audit below is historical.

# Latest audit — 8 November 2026

Completed6full-network v5epochs plus58separate frozen-feature residual-head epochs. Selected recipe:reports/direct_dense_head_long_20261108; deployed:True. Active architecture v7; v6/v7are FusDenseNet+ImageNet DenseNet121 hybrid variants, not a pure FusDenseNet result. Full measured audit and comparisons: SPATIAL_GENERALIZATION_RESULTS.md.

Actual active test accuracy59.26%, macroF10.5530, disgust recall43.04%. Source SHAa4ed3e54d3b863e6381a8de3ec16428020e58c6d006e48cdfbf72a03ee88c249. One candidate maximum selected after both recipes complete; no alternative test search. v6ifactive usesImageNet pretraining, disclosedCPU112-pixel preprocessing and frozen encoders; only new residual head trained. All official provenance/rights notices preserved, noFER-pretrained teacher orchangedlabels.

Full source suite67passed,15dependencywarnings,128.23seconds. LiveAPI/resource/metric verification is generated in reports/further_verification_summary.json by the final step. Earlier audit below is historical.

# Latest audit — 8 November 2026

Training finished: 13 additional full-network and 20 classifier-only epochs across four recipes. Exactly one frozen validation-qualified candidate was tested on all 6,561 retained images. Candidate test accuracy 57.2931%, macro F1 0.533686, disgust recall 35.4430%; current active model 57.4150%, 0.532040 and 36.7089%, respectively. The seven-metric gate rejected deployment because accuracy, macro recall and disgust recall declined. No subsequent candidate was selected using these test results. Active model and original data remain unchanged.

Verification: 60 tests passed (15 pre-existing dependency deprecation warnings), 136.12 seconds. Production frontend builds. Recomputed prediction/confusion metrics, live health/emotion/recommendation and controlled physiology checks passed. All 55 original upstream files remain byte-identical. External OpenCV teacher was screened on validation only and rejected; it did not train FusDenseNet. Full evidence: BALANCED_RESULTS.md and reports/deep_supervision_20261107/experiment_summary.json. Physiology primary fusion gain remains unproven. Validation/test reuse prevents fresh confirmatory claims.

The following earlier audit is historical.

# Current audit — 7 November 2026

The completed full emotion-path experiment trained5 additional epochs; validation selected student index124. The evaluated candidate passed the recorded integration gate and is active. Source checkpoint SHA `8fc1713541e49bac8fc6ec4f10a5d5ba4ca9ff3860e227caeda3b7b67870a618`.

Validation:58.0353% on3,970 fixed faces. Full test:6,561 retained faces, all7classes and all wrong predictions. The combined Kaggle test was historically inspected and validation repeatedly reused; neither cohort is a fresh identity-disjoint webcam benchmark. No80% accuracy or reliable webcam recognition is claimed.

| Metric | Before | Current | Difference |
|---|---:|---:|---:|
| accuracy | 0.561347 | 0.574150 | +0.012803 |
| macro f1 | 0.510547 | 0.532040 | +0.021494 |
| disgust recall | 0.240506 | 0.367089 | +0.126582 |
| disgust f1 | 0.336283 | 0.417266 | +0.080983 |
| macro precision | 0.546594 | 0.539191 | -0.007404 |
| macro recall | 0.498146 | 0.532511 | +0.034365 |
| disgust precision | 0.558824 | 0.483333 | -0.075490 |

Disgust true positives:19→29 of79; false positives:15→31. Recall/F1 improved, while macro and disgust precision declined. These regressions are visible in the app and deck. Detailed protocol, every new epoch and tensor-change audit: CONSISTENCY_RESULTS.md and reports/consistency_20261107/.

Verification:47 tests passed,15 existing Matplotlib/Pyparsing deprecation warnings,129.30 seconds. Production frontend builds. Live health, facial inference and recommendations returned200; research API identifies the new SHA, full_network_consistency stage and candidate_is_active=true. Every facial confusion-matrix/class metric was recomputed from retained predictions. All55 original upstream repository files are byte-identical. No user photograph was used for verification or training.

Real GamER fixed-game/exactly30-second/participant-held-out EEG/ECG experiments remain negative for primary fusion F1 improvement. Raw-window/fold audit passed. DREAMER official access is restricted; DENS source-size issues remain unresolved. There is no joint face/physiology/raga/weather efficacy result, real-sensor claim or defensible recommendation-quality metric from insufficient local feedback.

Editable presentation: output/RagaMoodSync_Research_Consistency_Reviewed.pptx, exactly13slides. All13rendered slides were visually reviewed; structural/layout/native-chart/table checks passed. SHA `9fc075902f19c682ec5f820094d8c2da228f91bcdd28cc724d83bb8a59b4e0d7`. Historical interface capture is labeled as such. Group details remain editable placeholders. No native PowerPoint-open verification is claimed.

The filtered publication checkout passed47tests (138.56 seconds) and its production frontend build. Portable weights and inference settings are identical to the evaluated source model; report checksums match. Publication files passed the private-resource/credential-pattern check. Dataset files, private SQLite, user images, original upstream assets and dependency environments are excluded.

Verified publication: main `766791d46bedd19b6b00efa6c07e5b39b7d18723` at https://github.com/sonu-2912/RagaMoodSync . Updated source archive: output/github/RagaMoodSync_GitHub_Ready_Consistency.zip.
