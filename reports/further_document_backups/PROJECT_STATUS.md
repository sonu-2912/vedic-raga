<!-- further-experiment-active -->
Further FusDenseNet training and fixed-context real physiology evaluation are in progress. The deployed 52.55% checkpoint is retained until a new candidate is evaluated. See reports/FURTHER_EXPERIMENT_PLAN.md.
<!-- further-experiment-active -->

<!-- actual-real-data-update -->
## Latest measured update — 3 November 2026

Active classifier: **52.55% unfiltered test accuracy**, macro F1 **0.4492**, disgust recall **12.66%**. New experiments, phase counts, all-class metrics, actual real-recording GamER fusion and its regressions/confounds are documented in [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md). Older figures below describe earlier checkpoints and stages. No result here establishes reliable webcam emotion, facial-plus-physiological fusion or raga/weather benefit. Current automated verification: 37/37 tests passed.

<!-- actual-real-data-update -->
# RagaMoodSync project status

Measured update: 03 November 2026 12:53, Asia/Kolkata. Prior progress history is preserved under reports/baseline_pilot/PROJECT_STATUS_history.md.

- Completed: paper-informed FusDenseNet v2, 223,776 parameters; dense fusion, cross-attention, cross-dense links, spatial/global pooling and reconstruction. Original paper author weights and exact architecture recovery are not claimed.
- Actual revised training: 23,503 train / 3,970 validation images after exact-pixel deduplication; 17 completed epochs across two phases. Best validation macro F1 selected epoch 16 (0.398849). Continuation restarted AdamW explicitly. Epoch 5 includes an overnight wall-clock interruption.
- Actual final test: 6,561 retained FER2013 combined-test faces; accuracy 50.13%, macro precision 0.4841, recall 0.4115, F1 0.3960. Checksum-matched evidence is in reports/emotion_evaluation.json. Some test-folder examples appeared in the old pilot evaluation; no fresh-dataset or identity-disjoint claim.
- Completed: validation-only inference selection and temperature fit; final calibration metrics reported separately. Webcam-domain reliability remains unverified.
- Completed: customer-facing music interface, FusDenseNet default, photo mood confirmation, recommendations, raga details/links, SQLite feedback and history. Separate research page displays live architecture, measured metrics, epoch history and plots.
- Verification: 29/29 automated tests passed; 0 failures, 0 errors. Production frontend builds under normal workspace permissions using Vite native config loading.
- Preserved: all 55 original extracted files byte-identical: True; PDF copy identical: True. Catalog: 72 ragas, 528 link entries, 516 unique URLs. Original source lacks a license file; attribution and rights notes remain.
- Completed: old pilot same-set reevaluation and fixed HOG/linear-SVM comparator; synthetic engineering tests remain separate. This is a combined training/data/architecture comparison, not a controlled architectural ablation.
- Completed: offline EEG/HRV and clearly synthetic sample mode, IKS provenance/missing-field handling and a trainable FDCF module. No sensors or real synchronized raga-response study are claimed.
- Reports updated: EXPERIMENTS.md, MODEL_CARDS.md, paper mapping, dataset access and literature comparator.
- Completed: output/RagaMoodSync_Research_Updated.pptx, exactly 13 editable slides, actual metrics/plots and customer-interface screenshot. All 13 rendered slides visually reviewed. Structural, layout, font, chart-workbook and import validation passed with zero findings/warnings. Original output filename also contains the same validated bytes. Receipt: reports/slides_build/validation-fer-full.json. Native PowerPoint opening is not claimed.

Remaining research: controlled equal-budget/multi-seed ablations; stronger/subject-disjoint FER protocol and webcam study; real interaction-trained FDCF and valid recommendation metrics; expert-reviewed missing raga metadata; synchronized multimodal data, calibrated sensors and clinical evidence. Group details in the presentation remain editable placeholders.

Checkpoint SHA-256: `04e2a0e288b071a576422ba5adbf02d8fad604e4f22747064eab6aa7f7e7f53e`.

- Backend launcher healthy-service reuse verified: repeated startup returns the existing application URL and exits successfully, without starting a second server.

## Multimodal correction — 3 November 2026

The full facial classifier remains 50.13% accurate on the retained test set; disgust recognition is weak. Conservative validation-based abstention replaces unsupported mood suggestions. Weather, EEG/HRV recordings, reference safeguards, listening goals and visible source contributions now reach the listener's recommendation path. This is experimental rule-based context integration, **not a validated improvement in physiological emotion recognition, recommendation quality or health**. Actual synthetic ablations and limitations are documented in [MULTIMODAL_STATUS.md](MULTIMODAL_STATUS.md). Current tests: 29/29 passed.

## Updated presentation and interface verification

Current deck: `output/RagaMoodSync_Research_Multimodal.pptx` (also copied to `output/RagaMoodSync_Research.pptx`). Exactly 13 editable slides, all rendered slides visually reviewed; package/layout/import/native-chart checks passed with zero findings. Previous FER-only deck is preserved as `output/RagaMoodSync_Research_Updated.pptx`. Native PowerPoint execution is not claimed. Desktop context explanations and 390px mobile layout were checked with isolated test data; no production feedback was written.

## Colab GPU training handoff — 3 November 2026

- Created `output/RagaMoodSync_GPU_Training.ipynb` and `output/RagaMoodSync_Colab_Project.zip`, with project source, the existing v2 checkpoint, immutable pixel split manifests and same-test baseline evidence. Dataset images, photos, personal SQLite history, credentials and upstream Navrasa weights are excluded.
- New CUDA-aware trainer supports mixed precision, CPU-portable model exports, validation macro-F1 checkpoint selection, class/disgust metrics, time/plateau stopping and completed-epoch recovery with model/optimizer/scheduler/scaler/Torch-Python RNG states on Drive. Existing CPU scripts and deployed model remain unchanged.
- GPU-aware validation/calibration/test/export produces complete seven-class metrics and plots with exact baseline record verification. Runs are locked against further training after test evaluation. No confidence filtering changes reported accuracy.
- Windows importer verifies archive paths/checksums, model/evaluation/policy provenance and CPU inference; backs up replaced artifacts; rejects default deployment when accuracy/macro F1 fail to improve or disgust recall decreases.
- Local tests exercised generated fixtures through actual training, epoch recovery, calibration, test evaluation, ZIP export and backed-up installation in an isolated project. These are software tests, not FER2013 results. CUDA execution cannot be tested on this Intel-only machine and must be verified by the notebook's Colab GPU check.
- No new model accuracy, physiological-fusion result or updated PPT metric is asserted by this handoff. Actual post-training results are exported as `COLAB_EXPERIMENTS.md`, comparison JSON, confusion matrix and curves for the subsequent research/app/PPT update. Full instructions: `COLAB_README.md`.

## Final real-data deliverables

Installed improved checkpoint: 52.55% unfiltered test accuracy, macro F1 0.4492; macro precision declined and disgust remains weak. Real GamER paired EEG/ECG experiments, offset sensitivity and game-context controls completed. Final verification: 37/37 tests passed; live model/evidence APIs and source checksums independently audited; production frontend build passed. Current deck: `output/RagaMoodSync_Research_Real_Data_Final.pptx` (13 editable slides), also copied to `output/RagaMoodSync_Research.pptx`. All 13 rendered slides reviewed; structural/layout/import/native-chart checks passed with zero findings. Native PowerPoint execution is not claimed. No aligned facial-physiology or raga/weather outcome validation is claimed.
