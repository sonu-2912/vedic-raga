# Actual model and physiology experiments in progress

Started 3 November 2026, Asia/Kolkata. This file records decisions before the new test evaluations.

## FusDenseNet continuation

The existing 223,776-parameter FusDenseNet v2 is warm-started from its epoch-16 weights. The original fixed pixel-disjoint training/validation manifest and FER2013 archive are verified. No new test images are read during training. Training uses AdamW (LR 0.0003, weight decay 0.0001), cosine schedule, class-frequency weight power 0.85, label smoothing 0.02 and reconstruction MSE weight 0.05. The previous optimizer is restarted explicitly. Eight additional epochs are requested, with validation patience 4 and a 100-minute epoch-boundary budget. Execution is on Windows CPU, despite the reusable trainer's `colab_train` filename. No GPU execution is claimed.

All epoch states are preserved. Selection uses validation macro F1; a candidate that also improves validation accuracy is preferred if one exists. Test evaluation happens after training and selection. All seven classes and all retained test records are scored, including wrong predictions. The historically inspected test corpus is not a fresh independent or identity-disjoint benchmark. Deployment requires higher full-test accuracy and macro F1 without lower disgust recall; failure retains the old model and is reported.

## Real physiological fusion

Official GamER v1: https://doi.org/10.5281/zenodo.8191877. Zenodo API metadata specifies CC-BY-4.0 and 919,686,572 bytes for data.zip; the original release checksum is verified after download. Over 600 GB free disk space was checked. No gated dataset agreement is signed, and no access restriction is bypassed.

The frozen machine-readable protocol is `reports/gamer_protocol.json`. Real single-lead EEG and ECG come from the same HDF5 recording and time interval. No face, weather or raga data are artificially paired. Primary task: source Angry versus Sad labels. Secondary task: the five source labels Angry, Sad, Bored, Stressed and Happy. Whole annotation intervals of at least 15 seconds are considered, with label-independent recording/beat quality gates. Short interval RR statistics are exploratory ultra-short features, not conventional validated HRV measurements.

EEG-only, cardiac-only, concatenated feature fusion and majority baseline use identical accepted intervals. Logistic regression C=1 and class weighting are fixed. Standardization is fitted on training subjects only. Leave-one-participant-out evaluation prevents participant overlap. Approximate paired participant-bootstrap uncertainty is reported; it does not constitute a clinical trial. Both subtraction and addition of the supplied VIDEO_OFFSET are evaluated because its sign cannot be independently confirmed from the omitted videos. No favorable sign is silently chosen after observing scores.

This benchmark cannot validate disgust recognition, seven FER expression classes, facial-plus-physiological fusion, weather-based emotion inference or Hindustani raga response. DREAMER and Emognition were reviewed but require access requests; no unauthorized mirrors are used.
