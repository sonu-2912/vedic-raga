<!-- actual-real-data-update -->
## Latest measured update — 3 November 2026

Active classifier: **52.55% unfiltered test accuracy**, macro F1 **0.4492**, disgust recall **12.66%**. New experiments, phase counts, all-class metrics, actual real-recording GamER fusion and its regressions/confounds are documented in [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md). Older figures below describe earlier checkpoints and stages. No result here establishes reliable webcam emotion, facial-plus-physiological fusion or raga/weather benefit. Current automated verification: 37/37 tests passed.

<!-- actual-real-data-update -->
# RagaMoodSync

GPU training handoff: open `output/RagaMoodSync_GPU_Training.ipynb` in Google Colab and upload `output/RagaMoodSync_Colab_Project.zip` plus the existing FER2013 image ZIP when prompted. See [COLAB_README.md](COLAB_README.md) for recovery, full evaluation, download and verified Windows import. Existing Windows CPU training code and deployed weights are preserved; no Colab training result is claimed before running the notebook.

Runnable Windows CPU research application based on [Pingle & Ragha (2025)](https://doi.org/10.1007/s41870-025-02531-y). Python 3.11, PyTorch CPU, FastAPI, React/Vite and SQLite. The original PDF and ZIP are preserved.

**Research status:** paper-informed mechanisms, not an exact reproduction. The default image model is the locally trained FusDenseNet version 2. The supplied nine-class EfficientNet-B2 Navrasa remains an optional, separately labeled baseline. See EXPERIMENTS.md and the Research tab for checksum-matched actual metrics, confusion matrix and training history. Recommendation quality needs real user interaction data. Synthetic results remain engineering checks only.

Actual revised FER2013 run: **17 completed epochs; selected epoch 16**, 23,503 training / 3,970 validation faces. All comparators below use the **same 6,561 retained test images**.

| Model | Test accuracy | Macro F1 |
|---|---:|---:|
| Archived two-epoch pilot | 26.52% | 0.1048 |
| Fixed HOG + linear SVM | 41.47% | 0.3470 |
| Revised FusDenseNet v2 | **50.13%** | **0.3960** |

This combined data/architecture/training comparison does not isolate individual architectural gains. Fear and disgust remain weak; webcam robustness and exact paper reproduction are unproven. Current artifacts: `output/RagaMoodSync_Research_Updated.pptx` (13 editable slides), EXPERIMENTS.md and MODEL_CARDS.md. The earlier pilot's 25.82% result used a different 1,375-image subset and is preserved separately.

**Current multimodal correction:** the facial model is still unsuitable for reliable seven-emotion recognition, especially disgust. Unsupported expression outputs are now withheld rather than labeled as a mood. The listener starts with your confirmed feeling (or “Not sure”) and listening goal, and accepts manual/city weather plus optional recorded EEG/HRV with reference safeguards. Open “More than a facial expression” on the home page. “What shaped this selection?” shows the inputs actually used. [MULTIMODAL_STATUS.md](MULTIMODAL_STATUS.md) separates functional integration from unmeasured recognition/health benefits.

## Run now (PowerShell)

From this directory:

```powershell
cd 'C:\Users\bindu\OneDrive\Desktop\VEDIC RAGA'
.\scripts\start-backend.ps1
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the built application. Interactive API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

For frontend development, keep the backend terminal running and use a second terminal:

```powershell
cd 'C:\Users\bindu\OneDrive\Desktop\VEDIC RAGA\frontend'
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite proxies `/api` to port 8000. Camera capture requires localhost or HTTPS and a browser camera permission. Upload/manual mood works when camera access is unavailable. Close the camera using Stop camera; tracks also stop after a capture.

Convenience scripts: `scripts/start-backend.ps1` and `scripts/start-frontend.ps1`. The backend launcher reuses an existing healthy RagaMoodSync service on port 8000, avoiding duplicate-start binding errors.

## Fresh installation

Install Python 3.11 x64 and Node 22, then:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm ci --no-audit --no-fund
npm run build
cd ..
```

This workspace also has a locally installed Python 3.11 under `.python/`, used to create `.venv`. It does not require CUDA, NVIDIA hardware or Intel GPU acceleration. Torch is explicitly CPU. The application never downloads model weights at startup. Keep `external/raag-ratnakar-main/` alongside backend code.

## Walk through the complete pipeline

The default interface is now designed for listeners: **For you**, **Explore ragas**, and **Your collection**. Choose a mood, select vocal/instrumental/both, and use **Find my ragas**. Raga cards provide listening performances, saved favourites and plain-language feedback. The optional photo check asks you to confirm or correct the suggested mood before recommendations.

Technical controls, model selection, EEG/HRV processing and research disclosures remain in the separate [research workspace](http://127.0.0.1:8000/?view=research&tab=Research), also linked inside About → About the collection. The detailed research workflow below applies to that workspace. The default listener screen does not display model names, confidence percentages, ranking scores, raw signal data or profile identifiers.

1. Use Discover to capture/upload a frontal facial image. Select FusDenseNet and click Predict emotion & discover ragas. Confidence is an uncalibrated model output; native label probabilities are in the API. Images are processed locally and never written to the database.
2. Correct mood manually if needed. Select vocal/instrumental/both, optional source-numbered Prahara and Ritu. Recommend.
3. Open music links. Link opens persist as listening history; completed playback is not measured.
4. Select Like/Dislike and mood change −2..+2, then Save feedback. Rankings immediately incorporate short/long explicit preference evidence. The profile identifier must contain letters, digits, `_` or `-`.
5. Signals lets you process explicitly synthetic samples or upload offline EEG/HRV JSON. Attached features participate in an optional, bounded and unvalidated arousal-context rule. No sensors are required or claimed.
6. History shows feedback, recommendations and listening link opens. Erasure removes the local profile's rows and disables any learned FDCF checkpoint.

Local profiles are not authenticated accounts. Bind to 127.0.0.1 for this single-computer prototype. Do not expose it publicly without authentication, authorization, consent and deployment hardening. SQLite is at `data/ragamood.sqlite3`; `RAGAMOOD_DB` can override it. Tests use isolated temporary databases.

## Tests and actual experiment commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests --junitxml=reports/pytest.xml
.\.venv\Scripts\python.exe -m scripts.smoke_experiments
.\.venv\Scripts\python.exe -m scripts.evaluate_recommendations
```

The smoke command generates artificial-texture classification metrics, confusion matrix/training curves, deterministic EEG/HRV results, sample rankings and synthetic feedback sensitivity. It **does not** train/save a deployable facial model. These outputs must not be presented as FER2013 or human-subject performance.

FER2013 version 1 is already downloaded locally after its advertised-license, size and disk checks. To reproduce revised full-partition CPU training (this replaces the deployable checkpoint):

```powershell
.\.venv\Scripts\python.exe -m scripts.train_emotion --data data/private/fer2013-v1.zip --epochs 12 --batch-size 64 --threads 8 --deduplicate --max-minutes 55
.\.venv\Scripts\python.exe -m scripts.calibrate_emotion --data data/private/fer2013-v1.zip
.\.venv\Scripts\python.exe -m scripts.evaluate_emotion --data data/private/fer2013-v1.zip --checkpoint models/fusdense.pt --split test --exclude-manifest reports/fer_training_manifest.json
```

To continue from saved real weights with validation plateau stopping:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_emotion --data data/private/fer2013-v1.zip --resume models/fusdense.pt --epochs 20 --batch-size 64 --threads 8 --deduplicate --learning-rate 0.001 --max-minutes 110 --patience 6 --min-delta 0.002
```

`--resume` warm-starts weights and explicitly restarts AdamW/cosine, preserving epoch history. It does not claim exact optimizer/RNG continuation. Best validation macro F1 is retained even if later epochs worsen. The continued project run began from the preserved phase-1 epoch-5 checkpoint. The overnight interruption appears in epoch-5 wall time; it is not active compute. Never rerun training against the test partition to improve reported scores.

The time limit is checked after an epoch; actual completed epochs are recorded. Best validation macro F1 selects the checkpoint. Calibration selects optional horizontal-flip averaging and fits temperature on validation only. Test evaluation uses the whole retained combined Kaggle test folder, excluding exact training/validation pixels and within-test duplicates. Some of this folder was evaluated in the old pilot; this is not a new benchmark or a subject-disjoint protocol.

To rerun the old checkpoint on precisely the revised test exclusions and regenerate current reports:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_emotion --data data/private/fer2013-v1.zip --checkpoint reports/baseline_pilot/fusdense.pt --split test --exclude-manifest reports/fer_training_manifest.json --output reports/baseline_re_evaluation
.\.venv\Scripts\python.exe -m scripts.verify_end_to_end
.\.venv\Scripts\python.exe -m scripts.update_research_results
```

The original two-epoch pilot checkpoint and reports are preserved in `reports/baseline_pilot/`. The revision changes architecture, data budget and optimizer together. Its comparison is not an isolated architecture ablation. Training curves, per-class precision/recall/F1, confusion matrix and calibration metrics are generated from real outputs. Original CSV `Usage` partitions and image-folder layouts are also supported. For a quick interface check, add `--limit 140 --val-limit 70 --epochs 1 --output reports/test_only.pt`; do not deploy tiny-run weights as a validated model.

Controlled ablations can use `--no-attention` / `--no-dilation`, separate checkpoint paths and equal fixed training budgets. Preserve each run's manifests/history before the next run. Such retraining ablations are pending; synthetic checks do not replace them.

After collecting at least four explicit like/dislike events with both classes:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_fdcf --epochs 80
```

This saves `models/fdcf.pt`. The running API reloads changed weights and blends learned scores for known profiles. Training loss is not a quality metric. The recommendation evaluation command performs rolling chronological testing with a four-event warmup for users with at least six events, and reports hit-rate/NDCG at 10 plus IKS ablation when held-out positive events exist. It evaluates the heuristic only to avoid contamination from a globally trained FDCF checkpoint. A separate time-split FDCF research protocol is still needed.

## API

| Endpoint | Purpose |
|---|---|
| GET `/api/health` | CPU/model/data health |
| GET `/api/research/results` | Checksum-matched real metrics, architecture, epoch history and baseline comparison |
| GET `/api/research/plots/{name}` | Measured training, confusion and comparison plots |
| POST `/api/emotion/predict` | Multipart image, model navrasa/fusdense, denoising, require_face |
| POST `/api/recommendations` | Emotion, local profile, metadata context, optional offline signal payloads |
| POST `/api/feedback` | Persist vote and mood change |
| POST `/api/listening` | Persist a catalog link open |
| GET/DELETE `/api/history/{user_id}` | Read/erase local profile history |
| GET `/api/ragas`, `/api/ragas/{id}` | Search/detail with provenance and explicit missing fields |
| GET `/api/signals/sample` | Deterministic synthetic signal requests |
| POST `/api/eeg/process`, `/api/hrv/process` | Offline feature extraction |

## Project documents and outputs

- `PAPER_ANALYSIS.md`: full-paper review and inconsistent reported values.
- `PAPER_IMPLEMENTATION_MAPPING.md`: equations, code and explicit assumptions.
- `LITERATURE_REVIEW.md`, `PRIOR_ART_AND_IP.md`: traceable research and preliminary patent search.
- `IKS_METHODOLOGY.md`, `DATASETS.md`: provenance, access, formats, rights and limitations.
- `EXPERIMENTS.md`: actual results versus paper/synthetic/pending experiments.
- `PROJECT_STATUS.md`: verified completion and remaining work.
- `THIRD_PARTY_NOTICES.md`: attribution and unresolved supplied resource license.
- `output/RagaMoodSync_Research.pptx`: exactly 13 editable slides, with relevant-slide citations and full references.

The 72-raga catalog has **528 link entries, 516 unique URLs**, not 537 songs. The supplied repository contains no license file. Resolve rights before public redistribution. No clinical efficacy, exact reproduction or real sensor readings are claimed. Measured FER results apply only to the documented dataset protocol; webcam robustness remains unverified.
