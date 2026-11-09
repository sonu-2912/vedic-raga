# RagaMoodSync

Windows CPU application and research implementation based on [Pingle & Ragha (2025)](https://doi.org/10.1007/s41870-025-02531-y). Paper-informed mechanisms with disclosed assumptions; exact reproduction is not claimed.

## Current measured results

The improved FusDenseNet checkpoint is installed. These are **unfiltered results on all 6,561 retained test faces**, including all seven classes and every wrong prediction.

| Metric | Previous model | Current model |
|---|---:|---:|
| Accuracy | 50.13% | 52.55% |
| Macro precision | 0.4841 | 0.4718 |
| Macro recall | 0.4115 | 0.4485 |
| Macro F1 | 0.3960 | 0.4492 |
| Disgust recall | 2.53% | 12.66% |
| Disgust F1 | 0.0488 | 0.1835 |

Accuracy, recall and F1 improved; macro precision fell. Disgust remains poor. Raw class probabilities are available in the photo check. No claim of reliable webcam emotion or clinical validity is made. The historically inspected test corpus is not fresh or identity-disjoint.

Training included three additional full-network epochs and two classifier-head experiments (34 and 30 completed epochs). Head epochs use frozen features and are cheaper than full-network epochs. Training-fitted normalization is folded into the existing MLP; the paper-informed 223,776-parameter FusDenseNet architecture remains. Selection uses validation only. The deployable checkpoint is SHA-256 `dbc4a9f9293fb4e1b28ae249958964cfe12e1b3b17a18f7e75bea77925b7a4ba`.

Real physiological evaluation uses official **GamER EEG/ECG recordings**, not simulation. Participant-held-out anger-versus-sadness accuracy: EEG 68.94%, cardiac 63.35%, fusion 74.53%. However, game ID alone scores 100%, revealing task/label confounding; the fusion-gain interval includes zero. Five-class fusion accuracy is 32.92%, below majority accuracy 38.75%. These results do not establish a validated emotion-specific physiology model. Short RR windows and uncertain annotation-offset sign remain limitations; both offset signs are reported.

Full class metrics, training details, real-data ablations, regressions, confidence intervals and reproduction commands: [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md). **37 tests passed**; the production frontend built successfully; live model and evidence APIs were verified.

## Start the application

```powershell
cd 'C:\Users\bindu\OneDrive\Desktop\VEDIC RAGA'
.\scripts\start-backend.ps1
```

Open http://127.0.0.1:8000/ for the built listener application. The launcher reuses an existing healthy service instead of starting a duplicate on port 8000. API documentation: http://127.0.0.1:8000/docs. Actual metrics: http://127.0.0.1:8000/?view=research&tab=Research.

For frontend development, keep the backend running and use another terminal:

```powershell
cd 'C:\Users\bindu\OneDrive\Desktop\VEDIC RAGA\frontend'
npm run dev
```

Open http://127.0.0.1:5173/. Vite proxies API requests to port 8000. Webcam capture needs browser permission; image upload and manual feeling selection work without a camera. No physical EEG/HRV sensor is required. Manual weather works offline; optional city lookup uses Open-Meteo. Uploaded recordings and explicit references can provide experimental listening context. Simulated mode is labelled and requires opt-in; it is separate from the real-recording benchmark.

## Evaluate and audit

```powershell
.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/selected_improvement_run --device cpu --threads 2
.\.venv\Scripts\python.exe -m scripts.gamer_experiment
.\.venv\Scripts\python.exe -m scripts.gamer_experiment --sensitivity
.\.venv\Scripts\python.exe -m scripts.verify_real_results
.\.venv\Scripts\python.exe -m pytest -q tests --junitxml=reports/pytest.xml
```

The selected FER run is locked after evaluation; its command reuses checksum-verified outputs. `colab` filenames identify reusable scripts; this experiment ran on Windows CPU. The audit checks official dataset checksums, participant separation, confusion matrices, macro F1 and live APIs. It uses the existing public demo fixture for endpoint plumbing, with no correctness claim about that single image.

## Installation and production build

Python 3.11 x64 and Node 22:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm ci --no-audit --no-fund
node node_modules/vite/bin/vite.js build --configLoader native
cd ..
```

Dependencies target CPU PyTorch; no NVIDIA/CUDA hardware is needed. For new training, use a new isolated run directory with `scripts.colab_train` or `scripts.refine_emotion_head`, select with validation and evaluate after selection. Do not overwrite the deployed model or continue tuning against its test results. Existing older Colab project/notebook snapshots are preserved; the latest verified model export is `reports/selected_improvement_run/RagaMoodSync_Trained_Model.zip`.

## Datasets, reports and presentation

FER2013 is already present in `data/private/fer2013-v1.zip`. GamER originals are in `data/private/gamer` with publisher checksums verified. Download/access, formats and licences are documented in [DATASETS.md](DATASETS.md). Restricted datasets require their custodians' manual access procedure; no unauthorized mirrors are used.

Current editable PowerPoint: **`output/RagaMoodSync_Research_Real_Data_Final.pptx`**, exactly 13 slides; also copied to `output/RagaMoodSync_Research.pptx`. All rendered slides were reviewed and structural/layout/import/chart validation passed. Native PowerPoint execution is not claimed. Previous decks remain preserved.

Research documentation: PAPER_ANALYSIS.md, PAPER_IMPLEMENTATION_MAPPING.md, LITERATURE_REVIEW.md, PRIOR_ART_AND_IP.md, IKS_METHODOLOGY.md, DATASETS.md, EXPERIMENTS.md, MODEL_CARDS.md, REAL_DATA_RESULTS.md and PROJECT_STATUS.md. Earlier README content is preserved in `reports/README_before_real_data_update.md`.

The original Raag Ratnakar resources remain preserved: **72 ragas, 528 link entries, 516 unique URLs**. Missing raga metadata is marked unknown. The supplied nine-Rasa model is a separate baseline, not FusDenseNet. Source attribution and unresolved redistribution rights are documented in THIRD_PARTY_NOTICES.md. Clinical efficacy, weather benefit, aligned face/physiology/raga response and strong recommendation-quality results remain unestablished.
