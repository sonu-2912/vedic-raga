# RagaMoodSync

Paper-informed FusDenseNet/FDCF app: Python3.11, PyTorch CPU, FastAPI, React/Vite and SQLite. Research source: https://doi.org/10.1007/s41870-025-02531-y . Architecture dimensions are disclosed project choices; exact reproduction is not claimed.

## Current measured results — 7 November 2026

| Metric | Previous | Current |
|---|---:|---:|
| Full test accuracy |52.58%|56.13%|
| Macro precision |0.4758|0.5466|
| Macro recall |0.4637|0.4981|
| Macro F1 |0.4658|0.5105|
| Disgust recall |22.78%|24.05%|
| Disgust F1 |0.2571|0.3363|

Validation: **57.41% with validation-selected flip averaging**,56.80% without it, on3,970 faces. Full test uses all6,561 retained faces, all7 classes and every error. Fixed training23,503 faces. The test set was historically inspected, not fresh or identity-disjoint. Disgust recall and webcam reliability remain limited.

**80% validation accuracy was not achieved.** Training later exceeded80% while validation remained near56%, showing overfitting. Training and validation accuracy are distinct. Detailed measured results, class tables, actual logs and repairs: [AUDIT_RESULTS.md](AUDIT_RESULTS.md).

FusDenseNet v4 preserves three fusion-dense stages, cross-attention, cross-dense links and prior pooling. It adds1,920 spatial features from the first dense blocks, feeding3,276→512→7 classification;1,828,896 total parameters. New training used frozen backbone features, with training-only normalization folded into the MLP. Two new experiments completed31 and32 classifier-only epochs. Logged epoch indices include earlier/rejected history and are not full-network epoch counts.

Real fixed-game/fixed30-second GamER EEG/ECG evaluation with held-out participants shows **no primary fusion F1 gain**. DENS scoring remains incomplete due to source dimension mismatches. No simulated measurement is reported as real. Joint facial/physiological, weather, raga-response and interaction-trained FDCF quality remain unvalidated.

## Start

This local project is now under `C:/Users/bindu/OneDrive - iitr.ac.in/VEDIC RAGA`.

```powershell
cd 'C:\Users\bindu\OneDrive - iitr.ac.in\VEDIC RAGA'
.\scripts\start-backend.ps1
```

Open http://127.0.0.1:8000/ . Research results: http://127.0.0.1:8000/?view=research&tab=Research . API docs: http://127.0.0.1:8000/docs . The backend serves the built frontend and the launcher reuses a healthy existing service.

Development frontend, second terminal:

```powershell
cd 'C:\Users\bindu\OneDrive - iitr.ac.in\VEDIC RAGA\frontend'
npm run dev
```

## Evaluation and tests

```powershell
.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/audit_20261107/selected_run --device cpu --threads 2 --baseline-report reports/audit_20261107/baseline/emotion_evaluation.json --baseline-manifest reports/audit_20261107/baseline/emotion_evaluation_manifest.json
.\.venv\Scripts\python.exe -m pytest -q --basetemp reports/test_tmp_new
```

Frozen evaluation is checksum-verified/reused on repeat runs; the evaluated run is not continued against test. Current tests: **41 passed**,15 dependency deprecation warnings. Production build and live emotion/recommendation API checks passed. All55 original repository files remain byte-identical. Model SHA-256 `9a733a87fc5ec0db544b74504d60ca9ea3d3c94c3648d85e61e5397ca10d43ed`.

## Data and deliverables

DATASETS.md documents access, formats, licences and manual downloads for FER2013, DEAP, WESAD, PhysioNet, Saraga, IndRaga, CompMusic and other attempted benchmarks. FER2013 and GamER are local under data/private/. Sample signals are explicitly synthetic. Missing raga fields remain unknown; the source has72 ragas,528 link entries and516 unique URLs. The supplied Navrasa weights remain an optional separate baseline.

Final audited editable13-slide deck: `output/RagaMoodSync_Research_Audited_Reviewed.pptx`. The earlier listener screenshot is explicitly labeled historical; current plots and numeric results are measured. Native PowerPoint execution is not claimed.

GitHub-ready source is prepared separately under output/github/RagaMoodSync, excluding original restricted assets, datasets, dependency environments and private feedback. Its portable model has bit-identical tensors and matching inference settings, with both serialization hashes recorded. Published and remote-verified as the private repository https://github.com/sonu-2912/RagaMoodSync , branch main, commit `5bc6d4432cb9fbf2a24a34b716d7ae7765709429`. The separate GitHub checkout also passed41 tests and its frontend build. Local Git credentials authenticated sonu-2912; no credential was printed or committed.

See AUDIT_RESULTS.md, PAPER_ANALYSIS.md, PAPER_IMPLEMENTATION_MAPPING.md, MODEL_CARDS.md, EXPERIMENTS.md, IKS_METHODOLOGY.md, LITERATURE_REVIEW.md, PRIOR_ART_AND_IP.md and THIRD_PARTY_NOTICES.md. Older numeric sections in historical reports are superseded by the current audit.
