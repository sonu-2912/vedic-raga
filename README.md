# RagaMoodSync

Paper-informed facial-expression and Hindustani raga research app. Python3.11, PyTorch CPU, FastAPI, React/Vite and SQLite. Primary paper: https://doi.org/10.1007/s41870-025-02531-y . Exact author reproduction is not claimed.

## Current measured results

The active model is **pure FusDenseNet**, with no external encoder or inference ensemble.

| Full test, all6,561faces | Earlier pure | Active pure |
|---|---:|---:|
| Accuracy |57.42%|58.34%|
| Macro F1 |0.5320|0.5554|
| Disgust recall |36.71%|46.84%|
| Disgust precision |0.4833|0.5068|
| Disgust F1 |0.4173|0.4868|

Disgust correct29to37outof79; all seven declared aggregate/disgust metrics improved. The gain comes from training the native classifier on original/mirrored TRAIN views from the same pure backbone, with TRAIN-only normalization folded into its Linear layer. The backbone is unchanged. Six independent full-network coverage epochs failed;72cheaper classifier epochs across512/256widths produced one frozen testcandidate, with the512head selected. They are not equivalent compute. No teacher, external features, inference class offset or prediction filter was used.

The primary test remains historically inspected and validation repeatedly reused, not fresh identity-disjoint evidence. Accuracy is still modest; no80%, reliablewebcam, clinical, weather/raga-response or joint physiology benefit is claimed. Real EEG/HRV controlled results remain unchanged. Full figures, all seven metrics and rejected epochs: [PURE_ORIENTATION_REPAIR_RESULTS.md](PURE_ORIENTATION_REPAIR_RESULTS.md). Failure analysis: [PURE_REPAIR_ANALYSIS.md](PURE_REPAIR_ANALYSIS.md).

## Run on Windows

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

For an existing healthy server, use `./scripts/start-backend.ps1` to avoid starting a duplicate instance. Open http://127.0.0.1:8000/ after a frontend build. For frontend development, in another terminal:

```powershell
cd frontend
npm run dev
```

Build production UI: `node node_modules/vite/bin/vite.js build --configLoader native` from frontend. Research results remain separate from the listener interface.

## Evaluation and verification

```powershell
.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/pure_orientation_20261108 --device cpu --threads 4 --single-view --require-disgust-recall-gain --baseline-report reports/pure_orientation_20261108/baseline_test.json --baseline-manifest reports/pure_orientation_20261108/baseline_test_manifest.json
.\.venv\Scripts\python.exe -m scripts.report_orientation_repair
.\.venv\Scripts\python.exe -m pytest -q --basetemp reports/test_tmp_new_run
```

A completed evaluator verifies/reuses its frozen bundle. It refuses changed weights after test access. Use a fresh temporary test directory on each pytest invocation. Local experiment artifacts and manually acquired FER2013 are required for exact evaluation; the filtered GitHub app checkout excludes private datasets and environments.

See [DATASETS.md](DATASETS.md) for official access, format, licence and manual download procedures. Dataset absence does not prevent inference with the included trained model. Provision the separately excluded Raag Ratnakar resources using `python -m scripts.bootstrap_resources --source-dir PATH` or `--download` and preserve upstream notices. Catalog:72ragas,528 link entries,516uniqueURLs.

The newest editable 13-slide deck is output/RagaMoodSync_Research_Pure_Orientation_Repair_Reviewed_v3.pptx. Historical reports/decks and all original resources are preserved.

Latest presentation: `output/RagaMoodSync_Research_Pure_Orientation_Repair_Reviewed_v3.pptx` (13 editable slides). Latest verification: 85 tests passed; production frontend and live API checks passed.


To independently re-evaluate the active pure checkpoint on the same deduplicated test cohort, use a fresh output prefix (this does not train or select another model):

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_emotion --data data/private/fer2013-v1.zip --checkpoint models/fusdense.pt --split test --exclude-manifest reports/fer_training_manifest.json --threads 4 --output reports/current_pure_recheck
```

The full source dataset and fixed manifest are needed. The filtered GitHub checkout excludes private data; existing measured summaries remain available there.

Class-level tradeoffs: happy, surprise and neutral recall declined. All per-class recalls and differences are reported in PURE_ORIENTATION_REPAIR_RESULTS.md and the app research view. No claim that every expression improved.
