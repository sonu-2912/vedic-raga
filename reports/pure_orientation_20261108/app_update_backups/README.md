# RagaMoodSync

Paper-informed facial-expression and Hindustani raga research app. Python3.11, PyTorch CPU, FastAPI, React/Vite and SQLite. Primary paper: https://doi.org/10.1007/s41870-025-02531-y . Exact author reproduction is not claimed.

## Current measured results

Pure experiments completed9full-network epochs across v4/v8recipes. Newpuredeployed:False. ActiveisPure:True. Active accuracy57.42%, macroF10.5320, disgustrecall36.71%. Formerhybrid59.26%is separately labelled reference-only. No new gain is claimed if no candidate qualified/passed.

Validation accuracy 58.04% on3,970faces. Full test 57.42% on all6,561retained faces. Macro precision 0.5392, recall 0.5325, F1 0.5320. Disgust precision 0.4833, recall 36.71%, F1 0.4173. All7classes/allerrors retained. Test historically inspected, validation reused; no fresh/identity-disjoint or reliablewebcam claim.

Purev4retains the earlier architecture; nativev8adds117,360residualdense feature parameters INSIDEfusionblocks with GroupNorm, same classifier widths. The separate93,080parameter contrastive projection is TRAINonly and removed at export. No external encoder/teacher or inference ensemble. One frozen validation-selected candidate maximum; single-view evaluation. Every failed epoch, purebaseline comparison and formerhybrid comparison: [PURE_FUSDENSENET_RESULTS.md](PURE_FUSDENSENET_RESULTS.md).

Real fixed-game/30-second/participant-held-out EEG/HRV results remain unchanged, with no primary fusionF1gain. No joint face/body/raga/weather efficacy result. Missing metadata and dataset access/source limitations stay explicit.

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
.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/pure_selected_20261108 --device cpu --threads 4 --single-view --require-disgust-recall-gain --baseline-report reports/pure_selected_20261108/baseline_test.json --baseline-manifest reports/pure_selected_20261108/baseline_test_manifest.json
.\.venv\Scripts\python.exe -m scripts.report_pure_fusdense
.\.venv\Scripts\python.exe -m pytest -q --basetemp reports/test_tmp_new_run
```

A completed evaluator verifies/reuses its frozen bundle. It refuses changed weights after test access. Use a fresh temporary test directory on each pytest invocation. Local experiment artifacts and manually acquired FER2013 are required for exact evaluation; the filtered GitHub app checkout excludes private datasets and environments.

See [DATASETS.md](DATASETS.md) for official access, format, licence and manual download procedures. Dataset absence does not prevent inference with the included trained model. Provision the separately excluded Raag Ratnakar resources using `python -m scripts.bootstrap_resources --source-dir PATH` or `--download` and preserve upstream notices. Catalog:72ragas,528 link entries,516uniqueURLs.

The newest editable 13-slide deck is output/RagaMoodSync_Research_Pure_FusDenseNet_Reviewed_v2.pptx. Historical reports/decks and all original resources are preserved.

Latest presentation: `output/RagaMoodSync_Research_Pure_FusDenseNet_Reviewed_v2.pptx` (13 editable slides). Latest verification: 78 tests passed; production frontend and live API checks passed.

No new pure candidate qualified, so the selected-run evaluation command is inapplicable; the earlier pure model was restored without a new test. All negative results remain reported.

To independently re-evaluate the active pure checkpoint on the same deduplicated test cohort, use a fresh output prefix (this does not train or select another model):

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_emotion --data data/private/fer2013-v1.zip --checkpoint models/fusdense.pt --split test --exclude-manifest reports/fer_training_manifest.json --threads 4 --output reports/current_pure_recheck
```

The full source dataset and fixed manifest are needed. The filtered GitHub checkout excludes private data; existing measured summaries remain available there.
