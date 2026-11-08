"""Integrate only completed measured evidence; preserve earlier reports and resources."""
import json, shutil
from pathlib import Path
import torch
from scripts.colab_common import digest, model_from, write_json
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/consistency_20261107'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    trial=read(RUN/'experiment_summary.json');active=read(ROOT/'reports/emotion_evaluation.json');val=read(ROOT/'reports/validation_evaluation.json')
    ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True)
    sha=digest(ROOT/'models/fusdense.pt');assert sha==active['checkpoint_sha256']==val['checkpoint_sha256']
    m=active['classification_report']['macro avg'];d=active['classification_report']['disgust'];history=read(ROOT/'reports/training_history.json')
    if trial['deployed']:
        write_json(ROOT/'reports/emotion_training_comparison.json',read(ROOT/'reports/latest_emotion_experiment.json')['test_comparison'])
    full=sum(not r.get('phase') or r.get('phase')=='full_network_consistency' for r in history)
    head=sum(str(r.get('phase','')).startswith('head_refinement') for r in history)
    summary={'accuracy':active['accuracy'],'macro_precision':m['precision'],'macro_recall':m['recall'],'macro_f1':m['f1-score'],'disgust':d,'test_n':active['n'],'checkpoint_sha256':sha,'parameters':sum(p.numel() for p in model_from(ck,torch.device('cpu')).parameters()),'classifier_inputs':next(layer.in_features for layer in model_from(ck,torch.device('cpu')).classifier.modules() if isinstance(layer,torch.nn.Linear)),'architecture':ck['architecture'],'selected_epoch':active['checkpoint_epoch'],'train_count':ck['train_count'],'val_count':ck['val_count'],'selected_lineage_full_epochs':full,'selected_lineage_head_epochs':head}
    write_json(ROOT/'reports/further_results_summary.json',summary)
    intro=f'''## Latest completed full-network experiment — 7 November 2026

Active model: **{100*active['accuracy']:.2f}% full-test accuracy**, macro precision **{m['precision']:.4f}**, recall **{m['recall']:.4f}**, F1 **{m['f1-score']:.4f}**; disgust recall **{100*d['recall']:.2f}%**, F1 **{d['f1-score']:.4f}**. Inference validation: **{100*val['accuracy']:.2f}%**. All 6,561 retained test faces and all seven classes remain included.

The new augmentation/distillation/averaged-weight experiment completed {trial['completion']['completed_additional_epochs']} full emotion-path epochs. Its candidate was {'deployed after passing the fixed integration gate' if trial['deployed'] else 'not deployed; the preceding evaluated model is retained'}. Results and regressions: [CONSISTENCY_RESULTS.md](CONSISTENCY_RESULTS.md). No 80% result is claimed. Validation is repeatedly reused and the combined FER test was historically inspected; this is not fresh webcam-domain evidence. Real fixed-game EEG/HRV fusion still shows no primary gain, and DREAMER official access is restricted. Earlier numerical sections below are historical.

'''
    for name in ['README.md','EXPERIMENTS.md','MODEL_CARDS.md','PAPER_IMPLEMENTATION_MAPPING.md','FURTHER_RESULTS.md','AUDIT_RESULTS.md']:
        path=ROOT/name;backup=RUN/'document_backups'/name;backup.parent.mkdir(exist_ok=True)
        if not backup.exists():shutil.copy2(path,backup)
        old=path.read_text(encoding='utf-8')
        marker='## Latest completed full-network experiment — 7 November 2026'
        if old.startswith(marker):old=old.split('Earlier numerical sections below are historical.\n\n',1)[1]
        path.write_text(intro+old,encoding='utf-8')
    readme=f"""# RagaMoodSync

Paper-informed facial-expression and Hindustani raga research app. Python3.11, PyTorch CPU, FastAPI, React/Vite and SQLite. Primary paper: https://doi.org/10.1007/s41870-025-02531-y . Exact author reproduction is not claimed.

## Current measured results

Validation {100*val['accuracy']:.2f}% ({val['n']} faces); full test {100*active['accuracy']:.2f}% ({active['n']} faces). Macro precision {m['precision']:.4f}, recall {m['recall']:.4f}, F1 {m['f1-score']:.4f}. Disgust recall {100*d['recall']:.2f}%, F1 {d['f1-score']:.4f}. All seven classes and all retained test errors are included. No80% result is claimed. The test is historically inspected, not fresh or identity-disjoint. Reliable webcam recognition remains unestablished.

The new full-network experiment completed {trial['completion']['completed_additional_epochs']} epochs; candidate deployed: {trial['deployed']}. See [CONSISTENCY_RESULTS.md](CONSISTENCY_RESULTS.md) for the protocol, complete metric comparison and regressions. Current source checkpoint SHA: `{sha}`.

Real participant-held-out EEG/ECG evaluation holds game and30-second duration constant; neither primary task shows a fusion F1 gain. DENS source-dimension problems and DREAMER restricted access prevent further validated results. No joint face/physiology/raga/weather efficacy is established. Sample signals are explicitly synthetic.

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
.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/consistency_20261107 --device cpu --threads 4 --baseline-report reports/consistency_20261107/baseline_test.json --baseline-manifest reports/consistency_20261107/baseline_test_manifest.json
.\.venv\Scripts\python.exe -m scripts.report_consistency
.\.venv\Scripts\python.exe -m pytest -q --basetemp reports/test_tmp_new_run
```

A completed evaluator verifies/reuses its frozen bundle. It refuses changed weights after test access. Use a fresh temporary test directory on each pytest invocation. Local experiment artifacts and manually acquired FER2013 are required for exact evaluation; the filtered GitHub app checkout excludes private datasets and environments.

See [DATASETS.md](DATASETS.md) for official access, format, licence and manual download procedures. Dataset absence does not prevent inference with the included trained model. Provision the separately excluded Raag Ratnakar resources using `python -m scripts.bootstrap_resources --source-dir PATH` or `--download` and preserve upstream notices. Catalog:72ragas,528 link entries,516uniqueURLs.

The newest editable13-slide deck is output/RagaMoodSync_Research_Consistency_Reviewed.pptx. Historical reports/decks and all original resources are retained. See MODEL_CARDS.md, PAPER_IMPLEMENTATION_MAPPING.md, THIRD_PARTY_NOTICES.md and PROJECT_STATUS.md for limitations and provenance. No clinical, patentability or publication claim is made.
"""
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    (ROOT/'PROJECT_STATUS.md').write_text('# Current project status — 7 November 2026\n\n'+intro+'Training has finished. Exact epoch logs, protocol and tensor-change audit are in reports/consistency_20261107/. Final regression, live API, PPT and GitHub verification are being completed. Original resources remain preserved.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
