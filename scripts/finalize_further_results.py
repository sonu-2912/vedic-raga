"""Publish completion only after actual metrics, live audit and PPT validation."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/n).read_text())
def main():
    e=read('reports/further_results_summary.json');v=read('reports/further_verification_summary.json');r=read('reports/slides_controlled/validation-controlled-final.json')
    assert v['failures']==0 and v['model_sha256']==e['checkpoint_sha256']
    assert r['packageIntegrity']['slide_count']==13 and r['packageIntegrity']['findingCount']==r['presentationLayout']['findingCount']==0
    ppt=ROOT/'output/RagaMoodSync_Research_Controlled_Final.pptx';assert hashlib.sha256(ppt.read_bytes()).hexdigest()==r['finalSha256']
    generic=ROOT/'output/RagaMoodSync_Research.pptx';backup=ROOT/'output/RagaMoodSync_Research_Before_Controlled_Update.pptx'
    if generic.exists() and not backup.exists():shutil.copy2(generic,backup)
    shutil.copy2(ppt,generic)
    previous=ROOT/'README.md';backup=ROOT/'reports/further_document_backups/README.md';backup.parent.mkdir(exist_ok=True)
    if not backup.exists():shutil.copy2(previous,backup)
    d=e['disgust'];summary=f'''# RagaMoodSync

Paper-informed Hindustani raga application, Windows / Python 3.11 / PyTorch CPU. Research authority: [Pingle & Ragha (2025)](https://doi.org/10.1007/s41870-025-02531-y). Layer dimensions and the v3 spatial extension are disclosed project choices, not exact reproduction.

## Actual current results

FusDenseNet v{e['architecture']['version']}, {e['parameters']:,} parameters. Full unfiltered test: {e['test_n']:,} faces, all seven classes and all errors included.

| Metric | Previous model | Current model |
|---|---:|---:|
| Accuracy | 52.55% | {100*e['accuracy']:.2f}% |
| Macro precision | 0.4718 | {e['macro_precision']:.4f} |
| Macro recall | 0.4485 | {e['macro_recall']:.4f} |
| Macro F1 | 0.4492 | {e['macro_f1']:.4f} |
| Disgust recall | 12.66% | {100*d['recall']:.2f}% |
| Disgust F1 | 0.1835 | {d['f1-score']:.4f} |

Overall accuracy improved by only 2 correctly classified faces out of 6,561. Disgust precision fell from 33.33% to 29.51%. Disgust and webcam reliability remain limited. This is a historically inspected combined FER2013 test folder, not a fresh or identity-disjoint benchmark. Fixed split: 23,503 training / 3,970 validation images. Checkpoint SHA-256 `{e['checkpoint_sha256']}`.

The new spatial path preserves the three fusion-dense stages, cross-attention and original pooling. Training-only normalization is folded into the MLP. Several further recipes regressed and are retained as negative results. Full-network and classifier-only epochs are reported separately. No guaranteed accuracy target or paper-level performance is claimed.

Real EEG/ECG evaluation holds **game identity and 30-second duration constant**, with participant-held-out testing. QWOP fusion macro F1 0.4583 equals EEG; CBAAD fusion macro F1 0.6061 equals EEG. Fusion accuracy is below majority accuracy in both tasks. There is no demonstrated primary fusion gain. Raw-window/fold audits passed. Annotation timing, artifacts, small cohorts and ultra-short HRV remain limitations. The older pooled game/label result is historical and confounded.

DENS game-free evaluation remains incomplete: only 2 of the predeclared 12 participant headers match their referenced signal dimensions. No DENS score is claimed. No simulated EEG/HRV result is presented as real. Facial-plus-physiological fusion, weather benefit, real raga outcomes, reliable disgust recognition and interaction-trained FDCF ranking quality remain research gaps.

**{v['passed']} automated tests passed** ({v['test_seconds']:.2f}s, 15 existing deprecation warnings). The six export workflow tests were rerun successfully after the deployment-rule update. Production frontend build, live model/checksum/API inference and recommendation checks passed. Original 55 repository files remain byte-identical.

## Run the app

```powershell
cd 'C:\\Users\\bindu\\OneDrive\\Desktop\\VEDIC RAGA'
.\\scripts\\start-backend.ps1
```

Open http://127.0.0.1:8000/ . The built React application is served by FastAPI. The launcher reuses a healthy existing backend and avoids duplicate port-8000 starts. API docs: http://127.0.0.1:8000/docs . Research evidence: http://127.0.0.1:8000/?view=research&tab=Research .

Frontend development, in a second terminal:

```powershell
cd 'C:\\Users\\bindu\\OneDrive\\Desktop\\VEDIC RAGA\\frontend'
npm run dev
```

## Exact evaluation and verification commands

```powershell
.\\.venv\\Scripts\\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/further_selected_run --device cpu --threads 2 --baseline-report reports/further_baseline/emotion_evaluation.json --baseline-manifest reports/further_baseline/emotion_evaluation_manifest.json
.\\.venv\\Scripts\\python.exe -m scripts.gamer_fixed_context
.\\.venv\\Scripts\\python.exe -m scripts.verify_fixed_context
.\\.venv\\Scripts\\python.exe -m scripts.verify_further_results
.\\.venv\\Scripts\\python.exe -m pytest -q --junitxml=reports/pytest.xml
```

The FER command verifies/reuses its frozen evaluated export instead of retraining against test results. Full predictions, per-class reports, confusion matrix and training curves remain in reports/. Current deployment bundle: reports/further_selected_run/RagaMoodSync_Trained_Model.zip .

## Datasets, reports and editable presentation

FER2013 and GamER originals are already local under data/private/. Access, licence, formats and manual procedures for FER2013, DEAP, WESAD, PhysioNet, Saraga, IndRaga and CompMusic are documented in [DATASETS.md](DATASETS.md). DENS integrity/access evidence is in reports/dens_access_outcome.json. Registration-controlled datasets remain pending. Do not redistribute source assets without resolving their rights.

Current PowerPoint: **output/RagaMoodSync_Research_Controlled_Final.pptx**, exactly 13 slides, with native editable chart/tables and actual results. All rendered slides were reviewed and structural/layout/import validation passed. Generic output/RagaMoodSync_Research.pptx is the same file. Older decks remain preserved. Native desktop PowerPoint execution is not claimed.

Detailed current evidence: [FURTHER_RESULTS.md](FURTHER_RESULTS.md). Also see PAPER_ANALYSIS.md, PAPER_IMPLEMENTATION_MAPPING.md, LITERATURE_REVIEW.md, PRIOR_ART_AND_IP.md, IKS_METHODOLOGY.md, DATASETS.md, EXPERIMENTS.md and MODEL_CARDS.md. Earlier numerical sections are retained as historical and superseded by the latest update.

72 ragas, 528 link entries, 516 unique URLs. Source: https://github.com/alronova/raag-ratnakar . Missing metadata stays unknown. The supplied Navrasa model remains separate from FusDenseNet. See THIRD_PARTY_NOTICES.md for attribution and unresolved source redistribution rights.
'''
    previous.write_text(summary,encoding='utf-8')
    status={'stage':'completed further training/evaluation/app/PPT update','updated':datetime.now().isoformat(),'emotion':e,'verification':v,'controlled_physiology':'Completed: real fixed-game/fixed-duration participant-held-out evaluation; no primary fusion gain','DENS':'Incomplete: source header/FDT mismatch leaves only2/12 participants; no metric','presentation':str(ppt.relative_to(ROOT)),'presentation_sha256':r['finalSha256'],'slides':13,'visual_review':'All13 rendered slides inspected individually, including changed source/output comparisons','limitations':['Weak disgust/webcam recognition','Historically inspected FER test','Short-window HRV and uncertain offset alignment','No validated face/physiology/raga/weather joint outcome or recommendation-quality result']}
    write=ROOT/'reports/further_finalization_status.json';write.write_text(json.dumps(status,indent=2),encoding='utf-8')
    for relative,value in [('reports/finalization_status.json',status),('reports/verification_summary.json',{**v,'real_fer2013_metrics':e,'context_evaluation':'Actual fixed-game and fixed-30s participant-held-out EEG/ECG; no primary fusion gain','clinical_results':None})]:
        dest=ROOT/relative;old=ROOT/'reports/further_document_backups'/dest.name
        if dest.exists() and not old.exists():shutil.copy2(dest,old)
        dest.write_text(json.dumps(value,indent=2),encoding='utf-8')
    (ROOT/'PROJECT_STATUS.md').write_text('# Current project status\n\n'+summary.split('## Actual current results')[1].split('## Run the app')[0]+'\n\nCompleted: trained/evaluated further FusDenseNet candidates, installed the qualifying checkpoint, updated real physiological controls/API/UI, documentation and editable13-slide PPT. Exact files/checksums and limitations: reports/further_finalization_status.json. Historical status is preserved in reports/further_document_backups/PROJECT_STATUS.md.\n',encoding='utf-8')
    print(json.dumps({'accuracy':e['accuracy'],'macro_f1':e['macro_f1'],'disgust_recall':d['recall'],'disgust_f1':d['f1-score'],'tests_passed':v['passed'],'ppt':str(ppt),'ppt_sha256':r['finalSha256']},indent=2))
if __name__=='__main__':main()
