"""Build a source-only GitHub checkout with a portable identical-weight model."""
import hashlib,json,shutil
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2),encoding='utf-8')
def main():
 out=ROOT/'output/github/RagaMoodSync'
 if (out/'.git').exists():raise SystemExit('Existing prepared repository preserved; create a new export revision.')
 out.mkdir(parents=True,exist_ok=True)
 for folder in ['backend','tests']:
  for p in (ROOT/folder).rglob('*.py'):
   if '__pycache__' not in p.parts:
    dest=out/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 for p in (ROOT/'frontend').rglob('*'):
  rel=p.relative_to(ROOT/'frontend')
  if p.is_file() and not any(x in ['node_modules','dist','.vite'] for x in rel.parts):
   dest=out/'frontend'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 scripts=['__init__.py','bootstrap_resources.py','colab_common.py','colab_train.py','colab_evaluate.py','refine_emotion_head.py','import_colab_model.py','train_emotion.py','evaluate_emotion.py','train_fdcf.py','evaluate_recommendations.py','gamer_experiment.py','gamer_controlled.py','gamer_fixed_context.py','verify_fixed_context.py','download_gamer.py','start-backend.ps1','start-frontend.ps1','package_colab.py']
 (out/'scripts').mkdir(exist_ok=True)
 for name in scripts:shutil.copy2(ROOT/'scripts'/name,out/'scripts'/name)
 for name in ['requirements.txt','requirements-colab.txt','pytest.ini','DATASETS.md','PAPER_ANALYSIS.md','PAPER_IMPLEMENTATION_MAPPING.md','LITERATURE_REVIEW.md','PRIOR_ART_AND_IP.md','IKS_METHODOLOGY.md','MODEL_CARDS.md','THIRD_PARTY_NOTICES.md','FURTHER_RESULTS.md','EXPERIMENTS.md','COLAB_README.md']:
  text=(ROOT/name).read_text(encoding='utf-8').replace(str(ROOT),'<PROJECT_ROOT>').replace('C:\\Users\\bindu\\OneDrive\\Desktop\\VEDIC RAGA','<PROJECT_ROOT>');(out/name).write_text(text,encoding='utf-8')
 for name in ['demo_face.png','sample_signals.json']:
  dest=out/'data'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'data'/name,dest)
 ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True)
 keys=['state_dict','classes','architecture','epoch','train_count','val_count','inference','pilot','synthetic','selection_metric']
 portable={k:ck[k] for k in keys if k in ck};portable['source_checkpoint_sha256']=sha(ROOT/'models/fusdense.pt');portable['head_refinement']=bool(ck.get('head_refinement'))
 dest=out/'models/fusdense.pt';dest.parent.mkdir(exist_ok=True);torch.save(portable,dest)
 restored=torch.load(dest,map_location='cpu',weights_only=True);assert all(torch.equal(v,restored['state_dict'][k]) for k,v in ck['state_dict'].items());newsha=sha(dest)
 for name in ['emotion_evaluation.json','validation_evaluation.json','expression_policy.json','calibration.json']:
  src=ROOT/'reports'/name
  if src.exists():
   value=json.loads(src.read_text());value.pop('predictions',None);value['source_checkpoint_sha256']=value.get('checkpoint_sha256');value['checkpoint_sha256']=newsha;write(out/'reports'/name,value)
 for name in ['training_history.json','gamer_metadata.json','gamer_fixed_context_evaluation.json','gamer_evaluation.json','gamer_evaluation_plus_offset.json']:
  src=ROOT/'reports'/name
  if src.exists():
   v=json.loads(src.read_text())
   if isinstance(v,dict) and 'tasks' in v:
    for task in v['tasks'].values():
     task.pop('predictions',None);task.pop('folds',None)
   write(out/'reports'/name,v)
 for name in ['emotion_evaluation.png','training_curves.png','gamer_fixed_context.png','gamer_fusion.png']:
  if (ROOT/'reports'/name).exists():shutil.copy2(ROOT/'reports'/name,out/'reports'/name)
 for name in ['RagaMoodSync_Research_Audited_Reviewed.pptx']:
  if (ROOT/'output'/name).exists():
   (out/'presentation').mkdir(exist_ok=True);shutil.copy2(ROOT/'output'/name,out/'presentation'/name)
 (out/'.gitignore').write_text('.venv/\n.python/\n**/node_modules/\n**/__pycache__/\n.pytest_cache/\nfrontend/dist/\nexternal/\ndata/private/\n**/*.sqlite*\n.env\n.env.*\n*.pem\n*.key\nreports/test_tmp*/\n',encoding='utf-8')
 e=json.loads((ROOT/'reports/emotion_evaluation.json').read_text());val=json.loads((ROOT/'reports/validation_evaluation.json').read_text());macro=e['classification_report']['macro avg']
 (out/'README.md').write_text(f"""# RagaMoodSync

Paper-informed FusDenseNet/FDCF research app, Python3.11, PyTorch CPU, FastAPI, React/Vite and SQLite. Original paper: https://doi.org/10.1007/s41870-025-02531-y . Exact reproduction is not claimed.

## Measured results

Validation accuracy {100*val['accuracy']:.2f}% on {val['n']} faces. Full test accuracy {100*e['accuracy']:.2f}% on all {e['n']} retained faces, macro F1 {macro['f1-score']:.4f}. No80% result is claimed. FER2013 test was historically inspected; it is not a fresh or identity-disjoint benchmark. Class metrics and confusion matrix include every error. Strong webcam/disgust reliability remains unestablished.

Real EEG/ECG tests hold game identity and30-second duration constant with held-out participants. Neither primary task shows a fusion macro-F1 gain. DENS evaluation is incomplete because of source-dimension mismatches. No simulated signal is reported as real. Weather/context ranking and FDCF recommendation-quality benefits remain unvalidated.

## Setup and run

Install Python3.11 and Node.js22 or newer compatible with the pinned Vite version.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.bootstrap_resources --download
cd frontend
npm ci
node node_modules/vite/bin/vite.js build --configLoader native
cd ..
.\scripts\start-backend.ps1
```

Open http://127.0.0.1:8000/ . Research metrics: http://127.0.0.1:8000/?view=research&tab=Research . For frontend development: cd frontend, then npm run dev.

The72-raga catalog/528 links is a separately provisioned Raag Ratnakar dependency. It is excluded from this repository because its supplied licence is unresolved. The bootstrap verifies the pinned inspected checksum; alternatively use --source-dir with an existing local source. Original Navrasa weights are optional and provisioned separately. No paper PDF, third-party archive, large dataset, personal photograph, private feedback database or dependency environment is committed.

The included portable model has bit-identical weights and identical inference settings to the evaluated source checkpoint. Its serialized metadata is minimized to omit local paths. reports/model_export_provenance.json records both hashes and the exact tensor-equality check. Metrics are copied from that equivalent model, never invented.

## Tests and datasets

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp reports/test_tmp_local
```

See DATASETS.md for FER2013, DEAP, WESAD, PhysioNet, Saraga, IndRaga, CompMusic and additional physiological data. Download FER2013 manually under data/private/ for new training. Training scripts require verified split manifests generated for your acquired data; the included app does not require downloading FER2013.

FusDenseNet retains dense fusion, cross-attention and multi-pooling. Project spatial/classifier dimensions and missing infrared assumptions are documented. FDCF contains user/music/emotion embeddings and dynamic preference focus; real interaction-based quality evaluation remains pending. Feedback and history persist locally in SQLite. EEG/HRV sample mode is explicitly synthetic; actual benchmark reports are separate.

See THIRD_PARTY_NOTICES.md and the research Markdown files for attribution, limitations and unresolved redistribution permissions. No clinical, patentability or publication claim is made.
""",encoding='utf-8')
 provenance={'source_checkpoint_sha256':sha(ROOT/'models/fusdense.pt'),'portable_checkpoint_sha256':newsha,'all_state_tensors_bit_identical':True,'inference_settings_identical':True,'metadata_local_paths_removed':True,'no_new_performance_claim':True};write(out/'reports/model_export_provenance.json',provenance)
 files=[p for p in out.rglob('*') if p.is_file()];assert all(p.stat().st_size<95*1024*1024 for p in files)
 assert not any(p.suffix in ['.sqlite3','.csv','.fdt','.set'] or p.name=='.env' for p in files)
 write(ROOT/'reports/audit_20261107/github_export_manifest.json',{'root':str(out),'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'publication_status':'not pushed; target account write access required'})
 print(out,len(files),'reviewed files; dataset/private files excluded')
if __name__=='__main__':main()
