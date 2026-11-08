"""Refresh the existing filtered publication checkout without touching Git history."""
import copy,json,shutil,subprocess
from pathlib import Path
import torch
from scripts.colab_common import digest,write_json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/github/RagaMoodSync'

def main():
    if not (OUT/'.git').is_dir():raise RuntimeError('Expected existing publication repository')
    status=subprocess.run(['git','status','--porcelain'],cwd=OUT,capture_output=True,text=True,check=True).stdout
    if status.strip():raise RuntimeError('Publication checkout has changes; inspect before refreshing')
    files=['backend/main.py','frontend/src/research/TrainingEvidence.jsx','scripts/train_consistency.py','scripts/report_consistency.py','scripts/integrate_consistency.py','tests/test_consistency.py','tests/test_experiment_provenance.py',
           'LITERATURE_REVIEW.md','DATASETS.md','README.md','EXPERIMENTS.md','MODEL_CARDS.md','PAPER_IMPLEMENTATION_MAPPING.md','FURTHER_RESULTS.md','AUDIT_RESULTS.md','CONSISTENCY_RESULTS.md']
    # Keep the publication-specific setup instructions, filtering notes and portable-model explanation.
    files.remove('README.md')
    for name in files:
        src=ROOT/name;dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True)
        text=src.read_text(encoding='utf-8-sig').replace(str(ROOT),'<PROJECT_ROOT>').replace(str(ROOT).replace('\\','/'),'<PROJECT_ROOT>')
        dest.write_text(text,encoding='utf-8')
    ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True)
    old=torch.load(OUT/'models/fusdense.pt',map_location='cpu',weights_only=True)
    identical=all(k in old['state_dict'] and torch.equal(v,old['state_dict'][k]) for k,v in ck['state_dict'].items()) and ck.get('inference')==old.get('inference')
    if not identical:
        keys=['state_dict','classes','architecture','epoch','train_count','val_count','inference','pilot','synthetic','selection_metric','training_stage']
        portable={k:ck[k] for k in keys if k in ck};portable['source_checkpoint_sha256']=digest(ROOT/'models/fusdense.pt');portable['head_refinement']=bool(ck.get('head_refinement'))
        torch.save(portable,OUT/'models/fusdense.pt')
    restored=torch.load(OUT/'models/fusdense.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,restored['state_dict'][k]) for k,v in ck['state_dict'].items()) and restored['inference']==ck['inference']
    source_sha=digest(ROOT/'models/fusdense.pt');portable_sha=digest(OUT/'models/fusdense.pt')
    for name in ['emotion_evaluation.json','validation_evaluation.json','expression_policy.json','calibration.json','emotion_training_comparison.json','latest_emotion_experiment.json']:
        value=json.loads((ROOT/'reports'/name).read_text());value.pop('predictions',None)
        def remap(v):
            if isinstance(v,dict):
                for k,val in list(v.items()):
                    if k in ['checkpoint_sha256','candidate_checkpoint_sha256'] and val==source_sha:v[k]=portable_sha
                    else:remap(val)
            elif isinstance(v,list):
                for item in v:remap(item)
        value['source_checkpoint_sha256']=value.get('checkpoint_sha256',source_sha);remap(value);write_json(OUT/'reports'/name,value)
    shutil.copy2(ROOT/'reports/training_history.json',OUT/'reports/training_history.json')
    for name in ['emotion_evaluation.png','training_curves.png']:shutil.copy2(ROOT/'reports'/name,OUT/'reports'/name)
    summary=json.loads((ROOT/'reports/consistency_20261107/experiment_summary.json').read_text())
    for key in ['validation_initial_raw','validation_selected_raw']:summary[key].pop('predictions',None)
    write_json(OUT/'reports/consistency_experiment.json',summary)
    write_json(OUT/'reports/model_export_provenance.json',{'source_checkpoint_sha256':source_sha,'portable_checkpoint_sha256':portable_sha,'state_tensors_bit_identical':True,'inference_settings_identical':True,'private_paths_removed':True,'metrics_not_changed':True})
    deck=ROOT/'output/RagaMoodSync_Research_Consistency_Reviewed.pptx';shutil.copy2(deck,OUT/'presentation'/deck.name)
    e=json.loads((ROOT/'reports/emotion_evaluation.json').read_text());v=json.loads((ROOT/'reports/validation_evaluation.json').read_text());m=e['classification_report']['macro avg']
    p=OUT/'README.md';text=p.read_text(encoding='utf-8');start=text.index('Validation accuracy');end=text.index('\n\nReal EEG/ECG',start)
    text=text[:start]+f"Validation accuracy {100*v['accuracy']:.2f}% on {v['n']} faces. Full test accuracy {100*e['accuracy']:.2f}% on all {e['n']} retained faces, macro F1 {m['f1-score']:.4f}. No80% result is claimed. FER2013 test was historically inspected; it is not a fresh or identity-disjoint benchmark. All class errors remain included. Strong webcam/disgust reliability remains unestablished. The completed full-network consistency experiment and any regressions are documented in CONSISTENCY_RESULTS.md. The latest editable13-slide deck is presentation/RagaMoodSync_Research_Consistency_Reviewed.pptx."+text[end:]
    p.write_text(text,encoding='utf-8')
    print(json.dumps({'source_sha':source_sha,'portable_sha':portable_sha,'bit_identical':True,'copied_files':len(files),'git_history_modified':False},indent=2))
if __name__=='__main__':main()
