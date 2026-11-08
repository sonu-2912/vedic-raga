"""Refresh the existing filtered publication tree after audit-driven spatial trials finish."""
import json,shutil,subprocess
from pathlib import Path
import torch
from scripts.colab_common import digest,write_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/github/RagaMoodSync'

def main():
    if not (OUT/'.git').is_dir():raise RuntimeError('Expected existing publication checkout')
    if subprocess.check_output(['git','status','--porcelain'],cwd=OUT,text=True).strip():raise RuntimeError('Inspect existing publication edits before refreshing')
    paths=['scripts/train_direct_dense_head.py','scripts/train_direct_dense_head_long.py','backend/inference.py','backend/emotion.py','scripts/train_dense_transfer_probe.py','scripts/select_spatial_repair.py','scripts/screen_translation_tta.py','licenses/Torchvision_BSD_3_Clause.txt','backend/models.py','scripts/train_spatial_generalization.py','scripts/report_spatial_generalization.py','scripts/audit_emotion_bottlenecks.py','tests/test_spatial_generalization.py','SPATIAL_GENERALIZATION_RESULTS.md','backend/main.py','frontend/src/research/TrainingEvidence.jsx','frontend/src/research/ContinuationEvidence.jsx','scripts/train_consistency.py','scripts/train_deep_supervision.py','scripts/train_pruned_head.py','scripts/train_sam_head.py','scripts/select_balanced_candidate.py','scripts/report_deep_supervision.py','scripts/colab_evaluate.py','scripts/screen_expression_teacher.py','scripts/download_expression_teacher.py','tests/test_deep_supervision.py','tests/test_pruned_head.py','tests/test_sam_head.py','tests/test_teacher_contract.py','BALANCED_RESULTS.md','DEEP_SUPERVISION_RESULTS.md','AUDIT_RESULTS.md','DATASETS.md','EXPERIMENTS.md','FURTHER_RESULTS.md','LITERATURE_REVIEW.md','MODEL_CARDS.md','PAPER_IMPLEMENTATION_MAPPING.md','THIRD_PARTY_NOTICES.md','licenses/OpenCV_Zoo_Apache_2.0.txt']
    for name in paths:
        src=ROOT/name;dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True)
        text=src.read_text(encoding='utf-8-sig').replace(str(ROOT),'<PROJECT_ROOT>').replace(str(ROOT).replace('\\','/'),'<PROJECT_ROOT>').replace('C:\\Users\\bindu\\OneDrive\\Desktop\\VEDIC RAGA','<PROJECT_ROOT>')
        dest.write_text(text,encoding='utf-8')
    ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True);old=torch.load(OUT/'models/fusdense.pt',map_location='cpu',weights_only=True)
    same=ck['architecture']==old['architecture'] and ck.get('inference')==old.get('inference') and all(k in old['state_dict'] and torch.equal(v,old['state_dict'][k]) for k,v in ck['state_dict'].items())
    if not same:
        keys=['state_dict','classes','architecture','epoch','train_count','val_count','inference','pilot','synthetic','selection_metric','training_stage'];portable={k:ck[k] for k in keys if k in ck};portable['source_checkpoint_sha256']=digest(ROOT/'models/fusdense.pt');portable['head_refinement']=bool(ck.get('head_refinement'));torch.save(portable,OUT/'models/fusdense.pt')
    restored=torch.load(OUT/'models/fusdense.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v,restored['state_dict'][k]) for k,v in ck['state_dict'].items()) and restored['inference']==ck['inference']
    source=digest(ROOT/'models/fusdense.pt');portable=digest(OUT/'models/fusdense.pt')
    def remap(v):
        if isinstance(v,dict):
            for k,val in list(v.items()):
                if k in ['checkpoint_sha256','candidate_checkpoint_sha256'] and val==source:v[k]=portable
                else:remap(val)
        elif isinstance(v,list):
            for item in v:remap(item)
    for name in ['emotion_evaluation.json','validation_evaluation.json','expression_policy.json','calibration.json','emotion_training_comparison.json','latest_emotion_experiment.json']:
        v=json.loads((ROOT/'reports'/name).read_text());v.pop('predictions',None);v['source_checkpoint_sha256']=v.get('checkpoint_sha256',source);remap(v);write_json(OUT/'reports'/name,v)
    for name in ['training_history.json']:shutil.copy2(ROOT/'reports'/name,OUT/'reports'/name)
    for name in ['emotion_evaluation.png','training_curves.png','spatial_generalization_trials.png']:shutil.copy2(ROOT/'reports'/name,OUT/'reports'/name)
    v=json.loads((ROOT/'reports/spatial_generalization_20261108/experiment_summary.json').read_text());v['active_source_sha256']=v.pop('active_source_sha256');v['active_portable_sha256']=portable;remap(v);write_json(OUT/'reports/spatial_generalization_experiment.json',v)
    write_json(OUT/'reports/bottleneck_audit_20261108/audit.json',json.loads((ROOT/'reports/bottleneck_audit_20261108/audit.json').read_text(encoding='utf-8')))
    write_json(OUT/'reports/dense_transfer112_20261108/pretrained_provenance.json',json.loads((ROOT/'reports/dense_transfer112_20261108/pretrained_provenance.json').read_text(encoding='utf-8')))
    for name in ['provenance.json','decision.json','val_evaluation.json']:
        v=json.loads((ROOT/'reports/teacher_screen_20261107'/name).read_text());v.pop('predictions',None);write_json(OUT/'reports/teacher_screen_20261107'/name,v)
    write_json(OUT/'reports/model_export_provenance.json',{'source_checkpoint_sha256':source,'portable_checkpoint_sha256':portable,'state_tensors_bit_identical':True,'inference_settings_identical':True,'private_paths_removed':True,'metrics_not_changed':True})
    deck=ROOT/'output/RagaMoodSync_Research_Spatial_Generalization_Reviewed_v3.pptx';shutil.copy2(deck,OUT/'presentation'/deck.name)
    e=json.loads((ROOT/'reports/emotion_evaluation.json').read_text());v=json.loads((ROOT/'reports/validation_evaluation.json').read_text());m=e['classification_report']['macro avg'];trial=json.loads((ROOT/'reports/latest_emotion_experiment.json').read_text())
    p=OUT/'README.md';s=p.read_text(encoding='utf-8');a=s.index('Validation accuracy');b=s.index('\n\nReal EEG/ECG',a)
    s=s[:a]+f"Validation accuracy {100*v['accuracy']:.2f}% on {v['n']} faces. Full unfiltered test accuracy {100*e['accuracy']:.2f}% on all {e['n']} retained faces; macro precision {m['precision']:.4f}, recall {m['recall']:.4f}, F1 {m['f1-score']:.4f}. Active architecture v{ck['architecture']['version']}; v7 is a probability ensemble retaining FusDenseNet plus a frozen ImageNet DenseNet121 encoder and independently trained classifier, not a pure FusDenseNet result. No80% result or reliable webcam claim. Test was historically inspected, not fresh/identity-disjoint. Latest trials: {trial['completed_full_epochs']} full-network and {trial['completed_head_epochs']} classifier-only epochs; all actual failures/regressions and any no-test outcome are in SPATIAL_GENERALIZATION_RESULTS.md. Exactly13editable slides: presentation/RagaMoodSync_Research_Spatial_Generalization_Reviewed_v3.pptx. External teacher screen was rejected and never used for FusDenseNet training."+s[b:]
    p.write_text(s,encoding='utf-8');print(json.dumps({'source_sha':source,'portable_sha':portable,'tensors_bit_identical':True,'architecture_unchanged_from_source':True,'files_updated':len(paths),'raw_private_data_exported':False},indent=2))
if __name__=='__main__':main()
