"""Refresh filtered publication only after verified pure improvement and PPT review."""
import json,shutil,subprocess
from pathlib import Path
import torch
from scripts.colab_common import digest,write_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/github/RagaMoodSync';RUN=ROOT/'reports/pure_orientation_20261108'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 torch.set_num_threads(2);decision=read(RUN/'deployment_decision.json');assert decision['eligible'] and decision['installed'];assert digest(ROOT/'models/fusdense.pt')==decision['checkpoint_sha256']
 if subprocess.check_output(['git','status','--porcelain'],cwd=OUT,text=True).strip():raise RuntimeError('Publication checkout has edits; inspect before refresh')
 review=read(ROOT/'reports/slides_orientation_repair_v3/visual_review.json');assert review['all_rendered_slides_reviewed_by_agent'] and review['slides']==13
 paths=['backend/main.py','frontend/src/research/TrainingEvidence.jsx','scripts/audit_pure_failure.py','scripts/audit_orientation_integrity.py','scripts/train_pure_coverage_repair.py','scripts/train_orientation_head_repair.py','scripts/evaluate_pure_coverage_repair.py','scripts/evaluate_orientation_repair.py','scripts/report_coverage_repair.py','scripts/report_orientation_repair.py','scripts/record_coverage_transition.py','scripts/keep_training_awake.py','scripts/continue_orientation_repair.py','tests/test_pure_coverage_repair.py','tests/test_orientation_repair.py','PURE_REPAIR_ANALYSIS.md','PURE_ORIENTATION_REPAIR_RESULTS.md','EXPERIMENTS.md','MODEL_CARDS.md','PAPER_IMPLEMENTATION_MAPPING.md','FURTHER_RESULTS.md','PROJECT_STATUS.md','AUDIT_RESULTS.md','LITERATURE_REVIEW.md']
 for name in paths:
  source=ROOT/name;target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);text=source.read_text(encoding='utf-8-sig').replace(str(ROOT),'<PROJECT_ROOT>').replace(str(ROOT).replace('\\','/'),'<PROJECT_ROOT>');target.write_text(text,encoding='utf-8')
 ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True);assert not any(k.startswith('transfer_') for k in ck['state_dict']);keys=['state_dict','classes','architecture','epoch','train_count','val_count','inference','pilot','synthetic','selection_metric','training_stage'];portable={k:ck[k] for k in keys if k in ck};portable['source_checkpoint_sha256']=digest(ROOT/'models/fusdense.pt');portable['head_refinement']=bool(ck.get('head_refinement'));torch.save(portable,OUT/'models/fusdense.pt');restored=torch.load(OUT/'models/fusdense.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v,restored['state_dict'][k]) for k,v in ck['state_dict'].items()) and restored['inference']==ck['inference'];source=digest(ROOT/'models/fusdense.pt');export=digest(OUT/'models/fusdense.pt')
 def remap(v):
  if isinstance(v,dict):
   for k,value in list(v.items()):
    if k in ['checkpoint_sha256','candidate_checkpoint_sha256'] and value==source:v[k]=export
    else:remap(value)
  elif isinstance(v,list):
   for item in v:remap(item)
 for name in ['emotion_evaluation.json','validation_evaluation.json','expression_policy.json','calibration.json','emotion_training_comparison.json','latest_emotion_experiment.json']:
  v=read(ROOT/'reports'/name);v.pop('predictions',None);v['source_checkpoint_sha256']=source;remap(v);write_json(OUT/'reports'/name,v)
 shutil.copy2(ROOT/'reports/training_history.json',OUT/'reports/training_history.json')
 for name in ['emotion_evaluation.png','training_curves.png','pure_orientation_validation.png','pure_orientation_validation_confusion.png']:shutil.copy2(ROOT/'reports'/name,OUT/'reports'/name)
 for source_path,destination in [(RUN/'repair_summary.json','orientation_repair_summary.json'),(RUN/'protocol.json','orientation_repair_protocol.json'),(RUN/'paired_uncertainty.json','orientation_repair_uncertainty.json'),(ROOT/'reports/pure_repair_20261108/failure_audit.json','pure_training_failure_audit.json'),(ROOT/'reports/pure_repair_20261108/sampling_audit.json','pure_training_sampling_audit.json'),(ROOT/'reports/pure_repair_20261108/budget_reallocation_protocol.json','coverage_budget_reallocation.json')]:
  v=read(source_path);remap(v);write_json(OUT/'reports'/destination,v)
 integrity=read(RUN/'mirror_integrity_audit.json');write_json(OUT/'reports/orientation_mirror_integrity.json',{k:v for k,v in integrity.items() if k not in ['mirror_pixel_hashes','blocked_mirror_train_indices']})
 write_json(OUT/'reports/model_export_provenance.json',{'source_checkpoint_sha256':source,'portable_checkpoint_sha256':export,'state_tensors_bit_identical':True,'inference_settings_identical':True,'private_paths_removed':True,'metrics_not_changed':True})
 deck=ROOT/'output/RagaMoodSync_Research_Pure_Orientation_Repair_Reviewed_v3.pptx';assert digest(deck)==review['sha256'];shutil.copy2(deck,OUT/'presentation'/deck.name)
 e=read(ROOT/'reports/emotion_evaluation.json');v=read(ROOT/'reports/validation_evaluation.json');m=e['classification_report']['macro avg'];d=e['classification_report']['disgust'];p=OUT/'README.md';text=p.read_text(encoding='utf-8');a=text.index('Validation accuracy');b=text.index('\n\nReal EEG/ECG',a)
 text=text[:a]+f'''Validation accuracy {100*v['accuracy']:.2f}% on3,970faces. Active **pure FusDenseNet** full test accuracy **{100*e['accuracy']:.2f}%** on all6,561retained faces; macro precision {m['precision']:.4f}, recall {m['recall']:.4f}, F1 **{m['f1-score']:.4f}**. Disgust precision **{d['precision']:.4f}**, recall **{100*d['recall']:.2f}%**, F1 **{d['f1-score']:.4f}**. Disgust correct29→37outof79; accuracy57.42%→58.34%; all seven tracked aggregate/disgust metrics improved against the earlier pure model.

The gain came from its native512-unit classifier learning original/mirrored TRAIN features from the same pure backbone. All non-classifier tensors remain bit-identical. No external encoder, teacher, inference ensemble or class threshold. Six independent full-network coverage epochs failed;72cheaper classifier-only epochs across512/256widths produced one frozen testcandidate. They are not equivalent compute or all ancestors of the chosen model.47,006views come from23,503source images, not new people. One single-view testcandidate, no alternate testsearch. Accuracy remains modest; no80%or reliablewebcam/clinical claim. Test historically inspected and validation reused, not fresh identity-disjoint confirmation. Complete gains, regressions, diagnostics and uncertainty: PURE_ORIENTATION_REPAIR_RESULTS.md.

Exactly13editable slides: presentation/RagaMoodSync_Research_Pure_Orientation_Repair_Reviewed_v3.pptx. Historical experiments/decks remain preserved.''' +text[b:]
 text='\n'.join(line for line in text.splitlines() if not line.startswith('Pure-training update:'))+'\n';text+='\nTraining scripts require the locally retained original pure checkpoint, fixed FER manifest and manually acquired FER2013. Private images/feature caches, feedback and environments are excluded from this filtered checkout. Included portable model tensors/inference are bit-identical to the evaluated source; source and portable hashes differ because private training metadata is removed.\n';p.write_text(text,encoding='utf-8')
 p=OUT/'PROJECT_STATUS.md';p.write_text(p.read_text(encoding='utf-8').replace('output/RagaMoodSync_Research_Pure_Orientation_Repair_Reviewed_v3.pptx','presentation/RagaMoodSync_Research_Pure_Orientation_Repair_Reviewed_v3.pptx'),encoding='utf-8');print(json.dumps({'source_sha256':source,'portable_sha256':export,'tensors_identical':True,'inference_identical':True,'raw_private_data_exported':False,'plaintext_files_refreshed':len(paths)},indent=2))
if __name__=='__main__':main()
