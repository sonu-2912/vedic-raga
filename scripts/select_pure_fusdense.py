"""Select one qualifying PURE checkpoint, before any new test evaluation."""
import json,shutil
from pathlib import Path
import torch
from scripts.colab_common import digest,write_json
from scripts.train_pure_fusdense import assert_pure,qualifies,selection_score
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'reports/pure_fusdense_20261108';B=ROOT/'reports/pure_native_residual_20261108';OUT=ROOT/'reports/pure_selected_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 if (OUT/'selection.json').exists():raise RuntimeError('Selection already frozen')
 baseline=read(A/'initial_validation.json');options=[]
 for folder in [A,B]:
  done=read(folder/'completion.json')
  if done['validation_improved']:
   r=read(folder/'best_validation.json');assert qualifies(r,baseline);ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);assert_pure(ck);options.append({'run':str(folder.relative_to(ROOT)).replace('\\','/'),'score':selection_score(r),'architecture':ck['architecture'],'validation_accuracy':r['accuracy'],'validation_macro_f1':r['classification_report']['macro avg']['f1-score'],'validation_disgust_recall':r['classification_report']['disgust']['recall']})
 OUT.mkdir(exist_ok=True);chosen=max(options,key=lambda x:x['score']) if options else None
 if chosen:
  folder=ROOT/chosen['run']
  for name in ['best.pt','best_validation.json','settings.json','training_history.json','fer_training_manifest.json','initialization_audit.json']:shutil.copy2(folder/name,OUT/name)
  for name in ['baseline_test.json','baseline_test_manifest.json','hybrid_reference_test.json']:shutil.copy2(A/name,OUT/name)
  write_json(OUT/'candidate_frozen.json',{'best_sha256':digest(OUT/'best.pt'),'test_accessed_at_freeze':False,'external_encoder':False,'teacher':False,'single_view_only':True})
 selection={'chosen':chosen,'candidates':options,'all_pure_recipes_completed':True,'test_accessed_at_selection':False,'rule':'One max-scoring validation-qualified purecheckpoint, no alternate testcandidate','timing':'Native residual-dense variant specified during v4training before anynewtest. V4budget capped/reallocated; both actualcounts preserved. Reused validation, not a fresh prospective benchmark.'}
 write_json(OUT/'selection.json',selection)
 for folder in [A,B]:write_json(folder/'candidate_frozen.json',{'training_locked':True,'selected_directory':str(OUT.relative_to(ROOT)).replace('\\','/'),'test_accessed_at_selection':False})
 print(json.dumps(selection,indent=2))
if __name__=='__main__':main()
