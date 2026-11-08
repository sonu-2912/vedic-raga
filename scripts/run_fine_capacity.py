"""Select and test at most one pure recovery candidate after all training stages."""
import json,subprocess,sys,time,shutil
from pathlib import Path
import torch
from scripts.colab_common import digest,write_json,model_from,fixed_split,predictions,metrics
from scripts.recall_recovery_common import eligible,objective
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_fine_head_20261109'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 prior=ROOT/'reports/pure_native_rollback_20261109';deadline=time.monotonic()+70*60
 while not (prior/'deployment_decision.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Prior experiment incomplete; no alternate candidate')
  time.sleep(10)
 previous=read(prior/'deployment_decision.json')
 if previous['test_evaluated'] or previous['eligible']:
  write_json(RUN/'skipped.json',{'reason':'Prior candidate already tested or qualified; no alternate test search.'});return
 with (RUN/'head_training.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-u','-m','scripts.train_fine_capacity','--run-dir',str(RUN.relative_to(ROOT)),'--threads','4'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 assert all(digest(ROOT/name)==sha for name,sha in read(RUN/'protected_artifacts.json').items())
 options=[]
 for name in ['head']:
  if (RUN/f'{name}_best.pt').exists():
   report=read(RUN/f'{name}_best_validation.json');assert eligible(report,read(RUN/'current_validation.json'),read(RUN/'earlier_validation.json'));options.append((objective(report),name,report))
 if not options:
  write_json(RUN/'deployment_decision.json',{'eligible':False,'test_evaluated':False,'installed':False,'reason':'No moderate fine-spatial native classifier checkpoint recovered all requested classes while retaining disgust and aggregate metrics. App/PPT unchanged.'});return
 _,stage,report=max(options,key=lambda x:x[0]);shutil.copy2(RUN/f'{stage}_best.pt',RUN/'best.pt');write_json(RUN/'best_validation.json',report);head=read(RUN/'head_history.json');full=[];old=read(RUN/'initial_history.json')
 if stage=='head':
  selection=read(RUN/'head_selection.json');name=selection['recipe']['name'];chosen=[r for r in head if r['recipe']==name];write_json(RUN/'training_history.json',old+chosen)
 else:write_json(RUN/'training_history.json',old+full)
 torch.set_num_threads(4);ck=torch.load(RUN/'best.pt',map_location='cpu',weights_only=True);network=model_from(ck,torch.device('cpu'));ds=fixed_split(ROOT/'data/private/fer2013-v1.zip',read(RUN/'fer_training_manifest.json'),'val');logits,y=predictions(network,ds,torch.device('cpu'));fresh=metrics(logits,y);write_json(RUN/'fresh_validation_preflight.json',fresh)
 if not eligible(fresh,read(RUN/'current_validation.json'),read(RUN/'earlier_validation.json')):
  write_json(RUN/'deployment_decision.json',{'eligible':False,'test_evaluated':False,'installed':False,'reason':'Fresh raw validation failed class/retention gate. No test or app/PPT update.'});return
 write_json(RUN/'candidate_frozen.json',{'best_sha256':digest(RUN/'best.pt'),'selected_stage':stage,'single_view_only':True,'test_accessed_at_freeze':False,'test_gate':read(RUN/'protocol.json')['test_gate']})
 cmd=[sys.executable,'-u','-m','scripts.colab_evaluate','--data','data/private/fer2013-v1.zip','--run-dir',str(RUN.relative_to(ROOT)),'--device','cpu','--threads','4','--single-view','--strict-all-metrics','--allow-equal-disgust-f1','--baseline-report',str(RUN.relative_to(ROOT)/'current_test.json'),'--baseline-manifest',str(RUN.relative_to(ROOT)/'current_test_manifest.json')]
 with (RUN/'test_evaluation.log').open('w',encoding='utf-8') as stream:subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 result=read(RUN/'deployment/reports/emotion_evaluation.json');current=read(RUN/'current_test.json');earlier=read(RUN/'earlier_test.json');passed=eligible(result,current,earlier);comparison=read(RUN/'deployment/reports/colab_comparison.json');assert not passed or comparison['recommended_for_integration'];write_json(RUN/'deployment_decision.json',{'eligible':passed,'test_evaluated':True,'installed':False,'selected_stage':stage,'checkpoint_sha256':result['checkpoint_sha256'],'reason':'All requested class recovery and disgust/aggregate retention conditions passed' if passed else 'One frozen candidate failed at least one requested full-test goal; no alternate candidate, app/PPT unchanged.','comparison':comparison,'class_recall':{name:{'current':current['classification_report'][name]['recall'],'earlier':earlier['classification_report'][name]['recall'],'candidate':result['classification_report'][name]['recall']} for name in ['happy','surprise','neutral','disgust']}})
 print(json.dumps(read(RUN/'deployment_decision.json'),indent=2),flush=True)
if __name__=='__main__':main()
