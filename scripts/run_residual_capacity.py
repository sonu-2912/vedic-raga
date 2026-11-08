"""Select and test at most one pure recovery candidate after all training stages."""
import json,subprocess,sys,time,shutil
from pathlib import Path
import torch
from scripts.colab_common import digest,write_json,save_checkpoint,fixed_split,predictions,metrics
from scripts.compact_residual_fusdense import model_from,initialize_from_compact
from scripts.recall_recovery_common import eligible,objective
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_residual_capacity_20261108'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 parent=ROOT/'reports/pure_capacity_20261108';bn=ROOT/'reports/pure_capacity_bn_20261108';deadline=time.monotonic()+160*60
 while not (bn/'deployment_decision.json').exists():
  if any((p/'candidate_frozen.json').exists() for p in [parent,bn]) or (bn/'skipped.json').exists():
   write_json(RUN/'skipped.json',{'reason':'An earlier candidate was already frozen/qualified/tested; no alternate search'});write_json(RUN/'completion.json',{'skipped':True});return
  if time.monotonic()>deadline:raise TimeoutError('Prior phase incomplete; no new candidate')
  time.sleep(10)
 previous=read(bn/'deployment_decision.json')
 if previous['test_evaluated'] or previous['eligible']:
  write_json(RUN/'skipped.json',{'reason':'Earlier candidate tested; no alternate search'});write_json(RUN/'completion.json',{'skipped':True});return
 protocol=read(RUN/'protocol.json');torch.set_num_threads(4);torch.manual_seed(protocol['seed']);warm=torch.load(parent/'warm_start.pt',map_location='cpu',weights_only=True);latest=torch.load(parent/'full_latest.pt',map_location='cpu',weights_only=True);full=read(parent/'full_history.json');kind=full[-1]['kind'];source={**warm,'state_dict':latest['ema_state'] if kind=='EMA' else latest['state_dict'],'epoch':full[-1]['epoch']};network,arch=initialize_from_compact(source);network.eval();original=model_from(source,torch.device('cpu')).eval();manifest=read(parent/'fer_training_manifest.json');validation=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'val');images=torch.stack([validation[i][0] for i in range(64)])
 with torch.no_grad():old_logits=original(images);new_logits=network(images);error=float((old_logits-new_logits).abs().max());assert torch.allclose(old_logits,new_logits,atol=2e-5,rtol=2e-5)
 write_json(RUN/'zero_initialization_preflight.json',{'validation_faces':64,'maximum_logit_error':error,'argmax_equal':bool(torch.equal(old_logits.argmax(1),new_logits.argmax(1))),'source_full_checkpoint_sha256':digest(parent/'full_latest.pt'),'source_kind':kind,'parameters':sum(p.numel() for p in network.parameters()),'test_images_read':False})
 protocol['epoch_offset']=full[-1]['epoch'];write_json(RUN/'protocol.json',protocol);settings={**protocol,'architecture':arch,'dataset_sha256':digest(ROOT/'data/private/fer2013-v1.zip'),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'device':'cpu','threads':4};write_json(RUN/'settings.json',settings);write_json(RUN/'fer_training_manifest.json',manifest);recipe=read(parent/'warm_recipe.json');history=[r for r in read(parent/'head_history.json') if r['recipe']==recipe['name'] and r['epoch']<=warm['epoch']]+full;write_json(RUN/'initial_history.json',history);write_json(RUN/'head_history.json',[]);write_json(RUN/'warm_recipe.json',{'name':'native_residual_capacity','head_width':arch['head_width'],'common_boost':1.,'source_compact_recipe':recipe['name']});write_json(RUN/'warm_validation.json',read(parent/f'full_validation_{len(full):02d}_{kind}.json'));base={k:v for k,v in source.items() if k not in ['state_dict','inference','calibration','colab_settings']};save_checkpoint(RUN/'warm_start.pt',{**base,'state_dict':network.state_dict(),'architecture':arch,'inference':{},'epoch':full[-1]['epoch'],'colab_settings':settings,'training_stage':'pure_residual_capacity_initialized','head_refinement':False});del network,original,latest,validation,images
 subprocess.Popen([sys.executable,'-m','scripts.keep_training_awake','--run-dir',str(RUN.relative_to(ROOT)),'--minutes','100'],cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
 with (RUN/'full_training.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-u','-m','scripts.train_residual_capacity_full','--run-dir',str(RUN.relative_to(ROOT)),'--threads','4'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 assert all(digest(ROOT/name)==sha for name,sha in read(RUN/'protected_artifacts.json').items())
 options=[]
 for name in ['head','full']:
  if (RUN/f'{name}_best.pt').exists():
   report=read(RUN/f'{name}_best_validation.json');assert eligible(report,read(RUN/'current_validation.json'),read(RUN/'earlier_validation.json'));options.append((objective(report),name,report))
 if not options:
  write_json(RUN/'deployment_decision.json',{'eligible':False,'test_evaluated':False,'installed':False,'reason':'No native residual compact validation checkpoint recovered all requested classes while retaining disgust and aggregate metrics. App/PPT unchanged.'});return
 _,stage,report=max(options,key=lambda x:x[0]);shutil.copy2(RUN/f'{stage}_best.pt',RUN/'best.pt');write_json(RUN/'best_validation.json',report);head=read(RUN/'head_history.json');full=read(RUN/'full_history.json');old=read(RUN/'initial_history.json')
 if stage=='head':
  selection=read(RUN/'head_selection.json');name=selection['recipe']['name'];chosen=[r for r in head if r['recipe']==name];write_json(RUN/'training_history.json',old+chosen)
 else:write_json(RUN/'training_history.json',old+full)
 torch.set_num_threads(4);ck=torch.load(RUN/'best.pt',map_location='cpu',weights_only=True);network=model_from(ck,torch.device('cpu'));ds=fixed_split(ROOT/'data/private/fer2013-v1.zip',read(RUN/'fer_training_manifest.json'),'val');logits,y=predictions(network,ds,torch.device('cpu'));fresh=metrics(logits,y);write_json(RUN/'fresh_validation_preflight.json',fresh)
 if not eligible(fresh,read(RUN/'current_validation.json'),read(RUN/'earlier_validation.json')):
  write_json(RUN/'deployment_decision.json',{'eligible':False,'test_evaluated':False,'installed':False,'reason':'Fresh raw validation failed class/retention gate. No test or app/PPT update.'});return
 write_json(RUN/'candidate_frozen.json',{'best_sha256':digest(RUN/'best.pt'),'selected_stage':stage,'single_view_only':True,'test_accessed_at_freeze':False,'test_gate':read(RUN/'protocol.json')['test_gate']})
 cmd=[sys.executable,'-u','-m','scripts.evaluate_residual_capacity','--data','data/private/fer2013-v1.zip','--run-dir',str(RUN.relative_to(ROOT)),'--device','cpu','--threads','4','--single-view','--strict-all-metrics','--allow-equal-disgust-f1','--baseline-report',str(RUN.relative_to(ROOT)/'current_test.json'),'--baseline-manifest',str(RUN.relative_to(ROOT)/'current_test_manifest.json')]
 with (RUN/'test_evaluation.log').open('w',encoding='utf-8') as stream:subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 result=read(RUN/'deployment/reports/emotion_evaluation.json');current=read(RUN/'current_test.json');earlier=read(RUN/'earlier_test.json');passed=eligible(result,current,earlier);comparison=read(RUN/'deployment/reports/colab_comparison.json');assert not passed or comparison['recommended_for_integration'];write_json(RUN/'deployment_decision.json',{'eligible':passed,'test_evaluated':True,'installed':False,'selected_stage':stage,'checkpoint_sha256':result['checkpoint_sha256'],'reason':'All requested class recovery and disgust/aggregate retention conditions passed' if passed else 'One frozen candidate failed at least one requested full-test goal; no alternate candidate, app/PPT unchanged.','comparison':comparison,'class_recall':{name:{'current':current['classification_report'][name]['recall'],'earlier':earlier['classification_report'][name]['recall'],'candidate':result['classification_report'][name]['recall']} for name in ['happy','surprise','neutral','disgust']}})
 print(json.dumps(read(RUN/'deployment_decision.json'),indent=2),flush=True)
if __name__=='__main__':main()
