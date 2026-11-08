"""Freeze at most one candidate across the two completed validation-only recipes."""
import json,shutil
from pathlib import Path
import torch
from torchvision.models import densenet121,DenseNet121_Weights
from backend.models import FusDenseNet
from scripts.colab_common import digest,write_json,save_checkpoint,fixed_split
from scripts.train_spatial_generalization import score,selected
ROOT=Path(__file__).resolve().parents[1];SPATIAL=ROOT/'reports/spatial_generalization_20261108';DENSE=ROOT/'reports/dense_transfer112_20261108';DIRECT=ROOT/'reports/direct_dense_head_long_20261108';OUT=ROOT/'reports/spatial_repair_selected_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 torch.set_num_threads(2)
 if (OUT/'selection.json').exists():raise RuntimeError('Selection already frozen')
 d1,d2,d3=read(SPATIAL/'completion.json'),read(DENSE/'completion.json'),read(DIRECT/'completion.json');baseline=read(SPATIAL/'initial_validation.json');candidates=[]
 for folder,done in [(SPATIAL,d1),(DENSE,d2),(DIRECT,d3)]:
  if done['validation_improved']:
   val=read(folder/'best_validation.json');assert selected(val,baseline);candidates.append({'run':str(folder.relative_to(ROOT)).replace('\\','/'),'score':score(val),'validation_accuracy':val['accuracy'],'validation_macro_f1':val['classification_report']['macro avg']['f1-score'],'validation_disgust_recall':val['classification_report']['disgust']['recall']})
 OUT.mkdir(exist_ok=True);chosen=max(candidates,key=lambda x:x['score']) if candidates else None
 if chosen:
  folder=ROOT/chosen['run']
  if folder in [DENSE,DIRECT]:
   head=torch.load(folder/'best_head.pt',map_location='cpu',weights_only=True);base=torch.load(SPATIAL/'baseline.pt',map_location='cpu',weights_only=True);architecture={**base['architecture'],'version':7 if folder==DIRECT else 6};architecture.update({'transfer_mix':head['transfer_mix'],'transfer_base_temperature':head['transfer_base_temperature']} if folder==DIRECT else {});model=FusDenseNet(**architecture);state=model.state_dict()
   for k,v in base['state_dict'].items():state[k]=v
   torch.hub.set_dir(str(ROOT/'external/torchvision_cache'));pre=densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
   for k,v in pre.features.state_dict().items():state['transfer_backbone.'+k]=v
   head=torch.load(folder/'best_head.pt',map_location='cpu',weights_only=True)
   for k,v in head['state_dict'].items():state['transfer_head.'+k]=v
   model.load_state_dict(state);model.eval()
   # Fresh composite inference must agree with cached validation computation.
   manifest=read(ROOT/'reports/fer_training_manifest.json');ds=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'val');x=torch.stack([ds[i][0] for i in range(16)])
   from scripts.colab_common import model_from
   original=model_from(base,torch.device('cpu'));original.eval();features=torch.load(DENSE/'val_features.pt',map_location='cpu',weights_only=True)['features'][:16]
   with torch.no_grad():
    core=original(x);extra=model.transfer_head(features);expected=(((1-head['transfer_mix'])*(core/head['transfer_base_temperature']).softmax(1)+head['transfer_mix']*extra.softmax(1)).clamp_min(1e-12).log() if folder==DIRECT else core+extra);actual=model(x)
   delta=float((expected-actual).abs().max());assert torch.allclose(expected,actual,atol=1e-4,rtol=1e-4) and torch.equal(expected.argmax(1),actual.argmax(1))
   write_json(OUT/'composite_contract.json',{'validation_examples':16,'max_logit_difference':delta,'predictions_identical':True,'not_an_accuracy_score':True})
   settings={**read(folder/'protocol.json'),'architecture':architecture,'dataset_sha256':digest(ROOT/'data/private/fer2013-v1.zip'),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'pretrained_provenance':read(DENSE/'pretrained_provenance.json'),'device':'cpu'}
   old_history=read(ROOT/'reports/training_history.json');offset=max(r['epoch'] for r in old_history);new_history=[{**r,'epoch':offset+r['epoch']} for r in read(folder/'training_history.json')];epoch=offset+head['epoch']
   save_checkpoint(OUT/'best.pt',{**{k:v for k,v in base.items() if k not in ['state_dict','architecture','inference','calibration','head_refinement']},'state_dict':model.state_dict(),'architecture':architecture,'inference':{},'epoch':epoch,'colab_settings':settings,'training_stage':'classifier_dense_probability' if folder==DIRECT else 'classifier_dense_transfer','head_refinement':False,'colab_improved':True,'selection_metric':settings['selection']})
   write_json(OUT/'settings.json',settings);write_json(OUT/'training_history.json',old_history+new_history)
  else:
   for name in ['best.pt','settings.json','training_history.json']:shutil.copy2(folder/name,OUT/name)
  for src,dst in [(folder/'best_validation.json','best_validation.json'),(ROOT/'reports/fer_training_manifest.json','fer_training_manifest.json'),(SPATIAL/'baseline_test.json','baseline_test.json'),(SPATIAL/'baseline_test_manifest.json','baseline_test_manifest.json')]:shutil.copy2(src,OUT/dst)
  write_json(OUT/'candidate_frozen.json',{'best_sha256':digest(OUT/'best.pt'),'selected_before_test':True,'at_most_one_test_candidate':True})
 write_json(OUT/'selection.json',{'chosen':chosen,'candidates':candidates,'all_recipes_completed':True,'test_accessed_at_selection':False,'rule':'Choose max validation .5accuracy+.3macroF1+.2disgustF1 among candidates whose accuracy and disgust recall improve, macro/disgustF1 not decline. No alternate test search.','disclosure':'Dense transfer and failed translation screen added after v5epoch2validation, before test. Reused validation; exploratory, not a fresh prospective benchmark.'})
 for folder in [SPATIAL,DENSE,DIRECT]:write_json(folder/'candidate_frozen.json',{'selection_directory':str(OUT.relative_to(ROOT)).replace('\\','/'),'all_training_finished':True,'training_locked':True,'selection_precedes_test':True})
 print(json.dumps({'chosen':chosen,'candidates':candidates},indent=2))
if __name__=='__main__':main()
