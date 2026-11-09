"""Train smaller native FusDenseNet heads on verified TRAIN representations."""
import argparse,copy,json,math,time,shutil
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from scripts.colab_common import digest,write_json,save_checkpoint,check_manifest,metrics,fixed_split
from scripts.compact_fusdense import CompactFusDenseNet,model_from,coarse_spatial_features,capacity_loss
from scripts.train_pruned_head import folded_head
from scripts.train_consistency import update_ema
from scripts.recall_recovery_common import eligible,objective,feasibility
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--threads',type=int,default=4);a=ap.parse_args();run=ROOT/a.run_dir
 if (run/'head_completion.json').exists() or (run/'candidate_frozen.json').exists():raise RuntimeError('Completed/frozen run preserved')
 protocol=read(run/'protocol.json');torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(protocol['seed']);ck=torch.load(run/'initial.pt',map_location='cpu',weights_only=True);assert digest(run/'initial.pt')==protocol['initial_checkpoint_sha256'];manifest=read(ROOT/'reports/fer_training_manifest.json');check_manifest(manifest);current=read(run/'current_validation.json');earlier=read(run/'earlier_validation.json');old=torch.load(ROOT/'reports/pure_orientation_20261108/pure_baseline.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v,ck['state_dict'][k]) for k,v in old['state_dict'].items() if not k.startswith('classifier.'));del old
 settings={**protocol,'architecture':{'version':9,'head_width':128},'dataset_sha256':digest(ROOT/'data/private/fer2013-v1.zip'),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'device':'cpu','threads':a.threads};write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',manifest)
 caches=[torch.load(ROOT/name,map_location='cpu',weights_only=True,mmap=True) for name in ['reports/pruned_head_20261107/train_features.pt','reports/pure_orientation_20261108/train_mirrored_features.pt','reports/pruned_head_20261107/val_features.pt']];raw,flip,val=caches;source=digest(ROOT/'reports/pure_orientation_20261108/pure_baseline.pt')
 for z in caches:assert z['source_sha256']==source and z['manifest_sha256']==settings['manifest_sha256']
 labels=torch.tensor([r['label'] for r in manifest['train']]);assert torch.equal(raw['labels'],labels) and torch.equal(flip['labels'],labels);assert val['labels'].tolist()==[r['label'] for r in manifest['val']]
 reference=torch.load(ROOT/'reports/pure_recall_recovery_v2_20261108/train_reference_logits.pt',map_location='cpu',weights_only=True);assert reference['source_checkpoint_sha256']==protocol['initial_checkpoint_sha256'] and reference['manifest_sha256']==settings['manifest_sha256'];assert torch.equal(reference['labels'],labels.repeat(2))
 original=model_from(ck,torch.device('cpu'));original.eval()
 with torch.no_grad():r=metrics(original.classifier(val['features']),val['labels'])
 assert r['accuracy']==current['accuracy'] and r['classification_report']['macro avg']['f1-score']==current['classification_report']['macro avg']['f1-score']
 candidate=CompactFusDenseNet(head_width=128);state=candidate.state_dict();state.update({k:v for k,v in ck['state_dict'].items() if not k.startswith('classifier.')});candidate.load_state_dict(state);candidate.eval();actual={};handle=candidate.classifier.register_forward_pre_hook(lambda module,args:actual.update(features=args[0].detach()))
 validation=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'val');images=torch.stack([validation[i][0] for i in range(64)])
 with torch.no_grad():candidate(images);expected=coarse_spatial_features(val['features'][:64]);error=float((actual['features']-expected).abs().max());assert torch.allclose(actual['features'],expected,atol=2e-4,rtol=2e-4)
 handle.remove();write_json(run/'cached_feature_preflight.json',{'validation_images':64,'maximum_absolute_feature_error':error,'coarse_feature_width':1260,'all_native_backbone_tensors_equal_to_active_model':True,'no_test_images_read':True});del original,candidate,validation,images
 x=torch.cat([coarse_spatial_features(raw['features']),coarse_spatial_features(flip['features'])]);y=labels.repeat(2);v=coarse_spatial_features(val['features']);vy=val['labels'];teacher=reference['logits'];del caches,raw,flip,val,reference
 mean=x.mean(0);raw_std=x.std(0);std=raw_std.clamp_min(protocol['feature_std_floor']);x.sub_(mean).div_(std);vn=(v-mean)/std;loader=DataLoader(TensorDataset(x,y,teacher),batch_size=512,shuffle=True);counts=torch.bincount(labels,minlength=7).float();base_weights=(counts.sum()/(7*counts)).sqrt();base_weights/=base_weights.mean();write_json(run/'normalization_audit.json',{'fit_on':'47006 original/mirrored TRAIN feature views from23503 source images','unique_train_source_images':len(labels),'feature_width':1260,'std_floor':protocol['feature_std_floor'],'features_below_std_floor':int((raw_std<protocol['feature_std_floor']).sum()),'class_weights':base_weights.tolist(),'original_parameter_count':sum(t.numel() for t in model_from(ck,torch.device('cpu')).parameters())})
 offset=max(r['epoch'] for r in read(run/'initial_history.json'));allrows=[];best=-float('inf');warm_best=-float('inf');selected=None
 def export(head,path,epoch,recipe,kind):
  folded=folded_head(head,mean,std);folded.eval()
  with torch.no_grad():lhs=head(vn);rhs=folded(v)
  assert float((lhs-rhs).abs().max())<2e-4 and torch.equal(lhs.argmax(1),rhs.argmax(1))
  network=CompactFusDenseNet(head_width=recipe['head_width']);state=network.state_dict();state.update({k:t for k,t in ck['state_dict'].items() if not k.startswith('classifier.')});network.load_state_dict(state);network.classifier.load_state_dict(folded.state_dict());arch={**ck['architecture'],'version':9,'head_width':recipe['head_width']};base={k:t for k,t in ck.items() if k not in ['state_dict','inference','calibration','colab_settings']};save_checkpoint(path,{**base,'state_dict':network.state_dict(),'architecture':arch,'inference':{},'epoch':epoch,'colab_settings':{**settings,'architecture':arch,'selected_recipe':recipe},'training_stage':'pure_capacity_coarse_classifier','training_reference':'Current pure classifier TRAIN correct disgust-vs-rest boundary; discarded at inference','head_refinement':True,'candidate_type':kind,'selection_metric':protocol['selection']})
 for number,recipe in enumerate(protocol['recipes']):
  torch.manual_seed(protocol['seed']+number);name=recipe['name'];folder=run/name;folder.mkdir(exist_ok=False);head=nn.Sequential(nn.Linear(1260,recipe['head_width']),nn.ReLU(),nn.Dropout(.5),nn.Linear(recipe['head_width'],7));ema=copy.deepcopy(head);ema.requires_grad_(False);weights=base_weights.clone();weights[[3,5,6]]*=recipe['common_boost'];optimizer=torch.optim.AdamW(head.parameters(),lr=protocol['head_lr'],weight_decay=protocol['head_weight_decay']);rows=[];diagnostic=-float('inf');stale=0;updates=0;began=time.perf_counter()
  for ep in range(protocol['head_epochs_max']):
   tick=time.perf_counter();head.train();total=0.;ce_total=0.;ret_total=0.;seen=0;correct=0
   for group in optimizer.param_groups:group['lr']=protocol['head_lr']*(.2+.8*.5*(1+math.cos(math.pi*ep/protocol['head_epochs_max'])))
   for xx,yy,tt in loader:
    optimizer.zero_grad(set_to_none=True);out=head(F.dropout(xx,p=protocol['feature_dropout'],training=True));det=head[3](head[1](head[0](xx)));loss,ce,ret=capacity_loss(out,det,yy,tt,weights,protocol['retention_weight'],protocol['label_smoothing']);assert bool(torch.isfinite(loss));loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step();update_ema(ema,head,updates);updates+=1;seen+=len(yy);correct+=int((det.argmax(1)==yy).sum());total+=float(loss.detach())*len(yy);ce_total+=float(ce.detach())*len(yy);ret_total+=float(ret.detach())*len(yy)
   options=[]
   for kind,module in [('student',head),('EMA',ema)]:
    module.eval()
    with torch.no_grad():r=metrics(module(vn),vy)
    ok=eligible(r,current,earlier);score=objective(r);feasible=feasibility(r,current,earlier);write_json(folder/f'validation_{ep+1:03d}_{kind}.json',r);options.append((feasible,r,kind,ok))
    if feasible>warm_best:export(module,run/'warm_start.pt',offset+ep+1,recipe,kind);write_json(run/'warm_validation.json',r);write_json(run/'warm_recipe.json',recipe);warm_best=feasible
    if ok and score>best:export(module,run/'head_best.pt',offset+ep+1,recipe,kind);write_json(run/'head_best_validation.json',r);selected={'recipe':recipe,'kind':kind,'epoch':offset+ep+1};write_json(run/'head_selection.json',selected);best=score
   feasible,r,kind,ok=max(options,key=lambda z:z[0]);stale=0 if feasible>diagnostic+1e-4 else stale+1;diagnostic=max(diagnostic,feasible);c=r['classification_report'];row={'epoch':offset+ep+1,'recipe_epoch':ep+1,'phase':'pure_capacity_head','recipe':name,'kind':kind,'loss':total/seen,'CE':ce_total/seen,'retention':ret_total/seen,'accuracy':correct/seen,'training_accuracy_kind':'Unmixed normalized TRAIN features, deterministic classifier','val_loss':r['negative_log_likelihood'],'val_accuracy':r['accuracy'],'val_macro_f1':c['macro avg']['f1-score'],'val_disgust_recall':c['disgust']['recall'],'val_disgust_f1':c['disgust']['f1-score'],'val_happy_recall':c['happy']['recall'],'val_surprise_recall':c['surprise']['recall'],'val_neutral_recall':c['neutral']['recall'],'feasibility_objective':feasible,'eligible':ok,'seconds':time.perf_counter()-tick};rows.append(row);allrows.append(row);write_json(folder/'rows.json',rows);write_json(run/'head_history.json',allrows);print(json.dumps(row),flush=True)
   if (len(rows)>=protocol['head_min_epochs'] and stale>=protocol['head_patience']) or time.perf_counter()-began>protocol['head_minutes_per_recipe']*60:break
  write_json(folder/'completion.json',{'completed_head_epochs':len(rows),'test_accessed':False})
 write_json(run/'head_completion.json',{'completed_head_epochs':len(allrows),'validation_improved':selected is not None,'selection':selected,'test_accessed':False,'external_encoder':False,'inference_hybrid':False})
if __name__=='__main__':main()
