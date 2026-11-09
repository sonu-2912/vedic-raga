"""Conditional pure-FusDenseNet orientation repair, no external encoder or teacher.

Run only if the coverage repair finishes with NO test candidate. Training pairs
raw/mirrored features from the original pure network; test inference is single view.
"""
import argparse,copy,json,math,shutil,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset,TensorDataset
from backend.models import FusDenseNet
from scripts.colab_common import digest,write_json,save_checkpoint,check_manifest,fixed_split,model_from,metrics,predictions
from scripts.train_pruned_head import rank_neurons,folded_head
from scripts.train_pure_fusdense import assert_pure,selection_score
from scripts.train_pure_coverage_repair import eligible
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def paired_training_features(raw,flipped,expected_labels):
 if raw['source_sha256']!=flipped['source_sha256'] or raw['manifest_sha256']!=flipped['manifest_sha256']:raise ValueError('Raw/flip provenance differs')
 labels=torch.as_tensor(expected_labels,dtype=torch.long)
 if not torch.equal(raw['labels'],labels) or not torch.equal(flipped['labels'],labels):raise ValueError('Training-view label order differs from fixed TRAIN manifest')
 if raw['features'].shape!=flipped['features'].shape or len(raw['features'])!=len(labels):raise ValueError('Training-view dimensions differ')
 return torch.cat([raw['features'],flipped['features']]),torch.cat([labels,labels])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--threads',type=int,default=4);args=ap.parse_args();run=ROOT/args.run_dir
 previous=read(ROOT/'reports/pure_repair_20261108/deployment_decision.json')
 if previous['test_evaluated']:raise RuntimeError('No alternate training/test search after a frozen candidate was tested')
 if (run/'candidate_frozen.json').exists() or (run/'completion.json').exists():raise RuntimeError('Finished orientation run preserved')
 torch.set_num_threads(args.threads);protocol=read(run/'protocol.json');torch.manual_seed(protocol['seed']);ck=torch.load(run/'pure_baseline.pt',map_location='cpu',weights_only=True);assert_pure(ck);source=digest(run/'pure_baseline.pt');assert source==protocol['initial_checkpoint_sha256'];manifest=read(ROOT/'reports/fer_training_manifest.json');check_manifest(manifest);manifest_sha=digest(ROOT/'reports/fer_training_manifest.json');settings={**protocol,'dataset_sha256':digest(ROOT/'data/private/fer2013-v1.zip'),'manifest_sha256':manifest_sha,'device':'cpu','threads':args.threads};write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',manifest)
 raw=torch.load(ROOT/'reports/pruned_head_20261107/train_features.pt',map_location='cpu',weights_only=True,mmap=True);validation=torch.load(ROOT/'reports/pruned_head_20261107/val_features.pt',map_location='cpu',weights_only=True,mmap=True)
 for cached in [raw,validation]:assert cached['source_sha256']==source and cached['manifest_sha256']==manifest_sha
 assert validation['labels'].tolist()==[r['label'] for r in manifest['val']]
 network=model_from(ck,torch.device('cpu'));network.eval();v,vy=validation['features'],validation['labels']
 with torch.no_grad():initial=metrics(network.classifier(v),vy)
 write_json(run/'initial_validation.json',initial)
 flip_file=run/'train_mirrored_features.pt'
 if flip_file.exists():flipped=torch.load(flip_file,map_location='cpu',weights_only=True,mmap=True)
 else:
  if shutil.disk_usage(run).free<1200*1024**2:raise RuntimeError('Need at least1.2GB free before private feature caching')
  ds=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'train');captured=[];ys=[];partial=run/'train_mirrored_features.partial.pt';done=0
  if partial.exists():
   cache=torch.load(partial,map_location='cpu',weights_only=True);assert cache['source_sha256']==source and cache['manifest_sha256']==manifest_sha;captured=[cache['features']];ys=[cache['labels']];done=len(cache['labels'])
  handle=network.classifier.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach().cpu()))
  with torch.no_grad():
   for n,(x,y) in enumerate(DataLoader(Subset(ds,range(done,len(ds))),batch_size=128)):
    network(x.flip(-1));ys.append(y)
    if (n+1)%50==0:
     save_checkpoint(partial,{'source_sha256':source,'manifest_sha256':manifest_sha,'features':torch.cat(captured),'labels':torch.cat(ys)});print(json.dumps({'cache':'TRAIN-only mirrored pure features','faces':sum(len(y) for y in ys),'total':len(ds)}),flush=True)
  handle.remove();save_checkpoint(flip_file,{'source_sha256':source,'manifest_sha256':manifest_sha,'features':torch.cat(captured),'labels':torch.cat(ys)});del captured,ys,ds;flipped=torch.load(flip_file,map_location='cpu',weights_only=True,mmap=True)
 x,y=paired_training_features(raw,flipped,[r['label'] for r in manifest['train']]);mean=x.mean(0);std=x.std(0).clamp_min(.01);reference_head=copy.deepcopy(network.classifier);reference_head.eval();old_history=read(run/'initial_history.json');offset=max(r['epoch'] for r in old_history);counts=torch.bincount(y,minlength=7).float();weights=(counts.sum()/(7*counts)).sqrt();weights/=weights.mean();write_json(run/'normalization_audit.json',{'source_sha256':source,'training_views':len(y),'unique_source_faces':len(manifest['train']),'train_only_fit':True,'class_weights':weights.tolist(),'normalized_CE_label_mass':(counts*weights/(counts*weights).sum()).tolist(),'external_encoder':False,'teacher':False,'note':'Initialization/ranking copies only this same FusDense classifier. No teacher logits, distillation or external network.'})
 all_rows=[];options=[]
 # Rankings use original TRAIN activations, never validation labels. Larger head is unpruned.
 selections={512:(torch.arange(512),reference_head[3].bias.detach().clone())}
 for width in protocol['widths']:
  if width!=512:selections[width]=rank_neurons(reference_head,x,y,width)
 x.sub_(mean).div_(std);vn=(v-mean)/std;del raw,flipped;loader=DataLoader(TensorDataset(x,y),batch_size=512,shuffle=True)
 for width in protocol['widths']:
  folder=run/f'head{width}';folder.mkdir(exist_ok=True)
  if (folder/'completion.json').exists():
   complete=read(folder/'completion.json');rows=read(folder/'rows.json');all_rows.extend(rows)
   if complete['validation_improved']:options.append({'folder':folder,'report':read(folder/'best_validation.json')})
   continue
  torch.manual_seed(protocol['seed']+width);indices,bias=selections[width];head=nn.Sequential(nn.Linear(3276,width),nn.ReLU(),nn.Dropout(.5),nn.Linear(width,7))
  with torch.no_grad():
   w=reference_head[0].weight[indices];head[0].weight.copy_(w*std);head[0].bias.copy_(reference_head[0].bias[indices]+w@mean);head[3].weight.copy_(reference_head[3].weight[:,indices]);head[3].bias.copy_(bias)
  head.eval();unfolded=folded_head(head,mean,std)
  with torch.no_grad():fold_error=float((head(vn)-unfolded(v)).abs().max())
  assert fold_error<2e-4
  write_json(folder/'initialization_audit.json',{'width':width,'train_only_selected_neuron_indices':indices.tolist(),'initial_fold_max_error':fold_error,'inference_backbone_frozen_pure_source':source,'no_teacher_loss':True})
  optimizer=torch.optim.AdamW(head.parameters(),lr=protocol['learning_rate'],weight_decay=.01);rows=[];best=selection_score(initial);diagnostic_best=float('-inf');stale=0;improved=False;began=time.perf_counter()
  for epoch in range(protocol['epochs_max']):
   tick=time.perf_counter();head.train();loss_total=0.;correct=0;seen=0
   for group in optimizer.param_groups:group['lr']=protocol['learning_rate']*(.1+.9*.5*(1+math.cos(math.pi*epoch/protocol['epochs_max'])))
   for xx,yy in loader:
    optimizer.zero_grad(set_to_none=True);logits=head(xx);loss=F.cross_entropy(logits,yy,weight=weights)
    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite orientation-head loss')
    loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step();loss_total+=float(loss.detach())*len(yy);correct+=int((logits.argmax(1)==yy).sum());seen+=len(yy)
   head.eval()
   with torch.no_grad():out=head(vn)
   r=metrics(out,vy);score=selection_score(r);qualifies=eligible(r,initial,protocol['validation_min_disgust_recall_gain']);selected=qualifies and score>best
   if selected:
    folded=folded_head(head,mean,std)
    with torch.no_grad():restored=folded(v)
    assert torch.equal(restored.argmax(1),out.argmax(1)) and float((restored-out).abs().max())<2e-4
    architecture={**ck['architecture'],'head_width':width};candidate=FusDenseNet(**architecture);state=candidate.state_dict()
    for name,value in ck['state_dict'].items():
     if not name.startswith('classifier.'):state[name]=value
    candidate.load_state_dict(state);candidate.classifier.load_state_dict(folded.state_dict());assert all(torch.equal(value,candidate.state_dict()[name]) for name,value in ck['state_dict'].items() if not name.startswith('classifier.'))
    base={k:v for k,v in ck.items() if k not in ['state_dict','architecture','inference','calibration','head_refinement']};save_checkpoint(folder/'best.pt',{**base,'state_dict':candidate.state_dict(),'architecture':architecture,'inference':{},'epoch':offset+epoch+1,'colab_settings':{**settings,'architecture':architecture},'colab_improved':True,'head_refinement':True,'training_stage':'classifier_orientation_repair','candidate_type':f'paired_views_head{width}','selection_metric':protocol['selection']});write_json(folder/'best_validation.json',r);best=score;improved=True
   current=score;stale=0 if current>diagnostic_best+1e-4 else stale+1;diagnostic_best=max(diagnostic_best,current);c=r['classification_report'];row={'epoch':offset+epoch+1,'phase':f'head_refinement_orientation{width}','loss':loss_total/seen,'accuracy':correct/seen,'val_loss':r['negative_log_likelihood'],'val_accuracy':r['accuracy'],'val_macro_f1':c['macro avg']['f1-score'],'val_disgust_recall':c['disgust']['recall'],'val_disgust_f1':c['disgust']['f1-score'],'saved_new_best':selected,'seconds':time.perf_counter()-tick,'train_views_seen':seen,'unique_source_faces':len(manifest['train'])};rows.append(row);write_json(folder/f'validation_{epoch+1:02d}.json',r);write_json(folder/'rows.json',rows);print(json.dumps(row),flush=True)
   if (len(rows)>=protocol['min_epochs'] and stale>=protocol['patience']) or time.perf_counter()-began>=protocol['max_minutes_per_head']*60:break
  write_json(folder/'completion.json',{'completed_head_epochs':len(rows),'validation_improved':improved,'backbone_unchanged':True,'test_accessed':False});all_rows.extend(rows)
  if improved:options.append({'folder':folder,'report':read(folder/'best_validation.json')})
 write_json(run/'all_head_history.json',all_rows);chosen=max(options,key=lambda o:selection_score(o['report'])) if options else None
 if chosen:
  for name in ['best.pt','best_validation.json']:shutil.copy2(chosen['folder']/name,run/name)
  rows=read(chosen['folder']/'rows.json');write_json(run/'training_history.json',old_history+rows);write_json(run/'settings.json',{**settings,'selected_head':chosen['folder'].name})
 write_json(run/'completion.json',{'completed_additional_epochs':len(all_rows),'completed_head_epochs':len(all_rows),'completed_full_epochs':0,'validation_improved':bool(chosen),'selected_head':chosen['folder'].name if chosen else None,'test_accessed':False,'external_encoder':False,'teacher':False,'synthetic':False,'source_backbone_unchanged':True,'reason':'Native FusDense classifier learned raw/mirror TRAIN representations; all inference features remain from the same pure network. Prior independent coverage run was not used as its backbone.'})
if __name__=='__main__':main()
