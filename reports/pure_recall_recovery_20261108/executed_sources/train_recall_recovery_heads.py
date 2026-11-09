"""Train the native pure FusDense classifier to recover common classes and retain disgust."""
import argparse,copy,json,math,time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scripts.colab_common import digest,write_json,save_checkpoint,model_from,metrics,check_manifest
from scripts.train_pruned_head import folded_head
from scripts.train_consistency import update_ema
from scripts.recall_recovery_common import eligible,objective,feasibility,recovery_loss
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--threads',type=int,default=4);a=ap.parse_args();run=ROOT/a.run_dir
 if (run/'head_completion.json').exists() or (run/'candidate_frozen.json').exists():raise RuntimeError('Existing completed/frozen run preserved')
 protocol=read(run/'protocol.json');torch.set_num_threads(a.threads);torch.manual_seed(protocol['seed']);ck=torch.load(run/'initial.pt',map_location='cpu',weights_only=True);assert digest(run/'initial.pt')==protocol['initial_checkpoint_sha256'];old=torch.load(ROOT/'reports/pure_orientation_20261108/pure_baseline.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v,ck['state_dict'][k]) for k,v in old['state_dict'].items() if not k.startswith('classifier.'));manifest=read(ROOT/'reports/fer_training_manifest.json');check_manifest(manifest);settings={**protocol,'architecture':ck['architecture'],'dataset_sha256':digest(ROOT/'data/private/fer2013-v1.zip'),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'device':'cpu','threads':a.threads};write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',manifest)
 raw=torch.load(ROOT/'reports/pruned_head_20261107/train_features.pt',map_location='cpu',weights_only=True,mmap=True);flip=torch.load(ROOT/'reports/pure_orientation_20261108/train_mirrored_features.pt',map_location='cpu',weights_only=True,mmap=True);val=torch.load(ROOT/'reports/pruned_head_20261107/val_features.pt',map_location='cpu',weights_only=True,mmap=True)
 source=digest(ROOT/'reports/pure_orientation_20261108/pure_baseline.pt')
 for cached in [raw,flip,val]:assert cached['source_sha256']==source and cached['manifest_sha256']==settings['manifest_sha256']
 labels=torch.tensor([r['label'] for r in manifest['train']]);assert torch.equal(raw['labels'],labels) and torch.equal(flip['labels'],labels);assert val['labels'].tolist()==[r['label'] for r in manifest['val']]
 network=model_from(ck,torch.device('cpu'));network.eval();reference=copy.deepcopy(network.classifier);reference.eval();reference.requires_grad_(False);x=torch.cat([raw['features'],flip['features']]);y=torch.cat([labels,labels]);mirrored=torch.arange(len(y))>=len(labels);v,vy=val['features'],val['labels'];current=read(run/'current_validation.json');earlier=read(run/'earlier_validation.json')
 with torch.no_grad():baseline=metrics(reference(v),vy)
 assert baseline['accuracy']==current['accuracy'] and baseline['classification_report']['macro avg']['f1-score']==current['classification_report']['macro avg']['f1-score']
 teacher=[]
 with torch.no_grad():
  for start in range(0,len(x),1024):teacher.append(reference(x[start:start+1024]))
 teacher=torch.cat(teacher);save_checkpoint(run/'train_reference_logits.pt',{'source_checkpoint_sha256':digest(run/'initial.pt'),'manifest_sha256':settings['manifest_sha256'],'logits':teacher,'labels':y});mean=x.mean(0);std=x.std(0).clamp_min(.01);base_weights=torch.tensor(read(ROOT/'reports/pure_orientation_20261108/normalization_audit.json')['class_weights']);x.sub_(mean).div_(std);vn=(v-mean)/std;loader=DataLoader(TensorDataset(x,y,teacher,mirrored),batch_size=512,shuffle=True);old_history=read(run/'initial_history.json');offset=max(r['epoch'] for r in old_history);best=float('-inf');warm_best=float('-inf');allrows=[];selected=None;warm_recipe=None
 def export(head,path,epoch,recipe,kind):
  folded=folded_head(head,mean,std);folded.eval()
  with torch.no_grad():z=head(vn);r=folded(v)
  assert float((z-r).abs().max())<2e-4 and torch.equal(z.argmax(1),r.argmax(1))
  candidate=model_from(ck,torch.device('cpu'));candidate.classifier.load_state_dict(folded.state_dict());base={k:value for k,value in ck.items() if k not in ['state_dict','inference','calibration','colab_settings']};save_checkpoint(path,{**base,'state_dict':candidate.state_dict(),'inference':{},'epoch':epoch,'colab_settings':{**settings,'selected_recipe':recipe},'training_stage':'classifier_recall_recovery','training_reference':'Same current pure classifier, TRAIN disgust only','candidate_type':kind,'colab_improved':False,'head_refinement':True,'selection_metric':protocol['selection']})
 for number,recipe in enumerate(protocol['recipes']):
  name=recipe['name'];folder=run/name;folder.mkdir(exist_ok=False);torch.manual_seed(protocol['seed']+number);head=copy.deepcopy(reference);head.requires_grad_(True);head[2].p=.5
  with torch.no_grad():w=reference[0].weight.clone();head[0].weight.copy_(w*std);head[0].bias.copy_(reference[0].bias+w@mean)
  head.eval()
  with torch.no_grad():assert float((head(vn)-reference(v)).abs().max())<2e-4
  ema=copy.deepcopy(head);ema.requires_grad_(False);weights=base_weights.clone();weights[[3,5,6]]*=recipe['common_boost'];optimizer=torch.optim.AdamW(head.parameters(),lr=protocol['head_lr'],weight_decay=.01);rows=[];stale=0;diagnostic=float('-inf');began=time.perf_counter();updates=0
  for ep in range(protocol['head_epochs_max']):
   tick=time.perf_counter();head.train();totals={'loss':0.,'CE':0.,'retention':0.};seen=0;correct=0
   for group in optimizer.param_groups:group['lr']=protocol['head_lr']*(.2+.8*.5*(1+math.cos(math.pi*ep/protocol['head_epochs_max'])))
   for xx,yy,tt,mirror in loader:
    optimizer.zero_grad(set_to_none=True);out=head(xx);loss,ce,retain=recovery_loss(out,yy,tt,weights,mirror,protocol['mirror_common_weight'],recipe['retention_weight']);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step();update_ema(ema,head,updates);updates+=1;seen+=len(yy);correct+=int((out.argmax(1)==yy).sum())
    for k,value in [('loss',loss),('CE',ce),('retention',retain)]:totals[k]+=float(value.detach())*len(yy)
   options=[]
   for kind,module in [('student',head),('EMA',ema)]:
    module.eval()
    with torch.no_grad():r=metrics(module(vn),vy)
    score=objective(r);feasible=feasibility(r,current,earlier);ok=eligible(r,current,earlier);options.append((feasible,r,kind,module,ok));write_json(folder/f'validation_{ep+1:02d}_{kind}.json',r)
    if feasible>warm_best:
     export(module,run/'warm_start.pt',offset+ep+1,recipe,kind);write_json(run/'warm_validation.json',r);warm_best=feasible;warm_recipe=recipe;write_json(run/'warm_recipe.json',recipe)
    if ok and score>best:
     export(module,run/'head_best.pt',offset+ep+1,recipe,kind);write_json(run/'head_best_validation.json',r);best=score;selected={'recipe':recipe,'kind':kind,'epoch':offset+ep+1,'score':score};write_json(run/'head_selection.json',selected)
   chosen=max(options,key=lambda o:o[0]);feasible,r,kind,module,ok=chosen;stale=0 if feasible>diagnostic+1e-4 else stale+1;diagnostic=max(diagnostic,feasible);c=r['classification_report'];row={'epoch':offset+ep+1,'phase':'head_recall_'+name,'recipe':name,'kind':kind,**{k:value/seen for k,value in totals.items()},'accuracy':correct/seen,'val_loss':r['negative_log_likelihood'],'val_accuracy':r['accuracy'],'val_macro_f1':c['macro avg']['f1-score'],'val_disgust_recall':c['disgust']['recall'],'val_disgust_f1':c['disgust']['f1-score'],'val_happy_recall':c['happy']['recall'],'val_surprise_recall':c['surprise']['recall'],'val_neutral_recall':c['neutral']['recall'],'eligible':ok,'feasibility_objective':feasible,'seconds':time.perf_counter()-tick};rows.append(row);allrows.append(row);write_json(folder/'rows.json',rows);write_json(run/'head_history.json',allrows);print(json.dumps(row),flush=True)
   if (len(rows)>=protocol['head_min_epochs'] and stale>=protocol['head_patience']) or time.perf_counter()-began>protocol['head_minutes_per_recipe']*60:break
  write_json(folder/'completion.json',{'completed_head_epochs':len(rows),'test_accessed':False})
 write_json(run/'head_completion.json',{'completed_head_epochs':len(allrows),'validation_improved':selected is not None,'selection':selected,'warm_recipe':warm_recipe,'test_accessed':False,'external_encoder':False,'training_reference':'Current pure native classifier on TRAIN disgust examples only','inference_hybrid':False})
if __name__=='__main__':main()
