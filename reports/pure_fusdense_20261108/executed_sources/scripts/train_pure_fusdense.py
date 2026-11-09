"""Full pure FusDenseNet training: class-paired SupCon and shared-feature R-Drop.

No hybrid/external encoder or teacher. Projection is TRAIN-only, not exported.
"""
import argparse,copy,json,math,random,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Sampler
from torchvision import transforms
from scripts.colab_common import digest,write_json,save_checkpoint,check_manifest,fixed_split,model_from,predictions,metrics,set_training_mode
from scripts.train_consistency import update_ema
ROOT=Path(__file__).resolve().parents[1]

class PairedClassBatches(Sampler):
 def __init__(self,labels,batch_size=64,per_class=2,batches=None):
  self.labels=torch.as_tensor(labels,dtype=torch.long);self.batch_size=batch_size;self.per_class=per_class;self.batches=batches or math.ceil(len(labels)/batch_size)
  self.pools=[torch.where(self.labels==c)[0] for c in range(7)];self.natural=batch_size-7*per_class
  if self.natural<0 or len(self.labels)<batch_size or any(len(p)<self.natural+per_class for p in self.pools):raise ValueError('Need enough unique training faces per class and batchsize >=7*per_class')
 def __len__(self):return self.batches
 def __iter__(self):
  for _ in range(self.batches):
   chosen=torch.randperm(len(self.labels))[:self.natural].tolist();used=set(chosen)
   for pool in self.pools:
    available=[int(i) for i in pool[torch.randperm(len(pool))].tolist() if i not in used]
    extras=available[:self.per_class];chosen.extend(extras);used.update(extras)
   yield [chosen[i] for i in torch.randperm(self.batch_size).tolist()]

def class_balanced_supcon(embedding,labels,temperature=.1):
 z=F.normalize(embedding,dim=1);logits=z@z.T/temperature;diagonal=torch.eye(len(z),dtype=torch.bool,device=z.device);positive=labels[:,None].eq(labels[None,:])&~diagonal
 valid=positive.any(1)
 if not bool(valid.any()):return embedding.sum()*0
 logits=logits-logits.max(1,keepdim=True).values.detach();logp=logits-logits.masked_fill(diagonal,float('-inf')).logsumexp(1,keepdim=True)
 per_anchor=-logp.masked_fill(~positive,0).sum(1)/positive.sum(1).clamp_min(1)
 return torch.stack([per_anchor[(labels==c)&valid].mean() for c in labels[valid].unique()]).mean()

def symmetric_kl(first,second):
 a,b=F.log_softmax(first,1),F.log_softmax(second,1)
 return .5*(F.kl_div(a,b.exp(),reduction='batchmean')+F.kl_div(b,a.exp(),reduction='batchmean'))

def selection_score(r):
 c=r['classification_report'];return .4*r['accuracy']+.4*c['macro avg']['f1-score']+.2*c['disgust']['f1-score']

def qualifies(r,base):
 c,b=r['classification_report'],base['classification_report']
 return r['accuracy']>base['accuracy'] and c['macro avg']['f1-score']>=b['macro avg']['f1-score'] and c['disgust']['recall']>b['disgust']['recall'] and c['disgust']['f1-score']>=b['disgust']['f1-score']

def assert_pure(ck):
 if ck['architecture'].get('version')!=4 or any(k.startswith(('transfer_','spatial_refine')) for k in ck['state_dict']):raise ValueError('Requires original purev4 FusDenseNet, no transfer backbone or hybrid checkpoint')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--recover',action='store_true');a=ap.parse_args();run=ROOT/a.run_dir
 if (run/'candidate_frozen.json').exists() or (run/'bundle_manifest.json').exists():raise RuntimeError('Frozen/evaluated experiment preserved; use a new declared run.')
 protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'));torch.set_num_threads(a.threads);torch.manual_seed(1108);random.seed(1108)
 ck=torch.load(run/'pure_baseline.pt',map_location='cpu',weights_only=True);assert_pure(ck);assert digest(run/'pure_baseline.pt')==protocol['initial_checkpoint_sha256']
 manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text(encoding='utf-8'));check_manifest(manifest);settings={**protocol,'architecture':ck['architecture'],'dataset_sha256':digest(ROOT/'data/private/fer2013-v1.zip'),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'seed':1108,'threads':a.threads,'device':'cpu'}
 write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',manifest)
 train=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'train');val=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'val');student=model_from(ck,torch.device('cpu'));ema=copy.deepcopy(student);ema.requires_grad_(False)
 logits,truth=predictions(student,val,torch.device('cpu'));initial=metrics(logits,truth);write_json(run/'initial_validation.json',initial)
 projection=nn.Sequential(nn.LayerNorm(588),nn.Linear(588,128),nn.ReLU(),nn.Linear(128,128));capture={}
 def hook(module,args):capture['features']=args[0]
 handle=student.classifier.register_forward_pre_hook(hook)
 for m in student.modules():
  if isinstance(m,nn.Dropout):m.p=protocol['dropout']
 head=list(student.classifier.parameters());head_ids={id(p) for p in head};backbone=[p for p in student.parameters() if id(p) not in head_ids]
 groups=[{'params':backbone,'lr':protocol['backbone_lr'],'weight_decay':.001},{'params':head,'lr':protocol['classifier_lr'],'weight_decay':.01},{'params':projection.parameters(),'lr':protocol['projection_lr'],'weight_decay':.001}];optimizer=torch.optim.AdamW(groups)
 rates=[g['lr'] for g in groups]
 labels=torch.tensor([y for _,y in train.rows]);counts=torch.bincount(labels,minlength=7).float();p=counts/counts.sum();q=(50*p+2)/64;importance=(p/q).sqrt();importance/=torch.dot(q,importance)
 sampler=PairedClassBatches(labels);loader=DataLoader(train,batch_sampler=sampler,num_workers=0)
 train.transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomAffine(5,translate=(.03,.03),scale=(.95,1.05),interpolation=transforms.InterpolationMode.BILINEAR,fill=128),transforms.ColorJitter(brightness=.12,contrast=.18),transforms.ToTensor()])
 old_history=json.loads((run/'initial_history.json').read_text(encoding='utf-8'));offset=max([r['epoch'] for r in old_history],default=ck['epoch']);rows=[];start=0;updates=0;best=selection_score(initial);diagnostic_best=float('-inf');stale=0;improved=False
 if a.recover:
  latest=torch.load(run/'latest.pt',map_location='cpu',weights_only=True);assert latest['settings']==settings
  student.load_state_dict(latest['state_dict']);ema.load_state_dict(latest['ema_state']);projection.load_state_dict(latest['projection_state']);optimizer.load_state_dict(latest['optimizer']);rows=latest['history'];start=len(rows);updates=latest['updates'];best=latest['best'];diagnostic_best=latest['diagnostic_best'];stale=latest['stale'];improved=latest['improved'];torch.set_rng_state(latest['torch_rng']);random.setstate(latest['python_rng'])
 elif (run/'latest.pt').exists():raise RuntimeError('Use recover or new run')
 audit={'architecture':ck['architecture'],'inference_parameters':sum(p.numel() for p in student.parameters()),'train_only_projection_parameters':sum(p.numel() for p in projection.parameters()),'counts':counts.int().tolist(),'sampling_prior':q.tolist(),'CE_importance':importance.tolist(),'source_sha256':digest(run/'pure_baseline.pt'),'external_encoder':False,'teacher':False,'logit_offsets_at_inference':False,'test_opened':False};write_json(run/'initialization_audit.json',audit);print(json.dumps(audit),flush=True)
 began=time.perf_counter()
 for ep in range(start,protocol['epochs_max']):
  t=time.perf_counter();cpu=time.process_time();set_training_mode(student,True);projection.train();totals={'CE':0.,'RDrop':0.,'SupCon':0.,'loss':0.};seen=0;correct=0;draws=torch.zeros(7,dtype=torch.long)
  for batch,(x,y) in enumerate(loader):
   warmup=min(1,(updates+1)/len(loader));decay=.1+.9*.5*(1+math.cos(math.pi*ep/protocol['epochs_max']))
   for group,lr in zip(optimizer.param_groups,rates):group['lr']=lr*warmup*decay
   optimizer.zero_grad(set_to_none=True);one=student(x);features=capture['features'];two=student.classifier(features)
   ce=(.5*(F.cross_entropy(one,y,reduction='none',label_smoothing=.01)+F.cross_entropy(two,y,reduction='none',label_smoothing=.01))*importance[y]).mean();rd=symmetric_kl(one,two);sc=class_balanced_supcon(projection(features[:,:588]),y,protocol['contrastive_temperature']);loss=ce+protocol['rdrop_weight']*rd+protocol['contrastive_weight']*sc
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; latest completed epoch remains preserved')
   loss.backward();nn.utils.clip_grad_norm_(backbone,2.);nn.utils.clip_grad_norm_(head,5.);nn.utils.clip_grad_norm_(projection.parameters(),5.);optimizer.step();update_ema(ema,student,updates);updates+=1
   for key,value in [('CE',ce),('RDrop',rd),('SupCon',sc),('loss',loss)]:totals[key]+=float(value.detach())*len(y)
   seen+=len(y);correct+=int(((one+two).argmax(1)==y).sum());draws+=torch.bincount(y,minlength=7)
   if (batch+1)%75==0:print(json.dumps({'epoch':offset+ep+1,'batch':batch+1,'batches':len(loader),'loss':totals['loss']/seen,'disgust_draws':int(draws[1])}),flush=True)
  options=[];chosen=None;epoch_improved=False
  for kind,model in [('student',student),('EMA',ema)]:
   out,truth=predictions(model,val,torch.device('cpu'));r=metrics(out,truth);write_json(run/f'validation_{ep+1:02d}_{kind}.json',r);item={'kind':kind,'report':r,'score':selection_score(r)};options.append(item)
   if qualifies(r,initial) and item['score']>best:
    best=item['score'];improved=True;epoch_improved=True;chosen=item;base={k:v for k,v in ck.items() if k not in ['state_dict','inference','calibration','head_refinement']}
    save_checkpoint(run/'best.pt',{**base,'state_dict':model.state_dict(),'inference':{},'epoch':offset+ep+1,'colab_settings':settings,'colab_improved':True,'head_refinement':False,'training_stage':'full_network_pure_supcon_rdrop','candidate_type':kind,'selection_metric':protocol['selection']});write_json(run/'best_validation.json',r)
  current=max(x['score'] for x in options);stale=0 if current>diagnostic_best+1e-4 else stale+1;diagnostic_best=max(diagnostic_best,current);chosen=chosen or max(options,key=lambda x:x['score']);r=chosen['report'];c=r['classification_report'];row={'epoch':offset+ep+1,'phase':'full_network_pure_supcon_rdrop',**{key:value/seen for key,value in totals.items()},'accuracy':correct/seen,'val_loss':r['negative_log_likelihood'],'val_accuracy':r['accuracy'],'val_macro_f1':c['macro avg']['f1-score'],'val_balanced_accuracy':c['macro avg']['recall'],'val_disgust_recall':c['disgust']['recall'],'val_disgust_f1':c['disgust']['f1-score'],'seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu,'sample_counts':draws.tolist(),'selected_kind':chosen['kind'],'saved_new_best':epoch_improved};rows.append(row)
  save_checkpoint(run/'latest.pt',{'settings':settings,'state_dict':student.state_dict(),'ema_state':ema.state_dict(),'projection_state':projection.state_dict(),'optimizer':optimizer.state_dict(),'history':rows,'updates':updates,'best':best,'diagnostic_best':diagnostic_best,'stale':stale,'improved':improved,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()});write_json(run/'training_history.json',old_history+rows);print(json.dumps(row),flush=True)
  if (len(rows)>=protocol['min_epochs'] and stale>=protocol['patience']) or time.perf_counter()-began>=protocol['max_minutes']*60:break
 handle.remove();write_json(run/'completion.json',{'completed_additional_epochs':len(rows),'validation_improved':improved,'test_accessed':False,'external_encoder':False,'teacher':False,'training_only_projection_removed_from_export':True,'synthetic':False})
if __name__=='__main__':main()
