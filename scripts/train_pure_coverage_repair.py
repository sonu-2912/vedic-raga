"""Pure FusDenseNet repair: complete train coverage, unit CE, gentler augmentation.

No auxiliary projection, regularizer, external teacher, logit offset or hybrid.
The baseline architecture and all inference widths are preserved.
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
from scripts.train_pure_fusdense import assert_pure,selection_score
ROOT=Path(__file__).resolve().parents[1]

class CoverageBatches(Sampler):
 """One shuffled natural pass plus cycling TRAIN-only minority/confusion samples."""
 def __init__(self,labels,seed=81109,natural=56,extras=None):
  self.labels=torch.as_tensor(labels,dtype=torch.long);self.seed=seed;self.epoch=0;self.natural=natural;self.extras=extras if extras is not None else {1:4,0:2,4:2};self.pools={c:torch.where(self.labels==c)[0].tolist() for c in self.extras};self.last_audit=None
  if len(self.labels)<natural or any(len(pool)<natural+max(self.extras.values()) for pool in self.pools.values()):raise ValueError('Insufficient distinct faces for unique batches')
 def __len__(self):return math.ceil(len(self.labels)/self.natural)
 def set_epoch(self,epoch):self.epoch=int(epoch)
 def __iter__(self):
  g=torch.Generator().manual_seed(self.seed+self.epoch);order=torch.randperm(len(self.labels),generator=g).tolist();queues={c:[] for c in self.extras};positions={c:0 for c in self.extras};covered=set();draws=torch.zeros(7,dtype=torch.long);natural_draws=0
  for start in range(0,len(order),self.natural):
   chunk=order[start:start+self.natural];chunk+=order[:self.natural-len(chunk)];natural_draws+=len(chunk);used=set(chunk)
   for c,k in self.extras.items():
    added=0
    while added<k:
     if positions[c]>=len(queues[c]):queues[c]=[self.pools[c][i] for i in torch.randperm(len(self.pools[c]),generator=g).tolist()];positions[c]=0
     idx=queues[c][positions[c]];positions[c]+=1
     if idx not in used:chunk.append(idx);used.add(idx);added+=1
   assert len(chunk)==len(set(chunk))==self.natural+sum(self.extras.values());covered.update(chunk);draws+=torch.bincount(self.labels[chunk],minlength=7)
   yield [chunk[i] for i in torch.randperm(len(chunk),generator=g).tolist()]
  self.last_audit={'unique_training_faces':len(covered),'training_pool':len(self.labels),'natural_draws':natural_draws,'natural_padding':natural_draws-len(self.labels),'class_draws':draws.tolist(),'all_training_faces_covered':len(covered)==len(self.labels),'no_duplicate_indices_within_batch':True}

def eligible(report,baseline,min_gain=.08):
 c,b=report['classification_report'],baseline['classification_report']
 return report['accuracy']>baseline['accuracy'] and c['macro avg']['f1-score']>b['macro avg']['f1-score'] and c['disgust']['recall']+1e-12>=b['disgust']['recall']+min_gain and c['disgust']['f1-score']>b['disgust']['f1-score']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--recover',action='store_true');a=ap.parse_args();run=ROOT/a.run_dir
 if (run/'candidate_frozen.json').exists() or (run/'bundle_manifest.json').exists():raise RuntimeError('Frozen experiment cannot be trained again')
 protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'));torch.set_num_threads(a.threads);torch.manual_seed(protocol['seed']);random.seed(protocol['seed'])
 ck=torch.load(run/'pure_baseline.pt',map_location='cpu',weights_only=True);assert_pure(ck);assert ck['architecture']['version']==4 and digest(run/'pure_baseline.pt')==protocol['initial_checkpoint_sha256']
 manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text(encoding='utf-8'));check_manifest(manifest);settings={**protocol,'architecture':ck['architecture'],'dataset_sha256':digest(ROOT/'data/private/fer2013-v1.zip'),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'threads':a.threads,'device':'cpu'};write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',manifest)
 train=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'train');val=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'val');student=model_from(ck,torch.device('cpu'));ema=copy.deepcopy(student);ema.requires_grad_(False)
 out,truth=predictions(student,val,torch.device('cpu'));baseline=metrics(out,truth);write_json(run/'initial_validation.json',baseline)
 head=list(student.classifier.parameters());ids={id(p) for p in head};backbone=[p for p in student.parameters() if id(p) not in ids];optimizer=torch.optim.AdamW([{'params':backbone,'lr':protocol['backbone_lr'],'weight_decay':.001},{'params':head,'lr':protocol['classifier_lr'],'weight_decay':.01}]);rates=[g['lr'] for g in optimizer.param_groups]
 labels=torch.tensor([y for _,y in train.rows]);sampler=CoverageBatches(labels,seed=protocol['seed']);loader=DataLoader(train,batch_sampler=sampler,num_workers=0)
 train.transform=transforms.Compose([transforms.RandomApply([transforms.ColorJitter(brightness=.08,contrast=.08)],p=.5),transforms.ToTensor()])
 initial_history=json.loads((run/'initial_history.json').read_text(encoding='utf-8'));offset=max(r['epoch'] for r in initial_history);rows=[];updates=0;best=selection_score(baseline);diagnostic_best=float('-inf');stale=0;improved=False;elapsed_previous=0.
 if a.recover:
  latest=torch.load(run/'latest.pt',map_location='cpu',weights_only=True);assert latest['settings']==settings;student.load_state_dict(latest['state_dict']);ema.load_state_dict(latest['ema_state']);optimizer.load_state_dict(latest['optimizer']);rows=latest['history'];updates=latest['updates'];best=latest['best'];diagnostic_best=latest['diagnostic_best'];stale=latest['stale'];improved=latest['improved'];elapsed_previous=latest['elapsed_wall_seconds'];torch.set_rng_state(latest['torch_rng']);random.setstate(latest['python_rng'])
 elif (run/'latest.pt').exists():raise RuntimeError('Use recover or a new declared run')
 initial_audit={'architecture':ck['architecture'],'inference_parameters':sum(p.numel() for p in student.parameters()),'source_sha256':digest(run/'pure_baseline.pt'),'external_encoder':False,'teacher':False,'train_only_projection_parameters':0,'inference_offsets':False,'class_loss_weights':[1.]*7,'sampling':{'natural_per_batch':56,'extra_train_class_indices':sampler.extras,'all_natural_faces_per_epoch':True},'dropout':student.classifier[2].p,'test_opened':False};write_json(run/'initialization_audit.json',initial_audit);print(json.dumps(initial_audit),flush=True)
 began=time.perf_counter();stop_reason='maximum_epochs'
 for ep in range(len(rows),protocol['epochs_max']):
  tick=time.perf_counter();cpu=time.process_time();sampler.set_epoch(ep);set_training_mode(student,True);loss_total=0.;correct=0;seen=0;grad_total=0.;class_loss=torch.zeros(7);class_seen=torch.zeros(7)
  for batch,(x,y) in enumerate(loader):
   warm=min(1.,(updates+1)/len(loader));decay=.2+.8*.5*(1+math.cos(math.pi*ep/protocol['epochs_max']))
   for group,lr in zip(optimizer.param_groups,rates):group['lr']=lr*warm*decay
   optimizer.zero_grad(set_to_none=True);logits=student(x);per=F.cross_entropy(logits,y,reduction='none');loss=per.mean()
   if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss; no partial epoch is claimed')
   loss.backward();grad_total+=float(nn.utils.clip_grad_norm_(backbone,2.));nn.utils.clip_grad_norm_(head,5.);optimizer.step();update_ema(ema,student,updates);updates+=1
   loss_total+=float(loss.detach())*len(y);correct+=int((logits.argmax(1)==y).sum());seen+=len(y);class_seen+=torch.bincount(y,minlength=7)
   class_loss.scatter_add_(0,y,per.detach())
   if (batch+1)%100==0:print(json.dumps({'epoch':offset+ep+1,'batch':batch+1,'batches':len(loader),'CE':loss_total/seen,'disgust_draws':int(class_seen[1])}),flush=True)
  assert sampler.last_audit['all_training_faces_covered'];options=[];chosen=None;new_best=False
  for kind,network in [('student',student),('EMA',ema)]:
   logits,y=predictions(network,val,torch.device('cpu'));report=metrics(logits,y);write_json(run/f'validation_{ep+1:02d}_{kind}.json',report);score=selection_score(report);item={'kind':kind,'report':report,'score':score};options.append(item)
   if eligible(report,baseline,protocol['validation_min_disgust_recall_gain']) and score>best:
    best=score;improved=True;chosen=item;new_best=True;base={k:v for k,v in ck.items() if k not in ['state_dict','inference','calibration','head_refinement']};save_checkpoint(run/'best.pt',{**base,'state_dict':network.state_dict(),'inference':{},'epoch':offset+ep+1,'colab_settings':settings,'colab_improved':True,'head_refinement':False,'training_stage':'full_network_pure_coverage_repair','candidate_type':kind,'selection_metric':protocol['selection']});write_json(run/'best_validation.json',report)
  current=max(i['score'] for i in options);stale=0 if current>diagnostic_best+1e-4 else stale+1;diagnostic_best=max(diagnostic_best,current);chosen=chosen or max(options,key=lambda i:i['score']);r=chosen['report'];c=r['classification_report'];row={'epoch':offset+ep+1,'phase':'full_network_pure_coverage_repair','loss':loss_total/seen,'CE':loss_total/seen,'accuracy':correct/seen,'val_accuracy':r['accuracy'],'val_macro_f1':c['macro avg']['f1-score'],'val_disgust_recall':c['disgust']['recall'],'val_disgust_f1':c['disgust']['f1-score'],'val_loss':r['negative_log_likelihood'],'selected_kind':chosen['kind'],'saved_new_best':new_best,'sample_counts':class_seen.int().tolist(),'mean_class_CE':(class_loss/class_seen.clamp_min(1)).tolist(),'sampling_audit':sampler.last_audit,'seconds':time.perf_counter()-tick,'cpu_seconds':time.process_time()-cpu};rows.append(row);elapsed=elapsed_previous+time.perf_counter()-began
  save_checkpoint(run/'latest.pt',{'settings':settings,'state_dict':student.state_dict(),'ema_state':ema.state_dict(),'optimizer':optimizer.state_dict(),'history':rows,'updates':updates,'best':best,'diagnostic_best':diagnostic_best,'stale':stale,'improved':improved,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'elapsed_wall_seconds':elapsed});write_json(run/'training_history.json',initial_history+rows);print(json.dumps(row),flush=True)
  if len(rows)>=protocol['min_epochs'] and stale>=protocol['patience']:stop_reason='validation_patience';break
  if elapsed>=protocol['max_minutes']*60:stop_reason='wall_budget_at_completed_epoch_boundary';break
 write_json(run/'completion.json',{'completed_additional_epochs':len(rows),'validation_improved':improved,'test_accessed':False,'external_encoder':False,'teacher':False,'stop_reason':stop_reason,'elapsed_wall_seconds':elapsed_previous+time.perf_counter()-began,'synthetic':False})
if __name__=='__main__':main()
