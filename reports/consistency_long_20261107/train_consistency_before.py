"""Full-path CPU fine-tuning; train-only augmentation/teacher; validation selection."""
import argparse,copy,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from scripts.colab_common import (fixed_split,check_manifest,digest,model_from,predictions,metrics,save_checkpoint,write_json,set_training_mode)
ROOT=Path(__file__).resolve().parents[1]
class Indexed(Dataset):
 def __init__(self,dataset):self.dataset=dataset
 def __len__(self):return len(self.dataset)
 def __getitem__(self,i):x,y=self.dataset[i];return x,y,i

def teacher_kl(logits,teacher,labels,temperature=2.):
 mask=teacher.argmax(1)==labels
 if not bool(mask.any()):return logits.sum()*0
 return F.kl_div(F.log_softmax(logits[mask]/temperature,1),F.softmax(teacher[mask]/temperature,1),reduction='batchmean')*temperature**2

@torch.no_grad()
def update_ema(ema,student,updates):
 decay=min(.995,(1+updates)/(10+updates))
 for target,current in zip(ema.parameters(),student.parameters()):target.mul_(decay).add_(current,alpha=1-decay)
 for target,current in zip(ema.buffers(),student.buffers()):target.copy_(current)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--initial',required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--recover',action='store_true');a=p.parse_args()
 run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=True)
 if (run/'bundle_manifest.json').exists():raise SystemExit('This run is locked after test evaluation.')
 torch.set_num_threads(a.threads);torch.manual_seed(211);np.random.seed(211);random.seed(211)
 protocol=json.loads((run/'protocol.json').read_text());assert digest(a.initial)==protocol['initial_checkpoint_sha256']
 manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text());check_manifest(manifest)
 settings={**protocol,'architecture':torch.load(a.initial,map_location='cpu',weights_only=True)['architecture'],'dataset_sha256':digest(a.data),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'device':'cpu','seed':211,'threads':a.threads,'batch_size':64}
 write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',manifest)
 train=fixed_split(a.data,manifest,'train');val=fixed_split(a.data,manifest,'val')
 ck=torch.load(a.initial,map_location='cpu',weights_only=True);student=model_from(ck,torch.device('cpu'));student.eval()
 teacher_file=run/'training_teacher.pt'
 if teacher_file.exists():
  cache=torch.load(teacher_file,map_location='cpu',weights_only=True)
  assert cache['source_sha256']==digest(a.initial) and cache['manifest_sha256']==settings['manifest_sha256']
 else:
  print('Caching only fixed-training teacher predictions; no validation/test targets.',flush=True)
  partial=run/'training_teacher.partial.pt';outputs=[];target=[];done=0
  if partial.exists():
   saved=torch.load(partial,map_location='cpu',weights_only=True);assert saved['source_sha256']==digest(a.initial) and saved['manifest_sha256']==settings['manifest_sha256'];outputs=[saved['logits']];target=[saved['labels']];done=len(saved['labels'])
  from torch.utils.data import Subset
  with torch.inference_mode():
   for n,(x,y) in enumerate(DataLoader(Subset(train,range(done,len(train))),batch_size=128)):
    outputs.append(((student(x)+student(x.flip(-1)))/2).cpu());target.append(y)
    if (n+1)%25==0:
     save_checkpoint(partial,{'source_sha256':digest(a.initial),'manifest_sha256':settings['manifest_sha256'],'logits':torch.cat(outputs),'labels':torch.cat(target)});print(json.dumps({'teacher_train_faces':sum(len(t) for t in target),'total_training_faces':len(train)}),flush=True)
  teacher,labels=torch.cat(outputs),torch.cat(target)
  cache={'source_sha256':digest(a.initial),'manifest_sha256':settings['manifest_sha256'],'logits':teacher,'labels':labels};save_checkpoint(teacher_file,cache)
  print('Training teacher cache complete.',flush=True)
 assert cache['labels'].tolist()==[label for _,label in train.rows]
 teacher=cache['logits']
 initial_logits,initial_y=predictions(student,val,torch.device('cpu'),128);initial=metrics(initial_logits,initial_y);write_json(run/'initial_validation.json',initial)
 def eligible(r):
  c=r['classification_report'];b=initial['classification_report']
  return r['accuracy']>=initial['accuracy'] and c['macro avg']['f1-score']>=b['macro avg']['f1-score'] and c['disgust']['f1-score']>=b['disgust']['f1-score'] and c['disgust']['recall']>=b['disgust']['recall']
 def score(r):return .4*r['accuracy']+.6*r['classification_report']['macro avg']['f1-score']
 history=json.loads((ROOT/'reports/training_history.json').read_text());offset=max(r['epoch'] for r in history);base={k:v for k,v in ck.items() if k not in ['state_dict','inference','calibration']}
 base.update(head_refinement=False,training_stage='full_network_consistency',colab_settings=settings,selection_metric=protocol['selection'])
 ema=copy.deepcopy(student);ema.requires_grad_(False)
 for module in student.modules():
  if isinstance(module,torch.nn.Dropout):module.p=protocol['dropout_during_training']
 head=list(student.classifier.parameters());head_ids={id(p) for p in head};backbone=[p for p in student.parameters() if id(p) not in head_ids]
 optimizer=torch.optim.AdamW([{'params':backbone,'lr':protocol['backbone_lr']},{'params':head,'lr':protocol['classifier_lr']}],weight_decay=.001)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['epochs_max'],eta_min=0)
 train.transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomAffine(7,translate=(.04,.04),scale=(.95,1.05),interpolation=transforms.InterpolationMode.BILINEAR),transforms.ColorJitter(brightness=.15,contrast=.15),transforms.ToTensor()])
 counts=torch.bincount(torch.tensor([y for _,y in train.rows]),minlength=7).float();weights=(counts.sum()/(7*counts)).pow(protocol['class_weight_power']);weights/=weights.mean()
 loader=DataLoader(Indexed(train),batch_size=64,shuffle=True,num_workers=0)
 completed=[];best_score=score(initial);updates=0;stale=0;start=0;improved_any=False
 if a.recover:
  state=torch.load(run/'latest.pt',map_location='cpu',weights_only=True);assert state['settings']==settings
  student.load_state_dict(state['state_dict']);ema.load_state_dict(state['ema_state']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);completed=state['history'];best_score=state['best_score'];updates=state['updates'];stale=state['stale'];start=len(completed);improved_any=state['validation_improved'];torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
 else:
  if (run/'best.pt').exists():
   seeded=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
   if (run/'latest.pt').exists() or seeded.get('colab_improved') or any(not torch.equal(t,seeded['state_dict'][k]) for k,t in ck['state_dict'].items()):raise SystemExit('Use --recover or a new run directory.')
  save_checkpoint(run/'best.pt',{**ck,'inference':{},'colab_settings':settings,'colab_improved':False})
 began=time.perf_counter()
 for epoch in range(start,protocol['epochs_max']):
  t=time.perf_counter();set_training_mode(student,True);loss_sum=0.;correct=0;seen=0
  for batch,(x,y,index) in enumerate(loader):
   optimizer.zero_grad(set_to_none=True);logits=student(x)
   ce=F.cross_entropy(logits,y,weight=weights,label_smoothing=.02);kd=teacher_kl(logits,teacher[index],y,protocol['distill_temperature']);loss=ce+protocol['distill_weight']*kd
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; retained last completed epoch.')
   loss.backward();torch.nn.utils.clip_grad_norm_(backbone,1.);torch.nn.utils.clip_grad_norm_(head,5.);optimizer.step();update_ema(ema,student,updates);updates+=1
   loss_sum+=float(loss.detach())*len(y);correct+=int((logits.argmax(1)==y).sum());seen+=len(y)
   if (batch+1)%75==0:print(json.dumps({'epoch':offset+epoch+1,'batch':batch+1,'batches':len(loader),'training_loss':loss_sum/seen}),flush=True)
  candidates=[];improved_epoch=False;chosen_kind=None
  for kind,model in [('student',student),('EMA',ema)]:
   out,truth=predictions(model,val,torch.device('cpu'),128);report=metrics(out,truth);c=report['classification_report'];item={'kind':kind,'accuracy':report['accuracy'],'macro_f1':c['macro avg']['f1-score'],'disgust_recall':c['disgust']['recall'],'disgust_f1':c['disgust']['f1-score'],'eligible':eligible(report),'score':score(report)};candidates.append(item)
   write_json(run/(f'validation_{epoch+1:02d}_{kind}.json'),report)
   if item['eligible'] and item['score']>best_score:
    best_score=item['score'];improved_any=True;improved_epoch=True;chosen_kind=kind;stale=0;save_checkpoint(run/'best.pt',{**base,'state_dict':model.state_dict(),'inference':{},'epoch':offset+epoch+1,'validation_macro_f1':item['macro_f1'],'colab_improved':True,'candidate_type':kind});write_json(run/'best_validation.json',report)
  best=next(c for c in candidates if c['kind']==chosen_kind) if chosen_kind else max(candidates,key=lambda r:r['score']);row={'epoch':offset+epoch+1,'phase':'full_network_consistency','loss':loss_sum/seen,'accuracy':correct/seen,'val_loss':float(torch.nn.functional.cross_entropy(out,truth)),'val_accuracy':best['accuracy'],'val_macro_f1':best['macro_f1'],'val_balanced_accuracy':report['classification_report']['macro avg']['recall'],'val_disgust_recall':best['disgust_recall'],'val_disgust_f1':best['disgust_f1'],'seconds':time.perf_counter()-t,'candidates':candidates}
  # Track loss/recall from the same student/EMA report as the displayed row.
  report=json.loads((run/(f'validation_{epoch+1:02d}_{best["kind"]}.json')).read_text());row['val_loss']=report['negative_log_likelihood'];row['val_balanced_accuracy']=report['classification_report']['macro avg']['recall']
  completed.append(row);stale=0 if improved_epoch else stale+1;scheduler.step()
  save_checkpoint(run/'latest.pt',{'state_dict':student.state_dict(),'ema_state':ema.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'history':completed,'best_score':best_score,'updates':updates,'stale':stale,'settings':settings,'validation_improved':improved_any,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()})
  write_json(run/'training_history.json',history+completed);print(json.dumps(row),flush=True)
  if stale>=protocol['patience'] or time.perf_counter()-began>=protocol['max_minutes']*60:break
 write_json(run/'completion.json',{'completed_additional_epochs':len(completed),'validation_improved':improved_any,'best_score':best_score,'test_accessed':False,'synthetic':False,'parameters_trained':'Full emotion path, not frozen-feature MLP-only training'})
if __name__=='__main__':main()
