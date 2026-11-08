"""Prospective validation-selected v5 refinement; no test pixels or targets."""
import argparse,copy,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from torchvision import transforms
from backend.models import FusDenseNet
from scripts.colab_common import check_manifest,fixed_split,digest,model_from,metrics,predictions,write_json,save_checkpoint,set_training_mode
from scripts.train_consistency import update_ema
ROOT=Path(__file__).resolve().parents[1]

def sampling_priors(counts,mix):
 p=counts.double()/counts.sum();q=(1-mix)*p+mix/len(p)
 return p.float(),q.float()

def initialize_v5(ck):
 model=FusDenseNet(**{**ck['architecture'],'version':5});state=model.state_dict()
 for k,v in ck['state_dict'].items():state[k]=v
 model.load_state_dict(state);return model

def natural_checkpoint_state(model,p,q):
 state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 state['classifier.3.bias']+=torch.log(p/q)
 return state

def selected(report,base):
 c,b=report['classification_report'],base['classification_report']
 return report['accuracy']>base['accuracy'] and c['macro avg']['f1-score']>=b['macro avg']['f1-score'] and c['disgust']['recall']>b['disgust']['recall'] and c['disgust']['f1-score']>=b['disgust']['f1-score']

def score(r):
 c=r['classification_report'];return .5*r['accuracy']+.3*c['macro avg']['f1-score']+.2*c['disgust']['f1-score']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--recover',action='store_true');a=ap.parse_args();run=ROOT/a.run_dir
 protocol=json.loads((run/'protocol.json').read_text());assert not (run/'bundle_manifest.json').exists() and not (run/'candidate_frozen.json').exists()
 torch.set_num_threads(a.threads);torch.manual_seed(808);np.random.seed(808);random.seed(808)
 initial=run/'baseline.pt';ck=torch.load(initial,map_location='cpu',weights_only=True);assert digest(initial)==protocol['initial_checkpoint_sha256']
 manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text());check_manifest(manifest)
 settings={**protocol,'architecture':{**ck['architecture'],'version':5},'dataset_sha256':digest(ROOT/'data/private/fer2013-v1.zip'),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'seed':808,'device':'cpu','threads':a.threads}
 write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',manifest)
 train=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'train');val=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'val')
 baseline=model_from(ck,torch.device('cpu'));bl,by=predictions(baseline,val,torch.device('cpu'));base=metrics(bl,by);write_json(run/'initial_validation.json',base)
 student=initialize_v5(ck);student.eval()
 fixture=torch.stack([val[i][0] for i in range(4)])
 with torch.no_grad():difference=float((baseline(fixture)-student(fixture)).abs().max())
 assert difference<1e-5
 counts=torch.bincount(torch.tensor([y for _,y in train.rows]),minlength=7).float();p,q=sampling_priors(counts,protocol['uniform_sampling_mix'])
 # Internal student learns the sampling distribution. Export folds an exact,
 # TRAIN-prior-only correction into output bias; no hand-set disgust threshold.
 with torch.no_grad():student.classifier[3].bias.add_(torch.log(q/p))
 ema=copy.deepcopy(student);ema.requires_grad_(False)
 for m in student.modules():
  if isinstance(m,nn.Dropout):m.p=protocol['dropout']
 head=list(student.classifier.parameters());refine=list(student.spatial_refine.parameters());special={id(v) for v in head+refine};backbone=[v for v in student.parameters() if id(v) not in special]
 opt=torch.optim.AdamW([{'params':backbone,'lr':protocol['backbone_lr'],'weight_decay':.003},{'params':head,'lr':protocol['head_lr'],'weight_decay':.02},{'params':refine,'lr':protocol['new_refinement_lr'],'weight_decay':.003}])
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=protocol['epochs_max'])
 train.transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomAffine(8,translate=(.06,.06),scale=(.9,1.1),interpolation=transforms.InterpolationMode.BILINEAR,fill=128),transforms.ColorJitter(brightness=.25,contrast=.3),transforms.ToTensor(),transforms.RandomErasing(p=.15,scale=(.015,.05),ratio=(.5,2.),value=.5)])
 sampler=WeightedRandomSampler((q/p)[torch.tensor([y for _,y in train.rows])].double(),len(train),replacement=True)
 loader=DataLoader(train,batch_size=64,sampler=sampler,num_workers=0)
 history=json.loads((ROOT/'reports/training_history.json').read_text());offset=max(r['epoch'] for r in history);rows=[];start=0;best_score=score(base);improved=False;stale=0;updates=0
 inference=initialize_v5(ck);inference.eval()
 audit={'initial_max_logit_difference':difference,'parameters':sum(v.numel() for v in student.parameters()),'classifier_parameters':sum(v.numel() for v in head),'counts':counts.int().tolist(),'natural_prior':p.tolist(),'sampling_prior':q.tolist(),'expected_disgust_examples_per_epoch':float(q[1]*len(train)),'test_opened':False};write_json(run/'initialization_audit.json',audit);print(json.dumps(audit),flush=True)
 if a.recover:
  latest=torch.load(run/'latest.pt',map_location='cpu',weights_only=True);assert latest['settings']==settings;student.load_state_dict(latest['state_dict']);ema.load_state_dict(latest['ema_state']);opt.load_state_dict(latest['optimizer']);scheduler.load_state_dict(latest['scheduler']);rows=latest['history'];start=len(rows);best_score=latest['best_score'];improved=latest['improved'];stale=latest['stale'];updates=latest['updates'];torch.set_rng_state(latest['torch_rng']);random.setstate(latest['python_rng'])
 elif (run/'latest.pt').exists():raise RuntimeError('Use recover; completed artifacts preserved')
 began=time.perf_counter()
 for ep in range(start,protocol['epochs_max']):
  t=time.perf_counter();set_training_mode(student,True)
  for m in student.spatial_refine.modules():
   if isinstance(m,nn.BatchNorm2d):m.train();m.momentum=.05
  loss_sum=0.;correct=0;seen=0;sample_counts=torch.zeros(7,dtype=torch.long)
  for batch,(x,y) in enumerate(loader):
   opt.zero_grad(set_to_none=True);out=student(x);loss=F.cross_entropy(out,y,label_smoothing=.02)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),5.);opt.step();update_ema(ema,student,updates);updates+=1
   loss_sum+=float(loss.detach())*len(y);correct+=int((out.argmax(1)==y).sum());seen+=len(y);sample_counts+=torch.bincount(y,minlength=7)
   if (batch+1)%75==0:print(json.dumps({'epoch':offset+ep+1,'batch':batch+1,'batches':len(loader),'loss':loss_sum/seen,'disgust_samples':int(sample_counts[1])}),flush=True)
  candidates=[];chosen=None;epoch_improved=False
  for name,m in [('student',student),('EMA',ema)]:
   inference.load_state_dict(natural_checkpoint_state(m,p,q));vl,vy=predictions(inference,val,torch.device('cpu'),128);report=metrics(vl,vy);write_json(run/f'validation_{ep+1:02d}_{name}.json',report)
   c=report['classification_report'];item={'kind':name,'report':report,'eligible':selected(report,base),'score':score(report)};candidates.append(item)
   if item['eligible'] and item['score']>best_score:
    best_score=item['score'];improved=True;epoch_improved=True;chosen=item
    base_ck={k:v for k,v in ck.items() if k not in ['state_dict','architecture','inference','calibration','head_refinement']}
    save_checkpoint(run/'best.pt',{**base_ck,'state_dict':inference.state_dict(),'architecture':settings['architecture'],'inference':{},'epoch':offset+ep+1,'colab_settings':settings,'training_stage':'full_network_spatial_generalization','colab_improved':True,'head_refinement':False,'candidate_type':name,'selection_metric':protocol['selection']});write_json(run/'best_validation.json',report)
  chosen=chosen or max(candidates,key=lambda x:x['score']);r=chosen['report'];c=r['classification_report'];row={'epoch':offset+ep+1,'phase':'full_network_spatial_generalization','loss':loss_sum/seen,'accuracy':correct/seen,'val_loss':r['negative_log_likelihood'],'val_accuracy':r['accuracy'],'val_macro_f1':c['macro avg']['f1-score'],'val_disgust_recall':c['disgust']['recall'],'val_disgust_f1':c['disgust']['f1-score'],'val_balanced_accuracy':c['macro avg']['recall'],'seconds':time.perf_counter()-t,'chosen_kind':chosen['kind'],'saved_new_best':epoch_improved,'sample_counts':sample_counts.tolist()};rows.append(row);stale=0 if epoch_improved else stale+1;scheduler.step()
  save_checkpoint(run/'latest.pt',{'settings':settings,'state_dict':student.state_dict(),'ema_state':ema.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'history':rows,'best_score':best_score,'improved':improved,'stale':stale,'updates':updates,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()});write_json(run/'training_history.json',history+rows);print(json.dumps(row),flush=True)
  if stale>=protocol['patience'] or time.perf_counter()-began>=protocol['max_minutes']*60:break
 write_json(run/'completion.json',{'completed_additional_epochs':len(rows),'validation_improved':improved,'best_score':best_score,'test_accessed':False,'synthetic':False,'parameters_trained':'Full FusDenseNet plus disclosed high-resolution residual refinement','complete':True})
if __name__=='__main__':main()
