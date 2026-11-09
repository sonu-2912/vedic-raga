"""Audit failed pure-training objectives on TRAIN faces only; no test access."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from scripts.colab_common import fixed_split,check_manifest,digest,model_from,write_json,set_training_mode
from scripts.train_pure_fusdense import PairedClassBatches,class_balanced_supcon,symmetric_kl
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/pure_repair_20261108'
def main():
 RUN.mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(81109)
 manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text(encoding='utf-8'));check_manifest(manifest)
 ds=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'train');labels=torch.tensor([y for _,y in ds.rows]);base=torch.load(ROOT/'models/fusdense.pt',weights_only=True,map_location='cpu');assert base['architecture']['version']==4
 previous=torch.load(ROOT/'reports/pure_fusdense_20261108/latest.pt',weights_only=True,map_location='cpu')
 counts=torch.bincount(labels,minlength=7).float();p=counts/counts.sum();q=(50*p+2)/64;importance=(p/q).sqrt();importance/=torch.dot(q,importance)
 audit={'source_sha256':digest(ROOT/'models/fusdense.pt'),'failed_source_sha256':digest(ROOT/'reports/pure_fusdense_20261108/latest.pt'),'train_count':len(ds),'natural_prior':p.tolist(),'old_sample_prior':q.tolist(),'old_CE_importance':importance.tolist(),'old_normalized_CE_class_mass':(q*importance/(q*importance).sum()).tolist(),'no_test_or_validation_images_opened':True,'diagnostic_seed':81109,'augmentation':{},'gradient_batches':[]}
 indices=torch.cat([torch.where(labels==c)[0][torch.randperm(int(counts[c]))[:330]] for c in range(7)]).tolist()
 old_aug=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomAffine(5,translate=(.03,.03),scale=(.95,1.05),interpolation=transforms.InterpolationMode.BILINEAR,fill=128),transforms.ColorJitter(brightness=.12,contrast=.18),transforms.ToTensor()])
 raw=transforms.ToTensor()
 for name,state in [('baseline',base['state_dict']),('failed_v4_final',previous['state_dict'])]:
  model=model_from(base,torch.device('cpu'));model.load_state_dict(state);model.eval();audit['augmentation'][name]={}
  for mode,transform in [('raw',raw),('old_augmented',old_aug)]:
   ds.transform=transform;truth=[];pred=[];loss=[];torch.manual_seed(81109)
   with torch.no_grad():
    for x,y in DataLoader(Subset(ds,indices),batch_size=128):
     logits=model(x);truth.extend(y.tolist());pred.extend(logits.argmax(1).tolist());loss.extend(F.cross_entropy(logits,y,reduction='none').tolist())
   truth=np.array(truth);pred=np.array(pred);loss=np.array(loss);audit['augmentation'][name][mode]={str(c):{'n':int((truth==c).sum()),'recall':float((pred[truth==c]==c).mean()),'CE':float(loss[truth==c].mean())} for c in range(7)}
  del model
 model=model_from(base,torch.device('cpu'));model.load_state_dict(previous['state_dict']);set_training_mode(model,True)
 for m in model.modules():
  if isinstance(m,nn.Dropout):m.p=.4
 projection=nn.Sequential(nn.LayerNorm(588),nn.Linear(588,128),nn.ReLU(),nn.Linear(128,128));projection.load_state_dict(previous['projection_state']);projection.train();captured={};handle=model.classifier.register_forward_pre_hook(lambda module,args:captured.update(features=args[0]))
 backbone=[p for n,p in model.named_parameters() if not n.startswith('classifier.') and p.requires_grad]
 ds.transform=old_aug;sampler=PairedClassBatches(labels,batches=6)
 def vector(loss):
  grads=torch.autograd.grad(loss,backbone,retain_graph=True,allow_unused=True)
  return torch.cat([(torch.zeros_like(p) if g is None else g).flatten().detach() for p,g in zip(backbone,grads)])
 def cosine(a,b):return float(F.cosine_similarity(a[None],b[None]).item())
 for i,(x,y) in enumerate(DataLoader(ds,batch_sampler=sampler,num_workers=0)):
  a=model(x);f=captured['features'];b=model.classifier(f);per=.5*(F.cross_entropy(a,y,reduction='none',label_smoothing=.01)+F.cross_entropy(b,y,reduction='none',label_smoothing=.01));ce=(per*importance[y]).mean();rd=.5*symmetric_kl(a,b);sc=.05*class_balanced_supcon(projection(f[:,:588]),y);rare=per[y==1].mean();vectors={name:vector(loss) for name,loss in [('CE',ce),('RDrop',rd),('SupCon',sc),('disgust_CE',rare)]}
  audit['gradient_batches'].append({'batch':i,'disgust_count':int((y==1).sum()),'loss':{'CE':float(ce),'RDrop_weighted':float(rd),'SupCon_weighted':float(sc)},'backbone_gradient_norms':{n:float(v.norm()) for n,v in vectors.items()},'cosines':{f'{n}_vs_disgust_CE':cosine(v,vectors['disgust_CE']) for n,v in vectors.items() if n!='disgust_CE'}})
  print(json.dumps(audit['gradient_batches'][-1]),flush=True)
 handle.remove();write_json(RUN/'failure_audit.json',audit);print('Train-only failure audit finished.',flush=True)
if __name__=='__main__':main()
