"""TRAIN-only diagnostic of frozen BatchNorm activation-statistic drift."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from scripts.colab_common import fixed_split,write_json,digest
from scripts.compact_fusdense import model_from
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_capacity_20261108'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
@torch.no_grad()
def collect(model,loader):
 model.eval();stats={};handles=[];buffers={}
 for name,module in model.named_modules():
  if not isinstance(module,nn.BatchNorm2d):continue
  buffers[name]=(module.running_mean.clone().double(),module.running_var.clone().double(),module.eps)
  def hook(m,args,key=name):
   x=args[0];dims=(0,2,3);n=x.shape[0]*x.shape[2]*x.shape[3];mu=x.mean(dims).double();var=x.var(dims,unbiased=False).double();record=stats.setdefault(key,{'n':0,'sum':torch.zeros_like(mu),'sum2':torch.zeros_like(mu)});record['n']+=n;record['sum']+=mu*n;record['sum2']+=(var+mu.square())*n
  handles.append(module.register_forward_pre_hook(hook))
 for x,_ in loader:model(x)
 for h in handles:h.remove()
 result={}
 for name,r in stats.items():
  mu=r['sum']/r['n'];var=(r['sum2']/r['n']-mu.square()).clamp_min(0);running_mean,running_var,eps=buffers[name];result[name]={'mean_absolute_mean_shift_in_stored_SD':float(((mu-running_mean).abs()/(running_var+eps).sqrt()).mean()),'mean_empirical_to_stored_variance_ratio':float((var/(running_var+eps)).mean())}
 return result

def main():
 torch.set_num_threads(1);initial=torch.load(RUN/'initial.pt',map_location='cpu',weights_only=True);before=digest(RUN/'full_latest.pt');latest=torch.load(RUN/'full_latest.pt',map_location='cpu',weights_only=True);after=digest(RUN/'full_latest.pt');assert before==after,'Checkpoint changed during diagnostic load; retry later';warm=torch.load(RUN/'warm_start.pt',map_location='cpu',weights_only=True);current={**warm,'state_dict':latest['state_dict']};manifest=read(RUN/'fer_training_manifest.json');train=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'train');loader=DataLoader(Subset(train,range(1024)),batch_size=64,shuffle=False);reports={}
 for name,ck in [('deployed_pure_baseline',initial),('latest_compact_student',current)]:
  m=model_from(ck,torch.device('cpu'));r=collect(m,loader);reports[name]={'layer_statistics':r,'mean_across_layers_of_SD_shift':sum(z['mean_absolute_mean_shift_in_stored_SD'] for z in r.values())/len(r)};del m
 result={'source_latest_sha256':before,'completed_full_epochs_at_audit':len(latest['history']),'cohort':'First1024fixedTRAINfaces only; no validation/test statistics used','interpretation':'Frozen-statistic mismatch hypothesis only. TRAIN drift does not establish that recalibration improves validation or test.','reports':reports};write_json(RUN/f"batchnorm_drift_after_epoch_{len(latest['history']):02d}.json",result);print({k:v['mean_across_layers_of_SD_shift'] for k,v in reports.items()})
if __name__=='__main__':main()
