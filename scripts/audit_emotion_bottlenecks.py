"""Measured training/validation audit. Never opens test images."""
import argparse,copy,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from scripts.colab_common import fixed_split,check_manifest,digest,model_from,metrics,predictions,write_json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();run=ROOT/a.run_dir;run.mkdir(parents=True,exist_ok=True)
 if (run/"audit.json").exists():raise RuntimeError("Completed audit preserved")
 torch.set_num_threads(4);torch.manual_seed(808)
 manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text());check_manifest(manifest)
 ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True);model=model_from(ck,torch.device('cpu'));model.eval()
 train=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'train');val=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'val')
 counts={s:dict(Counter(r['label'] for r in manifest[s])) for s in ['train','val']}
 cache=torch.load(ROOT/'reports/pruned_head_20261107/train_features.pt',map_location='cpu',weights_only=True)
 print('cache keys',list(cache),flush=True)
 features=cache.get('features');labels=cache.get('labels')
 assert cache['manifest_sha256']==digest(ROOT/'reports/fer_training_manifest.json');assert cache['source_sha256']==digest(ROOT/'models/fusdense.pt');assert labels.tolist()==[y for _,y in train.rows]
 with torch.inference_mode():trlog=torch.cat([model.classifier(x) for x in features.split(256)])
 tr=metrics(trlog,labels);write_json(run/'training_raw_evaluation.json',tr)
 vl,vy=predictions(model,val,torch.device('cpu'));vr=metrics(vl,vy);write_json(run/'validation_raw_evaluation.json',vr)
 # Exact feature contribution ablation is diagnostic, not a trained alternative.
 vcache=torch.load(ROOT/'reports/pruned_head_20261107/val_features.pt',map_location='cpu',weights_only=True);assert vcache['manifest_sha256']==digest(ROOT/'reports/fer_training_manifest.json');assert vcache['source_sha256']==digest(ROOT/'models/fusdense.pt');vf=vcache['features'];assert vcache['labels'].tolist()==vy.tolist()
 groups={'late_fused':(0,588),'stem_spatial':(588,1356),'first_dense_spatial':(1356,3276)};ablations={}
 with torch.inference_mode():
  for name,(lo,hi) in groups.items():
   f=vf.clone();f[:,lo:hi]=0;rep=metrics(model.classifier(f),vy);ablations[name]={'accuracy':rep['accuracy'],'macro_f1':rep['classification_report']['macro avg']['f1-score'],'disgust_recall':rep['classification_report']['disgust']['recall']}
 # Audit first BatchNorm input population mismatch on a deterministic training-only subset.
 bn=dict((n,m) for n,m in model.named_modules() if isinstance(m,nn.BatchNorm2d));observed={n:[] for n in bn};handles=[]
 for name,layer in bn.items():
  def hook(mod,args,name=name):
   x=args[0].detach();observed[name].append((x.mean((0,2,3)),x.var((0,2,3),unbiased=False)))
  handles.append(layer.register_forward_pre_hook(hook))
 indices=torch.randperm(len(train))[:2048].tolist()
 with torch.inference_mode():
  for x,_ in DataLoader(Subset(train,indices),batch_size=128):model(x)
 for h in handles:h.remove()
 drift={n:{'mean_abs_standardized_shift':float(((torch.stack([x[0] for x in vals]).mean(0)-bn[n].running_mean)/(bn[n].running_var+bn[n].eps).sqrt()).abs().mean()),'median_variance_ratio':float((torch.stack([x[1] for x in vals]).mean(0)/(bn[n].running_var+bn[n].eps)).median())} for n,vals in observed.items()}
 # Recalibration candidate uses training pixels only; validation determines whether it helps.
 recal=copy.deepcopy(model);recal.eval()
 for m in recal.modules():
  if isinstance(m,nn.BatchNorm2d):m.reset_running_stats();m.momentum=None;m.train()
 with torch.inference_mode():
  for x,_ in DataLoader(Subset(train,indices),batch_size=128):recal(x)
 rl,ry=predictions(recal,val,torch.device('cpu'));rr=metrics(rl,ry);write_json(run/'bn_recalibration_validation.json',rr)
 # Gradient contract: gradients can reach attention/dense path, distinguish weak influence from broken graph.
 model.train()
 for m in model.modules():
  if isinstance(m,nn.BatchNorm2d):m.eval()
 x,y=next(iter(DataLoader(Subset(train,indices),batch_size=32)));loss=nn.functional.cross_entropy(model(x),y);loss.backward()
 grads={n:float(v.grad.norm()) for n,v in model.named_parameters() if v.grad is not None and ('cross.in_proj_weight' in n or n=='stem.0.weight' or n=='classifier.0.weight')}
 out={'source_sha256':digest(ROOT/'models/fusdense.pt'),'train_counts_by_index':counts['train'],'validation_counts_by_index':counts['val'],'parameters':sum(p.numel() for p in model.parameters()),'classifier_parameters':sum(p.numel() for p in model.classifier.parameters()),'training_accuracy':tr['accuracy'],'training_macro_f1':tr['classification_report']['macro avg']['f1-score'],'training_disgust_recall':tr['classification_report']['disgust']['recall'],'validation_accuracy':vr['accuracy'],'validation_macro_f1':vr['classification_report']['macro avg']['f1-score'],'validation_disgust_recall':vr['classification_report']['disgust']['recall'],'validation_confusion':vr['confusion_matrix'],'feature_zero_ablation':ablations,'batchnorm_population_audit':drift,'bn_recalibration_validation_accuracy':rr['accuracy'],'bn_recalibration_validation_disgust_recall':rr['classification_report']['disgust']['recall'],'gradient_norms':grads,'test_images_opened':False,'limitations':'Zero ablations and2048-training-image BN recalibration are diagnostics, not causal controlled architectural or fresh test results.'}
 write_json(run/'audit.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
