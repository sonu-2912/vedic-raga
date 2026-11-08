"""Frozen ImageNet DenseNet feature probe alongside unchanged FusDenseNet logits.

All optimization/selection use TRAIN/validation only. No test images opened.
"""
import json,hashlib,shutil,time,copy
from pathlib import Path
import torch,torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from torchvision.models import DenseNet121_Weights,densenet121
from scripts.colab_common import digest,write_json,save_checkpoint,fixed_split,model_from
from scripts.colab_common import metrics
from scripts.train_spatial_generalization import score,selected
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/dense_transfer112_20261108'

def transfer_head():
 head=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,128),nn.ReLU(),nn.Dropout(.5),nn.Linear(128,7))
 nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias);return head

def main():
 RUN.mkdir(exist_ok=True)
 if (RUN/'completion.json').exists():raise RuntimeError('Completed experiment preserved')
 torch.set_num_threads(2);torch.manual_seed(906)
 source=ROOT/'reports/spatial_generalization_20261108/baseline.pt';sha=digest(source)
 manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text(encoding='utf-8'));manifest_sha=digest(ROOT/'reports/fer_training_manifest.json')
 protocol={'initial_checkpoint_sha256':sha,'method':'Frozen ImageNet1K DenseNet121 features add a trainable residual emotion head to unchanged FusDenseNet v4 logits. This is a hybrid project extension, not pure FusDenseNet or exact author reproduction.','train_counts':23503,'validation_count':3970,'epochs_max':20,'patience':8,'dropout':.5,'learning_rate':.001,'weight_decay':.01,'CE_class_weight_power':.25,'seed':906,'backbone_training':'Both feature extractors frozen; ONLY new LayerNorm/1024->128->7 residual head trained. No FER-pretrained teacher.','preprocessing':'CPU adaptation: grayscale replicated RGB, resize128/center112, official ImageNet mean/std. This differs from the source224-pixel recipe.','selection':'Same validation accuracy+disgust recall strict gains, macro/disgustF1 floors; .5accuracy+.3macroF1+.2disgustF1.','test':'No test opened; at most one final candidate will be frozen across this and the v5spatial run before one descriptive historically inspected-test evaluation.','disclosure':'Added after v5epoch2validation and failed translation screen, before any new test access. Validation repeatedly reused; not preregistered at project start or fresh confirmatory evidence.'}
 write_json(RUN/'protocol.json',protocol)
 weights=DenseNet121_Weights.IMAGENET1K_V1;hub=ROOT/'external/torchvision_cache';hub.mkdir(exist_ok=True);torch.hub.set_dir(str(hub))
 # Official weight file is30.8MB; do not download any ImageNet images.
 assert shutil.disk_usage(ROOT).free>300*1024**2
 import requests
 response=requests.head(weights.url,timeout=30);response.raise_for_status();size=int(response.headers.get('Content-Length',0));assert 0<size<40*1024**2
 license_path=hub/'Torchvision_BSD_3_Clause_LICENSE.txt'
 if not license_path.exists():
  response=requests.get('https://raw.githubusercontent.com/pytorch/vision/v0.20.1/LICENSE',timeout=30);response.raise_for_status();license_path.write_text(response.text,encoding='utf-8')
 net=densenet121(weights=weights);net.eval();net.requires_grad_(False);trunk=net.features;transform=weights.transforms(crop_size=112,resize_size=128);weight_file=hub/'checkpoints'/weights.url.split('/')[-1];weight_sha=digest(weight_file)
 write_json(RUN/'pretrained_provenance.json',{'source':'Official Torchvision DenseNet121 IMAGENET1K_V1','url':weights.url,'sha256':weight_sha,'bytes':weight_file.stat().st_size,'torchvision':str(torchvision.__version__),'training_corpus':'ImageNet1K; no FER expression supervision in this source','code_licence':'Torchvision BSD3Clause, preserved locally. ImageNet access/data rights are distinct; no ImageNet images downloaded or redistributed.','native_reported_accuracy':'74.434% is ImageNet1K top1, NOT FER or project accuracy','preprocessing':'weights.transforms(crop_size=112,resize_size=128), grayscale replicatedRGB; CPU adaptation, not native224-pixel inference','frozen':True,'test_images_used':False})
 def cache(split):
  path=RUN/f'{split}_features.pt'
  if path.exists():
   saved=torch.load(path,map_location='cpu',weights_only=True);assert saved['manifest_sha256']==manifest_sha and saved['weight_sha256']==weight_sha;return saved['features'],saved['labels']
  ds=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,split);features=[];labels=[];partial=RUN/f'{split}_features.partial.pt';done=0
  if partial.exists():
   saved=torch.load(partial,map_location='cpu',weights_only=True);assert saved['manifest_sha256']==manifest_sha and saved['weight_sha256']==weight_sha;features=[saved['features']];labels=[saved['labels']];done=len(saved['labels'])
  from torch.utils.data import Subset
  with torch.inference_mode():
   for i,(x,y) in enumerate(DataLoader(Subset(ds,range(done,len(ds))),batch_size=64)):
    rgb=transform(x.expand(-1,3,-1,-1));activation=F.relu(trunk(rgb));z=F.adaptive_avg_pool2d(activation,1).flatten(1);features.append(z.cpu());labels.append(y)
    if (i+1)%32==0:
     save_checkpoint(partial,{'manifest_sha256':manifest_sha,'weight_sha256':weight_sha,'features':torch.cat(features),'labels':torch.cat(labels)});print(json.dumps({'feature_split':split,'images':sum(len(y) for y in labels),'total':len(ds)}),flush=True)
  x,y=torch.cat(features),torch.cat(labels);assert y.tolist()==[label for _,label in ds.rows];save_checkpoint(path,{'manifest_sha256':manifest_sha,'weight_sha256':weight_sha,'features':x,'labels':y});return x,y
 x,y=cache('train');vx,vy=cache('val')
 del net,trunk
 ck=torch.load(source,map_location='cpu',weights_only=True);base=model_from(ck,torch.device('cpu'));base.eval()
 def baseline_logits(split):
  c=torch.load(ROOT/f'reports/pruned_head_20261107/{split}_features.pt',map_location='cpu',weights_only=True);assert c['source_sha256']==sha and c['manifest_sha256']==manifest_sha
  with torch.inference_mode():out=torch.cat([base.classifier(f) for f in c['features'].split(256)])
  return out,c['labels']
 bl,by=baseline_logits('train');bvl,bvy=baseline_logits('val');assert torch.equal(by,y) and torch.equal(bvy,vy);initial=metrics(bvl,vy);write_json(RUN/'initial_validation.json',initial)
 head=transfer_head();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20);counts=torch.bincount(y,minlength=7).float();w=(counts.sum()/(7*counts)).pow(.25);w/=w.mean();loader=DataLoader(TensorDataset(x,bl,y),batch_size=256,shuffle=True);rows=[];best=score(initial);improved=False;stale=0
 for ep in range(20):
  t=time.perf_counter();head.train();total=0.;correct=0
  for f,prior,target in loader:
   opt.zero_grad(set_to_none=True);out=prior+head(f);loss=F.cross_entropy(out,target,weight=w,label_smoothing=.02);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5.);opt.step();total+=float(loss.detach())*len(target);correct+=int((out.argmax(1)==target).sum())
  head.eval()
  with torch.inference_mode():vl=bvl+head(vx)
  report=metrics(vl,vy);write_json(RUN/f'validation_{ep+1:02d}_student.json',report);c=report['classification_report'];ok=selected(report,initial) and score(report)>best
  if ok:
   best=score(report);improved=True;stale=0;save_checkpoint(RUN/'best_head.pt',{'state_dict':head.state_dict(),'source_sha256':sha,'pretrained_sha256':weight_sha,'epoch':ep+1});write_json(RUN/'best_validation.json',report)
  else:stale+=1
  rows.append({'epoch':ep+1,'phase':'head_refinement_dense_transfer','loss':total/len(y),'accuracy':correct/len(y),'val_loss':report['negative_log_likelihood'],'val_accuracy':report['accuracy'],'val_macro_f1':c['macro avg']['f1-score'],'val_disgust_recall':c['disgust']['recall'],'val_disgust_f1':c['disgust']['f1-score'],'val_balanced_accuracy':c['macro avg']['recall'],'seconds':time.perf_counter()-t,'saved_new_best':ok});write_json(RUN/'training_history.json',rows);print(json.dumps(rows[-1]),flush=True);schedule.step()
  save_checkpoint(RUN/'latest_head.pt',{'state_dict':head.state_dict(),'source_sha256':sha,'pretrained_sha256':weight_sha,'epoch':ep+1})
  if stale>=8:break
 write_json(RUN/'completion.json',{'completed_additional_epochs':len(rows),'validation_improved':improved,'best_score':best,'training':'ONLY new residual classifier head; both encoders frozen','test_accessed':False,'synthetic':False})
if __name__=='__main__':main()
