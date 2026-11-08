"""Direct-label frozen-feature classifier; validation-only probability combination."""
import json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from scripts.colab_common import digest,model_from,metrics,write_json,save_checkpoint
from scripts.train_dense_transfer_probe import transfer_head
from scripts.train_spatial_generalization import score,selected
ROOT=Path(__file__).resolve().parents[1];FEATURES=ROOT/'reports/dense_transfer112_20261108';RUN=ROOT/'reports/direct_dense_head_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 RUN.mkdir(exist_ok=True)
 if (RUN/'completion.json').exists():raise RuntimeError('Completed run preserved')
 torch.set_num_threads(2);torch.manual_seed(907)
 ck=torch.load(ROOT/'reports/spatial_generalization_20261108/baseline.pt',map_location='cpu',weights_only=True);sha=digest(ROOT/'reports/spatial_generalization_20261108/baseline.pt');manifest_sha=digest(ROOT/'reports/fer_training_manifest.json');pre=read(FEATURES/'pretrained_provenance.json');temp=ck.get('inference',{}).get('temperature',1.);mixtures=[.25,.5,.75]
 protocol={'method':'Train DenseNet feature head DIRECTLY against FER TRAIN labels, independently of the overconfident FusDenseNet logits. Frozen feature extractors. Combine calibrated baseline and direct-head probabilities, not class thresholds.','initial_checkpoint_sha256':sha,'seed':907,'epochs_max':20,'patience':8,'class_weight_power':.5,'label_smoothing':0,'head':'LayerNorm1024->128->7, dropout.5','lr':.001,'weight_decay':.01,'probability_mixtures':mixtures,'baseline_temperature':temp,'selection':'Validation accuracy AND disgust recall strictly improve; macro/disgustF1 no lower. Max .5accuracy+.3macroF1+.2disgustF1 across20epochs/3predeclared mixtures.','disclosure':'Third exploratory recipe declared after failed residual-head validation, before ANY new test access. Highly reused validation, not fresh confirmatory evidence; all failed recipes kept.','test':'One final candidate across all completed recipes. No alternate test search.'};write_json(RUN/'protocol.json',protocol)
 base=model_from(ck,torch.device('cpu'));base.eval()
 def cache(split):
  c=torch.load(FEATURES/f'{split}_features.pt',map_location='cpu',weights_only=True);assert c['manifest_sha256']==manifest_sha and c['weight_sha256']==pre['sha256'];b=torch.load(ROOT/f'reports/pruned_head_20261107/{split}_features.pt',map_location='cpu',weights_only=True);assert b['source_sha256']==sha and b['manifest_sha256']==manifest_sha and torch.equal(c['labels'],b['labels'])
  with torch.inference_mode():logits=torch.cat([base.classifier(f) for f in b['features'].split(256)]).clone()
  return c['features'],c['labels'],logits
 x,y,bl=cache('train');vx,vy,bvl=cache('val');initial=metrics(bvl,vy);write_json(RUN/'initial_validation.json',initial);bp=(bvl/temp).softmax(1);head=transfer_head();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20);counts=torch.bincount(y,minlength=7).float();w=(counts.sum()/(7*counts)).sqrt();w/=w.mean();loader=DataLoader(TensorDataset(x,y),batch_size=256,shuffle=True);rows=[];best=score(initial);improved=False;stale=0
 for ep in range(20):
  t=time.perf_counter();head.train();loss_sum=0.;correct=0
  for f,target in loader:
   opt.zero_grad(set_to_none=True);out=head(f);loss=F.cross_entropy(out,target,weight=w);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5.);opt.step();loss_sum+=float(loss.detach())*len(target);correct+=int((out.argmax(1)==target).sum())
  head.eval()
  with torch.no_grad():hl=head(vx)
  standalone=metrics(hl,vy);write_json(RUN/f'standalone_validation_{ep+1:02d}.json',standalone);epoch_improved=False;options=[];best_current=None
  for mix in mixtures:
   combined=((1-mix)*bp+mix*hl.softmax(1)).clamp_min(1e-12).log();r=metrics(combined,vy);write_json(RUN/f'validation_{ep+1:02d}_mix{int(mix*100)}.json',r);options.append((score(r),mix,r))
   if selected(r,initial) and score(r)>best:
    best=score(r);improved=True;epoch_improved=True;best_current=(best,mix,r);save_checkpoint(RUN/'best_head.pt',{'state_dict':head.state_dict(),'source_sha256':sha,'pretrained_sha256':pre['sha256'],'epoch':ep+1,'transfer_mix':mix,'transfer_base_temperature':temp});write_json(RUN/'best_validation.json',r)
  _,mix,r=best_current or max(options,key=lambda x:x[0]);c=r['classification_report'];rows.append({'epoch':ep+1,'phase':'head_refinement_dense_probability','loss':loss_sum/len(y),'accuracy':correct/len(y),'val_loss':r['negative_log_likelihood'],'val_accuracy':r['accuracy'],'val_macro_f1':c['macro avg']['f1-score'],'val_disgust_recall':c['disgust']['recall'],'val_disgust_f1':c['disgust']['f1-score'],'val_balanced_accuracy':c['macro avg']['recall'],'transfer_mix':mix,'standalone_val_accuracy':standalone['accuracy'],'standalone_val_disgust_recall':standalone['classification_report']['disgust']['recall'],'seconds':time.perf_counter()-t,'saved_new_best':epoch_improved});write_json(RUN/'training_history.json',rows);print(json.dumps(rows[-1]),flush=True);sched.step();stale=0 if epoch_improved else stale+1
  save_checkpoint(RUN/'latest_head.pt',{'state_dict':head.state_dict(),'epoch':ep+1})
  if stale>=8:break
 write_json(RUN/'completion.json',{'completed_additional_epochs':len(rows),'validation_improved':improved,'best_score':best,'test_accessed':False,'synthetic':False,'training':'Only the new direct-label residual-feature classifier; both feature extractors frozen'})
if __name__=='__main__':main()
