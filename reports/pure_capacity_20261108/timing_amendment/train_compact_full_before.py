"""Full emotion-path pure FusDense recovery after native-classifier warm-up."""
import argparse,copy,json,math,time,shutil
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from scripts.colab_common import digest,write_json,save_checkpoint,check_manifest,fixed_split,predictions,metrics,set_training_mode
from scripts.train_consistency import update_ema
from scripts.recall_recovery_common import eligible,objective,feasibility
from scripts.compact_fusdense import model_from,capacity_loss
from torchvision import transforms
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
class RecoveryViews(Dataset):
 def __init__(self,ds,reference,seed):self.ds=ds;self.reference=reference;self.seed=seed;self.set_epoch(0);self.augmentation=transforms.Compose([transforms.RandomApply([transforms.RandomAffine(5,translate=(.02,.02),scale=(.95,1.05),interpolation=transforms.InterpolationMode.BILINEAR)],p=.6),transforms.RandomApply([transforms.ColorJitter(brightness=.1,contrast=.1)],p=.5),transforms.ToTensor()])
 def __len__(self):return len(self.ds)
 def set_epoch(self,epoch):
  g=torch.Generator().manual_seed(self.seed+epoch//2);self.flipped=torch.rand(len(self.ds),generator=g)<.5
  if epoch%2:self.flipped=~self.flipped
 def __getitem__(self,i):
  arr,y=self.ds.raw_image(i);flipped=bool(self.flipped[i]);image=Image.fromarray(arr);image=ImageOps.mirror(image) if flipped else image;return self.augmentation(image),y,self.reference[i+len(self.ds) if flipped else i],flipped

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--threads',type=int,default=4);a=ap.parse_args();run=ROOT/a.run_dir
 if (run/'full_completion.json').exists() or (run/'candidate_frozen.json').exists():raise RuntimeError('Completed/frozen run preserved')
 protocol=read(run/'protocol.json');torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(protocol['seed']+1000);current=read(run/'current_validation.json');earlier=read(run/'earlier_validation.json');warm=read(run/'warm_validation.json');use_warm=True;initial_path=run/'warm_start.pt';ck=torch.load(initial_path,map_location='cpu',weights_only=True);recipe=read(run/'warm_recipe.json');manifest=read(ROOT/'reports/fer_training_manifest.json');check_manifest(manifest);settings={**read(run/'settings.json'),'architecture':ck['architecture'],'selected_recipe':recipe};write_json(run/'full_settings.json',settings);dataset=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'train');validation=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'val');cache=torch.load(ROOT/'reports/pure_recall_recovery_v2_20261108/train_reference_logits.pt',map_location='cpu',weights_only=True);assert cache['source_checkpoint_sha256']==protocol['initial_checkpoint_sha256'] and cache['manifest_sha256']==digest(ROOT/'reports/fer_training_manifest.json');assert cache['labels'].tolist()==[r['label'] for r in manifest['train']]*2
 views=RecoveryViews(dataset,cache['logits'],protocol['seed']+400);loader=DataLoader(views,batch_size=64,shuffle=True,num_workers=0);student=model_from(ck,torch.device('cpu'));ema=copy.deepcopy(student);ema.requires_grad_(False)
 for module in student.modules():
  if isinstance(module,nn.Dropout):module.p=.5
 capture={}
 handle=student.classifier.register_forward_pre_hook(lambda module,args:capture.update(features=args[0]))
 head=list(student.classifier.parameters());ids={id(p) for p in head};backbone=[p for p in student.parameters() if id(p) not in ids];optimizer=torch.optim.AdamW([{'params':backbone,'lr':protocol['full_backbone_lr'],'weight_decay':.01},{'params':head,'lr':protocol['full_classifier_lr'],'weight_decay':.1}]);rates=[g['lr'] for g in optimizer.param_groups];weights=torch.tensor(read(ROOT/'reports/pure_orientation_20261108/normalization_audit.json')['class_weights']);weights[[3,5,6]]*=recipe['common_boost'];old_history=read(run/'initial_history.json');head_rows=read(run/'head_history.json');offset=max(r['epoch'] for r in head_rows);rows=[];best=float('-inf');updates=0;began=time.perf_counter();improved=False;diagnostic_best=-float('inf');stale=0
 snapshot=run/'executed_full_sources';snapshot.mkdir(exist_ok=False)
 for name in ['scripts/train_compact_full.py','scripts/compact_fusdense.py','scripts/recall_recovery_common.py','backend/models.py']:shutil.copy2(ROOT/name,snapshot/Path(name).name)
 write_json(run/'full_initialization.json',{'source_checkpoint_sha256':digest(initial_path),'warm_classifier_used':use_warm,'recipe':recipe,'all_emotion_path_parameters_trainable':True,'batchnorm_running_statistics_frozen':True,'external_encoder':False,'reference':'Cached current pure classifier on TRAIN only; discarded at inference','retention_mode':'TRAIN-only same-current-pure correct disgust-vs-rest boundary; deterministic retention branch', 'augmentation':'Complementary mirrors plus mild affine/brightness/contrast; TRAIN only','source_hashes':{p.name:digest(p) for p in snapshot.iterdir()}})
 for ep in range(protocol['full_epochs_max']):
  tick=time.perf_counter();cpu=time.process_time();views.set_epoch(ep);set_training_mode(student,True);seen=0;correct=0;total=0.;ce_total=0.;ret_total=0.
  for batch,(x,y,reference,mirrored) in enumerate(loader):
   warmup=min(1.,(updates+1)/len(loader));decay=.2+.8*.5*(1+math.cos(math.pi*ep/protocol['full_epochs_max']))
   for group,lr in zip(optimizer.param_groups,rates):group['lr']=lr*warmup*decay
   optimizer.zero_grad(set_to_none=True);out=student(x);features=capture['features'];deterministic=student.classifier[3](student.classifier[1](student.classifier[0](features)));loss,ce,ret=capacity_loss(out,deterministic,y,reference,weights,protocol['retention_weight'],protocol['label_smoothing'])
   if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite training loss; partial epoch not claimed')
   loss.backward();nn.utils.clip_grad_norm_(backbone,2.);nn.utils.clip_grad_norm_(head,5.);optimizer.step();update_ema(ema,student,updates);updates+=1;seen+=len(y);correct+=int((out.argmax(1)==y).sum());total+=float(loss.detach())*len(y);ce_total+=float(ce.detach())*len(y);ret_total+=float(ret.detach())*len(y)
   if (batch+1)%100==0:print(json.dumps({'epoch':offset+ep+1,'batch':batch+1,'batches':len(loader),'loss':total/seen}),flush=True)
  options=[]
  for kind,network in [('student',student),('EMA',ema)]:
   logits,y=predictions(network,validation,torch.device('cpu'));r=metrics(logits,y);write_json(run/f'full_validation_{ep+1:02d}_{kind}.json',r);ok=eligible(r,current,earlier);options.append((feasibility(r,current,earlier),r,kind))
   if ok and objective(r)>best:
    base={k:v for k,v in ck.items() if k not in ['state_dict','inference','calibration','colab_settings']};save_checkpoint(run/'full_best.pt',{**base,'state_dict':network.state_dict(),'inference':{},'epoch':offset+ep+1,'colab_settings':{**settings,'full_recipe':recipe},'training_stage':'pure_capacity_full_network','training_reference':'Current pure classifier TRAIN disgust-vs-rest boundary retention','candidate_type':kind,'colab_improved':True,'head_refinement':False,'selection_metric':protocol['selection']});write_json(run/'full_best_validation.json',r);best=objective(r);improved=True
  _,r,kind=max(options,key=lambda o:o[0]);c=r['classification_report'];row={'epoch':offset+ep+1,'phase':'pure_capacity_full_network','kind':kind,'loss':total/seen,'CE':ce_total/seen,'retention':ret_total/seen,'accuracy':correct/seen,'val_loss':r['negative_log_likelihood'],'val_accuracy':r['accuracy'],'val_macro_f1':c['macro avg']['f1-score'],'val_disgust_recall':c['disgust']['recall'],'val_disgust_f1':c['disgust']['f1-score'],'val_happy_recall':c['happy']['recall'],'val_surprise_recall':c['surprise']['recall'],'val_neutral_recall':c['neutral']['recall'],'eligible':eligible(r,current,earlier),'train_source_images_seen':seen,'seconds':time.perf_counter()-tick,'cpu_seconds':time.process_time()-cpu};rows.append(row);write_json(run/'full_history.json',rows);save_checkpoint(run/'full_latest.pt',{'state_dict':student.state_dict(),'ema_state':ema.state_dict(),'optimizer':optimizer.state_dict(),'history':rows,'settings':settings});print(json.dumps(row),flush=True)
  score=max(item[0] for item in options);stale=0 if score>diagnostic_best+1e-4 else stale+1;diagnostic_best=max(diagnostic_best,score)
  if (len(rows)>=protocol['full_min_epochs'] and stale>=protocol['full_patience']) or time.perf_counter()-began>=protocol['full_minutes']*60:break
 handle.remove()
 write_json(run/'full_completion.json',{'completed_full_epochs':len(rows),'validation_improved':improved,'test_accessed':False,'external_encoder':False,'inference_hybrid':False})
if __name__=='__main__':main()
