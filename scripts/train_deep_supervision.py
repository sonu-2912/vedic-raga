"""Train-only deep supervision of FusDenseNet's late fused feature path.

The deployed inference architecture/state schema is unchanged. No test pixels are
loaded here. Original selected runs remain immutable.
"""
import argparse, copy, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from scripts.colab_common import fixed_split,check_manifest,digest,model_from,predictions,metrics,save_checkpoint,write_json,set_training_mode
from scripts.train_consistency import update_ema
ROOT=Path(__file__).resolve().parents[1]

class FusedCompanion(nn.Module):
    """Companion classifier observes only the588late fused features, not shortcuts."""
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.LayerNorm(588),nn.Linear(588,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,7))
        self.features=None
        self.shortcut_drop=0.
    def capture(self,module,args):
        features=args[0]
        self.features=features[:,:588]
        if self.training and self.shortcut_drop:
            keep=1-self.shortcut_drop
            mask=(torch.rand((len(features),1),device=features.device)<keep).to(features.dtype)/keep
            return (torch.cat([features[:,:588],features[:,588:]*mask],1),)
    def forward(self):
        if self.features is None:raise RuntimeError('Capture a model forward first')
        return self.head(self.features)

def mixed_ce(logits,first,second,lam,weights):
    return lam*F.cross_entropy(logits,first,weight=weights,label_smoothing=.02)+(1-lam)*F.cross_entropy(logits,second,weight=weights,label_smoothing=.02)

def metric_vector(r):
    c=r['classification_report'];m=c['macro avg'];d=c['disgust']
    return {'accuracy':r['accuracy'],'macro_precision':m['precision'],'macro_recall':m['recall'],'macro_f1':m['f1-score'],'disgust_precision':d['precision'],'disgust_recall':d['recall'],'disgust_f1':d['f1-score']}

def qualifies(report,baseline):
    a,b=metric_vector(report),metric_vector(baseline)
    return all(a[k]+1e-12>=b[k] for k in a)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--initial',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--recover',action='store_true');a=p.parse_args()
    run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=True)
    if (run/'bundle_manifest.json').exists():raise RuntimeError('Evaluated run is locked; use a genuinely new protocol/run.')
    protocol=json.loads((run/'protocol.json').read_text());assert digest(a.initial)==protocol['initial_checkpoint_sha256']
    torch.set_num_threads(a.threads);torch.manual_seed(312);np.random.seed(312);random.seed(312)
    manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text());check_manifest(manifest)
    settings={**protocol,'dataset_sha256':digest(a.data),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'threads':a.threads,'seed':312,'device':'cpu'}
    write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',manifest)
    train=fixed_split(a.data,manifest,'train');val=fixed_split(a.data,manifest,'val')
    ck=torch.load(a.initial,map_location='cpu',weights_only=True);student=model_from(ck,torch.device('cpu'));ema=copy.deepcopy(student);ema.requires_grad_(False)
    raw,y=predictions(student,val,torch.device('cpu'),128);initial=metrics(raw,y);write_json(run/'initial_validation.json',initial)
    score=lambda r:.35*r['accuracy']+.65*r['classification_report']['macro avg']['f1-score']
    companion=FusedCompanion();companion.shortcut_drop=protocol['shortcut_dropout'];handle=student.classifier[0].register_forward_pre_hook(companion.capture)
    for layer in student.modules():
        if isinstance(layer,nn.Dropout):layer.p=.4
    groups={'early':[],'late':[],'classifier':[]}
    for name,param in student.named_parameters():
        key='classifier' if name.startswith('classifier.') else 'early' if name.startswith(('stem.','fusion.a.','fusion.b.')) else 'late'
        groups[key].append(param)
    lrs={'early':1e-5,'late':2e-4,'classifier':1e-4,'companion':1e-3}
    optimizer=torch.optim.AdamW([{'params':params,'lr':lrs[name],'group_name':name} for name,params in [*groups.items(),('companion',list(companion.parameters()))]],weight_decay=.001)
    counts=torch.bincount(torch.tensor([y for _,y in train.rows]),minlength=7).float();weights=(counts.sum()/(7*counts)).pow(.25);weights/=weights.mean()
    train.transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomAffine(7,translate=(.04,.04),scale=(.95,1.05),interpolation=transforms.InterpolationMode.BILINEAR),transforms.ColorJitter(brightness=.15,contrast=.15),transforms.ToTensor()])
    loader=DataLoader(train,batch_size=64,shuffle=True,num_workers=0)
    history=json.loads((ROOT/'reports/training_history.json').read_text());offset=max(r['epoch'] for r in history);completed=[];best=score(initial);stale=0;updates=0;start=0;improved_any=False
    base={k:v for k,v in ck.items() if k not in ['state_dict','inference','calibration']};base.update(head_refinement=False,training_stage='full_network_deep_supervision',colab_settings=settings,selection_metric=protocol['selection'])
    if a.recover:
        state=torch.load(run/'latest.pt',map_location='cpu',weights_only=True);assert state['settings']==settings
        student.load_state_dict(state['state_dict']);ema.load_state_dict(state['ema_state']);companion.load_state_dict(state['companion_state']);optimizer.load_state_dict(state['optimizer']);completed=state['history'];best=state['best'];stale=state['stale'];updates=state['updates'];start=len(completed);improved_any=state['validation_improved'];torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);np.random.set_state((state['numpy_rng_name'],np.array(state['numpy_rng_keys'],dtype=np.uint32),state['numpy_rng_pos'],state['numpy_rng_gauss'],state['numpy_rng_cached']))
    else:
        if (run/'latest.pt').exists() or (run/'best.pt').exists():raise RuntimeError('Existing run; recover or choose a new directory.')
        save_checkpoint(run/'best.pt',{**ck,'inference':{},'colab_settings':settings,'colab_improved':False})
    began=time.perf_counter();print(json.dumps({'initial_validation':metric_vector(initial),'train':len(train),'validation':len(val),'test_accessed':False}),flush=True)
    for epoch in range(start,protocol['epochs_max']):
        t=time.perf_counter();set_training_mode(student,True);companion.train();loss_sum=0.;correct=0.;seen=0;ce_sum=0.;aux_sum=0.
        factor=min(1,(epoch+1)/2)*(.1+.9*(1+math.cos(math.pi*epoch/protocol['epochs_max']))/2)
        for group in optimizer.param_groups:group['lr']=lrs[group['group_name']]*factor
        for batch,(x,y) in enumerate(loader):
            if random.random()<.5:
                lam=float(np.random.beta(.2,.2));permutation=torch.randperm(len(y));second=y[permutation];x=lam*x+(1-lam)*x[permutation]
            else:lam=1.;second=y
            optimizer.zero_grad(set_to_none=True);logits=student(x);aux=companion();ce=mixed_ce(logits,y,second,lam,weights);extra=mixed_ce(aux,y,second,lam,weights);loss=ce+.25*extra
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective; last completed checkpoint preserved.')
            loss.backward()
            for name,params in groups.items():torch.nn.utils.clip_grad_norm_(params,5. if name=='classifier' else 1.)
            torch.nn.utils.clip_grad_norm_(companion.parameters(),1.);optimizer.step();update_ema(ema,student,updates);updates+=1
            n=len(y);loss_sum+=float(loss.detach())*n;ce_sum+=float(ce.detach())*n;aux_sum+=float(extra.detach())*n;pred=logits.detach().argmax(1);correct+=float((lam*(pred==y).float()+(1-lam)*(pred==second).float()).sum());seen+=n;companion.features=None
            if (batch+1)%75==0:print(json.dumps({'epoch':offset+epoch+1,'batch':batch+1,'batches':len(loader),'main_loss':ce_sum/seen,'companion_loss':aux_sum/seen}),flush=True)
        companion.eval();candidates=[];improved_epoch=False;selected_kind=None
        for kind,network in [('student',student),('EMA',ema)]:
            out,truth=predictions(network,val,torch.device('cpu'),128);report=metrics(out,truth);vector=metric_vector(report);entry={'kind':kind,**vector,'score':score(report),'eligible':qualifies(report,initial)};candidates.append(entry);write_json(run/f'validation_{epoch+1:02d}_{kind}.json',report)
            if entry['eligible'] and entry['score']>best:
                best=entry['score'];improved_any=True;improved_epoch=True;selected_kind=kind
                save_checkpoint(run/'best.pt',{**base,'state_dict':network.state_dict(),'inference':{},'epoch':offset+epoch+1,'validation_macro_f1':vector['macro_f1'],'candidate_type':kind,'colab_improved':True});write_json(run/'best_validation.json',report)
        companion.features=None
        chosen=next(c for c in candidates if c['kind']==selected_kind) if selected_kind else max(candidates,key=lambda c:c['score']);report=json.loads((run/f'validation_{epoch+1:02d}_{chosen["kind"]}.json').read_text())
        row={'epoch':offset+epoch+1,'phase':'full_network_deep_supervision','loss':loss_sum/seen,'main_loss':ce_sum/seen,'companion_loss':aux_sum/seen,'accuracy':correct/seen,'training_accuracy_definition':'lambda-weighted mixed-label agreement with augmentation, shortcut dropout and student dropout','val_loss':report['negative_log_likelihood'],'val_accuracy':chosen['accuracy'],'val_macro_f1':chosen['macro_f1'],'val_balanced_accuracy':chosen['macro_recall'],'val_macro_precision':chosen['macro_precision'],'val_disgust_precision':chosen['disgust_precision'],'val_disgust_recall':chosen['disgust_recall'],'val_disgust_f1':chosen['disgust_f1'],'candidates':candidates,'seconds':time.perf_counter()-t}
        completed.append(row);stale=0 if improved_epoch else stale+1;npstate=np.random.get_state()
        save_checkpoint(run/'latest.pt',{'state_dict':student.state_dict(),'ema_state':ema.state_dict(),'companion_state':companion.state_dict(),'optimizer':optimizer.state_dict(),'history':completed,'best':best,'stale':stale,'updates':updates,'settings':settings,'validation_improved':improved_any,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng_name':npstate[0],'numpy_rng_keys':npstate[1].astype(np.int64).tolist(),'numpy_rng_pos':npstate[2],'numpy_rng_gauss':npstate[3],'numpy_rng_cached':npstate[4]})
        write_json(run/'training_history.json',history+completed);print(json.dumps(row),flush=True)
        if stale>=protocol['patience'] or time.perf_counter()-began>=protocol['max_minutes']*60:break
    handle.remove();write_json(run/'completion.json',{'completed_additional_epochs':len(completed),'validation_improved':improved_any,'best_score':best,'test_accessed':False,'synthetic':False,'parameters_trained':'Full emotion path plus training-only fused companion; no new inference parameters','budget_check':'Completed epoch boundaries; elapsed time can overshoot'})
if __name__=='__main__':main()
