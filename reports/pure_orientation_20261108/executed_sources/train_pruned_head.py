"""Validation-only capacity trial: prune the512-unit MLP to256units.

Backbone is frozen; no test access. Neuron ranking/normalization/teacher targets
use training data only. The smaller head is a disclosed project dimension choice.
"""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from backend.models import FusDenseNet
from scripts.colab_common import fixed_split,check_manifest,digest,model_from,metrics,save_checkpoint,write_json
from scripts.train_consistency import teacher_kl
from scripts.train_deep_supervision import qualifies,metric_vector
ROOT=Path(__file__).resolve().parents[1]

def rank_neurons(head,features,labels,width):
    with torch.no_grad():
        hidden=head[1](head[0](features))
        class_means=torch.stack([hidden[labels==c].mean(0) for c in range(7)])
        importance=(class_means*head[3].weight.abs()).sum(0)
        selected=importance.topk(width).indices.sort().values
        discarded=torch.ones(hidden.shape[1],dtype=torch.bool);discarded[selected]=False
        bias=head[3].bias+(hidden[:,discarded].mean(0)@head[3].weight[:,discarded].T)
    return selected,bias

def folded_head(head,mean,std):
    result=copy.deepcopy(head);result.eval()
    with torch.no_grad():result[0].weight.div_(std);result[0].bias.sub_(result[0].weight@mean)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--initial',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.manual_seed(419);np.random.seed(419);random.seed(419)
    run=Path(a.run_dir);protocol=json.loads((run/'protocol.json').read_text());assert digest(a.initial)==protocol['initial_checkpoint_sha256']
    if (run/'bundle_manifest.json').exists() or (run/'completion.json').exists() or (run/'best.pt').exists():raise RuntimeError('Existing trial preserved; use a new prospective run.')
    manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text());check_manifest(manifest);ck=torch.load(a.initial,map_location='cpu',weights_only=True);network=model_from(ck,torch.device('cpu'));network.eval();assert network.classifier[0].out_features==512
    settings={**protocol,'dataset_sha256':digest(a.data),'manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'device':'cpu','seed':419,'threads':a.threads};write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',manifest)
    features={}
    for split in ['train','val']:
        cache=run/f'{split}_features.pt'
        if cache.exists():
            saved=torch.load(cache,map_location='cpu',weights_only=True);assert saved['source_sha256']==digest(a.initial) and saved['manifest_sha256']==settings['manifest_sha256'];features[split]=(saved['features'],saved['labels']);continue
        ds=fixed_split(a.data,manifest,split);captured=[];labels=[]
        def hook(module,args):captured.append(args[0].detach().cpu())
        handle=network.classifier.register_forward_pre_hook(hook)
        with torch.no_grad():
            for n,(x,y) in enumerate(DataLoader(ds,batch_size=128)):
                network(x);labels.append(y)
                if (n+1)%25==0:print(json.dumps({'extract':split,'faces':sum(len(v) for v in labels),'total':len(ds)}),flush=True)
        handle.remove();x,y=torch.cat(captured),torch.cat(labels);features[split]=(x,y);save_checkpoint(cache,{'source_sha256':digest(a.initial),'manifest_sha256':settings['manifest_sha256'],'features':x,'labels':y});del ds
    x,y=features['train'];v,vy=features['val'];teacher=copy.deepcopy(network.classifier);teacher.eval()
    with torch.no_grad():teacher_logits=teacher(x);original=teacher(v)
    initial=metrics(original,vy);write_json(run/'initial_validation.json',initial)
    indices,bias=rank_neurons(teacher,x,y,256);mean=x.mean(0);std=x.std(0).clamp_min(.01)
    head=nn.Sequential(nn.Linear(3276,256),nn.ReLU(),nn.Dropout(.5),nn.Linear(256,7))
    with torch.no_grad():
        old=teacher[0].weight[indices].clone();head[0].weight.copy_(old*std);head[0].bias.copy_(teacher[0].bias[indices]+old@mean);head[3].weight.copy_(teacher[3].weight[:,indices]);head[3].bias.copy_(bias)
    write_json(run/'neuron_ranking.json',{'training_only':True,'selected_indices':indices.tolist(),'width_before':512,'width_after':256,'validation_labels_used_for_ranking':False})
    x=(x-mean)/std;vn=(v-mean)/std
    counts=torch.bincount(y,minlength=7).float();weights=(counts.sum()/(7*counts)).pow(.25);weights/=weights.mean()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.003);loader=DataLoader(TensorDataset(x,y,teacher_logits),batch_size=256,shuffle=True)
    history=json.loads((ROOT/'reports/training_history.json').read_text());offset=max(r['epoch'] for r in history);new=[];best=.35*initial['accuracy']+.65*initial['classification_report']['macro avg']['f1-score'];stale=0;improved=False
    save_checkpoint(run/'best.pt',{**ck,'inference':{},'colab_settings':settings,'colab_improved':False})
    for epoch in range(40):
        began=time.perf_counter();head.train();total=0.;correct=0
        for xx,yy,tt in loader:
            optimizer.zero_grad(set_to_none=True);out=head(xx);loss=torch.nn.functional.cross_entropy(out,yy,weight=weights,label_smoothing=.02)+.1*teacher_kl(out,tt,yy,2.)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(yy);correct+=int((out.detach().argmax(1)==yy).sum())
        head.eval()
        with torch.no_grad():out=head(vn)
        report=metrics(out,vy);vector=metric_vector(report);score=.35*vector['accuracy']+.65*vector['macro_f1'];eligible=qualifies(report,initial);selected=eligible and score>best
        if selected:
            folded=folded_head(head,mean,std)
            with torch.no_grad():restored=folded(v)
            error=float((out-restored).abs().max());assert error<2e-4 and torch.equal(out.argmax(1),restored.argmax(1))
            arch={**ck['architecture'],'head_width':256};candidate=FusDenseNet(**arch);state=candidate.state_dict()
            for name,tensor in ck['state_dict'].items():
                if not name.startswith('classifier.'):state[name]=tensor
            candidate.load_state_dict(state);candidate.classifier.load_state_dict(folded.state_dict())
            base={k:value for k,value in ck.items() if k not in ['state_dict','inference','calibration']}
            save_checkpoint(run/'best.pt',{**base,'state_dict':candidate.state_dict(),'architecture':arch,'inference':{},'epoch':offset+epoch+1,'validation_macro_f1':vector['macro_f1'],'colab_settings':settings,'colab_improved':True,'head_refinement':settings,'training_stage':'classifier_capacity_pruning','candidate_type':'pruned256','normalization_fold_max_error':error});write_json(run/'best_validation.json',report);best=score;stale=0;improved=True
        else:stale+=1
        row={'epoch':offset+epoch+1,'phase':'head_refinement_pruned256','loss':total/len(y),'accuracy':correct/len(y),'val_loss':report['negative_log_likelihood'],'val_accuracy':vector['accuracy'],'val_macro_f1':vector['macro_f1'],'val_balanced_accuracy':vector['macro_recall'],'val_macro_precision':vector['macro_precision'],'val_disgust_precision':vector['disgust_precision'],'val_disgust_recall':vector['disgust_recall'],'val_disgust_f1':vector['disgust_f1'],'eligible':eligible,'saved_new_best':selected,'seconds':time.perf_counter()-began};new.append(row);write_json(run/f'validation_{epoch+1:02d}_pruned.json',report);write_json(run/'training_history.json',history+new);print(json.dumps(row),flush=True)
        if stale>=10:break
    write_json(run/'completion.json',{'completed_additional_epochs':len(new),'validation_improved':improved,'best_score':best,'test_accessed':False,'synthetic':False,'parameters_trained':'Pruned256-unit classifier only; backbone byte-identical','inference_architecture_change':'Only classifier width512→256; original FusDenseNet mechanisms retained'})
if __name__=='__main__':main()
