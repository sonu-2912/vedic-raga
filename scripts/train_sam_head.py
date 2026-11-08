"""Frozen-feature SAM classifier trial; no test access and no architecture change."""
import argparse,copy,json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,TensorDataset
from scripts.colab_common import digest,model_from,metrics,save_checkpoint,write_json
from scripts.train_consistency import teacher_kl
from scripts.train_deep_supervision import qualifies,metric_vector
from scripts.train_pruned_head import folded_head
ROOT=Path(__file__).resolve().parents[1]

def sam_step(head,optimizer,objective,rho=.05):
    rng=torch.get_rng_state();optimizer.zero_grad(set_to_none=True);first,logits=objective()
    if not torch.isfinite(first):raise RuntimeError('Nonfinite first SAM loss')
    first.backward();parameters=[p for p in head.parameters() if p.grad is not None]
    norm=torch.stack([p.grad.detach().norm() for p in parameters]).norm()
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite SAM gradient')
    original=[p.detach().clone() for p in parameters]
    with torch.no_grad():
        for p in parameters:p.add_(p.grad*(rho/(norm+1e-12)))
    try:
        torch.set_rng_state(rng);optimizer.zero_grad(set_to_none=True);second,_=objective()
        if not torch.isfinite(second):raise RuntimeError('Nonfinite second SAM loss')
        second.backward()
    finally:
        with torch.no_grad():
            for p,before in zip(parameters,original):p.copy_(before)
    torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
    return float(first.detach()),logits.detach(),float(second.detach())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(521)
    run=Path(a.run_dir);protocol=json.loads((run/'protocol.json').read_text());source=ROOT/'reports/deep_supervision_20261107/baseline.pt'
    if (run/'best.pt').exists() or (run/'completion.json').exists():raise RuntimeError('Existing trial preserved')
    ck=torch.load(source,map_location='cpu',weights_only=True);network=model_from(ck,torch.device('cpu'));network.eval();sha=digest(source);assert sha==protocol['initial_checkpoint_sha256']
    features={}
    for split in ['train','val']:
        cached=torch.load(ROOT/f'reports/pruned_head_20261107/{split}_features.pt',map_location='cpu',weights_only=True);assert cached['source_sha256']==sha and cached['manifest_sha256']==digest(ROOT/'reports/fer_training_manifest.json');features[split]=(cached['features'],cached['labels'])
    teacher=torch.load(ROOT/'reports/consistency_long_20261107/training_teacher.pt',map_location='cpu',weights_only=True);assert teacher['source_sha256']==sha and teacher['manifest_sha256']==digest(ROOT/'reports/fer_training_manifest.json') and not teacher['teacher_horizontal_flip']
    x,y=features['train'];v,vy=features['val'];assert torch.equal(y,teacher['labels']);mean=x.mean(0);std=x.std(0).clamp_min(.01);head=copy.deepcopy(network.classifier)
    with torch.no_grad():w=head[0].weight.clone();head[0].weight.mul_(std);head[0].bias.add_(w@mean)
    x=(x-mean)/std;vn=(v-mean)/std;head.eval()
    with torch.no_grad():initial=metrics(head(vn),vy)
    write_json(run/'initial_validation.json',initial)
    for module in head.modules():
        if isinstance(module,torch.nn.Dropout):module.p=.45
    settings={**protocol,'device':'cpu','manifest_sha256':digest(ROOT/'reports/fer_training_manifest.json'),'dataset_sha256':json.loads((ROOT/'reports/pruned_head_20261107/settings.json').read_text())['dataset_sha256'],'seed':521,'threads':a.threads};write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',json.loads((ROOT/'reports/fer_training_manifest.json').read_text()))
    counts=torch.bincount(y,minlength=7).float();weights=(counts.sum()/(7*counts)).pow(.25);weights/=weights.mean();optimizer=torch.optim.AdamW(head.parameters(),lr=.0001,weight_decay=.003);loader=DataLoader(TensorDataset(x,y,teacher['logits']),batch_size=256,shuffle=True)
    old=json.loads((ROOT/'reports/training_history.json').read_text());offset=max(r['epoch'] for r in old);history=[];best=.35*initial['accuracy']+.65*initial['classification_report']['macro avg']['f1-score'];stale=0;improved=False
    save_checkpoint(run/'best.pt',{**ck,'inference':{},'colab_settings':settings,'colab_improved':False})
    for epoch in range(40):
        began=time.perf_counter();head.train();total=0.;sharp=0.;correct=0
        for xx,yy,tt in loader:
            def objective():
                out=head(xx);return torch.nn.functional.cross_entropy(out,yy,weight=weights,label_smoothing=.02)+.1*teacher_kl(out,tt,yy,2.),out
            loss,out,perturbed=sam_step(head,optimizer,objective);total+=loss*len(yy);sharp+=(perturbed-loss)*len(yy);correct+=int((out.argmax(1)==yy).sum())
        head.eval()
        with torch.no_grad():out=head(vn)
        report=metrics(out,vy);vector=metric_vector(report);score=.35*vector['accuracy']+.65*vector['macro_f1'];eligible=qualifies(report,initial);selected=eligible and score>best
        if selected:
            folded=folded_head(head,mean,std)
            with torch.no_grad():restored=folded(v)
            error=float((out-restored).abs().max());assert error<2e-4 and torch.equal(out.argmax(1),restored.argmax(1));network.classifier.load_state_dict(folded.state_dict())
            base={k:value for k,value in ck.items() if k not in ['state_dict','inference','calibration']}
            save_checkpoint(run/'best.pt',{**base,'state_dict':network.state_dict(),'inference':{},'epoch':offset+epoch+1,'validation_macro_f1':vector['macro_f1'],'colab_settings':settings,'colab_improved':True,'head_refinement':settings,'training_stage':'classifier_sharpness_finetune','candidate_type':'SAM512','normalization_fold_max_error':error});write_json(run/'best_validation.json',report);best=score;stale=0;improved=True
        else:stale+=1
        row={'epoch':offset+epoch+1,'phase':'head_refinement_sam512','loss':total/len(y),'sam_loss_increase':sharp/len(y),'accuracy':correct/len(y),'val_loss':report['negative_log_likelihood'],'val_accuracy':vector['accuracy'],'val_macro_f1':vector['macro_f1'],'val_balanced_accuracy':vector['macro_recall'],'val_macro_precision':vector['macro_precision'],'val_disgust_precision':vector['disgust_precision'],'val_disgust_recall':vector['disgust_recall'],'val_disgust_f1':vector['disgust_f1'],'eligible':eligible,'saved_new_best':selected,'seconds':time.perf_counter()-began};history.append(row);write_json(run/f'validation_{epoch+1:02d}_SAM.json',report);write_json(run/'training_history.json',old+history);print(json.dumps(row),flush=True)
        if stale>=10:break
    save_checkpoint(run/'last_head.pt',{'state_dict':head.state_dict(),'training_mean':mean,'training_std':std,'source_sha256':sha,'validation_qualified':improved})
    write_json(run/'completion.json',{'completed_additional_epochs':len(history),'validation_improved':improved,'best_score':best,'test_accessed':False,'synthetic':False,'parameters_trained':'Classifier only, SAM/AdamW; backbone frozen and architecture unchanged'})
if __name__=='__main__':main()
