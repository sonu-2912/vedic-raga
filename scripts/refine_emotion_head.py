"""Train the existing FusDenseNet MLP on frozen, train-normalized features.

Normalization is folded into the existing first Linear layer before export.
The deployed architecture and parameter count are unchanged. No test access.
"""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import classification_report
from scripts.colab_common import fixed_split,check_manifest,digest,model_from,save_checkpoint,write_json
from backend.models import FER_CLASSES

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--initial',default='models/fusdense.pt');p.add_argument('--initial-history',default='reports/training_history.json');p.add_argument('--epochs',type=int,default=40);p.add_argument('--threads',type=int,default=2);p.add_argument('--weight-power',type=float,default=.5);p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--feature-flips',action='store_true');p.add_argument('--priority-disgust',action='store_true');p.add_argument('--selection-accuracy',action='store_true');a=p.parse_args()
    torch.set_num_threads(a.threads);torch.manual_seed(67);np.random.seed(67);random.seed(67)
    run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=True)
    if (run/'best.pt').exists():raise ValueError('Use a new run directory')
    manifest=json.loads(Path('reports/fer_training_manifest.json').read_text());check_manifest(manifest)
    ck=torch.load(a.initial,map_location='cpu',weights_only=True);model=model_from(ck,torch.device('cpu'));model.eval()
    settings={'device':'cpu','dataset_sha256':digest(a.data),'manifest_sha256':digest('reports/fer_training_manifest.json'),'initial_checkpoint_sha256':digest(a.initial),
      'method':'Frozen backbone; existing MLP only; training-fitted feature normalization folded into Linear weights','seed':67,'epochs':a.epochs,'lr':a.learning_rate,'weight_power':a.weight_power,'batch_size':256,
      'selection':'Maximize .75 macro F1 + .25 disgust F1 with validation accuracy, macro F1 and disgust F1 no lower than initial' if a.priority_disgust else 'Maximum validation macro F1 among epochs with validation accuracy >= initial checkpoint; no test access','normalization':'training features only','architecture':ck['architecture'],'feature_flips':a.feature_flips,'priority_disgust':a.priority_disgust,'selection_accuracy':a.selection_accuracy}
    if a.selection_accuracy:settings['selection']='Maximum validation accuracy subject to no lower initial macro F1 and disgust F1; training features only'
    write_json(run/'settings.json',settings);write_json(run/'fer_training_manifest.json',manifest)
    features={}
    for split in ['train','val']:
        ds=fixed_split(a.data,manifest,split);captured=[];labels=[]
        def capture(module,args):captured.append(args[0].detach().cpu())
        handle=model.classifier.register_forward_pre_hook(capture)
        with torch.inference_mode():
            for index,(x,y) in enumerate(DataLoader(ds,batch_size=128)):
                model(x);labels.append(y)
                if split=='train' and a.feature_flips:model(x.flip(-1));labels.append(y)
                if index%25==0:print(json.dumps({'extract':split,'batch':index,'batches':int(np.ceil(len(ds)/128))}),flush=True)
        handle.remove();features[split]=(torch.cat(captured).clone(),torch.cat(labels));del ds
    x,y=features['train'];v,vy=features['val'];mean=x.mean(0);std=x.std(0).clamp_min(.01)
    head=copy.deepcopy(model.classifier)
    with torch.no_grad():
        w=head[0].weight.clone();head[0].weight.mul_(std);head[0].bias.add_(w@mean)
    x=(x-mean)/std;v=(v-mean)/std
    def evaluate():
        head.eval()
        with torch.inference_mode():out=head(v)
        return classification_report(vy,out.argmax(1),labels=list(range(7)),target_names=FER_CLASSES,output_dict=True,zero_division=0),float(torch.nn.functional.cross_entropy(out,vy))
    initial,initial_loss=evaluate();floor=initial['accuracy'];best=initial['macro avg']['f1-score'];best_weights=copy.deepcopy(head.state_dict());selected=ck['epoch'];stale=0
    initial_f1=best;initial_disgust=initial['disgust']['f1-score'];best_score=initial['accuracy'] if a.selection_accuracy else (.75*best+.25*initial_disgust if a.priority_disgust else best)
    write_json(run/'initial_validation.json',initial)
    counts=torch.bincount(y,minlength=7).float();weight=(counts.sum()/(7*counts)).pow(a.weight_power);weight/=weight.mean()
    opt=torch.optim.AdamW(head.parameters(),lr=a.learning_rate,weight_decay=.001);loader=DataLoader(TensorDataset(x,y),batch_size=256,shuffle=True)
    old=json.loads(Path(a.initial_history).read_text());history=[];offset=max(max(r['epoch'] for r in old),ck['epoch'])
    for epoch in range(a.epochs):
        start=time.perf_counter();head.train();loss_sum=0;correct=0
        for xx,yy in loader:
            opt.zero_grad(set_to_none=True);out=head(xx);loss=torch.nn.functional.cross_entropy(out,yy,weight=weight,label_smoothing=.02);loss.backward();opt.step();loss_sum+=float(loss)*len(yy);correct+=int((out.argmax(1)==yy).sum())
        report,val_loss=evaluate();f1=report['macro avg']['f1-score'];candidate_score=report['accuracy'] if a.selection_accuracy else (.75*f1+.25*report['disgust']['f1-score'] if a.priority_disgust else f1)
        improved=report['accuracy']>=floor and candidate_score>best_score and (not (a.priority_disgust or a.selection_accuracy) or (f1>=initial_f1 and report['disgust']['f1-score']>=initial_disgust))
        row={'epoch':offset+epoch+1,'phase':'head_refinement_frozen_backbone','loss':loss_sum/len(y),'accuracy':correct/len(y),'val_loss':val_loss,'val_accuracy':report['accuracy'],'val_macro_f1':f1,'val_balanced_accuracy':report['macro avg']['recall'],'val_disgust_recall':report['disgust']['recall'],'val_disgust_f1':report['disgust']['f1-score'],'seconds':time.perf_counter()-start};history.append(row)
        if improved:best=f1;best_score=candidate_score;best_weights=copy.deepcopy(head.state_dict());selected=row['epoch'];stale=0;write_json(run/'best_validation.json',report)
        else:stale+=1
        print(json.dumps({**row,'saved_new_best':improved}),flush=True)
        write_json(run/'training_history.json',old+history)
        if stale>=10:break
    head.load_state_dict(best_weights)
    with torch.no_grad():head[0].weight.div_(std);head[0].bias.sub_(head[0].weight@mean)
    model.classifier.load_state_dict(head.state_dict())
    base={k:value for k,value in ck.items() if k not in ['state_dict','inference','calibration']}
    save_checkpoint(run/'best.pt',{**base,'state_dict':model.state_dict(),'inference':{},'epoch':selected,'validation_macro_f1':best,'colab_settings':settings,'colab_improved':selected!=ck['epoch'],'head_refinement':settings})
    write_json(run/'completion.json',{'completed_additional_epochs':len(history),'best_macro_f1':best,'test_accessed':False,'synthetic':False})
if __name__=='__main__':main()
