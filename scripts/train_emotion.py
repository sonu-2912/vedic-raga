import argparse,json,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from backend.datasets import FER2013
from backend.models import FusDenseNet,FER_CLASSES
from backend.catalog import ROOT
from sklearn.metrics import f1_score,balanced_accuracy_score

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--limit',type=int);p.add_argument('--val-limit',type=int);p.add_argument('--deduplicate',action='store_true');p.add_argument('--denoising',choices=['none','nlm','bilateral','gaussian','hybrid'],default='none');p.add_argument('--vertical-flip',action='store_true');p.add_argument('--no-attention',action='store_true');p.add_argument('--no-dilation',action='store_true');p.add_argument('--output',default='models/fusdense.pt');p.add_argument('--threads',type=int,default=8);p.add_argument('--version',type=int,choices=[1,2],default=2);p.add_argument('--learning-rate',type=float,default=.002);p.add_argument('--max-minutes',type=float,default=60);p.add_argument('--resume',help='Warm-start best weights; optimizer/scheduler restart explicitly');p.add_argument('--patience',type=int,default=6);p.add_argument('--min-delta',type=float,default=.002);a=p.parse_args()
    # Additional knobs deliberately stay explicit in the run manifest.
    torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.set_num_threads(a.threads)
    if a.epochs<1:raise ValueError('epochs must be positive')
    train=FER2013(a.data,'train',True,a.denoising,a.limit,a.vertical_flip);val=FER2013(a.data,'val',False,a.denoising,a.val_limit or a.limit)
    seen=set();duplicates={}
    for name,ds in [('train',train),('val',val)]:
        rows=[];removed=0
        for row,h in zip(ds.rows,ds.fingerprints()):
            if a.deduplicate and h in seen:removed+=1;continue
            seen.add(h);rows.append(row)
        ds.rows=rows;duplicates[name]=removed
    manifest={'dataset':a.data,'seed':42,'args':vars(a),'train_count':len(train),'val_count':len(val),'duplicates_removed':duplicates,'train':[{'file':str(f),'label':y,'pixel_sha256':h} for (f,y),h in zip(train.rows,train.fingerprints())],'val':[{'file':str(f),'label':y,'pixel_sha256':h} for (f,y),h in zip(val.rows,val.fingerprints())]}
    (ROOT/'reports/fer_training_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ['train_count','val_count','duplicates_removed']}),flush=True)
    train.cache_images();val.cache_images()
    architecture={'attention':not a.no_attention,'dilated':not a.no_dilation,'version':a.version}
    model=FusDenseNet(**architecture);resume_provenance=None;offset=0;history=[];best=-1
    if a.resume:
        import hashlib
        resume_path=ROOT/a.resume;ck=torch.load(resume_path,map_location='cpu',weights_only=True)
        if ck.get('synthetic') or ck.get('architecture')!=architecture:raise ValueError('Resume needs a real checkpoint with identical architecture')
        if ck.get('train_count')!=len(train) or ck.get('val_count')!=len(val):raise ValueError('Resume split counts differ')
        model.load_state_dict(ck['state_dict']);best=ck.get('validation_macro_f1',-1)
        previous_history=ROOT/'reports/training_history.json'
        history=json.loads(previous_history.read_text()) if previous_history.exists() else []
        offset=max([r['epoch'] for r in history],default=ck.get('epoch',0))
        resume_provenance={'checkpoint_sha256':hashlib.sha256(resume_path.read_bytes()).hexdigest(),'source_epoch':ck.get('epoch'),'history_offset':offset,'optimizer':'AdamW restarted, not exact optimizer-state continuation'}
        print(json.dumps({'resume':resume_provenance}),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=5e-5)
    counts=torch.bincount(torch.tensor([y for _,y in train.rows]),minlength=7).float()
    weights=(counts.sum()/(7*counts)).sqrt();weights=weights/weights.mean()
    run_start=time.perf_counter();no_improvement=0;patience_best=best
    trainloader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0)
    valloader=DataLoader(val,batch_size=a.batch_size,num_workers=0)
    for epoch in range(a.epochs):
        start=time.perf_counter();cpu_start=time.process_time();model.train();loss_sum=0.;correct=0
        for step,(x,y) in enumerate(trainloader):
            opt.zero_grad();out,recon=model(x,return_reconstruction=True)
            loss=torch.nn.functional.cross_entropy(out,y,weight=weights,label_smoothing=.03)+.05*torch.nn.functional.mse_loss(recon,x)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();loss_sum+=loss.item()*len(y);correct+=(out.argmax(1)==y).sum().item()
            if (step+1)%100==0:print(json.dumps({'epoch':offset+epoch+1,'batch':step+1,'batches':len(trainloader),'running_accuracy':correct/min((step+1)*a.batch_size,len(train))}),flush=True)
        model.eval();vcorrect=0;vloss=0;truth=[];predicted=[]
        with torch.inference_mode():
            for x,y in valloader:
                out=model(x);vcorrect+=(out.argmax(1)==y).sum().item();vloss+=torch.nn.functional.cross_entropy(out,y).item()*len(y)
                truth+=y.tolist();predicted+=out.argmax(1).tolist()
        row={'epoch':offset+epoch+1,'loss':loss_sum/len(train),'accuracy':correct/len(train),'val_loss':vloss/len(val),'val_accuracy':vcorrect/len(val),'val_macro_f1':f1_score(truth,predicted,labels=list(range(7)),average='macro',zero_division=0),'val_balanced_accuracy':balanced_accuracy_score(truth,predicted),'learning_rate':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu_start};history.append(row);print(json.dumps(row),flush=True)
        if row['val_macro_f1']>best:
            best=row['val_macro_f1'];output=ROOT/a.output;output.parent.mkdir(parents=True,exist_ok=True)
            temporary=output.with_suffix('.training.tmp')
            torch.save({'state_dict':model.state_dict(),'classes':FER_CLASSES,'synthetic':False,'pilot':a.limit is not None,'architecture':architecture,'denoising':a.denoising,'epoch':offset+epoch+1,'train_count':len(train),'val_count':len(val),'selection_metric':'validation macro F1','validation_macro_f1':best,'training_args':vars(a),'resume_provenance':resume_provenance},temporary)
            temporary.replace(output)
        schedule.step()
        (ROOT/'reports/training_history.json').write_text(json.dumps(history,indent=2))
        if row['val_macro_f1']>patience_best+a.min_delta:patience_best=row['val_macro_f1'];no_improvement=0
        else:no_improvement+=1
        if a.patience>0 and no_improvement>=a.patience:
            print(json.dumps({'early_stop':'validation macro F1 plateau','patience':a.patience,'best_macro_f1':best}),flush=True);break
        if time.perf_counter()-run_start>a.max_minutes*60:break
    (ROOT/'reports/training_history.json').write_text(json.dumps(history,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(9,3.5))
    for key in ['loss','val_loss']:ax[0].plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    for key in ['accuracy','val_accuracy']:ax[1].plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    for axis in ax: axis.legend();axis.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(ROOT/'reports/training_curves.png',dpi=150)
if __name__=='__main__':main()
