import argparse,json,time,hashlib
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,balanced_accuracy_score
from backend.catalog import ROOT
from backend.datasets import FER2013
from backend.models import FusDenseNet,FER_CLASSES
from backend.inference import emotion_logits

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',default='models/fusdense.pt');p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--limit',type=int);p.add_argument('--exclude-manifest');p.add_argument('--batch-size',type=int,default=64);p.add_argument('--threads',type=int,default=8);p.add_argument('--output',default='reports/emotion_evaluation');a=p.parse_args()
    torch.set_num_threads(a.threads)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ck.get('synthetic'):raise ValueError('Synthetic checkpoint is not suitable for real facial evaluation')
    if ck.get('classes')!=FER_CLASSES:raise ValueError('Checkpoint class order differs from the FER2013 loader')
    model=FusDenseNet(**ck.get('architecture',{}));model.load_state_dict(ck['state_dict']);model.eval()
    ds=FER2013(a.data,a.split,denoising=ck.get('denoising','none'),limit=a.limit)
    removed=0
    if a.exclude_manifest:
        manifest=json.loads((ROOT/a.exclude_manifest).read_text())
        excluded_splits=['train'] if a.split=='val' else ['train','val']
        seen={r['pixel_sha256'] for name in excluded_splits for r in manifest[name]}
        kept=[]
        for row,h in zip(ds.rows,ds.fingerprints()):
            if h in seen:removed+=1;continue
            seen.add(h);kept.append(row)
        ds.rows=kept
    ds.cache_images()
    ytrue=[];ypred=[];probabilities=[];start=time.perf_counter()
    with torch.inference_mode():
        for x,y in DataLoader(ds,batch_size=a.batch_size,num_workers=0):
            out=emotion_logits(model,x,ck.get('inference',{}));ytrue+=y.tolist();ypred+=out.argmax(1).tolist();probabilities+=out.softmax(1).tolist()
    cm=confusion_matrix(ytrue,ypred,labels=list(range(7)))
    report={'dataset':a.data,'split':a.split,'n':len(ds),'duplicate_pixels_removed':removed,'checkpoint_epoch':ck.get('epoch'),'checkpoint_sha256':hashlib.sha256(open(a.checkpoint,'rb').read()).hexdigest(),'pilot':ck.get('pilot',False),'seconds':time.perf_counter()-start,'synthetic':False,'accuracy':accuracy_score(ytrue,ypred),'balanced_accuracy':balanced_accuracy_score(ytrue,ypred),'architecture':ck.get('architecture',{}),'classification_report':classification_report(ytrue,ypred,labels=list(range(7)),target_names=FER_CLASSES,zero_division=0,output_dict=True),'confusion_matrix':cm.tolist(),'predictions':{'truth':ytrue,'predicted':ypred}}
    probs=np.asarray(probabilities);truth=np.asarray(ytrue);confidence=probs.max(1);correct=np.asarray(ypred)==truth;bins=[]
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(confidence>=lo)&((confidence<hi) if hi<1 else (confidence<=hi));n=int(mask.sum())
        bins.append({'lower':float(lo),'upper':float(hi),'n':n,'accuracy':float(correct[mask].mean()) if n else None,'confidence':float(confidence[mask].mean()) if n else None})
    report.update({'inference':ck.get('inference',{}),'majority_class_baseline_accuracy':float(np.bincount(truth,minlength=7).max()/len(truth)),'negative_log_likelihood':float(-np.log(np.maximum(probs[np.arange(len(truth)),truth],1e-12)).mean()),'brier_score':float(((probs-np.eye(7)[truth])**2).sum(1).mean()),'expected_calibration_error_10_bins':sum(b['n']/len(ds)*abs(b['accuracy']-b['confidence']) for b in bins if b['n']),'reliability_bins':bins})
    prefix=ROOT/a.output;prefix.parent.mkdir(parents=True,exist_ok=True);prefix.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2))
    prefix.with_name(prefix.name+'_manifest').with_suffix('.json').write_text(json.dumps({'args':vars(a),'records':[{'file':str(f),'label':y,'pixel_sha256':h} for (f,y),h in zip(ds.rows,ds.fingerprints())]},indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(7,6));ConfusionMatrixDisplay(cm,display_labels=FER_CLASSES).plot(ax=ax,cmap='Greens',colorbar=False,text_kw={'fontsize':14});ax.tick_params(labelsize=12);ax.set_title(f'Actual {a.split} evaluation (n={len(ds)})',fontsize=14);fig.tight_layout();fig.savefig(prefix.with_suffix('.png'),dpi=150)
if __name__=='__main__':main()
