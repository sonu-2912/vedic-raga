"""Bounded synthetic mechanism checks. Never saves deployable emotion weights."""
import io,json,tempfile,time,os
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from backend.models import FusDenseNet,FDCF,FER_CLASSES
from backend.catalog import ROOT
from backend.signals import sample_signals,eeg_features,hrv_features
from backend.storage import init_db,connection
from backend.recommender import recommend
from backend.emotion import predict,load_model

def main():
    torch.manual_seed(42);torch.set_num_threads(2);rng=np.random.default_rng(42)
    # Seven artificial textures with independent random noise. No human faces.
    def generate(n):
        yy,xx=np.mgrid[:48,:48];xs=[];ys=[]
        for label in range(7):
            for j in range(n):
                a=.2+.6*(np.sin((label+1)*xx*.15+label)+np.cos((label+1)*yy*.12)>0)/2
                a=np.clip(a+rng.normal(0,.03,(48,48)),0,1)
                xs.append(a);ys.append(label)
        return torch.tensor(np.array(xs),dtype=torch.float32)[:,None],torch.tensor(ys)
    x,y=generate(3);xt,yt=generate(2);model=FusDenseNet();opt=torch.optim.Adam(model.parameters(),lr=.003);history=[]
    start=time.perf_counter()
    for step in range(16):
        model.train();opt.zero_grad();out,recon=model(x,return_reconstruction=True)
        loss=torch.nn.functional.cross_entropy(out,y)+.1*torch.nn.functional.mse_loss(recon,x);loss.backward();opt.step()
        model.eval()
        with torch.inference_mode():test=model(xt);acc=(test.argmax(1)==yt).float().mean().item()
        history.append({'step':step+1,'loss':loss.item(),'synthetic_holdout_accuracy':acc})
    with torch.inference_mode():pred=model(xt).argmax(1).numpy()
    truth=yt.numpy()
    pr,re,f1,_=precision_recall_fscore_support(truth,pred,average='macro',zero_division=0)
    cm=confusion_matrix(truth,pred,labels=list(range(7)))
    report={'kind':'SYNTHETIC ENGINEERING TEST ONLY - artificial textures, not FER2013 or faces','seed':42,'train_n':len(x),'test_n':len(xt),'steps':16,'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'accuracy':accuracy_score(truth,pred),'macro_precision':pr,'macro_recall':re,'macro_f1':f1,'confusion_matrix':cm.tolist(),'curves':history}
    (ROOT/'reports/synthetic_emotion_smoke.json').write_text(json.dumps(report,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(6,5));ConfusionMatrixDisplay(cm,display_labels=[f'T{i}' for i in range(7)]).plot(ax=ax,cmap='Greens',colorbar=False);ax.set_title('Synthetic texture classes only (n=14)');fig.tight_layout();fig.savefig(ROOT/'reports/synthetic_confusion_matrix.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(9,3.5));ax[0].plot([h['step'] for h in history],[h['loss'] for h in history],color='#38664c');ax[0].set_ylabel('CE + 0.1 MSE');ax[1].plot([h['step'] for h in history],[h['synthetic_holdout_accuracy'] for h in history],color='#af794b');ax[1].set_ylabel('Synthetic holdout accuracy')
    for a in ax:a.set_xlabel('Optimization step')
    fig.suptitle('FusDenseNet engineering smoke test, artificial textures');fig.tight_layout();fig.savefig(ROOT/'reports/synthetic_training_curves.png',dpi=140);plt.close(fig)
    s=sample_signals();e=eeg_features(s['eeg']['samples'],sample=True);h=hrv_features(s['hrv']['rr_ms'],sample=True)
    (ROOT/'reports/sample_signal_results.json').write_text(json.dumps({'eeg':e,'hrv':h},indent=2))
    (ROOT/'data/sample_signals.json').write_text(json.dumps(s,indent=2))
    old=os.environ.get('RAGAMOOD_DB');samples=[]
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['RAGAMOOD_DB']=tmp+'/demo.sqlite3';init_db()
        for emotion in ['sad','happy','neutral']:
            before=recommend('synthetic-demo',emotion,persist=False);r=before['recommendations'][0]
            with connection() as db:db.execute('INSERT INTO feedback(user_id,raga_id,emotion,liked,mood_improvement) VALUES(?,?,?,?,?)',('synthetic-demo',r['id'],emotion,1,2))
            after=recommend('synthetic-demo',emotion,persist=False)
            samples.append({'synthetic_feedback':True,'before':before,'after':after})
        on=recommend('ablation','sad',use_iks=True,persist=False);off=recommend('ablation','sad',use_iks=False,persist=False)
        ablation={'kind':'Engineering ranking sensitivity only, no user relevance labels','iks_on':[r['raga_name'] for r in on['recommendations']],'iks_off':[r['raga_name'] for r in off['recommendations']]}
    if old is None:os.environ.pop('RAGAMOOD_DB',None)
    else:os.environ['RAGAMOOD_DB']=old
    (ROOT/'reports/sample_recommendations.json').write_text(json.dumps(samples,indent=2))
    (ROOT/'reports/iks_sensitivity.json').write_text(json.dumps(ablation,indent=2))
    # Supplied pretrained baseline inference on artificial image checks weights only.
    b=io.BytesIO();Image.fromarray(rng.integers(0,255,(224,224,3),dtype=np.uint8)).save(b,format='PNG')
    start=time.perf_counter();baseline=predict(b.getvalue(),require_face=False)
    baseline.update(input='Synthetic random pixels, not a human face. Inference mechanics only.',seconds=time.perf_counter()-start)
    (ROOT/'reports/navrasa_inference_smoke.json').write_text(json.dumps(baseline,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['curves','confusion_matrix']},indent=2));print('Saved signal, recommendation, ablation and pretrained baseline smoke evidence.')
if __name__=='__main__':main()
