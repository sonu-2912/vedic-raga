"""Fixed DENS protocol: disjoint people AND movie clips in every outer fold."""
import hashlib,json,warnings
from pathlib import Path
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from backend.physiology_features import paired_features,EEG_NAMES,CARDIAC_NAMES
from scripts.colab_common import write_json
ROOT=Path(__file__).resolve().parents[1]

def features(t):
    data=np.fromfile(t['file'],dtype='<f4').reshape(t['sample_count'],t['channels']).astype(float)
    if not np.isfinite(data).all():raise ValueError('Nonfinite physical recording')
    eeg=data[:,t['eeg_indices']];std=eeg.std(0);keep=std>1e-8
    if keep.sum()<16:raise ValueError('Too few nonflat EEG channels')
    eeg=(eeg[:,keep]-eeg[:,keep].mean(0))/std[keep]
    fs=t['sampling_rate'];ecg=data[:,t['ecg_index']]
    if ecg.std()<1e-8:raise ValueError('Flat ECG')
    ecg=(ecg-ecg.mean())/ecg.std()
    # ECG beat detection shares the exact same 60s source interval as EEG.
    _,cardiac,quality=paired_features(eeg[:,0],ecg,fs)
    filt=signal.sosfiltfilt(signal.butter(4,[1,40],btype='bandpass',fs=fs,output='sos'),eeg,axis=0)
    f,p=signal.welch(filt,fs=fs,nperseg=int(4*fs),axis=0)
    def area(a,b):
        k=(f>=a)&(f<b);return np.trapz(p[k],f[k],axis=0)
    total=np.maximum(area(1,40),1e-18)
    spectral=p[(f>=1)&(f<40)];spectral/=np.maximum(spectral.sum(0),1e-18)
    mobility=np.sqrt(np.var(np.diff(filt,axis=0),axis=0)/np.maximum(np.var(filt,axis=0),1e-18))
    dm=np.sqrt(np.var(np.diff(filt,n=2,axis=0),axis=0)/np.maximum(np.var(np.diff(filt,axis=0),axis=0),1e-18))
    values=np.array([area(a,b)/total for a,b in [(4,8),(8,13),(13,30),(1,4),(30,40)]]+[(-spectral*np.log(spectral+1e-18)).sum(0)/np.log(len(spectral)),mobility,dm/np.maximum(mobility,1e-18)])
    v=np.r_[np.median(values,axis=1),np.subtract(*np.percentile(values,[75,25],axis=1))]
    if not np.isfinite(np.r_[v,cardiac]).all():raise ValueError('Nonfinite features')
    return {**{k:t[k] for k in ['id','subject','stimulus','valence','label']},'eeg':v.tolist(),'cardiac':cardiac,'quality':{**quality,'nonflat_eeg_channels':int(keep.sum()),'physically_scaled_input':True},'source_sha256':hashlib.sha256(Path(t['file']).read_bytes()).hexdigest()}

def score(y,p):
    c=classification_report(y,p,labels=[0,1],target_names=['low','high'],output_dict=True,zero_division=0)
    return {'n':len(y),'accuracy':float(np.mean(y==p)),'macro_precision':c['macro avg']['precision'],'macro_recall':c['macro avg']['recall'],'macro_f1':c['macro avg']['f1-score'],'classification_report':c,'confusion_matrix':confusion_matrix(y,p,labels=[0,1]).tolist()}

def main():
    if not (ROOT/'reports/dens_download_manifest.json').exists():
        raise SystemExit('DENS evaluation incomplete: no verified signal-range manifest. Resolve the source header/FDT mismatches recorded in dens_access_outcome.json before this planned benchmark can run. No performance is claimed.')
    plan=json.loads((ROOT/'reports/dens_download_plan.json').read_text());manifest=json.loads((ROOT/'reports/dens_download_manifest.json').read_text())
    refs={r['id']:r for r in manifest['windows']};cache=ROOT/'reports/dens_features.json'
    if cache.exists():feature=json.loads(cache.read_text())
    else:
        rows=[];excluded=[]
        for t in plan['trials']:
            try:
                assert hashlib.sha256(Path(t['file']).read_bytes()).hexdigest()==refs[t['id']]['sha256']
                row=features(t);rows.append(row);print(t['id'],len(rows),flush=True)
            except (ValueError,AssertionError) as exc:excluded.append({'id':t['id'],'reason':str(exc)})
        feature={'records':rows,'excluded':excluded,'feature_names':{'eeg':[prefix+n for prefix in ['median_','iqr_'] for n in EEG_NAMES],'cardiac':CARDIAC_NAMES},'protocol':plan['protocol']};write_json(cache,feature)
    rows=feature['records'];y=np.array([r['label']=='high' for r in rows],int)
    people=sorted({r['subject'] for r in rows});clips=sorted({r['stimulus'] for r in rows});assert len(people)>=6 and len(clips)>=6 and min(np.bincount(y))>=10
    rng=np.random.default_rng(94);rng.shuffle(people);rng.shuffle(clips)
    pg={x:i%4 for i,x in enumerate(people)};cg={x:i%4 for i,x in enumerate(clips)}
    predictions={k:np.full(len(rows),-1,int) for k in ['EEG','Cardiac','Fusion','Majority']};probs={k:np.full(len(rows),np.nan) for k in predictions};folds=[]
    x={k:np.array([r[k.lower()] for r in rows]) for k in ['EEG','Cardiac']};x['Fusion']=np.c_[x['EEG'],x['Cardiac']]
    for pgroup in range(4):
        for cgroup in range(4):
            train=np.array([i for i,r in enumerate(rows) if pg[r['subject']]!=pgroup and cg[r['stimulus']]!=cgroup]);test=np.array([i for i,r in enumerate(rows) if pg[r['subject']]==pgroup and cg[r['stimulus']]==cgroup])
            if len(test)==0:continue
            assert len(np.unique(y[train]))==2
            tp={rows[i]['subject'] for i in train};vp={rows[i]['subject'] for i in test};tc={rows[i]['stimulus'] for i in train};vc={rows[i]['stimulus'] for i in test};assert not(tp&vp) and not(tc&vc)
            folds.append({'participant_group':pgroup,'stimulus_group':cgroup,'train_indices':train.tolist(),'test_indices':test.tolist(),'train_participants':sorted(tp),'test_participants':sorted(vp),'train_stimuli':sorted(tc),'test_stimuli':sorted(vc)})
            for name in x:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=94));model.fit(x[name][train],y[train]);predictions[name][test]=model.predict(x[name][test]);probs[name][test]=model.predict_proba(x[name][test])[:,1]
            predictions['Majority'][test]=np.bincount(y[train],minlength=2).argmax();probs['Majority'][test]=y[train].mean()
    results={}
    for name,p in predictions.items():
        assert np.all(p>=0) and np.isfinite(probs[name]).all()
        results[name]={**score(y,p),'roc_auc':float(roc_auc_score(y,probs[name])),'predicted':p.tolist(),'probability_high':probs[name].tolist()}
    # Participant-cluster bootstrap of paired out-of-fold predictions. It does
    # not refit models and cannot establish significance in a small cohort.
    delta=[];rng=np.random.default_rng(94)
    for _ in range(1000):
        sample=rng.choice(people,len(people),replace=True);ix=np.concatenate([np.array([i for i,r in enumerate(rows) if r['subject']==s]) for s in sample])
        delta.append(score(y[ix],predictions['Fusion'][ix])['macro_f1']-score(y[ix],predictions['EEG'][ix])['macro_f1'])
    report={'protocol':plan['protocol'],'synthetic':False,'n':len(rows),'participants':len(people),'stimuli':len(clips),'class_counts':{'low':int((y==0).sum()),'high':int((y==1).sum())},'results':results,'truth':y.tolist(),'ids':[r['id'] for r in rows],'folds':folds,'excluded_source_recordings':plan['excluded'],'excluded_windows':feature['excluded'],'fusion_minus_eeg_macro_f1':results['Fusion']['macro_f1']-results['EEG']['macro_f1'],'paired_participant_bootstrap_95_interval':np.percentile(delta,[2.5,97.5]).tolist(),'interpretation':'Game-free real EEG/ECG. Participants and movie identities disjoint in every fold. Other artifacts, self-report noise and small-cohort uncertainty remain. No face/physiology/raga co-validation.'}
    write_json(ROOT/'reports/dens_evaluation.json',report)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));names=list(results);bars=ax.bar(names,[results[k]['macro_f1'] for k in names],color=['#829269','#b58157','#244f41','#ccc']);ax.bar_label(bars,fmt='%.3f');ax.set_ylim(0,1);ax.set_ylabel('Out-of-fold macro F1');ax.set_title(f'DENS real EEG/ECG: {len(rows)} windows / {len(people)} participants\nPeople AND movie clips held out');fig.tight_layout();fig.savefig(ROOT/'reports/dens_evaluation.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k in ['n','participants','stimuli','class_counts','fusion_minus_eeg_macro_f1','paired_participant_bootstrap_95_interval']},indent=2))
if __name__=='__main__':main()
