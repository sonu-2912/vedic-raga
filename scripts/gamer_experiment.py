"""Participant-disjoint real EEG/ECG feature-fusion benchmark on official GamER.

No faces/weather/raga responses exist in this experiment. Protocol is fixed in
advance; no model or hyperparameter selection uses outer-fold predictions.
"""
import argparse,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from backend.physiology_features import paired_features,EEG_NAMES,CARDIAC_NAMES

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL={'dataset':'GamER official v1; DOI 10.5281/zenodo.8191877','license':'CC-BY-4.0','seed':42,
 'primary_task':['Angry','Sad'],'secondary_task':['Angry','Sad','Bored','Stressed','Happy'],
 'intervals':'whole annotated intervals >=15 seconds; no padding or invented labels',
 'alignment':'signal time = video annotation time - VIDEO_OFFSET; offset-sign assumption disclosed; sensitivity +offset also reported',
 'quality':'both modalities finite/nonflat, ADC clipping <=2%, >=8 RR, >=90% RR in 300..2000ms; no label-based signal rejection',
 'model':'StandardScaler fitted only on training subjects + LogisticRegression C=1, balanced class weights, max_iter=2000',
 'evaluation':'leave-one-participant-out; identical accepted intervals for EEG, cardiac and concatenated fusion; no tuning',
 'features':{'eeg':EEG_NAMES,'cardiac':CARDIAC_NAMES},
 'limitations':['No disgust annotations','No face or weather data','Not a raga-response dataset','Single EEG lead and no ICA/manual artifact adjudication','Ultra-short interval HRV is noisy and not clinical HRV','Offset sign is an explicit assumption, not independently verified from omitted videos','Overlapping intervals remain in the same subject fold; effective n is participants, not intervals']}

def write(name,value):
    path=ROOT/'reports'/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value,indent=2,allow_nan=False))

def seconds(text):
    if not re.fullmatch(r'\d+:\d{2}',text.strip()):raise ValueError('Invalid annotation timestamp')
    minute,second=map(int,text.split(':'))
    if second>=60:raise ValueError('Invalid annotation seconds')
    return minute*60+second

def extract(sign=-1,window_seconds=None):
    folder=ROOT/'data/private/gamer';annotations=pd.read_csv(folder/'emotion_annotation.csv')
    offsets=pd.read_csv(folder/'video_offsets.csv',sep=';').set_index('SUB_ID').VIDEO_OFFSET.to_dict()
    rows=[];excluded=[];audit=[]
    with zipfile.ZipFile(folder/'data.zip') as archive:
        members=sorted(n for n in archive.namelist() if n.endswith('.h5') and n.startswith('data/'))
        for member in members:
            subject=int(member.split('/')[1]);selected=annotations[annotations.SUB_ID==subject]
            with h5py.File(io.BytesIO(archive.read(member)),'r') as f:
                device=f[next(iter(f))];fs=float(device.attrs['sampling rate']);raw=device['raw']
                sensors={str(v.attrs.get('sensor','')):k for k,v in raw.items()}
                if 'EEG' not in sensors or 'ECG' not in sensors:excluded.append({'subject':subject,'reason':'missing sensor'});continue
                eeg=raw[sensors['EEG']][:].flatten();ecg=raw[sensors['ECG']][:].flatten()
                audit.append({'subject':subject,'file':member,'sample_rate':fs,'duration_seconds':len(eeg)/fs,'sensors':sensors,'video_offset':offsets.get(subject)})
                for idx,row in selected.iterrows():
                    try:
                        interval=str(row.INTERVAL).split('-')
                        if len(interval)!=2:raise ValueError('Malformed annotation interval')
                        start,end=map(seconds,interval)
                        if end-start<15:raise ValueError('Annotation shorter than 15 seconds')
                        if window_seconds and end-start<window_seconds:raise ValueError('Annotation shorter than fixed window')
                        if subject not in offsets:raise ValueError('Missing video offset')
                        first,last=[int(round((t+sign*offsets[subject])*fs)) for t in [start,end]]
                        if first<0 or last>min(len(eeg),len(ecg)):raise ValueError('Annotation outside recording')
                        if window_seconds:
                            center=(first+last)//2;first=center-int(round(window_seconds*fs))//2;last=first+int(round(window_seconds*fs))
                        a,b,quality=paired_features(eeg[first:last],ecg[first:last],fs)
                        rows.append({'annotation_id':int(idx),'subject':subject,'label':row.EMOTION,'game':row.GAME_ID,'interval':row.INTERVAL,'recording':member,'first_sample':first,'last_sample':last,'sample_rate':fs,'eeg':a,'cardiac':b,'quality':quality})
                    except ValueError as error:excluded.append({'annotation_id':int(idx),'subject':subject,'reason':str(error)})
            print(json.dumps({'subject':subject,'accepted_so_far':len(rows),'offset_sign':sign}),flush=True)
    if len({row['annotation_id'] for row in rows})!=len(rows):raise ValueError('Ambiguous multiple recordings accept the same annotation; resolve recording identity before evaluation')
    result={'protocol':PROTOCOL,'offset_sign':sign,'fixed_window_seconds':window_seconds,'records':rows,'excluded':excluded,'recording_audit':audit}
    write(('gamer_fixed30_features' if window_seconds else 'gamer_features')+('_plus_offset' if sign==1 else '')+'.json',result)
    return result

def summarize(y,p,labels):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=labels,average='macro',zero_division=0)),
      'classification_report':classification_report(y,p,labels=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,p,labels=labels).tolist()}

def evaluate(features,labels):
    rows=[r for r in features['records'] if r['label'] in labels]
    y=np.array([r['label'] for r in rows]);groups=np.array([r['subject'] for r in rows])
    if len(set(groups))<5 or set(y)!=set(labels):raise ValueError('Insufficient real labelled subjects/classes')
    splits=list(LeaveOneGroupOut().split(y,y,groups));results={};predictions={};fold_manifest=[]
    matrices={'EEG':np.array([r['eeg'] for r in rows]),'Cardiac':np.array([r['cardiac'] for r in rows]),'Fusion':np.array([r['eeg']+r['cardiac'] for r in rows]),'Majority':np.zeros((len(rows),1)),'Game ID control':np.array([[str(r['game'])] for r in rows])}
    for name,x in matrices.items():
        pred=np.empty(len(y),dtype=y.dtype)
        estimator=DummyClassifier(strategy='most_frequent') if name=='Majority' else make_pipeline(OneHotEncoder(handle_unknown='ignore') if name=='Game ID control' else StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42))
        for train,test in splits:
            assert not set(groups[train])&set(groups[test])
            model=clone(estimator);model.fit(x[train],y[train]);pred[test]=model.predict(x[test])
            if name=='EEG':fold_manifest.append({'held_out_subject':int(groups[test][0]),'train_subjects':sorted(int(g) for g in set(groups[train])),'test_annotation_ids':[rows[i]['annotation_id'] for i in test]})
        results[name]=summarize(y,pred,labels);predictions[name]=pred.tolist()
    rng=np.random.default_rng(42);subjects=np.unique(groups);differences=[]
    for _ in range(1000):
        sampled=rng.choice(subjects,len(subjects),replace=True);indices=np.concatenate([np.where(groups==g)[0] for g in sampled])
        differences.append(f1_score(y[indices],np.array(predictions['Fusion'])[indices],labels=labels,average='macro',zero_division=0)-f1_score(y[indices],np.array(predictions['EEG'])[indices],labels=labels,average='macro',zero_division=0))
    return {'labels':labels,'n_intervals':len(y),'n_subjects':len(subjects),'class_counts':pd.Series(y).value_counts().to_dict(),'game_label_counts':pd.crosstab(pd.Series([r['game'] for r in rows],name='game'),pd.Series(y,name='label')).to_dict(),'ultrashort_intervals':sum(r['quality']['ultrashort_hrv'] for r in rows),'results':results,'folds':fold_manifest,'predictions':{'truth':y.tolist(),'subjects':groups.tolist(),**predictions},'fusion_minus_eeg_macro_f1':results['Fusion']['macro_f1']-results['EEG']['macro_f1'],'subject_bootstrap_95pct_delta_interval':np.percentile(differences,[2.5,97.5]).tolist(),'bootstrap_note':'Percentile bootstrap of existing outer-fold predictions by subject; approximate uncertainty, not nested refit or multiplicity correction','confounding_audit':'Game-ID-only negative control added before outer-fold scoring. Game/emotion association can make classification reflect task context or artifacts rather than emotion-specific physiology.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--features-only',action='store_true');parser.add_argument('--sensitivity',action='store_true');a=parser.parse_args()
    protocol_path=ROOT/'reports/gamer_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=PROTOCOL:raise ValueError('Protocol changed: use a separately named experiment')
    write('gamer_protocol.json',PROTOCOL)
    suffix='_plus_offset' if a.sensitivity else '';cache=ROOT/f'reports/gamer_features{suffix}.json'
    features=json.loads(cache.read_text()) if cache.exists() else extract(1 if a.sensitivity else -1)
    if a.features_only:return
    tasks={'anger_vs_sadness':evaluate(features,PROTOCOL['primary_task']),'five_class':evaluate(features,PROTOCOL['secondary_task'])}
    report={'protocol':PROTOCOL,'synthetic':False,'offset_sign':features['offset_sign'],'accepted_intervals':len(features['records']),'exclusion_count':len(features['excluded']),'tasks':tasks}
    write(f'gamer_evaluation{suffix}.json',report)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(task,item) in zip(axes,tasks.items()):
        names=list(item['results']);values=[item['results'][n]['macro_f1'] for n in names]
        bars=ax.bar(names,values,color=['#89956c','#b57b49','#24483b','#cccccc','#b7a4bb']);ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_ylim(0,1);ax.set_ylabel('Macro F1');ax.tick_params(axis='x',labelrotation=20);ax.set_title(f'{task.replace("_"," ")}\n{item["n_subjects"]} held-out participants, {item["n_intervals"]} intervals')
    fig.suptitle('Actual GamER recordings · leave-one-participant-out · no facial or raga data');fig.tight_layout();fig.savefig(ROOT/f'reports/gamer_fusion{suffix}.png',dpi=160);plt.close(fig)
    print(json.dumps({k:{'n':v['n_intervals'],'subjects':v['n_subjects'],'metrics':{m:{'accuracy':s['accuracy'],'macro_f1':s['macro_f1']} for m,s in v['results'].items()},'delta_ci':v['subject_bootstrap_95pct_delta_interval']} for k,v in tasks.items()},indent=2))
if __name__=='__main__':main()
