"""Within-game, participant-held-out EEG/cardiac fusion; no pooled game labels.

Primary windows >=30s; >=15s sensitivity. Source interval labels are retained.
These recordings still do not validate clinical HRV or internal mood inference.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL={'dataset':'GamER official v1','tasks':{'QWOP':['Angry','Happy'],'CBAAD':['Bored','Stressed']},'primary_minimum_seconds':30,'sensitivity_minimum_seconds':15,'evaluation':'Separate leave-one-participant-out classifiers within each constant game; fixed balanced LogisticRegression C=1; fold-fitted StandardScaler','controls':['Majority','Duration only','Game only (constant)'],'no_game_identity_feature':True,'minimum_participants_per_class':3,'seed':91,'selection':'No hyperparameter selection on these held-out scores','limitations':['Small cohorts and source psychologist labels','Short-window RR statistics are exploratory; conventional-duration HRV may not be estimable','EEG/ECG artifact and time-within-game confounding remain possible','Offset sign is not independently confirmed; both signs reported','No FER faces, disgust or raga response labels']}

def write(name,value):(ROOT/'reports'/name).write_text(json.dumps(value,indent=2,allow_nan=False))
def score(truth,pred,labels):
    r=classification_report(truth,pred,labels=labels,output_dict=True,zero_division=0)
    return {'accuracy':float(np.mean(np.asarray(truth)==np.asarray(pred))),'macro_f1':r['macro avg']['f1-score'],'classification_report':r,'confusion_matrix':confusion_matrix(truth,pred,labels=labels).tolist()}

def task(rows,game,labels,minimum):
    rows=[r for r in rows if r['game']==game and r['label'] in labels and r['quality']['duration_seconds']>=minimum]
    counts={label:len({r['subject'] for r in rows if r['label']==label}) for label in labels}
    if min(counts.values())<3:return {'status':'insufficient','n':len(rows),'class_participants':counts,'reason':'Fewer than three participants in a class; no valid claim from this cohort'}
    assert {r['game'] for r in rows}=={game}
    y=np.array([r['label'] for r in rows]);groups=np.array([r['subject'] for r in rows]);splits=list(LeaveOneGroupOut().split(y,y,groups))
    matrices={'EEG':np.array([r['eeg'] for r in rows]),'Cardiac':np.array([r['cardiac'] for r in rows]),'Fusion':np.array([r['eeg']+r['cardiac'] for r in rows]),'Majority':np.zeros((len(rows),1)),'Duration only':np.array([[r['quality']['duration_seconds']] for r in rows]),'Game only (constant)':np.zeros((len(rows),1))}
    results={};predictions={};folds=[]
    for name,x in matrices.items():
        pred=np.empty(len(y),dtype=y.dtype)
        estimator=DummyClassifier(strategy='most_frequent') if name in ['Majority','Game only (constant)'] else make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=91))
        for train,test in splits:
            assert not set(groups[train])&set(groups[test]);assert len(set(y[train]))==2
            model=clone(estimator);model.fit(x[train],y[train]);pred[test]=model.predict(x[test])
            if name=='EEG':folds.append({'subject':int(groups[test][0]),'training_subjects':sorted(map(int,set(groups[train]))),'annotation_ids':[rows[i]['annotation_id'] for i in test]})
        results[name]=score(y,pred,labels);predictions[name]=pred.tolist()
    unique=np.unique(groups);rng=np.random.default_rng(91);deltas=[]
    for _ in range(1000):
        idx=np.concatenate([np.flatnonzero(groups==g) for g in rng.choice(unique,len(unique),replace=True)])
        deltas.append(score(y[idx],np.array(predictions['Fusion'])[idx],labels)['macro_f1']-score(y[idx],np.array(predictions['EEG'])[idx],labels)['macro_f1'])
    return {'status':'evaluated','game':game,'labels':labels,'minimum_seconds':minimum,'n':len(y),'participants':len(unique),'class_participants':counts,'class_counts':{label:int(np.sum(y==label)) for label in labels},'participants_with_both_labels':sum(len(set(y[groups==g]))==2 for g in unique),'results':results,'fusion_minus_eeg_macro_f1':results['Fusion']['macro_f1']-results['EEG']['macro_f1'],'participant_bootstrap_95pct_interval':np.percentile(deltas,[2.5,97.5]).tolist(),'predictions':{'truth':y.tolist(),'subjects':groups.tolist(),**predictions},'folds':folds}

def main():
    path=ROOT/'reports/gamer_controlled_protocol.json'
    if path.exists() and json.loads(path.read_text())!=PROTOCOL:raise ValueError('Do not alter a scored protocol silently')
    write('gamer_controlled_protocol.json',PROTOCOL)
    reports={}
    for suffix in ['', '_plus_offset']:
        data=json.loads((ROOT/f'reports/gamer_features{suffix}.json').read_text());rows=data['records']
        assert len(rows)==len({r['annotation_id'] for r in rows})
        for minimum in [30,15,60]:
            for game,labels in PROTOCOL['tasks'].items():
                key=f'{game}_{minimum}s'+suffix;reports[key]=task(rows,game,labels,minimum);print(key,json.dumps({k:v for k,v in reports[key].items() if k in ['status','n','participants','class_counts','reason','fusion_minus_eeg_macro_f1']}),flush=True)
    write('gamer_controlled_evaluation.json',{'protocol':PROTOCOL,'synthetic':False,'tasks':reports})
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,game in zip(axes,PROTOCOL['tasks']):
        result=reports[f'{game}_30s']
        if result['status']!='evaluated':ax.text(.1,.5,result['reason']);continue
        names=['EEG','Cardiac','Fusion','Majority','Duration only'];bars=ax.bar(names,[result['results'][n]['macro_f1'] for n in names],color=['#819265','#b58157','#244f41','#ccc','#938a9f']);ax.bar_label(bars,fmt='%.3f');ax.set_ylim(0,1);ax.tick_params(axis='x',labelrotation=20);ax.set_ylabel('Macro F1');ax.set_title(f'{game} constant · {result["n"]} intervals / {result["participants"]} people')
    fig.suptitle('Real paired physiology · within-game LOSO · ≥30 seconds · no game identity feature');fig.tight_layout();fig.savefig(ROOT/'reports/gamer_controlled.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
