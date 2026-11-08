"""Actual engineering ablations with simulated physiology; no efficacy metrics."""
import os,json,tempfile
from fastapi.testclient import TestClient
from backend.main import app
from backend.catalog import ROOT

def main():
    previous=os.environ.get('RAGAMOOD_DB')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ['RAGAMOOD_DB']=tmp+'/multimodal.sqlite3'
            with TestClient(app) as c:
                sample=c.get('/api/signals/sample?profile=activated').json()
                base={'user_id':'synthetic-context-test','emotion':'fear','limit':6}
                body={'eeg':sample['eeg'],'hrv':sample['hrv'],'baseline':sample['baseline'],'use_sample_context':True}
                scenarios={'confirmed_mood_only':base,'weather_only':{**base,'weather':{'condition':'rain','source':'manual'}},'body_without_reference':{**base,'eeg':sample['eeg'],'hrv':sample['hrv']},'body_without_hrv':{**base,**body,'hrv':None},'body_with_reference':{**base,**body},'all_contexts':{**base,**body,'weather':{'condition':'rain','source':'manual'}}}
                results={}
                for name,payload in scenarios.items():
                    r=c.post('/api/recommendations',json=payload);r.raise_for_status();d=r.json()
                    results[name]={'effective_goal':d['context']['effective_goal'],'target_bhava':d['target_bhava'],'ragas':[{'name':r['raga_name'],'score':r['score'],'weather_adjustment':r['components']['weather_adjustment'],'physiology_adjustment':r['components']['physiology_adjustment']} for r in d['recommendations']],'context':d['context']}
                result={'kind':'Actual software execution with manual weather and SYNTHETIC EEG/HRV/reference; no participant or sensor evidence','scenarios':results,'recognition_accuracy':None,'recommendation_quality':None,'mood_improvement':None,'interpretation':'Ablations verify that optional inputs reach the scorer and change goals/scores where specified. They do not establish better recommendations or emotion recognition.'}
                (ROOT/'reports/multimodal_extension_evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{'goal':v['effective_goal'],'top_raga':v['ragas'][0]['name']} for k,v in results.items()},indent=2))
    finally:
        if previous is None:os.environ.pop('RAGAMOOD_DB',None)
        else:os.environ['RAGAMOOD_DB']=previous

if __name__=='__main__':main()
