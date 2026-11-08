"""An actual API pipeline on one unlabeled public-domain demo portrait.

This is an integration test, not facial-classification validation. Uses a temp DB.
"""
import os,json,tempfile,time
from fastapi.testclient import TestClient
from backend.main import app
from backend.catalog import ROOT

def main():
    source=ROOT/'data/demo_face.png'
    if not source.exists():raise SystemExit('Missing optional NASA/scikit-image portrait. See DATASETS.md.')
    old=os.environ.get('RAGAMOOD_DB')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ['RAGAMOOD_DB']=tmp+'/e2e.sqlite3'
            with TestClient(app) as c:
                start=time.perf_counter()
                prediction=c.post('/api/emotion/predict',files={'image':('demo_face.png',source.read_bytes(),'image/png')})
                prediction.raise_for_status();p=prediction.json()
                before=c.post('/api/recommendations',json={'user_id':'integration-demo','emotion':p['emotion']}).json()
                first=before['recommendations'][0]
                feedback=c.post('/api/feedback',json={'user_id':'integration-demo','emotion':p['emotion'],'raga_id':first['id'],'liked':1,'mood_improvement':1});feedback.raise_for_status()
                after=c.post('/api/recommendations',json={'user_id':'integration-demo','emotion':p['emotion']}).json()
                match=next(r for r in after['recommendations'] if r['id']==first['id'])
                assert match['score']>first['score']
                url=next(u for group in first['youtube_links'].values() for u in group)
                listened=c.post('/api/listening',json={'user_id':'integration-demo','raga_id':first['id'],'url':url});listened.raise_for_status()
                history=c.get('/api/history/integration-demo').json()
                report={'kind':'Actual integration test on one unlabeled public-domain portrait, with SYNTHETIC feedback','image_credit':'NASA Eileen Collins portrait via scikit-image. Public-domain status documented at https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut','prediction':p,'first_raga':first['raga_name'],'score_before':first['score'],'score_after':match['score'],'feedback_records':len(history['feedback']),'listening_records':len(history['listening']),'elapsed_seconds':time.perf_counter()-start,'classification_accuracy':None,'reason':'No ground-truth emotion label; cannot estimate accuracy from this demonstration.'}
                (ROOT/'reports/end_to_end.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    finally:
        if old is None:os.environ.pop('RAGAMOOD_DB',None)
        else:os.environ['RAGAMOOD_DB']=old
if __name__=='__main__':main()
