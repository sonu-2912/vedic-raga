import io,math
import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app
from backend.models import FusDenseNet,FDCF
from backend.signals import eeg_features,hrv_features,sample_signals

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('RAGAMOOD_DB',str(tmp_path/'test.sqlite3'))
    monkeypatch.setenv('RAGAMOOD_FDCF',str(tmp_path/'test-fdcf.pt'))
    with TestClient(app) as c: yield c

def test_catalog_counts_and_missing_metadata(client):
    health=client.get('/api/health').json();assert health['ragas']==72 and health['link_entries']==528
    rows=client.get('/api/ragas').json();assert len(rows)==72
    assert all(r['thaat'] is None for r in rows)
    assert client.get('/api/ragas/999').status_code==404

def test_feedback_personalization_and_persistence(client):
    payload={'user_id':'alice','emotion':'sad','limit':30}
    before=client.post('/api/recommendations',json=payload).json();r=before['recommendations'][0]
    bscore=r['score']
    assert client.post('/api/feedback',json={'user_id':'alice','raga_id':r['id'],'emotion':'sad','liked':1,'mood_improvement':2}).status_code==200
    after=client.post('/api/recommendations',json=payload).json()
    assert next(x['score'] for x in after['recommendations'] if x['id']==r['id'])>bscore
    other=client.post('/api/recommendations',json={**payload,'user_id':'bob'}).json()
    assert next(x['score'] for x in other['recommendations'] if x['id']==r['id'])==bscore
    assert len(client.get('/api/history/alice').json()['feedback'])==1

def test_dislike_lowers_score(client):
    p={'user_id':'alice','emotion':'neutral','limit':30}
    before=client.post('/api/recommendations',json=p).json()['recommendations'];r=before[0]
    client.post('/api/feedback',json={'user_id':'alice','raga_id':r['id'],'emotion':'neutral','liked':-1,'mood_improvement':-2})
    after=client.post('/api/recommendations',json=p).json()['recommendations']
    assert next(x['score'] for x in after if x['id']==r['id'])<r['score']

def test_listening_and_url_validation(client):
    r=client.get('/api/ragas/0').json();url=r['youtube_links']['vocal'][0]
    assert client.post('/api/listening',json={'raga_id':0,'url':url}).status_code==200
    assert len(client.get('/api/history/local-user').json()['listening'])==1
    assert client.post('/api/listening',json={'raga_id':0,'url':'https://example.com'}).status_code==422

@pytest.mark.parametrize('payload',[{'emotion':'invented'},{'limit':1000},{'prahara':9},{'user_id':"bad'user"}])
def test_invalid_recommendations(client,payload):
    assert client.post('/api/recommendations',json=payload).status_code==422

def test_invalid_feedback(client):
    assert client.post('/api/feedback',json={'raga_id':72,'emotion':'sad','liked':1}).status_code==422
    assert client.post('/api/feedback',json={'raga_id':0,'emotion':'sad','mood_improvement':3}).status_code==422

def test_signals_and_multimodal(client):
    s=client.get('/api/signals/sample').json()
    e=client.post('/api/eeg/process',json=s['eeg']);h=client.post('/api/hrv/process',json=s['hrv'])
    assert e.status_code==200 and h.status_code==200
    assert e.json()['sample_data'] and h.json()['rmssd_ms']>0
    assert e.json()['bands']['alpha']['relative'][0]>e.json()['bands']['beta']['relative'][0]
    out=client.post('/api/recommendations',json={'emotion':'neutral','eeg':s['eeg'],'hrv':s['hrv']})
    assert out.status_code==200 and out.json()['modalities']['physiology']

def test_hrv_known_values():
    rr=np.tile([800.,900.],20);r=hrv_features(rr)
    assert r['rmssd_ms']==pytest.approx(100.)
    assert r['sdnn_ms']==pytest.approx(np.std(rr,ddof=1))

def test_signal_validation(client):
    assert client.post('/api/eeg/process',json={'samples':[[0.]*1024],'sample_rate':128}).status_code==422
    assert client.post('/api/hrv/process',json={'rr_ms':[1.,2.]}).status_code==422

def test_image_validation(client):
    assert client.post('/api/emotion/predict',files={'image':('a.jpg',b'invalid','image/jpeg')}).status_code==422
    b=io.BytesIO();Image.new('RGB',(64,64),'black').save(b,format='PNG')
    assert client.post('/api/emotion/predict',files={'image':('a.png',b.getvalue(),'image/png')}).status_code==422

def test_fusdense_forward_backward():
    torch.manual_seed(42);model=FusDenseNet();x=torch.rand(2,1,48,48)
    logits,recon=model(x,return_reconstruction=True)
    assert logits.shape==(2,7) and recon.shape==x.shape
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1]))+.1*torch.nn.functional.mse_loss(recon,x)
    loss.backward()
    assert model.fusion.cross.in_proj_weight.grad is not None
    assert torch.isfinite(loss)

def test_fdcf_focus_and_gradient():
    model=FDCF(2,72);logits,d=model(torch.tensor([0,1]),torch.tensor([0,1]),torch.tensor([0,1]),return_details=True)
    assert logits.shape==(2,) and d['dynamic'].shape==(2,16)
    assert ((d['beta']>=0)&(d['beta']<=1)).all()
    logits.sum().backward();assert model.emotion.weight.grad is not None

def test_fusdense_v2_spatial_classifier_and_attention_gradients():
    model=FusDenseNet(version=2);x=torch.rand(3,1,48,48)
    logits,reconstruction=model(x,return_reconstruction=True)
    assert logits.shape==(3,7) and reconstruction.shape==x.shape
    assert model.classifier[0].in_features==84*7
    assert reconstruction.min()>=0 and reconstruction.max()<=1
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,3,6]))
    loss.backward()
    for block in [model.fusion,*model.more_fusion]:
        assert block.cross.in_proj_weight.grad is not None
        assert torch.isfinite(block.cross.in_proj_weight.grad).all()

def test_checkpoint_cache_refreshes_after_atomic_replacement(tmp_path,monkeypatch):
    from backend import emotion
    from backend.models import FER_CLASSES
    (tmp_path/'models').mkdir();path=tmp_path/'models/fusdense.pt'
    monkeypatch.setattr(emotion,'ROOT',tmp_path)
    monkeypatch.setattr(emotion,'_models',{});monkeypatch.setattr(emotion,'_infos',{});monkeypatch.setattr(emotion,'_versions',{})
    model=FusDenseNet(version=2)
    payload={'state_dict':model.state_dict(),'classes':FER_CLASSES,'architecture':{'version':2},'synthetic':False}
    torch.save(payload,path);first,_=emotion.load_model('fusdense')
    assert emotion.load_model('fusdense')[0] is first
    payload['epoch']=123;temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
    second,_=emotion.load_model('fusdense')
    assert first is not second and emotion._infos['fusdense']['epoch']==123

def test_research_evidence_refuses_mismatched_metrics(client):
    data=client.get('/api/research/results').json()
    assert data['status'] in {'untrained','evaluated','evaluation_pending'}
    if data['evaluation'] is not None:
        assert data['evaluation']['checkpoint_sha256']==data['checkpoint']['sha256']
    assert client.get('/api/research/plots/not-a-plot').status_code==404

def test_shared_inference_flip_and_temperature():
    from backend.inference import emotion_logits
    class Spatial(torch.nn.Module):
        def forward(self,x):return torch.stack([x[:,:,:,0].sum((1,2)),x[:,:,:,-1].sum((1,2))],1)
    model=Spatial();x=torch.zeros(2,1,4,4);x[:,:,:,0]=1
    assert torch.equal(emotion_logits(model,x),torch.tensor([[4.,0.],[4.,0.]]))
    out=emotion_logits(model,x,{'horizontal_flip_tta':True,'temperature':2.})
    assert torch.equal(out,torch.ones(2,2))

def test_forget_profile(client,monkeypatch):
    # No model present during tests, so erasure affects only isolated test data.
    client.post('/api/feedback',json={'raga_id':0,'emotion':'sad','liked':1})
    assert client.delete('/api/history/local-user').status_code==200
    assert client.get('/api/history/local-user').json()['feedback']==[]

def test_fer_zip_split_limit_and_images(tmp_path):
    import zipfile
    from backend.datasets import FER2013
    from backend.models import FER_CLASSES
    path=tmp_path/'fer.zip'
    with zipfile.ZipFile(path,'w') as z:
        for split,offset in [('train',0),('test',10)]:
            for label,name in enumerate(FER_CLASSES):
                for j in range(10):
                    b=io.BytesIO();Image.new('L',(48,48),label*25+j+offset).save(b,format='PNG')
                    z.writestr(f'{split}/{name}/{j}.png',b.getvalue())
    train=FER2013(path,'train',limit=14);val=FER2013(path,'val');test=FER2013(path,'test',limit=14)
    assert len(train)==14 and set(y for _,y in train.rows)==set(range(7))
    assert not set(f for f,_ in train.rows)&set(f for f,_ in val.rows)
    assert not set(train.fingerprints())&set(test.fingerprints())
    x,y=train[0];assert x.shape==(1,48,48) and 0<=x.min()<=x.max()<=1

def test_fer_csv_official_usage_split(tmp_path):
    import pandas as pd
    from backend.datasets import FER2013
    rows=[]
    for usage in ['Training','PublicTest','PrivateTest']:
        for label in range(7): rows.append({'emotion':label,'pixels':' '.join([str(label)]*2304),'Usage':usage})
    path=tmp_path/'fer.csv';pd.DataFrame(rows).to_csv(path,index=False)
    for split in ['train','val','test']:
        ds=FER2013(path,split);assert len(ds)==7
        assert [ds[i][1] for i in range(7)]==list(range(7))
