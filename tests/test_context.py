import json
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.context import body_context,weather_context
from backend.face_policy import expression_suggestion

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('RAGAMOOD_DB',str(tmp_path/'context.sqlite3'));monkeypatch.setenv('RAGAMOOD_FDCF',str(tmp_path/'absent.pt'))
    with TestClient(app) as c:yield c

def test_weather_contribution_and_invalid_input(client):
    p={'emotion':'neutral','limit':30}
    base=client.post('/api/recommendations',json=p).json()
    rainy=client.post('/api/recommendations',json={**p,'weather':{'condition':'rain','source':'manual'}}).json()
    assert rainy['modalities']['weather'] and rainy['context']['weather']['season_tags']==['varsha']
    assert any(r['components']['weather_adjustment']>0 for r in rainy['recommendations'])
    assert all(r['components']['weather_adjustment']==0 for r in base['recommendations'])
    assert client.post('/api/recommendations',json={**p,'weather':{'condition':'rain','temperature_c':200}}).status_code==422

def test_body_demonstration_requires_explicit_opt_in(client):
    s=client.get('/api/signals/sample?profile=activated').json();p={'emotion':'fear','eeg':s['eeg'],'hrv':s['hrv'],'baseline':s['baseline']}
    no=client.post('/api/recommendations',json=p).json()
    yes=client.post('/api/recommendations',json={**p,'use_sample_context':True}).json()
    assert no['context']['effective_goal']=='balance'
    assert yes['context']['effective_goal']=='calm' and yes['target_bhava']=='shama'
    assert yes['context']['physiology']['sample_data']
    assert any(r['components']['physiology_adjustment']>0 for r in yes['recommendations'])
    assert [r['id'] for r in no['recommendations']]!=[r['id'] for r in yes['recommendations']]

def test_missing_reference_retains_features_without_inference(client):
    s=client.get('/api/signals/sample').json()
    result=client.post('/api/recommendations',json={'emotion':'sad','hrv':s['hrv']}).json()
    assert result['context']['physiology']['features']['hrv']['rmssd_ms']>0
    assert not result['context']['physiology']['calming_context']
    assert result['context']['physiology']['arousal_hint'] is None

def test_unsynchronized_and_mixed_signals_are_not_joint_evidence():
    # Numerical fixtures test the guard, not participant/sensor data.
    eeg={'sample_data':False,'session_id':'a','duration_seconds':16,'bands':{'alpha':{'relative':[.1]},'beta':{'relative':[.8]}}}
    hrv={'sample_data':False,'session_id':'b','duration_seconds':90,'rmssd_ms':10}
    baseline={'alpha_beta_ratio':3,'rmssd_ms':30}
    assert not body_context(eeg,hrv,baseline)['calming_context']
    hrv['session_id']='a';assert body_context(eeg,hrv,baseline)['calming_context']
    hrv['sample_data']=True;assert not body_context(eeg,hrv,baseline,True)['calming_context']

def test_uncertain_neutral_is_not_called_at_ease(tmp_path,monkeypatch):
    import backend.face_policy as module
    monkeypatch.setattr(module,'ROOT',tmp_path);(tmp_path/'reports').mkdir()
    (tmp_path/'reports/expression_policy.json').write_text(json.dumps({'checkpoint_sha256':'fixture','eligible_classes':['happy'],'minimum_confidence':.7,'minimum_margin':.2}))
    p={'neutral':.99,'disgust':.01}
    assert expression_suggestion(p,'fixture')['suggested_emotion'] is None
    assert expression_suggestion({'happy':.95,'neutral':.05},'fixture')['suggested_emotion']=='happy'
    assert expression_suggestion({'happy':.95,'neutral':.05},'different-checkpoint')['abstained']

def test_city_lookup_parses_provider_without_network(monkeypatch):
    import backend.weather as module
    module._cache.clear()
    def response(base,params):
        if 'geocoding' in base:return {'results':[{'name':'Fixture city','latitude':18,'longitude':73,'country':'Fixture'}]}
        return {'current':{'weather_code':61,'temperature_2m':24,'relative_humidity_2m':80,'precipitation':1.2,'time':'2026-11-03T12:00'},'timezone':'Asia/Kolkata'}
    monkeypatch.setattr(module,'_json',response)
    weather=module.city_weather('Fixture city');assert weather['condition']=='rain' and weather['source']=='open_meteo'

def test_explicit_goal_overrides_automatic_body_goal(client):
    s=client.get('/api/signals/sample?profile=activated').json()
    r=client.post('/api/recommendations',json={'emotion':'fear','listening_goal':'energize','eeg':s['eeg'],'hrv':s['hrv'],'baseline':s['baseline'],'use_sample_context':True}).json()
    assert r['context']['effective_goal']=='energize' and r['target_bhava']=='utsaha'
