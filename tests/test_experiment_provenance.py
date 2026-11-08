"""Provenance fixtures; these scores are not model-performance evidence."""
import hashlib
import json
import pytest
import torch
from fastapi.testclient import TestClient
from backend.models import FusDenseNet

@pytest.mark.parametrize('matches',[False,True])
def test_experiment_cannot_claim_active_without_matching_weights(tmp_path,monkeypatch,matches):
    import backend.main as main
    (tmp_path/'models').mkdir();(tmp_path/'reports').mkdir()
    torch.set_num_threads(2)
    model=FusDenseNet()
    checkpoint=tmp_path/'models/fusdense.pt'
    torch.save({'state_dict':model.state_dict(),'architecture':{},'epoch':0},checkpoint)
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (tmp_path/'reports/latest_emotion_experiment.json').write_text(json.dumps({
        'candidate_checkpoint_sha256':sha if matches else '0'*64,
        'candidate_is_active':not matches, 'synthetic':True}))
    monkeypatch.setenv('RAGAMOOD_DB',str(tmp_path/'test.sqlite3'))
    monkeypatch.setenv('RAGAMOOD_FDCF',str(tmp_path/'absent.pt'))
    monkeypatch.setattr(main,'ROOT',tmp_path)
    with TestClient(main.app) as client:
        result=client.get('/api/research/results').json()
    assert result['latest_experiment']['candidate_is_active'] is matches
    assert result['evaluation'] is None
    assert result['status']=='evaluation_pending'
