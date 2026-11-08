"""Feature validity and evaluation isolation, not physiological efficacy claims."""
import numpy as np
import pytest
from backend.physiology_features import paired_features
from scripts.gamer_experiment import seconds,PROTOCOL

def test_recordings_need_duration_and_nonflat_data():
    with pytest.raises(ValueError,match='15 seconds'):
        paired_features(np.ones(100),np.ones(100),400)
    with pytest.raises(ValueError,match='Flat'):
        paired_features(np.ones(8000),np.ones(8000),400)
    a=np.ones(8000);a[100]=np.nan
    with pytest.raises(ValueError,match='Non-finite'):
        paired_features(a,np.ones(8000),400)

def test_annotations_do_not_invent_seconds_or_labels():
    assert seconds('10:20')==620
    for malformed in ['10:75','3.5','unknown','-1:05']:
        with pytest.raises(ValueError):seconds(malformed)
    assert PROTOCOL['primary_task']==['Angry','Sad']
    assert 'Disgust' not in PROTOCOL['secondary_task']
