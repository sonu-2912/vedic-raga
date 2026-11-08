import pytest
import torch
from scripts.train_orientation_head_repair import paired_training_features

def caches():
 labels=torch.tensor([0,1,2]);a={'source_sha256':'same-pure','manifest_sha256':'same-train','features':torch.arange(12).reshape(3,4).float(),'labels':labels};b={**a,'features':a['features']+100};return a,b,labels

def test_paired_training_views_preserve_exact_class_label_alignment():
 a,b,y=caches();x,target=paired_training_features(a,b,y)
 assert torch.equal(x[:3],a['features']) and torch.equal(x[3:],b['features'])
 assert torch.equal(target,torch.cat([y,y]))

@pytest.mark.parametrize('field',['source_sha256','manifest_sha256'])
def test_paired_training_views_reject_incompatible_provenance(field):
 a,b,y=caches();b[field]='incompatible'
 with pytest.raises(ValueError,match='provenance'):paired_training_features(a,b,y)

def test_paired_training_views_reject_shuffled_or_wrong_split_labels():
 a,b,y=caches();b['labels']=y.roll(1)
 with pytest.raises(ValueError,match='label order'):paired_training_features(a,b,y)
