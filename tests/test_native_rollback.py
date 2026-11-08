import torch
import pytest
from scripts.native_parameter_rollback import restore_native_state

def test_native_rollback_preserves_learned_residuals_and_tracks_classifier_source():
 old={'stem.weight':torch.ones(2,2),'bn.running_mean':torch.zeros(2),'classifier.0.weight':torch.ones(3,2)};trained={k:v+2 for k,v in old.items()};trained['fusion.refine_a.reduce.weight']=torch.full((2,2),5.)
 features,a=restore_native_state(trained,old,False);all_old,b=restore_native_state(trained,old,True)
 assert torch.equal(features['stem.weight'],old['stem.weight']) and torch.equal(features['classifier.0.weight'],trained['classifier.0.weight'])
 assert torch.equal(all_old['classifier.0.weight'],old['classifier.0.weight']) and torch.equal(all_old['fusion.refine_a.reduce.weight'],trained['fusion.refine_a.reduce.weight'])
 assert torch.equal(trained['stem.weight'],torch.full((2,2),3.)) and a['inference_networks']==1 and not a['external_encoder']
 with pytest.raises(ValueError):restore_native_state(trained,{'stem.weight':torch.ones(3,3)})
