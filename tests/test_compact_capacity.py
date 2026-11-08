import copy
import torch
import pytest
from backend.models import FusDenseNet,FER_CLASSES
from scripts.compact_fusdense import CompactFusDenseNet,coarse_spatial_features,model_from,capacity_loss

def test_coarse_pool_preserves_fused_features_and_rejects_wrong_shape():
 x=torch.randn(3,3276,requires_grad=True);y=coarse_spatial_features(x)
 assert y.shape==(3,1260) and torch.equal(x[:,:588],y[:,:588]);y.square().mean().backward();assert torch.isfinite(x.grad).all()
 with pytest.raises(ValueError):coarse_spatial_features(torch.zeros(2,1260))

def test_compact_model_matches_cached_native_features_and_updates_attention():
 torch.set_num_threads(2);torch.manual_seed(82);base=FusDenseNet(version=4,head_width=128);compact=CompactFusDenseNet(head_width=128);state=compact.state_dict();state.update({k:v for k,v in base.state_dict().items() if not k.startswith('classifier.')});compact.load_state_dict(state);base.eval();compact.eval();capture={};handle=base.classifier.register_forward_pre_hook(lambda module,args:capture.update(features=args[0]));x=torch.rand(2,1,48,48)
 base(x)
 # Calling the classifier directly must supply pre-pooling features, as its native hook pools them.
 expected=compact.classifier(capture['features']);actual=compact(x);assert torch.allclose(actual,expected,atol=1e-5);actual.square().mean().backward();assert compact.fusion.cross.in_proj_weight.grad.abs().sum()>0
 assert not hasattr(compact,'transfer_backbone') and compact.classifier[0].in_features==1260
 assert sum(p.numel() for p in compact.parameters())==309792;handle.remove()

def test_compact_checkpoint_roundtrip_uses_one_native_network(tmp_path):
 torch.manual_seed(7);m=CompactFusDenseNet(head_width=128).eval();p=tmp_path/'model.pt';torch.save({'state_dict':m.state_dict(),'architecture':{'version':9,'head_width':128},'classes':FER_CLASSES,'synthetic':False},p);loaded=model_from(torch.load(p,map_location='cpu',weights_only=True),torch.device('cpu')).eval();x=torch.rand(2,1,48,48)
 with torch.no_grad():assert torch.equal(m(x),loaded(x))

def test_capacity_loss_smooths_true_training_labels_without_reference_gradients():
 out=torch.randn(4,7,requires_grad=True);det=torch.randn(4,7,requires_grad=True);reference=torch.randn(4,7,requires_grad=True);labels=torch.tensor([1,3,5,6]);loss,ce,ret=capacity_loss(out,det,labels,reference,torch.ones(7),.25,.025);loss.backward();assert torch.isfinite(loss) and out.grad is not None and reference.grad is None
