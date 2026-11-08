import torch
from backend.models import FER_CLASSES
from scripts.compact_fusdense import CompactFusDenseNet
from scripts.compact_residual_fusdense import initialize_from_compact,model_from

def source():
 m=CompactFusDenseNet(head_width=256);return m,{'state_dict':m.state_dict(),'architecture':{'version':9,'head_width':256},'classes':FER_CLASSES,'synthetic':False}

def test_zero_residual_upgrade_preserves_predictions_without_an_external_encoder():
 torch.set_num_threads(2);torch.manual_seed(91);old,ck=source();new,arch=initialize_from_compact(ck);old.eval();new.eval();x=torch.rand(3,1,48,48)
 with torch.no_grad():assert torch.allclose(old(x),new(x),atol=1e-6)
 assert sum(p.numel() for p in new.parameters())==589456 and not hasattr(new,'transfer_backbone')
 assert all(torch.equal(new.state_dict()[k],v) for k,v in ck['state_dict'].items())

def test_new_residual_features_receive_gradients_after_zero_projection_opens():
 torch.manual_seed(2);_,ck=source();new,_=initialize_from_compact(ck);new.eval();x=torch.rand(3,1,48,48);optimizer=torch.optim.SGD(new.parameters(),lr=.01);new(x).square().sum().backward()
 assert new.fusion.refine_a.reduce.weight.grad.abs().sum()>0;optimizer.step();optimizer.zero_grad();new(x).square().sum().backward();assert new.fusion.refine_a.layers[0][2].weight.grad.abs().sum()>0

def test_residual_compact_checkpoint_is_one_portable_native_model(tmp_path):
 _,ck=source();new,arch=initialize_from_compact(ck);p=tmp_path/'model.pt';torch.save({'state_dict':new.state_dict(),'architecture':arch,'classes':FER_CLASSES,'synthetic':False},p);loaded=model_from(torch.load(p,map_location='cpu',weights_only=True),torch.device('cpu'));new.eval();loaded.eval();x=torch.rand(2,1,48,48)
 with torch.no_grad():assert torch.equal(new(x),loaded(x))
