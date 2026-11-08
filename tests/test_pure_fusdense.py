import copy
import pytest
import torch
from torch import nn
from scripts.train_pure_fusdense import PairedClassBatches,class_balanced_supcon,symmetric_kl,assert_pure
from scripts.colab_common import set_training_mode
from backend.models import FusDenseNet

def test_paired_batches_contain_all_classes_and_no_duplicate_faces():
 torch.manual_seed(1108);labels=torch.repeat_interleave(torch.arange(7),torch.tensor([200,30,80,300,150,90,120]));sampler=PairedClassBatches(labels,28,2,5)
 assert len(sampler)==5
 for batch in sampler:
  assert len(batch)==len(set(batch))==28
  assert (torch.bincount(labels[batch],minlength=7)>=2).all()

def test_supcon_favors_class_clusters_and_has_finite_gradient():
 z=torch.tensor([[1.,0,0],[1,.1,0],[0,1.,0],[.1,1,0],[0,0,1.],[0,.1,1.]],requires_grad=True);labels=torch.tensor([0,0,1,1,2,2]);good=class_balanced_supcon(z,labels);bad=class_balanced_supcon(z,torch.tensor([0,1,2,0,1,2]));assert good<bad
 good.backward();assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
 single=class_balanced_supcon(z,torch.arange(6));assert single==0

def test_rdrop_is_symmetric_and_zero_for_identical_predictions():
 a=torch.tensor([[1.,2.,3.],[2.,-1.,0.]],requires_grad=True);b=torch.tensor([[2.,1.,0.],[1.,0.,-1.]],requires_grad=True)
 assert torch.allclose(symmetric_kl(a,b),symmetric_kl(b,a));assert symmetric_kl(a,a).abs()<1e-7
 symmetric_kl(a,b).backward();assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0

@pytest.mark.parametrize("version",[4,8])
def test_shared_backbone_rdrop_matches_two_forward_gradients(version):
 torch.set_num_threads(2);torch.manual_seed(1108);one=FusDenseNet(version=version,head_width=32);two=copy.deepcopy(one);set_training_mode(one,True);set_training_mode(two,True);x=torch.rand(2,1,48,48);y=torch.tensor([0,1]);captured={}
 h=one.classifier.register_forward_pre_hook(lambda module,args:captured.update(features=args[0]));rng=torch.get_rng_state();a=one(x);b=one.classifier(captured['features']);loss=.5*(nn.functional.cross_entropy(a,y)+nn.functional.cross_entropy(b,y))+.5*symmetric_kl(a,b);loss.backward();h.remove()
 torch.set_rng_state(rng);aa=two(x);bb=two(x);other=.5*(nn.functional.cross_entropy(aa,y)+nn.functional.cross_entropy(bb,y))+.5*symmetric_kl(aa,bb);other.backward()
 assert torch.allclose(a,aa,atol=1e-6) and torch.allclose(b,bb,atol=1e-6)
 for (name,p),(other_name,q) in zip(one.named_parameters(),two.named_parameters()):
  assert name==other_name
  if p.grad is not None:assert torch.allclose(p.grad,q.grad,atol=1e-5,rtol=1e-4),name

def test_contrastive_projection_reaches_attention_but_not_classification_head():
 torch.set_num_threads(2);m=FusDenseNet(version=4,head_width=32);m.eval();captured={};h=m.classifier.register_forward_pre_hook(lambda module,args:captured.update(features=args[0]));m(torch.rand(4,1,48,48));projection=nn.Linear(588,128);loss=class_balanced_supcon(projection(captured['features'][:,:588]),torch.tensor([1,1,0,0]));loss.backward();h.remove()
 g=m.more_fusion[1].cross.in_proj_weight.grad;assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0;assert m.classifier[0].weight.grad is None

def test_pure_guard_rejects_hybrid_or_external_feature_weights():
 assert_pure({'architecture':{'version':4},'state_dict':{'stem.0.weight':torch.zeros(1)}})
 for ck in [{'architecture':{'version':7},'state_dict':{}},{'architecture':{'version':4},'state_dict':{'transfer_backbone.weight':torch.zeros(1)}}]:
  with pytest.raises(ValueError):assert_pure(ck)


def test_single_view_protocol_cannot_change_after_test(tmp_path,monkeypatch):
 import json,sys
 from scripts.colab_evaluate import main
 (tmp_path/'bundle_manifest.json').write_text(json.dumps({'strict_all_metrics':False,'single_view_only':False}),encoding='utf-8')
 monkeypatch.setattr(sys,'argv',['evaluate','--run-dir',str(tmp_path),'--data','missing-data-must-not-be-read.zip','--device','cpu','--single-view'])
 with pytest.raises(ValueError,match='Cannot change single-view'):main()


def test_pure_deployment_requires_actual_disgust_recall_gain():
 from scripts.colab_evaluate import integration_gate
 c={k:{'difference':.01} for k in ['accuracy','macro_f1','disgust_recall','disgust_f1','macro_precision','macro_recall','disgust_precision']}
 c['disgust_recall']['difference']=0.
 assert integration_gate(c)
 assert not integration_gate(c,False,True)
 c['disgust_recall']['difference']=.01;assert integration_gate(c,False,True)


def test_native_residual_variant_preserves_initial_logits_and_has_no_external_encoder(monkeypatch):
 import torchvision.models
 def forbid(*args,**kwargs):raise AssertionError('Pure FusDenseNet cannot construct an external encoder')
 monkeypatch.setattr(torchvision.models,'densenet121',forbid)
 torch.set_num_threads(2);torch.manual_seed(1108);old=FusDenseNet(version=4,head_width=32);new=FusDenseNet(version=8,head_width=32);state=new.state_dict();state.update(old.state_dict());new.load_state_dict(state);old.eval();new.eval();x=torch.rand(3,1,48,48)
 with torch.no_grad():assert torch.equal(old(x),new(x))
 assert not hasattr(new,'transfer_backbone')
 assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())==117360
 assert_pure({'architecture':{'version':8},'state_dict':new.state_dict()})

def test_native_refinement_learns_and_keeps_feature_width():
 from backend.models import ResidualDenseRefinement
 torch.manual_seed(1108);refiner=ResidualDenseRefinement(60);x=torch.rand(3,60,8,8);target=torch.rand_like(x);opt=torch.optim.SGD(refiner.parameters(),lr=.05)
 out=refiner(x);assert out.shape==x.shape and torch.equal(out,x)
 nn.functional.mse_loss(out,target).backward();assert refiner.reduce.weight.grad.abs().sum()>0;opt.step();opt.zero_grad();nn.functional.mse_loss(refiner(x),target).backward()
 assert refiner.layers[0][-1].weight.grad.abs().sum()>0
 assert not torch.equal(refiner(x),x)
