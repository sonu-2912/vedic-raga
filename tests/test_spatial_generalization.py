import torch
from backend.models import FusDenseNet,FER_CLASSES
from scripts.train_spatial_generalization import initialize_v5,sampling_priors,natural_checkpoint_state,selected

def test_v5_preserves_v4_logits_and_receives_gradient():
 torch.set_num_threads(2);torch.manual_seed(808)
 old=FusDenseNet(version=4,head_width=512);old.eval()
 ck={'architecture':{'version':4,'head_width':512},'state_dict':old.state_dict(),'classes':FER_CLASSES}
 new=initialize_v5(ck);new.eval();x=torch.rand(3,1,48,48)
 with torch.no_grad():assert torch.allclose(old(x),new(x),atol=1e-6,rtol=1e-6)
 torch.nn.functional.cross_entropy(new(x),torch.tensor([1,0,3])).backward()
 assert new.spatial_refine[-1].weight.grad.abs().sum()>0
 assert new.fusion.cross.in_proj_weight.grad.abs().sum()>0

def test_train_prior_sampling_and_export_are_exact_inverse():
 p,q=sampling_priors(torch.tensor([10.,1.,20.,30.,15.,9.,15.]),.3)
 assert q[1]>p[1] and torch.allclose(q.sum(),torch.tensor(1.))
 model=FusDenseNet(version=4);bias=model.classifier[3].bias.detach().clone()
 with torch.no_grad():model.classifier[3].bias.add_(torch.log(q/p))
 state=natural_checkpoint_state(model,p,q)
 assert torch.allclose(state['classifier.3.bias'],bias,atol=1e-6)
 assert not torch.equal(model.classifier[3].bias.detach(),state['classifier.3.bias'])

def test_validation_selection_requires_both_accuracy_and_disgust_recall():
 def rep(a,d,f=.55):return {'accuracy':a,'classification_report':{'macro avg':{'f1-score':f},'disgust':{'recall':d,'f1-score':.4}}}
 base=rep(.58,.28)
 assert selected(rep(.59,.3),base)
 assert not selected(rep(.59,.28),base)
 assert not selected(rep(.57,.5),base)
 assert not selected(rep(.59,.3,.54),base)


def test_flip_averaging_cannot_erase_selected_disgust_recall_gain():
 from scripts.colab_evaluate import choose_flip
 def rep(a,f,d):return {'accuracy':a,'classification_report':{'macro avg':{'f1-score':f,'precision':.5,'recall':.5},'disgust':{'recall':d,'f1-score':.4,'precision':.5}}}
 raw=rep(.58,.53,.34)
 assert not choose_flip(raw,rep(.59,.54,.32))
 assert choose_flip(raw,rep(.59,.54,.36))
 assert choose_flip(raw,rep(.59,.54,.36),True)


def test_translation_recipe_matches_declared_five_views_and_keeps_all_classes():
 from backend.inference import emotion_logits
 from torch.nn import functional as F
 class PositionModel(torch.nn.Module):
  def forward(self,x):return x[:,:,1:3,1:3].flatten(1).sum(1,keepdim=True)*torch.arange(1,8,dtype=x.dtype)
 m=PositionModel();x=torch.arange(72,dtype=torch.float32).reshape(2,1,6,6);p=F.pad(x,(2,2,2,2),mode='replicate')
 expected=torch.stack([m(p[:,:,2+dy:8+dy,2+dx:8+dx]) for dy,dx in [(0,0),(0,-2),(0,2),(-2,0),(2,0)]]).mean(0)
 assert torch.equal(emotion_logits(m,x,{'translation_tta_pixels':2}),expected)
 assert torch.equal(emotion_logits(m,x,{}),m(x))
 assert emotion_logits(m,x,{'translation_tta_pixels':2}).shape==(2,7)


def test_hybrid_keeps_fusdense_predictions_at_zero_initialization_without_download(monkeypatch):
 torch.set_num_threads(2);torch.manual_seed(906)
 def forbid(*args,**kwargs):raise AssertionError('Model construction must not download weights')
 monkeypatch.setattr(torch.hub,'load_state_dict_from_url',forbid)
 base=FusDenseNet(version=4,head_width=512);base.eval()
 hybrid=FusDenseNet(version=6,head_width=512);state=hybrid.state_dict()
 for k,v in base.state_dict().items():state[k]=v
 hybrid.load_state_dict(state);hybrid.eval();x=torch.rand(1,1,48,48)
 with torch.no_grad():before=base(x);after=hybrid(x)
 assert torch.equal(before,after)
 loss=torch.nn.functional.cross_entropy(hybrid(x),torch.tensor([1]));loss.backward()
 assert hybrid.transfer_head[-1].weight.grad.abs().sum()>0
 assert hybrid.fusion.cross.in_proj_weight.grad.abs().sum()>0


def test_probability_hybrid_matches_declared_mixture():
 torch.set_num_threads(2);torch.manual_seed(907)
 base=FusDenseNet(version=4,head_width=512);base.eval();model=FusDenseNet(version=7,head_width=512,transfer_mix=.75,transfer_base_temperature=1.3);state=model.state_dict()
 for k,v in base.state_dict().items():state[k]=v
 model.load_state_dict(state);model.eval();expert=torch.tensor([.2,-.5,.3,.7,1.,.4,-.2])
 with torch.no_grad():
  model.transfer_head[-1].bias.copy_(expert);x=torch.rand(1,1,48,48);expected=.25*(base(x)/1.3).softmax(1)+.75*expert.softmax(0);actual=model(x).softmax(1)
 assert torch.allclose(actual,expected,atol=1e-6)
 assert torch.allclose(actual.sum(1),torch.ones(1))
