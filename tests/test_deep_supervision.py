"""Training-contract fixtures, never facial accuracy measurements."""
import torch
from scripts.train_deep_supervision import FusedCompanion,mixed_ce,qualifies

def test_companion_gradient_reaches_fused_path_not_spatial_shortcuts():
    torch.set_num_threads(2);torch.manual_seed(312)
    x=torch.randn(5,3276,requires_grad=True);head=FusedCompanion();head.eval();head.capture(None,(x,))
    torch.nn.functional.cross_entropy(head(),torch.tensor([0,1,2,3,4])).backward()
    assert x.grad[:,:588].abs().sum()>0
    assert torch.equal(x.grad[:,588:],torch.zeros_like(x.grad[:,588:]))

def test_shortcut_dropout_keeps_fused_features_and_eval_is_unmodified():
    torch.manual_seed(312);x=torch.ones(128,3276,requires_grad=True);head=FusedCompanion();head.shortcut_drop=.2;head.train()
    changed=head.capture(None,(x,))[0]
    assert torch.equal(changed[:,:588],x[:,:588])
    assert bool((changed[:,588:]==0).any()) and bool((changed[:,588:]>1).any())
    head.eval();assert head.capture(None,(x,)) is None
    assert torch.equal(head.features,x[:,:588])

def test_mixed_ce_preserves_both_supervised_targets_and_gradient():
    logits=torch.tensor([[1.,2.,3.],[2.,1.,0.]],requires_grad=True);a=torch.tensor([0,1]);b=torch.tensor([2,0]);weights=torch.tensor([1.,2.,3.]);lam=.3
    expected=lam*torch.nn.functional.cross_entropy(logits,a,weight=weights,label_smoothing=.02)+(1-lam)*torch.nn.functional.cross_entropy(logits,b,weight=weights,label_smoothing=.02)
    actual=mixed_ce(logits,a,b,lam,weights);assert torch.allclose(actual,expected);actual.backward();assert logits.grad.abs().sum()>0

def test_precision_regression_cannot_qualify_despite_higher_accuracy():
    def report(accuracy,precision):return {'accuracy':accuracy,'classification_report':{'macro avg':{'precision':precision,'recall':.5,'f1-score':.5},'disgust':{'precision':.4,'recall':.3,'f1-score':.34}}}
    base=report(.58,.55)
    assert not qualifies(report(.61,.54),base)
    assert qualifies(report(.59,.56),base)


def test_strict_integration_retains_current_model_when_precision_regresses():
    from scripts.colab_evaluate import integration_gate
    c={k:{'difference':.01} for k in ['accuracy','macro_precision','macro_recall','macro_f1','disgust_precision','disgust_recall','disgust_f1']}
    c['macro_precision']['difference']=-.001
    assert integration_gate(c,False)
    assert not integration_gate(c,True)
    c['macro_precision']['difference']=0.
    assert integration_gate(c,True)


def test_real_fusdense_companion_updates_attention_without_inference_change():
    from backend.models import FusDenseNet
    torch.set_num_threads(2);torch.manual_seed(312)
    network=FusDenseNet(version=4,head_width=512);network.eval()
    x=torch.rand(2,1,48,48)
    with torch.no_grad():before=network(x)
    companion=FusedCompanion();companion.eval();companion.shortcut_drop=.2
    handle=network.classifier[0].register_forward_pre_hook(companion.capture)
    out=network(x)
    assert torch.equal(before,out.detach())
    torch.nn.functional.cross_entropy(companion(),torch.tensor([1,4])).backward()
    gradient=network.more_fusion[0].cross.in_proj_weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert network.classifier[0].weight.grad is None
    handle.remove()
    with torch.no_grad():assert torch.equal(network(x),before)
