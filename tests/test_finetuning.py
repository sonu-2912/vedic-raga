import torch
from scripts.colab_common import set_training_mode

def test_frozen_bn_keeps_references_but_learns_affine_and_keeps_dropout():
    torch.manual_seed(91)
    model=torch.nn.Sequential(torch.nn.BatchNorm2d(3),torch.nn.Dropout(.2),torch.nn.Conv2d(3,1,1))
    reference=model[0].running_mean.clone();weight=model[0].weight.detach().clone()
    set_training_mode(model,True)
    assert model[1].training and not model[0].training
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    model(torch.randn(4,3,8,8)+3).square().mean().backward();optimizer.step()
    assert torch.equal(model[0].running_mean,reference)
    assert not torch.equal(model[0].weight,weight)
    assert all(p.requires_grad for p in model.parameters())


def test_spatial_upgrade_preserves_logits_then_learns_new_path():
    from backend.models import FusDenseNet
    torch.set_num_threads(2);torch.manual_seed(39)
    before=FusDenseNet(version=2);after=FusDenseNet(version=3)
    copied=after.state_dict()
    for name,value in before.state_dict().items():
        if name=='classifier.0.weight':
            copied[name].zero_();copied[name][:,:588]=value
        else:copied[name]=value
    after.load_state_dict(copied);before.eval();after.eval()
    x=torch.rand(2,1,48,48)
    with torch.no_grad():assert torch.allclose(before(x),after(x),atol=1e-5)
    optimizer=torch.optim.SGD(after.parameters(),lr=.01)
    torch.nn.functional.cross_entropy(after(x),torch.tensor([1,3])).backward()
    assert after.classifier[0].weight.grad[:,588:].abs().sum()>0
    optimizer.step()
    assert after.classifier[0].weight[:,588:].abs().sum()>0


def test_dense_spatial_upgrade_preserves_predictions_and_learns_new_features():
    from backend.models import FusDenseNet
    torch.set_num_threads(2);torch.manual_seed(117)
    before=FusDenseNet(version=3);after=FusDenseNet(version=4)
    state=after.state_dict()
    for name,value in before.state_dict().items():
        if name=='classifier.0.weight':
            state[name].zero_();state[name][:,:1356]=value
        else:state[name]=value
    after.load_state_dict(state);before.eval();after.eval();x=torch.rand(2,1,48,48)
    with torch.no_grad():assert torch.allclose(before(x),after(x),atol=1e-5)
    torch.nn.functional.cross_entropy(after(x),torch.tensor([1,3])).backward()
    assert after.classifier[0].weight.grad[:,1356:].abs().sum()>0


def test_optional_baseline_missing_reports_clean_error(tmp_path,monkeypatch):
    import backend.emotion as emotion
    monkeypatch.setattr(emotion,'ROOT',tmp_path)
    monkeypatch.setattr(emotion,'_models',{})
    import pytest
    with pytest.raises(RuntimeError,match='Optional Navrasa baseline weights'):
        emotion.load_model('navrasa')
