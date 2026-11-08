import torch
import pytest
from scripts.recall_manifold_common import manifold_recovery_loss
from scripts.recall_recovery_common import boundary_recovery_loss

def test_mixup_identity_reduces_to_unmixed_supervision_without_reference_gradient():
 logits=torch.randn(4,7,requires_grad=True);ref=torch.randn(4,7,requires_grad=True);labels=torch.tensor([1,3,5,6]);mirror=torch.tensor([False,True,False,True]);weights=torch.ones(7);permutation=torch.tensor([1,0,3,2])
 loss,ce,ret=manifold_recovery_loss(logits,logits,labels,ref,weights,mirror,.5,1.,permutation,1.)
 expected,_,_=boundary_recovery_loss(logits,logits,labels,ref,weights,mirror,.5,1.)
 assert torch.allclose(loss,expected);loss.backward();assert logits.grad is not None and ref.grad is None

def test_mixup_uses_both_supervised_targets_and_preserves_gradients():
 mixed=torch.zeros(2,7,requires_grad=True);original=torch.zeros(2,7,requires_grad=True);labels=torch.tensor([3,5]);reference=torch.zeros(2,7,requires_grad=True);permutation=torch.tensor([1,0]);mirror=torch.zeros(2,dtype=torch.bool)
 loss,_,_=manifold_recovery_loss(mixed,original,labels,reference,torch.ones(7),mirror,.5,0.,permutation,.75);loss.backward()
 assert mixed.grad[0,3]<mixed.grad[0,5]<0 and original.grad is not None and reference.grad is None
 with pytest.raises(ValueError):manifold_recovery_loss(mixed,original,labels,reference,torch.ones(7),mirror,.5,0.,permutation,1.1)
