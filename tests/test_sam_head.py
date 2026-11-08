"""SAM optimizer contract fixtures; not model-performance results."""
import pytest
import torch
from torch import nn
from scripts.train_sam_head import sam_step

def test_sam_restores_exact_weights_before_zero_lr_step():
    torch.manual_seed(521);head=nn.Linear(2,2);old=[p.detach().clone() for p in head.parameters()];opt=torch.optim.SGD(head.parameters(),lr=0.)
    x=torch.tensor([[1.,2.],[2.,1.]]);y=torch.tensor([0,1])
    def objective():
        out=head(x);return nn.functional.cross_entropy(out,y),out
    first,_,second=sam_step(head,opt,objective)
    assert second>=first
    assert all(torch.equal(p,v) for p,v in zip(head.parameters(),old))

def test_sam_restores_parameters_even_if_second_objective_fails():
    torch.manual_seed(521);head=nn.Linear(2,2);old=[p.detach().clone() for p in head.parameters()];opt=torch.optim.AdamW(head.parameters(),lr=.001);calls=0
    def objective():
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('fixture second-pass failure')
        out=head(torch.ones(2,2));return out.square().mean(),out
    with pytest.raises(RuntimeError,match='fixture second-pass failure'):sam_step(head,opt,objective)
    assert all(torch.equal(p,v) for p,v in zip(head.parameters(),old))

def test_sam_reuses_same_dropout_mask_for_both_passes():
    torch.manual_seed(521);head=nn.Sequential(nn.Dropout(.5),nn.Linear(8,2));head.train();masks=[]
    hook=head[0].register_forward_hook(lambda module,args,out:masks.append(out.detach().clone()))
    opt=torch.optim.SGD(head.parameters(),lr=.001)
    def objective():
        out=head(torch.ones(4,8));return nn.functional.cross_entropy(out,torch.tensor([0,1,0,1])),out
    sam_step(head,opt,objective);hook.remove()
    assert len(masks)==2 and torch.equal(masks[0],masks[1])
