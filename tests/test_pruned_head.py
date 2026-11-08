"""Pruning conversion fixtures, not recognition-performance measurements."""
import torch
from torch import nn
from scripts.train_pruned_head import folded_head,rank_neurons

def test_folded_normalization_preserves_classifier_logits_and_predictions():
    torch.set_num_threads(2);torch.manual_seed(419)
    head=nn.Sequential(nn.Linear(3276,256),nn.ReLU(),nn.Dropout(.5),nn.Linear(256,7));head.eval()
    mean=torch.randn(3276)*.1;std=torch.rand(3276)+.1;x=torch.randn(64,3276)
    with torch.no_grad():before=head((x-mean)/std);folded=folded_head(head,mean,std);after=folded(x)
    assert torch.allclose(before,after,rtol=1e-4,atol=2e-5)
    assert torch.equal(before.argmax(1),after.argmax(1))
    assert folded[0].weight.data_ptr()!=head[0].weight.data_ptr()

def test_neuron_ranking_is_not_changed_by_duplicating_one_class():
    torch.manual_seed(419)
    head=nn.Sequential(nn.Linear(6,12),nn.ReLU(),nn.Dropout(.5),nn.Linear(12,7));head.eval()
    features=torch.randn(21,6);labels=torch.arange(7).repeat_interleave(3)
    a,_=rank_neurons(head,features,labels,8)
    repeated=torch.cat([features,features[labels==0].repeat(10,1)]);repeated_y=torch.cat([labels,torch.zeros(30,dtype=torch.long)])
    b,_=rank_neurons(head,repeated,repeated_y,8)
    assert torch.equal(a,b) and len(a.unique())==8
