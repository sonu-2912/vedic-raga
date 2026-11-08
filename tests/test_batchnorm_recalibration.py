import copy
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scripts.recalibrate_pure_batchnorm import reestimate_batchnorm

def test_population_recalibration_updates_only_bn_buffers_and_ignores_labels():
 torch.manual_seed(41);model=nn.Sequential(nn.BatchNorm2d(1),nn.Flatten(),nn.Dropout(.5),nn.Linear(16,7)).eval();other=copy.deepcopy(model);x=torch.rand(16,1,4,4)*2+4
 a=reestimate_batchnorm(model,DataLoader(TensorDataset(x,torch.zeros(16)),batch_size=4));b=reestimate_batchnorm(other,DataLoader(TensorDataset(x,torch.arange(16)),batch_size=4))
 assert a['all_learned_parameters_bit_identical'] and a['changed_buffers'] and torch.allclose(model[0].running_mean,x.mean((0,2,3)))
 assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items());assert not model.training

def test_model_without_bn_is_unchanged():
 model=nn.Sequential(nn.Flatten(),nn.Linear(16,7)).eval();before=copy.deepcopy(model.state_dict());report=reestimate_batchnorm(model,DataLoader(TensorDataset(torch.rand(4,1,4,4),torch.zeros(4)),batch_size=2))
 assert not report['changed_buffers'] and all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
