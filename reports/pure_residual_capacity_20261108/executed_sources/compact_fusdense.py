"""Experimental pure FusDenseNet capacity control, isolated from the deployed app."""
import torch
from torch import nn
from torch.nn import functional as F
from backend.models import FusDenseNet,FER_CLASSES
from scripts.colab_common import model_from as legacy_model_from
from scripts.recall_recovery_common import training_sample_weights,boundary_recovery_loss

def coarse_spatial_features(features):
 if features.ndim!=2 or features.shape[1]!=3276:raise ValueError('Expected 3276 native v4 FusDenseNet features')
 outputs=[features[:,:588]]
 for start,channels in [(588,24),(1356,60)]:
  n=channels*16;avg=features[:,start:start+n].reshape(-1,channels,4,4);mx=features[:,start+n:start+2*n].reshape(-1,channels,4,4)
  pooled=torch.cat([F.avg_pool2d(avg,2).flatten(1),F.max_pool2d(mx,2).flatten(1)],1)
  outputs.append(F.layer_norm(pooled,(channels*8,)))
 return torch.cat(outputs,1)

class CompactFusDenseNet(FusDenseNet):
 """v9: the native v4 feature extractor, 2x2 region pooling and a smaller MLP.

 An experimental project extension; not an exact author-specified architecture.
 A fixed differentiable pooling operation replaces the fine spatial head input.
 """
 def __init__(self,classes=7,attention=True,dilated=True,version=9,head_width=128,**kwargs):
  if version!=9:raise ValueError('Compact experimental architecture requires version9')
  super().__init__(classes=classes,attention=attention,dilated=dilated,version=4,head_width=head_width,**kwargs)
  self.classifier=nn.Sequential(nn.Linear(1260,head_width),nn.ReLU(),nn.Dropout(.5),nn.Linear(head_width,classes))
  self._coarse_hook=self.classifier.register_forward_pre_hook(lambda module,args:(coarse_spatial_features(args[0]),))

def model_from(checkpoint,device):
 if checkpoint['architecture'].get('version')!=9:return legacy_model_from(checkpoint,device)
 if checkpoint.get('synthetic') or checkpoint.get('classes')!=FER_CLASSES:raise ValueError('Real-data seven-class FER checkpoint required')
 m=CompactFusDenseNet(**checkpoint['architecture']).to(device);m.load_state_dict(checkpoint['state_dict']);return m

def capacity_loss(logits,deterministic,labels,reference,weights,retention_weight,smoothing):
 # Smooth within each labelled sample, then apply that sample's class cost.
 ce=(F.cross_entropy(logits,labels,label_smoothing=smoothing,reduction='none')*weights[labels]).sum()/weights[labels].sum()
 mirror=torch.zeros_like(labels,dtype=torch.bool)
 _,_,ret=boundary_recovery_loss(deterministic,deterministic,labels,reference,weights,mirror,1.,retention_weight)
 return ce+retention_weight*ret,ce,ret
