"""Compact pure FusDenseNet with native residual dense feature refinement."""
import torch
from torch import nn
from backend.models import FusDenseNet,FER_CLASSES
from scripts.compact_fusdense import coarse_spatial_features,model_from as previous_model_from

class CompactResidualFusDenseNet(FusDenseNet):
 def __init__(self,classes=7,attention=True,dilated=True,version=11,head_width=256,**kwargs):
  if version!=11:raise ValueError('Experimental native residual compact variant requires version11')
  super().__init__(classes=classes,attention=attention,dilated=dilated,version=8,head_width=head_width,**kwargs)
  self.classifier=nn.Sequential(nn.Linear(1260,head_width),nn.ReLU(),nn.Dropout(.5),nn.Linear(head_width,classes))
  self._coarse_hook=self.classifier.register_forward_pre_hook(lambda module,args:(coarse_spatial_features(args[0]),))

def model_from(checkpoint,device):
 if checkpoint['architecture'].get('version')!=11:return previous_model_from(checkpoint,device)
 if checkpoint.get('synthetic') or checkpoint.get('classes')!=FER_CLASSES:raise ValueError('Real-data seven-class checkpoint required')
 m=CompactResidualFusDenseNet(**checkpoint['architecture']).to(device);m.load_state_dict(checkpoint['state_dict']);return m

def initialize_from_compact(checkpoint):
 arch={**checkpoint['architecture'],'version':11};network=CompactResidualFusDenseNet(**arch);state=network.state_dict()
 for name,value in checkpoint['state_dict'].items():
  if name not in state or state[name].shape!=value.shape:raise ValueError('Incompatible compact checkpoint')
  state[name]=value
 network.load_state_dict(state);return network,arch
