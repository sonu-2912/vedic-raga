"""TRAIN-only BatchNorm buffer recalibration; no parameter or label fitting."""
import torch
from torch.optim.swa_utils import update_bn

def reestimate_batchnorm(model,loader):
 model.eval();parameters={name:p.detach().clone() for name,p in model.named_parameters()};before={name:t.detach().clone() for name,t in model.named_buffers()};update_bn(loader,model,device=torch.device('cpu'));model.eval()
 if not all(torch.equal(parameters[name],p) for name,p in model.named_parameters()):raise RuntimeError('BN recalibration unexpectedly changed learned parameters')
 changed=[name for name,t in model.named_buffers() if not torch.equal(before[name],t)]
 if any(not name.endswith(('running_mean','running_var','num_batches_tracked')) for name in changed):raise RuntimeError('Unexpected non-BatchNorm buffer change')
 return {'all_learned_parameters_bit_identical':True,'changed_buffers':changed,'method':'torch.optim.swa_utils.update_bn, installed torch2.5.1 implementation; TRAIN images only; labels ignored'}
