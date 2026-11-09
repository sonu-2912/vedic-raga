"""TRAIN-only manifold mixup for the native pure FusDenseNet classifier."""
import torch
from torch.nn import functional as F
from scripts.recall_recovery_common import training_sample_weights,boundary_recovery_loss

def manifold_recovery_loss(mixed_logits,original_logits,labels,reference,weights,mirrored,mirror_weight,retention_weight,permutation,lam):
 if not 0<=lam<=1:raise ValueError('Mixup coefficient outside [0,1]')
 w=training_sample_weights(labels,weights,mirrored,mirror_weight);other=labels[permutation];other_w=w[permutation]
 mixed=(lam*w*F.cross_entropy(mixed_logits,labels,reduction='none')+(1-lam)*other_w*F.cross_entropy(mixed_logits,other,reduction='none')).sum()/(lam*w+(1-lam)*other_w).sum()
 _,anchor,retention=boundary_recovery_loss(original_logits,original_logits,labels,reference,weights,mirrored,mirror_weight,retention_weight)
 ce=.75*mixed+.25*anchor
 return ce+retention_weight*retention,ce,retention
