"""Rollback retained parameters within one pure FusDenseNet training lineage."""
import torch

def restore_native_state(trained,reference,include_head=False):
 state={name:value.clone() for name,value in trained.items()};restored=[]
 for name,value in reference.items():
  if '.refine_' in name or (name.startswith('classifier.') and not include_head):continue
  if name not in state or state[name].shape!=value.shape:raise ValueError('Incompatible native checkpoint parameter: '+name)
  state[name]=value.clone();restored.append(name)
 if not restored:raise ValueError('No compatible native parameters restored')
 return state,{'restored_tensor_names':restored,'classifier_restored':include_head,'new_residual_modules_retained':True,'inference_networks':1,'external_encoder':False}
