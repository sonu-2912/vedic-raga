"""Measure the overfitting gap at one fixed completed compact-model checkpoint."""
import json,argparse
from pathlib import Path
import torch
from scripts.colab_common import fixed_split,predictions,metrics,write_json,digest
from scripts.compact_residual_fusdense import model_from
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_capacity_20261108'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 global RUN
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default='reports/pure_capacity_20261108');a=ap.parse_args();RUN=(ROOT/a.run_dir).resolve();assert RUN.is_relative_to((ROOT/'reports').resolve())
 decision=read(RUN/'deployment_decision.json');torch.set_num_threads(4);torch.set_num_interop_threads(1)
 if (RUN/'candidate_frozen.json').exists():
  ck=torch.load(RUN/'best.pt',map_location='cpu',weights_only=True);source={'file':'best.pt','sha256':digest(RUN/'best.pt'),'selection':'The already frozen candidate; no subsequent model search'};validation=read(RUN/'fresh_validation_preflight.json')
 else:
  warm=torch.load(RUN/'warm_start.pt',map_location='cpu',weights_only=True);latest=torch.load(RUN/'full_latest.pt',map_location='cpu',weights_only=True);history=read(RUN/'full_history.json');kind=history[-1]['kind'];ck={**warm,'state_dict':latest['ema_state'] if kind=='EMA' else latest['state_dict'],'inference':{}};source={'file':'full_latest.pt','sha256':digest(RUN/'full_latest.pt'),'kind':kind,'selection':'Final completed epoch diagnostic only; rejected for deployment'};validation=read(RUN/f"full_validation_{len(history):02d}_{kind}.json")
 network=model_from(ck,torch.device('cpu'));parameter_updates=None
 if (RUN/'warm_start.pt').exists():
  starting=torch.load(RUN/'warm_start.pt',map_location='cpu',weights_only=True)['state_dict'];details=[]
  for name,param in network.named_parameters():
   if name in starting and starting[name].shape==param.shape:
    delta=param.detach()-starting[name];details.append({'name':name,'changed':not torch.equal(param.detach(),starting[name]),'delta_L2':float(delta.norm()),'parameters':param.numel()})
  parameter_updates={'tensors':details,'changed_emotion_path_tensors':sum(r['changed'] for r in details if not r['name'].startswith('reconstruct.')),'changed_native_refinement_tensors':sum(r['changed'] for r in details if '.refine_' in r['name'])};write_json(RUN/'final_parameter_update_audit.json',parameter_updates)
 manifest=read(RUN/'fer_training_manifest.json');ds=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'train');logits,y=predictions(network,ds,torch.device('cpu'));actual=metrics(logits,y);cache=torch.load(ROOT/'reports/pure_recall_recovery_v2_20261108/train_reference_logits.pt',map_location='cpu',weights_only=True);assert cache['source_checkpoint_sha256']==digest(RUN/'initial.pt');baseline=metrics(cache['logits'][:len(y)],cache['labels'][:len(y)]);current=read(RUN/'current_validation.json')
 result={'checkpoint':source,'deployed':False,'no_model_selection_from_this_audit':True,'protocol':'One fixed model, eval mode, all23503original unaugmented TRAIN faces and all3970validation faces. No dropout, flip averaging, thresholds or test data. Epoch running training accuracy is not used as a fixed-model estimate. TRAIN metrics are in-sample, not evidence of generalization.','baseline_train':baseline,'baseline_validation':current,'candidate_train':actual,'candidate_validation':validation,'parameter_updates':parameter_updates,'accuracy_gap_before':baseline['accuracy']-current['accuracy'],'accuracy_gap_after':actual['accuracy']-validation['accuracy'],'deployment_decision':decision};write_json(RUN/'fixed_checkpoint_generalization_audit.json',result);print(json.dumps({'source':source,'train_accuracy':actual['accuracy'],'val_accuracy':validation['accuracy'],'train_minus_validation_gap':result['accuracy_gap_after'],'baseline_gap':result['accuracy_gap_before']},indent=2))
if __name__=='__main__':main()
