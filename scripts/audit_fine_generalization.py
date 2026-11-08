"""Fixed-checkpoint audit for a fine-spatial head with byte-identical backbone."""
import json
from pathlib import Path
import torch
from scripts.colab_common import model_from,metrics,digest,write_json
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_fine_head_20261109'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 torch.set_num_threads(4);decision=read(RUN/'deployment_decision.json');path=RUN/('best.pt' if (RUN/'candidate_frozen.json').exists() else 'warm_start.pt');ck=torch.load(path,map_location='cpu',weights_only=True);old=torch.load(RUN/'initial.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v,ck['state_dict'][k]) for k,v in old['state_dict'].items() if not k.startswith('classifier.'));network=model_from(ck,torch.device('cpu')).eval();train=torch.load(ROOT/'reports/pruned_head_20261107/train_features.pt',map_location='cpu',weights_only=True,mmap=True);val=torch.load(ROOT/'reports/pruned_head_20261107/val_features.pt',map_location='cpu',weights_only=True,mmap=True);manifest=read(ROOT/'reports/fer_training_manifest.json')
 for cache,split in [(train,'train'),(val,'val')]:assert cache['manifest_sha256']==digest(ROOT/'reports/fer_training_manifest.json') and cache['labels'].tolist()==[r['label'] for r in manifest[split]]
 outputs=[]
 with torch.no_grad():
  for start in range(0,len(train['labels']),512):outputs.append(network.classifier(train['features'][start:start+512]))
  actual=metrics(torch.cat(outputs),train['labels']);validation=metrics(network.classifier(val['features']),val['labels'])
 expected=read(RUN/('fresh_validation_preflight.json' if (RUN/'candidate_frozen.json').exists() else 'warm_validation.json'));assert validation['accuracy']==expected['accuracy'] and validation['classification_report']['macro avg']['f1-score']==expected['classification_report']['macro avg']['f1-score'];reference=torch.load(ROOT/'reports/pure_recall_recovery_v2_20261108/train_reference_logits.pt',map_location='cpu',weights_only=True);assert reference['source_checkpoint_sha256']==digest(RUN/'initial.pt');baseline=metrics(reference['logits'][:len(train['labels'])],reference['labels'][:len(train['labels'])]);current=read(RUN/'current_validation.json');result={'checkpoint':{'file':path.name,'sha256':digest(path),'eligible_for_deployment':decision['eligible']},'protocol':'All23503original TRAIN faces,3970validation faces, fixed eval-mode classifier. Cached native features valid because every backbone tensor is byte-identical to the verified source. No augmentation/dropout or new test data. TRAIN-only source/label/manifest provenance verified.','baseline_train':baseline,'baseline_validation':current,'candidate_train':actual,'candidate_validation':validation,'accuracy_gap_before':baseline['accuracy']-current['accuracy'],'accuracy_gap_after':actual['accuracy']-validation['accuracy'],'no_model_selection_from_this_audit':True};write_json(RUN/'fixed_checkpoint_generalization_audit.json',result);print(json.dumps({'train_accuracy':actual['accuracy'],'validation_accuracy':validation['accuracy'],'gap':result['accuracy_gap_after'],'deployed':False},indent=2))
if __name__=='__main__':main()
