"""Measure coverage separately from classifier accuracy; never hide abstentions."""
import json,hashlib
import torch
from torch.utils.data import DataLoader
from backend.catalog import ROOT
from backend.datasets import FER2013
from backend.models import FusDenseNet,FER_CLASSES
from backend.inference import emotion_logits

def main():
    torch.set_num_threads(4);checkpoint=ROOT/'models/fusdense.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    policy=json.loads((ROOT/'reports/expression_policy.json').read_text());assert policy['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest=json.loads((ROOT/'reports/emotion_evaluation_manifest.json').read_text());allowed={r['file'] for r in manifest['records']}
    ds=FER2013(ROOT/'data/private/fer2013-v1.zip','test');ds.rows=[r for r in ds.rows if str(r[0]) in allowed];assert len(ds)==len(allowed);ds.cache_images()
    model=FusDenseNet(**ck['architecture']);model.load_state_dict(ck['state_dict']);model.eval();n=correct=accepted=accepted_correct=0;classes={c:0 for c in FER_CLASSES}
    with torch.inference_mode():
        for x,y in DataLoader(ds,batch_size=64):
            probs=emotion_logits(model,x,ck.get('inference')).softmax(1);values,indices=probs.topk(2,dim=1)
            for truth,top,confidence,margin in zip(y.tolist(),indices[:,0].tolist(),values[:,0].tolist(),(values[:,0]-values[:,1]).tolist()):
                n+=1;correct+=int(truth==top)
                if FER_CLASSES[top] in policy['eligible_classes'] and confidence>=policy['minimum_confidence'] and margin>=policy['minimum_margin']:
                    accepted+=1;accepted_correct+=int(truth==top);classes[FER_CLASSES[top]]+=1
    report={'kind':'Actual selective-use evaluation; no model retraining or multimodal accuracy claim','checkpoint_sha256':policy['checkpoint_sha256'],'policy_fitted_on':'validation only','test_n':n,'all_image_classifier_accuracy':correct/n,'offered_suggestions':accepted,'withheld_suggestions':n-accepted,'coverage':accepted/n,'accuracy_among_offered_suggestions':accepted_correct/accepted if accepted else None,'offered_by_predicted_class':classes,'limitation':'Conditional accuracy covers only the accepted subset, not all seven emotions. Every suggestion still needs user confirmation. The full classifier remains about 50% accurate.'}
    (ROOT/'reports/selective_face_evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
