"""Build conservative expression-use limits from the fixed validation split."""
import json,hashlib
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from backend.catalog import ROOT
from backend.models import FusDenseNet,FER_CLASSES
from backend.datasets import FER2013
from backend.inference import emotion_logits

def main():
    torch.set_num_threads(4);path=ROOT/'models/fusdense.pt';raw=path.read_bytes();ck=torch.load(path,map_location='cpu',weights_only=True)
    manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text());allowed={r['file'] for r in manifest['val']}
    ds=FER2013(ROOT/'data/private/fer2013-v1.zip','val');ds.rows=[r for r in ds.rows if str(r[0]) in allowed];ds.cache_images()
    assert len(ds)==len(allowed)
    model=FusDenseNet(**ck['architecture']);model.load_state_dict(ck['state_dict']);model.eval();truth=[];pred=[]
    with torch.inference_mode():
        for x,y in DataLoader(ds,batch_size=64):truth+=y.tolist();pred+=emotion_logits(model,x,ck.get('inference')).argmax(1).tolist()
    report=classification_report(truth,pred,labels=list(range(7)),target_names=FER_CLASSES,output_dict=True,zero_division=0)
    policy={'checkpoint_sha256':hashlib.sha256(raw).hexdigest(),'split':'fixed validation only','n':len(ds),'minimum_confidence':.7,'minimum_margin':.2,'minimum_class_f1':.6,'minimum_class_recall':.6,'eligible_classes':[c for c in FER_CLASSES if report[c]['f1-score']>=.6 and report[c]['recall']>=.6],'class_report':report,'limitation':'Same validation partition used for checkpoint selection/calibration. Conservative operational abstention, not proof of recognition quality or multimodal improvement.'}
    (ROOT/'reports/expression_policy.json').write_text(json.dumps(policy,indent=2));print(json.dumps(policy,indent=2))

if __name__=='__main__':main()
