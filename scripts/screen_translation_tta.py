"""Pre-test validation-only robustness screen on one declared pixel recipe."""
import json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from backend.inference import emotion_logits
from scripts.colab_common import fixed_split,model_from,digest,metrics,write_json
from scripts.train_spatial_generalization import selected
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/translation_screen_20261108'
def main():
 RUN.mkdir(exist_ok=True)
 if (RUN/'validation.json').exists():raise RuntimeError('Completed screen preserved')
 torch.set_num_threads(2)
 write_json(RUN/'protocol.json',{'source_sha256':digest(ROOT/'models/fusdense.pt'),'recipe':{'translation_tta_pixels':2,'horizontal_flip_tta':False,'temperature':1.},'selection':'Both accuracy and disgust recall must strictly improve; macro/disgustF1 no lower than raw baseline. No test pixels opened.','disclosure':'Added during v5training, after raw baseline and epoch1validation were inspected, before any new test access. Five fixed whole-image views, no class thresholds, no exclusions.'})
 ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True);m=model_from(ck,torch.device('cpu'));m.eval();manifest=json.loads((ROOT/'reports/fer_training_manifest.json').read_text(encoding='utf-8'));val=fixed_split(ROOT/'data/private/fer2013-v1.zip',manifest,'val');outputs=[];truth=[];start=time.perf_counter()
 with torch.inference_mode():
  for i,(x,y) in enumerate(DataLoader(val,batch_size=128)):
   outputs.append(emotion_logits(m,x,{'translation_tta_pixels':2}));truth.append(y)
   if (i+1)%8==0:print(json.dumps({'validation_faces':sum(len(y) for y in truth),'total':len(val)}),flush=True)
 r=metrics(torch.cat(outputs),torch.cat(truth));r.update(source_checkpoint_sha256=digest(ROOT/'models/fusdense.pt'),split='val',recipe={'translation_tta_pixels':2,'temperature':1.},test_images_opened=False,seconds=time.perf_counter()-start)
 base=json.loads((ROOT/'reports/bottleneck_audit_20261108/validation_raw_evaluation.json').read_text(encoding='utf-8'));r['qualified']=selected(r,base);write_json(RUN/'validation.json',r);print(json.dumps({k:v for k,v in r.items() if k!='predictions'},indent=2),flush=True)
if __name__=='__main__':main()
