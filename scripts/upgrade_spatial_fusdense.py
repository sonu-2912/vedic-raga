"""Initialize a disclosed spatial-pooling FusDenseNet variant without changing logits."""
from pathlib import Path
import torch
from backend.models import FusDenseNet
from scripts.colab_common import save_checkpoint,digest,write_json
root=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
source=root/'reports/further_baseline/fusdense.pt';ck=torch.load(source,map_location='cpu',weights_only=True)
old=FusDenseNet(**ck['architecture']);old.load_state_dict(ck['state_dict']);old.eval()
architecture={**ck['architecture'],'version':3};new=FusDenseNet(**architecture)
state=new.state_dict()
for key,value in ck['state_dict'].items():
    if key=='classifier.0.weight':state[key].zero_();state[key][:,:value.shape[1]]=value
    else:state[key]=value
new.load_state_dict(state);new.eval()
torch.manual_seed(91)
with torch.inference_mode():
    fixture=torch.rand(8,1,48,48);difference=float((old(fixture)-new(fixture)).abs().max())
assert difference<1e-5
output=root/'reports/spatial_fusdense_initial.pt'
if output.exists():raise ValueError('Existing spatial initialization preserved')
base={k:v for k,v in ck.items() if k not in ['state_dict','inference','calibration']}
provenance={'variant':'Project FusDenseNet v3: higher-resolution stem mean/max spatial shortcut; all original fusion/dense/attention/pooling paths retained','source_sha256':digest(source),'initialization_contract_max_logit_difference':difference,'contract_fixture':'Random tensors test initialization equivalence only, not emotion performance','extra_classifier_features':768,'total_classifier_features':1356,'exact_paper_reproduction':False}
save_checkpoint(output,{**base,'state_dict':new.state_dict(),'architecture':architecture,'inference':{},'spatial_variant':provenance})
write_json(root/'reports/spatial_variant_initialization.json',{**provenance,'parameters':sum(p.numel() for p in new.parameters()),'output_sha256':digest(output)})
print(provenance)
