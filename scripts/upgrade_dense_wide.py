"""Disclosed v4: retain first dense-block spatial features; zero-init new path."""
from pathlib import Path
import torch
from backend.models import FusDenseNet
from scripts.colab_common import save_checkpoint,digest,write_json
root=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
source=root/'models/fusdense.pt';ck=torch.load(source,map_location='cpu',weights_only=True)
assert ck['architecture']['version']==3
old=FusDenseNet(**ck['architecture']);old.load_state_dict(ck['state_dict']);old.eval()
arch={**ck['architecture'],'version':4,'head_width':512};new=FusDenseNet(**arch);state=new.state_dict()
for key,value in ck['state_dict'].items():
 if key=='classifier.0.weight':state[key][:value.shape[0]].zero_();state[key][:value.shape[0],:value.shape[1]]=value
 elif key=='classifier.0.bias':state[key][:len(value)]=value
 elif key=='classifier.3.weight':state[key].zero_();state[key][:,:value.shape[1]]=value
 else:state[key]=value
new.load_state_dict(state);new.eval();torch.manual_seed(117)
with torch.inference_mode():
 x=torch.rand(4,1,48,48);difference=float((old(x)-new(x)).abs().max())
assert difference<1e-5
path=root/'reports/audit_20261107/dense_wide_initial.pt'
if path.exists():raise ValueError('Initialization already preserved')
base={k:v for k,v in ck.items() if k not in ['state_dict','inference','calibration']}
provenance={'source_sha256':digest(source),'added_features':1920,'classifier_features':3276,'parameters':sum(p.numel() for p in new.parameters()),'initial_logit_max_difference':difference,'synthetic_performance':False,'contract_fixture_only':True,'description':'Project v4 retains already-computed first dense-block maps before 8x8 pooling; 4x4 mean/max features and a widened512-unit MLP. All fusion, attention and prior features retained. No exact paper reproduction.'}
save_checkpoint(path,{**base,'state_dict':new.state_dict(),'architecture':arch,'inference':{},'dense_spatial_variant':provenance})
write_json(path.with_suffix('.json'),provenance);print(provenance)
