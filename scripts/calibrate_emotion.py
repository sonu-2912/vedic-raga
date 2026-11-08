"""Select optional flip inference and fit temperature on validation; never test."""
import argparse,json,time,hashlib
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,accuracy_score
from backend.models import FusDenseNet
from backend.datasets import FER2013
from backend.catalog import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',default='models/fusdense.pt');p.add_argument('--manifest',default='reports/fer_training_manifest.json');p.add_argument('--threads',type=int,default=8);a=p.parse_args();torch.set_num_threads(a.threads)
    path=ROOT/a.checkpoint;ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('synthetic'):raise ValueError('Calibration requires a real-data checkpoint')
    manifest=json.loads((ROOT/a.manifest).read_text());allowed={r['file'] for r in manifest['val']}
    ds=FER2013(a.data,'val',denoising=ck.get('denoising','none'));ds.rows=[row for row in ds.rows if str(row[0]) in allowed];ds.cache_images()
    if len(ds)!=len(allowed):raise ValueError('Validation manifest does not match dataset')
    model=FusDenseNet(**ck['architecture']);model.load_state_dict(ck['state_dict']);model.eval();raw=[];flipped=[];truth=[];start=time.perf_counter()
    with torch.inference_mode():
        for x,y in DataLoader(ds,batch_size=64):
            raw.append(model(x));flipped.append(model(torch.flip(x,[-1])));truth.append(y)
    logits=torch.cat(raw);flip_logits=(logits+torch.cat(flipped))/2;labels=torch.cat(truth)
    score=lambda z:float(f1_score(labels.tolist(),z.argmax(1).tolist(),labels=list(range(7)),average='macro',zero_division=0))
    single_f1,flip_f1=score(logits),score(flip_logits)
    use_flip=flip_f1>single_f1+.002
    chosen=flip_logits if use_flip else logits
    log_temperature=torch.zeros(1,requires_grad=True);optimizer=torch.optim.LBFGS([log_temperature],lr=.1,max_iter=60,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(chosen/torch.exp(log_temperature).clamp(.05,10),labels);loss.backward();return loss
    optimizer.step(closure);temperature=float(torch.exp(log_temperature).clamp(.05,10).detach())
    report={'split':'validation only, same fixed split used for checkpoint selection','n':len(ds),'single_macro_f1':single_f1,'flip_macro_f1':flip_f1,'flip_selection_rule':'Use two-view mean logits only if validation macro F1 improves by more than 0.002','horizontal_flip_tta':use_flip,'temperature':temperature,'nll_before':float(torch.nn.functional.cross_entropy(chosen,labels)),'nll_after':float(torch.nn.functional.cross_entropy(chosen/temperature,labels)),'accuracy_unchanged_by_temperature':float(accuracy_score(labels.tolist(),chosen.argmax(1).tolist())),'seconds':time.perf_counter()-start,'limitations':'Calibration fitting and checkpoint selection reuse validation. Test is not read. No webcam-domain or independent calibration validation.'}
    ck['inference']={'horizontal_flip_tta':use_flip,'temperature':temperature};ck['calibration']=report
    temporary=path.with_suffix('.calibrating.tmp');torch.save(ck,temporary);temporary.replace(path)
    report['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(ROOT/'reports/calibration.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
