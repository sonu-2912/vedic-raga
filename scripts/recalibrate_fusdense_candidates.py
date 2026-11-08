"""Training-only BN recalibration and within-run weight averaging candidates.

No teacher model, new architecture or test access. All selection remains on the
fixed validation split. Averaging is only between checkpoints from one recipe.
"""
import argparse,json,copy
from pathlib import Path
import torch
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from scripts.colab_common import fixed_split,model_from,predictions,save_checkpoint,write_json,digest
from backend.models import FER_CLASSES

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--data',required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(a.threads);run=Path(a.run_dir);assert (run/'completion.json').exists(),'Finish training before candidate construction'
    out=run/'natural_bn_candidates';out.mkdir(exist_ok=True)
    if (out/'candidate_manifest.json').exists():print('Existing candidate construction retained');return
    manifest=json.loads((run/'fer_training_manifest.json').read_text());settings=json.loads((run/'settings.json').read_text())
    initial=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    train=fixed_split(a.data,manifest,'train',augment=False);val=fixed_split(a.data,manifest,'val')
    snapshots=[]
    for path in (run/'epoch_snapshots').glob('*.pt'):
        state=torch.load(path,map_location='cpu',weights_only=True);snapshots.append((state['history'][-1]['epoch'],path))
    snapshots.sort();assert snapshots
    chosen=[snapshots[-1]]
    outputs=[]
    for mode in ['last_natural_bn','last3_average_natural_bn']:
        available=snapshots[-3:] if mode.startswith('last3') else snapshots[-1:]
        if mode.startswith('last3') and len(available)<2:continue
        states=[torch.load(path,map_location='cpu',weights_only=True)['state_dict'] for _,path in available]
        state={key:torch.stack([s[key] for s in states]).mean(0) if torch.is_floating_point(value) else states[-1][key] for key,value in states[0].items()}
        checkpoint={k:v for k,v in initial.items() if k not in ['state_dict','inference','calibration']};checkpoint.update(state_dict=state,inference={},colab_settings=settings,training_stage='full_network_balanced_finetune',epoch=available[-1][0])
        model=model_from(checkpoint,torch.device('cpu'))
        # update_bn changes running buffers only; all parameters are checked below.
        parameters=[parameter.detach().clone() for parameter in model.parameters()]
        update_bn(DataLoader(train,batch_size=128,shuffle=False),model,device=torch.device('cpu'))
        assert all(torch.equal(before,after) for before,after in zip(parameters,model.parameters()))
        logits,truth=predictions(model,val,torch.device('cpu'),128)
        report=classification_report(truth,logits.argmax(1),labels=list(range(7)),target_names=FER_CLASSES,output_dict=True,zero_division=0)
        provenance={'method':mode,'source_checkpoints':[{'epoch':epoch,'sha256':digest(path)} for epoch,path in available],'BN_data':'fixed training images only, natural unaugmented distribution; no test/validation images update statistics','teacher':None,'test_accessed':False}
        checkpoint.update(state_dict=model.state_dict(),validation_macro_f1=report['macro avg']['f1-score'],candidate_method=provenance)
        path=out/(mode+'.pt');save_checkpoint(path,checkpoint);write_json(out/(mode+'_validation.json'),report)
        outputs.append({'path':str(path),'report':str(out/(mode+'_validation.json')),'provenance':provenance})
        print(json.dumps({'mode':mode,'validation_accuracy':report['accuracy'],'macro_f1':report['macro avg']['f1-score'],'disgust_f1':report['disgust']['f1-score']}),flush=True)
    write_json(out/'candidate_manifest.json',{'candidates':outputs,'test_accessed':False})
if __name__=='__main__':main()
