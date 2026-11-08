"""Select completed experiments using validation only, before one test evaluation."""
import json,shutil
from pathlib import Path
import torch
from scripts.colab_common import save_checkpoint,write_json,digest

ROOT=Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(2)
    full=ROOT/'reports/real_improvement_run';head=ROOT/'reports/head_refinement_run';out=ROOT/'reports/selected_improvement_run'
    head_runs=[head,ROOT/'reports/head_refinement_continued_run']
    assert all((r/'completion.json').exists() for r in [full,*head_runs]),'All experiments must finish first'
    assert not (out/'bundle_manifest.json').exists(),'Selection is locked after test evaluation'
    initial=json.loads((full/'initial_validation.json').read_text());floor=initial['accuracy'];initial_f1=initial['macro avg']['f1-score']
    base=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True)
    candidates=[]
    for file in sorted((full/'epoch_snapshots').glob('*.pt')):
        snapshot=torch.load(file,map_location='cpu',weights_only=True);row=snapshot['history'][-1]
        candidates.append({'path':str(file),'kind':'full_network','epoch':row['epoch'],'val_accuracy':row['val_accuracy'],'val_macro_f1':row['val_macro_f1']})
    for hrun in head_runs:
        h=torch.load(hrun/'best.pt',map_location='cpu',weights_only=True)
        hr=json.loads((hrun/('best_validation.json' if (hrun/'best_validation.json').exists() else 'initial_validation.json')).read_text())
        candidates.append({'path':str(hrun/'best.pt'),'run_dir':str(hrun),'kind':'head_refinement','epoch':h['epoch'],'val_accuracy':hr['accuracy'],'val_macro_f1':hr['macro avg']['f1-score']})
    eligible=[c for c in candidates if c['val_accuracy']+1e-7>=floor and c['val_macro_f1']>initial_f1]
    chosen=max(eligible or candidates,key=lambda c:(c['val_macro_f1'],c['val_accuracy']))
    source=Path(chosen['run_dir']) if chosen['kind']=='head_refinement' else full
    if chosen['kind']=='head_refinement':checkpoint=torch.load(chosen['path'],map_location='cpu',weights_only=True)
    else:
        snapshot=torch.load(chosen['path'],map_location='cpu',weights_only=True)
        checkpoint={k:v for k,v in base.items() if k not in ['state_dict','calibration','inference']}
        checkpoint.update(state_dict=snapshot['state_dict'],epoch=chosen['epoch'],validation_macro_f1=chosen['val_macro_f1'],inference={},colab_settings=snapshot['settings'],colab_improved=True)
    checkpoint['experiment_selection']={'rule':'Maximum validation macro F1 among candidates at least matching initial validation accuracy; otherwise highest macro F1 with regression disclosed','selected':chosen,'initial_val_accuracy':floor,'initial_val_macro_f1':initial_f1,'test_accessed':False}
    out.mkdir(exist_ok=True);save_checkpoint(out/'best.pt',checkpoint)
    for name in ['settings.json','fer_training_manifest.json','training_history.json']:shutil.copy2(source/name,out/name)
    write_json(out/'selection.json',{'chosen':chosen,'eligible_count':len(eligible),'all_candidates':candidates,'test_accessed':False,'source_sha256':digest(chosen['path'])})
    print(json.dumps(checkpoint['experiment_selection'],indent=2))
if __name__=='__main__':main()
