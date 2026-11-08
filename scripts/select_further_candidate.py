"""Finalize one validation-only candidate before any new test-set access."""
import json,shutil,copy
from pathlib import Path
import torch
from scripts.colab_common import digest,save_checkpoint,write_json
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'reports/further_selected_run'
    if out.exists():raise ValueError('Selection already frozen; preserve its test protocol')
    baseline=json.loads((ROOT/'reports/further_fusdense_run/initial_validation.json').read_text())
    profiles=[];payloads={}
    for name in ['spatial_head_run','spatial_flip_head_run','spatial_disgust_head_run','spatial_full_run']:
        run=ROOT/'reports'/name
        assert (run/'completion.json').exists(),name+' still training'
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
        if name=='spatial_full_run' and ck.get('colab_improved'):
            ck['head_refinement']=False;ck['training_stage']='full_network_finetune'
        reportpath=run/'best_validation.json'
        report=json.loads(reportpath.read_text()) if reportpath.exists() else json.loads((run/'initial_validation.json').read_text())
        # A run that retained its starting weights has the starting report.
        if not ck.get('colab_improved',True):report=json.loads((run/'initial_validation.json').read_text())
        profiles.append({'name':name,'path':str(run/'best.pt'),'sha256':digest(run/'best.pt'),'accuracy':report['accuracy'],'macro_f1':report['macro avg']['f1-score'],'disgust_f1':report['disgust']['f1-score'],'disgust_recall':report['disgust']['recall'],'epoch':ck['epoch']})
        payloads[name]=ck
        if name=='spatial_full_run':
            for path in sorted((run/'epoch_snapshots').glob('*.pt')):
                state=torch.load(path,map_location='cpu',weights_only=True);row=state['history'][-1];key=name+'/'+path.stem
                profiles.append({'name':key,'path':str(path),'sha256':digest(path),'accuracy':row['val_accuracy'],'macro_f1':row['val_macro_f1'],'disgust_f1':row['val_disgust_f1'],'disgust_recall':row['val_disgust_recall'],'epoch':row['epoch']})
                value=copy.deepcopy(ck);value.update(state_dict=state['state_dict'],epoch=row['epoch'],head_refinement=False,training_stage='full_network_finetune',validation_macro_f1=row['val_macro_f1'],colab_improved=True);payloads[key]=value
            candidate_manifest=run/'natural_bn_candidates/candidate_manifest.json'
            if candidate_manifest.exists():
                for item in json.loads(candidate_manifest.read_text())['candidates']:
                    path=Path(item['path']);report=json.loads(Path(item['report']).read_text());key=name+'/'+path.stem
                    value=torch.load(path,map_location='cpu',weights_only=True);value['head_refinement']=False
                    profiles.append({'name':key,'path':str(path),'sha256':digest(path),'accuracy':report['accuracy'],'macro_f1':report['macro avg']['f1-score'],'disgust_f1':report['disgust']['f1-score'],'disgust_recall':report['disgust']['recall'],'epoch':value['epoch']})
                    payloads[key]=value
    for p in profiles:
        p['eligible']=p['accuracy']>=baseline['accuracy'] and p['macro_f1']>=baseline['macro avg']['f1-score'] and p['disgust_f1']>=baseline['disgust']['f1-score']
        p['selection_score']=.75*p['macro_f1']+.25*p['disgust_f1']
    eligible=[p for p in profiles if p['eligible']];assert eligible,'No validation-eligible candidate'
    chosen=max(eligible,key=lambda p:p['selection_score']);ck=payloads[chosen['name']]
    run=ROOT/'reports'/chosen['name'].split('/')[0];out.mkdir()
    decision={'selected':chosen,'all_candidates':profiles,'test_accessed':False,'rule':'.75 validation macro F1 + .25 validation disgust F1, requiring accuracy/macro F1/disgust F1 no lower than initial raw validation','baseline_validation':baseline,'limitation':'Repeated validation selection can overfit, especially the 50 disgust validation faces. Test historically inspected, not fresh.'}
    ck['selection_metric']=decision['rule'];ck['further_validation_selection']=decision
    save_checkpoint(out/'best.pt',ck)
    for file in ['settings.json','training_history.json','fer_training_manifest.json']:
        shutil.copy2(run/file,out/file)
    write_json(out/'completion.json',{'selection_frozen':True,'source_run':str(run),'test_accessed':False})
    write_json(out/'selection.json',decision);print(json.dumps(chosen,indent=2))
if __name__=='__main__':main()
