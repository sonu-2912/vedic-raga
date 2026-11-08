"""Freeze one validation-qualified candidate; never select on test performance."""
import json,shutil
from pathlib import Path
import torch
from scripts.colab_common import digest,write_json
from scripts.train_deep_supervision import metric_vector,qualifies
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/deep_supervision_20261107'
OUT=ROOT/'reports/balanced_selected_20261107'
def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'bundle_manifest.json').exists() or (OUT/'selection.json').exists():raise RuntimeError('Selection is already frozen; reuse its report/artifact.')
    protocol=json.loads((BASE/'candidate_selection_protocol.json').read_text());baseline=json.loads((BASE/'initial_validation.json').read_text());candidates=[]
    for relative in protocol['candidate_runs']:
        run=ROOT/relative;done=json.loads((run/'completion.json').read_text());settings=json.loads((run/'settings.json').read_text())
        assert settings['initial_checkpoint_sha256']==json.loads((BASE/'protocol.json').read_text())['initial_checkpoint_sha256']
        if not done['validation_improved']:continue
        report=json.loads((run/'best_validation.json').read_text());assert report['predictions']['truth']==baseline['predictions']['truth']
        if qualifies(report,baseline):candidates.append({'run':relative,'best_sha256':digest(run/'best.pt'),'metrics':metric_vector(report),'score':.35*report['accuracy']+.65*report['classification_report']['macro avg']['f1-score']})
    chosen=max(candidates,key=lambda c:c['score']) if candidates else None
    selection={'selection_protocol':protocol,'qualified_candidates':candidates,'chosen':chosen,'test_accessed':False,'synthetic':False}
    if chosen:
        run=ROOT/chosen['run']
        for name in ['best.pt','best_validation.json','fer_training_manifest.json','training_history.json','settings.json']:shutil.copy2(run/name,OUT/name)
        for name in ['baseline_test.json','baseline_test_manifest.json','baseline.pt']:shutil.copy2(BASE/name,OUT/name)
        shutil.copy2(BASE/'evaluation_protocol.json',OUT/'evaluation_protocol.json')
    write_json(OUT/'selection.json',selection)
    for relative in protocol['candidate_runs']:write_json(ROOT/relative/'candidate_frozen.json',{'selected_run':chosen['run'] if chosen else None,'source_best_sha256':digest(ROOT/relative/'best.pt'),'reason':'Validation selection ended before any new test. No source-run continuation or alternate candidate testing after test access.'})
    print(json.dumps(selection,indent=2))
if __name__=='__main__':main()
