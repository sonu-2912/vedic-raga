"""Freeze at most one validation-qualified pure repair, then measure its full test."""
import json,subprocess,sys,time
from pathlib import Path
from scripts.colab_common import digest,write_json
from scripts.train_pure_coverage_repair import eligible
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_repair_20261108'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def protected():return all(digest(ROOT/name)==sha for name,sha in read(RUN/'protected_artifacts.json').items())
def main():
 deadline=time.monotonic()+145*60
 while not (RUN/'completion.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Training incomplete; app and deck must remain unchanged')
  time.sleep(10)
 done=read(RUN/'completion.json');protocol=read(RUN/'protocol.json');assert protected(),'Protected app/deck changed before measured improvement'
 if not done['validation_improved']:
  write_json(RUN/'deployment_decision.json',{'eligible':False,'test_evaluated':False,'installed':False,'reason':'No pure checkpoint passed the predeclared validation accuracy/F1/recall gate. App and PPT unchanged.','protected_artifacts_unchanged':True,'new_epochs':done['completed_additional_epochs']});print('No qualifying pure repair; no test candidate or app/PPT changes.',flush=True);return
 assert eligible(read(RUN/'best_validation.json'),read(RUN/'initial_validation.json'),protocol['validation_min_disgust_recall_gain'])
 if not (RUN/'candidate_frozen.json').exists():write_json(RUN/'candidate_frozen.json',{'best_sha256':digest(RUN/'best.pt'),'test_accessed_at_freeze':False,'single_view_only':True,'external_encoder':False,'teacher':False,'minimum_test_disgust_recall_gain':protocol['test_min_disgust_recall_gain']})
 assert read(RUN/'candidate_frozen.json')['best_sha256']==digest(RUN/'best.pt')
 cmd=[sys.executable,'-u','-m','scripts.colab_evaluate','--data','data/private/fer2013-v1.zip','--run-dir','reports/pure_repair_20261108','--device','cpu','--threads','4','--single-view','--require-disgust-recall-gain','--baseline-report','reports/pure_repair_20261108/baseline_test.json','--baseline-manifest','reports/pure_repair_20261108/baseline_test_manifest.json']
 with (RUN/'evaluation.log').open('w',encoding='utf-8') as stream:subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 comparison=read(RUN/'deployment/reports/colab_comparison.json');gain=comparison['metrics']['disgust_recall']['difference'];passed=bool(comparison['recommended_for_integration'] and gain+1e-12>=protocol['test_min_disgust_recall_gain'])
 write_json(RUN/'deployment_decision.json',{'eligible':passed,'test_evaluated':True,'installed':False,'reason':'Passed full same-set test accuracy/macroF1/disgustF1 and at least5percentagepoint disgustrecall gain' if passed else 'Failed predeclared full test gate; no alternate candidate or app/PPT update allowed.','minimum_disgust_recall_gain':protocol['test_min_disgust_recall_gain'],'metrics':comparison['metrics'],'checkpoint_sha256':comparison['checkpoint_sha256'],'protected_artifacts_unchanged':protected(),'new_epochs':done['completed_additional_epochs'],'limitation':'Historically inspected test and reused validation, not fresh identity-disjoint confirmation or significance evidence.'})
 print(json.dumps(read(RUN/'deployment_decision.json'),indent=2),flush=True)
if __name__=='__main__':main()
