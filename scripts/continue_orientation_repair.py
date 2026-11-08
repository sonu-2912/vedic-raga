"""Continue only a not-yet-tested pure-model repair; never search alternate test candidates."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'reports/pure_repair_20261108';B=ROOT/'reports/pure_orientation_20261108'
def main():
 deadline=time.monotonic()+145*60
 while not (A/'deployment_decision.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Prior training/test decision incomplete; conditional stage not authorized by its protocol')
  time.sleep(10)
 decision=json.loads((A/'deployment_decision.json').read_text(encoding='utf-8'))
 if decision['test_evaluated']:
  (B/'skipped.json').write_text(json.dumps({'reason':'A new candidate was already tested; no alternate test search.','test_evaluated':False}),encoding='utf-8');print('Orientation fallback skipped: prior candidate already tested.',flush=True);return
 subprocess.run([sys.executable,'-m','scripts.report_coverage_repair'],cwd=ROOT,check=True)
 with (B/'awake_request.log').open('w',encoding='utf-8') as stream:subprocess.Popen([sys.executable,'-m','scripts.keep_training_awake','--run-dir','reports/pure_orientation_20261108','--minutes','65'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 with (B/'training.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-u','-m','scripts.train_orientation_head_repair','--run-dir','reports/pure_orientation_20261108','--threads','4'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 with (B/'evaluation.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-u','-m','scripts.evaluate_orientation_repair'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print('Conditional orientation repair and one-candidate decision finished.',flush=True)
if __name__=='__main__':main()
