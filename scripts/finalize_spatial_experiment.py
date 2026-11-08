"""Finalize this one declared CPU run; no alternative test search."""
import json,subprocess,sys,time
from pathlib import Path
from scripts.colab_common import digest,write_json
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/spatial_generalization_20261108'
def main():
 deadline=time.monotonic()+110*60
 while not ((RUN/'completion.json').exists() and (ROOT/'reports/dense_transfer112_20261108/completion.json').exists() and (ROOT/'reports/direct_dense_head_long_20261108/completion.json').exists()):
  if time.monotonic()>deadline:raise TimeoutError('Training not complete; do not publish partial results as finished.')
  time.sleep(10)
 subprocess.run([sys.executable,'-m','scripts.select_spatial_repair'],cwd=ROOT,check=True)
 selected=ROOT/'reports/spatial_repair_selected_20261108'
 selection=json.loads((selected/'selection.json').read_text(encoding='utf-8'))
 if selection['chosen']:
  command=[sys.executable,'-u','-m','scripts.colab_evaluate','--data','data/private/fer2013-v1.zip','--run-dir','reports/spatial_repair_selected_20261108','--device','cpu','--threads','4','--baseline-report','reports/spatial_repair_selected_20261108/baseline_test.json','--baseline-manifest','reports/spatial_repair_selected_20261108/baseline_test_manifest.json']
  with (selected/'evaluation.log').open('w',encoding='utf-8') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
  comparison=json.loads((selected/'deployment/reports/colab_comparison.json').read_text(encoding='utf-8'))
  if comparison['recommended_for_integration']:
   with (selected/'integration.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-m','scripts.import_colab_model','--bundle',str(selected/'RagaMoodSync_Trained_Model.zip'),'--apply'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 with (RUN/'report.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-m','scripts.report_spatial_generalization'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print('Completed training, at most one qualifying-candidate evaluation, conditional verified import and actual-results report.',flush=True)
if __name__=='__main__':main()
