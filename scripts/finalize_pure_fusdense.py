"""Freeze and test exactly one pure candidate after the declared training run."""
import json,subprocess,sys,time
from pathlib import Path
from scripts.colab_common import digest,write_json
from scripts.train_pure_fusdense import assert_pure
import torch
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_fusdense_20261108'
def main():
 deadline=time.monotonic()+145*60
 while not ((RUN/'completion.json').exists() and (ROOT/'reports/pure_native_residual_20261108/completion.json').exists()):
  if time.monotonic()>deadline:raise TimeoutError('Training incomplete; no finished result may be claimed')
  time.sleep(10)
 subprocess.run([sys.executable,'-m','scripts.select_pure_fusdense'],cwd=ROOT,check=True)
 selected=ROOT/'reports/pure_selected_20261108';selection=json.loads((selected/'selection.json').read_text(encoding='utf-8'))
 if selection['chosen']:
  command=[sys.executable,'-u','-m','scripts.colab_evaluate','--data','data/private/fer2013-v1.zip','--run-dir','reports/pure_selected_20261108','--device','cpu','--threads','4','--single-view','--require-disgust-recall-gain','--baseline-report','reports/pure_selected_20261108/baseline_test.json','--baseline-manifest','reports/pure_selected_20261108/baseline_test_manifest.json']
  with (selected/'evaluation.log').open('w',encoding='utf-8') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
  result=json.loads((selected/'deployment/reports/colab_comparison.json').read_text(encoding='utf-8'))
  if result['recommended_for_integration']:
   with (selected/'integration.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-m','scripts.import_colab_model','--bundle',str(selected/'RagaMoodSync_Trained_Model.zip'),'--apply'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 # User explicitly requests a pure default. Restore its verified baseline if no improved pure candidate passes.
 current=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True)
 if any(k.startswith('transfer_') for k in current['state_dict']):
  with (RUN/'pure_baseline_restoration.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-m','scripts.import_colab_model','--bundle','reports/consistency_20261107/RagaMoodSync_Trained_Model.zip','--apply'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
  write_json(RUN/'pure_baseline_restoration.json',{'restored':True,'reason':'User requests no hybrid reliance. No new pure gain is claimed if candidate fails. Restore earlier independently evaluated pure model; not another candidate test.','checkpoint_sha256':'8fc1713541e49bac8fc6ec4f10a5d5ba4ca9ff3860e227caeda3b7b67870a618'})
 with (RUN/'report.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-m','scripts.report_pure_fusdense'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print('Pure experiment completed; conditional pure installation and measured report finished.',flush=True)
if __name__=='__main__':main()
