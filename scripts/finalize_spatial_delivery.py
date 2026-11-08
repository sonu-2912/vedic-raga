"""Finish local reports/app/deck after the frozen experiment report exists.

GitHub publication remains a separate step after rendered-slide review.
"""
import json,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/spatial_generalization_20261108';SELECTED=ROOT/'reports/spatial_repair_selected_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 deadline=time.monotonic()+110*60
 while not (RUN/'experiment_summary.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Final actual experiment summary unavailable; do not label incomplete work finished.')
  time.sleep(10)
 # Report writer publishes summary before the remaining documents; wait for it.
 while not (ROOT/'SPATIAL_GENERALIZATION_RESULTS.md').exists() or not (ROOT/'reports/spatial_generalization_trials.png').exists():time.sleep(5)
 import xml.etree.ElementTree as ET
 suites=ET.parse(RUN/'pytest_67.xml').getroot().findall('.//testsuite');test_count=sum(int(x.get('tests',0)) for x in suites);test_seconds=sum(float(x.get('time',0)) for x in suites);assert test_count==67 and all(int(x.get('failures',0))+int(x.get('errors',0))==0 for x in suites)
 trial=read(ROOT/'reports/latest_emotion_experiment.json');e=read(ROOT/'reports/emotion_evaluation.json');v=read(ROOT/'reports/validation_evaluation.json');m=e['classification_report']['macro avg'];d=e['classification_report']['disgust'];selection=read(SELECTED/'selection.json');sha=e['checkpoint_sha256'];arch=e['architecture']
 intro=f"Completed{trial['completed_full_epochs']}full-network v5epochs plus{trial['completed_head_epochs']}separate frozen-feature residual-head epochs. Selected recipe:{trial['selected_recipe']}; deployed:{trial['deployed']}. Active architecture v{arch['version']}; v6/v7are FusDenseNet+ImageNet DenseNet121 hybrid variants, not a pure FusDenseNet result. Full measured audit and comparisons: SPATIAL_GENERALIZATION_RESULTS.md."
 p=ROOT/'README.md';s=p.read_text(encoding='utf-8');a=s.index('## Current measured results');b=s.index('## Run on Windows',a)
 s=s[:a]+f"""## Current measured results

Active model validation {100*v['accuracy']:.2f}% on3970faces; full test {100*e['accuracy']:.2f}% on all6561retained faces. Macro precision{m['precision']:.4f}, recall{m['recall']:.4f}, F1{m['f1-score']:.4f}. Disgust precision{d['precision']:.4f}, recall{100*d['recall']:.2f}%, F1{d['f1-score']:.4f}. All7classes and allerrors included. Historically inspected test and repeatedly reused validation, not fresh/identity-disjoint confirmation. No80%accuracy or reliablewebcam claim.

{intro}

The baseline audit measured83.01%TRAIN accuracy versus58.04%validation; disgust recall95.45%TRAIN versus28%validation.330train disgust versus6039happy;50validation disgust. BN recalibration and translation screens were rejected. New residual spatial refinement and a distinct frozen ImageNet hybrid test two hypotheses; their epochs/compute differ. Original architecture paths, data and previous results are preserved.

Real fixed-game/30-second participant-held-out EEG/ECG results remain negative for primary fusion F1 gain. No new joint face/physiology/raga/weather or therapeutic result. Sample signals remain explicitly synthetic.

"""+s[b:]
 s=s.replace('--run-dir reports/balanced_selected_20261107 --device cpu --threads 4 --strict-all-metrics --baseline-report reports/deep_supervision_20261107/baseline_test.json --baseline-manifest reports/deep_supervision_20261107/baseline_test_manifest.json','--run-dir reports/spatial_repair_selected_20261108 --device cpu --threads 4 --baseline-report reports/spatial_repair_selected_20261108/baseline_test.json --baseline-manifest reports/spatial_repair_selected_20261108/baseline_test_manifest.json').replace('-m scripts.report_deep_supervision','-m scripts.report_spatial_generalization').replace('output/RagaMoodSync_Research_Balanced_Reviewed.pptx','output/RagaMoodSync_Research_Spatial_Generalization_Reviewed.pptx').replace('60 tests passed','67 tests passed')
 if not selection['chosen']:s+='\nNo candidate qualified in this latest experiment, so the new selected-run evaluation command is inapplicable; no new test score was generated. Use the active checkpoint evaluation artifacts instead.\n'
 p.write_text(s,encoding='utf-8')
 audit=f"""# Latest audit — 8 November 2026

{intro}

Actual active test accuracy{100*e['accuracy']:.2f}%, macroF1{m['f1-score']:.4f}, disgust recall{100*d['recall']:.2f}%. Source SHA{sha}. One candidate maximum selected after both recipes complete; no alternative test search. v6ifactive usesImageNet pretraining, disclosedCPU112-pixel preprocessing and frozen encoders; only new residual head trained. All official provenance/rights notices preserved, noFER-pretrained teacher orchangedlabels.

Full source suite{test_count}passed,15dependencywarnings,{test_seconds:.2f}seconds. LiveAPI/resource/metric verification is generated in reports/further_verification_summary.json by the final step. Earlier audit below is historical.

"""
 p=ROOT/'AUDIT_RESULTS.md';backup=RUN/'prior_audit_results.md'
 if not backup.exists():shutil.copy2(p,backup)
 p.write_text(audit+backup.read_text(encoding='utf-8'),encoding='utf-8')
 shutil.copy2(RUN/'pytest_67.xml',ROOT/'reports/pytest.xml')
 with (RUN/'final_verification.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-m','scripts.verify_further_results'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 with (RUN/'frontend_build.log').open('w',encoding='utf-8') as stream:subprocess.run(['node','node_modules/vite/bin/vite.js','build','--configLoader','native'],cwd=ROOT/'frontend',stdout=stream,stderr=subprocess.STDOUT,check=True)
 node='C:/Users/bindu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
 with (RUN/'presentation_build.log').open('w',encoding='utf-8') as stream:subprocess.run([node,'reports/slides_nov7/edit_spatial_generalization.mjs'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 (RUN/'local_delivery_complete.json').write_text(json.dumps({'complete':True,'github_published':False,'slide_visual_review_pending':True,'active_sha256':sha,'active_accuracy':e['accuracy'],'model_version':arch['version']},indent=2),encoding='utf-8')
 print('Actual local app/reports/deck built. Rendered-slide review and filtered GitHub publication remain.',flush=True)
if __name__=='__main__':main()
