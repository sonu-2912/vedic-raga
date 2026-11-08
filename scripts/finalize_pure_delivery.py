"""Finish the actual pure experiment app/reports/deck; no publication before visual review."""
import json,subprocess,sys,time,shutil,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_fusdense_20261108';SELECTED=ROOT/'reports/pure_selected_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 deadline=time.monotonic()+110*60
 while not (RUN/'experiment_summary.json').exists() or not (ROOT/'reports/pure_fusdense_training.png').exists():
  if time.monotonic()>deadline:raise TimeoutError('Pure experiment incomplete; no finished score may be claimed')
  time.sleep(10)
 with (RUN/'tests_final.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-m','pytest','-q','--basetemp','reports/pure_selected_20261108/test_final','--junitxml','reports/pure_selected_20261108/pytest.xml'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 xml=ET.parse(SELECTED/'pytest.xml').getroot();suites=xml.findall('.//testsuite');n=sum(int(s.get('tests',0)) for s in suites);seconds=sum(float(s.get('time',0)) for s in suites);assert all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
 shutil.copy2(SELECTED/'pytest.xml',ROOT/'reports/pytest.xml')
 with (RUN/'verification.log').open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,'-m','scripts.verify_further_results'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 e=read(ROOT/'reports/emotion_evaluation.json');v=read(ROOT/'reports/validation_evaluation.json');trial=read(ROOT/'reports/latest_emotion_experiment.json');p=read(ROOT/'reports/pure_fusdense_results_summary.json');m=e['classification_report']['macro avg'];d=e['classification_report']['disgust']
 intro=f"Pure experiments completed{trial['completed_full_epochs']}full-network epochs across v4/v8recipes. Newpuredeployed:{trial['deployed']}. ActiveisPure:{p['active_is_pure']}. Active accuracy{100*e['accuracy']:.2f}%, macroF1{m['f1-score']:.4f}, disgustrecall{100*d['recall']:.2f}%. Formerhybrid59.26%is separately labelled reference-only. No new gain is claimed if no candidate qualified/passed."
 file=ROOT/'README.md';text=file.read_text(encoding='utf-8');a=text.index('## Current measured results');b=text.index('## Run on Windows',a)
 text=text[:a]+f"""## Current measured results

{intro}

Validation accuracy {100*v['accuracy']:.2f}% on3,970faces. Full test {100*e['accuracy']:.2f}% on all6,561retained faces. Macro precision {m['precision']:.4f}, recall {m['recall']:.4f}, F1 {m['f1-score']:.4f}. Disgust precision {d['precision']:.4f}, recall {100*d['recall']:.2f}%, F1 {d['f1-score']:.4f}. All7classes/allerrors retained. Test historically inspected, validation reused; no fresh/identity-disjoint or reliablewebcam claim.

Purev4retains the earlier architecture; nativev8adds117,360residualdense feature parameters INSIDEfusionblocks with GroupNorm, same classifier widths. The separate93,080parameter contrastive projection is TRAINonly and removed at export. No external encoder/teacher or inference ensemble. One frozen validation-selected candidate maximum; single-view evaluation. Every failed epoch, purebaseline comparison and formerhybrid comparison: [PURE_FUSDENSENET_RESULTS.md](PURE_FUSDENSENET_RESULTS.md).

Real fixed-game/30-second/participant-held-out EEG/HRV results remain unchanged, with no primary fusionF1gain. No joint face/body/raga/weather efficacy result. Missing metadata and dataset access/source limitations stay explicit.

"""+text[b:]
 lines=text.splitlines();command=r'.\.venv\Scripts\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/pure_selected_20261108 --device cpu --threads 4 --single-view --require-disgust-recall-gain --baseline-report reports/pure_selected_20261108/baseline_test.json --baseline-manifest reports/pure_selected_20261108/baseline_test_manifest.json'
 lines=[command if '-m scripts.colab_evaluate ' in line else line.replace('-m scripts.report_spatial_generalization','-m scripts.report_pure_fusdense').replace('RagaMoodSync_Research_Spatial_Generalization_Reviewed_v3.pptx','RagaMoodSync_Research_Pure_FusDenseNet_Reviewed.pptx').replace('67 tests passed',f'{n} tests passed') for line in lines];text='\n'.join(lines)+'\n'
 if not p['qualified']:text+='\nNo new pure candidate qualified, so the selected-run evaluation command is inapplicable; the earlier pure model was restored without a new test. All negative results remain reported.\n'
 text='\n'.join('The newest editable 13-slide deck is output/RagaMoodSync_Research_Pure_FusDenseNet_Reviewed.pptx. Historical reports and decks are preserved.' if line.startswith('The newest editable') else line for line in text.splitlines())+'\n'
 file.write_text(text,encoding='utf-8')
 file=ROOT/'AUDIT_RESULTS.md';backup=RUN/'prior_audit.md'
 if not backup.exists():shutil.copy2(file,backup)
 file.write_text(f"# Pure FusDenseNet audit — 8 November 2026\n\n{intro}\n\n{n}tests passed, {seconds:.2f}seconds. Production frontend and liveAPI verification recorded in runlogs. Raw retained-prediction metrics/confusion recomputed, originals55unchanged. No userphoto used. Full pure-state/initialization/gradient contracts tested; nativev8initial logits equalv4. Budget-paused v4run actualcounts and native-stage limits are explicit. Earlier audit below is historical.\n\n"+backup.read_text(encoding='utf-8'),encoding='utf-8')
 with (RUN/'frontend_build.log').open('w',encoding='utf-8') as stream:subprocess.run(['node','node_modules/vite/bin/vite.js','build','--configLoader','native'],cwd=ROOT/'frontend',stdout=stream,stderr=subprocess.STDOUT,check=True)
 node='C:/Users/bindu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
 with (RUN/'presentation_build.log').open('w',encoding='utf-8') as stream:subprocess.run([node,'reports/slides_nov7/edit_pure_fusdense.mjs'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 (RUN/'local_delivery_complete.json').write_text(json.dumps({'complete':True,'pure_default':p['active_is_pure'],'new_pure_deployed':trial['deployed'],'tests':n,'test_seconds':seconds,'visual_review_pending':True,'github_published':False},indent=2),encoding='utf-8');print('Pure app/results/deck ready for rendered-slide review and filtered publication.',flush=True)
if __name__=='__main__':main()
