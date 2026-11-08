"""Measured joint repair evidence; never changes the app, model or PowerPoint."""
import json
from pathlib import Path
import torch
import numpy as np
from scripts.colab_common import digest,write_json,model_from
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_orientation_20261108';A=ROOT/'reports/pure_repair_20261108'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def compact(r):
 c=r['classification_report'];return {'n':r['n'],'accuracy':r['accuracy'],'macro_precision':c['macro avg']['precision'],'macro_recall':c['macro avg']['recall'],'macro_f1':c['macro avg']['f1-score'],'disgust_precision':c['disgust']['precision'],'disgust_recall':c['disgust']['recall'],'disgust_f1':c['disgust']['f1-score']}
def main():
 torch.set_num_threads(4)
 if not (RUN/'deployment_decision.json').exists():raise RuntimeError('No completed evaluation decision yet')
 done=read(RUN/'completion.json');decision=read(RUN/'deployment_decision.json');history=read(RUN/'all_head_history.json');base=read(RUN/'baseline_test.json');bv=read(RUN/'initial_validation.json');test_path=RUN/'deployment/reports/emotion_evaluation.json';test=read(test_path) if test_path.exists() else None;validation=read(RUN/'fresh_validation_preflight.json') if (RUN/'fresh_validation_preflight.json').exists() else read(RUN/'best_validation.json') if (RUN/'best_validation.json').exists() else None
 if test:assert test['n']==base['n']==6561 and test['predictions']['truth']==base['predictions']['truth']
 deployed=bool(test and decision['eligible'] and digest(ROOT/'models/fusdense.pt')==test['checkpoint_sha256']);protected={name:digest(ROOT/name)==sha for name,sha in read(RUN/'protected_artifacts.json').items()};trained=None;robustness={};changes={};params=None;architecture=None
 if (RUN/'best.pt').exists():
  ck=torch.load(RUN/'best.pt',map_location='cpu',weights_only=True);old=torch.load(RUN/'pure_baseline.pt',map_location='cpu',weights_only=True);model=model_from(ck,torch.device('cpu'));reference=model_from(old,torch.device('cpu'));model.eval();reference.eval();params=sum(p.numel() for p in model.parameters());architecture=ck['architecture'];assert all(torch.equal(v,ck['state_dict'][k]) for k,v in old['state_dict'].items() if not k.startswith('classifier.'))
  for key,value in ck['state_dict'].items():
   previous=old['state_dict'][key]
   if value.shape!=previous.shape or not torch.equal(value,previous):changes[key]={'before_shape':list(previous.shape),'after_shape':list(value.shape),'bit_identical':False}
  raw=torch.load(ROOT/'reports/pruned_head_20261107/train_features.pt',map_location='cpu',weights_only=True,mmap=True);flip=torch.load(RUN/'train_mirrored_features.pt',map_location='cpu',weights_only=True,mmap=True);idx=raw['labels']==1
  for view,cache in [('original',raw),('mirrored',flip)]:
   f=cache['features'][idx]
   with torch.no_grad():old_out=reference.classifier(f);new_out=model.classifier(f)
   robustness[view]={'n_train_disgust':len(f),'baseline_correct':int((old_out.argmax(1)==1).sum()),'candidate_correct':int((new_out.argmax(1)==1).sum()),'baseline_recall':float((old_out.argmax(1)==1).float().mean()),'candidate_recall':float((new_out.argmax(1)==1).float().mean()),'scope':'TRAIN examples used for fitting; diagnostic only, not held-out evidence.'}
 summary={'method':'pure_orientation_repair','completion':done,'deployment_decision':decision,'deployed':deployed,'parameters':params,'architecture':architecture,'independent_failed_full_network_epochs':read(A/'completion.json')['completed_additional_epochs'],'completed_full_epochs':0,'completed_head_epochs':done['completed_head_epochs'],'selected_head':done['selected_head'],'backbone_bit_identical_to_earlier_pure':True,'changed_tensors':changes,'external_encoder':False,'teacher':False,'inference_ensemble':False,'single_view_only':True,'baseline_test':compact(base),'baseline_validation':compact(bv),'selected_validation':compact(validation) if validation else None,'candidate_test':compact(test) if test else None,'training_orientation_diagnostic':robustness,'all_head_history':history,'normalization_audit':read(RUN/'normalization_audit.json'),'mirror_integrity_counts':read(RUN/'mirror_integrity_audit.json')['overlap_counts'],'protected_artifacts_unchanged':protected,'limitation':'TRAIN augmentation diagnostics are fitting results. Reused validation and historically inspected test are exploratory, not fresh identity-disjoint confirmation.'};write_json(RUN/'repair_summary.json',summary)
 def table(before,after):return '\n'.join(f"| {k.replace('_',' ')} | {before[k]:.6f} | {after[k]:.6f} | {after[k]-before[k]:+.6f} |" for k in before if k!='n')
 trials='\n'.join(f"| {r['phase']} / {r['epoch']} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} | {r['val_disgust_recall']:.4f} | {r['val_disgust_f1']:.4f} |" for r in history)
 text=f'''# Pure FusDenseNet orientation repair: actual results

Independent full-network coverage stage: **{summary['independent_failed_full_network_epochs']} completed epochs**, no qualifying candidate and no test evaluation. The subsequent native-classifier stage completed **{done['completed_head_epochs']} classifier-only epochs** across two independent widths,512 and256. These are different compute budgets and are not equivalent full-network epochs. The accepted candidate's backbone, if any, comes directly from the earlier pure checkpoint, not from the failed coverage run.

Passed full test gate: **{decision['eligible']}**. New checkpoint installed: **{deployed}**. Candidate actually tested: **{test is not None}**. Selected classifier: **{done['selected_head']}**. Inference parameters: **{params}**. No external encoder, teacher logits/distillation, probability mixture, inference ensemble, class threshold or post-hoc class-logit offset is used.

## What the failure analysis established

The earlier sampler visited only62.63% of distinct training images in one measured epoch. Its disgust CE multiplier0.579 weakened the intended minority emphasis. However, fixing coverage and loss alone did not improve the held-out target: training loss fell while validation disgust recall remained around28%. Those fixes were not sufficient for generalization.

The baseline's TRAIN disgust recall was315/330(95.45%) on original views but117/330(35.45%) on their mirrors. The classifier contained1,681,415of1,828,896parameters and depended on orientation-sensitive spatial features. This motivated teaching its own classifier both views before changing its backbone further. These diagnostics motivate the repair; they do not establish an isolated causal attribution. Detailed six-batch gradient and augmentation audits: PURE_REPAIR_ANALYSIS.md and reports/pure_repair_20261108/.

## Training performed

Both raw and horizontally mirrored TRAIN representations were computed by the **same original pure FusDenseNet**, source SHA8fc1713541e49bac8fc6ec4f10a5d5ba4ca9ff3860e227caeda3b7b67870a618. They are not features from DenseNet121, Navrasa or another encoder. There are47,006augmented training views from23,503source images, including660disgust views from330source disgust images; these are not new people. Exact mirror-pixel overlaps with held-out raw validation/test input hashes were both zero. This is not an identity/near-duplicate audit.

The native MLP learned using sqrt-inverse-TRAIN-frequency weighted cross entropy, dropout0.5, AdamW and validation selection. No auxiliary teacher, RDrop or SupCon loss was used. One512-unit head retained its dimensions; the other used TRAIN-only activation ranking to prune to256units. Mean/std were fitted to TRAIN views only and folded algebraically into the existing Linear layer before export. Full-validation argmax equivalence of normalized versus folded inference was checked. Every non-classifier tensor remains bit-identical to the earlier pure model; this is a classifier improvement, not a claim that new fusion-block features were learned.

Selection required improved validation accuracy,macroF1,disgustF1 and at least8percentagepoints disgustrecall gain. Both head trials finished before freezing at most one candidate. A fresh full-network single-view validation preflight confirmed the selected cache-based result before any new test inference. If the earlier full-network stage had produced a test candidate, this entire fallback stage would have been skipped, regardless of its test result.

## Selected validation on all3,970faces

| Metric | Earlier pure | Selected pure | Difference |
|---|---:|---:|---:|
{table(compact(bv),compact(validation)) if validation else 'No qualifying validation checkpoint.'}

## Full same-set test on all6,561faces

All7classes,79disgust faces and every error are included. App/PPT changes require strictly positive accuracy,macroF1 and disgustF1 gains plus at least5percentagepoints disgust recall gain (at least4additional correct disgust cases outof79). No alternate test candidate, test-label relabelling, abstention filter or favorable subset was used. Validation-only positive temperature calibration cannot change the predicted class.

| Metric | Earlier pure test | New pure test | Difference |
|---|---:|---:|---:|
{table(compact(base),compact(test)) if test else 'No new candidate test result.'}

Deployment decision: {json.dumps(decision,ensure_ascii=False)}

## Original/mirrored TRAIN diagnostic

These examples were used in fitting and are **not** held-out accuracy. Measured correct counts/recalls: {json.dumps(robustness,ensure_ascii=False)}.

## Every completed classifier-only epoch

| Recipe / epoch index | Validation accuracy | Macro F1 | Disgust recall | Disgust F1 |
|---|---:|---:|---:|---:|
{trials}

## Boundaries

The repeatedly reused validation and historically inspected test do not provide fresh, identity-disjoint or statistical-significance confirmation. Architecture widths, pruning, normalization and data changes are project choices, not an isolated causal/equal-compute ablation or exact author reproduction. No80%accuracy, reliable webcam behavior, clinical, weather/raga-response or physiological-fusion improvement is claimed. EEG/HRV evaluation remains unchanged and negative for primary fusion gain. App/model/PPT hashes stayed unchanged until a positive deployment gate; the individual protection checks and changed native-classifier tensor names are retained in repair_summary.json. All earlier resources and failures remain preserved.

Primary architecture source: [Pingle and Ragha(2025)](https://doi.org/10.1007/s41870-025-02531-y). FER+ label-quality context is documented separately, not used to relabel this dataset: [Barsoum et al.(2016)](https://doi.org/10.48550/arXiv.1608.01041).
''';(ROOT/'PURE_ORIENTATION_REPAIR_RESULTS.md').write_text(text+('\n\nPer-class recall tradeoffs: '+json.dumps({name:{'baseline':base['classification_report'][name]['recall'],'candidate':test['classification_report'][name]['recall']} for name in ['angry','disgust','fear','happy','sad','surprise','neutral']})+'\nHappy, surprise and neutral recall declined in this run; aggregate/disgust gains do not mean every class improved.\n' if test else ''),encoding='utf-8')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13,3.8));baseline=compact(bv)
 for ax,key,title,bkey in zip(axes,['val_accuracy','val_macro_f1','val_disgust_recall'],['Validation accuracy','Validation macro F1','Validation disgust recall'],['accuracy','macro_f1','disgust_recall']):
  for phase in sorted({r['phase'] for r in history}):
   rows=[r for r in history if r['phase']==phase];ax.plot(range(1,len(rows)+1),[r[key] for r in rows],label=phase.replace('head_refinement_',''))
  ax.axhline(baseline[bkey],color='gray',linestyle='--',label='Earlier pure');ax.set_title(title);ax.set_xlabel('Independent classifier-only epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(RUN/'validation_curves.png',dpi=160);plt.close(fig)
 if validation:
  from sklearn.metrics import ConfusionMatrixDisplay
  fig,axes=plt.subplots(1,2,figsize=(12,5));names=['angry','disgust','fear','happy','sad','surprise','neutral']
  for ax,r,label in [(axes[0],bv,'Earlier pure validation'),(axes[1],validation,'Frozen pure candidate validation')]:ConfusionMatrixDisplay(np.asarray(r['confusion_matrix']),display_labels=names).plot(ax=ax,cmap='Blues',colorbar=False,xticks_rotation=35);ax.set_title(label)
  fig.suptitle('Fixed validation cohort: all3,970faces,50disgust examples');fig.tight_layout();fig.savefig(RUN/'validation_confusion.png',dpi=150);plt.close(fig)
 print(json.dumps({'test_available':test is not None,'passed_gate':decision['eligible'],'deployed':deployed,'new_head_epochs':done['completed_head_epochs'],'prior_failed_full_epochs':summary['independent_failed_full_network_epochs'],'parameters':params,'training_orientation_diagnostic':robustness},indent=2))
if __name__=='__main__':main()
