"""Measured pure-training repair findings; does not modify app/PPT/model artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.colab_common import digest,write_json
from scripts.train_pure_fusdense import selection_score
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_repair_20261108'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def compact(r):
 c=r['classification_report'];return {'n':r['n'],'accuracy':r['accuracy'],'macro_precision':c['macro avg']['precision'],'macro_recall':c['macro avg']['recall'],'macro_f1':c['macro avg']['f1-score'],'disgust_precision':c['disgust']['precision'],'disgust_recall':c['disgust']['recall'],'disgust_f1':c['disgust']['f1-score']}
def main():
 a=argparse.ArgumentParser();a.add_argument('--partial',action='store_true');args=a.parse_args()
 if not args.partial and not (RUN/'deployment_decision.json').exists():raise RuntimeError('A completed training/evaluation decision is required')
 audit=read(RUN/'failure_audit.json');sample=read(RUN/'sampling_audit.json');perturb=read(RUN/'augmentation_component_audit.json');baseline=read(RUN/'baseline_test.json');baseval=read(RUN/'initial_validation.json');protocol=read(RUN/'protocol.json');history=read(RUN/'training_history.json') if (RUN/'training_history.json').exists() else [];rows=[r for r in history if r.get('phase')=='full_network_pure_coverage_repair'];decision=read(RUN/'deployment_decision.json') if (RUN/'deployment_decision.json').exists() else None;done=read(RUN/'completion.json') if (RUN/'completion.json').exists() else None
 options=[(p,read(p)) for p in RUN.glob('validation_[0-9][0-9]_*.json')];observed=max(options,key=lambda pair:selection_score(pair[1])) if options else None
 testfile=RUN/'deployment/reports/emotion_evaluation.json';test=read(testfile) if testfile.exists() else None
 if test:assert test['predictions']['truth']==baseline['predictions']['truth'] and test['n']==6561
 protected={name:digest(ROOT/name)==sha for name,sha in read(RUN/'protected_artifacts.json').items()};deployed=bool(test and decision and decision['eligible'] and digest(ROOT/'models/fusdense.pt')==test['checkpoint_sha256'])
 gradient=audit['gradient_batches'];opp={name:sum(row['cosines'][f'{name}_vs_disgust_CE']<0 for row in gradient) for name in ['RDrop','SupCon']};summary={'method':'pure_coverage_repair','completed_new_epochs':len(rows),'completed':done is not None,'decision':decision,'deployed':deployed,'external_encoder':False,'teacher':False,'baseline_test':compact(baseline),'baseline_validation':compact(baseval),'best_observed_validation':compact(observed[1]) if observed else None,'best_observed_file':observed[0].name if observed else None,'candidate_test':compact(test) if test else None,'class_unit_CE_weights':[1.]*7,'old_loss_disgust_CE_mass':audit['old_normalized_CE_class_mass'][1],'old_sampler_unique_fraction':sample['old_unique_fraction'],'regularizer_gradient_conflict_counts':opp,'audited_gradient_batches':len(gradient),'protected_artifacts_unchanged':protected,'history':rows,'protocol':protocol,'limitation':'Perturbation/gradient tests are training-only diagnostics, not causal proof. Reused validation and historically inspected test are descriptive, not fresh identity-disjoint confirmation.'};write_json(RUN/'repair_summary.json',summary)
 def metric_rows(before,after):return '\n'.join(f"| {key.replace('_',' ')} | {before[key]:.6f} | {after[key]:.6f} | {after[key]-before[key]:+.6f} |" for key in before if key!='n')
 b=compact(baseval);o=compact(observed[1]) if observed else None;pd=perturb['train_only_disgust'];draws='\n'.join(f"| {i+1} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} | {r['val_disgust_recall']:.4f} | {r['val_disgust_f1']:.4f} | {r['sampling_audit']['unique_training_faces']} | {r['sample_counts'][1]} |" for i,r in enumerate(rows));text=f'''# Pure FusDenseNet training repair: measured findings

Status: {'finished' if done else 'training in progress'}. Completed new full-network epochs: **{len(rows)}**. Candidate tested: **{test is not None}**. Passed deployment gate: **{bool(decision and decision['eligible'])}**. Improved checkpoint installed: **{deployed}**. No hybrid, external encoder, teacher, inference ensemble, class threshold or logit offset is used.

## Why the earlier changes were not enough

1. The old sampling design repeatedly resampled natural faces within an epoch. In a measured epoch it drew {sample['old_draws']:,} samples but visited only **{sample['old_unique_faces']:,}/{sample['pool_faces']:,} distinct faces ({100*sample['old_unique_fraction']:.2f}%)**. This is a coverage limitation, not a PyTorch error. Repeating the same 330 disgust faces did not create new identities.
2. Its normalized disgust CE multiplier was **{audit['old_CE_importance'][1]:.4f}**. Consequently the expected CE class mass was **{100*audit['old_normalized_CE_class_mass'][1]:.2f}%**, despite a **{100*audit['old_sample_prior'][1]:.2f}%** sampling prior. That compensating weight weakened the intended minority emphasis.
3. The verified baseline was brittle to the chosen augmentation. On the same 330 TRAIN disgust faces, raw recall was **{100*pd['raw']['recall']:.2f}% ({pd['raw']['correct']}/330)**, mirrored recall **{100*pd['flip']['recall']:.2f}% ({pd['flip']['correct']}/330)**, affine recall **{100*pd['old_affine']['recall']:.2f}%**, and old photometric-only recall **{100*pd['old_color']['recall']:.2f}%**. These perturbations demonstrate orientation/spatial fragility, not held-out accuracy or proof that augmentation itself caused every failure.
4. On six training-only batches from the failed final v4 state, R-Drop's backbone gradient opposed the disgust CE direction in **{opp['RDrop']}/6** cases and SupCon in **{opp['SupCon']}/6**. Their gradient norms were smaller than CE; they are not established as the sole or dominant cause. The results show that adding auxiliary objectives was not uniformly helping the target.
5. The earlier baseline audit found 83.01% raw training versus58.04% validation accuracy, and95.45% versus28% disgust recall. Only330 unique training disgust faces and50 validation disgust faces are available. More native parameters did not remove this generalization problem. There is no evidence of a disconnected attention graph; earlier gradient contracts passed.

FER+ independently studies noisy facial-expression labels using multiple annotators: [Barsoum et al. (2016)](https://doi.org/10.48550/arXiv.1608.01041). Its [official annotation repository](https://github.com/microsoft/FERPlus) requires correspondence to the original FER2013 CSV row order. That mapping has not been verified for this workspace's JPEG archive, so no FER+ annotations are applied and no test labels are changed. This is context, not evidence that a particular project label is wrong.

## Repair implemented

The same1,828,896-parameter pure-v4 FusDenseNet path is fine-tuned end to end. Every training face appears in a shuffled natural pass each epoch. Each batch adds4disgust,2angry and2sad TRAIN-only exposures from cycling shuffled pools; angry/sad were among the original TRAIN disgust confusions. Every64-index batch is unique internally. Padding and actual unique/class counts are logged separately. These are repeated training exposures, not new images or people.

The objective is unit-weight supervised cross entropy, without compensating importance factors, label smoothing, SupCon or R-Drop. The classifier retains its original0.35dropout and old BatchNorm references; weights/affine parameters still learn. Half the examples are unmodified; half receive mild brightness/contrast jitter. No horizontal flip or affine augmentation is used in this repair. This preserves the available feature signal during fine-tuning; it does **not** establish orientation robustness. Backbone LR2e-5 and classifier LR3e-5 are lower than the failed recipe. Student and averaged-weight single-network checkpoints are selected on validation, never mixed at inference.

This combined repair is not an isolated equal-budget ablation. It addresses measured training weaknesses but cannot establish which individual change caused any gain. All source hashes, audit results, protocols and completed epoch states are preserved in reports/pure_repair_20261108/.

## Prospective selection and deployment rules

At most12epochs, minimum6before patience4,105-minute wall budget at completed-epoch boundaries. A conditional budget allocation declared after4completed validation epochs caps the run after at least6complete epochs ONLY if no checkpoint qualified; then the separate orientation-classifier repair can use remaining CPU time. The original protocol, adaptive allocation and discarded partial-next-epoch status are preserved. Validation accuracy, macroF1 and disgustF1 must exceed the earlier pure baseline, and disgust recall must gain at least8percentagepoints (28%to36%minimum). After all training, at most ONE qualifying checkpoint can be frozen for the full same-set test.

App/PPT changes require strictly better test accuracy,macroF1 and disgustF1, plus **at least5percentagepoints disgust recall gain** over36.71%. With79test disgust faces, this requires at least4additional correct disgust classifications. The same6,561retained test faces, all7classes and all errors are included. Single-view inference and positive validation-only temperature cannot change the predicted class. No alternate test candidate is allowed after the result. This is an engineering gate, not significance evidence. Historical test reuse and repeated validation prevent a fresh confirmatory claim.

## Every completed repair epoch

TRAIN agreement uses the augmented, oversampled distribution with dropout; validation uses raw natural-frequency images without dropout. They are not directly comparable generalization-gap estimates.

| New epoch | Validation accuracy | Macro F1 | Disgust recall | Disgust F1 | Unique train faces | Disgust draws |
|---|---:|---:|---:|---:|---:|---:|
{draws}

## Best observed validation diagnostic

{'No completed validation checkpoint yet.' if o is None else 'Best by the prespecified validation objective; it is not automatically eligible or deployed.'}

| Metric | Earlier pure validation | New diagnostic | Difference |
|---|---:|---:|---:|
{metric_rows(b,o) if o else ''}

## Full test result

{'No new candidate test result. The earlier pure test scores remain historical references; app/PPT must remain unchanged.' if test is None else 'One frozen pure candidate on all6,561same retained test faces. Passing the additional five-point recall requirement is required before deployment.'}

| Metric | Earlier pure test | New pure test | Difference |
|---|---:|---:|---:|
{metric_rows(compact(baseline),compact(test)) if test else ''}

Current deployment decision: {json.dumps(decision,ensure_ascii=False)}

The app/PPT protection checks before installation are recorded individually in repair_summary.json. Real EEG/HRV results and their negative fusion finding are unchanged. No new physiological, webcam-domain, weather/raga-response, therapeutic,80%accuracy or exact-author-reproduction claim is made. Primary architecture authority: https://doi.org/10.1007/s41870-025-02531-y .
''';(ROOT/'PURE_REPAIR_ANALYSIS.md').write_text(text,encoding='utf-8')
 if rows:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,3,figsize=(13,3.7));xx=list(range(1,len(rows)+1))
  for ax,key,title,reference in zip(axes,['val_accuracy','val_macro_f1','val_disgust_recall'],['Validation accuracy','Validation macro F1','Validation disgust recall'],[b['accuracy'],b['macro_f1'],b['disgust_recall']]):
   ax.plot(xx,[r[key] for r in rows],marker='o',label='Logged validation selection');ax.axhline(reference,color='gray',linestyle='--',label='Earlier pure baseline');ax.set_title(title);ax.set_xlabel('New full-network epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
  fig.tight_layout();fig.savefig(RUN/'validation_curves.png',dpi=160);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(10,3.5));axes[0].plot(xx,[r['CE'] for r in rows],label='Unit CE');axes[0].set_title('Actual training objective');axes[0].legend();axes[1].plot(xx,[r['sample_counts'][1] for r in rows],label='Repeated disgust exposures');axes[1].set_title('All23,503natural faces covered each epoch');axes[1].legend(fontsize=8)
  for ax in axes:ax.set_xlabel('New full-network epoch');ax.grid(alpha=.2)
  fig.tight_layout();fig.savefig(RUN/'training_curves.png',dpi=160);plt.close(fig)
 print(json.dumps({'epochs':len(rows),'completed':done is not None,'test_available':test is not None,'deployed':deployed,'app_ppt_unchanged':all(protected.values())},indent=2))
if __name__=='__main__':main()
