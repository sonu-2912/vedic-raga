"""Publish the completed audit-driven experiment and all actual failures."""
import json,shutil
from pathlib import Path
import torch
from scripts.colab_common import digest,model_from,write_json
from scripts.train_deep_supervision import metric_vector
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/spatial_generalization_20261108';TRANSFER=ROOT/'reports/dense_transfer112_20261108';DIRECT=ROOT/'reports/direct_dense_head_long_20261108';SHORT=ROOT/'reports/direct_dense_head_20261108';SELECTED=ROOT/'reports/spatial_repair_selected_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 torch.set_num_threads(2);done=read(RUN/'completion.json');audit=read(ROOT/'reports/bottleneck_audit_20261108/audit.json');base=read(RUN/'initial_validation.json');history=read(RUN/'training_history.json');rows=[r for r in history if r.get('phase')=='full_network_spatial_generalization'];reports=[(p,read(p)) for p in RUN.glob('validation_[0-9][0-9]_*.json')]
 from scripts.train_spatial_generalization import score
 transfer_done=read(TRANSFER/'completion.json');residual_rows=read(TRANSFER/'training_history.json');direct_done=read(DIRECT/'completion.json');short_done=read(SHORT/'completion.json');short_rows=read(SHORT/'training_history.json');direct_rows=read(DIRECT/'training_history.json');transfer_rows=residual_rows+short_rows+direct_rows;selection=read(SELECTED/'selection.json');chosen=selection['chosen'];reports += [(p,read(p)) for folder in [TRANSFER,DIRECT,SHORT] for p in folder.glob('validation_[0-9][0-9]_*.json')];observed_path,observed=max(reports,key=lambda pair:score(pair[1]));qualified=bool(chosen);selected=read(SELECTED/'best_validation.json') if qualified else observed
 a,b=metric_vector(base),metric_vector(selected);vc={'n':base['n'],'metrics':{k:{'baseline':a[k],'colab':b[k],'difference':b[k]-a[k]} for k in a}}
 test_path=SELECTED/'deployment/reports/colab_comparison.json';test=read(test_path) if test_path.exists() else None
 active=read(ROOT/'reports/emotion_evaluation.json');val=read(ROOT/'reports/validation_evaluation.json');sha=digest(ROOT/'models/fusdense.pt');assert active['checkpoint_sha256']==sha==val['checkpoint_sha256'];ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True);model=model_from(ck,torch.device('cpu'));deployed=bool(test and test['recommended_for_integration'] and test['checkpoint_sha256']==sha)
 trial={'method':'spatial_generalization','history':[{**r,'recipe':name} for name,rr in [('Spatial full',rows),('Residual head',residual_rows),('Direct short',short_rows),('Direct long',direct_rows)] for r in rr],'completed_epochs':len(rows)+len(transfer_rows),'completed_full_epochs':len(rows),'completed_head_epochs':len(transfer_rows),'selected_recipe':chosen['run'] if chosen else None,'curve_run':chosen['run'] if chosen else 'reports/spatial_generalization_20261108','curve_phase':'head_refinement_dense_probability' if chosen and 'direct_dense_head' in chosen['run'] else 'head_refinement_dense_transfer' if chosen and 'dense_transfer112' in chosen['run'] else 'full_network_spatial_generalization','candidate_checkpoint_sha256':test['checkpoint_sha256'] if test else None,'test_comparison':test,'validation_comparison':vc,'validation_comparison_kind':'Frozen selected candidate' if qualified else 'Best observed exploratory validation diagnostic; no qualifying candidate','validation_candidate_qualified':qualified,'deployed':deployed,'limitation':'Repeatedly reused validation and historically inspected test, not fresh confirmatory evidence. Full7classes and all6561retained errors remain included. v5 spatial refinement and v6 ImageNet hybrid are distinct project extensions. v5 training uses a different sampling prior and augmentation; v6 trains only the new head on frozen features. Their epochs/curves measure different conditions. Prior correction comes only from training counts. No webcam or face/physiology/raga efficacy claim.'}
 write_json(ROOT/'reports/latest_emotion_experiment.json',trial)
 if deployed:write_json(ROOT/'reports/emotion_training_comparison.json',test)
 summary={'protocol':read(RUN/'protocol.json'),'initialization_audit':read(RUN/'initialization_audit.json'),'completion':done,'transfer_completion':transfer_done,'direct_completion':direct_done,'short_direct_completion':short_done,'selection':selection,'pretrained_provenance':read(TRANSFER/'pretrained_provenance.json'),'bottleneck_audit':audit,'latest_experiment':trial,'active_source_sha256':sha,'best_observed_report':observed_path.name,'candidate_frozen_sha256':digest(SELECTED/'best.pt') if qualified else None,'active_architecture':ck['architecture'],'active_parameters':sum(p.numel() for p in model.parameters())}
 write_json(RUN/'experiment_summary.json',summary)
 m=active['classification_report']['macro avg'];d=active['classification_report']['disgust'];active_history=read(ROOT/'reports/training_history.json')
 write_json(ROOT/'reports/further_results_summary.json',{'accuracy':active['accuracy'],'macro_precision':m['precision'],'macro_recall':m['recall'],'macro_f1':m['f1-score'],'disgust':d,'test_n':active['n'],'checkpoint_sha256':sha,'parameters':sum(p.numel() for p in model.parameters()),'architecture':ck['architecture'],'selected_epoch':active['checkpoint_epoch'],'train_count':ck['train_count'],'val_count':ck['val_count'],'classifier_inputs':model.classifier[0].in_features,'selected_lineage_full_epochs':sum(not r.get('phase') or str(r.get('phase')).startswith('full_network_') for r in active_history),'selected_lineage_head_epochs':sum(str(r.get('phase','')).startswith('head_refinement') for r in active_history)})
 table=lambda comp:'\n'.join(f"| {k.replace('_',' ')} | {v['baseline']:.6f} | {v['colab']:.6f} | {v['difference']:+.6f} |" for k,v in comp['metrics'].items())
 log='\n'.join(f"| {r['epoch']} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} | {r['val_disgust_recall']:.4f} | {r['val_disgust_f1']:.4f} | {r['sample_counts'][1]} |" for r in rows)
 text=f"""# Audit-driven spatial generalization: actual results

Completed {len(rows)} full-network v5epochs and {len(transfer_rows)} separate hybrid residual-head epochs; validation-qualified candidate: {qualified}; new candidate deployed: {deployed}. Active architecture version **{ck['architecture']['version']}**; v6/v7 denote hybrid variants with a frozen ImageNet DenseNet121 branch; v7 averages probabilities from an independently trained head. Neither is a pure FusDenseNet result. Active full test accuracy **{100*active['accuracy']:.2f}%**, macro F1 **{m['f1-score']:.4f}**, disgust recall **{100*d['recall']:.2f}%**. Active validation accuracy **{100*val['accuracy']:.2f}%**. No80% claim or reliable webcam recognition result.

## What held the previous model back

Measured raw TRAIN accuracy83.0107% versus validation58.0353%; disgust TRAIN recall95.4545% versus validation28.00%. These are different data splits, not a training improvement. TRAIN disgust330faces, happy6039; validation disgust50, so one example changes its recall by2percentage points. The classifier holds1,681,415of1,828,896parameters (91.94%). This combination is consistent with overfitting on limited minority-class diversity. Exact duplicate hashes were already excluded. No label-order bug was found: folder labels map explicitly to angry/disgust/fear/happy/sad/surprise/neutral. Attention gradients are nonzero at every stage; fused and spatial feature-zeroing diagnostics both reduce validation accuracy. Those diagnostics do not prove causal feature importance or isolate architecture.

Recomputing BN statistics with2,048training-only images reduced validation accuracy to57.5819% and disgust recall24%; rejected. Max mean activation shift was0.0738standard deviations, providing no evidence that stale BN references are the primary bottleneck. These are reused-validation diagnostics, not held-out gains.

## First experiment: v5 spatial refinement

v5 retains all three fusion-dense stages, cross-dense concatenation, cross-attention and multi-pooling. It adds two60-channel3x3convolutions as a residual refinement of first dense24x24features before spatial pooling (65,040additional parameters). Last convolution is zero-initialized, and real validation initialization logits matched v4 exactly (max absolute difference0). This is a disclosed project extension; exact author reproduction is not claimed.

Full-network optimization; old BN statistics frozen and new branch BN learned; classifier dropout.5; backbone/head LR5e-5, new branch3e-4; AdamW decay.003/.02/.003; mild affine, photometric augmentation and15%small-patch erasing. No own-teacher distillation or foreign teacher. No generated face images. Sampling prior q=.7p+.3/7 is derived only from TRAIN counts, yielding about1,238disgust draws/epoch versus330under natural sampling. Draws repeat real training images with random augmentation; they are not new people. Internal output bias starts at log(q/p); exported bias adds log(p/q). This matches initialization exactly and accounts for the sampling shift without a hand-set disgust threshold. The prior-shift assumption is approximate after finite training, not a performance guarantee.

Seed808; at most8epochs,80minutes checked at epoch boundaries, patience6. Validation selection requires both accuracy and disgust recall strictly improve and macro/disgust F1 not decrease; score.5accuracy+.3macroF1+.2disgustF1; student/EMA. At most one frozen qualifying candidate can be tested. The prospectively saved recall-priority gate requires test accuracy/macroF1/disgustF1 improve and disgust recall not decrease; all precision regressions are still reported. This differs from the preceding experiment's all-seven-floor gate and was declared BEFORE test access. No alternative is tested after a candidate fails.

Primary architecture authority: https://doi.org/10.1007/s41870-025-02531-y . Class-prior adjustment background: Menon et al., ICLR2021, https://doi.org/10.48550/arXiv.2007.07314 . The sampling-mixture/bias-folding recipe and residual refinement are project choices, not a claim to reproduce that paper's protocol.

## Complete new validation log

Training accuracy in machine logs is measured on augmented, oversampled q-distributed examples with dropout. Validation uses natural-prior-corrected student/EMA logits. They measure different conditions; their gap is not an ordinary generalization estimate.

| Epoch | Validation accuracy | Macro F1 | Disgust recall | Disgust F1 | Actual disgust draws |
|---|---:|---:|---:|---:|---:|
{log}

## Validation comparison

{'Frozen qualified candidate' if qualified else 'Best observed diagnostic only; no candidate qualified'}; N=3970, not test.

| Metric | Baseline | Candidate/diagnostic | Difference |
|---|---:|---:|---:|
{table(vc)}

## Full test comparison

{('N='+str(test['n'])+'. All seven classes and all retained errors included.') if test else 'No new test score was produced because no candidate qualified on validation. The previous evaluated model remains active.'}

| Metric | Previous model | New candidate | Difference |
|---|---:|---:|---:|
{table(test) if test else ''}

The validation cohort is repeatedly reused and the combined Kaggle test is historically inspected. These are exploratory descriptive results, not fresh/identity-disjoint confirmation or significance evidence. FER2013 low-resolution expressions do not establish internal mood or webcam-domain accuracy. Real GamER fixed-game/30-second participant-held-out physiology results remain unchanged and show no primary fusion F1 gain. No new physiological, weather, raga-outcome or therapeutic efficacy is claimed.

Exact artifacts: reports/bottleneck_audit_20261108/audit.json and reports/spatial_generalization_20261108/. All original resources and prior failures preserved.
"""
 transfer_log='\n'.join(f"| {r['epoch']} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} | {r['val_disgust_recall']:.4f} | {r['val_disgust_f1']:.4f} |" for r in residual_rows)
 pre=read(TRANSFER/'pretrained_provenance.json');tt=read(ROOT/'reports/translation_screen_20261108/validation.json')
 extra=f"""## Separate hybrid transfer experiment

This is **FusDenseNet v4 plus a frozen ImageNet DenseNet121 residual feature branch**, a disclosed architecture extension. It retains every FusDenseNet feature/classifier path and adds1024-channel features through a LayerNorm/1024->128->7 residual head. ONLY the new residual head was trained for{len(residual_rows)}epochs; both encoders remained frozen. These epochs are not full-network training. Zero-initialized residual outputs preserve initial FusDenseNet predictions. No FER-pretrained teacher, FERPlus annotations, generated face data or altered evaluation labels were used. The ImageNet source was not trained with FER expression labels; complete upstream image overlap cannot be ruled out.

Official Torchvision ImageNet1K_V1 URL:{pre['url']}; SHA:{pre['sha256']}; bytes:{pre['bytes']}. Native74.434% is ImageNet top1, never a FER/project result. Torchvision BSD3Clause notice preserved; no ImageNet images downloaded or redistributed. The224-pixel extraction attempt was abandoned before any feature-cache/model result dueCPUcost and remains preserved. The actual probe uses resize128/center112 with official ImageNet RGB normalization, a disclosedCPUadaptation differing from native224-pixel preprocessing. Exact cache/source SHA checks bind each frozen feature/label to its training/validation manifest. Selected composite inference is checked against cached computation before test. No sensor/body data contributes to this facial classification experiment.

AdamW LR.001/decay.01, CE power.25/smoothing.02, dropout.5, cosine20maximumepochs, patience8. A new residual head can learn complementary visual features while the original emotion model is retained; this is a hypothesis tested here, not guaranteed generalization or a causal controlled ablation. Added after v5epoch2validation and the negative translation screen, before any new test access; validation is repeatedly reused. Both recipes finish before one highest-scoring qualified candidate is frozen and tested. No alternative is tested after failure. Selection:{chosen}.

| Classifier-only epoch | Validation accuracy | Macro F1 | Disgust recall | Disgust F1 |
|---|---:|---:|---:|---:|
{transfer_log}

The separate five-view2-pixel translation screen measured validation accuracy{100*tt['accuracy']:.2f}%, macroF1{tt['classification_report']['macro avg']['f1-score']:.4f}, disgust recall{100*tt['classification_report']['disgust']['recall']:.2f}%; qualified:{tt['qualified']}. It was rejected and not used for test selection or deployment. It is not a new training result.

"""
 direct_log='\n'.join(f"| {r['epoch']} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} | {r['val_disgust_recall']:.4f} | {r['transfer_mix']:.2f} |" for r in direct_rows)
 extra+=f"""## Direct-label hybrid classifier and early-stopping correction

The residual-head recipe improved aggregate validation accuracy but did not improve disgust recognition. A third recipe trained the SAME new head directly on FER TRAIN labels instead of adding already-confident FusDenseNet training logits to its training loss. Source backbone weights and feature caches are unchanged. The old model recognizes95.45%oftraining disgust examples; under residual-only correction this leaves little error signal for a new classifier to learn those expressions. This is a motivating optimization diagnosis, not a controlled causal proof.

The direct head uses inverse-square-root frequency CE weights, no label smoothing, dropout.5, AdamW.001/decay.01. Inference combines the old calibrated probabilities (validation-fitted baseline temperature recorded in protocol) with new-head probabilities at one of3declared weights .25/.5/.75. No class-specific thresholds or excluded predictions. The core retains at least25%weight. This is **v7probability-ensemble hybrid**, NOTpureFusDenseNet and NOTthev6residual-logit recipe. It uses no new dataset images, synthetic faces or EEG/HRV labels.

The first direct-head trial stopped at{len(short_rows)}epochs with no qualified candidate, despite improving validation objectives. Its original protocol/logs/negative result are preserved. A separate run corrects the early-stopping rule: patience12tracks unconstrained validation objective progress, while acceptance still requires BOTH accuracy and disgust recall improve plus macro/disgustF1 floors. Maximum60epochs; actual completed{len(direct_rows)}. Both feature extractors remain frozen; these are classifier-only epochs. Three probability weights and many epochs on repeatedly reused validation mean substantial selection bias; no fresh-confirmatory claim. All these choices were made before any new test access. At most ONE candidate across completed recipes is tested, with no alternate after failure.

| Direct-head epoch | Combined validation accuracy | Macro F1 | Disgust recall | New-head probability weight |
|---|---:|---:|---:|---:|
{direct_log}

Standalone new-head validation metrics are saved separately, not treated as combined-model metrics. Training accuracy is the standalone head with dropout; combined validation accuracy uses BOTHencoders. Curves have different definitions and are not an ordinary generalization-gap estimate.

"""
 text=text.replace('## Validation comparison',extra+'## Validation comparison')
 (ROOT/'SPATIAL_GENERALIZATION_RESULTS.md').write_text(text,encoding='utf-8')
 intro=f"Audit-driven experiments completed{len(rows)}full-network and{len(transfer_rows)}separate residual-head epochs. Active version{ck['architecture']['version']}; v6/v7are explicitly FusDenseNet+ImageNet DenseNet hybrid variants, v7uses probability combination. Qualifying validation replacement:{qualified}; deployed:{deployed}. Active test accuracy{100*active['accuracy']:.2f}%, macroF1{m['f1-score']:.4f}, disgust recall{100*d['recall']:.2f}%. Full measured audit and regressions: SPATIAL_GENERALIZATION_RESULTS.md. No80%, fresh-test or reliable-webcam result is claimed.\n\nEarlier numerical sections below are historical.\n\n"
 for name in ['EXPERIMENTS.md','MODEL_CARDS.md','PAPER_IMPLEMENTATION_MAPPING.md','FURTHER_RESULTS.md']:
  p=ROOT/name;backup=RUN/'document_backups'/name;backup.parent.mkdir(exist_ok=True)
  if not backup.exists():shutil.copy2(p,backup)
  p.write_text(intro+backup.read_text(encoding='utf-8'),encoding='utf-8')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13,3.8))
 for ax,key,title,bv in zip(axes,['val_accuracy','val_macro_f1','val_disgust_recall'],['Validation accuracy','Validation macro F1','Validation disgust recall'],[base['accuracy'],base['classification_report']['macro avg']['f1-score'],base['classification_report']['disgust']['recall']]):
  for rr,label in [(rows,'Spatial:full-network'),(residual_rows,'Residual:head only'),(short_rows,'Direct:short head'),(direct_rows,'Direct:long head')]:ax.plot(range(len(rr)+1),[bv]+[r[key] for r in rr],label=label)
  ax.set_title(title);ax.set_xlabel('Epoch within recipe; compute differs');ax.grid(alpha=.2)
 axes[0].legend(fontsize=7)
 fig.tight_layout();fig.savefig(ROOT/'reports/spatial_generalization_trials.png',dpi=160);plt.close(fig)
 print(json.dumps({'epochs':len(rows),'qualified':qualified,'deployed':deployed,'active_accuracy':active['accuracy'],'latest_test':test},indent=2))
if __name__=='__main__':main()
