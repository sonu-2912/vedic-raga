"""Report pure-network results separately from every historical hybrid score."""
import json,shutil
from pathlib import Path
import torch
from scripts.colab_common import digest,write_json,model_from
from scripts.train_pure_fusdense import assert_pure,selection_score
from scripts.train_deep_supervision import metric_vector
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'reports/pure_fusdense_20261108';NATIVE=ROOT/'reports/pure_native_residual_20261108';SELECTED=ROOT/'reports/pure_selected_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def stats(r):return {k:v for k,v in r.items() if k!='predictions'}
def compare(before,after):
 a,b=metric_vector(before),metric_vector(after)
 return {'n':after['n'],'metrics':{k:{'baseline':a[k],'colab':b[k],'difference':b[k]-a[k]} for k in a}}
def delta(old,new):
 result={}
 for k,v in new.items():
  if k not in old:
   d=result.setdefault(k.split('.')[0],{'changed_tensors':0,'changed_elements':0,'max_abs_change':0.});d['added_tensors']=d.get('added_tensors',0)+1;d['added_elements']=d.get('added_elements',0)+v.numel()
  elif not torch.equal(old[k],v):
   d=result.setdefault(k.split('.')[0],{'changed_tensors':0,'changed_elements':0,'max_abs_change':0.});d['changed_tensors']+=1;d['changed_elements']+=int((old[k]!=v).sum());d['max_abs_change']=max(d['max_abs_change'],float((old[k].float()-v.float()).abs().max()))
 return result

def main():
 torch.set_num_threads(2);done=read(RUN/'completion.json');history=read(RUN/'training_history.json');rows_a=[r for r in history if r.get('phase')=='full_network_pure_supcon_rdrop'];rows_b=[r for r in read(NATIVE/'training_history.json') if r.get('phase')=='full_network_pure_native_residual'];rows=[{**r,'recipe':name} for name,rr in [('v4regularized',rows_a),('v8native',rows_b)] for r in rr];baseval=read(RUN/'initial_validation.json');baseline=read(RUN/'baseline_test.json');hybrid=read(RUN/'hybrid_reference_test.json');selection=read(SELECTED/'selection.json');chosen=selection['chosen'];qualified=bool(chosen);observed_path,observed=max([(p,read(p)) for folder in [RUN,NATIVE] for p in folder.glob('validation_[0-9][0-9]_*.json')],key=lambda pair:selection_score(pair[1]));selectedval=read(SELECTED/'best_validation.json') if qualified else baseval
 testfile=SELECTED/'deployment/reports/emotion_evaluation.json';newtest=read(testfile) if testfile.exists() else None;comparison=read(SELECTED/'deployment/reports/colab_comparison.json') if newtest else None
 if newtest:
  assert newtest['predictions']['truth']==baseline['predictions']['truth']==hybrid['predictions']['truth'] and newtest['n']==baseline['n']==hybrid['n']==6561
  hybrid_comparison=compare(hybrid,newtest);hybrid_comparison.update(baseline_name='Former hybrid v7, reference only',candidate_name='New pure FusDenseNet',limitation='Same historically inspected test. Distinct architectures. Hybrid was not a teacher or feature source.')
 else:hybrid_comparison=None
 active=read(ROOT/'reports/emotion_evaluation.json');activeval=read(ROOT/'reports/validation_evaluation.json');active_sha=digest(ROOT/'models/fusdense.pt');active_ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True);assert active['checkpoint_sha256']==activeval['checkpoint_sha256']==active_sha;deployed=bool(newtest and newtest['checkpoint_sha256']==active_sha and comparison['recommended_for_integration'])
 pure_evaluation=newtest or baseline;pure_ck=torch.load((SELECTED/'best.pt' if qualified else RUN/'pure_baseline.pt'),map_location='cpu',weights_only=True);assert_pure(pure_ck);pure_model=model_from(pure_ck,torch.device('cpu'));old=torch.load(RUN/'pure_baseline.pt',map_location='cpu',weights_only=True);latest=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=True)
 if comparison:comparison.update(baseline_name='Earlier pure FusDenseNet v4',candidate_name='New pure FusDenseNet',comparison_kind='pure_baseline')
 trial={'method':'pure_fusdense','history':rows,'completed_epochs':len(rows),'completed_full_epochs':len(rows),'completed_head_epochs':0,'curve_run':chosen['run'] if chosen else str(observed_path.parent.relative_to(ROOT)).replace('\\','/'),'curve_phase':'full_network_pure_native_residual' if (chosen and 'native_residual' in chosen['run']) or (not chosen and observed_path.parent==NATIVE) else 'full_network_pure_supcon_rdrop','external_encoder':False,'teacher':False,'single_view_only':True,'training_only_projection_parameters':93080,'inference_parameters':sum(p.numel() for p in pure_model.parameters()),'candidate_checkpoint_sha256':newtest['checkpoint_sha256'] if newtest else None,'test_comparison':comparison,'former_hybrid_comparison':hybrid_comparison,'validation_comparison':compare(baseval,selectedval if qualified else observed),'validation_comparison_kind':'Frozen qualifying pure checkpoint' if qualified else 'Best observed validation diagnostic only; no candidate qualified','validation_candidate_qualified':qualified,'deployed':deployed,'limitation':'Pure FusDenseNet, no external encoder/teacher or inference ensemble. Auxiliary projection is stripped. Baseline comparison is earlier PUREmodel; the former hybrid is reported separately. Reused validation and historically inspected test are not fresh/identity-disjoint confirmation. Training uses paired batches, augmentation and two dropout draws; raw single-view validation has different conditions.'}
 write_json(ROOT/'reports/latest_emotion_experiment.json',trial)
 if deployed:write_json(ROOT/'reports/emotion_training_comparison.json',comparison)
 pure_summary={'architecture':pure_ck['architecture'],'parameters':sum(p.numel() for p in pure_model.parameters()),'classifier_inputs':pure_model.classifier[0].in_features,'pure_test':stats(pure_evaluation),'pure_validation':stats(selectedval),'diagnostic_validation':stats(observed),'qualified':qualified,'new_test_available':bool(newtest),'test_label':'New pure candidate' if newtest else 'Earlier pure baseline; no new test','pure_deployed':deployed,'active_is_pure':not any(k.startswith('transfer_') for k in active_ck['state_dict']),'active_test':stats(active),'hybrid_reference':stats(hybrid),'additional_full_epochs':len(rows),'checkpoint_epoch':pure_ck['epoch'],'external_encoder':False,'teacher':False,'single_view_only':True}
 write_json(ROOT/'reports/pure_fusdense_results_summary.json',pure_summary)
 summary={'protocol':read(RUN/'protocol.json'),'evaluation_protocol':read(RUN/'evaluation_protocol.json'),'completion':done,'native_completion':read(NATIVE/'completion.json'),'selection':selection,'native_protocol':read(NATIVE/'protocol.json'),'initialization_audit':read(RUN/'initialization_audit.json'),'latest_experiment':trial,'active_source_sha256':active_sha,'source_pure_baseline_sha256':digest(RUN/'pure_baseline.pt'),'frozen_candidate_sha256':digest(SELECTED/'best.pt') if qualified else None,'selected_weight_changes':delta(old['state_dict'],pure_ck['state_dict']),'trained_student_weight_changes':delta(old['state_dict'],latest['state_dict']),'native_trained_student_weight_changes':delta(old['state_dict'],torch.load(NATIVE/'latest.pt',map_location='cpu',weights_only=True)['state_dict']),'best_observed_validation':metric_vector(observed),'best_observed_file':observed_path.name,'results':pure_summary};write_json(RUN/'experiment_summary.json',summary)
 active_ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True);network=model_from(active_ck,torch.device('cpu'));m=active['classification_report']['macro avg'];d=active['classification_report']['disgust'];ah=read(ROOT/'reports/training_history.json')
 write_json(ROOT/'reports/further_results_summary.json',{'accuracy':active['accuracy'],'macro_precision':m['precision'],'macro_recall':m['recall'],'macro_f1':m['f1-score'],'disgust':d,'test_n':active['n'],'checkpoint_sha256':active_sha,'parameters':sum(p.numel() for p in network.parameters()),'architecture':active_ck['architecture'],'selected_epoch':active['checkpoint_epoch'],'train_count':active_ck['train_count'],'val_count':active_ck['val_count'],'classifier_inputs':network.classifier[0].in_features,'selected_lineage_full_epochs':sum(not r.get('phase') or str(r.get('phase')).startswith('full_network_') for r in ah),'selected_lineage_head_epochs':sum(str(r.get('phase','')).startswith('head_refinement') for r in ah)})
 table=lambda c:'\n'.join(f"| {k.replace('_',' ')} | {v['baseline']:.6f} | {v['colab']:.6f} | {v['difference']:+.6f} |" for k,v in c['metrics'].items())
 log='\n'.join(f"| {r['recipe']} / {r['epoch']} | {r['val_accuracy']:.4f} | {r['val_macro_f1']:.4f} | {r['val_disgust_recall']:.4f} | {r['val_disgust_f1']:.4f} | {r['sample_counts'][1]} |" for r in rows)
 vc=trial['validation_comparison'];pm=pure_evaluation['classification_report']['macro avg'];pd=pure_evaluation['classification_report']['disgust']
 text=f"""# Pure FusDenseNet: actual full-network experiment

Completed **{len(rows)} full-network epochs**. No frozen-feature-only epochs, external encoder, hybrid teacher, external pretrained weights, probability mixture or inference ensemble in this experiment. The training-only projection is removed from exported state. Selected pure model **{sum(p.numel() for p in pure_model.parameters()):,} parameters**, architecture v{pure_ck['architecture']['version']}. v4 has 1,828,896 parameters; v8 adds 117,360 native residual-dense parameters with the same classifier/input widths. Validation-qualified:{qualified}. New pure candidate deployed:{deployed}.

{pure_summary['test_label']}: accuracy **{100*pure_evaluation['accuracy']:.2f}%**, macro F1 **{pm['f1-score']:.4f}**, disgust recall **{100*pd['recall']:.2f}%**. {'This is a newly measured candidate test result.' if newtest else 'No new candidate qualified for test evaluation; these are historical baseline scores.'} Active app accuracy {100*active['accuracy']:.2f}%, macro F1 {m['f1-score']:.4f}, disgust recall {100*d['recall']:.2f}%; active architecture v{active_ck['architecture']['version']}. Former hybrid v7 reference: 59.26% accuracy, macro F1 0.5530, disgust recall 43.04%; it is not a pure FusDenseNet score.

## What changed inside the pure training path

All three fusion-dense stages, cross-dense connections, four-head attention, dilated dense convolutions, multi-pooling, early spatial paths and the existing classifier are retained. Full emotion-path weights are optimized, starting from the verified **pre-hybrid** pure checkpoint8fc1713541e49bac8fc6ec4f10a5d5ba4ca9ff3860e227caeda3b7b67870a618. No transfer_backbone or transfer_head keys are accepted. The older Navrasa model and pretrained DenseNet121 are not initialized or consulted by this trainer.

Each64-image training batch draws50natural-frequency images and2additional distinct images per FERclass. Each batch has unique indices and at least2examples per class; identities are unavailable, so cross-identity sampling is not claimed. The same23,503fixed training images are the only pool; repeats across batches are expected. Approximate class prior q=(50p+2)/64; CE uses normalized sqrt(p/q)importance, a disclosed partial cost-sensitive risk. No logit bias/prior offsets are folded into inference. Disgust sampling grows from1.4%natural to4.22%ofdraws; counts for every epoch are recorded, not presented as new people.

A TRAIN-only LayerNorm/588->128->128projection (93,080parameters) receives late fused features. Class-balanced supervised contrastive loss pulls same-labelled TRAIN images together and pushes different labels apart, with self-pairs excluded and temperature.1. It is a project adaptation of Khosla etal.(2020), **Supervised Contrastive Learning**, https://doi.org/10.48550/arXiv.2004.11362 . Projection and optimizer state are stored only for training recovery, not in the inference model. This loss uses the existing FERlabels; no automatic relabelling or claims that labels are verified are made.

RDrop uses mean CE from two independent classifier-dropout draws plus symmetric KL. Because dropout exists only in the classifier and old BatchNorm references are frozen, one deterministic backbone forward and two classifier forwards give the same parameter gradients as two full forwards. This is numerically tested. Basis: Liang etal.(2021), **R-Drop: Regularized Dropout for Neural Networks**, https://doi.org/10.48550/arXiv.2106.14448 . These are training extensions, not exact original-author loss settings. Primary architecture authority remains https://doi.org/10.1007/s41870-025-02531-y .

Objective: mean importance-weighted CE (smoothing.01) +.5symmetricKL +.05class-balancedSupCon. Classifier dropout.4; oldBNstatistics fixed (the preceding calibration audit found no large shift and recalibration worsened validation). AdamW LR8e-5backbone/6e-5classifier/5e-4projection,1epochlinear warmup/cosine, mild5degree affine/3%translation/5%scale and brightness/contrast augmentation. No erasing or mixed-face synthetic training examples.

## Native residual-dense FusDenseNet variant

A second pure recipe adds two dense growth layers (12channels each,3x3dilation1/2), GroupNorm6groups and a zero-initialized1x1localfusion projection inside EACHof6fusionbranches. Each branch still outputs60channels; all attention, concatenation, pooling and classifier input/output widths remain. These are native feature blocks, not a separate backbone/encoder or independently trained predictor. Added117,360params; purev8total1,946,256. Zero residual output preserves initialv4logits, numerically verified; newfeature gradients and exact shared-featureRDrop gradients are tested. GroupNorm has no running batch statistics. Basis: Wu,Y.andHe,K.(2018), GroupNormalization, https://doi.org/10.48550/arXiv.1803.08494 . This depth/normalization choice is an explicit project extension, not recovered author dimensions.

Both recipes independently initialize from the same pre-hybrid purebaseline, not eachother or a hybrid. The v4run budget was capped/reallocated during training before test; actual completedv4epochs:{len(rows_a)} (target12was a maximum, not an achievedcount). Actual nativev8epochs:{len(rows_b)}. Interrupted/unfinished epochs are not counted. The native stage uses backboneLR4e-5/newfeatures3e-4 and a45-minute epoch-boundary budget, maximum6epochs; the other losses/sampler/classifier are retained. This is not an isolated equal-budget architecture ablation because depth/LR/budget differ. Both recipes finish before one candidate is frozen. Full protocols and interruption records are preserved.

## Selection and test discipline

The initial v4protocol used at most12epochs, minimum6before patience6;110-minute wall budget checked at completed-epoch boundaries. Early stopping tracks unconstrained validation objective progress even below deployment floors. Student and EMA are evaluated on all3,970fixed validation faces, including50disgust faces. A checkpoint qualifies only if BOTHvalidation accuracy and disgust recall strictly exceed the earlier pure baseline58.0353%/28%, with macro/disgustF1 not lower. Score.4accuracy+.4macroF1+.2disgustF1. The saved final candidate is frozen before one full test evaluation; no alternate test search.

Inference is **single-view only** with validation temperature calibration; temperature cannot change the predicted class. No flip/crop/ensemble selection. Deployment additionally requires strictly positive TEST accuracy, macroF1, disgust recall and disgustF1 gains over the earlier pure model, specified before test. Former hybrid values are reported separately because the user explicitly prioritizes pure FusDenseNet. An accepted pure model may score below that hybrid on individual metrics; no such regression is hidden. If no candidate qualifies or passes, no improved pure deployment is claimed.

The validation cohort is repeatedly reused and the combined Kaggle test has been historically inspected. These are exploratory descriptive results, not fresh, identity-disjoint confirmation or isolated causal architecture ablations. All7classes and all6,561retained test errors are included. No80%or reliablewebcam result is claimed. Only330unique training disgust images and50validation examples limit minority-class evidence.

## Every completed full-network epoch

Training agreement averages the two dropout logits on augmented, class-paired samples. Validation uses one dropout-disabled raw image. The distributions/conditions differ; this is not a conventional train-validation gap comparison. The raw baseline audit83.01%train/58.04%val and95.45%/28%disgust recall used consistent raw inference conditions on its two cohorts.

| Epoch | Validation accuracy | Macro F1 | Disgust recall | Disgust F1 | Actual disgust draws |
|---|---:|---:|---:|---:|---:|
{log}

## Validation comparison ({trial['validation_comparison_kind']})

| Metric | Earlier pure | Candidate/diagnostic | Difference |
|---|---:|---:|---:|
{table(vc)}

## Full test: earlier pure versus new pure

{'All6561same retained records, including79disgust faces.' if newtest else 'No new test evaluation: no validation-qualified candidate. Earlier pure scores remain reference-only.'}

| Metric | Earlier pure | New pure | Difference |
|---|---:|---:|---:|
{table(comparison) if comparison else ''}

## Full test: former hybrid versus new pure (reference only)

| Metric | Former hybrid | New pure | Difference |
|---|---:|---:|---:|
{table(hybrid_comparison) if hybrid_comparison else 'No new pure test result; no comparison is claimed.'}

The actual current app checkpoint is SHA{active_sha}. Selected/trained weight-change audits, executed source hashes and complete protocols are retained in reports/pure_fusdense_20261108/. All originals and earlier failures are preserved. Real EEG/HRV fixed-context tests remain unchanged and show no primary fusionF1gain. No synchronized face/body/raga-response, weather efficacy, therapeutic or recommendation-quality gain is established by this facial experiment.
"""
 (ROOT/'PURE_FUSDENSENET_RESULTS.md').write_text(text,encoding='utf-8')
 intro=f"Latest pure FusDenseNet experiment: {len(rows)}full-network epochs; qualifying validation:{qualified}; puredeployed:{deployed}. No external encoder/teacher, hybrid weights or inference ensemble. Full measured results, failed epochs and separate pure/hybrid comparisons:PURE_FUSDENSENET_RESULTS.md. Active model accuracy{100*active['accuracy']:.2f}%, macroF1{m['f1-score']:.4f}, disgust recall{100*d['recall']:.2f}%. Earlier sections are historical.\n\n"
 for name in ['EXPERIMENTS.md','MODEL_CARDS.md','PAPER_IMPLEMENTATION_MAPPING.md','FURTHER_RESULTS.md']:
  p=ROOT/name;backup=RUN/'document_backups'/name;backup.parent.mkdir(exist_ok=True)
  if not backup.exists():shutil.copy2(p,backup)
  p.write_text(intro+backup.read_text(encoding='utf-8'),encoding='utf-8')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from sklearn.metrics import ConfusionMatrixDisplay
 import numpy as np
 fig,ax=plt.subplots(figsize=(7,6));ConfusionMatrixDisplay(np.asarray(pure_evaluation['confusion_matrix']),display_labels=['angry','disgust','fear','happy','sad','surprise','neutral']).plot(ax=ax,cmap='Greens',colorbar=False);ax.set_title(pure_summary['test_label']+f" (n={pure_evaluation['n']})");fig.tight_layout();fig.savefig(ROOT/'reports/pure_fusdense_test.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,5))
 for ax,report,title in [(axes[0],baseval,'Earlier pure baseline'),(axes[1],selectedval if qualified else observed,'Frozen pure candidate' if qualified else 'Best observed diagnostic; not qualified')]:
  ConfusionMatrixDisplay(np.asarray(report['confusion_matrix']),display_labels=['angry','disgust','fear','happy','sad','surprise','neutral']).plot(ax=ax,cmap='Blues',colorbar=False,xticks_rotation=35);ax.set_title(title)
 fig.suptitle('Fixed validation cohort: all 3,970 faces, including 50 disgust faces');fig.tight_layout();fig.savefig(ROOT/'reports/pure_fusdense_validation.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,3.8))
 for ax,key,title,bv in zip(axes,['val_accuracy','val_macro_f1','val_disgust_recall'],['Validation accuracy','Validation macro F1','Validation disgust recall'],[baseval['accuracy'],baseval['classification_report']['macro avg']['f1-score'],baseval['classification_report']['disgust']['recall']]):
  for label,rr in [('v4 regularized',rows_a),('v8 native residual',rows_b)]:
   ax.plot(range(len(rr)+1),[bv]+[r[key] for r in rr],marker='o',label=label)
  ax.set_title(title);ax.set_xlabel('Epochs from shared pure baseline');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(ROOT/'reports/pure_fusdense_training.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,3.6))
 for label,rr,style in [('v4',rows_a,'-'),('v8',rows_b,'--')]:
  for key in ['loss','CE','RDrop','SupCon']:ax.plot(range(1,len(rr)+1),[r[key] for r in rr],style,label=f'{label} {key}')
 ax.set_title('PureFus training losses; component scales differ');ax.set_xlabel('Epochs from shared pure baseline');ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(ROOT/'reports/pure_fusdense_losses.png',dpi=160);plt.close(fig)
 print(json.dumps({'pure_qualified':qualified,'pure_deployed':deployed,'epochs':len(rows),'test_comparison':comparison,'former_hybrid_comparison':hybrid_comparison},indent=2))
if __name__=='__main__':main()
