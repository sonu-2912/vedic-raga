"""Publish actual completed deep-supervision evidence, including failed trials."""
import json,shutil
from pathlib import Path
from datetime import datetime
import torch
from scripts.colab_common import digest,write_json,model_from
from scripts.train_deep_supervision import metric_vector
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/deep_supervision_20261107'
HEAD=ROOT/'reports/pruned_head_20261107'
LONG=ROOT/'reports/consistency_long_20261107'
SAM=ROOT/'reports/sam_head_20261107'
SELECTED=ROOT/'reports/balanced_selected_20261107'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def comparison(before,after,n):
    a,b=metric_vector(before),metric_vector(after)
    return {'n':n,'metrics':{k:{'baseline':a[k],'colab':b[k],'difference':b[k]-a[k]} for k in a}}
def changes(old,new):
    result={}
    for name,tensor in new.items():
        if tensor.shape!=old[name].shape:
            item=result.setdefault(name.split('.')[0],{'changed_tensors':0,'changed_elements':0,'max_absolute_change':0.})
            item.setdefault('shape_changes',[]).append({'tensor':name,'before':list(old[name].shape),'after':list(tensor.shape)})
            continue
        if not torch.equal(tensor,old[name]):
            item=result.setdefault(name.split('.')[0],{'changed_tensors':0,'changed_elements':0,'max_absolute_change':0.})
            item['changed_tensors']+=1;item['changed_elements']+=int((tensor!=old[name]).sum());item['max_absolute_change']=max(item['max_absolute_change'],float((tensor.float()-old[name].float()).abs().max()))
    return result

def main():
    torch.set_num_threads(2)
    done=read(RUN/'completion.json');initial=read(RUN/'initial_validation.json');history=read(RUN/'training_history.json');epochs=[r for r in history if r.get('phase')=='full_network_deep_supervision']
    long_done=read(LONG/'completion.json');long_history=[r for r in read(LONG/'training_history.json') if r.get('phase')=='full_network_consistency_long']
    head_done=read(HEAD/'completion.json');selection=read(SELECTED/'selection.json');chosen=selection['chosen'];qualified=bool(chosen)
    pruned_history=[r for r in read(HEAD/'training_history.json') if r.get('phase')=='head_refinement_pruned256']
    sam_done=read(SAM/'completion.json');sam_history=[r for r in read(SAM/'training_history.json') if r.get('phase')=='head_refinement_sam512'];head_history=pruned_history+sam_history
    selected_run=ROOT/chosen['run'] if chosen else RUN
    baseline=torch.load(RUN/'baseline.pt',map_location='cpu',weights_only=True);selected=torch.load(selected_run/'best.pt',map_location='cpu',weights_only=True);latest=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=True)
    reports=[(p,read(p)) for folder in [RUN,HEAD,LONG,SAM] for p in folder.glob('validation_[0-9][0-9]_*.json')]
    best_path,observed=max(reports,key=lambda pair:.35*pair[1]['accuracy']+.65*pair[1]['classification_report']['macro avg']['f1-score'])
    selected_val=read(selected_run/'best_validation.json') if qualified else initial
    display_val=selected_val if qualified else observed
    val_comparison=comparison(initial,display_val,initial['n']);val_comparison['limitation']='Validation selection/diagnostic only, not held-out test or webcam performance.'
    test_path=SELECTED/'deployment/reports/colab_comparison.json';test=read(test_path) if test_path.exists() else None
    active_sha=digest(ROOT/'models/fusdense.pt');deployed=bool(test and test['recommended_for_integration'] and test['checkpoint_sha256']==active_sha)
    summary={'method':'balanced_capacity_search','completed_epochs':len(epochs)+len(long_history)+len(head_history),'completed_full_epochs':len(epochs)+len(long_history),'completed_head_epochs':len(head_history),'head_run_completion':head_done,'sam_run_completion':sam_done,'long_run_completion':long_done,'auxiliary_full_epochs':len(epochs),'conservative_full_epochs':len(long_history),'selection':selection,'deep_run_completion':done,'protocol':read(RUN/'protocol.json'),'evaluation_protocol':read(RUN/'evaluation_protocol.json'),'source_sha256':digest(RUN/'baseline.pt'),'selected_best_sha256':digest(selected_run/'best.pt'),'active_sha256':active_sha,'deployed':deployed,'validation_candidate_qualified':qualified,'selected_candidate':selected.get('candidate_type','baseline'),'selected_epoch':selected.get('epoch'),'validation_initial_raw':metric_vector(initial),'validation_selected_raw':metric_vector(selected_val),'best_observed_validation':metric_vector(observed),'best_observed_file':best_path.name,'validation_comparison':val_comparison,'test_comparison':test,'selected_weight_changes':changes(baseline['state_dict'],selected['state_dict']),'trained_student_weight_changes':changes(baseline['state_dict'],latest['state_dict']),'synthetic':False}
    write_json(RUN/'experiment_summary.json',summary)
    write_json(ROOT/'reports/latest_emotion_experiment.json',{'method':summary['method'],'completed_epochs':len(epochs)+len(long_history)+len(head_history),'completed_full_epochs':len(epochs)+len(long_history),'completed_head_epochs':len(head_history),'curve_run':selected_run.relative_to(ROOT).as_posix() if chosen else RUN.relative_to(ROOT).as_posix(),'curve_phase':'head_refinement_pruned256' if chosen and selected_run==HEAD else 'head_refinement_sam512' if chosen and selected_run==SAM else 'full_network_consistency_long' if chosen and selected_run==LONG else 'full_network_deep_supervision','candidate_checkpoint_sha256':test['checkpoint_sha256'] if test else None,'test_comparison':test,'validation_comparison':val_comparison,'validation_candidate_qualified':qualified,'validation_comparison_kind':'selected qualifying candidate' if qualified else 'best observed diagnostic; did not qualify','synthetic':False,'limitation':'All seven classes/errors retained. Repeated validation/historically inspected test, not fresh or identity-disjoint. No reliable webcam, joint physiology/raga/weather or clinical claim.'})
    if deployed:write_json(ROOT/'reports/emotion_training_comparison.json',test)
    active=read(ROOT/'reports/emotion_evaluation.json');val=read(ROOT/'reports/validation_evaluation.json');ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True);network=model_from(ck,torch.device('cpu'));assert active_sha==active['checkpoint_sha256']==val['checkpoint_sha256']
    active_history=read(ROOT/'reports/training_history.json');m=active['classification_report']['macro avg'];d=active['classification_report']['disgust']
    write_json(ROOT/'reports/further_results_summary.json',{'accuracy':active['accuracy'],'macro_precision':m['precision'],'macro_recall':m['recall'],'macro_f1':m['f1-score'],'disgust':d,'test_n':active['n'],'checkpoint_sha256':active_sha,'parameters':sum(p.numel() for p in network.parameters()),'architecture':ck['architecture'],'selected_epoch':active['checkpoint_epoch'],'train_count':ck['train_count'],'val_count':ck['val_count'],'classifier_inputs':network.classifier[0].in_features,'selected_lineage_full_epochs':sum(not r.get('phase') or str(r.get('phase')).startswith('full_network_') for r in active_history),'selected_lineage_head_epochs':sum(str(r.get('phase','')).startswith('head_refinement') for r in active_history)})
    rows='\n'.join(f"| {r['epoch']} | {r['val_accuracy']:.4f} | {r['val_macro_precision']:.4f} | {r['val_macro_f1']:.4f} | {r['val_disgust_precision']:.4f} | {r['val_disgust_recall']:.4f} | {r['val_disgust_f1']:.4f} |" for r in epochs)
    def table(report):return '\n'.join(f"| {k.replace('_',' ')} | {v['baseline']:.6f} | {v['colab']:.6f} | {v['difference']:+.6f} |" for k,v in report['metrics'].items())
    test_text=table(test) if test else 'No new test evaluation was performed because no validation candidate passed all seven floors. Do not interpret the validation diagnostic as test performance.'
    text=f'''# Further balanced training experiments: actual results

Completed {len(epochs)} auxiliary-supervised and {len(long_history)} conservative full emotion-path epochs, plus {len(head_history)} separate frozen-backbone classifier epochs. Validation-qualified replacement: **{qualified}**. Selected recipe: **{chosen['run'] if chosen else 'none'}**. New candidate deployed: **{deployed}**. Active test accuracy **{100*active['accuracy']:.2f}%**, macro F1 **{m['f1-score']:.4f}**; active validation accuracy **{100*val['accuracy']:.2f}%**. No80% result is claimed. Active source SHA `{active_sha}`.

## Method and data

Same verified23,503training/3,970validation faces. All candidates retain FusDenseNet v4 mechanisms. Active inference: {sum(p.numel() for p in network.parameters()):,} parameters,3276→{ck['architecture'].get('head_width',128)}→7 classifier. The full-feature recipe keeps the512-unit head; the separate capacity recipe changes only its width to256, a disclosed project dimension choice. All three fusion stages, cross-attention and multi-pooling remain. A training-only LayerNorm/588→128→7 companion supervises only the late fused feature slice, so its gradients reach dense fusion/attention without using the spatial shortcut. The companion is absent from the saved inference state. In the auxiliary recipe only, training masks the early spatial shortcut in20%of examples with inverted scaling; evaluation never masks it. MixUp Beta(.2,.2) applies to half of batches and both labels receive weighted CE. Weight power.25, label smoothing.02, main+ .25companion loss. Early LR1e-5, late2e-4, main classifier1e-4, companion1e-3; two-epoch warmup then cosine decay, frozen BN running references, EMA/student validation. No external teacher or unverified FER+ labels.

This is a disclosed project training extension, not exact author reproduction or a controlled causal architecture ablation. Basis: Lee et al. (2015), Deeply-Supervised Nets, https://proceedings.mlr.press/v38/lee15a.html . MixUp: Zhang et al., https://arxiv.org/abs/1710.09412 . Primary authority remains https://doi.org/10.1007/s41870-025-02531-y .

## Separate capacity-control recipe

A second exploratory validation-only recipe was declared after two deep-supervision epochs, before any new test access. It freezes the baseline feature extractor and ranks512hidden neurons by class-balanced mean training activation multiplied by output-weight contribution. It retains256units, folds the mean discarded contribution into the output bias, and retrains that smaller MLP on training-normalized features. AdamW.001, weight decay.003, dropout.5, CE weight power.25/smoothing.02 plus.1correct-teacher KL T2. Neither normalization, neuron ranking nor teacher targets use validation/test labels. The final normalization is algebraically folded into the first Linear layer; full validation predictions must remain identical after conversion. This is classifier-only training, not additional full-network epochs. The initial full512-unit baseline defines the same seven validation floors.

Source-feature weights are copied unchanged for a pruned candidate. This is not a causal capacity ablation: regularization and optimization also differ. All four candidate runs and negative results are retained. Only the highest-scoring qualifying validation candidate is frozen for one test evaluation; no alternate candidate may be tested afterward. Validation is exploratory and repeatedly reused, not independent.

## Final fast classifier optimizer comparison

A fourth and final exploratory recipe, declared during the continuation before test access, keeps the full512-unit classifier and frozen baseline features. It uses sharpness-aware optimization (rho.05) with AdamW LR.0001/wd.003, dropout.45, weighted CE power.25/smoothing.02 and.1correct-own-teacher KL T2. Both gradient passes reuse the same dropout masks. Original parameters are restored exactly before the optimizer update, including on second-pass failure. Normalization uses training features only and is folded back with validation-logit/argmax checks. It is classifier-only training, not extra full-network epochs or an architectural change. Basis: Foret et al. (2021), https://doi.org/10.48550/arXiv.2010.01412 . This project recipe is not exact original SAM/FusDenseNet reproduction.

## Longer continuation and rejected foreign teacher

After the negative first recipes, a third exploratory run continued the previously successful own-teacher method from the same deployed baseline. Backbone LR2e-5, classifier5e-5, CE class-weight power.25/smoothing.02, dropout.45, mild affine/brightness/contrast augmentation and.2correct-teacher KL T2. Teacher inputs are only source/manifest-SHA-bound TRAIN features from the same baseline; raw teacher inference matches its deployed single-view mode. All training faces still receive CE. No MixUp, companion, shortcut dropout or external teacher contributes to this continuation. Student/EMA selection uses the same seven floors and.35accuracy+.65macroF1. Maximum12epochs, patience8,90-minute budget at completed-epoch boundaries.

An official Apache2-licensed OpenCV MobileFaceNet teacher was separately screened on all3,970fixed validation faces using replicated grayscale/112×112 inputs without invented landmarks. It measured43.4509% accuracy, below the declared screen threshold, and was **rejected and never used to train FusDenseNet**. Its published88.27% belongs to RAF-DB, not this project. The foreign checkpoint's complete training-image manifest is unavailable, so upstream overlap is unknown. This adapted diagnostic is not a clean independent benchmark or a FusDenseNet result. Original weights/notice/SHA were preserved locally; no foreign pixels or weights are published with the project. Exact screen: reports/teacher_screen_20261107/.

## Selection safeguards

The auxiliary full-network run uses maximum12epochs, patience5 and a90-minute budget checked after completed epochs (a long/interrupted epoch can overshoot). Score.35accuracy+.65macro F1, only if accuracy, macro precision/recall/F1 and disgust precision/recall/F1 all meet the initial raw-validation floors. Student and EMA are compared each epoch. Flip averaging/calibration use validation only; strict flip selection also prevents seven-metric regression versus raw. The separate evaluation protocol was saved before test access. Only one frozen qualifying candidate can be tested; strict integration requires positive accuracy/macro F1/disgust F1 and none of the seven declared metrics decreasing. The previous evaluated model remains active if no candidate qualifies or any strict test check fails.

## Complete new validation log

| Epoch | Accuracy | Macro precision | Macro F1 | Disgust precision | Disgust recall | Disgust F1 |
|---|---:|---:|---:|---:|---:|---:|
{rows}

Rows include rejected epochs. Raw student/EMA reports for every epoch are retained. Training agreement is lambda-weighted agreement with mixed labels under augmentation/dropout; it is not ordinary facial classification accuracy. Elapsed epoch seconds can include pauses and are not CPU-time measurements. Best observed diagnostic: {best_path.name}; it is not necessarily eligible. Selected inference checkpoint index: {selected.get('epoch')}.

## Validation comparison ({'qualifying candidate' if qualified else 'best observed, nonqualifying diagnostic'})

| Metric | Initial | Candidate/diagnostic | Difference |
|---|---:|---:|---:|
{table(val_comparison)}

## Full same-set test comparison

| Metric | Before | Candidate | Difference |
|---|---:|---:|---:|
{test_text}

Every test face and all7classes/errors are retained. The6,561-face combined Kaggle test was historically inspected and validation repeatedly reused. It is not a fresh identity-disjoint or webcam benchmark. No claim follows that every user's disgust or smile will be recognized correctly. A weight-change audit proves training occurred, not improved generalization by itself.

## Other metrics remain separate

The real GamER fixed-game/fixed30s/participant-held-out EEG/ECG results still show no primary fusion F1 gain. No new synchronized face/physiology/raga/weather dataset was acquired. DENS dimension issues and DREAMER restricted access remain unresolved. No simulated signal is reported as real, and no recommendation-quality or therapeutic metric is invented.

## Reproduce

The completed evaluator is locked after test access and verifies/reuses its saved bundle. New prospective training requires a separate protocol and run directory. Original datasets, weights, reports and source resources remain preserved.

```powershell
.\\.venv\\Scripts\\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/deep_supervision_20261107 --device cpu --threads4 --strict-all-metrics --baseline-report reports/deep_supervision_20261107/baseline_test.json --baseline-manifest reports/deep_supervision_20261107/baseline_test_manifest.json
.\\.venv\\Scripts\\python.exe -m scripts.report_deep_supervision
```

If no candidate qualified, do not run the evaluator to create a baseline-only test result; the report records that test was intentionally not accessed. Exact protocol, per-epoch reports, auxiliary optimizer state for recovery, tensor changes and manifest hashes are under reports/deep_supervision_20261107/. DATASETS.md contains access and licence information.
'''
    head_rows='\n'.join(f"| {'SAM512' if r['phase']=='head_refinement_sam512' else 'Pruned256'} | {r['epoch']} | {r['val_accuracy']:.4f} | {r['val_macro_precision']:.4f} | {r['val_macro_f1']:.4f} | {r['val_disgust_precision']:.4f} | {r['val_disgust_recall']:.4f} | {r['val_disgust_f1']:.4f} |" for r in head_history)
    text=text.replace('## Validation comparison', '## Complete classifier-only validation log\n\n| Recipe | Epoch | Accuracy | Macro precision | Macro F1 | Disgust precision | Disgust recall | Disgust F1 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'+head_rows+'\n\n## Validation comparison')
    long_lines=[]
    for i,r in enumerate(long_history,1):
        matching=[]
        for candidate in r['candidates']:
            candidate_report=read(LONG/f"validation_{i:02d}_{candidate['kind']}.json")
            if candidate_report['accuracy']==r['val_accuracy'] and candidate_report['classification_report']['macro avg']['f1-score']==r['val_macro_f1'] and candidate_report['negative_log_likelihood']==r['val_loss']:
                matching.append(candidate_report)
        if not matching:raise ValueError('Cannot identify exact logged validation report')
        if len(matching)>1 and any(v['classification_report']!=matching[0]['classification_report'] for v in matching):raise ValueError('Ambiguous report; do not invent precision')
        report=matching[0]
        precision=report['classification_report']['macro avg']['precision']
        long_lines.append(f"| {r['epoch']} | {r['val_accuracy']:.4f} | {precision:.4f} | {r['val_macro_f1']:.4f} | {r['val_disgust_recall']:.4f} | {r['val_disgust_f1']:.4f} |")
    long_rows='\n'.join(long_lines)
    text=text.replace('## Validation comparison', '## Complete conservative full-network validation log\n\n| Epoch | Accuracy | Macro precision | Macro F1 | Disgust recall | Disgust F1 |\n|---|---:|---:|---:|---:|---:|\n'+long_rows+'\n\n## Validation comparison')
    text=text.replace('--threads4','--threads 4').replace('--run-dir reports/deep_supervision_20261107','--run-dir reports/balanced_selected_20261107');(ROOT/'DEEP_SUPERVISION_RESULTS.md').write_text(text,encoding='utf-8');(ROOT/'BALANCED_RESULTS.md').write_text(text,encoding='utf-8')
    intro=f'''## Latest completed deep-supervision experiment — {datetime.now().date()}

Active model: test **{100*active['accuracy']:.2f}%**, macro precision **{m['precision']:.4f}**, recall **{m['recall']:.4f}**, F1 **{m['f1-score']:.4f}**, disgust precision **{d['precision']:.4f}**, recall **{d['recall']:.4f}**, F1 **{d['f1-score']:.4f}**. Validation **{100*val['accuracy']:.2f}%**. Trials completed {len(epochs)+len(long_history)}full-network and {len(head_history)}classifier-only epochs; qualifying validation replacement: {qualified}; deployed: {deployed}. All actual gains/regressions and no-test outcomes: [BALANCED_RESULTS.md](BALANCED_RESULTS.md). No80% or reliable webcam/physiology/raga/weather efficacy is claimed. Earlier numerical sections are historical.

'''
    for name in ['EXPERIMENTS.md','MODEL_CARDS.md','PAPER_IMPLEMENTATION_MAPPING.md','FURTHER_RESULTS.md']:
        p=ROOT/name;backup=RUN/'document_backups'/name;backup.parent.mkdir(exist_ok=True)
        if not backup.exists():shutil.copy2(p,backup)
        p.write_text(intro+p.read_text(encoding='utf-8'),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for rows,label in [(epochs,'Auxiliary full-network'),(pruned_history,'Pruned256 classifier only'),(sam_history,'SAM512 classifier only'),(long_history,'Conservative full-network')]:
        for ax,key,base_value,title in zip(axes,['val_accuracy','val_macro_f1','val_disgust_f1'],[initial['accuracy'],initial['classification_report']['macro avg']['f1-score'],initial['classification_report']['disgust']['f1-score']],['Validation accuracy','Validation macro F1','Validation disgust F1']):
            ax.plot(range(len(rows)+1),[base_value]+[r[key] for r in rows],marker='.',label=label)
            ax.set_title(title);ax.set_xlabel('New epoch within recipe');ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle('All validation trials; classifier-only and full-network epoch compute differ')
    fig.tight_layout();fig.savefig(ROOT/'reports/balanced_validation_trials.png',dpi=160);plt.close(fig)
    (ROOT/'PROJECT_STATUS.md').write_text('# Current project status\n\n'+intro+'Training finished. Final app/API, deck and GitHub verification are being completed.\n',encoding='utf-8')
    print(json.dumps({'epochs':len(epochs),'qualified':qualified,'deployed':deployed,'best_observed':metric_vector(observed),'test_comparison':test},indent=2))
if __name__=='__main__':main()
