"""Measured further-training report. Run after evaluation and verified import."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime
import torch
from scripts.colab_common import write_json
ROOT=Path(__file__).resolve().parents[1]
def read(n):return json.loads((ROOT/n).read_text())
def main():
    e=read('reports/emotion_evaluation.json');b=read('reports/further_baseline/emotion_evaluation.json');m=e['classification_report']['macro avg'];d=e['classification_report']['disgust'];ck=torch.load(ROOT/'models/fusdense.pt',map_location='cpu',weights_only=True)
    sha=hashlib.sha256((ROOT/'models/fusdense.pt').read_bytes()).hexdigest();assert sha==e['checkpoint_sha256']
    comparison=read('reports/further_selected_run/deployment/reports/colab_comparison.json')
    assert comparison['checkpoint_sha256']==sha and comparison['recommended_for_integration']
    for key in ['precision','recall']:
        before=b['classification_report']['macro avg'][key];after=m[key];comparison['metrics']['macro_'+key]={'baseline':before,'colab':after,'difference':after-before}
    write_json(ROOT/'reports/emotion_training_comparison.json',comparison)
    history=read('reports/training_history.json');parameters=sum(p.numel() for p in __import__('scripts.colab_common',fromlist=['model_from']).model_from(ck,torch.device('cpu')).parameters())
    summary={'accuracy':e['accuracy'],'macro_precision':m['precision'],'macro_recall':m['recall'],'macro_f1':m['f1-score'],'disgust':d,'test_n':e['n'],'checkpoint_sha256':sha,'parameters':parameters,'architecture':ck['architecture'],'selected_epoch':e['checkpoint_epoch'],'train_count':ck['train_count'],'val_count':ck['val_count'],'selected_lineage_full_epochs':sum(not r.get('phase') for r in history),'selected_lineage_head_epochs':sum(bool(r.get('phase')) for r in history)}
    write_json(ROOT/'reports/further_results_summary.json',summary)
    rows='\n'.join(f"| {name} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1-score']:.4f} | {int(c['support'])} |" for name,c in e['classification_report'].items() if name in ck['classes'])
    table='\n'.join(f"| {name.replace('_',' ')} | {v['baseline']:.6f} | {v['colab']:.6f} | {v['difference']:+.6f} |" for name,v in comparison['metrics'].items())
    experiments=[]
    for name in ['further_fusdense_run','frozen_bn_fusdense_run','spatial_head_run','spatial_flip_head_run','spatial_disgust_head_run','spatial_full_run']:
        c=read('reports/'+name+'/completion.json');experiments.append(f"| {name} | {c['completed_additional_epochs']} | {c['best_macro_f1']:.6f} | {'MLP only' if 'head' in name else 'Full network'} |")
    g=read('reports/gamer_fixed_context_evaluation.json');phys='\n'.join(f"| {name} | {g['tasks'][name]['n']} / {g['tasks'][name]['participants']} | {g['tasks'][name]['results']['EEG']['macro_f1']:.4f} | {g['tasks'][name]['results']['Cardiac']['macro_f1']:.4f} | {g['tasks'][name]['results']['Fusion']['macro_f1']:.4f} | {g['tasks'][name]['results']['Majority']['accuracy']:.4f} |" for name in ['QWOP','CBAAD'])
    text=f'''# Further actual results: FusDenseNet and controlled physiology

Generated {datetime.now().isoformat(timespec='seconds')}. This document supersedes earlier active-model summaries. Original paper results, historical experiments and simulated signal fixtures remain separate.

## Facial model: unfiltered same-set evaluation

All {e['n']:,} retained test faces and all seven classes are included, including wrong predictions. This is a historically inspected combined Kaggle test folder, not a fresh, identity-disjoint or PrivateTest-only evaluation. Fixed training {ck['train_count']:,}, validation {ck['val_count']:,}; exact pixel duplicates excluded. Repeated validation selection can overfit, particularly its 50 disgust examples.

| Metric | Previous 52.55% checkpoint | New checkpoint | Difference |
|---|---:|---:|---:|
{table}

| Expression | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{rows}

Deployed SHA-256 `{sha}`. Architecture version {ck['architecture']['version']}, {parameters:,} parameters. FusDenseNet retains all three fusion-dense stages, cross-attention, cross-dense links, dilation and global/spatial pooling. The disclosed project v3 adds 768 normalized 4×4 stem mean/max spatial features before the 8×8 bottleneck, concatenated with the original 588 features, followed by 1,356→128→7 classification. This adds 98,304 weights and is not an exact recovered author architecture or controlled equal-budget ablation. Initialization preserved v2 logits before training (see spatial_variant_initialization.json).

MLP refinement uses only fixed training features. Its feature normalization is folded into the existing first Linear layer. Training and validation phases share the same fixed pixel manifests. Selected index {e['checkpoint_epoch']} is an index across phases, not that many full-network epochs. The recorded source-run history contains {summary['selected_lineage_full_epochs']} full-network and {summary['selected_lineage_head_epochs']} MLP epochs, including historical epochs and later rejected checkpoints. These counts are logs, not the number of updates in the selected weights. Other experiments are separate, not added to the selected model's training history. Their completed new work is:

| Experiment | Additional completed epochs | Best validation macro F1 (including start) | Scope |
|---|---:|---:|---|
{chr(10).join(experiments)}

Partial interrupted epochs are excluded. Weighted sampling, frozen BN, flip features and stronger rare-class weighting are recorded even when rejected. Selection uses validation accuracy/macro F1/disgust F1 floors and .75 macro F1 + .25 disgust F1, before any new test access. Deployment requires improved unfiltered test accuracy, macro F1 and disgust F1, without lower disgust recall. This descriptive gate does not establish significance. Temperature fitting and flip-TTA selection use validation only, and do not exclude difficult faces. Strong disgust or webcam reliability remains unestablished. No approval was needed for authorized training.

## Real EEG/ECG fusion with game identity held constant

Official GamER CC BY 4.0 recordings, DOI https://doi.org/10.5281/zenodo.8191877. Separate fixed-game tasks, centered exactly 30-second labelled windows, leave-one-participant-out folds, fixed balanced logistic regression C=1 and fold-fitted StandardScaler. EEG theta/alpha/beta/delta/gamma relative powers, entropy and Hjorth features; real ECG-derived RR mean, RMSSD, SDNN, pNN50, median and IQR. EEG and ECG share the source interval. No simulated signal contributes to these results.

| Constant game | Windows / people | EEG macro F1 | Cardiac macro F1 | Fusion macro F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
{phys}

**Neither primary task shows a fusion macro-F1 gain.** Fusion accuracy is below majority accuracy. Participant-bootstrap paired intervals include zero. Game identity and window duration are constant within each task, so neither cue distinguishes labels. This controls those two particular confounds, not every artifact, movement or annotation error. Primary offset subtraction is fixed before evaluation; alternate addition is a sensitivity analysis, never selected for a better score. Exact raw-window recomputation, source annotation IDs, fold isolation, constant controls and independently recomputed confusion matrices passed `reports/gamer_fixed_context_audit.json`.

The older pooled Angry/Sad result is retained as historical and confounded (game identity alone classified all labels). It is not evidence of a physiology-specific emotion improvement. Thirty-second RR statistics are exploratory, not conventional five-minute clinical HRV. Longer-window within-game cohorts are insufficient. No facial/EEG alignment, disgust physiology, real raga responses, weather benefit or therapeutic efficacy is demonstrated.

## Independent game-free dataset attempt

DENS https://doi.org/10.18112/openneuro.ds003751.v1.0.2 and https://github.com/OpenNeuroDatasets/ds003751 contain EEG/ECG and movie-related self-ratings. A protocol fixed the first 12 released participants, 60-second windows and crossed participant/movie held-out folds. Only 2 official headers matched their annex-referenced float32 signal dimensions. The other 10 were excluded before signal interpretation. This cohort cannot support the planned benchmark, so **no DENS accuracy or fusion result is claimed**. Headers were fully MD5 checked; no partial signal is misrepresented as full-FDT checksum verified. See dens_download_plan.json, dens_access_outcome.json and dens_protocol.json. The modular evaluator is implemented but remains unvalidated on an adequate DENS cohort.

Dataset metadata says CC0 while its README/LICENSE specifies ODbL/DbCL. Preserve the source licence, attribution and more protective conditions pending clarification. No full signals or third-party assets are bundled into the PPT. No replacement subjects were chosen after observing scores.

## Reproduction

```powershell
.\\.venv\\Scripts\\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/further_selected_run --device cpu --threads 2 --baseline-report reports/further_baseline/emotion_evaluation.json --baseline-manifest reports/further_baseline/emotion_evaluation_manifest.json
.\\.venv\\Scripts\\python.exe -m scripts.gamer_fixed_context
.\\.venv\\Scripts\\python.exe -m scripts.verify_fixed_context
.\\.venv\\Scripts\\python.exe -m pytest -q
```

The frozen FER export is hash-verified and reused on repeat evaluation, preventing training/tuning of that run after test access. Exact metrics, confusion matrices and complete saved predictions remain in JSON. See DATASETS.md for access and licence instructions. The installed model can still misclassify a user's face; raw predictions and all class probabilities remain visible.
'''
    (ROOT/'FURTHER_RESULTS.md').write_text(text,encoding='utf-8')
    intro=f'''## Latest measured update: further training and controlled physiology

Active FusDenseNet v{ck['architecture']['version']}: accuracy **{100*e['accuracy']:.2f}%**, macro F1 **{m['f1-score']:.4f}**, disgust recall **{100*d['recall']:.2f}%**, disgust F1 **{d['f1-score']:.4f}** on all {e['n']:,} retained test faces. Previous baseline was 52.55%. This remains a limited historically inspected facial benchmark, not reliable webcam emotion recognition.

Game-constant, fixed-30s real EEG/ECG evaluation shows **no fusion F1 gain**. DENS participant/movie-held-out evaluation is incomplete because only 2/12 source headers match referenced signal dimensions. No DENS performance is invented. Full current results, regressions and commands: [FURTHER_RESULTS.md](FURTHER_RESULTS.md). Earlier numerical sections below are historical and may refer to previous checkpoints or confounded pooled tasks.

'''
    for name in ['MODEL_CARDS.md','EXPERIMENTS.md','PAPER_IMPLEMENTATION_MAPPING.md','PROJECT_STATUS.md','REAL_DATA_RESULTS.md']:
        p=ROOT/name;backup=ROOT/'reports/further_document_backups'/name;backup.parent.mkdir(exist_ok=True)
        if p.exists():
            if not backup.exists():shutil.copy2(p,backup)
            p.write_text(intro+p.read_text(encoding='utf-8'),encoding='utf-8')
    p=ROOT/'DATASETS.md';p.write_text(p.read_text(encoding='utf-8')+'\n\n## Further controlled real physiology and DENS access\n\n'+text[text.index('## Real EEG/ECG'):text.index('## Reproduction')],encoding='utf-8')
    p=ROOT/'LITERATURE_REVIEW.md';p.write_text(p.read_text(encoding='utf-8')+'\n\nDENS primary dataset: Mishra, S., Asif, M., Tiwary, U. S. and Srinivasan, N. Dataset on Emotion with Naturalistic Stimuli. OpenNeuro ds003751 v1.0.2. https://doi.org/10.18112/openneuro.ds003751.v1.0.2 . Source metadata and README inspected; conflicting licensing and signal/header mismatches are recorded, with no evaluation score claimed.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
