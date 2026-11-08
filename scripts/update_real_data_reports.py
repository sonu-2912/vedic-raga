"""Publish actual unfiltered FER and real paired-physiology experiment evidence."""
import hashlib,json
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
def read(relative):return json.loads((ROOT/relative).read_text(encoding='utf-8'))
def write(relative,value):
    path=ROOT/relative;path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def metric_table(report):
    return '\n'.join(f"| {name} | {100*m['accuracy']:.2f}% | {m['classification_report']['macro avg']['precision']:.4f} | {m['classification_report']['macro avg']['recall']:.4f} | {m['macro_f1']:.4f} |" for name,m in report['results'].items())
def main():
    selected=read('reports/selected_improvement_run/selection.json')
    result=read('reports/selected_improvement_run/deployment/reports/emotion_evaluation.json')
    comparison=read('reports/emotion_training_comparison.json')
    validation=read('reports/selected_improvement_run/deployment/reports/validation_evaluation.json')
    gamer=read('reports/gamer_evaluation.json');alternate=read('reports/gamer_evaluation_plus_offset.json')
    active=read('reports/emotion_evaluation.json');active_sha=hashlib.sha256((ROOT/'models/fusdense.pt').read_bytes()).hexdigest()
    assert active_sha==active['checkpoint_sha256']
    deployed=result['checkpoint_sha256']==active_sha
    suites=ET.parse(ROOT/'reports/pytest.xml').getroot()
    counts={key:sum(int(s.get(key,0)) for s in suites.findall('.//testsuite')) for key in ['tests','failures','errors']}
    metrics=result['classification_report'];macro=metrics['macro avg'];changes=comparison['metrics']
    text=f'''# Actual model improvement and real physiological fusion

Updated {datetime.now().strftime('%d %B %Y %H:%M')}, Asia/Kolkata.

## Unfiltered facial model results

All **{result['n']:,} retained FER2013 test images and all seven classes** were scored. No abstention, thresholding or removal of wrong predictions is used to calculate these results. The selected checkpoint is deployed: **{deployed}**. Active application accuracy is {100*active['accuracy']:.2f}%; the candidate results below are kept separate if deployment failed.

| Metric | Previous checkpoint | Selected candidate | Difference |
|---|---:|---:|---:|
'''
    for name,m in changes.items():
        percent=name in ['accuracy','disgust_recall'];fmt=lambda value:f'{100*value:.2f}%' if percent else f'{value:.4f}'
        text+=f"| {name.replace('_',' ')} | {fmt(m['baseline'])} | {fmt(m['colab'])} | {fmt(m['difference'])} |\n"
    text+=f'''
Candidate macro precision {macro['precision']:.4f}; macro recall {macro['recall']:.4f}. Selected checkpoint validation accuracy {100*validation['accuracy']:.2f}%, macro F1 {validation['classification_report']['macro avg']['f1-score']:.4f}. Validation and test are different measurements.

| Test class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
'''
    for name in ['angry','disgust','fear','happy','sad','surprise','neutral']:
        m=metrics[name];text+=f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support'])} |\n"
    full=read('reports/real_improvement_run/completion.json');heads=[read(f'reports/{name}/completion.json') for name in ['head_refinement_run','head_refinement_continued_run']]
    text+=f'''
### What was trained

FusDenseNet remains the existing paper-informed 223,776-parameter architecture, with three fusion-dense stages, cross-attention and spatial/global pooling. No unrelated network is relabelled as FusDenseNet. Windows **CPU**, not CUDA/Colab hardware, executed all new work.

Full-network continuation completed {full['completed_additional_epochs']} additional epochs using AdamW LR 0.0003, class-weight power 0.85, label smoothing 0.02, reconstruction weight 0.05 and fixed data augmentation. It warm-started epoch-16 weights, with explicit optimizer restart. Head-only experiments completed {heads[0]['completed_additional_epochs']} and {heads[1]['completed_additional_epochs']} epochs. They train the existing MLP on frozen full-training-split features with training-fitted normalization. The affine normalization is folded into its existing first Linear layer; dimensions and parameter count do not change. No external pretrained emotion weights are substituted. Head epochs are much cheaper than full-network epochs and must not be presented as equal compute.

Selected experiment: **{selected['chosen']['kind']}**, source `{selected['chosen']['path']}`, epoch index {selected['chosen']['epoch']}. Checkpoint selection maximizes validation macro F1 among candidates at least matching the initial validation accuracy. The selected candidate is tested once after selection. Deployment additionally requires higher test accuracy and macro F1 without lower disgust recall, as an engineering gate; it is not statistical evidence. Validation reuse across recipes can overfit that split. The combined FER test folder was historically inspected; it is not a newly acquired or identity-disjoint benchmark. Webcam generalization remains unmeasured. Better aggregate scores do not establish reliable disgust recognition.

Checkpoint SHA-256: `{result['checkpoint_sha256']}`. Exact unchanged train/validation pixel manifests and same-test-record comparison are in `reports/selected_improvement_run`. Earlier checkpoints/reports are preserved by the importer under `reports/colab_import_backups` if deployed. The reusable scripts retain `colab` names for compatibility; no GPU execution is implied.

## Real physiological fusion: GamER

Source: [Papadopoulos et al., GamER v1](https://doi.org/10.5281/zenodo.8191877), CC BY 4.0, official Zenodo release. The 919,686,572-byte ZIP and both source CSV files were acquired from the publisher and verified against its MD5 checksums after checking access, licence, size and free disk space. Source EEG and ECG are sampled together in each HDF5 recording. No synthetic arrays or fabricated participant responses enter this evaluation.

Whole annotated intervals of at least 15 seconds are used; both signals must pass quality gates. {gamer['accepted_intervals']} accepted intervals overall; {gamer['exclusion_count']} exclusions/invalid annotations are listed in `reports/gamer_features.json`. Each ablation uses exactly the same accepted cohort. Source labels are a psychologist's annotations based on self-reports and observation, not direct measurements of internal mood. Features: relative EEG theta/alpha/beta/delta/gamma, spectral entropy and Hjorth measures; ECG-derived mean/median RR, RMSSD, SDNN, pNN50 and RR IQR. Many intervals are ultra-short and do not support conventional clinical HRV interpretation. No frequency-domain HRV is claimed.

StandardScaler and fixed balanced LogisticRegression C=1 are fitted on training participants only. Leave-one-participant-out predictions are recorded for every retained interval. No participant occurs in both training and test within a fold. There is no hyperparameter search on outer-fold scores. An exploratory game-ID-only control was added before scoring to expose task/context confounding; it is not a physiological model.
'''
    for name,t in gamer['tasks'].items():
        other=alternate['tasks'][name]
        text+=f'''
### {name.replace('_',' ')}

{t['n_intervals']} real labelled intervals, {t['n_subjects']} participants; {t['ultrashort_intervals']} intervals shorter than 60 seconds. Class counts: {json.dumps(t['class_counts'])}.

| Method | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
{metric_table(t)}

Fusion minus EEG macro F1: **{t['fusion_minus_eeg_macro_f1']:+.4f}**. Approximate paired participant-bootstrap 95% interval: **[{t['subject_bootstrap_95pct_delta_interval'][0]:+.4f}, {t['subject_bootstrap_95pct_delta_interval'][1]:+.4f}]**. Bootstrap resamples participants in existing outer-fold predictions, not refitted studies; it is not multiplicity-adjusted significance evidence.

Offset-sign sensitivity (addition instead of subtraction), independently processed: {other['n_intervals']} intervals / {other['n_subjects']} participants. EEG F1 {other['results']['EEG']['macro_f1']:.4f}; cardiac F1 {other['results']['Cardiac']['macro_f1']:.4f}; fusion F1 {other['results']['Fusion']['macro_f1']:.4f}; game-only F1 {other['results']['Game ID control']['macro_f1']:.4f}. The favorable convention is not selected after scoring. Different accepted boundary/quality cohorts mean the two signs are not a controlled paired comparison.
'''
    text+='''
### Interpretation of the measured results

The primary anger-versus-sadness fusion score is numerically higher than EEG alone, but the participant-bootstrap interval for the F1 difference includes zero. Game ID alone classifies the two labels perfectly in this accepted cohort, demonstrating complete task/label confounding. This experiment cannot establish emotion-specific physiology. In the five-class task, fusion accuracy is below the majority baseline even though macro F1 is higher. These weaknesses and the macro-precision regression in the facial model must not be hidden behind the favorable accuracy or F1 figures. No learned physiology classifier is deployed as a validated mood detector.

### Limits of this physiological evidence

GamER does not contain disgust labels, FER2013 faces, local weather or Hindustani raga responses. Game identity is correlated with source emotion labels; features may encode task, movement or recording artifacts. A game-only score can be high for this reason. A positive fusion difference on these tasks does not establish emotion-specific physiological generalization, a facial-fusion improvement or mood benefit. No classifier trained on these source labels silently replaces the listener's seven FER classes. EEG/HRV files can still add explicit context in the listener; weather uses disclosed experimental Ritu preferences. No clinical or weather-outcome validation has been performed.

Video annotation time minus VIDEO_OFFSET is an explicit alignment assumption. The release does not independently confirm that sign and the original videos are absent. Both sign conventions are published; alignment needs author confirmation before stronger claims. Single-lead EEG does not support ICA, and there is no independent manual artifact adjudication or clinical beat annotation reference. The demonstrated result is a reproducible, limited real-recording benchmark, not a validated physiological emotion product. DREAMER and Emognition access restrictions were respected; no gated mirrors were used.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe -m scripts.gamer_experiment
.\\.venv\\Scripts\\python.exe -m scripts.gamer_experiment --sensitivity
.\\.venv\\Scripts\\python.exe -m scripts.colab_evaluate --data data/private/fer2013-v1.zip --run-dir reports/selected_improvement_run --device cpu --threads 2
.\\.venv\\Scripts\\python.exe -m pytest -q tests --junitxml=reports/pytest.xml
```

The selected FER run is locked after evaluation. Repeating its evaluation command reuses checksum-verified recorded outputs rather than repeatedly reading test labels. Real physiology evaluation reads the verified private source ZIP and source CSVs; no credentials are required for the open GamER release. Generated reports retain all accepted interval predictions and participant-disjoint fold manifests for audit, locally.
'''
    text+=f"\nVerification: {counts['tests']} tests, {counts['failures']} failures, {counts['errors']} errors. Production frontend build is checked separately.\n"
    (ROOT/'REAL_DATA_RESULTS.md').write_text(text,encoding='utf-8')
    summary=read('reports/verification_summary.json');summary.update(counts);summary['passed']=counts['tests']-counts['failures']-counts['errors'];summary['updated']=datetime.now().isoformat();summary['test_seconds']=sum(float(s.get('time',0)) for s in suites.findall('.//testsuite'))
    am=active['classification_report']['macro avg'];summary['real_fer2013_metrics'].update(n=active['n'],accuracy=active['accuracy'],macro_precision=am['precision'],macro_recall=am['recall'],macro_f1=am['f1-score'])
    if deployed:
        history=read('reports/training_history.json');summary['training'].update(epochs_completed=len(history),selected_epoch=active['checkpoint_epoch'])
    summary['context_evaluation']='Real GamER participant-held-out EEG/cardiac fusion benchmark, plus offset sensitivity and game-context control; no facial-fusion/raga/weather efficacy result'
    write('reports/verification_summary.json',summary)
    for relative in ['README.md','EXPERIMENTS.md','MODEL_CARDS.md','MULTIMODAL_STATUS.md','PAPER_IMPLEMENTATION_MAPPING.md','PROJECT_STATUS.md']:
        path=ROOT/relative;old=path.read_text(encoding='utf-8');marker='<!-- actual-real-data-update -->'
        if marker in old:old=old.split(marker,2)[-1].lstrip()
        banner=f'''{marker}
## Latest measured update — 3 November 2026

Active classifier: **{100*active['accuracy']:.2f}% unfiltered test accuracy**, macro F1 **{am['f1-score']:.4f}**, disgust recall **{100*active['classification_report']['disgust']['recall']:.2f}%**. New experiments, phase counts, all-class metrics, actual real-recording GamER fusion and its regressions/confounds are documented in [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md). Older figures below describe earlier checkpoints and stages. No result here establishes reliable webcam emotion, facial-plus-physiological fusion or raga/weather benefit. Current automated verification: {summary['passed']}/{summary['tests']} tests passed.

{marker}
'''
        path.write_text(banner+old,encoding='utf-8')
    print(json.dumps({'deployed':deployed,'accuracy':active['accuracy'],'macro_f1':am['f1-score'],'tests':summary['passed']}))
if __name__=='__main__':main()
