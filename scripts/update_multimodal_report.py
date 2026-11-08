"""Keep implemented context behavior separate from unmeasured efficacy."""
import json,re
from datetime import datetime
from xml.etree import ElementTree as ET
from backend.catalog import ROOT

def read(name):return json.loads((ROOT/'reports'/name).read_text())
def main():
    evaluation=read('emotion_evaluation.json');selective=read('selective_face_evaluation.json');policy=read('expression_policy.json');experiment=read('multimodal_extension_evaluation.json')
    assert selective['checkpoint_sha256']==policy['checkpoint_sha256']==evaluation['checkpoint_sha256']
    suite=ET.parse(ROOT/'reports/pytest.xml').getroot().find('testsuite');tests=int(suite.attrib['tests']);failures=int(suite.attrib['failures'])+int(suite.attrib['errors'])
    rows='\n'.join(f"| {name.replace('_',' ')} | {data['effective_goal']} | {data['ragas'][0]['name']} |" for name,data in experiment['scenarios'].items())
    text=f'''# Multimodal implementation status

Updated {datetime.now():%d %B %Y %H:%M}, Asia/Kolkata.

## Facial emotion remains weak

The full seven-class FER classifier remains **{100*evaluation['accuracy']:.2f}% test accuracy**, macro F1 {evaluation['classification_report']['macro avg']['f1-score']:.4f}. Validation accuracy at selected epoch 16 is 50.88%. These are different measurements; neither validates reliable seven-class recognition. Disgust test F1 is {evaluation['classification_report']['disgust']['f1-score']:.4f}, recall {100*evaluation['classification_report']['disgust']['recall']:.2f}%. No claim that disgust recognition is fixed is made.

The listener no longer interprets a neutral face output as feeling “At ease.” No mood is preselected. Weak or uncertain expression outputs are withheld, and the user chooses a feeling, including “Not sure.” Every offered expression still needs confirmation. Policy is based on class F1 >=0.6 and recall >=0.6 on the fixed **validation** split, confidence >=0.7 and top-two margin >=0.2. Eligible classes for this checkpoint: {', '.join(policy['eligible_classes'])}.

Actual test use of this policy: {selective['offered_suggestions']:,}/{selective['test_n']:,} images offered a suggestion (**{100*selective['coverage']:.2f}% coverage**), {selective['withheld_suggestions']:,} withheld. Accuracy on the offered subset is {100*selective['accuracy_among_offered_suggestions']:.2f}%; this is **not overall model accuracy**, covers only happy/surprise, and does not repair the other classes. Raw classifier outputs remain available for research. Sources: reports/expression_policy.json and selective_face_evaluation.json.

## What EEG / HRV now contribute

Previously they supplied offline features and a small absolute-threshold heuristic in the research workspace. The listener now accepts EEG and HRV recording JSON directly, displays theta/alpha/beta, RMSSD and SDNN, and includes processed features/provenance in recommendation context.

Joint physiology requires both recordings, the same declared session_id for real recordings, a resting reference, at least 10 seconds EEG and 60 seconds RR data, and a basic EEG amplitude screen. Matching session identifiers are user declarations, not proof of timing/subject synchronization. Do not combine unrelated DEAP and WESAD recordings. Ocular/muscle/ectopic-beat correction and validated signal-quality assessment remain incomplete.

The experimental rule compares EEG alpha/beta and RMSSD to the supplied reference. Both must be below 80% of their respective references before the automatic context mode suggests a calm **listening direction**. It never labels the person's mood or diagnoses stress. An explicit calm/energy goal overrides the automatic choice. A source-Shama preference can receive at most +0.03. These cutoffs and weights are project hypotheses, not established clinical or music-response relationships.

Missing references or one missing modality retain feature summaries without changing the goal. Simulated inputs are excluded unless the simulated demonstration is explicitly enabled, and cannot be silently mixed with real inputs. Personal baselines are never fabricated. Demo reference values are explicitly synthetic.

## Weather now contributes

Manual weather works offline. Optional city lookup calls the fixed [Open-Meteo geocoding](https://open-meteo.com/en/docs/geocoding-api) and [forecast](https://open-meteo.com/en/docs) endpoints; it does not request device location. Lookup failure leaves manual input available. A live backend lookup was verified with Pune as a public example, not the user's location (reports/live_weather_check.json).

An experimental weather-inspired policy gives up to +0.03 to matching source Ritu tags: rain/Varsha, heat/Grishma, cold/Hemanta or Shishira. Weather is not geographical season, and this is not a verified weather-to-raga benefit. Source metadata remains independently unverified. City weather is model-based, not an on-device sensor measurement. Open-Meteo attribution is displayed; free API use is non-commercial with CC BY 4.0 data under its [terms](https://open-meteo.com/en/terms).

## Actual functional ablations

All weather here is manual test input; all physiology and references are **synthetic**. These are executed software tests, not participant outcomes.

| Inputs | Effective listening goal | First raga |
|---|---|---|
{rows}

Source: reports/multimodal_extension_evaluation.json. Ranking scores and, for some scenarios, order changed as specified. This verifies integration and missing-modality behavior. **No multimodal recognition accuracy, recommendation-quality gain or mood-improvement result has been measured.** Weather/body signals do not improve the reported 50.13% classifier accuracy. A learned multimodal model requires consented, aligned facial/weather/physiology/listening-response data and independent evaluation.

## Verification and reproduction

{tests-failures}/{tests} automated tests passed ({failures} failures/errors). Production React build passed. Tests cover weather parsing, contribution bounds, explicit sample opt-in, joint references, mismatched sessions, missing modalities, user-goal priority and refusal to turn confident-but-unsupported neutral predictions into a mood.

Commands from project root:

```powershell
.\\.venv\\Scripts\\python.exe -m scripts.build_expression_policy
.\\.venv\\Scripts\\python.exe -m scripts.evaluate_expression_policy
.\\.venv\\Scripts\\python.exe -m scripts.multimodal_experiments
.\\.venv\\Scripts\\python.exe -m pytest -q tests --junitxml=reports/pytest.xml
```

Example recording fields: EEG samples (channels × samples, volts), sample_rate, sample_data=false, session_id; HRV rr_ms (milliseconds) or ECG/PPG waveform plus sample_rate/kind, sample_data=false, same session_id. Supply your own measured baseline alpha_beta_ratio and rmssd_ms. Numerical examples in sample mode are not personal reference values.
'''
    (ROOT/'MULTIMODAL_STATUS.md').write_text(text,encoding='utf-8')
    marker='## Multimodal correction — 3 November 2026'
    for filename in ['EXPERIMENTS.md','MODEL_CARDS.md','PAPER_IMPLEMENTATION_MAPPING.md','IKS_METHODOLOGY.md','PROJECT_STATUS.md']:
        path=ROOT/filename;old=path.read_text(encoding='utf-8').split(marker)[0].rstrip()
        old=re.sub(r'\*\*\d+/\d+ tests pass\*\*, \d+ failures and \d+ errors; [\d.]+ seconds',f"**{tests-failures}/{tests} tests pass**, {suite.attrib['failures']} failures and {suite.attrib['errors']} errors; {float(suite.attrib['time']):.2f} seconds",old)
        old=re.sub(r'Verification: \d+/\d+ automated tests passed; \d+ failures, \d+ errors',f"Verification: {tests-failures}/{tests} automated tests passed; {suite.attrib['failures']} failures, {suite.attrib['errors']} errors",old)
        block=f"\n\n{marker}\n\nThe full facial classifier remains 50.13% accurate on the retained test set; disgust recognition is weak. Conservative validation-based abstention replaces unsupported mood suggestions. Weather, EEG/HRV recordings, reference safeguards, listening goals and visible source contributions now reach the listener's recommendation path. This is experimental rule-based context integration, **not a validated improvement in physiological emotion recognition, recommendation quality or health**. Actual synthetic ablations and limitations are documented in [MULTIMODAL_STATUS.md](MULTIMODAL_STATUS.md). Current tests: {tests-failures}/{tests} passed.\n"
        path.write_text(old+block,encoding='utf-8')
    summary=read('verification_summary.json');summary.update(tests=tests,passed=tests-failures,failures=int(suite.attrib['failures']),errors=int(suite.attrib['errors']),test_seconds=float(suite.attrib['time']),context_evaluation='Synthetic functional ablations only; no quality/efficacy result')
    (ROOT/'reports/verification_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'tests':tests,'failures':failures,'report':'MULTIMODAL_STATUS.md'},indent=2))

if __name__=='__main__':main()
