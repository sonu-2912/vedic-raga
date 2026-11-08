<!-- actual-real-data-update -->
## Latest measured update — 3 November 2026

Active classifier: **52.55% unfiltered test accuracy**, macro F1 **0.4492**, disgust recall **12.66%**. New experiments, phase counts, all-class metrics, actual real-recording GamER fusion and its regressions/confounds are documented in [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md). Older figures below describe earlier checkpoints and stages. No result here establishes reliable webcam emotion, facial-plus-physiological fusion or raga/weather benefit. Current automated verification: 37/37 tests passed.

<!-- actual-real-data-update -->
# Multimodal implementation status

Updated 03 November 2026 14:35, Asia/Kolkata.

## Facial emotion remains weak

The full seven-class FER classifier remains **50.13% test accuracy**, macro F1 0.3960. Validation accuracy at selected epoch 16 is 50.88%. These are different measurements; neither validates reliable seven-class recognition. Disgust test F1 is 0.0488, recall 2.53%. No claim that disgust recognition is fixed is made.

The listener no longer interprets a neutral face output as feeling “At ease.” No mood is preselected. Weak or uncertain expression outputs are withheld, and the user chooses a feeling, including “Not sure.” Every offered expression still needs confirmation. Policy is based on class F1 >=0.6 and recall >=0.6 on the fixed **validation** split, confidence >=0.7 and top-two margin >=0.2. Eligible classes for this checkpoint: happy, surprise.

Actual test use of this policy: 1,464/6,561 images offered a suggestion (**22.31% coverage**), 5,097 withheld. Accuracy on the offered subset is 88.39%; this is **not overall model accuracy**, covers only happy/surprise, and does not repair the other classes. Raw classifier outputs remain available for research. Sources: reports/expression_policy.json and selective_face_evaluation.json.

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
| confirmed mood only | balance | Jait |
| weather only | balance | Malshree |
| body without reference | balance | Jait |
| body without hrv | balance | Jait |
| body with reference | calm | Chandrakauns |
| all contexts | calm | Chandrakauns |

Source: reports/multimodal_extension_evaluation.json. Ranking scores and, for some scenarios, order changed as specified. This verifies integration and missing-modality behavior. **No multimodal recognition accuracy, recommendation-quality gain or mood-improvement result has been measured.** Weather/body signals do not improve the reported 50.13% classifier accuracy. A learned multimodal model requires consented, aligned facial/weather/physiology/listening-response data and independent evaluation.

## Verification and reproduction

29/29 automated tests passed (0 failures/errors). Production React build passed. Tests cover weather parsing, contribution bounds, explicit sample opt-in, joint references, mismatched sessions, missing modalities, user-goal priority and refusal to turn confident-but-unsupported neutral predictions into a mood.

Commands from project root:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_expression_policy
.\.venv\Scripts\python.exe -m scripts.evaluate_expression_policy
.\.venv\Scripts\python.exe -m scripts.multimodal_experiments
.\.venv\Scripts\python.exe -m pytest -q tests --junitxml=reports/pytest.xml
```

Example recording fields: EEG samples (channels × samples, volts), sample_rate, sample_data=false, session_id; HRV rr_ms (milliseconds) or ECG/PPG waveform plus sample_rate/kind, sample_data=false, same session_id. Supply your own measured baseline alpha_beta_ratio and rmssd_ms. Numerical examples in sample mode are not personal reference values.

## Updated presentation and interface verification

Current deck: `output/RagaMoodSync_Research_Multimodal.pptx` (also copied to `output/RagaMoodSync_Research.pptx`). Exactly 13 editable slides, all rendered slides visually reviewed; package/layout/import/native-chart checks passed with zero findings. Previous FER-only deck is preserved as `output/RagaMoodSync_Research_Updated.pptx`. Native PowerPoint execution is not claimed. Desktop context explanations and 390px mobile layout were checked with isolated test data; no production feedback was written.
