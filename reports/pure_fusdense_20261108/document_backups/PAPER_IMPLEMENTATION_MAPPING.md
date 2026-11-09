Audit-driven experiments completed6full-network and58separate residual-head epochs. Active version7; v6/v7are explicitly FusDenseNet+ImageNet DenseNet hybrid variants, v7uses probability combination. Qualifying validation replacement:True; deployed:True. Active test accuracy59.26%, macroF10.5530, disgust recall43.04%. Full measured audit and regressions: SPATIAL_GENERALIZATION_RESULTS.md. No80%, fresh-test or reliable-webcam result is claimed.

Earlier numerical sections below are historical.

## Latest completed deep-supervision experiment — 2026-11-08

Active model: test **57.42%**, macro precision **0.5392**, recall **0.5325**, F1 **0.5320**, disgust precision **0.4833**, recall **0.3671**, F1 **0.4173**. Validation **58.04%**. Trials completed 13full-network and 20classifier-only epochs; qualifying validation replacement: True; deployed: False. All actual gains/regressions and no-test outcomes: [BALANCED_RESULTS.md](BALANCED_RESULTS.md). No80% or reliable webcam/physiology/raga/weather efficacy is claimed. Earlier numerical sections are historical.

## Latest completed full-network experiment — 7 November 2026

Active model: **57.42% full-test accuracy**, macro precision **0.5392**, recall **0.5325**, F1 **0.5320**; disgust recall **36.71%**, F1 **0.4173**. Inference validation: **58.04%**. All 6,561 retained test faces and all seven classes remain included.

The new augmentation/distillation/averaged-weight experiment completed 5 full emotion-path epochs. Its candidate was deployed after passing the fixed integration gate. Results and regressions: [CONSISTENCY_RESULTS.md](CONSISTENCY_RESULTS.md). No 80% result is claimed. Validation is repeatedly reused and the combined FER test was historically inspected; this is not fresh webcam-domain evidence. Real fixed-game EEG/HRV fusion still shows no primary gain, and DREAMER official access is restricted. Earlier numerical sections below are historical.

## Current audited result — 7 November 2026

Validation **57.41%**, full unfiltered test **56.13%**, macro F1 **0.5105**. The target80% validation accuracy was not achieved. Active version4 keeps dense fusion/attention and adds an early dense spatial feature path. New test and class results,63 additional classifier-only training epochs, repairs and limitations are in [AUDIT_RESULTS.md](AUDIT_RESULTS.md). **41 tests passed.** Earlier numbered results below are historical.

## Latest measured update: further training and controlled physiology

Active FusDenseNet v3: accuracy **52.58%**, macro F1 **0.4658**, disgust recall **22.78%**, disgust F1 **0.2571** on all 6,561 retained test faces. Previous baseline was 52.55%. This remains a limited historically inspected facial benchmark, not reliable webcam emotion recognition.

Overall accuracy changed by only two test faces. Disgust precision fell from 0.3333 to 0.2951, while recall and F1 improved. Version 3 adds 768 normalized early spatial stem features to the original 588 fusion features, followed by a 1,356-to-128-to-7 MLP. The selected improvement uses classifier-only refinement; further full-network trials were rejected on validation.

Game-constant, fixed-30s real EEG/ECG evaluation shows **no fusion F1 gain**. DENS participant/movie-held-out evaluation is incomplete because only 2/12 source headers match referenced signal dimensions. No DENS performance is invented. Full current results, regressions and commands: [FURTHER_RESULTS.md](FURTHER_RESULTS.md). Earlier numerical sections below are historical and may refer to previous checkpoints or confounded pooled tasks.

<!-- actual-real-data-update -->
## Latest measured update — 3 November 2026

Active classifier: **52.55% unfiltered test accuracy**, macro F1 **0.4492**, disgust recall **12.66%**. New experiments, phase counts, all-class metrics, actual real-recording GamER fusion and its regressions/confounds are documented in [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md). Older figures below describe earlier checkpoints and stages. No result here establishes reliable webcam emotion, facial-plus-physiological fusion or raga/weather benefit. Current automated verification: 37/37 tests passed.

<!-- actual-real-data-update -->
# Paper-to-implementation mapping

Primary paper: [Pingle & Ragha 2025](https://doi.org/10.1007/s41870-025-02531-y). All newly selected dimensions and policies below are assumptions, not architecture details claimed from the paper.

| Paper component | Implementation | Fidelity / assumption |
|---|---|---|
| Preprocessing and geometric augmentation, §3.1 | `backend/datasets.py`, `backend/emotion.py` | 48x48 grayscale FER, flips, ±15° rotation, random resized crop 0.85–1.0. Vertical flip implemented behind flag because inverted faces may be unrealistic. Validation has no random augmentation. |
| Hybrid denoising, Eqs 1–2 | `denoise` | OpenCV NLM → bilateral → Gaussian in hybrid mode. Kernel sizes/strengths selected locally. None is default to preserve supplied checkpoint's preprocessing. |
| Dense feature extraction, Figs 2–3 | `DenseBlock`, `FusDenseNet` | 24-channel stem, growth 12, three layers per dense block. Three fusion stages receive cross-dense concatenated prior outputs through 1x1 transitions. Figure 2's repeated blocks and concatenation are preserved, but depth/channel sizes are selected locally. |
| Cross attention, Eqs 3–5 | `FusionDenseBlock` | Two 32-channel 8x8 representations, four-head scaled dot-product attention. Queries are each view, keys/values concatenate both. Attention operator not specified in paper. |
| Cross self-attention / normalization, Eqs 6–7 | `FusionDenseBlock.forward` | Joint key/value attention supplies within/cross-view relationships. L2-normalized maps with epsilon and sigmoid gates. Not a literal reproduction of ambiguous Eq 6. |
| Structural image fusion, Eq 8 | original and 3x3 smoothed facial view | No infrared available. Two visible views are explicit substitutes, never described as measured infrared. Source features and gated features concatenate. |
| Dilated blocks, §4.5 | alternating dilation 1/2 in dense blocks | Dilation implemented. A full paper DRDB residual topology is not specified and not claimed. |
| Multi-pooling | spatial mean, maximum, root-mean-square and (v2) 2x2 spatial average pooling | Pooling functions are project choices because the paper lacks a full definition. V2 retains coarse eye/mouth layout alongside global statistics. |
| MSE, Eq 9 | reconstruction head + MSE | 1x1 projection and bilinear upsample reconstruct input. V1 used CE + 0.1 MSE. V2 uses class-weighted label-smoothed CE + 0.05 MSE and bounded sigmoid reconstruction. These objectives are disclosed assumptions. |
| Emotion classifier | `FusDenseNet.classifier` | Seven FER classes. V1 MLP 252 → 64 → 7; v2 MLP 588 → 128 → 7. Project-trained weights are distinct from supplied Navrasa and original author weights. See actual evaluation in EXPERIMENTS.md. |
| Representation learning, Eq 10 | `FDCF` user/music/emotion embeddings and linear layer | 16 latent dimensions, learned lookup embeddings + joint linear projection. Unknown users use cold start. |
| Focus, Eq 11 | `FDCF.focus` | Softmax over two logits using encoded features, P_l and P_s. beta is first focus probability. |
| Dynamic preference, Eq 12 | `FDCF.forward` | Exact algebraic convex interpolation. P_s = last five prior explicit feedback events; P_l = all prior feedback. Signed mood-adjusted embedding averages. Windows are project choices. |
| Interaction, Eq 13 | `interaction = e*m` | Elementwise emotion/music product. |
| MLP / score, Eqs 14–15 | `FDCF.mlp`, sigmoid at ranking | 32 → 32 → 16 → 1; BCE-with-logits on observed explicit like/dislike. No unlabeled items manufactured as negative examples. |
| Recommendation personalization | `backend/recommender.py`, SQLite | Immediate weighted preference changes on like/dislike and mood feedback. Trained FDCF contributes only after `train_fdcf` creates an observed-feedback checkpoint. Default IKS scoring is explicitly a cold-start heuristic. |
| Paper custom 12-raga data | unavailable | Separate supplied 72-raga catalog used as initial metadata. Not equated to the original data. |
| Supplied EfficientNet-B2 weights | `backend/emotion.py` Navrasa option | Reconstructed from supplied notebook and backend, `weights_only=True`, CPU. Native nine-label output exposed. Seven-label aggregation is a project heuristic and uncalibrated. Never called FusDenseNet. |
| EEG/HRV and fusion | `backend/signals.py` | Exploratory extensions. MNE filtering + Welch band integration, NeuroKit2 waveform peaks, RMSSD/SDNN. Optional bounded arousal heuristic, no learned multimodal affect model. |

## Runtime ranking formula

For cold start, IKS = 0.65 Bhava + 0.20 Prahara + 0.15 Ritu. These proportions follow the inspected Raag Ratnakar scoring approach, not FDCF equations. Unknown time/season receive 0.5. Primary Bhava match = 1, secondary = 0.6, other = 0. No IKS ablation substitutes 0.5 for this score.

For a raga, feedback value is like (−1/0/+1) + 0.25 mood change (−2..+2), shrunk by n+2. Short/long feedback scores combine with beta=0.7 when feedback exists (a fixed fallback policy). Base score = 0.65 IKS + 0.35[0.5 + 0.5 tanh(preference)]. The active extension uses matched EEG/HRV reference comparisons (without emotion inference) and a maximum 0.03 body-context adjustment, plus a maximum 0.03 weather-inspired Ritu adjustment. The earlier absolute-heart-rate heuristic is no longer used by the recommendation API. A user listening goal, or the explicitly enabled reference-context policy, can select the target Bhava; it does not change the confirmed feeling. A real trained FDCF score, when available for that user, contributes 25%, with base score at 75%. The cold-start fixed beta is **not the learned FDCF focus**.

Face/manual emotion determines the Bhava policy. Signal quality/provenance remains visible, missing physiology adds no adjustment, missing catalog metadata is never generated. Listening link opens persist as history but do not count as positive preference labels because an open is not evidence of liking.

## What is incomplete

Exact original architecture recovery, subject-disjoint FER validation and controlled multi-seed architectural ablations, paper 12-raga ratings, learned acoustic tempo embeddings, a validated multimodal emotional fusion model, clinical experiments, and hardware adapters. The original pilot, expanded training run and synthetic checks remain separate in EXPERIMENTS.md.

## Version 2 training revision (3 November 2026)

The revision retains all three dual-branch fusion-dense stages, four-head cross-attention, L2-normalized attention gates, dilation, cross-dense concatenation and reconstruction. It adds input scaling from [0,1] to [-1,1] internally, BatchNorm/ReLU after fusion, 2x2 spatial pooling, a larger classification head and dropout 0.35. Parameter count is 223,776 versus 163,744 in v1. Existing v1 checkpoints remain loadable through the explicit architecture version.

Training uses the complete seeded 85% partition of the archive training folder, not the test folder. Exact-pixel deduplication leaves 23,503 training and 3,970 validation images. Horizontal flips, rotations and crops apply only to training. Vertical flips stay off. AdamW, cosine learning-rate decay, inverse-square-root class-frequency weights, 0.03 label smoothing and gradient clipping are implementation choices. Best validation macro F1 selects the checkpoint; test results never select epochs. Cached images improve CPU throughput without changing labels or splits.

The revision changes architecture, data budget and optimizer together. Any improvement is an overall engineering comparison; it cannot establish which individual change caused the gain without equal-budget retraining ablations. Full Kaggle test-folder evaluation is not the original paper's protocol, and some of that folder appeared in the previous pilot evaluation.

Continuation uses only the project's FER-trained phase-1 checkpoint, with AdamW/cosine explicitly restarted. It does not import supplied Navrasa, ImageNet or original-paper author weights. Validation macro F1 selects weights; a six-epoch patience window uses minimum cumulative improvement 0.002. Optional horizontal-flip logit averaging is selected on validation (minimum macro-F1 gain 0.002); scalar temperature minimizes validation NLL and preserves class argmax. Both fit the same validation partition used for checkpoint selection, so independent/domain-specific calibration remains unverified. backend/inference.py supplies the same deployed recipe to API inference and final evaluation.

## Multimodal correction — 3 November 2026

The full facial classifier remains 50.13% accurate on the retained test set; disgust recognition is weak. Conservative validation-based abstention replaces unsupported mood suggestions. Weather, EEG/HRV recordings, reference safeguards, listening goals and visible source contributions now reach the listener's recommendation path. This is experimental rule-based context integration, **not a validated improvement in physiological emotion recognition, recommendation quality or health**. Actual synthetic ablations and limitations are documented in [MULTIMODAL_STATUS.md](MULTIMODAL_STATUS.md). Current tests: 29/29 passed.
