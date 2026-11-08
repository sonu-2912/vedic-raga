# IKS methodology

IKS here means traceable integration of Hindustani musical knowledge and aesthetic concepts, not a clinical causal model. The initial resource is [Raag Ratnakar](https://github.com/alronova/raag-ratnakar)'s supplied JSON. File integrity and counts are in `reports/source_audit.json`.

## Data confidence and scope

The 72 records supply names, primary/secondary Sthayi Bhava, numbered Praharas, Ritu and vocal/instrumental YouTube links. All are **source-supplied and independently unverified**. Structural verification establishes counts and formats, not musicological truth. The API attaches field-level status and provenance. Thaat, Aaroh, Avaroh, Vadi and Samvadi are null/missing throughout this source. Missing values remain visible in the UI. No additional raga-specific metadata is invented.

The native Navrasa mapping follows the supplied repository (Adbhuta/Vismaya, Bhayanaka/Bhaya, Hasya/Hasa, Karuna/Shoka, Raudra/Krodha, Shanta/Shama, Shringara/Rati, Veer/Utsaha). The repository's Bhibhatsa-to-Shama fallback is a practical policy because the catalog has no Jugupsa entries, not a traditional equivalence. This project's seven-emotion operational mapping collapses Shringara and Veer into happy for continuity and must not be treated as an academically validated ontology. Native probabilities remain accessible.

Manual FER labels map to target Bhava using a declared heuristic: angry/Krodha, fear/Bhaya, sad/Shoka, happy/Hasa, surprise/Vismaya, neutral/Shama and disgust/Shama fallback. Users can correct a facial prediction. No facial expression establishes a person's inner emotional condition.

## Ranking and personalization

Primary Bhava matches receive 1, secondary matches 0.6. Prahara and Ritu provide soft bonuses, never hard exclusions. Time and season are optional user context because sunrise-dependent Prahara boundaries and geography-specific Ritu require explicit assumptions. With missing context a neutral 0.5 score is used. The UI does not silently infer local weather, sunrise or seasonal truth.

Likes/dislikes and −2..+2 self-reported mood change update short (last five events) and long (all events) preferences. This personal evidence can counter a generic association. A listening URL open is recorded but does not imply enjoyment, completed playback or clinical mood improvement.

## Evaluation and expert review protocol

Before academic claims, two qualified Hindustani music experts should independently review every raga association and provide sources, spelling aliases, gharana/performance variation and confidence. Resolve disagreements while retaining both annotations and provenance. Validate links manually. Keep performance-level tempo, alaap/gat and musician context distinct from raga-level annotations.

For a future study, obtain consent, capture self-reported pre/post affect, randomize recommendations or a suitable comparator, record exposure duration, and hold out users chronologically. Compare identical candidates with IKS enabled/disabled. Do not conflate ranking changes with listener benefit. Current IKS sensitivity output only demonstrates that metadata changes rankings; it cannot establish quality.

Relevant empirical background: [Mathur et al. 2015](https://doi.org/10.3389/fpsyg.2015.00513), [Valla et al. 2017](https://doi.org/10.3389/fpsyg.2017.02115). These motivate performance- and culture-sensitive analysis, rather than universal fixed therapeutic prescriptions.

## Multimodal correction — 3 November 2026

The full facial classifier remains 50.13% accurate on the retained test set; disgust recognition is weak. Conservative validation-based abstention replaces unsupported mood suggestions. Weather, EEG/HRV recordings, reference safeguards, listening goals and visible source contributions now reach the listener's recommendation path. This is experimental rule-based context integration, **not a validated improvement in physiological emotion recognition, recommendation quality or health**. Actual synthetic ablations and limitations are documented in [MULTIMODAL_STATUS.md](MULTIMODAL_STATUS.md). Current tests: 29/29 passed.


## Latest physiological evidence boundary

Fixed-game, fixed-30-second real GamER EEG/ECG tests with held-out people found no primary fusion F1 gain. They do not measure raga-response or weather benefit. Ritu/Prahara and body context remain exploratory ranking inputs, with no validated therapeutic or aligned facial/physiological outcome. See FURTHER_RESULTS.md and the raw-window/fold audit.
