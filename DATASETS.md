# Dataset access and use

Access review: 2 November 2026, Asia/Kolkata. FER2013 version 1 was downloaded only after a public API metadata/license/size check and a disk-space check (about 646 GB free). All other research datasets remain undownloaded. Dataset rights, consent and access must still be checked per release. Sizes below marked unknown must be checked before transfer.

| Source | Format and purpose | Access and license | Current project status |
|---|---|---|---|
| [FER2013, msambare Kaggle mirror](https://www.kaggle.com/datasets/msambare/fer2013) | 48x48 grayscale facial images. Downloaded version 1 has 28,709 train and 7,178 combined test-folder images. Original CSV has emotion, pixels, Usage. | Public [metadata API](https://www.kaggle.com/api/v1/datasets/view/msambare/fer2013) advertises “Database: Open Database, Contents: Database Contents”. [ODbL](https://opendatacommons.org/licenses/odbl/1-0/) / [DbCL](https://opendatacommons.org/licenses/dbcl/1-0/) texts reviewed. Version 1 downloaded without credentials. Archive 63,252,113 bytes; contained files 56,510,189 bytes. Underlying image/personality rights are not independently audited. | Available locally in data/private/fer2013-v1.zip. Revised full training partition: 23,503 train / 3,970 validation after deduplication; expanded test results in EXPERIMENTS.md. No public redistribution. ZIP, folder and original CSV loaders implemented. |
| [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) | EEG and peripheral physiology, stimulus ratings; official preprocessed Python/MAT data and original recordings. Offline signal/affect research. | Official site timed out / returned 502 during review. Historically access uses a research request and end-user agreement. Obtain current authorization and verify terms from custodian. Current archive sizes and continued access unverified. | No download. Not synchronized Hindustani raga responses. Generic EEG adapter accepts selected signal epochs. |
| [WESAD](https://archive.ics.uci.edu/dataset/465/wesad) | Subject-wise physiological/motion recordings, stress/neutral/amusement labels. ECG/PPG feature validation. | UCI identifies an external dataset and explicitly defers licensing to linked original source. Follow original homepage/readme and acknowledgement policy. Size not checked at source. | Not downloaded. Does not provide EEG or a raga-response study. Generic ECG/PPG and RR adapters implemented. |
| [PhysioNet](https://physionet.org/) | A platform of separate datasets, often WFDB ECG/annotations, not one dataset. Select a specific open signal record for HRV validation. | Each record has its own license/access category. Open, restricted and credentialed resources differ. Credentialed datasets can require training, DUA and approval. Select only a suitable licensed resource after checking record page. | Platform reviewed, no dataset selected or downloaded. No implied mood labels. |
| [Saraga](https://mtg.github.io/saraga/) | MusicBrainz-linked metadata, mp3 audio, pitch/time-aligned music annotations. Future acoustic raga features and metadata review. | [License](https://github.com/MTG/saraga/blob/master/LICENSE.md): recordings/manual/automatic annotations CC BY-NC 4.0; repository code AGPL-3.0. Check editorial metadata's source terms separately. [Access instructions](https://mtg.github.io/saraga/access.html) recommend Zenodo or mirdata; older notebooks are discontinued. | Not downloaded. Neither listener affect nor user feedback ground truth is provided by this music collection alone. |
| [IndRaga DOI](https://doi.org/10.5281/zenodo.19978345) | User-identified precomputed Hindustani audio features. Intended for future music embedding research. | DOI/Zenodo record/API could not be opened by the browsing tool. NLM catalog independently lists the exact DOI/title and CC BY 4.0, but the primary release license, files, checksums and size remain unverified. [Catalog corroboration](https://datasetcatalog.nlm.nih.gov/dataset?q=0002981067). | Not downloaded. Do not treat its reported counts or metadata as verified project data. Check primary record before acquisition. |
| [CompMusic](https://compmusic.upf.edu/) | Research project and Dunya corpora/API ecosystem, not a single uniform dataset. Cultural music analysis and future metadata links. | Follow the specific collection's licensing and API registration requirements. Saraga is one separately licensed resource. No blanket license for all Dunya audio. | Website reviewed. No bulk download or API credentials required for this application. |
| Supplied Raag Ratnakar JSON | 72 records, 528 URL entries, 516 unique URLs. Initial raga/Bhava/Prahara/Ritu catalog. | No repository license file in supplied ZIP. Permission for distribution unresolved. YouTube links do not license recordings. See THIRD_PARTY_NOTICES.md. | Used locally, source unchanged. Field status explicitly unverified. |
| Synthetic samples | 16-second EEG at 128 Hz, 90 RR intervals; artificial image textures for engineering tests. | Generated deterministically by project code. No participants, clinical labels or sensor readings. | Used only in explicitly marked synthetic experiments. |
| One NASA demo portrait | `data/demo_face.png`, Eileen Collins portrait from scikit-image v0.24.0. One small, unlabeled real image for integration testing only. | [Official scikit-image provenance](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) states NASA origin and public-domain release. [Exact downloaded file](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png). | Downloaded after rights review. No emotion ground truth, no accuracy claim and no endorsement implied. |

## Manual FER2013 download

The exact version-1 archive is already present locally. SHA-256: `9f3422c7a45b676c2965ef4fc5f3099932aa98fd13eaaa2c254f620a9894a6e9`. Metadata and per-class counts are saved in `reports/fer2013_kaggle_metadata.json` and `reports/fer2013_archive_audit.json`. The API description mentions 3,589 public-test examples, but the actual archive test folder contains 7,178 images; this project reports its selected subset of that combined folder explicitly rather than claiming the original PrivateTest protocol.

The loader reads the ZIP directly, avoiding extraction of 35,887 small files. For example, pass `--data data/private/fer2013-v1.zip`. The pilot stratifies limited subsets with seed 42 and removes exact grayscale-pixel duplicates across training, validation and selected test samples. Identity disjointness is not established.

1. Open the Kaggle link above. Sign in if required and review Data Card, access conditions, file listing and license. Record dataset version and checksum in your experiment log.
2. Check archive size and available disk space. Download using the website. With the optional official Kaggle CLI configured locally, `kaggle datasets download -d msambare/fer2013 -p data/private` is an alternative. Do not place API tokens in this repository.
3. Either read the ZIP directly or extract so `data/fer2013/train/angry/`, `.../disgust/`, `fear/`, `happy/`, `sad/`, `surprise/`, `neutral/` and matching `test/` folders exist. The loader makes a seeded 15% stratified validation split from train when no `val/` folder exists. It never uses test to select a training checkpoint.
4. For original CSV, retain official Usage values Training/PublicTest/PrivateTest. Pass the CSV path instead of a folder.
5. Run training and evaluation from README.md. Audit near-duplicate and identity overlap before publishing. The Kaggle folder split is not established here as identity-disjoint.

## Other manual download workflows

- **DEAP**: visit official download page, obtain its current research access agreement and credentials from custodians, then download only the authorized preprocessed subset needed. Preserve subject IDs and split by participant. Never load untrusted Python pickle files. Convert trusted, authorized data offline to explicit channel arrays in volts with actual sample rate and epoch boundaries.
- **WESAD**: follow UCI's original dataset homepage to download/read the official README and license. Select ECG (chest, 700 Hz) or BVP (wrist, 64 Hz), convert a single contiguous quality-screened epoch to the JSON waveform format below. Keep participants separate in evaluation.
- **PhysioNet**: choose a named dataset and version, review access and license, and acquire only a needed record. Read units and peak annotations from its header. Do not label arrhythmia or synthetic waveforms as a mood measurement.
- **Saraga**: use the official Access page's Zenodo collection or mirdata Hindustani loader. Review CC BY-NC conditions and file sizes first. Acquire metadata before multi-hour audio. Preserve recording IDs and annotations, and acknowledge the collection paper [DOI](https://doi.org/10.18061/emr.v16i1.7641).
- **IndRaga**: retry the exact DOI on Zenodo, inspect metadata, files, checksums, license and version. If unresolved, contact authors rather than substituting an unrelated dataset. Do not infer rights to linked original audio from a feature release license.
- **CompMusic**: use Dunya to inspect the relevant corpus; acquire only resources whose permission is explicit. Current Saraga instructions deprecate older bulk scripts.

## Signal interface

POST `/api/eeg/process`: `{"samples":[[...]],"sample_rate":128,"sample_data":false}`. Shape is channels × samples, volts, at least 4 seconds, 80–4096 Hz. The theta [4,8), alpha [8,13), beta [13,30) integrals use Welch PSD after 1–35 Hz IIR filtering. Artifact rejection and calibrated affect classification are pending.

POST `/api/hrv/process`: `{"rr_ms":[...],"sample_data":false}` for at least 20 artifact-corrected intervals, 300–2000 ms. Alternatively `{"signal":[...],"sample_rate":250,"kind":"ecg","sample_data":false}` or kind `ppg`, at least 30 seconds and 50–4096 Hz. NeuroKit2 detects waveform peaks. Metrics use successive RR differences and sample standard deviation. Short recordings are exploratory estimates.

GET `/api/signals/sample` returns valid deterministic requests. The UI can process this mode without sensors or upload offline JSON. File-backed EDF/BrainVision/DEAP/WESAD-specific importers are not implemented; conversion to these explicit interfaces is required. SensorAdapter is a future hardware contract, not a connected device.

## Weather context extension — 3 November 2026

Open-Meteo [forecast documentation](https://open-meteo.com/en/docs), [city geocoding](https://open-meteo.com/en/docs/geocoding-api) and [terms](https://open-meteo.com/en/terms) were reviewed. The free API is for non-commercial uses; data are CC BY 4.0 with attribution. The app caches city lookups for five minutes and requests only small JSON responses. API access and current terms must be reviewed for commercial deployment. Model-based weather has uncertainty and is not a physical sensor reading. The app shares the explicitly entered city and resolved city coordinates with Open-Meteo; it does not obtain device GPS. The app shows source attribution, resolved city and provider time. Lookup failures leave manual condition/temperature entry available. A live lookup for Pune was an integration example, not an assumption of the user's location.

Weather/Ritu and body-reference rules are experimental extensions. They provide no validated mood or therapeutic labels. Real EEG/HRV uploads need personal resting references and declared matched sessions for joint scoring; equal IDs alone do not prove synchronization. Do not concatenate unrelated DEAP/WESAD records into a supposed raga-response dataset. Raw faces and uploaded physiological arrays are not persisted by the check-in flow.
# Real paired physiology acquired for this project: GamER

Official release: [Papadopoulos et al., GamER v1, Zenodo](https://doi.org/10.5281/zenodo.8191877), released 28 July 2023. The official API specifies **CC BY 4.0**, open access, a **919,686,572-byte** `data.zip`, `emotion_annotation.csv` and `video_offsets.csv`. Licence/access, size and over 600 GB free space were checked before acquisition. Originals are stored under `data/private/gamer`; the downloader verifies the publisher's MD5 checksums. No account, signed agreement or unauthorized mirror is used.

Recordings contain simultaneous EEG, ECG and other physiological channels in OpenSignals HDF5/TXT, organized by participant. Sampling rates and sensor labels are read from each file rather than assumed. Psychologist-produced annotations use gameplay/video intervals and source emotion names. They do not contain disgust, FER2013 images, weather or Hindustani raga-response labels. Annotation-offset sign is not independently confirmed; subtraction and addition are reported separately. Short annotated windows provide exploratory ultra-short RR statistics, not validated clinical HRV.

Reproduction: retain `reports/gamer_metadata.json`, then run `python -m scripts.download_gamer`, `python -m scripts.gamer_experiment` and `python -m scripts.gamer_experiment --sensitivity` using the project virtual environment. The downloader resumes retained byte-range chunks after interruptions and checks the assembled release. See `REAL_DATA_RESULTS.md` for participant-disjoint evaluation, exclusions and confounding controls.

Access alternatives reviewed: [DREAMER's official record](https://zenodo.org/records/546113) is restricted; [Emognition's author repository](https://github.com/Emognition/Emognition-wearable-dataset-2020) requires an academic EULA/access request. These restrictions were respected. Neither was downloaded from an unofficial mirror. [WESAD's original site](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) allows scientific, non-commercial use with attribution, but it has no EEG; it cannot provide an EEG–HRV experiment by itself.



## Further controlled real physiology and DENS access

## Real EEG/ECG fusion with game identity held constant

Official GamER CC BY 4.0 recordings, DOI https://doi.org/10.5281/zenodo.8191877. Separate fixed-game tasks, centered exactly 30-second labelled windows, leave-one-participant-out folds, fixed balanced logistic regression C=1 and fold-fitted StandardScaler. EEG theta/alpha/beta/delta/gamma relative powers, entropy and Hjorth features; real ECG-derived RR mean, RMSSD, SDNN, pNN50, median and IQR. EEG and ECG share the source interval. No simulated signal contributes to these results.

| Constant game | Windows / people | EEG macro F1 | Cardiac macro F1 | Fusion macro F1 | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| QWOP | 13 / 12 | 0.4583 | 0.3158 | 0.4583 | 0.7692 |
| CBAAD | 13 / 13 | 0.6061 | 0.5752 | 0.6061 | 0.6923 |

**Neither primary task shows a fusion macro-F1 gain.** Fusion accuracy is below majority accuracy. Participant-bootstrap paired intervals include zero. Game identity and window duration are constant within each task, so neither cue distinguishes labels. This controls those two particular confounds, not every artifact, movement or annotation error. Primary offset subtraction is fixed before evaluation; alternate addition is a sensitivity analysis, never selected for a better score. Exact raw-window recomputation, source annotation IDs, fold isolation, constant controls and independently recomputed confusion matrices passed `reports/gamer_fixed_context_audit.json`.

The older pooled Angry/Sad result is retained as historical and confounded (game identity alone classified all labels). It is not evidence of a physiology-specific emotion improvement. Thirty-second RR statistics are exploratory, not conventional five-minute clinical HRV. Longer-window within-game cohorts are insufficient. No facial/EEG alignment, disgust physiology, real raga responses, weather benefit or therapeutic efficacy is demonstrated.

## Independent game-free dataset attempt

DENS https://doi.org/10.18112/openneuro.ds003751.v1.0.2 and https://github.com/OpenNeuroDatasets/ds003751 contain EEG/ECG and movie-related self-ratings. A protocol fixed the first 12 released participants, 60-second windows and crossed participant/movie held-out folds. Only 2 official headers matched their annex-referenced float32 signal dimensions. The other 10 were excluded before signal interpretation. This cohort cannot support the planned benchmark, so **no DENS accuracy or fusion result is claimed**. Headers were fully MD5 checked; no partial signal is misrepresented as full-FDT checksum verified. See dens_download_plan.json, dens_access_outcome.json and dens_protocol.json. The modular evaluator is implemented but remains unvalidated on an adequate DENS cohort.

Dataset metadata says CC0 while its README/LICENSE specifies ODbL/DbCL. Preserve the source licence, attribution and more protective conditions pending clarification. No full signals or third-party assets are bundled into the PPT. No replacement subjects were chosen after observing scores.



## DREAMER access check — 7 November 2026

Official record https://zenodo.org/records/546113 (Katsigiannis and Ramzan; paper DOI https://doi.org/10.1109/JBHI.2017.2688239) describes paired EEG/ECG affect recordings. The official API currently reports restricted access, no accessible files and no machine-readable licence. Signal format, size and terms cannot be confirmed from an accessible file; no signal was downloaded and no score is claimed. Request access through the custodian and review the actual terms before use. This is another possible physiological dataset, not a synchronized Hindustani raga/face-response source. Metadata evidence is under reports/consistency_20261107/dreamer_access/.


## FER+ annotation access check

Official source: https://github.com/microsoft/FERPlus ; see its LICENSE.md and preserve Microsoft attribution before use. The extra CSV contains crowd annotation distributions in the original FER2013 CSV row order. It contains no original face pixels. It was inspected as a possible quality extension, but is **not used** in the deep-supervision experiment: the available Kaggle folder PNG names have no verified row-index crosswalk. Manual procedure: acquire the original FER2013 CSV through the official challenge access process, verify its permitted use, reconstruct pixels and match exact hashes to the existing fixed train/validation manifests, and apply only training-row annotations. Never relabel validation/test to inflate a seven-class FER2013 score; report any different FER+ task separately. No large additional dataset was downloaded or unverified label join performed.


## ImageNet feature pretraining, not an additional facial-response dataset

The optional hybrid probe uses official Torchvision DenseNet121 IMAGENET1K_V1 pretrained weights (30.8MiB), downloaded from download.pytorch.org after checking file size and available disk space. No ImageNet image dataset is downloaded. Frozen backbone provenance/SHA: reports/dense_transfer112_20261108/pretrained_provenance.json; code notice: licenses/Torchvision_BSD_3_Clause.txt. FER training/validation labels and images remain those in the fixed pixel-hash manifest; no FER-pretrained teacher is used. Input is replicated grayscale with resize128/center112 and ImageNet mean/std, a documented CPU adaptation from native224pixels. The224-pixel attempt produced no evaluation result and was abandoned before completed feature extraction dueCPUcost. New residual-head training uses only FER TRAIN; validation only selects candidates. This is a hybrid extension, not an exact paper reproduction.
