# Supplied repository audit

The root ZIP was extracted without replacing the original into `external/raag-ratnakar-main/`. `scripts/audit_sources.py` reads every source/configuration/notebook text file and hashes all files, including binaries. `reports/source_audit.json` is the machine-readable inventory. `reports/repository_source_review.txt` holds complete textual sources and notebook outputs. The separate 13-page Navrasa PDF is extracted in `reports/navrasa_document.txt`.

## Verified resources

| Resource | Finding |
|---|---|
| `raga_sthayi_bhava_dataset_enriched.json` | 72 raga objects, 528 link entries, 516 unique URLs. Keys: name, primary/secondary Sthayi Bhava, Prahara, Ritu, vocal/instrumental link groups. Missing other classical fields. |
| `best_navarasa_model.pth` | 101,850,648 bytes. Loads safely with PyTorch `weights_only=True`. Dictionary has model state, optimizer state, epoch, val_acc, class_names. Shapes load strictly into EfficientNet-B2 with 1408→512→9 head. |
| `navrasanet_model_code.ipynb` | ImageFolder-based nine-Rasa EfficientNet-B2 transfer learning. Resize 256/CenterCrop 224 and ImageNet normalization for validation/inference. Two-phase fine-tuning, label smoothing, dropout, cosine scheduling and training logs. Dataset directories reference external Google Drive and are not present. |
| `raga_recommender.py` | A deterministic weighted Bhava/Prahara/Ritu scoring policy with traditional, therapeutic and activity modes. It is not FDCF. Runtime source weights are 0.65/0.20/0.15. |
| `rasa_bhava_mapper.py` | Nine Rasa labels mapped to Bhava; Bhibhatsa uses Shama as a documented catalog-coverage fallback. |
| Navrasa PDF | Separate EfficientNet-B2/NFEC project description. Its claims and results are source-reported, not newly reproduced. It is not the selected RagaMoodSync paper. |

Checkpoint native class order is Adhbuta, Bhayanak, Bhibhatsa, Hasya, Karuna, Raudra, Shanta, Shringara, Veer after casing normalization. The serialized epoch is 28 and stored validation accuracy is 93.4250%. Those metadata values are **not independently measured project accuracy**, and say nothing about FER2013 accuracy or mapped seven-class calibration. `reports/checkpoint_audit.json` records them.

## Existing system

The backend is FastAPI with route/schema/service layers, SQLite users/recommendation history, custom signed bearer tokens and PBKDF2 password hashing, environmental context, lazy PyTorch inference and rule-based recommendations. Its environment module uses sunrise/sunset via Astral and Open-Meteo weather, with calendar Ritu policy. The frontend is React 19/Vite 6 with router-based authentication, dashboard, capture, recommendations, camera, metadata panels and CSS themes. Its requirements are unpinned. The package lock includes individual dependency notices, which are preserved.

The source PDF's implementation description differs from inspected code in places: it describes JWT/bcrypt while source uses a custom HMAC token and PBKDF2; it describes 70/20/10 scoring while source uses 65/20/15. This project follows actual inspected resources where reusing them and documents its own implementation. Source-reported claims of subject-disjoint performance and expert curation cannot be independently re-evaluated because NFEC images and subject splits are absent.

## Reuse decisions

Read the original JSON directly. Reuse verified checkpoint tensor architecture, class metadata, transforms and Rasa/Bhava policy with attribution. Implement FusDenseNet and FDCF independently from the selected paper's figures/equations. Provide a new local interface focused on feedback and offline operation. Do not use remote weather/GPS as a requirement for the pipeline, or inherit disease-management promises from source descriptions.

All original source files remain untouched. No repository LICENSE/LICENCE/COPYING file was found. This is a license gap, not an implied permissive grant. See THIRD_PARTY_NOTICES.md. Never count dependency license entries as a license for the data or model.
