# Attribution and resource rights

## Raag Ratnakar

Source: [alronova/raag-ratnakar](https://github.com/alronova/raag-ratnakar), supplied as `raag-ratnakar-main.zip`.

Original ZIP and extracted tree `external/raag-ratnakar-main/` remain unchanged. `reports/source_audit.json` records every file's byte count and SHA-256. `reports/repository_source_review.txt` collects all textual code/configuration and notebook cells/outputs for review. The source has a React/Vite frontend, FastAPI backend, tests, environment scoring, authentication code, a 72-raga JSON, a training notebook/PDF and an EfficientNet-B2 checkpoint.

Reuse: the JSON is read directly, no altered replacement is passed off as original. Native class ordering, preprocessing, classifier reconstruction and Rasa/Bhava mappings follow the inspected notebook and backend services. The cold-start ranking draws on `raga_recommender.py`'s Bhava/Prahara/Ritu approach. The new trainable FDCF implementation follows the selected paper, not that rule-based recommender. New UI/backend code is authored independently around these resources.

**No LICENSE, LICENCE or COPYING file exists in the supplied archive.** Dependency licenses in package-lock.json do not license the repository's own code, JSON or model. No open-source license is inferred. Obtain written authorization from its rights holders before external redistribution, publication of bundled assets or commercial use. This local research copy retains attribution and makes no new license grant for third-party assets.

The 528 URL entries include duplicates (516 unique URLs). Links are references, not 528 licensed songs. Availability, ownership, performance identity and metadata are not independently verified. Do not download or redistribute YouTube audio by assuming these links grant permission.

## Paper

Pingle and Ragha, 2025, [DOI](https://doi.org/10.1007/s41870-025-02531-y). Supplied PDF is the primary authority. Copyright and Springer notice remain intact. Figures are inspected locally; project architecture is independently described with citations. No claim of authorship of paper architecture or results.

## Software and other datasets

`data/demo_face.png` is the NASA Eileen Collins example portrait from scikit-image. The [official data documentation](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) states public-domain release. Used only for an unlabeled integration demonstration. No identity-recognition feature, participant study or endorsement is implied.

Python, PyTorch, torchvision, FastAPI, React, Vite, OpenCV, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, MNE, NeuroKit2, Pillow, pytest, httpx and PyMuPDF retain their upstream licenses. PyMuPDF is AGPL/commercial dual licensed: review distribution obligations if bundling this development dependency. Dataset-specific restrictions are in DATASETS.md. No external full dataset is redistributed here.

Weather: Open-Meteo APIs and model-derived data. Attribution appears in the check-in interface. Free API use is non-commercial; data terms: https://open-meteo.com/en/terms (CC BY 4.0). No paid subscription or account was created.

# GamER physiological recordings

Alexandros Papadopoulos, Chrysa Episkopou, Amaryllis Malegiannaki, Konstantinos Kyritsis, Ioannis Sarafis, Ioannis Papaefstathiou and Anastasios Delopoulos (2023), *GamER: Gameplay Physiological Signal Dataset for Emotion Recognition*, Zenodo v1, https://doi.org/10.5281/zenodo.8191877. Official release metadata: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. Original ZIP and annotation CSVs remain unmodified. This project creates derived EEG/cardiac features, participant-held-out predictions, plots and evaluation reports. These derived results are project work, with limitations documented in `REAL_DATA_RESULTS.md`; they do not imply creator endorsement or clinical validation. The h5py reader is used under its BSD-3-Clause licence.



## DENS access metadata and licence ambiguity

Official OpenNeuro ds003751, DOI https://doi.org/10.18112/openneuro.ds003751.v1.0.2 , by Sudhakar Mishra, Md. Asif, Uma Shanker Tiwary and Narayanan Srinivasan. Source README/LICENSE specify ODbL/DbCL while dataset_description says CC0. Both source records and the resolved full licence are preserved in reports/dens_access/. No DENS benchmark metric is claimed; only 2/12 predeclared participant headers match the referenced signal dimensions. No full DENS signals are redistributed in the presentation.


## OpenCV expression model: rejected validation-only teacher screen

Official source: https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition ; pinned commit48b10fa505aeb51fbd127fc60a5c660040e823b7. Model SHA2564f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9,4,791,892bytes. The directory licenses all files under Apache2.0. Original licence, model card and inference-source copyright notices were preserved in external/opencv_fer_teacher/. Copyright(C)2022,Shenzhen Institute of Artificial Intelligence and Robotics for Society; contributions credited in the upstream README include Jing Jiang and Chengrui Wang. The optional model was screened on adapted FER validation inputs, rejected, never used to train FusDenseNet and not used as the application's FusDenseNet implementation. No foreign model weights or original face pixels are redistributed in the filtered GitHub source. Upstream training-image overlap remains unknown. Its88.27%RAF-DB figure is not a project score.


## Optional frozen ImageNet DenseNet121 transfer branch (8 November 2026)

Torchvision0.20.1 provides the DenseNet121 architecture under BSD3Clause; the original notice is preserved in licenses/Torchvision_BSD_3_Clause.txt. Official IMAGENET1K_V1 weights: https://download.pytorch.org/models/densenet121-a639ec97.pth , SHA256 a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395 (32,342,954bytes). Documentation: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html . ImageNet data access/rights are distinct; no ImageNet images are bundled or downloaded. The project experiment freezes this feature extractor, adapts preprocessing to112pixels forCPU, and trains only a separate residual emotion classifier. It is a FusDenseNet+DenseNet121 hybrid extension, never relabelled as pure FusDenseNet or the original authors' model. Native74.434% is ImageNet accuracy, not a project emotion result. Original downloaded file is preserved locally.
