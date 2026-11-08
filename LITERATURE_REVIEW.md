# Literature review

Search performed 2 November 2026 through DOI, publisher, author repository, arXiv and dataset-provider pages. This is a focused preliminary review, not a systematic review. Exact queries and access limitations are summarized below. The selected RagaMoodSync paper remains the implementation authority.

| Verified work | Relevance | What it does not establish here |
|---|---|---|
| Pingle & Ragha (2025), RagaMoodSync. [10.1007/s41870-025-02531-y](https://doi.org/10.1007/s41870-025-02531-y) | Supplied complete paper, FusDenseNet/FDCF mechanisms. | Original reported accuracy is not a project result. Architecture and clinical protocol are incompletely specified. |
| Goodfellow et al. (2013), Challenges in Representation Learning: A report on three machine learning contests. [10.1007/978-3-642-42051-1_16](https://doi.org/10.1007/978-3-642-42051-1_16), [author preprint](https://arxiv.org/abs/1307.0414) | Source context for FER facial-expression benchmark. | Labels describe expressions and do not measure internal mood or therapeutic suitability. |
| Huang et al. (CVPR 2017), Densely Connected Convolutional Networks. [10.1109/CVPR.2017.243](https://doi.org/10.1109/CVPR.2017.243), [primary paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) | Dense concatenation and feature reuse. | Standard DenseNet is not automatically the paper's FusDenseNet. |
| He et al. (WWW 2017), Neural Collaborative Filtering. [10.1145/3038912.3052569](https://doi.org/10.1145/3038912.3052569), [author preprint](https://arxiv.org/abs/1708.05031) | Neural matching of learned user/item representations. | Does not validate this project's FDCF, sparse-feedback thresholds or emotion mapping. |
| Mathur, Vijayakumar, Chakrabarti & Singh (2015), Emotional responses to Hindustani raga music: the role of musical structure. [10.3389/fpsyg.2015.00513](https://doi.org/10.3389/fpsyg.2015.00513) | Empirical listener ratings motivate performance-level structure and context. | Does not justify universal clinical prescriptions from a raga name. |
| Valla, Alappatt, Mathur & Singh (2017), Music and Emotion—A Case for North Indian Classical Music. [10.3389/fpsyg.2017.02115](https://doi.org/10.3389/fpsyg.2017.02115) | Reviews NICM as a useful music/emotion research context. | Cultural interpretation is not a fixed face-to-raga ontology. |
| Koelstra et al. (2012), DEAP: A Database for Emotion Analysis Using Physiological Signals. [10.1109/T-AFFC.2011.15](https://doi.org/10.1109/T-AFFC.2011.15), [author manuscript](https://www.sanderkoelstra.nl/media/documents/tac_special_issue_2011_final.pdf) | Multimodal physiological affect benchmark with stimulus ratings. | Music-video responses are not synchronized Hindustani raga trials. |
| Schmidt et al. (2018), Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985), [provider record](https://archive.ics.uci.edu/dataset/465/wesad) | Physiological/motion data for stress-affect research. | No EEG or raga-response ground truth. |
| Srinivasamurthy, Gulati, Caro & Serra (2021), Saraga: Open Datasets for Research on Indian Art Music. [10.18061/emr.v16i1.7641](https://doi.org/10.18061/emr.v16i1.7641), [official collection](https://mtg.github.io/saraga/) | Open music data with cultural metadata and annotations. | Does not independently provide listener mood-improvement labels. |
| Gramfort et al. (2013), MEG and EEG data analysis with MNE-Python. [10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267) | Reproducible EEG processing tools. | Bandpower is not a validated emotional diagnosis. |
| Makowski et al. (2021), NeuroKit2: A Python toolbox for neurophysiological signal processing. [10.3758/s13428-020-01516-y](https://doi.org/10.3758/s13428-020-01516-y), [official docs](https://neuropsychology.github.io/NeuroKit/introduction.html) | ECG/PPG peak processing and physiological tooling. | HRV from short, noisy windows does not establish mood or clinical benefit. |

## Research gap and possible contribution

The executable project joins paper-informed mechanisms, traceable source metadata, explicit sensor simulation and persisted feedback in one local CPU application. This is an engineering integration, not a proven scientific novelty. A defensible research contribution would require expert-reviewed IKS annotations, subject-disjoint facial evaluation, prospective listening feedback, chronological recommendation evaluation and a controlled IKS ablation. An emotion-aware recommendation prototype alone is already prior art.

## Verification notes

Queries included the exact RagaMoodSync title/DOI, “Emotional responses to Hindustani raga music”, “Neural Collaborative Filtering”, “Densely Connected Convolutional Networks”, DEAP official download/license, WESAD provider, Saraga license, IndRaga DOI, MNE and NeuroKit2 citation records. The RagaMoodSync DOI resolver and some publisher pages were inaccessible, so its supplied PDF is the complete primary source. DEAP official pages timed out. IndRaga's primary release could not be accessed and is documented as unresolved, not promoted to verified academic findings. No unverified generic entries from the selected paper's bibliography were invented or copied into this review.

## Added experimental comparator — 3 November 2026

Dalal, N. and Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR, pp. 886–893. [Primary DOI record](https://doi.org/10.1109/CVPR.2005.177) verified. The paper establishes HOG descriptors in a human-detection setting, not facial-emotion benchmark performance. This project uses a fixed HOG + linear-SVM classifier only as an independently labeled FER2013 comparator. It does not replace FusDenseNet or establish paper architecture fidelity.

# Additional real-data resource evaluated

Papadopoulos, A., Episkopou, C., Malegiannaki, A., Kyritsis, K., Sarafis, I., Papaefstathiou, I., and Delopoulos, A. (2023). *GamER: Gameplay Physiological Signal Dataset for Emotion Recognition*, version 1. Zenodo. [DOI: 10.5281/zenodo.8191877](https://doi.org/10.5281/zenodo.8191877). This is a dataset release; no unverified accompanying-paper citation is invented. Official metadata identifies real simultaneous EEG/ECG and other sensors, psychologist-produced emotion annotations, open access and CC BY 4.0. The project's participant-held-out ablations, offset-sign sensitivity and game-ID control are reported in `REAL_DATA_RESULTS.md`. They are project results, not the original RagaMoodSync paper's results.

Saganowski et al. (2022), *Emognition dataset: emotion recognition with self-reports, facial expressions, and physiology using wearables*. Scientific Data 9, 158. [DOI: 10.1038/s41597-022-01262-0](https://doi.org/10.1038/s41597-022-01262-0). The [author repository](https://github.com/Emognition/Emognition-wearable-dataset-2020) requires an academic EULA and access request; it was reviewed, not downloaded. DREAMER ([DOI: 10.1109/JBHI.2017.2688239](https://doi.org/10.1109/JBHI.2017.2688239)) likewise has a restricted official record. Public descriptions must not be confused with permission to use gated recordings.



DENS primary dataset: Mishra, S., Asif, M., Tiwary, U. S. and Srinivasan, N. Dataset on Emotion with Naturalistic Stimuli. OpenNeuro ds003751 v1.0.2. https://doi.org/10.18112/openneuro.ds003751.v1.0.2 . Source metadata and README inspected; conflicting licensing and signal/header mismatches are recorded, with no evaluation score claimed.


## Full-network regularization and additional data-access review (7 November 2026)

Hinton, G., Vinyals, O. and Dean, J. (2015). Distilling the Knowledge in a Neural Network. https://doi.org/10.48550/arXiv.1503.02531 . This motivates soft-target distillation generally. Our small-temperature KL penalty uses the preceding FusDenseNet on training faces only; the correct-teacher mask, augmentation and EMA selection are disclosed project choices, not claims of a new method or exact replication of that paper. No external teacher or extra labelled face dataset was used.

Katsigiannis, S. and Ramzan, N. DREAMER: A Database for Emotion Recognition Through EEG and ECG Signals From Wireless Low-cost Off-the-Shelf Devices. IEEE Journal of Biomedical and Health Informatics22(1),98–107 (2017 online;2018 issue). https://doi.org/10.1109/JBHI.2017.2688239 . The authors describe23 participants and paired EEG/ECG with audiovisual elicitation. The official dataset record https://zenodo.org/records/546113 was restricted with no exposed files at our access check. We report no DREAMER score and do not transfer the accepted-paper manuscript licence to dataset signals.


## Deep supervision and MixUp follow-up

Lee, C.-Y., Xie, S., Gallagher, P., Zhang, Z. and Tu, Z. (2015). Deeply-Supervised Nets. PMLR38:562–570. https://proceedings.mlr.press/v38/lee15a.html . Companion objectives motivate direct supervision of intermediate representations. Our single training-only classifier on588late fused features is a disclosed adaptation; it does not replace the selected RagaMoodSync inference architecture or establish a causal benefit without a controlled ablation.

Zhang, H., Cisse, M., Dauphin, Y. N. and Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. ICLR; https://doi.org/10.48550/arXiv.1710.09412 . We interpolate training image pairs and both labels on half of batches using Beta(.2,.2), alongside the main and companion losses. Dataset-specific benefit must be measured here; the cited paper's results are not project results.

Microsoft FER+ official annotations: https://github.com/microsoft/FERPlus . Their rows correspond to the original fer2013.csv row order. Our acquired folder-format PNG archive does not provide a verified mapping to those row indices. We do not join labels by filename resemblance, guessed order or image-model predictions, and no FER+ annotation was used in this trial.


Jiang, J. and Deng, W. Boosting Facial Expression Recognition by A Semi-Supervised Progressive Teacher. https://arxiv.org/abs/2205.14361 ; OpenCV released MobileFaceNet model card: https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition . Its reported88.27%belongs to RAF-DB. Our separate adapted grayscale validation screen measured43.4509%and was rejected; no foreign teacher was used in FusDenseNet training, and no clean upstream-overlap claim is made. This negative cross-domain screen does not contradict its published RAF-DB result.


Foret, P., Kleiner, A., Mobahi, H. and Neyshabur, B. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR; https://doi.org/10.48550/arXiv.2010.01412 . The final classifier-only comparison uses a project SAM/AdamW recipe with identical dropout masks and exact perturbation restoration. Any benefit must be measured on this task; published results are not project results.


## Audit-driven imbalance experiment (8 November 2026)

Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., and Kumar, S. (2021). *Long-tail Learning via Logit Adjustment*. ICLR. https://doi.org/10.48550/arXiv.2007.07314 . Primary source: https://arxiv.org/abs/2007.07314 . This work motivates distinguishing class-prior effects from feature learning. The project uses a disclosed sampling-mixture/output-bias correction derived only from training frequencies; it does not reproduce the paper's experimental protocol or claim its reported results. Actual minority-class overfitting and the new residual spatial extension are measured separately in SPATIAL_GENERALIZATION_RESULTS.md.


## Pure FusDenseNet training extensions (8 November 2026)

Khosla, P. et al. (2020), *Supervised Contrastive Learning*, NeurIPS. https://doi.org/10.48550/arXiv.2004.11362 . The pure experiment uses an independently implemented class-balanced contrastive projection on late fused training features; sampling, loss weight and architecture dimensions are project choices. No claim to reproduce the authors' benchmark results.

Liang, X. et al. (2021), *R-Drop: Regularized Dropout for Neural Networks*, NeurIPS. https://doi.org/10.48550/arXiv.2106.14448 . Primary author source: https://github.com/dropreg/R-Drop . Bidirectional KL regularizes classifier dropout. With deterministic frozen-BN backbone and classifier-only dropout, shared features preserve two-forward parameter gradients; the numerical contract is tested. No external encoder, hybrid teacher or inference ensemble is introduced. RagaMoodSync2025 remains the architecture authority; these are explicit training extensions.


Wu, Y. and He, K. (2018), *Group Normalization*, ECCV. https://doi.org/10.48550/arXiv.1803.08494 . Native pure FusDenseNet v8 uses GroupNorm only in its newly added residual dense refinements, avoiding new running batch statistics. This is a project normalization/depth choice, not a recovered detail of the selected RagaMoodSync paper.


## Label-quality context reviewed during the pure repair

Barsoum, E., Zhang, C., Canton Ferrer, C. and Zhang, Z. (2016), *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution*, [DOI:10.48550/arXiv.1608.01041](https://doi.org/10.48550/arXiv.1608.01041), studies multiple-annotator label distributions. The [official FER+ repository](https://github.com/microsoft/FERPlus) maps annotations to the original FER2013 CSV row order. Correspondence to this workspace's JPEG archive has not been verified. No FER+ labels or pretrained model were used, and no project test labels were changed. This source supplies context, not evidence that a particular project label is wrong.
