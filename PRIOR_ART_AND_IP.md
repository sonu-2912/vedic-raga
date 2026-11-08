# Preliminary prior-art and IP review

Search date: 2 November 2026 (Asia/Kolkata). This is a preliminary technical search with a limited web interface, not a legal opinion, patentability conclusion or freedom-to-operate clearance.

## Search method

Queries: exact “RagaMoodSync” title/DOI, “music recommendation” “facial” “emotion” restricted to Google Patents, and neural collaborative filtering / Indian raga affect literature. Read the located patent publication's bibliographic page, description and claims. No exhaustive USPTO, EPO, WIPO, Indian Patent Office, family, prosecution history or non-English search was completed.

## Concrete relevant publication

[US20230153350A1, Music recommendation system by facial emotion using deep learning](https://patents.google.com/patent/US20230153350A1/en). Google Patents records publication on 18 May 2023 and filing/priority on 20 January 2023. Treat dates as bibliographic page information, not independently certified legal status. The title directly overlaps the broad application concept, while description/claims emphasize audio collection, fine-grained preprocessing, vocal separation and sparse attention. The exact claims, family and official prosecution record require specialist review before a legal conclusion. A title match alone does not determine infringement or validity.

Pingle & Ragha's 2025 [paper](https://doi.org/10.1007/s41870-025-02531-y), [He et al.'s neural collaborative filtering](https://doi.org/10.1145/3038912.3052569), [DenseNet](https://doi.org/10.1109/CVPR.2017.243), and the supplied [Raag Ratnakar repository](https://github.com/alronova/raag-ratnakar) also preclude treating the broad integration of face inference, music recommendations, learned matching or Rasa metadata as an unsupported new invention.

## Claim-oriented engineering comparison

| Subject | Existing disclosure | Project implementation / remaining issue |
|---|---|---|
| Facial emotion-driven music | Patent title and selected paper | Existing broad concept. No novelty claim. |
| Attention and dense feature reuse | DenseNet and RagaMoodSync | Project concretizes underspecified blocks; choice of dimensions alone is not established novelty. |
| Learned user/music matching | NCF and FDCF paper | Trainable embeddings/focus implementation. Actual recommendation benefit remains unmeasured. |
| IKS-informed catalog | Raga/emotion research and Raag Ratnakar | Local provenance/status integration. Expert adjudication and controlled outcome evidence pending. |
| EEG/HRV context | Established physiological affect literature | Offline extensions with generic features, no novel biosensor method or validated clinical intervention. |

## Rights and disclosure actions

No LICENSE file was found in the supplied repository. Retain original files and notices, seek rights-holder permission before distributing its JSON, weights or adapted source. A repository being publicly viewable does not establish a reuse license. Dataset permissions and recording copyrights remain separate from source-code rights. A license on derived audio features does not automatically cover original music recordings. See THIRD_PARTY_NOTICES.md and DATASETS.md.

Before publication or commercialization, commission a broader jurisdiction-specific search and official claim analysis, resolve third-party rights, obtain participant consent and ethics review where needed, and describe the contribution as an evaluated method only after valid experiments. No patent filing or legal clearance was performed.
