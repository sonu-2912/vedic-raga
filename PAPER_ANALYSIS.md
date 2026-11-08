# Paper analysis

Primary authority: Yogesh Prabhakar Pingle and Lakshmappa K. Ragha, *RagaMoodSync: a deep learning framework for Hindustani Raga recommendations driven by facial emotions*, International Journal of Information Technology 17(5), 2765–2778, 2025. [DOI](https://doi.org/10.1007/s41870-025-02531-y).

The supplied root PDF is preserved. A reference copy is `references/RagaMoodSync_Paper.pdf`; complete extraction of all 14 pages is `reports/paper_extracted.txt`. Page numbers below refer to PDF pages.

## Research problem and method

The work joins facial emotion recognition and Hindustani music recommendation. Pages 3–7 describe geometric augmentation, NLM/bilateral/Gaussian denoising, Fusion-DenseNet with cross-dense connections and cross-attention, feature concatenation, multi-pooling, and Focus Deep Collaborative Filtering. Figure 1 shows the full pipeline, Figures 2–3 the facial feature/fusion structure, and Figure 4 the user/music/emotion recommendation network. Hardware delivery using RPI_ZERO appears in the proposal but is outside this Windows software implementation.

## Equations

1–2 define Gaussian/bilateral filtering. 3–5 compute two attention maps from source features and their concatenation and then concatenate source and attention-weighted features. 6–7 describe cross-self attention and pixel weight normalization but leave index ranges and tensor dimensions ambiguous. 8 concatenates infrared and visible features, although FER2013 provides grayscale visible facial images. 9 defines reconstruction MSE. A classifier output, target coding, and combination of reconstruction and classification objectives are not fully specified.

10 defines linear embedding E = W_e x_e + b_e. 11 uses softmax focus over encoded features and long/short preferences. 12 forms P_d = beta P_s + (1-beta) P_l. 13 multiplies emotion and music embeddings elementwise. 14 matches concatenated dynamic preference and interaction through an MLP. 15 applies an output activation. 16–17 offer symbolic computational cost expressions without enough layer dimensions to derive exact FLOPs.

## Datasets and reported experiments

Section 4.1 names FER2013 and a custom 12-raga dataset with professional instrumental performances and emotional ratings. Neither the custom dataset, user interactions, nor paper model weights are supplied in this workspace. The Raag Ratnakar 72-raga catalog is a separate initial knowledge source.

Table 2 reports accuracy 0.9972, precision 0.9896, F1 0.9901, recall 0.990, sensitivity 0.9906 and specificity 0.9982. Table 5 instead reports full-model accuracy 0.9982, precision 0.9905, recall 0.9912, F1 0.9921. These are **paper-reported values**, not project measurements. The text additionally quotes precision 99.05%, which differs from Table 2. Preserve these disagreements rather than selecting the most favorable number.

Figure 5 is the paper confusion matrix. Figures 6–10 compare training, metrics and ROC behavior. The paper describes training through 300 epochs. It does not provide the complete executable architecture, seed, exact train/validation/test protocol, split integrity checks or confidence intervals needed to independently establish these results. Table 5 ablates cross-attention and dilated residual dense blocks, with a full FusDenseNet/FDCF model, but the separation of recommendation and facial classification evaluation is insufficiently clear.

## Limitations and critical reading

- Emotion terminology shifts between seven FER classes, happy/sad, and happy/sad/tensed/neutral. The project uses the standard seven-class FER order for FusDenseNet and explicitly labels the separate nine-class Navrasa baseline.
- Two-view attention and infrared/visible image fusion are not directly reconciled with grayscale FER2013. This implementation uses original and smoothed visible views and declares the substitution.
- Exact pooling operations, depth, growth rates, attention kernel, hidden widths, beta parameterization and preference windows are unspecified. Concrete CPU-sized choices are documented in the mapping.
- A randomized clinical protocol is mentioned, but the paper does not provide enough participant/outcome detail here to reproduce or clinically verify treatment effects. The project makes no therapeutic, disease-management or safety claim.
- One-second playback is described for mood assessment, while another passage describes eight weeks of 30-minute sessions. These descriptions should not be conflated into a single reproducible clinical experiment.
- No full recommendation relevance ground truth, held-out user protocol, negative sampling scheme or ranking-metric table supports a reproduced FDCF quality claim.
- Computational complexity is a stated limitation. Wearables, broader emotions, personalization and clinical trials are future work. EEG/HRV in this project are separate exploratory extensions.

## Bibliography audit

The PDF bibliography contains 43 numbered entries. Some include complete DOI links (e.g. the FER challenge report), while several list generic web domains, incomplete metadata or adjacent subjects. The two Krishna descriptions at references 21 and 37 differ and appear duplicative. Entries 31–35 need independent verification before use as academic authority. This project does not repeat unverified entries as established publications. `LITERATURE_REVIEW.md` cites independently located primary research sources and distinguishes original paper citations from verified project references.

## Reproduction claim

This is a **paper-informed executable implementation of its stated mechanisms**, with a separate pretrained operational baseline. It is not an exact reproduction and does not reproduce 99.72% facial accuracy or clinical improvement.
