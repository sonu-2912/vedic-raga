# Pure FusDenseNet: actual full-network experiment

Completed **9 full-network epochs**. No frozen-feature-only epochs, external encoder, hybrid teacher, external pretrained weights, probability mixture or inference ensemble in this experiment. The training-only projection is removed from exported state. Selected pure model **1,828,896 parameters**, architecture v4. v4 has 1,828,896 parameters; v8 adds 117,360 native residual-dense parameters with the same classifier/input widths. Validation-qualified:False. New pure candidate deployed:False.

Earlier pure baseline; no new test: accuracy **57.42%**, macro F1 **0.5320**, disgust recall **36.71%**. No new candidate qualified for test evaluation; these are historical baseline scores. Active app accuracy 57.42%, macro F1 0.5320, disgust recall 36.71%; active architecture v4. Former hybrid v7 reference: 59.26% accuracy, macro F1 0.5530, disgust recall 43.04%; it is not a pure FusDenseNet score.

## What changed inside the pure training path

All three fusion-dense stages, cross-dense connections, four-head attention, dilated dense convolutions, multi-pooling, early spatial paths and the existing classifier are retained. Full emotion-path weights are optimized, starting from the verified **pre-hybrid** pure checkpoint8fc1713541e49bac8fc6ec4f10a5d5ba4ca9ff3860e227caeda3b7b67870a618. No transfer_backbone or transfer_head keys are accepted. The older Navrasa model and pretrained DenseNet121 are not initialized or consulted by this trainer.

Each64-image training batch draws50natural-frequency images and2additional distinct images per FERclass. Each batch has unique indices and at least2examples per class; identities are unavailable, so cross-identity sampling is not claimed. The same23,503fixed training images are the only pool; repeats across batches are expected. Approximate class prior q=(50p+2)/64; CE uses normalized sqrt(p/q)importance, a disclosed partial cost-sensitive risk. No logit bias/prior offsets are folded into inference. Disgust sampling grows from1.4%natural to4.22%ofdraws; counts for every epoch are recorded, not presented as new people.

A TRAIN-only LayerNorm/588->128->128projection (93,080parameters) receives late fused features. Class-balanced supervised contrastive loss pulls same-labelled TRAIN images together and pushes different labels apart, with self-pairs excluded and temperature.1. It is a project adaptation of Khosla etal.(2020), **Supervised Contrastive Learning**, https://doi.org/10.48550/arXiv.2004.11362 . Projection and optimizer state are stored only for training recovery, not in the inference model. This loss uses the existing FERlabels; no automatic relabelling or claims that labels are verified are made.

RDrop uses mean CE from two independent classifier-dropout draws plus symmetric KL. Because dropout exists only in the classifier and old BatchNorm references are frozen, one deterministic backbone forward and two classifier forwards give the same parameter gradients as two full forwards. This is numerically tested. Basis: Liang etal.(2021), **R-Drop: Regularized Dropout for Neural Networks**, https://doi.org/10.48550/arXiv.2106.14448 . These are training extensions, not exact original-author loss settings. Primary architecture authority remains https://doi.org/10.1007/s41870-025-02531-y .

Objective: mean importance-weighted CE (smoothing.01) +.5symmetricKL +.05class-balancedSupCon. Classifier dropout.4; oldBNstatistics fixed (the preceding calibration audit found no large shift and recalibration worsened validation). AdamW LR8e-5backbone/6e-5classifier/5e-4projection,1epochlinear warmup/cosine, mild5degree affine/3%translation/5%scale and brightness/contrast augmentation. No erasing or mixed-face synthetic training examples.

## Native residual-dense FusDenseNet variant

A second pure recipe adds two dense growth layers (12channels each,3x3dilation1/2), GroupNorm6groups and a zero-initialized1x1localfusion projection inside EACHof6fusionbranches. Each branch still outputs60channels; all attention, concatenation, pooling and classifier input/output widths remain. These are native feature blocks, not a separate backbone/encoder or independently trained predictor. Added117,360params; purev8total1,946,256. Zero residual output preserves initialv4logits, numerically verified; newfeature gradients and exact shared-featureRDrop gradients are tested. GroupNorm has no running batch statistics. Basis: Wu,Y.andHe,K.(2018), GroupNormalization, https://doi.org/10.48550/arXiv.1803.08494 . This depth/normalization choice is an explicit project extension, not recovered author dimensions.

Both recipes independently initialize from the same pre-hybrid purebaseline, not eachother or a hybrid. The v4run budget was capped/reallocated during training before test; actual completedv4epochs:5 (target12was a maximum, not an achievedcount). Actual nativev8epochs:4. Interrupted/unfinished epochs are not counted. The native stage uses backboneLR4e-5/newfeatures3e-4 and a45-minute epoch-boundary budget, maximum6epochs; the other losses/sampler/classifier are retained. This is not an isolated equal-budget architecture ablation because depth/LR/budget differ. Both recipes finish before one candidate is frozen. Full protocols and interruption records are preserved.

## Selection and test discipline

The initial v4protocol used at most12epochs, minimum6before patience6;110-minute wall budget checked at completed-epoch boundaries. Early stopping tracks unconstrained validation objective progress even below deployment floors. Student and EMA are evaluated on all3,970fixed validation faces, including50disgust faces. A checkpoint qualifies only if BOTHvalidation accuracy and disgust recall strictly exceed the earlier pure baseline58.0353%/28%, with macro/disgustF1 not lower. Score.4accuracy+.4macroF1+.2disgustF1. The saved final candidate is frozen before one full test evaluation; no alternate test search.

Inference is **single-view only** with validation temperature calibration; temperature cannot change the predicted class. No flip/crop/ensemble selection. Deployment additionally requires strictly positive TEST accuracy, macroF1, disgust recall and disgustF1 gains over the earlier pure model, specified before test. Former hybrid values are reported separately because the user explicitly prioritizes pure FusDenseNet. An accepted pure model may score below that hybrid on individual metrics; no such regression is hidden. If no candidate qualifies or passes, no improved pure deployment is claimed.

The validation cohort is repeatedly reused and the combined Kaggle test has been historically inspected. These are exploratory descriptive results, not fresh, identity-disjoint confirmation or isolated causal architecture ablations. All7classes and all6,561retained test errors are included. No80%or reliablewebcam result is claimed. Only330unique training disgust images and50validation examples limit minority-class evidence.

## Every completed full-network epoch

Training agreement averages the two dropout logits on augmented, class-paired samples. Validation uses one dropout-disabled raw image. The distributions/conditions differ; this is not a conventional train-validation gap comparison. The raw baseline audit83.01%train/58.04%val and95.45%/28%disgust recall used consistent raw inference conditions on its two cohorts.

| Epoch | Validation accuracy | Macro F1 | Disgust recall | Disgust F1 | Actual disgust draws |
|---|---:|---:|---:|---:|---:|
| v4regularized / 126 | 0.5763 | 0.5199 | 0.2200 | 0.2895 | 977 |
| v4regularized / 127 | 0.5776 | 0.5225 | 0.2600 | 0.2921 | 1015 |
| v4regularized / 128 | 0.5783 | 0.5163 | 0.2000 | 0.2703 | 987 |
| v4regularized / 129 | 0.5801 | 0.5216 | 0.2200 | 0.2895 | 999 |
| v4regularized / 130 | 0.5844 | 0.5245 | 0.2200 | 0.2821 | 986 |
| v8native / 126 | 0.5811 | 0.5237 | 0.2200 | 0.2973 | 1006 |
| v8native / 127 | 0.5849 | 0.5270 | 0.2400 | 0.3038 | 1027 |
| v8native / 128 | 0.5836 | 0.5276 | 0.2400 | 0.3077 | 1008 |
| v8native / 129 | 0.5826 | 0.5252 | 0.2200 | 0.2933 | 991 |

## Validation comparison (Best observed validation diagnostic only; no candidate qualified)

| Metric | Earlier pure | Candidate/diagnostic | Difference |
|---|---:|---:|---:|
| accuracy | 0.580353 | 0.583627 | +0.003275 |
| macro precision | 0.524452 | 0.541768 | +0.017316 |
| macro recall | 0.525654 | 0.523077 | -0.002577 |
| macro f1 | 0.522483 | 0.527586 | +0.005102 |
| disgust precision | 0.333333 | 0.428571 | +0.095238 |
| disgust recall | 0.280000 | 0.240000 | -0.040000 |
| disgust f1 | 0.304348 | 0.307692 | +0.003344 |

## Full test: earlier pure versus new pure

No new test evaluation: no validation-qualified candidate. Earlier pure scores remain reference-only.

| Metric | Earlier pure | New pure | Difference |
|---|---:|---:|---:|


## Full test: former hybrid versus new pure (reference only)

| Metric | Former hybrid | New pure | Difference |
|---|---:|---:|---:|
No new pure test result; no comparison is claimed.

The actual current app checkpoint is SHA8fc1713541e49bac8fc6ec4f10a5d5ba4ca9ff3860e227caeda3b7b67870a618. Selected/trained weight-change audits, executed source hashes and complete protocols are retained in reports/pure_fusdense_20261108/. All originals and earlier failures are preserved. Real EEG/HRV fixed-context tests remain unchanged and show no primary fusionF1gain. No synchronized face/body/raga-response, weather efficacy, therapeutic or recommendation-quality gain is established by this facial experiment.
