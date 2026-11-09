# Project status — 8 November 2026

Training and frozen evaluation completed. The new checkpoint is active in the app.

| Full test, all6,561faces | Previous | Active hybrid |
|---|---:|---:|
| Accuracy |57.42%|59.26%|
| Macro precision |0.5392|0.5628|
| Macro recall |0.5325|0.5526|
| Macro F1 |0.5320|0.5530|
| Disgust precision |0.4833|0.5484|
| Disgust recall |36.71%|43.04%|
| Disgust F1 |0.4173|0.4823|

Disgust true positives29→34of79; false positives31→28. Active validation:60.28%accuracy, macroF1.5497, disgust recall34%. All7test metrics improved. Source checkpoint SHA:a4ed3e54d3b863e6381a8de3ec16428020e58c6d006e48cdfbf72a03ee88c249.

The gain is from a **FusDenseNet + frozen ImageNet DenseNet121 probability-ensemble hybrid (v7)**. It retains the complete v4FusDenseNet path and independently trains the new feature classifier. It is not a pure FusDenseNet or exact author-architecture reproduction. Separate v5full-network spatial training6epochs failed the joint validation gate; residual/direct-short heads8+8epochs also failed. Corrected early stopping allowed42direct-head epochs, yielding the only final candidate. Totals:6full-network +58classifier-only epochs; these are not equivalent compute. One validation-selected candidate was frozen and tested; no alternate test search.

Audit: baseline83.01%TRAIN vs58.04%validation accuracy; disgust95.45%vs28%.330training disgust vs6039happy,50validation disgust. Original attention gradients and fused/spatial paths work; BN recalibration and five-view translation screens failed and were rejected. Data scarcity/overfitting and premature new-head stopping were investigated, without claiming isolated causal architecture effects.

Live health/emotion/recommendation checks passed with the active SHA. Alltest confusion/class metrics recomputed.67source tests passed,15dependency warnings,128.23seconds. Production frontend built. All55original upstream files unchanged. No user photograph used for training or verification; no fabricated results or real-sensor claim. Both pure-model baseline and every failed experiment are preserved.

App and reports updated. Final editable13-slide deck:output/RagaMoodSync_Research_Spatial_Generalization_Reviewed_v3.pptx; default alias:output/RagaMoodSync_Research.pptx. All13rendered slides reviewed, references and architecture text fit; structural/layout checks0findings. Native PowerPoint opening was not verified. Source plots, full experiments and exact evaluation commands:SPATIAL_GENERALIZATION_RESULTS.md and README.md.

Limitations: accuracy/disgust recall remain modest; no validated webcam, clinical, weather/raga-outcome or joint face/physiology efficacy. Historically inspected test and repeatedly reused validation prevent fresh/identity-disjoint confirmation. Real fixed-game/30s/participant-held-out EEG/ECG tests show no primary fusion F1gain; DENS/DREAMER remain access/source-limited. Source annotations missing/unverified remain marked. GitHub publication complete and remote verified: 56d7eec99780a5542a640860f15da383db51b178. Filtered publication passed67tests (184.62seconds) and production frontend build; working tree clean. Archive:output/github/RagaMoodSync_GitHub_Ready_Spatial.zip. Receipt:reports/spatial_publication_receipt.json.
