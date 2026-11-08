# FusDenseNet GPU training package

Open `RagaMoodSync_GPU_Training.ipynb` in Google Colab (File > Upload notebook).
Choose Runtime > Change runtime type > GPU, then run the cells in order.

1. Upload `RagaMoodSync_Colab_Project.zip` when the project-upload cell asks.
2. Upload your existing `data/private/fer2013-v1.zip` when the dataset cell asks. The existing Windows file is `C:\Users\bindu\OneDrive\Desktop\VEDIC RAGA\data\private\fer2013-v1.zip`. Alternatively download the train/test image ZIP from https://www.kaggle.com/datasets/msambare/fer2013 after reviewing its access and license. Do not upload the original CSV or a differently named/organized image corpus to this fixed-split continuation notebook.
3. Default training: unchanged FusDenseNet v2, 60 **additional** epochs, batch 128, AdamW 0.0003, inverse-frequency class-weight power 0.75, patience 12, 180-minute epoch-boundary budget. Change the configuration cell before starting if needed. These are training hypotheses, not promised improvements. Reduce batch to 64 if memory is insufficient; use a new run name after changing configuration.
4. Run evaluation once after selecting the model on validation. It reports full unfiltered seven-class metrics and confusion matrix, including disgust, compares the exact same retained test records, and creates `RagaMoodSync_Trained_Model.zip`.
5. Download that ZIP in the final cell. Drive persistence is on by default and retains `latest.pt` every completed epoch for recovery. Rerun the notebook with the same run name/settings after a disconnect; it detects `latest.pt`. If an out-of-memory error interrupts an epoch, the previous completed epoch remains available. No within-epoch recovery or guaranteed Colab availability is claimed.

After test evaluation, the run is locked against further training and repeated evaluation reuses its recorded outputs. This prevents accidentally continuing training after reading that run's test results. A new scientific experiment still requires validation-based decisions and an appropriate independent test protocol.

## What is included

The training package contains current project Python source, frontend source, original 72-raga JSON and upstream README attribution, the existing FusDenseNet checkpoint, frozen train/validation pixel manifests, historical training records and baseline test evidence. It excludes FER images, personal images, SQLite history, secrets, virtual environments, node_modules, upstream Navrasa weights, the paper PDF and generated PowerPoint files. Navrasa is not required for FusDenseNet training. The upstream JSON has no supplied license grant; preserve attribution and resolve redistribution rights before sharing this package publicly. This private package is prepared for the existing project owner's Colab training.

`requirements-colab.txt` deliberately leaves CUDA torch/torchvision to Colab. Do not install the Windows CPU requirements unchanged in Colab. The notebook records actual runtime package versions. No Kaggle login tokens or secrets are required if you upload the dataset ZIP.

GPU trainer: `scripts/colab_train.py`. GPU validation/calibration/test/export: `scripts/colab_evaluate.py`. Existing CPU scripts are preserved unchanged. Training model/batches/loss weights use CUDA; mixed precision uses BF16 when supported, otherwise FP16 with GradScaler. Validation/test use FP32. Model, optimizer, scheduler, scaler and Torch/Python RNG states are saved at epoch boundaries. Backend and dataset augmentation share the original model code; no model is renamed as FusDenseNet.

## Integrate on Windows

Download `RagaMoodSync_Trained_Model.zip` into the existing project's `output` directory. The current workspace already contains the importer. The notebook also downloads a standalone copy as `import_colab_model.py`; if moving to a different project copy, place it in `scripts` first.

Inspect without replacing anything:

```powershell
cd 'C:\Users\bindu\OneDrive\Desktop\VEDIC RAGA'
.\.venv\Scripts\python.exe -m scripts.import_colab_model --bundle output\RagaMoodSync_Trained_Model.zip
```

Install the verified artifacts, with automatic backup of every replaced file:

```powershell
.\.venv\Scripts\python.exe -m scripts.import_colab_model --bundle output\RagaMoodSync_Trained_Model.zip --apply
```

The default deployment rule requires test accuracy and macro F1 to increase while disgust recall does not decrease. A failed rule retains the current model. Read `colab_comparison.json` before considering an explicit `--allow-regression` override. This guard is an engineering choice, not statistical significance or a reason to tune repeatedly on test. Stop your existing backend with Ctrl+C in its terminal and start it again:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The importer validates archive paths/hashes, `weights_only=True` loading, seven-class CPU inference and matching model/evaluation/policy checksums. It backs up old model/reports to `reports/colab_import_backups/<timestamp>`. The existing frontend uses matching metrics and rebuilt validation-based suggestion limits. Full test metrics never exclude abstained images. The app's existing PowerPoint is **not automatically updated**: actual new results must exist before replacing its figures and claims. The downloaded ZIP includes plots, comparison JSON and `COLAB_EXPERIMENTS.md` for that update.

## Scientific limits

Historical baseline: 50.13% test accuracy, macro F1 0.3960; disgust recall 2.53%. The best initial model is retained if validation does not improve. Temperature fitting does not change predicted labels. The fixed validation set is reused for checkpoint/flip/calibration selection. This is not a fresh or identity-disjoint test set, and previous project work has inspected it. No accuracy target is guaranteed. CPU/GPU numerical differences can occur, especially with mixed precision; strict bitwise cross-device reproducibility is not claimed.

This notebook trains facial expression only. It does not fabricate physiological recordings or evaluate EEG/HRV/weather/raga-response fusion. FER2013 lacks those aligned modalities. Real physiological fusion requires a separate properly labelled participant-disjoint experiment.

Sources: [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), [Colab runtime/access limits](https://research.google.com/colaboratory/faq.html), [FER2013](https://www.kaggle.com/datasets/msambare/fer2013), [selected paper](https://doi.org/10.1007/s41870-025-02531-y), and upstream [Raag Ratnakar](https://github.com/alronova/raag-ratnakar).
