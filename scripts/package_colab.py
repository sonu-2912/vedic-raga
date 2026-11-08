"""Build the self-contained notebook and a curated, private-data-free source ZIP."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook():
    cells = []
    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().splitlines(keepends=True)})
    def code(text):
        text = text.strip() + '\n'
        compile(text, '<notebook-cell>', 'exec')
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(keepends=True),
                      'execution_count': None, 'outputs': []})
    md('''# RagaMoodSync: FusDenseNet GPU training and Windows export

**Run cells top to bottom.** Select **Runtime → Change runtime type → GPU** first.
You need two files: `RagaMoodSync_Colab_Project.zip` supplied with this notebook, and your existing `fer2013-v1.zip` dataset (or the same Kaggle v1 image ZIP). No manual source editing is needed.

The unchanged FusDenseNet v2 continues from the supplied checkpoint. Best selection uses fixed validation macro F1, including all seven classes. Disgust recall/F1 are displayed. Test evaluation happens only after training, without filtering uncertain predictions. GPU speed does not guarantee higher accuracy.

Default checkpoints persist to your own Google Drive. Colab will ask you to authorize mounting it; this notebook does not upload files to another account. [Colab resources and runtimes are limited](https://research.google.com/colaboratory/faq.html).

This is **facial-expression training only**. It does not claim real EEG/HRV fusion or raga-response results. Source paper: [Pingle & Ragha, 2025](https://doi.org/10.1007/s41870-025-02531-y).''')
    md('## 1 · Configuration\nDefaults work without changes. For a different training recipe, choose a **new RUN_NAME**. EPOCHS means total additional epochs for this run, including any recovered epochs. Use batch 64 with a new run name if GPU memory is insufficient.')
    code('''from pathlib import Path
import os, sys, json, shutil, subprocess, zipfile, hashlib

PROJECT_ZIP = ""   # Optional existing path; otherwise the next cell opens upload.
DATASET_ZIP = ""   # Optional path on Drive; otherwise the dataset cell opens upload.
PERSIST_TO_DRIVE = True
RUN_NAME = "fer_v2_gpu_run_01"
EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 0.0003
CLASS_WEIGHT_POWER = 0.75
PATIENCE = 12
MAX_MINUTES = 180
USE_MIXED_PRECISION = True
ROOT = Path('/content/ragamoodsync')
''')
    md('## 2 · Verify GPU\nThe notebook stops here if no CUDA GPU is available. It does not silently run a long CPU training job.')
    code('''import torch
print('PyTorch:', torch.__version__)
assert torch.cuda.is_available(), 'Enable GPU: Runtime > Change runtime type > GPU, then rerun.'
DEVICE = 'cuda'
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('GPU memory GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
''')
    md('## 3 · Upload the supplied project ZIP\nChoose **RagaMoodSync_Colab_Project.zip**, not the original Raag Ratnakar ZIP. No dataset, personal photos, SQLite history or credentials are bundled.')
    code('''from google.colab import files
if not PROJECT_ZIP:
    print('Upload RagaMoodSync_Colab_Project.zip')
    uploaded = files.upload()
    assert len(uploaded) == 1, 'Select only the project ZIP.'
    PROJECT_ZIP = next(iter(uploaded))
    del uploaded
ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(PROJECT_ZIP) as z:
    assert sum(m.file_size for m in z.infolist()) < 100 * 1024**2, 'Unexpected project archive size.'
    for m in z.infolist():
        target = (ROOT / m.filename).resolve()
        assert target.is_relative_to(ROOT.resolve()), 'Unsafe archive path.'
    z.extractall(ROOT)
package = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
for relative, expected in package['files'].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
assert (ROOT / 'scripts/colab_train.py').exists()
os.chdir(ROOT)
print('Verified project files:', len(package['files']))
print('Baseline checkpoint:', package['baseline_checkpoint_sha256'])
''')
    md('## 4 · Install training dependencies\nColab’s existing **CUDA PyTorch and matching torchvision are preserved**. Do not install the Windows CPU requirements file in this runtime. If Colab asks for a runtime restart, restart, then rerun cells 1–4 and continue.')
    code('''subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                '--upgrade-strategy', 'only-if-needed', '-r', 'requirements-colab.txt'], check=True)
import torchvision
import numpy, pandas, scipy, sklearn, cv2, PIL, matplotlib
assert torch.cuda.is_available()
print('torch / torchvision:', torch.__version__, torchvision.__version__)
print('NumPy:', numpy.__version__, 'scikit-learn:', sklearn.__version__)
''')
    md('## 5 · Choose checkpoint storage\nDrive is recommended so completed epochs survive disconnection. A repeated run with the same name automatically recovers `latest.pt`. To start over, choose a new name; no previous run is deleted.')
    code('''assert RUN_NAME and '/' not in RUN_NAME and '\\\\' not in RUN_NAME and RUN_NAME not in ('.', '..')
if PERSIST_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    RUN_DIR = Path('/content/drive/MyDrive/RagaMoodSync_Colab') / RUN_NAME
else:
    RUN_DIR = Path('/content/ragamoodsync_runs') / RUN_NAME
    print('WARNING: Local runtime files can disappear when Colab disconnects. Download outputs promptly.')
RUN_DIR.mkdir(parents=True, exist_ok=True)
environment = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
(RUN_DIR / 'colab_environment.txt').write_text(environment)
print('Run directory:', RUN_DIR)
''')
    md('''## 6 · Upload FER2013 and check its format
Upload the existing `data/private/fer2013-v1.zip` from your Windows project. Alternatively obtain the same image ZIP from [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013), reviewing its terms. No automatic large download or Kaggle credentials are used.

The ZIP must contain `train/angry/...jpg`, etc., and `test/...`, directly at its root. CSV, renamed images and unrelated FER copies are not interchangeable with the supplied fixed manifest. Training verifies every retained train/validation pixel hash before updating weights. Images are read from local Colab disk rather than directly from Drive.''')
    code('''if not DATASET_ZIP:
    print('Upload fer2013-v1.zip (the train/test image ZIP, about 63 MB)')
    uploaded = files.upload()
    assert len(uploaded) == 1, 'Select only the FER2013 dataset ZIP.'
    DATASET_ZIP = str(Path(next(iter(uploaded))).resolve())
    del uploaded
DATA = ROOT / 'data/private/fer2013-v1.zip'
DATA.parent.mkdir(parents=True, exist_ok=True)
if Path(DATASET_ZIP).resolve() != DATA.resolve():
    assert shutil.disk_usage('/content').free > Path(DATASET_ZIP).stat().st_size + 1024**3, 'Insufficient disk space.'
    shutil.copy2(DATASET_ZIP, DATA)
with zipfile.ZipFile(DATA) as z:
    names = z.namelist()
    for split in ['train', 'test']:
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
            assert any(n.startswith(f'{split}/{emotion}/') and n.lower().endswith(('.jpg', '.jpeg', '.png')) for n in names), f'Missing {split}/{emotion}; use the existing fer2013-v1.zip.'
print('Dataset format accepted. Pixel/split identity is checked by the trainer.')
''')
    md('## 7 · Short GPU forward/backward check\nThis uses random tensors to check GPU plumbing, **not a training/evaluation result**. It does not modify the supplied checkpoint.')
    code('''from backend.models import FusDenseNet
from scripts.colab_common import autocast_for
torch.set_num_threads(2)
probe = FusDenseNet(version=2).cuda().train()
with autocast_for(torch.device('cuda'), USE_MIXED_PRECISION):
    probe_logits = probe(torch.rand(2, 1, 48, 48, device='cuda'))
    probe_loss = torch.nn.functional.cross_entropy(probe_logits, torch.tensor([0, 1], device='cuda'))
probe_loss.backward()
assert torch.isfinite(probe_loss)
assert all(torch.isfinite(p.grad).all() for p in probe.parameters() if p.grad is not None)
del probe, probe_logits, probe_loss
torch.cuda.empty_cache()
print('GPU forward/backward passed. This is a synthetic plumbing check, not model accuracy.')
''')
    md('''## 8 · Train / recover automatically
The source architecture is unchanged. New recipe: AdamW, class-weight power 0.75, smoothing 0.02, reconstruction weight 0.05 and cosine learning-rate schedule. These choices must be validated; no target accuracy is promised.

`best.pt` is selected by **validation macro F1**, with all classes included. The initial model is retained if no validation improvement occurs. `latest.pt` contains optimizer/scheduler/scaler and Torch/Python RNG states for completed-epoch recovery. The historical CPU phase used a different recipe; this is disclosed in the report. Stop this cell normally to interrupt; do not reset the runtime before a completed checkpoint is saved.''')
    code('''def run_logged(arguments, log_name):
    with (RUN_DIR / log_name).open('a', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', *map(str, arguments)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='')
                log.write(line); log.flush()
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate(); process.wait()
            raise
        if returncode:
            raise RuntimeError(f'Command failed ({returncode}); see {log_name}. No success is claimed.')

command = ['-m', 'scripts.colab_train', '--data', DATA,
           '--initial', ROOT / 'models/fusdense.pt',
           '--manifest', ROOT / 'reports/fer_training_manifest.json',
           '--initial-history', ROOT / 'reports/training_history.json',
           '--run-dir', RUN_DIR, '--device', DEVICE, '--epochs', EPOCHS,
           '--batch-size', BATCH_SIZE, '--lr', LEARNING_RATE,
           '--weight-power', CLASS_WEIGHT_POWER, '--patience', PATIENCE,
           '--max-minutes', MAX_MINUTES, '--threads', 2]
if not USE_MIXED_PRECISION:
    command.append('--no-amp')
if (RUN_DIR / 'latest.pt').exists():
    command.append('--recover')
    print('Recovering the last completed epoch; configuration must match.')
elif (RUN_DIR / 'best.pt').exists():
    raise RuntimeError('Initialization was interrupted before latest.pt was saved. Choose a new RUN_NAME; existing outputs are preserved.')
if (RUN_DIR / 'bundle_manifest.json').exists():
    print('This run has already been evaluated. Training is locked; reuse the evaluated download below.')
else:
    run_logged(command, 'training_console.log')
''')
    md('## 9 · Inspect validation before test\nUse validation for further recipe decisions. Do not repeatedly adjust the recipe after looking at test results. The next cell evaluates the already-selected checkpoint.')
    code('''import pandas as pd
from IPython.display import display, Image
history = pd.read_json(RUN_DIR / 'training_history.json')
display(history.tail(12))
best = torch.load(RUN_DIR / 'best.pt', map_location='cpu', weights_only=True)
print('Selected epoch:', best['epoch'])
print('Validation macro F1:', best['validation_macro_f1'])
print('Improved over initial validation:', best.get('colab_improved', False))
if 'val_disgust_recall' in history:
    display(history[['epoch', 'val_accuracy', 'val_macro_f1', 'val_disgust_recall', 'val_disgust_f1']].tail(12))
''')
    md('''## 10 · Calibrate, evaluate all test images and build download
Calibration/flip selection use validation only. The complete retained test set is evaluated with no confidence filtering. Exact pixel/file records must match the bundled baseline; otherwise comparison fails instead of reporting a misleading gain.

The original test set has been inspected in prior project work; it is not a new independent benchmark. Existing results are reused if this exact selected checkpoint has already been evaluated. Changing the selected checkpoint afterward requires a new independent evaluation plan, not repeated test-driven tuning.''')
    code('''run_logged(['-m', 'scripts.colab_evaluate', '--data', DATA, '--run-dir', RUN_DIR,
            '--device', DEVICE, '--batch-size', BATCH_SIZE, '--threads', 2], 'evaluation_console.log')
report_dir = RUN_DIR / 'deployment/reports'
comparison = json.loads((report_dir / 'colab_comparison.json').read_text())
display(pd.DataFrame(comparison['metrics']).T)
report = json.loads((report_dir / 'emotion_evaluation.json').read_text())
display(pd.DataFrame({c: report['classification_report'][c] for c in
                     ['angry','disgust','fear','happy','sad','surprise','neutral']}).T)
print('Full test images:', report['n'])
print('Recommended for integration:', comparison['recommended_for_integration'])
print('No recommendation means retain the local model and review regressions; it does not mean hide them.')
display(Image(filename=str(report_dir / 'emotion_evaluation.png')))
display(Image(filename=str(report_dir / 'training_curves.png')))
''')
    md('## 11 · Download the CPU-compatible model and results\nThe ZIP contains the trained checkpoint, matching evaluation/policy hashes, confusion matrix, curves, training history, comparison and Markdown report. It contains **no FER images**. Your complete runtime environment and recovery checkpoints remain in RUN_DIR.')
    code('''BUNDLE = RUN_DIR / 'RagaMoodSync_Trained_Model.zip'
assert BUNDLE.exists()
files.download(str(BUNDLE))
print('Saved bundle:', BUNDLE)
print('If your browser blocks the download, use Colab Files or retrieve it from the run folder on Drive.')
# Also provide the standalone importer for another copy of the Windows project.
files.download(str(ROOT / 'scripts/import_colab_model.py'))
''')
    md('''## 12 · Install in the Windows project
Save `RagaMoodSync_Trained_Model.zip` in the project's `output` folder. The existing project already includes the importer. If using another project copy, place the separately downloaded `import_colab_model.py` inside `scripts` first.

Run in PowerShell:
```powershell
cd 'C:\\Users\\bindu\\OneDrive\\Desktop\\VEDIC RAGA'
.\\.venv\\Scripts\\python.exe -m scripts.import_colab_model --bundle output\\RagaMoodSync_Trained_Model.zip
.\\.venv\\Scripts\\python.exe -m scripts.import_colab_model --bundle output\\RagaMoodSync_Trained_Model.zip --apply
```
The importer validates hashes and CPU inference, backs up replaced files, and installs only if accuracy and macro F1 improve and disgust recall does not decrease. Failed checks preserve the current model. `--allow-regression` is an explicit manual override after reviewing the report, not the default.

Restart your existing backend (Ctrl+C in its terminal, then your usual uvicorn command). No CUDA installation is needed on Windows. Metrics and expression-policy provenance match the imported checkpoint. Abstention does not affect the reported full test accuracy.

The existing PPT is **not rewritten with unmeasured numbers**. Use the downloaded actual plots and `COLAB_EXPERIMENTS.md` for its results update after training. Real EEG/HRV fusion remains a separate, unperformed experiment requiring aligned labelled recordings.

References: [FER2013 access](https://www.kaggle.com/datasets/msambare/fer2013), [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), [Colab FAQ](https://research.google.com/colaboratory/faq.html), [RagaMoodSync paper](https://doi.org/10.1007/s41870-025-02531-y), [upstream Raag Ratnakar](https://github.com/alronova/raag-ratnakar).''')
    for i, cell in enumerate(cells):
        cell['id'] = f'ragamood-{i:02d}'
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'accelerator': 'GPU', 'colab': {'name': 'RagaMoodSync_GPU_Training.ipynb', 'provenance': []},
                         'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.11'}}, 'cells': cells}


def main():
    output = ROOT / 'output'; output.mkdir(exist_ok=True)
    nb_path = output / 'RagaMoodSync_GPU_Training.ipynb'
    nb_path.write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    selected = set()
    for directory in ['backend', 'scripts', 'tests']:
        selected.update(f for f in (ROOT / directory).glob('*.py'))
    for pattern in ['*.md', 'requirements*.txt', 'pytest.ini']:
        selected.update(ROOT.glob(pattern))
    selected.update(f for f in (ROOT / 'frontend/src').rglob('*') if f.is_file())
    selected.update(ROOT / 'frontend' / name for name in ['package.json', 'package-lock.json', 'index.html', 'vite.config.js'])
    selected.update(ROOT / relative for relative in [
        'external/raag-ratnakar-main/raga_sthayi_bhava_dataset_enriched.json',
        'external/raag-ratnakar-main/README.md', 'models/fusdense.pt',
        'reports/fer_training_manifest.json', 'reports/training_history.json',
        'reports/emotion_evaluation.json', 'reports/emotion_evaluation_manifest.json',
        'reports/expression_policy.json', 'reports/calibration.json',
        'reports/emotion_evaluation.png', 'reports/training_curves.png'])
    selected = sorted(f for f in selected if f.is_file())
    filemap = {str(f.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(f.read_bytes()).hexdigest() for f in selected}
    filemap['RagaMoodSync_GPU_Training.ipynb'] = hashlib.sha256(nb_path.read_bytes()).hexdigest()
    manifest = {'schema': 'ragamoodsync-colab-project-v1', 'files': filemap,
                'baseline_checkpoint_sha256': filemap['models/fusdense.pt'],
                'excluded': ['datasets', 'user images', 'SQLite history', 'credentials', 'virtual environments', 'upstream Navrasa weights', 'paper PDF', 'PowerPoint outputs'],
                'cuda_execution_verified_locally': False}
    zip_path = output / 'RagaMoodSync_Colab_Project.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in selected: archive.write(file, file.relative_to(ROOT))
        archive.write(nb_path, 'RagaMoodSync_GPU_Training.ipynb')
        archive.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2))
    (output / 'RagaMoodSync_Colab_Package_Manifest.json').write_text(json.dumps({**manifest, 'archive_sha256': hashlib.sha256(zip_path.read_bytes()).hexdigest(), 'archive_bytes': zip_path.stat().st_size}, indent=2))
    print(json.dumps({'notebook': str(nb_path), 'project_zip': str(zip_path), 'zip_bytes': zip_path.stat().st_size, 'files': len(filemap)}))


if __name__ == '__main__':
    main()
