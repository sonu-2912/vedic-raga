"""Portable workflow tests use tiny generated fixtures, never scientific results."""
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from backend.datasets import FER2013
from backend.models import FER_CLASSES, FusDenseNet
from scripts.colab_common import check_manifest, device_for, fixed_split
from scripts.import_colab_model import validate_bundle
from scripts.package_colab import notebook

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def colab_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp('colab-fixture')
    dataset = directory / 'fixture.zip'
    rng = np.random.default_rng(52)
    with zipfile.ZipFile(dataset, 'w') as archive:
        for split, count in [('train', 10), ('test', 2)]:
            for label in FER_CLASSES:
                for i in range(count):
                    stream = io.BytesIO()
                    Image.fromarray(rng.integers(0, 256, (48, 48), dtype=np.uint8)).save(stream, format='PNG')
                    archive.writestr(f'{split}/{label}/{i}.png', stream.getvalue())
    manifest = {}
    for split in ['train', 'val']:
        ds = FER2013(dataset, split)
        manifest[split] = [{'file': str(f), 'label': label, 'pixel_sha256': sha}
                           for (f, label), sha in zip(ds.rows, ds.fingerprints())]
        manifest[split + '_count'] = len(ds)
    manifest_path = directory / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    torch.manual_seed(1)
    model = FusDenseNet(version=2)
    checkpoint = directory / 'initial.pt'
    torch.save({'state_dict': model.state_dict(), 'classes': FER_CLASSES, 'synthetic': False,
                'architecture': {'attention': True, 'dilated': True, 'version': 2}, 'epoch': 0,
                'train_count': manifest['train_count'], 'val_count': manifest['val_count'],
                'selection_metric': 'validation macro F1', 'pilot': True}, checkpoint)
    ds = FER2013(dataset, 'test')
    records = [{'file': str(f), 'label': label, 'pixel_sha256': sha}
               for (f, label), sha in zip(ds.rows, ds.fingerprints())]
    baseline_manifest = directory / 'baseline_manifest.json'
    baseline_manifest.write_text(json.dumps({'records': records}))
    baseline_report = directory / 'baseline.json'
    baseline_report.write_text(json.dumps({'accuracy': 0., 'classification_report': {
        'macro avg': {'f1-score': 0.}, 'disgust': {'recall': 0., 'f1-score': 0.}},
        'predictions': {'truth': [r['label'] for r in records]}}))
    return directory, dataset, manifest_path, checkpoint, baseline_report, baseline_manifest


def test_notebook_cells_are_valid_python_and_have_no_executed_results():
    nb = notebook()
    assert nb['metadata']['accelerator'] == 'GPU'
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<colab-cell>', 'exec')
            assert not cell['outputs'] and cell['execution_count'] is None
    text = json.dumps(nb)
    assert 'scripts.colab_train' in text and 'scripts.colab_evaluate' in text


def test_fixed_split_checks_pixel_identity(colab_fixture):
    _, dataset, path, _, _, _ = colab_fixture
    manifest = json.loads(path.read_text()); check_manifest(manifest)
    ds = fixed_split(dataset, manifest, 'val')
    assert len(ds) == manifest['val_count']
    manifest['val'][0]['pixel_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='pixel mismatch'):
        fixed_split(dataset, manifest, 'val')
    manifest['val'][0]['pixel_sha256'] = manifest['train'][0]['pixel_sha256']
    with pytest.raises(ValueError, match='pixel-disjoint'):
        check_manifest(manifest)


def test_cuda_request_fails_clearly_without_gpu():
    if torch.cuda.is_available():
        pytest.skip('Host has CUDA')
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        device_for('cuda')


def test_evaluated_run_cannot_continue_training(tmp_path):
    from scripts.colab_train import ensure_trainable_run
    ensure_trainable_run(tmp_path)
    (tmp_path / 'bundle_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='test-driven continuation'):
        ensure_trainable_run(tmp_path)


def test_train_recover_evaluate_export_import_cpu(colab_fixture, tmp_path, monkeypatch):
    _, data, manifest, initial, baseline, baseline_manifest = colab_fixture
    run = tmp_path / 'run'
    empty_history = tmp_path / 'history.json'; empty_history.write_text('[]')
    command = [sys.executable, '-m', 'scripts.colab_train', '--data', str(data), '--initial', str(initial),
               '--manifest', str(manifest), '--initial-history', str(empty_history), '--run-dir', str(run),
               '--device', 'cpu', '--epochs', '2', '--batch-size', '32', '--max-minutes', '.00001',
               '--patience', '0', '--threads', '2', '--no-amp']
    subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True, timeout=120)
    latest = torch.load(run / 'latest.pt', weights_only=True)
    assert latest['completed_additional_epochs'] == 1
    subprocess.run(command + ['--recover'], check=True, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert torch.load(run / 'latest.pt', weights_only=True)['completed_additional_epochs'] == 2
    evaluation = [sys.executable, '-m', 'scripts.colab_evaluate', '--data', str(data), '--run-dir', str(run),
                  '--baseline-report', str(baseline), '--baseline-manifest', str(baseline_manifest),
                  '--device', 'cpu', '--threads', '2', '--batch-size', '32']
    subprocess.run(evaluation, check=True, cwd=ROOT, capture_output=True, text=True, timeout=120)
    repeated = subprocess.run(evaluation, check=True, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert 'without reading test again' in repeated.stdout
    bundle = run / 'RagaMoodSync_Trained_Model.zip'
    verified, payload, report = validate_bundle(bundle)
    assert report['n'] == 14 and len(report['confusion_matrix']) == 7
    assert hashlib.sha256(payload['models/fusdense.pt']).hexdigest() == verified['checkpoint_sha256']
    assert len(json.loads(payload['reports/training_history.json'])) == 2
    # Exercise installation only in a disposable test project.
    import scripts.import_colab_model as importer
    destination = tmp_path / 'project'
    (destination / 'models').mkdir(parents=True)
    original = b'old model fixture'; (destination / 'models/fusdense.pt').write_bytes(original)
    monkeypatch.setattr(importer, 'ROOT', destination)
    monkeypatch.setattr(sys, 'argv', ['importer', '--bundle', str(bundle), '--apply', '--allow-regression'])
    importer.main()
    assert (destination / 'models/fusdense.pt').read_bytes() == payload['models/fusdense.pt']
    assert next((destination / 'reports/colab_import_backups').glob('*/models/fusdense.pt')).read_bytes() == original


def test_importer_rejects_traversal_before_writing(tmp_path):
    path = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('bundle_manifest.json', json.dumps({'schema': 'ragamoodsync-colab-v1',
                        'files': {'reports/../../outside.py': '0' * 64}}))
    with pytest.raises(ValueError, match='Unsafe bundle path'):
        validate_bundle(path)
