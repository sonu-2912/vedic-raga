"""Portable Colab helpers; no imports of Google APIs or automatic downloads."""
import contextlib
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from backend.datasets import FER2013
from backend.models import FER_CLASSES, FusDenseNet


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    """CPU tensors and primitive metadata work with Windows weights_only=True."""
    def cpu(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu()
        if isinstance(item, dict):
            return {k: cpu(v) for k, v in item.items()}
        if isinstance(item, list):
            return [cpu(v) for v in item]
        if isinstance(item, tuple):
            return tuple(cpu(v) for v in item)
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(cpu(value), temporary)
    temporary.replace(path)


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. In Colab select Runtime > Change runtime type > GPU.')
    return torch.device(name)


def model_from(checkpoint, device):
    if checkpoint.get('synthetic') or checkpoint.get('classes') != FER_CLASSES:
        raise ValueError('A real-data seven-class FER checkpoint is required')
    model = FusDenseNet(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def fixed_split(data, manifest, split, augment=False):
    """Verify every pixel hash, not just counts; never use test to build a split."""
    ds = FER2013(data, split, augment=augment)
    expected = manifest[split]
    rows = {str(file): (file, label) for file, label in ds.rows}
    if len({r['file'] for r in expected}) != len(expected):
        raise ValueError('Duplicate file names in fixed manifest')
    if any(r['file'] not in rows for r in expected):
        raise ValueError('Dataset filenames differ from the supplied fixed split. Upload the existing fer2013-v1.zip.')
    ds.rows = [rows[r['file']] for r in expected]
    ds.cache_images()
    for row, expected_row, pixel_hash in zip(ds.rows, expected, ds.fingerprints()):
        if row[1] != expected_row['label'] or pixel_hash != expected_row['pixel_sha256']:
            raise ValueError(f'Dataset label/pixel mismatch: {row[0]}')
    return ds


def check_manifest(manifest):
    train = {r['pixel_sha256'] for r in manifest['train']}
    val = {r['pixel_sha256'] for r in manifest['val']}
    if train & val or len(train) != len(manifest['train']) or len(val) != len(manifest['val']):
        raise ValueError('Training and validation must be pixel-disjoint and deduplicated')


def predictions(model, ds, device, batch_size=128, flip=False):
    model.eval()
    outputs, labels = [], []
    with torch.inference_mode():
        for x, y in DataLoader(ds, batch_size=batch_size, pin_memory=device.type == 'cuda'):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            if flip:
                logits = (logits + model(x.flip(-1))) / 2
            outputs.append(logits.float().cpu())
            labels.append(y)
    return torch.cat(outputs), torch.cat(labels)


def metrics(logits, labels, temperature=1.):
    probabilities = (logits / temperature).softmax(1).numpy()
    truth = labels.numpy()
    pred = probabilities.argmax(1)
    report = classification_report(truth, pred, labels=list(range(7)), target_names=FER_CLASSES,
                                   output_dict=True, zero_division=0)
    return {'n': len(truth), 'accuracy': float(accuracy_score(truth, pred)),
            'balanced_accuracy': float(report['macro avg']['recall']),
            'classification_report': report,
            'confusion_matrix': confusion_matrix(truth, pred, labels=list(range(7))).tolist(),
            'negative_log_likelihood': float(-np.log(np.maximum(probabilities[np.arange(len(truth)), truth], 1e-12)).mean()),
            'predictions': {'truth': truth.tolist(), 'predicted': pred.tolist()}}


def autocast_for(device, enabled):
    if device.type != 'cuda' or not enabled:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast('cuda', dtype=dtype)

def set_training_mode(model, freeze_batchnorm=False):
    """Keep dropout/gradients active while optionally retaining BN references."""
    model.train()
    if freeze_batchnorm:
        for module in model.modules():
            if isinstance(module,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.BatchNorm3d)):
                module.eval()
