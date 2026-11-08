"""Inspect/apply a Colab model bundle, backing up all replaced local artifacts."""
import argparse
import hashlib
import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

import torch
from backend.models import FusDenseNet, FER_CLASSES

ROOT = Path(__file__).resolve().parents[1]


def validate_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive entries')
        if sum(m.file_size for m in archive.infolist()) > 100 * 1024**2:
            raise ValueError('Unexpectedly large model bundle')
        manifest = json.loads(archive.read('bundle_manifest.json'))
        if manifest.get('schema') != 'ragamoodsync-colab-v1':
            raise ValueError('Unsupported bundle schema')
        payload = {}
        for relative, expected in manifest['files'].items():
            p = PurePosixPath(relative)
            if p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative or p.parts[0] not in ('models', 'reports'):
                raise ValueError('Unsafe bundle path')
            if relative.startswith('models/') and relative != 'models/fusdense.pt':
                raise ValueError('Unexpected model destination')
            raw = archive.read(relative)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f'Checksum mismatch: {relative}')
            payload[relative] = raw
        checkpoint = torch.load(io.BytesIO(payload['models/fusdense.pt']), map_location='cpu', weights_only=True)
        if checkpoint.get('synthetic') or checkpoint.get('classes') != FER_CLASSES:
            raise ValueError('Synthetic/incompatible checkpoint')
        model = FusDenseNet(**checkpoint['architecture'])
        model.load_state_dict(checkpoint['state_dict']); model.eval()
        with torch.inference_mode():
            logits = model(torch.zeros(1, 1, 48, 48))
        if logits.shape != (1, 7) or not torch.isfinite(logits).all():
            raise ValueError('CPU inference failed')
        actual_sha = hashlib.sha256(payload['models/fusdense.pt']).hexdigest()
        evaluation = json.loads(payload['reports/emotion_evaluation.json'])
        if actual_sha != manifest['checkpoint_sha256'] or evaluation['checkpoint_sha256'] != actual_sha:
            raise ValueError('Model/evaluation provenance mismatch')
        if json.loads(payload['reports/expression_policy.json'])['checkpoint_sha256'] != actual_sha:
            raise ValueError('Expression policy provenance mismatch')
        return manifest, payload, evaluation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--apply', action='store_true', help='Back up and install verified model/reports')
    p.add_argument('--allow-regression', action='store_true', help='Explicitly permit a bundle that fails the deployment metric rule')
    a = p.parse_args()
    manifest, payload, report = validate_bundle(a.bundle)
    print(json.dumps({'verified': True, 'accuracy': report['accuracy'], 'macro_f1': report['classification_report']['macro avg']['f1-score'], 'recommended_for_integration': manifest['recommended_for_integration']}, indent=2))
    if not a.apply:
        print('Inspection only. Add --apply to install with an automatic backup.'); return
    if not manifest['recommended_for_integration'] and not a.allow_regression:
        raise ValueError('Accuracy/macro F1/disgust deployment check failed. Current model retained. Read the comparison report before considering --allow-regression.')
    backup = ROOT / 'reports/colab_import_backups' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    for relative in payload:
        if not (ROOT / relative).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Destination resolves outside project; no files replaced')
    for relative in payload:
        destination = ROOT / relative
        if destination.exists():
            saved = backup / relative; saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
    for relative, raw in payload.items():
        destination = ROOT / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.importing.tmp')
        temporary.write_bytes(raw); temporary.replace(destination)
    print(f'Installed. Previous files backed up under {backup}. Restart the backend. See reports/COLAB_EXPERIMENTS.md; the existing PPT is not automatically rewritten.')


if __name__ == '__main__':
    main()
