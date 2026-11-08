"""CPU/GPU continuation of paper-informed FusDenseNet variants with fixed-split provenance.

Best model is selected by validation macro F1. latest.pt additionally stores
optimizer/scheduler/scaler/RNG state for epoch-boundary recovery on Drive.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from sklearn.metrics import classification_report

from backend.models import FER_CLASSES
from scripts.colab_common import (autocast_for, check_manifest, device_for, digest,
                                 fixed_split, model_from, predictions, save_checkpoint, write_json,set_training_mode)


def ensure_trainable_run(run):
    if (Path(run) / 'bundle_manifest.json').exists():
        raise ValueError('This run has completed test evaluation. Training is locked to avoid test-driven continuation. Reuse the evaluated export.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--initial', default='models/fusdense.pt')
    p.add_argument('--manifest', default='reports/fer_training_manifest.json')
    p.add_argument('--initial-history', default='reports/training_history.json')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--epochs', type=int, default=60, help='Total additional epochs for this run, including resumed epochs')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-power', type=float, default=.75)
    p.add_argument('--sampler-power', type=float, default=0., help='Training-only inverse-frequency sampling power; 0 keeps ordinary shuffle')
    p.add_argument('--freeze-bn',action='store_true',help='Retain BatchNorm running references while training affine weights and all other parameters')
    p.add_argument('--mild-augmentation',action='store_true',help='Small affine changes without random cropping, for spatial-feature fine-tuning')
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--max-minutes', type=float, default=180)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--recover', action='store_true')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--smoke-batches', type=int, default=0, help='Plumbing test only; checkpoint marked synthetic and cannot deploy')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 2 or not 0 <= a.weight_power <= 1 or not 0<=a.sampler_power<=1:
        raise ValueError('Positive epochs, batch size >=2, weight-power in [0,1] required')
    torch.set_num_threads(a.threads)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = device_for(a.device)
    run = Path(a.run_dir); run.mkdir(parents=True, exist_ok=True)
    ensure_trainable_run(run)
    manifest = json.loads(Path(a.manifest).read_text())
    check_manifest(manifest)
    manifest_hash, dataset_hash = digest(a.manifest), digest(a.data)
    train = fixed_split(a.data, manifest, 'train', augment=True)
    val = fixed_split(a.data, manifest, 'val')
    ck = torch.load(a.initial, map_location='cpu', weights_only=True)
    if ck.get('architecture', {}).get('version') not in [2,3,4]:
        raise ValueError('This trainer continues compatible project FusDenseNet v2/v3/v4')
    if ck.get('train_count') != len(train) or ck.get('val_count') != len(val):
        raise ValueError('Checkpoint and verified fixed split differ')
    if a.mild_augmentation:
        from torchvision import transforms
        train.transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomAffine(5,translate=(.02,.02),scale=(.98,1.02),interpolation=transforms.InterpolationMode.BILINEAR),transforms.ToTensor()])
    model = model_from(ck, device)
    # Do not retain temperature/TTA from the old model after updating weights.
    base = {k: v for k, v in ck.items() if k not in ('state_dict', 'inference', 'calibration')}
    base['training_args'] = vars(a)
    base['resume_provenance'] = {'checkpoint_sha256': digest(a.initial), 'source_epoch': ck['epoch'],
                                 'optimizer': 'AdamW restarted for Colab recipe; latest.pt supports subsequent epoch-boundary recovery'}
    old_history = json.loads(Path(a.initial_history).read_text()) if Path(a.initial_history).exists() else []
    offset = max([row['epoch'] for row in old_history], default=ck.get('epoch', 0))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and a.amp and not torch.cuda.is_bf16_supported())
    counts = torch.bincount(torch.tensor([y for _, y in train.rows]), minlength=7).float()
    if (counts == 0).any():
        raise ValueError('Training split must contain all seven classes')
    weights = (counts.sum() / (7 * counts)).pow(a.weight_power)
    weights = (weights / weights.mean()).to(device)
    settings = {'architecture': ck['architecture'], 'seed': 42, 'dataset_sha256': dataset_hash,
                'manifest_sha256': manifest_hash, 'initial_checkpoint_sha256': digest(a.initial),
                'epochs': a.epochs, 'lr': a.lr, 'weight_power': a.weight_power,'sampler_power':a.sampler_power,'freeze_bn':a.freeze_bn,'mild_augmentation':a.mild_augmentation,
                'batch_size': a.batch_size, 'amp': a.amp, 'smoke_batches': a.smoke_batches,
                'loss': 'weighted CE (label smoothing .02) + .05 reconstruction MSE',
                'selection': 'validation macro F1; no test access', 'class_counts': counts.int().tolist(),
                'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
                'warm_start': 'Model weights from previous best; optimizer restarted for new recipe'}
    history = []; start_epoch = 0; no_improvement = 0
    if a.recover:
        latest = torch.load(run / 'latest.pt', map_location='cpu', weights_only=True)
        for key in ['manifest_sha256', 'dataset_sha256', 'initial_checkpoint_sha256', 'epochs', 'lr', 'weight_power', 'batch_size', 'amp', 'smoke_batches']:
            if latest['settings'][key] != settings[key]:
                raise ValueError(f'Recovery setting changed: {key}. Keep settings or start a new run directory.')
        if latest['settings'].get('sampler_power',0.)!=a.sampler_power:raise ValueError('Recovery sampler setting changed')
        if latest['settings'].get('freeze_bn',False)!=a.freeze_bn:raise ValueError('Recovery BatchNorm setting changed')
        if latest['settings'].get('mild_augmentation',False)!=a.mild_augmentation:raise ValueError('Recovery augmentation setting changed')
        model.load_state_dict(latest['state_dict']); opt.load_state_dict(latest['optimizer'])
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value): state[key] = value.to(device)
        scheduler.load_state_dict(latest['scheduler']); scaler.load_state_dict(latest['scaler'])
        history = latest['history']; start_epoch = latest['completed_additional_epochs']
        best = latest['best']; no_improvement = latest['no_improvement']
        torch.set_rng_state(latest['torch_rng']); random.setstate(latest['python_rng'])
        if device.type == 'cuda' and latest['cuda_rng']:
            torch.cuda.set_rng_state_all(latest['cuda_rng'])
    else:
        if (run / 'latest.pt').exists() or (run / 'best.pt').exists():
            raise ValueError('Run directory already has checkpoints. Use --recover or a new run directory.')
        initial_logits, initial_labels = predictions(model, val, device, a.batch_size)
        initial_report = classification_report(initial_labels, initial_logits.argmax(1), labels=list(range(7)),
                                              target_names=FER_CLASSES, output_dict=True, zero_division=0)
        best = float(initial_report['macro avg']['f1-score'])
        write_json(run / 'initial_validation.json', initial_report)
        # Seed best with uncalibrated initial weights. No improvement still yields a valid artifact.
        save_checkpoint(run / 'best.pt', {**base, 'state_dict': model.state_dict(), 'inference': {},
                        'synthetic': bool(a.smoke_batches), 'validation_macro_f1': best,
                        'colab_settings': settings, 'colab_improved': False})
    write_json(run / 'settings.json', settings)
    write_json(run / 'fer_training_manifest.json', manifest)
    sampling_weights=counts.pow(-a.sampler_power)[torch.tensor([y for _,y in train.rows])].double()
    sampler=WeightedRandomSampler(sampling_weights,len(train),replacement=True) if a.sampler_power else None
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=sampler is None, sampler=sampler, pin_memory=device.type == 'cuda', num_workers=0)
    began = time.perf_counter()
    print(json.dumps({'device': settings['device'], 'train': len(train), 'val': len(val), 'start_additional_epoch': start_epoch, 'initial_or_recovered_best_macro_f1': best}), flush=True)
    for step_epoch in range(start_epoch, a.epochs):
        started = time.perf_counter();set_training_mode(model,a.freeze_bn); loss_sum = 0.; correct = 0; seen = 0
        for batch_index, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_for(device, a.amp):
                logits, reconstruction = model(x, return_reconstruction=True)
                loss = torch.nn.functional.cross_entropy(logits, y, weight=weights, label_smoothing=.02) + .05 * torch.nn.functional.mse_loss(reconstruction, x)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; last completed epoch remains saved. Retry with --no-amp in a new run directory.')
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.detach()) * len(y); correct += int((logits.argmax(1) == y).sum()); seen += len(y)
            if (batch_index + 1) % 50 == 0:
                print(json.dumps({'epoch': offset + step_epoch + 1, 'batch': batch_index + 1, 'batches': len(loader), 'loss': loss_sum / seen}), flush=True)
            if a.smoke_batches and batch_index + 1 >= a.smoke_batches: break
        validation_logits, labels = predictions(model, val, device, a.batch_size)
        report = classification_report(labels, validation_logits.argmax(1), labels=list(range(7)), target_names=FER_CLASSES, output_dict=True, zero_division=0)
        row = {'epoch': offset + step_epoch + 1, 'loss': loss_sum / seen, 'accuracy': correct / seen,
               'val_loss': float(torch.nn.functional.cross_entropy(validation_logits, labels)),
               'val_accuracy': float((validation_logits.argmax(1) == labels).float().mean()),
               'val_macro_f1': float(report['macro avg']['f1-score']), 'val_balanced_accuracy': float(report['macro avg']['recall']),
               'val_disgust_recall': float(report['disgust']['recall']), 'val_disgust_f1': float(report['disgust']['f1-score']),
               'learning_rate': opt.param_groups[0]['lr'], 'seconds': time.perf_counter() - started}
        history.append(row)
        improved = row['val_macro_f1'] > best
        if improved:
            best = row['val_macro_f1']; no_improvement = 0
            save_checkpoint(run / 'best.pt', {**base, 'state_dict': model.state_dict(), 'epoch': row['epoch'],
                            'synthetic': bool(a.smoke_batches), 'validation_macro_f1': best,
                            'colab_settings': settings, 'colab_improved': True, 'inference': {}})
            write_json(run / 'best_validation.json', report)
        else:
            no_improvement += 1
        scheduler.step()
        save_checkpoint(run / 'latest.pt', {'state_dict': model.state_dict(), 'optimizer': opt.state_dict(),
                        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'settings': settings,
                        'completed_additional_epochs': step_epoch + 1, 'best': best, 'no_improvement': no_improvement,
                        'history': history, 'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                        'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
        write_json(run / 'training_history.json', old_history + history)
        print(json.dumps({**row, 'saved_new_best': improved}), flush=True)
        if a.patience > 0 and no_improvement >= a.patience:
            print('Validation plateau; stopping. Best checkpoint retained.', flush=True); break
        if time.perf_counter() - began >= a.max_minutes * 60:
            print('Time budget reached at epoch boundary. Use recovery to continue.', flush=True); break
    write_json(run / 'completion.json', {'completed_additional_epochs': len(history),
               'best_macro_f1': best, 'test_accessed': False, 'synthetic': bool(a.smoke_batches)})


if __name__ == '__main__':
    main()
