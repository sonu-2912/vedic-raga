"""Shared inference recipe; any optional recipe is selected on validation only."""
import torch
from torch.nn import functional as F

def emotion_logits(model,images,recipe=None):
    recipe=recipe or {}
    shift=int(recipe.get('translation_tta_pixels',0))
    if shift not in (0,2):raise ValueError('Only the declared2-pixel translation recipe is supported')
    offsets=[(0,0),(0,-shift),(0,shift),(-shift,0),(shift,0)] if shift else [(0,0)]
    padded=F.pad(images,(shift,shift,shift,shift),mode='replicate') if shift else images
    logits=None
    for dy,dx in offsets:
        view=padded[:,:,shift+dy:shift+dy+images.shape[-2],shift+dx:shift+dx+images.shape[-1]] if shift else images
        out=model(view)
        if recipe.get('horizontal_flip_tta',False):out=(out+model(torch.flip(view,[-1])))/2
        logits=out if logits is None else logits+out
    logits=logits/len(offsets)
    return logits/max(float(recipe.get('temperature',1.)),1e-3)
