"""Prospective gates for class recovery with retained disgust performance."""
import torch
from torch.nn import functional as F
from scripts.train_consistency import teacher_kl
CLASSES=['angry','disgust','fear','happy','sad','surprise','neutral']
TARGETS=['happy','surprise','neutral']
def vector(r):
 c=r['classification_report'];return {'accuracy':r['accuracy'],'macro_f1':c['macro avg']['f1-score'],'macro_precision':c['macro avg']['precision'],'macro_recall':c['macro avg']['recall'],'disgust_precision':c['disgust']['precision'],'disgust_recall':c['disgust']['recall'],'disgust_f1':c['disgust']['f1-score']}
def eligible(r,current,earlier):
 x,b=vector(r),vector(current)
 return x['accuracy']>b['accuracy'] and x['macro_f1']>b['macro_f1'] and all(x[k]+1e-12>=b[k] for k in b if k not in ['accuracy','macro_f1']) and all(r['classification_report'][name]['recall']+1e-12>=max(current['classification_report'][name]['recall'],earlier['classification_report'][name]['recall']) for name in TARGETS)
def objective(r):
 v=vector(r);return .5*v['accuracy']+.5*v['macro_f1']
def feasibility(r,current,earlier):
 a,b=vector(r),vector(current);gap=sum(max(0.,b[k]-a[k]) for k in b)
 gap+=sum(max(0.,max(current['classification_report'][name]['recall'],earlier['classification_report'][name]['recall'])-r['classification_report'][name]['recall']) for name in TARGETS)
 return objective(r)-2*gap

def recovery_loss(logits,labels,reference_logits,weights,mirrored,mirror_common_weight,retention_weight):
 # Costs come from TRAIN labels; the fixed reference is the same current pure classifier.
 common=(labels==3)|(labels==5)|(labels==6);w=weights[labels]*torch.where(mirrored&common,torch.full_like(labels,mirror_common_weight,dtype=logits.dtype),torch.ones_like(labels,dtype=logits.dtype))
 ce=(F.cross_entropy(logits,labels,reduction='none')*w).sum()/w.sum()
 mask=labels==1
 retention=teacher_kl(logits[mask],reference_logits[mask].detach(),labels[mask],2.) if bool(mask.any()) else logits.sum()*0
 return ce+retention_weight*retention,ce,retention
