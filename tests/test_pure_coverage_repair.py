import torch,pytest
from scripts.train_pure_coverage_repair import CoverageBatches,eligible

def test_coverage_sampler_visits_every_training_face_and_never_duplicates_inside_batch():
 labels=torch.repeat_interleave(torch.arange(7),torch.tensor([100,80,90,140,100,90,103]));sampler=CoverageBatches(labels);batches=list(sampler)
 assert len(batches)==len(sampler)
 assert set(i for batch in batches for i in batch)==set(range(len(labels)))
 assert all(len(batch)==len(set(batch))==64 for batch in batches)
 assert sampler.last_audit['unique_training_faces']==len(labels)
 for batch in batches:
  counts=torch.bincount(labels[batch],minlength=7);assert counts[1]>=4 and counts[0]>=2 and counts[4]>=2
 assert sampler.last_audit['natural_padding']<56
 assert sum(sampler.last_audit['class_draws'])==64*len(sampler)

def test_coverage_sampler_resume_epoch_is_reproducible_and_changes_order():
 labels=torch.arange(700)%7;a=CoverageBatches(labels,seed=11);b=CoverageBatches(labels,seed=11);a.set_epoch(3);b.set_epoch(3)
 first=list(a);assert first==list(b)
 b.set_epoch(4);assert first!=list(b)
 assert a.last_audit['all_training_faces_covered'] and b.last_audit['all_training_faces_covered']

def test_pure_repair_selection_demands_minority_and_overall_improvement():
 def report(a,f,r,d):return {'accuracy':a,'classification_report':{'macro avg':{'f1-score':f},'disgust':{'recall':r,'f1-score':d}}}
 base=report(.58,.52,.28,.30)
 assert not eligible(report(.59,.53,.34,.33),base)
 assert not eligible(report(.57,.53,.40,.33),base)
 assert not eligible(report(.59,.51,.40,.33),base)
 assert eligible(report(.59,.53,.36,.33),base)
