import copy
import pytest
import torch
from scripts.recall_recovery_common import eligible,recovery_loss

def report():
 return {'accuracy':.59,'classification_report':{'macro avg':{'precision':.56,'recall':.56,'f1-score':.56},**{n:{'precision':.6,'recall':.6,'f1-score':.6} for n in ['angry','disgust','fear','happy','sad','surprise','neutral']}}}

def test_recovery_gate_requires_each_requested_class_and_preserves_disgust():
 current=report();earlier=report();earlier['classification_report']['happy']['recall']=.8;earlier['classification_report']['surprise']['recall']=.75
 candidate=copy.deepcopy(earlier);candidate['accuracy']=.61;candidate['classification_report']['macro avg']['f1-score']=.58
 assert eligible(candidate,current,earlier)
 for name in ['happy','surprise','neutral','disgust']:
  failed=copy.deepcopy(candidate);failed['classification_report'][name]['recall']-=.01
  assert not eligible(failed,current,earlier)

def test_retention_reference_is_detached_and_wrong_disgust_reference_is_not_taught():
 x=torch.randn(5,7,requires_grad=True);ref=torch.zeros(5,7,requires_grad=True);y=torch.tensor([1,3,5,6,0]);loss,ce,ret=recovery_loss(x,y,ref,torch.ones(7),torch.zeros(5,dtype=torch.bool),.5,1.)
 assert ret==0;loss.backward();assert x.grad is not None and ref.grad is None

@pytest.mark.parametrize('allow',[False,True])
def test_disgust_f1_retention_is_explicit_not_silently_relaxed(allow):
 from scripts.colab_evaluate import integration_gate
 c={k:{'difference':.01} for k in ['accuracy','macro_f1','macro_precision','macro_recall','disgust_precision','disgust_recall','disgust_f1']};c['disgust_f1']['difference']=0.
 assert integration_gate(c,True,False,allow)==allow
 c['disgust_f1']['difference']=-.001;assert not integration_gate(c,True,False,allow)

def test_retention_flag_is_locked_before_any_test_data(tmp_path,monkeypatch):
 import json,sys
 from scripts.colab_evaluate import main
 (tmp_path/'bundle_manifest.json').write_text(json.dumps({'strict_all_metrics':False,'single_view_only':False}),encoding='utf-8');monkeypatch.setattr(sys,'argv',['eval','--run-dir',str(tmp_path),'--data','must-not-open.zip','--allow-equal-disgust-f1'])
 with pytest.raises(ValueError,match='Cannot change disgust F1'):main()


def test_mirror_reallocation_preserves_total_class_cost():
 from scripts.recall_recovery_common import training_sample_weights
 labels=torch.tensor([1,3,5,6,1,3,5,6]);mirror=torch.tensor([False]*4+[True]*4);base=torch.tensor([1.,2.5,1.,.7,1.,1.1,.9]);w=training_sample_weights(labels,base,mirror,.5)
 for c in [1,3,5,6]:assert torch.allclose(w[labels==c].sum(),2*base[c])
 assert w[1]>w[5] and torch.allclose(w[1],2*w[5])


def test_boundary_retention_protects_negative_examples_and_has_no_reference_gradient():
 from scripts.recall_recovery_common import boundary_recovery_loss
 reference=torch.tensor([[0.,0.,0.,5.,0.,0.,0.],[0.,5.,0.,0.,0.,0.,0.]],requires_grad=True);labels=torch.tensor([3,1]);same=reference.detach().clone().requires_grad_();_,_,zero=boundary_recovery_loss(same,same,labels,reference,torch.ones(7),torch.zeros(2,dtype=torch.bool),.5,1.);assert abs(float(zero))<1e-6
 changed=same.detach().clone();changed[0,1]=6.;changed.requires_grad_();loss,_,penalty=boundary_recovery_loss(changed,changed,labels,reference,torch.ones(7),torch.zeros(2,dtype=torch.bool),.5,1.);assert penalty>.01;loss.backward();assert changed.grad is not None and reference.grad is None
