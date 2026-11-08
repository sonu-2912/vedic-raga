"""Engineering fixtures, never facial classification performance results."""
import copy
import torch
from torch.utils.data import Dataset,DataLoader
from scripts.train_consistency import Indexed,teacher_kl,update_ema

def test_wrong_teacher_returns_finite_zero_with_zero_gradient():
    logits=torch.tensor([[.1,.2,.3]],requires_grad=True)
    teacher=torch.tensor([[10.,0.,0.]])
    loss=teacher_kl(logits,teacher,torch.tensor([2]));loss.backward()
    assert loss.item()==0 and torch.equal(logits.grad,torch.zeros_like(logits))

def test_distillation_ignores_wrong_teacher_only_not_correct_examples():
    logits=torch.zeros(2,3,requires_grad=True)
    teacher=torch.tensor([[5.,0.,0.],[5.,0.,0.]])
    loss=teacher_kl(logits,teacher,torch.tensor([0,2]));loss.backward()
    assert loss.item()>0 and logits.grad[0].abs().sum()>0 and logits.grad[1].abs().sum()==0

def test_ema_has_independent_parameters_and_copies_normalization_buffers():
    torch.manual_seed(21)
    student=torch.nn.Sequential(torch.nn.BatchNorm2d(2),torch.nn.Conv2d(2,1,1))
    ema=copy.deepcopy(student);ema.requires_grad_(False);old=ema[1].weight.detach().clone()
    with torch.no_grad():student[1].weight.add_(1);student[0].running_mean.add_(2)
    update_ema(ema,student,0)
    assert torch.allclose(ema[1].weight,old+.9)
    assert torch.equal(ema[0].running_mean,student[0].running_mean)
    assert ema[1].weight.data_ptr()!=student[1].weight.data_ptr()
    assert not any(p.requires_grad for p in ema.parameters())

def test_training_teacher_indices_remain_aligned_after_shuffle():
    class Fixture(Dataset):
        def __len__(self):return 29
        def __getitem__(self,index):return torch.tensor([index],dtype=torch.float),index%7
    teacher=torch.eye(7)[torch.arange(29)%7]
    torch.manual_seed(21)
    seen=[]
    for x,y,index in DataLoader(Indexed(Fixture()),batch_size=5,shuffle=True):
        assert torch.equal(teacher[index].argmax(1),y)
        assert torch.equal(x[:,0].long(),index);seen.extend(index.tolist())
    assert sorted(seen)==list(range(29))
