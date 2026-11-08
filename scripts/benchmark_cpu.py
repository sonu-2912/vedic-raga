"""Synthetic-tensor throughput check only; no expression accuracy measurement."""
import json,time
import torch
from backend.models import FusDenseNet
from backend.catalog import ROOT

def main():
    torch.manual_seed(42);torch.set_num_threads(4)
    model=FusDenseNet(version=2);x=torch.rand(64,1,48,48);y=torch.randint(7,(64,));rows=[]
    for channels_last in [False,True]:
        if channels_last:model=model.to(memory_format=torch.channels_last);x=x.contiguous(memory_format=torch.channels_last)
        durations=[]
        for i in range(5):
            start=time.perf_counter();model.zero_grad();out=model(x);torch.nn.functional.cross_entropy(out,y).backward();durations.append(time.perf_counter()-start)
        row={'channels_last':channels_last,'median_forward_backward_seconds':sorted(durations[2:])[1],'timed_iterations':durations[2:]};rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'reports/cpu_memory_format_benchmark.json').write_text(json.dumps({'synthetic_tensors':True,'batch':64,'threads':4,'scope':'Throughput only, while training is running; contention/thermal state uncontrolled. No accuracy evidence.','runs':rows},indent=2))

if __name__=='__main__':main()
