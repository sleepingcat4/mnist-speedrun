"""Optimizer-only CPU microbenchmark; fixed gradients, not a training-quality test."""
import argparse,json,statistics,time
from pathlib import Path
import torch
from train import NanoCNN
from optimizer import DriftGate,FastDriftGate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',type=int,default=500);p.add_argument('--rounds',type=int,default=7);p.add_argument('--output',default='runs/optimizer-timing.json');a=p.parse_args()
    torch.set_num_threads(1)
    factories={'adamw':lambda ps:torch.optim.AdamW(ps,foreach=False),'driftgate':DriftGate,'fastdriftgate':FastDriftGate}
    samples={name:[] for name in factories}
    for r in range(a.rounds):
        names=list(factories);names=names[r%3:]+names[:r%3]
        for name in names:
            torch.manual_seed(1);model=NanoCNN();opt=factories[name](model.parameters())
            for param in model.parameters():param.grad=torch.randn_like(param)
            for _ in range(30):opt.step()
            started=time.perf_counter()
            for _ in range(a.steps):opt.step()
            samples[name].append((time.perf_counter()-started)*1000/a.steps)
    result={'device':'cpu','threads':1,'steps_per_round':a.steps,'rounds':a.rounds,'milliseconds_per_step':{k:{'median':statistics.median(v),'samples':v} for k,v in samples.items()}}
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
