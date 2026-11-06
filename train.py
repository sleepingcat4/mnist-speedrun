"""Small, paired MNIST optimizer benchmark. All times are measured, not promised."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import time
import ssl
from urllib.request import urlopen
import certifi
import torch
from torch import nn
from optimizer import DriftGate, FastDriftGate

FILES={
 'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873',
 'train-labels-idx1-ubyte.gz':'d53e105ee54ea40749a09fcbcd1e9432',
 't10k-images-idx3-ubyte.gz':'9fb629c4189551a2d022fa330f9573f3',
 't10k-labels-idx1-ubyte.gz':'ec29112dd5afa0611ce80d1b7f02629c',
}

def load_mnist(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    tensors=[]
    for name,checksum in FILES.items():
        path=root/name
        if not path.exists():
            with urlopen('https://ossci-datasets.s3.amazonaws.com/mnist/'+name,timeout=90,context=ssl.create_default_context(cafile=certifi.where())) as response:
                blob=response.read()
            if hashlib.md5(blob).hexdigest()!=checksum:raise ValueError('MNIST download checksum failed.')
            path.write_bytes(blob)
        blob=path.read_bytes()
        if hashlib.md5(blob).hexdigest()!=checksum:raise ValueError(f'Corrupt dataset: {path}')
        raw=gzip.decompress(blob);magic,count=struct.unpack('>II',raw[:8])
        if magic==2051:
            rows,cols=struct.unpack('>II',raw[8:16]);assert (rows,cols)==(28,28)
            value=torch.frombuffer(bytearray(raw[16:]),dtype=torch.uint8).reshape(count,1,28,28)
        elif magic==2049:value=torch.frombuffer(bytearray(raw[8:]),dtype=torch.uint8).long();assert value.numel()==count
        else:raise ValueError('Invalid MNIST format.')
        tensors.append(value)
    return tensors

class NanoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*7*7,64),nn.ReLU(),nn.Linear(64,10))
    def forward(self,x):return self.net(x)

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()
    elif device.type=='mps':torch.mps.synchronize()

def normalize(x):return (x.float()/255.-.1307)/.3081

@torch.no_grad()
def evaluate(model,x,y,device):
    model.eval();correct=0;total_loss=0.
    for start in range(0,len(y),512):
        labels=y[start:start+512].to(device)
        logits=model(normalize(x[start:start+512].to(device)))
        total_loss+=nn.functional.cross_entropy(logits,labels,reduction='sum').item()
        correct+=(logits.argmax(1)==labels).sum().item()
    return {'accuracy':correct/len(y),'nll':total_loss/len(y)}

def batch_order(size,steps,batch,seed):
    generator=torch.Generator().manual_seed(seed)
    batches=[]
    while len(batches)<steps:
        order=torch.randperm(size,generator=generator)
        batches.extend(order.split(batch))
    return batches[:steps]

def run_one(name,seed,args,datasets,indices,device):
    torch.manual_seed(seed);random.seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=NanoCNN().to(device)
    def factory(parameters):
        if name=='driftgate': return DriftGate(parameters,lr=args.drift_lr,alpha=args.alpha,weight_decay=args.weight_decay)
        if name=='fastdriftgate': return FastDriftGate(parameters,lr=args.fast_lr,mix=args.mix,clip=args.clip,weight_decay=args.weight_decay)
        return torch.optim.AdamW(parameters,lr=args.adam_lr,weight_decay=args.weight_decay,foreach=False)
    optimizer=factory(model.parameters())
    x,y,vx,vy,tx,ty=datasets
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    initialization_hash=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in initial.values())).hexdigest()
    # Warm up kernels/allocator for each contender, then reset parameters and optimizer state.
    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(normalize(x[:args.batch_size].to(device))),y[:args.batch_size].to(device)).backward();optimizer.step()
    sync(device);model.load_state_dict(initial);optimizer=factory(model.parameters())
    history=[];training_seconds=0.;target_seconds=None
    run_started=time.perf_counter()
    for step,index in enumerate(indices,1):
        model.train();sync(device);started=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits=model(normalize(x[index].to(device)))
        loss=nn.functional.cross_entropy(logits,y[index].to(device))
        loss.backward();optimizer.step();sync(device)
        training_seconds+=time.perf_counter()-started
        if step%args.eval_every==0 or step==args.steps:
            value=loss.item()
            if not math.isfinite(value):raise RuntimeError(f'{name} produced non-finite loss')
            validation=evaluate(model,vx,vy,device);sync(device)
            elapsed=time.perf_counter()-run_started
            if target_seconds is None and validation['accuracy']>=args.target_accuracy:target_seconds=elapsed
            row={'step':step,'train_loss':value,'validation_accuracy':validation['accuracy'],'train_seconds':training_seconds,'elapsed_seconds':elapsed}
            history.append(row);print(json.dumps({'optimizer':name,'seed':seed,**row}),flush=True)
    elapsed=time.perf_counter()-run_started
    test={'accuracy':None,'nll':None} if args.validation_only else evaluate(model,tx,ty,device)
    directory=Path(args.output)/f'{name}-seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.cpu().state_dict(),'optimizer':optimizer.state_dict(),'architecture':'NanoCNN','normalization':{'mean':.1307,'std':.3081},'seed':seed,'args':vars(args),'name':name},directory/'final.pt')
    result={'optimizer':name,'seed':seed,'steps':args.steps,'examples_seen':sum(len(i) for i in indices),'train_seconds':training_seconds,'elapsed_seconds':elapsed,'seconds_to_validation_target':target_seconds,'validation_accuracy':history[-1]['validation_accuracy'],'test_accuracy':test['accuracy'],'test_nll':test['nll'],'initialization_sha256':initialization_hash,'history':history}
    (directory/'metrics.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=200);parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);parser.add_argument('--train-size',type=int,default=10000)
    parser.add_argument('--eval-every',type=int,default=50);parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--adam-lr',type=float,default=.003);parser.add_argument('--drift-lr',type=float,default=.003);parser.add_argument('--alpha',type=float,default=.5);parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--target-accuracy',type=float,default=.95);parser.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    parser.add_argument('--threads',type=int,default=4);parser.add_argument('--data',default='data');parser.add_argument('--output',default='runs/compare')
    parser.add_argument('--optimizers',nargs='+',choices=['adamw','driftgate','fastdriftgate'],default=['adamw','driftgate','fastdriftgate'])
    parser.add_argument('--fast-lr',type=float,default=.003)
    parser.add_argument('--mix',type=float,default=.5);parser.add_argument('--clip',type=float,default=2.)
    parser.add_argument('--validation-only',action='store_true',help='Skip test metrics when tuning hyperparameters')
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.optimizers))!=len(args.optimizers):parser.error('Seeds and optimizer names must be unique.')
    if min(args.steps,args.batch_size,args.eval_every,args.threads)<1 or not 1<=args.train_size<=55000 or args.warmup<0:parser.error('Invalid size or iteration count.')
    if not 0<args.target_accuracy<=1:parser.error('Target accuracy must be in (0,1].')
    torch.set_num_threads(args.threads)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device)
    x,y,tx,ty=load_mnist(args.data)
    # Fixed train/validation partition shared by every seed and optimizer; test is untouched.
    partition=torch.randperm(len(y),generator=torch.Generator().manual_seed(2026))
    valid=partition[:5000];train=partition[5000:5000+args.train_size]
    datasets=(x[train],y[train],x[valid],y[valid],tx,ty)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps({'arguments':vars(args),'device':str(device),'torch':str(torch.__version__),'parameters':sum(p.numel() for p in NanoCNN().parameters()),'validation_size':5000,'test_size':len(ty),'partition_sha256':hashlib.sha256(partition.numpy().tobytes()).hexdigest()},indent=2))
    results=[]
    for i,seed in enumerate(args.seeds):
        indices=batch_order(len(train),args.steps,args.batch_size,seed)
        # Rotate order across seeds to balance each position among three contenders.
        shift=i%len(args.optimizers)
        order=args.optimizers[shift:]+args.optimizers[:shift]
        for name in order:results.append(run_one(name,seed,args,datasets,indices,device))
    summary={}
    for name in args.optimizers:
        rows=[r for r in results if r['optimizer']==name]
        summary[name]={}
        fields=('validation_accuracy','train_seconds','elapsed_seconds') if args.validation_only else ('test_accuracy','validation_accuracy','train_seconds','elapsed_seconds')
        for field in fields:
            values=[r[field] for r in rows];summary[name][field]={'mean':statistics.mean(values),'std':statistics.stdev(values) if len(values)>1 else None}
        summary[name]['target_reached_runs']=sum(r['seconds_to_validation_target'] is not None for r in rows)
    for seed in args.seeds:
        paired=[r for r in results if r['seed']==seed]
        assert len({r['initialization_sha256'] for r in paired})==1
    (out/'summary.json').write_text(json.dumps({'summary':summary,'runs':results},indent=2))
    accuracy_field='validation_accuracy' if args.validation_only else 'test_accuracy'
    report=['# MNIST optimizer comparison','',f'Device: {device}. Steps per run: {args.steps}. Seeds: {args.seeds}.','', '| Optimizer | Accuracy (mean; validation when --validation-only) | Train seconds (mean) | Target reached |','| --- | ---: | ---: | ---: |']
    for name,data in summary.items():report.append(f"| {name} | {data[accuracy_field]['mean']:.2%} | {data['train_seconds']['mean']:.2f} | {data['target_reached_runs']}/{len(args.seeds)} |")
    report+=['','A small fixed-budget experiment. No novelty, superiority, statistical significance or speed record is established.']
    (out/'comparison.md').write_text('\n'.join(report)+'\n');print('\n'.join(report))

if __name__=='__main__':main()
