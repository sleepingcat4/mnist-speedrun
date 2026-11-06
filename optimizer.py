"""DriftGate: an experimental dual-timescale AdamW variant, not proven novel."""
import math
import torch
from torch.optim import Optimizer

class DriftGate(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(.9,.99,.999), alpha=.5, rms_cap=1., eps=1e-8, weight_decay=.01):
        if lr<=0 or eps<=0 or weight_decay<0 or alpha<0 or rms_cap<=0:
            raise ValueError('Invalid optimizer hyperparameter.')
        if not 0<=betas[0]<betas[1]<1 or not 0<=betas[2]<1:
            raise ValueError('Expected 0 <= fast beta < slow beta < 1 and 0 <= variance beta < 1.')
        super().__init__(params,dict(lr=lr,betas=betas,alpha=alpha,rms_cap=rms_cap,eps=eps,weight_decay=weight_decay))

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            bf,bs,bv=group['betas']
            for p in group['params']:
                if p.grad is None:continue
                g=p.grad
                if g.is_sparse:raise RuntimeError('DriftGate does not support sparse gradients.')
                state=self.state[p]
                if not state:
                    state.update(step=0,fast=torch.zeros_like(p),slow=torch.zeros_like(p),variance=torch.zeros_like(p))
                state['step']+=1;t=state['step']
                state['fast'].lerp_(g,1-bf);state['slow'].lerp_(g,1-bs)
                state['variance'].mul_(bv).addcmul_(g,g,value=1-bv)
                fast=state['fast']/(1-bf**t);slow=state['slow']/(1-bs**t)
                variance=state['variance']/(1-bv**t)
                drift=fast-slow
                gate=drift.square()/(variance+drift.square()+group['eps'])
                direction=(fast+group['alpha']*gate*drift)/(variance.sqrt()+group['eps'])
                # Tensorwise cap, implemented on device without a host synchronization.
                direction=direction/(direction.square().mean().sqrt()/group['rms_cap']).clamp_min(1)
                p.mul_(1-group['lr']*group['weight_decay'])
                p.add_(direction,alpha=-group['lr'])
        return loss

class FastDriftGate(Optimizer):
    """Two-state, gradient-aligned variant. Real-arithmetic guarantees: lean/.

    Blend bias-corrected momentum with the current gradient. On sign disagreement,
    use the gradient. Precondition and clip coordinates, avoiding RMS reductions.
    This is an experimental variant, not a claim of algorithmic novelty.
    """
    def __init__(self,params,lr=.003,betas=(.9,.999),mix=.5,clip=2.,eps=1e-8,weight_decay=.01):
        if not all(math.isfinite(x) for x in (lr,*betas,mix,clip,eps,weight_decay)):
            raise ValueError('Hyperparameters must be finite.')
        if lr<=0 or eps<=0 or clip<=0 or weight_decay<0 or not 0<mix<=1 or any(not 0<=b<1 for b in betas):
            raise ValueError('Invalid FastDriftGate hyperparameter.')
        super().__init__(params,dict(lr=lr,betas=betas,mix=mix,clip=clip,eps=eps,weight_decay=weight_decay))

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            b1,b2=group['betas']
            for p in group['params']:
                if p.grad is None:continue
                g=p.grad
                if g.is_sparse or g.is_complex():raise RuntimeError('Dense real gradients required.')
                state=self.state[p]
                if not state:
                    state.update(step=0,momentum=torch.zeros_like(p),variance=torch.zeros_like(p))
                state['step']+=1;t=state['step']
                state['momentum'].lerp_(g,1-b1)
                state['variance'].mul_(b2).addcmul_(g,g,value=1-b2)
                direction=state['momentum'].div(1-b1**t).lerp_(g,group['mix'])
                direction=torch.where(direction*g>=0,direction,g)
                denom=state['variance'].sqrt().div_(math.sqrt(1-b2**t)).add_(group['eps'])
                direction.div_(denom).clamp_(-group['clip'],group['clip'])
                p.mul_(1-group['lr']*group['weight_decay']).add_(direction,alpha=-group['lr'])
        return loss
