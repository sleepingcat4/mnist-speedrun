import copy
import unittest
import torch
from optimizer import DriftGate
from train import batch_order, NanoCNN

class OptimizerTests(unittest.TestCase):
    def test_reduces_to_adamw_without_drift_or_cap(self):
        a=torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64));b=torch.nn.Parameter(a.detach().clone())
        custom=DriftGate([a],alpha=0,rms_cap=1e12,lr=.01)
        reference=torch.optim.AdamW([b],lr=.01,betas=(.9,.999),weight_decay=.01,foreach=False)
        for i in range(1,6):
            a.grad=torch.tensor([.1*i,-.3/i],dtype=torch.float64);b.grad=a.grad.clone()
            custom.step();reference.step();torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)

    def test_checkpoint_continuation(self):
        a=torch.nn.Parameter(torch.tensor([1.,-2.]));opt=DriftGate([a])
        a.grad=torch.tensor([.5,-.3]);opt.step()
        b=torch.nn.Parameter(a.detach().clone());restored=DriftGate([b]);restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        for _ in range(3):
            a.grad=a.detach()*2;b.grad=b.detach()*2;opt.step();restored.step()
        torch.testing.assert_close(a,b)

    def test_quadratic_loss_decreases(self):
        p=torch.nn.Parameter(torch.tensor([2.,-3.]));opt=DriftGate([p],lr=.05,weight_decay=0)
        start=p.square().sum().item()
        for _ in range(100):
            opt.zero_grad();p.square().sum().backward();opt.step()
        self.assertLess(p.square().sum().item(),start*.1)

    def test_zero_gradient_and_missing_gradient(self):
        p=torch.nn.Parameter(torch.ones(3));opt=DriftGate([p],weight_decay=0)
        opt.step();self.assertEqual(len(opt.state),0)
        p.grad=torch.zeros_like(p);opt.step();self.assertTrue(torch.isfinite(p).all());torch.testing.assert_close(p,torch.ones(3))

    def test_invalid_and_sparse(self):
        p=torch.nn.Parameter(torch.ones(3))
        with self.assertRaises(ValueError):DriftGate([p],betas=(.99,.9,.999))
        p.grad=torch.sparse_coo_tensor(torch.tensor([[0]]),torch.tensor([1.]),size=(3,))
        with self.assertRaises(RuntimeError):DriftGate([p]).step()

    def test_paired_batch_order_and_model_reload(self):
        left=batch_order(21,8,8,42);right=batch_order(21,8,8,42)
        for a,b in zip(left,right):torch.testing.assert_close(a,b)
        self.assertEqual(len(left),8)
        model=NanoCNN();other=NanoCNN();other.load_state_dict(model.state_dict())
        image=torch.rand(2,1,28,28);torch.testing.assert_close(model(image),other(image))


class FastOptimizerTests(unittest.TestCase):
    def test_alignment_and_coordinate_bound(self):
        from optimizer import FastDriftGate
        torch.manual_seed(7)
        p=torch.nn.Parameter(torch.randn(128,dtype=torch.float64))
        opt=FastDriftGate([p],lr=.03,clip=.7,weight_decay=0)
        for _ in range(40):
            gradient=torch.randn_like(p);p.grad=gradient;before=p.detach().clone();opt.step()
            direction=(before-p.detach())/.03
            self.assertGreaterEqual((gradient*direction).min().item(),-1e-12)
            self.assertLessEqual(direction.abs().max().item(),.7+1e-12)

    def test_state_reload(self):
        from optimizer import FastDriftGate
        a=torch.nn.Parameter(torch.tensor([1.,-2.]));opt=FastDriftGate([a])
        a.grad=torch.tensor([.4,-.7]);opt.step()
        b=torch.nn.Parameter(a.detach().clone());restored=FastDriftGate([b]);restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        for _ in range(4):
            a.grad=a.detach()*2;b.grad=b.detach()*2;opt.step();restored.step()
        torch.testing.assert_close(a,b)
        self.assertEqual(set(opt.state[a]),{'step','momentum','variance'})

    def test_invalid_hyperparameters(self):
        from optimizer import FastDriftGate
        p=torch.nn.Parameter(torch.ones(1))
        for options in ({'mix':0},{'clip':0},{'lr':float('nan')},{'weight_decay':-1}):
            with self.assertRaises(ValueError):FastDriftGate([p],**options)

if __name__=='__main__':unittest.main()
