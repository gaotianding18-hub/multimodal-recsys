import unittest
import torch
from torch import nn
from recsys.clip_lora import LoRALinear,symmetric_loss

class LoRATests(unittest.TestCase):
    def test_initial_equivalence_and_base_immutability(self):
        torch.manual_seed(1);base=nn.Linear(8,4);adapter=LoRALinear(base,rank=2,alpha=4)
        x=torch.randn(6,8);before=base.weight.detach().clone()
        self.assertTrue(torch.equal(base(x),adapter(x)))
        optimizer=torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=.01)
        loss=adapter(x).square().mean();loss.backward();optimizer.step()
        self.assertTrue(torch.equal(base.weight,before));self.assertGreater(float(adapter.b.detach().abs().sum()),0)
        self.assertIsNone(base.weight.grad)
    def test_duplicate_caption_multi_positive_loss(self):
        v=torch.randn(3,4,requires_grad=True);t=torch.randn(3,4,requires_grad=True)
        positive=torch.tensor([[True,True,False],[True,True,False],[False,False,True]])
        loss=symmetric_loss(v,t,positive);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(v.grad.abs().sum()),0)

if __name__=='__main__':unittest.main()
