import unittest
import numpy as np
import torch
from recsys.hm_multimodal import MultimodalRetrieval, candidate_masks
from recsys.advanced import contrastive_loss


class MultimodalTests(unittest.TestCase):
    def test_real_gradients_padding_and_missing_images(self):
        torch.manual_seed(0)
        model=MultimodalRetrieval(torch.randn(6,8),torch.randn(6,8),
                                  torch.tensor([[True,True],[False,True]]*3),dim=4)
        h=torch.tensor([[0,1,2],[0,3,4]])
        candidates=torch.tensor([1,2,3,4,5,6])
        positive=torch.tensor([[False,False,False,False,True,False],
                               [False,False,False,False,False,True]])
        loss=contrastive_loss(model.users(h),model.items(candidates),positive,torch.ones_like(positive))
        before=model.fusion.text.weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        loss.backward();optimizer.step()
        self.assertFalse(torch.equal(before,model.fusion.text.weight))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(model.users(h),model.users(h[:,1:]),atol=1e-6))
        self.assertTrue(torch.equal(model.items(torch.tensor([0])),torch.zeros(1,4)))
        first=model.items(torch.tensor([2])).detach()
        model.visual[2]=torch.randn(8)*1000
        self.assertTrue(torch.allclose(first,model.items(torch.tensor([2])),atol=1e-6))

    def test_duplicate_candidates_and_known_positive_mask(self):
        candidates,positive,allowed=candidate_masks(np.array([0,1]),[[1,2],[2]],['u','v'],
                 {'u':{1,2,3},'v':{2,4}},np.random.default_rng(0),4,256)
        self.assertEqual(len(candidates),len(set(candidates)))
        self.assertEqual(int(positive[0].sum()),2)
        self.assertFalse(bool(allowed[0,candidates==3].item()))
        self.assertTrue(bool(allowed[0,candidates==2].item()))
        self.assertFalse(bool(allowed[1,candidates==4].item()))


if __name__=='__main__':unittest.main()
