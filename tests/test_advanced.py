import unittest
import torch
from recsys.advanced import ModalityFusion, HistoryTower, InterestRanker, contrastive_loss


class AdvancedTests(unittest.TestCase):
    def test_missing_modality_and_padding_invariance(self):
        torch.manual_seed(7)
        fusion = ModalityFusion(8, 6, 4)
        image, text = torch.randn(3, 8), torch.randn(3, 6)
        available = torch.tensor([[True, False], [False, True], [True, True]])
        before, weights = fusion(image, text, available)
        image[1] = 1000; text[0] = -1000
        after, _ = fusion(image, text, available)
        torch.testing.assert_close(before, after)
        self.assertEqual(weights[0, 1].item(), 0)
        history = torch.randn(2, 3, 4)
        valid = torch.tensor([[True, False, False], [False, False, False]])
        candidate = torch.randn(2, 4)
        tower, ranker = HistoryTower(4), InterestRanker(4)
        a, b = tower(history, valid), ranker(history, candidate, valid)
        history[~valid] = 10000
        torch.testing.assert_close(a, tower(history, valid))
        torch.testing.assert_close(b, ranker(history, candidate, valid))
        self.assertTrue(torch.isfinite(b).all())

    def test_real_gradient_update_and_candidate_mask(self):
        torch.manual_seed(9)
        fusion = ModalityFusion(8, 6, 4)
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=.01)
        users = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
        items, _ = fusion(torch.randn(3, 8), torch.randn(3, 6), torch.ones(3, 2).bool())
        positive = torch.tensor([[True, False, False], [False, True, False]])
        allowed = torch.tensor([[True, True, False], [True, True, False]])
        loss = contrastive_loss(users, items, positive, allowed)
        changed = items.detach().clone(); changed[2] = 10000
        torch.testing.assert_close(loss.detach(), contrastive_loss(users, changed, positive, allowed))
        before = fusion.image.weight.detach().clone()
        loss.backward(); optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, fusion.image.weight))
