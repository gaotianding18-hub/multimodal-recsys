import unittest
import torch
from scripts.prepare_hm import build_queries
from recsys.hm_train import metric, Retrieval


class HMTests(unittest.TestCase):
    def test_daily_basket_never_enters_own_history(self):
        queries, history = build_queries([('2020-01-01','a'),('2020-01-02','b'),
                                         ('2020-01-02','c'),('2020-01-02','b')], 30)
        self.assertEqual(queries, [{'date':'2020-01-02','history':['a'],'targets':['b','c']}])
        self.assertEqual(history, ['a','b','c'])

    def test_out_of_catalogue_target_counts_as_miss(self):
        recall, ndcg, ap = metric(['a','b'], {'a','new-item'})
        self.assertEqual(recall, .5)
        self.assertEqual(ap, .5)
        self.assertLess(ndcg, 1)

    def test_zero_padding_does_not_change_history_vector(self):
        model = Retrieval(5, 4)
        torch.testing.assert_close(model.users(torch.tensor([[1,2]])),
                                   model.users(torch.tensor([[0,0,1,2]])))
