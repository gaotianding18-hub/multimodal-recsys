import unittest
import numpy as np
import torch
from recsys.core import load_data, TwoTower, metrics, tensor, query_scores
from recsys.serve import Engine
import json
from pathlib import Path


class ProtocolTests(unittest.TestCase):
    def test_metric(self):
        r,n=metrics(np.array([0,1,2]),np.array([5,2,1]),1)
        self.assertEqual(r,1);self.assertAlmostEqual(n,1)
        r,_=metrics(np.array([0,1]),np.array([1,2]),1)
        self.assertIsNone(r)

    def test_groups_and_inputs(self):
        data=load_data();queries=data[3]
        groups={s:{q['user_id'] for q in queries if q['split']==s} for s in ['train','val','test']}
        self.assertFalse(groups['train']&groups['val']);self.assertFalse(groups['test']&groups['val'])
        self.assertFalse(groups['test']&groups['train'])
        for q in queries:self.assertEqual(len(q['candidates']),12)
        q=queries[0];m=TwoTower()
        before=m.encode_user(tensor(q['user'])[None],tensor(q['context'])[None])
        q['ratings'][:]=1
        after=m.encode_user(tensor(q['user'])[None],tensor(q['context'])[None])
        torch.testing.assert_close(before,after)

    def test_service(self):
        r=json.loads(Path('reports/experiments.json').read_text())
        e=Engine(r['selected_for_service'])
        a=e.recommend('s9',1,3);b=e.recommend('s9',1,3)
        self.assertEqual(a['items'],b['items']);self.assertEqual(len(a['items']),3)
        self.assertTrue(e.recommend('unknown',2)['unknown_user_fallback'])
        data=load_data()
        for q in data[3]:
            scores,user,vectors=query_scores(e.model,tensor(data[0]),tensor(data[1]),q)
            pool=torch.argsort(scores,descending=True,stable=True)[:6]
            ranked=e.ranker(user,vectors[pool])
            order=torch.argsort(ranked,descending=True,stable=True)[:3]
            expected=[q['candidates'][pool[j].item()]+1 for j in order]
            actual=e.recommend(q['user_id'],q['scenario'],3)
            self.assertEqual(expected,[item['item_id'] for item in actual['items']])
        with self.assertRaises(ValueError):e.recommend('s9',7)
        with self.assertRaises(ValueError):e.recommend('s9',1,0)

if __name__=='__main__':unittest.main()
