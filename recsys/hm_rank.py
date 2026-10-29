"""Pairwise preference ranking on frozen retrieval features, real purchases."""
import copy
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .advanced import InterestRanker
from .hm_train import ROOT, DATA, Retrieval, load, metric


class Ranker(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.interest = InterestRanker(dim)
        self.behavior = nn.Sequential(nn.Linear(3,16),nn.ReLU(),nn.Linear(16,1))

    def forward(self, history, candidate, valid, behavior):
        return self.interest(history,candidate,valid) + self.behavior(behavior).squeeze(-1)


def features(h, candidate, popularity):
    match = h.eq(candidate[:, None]) & h.ne(0)
    return torch.stack([match.any(1).float(), match.sum(1).float() / 30,
                        popularity[candidate]], -1)


def main():
    torch.set_num_threads(4);torch.manual_seed(42);rng=np.random.default_rng(42)
    ids, mapping, histories, positives, keys, known, evaluation, config = load()
    choice=json.loads((ROOT/'reports/hm_id_summary.json').read_text())['selected_on_validation']
    directory=ROOT/f"artifacts/hm/id-{choice['negative_strategy']}-{choice['seed']}"
    checkpoint=torch.load(directory/'model.pt',weights_only=True)
    retriever=Retrieval(len(ids),choice['dim']);retriever.load_state_dict(checkpoint['state']);retriever.eval()
    retriever.requires_grad_(False)
    with torch.inference_mode(): vectors=retriever.items(torch.arange(len(ids)+1)).clone()
    counts=json.loads((DATA/'popularity.json').read_text())
    popularity=torch.tensor([0]+[math.log1p(counts[i]) for i in ids]);popularity/=popularity.max()
    ranker=Ranker(choice['dim']);optimizer=torch.optim.AdamW(ranker.parameters(),lr=.001)
    k=config['evaluation_k'];pool_size=100;best=-1;state=None;logs=[];started=time.perf_counter()

    def evaluate(split):
        total=np.zeros(3);coverage=0;ranker.eval()
        with torch.inference_mode():
            for start in range(0,len(evaluation[split]),64):
                rows=evaluation[split][start:start+64];h=torch.tensor([r[0] for r in rows])
                retrieval=retriever.users(h)@vectors[1:].T
                candidates=retrieval.topk(pool_size,dim=-1).indices+1
                expanded=h[:,None,:].expand(-1,pool_size,-1).reshape(-1,h.shape[1])
                cand=candidates.flatten()
                scores=ranker(vectors[expanded],vectors[cand],expanded.ne(0),features(expanded,cand,popularity)).reshape(len(rows),pool_size)
                order=scores.topk(k,dim=-1).indices
                top=candidates.gather(1,order).numpy()
                for selected,pool,(_,targets) in zip(top,candidates.numpy(),rows):
                    total+=metric([ids[i-1] for i in selected],targets)
                    coverage+=len(targets & {ids[i-1] for i in pool})/len(targets)
        result=dict(zip(['recall@12','ndcg@12','map@12'],(total/len(evaluation[split])).tolist()))
        result['retrieval_recall@100']=coverage/len(evaluation[split]);return result

    for epoch in range(3):
        ranker.train();losses=[]
        for positions in np.array_split(rng.permutation(len(histories)),math.ceil(len(histories)/256)):
            h=torch.from_numpy(histories[positions]);p=torch.tensor([rng.choice(positives[j]) for j in positions])
            n=rng.integers(1,len(ids)+1,len(positions))
            for i,j in enumerate(positions):
                while int(n[i]) in known[keys[j]]:n[i]=rng.integers(1,len(ids)+1)
            n=torch.from_numpy(n);optimizer.zero_grad(set_to_none=True)
            pos=ranker(vectors[h],vectors[p],h.ne(0),features(h,p,popularity))
            neg=ranker(vectors[h],vectors[n],h.ne(0),features(h,n,popularity))
            loss=F.softplus(neg-pos).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite ranking loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        val=evaluate('val');record={'epoch':epoch+1,'loss':float(np.mean(losses)),'val':val};logs.append(record)
        print(json.dumps(record),flush=True)
        if val['ndcg@12']>best:best=val['ndcg@12'];state=copy.deepcopy(ranker.state_dict())
    ranker.load_state_dict(state);test=evaluate('test')
    torch.save({'state':state,'dim':choice['dim'],'popularity':popularity,'retriever':choice,'candidate_pool':pool_size},directory/'ranker.pt')
    report={'test':test,'logs':logs,'selection_val_ndcg':best,'seconds':time.perf_counter()-started,
            'retriever':choice,'ranking_negatives':'uniform unobserved items excluding training-known positives',
            'limitations':['Frozen retriever fitted on same training histories; ranking training uses uniform items, not fitted top-k.',
                            'Training popularity is aggregate training-only, not causal per-day snapshot.',
                            'Exploratory test already inspected; no blind-test claim.','No CTR probabilities.']}
    (ROOT/'reports/hm_ranking.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
