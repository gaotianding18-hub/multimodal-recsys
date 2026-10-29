"""Causal-history multi-source candidates and pairwise interest ranking."""
import argparse
import copy
import hashlib
import json
import math
import sqlite3
import time

import numpy as np
import torch
from torch.nn import functional as F
from .hm_train import ROOT, DATA, Retrieval, load, metric
from .hm_multimodal import MultimodalRetrieval
from .hm_rank import Ranker, features


def load_retriever(directory):
    checkpoint=torch.load(directory/'model.pt',weights_only=True)
    if 'feature_sha256' in checkpoint:
        state=checkpoint['state']
        model=MultimodalRetrieval(state['visual'][1:],state['textual'][1:],state['available'][1:],
                                  checkpoint['dim'],checkpoint['args']['mode'])
    else: model=Retrieval(len(checkpoint['item_ids']),checkpoint['dim'])
    model.load_state_dict(checkpoint['state']);model.eval();model.requires_grad_(False)
    stored=np.load(directory/'index.npy',allow_pickle=False)
    with torch.inference_mode(): actual=model.items(torch.arange(1,len(checkpoint['item_ids'])+1)).numpy()
    if not np.allclose(actual,stored,rtol=1e-5,atol=1e-6):raise RuntimeError('Index/model mismatch')
    return model, checkpoint['item_ids'], torch.cat([torch.zeros(1,stored.shape[1]),torch.from_numpy(stored)])


def recent_order(ids, config):
    db=sqlite3.connect(DATA/'transactions.sqlite')
    counts=dict(db.execute('SELECT item,COUNT(*) FROM events WHERE date >= ? AND date < ? GROUP BY item',
                           ('2020-09-02',config['validation_start'])));db.close()
    return [i+1 for i in sorted(range(len(ids)),key=lambda i:(-counts.get(ids[i],0),ids[i]))]


def candidate_pool(model, vectors, h, recent, neural=100, popular=100, hybrid=True):
    with torch.inference_mode(): retrieval=(model.users(h)@vectors[1:].T).topk(neural,dim=-1).indices+1
    rows=[]
    for history, retrieved in zip(h.tolist(),retrieval.tolist()):
        additions=[x for x in reversed(history) if x]+recent[:popular] if hybrid else []
        rows.append(list(dict.fromkeys(retrieved+additions)))
    width=max(map(len,rows))
    return torch.tensor([row+[0]*(width-len(row)) for row in rows])


def ranked(model, ranker, vectors, h, recent, k=12, hybrid=True):
    candidates=candidate_pool(model,vectors,h,recent,hybrid=hybrid)
    expanded=h[:,None,:].expand(-1,candidates.shape[1],-1).reshape(-1,h.shape[1]);cand=candidates.flatten()
    # No aggregate-training popularity in ranking: causal history only.
    zero_pop=torch.zeros(len(vectors))
    with torch.inference_mode():
        scores=ranker(vectors[expanded],vectors[cand],expanded.ne(0),features(expanded,cand,zero_pop))
        scores=scores.reshape_as(candidates).masked_fill(candidates.eq(0),-torch.inf)
        order=scores.topk(k,dim=-1).indices
        return candidates.gather(1,order), candidates, scores.gather(1,order)


def main():
    p=argparse.ArgumentParser();p.add_argument('--retriever',default='artifacts/hm/id-random-43')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=3)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    ids,_,histories,positives,keys,known,evaluation,config=load()
    directory=ROOT/args.retriever;model,loaded_ids,vectors=load_retriever(directory)
    if loaded_ids!=ids:raise RuntimeError('Catalogue mismatch')
    recent=recent_order(ids,config);ranker=Ranker(vectors.shape[1]);optimizer=torch.optim.AdamW(ranker.parameters(),lr=.001)
    zero_pop=torch.zeros(len(vectors));best=-1;state=None;logs=[];start=time.perf_counter()
    def evaluate(split,hybrid):
        totals=np.zeros(3);coverage=0;ranker.eval()
        for offset in range(0,len(evaluation[split]),32):
            rows=evaluation[split][offset:offset+32];h=torch.tensor([r[0] for r in rows])
            top,pool,_=ranked(model,ranker,vectors,h,recent,hybrid=hybrid)
            for selected,candidates,(_,targets) in zip(top.tolist(),pool.tolist(),rows):
                totals+=metric([ids[i-1] for i in selected],targets)
                coverage+=len(targets & {ids[i-1] for i in candidates if i})/len(targets)
        out=dict(zip(['recall@12','ndcg@12','map@12'],(totals/len(evaluation[split])).tolist()))
        out['candidate_recall']=coverage/len(evaluation[split]);return out
    for epoch in range(args.epochs):
        ranker.train();losses=[]
        for positions in np.array_split(rng.permutation(len(histories)),math.ceil(len(histories)/256)):
            h=torch.from_numpy(histories[positions]);pids=torch.tensor([rng.choice(positives[j]) for j in positions])
            negatives=[]
            for j in positions:
                historical=[x for x in histories[j] if x and x not in positives[j]]
                n=int(rng.choice(historical)) if historical and rng.random()<.5 else int(rng.integers(1,len(ids)+1))
                while n in positives[j]:n=int(rng.integers(1,len(ids)+1))
                negatives.append(n)
            n=torch.tensor(negatives);optimizer.zero_grad(set_to_none=True)
            pos=ranker(vectors[h],vectors[pids],h.ne(0),features(h,pids,zero_pop))
            neg=ranker(vectors[h],vectors[n],h.ne(0),features(h,n,zero_pop))
            loss=F.softplus(neg-pos).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite ranking loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        val=evaluate('val',True);record={'epoch':epoch+1,'loss':float(np.mean(losses)),'val':val};logs.append(record)
        print(json.dumps(record),flush=True)
        if val['ndcg@12']>best:best=val['ndcg@12'];state=copy.deepcopy(ranker.state_dict())
    ranker.load_state_dict(state)
    test=evaluate('test',True);single=evaluate('test',False)
    source_sha=hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest()
    target=directory/f'hybrid-ranker-{args.seed}.pt'
    torch.save({'state':state,'dim':vectors.shape[1],'retriever_sha256':source_sha,'recent':recent,
                'seed':args.seed,'causal_behavior':True},target)
    report={'args':vars(args),'logs':logs,'test_hybrid':test,'test_neural_only':single,'selection_val_ndcg':best,
            'seconds':time.perf_counter()-start,'retriever_sha256':source_sha,
            'ranker_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'negative_sampling':'50% available historical non-targets, otherwise uniform non-targets; not explicit dislikes',
            'limitations':['Frozen recent popularity computed before validation','No blind test claim',
                           'Static metadata snapshot; no historical image/text snapshots',
                           'Ranking trained on sampled distractors, not production exposures']}
    (ROOT/f'reports/hm_hybrid_{directory.name}_{args.seed}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
