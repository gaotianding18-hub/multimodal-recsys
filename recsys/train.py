import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .core import ROOT, TwoTower, Ranker, load_data, tensor, query_scores, evaluate, metrics


def train(seed, mode, data, run_dir, epochs=100):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    images, texts, items, queries, histories, contexts = data
    image, text = tensor(images), tensor(texts)
    train_q = [q for q in queries if q['split'] == 'train']
    val_q = [q for q in queries if q['split'] == 'val']
    test_q = [q for q in queries if q['split'] == 'test']
    model = TwoTower(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, state, log = -1, None, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for q in train_q:
            positive = np.flatnonzero(q['ratings'] >= 4)
            if not len(positive):
                continue
            optimizer.zero_grad(set_to_none=True)
            scores, _, _ = query_scores(model, image, text, q)
            loss = -F.log_softmax(scores / 0.1, dim=0)[positive].mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        result = evaluate(model, image, text, val_q)
        log.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'val': result})
        if result['ndcg@3'] > best:
            best, state = result['ndcg@3'], copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    retrieval_test = evaluate(model, image, text, test_q)
    # Ranking uses frozen retrieval features. Full allowed catalogues, not top-k
    # selected by a retriever fitted on the same query. Final evaluation uses top6.
    ranker = Ranker()
    rank_opt = torch.optim.AdamW(ranker.parameters(), lr=1e-3, weight_decay=1e-3)
    best_rank, rank_state, rank_log = -1, None, []
    model.eval()
    for epoch in range(60):
        losses = []
        for q in train_q:
            positive, negative = np.flatnonzero(q['ratings'] >= 4), np.flatnonzero(q['ratings'] <= 2)
            if not len(positive) or not len(negative):
                continue
            with torch.no_grad():
                _, user, vectors = query_scores(model, image, text, q)
            rank_opt.zero_grad(set_to_none=True)
            scores = ranker(user, vectors)
            loss = F.softplus(-(scores[positive][:, None] - scores[negative][None, :])).mean()
            loss.backward()
            rank_opt.step()
            losses.append(float(loss.detach()))
        val = evaluate(model, image, text, val_q, ranker)
        rank_log.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'val': val})
        if val['ndcg@3'] > best_rank:
            best_rank, rank_state = val['ndcg@3'], copy.deepcopy(ranker.state_dict())
    ranker.load_state_dict(rank_state)
    final_test = evaluate(model, image, text, test_q, ranker)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / 'model.pt'
    torch.save({'model': state, 'ranker': rank_state, 'mode': mode}, checkpoint)
    model.eval()
    with torch.inference_mode():
        index = model.encode_items(image, text).numpy()
    np.save(run_dir / 'index.npy', index)
    np.savez(run_dir / 'features.npz', images=images, texts=texts)
    profiles = {k: v.tolist() for k, v in histories.items()}
    (run_dir / 'serving.json').write_text(json.dumps({'items': items, 'histories': profiles,
        'scenarios': {str(k):v.tolist() for k,v in contexts.items()}, 'seed': seed, 'mode': mode}))
    manifest = {'model_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'index_sha256': hashlib.sha256((run_dir/'index.npy').read_bytes()).hexdigest(),
        'features_sha256': hashlib.sha256((run_dir/'features.npz').read_bytes()).hexdigest(),
        'serving_sha256': hashlib.sha256((run_dir/'serving.json').read_bytes()).hexdigest(),
        'source_revision': '68edb8e1ac43f3718844e00a0f40f4f10173b24b'}
    (run_dir/'manifest.json').write_text(json.dumps(manifest, indent=2))
    result = {'seed': seed, 'mode': mode, 'retrieval_test': retrieval_test, 'reranked_test': final_test,
        'selection_val_ndcg3': best, 'ranking_selection_val_ndcg3': best_rank,
        'train_queries':len(train_q),'val_queries':len(val_q),'test_queries':len(test_q)}
    (run_dir/'metrics.json').write_text(json.dumps(result,indent=2))
    (run_dir/'training_log.json').write_text(json.dumps({'retrieval':log,'ranking':rank_log}))
    return result


def baseline(data):
    _, texts, _, queries, _, _ = data
    means = np.zeros(36)
    counts = np.zeros(36)
    for q in queries:
        if q['split']=='train':
            means[q['candidates']] += q['ratings']
            counts[q['candidates']] += 1
    means /= np.maximum(counts,1)
    results = {}
    for name in ['train_mean_rating', 'scenario_text_similarity']:
        vals=[]
        for q in queries:
            if q['split']!='test':continue
            scores = means[q['candidates']] if name=='train_mean_rating' else texts[q['candidates']] @ q['context']
            order=np.argsort(-scores,kind='stable')
            recall,ndcg=metrics(order,q['ratings'],3)
            vals.append((recall,ndcg))
        results[name]={'ndcg@3':float(np.mean([v[1] for v in vals])),
            'recall@3':float(np.mean([v[0] for v in vals if v[0] is not None]))}
    return results


if __name__ == '__main__':
    torch.set_num_threads(4)
    data = load_data()
    output=ROOT/'artifacts/runs'
    results=[]
    for mode in ['text','image','multimodal']:
        for seed in [42,43,44]:
            result=train(seed,mode,data,output/f'{mode}-{seed}')
            results.append(result)
            print(json.dumps(result),flush=True)
    summary={'baselines':baseline(data),'runs':results,'selected_for_service':None}
    # Select strictly on validation, never test, including modality choice.
    chosen=max(results,key=lambda r:r['ranking_selection_val_ndcg3'])
    summary['selected_for_service']=f"{chosen['mode']}-{chosen['seed']}"
    (ROOT/'reports/experiments.json').write_text(json.dumps(summary,indent=2))
