"""H&M real-data ID/history baseline. No image/VLM claims for this run."""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/hm/processed'


class Retrieval(nn.Module):
    def __init__(self, n, dim=64):
        super().__init__()
        self.item = nn.Embedding(n + 1, dim, padding_idx=0)
        self.user = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def items(self, ids):
        return F.normalize(self.item(ids), dim=-1)

    def users(self, history):
        valid = history.ne(0)
        pooled = (self.item(history) * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(self.user(pooled), dim=-1)


def metric(predictions, targets):
    hits = [int(p in targets) for p in predictions]
    recall = sum(hits) / len(targets)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(predictions), len(targets))))
    ap = sum(h * sum(hits[:i + 1]) / (i + 1) for i, h in enumerate(hits)) / min(len(targets), len(predictions))
    return recall, dcg / ideal, ap


def load():
    catalogue = json.loads((DATA / 'catalogue.json').read_text())
    ids = [r['article_id'] for r in catalogue]
    mapping = {item: i + 1 for i, item in enumerate(ids)}
    config = json.loads((ROOT / 'configs/hm.json').read_text())
    length = config['history_length']
    histories, positives, user_keys = [], [], []
    evaluation = {'val': [], 'test': []}; known = {}
    for line in (DATA / 'queries.jsonl').open():
        q = json.loads(line)
        encoded = [mapping[x] for x in q['history'] if x in mapping][-length:]
        h = [0] * (length - len(encoded)) + encoded
        if q['split'] == 'train':
            p = [mapping[x] for x in q['targets']]
            histories.append(h); positives.append(p); user_keys.append(q['user_id'])
            known.setdefault(q['user_id'], set()).update(p + encoded)
        else:
            evaluation[q['split']].append((h, set(q['targets'])))
    return ids, mapping, np.asarray(histories, dtype=np.int64), positives, user_keys, known, evaluation, config


def evaluate(model, evaluation, ids, device, k, batch=128):
    model.eval(); total = np.zeros(3)
    with torch.inference_mode():
        vectors = model.items(torch.arange(1, len(ids) + 1, device=device))
        for start in range(0, len(evaluation), batch):
            rows = evaluation[start:start + batch]
            h = torch.tensor([r[0] for r in rows], device=device)
            scores = model.users(h) @ vectors.T
            top = scores.topk(k, dim=-1).indices.cpu().numpy()
            for order, (_, targets) in zip(top, rows):
                total += metric([ids[i] for i in order], targets)
    return dict(zip([f'recall@{k}', f'ndcg@{k}', f'map@{k}'], (total / len(evaluation)).tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--negative-strategy', choices=['random', 'mixed'], default='random')
    args = parser.parse_args(); torch.set_num_threads(4)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    ids, mapping, histories, positives, keys, known, evaluation, config = load()
    k = config['evaluation_k']; output = ROOT / f'artifacts/hm/id-{args.negative_strategy}-{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    pop = json.loads((DATA / 'popularity.json').read_text())
    popular = sorted(ids, key=lambda x: (-pop[x], x))[:k]
    baselines = {s: dict(zip([f'recall@{k}', f'ndcg@{k}', f'map@{k}'],
                            np.mean([metric(popular, t) for _, t in rows], axis=0).tolist()))
                 for s, rows in evaluation.items()}
    device = torch.device(args.device); model = Retrieval(len(ids), args.dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    best, state, logs = -1, None, []; started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); losses = []
        for positions in np.array_split(rng.permutation(len(histories)), math.ceil(len(histories) / args.batch_size)):
            h = torch.as_tensor(histories[positions], device=device)
            p = np.array([rng.choice(positives[j]) for j in positions])
            pool_size = 32 if args.negative_strategy == 'mixed' else 8
            neg = rng.integers(1, len(ids) + 1, size=(len(positions), pool_size))
            for row, j in enumerate(positions):
                for col in range(pool_size):
                    while int(neg[row, col]) in known[keys[j]]:
                        neg[row, col] = rng.integers(1, len(ids) + 1)
            optimizer.zero_grad(set_to_none=True)
            u = model.users(h); positive = model.items(torch.as_tensor(p, device=device))
            negative = model.items(torch.as_tensor(neg, device=device))
            pos_score = (u * positive).sum(-1)
            neg_score = (u.unsqueeze(1) * negative).sum(-1)
            if args.negative_strategy == 'mixed':
                # Four uniform candidates plus four hard candidates from a
                # disjoint pool; extra scoring cost is reported, not hidden.
                hard = neg_score[:, 4:].detach().topk(4, dim=-1).indices + 4
                neg_score = torch.cat([neg_score[:, :4], neg_score.gather(1, hard)], dim=1)
            loss = F.softplus((neg_score - pos_score.unsqueeze(1)) / .1).mean()
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        val = evaluate(model, evaluation['val'], ids, device, k)
        record = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'val': val}
        logs.append(record); print(json.dumps(record), flush=True)
        if val[f'ndcg@{k}'] > best:
            best = val[f'ndcg@{k}']; state = copy.deepcopy(model.state_dict())
        (output / 'logs.json').write_text(json.dumps(logs, indent=2))
    model.load_state_dict(state)
    test = evaluate(model, evaluation['test'], ids, device, k)
    torch.save({'state': {key: value.cpu() for key, value in state.items()}, 'dim': args.dim,
                'item_ids': ids, 'args': vars(args)}, output / 'model.pt')
    model.eval()
    with torch.inference_mode():
        index = model.items(torch.arange(1, len(ids) + 1, device=device)).cpu().numpy()
    np.save(output / 'index.npy', index)
    report = {'model': 'ID/history BPR baseline, no multimodal encoder', 'args': vars(args),
              'candidate_items': len(ids), 'train_queries': len(histories), 'baselines': baselines,
              'selection_val_ndcg': best, 'test': test, 'elapsed_seconds': time.perf_counter() - started,
              'protocol': config, 'model_sha256': hashlib.sha256((output / 'model.pt').read_bytes()).hexdigest()}
    (ROOT / f'reports/hm_id_{args.negative_strategy}_{args.seed}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
