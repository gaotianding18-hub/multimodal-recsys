"""Task-trained retrieval on frozen, item-aligned CLIP features.

Independent implementation of multi-positive contrastive recommendation.
No CLIP backbone fine-tuning, CTR or official VLM2Vec execution claimed.
"""
import argparse
import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .advanced import ModalityFusion, HistoryTower, contrastive_loss
from .hm_train import ROOT, load, evaluate


class MultimodalRetrieval(nn.Module):
    def __init__(self, images, texts, available, dim=64, mode='gate'):
        super().__init__()
        n = len(texts)
        self.mode = mode
        self.register_buffer('visual', torch.cat([torch.zeros(1, images.shape[1]), images]))
        self.register_buffer('textual', torch.cat([torch.zeros(1, texts.shape[1]), texts]))
        self.register_buffer('available', torch.cat([torch.tensor([[False, True]]), available.bool()]))
        self.fusion = ModalityFusion(images.shape[1], texts.shape[1], dim,
                                     'mean' if mode == 'mean' else 'gate')
        self.identity = nn.Embedding(n + 1, dim, padding_idx=0)
        self.combine = nn.Linear(2 * dim, dim)
        self.history = HistoryTower(dim)

    def items(self, ids):
        shape = ids.shape
        unique, inverse = torch.unique(ids.flatten(), return_inverse=True)
        mask = self.available[unique].clone()
        if self.mode == 'text': mask[:, 0] = False
        if self.mode == 'image':
            # Missing images fall back to text; this is explicitly reported.
            mask[:, 1] = ~mask[:, 0]
        semantic, _ = self.fusion(self.visual[unique], self.textual[unique], mask)
        vectors = F.normalize(self.combine(torch.cat([self.identity(unique), semantic], -1)), dim=-1)
        vectors = vectors.masked_fill(unique.eq(0).unsqueeze(-1), 0)
        return vectors[inverse].reshape(*shape, vectors.shape[-1])

    def users(self, history):
        return self.history(self.items(history), history.ne(0))


def candidate_masks(positions, positives, keys, known, rng, n, random_count=256):
    # Include all positives in each basket. Deduplicate item IDs before CE.
    candidates = np.asarray(sorted(set(rng.integers(1, n + 1, random_count).tolist()) |
                                   {p for j in positions for p in positives[j]}), dtype=np.int64)
    positive = np.zeros((len(positions), len(candidates)), bool)
    allowed = np.ones_like(positive)
    for row, j in enumerate(positions):
        positive[row] = np.isin(candidates, positives[j])
        allowed[row] = ~np.isin(candidates, list(known[keys[j]])) | positive[row]
    return candidates, positive, allowed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', default='artifacts/hm/clip/features.npz')
    p.add_argument('--mode', choices=['text', 'image', 'mean', 'gate'], default='gate')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    p.add_argument('--max-train-queries', type=int, default=0)
    p.add_argument('--run-tag', default='')
    p.add_argument('--allow-partial-images', action='store_true')
    args = p.parse_args()
    if args.run_tag and not re.fullmatch(r'[a-z0-9-]{1,32}',args.run_tag):p.error('run-tag must be 1..32 lowercase letters/digits/hyphens')
    if args.epochs<1 or args.batch_size<1 or args.max_train_queries<0:p.error('Invalid training budget')
    tag='-'+args.run_tag if args.run_tag else ''
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ids, mapping, histories, positives, keys, known, evaluation, config = load()
    feature_path = ROOT / args.features
    with np.load(feature_path, allow_pickle=False) as data:
        if data['item_ids'].tolist() != ids: raise RuntimeError('Feature/catalogue ID alignment mismatch')
        images = torch.from_numpy(data['images'].copy()); texts = torch.from_numpy(data['texts'].copy())
        available = torch.from_numpy(data['available'].copy())
    if not torch.isfinite(images).all() or not torch.isfinite(texts).all():
        raise RuntimeError('Nonfinite features')
    coverage = float(available[:, 0].float().mean())
    if args.mode != 'text' and coverage < .95 and not args.allow_partial_images:
        raise RuntimeError('Image coverage below 95%; finish downloads or explicitly mark a partial-image diagnostic')
    feature_sha=hashlib.sha256(feature_path.read_bytes()).hexdigest()
    device = torch.device(args.device)
    model = MultimodalRetrieval(images, texts, available, mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    output = ROOT / f'artifacts/hm/mm-{args.mode}-{args.seed}-{feature_sha[:8]}{tag}'
    output.mkdir(parents=True, exist_ok=True)
    positions_all = np.arange(len(histories))
    if args.max_train_queries: positions_all = rng.choice(positions_all, min(args.max_train_queries,len(histories)), replace=False)
    best = -1; state = None; logs = []; start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); losses = []
        for positions in np.array_split(rng.permutation(positions_all), math.ceil(len(positions_all)/args.batch_size)):
            candidates, positive, allowed = candidate_masks(positions, positives, keys, known, rng, len(ids))
            h = torch.as_tensor(histories[positions], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = contrastive_loss(model.users(h), model.items(torch.as_tensor(candidates,device=device)),
                                    torch.as_tensor(positive,device=device),torch.as_tensor(allowed,device=device))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        val = evaluate(model, evaluation['val'], ids, device, config['evaluation_k'])
        record = {'epoch':epoch+1,'loss':float(np.mean(losses)),'val':val}; logs.append(record)
        print(json.dumps(record),flush=True)
        if val['ndcg@12'] > best: best=val['ndcg@12'];state=copy.deepcopy(model.state_dict())
        (output/'logs.json').write_text(json.dumps(logs,indent=2))
    model.load_state_dict(state); test=evaluate(model,evaluation['test'],ids,device,config['evaluation_k'])
    torch.save({'state':{k:v.cpu() for k,v in state.items()},'item_ids':ids,'dim':64,'args':vars(args),
                'feature_sha256':feature_sha},output/'model.pt')
    model.eval()
    with torch.inference_mode(): index=model.items(torch.arange(1,len(ids)+1,device=device)).cpu().numpy()
    np.save(output/'index.npy',index)
    report={'args':vars(args),'train_queries':len(positions_all),'candidate_items':len(ids),'image_coverage':coverage,
            'logs':logs,'selection_val_ndcg':best,'test':test,'seconds':time.perf_counter()-start,
            'frozen_clip':True,'feature_sha256':feature_sha,
            'trainer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'data_queries_sha256':hashlib.sha256((ROOT/'data/hm/processed/queries.jsonl').read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256((output/'model.pt').read_bytes()).hexdigest(),
            'limitations':['Exploratory test, not a blind holdout','Uniform sampled contrastive training candidates; full-catalogue evaluation',
                           'Training-known positives excluded from negatives; unobserved items are not dislike labels',
                           'Image-only arm uses text fallback where images are absent'],
            'protocol':config}
    (ROOT/f'reports/hm_mm_{args.mode}_{args.seed}_{feature_sha[:8]}{tag}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
