import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/vogue/data'
TEXT_DIM = 256
IMAGE_DIM = 768


def text_vector(text):
    v = np.zeros(TEXT_DIM, dtype=np.float32)
    for token in re.findall(r'[a-z]+', text.lower()):
        h = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little')
        v[h % TEXT_DIM] += 1
    return v / max(np.linalg.norm(v), 1e-8)


def load_data():
    images = []
    for i in range(1, 77):
        with Image.open(DATA / f'metadata/item_{i}.png') as im:
            a = np.asarray(im.convert('RGB').resize((16, 16)), dtype=np.float32) / 255
        images.append(a.reshape(-1))
    images = np.stack(images)
    texts, items = [], []
    for i in range(1, 37):
        item = json.loads((DATA / f'metadata/item_{i}.json').read_text())
        if item['id'] != i:
            raise ValueError('Item ID mismatch')
        texts.append(text_vector(item['product_name'] + ' ' + ' '.join(item.get('about_product') or [])))
        items.append({'item_id': i, 'name': item['product_name'], 'catalogue': item['catalogue']})
    histories = {}
    with (DATA / 'fashion_profiles/ratings.csv').open() as f:
        for row in csv.DictReader(f):
            if not row['participant_id'].startswith('s'):
                continue
            ratings = np.array([float(row[f'item_{i}']) for i in range(37, 77)])
            weights = np.maximum(ratings - 3, 0)
            # If no positive profile ratings, use all observed profile pictures.
            if weights.sum() == 0:
                weights = (ratings != -1).astype(float)
            histories[row['participant_id']] = (weights @ images[36:] / max(weights.sum(), 1)).astype(np.float32)
    scenarios = {int(x['Scenario']): text_vector(x['Body']) for x in json.loads((DATA / 'conversation_trials/scenarios.json').read_text())}
    queries = []
    with (DATA / 'conversation_trials/item_ratings/seeker_ratings.csv').open() as f:
        for row in csv.DictReader(f):
            uid = row['participant_id']
            candidates = [i for i, item in enumerate(items) if item['catalogue'] == row['catalogue']]
            ratings = np.array([float(row[f'item_{i+1}']) for i in candidates], dtype=np.float32)
            if len(candidates) != 12 or np.any(ratings < 1):
                raise ValueError('Unexpected scenario candidate/label layout')
            queries.append({'user_id': uid, 'scenario': int(row['scenario']), 'conversation_id': row['conversation_id'],
                            'split': 'train' if int(uid[1:]) <= 6 else ('val' if int(uid[1:]) <= 8 else 'test'),
                            'candidates': candidates, 'ratings': ratings,
                            'user': histories[uid], 'context': scenarios[int(row['scenario'])]})
    return images[:36], np.stack(texts), items, queries, histories, scenarios


class TwoTower(nn.Module):
    def __init__(self, mode='multimodal'):
        super().__init__()
        self.mode = mode
        self.image = nn.Sequential(nn.Linear(IMAGE_DIM, 64), nn.ReLU(), nn.Linear(64, 32))
        self.text = nn.Sequential(nn.Linear(TEXT_DIM, 64), nn.ReLU(), nn.Linear(64, 32))
        self.user = nn.Sequential(nn.Linear(IMAGE_DIM + TEXT_DIM, 64), nn.ReLU(), nn.Linear(64, 32))

    def encode_items(self, images, texts):
        a, b = self.image(images), self.text(texts)
        return F.normalize(b if self.mode == 'text' else a if self.mode == 'image' else a + b, dim=-1)

    def encode_user(self, history, context):
        return F.normalize(self.user(torch.cat([history, context], dim=-1)), dim=-1)


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, user, items):
        user = user.expand_as(items)
        return self.layers(torch.cat([user, items, user * items, (user - items).abs()], dim=-1)).squeeze(-1)


def tensor(a):
    return torch.as_tensor(a, dtype=torch.float32)


def query_scores(model, image, text, q, index=None):
    vectors = model.encode_items(image, text) if index is None else index
    user = model.encode_user(tensor(q['user'])[None], tensor(q['context'])[None])
    candidates = q['candidates']
    return (user @ vectors[candidates].T).squeeze(0), user, vectors[candidates]


def metrics(order, ratings, k):
    relevant = ratings >= 4
    selected = order[:k]
    recall = float(relevant[selected].sum() / relevant.sum()) if relevant.sum() else None
    gains = np.power(2.0, ratings - 1) - 1
    discount = 1 / np.log2(np.arange(min(k, len(order))) + 2)
    dcg = (gains[selected] * discount).sum()
    ideal = (np.sort(gains)[::-1][:k] * discount).sum()
    return recall, float(dcg / ideal) if ideal else 0.0


def evaluate(model, image, text, queries, ranker=None, candidate_k=6):
    out = {f'{metric}@{k}': [] for metric in ['recall', 'ndcg'] for k in [3, 6]}
    no_positive = 0
    model.eval()
    if ranker is not None:
        ranker.eval()
    with torch.inference_mode():
        index = model.encode_items(image, text)
        for q in queries:
            scores, user, vectors = query_scores(model, image, text, q, index)
            order = np.argsort(-scores.numpy(), kind='stable')
            if ranker is not None:
                pool = order[:candidate_k]
                ranked = ranker(user, vectors[pool]).numpy()
                order = np.concatenate([pool[np.argsort(-ranked, kind='stable')], order[candidate_k:]])
            if not np.any(q['ratings'] >= 4):
                no_positive += 1
            for k in [3, 6]:
                recall, ndcg = metrics(order, q['ratings'], k)
                if recall is not None:
                    out[f'recall@{k}'].append(recall)
                out[f'ndcg@{k}'].append(ndcg)
    return {**{key: float(np.mean(v)) if v else None for key, v in out.items()},
            'queries': len(queries), 'queries_without_rating_ge4': no_positive}
