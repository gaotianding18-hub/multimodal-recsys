"""Frozen official CLIP semantic encoding; no backbone training claimed."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from third_party.clip.model import build_model
from third_party.clip.simple_tokenizer import SimpleTokenizer

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'


def tokenize(texts, tokenizer):
    tokens = torch.zeros(len(texts), 77, dtype=torch.long)
    start, end = tokenizer.encoder['<|startoftext|>'], tokenizer.encoder['<|endoftext|>']
    for i, text in enumerate(texts):
        sequence = [start] + tokenizer.encode(text)[:75] + [end]
        tokens[i, :len(sequence)] = torch.tensor(sequence)
    return tokens


def preprocess(path):
    with Image.open(path) as source:
        image = source.convert('RGB')
        # CLIP: bicubic resize short side224, then centre crop224.
        w, h = image.size
        size = (224, int(224 * h / w)) if w <= h else (int(224 * w / h), 224)
        image = ImageOps.fit(image.resize(size, Image.Resampling.BICUBIC), (224, 224),
                             method=Image.Resampling.BICUBIC, centering=(.5, .5))
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    mean = torch.tensor([.48145466, .4578275, .40821073])[:, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711])[:, None, None]
    return (pixels - mean) / std


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu','mps'], default='cpu')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--limit', type=int, default=128)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(); torch.set_num_threads(4)
    weight = ROOT / 'artifacts/backbones/ViT-B-32.pt'
    if hashlib.sha256(weight.read_bytes()).hexdigest() != WEIGHT_SHA:
        raise RuntimeError('Official CLIP weight hash mismatch')
    model = build_model(torch.jit.load(str(weight), map_location='cpu').state_dict()).float().to(args.device)
    model.requires_grad_(False); model.eval(); tokenizer = SimpleTokenizer()
    catalogue = json.loads((ROOT / 'data/hm/processed/catalogue.json').read_text())
    if args.limit: catalogue = catalogue[:args.limit]
    images = np.zeros((len(catalogue), 512), np.float32)
    texts = np.zeros_like(images); available = np.zeros((len(catalogue), 2), bool)
    output = ROOT / 'artifacts/hm/clip'; output.mkdir(parents=True, exist_ok=True)
    name = 'features.npz' if args.limit == 0 else f'probe-{args.limit}.npz'
    item_ids = np.asarray([r['article_id'] for r in catalogue])
    if args.resume and (output/name).exists():
        with np.load(output/name, allow_pickle=False) as old:
            if not np.array_equal(old['item_ids'], item_ids): raise RuntimeError('Resume ID mismatch')
            images[:] = old['images']; texts[:] = old['texts']; available[:] = old['available']
    def save():
        temporary = output / (name + '.part')
        with temporary.open('wb') as f:
            np.savez(f, images=images, texts=texts, available=available, item_ids=item_ids)
        temporary.replace(output/name)
    started = time.perf_counter(); errors = []
    with torch.inference_mode():
        for start in range(0, len(catalogue), args.batch_size):
            rows = catalogue[start:start + args.batch_size]
            description = ['; '.join(str(r[x]) for x in ['prod_name','product_type_name',
                'colour_group_name','detail_desc'] if r.get(x)) for r in rows]
            missing_text = [i for i in range(len(rows)) if not available[start+i, 1]]
            if missing_text:
                vectors = F.normalize(model.encode_text(tokenize([description[i] for i in missing_text], tokenizer).to(args.device)), dim=-1)
                positions_text = [start+i for i in missing_text]
                texts[positions_text] = vectors.cpu().numpy(); available[positions_text, 1] = True
            visual, positions = [], []
            for i, row in enumerate(rows):
                if available[start+i, 0]: continue
                item = row['article_id']; path = ROOT / 'data/hm/raw/images' / item[:3] / f'{item}.jpg'
                if not path.exists(): continue
                try: visual.append(preprocess(path)); positions.append(start + i)
                except Exception as error: errors.append({'item_id':item,'error_type':type(error).__name__})
            if visual:
                vectors = F.normalize(model.encode_image(torch.stack(visual).to(args.device)), dim=-1).cpu().numpy()
                images[positions] = vectors; available[positions, 0] = True
            if (start // args.batch_size + 1) % 50 == 0:
                save(); print('encoded items',start + len(rows), 'images',int(available[:,0].sum()),flush=True)
    save()
    report = {'args':vars(args),'items':len(catalogue),'available_images':int(available[:,0].sum()),
              'seconds':time.perf_counter()-started,'weight_sha256':WEIGHT_SHA,'errors':errors,
              'frozen_backbone':True,'text_available':int(available[:,1].sum()),
              'mps_allocated_bytes_end':torch.mps.current_allocated_memory() if args.device == 'mps' else None,
              'mps_driver_bytes_end':torch.mps.driver_allocated_memory() if args.device == 'mps' else None,'feature_sha256':hashlib.sha256((output/name).read_bytes()).hexdigest()}
    (output / (name + '.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
