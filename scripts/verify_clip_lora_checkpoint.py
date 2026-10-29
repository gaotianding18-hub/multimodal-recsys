"""Reload actual CLIP LoRA and compare reproduced embeddings to saved cache."""
import hashlib,json
import numpy as np
import torch
from third_party.clip.model import build_model
from third_party.clip.simple_tokenizer import SimpleTokenizer
from recsys.clip_features import WEIGHT_SHA,preprocess,tokenize
from recsys.clip_lora import inject
from recsys.hm_train import ROOT

if __name__=='__main__':
    torch.set_num_threads(4);weight=ROOT/'artifacts/backbones/ViT-B-32.pt'
    if hashlib.sha256(weight.read_bytes()).hexdigest()!=WEIGHT_SHA:raise RuntimeError('Base hash mismatch')
    path=ROOT/'artifacts/hm/clip-lora-42/best.pt';checkpoint=torch.load(path,weights_only=True)
    if checkpoint['weight_sha256']!=WEIGHT_SHA:raise RuntimeError('Adapter/base mismatch')
    model=build_model(torch.jit.load(str(weight),map_location='cpu').state_dict()).float()
    adapters=inject(model,checkpoint['rank'],checkpoint['alpha'],checkpoint['blocks'])
    if set(adapters)!=set(checkpoint['adapters']):raise RuntimeError('Adapter keys mismatch')
    for n,p in adapters.items():p.data.copy_(checkpoint['adapters'][n])
    model=model.to('mps').eval();tokenizer=SimpleTokenizer()
    rows=json.loads((ROOT/'data/hm/visual/catalogue.json').read_text());indices=[]
    for split in ['train','val','test']:indices.append(next(i for i,r in enumerate(rows) if r['pair_split']==split))
    selected=[rows[i] for i in indices];description=['; '.join(str(r[k]) for k in ['prod_name','product_type_name','colour_group_name','detail_desc'] if r.get(k)) for r in selected]
    images=torch.stack([preprocess(ROOT/'data/hm/raw/images'/r['article_id'][:3]/(r['article_id']+'.jpg')) for r in selected])
    with torch.inference_mode():
        visual=torch.nn.functional.normalize(model.encode_image(images.to('mps')),dim=-1).cpu().numpy()
        text=torch.nn.functional.normalize(model.encode_text(tokenize(description,tokenizer).to('mps')),dim=-1).cpu().numpy()
    cache=ROOT/'artifacts/hm/clip-lora-42/features.npz'
    with np.load(cache) as data:
        if not np.allclose(visual,data['images'][indices],atol=1e-5,rtol=1e-5) or not np.allclose(text,data['texts'][indices],atol=1e-5,rtol=1e-5):raise RuntimeError('Reload embeddings differ')
    result={'checked_items':len(indices),'splits':['train','val','test'],'cache_consistency':True,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}
    (ROOT/'reports/clip_lora_checkpoint_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
