"""True CLIP-internal LoRA tuning on grouped public product image/text pairs."""
import argparse,copy,hashlib,json,time
import numpy as np
import torch
from torch.nn import functional as F
from third_party.clip.model import build_model
from third_party.clip.simple_tokenizer import SimpleTokenizer
from .clip_features import WEIGHT_SHA,preprocess,tokenize
from .clip_lora import inject,symmetric_loss
from .hm_train import ROOT


def base_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='mps',choices=['cpu','mps']);parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    weight=ROOT/'artifacts/backbones/ViT-B-32.pt'
    if hashlib.sha256(weight.read_bytes()).hexdigest()!=WEIGHT_SHA:raise RuntimeError('Weight SHA mismatch')
    model=build_model(torch.jit.load(str(weight),map_location='cpu').state_dict()).float()
    adapters=inject(model);before={n:p.detach().clone() for n,p in adapters.items()};base_before=base_digest(model)
    model=model.to(args.device);adapters={n:p for n,p in model.named_parameters() if p.requires_grad};tokenizer=SimpleTokenizer()
    source=ROOT/'data/hm/visual/catalogue.json';rows=json.loads(source.read_text())
    descriptions=['; '.join(str(r[k]) for k in ['prod_name','product_type_name','colour_group_name','detail_desc'] if r.get(k)) for r in rows]
    images=torch.stack([preprocess(ROOT/'data/hm/raw/images'/r['article_id'][:3]/(r['article_id']+'.jpg')) for r in rows])
    tokens=tokenize(descriptions,tokenizer)
    exact=[hashlib.sha256((ROOT/'data/hm/raw/images'/r['article_id'][:3]/(r['article_id']+'.jpg')).read_bytes()).hexdigest() for r in rows]
    positive=np.array([[descriptions[i]==descriptions[j] or exact[i]==exact[j] for j in range(len(rows))] for i in range(len(rows))])
    splits={s:np.array([i for i,r in enumerate(rows) if r['pair_split']==s]) for s in ['train','val','test']}
    output=ROOT/f'artifacts/hm/clip-lora-{args.seed}';output.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(adapters.values(),lr=1e-4,weight_decay=.01)
    def encode(indices):
        image,text=[],[];model.eval()
        with torch.inference_mode():
            for start in range(0,len(indices),args.batch_size):
                ix=indices[start:start+args.batch_size]
                image.append(F.normalize(model.encode_image(images[ix].to(args.device)),dim=-1).cpu())
                text.append(F.normalize(model.encode_text(tokens[ix].to(args.device)),dim=-1).cpu())
        return torch.cat(image).numpy(),torch.cat(text).numpy()
    def evaluate(indices):
        image,text=encode(indices);score=image@text.T;mask=positive[np.ix_(indices,indices)];result={}
        for direction,scores,valid in [('image_to_text',score,mask),('text_to_image',score.T,mask.T)]:
            order=np.argsort(-scores,axis=1,kind='stable')
            for k in [1,5,10]:result[f'{direction}_recall@{k}']=float(np.take_along_axis(valid,order[:,:k],axis=1).any(1).mean())
        return result
    started=time.perf_counter();baseline={'val':evaluate(splits['val']),'test':evaluate(splits['test'])};logs=[];best=-1;best_state=None;gradient_evidence={}
    for epoch in range(args.epochs):
        model.train();losses=[]
        for offset in range(0,len(splits['train']),args.batch_size):
            if offset==0:order=rng.permutation(splits['train'])
            ix=order[offset:offset+args.batch_size]
            if len(ix)<2:continue
            optimizer.zero_grad(set_to_none=True)
            visual=model.encode_image(images[ix].to(args.device));text=model.encode_text(tokens[ix].to(args.device))
            loss=symmetric_loss(visual,text,torch.as_tensor(positive[np.ix_(ix,ix)],device=args.device))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite LoRA loss')
            loss.backward()
            for tower in ['visual.','transformer.']:
                gradient_evidence[tower]=max(gradient_evidence.get(tower,0),sum(float(p.grad.detach().abs().sum().cpu()) for n,p in adapters.items() if n.startswith(tower) and p.grad is not None))
            optimizer.step();losses.append(float(loss.detach().cpu()))
        val=evaluate(splits['val']);record={'epoch':epoch+1,'loss':float(np.mean(losses)),'val':val};logs.append(record);print(json.dumps(record),flush=True)
        score=(val['image_to_text_recall@1']+val['text_to_image_recall@1'])/2
        if score>best:best=score;best_state={n:p.detach().cpu().clone() for n,p in adapters.items()}
        (output/'logs.json').write_text(json.dumps(logs,indent=2))
    torch.save({'adapters':{n:p.detach().cpu() for n,p in adapters.items()},'rank':4,'alpha':8,'blocks':1,'weight_sha256':WEIGHT_SHA},output/'last.pt')
    changed={n:float((p.detach().cpu()-before[n]).abs().sum()) for n,p in adapters.items()}
    base_after=base_digest(model)
    if base_after!=base_before:raise RuntimeError('Frozen base parameters changed')
    if not all(gradient_evidence[t]>0 for t in ['visual.','transformer.']):raise RuntimeError('Missing actual backbone adapter gradients')
    for n,p in adapters.items():p.data.copy_(best_state[n].to(args.device))
    test=evaluate(splits['test']);all_indices=np.arange(len(rows));visual,text=encode(all_indices)
    np.savez(output/'features.npz',images=visual,texts=text,available=np.ones((len(rows),2),bool),item_ids=np.array([r['article_id'] for r in rows]))
    torch.save({'adapters':best_state,'rank':4,'alpha':8,'blocks':1,'weight_sha256':WEIGHT_SHA},output/'best.pt')
    report={'args':vars(args),'task':'Grouped product image-text contrastive CLIP LoRA fine-tuning, not purchase prediction','split_items':{s:len(v) for s,v in splits.items()},
            'trainable_parameters':sum(p.numel() for p in adapters.values()),'target_modules':list(adapters),'base_sha_before':base_before,'base_sha_after':base_after,
            'adapter_l1_change_last':changed,'gradient_l1_evidence':gradient_evidence,'baseline':baseline,'logs':logs,'test':test,'seconds':time.perf_counter()-started,
            'mps_allocated_end_bytes':torch.mps.current_allocated_memory() if args.device=='mps' else None,'mps_driver_end_bytes':torch.mps.driver_allocated_memory() if args.device=='mps' else None,
            'checkpoint_sha256':hashlib.sha256((output/'best.pt').read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'limitations':['590 available-image products selected non-randomly','Per-run seed; stability assessed separately','84 held-out items, not large VLM fine-tuning','Pairs split by family/exact image; behavior protocol is separate','End memory values are not peak VRAM']}
    (ROOT/f'reports/clip_lora_{args.seed}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
