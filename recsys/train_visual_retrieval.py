"""Actual multimodal purchase retrieval on declared available-image subset."""
import argparse,copy,hashlib,json,time
import numpy as np
import torch
from recsys.hm_train import ROOT,metric,Retrieval
from recsys.hm_multimodal import MultimodalRetrieval,candidate_masks
from recsys.advanced import contrastive_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=['frozen','lora'],default='frozen');p.add_argument('--mode',choices=['id','text','image','mean','gate'],default='gate');p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=10);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    rows=json.loads((ROOT/'data/hm/visual/catalogue.json').read_text());ids=[r['article_id'] for r in rows];mapping={x:i+1 for i,x in enumerate(ids)}
    path=ROOT/'artifacts/hm/clip/features.npz' if args.source=='frozen' else ROOT/'artifacts/hm/clip-lora-42/features.npz'
    with np.load(path,allow_pickle=False) as data:
        lookup={item:i for i,item in enumerate(data['item_ids'].tolist())};positions=[lookup[i] for i in ids]
        images=torch.from_numpy(data['images'][positions].copy());texts=torch.from_numpy(data['texts'][positions].copy());available=torch.from_numpy(data['available'][positions].copy())
    if not available.all():raise RuntimeError('Visual subset requires actual image and text for every candidate')
    histories=[];positives=[];keys=[];known={};evaluation={'val':[],'test':[]}
    for line in (ROOT/'data/hm/visual/queries.jsonl').open():
        q=json.loads(line);h=[mapping[x] for x in q['history']][-10:];h=[0]*(10-len(h))+h;targets=[mapping[x] for x in q['targets']]
        if q['split']=='train':
            histories.append(h);positives.append(targets);keys.append(q['user_id']);known.setdefault(q['user_id'],set()).update([x for x in h if x]+targets)
        else:evaluation[q['split']].append((h,set(q['targets'])))
    histories=np.asarray(histories);model=Retrieval(len(ids)) if args.mode=='id' else MultimodalRetrieval(images,texts,available,mode=args.mode);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    output=ROOT/f'artifacts/hm/visual-{args.source}-{args.mode}-{args.seed}';output.mkdir(parents=True,exist_ok=True)
    def evaluate(split):
        model.eval();totals=np.zeros(3)
        with torch.inference_mode():
            index=model.items(torch.arange(1,len(ids)+1))
            for start in range(0,len(evaluation[split]),64):
                batch=evaluation[split][start:start+64];h=torch.tensor([r[0] for r in batch]);top=(model.users(h)@index.T).topk(12,dim=-1).indices.numpy()
                for order,(_,target) in zip(top,batch):totals+=metric([ids[i] for i in order],target)
        return dict(zip(['recall@12','ndcg@12','map@12'],(totals/len(evaluation[split])).tolist()))
    best=-1;state=None;logs=[];started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();losses=[];order=rng.permutation(len(histories))
        for start in range(0,len(order),64):
            pos=order[start:start+64];cand,positive,allowed=candidate_masks(pos,positives,keys,known,rng,len(ids),128)
            optimizer.zero_grad(set_to_none=True);loss=contrastive_loss(model.users(torch.from_numpy(histories[pos])),model.items(torch.from_numpy(cand)),torch.from_numpy(positive),torch.from_numpy(allowed))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite multimodal loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        val=evaluate('val');record={'epoch':epoch+1,'loss':float(np.mean(losses)),'val':val};logs.append(record)
        if val['ndcg@12']>best:best=val['ndcg@12'];state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);test=evaluate('test')
    torch.save({'state':state,'dim':64,'item_ids':ids,'args':{**vars(args),'mode':args.mode},'feature_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},output/'model.pt')
    model.eval()
    with torch.inference_mode():np.save(output/'index.npy',model.items(torch.arange(1,len(ids)+1)).numpy())
    report={'args':vars(args),'train_queries':len(histories),'candidates':len(ids),'logs':logs,'selection_val_ndcg':best,'test':test,'seconds':time.perf_counter()-started,
            'feature_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((output/'model.pt').read_bytes()).hexdigest(),
            'evaluation':'Conditional image-available histories and targets, 590 full candidates. NOT original 41585-catalogue metric.',
            'history_length':10,'limitations':['Nonrandom image availability selection','150val/124test query conditional subset','Image/text arms both include ID features','All modes have same training budget and sampling']}
    (ROOT/f'reports/visual_{args.source}_{args.mode}_{args.seed}.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['args','test','seconds']}),flush=True)


if __name__=='__main__':main()
