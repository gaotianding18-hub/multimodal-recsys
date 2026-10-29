"""Loopback-only H&M real-data retrieval/ranking research service."""
import argparse
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from .hm_train import ROOT, DATA, Retrieval
from .hm_rank import Ranker, features
from .hm_hybrid import recent_order, load_retriever, ranked as hybrid_ranked


class Engine:
    def __init__(self, release_path=None):
        torch.set_num_threads(1)
        choice=json.loads((ROOT/'reports/hm_id_summary.json').read_text())['selected_on_validation']
        directory=ROOT/f"artifacts/hm/id-{choice['negative_strategy']}-{choice['seed']}"
        report=json.loads((ROOT/f"reports/hm_id_{choice['negative_strategy']}_{choice['seed']}.json").read_text())
        actual=hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest()
        if actual!=report['model_sha256']:raise RuntimeError('Model version mismatch')
        checkpoint=torch.load(directory/'model.pt',weights_only=True)
        self.ids=checkpoint['item_ids'];self.mapping={x:i+1 for i,x in enumerate(self.ids)}
        self.model=Retrieval(len(self.ids),checkpoint['dim']);self.model.load_state_dict(checkpoint['state']);self.model.eval()
        with torch.inference_mode(): self.vectors=self.model.items(torch.arange(len(self.ids)+1)).clone()
        stored=np.load(directory/'index.npy')
        if not np.allclose(stored,self.vectors[1:].numpy(),rtol=1e-5,atol=1e-6):raise RuntimeError('Index inconsistent with model')
        rank=torch.load(directory/'ranker.pt',weights_only=True)
        if rank['retriever']!=choice:raise RuntimeError('Ranker retriever version mismatch')
        self.ranker=Ranker(rank['dim']);self.ranker.load_state_dict(rank['state']);self.ranker.eval()
        self.popularity=rank['popularity'];self.pool=rank['candidate_pool']
        self.version=hashlib.sha256((actual+hashlib.sha256((directory/'ranker.pt').read_bytes()).hexdigest()).encode()).hexdigest()[:12]
        catalogue=json.loads((DATA/'catalogue.json').read_text())
        if [r['article_id'] for r in catalogue]!=self.ids:raise RuntimeError('Catalogue/index ID order mismatch')
        self.articles={r['article_id']:r for r in catalogue}
        config=json.loads((ROOT/'configs/hm.json').read_text())
        if config!=report['protocol']:raise RuntimeError('Serving configuration differs from model protocol')
        self.recent=recent_order(self.ids,config)
        self.semantic=None
        release=release_path or ROOT/'artifacts/hm/release.json'
        if release.exists():
            manifest=json.loads(release.read_text());directory=ROOT/manifest['retriever']
            if hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest()!=manifest['retriever_sha256']:
                raise RuntimeError('Release retriever checksum mismatch')
            model,ids,vectors=load_retriever(directory)
            if ids!=self.ids:raise RuntimeError('Release catalogue mismatch')
            rankpath=directory/manifest['ranker_file']
            if hashlib.sha256(rankpath.read_bytes()).hexdigest()!=manifest['ranker_sha256']:
                raise RuntimeError('Release ranker checksum mismatch')
            checkpoint=torch.load(rankpath,weights_only=True)
            if checkpoint['retriever_sha256']!=manifest['retriever_sha256']:
                raise RuntimeError('Release ranker/retriever mismatch')
            ranker=Ranker(checkpoint['dim']);ranker.load_state_dict(checkpoint['state']);ranker.eval()
            self.semantic=(model,vectors,ranker,checkpoint['recent'],manifest)
            self.version=hashlib.sha256((self.version+manifest['retriever_sha256']+manifest['ranker_sha256']).encode()).hexdigest()[:12]
        self.visual=None
        visual_summary=ROOT/'reports/visual_summary.json'
        if visual_summary.exists():
            selected=json.loads(visual_summary.read_text())['multimodal_selected_on_validation']
            name=f"visual-{selected['source']}-{selected['mode']}-{selected['seed']}"
            visual_report=json.loads((ROOT/f"reports/visual_{selected['source']}_{selected['mode']}_{selected['seed']}.json").read_text())
            directory=ROOT/'artifacts/hm'/name
            if hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest()!=visual_report['model_sha256']:
                raise RuntimeError('Visual model checksum mismatch')
            model,ids,vectors=load_retriever(directory)
            mapping={x:i+1 for i,x in enumerate(ids)}
            if not set(ids)<=set(self.ids):raise RuntimeError('Unknown visual catalogue items')
            from collections import Counter
            counts=Counter()
            for line in (ROOT/'data/hm/visual/queries.jsonl').open():
                query=json.loads(line)
                if query['split']=='train':counts.update(query['targets'])
            popular=sorted(ids,key=lambda x:(-counts[x],x))
            self.visual=(model,ids,mapping,vectors,popular,selected)
            self.version=hashlib.sha256((self.version+visual_report['model_sha256']).encode()).hexdigest()[:12]
        self.demos={}
        for line in (DATA/'queries.jsonl').open():
            q=json.loads(line)
            if q['split']=='test':
                self.demos[f"demo-{len(self.demos)+1}"]=q['history']
                if len(self.demos)==30:break
        if self.visual:
            for line in (ROOT/'data/hm/visual/queries.jsonl').open():
                q=json.loads(line)
                if q['split']=='test':self.demos[f"visual-demo-{sum(x.startswith('visual-demo-') for x in self.demos)+1}"]=q['history']
                if sum(x.startswith('visual-demo-') for x in self.demos)>=10:break
        fingerprint=hashlib.sha256(self.version.encode())
        for path in [DATA/'catalogue.json',ROOT/'configs/hm.json',
                     ROOT/'recsys/hm_serve.py',ROOT/'recsys/hm_hybrid.py',
                     ROOT/'recsys/hm_rank.py',ROOT/'recsys/hm_train.py',ROOT/'recsys/advanced.py',ROOT/'recsys/hm_multimodal.py']:
            fingerprint.update(path.read_bytes())
        self.version=fingerprint.hexdigest()[:12]

    def recommend(self,payload):
        if not isinstance(payload,dict):raise ValueError('Expected a JSON object')
        method=payload.get('method','repeat')
        if method not in self.methods():raise ValueError('Unknown or unavailable method')
        k=payload.get('k',12)
        if type(k) is not int or not 1<=k<=50:raise ValueError('k must be an integer 1..50')
        if 'history' in payload:
            history=payload['history']
            if not isinstance(history,list) or len(history)>200 or any(not isinstance(x,str) for x in history):
                raise ValueError('history must be a list of at most 200 string item IDs')
        else:
            user=payload.get('user_id','demo-1')
            if user not in self.demos:raise ValueError('Unknown demo user')
            history=self.demos[user]
        h=[self.mapping[x] for x in history if x in self.mapping][-30:]
        unknown=sum(x not in self.mapping for x in history)
        label='Repeat-history + recent-popularity baseline'
        candidate_count=len(self.ids)
        if method=='visual':
            model,ids,mapping,vectors,popular,selected=self.visual
            vh=[mapping[x] for x in history if x in mapping][-10:]
            unknown=sum(x not in mapping for x in history);candidate_count=len(ids)
            label=f"Small multimodal subset: {selected['source']}/{selected['mode']}; 590 candidates"
            if vh:
                with torch.inference_mode():
                    query=torch.tensor([[0]*(10-len(vh))+vh]);values=(model.users(query)@vectors[1:].T).flatten()
                    scores,positions=values.topk(k);chosen=[ids[i] for i in positions.tolist()]
                ranked=scores;fallback=False
            else:chosen=popular[:k];ranked=torch.arange(k,0,-1).float();fallback=True
            order=torch.tensor([self.mapping[x] for x in chosen]);h=[self.mapping[ids[i-1]] for i in vh]
        elif not h or method=='repeat':
            order=torch.tensor(list(dict.fromkeys(list(reversed(h))+self.recent))[:k])
            ranked=torch.arange(k,0,-1).float();fallback=not h
        elif method=='semantic':
            model,vectors,ranker,recent,manifest=self.semantic
            hist=torch.tensor([[0]*(30-len(h))+h])
            top,_,scores=hybrid_ranked(model,ranker,vectors,hist,recent,k=k)
            order=top.flatten();ranked=scores.flatten();fallback=False
            label=manifest['description']
        else:
            label='ID/history retrieval + interest ranking (experimental)'
            with torch.inference_mode():
                hist=torch.tensor([[0]*(30-len(h))+h]);scores=self.model.users(hist)@self.vectors[1:].T
                candidates=scores.topk(self.pool,dim=-1).indices.flatten()+1
                expanded=hist.expand(self.pool,-1)
                rank_scores=self.ranker(self.vectors[expanded],self.vectors[candidates],expanded.ne(0),
                                       features(expanded,candidates,self.popularity))
                positions=rank_scores.topk(k).indices;order=candidates[positions];ranked=rank_scores[positions];fallback=False
        items=[]
        for index,score in zip(order.tolist(),ranked.tolist()):
            item=self.ids[index-1];row=self.articles[item]
            items.append({'item_id':item,'name':row['prod_name'],'type':row['product_type_name'],
                          'color':row['colour_group_name'],'description':row['detail_desc'],'score':score,
                          'image_url':f'/images/{item}' if (ROOT/'data/hm/raw/images'/item[:3]/f'{item}.jpg').exists() else None})
        return {'items':items,'version':self.version,'candidate_items':candidate_count,
                'history_used':len(h),'unknown_history_items':unknown,'fallback':fallback,
                'model':label,'method':method,'available_methods':self.methods(),
                'history_items':[{'item_id':self.ids[i-1],'name':self.articles[self.ids[i-1]]['prod_name']} for i in h],
                'as_of':'2020-09-16 for demo profiles','score_meaning':'ordinal baseline score' if fallback or method=='repeat' else 'preference logit, not purchase probability'}

    def methods(self):
        return ['repeat','id']+(['semantic'] if self.semantic else [])+(['visual'] if self.visual else [])


PAGE=(ROOT/'web/recommend.html').read_text()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    engine=Engine()
    logdir=ROOT/'artifacts/hm/service_logs';logdir.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('hm_service');logger.setLevel(logging.INFO)
    handler=RotatingFileHandler(logdir/'requests.jsonl',maxBytes=5*1024*1024,backupCount=2)
    logger.addHandler(handler)
    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,body,kind='application/json'):
            data=body if isinstance(body,bytes) else body.encode() if isinstance(body,str) else json.dumps(body,ensure_ascii=False).encode()
            self.response_status=status
            self.send_response(status);self.send_header('Content-Type',kind if kind.startswith('image/') else kind+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/':self.respond(200,(ROOT/'web/recommend.html').read_text(),'text/html')
            elif self.path=='/health':self.respond(200,{'status':'ok','version':engine.version,'candidate_items':len(engine.ids),'methods':engine.methods()})
            elif self.path=='/demo-users':self.respond(200,{'users':list(engine.demos)})
            elif self.path.startswith('/images/'):
                item=self.path[len('/images/'):]
                if item not in engine.articles:return self.respond(404,{'error':'Unknown item'})
                path=ROOT/'data/hm/raw/images'/item[:3]/f'{item}.jpg'
                if not path.exists():return self.respond(404,{'error':'Image unavailable'})
                self.respond(200,path.read_bytes(),'image/jpeg')
            else:self.respond(404,{'error':'Not found'})
        def do_POST(self):
            started=time.perf_counter();self.response_status=500;method='unknown'
            self.connection.settimeout(5)
            if self.path!='/recommend':return self.respond(404,{'error':'Not found'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(length));start=time.perf_counter()
                if isinstance(body,dict) and body.get('method','repeat') in engine.methods():method=body.get('method','repeat')
                result=engine.recommend(body)
                result['inference_ms']=(time.perf_counter()-start)*1000;self.respond(200,result)
            except (ValueError,TypeError) as error:self.respond(400,{'error':str(error)})
            except TimeoutError:self.respond(408,{'error':'Request body timeout'})
            except Exception:self.respond(500,{'error':'Internal inference error'})
            finally:
                logger.info(json.dumps({'unix_time':time.time(),'version':engine.version,'method':method,
                                        'status':self.response_status,'elapsed_ms':(time.perf_counter()-started)*1000}))
        def log_message(self,format,*args):pass
    print(f'H&M service http://127.0.0.1:{args.port}/ version {engine.version}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
