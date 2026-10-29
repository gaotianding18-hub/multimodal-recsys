import argparse
import hashlib
import json
import logging
import resource
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from torch.nn import functional as F

from .core import ROOT, DATA, TwoTower, Ranker, tensor


class Engine:
    def __init__(self, run):
        self.path=ROOT/'artifacts/runs'/run
        manifest=json.loads((self.path/'manifest.json').read_text())
        for name in ['model.pt','index.npy','features.npz','serving.json']:
            key=name.split('.')[0]+'_sha256'
            if hashlib.sha256((self.path/name).read_bytes()).hexdigest()!=manifest[key]:
                raise ValueError('Artifact checksum mismatch: '+name)
        checkpoint=torch.load(self.path/'model.pt',map_location='cpu',weights_only=True)
        self.model=TwoTower(checkpoint['mode']);self.model.load_state_dict(checkpoint['model']);self.model.eval()
        self.ranker=Ranker();self.ranker.load_state_dict(checkpoint['ranker']);self.ranker.eval()
        self.index=tensor(np.load(self.path/'index.npy',allow_pickle=False))
        self.meta=json.loads((self.path/'serving.json').read_text())
        self.version=manifest['model_sha256'][:12]
        features=np.load(self.path/'features.npz',allow_pickle=False)
        with torch.inference_mode():
            actual=self.model.encode_items(tensor(features['images']),tensor(features['texts']))
            if not torch.allclose(actual,self.index,atol=1e-6):raise ValueError('Offline/serving index mismatch')
        self.fallback=np.mean([v for k,v in self.meta['histories'].items() if int(k[1:])<=6],axis=0)

    def recommend(self,user_id,scenario,k=3):
        if not isinstance(user_id,str) or len(user_id)>64:raise ValueError('user_id must be a short string')
        if type(scenario) is not int or scenario not in range(1,7):raise ValueError('scenario must be integer 1..6')
        if type(k) is not int or not 1<=k<=6:raise ValueError('k must be integer 1..6')
        start=time.perf_counter()
        catalogue='a' if scenario<=2 else 'b' if scenario<=4 else 'c'
        candidates=[i for i,x in enumerate(self.meta['items']) if x['catalogue']==catalogue]
        known=user_id in self.meta['histories']
        history=self.meta['histories'].get(user_id,self.fallback)
        context=self.meta['scenarios'][str(scenario)]
        with torch.inference_mode():
            user=self.model.encode_user(tensor(history)[None],tensor(context)[None])
            retrieval=(user@self.index[candidates].T).squeeze(0)
            pool=torch.argsort(retrieval,descending=True,stable=True)[:6]
            ids=[candidates[i] for i in pool.tolist()]
            scores=self.ranker(user,self.index[ids])
            order=torch.argsort(scores,descending=True,stable=True)[:k].tolist()
        return {'model_version':self.version,'run':self.path.name,'unknown_user_fallback':not known,
            'latency_ms':(time.perf_counter()-start)*1000,'candidate_count':len(candidates),'retrieved_count':6,
            'items':[{**self.meta['items'][ids[i]],'score':float(scores[i]),'retrieval_score':float(retrieval[pool[i]]),
                      'image_url':f"/images/{ids[i]+1}.png"} for i in order],
            'score_meaning':'relative_preference_score_not_click_probability'}


HTML='''<!doctype html><meta charset="utf-8"><title>商品推荐研究演示</title>
<style>body{font-family:system-ui;max-width:900px;margin:40px auto}input,select,button{padding:8px;margin:5px}article{display:inline-block;width:260px;vertical-align:top;margin:12px}img{width:220px;height:220px;object-fit:contain}</style>
<h1>商品推荐研究演示</h1><p>VOGUE 小规模情境偏好数据 · 本机研究服务</p>
<p>依据调查前的偏好评分和当前情境推荐；不代表线上点击率或购买概率。</p>
<label>用户 <input id="user" value="s9"></label><label>情境 <select id="scenario"><option value="1">秋季农场出行</option><option value="2">校园日常</option><option value="3">办公室</option><option value="4">朋友聚餐</option><option value="5">冬季城市游览</option><option value="6">雨后公园</option></select></label><button id="go">推荐</button><p id="status"></p><section id="results"></section>
<script>document.getElementById('go').onclick=async()=>{let status=document.getElementById('status'),root=document.getElementById('results');root.replaceChildren();try{let r=await fetch('/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:document.getElementById('user').value,scenario:Number(document.getElementById('scenario').value),k:3})});let d=await r.json();if(!r.ok)throw Error(d.error);status.textContent='版本 '+d.model_version+' · '+d.latency_ms.toFixed(2)+'ms'+(d.unknown_user_fallback?' · 使用未知用户回退':'');for(let item of d.items){let a=document.createElement('article'),im=document.createElement('img'),p=document.createElement('p');im.src=item.image_url;p.textContent=item.name+' · 分数 '+item.score.toFixed(3);a.append(im,p);root.append(a)}}catch(e){status.textContent=e.message}};</script>'''


def handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,value,content_type='application/json'):
            data=json.dumps(value).encode() if content_type=='application/json' else value
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/health':return self.send(200,{'status':'ok','version':engine.version,
                'process_peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            if path=='/':return self.send(200,HTML.encode(),'text/html; charset=utf-8')
            if path.startswith('/images/'):
                name=path.removeprefix('/images/')
                if name in {f'{i}.png' for i in range(1,37)}:
                    return self.send(200,(DATA/f'metadata/item_{name}').read_bytes(),'image/png')
            self.send(404,{'error':'not_found'})
        def do_POST(self):
            if self.path!='/recommend':return self.send(404,{'error':'not_found'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=4096:raise ValueError('body size must be 1..4096 bytes')
                payload=json.loads(self.rfile.read(size))
                if not isinstance(payload,dict):raise ValueError('body must be JSON object')
                result=engine.recommend(payload.get('user_id'),payload.get('scenario'),payload.get('k',3))
                self.send(200,result)
            except (ValueError,TypeError,KeyError) as exc:self.send(400,{'error':str(exc)})
            except Exception:
                logging.exception('Inference failed');self.send(500,{'error':'inference_failed'})
        def log_message(self,format,*args):logging.info(format,*args)
    return Handler


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);parser.add_argument('--run')
    args=parser.parse_args();torch.set_num_threads(1)
    run=args.run or json.loads((ROOT/'reports/experiments.json').read_text())['selected_for_service']
    engine=Engine(run);logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    logging.info('Loaded %s version %s; loopback-only port %s',run,engine.version,args.port)
    ThreadingHTTPServer(('127.0.0.1',args.port),handler(engine)).serve_forever()
