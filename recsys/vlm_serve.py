"""Local-only structured attribute generation for known public catalogue items."""
import argparse,hashlib,json,threading,time
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import torch
from PIL import Image
from transformers import AutoProcessor,Idefics3ForConditionalGeneration
from recsys.clip_lora import LoRALinear
from recsys.hm_train import ROOT

PAGE=(ROOT/'web/attributes.html').read_text()

class AttributeEngine:
    def __init__(self,seed=42,device='mps'):
        self.device=device;torch.set_num_threads(4)
        directory=ROOT/'artifacts/backbones/smolvlm-500m'
        digest=hashlib.sha256()
        with (directory/'model.safetensors').open('rb') as source:
            while chunk:=source.read(1024*1024):digest.update(chunk)
        if digest.hexdigest()!='d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2':raise RuntimeError('Base weight checksum mismatch')
        checkpoint=ROOT/f'artifacts/hm/vlm-sft-{seed}/best.pt';report=json.loads((ROOT/f'reports/vlm_sft_{seed}.json').read_text())
        for split,expected in report['data_sha256'].items():
            if hashlib.sha256((ROOT/f'data/hm/vlm/{split}.jsonl').read_bytes()).hexdigest()!=expected:raise RuntimeError('Dataset/prompt checksum mismatch')
        sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if sha!=report['checkpoint_sha256']:raise RuntimeError('Adapter checksum mismatch')
        self.processor=AutoProcessor.from_pretrained(directory,local_files_only=True);self.processor.image_processor.do_image_splitting=False;self.processor.image_processor.size={'longest_edge':512}
        self.model=Idefics3ForConditionalGeneration.from_pretrained(directory,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager');self.model.requires_grad_(False)
        for name,module in list(self.model.named_modules()):
            if isinstance(module,torch.nn.Linear) and name.endswith(('q_proj','v_proj')) and 'text_model' in name:
                parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,LoRALinear(module,4,8))
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)['adapters'];parameters=dict(self.model.named_parameters())
        expected={n for n,p in parameters.items() if p.requires_grad}
        if expected!=set(state):raise RuntimeError('Adapter module mismatch')
        for name,value in state.items():parameters[name].data.copy_(value)
        self.model.to(device).eval();self.lock=threading.Lock();self.version=hashlib.sha256((sha+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()).encode()).hexdigest()[:12]
        self.rows={r['article_id']:r for split in ['train','val','test'] for r in [json.loads(x) for x in (ROOT/f'data/hm/vlm/{split}.jsonl').read_text().splitlines()]}
    def predict(self,item):
        if item not in self.rows:raise ValueError('Unknown article_id; available-image catalogue only')
        row=self.rows[item];started=time.perf_counter()
        with self.lock,torch.inference_mode():
            prompt=self.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':row['prompt']}]}],add_generation_prompt=True)
            batch=self.processor(text=prompt,images=[Image.open(ROOT/row['image']).convert('RGB')],return_tensors='pt');count=batch['input_ids'].shape[1];batch={k:v.to(self.device) for k,v in batch.items()}
            generated=self.model.generate(**batch,max_new_tokens=60,do_sample=False);output=self.processor.batch_decode(generated[:,count:],skip_special_tokens=True)[0]
        try:parsed=json.loads(output)
        except json.JSONDecodeError:parsed=None
        return {'article_id':item,'output':output,'attributes':parsed,'version':self.version,'elapsed_ms':(time.perf_counter()-started)*1000,'scope':'Experimental image-derived attributes; not verified facts'}

def main():
    log=logging.getLogger('vlm_requests');log.setLevel(logging.INFO)
    (ROOT/'logs').mkdir(exist_ok=True)
    handler=RotatingFileHandler(ROOT/'logs/vlm_requests.jsonl',maxBytes=1000000,backupCount=2);handler.setFormatter(logging.Formatter('%(message)s'));log.addHandler(handler)
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8767);args=p.parse_args();engine=AttributeEngine()
    class Handler(BaseHTTPRequestHandler):
        def respond(self,code,value,kind="application/json"):
            log.info(json.dumps({'time':time.time(),'method':self.command,'status':code,'version':engine.version}));body=value if isinstance(value,bytes) else value.encode() if isinstance(value,str) else json.dumps(value).encode();self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path=='/' or self.path.startswith('/?'):self.respond(200,(ROOT/'web/attributes.html').read_text(),'text/html; charset=utf-8')
            elif self.path=='/demo-items':
                items=[]
                for split in ['test','val','train']:
                    rows=[json.loads(x) for x in (ROOT/f'data/hm/vlm/{split}.jsonl').read_text().splitlines()]
                    items.extend({'article_id':r['article_id'],'split':split,'reference':json.loads(r['answer'])} for r in rows)
                self.respond(200,{'items':items})
            elif self.path.startswith('/images/'):
                item=self.path[len('/images/'):]
                if item not in engine.rows:self.respond(404,{'error':'Unknown item'});return
                self.respond(200,(ROOT/engine.rows[item]['image']).read_bytes(),'image/jpeg')
            elif self.path=='/health':self.respond(200,{'status':'ok','version':engine.version,'items':len(engine.rows),'device':engine.device})
            else:self.respond(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/attributes':self.respond(404,{'error':'Not found'});return
            try:
                self.connection.settimeout(5);size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=4096:raise ValueError('Invalid request size')
                body=json.loads(self.rfile.read(size))
                if not isinstance(body,dict):raise ValueError('JSON object required')
                item=body.get('article_id')
                if not isinstance(item,str):raise ValueError('article_id must be a string')
                self.respond(200,engine.predict(item))
            except (ValueError,TypeError,TimeoutError) as error:self.respond(400,{'error':str(error)})
            except Exception:self.respond(500,{'error':'Inference failed'})
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);print(f'VLM service http://127.0.0.1:{args.port} version {engine.version}',flush=True);server.serve_forever()

if __name__=='__main__':main()
