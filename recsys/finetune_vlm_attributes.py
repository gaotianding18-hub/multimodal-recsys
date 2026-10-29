"""Grouped attribute SFT with validation selection and held-out generation."""
import argparse,copy,hashlib,json,time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Idefics3ForConditionalGeneration
from recsys.hm_train import ROOT
from recsys.clip_lora import LoRALinear
from recsys.train_generative_vlm import frozen_digest
from recsys.vlm_metrics import attribute_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=3);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='mps');args=p.parse_args()
    torch.manual_seed(args.seed);torch.set_num_threads(4);rng=np.random.default_rng(args.seed);start=time.perf_counter()
    out=ROOT/f'artifacts/hm/vlm-sft-{args.seed}';out.mkdir(parents=True,exist_ok=True)
    processor=AutoProcessor.from_pretrained(ROOT/'artifacts/backbones/smolvlm-500m',local_files_only=True)
    processor.image_processor.do_image_splitting=False;processor.image_processor.size={'longest_edge':512}
    model=Idefics3ForConditionalGeneration.from_pretrained(ROOT/'artifacts/backbones/smolvlm-500m',local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager');model.requires_grad_(False)
    for name,module in list(model.named_modules()):
        if isinstance(module,torch.nn.Linear) and name.endswith(('q_proj','v_proj')) and 'text_model' in name:
            parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,LoRALinear(module,4,8))
    model.to(args.device);adapters={n:p for n,p in model.named_parameters() if p.requires_grad};base_sha=frozen_digest(model)
    before={n:p.detach().cpu().clone() for n,p in adapters.items()};opt=torch.optim.AdamW(adapters.values(),lr=1e-4)
    splits={s:[json.loads(x) for x in (ROOT/f'data/hm/vlm/{s}.jsonl').read_text().splitlines()] for s in ['train','val','test']}
    def encode(row,answer=False):
        image=Image.open(ROOT/row['image']).convert('RGB');messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['prompt']}]}]
        prompt=processor.apply_chat_template(messages,add_generation_prompt=True);prefix=processor(text=prompt,images=[image],return_tensors='pt');count=prefix['input_ids'].shape[1]
        if not answer:return {k:v.to(args.device) for k,v in prefix.items()},count
        batch=processor(text=prompt+row['answer']+processor.tokenizer.eos_token,images=[image],return_tensors='pt')
        if not torch.equal(prefix['input_ids'][0],batch['input_ids'][0,:count]):raise RuntimeError('Prompt boundary mismatch')
        labels=batch['input_ids'].clone();labels[:,:count]=-100
        return {k:v.to(args.device) for k,v in batch.items()},labels.to(args.device)
    def evaluate(rows,label):
        model.eval();predictions=[];records=[]
        with torch.inference_mode():
            for i,row in enumerate(rows):
                batch,count=encode(row);generated=model.generate(**batch,max_new_tokens=60,do_sample=False)
                answer=processor.batch_decode(generated[:,count:],skip_special_tokens=True)[0];predictions.append(answer);records.append({'article_id':row['article_id'],'output':answer,'target':json.loads(row['answer'])})
                if (i+1)%25==0:print(label,i+1,flush=True)
        (out/f'{label}.json').write_text(json.dumps(records,indent=2));return attribute_metrics(predictions,[json.loads(r['answer']) for r in rows])
    baseline={'val':evaluate(splits['val'],'baseline_val'),'test':evaluate(splits['test'],'baseline_test')}
    logs=[];best=-1;beststate=None;gradient=0
    for epoch in range(args.epochs):
        model.train();losses=[]
        for step,index in enumerate(rng.permutation(len(splits['train']))):
            batch,labels=encode(splits['train'][index],True);opt.zero_grad(set_to_none=True);loss=model(**batch,labels=labels).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();gradient=max(gradient,sum(float(p.grad.detach().abs().sum().cpu()) for p in adapters.values() if p.grad is not None));opt.step();losses.append(float(loss.detach().cpu()))
            if (step+1)%100==0:print('train',epoch+1,step+1,float(np.mean(losses)),flush=True)
        val=evaluate(splits['val'],f'epoch{epoch+1}_val');record={'epoch':epoch+1,'loss':float(np.mean(losses)),'val':val};logs.append(record);print(json.dumps(record),flush=True)
        score=(val['product_type_macro_recall']+val['color_macro_recall'])/2
        if score>best:
            best=score;beststate={n:p.detach().cpu().clone() for n,p in adapters.items()};torch.save({'adapters':beststate,'rank':4,'alpha':8,'epoch':epoch+1},out/'best.pt')
        (out/'logs.json').write_text(json.dumps(logs,indent=2))
    torch.save({'adapters':{n:p.detach().cpu() for n,p in adapters.items()},'rank':4,'alpha':8},out/'last.pt')
    for n,p in adapters.items():p.data.copy_(beststate[n].to(args.device))
    final=evaluate(splits['test'],'selected_test');after=frozen_digest(model)
    if after!=base_sha:raise RuntimeError('Frozen base changed')
    change=sum(float((p.detach().cpu()-before[n]).abs().sum()) for n,p in adapters.items())
    report={'args':vars(args),'task':'Generative VLM attribute SFT with catalogue weak labels','baseline':baseline,'logs':logs,'test':final,'trainable_parameters':sum(p.numel() for p in adapters.values()),'gradient_l1':gradient,'adapter_l1_change':change,'base_sha_before':base_sha,'base_sha_after':after,'seconds':time.perf_counter()-start,'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'data_sha256':{s:hashlib.sha256((ROOT/f'data/hm/vlm/{s}.jsonl').read_bytes()).hexdigest() for s in splits},'limitations':['Single seed','Nonrandom 590 items','Metadata agreement is not human visual accuracy','No representative recommendation gains','End-to-end generation evaluation; taxonomy synonyms counted as misses']}
    (ROOT/f'reports/vlm_sft_{args.seed}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
