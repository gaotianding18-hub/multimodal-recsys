"""Answer-only generative VLM LoRA pilot; local weights and data only."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Idefics3ForConditionalGeneration
from recsys.clip_lora import LoRALinear
from recsys.hm_train import ROOT


def frozen_digest(model):
    digest=hashlib.sha256()
    for name,value in model.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=4);p.add_argument('--device',default='mps');args=p.parse_args()
    torch.manual_seed(42);torch.set_num_threads(4);started=time.perf_counter()
    directory=ROOT/'artifacts/backbones/smolvlm-500m';processor=AutoProcessor.from_pretrained(directory,local_files_only=True)
    processor.image_processor.do_image_splitting=False
    processor.image_processor.size={'longest_edge':512}
    model=Idefics3ForConditionalGeneration.from_pretrained(directory,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
    model.requires_grad_(False);names=[]
    for name,module in list(model.named_modules()):
        if isinstance(module,torch.nn.Linear) and name.endswith(('q_proj','v_proj')) and 'text_model' in name:
            parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,LoRALinear(module,4,8));names.append(name)
    if not names:raise RuntimeError('No language attention adapters found')
    model.to(args.device);adapters={n:x for n,x in model.named_parameters() if x.requires_grad};before={n:x.detach().cpu().clone() for n,x in adapters.items()}
    base_before=frozen_digest(model)
    optimizer=torch.optim.AdamW(adapters.values(),lr=1e-4)
    rows=[json.loads(x) for x in (ROOT/'data/hm/vlm/train.jsonl').read_text().splitlines()];logs=[]
    for step in range(args.steps):
        row=rows[step%len(rows)];image=Image.open(ROOT/row['image']).convert('RGB')
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['prompt']}]}]
        prompt=processor.apply_chat_template(messages,add_generation_prompt=True)
        prefix=processor(text=prompt,images=[image],return_tensors='pt')
        batch=processor(text=prompt+row['answer']+processor.tokenizer.eos_token,images=[image],return_tensors='pt')
        count=prefix['input_ids'].shape[1]
        if not torch.equal(prefix['input_ids'][0],batch['input_ids'][0,:count]):raise RuntimeError('Prompt boundary is not token-prefix aligned')
        labels=batch['input_ids'].clone();labels[:,:count]=-100;labels[labels==model.config.image_token_id]=-100
        batch={k:v.to(args.device) for k,v in batch.items()};labels=labels.to(args.device)
        model.train();optimizer.zero_grad(set_to_none=True);loss=model(**batch,labels=labels).loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();grad=sum(float(x.grad.detach().abs().sum().cpu()) for x in adapters.values() if x.grad is not None);optimizer.step()
        record={'step':step+1,'loss':float(loss.detach().cpu()),'gradient_l1':grad,'tokens':int(labels.numel()),'answer_tokens':int((labels!=-100).sum()),'seconds':time.perf_counter()-started};logs.append(record);print(json.dumps(record),flush=True)
    changes=sum(float((x.detach().cpu()-before[n]).abs().sum()) for n,x in adapters.items())
    if changes<=0:raise RuntimeError('Adapters did not change')
    base_after=frozen_digest(model)
    if base_before!=base_after:raise RuntimeError('Frozen base changed')
    output=ROOT/'artifacts/hm/generative-vlm-pilot';output.mkdir(parents=True,exist_ok=True)
    torch.save({'adapters':{n:x.detach().cpu() for n,x in adapters.items()},'modules':names,'rank':4,'alpha':8},output/'adapter.pt')
    model.eval()
    with torch.inference_mode():
        generated=model.generate(**{k:v.to(args.device) for k,v in prefix.items()},max_new_tokens=60,do_sample=False)
    answer=processor.batch_decode(generated[:,count:],skip_special_tokens=True)[0]
    report={'args':vars(args),'task':'Feasibility pilot only, not completed validation experiment','trainable_parameters':sum(x.numel() for x in adapters.values()),'modules':names,'logs':logs,'adapter_l1_change':changes,'base_sha_before':base_before,'base_sha_after':base_after,'training_example_generation':answer,'seconds':time.perf_counter()-started,'mps_allocated_end_bytes':torch.mps.current_allocated_memory(),'mps_driver_end_bytes':torch.mps.driver_allocated_memory(),'limitations':['Training-example generation is not held-out evaluation','No quality gain claimed','Float32 batch1 pilot','End memory is not peak']}
    (ROOT/'reports/generative_vlm_pilot.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
