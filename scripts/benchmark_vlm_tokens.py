"""Validation-only post-connector token retention; no novelty or speed promise."""
import json,time
import numpy as np
import torch
from recsys.hm_train import ROOT
from recsys.vlm_serve import AttributeEngine
from recsys.vlm_metrics import attribute_metrics


def main():
    engine=AttributeEngine();model=engine.model;processor=engine.processor
    rows=[json.loads(x) for x in (ROOT/'data/hm/vlm/val.jsonl').read_text().splitlines()]
    from PIL import Image
    results={}
    # Same grouped validation set, no choosing a pruning method on held-out test.
    for mode,keep in [('full',64),('uniform32',32),('norm32',32),('uniform16',16)]:
        predictions=[];durations=[];records=[]
        with torch.inference_mode():
            for i,row in enumerate(rows):
                prompt=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':row['prompt']}]}],add_generation_prompt=True)
                batch=processor(text=prompt,images=[Image.open(ROOT/row['image']).convert('RGB')],return_tensors='pt');batch={k:v.to(engine.device) for k,v in batch.items()}
                torch.mps.synchronize();started=time.perf_counter()
                features=model.model.get_image_features(batch.pop('pixel_values'),batch.pop('pixel_attention_mask'))
                positions=(batch['input_ids'][0]==model.config.image_token_id).nonzero().flatten();total=len(positions)
                if total!=64:raise RuntimeError('Expected one image with 64 connector tokens')
                selected=torch.arange(total,device=engine.device)
                if keep<total:
                    selected=features.norm(dim=-1).topk(keep).indices.sort().values if mode.startswith('norm') else torch.linspace(0,total-1,keep,device=engine.device).round().long()
                mask=torch.ones(batch['input_ids'].shape[1],dtype=torch.bool,device=engine.device);mask[positions]=False;mask[positions[selected]]=True
                batch={k:v[:,mask] for k,v in batch.items()};features=features[selected];count=batch['input_ids'].shape[1]
                generated=model.generate(**batch,image_hidden_states=features,max_new_tokens=60,do_sample=False)
                torch.mps.synchronize();durations.append((time.perf_counter()-started)*1000)
                output=processor.batch_decode(generated[:,count:],skip_special_tokens=True)[0];predictions.append(output);records.append({'article_id':row['article_id'],'output':output,'elapsed_ms':durations[-1]})
                if (i+1)%25==0:print(mode,i+1,flush=True)
        results[mode]={'metrics':attribute_metrics(predictions,[json.loads(r['answer']) for r in rows]),'visual_tokens':keep,'p50_ms':float(np.median(durations)),'p95_ms':float(np.percentile(durations,95)),'mps_allocated_end_bytes':torch.mps.current_allocated_memory(),'mps_driver_end_bytes':torch.mps.driver_allocated_memory()}
        (ROOT/f'artifacts/hm/vlm-sft-42/token_{mode}.json').write_text(json.dumps(records,indent=2));print(mode,json.dumps(results[mode]),flush=True)
    report={'model_version':engine.version,'results':results,'protocol':'93 grouped validation items, fixed order, batch1 float32 MPS, greedy max60, includes vision encoder and generation but excludes image preprocessing','hardware':'MacBook Air M5 24GB shared host','limitations':['Single pass, no random order or repeats','Norm/uniform retention are existing simple baselines, not novel algorithms','No test-set selection','End memory not peak; token output length affects latency','Vision encoder cost unchanged; only language-side visual tokens reduced']}
    (ROOT/'reports/vlm_token_benchmark.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
