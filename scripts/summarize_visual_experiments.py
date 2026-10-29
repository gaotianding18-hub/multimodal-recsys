"""Conditional subset baselines and completed multimodal-head experiments."""
import json
from collections import Counter
import numpy as np
from recsys.hm_train import ROOT,metric

if __name__=='__main__':
    rows=json.loads((ROOT/'data/hm/visual/catalogue.json').read_text());ids=[r['article_id'] for r in rows]
    queries=[json.loads(x) for x in (ROOT/'data/hm/visual/queries.jsonl').read_text().splitlines()]
    counts=Counter(x for q in queries if q['split']=='train' for x in q['targets']);popular=sorted(ids,key=lambda x:(-counts[x],x))
    baselines={}
    for split in ['val','test']:
        baselines[split]={}
        selected=[q for q in queries if q['split']==split]
        for name in ['popularity','repeat_then_popularity']:
            values=[]
            for q in selected:
                predictions=popular[:12] if name=='popularity' else list(dict.fromkeys(list(reversed(q['history'][-10:]))+popular))[:12]
                values.append(metric(predictions,set(q['targets'])))
            baselines[split][name]=dict(zip(['recall@12','ndcg@12','map@12'],np.mean(values,axis=0).tolist()))
    runs=[json.loads(p.read_text()) for p in sorted((ROOT/'reports').glob('visual_*_*_4[234].json'))]
    aggregation={}
    for source in ['frozen','lora']:
        for mode in ['id','text','image','mean','gate']:
            group=[r for r in runs if r['args']['source']==source and r['args']['mode']==mode]
            if group:
                scores=[r['test']['ndcg@12'] for r in group]
                aggregation[source+'_'+mode]={'seeds':len(group),'ndcg_mean':float(np.mean(scores)),'seed_population_std':float(np.std(scores))}
    multimodal=[r for r in runs if r['args']['mode'] in ['mean','gate']]
    best=max(multimodal,key=lambda r:r['selection_val_ndcg'])
    result={'baselines':baselines,'aggregation':aggregation,'multimodal_selected_on_validation':best['args'],
            'selected_test':best['test'],'limitations':['Conditional subset results not comparable to full H&M metrics',
            'LoRA transfer uses fixed adapter seed42; the three seeds here are retrieval head seeds','ID baseline uses separate history architecture','Selection by validation, test already exploratory']}
    (ROOT/'reports/visual_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
