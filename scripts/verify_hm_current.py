"""Actual base-model offline/serving top-k parity before semantic release."""
import json
import torch
from recsys.hm_train import ROOT, load
from recsys.hm_serve import Engine
from recsys.hm_rank import features

if __name__=='__main__':
    engine=Engine();*_,evaluation,config=load();matches={'repeat':0,'id':0}
    with torch.inference_mode():
        for h,_ in evaluation['test'][:30]:
            history=[engine.ids[i-1] for i in h if i]
            repeat=list(dict.fromkeys([i for i in reversed(h) if i]+engine.recent))[:12]
            hist=torch.tensor([h]);scores=engine.model.users(hist)@engine.vectors[1:].T
            pool=scores.topk(engine.pool,dim=-1).indices.flatten()+1
            expanded=hist.expand(engine.pool,-1)
            rank=engine.ranker(engine.vectors[expanded],engine.vectors[pool],expanded.ne(0),features(expanded,pool,engine.popularity))
            order=pool[rank.topk(12).indices].tolist()
            for method,indices in [('repeat',repeat),('id',order)]:
                result=engine.recommend({'history':history,'method':method})
                expected=[engine.ids[i-1] for i in indices]
                if expected!=[x['item_id'] for x in result['items']]:raise RuntimeError(f'{method} parity failure')
                matches[method]+=1
        for method in engine.methods():
            assert engine.recommend({'history':[],'method':method})['fallback']
            assert engine.recommend({'history':['UNKNOWN'],'method':method})['unknown_history_items']==1
    report={'version':engine.version,'matched_histories_per_method':matches,'fallback_checked':True,
            'scope':'Base ID+rank and repeat. Not yet a full multimodal release.'}
    (ROOT/'reports/hm_current_consistency.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
