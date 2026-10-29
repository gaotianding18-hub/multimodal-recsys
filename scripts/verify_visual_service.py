"""Actual conditional multimodal model offline/serving comparison."""
import json
import torch
from recsys.hm_train import ROOT
from recsys.hm_serve import Engine

if __name__=='__main__':
    engine=Engine();model,ids,mapping,vectors,_,selection=engine.visual;matched=0
    with torch.inference_mode():
        for line in (ROOT/'data/hm/visual/queries.jsonl').open():
            q=json.loads(line)
            if q['split']!='test':continue
            h=[mapping[x] for x in q['history']][-10:]
            top=(model.users(torch.tensor([[0]*(10-len(h))+h]))@vectors[1:].T).topk(12,dim=-1).indices[0]
            expected=[ids[i] for i in top.tolist()]
            result=engine.recommend({'history':q['history'],'method':'visual'})
            if expected!=[x['item_id'] for x in result['items']]:raise RuntimeError('Visual offline/serving mismatch')
            if result['candidate_items']!=590:raise RuntimeError('Incorrect scope')
            matched+=1
    assert engine.recommend({'history':[],'method':'visual'})['fallback']
    assert engine.recommend({'history':['OUTSIDE'],'method':'visual'})['unknown_history_items']==1
    report={'version':engine.version,'matched_test_queries':matched,'candidate_items':590,'selection':selection,'fallback_checked':True}
    (ROOT/'reports/visual_service_consistency.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
