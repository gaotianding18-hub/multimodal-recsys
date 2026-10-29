"""Compare offline and serving computation on actual held-out input histories."""
import argparse
import json
import time
import resource
import numpy as np
import torch
from recsys.hm_train import ROOT, load
from recsys.hm_serve import Engine
from recsys.hm_hybrid import ranked


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='artifacts/hm/release.json');args=p.parse_args()
    engine=Engine(ROOT/args.manifest)
    _,_,_,_,_,_,evaluation,_=load();rows=evaluation['test'][:30]
    matches=0
    for h,targets in rows:
        history=[engine.ids[i-1] for i in h if i]
        online=engine.recommend({'history':history,'method':'semantic','k':12})
        model,vectors,ranker,recent,_=engine.semantic
        offline,_,_=ranked(model,ranker,vectors,torch.tensor([h]),recent)
        expected=[engine.ids[i-1] for i in offline[0].tolist()]
        if expected!=[x['item_id'] for x in online['items']]:raise RuntimeError('Offline/serving top12 mismatch')
        matches+=1
    for method in engine.methods():
        assert engine.recommend({'history':[],'method':method})['fallback']
    assert engine.recommend({'history':['UNKNOWN'],'method':'semantic'})['unknown_history_items']==1
    for _ in range(5):engine.recommend({'user_id':'demo-1','method':'semantic'})
    latencies=[]
    for i in range(100):
        start=time.perf_counter();engine.recommend({'user_id':f'demo-{i%30+1}','method':'semantic'})
        latencies.append((time.perf_counter()-start)*1000)
    report={'version':engine.version,'matching_test_histories':matches,'fallback_and_unknown_checks':True,
            'benchmark':'engine calls, not HTTP; one thread; 5 warmups then 100 calls across 30 users',
            'engine_p50_ms':float(np.percentile(latencies,50)),'engine_p95_ms':float(np.percentile(latencies,95)),
            'process_peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'candidate_items':len(engine.ids),'k':12,'manifest':args.manifest}
    (ROOT/'reports/hm_release_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
