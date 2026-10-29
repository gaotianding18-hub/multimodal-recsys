"""Measured warm loopback HTTP workload; no production extrapolation."""
import argparse
import concurrent.futures
import json
import platform
import time
import urllib.request
import numpy as np
from recsys.hm_train import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8766);p.add_argument('--requests',type=int,default=100);args=p.parse_args()
    base=f'http://127.0.0.1:{args.port}';health=json.load(urllib.request.urlopen(base+'/health',timeout=10));results=[]
    def request(method,i):
        start=time.perf_counter()
        user=f'visual-demo-{i%10+1}' if method=='visual' else f'demo-{i%30+1}'
        body=json.dumps({'user_id':user,'method':method,'k':12}).encode()
        req=urllib.request.Request(base+'/recommend',data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=10) as response:value=json.load(response)
        if value['version']!=health['version'] or len(value['items'])!=12:raise RuntimeError('Version/result mismatch')
        return (time.perf_counter()-start)*1000
    for method in health['methods']:
        for _ in range(5):request(method,0)
        for concurrency in [1,4]:
            started=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                latencies=list(executor.map(lambda i:request(method,i),range(args.requests)))
            elapsed=time.perf_counter()-started
            results.append({'method':method,'concurrency':concurrency,'requests':args.requests,'errors':0,
                            'p50_ms':float(np.percentile(latencies,50)),'p95_ms':float(np.percentile(latencies,95)),
                            'requests_per_second':args.requests/elapsed})
    report={'health':health,'platform':platform.platform(),'hardware':'MacBook Air M5 24GB, shared host',
            'workload':'localhost HTTP; 30 general histories, 10 visual-subset histories for visual method; k12; 5warmups per method; CPU torch threads1',
            'results':results,'limitations':['Short warm synthetic request workload, not real traffic or SLA',
                                             'Server RSS/peak memory not measured by this client benchmark']}
    path=ROOT/f"reports/hm_http_{health['version']}.json";path.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
