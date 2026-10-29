import json
import platform
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

URL='http://127.0.0.1:8765'
def request(i):
    start=time.perf_counter()
    req=urllib.request.Request(URL+'/recommend',data=json.dumps({'user_id':f's{i%10+1}','scenario':i%6+1,'k':3}).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=10) as r:result=json.load(r)
        assert len(result['items'])==3
        return (time.perf_counter()-start)*1000,None
    except Exception as e:return (time.perf_counter()-start)*1000,str(e)

if __name__=='__main__':
    for i in range(20):request(i)
    runs=[]
    for concurrency in [1,4]:
        start=time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:results=list(pool.map(request,range(200)))
        elapsed=time.perf_counter()-start
        times=[x[0] for x in results if x[1] is None]
        runs.append({'concurrency':concurrency,'requests':200,'warmup_requests':20,'errors':sum(x[1] is not None for x in results),
                     'error_messages':sorted({x[1] for x in results if x[1]}),'throughput_requests_s':len(times)/elapsed,
                     'p50_ms':float(np.percentile(times,50)) if times else None,'p95_ms':float(np.percentile(times,95)) if times else None,
                     'p99_ms':float(np.percentile(times,99)) if times else None})
    Path('reports/service_benchmark.json').write_text(json.dumps({'platform':platform.platform(),'device':'cpu','catalogue_items':36,'scenario_candidates':12,'retrieved_candidates':6,'k':3,'method':'warm local HTTP POST, closed-loop thread clients, includes localhost roundtrip','runs':runs},indent=2))
    print(json.dumps(runs,indent=2))
