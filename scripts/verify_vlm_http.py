"""Reloaded HTTP inference vs saved held-out predictions; no quality claim."""
import json,urllib.request,urllib.error,time
from recsys.hm_train import ROOT


def main():
    base='http://127.0.0.1:8767';health=json.load(urllib.request.urlopen(base+'/health',timeout=10));rows=json.loads((ROOT/'artifacts/hm/vlm-sft-42/selected_test.json').read_text());times=[]
    def request(value):
        req=urllib.request.Request(base+'/attributes',data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(req,timeout=60))
    for row in [rows[0],rows[len(rows)//2],rows[-1]]:
        value=request({'article_id':row['article_id']})
        if value['output']!=row['output'] or value['version']!=health['version']:raise RuntimeError('Offline/service mismatch')
        times.append(value['elapsed_ms'])
    for body in [{'article_id':'unknown'},{'article_id':123}]:
        try:request(body)
        except urllib.error.HTTPError as error:
            if error.code!=400:raise
        else:raise RuntimeError('Invalid request accepted')
    result={'version':health['version'],'checked_test_items':3,'offline_http_generation_equal':True,'invalid_requests_return_400':True,'elapsed_ms':times,'limitations':['Three deterministic samples, not full service equivalence or load benchmark']}
    (ROOT/'reports/vlm_http_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
