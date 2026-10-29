"""Existing official images only; item-family/exact-image grouped pair split."""
import hashlib,json
from pathlib import Path
from recsys.hm_train import ROOT,DATA

if __name__=='__main__':
    rows=json.loads((DATA/'catalogue.json').read_text());rows=[r for r in rows if (ROOT/'data/hm/raw/images'/r['article_id'][:3]/(r['article_id']+'.jpg')).exists()]
    parent={r['article_id']:r['article_id'] for r in rows}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):parent[find(a)]=find(b)
    families={};hashes={};captions={}
    for row in rows:
        item=row['article_id'];family=row['product_code']
        sha=hashlib.sha256((ROOT/'data/hm/raw/images'/item[:3]/(item+'.jpg')).read_bytes()).hexdigest()
        caption='; '.join(str(row[k]) for k in ['prod_name','product_type_name','colour_group_name','detail_desc'] if row.get(k))
        for key,groups in [(family,families),(sha,hashes),(caption,captions)]:
            if key in groups:union(item,groups[key])
            else:groups[key]=item
    groups={}
    for row in rows:groups.setdefault(find(row['article_id']),[]).append(row['article_id'])
    split={}
    for members in groups.values():
        value=int(hashlib.sha256(('42:'+min(members)).encode()).hexdigest()[:8],16)/2**32
        name='train' if value<.7 else 'val' if value<.85 else 'test'
        for item in members:split[item]=name
    for r in rows:r['pair_split']=split[r['article_id']]
    out=ROOT/'data/hm/visual';out.mkdir(parents=True,exist_ok=True)
    (out/'catalogue.json').write_text(json.dumps(rows))
    queries=[]
    for line in (DATA/'queries.jsonl').open():
        q=json.loads(line);history=[x for x in q['history'] if x in split];targets=[x for x in q['targets'] if x in split]
        if history and targets:queries.append({**q,'history':history,'targets':targets})
    (out/'queries.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries))
    report={'items':len(rows),'pair_groups':len(groups),'pair_splits':{s:sum(r['pair_split']==s for r in rows) for s in ['train','val','test']},
            'behavior_queries':{s:sum(q['split']==s for q in queries) for s in ['train','val','test']},
            'selection':'Existing official image subset, not representative. Pair split groups product_code, exact image SHA and exact caption. Behavior keeps original temporal split.',
            'limitations':['Filtered evaluation only, not original full-catalogue results','CLIP pair task and purchase recommendation are distinct protocols'],
            'hashes':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['catalogue.json','queries.jsonl']}}
    (ROOT/'reports/visual_subset.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
