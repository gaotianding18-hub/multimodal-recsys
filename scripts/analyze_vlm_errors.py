"""Catalogue disagreement diagnosis without claiming visual ground truth."""
import json
from collections import Counter
from recsys.hm_train import ROOT
from recsys.vlm_metrics import attribute_metrics


def main():
    train=[json.loads(x) for x in (ROOT/'data/hm/vlm/train.jsonl').read_text().splitlines()];known={k:{json.loads(x['answer'])[k] for x in train} for k in ['product_type','color']}
    rows=json.loads((ROOT/'artifacts/hm/vlm-sft-42/selected_test.json').read_text());counts=Counter();examples=[]
    for row in rows:
        try:value=json.loads(row['output'])
        except json.JSONDecodeError:value=None
        valid=isinstance(value,dict) and set(value)=={'product_type','color'} and all(isinstance(v,str) for v in value.values())
        reasons=[]
        if not valid:reasons.append('invalid_schema')
        else:
            for key in known:
                if value[key].strip().casefold()!=row['target'][key].strip().casefold():reasons.append(key+'_metadata_disagreement')
        for key in known:
            if row['target'][key] not in known[key]:counts['unseen_target_'+key]+=1
        counts.update(reasons)
        if reasons and len(examples)<10:examples.append({'article_id':row['article_id'],'reasons':reasons,'output':row['output'],'metadata':row['target']})
    subsets={}
    for key in known:
        for label in ['seen','unseen']:
            selected=[r for r in rows if (r['target'][key] in known[key])==(label=='seen')]
            subsets[key+'_'+label]=attribute_metrics([r['output'] for r in selected],[r['target'] for r in selected]) if selected else {'items':0}
    report={'counts':dict(counts),'subsets':subsets,'examples':examples,'limitations':['Examples are catalogue disagreements, not verified hallucinations','Fine-grained synonyms can count as errors','No manual visual audit performed']}
    # Item-linked outputs stay local artifacts, not redistributed with source.
    (ROOT/'artifacts/hm/vlm-sft-42/error_analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps({'counts':report['counts'],'subsets':subsets}))

if __name__=='__main__':main()
