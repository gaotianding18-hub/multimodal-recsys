"""Same target-query population baselines; no additional users or future counts."""
import json
from collections import Counter
import numpy as np
from recsys.hm_train import ROOT, DATA, load, metric


def main():
    ids,_,_,_,_,_,evaluation,config=load();total=Counter();recent=Counter()
    for line in (DATA/'queries.jsonl').open():
        q=json.loads(line)
        if q['split']!='train':continue
        total.update(q['targets'])
        if '2020-09-02'<=q['date']<config['validation_start']:recent.update(q['targets'])
    global_order=sorted(ids,key=lambda x:(-total[x],x));recent_order=sorted(ids,key=lambda x:(-recent[x],x))
    report={'training_information':'Counts only targets in the same 212658 selected-user training queries; no extra user population',
            'evaluation':'Full catalogue; week-start histories; OOC targets counted as misses','results':{}}
    for split,rows in evaluation.items():
        report['results'][split]={}
        for method in ['query_popularity','query_recent_popularity','repeat_then_query_recent']:
            totals=np.zeros(3)
            for history,targets in rows:
                if method=='query_popularity':predictions=global_order[:12]
                elif method=='query_recent_popularity':predictions=recent_order[:12]
                else:predictions=list(dict.fromkeys([ids[i-1] for i in reversed(history) if i]+recent_order[:12]))[:12]
                totals+=metric(predictions,targets)
            report['results'][split][method]=dict(zip(['recall@12','ndcg@12','map@12'],(totals/len(rows)).tolist()))
    report['limitations']=['Query counts are distinct-day basket occurrences, not unit sales counts.',
                           'Initial history-only baskets are not added to popularity counts.',
                           'Same-day history tie ordering remains deterministic item-ID order.']
    (ROOT/'reports/hm_matched_baselines.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
