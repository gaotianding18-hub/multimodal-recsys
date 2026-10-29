"""Aggregate completed comparable text runs; choose only by validation."""
import json
import numpy as np
from recsys.hm_train import ROOT

if __name__=='__main__':
    paths=[ROOT/'reports/hm_mm_text_42.json',*sorted((ROOT/'reports').glob('hm_mm_text_4[34]_*.json'))]
    runs=[json.loads(p.read_text()) for p in paths]
    if len(runs)!=3 or len({r['feature_sha256'] for r in runs})!=1:raise RuntimeError('Need three aligned text runs')
    best=max(runs,key=lambda r:r['selection_val_ndcg'])
    values=[r['test']['ndcg@12'] for r in runs]
    report={'runs':runs,'test_ndcg_mean':float(np.mean(values)),'seed_population_std':float(np.std(values)),
            'selected_by_validation':{'seed':best['args']['seed'],'test':best['test']},
            'matched_baselines':json.loads((ROOT/'reports/hm_matched_baselines.json').read_text()),
            'limitations':['Seed spread is not statistical confidence','Exploratory test, not blind','Not a full multimodal experiment']}
    (ROOT/'reports/hm_text_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['runs','matched_baselines']}))
