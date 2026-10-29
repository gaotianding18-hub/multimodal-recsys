"""Aggregate completed runs without inventing missing experiments."""
import json
import numpy as np
from recsys.hm_train import ROOT


def main():
    runs=[json.loads((ROOT/f'reports/clip_lora_{seed}.json').read_text()) for seed in [42,43,44]]
    if any(r['baseline']!=runs[0]['baseline'] or r['dataset_sha256']!=runs[0]['dataset_sha256'] for r in runs):
        raise ValueError('Cannot aggregate different datasets or baselines')
    result={'runs':[r['args']['seed'] for r in runs],'baseline':runs[0]['baseline']['test'],
            'test_mean':{k:float(np.mean([r['test'][k] for r in runs])) for k in runs[0]['test']},
            'test_seed_std':{k:float(np.std([r['test'][k] for r in runs])) for k in runs[0]['test']},
            'base_unchanged_all':all(r['base_sha_before']==r['base_sha_after'] for r in runs),
            'all_towers_received_gradient':all(all(v>0 for v in r['gradient_l1_evidence'].values()) for r in runs),
            'limitations':['Small grouped 84-item held-out catalogue','Three seeds do not establish statistical significance','Image-text matching is distinct from purchase recommendation']}
    (ROOT/'reports/clip_lora_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
