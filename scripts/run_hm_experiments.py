"""Reproducible ID baseline / negative sampling ablation, three seeds."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for strategy in ['random', 'mixed']:
    for seed in [42, 43, 44]:
        report = ROOT / f'reports/hm_id_{strategy}_{seed}.json'
        directory = ROOT / f'artifacts/hm/id-{strategy}-{seed}'
        if report.exists() and (directory / 'model.pt').exists() and (directory / 'index.npy').exists():
            continue
        with (ROOT / f'reports/hm_id_{strategy}_{seed}.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'recsys.hm_train', '--epochs', '3',
                                     '--seed', str(seed), '--negative-strategy', strategy],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Failed {strategy}/{seed}; see retained log')
        print(strategy, seed, 'completed', flush=True)
runs = [json.loads((ROOT / f'reports/hm_id_{s}_{seed}.json').read_text())
        for s in ['random', 'mixed'] for seed in [42,43,44]]
selected = max(runs, key=lambda x: x['selection_val_ndcg'])
(ROOT / 'reports/hm_id_summary.json').write_text(json.dumps({'runs': runs,
    'selected_on_validation': selected['args'],
    'limitations': ['3 epochs fixed before this ablation; mixed uses more scoring computation.',
                    'Exploratory test results already inspected; not a blind final test.',
                    'ID models do not establish multimodal benefit.']}, indent=2))
