"""Stronger fixed baselines under the same full catalogue protocol."""
import json
import sqlite3
import numpy as np
from recsys.hm_train import load, metric, DATA, ROOT

if __name__ == '__main__':
    ids, _, _, _, _, _, evaluation, config = load()
    k = config['evaluation_k']
    db = sqlite3.connect(DATA / 'transactions.sqlite')
    counts = dict(db.execute('SELECT item,COUNT(*) FROM events WHERE date >= ? AND date < ? GROUP BY item',
                            ('2020-09-02', config['validation_start'])))
    recent = sorted(ids, key=lambda x: (-counts.get(x, 0), x))[:k]
    report = {}
    for split, rows in evaluation.items():
        report[split] = {}
        for name in ['recent_popularity', 'repeat_recent_then_popularity']:
            results = []
            for history, targets in rows:
                predictions = recent if name == 'recent_popularity' else list(dict.fromkeys(
                    [ids[x - 1] for x in reversed(history) if x] + recent))[:k]
                results.append(metric(predictions, targets))
            report[split][name] = dict(zip([f'recall@{k}', f'ndcg@{k}', f'map@{k}'],
                np.mean(results, axis=0).tolist()))
    report['limitations'] = ['Popularity frozen before validation for both weeks.',
                            'Same-day history ties use deterministic item-ID order, not true temporal order.']
    (ROOT / 'reports/hm_stronger_baselines.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)); db.close()
