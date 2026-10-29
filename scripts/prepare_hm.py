"""Stream transactions to SQLite, select users on training data only.

No artificial interactions. Daily baskets use history strictly before that day.
All training-observed products form the candidate catalogue.
"""
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_queries(rows, history_length):
    history, queries = [], []
    baskets = defaultdict(list)
    for date, item in rows:
        baskets[date].append(item)
    for date in sorted(baskets):
        targets = sorted(set(baskets[date]))
        if history:
            queries.append({'date': date, 'history': history[-history_length:], 'targets': targets})
        # Stable ID order within a day is bookkeeping, not temporal evidence.
        history.extend(targets)
    return queries, history


def main():
    config = json.loads((ROOT / 'configs/hm.json').read_text())
    out = ROOT / 'data/hm/processed'; out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(out / 'transactions.sqlite')
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
    source = ROOT / 'data/hm/raw/transactions_train.csv'
    cache_key = json.dumps({'start': config['train_start'], 'end': config['end_exclusive'],
                            'source_size': source.stat().st_size, 'source_mtime_ns': source.stat().st_mtime_ns}, sort_keys=True)
    cached = db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()
    if cached is None or cached[0] != cache_key:
        # Derived data only. A crashed partial import must never be reused.
        db.execute('DROP TABLE IF EXISTS events')
        db.execute("DELETE FROM metadata WHERE key='complete'")
        db.commit()
    db.execute('CREATE TABLE IF NOT EXISTS events (date TEXT, user TEXT, item TEXT)')
    if not db.execute('SELECT COUNT(*) FROM events').fetchone()[0]:
        batch = []; count = 0
        with (ROOT / 'data/hm/raw/transactions_train.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                if config['train_start'] <= row['t_dat'] < config['end_exclusive']:
                    batch.append((row['t_dat'], row['customer_id'], row['article_id']))
                if len(batch) >= 50000:
                    db.executemany('INSERT INTO events VALUES (?,?,?)', batch)
                    count += len(batch); batch.clear(); db.commit()
        db.executemany('INSERT INTO events VALUES (?,?,?)', batch); db.commit()
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('complete',?)", (cache_key,)); db.commit()
        print('recent events imported', count + len(batch), flush=True)
    db.execute('CREATE INDEX IF NOT EXISTS by_user_date ON events(user,date)'); db.commit()
    val_start = config['validation_start']
    eligible = [r[0] for r in db.execute('SELECT user FROM events WHERE date < ? GROUP BY user HAVING COUNT(*) >= ?',
                                        (val_start, config['min_train_purchases']))]
    users = sorted(eligible, key=lambda u: hashlib.sha256(f"{config['selection_seed']}:{u}".encode()).digest())[:config['max_users']]
    counts = dict(db.execute('SELECT item, COUNT(*) FROM events WHERE date < ? GROUP BY item', (val_start,)))
    catalogue = sorted(counts); known = set(catalogue)
    with (ROOT / 'data/hm/raw/articles.csv').open(newline='') as f:
        articles = [r for r in csv.DictReader(f) if r['article_id'] in known]
    (out / 'catalogue.json').write_text(json.dumps(articles))
    (out / 'popularity.json').write_text(json.dumps(counts))
    stats = Counter(); target_coverage = Counter()
    with (out / 'queries.jsonl').open('w') as f:
        for i, user in enumerate(users):
            rows = list(db.execute('SELECT date,item FROM events WHERE user=? ORDER BY date,item', (user,)))
            train_rows = [(d, item) for d, item in rows if d < val_start]
            train_queries, history = build_queries(train_rows, config['history_length'])
            for q in train_queries:
                f.write(json.dumps(dict(q, user_id=user, split='train')) + '\n'); stats['train_queries'] += 1
            for split, start, end in [('val', val_start, config['test_start']),
                                       ('test', config['test_start'], config['end_exclusive'])]:
                # Frozen week-start history; no in-window purchases feed inputs.
                _, hist = build_queries([(d, x) for d, x in rows if d < start], config['history_length'])
                targets = sorted({x for d, x in rows if start <= d < end})
                if not targets: continue
                q = {'date': start, 'history': hist[-config['history_length']:], 'targets': targets,
                     'user_id': user, 'split': split}
                f.write(json.dumps(q) + '\n'); stats[split + '_queries'] += 1
                target_coverage[split + '_all'] += len(targets)
                target_coverage[split + '_in_catalogue'] += sum(x in known for x in targets)
            stats['selected_transactions'] += len(rows)
            if (i + 1) % 25000 == 0: print('users processed', i + 1, flush=True)
    report = {'config': config, 'source_cache_key': cache_key,
              'source_hashes': json.loads((ROOT / 'reports/hm_audit.json').read_text())['sha256'],
              'processed_hashes': {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                   for name in ['queries.jsonl', 'catalogue.json', 'popularity.json']},
              'eligible_train_users': len(eligible), 'selected_users': len(users),
              'candidate_items': len(catalogue), 'stats': dict(stats), 'target_coverage': dict(target_coverage),
              'protocol': 'train-active hashed user subset; all recent train-observed catalogue; repeat purchases allowed; out-of-catalogue targets retained in metric denominator'}
    (ROOT / 'reports/hm_preparation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True); db.close()


if __name__ == '__main__': main()
