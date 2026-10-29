"""Stream the official CSVs; do not treat repeat purchases as corruption."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/hm/raw'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    with (RAW / 'articles.csv').open(newline='') as f:
        articles = list(csv.DictReader(f))
    article_ids = {x['article_id'] for x in articles}
    with (RAW / 'customers.csv').open(newline='') as f:
        customers = {x['customer_id'] for x in csv.DictReader(f)}
    dates, users, items = Counter(), Counter(), Counter()
    unknown_item = unknown_user = invalid = total = 0
    with (RAW / 'transactions_train.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        for row in reader:
            total += 1
            user, item, date = row['customer_id'], row['article_id'], row['t_dat']
            dates[date] += 1; users[user] += 1; items[item] += 1
            unknown_item += item not in article_ids
            unknown_user += user not in customers
            invalid += len(item) != 10 or not item.isdigit() or len(date) != 10
            if total % 5000000 == 0:
                print(json.dumps({'rows_scanned': total}), flush=True)
    result = {
        'transaction_rows': total, 'transaction_columns': columns,
        'articles': len(articles), 'unique_article_ids': len(article_ids),
        'customers': len(customers), 'active_customers': len(users),
        'transacted_articles': len(items), 'first_date': min(dates),
        'last_date': max(dates), 'unknown_item_rows': unknown_item,
        'unknown_user_rows': unknown_user, 'invalid_id_or_date_shape_rows': invalid,
        'missing_descriptions': sum(not x['detail_desc'].strip() for x in articles),
        'daily_transactions': dict(sorted(dates.items())),
        'sha256': {name: digest(RAW / name) for name in
                   ['articles.csv', 'customers.csv', 'transactions_train.csv']},
        'limitations': ['Purchases, not exposure/click labels.',
                         'Date granularity is daily; no within-day ordering inferred.',
                         'Duplicate transaction rows may represent repeat purchases.',
                         'Catalogue metadata is a static snapshot.',
                         'Images are not acquired or audited yet.']}
    (ROOT / 'reports/hm_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'daily_transactions'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
