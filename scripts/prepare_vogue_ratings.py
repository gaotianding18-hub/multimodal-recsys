"""Convert observed survey cells to contextual records; no synthetic behavior."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/vogue/data'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/vogue_observed_ratings.jsonl'))
    args = parser.parse_args()
    source = args.data_root / 'conversation_trials/item_ratings/seeker_ratings.csv'
    records = []
    keys = set()
    with source.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            for item_id in range(1, 37):
                rating = float(row[f'item_{item_id}'])
                if rating == -1:
                    continue
                if rating not in {1, 2, 3, 4, 5}:
                    raise ValueError(f'Invalid rating: {rating}')
                key = (row['participant_id'], row['conversation_id'], item_id)
                if key in keys:
                    raise ValueError('Duplicate user/context/item key')
                keys.add(key)
                records.append({
                    'user_id': row['participant_id'],
                    'conversation_id': row['conversation_id'],
                    'scenario': row['scenario'], 'catalogue': row['catalogue'],
                    'item_id': item_id, 'rating': int(rating),
                    'label_source': 'human_contextual_survey',
                })
    if not records:
        raise ValueError('No observed ratings')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r) + '\n' for r in records))
    print(json.dumps({'observed_records': len(records), 'output': str(args.output),
                      'note': 'Not split for training; no invented timestamps or click labels.'}, indent=2))


if __name__ == '__main__':
    main()
