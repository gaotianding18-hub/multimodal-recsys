"""Audit observed survey ratings; never turn missing values into negatives."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def inspect_item(root, item_id):
    metadata_path = root / f'metadata/item_{item_id}.json'
    image_path = root / f'metadata/item_{item_id}.png'
    result = {'item_id': item_id, 'errors': []}
    if not metadata_path.is_file():
        result['errors'].append('metadata_missing')
    else:
        try:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('id') != item_id:
                result['errors'].append('metadata_id_mismatch')
            title = metadata.get('product_name')
            bullets = metadata.get('about_product') or []
            if not isinstance(title, str) or not title.strip():
                result['errors'].append('title_missing_or_invalid')
            if not isinstance(bullets, list) or any(not isinstance(x, str) for x in bullets):
                result['errors'].append('about_product_invalid')
            result['description_missing'] = metadata.get('product_description') is None
            result['text_rule'] = 'product_name + about_product; omit null description, ratings and review summaries'
        except (ValueError, TypeError, AttributeError) as exc:
            result['errors'].append(f'metadata_invalid: {type(exc).__name__}')
    if not image_path.is_file():
        result['errors'].append('image_missing')
    else:
        try:
            with Image.open(image_path) as image:
                result['image_size'] = list(image.size)
                image.verify()
            result['image_sha256'] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        except (OSError, ValueError, SyntaxError) as exc:
            result['errors'].append(f'image_invalid: {type(exc).__name__}')
    result['usable_under_current_rule'] = not result['errors']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/vogue/data'))
    parser.add_argument('--output', type=Path, default=Path('reports/vogue_data_audit.json'))
    args = parser.parse_args()
    source = args.data_root / 'conversation_trials/item_ratings/seeker_ratings.csv'
    with source.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Rating file contains no rows')
    columns = [f'item_{i}' for i in range(1, 37)]
    required = {'participant_id', 'conversation_id', *columns}
    if required - rows[0].keys():
        raise ValueError(f'Missing columns: {sorted(required - rows[0].keys())}')
    counts = Counter()
    rated_items = set()
    observed = 0
    for row in rows:
        for column in columns:
            value = float(row[column])
            if value not in {-1, 1, 2, 3, 4, 5}:
                raise ValueError(f'Unexpected rating: {value}')
            counts[str(int(value))] += 1
            if value != -1:
                observed += 1
                rated_items.add(int(column.removeprefix('item_')))
    joined = [i for i in sorted(rated_items)
              if (args.data_root / f'metadata/item_{i}.json').is_file()
              and (args.data_root / f'metadata/item_{i}.png').is_file()]
    item_checks = [inspect_item(args.data_root, i) for i in sorted(rated_items)]
    usable = [x['item_id'] for x in item_checks if x['usable_under_current_rule']]
    report = {
        'task': 'contextual_preference_survey_not_purchase_or_click_log',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows': len(rows),
        'unique_seeker_ids': len({row['participant_id'] for row in rows}),
        'observed_rating_cells': observed,
        'missing_rating_cells': counts['-1'],
        'rating_histogram': dict(sorted(counts.items())),
        'rated_item_count': len(rated_items),
        'local_items_with_metadata_and_image': joined,
        'image_text_join_complete_locally': len(joined) == len(rated_items),
        'content_validated_item_ids': usable,
        'all_rated_items_content_validated': len(usable) == len(rated_items),
        'item_content_checks': item_checks,
        'limitations': 'Ratings depend on conversation context; repeated user-item ratings are not independent. Missing means unobserved. File presence is not full content/license verification.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
