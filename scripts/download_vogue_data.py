"""Fetch all required source files, with Git blob hash verification."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from download_vogue_images import download

if __name__=='__main__':
    manifest=json.loads(Path('reports/vogue_manifest.json').read_text())
    required={'LICENSE.txt','data/fashion_profiles/ratings.csv','data/conversation_trials/scenarios.json',
              'data/conversation_trials/item_ratings/seeker_ratings.csv'}
    entries=[x for x in manifest['files'] if x['path'] in required or
             (x['path'].startswith('data/metadata/item_') and (x['path'].endswith('.png') or x['path'].endswith('.json')))]
    results=list(ThreadPoolExecutor(6).map(download,entries))
    Path('reports/vogue_full_download.json').write_text(json.dumps(results,indent=2))
    failed=[r for r in results if r['status']=='failed']
    print(json.dumps({'files':len(results),'failed':failed},indent=2))
    if failed:raise SystemExit(1)
