"""Download original dataset images by Git blob, verifying source Git SHA."""
import base64
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def download(entry):
    path = Path('data/vogue') / entry['path']
    def valid(data):
        return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == entry['sha']
    if path.exists() and valid(path.read_bytes()):
        return {'path': entry['path'], 'status': 'verified_cached'}
    for attempt in range(3):
        try:
            request = urllib.request.Request('https://raw.githubusercontent.com/D3Mlab/vogue/68edb8e1ac43f3718844e00a0f40f4f10173b24b/' + entry['path'], headers={'User-Agent': 'multimodal-recsys-research'})
            with urllib.request.urlopen(request, timeout=20) as response:
                data = response.read()
            if not valid(data):
                raise ValueError('Source hash mismatch')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return {'path': entry['path'], 'status': 'downloaded', 'bytes': len(data)}
        except Exception as exc:
            error = str(exc)
            time.sleep(attempt + 1)
    return {'path': entry['path'], 'status': 'failed', 'error': error}


if __name__ == '__main__':
    manifest = json.loads(Path('reports/vogue_manifest.json').read_text())
    entries = [x for x in manifest['files'] if x['path'].startswith('data/metadata/item_') and x['path'].endswith('.png')]
    results = list(ThreadPoolExecutor(6).map(download, entries))
    Path('reports/vogue_image_download.json').write_text(json.dumps(results, indent=2))
    failed = [r for r in results if r['status'] == 'failed']
    print(json.dumps({'images': len(results), 'failed': failed}, indent=2))
    if failed:
        raise SystemExit(1)
