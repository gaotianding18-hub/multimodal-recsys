"""Run with .venv-kaggle/bin/python after user configures local credentials.

Downloads only selected training-catalogue images from official competition.
Never prints authentication secrets. Missing official images stay missing.
"""
import argparse
import concurrent.futures
import threading
import hashlib
import io
import shutil
import json
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = 'h-and-m-personalized-fashion-recommendations'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--request-interval', type=float, default=2.0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: parser.error('workers must be 1..8')
    from kaggle.api.kaggle_api_extended import KaggleApi
    local = threading.local()
    request_lock = threading.Lock(); record_lock = threading.Lock()
    last_request = [0.0]; stopped = threading.Event()
    catalogue = json.loads((ROOT / 'data/hm/processed/catalogue.json').read_text())
    root = ROOT / 'data/hm/raw/images'; cache = ROOT / 'data/hm/download_cache'
    cache.mkdir(parents=True, exist_ok=True)
    records_path = ROOT / 'data/hm/image_download.jsonl'
    # limit=0 means entire selected catalogue, explicit CLI choice.
    started = time.perf_counter()
    latest = {}
    if records_path.exists():
        for line in records_path.open():
            record = json.loads(line); latest[record['item_id']] = record
    rows = catalogue[args.offset:args.offset + args.limit] if args.limit else catalogue[args.offset:]
    def download(row):
        if not hasattr(local, "api"):
            local.api = KaggleApi(); local.api.authenticate()
        api = local.api
        item = row['article_id']; target = root / item[:3] / f'{item}.jpg'
        if target.exists(): return None
        if latest.get(item, {}).get('http_status') == 404: return None
        if shutil.disk_usage(root.parent).free < 20 * 1024**3:
            raise RuntimeError('Less than 20 GiB disk free; stopping download safely')
        record = {'item_id': item, 'source': 'official Kaggle competition'}
        for attempt in range(3):
            try:
                if stopped.is_set(): raise RuntimeError('Download batch stopped')
                with request_lock:
                    time.sleep(max(0, args.request_interval - (time.monotonic()-last_request[0])))
                    last_request[0] = time.monotonic()
                api.competition_download_file(COMPETITION, f'images/{item[:3]}/{item}.jpg',
                                              path=str(cache), quiet=True)
                plain = cache / f'{item}.jpg'; zipped = cache / f'{item}.jpg.zip'
                if plain.exists(): data = plain.read_bytes()
                elif zipped.exists():
                    with zipfile.ZipFile(zipped) as z:
                        candidates = [n for n in z.namelist() if Path(n).name == f'{item}.jpg']
                        if len(candidates) != 1: raise ValueError('Unexpected image archive')
                        data = z.read(candidates[0])
                else: raise FileNotFoundError('Client did not produce an expected file')
                # Official API can return ZIP content under a .jpg filename.
                if zipfile.is_zipfile(io.BytesIO(data)):
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        candidates = [n for n in z.namelist() if Path(n).name == f'{item}.jpg']
                        if len(candidates) != 1: raise ValueError('Unexpected image archive')
                        data = z.read(candidates[0])
                if not data.startswith(b'\xff\xd8'): raise ValueError('Not JPEG')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.jpg.part'); temporary.write_bytes(data); temporary.replace(target)
                record.update(status='downloaded', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                plain.unlink(missing_ok=True); zipped.unlink(missing_ok=True)
                return record
            except Exception as error:
                # Exception URLs may contain signed query strings. Record only
                # the class and HTTP status, never the raw authentication URL.
                status = getattr(error, 'status', getattr(getattr(error, 'response', None), 'status_code', None))
                record.update(status='failed', error_type=type(error).__name__, http_status=status)
                if status == 429:
                    stopped.set()
                    record['retry_after'] = getattr(getattr(error, 'response', None), 'headers', {}).get('Retry-After')
                    with record_lock:
                        with records_path.open('a') as f: f.write(json.dumps(record) + '\n')
                    raise RuntimeError('Official API rate limited (429); stopped without further batch retries') from None
                if status in (401,403):
                    stopped.set()
                    raise RuntimeError('Kaggle access rejected; verify local credentials and accepted competition rules') from None
                if status == 404: break
                time.sleep(1 + attempt)
        return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for offset in range(0,len(rows),args.workers):
            chunk=rows[offset:offset+args.workers]
            for i, record in enumerate(executor.map(download,chunk),start=offset):
                if record is not None:
                    with records_path.open('a') as f: f.write(json.dumps(record) + '\n')
                if (i + 1) % 100 == 0: print('processed images', i + 1, '/', len(rows), flush=True)
    present = [root / r['article_id'][:3] / (r['article_id'] + '.jpg') for r in catalogue]
    report = {'catalogue_items':len(catalogue),'present_images':sum(p.exists() for p in present),
              'bytes':sum(p.stat().st_size for p in present if p.exists()),
              'seconds':time.perf_counter()-started,'args':vars(args),'status':'download_pass_finished'}
    (ROOT/'reports/hm_images_download.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__': main()
