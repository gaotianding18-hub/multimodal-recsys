"""Official fixed CLIP weight download with size/SHA and range-resume checks."""
import hashlib
import json
import re
import urllib.request
from recsys.hm_train import ROOT
from recsys.clip_features import WEIGHT_SHA

URL=f'https://openaipublic.azureedge.net/clip/models/{WEIGHT_SHA}/ViT-B-32.pt'
EXPECTED_BYTES=353976522


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    directory=ROOT/'artifacts/backbones';directory.mkdir(parents=True,exist_ok=True)
    target=directory/'ViT-B-32.pt';part=target.with_suffix('.pt.part')
    if target.exists():
        if target.stat().st_size!=EXPECTED_BYTES or digest(target)!=WEIGHT_SHA:raise RuntimeError('Existing weight invalid; preserve and inspect before replacing')
        print('Existing official weight verified');return
    offset=part.stat().st_size if part.exists() else 0
    req=urllib.request.Request(URL,headers={'Range':f'bytes={offset}-'} if offset else {})
    if offset<EXPECTED_BYTES:
        with urllib.request.urlopen(req,timeout=90) as response:
            if response.status==206:
                match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                if not match or int(match[1])!=offset or int(match[3])!=EXPECTED_BYTES:raise RuntimeError('Invalid ranged response')
                mode='ab'
            elif response.status==200:mode='wb';offset=0
            else:raise RuntimeError('Unexpected response status')
            with part.open(mode) as destination:
                for block in iter(lambda:response.read(4*1024*1024),b''):destination.write(block)
    if part.stat().st_size!=EXPECTED_BYTES or digest(part)!=WEIGHT_SHA:raise RuntimeError('Incomplete/corrupt weight; never publish it')
    part.replace(target)
    (ROOT/'reports/clip_weight_download.json').write_text(json.dumps({'source':URL,'bytes':EXPECTED_BYTES,'sha256':WEIGHT_SHA},indent=2))
    print('Official weight size and SHA verified')


if __name__=='__main__':main()
