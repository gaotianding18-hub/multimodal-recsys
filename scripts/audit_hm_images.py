"""Decode, hash and identify exact/identical-dHash groups; never delete items."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from recsys.hm_train import ROOT, DATA


def main():
    rows=json.loads((DATA/'catalogue.json').read_text());exact=defaultdict(list);perceptual=defaultdict(list)
    errors=[];present=0;size=0;start=time.perf_counter()
    target=ROOT/'data/hm/image_audit.jsonl'
    with target.open('w') as records:
        for row in rows:
            item=row['article_id'];path=ROOT/'data/hm/raw/images'/item[:3]/f'{item}.jpg'
            if not path.exists():continue
            try:
                with Image.open(path) as im:
                    im.load();pixels=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS))
                    dhash=np.packbits(pixels[:,1:]>pixels[:,:-1]).tobytes().hex();dimensions=im.size
                sha=hashlib.sha256(path.read_bytes()).hexdigest();exact[sha].append(item);perceptual[dhash].append(item)
                present+=1;size+=path.stat().st_size
                records.write(json.dumps({'item_id':item,'sha256':sha,'dhash':dhash,'size':dimensions})+'\n')
            except Exception as error:errors.append({'item_id':item,'error_type':type(error).__name__})
    report={'catalogue':len(rows),'valid_images':present,'coverage':present/len(rows),'bytes':size,
            'seconds':time.perf_counter()-start,'errors':errors,
            'exact_duplicate_groups':[group for group in exact.values() if len(group)>1],
            'identical_dhash_groups':[group for group in perceptual.values() if len(group)>1],
            'audit_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'interpretation':'Identical dHash is a candidate similarity group, not proof of duplicate. Temporal recommendation permits repeated items; no item-disjoint/cold-start claim.'}
    (ROOT/'reports/hm_image_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)}),flush=True)


if __name__=='__main__':main()
