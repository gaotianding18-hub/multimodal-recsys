"""Extract only training-catalogue images from a user-authorized official ZIP."""
import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from PIL import Image
from recsys.hm_train import ROOT, DATA


def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);args=p.parse_args()
    catalogue=json.loads((DATA/'catalogue.json').read_text());wanted={r['article_id'] for r in catalogue}
    destination=ROOT/'data/hm/raw/images';start=time.perf_counter();extracted=0;missing=set(wanted)
    with zipfile.ZipFile(args.archive) as archive:
        selected={}
        for member in archive.infolist():
            path=Path(member.filename);item=path.stem
            if item in wanted and path.suffix.lower()=='.jpg' and path.parent.name==item[:3]:
                if item in selected:raise RuntimeError('Duplicate archive entry for selected item')
                selected[item]=member
        plan={'selected_entries':len(selected),'selected_uncompressed_bytes':sum(m.file_size for m in selected.values()),
              'archive_bytes':args.archive.stat().st_size}
        print(json.dumps(plan),flush=True)
        for i,(item,member) in enumerate(selected.items()):
            target=destination/item[:3]/f'{item}.jpg';missing.discard(item)
            if target.exists():
                with Image.open(target) as image:image.verify()
                continue
            if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('Less than 20 GiB free; stopped safely')
            target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix('.jpg.part')
            # zipfile verifies the member CRC while streaming. No archive path is used as an output path.
            with archive.open(member) as source,temporary.open('wb') as output:shutil.copyfileobj(source,output,1024*1024)
            with Image.open(temporary) as image:image.verify()
            temporary.replace(target);extracted+=1
            if (i+1)%1000==0:print('selected entries processed',i+1,flush=True)
    report={**plan,'newly_extracted':extracted,'missing_item_ids':sorted(missing),'seconds':time.perf_counter()-start,
            'source':'official competition archive; no synthetic images','archive_retained':True}
    (ROOT/'reports/hm_selected_extraction.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
