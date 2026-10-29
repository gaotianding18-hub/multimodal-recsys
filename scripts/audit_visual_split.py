"""Check grouped pair split against original images and real captions."""
import hashlib,json
from recsys.hm_train import ROOT


def main():
    rows=json.loads((ROOT/'data/hm/visual/catalogue.json').read_text())
    seen={kind:{} for kind in ['family','image','caption']}
    for row in rows:
        item=row['article_id']
        keys={'family':row['product_code'],
              'image':hashlib.sha256((ROOT/'data/hm/raw/images'/item[:3]/(item+'.jpg')).read_bytes()).hexdigest(),
              'caption':'; '.join(str(row[k]) for k in ['prod_name','product_type_name','colour_group_name','detail_desc'] if row.get(k))}
        for kind,key in keys.items():
            previous=seen[kind].setdefault(key,row['pair_split'])
            if previous!=row['pair_split']:raise ValueError(f'Cross-split {kind} group')
    report={'items':len(rows),'checks':['product_code','exact_image_sha256','exact_caption'],'cross_split_groups':0,
            'limitations':['Near duplicates not ruled out','This audits pair matching split, not purchase temporal protocol']}
    (ROOT/'reports/visual_split_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
