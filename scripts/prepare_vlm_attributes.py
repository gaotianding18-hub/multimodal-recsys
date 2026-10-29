"""Directory labels as weak visual supervision; grouped split reused unchanged."""
import json,hashlib
from collections import Counter
from recsys.hm_train import ROOT

PROMPT='Look at the product image. Return only a JSON object with keys "product_type" and "color". Do not infer material, brand, or user preferences.'

def main():
    rows=json.loads((ROOT/'data/hm/visual/catalogue.json').read_text());out=ROOT/'data/hm/vlm';out.mkdir(parents=True,exist_ok=True)
    counts={};hashes={}
    for split in ['train','val','test']:
        records=[{'article_id':r['article_id'],'image':f"data/hm/raw/images/{r['article_id'][:3]}/{r['article_id']}.jpg",'prompt':PROMPT,
                  'answer':json.dumps({'product_type':r['product_type_name'],'color':r['colour_group_name']},ensure_ascii=False),
                  'label_source':'Official catalogue metadata; weak visual supervision'} for r in rows if r['pair_split']==split]
        p=out/f'{split}.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in records));counts[split]=len(records);hashes[split]=hashlib.sha256(p.read_bytes()).hexdigest()
    report={'counts':counts,'sha256':hashes,'split':'Existing product family/exact image/exact caption grouped split',
            'limitations':['Catalogue labels are not manually verified visual ground truth','Nonrandom 590 image subset','Fine color and product taxonomy may be ambiguous','No material labels or invented user behavior'],
            'acceptance':['Finite loss and actual adapter updates','Answer-only loss masking','Frozen base unchanged','Validation selects checkpoint; held-out metrics separate','Compare base/fine-tuned format validity and metadata agreement']}
    (ROOT/'reports/vlm_attributes_data.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
