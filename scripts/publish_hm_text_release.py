"""Verify then publish an explicitly text-only local experimental release."""
import hashlib
import json
import subprocess
import sys
from recsys.hm_train import ROOT

if __name__=='__main__':
    directory='artifacts/hm/mm-text-42-08e0dec9-budget10';run=ROOT/directory
    report=json.loads((ROOT/'reports/hm_hybrid_mm-text-42-08e0dec9-budget10_42.json').read_text())
    if report['retriever_sha256']!=hashlib.sha256((run/'model.pt').read_bytes()).hexdigest():raise RuntimeError('Stale ranking report')
    rankfile='hybrid-ranker-42.pt'
    if report['ranker_sha256']!=hashlib.sha256((run/rankfile).read_bytes()).hexdigest():raise RuntimeError('Stale ranker checkpoint')
    manifest={'retriever':directory,'retriever_sha256':report['retriever_sha256'],'ranker_file':rankfile,
              'ranker_sha256':report['ranker_sha256'],'scope':'text_only_experimental',
              'description':'Frozen CLIP text + trained contrastive retrieval + causal interest ranking (experimental)',
              'feature_sha256':json.loads((ROOT/'reports/hm_mm_text_42_08e0dec9-budget10.json').read_text())['feature_sha256'],
              'selection':'Best epoch chosen on validation; extra training budget; baseline remains default',
              'limitations':['Not a full multimodal release','One seed for longer training and ranking','Does not outperform repeat baseline']}
    candidate=ROOT/'artifacts/hm/release.text-candidate.json';candidate.write_text(json.dumps(manifest,indent=2))
    subprocess.run([sys.executable,'-m','scripts.verify_hm_release','--manifest','artifacts/hm/release.text-candidate.json'],cwd=ROOT,check=True)
    candidate.replace(ROOT/'artifacts/hm/release.json')
    print('Verified text-only experimental release published locally; restart service to load it',flush=True)
