"""Local serial pipeline. Progress is durable; no success metrics prewritten."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
PYTHON=ROOT/'.venv/bin/python'
progress=REPORTS/'hm_upgrade_progress.json'


def mark(stage,status='running',**extra):
    value={'stage':stage,'status':status,'updated_unix':time.time(),**extra}
    temporary=progress.with_suffix('.json.part');temporary.write_text(json.dumps(value,indent=2));temporary.replace(progress)
    print(json.dumps(value),flush=True)


def run(stage,argv):
    mark(stage)
    with (REPORTS/f'{stage}.log').open('w') as log:
        result=subprocess.run([str(x) for x in argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{stage} failed; inspect local log (exit {result.returncode})')


def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--download-pid',type=int)
    group.add_argument('--images-ready',action='store_true')
    args=p.parse_args()
    if args.download_pid:
        mark('waiting_for_official_images',download_pid=args.download_pid)
        while True:
            try:os.kill(args.download_pid,0)
            except ProcessLookupError:break
            time.sleep(10)
        summary=REPORTS/'hm_images_download.json'
        if not summary.exists():raise RuntimeError('Download exited without completion report')
        # A new API pass is deliberately not automatic after rate limiting.
    run('hm_image_audit',[PYTHON,'-m','scripts.audit_hm_images'])
    audit=json.loads((REPORTS/'hm_image_audit.json').read_text())
    if audit['coverage']<.95:raise RuntimeError('Audited image coverage below 95%; obtain official images before full experiments')
    run('hm_final_clip_encoding',[PYTHON,'-m','recsys.clip_features','--limit','0','--device','mps','--batch-size','64','--resume'])
    run('hm_frozen_text_baseline',[PYTHON,'-m','scripts.evaluate_hm_clip_baseline'])
    features=ROOT/'artifacts/hm/clip/features.npz'
    report=json.loads(features.with_suffix('.npz.json').read_text())
    if report['available_images']/report['items']<.95:raise RuntimeError('Final image coverage below 95%; do not publish full-image experiment')
    feature_sha=hashlib.sha256(features.read_bytes()).hexdigest();runs=[]
    for mode in ['text','image','mean','gate']:
        for seed in [42,43,44]:
            filename=REPORTS/f'hm_mm_{mode}_{seed}_{feature_sha[:8]}.json'
            existing=json.loads(filename.read_text()) if filename.exists() else {}
            if existing.get('args',{}).get('epochs')!=3 or existing.get('args',{}).get('max_train_queries')!=0:
                run(f'hm_mm_{mode}_{seed}_{feature_sha[:8]}',[PYTHON,'-m','recsys.hm_multimodal','--mode',mode,'--seed',seed,'--epochs','3'])
            value=json.loads(filename.read_text())
            if value['feature_sha256']!=feature_sha:raise RuntimeError('Stale feature experiment')
            runs.append(value)
    selected=max(runs,key=lambda r:r['selection_val_ndcg']);mode=selected['args']['mode'];seed=selected['args']['seed']
    directory=f'artifacts/hm/mm-{mode}-{seed}-{feature_sha[:8]}'
    ranking=[]
    for rank_seed in [42,43,44]:
        name=f'hm_hybrid_{Path(directory).name}_{rank_seed}'
        if not (REPORTS/f'{name}.json').exists():
            run(name,[PYTHON,'-m','recsys.hm_hybrid','--retriever',directory,'--seed',rank_seed])
        ranking.append(json.loads((REPORTS/f'{name}.json').read_text()))
    best=max(ranking,key=lambda r:r['selection_val_ndcg'])
    manifest={'retriever':directory,'retriever_sha256':best['retriever_sha256'],
              'ranker_file':f"hybrid-ranker-{best['args']['seed']}.pt",'ranker_sha256':best['ranker_sha256'],
              'description':f'Frozen CLIP {mode} + task-trained contrastive retrieval + causal-history interest ranking',
              'feature_sha256':feature_sha,'selection':'validation NDCG only; default baseline remains separately available'}
    (ROOT/'artifacts/hm/release.candidate.json').write_text(json.dumps(manifest,indent=2))
    import numpy as np
    summary={'runs':runs,'ranking_runs':ranking,'selected':manifest,'aggregation':{}}
    for mode in ['text','image','mean','gate']:
        values=[r['test']['ndcg@12'] for r in runs if r['args']['mode']==mode]
        summary['aggregation'][mode]={'test_ndcg_mean':float(np.mean(values)),'seed_population_std':float(np.std(values))}
    (REPORTS/'hm_multimodal_summary.json').write_text(json.dumps(summary,indent=2))
    run('hm_final_tests',[PYTHON,'-m','unittest','discover','-s','tests','-q'])
    run('hm_release_verification',[PYTHON,'-m','scripts.verify_hm_release','--manifest','artifacts/hm/release.candidate.json'])
    (ROOT/'artifacts/hm/release.candidate.json').replace(ROOT/'artifacts/hm/release.json')
    mark('experiments_and_release_verified','complete',release=manifest,
         remaining='Restart local HTTP service to load the release; HTTP performance and final report still need review')


if __name__=='__main__':
    try:main()
    except Exception as error:
        mark('pipeline_stopped','failed',error_type=type(error).__name__,reason=str(error));raise
