"""Recover evaluation from already-saved training output; never invent duration."""
import argparse
import hashlib
import json
import time
import torch
from recsys.hm_train import ROOT, load, evaluate
from recsys.hm_hybrid import load_retriever

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();torch.set_num_threads(4)
    directory=ROOT/args.directory
    checkpoint=torch.load(directory/'model.pt',weights_only=True)
    model,ids,_=load_retriever(directory)
    data_ids,_,histories,_,_,_,evaluation,config=load()
    if ids!=data_ids:raise RuntimeError('Data/model mapping mismatch')
    feature_path=ROOT/checkpoint['args']['features']
    if hashlib.sha256(feature_path.read_bytes()).hexdigest()!=checkpoint['feature_sha256']:raise RuntimeError('Feature version changed')
    logs=json.loads((directory/'logs.json').read_text());best=max(logs,key=lambda r:r['val']['ndcg@12'])
    # Confirm the stored best checkpoint agrees with its actual logged validation metrics.
    started=time.perf_counter();val=evaluate(model,evaluation['val'],ids,'cpu',12)
    if abs(val['ndcg@12']-best['val']['ndcg@12'])>1e-7:raise RuntimeError('Checkpoint/log validation mismatch')
    test=evaluate(model,evaluation['test'],ids,'cpu',12)
    source=directory/'trainer_at_report_failure.py'
    report={'args':checkpoint['args'],'train_queries':len(histories),'candidate_items':len(ids),
            'image_coverage':float(checkpoint['state']['available'][1:,0].float().mean()),'logs':logs,
            'selection_val_ndcg':best['val']['ndcg@12'],'test':test,'seconds':None,
            'recovery_eval_seconds':time.perf_counter()-started,'frozen_clip':True,
            'feature_sha256':checkpoint['feature_sha256'],'model_sha256':hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest(),
            'trainer_sha256':hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None,
            'report_recovered':True,'protocol':config,
            'limitations':['Original total duration not retained, not reconstructed','Exploratory test, not blind',
                           'Ten-epoch budget versus three-epoch run is not equal-compute algorithm ablation']}
    run=checkpoint['args'];tag='-'+run['run_tag'] if run['run_tag'] else ''
    name=f"hm_mm_{run['mode']}_{run['seed']}_{checkpoint['feature_sha256'][:8]}{tag}.json"
    (ROOT/'reports'/name).write_text(json.dumps(report,indent=2));print(json.dumps({'test':test,'selection_val_ndcg':report['selection_val_ndcg'],'report':name}),flush=True)
