"""Frozen text-only CLIP history cosine baseline, no training claims."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from recsys.hm_train import ROOT, load, evaluate


class FrozenText:
    def __init__(self,texts):self.vectors=torch.cat([torch.zeros(1,texts.shape[1]),texts])
    def eval(self):pass
    def items(self,ids):return self.vectors[ids]
    def users(self,h):
        valid=h.ne(0)
        mean=(self.vectors[h]*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        return F.normalize(mean,dim=-1)


if __name__=='__main__':
    torch.set_num_threads(4);ids,_,_,_,_,_,evaluation,config=load()
    with np.load(ROOT/'artifacts/hm/clip/features.npz') as data:
        assert data['item_ids'].tolist()==ids
        model=FrozenText(torch.from_numpy(data['texts'].copy()))
    report={split:evaluate(model,rows,ids,'cpu',12,batch=64) for split,rows in evaluation.items()}
    report['model']='Frozen CLIP text average history cosine; full catalogue; no fitted parameters'
    (ROOT/'reports/hm_frozen_clip_text_baseline.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
