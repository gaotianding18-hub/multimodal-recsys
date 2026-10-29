"""LoRA inside official CLIP final MLP blocks; backbone base weights frozen."""
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__();self.base=base;self.base.requires_grad_(False)
        self.a=nn.Parameter(torch.empty(rank,base.in_features));self.b=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.a,a=5**.5);self.scale=alpha/rank
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.a),self.b)*self.scale


def inject(model,rank=4,alpha=8,blocks=1):
    model.requires_grad_(False)
    for tower in [model.visual.transformer,model.transformer]:
        for block in tower.resblocks[-blocks:]:
            for name in ['c_fc','c_proj']:
                base=getattr(block.mlp,name);setattr(block.mlp,name,LoRALinear(base,rank,alpha))
    return {name:p for name,p in model.named_parameters() if p.requires_grad}


def symmetric_loss(images,texts,positive,temperature=.07):
    logits=F.normalize(images,dim=-1)@F.normalize(texts,dim=-1).T/temperature
    mask=positive.bool()
    def loss(values,mask):return -(values.log_softmax(-1).masked_fill(~mask,0).sum(-1)/mask.sum(-1)).mean()
    return (loss(logits,mask)+loss(logits.T,mask.T))/2
