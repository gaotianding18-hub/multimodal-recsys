"""Trainable integration boundary for external multimodal embeddings.

Independent PyTorch implementation; not a copy or full execution of either
VLM2Vec or TorchEasyRec. Inputs must be aligned by a versioned item-ID mapping.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ModalityFusion(nn.Module):
    """[N,Di], [N,Dt], availability [N,2] -> normalized [N,D]."""

    def __init__(self, image_dim, text_dim, dim=128, mode="gate"):
        super().__init__()
        if mode not in ("gate", "mean"):
            raise ValueError("mode must be gate or mean")
        self.mode = mode
        self.image = nn.Linear(image_dim, dim)
        self.text = nn.Linear(text_dim, dim)
        self.gate = nn.Sequential(nn.Linear(2 * dim + 2, dim), nn.ReLU(), nn.Linear(dim, 2))

    def forward(self, image, text, available):
        available = available.bool()
        if not available.any(dim=-1).all():
            raise ValueError("Each item requires at least one available modality")
        # Mask before projection: unavailable values cannot influence gates.
        image = image.masked_fill(~available[:, :1], 0)
        text = text.masked_fill(~available[:, 1:], 0)
        parts = torch.stack([self.image(image), self.text(text)], dim=1)
        logits = self.gate(torch.cat([parts.flatten(1), available.float()], -1))
        if self.mode == "mean":
            logits = torch.zeros_like(logits)
        weights = logits.masked_fill(~available, -torch.inf).softmax(-1)
        return F.normalize((parts * weights.unsqueeze(-1)).sum(1), dim=-1), weights


class HistoryTower(nn.Module):
    """Candidate-independent masked history pooling for indexable retrieval."""

    def __init__(self, dim=128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, history, valid):
        history = history.masked_fill(~valid.bool().unsqueeze(-1), 0)
        pooled = history.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return F.normalize(self.network(pooled), dim=-1)


class InterestRanker(nn.Module):
    """DIN-inspired candidate-dependent attention; returns preference logits.

    Not CTR probability. History [B,L,D], candidate [B,D], valid [B,L].
    Empty history has zero pooled interest, with no padded-item contribution.
    """

    def __init__(self, dim=128):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.score = nn.Sequential(nn.Linear(4 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, history, candidate, valid):
        valid = valid.bool()
        history = history.masked_fill(~valid.unsqueeze(-1), 0)
        query = candidate.unsqueeze(1).expand_as(history)
        features = torch.cat([history, query, history - query, history * query], -1)
        logits = self.attention(features).squeeze(-1).masked_fill(~valid, -1e9)
        weights = logits.softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        interest = (weights.unsqueeze(-1) * history).sum(1)
        return self.score(torch.cat([interest, candidate, interest - candidate,
                                     interest * candidate], -1)).squeeze(-1)


def contrastive_loss(users, items, positives, allowed, temperature=0.1):
    """Multi-positive softmax over explicit allowed training candidates.

    Caller constructs masks using training-only known positives. Repeated item
    IDs must be deduplicated before this function. Unobserved != disliked.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positives, allowed = positives.bool(), allowed.bool()
    if (positives & ~allowed).any() or not positives.any(-1).all():
        raise ValueError("Every query needs an allowed positive")
    logits = (users @ items.T / temperature).masked_fill(~allowed, -torch.inf)
    log_probs = logits.log_softmax(-1).masked_fill(~positives, 0)
    return -(log_probs.sum(-1) / positives.sum(-1)).mean()
