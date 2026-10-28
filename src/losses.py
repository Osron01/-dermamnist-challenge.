"""Loss functions for the imbalance experiments.

  ce              plain cross-entropy (baseline)
  weighted_ce     E1: cross-entropy with class weights proportional to 1 / n_c
  logit_adjusted  E2: logit-adjusted loss (Menon et al., ICLR 2021), which with
                  tau = 1 equals Balanced Softmax (Ren et al., NeurIPS 2020)
  focal           optional: focal loss (Lin et al., ICCV 2017)

Deferred re-weighting (DRW, Cao et al., NeurIPS 2019) is handled in train.py by
switching the class weights on after a fraction of the epochs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_frequency_weights(counts: np.ndarray, power: float = 1.0) -> torch.Tensor:
    """w_c proportional to (1 / n_c)^power, normalised so the mean weight is 1."""
    w = (1.0 / np.asarray(counts, dtype=float)) ** power
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def log_prior(counts: np.ndarray) -> torch.Tensor:
    p = np.asarray(counts, dtype=float) / np.sum(counts)
    return torch.tensor(np.log(p), dtype=torch.float32)


class LogitAdjustedCE(nn.Module):
    """Cross-entropy on logits + tau * log(prior). At test time the raw logits are used,
    which removes the prior and targets the balanced error."""

    def __init__(self, counts: np.ndarray, tau: float = 1.0):
        super().__init__()
        self.register_buffer("adj", tau * log_prior(counts))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adj, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_p, target, weight=self.weight, reduction="none")
        p_t = log_p.gather(1, target[:, None]).squeeze(1).exp()
        return ((1 - p_t) ** self.gamma * ce).mean()


def build_loss(cfg: dict, counts: np.ndarray, use_weights: bool = True) -> nn.Module:
    """`use_weights=False` lets DRW start without class weights."""
    lc = cfg.get("loss", {"name": "ce"})
    name = lc["name"]
    weight = inverse_frequency_weights(counts, lc.get("weight_power", 1.0)) if use_weights else None
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=lc.get("label_smoothing", 0.0))
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=lc.get("label_smoothing", 0.0))
    if name == "logit_adjusted":
        return LogitAdjustedCE(counts, tau=lc.get("tau", 1.0))
    if name == "focal":
        return FocalLoss(gamma=lc.get("gamma", 2.0), weight=weight if lc.get("weighted") else None)
    raise ValueError(f"unknown loss: {name}")
