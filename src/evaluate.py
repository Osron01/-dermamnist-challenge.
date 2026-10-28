"""Metrics and the bootstrap interval. Every number in the report comes from here.

All functions work on saved logits (numpy arrays), so tables can be rebuilt
without retraining anything.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from src.data import NUM_CLASSES

LABELS = list(range(NUM_CLASSES))


def predict(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits).argmax(axis=1)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    """Balanced accuracy (main metric), macro-F1, per-class sensitivity, accuracy."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = predict(logits)
    sens = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "sensitivity_per_class": [float(s) for s in sens],
    }


def confusion(y_true: np.ndarray, logits: np.ndarray, normalize: str | None = "true") -> np.ndarray:
    return confusion_matrix(np.asarray(y_true).reshape(-1), predict(logits), labels=LABELS, normalize=normalize)


def _bal_acc_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy without sklearn overhead (used inside the bootstrap loop).
    Classes absent from a resample are skipped, as sklearn does."""
    recalls = []
    for c in LABELS:
        mask = y_true == c
        if mask.any():
            recalls.append((y_pred[mask] == c).mean())
    return float(np.mean(recalls))


def bootstrap_difference_multiseed(
    y_true: np.ndarray,
    logits_variant: list[np.ndarray],
    logits_baseline: list[np.ndarray],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """95 % bootstrap CI of  mean_seeds BalAcc(variant) - mean_seeds BalAcc(baseline).

    Each resample draws val image indices with replacement and applies the SAME
    indices to every run of both models (paired bootstrap). This measures how
    much the difference could change with a different sample of val images.

    NOTE: the starter notebook ships `bootstrap_difference`. Compare it with this
    function on one pair of runs; if they differ, use the starter's and say so.
    """
    y_true = np.asarray(y_true).reshape(-1)
    preds_v = [predict(l) for l in logits_variant]
    preds_b = [predict(l) for l in logits_baseline]
    n = len(y_true)
    rng = np.random.default_rng(seed)

    def diff(idx: np.ndarray) -> float:
        yv = y_true[idx]
        mv = np.mean([_bal_acc_fast(yv, p[idx]) for p in preds_v])
        mb = np.mean([_bal_acc_fast(yv, p[idx]) for p in preds_b])
        return mv - mb

    observed = diff(np.arange(n))
    boots = np.array([diff(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(observed), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot}


def posthoc_logit_adjust(logits: np.ndarray, counts: np.ndarray, tau: float) -> np.ndarray:
    """E9: subtract tau * log(prior) from trained logits (Menon et al., ICLR 2021).
    Choose tau on VAL only."""
    prior = np.asarray(counts, dtype=float) / np.sum(counts)
    return np.asarray(logits) - tau * np.log(prior)
