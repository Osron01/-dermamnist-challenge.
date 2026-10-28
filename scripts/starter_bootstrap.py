"""Cross-check with the starter notebook's own bootstrap_difference.

    python scripts/starter_bootstrap.py

The starter compares ONE seed of the baseline with the SAME seed of a variant
(seed 0 vs seed 0, ...). src/evaluate.py pools all seeds of both models into
one interval instead. This script runs the starter's function, copied
unchanged, for every variant and seed, so both can be reported side by side.
Writes results/starter_bootstrap.csv and .md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import REPO_ROOT  # noqa: E402


# --- copied unchanged from notebooks/day4_challenge_starter.ipynb -----------
def bootstrap_difference(y_true, pred_a, pred_b, n_draws=2000, seed=0):
    """95 % interval of balanced_accuracy(B) - balanced_accuracy(A) on the same images (day 3)."""
    g = torch.Generator().manual_seed(seed)
    diffs = []
    for _ in range(n_draws):
        idx = torch.randint(0, len(y_true), (len(y_true),), generator=g)
        diffs.append(balanced_accuracy_score(y_true[idx], pred_b[idx])
                     - balanced_accuracy_score(y_true[idx], pred_a[idx]))
    low, high = torch.quantile(torch.tensor(diffs), torch.tensor([0.025, 0.975]))
    return round(low.item(), 3), round(high.item(), 3)
# ----------------------------------------------------------------------------


def preds(variant: str, seed_dir: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    d = REPO_ROOT / "results" / variant / seed_dir
    if not (d / "val_logits.npy").exists():
        return None
    y = torch.from_numpy(np.load(d / "val_labels.npy").reshape(-1).astype(np.int64))
    return y, torch.from_numpy(np.load(d / "val_logits.npy")).argmax(1)


def main(baseline: str = "baseline") -> None:
    root = REPO_ROOT / "results"
    rows = []
    for vdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in (baseline, "figures", "test")):
        for sdir in sorted(vdir.glob("seed*")):
            a, b = preds(baseline, sdir.name), preds(vdir.name, sdir.name)
            if a is None or b is None:
                continue
            assert torch.equal(a[0], b[0]), "val labels differ"
            low, high = bootstrap_difference(a[0], a[1], b[1])
            diff = balanced_accuracy_score(a[0], b[1]) - balanced_accuracy_score(a[0], a[1])
            rows.append({"variant": vdir.name, "seed": int(sdir.name[4:]), "delta": round(diff, 3),
                         "ci_low": low, "ci_high": high, "excludes_0": low > 0 or high < 0})
            print(f"{vdir.name:22s} seed {sdir.name[4:]}  {diff:+.3f} [{low:+.3f}, {high:+.3f}]")
    df = pd.DataFrame(rows)
    df.to_csv(root / "starter_bootstrap.csv", index=False)
    lines = ["| Variant | Seed | Δ vs baseline (same seed) | 95 % CI (starter's bootstrap_difference) |",
             "| --- | --- | --- | --- |"]
    lines += [f"| {r.variant} | {r.seed} | {r.delta:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |"
              for r in df.itertuples()]
    (root / "starter_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
