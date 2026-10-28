"""Build the results table from saved val logits (no retraining).

    python scripts/summarize.py                 # every variant in results/ vs 'baseline'
    python scripts/summarize.py --baseline baseline --n-boot 2000

Writes results/summary.csv and results/summary.md (paste the .md into notes,
the .csv into the LaTeX table).

A variant built on another variant (configs/e8_sam.yaml has `base:
e2_logit_adjusted.yaml`; a post-hoc row names its `source_variant`) also gets
a second interval against that parent, because that is the one change it makes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CLASS_SHORT  # noqa: E402
from src.evaluate import bootstrap_difference_multiseed, compute_metrics  # noqa: E402
from src.utils import REPO_ROOT  # noqa: E402


def load_runs(variant_dir: Path) -> tuple[np.ndarray | None, list[np.ndarray], list[int]]:
    labels, logits, seeds = None, [], []
    for seed_dir in sorted(variant_dir.glob("seed*")):
        f = seed_dir / "val_logits.npy"
        if f.exists():
            logits.append(np.load(f))
            labels = np.load(seed_dir / "val_labels.npy")
            seeds.append(int(seed_dir.name[4:]))
    return labels, logits, seeds


def parent_of(name: str, variant_dir: Path) -> str | None:
    """The variant this one changes: its config's `base`, or a post-hoc row's source."""
    cfg = REPO_ROOT / "configs" / f"{name}.yaml"
    if cfg.exists():
        base = (yaml.safe_load(cfg.read_text()) or {}).get("base")
        return Path(base).stem if base else None
    meta = next(iter(sorted(variant_dir.glob("seed*/metrics.json"))), None)
    return json.loads(meta.read_text()).get("source_variant") if meta else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(REPO_ROOT / "results"))
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    root = Path(a.results)

    runs = {d.name: load_runs(d) for d in sorted(root.iterdir()) if d.is_dir()}
    runs = {k: v for k, v in runs.items() if v[1]}
    if a.baseline not in runs:
        sys.exit(f"no runs for baseline '{a.baseline}' in {root}")
    y, base_logits, _ = runs[a.baseline]

    rows = []
    for name, (labels, logits, seeds) in runs.items():
        ms = [compute_metrics(labels, l) for l in logits]
        ba = np.array([m["balanced_accuracy"] for m in ms])
        f1 = np.array([m["macro_f1"] for m in ms])
        sens = np.mean([m["sensitivity_per_class"] for m in ms], axis=0)
        row = {"variant": name, "seeds": len(seeds),
               "bal_acc_mean": ba.mean(), "bal_acc_std": ba.std(ddof=1) if len(ba) > 1 else 0.0,
               "macro_f1_mean": f1.mean(), "macro_f1_std": f1.std(ddof=1) if len(f1) > 1 else 0.0}
        if name != a.baseline:
            assert np.array_equal(labels, y), "val labels differ between runs"
            bs = bootstrap_difference_multiseed(y, logits, base_logits, n_boot=a.n_boot)
            row.update(delta=bs["diff"], ci_low=bs["ci_low"], ci_high=bs["ci_high"])
            parent = parent_of(name, root / name)
            if parent and parent != a.baseline and parent in runs:
                bp = bootstrap_difference_multiseed(y, logits, runs[parent][1], n_boot=a.n_boot)
                row.update(parent=parent, delta_parent=bp["diff"],
                           ci_low_parent=bp["ci_low"], ci_high_parent=bp["ci_high"])
        row.update({f"sens_{c}": s for c, s in zip(CLASS_SHORT, sens)})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("bal_acc_mean", ascending=False)
    df.to_csv(root / "summary.csv", index=False)

    lines = ["| Variant | Seeds | Val BalAcc | Δ vs baseline [95 % CI] | Δ vs parent [95 % CI] | Val macro-F1 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in df.itertuples():
        delta = "—" if r.variant == a.baseline else f"{r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
        has_parent = isinstance(getattr(r, "parent", None), str)
        vs_parent = (f"{r.delta_parent:+.3f} [{r.ci_low_parent:+.3f}, {r.ci_high_parent:+.3f}] vs {r.parent}"
                     if has_parent else "—")
        lines.append(f"| {r.variant} | {r.seeds} | {r.bal_acc_mean:.3f} ± {r.bal_acc_std:.3f} | {delta} | "
                     f"{vs_parent} | {r.macro_f1_mean:.3f} ± {r.macro_f1_std:.3f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
