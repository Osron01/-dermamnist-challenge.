"""E9: post-hoc logit adjustment of an already trained variant (Menon et al., ICLR 2021).

    python scripts/posthoc_la.py                       # adjust the baseline's saved val logits
    python scripts/posthoc_la.py --source e3_flip_rot --name e3_posthoc_la

No retraining: subtract tau * log(class prior) from the saved val logits. tau is
chosen on VAL (one tau for all seeds, the one with the best mean val balanced
accuracy), so the val number of this row is slightly optimistic - say so in
the report. The test split is never touched.

Writes results/<name>/seed<k>/{val_logits,val_labels}.npy and metrics.json,
so scripts/summarize.py picks the row up like any trained variant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import compute_metrics, posthoc_logit_adjust  # noqa: E402
from src.utils import REPO_ROOT  # noqa: E402

TAUS = np.round(np.arange(0.0, 2.01, 0.25), 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="baseline", help="trained variant in results/")
    p.add_argument("--name", default="e9_posthoc_la", help="output variant name")
    a = p.parse_args()

    src_dir = REPO_ROOT / "results" / a.source
    seed_dirs = sorted(d for d in src_dir.glob("seed*") if (d / "val_logits.npy").exists())
    if not seed_dirs:
        sys.exit(f"no saved val logits in {src_dir}")
    runs = []
    for d in seed_dirs:
        meta = json.loads((d / "metrics.json").read_text())
        runs.append((d.name, np.load(d / "val_logits.npy"), np.load(d / "val_labels.npy"),
                     np.array(meta["train_class_counts"]), meta))

    # one tau for all seeds, chosen by mean val balanced accuracy
    curve = {}
    for tau in TAUS:
        curve[float(tau)] = float(np.mean([
            compute_metrics(y, posthoc_logit_adjust(l, c, tau))["balanced_accuracy"]
            for _, l, y, c, _ in runs]))
        print(f"tau {tau:4.2f}  mean val bal_acc {curve[float(tau)]:.4f}")
    best_tau = max(curve, key=curve.get)
    print(f"chosen on val: tau = {best_tau}")

    for seed_name, logits, y, counts, meta in runs:
        out = REPO_ROOT / "results" / a.name / seed_name
        out.mkdir(parents=True, exist_ok=True)
        adjusted = posthoc_logit_adjust(logits, counts, best_tau).astype(np.float32)
        np.save(out / "val_logits.npy", adjusted)
        np.save(out / "val_labels.npy", y)
        (out / "metrics.json").write_text(json.dumps({
            "variant": a.name, "seed": meta["seed"], "source_variant": a.source,
            "source_git_commit": meta["git_commit"], "posthoc_tau": best_tau,
            "tau_chosen_on": "val (mean balanced accuracy over seeds)",
            "tau_curve_val_bal_acc": curve, "train_class_counts": counts.tolist(),
            "val": compute_metrics(y, adjusted),
        }, indent=2))
    print(f"wrote results/{a.name}/")


if __name__ == "__main__":
    main()
