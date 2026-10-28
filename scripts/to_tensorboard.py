"""Write every saved run's per-epoch history to TensorBoard logs (for the report screenshot).

    python scripts/to_tensorboard.py
    tensorboard --logdir runs                  # then open http://localhost:6006

Each run in results/<variant>/seed<k>/metrics.json already stores its history
(train loss, val balanced accuracy, macro-F1 per epoch), so nothing is retrained.
Runs are named <variant>/seed<k>, so TensorBoard's run filter can show one
variant at a time. runs/ is git-ignored and can be rebuilt any time.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CLASS_SHORT  # noqa: E402
from src.utils import REPO_ROOT  # noqa: E402


def main() -> None:
    out = REPO_ROOT / "runs"
    shutil.rmtree(out, ignore_errors=True)
    n = 0
    for f in sorted((REPO_ROOT / "results").glob("*/seed*/metrics.json")):
        meta = json.loads(f.read_text())
        if "history" not in meta:          # post-hoc rows have no training curve
            continue
        w = SummaryWriter(log_dir=str(out / f.parent.parent.name / f.parent.name))
        for h in meta["history"]:
            w.add_scalar("loss/train", h["train_loss"], h["epoch"])
            w.add_scalar("val/balanced_accuracy", h["balanced_accuracy"], h["epoch"])
            w.add_scalar("val/macro_f1", h["macro_f1"], h["epoch"])
            w.add_scalar("val/accuracy", h["accuracy"], h["epoch"])
            for c, s in zip(CLASS_SHORT, h["sensitivity_per_class"]):
                w.add_scalar(f"val_sensitivity/{c}", s, h["epoch"])
        w.add_text("run", f"variant `{meta['variant']}`, seed {meta['seed']}, commit `{meta['git_commit']}`, "
                          f"{meta['minutes']:.1f} min on {meta['device']}")
        w.close()
        n += 1
    print(f"{n} runs written to {out}  ->  tensorboard --logdir runs")


if __name__ == "__main__":
    main()
