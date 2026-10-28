"""Load an exported TorchScript model the way the grader will and score it on VAL.

    python scripts/check_model.py model.pt

Loads the file on CPU in a fresh process, feeds the starter-normalised
(N, 3, 28, 28) val images and prints the val metrics, which must match the
results table (roadmap export checklist). Uses the val split only; the test
split is reserved for scripts/final_test.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CLASS_SHORT, build_transforms, get_mean_std, load_split  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="model.pt")
    a = p.parse_args()

    mean, std = get_mean_std()
    ds = load_split("val", build_transforms(None, mean, std, train=False))
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = np.asarray(ds.labels).reshape(-1)

    model = torch.jit.load(a.model, map_location="cpu").eval()
    with torch.no_grad():
        logits = torch.cat([model(x[i:i + 256]) for i in range(0, len(x), 256)]).numpy()
    assert logits.shape == (len(x), 7), logits.shape
    m = compute_metrics(y, logits)
    print(f"{a.model}: val balanced accuracy {m['balanced_accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
          f"accuracy {m['accuracy']:.4f}")
    print("  sensitivity: " + "  ".join(f"{c} {s:.2f}" for c, s in zip(CLASS_SHORT, m["sensitivity_per_class"])))
    print(f"  size {Path(a.model).stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
