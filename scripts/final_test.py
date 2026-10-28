"""Evaluate exported TorchScript models on the official TEST split.

Run this ONCE, after the final model is frozen (roadmap: Fri 6 Nov). The test
split must never be used to choose a model or a setting.

    python scripts/final_test.py --models baseline.pt model.pt --i-have-frozen-the-model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CLASS_SHORT, build_transforms, get_mean_std, load_split  # noqa: E402
from src.evaluate import compute_metrics, confusion  # noqa: E402
from src.utils import REPO_ROOT  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="TorchScript files, e.g. baseline.pt model.pt")
    p.add_argument("--i-have-frozen-the-model", action="store_true")
    a = p.parse_args()
    if not a.i_have_frozen_the_model:
        sys.exit("Refusing to touch the test split. Pass --i-have-frozen-the-model once the final model is fixed.")

    mean, std = get_mean_std()
    test = load_split("test", build_transforms(None, mean, std, train=False))
    loader = DataLoader(test, batch_size=512, shuffle=False)
    out_dir = REPO_ROOT / "results" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in a.models:
        model = torch.jit.load(path, map_location="cpu").eval()
        logits, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                logits.append(model(x))
                labels.append(torch.as_tensor(np.asarray(y)).reshape(-1))
        logits, labels = torch.cat(logits).numpy(), torch.cat(labels).numpy()
        m = compute_metrics(labels, logits)
        m["confusion_row_normalised"] = confusion(labels, logits).round(3).tolist()
        name = Path(path).stem
        (out_dir / f"{name}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        sens = "  ".join(f"{c} {s:.2f}" for c, s in zip(CLASS_SHORT, m["sensitivity_per_class"]))
        print(f"{name}: test bal_acc {m['balanced_accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}\n  {sens}")


if __name__ == "__main__":
    main()
