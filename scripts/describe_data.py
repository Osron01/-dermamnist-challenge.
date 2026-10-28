"""Phase A: class counts per split and an example grid (8 images per class).

    python scripts/describe_data.py

Writes results/figures/class_counts.csv, class_counts.png and examples.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CLASS_SHORT, NUM_CLASSES, class_counts, load_split  # noqa: E402
from src.utils import REPO_ROOT  # noqa: E402


def main() -> None:
    out = REPO_ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    splits = {s: load_split(s, transform=None) for s in ("train", "val", "test")}
    table = pd.DataFrame({s: class_counts(ds) for s, ds in splits.items()}, index=CLASS_SHORT)
    table.loc["total"] = table.sum()
    table["train_share_%"] = (100 * table["train"] / table.loc["total", "train"]).round(1)
    table.to_csv(out / "class_counts.csv")
    print(table)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(CLASS_SHORT, table.loc[CLASS_SHORT, "train"], color="#4a6fa5")
    ax.set_yscale("log")
    ax.set_ylabel("training images (log scale)")
    ax.set_title("DermaMNIST train split: 59 : 1 imbalance")
    fig.tight_layout()
    fig.savefig(out / "class_counts.png", dpi=200)

    train = splits["train"]
    labels = np.asarray(train.labels).reshape(-1)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(NUM_CLASSES, 8, figsize=(8, NUM_CLASSES * 1.05))
    for c in range(NUM_CLASSES):
        for j, i in enumerate(rng.choice(np.where(labels == c)[0], 8, replace=False)):
            img, _ = train[int(i)]
            axes[c, j].imshow(np.asarray(img), interpolation="nearest")
            axes[c, j].axis("off")
        axes[c, 0].set_title(CLASS_SHORT[c], fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=200)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
