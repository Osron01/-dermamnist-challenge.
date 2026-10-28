"""Report and poster figures from saved val logits (run scripts/summarize.py first).

    python scripts/make_figures.py                     # compare the best variant with the baseline
    python scripts/make_figures.py --compare e2_logit_adjusted

Writes to results/figures/:
  forest_delta.png      Δ val balanced accuracy vs baseline with 95 % CI, every variant
  sensitivity.png       per-class val sensitivity (mean over seeds), baseline vs --compare
  confusion_<v>.png     row-normalised val confusion matrix (mean over seeds)
  curves.png            val balanced accuracy per epoch (mean over seeds), every trained variant
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CLASS_SHORT  # noqa: E402
from src.evaluate import confusion  # noqa: E402
from src.utils import REPO_ROOT  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RESULTS, FIGS = REPO_ROOT / "results", REPO_ROOT / "results" / "figures"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def seed_dirs(variant: str) -> list[Path]:
    return sorted(d for d in (RESULTS / variant).glob("seed*") if (d / "val_logits.npy").exists())


def forest(df: pd.DataFrame, baseline: str) -> None:
    d = df[df.variant != baseline].sort_values("delta")
    fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(d) + 0.9))
    ax.axvline(0, color=MUTED, lw=1, ls="--", zorder=1)
    for i, r in enumerate(d.itertuples()):
        clear = r.ci_low > 0 or r.ci_high < 0
        ax.plot([r.ci_low, r.ci_high], [i, i], color=BLUE, lw=2, solid_capstyle="round", zorder=2)
        # filled marker = interval excludes 0; hollow = no clear effect (shape, not colour, carries it)
        ax.plot(r.delta, i, "o", ms=7, zorder=3, mec=BLUE, mew=2, mfc=BLUE if clear else "white")
    ax.set_yticks(range(len(d)), d.variant)
    ax.set_xlabel("Δ val balanced accuracy vs baseline (95 % bootstrap CI)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Filled: interval excludes 0", fontsize=8, color=MUTED, loc="left")
    fig.savefig(FIGS / "forest_delta.png")
    plt.close(fig)


def sensitivity(df: pd.DataFrame, baseline: str, other: str) -> None:
    cols = [f"sens_{c}" for c in CLASS_SHORT]
    a = df.set_index("variant").loc[baseline, cols].to_numpy(float)
    b = df.set_index("variant").loc[other, cols].to_numpy(float)
    x, w = np.arange(len(CLASS_SHORT)), 0.38
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.bar(x - w / 2 - 0.01, a, w, color=BLUE, label=baseline)
    ax.bar(x + w / 2 + 0.01, b, w, color=ORANGE, label=other)
    for xi, v in zip(x, b):
        ax.text(xi + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(x, CLASS_SHORT)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("val sensitivity (mean of seeds)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.savefig(FIGS / "sensitivity.png")
    plt.close(fig)


def confusion_plot(variant: str) -> None:
    cms = [confusion(np.load(d / "val_labels.npy"), np.load(d / "val_logits.npy")) for d in seed_dirs(variant)]
    cm = np.mean(cms, axis=0)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if cm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(CLASS_SHORT)), CLASS_SHORT, rotation=45)
    ax.set_yticks(range(len(CLASS_SHORT)), CLASS_SHORT)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{variant} (val, row-normalised)", fontsize=8, loc="left")
    fig.savefig(FIGS / f"confusion_{variant}.png")
    plt.close(fig)


def curves(variants: list[str], highlight: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for v in variants:
        hists = []
        for d in seed_dirs(v):
            meta = json.loads((d / "metrics.json").read_text())
            if "history" in meta:
                hists.append([h["balanced_accuracy"] for h in meta["history"]])
        if not hists or len({len(h) for h in hists}) != 1:
            continue
        m = np.mean(hists, axis=0)
        ep = np.arange(1, len(m) + 1)
        if v in highlight:
            color = BLUE if v == highlight[0] else ORANGE
            ax.plot(ep, m, color=color, lw=2, label=v, zorder=3)
            ax.text(ep[-1] + 0.2, m[-1], v, fontsize=7, color=INK, va="center")
        else:
            ax.plot(ep, m, color="#b5b3ad", lw=1, zorder=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val balanced accuracy (mean of seeds)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("grey: other variants", fontsize=8, color=MUTED, loc="left")
    fig.savefig(FIGS / "curves.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--compare", default=None, help="variant to compare with the baseline (default: best)")
    a = p.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "summary.csv")
    other = a.compare or df[df.variant != a.baseline].sort_values("bal_acc_mean").variant.iloc[-1]

    forest(df, a.baseline)
    sensitivity(df, a.baseline, other)
    for v in (a.baseline, other):
        confusion_plot(v)
    trained = [v for v in df.sort_values("bal_acc_mean").variant if (REPO_ROOT / "configs" / f"{v}.yaml").exists()]
    curves(list(df.variant), [a.baseline, trained[-1]])       # post-hoc rows have no curve
    print(f"figures in {FIGS} (compared: {a.baseline} vs {other})")


if __name__ == "__main__":
    main()
