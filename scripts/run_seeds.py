"""Train one config with several seeds (default 0, 1, 2), one after another.

    python scripts/run_seeds.py configs/e1_class_weights.yaml
    python scripts/run_seeds.py configs/e7_resnet18.yaml --seeds 0 1 2 --wandb

Works the same on Windows, macOS, Linux and Colab.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    scores = []
    for s in a.seeds:
        rec = train(cfg, s, fake=a.fake_data, use_wandb=a.wandb, epochs_override=a.epochs)
        scores.append(rec["val"]["balanced_accuracy"])
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print(f"\n{cfg['name']}: val balanced accuracy {mean:.4f} ± {std:.4f} over seeds {a.seeds}")


if __name__ == "__main__":
    main()
