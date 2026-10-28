"""Prepare what the app needs from the VAL split (never test):

    python scripts/app_assets.py            # after exporting model.pt

* the malignancy-flag threshold: the app warns when P(akiec) + P(bcc) + P(mel)
  reaches it. Chosen on val as the highest threshold that still flags at least
  90 % of the val images of those three classes.
* one example image per class (the first val image of that class, not picked
  by how well the model does on it), saved as PNG for the gallery.
* the model's val metrics per class, for the model-card tab.

Writes app/app_config.json and app/examples/<class>.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CLASS_SHORT, build_transforms, get_mean_std, load_split  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.utils import REPO_ROOT  # noqa: E402

MALIGNANT = [0, 1, 4]          # akiec, bcc, mel
TARGET_SENSITIVITY = 0.90


def main() -> None:
    app = REPO_ROOT / "app"
    model = torch.jit.load(str(REPO_ROOT / "model.pt"), map_location="cpu").eval()
    mean, std = get_mean_std()
    ds = load_split("val", build_transforms(None, mean, std, train=False))
    raw = load_split("val", transform=None)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = np.asarray(ds.labels).reshape(-1)
    with torch.no_grad():
        logits = torch.cat([model(x[i:i + 256]) for i in range(0, len(x), 256)])
    probs = F.softmax(logits, dim=1).numpy()

    p_mal = probs[:, MALIGNANT].sum(axis=1)
    is_mal = np.isin(y, MALIGNANT)
    # highest threshold whose flag still catches >= 90 % of the malignant val images
    threshold = float(np.quantile(p_mal[is_mal], 1 - TARGET_SENSITIVITY, method="lower"))
    threshold = np.floor(threshold * 1000) / 1000       # stored with 3 decimals: round down, keep >= 90 %
    flagged = p_mal >= threshold
    flag = {"threshold": float(threshold), "classes": [CLASS_SHORT[c] for c in MALIGNANT],
            "val_sensitivity": round(float(flagged[is_mal].mean()), 3),
            "val_flag_rate_benign": round(float(flagged[~is_mal].mean()), 3),
            "chosen_on": f"val: highest threshold flagging >= {TARGET_SENSITIVITY:.0%} of malignant images"}

    (app / "examples").mkdir(exist_ok=True)
    for c, name in enumerate(CLASS_SHORT):
        i = int(np.flatnonzero(y == c)[0])
        raw[i][0].save(app / "examples" / f"{name}.png")

    m = compute_metrics(y, logits.numpy())
    cfg = {"malignancy_flag": flag,
           "val_metrics": {"balanced_accuracy": round(m["balanced_accuracy"], 3),
                           "macro_f1": round(m["macro_f1"], 3),
                           "sensitivity_per_class": dict(zip(CLASS_SHORT, [round(s, 2) for s in m["sensitivity_per_class"]]))}}
    (app / "app_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
