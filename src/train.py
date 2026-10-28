"""Train one variant with one seed and save its val logits.

    python -m src.train --config configs/baseline.yaml --seed 0
    python -m src.train --config configs/e1_class_weights.yaml --seed 1 --wandb

Output: results/<variant>/seed<k>/{val_logits.npy, val_labels.npy, metrics.json}
plus model_state.pt (git-ignored). Everything later (tables, bootstrap, export)
is computed from these files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data import get_loaders, get_mean_std
from src.evaluate import compute_metrics
from src.losses import build_loss
from src.models import build_model
from src.sam import SAM
from src.utils import REPO_ROOT, get_device, git_commit, load_config, set_seed


def build_optimizer(cfg: dict, params):
    oc = cfg.get("optim", {})
    name, lr, wd = oc.get("name", "adam"), oc.get("lr", 1e-3), oc.get("weight_decay", 0.0)
    classes = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    kwargs = dict(lr=lr, weight_decay=wd)
    if name == "sgd":
        kwargs.update(momentum=oc.get("momentum", 0.9), nesterov=True)
    sam_cfg = oc.get("sam", {})
    if sam_cfg.get("enabled"):
        opt = SAM(params, classes[name], rho=sam_cfg.get("rho", 0.05), **kwargs)
        return opt, opt.base_optimizer, True
    opt = classes[name](params, **kwargs)
    return opt, opt, False


def build_scheduler(cfg: dict, optimizer, epochs: int, steps_per_epoch: int):
    sc = cfg.get("schedule", {"name": "none"})
    name = sc.get("name", "none")
    if name == "none":
        return None
    total = epochs * steps_per_epoch
    warmup = int(sc.get("warmup_epochs", 0) * steps_per_epoch)
    if name == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)  # stepped every batch
    raise ValueError(f"unknown schedule: {name}")


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def train(cfg: dict, seed: int, fake: bool = False, use_wandb: bool = False,
          out_root: Path | None = None, epochs_override: int | None = None) -> dict:
    gen = set_seed(seed)
    device = get_device()
    name = cfg["name"]
    epochs = epochs_override or cfg.get("train", {}).get("epochs", 30)
    out_dir = (out_root or REPO_ROOT / "results") / name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, counts = get_loaders(cfg, gen, fake=fake)
    mean, std = ((0.5,) * 3, (0.25,) * 3) if fake else get_mean_std()
    model = build_model(cfg, data_mean=mean, data_std=std).to(device)
    optimizer, sched_target, is_sam = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, sched_target, epochs, len(train_loader))

    drw_start = cfg.get("loss", {}).get("drw_start")  # e.g. 0.75 = weights on in the last 25 %
    drw_epoch = int(drw_start * epochs) if drw_start is not None else None
    loss_fn = build_loss(cfg, counts, use_weights=drw_epoch is None).to(device)

    run = None
    if use_wandb or cfg.get("wandb", {}).get("enabled"):
        import wandb
        run = wandb.init(project=cfg.get("wandb", {}).get("project", "dermamnist"),
                         group=name, name=f"{name}-s{seed}", reinit=True,
                         config={**cfg, "seed": seed, "git_commit": git_commit(), "fake_data": fake})

    keep = cfg.get("train", {}).get("checkpoint", "last")  # "last" or "best_val"
    best, history = -1.0, []
    best_logits = best_labels = None
    t0 = time.time()
    for epoch in range(epochs):
        if drw_epoch is not None and epoch == drw_epoch:
            loss_fn = build_loss(cfg, counts, use_weights=True).to(device)
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            if is_sam:
                loss.backward()
                optimizer.first_step()
                loss_fn(model(x), y).backward()
                optimizer.second_step()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * len(y)
            n += len(y)

        val_logits, val_labels = collect_logits(model, val_loader, device)
        m = compute_metrics(val_labels, val_logits)
        history.append({"epoch": epoch + 1, "train_loss": total / n, **m})
        print(f"[{name} s{seed}] epoch {epoch + 1:3d}/{epochs}  loss {total / n:.4f}  "
              f"val bal_acc {m['balanced_accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}", flush=True)
        if run:
            run.log({"epoch": epoch + 1, "train_loss": total / n, "val/balanced_accuracy": m["balanced_accuracy"],
                     "val/macro_f1": m["macro_f1"], "val/accuracy": m["accuracy"],
                     **{f"val/sens_{i}": s for i, s in enumerate(m["sensitivity_per_class"])}})
        if keep == "last" or m["balanced_accuracy"] > best:
            best = m["balanced_accuracy"]
            best_logits, best_labels = val_logits, val_labels
            torch.save(model.state_dict(), out_dir / "model_state.pt")

    final = compute_metrics(best_labels, best_logits)
    np.save(out_dir / "val_logits.npy", best_logits)
    np.save(out_dir / "val_labels.npy", best_labels)
    record = {"variant": name, "seed": seed, "git_commit": git_commit(), "fake_data": fake,
              "device": str(device), "epochs": epochs, "checkpoint_rule": keep,
              "train_class_counts": counts.tolist(), "minutes": (time.time() - t0) / 60,
              "val": final, "history": history, "config": cfg}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    if run:
        run.summary.update({"final/val_balanced_accuracy": final["balanced_accuracy"],
                            "final/val_macro_f1": final["macro_f1"]})
        run.finish()
    print(f"[{name} s{seed}] done: val bal_acc {final['balanced_accuracy']:.4f} -> {out_dir}")
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true", help="log this run to Weights & Biases")
    p.add_argument("--fake-data", action="store_true", help="random images; smoke test without downloading")
    p.add_argument("--epochs", type=int, default=None, help="override the config (smoke tests only)")
    p.add_argument("--out-root", default=None)
    a = p.parse_args()
    train(load_config(a.config), a.seed, fake=a.fake_data, use_wandb=a.wandb,
          out_root=Path(a.out_root) if a.out_root else None, epochs_override=a.epochs)


if __name__ == "__main__":
    main()
