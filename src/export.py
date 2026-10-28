"""Export the final model to TorchScript (model.pt) and check it.

    python -m src.export --config configs/final.yaml --seeds 0 1 2 --tau 0.0 --out model.pt

The exported module takes a normalised (N, 3, 28, 28) float tensor and returns
(N, 7) logits. Anything else the model needs (upsampling, seed ensemble,
post-hoc logit adjustment) is baked into the module, because the grader's
script only feeds normalised 28x28 images and reads logits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data import get_mean_std
from src.losses import log_prior
from src.models import build_model
from src.utils import REPO_ROOT, load_config


class FinalModel(nn.Module):
    """Averages the logits of one or more trained nets, then subtracts tau * log(prior)."""

    def __init__(self, nets: list[nn.Module], adjustment: torch.Tensor):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.register_buffer("adjustment", adjustment.view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs: list[torch.Tensor] = []
        for net in self.nets:  # explicit loop: TorchScript-friendly
            outs.append(net(x))
        return torch.stack(outs, dim=0).mean(dim=0) - self.adjustment


def build_final(cfg: dict, seeds: list[int], tau: float, counts: np.ndarray | None,
                results_root: Path, mean, std) -> FinalModel:
    nets = []
    for s in seeds:
        net = build_model(cfg, data_mean=mean, data_std=std)
        state = torch.load(results_root / cfg["name"] / f"seed{s}" / "model_state.pt", map_location="cpu")
        net.load_state_dict(state)
        nets.append(net.eval())
    adj = tau * log_prior(counts) if (tau and counts is not None) else torch.zeros(7)
    return FinalModel(nets, adj).eval().cpu()


def to_torchscript(model: nn.Module) -> torch.jit.ScriptModule:
    try:
        return torch.jit.script(model)
    except Exception as err:  # some layers do not script; tracing is fine for a fixed graph
        print(f"torch.jit.script failed ({type(err).__name__}); falling back to torch.jit.trace")
        return torch.jit.trace(model, torch.randn(2, 3, 28, 28))


def check_export(path: Path, reference: nn.Module) -> None:
    """Reload the file on CPU and compare it with the eager model."""
    loaded = torch.jit.load(str(path), map_location="cpu").eval()
    for n in (1, 256):
        x = torch.randn(n, 3, 28, 28)
        with torch.no_grad():
            out, ref = loaded(x), reference(x)
        assert out.shape == (n, 7), f"expected ({n}, 7), got {tuple(out.shape)}"
        diff = (out - ref).abs().max().item()
        assert diff < 1e-4, f"exported model differs from eager model by {diff}"
    size_mb = path.stat().st_size / 2**20
    print(f"OK: {path} loads on CPU, outputs (N, 7), matches the eager model. Size {size_mb:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--tau", type=float, default=0.0, help="post-hoc logit adjustment, chosen on VAL")
    p.add_argument("--out", default="model.pt")
    p.add_argument("--results-root", default=str(REPO_ROOT / "results"))
    p.add_argument("--fake-data", action="store_true")
    a = p.parse_args()

    cfg = load_config(a.config)
    mean, std = ((0.5,) * 3, (0.25,) * 3) if a.fake_data else get_mean_std()
    results_root = Path(a.results_root)
    rec =json.loads((results_root / cfg["name"] / f"seed{a.seeds[0]}" / "metrics.json").read_text())
    counts = np.array(rec["train_class_counts"])
    model = build_final(cfg, a.seeds, a.tau, counts, results_root, mean, std)
    scripted = to_torchscript(model)
    out = Path(a.out)
    scripted.save(str(out))
    check_export(out, model)


if __name__ == "__main__":
    main()
