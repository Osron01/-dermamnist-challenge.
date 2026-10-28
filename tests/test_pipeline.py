"""Fast checks that run without downloading DermaMNIST (random images).

    pytest -q
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.evaluate import bootstrap_difference_multiseed, compute_metrics, posthoc_logit_adjust
from src.export import build_final, check_export, to_torchscript
from src.losses import LogitAdjustedCE, build_loss, inverse_frequency_weights
from src.models import build_model
from src.train import train
from src.utils import REPO_ROOT, load_config

CONFIGS = REPO_ROOT / "configs"
COUNTS = np.array([228, 359, 769, 80, 779, 4693, 99])
FAKE_MEAN, FAKE_STD = (0.5,) * 3, (0.25,) * 3


def test_config_inheritance():
    cfg = load_config(CONFIGS / "e8_sam.yaml")
    assert cfg["name"] == "e8_sam"
    assert cfg["loss"]["name"] == "logit_adjusted"      # from e2
    assert cfg["optim"]["sam"]["enabled"] is True        # own change
    assert cfg["optim"]["lr"] == load_config(CONFIGS / "baseline.yaml")["optim"]["lr"]  # from baseline


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_every_config_builds_a_model_with_7_logits(name):
    cfg = load_config(CONFIGS / name)
    if cfg["model"]["name"] == "resnet_up":
        cfg["model"].update(pretrained=False, size=32)   # no download, faster
    model = build_model(cfg, FAKE_MEAN, FAKE_STD).eval()
    with torch.no_grad():
        assert model(torch.randn(4, 3, 28, 28)).shape == (4, 7)


def test_class_weights_and_logit_adjustment():
    w = inverse_frequency_weights(COUNTS)
    assert torch.isclose(w.mean(), torch.tensor(1.0))
    assert w[3] > w[5]                                   # dermatofibroma weighted above nevi
    logits, target = torch.randn(16, 7), torch.randint(0, 7, (16,))
    uniform = LogitAdjustedCE(np.ones(7), tau=1.0)       # uniform prior: same as plain CE
    assert torch.isclose(uniform(logits, target), F.cross_entropy(logits, target))
    for loss in ("ce", "weighted_ce", "logit_adjusted", "focal"):
        assert torch.isfinite(build_loss({"loss": {"name": loss}}, COUNTS)(logits, target))


def test_metrics_and_bootstrap():
    y = np.repeat(np.arange(7), 10)
    perfect = np.eye(7)[y] * 5
    assert compute_metrics(y, perfect)["balanced_accuracy"] == 1.0
    always_nevus = np.tile(np.eye(7)[5], (len(y), 1))
    assert np.isclose(compute_metrics(y, always_nevus)["balanced_accuracy"], 1 / 7)
    same = bootstrap_difference_multiseed(y, [perfect], [perfect], n_boot=50)
    assert same["diff"] == same["ci_low"] == same["ci_high"] == 0.0
    better = bootstrap_difference_multiseed(y, [perfect], [always_nevus], n_boot=200)
    assert better["ci_low"] > 0
    adjusted = posthoc_logit_adjust(np.zeros((1, 7)), COUNTS, tau=1.0)
    assert adjusted.argmax() == 3                        # rarest class wins on a tie


def _smoke_cfg(name: str) -> dict:
    cfg = copy.deepcopy(load_config(CONFIGS / name))
    cfg["data"]["num_workers"] = 0
    return cfg


@pytest.mark.parametrize("name", ["baseline.yaml", "e8_sam.yaml", "e6_cosine.yaml"])
def test_one_epoch_trains_and_saves(tmp_path: Path, name):
    rec = train(_smoke_cfg(name), seed=0, fake=True, out_root=tmp_path, epochs_override=1)
    run_dir = tmp_path / rec["variant"] / "seed0"
    assert np.load(run_dir / "val_logits.npy").shape == (128, 7)
    assert (run_dir / "metrics.json").exists() and (run_dir / "model_state.pt").exists()


def test_seeded_runs_are_reproducible(tmp_path: Path):
    a = train(_smoke_cfg("baseline.yaml"), seed=3, fake=True, out_root=tmp_path / "a", epochs_override=1)
    b = train(_smoke_cfg("baseline.yaml"), seed=3, fake=True, out_root=tmp_path / "b", epochs_override=1)
    assert a["history"][0]["train_loss"] == pytest.approx(b["history"][0]["train_loss"])


def test_export_roundtrip(tmp_path: Path):
    cfg = _smoke_cfg("baseline.yaml")
    for s in (0, 1):
        train(cfg, seed=s, fake=True, out_root=tmp_path, epochs_override=1)
    model = build_final(cfg, [0, 1], tau=1.0, counts=COUNTS, results_root=tmp_path,
                        mean=FAKE_MEAN, std=FAKE_STD)
    path = tmp_path / "model.pt"
    to_torchscript(model).save(str(path))
    check_export(path, model)                            # raises if shape or values are off
