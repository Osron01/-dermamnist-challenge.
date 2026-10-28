"""Sharpness-Aware Minimisation (Foret et al., ICLR 2021), used in E8 following
Rangwani et al., NeurIPS 2022 (SAM helps minority classes escape saddle points).

Usage per batch (two forward/backward passes):
    loss_fn(model(x), y).backward(); opt.first_step()
    loss_fn(model(x), y).backward(); opt.second_step()
"""
from __future__ import annotations

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        if rho < 0:
            raise ValueError("rho must be non-negative")
        super().__init__(params, dict(rho=rho, **kwargs))
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self) -> None:
        """Move the weights to the (approximate) worst point within radius rho."""
        grad_norm = torch.norm(torch.stack([
            p.grad.norm(2) for g in self.param_groups for p in g["params"] if p.grad is not None
        ]), 2)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        self.zero_grad()

    @torch.no_grad()
    def second_step(self) -> None:
        """Go back to the original weights and step with the gradient taken at the worst point."""
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p].pop("e_w"))
        self.base_optimizer.step()
        self.zero_grad()

    def step(self, closure=None):  # pragma: no cover - use first_step/second_step
        raise RuntimeError("SAM needs first_step() and second_step(); see the module docstring.")
