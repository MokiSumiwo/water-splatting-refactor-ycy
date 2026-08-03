"""Diagnostics for MCMC-style Gaussian density control."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor


def mcmc_tensor_stats(values: Tensor) -> Dict[str, float]:
    """Return compact robust stats for JSONL logging."""

    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"mean": 0.0, "p01": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(values.mean().item()),
        "p01": float(torch.quantile(values, 0.01).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
    }


def effective_gaussian_count(alpha: Tensor, eps: float = 1e-12) -> float:
    """Return ``(sum alpha)^2 / (sum alpha^2 + eps)``."""

    alpha = alpha.detach().float().reshape(-1).clamp_min(0.0)
    denom = alpha.square().sum().clamp_min(eps)
    return float((alpha.sum().square() / denom).item())


def parent_sampling_entropy(parent_indices: Tensor, num_alive: int) -> float:
    """Return normalized entropy over sampled parent ids."""

    parent_indices = parent_indices.detach().reshape(-1)
    if parent_indices.numel() == 0 or num_alive <= 1:
        return 0.0
    _, counts = torch.unique(parent_indices, return_counts=True)
    probs = counts.float() / counts.sum().float().clamp_min(1.0)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return float((entropy / math.log(float(num_alive))).clamp(0.0, 1.0).item())

