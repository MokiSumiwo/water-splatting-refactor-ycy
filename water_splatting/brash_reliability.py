"""Detached reliability statistics used by BRASH's weak gate prior."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def build_reliability_prior(
    *,
    support: Tensor,
    depth_variance: Tensor,
    transmission: Tensor,
    view_spread: Tensor,
    bank_size: int,
    minimum_support: int,
    transmission_floor: float,
    depth_temperature: float,
    view_spread_scale: float,
    gate_floor: float = 0.05,
) -> tuple[Tensor, Dict[str, Tensor]]:
    """Return a detached [N, 1] confidence and diagnostic components."""

    if bank_size < 1 or minimum_support < 1 or minimum_support > bank_size:
        raise ValueError("invalid BRASH support configuration")
    if depth_temperature <= 0.0 or view_spread_scale <= 0.0:
        raise ValueError("BRASH reliability temperatures must be positive")
    if not 0.0 <= transmission_floor < 1.0:
        raise ValueError("BRASH transmission floor must be in [0, 1)")
    gate_floor = float(min(max(gate_floor, 0.0), 1.0))

    support = support.detach().float().reshape(-1)
    depth_variance = depth_variance.detach().float().reshape(-1).clamp_min(0.0)
    transmission = transmission.detach().float().reshape(-1).clamp(0.0, 1.0)
    view_spread = view_spread.detach().float().reshape(-1).clamp(0.0, 1.0)
    support_scale = float(max(bank_size - minimum_support, 1))
    support_score = ((support - float(minimum_support)) / support_scale).clamp(0.0, 1.0)
    transmission_score = ((transmission - float(transmission_floor)) / max(1.0 - float(transmission_floor), 1e-6)).clamp(0.0, 1.0)
    depth_score = torch.exp(-depth_variance / float(depth_temperature)).clamp(0.0, 1.0)
    spread_score = (view_spread / float(view_spread_scale)).clamp(0.0, 1.0)
    confidence = support_score * transmission_score * depth_score * spread_score
    confidence = torch.where(support >= float(minimum_support), confidence, torch.zeros_like(confidence))
    target = gate_floor + (1.0 - gate_floor) * confidence
    components = {
        "support_score": support_score,
        "transmission_score": transmission_score,
        "depth_score": depth_score,
        "spread_score": spread_score,
        "confidence": confidence,
        "target": target,
    }
    return confidence.detach().view(-1, 1), {key: value.detach() for key, value in components.items()}
