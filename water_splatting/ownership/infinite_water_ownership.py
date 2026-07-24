"""Soft infinite-water ownership evidence for M2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


OwnershipMode = Literal["alpha_only", "alpha_depth", "alpha_depth_color"]


@dataclass
class InfiniteWaterOwnershipOutput:
    """Pixel-level ownership maps for infinite-water composition."""

    m_inf: Tensor
    m_inf_eff: Tensor
    alpha_evidence: Tensor
    depth_evidence: Tensor
    color_evidence: Tensor


def _normalize_depth(depth: Tensor, mode: Literal["max", "p95"]) -> Tensor:
    valid = torch.isfinite(depth) & (depth > 0)
    if not valid.any():
        return torch.zeros_like(depth)
    valid_depth = depth[valid]
    if mode == "p95":
        scale = torch.quantile(valid_depth, 0.95)
    elif mode == "max":
        scale = valid_depth.max()
    else:
        raise ValueError(f"Unknown depth normalize mode: {mode}")
    return (depth / scale.clamp_min(1e-6)).clamp(0.0, 2.0)


def compute_infinite_water_ownership(
    *,
    accumulation: Tensor,
    depth: Tensor,
    rgb_near: Tensor,
    b_inf: Tensor,
    mode: OwnershipMode,
    detach_evidence: bool,
    alpha_power: float,
    depth_mid: float,
    depth_temp: float,
    color_temp: float,
    depth_normalize_mode: Literal["max", "p95"],
    occupancy_limited: bool,
) -> InfiniteWaterOwnershipOutput:
    """Build soft infinite-water ownership from internal render evidence."""

    evidence_accumulation = accumulation.detach() if detach_evidence else accumulation
    evidence_depth = depth.detach() if detach_evidence else depth
    evidence_rgb = rgb_near.detach() if detach_evidence else rgb_near
    evidence_b_inf = b_inf.detach() if detach_evidence else b_inf

    alpha_evidence = (1.0 - evidence_accumulation).clamp(0.0, 1.0).pow(alpha_power)
    depth_norm = _normalize_depth(evidence_depth, depth_normalize_mode)
    depth_evidence = torch.sigmoid((depth_norm - depth_mid) / max(depth_temp, 1e-6))
    color_distance = torch.mean(torch.abs(evidence_rgb - evidence_b_inf), dim=-1, keepdim=True)
    color_evidence = torch.exp(-color_distance / max(color_temp, 1e-6)).clamp(0.0, 1.0)

    if mode == "alpha_only":
        m_inf = alpha_evidence
    elif mode == "alpha_depth":
        m_inf = alpha_evidence * depth_evidence
    elif mode == "alpha_depth_color":
        m_inf = alpha_evidence * depth_evidence * color_evidence
    else:
        raise ValueError(f"Unknown infinite_water_ownership_mode: {mode}")

    if occupancy_limited:
        occupancy_gate = (1.0 - evidence_accumulation).clamp(0.0, 1.0)
        m_inf_eff = m_inf * occupancy_gate
    else:
        m_inf_eff = m_inf

    return InfiniteWaterOwnershipOutput(
        m_inf=m_inf.clamp(0.0, 1.0),
        m_inf_eff=m_inf_eff.clamp(0.0, 1.0),
        alpha_evidence=alpha_evidence,
        depth_evidence=depth_evidence,
        color_evidence=color_evidence,
    )
