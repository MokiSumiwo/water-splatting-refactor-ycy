"""Background-attribution losses for high-precision water masks."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_rgb_l1_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean per-channel L1 loss over a single-channel pixel mask."""

    if mask.ndim == 2:
        mask = mask[..., None]
    mask = mask.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    if mask.shape[:2] != pred.shape[:2]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match pred shape {tuple(pred.shape)}")
    denom = mask.sum().clamp_min(1e-6) * float(pred.shape[-1])
    return (mask * torch.abs(pred - target)).sum() / denom


def effective_background_mask(
    *,
    water_mask: Tensor,
    boundary_mask: Tensor | None = None,
    hit_confidence: Tensor | None = None,
    hit_threshold: float = -1.0,
) -> Tensor:
    """Build a conservative background mask for clear-Gaussian suppression."""

    if water_mask.ndim == 2:
        water_mask = water_mask[..., None]
    mask = water_mask.float().clamp(0.0, 1.0)

    if boundary_mask is not None:
        if boundary_mask.ndim == 2:
            boundary_mask = boundary_mask[..., None]
        mask = mask * (1.0 - boundary_mask.to(device=mask.device, dtype=mask.dtype).clamp(0.0, 1.0))

    if hit_confidence is not None and hit_threshold >= 0.0:
        if hit_confidence.ndim == 2:
            hit_confidence = hit_confidence[..., None]
        hit_gate = (hit_confidence.to(device=mask.device, dtype=mask.dtype) < float(hit_threshold)).float()
        mask = mask * hit_gate

    return mask.clamp(0.0, 1.0)
