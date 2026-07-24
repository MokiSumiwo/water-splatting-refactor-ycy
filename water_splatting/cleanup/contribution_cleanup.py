"""M3 contribution-aware cleanup utilities.

The helper is intentionally tensor-only and side-effect-free so the model can run
diagnostic dry-runs before any destructive Gaussian pruning is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class GaussianCleanupStats:
    """Compact diagnostics for a cleanup decision at one refinement step."""

    step: int
    total_count: int
    candidate_count: int
    low_contribution_count: int
    opacity_gate_count: int
    alpha_gate_count: int
    depth_gate_count: int
    ownership_gate_count: int
    mean_contribution: float
    mean_opacity: float
    mean_depth: float
    mean_sampled_alpha: float
    mean_sampled_ownership: float
    dry_run: bool

    @property
    def candidate_fraction(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.candidate_count / self.total_count


def sample_pixel_map_at_gaussians(pixel_map: Tensor, xys: Tensor, radii: Tensor, height: int, width: int) -> Tensor:
    """Nearest-neighbor sample a [H, W, C] or [H, W] map at projected Gaussian centers."""

    if pixel_map.ndim == 3:
        values = pixel_map[..., 0]
    elif pixel_map.ndim == 2:
        values = pixel_map
    else:
        raise ValueError(f"Expected a 2D or 3D pixel map, got shape {tuple(pixel_map.shape)}")

    sampled = torch.zeros(xys.shape[0], device=xys.device, dtype=values.dtype)
    visible = radii.reshape(-1) > 0
    if not visible.any():
        return sampled

    xy_visible = xys.detach()[visible]
    xi = torch.round(xy_visible[:, 0]).long().clamp(0, width - 1)
    yi = torch.round(xy_visible[:, 1]).long().clamp(0, height - 1)
    sampled[visible] = values.detach()[yi, xi]
    return sampled


def build_cleanup_candidate_mask(
    *,
    step: int,
    opacities: Tensor,
    contribution: Tensor,
    visibility: Tensor,
    avg_depth: Tensor,
    sampled_alpha: Optional[Tensor],
    sampled_ownership: Optional[Tensor],
    min_visibility: int,
    contribution_threshold: float,
    opacity_threshold: float,
    alpha_threshold: float,
    depth_threshold: float,
    ownership_threshold: float,
    require_alpha_gate: bool,
    require_depth_gate: bool,
    require_ownership_gate: bool,
    dry_run: bool,
) -> tuple[Tensor, GaussianCleanupStats]:
    """Build a conservative candidate mask from low contribution plus water gates."""

    total = opacities.shape[0]
    device = opacities.device
    opacity = torch.sigmoid(opacities).reshape(-1)
    contribution_flat = contribution.reshape(-1)
    visibility_flat = visibility.reshape(-1)
    avg_depth_flat = avg_depth.reshape(-1)

    sampled_alpha_flat = (
        sampled_alpha.reshape(-1).to(device=device, dtype=opacity.dtype)
        if sampled_alpha is not None
        else torch.ones(total, device=device, dtype=opacity.dtype)
    )
    sampled_ownership_flat = (
        sampled_ownership.reshape(-1).to(device=device, dtype=opacity.dtype)
        if sampled_ownership is not None
        else torch.zeros(total, device=device, dtype=opacity.dtype)
    )

    visible_enough = visibility_flat >= max(min_visibility, 1)
    low_contribution = contribution_flat <= max(contribution_threshold, 0.0)
    opacity_gate = opacity <= max(opacity_threshold, 0.0)
    alpha_gate = sampled_alpha_flat <= alpha_threshold
    depth_gate = avg_depth_flat >= depth_threshold
    ownership_gate = sampled_ownership_flat >= ownership_threshold

    candidate = visible_enough & low_contribution & opacity_gate
    if require_alpha_gate:
        candidate = candidate & alpha_gate
    if require_depth_gate:
        candidate = candidate & depth_gate
    if require_ownership_gate:
        candidate = candidate & ownership_gate

    finite_depth = torch.isfinite(avg_depth_flat)
    depth_for_mean = avg_depth_flat[finite_depth] if finite_depth.any() else avg_depth_flat
    stats = GaussianCleanupStats(
        step=step,
        total_count=int(total),
        candidate_count=int(candidate.sum().item()),
        low_contribution_count=int((visible_enough & low_contribution).sum().item()),
        opacity_gate_count=int((visible_enough & opacity_gate).sum().item()),
        alpha_gate_count=int((visible_enough & alpha_gate).sum().item()),
        depth_gate_count=int((visible_enough & depth_gate).sum().item()),
        ownership_gate_count=int((visible_enough & ownership_gate).sum().item()),
        mean_contribution=float(contribution_flat.mean().item()) if total else 0.0,
        mean_opacity=float(opacity.mean().item()) if total else 0.0,
        mean_depth=float(depth_for_mean.mean().item()) if total else 0.0,
        mean_sampled_alpha=float(sampled_alpha_flat.mean().item()) if total else 0.0,
        mean_sampled_ownership=float(sampled_ownership_flat.mean().item()) if total else 0.0,
        dry_run=dry_run,
    )
    return candidate, stats


def format_cleanup_stats(stats: GaussianCleanupStats) -> str:
    """Console-friendly one-line summary."""

    mode = "dry-run" if stats.dry_run else "active"
    return (
        f"M3 cleanup {mode} step={stats.step} "
        f"candidates={stats.candidate_count}/{stats.total_count} "
        f"({stats.candidate_fraction:.6f}), "
        f"low_contrib={stats.low_contribution_count}, "
        f"opacity_gate={stats.opacity_gate_count}, "
        f"alpha_gate={stats.alpha_gate_count}, "
        f"depth_gate={stats.depth_gate_count}, "
        f"ownership_gate={stats.ownership_gate_count}, "
        f"mean_contrib={stats.mean_contribution:.6e}, "
        f"mean_opacity={stats.mean_opacity:.6f}, "
        f"mean_depth={stats.mean_depth:.6f}, "
        f"mean_alpha={stats.mean_sampled_alpha:.6f}, "
        f"mean_ownership={stats.mean_sampled_ownership:.6f}"
    )
