"""Closed-form local medium Jacobian actions for the CUDA forward compositor."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from water_splatting.utils import bin_and_sort_gaussians, compute_cumulative_intersects


def analytic_medium_jacobian_actions(
    *,
    xys: Tensor,
    depths: Tensor,
    radii: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    num_tiles_hit: Tensor,
    height: int,
    width: int,
    block_width: int,
    raw_medium: Tensor,
    raw_directions: Tensor,
    density_bias: float,
    chunk_size: int = 4096,
    pixel_indices: Optional[Tensor] = None,
) -> Tensor:
    """Evaluate ``J_p v_i`` for every pixel and every raw-medium direction.

    ``raw_directions`` has shape ``[modes, 9]`` and is interpreted in raw
    medium coordinates.  The implementation mirrors the checked CUDA forward
    compositor, including its alpha cutoff, stop condition, object attenuation,
    finite medium intervals, and terminal medium tail.  Geometry and branch
    decisions are detached, as required for a first-order capacity-control
    signal.  No autograd derivative is constructed.
    """

    raw = raw_medium.detach().float().reshape(-1, 9)
    directions = raw_directions.detach().float().reshape(-1, 9).to(device=raw.device)
    if int(chunk_size) < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if raw.shape[0] != height * width:
        raise ValueError(f"raw_medium has {raw.shape[0]} pixels, expected {height * width}")
    if directions.shape[1] != 9 or directions.shape[0] == 0:
        raise ValueError("raw_directions must have shape [modes, 9]")

    medium_rgb = torch.sigmoid(raw[:, :3])
    medium_bs = F.softplus(raw[:, 3:6] + float(density_bias))
    medium_attn = F.softplus(raw[:, 6:9] + float(density_bias))
    d_rgb = medium_rgb * (1.0 - medium_rgb)
    d_bs = torch.sigmoid(raw[:, 3:6] + float(density_bias))
    d_attn = torch.sigmoid(raw[:, 6:9] + float(density_bias))

    num_intersects, cumulative = compute_cumulative_intersects(num_tiles_hit.detach())
    if num_intersects < 1:
        return (d_rgb[:, None, :] * directions[None, :, :3]).detach()

    tile_bounds = (
        (int(width) + int(block_width) - 1) // int(block_width),
        (int(height) + int(block_width) - 1) // int(block_width),
        1,
    )
    _isect_unsorted, _ids_unsorted, _isect_sorted, gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(
        xys.shape[0],
        num_intersects,
        xys.detach(),
        depths.detach(),
        radii.detach(),
        cumulative,
        tile_bounds,
        int(block_width),
    )

    if pixel_indices is None:
        pixel_indices = torch.arange(height * width, device=raw.device, dtype=torch.long)
    else:
        pixel_indices = pixel_indices.to(device=raw.device, dtype=torch.long).reshape(-1)
        if bool((pixel_indices < 0).any() or (pixel_indices >= height * width).any()):
            raise ValueError("pixel_indices contains an out-of-range pixel")
    action_parts = []
    for start in range(0, pixel_indices.numel(), int(chunk_size)):
        flat = pixel_indices[start : start + int(chunk_size)]
        rows = torch.div(flat, int(width), rounding_mode="floor")
        cols = flat.remainder(int(width))
        tile_ids = (rows // int(block_width)) * tile_bounds[0] + cols // int(block_width)
        starts = tile_bins[tile_ids, 0].long()
        lengths = (tile_bins[tile_ids, 1] - tile_bins[tile_ids, 0]).long()
        max_length = int(lengths.max().item()) if lengths.numel() else 0
        if max_length == 0:
            action_parts.append(d_rgb[flat, None, :] * directions[None, :, :3])
            continue

        offsets = torch.arange(max_length, device=raw.device, dtype=torch.long)
        present = offsets[None, :] < lengths[:, None]
        gaussian_indices = (starts[:, None] + offsets[None, :]).clamp_max(len(gaussian_ids_sorted) - 1)
        gaussian_ids = gaussian_ids_sorted[gaussian_indices].long()
        center = xys.detach()[gaussian_ids]
        conic = conics.detach()[gaussian_ids]
        depth = depths.detach()[gaussian_ids]
        color = colors.detach()[gaussian_ids]
        opacity = opacities.detach()[gaussian_ids].reshape(-1, max_length)
        opacity = opacity
        delta_x = center[..., 0] - cols.float()[:, None]
        delta_y = center[..., 1] - rows.float()[:, None]
        sigma = 0.5 * (conic[..., 0] * delta_x.square() + conic[..., 2] * delta_y.square())
        sigma = sigma + conic[..., 1] * delta_x * delta_y
        alpha = torch.minimum(torch.full_like(sigma, 0.999), opacity * torch.exp(-sigma))

        min_attn = torch.minimum(torch.zeros_like(medium_attn[flat, 0]), medium_attn[flat].min(dim=-1).values)
        valid = present & (sigma >= 0.0) & (
            alpha * torch.exp(-min_attn[:, None] * depth) >= 1.0 / 255.0
        )
        factor = torch.where(valid, 1.0 - alpha, torch.ones_like(alpha))
        trans_before = torch.cat(
            [
                torch.ones((flat.numel(), 1), device=raw.device),
                torch.cumprod(factor[:, :-1], dim=-1),
            ],
            dim=-1,
        )
        stop = valid & (trans_before * (1.0 - alpha) <= 1e-4)
        prior_stop = torch.cat(
            [
                torch.zeros((flat.numel(), 1), device=raw.device, dtype=torch.bool),
                torch.cumsum(stop[:, :-1].to(torch.int32), dim=-1) > 0,
            ],
            dim=-1,
        )
        contributes = valid & ~prior_stop & ~stop
        contributes_f = contributes.to(dtype=raw.dtype)
        trans_factor = torch.where(contributes, 1.0 - alpha, torch.ones_like(alpha))
        trans_final = torch.cumprod(trans_factor, dim=-1)[:, -1]
        depth_for_prev = torch.where(contributes, depth, torch.zeros_like(depth))
        prev_before = torch.cat(
            [
                torch.zeros((flat.numel(), 1), device=raw.device),
                torch.cummax(depth_for_prev[:, :-1], dim=-1).values,
            ],
            dim=-1,
        )
        prev_final = torch.cummax(depth_for_prev, dim=-1).values[:, -1]
        vis = alpha * trans_before
        exp_attn = torch.exp(-medium_attn[flat, None, :] * depth[..., None])
        exp_bs_prev = torch.exp(-medium_bs[flat, None, :] * prev_before[..., None])
        exp_bs_depth = torch.exp(-medium_bs[flat, None, :] * depth[..., None])
        medium_factor = (
            contributes_f[..., None]
            * trans_before[..., None]
            * (exp_bs_prev - exp_bs_depth)
        ).sum(dim=1)
        medium_factor = medium_factor + trans_final[:, None] * torch.exp(-medium_bs[flat] * prev_final[:, None])
        object_attn_derivative = (
            -contributes_f[..., None]
            * vis[..., None]
            * color
            * exp_attn
            * depth[..., None]
        ).sum(dim=1)
        medium_bs_derivative = (
            contributes_f[..., None]
            * trans_before[..., None]
            * (-prev_before[..., None] * exp_bs_prev + depth[..., None] * exp_bs_depth)
        ).sum(dim=1)
        medium_bs_derivative = medium_bs_derivative - (
            trans_final[:, None] * prev_final[:, None] * torch.exp(-medium_bs[flat] * prev_final[:, None])
        )

        rgb_term = medium_factor[:, None, :] * d_rgb[flat, None, :] * directions[None, :, :3]
        bs_term = (
            medium_rgb[flat, None, :]
            * medium_bs_derivative[:, None, :]
            * d_bs[flat, None, :]
            * directions[None, :, 3:6]
        )
        attn_term = (
            object_attn_derivative[:, None, :]
            * d_attn[flat, None, :]
            * directions[None, :, 6:9]
        )
        effect = rgb_term + bs_term + attn_term
        if not bool(torch.isfinite(effect).all().item()):
            raise RuntimeError("Non-finite analytic medium Jacobian action")
        action_parts.append(effect)

    return torch.cat(action_parts, dim=0).detach()
