"""Shared types for GMVC diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class GMVCTrackConfig:
    """Configuration for offline GMVC surface-track diagnostics."""

    min_views: int = 3
    alpha_threshold: float = 0.95
    depth_rel_threshold: float = 0.02
    depth_std_rel_threshold: float = 0.25
    relative_depth_span: float = 0.05
    transmission_min: float = 0.10
    span_weight_high: float = 0.10
    depth_error_sigma: float = 0.01
    eps: float = 1e-4
    j_clamp_min: float = -0.25
    j_clamp_max: float = 1.25
    edge_margin: int = 8
    samples_per_view: int = 4096
    seed: int = 42
    target_neighbor_window: int = 0
    geometry_only_bank: bool = False
    signal_min: float = 0.02
    signal_max: float = 0.98
    signal_softness: float = 0.05


@dataclass
class GMVCView:
    """Rendered view data needed to build geometry-anchored tracks."""

    image_index: int
    camera_index: int
    camera_to_world: torch.Tensor
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    gt: torch.Tensor
    depth: torch.Tensor
    accumulation: torch.Tensor
    depth_std_relative: torch.Tensor
    medium_bs: torch.Tensor
    medium_attn: torch.Tensor
    b_inf: torch.Tensor
    transmission: torch.Tensor
    backscatter_endpoint: torch.Tensor
    actual_rgb_medium: torch.Tensor

    def metadata(self) -> Dict[str, Any]:
        return {
            "image_index": self.image_index,
            "camera_index": self.camera_index,
            "height": self.height,
            "width": self.width,
        }
