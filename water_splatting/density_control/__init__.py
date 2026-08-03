"""Density-control helpers for WaterSplatting experiments."""

from .mcmc_diagnostics import (
    effective_gaussian_count,
    mcmc_tensor_stats,
    parent_sampling_entropy,
)
from .mcmc_relocation import (
    alpha_after_coincident_split,
    alpha_to_logit,
    logit_to_alpha,
    relocation_alpha_and_scale,
    relocation_logits_and_log_scales,
    relocated_child_opacity_logits,
    split_parent_opacity_logits,
)

__all__ = [
    "alpha_after_coincident_split",
    "alpha_to_logit",
    "effective_gaussian_count",
    "logit_to_alpha",
    "mcmc_tensor_stats",
    "parent_sampling_entropy",
    "relocation_alpha_and_scale",
    "relocation_logits_and_log_scales",
    "relocated_child_opacity_logits",
    "split_parent_opacity_logits",
]
