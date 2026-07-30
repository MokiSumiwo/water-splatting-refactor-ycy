"""Scene-medium attribution helpers."""

from .medium_explainability import (
    MediumExplainabilitySupport,
    accumulation_clearance_amplifier,
    budgeted_capacity_loss,
    build_residual_gated_halo_support,
    build_route_capacity_support,
    build_training_routed_prediction,
    clear_proxy_chroma_loss,
    clear_proxy_luma_budget_loss,
    compute_far_depth_support,
    compute_image_structure_support,
    compute_medium_explainability,
    core_zero_capacity_loss,
    rgb_luma_budget_loss,
    support_coverage_stats,
    weighted_rgb_l1,
)

__all__ = [
    "MediumExplainabilitySupport",
    "accumulation_clearance_amplifier",
    "budgeted_capacity_loss",
    "build_residual_gated_halo_support",
    "build_route_capacity_support",
    "build_training_routed_prediction",
    "clear_proxy_chroma_loss",
    "clear_proxy_luma_budget_loss",
    "compute_far_depth_support",
    "compute_image_structure_support",
    "compute_medium_explainability",
    "core_zero_capacity_loss",
    "rgb_luma_budget_loss",
    "support_coverage_stats",
    "weighted_rgb_l1",
]
