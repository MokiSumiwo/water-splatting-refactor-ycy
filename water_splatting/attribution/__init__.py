"""Scene-medium attribution helpers."""

from .medium_explainability import (
    MediumExplainabilitySupport,
    budgeted_capacity_loss,
    build_route_capacity_support,
    build_training_routed_prediction,
    clear_proxy_chroma_loss,
    clear_proxy_luma_budget_loss,
    compute_far_depth_support,
    compute_image_structure_support,
    compute_medium_explainability,
    support_coverage_stats,
    weighted_rgb_l1,
)

__all__ = [
    "MediumExplainabilitySupport",
    "budgeted_capacity_loss",
    "build_route_capacity_support",
    "build_training_routed_prediction",
    "clear_proxy_chroma_loss",
    "clear_proxy_luma_budget_loss",
    "compute_far_depth_support",
    "compute_image_structure_support",
    "compute_medium_explainability",
    "support_coverage_stats",
    "weighted_rgb_l1",
]
