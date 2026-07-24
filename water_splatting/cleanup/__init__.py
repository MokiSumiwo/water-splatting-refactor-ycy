"""Contribution-aware Gaussian cleanup diagnostics."""

from .contribution_cleanup import (
    GaussianCleanupStats,
    build_cleanup_candidate_mask,
    format_cleanup_stats,
    sample_pixel_map_at_gaussians,
)

__all__ = [
    "GaussianCleanupStats",
    "build_cleanup_candidate_mask",
    "format_cleanup_stats",
    "sample_pixel_map_at_gaussians",
]
