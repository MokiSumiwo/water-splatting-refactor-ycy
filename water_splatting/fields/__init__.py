"""Field helpers for WaterSplatting."""

from .gaussian_appearance import (
    GaussianColorOutput,
    compute_bounded_gaussian_colors,
    compute_gaussian_colors,
    compute_gaussian_sh_residual,
)
from .medium_field import DirectionConditionedMediumField, MediumFieldOutput, get_medium_context_extra_dim

__all__ = [
    "DirectionConditionedMediumField",
    "GaussianColorOutput",
    "MediumFieldOutput",
    "compute_bounded_gaussian_colors",
    "compute_gaussian_colors",
    "compute_gaussian_sh_residual",
    "get_medium_context_extra_dim",
]
