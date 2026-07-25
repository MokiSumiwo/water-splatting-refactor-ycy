"""Field helpers for WaterSplatting."""

from .gaussian_appearance import (
    DualColorOutput,
    compute_dual_gaussian_colors,
    compute_gaussian_colors,
    compute_gaussian_sh_residual,
)
from .medium_field import DirectionConditionedMediumField, MediumFieldOutput, get_medium_context_extra_dim

__all__ = [
    "DualColorOutput",
    "DirectionConditionedMediumField",
    "MediumFieldOutput",
    "compute_dual_gaussian_colors",
    "compute_gaussian_colors",
    "compute_gaussian_sh_residual",
    "get_medium_context_extra_dim",
]
