"""Geometry helpers for WaterSplatting training modules."""

from .mvgar import (
    MVGAREvidence,
    build_mvgar_detail_map,
    build_mvgar_surface_evidence,
    load_mvgar_view_payload,
    mvgar_surface_anchor_loss,
    select_mvgar_candidates,
    tensor_stats,
)

__all__ = [
    "MVGAREvidence",
    "build_mvgar_detail_map",
    "build_mvgar_surface_evidence",
    "load_mvgar_view_payload",
    "mvgar_surface_anchor_loss",
    "select_mvgar_candidates",
    "tensor_stats",
]
