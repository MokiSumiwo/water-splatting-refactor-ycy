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
from .mcgr import (
    MCGREvidence,
    build_mcgr_detail_residual,
    build_mcgr_gaussian_evidence,
    build_mcgr_persistent_map,
    load_mcgr_correspondence_payload,
    select_mcgr_candidates,
    update_mcgr_residual_bank,
)

__all__ = [
    "MCGREvidence",
    "MVGAREvidence",
    "build_mcgr_detail_residual",
    "build_mcgr_gaussian_evidence",
    "build_mcgr_persistent_map",
    "build_mvgar_detail_map",
    "build_mvgar_surface_evidence",
    "load_mcgr_correspondence_payload",
    "load_mvgar_view_payload",
    "mvgar_surface_anchor_loss",
    "select_mcgr_candidates",
    "select_mvgar_candidates",
    "tensor_stats",
    "update_mcgr_residual_bank",
]
