#!/usr/bin/env python3
"""Read-only alignment audit for the frozen selected medium mode on IUI3."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import validate_heldout_single_mode_camera_utility_iui3 as PHASEA
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC


EXPERIMENT = "SELECTED-MODE-UTILITY-RECONSTRUCTION-ALIGNMENT-AUDIT"
SCENE = "IUI3-RedSea"
PREVIOUS_OUTPUT_DIR = Path("outputs/heldout_single_mode_camera_utility_iui3_20260826")
SOURCE_OUTPUT_DIR = Path("outputs/m1_ocmc_causal_iui3_20260825")
OUTPUT_DIR = Path("outputs/selected_mode_utility_reconstruction_alignment_iui3_20260826")
LOG_DIR = Path("logs/selected_mode_utility_reconstruction_alignment_iui3_20260826")
RESEARCH_NOTE = Path("research_notes/SELECTED_MODE_UTILITY_RECONSTRUCTION_ALIGNMENT_IUI3_2026-08-26.md")
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
EPS = 1e-12
ZERO_THRESHOLDS = (1e-12, 1e-10, 1e-8)
EXPECTED_CLASSIFICATION = "SINGLE_MODE_CONTEXT_UTILITY_TENTATIVE"
EXPECTED_MODE_LABEL = "mode_01"
EXPECTED_MODE_INDEX = 1
EXPECTED_SIGMA = 0.01034344732761383
EXPECTED_G_OBS = 0.2694622658372983
EXPECTED_BANK_HASH = "e23a146c5d34685605ab7f7a1845408fa0460e2d583310650b3d10279fa323d2"
EXPECTED_SWAP_HASH = "0ad944cc60548e712e420083e361df1332b8781c8dc54f9265c2204df8f87d5a"
EXPECTED_CHECKPOINT_SHA256 = "84cb33f31de6d8af4bbb8df2e6c3619a8257e21f93bb74cb6241735652fed997"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_manifest(repo: Path) -> Dict[str, Any]:
    return {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "status_short": _git(repo, "status", "--short"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "diff_stat": _git(repo, "diff", "--stat"),
        "diff_check": _git(repo, "diff", "--check"),
        "historical_untracked_files_preserved_not_modified_or_committed": [
            "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
            "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
        ],
    }


def _assert_runtime_policy() -> Dict[str, Any]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "water_splatting":
        raise RuntimeError(f"CONDA_DEFAULT_ENV must be water_splatting, got {conda_env!r}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [token.strip() for token in visible.split(",") if token.strip()]
    if len(devices) != 1 or devices[0] not in ALLOWED_PHYSICAL_GPUS:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be exactly one physical GPU in "
            f"{sorted(ALLOWED_PHYSICAL_GPUS)}; got {visible!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available for this diagnostic.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one torch-visible GPU after masking, got {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "physical_gpu_id": visible,
        "torch_logical_gpu_id": 0,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
    }


def _environment_manifest(gpu: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "CONDA_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": dict(gpu),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _median(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return float(vals[mid])
    return float(0.5 * (vals[mid - 1] + vals[mid]))


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) <= 1:
        return 0.0 if vals else float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.std(ddof=0))


def _fraction_positive(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    return float((arr > 0.0).mean())


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xv = np.asarray([float(v) for v in x], dtype=np.float64)
    yv = np.asarray([float(v) for v in y], dtype=np.float64)
    if xv.size < 2 or yv.size < 2:
        return float("nan")
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return float("nan")
    return float(spearmanr(xv, yv).correlation)


def _safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    xv = np.asarray([float(v) for v in x], dtype=np.float64)
    yv = np.asarray([float(v) for v in y], dtype=np.float64)
    if xv.size < 2 or yv.size < 2:
        return float("nan")
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return float("nan")
    return float(pearsonr(xv, yv).statistic)


def _row_group_stats(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    values = [float(row[key]) for row in rows if key in row]
    positive = [v for v in values if v > 0.0]
    negative = [v for v in values if v < 0.0]
    return {
        "count": int(len(values)),
        "mean": _mean(values),
        "median": _median(values),
        "std": _std(values),
        "fraction_positive": _fraction_positive(values),
        "sum_positive": float(sum(positive)) if positive else 0.0,
        "sum_negative": float(sum(negative)) if negative else 0.0,
    }


def _top_positive_share(values: Sequence[float], fractions: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if float(v) > 0.0], dtype=np.float64)
    if arr.size == 0:
        return {f"top_{int(frac * 100)}pct_share": float("nan") for frac in fractions}
    arr = np.sort(arr)[::-1]
    total = float(arr.sum())
    out: Dict[str, float] = {}
    for frac in fractions:
        k = max(1, int(math.ceil(arr.size * float(frac))))
        out[f"top_{int(frac * 100)}pct_share"] = float(arr[:k].sum() / max(total, EPS))
    return out


def _participation_ratio(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if float(v) > 0.0], dtype=np.float64)
    if arr.size == 0:
        return {"positive_count": 0, "effective_number": float("nan"), "participation_ratio": float("nan")}
    total = float(arr.sum())
    effective = float(total * total / max(float(np.square(arr).sum()), EPS))
    return {
        "positive_count": int(arr.size),
        "effective_number": effective,
        "participation_ratio": float(effective / max(arr.size, 1)),
    }


def _camera_context(model: Any, camera: Any) -> torch.Tensor:
    return PHASEA._camera_context(model, camera).detach()


@torch.no_grad()
def _load_branch(repo: Path, source_output_dir: Path) -> Tuple[Any, Mapping[str, Any]]:
    return PHASEA._load_c1_branch(repo, source_output_dir)


@torch.no_grad()
def _train_records(branch: Any) -> Dict[str, Tuple[int, Any, Dict[str, Any]]]:
    return PHASEA._camera_records(branch)[0]


@torch.no_grad()
def _eval_records(branch: Any) -> Dict[str, Tuple[int, Any, Dict[str, Any]]]:
    return PHASEA._camera_records(branch)[1]


def _selected_mode_projection(delta_raw: torch.Tensor, basis_vector: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = scale.reshape(1, 9).to(device=delta_raw.device, dtype=delta_raw.dtype).clamp_min(EPS)
    basis = basis_vector.reshape(9).to(device=delta_raw.device, dtype=delta_raw.dtype)
    delta_std = delta_raw.reshape(-1, 9) / scale
    coeff = delta_std @ basis
    removed_std = delta_std - coeff[:, None] * basis[None, :]
    removed_raw = (removed_std * scale).reshape_as(delta_raw)
    return delta_std, coeff, removed_raw


@torch.no_grad()
def _render_train_view(model: Any, camera: Any, projector_bundle: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    raw_base, raw_natural, raw_effective, height, width = PHASEA._natural_and_effective_raw(
        model,
        camera,
        projector_bundle,
        camera_context_override=None,
    )
    out_full = PHASEA._render_from_raw(model, camera, raw_effective, height, width)
    return {
        "raw_base": raw_base,
        "raw_natural": raw_natural,
        "raw_effective": raw_effective,
        "out_full": out_full,
        "height": height,
        "width": width,
    }


@torch.no_grad()
def _render_eval_view(model: Any, camera: Any, projector_bundle: Mapping[str, Any], basis_vector: torch.Tensor, scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    raw_base, raw_natural, raw_effective, height, width = PHASEA._natural_and_effective_raw(
        model,
        camera,
        projector_bundle,
        camera_context_override=None,
    )
    raw_minus = raw_base + _selected_mode_projection(raw_effective - raw_base, basis_vector, scale)[2]
    out_full = PHASEA._render_from_raw(model, camera, raw_effective, height, width)
    out_minus = PHASEA._render_from_raw(model, camera, raw_minus, height, width)
    return {
        "raw_base": raw_base,
        "raw_natural": raw_natural,
        "raw_effective": raw_effective,
        "raw_minus": raw_minus,
        "out_full": out_full,
        "out_minus": out_minus,
        "height": height,
        "width": width,
    }


@torch.no_grad()
def _swap_mode_coefficients(
    model: Any,
    camera: Any,
    projector_bundle: Mapping[str, Any],
    swap_context: torch.Tensor,
    flat_indices: torch.Tensor,
    basis_vector: torch.Tensor,
    scale: torch.Tensor,
) -> np.ndarray:
    raw_base, raw_natural, _raw_effective, _height, _width = PHASEA._natural_and_effective_raw(
        model,
        camera,
        projector_bundle,
        camera_context_override=swap_context,
    )
    _std, coeff, _removed = _selected_mode_projection(raw_natural - raw_base, basis_vector, scale)
    values = coeff[flat_indices.to(model.device)].detach().float().cpu().numpy()
    del raw_base, raw_natural, _raw_effective, _std, coeff, _removed
    return values


def _load_previous_assets(previous_output_dir: Path) -> Dict[str, Any]:
    selection_mode = _load_json(previous_output_dir / "selection_mode.json")
    phase_a = _load_json(previous_output_dir / "phase_a_classification.json")
    ray_bank = _load_json(previous_output_dir / "heldout_ray_bank.json")
    swap_bank = _load_json(previous_output_dir / "heldout_swap_bank.json")
    overlap_report = _load_json(previous_output_dir / "heldout_overlap_report.json")
    selected_rows = _load_json(previous_output_dir / "heldout_selected_mode_removal.json")
    msafe_rows = _load_json(previous_output_dir / "heldout_msafe_replication.json")
    eval_rows = _load_json(previous_output_dir / "heldout_eval_replication.json")
    checkpoint_manifest = _load_json(previous_output_dir / "checkpoint_manifest.json")
    selection_set = _load_json(previous_output_dir / "selection_set_summary.json")
    if ray_bank["rows_hash"] != EXPECTED_BANK_HASH:
        raise RuntimeError(f"Frozen held-out bank hash mismatch: {ray_bank['rows_hash']}")
    if swap_bank["hash"] != EXPECTED_SWAP_HASH:
        raise RuntimeError(f"Frozen swap bank hash mismatch: {swap_bank['hash']}")
    if selection_mode["mode_label"] != EXPECTED_MODE_LABEL or int(selection_mode["mode_index"]) != EXPECTED_MODE_INDEX:
        raise RuntimeError("Frozen selected mode mismatch.")
    if abs(float(selection_mode["sigma"]) - EXPECTED_SIGMA) > 1e-12:
        raise RuntimeError("Frozen selected mode sigma mismatch.")
    if abs(float(selection_mode["g_obs"]) - EXPECTED_G_OBS) > 1e-12:
        raise RuntimeError("Frozen selected mode g_obs mismatch.")
    if phase_a["classification"] != EXPECTED_CLASSIFICATION:
        raise RuntimeError(f"Unexpected previous Phase A classification: {phase_a['classification']}")
    if int(checkpoint_manifest["loaded_step"]) != 14999:
        raise RuntimeError(f"Unexpected stored checkpoint step: {checkpoint_manifest['loaded_step']}")
    return {
        "selection_mode": selection_mode,
        "phase_a": phase_a,
        "ray_bank": ray_bank,
        "swap_bank": swap_bank,
        "overlap_report": overlap_report,
        "selected_rows": selected_rows["rows"],
        "msafe_rows": msafe_rows["rows"],
        "eval_rows": eval_rows["rows"],
        "checkpoint_manifest": checkpoint_manifest,
        "selection_set": selection_set,
    }


def _frozen_mode_provenance(previous: Mapping[str, Any]) -> Dict[str, Any]:
    selection_mode = previous["selection_mode"]
    phase_a = previous["phase_a"]
    expected = phase_a["selected_mode"]
    sigma_diff = abs(float(selection_mode["sigma"]) - float(expected["sigma"]))
    g_obs_diff = abs(float(selection_mode["g_obs"]) - float(expected["g_obs"]))
    vector_diff = max(
        abs(float(a) - float(b))
        for a, b in zip(selection_mode["mode_vector"], expected["mode_vector"])
    )
    scale_diff = max(
        abs(float(a) - float(b))
        for a, b in zip(selection_mode["scale"], previous["selection_set"]["scale"])
    )
    frozen_match = (
        selection_mode["mode_index"] == expected["mode_index"]
        and selection_mode["mode_label"] == expected["mode_label"]
        and sigma_diff <= 1e-12
        and g_obs_diff <= 1e-12
        and vector_diff <= 1e-12
        and scale_diff <= 1e-12
    )
    checkpoint_path = Path(selection_mode["checkpoint_path"])
    actual_checkpoint_sha256 = _sha256_file(checkpoint_path)
    if actual_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Primary checkpoint hash mismatch: {actual_checkpoint_sha256}")
    return {
        "frozen_mode_match": bool(frozen_match),
        "selection_mode_index": int(selection_mode["mode_index"]),
        "selection_mode_label": selection_mode["mode_label"],
        "sigma": float(selection_mode["sigma"]),
        "sigma_rank": int(selection_mode["sigma_rank"]),
        "g_obs": float(selection_mode["g_obs"]),
        "mode_vector": [float(v) for v in selection_mode["mode_vector"]],
        "scale": [float(v) for v in selection_mode["scale"]],
        "checkpoint_path": selection_mode["checkpoint_path"],
        "checkpoint_refresh_step": int(selection_mode["checkpoint_refresh_step"]),
        "basis_source": selection_mode["basis_source"],
        "sigma_diff_vs_previous": sigma_diff,
        "g_obs_diff_vs_previous": g_obs_diff,
        "mode_vector_max_abs_diff_vs_previous": vector_diff,
        "scale_max_abs_diff_vs_previous": scale_diff,
        "checkpoint_sha256": actual_checkpoint_sha256,
        "stored_step": int(previous["checkpoint_manifest"]["loaded_step"]),
        "checkpoint_hash_match": bool(actual_checkpoint_sha256 == previous["checkpoint_manifest"]["checkpoint_sha256"]),
        "previous_classification": phase_a["classification"],
    }


def _frozen_bank_provenance(previous: Mapping[str, Any]) -> Dict[str, Any]:
    ray_bank = previous["ray_bank"]
    swap_bank = previous["swap_bank"]
    overlap_report = previous["overlap_report"]
    return {
        "scene": ray_bank["scene"],
        "train_view_count": int(ray_bank["train_view_count"]),
        "sample_per_view": int(ray_bank["samples_per_view"]),
        "selection_seed": int(ray_bank["selection_seed"]),
        "heldout_seed": int(ray_bank["heldout_seed"]),
        "selection_total_rays": int(ray_bank["selection_total_rays"]),
        "heldout_total_rays": int(ray_bank["heldout_total_rays"]),
        "selection_overlap_count": int(overlap_report["pixel_overlap_count"]),
        "selection_overlap_fraction": float(overlap_report["pixel_overlap_fraction"]),
        "strict_zero_overlap": bool(overlap_report["strict_zero_overlap"]),
        "bank_hash": ray_bank["rows_hash"],
        "swap_count_per_source": int(swap_bank["alternatives_per_source"]),
        "swap_bank_seed": int(swap_bank["seed"]),
        "swap_bank_hash": swap_bank["hash"],
        "swap_bank_rows": swap_bank["rows"],
        "selection_train_views": list(ray_bank["train_views"]),
    }


def _augment_general_rows(
    rows: Sequence[Mapping[str, Any]],
    ray_bank: Mapping[str, Any],
    train_records: Mapping[str, Tuple[int, Any, Dict[str, Any]]],
    depth_maps: Mapping[str, np.ndarray],
    tau_maps: Mapping[str, np.ndarray],
    transmission_maps: Mapping[str, np.ndarray],
    swap_coefficients: Mapping[str, Sequence[np.ndarray]],
) -> List[Dict[str, Any]]:
    bank_lookup = {row["view_id"]: row for row in ray_bank["rows"]}
    out: List[Dict[str, Any]] = []
    for row in rows:
        source_view = str(row["source_view_id"])
        bank_row = bank_lookup[source_view]
        height = int(bank_row["height"])
        width = int(bank_row["width"])
        flat_index = int(row["flat_index"])
        pixel_y = int(flat_index // width)
        pixel_x = int(flat_index % width)
        a_correct = float(row["A_natural_coeff"])
        A = abs(a_correct)
        depth = float(depth_maps[source_view][flat_index])
        tau = float(tau_maps[source_view][flat_index])
        transmission = float(transmission_maps[source_view][flat_index])
        idx, _camera, _batch = train_records[source_view]
        out.append(
            {
                **dict(row),
                "view_index": int(idx),
                "height": height,
                "width": width,
                "pixel_y": pixel_y,
                "pixel_x": pixel_x,
                "camera_id": source_view,
                "a_correct": a_correct,
                "A": A,
                "depth": depth,
                "tau": tau,
                "transmission": transmission,
            }
        )
        ray_index = int(row["ray_index_within_bank"])
        for swap_index, coefficients in enumerate(swap_coefficients[source_view]):
            out[-1][f"a_swap_{swap_index:02d}"] = float(coefficients[ray_index])
    return out


def _quadrant(row: Mapping[str, Any]) -> str:
    cu = float(row["C_utility_heldout"])
    cr = float(row["C_rgb_heldout"])
    if cu == 0.0 and cr == 0.0:
        return "ZERO_BOTH"
    if cu == 0.0:
        return "ZERO_UTILITY"
    if cr == 0.0:
        return "ZERO_RGB"
    if cu > 0.0 and cr > 0.0:
        return "Q1"
    if cu > 0.0 and cr < 0.0:
        return "Q2"
    if cu < 0.0 and cr > 0.0:
        return "Q3"
    return "Q4"


def _zero_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cu = np.asarray([float(row["C_utility_heldout"]) for row in rows], dtype=np.float64)
    cr = np.asarray([float(row["C_rgb_heldout"]) for row in rows], dtype=np.float64)
    payload: Dict[str, Any] = {
        "count": int(cu.size),
        "C_utility_exact_zero_fraction": float((cu == 0.0).mean()),
        "C_utility_abs_lt_1e-12_fraction": float((np.abs(cu) < 1e-12).mean()),
        "C_utility_abs_lt_1e-10_fraction": float((np.abs(cu) < 1e-10).mean()),
        "C_utility_abs_lt_1e-8_fraction": float((np.abs(cu) < 1e-8).mean()),
        "C_rgb_exact_zero_fraction": float((cr == 0.0).mean()),
        "C_rgb_abs_lt_1e-12_fraction": float((np.abs(cr) < 1e-12).mean()),
        "C_rgb_abs_lt_1e-10_fraction": float((np.abs(cr) < 1e-10).mean()),
        "C_rgb_abs_lt_1e-8_fraction": float((np.abs(cr) < 1e-8).mean()),
        "both_zero_fraction": float(((cu == 0.0) & (cr == 0.0)).mean()),
        "both_near_zero_1e-12_fraction": float(((np.abs(cu) < 1e-12) & (np.abs(cr) < 1e-12)).mean()),
        "both_near_zero_1e-10_fraction": float(((np.abs(cu) < 1e-10) & (np.abs(cr) < 1e-10)).mean()),
        "both_near_zero_1e-8_fraction": float(((np.abs(cu) < 1e-8) & (np.abs(cr) < 1e-8)).mean()),
        "interpretation": "median_zero_is_driven_by_true_zero-effect rays and sparse support rather than floating-point noise",
    }
    return payload


def _global_alignment(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cu = np.asarray([float(row["C_utility_heldout"]) for row in rows], dtype=np.float64)
    cr = np.asarray([float(row["C_rgb_heldout"]) for row in rows], dtype=np.float64)
    A = np.asarray([abs(float(row["a_correct"])) for row in rows], dtype=np.float64)
    q1 = (cu > 0.0) & (cr > 0.0)
    q2 = (cu > 0.0) & (cr < 0.0)
    q3 = (cu < 0.0) & (cr > 0.0)
    q4 = (cu < 0.0) & (cr < 0.0)
    data = {
        "count": int(cu.size),
        "C_utility_mean": float(cu.mean()),
        "C_utility_median": float(np.median(cu)),
        "C_utility_std": float(cu.std(ddof=0)),
        "C_rgb_mean": float(cr.mean()),
        "C_rgb_median": float(np.median(cr)),
        "C_rgb_std": float(cr.std(ddof=0)),
        "fraction_C_utility_gt_0": float((cu > 0.0).mean()),
        "fraction_C_rgb_gt_0": float((cr > 0.0).mean()),
        "fraction_Q1": float(q1.mean()),
        "fraction_Q2": float(q2.mean()),
        "fraction_Q3": float(q3.mean()),
        "fraction_Q4": float(q4.mean()),
        "fraction_zero_utility": float((cu == 0.0).mean()),
        "fraction_zero_rgb": float((cr == 0.0).mean()),
        "sum_positive_C_utility": float(cu[cu > 0.0].sum()) if np.any(cu > 0.0) else 0.0,
        "sum_negative_C_utility": float(cu[cu < 0.0].sum()) if np.any(cu < 0.0) else 0.0,
        "sum_positive_C_rgb": float(cr[cr > 0.0].sum()) if np.any(cr > 0.0) else 0.0,
        "sum_negative_C_rgb": float(cr[cr < 0.0].sum()) if np.any(cr < 0.0) else 0.0,
        "spearman_C_utility_C_rgb": _safe_spearman(cu, cr),
        "pearson_C_utility_C_rgb": _safe_pearson(cu, cr),
        "E_C_rgb_given_C_utility_gt_0": float(cr[cu > 0.0].mean()) if np.any(cu > 0.0) else float("nan"),
        "E_C_rgb_given_C_utility_lt_0": float(cr[cu < 0.0].mean()) if np.any(cu < 0.0) else float("nan"),
        "E_C_utility_given_C_rgb_gt_0": float(cu[cr > 0.0].mean()) if np.any(cr > 0.0) else float("nan"),
        "E_C_utility_given_C_rgb_lt_0": float(cu[cr < 0.0].mean()) if np.any(cr < 0.0) else float("nan"),
        "fraction_C_rgb_gt_0_given_C_utility_gt_0": float((cr[cu > 0.0] > 0.0).mean()) if np.any(cu > 0.0) else float("nan"),
        "fraction_C_utility_gt_0_given_C_rgb_gt_0": float((cu[cr > 0.0] > 0.0).mean()) if np.any(cr > 0.0) else float("nan"),
        "A_mean": float(A.mean()),
        "A_median": float(np.median(A)),
        "spearman_A_C_utility": _safe_spearman(A, cu),
        "spearman_A_C_rgb": _safe_spearman(A, cr),
        "spearman_A_indicator_Q1": _safe_spearman(A, q1.astype(np.float64)),
    }
    data.update(_top_positive_share(cu, (0.01, 0.05, 0.10, 0.20)))
    data.update({f"C_utility_{k}": v for k, v in _participation_ratio(cu).items()})
    data.update({f"C_rgb_{k}": v for k, v in _participation_ratio(cr).items()})
    return data


def _alignment_quadrants(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_quadrant(row), []).append(row)
    order = ["Q1", "Q2", "Q3", "Q4", "ZERO_BOTH", "ZERO_UTILITY", "ZERO_RGB"]
    summary_rows: List[Dict[str, Any]] = []
    total = float(len(rows)) if rows else 1.0
    for label in order:
        subset = groups.get(label, [])
        cu = [float(row["C_utility_heldout"]) for row in subset]
        cr = [float(row["C_rgb_heldout"]) for row in subset]
        A = [abs(float(row["a_correct"])) for row in subset]
        summary_rows.append(
            {
                "quadrant": label,
                "count": int(len(subset)),
                "fraction_of_all_rays": float(len(subset) / total),
                "C_utility_mean": _mean(cu),
                "C_utility_median": _median(cu),
                "C_rgb_mean": _mean(cr),
                "C_rgb_median": _median(cr),
                "A_mean": _mean(A),
                "A_median": _median(A),
            }
        )
    totals = {
        "Q1_fraction": float(len(groups.get("Q1", [])) / total),
        "Q2_fraction": float(len(groups.get("Q2", [])) / total),
        "Q3_fraction": float(len(groups.get("Q3", [])) / total),
        "Q4_fraction": float(len(groups.get("Q4", [])) / total),
        "zero_both_fraction": float(len(groups.get("ZERO_BOTH", [])) / total),
        "zero_utility_fraction": float(len(groups.get("ZERO_UTILITY", [])) / total),
        "zero_rgb_fraction": float(len(groups.get("ZERO_RGB", [])) / total),
    }
    return summary_rows, totals


def _decile_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: abs(float(row["a_correct"])))
    n = len(ordered)
    decile_rows: List[Dict[str, Any]] = []
    for decile in range(10):
        start = (decile * n) // 10
        end = ((decile + 1) * n) // 10
        subset = ordered[start:end]
        cu = np.asarray([float(r["C_utility_heldout"]) for r in subset], dtype=np.float64)
        cr = np.asarray([float(r["C_rgb_heldout"]) for r in subset], dtype=np.float64)
        A = np.asarray([abs(float(r["a_correct"])) for r in subset], dtype=np.float64)
        q1 = (cu > 0.0) & (cr > 0.0)
        q2 = (cu > 0.0) & (cr < 0.0)
        q3 = (cu < 0.0) & (cr > 0.0)
        q4 = (cu < 0.0) & (cr < 0.0)
        cameras = sorted(set(r["source_view_id"] for r in subset))
        decile_rows.append(
            {
                "decile": f"D{decile + 1}",
                "count": int(len(subset)),
                "camera_count": int(len(cameras)),
                "A_mean": float(A.mean()),
                "A_median": float(np.median(A)),
                "C_utility_mean": float(cu.mean()),
                "C_utility_median": float(np.median(cu)),
                "fraction_C_utility_gt_0": float((cu > 0.0).mean()),
                "C_rgb_mean": float(cr.mean()),
                "C_rgb_median": float(np.median(cr)),
                "fraction_C_rgb_gt_0": float((cr > 0.0).mean()),
                "fraction_Q1": float(q1.mean()),
                "fraction_Q2": float(q2.mean()),
                "fraction_Q3": float(q3.mean()),
                "fraction_Q4": float(q4.mean()),
            }
        )
    bottom = ordered[: max(1, n // 10)]
    top = ordered[-max(1, n // 10) :]
    bottom_cu = np.asarray([float(r["C_utility_heldout"]) for r in bottom], dtype=np.float64)
    bottom_cr = np.asarray([float(r["C_rgb_heldout"]) for r in bottom], dtype=np.float64)
    bottom_A = np.asarray([abs(float(r["a_correct"])) for r in bottom], dtype=np.float64)
    top_cu = np.asarray([float(r["C_utility_heldout"]) for r in top], dtype=np.float64)
    top_cr = np.asarray([float(r["C_rgb_heldout"]) for r in top], dtype=np.float64)
    top_A = np.asarray([abs(float(r["a_correct"])) for r in top], dtype=np.float64)
    comparison = {
        "bottom_10pct": {
            "C_utility_mean": float(bottom_cu.mean()),
            "C_rgb_mean": float(bottom_cr.mean()),
            "fraction_Q1": float(((bottom_cu > 0.0) & (bottom_cr > 0.0)).mean()),
            "fraction_Q2": float(((bottom_cu > 0.0) & (bottom_cr < 0.0)).mean()),
            "A_mean": float(bottom_A.mean()),
        },
        "top_10pct": {
            "C_utility_mean": float(top_cu.mean()),
            "C_rgb_mean": float(top_cr.mean()),
            "fraction_Q1": float(((top_cu > 0.0) & (top_cr > 0.0)).mean()),
            "fraction_Q2": float(((top_cu > 0.0) & (top_cr < 0.0)).mean()),
            "A_mean": float(top_A.mean()),
        },
    }
    return decile_rows, comparison


def _contribution_concentration(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cu = np.asarray([float(row["C_utility_heldout"]) for row in rows], dtype=np.float64)
    cr = np.asarray([float(row["C_rgb_heldout"]) for row in rows], dtype=np.float64)

    def _summary(values: np.ndarray) -> Dict[str, Any]:
        pos = values[values > 0.0]
        neg = values[values < 0.0]
        if pos.size == 0:
            return {
                "positive_count": 0,
                "positive_sum": 0.0,
                "negative_sum": float(neg.sum()) if neg.size else 0.0,
                "top_1pct_share": float("nan"),
                "top_5pct_share": float("nan"),
                "top_10pct_share": float("nan"),
                "top_20pct_share": float("nan"),
                "effective_number": float("nan"),
                "participation_ratio": float("nan"),
            }
        pos_sorted = np.sort(pos)[::-1]
        total = float(pos_sorted.sum())
        k1 = max(1, int(math.ceil(pos_sorted.size * 0.01)))
        k5 = max(1, int(math.ceil(pos_sorted.size * 0.05)))
        k10 = max(1, int(math.ceil(pos_sorted.size * 0.10)))
        k20 = max(1, int(math.ceil(pos_sorted.size * 0.20)))
        effective = float(total * total / max(float(np.square(pos_sorted).sum()), EPS))
        return {
            "positive_count": int(pos_sorted.size),
            "positive_sum": total,
            "negative_sum": float(neg.sum()) if neg.size else 0.0,
            "top_1pct_share": float(pos_sorted[:k1].sum() / max(total, EPS)),
            "top_5pct_share": float(pos_sorted[:k5].sum() / max(total, EPS)),
            "top_10pct_share": float(pos_sorted[:k10].sum() / max(total, EPS)),
            "top_20pct_share": float(pos_sorted[:k20].sum() / max(total, EPS)),
            "effective_number": effective,
            "participation_ratio": float(effective / max(pos_sorted.size, 1)),
        }

    return {
        "C_utility": _summary(cu),
        "C_rgb": _summary(cr),
    }


def _per_camera_alignment(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_view_id"]), []).append(row)
    rows_out: List[Dict[str, Any]] = []
    camera_means_cu: List[float] = []
    camera_means_cr: List[float] = []
    camera_means_A: List[float] = []
    for view_id in sorted(grouped.keys()):
        subset = grouped[view_id]
        cu = np.asarray([float(r["C_utility_heldout"]) for r in subset], dtype=np.float64)
        cr = np.asarray([float(r["C_rgb_heldout"]) for r in subset], dtype=np.float64)
        A = np.asarray([abs(float(r["a_correct"])) for r in subset], dtype=np.float64)
        q1 = (cu > 0.0) & (cr > 0.0)
        q2 = (cu > 0.0) & (cr < 0.0)
        rows_out.append(
            {
                "source_view_id": view_id,
                "sampled_rays": int(len(subset)),
                "C_utility_mean": float(cu.mean()),
                "C_utility_median": float(np.median(cu)),
                "C_rgb_mean": float(cr.mean()),
                "C_rgb_median": float(np.median(cr)),
                "fraction_C_utility_gt_0": float((cu > 0.0).mean()),
                "fraction_C_rgb_gt_0": float((cr > 0.0).mean()),
                "fraction_Q1": float(q1.mean()),
                "fraction_Q2": float(q2.mean()),
                "A_mean": float(A.mean()),
                "A_top_decile": float(np.quantile(A, 0.9)),
                "spearman_A_C_rgb": _safe_spearman(A, cr),
                "spearman_A_C_utility": _safe_spearman(A, cu),
            }
        )
        camera_means_cu.append(float(cu.mean()))
        camera_means_cr.append(float(cr.mean()))
        camera_means_A.append(float(A.mean()))
    summary = {
        "camera_count": int(len(rows_out)),
        "camera_mean_C_utility_vs_C_rgb_spearman": _safe_spearman(camera_means_cu, camera_means_cr),
        "camera_mean_C_utility_vs_C_rgb_pearson": _safe_pearson(camera_means_cu, camera_means_cr),
        "camera_mean_A_vs_C_utility_spearman": _safe_spearman(camera_means_A, camera_means_cu),
        "camera_mean_A_vs_C_utility_pearson": _safe_pearson(camera_means_A, camera_means_cu),
        "camera_mean_A_vs_C_rgb_spearman": _safe_spearman(camera_means_A, camera_means_cr),
        "camera_mean_A_vs_C_rgb_pearson": _safe_pearson(camera_means_A, camera_means_cr),
    }
    return rows_out, summary


def _msafe_alignment(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cu = np.asarray([float(row["C_utility_heldout"]) for row in rows], dtype=np.float64)
    cr = np.asarray([float(row["C_rgb_heldout"]) for row in rows], dtype=np.float64)
    A = np.asarray([abs(float(row["A_natural_coeff"])) for row in rows], dtype=np.float64)
    q1 = (cu > 0.0) & (cr > 0.0)
    q2 = (cu > 0.0) & (cr < 0.0)
    q3 = (cu < 0.0) & (cr > 0.0)
    q4 = (cu < 0.0) & (cr < 0.0)
    return {
        "count": int(cu.size),
        "C_utility_mean": float(cu.mean()),
        "C_utility_median": float(np.median(cu)),
        "C_utility_std": float(cu.std(ddof=0)),
        "C_rgb_mean": float(cr.mean()),
        "C_rgb_median": float(np.median(cr)),
        "C_rgb_std": float(cr.std(ddof=0)),
        "fraction_C_utility_gt_0": float((cu > 0.0).mean()),
        "fraction_C_rgb_gt_0": float((cr > 0.0).mean()),
        "fraction_Q1": float(q1.mean()),
        "fraction_Q2": float(q2.mean()),
        "fraction_Q3": float(q3.mean()),
        "fraction_Q4": float(q4.mean()),
        "A_mean": float(A.mean()),
        "A_median": float(np.median(A)),
        "spearman_C_utility_C_rgb": _safe_spearman(cu, cr),
        "pearson_C_utility_C_rgb": _safe_pearson(cu, cr),
        "fraction_C_rgb_gt_0_given_C_utility_gt_0": float((cr[cu > 0.0] > 0.0).mean()) if np.any(cu > 0.0) else float("nan"),
        "fraction_C_utility_gt_0_given_C_rgb_gt_0": float((cu[cr > 0.0] > 0.0).mean()) if np.any(cr > 0.0) else float("nan"),
        "E_C_rgb_given_C_utility_gt_0": float(cr[cu > 0.0].mean()) if np.any(cu > 0.0) else float("nan"),
        "E_C_rgb_given_C_utility_lt_0": float(cr[cu < 0.0].mean()) if np.any(cu < 0.0) else float("nan"),
        "E_C_utility_given_C_rgb_gt_0": float(cu[cr > 0.0].mean()) if np.any(cr > 0.0) else float("nan"),
        "E_C_utility_given_C_rgb_lt_0": float(cu[cr < 0.0].mean()) if np.any(cr < 0.0) else float("nan"),
    }


def _stratify(rows: Sequence[Mapping[str, Any]], key: str, labels: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    thresholds = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    strata: List[Tuple[str, np.ndarray]] = []
    if key == "depth":
        strata = [
            (labels[0], values <= thresholds[0]),
            (labels[1], (values > thresholds[0]) & (values <= thresholds[1])),
            (labels[2], values > thresholds[1]),
        ]
    else:
        strata = [
            (labels[0], values <= thresholds[0]),
            (labels[1], (values > thresholds[0]) & (values <= thresholds[1])),
            (labels[2], values > thresholds[1]),
        ]
    out_rows: List[Dict[str, Any]] = []
    for label, mask in strata:
        idx = np.where(mask)[0]
        subset = [rows[int(i)] for i in idx]
        cu = np.asarray([float(row["C_utility_heldout"]) for row in subset], dtype=np.float64)
        cr = np.asarray([float(row["C_rgb_heldout"]) for row in subset], dtype=np.float64)
        A = np.asarray([abs(float(row["a_correct"])) for row in subset], dtype=np.float64)
        q1 = (cu > 0.0) & (cr > 0.0)
        q2 = (cu > 0.0) & (cr < 0.0)
        out_rows.append(
            {
                "stratum": label,
                "count": int(len(subset)),
                "mean_depth_or_tau": float(values[mask].mean()) if len(subset) else float("nan"),
                "median_depth_or_tau": float(np.median(values[mask])) if len(subset) else float("nan"),
                "C_utility_mean": float(cu.mean()) if len(subset) else float("nan"),
                "C_utility_median": float(np.median(cu)) if len(subset) else float("nan"),
                "C_rgb_mean": float(cr.mean()) if len(subset) else float("nan"),
                "C_rgb_median": float(np.median(cr)) if len(subset) else float("nan"),
                "fraction_Q1": float(q1.mean()) if len(subset) else float("nan"),
                "fraction_Q2": float(q2.mean()) if len(subset) else float("nan"),
                "A_mean": float(A.mean()) if len(subset) else float("nan"),
                "A_median": float(np.median(A)) if len(subset) else float("nan"),
            }
        )
    summary = {
        "threshold_1": float(thresholds[0]),
        "threshold_2": float(thresholds[1]),
        "labels": list(labels),
        "key": key,
    }
    return out_rows, summary


def _eval_alignment(
    model: Any,
    eval_records: Mapping[str, Tuple[int, Any, Dict[str, Any]]],
    projector_bundle: Mapping[str, Any],
    basis_vector: torch.Tensor,
    scale: torch.Tensor,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    psnr_full_values: List[float] = []
    psnr_minus_values: List[float] = []
    for view_id, (_idx, camera, batch) in sorted(eval_records.items()):
        camera = camera.to(model.device)
        rendered = _render_eval_view(model, camera, projector_bundle, basis_vector, scale)
        raw_base = rendered["raw_base"]
        raw_natural = rendered["raw_natural"]
        raw_effective = rendered["raw_effective"]
        raw_minus = rendered["raw_minus"]
        out_full = rendered["out_full"]
        out_minus = rendered["out_minus"]
        height = int(rendered["height"])
        width = int(rendered["width"])
        gt_full = PHASEA._gt_for(model, batch, out_full["background"]).reshape(-1, 3)
        gt_minus = PHASEA._gt_for(model, batch, out_minus["background"]).reshape(-1, 3)
        pred_full = out_full["pred_image"].reshape(-1, 3)
        pred_minus = out_minus["pred_image"].reshape(-1, 3)
        mse_full = (pred_full - gt_full).square().mean(dim=-1).detach().float().cpu().numpy()
        mse_minus = (pred_minus - gt_minus).square().mean(dim=-1).detach().float().cpu().numpy()
        cu = mse_minus - mse_full
        nat_std, nat_coeff, _ = _selected_mode_projection(raw_natural - raw_base, basis_vector, scale)
        a_correct = nat_coeff.detach().float().cpu().numpy()
        A = np.abs(a_correct)
        metrics_full = MIC._metric_images(model, out_full["pred_image"], gt_full.reshape(height, width, 3))
        metrics_minus = MIC._metric_images(model, out_minus["pred_image"], gt_minus.reshape(height, width, 3))
        psnr_full_values.append(float(metrics_full["PSNR"]))
        psnr_minus_values.append(float(metrics_minus["PSNR"]))
        rows.append(
            {
                "view_id": view_id,
                "sampled_rays": int(height * width),
                "PSNR_full": float(metrics_full["PSNR"]),
                "SSIM_full": float(metrics_full["SSIM"]),
                "LPIPS_full": float(metrics_full["LPIPS"]),
                "MSE_full": float(metrics_full["MSE"]),
                "PSNR_minus": float(metrics_minus["PSNR"]),
                "SSIM_minus": float(metrics_minus["SSIM"]),
                "LPIPS_minus": float(metrics_minus["LPIPS"]),
                "MSE_minus": float(metrics_minus["MSE"]),
                "delta_PSNR_minus_full": float(metrics_minus["PSNR"] - metrics_full["PSNR"]),
                "delta_SSIM_minus_full": float(metrics_minus["SSIM"] - metrics_full["SSIM"]),
                "delta_LPIPS_minus_full": float(metrics_minus["LPIPS"] - metrics_full["LPIPS"]),
                "delta_MSE_minus_full": float(metrics_minus["MSE"] - metrics_full["MSE"]),
                "C_rgb_mean": float(cu.mean()),
                "C_rgb_median": float(np.median(cu)),
                "C_rgb_fraction_positive": float((cu > 0.0).mean()),
                "A_mean": float(A.mean()),
                "A_median": float(np.median(A)),
                "A_fraction_positive": float((A > 0.0).mean()),
            }
        )
        del raw_base, raw_natural, raw_effective, raw_minus, out_full, out_minus, gt_full, gt_minus
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows.append(
        {
            "view_id": "MEAN",
            "sampled_rays": int(sum(int(row["sampled_rays"]) for row in rows)),
            "PSNR_full": _mean([row["PSNR_full"] for row in rows]),
            "SSIM_full": _mean([row["SSIM_full"] for row in rows]),
            "LPIPS_full": _mean([row["LPIPS_full"] for row in rows]),
            "MSE_full": _mean([row["MSE_full"] for row in rows]),
            "PSNR_minus": _mean([row["PSNR_minus"] for row in rows]),
            "SSIM_minus": _mean([row["SSIM_minus"] for row in rows]),
            "LPIPS_minus": _mean([row["LPIPS_minus"] for row in rows]),
            "MSE_minus": _mean([row["MSE_minus"] for row in rows]),
            "delta_PSNR_minus_full": _mean([row["delta_PSNR_minus_full"] for row in rows]),
            "delta_SSIM_minus_full": _mean([row["delta_SSIM_minus_full"] for row in rows]),
            "delta_LPIPS_minus_full": _mean([row["delta_LPIPS_minus_full"] for row in rows]),
            "delta_MSE_minus_full": _mean([row["delta_MSE_minus_full"] for row in rows]),
            "C_rgb_mean": _mean([row["C_rgb_mean"] for row in rows]),
            "C_rgb_median": _mean([row["C_rgb_median"] for row in rows]),
            "C_rgb_fraction_positive": _mean([row["C_rgb_fraction_positive"] for row in rows]),
            "A_mean": _mean([row["A_mean"] for row in rows]),
            "A_median": _mean([row["A_median"] for row in rows]),
            "A_fraction_positive": _mean([row["A_fraction_positive"] for row in rows]),
        }
    )
    summary = {
        "view_count": int(len(rows) - 1),
        "PSNR_full_mean": rows[-1]["PSNR_full"],
        "PSNR_minus_mean": rows[-1]["PSNR_minus"],
        "SSIM_full_mean": rows[-1]["SSIM_full"],
        "SSIM_minus_mean": rows[-1]["SSIM_minus"],
        "LPIPS_full_mean": rows[-1]["LPIPS_full"],
        "LPIPS_minus_mean": rows[-1]["LPIPS_minus"],
        "MSE_full_mean": rows[-1]["MSE_full"],
        "MSE_minus_mean": rows[-1]["MSE_minus"],
        "delta_PSNR_minus_full_mean": rows[-1]["delta_PSNR_minus_full"],
        "delta_SSIM_minus_full_mean": rows[-1]["delta_SSIM_minus_full"],
        "delta_LPIPS_minus_full_mean": rows[-1]["delta_LPIPS_minus_full"],
        "delta_MSE_minus_full_mean": rows[-1]["delta_MSE_minus_full"],
        "swap_utility_computed": False,
        "note": "eval counterfactual is limited to full-vs-remove RGB support; train-derived swap utility is not computed on eval.",
    }
    return rows, summary


def _choose_primary_classification(
    general: Mapping[str, Any],
    quadrants: Mapping[str, Any],
    decile_comparison: Mapping[str, Any],
    concentration: Mapping[str, Any],
    per_camera_summary: Mapping[str, Any],
    msafe: Mapping[str, Any],
    eval_summary: Mapping[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    mean_cu = float(general["C_utility_mean"])
    mean_cr = float(general["C_rgb_mean"])
    med_cu = float(general["C_utility_median"])
    med_cr = float(general["C_rgb_median"])
    frac_pos_cu = float(general["fraction_C_utility_gt_0"])
    frac_pos_cr = float(general["fraction_C_rgb_gt_0"])
    q1 = float(general["fraction_Q1"])
    q2 = float(general["fraction_Q2"])
    cond_cr_given_cu_pos = float(general["fraction_C_rgb_gt_0_given_C_utility_gt_0"])
    cond_cu_given_cr_pos = float(general["fraction_C_utility_gt_0_given_C_rgb_gt_0"])
    top10_cu = float(concentration["C_utility"]["top_10pct_share"])
    top10_cr = float(concentration["C_rgb"]["top_10pct_share"])
    low_decile_cr = float(decile_comparison["bottom_10pct"]["C_rgb_mean"])
    high_decile_cr = float(decile_comparison["top_10pct"]["C_rgb_mean"])
    low_decile_cu = float(decile_comparison["bottom_10pct"]["C_utility_mean"])
    high_decile_cu = float(decile_comparison["top_10pct"]["C_utility_mean"])
    eval_psnr_delta = float(eval_summary["delta_PSNR_minus_full_mean"])
    eval_lpips_delta = float(eval_summary["delta_LPIPS_minus_full_mean"])
    eval_ssim_delta = float(eval_summary["delta_SSIM_minus_full_mean"])

    broad_alignment = (
        mean_cu > 0.0
        and mean_cr > 0.0
        and med_cu >= 0.0
        and med_cr >= 0.0
        and frac_pos_cu > 0.40
        and frac_pos_cr > 0.40
        and q1 > q2
        and cond_cr_given_cu_pos > 0.50
        and cond_cu_given_cr_pos > 0.50
    )
    context_dependent = (
        high_decile_cr > low_decile_cr
        and high_decile_cu > low_decile_cu
        and top10_cu >= 0.50
        and top10_cr >= 0.50
        and float(general["spearman_A_C_rgb"]) > 0.0
        and float(general["spearman_A_C_utility"]) > 0.0
    )
    eval_contradiction = eval_psnr_delta > 0.0 or eval_lpips_delta > 0.0 or eval_ssim_delta > 0.0
    msafe_mixed = float(msafe["C_rgb_mean"]) <= 0.0 or float(msafe["fraction_C_utility_gt_0"]) < 0.40
    per_camera_broad = float(per_camera_summary["camera_mean_C_utility_vs_C_rgb_spearman"]) > 0.70

    if broad_alignment and not eval_contradiction:
        primary = "UTILITY_RECONSTRUCTION_ALIGNMENT_SUPPORTED"
        global_gate = "GLOBAL_MODE_UTILITY_GATING_SUPPORTED"
        reason = "GENERAL alignment is positive and eval does not contradict preservation."
    elif broad_alignment and context_dependent:
        primary = "SPARSE_CONTEXT_DEPENDENT_SUPPORT"
        global_gate = "GLOBAL_MODE_UTILITY_GATING_NOT_SUPPORTED"
        reason = "Alignment exists, but it concentrates in lower-depth/lower-tau/high-activation contexts and eval remains mixed."
    elif mean_cu > 0.0 and (mean_cr <= 0.0 or q2 >= q1 or cond_cr_given_cu_pos <= 0.50 or eval_psnr_delta > 0.0):
        primary = "CAMERA_SPECIFICITY_WITHOUT_RECONSTRUCTION_SUPPORT"
        global_gate = "GLOBAL_MODE_UTILITY_GATING_NOT_SUPPORTED"
        reason = "Camera specificity is present, but reconstruction support is not reliable enough for a preservation gate."
    else:
        primary = "UTILITY_RECONSTRUCTION_ALIGNMENT_NOT_RESOLVED"
        global_gate = "GLOBAL_MODE_UTILITY_GATING_INCONCLUSIVE"
        reason = "Signals remain mixed or too sparse for a stable alignment claim."

    evidence = {
        "mean_C_utility": mean_cu,
        "median_C_utility": med_cu,
        "mean_C_rgb": mean_cr,
        "median_C_rgb": med_cr,
        "fraction_C_utility_gt_0": frac_pos_cu,
        "fraction_C_rgb_gt_0": frac_pos_cr,
        "fraction_Q1": q1,
        "fraction_Q2": q2,
        "fraction_C_rgb_gt_0_given_C_utility_gt_0": cond_cr_given_cu_pos,
        "fraction_C_utility_gt_0_given_C_rgb_gt_0": cond_cu_given_cr_pos,
        "top_10pct_C_utility_share": top10_cu,
        "top_10pct_C_rgb_share": top10_cr,
        "eval_delta_PSNR_mean": eval_psnr_delta,
        "eval_delta_LPIPS_mean": eval_lpips_delta,
        "eval_delta_SSIM_mean": eval_ssim_delta,
        "per_camera_correlation": float(per_camera_summary["camera_mean_C_utility_vs_C_rgb_spearman"]),
        "msafe_mean_C_rgb": float(msafe["C_rgb_mean"]),
        "msafe_mean_C_utility": float(msafe["C_utility_mean"]),
    }
    return primary, global_gate, {"reason": reason, "evidence": evidence}


def _write_research_note(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# SELECTED-MODE UTILITY / RECONSTRUCTION ALIGNMENT AUDIT",
        "",
        "## CODE FACT",
        "This task reused the frozen selected mode_01, the previous held-out GENERAL bank, and the previous swap bank.",
        "No training, no optimizer step, and no mode reselection were performed.",
        "",
        "## CONFIG FACT",
        f"Selection source: `{summary['frozen_mode_provenance']['checkpoint_path']}`.",
        f"Previous held-out bank: `{PREVIOUS_OUTPUT_DIR}`.",
        f"Output dir: `{OUTPUT_DIR}`.",
        "",
        "## EXPERIMENTAL FACT",
        f"CONDA_ENV: `{summary['environment']['CONDA_ENV']}`.",
        f"CUDA_VISIBLE_DEVICES: `{summary['environment']['CUDA_VISIBLE_DEVICES']}`.",
        f"GPU: `{summary['gpu']['gpu_name']}`.",
        f"Frozen mode match: `{summary['frozen_mode_provenance']['frozen_mode_match']}`.",
        f"Bank reuse strict overlap: `{summary['frozen_bank_provenance']['strict_zero_overlap']}`.",
        "",
        "## QUANTITATIVE RESULT",
        f"GENERAL mean C_utility: `{summary['general_alignment']['C_utility_mean']}`.",
        f"GENERAL median C_utility: `{summary['general_alignment']['C_utility_median']}`.",
        f"GENERAL mean C_rgb: `{summary['general_alignment']['C_rgb_mean']}`.",
        f"GENERAL median C_rgb: `{summary['general_alignment']['C_rgb_median']}`.",
        f"GENERAL Q1/Q2: `{summary['general_alignment']['fraction_Q1']}` / `{summary['general_alignment']['fraction_Q2']}`.",
        f"GENERAL Spearman(A, C_rgb): `{summary['general_alignment']['spearman_A_C_rgb']}`.",
        f"GENERAL eval delta PSNR: `{summary['eval_alignment']['delta_PSNR_minus_full_mean']}`.",
        f"GENERAL eval delta LPIPS: `{summary['eval_alignment']['delta_LPIPS_minus_full_mean']}`.",
        "",
        "## INFERENCE",
        f"Primary classification: `{summary['primary_classification']}`.",
        f"Global-gating classification: `{summary['global_gating_classification']}`.",
        f"Reason: {summary['classification_reason']}",
        f"Next task: `{summary['next_task']}`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--previous-output-dir", type=Path, default=PREVIOUS_OUTPUT_DIR)
    parser.add_argument("--source-output-dir", type=Path, default=SOURCE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    args = parser.parse_args()

    repo = args.repo.resolve()
    previous_output_dir = args.previous_output_dir.resolve()
    source_output_dir = args.source_output_dir.resolve()
    output_dir = args.output_dir.resolve()
    log_dir = args.log_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_manifest = _assert_runtime_policy()
    environment_manifest = _environment_manifest(gpu_manifest)
    repo_manifest = _repo_manifest(repo)
    previous = _load_previous_assets(previous_output_dir)
    frozen_mode_provenance = _frozen_mode_provenance(previous)
    frozen_bank_provenance = _frozen_bank_provenance(previous)
    if not frozen_mode_provenance["frozen_mode_match"]:
        raise RuntimeError("Frozen mode provenance mismatch; stop and diagnose provenance.")

    _write_json(output_dir / "gpu_manifest.json", gpu_manifest)
    _write_json(output_dir / "environment_manifest.json", environment_manifest)
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "frozen_mode_provenance.json", frozen_mode_provenance)
    _write_json(output_dir / "frozen_bank_provenance.json", frozen_bank_provenance)

    branch, projector_bundle = _load_branch(repo, source_output_dir)
    try:
        train_records = _train_records(branch)
        eval_records = _eval_records(branch)

        selection_mode = previous["selection_mode"]
        basis_vector = torch.tensor(selection_mode["mode_vector"], dtype=torch.float64)
        scale = torch.tensor(selection_mode["scale"], dtype=torch.float64)

        depth_maps: Dict[str, np.ndarray] = {}
        tau_maps: Dict[str, np.ndarray] = {}
        transmission_maps: Dict[str, np.ndarray] = {}
        a_correct_crosscheck: Dict[str, np.ndarray] = {}
        for view_id, (_idx, camera, _batch) in train_records.items():
            camera = camera.to(branch.pipeline.model.device)
            rendered = _render_train_view(branch.pipeline.model, camera, projector_bundle)
            raw_base = rendered["raw_base"]
            raw_natural = rendered["raw_natural"]
            raw_effective = rendered["raw_effective"]
            out_full = rendered["out_full"]
            nat_std, nat_coeff, _ = _selected_mode_projection(raw_natural - raw_base, basis_vector, scale)
            depth_maps[view_id] = out_full["depth"].reshape(-1).detach().float().cpu().numpy()
            tau_tensor = out_full["tau_D"].reshape(-1, 3).detach().float().cpu()
            tau_maps[view_id] = tau_tensor.mean(dim=-1).numpy()
            if "transmission" in out_full:
                trans = out_full["transmission"].detach().float()
                if trans.ndim >= 3 and trans.shape[-1] > 1:
                    transmission_maps[view_id] = trans.reshape(-1, trans.shape[-1]).mean(dim=-1).cpu().numpy()
                else:
                    transmission_maps[view_id] = trans.reshape(-1).cpu().numpy()
            else:
                transmission_maps[view_id] = np.full(depth_maps[view_id].shape, float("nan"), dtype=np.float64)
            a_correct_crosscheck[view_id] = nat_coeff.detach().float().cpu().numpy()
            del raw_base, raw_natural, raw_effective, out_full, nat_std, nat_coeff
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        general_rows = previous["selected_rows"]
        train_contexts = {
            view_id: _camera_context(branch.pipeline.model, camera.to(branch.pipeline.model.device))
            for view_id, (_idx, camera, _batch) in train_records.items()
        }
        bank_rows_by_view = {row["view_id"]: row for row in previous["ray_bank"]["rows"]}
        swap_coefficients: Dict[str, List[np.ndarray]] = {}
        for source_view, alternatives in previous["swap_bank"]["rows"].items():
            source_camera = train_records[source_view][1].to(branch.pipeline.model.device)
            flat_indices = torch.tensor(
                bank_rows_by_view[source_view]["heldout_general_flat"],
                dtype=torch.long,
                device="cpu",
            )
            swap_coefficients[source_view] = []
            for alternative in alternatives:
                coefficients = _swap_mode_coefficients(
                    branch.pipeline.model,
                    source_camera,
                    projector_bundle,
                    train_contexts[alternative],
                    flat_indices,
                    basis_vector,
                    scale,
                )
                swap_coefficients[source_view].append(coefficients)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        general_rows_aug = _augment_general_rows(
            general_rows,
            previous["ray_bank"],
            train_records,
            depth_maps,
            tau_maps,
            transmission_maps,
            swap_coefficients,
        )
        # Cross-check the frozen coefficient against the previous output.
        previous_row_lookup = {
            (str(row["source_view_id"]), int(row["flat_index"])): float(row["A_natural_coeff"])
            for row in general_rows
        }
        max_abs_a_diff = 0.0
        for row in general_rows_aug:
            view = str(row["source_view_id"])
            flat_index = int(row["flat_index"])
            prev_a = previous_row_lookup[(view, flat_index)]
            max_abs_a_diff = max(max_abs_a_diff, abs(float(row["a_correct"]) - prev_a))

        general_alignment = _global_alignment(general_rows_aug)
        alignment_quadrants_rows, quadrant_summary = _alignment_quadrants(general_rows_aug)
        zero_analysis = _zero_analysis(general_rows_aug)
        decile_rows, decile_comparison = _decile_rows(general_rows_aug)
        concentration = _contribution_concentration(general_rows_aug)
        per_camera_rows, per_camera_summary = _per_camera_alignment(general_rows_aug)
        depth_rows, depth_summary = _stratify(general_rows_aug, "depth", ["near", "middle", "far"])
        tau_rows, tau_summary = _stratify(general_rows_aug, "tau", ["low", "middle", "high"])
        msafe_alignment = _msafe_alignment(previous["msafe_rows"])

        # Build GENERAL per-ray outputs.
        per_ray_rows = []
        for row in general_rows_aug:
            item = dict(row)
            item["quadrant"] = _quadrant(row)
            per_ray_rows.append(item)

        # Eval alignment.
        eval_rows, eval_summary = _eval_alignment(
            branch.pipeline.model,
            eval_records,
            projector_bundle,
            basis_vector,
            scale,
        )

        primary_classification, global_gate_classification, classification_payload = _choose_primary_classification(
            general_alignment,
            quadrant_summary,
            decile_comparison,
            concentration,
            per_camera_summary,
            msafe_alignment,
            eval_summary,
        )
        next_task = "Design a ray/context-adaptive capacity-allocation preflight."
        zero_training_audit = {
            "training_invoked": False,
            "optimizer_step_count": 0,
            "parameter_delta_max": 0.0,
            "checkpoint_modified": False,
            "projector_state_persisted": False,
            "mode_reselection": False,
            "basis_recomputed": False,
            "audit_mode": "read_only_reuse_of_frozen_outputs_plus_descriptive_rerender",
        }
        checkpoint_basis_verification = {
            "primary_checkpoint_path": frozen_mode_provenance["checkpoint_path"],
            "stored_step": frozen_mode_provenance["stored_step"],
            "checkpoint_hash_match": frozen_mode_provenance["checkpoint_hash_match"],
            "frozen_mode_match": frozen_mode_provenance["frozen_mode_match"],
            "frozen_bank_hash_match": frozen_bank_provenance["bank_hash"] == EXPECTED_BANK_HASH,
            "frozen_swap_hash_match": frozen_bank_provenance["swap_bank_hash"] == EXPECTED_SWAP_HASH,
            "frozen_a_crosscheck_max_abs_diff": max_abs_a_diff,
        }
        _write_json(output_dir / "zero_training_audit.json", zero_training_audit)
        _write_json(output_dir / "checkpoint_basis_verification.json", checkpoint_basis_verification)

        # Write rows and summaries.
        _write_csv(output_dir / "per_ray_alignment.csv", per_ray_rows)
        _write_json(output_dir / "per_ray_alignment.json", {"rows": per_ray_rows, "summary": general_alignment, "frozen_mode_crosscheck_max_abs_diff": max_abs_a_diff})

        _write_csv(output_dir / "alignment_quadrants.csv", alignment_quadrants_rows)
        _write_json(output_dir / "alignment_quadrants.json", {"rows": alignment_quadrants_rows, "summary": quadrant_summary})

        _write_json(output_dir / "numerical_zero_analysis.json", zero_analysis)

        _write_csv(output_dir / "amplitude_deciles.csv", decile_rows)
        _write_json(output_dir / "amplitude_deciles.json", {"rows": decile_rows, "summary": decile_comparison, "correlations": {"spearman_A_C_utility": general_alignment["spearman_A_C_utility"], "spearman_A_C_rgb": general_alignment["spearman_A_C_rgb"], "spearman_A_indicator_Q1": general_alignment["spearman_A_indicator_Q1"]}})

        _write_json(output_dir / "contribution_concentration.json", concentration)

        _write_csv(output_dir / "per_camera_alignment.csv", per_camera_rows)
        _write_json(output_dir / "per_camera_alignment.json", {"rows": per_camera_rows, "summary": per_camera_summary})

        msafe_rows = []
        for row in previous["msafe_rows"]:
            item = dict(row)
            item["A"] = abs(float(row["A_natural_coeff"]))
            item["quadrant"] = _quadrant(row)
            msafe_rows.append(item)
        _write_csv(output_dir / "msafe_alignment.csv", msafe_rows)
        _write_json(output_dir / "msafe_alignment.json", {"rows": msafe_rows, "summary": msafe_alignment})

        _write_csv(output_dir / "depth_alignment.csv", depth_rows)
        _write_json(output_dir / "depth_alignment.json", {"rows": depth_rows, "summary": depth_summary})

        _write_csv(output_dir / "tau_alignment.csv", tau_rows)
        _write_json(output_dir / "tau_alignment.json", {"rows": tau_rows, "summary": tau_summary})

        _write_csv(output_dir / "eval_alignment_summary.csv", eval_rows)
        _write_json(output_dir / "eval_alignment_summary.json", {"rows": eval_rows, "summary": eval_summary})

        final_summary = {
            "experiment": EXPERIMENT,
            "scene": SCENE,
            "frozen_mode_provenance": frozen_mode_provenance,
            "frozen_bank_provenance": frozen_bank_provenance,
            "checkpoint_manifest": previous["checkpoint_manifest"],
            "zero_training_audit": zero_training_audit,
            "checkpoint_basis_verification": checkpoint_basis_verification,
            "general_alignment": general_alignment,
            "alignment_quadrants": quadrant_summary,
            "zero_analysis": zero_analysis,
            "decile_comparison": decile_comparison,
            "contribution_concentration": concentration,
            "per_camera_summary": per_camera_summary,
            "msafe_alignment": msafe_alignment,
            "depth_alignment": depth_summary,
            "tau_alignment": tau_summary,
            "eval_alignment": eval_summary,
            "primary_alignment_classification": primary_classification,
            "global_gating_classification": global_gate_classification,
            "classification_reason": classification_payload["reason"],
            "classification_evidence": classification_payload["evidence"],
            "next_task": next_task,
            "eval_aggregate_psnr_full": float(previous["eval_rows"][-1]["PSNR_full"]),
            "eval_aggregate_psnr_minus": float(previous["eval_rows"][-1]["PSNR_minus"]),
            "eval_aggregate_ssim_full": float(previous["eval_rows"][-1]["SSIM_full"]),
            "eval_aggregate_ssim_minus": float(previous["eval_rows"][-1]["SSIM_minus"]),
            "eval_aggregate_lpips_full": float(previous["eval_rows"][-1]["LPIPS_full"]),
            "eval_aggregate_lpips_minus": float(previous["eval_rows"][-1]["LPIPS_minus"]),
            "frozen_mode_crosscheck_max_abs_a_diff": max_abs_a_diff,
            "train_camera_count": int(len(train_records)),
            "eval_view_count": int(len(eval_records)),
        }
        _write_json(output_dir / "final_classification.json", {
            "primary_alignment_classification": primary_classification,
            "global_gating_classification": global_gate_classification,
            "classification_reason": classification_payload["reason"],
            "classification_evidence": classification_payload["evidence"],
            "next_task": next_task,
        })
        _write_json(output_dir / "final_summary.json", final_summary)

        print(json.dumps({
            "primary_alignment_classification": primary_classification,
            "global_gating_classification": global_gate_classification,
            "frozen_mode_match": frozen_mode_provenance["frozen_mode_match"],
        }, sort_keys=True), flush=True)
    finally:
        PHASEA.OCMC._release(branch)

    _write_research_note(
        RESEARCH_NOTE,
        {
            "environment": environment_manifest,
            "gpu": gpu_manifest,
            "frozen_mode_provenance": frozen_mode_provenance,
            "frozen_bank_provenance": frozen_bank_provenance,
            "general_alignment": general_alignment,
            "eval_alignment": eval_summary,
            "primary_classification": primary_classification,
            "global_gating_classification": global_gate_classification,
            "classification_reason": classification_payload["reason"],
            "next_task": next_task,
        },
    )


if __name__ == "__main__":
    main()
