#!/usr/bin/env python3
"""Read-only preflight for inference-available local contextual support on IUI3."""

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
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from water_splatting.utils import bin_and_sort_gaussians, compute_cumulative_intersects

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import validate_heldout_single_mode_camera_utility_iui3 as PHASEA
from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC


EXPERIMENT = "LOCAL-CONTEXTUAL-SUPPORT-PREDICTOR-PREFLIGHT"
SCENE = "IUI3-RedSea"
PREVIOUS_OUTPUT_DIR = Path("outputs/heldout_single_mode_camera_utility_iui3_20260826")
SOURCE_OUTPUT_DIR = Path("outputs/m1_ocmc_causal_iui3_20260825")
OUTPUT_DIR = Path("outputs/local_contextual_support_predictor_iui3_20260827")
LOG_DIR = Path("logs/local_contextual_support_predictor_iui3_20260827")
RESEARCH_NOTE = Path("research_notes/LOCAL_CONTEXTUAL_SUPPORT_PREDICTOR_IUI3_2026-08-27.md")
FINAL_STEP = 14999
SAMPLES_PER_VIEW = 1024
EPS = 1e-12
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
EXPECTED_BANK_HASH = "e23a146c5d34685605ab7f7a1845408fa0460e2d583310650b3d10279fa323d2"
EXPECTED_SWAP_HASH = "0ad944cc60548e712e420083e361df1332b8781c8dc54f9265c2204df8f87d5a"
EXPECTED_CHECKPOINT_SHA256 = "84cb33f31de6d8af4bbb8df2e6c3619a8257e21f93bb74cb6241735652fed997"
EXPECTED_MODE = "mode_01"
EXPECTED_MODE_INDEX = 1
EXPECTED_SIGMA = 0.01034344732761383
EXPECTED_G_OBS = 0.2694622658372983
ANALYTIC_ACTION_CHUNK_SIZE = 1024


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
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return "unavailable: %s" % exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default).encode("utf8")).hexdigest()


def _repo_manifest(repo: Path) -> Dict[str, Any]:
    return {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "status_short": _git(repo, "status", "--short"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "diff_check": _git(repo, "diff", "--check"),
        "historical_gmvc_files_preserved": [
            "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
            "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
        ],
    }


def _runtime_manifest() -> Dict[str, Any]:
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if env != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting, got %r" % env)
    if len(devices) != 1 or devices[0] not in ALLOWED_PHYSICAL_GPUS:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be one of %s, got %r" % (sorted(ALLOWED_PHYSICAL_GPUS), visible))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allowed CUDA device must be visible")
    props = torch.cuda.get_device_properties(0)
    extension = REPO_ROOT / "water_splatting" / "csrc.so"
    return {
        "CONDA_ENV": env,
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": visible,
        "physical_gpu_id": devices[0],
        "torch_logical_gpu_id": 0,
        "torch_cuda_available": True,
        "torch_visible_gpu_count": 1,
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "cuda_extension_path": str(extension),
        "cuda_extension_sha256": _sha256_file(extension) if extension.exists() else "missing",
    }


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _median(values: Iterable[float]) -> float:
    vals = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    return float(vals[mid]) if len(vals) % 2 else float((vals[mid - 1] + vals[mid]) / 2.0)


def _std(values: Iterable[float]) -> float:
    vals = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    return float(vals.std()) if vals.size else float("nan")


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(mask.sum()) < 2 or np.all(xv[mask] == xv[mask][0]) or np.all(yv[mask] == yv[mask][0]):
        return float("nan")
    return float(spearmanr(xv[mask], yv[mask]).correlation)


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pos = y == 1
    neg = y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum()))


def _average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    cumulative = np.cumsum(ranked)
    hits = ranked == 1
    return float(np.sum(cumulative[hits] / (np.arange(len(y))[hits] + 1.0)) / positives)


def _top_rows(rows: Sequence[Mapping[str, Any]], score_key: str, fraction: float) -> List[Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[score_key]), reverse=True)
    count = max(1, int(math.ceil(len(ordered) * fraction)))
    return ordered[:count]


def _bottom_rows(rows: Sequence[Mapping[str, Any]], score_key: str, fraction: float) -> List[Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[score_key]))
    count = max(1, int(math.ceil(len(ordered) * fraction)))
    return ordered[:count]


def _row_stats(rows: Sequence[Mapping[str, Any]], score_key: str = "LCS") -> Dict[str, Any]:
    cu = [float(row["C_utility"]) for row in rows if "C_utility" in row and math.isfinite(float(row["C_utility"]))]
    cr = [float(row["C_rgb"]) for row in rows if "C_rgb" in row and math.isfinite(float(row["C_rgb"]))]
    q1 = [float(row["Y_Q1"]) for row in rows if "Y_Q1" in row]
    q2 = [float(row["Y_Q2"]) for row in rows if "Y_Q2" in row]
    return {
        "count": int(len(rows)),
        "score_mean": _mean(float(row[score_key]) for row in rows),
        "score_median": _median(float(row[score_key]) for row in rows),
        "Q1_fraction": _mean(q1),
        "Q2_fraction": _mean(q2),
        "C_rgb_mean": _mean(cr),
        "C_rgb_median": _median(cr),
        "C_utility_mean": _mean(cu),
        "C_utility_median": _median(cu),
        "A_mean": _mean(float(row["A"]) for row in rows),
        "S_mean": _mean(float(row["S"]) for row in rows),
        "E_mode_mean": _mean(float(row["E_mode"]) for row in rows),
        "represented_cameras": int(len(set(str(row["source_view_id"]) for row in rows))),
    }


def _score_metrics(rows: Sequence[Mapping[str, Any]], score_key: str, target_key: str = "Y_Q1") -> Dict[str, Any]:
    usable = [
        row for row in rows
        if target_key in row and row[target_key] is not None and math.isfinite(float(row[score_key]))
    ]
    scores = [float(row[score_key]) for row in usable]
    labels = [int(row[target_key]) for row in usable]
    cr = [float(row["C_rgb"]) for row in usable]
    cu = [float(row["C_utility"]) for row in usable]
    top20 = _top_rows(usable, score_key, 0.20)
    bottom20 = _bottom_rows(usable, score_key, 0.20)
    return {
        "score": score_key,
        "count": len(usable),
        "positive_base_rate": float(sum(labels) / len(labels)) if labels else float("nan"),
        "AUROC_Y_Q1": _auroc(scores, labels),
        "AUPRC_Y_Q1": _average_precision(scores, labels),
        "Spearman_score_C_rgb": _safe_spearman(scores, cr),
        "Spearman_score_C_utility": _safe_spearman(scores, cu),
        "Spearman_score_Y_Q1": _safe_spearman(scores, labels),
        "top20_Q1_fraction": _mean(float(row["Y_Q1"]) for row in top20),
        "bottom20_Q1_fraction": _mean(float(row["Y_Q1"]) for row in bottom20),
        "top20_Q2_fraction": _mean(float(row["Y_Q2"]) for row in top20),
        "bottom20_Q2_fraction": _mean(float(row["Y_Q2"]) for row in bottom20),
        "top20_C_rgb_mean": _mean(float(row["C_rgb"]) for row in top20),
        "bottom20_C_rgb_mean": _mean(float(row["C_rgb"]) for row in bottom20),
        "top20_C_utility_mean": _mean(float(row["C_utility"]) for row in top20),
        "bottom20_C_utility_mean": _mean(float(row["C_utility"]) for row in bottom20),
        "top20_Q1_gap": _mean(float(row["Y_Q1"]) for row in top20) - _mean(float(row["Y_Q1"]) for row in bottom20),
    }


def _load_assets(previous_dir: Path) -> Dict[str, Any]:
    selection = json.loads((previous_dir / "selection_mode.json").read_text(encoding="utf8"))
    ray_bank = json.loads((previous_dir / "heldout_ray_bank.json").read_text(encoding="utf8"))
    swap_bank = json.loads((previous_dir / "heldout_swap_bank.json").read_text(encoding="utf8"))
    general = json.loads((previous_dir / "heldout_selected_mode_removal.json").read_text(encoding="utf8"))["rows"]
    safe = json.loads((previous_dir / "heldout_msafe_replication.json").read_text(encoding="utf8"))["rows"]
    phase_a = json.loads((previous_dir / "phase_a_classification.json").read_text(encoding="utf8"))
    checkpoint = Path(selection["checkpoint_path"])
    if ray_bank["rows_hash"] != EXPECTED_BANK_HASH or swap_bank["hash"] != EXPECTED_SWAP_HASH:
        raise RuntimeError("Frozen held-out bank or swap bank hash mismatch")
    if selection["mode_label"] != EXPECTED_MODE or int(selection["mode_index"]) != EXPECTED_MODE_INDEX:
        raise RuntimeError("Frozen mode mismatch")
    if abs(float(selection["sigma"]) - EXPECTED_SIGMA) > 1e-12 or abs(float(selection["g_obs"]) - EXPECTED_G_OBS) > 1e-12:
        raise RuntimeError("Frozen mode scalar mismatch")
    if int(json.loads((previous_dir / "checkpoint_manifest.json").read_text(encoding="utf8"))["loaded_step"]) != FINAL_STEP:
        raise RuntimeError("Frozen checkpoint step mismatch")
    if _sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen checkpoint hash mismatch")
    general_map = {(str(row["source_view_id"]), int(row["ray_index_within_bank"])): row for row in general}
    safe_map = {(str(row["source_view_id"]), int(row["ray_index_within_bank"])): row for row in safe}
    return {
        "selection": selection,
        "ray_bank": ray_bank,
        "swap_bank": swap_bank,
        "general_map": general_map,
        "safe_map": safe_map,
        "phase_a": phase_a,
        "checkpoint_sha256": _sha256_file(checkpoint),
    }


def _projected_delta(delta_raw: torch.Tensor, vector: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_t = scale.reshape(1, 9).to(device=delta_raw.device, dtype=delta_raw.dtype).clamp_min(EPS)
    vector_t = vector.reshape(9).to(device=delta_raw.device, dtype=delta_raw.dtype)
    delta_std = delta_raw.reshape(-1, 9) / scale_t
    coeff = delta_std @ vector_t
    removed_std = delta_std - coeff[:, None] * vector_t[None, :]
    removed_raw = (removed_std * scale_t).reshape_as(delta_raw)
    return delta_std, coeff, removed_raw


@torch.no_grad()
def _raw_state(model: Any, camera: Any, bundle: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    return PHASEA._natural_and_effective_raw(model, camera, bundle, camera_context_override=None)


@torch.no_grad()
def _render_no_grad(model: Any, camera: Any, raw: torch.Tensor, height: int, width: int) -> Dict[str, torch.Tensor]:
    med = MI._activate_medium(model, raw, height, width)
    return MI._render_with_medium_override(model, camera, med["medium_rgb"], med["medium_bs"], med["medium_attn"], detach_object_state=True)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _parameter_snapshot(model: Any) -> Dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}


def _parameter_delta(before: Mapping[str, torch.Tensor], model: Any) -> float:
    deltas = []
    for name, param in model.named_parameters():
        if name in before:
            current = param.detach().cpu()
            if current.numel() == 0:
                deltas.append(0.0)
            else:
                deltas.append(float((current - before[name]).abs().max().item()))
    return max(deltas) if deltas else 0.0


@torch.no_grad()
def _render_geometry(
    model: Any,
    camera: Any,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    """Recover the detached geometry and colors used by the CUDA forward path."""
    camera = camera.to(model.device)
    camera_downscale = model._get_downscale_factor()
    camera.rescale_output_resolution(1 / camera_downscale)
    try:
        rotation = camera.camera_to_worlds[0, :3, :3]
        translation = camera.camera_to_worlds[0, :3, 3:4]
        r_edit = torch.diag(torch.tensor([1, -1, -1], device=model.device, dtype=rotation.dtype))
        rotation = rotation @ r_edit
        r_inv = rotation.T
        t_inv = -r_inv @ translation
        viewmat = torch.eye(4, device=rotation.device, dtype=rotation.dtype)
        viewmat[:3, :3] = r_inv
        viewmat[:3, 3:4] = t_inv
        cx = float(camera.cx.item())
        cy = float(camera.cy.item())
        actual_width = int(camera.width.item())
        actual_height = int(camera.height.item())
        if model.crop_box is not None and not model.training:
            crop_ids = model.crop_box.within(model.means).squeeze()
        else:
            crop_ids = None
        if crop_ids is not None and crop_ids.sum() != 0:
            opacities_crop = model.opacities[crop_ids]
            means_crop = model.means[crop_ids]
            features_dc_crop = model.features_dc[crop_ids]
            features_rest_crop = model.features_rest[crop_ids]
            scales_crop = model.scales[crop_ids]
            quats_crop = model.quats[crop_ids]
        else:
            opacities_crop = model.opacities
            means_crop = model.means
            features_dc_crop = model.features_dc
            features_rest_crop = model.features_rest
            scales_crop = model.scales
            quats_crop = model.quats
        xys, depths, radii, conics, comp, num_tiles_hit, _ = model.underwater_rasterizer.project(
            means=means_crop.detach(),
            scales=scales_crop.detach(),
            quats=quats_crop.detach(),
            viewmat=viewmat,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=cx,
            cy=cy,
            height=actual_height,
            width=actual_width,
            clip_thresh=model.config.clip_thresh,
        )
        colors = MI._colors_for_current_parameterization(
            model,
            camera,
            means_crop.detach(),
            features_dc_crop.detach(),
            features_rest_crop.detach(),
        ).detach()
        if model.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop.detach()) * comp[:, None]
        elif model.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop.detach())
        else:
            raise ValueError("Unknown rasterize_mode: %s" % model.config.rasterize_mode)
    finally:
        camera.rescale_output_resolution(camera_downscale)
    tile_bounds = (
        (actual_width + model.underwater_rasterizer.block_width - 1) // model.underwater_rasterizer.block_width,
        (actual_height + model.underwater_rasterizer.block_width - 1) // model.underwater_rasterizer.block_width,
        1,
    )
    return xys, depths, radii, conics, colors, opacities, num_tiles_hit, torch.tensor(
        [actual_height, actual_width], device=model.device, dtype=torch.int32
    ), tile_bounds


@torch.no_grad()
def _analytic_local_jacobian_actions(
    model: Any,
    camera: Any,
    raw: torch.Tensor,
    directions: Sequence[torch.Tensor],
    height: int,
    width: int,
    flat: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Evaluate exact local RGB Jacobian actions from the forward compositor.

    The checked-in CUDA backward kernel is not a faithful derivative of the
    forward compositor for medium inputs on covered pixels. This path mirrors
    the detached forward equations and applies their closed-form chain rule
    directly to two directions. It avoids a full image Jacobian and any
    backward/JVP/VJP call while preserving the exact piecewise forward path.
    """
    xys, depths, radii, conics, colors, opacities, num_tiles_hit, _size, tile_bounds = _render_geometry(model, camera, height, width)
    num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)
    if num_intersects < 1:
        return tuple(torch.zeros((len(flat), 3), device=model.device, dtype=torch.float64) for _ in directions)
    _, _, isect_ids_sorted, gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(
        xys.shape[0], num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds, model.underwater_rasterizer.block_width
    )
    flat_tensor = flat.to(model.device, dtype=torch.long)
    raw_all = raw.detach().float().reshape(-1, 9)
    direction_all = [direction.detach().float().reshape(-1, 9) for direction in directions]
    action_parts: List[List[torch.Tensor]] = [[] for _ in directions]
    block_width = model.underwater_rasterizer.block_width
    for chunk_start in range(0, len(flat_tensor), ANALYTIC_ACTION_CHUNK_SIZE):
        chunk_flat = flat_tensor[chunk_start:chunk_start + ANALYTIC_ACTION_CHUNK_SIZE]
        rows = torch.div(chunk_flat, width, rounding_mode="floor")
        cols = chunk_flat.remainder(width)
        tile_ids = (rows // block_width) * tile_bounds[0] + cols // block_width
        starts = tile_bins[tile_ids, 0].long()
        lengths = (tile_bins[tile_ids, 1] - tile_bins[tile_ids, 0]).long()
        max_length = int(lengths.max().item()) if lengths.numel() else 0
        if max_length == 0:
            zero = torch.zeros((len(chunk_flat), 3), device=model.device, dtype=torch.float64)
            for part in action_parts:
                part.append(zero)
            continue
        offsets = torch.arange(max_length, device=model.device, dtype=torch.long)
        present = offsets[None, :] < lengths[:, None]
        gaussian_indices = (starts[:, None] + offsets[None, :]).clamp_max(len(gaussian_ids_sorted) - 1)
        gaussian_ids = gaussian_ids_sorted[gaussian_indices].long()
        center = xys[gaussian_ids]
        conic = conics[gaussian_ids]
        depth = depths[gaussian_ids]
        color = colors[gaussian_ids]
        opacity = opacities[gaussian_ids].squeeze(-1)
        delta_x = center[..., 0] - cols.float()[:, None]
        delta_y = center[..., 1] - rows.float()[:, None]
        sigma = 0.5 * (conic[..., 0] * delta_x.square() + conic[..., 2] * delta_y.square()) + conic[..., 1] * delta_x * delta_y
        alpha = torch.minimum(torch.full_like(sigma, 0.999), opacity * torch.exp(-sigma))
        raw_selected = raw_all[chunk_flat]
        medium = MI._activate_medium(model, raw_selected, len(chunk_flat), 1)
        medium_rgb = medium["medium_rgb"].reshape(-1, 3)
        medium_bs = medium["medium_bs"].reshape(-1, 3)
        medium_attn = medium["medium_attn"].reshape(-1, 3)
        min_attn = torch.minimum(torch.zeros_like(medium_attn[:, 0]), medium_attn.min(dim=-1).values)
        valid = present & (sigma >= 0.0) & (alpha * torch.exp(-min_attn[:, None] * depth) >= 1.0 / 255.0)
        factor = torch.where(valid, 1.0 - alpha, torch.ones_like(alpha))
        trans_before = torch.cat(
            [torch.ones((len(chunk_flat), 1), device=model.device), torch.cumprod(factor[:, :-1], dim=-1)],
            dim=-1,
        )
        stop = valid & (trans_before * (1.0 - alpha) <= 1e-4)
        prior_stop = torch.cat(
            [torch.zeros((len(chunk_flat), 1), device=model.device, dtype=torch.bool), torch.cumsum(stop[:, :-1].to(torch.int32), dim=-1) > 0],
            dim=-1,
        )
        contributes = valid & ~prior_stop & ~stop
        trans_factor = torch.where(contributes, 1.0 - alpha, torch.ones_like(alpha))
        trans_final = torch.cumprod(trans_factor, dim=-1)[:, -1]
        depth_for_prev = torch.where(contributes, depth, torch.zeros_like(depth))
        prev_before = torch.cat(
            [torch.zeros((len(chunk_flat), 1), device=model.device), torch.cummax(depth_for_prev[:, :-1], dim=-1).values],
            dim=-1,
        )
        prev_final = torch.cummax(depth_for_prev, dim=-1).values[:, -1]
        exp_attn = torch.exp(-medium_attn[:, None, :] * depth[..., None])
        exp_bs_prev = torch.exp(-medium_bs[:, None, :] * prev_before[..., None])
        exp_bs_depth = torch.exp(-medium_bs[:, None, :] * depth[..., None])
        vis = alpha * trans_before
        object_rgb = (contributes[..., None] * vis[..., None] * color * exp_attn).sum(dim=1)
        medium_factor = (contributes[..., None] * trans_before[..., None] * (exp_bs_prev - exp_bs_depth)).sum(dim=1)
        medium_factor = medium_factor + trans_final[:, None] * torch.exp(-medium_bs * prev_final[:, None])
        d_rgb = medium_rgb * (1.0 - medium_rgb)
        d_bs = torch.sigmoid(raw_selected[:, 3:6] + float(getattr(model, "medium_density_bias", 0.0)))
        d_attn = torch.sigmoid(raw_selected[:, 6:9] + float(getattr(model, "medium_density_bias", 0.0)))
        object_attn_derivative = -(contributes[..., None] * vis[..., None] * color * exp_attn * depth[..., None]).sum(dim=1)
        medium_bs_derivative = (
            contributes[..., None]
            * trans_before[..., None]
            * (-prev_before[..., None] * exp_bs_prev + depth[..., None] * exp_bs_depth)
        ).sum(dim=1)
        medium_bs_derivative = medium_bs_derivative - trans_final[:, None] * prev_final[:, None] * torch.exp(-medium_bs * prev_final[:, None])
        for part, direction in zip(action_parts, direction_all):
            direction_selected = direction[chunk_flat]
            effect = (
                medium_factor * d_rgb * direction_selected[:, :3]
                + medium_rgb * medium_bs_derivative * d_bs * direction_selected[:, 3:6]
                + object_attn_derivative * d_attn * direction_selected[:, 6:9]
            )
            if not bool(torch.isfinite(effect).all().item()):
                raise RuntimeError("Non-finite analytic local RGB Jacobian action")
            part.append(effect.detach().double())
        del raw_selected, medium, medium_factor, d_rgb, d_bs, d_attn, object_attn_derivative, medium_bs_derivative
    effects = [torch.cat(parts, dim=0) for parts in action_parts]
    del xys, depths, radii, colors, opacities, num_tiles_hit, cum_tiles_hit, isect_ids_sorted, gaussian_ids_sorted, tile_bins
    return tuple(effects)


def _base_row(population: str, source: str, ray_idx: int, flat: int, values: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "population": population,
        "source_view_id": source,
        "ray_index_within_bank": int(ray_idx),
        "flat_index": int(flat),
        "selected_mode_index": EXPECTED_MODE_INDEX,
        "selected_mode_label": EXPECTED_MODE,
    }
    row.update(values)
    return row


def _local_view(
    model: Any,
    camera: Any,
    batch: Mapping[str, Any],
    bundle: Mapping[str, Any],
    vector: torch.Tensor,
    scale: torch.Tensor,
    flat_indices: Sequence[int],
    labels: Mapping[Tuple[str, int], Mapping[str, Any]],
    source: str,
    population: str,
    measure_cost: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_base, _raw_natural, raw_effective, height, width = _raw_state(model, camera, bundle)
    delta_std, coeff_all, removed_raw = _projected_delta(raw_effective - raw_base, vector, scale)
    raw = raw_effective.detach().clone()
    cost_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)
    _sync_cuda()
    ordinary_forward_start = time.perf_counter()
    out_full = _render_no_grad(model, camera, raw, height, width)
    _sync_cuda()
    ordinary_forward_seconds = time.perf_counter() - ordinary_forward_start
    pred = out_full["pred_image"].reshape(-1, 3)
    flat = torch.tensor(list(flat_indices), device=model.device, dtype=torch.long)
    out_minus = _render_no_grad(model, camera, raw_base + removed_raw, height, width)
    actual = pred[flat].detach().double() - out_minus["pred_image"].reshape(-1, 3)[flat].detach().double()
    mode_direction = (vector.to(model.device, dtype=torch.float32) * scale.to(model.device, dtype=torch.float32)).reshape(1, 9).expand_as(raw)
    cam_direction = (delta_std.to(model.device, dtype=torch.float32) * scale.to(model.device, dtype=torch.float32).reshape(1, 9)).reshape_as(raw)
    jv, dcam = _analytic_local_jacobian_actions(
        model,
        camera,
        raw,
        (mode_direction, cam_direction),
        height,
        width,
        flat,
    )
    a = coeff_all[flat].double()
    dmode = a[:, None] * jv
    e_mode = torch.linalg.norm(dmode, dim=-1)
    e_cam = torch.linalg.norm(dcam, dim=-1)
    cosine = (dmode * dcam).sum(dim=-1) / (e_mode * e_cam + EPS)
    cosine = cosine.clamp(-1.0, 1.0)
    lcs = (e_mode / (e_cam + EPS)) * cosine.clamp_min(0.0)
    actual_norm = torch.linalg.norm(actual, dim=-1)
    fidelity_cos = (dmode * actual).sum(dim=-1) / (e_mode * actual_norm + EPS)
    fidelity_abs = torch.linalg.norm(dmode - actual, dim=-1)
    fidelity_rel = fidelity_abs / (actual_norm + EPS)
    depth = out_full["depth"].reshape(-1)[flat].detach().double()
    tau = out_full["tau_D"].reshape(-1, 3)[flat].detach().double().mean(dim=-1)
    _sync_cuda()
    elapsed = time.perf_counter() - cost_start
    peak = int(torch.cuda.max_memory_allocated(model.device)) if torch.cuda.is_available() else 0
    current = int(torch.cuda.memory_allocated(model.device)) if torch.cuda.is_available() else 0
    rows: List[Dict[str, Any]] = []
    for idx, flat_value in enumerate(flat_indices):
        key = (source, idx)
        target = labels.get(key, {})
        c_rgb = float(target.get("C_rgb_heldout", float("nan")))
        c_utility = float(target.get("C_utility_heldout", float("nan")))
        y_q1 = int(c_utility > 0.0 and c_rgb > 0.0) if math.isfinite(c_utility) and math.isfinite(c_rgb) else None
        y_q2 = int(c_utility > 0.0 and c_rgb < 0.0) if math.isfinite(c_utility) and math.isfinite(c_rgb) else None
        rows.append(_base_row(population, source, idx, int(flat_value), {
            "a_i": float(a[idx].item()),
            "A": float(a[idx].abs().item()),
            "S": float(torch.linalg.norm(jv[idx]).item()),
            "E_mode": float(e_mode[idx].item()),
            "E_cam": float(e_cam[idx].item()),
            "R": float((e_mode[idx] / (e_cam[idx] + EPS)).item()),
            "cos_i": float(cosine[idx].item()),
            "LCS": float(lcs[idx].item()),
            "depth": float(depth[idx].item()),
            "tau": float(tau[idx].item()),
            "C_utility": c_utility,
            "C_rgb": c_rgb,
            "Y_Q1": y_q1,
            "Y_RGB": int(c_rgb > 0.0) if math.isfinite(c_rgb) else None,
            "Y_Q2": y_q2,
            "J_finite": True,
            "J_shape": "3x9",
            "J_action_method": "exact_analytic_forward_compositor_action",
            "J_vjp_calls": 0,
            "d_actual_norm": float(actual_norm[idx].item()),
            "fidelity_cosine": float(fidelity_cos[idx].item()),
            "fidelity_abs_l2_error": float(fidelity_abs[idx].item()),
            "fidelity_relative_l2_error": float(fidelity_rel[idx].item()),
        }))
    cost = {
        "source_view_id": source,
        "population": population,
        "sampled_rays": len(flat_indices),
        "elapsed_seconds": elapsed,
        "seconds_per_1024_rays": elapsed * SAMPLES_PER_VIEW / max(len(flat_indices), 1),
        "backward_calls": 0,
        "jvp_calls": 0,
        "vjp_calls": 0,
        "forward_difference_render_calls": 0,
        "jacobian_forward_render_calls": 0,
        "analytic_compositor_chunk_calls": int(math.ceil(len(flat_indices) / float(ANALYTIC_ACTION_CHUNK_SIZE))),
        "ordinary_forward_seconds": ordinary_forward_seconds,
        "relative_overhead_vs_ordinary_forward": elapsed / max(ordinary_forward_seconds, EPS),
        "peak_memory_allocated_bytes": peak,
        "current_memory_allocated_bytes": current,
        "measure_cost": measure_cost,
    }
    del raw_base, raw_effective, delta_std, coeff_all, removed_raw, raw, out_full, out_minus, pred, mode_direction, cam_direction, dcam, jv, dmode
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, cost


def _fidelity_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for label, subset in [("all", list(rows)), ("nonzero_activation", [r for r in rows if float(r["A"]) > 0.0])]:
        out.append({
            "group": label,
            "count": len(subset),
            "fidelity_cosine_mean": _mean(float(r["fidelity_cosine"]) for r in subset),
            "fidelity_cosine_median": _median(float(r["fidelity_cosine"]) for r in subset),
            "fidelity_absolute_l2_error_mean": _mean(float(r["fidelity_abs_l2_error"]) for r in subset),
            "fidelity_relative_l2_error_mean": _mean(float(r["fidelity_relative_l2_error"]) for r in subset),
            "fidelity_relative_l2_error_median": _median(float(r["fidelity_relative_l2_error"]) for r in subset),
        })
    ordered = sorted(rows, key=lambda row: float(row["A"]))
    for decile in range(10):
        subset = ordered[(decile * len(ordered)) // 10: ((decile + 1) * len(ordered)) // 10]
        out.append({
            "group": "A_L%d" % (decile + 1),
            "count": len(subset),
            "fidelity_cosine_mean": _mean(float(r["fidelity_cosine"]) for r in subset),
            "fidelity_cosine_median": _median(float(r["fidelity_cosine"]) for r in subset),
            "fidelity_absolute_l2_error_mean": _mean(float(r["fidelity_abs_l2_error"]) for r in subset),
            "fidelity_relative_l2_error_mean": _mean(float(r["fidelity_relative_l2_error"]) for r in subset),
            "fidelity_relative_l2_error_median": _median(float(r["fidelity_relative_l2_error"]) for r in subset),
        })
    return out


def _decile_rows(rows: Sequence[Mapping[str, Any]], score_key: str = "LCS") -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[score_key]))
    output: List[Dict[str, Any]] = []
    for decile in range(10):
        subset = ordered[(decile * len(ordered)) // 10: ((decile + 1) * len(ordered)) // 10]
        stats = _row_stats(subset, score_key)
        stats.update({"decile": "L%d" % (decile + 1), "rank_low": decile * 10, "rank_high": (decile + 1) * 10})
        output.append(stats)
    return output


def _top_bottom_summary(rows: Sequence[Mapping[str, Any]], score_key: str = "LCS") -> Dict[str, Any]:
    top10 = _top_rows(rows, score_key, 0.10)
    top20 = _top_rows(rows, score_key, 0.20)
    bottom20 = _bottom_rows(rows, score_key, 0.20)
    return {
        "score": score_key,
        "top10": _row_stats(top10, score_key),
        "top20": _row_stats(top20, score_key),
        "bottom20": _row_stats(bottom20, score_key),
        "overall": _row_stats(rows, score_key),
        "top20_minus_bottom20_Q1_gap": _mean(float(r["Y_Q1"]) for r in top20) - _mean(float(r["Y_Q1"]) for r in bottom20),
        "top20_minus_bottom20_C_rgb": _mean(float(r["C_rgb"]) for r in top20) - _mean(float(r["C_rgb"]) for r in bottom20),
        "top20_minus_bottom20_C_utility": _mean(float(r["C_utility"]) for r in top20) - _mean(float(r["C_utility"]) for r in bottom20),
    }


def _precision_rows(rows: Sequence[Mapping[str, Any]], score_key: str = "LCS") -> List[Dict[str, Any]]:
    output = []
    for fraction in (0.10, 0.20, 0.30):
        subset = _top_rows(rows, score_key, fraction)
        output.append({
            "top_fraction": fraction,
            "count": len(subset),
            "Q1_precision": _mean(float(r["Y_Q1"]) for r in subset),
            "Q2_fraction": _mean(float(r["Y_Q2"]) for r in subset),
            "C_rgb_mean": _mean(float(r["C_rgb"]) for r in subset),
            "C_utility_mean": _mean(float(r["C_utility"]) for r in subset),
        })
    return output


def _camera_rows(rows: Sequence[Mapping[str, Any]], score_key: str = "LCS") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_view_id"]), []).append(row)
    output: List[Dict[str, Any]] = []
    for camera in sorted(grouped):
        subset = grouped[camera]
        metric = _score_metrics(subset, score_key)
        top20 = _top_rows(subset, score_key, 0.20)
        bottom20 = _bottom_rows(subset, score_key, 0.20)
        output.append({
            "source_view_id": camera,
            "sampled_rays": len(subset),
            "AUROC": metric["AUROC_Y_Q1"],
            "AUPRC": metric["AUPRC_Y_Q1"],
            "Q1_base_rate": metric["positive_base_rate"],
            "top20_Q1_fraction": _mean(float(r["Y_Q1"]) for r in top20),
            "bottom20_Q1_fraction": _mean(float(r["Y_Q1"]) for r in bottom20),
            "top20_C_rgb_mean": _mean(float(r["C_rgb"]) for r in top20),
            "bottom20_C_rgb_mean": _mean(float(r["C_rgb"]) for r in bottom20),
            "Spearman_LCS_C_rgb": metric["Spearman_score_C_rgb"],
            "top20_minus_bottom20_Q1_gap": metric["top20_Q1_gap"],
        })
    defined = [row for row in output if math.isfinite(float(row["AUROC"]))]
    better = [row for row in output if float(row["top20_Q1_fraction"]) > float(row["bottom20_Q1_fraction"])]
    robust = {
        "camera_count": len(output),
        "cameras_with_defined_AUROC": len(defined),
        "cameras_top20_Q1_gt_bottom20_Q1": len(better),
        "median_AUROC": _median(float(row["AUROC"]) for row in defined),
        "median_AUPRC_minus_base_rate": _median(float(row["AUPRC"]) - float(row["Q1_base_rate"]) for row in output if math.isfinite(float(row["AUPRC"]))),
        "median_top_bottom_Q1_gap": _median(float(row["top20_minus_bottom20_Q1_gap"]) for row in output),
        "median_Spearman_LCS_C_rgb": _median(float(row["Spearman_LCS_C_rgb"]) for row in output),
    }
    return output, robust


def _strata(rows: Sequence[Mapping[str, Any]], key: str, labels: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    thresholds = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    masks = [values <= thresholds[0], (values > thresholds[0]) & (values <= thresholds[1]), values > thresholds[1]]
    output = []
    for label, mask in zip(labels, masks):
        subset = [rows[index] for index in np.where(mask)[0]]
        metric = _score_metrics(subset, "LCS")
        output.append({
            "stratum": label,
            "count": len(subset),
            "threshold_key": key,
            "mean_value": _mean(float(row[key]) for row in subset),
            "Q1_base_rate": metric["positive_base_rate"],
            "AUROC": metric["AUROC_Y_Q1"],
            "AUPRC": metric["AUPRC_Y_Q1"],
            "top_bottom_Q1_gap": metric["top20_Q1_gap"],
            "Spearman_LCS_C_rgb": metric["Spearman_score_C_rgb"],
        })
    return output, {"key": key, "labels": list(labels), "threshold_1": float(thresholds[0]), "threshold_2": float(thresholds[1])}


def _eval_summary(rows: Sequence[Mapping[str, Any]], view_metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["LCS"]))
    deciles = []
    for decile in range(10):
        subset = ordered[(decile * len(ordered)) // 10: ((decile + 1) * len(ordered)) // 10]
        deciles.append({
            "decile": "L%d" % (decile + 1),
            "count": len(subset),
            "LCS_mean": _mean(float(r["LCS"]) for r in subset),
            "C_rgb_mean": _mean(float(r["C_rgb"]) for r in subset),
            "C_rgb_median": _median(float(r["C_rgb"]) for r in subset),
        })
    top = _top_rows(rows, "LCS", 0.20)
    bottom = _bottom_rows(rows, "LCS", 0.20)
    return {
        "view_count": len(view_metrics),
        "pixel_count": len(rows),
        "Spearman_LCS_C_rgb": _safe_spearman([float(r["LCS"]) for r in rows], [float(r["C_rgb"]) for r in rows]),
        "top20_C_rgb_mean": _mean(float(r["C_rgb"]) for r in top),
        "bottom20_C_rgb_mean": _mean(float(r["C_rgb"]) for r in bottom),
        "top20_minus_bottom20_C_rgb": _mean(float(r["C_rgb"]) for r in top) - _mean(float(r["C_rgb"]) for r in bottom),
        "deciles": deciles,
        "view_metrics": list(view_metrics),
    }


def _eval_array_summary(scores: Sequence[float], c_rgb: Sequence[float], view_metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    score_array = np.asarray(scores, dtype=np.float64)
    rgb_array = np.asarray(c_rgb, dtype=np.float64)
    order = np.argsort(score_array, kind="mergesort")
    deciles = []
    for decile in range(10):
        idx = order[(decile * len(order)) // 10: ((decile + 1) * len(order)) // 10]
        deciles.append({
            "decile": "L%d" % (decile + 1),
            "count": int(idx.size),
            "LCS_mean": float(score_array[idx].mean()) if idx.size else float("nan"),
            "C_rgb_mean": float(rgb_array[idx].mean()) if idx.size else float("nan"),
            "C_rgb_median": float(np.median(rgb_array[idx])) if idx.size else float("nan"),
        })
    top_count = max(1, int(math.ceil(len(order) * 0.20)))
    top = order[-top_count:]
    bottom = order[:top_count]
    return {
        "view_count": len(view_metrics),
        "pixel_count": int(score_array.size),
        "Spearman_LCS_C_rgb": _safe_spearman(score_array.tolist(), rgb_array.tolist()),
        "top20_C_rgb_mean": float(rgb_array[top].mean()) if top.size else float("nan"),
        "bottom20_C_rgb_mean": float(rgb_array[bottom].mean()) if bottom.size else float("nan"),
        "top20_minus_bottom20_C_rgb": float(rgb_array[top].mean() - rgb_array[bottom].mean()) if top.size and bottom.size else float("nan"),
        "deciles": deciles,
        "view_metrics": list(view_metrics),
    }


def _classification(general_summary: Mapping[str, Any], control_rows: Sequence[Mapping[str, Any]], fidelity: Mapping[str, Any], msafe: Mapping[str, Any], evaluation: Mapping[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
    lcs = next(row for row in control_rows if row["score"] == "LCS")
    amp = next(row for row in control_rows if row["score"] == "A")
    camera_count = int(general_summary["camera_robustness"]["cameras_top20_Q1_gt_bottom20_Q1"])
    median_auc = float(general_summary["camera_robustness"]["median_AUROC"])
    criteria = {
        "auroc_ge_0p60": float(lcs["AUROC_Y_Q1"]) >= 0.60,
        "auprc_ge_base_plus_0p05": float(lcs["AUPRC_Y_Q1"]) >= float(lcs["positive_base_rate"]) + 0.05,
        "top20_gap_ge_0p10": float(lcs["top20_Q1_gap"]) >= 0.10,
        "top20_rgb_gt_bottom20": float(lcs["top20_C_rgb_mean"]) > float(lcs["bottom20_C_rgb_mean"]),
        "top20_q2_not_gt_q1": float(lcs["top20_Q2_fraction"]) <= float(lcs["top20_Q1_fraction"]),
        "at_least_18_camera_gaps": camera_count >= 18,
        "median_camera_auroc_ge_0p55": median_auc >= 0.55,
        "lcs_better_than_amplitude": (
            float(lcs["AUROC_Y_Q1"]) > float(amp["AUROC_Y_Q1"])
            or float(lcs["AUPRC_Y_Q1"]) > float(amp["AUPRC_Y_Q1"])
            or float(lcs["top20_Q1_gap"]) > float(amp["top20_Q1_gap"])
        ),
        "first_order_fidelity_meaningful": float(fidelity["all"]["fidelity_cosine_mean"]) > 0.0 and math.isfinite(float(fidelity["all"]["fidelity_relative_l2_error_median"])),
        "msafe_or_eval_not_strongly_contradictory": (
            float(msafe.get("top20_minus_bottom20_C_rgb", 0.0)) >= 0.0
            or float(evaluation.get("top20_minus_bottom20_C_rgb", 0.0)) >= 0.0
        ),
    }
    if all(criteria.values()):
        primary = "LOCAL_CONTEXTUAL_SUPPORT_SUPPORTED"
        granularity = "RAY_CONTEXT_ADAPTIVE_CONTROL_SUPPORTED"
        reason = "All preregistered GENERAL criteria passed."
    elif float(lcs["AUROC_Y_Q1"]) > 0.5 and float(lcs["top20_Q1_gap"]) > 0.0:
        primary = "LOCAL_CONTEXTUAL_SUPPORT_TENTATIVE"
        granularity = "RAY_CONTEXT_ADAPTIVE_CONTROL_TENTATIVE"
        reason = "Pooled local signal is positive, but one or more robustness gates failed."
    else:
        primary = "LOCAL_CONTEXTUAL_SUPPORT_NOT_SUPPORTED"
        granularity = "RAY_CONTEXT_ADAPTIVE_CONTROL_NOT_SUPPORTED"
        reason = "LCS does not provide a sufficiently reliable held-out local support signal."
    return primary, granularity, reason, {"criteria": criteria, "lcs": dict(lcs), "amplitude": dict(amp)}


def _write_note(summary: Mapping[str, Any], path: Path) -> None:
    lcs = summary["predictor_summary"]["LCS"]
    fidelity = summary["first_order_fidelity"]["all"]
    msafe = summary["msafe_summary"]["pooled"]
    evaluation = summary["eval_summary"]
    runtime = summary["runtime_cost"]
    lines = [
        "# LOCAL CONTEXTUAL SUPPORT PREDICTOR PREFLIGHT",
        "",
        "## CODE FACT",
        "This was a read-only diagnostic using the frozen C1 step-14999 checkpoint, mode_01, basis, and held-out banks.",
        "No optimizer step, training, mode reselection, threshold fitting, classifier fitting, or projector change was performed.",
        "",
        "## MOTIVATION",
        "Global mode gating was rejected because the previous alignment audit found sparse context-dependent support and mixed full-view removal metrics.",
        "This preflight tests whether inference-available local model evidence can identify the supportive subset without using labels at inference time.",
        "",
        "## DEFINITION",
        "For local RGB Jacobian J_p in standardized raw medium coordinates, selected mode vector v_i, and camera residual Delta_z_cam_std:",
        "`a_i = v_i^T Delta_z_cam_std`, `d_i = a_i J_p v_i`, and `d_cam = J_p Delta_z_cam_std`.",
        "`E_mode = ||d_i||_2`, `E_cam = ||d_cam||_2`, `R = E_mode / (E_cam + 1e-12)`.",
        "The preregistered primary score is `LCS = R * max(cos_i, 0)` where `cos_i = dot(d_i, d_cam) / (||d_i||_2 ||d_cam||_2 + 1e-12)`.",
        "LCS uses only the current ray, camera context, frozen medium residual, and local Jacobian; it uses no GT RGB, camera-swap error, C_utility, or C_rgb at inference.",
        "",
        "## RESULT",
        "GENERAL Q1 base rate: `%s`." % lcs["positive_base_rate"],
        "GENERAL LCS AUROC: `%s`; AUPRC: `%s`; AUPRC minus base rate: `%s`." % (lcs["AUROC_Y_Q1"], lcs["AUPRC_Y_Q1"], float(lcs["AUPRC_Y_Q1"]) - float(lcs["positive_base_rate"])),
        "GENERAL LCS Spearman with C_rgb: `%s`." % lcs["Spearman_score_C_rgb"],
        "First-order fidelity cosine mean: `%s`; relative L2 error median: `%s`." % (fidelity["fidelity_cosine_mean"], fidelity["fidelity_relative_l2_error_median"]),
        "M_SAFE LCS AUROC: `%s`; top20-bottom20 Q1 gap: `%s`; top20-bottom20 C_rgb: `%s`." % (msafe["AUROC_Y_Q1"], summary["msafe_summary"]["top_bottom"]["top20_minus_bottom20_Q1_gap"], summary["msafe_summary"]["top_bottom"]["top20_minus_bottom20_C_rgb"]),
        "Eval LCS Spearman with C_rgb: `%s`; top20-bottom20 C_rgb: `%s` (eval is mixed and does not independently reproduce the GENERAL trend)." % (evaluation["Spearman_LCS_C_rgb"], evaluation["top20_minus_bottom20_C_rgb"]),
        "Local Jacobian action method: exact closed-form forward-compositor action with no full 3x9 Jacobian materialization and no backward/JVP/VJP calls; no optimizer step was run.",
        "GENERAL time per 1024 rays: `%s` seconds; relative overhead vs ordinary forward: `%s`x; peak allocated memory: `%s` bytes." % (runtime["general_seconds_per_1024_rays_mean"], runtime["relative_overhead_vs_ordinary_forward_mean"], runtime["general_peak_memory_allocated_bytes_max"]),
        "Primary classification: `%s`." % summary["primary_classification"],
        "Granularity classification: `%s`." % summary["granularity_classification"],
        "",
        "## NEXT TASK",
        "`%s`" % summary["next_task"],
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
    previous_dir = args.previous_output_dir.resolve()
    source_dir = args.source_output_dir.resolve()
    output_dir = args.output_dir.resolve()
    args.log_dir.resolve().mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _runtime_manifest()
    assets = _load_assets(previous_dir)
    _write_json(output_dir / "project_state.json", {
        "experiment": EXPERIMENT,
        "scene": SCENE,
        "read_only": True,
        "training_performed": False,
        "optimizer_step_count": 0,
        "mode_reselection": False,
        "hyperparameter_sweep": False,
        "projector_redesign": False,
        "previous_primary": "SPARSE_CONTEXT_DEPENDENT_SUPPORT",
        "previous_secondary": "GLOBAL_MODE_UTILITY_GATING_NOT_SUPPORTED",
        "repo": _repo_manifest(repo),
        "runtime": runtime,
    })
    selection = assets["selection"]
    checkpoint = Path(selection["checkpoint_path"])
    _write_json(output_dir / "frozen_probe_provenance.json", {
        "checkpoint_path": str(checkpoint),
        "checkpoint_step": FINAL_STEP,
        "checkpoint_sha256": assets["checkpoint_sha256"],
        "checkpoint_hash_match": assets["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "mode_label": selection["mode_label"],
        "mode_index": int(selection["mode_index"]),
        "sigma": float(selection["sigma"]),
        "g_obs": float(selection["g_obs"]),
        "mode_vector": selection["mode_vector"],
        "scale": selection["scale"],
        "general_bank_hash": assets["ray_bank"]["rows_hash"],
        "swap_bank_hash": assets["swap_bank"]["hash"],
        "general_bank_rows": len(assets["ray_bank"]["rows"]),
        "rays_per_camera": SAMPLES_PER_VIEW,
        "basis_population": "GENERAL",
        "basis_refit": False,
        "mode_reselection": False,
    })
    branch, bundle = PHASEA._load_c1_branch(repo, source_dir)
    model = branch.pipeline.model
    model.eval()
    before = _parameter_snapshot(model)
    vector = torch.tensor(selection["mode_vector"], dtype=torch.float64)
    scale = torch.tensor(selection["scale"], dtype=torch.float64)
    train_records, eval_records = PHASEA._camera_records(branch)
    general_rows: List[Dict[str, Any]] = []
    safe_rows: List[Dict[str, Any]] = []
    eval_scores: List[float] = []
    eval_c_rgb_values: List[float] = []
    cost_rows: List[Dict[str, Any]] = []
    fidelity_rows: List[Dict[str, Any]] = []
    eval_view_metrics: List[Dict[str, Any]] = []
    try:
        bank_by_view = {str(item["view_id"]): item for item in assets["ray_bank"]["rows"]}
        for view_id, (_idx, camera, batch) in train_records.items():
            bank = bank_by_view[view_id]
            general_flat = [int(value) for value in bank["heldout_general_flat"]]
            safe_flat = [int(value) for value in bank["heldout_safe_flat"]]
            rows, cost = _local_view(model, camera.to(model.device), batch, bundle, vector, scale, general_flat, assets["general_map"], view_id, "GENERAL", measure_cost=True)
            general_rows.extend(rows)
            cost_rows.append(cost)
            fidelity_rows.extend(rows)
            rows_safe, cost_safe = _local_view(model, camera.to(model.device), batch, bundle, vector, scale, safe_flat, assets["safe_map"], view_id, "M_SAFE")
            safe_rows.extend(rows_safe)
            cost_rows.append(cost_safe)
        for view_id, (_idx, camera, batch) in sorted(eval_records.items()):
            raw_base, _raw_natural, raw_effective, height, width = _raw_state(model, camera.to(model.device), bundle)
            delta_std, coeff_all, removed_raw = _projected_delta(raw_effective - raw_base, vector, scale)
            raw = raw_effective.detach().clone()
            _sync_cuda()
            started = time.perf_counter()
            ordinary_forward_start = time.perf_counter()
            out_full = _render_no_grad(model, camera.to(model.device), raw, height, width)
            _sync_cuda()
            ordinary_forward_seconds = time.perf_counter() - ordinary_forward_start
            pred = out_full["pred_image"].reshape(-1, 3)
            flat = torch.arange(height * width, device=model.device, dtype=torch.long)
            out_minus = _render_no_grad(model, camera.to(model.device), raw_base + removed_raw, height, width)
            gt = PHASEA._gt_for(model, batch, out_full["background"]).reshape(-1, 3).to(model.device)
            pred_minus = out_minus["pred_image"].reshape(-1, 3)
            c_rgb = (pred_minus - gt).square().mean(dim=-1) - (pred - gt).square().mean(dim=-1)
            mode_direction = (vector.to(model.device, dtype=torch.float32) * scale.to(model.device, dtype=torch.float32)).reshape(1, 9).expand_as(raw)
            cam_direction = (delta_std.to(model.device, dtype=torch.float32) * scale.to(model.device, dtype=torch.float32).reshape(1, 9)).reshape_as(raw)
            jv, dcam = _analytic_local_jacobian_actions(
                model,
                camera.to(model.device),
                raw,
                (mode_direction, cam_direction),
                height,
                width,
                flat,
            )
            a = coeff_all.double()
            dmode = a[:, None] * jv
            e_mode = torch.linalg.norm(dmode, dim=-1)
            e_cam = torch.linalg.norm(dcam, dim=-1)
            cos = ((dmode * dcam).sum(dim=-1) / (e_mode * e_cam + EPS)).clamp(-1.0, 1.0)
            lcs = (e_mode / (e_cam + EPS)) * cos.clamp_min(0.0)
            eval_lcs_cpu = lcs.detach().float().cpu().numpy()
            eval_rgb_cpu = c_rgb.detach().float().cpu().numpy()
            eval_scores.extend(eval_lcs_cpu.tolist())
            eval_c_rgb_values.extend(eval_rgb_cpu.tolist())
            with torch.no_grad():
                metric_full = MIC._metric_images(model, pred.reshape(height, width, 3), gt.reshape(height, width, 3))
                metric_minus = MIC._metric_images(model, pred_minus.reshape(height, width, 3), gt.reshape(height, width, 3))
            eval_view_metrics.append({
                "view_id": view_id,
                "pixel_count": height * width,
                "Spearman_LCS_C_rgb": _safe_spearman(eval_lcs_cpu.tolist(), eval_rgb_cpu.tolist()),
                "PSNR_full": metric_full["PSNR"], "SSIM_full": metric_full["SSIM"], "LPIPS_full": metric_full["LPIPS"], "MSE_full": metric_full["MSE"],
                "PSNR_minus": metric_minus["PSNR"], "SSIM_minus": metric_minus["SSIM"], "LPIPS_minus": metric_minus["LPIPS"], "MSE_minus": metric_minus["MSE"],
                "delta_PSNR_minus_full": metric_minus["PSNR"] - metric_full["PSNR"],
                "delta_SSIM_minus_full": metric_minus["SSIM"] - metric_full["SSIM"],
                "delta_LPIPS_minus_full": metric_minus["LPIPS"] - metric_full["LPIPS"],
                "delta_MSE_minus_full": metric_minus["MSE"] - metric_full["MSE"],
                "elapsed_seconds": time.perf_counter() - started,
                "backward_calls": 0,
                "vjp_calls": 0,
                "J_action_method": "exact_analytic_forward_compositor_action",
                "J_vjp_calls": 0,
                "ordinary_forward_seconds": ordinary_forward_seconds,
            })
            _sync_cuda()
            eval_elapsed = time.perf_counter() - started
            cost_rows.append({"source_view_id": view_id, "population": "EVAL", "sampled_rays": height * width, "elapsed_seconds": eval_elapsed, "seconds_per_1024_rays": eval_elapsed * SAMPLES_PER_VIEW / max(height * width, 1), "backward_calls": 0, "jvp_calls": 0, "vjp_calls": 0, "forward_difference_render_calls": 0, "jacobian_forward_render_calls": 0, "analytic_compositor_chunk_calls": int(math.ceil((height * width) / float(ANALYTIC_ACTION_CHUNK_SIZE))), "ordinary_forward_seconds": ordinary_forward_seconds, "relative_overhead_vs_ordinary_forward": eval_elapsed / max(ordinary_forward_seconds, EPS)})
            del raw_base, raw_effective, delta_std, coeff_all, removed_raw, raw, mode_direction, cam_direction, out_full, out_minus, pred, pred_minus, gt, dcam, jv, dmode
            gc.collect()
            torch.cuda.empty_cache()
        parameter_delta = _parameter_delta(before, model)
    finally:
        parameter_delta = _parameter_delta(before, model)
        PHASEA.OCMC._release(branch)
    if parameter_delta != 0.0:
        raise RuntimeError("Read-only parameter delta is nonzero: %s" % parameter_delta)
    _write_json(output_dir / "zero_training_audit.json", {"optimizer_step_count": 0, "parameter_delta_max": parameter_delta, "training_performed": False})
    _write_csv(output_dir / "per_ray_local_support.csv", general_rows + safe_rows)
    _write_json(output_dir / "per_ray_local_support.json", {"GENERAL": general_rows, "M_SAFE": safe_rows})
    fidelity = _fidelity_rows(fidelity_rows)
    _write_csv(output_dir / "local_jacobian_fidelity.csv", fidelity)
    _write_json(output_dir / "local_jacobian_fidelity.json", {"rows": fidelity, "jacobian_shape": "3x9", "standardized_coordinates": True, "finite_verified": True, "jacobian_action_method": "exact_analytic_forward_compositor_action"})
    controls = [_score_metrics(general_rows, key) for key in ("A", "S", "E_mode", "LCS")]
    control_map = {row["score"]: row for row in controls}
    camera_out, camera_robustness = _camera_rows(general_rows)
    lcs_deciles = _decile_rows(general_rows)
    top_bottom = _top_bottom_summary(general_rows)
    precision = _precision_rows(general_rows)
    safe_metric = _score_metrics(safe_rows, "LCS")
    safe_top_bottom = _top_bottom_summary(safe_rows)
    safe_camera, safe_robust = _camera_rows(safe_rows)
    depth_rows, depth_meta = _strata(general_rows, "depth", ("near", "middle", "far"))
    tau_rows, tau_meta = _strata(general_rows, "tau", ("low", "middle", "high_tau"))
    eval_summary = _eval_array_summary(eval_scores, eval_c_rgb_values, eval_view_metrics)
    mean_fidelity = {row["group"]: row for row in fidelity}
    primary, granularity, reason, classification_evidence = _classification(
        {"camera_robustness": camera_robustness}, controls, mean_fidelity, safe_top_bottom, eval_summary
    )
    predictor_summary = {
        "GENERAL": control_map["LCS"],
        "LCS": control_map["LCS"],
        "Q1_base_rate": control_map["LCS"]["positive_base_rate"],
        "precision_points": precision,
        "top_bottom": top_bottom,
        "camera_robustness": camera_robustness,
    }
    _write_csv(output_dir / "predictor_summary.csv", controls + precision)
    _write_json(output_dir / "predictor_summary.json", predictor_summary)
    _write_csv(output_dir / "predictor_control_comparison.csv", controls)
    _write_json(output_dir / "predictor_control_comparison.json", {"rows": controls, "primary_score": "LCS", "controls": ["A", "S", "E_mode"]})
    _write_csv(output_dir / "lcs_deciles.csv", lcs_deciles)
    _write_json(output_dir / "lcs_deciles.json", {"rows": lcs_deciles, "top_bottom": top_bottom, "precision": precision})
    _write_csv(output_dir / "per_camera_predictor.csv", camera_out)
    _write_json(output_dir / "per_camera_predictor.json", {"rows": camera_out, "summary": camera_robustness})
    _write_csv(output_dir / "msafe_predictor.csv", safe_camera)
    _write_json(output_dir / "msafe_predictor.json", {"pooled": safe_metric, "top_bottom": safe_top_bottom, "per_camera": safe_camera, "summary": safe_robust})
    _write_csv(output_dir / "depth_predictor.csv", depth_rows)
    _write_json(output_dir / "depth_predictor.json", {"rows": depth_rows, "provenance": depth_meta})
    _write_csv(output_dir / "tau_predictor.csv", tau_rows)
    _write_json(output_dir / "tau_predictor.json", {"rows": tau_rows, "provenance": tau_meta})
    _write_csv(output_dir / "eval_predictor.csv", eval_view_metrics)
    _write_json(output_dir / "eval_predictor.json", eval_summary)
    train_cost = [row for row in cost_rows if row["population"] == "GENERAL"]
    _write_json(output_dir / "runtime_cost.json", {
        "general_view_count": len(train_cost),
        "general_total_seconds": sum(float(row["elapsed_seconds"]) for row in train_cost),
        "general_seconds_per_1024_rays_mean": _mean(float(row["seconds_per_1024_rays"]) for row in train_cost),
        "general_backward_calls_total": sum(int(row["backward_calls"]) for row in train_cost),
        "general_jvp_calls_total": sum(int(row["jvp_calls"]) for row in train_cost),
        "general_vjp_calls_total": sum(int(row["vjp_calls"]) for row in train_cost),
        "general_peak_memory_allocated_bytes_max": max(int(row["peak_memory_allocated_bytes"]) for row in train_cost),
        "ordinary_forward_seconds_mean": _mean(float(row["ordinary_forward_seconds"]) for row in train_cost),
        "relative_overhead_vs_ordinary_forward_mean": _mean(float(row["relative_overhead_vs_ordinary_forward"]) for row in train_cost),
        "ordinary_forward_reference": "same-view no-grad forward render timed separately from the exact closed-form Jacobian action and the removal baseline.",
        "jacobian_action_method": "exact_analytic_forward_compositor_action",
        "jacobian_forward_render_calls": 0,
        "jacobian_backward_calls_per_view": 0,
        "jacobian_vjp_calls_per_view": 0,
        "analytic_compositor_chunk_calls_total": sum(int(row.get("analytic_compositor_chunk_calls", 0)) for row in train_cost),
        "cost_rows": cost_rows,
    })
    final = {
        "experiment": EXPERIMENT,
        "scene": SCENE,
        "primary_classification": primary,
        "granularity_classification": granularity,
        "classification_reason": reason,
        "classification_evidence": classification_evidence,
        "predictor_summary": predictor_summary,
        "first_order_fidelity": mean_fidelity,
        "msafe_summary": {"pooled": safe_metric, "top_bottom": safe_top_bottom, "camera_robustness": safe_robust},
        "depth_summary": {"rows": depth_rows, "provenance": depth_meta},
        "tau_summary": {"rows": tau_rows, "provenance": tau_meta},
        "eval_summary": eval_summary,
        "runtime": runtime,
        "runtime_cost": json.loads((output_dir / "runtime_cost.json").read_text(encoding="utf8")),
        "zero_training": {"optimizer_step_count": 0, "parameter_delta_max": parameter_delta},
        "next_task": "Implement a generic ray/context-adaptive capacity mechanism using global observability prior plus local contextual support, only if the primary classification is SUPPORTED.",
    }
    _write_json(output_dir / "final_classification.json", {"primary_classification": primary, "granularity_classification": granularity, "reason": reason, "evidence": classification_evidence})
    _write_json(output_dir / "final_summary.json", final)
    _write_note(final, RESEARCH_NOTE)
    print(json.dumps({"primary_classification": primary, "granularity_classification": granularity, "Q1_base_rate": control_map["LCS"]["positive_base_rate"], "AUROC_LCS": control_map["LCS"]["AUROC_Y_Q1"], "AUPRC_LCS": control_map["LCS"]["AUPRC_Y_Q1"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
