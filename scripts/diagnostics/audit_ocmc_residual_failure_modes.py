#!/usr/bin/env python3
"""Frozen four-scene residual failure-mode audit for formal OCMC states.

This script never constructs an optimizer update, calls backward, or mutates a
checkpoint.  Per-scene workers share one final-state render bank across all six
candidate diagnostics; the aggregate pass writes only compact statistics.
"""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.stats
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_m1_raoc_causal_scene as FORMAL
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAMERA
from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI

EXPERIMENT = "OCMC-RESIDUAL-FAILURE-MODE-AUDIT"
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "ocmc_residual_failure_audit_20260831"
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}
SCENES = tuple(SCENE_GPUS)
STEPS = (3000, 5000, 8000, 10000, 13000, 14999)
FINAL_STEP = 14999
RAYS_PER_VIEW = 16384
GAUSSIANS_PER_VIEW = 30000
PATCH_GRID = 8
EPS = 1e-12
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
PROTECTED = (
    "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
    "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
    "scripts/experiments/run_raoc_q50_q80_causal_scene.py",
    "scripts/experiments/run_raoc_q50_q80_causal_four_scene.py",
    "scripts/diagnostics/analyze_raoc_q50_q80_causal_four_scene.py",
)
CANDIDATE_NAMES = {
    "A": "Cross-View Intrinsic Inconsistency",
    "B": "Geometry-Medium Coupling",
    "C": "View-Dependent Residual Appearance",
    "D": "Spatially Structured Medium / RGB Residual",
    "E": "Late-Stage Gaussian Representation Allocation",
    "F": "Depth / Observability Conditioned Residual",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


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


def _write_table(root: Path, stem: str, rows: Sequence[Mapping[str, Any]], extra: Optional[Mapping[str, Any]] = None) -> None:
    _write_csv(root / f"{stem}.csv", rows)
    payload: Dict[str, Any] = {"rows": list(rows)}
    if extra:
        payload.update(extra)
    _write_json(root / f"{stem}.json", payload)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _run_text(command: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(list(command), cwd=str(cwd), text=True).strip()


def _file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*parts: str) -> int:
    data = "|".join(parts).encode("utf8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**31)


def _sample_indices(total: int, count: int, *seed_parts: str) -> np.ndarray:
    if total <= count:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(_stable_seed(*seed_parts))
    return np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))


def _finite_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    return a[valid], b[valid]


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _finite_pair(a, b)
    if a.size < 3 or np.ptp(a) <= EPS or np.ptp(b) <= EPS:
        return float("nan")
    return float(scipy.stats.spearmanr(a, b).statistic)


def _partial_rank_rho(x: np.ndarray, y: np.ndarray, controls: Sequence[np.ndarray]) -> float:
    arrays = [np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)] + [np.asarray(v).reshape(-1) for v in controls]
    valid = np.logical_and.reduce([np.isfinite(v) for v in arrays])
    if int(valid.sum()) < max(20, 4 + len(controls)):
        return float("nan")
    ranked = [scipy.stats.rankdata(v[valid]) for v in arrays]
    design = np.column_stack([np.ones(valid.sum())] + ranked[2:])
    rx = ranked[0] - design @ np.linalg.lstsq(design, ranked[0], rcond=None)[0]
    ry = ranked[1] - design @ np.linalg.lstsq(design, ranked[1], rcond=None)[0]
    return _rho(rx, ry)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, EPS))


def _predictiveness(score: np.ndarray, error: np.ndarray) -> Dict[str, float]:
    score, error = _finite_pair(score, error)
    if score.size < 20 or np.ptp(score) <= EPS or np.ptp(error) <= EPS:
        return {"sample_count": int(score.size), "spearman": float("nan"), "top_bottom_error_ratio": float("nan"), "auroc": float("nan"), "auprc": float("nan")}
    q_score = np.quantile(score, [0.2, 0.8])
    low = error[score <= q_score[0]]
    high = error[score >= q_score[1]]
    threshold = float(np.quantile(error, 0.8))
    target = error >= threshold
    try:
        auc = float(roc_auc_score(target, score))
        ap = float(average_precision_score(target, score))
    except ValueError:
        auc, ap = float("nan"), float("nan")
    return {
        "sample_count": int(score.size),
        "spearman": _rho(score, error),
        "top_bottom_error_ratio": _safe_ratio(float(np.mean(high)), float(np.mean(low))),
        "auroc": auc,
        "auprc": ap,
    }


def _stats(values: np.ndarray, prefix: str = "") -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return {f"{prefix}{key}": float("nan") for key in ("mean", "std", "p10", "p50", "p90", "p95", "p99")}
    out = {f"{prefix}mean": float(values.mean()), f"{prefix}std": float(values.std())}
    for q in (0.10, 0.50, 0.90, 0.95, 0.99):
        out[f"{prefix}p{int(q * 100)}"] = float(np.quantile(values, q))
    return out


def _tensor_stats(values: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    values = values.detach().float().reshape(-1).cpu().numpy()
    return _stats(values, prefix)


def _camera_context(model: Any, camera: Any) -> np.ndarray:
    return CAMERA._camera_context_for(model, camera.to(model.device), neutral=False).detach().float().cpu().numpy().reshape(3)


def _patch_vector(image: np.ndarray, grid: int = PATCH_GRID) -> np.ndarray:
    h, w = image.shape
    rows = []
    for ys in np.array_split(np.arange(h), grid):
        for xs in np.array_split(np.arange(w), grid):
            rows.append(float(image[np.ix_(ys, xs)].mean()))
    return np.asarray(rows, dtype=np.float64)


def _lag1_moran(image: np.ndarray) -> float:
    centered = image.astype(np.float64) - float(np.mean(image))
    denom = float(np.square(centered).sum())
    if denom <= EPS:
        return float("nan")
    horizontal = float((centered[:, :-1] * centered[:, 1:]).sum())
    vertical = float((centered[:-1, :] * centered[1:, :]).sum())
    edges = centered[:, :-1].size + centered[:-1, :].size
    return float(centered.size * (horizontal + vertical) / max(edges * denom, EPS))


def _pairwise_context_structure(rows: Sequence[Mapping[str, Any]], value_key: str) -> Dict[str, float]:
    n = len(rows)
    if n < 3:
        return {"camera_count": n, "distance_vs_abs_difference_rho": float("nan"), "nearest_over_all_abs_difference": float("nan"), "nearest_value_rho": float("nan")}
    contexts = np.asarray([row["camera_context"] for row in rows], dtype=np.float64)
    values = np.asarray([row[value_key] for row in rows], dtype=np.float64)
    distances = np.linalg.norm(contexts[:, None, :] - contexts[None, :, :], axis=-1)
    iu = np.triu_indices(n, 1)
    differences = np.abs(values[:, None] - values[None, :])
    nearest = np.argsort(np.where(np.eye(n, dtype=bool), np.inf, distances), axis=1)[:, 0]
    all_difference = float(np.mean(differences[iu]))
    nearest_difference = float(np.mean(np.abs(values - values[nearest])))
    return {
        "camera_count": n,
        "distance_vs_abs_difference_rho": _rho(distances[iu], differences[iu]),
        "nearest_over_all_abs_difference": _safe_ratio(nearest_difference, all_difference),
        "nearest_value_rho": _rho(values, values[nearest]),
    }


def _patch_context_structure(rows: Sequence[Mapping[str, Any]], patch_key: str = "patch_vector") -> Dict[str, float]:
    n = len(rows)
    if n < 3:
        return {"camera_count": n, "nearest_patch_correlation_mean": float("nan"), "all_pair_patch_correlation_mean": float("nan")}
    contexts = np.asarray([row["camera_context"] for row in rows], dtype=np.float64)
    patches = np.asarray([row[patch_key] for row in rows], dtype=np.float64)
    distances = np.linalg.norm(contexts[:, None, :] - contexts[None, :, :], axis=-1)
    nearest = np.argsort(np.where(np.eye(n, dtype=bool), np.inf, distances), axis=1)[:, 0]
    nearest_corr = [_rho(patches[i], patches[j]) for i, j in enumerate(nearest)]
    all_corr = [_rho(patches[i], patches[j]) for i in range(n) for j in range(i + 1, n)]
    return {
        "camera_count": n,
        "nearest_patch_correlation_mean": float(np.nanmean(nearest_corr)),
        "all_pair_patch_correlation_mean": float(np.nanmean(all_corr)),
    }


def _concat(parts: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=0) if parts else np.empty(0, dtype=np.float64)


def _lookup_archived_metrics(scene_dir: Path) -> Dict[Tuple[int, str, str], Dict[str, float]]:
    out: Dict[Tuple[int, str, str], Dict[str, float]] = {}
    for row in _read_csv(scene_dir / "per_view_eval.csv"):
        if row["branch"] != "C0":
            continue
        out[(int(row["absolute_step"]), row["split"], row["view_id"])] = {
            key: float(row[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")
        }
    return out


def _trajectory_persistence(scene_dir: Path, split: str) -> Dict[str, float]:
    metrics = _lookup_archived_metrics(scene_dir)
    ids = sorted({view for (step, candidate_split, view) in metrics if candidate_split == split and step == FINAL_STEP})
    correlations: List[float] = []
    overlaps: List[float] = []
    for step in (10000, 13000):
        common = [view for view in ids if (step, split, view) in metrics]
        if len(common) < 3:
            continue
        early = np.asarray([metrics[(step, split, view)]["MSE"] for view in common])
        final = np.asarray([metrics[(FINAL_STEP, split, view)]["MSE"] for view in common])
        correlations.append(_rho(early, final))
        n_top = max(1, int(math.ceil(len(common) * 0.2)))
        a = set(np.argsort(early)[-n_top:].tolist())
        b = set(np.argsort(final)[-n_top:].tolist())
        overlaps.append(len(a & b) / max(len(a | b), 1))
    return {
        "rank_persistence_mean": float(np.nanmean(correlations)) if correlations else float("nan"),
        "difficult_view_top20_jaccard_mean": float(np.nanmean(overlaps)) if overlaps else float("nan"),
    }


def _refinement_interval(scene_dir: Path, start: int, end: int) -> Dict[str, int]:
    rows = [row for row in _read_csv(scene_dir / "C0_refinement_events.csv") if start < int(row["absolute_step"]) <= end]
    return {
        "split_count": int(sum(int(row.get("K_split") or 0) for row in rows)),
        "duplicate_count": int(sum(int(row.get("K_duplicate") or 0) for row in rows)),
        "children_added": int(sum(int(row.get("children_added") or 0) for row in rows)),
        "pruned_count": int(sum(int(row.get("N_pruned") or 0) for row in rows)),
    }


def _topology_rows(branch: Any, scene: str, scene_dir: Path, eval_records: Sequence[Tuple[Any, ...]], archived: Mapping[Tuple[int, str, str], Mapping[str, float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    previous_step = 0
    representative = eval_records[0]
    for step in STEPS:
        checkpoint = scene_dir / "checkpoints" / "C0" / f"step-{step:09d}.ckpt"
        ckpt = FORMAL._load_checkpoint(branch, checkpoint)
        model = branch.pipeline.model
        with torch.no_grad():
            opacity = torch.sigmoid(model.opacities.detach().float())
            scales = torch.exp(model.scales.detach().float()).mean(dim=-1)
            outputs = model.get_outputs_for_camera(representative[2].to(model.device))
        radii = model.radii.detach().float()
        visible = radii > 0
        eval_values = [value for (s, split, _view), value in archived.items() if s == step and split == "eval"]
        refinements = _refinement_interval(scene_dir, previous_step, step)
        row: Dict[str, Any] = {
            "scene": scene,
            "absolute_step": step,
            "checkpoint_path": str(checkpoint),
            "refresh_step": int(ckpt.get("ocmc_bundle", {}).get("step", ckpt.get("refresh_step", -1))),
            "gaussian_count": int(model.means.shape[0]),
            "mean_opacity": float(opacity.mean().cpu()),
            "mean_scale": float(scales.mean().cpu()),
            "representative_view_id": representative[1],
            "visible_gaussian_count": int(visible.sum().cpu()),
            "mean_tau": float(outputs["tau_D"].detach().float().mean().cpu()),
            "mean_transmission": float(outputs["transmission"].detach().float().mean().cpu()),
            "mean_accumulation": float(outputs["accumulation"].detach().float().mean().cpu()),
            "mean_medium_rgb": float(outputs["medium_rgb"].detach().float().mean().cpu()),
            "mean_intrinsic_rendered_rgb": float(outputs["J_gaussian"].detach().float().mean().cpu()),
            "mean_visible_gaussian_intrinsic_rgb": float(outputs["gaussian_view_rgb"].detach().float()[visible].mean().cpu()),
            **_tensor_stats(opacity, "opacity_"),
            **_tensor_stats(scales, "scale_"),
            **_tensor_stats(radii[visible], "projected_radius_px_"),
            **refinements,
        }
        for key in ("PSNR", "SSIM", "LPIPS", "MSE"):
            row[key] = float(np.mean([value[key] for value in eval_values])) if eval_values else float("nan")
        if rows:
            delta_n = row["gaussian_count"] - rows[-1]["gaussian_count"]
            delta_psnr = row["PSNR"] - rows[-1]["PSNR"]
            delta_mse = row["MSE"] - rows[-1]["MSE"]
            row["delta_gaussian_count"] = delta_n
            row["delta_PSNR"] = delta_psnr
            row["delta_MSE"] = delta_mse
            row["delta_PSNR_per_100k_gaussians"] = delta_psnr / (delta_n / 100000.0) if delta_n else float("nan")
            row["delta_MSE_per_100k_gaussians"] = delta_mse / (delta_n / 100000.0) if delta_n else float("nan")
        rows.append(row)
        previous_step = step
        del outputs, ckpt
        gc.collect()
    return rows


def _bank_hash(rows: Sequence[Tuple[str, str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for split, view, indices in rows:
        digest.update(split.encode("utf8"))
        digest.update(view.encode("utf8"))
        digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _center_observations(
    model: Any,
    outputs: Mapping[str, torch.Tensor],
    error: np.ndarray,
    medium_delta: np.ndarray,
    scene: str,
    split: str,
    view_id: str,
) -> Dict[str, np.ndarray]:
    xy = model.xys.detach().float().cpu().numpy().reshape(-1, 2)
    radii = model.radii.detach().float().cpu().numpy().reshape(-1)
    n = min(xy.shape[0], radii.shape[0], int(model.means.shape[0]))
    xy, radii = xy[:n], radii[:n]
    h, w = error.shape
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(radii) & (radii > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    ids = np.flatnonzero(valid)
    if ids.size > GAUSSIANS_PER_VIEW:
        pick = _sample_indices(ids.size, GAUSSIANS_PER_VIEW, scene, split, view_id, "gaussian-centers")
        ids = ids[pick]
    px = np.clip(np.rint(xy[ids, 0]).astype(np.int64), 0, w - 1)
    py = np.clip(np.rint(xy[ids, 1]).astype(np.int64), 0, h - 1)
    scales = torch.exp(model.scales.detach().float()).mean(dim=-1).cpu().numpy()[:n]
    opacity = torch.sigmoid(model.opacities.detach().float()).reshape(-1).cpu().numpy()[:n]
    intrinsic = outputs["gaussian_view_rgb"].detach().float().mean(dim=-1).cpu().numpy()[:n]
    depths = outputs["projected_gaussian_depths"].detach().float().reshape(-1).cpu().numpy()[:n]
    return {
        "gaussian_id": ids,
        "error": error[py, px],
        "medium_delta": medium_delta[py, px],
        "depth": depths[ids],
        "scale": scales[ids],
        "opacity": opacity[ids],
        "radius": radii[ids],
        "intrinsic": intrinsic[ids],
    }


def _append_arrays(store: Dict[str, List[np.ndarray]], values: Mapping[str, np.ndarray]) -> None:
    for key, value in values.items():
        store.setdefault(key, []).append(np.asarray(value))


def _pooled(store: Mapping[str, Sequence[np.ndarray]]) -> Dict[str, np.ndarray]:
    return {key: _concat(values) for key, values in store.items()}


def _render_final_bank(
    branch: Any,
    scene: str,
    scene_dir: Path,
    archived: Mapping[Tuple[int, str, str], Mapping[str, float]],
) -> Dict[str, Any]:
    model = branch.pipeline.model
    train_records = FORMAL._train_records(branch.pipeline)
    eval_records = FORMAL._eval_records(branch.pipeline)
    n_gaussians = int(model.means.shape[0])
    color_sum = torch.zeros(n_gaussians, 3, dtype=torch.float64)
    color_sq_sum = torch.zeros(n_gaussians, 3, dtype=torch.float64)
    direction_sum = torch.zeros(n_gaussians, 3, dtype=torch.float64)
    visibility_count = torch.zeros(n_gaussians, dtype=torch.int32)
    ray_stores: Dict[str, Dict[str, List[np.ndarray]]] = {"train": {}, "eval": {}}
    geometry_stores: Dict[str, Dict[str, List[np.ndarray]]] = {"train": {}, "eval": {}}
    intrinsic_obs: Dict[str, Dict[str, List[np.ndarray]]] = {"train": {}, "eval": {}}
    view_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "eval": []}
    bank_rows: List[Tuple[str, str, np.ndarray]] = []
    safe_rows: List[Dict[str, Any]] = []
    locked_bank = _read_json(scene_dir / "calibration_bank.json")
    locked_safe = {str(row["view_id"]): np.asarray(row.get("M_SAFE_flat_pixel_indices", []), dtype=np.int64) for row in locked_bank.get("rows", [])}
    safe_hash = hashlib.sha256()

    def process(split: str, records: Sequence[Tuple[Any, ...]], collect_moments: bool) -> None:
        for ordinal, (_index, view_id, camera, batch) in enumerate(records):
            started = time.perf_counter()
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
            pred = outputs["pred_image"].detach().float().clamp(0, 1)
            residual_rgb = (pred - gt).cpu().numpy()
            error = np.mean(np.square(residual_rgb), axis=-1)
            abs_error = np.mean(np.abs(residual_rgb), axis=-1)
            h, w = error.shape
            indices = _sample_indices(h * w, RAYS_PER_VIEW, scene, split, view_id, "rays")
            bank_rows.append((split, view_id, indices))

            def flat(name: str, reduce: str = "mean") -> np.ndarray:
                value = outputs[name].detach().float().cpu().numpy()
                if value.ndim == 3 and value.shape[-1] > 1:
                    value = np.linalg.norm(value, axis=-1) if reduce == "norm" else value.mean(axis=-1)
                elif value.ndim == 3:
                    value = value[..., 0]
                return value.reshape(-1)[indices]

            camera_delta_image = np.linalg.norm(outputs["camera_medium_delta_projected_raw"].detach().float().cpu().numpy(), axis=-1)
            sampled = {
                "error": error.reshape(-1)[indices],
                "abs_error": abs_error.reshape(-1)[indices],
                "depth": flat("depth"),
                "accumulation": flat("accumulation"),
                "transmission": flat("transmission"),
                "tau": flat("tau_D"),
                "beta_B": flat("medium_bs"),
                "beta_D": flat("medium_attn"),
                "medium_rgb": flat("medium_rgb", "norm"),
                "medium_contribution": flat("rgb_medium", "norm"),
                "camera_residual": camera_delta_image.reshape(-1)[indices],
                "intrinsic_intensity": flat("J_gaussian"),
            }
            _append_arrays(ray_stores[split], sampled)
            center = _center_observations(model, outputs, error, camera_delta_image, scene, split, view_id)
            _append_arrays(geometry_stores[split], center)

            context = _camera_context(model, camera)
            metrics = dict(archived[(FINAL_STEP, split, view_id)])
            low_rgb = np.stack([scipy.ndimage.gaussian_filter(residual_rgb[..., channel], sigma=max(1.0, min(h, w) / 64.0), mode="reflect") for channel in range(3)], axis=-1)
            low_energy = float(np.square(low_rgb).sum())
            total_energy = float(np.square(residual_rgb).sum())
            patch = _patch_vector(error)
            medium_contribution_map = torch.linalg.norm(outputs["rgb_medium"].detach().float(), dim=-1).cpu().numpy()
            camera_residual_patch = _patch_vector(camera_delta_image)
            medium_contribution_patch = _patch_vector(medium_contribution_map)
            radii = model.radii.detach().float().cpu().numpy().reshape(-1)
            visible_radii = radii[radii > 0]
            row: Dict[str, Any] = {
                "scene": scene,
                "split": split,
                "view_id": view_id,
                **metrics,
                "mean_absolute_residual": float(abs_error.mean()),
                "p95_squared_residual": float(np.quantile(error, 0.95)),
                "p99_squared_residual": float(np.quantile(error, 0.99)),
                "mean_tau": float(outputs["tau_D"].detach().float().mean().cpu()),
                "mean_transmission": float(outputs["transmission"].detach().float().mean().cpu()),
                "mean_accumulation": float(outputs["accumulation"].detach().float().mean().cpu()),
                "mean_depth": float(outputs["depth"].detach().float().mean().cpu()),
                "visible_gaussian_count": int(visible_radii.size),
                "mean_projected_radius_px": float(visible_radii.mean()) if visible_radii.size else float("nan"),
                "camera_context": context.tolist(),
                "low_frequency_residual_energy_fraction": _safe_ratio(low_energy, total_energy),
                "residual_moran_lag1": _lag1_moran(error),
                "patch_vector": patch.tolist(),
                "camera_residual_moran_lag1": _lag1_moran(camera_delta_image),
                "camera_residual_patch_vector": camera_residual_patch.tolist(),
                "medium_contribution_moran_lag1": _lag1_moran(medium_contribution_map),
                "medium_contribution_patch_vector": medium_contribution_patch.tolist(),
                "render_seconds": time.perf_counter() - started,
                "height": h,
                "width": w,
            }
            view_rows[split].append(row)

            if scene == "IUI3-RedSea" and split == "train" and view_id in locked_safe:
                safe = locked_safe[view_id]
                safe = safe[(safe >= 0) & (safe < h * w)]
                safe_hash.update(view_id.encode("utf8"))
                safe_hash.update(safe.tobytes())
                if safe.size:
                    safe_rows.append({
                        "scene": scene,
                        "split": split,
                        "view_id": view_id,
                        "population": "M_SAFE",
                        "sample_count": int(safe.size),
                        "MSE": float(error.reshape(-1)[safe].mean()),
                        "mean_depth": float(outputs["depth"].detach().float().cpu().numpy().reshape(-1)[safe].mean()),
                        "mean_accumulation": float(outputs["accumulation"].detach().float().cpu().numpy().reshape(-1)[safe].mean()),
                        "provenance": "exact locked indices from source calibration_bank.json; no redefinition",
                    })

            visible = outputs["gaussian_visible_mask"].detach().bool().cpu().reshape(-1)
            colors = outputs["gaussian_view_rgb"].detach().float().cpu()
            n = min(n_gaussians, visible.numel(), colors.shape[0])
            ids = torch.where(visible[:n])[0]
            camera_position = camera.camera_to_worlds[0, :3, 3].detach().float().cpu()
            directions = camera_position[None, :] - model.means.detach().float().cpu()[:n]
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if collect_moments and ids.numel():
                c = colors[:n][ids].double()
                color_sum.index_add_(0, ids, c)
                color_sq_sum.index_add_(0, ids, c.square())
                direction_sum.index_add_(0, ids, directions[ids].double())
                visibility_count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))

            center_ids = center["gaussian_id"].astype(np.int64)
            if center_ids.size:
                obs = {
                    "gaussian_id": center_ids,
                    "error": center["error"],
                    "depth": center["depth"],
                    "radius": center["radius"],
                    "direction_x": directions[torch.from_numpy(center_ids), 0].numpy(),
                    "direction_y": directions[torch.from_numpy(center_ids), 1].numpy(),
                    "direction_z": directions[torch.from_numpy(center_ids), 2].numpy(),
                }
                _append_arrays(intrinsic_obs[split], obs)
            del outputs, gt, pred, low_rgb
            gc.collect()

    process("train", train_records, True)
    counts = visibility_count.double().clamp_min(1).reshape(-1, 1)
    color_mean = color_sum / counts
    color_variance = (color_sq_sum / counts - color_mean.square()).clamp_min(0).sum(dim=-1).sqrt()
    mean_direction = direction_sum / direction_sum.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    process("eval", eval_records, False)

    # Add train-derived intrinsic inconsistency and view-angle controls to both observation populations.
    intrinsic_pooled: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "eval"):
        obs = _pooled(intrinsic_obs[split])
        ids = obs.get("gaussian_id", np.empty(0, dtype=np.int64)).astype(np.int64)
        directions = np.column_stack([obs.get("direction_x", []), obs.get("direction_y", []), obs.get("direction_z", [])]) if ids.size else np.empty((0, 3))
        reference = mean_direction[torch.from_numpy(ids)].numpy() if ids.size else np.empty((0, 3))
        cosine = np.clip(np.sum(directions * reference, axis=1), -1.0, 1.0) if ids.size else np.empty(0)
        obs["intrinsic_variation"] = color_variance[torch.from_numpy(ids)].numpy() if ids.size else np.empty(0)
        obs["view_angle_from_train_mean"] = np.arccos(cosine) if ids.size else np.empty(0)
        obs["train_visibility_count"] = visibility_count[torch.from_numpy(ids)].numpy() if ids.size else np.empty(0)
        intrinsic_pooled[split] = obs

    return {
        "ray": {split: _pooled(ray_stores[split]) for split in ("train", "eval")},
        "geometry": {split: _pooled(geometry_stores[split]) for split in ("train", "eval")},
        "intrinsic": intrinsic_pooled,
        "views": view_rows,
        "bank_hash": _bank_hash(bank_rows),
        "bank_rows": [{"split": split, "view_id": view, "sample_count": int(indices.size), "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()} for split, view, indices in bank_rows],
        "safe_rows": safe_rows,
        "safe_hash": safe_hash.hexdigest() if safe_rows else None,
        "intrinsic_population": {
            "gaussian_count": n_gaussians,
            "visible_in_at_least_3_train_views": int((visibility_count >= 3).sum()),
            "visibility_observation_count": int(visibility_count.sum()),
        },
    }


def _candidate_a(scene: str, bank: Mapping[str, Any], model: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    sh_magnitude = model.features_rest.detach().float().reshape(model.features_rest.shape[0], -1).norm(dim=-1).cpu().numpy()
    scale = torch.exp(model.scales.detach().float()).mean(dim=-1).cpu().numpy()
    for split in ("train", "eval"):
        values = bank["intrinsic"][split]
        ids = values["gaussian_id"].astype(np.int64)
        valid = values["train_visibility_count"] >= 3
        score = values["intrinsic_variation"][valid]
        error = values["error"][valid]
        angle = values["view_angle_from_train_mean"][valid]
        depth = values["depth"][valid]
        radius = values["radius"][valid]
        pred = _predictiveness(score, error)
        summary = {
            "scene": scene,
            "candidate": "A",
            "row_type": "summary",
            "split": split,
            **pred,
            "partial_spearman_controlling_angle_depth_radius": _partial_rank_rho(score, error, [angle, depth, radius]),
            "intrinsic_variation_vs_view_angle_rho": _rho(score, angle),
            "intrinsic_variation_vs_depth_rho": _rho(score, depth),
            "intrinsic_variation_vs_SH_magnitude_rho": _rho(score, sh_magnitude[ids[valid]]),
            "intrinsic_variation_vs_scale_rho": _rho(score, scale[ids[valid]]),
            "eligible_observation_count": int(valid.sum()),
            "measurement": "train-view per-Gaussian RGB standard deviation; held-out residual sampled at projected Gaussian centers; pooled observations are visibility weighted because each visible view contributes one observation",
        }
        rows.append(summary)
        summaries[split] = summary
        if score.size:
            quantiles = np.quantile(score, np.linspace(0, 1, 6))
            for index in range(5):
                mask = (score >= quantiles[index]) & (score <= quantiles[index + 1] if index == 4 else score < quantiles[index + 1])
                rows.append({"scene": scene, "candidate": "A", "row_type": "quintile", "split": split, "quintile": index + 1, "sample_count": int(mask.sum()), "intrinsic_variation_mean": float(score[mask].mean()), "MSE_mean": float(error[mask].mean()), "view_angle_mean": float(angle[mask].mean())})
    evaluation = summaries["eval"]
    support = bool(evaluation["partial_spearman_controlling_angle_depth_radius"] >= 0.15 and evaluation["top_bottom_error_ratio"] >= 1.20 and evaluation["eligible_observation_count"] >= 1000)
    return rows, {"scene_support": support, "train_positive": bool(summaries["train"]["partial_spearman_controlling_angle_depth_radius"] >= 0.15), "held_out_positive": support, "primary_effect": evaluation["partial_spearman_controlling_angle_depth_radius"], "strongest_evidence": evaluation}


def _candidate_b(scene: str, bank: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "eval"):
        geometry = bank["geometry"][split]
        rays = bank["ray"][split]
        definitions: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
        for variable in ("depth", "scale", "opacity", "radius", "intrinsic"):
            definitions.append(("camera_residual", variable, geometry["medium_delta"], geometry[variable], geometry["error"]))
        definitions.extend(
            [
                ("camera_residual", "depth_ray", rays["camera_residual"], rays["depth"], rays["error"]),
                ("beta_D", "depth_ray", rays["beta_D"], rays["depth"], rays["error"]),
                ("beta_B", "accumulation", rays["beta_B"], rays["accumulation"], rays["error"]),
                ("tau", "intrinsic_intensity", rays["tau"], rays["intrinsic_intensity"], rays["error"]),
                ("medium_contribution", "accumulation", rays["medium_contribution"], rays["accumulation"], rays["error"]),
            ]
        )
        coupling: List[Dict[str, Any]] = []
        for medium_name, geometry_name, medium_values, geometry_values, error in definitions:
            rho = _rho(medium_values, geometry_values)
            error_rho = _rho(medium_values, error)
            item = {"medium_quantity": medium_name, "geometry_quantity": geometry_name, "spearman": rho, "medium_quantity_error_spearman": error_rho}
            coupling.append(item)
            rows.append({"scene": scene, "candidate": "B", "row_type": "coupling", "split": split, **item, "sample_count": int(min(medium_values.size, geometry_values.size, error.size))})
        strongest = max(coupling, key=lambda item: abs(item["spearman"]) if math.isfinite(item["spearman"]) else -1)
        summary = {"scene": scene, "candidate": "B", "row_type": "summary", "split": split, "strongest_medium_quantity": strongest["medium_quantity"], "strongest_coupling_variable": strongest["geometry_quantity"], "strongest_coupling_spearman": strongest["spearman"], "strongest_medium_quantity_error_spearman": strongest["medium_quantity_error_spearman"], "sample_count": int(rays["error"].size + geometry["error"].size), "caution": "association does not establish harmful coupling"}
        rows.append(summary)
        summaries[split] = summary
    train, evaluation = summaries["train"], summaries["eval"]
    same_pair = train["strongest_medium_quantity"] == evaluation["strongest_medium_quantity"] and train["strongest_coupling_variable"] == evaluation["strongest_coupling_variable"]
    same_error_direction = math.isfinite(train["strongest_medium_quantity_error_spearman"]) and train["strongest_medium_quantity_error_spearman"] * evaluation["strongest_medium_quantity_error_spearman"] > 0
    support = bool(same_pair and abs(evaluation["strongest_coupling_spearman"]) >= 0.30 and abs(evaluation["strongest_medium_quantity_error_spearman"]) >= 0.15 and same_error_direction)
    train_positive = bool(abs(train["strongest_coupling_spearman"]) >= 0.30 and abs(train["strongest_medium_quantity_error_spearman"]) >= 0.15)
    return rows, {"scene_support": support, "train_positive": train_positive, "held_out_positive": support, "primary_effect": abs(evaluation["strongest_medium_quantity_error_spearman"]), "strongest_evidence": evaluation}


def _candidate_c(scene: str, scene_dir: Path, bank: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "eval"):
        public_rows = []
        for item in bank["views"][split]:
            public = {key: value for key, value in item.items() if key not in ("patch_vector",)}
            public_rows.append(public)
            rows.append({"candidate": "C", "row_type": "view", **public})
        structure = _pairwise_context_structure(bank["views"][split], "MSE")
        persistence = _trajectory_persistence(scene_dir, split)
        mse = np.asarray([row["MSE"] for row in public_rows], dtype=np.float64)
        means = np.asarray([[row["mean_depth"], row["mean_tau"], row["mean_accumulation"], row["visible_gaussian_count"], row["mean_projected_radius_px"]] for row in public_rows], dtype=np.float64)
        context = np.asarray([row["camera_context"] for row in public_rows], dtype=np.float64)
        residualized_neighbor_ratio = float("nan")
        if len(public_rows) >= 4:
            design = np.column_stack([np.ones(len(public_rows)), np.column_stack([scipy.stats.rankdata(means[:, i]) for i in range(means.shape[1])])])
            target = scipy.stats.rankdata(mse)
            residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
            temp = [{"camera_context": context[i].tolist(), "residualized": residual[i]} for i in range(len(public_rows))]
            residualized_neighbor_ratio = _pairwise_context_structure(temp, "residualized")["nearest_over_all_abs_difference"]
        summary = {
            "scene": scene,
            "candidate": "C",
            "row_type": "summary",
            "split": split,
            "MSE_coefficient_of_variation": float(mse.std() / max(mse.mean(), EPS)),
            **structure,
            **persistence,
            "residualized_nearest_over_all_abs_difference": residualized_neighbor_ratio,
        }
        rows.append(summary)
        summaries[split] = summary
    evaluation = summaries["eval"]
    support = bool(evaluation["MSE_coefficient_of_variation"] >= 0.10 and evaluation["nearest_over_all_abs_difference"] <= 0.80 and summaries["train"]["rank_persistence_mean"] >= 0.50)
    return rows, {"scene_support": support, "train_positive": bool(summaries["train"]["nearest_over_all_abs_difference"] <= 0.80 and summaries["train"]["rank_persistence_mean"] >= 0.50), "held_out_positive": support, "primary_effect": 1.0 - evaluation["nearest_over_all_abs_difference"], "strongest_evidence": evaluation, "eval_camera_count": evaluation["camera_count"]}


def _candidate_d(scene: str, bank: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "eval"):
        for item in bank["views"][split]:
            rows.append({"scene": scene, "candidate": "D", "row_type": "view", "split": split, "view_id": item["view_id"], "low_frequency_residual_energy_fraction": item["low_frequency_residual_energy_fraction"], "residual_moran_lag1": item["residual_moran_lag1"], "camera_residual_moran_lag1": item["camera_residual_moran_lag1"], "medium_contribution_moran_lag1": item["medium_contribution_moran_lag1"]})
        structure = _patch_context_structure(bank["views"][split])
        camera_structure = _patch_context_structure(bank["views"][split], "camera_residual_patch_vector")
        medium_structure = _patch_context_structure(bank["views"][split], "medium_contribution_patch_vector")
        summary = {
            "scene": scene,
            "candidate": "D",
            "row_type": "summary",
            "split": split,
            **structure,
            "camera_residual_nearest_patch_correlation_mean": camera_structure["nearest_patch_correlation_mean"],
            "medium_contribution_nearest_patch_correlation_mean": medium_structure["nearest_patch_correlation_mean"],
            "mean_low_frequency_residual_energy_fraction": float(np.mean([row["low_frequency_residual_energy_fraction"] for row in bank["views"][split]])),
            "mean_residual_moran_lag1": float(np.mean([row["residual_moran_lag1"] for row in bank["views"][split]])),
            "mean_camera_residual_moran_lag1": float(np.mean([row["camera_residual_moran_lag1"] for row in bank["views"][split]])),
            "mean_medium_contribution_moran_lag1": float(np.mean([row["medium_contribution_moran_lag1"] for row in bank["views"][split]])),
        }
        rows.append(summary)
        summaries[split] = summary

    train = bank["views"]["train"]
    template_scores: List[float] = []
    template_errors: List[float] = []
    for evaluation in bank["views"]["eval"]:
        distances = [np.linalg.norm(np.asarray(evaluation["camera_context"]) - np.asarray(row["camera_context"])) for row in train]
        nearest = train[int(np.argmin(distances))]
        predicted = np.asarray(nearest["patch_vector"], dtype=np.float64)
        actual = np.asarray(evaluation["patch_vector"], dtype=np.float64)
        template_scores.extend(predicted.tolist())
        template_errors.extend(actual.tolist())
        rows.append({"scene": scene, "candidate": "D", "row_type": "eval_template", "split": "eval", "view_id": evaluation["view_id"], "nearest_train_view_id": nearest["view_id"], "patch_pattern_spearman": _rho(predicted, actual), "camera_context_distance": min(distances)})
    predict = _predictiveness(np.asarray(template_scores), np.asarray(template_errors))
    summary = {"scene": scene, "candidate": "D", "row_type": "held_out_predictiveness", "split": "eval", **predict, "score_definition": "nearest-train-camera normalized 8x8 residual patch template"}
    rows.append(summary)
    support = bool(predict["spearman"] >= 0.15 and predict["auroc"] >= 0.60 and summaries["eval"]["nearest_patch_correlation_mean"] >= 0.20)
    return rows, {"scene_support": support, "train_positive": bool(summaries["train"]["nearest_patch_correlation_mean"] >= 0.20), "held_out_positive": support, "primary_effect": predict["spearman"], "strongest_evidence": {**summaries["eval"], **predict}}


def _candidate_e(scene: str, topology: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [{"candidate": "E", "row_type": "checkpoint", **row} for row in topology]
    by_step = {int(row["absolute_step"]): row for row in topology}
    missing = sorted(set(STEPS) - set(by_step))
    if missing:
        raise RuntimeError(f"{scene} topology trajectory is missing {missing}; available={sorted(by_step)}")
    start, final = by_step[10000], by_step[14999]
    growth = (float(final["gaussian_count"]) - float(start["gaussian_count"])) / max(float(start["gaussian_count"]), 1.0)
    psnr_gain = float(final["PSNR"]) - float(start["PSNR"])
    mse_change = float(final["MSE"]) - float(start["MSE"])
    summary = {"scene": scene, "candidate": "E", "row_type": "summary", "late_stage_start": 10000, "late_stage_end": 14999, "late_gaussian_growth_fraction": growth, "late_PSNR_gain": psnr_gain, "late_MSE_change": mse_change, "large_growth": growth >= 0.10, "rgb_gain_saturated": psnr_gain <= 0.20, "specific_residual_regime_link_established": False}
    rows.append(summary)
    support = bool(summary["large_growth"] and summary["rgb_gain_saturated"] and summary["specific_residual_regime_link_established"])
    return rows, {"scene_support": support, "train_positive": bool(summary["large_growth"] and summary["rgb_gain_saturated"]), "held_out_positive": support, "primary_effect": growth, "strongest_evidence": summary}


def _candidate_f(scene: str, bank: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    variables = ("depth", "tau", "transmission", "accumulation")
    train = bank["ray"]["train"]
    evaluation = bank["ray"]["eval"]
    train_rhos = {variable: _rho(train[variable], train["error"]) for variable in variables}
    selected = max(variables, key=lambda key: abs(train_rhos[key]) if math.isfinite(train_rhos[key]) else -1)
    direction = 1.0 if train_rhos[selected] >= 0 else -1.0
    split_summaries: Dict[str, Dict[str, Any]] = {}
    for split, values in (("train", train), ("eval", evaluation)):
        for variable in variables:
            x, error = _finite_pair(values[variable], values["error"])
            quantiles = np.quantile(x, np.linspace(0, 1, 6))
            for index in range(5):
                mask = (x >= quantiles[index]) & (x <= quantiles[index + 1] if index == 4 else x < quantiles[index + 1])
                mse = float(error[mask].mean())
                rows.append({"scene": scene, "candidate": "F", "row_type": "quintile", "split": split, "variable": variable, "quintile": index + 1, "sample_count": int(mask.sum()), "MSE": mse, "MAE": float(values["abs_error"][: x.size][mask].mean()) if values["abs_error"].size == x.size else float("nan"), "PSNR_equivalent": float(-10.0 * math.log10(max(mse, EPS)))})
            rows.append({"scene": scene, "candidate": "F", "row_type": "association", "split": split, "variable": variable, "spearman_with_MSE": _rho(x, error)})
        score = direction * values[selected]
        predict = _predictiveness(score, values["error"])
        summary = {"scene": scene, "candidate": "F", "row_type": "summary", "split": split, "selected_regime_variable_from_train": selected, "high_error_direction": "high" if direction > 0 else "low", "raw_variable_spearman": _rho(values[selected], values["error"]), **predict}
        rows.append(summary)
        split_summaries[split] = summary

    # One preregistered-size interaction: selected factor plus strongest non-redundant factor.
    alternatives = [key for key in variables if key != selected and {key, selected} != {"tau", "transmission"}]
    second = max(alternatives, key=lambda key: abs(train_rhos[key]) if math.isfinite(train_rhos[key]) else -1)
    conditions = []
    for variable in (selected, second):
        rho = train_rhos[variable]
        q = float(np.quantile(evaluation[variable], 0.8 if rho >= 0 else 0.2))
        conditions.append(evaluation[variable] >= q if rho >= 0 else evaluation[variable] <= q)
    interaction = conditions[0] & conditions[1]
    ratio = _safe_ratio(float(evaluation["error"][interaction].mean()), float(evaluation["error"][~interaction].mean())) if interaction.any() and (~interaction).any() else float("nan")
    rows.append({"scene": scene, "candidate": "F", "row_type": "interaction", "split": "eval", "interaction": f"{selected}+{second}", "sample_count": int(interaction.sum()), "MSE_ratio_interaction_vs_other": ratio})
    for split in ("train", "eval"):
        radius = bank["geometry"][split]["radius"]
        error = bank["geometry"][split]["error"]
        quantiles = np.quantile(radius, np.linspace(0, 1, 6))
        for index in range(5):
            mask = (radius >= quantiles[index]) & (radius <= quantiles[index + 1] if index == 4 else radius < quantiles[index + 1])
            rows.append({"scene": scene, "candidate": "F", "row_type": "projected_footprint_quintile", "split": split, "variable": "projected_radius_px_at_gaussian_center", "quintile": index + 1, "sample_count": int(mask.sum()), "MSE": float(error[mask].mean()) if mask.any() else float("nan")})
        rows.append({"scene": scene, "candidate": "F", "row_type": "association", "split": split, "variable": "projected_radius_px_at_gaussian_center", "spearman_with_MSE": _rho(radius, error), "measurement_scope": "projected visible Gaussian centers; no approximate per-ray support reconstruction"})
    result = split_summaries["eval"]
    sign_consistent = split_summaries["train"]["raw_variable_spearman"] * result["raw_variable_spearman"] > 0
    support = bool(result["spearman"] >= 0.20 and result["auroc"] >= 0.60 and result["top_bottom_error_ratio"] >= 1.25 and sign_consistent)
    return rows, {"scene_support": support, "train_positive": bool(split_summaries["train"]["spearman"] >= 0.20), "held_out_positive": support, "primary_effect": result["spearman"], "strongest_evidence": {**result, "interaction": f"{selected}+{second}", "interaction_MSE_ratio": ratio}}


def run_scene(repo: Path, output_root: Path, scene: str, gpu: str) -> Dict[str, Any]:
    if scene not in SCENE_GPUS or SCENE_GPUS[scene] != str(gpu):
        raise RuntimeError(f"invalid scene/GPU assignment: {scene}/{gpu}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(gpu) or str(gpu) not in {"6", "7", "8", "9"}:
        raise RuntimeError(f"worker must expose exactly physical GPU {gpu}, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one CUDA device")
    scene_dir = SOURCE_ROOT / scene
    out_dir = output_root / "states" / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_cfg = FORMAL.SCENES[scene]
    runtime = {
        "scene": scene,
        "physical_gpu": str(gpu),
        "CUDA_VISIBLE_DEVICES": visible,
        "logical_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
    }
    started = time.perf_counter()
    branch = FORMAL._setup_branch(repo, scene_cfg, "C0")
    try:
        archived = _lookup_archived_metrics(scene_dir)
        train_records = FORMAL._train_records(branch.pipeline)
        eval_records = FORMAL._eval_records(branch.pipeline)
        topology = _topology_rows(branch, scene, scene_dir, eval_records, archived)
        _write_table(out_dir, "topology_trajectory", topology)
        final_path = scene_dir / "checkpoints" / "C0" / f"step-{FINAL_STEP:09d}.ckpt"
        ckpt = FORMAL._load_checkpoint(branch, final_path)
        config = branch.pipeline.model.config
        flags = {
            "intrinsic_color_parameterization": config.intrinsic_color_parameterization,
            "medium_context_mode": config.medium_context_mode,
            "camera_medium_observability_enabled": bool(config.camera_medium_observability_enabled),
            "camera_medium_ray_adaptive_observability_enabled": bool(config.camera_medium_ray_adaptive_observability_enabled),
            "camera_medium_observability_strength": float(config.camera_medium_observability_strength),
            "b_inf_mode": config.b_inf_mode,
            "infinite_water_enabled": bool(config.infinite_water_enabled),
            "medium_identifiability_enabled": bool(config.medium_identifiability_enabled),
        }
        if not flags["camera_medium_observability_enabled"] or flags["camera_medium_ray_adaptive_observability_enabled"]:
            raise RuntimeError(f"checkpoint is not frozen OCMC-only state: {flags}")
        bank = _render_final_bank(branch, scene, scene_dir, archived)
        candidate_rows: Dict[str, List[Dict[str, Any]]] = {}
        evidence: Dict[str, Dict[str, Any]] = {}
        candidate_rows["A"], evidence["A"] = _candidate_a(scene, bank, branch.pipeline.model)
        candidate_rows["B"], evidence["B"] = _candidate_b(scene, bank)
        candidate_rows["C"], evidence["C"] = _candidate_c(scene, scene_dir, bank)
        candidate_rows["D"], evidence["D"] = _candidate_d(scene, bank)
        candidate_rows["E"], evidence["E"] = _candidate_e(scene, topology)
        candidate_rows["F"], evidence["F"] = _candidate_f(scene, bank)
        for candidate, rows in candidate_rows.items():
            _write_table(out_dir, f"candidate_{candidate}", rows)
        _write_table(out_dir, "safe_population", bank["safe_rows"])
        checkpoint_manifest = {
            "scene": scene,
            "checkpoint_path": str(final_path),
            "absolute_step": FINAL_STEP,
            "source_config": str(branch.config_path),
            "gaussian_count": int(branch.pipeline.model.means.shape[0]),
            "train_camera_count": len(train_records),
            "eval_camera_count": len(eval_records),
            "image_resolutions": sorted({f"{int(record[2].width.item())}x{int(record[2].height.item())}" for record in list(train_records) + list(eval_records)}),
            "ocmc_bundle_present": ckpt.get("ocmc_bundle") is not None,
            "ocmc_refresh_step": int(ckpt.get("ocmc_bundle", {}).get("step", -1)),
            "state_provenance": "formal C0 branch of M1-RAOC four-scene causal experiment; C0 is OCMC enabled and RAOC disabled",
            "flags": flags,
        }
        diagnostic_manifest = {
            "scene": scene,
            "train_camera_count": len(train_records),
            "eval_camera_count": len(eval_records),
            "rays_per_view_cap": RAYS_PER_VIEW,
            "bank_sha256": bank["bank_hash"],
            "bank_rows": bank["bank_rows"],
            "intrinsic_population": bank["intrinsic_population"],
            "mask_population": "GENERAL plus exact locked M_SAFE supplemental train population" if scene == "IUI3-RedSea" else "GENERAL",
            "M_SAFE_sha256": bank["safe_hash"],
            "M_SAFE_provenance": str(scene_dir / "calibration_bank.json") if scene == "IUI3-RedSea" else None,
            "eval_M_SAFE": "not defined; locked protocol contains train-view indices only" if scene == "IUI3-RedSea" else "not applicable",
        }
        runtime["wall_seconds"] = time.perf_counter() - started
        result = {"scene": scene, "runtime": runtime, "checkpoint": checkpoint_manifest, "diagnostic_bank": diagnostic_manifest, "candidate_rows": candidate_rows, "candidate_evidence": evidence, "topology_rows": topology}
        _write_json(out_dir / "checkpoint_manifest.json", checkpoint_manifest)
        _write_json(out_dir / "diagnostic_bank_manifest.json", diagnostic_manifest)
        _write_json(out_dir / "scene_result.json", result)
        return result
    finally:
        FORMAL._release(branch)


def _classification(evidence: Sequence[Mapping[str, Any]]) -> Tuple[str, Dict[str, int]]:
    counts = {
        "scene_support_count": sum(bool(row.get("scene_support")) for row in evidence),
        "train_positive_count": sum(bool(row.get("train_positive")) for row in evidence),
        "held_out_positive_count": sum(bool(row.get("held_out_positive")) for row in evidence),
    }
    if counts["scene_support_count"] >= 3 and counts["held_out_positive_count"] >= 3:
        classification = "RESIDUAL_FAILURE_SUPPORTED"
    elif counts["scene_support_count"] == 2 or (counts["train_positive_count"] >= 3 and counts["held_out_positive_count"] >= 1):
        classification = "RESIDUAL_FAILURE_TENTATIVE"
    else:
        classification = "RESIDUAL_FAILURE_NOT_SUPPORTED"
    return classification, counts


def _priority(candidate: str, classification: str, counts: Mapping[str, int]) -> Dict[str, Any]:
    replication = int(counts["scene_support_count"])
    train = int(counts["train_positive_count"])
    held = int(counts["held_out_positive_count"])
    row = {
        "candidate": candidate,
        "name": CANDIDATE_NAMES[candidate],
        "classification": classification,
        "PERSISTENCE": min(3, train),
        "CROSS_SCENE_REPLICATION": min(3, replication),
        "HELD_OUT_RELEVANCE": min(3, held),
        "MECHANISTIC_CLARITY": {"A": 3, "B": 2, "C": 2, "D": 2, "E": 3, "F": 3}[candidate],
        "INDEPENDENCE_FROM_OCMC": {"A": 3, "B": 2, "C": 3, "D": 3, "E": 3, "F": 1}[candidate],
    }
    row["TOTAL"] = sum(int(row[key]) for key in ("PERSISTENCE", "CROSS_SCENE_REPLICATION", "HELD_OUT_RELEVANCE", "MECHANISTIC_CLARITY", "INDEPENDENCE_FROM_OCMC"))
    return row


def _primary_decision(classifications: Mapping[str, Mapping[str, Any]], priority: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    supported = [row for row in priority if row["classification"] == "RESIDUAL_FAILURE_SUPPORTED"]
    if supported:
        selected = max(supported, key=lambda row: (int(row["INDEPENDENCE_FROM_OCMC"]), int(row["TOTAL"]), -ord(str(row["candidate"]))))
        candidate = str(selected["candidate"])
        interventions = {
            "A": "At frozen render time, replace only the highest train-derived intrinsic-inconsistency Gaussian view residuals with their train-view mean while leaving geometry and OCMC unchanged.",
            "B": "At frozen render time, neutralize only the strongest measured geometry-coupled component of the camera-medium residual and compare matched controls.",
            "C": "Apply one frozen leave-one-camera-context residual correction estimated from neighboring train cameras and test held-out views without changing OCMC.",
            "D": "Apply one frozen low-frequency residual-template correction transferred from the nearest train camera and compare against a shuffled-camera template control.",
            "E": "At frozen state, prune only late-born Gaussians from the saturated interval and compare against a count-matched random prune control.",
            "F": "Clamp only the strongest residual-error observation regime to its adjacent train-derived quantile at frozen render time, with a count-matched random-ray control.",
        }
        return {
            "decision": "PRIMARY_RESIDUAL_FAILURE_IDENTIFIED",
            "primary_candidate": candidate,
            "primary_failure": CANDIDATE_NAMES[candidate],
            "selection_reason": "supported in at least three scenes with held-out relevance; selected by independence from OCMC then unweighted priority total",
            "minimal_causal_intervention": interventions[candidate],
            "next_task": f"FROZEN-{candidate}-MINIMAL-CAUSAL-INTERVENTION",
        }
    tentative = [row for row in priority if row["classification"] == "RESIDUAL_FAILURE_TENTATIVE"]
    strongest = max(tentative or list(priority), key=lambda row: int(row["TOTAL"]))
    return {
        "decision": "NO_SINGLE_DOMINANT_FAILURE_MODE",
        "primary_candidate": None,
        "primary_failure": None,
        "selection_reason": "no candidate met the supported, three-scene, held-out-relevance gate",
        "strongest_tentative_candidate": strongest["candidate"] if strongest["classification"] == "RESIDUAL_FAILURE_TENTATIVE" else None,
        "minimal_causal_intervention": None,
        "next_task": "FOCUSED-C-CAMERA-RESIDUAL-REPLICATION-DIAGNOSTIC" if strongest["candidate"] == "C" else f"FOCUSED-{strongest['candidate']}-RESIDUAL-DIAGNOSTIC",
    }


def _make_figures(root: Path, results: Sequence[Mapping[str, Any]], priority: Sequence[Mapping[str, Any]]) -> None:
    figure_dir = root / "aggregate" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(CANDIDATE_NAMES)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(candidates, [next(row["TOTAL"] for row in priority if row["candidate"] == candidate) for candidate in candidates], color=["#35618f", "#5c7c48", "#9b5a45", "#6f5b8c", "#777777", "#b08a2e"])
    ax.set_ylabel("Unweighted priority total")
    ax.set_ylim(0, 15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "priority_matrix.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for ax, result in zip(axes.flat, results):
        rows = result["topology_rows"]
        ax2 = ax.twinx()
        ax.plot([row["absolute_step"] for row in rows], [row["PSNR"] for row in rows], marker="o", color="#35618f", label="PSNR")
        ax2.plot([row["absolute_step"] for row in rows], [row["gaussian_count"] / 1e6 for row in rows], marker="s", color="#9b5a45", label="Gaussians")
        ax.set_title(result["scene"], fontsize=10)
        ax.set_ylabel("PSNR")
        ax2.set_ylabel("Gaussians (M)")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "topology_vs_psnr.png", dpi=160)
    plt.close(fig)


def _research_note(root: Path, classifications: Mapping[str, Mapping[str, Any]], priority: Sequence[Mapping[str, Any]], decision: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> str:
    by_scene = {result["scene"]: result for result in results}
    evidence = {candidate: {scene: by_scene[scene]["candidate_evidence"][candidate]["strongest_evidence"] for scene in SCENES} for candidate in CANDIDATE_NAMES}
    lines = [
        "# OCMC Residual Failure-Mode Audit (2026-08-31)",
        "",
        "## 1. Motivation",
        "",
        "This frozen-state audit asks which measurable failure remains after observability-controlled camera-conditioned medium context (OCMC). It does not design or train a new method.",
        "",
        "## 2. Locked OCMC Status",
        "",
        "CONFIG FACT: every audited state used bounded SH3 only as a controlled representation, `dir_xy_camera` scene-normalized camera-position context, OCMC enabled, and RAOC disabled.",
        "",
        "## 3. Why RAOC Was Closed",
        "",
        "EXPERIMENTAL FACT: the final hybrid CUDA feasibility attempt preserved tiny sensitivity error but failed the full modal reconstruction gate. RAOC remains scientifically archived and the formal line remains closed.",
        "",
        "## 4. Audit Protocol",
        "",
        "CODE FACT: all formal train and held-out eval cameras were rendered once at C0@14999. Deterministic per-view ray banks, projected Gaussian-center samples, normalized residual patches, and archived six-step topology trajectories were reused across candidates. No optimization, backward pass, or checkpoint mutation occurred.",
        "",
        "The exact audited train/eval camera counts were Curasao 18/3, IUI3-RedSea 25/4, JapaneseGradens-RedSea 17/3, and Panama 15/3. IUI3 reused the exact locked train-view M_SAFE indices; no eval-derived mask was defined.",
        "",
        "## 5. Candidate A: Cross-View Intrinsic Inconsistency",
        "",
        f"QUANTITATIVE RESULT: `{classifications['A']['classification']}` (0/4 held-out scene support). Held-out angle/depth/radius-controlled Spearman values were " + ", ".join(f"{scene} {evidence['A'][scene]['partial_spearman_controlling_angle_depth_radius']:.3f}" for scene in SCENES) + ".",
        "",
        "INFERENCE: intrinsic variation was strongly associated with SH magnitude, but it did not consistently enrich held-out RGB error. Only JapaneseGradens-RedSea had a raw held-out rho above 0.20, while the controlled effect remained 0.119. Candidate A is closed.",
        "",
        "## 6. Candidate B: Geometry-Medium Coupling",
        "",
        f"QUANTITATIVE RESULT: `{classifications['B']['classification']}` (2/4). IUI3 beta_D-depth coupling was rho {evidence['B']['IUI3-RedSea']['strongest_coupling_spearman']:.3f} with beta_D/error rho {evidence['B']['IUI3-RedSea']['strongest_medium_quantity_error_spearman']:.3f}; JapaneseGradens medium-contribution/accumulation coupling was rho {evidence['B']['JapaneseGradens-RedSea']['strongest_coupling_spearman']:.3f} with error rho {evidence['B']['JapaneseGradens-RedSea']['strongest_medium_quantity_error_spearman']:.3f}.",
        "",
        "INFERENCE: strong coupling exists in every scene for at least one pair, but only IUI3 and JapaneseGradens replicated the same pair across train/eval and also linked it to error. Correlation does not establish harmful coupling.",
        "",
        "## 7. Candidate C: View-Dependent Residual Appearance",
        "",
        f"QUANTITATIVE RESULT: `{classifications['C']['classification']}` (2/4). Eval-view MSE coefficients of variation were " + ", ".join(f"{scene} {evidence['C'][scene]['MSE_coefficient_of_variation']:.3f}" for scene in SCENES) + "; nearest-camera/all-pair error-difference ratios were " + ", ".join(f"{scene} {evidence['C'][scene]['nearest_over_all_abs_difference']:.3f}" for scene in SCENES) + ".",
        "",
        "INFERENCE: difficult views were late-stage persistent, but camera-neighbor structure replicated only in IUI3 and JapaneseGradens. Three or four eval cameras per scene leave substantial small-population uncertainty.",
        "",
        "## 8. Candidate D: Spatially Structured Medium / RGB Residual",
        "",
        f"QUANTITATIVE RESULT: `{classifications['D']['classification']}` (1/4 held-out predictive support). Eval RGB residual Moran-like values ranged from {min(evidence['D'][scene]['mean_residual_moran_lag1'] for scene in SCENES):.3f} to {max(evidence['D'][scene]['mean_residual_moran_lag1'] for scene in SCENES):.3f}, but nearest-train patch scores predicted held-out patch error only in IUI3.",
        "",
        "INFERENCE: individual residual maps and medium contributions are spatially structured, as expected, but similar structure was not consistently predictive of held-out error across views. This prevents a supported classification.",
        "",
        "## 9. Candidate E: Late-Stage Gaussian Representation Allocation",
        "",
        f"QUANTITATIVE RESULT: `{classifications['E']['classification']}` (0/4). From 10k to 14999, Gaussian populations changed by " + ", ".join(f"{scene} {100.0 * evidence['E'][scene]['late_gaussian_growth_fraction']:.2f}%" for scene in SCENES) + ".",
        "",
        "INFERENCE: every scene pruned 2.38-3.86% of its Gaussians instead of continuing large topology growth. The registered failure signature is absent, so Candidate E is closed.",
        "",
        "## 10. Candidate F: Depth / Observability Conditioned Residual",
        "",
        f"QUANTITATIVE RESULT: `{classifications['F']['classification']}` (2/4). IUI3 low-tau score achieved rho {evidence['F']['IUI3-RedSea']['spearman']:.3f}, AUROC {evidence['F']['IUI3-RedSea']['auroc']:.3f}, and top/bottom error ratio {evidence['F']['IUI3-RedSea']['top_bottom_error_ratio']:.2f}; JapaneseGradens low-depth achieved rho {evidence['F']['JapaneseGradens-RedSea']['spearman']:.3f}, AUROC {evidence['F']['JapaneseGradens-RedSea']['auroc']:.3f}, and ratio {evidence['F']['JapaneseGradens-RedSea']['top_bottom_error_ratio']:.2f}.",
        "",
        "INFERENCE: the effect is strong where present but the selected regime differs by scene and reverses or disappears in Curasao and Panama. It is also closest to OCMC observability, reducing independence.",
        "",
        "## 11. Cross-Scene Comparison",
        "",
        "QUANTITATIVE RESULT: " + "; ".join(f"{candidate}={classifications[candidate]['classification']} ({classifications[candidate]['scene_support_count']}/4)" for candidate in CANDIDATE_NAMES) + ".",
        "",
        "## 12. Priority Matrix",
        "",
        "| Candidate | Persistence | Cross-scene | Held-out | Clarity | OCMC independence | Total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in priority:
        lines.append(f"| {row['candidate']} | {row['PERSISTENCE']} | {row['CROSS_SCENE_REPLICATION']} | {row['HELD_OUT_RELEVANCE']} | {row['MECHANISTIC_CLARITY']} | {row['INDEPENDENCE_FROM_OCMC']} | {row['TOTAL']} |")
    lines.extend([
        "",
        "## 13. Primary Remaining Failure Mode",
        "",
        f"INFERENCE: `{decision['decision']}`. No primary failure is selected because no candidate reached supported status in at least 3/4 scenes with held-out relevance. Candidate C is the highest-priority tentative result at 11/15, not a supported mechanism.",
        "",
        "## 14. Closed Candidates",
        "",
        "Candidates A and E are formally closed as `RESIDUAL_FAILURE_NOT_SUPPORTED`. B, C, D, and F remain tentative only; none motivates module design yet.",
        "",
        "## 15. What Should Be Tested Next",
        "",
        f"HYPOTHESIS: the one next task is `{decision['next_task']}`: expand Candidate C coverage using deterministic leave-one-train-camera pseudo-held-out views plus all formal eval views, then retest difficult-view persistence and camera-neighbor structure while controlling depth, tau, accumulation, visibility, and footprint. This remains a focused diagnostic, not a module or training experiment.",
        "",
        "The audit stops before module design.",
    ])
    note = "\n".join(lines) + "\n"
    note_path = REPO_ROOT / "research_notes" / "OCMC_RESIDUAL_FAILURE_MODE_AUDIT_2026-08-31.md"
    note_path.write_text(note, encoding="utf8")
    return str(note_path)


def aggregate(output_root: Path) -> Dict[str, Any]:
    results = [_read_json(output_root / "states" / scene / "scene_result.json") for scene in SCENES]
    suffixes = {"A": "intrinsic", "B": "geometry_medium", "C": "view_residual", "D": "spatial_residual", "E": "topology", "F": "regime"}
    classifications: Dict[str, Dict[str, Any]] = {}
    cross_rows: List[Dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        rows = [row for result in results for row in result["candidate_rows"][candidate]]
        _write_table(output_root, f"candidate_{candidate}_{suffixes[candidate]}", rows)
        evidence = [result["candidate_evidence"][candidate] for result in results]
        classification, counts = _classification(evidence)
        payload = {"candidate": candidate, "name": CANDIDATE_NAMES[candidate], "classification": classification, **counts, "scene_evidence": {result["scene"]: result["candidate_evidence"][candidate] for result in results}}
        classifications[candidate] = payload
        for result in results:
            item = result["candidate_evidence"][candidate]
            cross_rows.append({"candidate": candidate, "scene": result["scene"], "scene_support": item["scene_support"], "train_positive": item["train_positive"], "held_out_positive": item["held_out_positive"], "primary_effect": item["primary_effect"]})
    priority = [_priority(candidate, classifications[candidate]["classification"], classifications[candidate]) for candidate in CANDIDATE_NAMES]
    decision = _primary_decision(classifications, priority)
    _write_table(output_root, "cross_scene_effects", cross_rows)
    _write_json(output_root / "candidate_classifications.json", classifications)
    _write_table(output_root, "priority_matrix", priority)
    _write_json(output_root / "primary_failure_mode.json", decision)
    _write_json(output_root / "ocmc_checkpoint_manifest.json", {"rows": [result["checkpoint"] for result in results]})
    _write_json(output_root / "diagnostic_bank_manifest.json", {"rows": [result["diagnostic_bank"] for result in results]})
    _make_figures(output_root, results, priority)
    note = _research_note(output_root, classifications, priority, decision, results)
    summary = {
        "experiment": EXPERIMENT,
        "locked_context": {"OCMC": "LOCKED MAIN MECHANISM", "RAOC_main_formal_line": "CLOSED", "RAOC_CUDA_hybrid": "NOT SUPPORTED", "pending_Q50_Q80": "CANCELLED", "BND_claim": "controlled representation setting only"},
        "scenes": list(SCENES),
        "candidate_classifications": classifications,
        "priority_matrix": priority,
        "primary_failure_mode": decision,
        "research_note": note,
        "no_training": True,
        "ocmc_frozen": True,
        "outputs_committed": False,
    }
    _write_json(output_root / "final_summary.json", summary)
    return summary


def _repo_manifest(repo: Path) -> Dict[str, Any]:
    return {
        "branch": _run_text(["git", "branch", "--show-current"], repo),
        "head": _run_text(["git", "rev-parse", "HEAD"], repo),
        "status_short": _run_text(["git", "status", "--short"], repo),
        "log_20": _run_text(["git", "log", "--oneline", "--decorate", "-20"], repo),
        "protected_files": {path: {"exists": (repo / path).exists(), "sha256": _file_sha256(repo / path)} for path in PROTECTED},
    }


def launch(repo: Path, output_root: Path) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "repo_state.json", _repo_manifest(repo))
    _write_json(output_root / "environment.json", {"python": sys.executable, "python_version": sys.version.split()[0], "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "visible_devices_launcher": torch.cuda.device_count(), "worker_policy": "exactly one of physical GPUs 6,7,8,9", "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "")})
    _write_json(output_root / "launcher_manifest.json", {"scene_gpu": SCENE_GPUS, "worker_script": str(Path(__file__).resolve()), "source_root": str(SOURCE_ROOT), "no_training": True})
    processes: Dict[str, subprocess.Popen[Any]] = {}
    handles: Dict[str, Any] = {}
    try:
        for scene, gpu in SCENE_GPUS.items():
            log = (output_root / "logs" / f"{scene}.log").open("w", encoding="utf8")
            handles[scene] = log
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [str(PYTHON), str(Path(__file__).resolve()), "--scene", scene, "--gpu", gpu, "--repo", str(repo), "--output-root", str(output_root)]
            processes[scene] = subprocess.Popen(command, cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        statuses = {scene: process.wait() for scene, process in processes.items()}
    finally:
        for handle in handles.values():
            handle.close()
    _write_json(output_root / "worker_status.json", statuses)
    if any(code != 0 for code in statuses.values()):
        raise RuntimeError(f"scene worker failure: {statuses}")
    summary = aggregate(output_root)
    ending = _repo_manifest(repo)
    _write_json(output_root / "repo_state_after.json", ending)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    repo, output_root = args.repo.resolve(), args.output_root.resolve()
    if args.scene:
        if args.gpu is None:
            raise ValueError("--scene requires --gpu")
        result = run_scene(repo, output_root, args.scene, args.gpu)
    elif args.aggregate_only:
        result = aggregate(output_root)
    else:
        result = launch(repo, output_root)
    if args.scene:
        printable = {"scene": result["scene"], "runtime": result["runtime"], "candidate_evidence": result["candidate_evidence"]}
    else:
        printable = result.get("primary_failure_mode", result)
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
