#!/usr/bin/env python3
"""Frozen cross-scene audit of low-training-view-support Gaussians.

Only locked OCMC checkpoints are loaded. This script never calls backward,
optimizer steps, refinement, or any parameter update.
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.diagnostics import audit_local_contextual_support_predictor_iui3 as LOCAL
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL
from scripts.experiments import run_ocmc_candidate_c_resplit_replication as REP

EXPERIMENT = "ISOLATE-FRACTION-VISIBLE-LOW-SUPPORT-PROXY"
SOURCE_ROOT = REPO_ROOT / "outputs" / "ocmc_candidate_c_resplit_replication_20260831"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "isolate_low_support_proxy_20260831"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "ISOLATE_LOW_SUPPORT_GAUSSIAN_PROXY_2026-08-31.md"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}
SCENES = tuple(SCENE_GPUS)
FORMAL_STEPS = (5000, 8000, 10000, 13000, 14999)
ALL_STEPS = (3000,) + FORMAL_STEPS
THRESHOLDS = (0, 1, 2, 3)
GROUPS = ("G0", "G1", "G2", "G3+")
GROUP_BOUNDS = {"G0": (0, 0), "G1": (1, 1), "G2": (2, 2), "G3+": (3, 32767)}
CONTROLS = (
    "mean_depth",
    "mean_tau",
    "mean_transmission",
    "mean_accumulation",
    "mean_footprint",
    "mean_opacity",
    "mean_scale",
    "visible_gaussian_count",
)
PROTECTED_HASHES = {
    "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py": "539f1c044f9ed136dce65b1dedc01746097cb2f3c4298c9682038019d23dfd7a",
    "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py": "fe3fd3ddcdbbff7904cfb7225a0ba024f928a9020777561252b66663c3c8ab32",
    "scripts/diagnostics/analyze_raoc_q50_q80_causal_four_scene.py": "b6a271372e68cd07fc566a3fde5ced5ba6463531278c31a6cfa47972aa15e8d6",
    "scripts/experiments/run_raoc_q50_q80_causal_four_scene.py": "d131428cc20ea76010e237abd91ac4cddfc5c6a78944c57c3317ed18bcdf60ef",
    "scripts/experiments/run_raoc_q50_q80_causal_scene.py": "3a924e88a606d34360a90348f3a392d0d12f80d43c98fe72b56cbec2d27ad6e7",
}
EPS = 1e-12
ACCUM_ATOL = 2e-6


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu()
        return cpu.item() if cpu.numel() == 1 else cpu.tolist()
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_table(
    root: Path, stem: str, rows: Sequence[Mapping[str, Any]], **metadata: Any
) -> None:
    _write_csv(root / f"{stem}.csv", rows)
    _write_json(root / f"{stem}.json", {"rows": list(rows), **metadata})


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: Sequence[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _quantile(values: Sequence[float], q: float) -> float:
    finite = _finite(values)
    return float(np.quantile(finite, q)) if finite.size else float("nan")


def _rho(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3 or np.ptp(a[valid]) <= EPS or np.ptp(b[valid]) <= EPS:
        return float("nan")
    return float(scipy.stats.spearmanr(a[valid], b[valid]).statistic)


def _rank_residualized_rho(
    predictor: Sequence[float], error: Sequence[float], control: Sequence[float]
) -> float:
    x = np.asarray(predictor, dtype=np.float64)
    y = np.asarray(error, dtype=np.float64)
    z = np.asarray(control, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(valid.sum()) < 4 or any(np.ptp(v[valid]) <= EPS for v in (x, y, z)):
        return float("nan")
    xr = scipy.stats.rankdata(x[valid]).astype(np.float64)
    yr = scipy.stats.rankdata(y[valid]).astype(np.float64)
    zr = scipy.stats.rankdata(z[valid]).astype(np.float64)
    design = np.column_stack((np.ones(zr.size), zr))
    x_res = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_res = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    return _rho(x_res, y_res)


def _stats(values: Sequence[float], prefix: str) -> Dict[str, Any]:
    finite = _finite(values)
    return {
        f"{prefix}_n": int(finite.size),
        f"{prefix}_mean": float(np.mean(finite)) if finite.size else float("nan"),
        f"{prefix}_median": _quantile(finite, 0.5),
        f"{prefix}_q10": _quantile(finite, 0.1),
        f"{prefix}_q90": _quantile(finite, 0.9),
    }


def _support_group_mask(support: torch.Tensor, group: str) -> torch.Tensor:
    lo, hi = GROUP_BOUNDS[group]
    return (support >= lo) & (support <= hi)


def _historical_visibility(
    model: Any, outputs: Mapping[str, Any], n_gaussians: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radii = model.radii.detach().reshape(-1)
    reported = outputs["gaussian_visible_mask"].detach().bool().reshape(-1)
    if radii.numel() != n_gaussians or reported.numel() != n_gaussians:
        raise RuntimeError(
            f"visibility shape mismatch: radii={radii.numel()} "
            f"reported={reported.numel()} n={n_gaussians}"
        )
    visible = radii > 0
    if not torch.equal(visible, reported):
        raise RuntimeError("radii > 0 differs from gaussian_visible_mask")
    depths = outputs["projected_gaussian_depths"].detach().reshape(-1)
    return visible, radii, depths


@torch.no_grad()
def _group_contribution_maps(
    model: Any, camera: Any, outputs: Mapping[str, Any], support: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], Dict[int, torch.Tensor], Dict[str, Any]]:
    geometry = LOCAL._render_geometry(
        model, camera, int(outputs["rgb"].shape[0]), int(outputs["rgb"].shape[1])
    )
    xys, depths, radii, conics, _colors, opacities, num_tiles_hit, size, _ = geometry
    height, width = int(size[0]), int(size[1])
    if support.numel() != radii.numel():
        raise RuntimeError("group contribution support/geometry shape mismatch")
    if not torch.equal(radii > 0, outputs["gaussian_visible_mask"].reshape(-1)):
        raise RuntimeError("reprojected visibility differs from formal output")
    zero_image = torch.zeros(height, width, 3, device=model.device)
    zero_background = torch.zeros(3, device=model.device)
    def render_indicator(mask: torch.Tensor, label: str) -> torch.Tensor:
        colors = mask.to(model.device, dtype=torch.float32)[:, None].expand(-1, 3)
        render = model.underwater_rasterizer.rasterize(
            xys=xys,
            xys_grad_abs=torch.zeros_like(xys),
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=colors,
            opacities=opacities,
            medium_rgb=zero_image,
            medium_bs=zero_image,
            medium_attn=zero_image,
            height=height,
            width=width,
            background=zero_background,
            step=model.step,
            force_white_background=False,
        )
        group_map = render.rgb_object[..., 0].detach()
        if not bool(torch.isfinite(group_map).all()):
            raise RuntimeError(f"non-finite standard RGB contribution map for {label}")
        if float((render.rgb_object - group_map[..., None]).abs().max()) != 0.0:
            raise RuntimeError(f"indicator RGB channels differ for {label}")
        return group_map

    maps = {
        group: render_indicator(_support_group_mask(support, group), group)
        for group in GROUPS
    }
    summed = sum(maps.values(), torch.zeros_like(next(iter(maps.values()))))
    accumulation = outputs["accumulation"].detach()[..., 0]
    difference = (summed - accumulation).abs()
    validation = {
        "finite": True,
        "standard_rgb_calls": len(GROUPS) + 1,
        "nd_cuda_path_used": False,
        "max_abs_accumulation_difference": float(difference.max()),
        "mean_abs_accumulation_difference": float(difference.mean()),
        "allclose_atol": ACCUM_ATOL,
        "allclose": bool(
            torch.allclose(summed, accumulation, atol=ACCUM_ATOL, rtol=ACCUM_ATOL)
        ),
    }
    if not validation["allclose"]:
        raise RuntimeError(
            f"support-group maps do not conserve formal accumulation: {validation}"
        )
    threshold_maps = {
        0: maps["G0"],
        1: maps["G0"] + maps["G1"],
        2: maps["G0"] + maps["G1"] + maps["G2"],
        3: render_indicator(support <= 3, "T3"),
    }
    return maps, threshold_maps, validation


def _checkpoint_path(scene: str, step: int) -> Path:
    return SOURCE_ROOT / scene / "checkpoints" / "C0" / f"step-{step:09d}.ckpt"


def _camera_row(
    scene: str,
    step: int,
    camera_id: str,
    model: Any,
    outputs: Mapping[str, Any],
    gt: torch.Tensor,
    support: torch.Tensor,
    group_maps: Mapping[str, torch.Tensor],
    threshold_maps: Mapping[int, torch.Tensor],
    checkpoint: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pred = outputs["pred_image"].detach().float().clamp(0, 1)
    gt = gt.detach().float().clamp(0, 1)
    metrics = MIC._metric_images(model, pred, gt)
    residual = (pred - gt).square().mean(dim=-1)
    visible, radii, _ = _historical_visibility(model, outputs, int(support.numel()))
    visible_cpu = visible.cpu()
    visible_support = support[visible_cpu].float()
    visible_ids = torch.where(visible)[0]
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)
    scale = torch.exp(model.scales.detach()).amax(dim=-1)
    total = sum(float(value.sum()) for value in group_maps.values())
    row: Dict[str, Any] = {
        "scene": scene,
        "absolute_step": step,
        "formal_checkpoint": step in FORMAL_STEPS,
        "camera_id": camera_id,
        "checkpoint_path": str(checkpoint),
        "E_cam": float(residual.mean()),
        "MSE": metrics["MSE"],
        "PSNR": metrics["PSNR"],
        "SSIM": metrics["SSIM"],
        "LPIPS": metrics["LPIPS"],
        "MAE": float((pred - gt).abs().mean()),
        "visible_gaussian_count": int(visible.sum()),
        "mean_train_support": (
            float(visible_support.mean()) if visible_support.numel() else float("nan")
        ),
        "median_train_support": (
            float(visible_support.median()) if visible_support.numel() else float("nan")
        ),
        "mean_depth": float(outputs["depth"].detach().float().mean()),
        "mean_tau": float(outputs["tau_D"].detach().float().mean()),
        "mean_transmission": float(outputs["transmission"].detach().float().mean()),
        "mean_accumulation": float(outputs["accumulation"].detach().float().mean()),
        "mean_footprint": (
            float(radii[visible].float().mean()) if bool(visible.any()) else float("nan")
        ),
        "mean_opacity": (
            float(opacity[visible_ids].mean()) if visible_ids.numel() else float("nan")
        ),
        "mean_scale": (
            float(scale[visible_ids].mean()) if visible_ids.numel() else float("nan")
        ),
        "mean_ocmc_projected_camera_residual": float(
            torch.linalg.vector_norm(
                outputs["camera_medium_delta_projected_raw"].detach().float(), dim=-1
            ).mean()
        ),
        "all_finite": True,
        "heldout_used_for_support": False,
        "parameter_update": False,
        "backward_called": False,
    }
    for threshold in THRESHOLDS:
        row[f"fraction_visible_support_le_{threshold}"] = (
            float((visible_support <= threshold).float().mean())
            if visible_support.numel()
            else float("nan")
        )
        numerator = float(threshold_maps[threshold].sum())
        row[f"cw_fraction_support_le_{threshold}"] = (
            numerator / total if total > EPS else float("nan")
        )
    for group in GROUPS:
        row[f"contribution_fraction_{group}"] = (
            float(group_maps[group].sum()) / total if total > EPS else float("nan")
        )

    cutoff = torch.quantile(residual.reshape(-1), 0.8)
    high = residual >= cutoff
    normal = ~high
    region_totals = {
        "high": sum(float(value[high].sum()) for value in group_maps.values()),
        "normal": sum(float(value[normal].sum()) for value in group_maps.values()),
    }
    enrichment_rows = []
    for group in GROUPS:
        high_fraction = (
            float(group_maps[group][high].sum()) / region_totals["high"]
            if region_totals["high"] > EPS
            else float("nan")
        )
        normal_fraction = (
            float(group_maps[group][normal].sum()) / region_totals["normal"]
            if region_totals["normal"] > EPS
            else float("nan")
        )
        enrichment_rows.append(
            {
                "scene": scene,
                "camera_id": camera_id,
                "absolute_step": step,
                "support_group": group,
                "high_residual_quantile": 0.8,
                "high_pixel_count": int(high.sum()),
                "normal_pixel_count": int(normal.sum()),
                "contribution_fraction_high": high_fraction,
                "contribution_fraction_normal": normal_fraction,
                "enrichment_ratio": (
                    high_fraction / normal_fraction
                    if normal_fraction > EPS
                    else float("nan")
                ),
                "diagnostic_uses_gt": True,
                "training_time_variable": False,
            }
        )
    return row, enrichment_rows


def _group_rows(
    scene: str,
    support: torch.Tensor,
    opacity: torch.Tensor,
    scale: torch.Tensor,
    sh_magnitude: torch.Tensor,
    heldout_visibility: torch.Tensor,
    footprint_sum: torch.Tensor,
    depth_sum: torch.Tensor,
    intrinsic_rgb_sum: torch.Tensor,
    contribution_sum: Mapping[str, float],
    heldout_camera_count: int,
) -> List[Dict[str, Any]]:
    n_gaussians = int(support.numel())
    total_contribution = sum(contribution_sum.values())
    rows = []
    for group in GROUPS:
        mask = _support_group_mask(support, group)
        count = int(mask.sum())
        visible = heldout_visibility[mask] > 0
        footprint = (
            footprint_sum[mask][visible] / heldout_visibility[mask][visible].float()
        )
        depth = depth_sum[mask][visible] / heldout_visibility[mask][visible].float()
        intrinsic_rgb = (
            intrinsic_rgb_sum[mask][visible]
            / heldout_visibility[mask][visible].float()
        )
        rows.append(
            {
                "scene": scene,
                "absolute_step": 14999,
                "support_group": group,
                "support_lower_inclusive": GROUP_BOUNDS[group][0],
                "support_upper_inclusive": GROUP_BOUNDS[group][1],
                "gaussian_count": count,
                "gaussian_fraction": count / n_gaussians,
                "heldout_visible_gaussian_count": int(visible.sum()),
                "heldout_contribution_fraction": (
                    contribution_sum[group] / total_contribution
                    if total_contribution > EPS
                    else float("nan")
                ),
                **_stats(
                    heldout_visibility[mask].numpy() / heldout_camera_count,
                    "heldout_visibility_frequency",
                ),
                **_stats(opacity[mask].numpy(), "opacity"),
                **_stats(scale[mask].numpy(), "scale"),
                **_stats(sh_magnitude[mask].numpy(), "sh_magnitude"),
                **_stats(intrinsic_rgb.numpy(), "intrinsic_rgb_magnitude"),
                **_stats(footprint.numpy(), "heldout_projected_footprint"),
                **_stats(depth.numpy(), "heldout_projected_depth"),
            }
        )
    return rows


def _stratification_bins(
    factors: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, Tuple[float, float]], List[Dict[str, Any]]]:
    bounds = {}
    rows = []
    for factor, values in factors.items():
        finite = values[torch.isfinite(values)]
        q1, q2 = torch.quantile(
            finite.float(), torch.tensor([1 / 3, 2 / 3])
        ).tolist()
        bounds[factor] = (q1, q2)
        for label, lo, hi in (
            ("low", -float("inf"), q1),
            ("middle", q1, q2),
            ("high", q2, float("inf")),
        ):
            mask = torch.isfinite(values) & (values > lo) & (values <= hi)
            rows.append(
                {
                    "stratification_factor": factor,
                    "bin": label,
                    "lower_bound": lo,
                    "upper_bound": hi,
                    "gaussian_count": int(mask.sum()),
                }
            )
    return bounds, rows


@torch.no_grad()
def _stratified_contribution_rows(
    scene: str,
    camera_id: str,
    model: Any,
    camera: Any,
    outputs: Mapping[str, Any],
    residual: torch.Tensor,
    support: torch.Tensor,
    factors: Mapping[str, torch.Tensor],
    bounds: Mapping[str, Tuple[float, float]],
) -> List[Dict[str, Any]]:
    geometry = LOCAL._render_geometry(
        model, camera, int(outputs["rgb"].shape[0]), int(outputs["rgb"].shape[1])
    )
    xys, depths, radii, conics, _colors, opacities, num_tiles_hit, size, _ = geometry
    height, width = int(size[0]), int(size[1])
    zero_image = torch.zeros(height, width, 3, device=model.device)
    zero_background = torch.zeros(3, device=model.device)
    high_error = residual >= torch.quantile(residual.reshape(-1), 0.8)
    accumulation = outputs["accumulation"].detach()[..., 0]
    region_totals = {
        "high": float(accumulation[high_error].sum()),
        "normal": float(accumulation[~high_error].sum()),
    }
    rows = []
    for factor, values in factors.items():
        q1, q2 = bounds[factor]
        for bin_label, lo, hi in (
            ("low", -float("inf"), q1),
            ("middle", q1, q2),
            ("high", q2, float("inf")),
        ):
            bin_mask = torch.isfinite(values) & (values > lo) & (values <= hi)
            maps = {}
            for support_label, support_mask in (
                ("low_support_s_le_1", support <= 1),
                ("high_support_s_ge_3", support >= 3),
            ):
                mask = (bin_mask & support_mask).to(
                    model.device, dtype=torch.float32
                )
                colors = mask[:, None].expand(-1, 3)
                render = model.underwater_rasterizer.rasterize(
                    xys=xys,
                    xys_grad_abs=torch.zeros_like(xys),
                    depths=depths,
                    radii=radii,
                    conics=conics,
                    num_tiles_hit=num_tiles_hit,
                    colors=colors,
                    opacities=opacities,
                    medium_rgb=zero_image,
                    medium_bs=zero_image,
                    medium_attn=zero_image,
                    height=height,
                    width=width,
                    background=zero_background,
                    step=model.step,
                    force_white_background=False,
                )
                maps[support_label] = render.rgb_object[..., 0]
            for support_label, contribution in maps.items():
                high = (
                    float(contribution[high_error].sum())
                    / region_totals["high"]
                    if region_totals["high"] > EPS
                    else float("nan")
                )
                normal = (
                    float(contribution[~high_error].sum())
                    / region_totals["normal"]
                    if region_totals["normal"] > EPS
                    else float("nan")
                )
                rows.append(
                    {
                        "scene": scene,
                        "camera_id": camera_id,
                        "absolute_step": 14999,
                        "stratification_factor": factor,
                        "fixed_quantile_bins": "scene heldout-visible tertiles",
                        "bin": bin_label,
                        "lower_bound": lo,
                        "upper_bound": hi,
                        "support_population": support_label,
                        "gaussian_count": int(
                            (
                                bin_mask
                                & (
                                    (support <= 1)
                                    if support_label == "low_support_s_le_1"
                                    else (support >= 3)
                                )
                            ).sum()
                        ),
                        "contribution_fraction_high_residual": high,
                        "contribution_fraction_normal_residual": normal,
                        "enrichment_ratio": high / normal if normal > EPS else float("nan"),
                        "diagnostic_uses_gt": True,
                    }
                )
    return rows


def worker(scene: str, assigned_gpu: str) -> Dict[str, Any]:
    if scene not in SCENES or assigned_gpu != SCENE_GPUS[scene]:
        raise RuntimeError(f"invalid scene/GPU assignment: {scene}/{assigned_gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != assigned_gpu:
        raise RuntimeError(f"worker must expose physical GPU {assigned_gpu} only")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"worker sees {torch.cuda.device_count()} devices instead of exactly one"
        )
    started = time.perf_counter()
    scene_dir = OUTPUT_ROOT / "workers" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_cfg = REP._scene_cfg(scene, SOURCE_ROOT)
    branch = FORMAL._setup_branch(REPO_ROOT, scene_cfg, "C0")
    try:
        model = branch.pipeline.model
        train_records = FORMAL._train_records(branch.pipeline)
        heldout_records = FORMAL._eval_records(branch.pipeline)
        train_ids = {row[1] for row in train_records}
        heldout_ids = {row[1] for row in heldout_records}
        if train_ids & heldout_ids:
            raise RuntimeError(f"train/heldout leakage in {scene}")

        source_checkpoint_rows = _read_json(
            SOURCE_ROOT / scene / "training_checkpoint_manifest.json"
        )["rows"]
        source_checkpoint_by_step = {
            int(row["absolute_step"]): row for row in source_checkpoint_rows
        }
        camera_rows: List[Dict[str, Any]] = []
        enrichment_rows: List[Dict[str, Any]] = []
        group_rows: List[Dict[str, Any]] = []
        stratified_rows: List[Dict[str, Any]] = []
        stratification_manifest: List[Dict[str, Any]] = []
        validation_rows: List[Dict[str, Any]] = []
        checkpoint_rows: List[Dict[str, Any]] = []
        visibility_checks = 0

        for step in ALL_STEPS:
            checkpoint = _checkpoint_path(scene, step)
            payload = FORMAL._load_checkpoint(branch, checkpoint)
            if (
                payload.get("experiment") != REP.EXPERIMENT
                or payload.get("branch") != "C0"
                or int(payload["absolute_step"]) != step
            ):
                raise RuntimeError(f"checkpoint provenance drift: {checkpoint}")
            if payload.get("raoc_state") is not None:
                raise RuntimeError(f"RAOC state unexpectedly present: {checkpoint}")
            if (
                not model.config.camera_medium_observability_enabled
                or model.config.camera_medium_ray_adaptive_observability_enabled
                or model.config.intrinsic_color_parameterization != "bounded_sh3"
                or model.config.rasterize_mode != "classic"
                or model.config.medium_context_mode != "dir_xy_camera"
                or int(model.config.sh_degree) != 3
            ):
                raise RuntimeError("locked checkpoint/model configuration drifted")
            n_gaussians = int(model.means.shape[0])
            support = torch.zeros(n_gaussians, dtype=torch.int16)
            for _index, _camera_id, camera, _batch in train_records:
                with torch.no_grad():
                    outputs = model.get_outputs_for_camera(camera.to(model.device))
                visible, _radii, _depths = _historical_visibility(
                    model, outputs, n_gaussians
                )
                support += visible.cpu().to(torch.int16)
                visibility_checks += 1
                del outputs
            if int(support.max()) > len(train_records):
                raise RuntimeError("support exceeds distinct training camera count")

            final = step == 14999
            heldout_visibility = (
                torch.zeros(n_gaussians, dtype=torch.int16) if final else None
            )
            footprint_sum = (
                torch.zeros(n_gaussians, dtype=torch.float32) if final else None
            )
            depth_sum = (
                torch.zeros(n_gaussians, dtype=torch.float32) if final else None
            )
            intrinsic_rgb_sum = (
                torch.zeros(n_gaussians, dtype=torch.float32) if final else None
            )
            contribution_sum = {group: 0.0 for group in GROUPS}
            for _index, camera_id, camera, batch in heldout_records:
                with torch.no_grad():
                    outputs = model.get_outputs_for_camera(camera.to(model.device))
                    gt = MI.PW._get_gt(model, batch, outputs["background"])
                visible, radii, projected_depths = _historical_visibility(
                    model, outputs, n_gaussians
                )
                visibility_checks += 1
                group_maps, threshold_maps, validation = _group_contribution_maps(
                    model, camera, outputs, support
                )
                validation_rows.append(
                    {
                        "scene": scene,
                        "absolute_step": step,
                        "camera_id": camera_id,
                        **validation,
                    }
                )
                row, per_camera_enrichment = _camera_row(
                    scene,
                    step,
                    camera_id,
                    model,
                    outputs,
                    gt,
                    support,
                    group_maps,
                    threshold_maps,
                    checkpoint,
                )
                camera_rows.append(row)
                if final:
                    enrichment_rows.extend(per_camera_enrichment)
                    assert (
                        heldout_visibility is not None
                        and footprint_sum is not None
                        and depth_sum is not None
                        and intrinsic_rgb_sum is not None
                    )
                    visible_cpu = visible.cpu()
                    heldout_visibility += visible_cpu.to(torch.int16)
                    footprint_sum += torch.where(
                        visible_cpu, radii.cpu().float(), 0.0
                    )
                    depth_sum += torch.where(
                        visible_cpu, projected_depths.cpu().float(), 0.0
                    )
                    view_rgb_magnitude = torch.linalg.vector_norm(
                        outputs["gaussian_view_rgb"].detach().float(), dim=-1
                    ).cpu()
                    intrinsic_rgb_sum += torch.where(
                        visible_cpu, view_rgb_magnitude, 0.0
                    )
                    for group in GROUPS:
                        contribution_sum[group] += float(group_maps[group].sum())
                del outputs, gt, group_maps, threshold_maps

            checkpoint_rows.append(
                {
                    "scene": scene,
                    "absolute_step": step,
                    "formal_checkpoint": step in FORMAL_STEPS,
                    "descriptive_only": step == 3000,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": source_checkpoint_by_step[step][
                        "checkpoint_sha256"
                    ],
                    "gaussian_count": n_gaussians,
                    "train_camera_count": len(train_records),
                    "heldout_camera_count": len(heldout_records),
                    "support_min": int(support.min()),
                    "support_max": int(support.max()),
                    "support_mean": float(support.float().mean()),
                    "ocmc_refresh_step": int(
                        (payload.get("ocmc_bundle") or {}).get("step", -1)
                    ),
                    "ocmc_enabled": True,
                    "raoc_enabled": False,
                    "frozen_read_only": True,
                }
            )

            if final:
                assert (
                    heldout_visibility is not None
                    and footprint_sum is not None
                    and depth_sum is not None
                    and intrinsic_rgb_sum is not None
                )
                opacity = (
                    torch.sigmoid(model.opacities.detach()).reshape(-1).cpu().float()
                )
                scale = (
                    torch.exp(model.scales.detach())
                    .amax(dim=-1)
                    .cpu()
                    .float()
                )
                sh_magnitude = (
                    torch.linalg.vector_norm(
                        model.features_rest.detach().reshape(n_gaussians, -1),
                        dim=-1,
                    )
                    .cpu()
                    .float()
                )
                group_rows = _group_rows(
                    scene,
                    support,
                    opacity,
                    scale,
                    sh_magnitude,
                    heldout_visibility,
                    footprint_sum,
                    depth_sum,
                    intrinsic_rgb_sum,
                    contribution_sum,
                    len(heldout_records),
                )
                heldout_seen = heldout_visibility > 0
                avg_footprint = torch.full_like(footprint_sum, float("nan"))
                avg_depth = torch.full_like(depth_sum, float("nan"))
                avg_footprint[heldout_seen] = (
                    footprint_sum[heldout_seen]
                    / heldout_visibility[heldout_seen].float()
                )
                avg_depth[heldout_seen] = (
                    depth_sum[heldout_seen]
                    / heldout_visibility[heldout_seen].float()
                )
                factors = {
                    "depth": avg_depth,
                    "scale": torch.where(
                        heldout_seen,
                        scale,
                        torch.full_like(scale, float("nan")),
                    ),
                    "opacity": torch.where(
                        heldout_seen,
                        opacity,
                        torch.full_like(opacity, float("nan")),
                    ),
                    "footprint": avg_footprint,
                }
                bounds, stratification_manifest = _stratification_bins(factors)
                for _index, camera_id, camera, batch in heldout_records:
                    with torch.no_grad():
                        outputs = model.get_outputs_for_camera(camera.to(model.device))
                        gt = MI.PW._get_gt(
                            model, batch, outputs["background"]
                        ).detach().float().clamp(0, 1)
                    pred = outputs["pred_image"].detach().float().clamp(0, 1)
                    residual = (pred - gt).square().mean(dim=-1)
                    stratified_rows.extend(
                        _stratified_contribution_rows(
                            scene,
                            camera_id,
                            model,
                            camera,
                            outputs,
                            residual,
                            support,
                            factors,
                            bounds,
                        )
                    )
                    del outputs, gt, pred, residual

            print(
                f"[{scene}] completed frozen checkpoint {step} "
                f"({n_gaussians} Gaussians)",
                flush=True,
            )
            del payload, support
            gc.collect()
            torch.cuda.empty_cache()

        result = {
            "experiment": EXPERIMENT,
            "scene": scene,
            "assigned_physical_gpu": assigned_gpu,
            "logical_gpu": 0,
            "gpu_name": torch.cuda.get_device_properties(0).name,
            "train_ids": sorted(train_ids),
            "heldout_ids": sorted(heldout_ids),
            "heldout_leakage": False,
            "checkpoint_rows": checkpoint_rows,
            "camera_rows": camera_rows,
            "enrichment_rows": enrichment_rows,
            "group_rows": group_rows,
            "stratified_rows": stratified_rows,
            "stratification_manifest": stratification_manifest,
            "contribution_validation_rows": validation_rows,
            "visibility_equivalence_checks": visibility_checks,
            "optimizer_step_called": False,
            "backward_called": False,
            "training_performed": False,
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json(scene_dir / "scene_result.json", result)
        return result
    finally:
        FORMAL._release(branch)


def _starting_repo_state() -> Dict[str, Any]:
    protected = {}
    for relative, expected in PROTECTED_HASHES.items():
        path = REPO_ROOT / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"protected file hash drift: {relative}: {actual}")
        protected[relative] = {
            "exists": True,
            "sha256": actual,
            "expected_sha256": expected,
            "untouched": True,
        }
    branch = _run_text(["git", "branch", "--show-current"])
    head = _run_text(["git", "rev-parse", "HEAD"])
    expected_head = "a7b93df54ad438d5c018344ad58d00dd33d437c4"
    if branch != "research/m1-bounded-intrinsic" or head != expected_head:
        raise RuntimeError(f"unexpected starting branch/HEAD: {branch} {head}")
    return {
        "starting_branch": branch,
        "starting_head": head,
        "remote_ref": "origin/research/m1-bounded-intrinsic",
        "protected_files": protected,
        "protected_files_untracked_at_start": True,
        "historical_gmvc_untouched": True,
        "q50_q80_untouched": True,
        "raoc_closed": True,
    }


def preflight() -> Dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    repo_state = _starting_repo_state()
    environment = {
        "python": str(PYTHON),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "launcher_visible_device_count": torch.cuda.device_count(),
        "allowed_physical_gpus": [6, 7, 8, 9],
        "worker_policy": "one assigned physical GPU exposed as logical cuda:0",
    }
    if (
        environment["python_version"] != "3.8.20"
        or environment["torch"] != "2.1.2+cu118"
        or not environment["cuda_available"]
    ):
        raise RuntimeError(f"environment drift: {environment}")

    checkpoint_rows = []
    for scene in SCENES:
        source_rows = _read_json(
            SOURCE_ROOT / scene / "training_checkpoint_manifest.json"
        )["rows"]
        source_by_step = {int(row["absolute_step"]): row for row in source_rows}
        locked_config = _read_json(
            SOURCE_ROOT / scene / "config" / "locked_config.json"
        )
        required_config = {
            "seed": 42,
            "intrinsic_color_parameterization": "bounded_sh3",
            "sh_degree": 3,
            "rasterize_mode": "classic",
            "medium_context_mode": "dir_xy_camera",
            "camera_medium_observability_enabled": True,
            "camera_medium_ray_adaptive_observability_enabled": False,
        }
        drift = {
            key: (locked_config.get(key), expected)
            for key, expected in required_config.items()
            if locked_config.get(key) != expected
        }
        if drift:
            raise RuntimeError(f"locked config drift for {scene}: {drift}")
        for step in ALL_STEPS:
            path = _checkpoint_path(scene, step)
            if not path.is_file() or step not in source_by_step:
                raise RuntimeError(f"missing checkpoint {scene}/{step}")
            actual_hash = _sha256(path)
            expected_hash = source_by_step[step]["checkpoint_sha256"]
            if actual_hash != expected_hash:
                raise RuntimeError(f"checkpoint hash drift: {path}")
            checkpoint_rows.append(
                {
                    "scene": scene,
                    "absolute_step": step,
                    "formal_checkpoint": step in FORMAL_STEPS,
                    "descriptive_only": step == 3000,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": actual_hash,
                    "source_manifest_sha256": expected_hash,
                    "locked_config_sha256": locked_config[
                        "config_lock_sha256"
                    ],
                    "present": True,
                }
            )

    cleanup_manifest = _read_json(
        Path("/tmp/outputs_cleanup_manifest_20260831.json")
    )
    approved = [
        row
        for row in cleanup_manifest["proposed_deletions"]
        if row["deletion_approved_by_task_policy"]
    ]
    if [row["path"] for row in approved] != [
        "outputs/m1_raoc_causal_four_scene_20260827_attempt1_oom"
    ]:
        raise RuntimeError("cleanup manifest differs from reviewed deletion")
    disk_before = {
        "captured_before_cleanup": True,
        "outputs_bytes": 455270005885,
        "outputs_du_h": "425G",
        "renders_bytes": 20070770153,
        "renders_du_h": "19G",
        "filesystem_available_bytes": 19642634240,
    }
    disk_after = {
        "captured_immediately_after_cleanup": True,
        "outputs_bytes": 441196859606,
        "renders_bytes": 20070770153,
        "filesystem_available_bytes": 33715249152,
        "reclaimed_path_size_bytes": sum(
            int(row["size_bytes"]) for row in approved
        ),
        "reclaimed_allocated_bytes": sum(
            int(row["size_du_kib"]) * 1024 for row in approved
        ),
        "deleted_output_paths": [row["path"] for row in approved],
        "deleted_render_paths": [],
    }
    support_definition = {
        "recovered_from": (
            "run_ocmc_candidate_c_resplit_replication.py::"
            "_historical_visible_mask/_render_final"
        ),
        "visible_gaussian": "model.radii > 0",
        "visibility_alias": "outputs['gaussian_visible_mask']",
        "alias_equality_asserted_every_render": True,
        "s_i": (
            "number of distinct preregistered training cameras in which "
            "Gaussian i has radii > 0 at the frozen checkpoint"
        ),
        "at_most_one_increment_per_training_camera": True,
        "heldout_cameras_excluded": True,
        "duplicate_pixels_within_camera_excluded": True,
        "future_views_excluded": True,
        "fraction_visible_unseen_train": (
            "heldout-visible fraction with s_i == 0"
        ),
        "fraction_visible_low_support": (
            "heldout-visible fraction with s_i <= 1"
        ),
        "mean_training_support": (
            "mean s_i over heldout-visible Gaussians"
        ),
        "median_training_support": (
            "median s_i over heldout-visible Gaussians"
        ),
        "thresholds": {
            f"T{threshold}": f"s_i <= {threshold}"
            for threshold in THRESHOLDS
        },
        "groups": {
            group: {"lower_inclusive": lo, "upper_inclusive": hi}
            for group, (lo, hi) in GROUP_BOUNDS.items()
        },
        "contribution_weight": (
            "exact transmittance-weighted alpha contribution from the "
            "standard 3-channel compositor with indicator RGB, zero medium, "
            "and black background"
        ),
        "contribution_validation": (
            "four mutually exclusive group maps must be finite and sum to "
            "formal accumulation at atol/rtol 2e-6; T3 uses its own s<=3 map"
        ),
        "high_residual_region": (
            "within-camera top 20% pixels by mean squared RGB residual; "
            "descriptive GT-only localization"
        ),
    }
    _write_json(OUTPUT_ROOT / "repo_state.json", repo_state)
    _write_json(OUTPUT_ROOT / "environment.json", environment)
    _write_json(OUTPUT_ROOT / "disk_cleanup_before.json", disk_before)
    _write_json(OUTPUT_ROOT / "disk_cleanup_manifest.json", cleanup_manifest)
    _write_json(OUTPUT_ROOT / "disk_cleanup_after.json", disk_after)
    _write_json(
        OUTPUT_ROOT / "checkpoint_manifest.json", {"rows": checkpoint_rows}
    )
    _write_json(OUTPUT_ROOT / "support_definition.json", support_definition)
    result = {
        "repo_state": repo_state,
        "environment": environment,
        "checkpoint_count": len(checkpoint_rows),
        "formal_checkpoint_count": sum(
            row["formal_checkpoint"] for row in checkpoint_rows
        ),
        "all_passed": True,
        "no_training": True,
    }
    _write_json(OUTPUT_ROOT / "preflight.json", result)
    return result


def _effect_rows(
    camera_rows: Sequence[Mapping[str, Any]], weighted: bool
) -> List[Dict[str, Any]]:
    rows = []
    for scene in SCENES:
        for step in ALL_STEPS:
            selected = [
                row
                for row in camera_rows
                if row["scene"] == scene
                and int(row["absolute_step"]) == step
            ]
            errors = [float(row["E_cam"]) for row in selected]
            for threshold in THRESHOLDS:
                key = (
                    "cw_fraction_support_le_"
                    if weighted
                    else "fraction_visible_support_le_"
                ) + str(threshold)
                values = [float(row[key]) for row in selected]
                rho = _rho(values, errors)
                rows.append(
                    {
                        "scene": scene,
                        "absolute_step": step,
                        "formal_checkpoint": step in FORMAL_STEPS,
                        "threshold": f"T{threshold}",
                        "support_condition": f"s_i <= {threshold}",
                        "weighting": (
                            "transmittance_weighted_alpha_contribution"
                            if weighted
                            else "visible_gaussian_count"
                        ),
                        "heldout_camera_count": len(selected),
                        "spearman_rho_E_cam": rho,
                        "positive_direction": bool(
                            math.isfinite(rho) and rho > 0
                        ),
                        "replication_threshold_met": bool(
                            math.isfinite(rho) and rho >= 0.4
                        ),
                        "camera_rank_order_by_proxy": [
                            row["camera_id"]
                            for row in sorted(
                                selected, key=lambda item: float(item[key])
                            )
                        ],
                        "camera_rank_order_by_E_cam": [
                            row["camera_id"]
                            for row in sorted(
                                selected,
                                key=lambda item: float(item["E_cam"]),
                            )
                        ],
                    }
                )
    return rows


def _threshold_decision(
    unweighted: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    final = [
        row for row in unweighted if int(row["absolute_step"]) == 14999
    ]
    summary = []
    for threshold in THRESHOLDS:
        selected = [
            row for row in final if row["threshold"] == f"T{threshold}"
        ]
        replication = sum(
            bool(row["replication_threshold_met"]) for row in selected
        )
        opposite = sum(
            float(row["spearman_rho_E_cam"]) <= -0.4
            for row in selected
            if math.isfinite(float(row["spearman_rho_E_cam"]))
        )
        summary.append(
            {
                "threshold": f"T{threshold}",
                "scene_count_positive": sum(
                    float(row["spearman_rho_E_cam"]) > 0
                    for row in selected
                    if math.isfinite(float(row["spearman_rho_E_cam"]))
                ),
                "scene_count_rho_at_least_0p4": replication,
                "scene_count_rho_at_most_minus_0p4": opposite,
                "median_rho": _quantile(
                    [
                        float(row["spearman_rho_E_cam"])
                        for row in selected
                    ],
                    0.5,
                ),
                "replicates_3_of_4": replication >= 3,
            }
        )
    passing = {
        int(row["threshold"][1:])
        for row in summary
        if row["replicates_3_of_4"]
        and row["scene_count_rho_at_most_minus_0p4"] <= 1
    }
    adjacent = [
        (left, left + 1)
        for left in range(3)
        if left in passing and left + 1 in passing
    ]
    if adjacent:
        classification = "LOW_SUPPORT_THRESHOLD_ROBUST"
    elif len(passing) == 1:
        classification = "LOW_SUPPORT_THRESHOLD_SENSITIVE"
    else:
        classification = "LOW_SUPPORT_THRESHOLD_NOT_SUPPORTED"
    return {
        "classification": classification,
        "threshold_rows": summary,
        "adjacent_replicating_thresholds": [
            f"T{left}/T{right}" for left, right in adjacent
        ],
    }


def _temporal_decision(
    unweighted: Sequence[Mapping[str, Any]],
    threshold_decision: Mapping[str, Any],
) -> Dict[str, Any]:
    candidates = [
        int(row["threshold"][1:])
        for row in threshold_decision["threshold_rows"]
        if row["replicates_3_of_4"]
    ]
    rows = []
    for scene in SCENES:
        for threshold in THRESHOLDS:
            selected = [
                row
                for row in unweighted
                if row["scene"] == scene
                and row["threshold"] == f"T{threshold}"
                and int(row["absolute_step"]) in FORMAL_STEPS
            ]
            by_step = {
                int(row["absolute_step"]): float(row["spearman_rho_E_cam"])
                for row in selected
            }
            intermediate_hits = [
                step
                for step in FORMAL_STEPS[:-1]
                if math.isfinite(by_step.get(step, float("nan")))
                and by_step[step] >= 0.4
            ]
            intermediate_positive = [
                step
                for step in FORMAL_STEPS[:-1]
                if math.isfinite(by_step.get(step, float("nan")))
                and by_step[step] > 0
            ]
            final_hit = (
                math.isfinite(by_step.get(14999, float("nan")))
                and by_step[14999] >= 0.4
            )
            first_positive = next(
                (
                    step
                    for step in FORMAL_STEPS
                    if math.isfinite(by_step.get(step, float("nan")))
                    and by_step[step] > 0
                ),
                None,
            )
            persistent = bool(
                first_positive is not None
                and all(
                    by_step[step] > 0
                    for step in FORMAL_STEPS
                    if step >= first_positive
                    and math.isfinite(by_step.get(step, float("nan")))
                )
            )
            rows.append(
                {
                    "scene": scene,
                    "threshold": f"T{threshold}",
                    "first_positive_checkpoint": first_positive,
                    "intermediate_checkpoints_rho_at_least_0p4": (
                        intermediate_hits
                    ),
                    "intermediate_positive_checkpoints": (
                        intermediate_positive
                    ),
                    "intermediate_hit_count": len(intermediate_hits),
                    "intermediate_positive_count": len(
                        intermediate_positive
                    ),
                    "final_rho_at_least_0p4": final_hit,
                    "final_positive": bool(
                        math.isfinite(by_step.get(14999, float("nan")))
                        and by_step[14999] > 0
                    ),
                    "positive_relation_persists_after_first": persistent,
                    "temporally_stable_by_protocol": (
                        len(intermediate_positive) >= 2
                        and math.isfinite(
                            by_step.get(14999, float("nan"))
                        )
                        and by_step[14999] > 0
                    ),
                    **{
                        f"rho_{step}": by_step.get(step, float("nan"))
                        for step in FORMAL_STEPS
                    },
                }
            )
    stable_counts = {
        threshold: sum(
            row["temporally_stable_by_protocol"]
            for row in rows
            if row["threshold"] == f"T{threshold}"
        )
        for threshold in candidates
    }
    if any(count >= 3 for count in stable_counts.values()):
        classification = "LOW_SUPPORT_TEMPORALLY_STABLE"
    elif candidates:
        late = any(
            sum(
                row[f"rho_{step}"] >= 0.4
                for step in (13000, 14999)
                if math.isfinite(row[f"rho_{step}"])
            )
            == 2
            and sum(
                row[f"rho_{step}"] >= 0.4
                for step in (5000, 8000, 10000)
                if math.isfinite(row[f"rho_{step}"])
            )
            <= 1
            for row in rows
            if int(row["threshold"][1:]) in candidates
        )
        classification = (
            "LOW_SUPPORT_LATE_STAGE_ONLY"
            if late
            else "LOW_SUPPORT_TEMPORALLY_MIXED"
        )
    elif any(row["final_rho_at_least_0p4"] for row in rows):
        classification = "LOW_SUPPORT_TEMPORALLY_MIXED"
    else:
        classification = "LOW_SUPPORT_TEMPORALLY_NOT_SUPPORTED"
    return {
        "classification": classification,
        "rows": rows,
        "stable_scene_counts_for_final_supported_thresholds": stable_counts,
    }


def _aggregate_enrichment(
    rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    result = []
    for scene in SCENES:
        for group in GROUPS:
            selected = [
                row
                for row in rows
                if row["scene"] == scene
                and row["support_group"] == group
            ]
            high = _finite(
                [float(row["contribution_fraction_high"]) for row in selected]
            )
            normal = _finite(
                [
                    float(row["contribution_fraction_normal"])
                    for row in selected
                ]
            )
            ratios = [
                float(row["enrichment_ratio"]) for row in selected
            ]
            mean_high = float(np.mean(high)) if high.size else float("nan")
            mean_normal = (
                float(np.mean(normal)) if normal.size else float("nan")
            )
            result.append(
                {
                    "scene": scene,
                    "support_group": group,
                    "heldout_camera_count": len(selected),
                    "mean_contribution_fraction_high": mean_high,
                    "mean_contribution_fraction_normal": mean_normal,
                    "ratio_of_mean_fractions": (
                        mean_high / mean_normal
                        if mean_normal > EPS
                        else float("nan")
                    ),
                    "median_camera_enrichment_ratio": _quantile(
                        ratios, 0.5
                    ),
                    "camera_count_enrichment_gt_1": sum(
                        value > 1
                        for value in ratios
                        if math.isfinite(value)
                    ),
                }
            )
    for scene in SCENES:
        by_camera: Dict[str, Dict[str, Mapping[str, Any]]] = {}
        for row in rows:
            if row["scene"] == scene:
                by_camera.setdefault(str(row["camera_id"]), {})[
                    str(row["support_group"])
                ] = row
        high = [
            sum(
                float(groups[group]["contribution_fraction_high"])
                for group in ("G0", "G1")
            )
            for groups in by_camera.values()
        ]
        normal = [
            sum(
                float(groups[group]["contribution_fraction_normal"])
                for group in ("G0", "G1")
            )
            for groups in by_camera.values()
        ]
        mean_high = float(np.mean(high))
        mean_normal = float(np.mean(normal))
        result.append(
            {
                "scene": scene,
                "support_group": "LOW(G0+G1)",
                "heldout_camera_count": len(by_camera),
                "mean_contribution_fraction_high": mean_high,
                "mean_contribution_fraction_normal": mean_normal,
                "ratio_of_mean_fractions": (
                    mean_high / mean_normal
                    if mean_normal > EPS
                    else float("nan")
                ),
                "median_camera_enrichment_ratio": _quantile(
                    [
                        left / right
                        for left, right in zip(high, normal)
                        if right > EPS
                    ],
                    0.5,
                ),
                "camera_count_enrichment_gt_1": sum(
                    left > right for left, right in zip(high, normal)
                ),
            }
        )
    return result


def _aggregate_stratified(
    rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    result = []
    for scene in SCENES:
        for factor in ("depth", "scale", "opacity", "footprint"):
            for bin_label in ("low", "middle", "high"):
                for population in (
                    "low_support_s_le_1",
                    "high_support_s_ge_3",
                ):
                    selected = [
                        row
                        for row in rows
                        if row["scene"] == scene
                        and row["stratification_factor"] == factor
                        and row["bin"] == bin_label
                        and row["support_population"] == population
                    ]
                    result.append(
                        {
                            "scene": scene,
                            "absolute_step": 14999,
                            "stratification_factor": factor,
                            "fixed_quantile_bins": (
                                "scene heldout-visible tertiles"
                            ),
                            "bin": bin_label,
                            "lower_bound": (
                                selected[0]["lower_bound"]
                                if selected
                                else float("nan")
                            ),
                            "upper_bound": (
                                selected[0]["upper_bound"]
                                if selected
                                else float("nan")
                            ),
                            "support_population": population,
                            "gaussian_count": (
                                selected[0]["gaussian_count"]
                                if selected
                                else 0
                            ),
                            "heldout_camera_count": len(selected),
                            "mean_enrichment_ratio": (
                                float(
                                    np.mean(
                                        _finite(
                                            [
                                                row["enrichment_ratio"]
                                                for row in selected
                                            ]
                                        )
                                    )
                                )
                                if _finite(
                                    [
                                        row["enrichment_ratio"]
                                        for row in selected
                                    ]
                                ).size
                                else float("nan")
                            ),
                            "median_enrichment_ratio": _quantile(
                                [
                                    row["enrichment_ratio"]
                                    for row in selected
                                ],
                                0.5,
                            ),
                            "camera_count_enrichment_gt_1": sum(
                                float(row["enrichment_ratio"]) > 1
                                for row in selected
                                if math.isfinite(
                                    float(row["enrichment_ratio"])
                                )
                            ),
                            "diagnostic_uses_gt": True,
                        }
                    )
    return result


def _control_rows(
    camera_rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    rows = []
    final = [
        row for row in camera_rows if int(row["absolute_step"]) == 14999
    ]
    for scene in SCENES:
        selected = [row for row in final if row["scene"] == scene]
        errors = [float(row["E_cam"]) for row in selected]
        for threshold in THRESHOLDS:
            key = f"fraction_visible_support_le_{threshold}"
            predictor = [float(row[key]) for row in selected]
            raw = _rho(predictor, errors)
            for control in CONTROLS:
                values = [float(row[control]) for row in selected]
                controlled = _rank_residualized_rho(
                    predictor, errors, values
                )
                rows.append(
                    {
                        "scene": scene,
                        "threshold": f"T{threshold}",
                        "control": control,
                        "heldout_camera_count": len(selected),
                        "raw_spearman_rho": raw,
                        "predictor_vs_control_spearman_rho": _rho(
                            predictor, values
                        ),
                        "control_vs_E_cam_spearman_rho": _rho(
                            values, errors
                        ),
                        "residualized_rank_spearman_rho": controlled,
                        "positive_after_control": bool(
                            math.isfinite(controlled) and controlled > 0
                        ),
                        "rho_at_least_0p4_after_control": bool(
                            math.isfinite(controlled)
                            and controlled >= 0.4
                        ),
                        "single_factor_only": True,
                    }
                )
    return rows


def _ocmc_independence(
    camera_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    rows = []
    positive = 0
    for scene in SCENES:
        selected = [
            row
            for row in camera_rows
            if row["scene"] == scene
            and int(row["absolute_step"]) == 14999
        ]
        predictor = [
            float(row["fraction_visible_support_le_1"])
            for row in selected
        ]
        errors = [float(row["E_cam"]) for row in selected]
        ocmc = [
            float(row["mean_ocmc_projected_camera_residual"])
            for row in selected
        ]
        controlled = _rank_residualized_rho(
            predictor, errors, ocmc
        )
        positive += int(math.isfinite(controlled) and controlled > 0)
        source_manifest = _read_json(
            SOURCE_ROOT / scene / "checkpoint_manifest.json"
        )
        rows.append(
            {
                "scene": scene,
                "selected_preregistered_threshold": "T1",
                "support_vs_ocmc_camera_residual_spearman": _rho(
                    predictor, ocmc
                ),
                "ocmc_camera_residual_vs_E_cam_spearman": _rho(
                    ocmc, errors
                ),
                "support_vs_E_cam_after_ocmc_camera_residual_control": (
                    controlled
                ),
                "positive_after_ocmc_control": bool(
                    math.isfinite(controlled) and controlled > 0
                ),
                "ocmc_global_gate": source_manifest["ocmc_global_gate"],
                "weak_ocmc_mode_count_gate_lt_0p5": source_manifest[
                    "ocmc_modes_below_half"
                ],
                "global_gate_is_scene_level_not_camera_level": True,
            }
        )
    if positive >= 3:
        classification = "LOW_SUPPORT_DISTINCT_FROM_OCMC"
    elif positive <= 1:
        classification = "LOW_SUPPORT_OCMC_COUPLED"
    else:
        classification = "LOW_SUPPORT_OCMC_INDEPENDENCE_INCONCLUSIVE"
    return {
        "classification": classification,
        "selected_preregistered_threshold": "T1",
        "positive_after_ocmc_control_scene_count": positive,
        "conceptual_independence": (
            "OCMC gates camera-conditioned medium mode capacity; support "
            "counts distinct training-view evidence for scene Gaussians."
        ),
        "g_obs_equivalence_rejected": True,
        "rows": rows,
    }


def _online_artifacts(
    actual_counts: Mapping[str, int],
    train_camera_counts: Mapping[str, int],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    memory_rows = []
    counts = {
        **actual_counts,
        "lower_bound_0p75M": 750000,
        "upper_bound_1p2M": 1200000,
    }
    for label, count in counts.items():
        for dtype, bytes_each in (
            ("uint8", 1),
            ("uint16", 2),
            ("int32", 4),
        ):
            memory_rows.append(
                {
                    "population": label,
                    "gaussian_count": count,
                    "dtype": dtype,
                    "bytes": count * bytes_each,
                    "decimal_MB": count * bytes_each / 1e6,
                    "MiB": count * bytes_each / (1024**2),
                }
            )
    for scene, count in actual_counts.items():
        bytes_each = math.ceil(train_camera_counts[scene] / 8)
        memory_rows.append(
            {
                "population": scene,
                "gaussian_count": count,
                "dtype": (
                    f"exact_camera_bitset_{train_camera_counts[scene]}_bits"
                ),
                "bytes": count * bytes_each,
                "decimal_MB": count * bytes_each / 1e6,
                "MiB": count * bytes_each / (1024**2),
            }
        )
    online = {
        "heldout_gt_required": False,
        "future_views_required": False,
        "per_ray_jacobian_required": False,
        "offline_full_scene_rerender_each_iteration_required": False,
        "reusable_visibility_signal": (
            "model.radii > 0 / gaussian_visible_mask from the current "
            "training forward"
        ),
        "existing_vis_counts_semantics": (
            "optimization observation count since the last refinement "
            "interval; not distinct-camera support"
        ),
        "existing_max_2Dsize_semantics": (
            "maximum projected radius since the last refinement interval; "
            "not distinct-camera support"
        ),
        "plain_scalar_counter_exact": False,
        "reason_plain_counter_not_exact": (
            "random sampling can revisit a camera non-consecutively, so "
            "increments require camera-identity deduplication"
        ),
        "exact_distinct_camera_online_feasible": True,
        "exact_state_candidate": (
            "per-Gaussian training-camera bitset plus topology-aware "
            "split/duplicate/prune operations"
        ),
        "approximate_state_candidate": (
            "documented sketch or bounded recent-camera structure; not "
            "equivalent to exact s_i"
        ),
        "estimated_update_cost": "LOW",
        "cost_basis": (
            "one visible-mask bit update for currently visible Gaussians; "
            "no extra render or Jacobian"
        ),
        "implemented_in_training": False,
    }
    lifecycle = {
        "explicit_age_in_locked_checkpoints": False,
        "explicit_lineage_in_locked_checkpoints": False,
        "load_state_dict_discards_legacy_gaussian_lineage_ids": True,
        "split_behavior": (
            "children append copied appearance/opacity and reduced scale; "
            "original split parents are then culled"
        ),
        "duplicate_behavior": "copies append and originals remain",
        "prune_behavior": "all Gaussian parameter arrays are masked",
        "future_state_requirement": (
            "support state must undergo the same append/mask topology "
            "operations as Gaussian parameters"
        ),
        "child_inherit_option": (
            "inherit parent camera bitset, preserving inherited geometric "
            "evidence but overstating independently observed child evidence"
        ),
        "child_reset_option": (
            "reset child state, preserving post-birth evidence but "
            "understating inherited constraints"
        ),
        "prune_option": (
            "mask support state exactly with pruned Gaussian indices"
        ),
        "inheritance_potentially_biased": True,
        "newborn_reset_potentially_biased": True,
        "final_design_selected": False,
        "age_confound_testable_from_checkpoint": False,
        "temporal_identity_tracking_valid": False,
        "temporal_analysis_scope": (
            "distributional camera/group statistics only"
        ),
    }
    return {"rows": memory_rows}, online, lifecycle


def _make_figures(
    unweighted: Sequence[Mapping[str, Any]],
    weighted: Sequence[Mapping[str, Any]],
    camera_rows: Sequence[Mapping[str, Any]],
    enrichment: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
) -> None:
    del weighted
    root = OUTPUT_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    colors = {
        "Curasao": "#0072B2",
        "IUI3-RedSea": "#D55E00",
        "JapaneseGradens-RedSea": "#009E73",
        "Panama": "#CC79A7",
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for scene in SCENES:
        rows = [
            row
            for row in unweighted
            if row["scene"] == scene
            and int(row["absolute_step"]) == 14999
        ]
        ax.plot(
            THRESHOLDS,
            [row["spearman_rho_E_cam"] for row in rows],
            marker="o",
            label=scene,
            color=colors[scene],
        )
    ax.axhline(0.4, color="#555555", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Support threshold T",
        ylabel="Spearman rho with E_cam",
        xticks=THRESHOLDS,
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "threshold_vs_rho.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    for ax, scene in zip(axes.flat, SCENES):
        for threshold in THRESHOLDS:
            rows = [
                row
                for row in unweighted
                if row["scene"] == scene
                and row["threshold"] == f"T{threshold}"
                and int(row["absolute_step"]) in FORMAL_STEPS
            ]
            ax.plot(
                FORMAL_STEPS,
                [row["spearman_rho_E_cam"] for row in rows],
                marker="o",
                label=f"T{threshold}",
            )
        ax.axhline(0.4, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_title(scene, fontsize=9)
    axes[1, 0].set_xlabel("Checkpoint")
    axes[1, 1].set_xlabel("Checkpoint")
    axes[0, 0].set_ylabel("rho")
    axes[1, 0].set_ylabel("rho")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "checkpoint_rho_trajectory.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    low = [
        row for row in enrichment if row["support_group"] == "LOW(G0+G1)"
    ]
    x = np.arange(len(SCENES))
    width = 0.36
    high_values = [
        next(row for row in low if row["scene"] == scene)[
            "mean_contribution_fraction_high"
        ]
        for scene in SCENES
    ]
    normal_values = [
        next(row for row in low if row["scene"] == scene)[
            "mean_contribution_fraction_normal"
        ]
        for scene in SCENES
    ]
    ax.bar(
        x - width / 2,
        high_values,
        width,
        label="top 20% residual",
        color="#D55E00",
    )
    ax.bar(
        x + width / 2,
        normal_values,
        width,
        label="remaining pixels",
        color="#56B4E9",
    )
    ax.set_xticks(x, ["Curasao", "IUI3", "Japanese", "Panama"])
    ax.set_ylabel("Low-support contribution fraction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "high_error_enrichment.png", dpi=160)
    plt.close(fig)

    for weighted_flag, filename, key in (
        (
            False,
            "low_support_fraction_vs_E_cam.png",
            "fraction_visible_support_le_1",
        ),
        (
            True,
            "contribution_weighted_vs_E_cam.png",
            "cw_fraction_support_le_1",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for scene in SCENES:
            selected = [
                row
                for row in camera_rows
                if row["scene"] == scene
                and int(row["absolute_step"]) == 14999
            ]
            ax.scatter(
                [row[key] for row in selected],
                [row["E_cam"] for row in selected],
                label=scene,
                color=colors[scene],
                s=28,
            )
        ax.set_xlabel(
            "Contribution-weighted support <=1"
            if weighted_flag
            else "Visible fraction support <=1"
        )
        ax.set_ylabel("E_cam (MSE)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    selected = [row for row in controls if row["threshold"] == "T1"]
    for scene in SCENES:
        rows = [row for row in selected if row["scene"] == scene]
        ax.plot(
            np.arange(len(CONTROLS)),
            [row["residualized_rank_spearman_rho"] for row in rows],
            marker="o",
            label=scene,
            color=colors[scene],
        )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(
        np.arange(len(CONTROLS)),
        [item.replace("mean_", "") for item in CONTROLS],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("T1 controlled rho")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "major_single_factor_controls.png", dpi=160)
    plt.close(fig)


def _research_note(summary: Mapping[str, Any]) -> None:
    threshold = summary["threshold_decision"]
    temporal = summary["temporal_decision"]
    final = summary["final_proxy_classification"]
    contribution = summary["contribution_comparison"]
    confounders = summary["confounder_summary"]
    threshold_lines = [
        (
            f"| {row['threshold']} | "
            f"{row['scene_count_rho_at_least_0p4']}/4 | "
            f"{row['median_rho']:.3f} |"
        )
        for row in threshold["threshold_rows"]
    ]
    scene_lines = []
    for scene in SCENES:
        effects = [
            row
            for row in summary["final_unweighted_effects"]
            if row["scene"] == scene
        ]
        scene_lines.append(
            "| "
            + scene
            + " | "
            + " | ".join(
                f"{row['spearman_rho_E_cam']:.3f}" for row in effects
            )
            + " |"
        )
    enrichment_lines = []
    for scene in SCENES:
        row = next(
            item
            for item in summary["high_error_enrichment"]
            if item["scene"] == scene
            and item["support_group"] == "LOW(G0+G1)"
        )
        localization = summary["gaussian_localization"]["scenes"][scene]
        enrichment_lines.append(
            f"| {scene} | "
            f"{row['mean_contribution_fraction_high']:.6f} | "
            f"{row['mean_contribution_fraction_normal']:.6f} | "
            f"{row['ratio_of_mean_fractions']:.3f} | "
            f"{row['camera_count_enrichment_gt_1']}/"
            f"{row['heldout_camera_count']} | "
            f"{'yes' if localization['low_support_enrichment_supported'] else 'no'} |"
        )
    temporal_lines = []
    for scene in SCENES:
        row = next(
            item
            for item in temporal["rows"]
            if item["scene"] == scene and item["threshold"] == "T1"
        )
        temporal_lines.append(
            f"| {scene} | {row['first_positive_checkpoint']} | "
            f"{row['rho_5000']:.3f} | {row['rho_14999']:.3f} | "
            f"{'yes' if row['positive_relation_persists_after_first'] else 'no'} |"
        )
    group_lines = []
    for scene in SCENES:
        rows = [
            row
            for row in summary["gaussian_support_group_stats"]
            if row["scene"] == scene
        ]
        by_group = {row["support_group"]: row for row in rows}
        group_lines.append(
            f"| {scene} | "
            + " | ".join(
                f"{100 * by_group[group]['gaussian_fraction']:.3f}%"
                for group in GROUPS
            )
            + " |"
        )
    camera_control_scenes = [
        scene
        for scene in SCENES
        if confounders["scene_survives_all_camera_single_factor_controls"][scene]
    ]
    stratified_control_scenes = [
        scene
        for scene in SCENES
        if confounders["scene_survives_all_stratified_controls"][scene]
    ]
    lines = [
        "# Isolation Audit of Low-Support Gaussian Failure Proxy (2026-08-31)",
        "",
        "## 1. Motivation",
        "",
        "HYPOTHESIS: Gaussians constrained by few distinct training views may be "
        "associated with heldout reconstruction error after locked OCMC. This "
        "audit is diagnostic and does not establish causality.",
        "",
        "## 2. Why Camera-Neighborhood Hypothesis Was Rejected",
        "",
        "LOCKED RESULT: center-space leave-one-out correlations were negative in "
        "all four prior scenes. Camera-neighborhood structure is not reused as "
        "the mechanism here.",
        "",
        "## 3. Current Low-Support Hypothesis",
        "",
        "The candidate signal is representation support across distinct "
        "preregistered training cameras, not camera-space proximity.",
        "",
        "## 4. Frozen OCMC States",
        "",
        "CONFIG FACT: all 5K/8K/10K/13K/14999 C0 checkpoints use bounded_sh3, "
        "SH degree 3, classic rasterization, dir_xy_camera, OCMC on, RAOC off, "
        "and seed 42. The 3K state is descriptive only. No optimization or "
        "backward pass was run.",
        "",
        "EXPERIMENTAL FACT: four frozen workers used physical GPUs 6/7/8/9 "
        "as logical cuda:0 for Curasao/IUI3-RedSea/JapaneseGradens-RedSea/"
        "Panama. All 20 formal checkpoints and four descriptive 3K checkpoints "
        "were present and hash-verified.",
        "",
        "## 5. Support Definition",
        "",
        "CODE FACT: visibility is model.radii > 0; exact equality with "
        "gaussian_visible_mask was asserted. s_i is the number of distinct "
        "preregistered training cameras in which Gaussian i is visible at that "
        "frozen checkpoint. Heldout cameras, duplicate pixels, and future views "
        "are excluded.",
        "",
        "Fixed thresholds are T0-T3 (s_i <= 0,1,2,3); groups are G0 (s=0), "
        "G1 (s=1), G2 (s=2), and G3+ (s>=3).",
        "",
        "## 6. Threshold Stability",
        "",
        "| Threshold | scenes rho >= 0.4 | median rho |",
        "| --- | ---: | ---: |",
        *threshold_lines,
        "",
        f"QUANTITATIVE RESULT: {threshold['classification']}. Adjacent "
        f"replicating pairs: "
        f"{threshold['adjacent_replicating_thresholds'] or 'none'}.",
        "",
        "| Scene | T0 | T1 | T2 | T3 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *scene_lines,
        "",
        "## 7. Temporal Stability",
        "",
        f"QUANTITATIVE RESULT: {temporal['classification']}. Comparisons are "
        "distributional because split/prune operations invalidate identity "
        "continuity.",
        "",
        "| Scene | first positive T1 | rho at 5K | rho at 14999 | positive persists |",
        "| --- | ---: | ---: | ---: | --- |",
        *temporal_lines,
        "",
        "T1 is positive by 5K in every scene and remains positive through "
        "14999. JapaneseGradens-RedSea first exceeds rho >= 0.4 at 8K; its "
        "5K rho is 0.395. Panama T0 reverses after 5K, but T1-T3 persist.",
        "",
        "## 8. Camera-Level Replication",
        "",
        f"QUANTITATIVE RESULT: final replication covers "
        f"{summary['heldout_camera_count']} preregistered heldout cameras. "
        "E_cam is heldout MSE; PSNR, SSIM, LPIPS, and MAE are descriptive.",
        "",
        "## 9. Gaussian-Level Localization",
        "",
        "| Scene | s=0 | s=1 | s=2 | s>=3 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *group_lines,
        "",
        f"QUANTITATIVE RESULT: low-support high-error enrichment is supported in "
        f"{summary['gaussian_localization']['supported_scene_count']}/4 scenes. "
        f"Approximate support-order monotonicity appears in "
        f"{summary['gaussian_localization']['approximately_monotonic_scene_count']}/4.",
        "",
        "## 10. Contribution Weighting",
        "",
        "CODE FACT: each group is rendered as a standard 3-channel indicator "
        "under the original projected geometry, depth order, opacity, and "
        "alpha/transmittance compositor. Four group maps sum to formal "
        "accumulation under 2e-6 tolerance. The unstable ND path was not used.",
        "",
        f"QUANTITATIVE RESULT: contribution weighting "
        f"{contribution['interpretation']} the cross-scene association: at "
        f"preregistered T1 the median rho changes from "
        f"{contribution['median_rho_unweighted']:.3f} to "
        f"{contribution['median_rho_weighted']:.3f} "
        f"(delta {contribution['median_rho_delta_weighted_minus_unweighted']:.3f}). "
        f"The strongest descriptive proxies are "
        f"{contribution['strongest_unweighted_proxy_descriptive']} and "
        f"{contribution['strongest_contribution_weighted_proxy_descriptive']}.",
        "",
        "## 11. High-Residual Enrichment",
        "",
        "| Scene | top-20% low-support fraction | remaining fraction | enrichment | cameras enriched | scene criterion |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        *enrichment_lines,
        "",
        "GT is used only to define diagnostic pixel regions; enrichment is not "
        "an online training variable. Curasao has aggregate enrichment but only "
        "1/6 cameras are enriched, so it does not pass the registered scene "
        "criterion; the other three scenes do.",
        "",
        "## 12. Confounders",
        "",
        f"QUANTITATIVE RESULT: preregistered T1 remains positive after every "
        f"single-factor camera control in "
        f"{confounders['scene_count_surviving_all_camera_controls']}/4 "
        "scenes. Controls are depth, tau, transmission, accumulation, "
        "footprint, opacity, scale, and visible count; no multivariable "
        f"regression was fit. Passing scenes: {', '.join(camera_control_scenes)}. "
        "Panama fails only because its scale control is constant, making the "
        "rank residual undefined.",
        "",
        f"QUANTITATIVE RESULT: fixed-tertile stratified localization survives "
        f"all four Gaussian factors in "
        f"{confounders['scene_count_surviving_stratified_controls']}/4 scenes: "
        f"{', '.join(stratified_control_scenes)}. The registered confounder "
        "criterion requires each control family to replicate in >=3 scenes; "
        "it does not require the same three scenes. A unique physical view "
        "angle is unavailable because no "
        "registered surface normal exists for an anisotropic Gaussian.",
        "",
        "## 13. OCMC Independence",
        "",
        f"INFERENCE: {summary['ocmc_independence']['classification']}. OCMC "
        "controls medium mode capacity while support records scene-Gaussian "
        "distinct-view evidence. T1 remains positive after OCMC-residual "
        f"control in "
        f"{summary['ocmc_independence']['positive_after_ocmc_control_scene_count']}/4.",
        "",
        "## 14. Online Computability",
        "",
        "CODE FACT: the current forward exposes radii > 0. Heldout GT, future "
        "views, Jacobians, and an extra render are unnecessary. Existing "
        "vis_counts counts optimization observations rather than distinct "
        "cameras. Exact deduplication needs camera-identity state such as a "
        "per-Gaussian bitset. Such a bitset is GT-free and feasible at LOW "
        "update cost, but it is not yet a reliable production statistic because "
        "its topology lifecycle is unresolved.",
        "",
        "## 15. Topology Lifecycle Considerations",
        "",
        "CODE FACT: split/duplicate append state, split parents are culled, and "
        "pruning masks state. Locked checkpoints contain no reliable age or "
        "lineage. Child inheritance can overstate independent evidence while "
        "reset can understate inherited evidence; no production policy is selected.",
        "",
        "LIMITATION: Gaussian age/newborn status cannot be controlled reliably "
        "from these checkpoints.",
        "",
        "## 16. Memory / Runtime Cost",
        "",
        "A scalar uint8/uint16/int32 counter costs 0.75-1.20 / 1.50-2.40 / "
        "3.00-4.80 decimal MB for 0.75M-1.2M Gaussians. Exact distinct-camera "
        "bitsets are reported separately at actual camera counts. Estimated "
        "per-iteration cost is LOW: one visible-mask bit update, no rerender.",
        "",
        "## 17. Final Proxy Classification",
        "",
        f"FINAL DECISION: {final['classification']}.",
        "",
        f"RESEARCH-LINE DECISION: {final['research_line_decision']}.",
        "",
        f"INFERENCE: {final['rationale']}",
        "",
        "## 18. ONE Next Task",
        "",
        final["one_next_task"],
        "",
        "No support-aware loss, pruning, refinement, counter, or other module "
        "is implemented in this task.",
        "",
        "## 19. Disk Cleanup Summary",
        "",
        "One reviewed excluded OOM attempt was deleted: "
        "outputs/m1_raoc_causal_four_scene_20260827_attempt1_oom "
        "(14,073,146,279 bytes; 13,743,756 KiB allocated). No render path was "
        "deleted. Every current resplit checkpoint was preserved.",
        "",
    ]
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def aggregate() -> Dict[str, Any]:
    results = [
        _read_json(
            OUTPUT_ROOT / "workers" / scene / "scene_result.json"
        )
        for scene in SCENES
    ]
    if not all(
        not result["optimizer_step_called"]
        and not result["backward_called"]
        and not result["training_performed"]
        and not result["heldout_leakage"]
        for result in results
    ):
        raise RuntimeError("worker frozen-state contract failed")
    camera_rows = [
        row for result in results for row in result["camera_rows"]
    ]
    checkpoint_rows = [
        row for result in results for row in result["checkpoint_rows"]
    ]
    group_rows = [
        row for result in results for row in result["group_rows"]
    ]
    raw_enrichment = [
        row for result in results for row in result["enrichment_rows"]
    ]
    raw_stratified = [
        row for result in results for row in result["stratified_rows"]
    ]
    validations = [
        row
        for result in results
        for row in result["contribution_validation_rows"]
    ]
    if not all(
        row["finite"]
        and row["allclose"]
        and not row["nd_cuda_path_used"]
        and float(row["max_abs_accumulation_difference"]) <= ACCUM_ATOL
        for row in validations
    ):
        raise RuntimeError("contribution compositor validation failed")
    expected_camera_rows = len(ALL_STEPS) * sum(
        len(result["heldout_ids"]) for result in results
    )
    if len(camera_rows) != expected_camera_rows:
        raise RuntimeError(
            f"camera row count mismatch: {len(camera_rows)} "
            f"!= {expected_camera_rows}"
        )
    if not all(bool(row["all_finite"]) for row in camera_rows):
        raise RuntimeError("non-finite camera result")

    unweighted = _effect_rows(camera_rows, weighted=False)
    weighted = _effect_rows(camera_rows, weighted=True)
    threshold_decision = _threshold_decision(unweighted)
    temporal_decision = _temporal_decision(
        unweighted, threshold_decision
    )
    enrichment = _aggregate_enrichment(raw_enrichment)
    stratified = _aggregate_stratified(raw_stratified)
    controls = _control_rows(camera_rows)
    ocmc = _ocmc_independence(camera_rows)

    final_unweighted = [
        row for row in unweighted if int(row["absolute_step"]) == 14999
    ]
    final_weighted = [
        row for row in weighted if int(row["absolute_step"]) == 14999
    ]
    unweighted_median = {
        threshold: _quantile(
            [
                row["spearman_rho_E_cam"]
                for row in final_unweighted
                if row["threshold"] == f"T{threshold}"
            ],
            0.5,
        )
        for threshold in THRESHOLDS
    }
    weighted_median = {
        threshold: _quantile(
            [
                row["spearman_rho_E_cam"]
                for row in final_weighted
                if row["threshold"] == f"T{threshold}"
            ],
            0.5,
        )
        for threshold in THRESHOLDS
    }
    strongest_unweighted = max(
        THRESHOLDS, key=lambda threshold: unweighted_median[threshold]
    )
    strongest_weighted = max(
        THRESHOLDS, key=lambda threshold: weighted_median[threshold]
    )
    delta = weighted_median[1] - unweighted_median[1]
    contribution_comparison = {
        "preregistered_comparison_threshold": "T1",
        "median_rho_unweighted": unweighted_median[1],
        "median_rho_weighted": weighted_median[1],
        "median_rho_delta_weighted_minus_unweighted": delta,
        "interpretation": (
            "strengthens"
            if delta > 0.05
            else "weakens"
            if delta < -0.05
            else "is similar to"
        ),
        "strongest_unweighted_proxy_descriptive": (
            f"T{strongest_unweighted}"
        ),
        "strongest_contribution_weighted_proxy_descriptive": (
            f"CW_T{strongest_weighted}"
        ),
        "threshold_optimization_performed": False,
    }

    localization_scene = {}
    monotonic_count = 0
    for scene in SCENES:
        low = next(
            row
            for row in enrichment
            if row["scene"] == scene
            and row["support_group"] == "LOW(G0+G1)"
        )
        supported = bool(
            float(low["ratio_of_mean_fractions"]) > 1
            and int(low["camera_count_enrichment_gt_1"])
            >= math.ceil(int(low["heldout_camera_count"]) / 2)
        )
        group_values = [
            next(
                row
                for row in enrichment
                if row["scene"] == scene
                and row["support_group"] == group
            )["ratio_of_mean_fractions"]
            for group in GROUPS
        ]
        finite_pairs = [
            (index, float(value))
            for index, value in enumerate(group_values)
            if math.isfinite(float(value))
        ]
        monotonic_rho = _rho(
            [item[0] for item in finite_pairs],
            [item[1] for item in finite_pairs],
        )
        approximately_monotonic = bool(
            math.isfinite(monotonic_rho) and monotonic_rho <= -0.4
        )
        monotonic_count += int(approximately_monotonic)
        localization_scene[scene] = {
            "low_support_enrichment_supported": supported,
            "low_support_enrichment_ratio": low[
                "ratio_of_mean_fractions"
            ],
            "support_order_vs_enrichment_spearman": monotonic_rho,
            "approximately_monotonic": approximately_monotonic,
        }
    gaussian_localization = {
        "scenes": localization_scene,
        "supported_scene_count": sum(
            row["low_support_enrichment_supported"]
            for row in localization_scene.values()
        ),
        "approximately_monotonic_scene_count": monotonic_count,
    }

    camera_control_survival = {}
    stratified_control_survival: Dict[str, Dict[str, bool]] = {}
    for scene in SCENES:
        selected = [
            row
            for row in controls
            if row["scene"] == scene and row["threshold"] == "T1"
        ]
        camera_control_survival[scene] = bool(
            selected and all(row["positive_after_control"] for row in selected)
        )
        stratified_control_survival[scene] = {}
        for factor in ("depth", "scale", "opacity", "footprint"):
            bins = [
                row
                for row in stratified
                if row["scene"] == scene
                and row["stratification_factor"] == factor
                and row["support_population"] == "low_support_s_le_1"
            ]
            stratified_control_survival[scene][factor] = (
                sum(
                    float(row["median_enrichment_ratio"]) > 1
                    for row in bins
                    if math.isfinite(
                        float(row["median_enrichment_ratio"])
                    )
                )
                >= 2
            )
    scene_survives_stratified = {
        scene: all(stratified_control_survival[scene].values())
        for scene in SCENES
    }
    confounder_summary = {
        "preregistered_threshold": "T1",
        "scene_survives_all_camera_single_factor_controls": (
            camera_control_survival
        ),
        "scene_count_surviving_all_camera_controls": sum(
            camera_control_survival.values()
        ),
        "stratified_factor_survival": stratified_control_survival,
        "scene_survives_all_stratified_controls": (
            scene_survives_stratified
        ),
        "scene_count_surviving_stratified_controls": sum(
            scene_survives_stratified.values()
        ),
        "scene_count_surviving_all_controls": sum(
            camera_control_survival[scene]
            and scene_survives_stratified[scene]
            for scene in SCENES
        ),
        "age_recoverable": False,
        "view_angle_control_available": False,
        "view_angle_reason": (
            "no registered surface normal defines one physical view angle "
            "for an anisotropic Gaussian"
        ),
    }

    actual_counts = {
        row["scene"]: int(row["gaussian_count"])
        for row in checkpoint_rows
        if int(row["absolute_step"]) == 14999
    }
    train_camera_counts = {
        result["scene"]: len(result["train_ids"]) for result in results
    }
    memory, online, lifecycle = _online_artifacts(
        actual_counts, train_camera_counts
    )
    camera_replication = max(
        (
            row["scene_count_rho_at_least_0p4"]
            for row in threshold_decision["threshold_rows"]
        ),
        default=0,
    )
    scientific_criteria = {
        "threshold_robust": (
            threshold_decision["classification"]
            == "LOW_SUPPORT_THRESHOLD_ROBUST"
        ),
        "temporally_stable": (
            temporal_decision["classification"]
            == "LOW_SUPPORT_TEMPORALLY_STABLE"
        ),
        "camera_level_replication_3_of_4": camera_replication >= 3,
        "gaussian_localization_3_of_4": (
            gaussian_localization["supported_scene_count"] >= 3
        ),
        "major_confounders_do_not_fully_explain_3_scenes": (
            confounder_summary[
                "scene_count_surviving_all_camera_controls"
            ]
            >= 3
            and confounder_summary[
                "scene_count_surviving_stratified_controls"
            ]
            >= 3
        ),
        "distinct_from_ocmc": (
            ocmc["classification"] == "LOW_SUPPORT_DISTINCT_FROM_OCMC"
        ),
    }
    online_reliable = bool(
        online["exact_distinct_camera_online_feasible"]
        and online["estimated_update_cost"] in ("NEGLIGIBLE", "LOW")
        and not lifecycle["inheritance_potentially_biased"]
    )
    actionable_criteria = {
        **scientific_criteria,
        "gt_free_online_statistic_low_cost_and_reliable": online_reliable,
        "no_per_ray_jacobian": not online["per_ray_jacobian_required"],
    }
    all_eight = all(actionable_criteria.values())
    scientific_pass_count = sum(scientific_criteria.values())
    camera_persists = bool(
        scientific_criteria["camera_level_replication_3_of_4"]
    )
    if all_eight:
        classification = "LOW_SUPPORT_PROXY_SUPPORTED_AND_ACTIONABLE"
        research_line = "PROCEED_TO_LOW_SUPPORT_MINIMAL_CAUSAL_TEST"
        next_task = "LOW-SUPPORT-GAUSSIAN-MINIMAL-CAUSAL-INTERVENTION"
        rationale = (
            "All eight registered scientific and online-actionability "
            "criteria pass."
        )
    elif all(scientific_criteria.values()):
        classification = "LOW_SUPPORT_PROXY_SUPPORTED_BUT_NOT_ACTIONABLE"
        research_line = "DEFER_LOW_SUPPORT_MODULE"
        next_task = "RESOLVE-LOW-SUPPORT-STATE-LIFECYCLE-PREFLIGHT"
        rationale = (
            "The scientific signal passes, but exact distinct-camera state "
            "is not yet reliable across split/duplicate/prune lifecycle."
        )
    elif camera_persists and scientific_pass_count >= 2:
        classification = "LOW_SUPPORT_PROXY_TENTATIVE"
        research_line = "DEFER_LOW_SUPPORT_MODULE"
        next_task = (
            "NO INTERVENTION; REASSESS ONLY WITH NEW PREREGISTERED "
            "CROSS-SCENE EVIDENCE"
        )
        rationale = (
            "Camera evidence persists, but robustness, temporal, "
            "localization, confounder, or independence criteria are mixed."
        )
    else:
        classification = "LOW_SUPPORT_PROXY_NOT_SUPPORTED"
        research_line = "CLOSE_LOW_SUPPORT_DIRECTION"
        next_task = "CLOSE LOW-SUPPORT DIRECTION"
        rationale = (
            "The registered proxy lacks stable replicated localized and "
            "controlled evidence; thresholds were not tuned to rescue it."
        )
    final_classification = {
        "classification": classification,
        "research_line_decision": research_line,
        "one_next_task": next_task,
        "rationale": rationale,
        "scientific_criteria": scientific_criteria,
        "actionable_criteria": actionable_criteria,
        "all_eight_required_criteria_pass": all_eight,
        "threshold_tuning_performed": False,
    }

    _write_table(
        OUTPUT_ROOT,
        "threshold_stability",
        threshold_decision["threshold_rows"],
        classification=threshold_decision["classification"],
        adjacent_replicating_thresholds=threshold_decision[
            "adjacent_replicating_thresholds"
        ],
    )
    _write_table(
        OUTPUT_ROOT,
        "temporal_stability",
        temporal_decision["rows"],
        classification=temporal_decision["classification"],
    )
    _write_table(
        OUTPUT_ROOT, "camera_level_support_effects", unweighted
    )
    _write_table(
        OUTPUT_ROOT,
        "contribution_weighted_effects",
        weighted,
        comparison=contribution_comparison,
    )
    _write_table(
        OUTPUT_ROOT, "gaussian_support_group_stats", group_rows
    )
    _write_table(
        OUTPUT_ROOT,
        "high_error_enrichment",
        enrichment,
        per_camera_rows_preserved_in="workers/<scene>/scene_result.json",
    )
    _write_table(
        OUTPUT_ROOT,
        "single_factor_controls",
        controls,
        summary=confounder_summary,
    )
    _write_table(
        OUTPUT_ROOT, "gaussian_stratified_controls", stratified
    )
    _write_json(OUTPUT_ROOT / "ocmc_independence.json", ocmc)
    _write_json(OUTPUT_ROOT / "online_computability.json", online)
    _write_json(OUTPUT_ROOT / "support_memory_cost.json", memory)
    _write_json(
        OUTPUT_ROOT / "support_topology_lifecycle.json", lifecycle
    )
    _write_json(
        OUTPUT_ROOT / "contribution_compositor_validation.json",
        {"rows": validations, "all_passed": True},
    )
    _write_json(
        OUTPUT_ROOT / "final_proxy_classification.json",
        final_classification,
    )
    summary = {
        "experiment": EXPERIMENT,
        "protocol_valid": True,
        "ocmc_frozen": True,
        "raoc_closed": True,
        "new_models_trained": False,
        "heldout_camera_count": sum(
            len(result["heldout_ids"]) for result in results
        ),
        "checkpoint_rows": checkpoint_rows,
        "threshold_decision": threshold_decision,
        "temporal_decision": temporal_decision,
        "final_unweighted_effects": final_unweighted,
        "final_weighted_effects": final_weighted,
        "contribution_comparison": contribution_comparison,
        "gaussian_support_group_stats": group_rows,
        "high_error_enrichment": enrichment,
        "gaussian_localization": gaussian_localization,
        "confounder_summary": confounder_summary,
        "ocmc_independence": ocmc,
        "online_computability": online,
        "support_topology_lifecycle": lifecycle,
        "final_proxy_classification": final_classification,
        "worker_wall_seconds": {
            result["scene"]: result["wall_seconds"] for result in results
        },
        "protected_files_untouched": True,
    }
    _make_figures(
        unweighted, weighted, camera_rows, enrichment, controls
    )
    _write_json(OUTPUT_ROOT / "final_summary.json", summary)
    _research_note(summary)
    return summary


def launch() -> Dict[str, Any]:
    preflight_result = preflight()
    processes = []
    logs = OUTPUT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for scene, gpu in SCENE_GPUS.items():
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        command = [
            str(PYTHON),
            str(Path(__file__).resolve()),
            "--worker",
            "--scene",
            scene,
            "--gpu",
            gpu,
        ]
        handle = (logs / f"{scene}.log").open("w", encoding="utf8")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((scene, gpu, process, handle))
    failures = []
    for scene, gpu, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append(
                {
                    "scene": scene,
                    "gpu": gpu,
                    "exit_code": code,
                    "log": str(logs / f"{scene}.log"),
                }
            )
    if failures:
        _write_json(
            OUTPUT_ROOT / "worker_failures.json", {"rows": failures}
        )
        raise RuntimeError(f"frozen workers failed: {failures}")
    return {"preflight": preflight_result, "summary": aggregate()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=tuple(SCENE_GPUS.values()))
    args = parser.parse_args()
    if args.worker:
        if args.scene is None or args.gpu is None:
            parser.error("--worker requires --scene and --gpu")
        result = worker(args.scene, args.gpu)
    elif args.preflight:
        result = preflight()
    elif args.aggregate:
        result = aggregate()
    else:
        result = launch()
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
