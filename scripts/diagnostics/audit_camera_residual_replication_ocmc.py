#!/usr/bin/env python3
"""Focused frozen-state replication audit of camera-level OCMC residuals.

The script inventories every source camera before rendering, uses only formal
eval or genuinely unused calibrated-GT cameras as residual targets, and never
performs optimization or checkpoint mutation. Training cameras contribute only
GT-free camera-manifold and Gaussian-visibility support information.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.stats
import torch
from PIL import Image
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary
from scripts.diagnostics import audit_ocmc_residual_failure_modes as BASE
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC

FORMAL = BASE.FORMAL
MI = BASE.MI

EXPERIMENT = "FOCUSED-C-CAMERA-RESIDUAL-REPLICATION-DIAGNOSTIC"
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
PREVIOUS_ROOT = REPO_ROOT / "outputs" / "ocmc_residual_failure_audit_20260831"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "focused_c_camera_residual_replication_20260831"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "FOCUSED_C_CAMERA_RESIDUAL_REPLICATION_2026-08-31.md"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
FINAL_STEP = 14999
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}
SCENES = tuple(SCENE_GPUS)
PROTECTED = BASE.PROTECTED
MIN_RELIABLE_HELDOUT = 5
PERMUTATIONS = 1000
EPS = 1e-12

PREDICTORS: Dict[str, int] = {
    "center_nearest_train": 1,
    "center_knn3_mean": 1,
    "context_nearest_train": 1,
    "context_knn3_mean": 1,
    "context_standardized_distance": 1,
    "view_direction_nearest_angle_deg": 1,
    "view_direction_knn3_angle_deg": 1,
    "visible_gaussian_count": -1,
    "mean_train_visibility_support": -1,
    "median_train_visibility_support": -1,
    "fraction_visible_unseen_train": 1,
    "fraction_visible_low_support": 1,
    "fraction_low_accumulation": 1,
}
CONTROLS = (
    "mean_depth",
    "mean_tau",
    "mean_transmission",
    "mean_accumulation",
    "mean_projected_radius_px",
    "visible_gaussian_count",
    "mean_train_visibility_support",
)


def _json_default(value: Any) -> Any:
    return BASE._json_default(value)


def _write_json(path: Path, value: Any) -> None:
    BASE._write_json(path, value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    BASE._write_csv(path, rows)


def _write_table(root: Path, stem: str, rows: Sequence[Mapping[str, Any]], **extra: Any) -> None:
    BASE._write_table(root, stem, rows, extra)


def _read_json(path: Path) -> Dict[str, Any]:
    return BASE._read_json(path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    return BASE._read_csv(path)


def _sha256(path: Path) -> Optional[str]:
    return BASE._file_sha256(path)


def _run_text(command: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    return BASE._run_text(command, cwd)


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _rho(a: Sequence[float], b: Sequence[float]) -> float:
    return BASE._rho(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _kendall(a: Sequence[float], b: Sequence[float]) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.ptp(left[valid]) <= EPS or np.ptp(right[valid]) <= EPS:
        return float("nan")
    return float(scipy.stats.kendalltau(left[valid], right[valid]).statistic)


def _quantile(values: np.ndarray, q: float) -> float:
    finite = _finite(values)
    return float(np.quantile(finite, q)) if finite.size else float("nan")


def _camera_arrays(records: Sequence[Tuple[Any, ...]], model: Any) -> Dict[str, Dict[str, np.ndarray]]:
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for _index, view_id, camera, _batch in records:
        c2w = camera.camera_to_worlds[0].detach().float().cpu().numpy()
        forward = -c2w[:3, 2]
        forward = forward / max(float(np.linalg.norm(forward)), EPS)
        result[view_id] = {
            "center": c2w[:3, 3],
            "context": BASE._camera_context(model, camera),
            "view_direction": forward,
        }
    return result


def _nearest_features(query: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
    distances = np.linalg.norm(reference - query.reshape(1, -1), axis=1)
    ordered = np.sort(distances)
    k = min(3, ordered.size)
    return float(ordered[0]), float(ordered[:k].mean())


def _angular_features(query: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
    cosine = np.clip(reference @ query.reshape(-1), -1.0, 1.0)
    angles = np.sort(np.degrees(np.arccos(cosine)))
    k = min(3, angles.size)
    return float(angles[0]), float(angles[:k].mean())


def _standardized_distance(query: np.ndarray, reference: np.ndarray) -> float:
    center = reference.mean(axis=0)
    covariance = np.cov(reference, rowvar=False)
    inverse = np.linalg.pinv(np.atleast_2d(covariance), rcond=1e-8)
    delta = query - center
    return float(np.sqrt(max(float(delta @ inverse @ delta), 0.0)))


def _repo_state(repo: Path) -> Dict[str, Any]:
    return {
        "branch": _run_text(["git", "branch", "--show-current"], repo),
        "head": _run_text(["git", "rev-parse", "HEAD"], repo),
        "status_short": _run_text(["git", "status", "--short"], repo),
        "log_20": _run_text(["git", "log", "--oneline", "--decorate", "-20"], repo),
        "protected_files": {
            path: {"exists": (repo / path).exists(), "sha256": _sha256(repo / path)}
            for path in PROTECTED
        },
    }


def _inventory_scene(repo: Path, scene: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scene_cfg = FORMAL.SCENES[scene]
    data_root = repo / str(scene_cfg["data_path"])
    calibration_path = data_root / "sparse" / "0" / "images.bin"
    image_root = data_root / "images" / "ColorImage"
    train_path = data_root / "train_list.txt"
    test_path = data_root / "test_list.txt"
    val_path = data_root / "val_list.txt"
    calibrated = read_images_binary(calibration_path)
    calibrated_by_name = {Path(image.name).name: image for image in calibrated.values()}
    rgb_by_name = {path.name: path for path in image_root.iterdir() if path.is_file()}
    train = set(train_path.read_text(encoding="utf8").split())
    test = set(test_path.read_text(encoding="utf8").split())
    val = set(val_path.read_text(encoding="utf8").split())
    rows: List[Dict[str, Any]] = []
    for image_name in sorted(set(calibrated_by_name) | set(rgb_by_name)):
        view_id = Path(image_name).stem
        has_calibration = image_name in calibrated_by_name
        has_gt = image_name in rgb_by_name
        in_train = image_name in train
        in_test = image_name in test
        if in_train and in_test:
            split = "OTHER / UNKNOWN"
            reason = "image appears in both formal train and test lists"
        elif in_train:
            split = "FORMAL_TRAIN"
            reason = "used by locked OCMC optimization"
        elif in_test:
            split = "FORMAL_EVAL"
            reason = "original formal held-out camera"
        elif has_calibration and has_gt:
            split = "UNUSED_CALIBRATED_GT"
            reason = "not listed in formal train or test split"
        elif has_calibration:
            split = "UNUSED_NO_GT"
            reason = "calibration exists but source RGB is missing"
        elif has_gt:
            split = "UNUSED_NO_CALIBRATION"
            reason = "source RGB exists but COLMAP calibration is missing"
        else:
            split = "OTHER / UNKNOWN"
            reason = "unclassified source record"
        eligible = split == "UNUSED_CALIBRATED_GT"
        calibration = calibrated_by_name.get(image_name)
        rows.append({
            "scene": scene,
            "camera_id": view_id,
            "image_name": image_name,
            "image_path": str(rgb_by_name.get(image_name, image_root / image_name)),
            "camera_calibration_path": f"{calibration_path}#image_id={getattr(calibration, 'id', '')}",
            "formal_split": split,
            "used_in_training": bool(in_train),
            "used_in_formal_eval": bool(in_test),
            "listed_in_val": bool(image_name in val),
            "valid_gt": bool(has_gt),
            "valid_calibration": bool(has_calibration),
            "eligible_additional_heldout": bool(eligible),
            "ineligibility_reason": "" if eligible else reason,
            "split_source_train": str(train_path),
            "split_source_eval": str(test_path),
            "calibration_source": str(calibration_path),
        })
    counts = {label: sum(row["formal_split"] == label for row in rows) for label in (
        "FORMAL_TRAIN", "FORMAL_EVAL", "UNUSED_CALIBRATED_GT", "UNUSED_NO_GT",
        "UNUSED_NO_CALIBRATION", "OTHER / UNKNOWN",
    )}
    summary = {
        "scene": scene,
        "data_root": str(data_root),
        "calibration_source": str(calibration_path),
        "source_rgb_root": str(image_root),
        "calibrated_camera_count": len(calibrated_by_name),
        "source_rgb_count": len(rgb_by_name),
        "N_formal_train": counts["FORMAL_TRAIN"],
        "N_formal_eval": counts["FORMAL_EVAL"],
        "N_unused_calibrated_GT": counts["UNUSED_CALIBRATED_GT"],
        "N_total_genuine_heldout_available": counts["FORMAL_EVAL"] + counts["UNUSED_CALIBRATED_GT"],
        "heldout_coverage_adequate": counts["FORMAL_EVAL"] + counts["UNUSED_CALIBRATED_GT"] >= MIN_RELIABLE_HELDOUT,
        "minimum_reliable_heldout": MIN_RELIABLE_HELDOUT,
        "all_calibrated_have_gt": set(calibrated_by_name) <= set(rgb_by_name),
        "all_source_gt_calibrated": set(rgb_by_name) <= set(calibrated_by_name),
        "train_eval_disjoint": not bool(train & test),
        "train_eval_cover_all_calibrated": set(calibrated_by_name) == train | test,
        "val_equals_test": val == test,
        "classification_counts": counts,
    }
    return rows, summary


def inventory(repo: Path, output_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for scene in SCENES:
        scene_rows, summary = _inventory_scene(repo, scene)
        rows.extend(scene_rows)
        summaries.append(summary)
    _write_table(output_root, "camera_inventory", rows,
                 classification_unit="source ColorImage/COLMAP camera; derived fake-air images are not distinct cameras")
    _write_json(output_root / "heldout_coverage_summary.json", {
        "rows": summaries,
        "total_formal_train": sum(row["N_formal_train"] for row in summaries),
        "total_formal_eval": sum(row["N_formal_eval"] for row in summaries),
        "total_unused_calibrated_GT": sum(row["N_unused_calibrated_GT"] for row in summaries),
        "coverage_decision": "INADEQUATE_FOR_RELIABLE_CAMERA_NEIGHBOR_TEST",
    })
    return rows, summaries


def _runtime(gpu: str) -> Dict[str, Any]:
    base = FORMAL._runtime(gpu)
    base.update({
        "experiment": EXPERIMENT,
        "no_optimizer_step": True,
        "no_backward": True,
        "logical_device": "cuda:0",
    })
    return base


def _render_scene(repo: Path, output_root: Path, scene: str, gpu: str) -> Dict[str, Any]:
    started = time.perf_counter()
    runtime = _runtime(gpu)
    scene_cfg = FORMAL.SCENES[scene]
    source_dir = SOURCE_ROOT / scene
    checkpoint = source_dir / "checkpoints" / "C0" / f"step-{FINAL_STEP:09d}.ckpt"
    out_dir = output_root / "states" / scene
    render_dir = output_root / "renders" / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    branch = FORMAL._setup_branch(repo, scene_cfg, "C0")
    try:
        ckpt = FORMAL._load_checkpoint(branch, checkpoint)
        model = branch.pipeline.model
        train_records = FORMAL._train_records(branch.pipeline)
        eval_records = FORMAL._eval_records(branch.pipeline)
        train_ids = {record[1] for record in train_records}
        eval_ids = {record[1] for record in eval_records}
        if train_ids & eval_ids:
            raise RuntimeError(f"formal train/eval overlap in {scene}: {sorted(train_ids & eval_ids)}")
        camera_data = {
            "train": _camera_arrays(train_records, model),
            "eval": _camera_arrays(eval_records, model),
        }
        n_gaussians = int(model.means.shape[0])
        visibility_count = torch.zeros(n_gaussians, dtype=torch.int16)
        train_radii: List[np.ndarray] = []
        render_rows: List[Dict[str, Any]] = []
        for _index, view_id, camera, _batch in train_records:
            tick = time.perf_counter()
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
            visible = outputs["gaussian_visible_mask"].detach().bool().cpu().reshape(-1)
            n = min(n_gaussians, visible.numel())
            visibility_count[:n] += visible[:n].to(torch.int16)
            radii = model.radii.detach().float().cpu().numpy().reshape(-1)[:n]
            train_radii.append(radii[radii > 0])
            render_rows.append({
                "scene": scene, "split": "FORMAL_TRAIN_SUPPORT_ONLY", "camera_id": view_id,
                "render_seconds": time.perf_counter() - tick, "residual_target": False,
            })
            del outputs
            gc.collect()
        pooled_train_radii = np.concatenate(train_radii) if train_radii else np.empty(0, dtype=np.float32)
        large_footprint_threshold = _quantile(pooled_train_radii, 0.95)
        del train_radii, pooled_train_radii

        train_centers = np.asarray([camera_data["train"][view]["center"] for view in sorted(train_ids)])
        train_contexts = np.asarray([camera_data["train"][view]["context"] for view in sorted(train_ids)])
        train_directions = np.asarray([camera_data["train"][view]["view_direction"] for view in sorted(train_ids)])
        metrics_rows: List[Dict[str, Any]] = []
        predictor_rows: List[Dict[str, Any]] = []
        for _index, view_id, camera, batch in eval_records:
            tick = time.perf_counter()
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
            pred = outputs["pred_image"].detach().float().clamp(0, 1)
            metrics = MIC._metric_images(model, pred, gt)
            residual_rgb = (pred - gt).detach().float().cpu().numpy()
            squared = np.mean(np.square(residual_rgb), axis=-1)
            absolute = np.mean(np.abs(residual_rgb), axis=-1)
            radii = model.radii.detach().float().cpu().numpy().reshape(-1)
            visible = radii > 0
            visible_ids = np.flatnonzero(visible)
            visible_radii = radii[visible]
            support = visibility_count[torch.from_numpy(visible_ids)].float().numpy() if visible_ids.size else np.empty(0)
            accumulation = outputs["accumulation"].detach().float().cpu().numpy().reshape(-1)
            depth = outputs["depth"].detach().float().cpu().numpy().reshape(-1)
            tau = outputs["tau_D"].detach().float().cpu().numpy().reshape(-1)
            transmission = outputs["transmission"].detach().float().cpu().numpy().reshape(-1)
            beta_b = outputs["medium_bs"].detach().float().cpu().numpy().reshape(-1)
            beta_d = outputs["medium_attn"].detach().float().cpu().numpy().reshape(-1)
            camera_residual = outputs.get("camera_medium_delta_projected_raw")
            if isinstance(camera_residual, torch.Tensor):
                ocmc_magnitude = float(torch.linalg.norm(camera_residual.detach().float(), dim=-1).mean().cpu())
            else:
                ocmc_magnitude = float("nan")
            info = camera_data["eval"][view_id]
            center_nn, center_k3 = _nearest_features(info["center"], train_centers)
            context_nn, context_k3 = _nearest_features(info["context"], train_contexts)
            direction_nn, direction_k3 = _angular_features(info["view_direction"], train_directions)
            image_path = repo / str(scene_cfg["data_path"]) / "images" / "ColorImage" / f"{view_id}.png"
            calibration_path = repo / str(scene_cfg["data_path"]) / "sparse" / "0" / "images.bin"
            metrics_rows.append({
                "scene": scene, "camera_id": view_id, "formal_split": "FORMAL_EVAL",
                "unused_heldout_eligibility": False, "used_in_training": False,
                "checkpoint": str(checkpoint), "checkpoint_branch": "C0", "absolute_step": FINAL_STEP,
                "render_config": "classic/bounded_sh3/dir_xy_camera/OCMC_ON/RAOC_OFF",
                "image_path": str(image_path), "calibration_source": str(calibration_path),
                **metrics, "MAE": float(absolute.mean()),
                "median_absolute_rgb_residual": _quantile(absolute, 0.50),
                "p90_absolute_rgb_residual": _quantile(absolute, 0.90),
                "p95_absolute_rgb_residual": _quantile(absolute, 0.95),
                "p99_absolute_rgb_residual": _quantile(absolute, 0.99),
                "E_cam": float(squared.mean()), "height": int(pred.shape[0]), "width": int(pred.shape[1]),
            })
            predictor_rows.append({
                "scene": scene, "camera_id": view_id, "formal_split": "FORMAL_EVAL",
                "center_x": float(info["center"][0]), "center_y": float(info["center"][1]), "center_z": float(info["center"][2]),
                "context_x": float(info["context"][0]), "context_y": float(info["context"][1]), "context_z": float(info["context"][2]),
                "view_direction_x": float(info["view_direction"][0]), "view_direction_y": float(info["view_direction"][1]), "view_direction_z": float(info["view_direction"][2]),
                "center_nearest_train": center_nn, "center_knn3_mean": center_k3,
                "context_nearest_train": context_nn, "context_knn3_mean": context_k3,
                "context_standardized_distance": _standardized_distance(info["context"], train_contexts),
                "view_direction_nearest_angle_deg": direction_nn,
                "view_direction_knn3_angle_deg": direction_k3,
                "visible_gaussian_count": int(visible_ids.size),
                "mean_train_visibility_support": float(support.mean()) if support.size else float("nan"),
                "median_train_visibility_support": _quantile(support, 0.50),
                "fraction_visible_unseen_train": float(np.mean(support == 0)) if support.size else float("nan"),
                "fraction_visible_low_support": float(np.mean(support <= 1)) if support.size else float("nan"),
                "mean_projected_radius_px": float(visible_radii.mean()) if visible_radii.size else float("nan"),
                "median_projected_radius_px": _quantile(visible_radii, 0.50),
                "p90_projected_radius_px": _quantile(visible_radii, 0.90),
                "p95_projected_radius_px": _quantile(visible_radii, 0.95),
                "fraction_large_footprint": float(np.mean(visible_radii > large_footprint_threshold)) if visible_radii.size else float("nan"),
                "large_footprint_train_p95_threshold": large_footprint_threshold,
                "mean_depth": float(np.mean(depth)), "p90_depth": _quantile(depth, 0.90),
                "mean_tau": float(np.mean(tau)), "p90_tau": _quantile(tau, 0.90),
                "mean_transmission": float(np.mean(transmission)),
                "mean_accumulation": float(np.mean(accumulation)), "p10_accumulation": _quantile(accumulation, 0.10),
                "fraction_low_accumulation": float(np.mean(accumulation < 0.1)),
                "mean_beta_B": float(np.mean(beta_b)), "mean_beta_D": float(np.mean(beta_d)),
                "mean_ocmc_projected_camera_residual": ocmc_magnitude,
                "predictors_use_gt": False, "per_ray_jacobian_required": False,
            })
            pred_path = render_dir / f"{view_id}_pred.png"
            pred_u8 = np.clip(pred.detach().float().cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
            Image.fromarray(pred_u8).save(pred_path)
            render_rows.append({
                "scene": scene, "split": "FORMAL_EVAL", "camera_id": view_id,
                "render_seconds": time.perf_counter() - tick, "residual_target": True,
                "prediction_path": str(pred_path), "ground_truth_path": str(image_path),
            })
            del outputs, gt, pred, residual_rgb, squared, absolute
            gc.collect()
        flags = {
            key: getattr(model.config, key, None) for key in (
                "intrinsic_color_parameterization", "rasterize_mode", "medium_context_mode",
                "camera_medium_observability_enabled", "camera_medium_ray_adaptive_observability_enabled",
                "camera_medium_observability_strength", "b_inf_mode", "infinite_water_enabled",
                "medium_identifiability_enabled",
            )
        }
        manifest = {
            "scene": scene, "checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_branch": "C0", "absolute_step": int(ckpt["absolute_step"]),
            "ocmc_bundle_present": "ocmc_bundle" in ckpt, "ocmc_refresh_step": int(ckpt.get("ocmc_bundle", {}).get("step", -1)),
            "state_provenance": "formal C0 OCMC checkpoint; RAOC disabled", "flags": flags,
            "formal_train_count": len(train_records), "formal_eval_count": len(eval_records),
            "formal_train_ids": sorted(train_ids), "formal_eval_ids": sorted(eval_ids),
            "large_footprint_threshold_source": "pooled visible radii across all formal training cameras",
            "large_footprint_train_p95_threshold": large_footprint_threshold,
        }
        runtime["wall_seconds"] = time.perf_counter() - started
        result = {
            "scene": scene, "runtime": runtime, "checkpoint": manifest,
            "metrics_rows": metrics_rows, "predictor_rows": predictor_rows, "render_rows": render_rows,
            "train_camera_geometry": [{"camera_id": view, **{key: value.tolist() for key, value in camera_data["train"][view].items()}} for view in sorted(train_ids)],
            "eval_camera_geometry": [{"camera_id": view, **{key: value.tolist() for key, value in camera_data["eval"][view].items()}} for view in sorted(eval_ids)],
        }
        _write_json(out_dir / "scene_result.json", result)
        _write_json(out_dir / "checkpoint_manifest.json", manifest)
        _write_table(out_dir, "per_camera_metrics", metrics_rows)
        _write_table(out_dir, "per_camera_gt_free_predictors", predictor_rows)
        _write_table(out_dir, "render_manifest", render_rows)
        return result
    finally:
        FORMAL._release(branch)


def _effect_rows(scene_rows: Sequence[Mapping[str, Any]], population: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        selected = [row for row in scene_rows if row["scene"] == scene]
        for predictor, expected_sign in PREDICTORS.items():
            errors = np.asarray([float(row["E_cam"]) for row in selected])
            values = np.asarray([float(row[predictor]) for row in selected])
            rho = _rho(values, errors)
            enough_quantiles = len(selected) >= MIN_RELIABLE_HELDOUT
            ratio = float("nan")
            if enough_quantiles:
                low, high = np.quantile(values, [1 / 3, 2 / 3])
                ratio = float(errors[values >= high].mean() / max(float(errors[values <= low].mean()), EPS))
            rows.append({
                "scene": scene, "population": population, "predictor": predictor,
                "expected_error_direction": "positive" if expected_sign > 0 else "negative",
                "heldout_camera_count": len(selected), "spearman_rho": rho,
                "kendall_tau": _kendall(values, errors), "top_vs_bottom_E_cam_ratio": ratio,
                "direction_consistent": bool(np.isfinite(rho) and rho * expected_sign > 0),
                "adequate_camera_coverage": len(selected) >= MIN_RELIABLE_HELDOUT,
                "inference_status": "DESCRIPTIVE_SMALL_N" if selected else "NO_CAMERAS",
            })
    return rows


def _rank_residualized_rho(predictor: np.ndarray, error: np.ndarray, control: np.ndarray) -> float:
    valid = np.isfinite(predictor) & np.isfinite(error) & np.isfinite(control)
    if int(valid.sum()) < 3:
        return float("nan")
    px = scipy.stats.rankdata(predictor[valid])
    ey = scipy.stats.rankdata(error[valid])
    cz = scipy.stats.rankdata(control[valid])
    design = np.column_stack([np.ones(cz.size), cz])
    px = px - design @ np.linalg.lstsq(design, px, rcond=None)[0]
    ey = ey - design @ np.linalg.lstsq(design, ey, rcond=None)[0]
    return _rho(px, ey)


def _control_rows(joined: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        selected = [row for row in joined if row["scene"] == scene]
        errors = np.asarray([float(row["E_cam"]) for row in selected])
        for predictor, expected_sign in PREDICTORS.items():
            values = np.asarray([float(row[predictor]) for row in selected])
            raw = _rho(values, errors)
            for control in CONTROLS:
                if predictor == control:
                    continue
                confounder = np.asarray([float(row[control]) for row in selected])
                adjusted = _rank_residualized_rho(values, errors, confounder)
                rows.append({
                    "scene": scene, "predictor": predictor, "control": control,
                    "heldout_camera_count": len(selected), "raw_spearman_rho": raw,
                    "residualized_rank_spearman_rho": adjusted,
                    "expected_direction": "positive" if expected_sign > 0 else "negative",
                    "raw_expected_sign": bool(np.isfinite(raw) and raw * expected_sign > 0),
                    "controlled_expected_sign": bool(np.isfinite(adjusted) and adjusted * expected_sign > 0),
                    "method": "univariate within-scene rank residualization",
                    "inference_status": "DESCRIPTIVE_SMALL_N_NOT_A_MULTIVARIATE_CONTROL",
                })
    return rows


def _distance_matrix(rows: Sequence[Mapping[str, Any]], space: str) -> np.ndarray:
    if space == "center":
        values = np.asarray([[row[f"center_{axis}"] for axis in "xyz"] for row in rows], dtype=np.float64)
        return np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    if space == "context":
        values = np.asarray([[row[f"context_{axis}"] for axis in "xyz"] for row in rows], dtype=np.float64)
        return np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    values = np.asarray([[row[f"view_direction_{axis}"] for axis in "xyz"] for row in rows], dtype=np.float64)
    cosine = np.clip(values @ values.T, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _neighbor_rows(joined: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    neighbor: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    for scene in SCENES:
        selected = [row for row in joined if row["scene"] == scene]
        errors = np.asarray([float(row["E_cam"]) for row in selected])
        n = len(selected)
        adequate = n >= MIN_RELIABLE_HELDOUT
        for space in ("center", "context", "view_direction"):
            distances = _distance_matrix(selected, space)
            predictions = np.full(n, np.nan)
            for index in range(n):
                other = np.arange(n) != index
                weights = 1.0 / (distances[index, other] + 1e-6)
                predictions[index] = float(np.sum(weights * errors[other]) / max(float(weights.sum()), EPS))
            observed = _rho(predictions, errors)
            permutation_values: List[float] = []
            if adequate:
                rng = np.random.default_rng(BASE._stable_seed(scene, space, "camera-label-permutation"))
                for _ in range(PERMUTATIONS):
                    permuted = rng.permutation(errors)
                    perm_predictions = np.full(n, np.nan)
                    for index in range(n):
                        other = np.arange(n) != index
                        weights = 1.0 / (distances[index, other] + 1e-6)
                        perm_predictions[index] = float(np.sum(weights * permuted[other]) / max(float(weights.sum()), EPS))
                    permutation_values.append(_rho(perm_predictions, permuted))
            finite_null = _finite(permutation_values)
            neighbor.append({
                "scene": scene, "distance_space": space, "heldout_camera_count": n,
                "weighting_rule": "inverse_distance_all_other_heldout_cameras_eps_1e-6",
                "observed_leave_one_view_out_spearman": observed,
                "null_median": float(np.median(finite_null)) if finite_null.size else float("nan"),
                "null_p95": _quantile(finite_null, 0.95),
                "empirical_percentile": float(np.mean(finite_null <= observed)) if finite_null.size else float("nan"),
                "permutation_count": int(finite_null.size),
                "status": "VALID" if adequate else "INSUFFICIENT_CAMERA_COUNT",
            })
            if n >= 2:
                iu = np.triu_indices(n, 1)
                differences = np.abs(errors[:, None] - errors[None, :])
                nearest = np.argsort(np.where(np.eye(n, dtype=bool), np.inf, distances), axis=1)[:, 0]
                all_difference = float(differences[iu].mean())
                nearest_difference = float(np.abs(errors - errors[nearest]).mean())
                pair_rho = _rho(distances[iu], differences[iu])
                pairs.append({
                    "scene": scene, "distance_space": space, "heldout_camera_count": n,
                    "camera_pair_count": int(len(iu[0])),
                    "distance_vs_absolute_E_cam_difference_spearman": pair_rho,
                    "nearest_over_all_absolute_E_cam_difference": nearest_difference / max(all_difference, EPS),
                    "expected_positive_relation": bool(np.isfinite(pair_rho) and pair_rho > 0),
                    "status": "DESCRIPTIVE_SMALL_N" if not adequate else "VALID",
                })
    return neighbor, pairs


def _persistence(scene: str) -> Dict[str, Any]:
    rows = [row for row in _read_csv(SOURCE_ROOT / scene / "per_view_eval.csv") if row["branch"] == "C0" and row["split"] == "eval"]
    by_step = {(int(row["absolute_step"]), row["view_id"]): float(row["MSE"]) for row in rows}
    final_ids = sorted(view for step, view in by_step if step == FINAL_STEP)
    correlations = []
    for step in (10000, 13000):
        ids = [view for view in final_ids if (step, view) in by_step]
        correlations.append({
            "early_step": step, "spearman_to_14999": _rho([by_step[(step, view)] for view in ids], [by_step[(FINAL_STEP, view)] for view in ids]),
            "camera_count": len(ids),
        })
    return {
        "scene": scene, "formal_eval_camera_count": len(final_ids), "rows": correlations,
        "mean_rank_persistence": float(np.nanmean([row["spearman_to_14999"] for row in correlations])),
        "additional_heldout_added": 0, "status": "DESCRIPTIVE_ORIGINAL_FORMAL_EVAL_ONLY",
    }


def _join_rows(metrics: Sequence[Mapping[str, Any]], predictors: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(row["scene"], row["camera_id"]): row for row in predictors}
    joined = []
    for metric in metrics:
        key = (metric["scene"], metric["camera_id"])
        if key not in by_key:
            raise RuntimeError(f"missing predictor row for {key}")
        joined.append({**metric, **by_key[key]})
    return joined


def _scene_classifications(joined: Sequence[Mapping[str, Any]], effects: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for scene in SCENES:
        selected = [row for row in joined if row["scene"] == scene]
        values = np.asarray([float(row["E_cam"]) for row in selected])
        scene_effects = [row for row in effects if row["scene"] == scene]
        expected = [row for row in scene_effects if row["direction_consistent"]]
        result[scene] = {
            "classification": "CAMERA_RESIDUAL_DATA_LIMITED",
            "heldout_camera_count": len(selected), "minimum_reliable_heldout": MIN_RELIABLE_HELDOUT,
            "coverage_adequate": False,
            "E_cam_mean": float(values.mean()), "E_cam_std": float(values.std()),
            "E_cam_min": float(values.min()), "E_cam_max": float(values.max()),
            "descriptive_expected_direction_predictor_count": len(expected),
            "descriptive_predictors": [row["predictor"] for row in expected],
            "replication_adjudication": "WITHHELD_BECAUSE_N_LT_5",
            "control_conclusion": "descriptive univariate rank residualization only; too few cameras for reliable confounder control",
        }
    return result


def _actionability(effects: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for predictor, expected_sign in PREDICTORS.items():
        candidate = [row for row in effects if row["predictor"] == predictor]
        signed = [float(row["spearman_rho"]) * expected_sign for row in candidate if np.isfinite(float(row["spearman_rho"]))]
        control_rows = [row for row in controls if row["predictor"] == predictor]
        rows.append({
            "predictor": predictor,
            "raw_expected_sign_scene_count": sum(row["direction_consistent"] for row in candidate),
            "median_expected_signed_rho": float(np.median(signed)) if signed else float("nan"),
            "controlled_expected_sign_count": sum(row["controlled_expected_sign"] for row in control_rows),
            "control_comparison_count": len(control_rows),
            "adequate_scene_count": sum(row["adequate_camera_coverage"] for row in candidate),
            "gt_free": True, "training_or_render_time_available": True,
            "second_order_derivatives_required": False, "per_ray_jacobian_required": False,
            "eval_label_tuning": False, "replicated_three_scenes": False, "actionable": False,
        })
    descriptive = max(rows, key=lambda row: (row["raw_expected_sign_scene_count"], row["median_expected_signed_rho"]))
    return {
        "rows": rows,
        "reliably_actionable_predictor": None,
        "strongest_descriptive_predictor": descriptive["predictor"],
        "strongest_descriptive_predictor_warning": "selected only to summarize small-N directions; not validated or actionable",
        "actionability_decision": "NO_ACTIONABLE_PREDICTOR_ESTABLISHED_BECAUSE_DATA_LIMITED",
    }


def _independence_rows(joined: Sequence[Mapping[str, Any]], predictor: str) -> List[Dict[str, Any]]:
    comparison_variables = (
        "mean_ocmc_projected_camera_residual",
        "mean_depth",
        "mean_tau",
        "mean_transmission",
        "mean_accumulation",
        "mean_projected_radius_px",
        "visible_gaussian_count",
        "mean_train_visibility_support",
    )
    rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        selected = [row for row in joined if row["scene"] == scene]
        predictor_values = [float(row[predictor]) for row in selected]
        errors = [float(row["E_cam"]) for row in selected]
        for comparison in comparison_variables:
            comparison_values = [float(row[comparison]) for row in selected]
            rows.append({
                "scene": scene,
                "strongest_descriptive_predictor": predictor,
                "comparison_variable": comparison,
                "heldout_camera_count": len(selected),
                "predictor_vs_comparison_spearman": _rho(predictor_values, comparison_values),
                "comparison_vs_E_cam_spearman": _rho(comparison_values, errors),
                "predictor_vs_E_cam_spearman": _rho(predictor_values, errors),
                "status": "DESCRIPTIVE_SMALL_N_INDEPENDENCE_NOT_ESTABLISHED",
            })
    return rows


def _make_figures(output_root: Path, results: Sequence[Mapping[str, Any]], joined: Sequence[Mapping[str, Any]], effects: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]]) -> None:
    plt = BASE.plt
    figure_dir = output_root / "cross_scene" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    for ax, result in zip(axes.flat, results):
        train = result["train_camera_geometry"]
        held = [row for row in joined if row["scene"] == result["scene"]]
        points = np.asarray([row["context"] for row in train] + [[row[f"context_{axis}"] for axis in "xyz"] for row in held])
        projected = PCA(n_components=2).fit_transform(points)
        n_train = len(train)
        ax.scatter(projected[:n_train, 0], projected[:n_train, 1], s=18, c="#777777", label="formal train")
        scatter = ax.scatter(projected[n_train:, 0], projected[n_train:, 1], s=58, c=[row["E_cam"] for row in held], cmap="magma", edgecolors="black", linewidths=0.5, label="formal eval")
        ax.set_title(result["scene"])
        ax.set_xlabel("camera-context PCA 1")
        ax.set_ylabel("camera-context PCA 2")
        fig.colorbar(scatter, ax=ax, label="E_cam")
    fig.tight_layout()
    fig.savefig(figure_dir / "camera_context_map_residual.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, scene in zip(axes.flat, SCENES):
        selected = [row for row in joined if row["scene"] == scene]
        ax.scatter([row["center_nearest_train"] for row in selected], [row["E_cam"] for row in selected], color="#35618f", s=45)
        for row in selected:
            ax.annotate(row["camera_id"], (row["center_nearest_train"], row["E_cam"]), fontsize=7)
        ax.set_title(scene)
        ax.set_xlabel("nearest training-camera center distance")
        ax.set_ylabel("E_cam")
    fig.tight_layout()
    fig.savefig(figure_dir / "camera_center_novelty_vs_error.png", dpi=160)
    plt.close(fig)

    effect_predictors = ("center_nearest_train", "view_direction_nearest_angle_deg", "mean_train_visibility_support", "fraction_visible_low_support")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(effect_predictors))
    offsets = np.linspace(-0.24, 0.24, len(SCENES))
    for offset, scene in zip(offsets, SCENES):
        selected = {row["predictor"]: row for row in effects if row["scene"] == scene}
        ax.scatter(x + offset, [selected[p]["spearman_rho"] for p in effect_predictors], label=scene, s=45)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, effect_predictors, rotation=18, ha="right")
    ax.set_ylabel("Spearman rho with E_cam")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "cross_scene_effect_sizes.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    center_pairs = [row for row in pair_rows if row["distance_space"] == "center"]
    ax.bar([row["scene"] for row in center_pairs], [row["distance_vs_absolute_E_cam_difference_spearman"] for row in center_pairs], color="#5c7c48")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("rho(center distance, |delta E_cam|)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figure_dir / "camera_pair_distance_effect.png", dpi=160)
    plt.close(fig)


def _research_note(summary: Mapping[str, Any]) -> None:
    coverage = {row["scene"]: row for row in summary["heldout_coverage"]}
    classes = summary["per_scene_classification"]
    persistence = {row["scene"]: row for row in summary["persistence"]}
    effects = summary["primary_effects"]
    neighbor = summary["center_neighbor_analysis"]
    pair_effects = summary["center_pair_analysis"]
    effect_by_key = {(row["scene"], row["predictor"]): row for row in effects}
    neighbor_by_scene = {row["scene"]: row for row in neighbor}
    pair_by_scene = {row["scene"]: row for row in pair_effects}
    center_rhos = "; ".join(
        f"{scene} {effect_by_key[(scene, 'center_nearest_train')]['spearman_rho']:.3f}"
        for scene in SCENES
    )
    context_rhos = "; ".join(
        f"{scene} {effect_by_key[(scene, 'context_nearest_train')]['spearman_rho']:.3f}"
        for scene in SCENES
    )
    direction_rhos = "; ".join(
        f"{scene} {effect_by_key[(scene, 'view_direction_nearest_angle_deg')]['spearman_rho']:.3f}"
        for scene in SCENES
    )
    support_rhos = "; ".join(
        f"{scene} {effect_by_key[(scene, 'fraction_visible_unseen_train')]['spearman_rho']:.3f}"
        for scene in SCENES
    )
    neighbor_scores = "; ".join(
        f"{scene} {neighbor_by_scene[scene]['observed_leave_one_view_out_spearman']:.3f} ({neighbor_by_scene[scene]['status']})"
        for scene in SCENES
    )
    pair_rhos = "; ".join(
        f"{scene} {pair_by_scene[scene]['distance_vs_absolute_E_cam_difference_spearman']:.3f}"
        for scene in SCENES
    )
    lines = [
        "# Focused Candidate-C Camera Residual Replication (2026-08-31)",
        "", "## 1. Motivation", "",
        "This frozen-state audit tests whether camera/view novelty or training-view support predicts genuine held-out camera error after locked OCMC. It does not train or design an intervention.",
        "", "## 2. Previous Candidate-C Evidence", "",
        "EXPERIMENTAL FACT: the preceding audit classified Candidate C as tentative: camera-neighborhood structure was descriptive in IUI3-RedSea and JapaneseGradens-RedSea, but only 13 formal eval cameras existed across four scenes.",
        "", "## 3. Held-out Camera Coverage Audit", "",
        "CODE FACT: COLMAP `images.bin`, source `ColorImage` files, formal split lists, and the dataparser-loaded train/eval IDs were cross-checked. Counts are " + "; ".join(f"{scene} {coverage[scene]['N_formal_train']}/{coverage[scene]['N_formal_eval']} train/eval" for scene in SCENES) + ".",
        "", "## 4. Additional Unused-Camera Discovery", "",
        "DATA FACT: no `UNUSED_CALIBRATED_GT` camera exists in any scene. Every calibrated source camera is already in formal train or formal eval, and every source RGB has calibration.",
        "", "## 5. Frozen OCMC Rendering Protocol", "",
        "CONFIG FACT: only formal C0 step-14999 checkpoints were loaded. Rendering used classic rasterization, bounded SH3, `dir_xy_camera`, OCMC on, and RAOC off. Formal train views supplied only GT-free camera/support geometry; their residuals were never held-out labels. No backward or optimizer step occurred.",
        "", "## 6. Camera-Level Error Statistics", "",
        "QUANTITATIVE RESULT: E_cam mean/std/range are " + "; ".join(f"{scene} {classes[scene]['E_cam_mean']:.6g}/{classes[scene]['E_cam_std']:.6g}/[{classes[scene]['E_cam_min']:.6g}, {classes[scene]['E_cam_max']:.6g}]" for scene in SCENES) + ".",
        "", "## 7. Camera-Center/Context Novelty", "",
        f"QUANTITATIVE RESULT: nearest-center Spearman rho with E_cam is {center_rhos}. Exact OCMC-context nearest-distance rho is {context_rhos}. All are descriptive because N=3/4/3/3. Center and context ranks can coincide because the OCMC context is an affine scene normalization of camera center.",
        "", "## 8. View-Direction Novelty", "",
        f"QUANTITATIVE RESULT: nearest angular-novelty rho with E_cam is {direction_rhos}. Minimum and fixed 3-NN angular novelty were evaluated independently, without a tuned center-angle score.",
        "", "## 9. Training-View Support", "",
        f"CODE FACT: each held-out visible Gaussian was assigned its exact formal-training visibility count. The fraction-visible-with-zero-training-support rho with E_cam is {support_rhos}. Mean/median support and zero/one-view support fractions are GT-free; contribution-weighted support was not claimed.",
        "", "## 10. Optical/Geometric Controls", "",
        "QUANTITATIVE RESULT: depth, tau, transmission, accumulation, projected footprint, visible count, and visibility support were examined by one-control-at-a-time within-scene rank residualization. With fewer than five cameras, these are descriptive controls rather than a reliable multivariate adjustment. Independence from these confounders and from OCMC projected camera-residual magnitude is not established.",
        "", "## 11. Leave-One-View-Out Neighbor Analysis", "",
        f"QUANTITATIVE RESULT: center-space leave-one-view-out scores are {neighbor_scores}. All four scenes are `INSUFFICIENT_CAMERA_COUNT`; no camera-label permutation test was run or interpreted. Center-distance versus absolute E_cam-difference rho is {pair_rhos}, also descriptive only.",
        "", "## 12. Formal-Eval vs Additional-Heldout Comparison", "",
        "DATA FACT: the additional-heldout population is empty, so combined genuine-heldout results equal formal-eval-only results and no independent expansion comparison exists.",
        "", "## 13. Cross-Scene Replication", "",
        "INFERENCE: no scene has adequate coverage for the protocol's reliable camera-neighbor test. The previous IUI3 and JapaneseGradens center-neighbor difference ratios reproduce numerically on the same formal-eval cameras, but this is not an expanded replication and cannot be upgraded to replicated evidence. Curasao and Panama do not newly establish Candidate C.",
        "", "## 14. GT-Free Actionability", "",
        f"INFERENCE: no GT-free predictor is actionable under this audit. `{summary['actionability']['strongest_descriptive_predictor']}` is only the strongest small-N directional summary, not a validated predictor.",
        "", "## 15. Final Candidate-C Decision", "",
        "INFERENCE: `C_DATA_LIMITED`. This does not support or refute Candidate C; it records that every scene remains below five genuine held-out cameras and no additional cameras exist.",
        "", "## 16. Closed / Remaining Uncertainties", "",
        "RAOC remains closed and OCMC remains frozen. Candidate C remains unresolved rather than supported. Camera-neighbor effects, optical-confounder independence, and OCMC independence cannot be reliably adjudicated with current held-out coverage.",
        "", "## 17. ONE Next Task", "",
        "HYPOTHESIS: `DATA-SPLIT-FEASIBILITY-AUDIT`. Audit whether moving enough currently trained cameras to reach at least five (preferably eight) held-out views per scene, followed by four fresh locked-OCMC retrains, is scientifically worth the cost. Do not retrain during that audit.",
        "",
    ]
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def aggregate(repo: Path, output_root: Path, inventory_rows: Optional[Sequence[Mapping[str, Any]]] = None, coverage_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    results = [_read_json(output_root / "states" / scene / "scene_result.json") for scene in SCENES]
    if inventory_rows is None:
        inventory_rows = _read_json(output_root / "camera_inventory.json")["rows"]
    if coverage_rows is None:
        coverage_rows = _read_json(output_root / "heldout_coverage_summary.json")["rows"]
    metrics = [row for result in results for row in result["metrics_rows"]]
    predictors = [row for result in results for row in result["predictor_rows"]]
    renders = [row for result in results for row in result["render_rows"]]
    joined = _join_rows(metrics, predictors)
    effects = _effect_rows(joined, "FORMAL_EVAL_ONLY")
    additional_effects = [{
        "scene": scene, "population": "ADDITIONAL_HELDOUT", "heldout_camera_count": 0,
        "status": "NO_UNUSED_CALIBRATED_GT_CAMERAS",
    } for scene in SCENES]
    combined_effects = [{**row, "population": "COMBINED_GENUINE_HELDOUT"} for row in effects]
    controls = _control_rows(joined)
    neighbor, pairs = _neighbor_rows(joined)
    persistence = [_persistence(scene) for scene in SCENES]
    classifications = _scene_classifications(joined, effects, controls)
    actionability = _actionability(effects, controls)
    independence_rows = _independence_rows(joined, actionability["strongest_descriptive_predictor"])
    independence = {
        "strongest_descriptive_predictor": actionability["strongest_descriptive_predictor"],
        "compared_against": sorted({row["comparison_variable"] for row in independence_rows}),
        "conclusion": "NOT_ESTABLISHED_DATA_LIMITED",
        "reason": "zero scenes have at least five genuine held-out cameras; OCMC global observability gate is scene-level and does not vary by camera",
        "rows": independence_rows,
    }
    cross_scene = {
        "adequate_scene_count": 0, "same_direction_replicated_scene_count": 0,
        "STRONG_CAMERA_RESIDUAL_REPLICATION": 0, "WEAK_CAMERA_RESIDUAL_REPLICATION": 0,
        "CAMERA_RESIDUAL_NOT_REPLICATED": 0, "CAMERA_RESIDUAL_DATA_LIMITED": 4,
        "raw_metrics_pooled_across_scenes": False,
        "conclusion": "CROSS_SCENE_REPLICATION_NOT_ADJUDICABLE",
    }
    decision = {
        "decision": "C_DATA_LIMITED", "candidate": "C", "OCMC_frozen": True, "RAOC_closed": True,
        "reason": "all scenes have fewer than five genuine held-out cameras and no unused calibrated-GT cameras were found",
        "actionable_gt_free_predictor": None,
        "one_next_task": "DATA-SPLIT-FEASIBILITY-AUDIT",
        "future_audit_only": {
            "minimum_heldout_per_scene": 5, "preferred_heldout_per_scene": 8,
            "additional_cameras_needed_for_minimum": {"Curasao": 2, "IUI3-RedSea": 1, "JapaneseGradens-RedSea": 2, "Panama": 2},
            "estimated_required_ocmc_retrains_if_resplit_is_later_approved": 4,
            "retraining_performed_in_this_task": False,
        },
    }
    _write_table(output_root, "per_camera_metrics", metrics)
    _write_table(output_root, "per_camera_gt_free_predictors", predictors)
    _write_table(output_root, "render_manifest", renders)
    _write_table(output_root, "formal_eval_only_effects", effects, warning="all scene-level effects are descriptive because N<5")
    _write_table(output_root, "additional_heldout_effects", additional_effects)
    _write_table(output_root, "combined_heldout_effects", combined_effects, identical_to="formal_eval_only_effects")
    _write_table(output_root, "camera_neighbor_analysis", neighbor)
    _write_table(output_root, "camera_pair_distance_analysis", pairs)
    _write_table(output_root, "optical_geometry_controls", controls)
    _write_table(output_root, "gt_free_predictor_summary", actionability["rows"])
    _write_table(output_root, "ocmc_independence_analysis", independence_rows)
    _write_json(output_root / "per_scene_classification.json", classifications)
    _write_json(output_root / "cross_scene_replication.json", cross_scene)
    _write_json(output_root / "ocmc_independence.json", independence)
    _write_json(output_root / "candidate_c_final_decision.json", decision)
    _write_json(output_root / "ocmc_checkpoint_manifest.json", {"rows": [result["checkpoint"] for result in results]})
    _write_json(output_root / "difficult_camera_persistence.json", {"rows": persistence})
    _make_figures(output_root, results, joined, effects, pairs)
    summary = {
        "experiment": EXPERIMENT, "scenes": list(SCENES),
        "heldout_coverage": list(coverage_rows), "camera_inventory_count": len(inventory_rows),
        "per_scene_classification": classifications, "cross_scene_replication": cross_scene,
        "actionability": actionability, "ocmc_independence": independence,
        "persistence": persistence, "candidate_c_final_decision": decision,
        "primary_effects": effects,
        "center_neighbor_analysis": [row for row in neighbor if row["distance_space"] == "center"],
        "center_pair_analysis": [row for row in pairs if row["distance_space"] == "center"],
        "no_training": True, "optimizer_step_called": False, "backward_called": False,
        "training_cameras_used_as_residual_targets": False,
        "formal_eval_only_equals_combined": True, "outputs_committed": False,
        "research_note": str(RESEARCH_NOTE),
    }
    _write_json(output_root / "final_summary.json", summary)
    _research_note(summary)
    return summary


def launch(repo: Path, output_root: Path) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    starting = _repo_state(repo)
    _write_json(output_root / "repo_state.json", starting)
    inventory_rows, coverage_rows = inventory(repo, output_root)
    _write_json(output_root / "environment.json", {
        "python": sys.executable, "python_version": sys.version.split()[0], "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "launcher_visible_device_count": torch.cuda.device_count(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""), "allowed_physical_gpus": [6, 7, 8, 9],
        "worker_policy": "one physical GPU exposed per worker; logical cuda:0",
    })
    _write_json(output_root / "launcher_manifest.json", {
        "scene_gpu_assignment": SCENE_GPUS, "source_root": str(SOURCE_ROOT),
        "previous_audit_root": str(PREVIOUS_ROOT), "no_training": True,
    })
    processes: Dict[str, subprocess.Popen[Any]] = {}
    handles: Dict[str, Any] = {}
    try:
        for scene, gpu in SCENE_GPUS.items():
            handle = (output_root / "logs" / f"{scene}.log").open("w", encoding="utf8")
            handles[scene] = handle
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [str(PYTHON), str(Path(__file__).resolve()), "--scene", scene, "--gpu", gpu, "--repo", str(repo), "--output-root", str(output_root)]
            processes[scene] = subprocess.Popen(command, cwd=str(repo), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        statuses = {scene: process.wait() for scene, process in processes.items()}
    finally:
        for handle in handles.values():
            handle.close()
    _write_json(output_root / "worker_status.json", statuses)
    if any(code != 0 for code in statuses.values()):
        raise RuntimeError(f"scene worker failure: {statuses}")
    summary = aggregate(repo, output_root, inventory_rows, coverage_rows)
    ending = _repo_state(repo)
    _write_json(output_root / "repo_state_after.json", ending)
    protected_unchanged = all(
        starting["protected_files"][path]["sha256"] == ending["protected_files"][path]["sha256"]
        for path in PROTECTED
    )
    _write_json(output_root / "protected_file_integrity.json", {
        "unchanged": protected_unchanged, "before": starting["protected_files"], "after": ending["protected_files"],
    })
    if not protected_unchanged:
        raise RuntimeError("protected historical/Q50-Q80 file hash changed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = args.output_root.resolve()
    if args.scene:
        if args.gpu is None:
            raise ValueError("--scene requires --gpu")
        result = _render_scene(repo, output_root, args.scene, args.gpu)
        printable = {"scene": result["scene"], "runtime": result["runtime"], "checkpoint": result["checkpoint"]}
    elif args.inventory_only:
        inventory_rows, coverage_rows = inventory(repo, output_root)
        printable = {"camera_count": len(inventory_rows), "coverage": coverage_rows}
    elif args.aggregate_only:
        result = aggregate(repo, output_root)
        printable = result["candidate_c_final_decision"]
    else:
        result = launch(repo, output_root)
        printable = result["candidate_c_final_decision"]
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
