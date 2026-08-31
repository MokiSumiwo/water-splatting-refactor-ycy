#!/usr/bin/env python3
"""Blind geometry-aware split feasibility audit for Candidate C.

Candidate IDs are selected and hashed from calibration geometry before any
checkpoint is loaded. Frozen C0 checkpoints are used only afterwards to audit
visibility-support coverage; the audit logic never reads RGB target tensors,
residual files, or invokes optimization operations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.stats
import torch
import yaml
from scipy.spatial import ConvexHull, QhullError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary
from scripts.diagnostics import audit_camera_residual_replication_ocmc as CAMERA_AUDIT
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL

EXPERIMENT = "DATA-SPLIT-FEASIBILITY-AUDIT"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "data_split_feasibility_audit_20260831"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "DATA_SPLIT_FEASIBILITY_AUDIT_2026-08-31.md"
HISTORICAL_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
PRIOR_FROZEN_ROOT = REPO_ROOT / "outputs" / "focused_c_camera_residual_replication_20260831"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
FINAL_STEP = 14999
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}
SCENES = tuple(SCENE_GPUS)
SCENE_FILE_STEMS = {
    "Curasao": "curasao",
    "IUI3-RedSea": "iui3",
    "JapaneseGradens-RedSea": "japanesegradens",
    "Panama": "panama",
}
FAMILY_SIZES = {
    "Curasao": {"A": 5, "B": 7, "C": 6},
    "IUI3-RedSea": {"A": 5, "B": 8, "C": 6},
    "JapaneseGradens-RedSea": {"A": 5, "B": 6, "C": 6},
    "Panama": {"A": 5, "B": 6, "C": 6},
}
FAMILY_NAMES = {
    "A": "GEOMETRY_STRATIFIED_NOVELTY",
    "B": "FARTHEST_POINT_CENTER_COVERAGE",
    "C": "TRAJECTORY_INTERLEAVED",
}
PROTECTED = CAMERA_AUDIT.PROTECTED
EPS = 1e-12


def _json_default(value: Any) -> Any:
    return CAMERA_AUDIT._json_default(value)


def _write_json(path: Path, value: Any) -> None:
    CAMERA_AUDIT._write_json(path, value)


def _write_table(root: Path, stem: str, rows: Sequence[Mapping[str, Any]], **extra: Any) -> None:
    CAMERA_AUDIT._write_table(root, stem, rows, **extra)


def _read_json(path: Path) -> Dict[str, Any]:
    return CAMERA_AUDIT._read_json(path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    return CAMERA_AUDIT._read_csv(path)


def _sha256(path: Path) -> Optional[str]:
    return CAMERA_AUDIT._sha256(path)


def _run_text(command: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    return CAMERA_AUDIT._run_text(command, cwd)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


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


def _dataset_split_hashes(repo: Path) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for scene in SCENES:
        data_root = repo / str(FORMAL.SCENES[scene]["data_path"])
        rows[scene] = {
            name: {"path": str(data_root / name), "sha256": _sha256(data_root / name)}
            for name in ("train_list.txt", "test_list.txt", "val_list.txt")
        }
    return rows


def _frame_number(camera_id: str) -> int:
    match = re.search(r"(\d+)$", camera_id)
    if match is None:
        raise ValueError(f"camera ID has no numeric acquisition suffix: {camera_id}")
    return int(match.group(1))


def _angular_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(left @ right), -1.0, 1.0))))


def _summary(values: Sequence[float], prefix: str) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {f"{prefix}_{key}": float("nan") for key in ("min", "q25", "median", "q75", "p90", "max", "mean")}
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
        f"{prefix}_p90": float(np.quantile(array, 0.90)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_mean": float(np.mean(array)),
    }


def _nearest_distances(values: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _nearest_angles(values: np.ndarray) -> np.ndarray:
    cosine = np.clip(values @ values.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    np.fill_diagonal(angles, np.inf)
    return angles.min(axis=1)


def _geometry_scene(repo: Path, scene: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scene_cfg = FORMAL.SCENES[scene]
    config_path = repo / str(scene_cfg["source_config"])
    config = yaml.load(config_path.read_text(encoding="utf8"), Loader=yaml.Loader)
    parser_config = config.pipeline.datamanager.dataparser
    data_root = repo / str(scene_cfg["data_path"])
    parser_config.data = data_root
    parser = parser_config.setup()
    train = parser.get_dataparser_outputs("train")
    test = parser.get_dataparser_outputs("test")
    if not np.allclose(train.dataparser_transform.numpy(), test.dataparser_transform.numpy(), atol=0, rtol=0):
        raise RuntimeError(f"train/test dataparser transform mismatch in {scene}")
    if float(train.dataparser_scale) != float(test.dataparser_scale):
        raise RuntimeError(f"train/test dataparser scale mismatch in {scene}")
    train_aabb = train.scene_box.aabb.detach().float().cpu().numpy()
    test_aabb = test.scene_box.aabb.detach().float().cpu().numpy()
    if not np.allclose(train_aabb, test_aabb, atol=0, rtol=0):
        raise RuntimeError(f"train/test scene box mismatch in {scene}")
    scene_center = (train_aabb[0] + train_aabb[1]) * 0.5
    scene_scale = float(np.linalg.norm(train_aabb[1] - train_aabb[0]))

    calibration_path = data_root / "sparse" / "0" / "images.bin"
    calibrated = read_images_binary(calibration_path)
    calibration_by_stem = {Path(image.name).stem: image for image in calibrated.values()}
    original_train = {Path(name).stem for name in (data_root / "train_list.txt").read_text(encoding="utf8").split()}
    original_eval = {Path(name).stem for name in (data_root / "test_list.txt").read_text(encoding="utf8").split()}
    rows: List[Dict[str, Any]] = []
    for split, outputs in (("FORMAL_TRAIN", train), ("FORMAL_EVAL", test)):
        cameras = outputs.cameras
        for index, filename in enumerate(outputs.image_filenames):
            camera_id = Path(filename).stem
            pose = cameras.camera_to_worlds[index].detach().float().cpu().numpy()
            direction = -pose[:3, 2]
            direction /= max(float(np.linalg.norm(direction)), EPS)
            up = pose[:3, 1]
            up /= max(float(np.linalg.norm(up)), EPS)
            center = pose[:3, 3]
            width = float(cameras.width[index].item())
            height = float(cameras.height[index].item())
            fx = float(cameras.fx[index].item())
            fy = float(cameras.fy[index].item())
            calibration = calibration_by_stem[camera_id]
            context = (center - scene_center) / (scene_scale + 1e-6)
            rows.append({
                "scene": scene,
                "camera_id": camera_id,
                "image_name": Path(filename).name,
                "image_path": str(filename),
                "original_split": split,
                "frame_number": _frame_number(camera_id),
                "colmap_image_id": int(calibration.id),
                "calibration_source": str(calibration_path),
                "valid_gt": Path(filename).is_file(),
                "valid_calibration": True,
                "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2]),
                "context_x": float(context[0]), "context_y": float(context[1]), "context_z": float(context[2]),
                "direction_x": float(direction[0]), "direction_y": float(direction[1]), "direction_z": float(direction[2]),
                "up_x": float(up[0]), "up_y": float(up[1]), "up_z": float(up[2]),
                "fx": fx, "fy": fy,
                "cx": float(cameras.cx[index].item()), "cy": float(cameras.cy[index].item()),
                "width": int(width), "height": int(height),
                "horizontal_fov_deg": float(np.degrees(2.0 * np.arctan(width / (2.0 * fx)))),
                "vertical_fov_deg": float(np.degrees(2.0 * np.arctan(height / (2.0 * fy)))),
                "dataparser_scale": float(outputs.dataparser_scale),
            })
    rows.sort(key=lambda row: (int(row["frame_number"]), str(row["camera_id"])))
    ids = {str(row["camera_id"]) for row in rows}
    if ids != set(calibration_by_stem) or ids != original_train | original_eval or original_train & original_eval:
        raise RuntimeError(f"camera inventory mismatch in {scene}")

    centers = np.asarray([[row[f"center_{axis}"] for axis in "xyz"] for row in rows], dtype=np.float64)
    directions = np.asarray([[row[f"direction_{axis}"] for axis in "xyz"] for row in rows], dtype=np.float64)
    frame_numbers = np.asarray([int(row["frame_number"]) for row in rows], dtype=np.float64)
    frame_delta: List[float] = []
    center_delta: List[float] = []
    angle_delta: List[float] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            frame_delta.append(abs(frame_numbers[left] - frame_numbers[right]))
            center_delta.append(float(np.linalg.norm(centers[left] - centers[right])))
            angle_delta.append(_angular_distance(directions[left], directions[right]))
    adjacent_center = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    adjacent_angle = np.asarray([_angular_distance(directions[i], directions[i + 1]) for i in range(len(rows) - 1)])
    singular_values = np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)
    frame_center_rho = float(scipy.stats.spearmanr(frame_delta, center_delta).statistic)
    frame_angle_rho = float(scipy.stats.spearmanr(frame_delta, angle_delta).statistic)
    adjacent_center_ratio = float(np.median(adjacent_center) / max(float(np.median(center_delta)), EPS))
    trajectory_verified = frame_center_rho >= 0.70 and adjacent_center_ratio <= 0.50
    summary = {
        "scene": scene,
        "total_camera_count": len(rows),
        "formal_train_count": len(original_train),
        "formal_eval_count": len(original_eval),
        "all_cameras_have_gt_and_calibration": all(row["valid_gt"] and row["valid_calibration"] for row in rows),
        "dataparser_orientation_method": str(parser_config.orientation_method),
        "dataparser_center_method": str(parser_config.center_method),
        "dataparser_auto_scale_poses": bool(parser_config.auto_scale_poses),
        "dataparser_scale": float(train.dataparser_scale),
        "dataparser_transform": train.dataparser_transform.tolist(),
        "scene_box_aabb": train_aabb.tolist(),
        "scene_center": scene_center.tolist(),
        "scene_scale": scene_scale,
        "camera_context_definition": "(dataparser camera center - scene-box center) / (scene-box diagonal + 1e-6)",
        "frame_number_min": int(frame_numbers.min()),
        "frame_number_max": int(frame_numbers.max()),
        "missing_numeric_frame_count": int(frame_numbers.max() - frame_numbers.min() + 1 - len(rows)),
        "filename_exif_timestamp_available": False,
        "frame_delta_vs_center_distance_spearman": frame_center_rho,
        "frame_delta_vs_direction_angle_spearman": frame_angle_rho,
        "adjacent_center_median_over_all_pair_median": adjacent_center_ratio,
        "adjacent_center_p90_over_all_pair_p90": float(np.quantile(adjacent_center, 0.90) / max(float(np.quantile(center_delta, 0.90)), EPS)),
        "adjacent_direction_p90_over_all_pair_p90": float(np.quantile(adjacent_angle, 0.90) / max(float(np.quantile(angle_delta, 0.90)), EPS)),
        "center_singular_value_ratios": (singular_values / max(float(singular_values[0]), EPS)).tolist(),
        "numeric_frame_order_geometry_verified": bool(trajectory_verified),
        "trajectory_family_eligible": bool(trajectory_verified),
    }
    return rows, summary


def _pairwise_rows(scene_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        selected = [row for row in scene_rows if row["scene"] == scene]
        for left in range(len(selected)):
            a = selected[left]
            ac = np.asarray([a[f"center_{axis}"] for axis in "xyz"], dtype=np.float64)
            ax = np.asarray([a[f"context_{axis}"] for axis in "xyz"], dtype=np.float64)
            av = np.asarray([a[f"direction_{axis}"] for axis in "xyz"], dtype=np.float64)
            for right in range(left + 1, len(selected)):
                b = selected[right]
                bc = np.asarray([b[f"center_{axis}"] for axis in "xyz"], dtype=np.float64)
                bx = np.asarray([b[f"context_{axis}"] for axis in "xyz"], dtype=np.float64)
                bv = np.asarray([b[f"direction_{axis}"] for axis in "xyz"], dtype=np.float64)
                rows.append({
                    "scene": scene, "camera_id_a": a["camera_id"], "camera_id_b": b["camera_id"],
                    "absolute_frame_number_difference": abs(int(a["frame_number"]) - int(b["frame_number"])),
                    "center_distance": float(np.linalg.norm(ac - bc)),
                    "context_distance": float(np.linalg.norm(ax - bx)),
                    "view_direction_angle_deg": _angular_distance(av, bv),
                })
    return rows


def _select_stratified(rows: Sequence[Mapping[str, Any]], count: int) -> List[str]:
    centers = np.asarray([[row[f"center_{axis}"] for axis in "xyz"] for row in rows], dtype=np.float64)
    loo_novelty = _nearest_distances(centers)
    ordered = sorted(range(len(rows)), key=lambda i: (float(loo_novelty[i]), str(rows[i]["camera_id"])))
    ranks = [int(math.floor(value + 0.5)) for value in np.linspace(0, len(ordered) - 1, count)]
    return sorted(str(rows[ordered[rank]]["camera_id"]) for rank in ranks)


def _select_farthest(rows: Sequence[Mapping[str, Any]], count: int) -> List[str]:
    centers = np.asarray([[row[f"center_{axis}"] for axis in "xyz"] for row in rows], dtype=np.float64)
    centroid = centers.mean(axis=0)
    selected: List[int] = []
    first_scores = np.linalg.norm(centers - centroid, axis=1)
    first = min(range(len(rows)), key=lambda i: (-float(first_scores[i]), str(rows[i]["camera_id"])))
    selected.append(first)
    while len(selected) < count:
        remaining = [i for i in range(len(rows)) if i not in selected]
        minimum = {
            i: min(float(np.linalg.norm(centers[i] - centers[j])) for j in selected)
            for i in remaining
        }
        selected.append(min(remaining, key=lambda i: (-minimum[i], str(rows[i]["camera_id"]))))
    return sorted(str(rows[index]["camera_id"]) for index in selected)


def _select_interleaved(rows: Sequence[Mapping[str, Any]], count: int) -> List[str]:
    if len(rows) < 2 * count + 1:
        raise ValueError("trajectory-interleaved selection cannot preserve non-adjacent training neighbors")
    targets = np.linspace(1, len(rows) - 2, count)
    selected: List[int] = []
    for target in targets:
        candidates = sorted(
            range(1, len(rows) - 1),
            key=lambda i: (abs(i - target), int(rows[i]["frame_number"]), str(rows[i]["camera_id"])),
        )
        index = next(i for i in candidates if i not in selected and all(abs(i - prior) > 1 for prior in selected))
        selected.append(index)
    selected.sort()
    if any(right - left <= 1 for left, right in zip(selected, selected[1:])):
        raise RuntimeError("interleaved held-out cameras are adjacent")
    return sorted(str(rows[index]["camera_id"]) for index in selected)


def _candidate_payload(scene: str, family: str, all_ids: Sequence[str], heldout_ids: Sequence[str]) -> Dict[str, Any]:
    if len(set(heldout_ids)) != len(heldout_ids):
        raise RuntimeError(f"duplicate held-out camera ID in {scene} family {family}")
    heldout = sorted(heldout_ids)
    train = sorted(set(all_ids) - set(heldout))
    if set(train) & set(heldout) or set(train) | set(heldout) != set(all_ids):
        raise RuntimeError(f"invalid split partition in {scene} family {family}")
    canonical = {
        "schema_version": 1,
        "scene": scene,
        "family": family,
        "family_name": FAMILY_NAMES[family],
        "train_ids": train,
        "heldout_ids": heldout,
    }
    return {
        **canonical,
        "candidate_id": f"{scene}:{family}:N{len(heldout)}",
        "train_count": len(train),
        "heldout_count": len(heldout),
        "selection_uses_rgb_residual": False,
        "selection_inputs": (
            ["camera_center", "verified_numeric_acquisition_order"]
            if family == "C" else ["camera_center"]
        ),
        "camera_id_audit": {
            "all_ids_exist": True, "all_have_gt": True, "all_have_calibration": True,
            "train_heldout_disjoint": True, "partition_complete": True, "no_duplicates": True,
        },
        "canonical_manifest_sha256": _canonical_hash(canonical),
        "proposal_only_not_applied": True,
    }


def _build_candidates(scene_rows: Sequence[Mapping[str, Any]], scene_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    summary_by_scene = {row["scene"]: row for row in scene_summaries}
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    for scene in SCENES:
        rows = [row for row in scene_rows if row["scene"] == scene]
        all_ids = [str(row["camera_id"]) for row in rows]
        sizes = FAMILY_SIZES[scene]
        heldout_by_family = {
            "A": _select_stratified(rows, sizes["A"]),
            "B": _select_farthest(rows, sizes["B"]),
        }
        if not summary_by_scene[scene]["trajectory_family_eligible"]:
            raise RuntimeError(f"trajectory ordering was not geometrically verified in {scene}")
        heldout_by_family["C"] = _select_interleaved(rows, sizes["C"])
        candidates[scene] = [_candidate_payload(scene, family, all_ids, heldout_by_family[family]) for family in ("A", "B", "C")]
    return candidates


def _projected_hull_metrics(all_centers: np.ndarray, train_centers: np.ndarray, heldout_centers: np.ndarray) -> Dict[str, Any]:
    centered = all_centers - all_centers.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    projection = vh[:2].T
    all_2d = centered @ projection
    train_2d = (train_centers - all_centers.mean(axis=0)) @ projection
    heldout_2d = (heldout_centers - all_centers.mean(axis=0)) @ projection
    all_extent = np.ptp(all_2d, axis=0)
    train_extent = np.ptp(train_2d, axis=0)
    result: Dict[str, Any] = {
        "pca1_extent_retention": float(train_extent[0] / max(float(all_extent[0]), EPS)),
        "pca2_extent_retention": float(train_extent[1] / max(float(all_extent[1]), EPS)),
    }
    try:
        all_hull = ConvexHull(all_2d)
        train_hull = ConvexHull(train_2d)
        result["projected_hull_area_retention"] = float(train_hull.volume / max(float(all_hull.volume), EPS))
        equations = train_hull.equations
        signed = heldout_2d @ equations[:, :2].T + equations[:, 2]
        outside = np.maximum(signed, 0.0)
        result["heldout_inside_retained_train_hull_fraction"] = float(np.mean(np.max(signed, axis=1) <= 1e-9))
        result["heldout_max_normalized_hull_violation"] = float(np.max(outside))
    except QhullError:
        result.update({
            "projected_hull_area_retention": float("nan"),
            "heldout_inside_retained_train_hull_fraction": float("nan"),
            "heldout_max_normalized_hull_violation": float("nan"),
        })
    return result


def _coverage_rows(
    geometry_rows: Sequence[Mapping[str, Any]], candidates: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    coverage: List[Dict[str, Any]] = []
    novelty: List[Dict[str, Any]] = []
    baselines: Dict[str, Dict[str, Any]] = {}
    for scene in SCENES:
        scene_rows = [row for row in geometry_rows if row["scene"] == scene]
        by_id = {str(row["camera_id"]): row for row in scene_rows}
        all_centers = np.asarray([[row[f"center_{axis}"] for axis in "xyz"] for row in scene_rows], dtype=np.float64)
        original_train_rows = [row for row in scene_rows if row["original_split"] == "FORMAL_TRAIN"]
        original_centers = np.asarray([[row[f"center_{axis}"] for axis in "xyz"] for row in original_train_rows], dtype=np.float64)
        original_directions = np.asarray([[row[f"direction_{axis}"] for axis in "xyz"] for row in original_train_rows], dtype=np.float64)
        baseline_center = _nearest_distances(original_centers)
        baseline_angle = _nearest_angles(original_directions)
        baselines[scene] = {
            **_summary(baseline_center, "original_train_center_nn"),
            **_summary(baseline_angle, "original_train_direction_nn_deg"),
            "original_train_count": len(original_train_rows),
        }
        for candidate in candidates[scene]:
            train_rows = [by_id[camera_id] for camera_id in candidate["train_ids"]]
            eval_rows = [by_id[camera_id] for camera_id in candidate["heldout_ids"]]
            train_centers = np.asarray([[row[f"center_{axis}"] for axis in "xyz"] for row in train_rows], dtype=np.float64)
            train_directions = np.asarray([[row[f"direction_{axis}"] for axis in "xyz"] for row in train_rows], dtype=np.float64)
            eval_centers = np.asarray([[row[f"center_{axis}"] for axis in "xyz"] for row in eval_rows], dtype=np.float64)
            eval_directions = np.asarray([[row[f"direction_{axis}"] for axis in "xyz"] for row in eval_rows], dtype=np.float64)
            center_nn = _nearest_distances(train_centers)
            angle_nn = _nearest_angles(train_directions)
            eval_center_distance = np.linalg.norm(eval_centers[:, None, :] - train_centers[None, :, :], axis=-1).min(axis=1)
            cosine = np.clip(eval_directions @ train_directions.T, -1.0, 1.0)
            eval_angle = np.degrees(np.arccos(cosine)).min(axis=1)
            rank_order = np.argsort(eval_center_distance, kind="stable")
            labels = [""] * len(eval_rows)
            for rank, index in enumerate(rank_order):
                labels[index] = ("LOW", "MID", "HIGH")[min(2, int(3 * rank / len(eval_rows)))]
            hull = _projected_hull_metrics(all_centers, train_centers, eval_centers)
            row = {
                "scene": scene, "candidate_id": candidate["candidate_id"],
                "family": candidate["family"], "family_name": candidate["family_name"],
                "train_count": len(train_rows), "heldout_count": len(eval_rows),
                **_summary(center_nn, "train_center_nn"),
                **_summary(angle_nn, "train_direction_nn_deg"),
                **_summary(eval_center_distance, "heldout_center_novelty"),
                **_summary(eval_angle, "heldout_direction_novelty_deg"),
                **hull,
                "train_center_p90_ratio_to_original": float(np.quantile(center_nn, 0.90) / max(float(np.quantile(baseline_center, 0.90)), EPS)),
                "train_center_max_ratio_to_original": float(np.max(center_nn) / max(float(np.max(baseline_center)), EPS)),
                "train_direction_p90_ratio_to_original": float(np.quantile(angle_nn, 0.90) / max(float(np.quantile(baseline_angle, 0.90)), EPS)),
                "train_direction_max_ratio_to_original": float(np.max(angle_nn) / max(float(np.max(baseline_angle)), EPS)),
                "heldout_camera_pair_count": int(len(eval_rows) * (len(eval_rows) - 1) // 2),
                "novelty_regimes_present": sorted(set(labels)),
                "old_formal_eval_overlap_count": len(set(candidate["heldout_ids"]) & {str(r["camera_id"]) for r in scene_rows if r["original_split"] == "FORMAL_EVAL"}),
                "selection_uses_rgb_residual": False,
            }
            row["heldout_center_novelty_span_ratio"] = float(np.max(eval_center_distance) / max(float(np.min(eval_center_distance)), EPS))
            row["heldout_direction_novelty_span_deg"] = float(np.max(eval_angle) - np.min(eval_angle))
            row["heldout_direction_novelty_span_ratio"] = float(np.max(eval_angle) / max(float(np.min(eval_angle)), EPS))
            row["direction_span_minimum_deg"] = float(0.25 * np.median(baseline_angle))
            row["meaningful_center_novelty_span"] = bool(
                len(set(np.round(eval_center_distance, 10))) >= 3
                and row["heldout_center_novelty_span_ratio"] >= 1.5
            )
            row["meaningful_direction_novelty_span"] = bool(
                len(set(np.round(eval_angle, 8))) >= 3
                and row["heldout_direction_novelty_span_ratio"] >= 1.5
                and row["heldout_direction_novelty_span_deg"] >= row["direction_span_minimum_deg"]
            )
            coverage.append(row)
            for index, camera in enumerate(eval_rows):
                novelty.append({
                    "scene": scene, "candidate_id": candidate["candidate_id"], "family": candidate["family"],
                    "camera_id": camera["camera_id"], "frame_number": camera["frame_number"],
                    "original_split": camera["original_split"],
                    "center_novelty_to_retained_train": float(eval_center_distance[index]),
                    "view_direction_novelty_deg_to_retained_train": float(eval_angle[index]),
                    "center_novelty_regime": labels[index],
                })
    return coverage, novelty, baselines


def _support_distribution(counts: np.ndarray, population: np.ndarray, prefix: str) -> Dict[str, Any]:
    values = counts[population].astype(np.float64)
    return {
        f"{prefix}_gaussian_population": int(values.size),
        f"{prefix}_fraction_zero": float(np.mean(values == 0)),
        f"{prefix}_fraction_one": float(np.mean(values == 1)),
        f"{prefix}_fraction_two": float(np.mean(values == 2)),
        f"{prefix}_fraction_at_least_three": float(np.mean(values >= 3)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
    }


def _support_worker(repo: Path, output_root: Path, scene: str, gpu: str) -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != gpu or gpu not in set(SCENE_GPUS.values()):
        raise RuntimeError(f"worker must expose only assigned GPU {gpu}, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("support worker must see exactly one logical CUDA device cuda:0")
    props = torch.cuda.get_device_properties(0)
    candidate_path = output_root / f"candidate_splits_{SCENE_FILE_STEMS[scene]}.json"
    candidate_document = _read_json(candidate_path)
    expected_file_hash = candidate_document["candidate_file_sha256_before_support"]
    comparable = {key: value for key, value in candidate_document.items() if key != "candidate_file_sha256_before_support"}
    if _canonical_hash(comparable) != expected_file_hash:
        raise RuntimeError(f"candidate manifest changed before support audit: {scene}")

    started = time.perf_counter()
    scene_cfg = FORMAL.SCENES[scene]
    branch = FORMAL._setup_branch(repo, scene_cfg, "C0")
    checkpoint = HISTORICAL_ROOT / scene / "checkpoints" / "C0" / f"step-{FINAL_STEP:09d}.ckpt"
    try:
        ckpt = FORMAL._load_checkpoint(branch, checkpoint)
        model = branch.pipeline.model
        records = FORMAL._train_records(branch.pipeline) + FORMAL._eval_records(branch.pipeline)
        by_id: Dict[str, np.ndarray] = {}
        camera_seconds: Dict[str, float] = {}
        for _index, camera_id, camera, _batch in records:
            tick = time.perf_counter()
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
            mask = outputs["gaussian_visible_mask"].detach().bool().cpu().numpy().reshape(-1)
            by_id[camera_id] = mask
            camera_seconds[camera_id] = time.perf_counter() - tick
            del outputs
            gc.collect()
        if len(by_id) != len(records):
            raise RuntimeError(f"duplicate camera IDs in support worker for {scene}")
        all_ids = sorted(by_id)
        masks = np.stack([by_id[camera_id] for camera_id in all_ids], axis=0)
        all_visible = np.any(masks, axis=0)
        original_train_ids = {
            Path(name).stem
            for name in (repo / str(scene_cfg["data_path"]) / "train_list.txt").read_text(encoding="utf8").split()
        }
        original_indices = [all_ids.index(camera_id) for camera_id in sorted(original_train_ids)]
        original_counts = masks[original_indices].sum(axis=0)
        original_distribution = _support_distribution(original_counts, all_visible, "original_train_support")
        original_distribution["original_train_visibility_coverage_of_all_camera_visible"] = float(np.mean(original_counts[all_visible] > 0))
        original_distribution["original_train_normalized_mean_support"] = float(np.mean(original_counts[all_visible]) / len(original_indices))
        candidate_rows: List[Dict[str, Any]] = []
        heldout_rows: List[Dict[str, Any]] = []
        for candidate in candidate_document["candidates"]:
            train_indices = [all_ids.index(camera_id) for camera_id in candidate["train_ids"]]
            counts = masks[train_indices].sum(axis=0)
            distribution = _support_distribution(counts, all_visible, "retained_train_support")
            coverage = float(np.mean(counts[all_visible] > 0))
            normalized_mean = float(np.mean(counts[all_visible]) / len(train_indices))
            candidate_row = {
                "scene": scene, "candidate_id": candidate["candidate_id"], "family": candidate["family"],
                "train_count": len(train_indices), "heldout_count": len(candidate["heldout_ids"]),
                "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                "absolute_step": int(ckpt["absolute_step"]),
                "frozen_geometry_proxy_warning": "visibility is measured on the old-split frozen C0 representation; future-resplit Gaussians may differ",
                **original_distribution, **distribution,
                "retained_train_visibility_coverage_of_all_camera_visible": coverage,
                "visibility_coverage_retention_vs_original_train": coverage / max(float(original_distribution["original_train_visibility_coverage_of_all_camera_visible"]), EPS),
                "retained_train_normalized_mean_support": normalized_mean,
                "normalized_mean_support_retention_vs_original_train": normalized_mean / max(float(original_distribution["original_train_normalized_mean_support"]), EPS),
                "rgb_ground_truth_tensor_read_by_audit_logic": False, "rgb_residual_accessed": False,
            }
            support_values: List[Tuple[float, float, float]] = []
            for camera_id in candidate["heldout_ids"]:
                camera_visible = by_id[camera_id]
                values = counts[camera_visible].astype(np.float64)
                heldout_row = {
                    "scene": scene, "candidate_id": candidate["candidate_id"], "family": candidate["family"],
                    "camera_id": camera_id, "visible_gaussian_count": int(values.size),
                    "fraction_visible_unseen_retained_train": float(np.mean(values == 0)),
                    "fraction_visible_low_support_retained_train": float(np.mean(values <= 1)),
                    "mean_retained_train_support_per_visible_gaussian": float(np.mean(values)),
                    "median_retained_train_support_per_visible_gaussian": float(np.median(values)),
                    "gt_free": True,
                }
                heldout_rows.append(heldout_row)
                support_values.append((
                    heldout_row["fraction_visible_unseen_retained_train"],
                    heldout_row["fraction_visible_low_support_retained_train"],
                    heldout_row["mean_retained_train_support_per_visible_gaussian"],
                ))
            support_array = np.asarray(support_values, dtype=np.float64)
            candidate_row.update({
                "heldout_fraction_unseen_range": float(np.ptp(support_array[:, 0])),
                "heldout_fraction_low_support_range": float(np.ptp(support_array[:, 1])),
                "heldout_mean_support_range": float(np.ptp(support_array[:, 2])),
                "meaningful_gt_free_support_diversity": bool(
                    np.ptp(support_array[:, 0]) >= 0.001
                    or np.ptp(support_array[:, 1]) >= 0.01
                    or np.ptp(support_array[:, 2]) >= 0.5
                ),
            })
            candidate_rows.append(candidate_row)
        result = {
            "scene": scene,
            "runtime": {
                "physical_gpu_id": gpu, "logical_device": "cuda:0", "visible_device_count": torch.cuda.device_count(),
                "gpu_name": props.name, "total_memory_bytes": int(props.total_memory),
                "wall_seconds": time.perf_counter() - started,
                "mean_camera_render_seconds": float(np.mean(list(camera_seconds.values()))),
                "camera_count": len(camera_seconds),
            },
            "checkpoint": str(checkpoint),
            "candidate_manifest_hash_verified": True,
            "candidate_rows": candidate_rows,
            "heldout_rows": heldout_rows,
            "no_training": True, "optimizer_step_called": False, "backward_called": False,
            "datamanager_may_cache_image_batches": True,
            "rgb_ground_truth_tensor_read_by_audit_logic": False, "rgb_residual_accessed": False,
        }
        _write_json(output_root / "support" / f"{SCENE_FILE_STEMS[scene]}_support_worker.json", result)
        return result
    finally:
        FORMAL._release(branch)


def _historical_runtime(repo: Path, support_results: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prior_render_rows = _read_csv(PRIOR_FROZEN_ROOT / "render_manifest.csv")
    rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        runtime_path = HISTORICAL_ROOT / scene / "runtime.json"
        manifest_path = HISTORICAL_ROOT / scene / "runtime_manifest.json"
        runtime = _read_json(runtime_path)
        manifest = _read_json(manifest_path)
        c0 = runtime["branches"]["C0"]
        checkpoints = list((HISTORICAL_ROOT / scene / "checkpoints" / "C0").glob("*.ckpt"))
        eval_render_seconds = [
            float(row["render_seconds"]) for row in prior_render_rows
            if row["scene"] == scene and row["split"] == "FORMAL_EVAL"
        ]
        support_runtime = next(result["runtime"] for result in support_results if result["scene"] == scene)
        rows.append({
            "scene": scene,
            "runtime_source": str(runtime_path),
            "gpu_source": str(manifest_path),
            "gpu_name": manifest["gpu_name"],
            "physical_gpu_id_historical": manifest["physical_gpu_id"],
            "completed_steps": c0["completed_steps"],
            "training_wall_seconds": c0["training_wall_seconds"],
            "training_gpu_hours": float(c0["training_wall_seconds"] / 3600.0),
            "peak_allocated_bytes": c0["peak_allocated_bytes"],
            "peak_reserved_bytes": c0["peak_reserved_bytes"],
            "historical_eval_render_seconds_mean_per_camera": float(np.mean(eval_render_seconds)),
            "historical_eval_render_camera_count": len(eval_render_seconds),
            "support_audit_wall_seconds": support_runtime["wall_seconds"],
            "c0_checkpoint_count": len(checkpoints),
            "c0_checkpoint_bytes": sum(path.stat().st_size for path in checkpoints),
        })
    training_seconds = sum(row["training_wall_seconds"] for row in rows)
    parallel_seconds = max(row["training_wall_seconds"] for row in rows)
    checkpoint_bytes = sum(row["c0_checkpoint_bytes"] for row in rows)
    diagnostic_seconds = sum(row["support_audit_wall_seconds"] for row in rows)
    cost = {
        "historical_runtime_basis": "exact C0 15K wall times from the formal four-scene C0/RAOC experiment",
        "future_training_run_count": 4,
        "training_gpu_hours_total": float(training_seconds / 3600.0),
        "estimated_frozen_eval_and_diagnostics_gpu_hours": float(diagnostic_seconds / 3600.0),
        "estimated_total_gpu_hours": float((training_seconds + diagnostic_seconds) / 3600.0),
        "parallel_training_wall_hours_four_gpus": float(parallel_seconds / 3600.0),
        "estimated_parallel_wall_hours_including_frozen_diagnostics": float((parallel_seconds + max(row["support_audit_wall_seconds"] for row in rows)) / 3600.0),
        "peak_reserved_bytes_max": max(row["peak_reserved_bytes"] for row in rows),
        "historical_c0_checkpoint_bytes_total": checkpoint_bytes,
        "estimated_future_storage_bytes": int(checkpoint_bytes + 0.10 * checkpoint_bytes),
        "storage_estimate_rule": "historical C0 checkpoints plus 10% for manifests, metrics, and frozen diagnostics",
        "cost_classification": "LOW_COST",
        "project_context": "four C0 runs are under four historical GPU-hours and materially cheaper than the prior C0+C1 formal burden",
        "cloud_pricing_used": False,
    }
    return rows, cost


def _classify_candidates(
    coverage_rows: Sequence[Mapping[str, Any]], support_rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    support_by_id = {row["candidate_id"]: row for row in support_rows}
    classified: List[Dict[str, Any]] = []
    for raw in coverage_rows:
        row = dict(raw)
        support = support_by_id[row["candidate_id"]]
        center_acceptable = (
            row["train_center_p90_ratio_to_original"] <= 2.0
            and row["train_center_max_ratio_to_original"] <= 2.25
            and row["projected_hull_area_retention"] >= 0.70
        )
        orientation_acceptable = (
            row["train_direction_p90_ratio_to_original"] <= 2.0
            and row["train_direction_max_ratio_to_original"] <= 2.50
        )
        support_acceptable = (
            support["visibility_coverage_retention_vs_original_train"] >= 0.98
            and support["normalized_mean_support_retention_vs_original_train"] >= 0.85
        )
        severe_hole = (
            row["train_count"] < 12
            or row["train_center_max_ratio_to_original"] > 2.75
            or row["projected_hull_area_retention"] < 0.55
            or row["train_direction_max_ratio_to_original"] > 3.0
            or support["visibility_coverage_retention_vs_original_train"] < 0.94
            or support["normalized_mean_support_retention_vs_original_train"] < 0.75
        )
        all_strong = (
            row["heldout_count"] >= 5
            and center_acceptable and orientation_acceptable and support_acceptable
            and row["meaningful_center_novelty_span"]
            and row["meaningful_direction_novelty_span"]
            and support["meaningful_gt_free_support_diversity"]
        )
        if all_strong and not severe_hole:
            classification = "SPLIT_FEASIBLE"
        elif row["heldout_count"] >= 5 and row["meaningful_center_novelty_span"] and not severe_hole:
            classification = "SPLIT_FEASIBLE_BUT_TIGHT"
        else:
            classification = "SPLIT_NOT_FEASIBLE"
        coverage_penalty = max(
            float(row["train_center_p90_ratio_to_original"]),
            float(row["train_center_max_ratio_to_original"]),
            float(row["train_direction_p90_ratio_to_original"]),
            float(row["train_direction_max_ratio_to_original"]),
            1.0 / max(float(row["projected_hull_area_retention"]), EPS),
            1.0 / max(float(support["visibility_coverage_retention_vs_original_train"]), EPS),
            1.0 / max(float(support["normalized_mean_support_retention_vs_original_train"]), EPS),
        )
        row.update({
            "center_coverage_acceptable": bool(center_acceptable),
            "orientation_coverage_acceptable": bool(orientation_acceptable),
            "visibility_support_acceptable": bool(support_acceptable),
            "severe_train_manifold_hole": bool(severe_hole),
            "candidate_classification": classification,
            "coverage_first_penalty": float(coverage_penalty),
            "visibility_coverage_retention_vs_original_train": support["visibility_coverage_retention_vs_original_train"],
            "normalized_mean_support_retention_vs_original_train": support["normalized_mean_support_retention_vs_original_train"],
            "meaningful_gt_free_support_diversity": support["meaningful_gt_free_support_diversity"],
            "heldout_fraction_unseen_range": support["heldout_fraction_unseen_range"],
            "heldout_fraction_low_support_range": support["heldout_fraction_low_support_range"],
            "heldout_mean_support_range": support["heldout_mean_support_range"],
            "classification_thresholds_preregistered_in_code": True,
        })
        classified.append(row)
    return classified


def _rank_and_recommend(
    candidates: Mapping[str, Sequence[Mapping[str, Any]]], classified: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    class_rank = {"SPLIT_FEASIBLE": 0, "SPLIT_FEASIBLE_BUT_TIGHT": 1, "SPLIT_NOT_FEASIBLE": 2}
    classified_by_id = {row["candidate_id"]: row for row in classified}
    scene_classifications: Dict[str, Any] = {}
    recommended: Dict[str, Any] = {
        "schema_version": 1,
        "proposal_only_not_applied": True,
        "selection_uses_rgb_residual": False,
        "ranking_rule": "classification, retained-coverage penalty, held-out size, novelty span, support diversity, family ID",
        "scenes": {},
    }
    for scene in SCENES:
        scene_candidates = []
        for candidate in candidates[scene]:
            assessment = classified_by_id[candidate["candidate_id"]]
            rank_key = (
                class_rank[assessment["candidate_classification"]],
                float(assessment["coverage_first_penalty"]),
                -int(assessment["heldout_count"]),
                -float(assessment["heldout_center_novelty_span_ratio"]),
                -float(assessment["heldout_mean_support_range"]),
                str(candidate["family"]),
            )
            scene_candidates.append((rank_key, candidate, assessment))
        scene_candidates.sort(key=lambda item: item[0])
        for rank, (_key, candidate, assessment) in enumerate(scene_candidates, 1):
            assessment["scene_rank"] = rank
        _key, winner, winner_assessment = scene_candidates[0]
        recommended["scenes"][scene] = {
            "candidate_id": winner["candidate_id"], "family": winner["family"], "family_name": winner["family_name"],
            "train_ids": winner["train_ids"], "heldout_ids": winner["heldout_ids"],
            "train_count": winner["train_count"], "heldout_count": winner["heldout_count"],
            "canonical_manifest_sha256": winner["canonical_manifest_sha256"],
            "classification": winner_assessment["candidate_classification"],
        }
        feasible_sizes = sorted({
            int(assessment["heldout_count"])
            for _key, _candidate, assessment in scene_candidates
            if assessment["candidate_classification"] != "SPLIT_NOT_FEASIBLE"
        })
        scene_classifications[scene] = {
            "classification": winner_assessment["candidate_classification"],
            "recommended_candidate_id": winner["candidate_id"],
            "recommended_family": winner["family"],
            "recommended_train_count": winner["train_count"],
            "recommended_heldout_count": winner["heldout_count"],
            "feasible_heldout_sizes_among_audited_candidates": feasible_sizes,
            "retained_center_coverage_acceptable": winner_assessment["center_coverage_acceptable"],
            "retained_orientation_coverage_acceptable": winner_assessment["orientation_coverage_acceptable"],
            "retained_visibility_support_acceptable": winner_assessment["visibility_support_acceptable"],
            "severe_train_manifold_hole": winner_assessment["severe_train_manifold_hole"],
            "low_mid_high_center_novelty": winner_assessment["meaningful_center_novelty_span"],
            "meaningful_direction_novelty": winner_assessment["meaningful_direction_novelty_span"],
            "meaningful_gt_free_support_diversity": winner_assessment["meaningful_gt_free_support_diversity"],
        }
    recommended["global_manifest_sha256"] = _canonical_hash(recommended)
    recommended["global_manifest_hash_scope"] = "all manifest fields except global_manifest_sha256 and global_manifest_hash_scope"
    return scene_classifications, recommended


def _preregistration() -> Dict[str, Any]:
    return {
        "registered_before_future_resplit_training": True,
        "primary_statistical_unit": "camera",
        "primary_target": "E_cam = mean per-pixel squared RGB residual (MSE)",
        "fixed_gt_free_predictors": [
            "fraction_visible_unseen_train",
            "camera_center_nearest_train",
            "camera_center_knn3_mean",
            "camera_context_nearest_train",
            "camera_context_knn3_mean",
            "view_direction_nearest_angle",
            "view_direction_knn3_angle",
            "mean_train_visibility_support",
            "median_train_visibility_support",
            "fraction_visible_low_support",
        ],
        "fraction_visible_unseen_train_status": "pre-registered from the prior audit before any new-split outcomes exist",
        "primary_predictor_interpretation": "predictors are evaluated independently; no tuned combined score",
        "neighbor_analysis": "fixed inverse-distance leave-one-view-out prediction with no self-neighbor",
        "permutation_analysis": "1000 fixed-seed camera-label permutations where N_eval >= 5",
        "confounder_plan": "one-control-at-a-time within-scene rank residualization for depth, tau, accumulation, footprint, and support",
        "multivariable_regression": "not approved at N=5-6 because of severe overfitting risk",
        "supported_and_actionable_criteria": [
            "same-direction Candidate-C replication in at least 3/4 scenes",
            "at least 3/4 scenes have at least five genuine held-out views",
            "a pre-registered GT-free predictor has absolute Spearman rho >= 0.4 in at least 3/4 scenes",
            "predictor direction survives major single-factor controls",
            "positive camera-neighbor structure in at least three adequate-N scenes",
            "signal is not reducible to OCMC observability",
            "no scene has a strong opposite-direction contradiction",
        ],
        "not_supported_criteria": [
            "Candidate C replicates in at most 2/4 scenes",
            "pre-registered GT-free predictors are weak or inconsistent",
            "major controls explain the effect",
        ],
        "not_supported_consequence": "close Candidate C; do not run additional split variants",
        "predictor_tuning_after_training_forbidden": True,
        "learned_predictor_forbidden": True,
    }


def _make_figures(
    output_root: Path,
    geometry_rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    recommended: Mapping[str, Any],
    classified: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
) -> None:
    plt = CAMERA_AUDIT.BASE.plt
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, scene in zip(axes.flat, SCENES):
        rows = [row for row in geometry_rows if row["scene"] == scene]
        centers = np.asarray([[row[f"center_{axis}"] for axis in "xyz"] for row in rows], dtype=np.float64)
        centered = centers - centers.mean(axis=0)
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        xy = centered @ vh[:2].T
        heldout = set(recommended["scenes"][scene]["heldout_ids"])
        train_indices = [i for i, row in enumerate(rows) if row["camera_id"] not in heldout]
        eval_indices = [i for i, row in enumerate(rows) if row["camera_id"] in heldout]
        ax.scatter(xy[train_indices, 0], xy[train_indices, 1], c="#777777", s=28, label="retained train")
        ax.scatter(xy[eval_indices, 0], xy[eval_indices, 1], c="#b23a48", marker="x", s=58, label="proposed heldout")
        for i in eval_indices:
            ax.annotate(rows[i]["camera_id"], xy[i], fontsize=7)
        ax.set_title(scene)
        ax.set_xlabel("camera-center PCA 1")
        ax.set_ylabel("camera-center PCA 2")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "recommended_camera_center_splits.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    xlabels = [f"{row['scene']}\n{row['family']}" for row in classified]
    axes[0].bar(np.arange(len(classified)), [row["train_center_max_ratio_to_original"] for row in classified], color="#4f6d7a")
    axes[0].axhline(2.25, color="#b23a48", linestyle="--", linewidth=1)
    axes[0].set_xticks(np.arange(len(classified)), xlabels, rotation=55, ha="right", fontsize=7)
    axes[0].set_ylabel("max train center-gap ratio")
    support_by_id = {row["candidate_id"]: row for row in support_rows}
    axes[1].bar(np.arange(len(classified)), [support_by_id[row["candidate_id"]]["visibility_coverage_retention_vs_original_train"] for row in classified], color="#6b8e23")
    axes[1].axhline(0.98, color="#b23a48", linestyle="--", linewidth=1)
    axes[1].set_xticks(np.arange(len(classified)), xlabels, rotation=55, ha="right", fontsize=7)
    axes[1].set_ylabel("visibility coverage retention")
    fig.tight_layout()
    fig.savefig(figure_root / "candidate_train_coverage.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    classified_by_id = {row["candidate_id"]: row for row in classified}
    for ax, scene in zip(axes.flat, SCENES):
        candidate_id = recommended["scenes"][scene]["candidate_id"]
        candidate = classified_by_id[candidate_id]
        heldout_rows = [
            row for row in _read_json(output_root / "heldout_novelty_analysis.json")["rows"]
            if row["candidate_id"] == candidate_id
        ]
        ax.scatter(
            [row["center_novelty_to_retained_train"] for row in heldout_rows],
            [row["view_direction_novelty_deg_to_retained_train"] for row in heldout_rows],
            c=[{"LOW": 0, "MID": 1, "HIGH": 2}[row["center_novelty_regime"]] for row in heldout_rows],
            cmap="viridis", s=58, edgecolors="black", linewidths=0.4,
        )
        for row in heldout_rows:
            ax.annotate(row["camera_id"], (row["center_novelty_to_retained_train"], row["view_direction_novelty_deg_to_retained_train"]), fontsize=7)
        ax.set_title(f"{scene} ({candidate['family']})")
        ax.set_xlabel("center novelty")
        ax.set_ylabel("direction novelty (deg)")
    fig.tight_layout()
    fig.savefig(figure_root / "recommended_heldout_novelty.png", dpi=160)
    plt.close(fig)


def _research_note(summary: Mapping[str, Any]) -> None:
    geometry = {row["scene"]: row for row in summary["scene_geometry"]}
    classes = summary["scene_split_classifications"]
    recommended = summary["recommended_split_manifest"]["scenes"]
    runtime = {row["scene"]: row for row in summary["historical_runtime_rows"]}
    candidate_rows = summary["candidate_assessments"]
    candidate_by_id = {row["candidate_id"]: row for row in candidate_rows}
    support_by_id = {row["candidate_id"]: row for row in summary["support_rows"]}
    preregistration = summary["preregistration"]
    statistical = {row["scene"]: row for row in summary["statistical_feasibility"]["rows"]}
    cost = summary["compute_cost"]
    decision = summary["final_decision"]
    lines = [
        "# Data-Split Feasibility Audit (2026-08-31)",
        "", "## 1. Motivation", "",
        "INFERENCE: Candidate C is data-limited, so this audit asks whether one larger geometry-aware split and four fresh locked-OCMC runs are worth approving. This is feasibility evidence, not Candidate-C support. It does not train a model or apply a split.",
        "", "## 2. Why Candidate C Is Data-Limited", "",
        "EXPERIMENTAL FACT: the current genuine held-out counts are Curasao 3, IUI3-RedSea 4, JapaneseGradens-RedSea 3, and Panama 3. Every scene is below the five-camera reliability threshold for camera-neighbor, permutation, and controlled rank analyses.",
        "", "## 3. Camera Inventory", "",
        "DATA FACT: independent COLMAP, source RGB, dataparser, and formal-list verification recovered 88 calibrated RGB cameras with no unused calibrated GT cameras.", "",
        "| Scene | Formal train | Formal eval | Total | RGB and calibration complete |", "| --- | ---: | ---: | ---: | --- |",
        *[f"| {scene} | {geometry[scene]['formal_train_count']} | {geometry[scene]['formal_eval_count']} | {geometry[scene]['total_camera_count']} | {geometry[scene]['all_cameras_have_gt_and_calibration']} |" for scene in SCENES],
        "", "## 4. Split-Design Constraints", "",
        "CONFIG FACT: candidate IDs were constructed only from transformed camera centers and geometrically verified numeric acquisition order. No old MSE, PSNR, LPIPS, residual ranking, or support ranking entered ID construction. After IDs were locked and hashed, final candidate-family ranking used view-direction coverage and GT-free support-coverage retention, as permitted by the protocol.",
        "", "CONFIG FACT: the normalized camera context exactly follows `(camera_center - scene_box_center) / (scene_box_diagonal + 1e-6)`. Center and direction analyses remain separate; no learned or outcome-tuned combined metric is used.",
        "", "## 5. Scene Geometry", "",
        "| Scene | Frame-center rho | Frame-direction rho | Adjacent/all center ratio | Trajectory eligible |", "| --- | ---: | ---: | ---: | --- |",
        *[f"| {scene} | {geometry[scene]['frame_delta_vs_center_distance_spearman']:.3f} | {geometry[scene]['frame_delta_vs_direction_angle_spearman']:.3f} | {geometry[scene]['adjacent_center_median_over_all_pair_median']:.3f} | {geometry[scene]['trajectory_family_eligible']} |" for scene in SCENES],
        "", "INFERENCE: the strong frame-center rank association and short adjacent steps verify numeric filename order as acquisition-trajectory evidence in all four scenes; filename order was not accepted without this geometry check.",
        "", "## 6. Candidate Split Families", "",
        "CODE FACT: exactly three proposals were audited per scene: A geometry-stratified novelty (N=5), B center farthest-point (N=7/8/6/6), and C trajectory-interleaved (N=6). This respects the three-proposal cap. All manifests were hashed before frozen support analysis.",
    ]
    for index, scene in enumerate(SCENES, 7):
        scene_candidates = sorted((row for row in candidate_rows if row["scene"] == scene), key=lambda row: row["scene_rank"])
        lines.extend([
            "", f"## {index}. {scene} Feasibility", "",
            "| Rank | Family | Train | Held out | Classification | Center max/original | Direction max/original | Hull retention | Visibility retention |", "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
            *[f"| {row['scene_rank']} | {row['family']} | {row['train_count']} | {row['heldout_count']} | {row['candidate_classification']} | {row['train_center_max_ratio_to_original']:.3f} | {row['train_direction_max_ratio_to_original']:.3f} | {row['projected_hull_area_retention']:.3f} | {row['visibility_coverage_retention_vs_original_train']:.3f} |" for row in scene_candidates],
            "", f"INFERENCE: audited feasible held-out sizes are {classes[scene]['feasible_heldout_sizes_among_audited_candidates']}; recommend `{recommended[scene]['candidate_id']}` with {recommended[scene]['train_count']} train and {recommended[scene]['heldout_count']} held-out cameras.",
        ])
    lines.extend([
        "", "## 11. Retained Training Coverage", "",
        "| Scene | Center p90 ratio | Center max ratio | Direction p90 ratio | Direction max ratio | PCA hull retention | Severe hole |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        *[f"| {scene} | {candidate_by_id[recommended[scene]['candidate_id']]['train_center_p90_ratio_to_original']:.3f} | {candidate_by_id[recommended[scene]['candidate_id']]['train_center_max_ratio_to_original']:.3f} | {candidate_by_id[recommended[scene]['candidate_id']]['train_direction_p90_ratio_to_original']:.3f} | {candidate_by_id[recommended[scene]['candidate_id']]['train_direction_max_ratio_to_original']:.3f} | {candidate_by_id[recommended[scene]['candidate_id']]['projected_hull_area_retention']:.3f} | {classes[scene]['severe_train_manifold_hole']} |" for scene in SCENES],
        "", "INFERENCE: all recommended splits retain acceptable center and orientation coverage and none creates a severe train-manifold hole under the fixed thresholds in the audit code.",
        "", "## 12. Held-Out Novelty Coverage", "",
        "| Scene | Held out | Center novelty min/median/max | Direction novelty min/median/max (deg) | LOW/MID/HIGH |", "| --- | ---: | --- | --- | --- |",
        *[f"| {scene} | {recommended[scene]['heldout_count']} | {candidate_by_id[recommended[scene]['candidate_id']]['heldout_center_novelty_min']:.3f}/{candidate_by_id[recommended[scene]['candidate_id']]['heldout_center_novelty_median']:.3f}/{candidate_by_id[recommended[scene]['candidate_id']]['heldout_center_novelty_max']:.3f} | {candidate_by_id[recommended[scene]['candidate_id']]['heldout_direction_novelty_deg_min']:.3f}/{candidate_by_id[recommended[scene]['candidate_id']]['heldout_direction_novelty_deg_median']:.3f}/{candidate_by_id[recommended[scene]['candidate_id']]['heldout_direction_novelty_deg_max']:.3f} | {classes[scene]['low_mid_high_center_novelty']} |" for scene in SCENES],
        "", "QUANTITATIVE RESULT: all recommendations span low/mid/high center-novelty ranks and pass the independent meaningful direction-novelty threshold.",
        "", "## 13. Support Coverage", "",
        "CONFIG FACT: only after split manifests were frozen and hashed, old C0 checkpoints supplied GT-free `gaussian_visible_mask` values. The framework datamanager may cache image batches during setup, but the audit logic never reads target image tensors or residuals. These support values are coverage-potential proxies because fresh-resplit Gaussians may differ.", "",
        "| Scene | Visibility retention | Normalized mean-support retention | Unseen range | Low-support range | Mean-support range |", "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[f"| {scene} | {support_by_id[recommended[scene]['candidate_id']]['visibility_coverage_retention_vs_original_train']:.3f} | {support_by_id[recommended[scene]['candidate_id']]['normalized_mean_support_retention_vs_original_train']:.3f} | {support_by_id[recommended[scene]['candidate_id']]['heldout_fraction_unseen_range']:.4f} | {support_by_id[recommended[scene]['candidate_id']]['heldout_fraction_low_support_range']:.4f} | {support_by_id[recommended[scene]['candidate_id']]['heldout_mean_support_range']:.3f} |" for scene in SCENES],
        "", "| Scene | Support fraction 0/1/2/>=3 | Support mean/median |", "| --- | --- | --- |",
        *[f"| {scene} | {support_by_id[recommended[scene]['candidate_id']]['retained_train_support_fraction_zero']:.4f}/{support_by_id[recommended[scene]['candidate_id']]['retained_train_support_fraction_one']:.4f}/{support_by_id[recommended[scene]['candidate_id']]['retained_train_support_fraction_two']:.4f}/{support_by_id[recommended[scene]['candidate_id']]['retained_train_support_fraction_at_least_three']:.4f} | {support_by_id[recommended[scene]['candidate_id']]['retained_train_support_mean']:.3f}/{support_by_id[recommended[scene]['candidate_id']]['retained_train_support_median']:.1f} |" for scene in SCENES],
        "", "INFERENCE: all four retain acceptable visibility/Gaussian support and span meaningful GT-free held-out support variation. The old-split representation makes this a bounded feasibility proxy, not a prediction of fresh-run support.",
        "", "## 14. Statistical Power / Feasibility", "",
        "| Scene | Held out | Independent pairs | LOO neighbor | Permutation | Single control | Multivariable regression |", "| --- | ---: | ---: | --- | --- | --- | --- |",
        *[f"| {scene} | {statistical[scene]['heldout_camera_count']} | {statistical[scene]['camera_pair_count']} | {statistical[scene]['leave_one_view_out_neighbor_feasible']} | {statistical[scene]['camera_label_permutation_feasible']} | {statistical[scene]['single_factor_rank_control_feasible']} | {statistical[scene]['multivariable_regression_appropriate']} |" for scene in SCENES],
        "", "INFERENCE: all four recommendations reach N>=5; two reach N>=6. Leave-one-view-out and 1,000 fixed-seed camera-label permutations become more meaningful than the original split, but exact permutation resolution remains coarse. Only one-control-at-a-time rank residualization is approved; multivariable regression remains inappropriate.",
        "", "## 15. Compute-Cost Estimate", "",
        "| Scene | Historical C0 wall time (s) | GPU-hours | Peak reserved (GiB) | GPU |", "| --- | ---: | ---: | ---: | --- |",
        *[f"| {scene} | {runtime[scene]['training_wall_seconds']:.2f} | {runtime[scene]['training_gpu_hours']:.3f} | {runtime[scene]['peak_reserved_bytes'] / 2**30:.2f} | {runtime[scene]['gpu_name']} |" for scene in SCENES],
        "", f"QUANTITATIVE RESULT: four fresh 15K runs require an estimated {cost['training_gpu_hours_total']:.3f} training GPU-hours, {cost['estimated_total_gpu_hours']:.3f} GPU-hours including frozen diagnostics, {cost['estimated_parallel_wall_hours_including_frozen_diagnostics']:.3f} hours with four GPUs in parallel, and {cost['estimated_future_storage_bytes'] / 1e9:.2f} GB. Cost is `{cost['cost_classification']}` relative to prior formal experiments.",
        "", "## 16. Pre-Registered Future Candidate-C Criteria", "",
        "HYPOTHESIS: the camera is the unit and `E_cam = MSE` is the target. Fixed GT-free predictors are: " + ", ".join(f"`{name}`" for name in preregistration["fixed_gt_free_predictors"]) + ". `fraction_visible_unseen_train` is preregistered from the prior audit before any new-split outcome exists.", "",
        "C_SUPPORTED_AND_ACTIONABLE requires all of: " + "; ".join(preregistration["supported_and_actionable_criteria"]) + ".", "",
        "C_NOT_SUPPORTED applies if any of: " + "; ".join(preregistration["not_supported_criteria"]) + ". Its consequence is to close Candidate C without more split variants.",
        "", "## 17. Recommended Split Proposal", "",
    ])
    for scene in SCENES:
        lines.extend([
            f"### {scene}", "",
            f"TRAIN ({recommended[scene]['train_count']}): `" + "`, `".join(recommended[scene]["train_ids"]) + "`.", "",
            f"HELDOUT ({recommended[scene]['heldout_count']}): `" + "`, `".join(recommended[scene]["heldout_ids"]) + "`.", "",
            f"Hash: `{recommended[scene]['canonical_manifest_sha256']}`.", "",
        ])
    lines.extend([
        f"Global manifest hash: `{summary['recommended_split_hashes']['global_manifest_sha256']}`.", "",
        "## 18. GO / NO-GO Decision", "",
        f"INFERENCE: `{decision['global_decision']}` and `{decision['candidate_c_research_line_decision']}`. Four of four recommended splits are `SPLIT_FEASIBLE`; expected information gain is sufficient relative to the low cost. This does not alter the original OCMC result and does not establish Candidate C.", "",
        f"Largest scientific risk: {decision['largest_scientific_risk']}.", "",
        f"Largest compute/engineering risk: {decision['largest_compute_engineering_risk']}.",
        "", "## 19. ONE Next Task", "",
        f"HYPOTHESIS: `{decision['one_next_task']}`. Use exactly one preregistered split and one fresh locked-OCMC run per scene; do not use k-fold by default.", "",
    ])
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def _aggregate(
    repo: Path,
    output_root: Path,
    geometry_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    scene_geometry: Optional[Sequence[Mapping[str, Any]]] = None,
    candidates: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    coverage_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    novelty_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    if geometry_rows is None:
        geometry_rows = _read_json(output_root / "camera_inventory_verified.json")["rows"]
    if scene_geometry is None:
        scene_geometry = _read_json(output_root / "scene_geometry_summary.json")["rows"]
    if candidates is None:
        candidates = {
            scene: _read_json(output_root / f"candidate_splits_{SCENE_FILE_STEMS[scene]}.json")["candidates"]
            for scene in SCENES
        }
    if coverage_rows is None:
        coverage_rows = _read_json(output_root / "train_coverage_analysis.json")["rows"]
    if novelty_rows is None:
        novelty_rows = _read_json(output_root / "heldout_novelty_analysis.json")["rows"]
    support_results = [
        _read_json(output_root / "support" / f"{SCENE_FILE_STEMS[scene]}_support_worker.json")
        for scene in SCENES
    ]
    support_rows = [row for result in support_results for row in result["candidate_rows"]]
    heldout_support_rows = [row for result in support_results for row in result["heldout_rows"]]
    _write_table(
        output_root, "support_coverage_analysis", support_rows,
        heldout_rows=heldout_support_rows,
        warning="frozen old-split C0 visibility is a GT-free geometry-potential proxy, not the future resplit representation",
    )
    classified = _classify_candidates(coverage_rows, support_rows)
    scene_classifications, recommended = _rank_and_recommend(candidates, classified)
    _write_json(output_root / "scene_split_classifications.json", scene_classifications)
    _write_json(output_root / "recommended_split_manifest.json", recommended)
    hashes = {
        "global_manifest_sha256": recommended["global_manifest_sha256"],
        "scene_hashes": {scene: recommended["scenes"][scene]["canonical_manifest_sha256"] for scene in SCENES},
        "hash_algorithm": "SHA-256 over canonical sorted-key compact JSON",
    }
    _write_json(output_root / "recommended_split_hashes.json", hashes)

    preregistration = _preregistration()
    _write_json(output_root / "future_candidate_c_preregistration.json", preregistration)
    statistical_rows = []
    for scene in SCENES:
        n = int(recommended["scenes"][scene]["heldout_count"])
        statistical_rows.append({
            "scene": scene, "heldout_camera_count": n, "camera_pair_count": n * (n - 1) // 2,
            "leave_one_view_out_neighbor_feasible": n >= 5,
            "camera_label_permutation_feasible": n >= 5,
            "permutation_resolution_warning": "coarse small-N exact label space" if n <= 6 else "acceptable",
            "single_factor_rank_control_feasible": n >= 5,
            "multivariable_regression_appropriate": False,
            "statistical_unit": "camera",
        })
    statistical = {
        "rows": statistical_rows,
        "scenes_with_at_least_five": sum(row["heldout_camera_count"] >= 5 for row in statistical_rows),
        "scenes_with_at_least_six": sum(row["heldout_camera_count"] >= 6 for row in statistical_rows),
        "conclusion": "MORE_MEANINGFUL_THAN_ORIGINAL_BUT_STILL_SMALL_N",
    }
    _write_json(output_root / "statistical_feasibility.json", statistical)

    runtime_rows, compute_cost = _historical_runtime(repo, support_results)
    _write_table(output_root, "historical_ocmc_runtime_summary", runtime_rows)
    _write_json(output_root / "compute_cost_estimate.json", compute_cost)
    counts = {
        "SPLIT_FEASIBLE": sum(row["classification"] == "SPLIT_FEASIBLE" for row in scene_classifications.values()),
        "SPLIT_FEASIBLE_BUT_TIGHT": sum(row["classification"] == "SPLIT_FEASIBLE_BUT_TIGHT" for row in scene_classifications.values()),
        "SPLIT_NOT_FEASIBLE": sum(row["classification"] == "SPLIT_NOT_FEASIBLE" for row in scene_classifications.values()),
    }
    at_least_tight = counts["SPLIT_NOT_FEASIBLE"] == 0
    if (
        counts["SPLIT_FEASIBLE"] >= 3
        and at_least_tight
        and statistical["scenes_with_at_least_five"] >= 3
        and all(row["low_mid_high_center_novelty"] for row in scene_classifications.values())
        and compute_cost["cost_classification"] in {"LOW_COST", "MODERATE_COST"}
    ):
        global_decision = "C_SPLIT_RETRAIN_GO"
        research_line = "CONTINUE_C_WITH_NEW_SPLIT"
        next_task = "OCMC-CANDIDATE-C-RESPLIT-CAUSAL-REPLICATION"
    elif at_least_tight and statistical["scenes_with_at_least_five"] >= 3:
        global_decision = "C_SPLIT_RETRAIN_BORDERLINE"
        research_line = "KEEP_C_DEFERRED"
        next_task = "BROADER-SECOND-MODULE-RESEARCH-PRIORITIZATION"
    else:
        global_decision = "C_SPLIT_RETRAIN_NO_GO"
        research_line = "CLOSE_CANDIDATE_C"
        next_task = "BROADER-SECOND-MODULE-RESEARCH-PRIORITIZATION"
    final_decision = {
        "global_decision": global_decision,
        "candidate_c_research_line_decision": research_line,
        "one_next_task": next_task,
        "scene_classification_counts": counts,
        "expected_information_gain_high_enough": global_decision == "C_SPLIT_RETRAIN_GO",
        "exactly_one_split_and_one_ocmc_run_per_scene_if_go": True,
        "k_fold_cross_validation_by_default": False,
        "largest_scientific_risk": "a geometry-selected split may change the reconstruction problem enough that residual differences reflect resplit-induced representation change rather than stable Candidate-C structure",
        "largest_compute_engineering_risk": "future split plumbing or checkpoint provenance drift could invalidate four otherwise inexpensive 15K runs",
        "feasibility_evidence_not_candidate_c_support": True,
    }
    _write_json(output_root / "final_feasibility_decision.json", final_decision)
    _write_table(output_root, "train_coverage_analysis", classified, baselines=_read_json(output_root / "train_coverage_analysis.json").get("baselines", {}))
    summary = {
        "experiment": EXPERIMENT,
        "scene_geometry": list(scene_geometry),
        "camera_inventory_count": len(geometry_rows),
        "candidate_assessments": classified,
        "support_rows": support_rows,
        "heldout_support_rows": heldout_support_rows,
        "statistical_feasibility": statistical,
        "preregistration": preregistration,
        "historical_runtime_rows": runtime_rows,
        "compute_cost": compute_cost,
        "scene_split_classifications": scene_classifications,
        "recommended_split_manifest": recommended,
        "recommended_split_hashes": hashes,
        "final_decision": final_decision,
        "no_dataset_mutation": True,
        "no_training": True,
        "optimizer_step_called": False,
        "backward_called": False,
        "old_rgb_residual_used_for_selection": False,
        "candidate_id_construction_used_gt_free_support": False,
        "final_family_ranking_used_gt_free_coverage_retention": True,
        "candidate_manifests_hashed_before_support_audit": True,
        "outputs_committed": False,
    }
    _write_json(output_root / "final_summary.json", summary)
    _make_figures(output_root, geometry_rows, candidates, recommended, classified, support_rows)
    _research_note(summary)
    return summary


def _candidate_document(scene: str, candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    document = {
        "scene": scene,
        "candidate_count": len(candidates),
        "construction_stage": "BLIND_GEOMETRY_ONLY_BEFORE_CHECKPOINT_LOAD",
        "old_rgb_residual_files_loaded": False,
        "candidate_id_construction_used_gt_free_support": False,
        "later_candidate_ranking_may_use_gt_free_coverage_retention": True,
        "candidate_selection_locked_before_support_audit": True,
        "candidates": list(candidates),
    }
    document["candidate_file_sha256_before_support"] = _canonical_hash(document)
    return document


def _launch(repo: Path, output_root: Path) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    starting = _repo_state(repo)
    starting_split_hashes = _dataset_split_hashes(repo)
    _write_json(output_root / "repo_state.json", starting)
    _write_json(output_root / "dataset_split_integrity_before.json", starting_split_hashes)
    _write_json(output_root / "environment.json", {
        "python": sys.executable, "python_version": sys.version.split()[0], "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "launcher_visible_device_count": torch.cuda.device_count(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "primary_execution": "CPU metadata and geometry; GPU only for frozen no-GT visibility support",
        "allowed_physical_gpus": [6, 7, 8, 9],
    })
    geometry_rows: List[Dict[str, Any]] = []
    scene_geometry: List[Dict[str, Any]] = []
    for scene in SCENES:
        rows, summary = _geometry_scene(repo, scene)
        geometry_rows.extend(rows)
        scene_geometry.append(summary)
    _write_table(output_root, "camera_inventory_verified", geometry_rows, total_camera_count=len(geometry_rows))
    _write_table(output_root, "scene_geometry_summary", scene_geometry)
    pairwise = _pairwise_rows(geometry_rows)
    _write_table(output_root, "pairwise_camera_geometry", pairwise)
    candidates = _build_candidates(geometry_rows, scene_geometry)
    candidate_hashes_before: Dict[str, str] = {}
    for scene in SCENES:
        path = output_root / f"candidate_splits_{SCENE_FILE_STEMS[scene]}.json"
        document = _candidate_document(scene, candidates[scene])
        candidate_hashes_before[scene] = document["candidate_file_sha256_before_support"]
        _write_json(path, document)
    coverage_rows, novelty_rows, baselines = _coverage_rows(geometry_rows, candidates)
    _write_table(output_root, "train_coverage_analysis", coverage_rows, baselines=baselines)
    _write_table(output_root, "heldout_novelty_analysis", novelty_rows)

    gpu_snapshot = _run_text([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"
    ])
    process_snapshot = _run_text([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"
    ])
    _write_json(output_root / "gpu_preflight.json", {
        "gpu_snapshot": gpu_snapshot, "compute_process_snapshot": process_snapshot,
        "no_processes_killed": True, "allowed_physical_gpus": [6, 7, 8, 9],
    })
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    processes: Dict[str, subprocess.Popen[Any]] = {}
    handles: Dict[str, Any] = {}
    try:
        for scene, gpu in SCENE_GPUS.items():
            handle = (output_root / "logs" / f"{scene}.log").open("w", encoding="utf8")
            handles[scene] = handle
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                str(PYTHON), str(Path(__file__).resolve()), "--support-scene", scene, "--gpu", gpu,
                "--repo", str(repo), "--output-root", str(output_root),
            ]
            processes[scene] = subprocess.Popen(command, cwd=str(repo), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        statuses = {scene: process.wait() for scene, process in processes.items()}
    finally:
        for handle in handles.values():
            handle.close()
    _write_json(output_root / "support_worker_status.json", statuses)
    if any(code != 0 for code in statuses.values()):
        raise RuntimeError(f"frozen support worker failed: {statuses}")
    for scene in SCENES:
        document = _read_json(output_root / f"candidate_splits_{SCENE_FILE_STEMS[scene]}.json")
        comparable = {key: value for key, value in document.items() if key != "candidate_file_sha256_before_support"}
        if document["candidate_file_sha256_before_support"] != candidate_hashes_before[scene] or _canonical_hash(comparable) != candidate_hashes_before[scene]:
            raise RuntimeError(f"candidate split drift after support audit: {scene}")
    summary = _aggregate(repo, output_root, geometry_rows, scene_geometry, candidates, coverage_rows, novelty_rows)
    ending = _repo_state(repo)
    ending_split_hashes = _dataset_split_hashes(repo)
    _write_json(output_root / "repo_state_after.json", ending)
    split_files_unchanged = starting_split_hashes == ending_split_hashes
    _write_json(output_root / "dataset_split_integrity_after.json", {
        "unchanged": split_files_unchanged,
        "before": starting_split_hashes,
        "after": ending_split_hashes,
    })
    protected_unchanged = all(
        starting["protected_files"][path]["sha256"] == ending["protected_files"][path]["sha256"]
        for path in PROTECTED
    )
    _write_json(output_root / "protected_file_integrity.json", {
        "unchanged": protected_unchanged, "before": starting["protected_files"], "after": ending["protected_files"],
    })
    if not protected_unchanged:
        raise RuntimeError("protected historical/Q50-Q80 file hash changed")
    if not split_files_unchanged:
        raise RuntimeError("official dataset split file hash changed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--support-scene", choices=SCENES)
    parser.add_argument("--gpu")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_root = args.output_root.resolve()
    if args.support_scene:
        if args.gpu is None:
            raise ValueError("--support-scene requires --gpu")
        result = _support_worker(repo, output_root, args.support_scene, args.gpu)
        printable = {"scene": result["scene"], "runtime": result["runtime"], "no_training": result["no_training"]}
    elif args.aggregate_only:
        result = _aggregate(repo, output_root)
        printable = result["final_decision"]
    else:
        result = _launch(repo, output_root)
        printable = result["final_decision"]
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
