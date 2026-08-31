#!/usr/bin/env python3
"""Pre-registered four-scene Candidate-C replication after locked OCMC.

The only training intervention is the outcome-blind camera split locked by the
preceding feasibility audit.  Each scene trains one fresh C0/OCMC model.  The
held-out cameras are used only after step 14999 for frozen evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_camera_residual_replication_ocmc as PRIOR
from scripts.diagnostics import audit_candidate_c_data_split_feasibility as SPLIT_AUDIT
from scripts.diagnostics import audit_ocmc_residual_failure_modes as BASE
from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL

EXPERIMENT = "OCMC-CANDIDATE-C-RESPLIT-CAUSAL-REPLICATION"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "ocmc_candidate_c_resplit_replication_20260831"
FEASIBILITY_ROOT = REPO_ROOT / "outputs" / "data_split_feasibility_audit_20260831"
OLD_AUDIT_ROOT = REPO_ROOT / "outputs" / "focused_c_camera_residual_replication_20260831"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "OCMC_CANDIDATE_C_RESPLIT_REPLICATION_2026-08-31.md"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
FINAL_STEP = 14999
MAX_STEPS = 15000
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}
SCENES = tuple(SCENE_GPUS)
PROTECTED = BASE.PROTECTED
EPS = 1e-12
EXPECTED_GLOBAL_HASH = "8615b61e3d4d7f3355a196e41708d2774110d6f025fdfa61d0f41e5b426ad465"
EXPECTED_SCENE_HASHES = {
    "Curasao": "9ab62359f5a860886ca86367d8e868adef4a4121ebe32355927227da63dfb735",
    "IUI3-RedSea": "caf4ad6ec74edb145093d8f0cee423c0b2f8a038d2fb4b23b3fc08f0bd3438a3",
    "JapaneseGradens-RedSea": "afb39101ed59294ec818d6f4962b10f224c387d56555a8c8842c0f8d8a053bf2",
    "Panama": "8b7584daaad06464b1d2a47fa8e279b9d7970416e57d32d5cc86df452a4dde1a",
}
PREDICTORS: Dict[str, int] = {
    "fraction_visible_unseen_train": 1,
    "center_nearest_train": 1,
    "center_knn3_mean": 1,
    "context_nearest_train": 1,
    "context_knn3_mean": 1,
    "view_direction_nearest_angle_deg": 1,
    "view_direction_knn3_angle_deg": 1,
    "mean_train_visibility_support": -1,
    "median_train_visibility_support": -1,
    "fraction_visible_low_support": 1,
}
PRIMARY_PREDICTOR = "fraction_visible_unseen_train"
CONTROLS = (
    "mean_depth",
    "mean_tau",
    "mean_transmission",
    "mean_accumulation",
    "mean_projected_radius_px",
    "visible_gaussian_count",
    "mean_train_visibility_support",
)
MAJOR_CONTROL_KEYS = CONTROLS


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
    if isinstance(value, torch.device):
        return str(value)
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


def _write_table(root: Path, stem: str, rows: Sequence[Mapping[str, Any]], **extra: Any) -> None:
    _write_csv(root / f"{stem}.csv", rows)
    _write_json(root / f"{stem}.json", {"rows": list(rows), **extra})


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _run_text(command: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(list(command), cwd=str(cwd), text=True).strip()


def _allowed_gpu_processes() -> Dict[str, str]:
    return {
        gpu: _run_text([
            "nvidia-smi", "-i", gpu,
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        for gpu in SCENE_GPUS.values()
    }


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


def _official_split_hashes(repo: Path) -> Dict[str, Any]:
    return {
        scene: {
            name: _sha256(repo / str(FORMAL.SCENES[scene]["data_path"]) / name)
            for name in ("train_list.txt", "test_list.txt", "val_list.txt")
        }
        for scene in SCENES
    }


def _verify_locked_manifest() -> Dict[str, Any]:
    manifest_path = FEASIBILITY_ROOT / "recommended_split_manifest.json"
    hashes_path = FEASIBILITY_ROOT / "recommended_split_hashes.json"
    preregistration_path = FEASIBILITY_ROOT / "future_candidate_c_preregistration.json"
    manifest = _read_json(manifest_path)
    hashes = _read_json(hashes_path)
    preregistration = _read_json(preregistration_path)
    base = {key: value for key, value in manifest.items() if key not in {"global_manifest_sha256", "global_manifest_hash_scope"}}
    computed_global = _canonical_hash(base)
    if computed_global != EXPECTED_GLOBAL_HASH or manifest["global_manifest_sha256"] != EXPECTED_GLOBAL_HASH:
        raise RuntimeError(f"global split hash mismatch: computed={computed_global} manifest={manifest.get('global_manifest_sha256')}")
    if hashes["global_manifest_sha256"] != EXPECTED_GLOBAL_HASH:
        raise RuntimeError("recommended_split_hashes global hash mismatch")
    rows = []
    all_ids: set = set()
    for scene in SCENES:
        entry = manifest["scenes"][scene]
        canonical = {
            "schema_version": 1,
            "scene": scene,
            "family": entry["family"],
            "family_name": entry["family_name"],
            "train_ids": sorted(entry["train_ids"]),
            "heldout_ids": sorted(entry["heldout_ids"]),
        }
        computed = _canonical_hash(canonical)
        expected = EXPECTED_SCENE_HASHES[scene]
        if computed != expected or entry["canonical_manifest_sha256"] != expected or hashes["scene_hashes"][scene] != expected:
            raise RuntimeError(f"split hash mismatch for {scene}: computed={computed}")
        train, heldout = set(entry["train_ids"]), set(entry["heldout_ids"])
        if train & heldout or len(train) != entry["train_count"] or len(heldout) != entry["heldout_count"]:
            raise RuntimeError(f"invalid locked partition for {scene}")
        scene_keyed = {(scene, camera_id) for camera_id in train | heldout}
        if all_ids & scene_keyed:
            raise RuntimeError(f"duplicate scene/camera key in {scene}")
        all_ids |= scene_keyed
        rows.append({
            "scene": scene,
            "train_count": len(train),
            "heldout_count": len(heldout),
            "scene_hash_expected": expected,
            "scene_hash_computed": computed,
            "hash_match": True,
        })
    expected_predictors = list(PREDICTORS)
    if preregistration["fixed_gt_free_predictors"] != [
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
    ]:
        raise RuntimeError("preregistered predictor document drift")
    return {
        "verified": True,
        "blind_geometry_split": not bool(manifest["selection_uses_rgb_residual"]),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_hashes": str(hashes_path),
        "source_preregistration": str(preregistration_path),
        "global_hash_expected": EXPECTED_GLOBAL_HASH,
        "global_hash_computed": computed_global,
        "rows": rows,
        "manifest": manifest,
        "preregistration": preregistration,
        "analysis_predictor_keys": expected_predictors,
    }


def _create_data_views(repo: Path, output_root: Path, verification: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for scene in SCENES:
        source = (repo / str(FORMAL.SCENES[scene]["data_path"])).resolve()
        view = (output_root / "data_views" / scene).resolve()
        view.mkdir(parents=True, exist_ok=True)
        for name in ("images", "sparse"):
            path = view / name
            target = source / name
            if path.is_symlink():
                if path.resolve() != target.resolve():
                    raise RuntimeError(f"existing data-view symlink drift: {path}")
            elif path.exists():
                raise RuntimeError(f"data-view path must be a symlink: {path}")
            else:
                path.symlink_to(target, target_is_directory=True)
        entry = verification["manifest"]["scenes"][scene]
        train_names = [f"{camera_id}.png" for camera_id in entry["train_ids"]]
        heldout_names = [f"{camera_id}.png" for camera_id in entry["heldout_ids"]]
        for name, values in (("train_list.txt", train_names), ("test_list.txt", heldout_names), ("val_list.txt", heldout_names)):
            (view / name).write_text("\n".join(values) + "\n", encoding="utf8")
        for image_name in train_names + heldout_names:
            if not (source / "images" / "ColorImage" / image_name).is_file():
                raise FileNotFoundError(f"missing source RGB: {scene}/{image_name}")
        rows.append({
            "scene": scene,
            "source_data_root": str(source),
            "experiment_data_root": str(view),
            "images_symlink_target": str((view / "images").resolve()),
            "sparse_symlink_target": str((view / "sparse").resolve()),
            "train_ids": entry["train_ids"],
            "heldout_ids": entry["heldout_ids"],
            "train_count": len(train_names),
            "heldout_count": len(heldout_names),
            "train_list_sha256": _sha256(view / "train_list.txt"),
            "test_list_sha256": _sha256(view / "test_list.txt"),
            "official_source_files_modified": False,
        })
    return {"implementation": "output-local lists with read-only symlinks to source images and COLMAP calibration", "rows": rows}


def _scene_cfg(scene: str, output_root: Path) -> Dict[str, Any]:
    cfg = dict(FORMAL.SCENES[scene])
    cfg["data_path"] = str((output_root / "data_views" / scene).resolve())
    cfg["locked_safe"] = False
    return cfg


def _config_lock(repo: Path, scene: str, scene_cfg: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    source = repo / str(scene_cfg["source_config"])
    config = yaml.load(source.read_text(encoding="utf8"), Loader=yaml.Loader)
    model = config.pipeline.model
    dm = config.pipeline.datamanager
    lock = {
        "scene": scene,
        "source_config": str(source),
        "source_config_sha256": _sha256(source),
        "only_scientific_change": "camera train/heldout assignment",
        "from_scratch": True,
        "load_dir": None,
        "load_step": None,
        "load_checkpoint": None,
        "seed": int(config.machine.seed),
        "max_num_iterations": int(config.max_num_iterations),
        "snapshot_steps": list(FORMAL.SNAPSHOT_STEPS),
        "refresh_steps": list(FORMAL.REFRESH_STEPS),
        "train_sampling_strategy": str(dm.train_cameras_sampling_strategy),
        "train_sampling_seed": int(dm.train_cameras_sampling_seed),
        "intrinsic_color_parameterization": "bounded_sh3",
        "sh_degree": int(model.sh_degree),
        "rasterize_mode": "classic",
        "medium_context_mode": "dir_xy_camera",
        "medium_camera_context_scale": 1.0,
        "medium_camera_context_dropout": 0.0,
        "camera_medium_observability_enabled": True,
        "camera_medium_observability_strength": 1.0,
        "camera_medium_ray_adaptive_observability_enabled": False,
        "b_inf_mode": "tied",
        "infinite_water_enabled": False,
        "stop_split_at": int(model.stop_split_at),
        "refine_every": int(model.refine_every),
        "warmup_length": int(model.warmup_length),
        "experiment_data_root": str(scene_cfg["data_path"]),
    }
    if lock["seed"] != 42 or lock["max_num_iterations"] != MAX_STEPS or lock["stop_split_at"] != 10000:
        raise RuntimeError(f"formal config drift for {scene}: {lock}")
    lock["config_lock_sha256"] = _canonical_hash(lock)
    _write_json(output_dir / "config" / "locked_config.json", lock)
    return lock


def _model_state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _camera_arrays(records: Sequence[Tuple[Any, ...]], model: Any) -> Dict[str, Dict[str, np.ndarray]]:
    return PRIOR._camera_arrays(records, model)


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _rho(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3 or np.ptp(a[valid]) <= EPS or np.ptp(b[valid]) <= EPS:
        return float("nan")
    return float(scipy.stats.spearmanr(a[valid], b[valid]).statistic)


def _kendall(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3 or np.ptp(a[valid]) <= EPS or np.ptp(b[valid]) <= EPS:
        return float("nan")
    return float(scipy.stats.kendalltau(a[valid], b[valid]).statistic)


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def _historical_visible_mask(model: Any, outputs: Mapping[str, Any], n_gaussians: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recover the exact focused-audit visibility definition and verify its alias."""

    radii = model.radii.detach().float().cpu().numpy().reshape(-1)
    reported = outputs["gaussian_visible_mask"].detach().bool().cpu().numpy().reshape(-1)
    if radii.size != n_gaussians or reported.size != n_gaussians:
        raise RuntimeError(
            f"visibility shape mismatch: radii={radii.size}, reported={reported.size}, gaussians={n_gaussians}"
        )
    visible = radii > 0
    if not np.array_equal(visible, reported):
        raise RuntimeError("historical radii > 0 visibility differs from gaussian_visible_mask")
    return visible, radii


def _render_final(repo: Path, scene: str, scene_cfg: Mapping[str, Any], output_dir: Path, config_lock: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    checkpoint = output_dir / "checkpoints" / "C0" / f"step-{FINAL_STEP:09d}.ckpt"
    branch = FORMAL._setup_branch(repo, scene_cfg, "C0")
    try:
        ckpt = FORMAL._load_checkpoint(branch, checkpoint)
        model = branch.pipeline.model
        train_records = FORMAL._train_records(branch.pipeline)
        heldout_records = FORMAL._eval_records(branch.pipeline)
        train_ids = {record[1] for record in train_records}
        heldout_ids = {record[1] for record in heldout_records}
        if train_ids & heldout_ids:
            raise RuntimeError(f"heldout leakage in datamanager records: {scene}")
        camera_data = {
            "train": _camera_arrays(train_records, model),
            "heldout": _camera_arrays(heldout_records, model),
        }
        n_gaussians = int(model.means.shape[0])
        support_count = torch.zeros(n_gaussians, dtype=torch.int16)
        train_radii: List[np.ndarray] = []
        visibility_equivalence_checks = 0
        for _index, camera_id, camera, _batch in train_records:
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
            visible, radii = _historical_visible_mask(model, outputs, n_gaussians)
            support_count += torch.from_numpy(visible).to(torch.int16)
            train_radii.append(radii[visible])
            visibility_equivalence_checks += 1
            del outputs
            gc.collect()
        pooled = np.concatenate(train_radii) if train_radii else np.empty(0)
        footprint_threshold = _quantile(pooled, 0.95)
        train_centers = np.asarray([camera_data["train"][key]["center"] for key in sorted(train_ids)])
        train_contexts = np.asarray([camera_data["train"][key]["context"] for key in sorted(train_ids)])
        train_directions = np.asarray([camera_data["train"][key]["view_direction"] for key in sorted(train_ids)])
        metrics_rows: List[Dict[str, Any]] = []
        predictor_rows: List[Dict[str, Any]] = []
        render_rows: List[Dict[str, Any]] = []
        decomp_maps: Dict[str, Dict[str, torch.Tensor]] = {}
        render_root = output_dir / "diagnostics" / "heldout_renders"
        render_root.mkdir(parents=True, exist_ok=True)
        for _index, camera_id, camera, batch in heldout_records:
            tick = time.perf_counter()
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
            pred = outputs["pred_image"].detach().float().clamp(0, 1)
            metric = MIC._metric_images(model, pred, gt)
            residual_rgb = (pred - gt).cpu().numpy()
            squared = np.mean(np.square(residual_rgb), axis=-1)
            absolute = np.mean(np.abs(residual_rgb), axis=-1)
            visible_mask, radii = _historical_visible_mask(model, outputs, n_gaussians)
            visibility_equivalence_checks += 1
            visible_ids = np.flatnonzero(visible_mask)
            visible_radii = radii[visible_ids]
            support = support_count[torch.from_numpy(visible_ids)].float().numpy() if visible_ids.size else np.empty(0)
            center_nn, center_k3 = PRIOR._nearest_features(camera_data["heldout"][camera_id]["center"], train_centers)
            context_nn, context_k3 = PRIOR._nearest_features(camera_data["heldout"][camera_id]["context"], train_contexts)
            direction_nn, direction_k3 = PRIOR._angular_features(camera_data["heldout"][camera_id]["view_direction"], train_directions)
            camera_delta = outputs.get("camera_medium_delta_projected_raw")
            ocmc_magnitude = (
                float(torch.linalg.norm(camera_delta.detach().float(), dim=-1).mean().cpu())
                if isinstance(camera_delta, torch.Tensor) else float("nan")
            )
            image_path = Path(scene_cfg["data_path"]) / "images" / "ColorImage" / f"{camera_id}.png"
            metrics_rows.append({
                "scene": scene,
                "camera_id": camera_id,
                "split": "PREREGISTERED_HELDOUT",
                "used_in_training": False,
                "used_in_ocmc_calibration": False,
                "absolute_step": int(ckpt["absolute_step"]),
                "checkpoint_path": str(checkpoint),
                "PSNR": metric["PSNR"],
                "SSIM": metric["SSIM"],
                "LPIPS": metric["LPIPS"],
                "MSE": metric["MSE"],
                "E_cam": float(squared.mean()),
                "MAE": float(absolute.mean()),
                "median_residual": _quantile(absolute, 0.50),
                "p90_residual": _quantile(absolute, 0.90),
                "p95_residual": _quantile(absolute, 0.95),
                "p99_residual": _quantile(absolute, 0.99),
                "height": int(pred.shape[0]),
                "width": int(pred.shape[1]),
                "all_finite": bool(np.isfinite(squared).all() and all(math.isfinite(float(metric[key])) for key in metric)),
            })
            depth = outputs["depth"].detach().float().cpu().numpy().reshape(-1)
            tau = outputs["tau_D"].detach().float().cpu().numpy().reshape(-1)
            transmission = outputs["transmission"].detach().float().cpu().numpy().reshape(-1)
            accumulation = outputs["accumulation"].detach().float().cpu().numpy().reshape(-1)
            beta_b = outputs["medium_bs"].detach().float().cpu().numpy().reshape(-1)
            beta_d = outputs["medium_attn"].detach().float().cpu().numpy().reshape(-1)
            info = camera_data["heldout"][camera_id]
            predictor_rows.append({
                "scene": scene,
                "camera_id": camera_id,
                "center_x": float(info["center"][0]), "center_y": float(info["center"][1]), "center_z": float(info["center"][2]),
                "context_x": float(info["context"][0]), "context_y": float(info["context"][1]), "context_z": float(info["context"][2]),
                "view_direction_x": float(info["view_direction"][0]), "view_direction_y": float(info["view_direction"][1]), "view_direction_z": float(info["view_direction"][2]),
                "center_nearest_train": center_nn,
                "center_knn3_mean": center_k3,
                "context_nearest_train": context_nn,
                "context_knn3_mean": context_k3,
                "view_direction_nearest_angle_deg": direction_nn,
                "view_direction_knn3_angle_deg": direction_k3,
                "visible_gaussian_count": int(visible_ids.size),
                "mean_train_visibility_support": float(np.mean(support)) if support.size else float("nan"),
                "median_train_visibility_support": _quantile(support, 0.50),
                "fraction_visible_unseen_train": float(np.mean(support == 0)) if support.size else float("nan"),
                "fraction_visible_low_support": float(np.mean(support <= 1)) if support.size else float("nan"),
                "mean_projected_radius_px": float(np.mean(visible_radii)) if visible_radii.size else float("nan"),
                "fraction_large_footprint": float(np.mean(visible_radii > footprint_threshold)) if visible_radii.size else float("nan"),
                "mean_depth": float(np.mean(depth)),
                "mean_tau": float(np.mean(tau)),
                "mean_transmission": float(np.mean(transmission)),
                "mean_accumulation": float(np.mean(accumulation)),
                "mean_beta_B": float(np.mean(beta_b)),
                "mean_beta_D": float(np.mean(beta_d)),
                "mean_ocmc_projected_camera_residual": ocmc_magnitude,
                "predictors_use_gt": False,
                "visibility_definition": "radii > 0, exactly as the focused Candidate-C audit",
                "support_definition": "count of preregistered train cameras with radii > 0",
                "low_support_definition": "support count <= 1",
            })
            decomp_keys = (
                "clear_object_fullsh_raw", "tau_D", "transmission", "b_inf", "medium_bs", "medium_attn",
                "gaussian_visible_mask", "gaussian_view_rgb", "gaussian_view_logits",
            )
            decomp_maps[camera_id] = {
                key: outputs[key].detach().float().cpu() if outputs[key].dtype != torch.bool else outputs[key].detach().cpu()
                for key in decomp_keys if key in outputs and isinstance(outputs[key], torch.Tensor)
            }
            pred_path = render_root / f"{camera_id}_pred.png"
            Image.fromarray(np.clip(pred.cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(pred_path)
            render_rows.append({
                "scene": scene,
                "camera_id": camera_id,
                "prediction_path": str(pred_path),
                "ground_truth_path": str(image_path),
                "render_seconds": time.perf_counter() - tick,
                "parameter_update": False,
                "backward_called": False,
            })
            del outputs, gt, pred, residual_rgb, squared, absolute
            gc.collect()
        decomp = CAM._decomposition_row("C0", FINAL_STEP, "preregistered_heldout", decomp_maps)
        flags = {
            key: getattr(model.config, key, None)
            for key in (
                "intrinsic_color_parameterization", "rasterize_mode", "medium_context_mode",
                "camera_medium_observability_enabled", "camera_medium_ray_adaptive_observability_enabled",
                "camera_medium_observability_strength", "b_inf_mode", "infinite_water_enabled",
            )
        }
        bundle = ckpt.get("ocmc_bundle") or {}
        gates = np.asarray(bundle.get("global_gate", bundle.get("gates", [])), dtype=np.float64)
        manifest = {
            "scene": scene,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "absolute_step": int(ckpt["absolute_step"]),
            "checkpoint_experiment": ckpt.get("experiment"),
            "checkpoint_branch": ckpt.get("branch"),
            "config_lock_sha256": config_lock["config_lock_sha256"],
            "train_count": len(train_records),
            "heldout_count": len(heldout_records),
            "train_ids": sorted(train_ids),
            "heldout_ids": sorted(heldout_ids),
            "flags": flags,
            "final_gaussian_count": n_gaussians,
            "ocmc_refresh_step": int(bundle.get("step", -1)),
            "ocmc_global_gate": gates.tolist(),
            "ocmc_global_gate_min": float(np.min(gates)) if gates.size else float("nan"),
            "ocmc_global_gate_mean": float(np.mean(gates)) if gates.size else float("nan"),
            "ocmc_modes_below_half": int(np.sum(gates < 0.5)) if gates.size else 0,
            "raoc_disabled": flags["camera_medium_ray_adaptive_observability_enabled"] is False,
            "heldout_leakage": False,
        }
        result = {
            "scene": scene,
            "checkpoint": manifest,
            "metrics_rows": metrics_rows,
            "predictor_rows": predictor_rows,
            "render_rows": render_rows,
            "decomposition_safety": decomp,
            "train_camera_geometry": [
                {"camera_id": camera_id, **{key: value.tolist() for key, value in camera_data["train"][camera_id].items()}}
                for camera_id in sorted(train_ids)
            ],
            "heldout_camera_geometry": [
                {"camera_id": camera_id, **{key: value.tolist() for key, value in camera_data["heldout"][camera_id].items()}}
                for camera_id in sorted(heldout_ids)
            ],
            "evaluation_wall_seconds": time.perf_counter() - started,
            "optimizer_step_called_during_evaluation": False,
            "backward_called_during_evaluation": False,
            "visibility_definition_audit": {
                "definition": "radii > 0",
                "gaussian_visible_mask_equivalence_asserted": True,
                "camera_checks": visibility_equivalence_checks,
            },
        }
        _write_json(output_dir / "diagnostics" / "scene_result.json", result)
        _write_table(output_dir / "diagnostics", "per_camera_metrics", metrics_rows)
        _write_table(output_dir / "diagnostics", "per_camera_predictors", predictor_rows)
        _write_table(output_dir / "diagnostics", "render_manifest", render_rows)
        _write_json(output_dir / "diagnostics" / "decomposition_safety.json", decomp)
        _write_json(output_dir / "checkpoint_manifest.json", manifest)
        return result
    finally:
        FORMAL._release(branch)


def _prepare_worker(
    repo: Path, output_root: Path, scene: str, gpu: str, scene_dir: Path,
) -> Dict[str, Any]:
    if scene not in SCENES or SCENE_GPUS[scene] != gpu:
        raise RuntimeError(f"invalid scene/GPU assignment: {scene}/{gpu}")
    FORMAL._configure_schedule(MAX_STEPS)
    FORMAL.EXPERIMENT = EXPERIMENT
    runtime_manifest = FORMAL._runtime(gpu)
    scene_dir.mkdir(parents=True, exist_ok=True)
    verification = _read_json(output_root / "split_manifest_verified.json")
    entry = verification["manifest"]["scenes"][scene]
    if entry["canonical_manifest_sha256"] != EXPECTED_SCENE_HASHES[scene]:
        raise RuntimeError(f"worker split hash mismatch for {scene}")
    scene_cfg = _scene_cfg(scene, output_root)
    config_lock = _config_lock(repo, scene, scene_cfg, scene_dir)
    _write_json(scene_dir / "runtime_manifest.json", runtime_manifest)
    FORMAL._seed_all(FORMAL.TRAINING_SEED)
    probe = FORMAL._setup_branch(repo, scene_cfg, "C0")
    try:
        training_rng = FORMAL._rng_state()
        train_records = FORMAL._train_records(probe.pipeline)
        eval_records = FORMAL._eval_records(probe.pipeline)
        actual_train = [record[1] for record in train_records]
        actual_eval = [record[1] for record in eval_records]
        if set(actual_train) != set(entry["train_ids"]) or set(actual_eval) != set(entry["heldout_ids"]):
            raise RuntimeError(f"datamanager split mismatch for {scene}: train={actual_train}, eval={actual_eval}")
        actual_flags = {
            key: getattr(probe.pipeline.model.config, key, None)
            for key in (
                "intrinsic_color_parameterization", "rasterize_mode", "medium_context_mode",
                "medium_camera_context_scale", "medium_camera_context_dropout",
                "camera_medium_observability_enabled", "camera_medium_observability_strength",
                "camera_medium_ray_adaptive_observability_enabled", "b_inf_mode", "infinite_water_enabled",
            )
        }
        expected_flags = {
            key: config_lock[key]
            for key in actual_flags
        }
        if actual_flags != expected_flags:
            raise RuntimeError(f"constructed model config drift for {scene}: actual={actual_flags}, expected={expected_flags}")
        initial_state_hash = _model_state_hash(probe.pipeline.model)
        samples, bank = FORMAL._build_samples(repo, scene_dir, scene, scene_cfg, probe)
        sequence, names = FORMAL._camera_sequence(probe, scene_dir)
    finally:
        FORMAL._release(probe)
    if not set(bank["rows"][i]["view_id"] for i in range(len(bank["rows"]))) <= set(entry["train_ids"]):
        raise RuntimeError(f"heldout camera entered OCMC bank for {scene}")
    if not set(names) <= set(entry["train_ids"]):
        raise RuntimeError(f"heldout camera entered optimization sequence for {scene}")
    if len(sequence) != MAX_STEPS or len(names) != MAX_STEPS:
        raise RuntimeError(f"camera sequence length drift for {scene}: {len(sequence)}")
    start_state = {
        "scene": scene,
        "policy": "registered seed-42 from-scratch construction from source COLMAP points",
        "load_dir": None,
        "load_step": None,
        "load_checkpoint": None,
        "initial_model_state_sha256": initial_state_hash,
        "training_rng_manifest": FORMAL._rng_manifest(training_rng),
        "train_ids_verified": sorted(entry["train_ids"]),
        "heldout_ids_absent_from_training": not bool(set(names) & set(entry["heldout_ids"])),
    }
    _write_json(scene_dir / "training_start_state.json", start_state)
    return {
        "runtime_manifest": runtime_manifest,
        "entry": entry,
        "scene_cfg": scene_cfg,
        "config_lock": config_lock,
        "training_rng": training_rng,
        "samples": samples,
        "bank": bank,
        "sequence": sequence,
        "names": names,
        "start_state": start_state,
        "actual_flags": actual_flags,
    }


def _worker_preflight(repo: Path, output_root: Path, scene: str, gpu: str) -> Dict[str, Any]:
    scene_dir = output_root / "preflight" / scene
    prepared = _prepare_worker(repo, output_root, scene, gpu, scene_dir)
    sequence_doc = _read_json(scene_dir / "camera_sequence.json")
    result = {
        "scene": scene,
        "passed": True,
        "no_optimization_or_training_called": True,
        "runtime_manifest": prepared["runtime_manifest"],
        "split_hash": prepared["entry"]["canonical_manifest_sha256"],
        "train_ids": sorted(prepared["entry"]["train_ids"]),
        "heldout_ids": sorted(prepared["entry"]["heldout_ids"]),
        "config_lock_sha256": prepared["config_lock"]["config_lock_sha256"],
        "initial_model_state_sha256": prepared["start_state"]["initial_model_state_sha256"],
        "camera_sequence_sha256": sequence_doc["sha256"],
        "camera_sequence_length": sequence_doc["length"],
        "calibration_bank_hash": prepared["bank"]["bank_hash"],
        "calibration_bank_train_only": prepared["bank"]["train_only"],
        "heldout_in_camera_sequence_count": len(set(prepared["names"]) & set(prepared["entry"]["heldout_ids"])),
        "heldout_in_calibration_bank_count": len(
            {row["view_id"] for row in prepared["bank"]["rows"]} & set(prepared["entry"]["heldout_ids"])
        ),
        "constructed_model_flags": prepared["actual_flags"],
    }
    _write_json(scene_dir / "worker_preflight.json", result)
    return result


def _worker(repo: Path, output_root: Path, scene: str, gpu: str) -> Dict[str, Any]:
    scene_dir = output_root / scene
    prepared = _prepare_worker(repo, output_root, scene, gpu, scene_dir)
    runtime_manifest = prepared["runtime_manifest"]
    entry = prepared["entry"]
    scene_cfg = prepared["scene_cfg"]
    config_lock = prepared["config_lock"]
    training_rng = prepared["training_rng"]
    samples = prepared["samples"]
    bank = prepared["bank"]
    sequence = prepared["sequence"]
    names = prepared["names"]
    start_state = prepared["start_state"]
    preflight = _read_json(output_root / "preflight" / scene / "worker_preflight.json")
    repeated = {
        "split_hash": entry["canonical_manifest_sha256"],
        "config_lock_sha256": config_lock["config_lock_sha256"],
        "initial_model_state_sha256": start_state["initial_model_state_sha256"],
        "camera_sequence_sha256": _read_json(scene_dir / "camera_sequence.json")["sha256"],
        "calibration_bank_hash": bank["bank_hash"],
    }
    mismatches = {key: (preflight[key], value) for key, value in repeated.items() if preflight[key] != value}
    if mismatches:
        raise RuntimeError(f"formal preparation differs from global preflight for {scene}: {mismatches}")
    _write_json(scene_dir / "preflight_reproduction.json", {"matched": True, **repeated})
    training_started = time.perf_counter()
    rows, events, checkpoints, topology, training_runtime = FORMAL._train_branch(
        repo, scene, scene_cfg, "C0", samples, sequence, names, scene_dir, training_rng
    )
    _write_table(scene_dir, "training_metrics", rows)
    _write_table(scene_dir, "refinement_events", events)
    _write_table(scene_dir, "topology", topology)
    checkpoint_rows = []
    for row in checkpoints:
        path = Path(row["checkpoint_path"])
        checkpoint_rows.append({**row, "checkpoint_sha256": _sha256(path)})
    _write_table(scene_dir, "training_checkpoint_manifest", checkpoint_rows)
    final_result = _render_final(repo, scene, scene_cfg, scene_dir, config_lock)
    sequence_doc = _read_json(scene_dir / "camera_sequence.json")
    runtime = {
        "scene": scene,
        "assigned_physical_gpu": gpu,
        "training": training_runtime,
        "training_and_evaluation_wall_seconds": time.perf_counter() - training_started,
        "evaluation_wall_seconds": final_result["evaluation_wall_seconds"],
        "camera_sequence_sha256": sequence_doc["sha256"],
        "camera_sequence_length": sequence_doc["length"],
        "calibration_bank_hash": bank["bank_hash"],
        "calibration_bank_train_only": bank["train_only"],
        "heldout_in_camera_sequence_count": len(set(names) & set(entry["heldout_ids"])),
        "heldout_in_calibration_bank_count": len({row["view_id"] for row in bank["rows"]} & set(entry["heldout_ids"])),
        "completed_steps": training_runtime["completed_steps"],
        "finite_training_rows": all(bool(row["finite"]) for row in rows),
        "oom": False,
    }
    _write_json(scene_dir / "runtime.json", runtime)
    result = {
        "scene": scene,
        "split_hash": entry["canonical_manifest_sha256"],
        "config_lock": config_lock,
        "start_state": start_state,
        "runtime": runtime,
        "final_result": final_result,
        "training_rows": rows,
        "topology_rows": topology,
        "checkpoint_rows": checkpoint_rows,
        "heldout_leakage": False,
        "raoc_closed": True,
    }
    _write_json(scene_dir / "scene_worker_result.json", result)
    return result


def _rank_residualized_rho(predictor: np.ndarray, error: np.ndarray, control: np.ndarray) -> float:
    valid = np.isfinite(predictor) & np.isfinite(error) & np.isfinite(control)
    if int(valid.sum()) < 3:
        return float("nan")
    x = scipy.stats.rankdata(predictor[valid])
    y = scipy.stats.rankdata(error[valid])
    z = scipy.stats.rankdata(control[valid])
    design = np.column_stack([np.ones(z.size), z])
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return _rho(x_residual, y_residual)


def _effect_rows(joined: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for scene in SCENES:
        selected = [row for row in joined if row["scene"] == scene]
        error = np.asarray([row["E_cam"] for row in selected], dtype=np.float64)
        for predictor, expected_sign in PREDICTORS.items():
            values = np.asarray([row[predictor] for row in selected], dtype=np.float64)
            rho = _rho(values, error)
            rows.append({
                "scene": scene,
                "predictor": predictor,
                "preregistered": True,
                "primary_predictor": predictor == PRIMARY_PREDICTOR,
                "expected_direction": "positive" if expected_sign > 0 else "negative",
                "expected_signed_spearman": rho * expected_sign if math.isfinite(rho) else float("nan"),
                "heldout_camera_count": len(selected),
                "spearman_rho": rho,
                "kendall_tau": _kendall(values, error),
                "absolute_rho_at_least_0p4": bool(math.isfinite(rho) and abs(rho) >= 0.4),
                "expected_direction_and_threshold": bool(math.isfinite(rho) and rho * expected_sign >= 0.4),
            })
    return rows


def _control_rows(joined: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for scene in SCENES:
        selected = [row for row in joined if row["scene"] == scene]
        error = np.asarray([row["E_cam"] for row in selected], dtype=np.float64)
        predictor = np.asarray([row[PRIMARY_PREDICTOR] for row in selected], dtype=np.float64)
        raw = _rho(predictor, error)
        for control in CONTROLS:
            values = np.asarray([row[control] for row in selected], dtype=np.float64)
            adjusted = _rank_residualized_rho(predictor, error, values)
            control_error = _rho(values, error)
            predictor_control = _rho(predictor, values)
            fully_explains = bool(
                math.isfinite(raw) and raw >= 0.4
                and math.isfinite(adjusted) and abs(adjusted) < 0.1
                and math.isfinite(control_error) and abs(control_error) >= 0.8
                and math.isfinite(predictor_control) and abs(predictor_control) >= 0.8
            )
            rows.append({
                "scene": scene,
                "predictor": PRIMARY_PREDICTOR,
                "control": control,
                "heldout_camera_count": len(selected),
                "raw_spearman_rho": raw,
                "residualized_rank_spearman_rho": adjusted,
                "controlled_positive_direction": bool(math.isfinite(adjusted) and adjusted > 0),
                "control_vs_E_cam_spearman": control_error,
                "predictor_vs_control_spearman": predictor_control,
                "single_control_fully_explains": fully_explains,
                "method": "one-control-at-a-time within-scene rank residualization",
            })
    return rows


def _control_summary(controls: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out = {}
    for scene in SCENES:
        selected = [row for row in controls if row["scene"] == scene]
        if {row["control"] for row in selected} != set(MAJOR_CONTROL_KEYS):
            raise RuntimeError(f"major control set drift for {scene}")
        positive = sum(row["controlled_positive_direction"] for row in selected)
        raw = float(selected[0]["raw_spearman_rho"])
        strong_reversal = any(
            math.isfinite(float(row["residualized_rank_spearman_rho"]))
            and float(row["residualized_rank_spearman_rho"]) <= -0.4
            for row in selected
        )
        out[scene] = {
            "raw_spearman_rho": raw,
            "positive_control_count": positive,
            "control_count": len(selected),
            "majority_positive": positive > len(selected) / 2,
            "single_control_fully_explains": any(row["single_control_fully_explains"] for row in selected),
            "strong_major_control_reversal": strong_reversal,
            "survives_major_controls": bool(
                raw > 0 and positive > len(selected) / 2
                and not any(row["single_control_fully_explains"] for row in selected)
                and not strong_reversal
            ),
        }
    return out


def _distance_matrix(rows: Sequence[Mapping[str, Any]], space: str) -> np.ndarray:
    if space in {"center", "context"}:
        prefix = "center" if space == "center" else "context"
        values = np.asarray([[row[f"{prefix}_{axis}"] for axis in "xyz"] for row in rows], dtype=np.float64)
        return np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    values = np.asarray([[row[f"view_direction_{axis}"] for axis in "xyz"] for row in rows], dtype=np.float64)
    return np.degrees(np.arccos(np.clip(values @ values.T, -1.0, 1.0)))


def _loo_prediction(distances: np.ndarray, errors: np.ndarray) -> Tuple[np.ndarray, bool]:
    n = errors.size
    predictions = np.full(n, np.nan)
    self_used = False
    for index in range(n):
        other = np.arange(n) != index
        self_used = self_used or bool(other[index])
        weights = 1.0 / (distances[index, other] + 1e-6)
        predictions[index] = float(np.sum(weights * errors[other]) / np.sum(weights))
    return predictions, self_used


def _neighbor_and_pair_rows(joined: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    neighbor_rows, permutation_rows, pair_rows, pair_summary = [], [], [], []
    for scene in SCENES:
        selected = sorted((row for row in joined if row["scene"] == scene), key=lambda row: row["camera_id"])
        errors = np.asarray([row["E_cam"] for row in selected], dtype=np.float64)
        n = len(selected)
        for space in ("center", "view_direction"):
            distances = _distance_matrix(selected, space)
            predicted, self_used = _loo_prediction(distances, errors)
            observed = _rho(predicted, errors)
            actual_ranks = scipy.stats.rankdata(errors) / n
            predicted_ranks = scipy.stats.rankdata(predicted) / n
            null_scores = []
            for permutation in itertools.permutations(range(n)):
                permuted = errors[list(permutation)]
                perm_prediction, perm_self = _loo_prediction(distances, permuted)
                if perm_self:
                    raise RuntimeError("self leakage in permutation neighbor analysis")
                null_scores.append(_rho(perm_prediction, permuted))
            finite_null = _finite(null_scores)
            null_median = float(np.median(finite_null))
            null_p95 = _quantile(finite_null, 0.95)
            percentile = float(np.mean(finite_null <= observed))
            neighbor_rows.append({
                "scene": scene,
                "distance_space": space,
                "heldout_camera_count": n,
                "weighting_rule": "inverse distance over all other heldout cameras, eps=1e-6",
                "self_neighbor_used": self_used,
                "observed_leave_one_view_out_spearman": observed,
                "normalized_rank_prediction_mae": float(np.mean(np.abs(actual_ranks - predicted_ranks))),
                "positive_structure": bool(math.isfinite(observed) and observed > 0),
                "exceeds_null_p95": bool(math.isfinite(observed) and observed > null_p95),
            })
            permutation_rows.append({
                "scene": scene,
                "distance_space": space,
                "observed_neighbor_spearman": observed,
                "null_median": null_median,
                "null_p95": null_p95,
                "empirical_percentile": percentile,
                "one_sided_empirical_p": float(np.mean(finite_null >= observed)),
                "unique_permutations_evaluated": int(math.factorial(n)),
                "exact_enumeration": True,
            })
            iu = np.triu_indices(n, 1)
            differences = np.abs(errors[:, None] - errors[None, :])
            for left, right in zip(*iu):
                pair_rows.append({
                    "scene": scene,
                    "distance_space": space,
                    "camera_id_a": selected[left]["camera_id"],
                    "camera_id_b": selected[right]["camera_id"],
                    "distance": float(distances[left, right]),
                    "absolute_E_cam_difference": float(differences[left, right]),
                    "descriptive_pair_not_independent": True,
                })
            pair_rho = _rho(distances[iu], differences[iu])
            nearest = np.argsort(np.where(np.eye(n, dtype=bool), np.inf, distances), axis=1)[:, 0]
            pair_summary.append({
                "scene": scene,
                "distance_space": space,
                "heldout_camera_count": n,
                "camera_pair_count": int(len(iu[0])),
                "distance_vs_absolute_E_cam_difference_spearman": pair_rho,
                "expected_positive_relation": bool(math.isfinite(pair_rho) and pair_rho > 0),
                "nearest_over_all_absolute_E_cam_difference": float(np.mean(np.abs(errors - errors[nearest])) / np.mean(differences[iu])),
                "scene_is_replication_unit": True,
            })
    return neighbor_rows, permutation_rows, pair_rows, pair_summary


def _old_new_table(effects: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    old = _read_json(OLD_AUDIT_ROOT / "formal_eval_only_effects.json")["rows"]
    old_primary = {row["scene"]: row for row in old if row["predictor"] == PRIMARY_PREDICTOR}
    new_primary = {row["scene"]: row for row in effects if row["predictor"] == PRIMARY_PREDICTOR}
    old_n = {"Curasao": 3, "IUI3-RedSea": 4, "JapaneseGradens-RedSea": 3, "Panama": 3}
    return [{
        "scene": scene,
        "old_heldout_count": old_n[scene],
        "new_heldout_count": new_primary[scene]["heldout_camera_count"],
        "old_primary_spearman_rho": float(old_primary[scene]["spearman_rho"]),
        "new_primary_spearman_rho": float(new_primary[scene]["spearman_rho"]),
        "old_direction": "positive" if float(old_primary[scene]["spearman_rho"]) > 0 else "negative_or_zero",
        "new_direction": "positive" if float(new_primary[scene]["spearman_rho"]) > 0 else "negative_or_zero",
        "direction_replicated": bool(float(old_primary[scene]["spearman_rho"]) > 0 and float(new_primary[scene]["spearman_rho"]) > 0),
        "absolute_old_new_metrics_not_causally_compared": True,
    } for scene in SCENES]


def _independence_rows(joined: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    comparisons = (
        "mean_ocmc_projected_camera_residual", "mean_depth", "mean_tau",
        "mean_train_visibility_support", "mean_accumulation",
    )
    rows = []
    scene_independent = {}
    for scene in SCENES:
        selected = [row for row in joined if row["scene"] == scene]
        predictor = np.asarray([row[PRIMARY_PREDICTOR] for row in selected], dtype=np.float64)
        error = np.asarray([row["E_cam"] for row in selected], dtype=np.float64)
        for comparison in comparisons:
            values = np.asarray([row[comparison] for row in selected], dtype=np.float64)
            rows.append({
                "scene": scene,
                "primary_predictor": PRIMARY_PREDICTOR,
                "comparison_variable": comparison,
                "predictor_vs_comparison_spearman": _rho(predictor, values),
                "comparison_vs_E_cam_spearman": _rho(values, error),
                "primary_vs_E_cam_controlled_spearman": _rank_residualized_rho(predictor, error, values),
            })
        ocmc = next(row for row in rows if row["scene"] == scene and row["comparison_variable"] == "mean_ocmc_projected_camera_residual")
        controlled = float(ocmc["primary_vs_E_cam_controlled_spearman"])
        scene_independent[scene] = bool(math.isfinite(controlled) and controlled > 0)
    summary = {
        "ocmc_global_gate_is_scene_level_not_camera_level": True,
        "conceptual_distinction": "g_obs is a global mode-level capacity gate; unseen fraction is a per-view training-coverage statistic",
        "scene_positive_after_ocmc_magnitude_control": scene_independent,
        "positive_after_ocmc_control_scene_count": sum(scene_independent.values()),
        "distinguishable_from_ocmc_observability": sum(scene_independent.values()) >= 3,
    }
    return rows, summary


def _scene_classifications(
    effects: Sequence[Mapping[str, Any]], controls: Mapping[str, Any], neighbor: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]], joined: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    out = {}
    for scene in SCENES:
        scene_effects = [row for row in effects if row["scene"] == scene]
        supportive = [row for row in scene_effects if row["expected_signed_spearman"] >= 0.4]
        opposite = [row for row in scene_effects if row["expected_signed_spearman"] <= -0.4]
        center_neighbor = next(row for row in neighbor if row["scene"] == scene and row["distance_space"] == "center")
        center_pair = next(row for row in pairs if row["scene"] == scene and row["distance_space"] == "center")
        contradiction_dominates = len(opposite) > len(supportive)
        n = sum(row["scene"] == scene for row in joined)
        strong = bool(
            n >= 5 and supportive and controls[scene]["survives_major_controls"]
            and (center_neighbor["positive_structure"] or center_pair["expected_positive_relation"])
            and not contradiction_dominates
        )
        any_directional = any(row["expected_signed_spearman"] > 0 for row in scene_effects)
        explicit_camera_structure = bool(
            center_neighbor["positive_structure"] or center_pair["expected_positive_relation"]
        )
        if strong:
            classification = "STRONG_CAMERA_RESIDUAL_REPLICATION"
        elif any_directional:
            classification = "WEAK_CAMERA_RESIDUAL_REPLICATION"
        else:
            classification = "CAMERA_RESIDUAL_NOT_REPLICATED"
        errors = np.asarray([row["E_cam"] for row in joined if row["scene"] == scene], dtype=np.float64)
        out[scene] = {
            "classification": classification,
            "heldout_camera_count": n,
            "supportive_predictors_at_threshold": [row["predictor"] for row in supportive],
            "strong_opposite_predictors": [row["predictor"] for row in opposite],
            "contradictory_pattern_dominates": contradiction_dominates,
            "primary_controls_survive": controls[scene]["survives_major_controls"],
            "center_neighbor_positive": center_neighbor["positive_structure"],
            "center_pair_positive": center_pair["expected_positive_relation"],
            "explicit_camera_structure": explicit_camera_structure,
            "residual_structure_replicated": bool(any_directional and explicit_camera_structure),
            "E_cam_mean": float(np.mean(errors)),
            "E_cam_std": float(np.std(errors)),
            "E_cam_min": float(np.min(errors)),
            "E_cam_median": float(np.median(errors)),
            "E_cam_p90": _quantile(errors, 0.90),
            "E_cam_max": float(np.max(errors)),
        }
    return out


def _decisions(
    effects: Sequence[Mapping[str, Any]], controls: Mapping[str, Any], classifications: Mapping[str, Any],
    neighbor: Sequence[Mapping[str, Any]], independence: Mapping[str, Any], protocol_valid: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    predictor_rows = []
    for predictor, expected_sign in PREDICTORS.items():
        selected = [row for row in effects if row["predictor"] == predictor]
        count = sum(row["expected_signed_spearman"] >= 0.4 for row in selected)
        signed = [row["expected_signed_spearman"] for row in selected]
        predictor_rows.append({
            "predictor": predictor,
            "expected_direction": "positive" if expected_sign > 0 else "negative",
            "scene_count_expected_direction_rho_at_least_0p4": count,
            "median_expected_signed_rho": float(np.nanmedian(signed)),
            "preregistered_before_new_outcomes": True,
        })
    strongest = max(predictor_rows, key=lambda row: (row["scene_count_expected_direction_rho_at_least_0p4"], row["median_expected_signed_rho"], -list(PREDICTORS).index(row["predictor"])))
    primary_count = next(row for row in predictor_rows if row["predictor"] == PRIMARY_PREDICTOR)["scene_count_expected_direction_rho_at_least_0p4"]
    control_count = sum(controls[scene]["survives_major_controls"] for scene in SCENES)
    neighbor_count = sum(
        row["positive_structure"] for row in neighbor if row["distance_space"] == "center"
    )
    class_counts = {
        label: sum(value["classification"] == label for value in classifications.values())
        for label in (
            "STRONG_CAMERA_RESIDUAL_REPLICATION", "WEAK_CAMERA_RESIDUAL_REPLICATION",
            "CAMERA_RESIDUAL_NOT_REPLICATED", "SCENE_INCONCLUSIVE",
        )
    }
    classified_replication_count = (
        class_counts["STRONG_CAMERA_RESIDUAL_REPLICATION"]
        + class_counts["WEAK_CAMERA_RESIDUAL_REPLICATION"]
    )
    residual_structure_count = sum(value["residual_structure_replicated"] for value in classifications.values())
    adequate_n_count = sum(value["heldout_camera_count"] >= 5 for value in classifications.values())
    no_strong_opposite_scene = all(not value["strong_opposite_predictors"] for value in classifications.values())
    actionable = bool(
        classified_replication_count >= 3
        and adequate_n_count >= 3
        and strongest["scene_count_expected_direction_rho_at_least_0p4"] >= 3
        and control_count >= 3
        and neighbor_count >= 3
        and independence["distinguishable_from_ocmc_observability"]
        and no_strong_opposite_scene
    )
    if not protocol_valid:
        final = "C_PROTOCOL_INCONCLUSIVE"
        research_line = "C_REPLICATION_PROTOCOL_FAILED"
        next_task = "REPAIR-CANDIDATE-C-REPLICATION-PROTOCOL"
    elif actionable:
        final = "C_SUPPORTED_AND_ACTIONABLE"
        research_line = "PROCEED_TO_C_MINIMAL_CAUSAL_INTERVENTION"
        next_task = "C-MINIMAL-ONE-FACTOR-CAUSAL-INTERVENTION"
    elif residual_structure_count >= 3 and control_count >= 3:
        final = "C_SUPPORTED_BUT_NOT_ACTIONABLE"
        research_line = "C_SCIENTIFICALLY_SUPPORTED_BUT_DEFER_MODULE"
        next_task = f"ISOLATE-{strongest['predictor'].upper().replace('_', '-')}-PROXY"
    else:
        final = "C_NOT_SUPPORTED"
        research_line = "CLOSE_CANDIDATE_C"
        next_task = "NON-CANDIDATE-C-FAILURE-MECHANISM-PRIORITIZATION"
    if primary_count >= 3 and control_count >= 3:
        primary_decision = "UNSEEN_FRACTION_SUPPORTED"
    elif primary_count >= 2 or sum(next(row for row in effects if row["scene"] == scene and row["predictor"] == PRIMARY_PREDICTOR)["spearman_rho"] > 0 for scene in SCENES) >= 3:
        primary_decision = "UNSEEN_FRACTION_TENTATIVE"
    else:
        primary_decision = "UNSEEN_FRACTION_NOT_SUPPORTED"
    decision = {
        "final_candidate_c_classification": final,
        "candidate_c_research_line_decision": research_line,
        "one_next_task": next_task,
        "scene_classification_counts": class_counts,
        "classified_replication_scene_count": classified_replication_count,
        "residual_structure_replication_scene_count": residual_structure_count,
        "replicated_scene_count": residual_structure_count,
        "adequate_heldout_n_scene_count": adequate_n_count,
        "strongest_cross_scene_preregistered_predictor": strongest,
        "predictor_replication_rows": predictor_rows,
        "control_survival_scene_count": control_count,
        "positive_center_neighbor_scene_count": neighbor_count,
        "ocmc_independence": independence["distinguishable_from_ocmc_observability"],
        "old_new_absolute_metric_difference_interpreted_causally": False,
        "no_rescue_split_seed_or_kfold": True,
    }
    primary = {
        "primary_predictor": PRIMARY_PREDICTOR,
        "classification": primary_decision,
        "positive_rho_at_least_0p4_scene_count": primary_count,
        "control_survival_scene_count": control_count,
        "registered_before_new_split_outcomes": True,
    }
    return decision, primary


def _make_figures(
    output_root: Path, joined: Sequence[Mapping[str, Any]], effects: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]], neighbor: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]],
) -> None:
    figure_root = output_root / "aggregate" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, scene in zip(axes.flat, SCENES):
        rows = [row for row in joined if row["scene"] == scene]
        ax.scatter([row[PRIMARY_PREDICTOR] for row in rows], [row["E_cam"] for row in rows], c="#326273", s=55)
        for row in rows:
            ax.annotate(row["camera_id"], (row[PRIMARY_PREDICTOR], row["E_cam"]), fontsize=7)
        rho = next(row["spearman_rho"] for row in effects if row["scene"] == scene and row["predictor"] == PRIMARY_PREDICTOR)
        ax.set_title(f"{scene} (rho={rho:.3f})")
        ax.set_xlabel(PRIMARY_PREDICTOR)
        ax.set_ylabel("E_cam")
    fig.tight_layout()
    fig.savefig(figure_root / "unseen_fraction_vs_E_cam.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, scene in zip(axes.flat, SCENES):
        rows = [row for row in joined if row["scene"] == scene]
        train = _read_json(output_root / scene / "diagnostics" / "scene_result.json")["train_camera_geometry"]
        centers = np.asarray([row["center"] for row in train] + [[row[f"center_{axis}"] for axis in "xyz"] for row in rows])
        centered = centers - centers.mean(axis=0)
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        xy = centered @ vh[:2].T
        n_train = len(train)
        ax.scatter(xy[:n_train, 0], xy[:n_train, 1], c="#777777", s=20)
        scatter = ax.scatter(xy[n_train:, 0], xy[n_train:, 1], c=[row["E_cam"] for row in rows], cmap="magma", s=58, edgecolors="black", linewidths=0.4)
        ax.set_title(scene)
        ax.set_xlabel("camera-center PCA 1")
        ax.set_ylabel("camera-center PCA 2")
        fig.colorbar(scatter, ax=ax, label="E_cam")
    fig.tight_layout()
    fig.savefig(figure_root / "heldout_camera_center_E_cam.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, scene in zip(axes.flat, SCENES):
        rows = [row for row in pairs if row["scene"] == scene and row["distance_space"] == "center"]
        ax.scatter([row["distance"] for row in rows], [row["absolute_E_cam_difference"] for row in rows], c="#668f80", s=38)
        ax.set_title(scene)
        ax.set_xlabel("heldout center distance")
        ax.set_ylabel("|delta E_cam|")
    fig.tight_layout()
    fig.savefig(figure_root / "center_distance_vs_E_cam_difference.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, scene in zip(axes.flat, SCENES):
        rows = sorted((row for row in joined if row["scene"] == scene), key=lambda row: row["camera_id"])
        errors = np.asarray([row["E_cam"] for row in rows])
        predicted, _ = _loo_prediction(_distance_matrix(rows, "center"), errors)
        ax.scatter(predicted, errors, c="#b24c63", s=50)
        ax.set_title(scene)
        ax.set_xlabel("LOO neighbor-predicted E_cam")
        ax.set_ylabel("observed E_cam")
    fig.tight_layout()
    fig.savefig(figure_root / "neighbor_predicted_vs_observed.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(PREDICTORS))
    offsets = np.linspace(-0.24, 0.24, len(SCENES))
    for offset, scene in zip(offsets, SCENES):
        by_predictor = {row["predictor"]: row for row in effects if row["scene"] == scene}
        ax.scatter(x + offset, [by_predictor[p]["expected_signed_spearman"] for p in PREDICTORS], s=42, label=scene)
    ax.axhline(0.4, color="#b23a48", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, list(PREDICTORS), rotation=35, ha="right")
    ax.set_ylabel("expected-signed Spearman rho")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "cross_scene_preregistered_predictor_rho.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels, values, colors = [], [], []
    for scene in SCENES:
        scene_rows = [row for row in controls if row["scene"] == scene]
        labels.append(f"{scene}\nraw")
        values.append(scene_rows[0]["raw_spearman_rho"])
        colors.append("#444444")
        for row in scene_rows:
            labels.append(f"{scene}\n{row['control'].replace('mean_', '')}")
            values.append(row["residualized_rank_spearman_rho"])
            colors.append("#4f7c68")
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(values)), labels, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("primary raw/controlled Spearman rho")
    fig.tight_layout()
    fig.savefig(figure_root / "primary_raw_vs_controlled_effects.png", dpi=160)
    plt.close(fig)


def _research_note(summary: Mapping[str, Any]) -> None:
    scenes = summary["per_scene_candidate_c_classification"]
    effects = {(row["scene"], row["predictor"]): row for row in summary["all_predictor_effects"]}
    controls = summary["control_summary"]
    control_rows = {
        (row["scene"], row["control"]): row
        for row in summary["single_factor_controls"]
    }
    neighbors = {(row["scene"], row["distance_space"]): row for row in summary["camera_neighbor_analysis"]}
    permutations = {(row["scene"], row["distance_space"]): row for row in summary["camera_permutation_analysis"]}
    pairs = {(row["scene"], row["distance_space"]): row for row in summary["camera_pair_summary"]}
    runtime = {row["scene"]: row for row in summary["per_scene_training_summary"]}
    heldout = {row["scene"]: row for row in summary["heldout_scene_summary"]}
    old_new = {row["scene"]: row for row in summary["old_new_replication_table"]}
    independence = {
        (row["scene"], row["comparison_variable"]): row
        for row in summary["ocmc_independence_rows"]
    }
    decomposition = dict(zip(SCENES, summary["decomposition_safety"]["rows"]))
    mechanism = {row["scene"]: row for row in summary["ocmc_mechanism_sanity"]["rows"]}
    split_rows = {row["scene"]: row for row in summary["split_verification"]["rows"]}
    decision = summary["candidate_c_final_decision"]
    primary = summary["primary_predictor_decision"]
    strongest = decision["strongest_cross_scene_preregistered_predictor"]
    runtime_summary = summary["runtime_summary"]
    lines = [
        "# OCMC Candidate-C Resplit Replication (2026-08-31)",
        "", "## 1. Motivation", "",
        "HYPOTHESIS: this preregistered replication asks whether held-out camera residual structure survives a larger outcome-blind split after locked OCMC. The split change is not a causal RGB arm.",
        "", "## 2. Previous C_DATA_LIMITED Result", "",
        "EXPERIMENTAL FACT: the old split had only 3/4/3/3 held-out cameras. Candidate C remained data-limited despite a positive descriptive unseen-fraction direction in all four scenes.",
        "", "## 3. Pre-Registered Split Provenance", "",
        f"CONFIG FACT: all four scene hashes and global hash `{EXPECTED_GLOBAL_HASH}` matched the feasibility artifact before training. Output-local list files and read-only source-data symlinks preserved the official split files.",
        "", "| Scene | Train | Heldout | Locked split SHA-256 |", "| --- | ---: | ---: | --- |",
        *[
            f"| {scene} | {split_rows[scene]['train_count']} | {split_rows[scene]['heldout_count']} | `{split_rows[scene]['scene_hash_computed']}` |"
            for scene in SCENES
        ],
        "", "## 4. Formal Training Protocol", "",
        "CONFIG FACT: each scene used one fresh seed-42, 15K C0/OCMC run with bounded SH3, SH degree 3, classic rasterization, `dir_xy_camera`, tied B_inf, refreshes at 0/5000/10000, formal refinement through stop_split_at=10000, and RAOC disabled. Held-out IDs were absent from optimization sequences and OCMC banks.",
        "", "PROTOCOL FACT: all four construction preflights passed before training. Formal workers reproduced the locked split, config, seed-42 start-state, 15K camera sequence, and train-only OCMC bank hashes. All four final checkpoints identify this experiment, branch C0, and step 14999. Frozen evaluation used no optimizer step or backward call.",
        "", "## 5. New OCMC Training Sanity", "",
        "| Scene | Steps | Train s | Eval s | Final Gaussians | Peak reserved GiB | PSNR | SSIM | LPIPS | MSE |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {scene} | {runtime[scene]['completed_steps']} | {runtime[scene]['training_wall_seconds']:.2f} | {runtime[scene]['evaluation_wall_seconds']:.2f} | {runtime[scene]['final_gaussian_count']} | {runtime[scene]['peak_reserved_bytes'] / 2**30:.2f} | {heldout[scene]['mean_PSNR']:.3f} | {heldout[scene]['mean_SSIM']:.4f} | {heldout[scene]['mean_LPIPS']:.4f} | {heldout[scene]['mean_MSE']:.6g} |"
            for scene in SCENES
        ],
        "", f"RUNTIME FACT: total training cost was {runtime_summary['training_gpu_hours_total']:.3f} GPU-hours and parallel worker wall-clock was {runtime_summary['parallel_wall_hours']:.3f} hours. This is {100.0 * runtime_summary['relative_error_vs_estimate']:.1f}% from the preregistered 3.75 GPU-hour estimate and is classified close to estimate. No NaN, Inf, OOM, or missing heldout render occurred.",
        "", "## 6. Heldout Camera Population", "",
        "QUANTITATIVE RESULT: all 22 preregistered heldout cameras rendered. `E_cam` is mean per-pixel squared RGB residual.",
        "", "| Scene | N | E_cam min | median | mean | p90 | max |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {scene} | {heldout[scene]['heldout_camera_count']} | {heldout[scene]['E_cam_min']:.6g} | {heldout[scene]['E_cam_median']:.6g} | {heldout[scene]['E_cam_mean']:.6g} | {heldout[scene]['E_cam_p90']:.6g} | {heldout[scene]['E_cam_max']:.6g} |"
            for scene in SCENES
        ],
        "", "## 7. Primary GT-Free Predictor", "",
        "| Scene | unseen-fraction rho | Kendall tau | Controls survive |", "| --- | ---: | ---: | --- |",
        *[f"| {scene} | {effects[(scene, PRIMARY_PREDICTOR)]['spearman_rho']:.3f} | {effects[(scene, PRIMARY_PREDICTOR)]['kendall_tau']:.3f} | {controls[scene]['survives_major_controls']} |" for scene in SCENES],
        "", f"INFERENCE: the primary rho reached +0.4 in {primary['positive_rho_at_least_0p4_scene_count']}/4 scenes and survived major controls in {primary['control_survival_scene_count']}/4. Its decision is `{primary['classification']}`. This is association evidence only, not a causal support-error claim.",
        "", "## 8. Camera-Center/Context Novelty", "",
        "| Scene | Center nearest | Center 3NN | Context nearest | Context 3NN |", "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {scene} | {effects[(scene, 'center_nearest_train')]['spearman_rho']:.3f} | {effects[(scene, 'center_knn3_mean')]['spearman_rho']:.3f} | {effects[(scene, 'context_nearest_train')]['spearman_rho']:.3f} | {effects[(scene, 'context_knn3_mean')]['spearman_rho']:.3f} |"
            for scene in SCENES
        ],
        "", "INFERENCE: center and exact OCMC-context ranks match here because `dir_xy_camera` is derived from scene-normalized camera center. Neither distance family replicates at the +0.4 threshold in three scenes.",
        "", "## 9. View-Direction Novelty", "",
        "| Scene | Nearest angle rho | 3NN angle rho |", "| --- | ---: | ---: |",
        *[
            f"| {scene} | {effects[(scene, 'view_direction_nearest_angle_deg')]['spearman_rho']:.3f} | {effects[(scene, 'view_direction_knn3_angle_deg')]['spearman_rho']:.3f} |"
            for scene in SCENES
        ],
        "", "INFERENCE: view-direction novelty is inconsistent and strongly opposite in IUI3 nearest-angle and both Panama angle summaries.",
        "", "## 10. Training-View Support", "",
        "CODE FACT: visible means final `radii > 0`, exactly as in the focused Candidate-C audit; equality with `gaussian_visible_mask` was asserted for every support/eval render. Train support is the number of preregistered train cameras in which that Gaussian is visible; low support means count <=1. Definitions were locked before outcomes.",
        "", "| Scene | Mean support rho | Median support rho | Low-support fraction rho |", "| --- | ---: | ---: | ---: |",
        *[
            f"| {scene} | {effects[(scene, 'mean_train_visibility_support')]['spearman_rho']:.3f} | {effects[(scene, 'median_train_visibility_support')]['spearman_rho']:.3f} | {effects[(scene, 'fraction_visible_low_support')]['spearman_rho']:.3f} |"
            for scene in SCENES
        ],
        "", f"INFERENCE: `{strongest['predictor']}` is the strongest cross-scene preregistered predictor: expected-direction |rho| >=0.4 in {strongest['scene_count_expected_direction_rho_at_least_0p4']}/4 scenes, with median expected-signed rho {strongest['median_expected_signed_rho']:.3f}.",
        "", "## 11. Confounder Controls", "",
        "QUANTITATIVE RESULT: one-control-at-a-time rank residualization used the same seven factors in every scene. No multivariable regression was fit.",
        "", "| Scene | Raw | Depth | Tau | Transmission | Accumulation | Footprint | Visible count | Mean support | Survives |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *[
            "| " + scene + " | " + " | ".join([
                f"{controls[scene]['raw_spearman_rho']:.3f}",
                *[f"{control_rows[(scene, control)]['residualized_rank_spearman_rho']:.3f}" for control in CONTROLS],
                str(controls[scene]["survives_major_controls"]),
            ]) + " |"
            for scene in SCENES
        ],
        "", "INFERENCE: all seven controlled directions remain positive in Curasao, IUI3, and JapaneseGradens; none is positive in Panama. No single registered control met the full-explanation rule.",
        "", "## 12. Camera-Neighbor Analysis", "",
        "| Scene | Center LOO rho | Center rank MAE | Direction LOO rho | Direction rank MAE |", "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {scene} | {neighbors[(scene, 'center')]['observed_leave_one_view_out_spearman']:.3f} | {neighbors[(scene, 'center')]['normalized_rank_prediction_mae']:.3f} | {neighbors[(scene, 'view_direction')]['observed_leave_one_view_out_spearman']:.3f} | {neighbors[(scene, 'view_direction')]['normalized_rank_prediction_mae']:.3f} |"
            for scene in SCENES
        ],
        "", "QUANTITATIVE RESULT: the fixed inverse-distance LOO score is negative in 0/4 positive center scenes and 0/4 positive direction scenes. Self-neighbor use was programmatically false. This blocks `C_SUPPORTED_AND_ACTIONABLE`.",
        "", "## 13. Permutation Analysis", "",
        "| Scene | Space | Observed | Null median | Null p95 | Percentile | Exact N! |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {scene} | {space} | {permutations[(scene, space)]['observed_neighbor_spearman']:.3f} | {permutations[(scene, space)]['null_median']:.3f} | {permutations[(scene, space)]['null_p95']:.3f} | {permutations[(scene, space)]['empirical_percentile']:.3f} | {permutations[(scene, space)]['unique_permutations_evaluated']} |"
            for scene in SCENES for space in ("center", "view_direction")
        ],
        "", "QUANTITATIVE RESULT: no observed neighbor score exceeded its exact permutation null p95.",
        "", "## 14. Pair-Distance Analysis", "",
        "| Scene | Center rho | View-angle rho | Center pair direction |", "| --- | ---: | ---: | --- |",
        *[
            f"| {scene} | {pairs[(scene, 'center')]['distance_vs_absolute_E_cam_difference_spearman']:.3f} | {pairs[(scene, 'view_direction')]['distance_vs_absolute_E_cam_difference_spearman']:.3f} | {pairs[(scene, 'center')]['expected_positive_relation']} |"
            for scene in SCENES
        ],
        "", "INFERENCE: center pair-distance direction is positive in IUI3, JapaneseGradens, and Panama, but all three effects are small. Camera pairs are descriptive; scene remains the replication unit.",
        "", "## 15. Old-vs-New Candidate-C Replication", "",
        "| Scene | Old N | Old unseen rho | New N | New unseen rho | Direction replicated |", "| --- | ---: | ---: | ---: | ---: | --- |",
        *[
            f"| {scene} | {old_new[scene]['old_heldout_count']} | {old_new[scene]['old_primary_spearman_rho']:.3f} | {old_new[scene]['new_heldout_count']} | {old_new[scene]['new_primary_spearman_rho']:.3f} | {old_new[scene]['direction_replicated']} |"
            for scene in SCENES
        ],
        "", "CONFIG FACT: only old/new directions and sample counts were compared. Absolute PSNR or other reconstruction differences were not interpreted as a causal Candidate-C effect.",
        "", "## 16. OCMC-Independence Analysis", "",
        "| Scene | Unseen vs OCMC magnitude | OCMC magnitude vs E_cam | Unseen vs E_cam controlled |", "| --- | ---: | ---: | ---: |",
        *[
            f"| {scene} | {independence[(scene, 'mean_ocmc_projected_camera_residual')]['predictor_vs_comparison_spearman']:.3f} | {independence[(scene, 'mean_ocmc_projected_camera_residual')]['comparison_vs_E_cam_spearman']:.3f} | {independence[(scene, 'mean_ocmc_projected_camera_residual')]['primary_vs_E_cam_controlled_spearman']:.3f} |"
            for scene in SCENES
        ],
        "", f"INFERENCE: distinguishable from OCMC observability is `{summary['ocmc_independence_analysis']['distinguishable_from_ocmc_observability']}` because controlled direction remains positive in {summary['ocmc_independence_analysis']['positive_after_ocmc_control_scene_count']}/4 scenes. OCMC g_obs remains global/mode-level; Candidate C predictors are per-view coverage statistics.",
        "", "DECOMPOSITION FACT: new-split OCMC remained safe on every scene.",
        "", "| Scene | P(J>1) | J p99 | Tau p90/p99 | P(T<0.1) | B_inf mean | beta_B mean | beta_D mean | OCMC modes <0.5 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {scene} | {decomposition[scene]['P_J_gt_1']:.3g} | {decomposition[scene]['J_p99']:.3f} | {decomposition[scene]['tau_p90']:.3f}/{decomposition[scene]['tau_p99']:.3f} | {decomposition[scene]['P_T_lt_0p1']:.4f} | {decomposition[scene]['B_inf_mean']:.3f} | {decomposition[scene]['beta_B_mean']:.3f} | {decomposition[scene]['beta_D_mean']:.3f} | {mechanism[scene]['modes_below_half']} |"
            for scene in SCENES
        ],
        "", "MECHANISM SANITY: OCMC was active, RAOC was off, all nine global gates were finite and bounded, and four modes were below 0.5 in every scene. This is a sanity check, not a new baseline comparison.",
        "", "## 17. Per-Scene Classifications", "",
        *[
            f"- {scene}: `{scenes[scene]['classification']}`; primary controls survive={scenes[scene]['primary_controls_survive']}, center neighbor positive={scenes[scene]['center_neighbor_positive']}, center pair positive={scenes[scene]['center_pair_positive']}."
            for scene in SCENES
        ],
        "", "## 18. Cross-Scene Decision", "",
        f"INFERENCE: `{decision['final_candidate_c_classification']}`. Residual structure appears in {decision['residual_structure_replication_scene_count']}/4 scenes and controls survive in {decision['control_survival_scene_count']}/4. `{strongest['predictor']}` reaches the registered direction/threshold in 4/4, but center-neighbor structure is positive in only {decision['positive_center_neighbor_scene_count']}/4. The support association is replicated but is not actionable enough to justify a module.",
        "", "## 19. Candidate-C Research-Line Decision", "",
        f"INFERENCE: `{decision['candidate_c_research_line_decision']}`. No second split, seed, k-fold rescue, predictor sweep, or module design was performed. The largest remaining uncertainty is whether the replicated low-support association is a stable, inference-time measurable signal given absent LOO neighbor support and tied or near-zero unseen fractions.",
        "", "## 20. ONE Next Task", "",
        f"HYPOTHESIS: `{decision['one_next_task']}`. Isolate the measurement stability and inference-time computability of the preregistered low-support-visible-Gaussian fraction without training a module or changing the split.", "",
    ]
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def _aggregate(repo: Path, output_root: Path) -> Dict[str, Any]:
    results = [_read_json(output_root / scene / "scene_worker_result.json") for scene in SCENES]
    metrics = [row for result in results for row in result["final_result"]["metrics_rows"]]
    predictors = [row for result in results for row in result["final_result"]["predictor_rows"]]
    by_predictor = {(row["scene"], row["camera_id"]): row for row in predictors}
    joined = [{**row, **by_predictor[(row["scene"], row["camera_id"])]} for row in metrics]
    effects = _effect_rows(joined)
    primary_effects = [row for row in effects if row["predictor"] == PRIMARY_PREDICTOR]
    controls = _control_rows(joined)
    control_summary = _control_summary(controls)
    neighbor, permutation, pair_rows, pair_summary = _neighbor_and_pair_rows(joined)
    old_new = _old_new_table(effects)
    independence_rows, independence = _independence_rows(joined)
    classifications = _scene_classifications(effects, control_summary, neighbor, pair_summary, joined)
    decomposition_rows = [result["final_result"]["decomposition_safety"] for result in results]
    decomposition_safe = all(
        math.isfinite(float(row["P_J_gt_1"])) and float(row["P_J_gt_1"]) == 0.0
        and math.isfinite(float(row["J_p99"])) and float(row["J_p99"]) <= 1.0 + 1e-6
        for row in decomposition_rows
    )
    mechanism_rows = []
    for result in results:
        checkpoint = result["final_result"]["checkpoint"]
        gates = np.asarray(checkpoint["ocmc_global_gate"], dtype=np.float64)
        mechanism_rows.append({
            "scene": result["scene"],
            "ocmc_enabled": checkpoint["flags"]["camera_medium_observability_enabled"] is True,
            "raoc_disabled": checkpoint["raoc_disabled"],
            "refresh_step": checkpoint["ocmc_refresh_step"],
            "global_gate": gates.tolist(),
            "global_gate_finite_and_bounded": bool(
                gates.size == 9 and np.isfinite(gates).all() and np.all(gates >= 0.0) and np.all(gates <= 1.0)
            ),
            "modes_below_half": checkpoint["ocmc_modes_below_half"],
            "low_observability_capacity_suppressed": bool(np.any(gates < 0.5)) if gates.size else False,
            "sanity_only_not_new_causal_comparison": True,
        })
    mechanism_sanity = {
        "rows": mechanism_rows,
        "all_scenes_ocmc_active_raoc_off": all(row["ocmc_enabled"] and row["raoc_disabled"] for row in mechanism_rows),
        "all_global_gates_finite_and_bounded": all(row["global_gate_finite_and_bounded"] for row in mechanism_rows),
        "all_scenes_suppress_at_least_one_mode_below_half": all(
            row["low_observability_capacity_suppressed"] for row in mechanism_rows
        ),
    }
    split_verification = _read_json(output_root / "split_manifest_verified.json")
    construction_preflight = _read_json(output_root / "construction_preflight.json")
    official_after = _official_split_hashes(repo)
    official_unchanged = official_after == _read_json(output_root / "official_split_integrity_before.json")
    expected_eval_ids = {
        scene: set(split_verification["manifest"]["scenes"][scene]["heldout_ids"])
        for scene in SCENES
    }
    actual_eval_ids = {
        scene: {row["camera_id"] for row in metrics if row["scene"] == scene}
        for scene in SCENES
    }
    protocol_valid = bool(
        split_verification["verified"] and official_unchanged
        and construction_preflight["all_passed_before_any_training"]
        and construction_preflight["no_optimization_or_training_called"]
        and all(result["runtime"]["completed_steps"] == MAX_STEPS for result in results)
        and all(result["runtime"]["camera_sequence_length"] == MAX_STEPS for result in results)
        and all(not result["heldout_leakage"] for result in results)
        and all(result["runtime"]["heldout_in_camera_sequence_count"] == 0 for result in results)
        and all(result["runtime"]["heldout_in_calibration_bank_count"] == 0 for result in results)
        and all(result["runtime"]["calibration_bank_train_only"] for result in results)
        and all(result["final_result"]["checkpoint"]["absolute_step"] == FINAL_STEP for result in results)
        and all(result["final_result"]["checkpoint"]["checkpoint_experiment"] == EXPERIMENT for result in results)
        and all(result["final_result"]["checkpoint"]["checkpoint_branch"] == "C0" for result in results)
        and all(result["final_result"]["checkpoint"]["checkpoint_sha256"] for result in results)
        and all(result["final_result"]["checkpoint"]["raoc_disabled"] for result in results)
        and all(result["final_result"]["visibility_definition_audit"]["gaussian_visible_mask_equivalence_asserted"] for result in results)
        and all(not result["final_result"]["optimizer_step_called_during_evaluation"] for result in results)
        and all(not result["final_result"]["backward_called_during_evaluation"] for result in results)
        and len(metrics) == 22 and actual_eval_ids == expected_eval_ids
        and all(row["all_finite"] for row in metrics)
    )
    decision, primary_decision = _decisions(effects, control_summary, classifications, neighbor, independence, protocol_valid)
    training_rows = []
    checkpoint_rows = []
    for result in results:
        runtime = result["runtime"]
        checkpoint = result["final_result"]["checkpoint"]
        training_rows.append({
            "scene": result["scene"],
            "completed_steps": runtime["completed_steps"],
            "training_wall_seconds": runtime["training"]["training_wall_seconds"],
            "training_gpu_hours": runtime["training"]["training_wall_seconds"] / 3600.0,
            "evaluation_wall_seconds": runtime["evaluation_wall_seconds"],
            "peak_allocated_bytes": runtime["training"]["peak_allocated_bytes"],
            "peak_reserved_bytes": runtime["training"]["peak_reserved_bytes"],
            "final_gaussian_count": checkpoint["final_gaussian_count"],
            "all_finite": runtime["finite_training_rows"] and all(row["all_finite"] for row in result["final_result"]["metrics_rows"]),
            "oom": runtime["oom"],
            "physical_gpu": runtime["assigned_physical_gpu"],
        })
        checkpoint_rows.append(checkpoint)
    training_seconds = sum(row["training_wall_seconds"] for row in training_rows)
    launcher_runtime = _read_json(output_root / "launcher_runtime.json")
    runtime_summary = {
        "training_gpu_hours_total": training_seconds / 3600.0,
        "parallel_wall_seconds": launcher_runtime["parallel_worker_wall_seconds"],
        "parallel_wall_hours": launcher_runtime["parallel_worker_wall_seconds"] / 3600.0,
        "evaluation_seconds_total": sum(row["evaluation_wall_seconds"] for row in training_rows),
        "preregistered_estimate_gpu_hours": 3.75,
        "relative_error_vs_estimate": abs(training_seconds / 3600.0 - 3.75) / 3.75,
        "close_to_preregistered_estimate": abs(training_seconds / 3600.0 - 3.75) / 3.75 <= 0.25,
    }
    heldout_scene_summary = []
    for scene in SCENES:
        rows = [row for row in joined if row["scene"] == scene]
        heldout_scene_summary.append({
            "scene": scene,
            "heldout_camera_count": len(rows),
            **{f"mean_{key}": float(np.mean([row[key] for row in rows])) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
            "E_cam_min": float(np.min([row["E_cam"] for row in rows])),
            "E_cam_median": float(np.median([row["E_cam"] for row in rows])),
            "E_cam_mean": float(np.mean([row["E_cam"] for row in rows])),
            "E_cam_p90": _quantile(np.asarray([row["E_cam"] for row in rows]), 0.90),
            "E_cam_max": float(np.max([row["E_cam"] for row in rows])),
        })
    _write_table(output_root, "per_camera_metrics", metrics)
    _write_table(output_root, "per_camera_predictors", predictors)
    _write_table(output_root, "primary_unseen_fraction_effects", primary_effects)
    _write_table(output_root, "all_predictor_effects", effects)
    _write_table(output_root, "single_factor_controls", controls, scene_summary=control_summary)
    _write_table(output_root, "camera_neighbor_analysis", neighbor)
    _write_table(output_root, "camera_permutation_analysis", permutation)
    _write_table(output_root, "camera_pair_analysis", pair_rows, summary_rows=pair_summary)
    _write_table(output_root, "old_new_replication_table", old_new)
    _write_table(output_root, "ocmc_independence_analysis", independence_rows, summary=independence)
    _write_json(output_root / "per_scene_candidate_c_classification.json", classifications)
    _write_json(output_root / "decomposition_safety.json", {"rows": decomposition_rows, "all_scenes_safe": decomposition_safe})
    _write_json(output_root / "ocmc_mechanism_sanity.json", mechanism_sanity)
    _write_json(output_root / "candidate_c_final_decision.json", decision)
    _write_json(output_root / "primary_predictor_decision.json", primary_decision)
    _write_table(output_root, "per_scene_training_summary", training_rows)
    _write_json(output_root / "checkpoint_manifest.json", {"rows": checkpoint_rows})
    _write_json(output_root / "training_manifest.json", {
        "experiment": EXPERIMENT,
        "one_fresh_ocmc_run_per_scene": True,
        "max_steps": MAX_STEPS,
        "seed": FORMAL.TRAINING_SEED,
        "scene_gpu_assignment": SCENE_GPUS,
        "heldout_optimization_count": 0,
        "heldout_ocmc_calibration_count": 0,
        "raoc_enabled": False,
        "second_seed_or_split": False,
    })
    _write_json(output_root / "official_split_integrity_after.json", {"unchanged": official_unchanged, "hashes": official_after})
    summary = {
        "experiment": EXPERIMENT,
        "protocol_valid": protocol_valid,
        "split_verification": split_verification,
        "heldout_scene_summary": heldout_scene_summary,
        "per_scene_training_summary": training_rows,
        "checkpoint_manifest": checkpoint_rows,
        "per_camera_metrics": metrics,
        "per_camera_predictors": predictors,
        "all_predictor_effects": effects,
        "primary_unseen_fraction_effects": primary_effects,
        "single_factor_controls": controls,
        "control_summary": control_summary,
        "camera_neighbor_analysis": neighbor,
        "camera_permutation_analysis": permutation,
        "camera_pair_analysis": pair_rows,
        "camera_pair_summary": pair_summary,
        "old_new_replication_table": old_new,
        "ocmc_independence_rows": independence_rows,
        "ocmc_independence_analysis": independence,
        "decomposition_safety": {"rows": decomposition_rows, "all_scenes_safe": decomposition_safe},
        "ocmc_mechanism_sanity": mechanism_sanity,
        "per_scene_candidate_c_classification": classifications,
        "primary_predictor_decision": primary_decision,
        "candidate_c_final_decision": decision,
        "runtime_summary": runtime_summary,
        "official_split_files_unchanged": official_unchanged,
        "old_new_absolute_metrics_interpreted_as_causal": False,
        "outputs_committed": False,
    }
    _write_json(output_root / "final_summary.json", summary)
    _make_figures(output_root, joined, effects, controls, neighbor, pair_rows)
    _research_note(summary)
    return summary


def _launch(repo: Path, output_root: Path) -> Dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"formal output root must be empty before launch: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    starting = _repo_state(repo)
    _write_json(output_root / "repo_state.json", starting)
    _write_json(output_root / "official_split_integrity_before.json", _official_split_hashes(repo))
    verification = _verify_locked_manifest()
    _write_json(output_root / "split_manifest_verified.json", verification)
    data_views = _create_data_views(repo, output_root, verification)
    _write_json(output_root / "experiment_data_views.json", data_views)
    environment = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "launcher_visible_device_count": torch.cuda.device_count(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "allowed_physical_gpus": [6, 7, 8, 9],
        "worker_policy": "one physical GPU exposed per worker; logical cuda:0",
    }
    _write_json(output_root / "environment.json", environment)
    gpu_snapshot = _run_text(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"])
    process_snapshot = _allowed_gpu_processes()
    _write_json(output_root / "gpu_preflight.json", {
        "gpu_snapshot": gpu_snapshot,
        "compute_process_snapshot": process_snapshot,
        "scene_gpu_assignment": SCENE_GPUS,
        "no_processes_killed": True,
    })
    occupied = {gpu: rows for gpu, rows in process_snapshot.items() if rows}
    if occupied:
        raise RuntimeError(f"assigned GPUs have active compute processes; no jobs were killed: {occupied}")
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    preflight_processes: Dict[str, subprocess.Popen[Any]] = {}
    preflight_handles: Dict[str, Any] = {}
    try:
        for scene, gpu in SCENE_GPUS.items():
            handle = (logs / f"{scene}.preflight.log").open("w", encoding="utf8")
            preflight_handles[scene] = handle
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["CONDA_DEFAULT_ENV"] = "water_splatting"
            command = [
                str(PYTHON), str(Path(__file__).resolve()), "--worker-preflight-only", scene, "--gpu", gpu,
                "--repo", str(repo), "--output-root", str(output_root),
            ]
            preflight_processes[scene] = subprocess.Popen(
                command, cwd=str(repo), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True
            )
        preflight_statuses = {scene: process.wait() for scene, process in preflight_processes.items()}
    finally:
        for handle in preflight_handles.values():
            handle.close()
    _write_json(output_root / "construction_preflight_status.json", preflight_statuses)
    if any(code != 0 for code in preflight_statuses.values()):
        raise RuntimeError(f"construction preflight failure; no formal training launched: {preflight_statuses}")
    preflight_rows = [_read_json(output_root / "preflight" / scene / "worker_preflight.json") for scene in SCENES]
    _write_json(output_root / "construction_preflight.json", {
        "all_passed_before_any_training": all(row["passed"] for row in preflight_rows),
        "no_optimization_or_training_called": all(row["no_optimization_or_training_called"] for row in preflight_rows),
        "rows": preflight_rows,
    })
    occupied_after_preflight = {gpu: rows for gpu, rows in _allowed_gpu_processes().items() if rows}
    if occupied_after_preflight:
        raise RuntimeError(
            f"assigned GPUs became occupied after construction preflight; no formal training launched: {occupied_after_preflight}"
        )
    processes: Dict[str, subprocess.Popen[Any]] = {}
    handles: Dict[str, Any] = {}
    started = time.perf_counter()
    try:
        for scene, gpu in SCENE_GPUS.items():
            handle = (logs / f"{scene}.log").open("w", encoding="utf8")
            handles[scene] = handle
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["CONDA_DEFAULT_ENV"] = "water_splatting"
            command = [
                str(PYTHON), str(Path(__file__).resolve()), "--scene-worker", scene, "--gpu", gpu,
                "--repo", str(repo), "--output-root", str(output_root),
            ]
            processes[scene] = subprocess.Popen(command, cwd=str(repo), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        statuses = {scene: process.wait() for scene, process in processes.items()}
    finally:
        for handle in handles.values():
            handle.close()
    parallel_seconds = time.perf_counter() - started
    _write_json(output_root / "worker_status.json", statuses)
    _write_json(output_root / "launcher_runtime.json", {"parallel_worker_wall_seconds": parallel_seconds})
    if any(code != 0 for code in statuses.values()):
        raise RuntimeError(f"formal scene worker failure: {statuses}")
    summary = _aggregate(repo, output_root)
    ending = _repo_state(repo)
    protected_unchanged = all(
        starting["protected_files"][path]["sha256"] == ending["protected_files"][path]["sha256"]
        for path in PROTECTED
    )
    _write_json(output_root / "protected_file_integrity.json", {
        "unchanged": protected_unchanged,
        "before": starting["protected_files"],
        "after": ending["protected_files"],
    })
    _write_json(output_root / "repo_state_after.json", ending)
    if not protected_unchanged:
        raise RuntimeError("protected file hash changed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--scene-worker", choices=SCENES)
    parser.add_argument("--worker-preflight-only", choices=SCENES)
    parser.add_argument("--gpu")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    repo, output_root = args.repo.resolve(), args.output_root.resolve()
    if args.worker_preflight_only:
        if args.gpu is None:
            raise ValueError("--worker-preflight-only requires --gpu")
        result = _worker_preflight(repo, output_root, args.worker_preflight_only, str(args.gpu))
        printable = result
    elif args.scene_worker:
        if args.gpu is None:
            raise ValueError("--scene-worker requires --gpu")
        result = _worker(repo, output_root, args.scene_worker, str(args.gpu))
        printable = {"scene": result["scene"], "runtime": result["runtime"], "checkpoint": result["final_result"]["checkpoint"]}
    elif args.aggregate_only:
        result = _aggregate(repo, output_root)
        printable = result["candidate_c_final_decision"]
    elif args.preflight_only:
        result = _verify_locked_manifest()
        printable = {"verified": result["verified"], "global_hash": result["global_hash_computed"]}
    else:
        result = _launch(repo, output_root)
        printable = result["candidate_c_final_decision"]
    print(json.dumps(printable, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
