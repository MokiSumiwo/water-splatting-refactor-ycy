#!/usr/bin/env python3
"""Formal M1 camera-context identifiability audit on IUI3.

Phase A has one causal arm pair:

* C0 keeps the formal 22-D M1 medium MLP architecture but replaces only the
  final 3-D camera-center context feature with zero.
* C1 keeps the current formal camera-center context.

The script then runs read-only camera-context swap and identifiability audits on
the trained snapshots. Phase B is deliberately not implemented here; the gate is
decided from the Phase-A summary.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.train import _set_random_seed

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC


PW = MI.PW

SCENE = "IUI3-RedSea"
OUTPUT_DIR = Path("outputs/m1_camera_context_identifiability_iui3_20260825")
RESEARCH_NOTE = Path("research_notes/M1_CAMERA_CONTEXT_IDENTIFIABILITY_IUI3_2026-08-25.md")
BND_CONFIG = PW.BND_CONFIG
BRANCHES = ("C0", "C1")
POPULATIONS = ("GENERAL", "M_SAFE")
FINAL_NOMINAL_STEP = 15000
FINAL_ACTUAL_STEP = 14999
SNAPSHOT_STEPS = (3000, 5000, 8000, 10000, 13000, 14999)
IDENTIFIABILITY_STEPS = (5000, 10000, 14999)
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
TRAINING_RNG_SEED = 202608252
SAMPLES_PER_VIEW = 1024
SAMPLE_SEED = 20260825
SWAP_SEED = 202608253
ALT_CONTEXT_COUNT = 8
COUNTERFACTUAL_EPSILON = 0.25
RANDOM_DIRECTIONS = 8
LOG_INTERVAL = 500
QUANTILE_MAX_N = 1_000_000
EPS = 1e-12


@dataclass
class BranchState:
    branch: str
    config_path: Path
    config: Any
    pipeline: Any
    optimizers: Optimizers
    scalers: Mapping[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object_hash(value: Any) -> str:
    import pickle

    return _sha256_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


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
        raise RuntimeError("CUDA must be available for this formal task.")
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


def _release(branch: Optional[BranchState]) -> None:
    if branch is None:
        return
    try:
        del branch.pipeline
    except Exception:
        pass
    try:
        del branch.optimizers
    except Exception:
        pass
    del branch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _config_from_yaml(config_path: Path, branch: str) -> Any:
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = None
    config.load_step = None
    config.load_checkpoint = None
    config.pipeline.datamanager.load_depths = False
    model_cfg = config.pipeline.model
    model_cfg.intrinsic_color_parameterization = "bounded_sh3"
    model_cfg.rasterize_mode = "classic"
    model_cfg.medium_context_mode = "dir_xy_camera"
    model_cfg.medium_camera_context_scale = 1.0
    model_cfg.medium_camera_context_dropout = 0.0
    model_cfg.medium_camera_context_ablation = branch == "C0"
    model_cfg.b_inf_mode = "tied"
    model_cfg.infinite_water_enabled = False
    model_cfg.coarse_depth_supervision_enabled = False
    model_cfg.medium_identifiability_enabled = False
    model_cfg.medium_identifiability_weight = 0.0
    return config


def _configure_model(model: Any, branch: str) -> None:
    model.config.intrinsic_color_parameterization = "bounded_sh3"
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.medium_camera_context_scale = 1.0
    model.config.medium_camera_context_dropout = 0.0
    model.config.medium_camera_context_ablation = branch == "C0"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.coarse_depth_supervision_enabled = False
    model.config.medium_identifiability_enabled = False
    model.config.medium_identifiability_weight = 0.0


def _setup_branch(repo: Path, branch: str) -> BranchState:
    config_path = repo / BND_CONFIG
    config = _config_from_yaml(config_path, branch)
    _set_random_seed(int(config.machine.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="test")
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected Pipeline, got {type(pipeline)}")
    _configure_model(pipeline.model, branch)
    pipeline.model.step = 0
    optimizers = Optimizers(MIC._optimizer_groups(config, pipeline.model), pipeline.model.get_param_groups())
    pipeline.eval()
    return BranchState(branch=branch, config_path=config_path, config=config, pipeline=pipeline, optimizers=optimizers, scalers={})


def _train_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return PW._records(pipeline)["train"]


def _eval_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return PW._records(pipeline)["eval"]


def _topology_snapshot(branch_name: str, model: Any, abs_step: int) -> Dict[str, Any]:
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacities.detach().float())
        scale = torch.exp(model.scales.detach().float())
    return {
        "branch": branch_name,
        "absolute_step": int(abs_step),
        "gaussian_count": int(model.means.shape[0]),
        "mean_opacity": float(opacity.mean().cpu().item()),
        "p99_opacity": float(torch.quantile(opacity.reshape(-1).cpu(), 0.99).item()),
        "mean_scale": float(scale.mean().cpu().item()),
        "max_scale": float(scale.max().cpu().item()),
        "p99_scale": float(torch.quantile(scale.reshape(-1).cpu(), 0.99).item()),
    }


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _seed_all(seed: int) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return _rng_state()


def _rng_manifest(state: Mapping[str, Any]) -> Dict[str, Any]:
    out = {
        "python_rng_sha256": _object_hash(state["python"]),
        "numpy_rng_sha256": _object_hash(state["numpy"]),
        "torch_cpu_rng_sha256": hashlib.sha256(state["torch_cpu"].detach().cpu().numpy().tobytes()).hexdigest(),
        "cuda_count": len(state.get("torch_cuda", [])),
    }
    if "torch_cuda" in state:
        out["torch_cuda_rng_sha256"] = [
            hashlib.sha256(item.detach().cpu().numpy().tobytes()).hexdigest() for item in state["torch_cuda"]
        ]
    return out


def _max_abs_diff(a: Tensor, b: Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float((a.detach().float() - b.detach().float()).abs().max().cpu().item())


def _compare_tensor_dict(a: Mapping[str, Tensor], b: Mapping[str, Tensor], key_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            rows.append({key_name: name, "max_abs_diff": float("nan"), "pass": False})
            continue
        if a[name].shape != b[name].shape:
            rows.append({key_name: name, "shape_C0": list(a[name].shape), "shape_C1": list(b[name].shape), "pass": False})
            continue
        diff = _max_abs_diff(a[name], b[name]) if a[name].numel() else 0.0
        rows.append({key_name: name, "shape": list(a[name].shape), "max_abs_diff": diff, "pass": bool(diff == 0.0)})
    return rows


def _model_param_tensors(model: Any) -> Dict[str, Tensor]:
    out = {
        name: getattr(model, name).detach().cpu().clone()
        for name in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")
    }
    out["medium_mlp"] = MIC._flatten_module_params(model.medium_mlp)
    out["direction_encoding"] = MIC._flatten_module_params(model.direction_encoding)
    return out


def _start_state_audit(repo: Path, output_dir: Path, training_rng: Mapping[str, Any]) -> Dict[str, Any]:
    c0: Optional[BranchState] = None
    c1: Optional[BranchState] = None
    try:
        c0 = _setup_branch(repo, "C0")
        c1 = _setup_branch(repo, "C1")
        param_rows = _compare_tensor_dict(_model_param_tensors(c0.pipeline.model), _model_param_tensors(c1.pipeline.model), "parameter_group")
        opt_rows = MIC._compare_nested_state_dict(
            MIC._optimizer_state_tensors(c0.optimizers),
            MIC._optimizer_state_tensors(c1.optimizers),
            "optimizer_group",
        )
        sched_rows = _compare_tensor_dict(
            MIC._scheduler_state_tensors(c0.optimizers),
            MIC._scheduler_state_tensors(c1.optimizers),
            "scheduler_group",
        )
        rng_a = _rng_manifest(training_rng)
        _set_rng_state(training_rng)
        rng_b = _rng_manifest(_rng_state())
        rng_rows = [
            {"quantity": key, "C0_planned_hash": value, "C1_after_reset_hash": rng_b.get(key), "pass": value == rng_b.get(key)}
            for key, value in rng_a.items()
            if not isinstance(value, list)
        ]
        if "torch_cuda_rng_sha256" in rng_a:
            rng_rows.append(
                {
                    "quantity": "torch_cuda_rng_sha256",
                    "C0_planned_hash": rng_a["torch_cuda_rng_sha256"],
                    "C1_after_reset_hash": rng_b.get("torch_cuda_rng_sha256"),
                    "pass": rng_a["torch_cuda_rng_sha256"] == rng_b.get("torch_cuda_rng_sha256"),
                }
            )
        all_pass = all(row["pass"] for row in param_rows + opt_rows + sched_rows + rng_rows)
        payload = {
            "PARAMETER_STATE_EQUIVALENCE": all(row["pass"] for row in param_rows),
            "OPTIMIZER_STATE_EQUIVALENCE": all(row["pass"] for row in opt_rows),
            "SCHEDULER_STATE_EQUIVALENCE": all(row["pass"] for row in sched_rows),
            "RNG_EQUIVALENCE": all(row["pass"] for row in rng_rows),
            "START_STATE_EQUIVALENCE_FOR_ALLOWED_CATEGORIES": bool(all_pass),
            "forward_equivalence_required": False,
            "forward_equivalence_reason": "Camera-context input is the experimental intervention, so C0/C1 forward outputs may differ.",
        }
        _write_csv(output_dir / "start_state_parameter_equivalence.csv", param_rows)
        _write_json(output_dir / "start_state_parameter_equivalence.json", {"rows": param_rows})
        _write_csv(output_dir / "start_state_optimizer_equivalence.csv", opt_rows)
        _write_json(output_dir / "start_state_optimizer_equivalence.json", {"rows": opt_rows})
        _write_csv(output_dir / "start_state_scheduler_equivalence.csv", sched_rows)
        _write_json(output_dir / "start_state_scheduler_equivalence.json", {"rows": sched_rows})
        _write_csv(output_dir / "start_state_rng_equivalence.csv", rng_rows)
        _write_json(output_dir / "start_state_rng_equivalence.json", {"rows": rng_rows})
        _write_json(output_dir / "start_state_audit.json", payload)
        if not all_pass:
            raise RuntimeError("Start-state equivalence failed for parameters/optimizer/scheduler/RNG.")
        return payload
    finally:
        _release(c0)
        _release(c1)


def _camera_context_for(model: Any, camera: Cameras, *, neutral: bool = False) -> Tensor:
    device = model.device
    dtype = camera.camera_to_worlds.dtype
    if neutral:
        return torch.zeros(3, device=device, dtype=dtype)
    scene_center, scene_scale = model._get_scene_normalization(dtype=dtype, device=device)
    camera_feature = camera.camera_to_worlds[0, :3, 3].to(device=device, dtype=dtype)
    camera_feature = (camera_feature - scene_center) / (scene_scale + 1e-6)
    return camera_feature * float(getattr(model.config, "medium_camera_context_scale", 1.0))


def _medium_raw_for_camera(
    model: Any,
    camera: Cameras,
    *,
    camera_context_override: Optional[Tensor] = None,
    force_real_camera_context: bool = False,
) -> Tuple[Tensor, int, int, Dict[str, Tensor]]:
    camera, rotation, cx, cy, height, width = MI._camera_geometry(model, camera)
    dtype = rotation.dtype
    device = rotation.device
    y = torch.linspace(0.0, height, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, width, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = (yy - cy) / camera.fy.item()
    xx = (xx - cx) / camera.fx.item()
    directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1e-12)
    directions = directions @ rotation.T
    directions_encoded = model.direction_encoding(directions.reshape(-1, 3))
    image_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    image_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    image_yy, image_xx = torch.meshgrid(image_y, image_x, indexing="ij")
    radius = torch.sqrt(image_xx.square() + image_yy.square())
    xy_context = torch.stack([image_xx, image_yy, radius], dim=-1).reshape(-1, 3)
    mode = getattr(model.config, "medium_context_mode", "dir_only")
    if mode != "dir_xy_camera":
        raise ValueError(f"This formal audit expects dir_xy_camera, got {mode}")
    if camera_context_override is not None:
        camera_context = camera_context_override.to(device=device, dtype=dtype).reshape(1, 3).expand(height * width, 3)
    else:
        neutral = bool(getattr(model.config, "medium_camera_context_ablation", False)) and not force_real_camera_context
        camera_context = _camera_context_for(model, camera, neutral=neutral).reshape(1, 3).expand(height * width, 3)
    mlp_input = torch.cat([directions_encoded, xy_context, camera_context], dim=-1).contiguous()
    if model.config.mlp_type == "tcnn":
        raw = model.medium_mlp(mlp_input)
    else:
        raw = model.medium_mlp(mlp_input.float())
    features = {
        "directions_encoded": directions_encoded.detach(),
        "xy_context": xy_context.detach(),
        "camera_context": camera_context.detach(),
        "mlp_input": mlp_input.detach(),
    }
    return raw, height, width, features


def _activate_medium(model: Any, raw: Tensor, height: int, width: int) -> Dict[str, Tensor]:
    return MI._activate_medium(model, raw, height, width)


def _generate_camera_sequence(branch: BranchState, output_dir: Path, final_step: int) -> Tuple[List[int], List[str]]:
    dm = branch.pipeline.datamanager
    filenames = list(getattr(dm.train_dataset, "image_filenames", []))
    names = [Path(path).stem for path in filenames]
    rows: List[Dict[str, Any]] = []
    indices: List[int] = []
    view_ids: List[str] = []
    for abs_step in range(0, final_step + 1):
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        index = int(dm.train_unseen_cameras.pop(0))
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        view_id = names[index]
        indices.append(index)
        view_ids.append(view_id)
        rows.append({"absolute_step": abs_step, "camera_index": index, "camera_name": view_id})
    encoded = json.dumps(rows, sort_keys=True).encode("utf8")
    _write_csv(output_dir / "paired_camera_sequence.csv", rows)
    _write_json(output_dir / "paired_camera_sequence.json", {"rows": rows, "length": len(rows)})
    audit = {
        "CAMERA_SEQUENCE_MATCH": True,
        "CAMERA_SEQUENCE_EXACT_MATCH": True,
        "length": len(rows),
        "mismatch_count": 0,
        "sha256": _sha256_bytes(encoded),
        "basis": "Both arms consume this explicit camera_index list; datamanager stochastic sampling is not used inside branch training.",
    }
    _write_json(output_dir / "camera_sequence_audit.json", audit)
    return indices, view_ids


def _save_checkpoint(branch: BranchState, abs_step: int, output_dir: Path) -> Path:
    path = output_dir / "checkpoints" / branch.branch / f"step-{abs_step:09d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": "M1-CAMERA-CONTEXT-IDENTIFIABILITY-AUDIT",
            "branch": branch.branch,
            "absolute_step": int(abs_step),
            "pipeline": branch.pipeline.state_dict(),
            "optimizers": {group: opt.state_dict() for group, opt in branch.optimizers.optimizers.items()},
            "schedulers": {group: sched.state_dict() for group, sched in branch.optimizers.schedulers.items()},
            "scalers": dict(branch.scalers),
            "metadata": {
                "medium_camera_context_ablation": branch.branch == "C0",
                "only_intervention": "camera_context_zero_vs_formal_camera_center_context",
                "matched_camera_sequence": True,
                "normal_topology_enabled": True,
            },
        },
        path,
    )
    return path


def _ckpt_path(output_dir: Path, branch: str, abs_step: int) -> Path:
    return output_dir / "checkpoints" / branch / f"step-{abs_step:09d}.ckpt"


def _require_existing_checkpoints(output_dir: Path, steps: Sequence[int]) -> None:
    missing = [
        str(_ckpt_path(output_dir, branch, int(step)))
        for branch in BRANCHES
        for step in steps
        if not _ckpt_path(output_dir, branch, int(step)).exists()
    ]
    if missing:
        raise FileNotFoundError("analysis-only mode requires existing formal checkpoints: " + ", ".join(missing))


def _load_snapshot(branch: BranchState, output_dir: Path, abs_step: int) -> None:
    ckpt = torch.load(_ckpt_path(output_dir, branch.branch, abs_step), map_location="cpu")
    branch.pipeline.load_pipeline(ckpt["pipeline"], int(ckpt["absolute_step"]))
    for group, optimizer in branch.optimizers.optimizers.items():
        if group in ckpt["optimizers"]:
            optimizer.load_state_dict(ckpt["optimizers"][group])
    for group, scheduler in branch.optimizers.schedulers.items():
        if group in ckpt["schedulers"]:
            scheduler.load_state_dict(ckpt["schedulers"][group])
    branch.pipeline.model.step = int(ckpt["absolute_step"])
    _configure_model(branch.pipeline.model, branch.branch)
    branch.pipeline.eval()


def _train_branch(
    repo: Path,
    branch_name: str,
    *,
    camera_indices: Sequence[int],
    camera_names: Sequence[str],
    training_rng: Mapping[str, Any],
    snapshot_steps: Sequence[int],
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    branch = _setup_branch(repo, branch_name)
    _set_rng_state(training_rng)
    model = branch.pipeline.model
    dm = branch.pipeline.datamanager
    cached_train = dm.cached_train
    train_cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
    snapshot_set = set(int(x) for x in snapshot_steps)
    training_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    ckpt_rows: List[Dict[str, Any]] = []
    topology_rows: List[Dict[str, Any]] = []
    try:
        for abs_step, (camera_index, camera_name) in enumerate(zip(camera_indices, camera_names)):
            branch.pipeline.train()
            model.train()
            _configure_model(model, branch_name)
            MIC._run_before(model, branch.optimizers, abs_step)
            branch.optimizers.zero_grad_all()
            batch = _batch_to_device(cached_train[camera_index].copy(), model.device)
            camera = train_cameras[camera_index : camera_index + 1]
            lrs = MIC._optimizer_lrs(branch.optimizers)
            outputs = model.get_outputs(camera)
            components = MIC._compute_loss_components(model, outputs, batch)
            metrics: Dict[str, Tensor] = {}
            losses = model.get_loss_dict(outputs, batch, metrics)
            total_loss = sum(losses.values())
            if not bool(torch.isfinite(total_loss).detach().cpu().item()):
                raise RuntimeError(f"Non-finite loss in {branch_name} step={abs_step}")
            total_loss.backward()
            total_grad_stats = MIC._param_group_grad_stats(model)
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(abs_step)
            event = MIC._run_after(model, branch.optimizers, abs_step)
            row: Dict[str, Any] = {
                "branch": branch_name,
                "absolute_step": abs_step,
                "camera_index": int(camera_index),
                "camera_name": camera_name,
                "L_total": float(total_loss.detach().cpu().item()),
                "L_RGB": float(losses["main_loss"].detach().cpu().item()),
                "reg_l1": float(components["reg_l1"].detach().cpu().item()),
                "reg_ssim": float(components["reg_ssim"].detach().cpu().item()),
                "medium_total_grad_l2": float(total_grad_stats["medium_branch"]["grad_l2"]),
                "gaussian_count": int(model.means.shape[0]),
                "mean_opacity": float(torch.sigmoid(model.opacities.detach()).mean().cpu().item()),
                "mean_scale": float(torch.exp(model.scales.detach()).mean().cpu().item()),
                "max_scale": float(torch.exp(model.scales.detach()).max().cpu().item()),
                "stable": True,
                "medium_camera_context_ablation": branch_name == "C0",
            }
            for group, lr in lrs.items():
                row[f"lr_{group}"] = lr
            if abs_step % LOG_INTERVAL == 0 or abs_step in snapshot_set:
                training_rows.append(row)
                topology_rows.append(_topology_snapshot(branch_name, model, abs_step))
            if event.get("refinement_called"):
                event = dict(event)
                event["branch"] = branch_name
                event["absolute_step"] = abs_step
                event["camera_name"] = camera_name
                event_rows.append(event)
            if abs_step in snapshot_set:
                ckpt = _save_checkpoint(branch, abs_step, output_dir)
                ckpt_rows.append({"branch": branch_name, "absolute_step": abs_step, "checkpoint_path": str(ckpt)})
                _write_csv(output_dir / f"{branch_name.lower()}_training_log.csv", training_rows)
                _write_json(output_dir / f"{branch_name.lower()}_training_log.json", {"rows": training_rows})
                _write_csv(output_dir / f"{branch_name.lower()}_refinement_events.csv", event_rows)
                _write_json(output_dir / f"{branch_name.lower()}_refinement_events.json", {"rows": event_rows})
        return training_rows, event_rows, ckpt_rows, topology_rows
    finally:
        _release(branch)


def _metric_images(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    return MIC._metric_images(model, pred, gt)


def _render_records(pipeline: Any, records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]]) -> Dict[str, Dict[str, Tensor]]:
    return MIC._render_records(pipeline, records)


def _quantile_flat(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if flat.numel() > QUANTILE_MAX_N:
        idx = torch.linspace(0, flat.numel() - 1, QUANTILE_MAX_N, device=flat.device).long()
        flat = flat[idx]
    return float(torch.quantile(flat.cpu(), q).item())


def _decomposition_row(branch: str, abs_step: int, split: str, maps: Mapping[str, Mapping[str, Tensor]]) -> Dict[str, Any]:
    j_vals: List[Tensor] = []
    tau_vals: List[Tensor] = []
    t_vals: List[Tensor] = []
    b_inf_vals: List[Tensor] = []
    beta_b_vals: List[Tensor] = []
    beta_d_vals: List[Tensor] = []
    c_vals: List[Tensor] = []
    logit_vals: List[Tensor] = []
    for data in maps.values():
        if "clear_object_fullsh_raw" in data:
            j_vals.append(data["clear_object_fullsh_raw"].reshape(-1, 3))
        if "tau_D" in data:
            tau_vals.append(data["tau_D"].reshape(-1, 3))
        if "transmission" in data:
            t_vals.append(data["transmission"].reshape(-1, 3))
        if "b_inf" in data:
            b_inf_vals.append(data["b_inf"].reshape(-1, 3))
        if "medium_bs" in data:
            beta_b_vals.append(data["medium_bs"].reshape(-1, 3))
        if "medium_attn" in data:
            beta_d_vals.append(data["medium_attn"].reshape(-1, 3))
        visible = data.get("gaussian_visible_mask")
        if "gaussian_view_rgb" in data and isinstance(visible, Tensor):
            colors = data["gaussian_view_rgb"]
            if visible.numel() == colors.shape[0] and int(visible.bool().sum().item()) > 0:
                c_vals.append(colors[visible.bool()].reshape(-1, 3))
        if "gaussian_view_logits" in data and isinstance(visible, Tensor):
            logits = data["gaussian_view_logits"]
            if visible.numel() == logits.shape[0] and int(visible.bool().sum().item()) > 0:
                logit_vals.append(logits[visible.bool()].reshape(-1, 3))

    j = torch.cat(j_vals, dim=0) if j_vals else torch.empty(0, 3)
    tau = torch.cat(tau_vals, dim=0) if tau_vals else torch.empty(0, 3)
    trans = torch.cat(t_vals, dim=0) if t_vals else torch.empty(0, 3)
    b_inf = torch.cat(b_inf_vals, dim=0) if b_inf_vals else torch.empty(0, 3)
    beta_b = torch.cat(beta_b_vals, dim=0) if beta_b_vals else torch.empty(0, 3)
    beta_d = torch.cat(beta_d_vals, dim=0) if beta_d_vals else torch.empty(0, 3)
    colors = torch.cat(c_vals, dim=0) if c_vals else torch.empty(0, 3)
    logits_all = torch.cat(logit_vals, dim=0) if logit_vals else torch.empty(0, 3)
    row: Dict[str, Any] = {
        "branch": branch,
        "absolute_step": int(abs_step),
        "split": split,
        "J_p99": _quantile_flat(j, 0.99),
        "J_amax_p99": _quantile_flat(j.amax(dim=-1), 0.99) if j.numel() else float("nan"),
        "P_J_gt_1": float((j > 1.0).float().mean().item()) if j.numel() else float("nan"),
        "tau_p90": _quantile_flat(tau, 0.90),
        "tau_p99": _quantile_flat(tau, 0.99),
        "P_T_lt_0p1": float((trans < 0.1).float().mean().item()) if trans.numel() else float("nan"),
        "P_c_gt_0p99": float((colors > 0.99).float().mean().item()) if colors.numel() else float("nan"),
        "P_abs_s_full_gt_5": float((logits_all.abs() > 5.0).float().mean().item()) if logits_all.numel() else float("nan"),
        "visible_gaussian_color_count": int(colors.shape[0]),
    }
    row.update(_stats(b_inf, "B_inf_"))
    row.update(_stats(beta_b, "beta_B_"))
    row.update(_stats(beta_d, "beta_D_"))
    return row


def _evaluate_snapshots(repo: Path, output_dir: Path, snapshot_steps: Sequence[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    global_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        branch = _setup_branch(repo, branch_name)
        try:
            for step in snapshot_steps:
                _load_snapshot(branch, output_dir, int(step))
                for split, records in (("train", _train_records(branch.pipeline)), ("eval", _eval_records(branch.pipeline))):
                    maps = _render_records(branch.pipeline, records)
                    accum: Dict[str, List[float]] = {"PSNR": [], "SSIM": [], "LPIPS": [], "MSE": []}
                    for _idx, view_id, _camera, _batch in records:
                        metrics = _metric_images(branch.pipeline.model, maps[view_id]["pred"], maps[view_id]["gt"])
                        per_view_rows.append(
                            {
                                "branch": branch_name,
                                "absolute_step": int(step),
                                "split": split,
                                "view_id": view_id,
                                **metrics,
                            }
                        )
                        for key, value in metrics.items():
                            accum[key].append(float(value))
                    row = {"branch": branch_name, "absolute_step": int(step), "split": split, "view_count": len(accum["PSNR"])}
                    for key, vals in accum.items():
                        row[key] = float(sum(vals) / len(vals)) if vals else float("nan")
                    global_rows.append(row)
                    decomp_rows.append(_decomposition_row(branch_name, int(step), split, maps))
                    del maps
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            _release(branch)
    return global_rows, per_view_rows, decomp_rows


def _analyse_loaded_branch(branch: BranchState, samples: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model = branch.pipeline.model
    model.eval()
    records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
    store = MI._init_store()
    view_slices: Dict[str, Dict[str, Tuple[int, int]]] = {pop: {} for pop in POPULATIONS}
    per_view_elapsed: List[Dict[str, Any]] = []
    for view_ord, (view_id, sample) in enumerate(samples.items()):
        if view_id not in records:
            raise RuntimeError(f"Missing train view {view_id}")
        _idx, camera, batch = records[view_id]
        t0 = time.time()
        raw_base, height, width, _features = _medium_raw_for_camera(model, camera)
        if (height, width) != (sample.height, sample.width):
            raise RuntimeError(f"View size changed for {view_id}: {(height, width)} vs {(sample.height, sample.width)}")
        raw = raw_base.detach().clone().requires_grad_(True)
        med = _activate_medium(model, raw, height, width)
        outputs = MI._render_with_medium_override(
            model,
            camera,
            med["medium_rgb"],
            med["medium_bs"],
            med["medium_attn"],
            detach_object_state=True,
        )
        gt = PW._get_gt(model, batch, outputs["background"]).reshape(-1, 3).detach().float().cpu()
        union_flat = torch.unique(torch.cat([sample.general_flat, sample.safe_flat], dim=0)).sort().values.long()
        if union_flat.numel() == 0:
            continue
        union_dev = union_flat.to(model.device)
        pred_flat = outputs["pred_image"].reshape(-1, 3)
        grad_parts: List[Tensor] = []
        for channel in range(3):
            scalar = pred_flat[union_dev, channel].sum()
            grad_raw = torch.autograd.grad(
                scalar,
                raw,
                retain_graph=channel < 2,
                create_graph=False,
                allow_unused=False,
            )[0]
            grad_parts.append(grad_raw[union_dev].detach().float().cpu())
        grad_union = torch.stack(grad_parts, dim=1)
        MI._append_population_values(
            store,
            view_slices,
            "GENERAL",
            view_id,
            sample.general_flat,
            union_flat,
            grad_union,
            raw,
            med,
            outputs,
            gt,
        )
        MI._append_population_values(
            store,
            view_slices,
            "M_SAFE",
            view_id,
            sample.safe_flat,
            union_flat,
            grad_union,
            raw,
            med,
            outputs,
            gt,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per_view_elapsed.append({"view_id": view_id, "ordinal": view_ord, "seconds": time.time() - t0, "union_sampled_rays": int(union_flat.numel())})
        del raw_base, raw, med, outputs, grad_union, grad_parts, gt
        gc.collect()
        torch.cuda.empty_cache()
    analyses = {pop: MI._finalize_population(store[pop], view_slices[pop]) for pop in POPULATIONS}
    meta = {
        "branch": branch.branch,
        "loaded_step": int(model.step),
        "gaussian_count": int(model.means.shape[0]),
        "medium_camera_context_ablation": bool(getattr(model.config, "medium_camera_context_ablation", False)),
        "per_view_elapsed": per_view_elapsed,
        "structured_jacobian_source": "same aggregate Jacobian semantics as audit_bnd_medium_identifiability_iui3.py, with ablation-aware medium input reconstruction.",
    }
    return analyses, meta


def _run_identifiability_audit(repo: Path, output_dir: Path, steps: Sequence[int], samples: Mapping[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[Tuple[str, int, str], Dict[str, Tensor]]]:
    natural_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    weak_rows: List[Dict[str, Any]] = []
    camera_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    nvo_rows: List[Dict[str, Any]] = []
    counter_rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []
    basis: Dict[Tuple[str, int, str], Dict[str, Tensor]] = {}
    random_dirs = MI._unit_random_directions(SWAP_SEED + 17, RANDOM_DIRECTIONS)
    for branch_name in BRANCHES:
        branch = _setup_branch(repo, branch_name)
        try:
            for step in steps:
                print(f"[M1-CAMCTX] identifiability {branch_name} step={step}", flush=True)
                _load_snapshot(branch, output_dir, int(step))
                analyses, meta = _analyse_loaded_branch(branch, samples)
                meta_rows.append({"branch": branch_name, "absolute_step": int(step), **meta})
                for pop, analysis in analyses.items():
                    natural_rows.append(MI._natural_stats_rows(branch_name, int(step), int(step), pop, analysis))
                    aggregate_rows.append(MI._aggregate_rows(branch_name, int(step), int(step), pop, analysis))
                    weak_rows.append(MI._weak_mode_row(branch_name, int(step), int(step), pop, analysis))
                    camera_rows.extend(MI._camera_rows(branch_name, int(step), int(step), pop, analysis))
                    strata_rows.extend(MI._strata_rows(branch_name, int(step), int(step), pop, analysis))
                    nvo_rows.extend(MI._natural_variance_observability_rows(branch_name, int(step), int(step), pop, analysis))
                    basis[(branch_name, int(step), pop)] = {
                        "v_min": analysis.v_min.detach().double().cpu(),
                        "v_max": analysis.v_max.detach().double().cpu(),
                        "scale": analysis.scale.detach().double().cpu(),
                        "depth": analysis.depth.detach().float().cpu(),
                        "tau": analysis.tau.detach().float().cpu(),
                    }
                counter_rows.extend(_counterfactual_for_branch(branch, analyses, samples, random_dirs))
                _write_csv(output_dir / "identifiability_summary.csv", aggregate_rows)
                _write_json(output_dir / "identifiability_summary.json", {"rows": aggregate_rows})
        finally:
            _release(branch)
    counter_ratio_rows = _counterfactual_ratio_rows(counter_rows)
    outputs = {
        "natural_medium_output_statistics": natural_rows,
        "identifiability_summary": aggregate_rows,
        "weak_mode_summary": weak_rows,
        "camera_context_stability": camera_rows,
        "depth_tau_identifiability_strata": strata_rows,
        "natural_variance_vs_observability": nvo_rows,
        "counterfactual_perturbation": counter_rows,
        "counterfactual_weak_mode_sensitivity": counter_ratio_rows,
        "identifiability_checkpoint_meta": meta_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", {"rows": rows})
    return outputs, basis


def _counterfactual_for_branch(branch: BranchState, analyses: Mapping[str, Any], samples: Mapping[str, Any], random_dirs: Sequence[Tensor]) -> List[Dict[str, Any]]:
    model = branch.pipeline.model
    records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
    direction_bank: Dict[str, List[Tuple[str, Tensor]]] = {}
    for pop, analysis in analyses.items():
        direction_bank[pop] = [("v_min", analysis.v_min.detach().double().cpu()), ("v_max", analysis.v_max.detach().double().cpu())]
        for idx, direction in enumerate(random_dirs):
            direction_bank[pop].append((f"random_{idx:02d}", direction.detach().double().cpu()))
    accum: Dict[Tuple[str, str], Dict[str, float]] = {}
    for pop in POPULATIONS:
        for label, _direction in direction_bank[pop]:
            accum[(pop, label)] = {key: 0.0 for key in ("ray_count", "rgb_abs", "rgb_sq", "act_sq", "bd_sq")}
    for view_id, sample in samples.items():
        _idx, camera, batch = records[view_id]
        with torch.no_grad():
            raw_base, height, width, _features = _medium_raw_for_camera(model, camera)
            med_base = _activate_medium(model, raw_base, height, width)
            base_out = MI._render_with_medium_override(model, camera, med_base["medium_rgb"], med_base["medium_bs"], med_base["medium_attn"], detach_object_state=True)
            base_pred_flat = base_out["pred_image"].reshape(-1, 3).detach().float().cpu()
            base_act_flat = torch.cat([med_base["medium_rgb"].reshape(-1, 3), med_base["medium_bs"].reshape(-1, 3), med_base["medium_attn"].reshape(-1, 3)], dim=-1).detach().float().cpu()
            for pop in POPULATIONS:
                flat = sample.flat_for(pop)
                if flat.numel() == 0:
                    continue
                flat_dev = flat.to(model.device)
                for label, direction in direction_bank[pop]:
                    raw_pert = raw_base.detach().clone()
                    delta = analyses[pop].scale.detach().to(device=model.device, dtype=raw_pert.dtype) * float(COUNTERFACTUAL_EPSILON) * direction.to(device=model.device, dtype=raw_pert.dtype)
                    raw_pert[flat_dev] = raw_pert[flat_dev] + delta.reshape(1, 9)
                    med_pert = _activate_medium(model, raw_pert, height, width)
                    pert_out = MI._render_with_medium_override(model, camera, med_pert["medium_rgb"], med_pert["medium_bs"], med_pert["medium_attn"], detach_object_state=True)
                    pert_pred = pert_out["pred_image"].reshape(-1, 3).detach().float().cpu()[flat]
                    pert_act = torch.cat([med_pert["medium_rgb"].reshape(-1, 3), med_pert["medium_bs"].reshape(-1, 3), med_pert["medium_attn"].reshape(-1, 3)], dim=-1).detach().float().cpu()[flat]
                    rgb_delta = pert_pred - base_pred_flat[flat]
                    act_delta = pert_act - base_act_flat[flat]
                    item = accum[(pop, label)]
                    n = float(flat.numel())
                    item["ray_count"] += n
                    item["rgb_abs"] += float(rgb_delta.abs().sum().item())
                    item["rgb_sq"] += float(rgb_delta.square().sum().item())
                    item["act_sq"] += float(act_delta.square().sum().item())
                    item["bd_sq"] += float(act_delta[:, 6:9].square().sum().item())
                    del raw_pert, med_pert, pert_out
        del raw_base, med_base, base_out, base_pred_flat, base_act_flat
        gc.collect()
        torch.cuda.empty_cache()
    rows: List[Dict[str, Any]] = []
    for pop in POPULATIONS:
        for label, direction in direction_bank[pop]:
            item = accum[(pop, label)]
            n = max(item["ray_count"], 1.0)
            rows.append(
                {
                    "branch": branch.branch,
                    "absolute_step": int(model.step),
                    "population": pop,
                    "direction_label": label,
                    "epsilon_standardized": COUNTERFACTUAL_EPSILON,
                    "sampled_rays": int(item["ray_count"]),
                    "mean_abs_rendered_rgb_change": item["rgb_abs"] / max(n * 3.0, 1.0),
                    "rendered_rgb_delta_mse_vs_baseline": item["rgb_sq"] / max(n * 3.0, 1.0),
                    "rms_medium_output_change_9d": math.sqrt(item["act_sq"] / max(n * 9.0, 1.0)),
                    "rms_beta_D_change": math.sqrt(item["bd_sq"] / max(n * 3.0, 1.0)),
                    **{f"direction_{idx}": float(direction[idx].item()) for idx in range(9)},
                }
            )
    return rows


def _counterfactual_ratio_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keyed = {(row["branch"], int(row["absolute_step"]), row["population"], row["direction_label"]): row for row in rows}
    out: List[Dict[str, Any]] = []
    for branch in BRANCHES:
        steps = sorted({int(row["absolute_step"]) for row in rows if row["branch"] == branch})
        for step in steps:
            for pop in POPULATIONS:
                vmin = keyed.get((branch, step, pop, "v_min"))
                vmax = keyed.get((branch, step, pop, "v_max"))
                if not vmin or not vmax:
                    continue
                out.append(
                    {
                        "branch": branch,
                        "absolute_step": step,
                        "population": pop,
                        "vmin_mean_abs_rgb_change": float(vmin["mean_abs_rendered_rgb_change"]),
                        "vmax_mean_abs_rgb_change": float(vmax["mean_abs_rendered_rgb_change"]),
                        "vmin_over_vmax_rgb_change": float(vmin["mean_abs_rendered_rgb_change"]) / max(float(vmax["mean_abs_rendered_rgb_change"]), EPS),
                        "vmin_rms_medium_output_change_9d": float(vmin["rms_medium_output_change_9d"]),
                        "vmin_rms_beta_D_change": float(vmin["rms_beta_D_change"]),
                    }
                )
    return out


def _channel_var_row(values: Tensor, prefix: str) -> Dict[str, Any]:
    names = ("r", "g", "b")
    out: Dict[str, Any] = {}
    flat = values.detach().float().reshape(-1)
    out[f"{prefix}_variance_pooled"] = float(flat.var(unbiased=False).item()) if flat.numel() else float("nan")
    out[f"{prefix}_std_pooled"] = float(flat.std(unbiased=False).item()) if flat.numel() else float("nan")
    for idx, name in enumerate(names):
        vals = values[:, idx].detach().float() if values.numel() else torch.empty(0)
        out[f"{prefix}_{name}_variance"] = float(vals.var(unbiased=False).item()) if vals.numel() else float("nan")
        out[f"{prefix}_{name}_std"] = float(vals.std(unbiased=False).item()) if vals.numel() else float("nan")
        out[f"{prefix}_{name}_mean"] = float(vals.mean().item()) if vals.numel() else float("nan")
    return out


def _stats(values: Tensor, prefix: str) -> Dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in ("mean", "std", "p01", "p50", "p90", "p99", "max")}
    if flat.numel() > QUANTILE_MAX_N:
        idx = torch.linspace(0, flat.numel() - 1, QUANTILE_MAX_N, device=flat.device).long()
        flat = flat[idx]
    return {
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        f"{prefix}p01": float(torch.quantile(flat.cpu(), 0.01).item()),
        f"{prefix}p50": float(torch.quantile(flat.cpu(), 0.50).item()),
        f"{prefix}p90": float(torch.quantile(flat.cpu(), 0.90).item()),
        f"{prefix}p99": float(torch.quantile(flat.cpu(), 0.99).item()),
        f"{prefix}max": float(flat.max().item()),
    }


def _medium_distribution_audit(repo: Path, output_dir: Path, steps: Sequence[int], samples: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    per_camera_rows: List[Dict[str, Any]] = []
    variance_rows: List[Dict[str, Any]] = []
    between_rows: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        branch = _setup_branch(repo, branch_name)
        try:
            for step in steps:
                print(f"[M1-CAMCTX] medium distribution {branch_name} step={step}", flush=True)
                _load_snapshot(branch, output_dir, int(step))
                model = branch.pipeline.model
                records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
                accum: Dict[str, Dict[str, List[Tensor]]] = {pop: {"raw": [], "act": [], "depth": [], "tau": [], "trans": [], "rgb_residual": []} for pop in POPULATIONS}
                for view_id, sample in samples.items():
                    _idx, camera, batch = records[view_id]
                    with torch.no_grad():
                        raw, height, width, features = _medium_raw_for_camera(model, camera)
                        med = _activate_medium(model, raw, height, width)
                        outputs = MI._render_with_medium_override(model, camera, med["medium_rgb"], med["medium_bs"], med["medium_attn"], detach_object_state=True)
                        gt = PW._get_gt(model, batch, outputs["background"]).detach().float()
                    for pop in POPULATIONS:
                        flat = sample.flat_for(pop)
                        if flat.numel() == 0:
                            continue
                        flat_dev = flat.to(model.device)
                        raw_s = raw.reshape(-1, 9)[flat_dev].detach().float().cpu()
                        act_s = torch.cat([med["medium_rgb"].reshape(-1, 3)[flat_dev], med["medium_bs"].reshape(-1, 3)[flat_dev], med["medium_attn"].reshape(-1, 3)[flat_dev]], dim=-1).detach().float().cpu()
                        pred = outputs["pred_image"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        target = gt.reshape(-1, 3)[flat_dev].detach().float().cpu()
                        residual = pred - target
                        depth = outputs["depth"].reshape(-1, 1)[flat_dev].detach().float().cpu()
                        tau = outputs["tau_D"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        trans = outputs["transmission"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        accum[pop]["raw"].append(raw_s)
                        accum[pop]["act"].append(act_s)
                        accum[pop]["depth"].append(depth)
                        accum[pop]["tau"].append(tau)
                        accum[pop]["trans"].append(trans)
                        accum[pop]["rgb_residual"].append(residual)
                        row: Dict[str, Any] = {
                            "branch": branch_name,
                            "absolute_step": int(step),
                            "population": pop,
                            "view_id": view_id,
                            "sampled_rays": int(flat.numel()),
                            "camera_context_x": float(features["camera_context"][0, 0].detach().cpu().item()),
                            "camera_context_y": float(features["camera_context"][0, 1].detach().cpu().item()),
                            "camera_context_z": float(features["camera_context"][0, 2].detach().cpu().item()),
                            "rgb_mse": float(residual.square().mean().item()),
                        }
                        names = ("Binf_r", "Binf_g", "Binf_b", "betaB_r", "betaB_g", "betaB_b", "betaD_r", "betaD_g", "betaD_b")
                        for idx_name, name in enumerate(names):
                            row[f"raw_{name}_mean"] = float(raw_s[:, idx_name].mean().item())
                            row[f"raw_{name}_variance"] = float(raw_s[:, idx_name].var(unbiased=False).item())
                            row[f"activated_{name}_mean"] = float(act_s[:, idx_name].mean().item())
                            row[f"activated_{name}_variance"] = float(act_s[:, idx_name].var(unbiased=False).item())
                        per_camera_rows.append(row)
                for pop in POPULATIONS:
                    raw_all = torch.cat(accum[pop]["raw"], dim=0)
                    act_all = torch.cat(accum[pop]["act"], dim=0)
                    tau_all = torch.cat(accum[pop]["tau"], dim=0)
                    trans_all = torch.cat(accum[pop]["trans"], dim=0)
                    residual_all = torch.cat(accum[pop]["rgb_residual"], dim=0)
                    row = {"branch": branch_name, "absolute_step": int(step), "population": pop, "sampled_rays": int(raw_all.shape[0]), "rgb_mse": float(residual_all.square().mean().item())}
                    row.update(_channel_var_row(raw_all[:, 0:3], "raw_Binf"))
                    row.update(_channel_var_row(raw_all[:, 3:6], "raw_betaB"))
                    row.update(_channel_var_row(raw_all[:, 6:9], "raw_betaD"))
                    row.update(_channel_var_row(act_all[:, 0:3], "B_inf"))
                    row.update(_channel_var_row(act_all[:, 3:6], "beta_B"))
                    row.update(_channel_var_row(act_all[:, 6:9], "beta_D"))
                    row.update(_stats(tau_all, "tau_"))
                    row.update(_stats(trans_all, "T_"))
                    variance_rows.append(row)
        finally:
            _release(branch)

    grouped: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = {}
    for row in per_camera_rows:
        grouped.setdefault((str(row["branch"]), int(row["absolute_step"]), str(row["population"])), []).append(row)
    for (branch, step, pop), rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        item: Dict[str, Any] = {"branch": branch, "absolute_step": int(step), "population": pop, "camera_count": len(rows)}
        for prefix in ("raw", "activated"):
            for name in ("Binf_r", "Binf_g", "Binf_b", "betaB_r", "betaB_g", "betaB_b", "betaD_r", "betaD_g", "betaD_b"):
                means = torch.tensor([float(r[f"{prefix}_{name}_mean"]) for r in rows], dtype=torch.float64)
                within = torch.tensor([float(r[f"{prefix}_{name}_variance"]) for r in rows], dtype=torch.float64)
                across = float(means.var(unbiased=False).item()) if means.numel() else float("nan")
                within_mean = float(within.mean().item()) if within.numel() else float("nan")
                item[f"{prefix}_{name}_between_camera_variance"] = across
                item[f"{prefix}_{name}_within_camera_variance"] = within_mean
                item[f"{prefix}_{name}_between_over_within"] = across / max(within_mean, EPS)
        between_rows.append(item)
    outputs = {
        "per_camera_medium_statistics": per_camera_rows,
        "medium_output_statistics": variance_rows,
        "between_within_camera_variance": between_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", {"rows": rows})
    return outputs


def _delta_energy(delta: Tensor, prefix: str) -> Dict[str, Any]:
    d = delta.detach().float().reshape(-1, 9)
    sq = d.square()
    total = float(sq.sum().item())
    total = max(total, EPS)
    out = {
        f"{prefix}_rms_9d": float(torch.sqrt(sq.mean()).item()) if sq.numel() else float("nan"),
        f"{prefix}_B_inf_energy_fraction": float(sq[:, 0:3].sum().item() / total),
        f"{prefix}_beta_B_energy_fraction": float(sq[:, 3:6].sum().item() / total),
        f"{prefix}_beta_D_energy_fraction": float(sq[:, 6:9].sum().item() / total),
    }
    labels = ("Binf_r", "Binf_g", "Binf_b", "betaB_r", "betaB_g", "betaB_b", "betaD_r", "betaD_g", "betaD_b")
    for idx, label in enumerate(labels):
        out[f"{prefix}_{label}_energy_fraction"] = float(sq[:, idx].sum().item() / total)
    return out


def _projection_stats(delta_raw: Tensor, scale: Tensor, v_min: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    delta_std = delta_raw.detach().double() / scale.reshape(1, 9).clamp_min(EPS)
    v = v_min.detach().double().reshape(9)
    v = v / v.norm().clamp_min(EPS)
    coeff = delta_std @ v
    weak_std = coeff[:, None] * v.reshape(1, 9)
    orth_std = delta_std - weak_std
    frac = coeff.square() / delta_std.square().sum(dim=-1).clamp_min(EPS)
    stats = {
        "weak_energy_fraction_mean": float(frac.mean().item()),
        "weak_energy_fraction_median": float(torch.quantile(frac.float().cpu(), 0.5).item()),
        "weak_projection_over_random_1over9": float(frac.mean().item() / (1.0 / 9.0)),
        "delta_std_rms": float(torch.sqrt(delta_std.square().mean()).item()),
        "weak_std_rms": float(torch.sqrt(weak_std.square().mean()).item()),
        "orth_std_rms": float(torch.sqrt(orth_std.square().mean()).item()),
    }
    return delta_std, weak_std, orth_std, stats


def _render_from_raw(model: Any, camera: Cameras, raw: Tensor, height: int, width: int) -> Dict[str, Tensor]:
    med = _activate_medium(model, raw, height, width)
    return MI._render_with_medium_override(model, camera, med["medium_rgb"], med["medium_bs"], med["medium_attn"], detach_object_state=True)


def _camera_swap_and_weak_component_audit(
    repo: Path,
    output_dir: Path,
    steps: Sequence[int],
    samples: Mapping[str, Any],
    basis: Mapping[Tuple[str, int, str], Mapping[str, Tensor]],
) -> Dict[str, List[Dict[str, Any]]]:
    utility_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    projection_rows: List[Dict[str, Any]] = []
    sensitivity_rows: List[Dict[str, Any]] = []
    removal_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    rng = random.Random(SWAP_SEED)
    branch = _setup_branch(repo, "C1")
    try:
        train_records = _train_records(branch.pipeline)
        train_view_ids = [view_id for _idx, view_id, _cam, _batch in train_records]
        alt_map: Dict[str, List[str]] = {}
        for view_id in train_view_ids:
            candidates = [v for v in train_view_ids if v != view_id]
            local = candidates[:]
            rng.shuffle(local)
            alt_map[view_id] = local[:ALT_CONTEXT_COUNT]
        _write_json(output_dir / "real_camera_swap_bank.json", {"seed": SWAP_SEED, "alternatives_per_source": ALT_CONTEXT_COUNT, "rows": alt_map})

        for step in steps:
            print(f"[M1-CAMCTX] camera swap C1 step={step}", flush=True)
            _load_snapshot(branch, output_dir, int(step))
            model = branch.pipeline.model
            records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
            context_bank = {
                view_id: _camera_context_for(model, camera.to(model.device), neutral=False).detach()
                for view_id, (_idx, camera, _batch) in records.items()
            }
            for source_view, sample in samples.items():
                _idx, camera, batch = records[source_view]
                camera = camera.to(model.device)
                with torch.no_grad():
                    raw_correct, height, width, _features = _medium_raw_for_camera(model, camera)
                    out_correct = _render_from_raw(model, camera, raw_correct, height, width)
                    gt_correct = PW._get_gt(model, batch, out_correct["background"]).reshape(-1, 3).detach().float().cpu()
                    pred_correct = out_correct["pred_image"].reshape(-1, 3).detach().float().cpu()
                    depth_correct = out_correct["depth"].reshape(-1, 1).detach().float().cpu()
                    tau_correct = out_correct["tau_D"].reshape(-1, 3).detach().float().cpu().mean(dim=-1, keepdim=True)
                raw_ref_sum = torch.zeros_like(raw_correct.detach())
                ref_count = 0
                for alt_view in alt_map[source_view]:
                    with torch.no_grad():
                        raw_swap, _h2, _w2, _ = _medium_raw_for_camera(model, camera, camera_context_override=context_bank[alt_view])
                        raw_ref_sum += raw_swap.detach()
                        ref_count += 1
                        out_swap = _render_from_raw(model, camera, raw_swap, height, width)
                        pred_swap = out_swap["pred_image"].reshape(-1, 3).detach().float().cpu()
                        gt_swap = PW._get_gt(model, batch, out_swap["background"]).reshape(-1, 3).detach().float().cpu()
                    for pop in POPULATIONS:
                        flat = sample.flat_for(pop)
                        if flat.numel() == 0:
                            continue
                        flat_dev = flat.to(model.device)
                        err_correct = (pred_correct[flat] - gt_correct[flat]).square().mean(dim=-1)
                        err_swap = (pred_swap[flat] - gt_swap[flat]).square().mean(dim=-1)
                        delta_e = err_swap - err_correct
                        raw_delta = raw_correct.reshape(-1, 9)[flat_dev].detach().float().cpu() - raw_swap.reshape(-1, 9)[flat_dev].detach().float().cpu()
                        med_c = _activate_medium(model, raw_correct, height, width)
                        med_s = _activate_medium(model, raw_swap, height, width)
                        act_delta = torch.cat(
                            [
                                med_c["medium_rgb"].reshape(-1, 3)[flat_dev] - med_s["medium_rgb"].reshape(-1, 3)[flat_dev],
                                med_c["medium_bs"].reshape(-1, 3)[flat_dev] - med_s["medium_bs"].reshape(-1, 3)[flat_dev],
                                med_c["medium_attn"].reshape(-1, 3)[flat_dev] - med_s["medium_attn"].reshape(-1, 3)[flat_dev],
                            ],
                            dim=-1,
                        ).detach().float().cpu()
                        v_min = basis[("C1", int(step), pop)]["v_min"]
                        scale = basis[("C1", int(step), pop)]["scale"]
                        _delta_std, _weak_std, _orth_std, proj = _projection_stats(raw_delta, scale, v_min)
                        utility_rows.append(
                            {
                                "branch": "C1",
                                "absolute_step": int(step),
                                "population": pop,
                                "source_view_id": source_view,
                                "swapped_view_id": alt_view,
                                "sampled_rays": int(flat.numel()),
                                "E_correct_mean": float(err_correct.mean().item()),
                                "E_swap_mean": float(err_swap.mean().item()),
                                "Delta_E_swap_mean": float(delta_e.mean().item()),
                                "Delta_E_swap_median": float(torch.quantile(delta_e.float(), 0.5).item()),
                                "fraction_Delta_E_swap_gt_0": float((delta_e > 0).float().mean().item()),
                                "rgb_change_mean_abs": float((pred_swap[flat] - pred_correct[flat]).abs().mean().item()),
                            }
                        )
                        drow = {"branch": "C1", "absolute_step": int(step), "population": pop, "source_view_id": source_view, "swapped_view_id": alt_view, "sampled_rays": int(flat.numel())}
                        drow.update(_delta_energy(raw_delta, "raw_delta_z_cam"))
                        drow.update(_delta_energy(act_delta, "activated_delta_cam"))
                        delta_rows.append(drow)
                        projection_rows.append({"branch": "C1", "absolute_step": int(step), "population": pop, "source_view_id": source_view, "swapped_view_id": alt_view, **proj})

                raw_ref = raw_ref_sum / max(ref_count, 1)
                with torch.no_grad():
                    out_ref = _render_from_raw(model, camera, raw_ref, height, width)
                    pred_ref = out_ref["pred_image"].reshape(-1, 3).detach().float().cpu()
                    gt_ref = PW._get_gt(model, batch, out_ref["background"]).reshape(-1, 3).detach().float().cpu()
                for pop in POPULATIONS:
                    flat = sample.flat_for(pop)
                    if flat.numel() == 0:
                        continue
                    flat_dev = flat.to(model.device)
                    v_min = basis[("C1", int(step), pop)]["v_min"]
                    scale = basis[("C1", int(step), pop)]["scale"].to(dtype=torch.float32)
                    raw_delta = raw_correct.reshape(-1, 9)[flat_dev].detach().float().cpu() - raw_ref.reshape(-1, 9)[flat_dev].detach().float().cpu()
                    delta_std, weak_std, orth_std, proj = _projection_stats(raw_delta, scale.double(), v_min)
                    target_rms = float(torch.sqrt(delta_std.square().mean()).item())
                    weak_rms = float(torch.sqrt(weak_std.square().mean()).item())
                    orth_rms = float(torch.sqrt(orth_std.square().mean()).item())
                    weak_gain = target_rms / max(weak_rms, EPS)
                    orth_gain = target_rms / max(orth_rms, EPS)
                    raw_ref_flat = raw_ref.reshape(-1, 9).detach().clone()
                    weak_map = raw_ref_flat.clone()
                    orth_map = raw_ref_flat.clone()
                    removed_map = raw_correct.reshape(-1, 9).detach().clone()
                    map_dtype = raw_ref_flat.dtype
                    scale_dev = scale.to(device=model.device, dtype=map_dtype).reshape(1, 9)
                    weak_component = weak_std.to(device=model.device, dtype=map_dtype) * scale_dev
                    orth_component = orth_std.to(device=model.device, dtype=map_dtype) * scale_dev
                    weak_map[flat_dev] = raw_ref_flat[flat_dev] + weak_component * weak_gain
                    orth_map[flat_dev] = raw_ref_flat[flat_dev] + orth_component * orth_gain
                    removed_map[flat_dev] = raw_ref_flat[flat_dev] + orth_component
                    weak_map = weak_map.view(height, width, 9)
                    orth_map = orth_map.view(height, width, 9)
                    removed_map = removed_map.view(height, width, 9)
                    with torch.no_grad():
                        out_weak = _render_from_raw(model, camera, weak_map, height, width)
                        out_orth = _render_from_raw(model, camera, orth_map, height, width)
                        out_removed = _render_from_raw(model, camera, removed_map, height, width)
                    weak_pred = out_weak["pred_image"].reshape(-1, 3).detach().float().cpu()[flat]
                    orth_pred = out_orth["pred_image"].reshape(-1, 3).detach().float().cpu()[flat]
                    removed_pred = out_removed["pred_image"].reshape(-1, 3).detach().float().cpu()[flat]
                    ref_pred = pred_ref[flat]
                    correct_pred = pred_correct[flat]
                    target = gt_correct[flat]
                    weak_rgb = (weak_pred - ref_pred).abs().mean()
                    orth_rgb = (orth_pred - ref_pred).abs().mean()
                    sensitivity_rows.append(
                        {
                            "branch": "C1",
                            "absolute_step": int(step),
                            "population": pop,
                            "source_view_id": source_view,
                            "reference": "mean_of_8_real_swapped_camera_contexts",
                            "sampled_rays": int(flat.numel()),
                            "target_delta_std_rms": target_rms,
                            "weak_component_gain_for_matched_rms": weak_gain,
                            "orth_component_gain_for_matched_rms": orth_gain,
                            "weak_component_rgb_change_mean_abs": float(weak_rgb.item()),
                            "orth_component_rgb_change_mean_abs": float(orth_rgb.item()),
                            "weak_over_orth_rgb_change": float(weak_rgb.item()) / max(float(orth_rgb.item()), EPS),
                            **proj,
                        }
                    )
                    err_correct = (correct_pred - target).square().mean(dim=-1)
                    err_removed = (removed_pred - target).square().mean(dim=-1)
                    removal_rows.append(
                        {
                            "branch": "C1",
                            "absolute_step": int(step),
                            "population": pop,
                            "source_view_id": source_view,
                            "reference": "mean_of_8_real_swapped_camera_contexts",
                            "sampled_rays": int(flat.numel()),
                            "E_correct_mean": float(err_correct.mean().item()),
                            "E_remove_weak_mean": float(err_removed.mean().item()),
                            "Delta_E_remove_weak_mean": float((err_removed - err_correct).mean().item()),
                            "Delta_E_remove_weak_median": float(torch.quantile((err_removed - err_correct).float(), 0.5).item()),
                            "fraction_remove_weak_improves_or_equal": float((err_removed <= err_correct).float().mean().item()),
                        }
                    )
                    depth_vals = depth_correct[flat, 0]
                    tau_vals = tau_correct[flat, 0]
                    for basis_name, values in (("depth", depth_vals), ("tau", tau_vals)):
                        q1 = torch.quantile(values.float(), 1.0 / 3.0)
                        q2 = torch.quantile(values.float(), 2.0 / 3.0)
                        for stratum, mask in (
                            ("near_or_low", values <= q1),
                            ("middle", (values > q1) & (values <= q2)),
                            ("far_or_high", values > q2),
                        ):
                            if int(mask.sum().item()) == 0:
                                continue
                            strata_rows.append(
                                {
                                    "branch": "C1",
                                    "absolute_step": int(step),
                                    "population": pop,
                                    "source_view_id": source_view,
                                    "stratification_basis": basis_name,
                                    "stratum": stratum,
                                    "sampled_rays": int(mask.sum().item()),
                                    "camera_delta_std_rms": float(torch.sqrt(delta_std.float()[mask].square().mean()).item()),
                                    "weak_energy_fraction_mean": float(((delta_std.float()[mask] @ v_min.float()).square() / delta_std.float()[mask].square().sum(dim=-1).clamp_min(EPS)).mean().item()),
                                    "weak_component_rgb_change_mean_abs": float((weak_pred[mask] - ref_pred[mask]).abs().mean().item()),
                                    "orth_component_rgb_change_mean_abs": float((orth_pred[mask] - ref_pred[mask]).abs().mean().item()),
                                    "Delta_E_remove_weak_mean": float((err_removed[mask] - err_correct[mask]).mean().item()),
                                }
                            )
                    del out_weak, out_orth, out_removed
                del raw_correct, raw_ref, raw_ref_sum, out_correct, pred_correct, gt_correct, pred_ref, gt_ref
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        _release(branch)
    outputs = {
        "correct_vs_swapped_context_utility": utility_rows,
        "camera_induced_medium_delta": delta_rows,
        "camera_delta_weak_projection": projection_rows,
        "weak_component_rgb_sensitivity": sensitivity_rows,
        "weak_component_removal_counterfactual": removal_rows,
        "depth_tau_camera_context_strata": strata_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", {"rows": rows})
    per_camera = _per_camera_stability(utility_rows, delta_rows, projection_rows, sensitivity_rows, removal_rows)
    _write_csv(output_dir / "per_camera_stability.csv", per_camera)
    _write_json(output_dir / "per_camera_stability.json", {"rows": per_camera})
    outputs["per_camera_stability"] = per_camera
    return outputs


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = sorted(float(row[key]) for row in rows if key in row and math.isfinite(float(row[key])))
    if not vals:
        return float("nan")
    return vals[len(vals) // 2]


def _per_camera_stability(*row_groups: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str, str], Dict[str, List[Mapping[str, Any]]]] = {}
    names = ("utility", "delta", "projection", "sensitivity", "removal")
    for name, rows in zip(names, row_groups):
        for row in rows:
            key = (int(row["absolute_step"]), str(row["population"]), str(row["source_view_id"]))
            grouped.setdefault(key, {n: [] for n in names})[name].append(row)
    out: List[Dict[str, Any]] = []
    for (step, pop, view), groups in sorted(grouped.items()):
        item = {"absolute_step": step, "population": pop, "source_view_id": view}
        item["Delta_E_swap_mean"] = _mean(groups["utility"], "Delta_E_swap_mean")
        item["Delta_E_swap_median"] = _median(groups["utility"], "Delta_E_swap_mean")
        item["raw_delta_z_cam_rms_9d_mean"] = _mean(groups["delta"], "raw_delta_z_cam_rms_9d")
        item["raw_delta_beta_D_energy_fraction_mean"] = _mean(groups["delta"], "raw_delta_z_cam_beta_D_energy_fraction")
        item["weak_energy_fraction_mean"] = _mean(groups["projection"], "weak_energy_fraction_mean")
        item["weak_over_orth_rgb_change_mean"] = _mean(groups["sensitivity"], "weak_over_orth_rgb_change")
        item["Delta_E_remove_weak_mean"] = _mean(groups["removal"], "Delta_E_remove_weak_mean")
        item["remove_weak_improves_fraction_mean"] = _mean(groups["removal"], "fraction_remove_weak_improves_or_equal")
        out.append(item)
    return out


def _delta_rows(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str], metrics: Sequence[str], branch_field: str = "branch") -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped.setdefault(key, {})[str(row[branch_field])] = row
    out: List[Dict[str, Any]] = []
    for key, branches in sorted(grouped.items(), key=lambda kv: kv[0]):
        if "C0" not in branches or "C1" not in branches:
            continue
        item = {field: value for field, value in zip(key_fields, key)}
        for metric in metrics:
            try:
                c0 = float(branches["C0"][metric])
                c1 = float(branches["C1"][metric])
            except Exception:
                continue
            item[f"C0_{metric}"] = c0
            item[f"C1_{metric}"] = c1
            item[f"delta_C1_minus_C0_{metric}"] = c1 - c0
            item[f"ratio_C1_over_C0_{metric}"] = c1 / max(abs(c0), EPS)
        out.append(item)
    return out


def _source_semantics(repo: Path, output_dir: Path) -> Dict[str, Any]:
    audit = {
        "CODE_FACT": True,
        "camera_context_name": "camera context",
        "not_learned_latent": True,
        "definition": "(camera.camera_to_worlds[0,:3,3] - scene_center) / (scene_scale + 1e-6) * medium_camera_context_scale",
        "dimensions": 3,
        "trainable": False,
        "scene_specific": True,
        "normalization": "scene-box normalization from WaterSplattingModel._get_scene_normalization",
        "zero_reference": "scene-normalized scene center; used as neutral constant for C0 ablation",
        "train_eval_construction": "constructed from each Cameras camera_to_worlds translation in WaterSplattingModel._predict_medium at train/eval/render time",
        "per_image_unique": "unique when camera centers differ; repeated over all pixels in that image",
        "input_order": "16-D SH direction encoding, 3-D XY/r context, 3-D camera context",
        "medium_raw_output": "z_med[0:3]->sigmoid B_inf/medium_rgb, z_med[3:6]->softplus beta_B, z_med[6:9]->softplus beta_D",
        "source_files": [
            "water_splatting/fields/medium_field.py",
            "water_splatting/water_splatting.py",
        ],
    }
    _write_json(output_dir / "exact_camera_context_semantics.json", audit)
    return audit


def _historical_m1_evidence(repo: Path, output_dir: Path) -> Dict[str, Any]:
    evidence = {
        "git_log_hits": _git(repo, "log", "--all", "--oneline", "--grep=M1\\|medium context\\|dir_xy_camera\\|camera context"),
        "introduced_by_blame": {
            "medium_field_append_context": "72927e7 Refactor WaterSplatting and add M1-M4 ablations",
            "formal_clean_bnd_baseline": "62294d6 Build clean M1 bounded-intrinsic baseline",
            "previous_medctx_audit": "4f9cffa Audit bounded medium context utilization",
        },
        "original_hypothesis_recovered": "M1 introduced extra direction/XY/camera-conditioned medium inputs to increase view-dependent medium modeling capacity.",
        "pre_m1_baseline_recovered": "The pre-M1 medium field used direction-conditioned medium input; current code names this mode dir_only.",
        "prior_interpretation": "Previous MEDCTX audit classified extra context as used without hard-region association; it closed the Panama hard-region explanation question, not this IUI3 camera-context expressiveness/ambiguity question.",
    }
    _write_json(output_dir / "recovered_historical_m1_evidence.json", evidence)
    return evidence


def _final_summary(
    output_dir: Path,
    env: Mapping[str, Any],
    gpu: Mapping[str, Any],
    start_audit: Mapping[str, Any],
    camera_audit: Mapping[str, Any],
    global_rows: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
    decomp_rows: Sequence[Mapping[str, Any]],
    medium_rows: Sequence[Mapping[str, Any]],
    between_rows: Sequence[Mapping[str, Any]],
    ident_rows: Sequence[Mapping[str, Any]],
    weak_rows: Sequence[Mapping[str, Any]],
    swap_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    final_step: int,
) -> Dict[str, Any]:
    g = {(row["branch"], int(row["absolute_step"]), row["split"]): row for row in global_rows}
    c0_eval = g[("C0", int(final_step), "eval")]
    c1_eval = g[("C1", int(final_step), "eval")]
    c0_train = g[("C0", int(final_step), "train")]
    c1_train = g[("C1", int(final_step), "train")]
    delta_eval = {k: float(c1_eval[k]) - float(c0_eval[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")}
    per_eval_delta = _delta_rows(per_view_rows, ("absolute_step", "split", "view_id"), ("PSNR", "SSIM", "LPIPS", "MSE"))
    per_eval_final = [r for r in per_eval_delta if int(r["absolute_step"]) == int(final_step) and r["split"] == "eval"]
    positive_psnr_views = sum(1 for row in per_eval_final if float(row["delta_C1_minus_C0_PSNR"]) > 0.0)
    c1_decomp_final = [row for row in decomp_rows if row["branch"] == "C1" and int(row["absolute_step"]) == int(final_step)]
    decomp_safe = all(float(row.get("P_J_gt_1", 1.0)) == 0.0 for row in c1_decomp_final)
    utility_final = [row for row in swap_outputs["correct_vs_swapped_context_utility"] if int(row["absolute_step"]) == int(final_step)]
    utility_positive = _mean(utility_final, "fraction_Delta_E_swap_gt_0")
    utility_delta = _mean(utility_final, "Delta_E_swap_mean")
    c1_used = bool(utility_final and utility_positive > 0.5 and utility_delta > 0.0)
    if delta_eval["PSNR"] > 0.0 and (delta_eval["SSIM"] > 0.0 or delta_eval["LPIPS"] < 0.0) and positive_psnr_views >= 2 and decomp_safe and c1_used:
        express = "CAMERA_CONTEXT_EXPRESSIVENESS_SUPPORTED"
    elif delta_eval["PSNR"] > -0.05 or positive_psnr_views >= 2 or c1_used:
        express = "CAMERA_CONTEXT_EXPRESSIVENESS_TENTATIVE"
    else:
        express = "CAMERA_CONTEXT_EXPRESSIVENESS_NOT_SUPPORTED"

    proj_rows = list(swap_outputs["camera_delta_weak_projection"])
    sens_rows = list(swap_outputs["weak_component_rgb_sensitivity"])
    removal_rows = list(swap_outputs["weak_component_removal_counterfactual"])
    projection_by_step = {
        step: _mean([r for r in proj_rows if int(r["absolute_step"]) == step and r["population"] == "M_SAFE"], "weak_projection_over_random_1over9")
        for step in IDENTIFIABILITY_STEPS
        if step <= final_step
    }
    sens_by_step = {
        step: _mean([r for r in sens_rows if int(r["absolute_step"]) == step and r["population"] == "M_SAFE"], "weak_over_orth_rgb_change")
        for step in IDENTIFIABILITY_STEPS
        if step <= final_step
    }
    removal_by_step = {
        step: _mean([r for r in removal_rows if int(r["absolute_step"]) == step and r["population"] == "M_SAFE"], "Delta_E_remove_weak_mean")
        for step in IDENTIFIABILITY_STEPS
        if step <= final_step
    }
    nontrivial_delta = _mean([r for r in swap_outputs["camera_induced_medium_delta"] if r["population"] == "M_SAFE"], "raw_delta_z_cam_rms_9d") > 1e-4
    proj_supported_steps = sum(1 for v in projection_by_step.values() if math.isfinite(v) and v > 1.25)
    weak_less_sensitive_steps = sum(1 for v in sens_by_step.values() if math.isfinite(v) and v < 0.5)
    removal_safe_steps = sum(1 for v in removal_by_step.values() if math.isfinite(v) and v <= 1e-6)
    if nontrivial_delta and proj_supported_steps >= 2 and weak_less_sensitive_steps >= 2 and removal_safe_steps >= 2:
        ambiguity = "CAMERA_CONTEXT_AMBIGUITY_SUPPORTED"
    elif nontrivial_delta and (proj_supported_steps >= 1 or weak_less_sensitive_steps >= 1 or removal_safe_steps >= 1):
        ambiguity = "CAMERA_CONTEXT_AMBIGUITY_TENTATIVE"
    else:
        ambiguity = "CAMERA_CONTEXT_AMBIGUITY_NOT_SUPPORTED"

    if express == "CAMERA_CONTEXT_EXPRESSIVENESS_SUPPORTED" and ambiguity == "CAMERA_CONTEXT_AMBIGUITY_SUPPORTED":
        combined = "CAMERA_CONDITIONED_IDENTIFIABILITY_TRADEOFF_SUPPORTED"
    elif "TENTATIVE" in express or "TENTATIVE" in ambiguity:
        combined = "CAMERA_CONDITIONED_IDENTIFIABILITY_TRADEOFF_TENTATIVE"
    else:
        combined = "CAMERA_CONDITIONED_IDENTIFIABILITY_TRADEOFF_NOT_SUPPORTED"

    phase_b_entered = combined == "CAMERA_CONDITIONED_IDENTIFIABILITY_TRADEOFF_SUPPORTED"
    phase_b_class = "NOT_ENTERED"
    next_experiment = (
        "formal matched causal training: camera-conditioned baseline vs camera-conditioned + observability module on IUI3"
        if phase_b_entered
        else "Run the one missing diagnostic indicated by the tentative side; do not implement a new camera-context regularizer."
        if combined == "CAMERA_CONDITIONED_IDENTIFIABILITY_TRADEOFF_TENTATIVE"
        else "Do not build another camera-context regularizer."
    )
    summary = {
        "experiment": "M1-CAMERA-CONTEXT-IDENTIFIABILITY-AUDIT",
        "scene": SCENE,
        "CONDA_ENV": env["CONDA_ENV"],
        "PYTHON_PATH": env["PYTHON_PATH"],
        "TORCH_VERSION": env["TORCH_VERSION"],
        "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
        "gpu": dict(gpu),
        "final_step": int(final_step),
        "PARAMETER_STATE_EQUIVALENCE": bool(start_audit["PARAMETER_STATE_EQUIVALENCE"]),
        "OPTIMIZER_STATE_EQUIVALENCE": bool(start_audit["OPTIMIZER_STATE_EQUIVALENCE"]),
        "RNG_EQUIVALENCE": bool(start_audit["RNG_EQUIVALENCE"]),
        "CAMERA_SEQUENCE_MATCH": bool(camera_audit["CAMERA_SEQUENCE_MATCH"]),
        "final_train_C0": {k: float(c0_train[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_train_C1": {k: float(c1_train[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_eval_C0": {k: float(c0_eval[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_eval_C1": {k: float(c1_eval[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_eval_delta_C1_minus_C0": delta_eval,
        "final_eval_views_positive_PSNR": int(positive_psnr_views),
        "final_eval_view_count": len(per_eval_final),
        "correct_context_utility_fraction_positive_final": utility_positive,
        "correct_context_utility_delta_E_final": utility_delta,
        "M_SAFE_weak_projection_over_random_by_step": projection_by_step,
        "M_SAFE_weak_over_orth_rgb_sensitivity_by_step": sens_by_step,
        "M_SAFE_remove_weak_delta_E_by_step": removal_by_step,
        "decomposition_safety_intact": bool(decomp_safe),
        "camera_context_expressiveness_classification": express,
        "camera_context_ambiguity_classification": ambiguity,
        "combined_tradeoff_classification": combined,
        "PHASE_B_ENTERED": bool(phase_b_entered),
        "phase_b_classification": phase_b_class,
        "next_single_formal_experiment": next_experiment,
    }
    _write_json(output_dir / "final_summary.json", summary)
    _write_csv(output_dir / "final_summary.csv", [{"key": k, "value": v} for k, v in summary.items() if not isinstance(v, (dict, list))])
    _write_csv(output_dir / "global_rgb_deltas.csv", _delta_rows(global_rows, ("absolute_step", "split"), ("PSNR", "SSIM", "LPIPS", "MSE")))
    _write_csv(output_dir / "per_view_rgb_deltas.csv", per_eval_final)
    _write_json(output_dir / "per_view_rgb_deltas.json", {"rows": per_eval_final})
    _write_csv(output_dir / "identifiability_deltas.csv", _delta_rows(ident_rows, ("nominal_step", "population"), ("sigma_min_over_sigma_max", "condition_number", "effective_rank"), branch_field="run"))
    _write_csv(output_dir / "medium_output_deltas.csv", _delta_rows(medium_rows, ("absolute_step", "population"), ("raw_betaD_variance_pooled", "beta_D_variance_pooled", "rgb_mse")))
    _write_csv(output_dir / "between_within_deltas.csv", _delta_rows(between_rows, ("absolute_step", "population"), ("raw_betaD_r_between_over_within", "raw_betaD_g_between_over_within", "raw_betaD_b_between_over_within")))
    return summary


def _write_research_note(repo: Path, output_dir: Path, summary: Mapping[str, Any], source: Mapping[str, Any], history: Mapping[str, Any]) -> None:
    delta = summary["final_eval_delta_C1_minus_C0"]
    lines = [
        "# M1-CAMERA-CONTEXT-IDENTIFIABILITY-AUDIT",
        "",
        "## CODE FACT",
        f"The current 3-D camera context is `{source['definition']}`.",
        "It is not a learned latent, is not trainable, and is constructed from each camera center at train/eval time.",
        "C0 keeps the 22-D medium MLP input but sets only the final camera-context feature to zero.",
        "",
        "## CONFIG FACT",
        "Both arms use BND as a controlled bounded intrinsic-color parameterization: `bounded_sh3`, SH degree 3, `dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=False`.",
        "Only intervention: C0 neutral camera context vs C1 formal scene-normalized camera-center context.",
        "",
        "## EXPERIMENTAL FACT",
        f"Parameter/optimizer/RNG equivalence: `{summary['PARAMETER_STATE_EQUIVALENCE']}`, `{summary['OPTIMIZER_STATE_EQUIVALENCE']}`, `{summary['RNG_EQUIVALENCE']}`.",
        f"Camera sequence match: `{summary['CAMERA_SEQUENCE_MATCH']}`.",
        f"Outputs: `{output_dir}`.",
        "",
        "## QUANTITATIVE RESULT",
        f"Final eval C1-C0: PSNR `{delta['PSNR']:.6f}` dB, SSIM `{delta['SSIM']:.6f}`, LPIPS `{delta['LPIPS']:.6f}`, MSE `{delta['MSE']:.8f}`.",
        f"Final eval PSNR positive views: `{summary['final_eval_views_positive_PSNR']}/{summary['final_eval_view_count']}`.",
        f"Correct-context swap utility final fraction positive: `{summary['correct_context_utility_fraction_positive_final']}`.",
        f"M_SAFE weak projection over 1/9 random reference by step: `{summary['M_SAFE_weak_projection_over_random_by_step']}`.",
        f"M_SAFE weak/orth RGB sensitivity by step: `{summary['M_SAFE_weak_over_orth_rgb_sensitivity_by_step']}`.",
        f"M_SAFE weak-removal delta_E by step: `{summary['M_SAFE_remove_weak_delta_E_by_step']}`.",
        "",
        "## INFERENCE",
        f"Expressiveness classification: `{summary['camera_context_expressiveness_classification']}`.",
        f"Ambiguity classification: `{summary['camera_context_ambiguity_classification']}`.",
        f"Combined tradeoff classification: `{summary['combined_tradeoff_classification']}`.",
        f"Phase B entered: `{summary['PHASE_B_ENTERED']}`; Phase-B classification: `{summary['phase_b_classification']}`.",
        "No true-medium, true-color, or true-geometry claim is made.",
        "",
        "## HYPOTHESIS",
        f"Next single formal experiment: {summary['next_single_formal_experiment']}",
        "",
        "## RECOVERED HISTORICAL M1 EVIDENCE",
        str(history.get("introduced_by_blame", {})),
        "",
    ]
    path = repo / RESEARCH_NOTE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _prepare_output_dir(output_dir: Path, allow_existing: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise RuntimeError(f"Output directory exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def run(repo: Path, output_dir: Path, final_step: int, allow_existing_output: bool) -> Dict[str, Any]:
    gpu = _assert_runtime_policy()
    output_dir = output_dir if output_dir.is_absolute() else repo / output_dir
    _prepare_output_dir(output_dir, allow_existing_output)
    env = _environment_manifest(gpu)
    repo_manifest = _repo_manifest(repo)
    _write_json(output_dir / "gpu_manifest.json", gpu)
    _write_json(output_dir / "environment_manifest.json", env)
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    shutil.copy2(repo / BND_CONFIG, output_dir / "source_bnd_config.yml")
    source = _source_semantics(repo, output_dir)
    history = _historical_m1_evidence(repo, output_dir)

    snapshot_steps = tuple(step for step in SNAPSHOT_STEPS if step <= final_step)
    ident_steps = tuple(step for step in IDENTIFIABILITY_STEPS if step <= final_step)
    if final_step not in snapshot_steps:
        snapshot_steps = tuple(sorted(set(snapshot_steps + (final_step,))))
    if final_step >= 5000 and final_step not in ident_steps:
        ident_steps = tuple(sorted(set(ident_steps + (final_step,))))

    seq_branch = _setup_branch(repo, "C1")
    try:
        camera_indices, camera_names = _generate_camera_sequence(seq_branch, output_dir, int(final_step))
    finally:
        _release(seq_branch)
    camera_audit = json.loads((output_dir / "camera_sequence_audit.json").read_text(encoding="utf8"))
    training_rng = _seed_all(TRAINING_RNG_SEED)
    _write_json(output_dir / "rng_state_manifest.json", {"seed": TRAINING_RNG_SEED, **_rng_manifest(training_rng)})
    start_audit = _start_state_audit(repo, output_dir, training_rng)

    all_training: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    all_ckpts: List[Dict[str, Any]] = []
    all_topology: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        print(f"[M1-CAMCTX] training {branch_name}", flush=True)
        rows, events, ckpts, topology = _train_branch(
            repo,
            branch_name,
            camera_indices=camera_indices,
            camera_names=camera_names,
            training_rng=training_rng,
            snapshot_steps=snapshot_steps,
            output_dir=output_dir,
        )
        all_training.extend(rows)
        all_events.extend(events)
        all_ckpts.extend(ckpts)
        all_topology.extend(topology)
    _write_csv(output_dir / "training_metrics.csv", all_training)
    _write_json(output_dir / "training_metrics.json", {"rows": all_training})
    _write_csv(output_dir / "refinement_events.csv", all_events)
    _write_json(output_dir / "refinement_events.json", {"rows": all_events})
    _write_csv(output_dir / "checkpoint_manifest.csv", all_ckpts)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": all_ckpts})
    _write_csv(output_dir / "gaussian_population.csv", all_topology)
    _write_json(output_dir / "gaussian_population.json", {"rows": all_topology})

    print("[M1-CAMCTX] evaluating snapshots", flush=True)
    global_rows, per_view_rows, decomp_rows = _evaluate_snapshots(repo, output_dir, snapshot_steps)
    _write_csv(output_dir / "global_rgb_metrics.csv", global_rows)
    _write_json(output_dir / "global_rgb_metrics.json", {"rows": global_rows})
    _write_csv(output_dir / "per_view_rgb_metrics.csv", per_view_rows)
    _write_json(output_dir / "per_view_rgb_metrics.json", {"rows": per_view_rows})
    _write_csv(output_dir / "decomposition_safety.csv", decomp_rows)
    _write_json(output_dir / "decomposition_safety.json", {"rows": decomp_rows})

    print("[M1-CAMCTX] building deterministic samples", flush=True)
    samples, sampling_meta, sampling_rows = MI._build_samples(repo, output_dir, SAMPLES_PER_VIEW, SAMPLE_SEED)
    _write_csv(output_dir / "deterministic_ray_sampling.csv", sampling_rows)
    _write_json(output_dir / "deterministic_ray_sampling.json", sampling_meta)

    medium_outputs = _medium_distribution_audit(repo, output_dir, snapshot_steps, samples)
    if ident_steps:
        ident_outputs, weak_basis = _run_identifiability_audit(repo, output_dir, ident_steps, samples)
        swap_outputs = _camera_swap_and_weak_component_audit(repo, output_dir, ident_steps, samples, weak_basis)
    else:
        ident_outputs = {
            "identifiability_summary": [],
            "weak_mode_summary": [],
        }
        swap_outputs = {
            "correct_vs_swapped_context_utility": [],
            "camera_induced_medium_delta": [],
            "camera_delta_weak_projection": [],
            "weak_component_rgb_sensitivity": [],
            "weak_component_removal_counterfactual": [],
            "depth_tau_camera_context_strata": [],
            "per_camera_stability": [],
        }

    summary = _final_summary(
        output_dir,
        env,
        gpu,
        start_audit,
        camera_audit,
        global_rows,
        per_view_rows,
        decomp_rows,
        medium_outputs["medium_output_statistics"],
        medium_outputs["between_within_camera_variance"],
        ident_outputs["identifiability_summary"],
        ident_outputs["weak_mode_summary"],
        swap_outputs,
        int(final_step),
    )
    _write_research_note(repo, output_dir, summary, source, history)
    return summary


def run_analysis_only(repo: Path, output_dir: Path, final_step: int) -> Dict[str, Any]:
    gpu = _assert_runtime_policy()
    output_dir = output_dir if output_dir.is_absolute() else repo / output_dir
    if not output_dir.exists():
        raise FileNotFoundError(f"Missing output directory for analysis-only mode: {output_dir}")
    env = _environment_manifest(gpu)
    _write_json(output_dir / "gpu_manifest.json", gpu)
    _write_json(output_dir / "environment_manifest.json", env)
    _write_json(output_dir / "repo_manifest.json", _repo_manifest(repo))
    source = _source_semantics(repo, output_dir)
    history = _historical_m1_evidence(repo, output_dir)

    snapshot_steps = tuple(step for step in SNAPSHOT_STEPS if step <= final_step)
    ident_steps = tuple(step for step in IDENTIFIABILITY_STEPS if step <= final_step)
    if final_step not in snapshot_steps:
        snapshot_steps = tuple(sorted(set(snapshot_steps + (final_step,))))
    if final_step >= 5000 and final_step not in ident_steps:
        ident_steps = tuple(sorted(set(ident_steps + (final_step,))))
    _require_existing_checkpoints(output_dir, snapshot_steps)

    start_audit = json.loads((output_dir / "start_state_audit.json").read_text(encoding="utf8"))
    camera_audit = json.loads((output_dir / "camera_sequence_audit.json").read_text(encoding="utf8"))
    global_rows = _read_csv(output_dir / "global_rgb_metrics.csv")
    per_view_rows = _read_csv(output_dir / "per_view_rgb_metrics.csv")
    decomp_rows = _read_csv(output_dir / "decomposition_safety.csv")

    print("[M1-CAMCTX] analysis-only building deterministic samples", flush=True)
    samples, sampling_meta, sampling_rows = MI._build_samples(repo, output_dir, SAMPLES_PER_VIEW, SAMPLE_SEED)
    _write_csv(output_dir / "deterministic_ray_sampling.csv", sampling_rows)
    _write_json(output_dir / "deterministic_ray_sampling.json", sampling_meta)

    medium_outputs = _medium_distribution_audit(repo, output_dir, snapshot_steps, samples)
    ident_outputs, weak_basis = _run_identifiability_audit(repo, output_dir, ident_steps, samples)
    swap_outputs = _camera_swap_and_weak_component_audit(repo, output_dir, ident_steps, samples, weak_basis)

    summary = _final_summary(
        output_dir,
        env,
        gpu,
        start_audit,
        camera_audit,
        global_rows,
        per_view_rows,
        decomp_rows,
        medium_outputs["medium_output_statistics"],
        medium_outputs["between_within_camera_variance"],
        ident_outputs["identifiability_summary"],
        ident_outputs["weak_mode_summary"],
        swap_outputs,
        int(final_step),
    )
    _write_research_note(repo, output_dir, summary, source, history)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--final-step", type=int, default=FINAL_ACTUAL_STEP)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis_only:
        summary = run_analysis_only(args.repo.resolve(), args.output_dir, int(args.final_step))
    else:
        summary = run(args.repo.resolve(), args.output_dir, int(args.final_step), bool(args.allow_existing_output))
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
