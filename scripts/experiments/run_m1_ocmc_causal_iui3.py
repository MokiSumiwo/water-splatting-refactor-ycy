#!/usr/bin/env python3
"""Formal M1-OCMC causal training experiment on IUI3.

This driver runs the single-factor comparison requested for
M1-OCMC-CAUSAL-IUI3:

* C0: camera-conditioned M1/BND baseline.
* C1: identical camera-conditioned M1/BND model with the existing
  observability-controlled medium context (OCMC) projector enabled.

No OCMC architecture, gate formula, medium MLP, renderer, topology schedule, or
loss is redesigned here.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
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
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM


PW = MI.PW

SCENE = "IUI3-RedSea"
EXPERIMENT = "M1-OCMC-CAUSAL-IUI3"
OUTPUT_DIR = Path("outputs/m1_ocmc_causal_iui3_20260825")
PHASE_A_OUTPUT_DIR = Path("outputs/m1_camera_context_identifiability_iui3_20260825")
RESEARCH_NOTE = Path("research_notes/M1_OCMC_CAUSAL_IUI3_2026-08-25.md")
BND_CONFIG = PW.BND_CONFIG
BRANCHES = ("C0", "C1")
POPULATIONS = ("GENERAL", "M_SAFE")
FINAL_NOMINAL_STEP = 15000
FINAL_ACTUAL_STEP = 14999
SNAPSHOT_STEPS = (3000, 5000, 8000, 10000, 13000, 14999)
IDENTIFIABILITY_STEPS = (5000, 10000, 14999)
PHASE_A_STABILITY_STEPS = (5000, 10000, 14999)
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
TRAINING_RNG_SEED = 202608254
SAMPLES_PER_VIEW = 1024
SAMPLE_SEED = 20260825
SWAP_SEED = 202608255
ALT_CONTEXT_COUNT = 8
COUNTERFACTUAL_EPSILON = 0.25
RANDOM_DIRECTIONS = 8
LOG_INTERVAL = 500
PROJECTOR_POPULATION = "GENERAL"
PROJECTOR_STABLE_REL_FRO_MAX = 0.35
PROJECTOR_STABLE_VMIN_COS_MIN = 0.95
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


def _repo_path(path: Path, repo: Path) -> Path:
    return path if path.is_absolute() else repo / path


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
        raise RuntimeError("CUDA must be available for this formal experiment.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one torch-visible GPU after masking, got {torch.cuda.device_count()}")
    logical = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(logical)
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "physical_gpu_id": visible,
        "torch_logical_gpu_id": logical,
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


def _train_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return PW._records(pipeline)["train"]


def _eval_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return PW._records(pipeline)["eval"]


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
    model_cfg.medium_camera_context_ablation = False
    model_cfg.camera_medium_observability_enabled = branch == "C1"
    model_cfg.camera_medium_observability_strength = 1.0
    model_cfg.b_inf_mode = "tied"
    model_cfg.infinite_water_enabled = False
    model_cfg.coarse_depth_supervision_enabled = False
    model_cfg.medium_identifiability_enabled = False
    model_cfg.medium_identifiability_weight = 0.0
    return config


def _configure_model(model: Any, branch: str, *, projector_enabled: Optional[bool] = None) -> None:
    enabled = branch == "C1" if projector_enabled is None else bool(projector_enabled)
    model.config.intrinsic_color_parameterization = "bounded_sh3"
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.medium_camera_context_scale = 1.0
    model.config.medium_camera_context_dropout = 0.0
    model.config.medium_camera_context_ablation = False
    model.config.camera_medium_observability_enabled = enabled
    model.config.camera_medium_observability_strength = 1.0
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
    pipeline.model.set_camera_medium_observability_projector(None)
    pipeline.model.step = 0
    optimizers = Optimizers(MIC._optimizer_groups(config, pipeline.model), pipeline.model.get_param_groups())
    pipeline.eval()
    return BranchState(branch=branch, config_path=config_path, config=config, pipeline=pipeline, optimizers=optimizers, scalers={})


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


def _config_snapshot(branch: BranchState) -> Dict[str, Any]:
    cfg = branch.pipeline.model.config
    keys = (
        "intrinsic_color_parameterization",
        "sh_degree",
        "rasterize_mode",
        "medium_context_mode",
        "medium_camera_context_scale",
        "medium_camera_context_dropout",
        "medium_camera_context_ablation",
        "camera_medium_observability_enabled",
        "camera_medium_observability_strength",
        "b_inf_mode",
        "infinite_water_enabled",
        "coarse_depth_supervision_enabled",
        "medium_identifiability_enabled",
        "medium_identifiability_weight",
        "refine_every",
        "stop_split_at",
        "reset_alpha_every",
        "continue_cull_post_densification",
    )
    return {
        "branch": branch.branch,
        "config_path": str(branch.config_path),
        "optimizer_groups": sorted(branch.optimizers.optimizers.keys()),
        "scheduler_groups": sorted(branch.optimizers.schedulers.keys()),
        "model_config": {key: getattr(cfg, key, None) for key in keys},
    }


def _start_state_audit(repo: Path, output_dir: Path, training_rng: Mapping[str, Any]) -> Dict[str, Any]:
    c0: Optional[BranchState] = None
    c1: Optional[BranchState] = None
    try:
        c0 = _setup_branch(repo, "C0")
        c1 = _setup_branch(repo, "C1")
        c0.pipeline.model.set_camera_medium_observability_projector(None)
        c1.pipeline.model.set_camera_medium_observability_projector(None)
        _write_json(output_dir / "config_snapshot_C0.json", _config_snapshot(c0))
        _write_json(output_dir / "config_snapshot_C1.json", _config_snapshot(c1))

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
        scaler_rows = [
            {
                "state": "scalers",
                "C0_hash": _object_hash(c0.scalers),
                "C1_hash": _object_hash(c1.scalers),
                "pass": _object_hash(c0.scalers) == _object_hash(c1.scalers),
            }
        ]

        idx0, view0, camera0, batch0 = _train_records(c0.pipeline)[0]
        idx1, view1, camera1, batch1 = _train_records(c1.pipeline)[0]
        if idx0 != idx1 or view0 != view1:
            raise RuntimeError(f"Start audit train record mismatch: {(idx0, view0)} vs {(idx1, view1)}")
        model0 = c0.pipeline.model
        model1 = c1.pipeline.model
        model0.eval()
        model1.eval()
        batch0 = _batch_to_device(batch0.copy(), model0.device)
        batch1 = _batch_to_device(batch1.copy(), model1.device)
        with torch.no_grad():
            out0 = model0.get_outputs_for_camera(camera0.to(model0.device))
            out1 = model1.get_outputs_for_camera(camera1.to(model1.device))
            raw0, h0, w0, _features0 = CAM._medium_raw_for_camera(model0, camera0, force_real_camera_context=True)
            raw1, h1, w1, _features1 = CAM._medium_raw_for_camera(model1, camera1, force_real_camera_context=True)
            loss0 = model0.get_loss_dict(out0, batch0, {})["main_loss"]
            loss1 = model1.get_loss_dict(out1, batch1, {})["main_loss"]

        key_map = (
            ("pred_image", "pred_image"),
            ("depth", "depth"),
            ("accumulation", "accumulation"),
            ("medium_rgb", "medium_rgb"),
            ("beta_B", "medium_bs"),
            ("beta_D", "medium_attn"),
            ("raw z_med", "manual_medium_mlp_raw"),
        )
        forward_rows: List[Dict[str, Any]] = []
        for label, key in key_map:
            if key == "manual_medium_mlp_raw":
                diff = _max_abs_diff(raw0.view(h0, w0, 9), raw1.view(h1, w1, 9))
            else:
                diff = _max_abs_diff(out0[key], out1[key])
            forward_rows.append({"quantity": label, "source_key": key, "max_abs_diff": diff, "pass": bool(diff == 0.0)})
        main_diff = abs(float(loss0.detach().cpu().item()) - float(loss1.detach().cpu().item()))
        forward_rows.append({"quantity": "main RGB loss", "source_key": "loss_dict.main_loss", "max_abs_diff": main_diff, "pass": bool(main_diff == 0.0)})

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

        _write_csv(output_dir / "start_state_parameter_equivalence.csv", param_rows)
        _write_json(output_dir / "start_state_parameter_equivalence.json", {"rows": param_rows})
        _write_csv(output_dir / "start_state_optimizer_equivalence.csv", opt_rows)
        _write_json(output_dir / "start_state_optimizer_equivalence.json", {"rows": opt_rows})
        _write_csv(output_dir / "start_state_scheduler_equivalence.csv", sched_rows)
        _write_json(output_dir / "start_state_scheduler_equivalence.json", {"rows": sched_rows})
        _write_csv(output_dir / "start_state_scaler_equivalence.csv", scaler_rows)
        _write_json(output_dir / "start_state_scaler_equivalence.json", {"rows": scaler_rows})
        _write_csv(output_dir / "start_state_rng_equivalence.csv", rng_rows)
        _write_json(output_dir / "start_state_rng_equivalence.json", {"rows": rng_rows})
        _write_csv(output_dir / "start_state_forward_equivalence.csv", forward_rows)
        _write_json(output_dir / "start_state_forward_equivalence.json", {"rows": forward_rows})

        payload = {
            "MODEL_PARAMETER_EQUIVALENCE": all(row["pass"] for row in param_rows),
            "OPTIMIZER_STATE_EQUIVALENCE": all(row["pass"] for row in opt_rows),
            "SCHEDULER_STATE_EQUIVALENCE": all(row["pass"] for row in sched_rows),
            "SCALER_STATE_EQUIVALENCE": all(row["pass"] for row in scaler_rows),
            "RNG_EQUIVALENCE": all(row["pass"] for row in rng_rows),
            "FORWARD_EQUIVALENCE_BEFORE_PROJECTOR_ACTION": all(row["pass"] for row in forward_rows),
            "START_STATE_EQUIVALENCE": False,
            "first_intervention_step": 0,
            "forward_equivalence_reason": "C1 has OCMC enabled but no projector installed during the start audit, so the implemented code path is mathematically identical to C0.",
            "probe_train_index": int(idx0),
            "probe_view_id": view0,
        }
        payload["START_STATE_EQUIVALENCE"] = all(
            bool(payload[key])
            for key in (
                "MODEL_PARAMETER_EQUIVALENCE",
                "OPTIMIZER_STATE_EQUIVALENCE",
                "SCHEDULER_STATE_EQUIVALENCE",
                "SCALER_STATE_EQUIVALENCE",
                "RNG_EQUIVALENCE",
                "FORWARD_EQUIVALENCE_BEFORE_PROJECTOR_ACTION",
            )
        )
        _write_json(output_dir / "start_state_audit.json", payload)
        if not payload["START_STATE_EQUIVALENCE"]:
            raise RuntimeError("START_STATE_EQUIVALENCE=false; stopping before training.")
        return payload
    finally:
        _release(c0)
        _release(c1)


def _source_semantics(output_dir: Path) -> Dict[str, Any]:
    data = {
        "CODE_FACT": True,
        "ocmc_flags": {
            "camera_medium_observability_enabled": "default False; when True and a 9x9 projector exists, raw camera residual projection is used.",
            "camera_medium_observability_strength": "default 1.0; clamped to [0,1] inside DirectionConditionedMediumField.",
        },
        "camera_context": {
            "definition": "(camera.camera_to_worlds[0,:3,3] - scene_center) / (scene_scale + 1e-6) * medium_camera_context_scale",
            "trainable": False,
            "not_learned_latent": True,
            "input_order": "16-D direction encoding + 3-D image XY/r + 3-D scene-normalized camera center",
        },
        "raw_medium_semantics": {
            "z_med[0:3]": "sigmoid -> B_inf / medium_rgb because b_inf_mode=tied",
            "z_med[3:6]": "softplus(z+density_bias) -> beta_B / medium_bs",
            "z_med[6:9]": "softplus(z+density_bias) -> beta_D / medium_attn",
        },
        "implemented_ocmc_equations": [
            "z_full = f(dir, xy, camera_context)",
            "z_base = f(dir, xy, zero_camera_context)",
            "Delta_z_cam = z_full - z_base",
            "delta_std = Delta_z_cam / scale if scale is available",
            "Delta_projected = (delta_std @ P_obs.T) * scale",
            "z_effective = z_base + Delta_z_cam + strength * (Delta_projected - Delta_z_cam)",
            "with strength=1.0, z_effective = z_base + Delta_projected",
        ],
        "projector_gate_rule": "P_obs = V diag(g) V^T with g_i=sigma_i^2/(sigma_i^2+median(sigma)^2) in standardized 9-D raw medium space.",
        "source_files": [
            "water_splatting/fields/medium_field.py",
            "water_splatting/water_splatting.py",
        ],
    }
    _write_json(output_dir / "recovered_ocmc_semantics.json", data)
    return data


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


def _principal_angle_row(a: Tensor, b: Tensor, prefix: str) -> Dict[str, Any]:
    qa = a.detach().double()
    qb = b.detach().double()
    s = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
    angles = torch.arccos(s) * (180.0 / math.pi)
    return {
        f"{prefix}_subspace_similarity_mean_singular": float(s.mean().item()),
        f"{prefix}_principal_angle_max_deg": float(angles.max().item()),
        f"{prefix}_principal_angle_mean_deg": float(angles.mean().item()),
    }


def _projector_bundle_from_analysis(
    analysis: MI.PopAnalysis,
    *,
    branch: str,
    step: int,
    population: str,
    source: str,
) -> Dict[str, Any]:
    singular = analysis.singular_values_per_sqrt_ray.detach().float().cpu()
    eigvecs = analysis.eigvecs.detach().float().cpu()
    scale = analysis.scale.detach().float().cpu()
    sigma_ref = torch.median(singular).clamp_min(1e-12)
    gates = singular.square() / (singular.square() + sigma_ref.square())
    projector = eigvecs @ torch.diag(gates) @ eigvecs.T
    projector = 0.5 * (projector + projector.T)
    return {
        "branch": branch,
        "step": int(step),
        "population": population,
        "source": source,
        "projector": projector,
        "singular_values": singular,
        "gates": gates,
        "scale": scale,
        "eigvecs": eigvecs,
        "v_min": eigvecs[:, 0],
        "sigma_ref": float(sigma_ref.item()),
        "trace": float(torch.trace(projector).item()),
        "fro_norm": float(torch.linalg.norm(projector).item()),
        "gate_min": float(gates.min().item()),
        "gate_median": float(torch.median(gates).item()),
        "gate_max": float(gates.max().item()),
        "sampled_rays": int(analysis.z.shape[0]),
        "gate_rule": "g_i=sigma_i^2/(sigma_i^2+median(sigma)^2)",
        "scale_rule": "S_j=max(std(z_med_j),1e-3); project Delta_z_cam/S in standardized raw medium coordinates.",
    }


def _install_projector(model: Any, bundle: Optional[Mapping[str, Any]]) -> None:
    if bundle is None:
        model.set_camera_medium_observability_projector(None)
        model._formal_ocmc_projector_bundle = None
        return
    model.set_camera_medium_observability_projector(bundle["projector"], bundle["singular_values"], bundle["scale"])
    model._formal_ocmc_projector_bundle = {
        key: value.detach().cpu().clone() if isinstance(value, Tensor) else value for key, value in bundle.items()
    }


def _projector_tensor(bundle: Mapping[str, Any], key: str, device: torch.device, dtype: torch.dtype) -> Tensor:
    return bundle[key].to(device=device, dtype=dtype) if isinstance(bundle[key], Tensor) else torch.tensor(bundle[key], device=device, dtype=dtype)


def _apply_projector_to_delta(delta: Tensor, bundle: Mapping[str, Any]) -> Tensor:
    projector = _projector_tensor(bundle, "projector", delta.device, delta.dtype)
    scale = _projector_tensor(bundle, "scale", delta.device, delta.dtype).reshape(1, 9).clamp_min(1e-6)
    delta_std = delta.reshape(-1, 9) / scale
    projected = (delta_std @ projector.T) * scale
    return projected.reshape_as(delta)


def _bundle_summary_row(bundle: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    row: Dict[str, Any] = {
        f"{prefix}population": bundle.get("population"),
        f"{prefix}trace": float(bundle["trace"]),
        f"{prefix}fro_norm": float(bundle["fro_norm"]),
        f"{prefix}gate_min": float(bundle["gate_min"]),
        f"{prefix}gate_median": float(bundle["gate_median"]),
        f"{prefix}gate_max": float(bundle["gate_max"]),
        f"{prefix}sigma_ref": float(bundle["sigma_ref"]),
    }
    singular = bundle["singular_values"]
    gates = bundle["gates"]
    scale = bundle["scale"]
    for idx in range(9):
        row[f"{prefix}sigma_{idx}"] = float(singular[idx].item())
        row[f"{prefix}gate_{idx}"] = float(gates[idx].item())
        row[f"{prefix}scale_{idx}"] = float(scale[idx].item())
    return row


def _projector_pair_row(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    pa = a["projector"].detach().double()
    pb = b["projector"].detach().double()
    fro = float(torch.linalg.norm(pa - pb).item())
    denom = max(float(torch.linalg.norm(pa).item()), float(torch.linalg.norm(pb).item()), EPS)
    vmin_cos = float(torch.abs(torch.dot(a["v_min"].double(), b["v_min"].double())).item())
    gate_diff = (a["gates"].double() - b["gates"].double()).abs()
    row: Dict[str, Any] = {
        "population": a["population"],
        "step_a": int(a["step"]),
        "step_b": int(b["step"]),
        "projector_frobenius_difference": fro,
        "projector_relative_frobenius_difference": fro / denom,
        "v_min_abs_cosine": vmin_cos,
        "gate_mean_abs_difference": float(gate_diff.mean().item()),
        "gate_max_abs_difference": float(gate_diff.max().item()),
        "trace_a": float(a["trace"]),
        "trace_b": float(b["trace"]),
        "trace_abs_difference": abs(float(a["trace"]) - float(b["trace"])),
    }
    row.update(_principal_angle_row(a["eigvecs"][:, :3], b["eigvecs"][:, :3], "low3"))
    row.update(_principal_angle_row(a["eigvecs"][:, -3:], b["eigvecs"][:, -3:], "top3"))
    return row


def _projector_stability_preflight(
    repo: Path,
    output_dir: Path,
    phase_a_output_dir: Path,
    samples: Mapping[str, MI.ViewSample],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    phase_a_output_dir = _repo_path(phase_a_output_dir, repo)
    bundles: Dict[Tuple[int, str], Dict[str, Any]] = {}
    spectrum_rows: List[Dict[str, Any]] = []
    for step in PHASE_A_STABILITY_STEPS:
        branch: Optional[CAM.BranchState] = None
        try:
            print(f"[M1-OCMC] projector stability preflight C1 phase-A step={step}", flush=True)
            branch = CAM._setup_branch(repo, "C1")
            CAM._load_snapshot(branch, phase_a_output_dir, int(step))
            branch.pipeline.eval()
            branch.pipeline.model.eval()
            analyses, _meta = CAM._analyse_loaded_branch(branch, samples)
            for pop, analysis in analyses.items():
                bundle = _projector_bundle_from_analysis(
                    analysis,
                    branch="phaseA_C1",
                    step=int(step),
                    population=pop,
                    source="existing_phase_A_C1_checkpoint",
                )
                bundles[(int(step), pop)] = bundle
                row = {"step": int(step), "population": pop}
                row.update(_bundle_summary_row(bundle))
                spectrum_rows.append(row)
        finally:
            CAM._release(branch)

    pair_rows: List[Dict[str, Any]] = []
    for pop in POPULATIONS:
        for i, a_step in enumerate(PHASE_A_STABILITY_STEPS):
            for b_step in PHASE_A_STABILITY_STEPS[i + 1 :]:
                pair_rows.append(_projector_pair_row(bundles[(int(a_step), pop)], bundles[(int(b_step), pop)]))

    general_pairs = [row for row in pair_rows if row["population"] == PROJECTOR_POPULATION]
    max_rel_fro = max(float(row["projector_relative_frobenius_difference"]) for row in general_pairs)
    min_vmin_cos = min(float(row["v_min_abs_cosine"]) for row in general_pairs)
    stable = bool(max_rel_fro <= PROJECTOR_STABLE_REL_FRO_MAX and min_vmin_cos >= PROJECTOR_STABLE_VMIN_COS_MIN)
    if stable:
        refresh_steps = [0, 5000, 10000]
        protocol_name = "LOW_FREQUENCY_PERIODIC_REFRESH_CURRENT_GENERAL"
        protocol_reason = (
            "GENERAL full soft projector was stable enough across existing 5k/10k/14999 Phase-A checkpoints, "
            "so the formal C1 arm uses detached current-state refreshes at 0, 5000, and 10000 only."
        )
    else:
        refresh_steps = [0, 3000, 5000, 8000, 10000, 13000]
        protocol_name = "PERIODIC_REFRESH_CURRENT_GENERAL"
        protocol_reason = (
            "GENERAL projector temporal variation exceeded the pre-registered stability threshold, "
            "so the formal C1 arm uses fixed periodic current-state refreshes at saved topology stages."
        )
    protocol = {
        "PROJECTOR_UPDATE_PROTOCOL": protocol_name,
        "decision_rule_registered_before_formal_training": True,
        "population": PROJECTOR_POPULATION,
        "data_used": "TRAIN observations only; deterministic GENERAL rays from 25 train cameras, capped at 1024 rays per camera.",
        "sampled_rays_per_camera": SAMPLES_PER_VIEW,
        "camera_selection": "validated deterministic IUI3 train-view sampling from audit_bnd_medium_identifiability_iui3.py",
        "refresh_steps": refresh_steps,
        "refresh_interval_description": "low frequency fixed steps; no RGB-dependent tuning",
        "detached_projector": True,
        "uses_current_model_state": True,
        "gate_rule_unchanged": "g_i=sigma_i^2/(sigma_i^2+median(sigma)^2)",
        "strength": 1.0,
        "population_choice_reason": "Existing prototype is a raw 9-D medium-output projector, not an M_SAFE-only implementation; formal training therefore uses the preferred GENERAL train-ray population. M_SAFE remains diagnostic only.",
        "stability_thresholds": {
            "max_relative_frobenius_difference": PROJECTOR_STABLE_REL_FRO_MAX,
            "min_v_min_abs_cosine": PROJECTOR_STABLE_VMIN_COS_MIN,
        },
        "observed_general_max_relative_frobenius_difference": max_rel_fro,
        "observed_general_min_v_min_abs_cosine": min_vmin_cos,
        "reason": protocol_reason,
    }
    preflight = {
        "experiment": EXPERIMENT,
        "source_phase_a_output_dir": str(phase_a_output_dir),
        "steps": list(PHASE_A_STABILITY_STEPS),
        "populations": list(POPULATIONS),
        "stable_under_registered_rule": stable,
        "general_max_relative_frobenius_difference": max_rel_fro,
        "general_min_v_min_abs_cosine": min_vmin_cos,
        "protocol": protocol,
    }
    _write_csv(output_dir / "projector_stability_spectrum.csv", spectrum_rows)
    _write_json(output_dir / "projector_stability_spectrum.json", {"rows": spectrum_rows})
    _write_csv(output_dir / "projector_stability_pairwise.csv", pair_rows)
    _write_json(output_dir / "projector_stability_pairwise.json", {"rows": pair_rows})
    _write_json(output_dir / "projector_stability_preflight.json", preflight)
    _write_json(output_dir / "registered_projector_update_protocol.json", protocol)
    return preflight, protocol


def _estimate_current_projector(
    branch: BranchState,
    samples: Mapping[str, MI.ViewSample],
    *,
    refresh_step: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    t0 = time.time()
    was_training = branch.pipeline.model.training
    branch.pipeline.eval()
    branch.pipeline.model.eval()
    analyses, meta = CAM._analyse_loaded_branch(branch, samples)
    analysis = analyses[PROJECTOR_POPULATION]
    bundle = _projector_bundle_from_analysis(
        analysis,
        branch=branch.branch,
        step=int(refresh_step),
        population=PROJECTOR_POPULATION,
        source="formal_current_training_state_train_rays",
    )
    if was_training:
        branch.pipeline.train()
        branch.pipeline.model.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return bundle, meta, time.time() - t0


def _projector_bundle_for_checkpoint(bundle: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if bundle is None:
        return None
    return {key: value.detach().cpu().clone() if isinstance(value, Tensor) else value for key, value in bundle.items()}


def _save_checkpoint(branch: BranchState, abs_step: int, output_dir: Path, projector_bundle: Optional[Mapping[str, Any]]) -> Path:
    path = output_dir / "checkpoints" / branch.branch / f"step-{abs_step:09d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": EXPERIMENT,
            "branch": branch.branch,
            "absolute_step": int(abs_step),
            "model": branch.pipeline.model.state_dict(),
            "ocmc_projector_bundle": _projector_bundle_for_checkpoint(projector_bundle) if branch.branch == "C1" else None,
            "metadata": {
                "camera_context_enabled": True,
                "camera_medium_observability_enabled": branch.branch == "C1",
                "camera_medium_observability_strength": 1.0,
                "only_intervention": "C1 applies existing OCMC projector to camera-conditioned medium residual",
                "matched_camera_sequence": True,
                "normal_topology_enabled": True,
            },
        },
        path,
    )
    return path


def _ckpt_path(output_dir: Path, branch: str, abs_step: int) -> Path:
    return output_dir / "checkpoints" / branch / f"step-{abs_step:09d}.ckpt"


def _load_snapshot(branch: BranchState, output_dir: Path, abs_step: int) -> Optional[Dict[str, Any]]:
    ckpt = torch.load(_ckpt_path(output_dir, branch.branch, abs_step), map_location="cpu")
    branch.pipeline.model.load_state_dict(ckpt["model"], strict=True)
    branch.pipeline.model.step = int(ckpt["absolute_step"])
    _configure_model(branch.pipeline.model, branch.branch)
    bundle = ckpt.get("ocmc_projector_bundle")
    if branch.branch == "C1":
        if bundle is None:
            raise RuntimeError(f"Missing OCMC projector bundle in C1 checkpoint {abs_step}")
        _install_projector(branch.pipeline.model, bundle)
    else:
        _install_projector(branch.pipeline.model, None)
        branch.pipeline.model.config.camera_medium_observability_enabled = False
    branch.pipeline.eval()
    branch.pipeline.model.eval()
    return bundle


def _require_existing_checkpoints(output_dir: Path, steps: Sequence[int]) -> None:
    missing = [
        str(_ckpt_path(output_dir, branch, int(step)))
        for branch in BRANCHES
        for step in steps
        if not _ckpt_path(output_dir, branch, int(step)).exists()
    ]
    if missing:
        raise FileNotFoundError("analysis-only mode requires existing checkpoints: " + ", ".join(missing))


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


def _residual_metric(outputs: Mapping[str, Tensor]) -> Dict[str, Any]:
    if "camera_medium_delta_raw" not in outputs:
        return {}
    delta = outputs["camera_medium_delta_raw"].detach().float()
    projected = outputs["camera_medium_delta_projected_raw"].detach().float()
    suppressed = outputs["camera_medium_delta_suppressed_raw"].detach().float()
    full_rms = math.sqrt(float(delta.square().mean().cpu().item()))
    proj_rms = math.sqrt(float(projected.square().mean().cpu().item()))
    supp_rms = math.sqrt(float(suppressed.square().mean().cpu().item()))
    return {
        "camera_residual_rms": full_rms,
        "camera_projected_residual_rms": proj_rms,
        "camera_suppressed_residual_rms": supp_rms,
        "camera_projected_over_full": proj_rms / max(full_rms, EPS),
        "camera_suppressed_over_full": supp_rms / max(full_rms, EPS),
    }


def _direct_ocmc_gradient_probe(branch: BranchState, camera: Cameras, batch: Mapping[str, Any], abs_step: int) -> Dict[str, Any]:
    model = branch.pipeline.model
    branch.optimizers.zero_grad_all()
    model.zero_grad(set_to_none=True)
    outputs = model.get_outputs(camera)
    if "camera_medium_delta_suppressed_raw" not in outputs:
        return {
            "branch": branch.branch,
            "absolute_step": int(abs_step),
            "direct_metric_available": False,
            "NEW_MECHANISM_GRAD_OBJECT": "not_applicable",
        }
    metric = outputs["camera_medium_delta_suppressed_raw"].float().square().mean()
    metric.backward()
    stats = MIC._param_group_grad_stats(model)
    gaussian_groups = ("means", "scales", "quats", "features_dc", "features_rest", "opacities")
    gaussian_l2 = sum(float(stats[group]["grad_l2"]) for group in gaussian_groups if group in stats)
    row = {
        "branch": branch.branch,
        "absolute_step": int(abs_step),
        "direct_metric_available": True,
        "direct_suppressed_delta_metric": float(metric.detach().cpu().item()),
        "medium_mlp_grad_l2": float(stats["medium_mlp"]["grad_l2"]),
        "direction_encoding_grad_l2": float(stats["direction_encoding"]["grad_l2"]),
        "medium_branch_grad_l2": float(stats["medium_branch"]["grad_l2"]),
        "gaussian_grad_l2_sum": gaussian_l2,
        "NEW_MECHANISM_GRAD_OBJECT": "nonzero" if gaussian_l2 > 0.0 else "0",
        "grad_stats": stats,
    }
    branch.optimizers.zero_grad_all()
    model.zero_grad(set_to_none=True)
    return row


def _train_branch(
    repo: Path,
    branch_name: str,
    *,
    camera_indices: Sequence[int],
    camera_names: Sequence[str],
    training_rng: Mapping[str, Any],
    snapshot_steps: Sequence[int],
    output_dir: Path,
    samples: Mapping[str, MI.ViewSample],
    protocol: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    branch = _setup_branch(repo, branch_name)
    _set_rng_state(training_rng)
    model = branch.pipeline.model
    dm = branch.pipeline.datamanager
    cached_train = dm.cached_train
    train_cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
    snapshot_set = set(int(x) for x in snapshot_steps)
    refresh_set = set(int(x) for x in protocol["refresh_steps"] if int(x) <= max(snapshot_steps))
    training_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    ckpt_rows: List[Dict[str, Any]] = []
    topology_rows: List[Dict[str, Any]] = []
    projector_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    current_projector: Optional[Dict[str, Any]] = None
    prev_projector: Optional[Dict[str, Any]] = None
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for abs_step, (camera_index, camera_name) in enumerate(zip(camera_indices, camera_names)):
            branch.pipeline.train()
            model.train()
            _configure_model(model, branch_name)
            if branch_name == "C1" and abs_step in refresh_set:
                bundle, meta, seconds = _estimate_current_projector(branch, samples, refresh_step=abs_step)
                current_projector = bundle
                _install_projector(model, current_projector)
                row = {
                    "branch": branch_name,
                    "refresh_step": int(abs_step),
                    "projector_compute_seconds": seconds,
                    "analysis_loaded_step": int(meta["loaded_step"]),
                    "gaussian_count": int(meta["gaussian_count"]),
                    "projector_change_from_previous": float("nan"),
                    "weak_direction_abs_cosine_from_previous": float("nan"),
                }
                if prev_projector is not None:
                    pair = _projector_pair_row(prev_projector, current_projector)
                    row["projector_change_from_previous"] = pair["projector_frobenius_difference"]
                    row["projector_relative_change_from_previous"] = pair["projector_relative_frobenius_difference"]
                    row["weak_direction_abs_cosine_from_previous"] = pair["v_min_abs_cosine"]
                row.update(_bundle_summary_row(current_projector))
                projector_rows.append(row)
                prev_projector = _projector_bundle_for_checkpoint(current_projector)
                _write_csv(output_dir / "projector_dynamics.csv", projector_rows)
                _write_json(output_dir / "projector_dynamics.json", {"rows": projector_rows})
                branch.pipeline.train()
                model.train()
            elif branch_name == "C0":
                _install_projector(model, None)
                model.config.camera_medium_observability_enabled = False
            elif branch_name == "C1" and current_projector is None:
                raise RuntimeError("C1 reached training without an installed projector; refresh step 0 is required.")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            MIC._run_before(model, branch.optimizers, abs_step)
            branch.optimizers.zero_grad_all()
            batch = _batch_to_device(cached_train[camera_index].copy(), model.device)
            camera = train_cameras[camera_index : camera_index + 1]

            if branch_name == "C1" and (abs_step in snapshot_set or abs_step in refresh_set):
                gradient_rows.append(_direct_ocmc_gradient_probe(branch, camera, batch, abs_step))

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
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - t0
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
                "step_time_seconds": elapsed,
                "stable": True,
                "camera_context_enabled": True,
                "camera_medium_observability_enabled": branch_name == "C1",
                "projector_refresh_step": int(current_projector["step"]) if current_projector is not None else "",
            }
            row.update(_residual_metric(outputs))
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
                ckpt = _save_checkpoint(branch, abs_step, output_dir, current_projector)
                ckpt_rows.append(
                    {
                        "branch": branch_name,
                        "absolute_step": abs_step,
                        "checkpoint_path": str(ckpt),
                        "ocmc_projector_refresh_step": int(current_projector["step"]) if current_projector is not None else "",
                    }
                )
                _write_csv(output_dir / f"{branch_name.lower()}_training_log.csv", training_rows)
                _write_json(output_dir / f"{branch_name.lower()}_training_log.json", {"rows": training_rows})
                _write_csv(output_dir / f"{branch_name.lower()}_refinement_events.csv", event_rows)
                _write_json(output_dir / f"{branch_name.lower()}_refinement_events.json", {"rows": event_rows})
                _write_csv(output_dir / f"{branch_name.lower()}_gradient_pathway.csv", gradient_rows)
                _write_json(output_dir / f"{branch_name.lower()}_gradient_pathway.json", {"rows": gradient_rows})
        return training_rows, event_rows, ckpt_rows, topology_rows, projector_rows, gradient_rows
    finally:
        _release(branch)


def _render_records(pipeline: Any, records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]]) -> Dict[str, Dict[str, Tensor]]:
    return MIC._render_records(pipeline, records)


def _decomposition_row(branch: str, abs_step: int, split: str, maps: Mapping[str, Mapping[str, Tensor]]) -> Dict[str, Any]:
    return CAM._decomposition_row(branch, abs_step, split, maps)


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
                        metrics = MIC._metric_images(branch.pipeline.model, maps[view_id]["pred"], maps[view_id]["gt"])
                        per_view_rows.append({"branch": branch_name, "absolute_step": int(step), "split": split, "view_id": view_id, **metrics})
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
    _write_csv(output_dir / "global_rgb_metrics.csv", global_rows)
    _write_json(output_dir / "global_rgb_metrics.json", {"rows": global_rows})
    _write_csv(output_dir / "per_view_rgb_metrics.csv", per_view_rows)
    _write_json(output_dir / "per_view_rgb_metrics.json", {"rows": per_view_rows})
    _write_csv(output_dir / "decomposition_safety.csv", decomp_rows)
    _write_json(output_dir / "decomposition_safety.json", {"rows": decomp_rows})
    _write_csv(output_dir / "global_rgb_deltas.csv", CAM._delta_rows(global_rows, ("absolute_step", "split"), ("PSNR", "SSIM", "LPIPS", "MSE")))
    _write_csv(output_dir / "per_view_rgb_deltas.csv", CAM._delta_rows(per_view_rows, ("absolute_step", "split", "view_id"), ("PSNR", "SSIM", "LPIPS", "MSE")))
    return global_rows, per_view_rows, decomp_rows


def _effective_raw_for_context(
    model: Any,
    camera: Cameras,
    *,
    camera_context_override: Optional[Tensor] = None,
    projector_bundle: Optional[Mapping[str, Any]] = None,
) -> Tuple[Tensor, int, int, Dict[str, Tensor]]:
    raw_ctx, height, width, features = CAM._medium_raw_for_camera(
        model,
        camera,
        camera_context_override=camera_context_override,
        force_real_camera_context=camera_context_override is None,
    )
    if projector_bundle is None:
        return raw_ctx, height, width, features
    zero_context = torch.zeros(3, device=model.device, dtype=raw_ctx.dtype)
    raw_base, _h2, _w2, _features_base = CAM._medium_raw_for_camera(model, camera, camera_context_override=zero_context)
    delta = raw_ctx - raw_base
    raw_eff = raw_base + _apply_projector_to_delta(delta, projector_bundle)
    return raw_eff, height, width, features


def _medium_distribution_audit(
    repo: Path,
    output_dir: Path,
    steps: Sequence[int],
    samples: Mapping[str, MI.ViewSample],
) -> Dict[str, List[Dict[str, Any]]]:
    per_camera_rows: List[Dict[str, Any]] = []
    variance_rows: List[Dict[str, Any]] = []
    between_rows: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        branch = _setup_branch(repo, branch_name)
        try:
            for step in steps:
                print(f"[M1-OCMC] medium distribution {branch_name} step={step}", flush=True)
                bundle = _load_snapshot(branch, output_dir, int(step))
                model = branch.pipeline.model
                records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
                accum: Dict[str, Dict[str, List[Tensor]]] = {
                    pop: {"raw": [], "act": [], "tau": [], "trans": [], "rgb_residual": []} for pop in POPULATIONS
                }
                for view_id, sample in samples.items():
                    _idx, camera, batch = records[view_id]
                    camera = camera.to(model.device)
                    with torch.no_grad():
                        raw, height, width, features = _effective_raw_for_context(model, camera, projector_bundle=bundle if branch_name == "C1" else None)
                        med = CAM._activate_medium(model, raw, height, width)
                        outputs = MI._render_with_medium_override(
                            model,
                            camera,
                            med["medium_rgb"],
                            med["medium_bs"],
                            med["medium_attn"],
                            detach_object_state=True,
                        )
                        gt = PW._get_gt(model, batch, outputs["background"]).detach().float()
                    for pop in POPULATIONS:
                        flat = sample.flat_for(pop)
                        if flat.numel() == 0:
                            continue
                        flat_dev = flat.to(model.device)
                        raw_s = raw.reshape(-1, 9)[flat_dev].detach().float().cpu()
                        act_s = torch.cat(
                            [
                                med["medium_rgb"].reshape(-1, 3)[flat_dev],
                                med["medium_bs"].reshape(-1, 3)[flat_dev],
                                med["medium_attn"].reshape(-1, 3)[flat_dev],
                            ],
                            dim=-1,
                        ).detach().float().cpu()
                        pred = outputs["pred_image"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        target = gt.reshape(-1, 3)[flat_dev].detach().float().cpu()
                        residual = pred - target
                        tau = outputs["tau_D"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        trans = outputs["transmission"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        accum[pop]["raw"].append(raw_s)
                        accum[pop]["act"].append(act_s)
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
                            "effective_raw_uses_ocmc_projector": branch_name == "C1",
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
                    row = {
                        "branch": branch_name,
                        "absolute_step": int(step),
                        "population": pop,
                        "sampled_rays": int(raw_all.shape[0]),
                        "rgb_mse": float(residual_all.square().mean().item()),
                    }
                    row.update(CAM._channel_var_row(raw_all[:, 0:3], "raw_Binf"))
                    row.update(CAM._channel_var_row(raw_all[:, 3:6], "raw_betaB"))
                    row.update(CAM._channel_var_row(raw_all[:, 6:9], "raw_betaD"))
                    row.update(CAM._channel_var_row(act_all[:, 0:3], "B_inf"))
                    row.update(CAM._channel_var_row(act_all[:, 3:6], "beta_B"))
                    row.update(CAM._channel_var_row(act_all[:, 6:9], "beta_D"))
                    row.update(CAM._stats(tau_all, "tau_"))
                    row.update(CAM._stats(trans_all, "T_"))
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
    _write_csv(output_dir / "medium_output_deltas.csv", CAM._delta_rows(variance_rows, ("absolute_step", "population"), ("raw_betaD_variance_pooled", "beta_D_variance_pooled", "rgb_mse")))
    _write_csv(output_dir / "between_within_deltas.csv", CAM._delta_rows(between_rows, ("absolute_step", "population"), ("raw_betaD_r_between_over_within", "raw_betaD_g_between_over_within", "raw_betaD_b_between_over_within")))
    return outputs


def _camera_delta_for_samples(
    model: Any,
    records: Mapping[str, Tuple[int, Cameras, Dict[str, Any]]],
    samples: Mapping[str, MI.ViewSample],
) -> Dict[str, Tensor]:
    chunks: Dict[str, List[Tensor]] = {pop: [] for pop in POPULATIONS}
    for view_id, sample in samples.items():
        _idx, camera, _batch = records[view_id]
        camera = camera.to(model.device)
        with torch.no_grad():
            raw_full, _height, _width, _features = CAM._medium_raw_for_camera(model, camera, force_real_camera_context=True)
            zero_context = torch.zeros(3, device=model.device, dtype=raw_full.dtype)
            raw_base, _h2, _w2, _features_base = CAM._medium_raw_for_camera(model, camera, camera_context_override=zero_context)
            delta = raw_full.reshape(-1, 9) - raw_base.reshape(-1, 9)
        for pop in POPULATIONS:
            flat = sample.flat_for(pop)
            if flat.numel() > 0:
                chunks[pop].append(delta[flat.to(model.device)].detach().float().cpu())
    return {pop: torch.cat(chunks[pop], dim=0) if chunks[pop] else torch.empty(0, 9) for pop in POPULATIONS}


def _projected_decomposition_stats(delta: Tensor, bundle: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    if delta.numel() == 0:
        return {
            f"{prefix}_full_rms": float("nan"),
            f"{prefix}_projected_rms": float("nan"),
            f"{prefix}_suppressed_rms": float("nan"),
            f"{prefix}_observable_over_full": float("nan"),
            f"{prefix}_suppressed_over_full": float("nan"),
        }
    projected = _apply_projector_to_delta(delta, bundle).detach().float().cpu()
    suppressed = delta.detach().float().cpu() - projected
    full_rms = math.sqrt(float(delta.detach().float().square().mean().item()))
    proj_rms = math.sqrt(float(projected.square().mean().item()))
    supp_rms = math.sqrt(float(suppressed.square().mean().item()))
    return {
        f"{prefix}_full_rms": full_rms,
        f"{prefix}_projected_rms": proj_rms,
        f"{prefix}_suppressed_rms": supp_rms,
        f"{prefix}_observable_over_full": proj_rms / max(full_rms, EPS),
        f"{prefix}_suppressed_over_full": supp_rms / max(full_rms, EPS),
    }


def _run_mechanism_audit(
    repo: Path,
    output_dir: Path,
    snapshot_steps: Sequence[int],
    ident_steps: Sequence[int],
    samples: Mapping[str, MI.ViewSample],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[Tuple[str, int, str], Dict[str, Tensor]]]:
    natural_rows: List[Dict[str, Any]] = []
    ident_rows: List[Dict[str, Any]] = []
    weak_rows: List[Dict[str, Any]] = []
    camera_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    nvo_rows: List[Dict[str, Any]] = []
    perturb_rows: List[Dict[str, Any]] = []
    residual_rows: List[Dict[str, Any]] = []
    weak_capacity_rows: List[Dict[str, Any]] = []
    diagnostic_projector_rows: List[Dict[str, Any]] = []
    basis: Dict[Tuple[str, int, str], Dict[str, Tensor]] = {}
    random_dirs = MI._unit_random_directions(SWAP_SEED + 17, RANDOM_DIRECTIONS)

    for branch_name in BRANCHES:
        branch = _setup_branch(repo, branch_name)
        try:
            for step in snapshot_steps:
                print(f"[M1-OCMC] mechanism audit {branch_name} step={step}", flush=True)
                formal_bundle = _load_snapshot(branch, output_dir, int(step))
                analyses, meta = CAM._analyse_loaded_branch(branch, samples)
                records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
                deltas = _camera_delta_for_samples(branch.pipeline.model, records, samples)
                for pop, analysis in analyses.items():
                    diag_bundle = _projector_bundle_from_analysis(
                        analysis,
                        branch=branch_name,
                        step=int(step),
                        population=pop,
                        source="diagnostic_current_branch_checkpoint",
                    )
                    prow = {"branch": branch_name, "absolute_step": int(step), "population": pop}
                    prow.update(_bundle_summary_row(diag_bundle))
                    diagnostic_projector_rows.append(prow)
                    natural_rows.append(MI._natural_stats_rows(branch_name, int(step), int(step), pop, analysis))
                    if int(step) in ident_steps:
                        ident_rows.append(MI._aggregate_rows(branch_name, int(step), int(step), pop, analysis))
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
                            "diagnostic_projector": diag_bundle["projector"].detach().float().cpu(),
                        }
                    delta = deltas[pop]
                    row = {
                        "branch": branch_name,
                        "absolute_step": int(step),
                        "population": pop,
                        "sampled_rays": int(delta.shape[0]),
                        "projector_source": "diagnostic current branch structured Jacobian",
                    }
                    row.update(_projected_decomposition_stats(delta, diag_bundle, "unprojected_camera_delta_diagP"))
                    if branch_name == "C1" and formal_bundle is not None:
                        effective_delta = _apply_projector_to_delta(delta.to(branch.pipeline.model.device), formal_bundle).detach().cpu()
                        row.update(_projected_decomposition_stats(effective_delta, diag_bundle, "effective_camera_delta_diagP"))
                        row["formal_projector_refresh_step"] = int(formal_bundle["step"])
                    else:
                        effective_delta = delta
                        row.update(_projected_decomposition_stats(effective_delta, diag_bundle, "effective_camera_delta_diagP"))
                        row["formal_projector_refresh_step"] = ""
                    residual_rows.append(row)
                    if int(step) in ident_steps:
                        _delta_std, _weak_std, _orth_std, proj = CAM._projection_stats(effective_delta, analysis.scale.detach().double().cpu(), analysis.v_min.detach().double().cpu())
                        weak_capacity_rows.append(
                            {
                                "branch": branch_name,
                                "absolute_step": int(step),
                                "population": pop,
                                "delta_source": "effective camera-conditioned residual after formal OCMC projector for C1; full residual for C0",
                                "sampled_rays": int(effective_delta.shape[0]),
                                **proj,
                            }
                        )
                if int(step) in ident_steps:
                    perturb_rows.extend(CAM._counterfactual_for_branch(branch, analyses, samples, random_dirs))
                    _write_csv(output_dir / "identifiability_summary.csv", ident_rows)
                    _write_json(output_dir / "identifiability_summary.json", {"rows": ident_rows})
        finally:
            _release(branch)

    outputs = {
        "natural_medium_output_statistics": natural_rows,
        "identifiability_summary": ident_rows,
        "weak_mode_summary": weak_rows,
        "camera_context_stability": camera_rows,
        "depth_tau_identifiability_strata": strata_rows,
        "natural_variance_vs_observability": nvo_rows,
        "counterfactual_perturbation": perturb_rows,
        "camera_residual_decomposition": residual_rows,
        "weak_camera_capacity": weak_capacity_rows,
        "diagnostic_projector_summary": diagnostic_projector_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", {"rows": rows})
    _write_csv(output_dir / "camera_residual_decomposition_deltas.csv", CAM._delta_rows(residual_rows, ("absolute_step", "population"), ("effective_camera_delta_diagP_suppressed_over_full", "effective_camera_delta_diagP_full_rms")))
    _write_csv(output_dir / "weak_camera_capacity_deltas.csv", CAM._delta_rows(weak_capacity_rows, ("absolute_step", "population"), ("weak_energy_fraction_mean", "weak_projection_over_random_1over9")))
    _write_csv(output_dir / "identifiability_deltas.csv", CAM._delta_rows(ident_rows, ("nominal_step", "population"), ("sigma_min_over_sigma_max", "condition_number", "effective_rank"), branch_field="run"))
    return outputs, basis


def _camera_swap_and_weak_component_audit(
    repo: Path,
    output_dir: Path,
    steps: Sequence[int],
    samples: Mapping[str, MI.ViewSample],
    basis: Mapping[Tuple[str, int, str], Mapping[str, Tensor]],
) -> Dict[str, List[Dict[str, Any]]]:
    utility_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    projection_rows: List[Dict[str, Any]] = []
    sensitivity_rows: List[Dict[str, Any]] = []
    removal_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    rng = random.Random(SWAP_SEED)

    # The same deterministic real-camera swap bank is shared by both arms.
    setup = _setup_branch(repo, "C0")
    try:
        train_view_ids = [view_id for _idx, view_id, _cam, _batch in _train_records(setup.pipeline)]
    finally:
        _release(setup)
    alt_map: Dict[str, List[str]] = {}
    for view_id in train_view_ids:
        candidates = [v for v in train_view_ids if v != view_id]
        local = candidates[:]
        rng.shuffle(local)
        alt_map[view_id] = local[:ALT_CONTEXT_COUNT]
    _write_json(output_dir / "real_camera_swap_bank.json", {"seed": SWAP_SEED, "alternatives_per_source": ALT_CONTEXT_COUNT, "rows": alt_map})

    for branch_name in BRANCHES:
        branch = _setup_branch(repo, branch_name)
        try:
            for step in steps:
                print(f"[M1-OCMC] camera swap {branch_name} step={step}", flush=True)
                formal_bundle = _load_snapshot(branch, output_dir, int(step))
                model = branch.pipeline.model
                records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
                context_bank = {
                    view_id: CAM._camera_context_for(model, camera.to(model.device), neutral=False).detach()
                    for view_id, (_idx, camera, _batch) in records.items()
                }
                for source_view, sample in samples.items():
                    _idx, camera, batch = records[source_view]
                    camera = camera.to(model.device)
                    with torch.no_grad():
                        raw_correct, height, width, _features = _effective_raw_for_context(
                            model,
                            camera,
                            projector_bundle=formal_bundle if branch_name == "C1" else None,
                        )
                        out_correct = CAM._render_from_raw(model, camera, raw_correct, height, width)
                        gt_correct = PW._get_gt(model, batch, out_correct["background"]).reshape(-1, 3).detach().float().cpu()
                        pred_correct = out_correct["pred_image"].reshape(-1, 3).detach().float().cpu()
                        depth_correct = out_correct["depth"].reshape(-1, 1).detach().float().cpu()
                        tau_correct = out_correct["tau_D"].reshape(-1, 3).detach().float().cpu().mean(dim=-1, keepdim=True)
                    raw_ref_sum = torch.zeros_like(raw_correct.detach())
                    ref_count = 0
                    for alt_view in alt_map[source_view]:
                        with torch.no_grad():
                            raw_swap, _h2, _w2, _ = _effective_raw_for_context(
                                model,
                                camera,
                                camera_context_override=context_bank[alt_view],
                                projector_bundle=formal_bundle if branch_name == "C1" else None,
                            )
                            raw_ref_sum += raw_swap.detach()
                            ref_count += 1
                            out_swap = CAM._render_from_raw(model, camera, raw_swap, height, width)
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
                            med_c = CAM._activate_medium(model, raw_correct, height, width)
                            med_s = CAM._activate_medium(model, raw_swap, height, width)
                            act_delta = torch.cat(
                                [
                                    med_c["medium_rgb"].reshape(-1, 3)[flat_dev] - med_s["medium_rgb"].reshape(-1, 3)[flat_dev],
                                    med_c["medium_bs"].reshape(-1, 3)[flat_dev] - med_s["medium_bs"].reshape(-1, 3)[flat_dev],
                                    med_c["medium_attn"].reshape(-1, 3)[flat_dev] - med_s["medium_attn"].reshape(-1, 3)[flat_dev],
                                ],
                                dim=-1,
                            ).detach().float().cpu()
                            v_min = basis[(branch_name, int(step), pop)]["v_min"]
                            scale = basis[(branch_name, int(step), pop)]["scale"]
                            _delta_std, _weak_std, _orth_std, proj = CAM._projection_stats(raw_delta, scale, v_min)
                            utility_rows.append(
                                {
                                    "branch": branch_name,
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
                            drow = {"branch": branch_name, "absolute_step": int(step), "population": pop, "source_view_id": source_view, "swapped_view_id": alt_view, "sampled_rays": int(flat.numel())}
                            drow.update(CAM._delta_energy(raw_delta, "raw_delta_z_cam"))
                            drow.update(CAM._delta_energy(act_delta, "activated_delta_cam"))
                            delta_rows.append(drow)
                            projection_rows.append({"branch": branch_name, "absolute_step": int(step), "population": pop, "source_view_id": source_view, "swapped_view_id": alt_view, **proj})

                    raw_ref = raw_ref_sum / max(ref_count, 1)
                    with torch.no_grad():
                        out_ref = CAM._render_from_raw(model, camera, raw_ref, height, width)
                        pred_ref = out_ref["pred_image"].reshape(-1, 3).detach().float().cpu()
                    for pop in POPULATIONS:
                        flat = sample.flat_for(pop)
                        if flat.numel() == 0:
                            continue
                        flat_dev = flat.to(model.device)
                        v_min = basis[(branch_name, int(step), pop)]["v_min"]
                        scale = basis[(branch_name, int(step), pop)]["scale"].to(dtype=torch.float32)
                        raw_delta = raw_correct.reshape(-1, 9)[flat_dev].detach().float().cpu() - raw_ref.reshape(-1, 9)[flat_dev].detach().float().cpu()
                        delta_std, weak_std, orth_std, proj = CAM._projection_stats(raw_delta, scale.double(), v_min)
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
                            out_weak = CAM._render_from_raw(model, camera, weak_map, height, width)
                            out_orth = CAM._render_from_raw(model, camera, orth_map, height, width)
                            out_removed = CAM._render_from_raw(model, camera, removed_map, height, width)
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
                                "branch": branch_name,
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
                                "branch": branch_name,
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
                                        "branch": branch_name,
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
                    del raw_correct, raw_ref, raw_ref_sum, out_correct, pred_correct, gt_correct, pred_ref
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
    per_camera = CAM._per_camera_stability(utility_rows, delta_rows, projection_rows, sensitivity_rows, removal_rows)
    for row in per_camera:
        row["branch"] = next(
            (r["branch"] for r in utility_rows if int(r["absolute_step"]) == int(row["absolute_step"]) and r["population"] == row["population"] and r["source_view_id"] == row["source_view_id"]),
            "",
        )
    _write_csv(output_dir / "per_camera_stability.csv", per_camera)
    _write_json(output_dir / "per_camera_stability.json", {"rows": per_camera})
    outputs["per_camera_stability"] = per_camera
    _write_csv(output_dir / "correct_context_utility_deltas.csv", CAM._delta_rows(utility_rows, ("absolute_step", "population", "source_view_id", "swapped_view_id"), ("Delta_E_swap_mean", "fraction_Delta_E_swap_gt_0")))
    _write_csv(output_dir / "weak_component_removal_deltas.csv", CAM._delta_rows(removal_rows, ("absolute_step", "population", "source_view_id"), ("Delta_E_remove_weak_mean", "fraction_remove_weak_improves_or_equal")))
    _write_csv(output_dir / "weak_component_sensitivity_deltas.csv", CAM._delta_rows(sensitivity_rows, ("absolute_step", "population", "source_view_id"), ("weak_over_orth_rgb_change",)))
    return outputs


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals: List[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except Exception:
            continue
        if math.isfinite(value):
            vals.append(value)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _latest_row(rows: Sequence[Mapping[str, Any]], branch: str, step: int, **filters: Any) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if str(row.get("branch", row.get("run", ""))) != branch:
            continue
        if int(row.get("absolute_step", row.get("nominal_step", -1))) != int(step):
            continue
        ok = True
        for key, value in filters.items():
            if str(row.get(key)) != str(value):
                ok = False
                break
        if ok:
            return row
    return None


def _rgb_classification(delta_eval: Mapping[str, float]) -> str:
    if delta_eval["PSNR"] > 0.0 and delta_eval["SSIM"] >= -0.001 and delta_eval["LPIPS"] <= 0.002:
        return "RGB_IMPROVED"
    if delta_eval["PSNR"] >= -0.05 and delta_eval["SSIM"] >= -0.002 and delta_eval["LPIPS"] <= 0.005:
        return "RGB_NEUTRAL"
    return "RGB_DEGRADED"


def _classification(
    summary_metrics: Mapping[str, Any],
    weak_rows: Sequence[Mapping[str, Any]],
    utility_rows: Sequence[Mapping[str, Any]],
    removal_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    residual_rows: Sequence[Mapping[str, Any]],
    between_rows: Sequence[Mapping[str, Any]],
    decomp_rows: Sequence[Mapping[str, Any]],
    final_step: int,
) -> Tuple[str, str, Dict[str, Any]]:
    rgb_class = summary_metrics["rgb_safety_classification"]
    mechanism_checks: Dict[str, Any] = {}
    decreases = 0
    for metric in ("weak_energy_fraction_mean", "weak_projection_over_random_1over9"):
        good = 0
        compared = 0
        for step in IDENTIFIABILITY_STEPS:
            if step > final_step:
                continue
            for pop in POPULATIONS:
                c0 = _latest_row(weak_rows, "C0", step, population=pop)
                c1 = _latest_row(weak_rows, "C1", step, population=pop)
                if c0 and c1:
                    compared += 1
                    if float(c1[metric]) < float(c0[metric]):
                        good += 1
        mechanism_checks[f"{metric}_decreased_pairs"] = good
        mechanism_checks[f"{metric}_compared_pairs"] = compared
        if compared and good >= max(1, math.ceil(0.5 * compared)):
            decreases += 1

    residual_good = 0
    residual_compared = 0
    for step in SNAPSHOT_STEPS:
        if step > final_step:
            continue
        for pop in POPULATIONS:
            c0 = _latest_row(residual_rows, "C0", step, population=pop)
            c1 = _latest_row(residual_rows, "C1", step, population=pop)
            if c0 and c1:
                residual_compared += 1
                if float(c1["effective_camera_delta_diagP_suppressed_over_full"]) < float(c0["effective_camera_delta_diagP_suppressed_over_full"]):
                    residual_good += 1
    mechanism_checks["suppressed_over_full_decreased_pairs"] = residual_good
    mechanism_checks["suppressed_over_full_compared_pairs"] = residual_compared
    if residual_compared and residual_good >= math.ceil(0.5 * residual_compared):
        decreases += 1

    utility_c0 = _mean([r for r in utility_rows if r["branch"] == "C0" and int(r["absolute_step"]) == final_step], "Delta_E_swap_mean")
    utility_c1 = _mean([r for r in utility_rows if r["branch"] == "C1" and int(r["absolute_step"]) == final_step], "Delta_E_swap_mean")
    utility_preserved = bool(math.isfinite(utility_c0) and math.isfinite(utility_c1) and utility_c1 >= utility_c0 - max(abs(utility_c0) * 0.25, 1e-6))
    mechanism_checks["final_correct_context_utility_C0"] = utility_c0
    mechanism_checks["final_correct_context_utility_C1"] = utility_c1
    mechanism_checks["correct_context_utility_preserved"] = utility_preserved

    removal_c0 = _mean([r for r in removal_rows if r["branch"] == "C0" and int(r["absolute_step"]) == final_step], "Delta_E_remove_weak_mean")
    removal_c1 = _mean([r for r in removal_rows if r["branch"] == "C1" and int(r["absolute_step"]) == final_step], "Delta_E_remove_weak_mean")
    mechanism_checks["final_remove_weak_delta_E_C0"] = removal_c0
    mechanism_checks["final_remove_weak_delta_E_C1"] = removal_c1
    mechanism_checks["removable_low_observability_component_decreased"] = bool(math.isfinite(removal_c0) and math.isfinite(removal_c1) and abs(removal_c1) < abs(removal_c0))

    final_c1_decomp = [row for row in decomp_rows if row["branch"] == "C1" and int(row["absolute_step"]) == final_step]
    decomposition_safe = bool(final_c1_decomp and all(float(row.get("P_J_gt_1", 1.0)) == 0.0 for row in final_c1_decomp))
    mechanism_checks["bounded_decomposition_safety_intact"] = decomposition_safe

    expressive_rows = [row for row in between_rows if int(row["absolute_step"]) == final_step and row["population"] == PROJECTOR_POPULATION]
    c0_expr = [row for row in expressive_rows if row["branch"] == "C0"]
    c1_expr = [row for row in expressive_rows if row["branch"] == "C1"]
    expressiveness_preserved = bool(c0_expr and c1_expr)
    if c0_expr and c1_expr:
        for key in ("activated_betaD_r_within_camera_variance", "activated_betaD_g_within_camera_variance", "activated_betaD_b_within_camera_variance"):
            if key in c0_expr[0] and float(c1_expr[0][key]) < 0.1 * max(float(c0_expr[0][key]), EPS):
                expressiveness_preserved = False
    mechanism_checks["camera_expressiveness_preserved"] = expressiveness_preserved

    if decreases >= 2 and utility_preserved and expressiveness_preserved and rgb_class in ("RGB_IMPROVED", "RGB_NEUTRAL") and decomposition_safe:
        ocmc = "OCMC_ACTIONABLE"
    elif decreases >= 1 and decomposition_safe and rgb_class != "RGB_DEGRADED":
        ocmc = "OCMC_PARTIALLY_ACTIONABLE"
    else:
        ocmc = "OCMC_NOT_ACTIONABLE"

    if ocmc == "OCMC_ACTIONABLE":
        capacity = "CAMERA_CONTEXT_CAPACITY_ALLOCATION_VALIDATED"
    elif ocmc == "OCMC_PARTIALLY_ACTIONABLE":
        capacity = "CAMERA_CONTEXT_CAPACITY_ALLOCATION_TENTATIVE"
    else:
        capacity = "CAMERA_CONTEXT_CAPACITY_ALLOCATION_NOT_SUPPORTED"
    return ocmc, capacity, mechanism_checks


def _final_summary(
    output_dir: Path,
    env: Mapping[str, Any],
    gpu: Mapping[str, Any],
    start_audit: Mapping[str, Any],
    camera_audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
    preflight: Mapping[str, Any],
    global_rows: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
    decomp_rows: Sequence[Mapping[str, Any]],
    medium_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    mechanism_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    swap_outputs: Mapping[str, Sequence[Mapping[str, Any]]],
    projector_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    final_step: int,
) -> Dict[str, Any]:
    g = {(row["branch"], int(row["absolute_step"]), row["split"]): row for row in global_rows}
    c0_eval = g[("C0", int(final_step), "eval")]
    c1_eval = g[("C1", int(final_step), "eval")]
    c0_train = g[("C0", int(final_step), "train")]
    c1_train = g[("C1", int(final_step), "train")]
    delta_eval = {k: float(c1_eval[k]) - float(c0_eval[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")}
    delta_train = {k: float(c1_train[k]) - float(c0_train[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")}
    rgb_class = _rgb_classification(delta_eval)
    per_eval_delta = CAM._delta_rows(per_view_rows, ("absolute_step", "split", "view_id"), ("PSNR", "SSIM", "LPIPS", "MSE"))
    per_eval_final = [r for r in per_eval_delta if int(r["absolute_step"]) == int(final_step) and r["split"] == "eval"]
    positive_psnr_views = sum(1 for row in per_eval_final if float(row["delta_C1_minus_C0_PSNR"]) > 0.0)

    train_times = {
        branch: _mean([row for row in training_rows if row["branch"] == branch], "step_time_seconds")
        for branch in BRANCHES
    }
    refresh_time_total = sum(float(row.get("projector_compute_seconds", 0.0)) for row in projector_rows)
    amortized = refresh_time_total / max(len([row for row in training_rows if row["branch"] == "C1"]), 1)
    overhead = {
        "baseline_step_time_seconds": train_times.get("C0", float("nan")),
        "ocmc_step_time_seconds": train_times.get("C1", float("nan")),
        "ocmc_step_time_delta_seconds": train_times.get("C1", float("nan")) - train_times.get("C0", float("nan")),
        "projector_refresh_time_total_seconds": refresh_time_total,
        "projector_refresh_time_mean_seconds": _mean(projector_rows, "projector_compute_seconds"),
        "amortized_projector_overhead_seconds_per_logged_step": amortized,
    }

    summary_metrics = {"rgb_safety_classification": rgb_class}
    ocmc_class, capacity_class, mechanism_checks = _classification(
        summary_metrics,
        mechanism_outputs["weak_camera_capacity"],
        swap_outputs["correct_vs_swapped_context_utility"],
        swap_outputs["weak_component_removal_counterfactual"],
        swap_outputs["weak_component_rgb_sensitivity"],
        mechanism_outputs["camera_residual_decomposition"],
        medium_outputs["between_within_camera_variance"],
        decomp_rows,
        final_step,
    )
    if ocmc_class == "OCMC_ACTIONABLE" and capacity_class == "CAMERA_CONTEXT_CAPACITY_ALLOCATION_VALIDATED":
        next_experiment = "Cross-scene causal validation of M1+OCMC on Curasao with the exact locked OCMC configuration."
    elif ocmc_class == "OCMC_PARTIALLY_ACTIONABLE":
        next_experiment = "One diagnostic to separate projector temporal mismatch from over-suppression of useful context, without a sweep."
    else:
        next_experiment = "Close the current OCMC projector formulation; do not invent a second gate in this task."

    final_c1_decomp = [row for row in decomp_rows if row["branch"] == "C1" and int(row["absolute_step"]) == final_step]
    summary = {
        "experiment": EXPERIMENT,
        "scene": SCENE,
        "CONDA_ENV": env["CONDA_ENV"],
        "PYTHON_PATH": env["PYTHON_PATH"],
        "TORCH_VERSION": env["TORCH_VERSION"],
        "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
        "gpu": dict(gpu),
        "final_step": int(final_step),
        "START_STATE_EQUIVALENCE": bool(start_audit["START_STATE_EQUIVALENCE"]),
        "MODEL_PARAMETER_EQUIVALENCE": bool(start_audit["MODEL_PARAMETER_EQUIVALENCE"]),
        "OPTIMIZER_STATE_EQUIVALENCE": bool(start_audit["OPTIMIZER_STATE_EQUIVALENCE"]),
        "SCHEDULER_STATE_EQUIVALENCE": bool(start_audit["SCHEDULER_STATE_EQUIVALENCE"]),
        "SCALER_STATE_EQUIVALENCE": bool(start_audit["SCALER_STATE_EQUIVALENCE"]),
        "RNG_EQUIVALENCE": bool(start_audit["RNG_EQUIVALENCE"]),
        "CAMERA_SEQUENCE_MATCH": bool(camera_audit["CAMERA_SEQUENCE_MATCH"]),
        "PROJECTOR_UPDATE_PROTOCOL": protocol["PROJECTOR_UPDATE_PROTOCOL"],
        "projector_temporally_stable_preflight": bool(preflight["stable_under_registered_rule"]),
        "projector_refresh_count": len(projector_rows),
        "projector_dynamics": {
            "mean_refresh_seconds": overhead["projector_refresh_time_mean_seconds"],
            "max_relative_change_from_previous": max(
                [float(row.get("projector_relative_change_from_previous", 0.0)) for row in projector_rows if str(row.get("projector_relative_change_from_previous", "")) not in ("", "nan")],
                default=float("nan"),
            ),
            "min_weak_direction_cosine_from_previous": min(
                [float(row.get("weak_direction_abs_cosine_from_previous", 1.0)) for row in projector_rows if str(row.get("weak_direction_abs_cosine_from_previous", "")) not in ("", "nan")],
                default=float("nan"),
            ),
        },
        "final_train_C0": {k: float(c0_train[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_train_C1": {k: float(c1_train[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_train_delta_C1_minus_C0": delta_train,
        "final_eval_C0": {k: float(c0_eval[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_eval_C1": {k: float(c1_eval[k]) for k in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_eval_delta_C1_minus_C0": delta_eval,
        "rgb_safety_classification": rgb_class,
        "final_eval_views_positive_PSNR": int(positive_psnr_views),
        "final_eval_view_count": len(per_eval_final),
        "mechanism_checks": mechanism_checks,
        "decomposition_safety": {
            "C1_P_J_gt_1_all_final_rows_zero": bool(final_c1_decomp and all(float(row.get("P_J_gt_1", 1.0)) == 0.0 for row in final_c1_decomp)),
            "C1_final_rows": final_c1_decomp,
        },
        "computational_overhead": overhead,
        "gradient_pathway_rows": len(gradient_rows),
        "OCMC_CLASSIFICATION": ocmc_class,
        "CAPACITY_ALLOCATION_CLASSIFICATION": capacity_class,
        "next_single_experiment": next_experiment,
    }
    _write_json(output_dir / "final_summary.json", summary)
    _write_csv(output_dir / "final_summary.csv", [{"key": k, "value": v} for k, v in summary.items() if not isinstance(v, (dict, list))])
    _write_json(output_dir / "computational_overhead_summary.json", overhead)
    _write_csv(output_dir / "computational_overhead_summary.csv", [{"key": k, "value": v} for k, v in overhead.items()])
    _write_json(output_dir / "camera_context_utility_summary.json", {"rows": swap_outputs["correct_vs_swapped_context_utility"], "mechanism_checks": mechanism_checks})
    _write_json(output_dir / "weak_capacity_summary.json", {"rows": mechanism_outputs["weak_camera_capacity"], "mechanism_checks": mechanism_checks})
    _write_json(output_dir / "final_train_eval_table.json", {"train_C0": summary["final_train_C0"], "train_C1": summary["final_train_C1"], "eval_C0": summary["final_eval_C0"], "eval_C1": summary["final_eval_C1"], "eval_delta": delta_eval})
    _write_json(output_dir / "per_view_result_table.json", {"rows": per_eval_final})
    return summary


def _write_research_note(repo: Path, output_dir: Path, summary: Mapping[str, Any], source: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    delta = summary["final_eval_delta_C1_minus_C0"]
    train_delta = summary["final_train_delta_C1_minus_C0"]
    checks = summary["mechanism_checks"]
    lines = [
        "# M1-OCMC-CAUSAL-IUI3",
        "",
        "## CODE FACT",
        "OCMC is implemented in `water_splatting/fields/medium_field.py` as a detached 9-D raw-medium projector on the camera-conditioned residual.",
        "The current camera context is scene-normalized camera position, not a learned latent.",
        f"Implemented equations: `{source['implemented_ocmc_equations']}`.",
        "",
        "## CONFIG FACT",
        "Both arms use `bounded_sh3`, SH degree 3, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, and `infinite_water_enabled=False`.",
        "C0 keeps camera context and disables OCMC. C1 keeps the same camera context and enables OCMC with strength `1.0`.",
        f"Projector protocol: `{protocol['PROJECTOR_UPDATE_PROTOCOL']}` with refresh steps `{protocol['refresh_steps']}` and population `{protocol['population']}`.",
        "",
        "## EXPERIMENTAL FACT",
        f"Start-state equivalence: `{summary['START_STATE_EQUIVALENCE']}`.",
        f"Camera sequence match: `{summary['CAMERA_SEQUENCE_MATCH']}`.",
        f"Outputs: `{output_dir}`.",
        "",
        "## QUANTITATIVE RESULT",
        f"Final train C1-C0: PSNR `{train_delta['PSNR']:.6f}` dB, SSIM `{train_delta['SSIM']:.6f}`, LPIPS `{train_delta['LPIPS']:.6f}`, MSE `{train_delta['MSE']:.8f}`.",
        f"Final eval C1-C0: PSNR `{delta['PSNR']:.6f}` dB, SSIM `{delta['SSIM']:.6f}`, LPIPS `{delta['LPIPS']:.6f}`, MSE `{delta['MSE']:.8f}`.",
        f"RGB safety classification: `{summary['rgb_safety_classification']}`.",
        f"Mechanism checks: `{checks}`.",
        f"Decomposition safety C1 P(J>1)=0: `{summary['decomposition_safety']['C1_P_J_gt_1_all_final_rows_zero']}`.",
        "",
        "## INFERENCE",
        f"OCMC classification: `{summary['OCMC_CLASSIFICATION']}`.",
        f"Capacity-allocation classification: `{summary['CAPACITY_ALLOCATION_CLASSIFICATION']}`.",
        "No true-color, true-medium, or true-geometry claim is made.",
        "",
        "## HYPOTHESIS",
        f"Next single experiment: {summary['next_single_experiment']}",
        "",
    ]
    path = repo / RESEARCH_NOTE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _prepare_output_dir(output_dir: Path, allow_existing: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise RuntimeError(f"Output directory exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _snapshot_steps(final_step: int) -> Tuple[int, ...]:
    steps = tuple(step for step in SNAPSHOT_STEPS if step <= final_step)
    if final_step not in steps:
        steps = tuple(sorted(set(steps + (final_step,))))
    return steps


def _ident_steps(final_step: int) -> Tuple[int, ...]:
    steps = tuple(step for step in IDENTIFIABILITY_STEPS if step <= final_step)
    if final_step >= 5000 and final_step not in steps:
        steps = tuple(sorted(set(steps + (final_step,))))
    return steps


def run(repo: Path, output_dir: Path, final_step: int, allow_existing_output: bool, phase_a_output_dir: Path) -> Dict[str, Any]:
    gpu = _assert_runtime_policy()
    repo = repo.resolve()
    output_dir = _repo_path(output_dir, repo)
    _prepare_output_dir(output_dir, allow_existing_output)
    env = _environment_manifest(gpu)
    _write_json(output_dir / "gpu_manifest.json", gpu)
    _write_json(output_dir / "environment_manifest.json", env)
    _write_json(output_dir / "repo_manifest.json", _repo_manifest(repo))
    shutil.copy2(repo / BND_CONFIG, output_dir / "source_bnd_config.yml")
    source = _source_semantics(output_dir)

    snapshot_steps = _snapshot_steps(int(final_step))
    ident_steps = _ident_steps(int(final_step))

    print("[M1-OCMC] building deterministic train samples", flush=True)
    samples, sampling_meta, sampling_rows = MI._build_samples(repo, output_dir, SAMPLES_PER_VIEW, SAMPLE_SEED)
    _write_csv(output_dir / "deterministic_ray_sampling.csv", sampling_rows)
    _write_json(output_dir / "deterministic_ray_sampling.json", sampling_meta)

    preflight, protocol = _projector_stability_preflight(repo, output_dir, phase_a_output_dir, samples)

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
    all_projectors: List[Dict[str, Any]] = []
    all_gradients: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        print(f"[M1-OCMC] training {branch_name}", flush=True)
        rows, events, ckpts, topology, projectors, gradients = _train_branch(
            repo,
            branch_name,
            camera_indices=camera_indices,
            camera_names=camera_names,
            training_rng=training_rng,
            snapshot_steps=snapshot_steps,
            output_dir=output_dir,
            samples=samples,
            protocol=protocol,
        )
        all_training.extend(rows)
        all_events.extend(events)
        all_ckpts.extend(ckpts)
        all_topology.extend(topology)
        all_projectors.extend(projectors)
        all_gradients.extend(gradients)
    _write_csv(output_dir / "training_metrics.csv", all_training)
    _write_json(output_dir / "training_metrics.json", {"rows": all_training})
    _write_csv(output_dir / "refinement_events.csv", all_events)
    _write_json(output_dir / "refinement_events.json", {"rows": all_events})
    _write_csv(output_dir / "checkpoint_manifest.csv", all_ckpts)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": all_ckpts})
    _write_csv(output_dir / "gaussian_population.csv", all_topology)
    _write_json(output_dir / "gaussian_population.json", {"rows": all_topology})
    _write_csv(output_dir / "projector_dynamics.csv", all_projectors)
    _write_json(output_dir / "projector_dynamics.json", {"rows": all_projectors})
    _write_csv(output_dir / "gradient_pathway.csv", all_gradients)
    _write_json(output_dir / "gradient_pathway.json", {"rows": all_gradients})

    print("[M1-OCMC] evaluating train/eval snapshots", flush=True)
    global_rows, per_view_rows, decomp_rows = _evaluate_snapshots(repo, output_dir, snapshot_steps)

    medium_outputs = _medium_distribution_audit(repo, output_dir, snapshot_steps, samples)
    mechanism_outputs, weak_basis = _run_mechanism_audit(repo, output_dir, snapshot_steps, ident_steps, samples)
    swap_outputs = _camera_swap_and_weak_component_audit(repo, output_dir, ident_steps, samples, weak_basis)

    summary = _final_summary(
        output_dir,
        env,
        gpu,
        start_audit,
        camera_audit,
        protocol,
        preflight,
        global_rows,
        per_view_rows,
        decomp_rows,
        medium_outputs,
        mechanism_outputs,
        swap_outputs,
        all_projectors,
        all_training,
        all_gradients,
        int(final_step),
    )
    _write_research_note(repo, output_dir, summary, source, protocol)
    return summary


def run_analysis_only(repo: Path, output_dir: Path, final_step: int, phase_a_output_dir: Path) -> Dict[str, Any]:
    gpu = _assert_runtime_policy()
    repo = repo.resolve()
    output_dir = _repo_path(output_dir, repo)
    if not output_dir.exists():
        raise FileNotFoundError(f"Missing output directory for analysis-only mode: {output_dir}")
    env = _environment_manifest(gpu)
    _write_json(output_dir / "gpu_manifest.json", gpu)
    _write_json(output_dir / "environment_manifest.json", env)
    _write_json(output_dir / "repo_manifest.json", _repo_manifest(repo))
    source = _source_semantics(output_dir)
    snapshot_steps = _snapshot_steps(int(final_step))
    ident_steps = _ident_steps(int(final_step))
    _require_existing_checkpoints(output_dir, snapshot_steps)
    start_audit = json.loads((output_dir / "start_state_audit.json").read_text(encoding="utf8"))
    camera_audit = json.loads((output_dir / "camera_sequence_audit.json").read_text(encoding="utf8"))
    protocol = json.loads((output_dir / "registered_projector_update_protocol.json").read_text(encoding="utf8"))
    preflight = json.loads((output_dir / "projector_stability_preflight.json").read_text(encoding="utf8"))
    samples, sampling_meta, sampling_rows = MI._build_samples(repo, output_dir, SAMPLES_PER_VIEW, SAMPLE_SEED)
    _write_csv(output_dir / "deterministic_ray_sampling.csv", sampling_rows)
    _write_json(output_dir / "deterministic_ray_sampling.json", sampling_meta)
    global_rows, per_view_rows, decomp_rows = _evaluate_snapshots(repo, output_dir, snapshot_steps)
    medium_outputs = _medium_distribution_audit(repo, output_dir, snapshot_steps, samples)
    mechanism_outputs, weak_basis = _run_mechanism_audit(repo, output_dir, snapshot_steps, ident_steps, samples)
    swap_outputs = _camera_swap_and_weak_component_audit(repo, output_dir, ident_steps, samples, weak_basis)
    projector_rows = _read_csv(output_dir / "projector_dynamics.csv") if (output_dir / "projector_dynamics.csv").exists() else []
    training_rows = _read_csv(output_dir / "training_metrics.csv") if (output_dir / "training_metrics.csv").exists() else []
    gradient_rows = _read_csv(output_dir / "gradient_pathway.csv") if (output_dir / "gradient_pathway.csv").exists() else []
    summary = _final_summary(
        output_dir,
        env,
        gpu,
        start_audit,
        camera_audit,
        protocol,
        preflight,
        global_rows,
        per_view_rows,
        decomp_rows,
        medium_outputs,
        mechanism_outputs,
        swap_outputs,
        projector_rows,
        training_rows,
        gradient_rows,
        int(final_step),
    )
    _write_research_note(repo, output_dir, summary, source, protocol)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--phase-a-output-dir", type=Path, default=PHASE_A_OUTPUT_DIR)
    parser.add_argument("--final-step", type=int, default=FINAL_ACTUAL_STEP)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis_only:
        summary = run_analysis_only(args.repo, args.output_dir, int(args.final_step), args.phase_a_output_dir)
    else:
        summary = run(args.repo, args.output_dir, int(args.final_step), bool(args.allow_existing_output), args.phase_a_output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
