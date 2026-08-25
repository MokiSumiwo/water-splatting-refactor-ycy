#!/usr/bin/env python3
"""Formal BND-MIC causal continuation experiment on IUI3.

This driver starts both arms from the same formal BND@3000 checkpoint, replays
one explicit train-camera sequence, and changes only the C1 loss by adding the
pre-registered beta_D raw variance MIC term.
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
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.eval_utils import eval_setup

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI


PW = MI.PW

SCENE = "IUI3-RedSea"
OUTPUT_DIR = Path("outputs/bnd_mic_causal_iui3_20260825")
RESEARCH_NOTE = Path("research_notes/BND_MIC_CAUSAL_IUI3_2026-08-25.md")

START_NOMINAL_STEP = 3000
FINAL_NOMINAL_STEP = 15000
SNAPSHOT_ABS_NOMINAL = (3000, 5000, 8000, 10000, 13000, 15000)
IDENTIFIABILITY_ABS_NOMINAL = (5000, 10000, 15000)
BRANCHES = ("C0", "C1")
POPULATIONS = ("GENERAL", "M_SAFE")
ALLOWED_PHYSICAL_GPUS = frozenset({"6", "7", "8", "9"})
MIC_WEIGHT = 1.5118506741538569
MIC_TARGET = "beta_D_raw_variance"
TRAINING_RNG_SEED = 202608251
SAMPLES_PER_VIEW = MI.SAMPLES_PER_VIEW
SAMPLE_SEED = MI.RNG_SEED
COUNTERFACTUAL_RANDOM_SEED = SAMPLE_SEED + 404
COUNTERFACTUAL_EPSILON = MI.COUNTERFACTUAL_EPSILON
LOG_INTERVAL = 500
CAMERA_CONTEXT_SAMPLE_RAYS = 256
QUANTILE_MAX_N = 1_000_000
EPS = 1e-12

RGB_SAFE_RULE = {
    "dPSNR_min": -0.05,
    "dSSIM_min": -0.002,
    "dLPIPS_max": 0.005,
    "registered_before_training": True,
}


@dataclass
class LoadedBranch:
    branch: str
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
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


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _environment_manifest(gpu_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "CONDA_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu": dict(gpu_manifest),
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


def _available_steps(config_path: Path) -> Dict[int, Path]:
    ckpt_dir = config_path.parent / "nerfstudio_models"
    out: Dict[int, Path] = {}
    if ckpt_dir.exists():
        for path in ckpt_dir.glob("step-*.ckpt"):
            try:
                out[int(path.stem.split("-")[1])] = path
            except Exception:
                continue
    return out


def _actual_step(config_path: Path, nominal_step: int) -> int:
    steps = _available_steps(config_path)
    if nominal_step in steps:
        return int(nominal_step)
    if nominal_step == FINAL_NOMINAL_STEP and 14999 in steps:
        return 14999
    raise FileNotFoundError(f"Missing checkpoint step {nominal_step} for {config_path}; available={sorted(steps)}")


def _snapshot_abs_steps(final_actual_step: int) -> Tuple[int, ...]:
    out: List[int] = []
    for nominal in SNAPSHOT_ABS_NOMINAL:
        actual = final_actual_step if nominal == FINAL_NOMINAL_STEP else int(nominal)
        if START_NOMINAL_STEP <= actual <= final_actual_step:
            out.append(actual)
    out.append(final_actual_step)
    return tuple(dict.fromkeys(out))


def _identifiability_abs_steps(final_actual_step: int) -> Tuple[int, ...]:
    out: List[int] = []
    for nominal in IDENTIFIABILITY_ABS_NOMINAL:
        actual = final_actual_step if nominal == FINAL_NOMINAL_STEP else int(nominal)
        if START_NOMINAL_STEP <= actual <= final_actual_step:
            out.append(actual)
    out.append(final_actual_step)
    return tuple(dict.fromkeys(out))


def _rel(abs_step: int) -> int:
    return int(abs_step) - START_NOMINAL_STEP


def _snapshot_rel_steps(final_actual_step: int) -> Tuple[int, ...]:
    return tuple(_rel(step) for step in _snapshot_abs_steps(final_actual_step))


def _optimizer_groups(config: Any, model: Any) -> Dict[str, Any]:
    groups = model.get_param_groups()
    return {name: config.optimizers[name] for name in groups}


def _configure_formal_model(model: Any, branch: str) -> None:
    enabled = branch == "C1"
    model.config.intrinsic_color_parameterization = "bounded_sh3"
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.coarse_depth_supervision_enabled = False
    model.config.medium_identifiability_enabled = enabled
    model.config.medium_identifiability_weight = float(MIC_WEIGHT if enabled else 0.0)
    model.config.medium_identifiability_start_step = 0
    model.config.medium_identifiability_end_step = -1
    model.config.medium_identifiability_target = MIC_TARGET


def _load_branch(repo: Path, branch: str, step: int = START_NOMINAL_STEP) -> LoadedBranch:
    config_path = repo / PW.BND_CONFIG
    actual = _actual_step(config_path, step)
    mic_enabled = branch == "C1"

    def update_config(config: Any) -> Any:
        config.load_step = actual
        model_cfg = config.pipeline.model
        model_cfg.intrinsic_color_parameterization = "bounded_sh3"
        model_cfg.rasterize_mode = "classic"
        model_cfg.medium_context_mode = "dir_xy_camera"
        model_cfg.b_inf_mode = "tied"
        model_cfg.infinite_water_enabled = False
        model_cfg.coarse_depth_supervision_enabled = False
        model_cfg.medium_identifiability_enabled = mic_enabled
        model_cfg.medium_identifiability_weight = float(MIC_WEIGHT if mic_enabled else 0.0)
        model_cfg.medium_identifiability_start_step = 0
        model_cfg.medium_identifiability_end_step = -1
        model_cfg.medium_identifiability_target = MIC_TARGET
        config.pipeline.datamanager.load_depths = False
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    model = pipeline.model
    _configure_formal_model(model, branch)
    model.step = int(loaded_step)
    optimizers = Optimizers(_optimizer_groups(config, model), model.get_param_groups())
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    for group in optimizers.optimizers:
        optimizers.optimizers[group].load_state_dict(ckpt["optimizers"][group])
    for group in optimizers.schedulers:
        optimizers.schedulers[group].load_state_dict(ckpt["schedulers"][group])
    pipeline.eval()
    return LoadedBranch(
        branch=branch,
        config_path=config_path,
        checkpoint_path=Path(checkpoint_path),
        loaded_step=int(loaded_step),
        config=config,
        pipeline=pipeline,
        optimizers=optimizers,
        scalers=ckpt.get("scalers", {}),
    )


def _release(obj: Optional[LoadedBranch]) -> None:
    if obj is None:
        return
    try:
        del obj.pipeline
    except Exception:
        pass
    try:
        del obj.optimizers
    except Exception:
        pass
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _train_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return PW._records(pipeline)["train"]


def _eval_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return PW._records(pipeline)["eval"]


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


def _with_rng_preserved(func: Any) -> Any:
    state = _rng_state()
    try:
        return func()
    finally:
        _set_rng_state(state)


def _max_abs_diff(a: Tensor, b: Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float((a.detach().float() - b.detach().float()).abs().max().cpu().item())


def _flatten_module_params(module: Any) -> Tensor:
    pieces = [p.detach().reshape(-1).cpu() for p in module.parameters()]
    return torch.cat(pieces) if pieces else torch.empty(0)


def _model_param_tensors(model: Any) -> Dict[str, Tensor]:
    out = {
        name: getattr(model, name).detach().cpu().clone()
        for name in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")
    }
    out["medium_mlp"] = _flatten_module_params(model.medium_mlp)
    out["direction_encoding"] = _flatten_module_params(model.direction_encoding)
    return out


def _optimizer_state_tensors(optimizers: Optimizers) -> Dict[str, Dict[str, Tensor]]:
    out: Dict[str, Dict[str, Tensor]] = {}
    for group, optimizer in optimizers.optimizers.items():
        pieces: Dict[str, List[Tensor]] = {"exp_avg": [], "exp_avg_sq": [], "step": []}
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state.get(param, {})
                for key in pieces:
                    if key not in state:
                        continue
                    value = state[key]
                    if isinstance(value, Tensor):
                        pieces[key].append(value.detach().reshape(-1).cpu().float())
                    else:
                        pieces[key].append(torch.tensor([float(value)], dtype=torch.float32))
        out[group] = {key: torch.cat(vals) if vals else torch.empty(0) for key, vals in pieces.items()}
    return out


def _scheduler_state_tensors(optimizers: Optimizers) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    for group, scheduler in optimizers.schedulers.items():
        pieces: List[Tensor] = []
        for value in scheduler.state_dict().values():
            if isinstance(value, Tensor):
                pieces.append(value.detach().reshape(-1).cpu().float())
            elif isinstance(value, (int, float, bool)):
                pieces.append(torch.tensor([float(value)], dtype=torch.float32))
            elif isinstance(value, list) and all(isinstance(x, (int, float, bool)) for x in value):
                pieces.append(torch.tensor([float(x) for x in value], dtype=torch.float32))
        out[group] = torch.cat(pieces) if pieces else torch.empty(0)
    return out


def _compare_tensor_dict(a: Mapping[str, Tensor], b: Mapping[str, Tensor], name_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            rows.append({name_key: name, "max_abs_diff": float("nan"), "pass": False})
            continue
        if a[name].shape != b[name].shape:
            rows.append(
                {
                    name_key: name,
                    "max_abs_diff": float("nan"),
                    "pass": False,
                    "shape_C0": list(a[name].shape),
                    "shape_C1": list(b[name].shape),
                }
            )
            continue
        diff = _max_abs_diff(a[name], b[name]) if a[name].numel() else 0.0
        rows.append({name_key: name, "shape": list(a[name].shape), "max_abs_diff": diff, "pass": bool(diff == 0.0)})
    return rows


def _compare_nested_state_dict(
    a: Mapping[str, Mapping[str, Tensor]],
    b: Mapping[str, Mapping[str, Tensor]],
    group_key: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in sorted(set(a) | set(b)):
        keys = sorted(set(a.get(group, {})) | set(b.get(group, {})))
        for key in keys:
            if group not in a or group not in b or key not in a.get(group, {}) or key not in b.get(group, {}):
                rows.append({group_key: group, "state_key": key, "max_abs_diff": float("nan"), "pass": False})
                continue
            ta, tb = a[group][key], b[group][key]
            if ta.shape != tb.shape:
                rows.append(
                    {
                        group_key: group,
                        "state_key": key,
                        "shape_C0": list(ta.shape),
                        "shape_C1": list(tb.shape),
                        "max_abs_diff": float("nan"),
                        "pass": False,
                    }
                )
                continue
            diff = _max_abs_diff(ta, tb) if ta.numel() else 0.0
            rows.append({group_key: group, "state_key": key, "shape": list(ta.shape), "max_abs_diff": diff, "pass": bool(diff == 0.0)})
    return rows


def _manual_mic_loss_from_raw(raw: Tensor) -> Tensor:
    beta_d_raw = raw.reshape(-1, 9)[..., 6:9]
    mean = beta_d_raw.mean(dim=0, keepdim=True).detach()
    return (beta_d_raw - mean).square().mean()


def _config_snapshot(branch: LoadedBranch) -> Dict[str, Any]:
    model = branch.pipeline.model
    cfg = model.config
    keys = (
        "intrinsic_color_parameterization",
        "sh_degree",
        "rasterize_mode",
        "medium_context_mode",
        "b_inf_mode",
        "infinite_water_enabled",
        "coarse_depth_supervision_enabled",
        "medium_identifiability_enabled",
        "medium_identifiability_weight",
        "medium_identifiability_start_step",
        "medium_identifiability_end_step",
        "medium_identifiability_target",
        "medium_camera_context_scale",
        "medium_camera_context_dropout",
        "refine_every",
        "stop_split_at",
        "reset_alpha_every",
        "continue_cull_post_densification",
    )
    return {
        "branch": branch.branch,
        "config_path": str(branch.config_path),
        "checkpoint_path": str(branch.checkpoint_path),
        "loaded_step": int(branch.loaded_step),
        "optimizer_groups": sorted(branch.optimizers.optimizers.keys()),
        "scheduler_groups": sorted(branch.optimizers.schedulers.keys()),
        "model_config": {key: getattr(cfg, key, None) for key in keys},
    }


def _start_state_equivalence(repo: Path, output_dir: Path, training_rng: Mapping[str, Any]) -> Dict[str, Any]:
    c0: Optional[LoadedBranch] = None
    c1: Optional[LoadedBranch] = None
    try:
        c0 = _load_branch(repo, "C0")
        c1 = _load_branch(repo, "C1")
        _write_json(output_dir / "config_snapshot_C0.json", _config_snapshot(c0))
        _write_json(output_dir / "config_snapshot_C1.json", _config_snapshot(c1))

        param_rows = _compare_tensor_dict(
            _model_param_tensors(c0.pipeline.model),
            _model_param_tensors(c1.pipeline.model),
            "parameter_group",
        )
        opt_rows = _compare_nested_state_dict(
            _optimizer_state_tensors(c0.optimizers),
            _optimizer_state_tensors(c1.optimizers),
            "optimizer_group",
        )
        sched_rows = _compare_tensor_dict(
            _scheduler_state_tensors(c0.optimizers),
            _scheduler_state_tensors(c1.optimizers),
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

        records0 = _train_records(c0.pipeline)
        records1 = _train_records(c1.pipeline)
        idx0, view0, camera0, batch0 = records0[0]
        idx1, view1, camera1, batch1 = records1[0]
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
            raw0, h0, w0 = MI._medium_raw_for_camera(model0, camera0)
            raw1, h1, w1 = MI._medium_raw_for_camera(model1, camera1)
            loss0 = model0.get_loss_dict(out0, batch0, {})["main_loss"]
            loss1 = model1.get_loss_dict(out1, batch1, {})["main_loss"]

        forward_rows: List[Dict[str, Any]] = []
        key_map = (
            ("rendered RGB", "pred_image"),
            ("depth", "depth"),
            ("accumulation", "accumulation"),
            ("medium_rgb", "medium_rgb"),
            ("medium_bs", "medium_bs"),
            ("medium_attn", "medium_attn"),
            ("b_inf", "b_inf"),
            ("beta_B", "medium_bs"),
            ("beta_D", "medium_attn"),
        )
        for label, key in key_map:
            diff = _max_abs_diff(out0[key], out1[key])
            forward_rows.append({"quantity": label, "source_key": key, "max_abs_diff": diff, "pass": bool(diff == 0.0)})
        raw_diff = _max_abs_diff(raw0.view(h0, w0, 9), raw1.view(h1, w1, 9))
        main_diff = abs(float(loss0.detach().cpu().item()) - float(loss1.detach().cpu().item()))
        forward_rows.append({"quantity": "raw z_med", "source_key": "manual_medium_mlp_raw", "max_abs_diff": raw_diff, "pass": bool(raw_diff == 0.0)})
        forward_rows.append({"quantity": "main RGB loss", "source_key": "loss_dict.main_loss", "max_abs_diff": main_diff, "pass": bool(main_diff == 0.0)})

        _write_csv(output_dir / "start_state_forward_equivalence.csv", forward_rows)
        _write_json(output_dir / "start_state_forward_equivalence.json", {"rows": forward_rows})
        _write_csv(output_dir / "start_state_parameter_equivalence.csv", param_rows)
        _write_json(output_dir / "start_state_parameter_equivalence.json", {"rows": param_rows})
        _write_csv(output_dir / "start_state_optimizer_equivalence.csv", opt_rows)
        _write_json(output_dir / "start_state_optimizer_equivalence.json", {"rows": opt_rows})
        _write_csv(output_dir / "start_state_scheduler_equivalence.csv", sched_rows)
        _write_json(output_dir / "start_state_scheduler_equivalence.json", {"rows": sched_rows})
        _write_csv(output_dir / "start_state_scaler_equivalence.csv", scaler_rows)
        _write_json(output_dir / "start_state_scaler_equivalence.json", {"rows": scaler_rows})

        rng_a = _rng_manifest(training_rng)
        _set_rng_state(training_rng)
        rng_b = _rng_manifest(_rng_state())
        rng_rows = [
            {
                "quantity": key,
                "C0_planned_hash": value,
                "C1_after_reset_hash": rng_b.get(key),
                "pass": value == rng_b.get(key),
            }
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
        _write_csv(output_dir / "start_state_rng_equivalence.csv", rng_rows)
        _write_json(output_dir / "start_state_rng_equivalence.json", {"rows": rng_rows})

        all_pass = (
            all(row["pass"] for row in forward_rows)
            and all(row["pass"] for row in param_rows)
            and all(row["pass"] for row in opt_rows)
            and all(row["pass"] for row in sched_rows)
            and all(row["pass"] for row in scaler_rows)
            and all(row["pass"] for row in rng_rows)
        )
        payload = {
            "START_STATE_EQUIVALENCE": bool(all_pass),
            "probe_train_index": int(idx0),
            "probe_view_id": view0,
            "model_parameter_equivalence": all(row["pass"] for row in param_rows),
            "optimizer_state_equivalence": all(row["pass"] for row in opt_rows),
            "scheduler_state_equivalence": all(row["pass"] for row in sched_rows),
            "scaler_state_equivalence": all(row["pass"] for row in scaler_rows),
            "rng_state_equivalence": all(row["pass"] for row in rng_rows),
            "forward_equivalence": all(row["pass"] for row in forward_rows),
            "max_forward_diff": max(float(row["max_abs_diff"]) for row in forward_rows),
            "raw_z_med_shape": [int(h0), int(w0), 9],
        }
        _write_json(output_dir / "start_state_audit.json", payload)
        if not payload["START_STATE_EQUIVALENCE"]:
            raise RuntimeError("START_STATE_EQUIVALENCE=false; stopping before training.")
        return payload
    finally:
        _release(c0)
        _release(c1)


def _config_diff_audit(output_dir: Path) -> Dict[str, Any]:
    c0 = json.loads((output_dir / "config_snapshot_C0.json").read_text(encoding="utf8"))
    c1 = json.loads((output_dir / "config_snapshot_C1.json").read_text(encoding="utf8"))
    c0_cfg = c0["model_config"]
    c1_cfg = c1["model_config"]
    allowed = {
        "medium_identifiability_enabled",
        "medium_identifiability_weight",
    }
    diffs = []
    for key in sorted(set(c0_cfg) | set(c1_cfg)):
        if c0_cfg.get(key) != c1_cfg.get(key):
            diffs.append({"key": key, "C0": c0_cfg.get(key), "C1": c1_cfg.get(key), "allowed": key in allowed})
    payload = {
        "ONLY_INTERVENTION_MIC": all(item["allowed"] for item in diffs),
        "allowed_difference": "C1 adds lambda_mic * beta_D_raw_variance MIC loss; C0 leaves MIC disabled.",
        "lambda_mic": MIC_WEIGHT,
        "medium_identifiability_target": MIC_TARGET,
        "forbidden_interventions_enabled": [],
        "diffs": diffs,
    }
    _write_json(output_dir / "config_diff_audit.json", payload)
    return payload


def _generate_camera_sequence(branch: LoadedBranch, output_dir: Path, final_actual_step: int) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
    dm = branch.pipeline.datamanager
    filenames = list(getattr(dm.train_dataset, "image_filenames", []))
    names = [Path(path).stem for path in filenames]
    rows: List[Dict[str, Any]] = []
    indices: List[int] = []
    view_ids: List[str] = []
    for abs_step in range(START_NOMINAL_STEP + 1, final_actual_step + 1):
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        index = int(dm.train_unseen_cameras.pop(0))
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        view_id = names[index]
        indices.append(index)
        view_ids.append(view_id)
        rows.append(
            {
                "relative_step": _rel(abs_step),
                "absolute_step": abs_step,
                "camera_index": index,
                "camera_name": view_id,
            }
        )
    payload = {
        "scene": SCENE,
        "length": len(rows),
        "start_abs_step_exclusive": START_NOMINAL_STEP,
        "final_abs_step_inclusive": int(final_actual_step),
        "rows": rows,
    }
    encoded = json.dumps(rows, sort_keys=True).encode("utf8")
    audit = {
        "CAMERA_SEQUENCE_MATCH": True,
        "CAMERA_SEQUENCE_EXACT_MATCH": True,
        "mismatch_count": 0,
        "length": len(rows),
        "sha256": _sha256_bytes(encoded),
        "basis": "Both arms consume this explicit camera_index list; datamanager random sampling is not used inside branch training.",
    }
    _write_json(output_dir / "paired_camera_sequence.json", payload)
    _write_json(output_dir / "camera_sequence_audit.json", audit)
    _write_csv(output_dir / "paired_camera_sequence.csv", rows)
    return indices, view_ids, rows


def _run_before(model: Any, optimizers: Optimizers, abs_step: int) -> None:
    model.step_cb(step=abs_step)
    model.aopt_before_train_iteration(optimizers, step=abs_step)
    model.medium_hold_before_train_iteration(optimizers, step=abs_step)


def _run_after(model: Any, optimizers: Optimizers, abs_step: int) -> Mapping[str, Any]:
    model.aopt_after_train_iteration(step=abs_step)
    model.medium_hold_after_train_iteration(optimizers, step=abs_step)
    model.after_train(step=abs_step)
    if abs_step % int(model.config.refine_every) == 0:
        model.refinement_after(optimizers, step=abs_step)
        return dict(getattr(model, "_refinement_last_event", {}))
    return {
        "step": abs_step,
        "refinement_called": False,
        "priority_mode": getattr(model.config, "refinement_priority_mode", "baseline"),
        "N_after": int(model.means.shape[0]),
    }


def _optimizer_lrs(optimizers: Optimizers) -> Dict[str, float]:
    return {group: float(opt.param_groups[0]["lr"]) for group, opt in optimizers.optimizers.items()}


def _grad_stats_for_params(params: Iterable[torch.nn.Parameter]) -> Dict[str, Any]:
    sq_sum = 0.0
    abs_sum = 0.0
    count = 0
    max_abs = 0.0
    has_grad = False
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        has_grad = True
        sq_sum += float(grad.square().sum().item())
        abs_sum += float(grad.abs().sum().item())
        count += int(grad.numel())
        max_abs = max(max_abs, float(grad.abs().max().item()) if grad.numel() else 0.0)
    return {
        "grad_l2": math.sqrt(sq_sum),
        "grad_mean_abs": abs_sum / max(count, 1),
        "grad_max_abs": max_abs,
        "has_grad": bool(has_grad and sq_sum > 0.0),
    }


def _param_group_grad_stats(model: Any) -> Dict[str, Dict[str, Any]]:
    groups = model.get_param_groups()
    out = {name: _grad_stats_for_params(params) for name, params in groups.items()}
    medium_params: List[torch.nn.Parameter] = []
    for name in ("direction_encoding", "medium_mlp"):
        medium_params.extend(groups.get(name, []))
    out["medium_branch"] = _grad_stats_for_params(medium_params)
    return out


def _compute_loss_components(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any]) -> Dict[str, Tensor]:
    gt_img = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
    pred_img = outputs["pred_image"]
    if "mask" in batch:
        mask = model._downscale_if_required(batch["mask"]).to(model.device)
        gt_img = gt_img * mask
        pred_img = pred_img * mask
    recon = torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
    sim = 1 - model.ssim(
        (gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
        (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
    )
    return {"reg_l1": recon.detach(), "reg_ssim": sim.detach()}


def _mic_metric_no_grad(model: Any, camera: Cameras) -> float:
    with torch.no_grad():
        raw, _height, _width = MI._medium_raw_for_camera(model, camera)
        return float(_manual_mic_loss_from_raw(raw).detach().cpu().item())


def _gradient_probe(branch: LoadedBranch, camera: Cameras, batch: Mapping[str, Any]) -> Dict[str, Any]:
    model = branch.pipeline.model

    def run_probe() -> Dict[str, Any]:
        branch.optimizers.zero_grad_all()
        outputs = model.get_outputs(camera)
        metrics: Dict[str, Tensor] = {}
        losses = model.get_loss_dict(outputs, batch, metrics)
        rgb_loss = losses["main_loss"]
        rgb_loss.backward()
        rgb_stats = _param_group_grad_stats(model)

        branch.optimizers.zero_grad_all()
        if branch.branch == "C1":
            outputs = model.get_outputs(camera)
            mic_raw = model._medium_identifiability_loss(outputs)
            mic_raw.backward()
            mic_stats = _param_group_grad_stats(model)
            mic_raw_value = float(mic_raw.detach().cpu().item())
        else:
            mic_stats = {key: {"grad_l2": 0.0, "grad_mean_abs": 0.0, "grad_max_abs": 0.0, "has_grad": False} for key in rgb_stats}
            mic_raw_value = _mic_metric_no_grad(model, camera)

        branch.optimizers.zero_grad_all()
        rgb_medium = rgb_stats["medium_branch"]["grad_l2"]
        mic_medium = mic_stats["medium_branch"]["grad_l2"]
        return {
            "probe_enabled": True,
            "probe_L_RGB": float(rgb_loss.detach().cpu().item()),
            "probe_L_MIC_raw_candidate": mic_raw_value,
            "probe_L_MIC_weighted": float(MIC_WEIGHT * mic_raw_value) if branch.branch == "C1" else 0.0,
            "probe_medium_RGB_grad_l2": rgb_medium,
            "probe_medium_MIC_grad_l2": mic_medium,
            "probe_grad_MIC_over_RGB": mic_medium / max(rgb_medium, EPS),
            "probe_grad_stats_RGB": rgb_stats,
            "probe_grad_stats_MIC": mic_stats,
        }

    return _with_rng_preserved(run_probe)


def _topology_snapshot(branch: LoadedBranch, rel_step: int) -> Dict[str, Any]:
    model = branch.pipeline.model
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacities.detach().float())
        scale = torch.exp(model.scales.detach().float())
    return {
        "branch": branch.branch,
        "relative_step": int(rel_step),
        "absolute_step": START_NOMINAL_STEP + int(rel_step),
        "gaussian_count": int(model.means.shape[0]),
        "mean_opacity": float(opacity.mean().cpu().item()),
        "p99_opacity": float(torch.quantile(opacity.reshape(-1).cpu(), 0.99).item()),
        "mean_scale": float(scale.mean().cpu().item()),
        "max_scale": float(scale.max().cpu().item()),
        "p99_scale": float(torch.quantile(scale.reshape(-1).cpu(), 0.99).item()),
    }


def _save_checkpoint(branch: LoadedBranch, rel_step: int, output_dir: Path) -> Path:
    path = output_dir / "continuation_checkpoints" / branch.branch / f"relative-{rel_step:06d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "branch": branch.branch,
            "relative_step": int(rel_step),
            "absolute_step": START_NOMINAL_STEP + int(rel_step),
            "pipeline": branch.pipeline.state_dict(),
            "optimizers": {group: opt.state_dict() for group, opt in branch.optimizers.optimizers.items()},
            "schedulers": {group: sched.state_dict() for group, sched in branch.optimizers.schedulers.items()},
            "scalers": dict(branch.scalers),
            "metadata": {
                "experiment": "BND-MIC-CAUSAL-IUI3",
                "medium_identifiability_enabled": branch.branch == "C1",
                "medium_identifiability_weight": MIC_WEIGHT if branch.branch == "C1" else 0.0,
                "medium_identifiability_target": MIC_TARGET,
                "matched_camera_sequence": True,
                "normal_topology_enabled": True,
            },
        },
        path,
    )
    return path


def _ckpt_path(output_dir: Path, branch: str, rel_step: int) -> Path:
    return output_dir / "continuation_checkpoints" / branch / f"relative-{rel_step:06d}.ckpt"


def _load_snapshot(branch: LoadedBranch, output_dir: Path, rel_step: int) -> None:
    ckpt = torch.load(_ckpt_path(output_dir, branch.branch, rel_step), map_location="cpu")
    branch.pipeline.load_pipeline(ckpt["pipeline"], int(ckpt["absolute_step"]))
    model = branch.pipeline.model
    model.step = int(ckpt["absolute_step"])
    _configure_formal_model(model, branch.branch)
    branch.pipeline.eval()


def _train_branch(
    repo: Path,
    branch_name: str,
    *,
    camera_indices: Sequence[int],
    camera_names: Sequence[str],
    training_rng: Mapping[str, Any],
    snapshot_rels: Sequence[int],
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    branch = _load_branch(repo, branch_name)
    _set_rng_state(training_rng)
    model = branch.pipeline.model
    _configure_formal_model(model, branch_name)
    dm = branch.pipeline.datamanager
    cached_train = dm.cached_train
    train_cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
    snapshot_set = set(int(x) for x in snapshot_rels)
    training_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    ckpt_rows: List[Dict[str, Any]] = []
    topology_rows: List[Dict[str, Any]] = []
    try:
        if 0 in snapshot_set:
            ckpt = _save_checkpoint(branch, 0, output_dir)
            ckpt_rows.append({"branch": branch_name, "relative_step": 0, "absolute_step": START_NOMINAL_STEP, "checkpoint_path": str(ckpt)})
            topology_rows.append(_topology_snapshot(branch, 0))

        for rel_step, (camera_index, camera_name) in enumerate(zip(camera_indices, camera_names), start=1):
            abs_step = START_NOMINAL_STEP + rel_step
            branch.pipeline.train()
            model.train()
            _configure_formal_model(model, branch_name)
            _run_before(model, branch.optimizers, abs_step)
            branch.optimizers.zero_grad_all()
            batch = _batch_to_device(cached_train[camera_index].copy(), model.device)
            camera = train_cameras[camera_index : camera_index + 1]
            lrs = _optimizer_lrs(branch.optimizers)
            should_probe = rel_step == 1 or abs_step % LOG_INTERVAL == 0 or rel_step in snapshot_set
            probe = _gradient_probe(branch, camera, batch) if should_probe else {"probe_enabled": False}

            outputs = model.get_outputs(camera)
            components = _compute_loss_components(model, outputs, batch)
            metrics: Dict[str, Tensor] = {}
            losses = model.get_loss_dict(outputs, batch, metrics)
            total_loss = sum(losses.values())
            if not bool(torch.isfinite(total_loss).detach().cpu().item()):
                raise RuntimeError(f"Non-finite loss in {branch_name} rel_step={rel_step}")
            total_loss.backward()
            total_grad_stats = _param_group_grad_stats(model)

            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(abs_step)
            event = _run_after(model, branch.optimizers, abs_step)

            mic_weighted = losses.get("medium_identifiability_loss", total_loss.new_tensor(0.0))
            mic_raw_value = metrics.get("medium_identifiability_loss_raw")
            if mic_raw_value is None:
                mic_raw_float = float(probe.get("probe_L_MIC_raw_candidate", 0.0)) if should_probe else 0.0
            else:
                mic_raw_float = float(mic_raw_value.detach().cpu().item())
            row: Dict[str, Any] = {
                "branch": branch_name,
                "relative_step": rel_step,
                "absolute_step": abs_step,
                "camera_index": int(camera_index),
                "camera_name": camera_name,
                "L_total": float(total_loss.detach().cpu().item()),
                "L_RGB": float(losses["main_loss"].detach().cpu().item()),
                "L_MIC_raw": mic_raw_float,
                "lambda_mic": float(MIC_WEIGHT if branch_name == "C1" else 0.0),
                "L_MIC_weighted": float(mic_weighted.detach().cpu().item()),
                "MIC_enabled": branch_name == "C1",
                "reg_l1": float(components["reg_l1"].detach().cpu().item()),
                "reg_ssim": float(components["reg_ssim"].detach().cpu().item()),
                "medium_total_grad_l2": float(total_grad_stats["medium_branch"]["grad_l2"]),
                "gaussian_count": int(model.means.shape[0]),
                "mean_opacity": float(torch.sigmoid(model.opacities.detach()).mean().cpu().item()),
                "mean_scale": float(torch.exp(model.scales.detach()).mean().cpu().item()),
                "max_scale": float(torch.exp(model.scales.detach()).max().cpu().item()),
                "stable": True,
                "probe_enabled": bool(probe.get("probe_enabled", False)),
                "medium_RGB_grad_l2": probe.get("probe_medium_RGB_grad_l2", ""),
                "MIC_medium_grad_l2": probe.get("probe_medium_MIC_grad_l2", ""),
                "grad_MIC_over_RGB": probe.get("probe_grad_MIC_over_RGB", ""),
            }
            for group, lr in lrs.items():
                row[f"lr_{group}"] = lr
            training_rows.append(row)

            if should_probe:
                probe_row = {
                    "branch": branch_name,
                    "relative_step": rel_step,
                    "absolute_step": abs_step,
                    "camera_name": camera_name,
                    **{key: value for key, value in probe.items() if not key.startswith("probe_grad_stats")},
                }
                _write_json(output_dir / "latest_gradient_probe.json", probe_row)

            if event.get("refinement_called"):
                event = dict(event)
                event["branch"] = branch_name
                event["relative_step"] = rel_step
                event["absolute_step"] = abs_step
                event["camera_name"] = camera_name
                event_rows.append(event)

            if abs_step % LOG_INTERVAL == 0 or rel_step in snapshot_set:
                topology_rows.append(_topology_snapshot(branch, rel_step))

            if rel_step in snapshot_set:
                ckpt = _save_checkpoint(branch, rel_step, output_dir)
                ckpt_rows.append({"branch": branch_name, "relative_step": rel_step, "absolute_step": abs_step, "checkpoint_path": str(ckpt)})
                _write_csv(output_dir / f"{branch_name.lower()}_training_log.csv", training_rows)
                _write_json(output_dir / f"{branch_name.lower()}_training_log.json", {"rows": training_rows})
                _write_csv(output_dir / f"{branch_name.lower()}_refinement_events.csv", event_rows)
                _write_json(output_dir / f"{branch_name.lower()}_refinement_events.json", {"rows": event_rows})
        return training_rows, event_rows, ckpt_rows, topology_rows
    finally:
        _release(branch)


def _safe_outputs(outputs: Mapping[str, Any]) -> Dict[str, Tensor]:
    keys = (
        "pred_image",
        "background",
        "rgb_object",
        "direct_object_signal",
        "rgb_medium",
        "rgb_medium_finite",
        "rgb_tail",
        "accumulation",
        "clear_object_fullsh_raw",
        "rgb_clear",
        "rgb_clear_clamp",
        "depth",
        "tau_D",
        "transmission",
        "medium_rgb",
        "medium_bs",
        "medium_attn",
        "b_inf",
        "gaussian_view_rgb",
        "gaussian_view_logits",
        "gaussian_visible_mask",
    )
    return {key: outputs[key].detach().float().cpu() for key in keys if isinstance(outputs.get(key), Tensor)}


def _gt_for(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return model.composite_with_background(model.get_gt_img(batch["image"]), background.to(model.device)).detach().float().cpu()


def _render_records(pipeline: Any, records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]]) -> Dict[str, Dict[str, Tensor]]:
    model = pipeline.model
    model.eval()
    out: Dict[str, Dict[str, Tensor]] = {}
    for _idx, view_id, camera, batch in records:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
            safe = _safe_outputs(outputs)
            gt = _gt_for(model, batch, outputs["background"])
        pred = safe["pred_image"].clamp(0.0, 1.0)
        gt = gt.clamp(0.0, 1.0)
        residual = (pred - gt).square().mean(dim=-1)
        out[view_id] = {
            **safe,
            "gt": gt,
            "pred": pred,
            "residual": residual,
        }
    return out


def _metric_images(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0).to(model.device)
    gt = gt.detach().float().clamp(0.0, 1.0).to(model.device)
    pred_nchw = torch.moveaxis(pred, -1, 0)[None, ...]
    gt_nchw = torch.moveaxis(gt, -1, 0)[None, ...]
    mse = float(((pred - gt) ** 2).mean().item())
    return {
        "PSNR": float(model.psnr(gt_nchw, pred_nchw).item()),
        "SSIM": float(model.ssim(gt_nchw, pred_nchw).item()),
        "LPIPS": float(model.lpips(gt_nchw, pred_nchw).item()),
        "MSE": mse,
    }


def _quantile_flat(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if flat.numel() > QUANTILE_MAX_N:
        idx = torch.linspace(0, flat.numel() - 1, QUANTILE_MAX_N, device=flat.device).long()
        idx = idx.clamp(0, flat.numel() - 1)
        flat = flat[idx]
    return float(torch.quantile(flat.cpu(), q).item())


def _stats(values: Tensor, prefix: str = "") -> Dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in ("mean", "std", "p01", "p50", "p90", "p99", "max")}
    return {
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        f"{prefix}p01": _quantile_flat(flat, 0.01),
        f"{prefix}p50": _quantile_flat(flat, 0.50),
        f"{prefix}p90": _quantile_flat(flat, 0.90),
        f"{prefix}p99": _quantile_flat(flat, 0.99),
        f"{prefix}max": float(flat.max().item()),
    }


def _channel_var_row(values: Tensor, prefix: str) -> Dict[str, Any]:
    names = ("r", "g", "b")
    out: Dict[str, Any] = {
        f"{prefix}_variance_pooled": float(values.detach().float().reshape(-1).var(unbiased=False).item()) if values.numel() else float("nan"),
        f"{prefix}_std_pooled": float(values.detach().float().reshape(-1).std(unbiased=False).item()) if values.numel() else float("nan"),
    }
    for idx, name in enumerate(names):
        vals = values[:, idx].detach().float() if values.numel() else torch.empty(0)
        out[f"{prefix}_{name}_variance"] = float(vals.var(unbiased=False).item()) if vals.numel() else float("nan")
        out[f"{prefix}_{name}_std"] = float(vals.std(unbiased=False).item()) if vals.numel() else float("nan")
        out[f"{prefix}_{name}_mean"] = float(vals.mean().item()) if vals.numel() else float("nan")
    return out


def _decomposition_row(branch: str, rel_step: int, split: str, maps: Mapping[str, Mapping[str, Tensor]]) -> Dict[str, Any]:
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
        "relative_step": int(rel_step),
        "absolute_step": START_NOMINAL_STEP + int(rel_step),
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


def _evaluate_snapshots(
    repo: Path,
    output_dir: Path,
    snapshot_rels: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    global_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        branch = _load_branch(repo, branch_name)
        try:
            for rel_step in snapshot_rels:
                _load_snapshot(branch, output_dir, rel_step)
                for split, records in (("train", _train_records(branch.pipeline)), ("eval", _eval_records(branch.pipeline))):
                    maps = _render_records(branch.pipeline, records)
                    metric_accum: Dict[str, List[float]] = {"PSNR": [], "SSIM": [], "LPIPS": [], "MSE": []}
                    for _idx, view_id, _camera, _batch in records:
                        metrics = _metric_images(branch.pipeline.model, maps[view_id]["pred"], maps[view_id]["gt"])
                        per_row = {
                            "branch": branch_name,
                            "relative_step": int(rel_step),
                            "absolute_step": START_NOMINAL_STEP + int(rel_step),
                            "split": split,
                            "view_id": view_id,
                            **metrics,
                        }
                        per_view_rows.append(per_row)
                        for key, value in metrics.items():
                            metric_accum[key].append(float(value))
                    global_row: Dict[str, Any] = {
                        "branch": branch_name,
                        "relative_step": int(rel_step),
                        "absolute_step": START_NOMINAL_STEP + int(rel_step),
                        "split": split,
                        "view_count": len(metric_accum["PSNR"]),
                    }
                    for key, vals in metric_accum.items():
                        global_row[key] = float(sum(vals) / len(vals)) if vals else float("nan")
                    global_rows.append(global_row)
                    decomp_rows.append(_decomposition_row(branch_name, int(rel_step), split, maps))
                    del maps
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            _release(branch)
    return global_rows, per_view_rows, decomp_rows


def _analyse_loaded_branch(
    loaded: LoadedBranch,
    samples: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model = loaded.pipeline.model
    model.eval()
    records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(loaded.pipeline)}
    store = MI._init_store()
    view_slices: Dict[str, Dict[str, Tuple[int, int]]] = {pop: {} for pop in POPULATIONS}
    per_view_elapsed: List[Dict[str, Any]] = []
    for view_ord, (view_id, sample) in enumerate(samples.items()):
        if view_id not in records:
            raise RuntimeError(f"Missing train view {view_id} in loaded pipeline records")
        _idx, camera, batch = records[view_id]
        t0 = time.time()
        raw_base, height, width = MI._medium_raw_for_camera(model, camera)
        if (height, width) != (sample.height, sample.width):
            raise RuntimeError(f"View size changed for {view_id}: {(height, width)} vs {(sample.height, sample.width)}")
        raw = raw_base.detach().clone().requires_grad_(True)
        med = MI._activate_medium(model, raw, height, width)
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
        per_view_elapsed.append(
            {
                "view_id": view_id,
                "ordinal": view_ord,
                "seconds": time.time() - t0,
                "union_sampled_rays": int(union_flat.numel()),
            }
        )
        del raw_base, raw, med, outputs, grad_union, grad_parts, gt
        gc.collect()
        torch.cuda.empty_cache()

    analyses = {pop: MI._finalize_population(store[pop], view_slices[pop]) for pop in POPULATIONS}
    meta = {
        "branch": loaded.branch,
        "loaded_step": int(loaded.pipeline.model.step),
        "gaussian_count": int(model.means.shape[0]),
        "per_view_elapsed": per_view_elapsed,
        "detached_object_state": True,
        "structured_jacobian_source": "same code path as Phase-A audit_bnd_medium_identifiability_iui3.py",
    }
    return analyses, meta


def _far_high_tau_rows(branch: str, abs_step: int, analyses: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pop, analysis in analyses.items():
        for basis, values in (("depth", analysis.depth[:, 0]), ("tau", analysis.tau.mean(dim=-1))):
            q1 = torch.quantile(values.float(), 1.0 / 3.0)
            q2 = torch.quantile(values.float(), 2.0 / 3.0)
            strata = (
                ("near_or_low", values <= q1),
                ("middle", (values > q1) & (values <= q2)),
                ("far_or_high", values > q2),
            )
            for name, mask in strata:
                local_j = analysis.local_jacobian[mask]
                ratio, cond, erank, v = MI._gram_for_subset(local_j, analysis.scale)
                raw_beta = analysis.z[mask, 6:9]
                act_beta = analysis.activated[mask, 6:9]
                rgb_mse = float(analysis.rgb_residual[mask].square().mean().item()) if int(mask.sum().item()) > 0 else float("nan")
                row: Dict[str, Any] = {
                    "branch": branch,
                    "absolute_step": int(abs_step),
                    "relative_step": _rel(abs_step),
                    "population": pop,
                    "stratification_basis": basis,
                    "stratum": name,
                    "q1": float(q1.item()),
                    "q2": float(q2.item()),
                    "sampled_rays": int(mask.sum().item()),
                    "sigma_min_over_sigma_max": ratio,
                    "condition_number": cond,
                    "effective_rank": erank,
                    "rgb_mse": rgb_mse,
                    "weak_mode_family": MI._dominant_family(v) if torch.isfinite(v).all() else "WEAK_MODE_UNSTABLE",
                }
                row.update(_channel_var_row(raw_beta, "raw_betaD"))
                row.update(_channel_var_row(act_beta, "activated_betaD"))
                rows.append(row)
    return rows


def _counterfactual_ratio_rows(counter_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keyed = {
        (row["run"], int(row["nominal_step"]), row["population"], row["direction_label"]): row
        for row in counter_rows
    }
    rows: List[Dict[str, Any]] = []
    for branch in BRANCHES:
        steps = sorted({int(row["nominal_step"]) for row in counter_rows if row["run"] == branch})
        for step in steps:
            for pop in POPULATIONS:
                vmin = keyed.get((branch, step, pop, "v_min"))
                vmax = keyed.get((branch, step, pop, "v_max"))
                if not vmin or not vmax:
                    continue
                rows.append(
                    {
                        "branch": branch,
                        "absolute_step": step,
                        "relative_step": _rel(step),
                        "population": pop,
                        "vmin_mean_abs_rgb_change": float(vmin["mean_abs_rendered_rgb_change"]),
                        "vmax_mean_abs_rgb_change": float(vmax["mean_abs_rendered_rgb_change"]),
                        "vmin_over_vmax_rgb_change": float(vmin["mean_abs_rendered_rgb_change"]) / max(float(vmax["mean_abs_rendered_rgb_change"]), EPS),
                        "vmin_rms_medium_output_change_9d": float(vmin["rms_medium_output_change_9d"]),
                        "vmin_rms_beta_D_change": float(vmin["rms_beta_D_change"]),
                    }
                )
    return rows


def _run_identifiability_audit(
    repo: Path,
    output_dir: Path,
    ident_rels: Sequence[int],
    samples: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    natural_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    weak_rows: List[Dict[str, Any]] = []
    camera_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    nvo_rows: List[Dict[str, Any]] = []
    counter_rows: List[Dict[str, Any]] = []
    far_rows: List[Dict[str, Any]] = []
    checkpoint_meta: List[Dict[str, Any]] = []
    random_dirs = MI._unit_random_directions(COUNTERFACTUAL_RANDOM_SEED)
    for branch_name in BRANCHES:
        branch = _load_branch(repo, branch_name)
        try:
            for rel_step in ident_rels:
                abs_step = START_NOMINAL_STEP + int(rel_step)
                print(f"[BND-MIC] identifiability audit {branch_name} abs_step={abs_step}", flush=True)
                _load_snapshot(branch, output_dir, int(rel_step))
                analyses, meta = _analyse_loaded_branch(branch, samples)
                checkpoint_meta.append({"branch": branch_name, "absolute_step": abs_step, "relative_step": int(rel_step), **meta})
                for pop, analysis in analyses.items():
                    natural_rows.append(MI._natural_stats_rows(branch_name, abs_step, abs_step, pop, analysis))
                    aggregate_rows.append(MI._aggregate_rows(branch_name, abs_step, abs_step, pop, analysis))
                    weak_rows.append(MI._weak_mode_row(branch_name, abs_step, abs_step, pop, analysis))
                    camera_rows.extend(MI._camera_rows(branch_name, abs_step, abs_step, pop, analysis))
                    strata_rows.extend(MI._strata_rows(branch_name, abs_step, abs_step, pop, analysis))
                    nvo_rows.extend(MI._natural_variance_observability_rows(branch_name, abs_step, abs_step, pop, analysis))
                far_rows.extend(_far_high_tau_rows(branch_name, abs_step, analyses))
                counter_loaded = SimpleNamespace(
                    run=branch_name,
                    nominal_step=abs_step,
                    loaded_step=abs_step,
                    pipeline=branch.pipeline,
                )
                counter_rows.extend(MI._counterfactual_for_checkpoint(counter_loaded, analyses, samples, random_dirs))
                _write_csv(output_dir / "identifiability_summary.csv", aggregate_rows)
                _write_json(output_dir / "identifiability_summary.json", {"rows": aggregate_rows})
                _write_csv(output_dir / "counterfactual_weak_mode_sensitivity.csv", _counterfactual_ratio_rows(counter_rows))
                _write_json(output_dir / "counterfactual_perturbation.json", {"rows": counter_rows})
        finally:
            _release(branch)

    counter_ratio = _counterfactual_ratio_rows(counter_rows)
    outputs = {
        "natural_medium_output_statistics": natural_rows,
        "identifiability_summary": aggregate_rows,
        "weak_mode_summary": weak_rows,
        "camera_context_stability": camera_rows,
        "depth_tau_stratification": strata_rows,
        "natural_variance_vs_observability": nvo_rows,
        "counterfactual_perturbation": counter_rows,
        "counterfactual_weak_mode_sensitivity": counter_ratio,
        "far_high_tau_analysis": far_rows,
        "identifiability_checkpoint_meta": checkpoint_meta,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", {"rows": rows})
    return outputs


def _sample_map(value: Tensor, flat: Tensor) -> Tensor:
    if flat.numel() == 0:
        return value.new_empty((0, value.shape[-1] if value.ndim == 3 else 1))
    return value.reshape(-1, *value.shape[2:])[flat.to(value.device)]


def _simple_medium_sample_audit(
    repo: Path,
    output_dir: Path,
    snapshot_rels: Sequence[int],
    samples: Mapping[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    per_camera_rows: List[Dict[str, Any]] = []
    variance_rows: List[Dict[str, Any]] = []
    responsibility_rows: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        branch = _load_branch(repo, branch_name)
        try:
            for rel_step in snapshot_rels:
                abs_step = START_NOMINAL_STEP + int(rel_step)
                print(f"[BND-MIC] medium variance audit {branch_name} abs_step={abs_step}", flush=True)
                _load_snapshot(branch, output_dir, int(rel_step))
                model = branch.pipeline.model
                records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
                accum: Dict[str, Dict[str, List[Tensor]]] = {
                    pop: {
                        "raw_beta": [],
                        "act_beta": [],
                        "b_inf": [],
                        "beta_b": [],
                        "tau": [],
                        "transmission": [],
                        "accumulation": [],
                        "direct_object_signal": [],
                        "rgb_medium": [],
                        "rgb_residual": [],
                    }
                    for pop in POPULATIONS
                }
                for view_id, sample in samples.items():
                    _idx, camera, batch = records[view_id]
                    with torch.no_grad():
                        raw, height, width = MI._medium_raw_for_camera(model, camera)
                        med = MI._activate_medium(model, raw, height, width)
                        outputs = model.get_outputs_for_camera(camera.to(model.device))
                        gt = PW._get_gt(model, batch, outputs["background"]).detach().float()
                    for pop in POPULATIONS:
                        flat = sample.flat_for(pop)
                        if flat.numel() == 0:
                            continue
                        flat_dev = flat.to(model.device)
                        raw_beta = raw.reshape(-1, 9)[flat_dev, 6:9].detach().float().cpu()
                        act_beta = med["medium_attn"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        b_inf = med["b_inf"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        beta_b = med["medium_bs"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        tau = outputs["tau_D"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        trans = outputs["transmission"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        acc = outputs["accumulation"].reshape(-1, 1)[flat_dev].detach().float().cpu()
                        direct = outputs["direct_object_signal"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        rgb_medium = outputs["rgb_medium"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        pred = outputs["pred_image"].reshape(-1, 3)[flat_dev].detach().float().cpu()
                        gt_flat = gt.reshape(-1, 3)[flat_dev].detach().float().cpu()
                        residual = pred - gt_flat
                        for key, value in (
                            ("raw_beta", raw_beta),
                            ("act_beta", act_beta),
                            ("b_inf", b_inf),
                            ("beta_b", beta_b),
                            ("tau", tau),
                            ("transmission", trans),
                            ("accumulation", acc),
                            ("direct_object_signal", direct),
                            ("rgb_medium", rgb_medium),
                            ("rgb_residual", residual),
                        ):
                            accum[pop][key].append(value)
                        row: Dict[str, Any] = {
                            "branch": branch_name,
                            "absolute_step": abs_step,
                            "relative_step": int(rel_step),
                            "population": pop,
                            "view_id": view_id,
                            "sampled_rays": int(flat.numel()),
                            "raw_betaD_r_mean": float(raw_beta[:, 0].mean().item()),
                            "raw_betaD_g_mean": float(raw_beta[:, 1].mean().item()),
                            "raw_betaD_b_mean": float(raw_beta[:, 2].mean().item()),
                            "activated_betaD_r_mean": float(act_beta[:, 0].mean().item()),
                            "activated_betaD_g_mean": float(act_beta[:, 1].mean().item()),
                            "activated_betaD_b_mean": float(act_beta[:, 2].mean().item()),
                            "B_inf_r_mean": float(b_inf[:, 0].mean().item()),
                            "B_inf_g_mean": float(b_inf[:, 1].mean().item()),
                            "B_inf_b_mean": float(b_inf[:, 2].mean().item()),
                            "beta_B_r_mean": float(beta_b[:, 0].mean().item()),
                            "beta_B_g_mean": float(beta_b[:, 1].mean().item()),
                            "beta_B_b_mean": float(beta_b[:, 2].mean().item()),
                            "tau_r_mean": float(tau[:, 0].mean().item()),
                            "tau_g_mean": float(tau[:, 1].mean().item()),
                            "tau_b_mean": float(tau[:, 2].mean().item()),
                            "rgb_mse": float(residual.square().mean().item()),
                            "accumulation_mean": float(acc.mean().item()),
                            "direct_object_signal_l2_mean": float(torch.linalg.norm(direct, dim=-1).mean().item()),
                            "rgb_medium_l2_mean": float(torch.linalg.norm(rgb_medium, dim=-1).mean().item()),
                        }
                        row.update(_channel_var_row(raw_beta, "raw_betaD"))
                        row.update(_channel_var_row(act_beta, "activated_betaD"))
                        per_camera_rows.append(row)
                for pop in POPULATIONS:
                    if not accum[pop]["raw_beta"]:
                        continue
                    raw_beta = torch.cat(accum[pop]["raw_beta"], dim=0)
                    act_beta = torch.cat(accum[pop]["act_beta"], dim=0)
                    b_inf = torch.cat(accum[pop]["b_inf"], dim=0)
                    beta_b = torch.cat(accum[pop]["beta_b"], dim=0)
                    tau = torch.cat(accum[pop]["tau"], dim=0)
                    trans = torch.cat(accum[pop]["transmission"], dim=0)
                    residual = torch.cat(accum[pop]["rgb_residual"], dim=0)
                    row = {
                        "branch": branch_name,
                        "absolute_step": abs_step,
                        "relative_step": int(rel_step),
                        "population": pop,
                        "sampled_rays": int(raw_beta.shape[0]),
                        "rgb_mse": float(residual.square().mean().item()),
                    }
                    row.update(_channel_var_row(raw_beta, "raw_betaD"))
                    row.update(_channel_var_row(act_beta, "activated_betaD"))
                    row.update(_stats(b_inf, "B_inf_"))
                    row.update(_stats(beta_b, "beta_B_"))
                    row.update(_stats(act_beta, "beta_D_"))
                    row.update(_stats(tau, "tau_"))
                    row.update(_stats(trans, "T_"))
                    variance_rows.append(row)
                    responsibility_rows.append(
                        {
                            "branch": branch_name,
                            "absolute_step": abs_step,
                            "relative_step": int(rel_step),
                            "population": pop,
                            "accumulation_mean": float(torch.cat(accum[pop]["accumulation"], dim=0).mean().item()),
                            "direct_object_signal_l2_mean": float(
                                torch.linalg.norm(torch.cat(accum[pop]["direct_object_signal"], dim=0), dim=-1).mean().item()
                            ),
                            "rgb_medium_l2_mean": float(torch.linalg.norm(torch.cat(accum[pop]["rgb_medium"], dim=0), dim=-1).mean().item()),
                            "rgb_mse": float(residual.square().mean().item()),
                        }
                    )
        finally:
            _release(branch)
    camera_summary = _camera_expressiveness_summary(per_camera_rows)
    outputs = {
        "per_camera_medium_expressiveness": per_camera_rows,
        "raw_betad_contextual_variance": variance_rows,
        "camera_expressiveness_summary": camera_summary,
        "responsibility_context": responsibility_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", {"rows": rows})
    return outputs


def _camera_expressiveness_summary(per_camera_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = {}
    for row in per_camera_rows:
        grouped.setdefault((str(row["branch"]), int(row["absolute_step"]), str(row["population"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (branch, abs_step, pop), rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        item: Dict[str, Any] = {
            "branch": branch,
            "absolute_step": abs_step,
            "relative_step": _rel(abs_step),
            "population": pop,
            "camera_count": len(rows),
        }
        for basis in ("raw_betaD", "activated_betaD"):
            for channel in ("r", "g", "b"):
                means = torch.tensor([float(row[f"{basis}_{channel}_mean"]) for row in rows], dtype=torch.float64)
                within = torch.tensor([float(row[f"{basis}_{channel}_variance"]) for row in rows], dtype=torch.float64)
                across = float(means.var(unbiased=False).item()) if means.numel() else float("nan")
                within_mean = float(within.mean().item()) if within.numel() else float("nan")
                item[f"{basis}_{channel}_across_camera_variance"] = across
                item[f"{basis}_{channel}_within_camera_spatial_variance"] = within_mean
                item[f"{basis}_{channel}_between_over_within_ratio"] = across / max(within_mean, EPS)
        out.append(item)
    return out


def _clone_camera(camera: Cameras) -> Cameras:
    def maybe_clone(value: Any) -> Any:
        return value.clone() if isinstance(value, Tensor) else value

    return Cameras(
        camera_to_worlds=camera.camera_to_worlds.clone(),
        fx=maybe_clone(camera.fx),
        fy=maybe_clone(camera.fy),
        cx=maybe_clone(camera.cx),
        cy=maybe_clone(camera.cy),
        width=maybe_clone(camera.width),
        height=maybe_clone(camera.height),
        distortion_params=maybe_clone(camera.distortion_params),
        camera_type=maybe_clone(camera.camera_type),
        times=maybe_clone(camera.times),
        metadata=camera.metadata,
    )


def _camera_context_sensitivity(
    repo: Path,
    output_dir: Path,
    ident_rels: Sequence[int],
    samples: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        branch = _load_branch(repo, branch_name)
        try:
            for rel_step in ident_rels:
                abs_step = START_NOMINAL_STEP + int(rel_step)
                _load_snapshot(branch, output_dir, int(rel_step))
                model = branch.pipeline.model
                records = _train_records(branch.pipeline)
                records_by_name = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in records}
                view_ids = list(samples.keys())
                for source_idx, source_view in enumerate(view_ids[:5]):
                    target_view = view_ids[(source_idx + 7) % len(view_ids)]
                    _idx_s, camera_s, _batch_s = records_by_name[source_view]
                    _idx_t, camera_t, _batch_t = records_by_name[target_view]
                    flat = samples[source_view].general_flat[:CAMERA_CONTEXT_SAMPLE_RAYS]
                    if flat.numel() == 0:
                        continue
                    cam_swap = _clone_camera(camera_s.to(model.device))
                    cam_swap.camera_to_worlds[0, :3, 3] = camera_t.to(model.device).camera_to_worlds[0, :3, 3]
                    with torch.no_grad():
                        raw_base, height, width = MI._medium_raw_for_camera(model, camera_s)
                        raw_swap, height2, width2 = MI._medium_raw_for_camera(model, cam_swap)
                        if (height, width) != (height2, width2):
                            raise RuntimeError("Camera-context sensitivity clone changed image size")
                        med_base = MI._activate_medium(model, raw_base, height, width)
                        med_swap = MI._activate_medium(model, raw_swap, height, width)
                    flat_dev = flat.to(model.device)
                    base_raw = raw_base.reshape(-1, 9)[flat_dev].detach().float().cpu()
                    swap_raw = raw_swap.reshape(-1, 9)[flat_dev].detach().float().cpu()
                    base_act = torch.cat(
                        [
                            med_base["medium_rgb"].reshape(-1, 3)[flat_dev],
                            med_base["medium_bs"].reshape(-1, 3)[flat_dev],
                            med_base["medium_attn"].reshape(-1, 3)[flat_dev],
                        ],
                        dim=-1,
                    ).detach().float().cpu()
                    swap_act = torch.cat(
                        [
                            med_swap["medium_rgb"].reshape(-1, 3)[flat_dev],
                            med_swap["medium_bs"].reshape(-1, 3)[flat_dev],
                            med_swap["medium_attn"].reshape(-1, 3)[flat_dev],
                        ],
                        dim=-1,
                    ).detach().float().cpu()
                    raw_delta = swap_raw - base_raw
                    act_delta = swap_act - base_act
                    rows.append(
                        {
                            "branch": branch_name,
                            "absolute_step": abs_step,
                            "relative_step": int(rel_step),
                            "source_view_id": source_view,
                            "swapped_camera_context_view_id": target_view,
                            "sampled_rays": int(flat.numel()),
                            "raw_z_med_rms_delta_9d": float(torch.sqrt(raw_delta.square().mean()).item()),
                            "raw_betaD_rms_delta": float(torch.sqrt(raw_delta[:, 6:9].square().mean()).item()),
                            "activated_medium_rms_delta_9d": float(torch.sqrt(act_delta.square().mean()).item()),
                            "B_inf_rms_delta": float(torch.sqrt(act_delta[:, 0:3].square().mean()).item()),
                            "beta_B_rms_delta": float(torch.sqrt(act_delta[:, 3:6].square().mean()).item()),
                            "beta_D_rms_delta": float(torch.sqrt(act_delta[:, 6:9].square().mean()).item()),
                        }
                    )
        finally:
            _release(branch)
    _write_csv(output_dir / "camera_context_sensitivity.csv", rows)
    _write_json(output_dir / "camera_context_sensitivity.json", {"rows": rows})
    return rows


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


def _index_rows(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Dict[Tuple[Any, ...], Mapping[str, Any]]:
    return {tuple(row[field] for field in fields): row for row in rows}


def _summarize_results(
    output_dir: Path,
    *,
    final_actual: int,
    start_audit: Mapping[str, Any],
    config_diff: Mapping[str, Any],
    camera_audit: Mapping[str, Any],
    global_rows: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
    decomp_rows: Sequence[Mapping[str, Any]],
    medium_rows: Sequence[Mapping[str, Any]],
    camera_summary_rows: Sequence[Mapping[str, Any]],
    ident_rows: Sequence[Mapping[str, Any]],
    nvo_rows: Sequence[Mapping[str, Any]],
    counter_ratio_rows: Sequence[Mapping[str, Any]],
    far_rows: Sequence[Mapping[str, Any]],
    topology_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    final_rel = _rel(final_actual)
    g = _index_rows(global_rows, ("branch", "absolute_step", "split"))
    final_eval_c0 = g[("C0", final_actual, "eval")]
    final_eval_c1 = g[("C1", final_actual, "eval")]
    final_train_c0 = g[("C0", final_actual, "train")]
    final_train_c1 = g[("C1", final_actual, "train")]
    dpsnr = float(final_eval_c1["PSNR"]) - float(final_eval_c0["PSNR"])
    dssim = float(final_eval_c1["SSIM"]) - float(final_eval_c0["SSIM"])
    dlpips = float(final_eval_c1["LPIPS"]) - float(final_eval_c0["LPIPS"])
    dmse = float(final_eval_c1["MSE"]) - float(final_eval_c0["MSE"])
    rgb_safe = dpsnr >= RGB_SAFE_RULE["dPSNR_min"] and dssim >= RGB_SAFE_RULE["dSSIM_min"] and dlpips <= RGB_SAFE_RULE["dLPIPS_max"]

    medium_idx = _index_rows(medium_rows, ("branch", "absolute_step", "population"))
    raw_reduction_steps = []
    raw_ratios = []
    for abs_step in _snapshot_abs_steps(final_actual):
        if abs_step == START_NOMINAL_STEP:
            continue
        key0 = ("C0", abs_step, "M_SAFE")
        key1 = ("C1", abs_step, "M_SAFE")
        if key0 in medium_idx and key1 in medium_idx:
            c0 = float(medium_idx[key0]["raw_betaD_variance_pooled"])
            c1 = float(medium_idx[key1]["raw_betaD_variance_pooled"])
            ratio = c1 / max(c0, EPS)
            raw_ratios.append({"absolute_step": abs_step, "ratio": ratio, "delta": c1 - c0})
            if c1 < c0:
                raw_reduction_steps.append(abs_step)
    raw_variance_reduced = len(raw_reduction_steps) >= max(1, len(raw_ratios) - 1)

    ident_idx = _index_rows(ident_rows, ("run", "nominal_step", "population"))
    ident_improved_steps = []
    for abs_step in _identifiability_abs_steps(final_actual):
        key0 = ("C0", abs_step, "M_SAFE")
        key1 = ("C1", abs_step, "M_SAFE")
        if key0 in ident_idx and key1 in ident_idx:
            c0 = ident_idx[key0]
            c1 = ident_idx[key1]
            ratio_improved = float(c1["sigma_min_over_sigma_max"]) > float(c0["sigma_min_over_sigma_max"])
            cond_improved = float(c1["condition_number"]) < float(c0["condition_number"])
            if ratio_improved or cond_improved:
                ident_improved_steps.append(abs_step)
    identifiability_improved = len(ident_improved_steps) >= 2

    nvo_idx = _index_rows(nvo_rows, ("run", "nominal_step", "population", "mode"))
    weak_variation_reduced_steps = []
    for abs_step in _identifiability_abs_steps(final_actual):
        key0 = ("C0", abs_step, "M_SAFE", "v_min")
        key1 = ("C1", abs_step, "M_SAFE", "v_min")
        if key0 in nvo_idx and key1 in nvo_idx:
            if float(nvo_idx[key1]["natural_std"]) < float(nvo_idx[key0]["natural_std"]):
                weak_variation_reduced_steps.append(abs_step)
    weak_ambiguity_reduced = len(weak_variation_reduced_steps) >= 2

    decomp_safe = True
    for row in decomp_rows:
        if row["branch"] == "C1" and int(row["absolute_step"]) in _identifiability_abs_steps(final_actual):
            if float(row.get("P_J_gt_1", 1.0)) != 0.0:
                decomp_safe = False

    cam_idx = _index_rows(camera_summary_rows, ("branch", "absolute_step", "population"))
    final_cam_c0 = cam_idx.get(("C0", final_actual, "M_SAFE"), {})
    final_cam_c1 = cam_idx.get(("C1", final_actual, "M_SAFE"), {})
    channel_ratios = []
    for basis in ("raw_betaD", "activated_betaD"):
        for channel in ("r", "g", "b"):
            for source in ("across_camera_variance", "within_camera_spatial_variance"):
                key = f"{basis}_{channel}_{source}"
                if key in final_cam_c0 and key in final_cam_c1:
                    channel_ratios.append(float(final_cam_c1[key]) / max(float(final_cam_c0[key]), EPS))
    broad_collapse = bool(channel_ratios and float(np.median(channel_ratios)) < 0.05)

    per_eval = [
        row
        for row in _delta_rows(
            per_view_rows,
            ("absolute_step", "split", "view_id"),
            ("PSNR", "SSIM", "LPIPS", "MSE"),
        )
        if int(row["absolute_step"]) == final_actual and row["split"] == "eval"
    ]
    eval_positive_psnr = sum(1 for row in per_eval if float(row.get("delta_C1_minus_C0_PSNR", 0.0)) > 0.0)
    view_consistent = eval_positive_psnr >= 2 or abs(dpsnr) < 0.05

    far_delta = _delta_rows(
        far_rows,
        ("absolute_step", "population", "stratification_basis", "stratum"),
        ("raw_betaD_variance_pooled", "sigma_min_over_sigma_max", "rgb_mse"),
    )
    far_effect_steps = []
    for abs_step in _identifiability_abs_steps(final_actual):
        for basis in ("depth", "tau"):
            near = next(
                (
                    row
                    for row in far_delta
                    if int(row["absolute_step"]) == abs_step
                    and row["population"] == "M_SAFE"
                    and row["stratification_basis"] == basis
                    and row["stratum"] == "near_or_low"
                ),
                None,
            )
            far = next(
                (
                    row
                    for row in far_delta
                    if int(row["absolute_step"]) == abs_step
                    and row["population"] == "M_SAFE"
                    and row["stratification_basis"] == basis
                    and row["stratum"] == "far_or_high"
                ),
                None,
            )
            if near and far:
                near_reduction = -float(near.get("delta_C1_minus_C0_raw_betaD_variance_pooled", 0.0))
                far_reduction = -float(far.get("delta_C1_minus_C0_raw_betaD_variance_pooled", 0.0))
                if far_reduction >= near_reduction:
                    far_effect_steps.append({"absolute_step": abs_step, "basis": basis})

    if (
        raw_variance_reduced
        and identifiability_improved
        and weak_ambiguity_reduced
        and rgb_safe
        and decomp_safe
        and not broad_collapse
        and view_consistent
    ):
        mic_class = "MIC_ACTIONABLE"
    elif (raw_variance_reduced and (identifiability_improved or weak_ambiguity_reduced) and decomp_safe and rgb_safe) or (
        raw_variance_reduced and broad_collapse and decomp_safe
    ):
        mic_class = "MIC_PARTIALLY_ACTIONABLE"
    else:
        mic_class = "MIC_NOT_ACTIONABLE"

    if mic_class == "MIC_ACTIONABLE" and not broad_collapse:
        variance_class = "BETAD_VARIANCE_PROBE_VALID"
    elif mic_class in {"MIC_ACTIONABLE", "MIC_PARTIALLY_ACTIONABLE"} and broad_collapse:
        variance_class = "BETAD_VARIANCE_PROBE_TOO_COARSE"
    elif mic_class == "MIC_PARTIALLY_ACTIONABLE":
        variance_class = "BETAD_VARIANCE_PROBE_TOO_COARSE"
    else:
        variance_class = "BETAD_VARIANCE_PROBE_FAILED"

    pop_final = {
        row["branch"]: row
        for row in topology_rows
        if int(row["absolute_step"]) == final_actual
    }
    summary = {
        "experiment": "BND-MIC-CAUSAL-IUI3",
        "scene": SCENE,
        "final_absolute_step": int(final_actual),
        "final_relative_step": int(final_rel),
        "START_STATE_EQUIVALENCE": bool(start_audit["START_STATE_EQUIVALENCE"]),
        "CAMERA_SEQUENCE_MATCH": bool(camera_audit["CAMERA_SEQUENCE_MATCH"]),
        "ONLY_INTERVENTION_MIC": bool(config_diff["ONLY_INTERVENTION_MIC"]),
        "MIC_WEIGHT": MIC_WEIGHT,
        "MIC_TARGET": MIC_TARGET,
        "RGB_SAFE_RULE": RGB_SAFE_RULE,
        "final_eval_C0": {key: float(final_eval_c0[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_eval_C1": {key: float(final_eval_c1[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_eval_delta_C1_minus_C0": {"PSNR": dpsnr, "SSIM": dssim, "LPIPS": dlpips, "MSE": dmse},
        "final_train_C0": {key: float(final_train_c0[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_train_C1": {key: float(final_train_c1[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "rgb_safe": bool(rgb_safe),
        "raw_betaD_contextual_variance_reduced": bool(raw_variance_reduced),
        "raw_betaD_reduction_steps_M_SAFE": raw_reduction_steps,
        "raw_betaD_reduction_ratios_M_SAFE": raw_ratios,
        "aggregate_identifiability_improved": bool(identifiability_improved),
        "aggregate_identifiability_improved_steps_M_SAFE": ident_improved_steps,
        "counterfactual_weak_mode_ambiguity_reduced": bool(weak_ambiguity_reduced),
        "weak_vmin_natural_std_reduction_steps_M_SAFE": weak_variation_reduced_steps,
        "decomposition_safety_intact": bool(decomp_safe),
        "camera_conditioned_variation_broad_collapse": bool(broad_collapse),
        "camera_variation_ratio_median_final_M_SAFE": float(np.median(channel_ratios)) if channel_ratios else float("nan"),
        "final_eval_views_positive_PSNR_count": int(eval_positive_psnr),
        "final_eval_view_count": len(per_eval),
        "effect_not_single_view_dominated": bool(view_consistent),
        "far_high_tau_stronger_raw_variance_effect_entries": far_effect_steps,
        "far_high_tau_effect_stronger": len(far_effect_steps) >= 3,
        "final_population": {
            branch: int(row["gaussian_count"]) for branch, row in pop_final.items()
        },
        "MIC_actionability_classification": mic_class,
        "variance_probe_classification": variance_class,
        "next_single_experiment": (
            "Design Observability-Guided Contextual Medium Control for the camera-conditioned medium field."
            if mic_class in {"MIC_ACTIONABLE", "MIC_PARTIALLY_ACTIONABLE"}
            else "Close the betaD identifiability regularization line; do not sweep lambda."
        ),
    }
    _write_json(output_dir / "final_summary.json", summary)
    _write_csv(output_dir / "final_summary.csv", [{"key": key, "value": value} for key, value in summary.items() if not isinstance(value, (dict, list))])
    _write_csv(output_dir / "global_rgb_deltas.csv", _delta_rows(global_rows, ("absolute_step", "split"), ("PSNR", "SSIM", "LPIPS", "MSE")))
    _write_csv(output_dir / "per_view_rgb_deltas.csv", per_eval)
    _write_json(output_dir / "per_view_rgb_deltas.json", {"rows": per_eval})
    _write_csv(
        output_dir / "raw_betad_contextual_variance_deltas.csv",
        _delta_rows(medium_rows, ("absolute_step", "population"), ("raw_betaD_variance_pooled", "activated_betaD_variance_pooled", "rgb_mse")),
    )
    _write_csv(
        output_dir / "identifiability_deltas.csv",
        _delta_rows(ident_rows, ("nominal_step", "population"), ("sigma_min_over_sigma_max", "condition_number", "effective_rank"), branch_field="run"),
    )
    _write_csv(output_dir / "far_high_tau_deltas.csv", far_delta)
    _write_json(output_dir / "far_high_tau_deltas.json", {"rows": far_delta})
    return summary


def _checkpoint_metric_summary(
    output_dir: Path,
    global_rows: Sequence[Mapping[str, Any]],
    medium_rows: Sequence[Mapping[str, Any]],
    ident_rows: Sequence[Mapping[str, Any]],
    counter_ratio_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    medium_idx = _index_rows(medium_rows, ("branch", "absolute_step", "population"))
    ident_idx = _index_rows(ident_rows, ("run", "nominal_step", "population"))
    counter_idx = _index_rows(counter_ratio_rows, ("branch", "absolute_step", "population"))
    rows: List[Dict[str, Any]] = []
    for row in global_rows:
        if row["split"] != "eval":
            continue
        branch = row["branch"]
        abs_step = int(row["absolute_step"])
        item = dict(row)
        med = medium_idx.get((branch, abs_step, "M_SAFE"))
        if med:
            item["M_SAFE_raw_betaD_variance_pooled"] = med["raw_betaD_variance_pooled"]
            item["M_SAFE_activated_betaD_variance_pooled"] = med["activated_betaD_variance_pooled"]
        ident = ident_idx.get((branch, abs_step, "M_SAFE"))
        if ident:
            item["M_SAFE_sigma_min_over_sigma_max"] = ident["sigma_min_over_sigma_max"]
            item["M_SAFE_condition_number"] = ident["condition_number"]
            item["M_SAFE_effective_rank"] = ident["effective_rank"]
        counter = counter_idx.get((branch, abs_step, "M_SAFE"))
        if counter:
            item["M_SAFE_vmin_over_vmax_rgb_change"] = counter["vmin_over_vmax_rgb_change"]
            item["M_SAFE_vmin_rms_medium_output_change_9d"] = counter["vmin_rms_medium_output_change_9d"]
        rows.append(item)
    _write_csv(output_dir / "checkpoint_metric_summary.csv", rows)
    _write_json(output_dir / "checkpoint_metric_summary.json", {"rows": rows})
    return rows


def _write_research_note(path: Path, summary: Mapping[str, Any]) -> None:
    delta = summary["final_eval_delta_C1_minus_C0"]
    lines = [
        "# BND-MIC-CAUSAL-IUI3",
        "",
        "## CODE FACT",
        "The MIC prototype is implemented behind `WaterSplattingModelConfig.medium_identifiability_enabled`; the default remains `False` with zero weight.",
        "The loss reads `outputs['medium_raw'][..., 6:9]`, the raw pre-softplus beta_D channels, and penalizes variance around a stop-gradient per-channel mean.",
        "The causal driver is `scripts/experiments/run_bnd_mic_causal_iui3.py` and restores pipeline, optimizer, scheduler, scaler, and explicit camera sequence state from BND@3000.",
        "",
        "## CONFIG FACT",
        "Both arms use `bounded_sh3`, SH degree 3, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, and `infinite_water_enabled=False`.",
        f"C0 has MIC disabled. C1 uses `lambda_mic={MIC_WEIGHT}` with target `{MIC_TARGET}` for every continuation update from 3001 through the final step.",
        "No CB-FG, CB-BG, BAP, UNORM, LOSSRESP, CDEPTH, OMVC, depth prior, depth residual, depth-aware alpha, or medium-context removal is enabled.",
        "",
        "## EXPERIMENTAL FACT",
        f"`START_STATE_EQUIVALENCE={summary['START_STATE_EQUIVALENCE']}` and `CAMERA_SEQUENCE_MATCH={summary['CAMERA_SEQUENCE_MATCH']}`.",
        "Outputs are written under `outputs/bnd_mic_causal_iui3_20260825/` and are intentionally not committed.",
        "",
        "## QUANTITATIVE RESULT",
        f"Final eval delta C1-C0: PSNR `{delta['PSNR']:.6f}` dB, SSIM `{delta['SSIM']:.6f}`, LPIPS `{delta['LPIPS']:.6f}`, MSE `{delta['MSE']:.8f}`.",
        f"Raw beta_D contextual variance reduced on M_SAFE: `{summary['raw_betaD_contextual_variance_reduced']}`; steps `{summary['raw_betaD_reduction_steps_M_SAFE']}`.",
        f"Aggregate identifiability improved on M_SAFE: `{summary['aggregate_identifiability_improved']}`; steps `{summary['aggregate_identifiability_improved_steps_M_SAFE']}`.",
        f"Weak-mode natural variation reduced on M_SAFE: `{summary['counterfactual_weak_mode_ambiguity_reduced']}`; steps `{summary['weak_vmin_natural_std_reduction_steps_M_SAFE']}`.",
        f"Decomposition safety intact: `{summary['decomposition_safety_intact']}`.",
        "",
        "## INFERENCE",
        f"MIC actionability classification: `{summary['MIC_actionability_classification']}`.",
        f"Simple variance-probe classification: `{summary['variance_probe_classification']}`.",
        "The experiment does not prove true attenuation, true colors, or true geometry; it tests whether suppressing a measured beta_D-dominated low-observability freedom is useful under BND.",
        "",
        "## HYPOTHESIS",
        f"Next single experiment: {summary['next_single_experiment']}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf8")


def _prepare_output_dir(output_dir: Path, allow_existing: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise RuntimeError(f"Output directory already exists and is non-empty: {output_dir}. Pass --allow-existing-output only for an intentional rerun.")
    output_dir.mkdir(parents=True, exist_ok=True)


def run(repo: Path, output_dir: Path, *, final_step: int, allow_existing_output: bool) -> Dict[str, Any]:
    gpu_manifest = _assert_runtime_policy()
    output_dir = output_dir if output_dir.is_absolute() else repo / output_dir
    _prepare_output_dir(output_dir, allow_existing_output)

    env_manifest = _environment_manifest(gpu_manifest)
    repo_manifest = _repo_manifest(repo)
    _write_json(output_dir / "gpu_manifest.json", gpu_manifest)
    _write_json(output_dir / "environment_manifest.json", env_manifest)
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    config_path = repo / PW.BND_CONFIG
    start_actual = _actual_step(config_path, START_NOMINAL_STEP)
    formal_final_actual = _actual_step(config_path, FINAL_NOMINAL_STEP)
    final_actual = formal_final_actual if int(final_step) == FINAL_NOMINAL_STEP else int(final_step)
    if final_actual <= START_NOMINAL_STEP or final_actual > formal_final_actual:
        raise ValueError(f"final-step must be in ({START_NOMINAL_STEP}, {formal_final_actual}], got {final_step}")
    snapshot_rels = _snapshot_rel_steps(final_actual)
    ident_rels = tuple(_rel(step) for step in _identifiability_abs_steps(final_actual))
    start_ckpt = _available_steps(config_path)[start_actual]
    ckpt_state = torch.load(start_ckpt, map_location="cpu")
    start_manifest = {
        "START_BND_CHECKPOINT_VALID": True,
        "config_path": str(config_path),
        "start_checkpoint_path": str(start_ckpt),
        "start_nominal_step": START_NOMINAL_STEP,
        "start_actual_step": int(start_actual),
        "checkpoint_step_field": int(ckpt_state.get("step", -1)),
        "final_requested_step": int(final_step),
        "final_actual_step": int(final_actual),
        "snapshot_absolute_steps": [START_NOMINAL_STEP + int(rel) for rel in snapshot_rels],
        "identifiability_absolute_steps": [START_NOMINAL_STEP + int(rel) for rel in ident_rels],
        "gaussian_count": int(ckpt_state["pipeline"]["_model.gauss_params.means"].shape[0]),
        "optimizer_state_available": bool(ckpt_state.get("optimizers")),
        "scheduler_state_available": bool(ckpt_state.get("schedulers")),
        "scaler_state_available": "scalers" in ckpt_state,
        "checkpoint_sha256": _sha256_path(start_ckpt),
        "config_sha256": _sha256_path(config_path),
    }
    _write_json(output_dir / "start_checkpoint_manifest.json", start_manifest)
    shutil.copy2(config_path, output_dir / "source_bnd_config.yml")

    seq_branch = _load_branch(repo, "C0")
    try:
        camera_indices, camera_names, _sequence_rows = _generate_camera_sequence(seq_branch, output_dir, final_actual)
    finally:
        _release(seq_branch)
    camera_audit = json.loads((output_dir / "camera_sequence_audit.json").read_text(encoding="utf8"))
    training_rng = _seed_all(TRAINING_RNG_SEED)
    _write_json(output_dir / "rng_state_manifest.json", {"seed": TRAINING_RNG_SEED, **_rng_manifest(training_rng)})

    start_audit = _start_state_equivalence(repo, output_dir, training_rng)
    config_diff = _config_diff_audit(output_dir)
    if not config_diff["ONLY_INTERVENTION_MIC"]:
        raise RuntimeError("ONLY_INTERVENTION_MIC=false; stopping before training.")

    all_training: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    all_ckpts: List[Dict[str, Any]] = []
    all_topology: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        print(f"[BND-MIC] training {branch_name}", flush=True)
        rows, events, ckpts, topology = _train_branch(
            repo,
            branch_name,
            camera_indices=camera_indices,
            camera_names=camera_names,
            training_rng=training_rng,
            snapshot_rels=snapshot_rels,
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
    _write_csv(output_dir / "continuation_checkpoint_manifest.csv", all_ckpts)
    _write_json(output_dir / "continuation_checkpoint_manifest.json", {"rows": all_ckpts})
    _write_csv(output_dir / "gaussian_population.csv", all_topology)
    _write_json(output_dir / "gaussian_population.json", {"rows": all_topology})

    print("[BND-MIC] evaluating RGB/decomposition snapshots", flush=True)
    global_rows, per_view_rows, decomp_rows = _evaluate_snapshots(repo, output_dir, snapshot_rels)
    _write_csv(output_dir / "global_rgb_metrics.csv", global_rows)
    _write_json(output_dir / "global_rgb_metrics.json", {"rows": global_rows})
    _write_csv(output_dir / "per_view_rgb_metrics.csv", per_view_rows)
    _write_json(output_dir / "per_view_rgb_metrics.json", {"rows": per_view_rows})
    _write_csv(output_dir / "decomposition_safety.csv", decomp_rows)
    _write_json(output_dir / "decomposition_safety.json", {"rows": decomp_rows})

    print("[BND-MIC] building deterministic Phase-A samples", flush=True)
    samples, sampling_meta, sampling_rows = MI._build_samples(repo, output_dir, SAMPLES_PER_VIEW, SAMPLE_SEED)
    _write_csv(output_dir / "deterministic_ray_sampling.csv", sampling_rows)
    _write_json(output_dir / "deterministic_ray_sampling.json", sampling_meta)

    medium_outputs = _simple_medium_sample_audit(repo, output_dir, snapshot_rels, samples)
    ident_outputs = _run_identifiability_audit(repo, output_dir, ident_rels, samples)
    camera_context_rows = _camera_context_sensitivity(repo, output_dir, ident_rels, samples)

    _checkpoint_metric_summary(
        output_dir,
        global_rows,
        medium_outputs["raw_betad_contextual_variance"],
        ident_outputs["identifiability_summary"],
        ident_outputs["counterfactual_weak_mode_sensitivity"],
    )

    summary = _summarize_results(
        output_dir,
        final_actual=final_actual,
        start_audit=start_audit,
        config_diff=config_diff,
        camera_audit=camera_audit,
        global_rows=global_rows,
        per_view_rows=per_view_rows,
        decomp_rows=decomp_rows,
        medium_rows=medium_outputs["raw_betad_contextual_variance"],
        camera_summary_rows=medium_outputs["camera_expressiveness_summary"],
        ident_rows=ident_outputs["identifiability_summary"],
        nvo_rows=ident_outputs["natural_variance_vs_observability"],
        counter_ratio_rows=ident_outputs["counterfactual_weak_mode_sensitivity"],
        far_rows=ident_outputs["far_high_tau_analysis"],
        topology_rows=all_topology,
    )
    summary.update(
        {
            "CONDA_ENV": env_manifest["CONDA_ENV"],
            "PYTHON_PATH": env_manifest["PYTHON_PATH"],
            "TORCH_VERSION": env_manifest["TORCH_VERSION"],
            "CUDA_VISIBLE_DEVICES": env_manifest["CUDA_VISIBLE_DEVICES"],
            "gpu": dict(gpu_manifest),
            "camera_context_sensitivity_rows": len(camera_context_rows),
        }
    )
    _write_json(output_dir / "final_summary.json", summary)
    _write_research_note(repo / RESEARCH_NOTE, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--final-step", type=int, default=FINAL_NOMINAL_STEP)
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        args.repo.resolve(),
        args.output_dir,
        final_step=int(args.final_step),
        allow_existing_output=bool(args.allow_existing_output),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
