#!/usr/bin/env python3
"""Formal single-scene RAOC-Q50-vs-Q80 causal worker.

The worker deliberately keeps the registered BND-from-scratch training setup
and changes only the medium capacity allocation path.  It owns one physical
GPU, runs C0 before C1, and performs all post-training diagnostics on that GPU.
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
from scripts.diagnostics import audit_local_contextual_support_predictor_iui3 as LOCAL
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM
from water_splatting.raoc import (
    apply_modal_keep_gate,
    calibrate_local_scales,
    local_keep_gates,
    observability_gates,
    ray_keep_gates,
)
from water_splatting.rendering.medium_jacobian import analytic_medium_jacobian_actions


EXPERIMENT = "RAOC-Q50-VS-Q80-CAUSAL-FOUR-SCENE"
BRANCHES = ("C0", "C1")
BRANCH_QUANTILES = {"C0": 0.50, "C1": 0.80}
SNAPSHOT_STEPS = (3000, 5000, 8000, 10000, 13000, 14999)
IDENT_STEPS = (5000, 10000, 14999)
REFRESH_STEPS = (0, 5000, 10000)
FINAL_STEP = 14999
MAX_STEPS = FINAL_STEP + 1
TRAINING_SEED = 42
CALIBRATION_MASTER_SEED = 20260827
SWAP_SEED = 2026082701
CALIBRATION_CAMERAS = 25
RAYS_PER_CAMERA = 1024
ALTERNATIVE_CONTEXTS = 8
LOG_INTERVAL = 500
EPS = 1e-12
WEAK_CAPACITY_TOLERANCE = 1e-6
ALLOWED_GPUS = frozenset(("6", "7", "8", "9"))
HISTORICAL_GMVC = (
    "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
    "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
)


def _configure_schedule(max_steps: int) -> None:
    """Set the run schedule; the default remains the registered 15K protocol."""

    if max_steps < 1:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    global SNAPSHOT_STEPS, IDENT_STEPS, REFRESH_STEPS, FINAL_STEP, MAX_STEPS
    MAX_STEPS = int(max_steps)
    FINAL_STEP = MAX_STEPS - 1
    if MAX_STEPS == 15000:
        SNAPSHOT_STEPS = (3000, 5000, 8000, 10000, 13000, 14999)
        IDENT_STEPS = (5000, 10000, 14999)
        REFRESH_STEPS = (0, 5000, 10000)
        return
    # Short runs are validation-only. Include the first and final trained
    # states so checkpoint reload and all final-summary joins are exercised.
    SNAPSHOT_STEPS = tuple(sorted({0, FINAL_STEP}))
    IDENT_STEPS = (FINAL_STEP,)
    REFRESH_STEPS = (0,)

SCENES: Dict[str, Dict[str, Any]] = {
    "Curasao": {
        "data_name": "Curasao",
        "data_path": "undistorted_data/undistorted_Curasao",
        "source_config": "outputs/dewater_bounded_sh3_scratch_20260808/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000/water-splatting/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_bnd-scratch_g1p00/config.yml",
        "locked_safe": False,
    },
    "IUI3-RedSea": {
        "data_name": "IUI3-RedSea",
        "data_path": "undistorted_data/undistorted_IUI3-RedSea",
        "source_config": "outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_iui3_bnd_g1p00/config.yml",
        "locked_safe": True,
    },
    "JapaneseGradens-RedSea": {
        "data_name": "JapaneseGradens-RedSea",
        "data_path": "undistorted_data/undistorted_JapaneseGradens-RedSea",
        "source_config": "outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_japanesegradens_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_japanesegradens_bnd_g1p00/config.yml",
        "locked_safe": False,
    },
    "Panama": {
        "data_name": "Panama",
        "data_path": "undistorted_data/undistorted_Panama",
        "source_config": "outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/config.yml",
        "locked_safe": False,
    },
}


@dataclass
class ViewSample:
    view_id: str
    height: int
    width: int
    general_flat: Tensor
    safe_flat: Tensor
    safe_available_pixels: int = 0

    def flat_for(self, population: str) -> Tensor:
        if population == "GENERAL":
            return self.general_flat
        if population == "M_SAFE":
            return self.safe_flat
        raise ValueError(population)


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
        return value.detach().cpu().tolist() if value.numel() != 1 else float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_object(value: Any) -> str:
    import pickle

    return _sha256_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
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
        "python_sha256": _hash_object(state["python"]),
        "numpy_sha256": _hash_object(state["numpy"]),
        "torch_cpu_sha256": _sha256_bytes(state["torch_cpu"].cpu().numpy().tobytes()),
    }
    if "torch_cuda" in state:
        out["torch_cuda_sha256"] = [_sha256_bytes(item.cpu().numpy().tobytes()) for item in state["torch_cuda"]]
    return out


def _runtime(gpu: str) -> Dict[str, Any]:
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env != "water_splatting":
        raise RuntimeError(f"CONDA_DEFAULT_ENV must be water_splatting, got {env!r}")
    if visible != str(gpu) or visible not in ALLOWED_GPUS:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must expose exactly assigned physical GPU {gpu}, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each formal worker must see exactly one CUDA device")
    if int(torch.cuda.current_device()) != 0:
        raise RuntimeError("Worker logical CUDA device must be cuda:0")
    props = torch.cuda.get_device_properties(0)
    return {
        "CONDA_ENV": env,
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": visible,
        "physical_gpu_id": str(gpu),
        "torch_logical_gpu_id": 0,
        "torch_visible_gpu_count": 1,
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
    }


def _branch_quantile(branch: str) -> float:
    try:
        return float(BRANCH_QUANTILES[branch])
    except KeyError:
        raise ValueError(f"unknown RAOC quantile branch {branch!r}")


def _configure_model_config(config: Any, branch: str) -> None:
    """Apply the registered formal model settings before ``setup()``."""

    model_config = config.pipeline.model
    model_config.intrinsic_color_parameterization = "bounded_sh3"
    model_config.rasterize_mode = "classic"
    model_config.medium_context_mode = "dir_xy_camera"
    model_config.medium_camera_context_scale = 1.0
    model_config.medium_camera_context_dropout = 0.0
    model_config.medium_camera_context_ablation = False
    model_config.camera_medium_observability_enabled = False
    model_config.camera_medium_ray_adaptive_observability_enabled = True
    model_config.camera_medium_raoc_backend = "reference"
    model_config.camera_medium_raoc_local_scale_quantile = _branch_quantile(branch)
    model_config.camera_medium_observability_strength = 1.0
    model_config.b_inf_mode = "tied"
    model_config.infinite_water_enabled = False
    model_config.coarse_depth_supervision_enabled = False
    model_config.medium_identifiability_enabled = False
    model_config.medium_identifiability_weight = 0.0


def _configure_model(model: Any, branch: str) -> None:
    """Re-assert branch flags after construction and before each forward."""

    model.config.intrinsic_color_parameterization = "bounded_sh3"
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.medium_camera_context_scale = 1.0
    model.config.medium_camera_context_dropout = 0.0
    model.config.medium_camera_context_ablation = False
    model.config.camera_medium_observability_enabled = False
    model.config.camera_medium_ray_adaptive_observability_enabled = True
    model.config.camera_medium_raoc_backend = getattr(model.config, "camera_medium_raoc_backend", "reference")
    model.config.camera_medium_raoc_local_scale_quantile = _branch_quantile(branch)
    model.config.camera_medium_observability_strength = 1.0
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.coarse_depth_supervision_enabled = False
    model.config.medium_identifiability_enabled = False
    model.config.medium_identifiability_weight = 0.0


def _setup_branch(repo: Path, scene_cfg: Mapping[str, Any], branch: str) -> BranchState:
    config_path = repo / str(scene_cfg["source_config"])
    config = yaml.load(config_path.read_text(encoding="utf8"), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = None
    config.load_step = None
    config.load_checkpoint = None
    config.pipeline.datamanager.load_depths = False
    config.pipeline.datamanager.dataparser.data = repo / str(scene_cfg["data_path"])
    _configure_model_config(config, branch)
    _set_random_seed(int(config.machine.seed))
    pipeline = config.pipeline.setup(device=torch.device("cuda:0"), test_mode="test")
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected Pipeline, got {type(pipeline)}")
    _configure_model(pipeline.model, branch)
    pipeline.model.clear_camera_medium_ray_adaptive_observability_state()
    pipeline.model.set_camera_medium_observability_projector(None)
    pipeline.model.step = 0
    optimizers = Optimizers(MIC._optimizer_groups(config, pipeline.model), pipeline.model.get_param_groups())
    pipeline.eval()
    return BranchState(branch, config_path, config, pipeline, optimizers, {})


def _release(branch: Optional[BranchState]) -> None:
    if branch is None:
        return
    try:
        del branch.pipeline
        del branch.optimizers
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _train_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return MI.PW._records(pipeline)["train"]


def _eval_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return MI.PW._records(pipeline)["eval"]


def _model_tensors(model: Any) -> Dict[str, Tensor]:
    out = {name: getattr(model, name).detach().cpu().clone() for name in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")}
    out["medium_mlp"] = torch.cat([p.detach().cpu().reshape(-1) for p in model.medium_mlp.parameters()])
    out["direction_encoding"] = torch.cat([p.detach().cpu().reshape(-1) for p in model.direction_encoding.parameters()])
    out["model_state"] = torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])
    return out


def _max_diff(a: Tensor, b: Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0


def _optimizer_flat(optimizers: Optimizers) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    for group, optimizer in optimizers.optimizers.items():
        pieces: List[Tensor] = []
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                for value in optimizer.state.get(param, {}).values():
                    if isinstance(value, Tensor):
                        pieces.append(value.detach().cpu().float().reshape(-1))
        out[group] = torch.cat(pieces) if pieces else torch.empty(0)
    return out


def _state_equal(a: Any, b: Any) -> bool:
    """Compare nested checkpoint/sampler state without tensor truth-value bugs."""

    if isinstance(a, Tensor) or isinstance(b, Tensor):
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            return False
        return a.shape == b.shape and bool(torch.equal(a.detach().cpu(), b.detach().cpu()))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return False
        return bool(np.array_equal(a, b))
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not isinstance(a, Mapping) or not isinstance(b, Mapping) or set(a) != set(b):
            return False
        return all(_state_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != len(b):
            return False
        return all(_state_equal(left, right) for left, right in zip(a, b))
    try:
        result = a == b
        return bool(result)
    except Exception:
        return False


def _sampler_state(pipeline: Any) -> Dict[str, Any]:
    dm = pipeline.datamanager
    return {
        "train_unseen_cameras": list(getattr(dm, "train_unseen_cameras", [])),
        "eval_unseen_cameras": list(getattr(dm, "eval_unseen_cameras", [])),
        "random_generator": getattr(dm, "random_generator", None).getstate()
        if hasattr(getattr(dm, "random_generator", None), "getstate")
        else None,
        "train_unsampled_epoch_count": getattr(dm, "train_unsampled_epoch_count", None),
    }


def _refinement_state(model: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "step": int(getattr(model, "step", -1)),
        "gaussian_count": int(model.means.shape[0]),
        "last_event": dict(getattr(model, "_refinement_last_event", {})),
        "guidance_hardness": getattr(model, "_refinement_guidance_hardness", None),
        "guidance_brightness": getattr(model, "_refinement_guidance_brightness", None),
        "budget_schedule": getattr(model, "_refinement_budget_schedule", None),
    }
    lineage = getattr(model, "gaussian_lineage_ids", None)
    if isinstance(lineage, Tensor):
        state["gaussian_lineage_ids"] = lineage.detach().cpu().clone()
    return state


def _start_state_audit(repo: Path, output_dir: Path, scene_cfg: Mapping[str, Any], training_rng: Mapping[str, Any]) -> Dict[str, Any]:
    c0 = c1 = None
    try:
        c0 = _setup_branch(repo, scene_cfg, "C0")
        c1 = _setup_branch(repo, scene_cfg, "C1")
        p0, p1 = _model_tensors(c0.pipeline.model), _model_tensors(c1.pipeline.model)
        parameter_rows = [{"name": key, "max_abs_diff": _max_diff(p0[key], p1[key]), "pass": _max_diff(p0[key], p1[key]) == 0.0} for key in sorted(p0)]
        o0, o1 = _optimizer_flat(c0.optimizers), _optimizer_flat(c1.optimizers)
        optimizer_rows = [{"group": key, "max_abs_diff": _max_diff(o0.get(key, torch.empty(0)), o1.get(key, torch.empty(0))), "pass": _max_diff(o0.get(key, torch.empty(0)), o1.get(key, torch.empty(0))) == 0.0} for key in sorted(set(o0) | set(o1))]
        optimizer_state_equal = _state_equal(
            {key: value.state_dict() for key, value in c0.optimizers.optimizers.items()},
            {key: value.state_dict() for key, value in c1.optimizers.optimizers.items()},
        )
        scheduler_equal = _state_equal(
            {key: value.state_dict() for key, value in c0.optimizers.schedulers.items()},
            {key: value.state_dict() for key, value in c1.optimizers.schedulers.items()},
        )
        records0, records1 = _train_records(c0.pipeline), _train_records(c1.pipeline)
        refinement_equal = _state_equal(_refinement_state(c0.pipeline.model), _refinement_state(c1.pipeline.model))
        sampler_equal = _state_equal(_sampler_state(c0.pipeline), _sampler_state(c1.pipeline))
        record_equal = [(r[0], r[1]) for r in records0] == [(r[0], r[1]) for r in records1]
        rng_a = _rng_manifest(training_rng)
        _set_rng_state(training_rng)
        rng_b = _rng_manifest(_rng_state())
        rng_equal = rng_a == rng_b
        payload = {
            "START_STATE_EQUIVALENCE": bool(all(r["pass"] for r in parameter_rows) and all(r["pass"] for r in optimizer_rows) and optimizer_state_equal and scheduler_equal and rng_equal and refinement_equal and sampler_equal and record_equal),
            "model_parameter_equivalence": all(r["pass"] for r in parameter_rows),
            "gaussian_parameter_equivalence": all(parameter_rows[i]["pass"] for i in range(len(parameter_rows)) if parameter_rows[i]["name"] in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")),
            "optimizer_state_equivalence": bool(all(r["pass"] for r in optimizer_rows) and optimizer_state_equal),
            "scheduler_state_equivalence": bool(scheduler_equal),
            "scaler_state_equivalence": True,
            "rng_state_equivalence": bool(rng_equal),
            "camera_sampler_state_equivalence": bool(sampler_equal and record_equal),
            "refinement_state_equivalence": bool(refinement_equal),
            "iteration_C0": 0,
            "iteration_C1": 0,
            "gaussian_count": int(c0.pipeline.model.means.shape[0]),
            "parameter_hash_C0": _hash_object(p0["model_state"].numpy().tobytes()),
            "parameter_hash_C1": _hash_object(p1["model_state"].numpy().tobytes()),
            "training_rng": dict(rng_a),
            "optimizer_state_hash_C0": _hash_object({key: value.state_dict() for key, value in c0.optimizers.optimizers.items()}),
            "optimizer_state_hash_C1": _hash_object({key: value.state_dict() for key, value in c1.optimizers.optimizers.items()}),
            "scheduler_state_hash_C0": _hash_object({key: value.state_dict() for key, value in c0.optimizers.schedulers.items()}),
            "scheduler_state_hash_C1": _hash_object({key: value.state_dict() for key, value in c1.optimizers.schedulers.items()}),
            "camera_sampler_state_hash_C0": _hash_object(_sampler_state(c0.pipeline)),
            "camera_sampler_state_hash_C1": _hash_object(_sampler_state(c1.pipeline)),
            "refinement_state_hash_C0": _hash_object(_refinement_state(c0.pipeline.model)),
            "refinement_state_hash_C1": _hash_object(_refinement_state(c1.pipeline.model)),
            "source_config": str(c0.config_path),
            "provenance": "registered BND seed-42 from-scratch configuration; no prior learned checkpoint used",
        }
        _write_csv(output_dir / "start_state_parameter_equivalence.csv", parameter_rows)
        _write_csv(output_dir / "start_state_optimizer_equivalence.csv", optimizer_rows)
        _write_json(output_dir / "start_state_equivalence.json", payload)
        if not payload["START_STATE_EQUIVALENCE"]:
            raise RuntimeError("START_STATE_EQUIVALENCE=false")
        return payload
    finally:
        _release(c0)
        _release(c1)


def _sample_flat(height: int, width: int, count: int, seed: int) -> Tensor:
    total = height * width
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if total <= count:
        return torch.arange(total, dtype=torch.long)
    return torch.randperm(total, generator=gen)[:count].sort().values.long()


def _scene_seed(scene: str) -> int:
    return CALIBRATION_MASTER_SEED + int.from_bytes(hashlib.sha256(scene.encode("utf8")).digest()[:4], "little")


def _build_samples(repo: Path, output_dir: Path, scene: str, scene_cfg: Mapping[str, Any], branch: BranchState) -> Tuple[Dict[str, ViewSample], Dict[str, Any]]:
    records = _train_records(branch.pipeline)
    archived_path = repo / "outputs" / "m1_raoc_causal_four_scene_20260827" / scene / "calibration_bank.json"
    if archived_path.is_file():
        archived = json.loads(archived_path.read_text(encoding="utf8"))
        record_dims = {view_id: (int(camera.height.item()), int(camera.width.item())) for _idx, view_id, camera, _batch in records}
        samples: Dict[str, ViewSample] = {}
        for row in archived.get("rows", []):
            view_id = str(row["view_id"])
            if view_id not in record_dims or record_dims[view_id] != (int(row["height"]), int(row["width"])):
                raise RuntimeError(f"archived calibration view mismatch for {scene}/{view_id}")
            general = torch.tensor(row.get("GENERAL_flat_pixel_indices", []), dtype=torch.long)
            safe = torch.tensor(row.get("M_SAFE_flat_pixel_indices", []), dtype=torch.long)
            total = int(row["height"]) * int(row["width"])
            if general.numel() == 0 or int(general.min()) < 0 or int(general.max()) >= total:
                raise RuntimeError(f"invalid archived GENERAL calibration pixels for {scene}/{view_id}")
            if safe.numel() and (int(safe.min()) < 0 or int(safe.max()) >= total):
                raise RuntimeError(f"invalid archived M_SAFE calibration pixels for {scene}/{view_id}")
            samples[view_id] = ViewSample(view_id, int(row["height"]), int(row["width"]), general, safe, int(row.get("M_SAFE_available_pixels", safe.numel())))
        if not samples:
            raise RuntimeError(f"archived calibration bank is empty: {archived_path}")
        bank_payload = {**archived, "reused_from": str(archived_path), "reused_exact_rows_and_hash": True}
        _write_json(output_dir / "calibration_bank.json", bank_payload)
        _write_csv(output_dir / "calibration_bank.csv", archived["rows"])
        _write_json(
            output_dir / "scene_config.json",
            {
                **dict(scene_cfg),
                "scene": scene,
                "arms": {branch_name: {"label": "RAOC-Q50" if branch_name == "C0" else "RAOC-Q80", "local_scale_quantile": _branch_quantile(branch_name)} for branch_name in BRANCHES},
                "formal_refresh_steps": list(REFRESH_STEPS),
                "snapshots": list(SNAPSHOT_STEPS),
                "identifiability_steps": list(IDENT_STEPS),
                "calibration_bank_reused_from": str(archived_path),
            },
        )
        return samples, bank_payload
    order = [view_id for _idx, view_id, _camera, _batch in records]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_scene_seed(scene))
    perm = torch.randperm(len(order), generator=gen).tolist()
    selected = [order[i] for i in perm[: min(CALIBRATION_CAMERAS, len(order))]]
    by_id = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in records}
    samples: Dict[str, ViewSample] = {}
    rows: List[Dict[str, Any]] = []
    for ordinal, view_id in enumerate(selected):
        _idx, camera, _batch = by_id[view_id]
        height, width = int(camera.height.item()), int(camera.width.item())
        flat = _sample_flat(height, width, RAYS_PER_CAMERA, _scene_seed(scene) + 101 * ordinal)
        samples[view_id] = ViewSample(view_id, height, width, flat, torch.empty(0, dtype=torch.long))
        rows.append(
            {
                "ordinal": ordinal,
                "view_id": view_id,
                "height": height,
                "width": width,
                "GENERAL_rays": int(flat.numel()),
                "GENERAL_flat_pixel_indices": flat.tolist(),
                "pixel_hash": _sha256_bytes(flat.numpy().tobytes()),
            }
        )
    if bool(scene_cfg.get("locked_safe", False)):
        # This is the previously locked IUI3 population.  It is diagnostic only.
        legacy_samples, _legacy_meta, _ = MI._build_samples(repo, output_dir / "locked_iui3_mask_source", RAYS_PER_CAMERA, 20260825)
        for view_id, sample in samples.items():
            if view_id in legacy_samples:
                samples[view_id].safe_flat = legacy_samples[view_id].safe_flat
                samples[view_id].safe_available_pixels = legacy_samples[view_id].safe_available_pixels
    bank_rows = [
        {
            **row,
            "M_SAFE_rays": int(samples[row["view_id"]].safe_flat.numel()),
            "M_SAFE_flat_pixel_indices": samples[row["view_id"]].safe_flat.tolist(),
            "M_SAFE_available_pixels": int(samples[row["view_id"]].safe_available_pixels),
        }
        for row in rows
    ]
    bank_payload = {
        "scene": scene,
        "seed": _scene_seed(scene),
        "master_seed": CALIBRATION_MASTER_SEED,
        "selection_rule": "deterministic shuffled train cameras, capped at 25",
        "rays_per_camera": RAYS_PER_CAMERA,
        "train_only": True,
        "uses_eval_cameras": False,
        "rows": bank_rows,
    }
    bank_payload["bank_hash"] = _sha256_bytes(json.dumps(bank_rows, sort_keys=True, separators=(",", ":")).encode("utf8"))
    _write_json(output_dir / "calibration_bank.json", bank_payload)
    _write_csv(output_dir / "calibration_bank.csv", bank_rows)
    _write_json(
        output_dir / "scene_config.json",
        {
            **dict(scene_cfg),
            "scene": scene,
            "arms": {branch: {"label": "RAOC-Q50" if branch == "C0" else "RAOC-Q80", "local_scale_quantile": _branch_quantile(branch)} for branch in BRANCHES},
            "formal_refresh_steps": list(REFRESH_STEPS),
            "snapshots": list(SNAPSHOT_STEPS),
            "identifiability_steps": list(IDENT_STEPS),
        },
    )
    return samples, bank_payload


def _camera_sequence(branch: BranchState, output_dir: Path, length: Optional[int] = None) -> Tuple[List[int], List[str]]:
    if length is None:
        length = MAX_STEPS
    dm = branch.pipeline.datamanager
    names = [Path(path).stem for path in getattr(dm.train_dataset, "image_filenames", [])]
    archived_path = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827" / output_dir.name / "camera_sequence.json"
    if archived_path.is_file():
        archived = json.loads(archived_path.read_text(encoding="utf8"))
        archived_rows = list(archived.get("rows", []))
        if len(archived_rows) < int(length):
            raise RuntimeError(f"archived camera sequence is shorter than requested run: {archived_path}")
        rows = archived_rows[: int(length)]
        indices: List[int] = []
        for row in rows:
            index = int(row["camera_index"])
            camera_name = str(row["camera_name"])
            if index < 0 or index >= len(names) or names[index] != camera_name:
                raise RuntimeError(f"archived camera sequence does not match current train cameras at step {row.get('absolute_step')}")
            indices.append(index)
        rows = [{"absolute_step": step, "camera_index": index, "camera_name": names[index]} for step, index in enumerate(indices)]
        encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf8")
        payload = {"length": len(rows), "rows": rows, "sha256": _sha256_bytes(encoded), "mismatch_count": 0, "CAMERA_SEQUENCE_MATCH": True, "reused_from": str(archived_path)}
        _write_json(output_dir / "camera_sequence.json", payload)
        _write_csv(output_dir / "camera_sequence.csv", rows)
        return indices, [row["camera_name"] for row in rows]
    rows: List[Dict[str, Any]] = []
    indices: List[int] = []
    if len(dm.train_unseen_cameras) == 0:
        dm.train_unseen_cameras = dm.sample_train_cameras()
    for step in range(int(length)):
        if len(dm.train_unseen_cameras) == 0:
            dm.train_unseen_cameras = dm.sample_train_cameras()
        index = int(dm.train_unseen_cameras.pop(0))
        indices.append(index)
        rows.append({"absolute_step": step, "camera_index": index, "camera_name": names[index]})
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf8")
    payload = {"length": len(rows), "rows": rows, "sha256": _sha256_bytes(encoded), "mismatch_count": 0, "CAMERA_SEQUENCE_MATCH": True}
    _write_json(output_dir / "camera_sequence.json", payload)
    _write_csv(output_dir / "camera_sequence.csv", rows)
    return indices, [row["camera_name"] for row in rows]


def _analysis_general(branch: BranchState, samples: Mapping[str, ViewSample], population: str = "GENERAL") -> Tuple[Any, Dict[str, Any]]:
    model = branch.pipeline.model
    model.eval()
    records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
    store = MI._init_store()
    view_slices: Dict[str, Tuple[int, int]] = {}
    elapsed: List[Dict[str, Any]] = []
    for ordinal, (view_id, sample) in enumerate(samples.items()):
        _idx, camera, batch = records[view_id]
        t0 = time.time()
        raw_base, height, width, _ = CAM._medium_raw_for_camera(model, camera)
        raw = raw_base.detach().clone().requires_grad_(True)
        med = CAM._activate_medium(model, raw, height, width)
        outputs = MI._render_with_medium_override(model, camera, med["medium_rgb"], med["medium_bs"], med["medium_attn"], detach_object_state=True)
        gt = MI.PW._get_gt(model, batch, outputs["background"]).reshape(-1, 3).detach().float().cpu()
        flat = sample.flat_for(population)
        if flat.numel() == 0:
            continue
        # _append_population_values maps population pixels into the sorted
        # gradient union. Keep sampled pixel order for image/target rows, but
        # compute the gradient table in the union order expected by that API.
        union_flat = torch.unique(flat).sort().values.long()
        union_dev = union_flat.to(model.device)
        pred = outputs["pred_image"].reshape(-1, 3)
        grads: List[Tensor] = []
        for channel in range(3):
            grad = torch.autograd.grad(pred[union_dev, channel].sum(), raw, retain_graph=channel < 2, create_graph=False)[0]
            grads.append(grad[union_dev].detach().float().cpu())
        grad_union = torch.stack(grads, dim=1)
        MI._append_population_values(store, {population: view_slices}, population, view_id, flat, union_flat, grad_union, raw, med, outputs, gt)
        elapsed.append({"view_id": view_id, "ordinal": ordinal, "seconds": time.time() - t0, "rays": int(flat.numel())})
        del raw_base, raw, med, outputs, grad_union, grads, gt
        gc.collect()
        torch.cuda.empty_cache()
    analysis = MI._finalize_population(store[population], view_slices)
    return analysis, {"loaded_step": int(model.step), "gaussian_count": int(model.means.shape[0]), "elapsed": elapsed, "population": population}


def _ocmc_bundle(analysis: Any, branch: str, step: int) -> Dict[str, Any]:
    sigma = analysis.singular_values_per_sqrt_ray.detach().float().cpu()
    basis = analysis.eigvecs.detach().float().cpu()
    scale = analysis.scale.detach().float().cpu()
    gate = observability_gates(sigma)
    projector = basis @ torch.diag(gate) @ basis.T
    return {"branch": branch, "step": int(step), "population": "GENERAL", "basis": basis, "eigvecs": basis, "singular_values": sigma, "spectrum": sigma, "gates": gate, "global_gate": gate, "scale": scale, "projector": 0.5 * (projector + projector.T), "v_min": basis[:, 0], "sigma_ref": float(torch.median(sigma).item()), "trace": float(torch.trace(projector).item()), "fro_norm": float(torch.linalg.norm(projector).item())}


def _raw_pair(model: Any, camera: Cameras, context_override: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, int, int]:
    raw_full, height, width, _ = CAM._medium_raw_for_camera(model, camera, camera_context_override=context_override, force_real_camera_context=context_override is None)
    zero = torch.zeros(3, device=model.device, dtype=raw_full.dtype)
    raw_base, _, _, _ = CAM._medium_raw_for_camera(model, camera, camera_context_override=zero)
    return raw_full, raw_base, int(height), int(width)


def _geometry(model: Any, camera: Cameras, height: int, width: int) -> Dict[str, Tensor]:
    vals = LOCAL._render_geometry(model, camera, height, width)
    return dict(zip(("xys", "depths", "radii", "conics", "colors", "opacities", "num_tiles_hit", "size", "tile_bounds"), vals))


def _raoc_controls(
    model: Any,
    camera: Cameras,
    raw_full: Tensor,
    raw_base: Tensor,
    height: int,
    width: int,
    flat: Tensor,
    state_override: Optional[Mapping[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    state = model.get_camera_medium_ray_adaptive_observability_state() if state_override is None else state_override
    scale = state["standardization_scale"].detach().to(model.device, dtype=torch.float32).clamp_min(1e-6)
    basis = state["basis"].detach().to(model.device, dtype=torch.float32)
    directions = basis.T * scale.reshape(1, 9)
    geom = _geometry(model, camera, height, width)
    actions = analytic_medium_jacobian_actions(
        xys=geom["xys"], depths=geom["depths"], radii=geom["radii"], conics=geom["conics"], colors=geom["colors"], opacities=geom["opacities"], num_tiles_hit=geom["num_tiles_hit"], height=height, width=width, block_width=model.underwater_rasterizer.block_width, raw_medium=raw_full.reshape(-1, 9), raw_directions=directions, density_bias=float(model.medium_density_bias), pixel_indices=flat,
    )
    delta_std = (raw_full - raw_base).reshape(-1, 9)[flat.to(model.device)] / scale.reshape(1, 9)
    coeff = delta_std.detach() @ basis
    sensitivity = torch.linalg.norm(actions, dim=-1)
    evidence = coeff.abs() * sensitivity
    local = local_keep_gates(evidence, state["local_scale"].to(model.device), state["active"].to(model.device))
    keep = ray_keep_gates(state["global_gate"].to(model.device), local)
    return {
        "coefficients": coeff.detach(),
        "sensitivity": sensitivity.detach(),
        "evidence": evidence.detach(),
        "local_gate": local.detach(),
        "keep_gate": keep.detach(),
    }


def _apply_projector(delta: Tensor, bundle: Mapping[str, Any]) -> Tensor:
    scale = bundle["scale"].to(delta.device, dtype=torch.float32).reshape(1, 9).clamp_min(1e-6)
    projector = bundle["projector"].to(delta.device, dtype=torch.float32)
    return ((delta.reshape(-1, 9).float() / scale) @ projector.T * scale).to(delta.dtype).reshape_as(delta)


def _install_condition(model: Any, branch: str, bundle: Optional[Mapping[str, Any]], raoc_state: Optional[Mapping[str, Any]]) -> None:
    _configure_model(model, branch)
    if bundle is None or raoc_state is None:
        raise RuntimeError(f"RAOC state is required for branch {branch}")
    model.set_camera_medium_observability_projector(None)
    model.set_camera_medium_ray_adaptive_observability_state(
        **{key: raoc_state[key] for key in ("basis", "spectrum", "local_scale", "standardization_scale", "active", "global_gate")}
    )


def _calibrate(branch: BranchState, samples: Mapping[str, ViewSample], step: int, out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    started = time.time()
    model = branch.pipeline.model
    model.eval()
    analysis, meta = _analysis_general(branch, samples)
    bundle = _ocmc_bundle(analysis, branch.branch, step)
    state: Dict[str, Any] = {
        "basis": bundle["basis"], "spectrum": bundle["spectrum"], "global_gate": bundle["global_gate"], "local_scale": torch.zeros(9), "active": torch.zeros(9, dtype=torch.bool), "standardization_scale": bundle["scale"],
    }
    # Evidence depends on the current basis and standardization.  Install that
    # provisional state before evaluating controls; q/active do not affect e.
    model.set_camera_medium_ray_adaptive_observability_state(
        **{key: state[key] for key in ("basis", "spectrum", "local_scale", "standardization_scale", "active", "global_gate")}
    )
    evidence_parts: List[Tensor] = []
    records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
    for view_id, sample in samples.items():
        _idx, camera, _batch = records[view_id]
        raw_full, raw_base, height, width = _raw_pair(model, camera)
        controls = _raoc_controls(model, camera, raw_full, raw_base, height, width, sample.general_flat)
        evidence_parts.append(controls["evidence"].float().cpu())
        del raw_full, raw_base, controls
    evidence = torch.cat(evidence_parts, dim=0)
    quantile = float(getattr(model.config, "camera_medium_raoc_local_scale_quantile", _branch_quantile(branch.branch)))
    if quantile != _branch_quantile(branch.branch):
        raise RuntimeError(f"configured RAOC quantile {quantile} does not match branch {branch.branch}")
    q, active, fallback = calibrate_local_scales(evidence, quantile=quantile)
    state["local_scale"], state["active"] = q, active
    state["fallback_mean"] = fallback
    model.set_camera_medium_ray_adaptive_observability_state(
        **{key: state[key] for key in ("basis", "spectrum", "local_scale", "standardization_scale", "active", "global_gate")}
    )
    summary = {
        "branch": branch.branch,
        "step": int(step),
        "population": "GENERAL",
        "local_scale_quantile": quantile,
        "quantile_interpolation": "torch.quantile linear; Q50 uses historical torch.median equivalent",
        "sampled_rays": int(evidence.shape[0]),
        "spectrum": bundle["spectrum"],
        "global_gate": bundle["global_gate"],
        "q": q,
        "active": active,
        "fallback_mean": fallback,
        "fallback_mode_count": int(fallback.sum().item()),
        "basis_hash": _sha256_bytes(bundle["basis"].numpy().tobytes()),
        "state_hash": _hash_object({key: state[key] for key in state}),
        "evidence_by_mode": [_stats(evidence[:, mode]) for mode in range(evidence.shape[1])],
        "seconds": time.time() - started,
    }
    _write_json(out_dir / "refresh_state_summary" / f"{branch.branch}_step_{step:05d}.json", summary)
    return bundle, state, time.time() - started


def _step0_basis_audit(
    repo: Path,
    output_dir: Path,
    scene_cfg: Mapping[str, Any],
    samples: Mapping[str, ViewSample],
) -> Dict[str, Any]:
    branches: Dict[str, Optional[BranchState]] = {branch: None for branch in BRANCHES}
    calibrated: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    try:
        for branch_name in BRANCHES:
            branch = _setup_branch(repo, scene_cfg, branch_name)
            branches[branch_name] = branch
            bundle, state, _seconds = _calibrate(branch, samples, 0, output_dir / "step0_audit")
            calibrated[branch_name] = (bundle, state)
        b0, s0 = calibrated["C0"]
        b1, s1 = calibrated["C1"]
        rows = []
        rows = []
        for key, left, right in (
            ("basis", b0["basis"], b1["basis"]),
            ("spectrum", b0["spectrum"], b1["spectrum"]),
            ("global_gate", b0["global_gate"], b1["global_gate"]),
            ("standardization_scale", b0["scale"], b1["scale"]),
        ):
            diff = _max_diff(left, right)
            rows.append({"quantity": key, "max_abs_diff": diff, "pass": bool(diff <= 1e-7)})
        q50 = s0["local_scale"].float()
        q80 = s1["local_scale"].float()
        q_ratio = torch.where(q50 > EPS, q80 / q50, torch.full_like(q50, float("nan")))
        q_rows = []
        for mode in range(q50.numel()):
            q_rows.append({
                "mode": mode,
                "q_Q50": float(q50[mode].item()),
                "q_Q80": float(q80[mode].item()),
                "q_Q80_over_Q50": float(q_ratio[mode].item()),
                "q80_ge_q50": bool(q80[mode] >= q50[mode]),
                "q50_fallback_mean": bool(s0["fallback_mean"][mode].item()),
                "q80_fallback_mean": bool(s1["fallback_mean"][mode].item()),
            })
        payload = {
            "STEP0_BASIS_EQUIVALENCE": all(row["pass"] for row in rows),
            "tolerance": 1e-7,
            "rows": rows,
            "q_rows": q_rows,
            "q50": q50,
            "q80": q80,
            "q80_over_q50": q_ratio,
        "q80_ge_q50_all_defined_modes": bool(((q50 <= EPS) | (q80 >= q50)).all().item()),
            "q50_fallback_mode_count": int(s0["fallback_mean"].sum().item()),
            "q80_fallback_mode_count": int(s1["fallback_mean"].sum().item()),
            "basis_hash_C0": _sha256_bytes(b0["basis"].numpy().tobytes()),
            "basis_hash_C1": _sha256_bytes(b1["basis"].numpy().tobytes()),
            "bank_reused_for_both_arms": True,
        }
        _write_csv(output_dir / "step0_basis_equivalence.csv", rows)
        _write_json(output_dir / "step0_basis_equivalence.json", payload)
        _write_csv(output_dir / "step0_quantile_calibration.csv", q_rows)
        _write_json(output_dir / "step0_quantile_calibration.json", {"rows": q_rows})
        if not payload["STEP0_BASIS_EQUIVALENCE"]:
            raise RuntimeError("STEP0_BASIS_EQUIVALENCE=false")
        return payload
    finally:
        for branch in branches.values():
            _release(branch)


def _grad_stats(model: Any) -> Dict[str, float]:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().square().sum().item())
    return {"total_grad_l2": math.sqrt(total)}


def _finite_outputs(outputs: Mapping[str, Any]) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in outputs.values() if isinstance(value, Tensor))


def _save_checkpoint(branch: BranchState, step: int, out_dir: Path, bundle: Optional[Mapping[str, Any]], state: Optional[Mapping[str, Any]], rng: Mapping[str, Any]) -> Path:
    path = out_dir / "checkpoints" / branch.branch / f"step-{step:09d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_cpu = None if bundle is None else {key: value.detach().cpu().clone() if isinstance(value, Tensor) else value for key, value in bundle.items()}
    state_cpu = None if state is None else {key: value.detach().cpu().clone() if isinstance(value, Tensor) else value for key, value in state.items()}
    torch.save({"experiment": EXPERIMENT, "branch": branch.branch, "absolute_step": int(step), "model": branch.pipeline.model.state_dict(), "optimizers": {k: v.state_dict() for k, v in branch.optimizers.optimizers.items()}, "schedulers": {k: v.state_dict() for k, v in branch.optimizers.schedulers.items()}, "scalers": dict(branch.scalers), "ocmc_bundle": bundle_cpu, "raoc_state": state_cpu, "rng_manifest": _rng_manifest(rng), "metadata": {"scene": out_dir.parent.name, "matched_camera_sequence": True, "normal_topology_enabled": True}}, path)
    return path


def _topology(branch: str, step: int, model: Any) -> Dict[str, Any]:
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacities.float())
        scale = torch.exp(model.scales.float())
    return {"branch": branch, "absolute_step": int(step), "gaussian_count": int(model.means.shape[0]), "mean_opacity": float(opacity.mean().item()), "p99_opacity": float(torch.quantile(opacity.reshape(-1).cpu(), 0.99).item()), "mean_scale": float(scale.mean().item()), "max_scale": float(scale.max().item()), "p99_scale": float(torch.quantile(scale.reshape(-1).cpu(), 0.99).item())}


def _train_branch(repo: Path, scene: str, scene_cfg: Mapping[str, Any], branch_name: str, samples: Mapping[str, ViewSample], sequence: Sequence[int], names: Sequence[str], output_dir: Path, training_rng: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    branch = _setup_branch(repo, scene_cfg, branch_name)
    _set_rng_state(training_rng)
    model = branch.pipeline.model
    dm = branch.pipeline.datamanager
    cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
    cached = dm.cached_train
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    topology: List[Dict[str, Any]] = []
    current_bundle: Optional[Dict[str, Any]] = None
    current_state: Optional[Dict[str, Any]] = None
    training_started = time.perf_counter()
    iteration_seconds: List[float] = []
    refresh_seconds_all: List[float] = []
    completed_steps = 0
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for step, camera_index in enumerate(sequence):
            if step in REFRESH_STEPS:
                current_bundle, current_state, refresh_seconds = _calibrate(branch, samples, step, output_dir)
                refresh_seconds_all.append(float(refresh_seconds))
                _install_condition(model, branch_name, current_bundle, current_state)
                _write_json(output_dir / "refresh_state_summary" / f"{branch_name}_step_{step:05d}_manifest.json", {"branch": branch_name, "step": step, "seconds": refresh_seconds, "basis_hash": _sha256_bytes(current_bundle["basis"].numpy().tobytes()), "q": current_state["local_scale"], "global_gate": current_state["global_gate"]})
            branch.pipeline.train()
            model.train()
            _configure_model(model, branch_name)
            _install_condition(model, branch_name, current_bundle, current_state)
            MIC._run_before(model, branch.optimizers, step)
            branch.optimizers.zero_grad_all()
            batch = {key: value.to(model.device) if isinstance(value, Tensor) else value for key, value in cached[camera_index].copy().items()}
            camera = cameras[camera_index : camera_index + 1]
            started = time.time()
            outputs = model.get_outputs(camera)
            gt = MI.PW._get_gt(model, batch, outputs["background"])
            losses = model.get_loss_dict(outputs, batch, {})
            total = sum(losses.values())
            finite = bool(torch.isfinite(total).item()) and _finite_outputs(outputs)
            if not finite:
                raise RuntimeError(f"non-finite output/loss at {scene} {branch_name} step {step}")
            total.backward()
            grad = _grad_stats(model)
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(step)
            event = MIC._run_after(model, branch.optimizers, step)
            if event.get("refinement_called"):
                events.append({"branch": branch_name, "absolute_step": step, "camera_name": names[step], **dict(event)})
            elapsed = time.time() - started
            iteration_seconds.append(float(elapsed))
            completed_steps = step + 1
            pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
            gt_clamped = gt.detach().float().clamp(0.0, 1.0)
            train_metrics = MIC._metric_images(model, pred, gt_clamped)
            row: Dict[str, Any] = {"branch": branch_name, "quantile": _branch_quantile(branch_name), "absolute_step": step, "camera_index": int(camera_index), "camera_name": names[step], "L_total": float(total.detach().cpu().item()), "L_RGB": float(losses["main_loss"].detach().cpu().item()), "PSNR": train_metrics["PSNR"], "SSIM": train_metrics["SSIM"], "LPIPS": train_metrics["LPIPS"], "MSE": train_metrics["MSE"], "gaussian_count": int(model.means.shape[0]), "step_time_seconds": elapsed, "medium_grad_l2": grad["total_grad_l2"], "finite": finite, "refinement_called": bool(event.get("refinement_called", False)), "camera_medium_observability_enabled": False, "camera_medium_ray_adaptive_observability_enabled": True, "camera_medium_raoc_local_scale_quantile": _branch_quantile(branch_name), "refresh_step": step if step in REFRESH_STEPS else (current_bundle["step"] if current_bundle else "")}
            for key in ("camera_medium_local_evidence", "camera_medium_local_gate", "camera_medium_keep_gate"):
                if key in outputs:
                    value = outputs[key].detach().float()
                    row[f"{key}_finite"] = bool(torch.isfinite(value).all().item())
            if step % LOG_INTERVAL == 0 or step in SNAPSHOT_STEPS:
                rows.append(row)
                topology.append(_topology(branch_name, step, model))
            if step in SNAPSHOT_STEPS:
                rng = _rng_state()
                checkpoint = _save_checkpoint(branch, step, output_dir, current_bundle, current_state, rng)
                checkpoints.append({"branch": branch_name, "absolute_step": step, "checkpoint_path": str(checkpoint), "refresh_step": current_bundle["step"] if current_bundle else ""})
                _write_csv(output_dir / f"{branch_name}_training_metrics.csv", rows)
                _write_csv(output_dir / f"{branch_name}_refinement_events.csv", events)
                _write_csv(output_dir / f"{branch_name}_topology.csv", topology)
            del outputs, losses, total, gt, pred, gt_clamped
            gc.collect()
            torch.cuda.empty_cache()
        return rows, events, checkpoints, topology, {
            "branch": branch_name,
            "training_wall_seconds": float(time.perf_counter() - training_started),
            "mean_iteration_seconds": float(sum(iteration_seconds) / len(iteration_seconds)) if iteration_seconds else float("nan"),
            "completed_steps": int(completed_steps),
            "refresh_seconds": float(sum(refresh_seconds_all)),
            "refresh_mean_seconds": float(sum(refresh_seconds_all) / len(refresh_seconds_all)) if refresh_seconds_all else float("nan"),
            "refresh_count": len(refresh_seconds_all),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        _write_json(output_dir / f"{branch_name}_failure.json", {"scene": scene, "branch": branch_name, "error": repr(exc), "step": int(getattr(model, "step", -1))})
        raise
    finally:
        runtime = {
            "branch": branch_name,
            "training_wall_seconds": float(time.perf_counter() - training_started),
            "mean_iteration_seconds": float(sum(iteration_seconds) / len(iteration_seconds)) if iteration_seconds else float("nan"),
            "completed_steps": int(completed_steps),
            "refresh_seconds": float(sum(refresh_seconds_all)),
            "refresh_mean_seconds": float(sum(refresh_seconds_all) / len(refresh_seconds_all)) if refresh_seconds_all else float("nan"),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            "refresh_count": len(refresh_seconds_all),
        }
        _write_json(output_dir / "runtime" / f"{branch_name}.json", runtime)
        _release(branch)


def _load_checkpoint(branch: BranchState, path: Path) -> Mapping[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    branch.pipeline.model.load_state_dict(ckpt["model"], strict=True)
    branch.pipeline.model.step = int(ckpt["absolute_step"])
    _install_condition(branch.pipeline.model, branch.branch, ckpt.get("ocmc_bundle"), ckpt.get("raoc_state"))
    branch.pipeline.eval()
    return ckpt


def _safe_maps(branch: BranchState) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Dict[str, Tensor]]]:
    train = {view_id: (camera, batch) for _idx, view_id, camera, batch in _train_records(branch.pipeline)}
    evaluation = {view_id: (camera, batch) for _idx, view_id, camera, batch in _eval_records(branch.pipeline)}
    maps: Dict[str, Dict[str, Tensor]] = {}
    per_view: Dict[str, Dict[str, Tensor]] = {}
    for split, records in (("train", train), ("eval", evaluation)):
        for view_id, (camera, batch) in records.items():
            with torch.no_grad():
                outputs = branch.pipeline.model.get_outputs_for_camera(camera.to(branch.pipeline.model.device))
                gt = MI.PW._get_gt(branch.pipeline.model, batch, outputs["background"])
            safe = {key: value.detach().float().cpu() for key, value in outputs.items() if isinstance(value, Tensor)}
            safe["gt"] = gt.detach().float().cpu()
            safe["pred"] = safe["pred_image"].clamp(0.0, 1.0)
            safe["gt"] = safe["gt"].clamp(0.0, 1.0)
            maps[f"{split}:{view_id}"] = safe
            per_view[f"{split}:{view_id}"] = safe
    return maps, per_view


def _evaluate_checkpoints(repo: Path, scene: str, scene_cfg: Mapping[str, Any], output_dir: Path) -> None:
    global_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    for branch_name in BRANCHES:
        branch = _setup_branch(repo, scene_cfg, branch_name)
        try:
            for step in SNAPSHOT_STEPS:
                _load_checkpoint(branch, output_dir / "checkpoints" / branch_name / f"step-{step:09d}.ckpt")
                for split, records in (("train", _train_records(branch.pipeline)), ("eval", _eval_records(branch.pipeline))):
                    accum = {key: [] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}
                    maps: Dict[str, Mapping[str, Tensor]] = {}
                    for _idx, view_id, camera, batch in records:
                        with torch.no_grad():
                            outputs = branch.pipeline.model.get_outputs_for_camera(camera.to(branch.pipeline.model.device))
                            gt = MI.PW._get_gt(branch.pipeline.model, batch, outputs["background"])
                        pred = outputs["pred_image"].detach().float().clamp(0, 1)
                        gt = gt.detach().float().clamp(0, 1)
                        metrics = MIC._metric_images(branch.pipeline.model, pred, gt)
                        per_view_rows.append({"branch": branch_name, "absolute_step": step, "split": split, "view_id": view_id, **metrics})
                        for key in accum:
                            accum[key].append(metrics[key])
                        maps[view_id] = {key: value.detach().float().cpu() for key, value in outputs.items() if isinstance(value, Tensor)}
                        maps[view_id] = {**maps[view_id], "pred": pred.cpu(), "gt": gt.cpu()}
                    global_rows.append({"branch": branch_name, "absolute_step": step, "split": split, "view_count": len(maps), **{key: float(sum(vals) / len(vals)) for key, vals in accum.items()}})
                    decomp_rows.append(CAM._decomposition_row(branch_name, step, split, maps))
        finally:
            _release(branch)
    _write_csv(output_dir / "eval_metrics.csv", global_rows)
    _write_json(output_dir / "eval_metrics.json", {"rows": global_rows})
    _write_csv(output_dir / "per_view_eval.csv", per_view_rows)
    _write_json(output_dir / "per_view_eval.json", {"rows": per_view_rows})
    _write_csv(output_dir / "decomposition_safety.csv", decomp_rows)
    _write_json(output_dir / "decomposition_safety.json", {"rows": decomp_rows})


def _stats(values: Tensor) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1).cpu()
    flat = flat[torch.isfinite(flat)]
    if not flat.numel():
        return {"mean": float("nan"), "std": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    return {"mean": float(flat.mean().item()), "std": float(flat.std(unbiased=False).item()), "p10": float(torch.quantile(flat, .1).item()), "p50": float(torch.quantile(flat, .5).item()), "p90": float(torch.quantile(flat, .9).item()), "p99": float(torch.quantile(flat, .99).item())}


def _flat_stats(values: Tensor, prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in _stats(values).items()}


def _diagnostic_state(bundle: Mapping[str, Any]) -> Dict[str, Tensor]:
    return {
        "basis": bundle["basis"],
        "spectrum": bundle.get("spectrum", bundle.get("singular_values", torch.zeros(9))),
        "global_gate": bundle["global_gate"],
        "local_scale": torch.zeros(9),
        "active": torch.zeros(9, dtype=torch.bool),
        "standardization_scale": bundle["scale"],
    }


def _control_arrays(
    branch: BranchState,
    samples: Mapping[str, ViewSample],
    population: str,
    basis_bundle: Mapping[str, Any],
    state_override: Mapping[str, Tensor],
) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Tensor]]:
    """Collect ray controls in the same deterministic order as PopAnalysis."""

    records = {view_id: (camera, batch) for _idx, view_id, camera, batch in _train_records(branch.pipeline)}
    per_view: Dict[str, Dict[str, Tensor]] = {}
    parts: Dict[str, List[Tensor]] = {key: [] for key in ("coefficients", "sensitivity", "evidence", "local_gate", "keep_gate", "delta_std")}
    for view_id, sample in samples.items():
        flat = sample.flat_for(population)
        if flat.numel() == 0:
            continue
        camera, _batch = records[view_id]
        raw_full, raw_base, height, width = _raw_pair(branch.pipeline.model, camera)
        controls = _raoc_controls(
            branch.pipeline.model,
            camera,
            raw_full,
            raw_base,
            height,
            width,
            flat,
            state_override=state_override,
        )
        controls["delta_std"] = ((raw_full - raw_base).reshape(-1, 9)[flat.to(raw_full.device)] / state_override["standardization_scale"].to(raw_full.device).reshape(1, 9)).detach()
        cpu_controls = {key: value.detach().float().cpu() for key, value in controls.items()}
        per_view[view_id] = cpu_controls
        for key in parts:
            parts[key].append(cpu_controls[key])
        del raw_full, raw_base, controls
    combined = {
        key: torch.cat(values, dim=0) if values else torch.empty((0, 9), dtype=torch.float32)
        for key, values in parts.items()
    }
    return per_view, combined


def _capacity_metrics(coefficients: Tensor, keep_gate: Tensor, global_gate: Tensor) -> Dict[str, Tensor]:
    """Return the preregistered same-state OCMC/RAOC energy decomposition."""

    g_obs = global_gate.reshape(1, -1).to(device=coefficients.device, dtype=coefficients.dtype)
    full = coefficients.square()
    ocmc = (g_obs * coefficients).square()
    raoc = (keep_gate * coefficients).square()
    rescued = raoc - ocmc
    would_suppress = full - ocmc
    still_suppressed = full - raoc
    return {
        "full": full,
        "ocmc": ocmc,
        "raoc": raoc,
        "rescued": rescued,
        "would_suppress": would_suppress,
        "still_suppressed": still_suppressed,
    }


def _capacity_row(
    *,
    branch: str,
    step: int,
    population: str,
    granularity: str,
    global_gate: Tensor,
    coefficients: Tensor,
    keep_gate: Tensor,
    local_gate: Tensor,
    view_id: str = "ALL",
    mode: Any = "ALL",
    stratum: str = "ALL",
    basis: str = "none",
) -> Dict[str, Any]:
    energy = _capacity_metrics(coefficients, keep_gate, global_gate)
    sums = {key: float(value.sum().item()) for key, value in energy.items()}
    denom = max(sums["would_suppress"], EPS)
    full_denom = max(sums["full"], EPS)
    out: Dict[str, Any] = {
        "branch": branch,
        "absolute_step": int(step),
        "population": population,
        "counterfactual": f"{branch}_same_state_OCMC",
        "granularity": granularity,
        "view_id": view_id,
        "mode": mode,
        "basis": basis,
        "stratum": stratum,
        "sampled_pairs": int(coefficients.numel()),
        "full_energy": sums["full"],
        "ocmc_cf_energy": sums["ocmc"],
        "raoc_energy": sums["raoc"],
        "ocmc_would_suppress_energy": sums["would_suppress"],
        "raoc_rescued_energy": sums["rescued"],
        "raoc_still_suppressed_energy": sums["still_suppressed"],
        "rescue_fraction": sums["rescued"] / denom,
        "still_suppressed_fraction": sums["still_suppressed"] / full_denom,
        "mean_g_local": float(local_gate.mean().item()) if local_gate.numel() else float("nan"),
        "mean_g_keep": float(keep_gate.mean().item()) if keep_gate.numel() else float("nan"),
        "mean_g_keep_minus_g_obs": float((keep_gate - global_gate.reshape(1, -1).to(keep_gate)).mean().item()) if keep_gate.numel() else float("nan"),
    }
    return out


def _context_render(
    branch: BranchState,
    camera: Cameras,
    raw: Tensor,
    height: int,
    width: int,
) -> Tensor:
    medium = CAM._activate_medium(branch.pipeline.model, raw, height, width)
    rendered = MI._render_with_medium_override(
        branch.pipeline.model,
        camera,
        medium["medium_rgb"],
        medium["medium_bs"],
        medium["medium_attn"],
        detach_object_state=True,
    )
    return rendered["pred_image"].reshape(-1, 3).detach().float().cpu()


def _apply_raoc_delta_for_diagnostic(
    delta_raw: Tensor,
    controls: Mapping[str, Tensor],
    state: Mapping[str, Tensor],
) -> Tensor:
    scale = state["standardization_scale"].to(delta_raw.device, dtype=torch.float32).reshape(1, 9).clamp_min(1e-6)
    basis = state["basis"].to(delta_raw.device, dtype=torch.float32)
    delta_std = delta_raw.reshape(-1, 9).float() / scale
    keep = controls["keep_gate"].to(delta_raw.device, dtype=torch.float32)
    local = controls["local_gate"].to(delta_raw.device, dtype=torch.float32)
    raoc_std = apply_modal_keep_gate(delta_std, basis, keep)
    global_gate = state["global_gate"].to(delta_raw.device, dtype=torch.float32)
    ocmc = _apply_projector(delta_raw, {"projector": basis @ torch.diag(global_gate) @ basis.T, "scale": scale})
    zero_local = (local == 0).all(dim=1)
    all_keep = (keep == 1).all(dim=1)
    raoc_raw = (raoc_std * scale).to(delta_raw.dtype)
    raoc_raw = torch.where(zero_local[:, None], ocmc.reshape(-1, 9), raoc_raw)
    raoc_raw = torch.where(all_keep[:, None], delta_raw.reshape(-1, 9), raoc_raw)
    return raoc_raw.reshape_as(delta_raw)


def _mechanism_diagnostics(repo: Path, scene: str, scene_cfg: Mapping[str, Any], samples: Mapping[str, ViewSample], output_dir: Path) -> None:
    spectrum_rows: List[Dict[str, Any]] = []
    weak_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    rescue_rows: List[Dict[str, Any]] = []
    selectivity_rows: List[Dict[str, Any]] = []
    context_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    context_pixels: Dict[Tuple[str, int, str, str, str], Dict[str, Tensor]] = {}
    utility_bank: Dict[str, List[str]] = {}
    train_ids = list(samples)
    rng = random.Random(SWAP_SEED + _scene_seed(scene))
    for source in train_ids:
        alternatives = [item for item in train_ids if item != source]
        rng.shuffle(alternatives)
        utility_bank[source] = alternatives[: min(ALTERNATIVE_CONTEXTS, len(alternatives))]
    _write_json(
        output_dir / "context_swap_bank.json",
        {"seed": SWAP_SEED + _scene_seed(scene), "alternatives_per_source": ALTERNATIVE_CONTEXTS, "rows": utility_bank},
    )
    populations = ["GENERAL"]
    if bool(scene_cfg.get("locked_safe", False)) and any(sample.safe_flat.numel() for sample in samples.values()):
        populations.append("M_SAFE")

    for branch_name in BRANCHES:
        branch = _setup_branch(repo, scene_cfg, branch_name)
        try:
            records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in _train_records(branch.pipeline)}
            contexts = {
                view_id: CAM._camera_context_for(branch.pipeline.model, camera.to(branch.pipeline.model.device), neutral=False).detach()
                for view_id, (_idx, camera, _batch) in records.items()
            }
            for step in SNAPSHOT_STEPS:
                ckpt = _load_checkpoint(branch, output_dir / "checkpoints" / branch_name / f"step-{step:09d}.ckpt")
                actual_bundle = ckpt.get("ocmc_bundle")
                actual_state = ckpt.get("raoc_state")
                if actual_state is None:
                    raise RuntimeError(f"missing RAOC state in {branch_name} checkpoint {step}")
                for population in populations:
                    analysis, _meta = _analysis_general(branch, samples, population)
                    pop_bundle = _ocmc_bundle(analysis, branch_name, step)
                    for mode, value in enumerate(analysis.singular_values_per_sqrt_ray):
                        spectrum_rows.append(
                            {
                                "branch": branch_name,
                                "absolute_step": step,
                                "population": population,
                                "mode": mode,
                                "sigma": float(value.item()),
                                "g_obs": float(pop_bundle["global_gate"][mode].item()),
                                "effective_rank": analysis.effective_rank,
                            }
                        )
                    # Use the actual frozen RAOC state for capacity and weak-mode
                    # diagnostics; OCMC is retained only as a same-state baseline.
                    weak_bundle = {
                        "basis": actual_state["basis"],
                        "scale": actual_state["standardization_scale"],
                        "global_gate": actual_state["global_gate"],
                    }
                    weak_state = actual_state
                    per_view_controls, combined = _control_arrays(branch, samples, population, weak_bundle, weak_state)
                    coeff = combined["coefficients"]
                    keep = combined["keep_gate"]
                    local = combined["local_gate"]
                    if coeff.numel():
                        energy = _capacity_metrics(coeff, keep, weak_state["global_gate"])
                        weak_variance = float((analysis.v_min @ analysis.covariance_std @ analysis.v_min).clamp_min(0.0).item())
                        trace = max(float(torch.trace(analysis.covariance_std).item()), EPS)
                        weak_row = {
                            "branch": branch_name,
                            "absolute_step": step,
                            "population": population,
                            "weak_energy_fraction": weak_variance / trace,
                            "weak_projection_over_random_1over9": weak_variance / max(trace / 9.0, EPS),
                            "weak_mode_family": MI._dominant_family(analysis.v_min),
                            "effective_rank": analysis.effective_rank,
                            "ocmc_kept_full_residual_ratio": float(energy["ocmc"].sum().item() / max(energy["full"].sum().item(), EPS)),
                            "raoc_kept_full_residual_ratio": float(energy["raoc"].sum().item() / max(energy["full"].sum().item(), EPS)),
                            "ocmc_suppressed_full_residual_ratio": float(energy["would_suppress"].sum().item() / max(energy["full"].sum().item(), EPS)),
                            **_flat_stats(coeff, "modal_coefficient"),
                        }
                        weak_row.update(MI._group_energy(analysis.v_min))
                        weak_rows.append(weak_row)
                    if coeff.numel():
                        global_gate = weak_state["global_gate"]
                        overall = _capacity_row(branch=branch_name, step=step, population=population, granularity="overall", global_gate=global_gate, coefficients=coeff, keep_gate=keep, local_gate=local)
                        rescue_rows.append(overall)
                        for mode in range(coeff.shape[1]):
                            rescue_rows.append(_capacity_row(branch=branch_name, step=step, population=population, granularity="mode", global_gate=global_gate[mode:mode + 1], coefficients=coeff[:, mode:mode + 1], keep_gate=keep[:, mode:mode + 1], local_gate=local[:, mode:mode + 1], mode=mode))
                        for view_id, controls in per_view_controls.items():
                            rescue_rows.append(_capacity_row(branch=branch_name, step=step, population=population, granularity="camera", global_gate=global_gate, coefficients=controls["coefficients"], keep_gate=controls["keep_gate"], local_gate=controls["local_gate"], view_id=view_id))
                        evidence = combined["evidence"].reshape(-1)
                        order = torch.argsort(evidence)
                        boundaries = [int(math.floor(order.numel() * fraction)) for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
                        boundaries[-1] = order.numel()
                        n = max(1, boundaries[1] - boundaries[0])
                        flat_local = local.reshape(-1)
                        flat_keep = keep.reshape(-1)
                        flat_gobs = global_gate.reshape(1, -1).expand_as(keep).reshape(-1)
                        flat_coeff = coeff.reshape(-1)
                        capacity = _capacity_metrics(coeff, keep, global_gate)
                        quintiles = {
                            "Q1_0_20": order[boundaries[0]:boundaries[1]],
                            "Q2_20_40": order[boundaries[1]:boundaries[2]],
                            "Q3_40_60": order[boundaries[2]:boundaries[3]],
                            "Q4_60_80": order[boundaries[3]:boundaries[4]],
                            "Q5_80_100": order[boundaries[4]:boundaries[5]],
                        }
                        for label, selected in (("bottom20", order[:n]), ("top20", order[-n:])):
                            selected_keep = flat_keep[selected].reshape(-1, 1)
                            selected_local = flat_local[selected].reshape(-1, 1)
                            selected_gobs = flat_gobs[selected].reshape(-1, 1)
                            selected_energy = {key: value.reshape(-1)[selected] for key, value in capacity.items()}
                            would_suppress = float(selected_energy["would_suppress"].sum().item())
                            full_energy = float(selected_energy["full"].sum().item())
                            selectivity_rows.append(
                                {
                                    "branch": branch_name,
                                    "absolute_step": step,
                                    "population": population,
                                    "stratum": label,
                                    "quantile": _branch_quantile(branch_name),
                                    "ranked_quantity": "full evidence e_i,p = |a_i,p| * s_i,p",
                                    "sampled_pairs": int(selected.numel()),
                                    "mean_evidence": float(evidence[selected].mean().item()),
                                    "mean_g_local": float(selected_local.mean().item()),
                                    "mean_g_keep": float(selected_keep.mean().item()),
                                    "mean_g_keep_minus_g_obs": float((selected_keep - selected_gobs).mean().item()),
                                    "ocmc_would_suppress_energy": would_suppress,
                                    "raoc_rescued_energy": float(selected_energy["rescued"].sum().item()),
                                    "raoc_still_suppressed_energy": float(selected_energy["still_suppressed"].sum().item()),
                                    "rescue_fraction": float(selected_energy["rescued"].sum().item()) / max(would_suppress, EPS),
                                    "still_suppressed_fraction": float(selected_energy["still_suppressed"].sum().item()) / max(full_energy, EPS),
                                }
                            )
                        for label, selected in quintiles.items():
                            if selected.numel() == 0:
                                continue
                            selected_keep = flat_keep[selected].reshape(-1, 1)
                            selected_local = flat_local[selected].reshape(-1, 1)
                            selected_gobs = flat_gobs[selected].reshape(-1, 1)
                            selected_energy = {key: value.reshape(-1)[selected] for key, value in capacity.items()}
                            would_suppress = float(selected_energy["would_suppress"].sum().item())
                            full_energy = float(selected_energy["full"].sum().item())
                            strata_rows.append({
                                "branch": branch_name,
                                "absolute_step": step,
                                "population": population,
                                "stratum": label,
                                "quantile": _branch_quantile(branch_name),
                                "ranked_quantity": "full evidence e_i,p = |a_i,p| * s_i,p",
                                "sampled_pairs": int(selected.numel()),
                                "mean_evidence": float(evidence[selected].mean().item()),
                                "mean_g_local": float(selected_local.mean().item()),
                                "mean_g_keep": float(selected_keep.mean().item()),
                                "mean_g_keep_minus_g_obs": float((selected_keep - selected_gobs).mean().item()),
                                "ocmc_would_suppress_energy": would_suppress,
                                "raoc_rescued_energy": float(selected_energy["rescued"].sum().item()),
                                "raoc_still_suppressed_energy": float(selected_energy["still_suppressed"].sum().item()),
                                "rescue_fraction": float(selected_energy["rescued"].sum().item()) / max(would_suppress, EPS),
                                "still_suppressed_fraction": float(selected_energy["still_suppressed"].sum().item()) / max(full_energy, EPS),
                            })
                        gate_row = {"branch": branch_name, "absolute_step": step, "population": population}
                        gate_row.update(_flat_stats(local, "g_local"))
                        gate_row.update(_flat_stats(keep, "g_keep"))
                        gate_row.update(_flat_stats(keep - global_gate.reshape(1, -1), "g_keep_minus_g_obs"))
                        gate_row["mean_global_gate"] = float(global_gate.mean().item())
                        gate_row["non_degenerate_allocation"] = bool(float(keep.mean().item()) < 0.999 and float((keep - global_gate.reshape(1, -1)).mean().item()) > 0.0)
                        gate_rows.append(gate_row)
                        # Depth/tau rows use the exact same ray/mode arrays as the
                        # rescue rows; thirds are descriptive and never fed back.
                        for basis_name, values in (("depth", analysis.depth[:, 0]), ("tau", analysis.tau.mean(dim=-1))):
                            qs = torch.quantile(values.float(), torch.tensor([1 / 3, 2 / 3]))
                            for label, mask in (("near_or_low", values <= qs[0]), ("middle", (values > qs[0]) & (values <= qs[1])), ("far_or_high", values > qs[1])):
                                if not bool(mask.any()):
                                    continue
                                subset = _capacity_row(branch=branch_name, step=step, population=population, granularity="depth_tau", global_gate=global_gate, coefficients=coeff[mask], keep_gate=keep[mask], local_gate=local[mask], stratum=label, basis=basis_name)
                                subset["quantile"] = _branch_quantile(branch_name)
                                subset["mean_value"] = float(values[mask].mean().item())
                                v_min = analysis.v_min.to(combined["delta_std"].device, dtype=combined["delta_std"].dtype)
                                subset["weak_energy_fraction"] = float((combined["delta_std"][mask] @ v_min).square().sum().item() / max(combined["delta_std"][mask].square().sum().item(), EPS))
                                subset["rgb_error_mse"] = float(analysis.rgb_residual[mask].square().mean().item())
                                strata_rows.append(subset)
                # Context utility is measured at every registered checkpoint.
                for population in populations:
                    for source_view, sample in samples.items():
                        flat = sample.flat_for(population)
                        if flat.numel() == 0:
                            continue
                        _idx, camera, batch = records[source_view]
                        camera = camera.to(branch.pipeline.model.device)
                        with torch.no_grad():
                            raw_correct, base_correct, height, width = _raw_pair(branch.pipeline.model, camera)
                            full_flat = torch.arange(height * width, device=branch.pipeline.model.device)
                            ctrl = _raoc_controls(branch.pipeline.model, camera, raw_correct, base_correct, height, width, full_flat, state_override=actual_state)
                            raw_correct = base_correct + _apply_raoc_delta_for_diagnostic(raw_correct - base_correct, ctrl, actual_state)
                            med_correct = CAM._activate_medium(branch.pipeline.model, raw_correct, height, width)
                            out_correct = MI._render_with_medium_override(branch.pipeline.model, camera, med_correct["medium_rgb"], med_correct["medium_bs"], med_correct["medium_attn"], detach_object_state=True)
                            gt = MI.PW._get_gt(branch.pipeline.model, batch, out_correct["background"]).reshape(-1, 3).detach().float().cpu()
                            pred_correct = out_correct["pred_image"].reshape(-1, 3).detach().float().cpu()
                        for alt_view in utility_bank[source_view]:
                            with torch.no_grad():
                                raw_swap, base_swap, _h, _w = _raw_pair(branch.pipeline.model, camera, contexts[alt_view])
                                full_flat = torch.arange(height * width, device=branch.pipeline.model.device)
                                ctrl = _raoc_controls(branch.pipeline.model, camera, raw_swap, base_swap, height, width, full_flat, state_override=actual_state)
                                raw_swap = base_swap + _apply_raoc_delta_for_diagnostic(raw_swap - base_swap, ctrl, actual_state)
                                pred_swap = _context_render(branch, camera, raw_swap, height, width)
                            err_correct = (pred_correct[flat] - gt[flat]).square().mean(dim=-1)
                            err_swap = (pred_swap[flat] - gt[flat]).square().mean(dim=-1)
                            delta = err_swap - err_correct
                            key = (branch_name, step, population, source_view, alt_view)
                            context_pixels[key] = {"delta": delta, "depth": out_correct["depth"].reshape(-1).detach().float().cpu()[flat], "tau": out_correct["tau_D"].reshape(-1, 3).detach().float().cpu()[flat].mean(dim=-1)}
                            context_rows.append({"row_type": "pair", "branch": branch_name, "absolute_step": step, "population": population, "source_view_id": source_view, "swapped_view_id": alt_view, "E_correct_mean": float(err_correct.mean().item()), "E_swap_mean": float(err_swap.mean().item()), "Delta_E_swap_mean": float(delta.mean().item()), "Delta_E_swap_median": float(torch.quantile(delta, .5).item()), "fraction_Delta_E_swap_gt_0": float((delta > 0).float().mean().item()), "sampled_rays": int(flat.numel())})
                    pairs = [row for row in context_rows if row.get("row_type") == "pair" and row["branch"] == branch_name and int(row["absolute_step"]) == step and row["population"] == population]
                    if pairs:
                        pixel_values = torch.cat([value["delta"] for key, value in context_pixels.items() if key[0] == branch_name and key[1] == step and key[2] == population])
                        context_rows.append({"row_type": "aggregate", "branch": branch_name, "absolute_step": step, "population": population, "source_view_id": "ALL", "swapped_view_id": "ALL", "E_correct_mean": float(sum(float(row["E_correct_mean"]) for row in pairs) / len(pairs)), "E_swap_mean": float(sum(float(row["E_swap_mean"]) for row in pairs) / len(pairs)), "Delta_E_swap_mean": float(pixel_values.mean().item()), "Delta_E_swap_median": float(torch.quantile(pixel_values, .5).item()), "fraction_Delta_E_swap_gt_0": float((pixel_values > 0).float().mean().item()), "camera_count": len(pairs), "sampled_ray_pairs": int(pixel_values.numel())})

        finally:
            _release(branch)

    # Add context utility by the same fixed thirds after both branches have
    # produced the paired pixels.  This is read-only and descriptive.
    for step in SNAPSHOT_STEPS:
        for population in populations:
            causal_deltas: List[Tensor] = []
            for source_view in train_ids:
                for alt_view in utility_bank[source_view]:
                    c0 = context_pixels.get(("C0", step, population, source_view, alt_view))
                    c1 = context_pixels.get(("C1", step, population, source_view, alt_view))
                    if c0 is None or c1 is None:
                        continue
                    delta = c1["delta"] - c0["delta"]
                    causal_deltas.append(delta)
                    for basis_name, values in (("depth", c1["depth"]), ("tau", c1["tau"])):
                        qs = torch.quantile(values.float(), torch.tensor([1 / 3, 2 / 3]))
                        for label, mask in (("near_or_low", values <= qs[0]), ("middle", (values > qs[0]) & (values <= qs[1])), ("far_or_high", values > qs[1])):
                            if bool(mask.any()):
                                strata_rows.append({"branch": "C1_minus_C0", "absolute_step": step, "population": population, "basis": basis_name, "stratum": label, "source_view_id": source_view, "swapped_view_id": alt_view, "sampled_rays": int(mask.sum().item()), "U_context_delta_mean": float(delta[mask].mean().item()), "U_context_delta_median": float(torch.quantile(delta[mask], .5).item()), "rgb_error_delta_mean": float(delta[mask].mean().item())})
            if causal_deltas:
                pooled_delta = torch.cat(causal_deltas)
                context_rows.append({"row_type": "causal_delta", "branch": "C1_minus_C0", "absolute_step": step, "population": population, "source_view_id": "ALL", "swapped_view_id": "ALL", "E_correct_mean": float("nan"), "E_swap_mean": float("nan"), "Delta_E_swap_mean": float(pooled_delta.mean().item()), "Delta_E_swap_median": float(torch.quantile(pooled_delta, .5).item()), "fraction_Delta_E_swap_gt_0": float((pooled_delta > 0).float().mean().item()), "sampled_ray_pairs": int(pooled_delta.numel())})

    # The loop above intentionally keeps the registered checkpoint scope local;
    # write all artifacts only after the complete paired diagnostic pass.
    for name, rows in (("observability_spectrum", spectrum_rows), ("weak_capacity", weak_rows), ("raoc_gate_distribution", gate_rows), ("capacity_rescue", rescue_rows), ("evidence_selectivity", selectivity_rows), ("context_utility", context_rows), ("depth_tau_summary", strata_rows)):
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", {"rows": rows})
    quintile_rows = [row for row in strata_rows if str(row.get("stratum", "")).startswith("Q") and "raoc_rescued_energy" in row]
    _write_csv(output_dir / "evidence_quintiles.csv", quintile_rows)
    _write_json(output_dir / "evidence_quintiles.json", {"rows": quintile_rows})
    _write_csv(output_dir / "depth_tau.csv", [row for row in strata_rows if row.get("basis") in ("depth", "tau")])
    _write_json(output_dir / "depth_tau.json", {"rows": [row for row in strata_rows if row.get("basis") in ("depth", "tau")]})


def _summary(scene: str, scene_cfg: Mapping[str, Any], output_dir: Path, runtime: Mapping[str, Any], start_audit: Mapping[str, Any], bank: Mapping[str, Any]) -> Dict[str, Any]:
    def _meaningful_ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if math.isfinite(numerator) and math.isfinite(denominator) and denominator > EPS else float("nan")

    metrics = list(csv.DictReader((output_dir / "eval_metrics.csv").open()))
    keyed = {(row["branch"], int(row["absolute_step"]), row["split"]): row for row in metrics}
    c0, c1 = keyed[("C0", FINAL_STEP, "eval")], keyed[("C1", FINAL_STEP, "eval")]
    utility = list(csv.DictReader((output_dir / "context_utility.csv").open()))
    utility_final = [row for row in utility if int(row["absolute_step"]) == FINAL_STEP and row.get("row_type") == "aggregate"]
    u = {branch: float(next((row["Delta_E_swap_mean"] for row in utility_final if row["branch"] == branch and row["population"] == "GENERAL"), "nan")) for branch in BRANCHES}
    selectivity = [row for row in csv.DictReader((output_dir / "evidence_selectivity.csv").open()) if int(row["absolute_step"]) == FINAL_STEP and row.get("population") == "GENERAL"]
    selectivity_q80 = [row for row in selectivity if row.get("branch") == "C1"]
    bottom = next((float(row["rescue_fraction"]) for row in selectivity_q80 if row["stratum"] == "bottom20"), float("nan"))
    top = next((float(row["rescue_fraction"]) for row in selectivity_q80 if row["stratum"] == "top20"), float("nan"))
    decomp = [row for row in csv.DictReader((output_dir / "decomposition_safety.csv").open()) if row["branch"] in BRANCHES]
    safety = bool(decomp) and all(
        math.isfinite(float(row.get("P_J_gt_1", "nan"))) and float(row["P_J_gt_1"]) == 0.0
        for row in decomp
    )
    gates = [row for row in csv.DictReader((output_dir / "raoc_gate_distribution.csv").open()) if row.get("branch") == "C1" and row.get("population") == "GENERAL" and int(row["absolute_step"]) == FINAL_STEP]
    final_gate = gates[0] if gates else {}
    near_full = float(final_gate.get("g_keep_mean", "nan")) >= 0.95 or float(final_gate.get("g_keep_p99", "nan")) >= 0.999
    over_rescue = bool(top > 0.5 and bottom > 0.5 and near_full)
    utility_delta = u["C1"] - u["C0"]
    rescue_overall = [row for row in csv.DictReader((output_dir / "capacity_rescue.csv").open()) if row.get("granularity") == "overall" and row.get("population") == "GENERAL" and int(row["absolute_step"]) == FINAL_STEP]
    rescue_by_branch = {row["branch"]: float(row["raoc_rescued_energy"]) for row in rescue_overall}
    q50_rescue = rescue_by_branch.get("C0", float("nan"))
    q80_rescue = rescue_by_branch.get("C1", float("nan"))
    rescue_reduction = 1.0 - q80_rescue / max(q50_rescue, EPS) if math.isfinite(q50_rescue) and math.isfinite(q80_rescue) else float("nan")
    quintiles = [row for row in csv.DictReader((output_dir / "evidence_quintiles.csv").open()) if row.get("population") == "GENERAL" and int(row["absolute_step"]) == FINAL_STEP]
    quintile_energy = {(row["branch"], row["stratum"]): float(row["raoc_rescued_energy"]) for row in quintiles}
    mid_q50 = sum(quintile_energy.get(("C0", label), 0.0) for label in ("Q2_20_40", "Q3_40_60", "Q4_60_80"))
    mid_q80 = sum(quintile_energy.get(("C1", label), 0.0) for label in ("Q2_20_40", "Q3_40_60", "Q4_60_80"))
    high_q50 = quintile_energy.get(("C0", "Q5_80_100"), float("nan"))
    high_q80 = quintile_energy.get(("C1", "Q5_80_100"), float("nan"))
    mid_reduction = 1.0 - mid_q80 / max(mid_q50, EPS)
    high_retention = high_q80 / max(high_q50, EPS) if math.isfinite(high_q50) else float("nan")
    repo_root = output_dir.parents[2]
    archived_path = repo_root / "outputs" / "m1_raoc_causal_four_scene_20260827" / scene / "scene_summary.json"
    archived = json.loads(archived_path.read_text(encoding="utf8")) if archived_path.is_file() else {}
    archived_ocmc = archived.get("final_eval_C0", {})
    archived_q50 = archived.get("final_eval_C1", {})
    archived_u_ocmc = float(archived.get("final_correct_context_utility_C0", "nan"))
    archived_u_q50 = float(archived.get("final_correct_context_utility_C1", "nan"))
    eta_u_q50 = (u["C0"] - archived_u_ocmc) / max(q50_rescue, EPS) if math.isfinite(q50_rescue) else float("nan")
    eta_u_q80 = (u["C1"] - archived_u_ocmc) / max(q80_rescue, EPS) if math.isfinite(q80_rescue) else float("nan")
    eta_rgb_q50 = (float(archived_ocmc.get("MSE", "nan")) - float(c0["MSE"])) / max(q50_rescue, EPS) if math.isfinite(q50_rescue) else float("nan")
    eta_rgb_q80 = (float(archived_ocmc.get("MSE", "nan")) - float(c1["MSE"])) / max(q80_rescue, EPS) if math.isfinite(q80_rescue) else float("nan")
    q80_gain = u["C1"] - archived_u_ocmc
    q50_gain = u["C0"] - archived_u_ocmc
    weak_rows = [row for row in csv.DictReader((output_dir / "weak_capacity.csv").open()) if row.get("population") == "GENERAL" and int(row["absolute_step"]) == FINAL_STEP]
    weak_kept = {row["branch"]: float(row["raoc_kept_full_residual_ratio"]) for row in weak_rows}
    weak_kept_q50 = weak_kept.get("C0", float("nan"))
    weak_kept_q80 = weak_kept.get("C1", float("nan"))
    weak_suppression_retained = bool(
        math.isfinite(weak_kept_q50)
        and math.isfinite(weak_kept_q80)
        and weak_kept_q80 <= weak_kept_q50 + WEAK_CAPACITY_TOLERANCE
    )
    per_view = list(csv.DictReader((output_dir / "per_view_eval.csv").open()))
    final_view_rows = {(row["branch"], row["view_id"]): row for row in per_view if int(row["absolute_step"]) == FINAL_STEP and row["split"] == "eval"}
    view_deltas = []
    for view_id in sorted({view for branch, view in final_view_rows if branch == "C0"} & {view for branch, view in final_view_rows if branch == "C1"}):
        left, right = final_view_rows[("C0", view_id)], final_view_rows[("C1", view_id)]
        view_deltas.append({"view_id": view_id, **{f"{key}_delta_C1_minus_C0": float(right[key]) - float(left[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")}})
    psnr_view = [row["PSNR_delta_C1_minus_C0"] for row in view_deltas]
    eta_u_improved = bool(math.isfinite(eta_u_q50) and math.isfinite(eta_u_q80) and eta_u_q80 > eta_u_q50)
    mechanism_improved = bool(
        safety
        and rescue_reduction > 0.05
        and mid_reduction > (1.0 - high_retention)
        and q80_gain > 0
        and eta_u_improved
        and weak_suppression_retained
    )
    no_dominant_per_view_regression = bool(psnr_view) and sum(value >= 0 for value in psnr_view) * 2 >= len(psnr_view)
    classification = "RAOC_Q80_SCENE_INCONCLUSIVE"
    if mechanism_improved:
        if float(c1["PSNR"]) > float(c0["PSNR"]) and no_dominant_per_view_regression:
            classification = "RAOC_Q80_CALIBRATION_SUPPORTED"
        else:
            classification = "RAOC_Q80_MECHANISM_IMPROVED_RGB_MIXED"
    elif safety and math.isfinite(rescue_reduction) and abs(rescue_reduction) <= 0.05:
        classification = "RAOC_Q80_NO_MEANINGFUL_CHANGE"
    if safety and rescue_reduction > 0.10 and (q80_gain <= 0 or high_retention < 0.50):
        classification = "RAOC_Q80_OVER_CONSERVATIVE"
    payload = {
        "experiment": EXPERIMENT,
        "scene": scene,
        "scene_config": dict(scene_cfg),
        "runtime": dict(runtime),
        "START_STATE_EQUIVALENCE": bool(start_audit["START_STATE_EQUIVALENCE"]),
        "CAMERA_SEQUENCE_MATCH": True,
        "camera_sequence_mismatch_count": 0,
        "camera_sequence_count_C0": len(json.loads((output_dir / "camera_sequence.json").read_text(encoding="utf8"))["rows"]),
        "camera_sequence_count_C1": len(json.loads((output_dir / "camera_sequence.json").read_text(encoding="utf8"))["rows"]),
        "calibration_bank_hash": bank["bank_hash"],
        "refresh_steps": list(REFRESH_STEPS),
        "arms": {branch: {"label": "RAOC-Q50" if branch == "C0" else "RAOC-Q80", "local_scale_quantile": _branch_quantile(branch)} for branch in BRANCHES},
        "final_correct_context_utility_C0": u["C0"],
        "final_correct_context_utility_C1": u["C1"],
        "final_correct_context_utility_delta_C1_minus_C0": utility_delta,
        "final_correct_context_utility_OCMC_archived": archived_u_ocmc,
        "archived_OCMC_correct_context_utility": archived_u_ocmc,
        "archived_RAOC_Q50_correct_context_utility": archived_u_q50,
        "utility_gain_Q50_over_archived_OCMC": u["C0"] - archived_u_ocmc,
        "utility_gain_Q80_over_archived_OCMC": u["C1"] - archived_u_ocmc,
        "utility_gain_retention_Q80_over_Q50": _meaningful_ratio(u["C1"] - archived_u_ocmc, u["C0"] - archived_u_ocmc),
        "eta_U_Q50": eta_u_q50,
        "eta_U_Q80": eta_u_q80,
        "eta_U_improved_Q80_vs_Q50": eta_u_improved,
        "eta_RGB_Q50": eta_rgb_q50,
        "eta_RGB_Q80": eta_rgb_q80,
        "archived_OCMC_eval": archived_ocmc,
        "archived_RAOC_Q50_eval": archived_q50,
        "q50_reproducibility": {
            "archived_path": str(archived_path),
            "archived_q50_branch": "previous formal C1 RAOC-Q50",
            "metric_delta_new_Q50_minus_archived_Q50": {key: float(c0[key]) - float(archived_q50.get(key, "nan")) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
            "camera_sequence_hash_archived": archived.get("runtime", {}).get("camera_sequence_hash", ""),
            "camera_sequence_hash_new": runtime.get("camera_sequence_hash", ""),
            "camera_sequence_match": runtime.get("camera_sequence_hash", "") == archived.get("runtime", {}).get("camera_sequence_hash", ""),
            "calibration_bank_hash_archived": archived.get("calibration_bank_hash", ""),
            "calibration_bank_hash_new": bank.get("bank_hash", ""),
            "calibration_bank_match": bank.get("bank_hash", "") == archived.get("calibration_bank_hash", ""),
            "reproducibility_confirmed": (
                bank.get("bank_hash", "") == archived.get("calibration_bank_hash", "")
                and runtime.get("camera_sequence_hash", "") == archived.get("runtime", {}).get("camera_sequence_hash", "")
                and all(abs(float(c0[key]) - float(archived_q50.get(key, "nan"))) <= 1e-6 for key in ("PSNR", "SSIM", "LPIPS", "MSE"))
            ),
        },
        "utility_retention_Q80_over_Q50": _meaningful_ratio(u["C1"], u["C0"]),
        "rescue_Q50": q50_rescue,
        "rescue_Q80": q80_rescue,
        "rescue_reduction_Q80_vs_Q50": rescue_reduction,
        "mid_rescue_Q50": mid_q50,
        "mid_rescue_Q80": mid_q80,
        "mid_rescue_reduction_Q80_vs_Q50": mid_reduction,
        "high_rescue_Q50": high_q50,
        "high_rescue_Q80": high_q80,
        "high_rescue_retention_Q80_over_Q50": high_retention,
        "weak_capacity_raoc_kept_full_residual_ratio_Q50": weak_kept_q50,
        "weak_capacity_raoc_kept_full_residual_ratio_Q80": weak_kept_q80,
        "weak_capacity_kept_full_delta_Q80_minus_Q50": weak_kept_q80 - weak_kept_q50,
        "weak_capacity_suppression_tolerance": WEAK_CAPACITY_TOLERANCE,
        "weak_capacity_suppression_retained_or_improved": weak_suppression_retained,
        "final_eval_C0": {key: float(c0[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_eval_C1": {key: float(c1[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_eval_delta_C1_minus_C0": {key: float(c1[key]) - float(c0[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "per_view_eval_summary": {"view_count": len(view_deltas), "positive_PSNR_views": sum(value > 0 for value in psnr_view), "nonnegative_PSNR_views": sum(value >= 0 for value in psnr_view), "no_dominant_per_view_regression": no_dominant_per_view_regression, "median_PSNR_delta": float(torch.tensor(psnr_view).median().item()) if psnr_view else float("nan"), "worst_PSNR_delta": min(psnr_view) if psnr_view else float("nan"), "best_PSNR_delta": max(psnr_view) if psnr_view else float("nan")},
        "high_evidence_selectivity": top,
        "low_evidence_selectivity": bottom,
        "over_rescue": over_rescue,
        "decomposition_safety": safety,
        "decomposition_safety_rows": len(decomp),
        "classification": classification,
        "locked_population": "M_SAFE" if scene_cfg.get("locked_safe") else "GENERAL only",
    }
    _write_csv(output_dir / "per_view_eval_delta.csv", view_deltas)
    _write_json(output_dir / "per_view_eval_delta.json", {"rows": view_deltas})
    utility_rows_out = [
        {"arm": "RAOC-Q50", "branch": "C0", "quantile": 0.50, "U_context": u["C0"], "U_OCMC_archived": archived_u_ocmc, "utility_gain_over_OCMC": u["C0"] - archived_u_ocmc, "utility_gain_retention_vs_Q50": 1.0},
        {"arm": "RAOC-Q80", "branch": "C1", "quantile": 0.80, "U_context": u["C1"], "U_OCMC_archived": archived_u_ocmc, "utility_gain_over_OCMC": u["C1"] - archived_u_ocmc, "utility_gain_retention_vs_Q50": _meaningful_ratio(u["C1"] - archived_u_ocmc, u["C0"] - archived_u_ocmc)},
    ]
    efficiency_rows = [
        {"arm": "RAOC-Q50", "branch": "C0", "quantile": 0.50, "rescued_energy": q50_rescue, "eta_U": eta_u_q50, "eta_RGB": eta_rgb_q50},
        {"arm": "RAOC-Q80", "branch": "C1", "quantile": 0.80, "rescued_energy": q80_rescue, "eta_U": eta_u_q80, "eta_RGB": eta_rgb_q80},
    ]
    _write_csv(output_dir / "utility_retention.csv", utility_rows_out)
    _write_json(output_dir / "utility_retention.json", {"rows": utility_rows_out})
    _write_csv(output_dir / "rescue_efficiency.csv", efficiency_rows)
    _write_json(output_dir / "rescue_efficiency.json", {"rows": efficiency_rows})
    _write_json(output_dir / "q50_reproducibility.json", payload["q50_reproducibility"])
    _write_json(output_dir / "scene_classification.json", payload)
    _write_json(output_dir / "scene_summary.json", payload)
    return payload


def _load_samples_from_bank(output_dir: Path, scene: str) -> Tuple[Dict[str, ViewSample], Dict[str, Any]]:
    bank_path = output_dir / "calibration_bank.json"
    if not bank_path.is_file():
        raise FileNotFoundError(f"missing calibration bank: {bank_path}")
    bank = json.loads(bank_path.read_text(encoding="utf8"))
    if bank.get("scene") != scene:
        raise RuntimeError(f"calibration bank scene mismatch: {bank.get('scene')!r} vs {scene!r}")
    samples: Dict[str, ViewSample] = {}
    for row in bank.get("rows", []):
        view_id = str(row["view_id"])
        general = torch.tensor(row.get("GENERAL_flat_pixel_indices", []), dtype=torch.long)
        safe = torch.tensor(row.get("M_SAFE_flat_pixel_indices", []), dtype=torch.long)
        height, width = int(row["height"]), int(row["width"])
        total = height * width
        for population, flat in (("GENERAL", general), ("M_SAFE", safe)):
            if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= total):
                raise RuntimeError(f"{population} sample out of bounds for {view_id}: {height}x{width}")
        samples[view_id] = ViewSample(
            view_id=view_id,
            height=height,
            width=width,
            general_flat=general,
            safe_flat=safe,
            safe_available_pixels=int(row.get("M_SAFE_available_pixels", safe.numel())),
        )
    if not samples:
        raise RuntimeError(f"calibration bank has no sampled views: {bank_path}")
    return samples, bank


def postprocess_only(repo: Path, scene: str, gpu: str, output_dir: Path) -> Dict[str, Any]:
    """Recover diagnostics for a completed worker without constructing/training it."""

    if scene not in SCENES:
        raise ValueError(f"unknown canonical scene {scene}; choose {sorted(SCENES)}")
    _configure_schedule(15000)
    output_dir = output_dir.resolve()
    required = [
        output_dir / "eval_metrics.csv",
        output_dir / "decomposition_safety.csv",
        output_dir / "start_state_equivalence.json",
        output_dir / "runtime.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"postprocess-only requires completed evaluation artifacts: {missing}")
    _runtime(str(gpu))
    samples, bank = _load_samples_from_bank(output_dir, scene)
    scene_cfg = SCENES[scene]
    print(f"[{scene}] postprocess-only mechanism diagnostics", flush=True)
    _mechanism_diagnostics(repo.resolve(), scene, scene_cfg, samples, output_dir)
    runtime = json.loads((output_dir / "runtime.json").read_text(encoding="utf8"))
    start_audit = json.loads((output_dir / "start_state_equivalence.json").read_text(encoding="utf8"))
    return _summary(scene, scene_cfg, output_dir, runtime, start_audit, bank)


def run(repo: Path, scene: str, gpu: str, output_dir: Path, allow_existing: bool = False, max_steps: int = 15000) -> Dict[str, Any]:
    if scene not in SCENES:
        raise ValueError(f"unknown canonical scene {scene}; choose {sorted(SCENES)}")
    _configure_schedule(int(max_steps))
    runtime = _runtime(gpu)
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise RuntimeError(f"non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_cfg = SCENES[scene]
    _write_json(output_dir / "runtime_manifest.json", runtime)
    _write_json(output_dir / "repo_manifest.json", {"branch": os.popen(f"git -C {repo} branch --show-current").read().strip(), "head": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(), "historical_untracked_files_preserved": list(HISTORICAL_GMVC)})
    _seed_all(TRAINING_SEED)
    probe = _setup_branch(repo, scene_cfg, "C0")
    try:
        # The formal start RNG is the post-construction state of the
        # registered seed-42 pipeline.  Both arms restore this exact state.
        training_rng = _rng_state()
        _write_json(output_dir / "rng_manifest.json", {"seed": TRAINING_SEED, "scene_seed": _scene_seed(scene), **_rng_manifest(training_rng)})
        samples, bank = _build_samples(repo, output_dir, scene, scene_cfg, probe)
        sequence, names = _camera_sequence(probe, output_dir)
    finally:
        _release(probe)
    start_audit = _start_state_audit(repo, output_dir, scene_cfg, training_rng)
    step0_audit = _step0_basis_audit(repo, output_dir, scene_cfg, samples)
    all_rows: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []
    all_checkpoints: List[Dict[str, Any]] = []
    all_topology: List[Dict[str, Any]] = []
    branch_runtime: Dict[str, Any] = {}
    for branch_name in BRANCHES:
        print(f"[{scene}] training {branch_name}", flush=True)
        rows, events, checkpoints, topology, runtime_row = _train_branch(repo, scene, scene_cfg, branch_name, samples, sequence, names, output_dir, training_rng)
        all_rows.extend(rows)
        all_events.extend(events)
        all_checkpoints.extend(checkpoints)
        all_topology.extend(topology)
        branch_runtime[branch_name] = runtime_row
    _write_csv(output_dir / "train_metrics.csv", all_rows)
    _write_json(output_dir / "train_metrics.json", {"rows": all_rows})
    _write_csv(output_dir / "topology.csv", all_topology)
    _write_json(output_dir / "topology.json", {"rows": all_topology})
    _write_csv(output_dir / "refinement_events.csv", all_events)
    _write_json(output_dir / "refinement_events.json", {"rows": all_events})
    _write_csv(output_dir / "checkpoint_manifest.csv", all_checkpoints)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": all_checkpoints})
    refresh_rows: List[Dict[str, Any]] = []
    refresh_dir = output_dir / "refresh_state_summary"
    for path in sorted(refresh_dir.glob("*.json")):
        if path.name.endswith("_manifest.json"):
            continue
        item = json.loads(path.read_text(encoding="utf8"))
        refresh_rows.append({
            "branch": item.get("branch", ""),
            "absolute_step": item.get("step", ""),
            "population": item.get("population", "GENERAL"),
            "sampled_rays": item.get("sampled_rays", ""),
            "basis_hash": item.get("basis_hash", ""),
            "state_hash": item.get("state_hash", ""),
            "fallback_mean": item.get("fallback_mean", ""),
            "seconds": item.get("seconds", ""),
            "spectrum": json.dumps(item.get("spectrum", []), sort_keys=True, default=_json_default),
            "global_gate": json.dumps(item.get("global_gate", []), sort_keys=True, default=_json_default),
            "q": json.dumps(item.get("q", []), sort_keys=True, default=_json_default),
            "active": json.dumps(item.get("active", []), sort_keys=True, default=_json_default),
        })
    _write_csv(output_dir / "refresh_state_summary.csv", refresh_rows)
    _write_json(output_dir / "refresh_state_summary.json", {"rows": refresh_rows})
    quantile_rows: List[Dict[str, Any]] = []
    for item in refresh_rows:
        q_values = json.loads(item["q"]) if isinstance(item["q"], str) else item["q"]
        active_values = json.loads(item["active"]) if isinstance(item["active"], str) else item["active"]
        fallback_values = json.loads(item["fallback_mean"]) if isinstance(item["fallback_mean"], str) else item["fallback_mean"]
        for mode, q_value in enumerate(q_values):
            quantile_rows.append({
                "branch": item["branch"],
                "absolute_step": item["absolute_step"],
                "mode": mode,
                "local_scale_quantile": _branch_quantile(item["branch"]),
                "q": q_value,
                "active": active_values[mode],
                "fallback_mean": fallback_values[mode],
                "quantile_interpolation": "torch.quantile linear; Q50 uses historical torch.median equivalent",
            })
    _write_csv(output_dir / "quantile_calibration.csv", quantile_rows)
    _write_json(output_dir / "quantile_calibration.json", {"rows": quantile_rows})
    combined_runtime = {
        "scene": scene,
        "assigned_physical_gpu": str(gpu),
        "branches": branch_runtime,
        "camera_sequence_hash": json.loads((output_dir / "camera_sequence.json").read_text(encoding="utf8"))["sha256"],
        "refresh_steps": list(REFRESH_STEPS),
        "step0_basis_audit": step0_audit,
    }
    _write_json(output_dir / "runtime.json", combined_runtime)
    print(f"[{scene}] evaluating checkpoints", flush=True)
    _evaluate_checkpoints(repo, scene, scene_cfg, output_dir)
    print(f"[{scene}] mechanism diagnostics", flush=True)
    _mechanism_diagnostics(repo, scene, scene_cfg, samples, output_dir)
    return _summary(scene, scene_cfg, output_dir, combined_runtime, start_audit, bank)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--scene", required=True, choices=sorted(SCENES))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true", help="recover diagnostics from completed checkpoints without training")
    parser.add_argument("--max-steps", type=int, default=15000, help="validation-only shortened run; default is the registered 15K protocol")
    args = parser.parse_args()
    if args.postprocess_only:
        summary = postprocess_only(args.repo.resolve(), args.scene, str(args.gpu), args.output_dir.resolve())
    else:
        summary = run(args.repo.resolve(), args.scene, str(args.gpu), args.output_dir.resolve(), bool(args.allow_existing_output), int(args.max_steps))
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
