#!/usr/bin/env python3
"""Formal one-scene MDRR/CICA continuation worker.

The worker starts one arm from the registered OCMC checkpoint at step 3000.
MDRR and CICA are deliberately implemented here, outside the production
renderer and loss.  All controller quantities are detached.  The renderer's
forward RGB is therefore unchanged; only the gradients handed to the
existing optimizers are changed for the registered arm.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.train import _set_random_seed

from scripts.diagnostics import audit_full_medium_degradation_responsibility as FULL
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_identifiability_module_causal_scene as AUX
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL
from water_splatting.brash_reliability import build_reliability_prior
from water_splatting.bvam import bvam_gate_statistics
from water_splatting.taba import taba_statistics
from water_splatting.grac import GRAC_CONFIG, config_values as grac_config_values, grac_statistics
from water_splatting.gcrc import GCRC_CONFIG, config_values as gcrc_config_values, gcrc_statistics
from water_splatting.crcc import CRCC_CONFIG, config_values as crcc_config_values, crcc_statistics


def _route_renderer_backend(mode: str) -> Dict[str, Any]:
    """Change only rasterizer backward; all arms retain production forward."""
    if mode not in ("legacy", "fixed", "corrected_v1"):
        raise ValueError(f"unsupported renderer backward mode: {mode}")
    import water_splatting.cuda._backend as backend
    from water_splatting import csrc as production
    if mode == "corrected_v1":
        from water_splatting.cuda.gradient_backend import build_corrected_backend
        backend._C = build_corrected_backend(production)
        return {"mode": mode, "extension": backend._C.corrected.__name__,
                "build": str(Path(backend._C.corrected.__file__).parent),
                "forward_backend": production.__name__,
                "changed_functions": ["rasterize_backward"], "gradient_abi_version": 1}
    if mode == "legacy":
        if os.environ.get("WATER_SPLATTING_GRADIENT_BACKEND") == "corrected_v1":
            raise ValueError("Explicit corrected_v1 environment conflicts with legacy renderer mode; select --renderer-backward-mode corrected_v1")
        backend._C = production
        return {"mode": mode, "extension": production.__name__, "build": None,
                "forward_backend": production.__name__, "changed_functions": []}
    import glob
    from torch.utils.cpp_extension import _get_build_directory, load
    srcdir = REPO_ROOT / "water_splatting" / "cuda" / "csrc"
    name = "water_splatting_cuda_fsb_fix_20260915"
    build = Path(_get_build_directory(name, verbose=False))
    sources = sorted(glob.glob(str(srcdir / "*.cu")) + glob.glob(str(srcdir / "*.cpp")))
    ext = load(
        name=name,
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr"],
        extra_include_paths=[str(srcdir / "third_party" / "glm")],
        verbose=False,
    )
    class BackwardOnlyBackend:
        def __getattr__(self, function: str) -> Any:
            target = ext if function in ("rasterize_backward", "nd_rasterize_backward") else production
            return getattr(target, function)

    backend._C = BackwardOnlyBackend()
    return {"mode": mode, "extension": name, "build": str(build),
            "forward_backend": production.__name__,
            "changed_functions": ["rasterize_backward", "nd_rasterize_backward"]}


EXPERIMENT = "DIRECT_TRAINING_MDRR_CICA_AND_COMBINED_FOUR_SCENE"
ARMS = ("A1", "A2", "A3", "B", "N")
START_STEP = 3000
FINAL_STEP = 14999
SNAPSHOT_STEPS = (5000, 8000, 10000, 13000, 14999)
ACTIVE_SNAPSHOT_STEPS = SNAPSHOT_STEPS
TRAINING_SEED = 42
ACTIVE_TRAINING_SEED = TRAINING_SEED
ACTIVE_COMPACT_OUTPUT = False
ACTIVE_CHECKPOINT_DIRECTORY: Optional[Path] = None
ACTIVE_SAVE_COMPACT_RENDERS = False
MDRR_START_STEP = 5000
CICA_START_STEP = 10000
CICA_REFRESH_STEPS = (10000, 12000, 14000)
CICA_BANK_MAX = 6
MDRR_MIN_SHARED_VISIBLE = 32
RESPONSIBILITY_FLOOR = 1e-6
CICA_INFORMATION_FLOOR = 1e-12
CICA_HUBER_DELTA = 0.05
CICA_GRADIENT_FRACTION = 0.10
CICA_DIRECTION_COLLAPSE_FRACTION = 0.95
CICA_DIRECTION_EPS = 1e-6
CICA_MIN_SUPPORT = 3
CICA_RAMP_STEPS = 0
RESPONSIBILITY_REFERENCE_STEP = 10000
RESPONSIBILITY_EVENT_START = 10001
RESPONSIBILITY_EVENT_END = 14000
RESPONSIBILITY_BLOCK = 200
RESPONSIBILITY_M1_ONLY = 32
RESPONSIBILITY_EVENTS = 20
RESPONSIBILITY_RHO_MAX = 0.75
RESPONSIBILITY_MAX_CANDIDATES = 4096
RESPONSIBILITY_ENABLED = False
RESPONSIBILITY_SMOKE = False
AUXILIARY_APPEARANCE_ENABLED = True
AUXILIARY_APPEARANCE_STRENGTH = 1.0
OCMC_STRENGTH = 1.0
OCMC_STRENGTH_START = 1.0
OCMC_STRENGTH_RAMP_STEPS = 0
OCMC_ENABLED = True
CICA_ENABLED = True
CICA_DERIVATIVE_MODE = "legacy"
RENDERER_BACKWARD_MODE = "legacy"
# Optional overrides used by strict paired diagnostics.  The historical A1/A2/A3
# arms keep their original semantics when these are None.
ACTIVE_MDRR_OVERRIDE: Optional[bool] = None
ACTIVE_CICA_OVERRIDE: Optional[bool] = None
# Optional training-time hook used by independently registered experiments.
# The hook is called after the native/MDRR/CICA gradient has been assembled and
# may add a separately audited, parameter-group-limited gradient.  Keeping this
# disabled preserves the historical route bit-for-bit for all existing workers.
EXTRA_GRADIENT_HOOK = None
EPS = 1e-12
SH_C0 = 0.28209479177387814
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
SOURCE_CONFIG = REPO_ROOT / "outputs" / "m1_ocmc_causal_iui3_20260825" / "source_bnd_config.yml"
ACTIVE_SOURCE_ROOT = SOURCE_ROOT
ACTIVE_SOURCE_CONFIG = SOURCE_CONFIG
RAW_DATA_ONLY = False
ACTIVE_INITIALIZATION_DIR: Optional[Path] = None
ACTIVE_PARAMETERIZATION = "bounded_sh3"
HEADROOM_ALPHA = 1.0
DEGREEWISE_HEADROOM_ALPHAS = (1.0, 1.0, 1.0)
HEADROOM_SCHEDULE = "constant"
HEADROOM_START_STEP = 3000
HEADROOM_END_STEP = 8000
BRASH_ENABLED = False
BRASH_RESIDUAL_SCALE = 1.0
BRASH_GATE_INIT_LOGIT = 5.0
BRASH_GATE_MODE = "learned"
BRASH_GATE_PRIOR_ENABLED = False
BRASH_GATE_PRIOR_WEIGHT = 0.0
BRASH_MIN_SUPPORT = 3
BRASH_TRANSMISSION_FLOOR = 0.0
BRASH_DEPTH_TEMPERATURE = 1.0
BRASH_VIEW_SPREAD_SCALE = 1.0
BRASH_GATE_START_STEP = 10000
BRASH_GATE_RAMP_STEPS = 5000
BVAM_ENABLED = False
BVAM_NUM_EXPERTS = 2
BVAM_DELTA_SCALE = 1.0
BVAM_GATE_DEGREE = 1
BVAM_GATE_TEMPERATURE = 1.0
BVAM_MIN_SUPPORT = 3
BVAM_MIN_VIEW_SPREAD = 0.01
BVAM_PSEUDOGT_ENABLED = False
BVAM_PSEUDOGT_TEMPERATURE = 0.1
BVAM_PSEUDOGT_MIN_CONFIDENCE = 0.1
BVAM_GATE_LOSS_WEIGHT = 0.0
BVAM_DIVERSITY_LOSS_WEIGHT = 0.0
BVAM_GATE_ENTROPY_WEIGHT = 0.0
BVAM_DELTA_REGULARIZER_WEIGHT = 0.0
BVAM_START_STEP = 0
BVAM_RAMP_STEPS = 0
TABA_EXPERIMENT = False
TABA_ENABLED = False
TABA_MODE = "disabled"
TABA_DIAGNOSTIC_ONLY = False
TABA_TRANSMISSION_MID = 0.50
TABA_TRANSMISSION_TEMPERATURE = 0.15
TABA_CHROMA_FLOOR = 0.20
TABA_UNCERTAINTY_LAMBDA = 0.50
TABA_WATER_AXIS_EPS = 1e-4
TABA_LOG_PATH = None
GRAC_EXPERIMENT = False
GRAC_OPTIONS = dict(GRAC_CONFIG)
GRAC_LOG_PATH = None
GCRC_EXPERIMENT = False
GCRC_OPTIONS = dict(GCRC_CONFIG)
GCRC_LOG_PATH = None
CRCC_EXPERIMENT = False
CRCC_OPTIONS = dict(CRCC_CONFIG)
CRCC_LOG_PATH = None
OUTPUT_ROOT = REPO_ROOT / "outputs" / "direct_mdrr_cica_four_scene_20260903"
RENDER_ROOT = REPO_ROOT / "renders" / "direct_mdrr_cica_four_scene_20260903"
ACTIVE_RENDER_ROOT = RENDER_ROOT
ALLOWED_GPUS = frozenset(str(index) for index in range(10))
ENFORCE_SCENE_GPU_ASSIGNMENT = True
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}


def _arm_use_mdrr(arm: str) -> bool:
    if arm in ("B", "N"):
        return False
    return bool(ACTIVE_MDRR_OVERRIDE) if ACTIVE_MDRR_OVERRIDE is not None else arm in ("A1", "A3")


def _arm_use_cica(arm: str) -> bool:
    if arm in ("B", "N"):
        return True
    enabled = ACTIVE_CICA_OVERRIDE if ACTIVE_CICA_OVERRIDE is not None else arm in ("A2", "A3")
    return bool(CICA_ENABLED and enabled)


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        return _sanitize(value.item() if value.numel() == 1 else value.tolist())
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, np.generic):
        return _sanitize(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")


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
            writer.writerow({key: _sanitize(row.get(key, "")) for key in fields})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_object(value: Any) -> str:
    """Hash nested checkpoint values deterministically across Tensor wrappers."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().contiguous().cpu()
            digest.update(b"tensor:")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray:")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
            return
        if isinstance(item, np.generic):
            update(item.item())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping:")
            for key in sorted(item, key=str):
                update(str(key))
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence:")
            digest.update(type(item).__name__.encode("ascii"))
            for child in item:
                update(child)
            return
        if isinstance(item, bytes):
            digest.update(b"bytes:")
            digest.update(item)
            return
        digest.update(type(item).__name__.encode("ascii"))
        digest.update(repr(item).encode("utf8"))

    update(value)
    return digest.hexdigest()


def _cpu_object(value: Any) -> Any:
    """Normalize checkpoint state before hashing so device placement is irrelevant."""

    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {str(key): _cpu_object(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_object(item) for item in value)
    return value


def _hash_tensor(value: Tensor) -> str:
    return _hash_bytes(value.detach().contiguous().cpu().numpy().tobytes())


def _hash_optional_tensor(value: Optional[Tensor]) -> Optional[str]:
    return None if value is None else _hash_tensor(value)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rng_manifest(state: Mapping[str, Any]) -> Dict[str, Any]:
    # Match the historical A0 manifest representation exactly. Model and
    # optimizer hashes continue to use the deterministic canonical hasher.
    result = {
        "python_sha256": _hash_bytes(pickle.dumps(state["python"], protocol=pickle.HIGHEST_PROTOCOL)),
        "numpy_sha256": _hash_bytes(pickle.dumps(state["numpy"], protocol=pickle.HIGHEST_PROTOCOL)),
        "torch_cpu_sha256": _hash_tensor(state["torch_cpu"]),
    }
    result["torch_cuda_sha256"] = [_hash_tensor(x) for x in state.get("torch_cuda", [])]
    return result


def _runtime(scene: str, gpu: str) -> Dict[str, Any]:
    if scene not in SCENE_GPUS:
        raise RuntimeError(f"unknown scene: {scene}")
    if ENFORCE_SCENE_GPU_ASSIGNMENT and SCENE_GPUS[scene] != str(gpu):
        raise RuntimeError(f"invalid scene/GPU assignment: {scene}/{gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose only the assigned physical GPU")
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("worker must see exactly logical cuda:0")
    props = torch.cuda.get_device_properties(0)
    return {
        "scene": scene,
        "physical_gpu": str(gpu),
        "logical_gpu": 0,
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _source_checkpoint(scene: str) -> Path:
    if RAW_DATA_ONLY:
        raise RuntimeError("raw_data_only runs do not have a source training checkpoint")
    return ACTIVE_SOURCE_ROOT / scene / "checkpoints" / "C0" / f"step-{START_STEP:09d}.ckpt"


def _source_sequence(scene: str) -> Path:
    if RAW_DATA_ONLY:
        raise RuntimeError("raw_data_only runs generate their own camera sequence")
    return ACTIVE_SOURCE_ROOT / scene / "camera_sequence.json"


def _active_snapshots(max_step: int) -> Tuple[int, ...]:
    if ACTIVE_COMPACT_OUTPUT:
        return (int(max_step),)
    return tuple(dict.fromkeys([step for step in ACTIVE_SNAPSHOT_STEPS if step <= int(max_step)] + [int(max_step)]))


def _scene_config(scene: str) -> Dict[str, Any]:
    return {
        "data_path": f"undistorted_data/undistorted_{scene}",
        "source_checkpoint": None if RAW_DATA_ONLY else str(_source_checkpoint(scene)),
    }


def _configure_model_config(config: Any) -> None:
    model = config.pipeline.model
    model.intrinsic_color_parameterization = "bvam_bounded_sh3" if BVAM_ENABLED else ("brash_bounded_sh3" if BRASH_ENABLED else ACTIVE_PARAMETERIZATION)
    model.bounded_headroom_alpha = HEADROOM_ALPHA
    model.degreewise_headroom_alphas = DEGREEWISE_HEADROOM_ALPHAS
    model.headroom_schedule = HEADROOM_SCHEDULE
    model.headroom_start_step = HEADROOM_START_STEP
    model.headroom_end_step = HEADROOM_END_STEP
    model.sh_degree = 3
    model.rasterize_mode = "classic"
    model.medium_context_mode = "dir_xy_camera"
    model.b_inf_mode = "tied"
    model.infinite_water_enabled = False
    model.camera_medium_observability_enabled = OCMC_ENABLED
    model.camera_medium_observability_strength = OCMC_STRENGTH
    model.camera_medium_observability_strength_start = OCMC_STRENGTH_START
    model.camera_medium_observability_strength_ramp_start_step = START_STEP
    model.camera_medium_observability_strength_ramp_steps = OCMC_STRENGTH_RAMP_STEPS
    model.camera_medium_ray_adaptive_observability_enabled = False
    model.medium_identifiability_enabled = False
    model.medium_identifiability_weight = 0.0
    model.coarse_depth_supervision_enabled = False
    model.stop_split_at = 10000
    model.appearance_lr_scale = 1.0
    model.brash_enabled = BRASH_ENABLED
    model.brash_residual_scale = BRASH_RESIDUAL_SCALE
    model.brash_gate_init_logit = BRASH_GATE_INIT_LOGIT
    model.brash_gate_mode = BRASH_GATE_MODE
    model.brash_gate_prior_enabled = BRASH_GATE_PRIOR_ENABLED
    model.brash_gate_prior_weight = BRASH_GATE_PRIOR_WEIGHT
    model.brash_min_support = BRASH_MIN_SUPPORT
    model.brash_transmission_floor = BRASH_TRANSMISSION_FLOOR
    model.brash_depth_stability_temperature = BRASH_DEPTH_TEMPERATURE
    model.brash_view_spread_scale = BRASH_VIEW_SPREAD_SCALE
    model.brash_gate_regularizer_start_step = BRASH_GATE_START_STEP
    model.brash_gate_regularizer_ramp_steps = BRASH_GATE_RAMP_STEPS
    model.bvam_enabled = BVAM_ENABLED
    model.bvam_num_experts = BVAM_NUM_EXPERTS
    model.bvam_delta_scale = BVAM_DELTA_SCALE
    model.bvam_gate_degree = BVAM_GATE_DEGREE
    model.bvam_gate_temperature = BVAM_GATE_TEMPERATURE
    model.bvam_min_support = BVAM_MIN_SUPPORT
    model.bvam_min_view_spread = BVAM_MIN_VIEW_SPREAD
    model.bvam_pseudogt_enabled = BVAM_PSEUDOGT_ENABLED
    model.bvam_pseudogt_temperature = BVAM_PSEUDOGT_TEMPERATURE
    model.bvam_pseudogt_min_confidence = BVAM_PSEUDOGT_MIN_CONFIDENCE
    model.bvam_gate_loss_weight = BVAM_GATE_LOSS_WEIGHT
    model.bvam_diversity_loss_weight = BVAM_DIVERSITY_LOSS_WEIGHT
    model.bvam_gate_entropy_weight = BVAM_GATE_ENTROPY_WEIGHT
    model.bvam_delta_regularizer_weight = BVAM_DELTA_REGULARIZER_WEIGHT
    model.bvam_start_step = BVAM_START_STEP
    model.bvam_ramp_steps = BVAM_RAMP_STEPS
    _configure_taba(model)


def _configure_model(model: Any) -> None:
    model.config.intrinsic_color_parameterization = "bvam_bounded_sh3" if BVAM_ENABLED else ("brash_bounded_sh3" if BRASH_ENABLED else ACTIVE_PARAMETERIZATION)
    model.config.bounded_headroom_alpha = HEADROOM_ALPHA
    model.config.degreewise_headroom_alphas = DEGREEWISE_HEADROOM_ALPHAS
    model.config.headroom_schedule = HEADROOM_SCHEDULE
    model.config.headroom_start_step = HEADROOM_START_STEP
    model.config.headroom_end_step = HEADROOM_END_STEP
    model.config.sh_degree = 3
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.camera_medium_observability_enabled = OCMC_ENABLED
    model.config.camera_medium_observability_strength = OCMC_STRENGTH
    model.config.camera_medium_observability_strength_start = OCMC_STRENGTH_START
    model.config.camera_medium_observability_strength_ramp_start_step = START_STEP
    model.config.camera_medium_observability_strength_ramp_steps = OCMC_STRENGTH_RAMP_STEPS
    model.config.camera_medium_ray_adaptive_observability_enabled = False
    model.config.medium_identifiability_enabled = False
    model.config.medium_identifiability_weight = 0.0
    model.config.coarse_depth_supervision_enabled = False
    model.config.stop_split_at = 10000
    model.config.appearance_lr_scale = 1.0
    model.config.brash_enabled = BRASH_ENABLED
    model.config.brash_residual_scale = BRASH_RESIDUAL_SCALE
    model.config.brash_gate_init_logit = BRASH_GATE_INIT_LOGIT
    model.config.brash_gate_mode = BRASH_GATE_MODE
    model.config.brash_gate_prior_enabled = BRASH_GATE_PRIOR_ENABLED
    model.config.brash_gate_prior_weight = BRASH_GATE_PRIOR_WEIGHT
    model.config.brash_min_support = BRASH_MIN_SUPPORT
    model.config.brash_transmission_floor = BRASH_TRANSMISSION_FLOOR
    model.config.brash_depth_stability_temperature = BRASH_DEPTH_TEMPERATURE
    model.config.brash_view_spread_scale = BRASH_VIEW_SPREAD_SCALE
    model.config.brash_gate_regularizer_start_step = BRASH_GATE_START_STEP
    model.config.brash_gate_regularizer_ramp_steps = BRASH_GATE_RAMP_STEPS
    model.config.bvam_enabled = BVAM_ENABLED
    model.config.bvam_num_experts = BVAM_NUM_EXPERTS
    model.config.bvam_delta_scale = BVAM_DELTA_SCALE
    model.config.bvam_gate_degree = BVAM_GATE_DEGREE
    model.config.bvam_gate_temperature = BVAM_GATE_TEMPERATURE
    model.config.bvam_min_support = BVAM_MIN_SUPPORT
    model.config.bvam_min_view_spread = BVAM_MIN_VIEW_SPREAD
    model.config.bvam_pseudogt_enabled = BVAM_PSEUDOGT_ENABLED
    model.config.bvam_pseudogt_temperature = BVAM_PSEUDOGT_TEMPERATURE
    model.config.bvam_pseudogt_min_confidence = BVAM_PSEUDOGT_MIN_CONFIDENCE
    model.config.bvam_gate_loss_weight = BVAM_GATE_LOSS_WEIGHT
    model.config.bvam_diversity_loss_weight = BVAM_DIVERSITY_LOSS_WEIGHT
    model.config.bvam_gate_entropy_weight = BVAM_GATE_ENTROPY_WEIGHT
    model.config.bvam_delta_regularizer_weight = BVAM_DELTA_REGULARIZER_WEIGHT
    model.config.bvam_start_step = BVAM_START_STEP
    model.config.bvam_ramp_steps = BVAM_RAMP_STEPS
    _configure_taba(model.config)


def _configure_taba(config: Any) -> None:
    config.taba_enabled = TABA_ENABLED
    config.taba_mode = TABA_MODE
    config.taba_diagnostic_only = TABA_DIAGNOSTIC_ONLY
    config.taba_gate_mode = "detached_transmission"
    config.taba_transmission_mid = TABA_TRANSMISSION_MID
    config.taba_transmission_temperature = TABA_TRANSMISSION_TEMPERATURE
    config.taba_chroma_floor = TABA_CHROMA_FLOOR
    config.taba_uncertainty_lambda = TABA_UNCERTAINTY_LAMBDA
    config.taba_water_axis_eps = TABA_WATER_AXIS_EPS
    config.taba_log_path = TABA_LOG_PATH
    for key, value in GRAC_OPTIONS.items():
        setattr(config, "grac_" + key, value)
    config.grac_log_path = GRAC_LOG_PATH
    for key, value in GCRC_OPTIONS.items():
        setattr(config, "gcrc_" + key, value)
    config.gcrc_log_path = GCRC_LOG_PATH
    for key, value in CRCC_OPTIONS.items():
        setattr(config, "crcc_" + key, value)
    config.crcc_log_path = CRCC_LOG_PATH


def _optimizer_groups(config: Any, model: Any) -> Dict[str, Any]:
    """Resolve the canonical optimizer groups plus BRASH's optional gate."""

    groups = model.get_param_groups()
    result = {}
    for name in groups:
        if name in config.optimizers:
            result[name] = config.optimizers[name]
            continue
        if name not in ("brash_gate", "bvam_delta", "bvam_gate"):
            raise KeyError(f"missing optimizer configuration for group {name}")
        template_name = "features_dc" if name in ("brash_gate", "bvam_gate") else "features_rest"
        if template_name not in config.optimizers:
            raise KeyError(f"{name} needs the {template_name} optimizer template")
        gate_config = copy.deepcopy(config.optimizers[template_name])
        optimizer_config = gate_config["optimizer"] if isinstance(gate_config, Mapping) else gate_config.optimizer
        optimizer_config.lr = min(float(optimizer_config.lr), 1e-3)
        scheduler_config = gate_config.get("scheduler") if isinstance(gate_config, Mapping) else gate_config.scheduler
        if scheduler_config is not None:
            scheduler_config.lr_final = min(float(scheduler_config.lr_final), 1e-3)
        result[name] = gate_config
    return result


def _new_branch(scene: str) -> Any:
    config = yaml.load(ACTIVE_SOURCE_CONFIG.read_text(encoding="utf8"), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = None
    config.load_step = None
    config.load_checkpoint = None
    config.pipeline.datamanager.load_depths = False
    config.pipeline.datamanager.dataparser.data = REPO_ROOT / _scene_config(scene)["data_path"]
    _configure_model_config(config)
    _set_random_seed(int(ACTIVE_TRAINING_SEED))
    pipeline = config.pipeline.setup(device=torch.device("cuda:0"), test_mode="test")
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"unexpected pipeline type: {type(pipeline)}")
    _configure_model(pipeline.model)
    pipeline.model.clear_camera_medium_ray_adaptive_observability_state()
    pipeline.model.set_camera_medium_observability_projector(None)
    pipeline.model.step = 0
    optimizers = Optimizers(_optimizer_groups(config, pipeline.model), pipeline.model.get_param_groups())
    pipeline.eval()
    branch = type("Branch", (), {"config": config, "pipeline": pipeline, "optimizers": optimizers, "scalers": {}})()
    if RAW_DATA_ONLY:
        _save_raw_initialization(branch, scene)
    return branch


def _save_raw_initialization(branch: Any, scene: str) -> Path:
    if ACTIVE_INITIALIZATION_DIR is None:
        raise RuntimeError("raw initialization directory was not configured")
    path = ACTIVE_INITIALIZATION_DIR / scene / "step-000000000.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    model = branch.pipeline.model
    data_path = REPO_ROOT / _scene_config(scene)["data_path"]
    payload = {"scene": scene, "seed": ACTIVE_TRAINING_SEED, "step": 0,
               "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "data_path": str(data_path), "data_identity": _sha256(data_path / "sparse/0" / "points3D.bin") if (data_path / "sparse/0/points3D.bin").is_file() else None,
               "camera_names": [str(row[1]) for row in _train_records(branch)],
               "initialization": "raw_colmap_and_random_network_only"}
    if not path.exists():
        torch.save(payload, path)
    elif _hash_object(torch.load(path, map_location="cpu", weights_only=False)["model"]) != _hash_object(payload["model"]):
        raise RuntimeError(f"step-0 initialization drift for {scene}: {path}")
    return path


def _release(branch: Optional[Any]) -> None:
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


def _load_source(branch: Any, scene: str) -> Mapping[str, Any]:
    if RAW_DATA_ONLY:
        model = branch.pipeline.model
        _configure_model(model)
        model.step = 0
        # OCMC is calibrated from this run's fresh raw initialization with the
        # same fixed-camera analysis used by the accepted native C0 path.
        ocmc_dir = ACTIVE_INITIALIZATION_DIR / scene / "ocmc_step0"
        samples, bank = FORMAL._build_samples(
            REPO_ROOT, ocmc_dir, scene, {**FORMAL.SCENES[scene], "locked_safe": False}, branch)
        analysis, analysis_meta = FORMAL._analysis_general(branch, samples)
        bundle = FORMAL._ocmc_bundle(analysis, "C0", 0)
        FORMAL._install_condition(model, "C0", bundle, None)
        _configure_model(model)
        bundle_hash = _hash_object(bundle)
        return {"raw_data_only": True, "ocmc_bundle": bundle, "ocmc_bank": bank,
                "ocmc_analysis_meta": analysis_meta, "ocmc_bundle_hash": bundle_hash,
                "raoc_state": None,
                "checkpoint": None, "initialization": str(_save_raw_initialization(branch, scene))}
    checkpoint = _source_checkpoint(scene)
    expected = {
        "Curasao": "06eeaa97550a5a49abdf483e291e637b58f1e6c608f7743935ebe57fbbaed950",
        "IUI3-RedSea": "73b7c7f5739b44950e63f815c0f24873db3fb00e68a79ac8133e3e6a298e86b3",
        "JapaneseGradens-RedSea": "5f52b47704b075a72de3ed14ef0147fdad194cd2311756325fc5e9338df266a4",
        "Panama": "f1b9a515c6cc4ffd30698d50f42f7b01291a3238dd9d38b33e05294b22cd4a23",
    }[scene]
    if ACTIVE_SOURCE_ROOT == SOURCE_ROOT and _sha256(checkpoint) != expected:
        raise RuntimeError(f"registered source checkpoint hash drift: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("branch") != "C0" or int(payload.get("absolute_step", -1)) != START_STEP:
        raise RuntimeError("source checkpoint is not OCMC C0@3000")
    if payload.get("raoc_state") is not None or payload.get("ocmc_bundle") is None:
        raise RuntimeError("source checkpoint must contain OCMC and no RAOC state")
    metadata = payload.get("metadata", {})
    source_parameterization = metadata.get("intrinsic_color_parameterization")
    compatible_source = {"bounded_selective_sh3": {"bounded_sh3", "bounded_selective_sh3"},
                         "bounded_sh3": {"bounded_sh3"},
                         # FSB keeps the same tensor layout as BND-v1 and can
                         # continue directly from a bounded_sh3 checkpoint.
                         "bounded_headroom_sh3": {"bounded_headroom_sh3", "bounded_sh3"}}
    if source_parameterization is not None and source_parameterization not in compatible_source.get(ACTIVE_PARAMETERIZATION, {ACTIVE_PARAMETERIZATION}):
        raise RuntimeError(
            f"source parameterization mismatch: checkpoint={source_parameterization!r}, "
            f"requested={ACTIVE_PARAMETERIZATION!r}"
        )
    branch.pipeline.model.load_state_dict(payload["model"], strict=True)
    branch.pipeline.model.step = START_STEP
    _configure_model(branch.pipeline.model)
    if payload.get("ocmc_bundle") is not None:
        FORMAL._install_condition(branch.pipeline.model, "C0", payload["ocmc_bundle"], None)
    # The formal loader installs the historical C0 condition with OCMC on.
    # Re-apply this worker's explicit switch so E1 can disable OCMC without
    # changing the source checkpoint or the frozen projector bundle.
    _configure_model(branch.pipeline.model)
    branch.optimizers = Optimizers(_optimizer_groups(branch.config, branch.pipeline.model), branch.pipeline.model.get_param_groups())
    source_optimizer_groups = set(payload["optimizers"])
    current_optimizer_groups = set(branch.optimizers.optimizers)
    missing_source_groups = source_optimizer_groups - current_optimizer_groups
    extra_current_groups = current_optimizer_groups - source_optimizer_groups
    if missing_source_groups or extra_current_groups - {"brash_gate", "bvam_delta", "bvam_gate"}:
        raise RuntimeError(
            "optimizer group mismatch at source checkpoint: "
            f"missing={sorted(missing_source_groups)}, extra={sorted(extra_current_groups)}"
        )
    for group, optimizer in branch.optimizers.optimizers.items():
        if group not in payload["optimizers"]:
            continue
        optimizer.load_state_dict(payload["optimizers"][group])
    for group, scheduler in branch.optimizers.schedulers.items():
        if group not in payload["schedulers"]:
            continue
        scheduler.load_state_dict(payload["schedulers"][group])
    branch.scalers = dict(payload.get("scalers", {}))
    branch.pipeline.eval()
    return payload


def _train_records(branch: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return FORMAL._train_records(branch.pipeline)


def _eval_records(branch: Any) -> List[Tuple[int, str, Cameras, Dict[str, Any]]]:
    return FORMAL._eval_records(branch.pipeline)


def _model_state_hash(model: Any) -> str:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return _hash_object(state)


def _optimizer_hash(optimizers: Optimizers, exclude_groups: Sequence[str] = ()) -> str:
    excluded = set(exclude_groups)
    return _hash_object({key: _canonical_optimizer_state(value.state_dict()) for key, value in optimizers.optimizers.items() if key not in excluded})


def _canonical_optimizer_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Hash optimizer values by group order, independent of runtime param IDs."""

    groups: List[Dict[str, Any]] = []
    state_values = state.get("state", {})
    for group in state.get("param_groups", []):
        parameter_ids = list(group.get("params", []))
        group_values = {key: value for key, value in group.items() if key != "params"}
        groups.append(
            {
                "group": group_values,
                "state": [_cpu_object(state_values.get(parameter_id, {})) for parameter_id in parameter_ids],
            }
        )
    return {"param_groups": groups}


def _scheduler_hash(optimizers: Optimizers, exclude_groups: Sequence[str] = ()) -> str:
    excluded = set(exclude_groups)
    return _hash_object({key: value.state_dict() for key, value in optimizers.schedulers.items() if key not in excluded})


def _locked_config(model: Any) -> Dict[str, Any]:
    cfg = model.config
    return {
        "intrinsic_color_parameterization": cfg.intrinsic_color_parameterization,
        "bounded_headroom_alpha": float(getattr(cfg, "bounded_headroom_alpha", 1.0)),
        "degreewise_headroom_alphas": list(getattr(cfg, "degreewise_headroom_alphas", (1.0, 1.0, 1.0))),
        "headroom_schedule": getattr(cfg, "headroom_schedule", "constant"),
        "headroom_start_step": int(getattr(cfg, "headroom_start_step", 3000)),
        "headroom_end_step": int(getattr(cfg, "headroom_end_step", 8000)),
        "sh_degree": int(cfg.sh_degree),
        "rasterize_mode": cfg.rasterize_mode,
        "medium_context_mode": cfg.medium_context_mode,
        "b_inf_mode": cfg.b_inf_mode,
        "infinite_water_enabled": bool(cfg.infinite_water_enabled),
        "camera_medium_observability_enabled": bool(cfg.camera_medium_observability_enabled),
        "camera_medium_observability_strength": float(cfg.camera_medium_observability_strength),
        "camera_medium_observability_strength_start": float(cfg.camera_medium_observability_strength_start),
        "camera_medium_observability_strength_ramp_start_step": int(cfg.camera_medium_observability_strength_ramp_start_step),
        "camera_medium_observability_strength_ramp_steps": int(cfg.camera_medium_observability_strength_ramp_steps),
        "camera_medium_ray_adaptive_observability_enabled": bool(cfg.camera_medium_ray_adaptive_observability_enabled),
        "stop_split_at": int(cfg.stop_split_at),
        "appearance_lr_scale": float(getattr(cfg, "appearance_lr_scale", 1.0)),
        "brash_enabled": bool(BRASH_ENABLED),
        "brash_gate_mode": BRASH_GATE_MODE,
        "brash_gate_prior_enabled": bool(BRASH_GATE_PRIOR_ENABLED),
        "brash_gate_prior_weight": float(BRASH_GATE_PRIOR_WEIGHT),
        "brash_residual_scale": float(BRASH_RESIDUAL_SCALE),
        "bvam_enabled": bool(BVAM_ENABLED),
        "bvam_num_experts": int(BVAM_NUM_EXPERTS),
        "bvam_delta_scale": float(BVAM_DELTA_SCALE),
        "bvam_gate_degree": int(BVAM_GATE_DEGREE),
        "bvam_gate_temperature": float(BVAM_GATE_TEMPERATURE),
        "bvam_pseudogt_enabled": bool(BVAM_PSEUDOGT_ENABLED),
        "bvam_gate_loss_weight": float(BVAM_GATE_LOSS_WEIGHT),
        "taba_enabled": bool(cfg.taba_enabled),
        "taba_mode": cfg.taba_mode,
        "taba_diagnostic_only": bool(cfg.taba_diagnostic_only),
        "taba_gate_mode": cfg.taba_gate_mode,
        "taba_transmission_mid": float(cfg.taba_transmission_mid),
        "taba_transmission_temperature": float(cfg.taba_transmission_temperature),
        "taba_chroma_floor": float(cfg.taba_chroma_floor),
        "taba_uncertainty_lambda": float(cfg.taba_uncertainty_lambda),
        "taba_water_axis_eps": float(cfg.taba_water_axis_eps),
        **{"grac_" + key: value for key, value in grac_config_values(cfg).items()},
        **{"gcrc_" + key: value for key, value in gcrc_config_values(cfg).items()},
        **{"crcc_" + key: value for key, value in crcc_config_values(cfg).items()},
    }


def _start_audit(
    branch: Any,
    payload: Mapping[str, Any],
    output_dir: Path,
    scene: str,
    arm: str,
    training_rng: Mapping[str, Any],
) -> Dict[str, Any]:
    model = branch.pipeline.model
    if payload.get("raw_data_only"):
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        audit = {"scene": scene, "arm": arm, "absolute_step": 0,
                 "initialization": "raw_colmap_and_random_network_only",
                 "model_hash": _hash_object(state), "optimizer_groups": sorted(branch.optimizers.optimizers),
                 "optimizer_steps": {name: 0 for name in branch.optimizers.optimizers},
                 "training_rng_manifest": _rng_manifest(training_rng),
                 "forward_probe": _forward_probe(model, branch.pipeline.datamanager),
                 "no_training_checkpoint_loaded": True}
        if not audit["forward_probe"]["finite"]:
            raise RuntimeError("raw initialization forward probe is non-finite")
        _write_json(output_dir / "start_state_equivalence.json", audit)
        return audit
    checkpoint_model_hash = _hash_object({key: value.detach().cpu() for key, value in payload["model"].items()})
    loaded_model_hash = _model_state_hash(model)
    loaded_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    source_state = {key: value.detach().cpu() for key, value in payload["model"].items()}
    compatibility_keys = []
    if GRAC_EXPERIMENT:
        compatibility_keys.extend(key for key in loaded_state if key.startswith("_grac_state.") and key not in source_state)
    if GCRC_EXPERIMENT:
        compatibility_keys.extend(key for key in loaded_state if key.startswith(("_gcrc_state.", "_grac_state.")) and key not in source_state)
    if CRCC_EXPERIMENT:
        compatibility_keys.extend(key for key in loaded_state if key.startswith("_crcc_state.") and key not in source_state)
    if BRASH_ENABLED and "gauss_params.brash_gate_logit" not in payload["model"]:
        compatibility_keys.append("gauss_params.brash_gate_logit")
    if BVAM_ENABLED:
        compatibility_keys.extend(
            key for key in ("gauss_params.bvam_delta", "gauss_params.bvam_gate")
            if key not in payload["model"]
        )
    for key in compatibility_keys:
        loaded_state.pop(key, None)
    topology_compatibility = bool(compatibility_keys)
    loaded_model_hash_without_new_params = _hash_object(loaded_state)
    source_model_hash = _hash_object(source_state)
    excluded_groups = tuple(
        group for group, enabled in (
            ("brash_gate", BRASH_ENABLED),
            ("bvam_delta", BVAM_ENABLED),
            ("bvam_gate", BVAM_ENABLED),
        ) if enabled
    )
    optimizer_hash = _optimizer_hash(branch.optimizers, excluded_groups)
    scheduler_hash = _scheduler_hash(branch.optimizers, excluded_groups)
    source_optimizer_hash = _hash_object(
        {key: _canonical_optimizer_state(value) for key, value in payload["optimizers"].items()}
    )
    source_scheduler_hash = _hash_object(payload["schedulers"])
    scaler_hash = _hash_object(branch.scalers)
    source_scaler_hash = _hash_object(payload.get("scalers", {}))
    rng_hash = _rng_manifest(training_rng)
    projector_hash = _hash_optional_tensor(model._camera_medium_observability_projector)
    forward_probe = _forward_probe(model, branch.pipeline.datamanager)
    audit = {
        "scene": scene,
        "arm": arm,
        "START_STATE_EQUIVALENCE": bool(
            (checkpoint_model_hash == loaded_model_hash if not topology_compatibility else loaded_model_hash_without_new_params == source_model_hash)
            and optimizer_hash == source_optimizer_hash
            and scheduler_hash == source_scheduler_hash
            and scaler_hash == source_scaler_hash
        ),
        "absolute_step": START_STEP,
        "source_checkpoint": str(_source_checkpoint(scene)),
        "source_checkpoint_sha256": _sha256(_source_checkpoint(scene)),
        "checkpoint_model_hash": checkpoint_model_hash,
        "loaded_model_hash": loaded_model_hash,
        "optimizer_hash": optimizer_hash,
        "source_optimizer_hash": source_optimizer_hash,
        "scheduler_hash": scheduler_hash,
        "source_scheduler_hash": source_scheduler_hash,
        "scaler_hash": scaler_hash,
        "source_scaler_hash": source_scaler_hash,
        "training_rng_manifest": rng_hash,
        "ocmc_projector_sha256": projector_hash,
        "locked_config": _locked_config(model),
        "raoc_enabled": False,
        "ocmc_enabled": OCMC_ENABLED,
        "scaler_state_present": bool(payload.get("scalers", {})),
        "rng_provenance": f"seed-{ACTIVE_TRAINING_SEED} continuation RNG is restored after branch setup; source checkpoint stores only RNG hashes",
        "forward_probe": forward_probe,
        "topology_compatibility_initialization": topology_compatibility,
        "initialized_missing_parameter_keys": compatibility_keys,
        "loaded_model_hash_without_new_params": loaded_model_hash_without_new_params,
    }
    _write_json(output_dir / "start_state_equivalence.json", audit)
    if not audit["START_STATE_EQUIVALENCE"]:
        raise RuntimeError("start-state equivalence failed")
    return audit


@torch.no_grad()
def _forward_probe(model: Any, datamanager: Any) -> Dict[str, Any]:
    """Record a deterministic source forward probe for the start-state audit."""

    camera = getattr(datamanager, "train_cameras", datamanager.train_dataset.cameras).to(model.device)
    index = 0
    outputs = model.get_outputs(camera[index : index + 1])
    names = ("pred_image", "rgb_object", "rgb_medium_finite", "rgb_tail", "medium_attn", "gaussian_view_rgb")
    hashes = {name: _hash_tensor(outputs[name]) for name in names if isinstance(outputs.get(name), Tensor)}
    finite = all(bool(torch.isfinite(value).all().item()) for value in outputs.values() if isinstance(value, Tensor))
    return {"camera_index": index, "finite": finite, "tensor_sha256": hashes}


def _camera_sequence(scene: str, output_dir: Path) -> List[Dict[str, Any]]:
    if RAW_DATA_ONLY:
        branch = _new_branch(scene)
        try:
            records = _train_records(branch)
            names = [str(row[1]) for row in records]
            rng = random.Random(ACTIVE_TRAINING_SEED + 1009)
            rows = []
            for step in range(1, FINAL_STEP + 1):
                index = rng.randrange(len(records))
                rows.append({"absolute_step": step, "camera_index": index, "camera_name": names[index]})
            payload = {"scene": scene, "rows": rows, "length": len(rows), "start_step_exclusive": 0,
                       "final_step_inclusive": FINAL_STEP, "CAMERA_SEQUENCE_RAW": True,
                       "sha256": _hash_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())}
            _write_json(output_dir / "camera_sequence_hashes.json", payload)
            _write_csv(output_dir / "camera_sequence.csv", rows)
            return rows
        finally:
            _release(branch)
    source = json.loads(_source_sequence(scene).read_text(encoding="utf8"))
    rows = [dict(row) for row in source["rows"] if START_STEP < int(row["absolute_step"]) <= FINAL_STEP]
    expected = FINAL_STEP - START_STEP
    if len(rows) != expected or [int(row["absolute_step"]) for row in rows] != list(range(START_STEP + 1, FINAL_STEP + 1)):
        raise RuntimeError("source camera sequence is not the registered contiguous continuation")
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf8")
    payload = {
        "scene": scene,
        "rows": rows,
        "length": len(rows),
        "start_step_exclusive": START_STEP,
        "final_step_inclusive": FINAL_STEP,
        "CAMERA_SEQUENCE_EXACT_MATCH": True,
        "sha256": _hash_bytes(encoded),
        "source_path": str(_source_sequence(scene)),
        "source_sha256": _sha256(_source_sequence(scene)),
    }
    _write_json(output_dir / "camera_sequence_hashes.json", payload)
    _write_csv(output_dir / "camera_sequence.csv", rows)
    return rows


def _camera_bank(branch: Any, output_dir: Path) -> Dict[str, Any]:
    records = _train_records(branch)
    ordered = sorted(enumerate(records), key=lambda item: (str(item[1][1]), item[0]))
    count = min(CICA_BANK_MAX, len(ordered))
    chosen: List[int] = []
    if count == 1:
        chosen = [ordered[0][0]]
    elif count > 1:
        positions = [int(round(i * (len(ordered) - 1) / float(count - 1))) for i in range(count)]
        chosen = [ordered[position][0] for position in positions]
    names = [records[index][1] for index in chosen]
    rows = [{"ordinal": i, "camera_index": index, "camera_id": records[index][1]} for i, index in enumerate(chosen)]
    payload = {
        "bank_max": CICA_BANK_MAX,
        "selected_count": len(chosen),
        "camera_indices": chosen,
        "camera_ids": names,
        "camera_id_sorted_uniform_sampling": True,
        "sorted_camera_ids": [str(item[1][1]) for item in ordered],
        "rows": rows,
        "bank_sha256": _hash_bytes(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf8")),
    }
    _write_json(output_dir / "cica_camera_bank.json", payload)
    return payload


@torch.no_grad()
def _build_brash_reliability_prior(
    model: Any,
    bank_records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]],
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    """Build detached training-camera reliability statistics for BRASH."""

    count = int(model.num_points)
    device = model.device
    support = torch.zeros(count, device=device, dtype=torch.float32)
    depth_sum = torch.zeros(count, device=device, dtype=torch.float32)
    depth_sq_sum = torch.zeros(count, device=device, dtype=torch.float32)
    transmission_sum = torch.zeros(count, device=device, dtype=torch.float32)
    direction_sum = torch.zeros(count, 3, device=device, dtype=torch.float32)
    for _index, _view_id, camera, _batch in bank_records:
        outputs = model.get_outputs_for_camera(camera.to(device))
        visible = outputs["gaussian_visible_mask"].reshape(-1).bool()
        if not bool(visible.any()):
            continue
        depths = outputs["projected_gaussian_depths"].reshape(-1).float()
        xys = getattr(model, "xys", None)
        if xys is None:
            raise RuntimeError("BRASH reliability requires projected Gaussian coordinates")
        xys = xys.reshape(-1, 2)
        transmission = outputs["transmission"].detach().float()
        height, width = int(transmission.shape[0]), int(transmission.shape[1])
        px = xys[:, 0].round().long().clamp(0, width - 1)
        py = xys[:, 1].round().long().clamp(0, height - 1)
        transmission_scalar = transmission[py, px].mean(dim=-1)
        center = camera.camera_to_worlds[0, :3, 3].to(device=device, dtype=torch.float32)
        directions = model.means.detach().float() - center[None, :]
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        mask = visible.float()
        support += mask
        depth_sum += depths * mask
        depth_sq_sum += depths.square() * mask
        transmission_sum += transmission_scalar * mask
        direction_sum += directions * mask[:, None]
    denom = support.clamp_min(1.0)
    depth_mean = depth_sum / denom
    depth_variance = (depth_sq_sum / denom - depth_mean.square()).clamp_min(0.0)
    transmission_mean = (transmission_sum / denom).clamp(0.0, 1.0)
    direction_mean = direction_sum / denom[:, None]
    view_spread = (1.0 - direction_mean.norm(dim=-1)).clamp(0.0, 1.0)
    confidence, components = build_reliability_prior(
        support=support,
        depth_variance=depth_variance,
        transmission=transmission_mean,
        view_spread=view_spread,
        bank_size=len(bank_records),
        minimum_support=int(BRASH_MIN_SUPPORT),
        transmission_floor=float(BRASH_TRANSMISSION_FLOOR),
        depth_temperature=float(BRASH_DEPTH_TEMPERATURE),
        view_spread_scale=float(BRASH_VIEW_SPREAD_SCALE),
    )
    prior = {
        "support": support.detach(),
        "depth_variance": depth_variance.detach(),
        "transmission": transmission_mean.detach(),
        "view_spread": view_spread.detach(),
        "confidence": confidence.reshape(-1).detach(),
        "target": components["target"].detach(),
    }
    stats = {
        "support_mean": float(support.mean().item()),
        "support_p10": float(torch.quantile(support, 0.10).item()),
        "support_p50": float(torch.quantile(support, 0.50).item()),
        "support_p90": float(torch.quantile(support, 0.90).item()),
        "transmission_mean": float(transmission_mean.mean().item()),
        "transmission_p90": float(torch.quantile(transmission_mean, 0.90).item()),
        "depth_stability_mean": float(torch.exp(-depth_variance / float(BRASH_DEPTH_TEMPERATURE)).mean().item()),
        "view_spread_mean": float(view_spread.mean().item()),
        "confidence_mean": float(confidence.mean().item()),
        "confidence_p10": float(torch.quantile(confidence.reshape(-1), 0.10).item()),
        "confidence_p50": float(torch.quantile(confidence.reshape(-1), 0.50).item()),
        "confidence_p90": float(torch.quantile(confidence.reshape(-1), 0.90).item()),
        "high_confidence_fraction": float((confidence.reshape(-1) >= 0.5).float().mean().item()),
        "population": count,
    }
    return prior, stats


def _partner_mapping(branch: Any, output_dir: Path) -> Dict[str, Any]:
    records = _train_records(branch)
    visibility: List[Tensor] = []
    for _index, _view_id, camera, _batch in records:
        geometry = FULL._project_for_camera(branch.pipeline.model, camera)
        visibility.append((geometry["radii"] > 0).reshape(-1).detach())
    items: List[Dict[str, Any]] = []
    for index, primary in enumerate(records):
        primary_camera = primary[2]
        cp = primary_camera.camera_to_worlds[0, :3, 3].detach().float()
        vp = primary_camera.camera_to_worlds[0, :3, 2].detach().float()
        candidates: List[Tuple[int, float, float, str]] = []
        for candidate_index, candidate in enumerate(records):
            if candidate_index == index:
                continue
            cc = candidate[2].camera_to_worlds[0, :3, 3].detach().float()
            vc = candidate[2].camera_to_worlds[0, :3, 2].detach().float()
            angle = float(torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(vp[None], vc[None]).squeeze(), -1.0, 1.0))).item())
            baseline = float(torch.linalg.vector_norm(cp - cc).item())
            candidates.append((candidate_index, angle, baseline, str(candidate[1])))
        preferred = [item for item in candidates if item[1] < 60.0]
        pool = sorted(preferred if preferred else candidates, key=lambda item: (-item[2], item[1], item[3], item[0]))
        selected = pool[0]
        for candidate in pool:
            shared = int((visibility[index] & visibility[candidate[0]]).sum().item())
            if shared >= MDRR_MIN_SHARED_VISIBLE:
                selected = candidate
                break
        shared_count = int((visibility[index] & visibility[selected[0]]).sum().item())
        items.append(
            {
                "primary_index": index,
                "primary_camera_id": str(primary[1]),
                "partner_index": selected[0],
                "partner_camera_id": selected[3],
                "direction_angle_deg": selected[1],
                "camera_center_baseline": selected[2],
                "angle_priority_lt_60_deg": bool(preferred),
                "shared_visible_gaussian_count": shared_count,
                "shared_visibility_requirement": MDRR_MIN_SHARED_VISIBLE,
                "shared_visibility_requirement_met": shared_count >= MDRR_MIN_SHARED_VISIBLE,
            }
        )
    payload = {
        "selection_rule": "prefer angle<60 degrees; among candidates prefer those meeting the fixed shared-visible requirement, then shared-visible count, baseline descending, angle, and deterministic camera-id/index tie break",
        "shared_visibility_requirement": MDRR_MIN_SHARED_VISIBLE,
        "rows": items,
        "mapping_sha256": _hash_bytes(json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf8")),
    }
    del visibility
    _write_json(output_dir / "mdrr_partner_mapping.json", payload)
    return payload


def _add_crcc_gradients(loss, params, group_indices):
    """Keep CRCC outside photometric routing and CICA's gradient calibration."""
    if loss is None:
        return {}
    gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    allowed = set(group_indices.get("features_dc", []) + group_indices.get("features_rest", []))
    if any(value is not None and not bool(torch.isfinite(value).all()) for value in gradients):
        raise RuntimeError("non-finite CRCC gradient")
    if _nonzero_group_norm(gradients, allowed) != 0:
        raise RuntimeError("CRCC gradient leaked outside appearance")
    for index in allowed:
        if gradients[index] is not None:
            value = gradients[index].detach()
            params[index].grad = value.clone() if params[index].grad is None else params[index].grad + value
    return _group_norms(gradients, group_indices)


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _gt(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return model.composite_with_background(model.get_gt_img(batch["image"]), background)


@torch.no_grad()
def _taba_snapshot_rows(model: Any, records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]], step: int) -> List[Dict[str, Any]]:
    """Detached per-training-view TABA diagnostics for an exact checkpoint state."""
    rows: List[Dict[str, Any]] = []
    for index, view_id, camera, batch in records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        gt = _gt(model, _batch_to_device(batch, model.device), outputs["background"]).float().clamp(0.0, 1.0)
        pred = outputs["pred_image"].float().clamp(0.0, 1.0)
        obj = outputs["rgb_object"].float()
        med = outputs["rgb_medium"].float()
        clear = outputs["rgb_clear"].float()
        transmission = outputs["transmission"].float().mean(dim=-1)
        residual = pred - gt
        obj_residual = (obj - gt).abs().mean(dim=-1)
        medium_residual = (med - (gt - obj)).abs().mean(dim=-1)
        clear_residual = (clear - obj).abs().mean(dim=-1)
        luma = residual.mean(dim=-1).abs()
        chroma = (residual - residual.mean(dim=-1, keepdim=True)).norm(dim=-1)
        bins = ((transmission < 0.33), (transmission >= 0.33) & (transmission < 0.66), transmission >= 0.66)
        visible = outputs.get("gaussian_visible_mask", torch.zeros(0, device=model.device)).reshape(-1).bool()
        support = float(visible.float().sum().item())
        if TABA_EXPERIMENT:
            rows.append({
                "absolute_step": int(step), "camera_index": int(index), "view_id": str(view_id),
                "bucket": "all", "pixel_count": int(transmission.numel()),
                **taba_statistics(outputs, xys=model.xys, radii=model.radii),
                **(grac_statistics(outputs) if GRAC_EXPERIMENT else {}),
                **(gcrc_statistics(outputs) if GCRC_EXPERIMENT else {}),
                **(crcc_statistics(outputs) if CRCC_EXPERIMENT else {}),
                "rgb_object_residual": float(obj_residual.mean()),
                "rgb_medium_residual": float(medium_residual.mean()),
            })
        for bucket, mask in zip(("low", "middle", "high"), bins):
            if not bool(mask.any()):
                continue
            rows.append({
                "absolute_step": int(step), "camera_index": int(index), "view_id": str(view_id), "bucket": bucket,
                "pixel_count": int(mask.sum().item()), "photometric_residual": float(residual[mask].abs().mean().item()),
                "rgb_object_residual": float(obj_residual[mask].mean().item()), "rgb_medium_residual": float(medium_residual[mask].mean().item()),
                "clear_residual": float(clear_residual[mask].mean().item()), "luminance_residual": float(luma[mask].mean().item()),
                "chromatic_residual": float(chroma[mask].mean().item()), "mean_transmission": float(transmission[mask].mean().item()),
                "support_count": support, "depth_variance": float(outputs["depth"][mask].float().var().item()),
                "medium_attenuation": float(outputs["medium_attn"][mask].float().mean().item()),
                "P_J_gt_1": float((outputs["J_gaussian_raw" if TABA_EXPERIMENT else "J_gaussian"][mask] > 1.0).float().mean().item()),
                "J_p99": float(torch.quantile(outputs["J_gaussian_raw" if TABA_EXPERIMENT else "J_gaussian"][mask].float().reshape(-1), 0.99).item()),
                "tau_D_p90": float(torch.quantile(outputs["tau_D"][mask].float().reshape(-1), 0.90).item()),
            })
    return rows


def _responsibility_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = FULL._mdrr_extension()
    return _EXTENSION


_EXTENSION: Any = None


def _cica_extension() -> Any:
    """Build the exact classic-compositing diagonal DC Jacobian accumulator."""

    global _CICA_EXTENSION
    if _CICA_EXTENSION is not None:
        return _CICA_EXTENSION
    from torch.utils.cpp_extension import load_inline

    cpp = r'''
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> cica_forward(
    torch::Tensor tile_bounds, torch::Tensor img_size,
    torch::Tensor gaussian_ids_sorted, torch::Tensor tile_bins,
    torch::Tensor xys, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, torch::Tensor medium_attn, torch::Tensor depths,
    torch::Tensor residual, int block_width, int legacy_mode);
'''
    cuda = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

constexpr float SH_C0 = 0.28209479177387814f;

__global__ void cica_kernel(
    int tile_x, int width, int height, const int* ids, const int2* bins,
    const float2* xys, const float3* conics, const float* colors,
    const float* opacities, const float* medium_attn, const float* depths,
    const float* residual, float* numerator, float* information, int legacy_mode) {
    const int tx = blockIdx.x, ty = blockIdx.y;
    const int px = tx * blockDim.x + threadIdx.x;
    const int py = ty * blockDim.y + threadIdx.y;
    const bool inside = px < width && py < height;
    const int pixel = py * width + px;
    const int2 range = bins[ty * tile_x + tx];
    const int batch_size_max = blockDim.x * blockDim.y;
    extern __shared__ unsigned char shared_raw[];
    int* id_batch = reinterpret_cast<int*>(shared_raw);
    float3* xy_opacity_batch = reinterpret_cast<float3*>(id_batch + batch_size_max);
    float3* conic_batch = reinterpret_cast<float3*>(xy_opacity_batch + batch_size_max);
    float* depth_batch = reinterpret_cast<float*>(conic_batch + batch_size_max);
    float transmittance = 1.0f;
    bool done = !inside;
    const int batches = (range.y - range.x + batch_size_max - 1) / batch_size_max;
    for (int batch = 0; batch < batches; ++batch) {
        if (__syncthreads_count(done) >= batch_size_max) break;
        const int start = range.x + batch * batch_size_max;
        const int local = threadIdx.y * blockDim.x + threadIdx.x;
        const int index = start + local;
        if (index < range.y) {
            const int gaussian = ids[index];
            id_batch[local] = gaussian;
            xy_opacity_batch[local] = make_float3(xys[gaussian].x, xys[gaussian].y, opacities[gaussian]);
            conic_batch[local] = conics[gaussian];
            depth_batch[local] = depths[gaussian];
        }
        __syncthreads();
        const int batch_size = min(batch_size_max, range.y - start);
        for (int item = 0; item < batch_size && !done; ++item) {
            const float3 conic = conic_batch[item];
            const float3 xy_opacity = xy_opacity_batch[item];
            const float depth = depth_batch[item];
            const float dx = xy_opacity.x - static_cast<float>(px);
            const float dy = xy_opacity.y - static_cast<float>(py);
            const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
            const float alpha = fminf(0.999f, xy_opacity.z * __expf(-sigma));
            float minimum_attenuation = inside
                ? fminf(medium_attn[3 * pixel], fminf(medium_attn[3 * pixel + 1], medium_attn[3 * pixel + 2]))
                : 0.0f;
            minimum_attenuation = fminf(0.0f, minimum_attenuation);
            if (!inside || sigma < 0.0f || alpha * __expf(-minimum_attenuation * depth) < 1.0f / 255.0f) continue;
            const float next_transmittance = transmittance * (1.0f - alpha);
            if (next_transmittance <= 1e-4f) {
                done = true;
                break;
            }
            const int gaussian = id_batch[item];
            const float visibility = alpha * transmittance;
            for (int channel = 0; channel < 3; ++channel) {
                const float color = colors[3 * gaussian + channel];
                const float attenuation = __expf(-medium_attn[3 * pixel + channel] * depth);
                const float jacobian = visibility * attenuation * (legacy_mode ? SH_C0 : color * (1.0f - color) * SH_C0);
                atomicAdd(&numerator[3 * gaussian + channel], jacobian * residual[3 * pixel + channel]);
                atomicAdd(&information[3 * gaussian + channel], jacobian * jacobian);
            }
            transmittance = next_transmittance;
        }
        __syncthreads();
    }
}

std::vector<torch::Tensor> cica_forward(
    torch::Tensor tile_bounds, torch::Tensor img_size,
    torch::Tensor gaussian_ids_sorted, torch::Tensor tile_bins,
    torch::Tensor xys, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, torch::Tensor medium_attn, torch::Tensor depths,
    torch::Tensor residual, int block_width, int legacy_mode) {
    auto numerator = torch::zeros_like(colors);
    auto information = torch::zeros_like(colors);
    const dim3 grid(tile_bounds[0].item<int>(), tile_bounds[1].item<int>(), 1);
    const dim3 block(block_width, block_width, 1);
    const int batch_size = block_width * block_width;
    const size_t shared_bytes = batch_size * (sizeof(int) + sizeof(float3) + sizeof(float3) + sizeof(float));
    cica_kernel<<<grid, block, shared_bytes>>>(
        grid.x, img_size[0].item<int>(), img_size[1].item<int>(),
        gaussian_ids_sorted.data_ptr<int>(), reinterpret_cast<const int2*>(tile_bins.data_ptr<int>()),
        reinterpret_cast<const float2*>(xys.data_ptr<float>()), reinterpret_cast<const float3*>(conics.data_ptr<float>()),
        colors.data_ptr<float>(), opacities.data_ptr<float>(), medium_attn.data_ptr<float>(), depths.data_ptr<float>(),
        residual.data_ptr<float>(), numerator.data_ptr<float>(), information.data_ptr<float>(), legacy_mode);
    const auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
    return {numerator, information};
}
'''
    _CICA_EXTENSION = load_inline(
        name="cica_exact_dc_jacobian_cuda_v1",
        cpp_sources=cpp,
        cuda_sources=cuda,
        functions=["cica_forward"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _CICA_EXTENSION


_CICA_EXTENSION: Any = None


@torch.no_grad()
def _responsibility_components(
    model: Any,
    camera: Cameras,
    outputs: Mapping[str, Tensor],
    residual: Tensor,
    additive: Tensor,
) -> Dict[str, Tensor]:
    """Extract exact classic compositing responsibilities and full response."""

    geometry = FULL._project_for_camera(model, camera)
    colors = outputs["gaussian_view_rgb"].detach().float()
    medium_attn = outputs["medium_attn"].detach().float()
    opacities = torch.sigmoid(model.opacities.detach()).reshape(-1)
    selected = torch.arange(int(model.num_points), device=model.device, dtype=torch.long)
    extension = _responsibility_extension()
    zeros = torch.zeros_like(residual[..., 0])
    stats, pixel_weight, selected_weight, _score = FULL._responsibility_forward(
        extension, geometry, colors, opacities, medium_attn, selected, residual, zeros, zeros
    )
    add_stats, _pw, _sw, _score = FULL._responsibility_forward(
        extension, geometry, colors, opacities, medium_attn, selected, additive, zeros, zeros
    )
    denom = stats[:, 0].double().clamp_min(EPS)
    effective = stats[:, 0].double() > RESPONSIBILITY_FLOOR
    e = torch.where(effective[:, None], stats[:, 1:4].double() / denom[:, None], torch.full((int(model.num_points), 3), float("nan"), device=model.device, dtype=torch.float64))
    d_direct = torch.where(effective[:, None], stats[:, 4:7].double() / denom[:, None], torch.full_like(e, float("nan")))
    d_add = torch.where(effective[:, None], add_stats[:, 1:4].double() / denom[:, None], torch.full_like(e, float("nan")))
    return {
        "geometry": geometry,
        "weights": stats[:, 0].detach(),
        "selected_weight": selected_weight.detach(),
        "e": e.detach(),
        "transmission": torch.where(effective, stats[:, 9].double() / denom, torch.zeros_like(denom)).detach(),
        "d": (d_direct + d_add).detach(),
        "d_direct": d_direct.detach(),
        "d_add": d_add.detach(),
        "pixel_weight": pixel_weight.detach(),
    }


@torch.no_grad()
def _cica_local_evidence(
    model: Any,
    camera: Cameras,
    outputs: Mapping[str, Tensor],
    residual: Tensor,
) -> Dict[str, Tensor]:
    """Accumulate the exact per-pixel DC-logit normal equation for one view."""

    geometry = FULL._project_for_camera(model, camera)
    gaussian_ids, tile_bins, tile_bounds = FULL._geometry_bins(geometry)
    height, width = int(geometry["height"]), int(geometry["width"])
    image_size = torch.tensor([width, height, 1], dtype=torch.int32, device=model.device)
    tile_size = torch.tensor(tile_bounds, dtype=torch.int32, device=model.device)
    # The CUDA accumulator returns q = d(render)/d(color) when legacy_mode=1.
    # For FSB, the exact DC-logit derivative is applied below in Python.  This
    # keeps the renderer kernel unchanged while matching the asymmetric
    # headroom map used by the forward model.
    kernel_legacy = (CICA_DERIVATIVE_MODE == "exact_fsb" and ACTIVE_PARAMETERIZATION == "bounded_headroom_sh3") or ACTIVE_PARAMETERIZATION == "legacy"
    numerator, information = _cica_extension().cica_forward(
        tile_size,
        image_size,
        gaussian_ids.contiguous(),
        tile_bins.contiguous(),
        geometry["xys"].float().contiguous(),
        geometry["conics"].float().contiguous(),
        outputs["gaussian_view_rgb"].detach().float().contiguous(),
        torch.sigmoid(model.opacities.detach()).reshape(-1).float().contiguous(),
        outputs["medium_attn"].detach().float().contiguous(),
        geometry["depths"].reshape(-1).float().contiguous(),
        residual.detach().reshape(-1, 3).float().contiguous(),
        16,
        int(kernel_legacy),
    )
    if CICA_DERIVATIVE_MODE == "exact_fsb" and ACTIVE_PARAMETERIZATION == "bounded_headroom_sh3":
        full_logits = outputs.get("gaussian_view_logits")
        dc_logits = outputs.get("gaussian_view_dc_logits")
        if not isinstance(full_logits, Tensor) or not isinstance(dc_logits, Tensor):
            raise RuntimeError("FSB exact CICA requires gaussian_view_logits and gaussian_view_dc_logits")
        c0 = torch.sigmoid(dc_logits.detach().float())
        residual_logits = full_logits.detach().float() - dc_logits.detach().float()
        pos = residual_logits >= 0.0
        r = residual_logits
        h_pos = torch.tanh(c0 * r)
        h_neg = torch.tanh((1.0 - c0) * r)
        d_pos = c0 * (1.0 - c0) * (1.0 - h_pos + (1.0 - c0) * r * (1.0 - h_pos.square()))
        d_neg = c0 * (1.0 - c0) * (1.0 + h_neg - c0 * r * (1.0 - h_neg.square()))
        d_fsb = torch.where(pos, d_pos, d_neg)
        numerator = numerator * d_fsb
        information = information * d_fsb.square()
    finite = torch.isfinite(numerator).all(dim=-1) & torch.isfinite(information).all(dim=-1)
    information_sum = information.sum(dim=-1)
    valid = finite & (information_sum > CICA_INFORMATION_FLOOR)
    # The accumulator solves for a feature-space update.  Bounded SH uses the
    # resulting SH-logit displacement for its sigmoid DC; Legacy uses the
    # linear SH displacement and therefore converts it to RGB with SH_C0.
    delta_parameter = numerator / information.clamp_min(CICA_INFORMATION_FLOOR)
    delta_parameter = torch.where(valid[:, None], delta_parameter, torch.zeros_like(delta_parameter))
    return {
        "delta_parameter": delta_parameter.detach(),
        "information": information.detach(),
        "information_sum": information_sum.detach(),
        "valid": valid.detach(),
    }


@torch.no_grad()
def _mdrr_control(
    model: Any,
    primary_camera: Cameras,
    primary_outputs: Mapping[str, Tensor],
    primary_batch: Mapping[str, Any],
    partner_camera: Cameras,
    partner_batch: Mapping[str, Any],
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    primary_gt = _gt(model, primary_batch, primary_outputs["background"]).detach().float()
    primary_pred = primary_outputs["pred_image"].detach().float()
    primary_residual = primary_gt - primary_pred
    primary_add = primary_outputs["rgb_medium_finite"].detach().float() + primary_outputs["rgb_tail"].detach().float()
    primary_renderer_state = {
        name: getattr(model, name, None)
        for name in ("xys", "radii", "depths", "xys_grad_abs", "last_size", "last_fx", "last_fy", "crop_box")
    }
    with torch.no_grad():
        was_training = model.training
        # Match the primary forward's resolution schedule.  At steps before
        # the final resolution stage, training mode intentionally renders at
        # half resolution; eval mode would make the partner residual shape
        # incompatible with the primary residual.
        try:
            model.train(mode=True)
            model.set_crop(None)
            # get_outputs_for_camera is a no-grad evaluation helper, whereas
            # the training-mode renderer retains screen-space gradients. Call
            # the underlying forward with gradients enabled, then discard its
            # graph. The primary renderer state is restored before returning
            # so normal densification consumes the primary view's buffers.
            with torch.enable_grad():
                partner_outputs = model.get_outputs(partner_camera.to(model.device))
            partner_outputs = {
                key: value.detach() if isinstance(value, Tensor) else value
                for key, value in partner_outputs.items()
            }
        finally:
            for name, value in primary_renderer_state.items():
                setattr(model, name, value)
            model.train(mode=was_training)
        partner_gt = _gt(model, partner_batch, partner_outputs["background"]).detach().float()
        partner_pred = partner_outputs["pred_image"].float()
        partner_residual = partner_gt - partner_pred
        partner_add = partner_outputs["rgb_medium_finite"].float() + partner_outputs["rgb_tail"].float()
    a = _responsibility_components(model, primary_camera, primary_outputs, primary_residual, primary_add)
    b = _responsibility_components(model, partner_camera, partner_outputs, partner_residual, partner_add)
    de = a["e"] - b["e"]
    dd = a["d"] - b["d"]
    de_norm = torch.linalg.vector_norm(de, dim=-1)
    dd_norm = torch.linalg.vector_norm(dd, dim=-1)
    q = (de * dd).sum(dim=-1) / (de_norm * dd_norm + EPS)
    valid = torch.isfinite(q) & (a["weights"] > RESPONSIBILITY_FLOOR) & (b["weights"] > RESPONSIBILITY_FLOOR) & (de_norm > 1e-8) & (dd_norm > 1e-8)
    q_pos = torch.where(valid, q.clamp(0.0, 1.0), torch.zeros_like(q)).float()
    selected = torch.arange(int(model.num_points), device=model.device, dtype=torch.long)
    zero = torch.zeros_like(primary_residual)
    extension = _responsibility_extension()
    score_stats, _pw, selected_weight, pixel_score = FULL._responsibility_forward(
        extension,
        a["geometry"],
        primary_outputs["gaussian_view_rgb"].detach().float(),
        torch.sigmoid(model.opacities.detach()).reshape(-1),
        primary_outputs["medium_attn"].detach().float(),
        selected,
        zero,
        torch.zeros_like(zero[..., 0]),
        torch.zeros_like(zero[..., 0]),
        q_pos,
    )
    del score_stats, _pw, selected_weight
    g = (pixel_score / a["pixel_weight"].clamp_min(EPS)).clamp(0.0, 1.0).detach()
    info = {
        "enabled": True,
        "valid_gaussian_fraction": float(valid.float().mean().item()),
        "q_mean": float(q[valid].mean().item()) if bool(valid.any()) else 0.0,
        "q_positive_fraction": float((q_pos > 0).float().mean().item()),
        "q_p50": float(torch.quantile(q[valid].float(), 0.50).item()) if bool(valid.any()) else 0.0,
        "g_mean": float(g.mean().item()),
        "g_p50": float(torch.quantile(g.float().reshape(-1), 0.50).item()),
        "g_p10": float(torch.quantile(g.float().reshape(-1), 0.10).item()),
        "g_p90": float(torch.quantile(g.float().reshape(-1), 0.90).item()),
        "g_p99": float(torch.quantile(g.float().reshape(-1), 0.99).item()),
        "full_response_used": True,
        "valid_gaussian_count": int(valid.sum().item()),
    }
    del partner_outputs, a, b, de, dd, q, q_pos
    return {"pixel_g": g, "q": valid.float()}, info


def _sample_image(value: Tensor, xys: Tensor) -> Tensor:
    height, width = int(value.shape[0]), int(value.shape[1])
    x = xys[:, 0].round().long().clamp(0, width - 1)
    y = xys[:, 1].round().long().clamp(0, height - 1)
    return value[y, x]


@torch.no_grad()
def _build_cica_controller(model: Any, bank_records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]]) -> Dict[str, Tensor]:
    """Build detached DC log-chroma targets using the direct compositing Jacobian.

    The target is intentionally built from the degree-zero bounded intrinsic
    color only.  View-dependent SH residuals are evidence for the base RGB
    objective, not part of the canonical chromaticity target.
    """

    count = int(model.num_points)
    chroma_views: List[Tensor] = []
    jacobian_views: List[Tensor] = []
    support = torch.zeros(count, device=model.device, dtype=torch.int16)
    for _index, _view_id, camera, _batch in bank_records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        gt = _gt(model, _batch_to_device(_batch, model.device), outputs["background"]).detach().float()
        residual = gt - outputs["pred_image"].detach().float()
        evidence = _cica_local_evidence(model, camera, outputs, residual)
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1).bool() & evidence["valid"]
        if not bool(visible.any()):
            del outputs, evidence
            continue
        if ACTIVE_PARAMETERIZATION == "legacy":
            dc = torch.clamp(SH_C0 * model.features_dc.detach().float() + 0.5, min=1e-6)
            corrected_dc = torch.clamp(dc + SH_C0 * evidence["delta_parameter"], min=1e-6)
        else:
            dc = outputs["gaussian_view_dc_rgb"].detach().float()
            dc_logits = outputs["gaussian_view_dc_logits"].detach().float()
            corrected_dc = torch.sigmoid(dc_logits + SH_C0 * evidence["delta_parameter"]).clamp(1e-6, 1.0 - 1e-6)
        information = torch.where(visible, evidence["information_sum"], torch.zeros_like(evidence["information_sum"]))
        chroma = torch.stack(
            [torch.log((corrected_dc[:, 0] + 1e-6) / (corrected_dc[:, 1] + 1e-6)), torch.log((corrected_dc[:, 2] + 1e-6) / (corrected_dc[:, 1] + 1e-6))], dim=-1
        ).double()
        chroma = torch.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
        chroma_views.append(chroma)
        jacobian_views.append(information.double())
        support += visible.to(torch.int16)
        del outputs, gt, residual, evidence, dc, corrected_dc, information, chroma
    valid = support >= int(CICA_MIN_SUPPORT)
    if chroma_views:
        values = torch.stack(chroma_views, dim=0)
        weights = torch.stack(jacobian_views, dim=0)
        order = torch.argsort(values, dim=0, stable=True)
        sorted_values = torch.gather(values, 0, order)
        sorted_weights = torch.gather(weights[..., None].expand(-1, -1, 2), 0, order)
        half = sorted_weights.sum(dim=0) * 0.5
        cumulative = sorted_weights.cumsum(dim=0)
        median_index = (cumulative >= half[None]).to(torch.int64).argmax(dim=0)
        target = torch.gather(sorted_values, 0, median_index[None].expand(1, -1, -1)).squeeze(0)
        weight_sum = weights.sum(dim=0)
    else:
        target = torch.zeros(count, 2, device=model.device, dtype=torch.float64)
        weight_sum = torch.zeros(count, device=model.device, dtype=torch.float64)
    target = torch.where(valid[:, None], target, torch.zeros_like(target)).detach()
    gate = valid.float().detach()
    return {
        "target_log_chroma": target,
        "weight": weight_sum.float().detach(),
        "support": support.detach(),
        "gate": gate,
        "valid_count": valid.sum().detach(),
        "population": torch.tensor(count, device=model.device),
        "valid_target_fraction": valid.float().mean().detach(),
    }


def _cica_loss(model: Any, controller: Optional[Mapping[str, Tensor]]) -> Tuple[Tensor, Dict[str, Any]]:
    zero = model.features_dc.sum() * 0.0
    if controller is None:
        return zero, {"enabled": False, "valid_gaussians": 0, "valid_fraction": 0.0, "raw_loss": 0.0}
    if int(controller["population"].item()) != int(model.num_points):
        raise RuntimeError("CICA target population is stale")
    if ACTIVE_PARAMETERIZATION == "legacy":
        dc = torch.clamp(model.features_dc * SH_C0 + 0.5, min=1e-6)
    else:
        dc = torch.sigmoid(model.features_dc * SH_C0)
    current = torch.stack(
        [torch.log((dc[:, 0] + 1e-6) / (dc[:, 1] + 1e-6)), torch.log((dc[:, 2] + 1e-6) / (dc[:, 1] + 1e-6))], dim=-1
    )
    target = controller["target_log_chroma"].to(device=model.device, dtype=current.dtype)
    gate = controller["gate"].to(device=model.device, dtype=current.dtype)
    weight = controller["weight"].to(device=model.device, dtype=current.dtype).clamp_min(0.0)
    residual = current - target
    huber = F.huber_loss(residual, target.new_zeros(residual.shape), reduction="none", delta=CICA_HUBER_DELTA).mean(dim=-1)
    coefficients = gate * weight.clamp_min(1e-6)
    loss = (coefficients * huber).sum() / coefficients.sum().clamp_min(EPS)
    valid_correction = (-residual[gate > 0]).detach()
    if valid_correction.numel():
        delta_rg = valid_correction[:, 0]
        delta_bg = valid_correction[:, 1]
        delta_rg_mean = float(delta_rg.mean().item())
        delta_bg_mean = float(delta_bg.mean().item())
        delta_rg_abs_mean = float(delta_rg.abs().mean().item())
        delta_bg_abs_mean = float(delta_bg.abs().mean().item())
        median_abs_delta_chi = float(torch.quantile(valid_correction.abs().reshape(-1), 0.50).item())
        delta_rg_p10 = float(torch.quantile(delta_rg.float(), 0.10).item())
        delta_rg_p50 = float(torch.quantile(delta_rg.float(), 0.50).item())
        delta_rg_p90 = float(torch.quantile(delta_rg.float(), 0.90).item())
        delta_bg_p10 = float(torch.quantile(delta_bg.float(), 0.10).item())
        delta_bg_p50 = float(torch.quantile(delta_bg.float(), 0.50).item())
        delta_bg_p90 = float(torch.quantile(delta_bg.float(), 0.90).item())
        delta_rg_positive_fraction = float((delta_rg > CICA_DIRECTION_EPS).float().mean().item())
        delta_rg_negative_fraction = float((delta_rg < -CICA_DIRECTION_EPS).float().mean().item())
        delta_bg_positive_fraction = float((delta_bg > CICA_DIRECTION_EPS).float().mean().item())
        delta_bg_negative_fraction = float((delta_bg < -CICA_DIRECTION_EPS).float().mean().item())
    else:
        delta_rg_mean = delta_bg_mean = delta_rg_abs_mean = delta_bg_abs_mean = median_abs_delta_chi = 0.0
        delta_rg_p10 = delta_rg_p50 = delta_rg_p90 = 0.0
        delta_bg_p10 = delta_bg_p50 = delta_bg_p90 = 0.0
        delta_rg_positive_fraction = delta_rg_negative_fraction = 0.0
        delta_bg_positive_fraction = delta_bg_negative_fraction = 0.0
    color_prior_collapse_warning = bool(
        delta_bg_negative_fraction >= CICA_DIRECTION_COLLAPSE_FRACTION
        and delta_bg_abs_mean > CICA_DIRECTION_EPS
    )
    return loss, {
        "enabled": True,
        "valid_gaussians": int(controller["valid_count"].item()),
        "valid_fraction": float(gate.mean().item()),
        "raw_loss": float(loss.detach().item()),
        "delta_rg_mean": delta_rg_mean,
        "delta_bg_mean": delta_bg_mean,
        "delta_rg_abs_mean": delta_rg_abs_mean,
        "delta_bg_abs_mean": delta_bg_abs_mean,
        "delta_rg_p10": delta_rg_p10,
        "delta_rg_p50": delta_rg_p50,
        "delta_rg_p90": delta_rg_p90,
        "delta_bg_p10": delta_bg_p10,
        "delta_bg_p50": delta_bg_p50,
        "delta_bg_p90": delta_bg_p90,
        "delta_rg_positive_fraction": delta_rg_positive_fraction,
        "delta_rg_negative_fraction": delta_rg_negative_fraction,
        "delta_bg_positive_fraction": delta_bg_positive_fraction,
        "delta_bg_negative_fraction": delta_bg_negative_fraction,
        "color_prior_collapse_warning": color_prior_collapse_warning,
        "median_abs_delta_chi": median_abs_delta_chi,
    }


@torch.no_grad()
def _build_bvam_controller(
    model: Any,
    bank_records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build detached two-state BVAM gate targets from training-camera evidence."""

    count = int(model.num_points)
    views = len(bank_records)
    q_views: List[Tensor] = []
    directions = torch.zeros(count, 3, device=model.device, dtype=torch.float32)
    support = torch.zeros(count, device=model.device, dtype=torch.float32)
    for _index, _view_id, camera, batch in bank_records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        gt = _gt(model, _batch_to_device(batch, model.device), outputs["background"]).detach().float()
        residual = (gt - outputs["pred_image"].detach().float()).abs().mean(dim=-1)
        visible = outputs["gaussian_visible_mask"].reshape(-1).bool()
        xys = model.xys.detach().reshape(-1, 2)
        transmission = outputs["transmission"].detach().float().mean(dim=-1)
        px = xys[:, 0].round().long().clamp(0, transmission.shape[1] - 1)
        py = xys[:, 1].round().long().clamp(0, transmission.shape[0] - 1)
        pixel_residual = residual[py, px]
        pixel_transmission = transmission[py, px]
        opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)
        depth = outputs["projected_gaussian_depths"].detach().float().reshape(-1)
        sensitivity = opacity * torch.exp(-depth.clamp_min(0.0) * outputs["medium_attn"].detach().float()[py, px].mean(dim=-1))
        q = (pixel_residual * pixel_transmission * sensitivity).clamp_min(0.0)
        q_views.append(torch.where(visible, q, torch.zeros_like(q)).detach())
        center = camera.camera_to_worlds[0, :3, 3].to(device=model.device, dtype=torch.float32)
        direction = model.means.detach().float() - center[None, :]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        directions += direction * visible[:, None].float()
        support += visible.float()
        del outputs, gt, residual, transmission, pixel_residual, pixel_transmission, opacity, depth, sensitivity, q

    q_values = torch.stack(q_views, dim=0) if q_views else torch.zeros(0, count, device=model.device)
    q_mean = q_values.mean(dim=0) if views else torch.zeros(count, device=model.device)
    q_std = q_values.std(dim=0, unbiased=False) if views else torch.zeros(count, device=model.device)
    if views:
        center_1 = torch.quantile(q_values, 0.25, dim=0)
        center_2 = torch.quantile(q_values, 0.75, dim=0)
    else:
        center_1 = torch.zeros(count, device=model.device)
        center_2 = torch.zeros(count, device=model.device)
    separation = (center_2 - center_1).abs() / q_mean.abs().clamp_min(1e-6)
    stability = torch.exp(-q_std / q_mean.abs().clamp_min(1e-6)).clamp(0.0, 1.0)
    support_conf = (support / float(max(views, 1))).clamp(0.0, 1.0)
    spread = (1.0 - (directions / support.clamp_min(1.0)[:, None]).norm(dim=-1)).clamp(0.0, 1.0)
    spread_conf = (spread / max(float(BVAM_MIN_VIEW_SPREAD), 1e-6)).clamp(0.0, 1.0)
    confidence = support_conf * spread_conf * separation.clamp(0.0, 1.0) * stability
    valid = (support >= float(BVAM_MIN_SUPPORT)) & (spread >= float(BVAM_MIN_VIEW_SPREAD))
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence)).detach()
    if views:
        logits = torch.stack(
            [-(q_values - center_1[None, :]).abs(), -(q_values - center_2[None, :]).abs()],
            dim=-1,
        ) / float(max(BVAM_PSEUDOGT_TEMPERATURE, 1e-6))
        assignments = torch.softmax(logits, dim=-1)
        assignment_confidence = (assignments.max(dim=-1).values * confidence[None, :]).detach()
        assignments = assignments.detach()
    else:
        assignments = torch.zeros(0, count, 2, device=model.device)
        assignment_confidence = assignments.clone()
    camera_to_bank = {int(index): row for row, (index, _view_id, _camera, _batch) in enumerate(bank_records)}
    return {
        "population": count,
        "bank_indices": [int(index) for index, _view_id, _camera, _batch in bank_records],
        "camera_to_bank": camera_to_bank,
        "support": support.detach(),
        "view_spread": spread.detach(),
        "q_mean": q_mean.detach(),
        "q_std": q_std.detach(),
        "center_1": center_1.detach(),
        "center_2": center_2.detach(),
        "confidence": confidence.detach(),
        "valid": valid.detach(),
        "assignments": assignments,
        "assignment_confidence": assignment_confidence,
        "stats": {
            "support_mean": float(support.mean().item()),
            "support_p50": float(torch.quantile(support, 0.50).item()),
            "view_spread_mean": float(spread.mean().item()),
            "q_mean": float(q_mean.mean().item()),
            "q_std_mean": float(q_std.mean().item()),
            "q_separation_mean": float(separation.mean().item()),
            "pseudo_confidence_mean": float(confidence.mean().item()),
            "pseudo_confidence_p90": float(torch.quantile(confidence, 0.90).item()),
            "valid_gaussian_fraction": float(valid.float().mean().item()),
            "population": count,
        },
    }


def _bvam_loss(
    model: Any,
    controller: Optional[Mapping[str, Any]],
    outputs: Mapping[str, Tensor],
    camera_index: int,
    step: int,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Compute BVAM-only auxiliary losses against detached assignments."""

    zero = model.features_dc.sum() * 0.0
    if controller is None or not BVAM_ENABLED or step < int(BVAM_START_STEP):
        return zero, {"enabled": False, "progress": 0.0, "pseudo_valid_fraction": 0.0}
    progress = 1.0 if BVAM_RAMP_STEPS <= 0 else min(max((step - BVAM_START_STEP) / float(BVAM_RAMP_STEPS), 0.0), 1.0)
    gate = outputs.get("bvam_gate_probs")
    deltas = outputs.get("bvam_deltas")
    if not isinstance(gate, Tensor) or not isinstance(deltas, Tensor):
        raise RuntimeError("BVAM outputs are missing from the renderer")
    bank_row = controller["camera_to_bank"].get(int(camera_index))
    if BVAM_PSEUDOGT_ENABLED and bank_row is not None:
        target = controller["assignments"][bank_row].to(device=gate.device, dtype=gate.dtype)
        confidence = controller["assignment_confidence"][bank_row].to(device=gate.device, dtype=gate.dtype)
        mask = confidence >= float(BVAM_PSEUDOGT_MIN_CONFIDENCE)
        log_gate = gate.clamp_min(1e-7).log()
        gate_loss = -(target * log_gate).sum(dim=-1)
        gate_loss = (gate_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        pseudo_valid_fraction = float(mask.float().mean().item())
    else:
        gate_loss = zero
        pseudo_valid_fraction = 0.0
    delta_norm = deltas.float().norm(dim=-1)
    delta_regularizer = delta_norm.square().mean()
    expert_difference = (deltas[:, 0] - deltas[:, 1]).float().norm(dim=-1)
    diversity_loss = torch.exp(-expert_difference.square()).mean()
    entropy = -(gate.clamp_min(1e-7) * gate.clamp_min(1e-7).log()).sum(dim=-1).mean()
    total = progress * (
        float(BVAM_GATE_LOSS_WEIGHT) * gate_loss
        + float(BVAM_DIVERSITY_LOSS_WEIGHT) * diversity_loss
        + float(BVAM_GATE_ENTROPY_WEIGHT) * entropy
        + float(BVAM_DELTA_REGULARIZER_WEIGHT) * delta_regularizer
    )
    return total, {
        "enabled": bool(progress > 0.0),
        "progress": float(progress),
        "gate_loss": float(gate_loss.detach().item()),
        "diversity_loss": float(diversity_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "delta_regularizer": float(delta_regularizer.detach().item()),
        "pseudo_valid_fraction": pseudo_valid_fraction,
        "expert_usage_1": float(gate[:, 0].detach().mean().item()),
        "expert_usage_2": float(gate[:, 1].detach().mean().item()),
        "expert_entropy": float(entropy.detach().item()),
        "expert_collapse_fraction": float((expert_difference < 1e-4).float().mean().item()),
        "delta_norm_mean": float(delta_norm.detach().mean().item()),
        "delta_norm_p90": float(torch.quantile(delta_norm.detach().reshape(-1), 0.90).item()),
    }


def _parameter_entries(model: Any) -> Tuple[List[torch.nn.Parameter], Dict[str, List[int]]]:
    groups = model.get_param_groups()
    params: List[torch.nn.Parameter] = []
    group_indices: Dict[str, List[int]] = {}
    seen: Dict[int, int] = {}
    for name, values in groups.items():
        indices: List[int] = []
        for param in values:
            identifier = id(param)
            if identifier not in seen:
                seen[identifier] = len(params)
                params.append(param)
            indices.append(seen[identifier])
        group_indices[name] = indices
    return params, group_indices


def _norm(values: Sequence[Optional[Tensor]]) -> float:
    return math.sqrt(sum(float(value.detach().float().square().sum().item()) for value in values if value is not None))


def _nonzero_group_norm(values: Sequence[Optional[Tensor]], allowed: set[int]) -> float:
    return _norm([value for index, value in enumerate(values) if index not in allowed])


def _output_mean_abs(outputs: Mapping[str, Any], name: str) -> float:
    value = outputs.get(name)
    if not isinstance(value, Tensor):
        return 0.0
    return float(value.detach().float().abs().mean().item())


def _clear_distribution(value: Tensor) -> Dict[str, float]:
    """Summarize the native clear render without applying color correction."""

    raw = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    flat = raw.reshape(-1, 3)
    if flat.numel() == 0:
        return {
            "clear_raw_mean_r": 0.0,
            "clear_raw_mean_g": 0.0,
            "clear_raw_mean_b": 0.0,
            "clear_raw_p99": 0.0,
            "clear_raw_blue_minus_red": 0.0,
            "clear_raw_blue_minus_green": 0.0,
        }
    means = flat.mean(dim=0)
    return {
        "clear_raw_mean_r": float(means[0].item()),
        "clear_raw_mean_g": float(means[1].item()),
        "clear_raw_mean_b": float(means[2].item()),
        "clear_raw_p99": float(torch.quantile(flat, 0.99).item()),
        "clear_raw_blue_minus_red": float((means[2] - means[0]).item()),
        "clear_raw_blue_minus_green": float((means[2] - means[1]).item()),
    }


def _save_native_rgb(path: Path, value: Tensor) -> None:
    """Encode an already-rendered RGB tensor; no enhancement or recoloring."""

    array = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=1.0, neginf=0.0)
    array = array.clamp(0.0, 1.0).cpu().numpy()
    Image.fromarray((array * 255.0).round().astype(np.uint8)).save(path)


def _route_loss(model: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any], pixel_weight: Tensor) -> Tensor:
    gt = _gt(model, batch, outputs["background"])
    pred = outputs["pred_image"]
    weight = pixel_weight.to(device=pred.device, dtype=pred.dtype)
    if weight.ndim == 2:
        weight = weight[..., None]
    if weight.shape[:2] != pred.shape[:2]:
        raise RuntimeError(f"MDRR pixel map shape {tuple(weight.shape)} does not match RGB {tuple(pred.shape)}")
    if "mask" in batch:
        mask = model._downscale_if_required(batch["mask"]).to(model.device)
        gt = gt * mask
        pred = pred * mask
        weight = weight * mask

    denominator = pred.detach() + 1e-3
    if model.config.main_loss == "l1":
        reconstruction = torch.abs(gt - pred).mean(dim=-1, keepdim=True)
    elif model.config.main_loss == "reg_l1":
        reconstruction = torch.abs((gt - pred) / denominator).mean(dim=-1, keepdim=True)
    elif model.config.main_loss == "reg_l2":
        reconstruction = (((pred - gt) / denominator) ** 2).mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown main_loss: {model.config.main_loss}")

    if model.config.ssim_loss != "ssim":
        simloss = 1 - model.ssim(
            (gt / denominator).permute(2, 0, 1)[None, ...],
            (pred / denominator).permute(2, 0, 1)[None, ...],
        )
    else:
        simloss = 1 - model.ssim(
            gt.permute(2, 0, 1)[None, ...], pred.permute(2, 0, 1)[None, ...]
        )

    # The complementary routes sum exactly to the unweighted base loss:
    # route means are retained only to apportion the global SSIM scalar.
    return (1.0 - model.config.ssim_lambda) * (weight * reconstruction).mean() + model.config.ssim_lambda * simloss * weight.mean()


def _assign_grads(params: Sequence[torch.nn.Parameter], grads: Sequence[Optional[Tensor]]) -> None:
    for param, grad in zip(params, grads):
        param.grad = None if grad is None else grad.detach().clone()


def _combine_gradients(
    params: Sequence[torch.nn.Parameter],
    groups: Mapping[str, List[int]],
    base: Sequence[Optional[Tensor]],
    surface: Sequence[Optional[Tensor]],
    medium: Sequence[Optional[Tensor]],
    auxiliary: Sequence[Optional[Tensor]],
    cica: Sequence[Optional[Tensor]],
    use_mdrr: bool,
    brash: Optional[Sequence[Optional[Tensor]]] = None,
    bvam: Optional[Sequence[Optional[Tensor]]] = None,
) -> Dict[str, float]:
    appearance = set(groups.get("features_dc", [])) | set(groups.get("features_rest", []))
    medium_ids = set(groups.get("medium_mlp", [])) | set(groups.get("direction_encoding", []))
    geometry = set(groups.get("means", [])) | set(groups.get("scales", [])) | set(groups.get("quats", [])) | set(groups.get("opacities", []))
    features_dc = set(groups.get("features_dc", []))
    features_rest = set(groups.get("features_rest", []))
    brash_gate = set(groups.get("brash_gate", []))
    bvam_params = set(groups.get("bvam_delta", [])) | set(groups.get("bvam_gate", []))
    brash = brash if brash is not None else [None] * len(params)
    bvam = bvam if bvam is not None else [None] * len(params)
    for name, gradients, allowed in (
        ("auxiliary appearance", auxiliary, features_rest),
        ("CICA", cica, features_dc),
        ("BRASH", brash, brash_gate),
        ("BVAM", bvam, bvam_params),
    ):
        for index, gradient in enumerate(gradients):
            if gradient is not None and index not in allowed and float(gradient.detach().float().abs().max().item()) > 0.0:
                raise RuntimeError(f"{name} gradient escaped its registered parameter group")
    final: List[Optional[Tensor]] = []
    for index, _param in enumerate(params):
        if use_mdrr and index in appearance:
            value = surface[index]
        elif use_mdrr and index in medium_ids:
            value = medium[index]
        else:
            value = base[index]
        if index in geometry:
            value = base[index]
        if index in features_rest:
            if value is None:
                value = auxiliary[index]
            elif auxiliary[index] is not None:
                value = value + auxiliary[index]
        if index in features_dc:
            if value is None:
                value = cica[index]
            elif cica[index] is not None:
                value = value + cica[index]
        if index in brash_gate and brash[index] is not None:
            if value is None:
                value = brash[index]
            else:
                value = value + brash[index]
        if index in bvam_params and bvam[index] is not None:
            if value is None:
                value = bvam[index]
            else:
                value = value + bvam[index]
        final.append(value)
    _assign_grads(params, final)
    return {
        "base_grad_l2": _norm(base),
        "surface_grad_l2": _norm(surface),
        "medium_grad_l2": _norm(medium),
        "auxiliary_grad_l2": _norm(auxiliary),
        "cica_grad_l2": _norm(cica),
        "auxiliary_escape_grad_l2": _nonzero_group_norm(auxiliary, features_rest),
        "cica_escape_grad_l2": _nonzero_group_norm(cica, features_dc),
        "brash_grad_l2": _norm(brash),
        "brash_escape_grad_l2": _nonzero_group_norm(brash, brash_gate),
        "bvam_grad_l2": _norm(bvam),
        "bvam_escape_grad_l2": _nonzero_group_norm(bvam, bvam_params),
        "final_grad_l2": _norm(final),
        "geometry_base_grad_l2": _norm([base[index] for index in geometry]),
    }


def _group_norms(values: Sequence[Optional[Tensor]], groups: Mapping[str, List[int]]) -> Dict[str, float]:
    return {
        name: _norm([values[index] for index in indices])
        for name, indices in groups.items()
    }


def _gradient_routing_audit(
    branch: Any,
    scene: str,
    arm: str,
    output_dir: Path,
    records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]],
    partner: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> Dict[str, Any]:
    """Exercise both module routes before training and record direct gradients.

    The audit uses the common start state and never steps an optimizer.  It is
    intentionally separate from the training loop so a routing regression
    fails before any formal checkpoint is produced.
    """

    model = branch.pipeline.model
    original_step = int(model.step)
    was_training = model.training
    model.train()
    try:
        # Probe after every registered auxiliary route is active.  In
        # particular, BRASH's reliability prior starts later than MDRR, so a
        # probe at MDRR_START_STEP would only audit a zero regularizer.
        routing_probe_step = max(
            MDRR_START_STEP,
            BRASH_GATE_START_STEP if BRASH_ENABLED and BRASH_GATE_PRIOR_ENABLED else MDRR_START_STEP,
            BVAM_START_STEP if BVAM_ENABLED else MDRR_START_STEP,
        )
        if BRASH_ENABLED and BRASH_GATE_PRIOR_ENABLED:
            routing_probe_step += 1
        model.step = routing_probe_step
        camera_index, _view_id, camera, batch = records[0]
        partner_index = int(partner["rows"][camera_index]["partner_index"])
        partner_record = records[partner_index]
        outputs = model.get_outputs(camera.to(model.device))
        losses = model.get_loss_dict(outputs, _batch_to_device(batch.copy(), model.device), {})
        crcc_weighted = losses.pop("crcc_loss", None)
        base_loss = sum(losses.values())
        model.eval()
        auxiliary_controller = (
            AUX._refresh_controller_rng_preserved(model, records)
            if AUXILIARY_APPEARANCE_ENABLED
            else None
        )
        auxiliary_raw, _auxiliary_info = AUX._module_loss(model, auxiliary_controller)
        cica_controller = (
            _build_cica_controller(
                model,
                [records[int(index)] for index in bank["camera_indices"]],
            )
            if _arm_use_cica(arm)
            else None
        )
        model.train()
        cica_raw, cica_info = _cica_loss(model, cica_controller)
        brash_prior: Optional[Dict[str, Tensor]] = None
        brash_stats: Dict[str, Any] = {"enabled": False}
        if BRASH_ENABLED and BRASH_GATE_PRIOR_ENABLED:
            was_training = model.training
            model.eval()
            brash_prior, brash_stats = _build_brash_reliability_prior(model, [records[int(index)] for index in bank["camera_indices"]])
            if was_training:
                model.train()
        brash_raw, brash_info = model.brash_gate_regularizer(brash_prior, step=routing_probe_step)
        bvam_controller = (
            _build_bvam_controller(model, [records[int(index)] for index in bank["camera_indices"]])
            if BVAM_ENABLED else None
        )
        bvam_raw, bvam_info = _bvam_loss(model, bvam_controller, outputs, camera_index, routing_probe_step)
        mdrr_control, mdrr_info = _mdrr_control(
            model,
            camera,
            outputs,
            _batch_to_device(batch.copy(), model.device),
            partner_record[2].to(model.device),
            _batch_to_device(partner_record[3].copy(), model.device),
        )
        surface_loss = _route_loss(
            model,
            outputs,
            _batch_to_device(batch.copy(), model.device),
            1.0 - mdrr_control["pixel_g"],
        )
        medium_loss = _route_loss(
            model,
            outputs,
            _batch_to_device(batch.copy(), model.device),
            mdrr_control["pixel_g"],
        )
        params, group_indices = _parameter_entries(model)
        base_grads = torch.autograd.grad(base_loss, params, retain_graph=True, allow_unused=True)
        surface_grads = torch.autograd.grad(surface_loss, params, retain_graph=True, allow_unused=True)
        medium_grads = torch.autograd.grad(medium_loss, params, retain_graph=True, allow_unused=True)
        auxiliary_grads = torch.autograd.grad(auxiliary_raw, params, retain_graph=True, allow_unused=True)
        cica_grads = torch.autograd.grad(cica_raw, params, retain_graph=True, allow_unused=True)
        brash_grads = torch.autograd.grad(brash_raw, params, retain_graph=True, allow_unused=True)
        bvam_grads = torch.autograd.grad(bvam_raw, params, retain_graph=True, allow_unused=True)
        combined_info = _combine_gradients(
            params,
            group_indices,
            base_grads,
            surface_grads,
            medium_grads,
            auxiliary_grads,
            cica_grads,
            True,
            brash_grads,
            bvam_grads,
        )
        crcc_norms = _add_crcc_gradients(crcc_weighted, params, group_indices)
        geometry_ids = set(
            group_indices.get("means", [])
            + group_indices.get("scales", [])
            + group_indices.get("quats", [])
            + group_indices.get("opacities", [])
        )
        geometry_diff = max(
            [
                float((params[index].grad - base_grads[index]).detach().float().abs().max().item())
                for index in geometry_ids
                if params[index].grad is not None and base_grads[index] is not None
            ]
            or [0.0]
        )
        direct = {
            "base": _group_norms(base_grads, group_indices),
            "surface_route": _group_norms(surface_grads, group_indices),
            "medium_route": _group_norms(medium_grads, group_indices),
            "auxiliary_appearance": _group_norms(auxiliary_grads, group_indices),
            "CICA": _group_norms(cica_grads, group_indices),
            "BRASH": _group_norms(brash_grads, group_indices),
            "BVAM": _group_norms(bvam_grads, group_indices),
        }
        auxiliary_escape = combined_info["auxiliary_escape_grad_l2"]
        cica_escape = combined_info["cica_escape_grad_l2"]
        brash_non_gate = combined_info["brash_escape_grad_l2"]
        bvam_non_bvam = combined_info["bvam_escape_grad_l2"]
        geometry_ids = set(
            group_indices.get("means", [])
            + group_indices.get("scales", [])
            + group_indices.get("quats", [])
            + group_indices.get("opacities", [])
        )
        medium_ids = set(group_indices.get("medium_mlp", []) + group_indices.get("direction_encoding", []))
        appearance_ids = set(group_indices.get("features_dc", []) + group_indices.get("features_rest", []))
        gate_ids = set(group_indices.get("brash_gate", []))
        brash_geometry_escape = _norm([brash_grads[index] for index in geometry_ids])
        brash_medium_escape = _norm([brash_grads[index] for index in medium_ids])
        brash_appearance_escape = _norm([brash_grads[index] for index in appearance_ids])
        bvam_delta_ids = set(group_indices.get("bvam_delta", []))
        bvam_gate_ids = set(group_indices.get("bvam_gate", []))
        bvam_geometry_escape = _norm([bvam_grads[index] for index in geometry_ids])
        bvam_medium_escape = _norm([bvam_grads[index] for index in medium_ids])
        bvam_features_rest_escape = _norm([bvam_grads[index] for index in group_indices.get("features_rest", [])])
        bvam_features_dc_escape = _norm([bvam_grads[index] for index in group_indices.get("features_dc", [])])
        finite = all(
            value is None or bool(torch.isfinite(value).all().item())
            for gradients in (base_grads, surface_grads, medium_grads, auxiliary_grads, cica_grads, brash_grads, bvam_grads)
            for value in gradients
        )
        surface_total = _norm(surface_grads)
        medium_total = _norm(medium_grads)
        appearance_ids = set(group_indices.get("features_dc", []) + group_indices.get("features_rest", []))
        medium_ids = set(group_indices.get("medium_mlp", []) + group_indices.get("direction_encoding", []))
        surface_appearance = _norm([surface_grads[index] for index in appearance_ids])
        medium_branch = _norm([medium_grads[index] for index in medium_ids])
        effective_surface = [value if index in appearance_ids else None for index, value in enumerate(surface_grads)]
        effective_medium = [value if index in medium_ids else None for index, value in enumerate(medium_grads)]
        effective_surface_escape = _nonzero_group_norm(effective_surface, appearance_ids)
        effective_medium_escape = _nonzero_group_norm(effective_medium, medium_ids)
        route_loss_difference = float((surface_loss + medium_loss - losses["main_loss"]).detach().abs().item())
        audit = {
            "scene": scene,
            "arm": arm,
            "absolute_step_probe": routing_probe_step,
            "optimizer_step_performed": False,
            "finite_direct_gradients": finite,
            "auxiliary_gradient_escape_l2": auxiliary_escape,
            "cica_gradient_escape_l2": cica_escape,
            "brash_gradient_escape_l2": combined_info["brash_escape_grad_l2"],
            "brash_geometry_escape_l2": brash_geometry_escape,
            "brash_medium_escape_l2": brash_medium_escape,
            "brash_appearance_escape_l2": brash_appearance_escape,
            "brash_prior_statistics": brash_stats,
            "brash_regularizer": brash_info,
            "bvam_prior_statistics": bvam_controller["stats"] if bvam_controller is not None else {"enabled": False},
            "bvam_regularizer": bvam_info,
            "bvam_geometry_escape_l2": bvam_geometry_escape,
            "bvam_medium_escape_l2": bvam_medium_escape,
            "bvam_features_rest_escape_l2": bvam_features_rest_escape,
            "bvam_features_dc_escape_l2": bvam_features_dc_escape,
            "combined_geometry_max_abs_difference_from_base": geometry_diff,
            "surface_route_appearance_fraction": surface_appearance / max(surface_total, EPS),
            "medium_route_medium_branch_fraction": medium_branch / max(medium_total, EPS),
            "effective_surface_route_appearance_fraction": _norm(effective_surface) / max(_norm(effective_surface), EPS),
            "effective_medium_route_medium_branch_fraction": _norm(effective_medium) / max(_norm(effective_medium), EPS),
            "effective_surface_route_escape_l2": effective_surface_escape,
            "effective_medium_route_escape_l2": effective_medium_escape,
            "complementary_route_loss_difference_from_base_main": route_loss_difference,
            "cica_valid_gaussians": cica_info.get("valid_gaussians", 0),
            "cica_correction_direction": {
                key: cica_info.get(key, 0.0)
                for key in (
                    "delta_rg_mean",
                    "delta_bg_mean",
                    "delta_rg_p10",
                    "delta_rg_p50",
                    "delta_rg_p90",
                    "delta_bg_p10",
                    "delta_bg_p50",
                    "delta_bg_p90",
                    "delta_rg_positive_fraction",
                    "delta_rg_negative_fraction",
                    "delta_bg_positive_fraction",
                    "delta_bg_negative_fraction",
                    "color_prior_collapse_warning",
                )
            },
            "mdrr_valid_gaussian_count": mdrr_info.get("valid_gaussian_count", 0),
            "mdrr_g_mean": mdrr_info.get("g_mean", 0.0),
            "direct_group_norms": direct,
            "assertions": {
                "TABA_STATISTICS_DETACHED": all(
                    not value.requires_grad for key, value in outputs.items()
                    if key.startswith("taba_") and isinstance(value, Tensor)
                ),
                "GRAC_STATISTICS_DETACHED": all(not value.requires_grad for key, value in outputs.items()
                    if key.startswith("grac_") and isinstance(value, Tensor)),
                "GCRC_STATISTICS_DETACHED": all(not value.requires_grad for key, value in outputs.items()
                    if key.startswith("gcrc_") and isinstance(value, Tensor)),
                "OCMC_PROJECTOR_DETACHED": model._camera_medium_observability_projector is None or not model._camera_medium_observability_projector.requires_grad,
                "AUXILIARY_ONLY_FEATURES_REST": auxiliary_escape == 0.0,
                "CICA_ONLY_FEATURES_DC": cica_escape == 0.0,
                "BRASH_ONLY_GATE": (
                    (direct["BRASH"].get("brash_gate", 0.0) > EPS)
                    if bool(brash_info.get("enabled", False))
                    else _norm(brash_grads) <= EPS
                ),
                "BRASH_NO_GEOMETRY_ESCAPE": brash_geometry_escape <= EPS,
                "BRASH_NO_MEDIUM_ESCAPE": brash_medium_escape <= EPS,
                "BRASH_NO_APPEARANCE_ESCAPE": brash_appearance_escape <= EPS,
                "BRASH_GRADIENT_ONLY_REGISTERED_GATE": (
                    brash_non_gate <= EPS
                    and (not bool(brash_info.get("enabled", False)) or bool(gate_ids))
                ),
                "BVAM_AUX_ONLY_REGISTERED_PARAMS": (
                    bvam_non_bvam <= EPS
                    and (not bool(bvam_info.get("enabled", False)) or bool(bvam_delta_ids or bvam_gate_ids))
                ),
                "BVAM_NO_GEOMETRY_ESCAPE": bvam_geometry_escape <= EPS,
                "BVAM_NO_MEDIUM_ESCAPE": bvam_medium_escape <= EPS,
                "BVAM_NO_FEATURES_REST_ESCAPE": bvam_features_rest_escape <= EPS,
                "BVAM_NO_FEATURES_DC_ESCAPE": bvam_features_dc_escape <= EPS,
                "MODULES_DO_NOT_DIRECTLY_CHANGE_GEOMETRY": geometry_diff <= 1e-12,
                "CICA_ROUTE_MATCHES_SWITCH": (
                    direct["CICA"].get("features_dc", 0.0) > EPS
                    if _arm_use_cica(arm)
                    else _norm(cica_grads) <= EPS
                ),
                "AUXILIARY_ROUTE_MATCHES_SWITCH": (
                    direct["auxiliary_appearance"].get("features_rest", 0.0) > EPS
                    if AUXILIARY_APPEARANCE_ENABLED
                    else _norm(auxiliary_grads) <= EPS
                ),
                "MDRR_SURFACE_ROUTE_ONLY_APPEARANCE": effective_surface_escape == 0.0 and _norm(effective_surface) > EPS,
                "MDRR_MEDIUM_ROUTE_ONLY_MEDIUM": effective_medium_escape == 0.0 and _norm(effective_medium) > EPS,
                "MDRR_PRESERVES_PHOTOMETRIC_LOSS_SCALE": route_loss_difference <= 1e-6,
            },
        }
        audit["GRADIENT_ROUTING_AUDIT"] = bool(
            audit["finite_direct_gradients"]
            and all(audit["assertions"].values())
        )
        if CRCC_EXPERIMENT:
            audit["direct_group_norms"]["CRCC"] = crcc_norms
            audit["assertions"]["CRCC_STATISTICS_DETACHED"] = all(
                not value.requires_grad for key, value in outputs.items()
                if key.startswith("crcc_") and key != "crcc_loss" and isinstance(value, Tensor))
            audit["assertions"]["CRCC_NO_MEDIUM_OR_GEOMETRY_ESCAPE"] = all(
                value == 0 for key, value in crcc_norms.items() if key not in ("features_dc", "features_rest"))
            audit["GRADIENT_ROUTING_AUDIT"] &= all(audit["assertions"].values())
        rows: List[Dict[str, Any]] = []
        for route, values in direct.items():
            for group, norm_value in values.items():
                rows.append({"scene": scene, "arm": arm, "route": route, "parameter_group": group, "gradient_l2": norm_value})
        rows.extend(
            [
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "auxiliary_escape", "gradient_l2": auxiliary_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "cica_escape", "gradient_l2": cica_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "brash_escape", "gradient_l2": combined_info["brash_escape_grad_l2"]},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "brash_geometry_escape", "gradient_l2": brash_geometry_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "brash_medium_escape", "gradient_l2": brash_medium_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "brash_appearance_escape", "gradient_l2": brash_appearance_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "bvam_escape", "gradient_l2": combined_info["bvam_escape_grad_l2"]},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "bvam_geometry_escape", "gradient_l2": bvam_geometry_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "bvam_medium_escape", "gradient_l2": bvam_medium_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "bvam_features_rest_escape", "gradient_l2": bvam_features_rest_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "bvam_features_dc_escape", "gradient_l2": bvam_features_dc_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "geometry_max_abs_difference", "gradient_l2": geometry_diff},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "effective_surface_escape", "gradient_l2": effective_surface_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "effective_medium_escape", "gradient_l2": effective_medium_escape},
            ]
        )
        _write_csv(output_dir / "gradient_routing_audit.csv", rows)
        _write_json(output_dir / "gradient_routing_audit.json", audit)
        if not audit["GRADIENT_ROUTING_AUDIT"]:
            raise RuntimeError("gradient routing audit failed")
        return audit
    finally:
        model.step = original_step
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        else:
            model.eval()


def _topology_row(arm: str, step: int, model: Any, cumulative: Mapping[str, int]) -> Dict[str, Any]:
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacities.detach()).reshape(-1).float().cpu()
    return {
        "arm": arm,
        "absolute_step": int(step),
        "gaussian_count": int(model.num_points),
        "opacity_mean": float(opacity.mean().item()),
        "opacity_median": float(opacity.median().item()),
        "split_count_cumulative": int(cumulative["split"]),
        "duplicate_count_cumulative": int(cumulative["duplicate"]),
        "prune_count_cumulative": int(cumulative["prune"]),
        "opacity_reset_count_cumulative": int(cumulative["reset"]),
    }


def _opacity_cap_state(model: Any, candidates: Tensor, rho: float = 0.0) -> Dict[str, Tensor]:
    """Create immutable reference opacity and optical-thickness caps."""
    opacity = torch.sigmoid(model.opacities.detach().float()).reshape(-1)
    tau = -torch.log1p(-opacity.clamp(max=1.0 - 1e-7))
    candidates = candidates.to(device=opacity.device, dtype=torch.bool).reshape(-1)
    if candidates.numel() != opacity.numel():
        raise ValueError("candidate mask does not match Gaussian population")
    return {"reference_opacity": opacity.clone(), "reference_tau": tau.clone(),
            "candidate_mask": candidates.clone(), "rho": torch.tensor(float(rho), device=opacity.device)}


@torch.no_grad()
def _project_opacity_(model: Any, optimizers: Any, state: Mapping[str, Tensor], rho: float) -> Dict[str, Any]:
    """Project stored logits to shared/reference and candidate optical caps."""
    ref_tau = state["reference_tau"].to(model.opacities.device)
    candidates = state["candidate_mask"].to(model.opacities.device)
    if ref_tau.numel() != model.opacities.shape[0]:
        raise RuntimeError("opacity cap population changed; synchronize permanent Gaussian IDs before projection")
    cap_tau = ref_tau.clone()
    cap_tau[candidates] *= 1.0 - float(rho)
    cap = -torch.expm1(-cap_tau.clamp_min(0.0))
    cap_logit = torch.logit(cap.clamp(min=1e-8, max=1.0 - 1e-7))
    before = model.opacities.detach().reshape(-1)
    hit = before > cap_logit
    if hit.any():
        model.opacities.reshape(-1).copy_(torch.minimum(before, cap_logit))
        optimizer = optimizers.optimizers.get("opacities")
        if optimizer is not None:
            adam = optimizer.state.get(model.opacities, {})
            moment = adam.get("exp_avg")
            if isinstance(moment, Tensor):
                # Adam subtracts exp_avg; a negative moment pushes through the upper bound.
                moment.reshape(-1)[hit] = moment.reshape(-1)[hit].clamp_min(0.0)
    return {"rho": float(rho), "projected_count": int(hit.sum().item()),
            "projected_fraction": float(hit.float().mean().item()),
            "projected_reference_tau": float(ref_tau[hit].sum().item()) if hit.any() else 0.0,
            "max_opacity_violation": float((torch.sigmoid(model.opacities.detach().reshape(-1)) - cap).clamp_min(0).max().item())}


def _medium_optimizer_groups(optimizers: Any) -> set[str]:
    names = set(optimizers.optimizers)
    selected = names.intersection({"medium_mlp", "direction_encoding", "gmvc_bounded_medium"})
    if "medium_mlp" not in selected:
        raise RuntimeError(f"M1 optimizer group is absent: available={sorted(names)}")
    return selected


TRAINING_RUNTIME_ATTRIBUTES = (
    "step", "xys_grad_norm", "vis_counts", "depths_accum", "max_2Dsize",
    "last_size", "last_fx", "last_fy", "background_color",
    "_appearance_lr_scale_applied", "_medium_hold_last_active",
    "_refinement_last_event", "_refinement_guidance_hardness",
    "_refinement_guidance_brightness", "_refinement_budget_schedule",
)


def _training_runtime_state(model: Any) -> Dict[str, Any]:
    return {name: (getattr(model, name).detach().cpu().clone()
                   if isinstance(getattr(model, name), Tensor) else copy.deepcopy(getattr(model, name)))
            for name in TRAINING_RUNTIME_ATTRIBUTES if hasattr(model, name)}


def _restore_training_checkpoint(branch: Any, payload: Mapping[str, Any], strict_runtime: bool) -> Dict[str, Any]:
    model = branch.pipeline.model
    if strict_runtime and ("rng_state" not in payload or "training_runtime_state" not in payload):
        raise RuntimeError("strict fork needs actual RNG and refinement state, not just their hashes")
    model.load_state_dict(payload["model"], strict=True)
    if RESPONSIBILITY_ENABLED:
        FORMAL._install_condition(model, "C0", payload["ocmc_bundle"], None)
    model.step = int(payload["absolute_step"])
    # load_state_dict replaces Gaussian Parameter objects. Optimizers must be
    # rebuilt against these objects before restoring moments and schedulers.
    branch.optimizers = Optimizers(_optimizer_groups(branch.config, model), model.get_param_groups())
    for group, optimizer in branch.optimizers.optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][group])
    for group, scheduler in branch.optimizers.schedulers.items():
        scheduler.load_state_dict(payload["schedulers"][group])
    branch.scalers = dict(payload.get("scalers", {}))
    for name, value in payload.get("training_runtime_state", {}).items():
        if name not in TRAINING_RUNTIME_ATTRIBUTES:
            raise RuntimeError(f"unknown training runtime attribute: {name}")
        setattr(model, name, value.to(model.device) if isinstance(value, Tensor) else copy.deepcopy(value))
    if "rng_state" in payload:
        _set_rng_state(payload["rng_state"])
    groups = model.get_param_groups()
    binding = {name: [id(p) for pg in opt.param_groups for p in pg["params"]] == [id(p) for p in groups[name]]
               for name, opt in branch.optimizers.optimizers.items()}
    hashes = {
        "model": _model_state_hash(model),
        "optimizers": _optimizer_hash(branch.optimizers),
        "schedulers": _scheduler_hash(branch.optimizers),
        "scalers": _hash_object(branch.scalers),
        "runtime": _hash_object(_training_runtime_state(model)),
        "rng": _rng_manifest(_rng_state()),
        "ocmc_projector": _hash_optional_tensor(model._camera_medium_observability_projector),
    }
    expected = {
        "model": _hash_object(payload["model"]),
        "optimizers": _hash_object({k: _canonical_optimizer_state(v) for k, v in payload["optimizers"].items()}),
        "schedulers": _hash_object(payload["schedulers"]),
        "scalers": _hash_object(payload.get("scalers", {})),
    }
    if "training_runtime_state" in payload:
        expected["runtime"] = _hash_object(payload["training_runtime_state"])
    if "rng_state" in payload:
        expected["rng"] = _rng_manifest(payload["rng_state"])
    checks = {k: hashes[k] == v for k, v in expected.items()}
    if not all(binding.values()) or not all(checks.values()):
        raise RuntimeError(f"training checkpoint restore failed: {checks}, binding={binding}")
    return {"all_pass": True, "absolute_step": model.step, "hashes": hashes,
            "state_matches": checks, "optimizer_parameter_binding": binding}


def _save_checkpoint(branch: Any, scene: str, arm: str, step: int, output_dir: Path, source_payload: Mapping[str, Any], mdrr_hash: str, cica_hash: str, auxiliary_hash: str, lambda_cica: Optional[float], brash_stats: Mapping[str, Any], taba_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> Path:
    path = output_dir / "checkpoints" / f"step-{step:09d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": EXPERIMENT,
            "scene": scene,
            "arm": arm,
            "absolute_step": int(step),
            "model": branch.pipeline.model.state_dict(),
            "optimizers": {key: value.state_dict() for key, value in branch.optimizers.optimizers.items()},
            "schedulers": {key: value.state_dict() for key, value in branch.optimizers.schedulers.items()},
            "scalers": dict(branch.scalers),
            "rng_manifest": _rng_manifest(_rng_state()),
            "rng_state": _rng_state(),
            "training_runtime_state": _training_runtime_state(branch.pipeline.model),
            "responsibility_state": branch.responsibility_controller.to_cpu(branch.responsibility_controller.state) if getattr(branch, "responsibility_controller", None) is not None else None,
            "cica_controller": _cpu_object(getattr(branch, "cica_controller", None)),
            "ocmc_bundle": source_payload["ocmc_bundle"],
            "raoc_state": None,
            "metadata": {
                "matched_updates": int(step if RAW_DATA_ONLY else step - START_STEP),
                "mdrr_enabled": _arm_use_mdrr(arm) and step >= MDRR_START_STEP,
                "cica_enabled": _arm_use_cica(arm) and step >= CICA_START_STEP,
                "cica_derivative_mode": CICA_DERIVATIVE_MODE,
                "renderer_backward_mode": RENDERER_BACKWARD_MODE,
                "cica_ramp_steps": CICA_RAMP_STEPS,
                "cica_minimum_support": CICA_MIN_SUPPORT,
                "ocmc_enabled": OCMC_ENABLED,
                "ocmc_strength": OCMC_STRENGTH,
                "ocmc_strength_start": OCMC_STRENGTH_START,
                "ocmc_strength_ramp_steps": OCMC_STRENGTH_RAMP_STEPS,
                "mdrr_control_hash": mdrr_hash,
                "cica_target_hash": cica_hash,
                "auxiliary_appearance_target_hash": auxiliary_hash,
                "auxiliary_appearance_enabled": AUXILIARY_APPEARANCE_ENABLED,
                "auxiliary_appearance_strength": AUXILIARY_APPEARANCE_STRENGTH,
                "resolved_lambda_CICA": lambda_cica,
        "ocmc_frozen": True,
        "intrinsic_color_parameterization": ACTIVE_PARAMETERIZATION,
                "headroom_config": {k: v for k, v in _locked_config(branch.pipeline.model).items()
                                    if "headroom" in k},
                "raoc_enabled": False,
                "normal_topology_enabled": True,
                "taba_diagnostic_rows": list(taba_rows or []),
                "taba_config": {key: value for key, value in _locked_config(branch.pipeline.model).items() if key.startswith("taba_")},
                "grac_config": grac_config_values(branch.pipeline.model.config),
                "gcrc_config": gcrc_config_values(branch.pipeline.model.config),
                "crcc_config": crcc_config_values(branch.pipeline.model.config),
                "brash_enabled": bool(BRASH_ENABLED),
                "brash_gate_mode": BRASH_GATE_MODE,
                "brash_gate_prior_enabled": bool(BRASH_GATE_PRIOR_ENABLED),
                "brash_gate_prior_weight": float(BRASH_GATE_PRIOR_WEIGHT),
                "brash_reliability_stats": dict(brash_stats),
                "bvam_enabled": bool(BVAM_ENABLED),
                "bvam_num_experts": int(BVAM_NUM_EXPERTS),
                "bvam_delta_scale": float(BVAM_DELTA_SCALE),
                "bvam_gate_degree": int(BVAM_GATE_DEGREE),
                "bvam_gate_temperature": float(BVAM_GATE_TEMPERATURE),
            },
        },
        path,
    )
    return path


def _train_arm(
    scene: str,
    arm: str,
    output_dir: Path,
    sequence: Sequence[Mapping[str, Any]],
    partner: Mapping[str, Any],
    bank: Mapping[str, Any],
    training_rng: Mapping[str, Any],
    max_step: int = FINAL_STEP,
    resume_checkpoint: Optional[Path] = None,
    fork_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    branch = None
    started = time.perf_counter()
    try:
        branch = _new_branch(scene)
        source_payload = _load_source(branch, scene)
        resume_step = 0 if RAW_DATA_ONLY else START_STEP
        resume_metadata: Mapping[str, Any] = {}
        responsibility_resume = None
        cica_resume = None
        restore_path = fork_checkpoint or resume_checkpoint
        if restore_path is not None:
            resume_payload = torch.load(restore_path, map_location="cpu", weights_only=False)
            if fork_checkpoint is None and resume_payload.get("arm") != arm:
                raise RuntimeError(f"resume checkpoint arm mismatch: {restore_path}")
            resume_step = int(resume_payload.get("absolute_step", -1))
            if resume_step <= (0 if RAW_DATA_ONLY else START_STEP) or resume_step >= int(max_step):
                raise RuntimeError(f"resume checkpoint step is not resumable: {restore_path}")
            resume_metadata = dict(resume_payload.get("metadata", {}))
            responsibility_resume = resume_payload.get("responsibility_state")
            cica_resume = resume_payload.get("cica_controller")
            if fork_checkpoint is not None:
                if (resume_step >= CICA_START_STEP or resume_metadata.get("cica_enabled")
                        or resume_metadata.get("mdrr_enabled") or resume_payload.get("scene") != scene
                        or resume_metadata.get("renderer_backward_mode") != RENDERER_BACKWARD_MODE
                        or resume_metadata.get("intrinsic_color_parameterization") != ACTIVE_PARAMETERIZATION):
                    raise RuntimeError("fork requires a matching renderer/scene/appearance before CICA activates")
            restore_audit = _restore_training_checkpoint(branch, resume_payload, fork_checkpoint is not None)
            restore_audit.update(checkpoint=str(restore_path), checkpoint_sha256=_sha256(restore_path),
                                 fork=fork_checkpoint is not None, source_arm=resume_payload["arm"], target_arm=arm)
            _write_json(output_dir / "resume_state_audit.json", restore_audit)
            if "rng_state" not in resume_payload:
                _set_rng_state(training_rng)
            del resume_payload
        else:
            _set_rng_state(training_rng)
        model = branch.pipeline.model
        dm = branch.pipeline.datamanager
        cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
        cached = dm.cached_train
        records = _train_records(branch)
        by_index = {index: record for index, record in enumerate(records)}
        partner_by_index = {int(row["primary_index"]): int(row["partner_index"]) for row in partner["rows"]}
        bank_indices = [int(x) for x in bank["camera_indices"]]
        bank_records = [records[index] for index in bank_indices]
        brash_prior: Optional[Dict[str, Tensor]] = None
        brash_stats: Dict[str, Any] = {"enabled": False}
        if BRASH_ENABLED and BRASH_GATE_PRIOR_ENABLED:
            was_training = model.training
            model.eval()
            brash_prior, brash_stats = _build_brash_reliability_prior(model, bank_records)
            if was_training:
                model.train()
            module_rows = [{"arm": arm, "absolute_step": START_STEP, "module": "BRASH", "event": "reliability_refresh", **brash_stats}]
        else:
            module_rows = []
        bvam_controller: Optional[Dict[str, Any]] = None
        if BVAM_ENABLED:
            was_training = model.training
            model.eval()
            bvam_controller = _build_bvam_controller(model, bank_records)
            model.set_bvam_valid_mask(bvam_controller["valid"])
            if was_training:
                model.train()
            module_rows.append({"arm": arm, "absolute_step": START_STEP, "module": "BVAM", "event": "pseudo_target_refresh", **bvam_controller["stats"]})
        use_mdrr = _arm_use_mdrr(arm)
        use_cica = _arm_use_cica(arm)
        controller: Optional[Dict[str, Tensor]] = None
        auxiliary_controller: Optional[Dict[str, Any]] = None
        cica_controller: Optional[Dict[str, Tensor]] = (
            {k: v.to(model.device) if isinstance(v, Tensor) else v for k,v in cica_resume.items()}
            if cica_resume is not None else None)
        lambda_cica: Optional[float] = (
            float(resume_metadata["resolved_lambda_CICA"])
            if resume_metadata.get("resolved_lambda_CICA") is not None
            else None
        )
        mdrr_control_hash = ""
        mdrr_info: Dict[str, Any] = {"enabled": False}
        rows: List[Dict[str, Any]] = []
        module_rows = list(module_rows)
        topology_rows: List[Dict[str, Any]] = [_topology_row(arm, resume_step, model, {"split": 0, "duplicate": 0, "prune": 0, "reset": 0})]
        event_rows: List[Dict[str, Any]] = []
        checkpoints: List[Dict[str, Any]] = []
        cumulative = {"split": 0, "duplicate": 0, "prune": 0, "reset": 0}
        projector_hash_start = _hash_optional_tensor(model._camera_medium_observability_projector)
        snapshot_steps = _active_snapshots(max_step)
        sequence = [
            row
            for row in sequence
            if resume_step < int(row["absolute_step"]) <= int(max_step)
        ]
        if not sequence:
            raise RuntimeError("empty continuation sequence")
        if AUXILIARY_APPEARANCE_ENABLED:
            model.eval()
            auxiliary_controller = AUX._refresh_controller_rng_preserved(model, records)
            model.train()
            module_rows.append({"arm": arm, "absolute_step": START_STEP, "module": "AUXILIARY_APPEARANCE", "event": "refresh", "reason": "common_start", "active_gaussians": int(auxiliary_controller["active_gaussians"]), "target_hash": _hash_tensor(auxiliary_controller["targets"])})
        responsibility = None
        if RESPONSIBILITY_ENABLED:
            from scripts.experiments.native_responsibility_controller import ResponsibilityController
            responsibility = ResponsibilityController(sys.modules[__name__], branch, records, arm, output_dir,
                smoke=RESPONSIBILITY_SMOKE, state=responsibility_resume)
            branch.responsibility_controller = responsibility
            responsibility.project()
        if RAW_DATA_ONLY and resume_step == 0:
            _save_checkpoint(branch, scene, arm, 0, output_dir, source_payload, "", "", "", None, brash_stats)
        for sequence_row in sequence:
            step = int(sequence_row["absolute_step"])
            phase = responsibility.before(step) if responsibility is not None else "native"
            camera_index = responsibility.camera(step, int(sequence_row["camera_index"])) if responsibility is not None else int(sequence_row["camera_index"])
            camera = cameras[camera_index : camera_index + 1]
            batch = _batch_to_device(cached[camera_index].copy(), model.device)
            branch.pipeline.train()
            model.train()
            model.step_cb(step)
            model.aopt_before_train_iteration(branch.optimizers, step)
            model.medium_hold_before_train_iteration(branch.optimizers, step)
            branch.optimizers.zero_grad_all()

            if use_cica and phase != "m1_only" and step >= CICA_START_STEP and (step in CICA_REFRESH_STEPS or cica_controller is None or int(cica_controller["population"].item()) != int(model.num_points)):
                refresh_start = time.perf_counter()
                was_training = model.training
                model.eval()
                cica_controller = _build_cica_controller(model, bank_records)
                if was_training:
                    model.train()
                module_rows.append({"arm": arm, "absolute_step": step, "module": "CICA", "event": "refresh", "refresh_seconds": time.perf_counter() - refresh_start, "population": int(model.num_points), "valid_gaussians": int(cica_controller["valid_count"].item()), "target_hash": _hash_tensor(cica_controller["target_log_chroma"]), "bank_hash": bank["bank_sha256"]})

            started_iteration = time.perf_counter()
            outputs = model.get_outputs(camera)
            losses = model.get_loss_dict(outputs, batch, {})
            crcc_weighted = losses.pop("crcc_loss", None)
            base_loss = sum(losses.values())
            if not bool(torch.isfinite(base_loss).item()) or not FORMAL._finite_outputs(outputs):
                raise RuntimeError(f"non-finite base state at {scene}/{arm}/{step}")
            if TABA_EXPERIMENT:
                if bool((outputs["J_gaussian_raw"].detach() > 1.0).any()) or bool((outputs["gaussian_view_rgb"] > 1.0).any()):
                    raise RuntimeError(f"bounded appearance violation at {scene}/{arm}/{step}")

            inner_info = {}
            if phase == "m1_only":
                inner_loss, inner_info = responsibility.inner_loss(outputs, camera_index)
                base_loss = base_loss + inner_loss
            auxiliary_raw, auxiliary_info = AUX._module_loss(model, auxiliary_controller if AUXILIARY_APPEARANCE_ENABLED else None)
            auxiliary_weighted = float(AUXILIARY_APPEARANCE_STRENGTH) * auxiliary_raw
            cica_active = use_cica and step >= CICA_START_STEP and phase != "m1_only"
            cica_progress = 1.0
            if cica_active and CICA_RAMP_STEPS > 0:
                cica_progress = min(max((float(step) - float(CICA_START_STEP)) / float(CICA_RAMP_STEPS), 0.0), 1.0)
            cica_raw, cica_info = _cica_loss(model, cica_controller if cica_active else None)
            cica_weighted = cica_raw if lambda_cica is None else float(lambda_cica) * cica_progress * cica_raw
            brash_regularizer, brash_info = model.brash_gate_regularizer(brash_prior, step=step)
            bvam_regularizer, bvam_info = _bvam_loss(model, bvam_controller, outputs, camera_index, step)
            active_mdrr = False
            gmap = None
            if use_mdrr and step >= MDRR_START_STEP:
                partner_index = partner_by_index[camera_index]
                partner_record = by_index[partner_index]
                control, mdrr_info = _mdrr_control(model, camera, outputs, batch, partner_record[2].to(model.device), _batch_to_device(cached[partner_index].copy(), model.device))
                gmap = control["pixel_g"].detach()
                mdrr_control_hash = _hash_tensor(gmap)
                module_rows.append({"arm": arm, "absolute_step": step, "module": "MDRR", "event": "control_update", "primary_camera_index": camera_index, "partner_camera_index": partner_index, **mdrr_info, "refresh_rule": "every active primary update from the current model state"})
                active_mdrr = True

            if cica_active and lambda_cica is None:
                base_probe = torch.autograd.grad(base_loss, [model.features_dc], retain_graph=True, allow_unused=True)[0]
                cica_probe = torch.autograd.grad(cica_raw, [model.features_dc], retain_graph=True, allow_unused=True)[0]
                base_norm = float(base_probe.detach().float().norm().item()) if base_probe is not None else 0.0
                cica_norm = float(cica_probe.detach().float().norm().item()) if cica_probe is not None else 0.0
                if cica_norm <= EPS or not math.isfinite(cica_norm):
                    raise RuntimeError("CICA calibration has zero/non-finite features_dc gradient")
                lambda_cica = CICA_GRADIENT_FRACTION * max(base_norm, EPS) / cica_norm
                cica_weighted = float(lambda_cica) * cica_progress * cica_raw
                module_rows.append({"arm": arm, "absolute_step": step, "module": "CICA", "event": "calibration", "resolved_lambda_CICA": lambda_cica, "photometric_features_dc_grad_l2": base_norm, "raw_CICA_features_dc_grad_l2": cica_norm, "target_gradient_fraction": CICA_GRADIENT_FRACTION})

            if active_mdrr:
                surface_weight = (1.0 - gmap).clamp(0.0, 1.0)
                medium_weight = gmap.clamp(0.0, 1.0)
                surface_loss = _route_loss(model, outputs, batch, surface_weight)
                medium_loss = _route_loss(model, outputs, batch, medium_weight)
                params, group_indices = _parameter_entries(model)
                base_grads = torch.autograd.grad(base_loss, params, retain_graph=True, allow_unused=True)
                surface_grads = torch.autograd.grad(surface_loss, params, retain_graph=True, allow_unused=True)
                medium_grads = torch.autograd.grad(medium_loss, params, retain_graph=True, allow_unused=True)
                auxiliary_grads = torch.autograd.grad(auxiliary_weighted, params, retain_graph=True, allow_unused=True)
                cica_grads = torch.autograd.grad(cica_weighted, params, retain_graph=True, allow_unused=True) if cica_active else [None] * len(params)
                brash_grads = torch.autograd.grad(brash_regularizer, params, retain_graph=True, allow_unused=True)
                bvam_grads = torch.autograd.grad(bvam_regularizer, params, retain_graph=True, allow_unused=True)
                grad_info = _combine_gradients(params, group_indices, base_grads, surface_grads, medium_grads, auxiliary_grads, cica_grads, True, brash_grads, bvam_grads)
            else:
                params, group_indices = _parameter_entries(model)
                base_grads = torch.autograd.grad(base_loss, params, retain_graph=True, allow_unused=True)
                auxiliary_grads = torch.autograd.grad(auxiliary_weighted, params, retain_graph=True, allow_unused=True)
                cica_grads = torch.autograd.grad(cica_weighted, params, retain_graph=True, allow_unused=True) if cica_active else [None] * len(params)
                brash_grads = torch.autograd.grad(brash_regularizer, params, retain_graph=True, allow_unused=True)
                bvam_grads = torch.autograd.grad(bvam_regularizer, params, retain_graph=True, allow_unused=True)
                grad_info = _combine_gradients(params, group_indices, base_grads, base_grads, base_grads, auxiliary_grads, cica_grads, False, brash_grads, bvam_grads)

            geometry_hook = globals().get("GEOMETRY_GRADIENT_HOOK")
            if geometry_hook is not None:
                geometry_hook(branch, outputs, base_loss, step, output_dir)

            extra_info: Dict[str, Any] = {}
            extra_hook = globals().get("EXTRA_GRADIENT_HOOK")
            if extra_hook is not None:
                extra_info = extra_hook(
                    branch,
                    outputs,
                    batch,
                    base_loss,
                    step,
                    output_dir,
                    params,
                    group_indices,
                    camera_index,
                    sequence_row,
                ) or {}
                if not isinstance(extra_info, Mapping):
                    raise RuntimeError("EXTRA_GRADIENT_HOOK must return a mapping")

            crcc_norms = _add_crcc_gradients(crcc_weighted, params, group_indices)
            if CRCC_EXPERIMENT:
                grad_info.update({"crcc_"+key+"_grad_l2": value for key,value in crcc_norms.items()})
            total_loss_value = float((base_loss + auxiliary_weighted + cica_weighted + brash_regularizer + bvam_regularizer
                                      + (crcc_weighted if crcc_weighted is not None else 0)).detach().item())
            groups = set(branch.optimizers.optimizers)
            if phase == "m1_only":
                groups = _medium_optimizer_groups(branch.optimizers)
            elif responsibility is not None and step > responsibility.reference_step:
                groups -= {"means", "scales", "quats"}
            frozen_before = ({name: [p.detach().clone() for pg in opt.param_groups for p in pg['params']]
                              for name, opt in branch.optimizers.optimizers.items() if name not in groups}
                             if RESPONSIBILITY_SMOKE else {})
            for name, optimizer in branch.optimizers.optimizers.items():
                if name not in groups:
                    for pg in optimizer.param_groups:
                        for parameter in pg["params"]:
                            parameter.grad = None
                    continue
                max_norm = branch.optimizers.config[name]["optimizer"].max_norm
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(branch.optimizers.parameters[name], max_norm)
                optimizer.step()
                if responsibility is not None:
                    calls = responsibility.state["optimizer_calls"]
                    calls[name] = calls.get(name, 0) + 1
            branch.optimizers.scheduler_step_all(step)
            model.aopt_after_train_iteration(step)
            model.medium_hold_after_train_iteration(branch.optimizers, step)
            model.after_train(step=step)
            for name, copies in frozen_before.items():
                current = [p for pg in branch.optimizers.optimizers[name].param_groups for p in pg['params']]
                if not all(torch.equal(a, b) for a, b in zip(copies, current)):
                    raise RuntimeError(f'frozen group changed: {scene}/{arm}/{step}/{name}')
            event: Dict[str, Any]
            if step % int(model.config.refine_every) == 0 and (responsibility is None or step < responsibility.reference_step):
                model.refinement_after(branch.optimizers, step=step)
                event = dict(getattr(model, "_refinement_last_event", {}))
                if event.get("refinement_called"):
                    event_rows.append({"arm": arm, "absolute_step": step, **event})
                    cumulative["split"] += int(event.get("K_split", 0))
                    cumulative["duplicate"] += int(event.get("K_duplicate", 0))
                    cumulative["prune"] += int(event.get("N_pruned", 0))
                    cumulative["reset"] += int(bool(event.get("opacity_reset", False)))
                    if use_cica and step >= CICA_START_STEP and int(event.get("N_before", model.num_points)) != int(event.get("N_after", model.num_points)):
                        was_training = model.training
                        model.eval()
                        cica_controller = _build_cica_controller(model, bank_records)
                        if was_training:
                            model.train()
                        module_rows.append({"arm": arm, "absolute_step": step, "module": "CICA", "event": "topology_refresh", "population": int(model.num_points), "valid_gaussians": int(cica_controller["valid_count"].item()), "target_hash": _hash_tensor(cica_controller["target_log_chroma"]), "bank_hash": bank["bank_sha256"]})
                    if BRASH_ENABLED and BRASH_GATE_PRIOR_ENABLED:
                        was_training = model.training
                        model.eval()
                        brash_prior, brash_stats = _build_brash_reliability_prior(model, bank_records)
                        if was_training:
                            model.train()
                        module_rows.append({"arm": arm, "absolute_step": step, "module": "BRASH", "event": "reliability_refresh", **brash_stats})
                    if BVAM_ENABLED:
                        was_training = model.training
                        model.eval()
                        bvam_controller = _build_bvam_controller(model, bank_records)
                        model.set_bvam_valid_mask(bvam_controller["valid"])
                        if was_training:
                            model.train()
                        module_rows.append({"arm": arm, "absolute_step": step, "module": "BVAM", "event": "pseudo_target_refresh", **bvam_controller["stats"]})
                    if AUXILIARY_APPEARANCE_ENABLED:
                        was_training = model.training
                        model.eval()
                        auxiliary_controller = AUX._refresh_controller_rng_preserved(model, records)
                        if was_training:
                            model.train()
                        module_rows.append({"arm": arm, "absolute_step": step, "module": "AUXILIARY_APPEARANCE", "event": "refresh", "reason": "topology_change", "active_gaussians": int(auxiliary_controller["active_gaussians"]), "target_hash": _hash_tensor(auxiliary_controller["targets"])})
            else:
                event = {"refinement_called": False, "N_after": int(model.num_points)}

            projection_info = responsibility.after(step) if responsibility is not None else {}
            branch.cica_controller = cica_controller
            if responsibility is not None:
                with (output_dir / "responsibility_iterations.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(step=step, phase=phase, camera_index=camera_index,
                        event=responsibility.state["event_index"], rho=responsibility.state["rho_effective"],
                        inner=inner_info, projection=projection_info)) + "\n")
            if step % 500 == 0 or step in snapshot_steps or step == (1 if RAW_DATA_ONLY else START_STEP + 1):
                pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
                target = _gt(model, batch, outputs["background"]).detach().float().clamp(0.0, 1.0)
                metrics = MIC._metric_images(model, pred, target)
                rows.append({
                    "arm": arm,
                    "absolute_step": step,
                    "camera_index": camera_index,
                    "camera_name": sequence_row["camera_name"],
                    "L_total": total_loss_value,
                    "headroom_alpha": float(outputs["headroom_alpha"].detach()) if "headroom_alpha" in outputs else None,
                    "L_RGB": float(base_loss.detach().item()),
                    "L_AUXILIARY_APPEARANCE": float(auxiliary_weighted.detach().item()),
                    "L_CICA_raw": float(cica_raw.detach().item()),
                    "L_CICA_weighted": float(cica_weighted.detach().item()),
                    "L_BRASH_gate_prior": float(brash_regularizer.detach().item()),
                    "L_BVAM_auxiliary": float(bvam_regularizer.detach().item()),
                    "BRASH_enabled": bool(BRASH_ENABLED),
                    "BRASH_gate_mode": BRASH_GATE_MODE,
                    "BRASH_gate_prior_enabled": bool(BRASH_GATE_PRIOR_ENABLED),
                    "BRASH_gate_regularizer_progress": brash_info.get("progress", 0.0),
                    "BRASH_gate_mean": brash_info.get("gate_mean", 0.0),
                    "BRASH_gate_p10": brash_info.get("gate_p10", 0.0),
                    "BRASH_gate_p50": brash_info.get("gate_p50", 0.0),
                    "BRASH_gate_p90": brash_info.get("gate_p90", 0.0),
                    "BRASH_gate_near_zero_fraction": brash_info.get("gate_near_zero_fraction", 0.0),
                    "BRASH_gate_near_one_fraction": brash_info.get("gate_near_one_fraction", 0.0),
                    "BRASH_base_logit_abs_mean": float(outputs.get("brash_base_logits", outputs["gaussian_view_rgb"]).detach().float().abs().mean().item()),
                    "BRASH_residual_logit_abs_mean": float(outputs.get("brash_residual_logits", outputs["gaussian_view_rgb"]).detach().float().abs().mean().item()),
                    "BRASH_sigmoid_saturation_fraction": float(((outputs["gaussian_view_rgb"].detach().float() < 0.01) | (outputs["gaussian_view_rgb"].detach().float() > 0.99)).float().mean().item()),
                    "BVAM_enabled": bool(BVAM_ENABLED),
                    "BVAM_expert_usage_1": bvam_info.get("expert_usage_1", 0.0),
                    "BVAM_expert_usage_2": bvam_info.get("expert_usage_2", 0.0),
                    "BVAM_expert_entropy": bvam_info.get("expert_entropy", 0.0),
                    "BVAM_expert_collapse_fraction": bvam_info.get("expert_collapse_fraction", 0.0),
                    "BVAM_delta_norm_mean": bvam_info.get("delta_norm_mean", 0.0),
                    "BVAM_delta_norm_p90": bvam_info.get("delta_norm_p90", 0.0),
                    "BVAM_pseudo_valid_fraction": bvam_info.get("pseudo_valid_fraction", 0.0),
                    "resolved_lambda_CICA": lambda_cica if lambda_cica is not None else 0.0,
                    "MDRR_enabled": active_mdrr,
                    "CICA_enabled": bool(cica_active),
                    "CICA_progress": float(cica_progress if cica_active else 0.0),
                    "MDRR_g_mean": mdrr_info.get("g_mean", 0.0) if active_mdrr else 0.0,
                    "MDRR_g_p50": mdrr_info.get("g_p50", 0.0) if active_mdrr else 0.0,
                    "MDRR_q_mean": mdrr_info.get("q_mean", 0.0) if active_mdrr else 0.0,
                    "MDRR_g_p10": mdrr_info.get("g_p10", 0.0) if active_mdrr else 0.0,
                    "MDRR_g_p90": mdrr_info.get("g_p90", 0.0) if active_mdrr else 0.0,
                    "CICA_valid_fraction": cica_info.get("valid_fraction", 0.0),
                    "CICA_median_abs_delta_chi": cica_info.get("median_abs_delta_chi", 0.0),
                    "CICA_delta_R_over_G": cica_info.get("delta_rg_mean", 0.0),
                    "CICA_delta_B_over_G": cica_info.get("delta_bg_mean", 0.0),
                    "CICA_delta_R_over_G_p10": cica_info.get("delta_rg_p10", 0.0),
                    "CICA_delta_R_over_G_p50": cica_info.get("delta_rg_p50", 0.0),
                    "CICA_delta_R_over_G_p90": cica_info.get("delta_rg_p90", 0.0),
                    "CICA_delta_B_over_G_p10": cica_info.get("delta_bg_p10", 0.0),
                    "CICA_delta_B_over_G_p50": cica_info.get("delta_bg_p50", 0.0),
                    "CICA_delta_B_over_G_p90": cica_info.get("delta_bg_p90", 0.0),
                    "CICA_delta_R_over_G_positive_fraction": cica_info.get("delta_rg_positive_fraction", 0.0),
                    "CICA_delta_R_over_G_negative_fraction": cica_info.get("delta_rg_negative_fraction", 0.0),
                    "CICA_delta_B_over_G_positive_fraction": cica_info.get("delta_bg_positive_fraction", 0.0),
                    "CICA_delta_B_over_G_negative_fraction": cica_info.get("delta_bg_negative_fraction", 0.0),
                    "CICA_color_prior_collapse_warning": cica_info.get("color_prior_collapse_warning", False),
                    "SH_DC_energy_mean": float(model.features_dc.detach().float().square().mean().item()),
                    "SH_nonDC_energy_mean": float(model.features_rest.detach().float().square().mean().item()),
                    "medium_attn_abs_mean": _output_mean_abs(outputs, "medium_attn"),
                    "medium_bs_abs_mean": _output_mean_abs(outputs, "medium_bs"),
                    "ocmc_projected_raw_abs_mean": _output_mean_abs(outputs, "camera_medium_delta_projected_raw"),
                    "gaussian_count": int(model.num_points),
                    "iteration_seconds": time.perf_counter() - started_iteration,
                    **extra_info,
                    **grad_info,
                    **metrics,
                })
                module_rows.append({
                    "arm": arm,
                    "absolute_step": step,
                    "module": "combined",
                    "event": "training_log",
                    "MDRR_enabled": active_mdrr,
                    "CICA_enabled": bool(cica_active),
                    "CICA_progress": float(cica_progress if cica_active else 0.0),
                    "resolved_lambda_CICA": lambda_cica if lambda_cica is not None else 0.0,
                    "population": int(model.num_points),
                    "g_mean": mdrr_info.get("g_mean", 0.0) if active_mdrr else 0.0,
                    "q_mean": mdrr_info.get("q_mean", 0.0) if active_mdrr else 0.0,
                    "CICA_valid_fraction": cica_info.get("valid_fraction", 0.0),
                    "CICA_delta_R_over_G": cica_info.get("delta_rg_mean", 0.0),
                    "CICA_delta_B_over_G": cica_info.get("delta_bg_mean", 0.0),
                    "CICA_delta_R_over_G_positive_fraction": cica_info.get("delta_rg_positive_fraction", 0.0),
                    "CICA_delta_R_over_G_negative_fraction": cica_info.get("delta_rg_negative_fraction", 0.0),
                    "CICA_delta_B_over_G_positive_fraction": cica_info.get("delta_bg_positive_fraction", 0.0),
                    "CICA_delta_B_over_G_negative_fraction": cica_info.get("delta_bg_negative_fraction", 0.0),
                    "color_prior_collapse_warning": cica_info.get("color_prior_collapse_warning", False),
                    "BVAM_enabled": bool(BVAM_ENABLED),
                    "BVAM_expert_usage_1": bvam_info.get("expert_usage_1", 0.0),
                    "BVAM_expert_usage_2": bvam_info.get("expert_usage_2", 0.0),
                    "BVAM_expert_entropy": bvam_info.get("expert_entropy", 0.0),
                    "BVAM_expert_collapse_fraction": bvam_info.get("expert_collapse_fraction", 0.0),
                    "BVAM_delta_norm_p90": bvam_info.get("delta_norm_p90", 0.0),
                })
            topology_rows.append(_topology_row(arm, step, model, cumulative)) if step % 500 == 0 or step in snapshot_steps else None
            if step in snapshot_steps or (RESPONSIBILITY_SMOKE and step in (6, 7)):
                mdrr_hash = mdrr_control_hash
                cica_hash = _hash_tensor(cica_controller["target_log_chroma"]) if cica_controller is not None else ""
                auxiliary_hash = _hash_tensor(auxiliary_controller["targets"]) if auxiliary_controller is not None else ""
                taba_rows = _taba_snapshot_rows(model, records, step) if bool(globals().get("TABA_DIAGNOSTIC_ENABLED", False)) else []
                checkpoint = _save_checkpoint(branch, scene, arm, step, output_dir, source_payload, mdrr_hash, cica_hash, auxiliary_hash, lambda_cica, brash_stats, taba_rows)
                checkpoints.append({"arm": arm, "absolute_step": step, "path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": _sha256(checkpoint)})
                if taba_rows:
                    _write_csv(output_dir / "taba_training_view_statistics.csv", [dict(row, scene=scene, variant=ACTIVE_PARAMETERIZATION) for row in taba_rows])
                    if TABA_EXPERIMENT:
                        _write_csv(output_dir / f"taba_statistics_step_{step:05d}.csv", taba_rows)
                    if GRAC_EXPERIMENT:
                        _write_csv(output_dir / f"grac_statistics_step_{step:05d}.csv", taba_rows)
                    if GCRC_EXPERIMENT:
                        _write_csv(output_dir / f"gcrc_statistics_step_{step:05d}.csv", taba_rows)
                    if CRCC_EXPERIMENT:
                        _write_csv(output_dir / f"crcc_statistics_step_{step:05d}.csv", taba_rows)
                _write_csv(output_dir / f"training_summary_{arm}.csv", rows)
                _write_csv(output_dir / "module_statistics.csv", module_rows)
                _write_csv(output_dir / "topology_statistics.csv", topology_rows)
            del outputs, losses, base_loss, auxiliary_raw, auxiliary_weighted, cica_raw, cica_weighted, brash_regularizer, bvam_regularizer, batch
            if step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        projector_hash_end = _hash_optional_tensor(model._camera_medium_observability_projector)
        result = {
            "scene": scene,
            "arm": arm,
            "completed_updates": len(sequence),
            "formal_global_step": int(sequence[-1]["absolute_step"]),
            "responsibility_state_summary": ({k: responsibility.state[k] for k in ("phase", "event_index", "rho_planned", "rho_effective", "paused", "optimizer_calls")} if responsibility is not None else None),
            "first_step": int(sequence[0]["absolute_step"]),
            "final_step": int(sequence[-1]["absolute_step"]),
            "training_wall_seconds": time.perf_counter() - started,
            "ocmc_projector_hash_start": projector_hash_start,
            "ocmc_projector_hash_end": projector_hash_end,
            "ocmc_projector_unchanged": projector_hash_start == projector_hash_end,
            "ocmc_enabled": OCMC_ENABLED,
            "raoc_enabled": False,
            "mdrr_enabled": use_mdrr,
            "cica_enabled": use_cica,
            "resolved_lambda_CICA": lambda_cica,
            "checkpoint_count": len(checkpoints),
            "checkpoint_rows": checkpoints,
            "training_rows": rows,
            "module_rows": module_rows,
            "topology_rows": topology_rows,
            "event_rows": event_rows,
            "final_population": int(model.num_points),
            "snapshot_steps": list(snapshot_steps),
            "auxiliary_appearance_enabled": AUXILIARY_APPEARANCE_ENABLED,
            "brash_enabled": bool(BRASH_ENABLED),
            "brash_gate_mode": BRASH_GATE_MODE,
            "brash_gate_prior_enabled": bool(BRASH_GATE_PRIOR_ENABLED),
            "brash_gate_prior_weight": float(BRASH_GATE_PRIOR_WEIGHT),
            "brash_reliability_stats": dict(brash_stats),
            "bvam_enabled": bool(BVAM_ENABLED),
            "bvam_num_experts": int(BVAM_NUM_EXPERTS),
            "bvam_pseudogt_enabled": bool(BVAM_PSEUDOGT_ENABLED),
            "bvam_pseudogt_stats": bvam_controller["stats"] if bvam_controller is not None else {"enabled": False},
        }
        _write_json(output_dir / "training_summary.json", result)
        _write_csv(output_dir / "refinement_events.csv", event_rows)
        return result
    except Exception as exc:
        _write_json(output_dir / "failure.json", {"scene": scene, "arm": arm, "error": repr(exc)})
        raise
    finally:
        _release(branch)


def _load_snapshot(branch: Any, scene: str, path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_headroom = payload.get("metadata", {}).get("headroom_config")
    if saved_headroom is not None:
        configured = _locked_config(branch.pipeline.model)
        for key, value in saved_headroom.items():
            if _sanitize(configured[key]) != _sanitize(value):
                raise RuntimeError(f"Checkpoint headroom setting mismatch for {key}: {value} != {configured[key]}")
    branch.pipeline.model.load_state_dict(payload["model"], strict=True)
    branch.pipeline.model.step = int(payload["absolute_step"])
    _configure_model(branch.pipeline.model)
    FORMAL._install_condition(branch.pipeline.model, "C0", payload["ocmc_bundle"], None)
    _configure_model(branch.pipeline.model)
    branch.pipeline.eval()
    return payload


def _restore_bvam_support_mask(
    branch: Any,
    output_dir: Path,
) -> Dict[str, Any]:
    """Rebuild BVAM's detached support state after checkpoint reload.

    The mask is derived from training cameras and is intentionally not a
    learnable checkpoint parameter.  It must therefore be reconstructed after
    loading a snapshot whose Gaussian population may differ from the start
    state because of refinement.
    """

    if not BVAM_ENABLED:
        return {"enabled": False}
    bank_path = output_dir / "cica_camera_bank.json"
    if not bank_path.is_file():
        raise FileNotFoundError(f"BVAM support bank is missing: {bank_path}")
    bank = json.loads(bank_path.read_text(encoding="utf8"))
    records = _train_records(branch)
    bank_records = [records[int(index)] for index in bank["camera_indices"]]
    model = branch.pipeline.model
    was_training = model.training
    model.eval()
    controller = _build_bvam_controller(model, bank_records)
    model.set_bvam_valid_mask(controller["valid"])
    if was_training:
        model.train()
    return dict(controller["stats"])


def _evaluate_checkpoint(branch: Any, arm: str, step: int, records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]], render_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    model = branch.pipeline.model
    metric_values = {key: [] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}
    per_view: List[Dict[str, Any]] = []
    render_rows: List[Dict[str, Any]] = []
    maps: Dict[str, Dict[str, Tensor]] = {}
    for _index, view_id, camera, batch in records:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        render_started = time.perf_counter()
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
            torch.cuda.synchronize()
            render_seconds = time.perf_counter() - render_started
            render_peak_bytes = torch.cuda.max_memory_allocated()
            gt = _gt(model, _batch_to_device(batch, model.device), outputs["background"]).detach().float().clamp(0.0, 1.0)
        pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
        metrics = _taba_rgb_metrics(pred, gt) if (TABA_EXPERIMENT or RESPONSIBILITY_ENABLED) else MIC._metric_images(model, pred, gt)
        clear = outputs.get("rgb_clear")
        clear_raw = outputs.get("clear_object_fullsh_raw")
        if not isinstance(clear, Tensor) or not isinstance(clear_raw, Tensor):
            raise RuntimeError("native clear outputs are missing from the renderer")
        clear = clear.detach().float().clamp(0.0, 1.0)
        clear_raw = clear_raw.detach().float()
        per_view.append({"arm": arm, "absolute_step": step, "split": "eval", "view_id": str(view_id),
                         "render_seconds": render_seconds, "render_peak_cuda_bytes": render_peak_bytes,
                         **metrics, **_clear_distribution(clear_raw)})
        if TABA_EXPERIMENT:
            per_view[-1].update(taba_statistics(outputs, xys=model.xys, radii=model.radii))
            if GRAC_EXPERIMENT:
                per_view[-1].update(grac_statistics(outputs))
            if GCRC_EXPERIMENT:
                per_view[-1].update(gcrc_statistics(outputs))
            if CRCC_EXPERIMENT:
                per_view[-1].update(crcc_statistics(outputs))
            per_view[-1]["rgb_object_residual"] = float((outputs["rgb_object"] - gt).abs().mean())
            per_view[-1]["rgb_medium_residual"] = float((outputs["rgb_medium"] - (gt - outputs["rgb_object"])).abs().mean())
        for key in metric_values:
            metric_values[key].append(metrics[key])
        maps[str(view_id)] = {key: value.detach().float().cpu() for key, value in outputs.items() if isinstance(value, Tensor)}
        maps[str(view_id)]["pred"] = pred.cpu()
        maps[str(view_id)]["gt"] = gt.cpu()
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)
            scene_dir = render_dir / f"step_{step:05d}" / str(view_id) / arm
            scene_dir.mkdir(parents=True, exist_ok=True)
            _save_native_rgb(scene_dir / "underwater.png", pred)
            _save_native_rgb(scene_dir / "gt.png", gt)
            _save_native_rgb(scene_dir / "clear.png", clear)
            _save_native_rgb(scene_dir / "clear_raw_display_clamp01.png", clear_raw)
            if RESPONSIBILITY_ENABLED:
                np.savez_compressed(scene_dir / 'native_float.npz',
                    underwater=pred.cpu().numpy(), gt=gt.cpu().numpy(),
                    clear=clear.cpu().numpy(), clear_raw=clear_raw.cpu().numpy())
            render_rows.extend(
                [
                    {"scene": render_dir.name, "arm": arm, "absolute_step": step, "view_id": str(view_id), "output_type": "underwater", "path": str(scene_dir / "underwater.png")},
                    {"scene": render_dir.name, "arm": arm, "absolute_step": step, "view_id": str(view_id), "output_type": "clear_native", "path": str(scene_dir / "clear.png")},
                    {"scene": render_dir.name, "arm": arm, "absolute_step": step, "view_id": str(view_id), "output_type": "clear_raw_display_clamp01", "path": str(scene_dir / "clear_raw_display_clamp01.png")},
                ]
            )
        del outputs, gt, pred
    global_row = {"arm": arm, "absolute_step": step, "split": "eval", "view_count": len(per_view), **{key: sum(values) / max(len(values), 1) for key, values in metric_values.items()}}
    decomposition = CAM._decomposition_row(arm, step, "eval", maps)
    return global_row, per_view, decomposition, render_rows


@torch.no_grad()
def _taba_rgb_metrics(pred: Tensor, gt: Tensor) -> Dict[str, float]:
    """Project-local RGB metrics on exactly paired native-resolution tensors."""
    import metrics as local_metrics

    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"RGB pair shape mismatch: {pred.shape} / {gt.shape}")
    if not bool(torch.isfinite(pred).all() & torch.isfinite(gt).all()):
        raise ValueError("nonfinite RGB metrics input")
    if not bool((pred >= 0).all() & (pred <= 1).all() & (gt >= 0).all() & (gt <= 1).all()):
        raise ValueError("RGB metrics inputs must be in [0, 1]")
    pred, gt = pred.permute(2, 0, 1)[None].contiguous(), gt.permute(2, 0, 1)[None].contiguous()
    return {
        "PSNR": float(local_metrics.psnr(pred, gt).mean()),
        "SSIM": float(local_metrics.ssim(pred, gt).mean()),
        "LPIPS": float(local_metrics.lpips(pred, gt, net_type="vgg").mean()),
        "MSE": float((pred - gt).square().mean()),
    }


def _evaluate_arm(scene: str, arm: str, output_dir: Path, render_dir: Path) -> Dict[str, Any]:
    branch = None
    try:
        branch = _new_branch(scene)
        records_eval = _eval_records(branch)
        global_rows: List[Dict[str, Any]] = []
        per_view_rows: List[Dict[str, Any]] = []
        decomposition_rows: List[Dict[str, Any]] = []
        render_rows: List[Dict[str, Any]] = []
        for step in ((15000,) if RESPONSIBILITY_ENABLED else _active_snapshots(FINAL_STEP)):
            checkpoint = output_dir / "checkpoints" / f"step-{step:09d}.ckpt"
            payload = _load_snapshot(branch, scene, checkpoint)
            _restore_bvam_support_mask(branch, output_dir)
            row, views, decomp, saved_renders = _evaluate_checkpoint(
                branch,
                arm,
                step,
                records_eval,
                render_dir if (TABA_EXPERIMENT or step == FINAL_STEP) and (not ACTIVE_COMPACT_OUTPUT or ACTIVE_SAVE_COMPACT_RENDERS) else None,
            )
            global_rows.append(row)
            per_view_rows.extend(views)
            decomposition_rows.append(decomp)
            render_rows.extend(saved_renders)
            if payload.get("raoc_state") is not None:
                raise RuntimeError("RAOC state unexpectedly present in new checkpoint")
        _write_csv(output_dir / "evaluation_metrics.csv", global_rows)
        _write_csv(output_dir / "per_view_metrics.csv", per_view_rows)
        _write_json(output_dir / "decomposition_safety.json", {"rows": decomposition_rows, "all_P_J_gt_1_zero": all(float(row.get("P_J_gt_1", 0.0)) == 0.0 for row in decomposition_rows)})
        _write_csv(output_dir / "render_manifest.csv", render_rows)
        return {"global_rows": global_rows, "per_view_rows": per_view_rows, "decomposition_rows": decomposition_rows, "render_rows": render_rows}
    finally:
        _release(branch)


def _smoke_reload_audit(scene: str, arm: str, output_dir: Path, checkpoint: Path) -> Dict[str, Any]:
    """Reload the shortened-run endpoint and probe one rendered view."""

    branch = None
    try:
        branch = _new_branch(scene)
        payload = _load_snapshot(branch, scene, checkpoint)
        model = branch.pipeline.model
        records = _train_records(branch)
        camera_index, _view_id, camera, _batch = records[0]
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
        finite = FORMAL._finite_outputs(outputs)
        model_hash = _model_state_hash(model)
        checkpoint_hash = _hash_object(payload["model"])
        audit = {
            "scene": scene,
            "arm": arm,
            "absolute_step": int(payload["absolute_step"]),
            "checkpoint": str(checkpoint),
            "checkpoint_model_hash": checkpoint_hash,
            "reloaded_model_hash": model_hash,
            "model_hash_match": checkpoint_hash == model_hash,
            "forward_probe_camera_index": int(camera_index),
            "forward_finite": bool(finite),
            "tensor_sha256": {name: _hash_tensor(value) for name, value in outputs.items() if isinstance(value, Tensor) and name in ("pred_image", "rgb_object", "rgb_medium_finite", "rgb_tail")},
        }
        _write_json(output_dir / "smoke_reload_audit.json", audit)
        if not audit["model_hash_match"] or not audit["forward_finite"]:
            raise RuntimeError("smoke checkpoint reload audit failed")
        return audit
    finally:
        _release(branch)


def run(
    scene: str,
    arm: str,
    gpu: str,
    output_dir: Path,
    max_step: int = FINAL_STEP,
    source_root: Optional[Path] = None,
    source_config: Optional[Path] = None,
    parameterization: str = "bounded_sh3",
    resume_checkpoint: Optional[Path] = None,
    training_seed: int = TRAINING_SEED,
    compact_output: bool = False,
    fork_checkpoint: Optional[Path] = None,
    raw_data_only: bool = False,
) -> Dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    if fork_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("choose fork or resume, not both")
    global ACTIVE_SOURCE_ROOT, ACTIVE_SOURCE_CONFIG, ACTIVE_PARAMETERIZATION, RAW_DATA_ONLY, ACTIVE_INITIALIZATION_DIR
    global ACTIVE_TRAINING_SEED, ACTIVE_COMPACT_OUTPUT, ACTIVE_RENDER_ROOT
    global AUXILIARY_APPEARANCE_ENABLED
    if parameterization not in ("legacy", "bounded_sh3", "bounded_headroom_sh3", "bounded_selective_sh3", "bvam_bounded_sh3"):
        raise ValueError(f"unsupported intrinsic color parameterization: {parameterization}")
    ACTIVE_SOURCE_ROOT = (source_root or SOURCE_ROOT).resolve()
    ACTIVE_SOURCE_CONFIG = (source_config or SOURCE_CONFIG).resolve()
    ACTIVE_PARAMETERIZATION = parameterization
    RAW_DATA_ONLY = bool(raw_data_only)
    # The historical detached auxiliary appearance route is not part of the
    # fresh OCMC+M1+CICA responsibility objective.  Disable it for raw runs so
    # the audit and optimizer route describe exactly the requested experiment.
    if RAW_DATA_ONLY:
        AUXILIARY_APPEARANCE_ENABLED = False
        global FINAL_STEP, ACTIVE_SNAPSHOT_STEPS
        FINAL_STEP = 15000
        ACTIVE_SNAPSHOT_STEPS = (5000, 10000, 12000, 14000, 15000)
    ACTIVE_INITIALIZATION_DIR = output_dir / "initialization"
    if RESPONSIBILITY_ENABLED:
        ACTIVE_RENDER_ROOT = output_dir / "renders"
    ACTIVE_TRAINING_SEED = int(training_seed)
    ACTIVE_COMPACT_OUTPUT = bool(compact_output)
    backend_info = _route_renderer_backend(RENDERER_BACKWARD_MODE)
    FORMAL.ACTIVE_PARAMETERIZATION = ACTIVE_PARAMETERIZATION
    AUX.ACTIVE_PARAMETERIZATION = ACTIVE_PARAMETERIZATION
    runtime = _runtime(scene, gpu)
    if not ACTIVE_SOURCE_CONFIG.is_file():
        raise FileNotFoundError(ACTIVE_SOURCE_CONFIG)
    if output_dir.exists() and any(output_dir.iterdir()) and resume_checkpoint is None:
        raise RuntimeError(f"non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if ACTIVE_CHECKPOINT_DIRECTORY is not None:
        ACTIVE_CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=False)
        (output_dir / "checkpoints").symlink_to(ACTIVE_CHECKPOINT_DIRECTORY, target_is_directory=True)
    shutil.copy2(ACTIVE_SOURCE_CONFIG, output_dir / "source_config.yml")
    _seed_all(ACTIVE_TRAINING_SEED)
    # Historical A0 records the continuation RNG immediately after seeding,
    # before any pipeline setup consumes random values. Every new arm restores
    # this exact state immediately before its first continuation update.
    training_rng = _rng_state()
    probe = None
    try:
        probe = _new_branch(scene)
        source_payload = _load_source(probe, scene)
        start = _start_audit(probe, source_payload, output_dir, scene, arm, training_rng)
        sequence = _camera_sequence(scene, output_dir)
        partner = _partner_mapping(probe, output_dir)
        bank = _camera_bank(probe, output_dir)
        routing_audit = _gradient_routing_audit(
            probe,
            scene,
            arm,
            output_dir,
            _train_records(probe),
            partner,
            bank,
        )
        _write_json(output_dir / "experiment_manifest.json", {
            "experiment": EXPERIMENT,
            "scene": scene,
            "arm": arm,
            "runtime": runtime,
            "source_checkpoint": None if RAW_DATA_ONLY else str(_source_checkpoint(scene)),
            "source_checkpoint_sha256": None if RAW_DATA_ONLY else _sha256(_source_checkpoint(scene)),
            "source_config_sha256": _sha256(ACTIVE_SOURCE_CONFIG),
            "source_root": str(ACTIVE_SOURCE_ROOT),
            "fork_checkpoint": str(fork_checkpoint) if fork_checkpoint is not None else None,
            "intrinsic_color_parameterization": ACTIVE_PARAMETERIZATION,
            "start_step": 0 if RAW_DATA_ONLY else START_STEP,
            "final_step": max_step,
                "matched_updates": max_step if RAW_DATA_ONLY else max_step - START_STEP,
            "training_seed": ACTIVE_TRAINING_SEED,
            "compact_output": ACTIVE_COMPACT_OUTPUT,
            "temporary_checkpoint_directory": ACTIVE_CHECKPOINT_DIRECTORY,
            "training_rng_manifest": _rng_manifest(training_rng),
            "module_schedule": {
                "AUXILIARY_APPEARANCE": {"enabled": AUXILIARY_APPEARANCE_ENABLED, "name": "A_DETACHED_SH_OPACITY_TANGENT_ORTHOGONALIZATION", "strength": AUXILIARY_APPEARANCE_STRENGTH, "role": "CURRENT_BASE auxiliary appearance regularization; not claimed as identifiability innovation"},
                "MDRR": {"enabled": _arm_use_mdrr(arm), "start_step": MDRR_START_STEP, "refresh_rule": "every active primary update from the current model state", "full_medium_response": True},
                "CICA": {"enabled": _arm_use_cica(arm), "derivative_mode": CICA_DERIVATIVE_MODE, "start_step": CICA_START_STEP, "ramp_steps": CICA_RAMP_STEPS, "refresh_steps": list(CICA_REFRESH_STEPS), "camera_bank_max": CICA_BANK_MAX, "minimum_support": CICA_MIN_SUPPORT, "huber_delta": CICA_HUBER_DELTA, "target_gradient_fraction": CICA_GRADIENT_FRACTION},
            },
            "renderer_backward_mode": RENDERER_BACKWARD_MODE,
            "renderer_backend": backend_info,
            "ocmc_schedule": {"target_strength": OCMC_STRENGTH, "start_strength": OCMC_STRENGTH_START, "ramp_steps": OCMC_STRENGTH_RAMP_STEPS},
            "locked_config": _locked_config(probe.pipeline.model),
            "ocmc_on": OCMC_ENABLED,
            "raoc_off": True,
            "topology_unchanged_logic": "normal existing refinement callbacks; no module topology intervention",
            "gradient_routing_audit": routing_audit,
        })
    finally:
        _release(probe)
    result = _train_arm(
        scene,
        arm,
        output_dir,
        sequence,
        partner,
        bank,
        training_rng,
        max_step=max_step,
        resume_checkpoint=resume_checkpoint,
        fork_checkpoint=fork_checkpoint,
    )
    if RESPONSIBILITY_SMOKE and resume_checkpoint is None:
        replay_dir = output_dir / "resume_replay"
        replay_dir.mkdir(exist_ok=True)
        _train_arm(scene, arm, replay_dir, sequence, partner, bank, training_rng,
            max_step=7, resume_checkpoint=output_dir / "checkpoints/step-000000006.ckpt")
        uninterrupted = torch.load(output_dir / "checkpoints/step-000000007.ckpt", map_location="cpu", weights_only=False)
        replayed = torch.load(replay_dir / "checkpoints/step-000000007.ckpt", map_location="cpu", weights_only=False)
        differences = {k: float((v.float()-replayed["model"][k].float()).abs().max()) if v.numel() else 0.
                       for k,v in uninterrupted["model"].items()}
        matching = all(torch.allclose(v.float(), replayed["model"][k].float(), atol=1e-6, rtol=1e-5)
                       for k,v in uninterrupted["model"].items())
        state_matches = _hash_object(uninterrupted["responsibility_state"]) == _hash_object(replayed["responsibility_state"])
        replay_audit = dict(passed=matching and state_matches, atol=1e-6, rtol=1e-5,
            max_model_difference=max(differences.values()), event_state_matches=state_matches,
            mainline_updates=16, replay_updates=1, actual_updates=17)
        _write_json(output_dir / "resume_next_update_audit.json", replay_audit)
        if not replay_audit["passed"]:
            raise RuntimeError(f"smoke next-update restore mismatch: {replay_audit}")
    if max_step == FINAL_STEP:
        render_dir = ACTIVE_RENDER_ROOT / scene
        evaluation = _evaluate_arm(scene, arm, output_dir, render_dir)
        result["evaluation_rows"] = len(evaluation["global_rows"])
        result["per_view_rows"] = len(evaluation["per_view_rows"])
        _write_json(output_dir / "decomposition_safety.json", {"rows": evaluation["decomposition_rows"], "all_P_J_gt_1_zero": all(float(row.get("P_J_gt_1", 0.0)) == 0.0 for row in evaluation["decomposition_rows"])})
        if ACTIVE_COMPACT_OUTPUT:
            removed_checkpoints = []
            for checkpoint in sorted((output_dir / "checkpoints").glob("*.ckpt")):
                removed_checkpoints.append({"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size})
                checkpoint.unlink()
            result["checkpoint_retained"] = False
            result["removed_checkpoints"] = removed_checkpoints
        else:
            result["checkpoint_retained"] = True
        _write_json(output_dir / "final_summary.json", result)
    else:
        smoke_checkpoint = output_dir / "checkpoints" / f"step-{max_step:09d}.ckpt"
        reload_audit = _smoke_reload_audit(scene, arm, output_dir, smoke_checkpoint)
        result["smoke_reload_audit"] = reload_audit
        _write_json(output_dir / "final_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(SCENE_GPUS), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--gpu", choices=tuple(sorted(ALLOWED_GPUS)), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=FINAL_STEP)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--parameterization", choices=("legacy", "bounded_sh3", "bounded_headroom_sh3", "bounded_selective_sh3"), default="bounded_sh3")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--raw-data-only", action="store_true")
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--compact-output", action="store_true")
    parser.add_argument("--cica-derivative-mode", choices=("legacy", "exact_fsb"), default="legacy")
    parser.add_argument("--renderer-backward-mode", choices=("legacy", "fixed", "corrected_v1"), default="legacy")
    args = parser.parse_args()
    global CICA_DERIVATIVE_MODE, RENDERER_BACKWARD_MODE
    CICA_DERIVATIVE_MODE = args.cica_derivative_mode
    RENDERER_BACKWARD_MODE = args.renderer_backward_mode
    result = run(
        args.scene,
        args.arm,
        args.gpu,
        args.output_dir.resolve(),
        int(args.max_step),
        source_root=args.source_root.resolve(),
        source_config=args.source_config.resolve(),
        parameterization=args.parameterization,
        resume_checkpoint=args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None,
        training_seed=int(args.seed),
        compact_output=bool(args.compact_output),
        raw_data_only=bool(args.raw_data_only),
    )
    print(json.dumps(_sanitize(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
