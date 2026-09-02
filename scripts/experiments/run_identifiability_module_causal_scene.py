#!/usr/bin/env python3
"""Run one matched OCMC identifiability-module causal continuation.

Both arms restore the same registered OCMC C0 checkpoint at step 3000.  The
only intervention is an external, detached SH-opacity tangent regularizer in
C1.  This script does not modify the production model, renderer, optimizer,
or training callbacks.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.engine.optimizers import Optimizers

from scripts.diagnostics import audit_gaussian_parameter_identifiability_ocmc as AUDIT
from scripts.diagnostics import audit_low_support_causal_intervention as CAUSAL
from scripts.diagnostics import preflight_gaussian_identifiability_module as PREFLIGHT
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL
from water_splatting._torch_impl import eval_sh_bases


EXPERIMENT = "IDENTIFIABILITY_MODULE_CAUSAL_EXPERIMENT"
MECHANISM = "A_DETACHED_SH_OPACITY_TANGENT_ORTHOGONALIZATION"
EXPECTED_BRANCH = "research/m1-bounded-intrinsic"
ARMS = ("C0", "C1")
START_STEP = 3000
FINAL_STEP = 14999
FORMAL_SNAPSHOT_STEPS = (5000, 8000, 10000, 13000, 14999)
DIAGNOSTIC_STEPS = (3000,) + FORMAL_SNAPSHOT_STEPS
MODULE_STRENGTH = 1.0
OVERLAP_THRESHOLD = 0.80
TRAINING_SEED = 42
LOG_INTERVAL = 500
DIAGNOSTIC_SAMPLE_COUNT = 256
EPS = 1e-12
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
ALLOWED_GPUS = frozenset(("6", "7", "8", "9"))
SCENE_GPUS = dict(CAUSAL.SCENE_GPUS)
SCENES = tuple(SCENE_GPUS)
EXPECTED_SOURCE_CHECKPOINT_HASHES = {
    "Curasao": "06eeaa97550a5a49abdf483e291e637b58f1e6c608f7743935ebe57fbbaed950",
    "IUI3-RedSea": "73b7c7f5739b44950e63f815c0f24873db3fb00e68a79ac8133e3e6a298e86b3",
    "JapaneseGradens-RedSea": "5f52b47704b075a72de3ed14ef0147fdad194cd2311756325fc5e9338df266a4",
    "Panama": "f1b9a515c6cc4ffd30698d50f42f7b01291a3238dd9d38b33e05294b22cd4a23",
}

# Registered before the formal run.  These are safety/classification gates,
# not tunable module parameters.
MIN_CAPACITY_RATIO = 0.75
MIN_OPACITY_RATIO = 0.75
MAX_OPACITY_RATIO = 1.25
MAX_FINAL_POPULATION_RELATIVE_GAP = 0.10
HISTORICAL_UNTRACKED_HASHES = {
    "scripts/diagnostics/analyze_raoc_q50_q80_causal_four_scene.py": "b6a271372e68cd07fc566a3fde5ced5ba6463531278c31a6cfa47972aa15e8d6",
    "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py": "539f1c044f9ed136dce65b1dedc01746097cb2f3c4298c9682038019d23dfd7a",
    "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py": "fe3fd3ddcdbbff7904cfb7225a0ba024f928a9020777561252b66663c3c8ab32",
    "scripts/experiments/run_raoc_q50_q80_causal_four_scene.py": "d131428cc20ea76010e237abd91ac4cddfc5c6a78944c57c3317ed18bcdf60ef",
    "scripts/experiments/run_raoc_q50_q80_causal_scene.py": "3a924e88a606d34360a90348f3a392d0d12f80d43c98fe72b56cbec2d27ad6e7",
}
REGISTERED_PROTECTED_HASHES = {
    **PREFLIGHT.PROTECTED_HASHES,
    **HISTORICAL_UNTRACKED_HASHES,
}


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Tensor):
        cpu = value.detach().cpu()
        return _sanitize(cpu.item() if cpu.numel() == 1 else cpu.tolist())
    if isinstance(value, np.generic):
        return _sanitize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf8",
    )


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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_object(value: Any) -> str:
    return _hash_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _hash_tensor(value: Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    return _hash_bytes(tensor.numpy().tobytes())


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _protected_hashes() -> Dict[str, str]:
    hashes = {relative: _sha256(REPO_ROOT / relative) for relative in REGISTERED_PROTECTED_HASHES}
    for relative, expected in REGISTERED_PROTECTED_HASHES.items():
        if hashes[relative] != expected:
            raise RuntimeError(f"protected source changed: {relative}")
    return hashes


def _runtime(scene: str, gpu: str) -> Dict[str, Any]:
    if scene not in SCENES or SCENE_GPUS[scene] != gpu:
        raise RuntimeError(f"invalid formal scene/GPU assignment: {scene}/{gpu}")
    if gpu not in ALLOWED_GPUS or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError(f"worker must expose only physical GPU {gpu}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("worker must see exactly one CUDA device")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("worker must use logical cuda:0")
    props = torch.cuda.get_device_properties(0)
    return {
        "scene": scene,
        "physical_gpu": gpu,
        "logical_gpu": 0,
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": True,
    }


def _stable_seed(scene: str, purpose: str) -> int:
    encoded = f"{TRAINING_SEED}:{scene}:{purpose}".encode("utf8")
    return int(hashlib.sha256(encoded).hexdigest()[:8], 16)


def _rng_state() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def _set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _seed_rng(seed: int) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return _rng_state()


def _rng_manifest(state: Mapping[str, Any]) -> Dict[str, Any]:
    result = {
        "python_sha256": _hash_object(state["python"]),
        "numpy_sha256": _hash_object(state["numpy"]),
        "torch_cpu_sha256": _hash_tensor(state["torch_cpu"]),
    }
    result["torch_cuda_sha256"] = [_hash_tensor(item) for item in state.get("torch_cuda", [])]
    return result


def _source_checkpoint(scene: str) -> Path:
    return SOURCE_ROOT / scene / "checkpoints" / "C0" / f"step-{START_STEP:09d}.ckpt"


def _source_sequence(scene: str) -> Path:
    return SOURCE_ROOT / scene / "camera_sequence.json"


def _locked_config(model: Any) -> Dict[str, Any]:
    cfg = model.config
    return {
        "intrinsic_color_parameterization": cfg.intrinsic_color_parameterization,
        "sh_degree": int(cfg.sh_degree),
        "rasterize_mode": cfg.rasterize_mode,
        "medium_context_mode": cfg.medium_context_mode,
        "b_inf_mode": cfg.b_inf_mode,
        "camera_medium_observability_enabled": bool(cfg.camera_medium_observability_enabled),
        "camera_medium_ray_adaptive_observability_enabled": bool(
            cfg.camera_medium_ray_adaptive_observability_enabled
        ),
        "infinite_water_enabled": bool(cfg.infinite_water_enabled),
        "coarse_depth_supervision_enabled": bool(cfg.coarse_depth_supervision_enabled),
        "medium_identifiability_enabled": bool(cfg.medium_identifiability_enabled),
        "appearance_lr_scale": float(getattr(cfg, "appearance_lr_scale", 1.0)),
        "refine_every": int(cfg.refine_every),
        "stop_split_at": int(cfg.stop_split_at),
        "reset_alpha_every": int(cfg.reset_alpha_every),
        "continue_cull_post_densification": bool(cfg.continue_cull_post_densification),
    }


def _assert_locked_config(model: Any) -> Dict[str, Any]:
    snapshot = _locked_config(model)
    expected = {
        "intrinsic_color_parameterization": "bounded_sh3",
        "sh_degree": 3,
        "rasterize_mode": "classic",
        "medium_context_mode": "dir_xy_camera",
        "b_inf_mode": "tied",
        "camera_medium_observability_enabled": True,
        "camera_medium_ray_adaptive_observability_enabled": False,
        "infinite_water_enabled": False,
        "coarse_depth_supervision_enabled": False,
        "medium_identifiability_enabled": False,
        "appearance_lr_scale": 1.0,
    }
    for key, value in expected.items():
        if snapshot[key] != value:
            raise RuntimeError(f"locked configuration drift: {key}={snapshot[key]!r}, expected {value!r}")
    return snapshot


def _release(branch: Optional[Any]) -> None:
    if branch is None:
        return
    FORMAL._release(branch)


def _load_branch(scene: str, arm: str, load_optimizer: bool = True) -> Tuple[Any, Mapping[str, Any]]:
    if arm not in ARMS:
        raise ValueError(arm)
    scene_cfg = FORMAL.SCENES[scene]
    branch = FORMAL._setup_branch(REPO_ROOT, scene_cfg, "C0")
    checkpoint = _source_checkpoint(scene)
    payload = torch.load(checkpoint, map_location="cpu")
    if (
        payload.get("branch") != "C0"
        or int(payload.get("absolute_step", -1)) != START_STEP
        or payload.get("ocmc_bundle") is None
        or payload.get("raoc_state") is not None
    ):
        raise RuntimeError(f"invalid common OCMC start checkpoint: {checkpoint}")
    model = branch.pipeline.model
    model.load_state_dict(payload["model"], strict=True)
    model.step = START_STEP
    FORMAL._configure_model(model, "C0")
    FORMAL._install_condition(model, "C0", payload["ocmc_bundle"], None)
    _assert_locked_config(model)
    branch.branch = arm
    branch.scalers = dict(payload.get("scalers", {}))
    if load_optimizer:
        del branch.optimizers
        branch.optimizers = Optimizers(MIC._optimizer_groups(branch.config, model), model.get_param_groups())
        if set(branch.optimizers.optimizers) != set(payload["optimizers"]):
            raise RuntimeError("optimizer group mismatch at common start")
        if set(branch.optimizers.schedulers) != set(payload["schedulers"]):
            raise RuntimeError("scheduler group mismatch at common start")
        for group, optimizer in branch.optimizers.optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][group])
        for group, scheduler in branch.optimizers.schedulers.items():
            scheduler.load_state_dict(payload["schedulers"][group])
    branch.pipeline.eval()
    return branch, payload


def _model_tensors(model: Any) -> Dict[str, Tensor]:
    result = {
        name: getattr(model, name).detach().cpu().clone()
        for name in ("means", "features_dc", "features_rest", "scales", "quats", "opacities")
    }
    result["medium_mlp"] = torch.cat(
        [parameter.detach().cpu().reshape(-1) for parameter in model.medium_mlp.parameters()]
    )
    result["direction_encoding"] = torch.cat(
        [parameter.detach().cpu().reshape(-1) for parameter in model.direction_encoding.parameters()]
    )
    return result


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_nested_equal(a, b) for a, b in zip(left, right))
        )
    try:
        return bool(left == right)
    except Exception:
        return False


def _optimizer_state(branch: Any) -> Dict[str, Any]:
    return {name: optimizer.state_dict() for name, optimizer in branch.optimizers.optimizers.items()}


def _scheduler_state(branch: Any) -> Dict[str, Any]:
    return {name: scheduler.state_dict() for name, scheduler in branch.optimizers.schedulers.items()}


def _start_snapshot(scene: str, arm: str) -> Dict[str, Any]:
    branch = None
    try:
        branch, payload = _load_branch(scene, arm)
        model = branch.pipeline.model
        records = FORMAL._train_records(branch.pipeline)
        _index, view_id, camera, batch = records[0]
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
            gt = AUDIT.VC.MI.PW._get_gt(model, batch, outputs["background"])
            loss = model.get_loss_dict(outputs, {**batch, "image": batch["image"].to(model.device)}, {})[
                "main_loss"
            ]
        forward = {
            key: value.detach().cpu().clone()
            for key, value in outputs.items()
            if isinstance(value, Tensor) and key in ("pred_image", "depth", "accumulation", "medium_rgb", "medium_bs", "medium_attn", "b_inf")
        }
        return {
            "arm": arm,
            "model": _model_tensors(model),
            "optimizer": _optimizer_state(branch),
            "scheduler": _scheduler_state(branch),
            "scalers": dict(branch.scalers),
            "forward": forward,
            "main_loss": float(loss.detach().cpu()),
            "probe_view": view_id,
            "gaussian_count": int(model.num_points),
            "ocmc_projector": model._camera_medium_observability_projector.detach().cpu().clone(),
            "locked_config": _locked_config(model),
            "checkpoint_rng_manifest": dict(payload.get("rng_manifest", {})),
        }
    finally:
        _release(branch)


def _max_diff(left: Tensor, right: Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max()) if left.numel() else 0.0


def _start_state_audit(scene: str, output_dir: Path, training_rng: Mapping[str, Any]) -> Dict[str, Any]:
    left = _start_snapshot(scene, "C0")
    right = _start_snapshot(scene, "C1")
    parameter_rows = []
    for name in sorted(left["model"]):
        diff = _max_diff(left["model"][name], right["model"][name])
        parameter_rows.append({"parameter": name, "max_abs_diff": diff, "pass": diff == 0.0})
    forward_rows = []
    for name in sorted(left["forward"]):
        diff = _max_diff(left["forward"][name], right["forward"][name])
        forward_rows.append({"quantity": name, "max_abs_diff": diff, "pass": diff == 0.0})
    loss_diff = abs(left["main_loss"] - right["main_loss"])
    forward_rows.append({"quantity": "main_loss", "max_abs_diff": loss_diff, "pass": loss_diff == 0.0})
    optimizer_equal = _nested_equal(left["optimizer"], right["optimizer"])
    scheduler_equal = _nested_equal(left["scheduler"], right["scheduler"])
    scaler_equal = _nested_equal(left["scalers"], right["scalers"])
    projector_diff = _max_diff(left["ocmc_projector"], right["ocmc_projector"])
    config_equal = left["locked_config"] == right["locked_config"]
    planned_rng = _rng_manifest(training_rng)
    _set_rng_state(training_rng)
    reset_rng = _rng_manifest(_rng_state())
    rng_equal = planned_rng == reset_rng
    all_pass = bool(
        all(row["pass"] for row in parameter_rows)
        and all(row["pass"] for row in forward_rows)
        and optimizer_equal
        and scheduler_equal
        and scaler_equal
        and projector_diff == 0.0
        and config_equal
        and rng_equal
    )
    result = {
        "START_STATE_EQUIVALENCE": all_pass,
        "scene": scene,
        "common_checkpoint": str(_source_checkpoint(scene)),
        "common_checkpoint_sha256": _sha256(_source_checkpoint(scene)),
        "absolute_step_C0": START_STEP,
        "absolute_step_C1": START_STEP,
        "max_abs_model_diff": max(row["max_abs_diff"] for row in parameter_rows),
        "max_abs_forward_diff": max(row["max_abs_diff"] for row in forward_rows),
        "optimizer_equivalent": optimizer_equal,
        "scheduler_equivalent": scheduler_equal,
        "scaler_equivalent": scaler_equal,
        "rng_equivalent": rng_equal,
        "ocmc_projector_max_abs_diff": projector_diff,
        "ocmc_config_equivalent": config_equal,
        "ocmc_projector_sha256_C0": _hash_tensor(left["ocmc_projector"]),
        "ocmc_projector_sha256_C1": _hash_tensor(right["ocmc_projector"]),
        "ocmc_configuration_sha256_C0": _hash_object(left["locked_config"]),
        "ocmc_configuration_sha256_C1": _hash_object(right["locked_config"]),
        "gaussian_count_C0": left["gaussian_count"],
        "gaussian_count_C1": right["gaussian_count"],
        "probe_view_C0": left["probe_view"],
        "probe_view_C1": right["probe_view"],
        "continuation_rng_manifest": planned_rng,
        "source_checkpoint_rng_manifest": left["checkpoint_rng_manifest"],
        "rng_provenance": (
            "The source checkpoint stores RNG hashes rather than serializable states. Both arms explicitly "
            "restore the same registered seed-42-derived continuation RNG state before their first update."
        ),
        "model_parameter_rows": parameter_rows,
        "forward_rows": forward_rows,
    }
    _write_json(output_dir / "start_state_equivalence.json", result)
    _write_csv(output_dir / "start_state_parameter_equivalence.csv", parameter_rows)
    _write_csv(output_dir / "start_state_forward_equivalence.csv", forward_rows)
    if not all_pass:
        raise RuntimeError("START_STATE_EQUIVALENCE=false; formal training is forbidden")
    return result


def _camera_sequence(scene: str, final_step: int, output_dir: Path) -> List[Dict[str, Any]]:
    source = _read_json(_source_sequence(scene))
    rows = [
        dict(row)
        for row in source["rows"]
        if START_STEP < int(row["absolute_step"]) <= int(final_step)
    ]
    expected = final_step - START_STEP
    if len(rows) != expected:
        raise RuntimeError(f"camera continuation length {len(rows)} != {expected}")
    if [int(row["absolute_step"]) for row in rows] != list(range(START_STEP + 1, final_step + 1)):
        raise RuntimeError("source camera sequence is not contiguous")
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf8")
    payload = {
        "scene": scene,
        "rows": rows,
        "length_C0": len(rows),
        "length_C1": len(rows),
        "camera_mismatch_count": 0,
        "CAMERA_SEQUENCE_EXACT_MATCH": True,
        "sha256_C0": _hash_bytes(encoded),
        "sha256_C1": _hash_bytes(encoded),
        "source_path": str(_source_sequence(scene)),
        "source_sha256": _sha256(_source_sequence(scene)),
    }
    _write_json(output_dir / "camera_sequence.json", payload)
    _write_csv(output_dir / "camera_sequence.csv", rows)
    return rows


def _sample_image(image: Tensor, xys: Tensor) -> Tensor:
    height, width = int(image.shape[0]), int(image.shape[1])
    x = xys[:, 0].round().long().clamp(0, width - 1)
    y = xys[:, 1].round().long().clamp(0, height - 1)
    return image[y, x]


def _view_response_terms(
    model: Any,
    camera: Any,
    outputs: Mapping[str, Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return visible IDs, SH bases, Jacobian scales, opacity tangent, and SH response."""

    ids = torch.nonzero(outputs["gaussian_visible_mask"].reshape(-1).bool(), as_tuple=False).reshape(-1)
    means = model.means.detach()[ids]
    position = camera.camera_to_worlds[..., :3, 3].to(model.device)
    viewdirs = means - position.detach()
    viewdirs = viewdirs / torch.linalg.vector_norm(viewdirs, dim=-1, keepdim=True).clamp_min(1e-6)
    bases = eval_sh_bases(16, viewdirs).float()[:, 1:]
    color = outputs["gaussian_view_rgb"].detach().float()[ids]
    dc_color = outputs["gaussian_view_dc_rgb"].detach().float()[ids]
    depth = outputs["projected_gaussian_depths"].detach().reshape(-1)[ids].float()
    xys = model.xys.detach()[ids].float()
    beta_d = _sample_image(outputs["medium_attn"].detach().float(), xys)
    beta_b = _sample_image(outputs["medium_bs"].detach().float(), xys)
    b_inf = _sample_image(outputs["b_inf"].detach().float(), xys)
    transmission_d = torch.exp(-(beta_d * depth[:, None]).clamp_min(0.0)).clamp(0.0, 1.0)
    transmission_b = torch.exp(-(beta_b * depth[:, None]).clamp_min(0.0)).clamp(0.0, 1.0)
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)[ids].float()
    tangent = opacity[:, None] * (1.0 - opacity[:, None]) * (
        transmission_d * color - transmission_b * b_inf
    )
    jacobian_scale = opacity[:, None] * transmission_d * color * (1.0 - color)
    response = opacity[:, None] * transmission_d * (color - dc_color)
    return ids, bases, jacobian_scale, tangent, response


@torch.no_grad()
def _refresh_controller(model: Any, train_records: Sequence[Tuple[int, str, Any, Any]]) -> Dict[str, Any]:
    """Build one detached multi-view shared direction for every Gaussian."""

    was_training = model.training
    model.eval()
    count = int(model.num_points)
    shared = torch.zeros_like(model.features_rest.detach(), dtype=torch.float32)
    response_dot = torch.zeros(count, device=model.device, dtype=torch.float32)
    tangent_norm_sq = torch.zeros_like(response_dot)
    support = torch.zeros(count, device=model.device, dtype=torch.int16)
    for _index, _view_id, camera, _batch in train_records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        ids, bases, scale, tangent, response = _view_response_terms(model, camera, outputs)
        shared[ids] += bases[:, :, None] * (scale * tangent)[:, None, :]
        response_dot[ids] += (response * tangent).sum(dim=-1)
        tangent_norm_sq[ids] += tangent.square().sum(dim=-1)
        support[ids] += 1
        del outputs, ids, bases, scale, tangent, response
    shared_norm = torch.linalg.vector_norm(shared.reshape(count, -1), dim=-1)
    directions = shared / shared_norm[:, None, None].clamp_min(EPS)
    anchor = model.features_rest.detach().float()
    anchor_projection = (anchor * directions).sum(dim=(1, 2))
    targets = anchor_projection - response_dot / shared_norm.clamp_min(EPS)

    predicted_dot = torch.zeros(count, device=model.device, dtype=torch.float32)
    predicted_norm_sq = torch.zeros_like(predicted_dot)
    for _index, _view_id, camera, _batch in train_records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        ids, bases, scale, tangent, _response = _view_response_terms(model, camera, outputs)
        predicted = scale * torch.einsum("nkc,nk->nc", directions[ids], bases)
        predicted_dot[ids] += (predicted * tangent).sum(dim=-1)
        predicted_norm_sq[ids] += predicted.square().sum(dim=-1)
        del outputs, ids, bases, scale, tangent, predicted
    overlap = predicted_dot.abs() / torch.sqrt(predicted_norm_sq * tangent_norm_sq).clamp_min(EPS)
    valid = (
        (support >= 2)
        & (shared_norm > EPS)
        & (tangent_norm_sq > EPS)
        & (predicted_norm_sq > EPS)
        & torch.isfinite(overlap)
        & torch.isfinite(targets)
    )
    gates = torch.zeros_like(overlap)
    gates[valid] = ((overlap[valid] - OVERLAP_THRESHOLD) / (1.0 - OVERLAP_THRESHOLD)).clamp(0, 1)
    active = gates > 0
    if was_training:
        model.train()
    return {
        "directions": directions.detach(),
        "targets": targets.detach(),
        "gates": gates.detach(),
        "gaussian_count": count,
        "training_view_count": len(train_records),
        "finite_direction_count": int(valid.sum()),
        "active_gaussians": int(active.sum()),
        "active_gate_fraction": float(active.float().mean()),
        "mean_gate": float(gates.mean()),
        "median_overlap": float(overlap[valid].median()) if bool(valid.any()) else 0.0,
        "minimum_support": int(support[valid].min()) if bool(valid.any()) else 0,
        "maximum_support": int(support[valid].max()) if bool(valid.any()) else 0,
    }


def _refresh_controller_rng_preserved(
    model: Any,
    train_records: Sequence[Tuple[int, str, Any, Any]],
) -> Dict[str, Any]:
    state = _rng_state()
    try:
        return _refresh_controller(model, train_records)
    finally:
        _set_rng_state(state)


def _module_loss(model: Any, controller: Optional[Mapping[str, Any]]) -> Tuple[Tensor, Dict[str, Any]]:
    zero = model.features_rest.sum() * 0.0
    if controller is None:
        return zero, {
            "enabled": False,
            "active_gaussians": 0,
            "active_gate_fraction": 0.0,
            "mean_gate": 0.0,
            "median_overlap": 0.0,
            "raw_loss": 0.0,
        }
    if int(controller["gaussian_count"]) != int(model.num_points):
        raise RuntimeError("stale identifiability controller after topology change")
    projection = (model.features_rest * controller["directions"]).sum(dim=(1, 2))
    residual = projection - controller["targets"]
    gates = controller["gates"]
    loss = 0.5 * (gates * residual.square()).sum() / gates.sum().clamp_min(1.0)
    return loss, {
        "enabled": True,
        "active_gaussians": int(controller["active_gaussians"]),
        "active_gate_fraction": float(controller["active_gate_fraction"]),
        "mean_gate": float(controller["mean_gate"]),
        "median_overlap": float(controller["median_overlap"]),
        "raw_loss": float(loss.detach()),
    }


def _direct_gradient_rows(
    model: Any,
    module_loss: Tensor,
    arm: str,
    absolute_step: int,
) -> List[Dict[str, Any]]:
    names = ("means", "features_dc", "features_rest", "scales", "quats", "opacities")
    parameters = [getattr(model, name) for name in names]
    medium_mlp = list(model.medium_mlp.parameters())
    direction = list(model.direction_encoding.parameters())
    all_parameters = parameters + medium_mlp + direction
    grads = torch.autograd.grad(module_loss, all_parameters, retain_graph=True, allow_unused=True)
    grouped: Dict[str, List[Optional[Tensor]]] = {
        name: [gradient] for name, gradient in zip(names, grads[: len(names)])
    }
    offset = len(names)
    grouped["medium_mlp"] = list(grads[offset : offset + len(medium_mlp)])
    offset += len(medium_mlp)
    grouped["direction_encoding"] = list(grads[offset:])
    grouped["medium_branch"] = grouped["medium_mlp"] + grouped["direction_encoding"]
    rows = []
    for group, values in grouped.items():
        finite = [value.detach().float() for value in values if value is not None]
        norm = math.sqrt(sum(float(value.square().sum()) for value in finite))
        maximum = max((float(value.abs().max()) for value in finite if value.numel()), default=0.0)
        rows.append(
            {
                "arm": arm,
                "absolute_step": absolute_step,
                "parameter_group": group,
                "module_direct_gradient_l2": norm,
                "module_direct_gradient_max_abs": maximum,
                "has_direct_gradient": norm > 0.0,
            }
        )
    return rows


def _topology_row(arm: str, step: int, model: Any, cumulative: Mapping[str, int]) -> Dict[str, Any]:
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacities.detach()).reshape(-1).float().cpu()
    return {
        "arm": arm,
        "absolute_step": step,
        "gaussian_count": int(model.num_points),
        "opacity_mean": float(opacity.mean()),
        "opacity_median": float(opacity.median()),
        "split_count_cumulative": int(cumulative["split"]),
        "duplicate_count_cumulative": int(cumulative["duplicate"]),
        "prune_count_cumulative": int(cumulative["prune"]),
        "opacity_reset_count_cumulative": int(cumulative["reset"]),
    }


def _save_checkpoint(
    branch: Any,
    arm: str,
    step: int,
    output_dir: Path,
    source_payload: Mapping[str, Any],
) -> Path:
    path = output_dir / "checkpoints" / arm / f"step-{step:09d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": EXPERIMENT,
            "mechanism": MECHANISM,
            "branch": arm,
            "absolute_step": step,
            "model": branch.pipeline.model.state_dict(),
            "optimizers": _optimizer_state(branch),
            "schedulers": _scheduler_state(branch),
            "scalers": dict(branch.scalers),
            "ocmc_bundle": source_payload["ocmc_bundle"],
            "raoc_state": None,
            "rng_state": _rng_state(),
            "rng_manifest": _rng_manifest(_rng_state()),
            "metadata": {
                "scene": output_dir.name,
                "common_start_step": START_STEP,
                "module_enabled": arm == "C1",
                "module_strength": MODULE_STRENGTH if arm == "C1" else 0.0,
                "matched_camera_sequence": True,
                "ocmc_frozen": True,
                "raoc_enabled": False,
            },
        },
        path,
    )
    return path


def _checkpoint_path(scene: str, arm: str, step: int, output_dir: Path) -> Path:
    if step == START_STEP:
        return _source_checkpoint(scene)
    return output_dir / "checkpoints" / arm / f"step-{step:09d}.ckpt"


def _load_snapshot(branch: Any, scene: str, arm: str, step: int, output_dir: Path) -> Mapping[str, Any]:
    payload = torch.load(_checkpoint_path(scene, arm, step, output_dir), map_location="cpu")
    branch.pipeline.model.load_state_dict(payload["model"], strict=True)
    branch.pipeline.model.step = step
    FORMAL._configure_model(branch.pipeline.model, "C0")
    FORMAL._install_condition(branch.pipeline.model, "C0", payload["ocmc_bundle"], None)
    _assert_locked_config(branch.pipeline.model)
    branch.pipeline.eval()
    return payload


def _train_arm(
    scene: str,
    arm: str,
    rows: Sequence[Mapping[str, Any]],
    snapshot_steps: Sequence[int],
    training_rng: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    branch = None
    started = time.perf_counter()
    training_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    topology_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    refresh_rows: List[Dict[str, Any]] = []
    snapshot_set = set(snapshot_steps)
    cumulative = {"split": 0, "duplicate": 0, "prune": 0, "reset": 0}
    completed = 0
    try:
        branch, source_payload = _load_branch(scene, arm)
        _set_rng_state(training_rng)
        model = branch.pipeline.model
        dm = branch.pipeline.datamanager
        cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
        cached = dm.cached_train
        train_records = FORMAL._train_records(branch.pipeline)
        controller: Optional[Dict[str, Any]] = None
        if arm == "C1":
            refresh_started = time.perf_counter()
            controller = _refresh_controller_rng_preserved(model, train_records)
            refresh_rows.append(
                {
                    "arm": arm,
                    "absolute_step": START_STEP,
                    "reason": "common_start",
                    "refresh_seconds": time.perf_counter() - refresh_started,
                    **{key: value for key, value in controller.items() if not isinstance(value, Tensor)},
                }
            )
        projector_hash_start = _hash_tensor(model._camera_medium_observability_projector)
        topology_rows.append(_topology_row(arm, START_STEP, model, cumulative))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for sequence_row in rows:
            absolute_step = int(sequence_row["absolute_step"])
            camera_index = int(sequence_row["camera_index"])
            camera_name = str(sequence_row["camera_name"])
            branch.pipeline.train()
            model.train()
            FORMAL._configure_model(model, "C0")
            MIC._run_before(model, branch.optimizers, absolute_step)
            branch.optimizers.zero_grad_all()
            batch = {
                key: value.to(model.device) if isinstance(value, Tensor) else value
                for key, value in cached[camera_index].copy().items()
            }
            camera = cameras[camera_index : camera_index + 1]
            iteration_started = time.perf_counter()
            outputs = model.get_outputs(camera)
            losses = model.get_loss_dict(outputs, batch, {})
            base_loss = sum(losses.values())
            module_raw, module_info = _module_loss(model, controller)
            module_weighted = MODULE_STRENGTH * module_raw if arm == "C1" else module_raw
            total_loss = base_loss + module_weighted
            if not bool(torch.isfinite(total_loss)) or not FORMAL._finite_outputs(outputs):
                raise RuntimeError(f"non-finite training state at {scene}/{arm}/{absolute_step}")
            audit_gradient = absolute_step == START_STEP + 1 or absolute_step in snapshot_set
            if audit_gradient:
                audit_row_start = len(gradient_rows)
                gradient_rows.extend(_direct_gradient_rows(model, module_weighted, arm, absolute_step))
            total_loss.backward()
            if audit_gradient:
                total_stats = MIC._param_group_grad_stats(model)
                for audit_row in gradient_rows[audit_row_start:]:
                    stats = total_stats[audit_row["parameter_group"]]
                    audit_row["total_loss_gradient_l2"] = stats["grad_l2"]
                    audit_row["total_loss_gradient_max_abs"] = stats["grad_max_abs"]
                    audit_row["total_loss_has_gradient"] = stats["has_grad"]
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(absolute_step)
            event = MIC._run_after(model, branch.optimizers, absolute_step)
            if event.get("refinement_called"):
                event_row = {"arm": arm, "absolute_step": absolute_step, "camera_name": camera_name, **dict(event)}
                event_rows.append(event_row)
                cumulative["split"] += int(event.get("K_split", 0))
                cumulative["duplicate"] += int(event.get("K_duplicate", 0))
                cumulative["prune"] += int(event.get("N_pruned", 0))
                cumulative["reset"] += int(bool(event.get("opacity_reset", False)))
                if arm == "C1":
                    refresh_started = time.perf_counter()
                    controller = _refresh_controller_rng_preserved(model, train_records)
                    refresh_rows.append(
                        {
                            "arm": arm,
                            "absolute_step": absolute_step,
                            "reason": (
                                "topology_change"
                                if int(event.get("N_before", model.num_points))
                                != int(event.get("N_after", model.num_points))
                                else "periodic_refinement_boundary"
                            ),
                            "refresh_seconds": time.perf_counter() - refresh_started,
                            **{key: value for key, value in controller.items() if not isinstance(value, Tensor)},
                        }
                    )
            completed += 1
            if absolute_step % LOG_INTERVAL == 0 or absolute_step in snapshot_set or completed == 1:
                pred = outputs["pred_image"].detach().float().clamp(0, 1)
                gt = AUDIT.VC.MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
                metrics = MIC._metric_images(model, pred, gt)
                training_rows.append(
                    {
                        "arm": arm,
                        "absolute_step": absolute_step,
                        "camera_index": camera_index,
                        "camera_name": camera_name,
                        "L_base": float(base_loss.detach()),
                        "L_ident_raw": float(module_raw.detach()),
                        "module_strength": MODULE_STRENGTH if arm == "C1" else 0.0,
                        "L_total": float(total_loss.detach()),
                        "gaussian_count": int(model.num_points),
                        "iteration_seconds": time.perf_counter() - iteration_started,
                        **module_info,
                        **metrics,
                    }
                )
            if absolute_step in snapshot_set:
                topology_rows.append(_topology_row(arm, absolute_step, model, cumulative))
                checkpoint = _save_checkpoint(branch, arm, absolute_step, output_dir, source_payload)
                checkpoint_rows.append(
                    {
                        "arm": arm,
                        "absolute_step": absolute_step,
                        "path": str(checkpoint),
                        "size_bytes": checkpoint.stat().st_size,
                        "sha256": _sha256(checkpoint),
                    }
                )
                _write_csv(output_dir / f"{arm}_training_progress.csv", training_rows)
                _write_csv(output_dir / f"{arm}_gradient_progress.csv", gradient_rows)
                _write_csv(output_dir / f"{arm}_topology_progress.csv", topology_rows)
            del outputs, losses, base_loss, module_raw, module_weighted, total_loss, batch
            if absolute_step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        projector_hash_end = _hash_tensor(model._camera_medium_observability_projector)
        result = {
            "arm": arm,
            "completed_updates": completed,
            "first_absolute_step": int(rows[0]["absolute_step"]),
            "final_absolute_step": int(rows[-1]["absolute_step"]),
            "training_wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "ocmc_projector_hash_start": projector_hash_start,
            "ocmc_projector_hash_end": projector_hash_end,
            "ocmc_projector_unchanged": projector_hash_start == projector_hash_end,
            "controller_refresh_count": len(refresh_rows),
            "controller_refresh_seconds": sum(float(row["refresh_seconds"]) for row in refresh_rows),
            "training_rows": training_rows,
            "gradient_rows": gradient_rows,
            "topology_rows": topology_rows,
            "event_rows": event_rows,
            "checkpoint_rows": checkpoint_rows,
            "refresh_rows": refresh_rows,
        }
        return result
    except Exception as exc:
        _write_json(
            output_dir / f"{arm}_failure.json",
            {"scene": scene, "arm": arm, "completed_updates": completed, "error": repr(exc)},
        )
        raise
    finally:
        _release(branch)


def _metric_records(branch: Any, split: str) -> Sequence[Tuple[int, str, Any, Mapping[str, Any]]]:
    return FORMAL._train_records(branch.pipeline) if split == "train" else FORMAL._eval_records(branch.pipeline)


def _evaluate_loaded(branch: Any, arm: str, step: int, split: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    model = branch.pipeline.model
    values = {key: [] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}
    per_view = []
    maps: Dict[str, Mapping[str, Tensor]] = {}
    ocmc_sq = 0.0
    ocmc_count = 0
    for _index, view_id, camera, batch in _metric_records(branch, split):
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
            gt = AUDIT.VC.MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
        pred = outputs["pred_image"].detach().float().clamp(0, 1)
        metrics = MIC._metric_images(model, pred, gt)
        per_view.append({"arm": arm, "absolute_step": step, "split": split, "view_id": view_id, **metrics})
        for key in values:
            values[key].append(metrics[key])
        maps[view_id] = {
            **{key: value.detach().float().cpu() for key, value in outputs.items() if isinstance(value, Tensor)},
            "pred": pred.cpu(),
            "gt": gt.cpu(),
        }
        projected = outputs.get("camera_medium_delta_projected_raw")
        if isinstance(projected, Tensor):
            ocmc_sq += float(projected.detach().float().square().sum())
            ocmc_count += int(projected.numel())
        del outputs, gt, pred
    global_row = {
        "arm": arm,
        "absolute_step": step,
        "split": split,
        "view_count": len(per_view),
        **{key: sum(items) / len(items) for key, items in values.items()},
    }
    decomp = CAM._decomposition_row(arm, step, split, maps)
    ocmc = {
        "arm": arm,
        "absolute_step": step,
        "split": split,
        "ocmc_projected_raw_rms": math.sqrt(ocmc_sq / max(ocmc_count, 1)),
        "sample_count": ocmc_count,
    }
    return global_row, per_view, {"decomposition": decomp, "ocmc": ocmc}


def _opacity_stats(opacity: Tensor) -> Dict[str, Any]:
    flat = opacity.detach().float().reshape(-1).cpu()
    quantiles = torch.quantile(flat, torch.tensor([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]))
    histogram = torch.histc(flat, bins=20, min=0.0, max=1.0).to(torch.int64)
    return {
        "opacity_mean": float(flat.mean()),
        "opacity_std": float(flat.std(unbiased=False)),
        **{f"opacity_q{label}": float(value) for label, value in zip(("01", "10", "25", "50", "75", "90", "99"), quantiles)},
        "opacity_histogram_20_bins": histogram.tolist(),
    }


def _mechanism_loaded(
    branch: Any,
    scene: str,
    arm: str,
    step: int,
    sample_count: int,
) -> Tuple[Dict[str, Any], Dict[str, Tensor]]:
    model = branch.pipeline.model
    train_records = FORMAL._train_records(branch.pipeline)
    support = AUDIT.VC._support_counts(model, train_records)
    available = int((support >= 2).sum())
    selected_count = min(sample_count, available)
    selected, sampling = AUDIT._sample_gaussians(scene, step, support, selected_count)
    observations = PREFLIGHT._collect_observations(model, train_records, selected)
    ids = selected.to(model.device)
    rest = model.features_rest.detach()[ids].float().clone()
    dc = model.features_dc.detach()[ids].float().clone()
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)[ids].float().clone()
    directions, targets, gates, direction_summary = PREFLIGHT._directions_and_gates(
        rest, dc, opacity, observations
    )
    response, tangent, _color, _transmission = PREFLIGHT._responses(rest, dc, opacity, observations)
    projection = (rest * directions).sum(dim=(1, 2))
    orthogonal = rest - projection[:, None, None] * directions
    active = gates > 0
    weights = gates[active]

    def weighted(values: Tensor) -> float:
        if not bool(active.any()):
            return float("nan")
        return float((weights * values[active]).sum() / weights.sum().clamp_min(EPS))

    row = {
        "arm": arm,
        "absolute_step": step,
        "sample_count": int(selected.numel()),
        "eligible_count": available,
        "selected_ids_sha256": sampling["selected_ids_sha256"],
        "finite_direction_count": direction_summary["finite_direction_count"],
        "median_tangent_overlap": direction_summary["median_overlap"],
        "active_gate_fraction": direction_summary["active_gate_fraction"],
        "SH_DC_energy": float(dc.square().sum(dim=-1).mean()),
        "SH_nonDC_energy": float(rest.square().sum(dim=(1, 2)).mean()),
        "SH_total_energy": float(dc.square().sum(dim=-1).mean() + rest.square().sum(dim=(1, 2)).mean()),
        "SH_shared_coefficient_energy": weighted(projection.square()),
        "SH_orthogonal_energy": weighted(orthogonal.square().sum(dim=(1, 2))),
        "SH_opacity_shared_response_energy": PREFLIGHT._shared_response_energy(response, tangent, gates),
        **_opacity_stats(torch.sigmoid(model.opacities.detach())),
    }
    state = {
        "selected": selected,
        "rest": rest,
        "directions": directions,
        "targets": targets,
        "gates": gates,
    }
    return row, state


def _counterfactual_loaded(
    branch: Any,
    arm: str,
    state: Mapping[str, Tensor],
) -> List[Dict[str, Any]]:
    model = branch.pipeline.model
    selected = state["selected"].to(model.device)
    before = model.features_rest.detach()[selected].clone()
    controlled = PREFLIGHT._controlled_coefficients(
        state["rest"], state["directions"], state["targets"], state["gates"], 1.0
    ).to(model.device, dtype=model.features_rest.dtype)
    projection_before = (state["rest"] * state["directions"]).sum(dim=(1, 2))
    projection_after = (controlled.float() * state["directions"]).sum(dim=(1, 2))
    orth_before = state["rest"] - projection_before[:, None, None] * state["directions"]
    orth_after = controlled.float() - projection_after[:, None, None] * state["directions"]
    rows = []
    try:
        full_metrics: Dict[str, Dict[str, float]] = {}
        for _index, view_id, camera, batch in FORMAL._eval_records(branch.pipeline):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = AUDIT.VC.MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
            pred = outputs["pred_image"].detach().float().clamp(0, 1)
            full_metrics[view_id] = MIC._metric_images(model, pred, gt)
        with torch.no_grad():
            model.features_rest[selected] = controlled
        for _index, view_id, camera, batch in FORMAL._eval_records(branch.pipeline):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = AUDIT.VC.MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
            pred = outputs["pred_image"].detach().float().clamp(0, 1)
            removed = MIC._metric_images(model, pred, gt)
            full = full_metrics[view_id]
            row = {
                "arm": arm,
                "absolute_step": FINAL_STEP,
                "split": "eval",
                "view_id": view_id,
                "counterfactual": "training_anchored_sampled_shared_component_removal",
                "sample_count": int(selected.numel()),
                "orthogonal_relative_drift": float(
                    torch.linalg.vector_norm(orth_after - orth_before)
                    / torch.linalg.vector_norm(orth_before).clamp_min(EPS)
                ),
            }
            for key in ("PSNR", "SSIM", "LPIPS", "MSE"):
                row[f"FULL_{key}"] = full[key]
                row[f"SHARED_REMOVED_{key}"] = removed[key]
                row[f"SHARED_REMOVED_minus_FULL_{key}"] = removed[key] - full[key]
            rows.append(row)
    finally:
        with torch.no_grad():
            model.features_rest[selected] = before
    return rows


def _postprocess(
    scene: str,
    output_dir: Path,
    diagnostic_steps: Sequence[int],
    sample_count: int,
) -> Dict[str, Any]:
    global_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    mechanism_rows: List[Dict[str, Any]] = []
    decomposition_rows: List[Dict[str, Any]] = []
    ocmc_rows: List[Dict[str, Any]] = []
    counterfactual_rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        branch = None
        try:
            branch, _payload = _load_branch(scene, arm, load_optimizer=False)
            for step in diagnostic_steps:
                payload = _load_snapshot(branch, scene, arm, step, output_dir)
                for split in ("train", "eval"):
                    global_row, view_rows, extra = _evaluate_loaded(branch, arm, step, split)
                    global_rows.append(global_row)
                    per_view_rows.extend(view_rows)
                    decomposition_rows.append(extra["decomposition"])
                    ocmc_rows.append(extra["ocmc"])
                mechanism, state = _mechanism_loaded(branch, scene, arm, step, sample_count)
                mechanism_rows.append(mechanism)
                if step == FINAL_STEP:
                    counterfactual_rows.extend(_counterfactual_loaded(branch, arm, state))
                if payload.get("raoc_state") is not None:
                    raise RuntimeError("RAOC state unexpectedly present")
        finally:
            _release(branch)

    _write_csv(output_dir / "evaluation_metrics.csv", global_rows)
    _write_csv(output_dir / "per_view_metrics.csv", per_view_rows)
    _write_csv(output_dir / "mechanism_metrics.csv", mechanism_rows)
    _write_csv(output_dir / "decomposition_safety.csv", decomposition_rows)
    _write_json(output_dir / "decomposition_safety.json", {"rows": decomposition_rows})
    _write_csv(output_dir / "ocmc_magnitude.csv", ocmc_rows)
    _write_csv(output_dir / "counterfactual_metrics.csv", counterfactual_rows)
    return {
        "global_rows": global_rows,
        "per_view_rows": per_view_rows,
        "mechanism_rows": mechanism_rows,
        "decomposition_rows": decomposition_rows,
        "ocmc_rows": ocmc_rows,
        "counterfactual_rows": counterfactual_rows,
    }


def run(
    scene: str,
    gpu: str,
    output_dir: Path,
    final_step: int = FINAL_STEP,
    diagnostic_sample_count: int = DIAGNOSTIC_SAMPLE_COUNT,
) -> Dict[str, Any]:
    if final_step <= START_STEP or final_step > FINAL_STEP:
        raise ValueError(f"final_step must be in [{START_STEP + 1}, {FINAL_STEP}]")
    formal = final_step == FINAL_STEP
    snapshot_steps = FORMAL_SNAPSHOT_STEPS if formal else (final_step,)
    diagnostic_steps = DIAGNOSTIC_STEPS if formal else (START_STEP, final_step)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _runtime(scene, gpu)
    continuation_seed = TRAINING_SEED
    training_rng = _seed_rng(continuation_seed)
    start_head = _git("rev-parse", "HEAD")
    branch_name = _git("branch", "--show-current")
    if branch_name != EXPECTED_BRANCH:
        raise RuntimeError(f"unexpected branch: {branch_name}")
    protected_hashes = _protected_hashes()
    worker_script_sha256 = _sha256(Path(__file__).resolve())
    manifest = {
        "experiment": EXPERIMENT,
        "scene": scene,
        "formal": formal,
        "START_HEAD": start_head,
        "branch": branch_name,
        "git_status": _git("status", "--short"),
        "worker_script": str(Path(__file__).resolve()),
        "worker_script_sha256": worker_script_sha256,
        "runtime": runtime,
        "common_start_step": START_STEP,
        "final_step": final_step,
        "matched_updates_per_arm": final_step - START_STEP,
        "snapshot_steps": list(snapshot_steps),
        "diagnostic_steps": list(diagnostic_steps),
        "module": {
            "name": MECHANISM,
            "strength": MODULE_STRENGTH,
            "overlap_threshold": OVERLAP_THRESHOLD,
            "C0_enabled": False,
            "C1_enabled": True,
            "direct_gradient_target": "features_rest",
            "uses_gt_for_control": False,
            "uses_heldout_for_control": False,
            "inference_time_path": False,
        },
        "preregistered_safety_gates": {
            "minimum_SH_nonDC_and_orthogonal_capacity_ratio_C1_over_C0": MIN_CAPACITY_RATIO,
            "minimum_opacity_mean_and_median_ratio_C1_over_C0": MIN_OPACITY_RATIO,
            "maximum_opacity_mean_and_median_ratio_C1_over_C0": MAX_OPACITY_RATIO,
            "maximum_final_population_relative_gap": MAX_FINAL_POPULATION_RELATIVE_GAP,
            "decomposition_P_J_gt_1": 0.0,
        },
        "source_checkpoint": str(_source_checkpoint(scene)),
        "source_checkpoint_sha256": _sha256(_source_checkpoint(scene)),
        "source_camera_sequence": str(_source_sequence(scene)),
        "source_camera_sequence_sha256": _sha256(_source_sequence(scene)),
        "continuation_seed": continuation_seed,
        "continuation_rng_manifest": _rng_manifest(training_rng),
        "protected_hashes": protected_hashes,
        "disk_before": subprocess.check_output(["df", "-B1", str(REPO_ROOT)], text=True).splitlines()[-1],
    }
    expected_checkpoint = EXPECTED_SOURCE_CHECKPOINT_HASHES[scene]
    expected_sequence = AUDIT.VC.EXPECTED_CAMERA_SEQUENCE_HASHES[scene]
    if manifest["source_checkpoint_sha256"] != expected_checkpoint:
        raise RuntimeError("registered step-3000 checkpoint hash drift")
    if manifest["source_camera_sequence_sha256"] != expected_sequence:
        raise RuntimeError("registered camera sequence hash drift")
    _write_json(output_dir / "run_manifest.json", manifest)
    sequence = _camera_sequence(scene, final_step, output_dir)
    start = _start_state_audit(scene, output_dir, training_rng)
    arm_results = {}
    for arm in ARMS:
        print(f"[{scene}] {arm}: {len(sequence)} matched continuation updates", flush=True)
        arm_results[arm] = _train_arm(
            scene, arm, sequence, snapshot_steps, training_rng, output_dir
        )
    training_rows = [row for arm in ARMS for row in arm_results[arm]["training_rows"]]
    gradient_rows = [row for arm in ARMS for row in arm_results[arm]["gradient_rows"]]
    topology_rows = [row for arm in ARMS for row in arm_results[arm]["topology_rows"]]
    event_rows = [row for arm in ARMS for row in arm_results[arm]["event_rows"]]
    checkpoint_rows = [row for arm in ARMS for row in arm_results[arm]["checkpoint_rows"]]
    refresh_rows = [row for arm in ARMS for row in arm_results[arm]["refresh_rows"]]
    _write_csv(output_dir / "training_metrics.csv", training_rows)
    _write_csv(output_dir / "gradient_audit.csv", gradient_rows)
    _write_csv(output_dir / "topology_metrics.csv", topology_rows)
    _write_csv(output_dir / "refinement_events.csv", event_rows)
    _write_csv(output_dir / "checkpoint_manifest.csv", checkpoint_rows)
    _write_csv(output_dir / "controller_refresh.csv", refresh_rows)
    training_summary = {
        "scene": scene,
        "formal": formal,
        "START_STATE_EQUIVALENCE": start["START_STATE_EQUIVALENCE"],
        "CAMERA_SEQUENCE_EXACT_MATCH": True,
        "camera_mismatch_count": 0,
        "camera_sequence_hash": _read_json(output_dir / "camera_sequence.json")["sha256_C0"],
        "matched_updates_per_arm": len(sequence),
        "arms": {
            arm: {
                key: value
                for key, value in arm_results[arm].items()
                if key
                not in (
                    "training_rows",
                    "gradient_rows",
                    "topology_rows",
                    "event_rows",
                    "checkpoint_rows",
                    "refresh_rows",
                )
            }
            for arm in ARMS
        },
    }
    _write_json(output_dir / "training_summary.json", training_summary)
    print(f"[{scene}] evaluating temporal and mechanism metrics", flush=True)
    post = _postprocess(scene, output_dir, diagnostic_steps, diagnostic_sample_count)
    if _sha256(Path(__file__).resolve()) != worker_script_sha256:
        raise RuntimeError("worker script changed during execution")
    if _protected_hashes() != protected_hashes:
        raise RuntimeError("protected source changed during execution")
    if _sha256(_source_checkpoint(scene)) != manifest["source_checkpoint_sha256"]:
        raise RuntimeError("source checkpoint changed during execution")
    if _sha256(_source_sequence(scene)) != manifest["source_camera_sequence_sha256"]:
        raise RuntimeError("source camera sequence changed during execution")
    final = {
        "scene": scene,
        "formal": formal,
        "training_summary": training_summary,
        "evaluation_rows": len(post["global_rows"]),
        "per_view_rows": len(post["per_view_rows"]),
        "mechanism_rows": len(post["mechanism_rows"]),
        "decomposition_rows": len(post["decomposition_rows"]),
        "counterfactual_rows": len(post["counterfactual_rows"]),
        "integrity": {
            "worker_script_unchanged": True,
            "protected_hashes_unchanged": True,
            "source_checkpoint_unchanged": True,
            "source_camera_sequence_unchanged": True,
        },
        "disk_after": subprocess.check_output(["df", "-B1", str(REPO_ROOT)], text=True).splitlines()[-1],
    }
    _write_json(output_dir / "scene_complete.json", final)
    print(json.dumps(_sanitize(final), indent=2, sort_keys=True), flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--gpu", choices=tuple(sorted(ALLOWED_GPUS)), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-step", type=int, default=FINAL_STEP)
    parser.add_argument("--diagnostic-sample-count", type=int, default=DIAGNOSTIC_SAMPLE_COUNT)
    args = parser.parse_args()
    run(
        args.scene,
        args.gpu,
        args.output_dir.resolve(),
        final_step=args.final_step,
        diagnostic_sample_count=args.diagnostic_sample_count,
    )


if __name__ == "__main__":
    main()
