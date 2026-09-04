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


EXPERIMENT = "DIRECT_TRAINING_MDRR_CICA_AND_COMBINED_FOUR_SCENE"
ARMS = ("A1", "A2", "A3")
START_STEP = 3000
FINAL_STEP = 14999
SNAPSHOT_STEPS = (5000, 8000, 10000, 13000, 14999)
TRAINING_SEED = 42
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
AUXILIARY_APPEARANCE_ENABLED = True
AUXILIARY_APPEARANCE_STRENGTH = 1.0
EPS = 1e-12
SH_C0 = 0.28209479177387814
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
SOURCE_CONFIG = REPO_ROOT / "outputs" / "m1_ocmc_causal_iui3_20260825" / "source_bnd_config.yml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "direct_mdrr_cica_four_scene_20260903"
RENDER_ROOT = REPO_ROOT / "renders" / "direct_mdrr_cica_four_scene_20260903"
ALLOWED_GPUS = frozenset(("6", "7", "8", "9"))
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}


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
    if scene not in SCENE_GPUS or SCENE_GPUS[scene] != str(gpu):
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
    return SOURCE_ROOT / scene / "checkpoints" / "C0" / f"step-{START_STEP:09d}.ckpt"


def _source_sequence(scene: str) -> Path:
    return SOURCE_ROOT / scene / "camera_sequence.json"


def _scene_config(scene: str) -> Dict[str, Any]:
    return {
        "data_path": f"undistorted_data/undistorted_{scene}",
        "source_checkpoint": str(_source_checkpoint(scene)),
    }


def _configure_model_config(config: Any) -> None:
    model = config.pipeline.model
    model.intrinsic_color_parameterization = "bounded_sh3"
    model.sh_degree = 3
    model.rasterize_mode = "classic"
    model.medium_context_mode = "dir_xy_camera"
    model.b_inf_mode = "tied"
    model.infinite_water_enabled = False
    model.camera_medium_observability_enabled = True
    model.camera_medium_ray_adaptive_observability_enabled = False
    model.medium_identifiability_enabled = False
    model.medium_identifiability_weight = 0.0
    model.coarse_depth_supervision_enabled = False
    model.stop_split_at = 10000
    model.appearance_lr_scale = 1.0


def _configure_model(model: Any) -> None:
    model.config.intrinsic_color_parameterization = "bounded_sh3"
    model.config.sh_degree = 3
    model.config.rasterize_mode = "classic"
    model.config.medium_context_mode = "dir_xy_camera"
    model.config.b_inf_mode = "tied"
    model.config.infinite_water_enabled = False
    model.config.camera_medium_observability_enabled = True
    model.config.camera_medium_ray_adaptive_observability_enabled = False
    model.config.medium_identifiability_enabled = False
    model.config.medium_identifiability_weight = 0.0
    model.config.coarse_depth_supervision_enabled = False
    model.config.stop_split_at = 10000
    model.config.appearance_lr_scale = 1.0


def _new_branch(scene: str) -> Any:
    config = yaml.load(SOURCE_CONFIG.read_text(encoding="utf8"), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = None
    config.load_step = None
    config.load_checkpoint = None
    config.pipeline.datamanager.load_depths = False
    config.pipeline.datamanager.dataparser.data = REPO_ROOT / _scene_config(scene)["data_path"]
    _configure_model_config(config)
    _set_random_seed(int(config.machine.seed))
    pipeline = config.pipeline.setup(device=torch.device("cuda:0"), test_mode="test")
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"unexpected pipeline type: {type(pipeline)}")
    _configure_model(pipeline.model)
    pipeline.model.clear_camera_medium_ray_adaptive_observability_state()
    pipeline.model.set_camera_medium_observability_projector(None)
    pipeline.model.step = 0
    optimizers = Optimizers(MIC._optimizer_groups(config, pipeline.model), pipeline.model.get_param_groups())
    pipeline.eval()
    return type("Branch", (), {"config": config, "pipeline": pipeline, "optimizers": optimizers, "scalers": {}})()


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
    checkpoint = _source_checkpoint(scene)
    expected = {
        "Curasao": "06eeaa97550a5a49abdf483e291e637b58f1e6c608f7743935ebe57fbbaed950",
        "IUI3-RedSea": "73b7c7f5739b44950e63f815c0f24873db3fb00e68a79ac8133e3e6a298e86b3",
        "JapaneseGradens-RedSea": "5f52b47704b075a72de3ed14ef0147fdad194cd2311756325fc5e9338df266a4",
        "Panama": "f1b9a515c6cc4ffd30698d50f42f7b01291a3238dd9d38b33e05294b22cd4a23",
    }[scene]
    if _sha256(checkpoint) != expected:
        raise RuntimeError(f"registered source checkpoint hash drift: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("branch") != "C0" or int(payload.get("absolute_step", -1)) != START_STEP:
        raise RuntimeError("source checkpoint is not OCMC C0@3000")
    if payload.get("raoc_state") is not None or payload.get("ocmc_bundle") is None:
        raise RuntimeError("source checkpoint must contain OCMC and no RAOC state")
    branch.pipeline.model.load_state_dict(payload["model"], strict=True)
    branch.pipeline.model.step = START_STEP
    _configure_model(branch.pipeline.model)
    FORMAL._install_condition(branch.pipeline.model, "C0", payload["ocmc_bundle"], None)
    branch.optimizers = Optimizers(MIC._optimizer_groups(branch.config, branch.pipeline.model), branch.pipeline.model.get_param_groups())
    if set(branch.optimizers.optimizers) != set(payload["optimizers"]):
        raise RuntimeError("optimizer group mismatch at source checkpoint")
    for group, optimizer in branch.optimizers.optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][group])
    for group, scheduler in branch.optimizers.schedulers.items():
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


def _optimizer_hash(optimizers: Optimizers) -> str:
    return _hash_object({key: _canonical_optimizer_state(value.state_dict()) for key, value in optimizers.optimizers.items()})


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


def _scheduler_hash(optimizers: Optimizers) -> str:
    return _hash_object({key: value.state_dict() for key, value in optimizers.schedulers.items()})


def _locked_config(model: Any) -> Dict[str, Any]:
    cfg = model.config
    return {
        "intrinsic_color_parameterization": cfg.intrinsic_color_parameterization,
        "sh_degree": int(cfg.sh_degree),
        "rasterize_mode": cfg.rasterize_mode,
        "medium_context_mode": cfg.medium_context_mode,
        "b_inf_mode": cfg.b_inf_mode,
        "infinite_water_enabled": bool(cfg.infinite_water_enabled),
        "camera_medium_observability_enabled": bool(cfg.camera_medium_observability_enabled),
        "camera_medium_ray_adaptive_observability_enabled": bool(cfg.camera_medium_ray_adaptive_observability_enabled),
        "stop_split_at": int(cfg.stop_split_at),
        "appearance_lr_scale": float(getattr(cfg, "appearance_lr_scale", 1.0)),
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
    checkpoint_model_hash = _hash_object({key: value.detach().cpu() for key, value in payload["model"].items()})
    loaded_model_hash = _model_state_hash(model)
    optimizer_hash = _optimizer_hash(branch.optimizers)
    scheduler_hash = _scheduler_hash(branch.optimizers)
    source_optimizer_hash = _hash_object(
        {key: _canonical_optimizer_state(value) for key, value in payload["optimizers"].items()}
    )
    source_scheduler_hash = _hash_object(payload["schedulers"])
    scaler_hash = _hash_object(branch.scalers)
    source_scaler_hash = _hash_object(payload.get("scalers", {}))
    rng_hash = _rng_manifest(training_rng)
    projector_hash = _hash_tensor(model._camera_medium_observability_projector)
    forward_probe = _forward_probe(model, branch.pipeline.datamanager)
    audit = {
        "scene": scene,
        "arm": arm,
        "START_STATE_EQUIVALENCE": bool(
            checkpoint_model_hash == loaded_model_hash
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
        "ocmc_enabled": True,
        "scaler_state_present": bool(payload.get("scalers", {})),
        "rng_provenance": "seed-42 continuation RNG is restored before every arm; source checkpoint stores only RNG hashes",
        "forward_probe": forward_probe,
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


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _gt(model: Any, batch: Mapping[str, Any], background: Tensor) -> Tensor:
    return model.composite_with_background(model.get_gt_img(batch["image"]), background)


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
    torch::Tensor residual, int block_width);
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
    const float* residual, float* numerator, float* information) {
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
                const float jacobian = visibility * attenuation * color * (1.0f - color) * SH_C0;
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
    torch::Tensor residual, int block_width) {
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
        residual.data_ptr<float>(), numerator.data_ptr<float>(), information.data_ptr<float>());
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
    )
    finite = torch.isfinite(numerator).all(dim=-1) & torch.isfinite(information).all(dim=-1)
    information_sum = information.sum(dim=-1)
    valid = finite & (information_sum > CICA_INFORMATION_FLOOR)
    # The CUDA accumulator differentiates with respect to features_dc. Convert
    # the least-squares feature update back to the degree-zero SH logit domain.
    delta_dc_logits = SH_C0 * numerator / information.clamp_min(CICA_INFORMATION_FLOOR)
    delta_dc_logits = torch.where(valid[:, None], delta_dc_logits, torch.zeros_like(delta_dc_logits))
    return {
        "delta_dc_logits": delta_dc_logits.detach(),
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
    with torch.no_grad():
        was_training = model.training
        model.eval()
        partner_outputs = model.get_outputs_for_camera(partner_camera.to(model.device))
        if was_training:
            model.train()
        partner_gt = _gt(model, partner_batch, partner_outputs["background"]).detach().float()
        partner_pred = partner_outputs["pred_image"].detach().float()
        partner_residual = partner_gt - partner_pred
        partner_add = partner_outputs["rgb_medium_finite"].detach().float() + partner_outputs["rgb_tail"].detach().float()
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
        dc = outputs["gaussian_view_dc_rgb"].detach().float()
        dc_logits = outputs["gaussian_view_dc_logits"].detach().float()
        delta_logits = evidence["delta_dc_logits"]
        corrected_dc = torch.sigmoid(dc_logits + delta_logits).clamp(1e-6, 1.0 - 1e-6)
        information = torch.where(visible, evidence["information_sum"], torch.zeros_like(evidence["information_sum"]))
        chroma = torch.stack(
            [torch.log((corrected_dc[:, 0] + 1e-6) / (corrected_dc[:, 1] + 1e-6)), torch.log((corrected_dc[:, 2] + 1e-6) / (corrected_dc[:, 1] + 1e-6))], dim=-1
        ).double()
        chroma = torch.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
        chroma_views.append(chroma)
        jacobian_views.append(information.double())
        support += visible.to(torch.int16)
        del outputs, gt, residual, evidence, dc, dc_logits, delta_logits, corrected_dc, information, chroma
    valid = support >= 3
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
) -> Dict[str, float]:
    appearance = set(groups.get("features_dc", [])) | set(groups.get("features_rest", []))
    medium_ids = set(groups.get("medium_mlp", [])) | set(groups.get("direction_encoding", []))
    geometry = set(groups.get("means", [])) | set(groups.get("scales", [])) | set(groups.get("quats", [])) | set(groups.get("opacities", []))
    features_dc = set(groups.get("features_dc", []))
    features_rest = set(groups.get("features_rest", []))
    for name, gradients, allowed in (
        ("auxiliary appearance", auxiliary, features_rest),
        ("CICA", cica, features_dc),
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
        model.step = CICA_START_STEP
        camera_index, _view_id, camera, batch = records[0]
        partner_index = int(partner["rows"][camera_index]["partner_index"])
        partner_record = records[partner_index]
        outputs = model.get_outputs(camera.to(model.device))
        losses = model.get_loss_dict(outputs, _batch_to_device(batch.copy(), model.device), {})
        base_loss = sum(losses.values())
        model.eval()
        auxiliary_controller = AUX._refresh_controller_rng_preserved(model, records)
        auxiliary_raw, _auxiliary_info = AUX._module_loss(model, auxiliary_controller)
        cica_controller = _build_cica_controller(
            model,
            [records[int(index)] for index in bank["camera_indices"]],
        )
        model.train()
        cica_raw, cica_info = _cica_loss(model, cica_controller)
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
        combined_info = _combine_gradients(
            params,
            group_indices,
            base_grads,
            surface_grads,
            medium_grads,
            auxiliary_grads,
            cica_grads,
            True,
        )
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
        }
        auxiliary_escape = combined_info["auxiliary_escape_grad_l2"]
        cica_escape = combined_info["cica_escape_grad_l2"]
        finite = all(
            value is None or bool(torch.isfinite(value).all().item())
            for gradients in (base_grads, surface_grads, medium_grads, auxiliary_grads, cica_grads)
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
            "absolute_step_probe": CICA_START_STEP,
            "optimizer_step_performed": False,
            "finite_direct_gradients": finite,
            "auxiliary_gradient_escape_l2": auxiliary_escape,
            "cica_gradient_escape_l2": cica_escape,
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
                "AUXILIARY_ONLY_FEATURES_REST": auxiliary_escape == 0.0,
                "CICA_ONLY_FEATURES_DC": cica_escape == 0.0,
                "MODULES_DO_NOT_DIRECTLY_CHANGE_GEOMETRY": geometry_diff <= 1e-12,
                "CICA_HAS_NONZERO_FEATURES_DC_GRADIENT": direct["CICA"].get("features_dc", 0.0) > EPS,
                "AUXILIARY_HAS_NONZERO_FEATURES_REST_GRADIENT": direct["auxiliary_appearance"].get("features_rest", 0.0) > EPS,
                "MDRR_SURFACE_ROUTE_ONLY_APPEARANCE": effective_surface_escape == 0.0 and _norm(effective_surface) > EPS,
                "MDRR_MEDIUM_ROUTE_ONLY_MEDIUM": effective_medium_escape == 0.0 and _norm(effective_medium) > EPS,
                "MDRR_PRESERVES_PHOTOMETRIC_LOSS_SCALE": route_loss_difference <= 1e-6,
            },
        }
        audit["GRADIENT_ROUTING_AUDIT"] = bool(
            audit["finite_direct_gradients"]
            and all(audit["assertions"].values())
        )
        rows: List[Dict[str, Any]] = []
        for route, values in direct.items():
            for group, norm_value in values.items():
                rows.append({"scene": scene, "arm": arm, "route": route, "parameter_group": group, "gradient_l2": norm_value})
        rows.extend(
            [
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "auxiliary_escape", "gradient_l2": auxiliary_escape},
                {"scene": scene, "arm": arm, "route": "audit", "parameter_group": "cica_escape", "gradient_l2": cica_escape},
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


def _save_checkpoint(branch: Any, scene: str, arm: str, step: int, output_dir: Path, source_payload: Mapping[str, Any], mdrr_hash: str, cica_hash: str, auxiliary_hash: str, lambda_cica: Optional[float]) -> Path:
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
            "ocmc_bundle": source_payload["ocmc_bundle"],
            "raoc_state": None,
            "metadata": {
                "matched_updates": int(step - START_STEP),
                "mdrr_enabled": arm in ("A1", "A3") and step >= MDRR_START_STEP,
                "cica_enabled": arm in ("A2", "A3") and step >= CICA_START_STEP,
                "mdrr_control_hash": mdrr_hash,
                "cica_target_hash": cica_hash,
                "auxiliary_appearance_target_hash": auxiliary_hash,
                "auxiliary_appearance_enabled": AUXILIARY_APPEARANCE_ENABLED,
                "auxiliary_appearance_strength": AUXILIARY_APPEARANCE_STRENGTH,
                "resolved_lambda_CICA": lambda_cica,
                "ocmc_frozen": True,
                "raoc_enabled": False,
                "normal_topology_enabled": True,
            },
        },
        path,
    )
    return path


def _train_arm(scene: str, arm: str, output_dir: Path, sequence: Sequence[Mapping[str, Any]], partner: Mapping[str, Any], bank: Mapping[str, Any], training_rng: Mapping[str, Any], max_step: int = FINAL_STEP) -> Dict[str, Any]:
    branch = None
    started = time.perf_counter()
    try:
        branch = _new_branch(scene)
        source_payload = _load_source(branch, scene)
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
        use_mdrr = arm in ("A1", "A3")
        use_cica = arm in ("A2", "A3")
        controller: Optional[Dict[str, Tensor]] = None
        auxiliary_controller: Optional[Dict[str, Any]] = None
        cica_controller: Optional[Dict[str, Tensor]] = None
        lambda_cica: Optional[float] = None
        mdrr_control_hash = ""
        mdrr_info: Dict[str, Any] = {"enabled": False}
        rows: List[Dict[str, Any]] = []
        module_rows: List[Dict[str, Any]] = []
        topology_rows: List[Dict[str, Any]] = [_topology_row(arm, START_STEP, model, {"split": 0, "duplicate": 0, "prune": 0, "reset": 0})]
        event_rows: List[Dict[str, Any]] = []
        checkpoints: List[Dict[str, Any]] = []
        cumulative = {"split": 0, "duplicate": 0, "prune": 0, "reset": 0}
        projector_hash_start = _hash_tensor(model._camera_medium_observability_projector)
        snapshot_steps = tuple(dict.fromkeys([step for step in SNAPSHOT_STEPS if step <= int(max_step)] + [int(max_step)]))
        sequence = [row for row in sequence if int(row["absolute_step"]) <= int(max_step)]
        if not sequence:
            raise RuntimeError("empty continuation sequence")
        if AUXILIARY_APPEARANCE_ENABLED:
            model.eval()
            auxiliary_controller = AUX._refresh_controller_rng_preserved(model, records)
            model.train()
            module_rows.append({"arm": arm, "absolute_step": START_STEP, "module": "AUXILIARY_APPEARANCE", "event": "refresh", "reason": "common_start", "active_gaussians": int(auxiliary_controller["active_gaussians"]), "target_hash": _hash_tensor(auxiliary_controller["targets"])})
        for sequence_row in sequence:
            step = int(sequence_row["absolute_step"])
            camera_index = int(sequence_row["camera_index"])
            camera = cameras[camera_index : camera_index + 1]
            batch = _batch_to_device(cached[camera_index].copy(), model.device)
            branch.pipeline.train()
            model.train()
            model.step_cb(step)
            model.aopt_before_train_iteration(branch.optimizers, step)
            model.medium_hold_before_train_iteration(branch.optimizers, step)
            branch.optimizers.zero_grad_all()

            if use_cica and step >= CICA_START_STEP and (step in CICA_REFRESH_STEPS or cica_controller is None or int(cica_controller["population"].item()) != int(model.num_points)):
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
            base_loss = sum(losses.values())
            if not bool(torch.isfinite(base_loss).item()) or not FORMAL._finite_outputs(outputs):
                raise RuntimeError(f"non-finite base state at {scene}/{arm}/{step}")

            auxiliary_raw, auxiliary_info = AUX._module_loss(model, auxiliary_controller if AUXILIARY_APPEARANCE_ENABLED else None)
            auxiliary_weighted = float(AUXILIARY_APPEARANCE_STRENGTH) * auxiliary_raw
            cica_raw, cica_info = _cica_loss(model, cica_controller if use_cica and step >= CICA_START_STEP else None)
            cica_weighted = cica_raw if lambda_cica is None else float(lambda_cica) * cica_raw
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

            if use_cica and step == CICA_START_STEP:
                base_probe = torch.autograd.grad(base_loss, [model.features_dc], retain_graph=True, allow_unused=True)[0]
                cica_probe = torch.autograd.grad(cica_raw, [model.features_dc], retain_graph=True, allow_unused=True)[0]
                base_norm = float(base_probe.detach().float().norm().item()) if base_probe is not None else 0.0
                cica_norm = float(cica_probe.detach().float().norm().item()) if cica_probe is not None else 0.0
                if cica_norm <= EPS or not math.isfinite(cica_norm):
                    raise RuntimeError("CICA calibration has zero/non-finite features_dc gradient")
                lambda_cica = CICA_GRADIENT_FRACTION * max(base_norm, EPS) / cica_norm
                cica_weighted = float(lambda_cica) * cica_raw
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
                cica_grads = torch.autograd.grad(cica_weighted, params, retain_graph=True, allow_unused=True) if use_cica and step >= CICA_START_STEP else [None] * len(params)
                grad_info = _combine_gradients(params, group_indices, base_grads, surface_grads, medium_grads, auxiliary_grads, cica_grads, True)
            else:
                params, group_indices = _parameter_entries(model)
                base_grads = torch.autograd.grad(base_loss, params, retain_graph=True, allow_unused=True)
                auxiliary_grads = torch.autograd.grad(auxiliary_weighted, params, retain_graph=True, allow_unused=True)
                cica_grads = torch.autograd.grad(cica_weighted, params, retain_graph=True, allow_unused=True) if use_cica and step >= CICA_START_STEP else [None] * len(params)
                grad_info = _combine_gradients(params, group_indices, base_grads, base_grads, base_grads, auxiliary_grads, cica_grads, False)

            total_loss_value = float((base_loss + auxiliary_weighted + cica_weighted).detach().item())
            branch.optimizers.optimizer_step_all()
            branch.optimizers.scheduler_step_all(step)
            model.aopt_after_train_iteration(step)
            model.medium_hold_after_train_iteration(branch.optimizers, step)
            model.after_train(step=step)
            event: Dict[str, Any]
            if step % int(model.config.refine_every) == 0:
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
                    if AUXILIARY_APPEARANCE_ENABLED:
                        was_training = model.training
                        model.eval()
                        auxiliary_controller = AUX._refresh_controller_rng_preserved(model, records)
                        if was_training:
                            model.train()
                        module_rows.append({"arm": arm, "absolute_step": step, "module": "AUXILIARY_APPEARANCE", "event": "refresh", "reason": "topology_change", "active_gaussians": int(auxiliary_controller["active_gaussians"]), "target_hash": _hash_tensor(auxiliary_controller["targets"])})
            else:
                event = {"refinement_called": False, "N_after": int(model.num_points)}

            if step % 500 == 0 or step in snapshot_steps or step == START_STEP + 1:
                pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
                target = _gt(model, batch, outputs["background"]).detach().float().clamp(0.0, 1.0)
                metrics = MIC._metric_images(model, pred, target)
                rows.append({
                    "arm": arm,
                    "absolute_step": step,
                    "camera_index": camera_index,
                    "camera_name": sequence_row["camera_name"],
                    "L_total": total_loss_value,
                    "L_RGB": float(base_loss.detach().item()),
                    "L_AUXILIARY_APPEARANCE": float(auxiliary_weighted.detach().item()),
                    "L_CICA_raw": float(cica_raw.detach().item()),
                    "L_CICA_weighted": float(cica_weighted.detach().item()),
                    "resolved_lambda_CICA": lambda_cica if lambda_cica is not None else 0.0,
                    "MDRR_enabled": active_mdrr,
                    "CICA_enabled": bool(use_cica and step >= CICA_START_STEP),
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
                    **grad_info,
                    **metrics,
                })
                module_rows.append({
                    "arm": arm,
                    "absolute_step": step,
                    "module": "combined",
                    "event": "training_log",
                    "MDRR_enabled": active_mdrr,
                    "CICA_enabled": bool(use_cica and step >= CICA_START_STEP),
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
                })
            topology_rows.append(_topology_row(arm, step, model, cumulative)) if step % 500 == 0 or step in snapshot_steps else None
            if step in snapshot_steps:
                mdrr_hash = mdrr_control_hash
                cica_hash = _hash_tensor(cica_controller["target_log_chroma"]) if cica_controller is not None else ""
                auxiliary_hash = _hash_tensor(auxiliary_controller["targets"]) if auxiliary_controller is not None else ""
                checkpoint = _save_checkpoint(branch, scene, arm, step, output_dir, source_payload, mdrr_hash, cica_hash, auxiliary_hash, lambda_cica)
                checkpoints.append({"arm": arm, "absolute_step": step, "path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": _sha256(checkpoint)})
                _write_csv(output_dir / f"training_summary_{arm}.csv", rows)
                _write_csv(output_dir / "module_statistics.csv", module_rows)
                _write_csv(output_dir / "topology_statistics.csv", topology_rows)
            del outputs, losses, base_loss, auxiliary_raw, auxiliary_weighted, cica_raw, cica_weighted, batch
            if step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        projector_hash_end = _hash_tensor(model._camera_medium_observability_projector)
        result = {
            "scene": scene,
            "arm": arm,
            "completed_updates": len(sequence),
            "first_step": int(sequence[0]["absolute_step"]),
            "final_step": int(sequence[-1]["absolute_step"]),
            "training_wall_seconds": time.perf_counter() - started,
            "ocmc_projector_hash_start": projector_hash_start,
            "ocmc_projector_hash_end": projector_hash_end,
            "ocmc_projector_unchanged": projector_hash_start == projector_hash_end,
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
    branch.pipeline.model.load_state_dict(payload["model"], strict=True)
    branch.pipeline.model.step = int(payload["absolute_step"])
    _configure_model(branch.pipeline.model)
    FORMAL._install_condition(branch.pipeline.model, "C0", payload["ocmc_bundle"], None)
    branch.pipeline.eval()
    return payload


def _evaluate_checkpoint(branch: Any, arm: str, step: int, records: Sequence[Tuple[int, str, Cameras, Dict[str, Any]]], render_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    model = branch.pipeline.model
    metric_values = {key: [] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}
    per_view: List[Dict[str, Any]] = []
    render_rows: List[Dict[str, Any]] = []
    maps: Dict[str, Dict[str, Tensor]] = {}
    for _index, view_id, camera, batch in records:
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera.to(model.device))
            gt = _gt(model, _batch_to_device(batch, model.device), outputs["background"]).detach().float().clamp(0.0, 1.0)
        pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
        metrics = MIC._metric_images(model, pred, gt)
        clear = outputs.get("rgb_clear")
        clear_raw = outputs.get("clear_object_fullsh_raw")
        if not isinstance(clear, Tensor) or not isinstance(clear_raw, Tensor):
            raise RuntimeError("native clear outputs are missing from the renderer")
        clear = clear.detach().float().clamp(0.0, 1.0)
        clear_raw = clear_raw.detach().float()
        per_view.append({"arm": arm, "absolute_step": step, "split": "eval", "view_id": str(view_id), **metrics, **_clear_distribution(clear_raw)})
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
            _save_native_rgb(scene_dir / "clear.png", clear)
            _save_native_rgb(scene_dir / "clear_raw_display_clamp01.png", clear_raw)
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


def _evaluate_arm(scene: str, arm: str, output_dir: Path, render_dir: Path) -> Dict[str, Any]:
    branch = None
    try:
        branch = _new_branch(scene)
        records_eval = _eval_records(branch)
        global_rows: List[Dict[str, Any]] = []
        per_view_rows: List[Dict[str, Any]] = []
        decomposition_rows: List[Dict[str, Any]] = []
        render_rows: List[Dict[str, Any]] = []
        for step in SNAPSHOT_STEPS:
            checkpoint = output_dir / "checkpoints" / f"step-{step:09d}.ckpt"
            payload = _load_snapshot(branch, scene, checkpoint)
            row, views, decomp, saved_renders = _evaluate_checkpoint(branch, arm, step, records_eval, render_dir if step == FINAL_STEP else None)
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


def run(scene: str, arm: str, gpu: str, output_dir: Path, max_step: int = FINAL_STEP) -> Dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    runtime = _runtime(scene, gpu)
    if not SOURCE_CONFIG.is_file():
        raise FileNotFoundError(SOURCE_CONFIG)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_CONFIG, output_dir / "source_bnd_config.yml")
    _seed_all(TRAINING_SEED)
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
            "source_checkpoint": str(_source_checkpoint(scene)),
            "source_checkpoint_sha256": _sha256(_source_checkpoint(scene)),
            "source_config_sha256": _sha256(SOURCE_CONFIG),
            "start_step": START_STEP,
            "final_step": max_step,
            "matched_updates": max_step - START_STEP,
            "training_seed": TRAINING_SEED,
            "training_rng_manifest": _rng_manifest(training_rng),
            "module_schedule": {
                "AUXILIARY_APPEARANCE": {"enabled": AUXILIARY_APPEARANCE_ENABLED, "name": "A_DETACHED_SH_OPACITY_TANGENT_ORTHOGONALIZATION", "strength": AUXILIARY_APPEARANCE_STRENGTH, "role": "CURRENT_BASE auxiliary appearance regularization; not claimed as identifiability innovation"},
                "MDRR": {"enabled": arm in ("A1", "A3"), "start_step": MDRR_START_STEP, "refresh_rule": "every active primary update from the current model state", "full_medium_response": True},
                "CICA": {"enabled": arm in ("A2", "A3"), "start_step": CICA_START_STEP, "refresh_steps": list(CICA_REFRESH_STEPS), "camera_bank_max": CICA_BANK_MAX, "huber_delta": CICA_HUBER_DELTA, "target_gradient_fraction": CICA_GRADIENT_FRACTION},
            },
            "locked_config": _locked_config(probe.pipeline.model),
            "ocmc_on": True,
            "raoc_off": True,
            "topology_unchanged_logic": "normal existing refinement callbacks; no module topology intervention",
            "gradient_routing_audit": routing_audit,
        })
    finally:
        _release(probe)
    result = _train_arm(scene, arm, output_dir, sequence, partner, bank, training_rng, max_step=max_step)
    if max_step == FINAL_STEP:
        render_dir = RENDER_ROOT / scene
        evaluation = _evaluate_arm(scene, arm, output_dir, render_dir)
        result["evaluation_rows"] = len(evaluation["global_rows"])
        result["per_view_rows"] = len(evaluation["per_view_rows"])
        _write_json(output_dir / "decomposition_safety.json", {"rows": evaluation["decomposition_rows"], "all_P_J_gt_1_zero": all(float(row.get("P_J_gt_1", 0.0)) == 0.0 for row in evaluation["decomposition_rows"])})
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
    args = parser.parse_args()
    result = run(args.scene, args.arm, args.gpu, args.output_dir.resolve(), int(args.max_step))
    print(json.dumps(_sanitize(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
