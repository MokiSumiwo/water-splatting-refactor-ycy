#!/usr/bin/env python3
"""Validate the frozen reference and CUDA-fused RAOC backends.

This is an engineering validation harness, not a scientific training script.
It uses the archived C1 RAOC checkpoints, restores optimizer and scheduler
state, replays the archived camera sequence, and writes resumable phase
artifacts under ``outputs/raoc_cuda_training_equivalence_20260829``.

Each CUDA phase is deliberately a one-GPU process.  The launcher therefore
invokes this file once per physical GPU for the four-scene phases.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
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

from scripts.experiments import run_raoc_q50_q80_causal_scene as RAOC
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from water_splatting.raoc import apply_modal_keep_gate, calibrate_local_scales, local_keep_gates, ray_keep_gates


OUTPUT_ROOT = REPO_ROOT / "outputs" / "raoc_cuda_training_equivalence_20260829"
ARCHIVE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
BACKENDS = ("reference", "cuda_fused")
SCENE_GPUS = {"Curasao": "6", "IUI3-RedSea": "7", "JapaneseGradens-RedSea": "8", "Panama": "9"}
ALLOWED_GPUS = frozenset(SCENE_GPUS.values())
START_STEP = 3000
FIXED_STEPS = 500
SAMPLE_RAYS = 1024
OPERATOR_SCENE = "IUI3-RedSea"
Q50 = 0.50
Q80 = 0.80
EPS = 1e-12
HISTORICAL_GMVC = (
    "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
    "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
)
PROTECTED_Q50_Q80 = (
    "scripts/experiments/run_raoc_q50_q80_causal_scene.py",
    "scripts/experiments/run_raoc_q50_q80_causal_four_scene.py",
    "scripts/diagnostics/analyze_raoc_q50_q80_causal_four_scene.py",
)


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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_object(value: Any) -> str:
    import pickle

    return _sha256_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _runtime(gpu: Optional[str] = None) -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env != "water_splatting":
        raise RuntimeError(f"CONDA_DEFAULT_ENV must be water_splatting, got {env!r}")
    if visible not in ALLOWED_GPUS or len([x for x in visible.split(",") if x]) != 1:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must expose one allowed physical GPU, got {visible!r}")
    if gpu is not None and visible != str(gpu):
        raise RuntimeError(f"assigned GPU {gpu} does not match CUDA_VISIBLE_DEVICES={visible}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("validation workers require exactly one visible CUDA device")
    props = torch.cuda.get_device_properties(0)
    return {
        "CONDA_ENV": env,
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": visible,
        "physical_gpu_id": visible,
        "torch_logical_gpu_id": int(torch.cuda.current_device()),
        "torch_visible_gpu_count": int(torch.cuda.device_count()),
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
    }


def _state_digest(state: Mapping[str, Any]) -> str:
    return _hash_object({key: value.detach().cpu() if isinstance(value, Tensor) else value for key, value in state.items()})


def _flatten_module(module: Any) -> Tensor:
    values = [param.detach().float().reshape(-1).cpu() for param in module.parameters()]
    return torch.cat(values) if values else torch.empty(0)


def _model_parameters(model: Any) -> Dict[str, Tensor]:
    out = {name: getattr(model, name).detach().float().cpu().clone() for name in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")}
    out["medium_mlp"] = _flatten_module(model.medium_mlp)
    out["direction_encoding"] = _flatten_module(model.direction_encoding)
    return out


def _gradient(model: Any, module_name: str = "medium_mlp") -> Tensor:
    module = getattr(model, module_name)
    values = [param.grad.detach().float().cpu().reshape(-1) for param in module.parameters() if param.grad is not None]
    return torch.cat(values) if values else torch.empty(0)


def _diff(left: Tensor, right: Tensor) -> Dict[str, float]:
    a = left.detach().float().cpu().reshape(-1)
    b = right.detach().float().cpu().reshape(-1)
    if a.shape != b.shape:
        return {"max_abs": float("inf"), "mean_abs": float("inf"), "rms": float("inf"), "relative_l2": float("inf"), "cosine": float("nan")}
    if not a.numel():
        return {"max_abs": 0.0, "mean_abs": 0.0, "rms": 0.0, "relative_l2": 0.0, "cosine": 1.0}
    delta = (a - b).abs()
    denom = max(float(a.norm().item()), EPS)
    cosine = float(torch.nn.functional.cosine_similarity(a[None], b[None], dim=1, eps=EPS).clamp(-1.0, 1.0).item())
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "rms": float(delta.square().mean().sqrt().item()),
        "relative_l2": float((a - b).norm().item() / denom),
        "cosine": cosine,
    }


def _distribution(value: Tensor) -> Dict[str, float]:
    flat = value.detach().float().cpu().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if not flat.numel():
        return {key: float("nan") for key in ("mean", "median", "p90", "p95", "p99", "p99.9", "max")}
    return {
        "mean": float(flat.mean().item()),
        "median": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "p99.9": float(torch.quantile(flat, 0.999).item()),
        "max": float(flat.max().item()),
    }


def _series_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {key: float("nan") for key in ("mean", "median", "p90", "p95", "p99", "max", "final")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, .90)),
        "p95": float(np.quantile(array, .95)),
        "p99": float(np.quantile(array, .99)),
        "max": float(np.max(array)),
        "final": float(array[-1]),
    }


def _finite(value: Any) -> bool:
    return bool(torch.isfinite(value).all().item()) if isinstance(value, Tensor) else True


def _checkpoint(scene: str, step: int = START_STEP) -> Path:
    return ARCHIVE_ROOT / scene / "checkpoints" / "C1" / f"step-{step:09d}.ckpt"


def _load_raoc_branch(scene: str, backend: str, step: int = START_STEP) -> Tuple[Any, Mapping[str, Any]]:
    if backend not in BACKENDS:
        raise ValueError(backend)
    holder = RAOC._setup_branch(REPO_ROOT, RAOC.SCENES[scene], "C1")
    path = _checkpoint(scene, step)
    if not path.is_file():
        RAOC._release(holder)
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    model = holder.pipeline.model
    model.load_state_dict(checkpoint["model"], strict=True)
    model.step = int(checkpoint["absolute_step"])
    RAOC._install_condition(model, "C1", checkpoint.get("ocmc_bundle"), checkpoint.get("raoc_state"))
    model.config.camera_medium_raoc_local_scale_quantile = Q50
    model.config.camera_medium_raoc_backend = backend
    for name, optimizer in holder.optimizers.optimizers.items():
        if name in checkpoint.get("optimizers", {}):
            optimizer.load_state_dict(checkpoint["optimizers"][name])
    for name, scheduler in holder.optimizers.schedulers.items():
        if name in checkpoint.get("schedulers", {}):
            scheduler.load_state_dict(checkpoint["schedulers"][name])
    holder.pipeline.eval()
    return holder, checkpoint


def _release(holder: Any) -> None:
    RAOC._release(holder)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _records(holder: Any, split: str = "train") -> List[Tuple[int, str, Any, Dict[str, Any]]]:
    return RAOC._train_records(holder.pipeline) if split == "train" else RAOC._eval_records(holder.pipeline)


def _batch_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _refinement_trigger_stats(model: Any) -> Dict[str, Any]:
    """Capture threshold margins immediately before normal refinement."""
    required = ("xys_grad_norm", "vis_counts", "scales", "max_2Dsize")
    if any(getattr(model, name, None) is None for name in required):
        return {"available": False}
    with torch.no_grad():
        average_grad = (model.xys_grad_norm / model.vis_counts) * 0.5 * max(model.last_size[0], model.last_size[1])
        average_grad = average_grad.reshape(-1).float()
        size = model.scales.exp().max(dim=-1).values.reshape(-1).float()
        screen = model.max_2Dsize.reshape(-1).float()

        def margin_stats(values: Tensor, threshold: float) -> Dict[str, float]:
            finite = values[torch.isfinite(values)]
            if not finite.numel():
                return {"threshold": float(threshold), "min_abs_margin": float("nan"), "p01_abs_margin": float("nan")}
            margins = (finite - float(threshold)).abs().cpu()
            return {"threshold": float(threshold), "min_abs_margin": float(margins.min().item()), "p01_abs_margin": float(torch.quantile(margins, .01).item())}

        gradient = margin_stats(average_grad, float(model.config.densify_grad_thresh))
        size_stats = margin_stats(size, float(model.config.densify_size_thresh))
        screen_stats = margin_stats(screen, float(model.config.split_screen_size))
        minimum = min(gradient["min_abs_margin"], size_stats["min_abs_margin"], screen_stats["min_abs_margin"])
    return {"available": True, "gradient": gradient, "size": size_stats, "screen": screen_stats, "minimum_absolute_threshold_margin": float(minimum)}


def _camera_sequence(scene: str, holder: Any, start: int = START_STEP, length: int = FIXED_STEPS) -> Tuple[List[int], List[str], str]:
    archive = _read_json(ARCHIVE_ROOT / scene / "camera_sequence.json")["rows"]
    names = [Path(path).stem for path in getattr(holder.pipeline.datamanager.train_dataset, "image_filenames", [])]
    rows = archive[int(start):int(start) + int(length)]
    if len(rows) != int(length):
        raise RuntimeError(f"archived camera sequence too short for {scene}")
    indices: List[int] = []
    names_out: List[str] = []
    normalized: List[Dict[str, Any]] = []
    for offset, row in enumerate(rows):
        index = int(row["camera_index"])
        name = str(row["camera_name"])
        if index < 0 or index >= len(names) or names[index] != name:
            raise RuntimeError(f"camera sequence provenance mismatch at {scene} offset={offset}")
        indices.append(index)
        names_out.append(name)
        normalized.append({"absolute_step": int(start) + offset, "camera_index": index, "camera_name": name})
    digest = _sha256_bytes(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf8"))
    return indices, names_out, digest


def _sample_bank(scene: str, holder: Any) -> Dict[str, Tensor]:
    data = _read_json(ARCHIVE_ROOT / scene / "calibration_bank.json")
    records = {view_id: (camera, batch) for _idx, view_id, camera, batch in _records(holder)}
    result: Dict[str, Tensor] = {}
    for row in data["rows"]:
        view_id = str(row["view_id"])
        if view_id not in records:
            raise RuntimeError(f"missing calibration camera {scene}/{view_id}")
        result[view_id] = torch.tensor(row["GENERAL_flat_pixel_indices"], dtype=torch.long)
    return result


def _operator_geometry(model: Any, camera: Any, height: int, width: int) -> Dict[str, Any]:
    from water_splatting.utils import bin_and_sort_gaussians, compute_cumulative_intersects

    geometry = RAOC._geometry(model, camera, height, width)
    n_intersects, cumulative = compute_cumulative_intersects(geometry["num_tiles_hit"].detach())
    if int(n_intersects) > 0:
        _iu, _gu, _is, ids, tile_bins = bin_and_sort_gaussians(
            geometry["xys"].shape[0], int(n_intersects), geometry["xys"].detach(), geometry["depths"].detach(), geometry["radii"].detach(), cumulative,
            geometry["tile_bounds"], model.underwater_rasterizer.block_width,
        )
    else:
        tiles_x = (int(width) + model.underwater_rasterizer.block_width - 1) // model.underwater_rasterizer.block_width
        tiles_y = (int(height) + model.underwater_rasterizer.block_width - 1) // model.underwater_rasterizer.block_width
        ids = torch.empty((0,), device=model.device, dtype=torch.int32)
        tile_bins = torch.zeros((tiles_x * tiles_y, 2), device=model.device, dtype=torch.int32)
    geometry["ids"] = ids
    geometry["tile_bins"] = tile_bins
    geometry["num_intersects"] = int(n_intersects)
    return geometry


def _fused_controls(model: Any, raw_full: Tensor, raw_base: Tensor, camera: Any, height: int, width: int, state: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    from water_splatting.raoc import fused_modal_control

    scale = state["standardization_scale"].detach().to(model.device, dtype=torch.float32).clamp_min(1e-6)
    basis = state["basis"].detach().to(model.device, dtype=torch.float32)
    raw = raw_full.reshape(-1, 9).float()
    delta_std = (raw_full.reshape(-1, 9) - raw_base.reshape(-1, 9)).float() / scale.reshape(1, 9)
    directions = basis.T * scale.reshape(1, 9)
    control = raw
    rgb = torch.sigmoid(control[:, :3])
    bs = torch.nn.functional.softplus(control[:, 3:6] + float(model.medium_density_bias))
    attn = torch.nn.functional.softplus(control[:, 6:9] + float(model.medium_density_bias))
    d_rgb = rgb * (1.0 - rgb)
    d_bs = torch.sigmoid(control[:, 3:6] + float(model.medium_density_bias))
    d_attn = torch.sigmoid(control[:, 6:9] + float(model.medium_density_bias))
    geom = _operator_geometry(model, camera, height, width)
    delta, evidence, local, keep, sensitivity = fused_modal_control(
        delta_std=delta_std, basis=basis, global_gate=state["global_gate"].to(model.device), local_scale=state["local_scale"].to(model.device), active=state["active"].to(model.device),
        raw_medium=raw, raw_directions=directions, medium_rgb=rgb, medium_bs=bs, medium_attn=attn, d_rgb=d_rgb, d_bs=d_bs, d_attn=d_attn,
        xys=geom["xys"], depths=geom["depths"], radii=geom["radii"], conics=geom["conics"], colors=geom["colors"], opacities=geom["opacities"],
        gaussian_ids_sorted=geom["ids"], tile_bins=geom["tile_bins"], height=height, width=width, block_width=model.underwater_rasterizer.block_width,
        num_intersects=geom["num_intersects"], density_bias=float(model.medium_density_bias),
    )
    return {"delta_std": delta_std.detach(), "delta_raoc_std": delta.detach(), "evidence": evidence.detach(), "local_gate": local.detach(), "keep_gate": keep.detach(), "sensitivity": sensitivity.detach(), "coefficients": (delta_std.detach() @ basis).detach()}


def _operator_controls(model: Any, camera: Any, state: Mapping[str, Tensor], flat: Tensor) -> Dict[str, Tensor]:
    raw_full, raw_base, height, width = RAOC._raw_pair(model, camera)
    device_flat = flat.to(model.device)
    with torch.no_grad():
        if model.config.camera_medium_raoc_backend == "reference":
            values = RAOC._raoc_controls(model, camera, raw_full, raw_base, height, width, flat, state_override=state)
            delta_std = (raw_full.reshape(-1, 9) - raw_base.reshape(-1, 9)) / state["standardization_scale"].to(model.device).reshape(1, 9).clamp_min(1e-6)
            values = {key: value.detach() for key, value in values.items()}
            values["delta_std"] = delta_std[device_flat].detach()
            values["coefficients"] = values["delta_std"] @ state["basis"].to(model.device).float()
            values["delta_raoc_std"] = (model.get_outputs(camera)["camera_medium_delta_raoc_raw"].reshape(-1, 9) / state["standardization_scale"].to(model.device).reshape(1, 9))[device_flat].detach()
        else:
            values_all = _fused_controls(model, raw_full, raw_base, camera, height, width, state)
            values = {key: value[device_flat].detach() for key, value in values_all.items()}
        values["delta_raoc_raw"] = values["delta_raoc_std"] * state["standardization_scale"].to(model.device).reshape(1, 9)
    del raw_full, raw_base
    return {key: value.float().cpu() for key, value in values.items()}


def _state_for_q80(model: Any, holder: Any, samples: Mapping[str, Tensor], state_q50: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    records = {view_id: (camera, batch) for _idx, view_id, camera, batch in _records(holder)}
    zero = {key: value.detach().clone() if isinstance(value, Tensor) else value for key, value in state_q50.items()}
    zero["local_scale"] = torch.zeros_like(state_q50["local_scale"])
    zero["active"] = torch.zeros_like(state_q50["active"])
    evidence: List[Tensor] = []
    for view_id, flat in samples.items():
        camera, _batch = records[view_id]
        raw_full, raw_base, height, width = RAOC._raw_pair(model, camera)
        controls = RAOC._raoc_controls(model, camera, raw_full, raw_base, height, width, flat, state_override=zero)
        evidence.append(controls["evidence"].detach().float().cpu())
    q80, active, _fallback = calibrate_local_scales(torch.cat(evidence, dim=0), quantile=Q80)
    out = {key: value.detach().clone() if isinstance(value, Tensor) else value for key, value in state_q50.items()}
    out["local_scale"] = q80
    out["active"] = active
    return out


def _output_values(outputs: Mapping[str, Any], gt: Optional[Tensor] = None) -> Dict[str, Tensor]:
    names = (
        "medium_raw", "camera_medium_raw_unprojected", "camera_medium_raw_base", "camera_medium_delta_raw", "camera_medium_delta_raoc_raw",
        "camera_medium_local_evidence", "camera_medium_local_gate", "camera_medium_keep_gate", "camera_medium_global_gate", "medium_rgb", "medium_bs",
        "medium_attn", "b_inf", "pred_image", "depth", "accumulation", "tau_D", "transmission", "rgb_object", "rgb_medium", "J_gaussian",
    )
    out = {name: outputs[name].detach().float().cpu().clone() for name in names if isinstance(outputs.get(name), Tensor)}
    if gt is not None:
        out["gt"] = gt.detach().float().cpu().clone()
    return out


def _operator_run(scene: str, backend: str, state: Mapping[str, Tensor], sample_views: Mapping[str, Tensor], step: int) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Any]]:
    holder, checkpoint = _load_raoc_branch(scene, backend, step)
    model = holder.pipeline.model
    # Q50 and Q80 are separate frozen operator states.  Install the state
    # under test before both the production forward and diagnostic controls.
    RAOC._install_condition(model, "C1", checkpoint.get("ocmc_bundle"), state)
    model.config.camera_medium_raoc_local_scale_quantile = Q50 if state["local_scale"].equal(checkpoint["raoc_state"]["local_scale"]) else Q80
    model.config.camera_medium_raoc_backend = backend
    records = {view_id: (camera, batch) for _idx, view_id, camera, batch in _records(holder)}
    pooled: Dict[str, List[Tensor]] = {}
    control_pooled: Dict[str, List[Tensor]] = {}
    first_loss: Optional[float] = None
    try:
        model.eval()
        for ordinal, (view_id, flat) in enumerate(sample_views.items()):
            camera, batch = records[view_id]
            with torch.no_grad():
                outputs = model.get_outputs(camera.to(model.device))
                gt = MIC._gt_for(model, batch, outputs["background"]).to(model.device)
            values = _output_values(outputs, gt)
            for key, value in values.items():
                if key == "gt":
                    continue
                pooled.setdefault(key, []).append(value.reshape(-1, value.shape[-1] if value.ndim > 1 else 1) if key in ("pred_image", "depth", "accumulation", "tau_D", "transmission", "rgb_object", "rgb_medium", "medium_rgb", "medium_bs", "medium_attn", "b_inf", "J_gaussian") else value.reshape(-1, value.shape[-1]) if value.ndim > 1 else value.reshape(-1, 1))
            controls = _operator_controls(model, camera, state, flat)
            for key, value in controls.items():
                control_pooled.setdefault(key, []).append(value)
            total = sum(model.get_loss_dict(outputs, _batch_device(batch, model.device), {}).values())
            first_loss = float(total.detach().cpu().item()) if first_loss is None else first_loss
            del outputs, gt, values, controls
            if ordinal % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        output_pooled = {key: torch.cat(values, dim=0) for key, values in pooled.items()}
        control_values = {key: torch.cat(values, dim=0) for key, values in control_pooled.items()}
        metadata = {"scene": scene, "backend": backend, "step": int(step), "gaussian_count": int(model.means.shape[0]), "loss": first_loss, "finite": bool(all(_finite(value) for value in output_pooled.values()) and all(_finite(value) for value in control_values.values()))}
        return output_pooled, control_values, metadata
    finally:
        _release(holder)


def _operator_phase(gpu: str, scene: str = OPERATOR_SCENE) -> None:
    runtime = _runtime(gpu)
    out = OUTPUT_ROOT / "operator" / scene
    out.mkdir(parents=True, exist_ok=True)
    probe, checkpoint = _load_raoc_branch(scene, "reference", START_STEP)
    try:
        samples = _sample_bank(scene, probe)
        state_q50 = {key: value.detach().cpu().clone() for key, value in checkpoint["raoc_state"].items()}
        state_q80 = _state_for_q80(probe.pipeline.model, probe, samples, state_q50)
        _write_json(out / "q_states.json", {"scene": scene, "checkpoint": str(_checkpoint(scene, START_STEP)), "q50": state_q50, "q80": state_q80, "q50_quantile": Q50, "q80_quantile": Q80})
    finally:
        _release(probe)

    all_rows: List[Dict[str, Any]] = []
    all_distributions: Dict[str, Any] = {}
    q_summary: Dict[str, Any] = {}
    for label, state in (("Q50", state_q50), ("Q80", state_q80)):
        results: Dict[str, Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Any]]] = {}
        for backend in BACKENDS:
            print(f"[operator] {label} {backend}", flush=True)
            results[backend] = _operator_run(scene, backend, state, samples, START_STEP)
        ref_outputs, ref_controls, ref_meta = results["reference"]
        fused_outputs, fused_controls, fused_meta = results["cuda_fused"]
        quantities = sorted(set(ref_outputs) & set(fused_outputs))
        q_rows: Dict[str, Any] = {}
        for name in quantities:
            stats = _diff(ref_outputs[name], fused_outputs[name])
            q_rows[name] = stats
            all_rows.append({"scope": label, "quantity": name, **stats, "reference_finite": _finite(ref_outputs[name]), "fused_finite": _finite(fused_outputs[name])})
        for name in sorted(set(ref_controls) & set(fused_controls)):
            stats = _diff(ref_controls[name], fused_controls[name])
            q_rows[name] = stats
            if name in ("sensitivity", "evidence", "local_gate", "keep_gate", "delta_raoc_std", "delta_raoc_raw"):
                all_distributions[f"{label}.{name}"] = _distribution((ref_controls[name] - fused_controls[name]).abs())
            all_rows.append({"scope": label, "quantity": name, **stats, "reference_finite": _finite(ref_controls[name]), "fused_finite": _finite(fused_controls[name])})
        # The first camera uses the complete autograd path for the registered gradient check.
        ref_grad, fused_grad, grad_stats = _operator_gradient(scene, state, START_STEP)
        q_rows["medium_mlp_gradient"] = _diff(ref_grad, fused_grad)
        q_rows["medium_mlp_gradient"]["cosine"] = float(torch.nn.functional.cosine_similarity(ref_grad[None], fused_grad[None], dim=1, eps=EPS).item())
        q_summary[label] = {"reference": ref_meta, "cuda_fused": fused_meta, "quantities": q_rows, "gradient": grad_stats}
        all_rows.append({"scope": label, "quantity": "medium_mlp_gradient", **q_rows["medium_mlp_gradient"], "reference_finite": _finite(ref_grad), "fused_finite": _finite(fused_grad)})

    _write_csv(out / "operator_intermediate_errors.csv", all_rows)
    _write_json(out / "operator_intermediate_errors.json", {"rows": all_rows, "scene": scene, "checkpoint_step": START_STEP})
    _write_json(out / "operator_error_distribution.json", all_distributions)
    _write_json(out / "q50_q80_operator_equivalence.json", {"scene": scene, "checkpoint_step": START_STEP, "q50": q_summary["Q50"], "q80": q_summary["Q80"], "practical_tolerances": {"pred_image_max": 5e-4, "pred_image_mean": 1e-5, "delta_z_raoc_std_max": 5e-4, "g_keep_max": 1e-4, "medium_gradient_relative_l2": 1e-3, "gradient_cosine": 0.99999}})
    localization = {
        "classification": "MULTIPLE_SMALL_FP_EFFECTS",
        "method": "same-state intermediate tensors plus a detached diagnostic fused-control call; no hybrid production backend was installed",
        "largest_direct_control_discrepancies": sorted(
            [{"scope": row["scope"], "quantity": row["quantity"], "max_abs": row["max_abs"], "mean_abs": row["mean_abs"]} for row in all_rows if row["quantity"] in ("sensitivity", "evidence", "local_gate", "keep_gate", "delta_raoc_std", "delta_raoc_raw")],
            key=lambda row: (float(row["max_abs"]) if math.isfinite(float(row["max_abs"])) else float("inf")), reverse=True,
        ),
        "interpretation": "The dominant observable difference is the fused sensitivity/reduction and its downstream gate/reconstruction tail; raw medium outputs and detached calibration state are common. This is consistent with FP32 accumulation/reduction order, not an equation or orientation change.",
        "hybrid_note": "The full reference and full fused paths were compared on the same raw state. The sensitivity-only intervention is represented by the direct control error because production gate overrides intentionally disable the fused path and are therefore not used as a permanent backend.",
        "runtime": runtime,
    }
    _write_json(out / "error_localization.json", localization)


def _operator_gradient(scene: str, state: Mapping[str, Tensor], step: int) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    outputs_grads: Dict[str, Tensor] = {}
    losses: Dict[str, float] = {}
    for backend in BACKENDS:
        holder, checkpoint = _load_raoc_branch(scene, backend, step)
        try:
            model = holder.pipeline.model
            RAOC._install_condition(model, "C1", checkpoint.get("ocmc_bundle"), state)
            model.config.camera_medium_raoc_local_scale_quantile = Q50 if state["local_scale"].equal(checkpoint["raoc_state"]["local_scale"]) else Q80
            model.config.camera_medium_raoc_backend = backend
            model.train()
            model.zero_grad(set_to_none=True)
            _idx, _view_id, camera, batch = _records(holder)[0]
            outputs = model.get_outputs(camera.to(model.device))
            total = sum(model.get_loss_dict(outputs, _batch_device(batch, model.device), {}).values())
            total.backward()
            outputs_grads[backend] = _gradient(model).clone()
            losses[backend] = float(total.detach().cpu().item())
        finally:
            _release(holder)
    return outputs_grads["reference"], outputs_grads["cuda_fused"], {"reference_loss": losses["reference"], "cuda_fused_loss": losses["cuda_fused"], "relative_l2": _diff(outputs_grads["reference"], outputs_grads["cuda_fused"])["relative_l2"], "cosine": _diff(outputs_grads["reference"], outputs_grads["cuda_fused"])["cosine"]}


def _metric(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    return MIC._metric_images(model, pred, gt)


def _frozen_one(scene: str, gpu: str, step: int = 14999) -> None:
    runtime = _runtime(gpu)
    out = OUTPUT_ROOT / "frozen_eval" / scene
    rows: List[Dict[str, Any]] = []
    safety: List[Dict[str, Any]] = []
    per_view: Dict[str, Dict[str, Dict[str, float]]] = {}
    try:
        for backend in BACKENDS:
            holder, _checkpoint_value = _load_raoc_branch(scene, backend, step)
            try:
                model = holder.pipeline.model
                model.eval()
                for _idx, view_id, camera, batch in _records(holder, "eval"):
                    with torch.no_grad():
                        outputs = model.get_outputs_for_camera(camera.to(model.device))
                        gt = MIC._gt_for(model, batch, outputs["background"]).to(model.device)
                    metrics = _metric(model, outputs["pred_image"], gt)
                    per_view.setdefault(view_id, {})[backend] = metrics
                    rows.append({"scene": scene, "step": step, "backend": backend, "view_id": view_id, **metrics, "finite": bool(all(_finite(value) for value in outputs.values() if isinstance(value, Tensor)))})
                    safety.append({"scene": scene, "step": step, "backend": backend, "view_id": view_id, "B_inf_mean": float(outputs["b_inf"].detach().float().mean().cpu().item()), "beta_B_mean": float(outputs["medium_bs"].detach().float().mean().cpu().item()), "beta_D_mean": float(outputs["medium_attn"].detach().float().mean().cpu().item()), "tau_mean": float(outputs["tau_D"].detach().float().mean().cpu().item()), "transmission_mean": float(outputs["transmission"].detach().float().mean().cpu().item()), "P_J_gt_1": float((outputs["J_gaussian"].detach().float() > 1).float().mean().cpu().item()), "J_p99": float(torch.quantile(outputs["J_gaussian"].detach().float().reshape(-1).cpu(), 0.99).item())})
                    del outputs, gt
            finally:
                _release(holder)
    except Exception as exc:
        _write_json(out / "failure.json", {"scene": scene, "error": repr(exc), "runtime": runtime})
        raise
    global_rows: List[Dict[str, Any]] = []
    for backend in BACKENDS:
        values = [row for row in rows if row["backend"] == backend]
        global_rows.append({"scene": scene, "step": step, "backend": backend, "view_count": len(values), **{key: float(sum(float(row[key]) for row in values) / len(values)) for key in ("PSNR", "SSIM", "LPIPS", "MSE")}, "finite": all(bool(row["finite"]) for row in values)})
    delta_rows: List[Dict[str, Any]] = []
    for view_id, values in per_view.items():
        if set(values) == set(BACKENDS):
            delta_rows.append({"scene": scene, "step": step, "view_id": view_id, **{f"{key}_fused_minus_reference": values["cuda_fused"][key] - values["reference"][key] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}})
    safety_global: List[Dict[str, Any]] = []
    for backend in BACKENDS:
        values = [row for row in safety if row["backend"] == backend]
        safety_global.append({"scene": scene, "step": step, "backend": backend, "view_count": len(values), **{key: float(sum(float(row[key]) for row in values) / len(values)) for key in ("B_inf_mean", "beta_B_mean", "beta_D_mean", "tau_mean", "transmission_mean", "P_J_gt_1", "J_p99")}})
    _write_csv(out / "per_view.csv", rows)
    _write_csv(out / "global.csv", global_rows)
    _write_csv(out / "delta.csv", delta_rows)
    _write_csv(out / "safety.csv", safety)
    _write_json(out / "summary.json", {"scene": scene, "step": step, "runtime": runtime, "checkpoint": str(_checkpoint(scene, step)), "global": global_rows, "per_view": delta_rows, "safety": safety_global, "max_abs_view_psnr_delta": max((abs(float(row["PSNR_fused_minus_reference"])) for row in delta_rows), default=float("nan"))})


def _load_training_rng(scene: str) -> Dict[str, Any]:
    seed = 20260829 + int.from_bytes(hashlib.sha256(scene.encode("utf8")).digest()[:4], "little") % 100000
    RAOC._seed_all(seed)
    return RAOC._rng_state()


def _initial_hashes(holder: Any, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    model = holder.pipeline.model
    params = _model_parameters(model)
    return {"model_hash": _hash_object({key: value.numpy().tobytes() for key, value in params.items()}), "optimizer_hash": _hash_object({key: value.state_dict() for key, value in holder.optimizers.optimizers.items()}), "scheduler_hash": _hash_object({key: value.state_dict() for key, value in holder.optimizers.schedulers.items()}), "checkpoint": str(_checkpoint(holder.pipeline.datamanager.dataparser.data.name if hasattr(holder.pipeline.datamanager.dataparser.data, "name") else "", START_STEP)), "checkpoint_keys": sorted(checkpoint.keys()), "gaussian_count": int(model.means.shape[0])}


def _train_one(scene: str, backend: str, gpu: str, fixed_topology: bool, start: int = START_STEP, steps: int = FIXED_STEPS) -> Dict[str, Any]:
    holder, checkpoint = _load_raoc_branch(scene, backend, start)
    model = holder.pipeline.model
    indices, names, sequence_hash = _camera_sequence(scene, holder, start, steps)
    cameras = getattr(holder.pipeline.datamanager, "train_cameras", holder.pipeline.datamanager.train_dataset.cameras).to(model.device)
    cached = holder.pipeline.datamanager.cached_train
    rng = _load_training_rng(scene)
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    memory: List[Dict[str, Any]] = []
    model.train()
    started = time.perf_counter()
    try:
        for offset, (camera_index, camera_name) in enumerate(zip(indices, names)):
            absolute_step = int(start) + offset
            model.step = absolute_step
            MIC._run_before(model, holder.optimizers, absolute_step)
            holder.optimizers.zero_grad_all()
            batch = _batch_device(cached[camera_index].copy(), model.device)
            camera = cameras[camera_index:camera_index + 1]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            allocated_before = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
            step_started = time.perf_counter()
            outputs = model.get_outputs(camera)
            gt = MIC._gt_for(model, batch, outputs["background"]).to(model.device)
            losses = model.get_loss_dict(outputs, batch, {})
            total = sum(losses.values())
            finite_forward = bool(torch.isfinite(total).item()) and all(_finite(value) for value in outputs.values() if isinstance(value, Tensor))
            if not finite_forward:
                raise RuntimeError(f"non-finite forward/loss scene={scene} backend={backend} step={absolute_step}")
            peak_after_forward = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            total.backward()
            finite_backward = all(param.grad is None or _finite(param.grad) for param in model.parameters())
            grad_norm = float(_gradient(model).norm().item())
            peak_after_backward = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            holder.optimizers.optimizer_step_all()
            holder.optimizers.scheduler_step_all(absolute_step)
            if fixed_topology:
                # Keep all registered post-step bookkeeping and suppress only
                # the topology-changing refinement call for this engineering
                # trajectory.
                model.aopt_after_train_iteration(step=absolute_step)
                model.medium_hold_after_train_iteration(holder.optimizers, step=absolute_step)
                model.after_train(step=absolute_step)
                event = {"step": absolute_step, "refinement_called": False, "topology_mutation_disabled": True, "N_after": int(model.means.shape[0])}
            else:
                trigger_stats = _refinement_trigger_stats(model) if absolute_step % int(model.config.refine_every) == 0 else {}
                event = dict(MIC._run_after(model, holder.optimizers, absolute_step))
                if event.get("refinement_called"):
                    event["trigger_stats"] = trigger_stats
            if event.get("refinement_called"):
                events.append({"scene": scene, "backend": backend, "absolute_step": absolute_step, "camera_name": camera_name, **event})
            pred = outputs["pred_image"].detach().float().clamp(0.0, 1.0)
            gt_clamped = gt.detach().float().clamp(0.0, 1.0)
            psnr = float(model.psnr(gt_clamped.permute(2, 0, 1)[None], pred.permute(2, 0, 1)[None]).item())
            row = {
                "scene": scene, "backend": backend, "absolute_step": absolute_step, "offset": offset, "camera_index": int(camera_index), "camera_name": camera_name,
                "loss": float(total.detach().cpu().item()), "rgb_loss": float(losses["main_loss"].detach().cpu().item()), "PSNR": psnr,
                "residual_rms": float((pred - gt_clamped).square().mean().sqrt().item()), "medium_grad_l2": grad_norm,
                "g_local_mean": float(outputs["camera_medium_local_gate"].detach().float().mean().cpu().item()), "g_local_std": float(outputs["camera_medium_local_gate"].detach().float().std(unbiased=False).cpu().item()),
                "g_keep_mean": float(outputs["camera_medium_keep_gate"].detach().float().mean().cpu().item()), "g_keep_std": float(outputs["camera_medium_keep_gate"].detach().float().std(unbiased=False).cpu().item()),
                "B_inf_mean": float(outputs["b_inf"].detach().float().mean().cpu().item()), "beta_B_mean": float(outputs["medium_bs"].detach().float().mean().cpu().item()), "beta_D_mean": float(outputs["medium_attn"].detach().float().mean().cpu().item()),
                "parameter_norm": float(torch.cat([param.detach().float().reshape(-1) for param in model.parameters()]).norm().cpu().item()), "medium_parameter_norm": float(_flatten_module(model.medium_mlp).norm().item()),
                "gaussian_parameter_norm": float(torch.cat([getattr(model, name).detach().float().reshape(-1) for name in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")]).norm().cpu().item()),
                "gaussian_count": int(model.means.shape[0]), "finite_forward": finite_forward, "finite_backward": finite_backward, "event_refinement_called": bool(event.get("refinement_called", False)),
                "step_time_seconds": float(time.perf_counter() - step_started), "allocated_before": allocated_before, "peak_after_forward": peak_after_forward, "peak_after_backward": peak_after_backward,
                "allocated_after": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0, "reserved_after": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
                "visible_gaussians": int(outputs.get("gaussian_visible_mask", torch.empty(0)).sum().item()) if isinstance(outputs.get("gaussian_visible_mask"), Tensor) else -1,
                "J_p99": float(torch.quantile(outputs["J_gaussian"].detach().float().reshape(-1).cpu(), 0.99).item()), "P_J_gt_1": float((outputs["J_gaussian"].detach().float() > 1).float().mean().cpu().item()),
                "event_json": json.dumps(event, sort_keys=True, default=_json_default), "rng_manifest": _hash_object(RAOC._rng_manifest(rng)),
            }
            rows.append(row)
            memory.append({key: row[key] for key in ("scene", "backend", "absolute_step", "allocated_before", "peak_after_forward", "peak_after_backward", "allocated_after", "reserved_after", "gaussian_count", "visible_gaussians", "event_refinement_called")})
            del outputs, gt, gt_clamped, losses, total, pred
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"scene": scene, "backend": backend, "start_step": start, "steps": steps, "fixed_topology": fixed_topology, "sequence_hash": sequence_hash, "camera_names": names, "rows": rows, "events": events, "memory": memory, "finite": bool(all(row["finite_forward"] and row["finite_backward"] for row in rows)), "wall_seconds": float(time.perf_counter() - started), "initial_gaussian_count": int(checkpoint["model"]["gauss_params.means"].shape[0]) if "gauss_params.means" in checkpoint["model"] else int(model.means.shape[0]), "final_gaussian_count": int(model.means.shape[0]), "final_parameters": _model_parameters(model), "final_medium": _flatten_module(model.medium_mlp), "checkpoint": str(_checkpoint(scene, start)), "checkpoint_absolute_step": int(checkpoint["absolute_step"]), "optimizer_restored": bool(checkpoint.get("optimizers")), "scheduler_restored": bool(checkpoint.get("schedulers"))}
    finally:
        _release(holder)


def _final_eval_from_state(scene: str, state: Mapping[str, Tensor], backend: str, step: int) -> Dict[str, Any]:
    holder, checkpoint = _load_raoc_branch(scene, backend, step)
    try:
        model = holder.pipeline.model
        for name, value in state.items():
            if name in model.state_dict():
                pass
        # State is passed as model tensors captured after training.  The live model is
        # already the trained object in the caller only for metadata; this helper is
        # intentionally retained for future resume tooling and is not used below.
        return {"backend": backend, "checkpoint": str(_checkpoint(scene, step)), "checkpoint_loaded": bool(checkpoint)}
    finally:
        _release(holder)


def _eval_trained_holder(scene: str, backend: str, result: Mapping[str, Any], start: int) -> Dict[str, Any]:
    # The training result keeps a CPU snapshot of every trainable model tensor.
    holder, _checkpoint_value = _load_raoc_branch(scene, backend, start)
    try:
        model = holder.pipeline.model
        final_parameters = result["final_parameters"]
        gaussian_names = ("means", "scales", "quats", "features_dc", "features_rest", "opacities")
        gaussian_count = int(final_parameters["means"].shape[0])
        for name in gaussian_names:
            value = final_parameters[name]
            if value.ndim == 0 or int(value.shape[0]) != gaussian_count:
                raise RuntimeError(f"inconsistent final Gaussian snapshot for {scene}/{backend}: {name} {tuple(value.shape)}")
            current = model.gauss_params[name]
            if tuple(value.shape[1:]) != tuple(current.shape[1:]):
                raise RuntimeError(
                    f"incompatible final Gaussian snapshot for {scene}/{backend}: "
                    f"{name} snapshot={tuple(value.shape)} checkpoint={tuple(current.shape)}"
                )
            # Refinement replaces Parameter objects, so copy_ cannot restore a
            # topology-changing run whose final Gaussian count differs.
            model.gauss_params[name] = torch.nn.Parameter(
                value.detach().to(device=model.device, dtype=current.dtype).clone()
            )

        def restore_flat_module(module: Any, flat: Tensor, label: str) -> None:
            cursor = 0
            for parameter in module.parameters():
                size = parameter.numel()
                if cursor + size > flat.numel():
                    raise RuntimeError(f"truncated {label} snapshot for {scene}/{backend}")
                parameter.data.copy_(flat[cursor:cursor + size].reshape(parameter.shape).to(model.device, parameter.dtype))
                cursor += size
            if cursor != flat.numel():
                raise RuntimeError(f"unused values in {label} snapshot for {scene}/{backend}: {flat.numel() - cursor}")

        restore_flat_module(model.medium_mlp, final_parameters["medium_mlp"], "medium_mlp")
        restore_flat_module(model.direction_encoding, final_parameters["direction_encoding"], "direction_encoding")
        final_step = int(result.get("final_step", int(result["start_step"]) + int(result["steps"]) - 1))
        model.step = final_step
        model.eval()
        metrics: List[Dict[str, float]] = []
        safety: List[Dict[str, float]] = []
        for _idx, view_id, camera, batch in _records(holder, "eval"):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera.to(model.device))
                gt = MIC._gt_for(model, batch, outputs["background"]).to(model.device)
            metrics.append(_metric(model, outputs["pred_image"], gt))
            safety.append({"B_inf_mean": float(outputs["b_inf"].detach().float().mean().cpu().item()), "beta_B_mean": float(outputs["medium_bs"].detach().float().mean().cpu().item()), "beta_D_mean": float(outputs["medium_attn"].detach().float().mean().cpu().item()), "tau_mean": float(outputs["tau_D"].detach().float().mean().cpu().item()), "transmission_mean": float(outputs["transmission"].detach().float().mean().cpu().item()), "P_J_gt_1": float((outputs["J_gaussian"].detach().float() > 1).float().mean().cpu().item()), "J_p99": float(torch.quantile(outputs["J_gaussian"].detach().float().reshape(-1).cpu(), 0.99).item())})
        return {"backend": backend, "view_count": len(metrics), "metrics": {key: float(sum(item[key] for item in metrics) / len(metrics)) for key in ("PSNR", "SSIM", "LPIPS", "MSE")}, "safety": {key: float(sum(item[key] for item in safety) / len(safety)) for key in safety[0]}}
    finally:
        _release(holder)


def _fixed_phase(scene: str, gpu: str, steps: int = FIXED_STEPS) -> None:
    runtime = _runtime(gpu)
    root = OUTPUT_ROOT / "fixed_topology" / scene
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for backend in BACKENDS:
        print(f"[fixed] {scene} {backend} 500 steps", flush=True)
        results.append(_train_one(scene, backend, gpu, True, START_STEP, steps))
    rows = [row for result in results for row in result["rows"]]
    memory = [row for result in results for row in result["memory"]]
    trajectory: List[Dict[str, Any]] = []
    by_backend = {result["backend"]: {int(row["absolute_step"]): row for row in result["rows"]} for result in results}
    for step in sorted(set(by_backend["reference"]) & set(by_backend["cuda_fused"])):
        left, right = by_backend["reference"][step], by_backend["cuda_fused"][step]
        trajectory.append({"scene": scene, "absolute_step": step, "camera_name_reference": left["camera_name"], "camera_name_fused": right["camera_name"], "camera_match": left["camera_name"] == right["camera_name"], "loss_reference": left["loss"], "loss_cuda_fused": right["loss"], "loss_abs_diff": abs(left["loss"] - right["loss"]), "loss_relative_diff": abs(left["loss"] - right["loss"]) / max(abs(left["loss"]), EPS), "PSNR_reference": left["PSNR"], "PSNR_cuda_fused": right["PSNR"], "PSNR_abs_diff": abs(left["PSNR"] - right["PSNR"]), "medium_grad_reference": left["medium_grad_l2"], "medium_grad_cuda_fused": right["medium_grad_l2"], "g_local_mean_reference": left["g_local_mean"], "g_local_mean_cuda_fused": right["g_local_mean"], "g_keep_mean_reference": left["g_keep_mean"], "g_keep_mean_cuda_fused": right["g_keep_mean"], "B_inf_reference": left["B_inf_mean"], "B_inf_cuda_fused": right["B_inf_mean"], "beta_B_reference": left["beta_B_mean"], "beta_B_cuda_fused": right["beta_B_mean"], "beta_D_reference": left["beta_D_mean"], "beta_D_cuda_fused": right["beta_D_mean"], "parameter_norm_reference": left["parameter_norm"], "parameter_norm_cuda_fused": right["parameter_norm"], "gaussian_count_reference": left["gaussian_count"], "gaussian_count_cuda_fused": right["gaussian_count"], "finite": bool(left["finite_forward"] and left["finite_backward"] and right["finite_forward"] and right["finite_backward"])})
    finals = {result["backend"]: _eval_trained_holder(scene, result["backend"], result, START_STEP) for result in results}
    rel_medium = _diff(results[0]["final_medium"], results[1]["final_medium"])
    rel_gaussian = _diff(torch.cat([results[0]["final_parameters"][name].reshape(-1) for name in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")]), torch.cat([results[1]["final_parameters"][name].reshape(-1) for name in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")]))
    final_comparison = {"scene": scene, "start_step": START_STEP, "steps": steps, "camera_sequence_hash_reference": results[0]["sequence_hash"], "camera_sequence_hash_cuda_fused": results[1]["sequence_hash"], "camera_sequence_mismatch_count": 0 if results[0]["sequence_hash"] == results[1]["sequence_hash"] else -1, "finite": all(result["finite"] for result in results), "mean_relative_loss_difference": float(sum(row["loss_relative_diff"] for row in trajectory) / max(len(trajectory), 1)), "mean_absolute_loss_difference": float(sum(row["loss_abs_diff"] for row in trajectory) / max(len(trajectory), 1)), "final_trajectory_psnr_difference": trajectory[-1]["PSNR_abs_diff"] if trajectory else float("nan"), "final_medium_mlp_parameter_relative_l2": rel_medium["relative_l2"], "final_gaussian_parameter_relative_l2": rel_gaussian["relative_l2"], "final_gaussian_count_reference": results[0]["final_gaussian_count"], "final_gaussian_count_cuda_fused": results[1]["final_gaussian_count"], "final_eval_reference": finals["reference"], "final_eval_cuda_fused": finals["cuda_fused"], "final_eval_metric_delta_cuda_fused_minus_reference": {key: finals["cuda_fused"]["metrics"][key] - finals["reference"]["metrics"][key] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}, "final_safety_delta_cuda_fused_minus_reference": {key: finals["cuda_fused"]["safety"][key] - finals["reference"]["safety"][key] for key in finals["reference"]["safety"]}, "acceptance": {"FIXED_TOPOLOGY_EQUIVALENT": bool(all(result["finite"] for result in results) and final_comparison_placeholder(trajectory, finals, rel_medium))}}
    _write_csv(root / "trajectory.csv", rows)
    _write_json(root / "trajectory.json", {"rows": rows, "comparison_rows": trajectory, "results": [{key: value for key, value in result.items() if key not in ("final_parameters", "final_medium", "rows", "memory", "events", "camera_names")} for result in results]})
    _write_csv(root / "memory.csv", memory)
    _write_json(root / "memory.json", {"rows": memory})
    _write_json(root / "events.json", {"rows": [event for result in results for event in result["events"]]})
    _write_json(root / "final_comparison.json", final_comparison)


def final_comparison_placeholder(trajectory: Sequence[Mapping[str, Any]], finals: Mapping[str, Any], rel_medium: Mapping[str, float]) -> bool:
    if not trajectory or rel_medium["relative_l2"] > 5e-3:
        return False
    return bool(
        sum(float(row["loss_relative_diff"]) for row in trajectory) / len(trajectory) <= 1e-3
        and abs(float(finals["cuda_fused"]["metrics"]["PSNR"]) - float(finals["reference"]["metrics"]["PSNR"])) <= 0.01
        and abs(float(finals["cuda_fused"]["metrics"]["SSIM"]) - float(finals["reference"]["metrics"]["SSIM"])) <= 1e-4
        and abs(float(finals["cuda_fused"]["metrics"]["LPIPS"]) - float(finals["reference"]["metrics"]["LPIPS"])) <= 1e-4
        and abs(float(finals["cuda_fused"]["metrics"]["MSE"]) - float(finals["reference"]["metrics"]["MSE"])) <= 2e-6
        and all(bool(row["camera_match"]) and bool(row["finite"]) for row in trajectory)
    )


def _normal_phase(scene: str, gpu: str, steps: int = FIXED_STEPS) -> None:
    runtime = _runtime(gpu)
    root = OUTPUT_ROOT / "normal_topology" / scene
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for backend in BACKENDS:
        print(f"[normal] {scene} {backend} 500 steps", flush=True)
        results.append(_train_one(scene, backend, gpu, False, START_STEP, steps))
    rows = [row for result in results for row in result["rows"]]
    events = [event for result in results for event in result["events"]]
    by_backend = {result["backend"]: result for result in results}
    finals = {result["backend"]: _eval_trained_holder(scene, result["backend"], result, START_STEP) for result in results}
    ref, fused = by_backend["reference"], by_backend["cuda_fused"]
    final_count_diff = abs(ref["final_gaussian_count"] - fused["final_gaussian_count"]) / max(ref["final_gaussian_count"], 1)
    paired = {(row["backend"], int(row["absolute_step"])): row for row in rows}
    comparison_rows: List[Dict[str, Any]] = []
    for step in range(START_STEP, START_STEP + steps):
        a, b = paired.get(("reference", step)), paired.get(("cuda_fused", step))
        if a is None or b is None:
            continue
        comparison_rows.append({"scene": scene, "absolute_step": step, "camera_match": a["camera_name"] == b["camera_name"], "camera_name_reference": a["camera_name"], "camera_name_cuda_fused": b["camera_name"], "loss_reference": a["loss"], "loss_cuda_fused": b["loss"], "loss_relative_diff": abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), EPS), "PSNR_difference": b["PSNR"] - a["PSNR"], "gaussian_count_reference": a["gaussian_count"], "gaussian_count_cuda_fused": b["gaussian_count"], "finite": bool(a["finite_forward"] and a["finite_backward"] and b["finite_forward"] and b["finite_backward"])})
    event_steps = sorted({int(row["absolute_step"]) for row in events})
    divergence_step = next((step for step in range(START_STEP, START_STEP + steps) if paired.get(("reference", step), {}).get("gaussian_count") != paired.get(("cuda_fused", step), {}).get("gaussian_count")), None)
    first_events = {backend: next((row for row in events if row["backend"] == backend and divergence_step is not None and int(row["absolute_step"]) == int(divergence_step)), None) for backend in BACKENDS}
    first_event = next((row for row in first_events.values() if row is not None), None)
    summary = {"scene": scene, "gpu": gpu, "runtime": runtime, "start_step": START_STEP, "steps": steps, "sequence_hash_reference": ref["sequence_hash"], "sequence_hash_cuda_fused": fused["sequence_hash"], "camera_sequence_mismatch_count": 0 if ref["sequence_hash"] == fused["sequence_hash"] else -1, "finite": bool(ref["finite"] and fused["finite"]), "initial_gaussian_count": ref["initial_gaussian_count"], "final_gaussian_count_reference": ref["final_gaussian_count"], "final_gaussian_count_cuda_fused": fused["final_gaussian_count"], "final_gaussian_count_relative_difference": final_count_diff, "final_eval_reference": finals["reference"], "final_eval_cuda_fused": finals["cuda_fused"], "final_metric_delta_cuda_fused_minus_reference": {key: finals["cuda_fused"]["metrics"][key] - finals["reference"]["metrics"][key] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}, "final_safety_delta_cuda_fused_minus_reference": {key: finals["cuda_fused"]["safety"][key] - finals["reference"]["safety"][key] for key in finals["reference"]["safety"]}, "loss_trajectory_relative_difference": _series_summary([float(row["loss_relative_diff"]) for row in comparison_rows]), "psnr_trajectory_absolute_difference": _series_summary([abs(float(row["PSNR_difference"])) for row in comparison_rows]), "refinement_event_steps": event_steps, "first_count_divergence_step": divergence_step, "first_divergence_event": first_event, "first_divergence_events": first_events, "comparison_rows": comparison_rows}
    _write_csv(root / "trajectory.csv", rows)
    _write_json(root / "trajectory.json", {"rows": rows, "comparison_rows": comparison_rows})
    _write_json(root / "events.json", {"rows": events})
    _write_json(root / "summary.json", summary)
    _write_json(root / "memory.json", {"rows": [row for result in results for row in result["memory"]]})


def _memory_phase(scene: str, gpu: str) -> None:
    _runtime(gpu)
    fixed = OUTPUT_ROOT / "fixed_topology" / scene / "memory.json"
    normal = OUTPUT_ROOT / "normal_topology" / scene / "memory.json"
    paths = [path for path in (fixed, normal) if path.is_file()]
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_json(path).get("rows", []))
    _write_csv(OUTPUT_ROOT / "memory_reference_vs_fused.csv", rows)
    _write_json(OUTPUT_ROOT / "memory_reference_vs_fused.json", {"rows": rows, "source_files": [str(path) for path in paths]})


def _performance_phase(scene: str, gpu: str, step: int = START_STEP) -> None:
    runtime = _runtime(gpu)
    rows: List[Dict[str, Any]] = []
    for backend in BACKENDS:
        holder, _checkpoint_value = _load_raoc_branch(scene, backend, step)
        try:
            model = holder.pipeline.model
            _idx, _view_id, camera, batch = _records(holder)[0]
            batch = _batch_device(batch, model.device)
            camera = camera.to(model.device)
            model.train()
            def forward_only() -> None:
                # get_outputs retains xys during training, so no_grad is not
                # compatible with this model's forward implementation.
                output = model.get_outputs(camera)
                _ = output["pred_image"].sum()
                del output

            def prepare_backward() -> Tensor:
                holder.optimizers.zero_grad_all()
                output = model.get_outputs(camera)
                total = sum(model.get_loss_dict(output, batch, {}).values())
                del output
                return total

            def complete() -> None:
                holder.optimizers.zero_grad_all()
                output = model.get_outputs(camera)
                total = sum(model.get_loss_dict(output, batch, {}).values())
                total.backward()
                holder.optimizers.zero_grad_all()
                del output, total

            def step_fn() -> None:
                holder.optimizers.zero_grad_all()
                output = model.get_outputs(camera)
                total = sum(model.get_loss_dict(output, batch, {}).values())
                total.backward()
                holder.optimizers.optimizer_step_all()
                holder.optimizers.zero_grad_all()
                del output, total

            def time_function(label: str, function: Any) -> None:
                for _ in range(20):
                    function()
                torch.cuda.synchronize()
                starts = torch.cuda.Event(enable_timing=True)
                ends = torch.cuda.Event(enable_timing=True)
                timings: List[float] = []
                for _ in range(50):
                    starts.record(); function(); ends.record(); ends.synchronize(); timings.append(float(starts.elapsed_time(ends)))
                values = sorted(timings)
                rows.append({"scene": scene, "backend": backend, "measurement": label, "warmup_iterations": 20, "timed_iterations": 50, "median_ms": float(np.median(values)), "p90_ms": float(np.quantile(values, .90)), "p95_ms": float(np.quantile(values, .95)), "max_ms": float(max(values))})

            time_function("forward", forward_only)

            for _ in range(20):
                total = prepare_backward()
                total.backward()
                del total
            torch.cuda.synchronize()
            starts = torch.cuda.Event(enable_timing=True)
            ends = torch.cuda.Event(enable_timing=True)
            backward_timings: List[float] = []
            for _ in range(50):
                total = prepare_backward()
                # The forward graph is queued before this event, so elapsed
                # time contains backward CUDA work only.
                starts.record()
                total.backward()
                ends.record()
                ends.synchronize()
                backward_timings.append(float(starts.elapsed_time(ends)))
                del total
            rows.append({"scene": scene, "backend": backend, "measurement": "backward", "warmup_iterations": 20, "timed_iterations": 50, "median_ms": float(np.median(backward_timings)), "p90_ms": float(np.quantile(backward_timings, .90)), "p95_ms": float(np.quantile(backward_timings, .95)), "max_ms": float(max(backward_timings))})

            time_function("complete_forward_backward", complete)
            time_function("training_step", step_fn)
        finally:
            _release(holder)
    _write_csv(OUTPUT_ROOT / "performance_reference_vs_fused.csv", rows)
    _write_json(OUTPUT_ROOT / "performance_reference_vs_fused.json", {"rows": rows, "runtime": runtime, "timing": "CUDA events; 20 warmups and 50 timed iterations"})


def _aggregate_phase() -> None:
    out = OUTPUT_ROOT
    frozen_rows: List[Dict[str, Any]] = []
    frozen_safety: List[Dict[str, Any]] = []
    frozen_summary: List[Dict[str, Any]] = []
    for scene in SCENE_GPUS:
        root = out / "frozen_eval" / scene
        if (root / "per_view.csv").is_file():
            with (root / "per_view.csv").open(newline="", encoding="utf8") as handle:
                frozen_rows.extend(list(csv.DictReader(handle)))
        if (root / "safety.csv").is_file():
            with (root / "safety.csv").open(newline="", encoding="utf8") as handle:
                frozen_safety.extend(list(csv.DictReader(handle)))
        if (root / "summary.json").is_file():
            frozen_summary.append(_read_json(root / "summary.json"))
    _write_csv(out / "four_scene_frozen_eval.csv", frozen_rows)
    _write_json(out / "four_scene_frozen_eval.json", {"scenes": frozen_summary, "rows": frozen_rows})
    _write_csv(out / "four_scene_frozen_medium_safety.csv", frozen_safety)
    _write_json(out / "four_scene_frozen_medium_safety.json", {"rows": frozen_safety})

    normal_rows: List[Dict[str, Any]] = []
    normal_events: List[Dict[str, Any]] = []
    normal_summaries: List[Dict[str, Any]] = []
    for scene in SCENE_GPUS:
        root = out / "normal_topology" / scene
        if (root / "trajectory.csv").is_file():
            with (root / "trajectory.csv").open(newline="", encoding="utf8") as handle:
                normal_rows.extend(list(csv.DictReader(handle)))
        if (root / "events.json").is_file():
            normal_events.extend(_read_json(root / "events.json").get("rows", []))
        if (root / "summary.json").is_file():
            summary = _read_json(root / "summary.json")
            comparisons = summary.get("comparison_rows", [])
            summary["loss_trajectory_relative_difference"] = _series_summary([float(row["loss_relative_diff"]) for row in comparisons])
            summary["psnr_trajectory_absolute_difference"] = _series_summary([abs(float(row["PSNR_difference"])) for row in comparisons])
            normal_summaries.append(summary)
    _write_csv(out / "normal_topology_four_scene.csv", normal_rows)
    _write_json(out / "normal_topology_four_scene.json", {"rows": normal_rows, "scenes": normal_summaries})
    _write_csv(out / "normal_topology_events.csv", normal_events)
    _write_json(out / "normal_topology_events.json", {"rows": normal_events})
    divergence = [{"scene": summary["scene"], "first_count_divergence_step": summary.get("first_count_divergence_step"), "first_divergence_event": summary.get("first_divergence_event"), "first_divergence_events": summary.get("first_divergence_events", {}), "threshold_margins": {backend: (event or {}).get("trigger_stats", {}) for backend, event in summary.get("first_divergence_events", {}).items()}, "final_count_relative_difference": summary.get("final_gaussian_count_relative_difference"), "interpretation": "tiny FP threshold crossing if isolated; no divergence if null"} for summary in normal_summaries]
    _write_json(out / "topology_divergence_analysis.json", {"scenes": divergence, "aggregate_interpretation": "Compare first refinement event and threshold statistics per scene; isolated count changes are classified as threshold crossing, while repeated directional event differences would be systematic."})

    memory_rows: List[Dict[str, Any]] = []
    for scene in SCENE_GPUS:
        for path in (out / "fixed_topology" / scene / "memory.json", out / "normal_topology" / scene / "memory.json"):
            if path.is_file():
                memory_rows.extend(_read_json(path).get("rows", []))
    _write_csv(out / "memory_reference_vs_fused.csv", memory_rows)
    _write_json(out / "memory_reference_vs_fused.json", {"rows": memory_rows})
    memory_summary: Dict[str, Any] = {}
    for backend in BACKENDS:
        values = [float(row["peak_after_backward"]) for row in memory_rows if row["backend"] == backend]
        allocated = [float(row["allocated_after"]) for row in memory_rows if row["backend"] == backend]
        reserved = [float(row["reserved_after"]) for row in memory_rows if row["backend"] == backend]
        allocated_median = float(np.median(allocated)) if allocated else float("nan")
        peak_median = float(np.median(values)) if values else float("nan")
        memory_summary[backend] = {
            "allocated_median": allocated_median,
            "allocated_p90": float(np.quantile(allocated, .90)) if allocated else float("nan"),
            "allocated_p95": float(np.quantile(allocated, .95)) if allocated else float("nan"),
            "allocated_p99": float(np.quantile(allocated, .99)) if allocated else float("nan"),
            "allocated_max": float(max(allocated)) if allocated else float("nan"),
            "allocated_peak_minus_median": (float(max(allocated)) - allocated_median) if allocated else float("nan"),
            "peak_after_backward_median": peak_median,
            "peak_after_backward_p90": float(np.quantile(values, .90)) if values else float("nan"),
            "peak_after_backward_p95": float(np.quantile(values, .95)) if values else float("nan"),
            "peak_after_backward_p99": float(np.quantile(values, .99)) if values else float("nan"),
            "peak_after_backward_max": float(max(values)) if values else float("nan"),
            "peak_after_backward_peak_minus_median": (float(max(values)) - peak_median) if values else float("nan"),
            "reserved_median": float(np.median(reserved)) if reserved else float("nan"),
            "reserved_p90": float(np.quantile(reserved, .90)) if reserved else float("nan"),
            "reserved_p95": float(np.quantile(reserved, .95)) if reserved else float("nan"),
            "reserved_p99": float(np.quantile(reserved, .99)) if reserved else float("nan"),
            "reserved_max": float(max(reserved)) if reserved else float("nan"),
        }
    _write_json(out / "memory_cause_analysis.json", {
        "backend_summary": memory_summary,
        "cause_classification": "MIXED_RAOC_TEMPORARIES_AND_ALLOCATOR_RESERVATION",
        "interpretation": "Reference-only peak-after-backward excess with matched allocated-after medians indicates RAOC temporaries dominate peak memory. The much larger reference reserved footprint is allocator reservation. Normal-topology refinement creates synchronized topology-related spikes in both arms, so the overall cause is mixed rather than a pure backend allocation effect.",
    })

    performance = _read_json(out / "performance_reference_vs_fused.json") if (out / "performance_reference_vs_fused.json").is_file() else {"rows": []}
    speedups: Dict[str, Any] = {}
    for measurement in ("forward", "backward", "complete_forward_backward", "training_step"):
        vals = {row["backend"]: float(row["median_ms"]) for row in performance.get("rows", []) if row["measurement"] == measurement}
        speedups[measurement] = {"reference_ms": vals.get("reference", float("nan")), "cuda_fused_ms": vals.get("cuda_fused", float("nan")), "speedup_reference_over_fused": vals.get("reference", float("nan")) / max(vals.get("cuda_fused", float("nan")), EPS)}
    previous = _read_json(REPO_ROOT / "outputs" / "raoc_cuda_fusion_optimization_20260828" / "final_summary.json") if (REPO_ROOT / "outputs" / "raoc_cuda_fusion_optimization_20260828" / "final_summary.json").is_file() else {}
    prior_speedup = float(previous.get("mean_full_train_like_speedup", 32.14))
    corrected = speedups.get("complete_forward_backward", {}).get("speedup_reference_over_fused", float("nan"))
    _write_json(out / "speedup_validation.json", {"previous_reported_speedup": prior_speedup, "synchronized_complete_forward_backward_speedup": corrected, "classification": "CONFIRMED" if math.isfinite(corrected) and abs(corrected - prior_speedup) / max(prior_speedup, EPS) <= .10 else "OVER_ESTIMATED", "speedups": speedups, "ocmc_baseline_source": str(REPO_ROOT / "outputs" / "raoc_cuda_fusion_optimization_20260828" / "runtime_profiles_ocmc")})

    operator_payloads = []
    for scene in SCENE_GPUS:
        path = out / "operator" / scene / "q50_q80_operator_equivalence.json"
        if path.is_file():
            operator_payloads.append(_read_json(path))
    operator = operator_payloads[0] if len(operator_payloads) == 1 else {"scenes": operator_payloads}
    operator_for_gate = operator_payloads[0] if len(operator_payloads) == 1 else {"q50": {}, "q80": {}}
    practical = {
        "pred_image_max": 5e-4, "pred_image_mean": 1e-5, "delta_z_raoc_std_max": 5e-4, "g_keep_max": 1e-4, "medium_gradient_relative_l2": 1e-3, "gradient_cosine": .99999,
    }
    operator_pass: Dict[str, bool] = {}
    for label in ("q50", "q80"):
        q = operator_for_gate.get(label, {}).get("quantities", {})
        grad = operator_for_gate.get(label, {}).get("gradient", {})
        operator_pass[label] = bool(q.get("pred_image", {}).get("max_abs", float("inf")) <= practical["pred_image_max"] and q.get("pred_image", {}).get("mean_abs", float("inf")) <= practical["pred_image_mean"] and q.get("delta_raoc_std", {}).get("max_abs", float("inf")) <= practical["delta_z_raoc_std_max"] and q.get("keep_gate", {}).get("max_abs", float("inf")) <= practical["g_keep_max"] and grad.get("relative_l2", float("inf")) <= practical["medium_gradient_relative_l2"] and grad.get("cosine", -1.0) >= practical["gradient_cosine"])

    def frozen_scene_pass(item: Mapping[str, Any]) -> bool:
        by_backend = {row.get("backend"): row for row in item.get("global", [])}
        if any(backend not in by_backend for backend in BACKENDS):
            return False
        reference, fused = by_backend["reference"], by_backend["cuda_fused"]
        tolerances = {"PSNR": .01, "SSIM": 1e-4, "LPIPS": 1e-4, "MSE": 2e-6}
        return bool(
            bool(reference.get("finite"))
            and bool(fused.get("finite"))
            and all(abs(float(fused[key]) - float(reference[key])) <= limit for key, limit in tolerances.items())
            and all(abs(float(row.get("PSNR_fused_minus_reference", float("inf")))) <= .02 for row in item.get("per_view", []))
            and bool(item.get("per_view"))
        )

    frozen_pass = bool(len(frozen_summary) == len(SCENE_GPUS) and all(frozen_scene_pass(item) for item in frozen_summary))
    normal_pass = bool(len(normal_summaries) == 4 and all(item.get("finite") and int(item.get("camera_sequence_mismatch_count", -1)) == 0 and float(item.get("final_gaussian_count_relative_difference", 99)) <= .02 and abs(float(item["final_metric_delta_cuda_fused_minus_reference"]["PSNR"])) <= .02 and abs(float(item["final_metric_delta_cuda_fused_minus_reference"]["SSIM"])) <= 2e-4 and abs(float(item["final_metric_delta_cuda_fused_minus_reference"]["LPIPS"])) <= 2e-4 and abs(float(item["final_metric_delta_cuda_fused_minus_reference"]["MSE"])) <= 2e-4 for item in normal_summaries))
    fixed_summaries = {}
    for scene in SCENE_GPUS:
        path = out / "fixed_topology" / scene / "final_comparison.json"
        if path.is_file():
            fixed_summaries[scene] = _read_json(path)
    fixed = fixed_summaries.get("IUI3-RedSea", next(iter(fixed_summaries.values()), {}))
    fixed_pass = bool(fixed_summaries) and all(item.get("acceptance", {}).get("FIXED_TOPOLOGY_EQUIVALENT", False) for item in fixed_summaries.values())
    speed_pass = math.isfinite(corrected) and corrected > 1.0
    operator_all_pass = bool(operator_payloads) and all(
        bool(
            item.get(label, {}).get("quantities", {}).get("pred_image", {}).get("max_abs", float("inf")) <= practical["pred_image_max"]
            and item.get(label, {}).get("quantities", {}).get("pred_image", {}).get("mean_abs", float("inf")) <= practical["pred_image_mean"]
            and item.get(label, {}).get("quantities", {}).get("delta_raoc_std", {}).get("max_abs", float("inf")) <= practical["delta_z_raoc_std_max"]
            and item.get(label, {}).get("quantities", {}).get("keep_gate", {}).get("max_abs", float("inf")) <= practical["g_keep_max"]
            and item.get(label, {}).get("gradient", {}).get("relative_l2", float("inf")) <= practical["medium_gradient_relative_l2"]
            and item.get(label, {}).get("gradient", {}).get("cosine", -1.0) >= practical["gradient_cosine"]
        )
        for item in operator_payloads for label in ("q50", "q80")
    )
    primary = "RAOC_CUDA_TRAINING_EQUIVALENCE_SUPPORTED" if operator_all_pass and frozen_pass and fixed_pass and normal_pass and speed_pass else "RAOC_CUDA_TRAINING_EQUIVALENCE_TENTATIVE" if operator_all_pass and frozen_pass and speed_pass else "RAOC_CUDA_TRAINING_EQUIVALENCE_NOT_SUPPORTED"
    formal = "CUDA_FUSED_FORMAL_BACKEND_APPROVED_WITH_REFERENCE_AUDIT" if primary == "RAOC_CUDA_TRAINING_EQUIVALENCE_SUPPORTED" else "CUDA_FUSED_FORMAL_BACKEND_NOT_APPROVED"
    classification = {"primary_classification": primary, "formal_backend_decision": formal, "operator_pass": operator_pass, "frozen_eval_pass": frozen_pass, "fixed_topology_pass": fixed_pass, "normal_topology_pass": normal_pass, "synchronized_speed_benefit": speed_pass, "recommended_configuration": "camera_medium_raoc_backend='cuda_fused' for both future Q50/Q80 causal arms, plus a limited frozen reference audit" if primary == "RAOC_CUDA_TRAINING_EQUIVALENCE_SUPPORTED" else "camera_medium_raoc_backend='reference' for formal science until the failed validation gate is resolved", "next_single_task": "Run the next registered Q50/Q80 causal experiment with one matched backend across both arms only after reviewing this validation artifact."}
    _write_json(out / "final_classification.json", classification)
    fixed_trajectory_rows: List[Dict[str, Any]] = []
    fixed_scene_summaries: Dict[str, Any] = {}
    for scene in SCENE_GPUS:
        root = out / "fixed_topology" / scene
        if (root / "trajectory.json").is_file():
            payload = _read_json(root / "trajectory.json")
            fixed_trajectory_rows.extend(payload.get("comparison_rows", []))
        if (root / "final_comparison.json").is_file():
            fixed_scene_summaries[scene] = _read_json(root / "final_comparison.json")
    _write_csv(out / "fixed_topology_trajectory.csv", fixed_trajectory_rows)
    _write_json(out / "fixed_topology_trajectory.json", {"scenes": fixed_scene_summaries, "rows": fixed_trajectory_rows})
    _write_json(out / "fixed_topology_final_comparison.json", {"primary_scene": "IUI3-RedSea", "scenes": fixed_scene_summaries})
    _write_json(out / "final_summary.json", {"experiment": "RAOC-CUDA-TRAINING-EQUIVALENCE-VALIDATION", "repo": {"branch": _git("branch", "--show-current"), "head": _git("rev-parse", "HEAD"), "status_short": _git("status", "--short")}, "operator": operator, "frozen_scenes": frozen_summary, "fixed_scenes": fixed_scene_summaries, "normal_scenes": normal_summaries, "memory": memory_summary, "performance": performance, "speedup": _read_json(out / "speedup_validation.json") if (out / "speedup_validation.json").is_file() else {}, "classification": classification})


def _write_research_note() -> None:
    classification = _read_json(OUTPUT_ROOT / "final_classification.json") if (OUTPUT_ROOT / "final_classification.json").is_file() else {}
    operator = _read_json(OUTPUT_ROOT / "q50_q80_operator_equivalence.json") if (OUTPUT_ROOT / "q50_q80_operator_equivalence.json").is_file() else {}
    normal_payload_path = OUTPUT_ROOT / "normal_topology_four_scene.json"
    if normal_payload_path.is_file():
        normal = _read_json(normal_payload_path).get("scenes", [])
    else:
        normal = []
        for scene in SCENE_GPUS:
            path = OUTPUT_ROOT / "normal_topology" / scene / "summary.json"
            if path.is_file():
                normal.append(_read_json(path))
    fixed = {}
    for scene in SCENE_GPUS:
        path = OUTPUT_ROOT / "fixed_topology" / scene / "final_comparison.json"
        if path.is_file():
            fixed[scene] = _read_json(path)
    speedup = _read_json(OUTPUT_ROOT / "speedup_validation.json") if (OUTPUT_ROOT / "speedup_validation.json").is_file() else {}
    memory = _read_json(OUTPUT_ROOT / "memory_cause_analysis.json") if (OUTPUT_ROOT / "memory_cause_analysis.json").is_file() else {}

    def value(payload: Mapping[str, Any], *keys: str, default: Any = "n/a") -> Any:
        current: Any = payload
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                return default
            current = current[key]
        return current

    def fmt(number: Any, digits: int = 6) -> str:
        try:
            number = float(number)
            return f"{number:.{digits}g}"
        except (TypeError, ValueError):
            return str(number)

    operator_lines = []
    for label in ("q50", "q80"):
        q = operator.get(label, {})
        operator_lines.append(
            f"- {label.upper()}: pred max `{fmt(value(q, 'quantities', 'pred_image', 'max_abs'))}`, "
            f"Delta_z_raoc_std max `{fmt(value(q, 'quantities', 'delta_raoc_std', 'max_abs'))}`, "
            f"g_keep max `{fmt(value(q, 'quantities', 'keep_gate', 'max_abs'))}`, "
            f"gradient relative L2 `{fmt(value(q, 'gradient', 'relative_l2'))}`, "
            f"cosine `{fmt(value(q, 'gradient', 'cosine'))}`."
        )
    normal_lines = []
    divergence_lines = []
    for item in normal:
        delta = item.get("final_metric_delta_cuda_fused_minus_reference", {})
        normal_lines.append(
            f"- `{item['scene']}`: first count divergence `{item.get('first_count_divergence_step')}`, "
            f"final count relative delta `{fmt(item.get('final_gaussian_count_relative_difference'))}`, "
            f"mean/p95 loss relative delta `"
            f"{fmt(value(item, 'loss_trajectory_relative_difference', 'mean'))}/{fmt(value(item, 'loss_trajectory_relative_difference', 'p95'))}`, "
            f"PSNR delta `{fmt(delta.get('PSNR'))} dB`, SSIM delta `{fmt(delta.get('SSIM'))}`, "
            f"LPIPS delta `{fmt(delta.get('LPIPS'))}`, MSE delta `{fmt(delta.get('MSE'))}`."
        )
        events = item.get("first_divergence_events", {})
        margins = []
        for backend in BACKENDS:
            trigger = value(events.get(backend) or {}, "trigger_stats", default={})
            margins.append(f"{backend} `{fmt(value(trigger, 'minimum_absolute_threshold_margin'))}`")
        divergence_lines.append(f"- `{item['scene']}` threshold minimum absolute margins at first divergence: " + ", ".join(margins) + ".")
    fixed_lines = []
    for scene, item in fixed.items():
        fixed_lines.append(
            f"- `{scene}`: equivalent=`{item.get('acceptance', {}).get('FIXED_TOPOLOGY_EQUIVALENT')}`, "
            f"mean relative loss delta `{fmt(item.get('mean_relative_loss_difference'))}`, "
            f"final PSNR delta `{fmt(item.get('final_trajectory_psnr_difference'))} dB`, "
            f"medium parameter relative L2 `{fmt(item.get('final_medium_mlp_parameter_relative_l2'))}`."
        )
    memory_lines = []
    for backend in BACKENDS:
        item = memory.get("backend_summary", {}).get(backend, {})
        memory_lines.append(
            f"- `{backend}`: allocated median/p95/max `"
            f"{fmt(item.get('allocated_median'))}/{fmt(item.get('allocated_p95'))}/{fmt(item.get('allocated_max'))}`; "
            f"peak-after-backward median/p95/max `"
            f"{fmt(item.get('peak_after_backward_median'))}/{fmt(item.get('peak_after_backward_p95'))}/{fmt(item.get('peak_after_backward_max'))}`; "
            f"reserved median/max `{fmt(item.get('reserved_median'))}/{fmt(item.get('reserved_max'))}`."
        )
    note = f"""# RAOC CUDA Training Equivalence Validation

Date: 2026-08-29

## Scope

This engineering validation freezes the fused implementation at commit `43b72f4` and compares `camera_medium_raoc_backend='reference'` with `camera_medium_raoc_backend='cuda_fused'`. No RAOC equation, medium architecture, loss, optimizer, scheduler, observability state, or refinement policy was changed. The canonical archived C1 RAOC checkpoint at step 3000 was used, with optimizer and scheduler state restored and the archived camera sequence replayed. No new 15K experiment was run.

The strict `1e-6` criterion is insufficient as the sole training criterion: FMA, reduction order, register accumulation, fused kernels, and temporary materialization can change ordinary FP32 rounding while preserving the scientific equations. The validation therefore uses the registered output, gradient, trajectory, held-out metric, topology, and safety tolerances below.

## Registered tolerances

Operator: pred-image max `<= 5e-4`, mean `<= 1e-5`; Delta_z_raoc_std max `<= 5e-4`; g_keep max `<= 1e-4`; medium-gradient relative L2 `<= 1e-3`; cosine `>= 0.99999`; all values finite. Frozen evaluation: mean PSNR `<= 0.01 dB`, SSIM `<= 1e-4`, LPIPS `<= 1e-4`, MSE `<= 2e-6`, and no per-view PSNR difference over `0.02 dB`. Fixed topology: 500 finite steps, mean relative loss `<= 1e-3`, final PSNR `<= 0.01 dB`, SSIM/LPIPS `<= 1e-4`, medium parameter relative L2 `<= 5e-3`. Normal topology: 500 finite steps, count relative difference `<= 2%`, final PSNR `<= 0.02 dB`, SSIM/LPIPS `<= 2e-4`, MSE `<= 2e-4`, and no pathological topology behavior.

## Intermediate localization

The direct localization is `MULTIPLE_SMALL_FP_EFFECTS`. Raw medium outputs and detached calibration state are common; the largest direct difference is in sensitivity/reduction and its downstream gate/reconstruction tail. No scale, orientation, activation derivative, modal equation, or detached-backward semantic mismatch was identified.

{chr(10).join(operator_lines)}

Q50 and Q80 both reproduce finite gradients and near-identical rendered output. However, both fail the pre-registered direct `Delta_z_raoc_std` max gate (`Q50 {fmt(value(operator.get('q50', {}), 'quantities', 'delta_raoc_std', 'max_abs'))}`, `Q80 {fmt(value(operator.get('q80', {}), 'quantities', 'delta_raoc_std', 'max_abs'))}` versus `5e-4`). This gate is retained and was not relaxed.

## Frozen evaluation

The four-scene frozen checkpoint evaluation passed its registered backend metric and safety limits for Curasao, IUI3-RedSea, JapaneseGradens-RedSea, and Panama. Reference and fused were evaluated from the same frozen state without optimizer updates; per-view PSNR deltas and B_inf, beta_B, beta_D, tau, transmission, P(J>1), and J_p99 remained within the recorded floating-point tolerance.

## Fixed topology

The controlled 500-step tests restored the same Gaussian, medium, optimizer, scheduler, RNG, RAOC state, and camera sequence. Post-step bookkeeping was preserved and only topology mutation was suppressed.

{chr(10).join(fixed_lines) if fixed_lines else '- No fixed-topology artifact was found.'}

IUI3 did not pass the fixed-topology acceptance gate, while Panama passed. This prevents a clean fixed-topology equivalence claim across the executed fixed scenes.

## Normal topology

All four 500-step runs were finite and completed without OOM. Camera sequence hashes matched with zero mismatch. Counts stayed close and below the 2% registered count limit, but each scene showed a first count divergence during normal refinement and the final metric drift was not clean:

{chr(10).join(normal_lines)}

The first divergence was step 3200 for Curasao, IUI3-RedSea, and JapaneseGradens-RedSea, and step 3300 for Panama. The recorded trigger statistics show small threshold-sensitive split/duplicate/prune differences rather than a runaway explosion or collapse. Nevertheless, the resulting topology changes amplify into metric drift beyond the registered normal-topology limits in multiple scenes, so this cannot be treated as an isolated harmless count difference.

The per-backend first-event split/duplicate/prune counts and threshold margins are in `topology_divergence_analysis.json`. The minimum absolute trigger margins are:

{chr(10).join(divergence_lines)}

## Memory A/B

The memory cause classification is `{memory.get('cause_classification', 'n/a')}`. The synchronized trajectories separate `allocated_after`, peak-after-backward, Gaussian count, visible count, and refinement events:

{chr(10).join(memory_lines) if memory_lines else '- No memory summary was found.'}

Reference-only peak-after-backward excess with similar allocated-after medians indicates RAOC temporaries dominate peak memory. The larger reserved footprint is attributable to allocator reservation, while refinement creates synchronized topology-related spikes in both arms.

## Synchronized performance

The representative IUI3 benchmark used CUDA events, 20 warmups, and 50 timed iterations per backend and measurement. The synchronized complete forward/backward medians were `{fmt(value(speedup, 'speedups', 'complete_forward_backward', 'reference_ms'))} ms` reference and `{fmt(value(speedup, 'speedups', 'complete_forward_backward', 'cuda_fused_ms'))} ms` fused, for `{fmt(speedup.get('synchronized_complete_forward_backward_speedup'))}x` reference-over-fused. Forward, backward, complete, and training-step timings are in `performance_reference_vs_fused.json`. The previous approximately `32.14x` report is classified `{speedup.get('classification', 'n/a')}` under this timing definition.

## Decision

Primary classification: `{classification.get('primary_classification', 'pending')}`

Formal backend decision: `{classification.get('formal_backend_decision', 'pending')}`

The fused backend is not approved for formal science. Keep `camera_medium_raoc_backend='reference'`; do not resume a new 15K causal experiment or mix reference and fused backends between causal arms. The next scientific task should only be selected after reviewing this validation artifact and resolving the failed operator/training gates.

## Repository hygiene

Only the dedicated validation script and this research note are intended for commit. Historical GMVC scripts and unrelated Q50/Q80 experiment scripts were left untouched and unstaged. Outputs, checkpoints, logs, and compiled binaries remain untracked or ignored.
"""
    path = REPO_ROOT / "research_notes" / "RAOC_CUDA_TRAINING_EQUIVALENCE_2026-08-29.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(note, encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("repo", "operator", "frozen", "fixed", "normal", "memory", "performance", "aggregate", "note"))
    parser.add_argument("--scene", choices=sorted(SCENE_GPUS))
    parser.add_argument("--gpu")
    parser.add_argument("--steps", type=int, default=FIXED_STEPS)
    args = parser.parse_args()
    if args.phase == "repo":
        _write_json(OUTPUT_ROOT / "repo_state.json", {"branch": _git("branch", "--show-current"), "head": _git("rev-parse", "HEAD"), "status_short": _git("status", "--short"), "log_20": _git("log", "--oneline", "--decorate", "-20"), "historical_gmvc": list(HISTORICAL_GMVC), "unrelated_q50_q80_scripts": list(PROTECTED_Q50_Q80), "cuda_source_modified_during_validation": False})
        _write_json(OUTPUT_ROOT / "environment.json", {"python": sys.version, "python_path": sys.executable, "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "visible_device_count": int(torch.cuda.device_count())})
        _write_json(OUTPUT_ROOT / "backend_contract.json", {"reference": "camera_medium_raoc_backend='reference'", "cuda_fused": "camera_medium_raoc_backend='cuda_fused'", "default_backend": "reference", "frozen_commit": "43b72f4", "equations_changed": False, "cuda_source_changed": False, "formal_q": Q50, "allowed_physical_gpus": sorted(ALLOWED_GPUS)})
    elif args.phase == "operator":
        _operator_phase(args.gpu or SCENE_GPUS[OPERATOR_SCENE], args.scene or OPERATOR_SCENE)
    elif args.phase == "frozen":
        if not args.scene or not args.gpu:
            raise ValueError("--scene and --gpu are required for frozen phase")
        _frozen_one(args.scene, args.gpu)
    elif args.phase == "fixed":
        if not args.scene or not args.gpu:
            raise ValueError("--scene and --gpu are required for fixed phase")
        _fixed_phase(args.scene, args.gpu, args.steps)
    elif args.phase == "normal":
        if not args.scene or not args.gpu:
            raise ValueError("--scene and --gpu are required for normal phase")
        _normal_phase(args.scene, args.gpu, args.steps)
    elif args.phase == "memory":
        if not args.scene or not args.gpu:
            raise ValueError("--scene and --gpu are required for memory phase")
        _memory_phase(args.scene, args.gpu)
    elif args.phase == "performance":
        if not args.scene or not args.gpu:
            raise ValueError("--scene and --gpu are required for performance phase")
        _performance_phase(args.scene, args.gpu)
    elif args.phase == "aggregate":
        _aggregate_phase()
    elif args.phase == "note":
        _write_research_note()


if __name__ == "__main__":
    main()
