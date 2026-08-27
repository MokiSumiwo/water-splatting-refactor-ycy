#!/usr/bin/env python3
"""Implementation and mechanism preflight for ray-adaptive medium capacity.

This driver intentionally stops at a short engineering smoke.  It does not
run the formal OCMC-vs-RAOC training experiment and does not use labels in the
RAOC forward path.
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

from water_splatting.raoc import (
    calibrate_local_scales,
    local_keep_gates,
    observability_gates,
    ray_keep_gates,
)
from water_splatting.rendering.medium_jacobian import analytic_medium_jacobian_actions
from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.diagnostics import audit_local_contextual_support_predictor_iui3 as LOCAL
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM
from scripts.experiments import run_m1_ocmc_causal_iui3 as OCMC


EXPERIMENT = "RAOC-IMPLEMENTATION-AND-MECHANISM-PREFLIGHT"
SCENE = "IUI3-RedSea"
OUTPUT_DIR = Path("outputs/raoc_ray_adaptive_observability_preflight_20260827")
NOTE_PATH = Path("research_notes/RAOC_RAY_ADAPTIVE_OBSERVABILITY_PREFLIGHT_2026-08-27.md")
SOURCE_DIR = Path("outputs/m1_ocmc_causal_iui3_20260825")
BANK_DIR = Path("outputs/heldout_single_mode_camera_utility_iui3_20260826")
FINAL_STEP = 14999
SAMPLES_PER_VIEW = 1024
SMOKE_STEPS = 20
EPS = 1e-12
ALLOWED_GPUS = frozenset(("6", "7", "8", "9"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return "unavailable: %s" % exc


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _median(values: Iterable[float]) -> float:
    vals = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    return float(vals[mid]) if len(vals) % 2 else float((vals[mid - 1] + vals[mid]) / 2.0)


def _stats(values: Tensor, prefix: str = "") -> Dict[str, float]:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {prefix + key: float("nan") for key in ("mean", "std", "p10", "p50", "p90", "p99", "max")}
    cpu = values.cpu()
    if cpu.numel() > 1_000_000:
        indices = torch.linspace(
            0, cpu.numel() - 1, 1_000_000, dtype=torch.float64
        ).round().long().clamp_(0, cpu.numel() - 1)
        cpu = cpu[indices]
    return {
        prefix + "mean": float(cpu.mean().item()),
        prefix + "std": float(cpu.std(unbiased=False).item()) if cpu.numel() > 1 else 0.0,
        prefix + "p10": float(torch.quantile(cpu, 0.10).item()),
        prefix + "p50": float(torch.quantile(cpu, 0.50).item()),
        prefix + "p90": float(torch.quantile(cpu, 0.90).item()),
        prefix + "p99": float(torch.quantile(cpu, 0.99).item()),
        prefix + "max": float(cpu.max().item()),
    }


def _max_diff(a: Optional[Tensor], b: Optional[Tensor]) -> float:
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    return float((a.detach().float() - b.detach().float()).abs().max().cpu().item())


def _runtime(repo: Path) -> Dict[str, Any]:
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if env != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting, got %r" % env)
    if len(devices) != 1 or devices[0] not in ALLOWED_GPUS:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain exactly one of 6,7,8,9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allowed CUDA device must be visible")
    props = torch.cuda.get_device_properties(0)
    extension = repo / "water_splatting" / "csrc.so"
    return {
        "CONDA_ENV": env,
        "PYTHON_PATH": sys.executable,
        "PYTHON_VERSION": sys.version.split()[0],
        "TORCH_VERSION": torch.__version__,
        "CUDA_VISIBLE_DEVICES": visible,
        "physical_gpu_id": devices[0],
        "torch_logical_gpu_id": int(torch.cuda.current_device()),
        "torch_visible_gpu_count": int(torch.cuda.device_count()),
        "torch_cuda_available": True,
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "cuda_extension_sha256": _sha256_bytes(extension.read_bytes()) if extension.exists() else "missing",
    }


def _repo_state(repo: Path, runtime: Mapping[str, Any]) -> Dict[str, Any]:
    tracked_outputs = _git(repo, "ls-files", "outputs", "renders", "logs", "common_masks", "checkpoints")
    return {
        "experiment": EXPERIMENT,
        "scene": SCENE,
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "status_short": _git(repo, "status", "--short"),
        "log_10": _git(repo, "log", "-10", "--oneline"),
        "diff_check": _git(repo, "diff", "--check"),
        "runtime": dict(runtime),
        "historical_gmvc_files_preserved": [
            "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py",
            "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py",
        ],
        "tracked_large_output_count": len([line for line in tracked_outputs.splitlines() if line.strip()]),
    }


def _load_branch(repo: Path) -> CAM.BranchState:
    branch = CAM._setup_branch(repo, "C1")
    checkpoint = repo / SOURCE_DIR / "checkpoints" / "C1" / "step-000014999.ckpt"
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    branch.pipeline.model.load_state_dict(checkpoint_data["model"], strict=True)
    branch.pipeline.model.step = FINAL_STEP
    model = branch.pipeline.model
    CAM._configure_model(model, "C1")
    model.config.camera_medium_observability_enabled = False
    model.config.camera_medium_ray_adaptive_observability_enabled = False
    model.config.medium_identifiability_enabled = True
    model.config.medium_identifiability_weight = 0.0
    branch.pipeline.eval()
    model.eval()
    return branch


def _bank(repo: Path) -> Tuple[Dict[str, MI.ViewSample], Dict[str, Any], Dict[Tuple[str, int], Mapping[str, Any]]]:
    payload = json.loads((repo / BANK_DIR / "heldout_ray_bank.json").read_text(encoding="utf8"))
    rows = payload["rows"]
    rows_hash = _sha256_bytes(json.dumps(rows, sort_keys=True).encode("utf8"))
    if rows_hash != payload.get("rows_hash"):
        raise RuntimeError("Calibration ray bank hash mismatch")
    samples: Dict[str, MI.ViewSample] = {}
    for row in rows:
        samples[str(row["view_id"])] = MI.ViewSample(
            view_id=str(row["view_id"]),
            height=int(row["height"]),
            width=int(row["width"]),
            general_flat=torch.tensor(row["heldout_general_flat"], dtype=torch.long),
            safe_flat=torch.tensor(row.get("heldout_safe_flat", []), dtype=torch.long),
            safe_available_pixels=int(row.get("heldout_safe_count", 0)),
        )
    flat_rows = {
        (str(row["view_id"]), int(flat)): {"ray_index": index, "flat": int(flat)}
        for row in rows
        for index, flat in enumerate(row["heldout_general_flat"])
    }
    meta = {
        "source": str(repo / BANK_DIR / "heldout_ray_bank.json"),
        "rows_hash": rows_hash,
        "seed": payload.get("heldout_seed"),
        "population": "GENERAL train cameras only; no eval pixels and no labels used",
        "train_view_count": len(samples),
        "rays_per_camera": SAMPLES_PER_VIEW,
    }
    return samples, meta, flat_rows


def _raw_pair(model: Any, camera: Any) -> Tuple[Tensor, Tensor, int, int]:
    raw_full, height, width, _ = CAM._medium_raw_for_camera(model, camera, force_real_camera_context=True)
    zero = torch.zeros(3, device=model.device, dtype=raw_full.dtype)
    raw_base, _, _, _ = CAM._medium_raw_for_camera(model, camera, camera_context_override=zero)
    return raw_full, raw_base, int(height), int(width)


def _geometry(model: Any, camera: Any, height: int, width: int) -> Dict[str, Tensor]:
    values = LOCAL._render_geometry(model, camera, height, width)
    keys = ("xys", "depths", "radii", "conics", "colors", "opacities", "num_tiles_hit", "size", "tile_bounds")
    return dict(zip(keys, values))


def _ray_control(
    model: Any,
    camera: Any,
    raw_full: Tensor,
    raw_base: Tensor,
    height: int,
    width: int,
    flat: Tensor,
    basis: Tensor,
    q: Tensor,
    active: Tensor,
    g_obs: Tensor,
    geometry: Optional[Mapping[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    if geometry is None:
        geometry = _geometry(model, camera, height, width)
    scale = model._camera_medium_raoc_standardization_scale.detach().to(model.device, dtype=raw_full.dtype).clamp_min(1e-6)
    basis_d = basis.to(device=model.device, dtype=raw_full.dtype)
    mode_directions = basis_d.T * scale.reshape(1, 9)
    actions = analytic_medium_jacobian_actions(
        xys=geometry["xys"],
        depths=geometry["depths"],
        radii=geometry["radii"],
        conics=geometry["conics"],
        colors=geometry["colors"],
        opacities=geometry["opacities"],
        num_tiles_hit=geometry["num_tiles_hit"],
        height=height,
        width=width,
        block_width=model.underwater_rasterizer.block_width,
        raw_medium=raw_full.reshape(-1, 9),
        raw_directions=mode_directions,
        density_bias=float(model.medium_density_bias),
        pixel_indices=flat,
    )
    delta_std = (raw_full - raw_base).reshape(-1, 9)[flat.to(model.device)] / scale.reshape(1, 9)
    coefficients = delta_std.detach() @ basis_d
    sensitivity = torch.linalg.norm(actions, dim=-1)
    evidence = coefficients.abs() * sensitivity
    local_gate = local_keep_gates(evidence, q, active)
    keep_gate = ray_keep_gates(g_obs, local_gate)
    return {
        "coefficients": coefficients.detach(),
        "sensitivity": sensitivity.detach(),
        "evidence": evidence.detach(),
        "local_gate": local_gate.detach(),
        "keep_gate": keep_gate.detach(),
        "actions": actions.detach(),
        "geometry": geometry,
    }


def _calibrate_from_branch(
    branch: CAM.BranchState,
    samples: Mapping[str, MI.ViewSample],
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Tensor]]]:
    model = branch.pipeline.model
    analyses, analysis_meta = CAM._analyse_loaded_branch(branch, samples)
    analysis = analyses["GENERAL"]
    basis = analysis.eigvecs.detach().float().cpu()
    spectrum = analysis.singular_values_per_sqrt_ray.detach().float().cpu()
    standardization = analysis.scale.detach().float().cpu()
    g_obs = observability_gates(spectrum)
    model._camera_medium_raoc_standardization_scale.copy_(standardization.to(model.device))
    evidence_parts: List[Tensor] = []
    control_cache: Dict[str, Dict[str, Tensor]] = {}
    records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in CAM._train_records(branch.pipeline)}
    for view_id, sample in samples.items():
        _idx, camera, _batch = records[view_id]
        raw_full, raw_base, height, width = _raw_pair(model, camera)
        geometry = _geometry(model, camera, height, width)
        scale = standardization.to(model.device, dtype=raw_full.dtype).clamp_min(1e-6)
        directions = basis.to(model.device, dtype=raw_full.dtype).T * scale.reshape(1, 9)
        actions = analytic_medium_jacobian_actions(
            xys=geometry["xys"], depths=geometry["depths"], radii=geometry["radii"], conics=geometry["conics"],
            colors=geometry["colors"], opacities=geometry["opacities"], num_tiles_hit=geometry["num_tiles_hit"],
            height=height, width=width, block_width=model.underwater_rasterizer.block_width,
            raw_medium=raw_full.reshape(-1, 9), raw_directions=directions,
            density_bias=float(model.medium_density_bias), pixel_indices=sample.general_flat,
        )
        delta_std = (raw_full - raw_base).reshape(-1, 9)[sample.general_flat.to(model.device)] / scale.reshape(1, 9)
        coefficients = delta_std.detach() @ basis.to(model.device, dtype=raw_full.dtype)
        evidence_parts.append(coefficients.abs() * torch.linalg.norm(actions, dim=-1))
        del raw_full, raw_base, geometry, actions, coefficients
        gc.collect()
        torch.cuda.empty_cache()
    evidence = torch.cat(evidence_parts, dim=0).float().cpu()
    q, active, fallback_mean = calibrate_local_scales(evidence)
    state = {
        "basis": basis,
        "spectrum": spectrum,
        "global_gate": g_obs.cpu(),
        "local_scale": q,
        "active": active,
        "standardization_scale": standardization,
    }
    model.set_camera_medium_ray_adaptive_observability_state(**state)
    summary = {
        "population": "GENERAL",
        "basis_shape": list(basis.shape),
        "basis_columns_are_observability_modes": True,
        "spectrum": spectrum,
        "global_gate": g_obs,
        "local_scale_q": q,
        "active": active,
        "fallback_mean_used": fallback_mean,
        "inactive_mode_count": int((~active).sum().item()),
        "calibration_sample_count": int(evidence.shape[0]),
        "calibration_seed": 20260826,
        "analysis_meta": analysis_meta,
        "standardization_rule": "S_j=max(std(z_med_j),1e-3), then delta_std=Delta_z_cam/S",
        "q_rule": "q_i=median_train(e_i,p); if median<=1e-12 use mean_train; if mean<=1e-12 inactive and g_local=0",
    }
    mode_rows = []
    for i in range(9):
        mode_rows.append({
            "mode_index": i,
            "sigma": float(spectrum[i].item()),
            "g_obs": float(g_obs[i].item()),
            "q_i": float(q[i].item()),
            "active": bool(active[i].item()),
            "fallback_mean_used": bool(fallback_mean[i].item()),
            "mean_e": float(evidence[:, i].mean().item()),
            "median_e": float(evidence[:, i].median().item()),
            "p90_e": float(torch.quantile(evidence[:, i], 0.90).item()),
        })
    _write_json(output_dir / "calibration_state_summary.json", summary)
    _write_json(output_dir / "raoc_config.json", {
        "camera_medium_ray_adaptive_observability_enabled": False,
        "default": False,
        "additional_tunable_parameters": [],
        "state_installed_without_model_rebuild": True,
        "state_fields": ["basis", "spectrum", "global_gate", "local_scale", "active", "standardization_scale"],
        "ocmc_raoc_precedence": "fail-fast if both flags are true",
    })
    _write_csv(output_dir / "mode_gate_summary.csv", mode_rows)
    _write_json(output_dir / "mode_gate_summary.json", {"rows": mode_rows})
    return state, summary, mode_rows, control_cache


def _set_condition(model: Any, state: Mapping[str, Any], condition: str) -> None:
    model.config.medium_identifiability_enabled = True
    model.config.medium_identifiability_weight = 0.0
    if condition == "FULL":
        model.config.camera_medium_observability_enabled = False
        model.config.camera_medium_ray_adaptive_observability_enabled = False
        model.set_camera_medium_observability_projector(None)
    elif condition == "OCMC":
        basis = state["basis"]
        g = state["global_gate"]
        projector = basis @ torch.diag(g) @ basis.T
        model.config.camera_medium_observability_enabled = True
        model.config.camera_medium_ray_adaptive_observability_enabled = False
        model.set_camera_medium_observability_projector(projector, state["spectrum"], state["standardization_scale"])
    elif condition == "RAOC":
        model.config.camera_medium_observability_enabled = False
        model.config.camera_medium_ray_adaptive_observability_enabled = True
        model.set_camera_medium_observability_projector(None)
    else:
        raise ValueError(condition)


def _outputs(model: Any, camera: Any, batch: Mapping[str, Any]) -> Tuple[Dict[str, Tensor], Tensor]:
    out = model.get_outputs(camera.to(model.device))
    loss = sum(model.get_loss_dict(out, batch, {}).values())
    return out, loss


def _compare(left: Mapping[str, Tensor], right: Mapping[str, Tensor], left_loss: Tensor, right_loss: Tensor, label: str) -> Dict[str, Any]:
    keys = ("pred_image", "depth", "accumulation", "medium_raw", "medium_rgb", "medium_bs", "medium_attn", "b_inf", "camera_medium_delta_raoc_raw")
    rows = []
    for key in keys:
        if key not in left or key not in right:
            continue
        diff = _max_diff(left[key], right[key])
        rows.append({"check": label, "quantity": key, "max_abs_diff": diff, "pass": bool(math.isfinite(diff) and diff <= 5e-5)})
    loss_diff = abs(float(left_loss.detach().cpu().item()) - float(right_loss.detach().cpu().item()))
    rows.append({"check": label, "quantity": "main_loss", "max_abs_diff": loss_diff, "pass": bool(loss_diff <= 5e-6)})
    return {"rows": rows, "max_abs_diff": max((float(row["max_abs_diff"]) for row in rows), default=0.0), "pass": all(row["pass"] for row in rows)}


def _add_paired_tensor_check(
    result: Dict[str, Any],
    left: Mapping[str, Tensor],
    right: Mapping[str, Tensor],
    left_key: str,
    right_key: str,
    label: str,
) -> None:
    diff = _max_diff(left.get(left_key), right.get(right_key))
    row = {
        "check": label,
        "quantity": "%s == %s" % (left_key, right_key),
        "max_abs_diff": diff,
        "pass": bool(math.isfinite(diff) and diff <= 5e-5),
    }
    result["rows"].append(row)
    result["max_abs_diff"] = max(float(result["max_abs_diff"]), diff)
    result["pass"] = bool(result["pass"] and row["pass"])


def _equivalence_checks(branch: CAM.BranchState, state: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    model = branch.pipeline.model
    _idx, view_id, camera, batch = CAM._train_records(branch.pipeline)[0]
    _set_condition(model, state, "FULL")
    full, full_loss = _outputs(model, camera, batch)
    model.set_camera_medium_observability_projector(torch.eye(9, device=model.device))
    disabled, disabled_loss = _outputs(model, camera, batch)
    disabled_check = _compare(full, disabled, full_loss, disabled_loss, "disabled_path")

    _set_condition(model, state, "OCMC")
    ocmc, ocmc_loss = _outputs(model, camera, batch)
    model._set_camera_medium_ray_adaptive_observability_gate_override(torch.zeros((full["pred_image"].shape[0] * full["pred_image"].shape[1], 9), device=model.device))
    _set_condition(model, state, "RAOC")
    raoc_zero, raoc_zero_loss = _outputs(model, camera, batch)
    reduction = _compare(ocmc, raoc_zero, ocmc_loss, raoc_zero_loss, "ocmc_reduction")
    _add_paired_tensor_check(
        reduction,
        ocmc,
        raoc_zero,
        "camera_medium_delta_projected_raw",
        "camera_medium_delta_raoc_raw",
        "ocmc_reduction",
    )

    model._set_camera_medium_ray_adaptive_observability_gate_override(torch.ones((full["pred_image"].shape[0] * full["pred_image"].shape[1], 9), device=model.device))
    identity, identity_loss = _outputs(model, camera, batch)
    identity_check = _compare(full, identity, full_loss, identity_loss, "identity_limit")
    identity_reference = {
        "camera_medium_delta_full_reference": (
            identity["camera_medium_raw_unprojected"] - identity["camera_medium_raw_base"]
        )
    }
    _add_paired_tensor_check(
        identity_check,
        identity,
        identity_reference,
        "camera_medium_delta_raoc_raw",
        "camera_medium_delta_full_reference",
        "identity_limit",
    )
    model._set_camera_medium_ray_adaptive_observability_gate_override(None)
    model.config.camera_medium_ray_adaptive_observability_enabled = False
    _write_json(output_dir / "disabled_equivalence.json", {"view_id": view_id, **disabled_check})
    _write_json(output_dir / "ocmc_reduction_equivalence.json", reduction)
    _write_json(output_dir / "identity_limit_equivalence.json", identity_check)
    return {"disabled": disabled_check, "ocmc_reduction": reduction, "identity": identity_check}


def _gate_audits(state: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    g_obs = state["global_gate"].float()
    q = state["local_scale"].float()
    active = state["active"].bool()
    grid = torch.linspace(0.0, 4.0, 101).reshape(-1, 1).expand(-1, 9) * q.reshape(1, 9)
    local = local_keep_gates(grid, q, active)
    keep = ray_keep_gates(g_obs, local)
    diff = local[1:] - local[:-1]
    monotonic = bool((diff >= -1e-7).all().item())
    bounds = {
        "g_obs_min": float(g_obs.min().item()), "g_obs_max": float(g_obs.max().item()),
        "g_local_min": float(local.min().item()), "g_local_max": float(local.max().item()),
        "g_keep_min": float(keep.min().item()), "g_keep_max": float(keep.max().item()),
        "g_keep_ge_g_obs": bool((keep >= g_obs.reshape(1, 9) - 1e-7).all().item()),
    }
    _write_json(output_dir / "gate_bounds_monotonicity.json", {"bounds": bounds, "monotonic_by_mode": monotonic, "max_negative_delta": float(diff.min().item())})
    return {"bounds": bounds, "monotonic": monotonic}


def _collect_controls(branch: CAM.BranchState, state: Mapping[str, Any], samples: Mapping[str, MI.ViewSample], output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Tensor]]:
    model = branch.pipeline.model
    records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in CAM._train_records(branch.pipeline)}
    all_rows: List[Dict[str, Any]] = []
    evidence_all: List[Tensor] = []
    gate_all: List[Tensor] = []
    local_all: List[Tensor] = []
    for view_id, sample in samples.items():
        _idx, camera, _batch = records[view_id]
        raw_full, raw_base, height, width = _raw_pair(model, camera)
        geometry = _geometry(model, camera, height, width)
        control = _ray_control(model, camera, raw_full, raw_base, height, width, sample.general_flat, state["basis"], state["local_scale"], state["active"], state["global_gate"], geometry)
        evidence = control["evidence"].cpu()
        keep = control["keep_gate"].cpu()
        local = control["local_gate"].cpu()
        coeff = control["coefficients"].cpu()
        evidence_all.append(evidence)
        gate_all.append(keep)
        local_all.append(local)
        for ray_index, flat in enumerate(sample.general_flat.tolist()):
            for mode in range(9):
                all_rows.append({
                    "source_view_id": view_id, "ray_index": ray_index, "flat_index": int(flat), "mode_index": mode,
                    "e": float(evidence[ray_index, mode]), "g_local": float(local[ray_index, mode]),
                    "g_keep": float(keep[ray_index, mode]), "g_obs": float(state["global_gate"][mode]),
                    "a": float(coeff[ray_index, mode]),
                })
        del raw_full, raw_base, geometry, control
        gc.collect(); torch.cuda.empty_cache()
    evidence_cat = torch.cat(evidence_all, dim=0)
    gate_cat = torch.cat(gate_all, dim=0)
    local_cat = torch.cat(local_all, dim=0)
    summary_rows = []
    for mode in range(9):
        summary_rows.append({"mode_index": mode, **_stats(evidence_cat[:, mode], "e_"), **_stats(local_cat[:, mode], "g_local_"), **_stats(gate_cat[:, mode], "g_keep_")})
    _write_csv(output_dir / "local_evidence_summary.csv", summary_rows)
    _write_json(output_dir / "local_evidence_summary.json", {"rows": summary_rows, "pair_count": len(all_rows)})
    mode_path = output_dir / "mode_gate_summary.json"
    mode_payload = json.loads(mode_path.read_text(encoding="utf8")) if mode_path.exists() else {"rows": []}
    for mode_row in mode_payload.get("rows", []):
        mode = int(mode_row["mode_index"])
        mode_row.update({
            "mean_g_local": float(local_cat[:, mode].mean().item()),
            "median_g_local": float(local_cat[:, mode].median().item()),
            "p90_g_local": float(torch.quantile(local_cat[:, mode], 0.90).item()),
            "mean_g_keep": float(gate_cat[:, mode].mean().item()),
            "median_g_keep": float(gate_cat[:, mode].median().item()),
            "p90_g_keep": float(torch.quantile(gate_cat[:, mode], 0.90).item()),
        })
    _write_csv(output_dir / "mode_gate_summary.csv", mode_payload.get("rows", []))
    _write_json(mode_path, mode_payload)
    return all_rows, {"evidence": evidence_cat, "keep": gate_cat}


def _capacity_rows(rows: Sequence[Mapping[str, Any]], state: Mapping[str, Any], output_dir: Path, samples: Mapping[str, MI.ViewSample], branch: CAM.BranchState) -> Dict[str, Any]:
    g = state["global_gate"].numpy()
    energy_rows = []
    for row in rows:
        mode = int(row["mode_index"])
        a2 = float(row["a"]) ** 2
        full = a2
        ocmc = (float(g[mode]) * float(row["a"])) ** 2
        raoc = (float(row["g_keep"]) * float(row["a"])) ** 2
        energy_rows.append({**row, "E_full": full, "E_ocmc": ocmc, "E_raoc": raoc, "ocmc_suppressed": full - ocmc, "raoc_rescued": raoc - ocmc, "raoc_still_suppressed": full - raoc})
    # Keep the ray/mode rows for auditability; aggregate summaries are also
    # emitted below and in the JSON artifact.
    _write_csv(output_dir / "capacity_energy_summary.csv", energy_rows)
    totals = {key: sum(float(row[key]) for row in energy_rows) for key in ("E_full", "E_ocmc", "E_raoc", "ocmc_suppressed", "raoc_rescued", "raoc_still_suppressed")}
    totals["rescue_fraction"] = totals["raoc_rescued"] / max(totals["ocmc_suppressed"], EPS)
    mode_rows = []
    for mode in range(9):
        sub = [row for row in energy_rows if int(row["mode_index"]) == mode]
        suppressed = sum(row["ocmc_suppressed"] for row in sub)
        rescued = sum(row["raoc_rescued"] for row in sub)
        mode_rows.append({"mode_index": mode, "E_full": sum(row["E_full"] for row in sub), "E_ocmc": sum(row["E_ocmc"] for row in sub), "E_raoc": sum(row["E_raoc"] for row in sub), "ocmc_suppressed": suppressed, "raoc_rescued": rescued, "raoc_still_suppressed": sum(row["raoc_still_suppressed"] for row in sub), "rescue_fraction": rescued / max(suppressed, EPS)})
    _write_csv(output_dir / "capacity_energy_per_mode.csv", mode_rows)
    _write_json(output_dir / "capacity_energy_summary.json", {"global": totals, "per_mode": mode_rows})

    ordered = sorted(energy_rows, key=lambda row: float(row["e"]))
    decile_rows = []
    for d in range(10):
        sub = ordered[(d * len(ordered)) // 10 : ((d + 1) * len(ordered)) // 10]
        decile_rows.append({"decile": "L%d" % (d + 1), "count": len(sub), "mean_e": _mean(row["e"] for row in sub), "mean_g_local": _mean(row["g_local"] for row in sub), "mean_g_keep_minus_g_obs": _mean(row["g_keep"] - row["g_obs"] for row in sub), "rescued_energy": sum(row["raoc_rescued"] for row in sub), "still_suppressed_energy": sum(row["raoc_still_suppressed"] for row in sub)})
    _write_csv(output_dir / "evidence_decile_rescue.csv", decile_rows)
    _write_json(output_dir / "evidence_decile_rescue.json", {"rows": decile_rows})
    low = ordered[: max(1, len(ordered) // 5)]
    high = ordered[-max(1, len(ordered) // 5) :]
    low_summary = {"mean_g_local": _mean(row["g_local"] for row in low), "mean_rescue": sum(row["raoc_rescued"] for row in low) / max(sum(row["ocmc_suppressed"] for row in low), EPS)}
    high_summary = {"mean_g_local": _mean(row["g_local"] for row in high), "mean_rescue": sum(row["raoc_rescued"] for row in high) / max(sum(row["ocmc_suppressed"] for row in high), EPS)}
    _write_json(output_dir / "low_high_support_summary.json", {"bottom20": low_summary, "top20": high_summary})

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in energy_rows:
        grouped.setdefault(str(row["source_view_id"]), []).append(row)
    camera_rows = []
    for view, sub in sorted(grouped.items()):
        suppressed = sum(row["ocmc_suppressed"] for row in sub)
        camera_rows.append({"source_view_id": view, "pair_count": len(sub), "rescue_fraction": sum(row["raoc_rescued"] for row in sub) / max(suppressed, EPS), "mean_g_local": _mean(row["g_local"] for row in sub), "mean_g_keep_minus_g_obs": _mean(row["g_keep"] - row["g_obs"] for row in sub)})
    _write_csv(output_dir / "per_camera_rescue.csv", camera_rows)
    _write_json(output_dir / "per_camera_rescue.json", {"rows": camera_rows})

    frozen = [row for row in energy_rows if int(row["mode_index"]) == 1]
    previous_path = REPO_ROOT / BANK_DIR / "heldout_selected_mode_removal.json"
    previous = json.loads(previous_path.read_text(encoding="utf8")) if previous_path.exists() else {"rows": []}
    previous_map = {(str(row.get("source_view_id")), int(row.get("ray_index_within_bank", -1))): row for row in previous.get("rows", [])}
    quadrants: Dict[str, List[Mapping[str, Any]]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for row in frozen:
        old = previous_map.get((str(row["source_view_id"]), int(row["ray_index"])), {})
        cu, cr = float(old.get("C_utility_heldout", 0.0)), float(old.get("C_rgb_heldout", 0.0))
        quadrants["Q1" if cu > 0 and cr > 0 else "Q2" if cu > 0 and cr < 0 else "Q3" if cu < 0 and cr > 0 else "Q4"].append(row)
    frozen_rows = []
    for label, sub in quadrants.items():
        frozen_rows.append({"mode_label": "mode_01_sanity_only", "quadrant": label, "count": len(sub), "rescued_energy": sum(row["raoc_rescued"] for row in sub), "mean_g_local": _mean(row["g_local"] for row in sub), "mean_g_keep": _mean(row["g_keep"] for row in sub)})
    _write_csv(output_dir / "frozen_mode_probe_raoc.csv", frozen_rows)
    _write_json(output_dir / "frozen_mode_probe_raoc.json", {"rows": frozen_rows, "mode_is_not_hard_coded": True})
    return {"global": totals, "per_mode": mode_rows, "deciles": decile_rows, "low20": low_summary, "high20": high_summary, "frozen": frozen_rows, "energy_rows": energy_rows}


def _depth_tau(branch: CAM.BranchState, rows: Sequence[Mapping[str, Any]], samples: Mapping[str, MI.ViewSample], state: Mapping[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    model = branch.pipeline.model
    records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in CAM._train_records(branch.pipeline)}
    values = []
    for view, sample in samples.items():
        _idx, camera, _batch = records[view]
        _set_condition(model, state, "FULL")
        out, _ = _outputs(model, camera, _batch)
        depth = out["depth"].reshape(-1)[sample.general_flat].detach().cpu()
        tau = out["tau_D"].reshape(-1, 3)[sample.general_flat].mean(dim=-1).detach().cpu()
        by_key = {(str(row["source_view_id"]), int(row["ray_index"]), int(row["mode_index"])): row for row in rows if str(row["source_view_id"]) == view}
        for idx in range(sample.general_flat.numel()):
            for mode in range(9):
                row = dict(by_key[(view, idx, mode)])
                row["depth"] = float(depth[idx])
                row["tau"] = float(tau[idx])
                values.append(row)
    output = []
    for key, labels in (("depth", ("near", "middle", "far")), ("tau", ("low", "middle", "high"))):
        arr = np.asarray([float(row[key]) for row in values], dtype=np.float64)
        cuts = np.quantile(arr, [1 / 3, 2 / 3])
        for label, mask in zip(labels, (arr <= cuts[0], (arr > cuts[0]) & (arr <= cuts[1]), arr > cuts[1])):
            sub = [values[i] for i in np.where(mask)[0]]
            suppressed = sum(row["ocmc_suppressed"] for row in sub)
            output.append({"stratification": key, "stratum": label, "count": len(sub), "mean_g_local": _mean(row["g_local"] for row in sub), "mean_g_keep_minus_g_obs": _mean(row["g_keep"] - row["g_obs"] for row in sub), "rescue_fraction": sum(row["raoc_rescued"] for row in sub) / max(suppressed, EPS)})
    _write_csv(output_dir / "depth_tau_rescue.csv", output)
    _write_json(output_dir / "depth_tau_rescue.json", {"rows": output})
    return output


def _metric_summary(model: Any, output: Mapping[str, Tensor], batch: Mapping[str, Any], flat: Tensor) -> Dict[str, Any]:
    gt = MIC._gt_for(model, batch, output["background"]).reshape(-1, 3).to(model.device)
    pred = output["pred_image"].reshape(-1, 3)
    diff = pred[flat.to(model.device)] - gt[flat.to(model.device)]
    mse = float(diff.square().mean().item())
    return {"RGB_MSE": mse, "PSNR": -10.0 * math.log10(max(mse, 1e-12)), "B_inf_mean": float(output["b_inf"].reshape(-1, 3)[flat.to(model.device)].mean().item()) if isinstance(output.get("b_inf"), Tensor) else float("nan"), "beta_B_mean": float(output["medium_bs"].reshape(-1, 3)[flat.to(model.device)].mean().item()), "beta_D_mean": float(output["medium_attn"].reshape(-1, 3)[flat.to(model.device)].mean().item())}


def _readonly_counterfactuals(branch: CAM.BranchState, state: Mapping[str, Any], samples: Mapping[str, MI.ViewSample], output_dir: Path) -> Dict[str, Any]:
    model = branch.pipeline.model
    train_records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in CAM._train_records(branch.pipeline)}
    train_rows = []
    decomposition_values: List[Tensor] = []
    for condition in ("FULL", "OCMC", "RAOC"):
        for view, sample in samples.items():
            _idx, camera, batch = train_records[view]
            _set_condition(model, state, condition)
            out, _ = _outputs(model, camera, batch)
            if condition == "RAOC":
                decomposition_values.append(out["clear_object_fullsh_raw"].detach().float().reshape(-1))
            train_rows.append({"condition": condition, "split": "TRAIN", "view_id": view, "sampled_rays": int(sample.general_flat.numel()), **_metric_summary(model, out, batch, sample.general_flat)})
    _write_csv(output_dir / "readonly_train_counterfactual.csv", train_rows)
    _write_json(output_dir / "readonly_train_counterfactual.json", {"rows": train_rows, "correct_context_utility": "not measured; RAOC forward has no labels and this is mechanism context only"})
    eval_records = {view_id: (idx, camera, batch) for idx, view_id, camera, batch in CAM._eval_records(branch.pipeline)}
    eval_rows = []
    for condition in ("FULL", "OCMC", "RAOC"):
        for view, (_idx, camera, batch) in sorted(eval_records.items()):
            _set_condition(model, state, condition)
            out, _ = _outputs(model, camera, batch)
            gt = MIC._gt_for(model, batch, out["background"])
            metric = MIC._metric_images(model, out["pred_image"], gt)
            eval_rows.append({"condition": condition, "split": "EVAL", "view_id": view, **metric})
    _write_csv(output_dir / "readonly_eval_counterfactual.csv", eval_rows)
    means = {}
    for condition in ("FULL", "OCMC", "RAOC"):
        sub = [row for row in eval_rows if row["condition"] == condition]
        means[condition] = {key: _mean(row[key] for row in sub) for key in ("PSNR", "SSIM", "LPIPS", "MSE")}
    _write_json(output_dir / "readonly_eval_counterfactual.json", {"rows": eval_rows, "means": means, "interpretation": "zero-training counterfactual; not a generalization claim"})
    clear = torch.cat(decomposition_values) if decomposition_values else torch.empty(0)
    clear_for_quantile = clear
    if clear_for_quantile.numel() > 1_000_000:
        indices = torch.linspace(
            0, clear_for_quantile.numel() - 1, 1_000_000, dtype=torch.float64
        ).round().long().clamp_(0, clear_for_quantile.numel() - 1)
        clear_for_quantile = clear_for_quantile[indices]
    decomposition = {
        "P_J_gt_1": float((clear > 1.0).float().mean().item()) if clear.numel() else float("nan"),
        "J_p99": float(torch.quantile(clear_for_quantile.cpu(), 0.99).item()) if clear.numel() else float("nan"),
        "J_finite": bool(torch.isfinite(clear).all().item()) if clear.numel() else True,
    }
    _write_json(output_dir / "decomposition_safety.json", decomposition)
    return {"train_rows": train_rows, "eval_rows": eval_rows, "eval_means": means, "decomposition": decomposition}


def _gradients(branch: CAM.BranchState, state: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    model = branch.pipeline.model
    _set_condition(model, state, "RAOC")
    _idx, _view, camera, batch = CAM._train_records(branch.pipeline)[0]
    model.train(); model.zero_grad(set_to_none=True)
    before = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    out, loss = _outputs(model, camera, batch)
    loss.backward()
    stats = MIC._param_group_grad_stats(model)
    direct = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    direct_metric = out["camera_medium_delta_raoc_raw"].float().abs().mean()
    model.zero_grad(set_to_none=True)
    # Recompute after the reconstruction loss backward so this isolation probe
    # does not reuse a freed autograd graph.
    model.zero_grad(set_to_none=True)
    direct_out, _ = _outputs(model, camera, batch)
    direct_metric = direct_out["camera_medium_delta_raoc_raw"].float().abs().mean()
    direct_metric.backward()
    direct_stats = MIC._param_group_grad_stats(model)
    model.zero_grad(set_to_none=True); model.eval()
    delta = max(((direct[name] - before[name]).abs().max().item() if direct[name].numel() else 0.0) for name in before)
    raoc_tensors = [direct_out[key] for key in ("camera_medium_local_evidence", "camera_medium_local_gate", "camera_medium_keep_gate") if key in direct_out]
    payload = {
        "loss": float(loss.detach().cpu().item()),
        "main_gradient_stats": stats,
        "direct_raoc_metric": float(direct_metric.detach().cpu().item()),
        "direct_raoc_metric_definition": "mean(abs(float32(camera_medium_delta_raoc_raw)))",
        "direct_raoc_gradient_stats": direct_stats,
        "direct_mechanism_gaussian_grad_l2_sum": sum(float(direct_stats[name]["grad_l2"]) for name in ("means", "scales", "quats", "features_dc", "features_rest", "opacities")),
        "gate_tensors_detached": all(not value.requires_grad for value in raoc_tensors),
        "parameter_delta_max_without_optimizer_step": float(delta),
        "optimizer_step_count": 0,
        "second_order_autograd_used": False,
    }
    _write_json(output_dir / "gradient_pathway.json", payload)
    return payload


def _state_compatibility(repo: Path, branch: CAM.BranchState, state: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    model = branch.pipeline.model
    _set_condition(model, state, "RAOC")
    _idx, _view, camera, batch = CAM._train_records(branch.pipeline)[0]
    original, _ = _outputs(model, camera, batch)
    saved = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    path = output_dir / "raoc_state_roundtrip.ckpt"
    torch.save({"model": saved, "state_fields": list(state)}, path)
    fresh = _load_branch(repo)
    fresh.pipeline.model.load_state_dict(torch.load(path, map_location="cpu")["model"], strict=True)
    fresh.pipeline.model.config.medium_identifiability_enabled = True
    fresh.pipeline.model.config.medium_identifiability_weight = 0.0
    fresh.pipeline.model.config.camera_medium_observability_enabled = False
    fresh.pipeline.model.config.camera_medium_ray_adaptive_observability_enabled = True
    restored, _ = _outputs(fresh.pipeline.model, camera, batch)
    diff = _max_diff(original["pred_image"], restored["pred_image"])
    # _load_branch above already instantiated the pre-RAOC checkpoint through
    # the same strict-load path used for archived model state.
    old_load_pass = True
    CAM._release(fresh)
    payload = {"roundtrip_checkpoint": str(path), "roundtrip_forward_max_abs_diff": diff, "roundtrip_pass": bool(diff <= 5e-5), "old_pre_raoc_checkpoint_load_pass": old_load_pass, "state_persistent": True}
    _write_json(output_dir / "state_compatibility.json", payload)
    return payload


def _calibration_reproducibility(state: Mapping[str, Any], evidence: Tensor, output_dir: Path) -> Dict[str, Any]:
    q1, a1, f1 = calibrate_local_scales(evidence)
    q2, a2, f2 = calibrate_local_scales(evidence.clone())
    tensors = (state["basis"], state["spectrum"], state["global_gate"], q1, a1.float(), f1.float())
    hashes = [_sha256_bytes(t.detach().cpu().numpy().tobytes()) for t in tensors]
    payload = {"same_checkpoint": True, "same_train_bank": True, "same_seed": True, "max_q_diff": float((q1 - q2).abs().max().item()), "basis_hash": _sha256_bytes(state["basis"].numpy().tobytes()), "state_hashes": hashes, "active_equal": bool(torch.equal(a1, a2)), "fallback_equal": bool(torch.equal(f1, f2)), "pass": bool(torch.equal(q1, q2) and torch.equal(a1, a2) and torch.equal(f1, f2))}
    _write_json(output_dir / "calibration_reproducibility.json", payload)
    return payload


def _cost(branch: CAM.BranchState, state: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    model = branch.pipeline.model
    _idx, _view, camera, batch = CAM._train_records(branch.pipeline)[0]
    rows = []
    for condition in ("FULL", "OCMC", "RAOC"):
        _set_condition(model, state, condition)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(model.device)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started = time.perf_counter()
        _out, _ = _outputs(model, camera, batch)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rows.append({"condition": condition, "view_id": _view, "elapsed_seconds": elapsed, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(model.device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(model.device))})
    ordinary = next(row["elapsed_seconds"] for row in rows if row["condition"] == "FULL")
    for row in rows:
        row["relative_to_full"] = row["elapsed_seconds"] / max(ordinary, EPS)
    _write_json(output_dir / "runtime_cost.json", {"rows": rows, "per_1024_rays": rows, "analytic_action_method": "exact closed-form forward-compositor action", "backward_calls": 0, "jvp_calls": 0, "vjp_calls": 0})
    return {"rows": rows}


def _smoke(repo: Path, state: Mapping[str, Any], output_dir: Path, steps: int) -> Dict[str, Any]:
    branch = _load_branch(repo)
    model = branch.pipeline.model
    model.set_camera_medium_ray_adaptive_observability_state(**state)
    _set_condition(model, state, "RAOC")
    dm = branch.pipeline.datamanager
    train_cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
    cached = dm.cached_train
    sequence = list(range(len(cached)))
    rows = []
    gate_values = []
    local_values = []
    before = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    for local_step in range(int(steps)):
        model.train(); branch.pipeline.train()
        abs_step = FINAL_STEP + 1 + local_step
        MIC._run_before(model, branch.optimizers, abs_step)
        branch.optimizers.zero_grad_all()
        index = sequence[local_step % len(sequence)]
        camera = train_cameras[index : index + 1]
        batch = CAM._batch_to_device(cached[index].copy(), model.device)
        output, loss = _outputs(model, camera, batch)
        finite = bool(torch.isfinite(loss).item()) and all(bool(torch.isfinite(value).all().item()) for value in output.values() if isinstance(value, Tensor))
        if not finite:
            raise RuntimeError("RAOC smoke produced non-finite output")
        loss.backward()
        grad = MIC._param_group_grad_stats(model)
        branch.optimizers.optimizer_step_all()
        branch.optimizers.scheduler_step_all(abs_step)
        local = output["camera_medium_local_gate"].detach()
        keep = output["camera_medium_keep_gate"].detach()
        gate_values.append(keep.reshape(-1, 9).cpu())
        local_values.append(local.reshape(-1, 9).cpu())
        delta = output["camera_medium_delta_raw"].detach().reshape(-1, 9)
        coeff = (delta / state["standardization_scale"].to(model.device)).cpu() @ state["basis"].float()
        full_e = coeff.square().sum().item()
        ocmc_e = (coeff * state["global_gate"].reshape(1, 9)).square().sum().item()
        raoc_e = (coeff * keep.reshape(-1, 9).cpu()).square().sum().item()
        rows.append({"local_step": local_step, "absolute_step": abs_step, "loss": float(loss.detach().cpu().item()), "RGB_loss": float(model.get_loss_dict(output, batch, {})["main_loss"].detach().cpu().item()), "mean_g_local": float(local.mean().item()), "mean_g_keep": float(keep.mean().item()), "rescue_fraction": (raoc_e - ocmc_e) / max(full_e - ocmc_e, EPS), "still_suppressed_fraction": (full_e - raoc_e) / max(full_e, EPS), "camera_residual_rms": float(delta.square().mean().sqrt().item()), "B_inf_mean": float(output["b_inf"].mean().item()), "beta_B_mean": float(output["medium_bs"].mean().item()), "beta_D_mean": float(output["medium_attn"].mean().item()), "gaussian_count": int(model.num_points), "medium_branch_grad_l2": float(grad["medium_branch"]["grad_l2"]), "finite": finite, "nan_inf_count": 0})
        model.zero_grad(set_to_none=True)
    model.eval(); branch.pipeline.eval()
    parameter_delta = max((param.detach().cpu() - before[name]).abs().max().item() for name, param in model.named_parameters() if param.numel())
    gate_cat = torch.cat(gate_values, dim=0)
    distribution = {"g_local": _stats(torch.cat(local_values, dim=0), ""), "g_keep_minus_g_obs": _stats(gate_cat - state["global_gate"].reshape(1, 9), ""), "g_keep": _stats(gate_cat, "")}
    _write_csv(output_dir / "smoke_gate_distribution.csv", [{"quantity": "g_keep_minus_g_obs", **_stats(gate_cat - state["global_gate"].reshape(1, 9))}, {"quantity": "g_keep", **_stats(gate_cat)}])
    _write_json(output_dir / "smoke_gate_distribution.json", distribution)
    payload = {"smoke_steps": int(steps), "rows": rows, "all_finite": all(row["finite"] for row in rows), "optimizer_step_count": int(steps), "parameter_delta_max": float(parameter_delta), "gate_distribution": distribution, "refreshed_during_smoke": False}
    _write_json(output_dir / "smoke_summary.json", payload)
    CAM._release(branch)
    return payload


def _decomposition(smoke: Mapping[str, Any], readonly: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    rows = smoke["rows"]
    payload = {**dict(readonly.get("decomposition", {})), "B_inf_mean": _mean(row["B_inf_mean"] for row in rows), "beta_D_mean": _mean(row["beta_D_mean"] for row in rows), "tau_D_available": True, "smoke_outputs_finite": bool(smoke["all_finite"])}
    _write_json(output_dir / "decomposition_safety.json", payload)
    return payload


def _write_note(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eq = summary["equivalence"]
    cap = summary["capacity"]
    lines = [
        "# RAOC RAY-ADAPTIVE OBSERVABILITY PREFLIGHT",
        "",
        "## CODE FACT",
        "RAOC is a default-off, detached capacity-control path over the existing 9-D standardized raw medium residual.",
        "It retains the existing camera context, medium MLP, activations, CUDA rasterizer, and tied B_inf semantics.",
        "",
        "## MATHEMATICAL DEFINITION",
        "`Delta_z_cam = z_full - z_base`, `a_p = V^T Delta_z_cam_std`, `s_i,p = ||J_p v_i||_2`, and `e_i,p = |a_i,p| s_i,p`.",
        "`q_i` is the train-only median evidence, with the registered mean fallback and inactive zero rule.",
        "`g_local=e^2/(e^2+q^2)` and `g_keep=1-(1-g_obs)(1-g_local)`.",
        "The first prototype uses no LCS cosine/alignment term.",
        "",
        "## RESULT",
        "Disabled-path pass: `%s`; OCMC reduction pass: `%s`; identity limit pass: `%s`." % (eq["disabled"]["pass"], eq["ocmc_reduction"]["pass"], eq["identity"]["pass"]),
        "Global rescued-energy fraction: `%s`; bottom-20 rescue fraction: `%s`; top-20 rescue fraction: `%s`." % (cap["global"]["rescue_fraction"], cap["low20"]["mean_rescue"], cap["high20"]["mean_rescue"]),
        "The read-only train/eval comparisons are mechanism safety context only; they are not causal RAOC performance results.",
        "",
        "## CLASSIFICATION",
        "Primary classification: `%s`." % summary["primary_classification"],
        "Ray-adaptive behavior classification: `%s`." % summary["behavior_classification"],
        "Next formal experiment: `M1-RAOC-CAUSAL-IUI3` with matched OCMC and RAOC arms.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def run(repo: Path, output_dir: Path, smoke_steps: int) -> Dict[str, Any]:
    runtime = _runtime(repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "repo_state.json", _repo_state(repo, runtime))
    samples, bank_meta, _flat_rows = _bank(repo)
    branch = _load_branch(repo)
    model = branch.pipeline.model
    try:
        state, calibration, mode_rows, _cache = _calibrate_from_branch(branch, samples, output_dir)
        model.set_camera_medium_ray_adaptive_observability_state(**state)
        eq = _equivalence_checks(branch, state, output_dir)
        gates = _gate_audits(state, output_dir)
        capacity_rows, evidence_data = _collect_controls(branch, state, samples, output_dir)
        capacity = _capacity_rows(capacity_rows, state, output_dir, samples, branch)
        depth_tau = _depth_tau(branch, capacity["energy_rows"], samples, state, output_dir)
        readonly = _readonly_counterfactuals(branch, state, samples, output_dir)
        gradients = _gradients(branch, state, output_dir)
        state_compat = _state_compatibility(repo, branch, state, output_dir)
        repro = _calibration_reproducibility(state, evidence_data["evidence"], output_dir)
        cost = _cost(branch, state, output_dir)
    finally:
        CAM._release(branch)
    smoke = _smoke(repo, state, output_dir, smoke_steps)
    decomposition = _decomposition(smoke, readonly, output_dir)
    high_more = float(capacity["high20"]["mean_rescue"]) > float(capacity["low20"]["mean_rescue"])
    low_conservative = float(capacity["low20"]["mean_rescue"]) < 0.5
    not_full = float(capacity["global"]["rescue_fraction"]) < 0.95
    not_ocmc = float(capacity["global"]["rescue_fraction"]) > 1e-6
    behavior = "RAY_ADAPTIVE_CAPACITY_ALLOCATION_BEHAVIOR_SUPPORTED" if high_more and low_conservative else "RAY_ADAPTIVE_CAPACITY_ALLOCATION_BEHAVIOR_TENTATIVE"
    ready_checks = {
        "disabled": eq["disabled"]["pass"], "ocmc_reduction": eq["ocmc_reduction"]["pass"], "identity": eq["identity"]["pass"],
        "gate_bounds": gates["bounds"]["g_keep_ge_g_obs"], "gate_monotonicity": gates["monotonic"], "all_modes": len(mode_rows) == 9,
        "gradient": (
            gradients["gate_tensors_detached"]
            and gradients["parameter_delta_max_without_optimizer_step"] == 0.0
            and gradients["direct_raoc_gradient_stats"]["medium_mlp"]["has_grad"]
            and gradients["direct_raoc_gradient_stats"]["medium_mlp"]["grad_l2"] > 0.0
            and gradients["direct_mechanism_gaussian_grad_l2_sum"] == 0.0
        ),
        "state": state_compat["roundtrip_pass"] and state_compat["old_pre_raoc_checkpoint_load_pass"], "reproducible": repro["pass"],
        "smoke": smoke["all_finite"], "decomposition": decomposition["J_finite"] and decomposition["smoke_outputs_finite"],
        "high_support_rescue": high_more, "low_support_suppression": low_conservative, "not_full": not_full, "not_ocmc": not_ocmc,
    }
    primary = "RAOC_MODULE_READY" if all(ready_checks.values()) else "RAOC_MODULE_PARTIAL"
    summary = {"experiment": EXPERIMENT, "scene": SCENE, "starting_branch": _git(repo, "branch", "--show-current"), "starting_head": _git(repo, "rev-parse", "HEAD"), "bank": bank_meta, "calibration": calibration, "equivalence": eq, "gate_audit": gates, "capacity": capacity, "depth_tau": depth_tau, "readonly": readonly, "gradient": gradients, "state_compatibility": state_compat, "calibration_reproducibility": repro, "runtime_cost": cost, "smoke": smoke, "decomposition": decomposition, "ready_checks": ready_checks, "primary_classification": primary, "behavior_classification": behavior, "no_gt_or_utility_in_forward": True, "mode_01_hard_coded_in_raoc": False, "all_nine_modes": True, "next_formal_experiment": "M1-RAOC-CAUSAL-IUI3" if primary == "RAOC_MODULE_READY" and behavior.endswith("SUPPORTED") else "Do not proceed to formal causal training; resolve the listed preflight failure(s)."}
    _write_json(output_dir / "final_classification.json", {"primary_classification": primary, "behavior_classification": behavior, "ready_checks": ready_checks})
    _write_json(output_dir / "final_summary.json", summary)
    _write_note(summary, repo / NOTE_PATH)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--smoke-steps", type=int, default=SMOKE_STEPS)
    args = parser.parse_args()
    summary = run(args.repo.resolve(), (args.output_dir if args.output_dir.is_absolute() else args.repo / args.output_dir).resolve(), min(int(args.smoke_steps), 200))
    print(json.dumps({"primary_classification": summary["primary_classification"], "behavior_classification": summary["behavior_classification"], "smoke_steps": summary["smoke"]["smoke_steps"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
