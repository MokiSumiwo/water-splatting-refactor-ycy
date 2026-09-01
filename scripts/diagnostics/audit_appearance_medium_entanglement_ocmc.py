#!/usr/bin/env python3
"""Frozen appearance-medium entanglement audit for registered OCMC checkpoints.

Training-view SH/medium metrics and Gaussian sampling are frozen before heldout
ground truth is accessed. The script performs detached forward rendering only.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.stats
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.diagnostics import audit_gaussian_view_consistency_ocmc as VC
from scripts.diagnostics import audit_low_support_causal_intervention as CAUSAL
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL


EXPERIMENT = "APPEARANCE_MEDIUM_ENTANGLEMENT_AUDIT_OCMC"
EXPECTED_BRANCH = "research/m1-bounded-intrinsic"
EXPECTED_HEAD = "36f965f52d34a5993dfeb7c007f8339171445ead"
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "appearance_medium_entanglement_audit_20260901"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "APPEARANCE_MEDIUM_ENTANGLEMENT_AUDIT_2026-09-01.md"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENE_GPUS = dict(CAUSAL.SCENE_GPUS)
SCENES = tuple(SCENE_GPUS)
STEPS = (5000, 8000, 10000, 13000, 14999)
FINAL_STEP = 14999
SEED = 42
TEMPORAL_SAMPLE_COUNT = 2048
FINAL_SAMPLE_COUNT = 4096
MIN_VALID_SAMPLE = 512
EPS = 1e-12

REQUIRED_CONTROLS = (
    "train_depth_mean",
    "train_tau_mean",
    "train_transmission_mean",
    "opacity",
    "train_footprint_mean",
    "train_ocmc_active_magnitude_mean",
)
DEPTH_MEDIUM_CONTROLS = (
    "train_depth_mean",
    "train_tau_mean",
    "train_transmission_mean",
)
OCMC_INDEPENDENCE_CONTROLS = (
    "train_ocmc_active_magnitude_mean",
    "train_medium_suppressed_residual_mean",
)
ALL_CONTROLS = REQUIRED_CONTROLS + ("train_medium_suppressed_residual_mean",)
PREDICTORS = (
    "VC_SH",
    "VC_medium",
    "compensation_score",
    "heldout_SH_alignment",
    "heldout_medium_alignment",
    "heldout_interaction_alignment",
    "heldout_joint_alignment",
)

PROTECTED_HASHES = {
    "scripts/diagnostics/analyze_raoc_q50_q80_causal_four_scene.py": "b6a271372e68cd07fc566a3fde5ced5ba6463531278c31a6cfa47972aa15e8d6",
    "scripts/diagnostics/render_gmvc_curasao_contact_sheet.py": "539f1c044f9ed136dce65b1dedc01746097cb2f3c4298c9682038019d23dfd7a",
    "scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py": "fe3fd3ddcdbbff7904cfb7225a0ba024f928a9020777561252b66663c3c8ab32",
    "scripts/experiments/run_raoc_q50_q80_causal_four_scene.py": "d131428cc20ea76010e237abd91ac4cddfc5c6a78944c57c3317ed18bcdf60ef",
    "scripts/experiments/run_raoc_q50_q80_causal_scene.py": "3a924e88a606d34360a90348f3a392d0d12f80d43c98fe72b56cbec2d27ad6e7",
    "scripts/diagnostics/audit_gaussian_view_consistency_ocmc.py": "4040207673fcb83e43d49effa581cc4601f6b0fd671d7ec6876690805f82952e",
    "scripts/diagnostics/audit_low_support_causal_intervention.py": "92a45a7d17621b6f44b882e919ea2d65f9916669a0e94d75c8c72d03249d0ee3",
    "scripts/diagnostics/audit_local_contextual_support_predictor_iui3.py": "2f88afc2174f5753ee6cee494041b1f793529a4ea13742c425ad2928023a3479",
    "scripts/experiments/run_m1_raoc_causal_scene.py": "79930754f41887c0530e6b033eef5f0f26b692795a4f3abd078358ad800f9f2a",
    "water_splatting/water_splatting.py": "1a9930c0e74b4f235fc5ae5e819823fe9e2cdd828e8764ca73e43d0f67aa63e1",
    "water_splatting/fields/medium_field.py": "43a610d67921c00b171b9285e0fe3138f0e8eff6d84edf3d9b1f79e373bbfdef",
    "water_splatting/rendering/underwater_rasterizer.py": "04e6d1c6d136ee46ea32ea2abd666e688d6a35c00c787e31f17aa5f5ba17beba",
    "water_splatting/raoc.py": "e2ffe7f0e457ef1ef67b478b0638ea3afc6a585557886fa95a933d65f6b0ba08",
    "water_splatting/cuda/csrc/raoc.cu": "5599222dedf658885889d86af6b24a5ba2f6e6760818f0889b392dccd0a6d24d",
    "water_splatting/utils.py": "cb5ae8538bdf9bd6a36f15b6a819a63e750c1d6cc306574e11a85511fa4295ea",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Tensor):
        cpu = value.detach().cpu()
        return cpu.item() if cpu.numel() == 1 else cpu.tolist()
    return str(value)


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _sanitize(value.item())
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize(value), indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = {}
            for key in fields:
                value = source.get(key, "")
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                row[key] = value
            writer.writerow(row)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: Sequence[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def _checkpoint(scene: str, step: int) -> Path:
    return SOURCE_ROOT / scene / "checkpoints" / "C0" / f"step-{step:09d}.ckpt"


def _strict_repo() -> Dict[str, Any]:
    branch = _run_text(["git", "branch", "--show-current"])
    head = _run_text(["git", "rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected repository state: {branch}@{head}")
    hashes = {}
    for relative, expected in PROTECTED_HASHES.items():
        actual = _sha256(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"protected source changed: {relative}")
        hashes[relative] = actual
    return {
        "branch": branch,
        "starting_head": head,
        "status_short": _run_text(["git", "status", "--short"]),
        "protected_hashes": hashes,
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
    }


def preflight() -> Dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    repo = _strict_repo()
    rows = []
    for scene in SCENES:
        scene_config = CAUSAL._scene_config(scene)
        config = REPO_ROOT / scene_config["source_config"]
        sequence = SOURCE_ROOT / scene / "camera_sequence.json"
        config_text = config.read_text(encoding="utf8")
        serialized_conditions = (
            "intrinsic_color_parameterization: sigmoid_sh",
            "medium_context_mode: dir_xy_camera",
            "b_inf_mode: tied",
            "infinite_water_enabled: false",
            "rasterize_mode: classic",
        )
        if not all(condition in config_text for condition in serialized_conditions):
            raise RuntimeError(f"serialized source configuration drift for {scene}")
        if _sha256(config) != VC.EXPECTED_CONFIG_HASHES[scene]:
            raise RuntimeError(f"source config provenance drift for {scene}")
        if _sha256(sequence) != VC.EXPECTED_CAMERA_SEQUENCE_HASHES[scene]:
            raise RuntimeError(f"camera sequence provenance drift for {scene}")
        for step in STEPS:
            checkpoint = _checkpoint(scene, step)
            actual = _sha256(checkpoint)
            if actual != VC.EXPECTED_CHECKPOINT_HASHES[scene][step]:
                raise RuntimeError(f"checkpoint provenance drift: {checkpoint}")
            rows.append(
                {
                    "scene": scene,
                    "absolute_step": step,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": actual,
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                    "source_config": str(config),
                    "source_config_sha256": _sha256(config),
                    "source_config_serialized_intrinsic_name": "sigmoid_sh",
                    "runtime_canonical_intrinsic_name": "bounded_sh3",
                    "camera_sequence": str(sequence),
                    "camera_sequence_sha256": _sha256(sequence),
                }
            )
    result = {
        "experiment": EXPERIMENT,
        "repo": repo,
        "launcher_environment": {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "worker_physical_gpus": SCENE_GPUS,
            "worker_logical_device": "cuda:0",
        },
        "checkpoint_rows": rows,
        "metric": {
            "weight": "w_i(v)=sigmoid(opacity_i)*pi*projected_radius_i(v)^2/image_area",
            "C_SH": "w_i(v)*(gaussian_view_rgb_i(v)-gaussian_view_dc_rgb_i)",
            "C_medium_direct": "w_i(v)*(exp(-beta_D(v)*depth_i(v))-1)*gaussian_view_dc_rgb_i",
            "C_medium_scatter": "w_i(v)*rgb_medium(v) sampled at the projected Gaussian center",
            "C_medium": "C_medium_direct+C_medium_scatter",
            "C_interaction": "w_i(v)*(exp(-beta_D(v)*depth_i(v))-1)*(gaussian_view_rgb_i(v)-gaussian_view_dc_rgb_i)",
            "VC_SH": "population mean squared RGB L2 deviation of C_SH over visible training cameras",
            "VC_medium": "population mean squared RGB L2 deviation of C_medium over visible training cameras",
            "SH_medium_corr": "centered vector correlation sum_v dot(delta C_SH,delta C_medium)/sqrt(sum_v||delta C_SH||^2 sum_v||delta C_medium||^2)",
            "compensation_score": "-SH_medium_corr; positive means opposing paired training-view variation",
            "joint_residual": "C_SH+C_medium+C_interaction, relative to the clear DC appearance proxy",
            "heldout_alignment": "mean heldout deviation from the frozen training-view component mean",
            "heldout_error": "mean heldout RGB MSE in each Gaussian's clipped projected-radius footprint box",
            "sampling": "deterministic support-stratified sample using training visibility only; support>=2",
            "gt_boundary": "sampling and all training metrics are frozen before the first heldout GT access",
        },
        "classification_rule": {
            "checkpoint_pass": "median SH_medium_corr<0, rho(compensation_score,error)>0, joint depth/tau/transmission partial rank rho>0, all-required-control partial rank rho>0, and OCMC-control partial rank rho>0",
            "scene_pass": "final checkpoint passes and checkpoint_pass recurs at >=3/5 checkpoints",
            "SUPPORTED": ">=3/4 scene passes with quality gate passing",
            "TENTATIVE": "exactly 2/4 scene passes with quality gate passing",
            "NOT_SUPPORTED": "0-1/4 scene passes with quality gate passing",
            "module_design_authorized": "classification is SUPPORTED only",
        },
        "lineage": {
            "available": False,
            "temporal_scope": "checkpoint-population distributions only",
            "identity_matching_forbidden": True,
        },
        "frozen_forward_only": True,
        "ocmc_enabled": True,
        "raoc_enabled": False,
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
    }
    _write_json(OUTPUT_ROOT / "preflight.json", result)
    _write_json(OUTPUT_ROOT / "checkpoint_manifest.json", {"rows": rows})
    return result


def _runtime(scene: str, gpu: str) -> Dict[str, Any]:
    if scene not in SCENES or SCENE_GPUS[scene] != gpu:
        raise RuntimeError(f"invalid scene/GPU assignment: {scene}/{gpu}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError(f"worker must expose physical GPU {gpu} only")
    if os.environ.get("CONDA_DEFAULT_ENV") != "water_splatting":
        raise RuntimeError("CONDA_DEFAULT_ENV must be water_splatting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("worker must see exactly logical cuda:0")
    props = torch.cuda.get_device_properties(0)
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "physical_gpu": gpu,
        "logical_gpu": 0,
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
    }


def _camera_split(
    train_records: Sequence[Tuple[int, str, Any, Any]],
    eval_records: Sequence[Tuple[int, str, Any, Any]],
) -> Dict[str, Any]:
    train_ids = [str(row[1]) for row in train_records]
    eval_ids = [str(row[1]) for row in eval_records]
    if len(train_ids) != len(set(train_ids)) or len(eval_ids) != len(set(eval_ids)):
        raise RuntimeError("duplicate camera ID within a split")
    if set(train_ids) & set(eval_ids):
        raise RuntimeError("train/eval camera leakage")
    payload = json.dumps({"train": train_ids, "eval": eval_ids}, sort_keys=True).encode("utf8")
    return {
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "train_count": len(train_ids),
        "eval_count": len(eval_ids),
        "camera_split_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _sample_image(values: Tensor, xys: Tensor) -> Tensor:
    height, width = int(values.shape[0]), int(values.shape[1])
    x = xys[:, 0].round().long().clamp(0, width - 1)
    y = xys[:, 1].round().long().clamp(0, height - 1)
    return values[y, x]


@torch.no_grad()
def _components(
    model: Any,
    outputs: Mapping[str, Tensor],
    global_ids: Tensor,
    opacity: Tensor,
) -> Dict[str, Tensor]:
    xys = model.xys.detach()[global_ids]
    radii = model.radii.detach().reshape(-1)[global_ids].float()
    depth = outputs["projected_gaussian_depths"].detach().reshape(-1)[global_ids].float()
    height, width = int(outputs["pred_image"].shape[0]), int(outputs["pred_image"].shape[1])
    weight = opacity * math.pi * radii.square() / float(height * width)
    rgb = outputs["gaussian_view_rgb"].detach().float()[global_ids]
    dc = outputs["gaussian_view_dc_rgb"].detach().float()[global_ids]
    sh_residual = outputs["gaussian_color_residual"].detach().float()[global_ids]
    if not torch.allclose(rgb - dc, sh_residual, rtol=2e-5, atol=2e-6):
        raise RuntimeError("bounded SH RGB residual alias mismatch")
    medium_attn = _sample_image(outputs["medium_attn"].detach().float(), xys)
    transmission_rgb = torch.exp(-(medium_attn * depth[:, None]).clamp_min(0.0)).clamp(0.0, 1.0)
    medium_scatter_rgb = _sample_image(outputs["rgb_medium"].detach().float(), xys)
    c_sh = weight[:, None] * sh_residual
    c_medium_direct = weight[:, None] * (transmission_rgb - 1.0) * dc
    c_medium_scatter = weight[:, None] * medium_scatter_rgb
    c_medium = c_medium_direct + c_medium_scatter
    c_interaction = weight[:, None] * (transmission_rgb - 1.0) * sh_residual
    c_joint = c_sh + c_medium + c_interaction
    return {
        "sh": c_sh,
        "medium": c_medium,
        "medium_direct": c_medium_direct,
        "medium_scatter": c_medium_scatter,
        "interaction": c_interaction,
        "joint": c_joint,
        "depth": depth,
        "tau": (medium_attn * depth[:, None]).mean(dim=-1),
        "transmission": transmission_rgb.mean(dim=-1),
        "footprint": radii,
        "ocmc_active_magnitude": torch.linalg.vector_norm(
            _sample_image(outputs["camera_medium_delta_projected_raw"].detach().float(), xys), dim=-1
        ),
        "medium_suppressed_residual": torch.linalg.vector_norm(
            _sample_image(outputs["camera_medium_delta_suppressed_raw"].detach().float(), xys), dim=-1
        ),
    }


def _consistency(values: Tensor, observed: Tensor) -> Dict[str, Tensor]:
    valid = torch.isfinite(values[..., 0])
    safe = torch.where(valid[..., None], values.double(), torch.zeros_like(values, dtype=torch.float64))
    mean = safe.sum(dim=0) / observed.double().clamp_min(1)[:, None]
    delta = torch.where(valid[..., None], safe - mean[None, ...], torch.zeros_like(safe))
    variance = delta.square().sum(dim=-1).sum(dim=0) / observed.double().clamp_min(1)
    deviation = torch.linalg.vector_norm(delta, dim=-1).sum(dim=0) / observed.double().clamp_min(1)
    return {"mean": mean, "delta": delta, "variance": variance, "mean_deviation": deviation}


@torch.no_grad()
def _training_statistics(
    model: Any,
    records: Sequence[Tuple[int, str, Any, Any]],
    selected: Tensor,
    support: Tensor,
) -> Dict[str, Tensor]:
    count = int(selected.numel())
    selected_gpu = selected.to(model.device)
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)[selected_gpu].float()
    observed = torch.zeros(count, dtype=torch.int16)
    names = ("sh", "medium", "medium_direct", "medium_scatter", "interaction", "joint")
    values = {
        name: torch.full((len(records), count, 3), float("nan"), dtype=torch.float32)
        for name in names
    }
    sum_names = (
        "depth",
        "tau",
        "transmission",
        "footprint",
        "ocmc_active_magnitude",
        "medium_suppressed_residual",
    )
    sums = {name: torch.zeros(count, dtype=torch.float64) for name in sum_names}
    for view_index, (_index, _camera_id, camera, _batch) in enumerate(records):
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1)[selected_gpu].bool()
        visible_cpu = visible.cpu()
        if bool(visible.any()):
            local = torch.nonzero(visible, as_tuple=False).reshape(-1)
            global_ids = selected_gpu[local]
            components = _components(model, outputs, global_ids, opacity[local])
            for name in names:
                values[name][view_index, visible_cpu] = components[name].cpu()
            for name in sum_names:
                sums[name][visible_cpu] += components[name].double().cpu()
            observed += visible_cpu.to(torch.int16)
        del outputs
    if not torch.equal(observed, support[selected]):
        raise RuntimeError("training metric visibility differs from frozen support")
    stats = {name: _consistency(values[name], observed) for name in names}
    sh_delta = stats["sh"]["delta"]
    medium_delta = stats["medium"]["delta"]
    numerator = (sh_delta * medium_delta).sum(dim=(0, 2))
    denominator = torch.sqrt(
        sh_delta.square().sum(dim=(0, 2)) * medium_delta.square().sum(dim=(0, 2))
    )
    correlation = torch.full((count,), float("nan"), dtype=torch.float64)
    finite_corr = denominator > EPS
    correlation[finite_corr] = numerator[finite_corr] / denominator[finite_corr]
    paired_dot = (sh_delta * medium_delta).sum(dim=-1)
    paired_nonzero = (
        torch.linalg.vector_norm(sh_delta, dim=-1) > EPS
    ) & (torch.linalg.vector_norm(medium_delta, dim=-1) > EPS)
    opposite = ((paired_dot < 0.0) & paired_nonzero).sum(dim=0).double() / paired_nonzero.sum(dim=0).clamp_min(1)
    result: Dict[str, Tensor] = {
        "observed_train_views": observed,
        "VC_SH": stats["sh"]["variance"],
        "VC_medium": stats["medium"]["variance"],
        "VC_medium_direct": stats["medium_direct"]["variance"],
        "VC_medium_scatter": stats["medium_scatter"]["variance"],
        "VC_interaction": stats["interaction"]["variance"],
        "VC_joint": stats["joint"]["variance"],
        "SH_mean_deviation": stats["sh"]["mean_deviation"],
        "medium_mean_deviation": stats["medium"]["mean_deviation"],
        "SH_medium_corr": correlation,
        "compensation_score": -correlation,
        "opposite_variation_fraction": opposite,
        "train_sh_mean_rgb": stats["sh"]["mean"],
        "train_medium_mean_rgb": stats["medium"]["mean"],
        "train_interaction_mean_rgb": stats["interaction"]["mean"],
        "train_joint_mean_rgb": stats["joint"]["mean"],
    }
    for name, total in sums.items():
        result[f"train_{name}_mean"] = total / observed.double().clamp_min(1)
    return result


@torch.no_grad()
def _heldout_statistics(
    scene: str,
    step: int,
    model: Any,
    records: Sequence[Tuple[int, str, Any, Any]],
    selected: Tensor,
    training: Mapping[str, Tensor],
) -> Tuple[Dict[str, Tensor], List[Dict[str, Any]]]:
    count = int(selected.numel())
    selected_gpu = selected.to(model.device)
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)[selected_gpu].float()
    names = ("SH", "medium", "interaction", "joint")
    sums = {name: torch.zeros(count, dtype=torch.float64) for name in names}
    residual_sum = torch.zeros(count, dtype=torch.float64)
    high_fraction_sum = torch.zeros(count, dtype=torch.float64)
    observed = torch.zeros(count, dtype=torch.int16)
    camera_rows: List[Dict[str, Any]] = []
    training_means = {
        "SH": training["train_sh_mean_rgb"],
        "medium": training["train_medium_mean_rgb"],
        "interaction": training["train_interaction_mean_rgb"],
        "joint": training["train_joint_mean_rgb"],
    }
    component_keys = {"SH": "sh", "medium": "medium", "interaction": "interaction", "joint": "joint"}
    for _index, camera_id, camera, batch in records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1)[selected_gpu].bool()
        local = torch.nonzero(visible, as_tuple=False).reshape(-1)
        global_ids = selected_gpu[local]
        # Sampling and every training-view metric are frozen before this first GT access.
        gt = MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
        pred = outputs["pred_image"].detach().float().clamp(0, 1)
        residual = (pred - gt).square().mean(dim=-1)
        threshold = torch.quantile(residual.reshape(-1), 0.80)
        high = residual >= threshold
        camera_scores = {name: torch.empty(0) for name in names}
        if bool(visible.any()):
            components = _components(model, outputs, global_ids, opacity[local])
            xys = model.xys.detach()[global_ids]
            radii = model.radii.detach().reshape(-1)[global_ids]
            projected_residual, high_fraction, _area = VC._box_statistics(residual, high, xys, radii)
            visible_cpu = visible.cpu()
            residual_sum[visible_cpu] += projected_residual.double().cpu()
            high_fraction_sum[visible_cpu] += high_fraction.double().cpu()
            for name in names:
                mean = training_means[name][visible_cpu].to(model.device, dtype=torch.float32)
                score = torch.linalg.vector_norm(components[component_keys[name]] - mean, dim=-1)
                sums[name][visible_cpu] += score.double().cpu()
                camera_scores[name] = score
            observed += visible_cpu.to(torch.int16)
        camera_mse = float(residual.mean())
        camera_psnr = -10.0 * math.log10(max(camera_mse, EPS))
        camera_rows.append(
            {
                "scene": scene,
                "absolute_step": step,
                "camera_id": camera_id,
                "sampled_visible_gaussian_count": int(visible.sum()),
                "camera_MSE": camera_mse,
                "camera_PSNR": camera_psnr,
                "camera_PSNR_error": -camera_psnr,
                "camera_high_error_threshold_MSE": float(threshold),
                "camera_VC_SH_mean": float(training["VC_SH"][visible.cpu()].mean()) if bool(visible.any()) else float("nan"),
                "camera_VC_medium_mean": float(training["VC_medium"][visible.cpu()].mean()) if bool(visible.any()) else float("nan"),
                "camera_compensation_score_mean": float(training["compensation_score"][visible.cpu()].nanmean()) if bool(visible.any()) else float("nan"),
                "camera_SH_alignment_mean": float(camera_scores["SH"].mean()) if camera_scores["SH"].numel() else float("nan"),
                "camera_medium_alignment_mean": float(camera_scores["medium"].mean()) if camera_scores["medium"].numel() else float("nan"),
                "camera_interaction_alignment_mean": float(camera_scores["interaction"].mean()) if camera_scores["interaction"].numel() else float("nan"),
                "camera_joint_alignment_mean": float(camera_scores["joint"].mean()) if camera_scores["joint"].numel() else float("nan"),
                "metric_uses_heldout_or_gt": False,
                "gt_used_for_error_outcome_only": True,
            }
        )
        del outputs, gt, pred, residual, high
    seen = observed > 0

    def mean_or_nan(total: Tensor) -> Tensor:
        output = torch.full_like(total, float("nan"), dtype=torch.float64)
        output[seen] = total[seen] / observed[seen].double()
        return output

    result = {
        "heldout_visible_views": observed,
        "heldout_projected_residual": mean_or_nan(residual_sum),
        "heldout_high_error_pixel_fraction": mean_or_nan(high_fraction_sum),
    }
    for name in names:
        result[f"heldout_{name}_alignment"] = mean_or_nan(sums[name])
    return result, camera_rows


def _rho(left: Sequence[float], right: Sequence[float], minimum_count: int = 12) -> Tuple[float, float, int]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < minimum_count or np.ptp(a[valid]) <= 0.0 or np.ptp(b[valid]) <= 0.0:
        return float("nan"), float("nan"), int(valid.sum())
    result = scipy.stats.spearmanr(a[valid], b[valid])
    return float(result.statistic), float(result.pvalue), int(valid.sum())


def _partial_rank_multi(
    predictor: Sequence[float],
    target: Sequence[float],
    controls: Sequence[Sequence[float]],
) -> Tuple[float, float, int]:
    arrays = [np.asarray(predictor, dtype=np.float64), np.asarray(target, dtype=np.float64)]
    arrays.extend(np.asarray(control, dtype=np.float64) for control in controls)
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    if int(valid.sum()) < 12:
        return float("nan"), float("nan"), int(valid.sum())
    ranked = [scipy.stats.rankdata(array[valid]) for array in arrays]
    design = np.column_stack([np.ones(int(valid.sum()))] + ranked[2:])
    x_residual = ranked[0] - design @ np.linalg.lstsq(design, ranked[0], rcond=None)[0]
    y_residual = ranked[1] - design @ np.linalg.lstsq(design, ranked[1], rcond=None)[0]
    return _rho(x_residual, y_residual)


def _rank_r2(predictors: Sequence[Sequence[float]], target: Sequence[float]) -> float:
    arrays = [np.asarray(target, dtype=np.float64)] + [np.asarray(item, dtype=np.float64) for item in predictors]
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    if int(valid.sum()) < 12:
        return float("nan")
    y = scipy.stats.rankdata(arrays[0][valid])
    x = np.column_stack([np.ones(int(valid.sum()))] + [scipy.stats.rankdata(item[valid]) for item in arrays[1:]])
    fit = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    total = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - fit) ** 2).sum()) / max(total, EPS)


def _rows_for_checkpoint(
    scene: str,
    step: int,
    selected: Tensor,
    support: Tensor,
    model: Any,
    training: Mapping[str, Tensor],
    heldout: Mapping[str, Tensor],
) -> List[Dict[str, Any]]:
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1).cpu()[selected]
    valid = (
        torch.isfinite(heldout["heldout_projected_residual"])
        & torch.isfinite(training["SH_medium_corr"])
        & (heldout["heldout_visible_views"] > 0)
    )
    valid_local = torch.nonzero(valid, as_tuple=False).reshape(-1)
    high_fraction = heldout["heldout_high_error_pixel_fraction"]
    threshold = float(torch.quantile(high_fraction[valid].float(), 0.80))
    scalar_training = (
        "observed_train_views",
        "VC_SH",
        "VC_medium",
        "VC_medium_direct",
        "VC_medium_scatter",
        "VC_interaction",
        "VC_joint",
        "SH_mean_deviation",
        "medium_mean_deviation",
        "SH_medium_corr",
        "compensation_score",
        "opposite_variation_fraction",
        "train_depth_mean",
        "train_tau_mean",
        "train_transmission_mean",
        "train_footprint_mean",
        "train_ocmc_active_magnitude_mean",
        "train_medium_suppressed_residual_mean",
    )
    rows = []
    for local in valid_local.tolist():
        gaussian_id = int(selected[local])
        error = float(heldout["heldout_projected_residual"][local])
        row: Dict[str, Any] = {
            "gaussian_id": gaussian_id,
            "scene": scene,
            "absolute_step": step,
            "gaussian_identity_persistent_across_checkpoints": False,
            "support_count": int(support[gaussian_id]),
            "opacity": float(opacity[local]),
            "heldout_PSNR_error": 10.0 * math.log10(max(error, EPS)),
            "high_error_label_top20": bool(float(high_fraction[local]) >= threshold),
            "metric_uses_heldout_or_gt": False,
            "gt_used_after_metric_for_error_only": True,
        }
        for key in scalar_training:
            row[key] = float(training[key][local])
        for key, values in heldout.items():
            row[key] = float(values[local])
        rows.append(row)
    return rows


def _checkpoint_summary(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    error = [float(row["heldout_projected_residual"]) for row in rows]
    labels = [bool(row["high_error_label_top20"]) for row in rows]
    correlations: Dict[str, Dict[str, Any]] = {}
    control_rows: List[Dict[str, Any]] = []
    for predictor in PREDICTORS:
        values = [float(row[predictor]) for row in rows]
        rho, pvalue, count = _rho(values, error)
        correlations[predictor] = {
            "rho_vs_heldout_MSE": rho,
            "rho_vs_heldout_PSNR_error": rho,
            "pvalue": pvalue,
            "valid_count": count,
            "auroc_top20_error": VC._auroc(values, labels),
        }
        control_rows.append(
            {
                "scene": rows[0]["scene"],
                "absolute_step": int(rows[0]["absolute_step"]),
                "predictor": predictor,
                "control_set": "raw",
                "controls": "",
                "rho": rho,
                "pvalue": pvalue,
                "valid_count": count,
            }
        )
        sets = [(name, (name,)) for name in ALL_CONTROLS]
        sets.extend(
            [
                ("depth_tau_transmission_joint", DEPTH_MEDIUM_CONTROLS),
                ("ocmc_independence_joint", OCMC_INDEPENDENCE_CONTROLS),
                ("all_required_joint", ALL_CONTROLS),
            ]
        )
        for set_name, names in sets:
            controlled, controlled_p, controlled_count = _partial_rank_multi(
                values,
                error,
                [[float(row[name]) for row in rows] for name in names],
            )
            control_rows.append(
                {
                    "scene": rows[0]["scene"],
                    "absolute_step": int(rows[0]["absolute_step"]),
                    "predictor": predictor,
                    "control_set": set_name,
                    "controls": ";".join(names),
                    "rho": controlled,
                    "pvalue": controlled_p,
                    "valid_count": controlled_count,
                }
            )
    comp_controls = {
        row["control_set"]: float(row["rho"])
        for row in control_rows
        if row["predictor"] == "compensation_score"
    }
    corr_values = np.asarray([float(row["SH_medium_corr"]) for row in rows], dtype=np.float64)
    comp_rho = correlations["compensation_score"]["rho_vs_heldout_MSE"]
    checkpoint_pass = bool(
        float(np.median(corr_values)) < 0.0
        and math.isfinite(comp_rho)
        and comp_rho > 0.0
        and comp_controls["depth_tau_transmission_joint"] > 0.0
        and comp_controls["ocmc_independence_joint"] > 0.0
        and comp_controls["all_required_joint"] > 0.0
    )
    alignment_names = (
        "heldout_SH_alignment",
        "heldout_medium_alignment",
        "heldout_interaction_alignment",
        "heldout_joint_alignment",
    )
    rank_r2 = {
        name: _rank_r2([[float(row[name]) for row in rows]], error)
        for name in alignment_names
    }
    rank_r2["SH_medium_interaction_multivariate"] = _rank_r2(
        [[float(row[name]) for row in rows] for name in alignment_names[:3]], error
    )
    winner = max(alignment_names, key=lambda name: correlations[name]["rho_vs_heldout_MSE"])
    summary = {
        "scene": rows[0]["scene"],
        "absolute_step": int(rows[0]["absolute_step"]),
        "gaussian_count": len(rows),
        "median_SH_medium_corr": float(np.median(corr_values)),
        "negative_SH_medium_corr_fraction": float((corr_values < 0.0).mean()),
        "median_compensation_score": float(np.median(-corr_values)),
        "median_VC_SH": float(np.median([float(row["VC_SH"]) for row in rows])),
        "median_VC_medium": float(np.median([float(row["VC_medium"]) for row in rows])),
        "VC_SH_positive_fraction": float(np.mean([float(row["VC_SH"]) > 0.0 for row in rows])),
        "VC_medium_positive_fraction": float(np.mean([float(row["VC_medium"]) > 0.0 for row in rows])),
        "correlations": correlations,
        "compensation_control_rhos": comp_controls,
        "residual_decomposition_rank_R2": rank_r2,
        "residual_decomposition_best_single_predictor": winner,
        "checkpoint_entanglement_pass": checkpoint_pass,
    }
    return summary, control_rows


@torch.no_grad()
def worker(
    scene: str,
    gpu: str,
    steps: Sequence[int] = STEPS,
    sample_count: Optional[int] = None,
) -> Dict[str, Any]:
    runtime = _runtime(scene, gpu)
    started = time.perf_counter()
    scene_dir = OUTPUT_ROOT / "workers" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    branch = FORMAL._setup_branch(REPO_ROOT, CAUSAL._scene_config(scene), "C0")
    try:
        model = branch.pipeline.model
        train_records = FORMAL._train_records(branch.pipeline)
        eval_records = FORMAL._eval_records(branch.pipeline)
        split = _camera_split(train_records, eval_records)
        all_rows: List[Dict[str, Any]] = []
        all_camera_rows: List[Dict[str, Any]] = []
        all_control_rows: List[Dict[str, Any]] = []
        checkpoint_rows: List[Dict[str, Any]] = []
        for step in steps:
            checkpoint = _checkpoint(scene, step)
            payload = FORMAL._load_checkpoint(branch, checkpoint)
            if (
                payload.get("experiment") != FORMAL.EXPERIMENT
                or payload.get("branch") != "C0"
                or int(payload.get("absolute_step", -1)) != step
                or payload.get("ocmc_bundle") is None
                or payload.get("raoc_state") is not None
            ):
                raise RuntimeError(f"checkpoint condition provenance drift: {checkpoint}")
            if (
                not model.config.camera_medium_observability_enabled
                or model.config.camera_medium_ray_adaptive_observability_enabled
                or model.config.intrinsic_color_parameterization != "bounded_sh3"
                or model.config.rasterize_mode != "classic"
                or model.config.medium_context_mode != "dir_xy_camera"
                or int(model.config.sh_degree) != 3
                or model.config.b_inf_mode != "tied"
                or model.config.infinite_water_enabled
            ):
                raise RuntimeError("locked OCMC-on RAOC-off model configuration drift")
            state_before = CAUSAL._model_state_hash(model)
            projector_before = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
            support = VC._support_counts(model, train_records)
            requested = sample_count if sample_count is not None else (
                FINAL_SAMPLE_COUNT if step == FINAL_STEP else TEMPORAL_SAMPLE_COUNT
            )
            selected, sampling = VC._sample_gaussians(scene, step, support, requested)
            training = _training_statistics(model, train_records, selected, support)
            # This is the heldout/GT boundary; no GT-dependent selection is possible.
            heldout, camera_rows = _heldout_statistics(scene, step, model, eval_records, selected, training)
            rows = _rows_for_checkpoint(scene, step, selected, support, model, training, heldout)
            summary, control_rows = _checkpoint_summary(rows)
            state_after = CAUSAL._model_state_hash(model)
            projector_after = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
            if state_before != state_after or projector_before != projector_after:
                raise RuntimeError("frozen model or OCMC projector changed")
            sampling["heldout_visible_finite_compensation_count"] = len(rows)
            checkpoint_rows.append(
                {
                    **summary,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "sampling": sampling,
                    "ocmc_enable_flag": True,
                    "raoc_enable_flag": False,
                    "raoc_state_present": False,
                    "runtime_intrinsic_color_parameterization": str(
                        model.config.intrinsic_color_parameterization
                    ),
                    "source_config_serialized_intrinsic_name": "sigmoid_sh",
                    "runtime_rasterize_mode": str(model.config.rasterize_mode),
                    "runtime_medium_context_mode": str(model.config.medium_context_mode),
                    "runtime_b_inf_mode": str(model.config.b_inf_mode),
                    "runtime_infinite_water_enabled": bool(model.config.infinite_water_enabled),
                    "camera_medium_raoc_backend_config": str(
                        getattr(model.config, "camera_medium_raoc_backend", "not_configured")
                    ),
                    "camera_medium_raoc_effective_status": "disabled_by_enable_flag_and_absent_state",
                    "model_state_sha256_before": state_before,
                    "model_state_sha256_after": state_after,
                    "ocmc_projector_sha256_before": projector_before,
                    "ocmc_projector_sha256_after": projector_after,
                    "model_and_ocmc_unchanged": True,
                }
            )
            all_rows.extend(rows)
            all_camera_rows.extend(camera_rows)
            all_control_rows.extend(control_rows)
            print(
                f"[{scene}] step {step}: n={len(rows)} median_corr={summary['median_SH_medium_corr']:.6f} "
                f"rho_comp_error={summary['correlations']['compensation_score']['rho_vs_heldout_MSE']:.6f} "
                f"pass={summary['checkpoint_entanglement_pass']}",
                flush=True,
            )
            del payload, support, selected, training, heldout, rows
            gc.collect()
            torch.cuda.empty_cache()
        result = {
            "experiment": EXPERIMENT,
            "scene": scene,
            "runtime": runtime,
            "camera_split": split,
            "checkpoint_rows": checkpoint_rows,
            "gaussian_rows": len(all_rows),
            "camera_rows": len(all_camera_rows),
            "control_rows": len(all_control_rows),
            "steps": list(steps),
            "sample_count_override": sample_count,
            "elapsed_seconds": time.perf_counter() - started,
            "frozen_forward_only": True,
            "backward_calls": 0,
            "optimizer_step_calls": 0,
            "checkpoint_writes": 0,
            "render_writes": 0,
        }
        suffix = "" if tuple(steps) == STEPS and sample_count is None else "_smoke"
        _write_csv(scene_dir / f"per_gaussian_sh_medium_metrics{suffix}.csv", all_rows)
        _write_csv(scene_dir / f"per_camera_error_alignment{suffix}.csv", all_camera_rows)
        _write_csv(scene_dir / f"control_analysis{suffix}.csv", all_control_rows)
        _write_json(scene_dir / f"worker_summary{suffix}.json", result)
        return result
    finally:
        FORMAL._release(branch)


def _coerce_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    text = {"scene", "camera_id", "predictor", "control_set", "controls"}
    boolean = {
        "gaussian_identity_persistent_across_checkpoints",
        "high_error_label_top20",
        "metric_uses_heldout_or_gt",
        "gt_used_after_metric_for_error_only",
        "gt_used_for_error_outcome_only",
    }
    integer = {
        "gaussian_id",
        "absolute_step",
        "support_count",
        "observed_train_views",
        "heldout_visible_views",
        "sampled_visible_gaussian_count",
        "valid_count",
    }
    output = []
    for source in rows:
        row: Dict[str, Any] = {}
        for key, value in source.items():
            if key in text:
                row[key] = value
            elif key in boolean:
                row[key] = value == "True"
            elif key in integer:
                row[key] = int(float(value))
            else:
                row[key] = float(value) if value != "" else float("nan")
        output.append(row)
    return output


def _camera_analysis(camera_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    final = [row for row in camera_rows if int(row["absolute_step"]) == FINAL_STEP]
    metrics = (
        "camera_compensation_score_mean",
        "camera_SH_alignment_mean",
        "camera_medium_alignment_mean",
        "camera_interaction_alignment_mean",
        "camera_joint_alignment_mean",
    )
    scene_rows = []
    pooled: Dict[str, List[float]] = {name: [] for name in metrics}
    pooled_error: List[float] = []
    for scene in SCENES:
        rows = [row for row in final if row["scene"] == scene]
        error = [float(row["camera_MSE"]) for row in rows]
        entry: Dict[str, Any] = {"scene": scene, "heldout_camera_count": len(rows), "small_n_descriptive_only": True}
        for name in metrics:
            values = [float(row[name]) for row in rows]
            entry[f"rho_{name}_vs_camera_MSE"] = _rho(values, error, minimum_count=3)[0]
            pooled[name].extend((scipy.stats.rankdata(values) / max(len(values), 1)).tolist())
        pooled_error.extend((scipy.stats.rankdata(error) / max(len(error), 1)).tolist())
        scene_rows.append(entry)
    return {
        "scene_rows": scene_rows,
        "pooled_camera_count": len(pooled_error),
        "pooled_within_scene_rank_rhos": {
            name: _rho(values, pooled_error)[0] for name, values in pooled.items()
        },
        "pooling_rule": "Spearman correlation over within-scene camera percentile ranks",
    }


def _classify(checkpoint_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    scene_rows = []
    temporal_rows = []
    quality_rows = []
    for scene in SCENES:
        trajectory = sorted(
            [row for row in checkpoint_rows if row["scene"] == scene],
            key=lambda row: int(row["absolute_step"]),
        )
        final = next(row for row in trajectory if int(row["absolute_step"]) == FINAL_STEP)
        pass_count = sum(bool(row["checkpoint_entanglement_pass"]) for row in trajectory)
        temporal_stable = pass_count >= 3
        quality = bool(
            len(trajectory) == len(STEPS)
            and all(int(row["gaussian_count"]) >= MIN_VALID_SAMPLE for row in trajectory)
            and all(float(row["VC_SH_positive_fraction"]) >= 0.90 for row in trajectory)
            and all(float(row["VC_medium_positive_fraction"]) >= 0.90 for row in trajectory)
            and all(
                math.isfinite(float(row["correlations"]["compensation_score"]["rho_vs_heldout_MSE"]))
                and math.isfinite(float(row["compensation_control_rhos"]["all_required_joint"]))
                for row in trajectory
            )
        )
        scene_pass = bool(final["checkpoint_entanglement_pass"] and temporal_stable and quality)
        scene_rows.append(
            {
                "scene": scene,
                "final_median_SH_medium_corr": float(final["median_SH_medium_corr"]),
                "final_negative_corr_fraction": float(final["negative_SH_medium_corr_fraction"]),
                "final_rho_compensation_vs_error": float(final["correlations"]["compensation_score"]["rho_vs_heldout_MSE"]),
                "final_auroc_compensation": float(final["correlations"]["compensation_score"]["auroc_top20_error"]),
                "final_rho_VC_SH_vs_error": float(final["correlations"]["VC_SH"]["rho_vs_heldout_MSE"]),
                "final_auroc_VC_SH": float(final["correlations"]["VC_SH"]["auroc_top20_error"]),
                "final_rho_VC_medium_vs_error": float(final["correlations"]["VC_medium"]["rho_vs_heldout_MSE"]),
                "final_auroc_VC_medium": float(final["correlations"]["VC_medium"]["auroc_top20_error"]),
                "depth_tau_transmission_control_rho": float(final["compensation_control_rhos"]["depth_tau_transmission_joint"]),
                "all_required_control_rho": float(final["compensation_control_rhos"]["all_required_joint"]),
                "ocmc_independence_control_rho": float(final["compensation_control_rhos"]["ocmc_independence_joint"]),
                "temporal_pass_count": pass_count,
                "temporal_stable": temporal_stable,
                "quality_pass": quality,
                "scene_supported": scene_pass,
            }
        )
        quality_rows.append(
            {
                "scene": scene,
                "checkpoint_count": len(trajectory),
                "minimum_valid_gaussians": min(int(row["gaussian_count"]) for row in trajectory),
                "minimum_VC_SH_positive_fraction": min(float(row["VC_SH_positive_fraction"]) for row in trajectory),
                "minimum_VC_medium_positive_fraction": min(float(row["VC_medium_positive_fraction"]) for row in trajectory),
                "quality_pass": quality,
            }
        )
        for row in trajectory:
            temporal_rows.append(
                {
                    "scene": scene,
                    "absolute_step": int(row["absolute_step"]),
                    "gaussian_count": int(row["gaussian_count"]),
                    "median_SH_medium_corr": float(row["median_SH_medium_corr"]),
                    "negative_SH_medium_corr_fraction": float(row["negative_SH_medium_corr_fraction"]),
                    "rho_compensation_vs_heldout_error": float(row["correlations"]["compensation_score"]["rho_vs_heldout_MSE"]),
                    "rho_after_depth_tau_transmission_control": float(row["compensation_control_rhos"]["depth_tau_transmission_joint"]),
                    "rho_after_all_required_controls": float(row["compensation_control_rhos"]["all_required_joint"]),
                    "rho_after_ocmc_independence_controls": float(row["compensation_control_rhos"]["ocmc_independence_joint"]),
                    "checkpoint_entanglement_pass": bool(row["checkpoint_entanglement_pass"]),
                    "temporal_scope": "distribution_level_no_lineage",
                }
            )
    quality_pass = all(row["quality_pass"] for row in quality_rows)
    supported_count = sum(row["scene_supported"] for row in scene_rows)
    if not quality_pass:
        label = "DATA_LIMITED"
    elif supported_count >= 3:
        label = "SUPPORTED"
    elif supported_count == 2:
        label = "TENTATIVE"
    else:
        label = "NOT_SUPPORTED"
    next_task = {
        "SUPPORTED": "APPEARANCE_MEDIUM_ENTANGLEMENT_MODULE_DESIGN",
        "TENTATIVE": "INDEPENDENT_FROZEN_SCENE_REPLICATION_BEFORE_ANY_MODULE_DESIGN",
        "NOT_SUPPORTED": "CLOSE_APPEARANCE_MEDIUM_ENTANGLEMENT_AND_RETURN_TO_FAILURE_HYPOTHESIS_SELECTION",
        "DATA_LIMITED": "RESOLVE_AUDIT_DATA_QUALITY_ONLY",
    }[label]
    classification = {
        "experiment": EXPERIMENT,
        "classification": label,
        "supported_scene_count": supported_count,
        "required_supported_scene_count": 3,
        "quality_gate_passed": quality_pass,
        "module_design_authorized": label == "SUPPORTED",
        "direction_closed": label == "NOT_SUPPORTED",
        "next_unique_task": next_task,
        "scene_rows": scene_rows,
        "quality_rows": quality_rows,
    }
    return classification, temporal_rows


def _integrity_after(preflight_result: Mapping[str, Any]) -> Dict[str, Any]:
    if _sha256(Path(__file__).resolve()) != preflight_result["repo"]["audit_script_sha256"]:
        raise RuntimeError("audit script changed during execution")
    protected_after = {relative: _sha256(REPO_ROOT / relative) for relative in PROTECTED_HASHES}
    if protected_after != preflight_result["repo"]["protected_hashes"]:
        raise RuntimeError("protected source changed during execution")
    checkpoints_after = []
    for row in preflight_result["checkpoint_rows"]:
        actual = _sha256(Path(row["checkpoint"]))
        if actual != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed during execution: {row['checkpoint']}")
        checkpoints_after.append({"checkpoint": row["checkpoint"], "sha256": actual})
    return {
        "audit_script_sha256_before_after_match": True,
        "protected_source_hashes_before_after_match": True,
        "checkpoint_hashes_before_after_match": True,
        "checkpoint_count": len(checkpoints_after),
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
    }


def _research_note(summary: Mapping[str, Any]) -> None:
    classification = summary["classification"]
    final = {row["scene"]: row for row in summary["final_checkpoint_rows"]}
    lines = [
        "# Appearance-Medium Entanglement Audit Under OCMC",
        "",
        "Date: 2026-09-01",
        f"Experiment: `{EXPERIMENT}`",
        f"Classification: `{classification['classification']}`",
        "",
        "## Hypothesis",
        "",
        "The frozen bounded-SH appearance residual and classic underwater medium representation may vary in opposite directions across training views. If this paired compensation predicts heldout RGB error after geometry, attenuation, opacity, footprint, and OCMC controls, it is a candidate failure mechanism. This audit does not identify physical ground-truth appearance or medium parameters.",
        "",
        "## Frozen Protocol",
        "",
        "All 20 registered C0 checkpoints use OCMC on, RAOC off, runtime-canonical `bounded_sh3`, `dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=false`, and classic rasterization. Historical source YAML files serialize the same bounded sigmoid-SH parameterization under its old `sigmoid_sh` name; the protected formal setup normalizes it to `bounded_sh3` before checkpoint loading, and every worker verifies the canonical runtime value. The audit uses detached forward passes only. It does not train, call backward or optimizer.step, modify model code, write checkpoints, or write renders.",
        "",
        "Sampling uses training visibility only with support at least two. SH/medium metrics and samples are frozen before heldout GT is accessed. Heldout GT is used only as the error outcome in projected Gaussian footprint boxes and camera RGB MSE.",
        "",
        "## Metric Definition",
        "",
        "For opacity-area weight `w`, `C_SH = w*(RGB_fullSH-RGB_DC)`. `C_medium` is the sum of the DC direct attenuation residual `w*(T_D-1)*RGB_DC` and the existing renderer-integrated `rgb_medium` sampled at the projected Gaussian center and weighted by `w`. The explicit SH-attenuation interaction is `w*(T_D-1)*(RGB_fullSH-RGB_DC)`. `VC_SH` and `VC_medium` are population RGB variances over visible training cameras.",
        "",
        "`SH_medium_corr` is the centered vector correlation of paired `C_SH` and `C_medium` across training views. `compensation_score=-SH_medium_corr`, so a positive score means opposite variation. This is a representational association proxy, not a causal intervention.",
        "",
        "## Final Results",
        "",
        "| Scene | median SH-medium corr | rho(comp,error) | controlled depth/tau/T | controlled OCMC | rho VC_SH | AUROC SH | rho VC_med | AUROC med | temporal | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in classification["scene_rows"]:
        lines.append(
            f"| {row['scene']} | {row['final_median_SH_medium_corr']:.6f} | {row['final_rho_compensation_vs_error']:.6f} | {row['depth_tau_transmission_control_rho']:.6f} | {row['ocmc_independence_control_rho']:.6f} | {row['final_rho_VC_SH_vs_error']:.6f} | {row['final_auroc_VC_SH']:.6f} | {row['final_rho_VC_medium_vs_error']:.6f} | {row['final_auroc_VC_medium']:.6f} | {row['temporal_pass_count']}/5 | {'yes' if row['scene_supported'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Residual Decomposition",
            "",
            "| Scene | rho SH align | rho medium align | rho interaction | rho joint | best single | rank R2 multivariate |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for scene in SCENES:
        row = final[scene]
        corr = row["correlations"]
        r2 = row["residual_decomposition_rank_R2"]
        lines.append(
            f"| {scene} | {corr['heldout_SH_alignment']['rho_vs_heldout_MSE']:.6f} | {corr['heldout_medium_alignment']['rho_vs_heldout_MSE']:.6f} | {corr['heldout_interaction_alignment']['rho_vs_heldout_MSE']:.6f} | {corr['heldout_joint_alignment']['rho_vs_heldout_MSE']:.6f} | {row['residual_decomposition_best_single_predictor']} | {r2['SH_medium_interaction_multivariate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "These decomposition values measure rank association with heldout local error; they are not causal MSE attribution and do not imply true component ownership.",
            "",
            "## Temporal Stability",
            "",
            "No checkpoint contains persistent Gaussian lineage IDs. Temporal recurrence is therefore population-level only; array-index and nearest-geometry identity matching were not used.",
            "",
            "| Scene | 5k | 8k | 10k | 13k | 14999 | passes |",
            "|---|:---:|:---:|:---:|:---:|:---:|---:|",
        ]
    )
    for scene in SCENES:
        rows = [row for row in summary["temporal_stability"] if row["scene"] == scene]
        flags = ["yes" if row["checkpoint_entanglement_pass"] else "no" for row in rows]
        lines.append(f"| {scene} | {' | '.join(flags)} | {sum(row['checkpoint_entanglement_pass'] for row in rows)}/5 |")
    lines.extend(
        [
            "",
            "## OCMC Independence",
            "",
            "The OCMC independence test jointly rank-residualizes compensation score and heldout error against OCMC active projected magnitude and suppressed medium residual. The stricter all-control result additionally includes depth, tau, transmission, opacity, and footprint. OCMC and model state hashes remain unchanged for every checkpoint.",
            "",
            "## Limitations",
            "",
            "`rgb_medium` is an exact renderer-integrated ray contribution but not a per-Gaussian physical attribution; weighting its projected-center sample by the Gaussian opacity-area proxy only associates that ray contribution with a Gaussian. Projected footprint boxes overlap, occlusion is not assigned exactly, heldout camera counts are small, and no true appearance/medium labels exist. The audit is observational and cannot establish causal compensation.",
            "",
            "## Classification",
            "",
            f"The formal result is `{classification['classification']}` with {classification['supported_scene_count']}/4 supported scenes. Module design authorization is `{str(classification['module_design_authorized']).lower()}`.",
            "",
        ]
    )
    if classification["classification"] == "SUPPORTED":
        lines.append("The next and only authorized task is a separate appearance-medium entanglement module-design phase. No module was implemented here.")
    elif classification["classification"] == "TENTATIVE":
        lines.append("The direction remains only possible. The next task is an independent frozen-scene replication; module design is not authorized.")
    elif classification["classification"] == "NOT_SUPPORTED":
        lines.append("Appearance-medium entanglement is not supported as a novel-view failure mechanism under this protocol. Close this direction and return to failure-hypothesis selection; do not design a module from this signal.")
    else:
        lines.append("The audit is data-limited. Resolve only the stated data-quality failure before drawing a mechanism conclusion.")
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"Analyzed {summary['per_gaussian_rows']} Gaussian-checkpoint rows and {summary['per_camera_rows']} heldout camera-checkpoint rows. All 20 checkpoint hashes and protected source hashes matched before and after execution. Backward calls, optimizer steps, checkpoint writes, and render writes were all zero.",
            "",
        ]
    )
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def aggregate() -> Dict[str, Any]:
    preflight_result = _read_json(OUTPUT_ROOT / "preflight.json")
    worker_summaries = [_read_json(OUTPUT_ROOT / "workers" / scene / "worker_summary.json") for scene in SCENES]
    if not all(
        row["backward_calls"] == 0
        and row["optimizer_step_calls"] == 0
        and row["checkpoint_writes"] == 0
        and row["render_writes"] == 0
        and len(row["checkpoint_rows"]) == len(STEPS)
        and all(item["model_and_ocmc_unchanged"] for item in row["checkpoint_rows"])
        for row in worker_summaries
    ):
        raise RuntimeError("worker integrity gate failed")
    gaussian_raw: List[Dict[str, str]] = []
    camera_raw: List[Dict[str, str]] = []
    control_raw: List[Dict[str, str]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    for scene, worker_summary in zip(SCENES, worker_summaries):
        scene_dir = OUTPUT_ROOT / "workers" / scene
        gaussian_raw.extend(_read_csv(scene_dir / "per_gaussian_sh_medium_metrics.csv"))
        camera_raw.extend(_read_csv(scene_dir / "per_camera_error_alignment.csv"))
        control_raw.extend(_read_csv(scene_dir / "control_analysis.csv"))
        checkpoint_rows.extend(worker_summary["checkpoint_rows"])
    gaussian_rows = _coerce_rows(gaussian_raw)
    camera_rows = _coerce_rows(camera_raw)
    control_rows = _coerce_rows(control_raw)
    classification, temporal_rows = _classify(checkpoint_rows)
    camera_analysis = _camera_analysis(camera_rows)
    integrity = _integrity_after(preflight_result)
    final_checkpoint_rows = [row for row in checkpoint_rows if int(row["absolute_step"]) == FINAL_STEP]
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "final_checkpoint_rows": final_checkpoint_rows,
        "all_checkpoint_rows": checkpoint_rows,
        "temporal_stability": temporal_rows,
        "camera_analysis": camera_analysis,
        "worker_summaries": worker_summaries,
        "per_gaussian_rows": len(gaussian_rows),
        "per_camera_rows": len(camera_rows),
        "control_rows": len(control_rows),
        "integrity": integrity,
        "frozen_forward_only": True,
        "ocmc_enabled": True,
        "raoc_enabled": False,
        "module_design_started": False,
        "backward_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
    }
    _write_csv(OUTPUT_ROOT / "per_gaussian_sh_medium_metrics.csv", gaussian_rows)
    _write_csv(OUTPUT_ROOT / "per_camera_error_alignment.csv", camera_rows)
    _write_csv(OUTPUT_ROOT / "temporal_stability.csv", temporal_rows)
    _write_csv(OUTPUT_ROOT / "control_analysis.csv", control_rows)
    _write_json(OUTPUT_ROOT / "classification.json", classification)
    _write_json(OUTPUT_ROOT / "final_summary.json", summary)
    _research_note(summary)
    return summary


def launch() -> Dict[str, Any]:
    preflight_result = preflight()
    logs = OUTPUT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for scene, gpu in SCENE_GPUS.items():
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        command = [str(PYTHON), str(Path(__file__).resolve()), "--worker", "--scene", scene, "--gpu", gpu]
        handle = (logs / f"{scene}.log").open("w", encoding="utf8")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((scene, gpu, process, handle))
    failures = []
    for scene, gpu, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append({"scene": scene, "gpu": gpu, "exit_code": code, "log": str(logs / f"{scene}.log")})
    if failures:
        _write_json(OUTPUT_ROOT / "worker_failures.json", {"rows": failures})
        raise RuntimeError(f"appearance-medium audit workers failed: {failures}")
    return {"preflight": preflight_result, "summary": aggregate()}


def _parse_steps(value: str) -> Tuple[int, ...]:
    steps = tuple(int(item) for item in value.split(",") if item)
    if not steps or any(step not in STEPS for step in steps):
        raise argparse.ArgumentTypeError(f"steps must be comma-separated members of {STEPS}")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=tuple(SCENE_GPUS.values()))
    parser.add_argument("--steps", type=_parse_steps, default=STEPS)
    parser.add_argument("--sample-count", type=int)
    args = parser.parse_args()
    if args.worker:
        if args.scene is None or args.gpu is None:
            parser.error("--worker requires --scene and --gpu")
        if args.sample_count is not None and args.sample_count < 20:
            parser.error("--sample-count must be at least 20")
        result = worker(args.scene, args.gpu, args.steps, args.sample_count)
    elif args.preflight:
        result = preflight()
    elif args.aggregate:
        result = aggregate()
    else:
        result = launch()
    print(json.dumps(_sanitize(result), indent=2, sort_keys=True, default=_json_default, allow_nan=False))


if __name__ == "__main__":
    main()
