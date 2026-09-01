#!/usr/bin/env python3
"""Frozen Gaussian parameter identifiability audit under registered OCMC.

The audit constructs local parameter-group response subspaces from detached
forward quantities. Sampling and every ambiguity metric are frozen before
heldout ground truth is opened. No autograd backward/JVP/VJP is used.
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

from scripts.diagnostics import audit_appearance_medium_entanglement_ocmc as AME
from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.diagnostics import audit_gaussian_view_consistency_ocmc as VC
from scripts.diagnostics import audit_low_support_causal_intervention as CAUSAL
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL
from water_splatting._torch_impl import eval_sh_bases


EXPERIMENT = "GAUSSIAN_PARAMETER_IDENTIFIABILITY_AUDIT_OCMC"
EXPECTED_BRANCH = "research/m1-bounded-intrinsic"
EXPECTED_HEAD = "3d5c461fffcbb158465f9aa7fe37cb6db9fed5d0"
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "gaussian_parameter_identifiability_audit_20260901"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "GAUSSIAN_PARAMETER_IDENTIFIABILITY_AUDIT_2026-09-01.md"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENE_GPUS = dict(CAUSAL.SCENE_GPUS)
SCENES = tuple(SCENE_GPUS)
STEPS = (5000, 8000, 10000, 13000, 14999)
FINAL_STEP = 14999
SEED = 42
TEMPORAL_SAMPLE_COUNT = 256
FINAL_SAMPLE_COUNT = 512
SUPPORT_STRATA = 5
MIN_VALID_SAMPLE = 128
FOOTPRINT_OFFSETS = ((0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5))
GROUPS = ("Medium", "SH", "Opacity", "Geometry")
PAIRS = (
    ("Medium", "SH"),
    ("Medium", "Opacity"),
    ("Medium", "Geometry"),
    ("SH", "Opacity"),
    ("SH", "Geometry"),
    ("Opacity", "Geometry"),
)
OVERLAP_THRESHOLD = 0.80
ANGLE_THRESHOLD_DEGREES = 20.0
SVD_RELATIVE_THRESHOLD = 1e-4
SVD_ENERGY_THRESHOLD = 0.999
EPS = 1e-12

CONTROLS = (
    "train_depth_mean",
    "train_tau_mean",
    "train_transmission_mean",
    "train_accumulation_mean",
    "opacity",
    "train_footprint_mean",
    "train_ocmc_active_magnitude_mean",
    "train_medium_suppressed_residual_mean",
)
DEPTH_MEDIUM_CONTROLS = ("train_depth_mean", "train_tau_mean", "train_transmission_mean")
OCMC_CONTROLS = ("train_ocmc_active_magnitude_mean", "train_medium_suppressed_residual_mean")
REQUIRED_JOINT_CONTROLS = (
    "train_depth_mean",
    "train_tau_mean",
    "train_transmission_mean",
    "train_accumulation_mean",
    "opacity",
    "train_footprint_mean",
    "train_ocmc_active_magnitude_mean",
    "train_medium_suppressed_residual_mean",
)

PROTECTED_HASHES = {
    **AME.PROTECTED_HASHES,
    "scripts/diagnostics/audit_appearance_medium_entanglement_ocmc.py": "60a812b027258dbb19897ee31bd530a8d86dc9beff9bc5305a318bee81e91cb6",
    "water_splatting/_torch_impl.py": "e43def3026272cc168c96a68d9949678da68aaeb7136946be712e406e95c5977",
    "water_splatting/fields/gaussian_appearance.py": "a15a939d2023e4659184c460bc9010002c18dde11247e4eeedc4a353e814385b",
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
            "observation_space": "For each Gaussian, stack RGB responses at five fixed local footprint offsets over every visible training camera; shape [15*support, parameter dimension].",
            "isolated_signal": "s=alpha*(exp(-beta_D*d)*bounded_SH_RGB-exp(-beta_B*d)*B_inf), relative to pure tied-B_inf background",
            "medium_group": "analytic derivatives of s with respect to physical B_inf[3], beta_B[3], beta_D[3]",
            "sh_group": "analytic derivatives through bounded_sh3 sigmoid with respect to the full 16x3 appearance coefficient group, including DC",
            "opacity_group": "analytic derivative with respect to raw opacity logit",
            "geometry_group": "analytic screen-space center displacement derivatives x/y with conic and depth fixed",
            "subspace_rank": "minimum rank retaining 99.9% response energy among singular values >=1e-4 of the largest singular value",
            "normalized_overlap": "||U_A^T U_B||_F/sqrt(min(rank_A,rank_B)); range [0,1]",
            "principal_angle": "minimum principal angle in degrees from the maximum singular value of U_A^T U_B",
            "ambiguity_score": "maximum pairwise normalized overlap multiplied by total group sensitivity sqrt(sum_group mean(J_group^2))",
            "removal": {
                "medium": "set beta_D=beta_B=0 and B_inf=0 in the isolated local signal and measure the resulting RGB delta",
                "SH": "replace full bounded SH RGB by bounded DC RGB, removing only the non-DC SH contribution",
                "opacity": "set local Gaussian opacity/alpha to zero",
                "geometry": "replace five footprint samples by their center signal, removing screen-displacement variation",
            },
            "removal_error_alignment": "Spearman correlation between absolute local removal-delta RGB RMS and heldout error; baseline-relative removal sensitivity is also reported per Gaussian but is not used for opacity alignment because opacity removal is identically the full isolated signal",
            "heldout_error": "mean heldout RGB MSE in clipped projected-radius footprint boxes",
            "cross_view_stability": "per-Gaussian coefficient of variation of each parameter-group response RMS over visible training views; medians and heldout-error correlations are descriptive and do not alter the preregistered classification gate",
            "gt_boundary": "sampling, local Jacobians, overlap, ambiguity, and removal metrics are frozen before heldout GT access",
        },
        "classification_rule": {
            "obvious_overlap": f"checkpoint population median maximum overlap >= {OVERLAP_THRESHOLD} and median minimum angle <= {ANGLE_THRESHOLD_DEGREES} degrees",
            "checkpoint_pass": "obvious overlap, rho(ambiguity,error)>0, depth/tau/transmission joint controlled rho>0, OCMC+suppressed-residual controlled rho>0, and all-required-control rho>0",
            "scene_pass": "final checkpoint passes and checkpoint_pass recurs at >=3/5 checkpoints",
            "SUPPORTED": ">=3/4 scene passes with quality gate passing",
            "TENTATIVE": "exactly 2/4 scene passes, or >=2/4 partial candidate scenes, with quality gate passing",
            "NOT_SUPPORTED": "otherwise with quality gate passing",
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
        "jvp_calls": 0,
        "vjp_calls": 0,
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


def _stable_seed(*parts: Any) -> int:
    payload = ":".join([str(SEED)] + [str(part) for part in parts]).encode("utf8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


@torch.no_grad()
def _sample_gaussians(
    scene: str,
    step: int,
    support: Tensor,
    requested: int,
) -> Tuple[Tensor, Dict[str, Any]]:
    eligible = torch.nonzero(support >= 2, as_tuple=False).reshape(-1)
    if int(eligible.numel()) < requested:
        raise RuntimeError(f"{scene}/{step} only has {eligible.numel()} eligible Gaussians")
    order = eligible[torch.argsort(support[eligible].to(torch.int32), stable=True)]
    strata = torch.tensor_split(order, SUPPORT_STRATA)
    generator = torch.Generator(device="cpu")
    seed = _stable_seed(scene, step, "identifiability-support-stratified-sample")
    generator.manual_seed(seed)
    base = requested // SUPPORT_STRATA
    remainder = requested % SUPPORT_STRATA
    selections = []
    rows = []
    for index, pool in enumerate(strata):
        count = base + int(index < remainder)
        chosen = pool[torch.randperm(int(pool.numel()), generator=generator)[:count]]
        selections.append(chosen)
        rows.append(
            {
                "stratum": index,
                "pool_count": int(pool.numel()),
                "sample_count": count,
                "support_min": int(support[pool].min()),
                "support_max": int(support[pool].max()),
            }
        )
    selected = torch.cat(selections).sort().values
    return selected, {
        "seed": seed,
        "requested_count": requested,
        "selected_count": int(selected.numel()),
        "eligible_count": int(eligible.numel()),
        "eligibility": "training support>=2; no heldout camera or GT used",
        "strata": rows,
        "selected_ids_sha256": CAUSAL._tensor_hash(selected),
    }


def _sample_image(values: Tensor, xys: Tensor) -> Tensor:
    height, width = int(values.shape[0]), int(values.shape[1])
    x = xys[:, 0].round().long().clamp(0, width - 1)
    y = xys[:, 1].round().long().clamp(0, height - 1)
    return values[y, x]


def _viewdirs_and_bases(model: Any, camera: Any, global_ids: Tensor) -> Tuple[Tensor, Tensor]:
    position = camera.camera_to_worlds[..., :3, 3].to(model.device)
    viewdirs = model.means.detach()[global_ids] - position.detach()
    viewdirs = viewdirs / torch.linalg.vector_norm(viewdirs, dim=-1, keepdim=True).clamp_min(1e-6)
    return viewdirs, eval_sh_bases(16, viewdirs).float()


def _local_response(
    model: Any,
    camera: Any,
    outputs: Mapping[str, Tensor],
    global_ids: Tensor,
    opacity: Tensor,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    xys = model.xys.detach()[global_ids].float()
    depth = outputs["projected_gaussian_depths"].detach().reshape(-1)[global_ids].float()
    radii = model.radii.detach().reshape(-1)[global_ids].float()
    colors = outputs["gaussian_view_rgb"].detach().float()[global_ids]
    dc = outputs["gaussian_view_dc_rgb"].detach().float()[global_ids]
    b_inf = _sample_image(outputs["b_inf"].detach().float(), xys)
    beta_b = _sample_image(outputs["medium_bs"].detach().float(), xys)
    beta_d = _sample_image(outputs["medium_attn"].detach().float(), xys)
    active = torch.linalg.vector_norm(
        _sample_image(outputs["camera_medium_delta_projected_raw"].detach().float(), xys), dim=-1
    )
    suppressed = torch.linalg.vector_norm(
        _sample_image(outputs["camera_medium_delta_suppressed_raw"].detach().float(), xys), dim=-1
    )
    transmission_d = torch.exp(-(beta_d * depth[:, None]).clamp_min(0.0)).clamp(0.0, 1.0)
    transmission_b = torch.exp(-(beta_b * depth[:, None]).clamp_min(0.0)).clamp(0.0, 1.0)
    viewdirs, bases = _viewdirs_and_bases(model, camera, global_ids)
    sigmoid_derivative = colors * (1.0 - colors)
    offsets = torch.tensor(FOOTPRINT_OFFSETS, device=model.device, dtype=torch.float32)
    scale = radii.clamp_min(0.5)[:, None, None] * 0.5
    local_xy = xys[:, None, :] + offsets[None, :, :] * scale
    delta = local_xy - xys[:, None, :]
    # Isotropic local EWA proxy uses the measured projected radius. This keeps
    # geometry in legal screen-displacement coordinates without changing geometry.
    sigma = delta.square().sum(dim=-1) / (2.0 * radii.clamp_min(0.5).square()[:, None])
    gaussian = torch.exp(-sigma)
    alpha_uncapped = opacity[:, None] * gaussian
    alpha = alpha_uncapped.clamp(max=0.999)
    cap_grad = (alpha_uncapped < 0.999).float()
    object_term = transmission_d * colors
    background_term = transmission_b * b_inf
    contrast = object_term - background_term
    signal = alpha[..., None] * contrast[:, None, :]

    count = int(global_ids.numel())
    medium = torch.zeros(count, len(FOOTPRINT_OFFSETS), 3, 9, device=model.device)
    for channel in range(3):
        medium[:, :, channel, channel] = -alpha * transmission_b[:, channel][:, None]
        medium[:, :, channel, 3 + channel] = (
            alpha * depth[:, None] * transmission_b[:, channel][:, None] * b_inf[:, channel][:, None]
        )
        medium[:, :, channel, 6 + channel] = (
            -alpha * depth[:, None] * transmission_d[:, channel][:, None] * colors[:, channel][:, None]
        )

    sh = torch.zeros(count, len(FOOTPRINT_OFFSETS), 3, 48, device=model.device)
    for channel in range(3):
        columns = slice(channel, 48, 3)
        derivative = (
            alpha[:, :, None]
            * transmission_d[:, channel][:, None, None]
            * sigmoid_derivative[:, channel][:, None, None]
            * bases[:, None, :]
        )
        sh[:, :, channel, columns] = derivative

    dalpha_dlogit = cap_grad * gaussian * (opacity * (1.0 - opacity))[:, None]
    opacity_j = dalpha_dlogit[..., None, None] * contrast[:, None, :, None]
    geometry = torch.zeros(count, len(FOOTPRINT_OFFSETS), 3, 2, device=model.device)
    ds_dx = gaussian * delta[..., 0] / radii.clamp_min(0.5).square()[:, None]
    ds_dy = gaussian * delta[..., 1] / radii.clamp_min(0.5).square()[:, None]
    geometry[..., 0] = (cap_grad * opacity[:, None] * ds_dx)[..., None] * contrast[:, None, :]
    geometry[..., 1] = (cap_grad * opacity[:, None] * ds_dy)[..., None] * contrast[:, None, :]

    remove_medium = alpha[..., None] * (
        (1.0 - transmission_d)[:, None, :] * colors[:, None, :]
        + transmission_b[:, None, :] * b_inf[:, None, :]
    )
    remove_sh = alpha[..., None] * transmission_d[:, None, :] * (colors - dc)[:, None, :]
    remove_opacity = signal
    remove_geometry = signal - signal[:, :1, :].expand_as(signal)
    sensitivity = {
        "Medium": medium,
        "SH": sh,
        "Opacity": opacity_j,
        "Geometry": geometry,
    }
    meta = {
        "depth": depth,
        "tau": (beta_d * depth[:, None]).mean(dim=-1),
        "transmission": transmission_d.mean(dim=-1),
        "accumulation": _sample_image(
            outputs["accumulation"].detach().float(), xys
        ).reshape(-1),
        "footprint": radii,
        "ocmc_active_magnitude": active,
        "medium_suppressed_residual": suppressed,
        "remove_medium": remove_medium,
        "remove_sh": remove_sh,
        "remove_opacity": remove_opacity,
        "remove_geometry": remove_geometry,
        "baseline_signal": signal,
        "viewdirs": viewdirs,
    }
    return sensitivity, meta


def _basis(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0 or not np.isfinite(matrix).all():
        return np.empty((matrix.shape[0], 0)), np.empty(0)
    u, singular, _vh = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] <= EPS:
        return np.empty((matrix.shape[0], 0)), singular
    valid = singular >= singular[0] * SVD_RELATIVE_THRESHOLD
    energy = np.square(singular)
    cumulative = np.cumsum(energy) / max(float(energy.sum()), EPS)
    energy_rank = int(np.searchsorted(cumulative, SVD_ENERGY_THRESHOLD) + 1)
    rank = min(int(valid.sum()), energy_rank)
    rank = max(rank, 1)
    return u[:, :rank], singular


def _coefficient_of_variation(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return float("nan")
    mean = abs(float(np.mean(array)))
    if mean <= EPS:
        return float("nan")
    return float(np.std(array, ddof=0) / mean)


def _finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_median(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def _overlap(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    ua, sa = _basis(left)
    ub, sb = _basis(right)
    if ua.shape[1] == 0 or ub.shape[1] == 0:
        return {
            "principal_angle": float("nan"),
            "overlap": float("nan"),
            "rank_left": ua.shape[1],
            "rank_right": ub.shape[1],
            "left_largest_singular": float(sa[0]) if sa.size else 0.0,
            "right_largest_singular": float(sb[0]) if sb.size else 0.0,
        }
    singular = np.linalg.svd(ua.T @ ub, compute_uv=False).clip(0.0, 1.0)
    return {
        "principal_angle": math.degrees(math.acos(float(singular.max()))),
        "overlap": float(np.linalg.norm(singular) / math.sqrt(min(ua.shape[1], ub.shape[1]))),
        "rank_left": ua.shape[1],
        "rank_right": ub.shape[1],
        "left_largest_singular": float(sa[0]),
        "right_largest_singular": float(sb[0]),
    }


@torch.no_grad()
def _training_metrics(
    model: Any,
    records: Sequence[Tuple[int, str, Any, Any]],
    selected: Tensor,
    support: Tensor,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    count = int(selected.numel())
    selected_gpu = selected.to(model.device)
    opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)[selected_gpu].float()
    response: Dict[str, List[List[np.ndarray]]] = {
        group: [[] for _ in range(count)] for group in GROUPS
    }
    removal_names = ("medium", "sh", "opacity", "geometry")
    removal_relative: Dict[str, List[List[float]]] = {
        name: [[] for _ in range(count)] for name in removal_names
    }
    removal_absolute: Dict[str, List[List[float]]] = {
        name: [[] for _ in range(count)] for name in removal_names
    }
    response_norms: Dict[str, List[List[float]]] = {
        group: [[] for _ in range(count)] for group in GROUPS
    }
    meta_names = (
        "depth",
        "tau",
        "transmission",
        "accumulation",
        "footprint",
        "ocmc_active_magnitude",
        "medium_suppressed_residual",
    )
    meta_values: Dict[str, List[List[float]]] = {
        name: [[] for _ in range(count)] for name in meta_names
    }
    observed = torch.zeros(count, dtype=torch.int16)
    for _index, _camera_id, camera, _batch in records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1)[selected_gpu].bool()
        if bool(visible.any()):
            local = torch.nonzero(visible, as_tuple=False).reshape(-1)
            global_ids = selected_gpu[local]
            sensitivity, meta = _local_response(model, camera, outputs, global_ids, opacity[local])
            for position, local_index in enumerate(local.tolist()):
                for group in GROUPS:
                    matrix = sensitivity[group][position].reshape(-1, sensitivity[group].shape[-1])
                    response[group][local_index].append(matrix.double().cpu().numpy())
                    response_norms[group][local_index].append(float(torch.sqrt(matrix.square().mean())))
                baseline_rms = float(torch.sqrt(meta["baseline_signal"][position].square().mean()))
                for name in removal_names:
                    value = float(torch.sqrt(meta[f"remove_{name}"][position].square().mean()))
                    removal_absolute[name][local_index].append(value)
                    removal_relative[name][local_index].append(value / max(baseline_rms, EPS))
                for name in meta_names:
                    meta_values[name][local_index].append(float(meta[name][position]))
            observed += visible.cpu().to(torch.int16)
        del outputs
    if not torch.equal(observed, support[selected]):
        raise RuntimeError("training response visibility differs from frozen support")

    gaussian_rows = []
    overlap_rows = []
    scale = torch.exp(model.scales.detach()).amax(dim=-1).cpu()[selected]
    for local, gaussian_id in enumerate(selected.tolist()):
        matrices = {group: np.concatenate(response[group][local], axis=0) for group in GROUPS}
        pair_values = {}
        for left, right in PAIRS:
            values = _overlap(matrices[left], matrices[right])
            pair = f"{left}-{right}"
            pair_values[pair] = values
            overlap_rows.append(
                {
                    "scene": "",
                    "checkpoint": "",
                    "absolute_step": 0,
                    "gaussian_id": gaussian_id,
                    "pair": pair,
                    **values,
                    "observation_rows": int(matrices[left].shape[0]),
                }
            )
        finite_pairs = [item for item in pair_values.items() if math.isfinite(item[1]["overlap"])]
        max_pair, max_values = max(finite_pairs, key=lambda item: item[1]["overlap"])
        sensitivities = {
            group: float(np.mean(response_norms[group][local])) for group in GROUPS
        }
        cross_view_cv = {
            group: _coefficient_of_variation(response_norms[group][local]) for group in GROUPS
        }
        total_sensitivity = math.sqrt(sum(value * value for value in sensitivities.values()))
        row = {
            "gaussian_id": gaussian_id,
            "scene": "",
            "checkpoint": "",
            "absolute_step": 0,
            "gaussian_identity_persistent_across_checkpoints": False,
            "support_count": int(support[gaussian_id]),
            "ambiguity_score": float(max_values["overlap"] * total_sensitivity),
            "maximum_pairwise_overlap": float(max_values["overlap"]),
            "minimum_principal_angle": float(min(v["principal_angle"] for _p, v in finite_pairs)),
            "maximum_overlap_pair": max_pair,
            "medium_sensitivity": sensitivities["Medium"],
            "SH_sensitivity": sensitivities["SH"],
            "opacity_sensitivity": sensitivities["Opacity"],
            "geometry_sensitivity": sensitivities["Geometry"],
            "total_sensitivity": total_sensitivity,
            "medium_cross_view_cv": cross_view_cv["Medium"],
            "SH_cross_view_cv": cross_view_cv["SH"],
            "opacity_cross_view_cv": cross_view_cv["Opacity"],
            "geometry_cross_view_cv": cross_view_cv["Geometry"],
            "mean_cross_view_response_cv": _finite_mean(list(cross_view_cv.values())),
            "opacity": float(opacity[local]),
            "scale": float(scale[local]),
            "observed_train_views": int(observed[local]),
            "remove_medium_RGB_sensitivity": float(np.mean(removal_absolute["medium"][local])),
            "remove_SH_RGB_sensitivity": float(np.mean(removal_absolute["sh"][local])),
            "remove_opacity_RGB_sensitivity": float(np.mean(removal_absolute["opacity"][local])),
            "remove_geometry_RGB_sensitivity": float(np.mean(removal_absolute["geometry"][local])),
            "remove_medium_relative_RGB_sensitivity": float(np.mean(removal_relative["medium"][local])),
            "remove_SH_relative_RGB_sensitivity": float(np.mean(removal_relative["sh"][local])),
            "remove_opacity_relative_RGB_sensitivity": float(np.mean(removal_relative["opacity"][local])),
            "remove_geometry_relative_RGB_sensitivity": float(np.mean(removal_relative["geometry"][local])),
            "metric_uses_heldout_or_gt": False,
            "gt_used_after_metric_for_error_only": True,
        }
        for name in meta_names:
            row[f"train_{name}_mean"] = float(np.mean(meta_values[name][local]))
        for pair, values in pair_values.items():
            key = pair.lower().replace("-", "_")
            row[f"overlap_{key}"] = float(values["overlap"])
            row[f"angle_{key}"] = float(values["principal_angle"])
        gaussian_rows.append(row)
    return gaussian_rows, overlap_rows


@torch.no_grad()
def _heldout_error(
    scene: str,
    step: int,
    model: Any,
    records: Sequence[Tuple[int, str, Any, Any]],
    selected: Tensor,
    metric_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    selected_gpu = selected.to(model.device)
    totals: Dict[int, Dict[str, float]] = {
        int(gaussian_id): {"error": 0.0, "high": 0.0, "count": 0.0} for gaussian_id in selected.tolist()
    }
    by_id = {int(row["gaussian_id"]): row for row in metric_rows}
    camera_rows = []
    for _index, camera_id, camera, batch in records:
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1)[selected_gpu].bool()
        local = torch.nonzero(visible, as_tuple=False).reshape(-1)
        global_ids = selected_gpu[local]
        # The training response metric and sample are frozen before this GT access.
        gt = MI.PW._get_gt(model, batch, outputs["background"]).detach().float().clamp(0, 1)
        pred = outputs["pred_image"].detach().float().clamp(0, 1)
        residual = (pred - gt).square().mean(dim=-1)
        threshold = torch.quantile(residual.reshape(-1), 0.80)
        high = residual >= threshold
        if bool(visible.any()):
            projected, high_fraction, _area = VC._box_statistics(
                residual,
                high,
                model.xys.detach()[global_ids],
                model.radii.detach().reshape(-1)[global_ids],
            )
            for position, gaussian_id in enumerate(global_ids.tolist()):
                totals[gaussian_id]["error"] += float(projected[position])
                totals[gaussian_id]["high"] += float(high_fraction[position])
                totals[gaussian_id]["count"] += 1.0
        camera_ambiguity = [
            float(by_id[int(gaussian_id)]["ambiguity_score"]) for gaussian_id in global_ids.tolist()
        ]
        camera_rows.append(
            {
                "scene": scene,
                "checkpoint": str(_checkpoint(scene, step)),
                "absolute_step": step,
                "camera_id": camera_id,
                "sampled_visible_gaussian_count": int(visible.sum()),
                "camera_ambiguity_mean": float(np.mean(camera_ambiguity)) if camera_ambiguity else float("nan"),
                "camera_MSE": float(residual.mean()),
                "camera_PSNR_error": 10.0 * math.log10(max(float(residual.mean()), EPS)),
                "metric_uses_heldout_or_gt": False,
                "gt_used_for_error_outcome_only": True,
            }
        )
        del outputs, gt, pred, residual, high
    result = {}
    for gaussian_id, item in totals.items():
        if item["count"] > 0:
            result[gaussian_id] = {
                "heldout_visible_views": int(item["count"]),
                "heldout_projected_residual": item["error"] / item["count"],
                "heldout_high_error_pixel_fraction": item["high"] / item["count"],
            }
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


def _checkpoint_summary(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ambiguity = [float(row["ambiguity_score"]) for row in rows]
    error = [float(row["heldout_projected_residual"]) for row in rows]
    psnr_error = [float(row["heldout_PSNR_error"]) for row in rows]
    labels = [bool(row["high_error_label_top20"]) for row in rows]
    rho, pvalue, count = _rho(ambiguity, error)
    psnr_rho, psnr_pvalue, psnr_count = _rho(ambiguity, psnr_error)
    control_rows = [
        {
            "scene": rows[0]["scene"],
            "checkpoint": rows[0]["checkpoint"],
            "absolute_step": int(rows[0]["absolute_step"]),
            "control_set": "raw",
            "controls": "",
            "rho": rho,
            "pvalue": pvalue,
            "valid_count": count,
        }
    ]
    sets = [(name, (name,)) for name in CONTROLS]
    sets.extend(
        [
            ("depth_tau_transmission_joint", DEPTH_MEDIUM_CONTROLS),
            ("ocmc_independence_joint", OCMC_CONTROLS),
            ("all_required_joint", REQUIRED_JOINT_CONTROLS),
        ]
    )
    controls = {}
    for set_name, names in sets:
        value, value_p, value_count = _partial_rank_multi(
            ambiguity,
            error,
            [[float(row[name]) for row in rows] for name in names],
        )
        controls[set_name] = value
        control_rows.append(
            {
                "scene": rows[0]["scene"],
                "checkpoint": rows[0]["checkpoint"],
                "absolute_step": int(rows[0]["absolute_step"]),
                "control_set": set_name,
                "controls": ";".join(names),
                "rho": value,
                "pvalue": value_p,
                "valid_count": value_count,
            }
        )
    overlaps = np.asarray([float(row["maximum_pairwise_overlap"]) for row in rows])
    angles = np.asarray([float(row["minimum_principal_angle"]) for row in rows])
    obvious = bool(np.median(overlaps) >= OVERLAP_THRESHOLD and np.median(angles) <= ANGLE_THRESHOLD_DEGREES)
    checkpoint_pass = bool(
        obvious
        and rho > 0.0
        and controls["depth_tau_transmission_joint"] > 0.0
        and controls["ocmc_independence_joint"] > 0.0
        and controls["all_required_joint"] > 0.0
    )
    pair_medians = {}
    pair_finite_fractions = {}
    for left, right in PAIRS:
        pair = f"{left}-{right}"
        values = np.asarray(
            [float(row[f"overlap_{left.lower()}_{right.lower()}"]) for row in rows], dtype=np.float64
        )
        pair_medians[pair] = _finite_median(values)
        pair_finite_fractions[pair] = float(np.mean(np.isfinite(values)))
    finite_pair_medians = {key: value for key, value in pair_medians.items() if math.isfinite(value)}
    if not finite_pair_medians:
        raise RuntimeError("checkpoint has no finite parameter-pair overlap medians")
    removal_rhos = {}
    for name in ("medium", "SH", "opacity", "geometry"):
        key = f"remove_{name}_RGB_sensitivity"
        removal_rhos[name] = _rho([float(row[key]) for row in rows], error)[0]
    cross_view_cv_medians = {}
    cross_view_cv_error_rhos = {}
    for name in ("medium", "SH", "opacity", "geometry"):
        key = f"{name}_cross_view_cv"
        values = [float(row[key]) for row in rows]
        cross_view_cv_medians[name] = _finite_median(values)
        cross_view_cv_error_rhos[name] = _rho(values, error)[0]
    mean_cross_view_cv = [float(row["mean_cross_view_response_cv"]) for row in rows]
    return {
        "scene": rows[0]["scene"],
        "checkpoint": rows[0]["checkpoint"],
        "absolute_step": int(rows[0]["absolute_step"]),
        "gaussian_count": len(rows),
        "median_ambiguity_score": float(np.median(ambiguity)),
        "median_maximum_pairwise_overlap": float(np.median(overlaps)),
        "median_minimum_principal_angle": float(np.median(angles)),
        "obvious_parameter_overlap": obvious,
        "pair_median_overlaps": pair_medians,
        "pair_finite_fractions": pair_finite_fractions,
        "minimum_pair_finite_fraction": min(pair_finite_fractions.values()),
        "largest_median_overlap_pair": max(finite_pair_medians, key=finite_pair_medians.get),
        "rho_ambiguity_vs_heldout_error": rho,
        "pvalue_ambiguity_vs_heldout_error": pvalue,
        "rho_ambiguity_vs_heldout_PSNR_error": psnr_rho,
        "pvalue_ambiguity_vs_heldout_PSNR_error": psnr_pvalue,
        "PSNR_error_valid_count": psnr_count,
        "auroc_ambiguity_top20_error": VC._auroc(ambiguity, labels),
        "control_rhos": controls,
        "removal_error_alignment_rhos": removal_rhos,
        "cross_view_cv_medians": cross_view_cv_medians,
        "cross_view_cv_error_rhos": cross_view_cv_error_rhos,
        "median_mean_cross_view_response_cv": _finite_median(mean_cross_view_cv),
        "rho_mean_cross_view_response_cv_vs_heldout_error": _rho(mean_cross_view_cv, error)[0],
        "checkpoint_pass": checkpoint_pass,
        "ambiguity_positive_fraction": float(np.mean(np.asarray(ambiguity) > 0.0)),
        "finite_overlap_fraction": float(np.mean(np.isfinite(overlaps))),
        "finite_pairwise_overlap_fraction": float(np.mean(list(pair_finite_fractions.values()))),
    }, control_rows


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
        split = AME._camera_split(train_records, eval_records)
        all_gaussian_rows = []
        all_overlap_rows = []
        all_error_rows = []
        all_camera_rows = []
        all_control_rows = []
        checkpoint_rows = []
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
                or int(model.config.sh_degree) != 3
                or model.config.rasterize_mode != "classic"
                or model.config.medium_context_mode != "dir_xy_camera"
                or model.config.b_inf_mode != "tied"
                or model.config.infinite_water_enabled
            ):
                raise RuntimeError("locked OCMC-on RAOC-off configuration drift")
            state_before = CAUSAL._model_state_hash(model)
            projector_before = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
            support = VC._support_counts(model, train_records)
            requested = sample_count if sample_count is not None else (
                FINAL_SAMPLE_COUNT if step == FINAL_STEP else TEMPORAL_SAMPLE_COUNT
            )
            selected, sampling = _sample_gaussians(scene, step, support, requested)
            metric_rows, overlap_rows = _training_metrics(model, train_records, selected, support)
            for row in metric_rows:
                row.update({"scene": scene, "checkpoint": str(checkpoint), "absolute_step": step})
            for row in overlap_rows:
                row.update({"scene": scene, "checkpoint": str(checkpoint), "absolute_step": step})
            # No heldout GT has been accessed before this boundary.
            heldout, camera_rows = _heldout_error(scene, step, model, eval_records, selected, metric_rows)
            valid_rows = []
            for row in metric_rows:
                gaussian_id = int(row["gaussian_id"])
                if gaussian_id not in heldout:
                    continue
                row.update(heldout[gaussian_id])
                valid_rows.append(row)
            high_values = np.asarray([float(row["heldout_projected_residual"]) for row in valid_rows])
            high_threshold = float(np.quantile(high_values, 0.80))
            for row in valid_rows:
                error = float(row["heldout_projected_residual"])
                row["heldout_PSNR_error"] = 10.0 * math.log10(max(error, EPS))
                row["high_error_label_top20"] = bool(error >= high_threshold)
            summary, control_rows = _checkpoint_summary(valid_rows)
            state_after = CAUSAL._model_state_hash(model)
            projector_after = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
            if state_before != state_after or projector_before != projector_after:
                raise RuntimeError("frozen model or OCMC projector changed")
            sampling["heldout_visible_analysis_count"] = len(valid_rows)
            checkpoint_rows.append(
                {
                    **summary,
                    "sampling": sampling,
                    "checkpoint_sha256": _sha256(checkpoint),
                    "runtime_intrinsic_color_parameterization": str(model.config.intrinsic_color_parameterization),
                    "source_config_serialized_intrinsic_name": "sigmoid_sh",
                    "runtime_rasterize_mode": str(model.config.rasterize_mode),
                    "runtime_medium_context_mode": str(model.config.medium_context_mode),
                    "runtime_b_inf_mode": str(model.config.b_inf_mode),
                    "runtime_infinite_water_enabled": bool(model.config.infinite_water_enabled),
                    "ocmc_enable_flag": True,
                    "raoc_enable_flag": False,
                    "raoc_state_present": False,
                    "model_state_sha256_before": state_before,
                    "model_state_sha256_after": state_after,
                    "ocmc_projector_sha256_before": projector_before,
                    "ocmc_projector_sha256_after": projector_after,
                    "model_and_ocmc_unchanged": True,
                }
            )
            all_gaussian_rows.extend(valid_rows)
            all_overlap_rows.extend([row for row in overlap_rows if int(row["gaussian_id"]) in heldout])
            all_error_rows.extend(valid_rows)
            all_camera_rows.extend(camera_rows)
            all_control_rows.extend(control_rows)
            print(
                f"[{scene}] step {step}: n={len(valid_rows)} overlap={summary['median_maximum_pairwise_overlap']:.6f} "
                f"angle={summary['median_minimum_principal_angle']:.3f} rho={summary['rho_ambiguity_vs_heldout_error']:.6f} "
                f"pass={summary['checkpoint_pass']}",
                flush=True,
            )
            del payload, support, selected, metric_rows, overlap_rows, heldout, valid_rows
            gc.collect()
            torch.cuda.empty_cache()
        result = {
            "experiment": EXPERIMENT,
            "scene": scene,
            "runtime": runtime,
            "camera_split": split,
            "checkpoint_rows": checkpoint_rows,
            "per_gaussian_rows": len(all_gaussian_rows),
            "overlap_rows": len(all_overlap_rows),
            "error_rows": len(all_error_rows),
            "camera_rows": len(all_camera_rows),
            "control_rows": len(all_control_rows),
            "elapsed_seconds": time.perf_counter() - started,
            "frozen_forward_only": True,
            "backward_calls": 0,
            "jvp_calls": 0,
            "vjp_calls": 0,
            "optimizer_step_calls": 0,
            "checkpoint_writes": 0,
            "render_writes": 0,
        }
        suffix = "" if tuple(steps) == STEPS and sample_count is None else "_smoke"
        _write_csv(scene_dir / f"per_gaussian_ambiguity{suffix}.csv", all_gaussian_rows)
        _write_csv(scene_dir / f"parameter_overlap{suffix}.csv", all_overlap_rows)
        _write_csv(scene_dir / f"error_alignment{suffix}.csv", all_error_rows)
        _write_csv(scene_dir / f"camera_alignment{suffix}.csv", all_camera_rows)
        _write_csv(scene_dir / f"control_analysis{suffix}.csv", all_control_rows)
        _write_json(scene_dir / f"worker_summary{suffix}.json", result)
        return result
    finally:
        FORMAL._release(branch)


def _coerce_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    text = {"scene", "checkpoint", "pair", "maximum_overlap_pair", "camera_id", "control_set", "controls"}
    boolean = {
        "gaussian_identity_persistent_across_checkpoints",
        "metric_uses_heldout_or_gt",
        "gt_used_after_metric_for_error_only",
        "gt_used_for_error_outcome_only",
        "high_error_label_top20",
    }
    integer = {
        "gaussian_id",
        "absolute_step",
        "support_count",
        "observed_train_views",
        "heldout_visible_views",
        "sampled_visible_gaussian_count",
        "valid_count",
        "rank_left",
        "rank_right",
        "observation_rows",
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
        pass_count = sum(bool(row["checkpoint_pass"]) for row in trajectory)
        overlap_count = sum(bool(row["obvious_parameter_overlap"]) for row in trajectory)
        temporal_stable = pass_count >= 3
        quality = bool(
            len(trajectory) == len(STEPS)
            and all(int(row["gaussian_count"]) >= MIN_VALID_SAMPLE for row in trajectory)
            and all(float(row["ambiguity_positive_fraction"]) >= 0.90 for row in trajectory)
            and all(float(row["finite_overlap_fraction"]) >= 0.99 for row in trajectory)
            and all(math.isfinite(float(row["rho_ambiguity_vs_heldout_error"])) for row in trajectory)
        )
        scene_pass = bool(final["checkpoint_pass"] and temporal_stable and quality)
        partial = bool(
            quality
            and overlap_count >= 3
            and sum(float(row["rho_ambiguity_vs_heldout_error"]) > 0.0 for row in trajectory) >= 3
        )
        scene_rows.append(
            {
                "scene": scene,
                "final_median_maximum_overlap": float(final["median_maximum_pairwise_overlap"]),
                "final_median_minimum_angle": float(final["median_minimum_principal_angle"]),
                "final_largest_overlap_pair": final["largest_median_overlap_pair"],
                "final_pair_median_overlaps": final["pair_median_overlaps"],
                "final_pair_finite_fractions": final["pair_finite_fractions"],
                "final_rho_ambiguity_vs_error": float(final["rho_ambiguity_vs_heldout_error"]),
                "final_rho_ambiguity_vs_PSNR_error": float(
                    final["rho_ambiguity_vs_heldout_PSNR_error"]
                ),
                "final_auroc": float(final["auroc_ambiguity_top20_error"]),
                "depth_tau_transmission_control_rho": float(final["control_rhos"]["depth_tau_transmission_joint"]),
                "ocmc_control_rho": float(final["control_rhos"]["ocmc_independence_joint"]),
                "all_required_control_rho": float(final["control_rhos"]["all_required_joint"]),
                "removal_error_alignment_rhos": final["removal_error_alignment_rhos"],
                "cross_view_cv_medians": final["cross_view_cv_medians"],
                "cross_view_cv_error_rhos": final["cross_view_cv_error_rhos"],
                "median_mean_cross_view_response_cv": float(final["median_mean_cross_view_response_cv"]),
                "rho_mean_cross_view_response_cv_vs_heldout_error": float(
                    final["rho_mean_cross_view_response_cv_vs_heldout_error"]
                ),
                "checkpoint_pass_count": pass_count,
                "overlap_checkpoint_count": overlap_count,
                "temporal_stable": temporal_stable,
                "quality_pass": quality,
                "partial_candidate": partial,
                "scene_supported": scene_pass,
            }
        )
        quality_rows.append(
            {
                "scene": scene,
                "checkpoint_count": len(trajectory),
                "minimum_valid_gaussians": min(int(row["gaussian_count"]) for row in trajectory),
                "minimum_ambiguity_positive_fraction": min(float(row["ambiguity_positive_fraction"]) for row in trajectory),
                "minimum_finite_overlap_fraction": min(float(row["finite_overlap_fraction"]) for row in trajectory),
                "minimum_pair_finite_fraction": min(float(row["minimum_pair_finite_fraction"]) for row in trajectory),
                "quality_pass": quality,
            }
        )
        for row in trajectory:
            temporal_rows.append(
                {
                    "scene": scene,
                    "checkpoint": row["checkpoint"],
                    "absolute_step": int(row["absolute_step"]),
                    "gaussian_count": int(row["gaussian_count"]),
                    "median_ambiguity_score": float(row["median_ambiguity_score"]),
                    "median_maximum_pairwise_overlap": float(row["median_maximum_pairwise_overlap"]),
                    "median_minimum_principal_angle": float(row["median_minimum_principal_angle"]),
                    "largest_median_overlap_pair": row["largest_median_overlap_pair"],
                    "rho_ambiguity_vs_error": float(row["rho_ambiguity_vs_heldout_error"]),
                    "rho_ambiguity_vs_PSNR_error": float(
                        row["rho_ambiguity_vs_heldout_PSNR_error"]
                    ),
                    "rho_after_depth_tau_transmission_control": float(row["control_rhos"]["depth_tau_transmission_joint"]),
                    "rho_after_ocmc_control": float(row["control_rhos"]["ocmc_independence_joint"]),
                    "rho_after_all_required_controls": float(row["control_rhos"]["all_required_joint"]),
                    "obvious_parameter_overlap": bool(row["obvious_parameter_overlap"]),
                    "checkpoint_pass": bool(row["checkpoint_pass"]),
                    "temporal_scope": "distribution_level_no_lineage",
                }
            )
    quality_pass = all(row["quality_pass"] for row in quality_rows)
    supported_count = sum(row["scene_supported"] for row in scene_rows)
    partial_count = sum(row["partial_candidate"] for row in scene_rows)
    if not quality_pass:
        label = "DATA_LIMITED"
    elif supported_count >= 3:
        label = "SUPPORTED"
    elif supported_count == 2 or partial_count >= 2:
        label = "TENTATIVE"
    else:
        label = "NOT_SUPPORTED"
    next_task = {
        "SUPPORTED": "GAUSSIAN_IDENTIFIABILITY_MODULE_DESIGN",
        "TENTATIVE": "INDEPENDENT_IDENTIFIABILITY_REPLICATION_BEFORE_MODULE_DESIGN",
        "NOT_SUPPORTED": "CLOSE_GAUSSIAN_IDENTIFIABILITY_AND_RETURN_TO_FAILURE_HYPOTHESIS_SELECTION",
        "DATA_LIMITED": "RESOLVE_IDENTIFIABILITY_AUDIT_DATA_QUALITY_ONLY",
    }[label]
    return {
        "experiment": EXPERIMENT,
        "classification": label,
        "supported_scene_count": supported_count,
        "partial_candidate_scene_count": partial_count,
        "required_supported_scene_count": 3,
        "quality_gate_passed": quality_pass,
        "module_design_authorized": label == "SUPPORTED",
        "direction_closed": label == "NOT_SUPPORTED",
        "next_unique_task": next_task,
        "scene_rows": scene_rows,
        "quality_rows": quality_rows,
    }, temporal_rows


def _camera_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    final = [row for row in rows if int(row["absolute_step"]) == FINAL_STEP]
    scene_rows = []
    pooled_x = []
    pooled_y = []
    for scene in SCENES:
        subset = [row for row in final if row["scene"] == scene]
        x = [float(row["camera_ambiguity_mean"]) for row in subset]
        y = [float(row["camera_MSE"]) for row in subset]
        rho, pvalue, count = _rho(x, y, minimum_count=3)
        scene_rows.append(
            {
                "scene": scene,
                "heldout_camera_count": len(subset),
                "rho_camera_ambiguity_vs_MSE": rho,
                "pvalue": pvalue,
                "small_n_descriptive_only": True,
            }
        )
        pooled_x.extend((scipy.stats.rankdata(x) / max(len(x), 1)).tolist())
        pooled_y.extend((scipy.stats.rankdata(y) / max(len(y), 1)).tolist())
    return {
        "scene_rows": scene_rows,
        "pooled_camera_count": len(pooled_x),
        "pooled_within_scene_rank_rho": _rho(pooled_x, pooled_y)[0],
        "pooling_rule": "Spearman over within-scene camera percentile ranks",
    }


def _integrity_after(preflight_result: Mapping[str, Any]) -> Dict[str, Any]:
    if _sha256(Path(__file__).resolve()) != preflight_result["repo"]["audit_script_sha256"]:
        raise RuntimeError("audit script changed during execution")
    after = {relative: _sha256(REPO_ROOT / relative) for relative in PROTECTED_HASHES}
    if after != preflight_result["repo"]["protected_hashes"]:
        raise RuntimeError("protected source changed during execution")
    for row in preflight_result["checkpoint_rows"]:
        if _sha256(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed during execution: {row['checkpoint']}")
    return {
        "audit_script_sha256_before_after_match": True,
        "protected_source_hashes_before_after_match": True,
        "checkpoint_hashes_before_after_match": True,
        "checkpoint_count": len(preflight_result["checkpoint_rows"]),
        "backward_calls": 0,
        "jvp_calls": 0,
        "vjp_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
    }


def _research_note(summary: Mapping[str, Any]) -> None:
    classification = summary["classification"]
    lines = [
        "# Gaussian Parameter Identifiability Audit Under OCMC",
        "",
        "Date: 2026-09-01",
        f"Experiment: `{EXPERIMENT}`",
        f"Classification: `{classification['classification']}`",
        "",
        "## Hypothesis",
        "",
        "Medium, bounded-SH appearance, opacity, and screen-space geometry responses may occupy overlapping local RGB observation subspaces. Such overlap is a representation ambiguity only if it is substantial, predicts heldout error after controls, remains independent of OCMC, and recurs over checkpoint populations.",
        "",
        "## Frozen Protocol",
        "",
        "All 20 registered C0 checkpoints use OCMC on, RAOC off, runtime `bounded_sh3`, SH degree 3, `dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=false`, and classic rasterization. Historical YAML uses the old `sigmoid_sh` name, which the protected setup normalizes to `bounded_sh3`. The audit uses detached analytic forward sensitivities only: zero backward, JVP, VJP, optimizer steps, checkpoint writes, and render writes.",
        "",
        "Sampling and every ambiguity/removal metric use training visibility only and are frozen before heldout GT access. Heldout GT is used only as the projected-footprint error outcome.",
        "",
        "## Metric Definition",
        "",
        "For each Gaussian, local RGB responses are evaluated at five fixed footprint offsets over all visible training cameras and stacked into a common observation matrix. The isolated signal relative to tied pure-medium background is `alpha*(exp(-beta_D*d)*c_SH-exp(-beta_B*d)*B_inf)`. Analytic response groups are physical 9-D medium, the full 48-D bounded SH3 appearance coefficient group (including DC), raw-opacity logit, and 2-D screen-center displacement with conic/depth fixed. The separate SH removal proxy replaces full bounded RGB with bounded DC RGB and therefore removes only the non-DC contribution.",
        "",
        "Each response matrix is truncated to the effective left-singular subspace retaining 99.9% energy with relative singular values at least `1e-4`. Normalized overlap is `||U_A^T U_B||_F/sqrt(min(rank_A,rank_B))`; principal angle is the minimum angle. Obvious overlap requires population median maximum overlap at least `0.8` and median minimum angle at most `20 degrees`. Ambiguity score is maximum overlap times total response sensitivity. PSNR error is `10*log10(MSE)`, so its rank correlation is expected to equal the heldout-MSE rank correlation and is recorded explicitly in the formal outputs.",
        "",
        "## Final Results",
        "",
        "| Scene | max overlap median | min angle | largest pair | rho(A,MSE) | rho(A,PSNR error) | AUROC | depth/tau/T ctrl | OCMC ctrl | temporal | pass |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in classification["scene_rows"]:
        lines.append(
            f"| {row['scene']} | {row['final_median_maximum_overlap']:.6f} | {row['final_median_minimum_angle']:.3f} | {row['final_largest_overlap_pair']} | {row['final_rho_ambiguity_vs_error']:.6f} | {row['final_rho_ambiguity_vs_PSNR_error']:.6f} | {row['final_auroc']:.6f} | {row['depth_tau_transmission_control_rho']:.6f} | {row['ocmc_control_rho']:.6f} | {row['checkpoint_pass_count']}/5 | {'yes' if row['scene_supported'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Overlap",
            "",
            "| Scene | Medium-SH | Medium-Opacity | Medium-Geometry | SH-Opacity | SH-Geometry | Opacity-Geometry |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in classification["scene_rows"]:
        pair = row["final_pair_median_overlaps"]
        lines.append(
            f"| {row['scene']} | {pair['Medium-SH']:.6f} | {pair['Medium-Opacity']:.6f} | {pair['Medium-Geometry']:.6f} | {pair['SH-Opacity']:.6f} | {pair['SH-Geometry']:.6f} | {pair['Opacity-Geometry']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Counterfactual Removal Alignment",
            "",
            "| Scene | remove medium | remove SH | remove opacity | remove geometry |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in classification["scene_rows"]:
        removal = row["removal_error_alignment_rhos"]
        lines.append(
            f"| {row['scene']} | {removal['medium']:.6f} | {removal['SH']:.6f} | {removal['opacity']:.6f} | {removal['geometry']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Removal/error alignment uses absolute local removal-delta RGB RMS. Baseline-relative sensitivity is also retained per Gaussian, but opacity alignment cannot use it because removing opacity is identically the full isolated signal and therefore has relative magnitude one. These are frozen read-only counterfactual proxies, not retraining, not physical component ground truth, and not causal error attribution.",
            "",
            "## Cross-view Stability",
            "",
            "Response stability is the coefficient of variation (CV) of each group response RMS over visible training cameras. Lower CV means more stable training-view sensitivity magnitude. Its correlation with heldout error is descriptive; no post-hoc CV threshold changes the classification gate.",
            "",
            "| Scene | medium CV | SH CV | opacity CV | geometry CV | mean-CV/error rho |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in classification["scene_rows"]:
        cv = row["cross_view_cv_medians"]
        lines.append(
            f"| {row['scene']} | {cv['medium']:.6f} | {cv['SH']:.6f} | {cv['opacity']:.6f} | {cv['geometry']:.6f} | {row['rho_mean_cross_view_response_cv_vs_heldout_error']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The stacked subspace audit asks whether parameter explanations overlap across the training-view observation set; CV separately tests whether their response magnitudes are stable across those views. Thus a training-view-stable but novel-view-ambiguous pattern requires low response CV together with positive ambiguity/error alignment, rather than overlap alone.",
            "",
            "## Temporal Stability",
            "",
            "Checkpoint populations have no persistent Gaussian lineage. Temporal recurrence is distribution-level only; array index and nearest-geometry identity matching were not used.",
            "",
            "| Scene | 5k | 8k | 10k | 13k | 14999 | checkpoint passes |",
            "|---|:---:|:---:|:---:|:---:|:---:|---:|",
        ]
    )
    for scene in SCENES:
        rows = [row for row in summary["temporal_stability"] if row["scene"] == scene]
        flags = ["yes" if row["checkpoint_pass"] else "no" for row in rows]
        lines.append(f"| {scene} | {' | '.join(flags)} | {sum(row['checkpoint_pass'] for row in rows)}/5 |")
    lines.extend(
        [
            "",
            "## OCMC Independence",
            "",
            "OCMC independence jointly rank-residualizes ambiguity and heldout error against projected OCMC active magnitude and suppressed medium residual. The all-required control additionally includes depth, tau, transmission, accumulation, opacity, and footprint. A scene cannot pass based on overlap alone.",
            "",
            "## Limitations",
            "",
            "This is local first-order identifiability, not global optimization equivalence. The isolated-Gaussian observation ignores other-Gaussian occlusion, uses an isotropic radius-based footprint for legal screen-displacement sensitivity, and associates heldout error through overlapping projected boxes. Medium parameters are local physical activations rather than all MLP weights. Geometry covers screen translation only, not full 3-D position/scale/rotation. Pair medians use finite overlaps only; zero-rank groups remain undefined and their pairwise finite fractions are retained in `final_summary.json`. No true parameter or medium labels exist.",
            "",
            "## Classification",
            "",
            f"The formal result is `{classification['classification']}` with {classification['supported_scene_count']}/4 supported scenes and {classification['partial_candidate_scene_count']}/4 partial candidate scenes. Module design authorization is `{str(classification['module_design_authorized']).lower()}`.",
            "",
        ]
    )
    if classification["classification"] == "SUPPORTED":
        lines.append("The next and only authorized task is a separate Gaussian-identifiability module-design phase. No module was implemented here.")
    elif classification["classification"] == "TENTATIVE":
        lines.append("Gaussian identifiability remains a candidate mechanism only. The next task is an independent replication; module design is not authorized.")
    elif classification["classification"] == "NOT_SUPPORTED":
        lines.append("Gaussian parameter identifiability ambiguity is not supported as a novel-view failure mechanism under this protocol. Close this direction and return to failure-hypothesis selection; do not design a module from this audit.")
    else:
        lines.append("The audit is data-limited. Resolve only the stated quality failure before drawing a mechanism conclusion.")
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"Analyzed {summary['per_gaussian_rows']} Gaussian-checkpoint rows, {summary['parameter_overlap_rows']} pairwise-overlap rows, and {summary['per_camera_rows']} heldout camera-checkpoint rows. All 20 checkpoint and protected source hashes matched before and after execution. Backward, JVP, VJP, optimizer-step, checkpoint-write, and render-write counts were zero.",
            "",
        ]
    )
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def aggregate() -> Dict[str, Any]:
    preflight_result = _read_json(OUTPUT_ROOT / "preflight.json")
    worker_summaries = [_read_json(OUTPUT_ROOT / "workers" / scene / "worker_summary.json") for scene in SCENES]
    if not all(
        row["backward_calls"] == 0
        and row["jvp_calls"] == 0
        and row["vjp_calls"] == 0
        and row["optimizer_step_calls"] == 0
        and row["checkpoint_writes"] == 0
        and row["render_writes"] == 0
        and len(row["checkpoint_rows"]) == len(STEPS)
        and all(item["model_and_ocmc_unchanged"] for item in row["checkpoint_rows"])
        for row in worker_summaries
    ):
        raise RuntimeError("worker integrity gate failed")
    gaussian_raw = []
    overlap_raw = []
    error_raw = []
    camera_raw = []
    control_raw = []
    checkpoint_rows = []
    for scene, worker_summary in zip(SCENES, worker_summaries):
        scene_dir = OUTPUT_ROOT / "workers" / scene
        gaussian_raw.extend(_read_csv(scene_dir / "per_gaussian_ambiguity.csv"))
        overlap_raw.extend(_read_csv(scene_dir / "parameter_overlap.csv"))
        error_raw.extend(_read_csv(scene_dir / "error_alignment.csv"))
        camera_raw.extend(_read_csv(scene_dir / "camera_alignment.csv"))
        control_raw.extend(_read_csv(scene_dir / "control_analysis.csv"))
        checkpoint_rows.extend(worker_summary["checkpoint_rows"])
    gaussian_rows = _coerce_rows(gaussian_raw)
    overlap_rows = _coerce_rows(overlap_raw)
    error_rows = _coerce_rows(error_raw)
    camera_rows = _coerce_rows(camera_raw)
    control_rows = _coerce_rows(control_raw)
    classification, temporal_rows = _classify(checkpoint_rows)
    camera_analysis = _camera_analysis(camera_rows)
    integrity = _integrity_after(preflight_result)
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "final_checkpoint_rows": [row for row in checkpoint_rows if int(row["absolute_step"]) == FINAL_STEP],
        "all_checkpoint_rows": checkpoint_rows,
        "temporal_stability": temporal_rows,
        "camera_analysis": camera_analysis,
        "worker_summaries": worker_summaries,
        "per_gaussian_rows": len(gaussian_rows),
        "parameter_overlap_rows": len(overlap_rows),
        "error_alignment_rows": len(error_rows),
        "per_camera_rows": len(camera_rows),
        "control_rows": len(control_rows),
        "integrity": integrity,
        "frozen_forward_only": True,
        "ocmc_enabled": True,
        "raoc_enabled": False,
        "module_design_started": False,
        "backward_calls": 0,
        "jvp_calls": 0,
        "vjp_calls": 0,
        "optimizer_step_calls": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
    }
    _write_csv(OUTPUT_ROOT / "parameter_overlap.csv", overlap_rows)
    _write_csv(OUTPUT_ROOT / "per_gaussian_ambiguity.csv", gaussian_rows)
    _write_csv(OUTPUT_ROOT / "error_alignment.csv", error_rows)
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
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
        processes.append((scene, gpu, process, handle))
    failures = []
    for scene, gpu, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append({"scene": scene, "gpu": gpu, "exit_code": code, "log": str(logs / f"{scene}.log")})
    if failures:
        _write_json(OUTPUT_ROOT / "worker_failures.json", {"rows": failures})
        raise RuntimeError(f"identifiability audit workers failed: {failures}")
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
        if args.sample_count is not None and args.sample_count < 12:
            parser.error("--sample-count must be at least 12")
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
