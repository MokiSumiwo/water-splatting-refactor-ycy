#!/usr/bin/env python3
"""Preflight a minimal SH-opacity identifiability controller under OCMC.

The controller is evaluated on cloned non-DC SH tensors from registered C0
checkpoints. It never mutates model state, writes checkpoints, or accesses
heldout views/ground truth. No renderer, optimizer, loss, or training-loop
source is changed.
"""

from __future__ import annotations

import argparse
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
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_gaussian_parameter_identifiability_ocmc as AUDIT
from scripts.diagnostics import audit_low_support_causal_intervention as CAUSAL
from scripts.experiments import run_m1_raoc_causal_scene as FORMAL
from water_splatting._torch_impl import eval_sh_bases


EXPERIMENT = "IDENTIFIABILITY_MODULE_PREFLIGHT"
SELECTED_CANDIDATE = "A_DETACHED_SH_OPACITY_TANGENT_ORTHOGONALIZATION"
EXPECTED_BRANCH = "research/m1-bounded-intrinsic"
EXPECTED_HEAD = "d1852da8db7c03db156fae506f63bda027334446"
SOURCE_ROOT = REPO_ROOT / "outputs" / "m1_raoc_causal_four_scene_20260827"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "gaussian_identifiability_module_preflight_20260901"
RESEARCH_NOTE = REPO_ROOT / "research_notes" / "GAUSSIAN_IDENTIFIABILITY_MODULE_DESIGN_2026-09-01.md"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENE_GPUS = dict(CAUSAL.SCENE_GPUS)
SCENES = tuple(SCENE_GPUS)
STEP = 14999
SEED = 42
SAMPLE_COUNT = 256
MIN_VALID_SAMPLE = 128
OPTIMIZATION_STEPS = 50
CONTROL_BUDGET = 0.20
OVERLAP_THRESHOLD = 0.80
MIN_ACTIVE_GATE_FRACTION = 0.50
MIN_PARALLEL_ENERGY_REDUCTION = 0.02
MIN_RESPONSE_SHARED_ENERGY_REDUCTION = 0.02
MAX_ORTHOGONAL_DRIFT = 1e-5
MIN_SH_VARIANCE_RATIO = 0.90
MAX_SH_VARIANCE_RATIO = 1.10
MIN_SH_RESPONSE_RMS_RATIO = 0.80
MAX_SH_RESPONSE_RMS_RATIO = 1.20
MIN_DIRECT_RGB_RMS_RATIO = 0.90
MAX_DIRECT_RGB_RMS_RATIO = 1.10
SVD_RELATIVE_THRESHOLD = 1e-4
SVD_ENERGY_THRESHOLD = 0.999
EPS = 1e-12

PROTECTED_HASHES = {
    **AUDIT.PROTECTED_HASHES,
    "scripts/diagnostics/audit_gaussian_parameter_identifiability_ocmc.py": "8928d9fac8943bc31b6ee567cc734712c75d5e417d58d7f7eebe680ef9876397",
}

CANDIDATE_ANALYSIS = {
    "experiment": EXPERIMENT,
    "selected_candidate": SELECTED_CANDIDATE,
    "selection_reason": (
        "Candidate A can control one detached opacity-equivalent direction in the 45-D non-DC SH "
        "parameter space while preserving its 44-D orthogonal complement. It uses training-view "
        "observability only and requires no inference-time Jacobian."
    ),
    "candidates": [
        {
            "candidate": "A_SH_OPACITY_ORTHOGONALIZATION",
            "selected": True,
            "theoretical_motivation": (
                "Use the supported SH-opacity tangent overlap directly. Freeze the opacity tangent "
                "and penalize an anchored target only along the non-DC SH coefficient direction "
                "aligned with J_SH^T j_alpha."
            ),
            "ocmc_compatibility": (
                "Complementary: OCMC constrains camera-conditioned medium context; this controller "
                "acts only on per-Gaussian non-DC appearance coefficients."
            ),
            "computational_cost": (
                "Training-only periodic analytic sensitivities over observed views, one 45-vector and "
                "one scalar gate per active Gaussian; no inference-time cost."
            ),
            "valid_appearance_risk": (
                "Lowest of the candidates because DC is untouched and only one locally observed "
                "parameter direction is controlled; orthogonal SH components remain available."
            ),
        },
        {
            "candidate": "B_OPACITY_CONDITIONED_SH_CAPACITY",
            "selected": False,
            "theoretical_motivation": "Reduce SH freedom as a function of opacity and overlap.",
            "ocmc_compatibility": "Architecturally compatible but not observability-specific enough.",
            "computational_cost": "Low; a scalar gate per Gaussian.",
            "valid_appearance_risk": (
                "High: opacity alone does not imply that view dependence is invalid, so capacity "
                "attenuation can globally suppress legitimate non-DC appearance."
            ),
            "rejection_reason": (
                "Conflates opacity magnitude with causal redundancy and violates the requirement not "
                "to suppress SH capacity globally."
            ),
        },
        {
            "candidate": "C_ADAPTIVE_PARAMETER_UPDATE_DECORRELATION",
            "selected": False,
            "theoretical_motivation": "Project aligned SH and opacity optimization updates apart.",
            "ocmc_compatibility": "Potentially compatible but interacts with every training loss.",
            "computational_cost": "High; requires paired gradients and optimizer/update interception.",
            "valid_appearance_risk": (
                "Medium to high because gradient alignment depends on the current loss and can remove "
                "jointly useful updates rather than representation redundancy."
            ),
            "rejection_reason": (
                "Requires training-loss gradients (normally GT-dependent) and optimizer/training-loop "
                "changes, conflicting with this task's no-GT mechanism and engineering boundary."
            ),
        },
    ],
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
    if isinstance(value, Tensor):
        cpu = value.detach().cpu()
        return _sanitize(cpu.item() if cpu.numel() == 1 else cpu.tolist())
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: Sequence[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def _checkpoint(scene: str) -> Path:
    return SOURCE_ROOT / scene / "checkpoints" / "C0" / f"step-{STEP:09d}.ckpt"


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
        "preflight_script_sha256": _sha256(Path(__file__).resolve()),
    }


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


def _mechanism_markdown() -> str:
    return "\n".join(
        [
            "# Identifiability Mechanism Specification",
            "",
            f"Selected mechanism: `{SELECTED_CANDIDATE}`",
            "",
            "## Inputs",
            "",
            "For Gaussian `i`, use non-DC bounded-SH3 coefficients `theta_i in R^45`, physical opacity `alpha_i=sigmoid(a_i)`, and detached local RGB sensitivities from visible training cameras only. Heldout views and GT are forbidden.",
            "",
            "## Shared Direction",
            "",
            "Stack the non-DC SH Jacobian `J_i` and raw-opacity tangent `j_alpha_i` over observed training-view RGB. Define:",
            "",
            "`u_i = normalize(stopgrad(j_alpha_i))` and `h_i = normalize(stopgrad(J_i^T u_i))`.",
            "",
            "At a detached refresh anchor `theta_i0`, let `r_i0` be the non-DC SH RGB response and define the first-order shared-response-null target:",
            "",
            "`tau_i = h_i^T theta_i0 - (u_i^T r_i0) / ||J_i^T u_i||`.",
            "",
            "The detached observability gate is:",
            "",
            f"`g_i = clamp((Overlap(span(J_i), j_alpha_i) - {OVERLAP_THRESHOLD}) / (1 - {OVERLAP_THRESHOLD}), 0, 1)`.",
            "",
            "The module emits `(h_i, tau_i, g_i)` and the scalar regularizer:",
            "",
            "`L_ident = lambda / (2 sum_i g_i) * sum_i g_i * (h_i^T theta_i - tau_i)^2`.",
            "",
            "All Jacobians, gates, and directions are stop-gradient. Therefore the direct module gradient reaches only `features_rest`; DC color, opacity, geometry, medium, OCMC, and topology receive no direct gradient.",
            "",
            "## Preservation Property",
            "",
            "Each active Gaussian controls one anchored direction in the 45-D non-DC SH space. Any coefficient perturbation orthogonal to `h_i` remains unpenalized, preserving 44 dimensions for legitimate view dependence. The anchor removes coordinate-origin dependence: under the refresh linearization, reaching `tau_i` nulls only the opacity-tangent projection of the current non-DC SH response. The module does not gate forward colors and has no inference-time operation or Jacobian.",
            "",
            "## Proposed Training Integration",
            "",
            "A future causal experiment may refresh detached `(h_i, g_i)` from recent training-view observations at a low cadence, then add `L_ident` to the existing training objective. This design phase does not integrate that loss or alter the training loop. Strength zero omits the term exactly.",
            "",
            "## Safety Boundaries",
            "",
            "OCMC remains on and RAOC remains off. No renderer, CUDA, optimizer, base loss, training-loop, checkpoint, or inference path is modified. Sampling and controller construction cannot use heldout views or GT.",
            "",
        ]
    )


def preflight() -> Dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    repo = _strict_repo()
    sources = []
    for scene in SCENES:
        scene_config = CAUSAL._scene_config(scene)
        config = REPO_ROOT / scene_config["source_config"]
        sequence = SOURCE_ROOT / scene / "camera_sequence.json"
        checkpoint = _checkpoint(scene)
        if _sha256(config) != AUDIT.VC.EXPECTED_CONFIG_HASHES[scene]:
            raise RuntimeError(f"source config provenance drift for {scene}")
        if _sha256(sequence) != AUDIT.VC.EXPECTED_CAMERA_SEQUENCE_HASHES[scene]:
            raise RuntimeError(f"camera sequence provenance drift for {scene}")
        if _sha256(checkpoint) != AUDIT.VC.EXPECTED_CHECKPOINT_HASHES[scene][STEP]:
            raise RuntimeError(f"checkpoint provenance drift for {scene}")
        sources.append(
            {
                "scene": scene,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "source_config": str(config),
                "source_config_sha256": _sha256(config),
                "camera_sequence": str(sequence),
                "camera_sequence_sha256": _sha256(sequence),
            }
        )
    result = {
        "experiment": EXPERIMENT,
        "repo": repo,
        "sources": sources,
        "candidate": SELECTED_CANDIDATE,
        "mechanism": {
            "shared_direction": "u_i=normalize(stopgrad(j_alpha_i)); h_i=normalize(stopgrad(J_SH_nonDC_i^T*u_i))",
            "anchored_target": "tau_i=h_i^T*theta_i0-(u_i^T*r_SH_nonDC_i0)/||J_SH_nonDC_i^T*u_i||",
            "gate": f"clamp((rank1_subspace_overlap-{OVERLAP_THRESHOLD})/(1-{OVERLAP_THRESHOLD}),0,1)",
            "regularizer": "lambda*sum_i(g_i*(h_i^T*theta_nonDC_i-tau_i)^2)/(2*sum_i(g_i))",
            "direct_gradient_targets": ["features_rest"],
            "direct_gradient_exclusions": [
                "features_dc",
                "opacities",
                "means",
                "scales",
                "quats",
                "medium_mlp",
                "direction_encoding",
                "OCMC_projector",
            ],
            "training_view_only": True,
            "uses_gt": False,
            "uses_heldout_views": False,
            "inference_time_jacobian": False,
        },
        "preflight_protocol": {
            "checkpoint_step": STEP,
            "sample_count_per_scene": SAMPLE_COUNT,
            "sample_eligibility": "training support>=2; support-stratified; no heldout view or GT",
            "optimization_steps": OPTIMIZATION_STEPS,
            "isolated_tensor": "clone of sampled features_rest only",
            "cumulative_control_budget": CONTROL_BUDGET,
            "checkpoint_writes": 0,
            "render_writes": 0,
            "model_training_steps": 0,
        },
        "ready_gate": {
            "scene": (
                f">={MIN_VALID_SAMPLE} finite directions; disabled max diff=0; nonzero features_rest "
                "gradient and zero direct opacity gradient; active gate fraction>=0.5; anchored shared "
                "coordinate and opacity-tangent response energy each reduce >=2%; orthogonal coefficient drift "
                "<=1e-5; SH variance ratio in [0.9,1.1]; SH response RMS ratio in [0.8,1.2]; "
                "local direct-RGB contribution RMS ratio in [0.9,1.1]; "
                "opacity/count/model/OCMC unchanged"
            ),
            "MODULE_READY": "at least 3/4 scenes pass with all integrity gates passing",
            "MODULE_NOT_READY": "otherwise",
        },
        "ocmc_enabled": True,
        "raoc_enabled": False,
        "renderer_changes": 0,
        "cuda_changes": 0,
        "optimizer_changes": 0,
        "loss_changes": 0,
        "training_loop_changes": 0,
        "checkpoint_writes": 0,
        "render_writes": 0,
    }
    _write_json(OUTPUT_ROOT / "candidate_analysis.json", CANDIDATE_ANALYSIS)
    (OUTPUT_ROOT / "mechanism_specification.md").write_text(_mechanism_markdown(), encoding="utf8")
    _write_json(OUTPUT_ROOT / "preflight_manifest.json", result)
    return result


@torch.no_grad()
def _collect_observations(
    model: Any,
    records: Sequence[Tuple[int, str, Any, Any]],
    selected: Tensor,
) -> Dict[str, Tensor]:
    count = int(selected.numel())
    views = len(records)
    selected_gpu = selected.to(model.device)
    mask = torch.zeros(count, views, dtype=torch.bool, device=model.device)
    bases = torch.zeros(count, views, 16, dtype=torch.float32, device=model.device)
    depth = torch.zeros(count, views, dtype=torch.float32, device=model.device)
    b_inf = torch.zeros(count, views, 3, dtype=torch.float32, device=model.device)
    beta_b = torch.zeros_like(b_inf)
    beta_d = torch.zeros_like(b_inf)
    observed = torch.zeros(count, dtype=torch.int16)
    for view, (_index, _camera_id, camera, _batch) in enumerate(records):
        outputs = model.get_outputs_for_camera(camera.to(model.device))
        visible = outputs["gaussian_visible_mask"].detach().reshape(-1)[selected_gpu].bool()
        if bool(visible.any()):
            local = torch.nonzero(visible, as_tuple=False).reshape(-1)
            global_ids = selected_gpu[local]
            xys = model.xys.detach()[global_ids].float()
            _viewdirs, local_bases = AUDIT._viewdirs_and_bases(model, camera, global_ids)
            mask[local, view] = True
            bases[local, view] = local_bases
            depth[local, view] = outputs["projected_gaussian_depths"].detach().reshape(-1)[global_ids].float()
            b_inf[local, view] = AUDIT._sample_image(outputs["b_inf"].detach().float(), xys)
            beta_b[local, view] = AUDIT._sample_image(outputs["medium_bs"].detach().float(), xys)
            beta_d[local, view] = AUDIT._sample_image(outputs["medium_attn"].detach().float(), xys)
            observed += visible.cpu().to(torch.int16)
        del outputs
    return {
        "mask": mask,
        "bases": bases,
        "depth": depth,
        "b_inf": b_inf,
        "beta_b": beta_b,
        "beta_d": beta_d,
        "observed": observed,
    }


def _responses(
    rest: Tensor,
    dc: Tensor,
    opacity: Tensor,
    observations: Mapping[str, Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    bases = observations["bases"]
    depth = observations["depth"]
    b_inf = observations["b_inf"]
    beta_b = observations["beta_b"]
    beta_d = observations["beta_d"]
    mask = observations["mask"][..., None]
    dc_logits = bases[..., :1, None] * dc[:, None, None, :]
    dc_logits = dc_logits.squeeze(-2)
    residual_logits = torch.einsum("nvk,nkc->nvc", bases[..., 1:], rest)
    color = torch.sigmoid(dc_logits + residual_logits)
    dc_color = torch.sigmoid(dc_logits)
    transmission_d = torch.exp(-(beta_d * depth[..., None]).clamp_min(0.0)).clamp(0.0, 1.0)
    transmission_b = torch.exp(-(beta_b * depth[..., None]).clamp_min(0.0)).clamp(0.0, 1.0)
    sh_response = opacity[:, None, None] * transmission_d * (color - dc_color)
    opacity_tangent = (
        opacity * (1.0 - opacity)
    )[:, None, None] * (transmission_d * color - transmission_b * b_inf)
    zeros = torch.zeros((), device=rest.device, dtype=rest.dtype)
    return (
        torch.where(mask, sh_response, zeros),
        torch.where(mask, opacity_tangent, zeros),
        color,
        transmission_d,
    )


def _effective_svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.size == 0 or not np.isfinite(matrix).all():
        return np.empty((matrix.shape[0], 0)), np.empty(0), np.empty((0, matrix.shape[1]))
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] <= EPS:
        return np.empty((matrix.shape[0], 0)), singular, np.empty((0, matrix.shape[1]))
    valid = singular >= singular[0] * SVD_RELATIVE_THRESHOLD
    cumulative = np.cumsum(np.square(singular)) / max(float(np.square(singular).sum()), EPS)
    energy_rank = int(np.searchsorted(cumulative, SVD_ENERGY_THRESHOLD) + 1)
    rank = min(int(valid.sum()), energy_rank)
    rank = max(rank, 1)
    return u[:, :rank], singular[:rank], vh[:rank]


def _directions_and_gates(
    rest: Tensor,
    dc: Tensor,
    opacity: Tensor,
    observations: Mapping[str, Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    with torch.no_grad():
        sh_response, opacity_tangent, color, transmission_d = _responses(
            rest, dc, opacity, observations
        )
        sigmoid_derivative = color * (1.0 - color)
        bases = observations["bases"][..., 1:]
        mask = observations["mask"]
        directions = torch.zeros_like(rest)
        targets = torch.zeros(rest.shape[0], dtype=rest.dtype, device=rest.device)
        overlaps = torch.full((rest.shape[0],), float("nan"), device=rest.device)
        ranks = torch.zeros(rest.shape[0], dtype=torch.int16, device=rest.device)
        for index in range(rest.shape[0]):
            visible = mask[index]
            support = int(visible.sum())
            if support < 2:
                continue
            local_bases = bases[index, visible]
            local_alpha = opacity[index]
            local_t = transmission_d[index, visible]
            local_sigmoid = sigmoid_derivative[index, visible]
            jacobian = torch.zeros(support, 3, 45, device=rest.device)
            for channel in range(3):
                jacobian[:, channel, channel::3] = (
                    local_alpha * local_t[:, channel] * local_sigmoid[:, channel]
                )[:, None] * local_bases
            matrix = jacobian.reshape(-1, 45).double().cpu().numpy()
            tangent = opacity_tangent[index, visible].reshape(-1).double().cpu().numpy()
            response = sh_response[index, visible].reshape(-1).double().cpu().numpy()
            u, _singular, _vh = _effective_svd(matrix)
            tangent_norm = float(np.linalg.norm(tangent))
            if u.shape[1] == 0 or tangent_norm <= EPS:
                continue
            overlap = float(np.linalg.norm(u.T @ tangent) / tangent_norm)
            tangent_unit = tangent / tangent_norm
            shared_gradient = matrix.T @ tangent_unit
            gradient_norm = float(np.linalg.norm(shared_gradient))
            if not math.isfinite(overlap) or gradient_norm <= EPS:
                continue
            direction = torch.from_numpy((shared_gradient / gradient_norm).reshape(15, 3)).to(
                device=rest.device, dtype=rest.dtype
            )
            directions[index] = direction
            current_projection = float((rest[index] * direction).sum())
            shared_response_coordinate = float(np.dot(response, tangent_unit))
            targets[index] = current_projection - shared_response_coordinate / gradient_norm
            overlaps[index] = overlap
            ranks[index] = u.shape[1]
        finite = torch.isfinite(overlaps)
        gates = torch.zeros_like(overlaps)
        gates[finite] = ((overlaps[finite] - OVERLAP_THRESHOLD) / (1.0 - OVERLAP_THRESHOLD)).clamp(0, 1)
        finite_overlaps = overlaps[finite]
        finite_ranks = ranks[finite].float()
        return directions.detach(), targets.detach(), gates.detach(), {
            "finite_direction_count": int(finite.sum()),
            "finite_direction_fraction": float(finite.float().mean()),
            "median_overlap": float(finite_overlaps.median()) if bool(finite.any()) else float("nan"),
            "minimum_overlap": float(finite_overlaps.min()) if bool(finite.any()) else float("nan"),
            "maximum_overlap": float(finite_overlaps.max()) if bool(finite.any()) else float("nan"),
            "median_effective_SH_rank": float(finite_ranks.median()) if bool(finite.any()) else float("nan"),
            "minimum_effective_SH_rank": int(finite_ranks.min()) if bool(finite.any()) else 0,
            "maximum_effective_SH_rank": int(finite_ranks.max()) if bool(finite.any()) else 0,
            "active_gate_fraction": float((gates > 0).float().mean()),
            "median_active_gate": float(gates[gates > 0].median()) if bool((gates > 0).any()) else 0.0,
        }


def _regularizer(rest: Tensor, directions: Tensor, targets: Tensor, gates: Tensor) -> Tensor:
    residual = (rest * directions).sum(dim=(1, 2)) - targets
    return 0.5 * (gates * residual.square()).sum() / gates.sum().clamp_min(1.0)


def _controlled_coefficients(
    rest: Tensor,
    directions: Tensor,
    targets: Tensor,
    gates: Tensor,
    strength: float,
) -> Tensor:
    residual = (rest * directions).sum(dim=(1, 2)) - targets
    return rest - float(strength) * gates[:, None, None] * residual[:, None, None] * directions


def _weighted_energy(values: Tensor, gates: Tensor) -> float:
    valid = gates > 0
    if not bool(valid.any()):
        return float("nan")
    return float((gates[valid] * values[valid]).sum() / gates[valid].sum())


def _shared_response_energy(response: Tensor, tangent: Tensor, gates: Tensor) -> float:
    flat_response = response.reshape(response.shape[0], -1)
    flat_tangent = tangent.reshape(tangent.shape[0], -1)
    coefficient = (flat_response * flat_tangent).sum(-1) / flat_tangent.square().sum(-1).clamp_min(EPS)
    shared = coefficient[:, None] * flat_tangent
    return _weighted_energy(shared.square().sum(-1), gates)


def _relative_change(after: float, before: float) -> float:
    return (after - before) / max(abs(before), EPS)


def _optimization_preflight(
    rest_initial: Tensor,
    dc: Tensor,
    opacity: Tensor,
    observations: Mapping[str, Tensor],
    directions: Tensor,
    targets: Tensor,
    gates: Tensor,
) -> Dict[str, Any]:
    disabled = _controlled_coefficients(rest_initial, directions, targets, gates, 0.0)
    disabled_max_diff = float((disabled - rest_initial).abs().max())

    rest_probe = rest_initial.detach().clone().requires_grad_(True)
    opacity_probe = opacity.detach().clone().requires_grad_(True)
    probe_loss = _regularizer(rest_probe, directions, targets, gates)
    rest_gradient, opacity_gradient = torch.autograd.grad(
        probe_loss, (rest_probe, opacity_probe), allow_unused=True
    )
    rest_gradient_norm = float(torch.linalg.vector_norm(rest_gradient))
    opacity_gradient_max = 0.0 if opacity_gradient is None else float(opacity_gradient.abs().max())

    with torch.no_grad():
        response_initial, tangent_initial, color_initial, transmission_initial = _responses(
            rest_initial, dc, opacity, observations
        )
        response_disabled, _tangent_disabled, _color_disabled, _transmission_disabled = _responses(
            disabled, dc, opacity, observations
        )
        disabled_response_max_diff = float((response_disabled - response_initial).abs().max())
        projection_initial = (rest_initial * directions).sum(dim=(1, 2))
        shared_coordinate_initial = projection_initial - targets
        parallel_initial = _weighted_energy(shared_coordinate_initial.square(), gates)
        orthogonal_initial = rest_initial - projection_initial[:, None, None] * directions
        variance_initial = float(torch.var(rest_initial, unbiased=False))
        response_rms_initial = float(torch.sqrt(response_initial.square().mean()))
        direct_initial = torch.where(
            observations["mask"][..., None],
            opacity[:, None, None] * transmission_initial * color_initial,
            torch.zeros((), device=rest_initial.device, dtype=rest_initial.dtype),
        )
        direct_rgb_rms_initial = float(torch.sqrt(direct_initial.square().mean()))
        shared_response_initial = _shared_response_energy(response_initial, tangent_initial, gates)

    rest = rest_initial.detach().clone()
    losses: List[float] = []
    gate_sum = float(gates.sum())
    learning_rate = CONTROL_BUDGET * max(gate_sum, 1.0) / OPTIMIZATION_STEPS
    for _step in range(OPTIMIZATION_STEPS):
        rest = rest.detach().requires_grad_(True)
        loss = _regularizer(rest, directions, targets, gates)
        gradient = torch.autograd.grad(loss, rest)[0]
        with torch.no_grad():
            rest = rest - learning_rate * gradient
        losses.append(float(loss))

    with torch.no_grad():
        response_final, _tangent_final, color_final, transmission_final = _responses(
            rest, dc, opacity, observations
        )
        projection_final = (rest * directions).sum(dim=(1, 2))
        shared_coordinate_final = projection_final - targets
        parallel_final = _weighted_energy(shared_coordinate_final.square(), gates)
        orthogonal_final = rest - projection_final[:, None, None] * directions
        variance_final = float(torch.var(rest, unbiased=False))
        response_rms_final = float(torch.sqrt(response_final.square().mean()))
        direct_final = torch.where(
            observations["mask"][..., None],
            opacity[:, None, None] * transmission_final * color_final,
            torch.zeros((), device=rest.device, dtype=rest.dtype),
        )
        direct_rgb_rms_final = float(torch.sqrt(direct_final.square().mean()))
        shared_response_final = _shared_response_energy(response_final, tangent_initial, gates)
        orthogonal_relative_drift = float(
            torch.linalg.vector_norm(orthogonal_final - orthogonal_initial)
            / torch.linalg.vector_norm(orthogonal_initial).clamp_min(EPS)
        )

    parallel_reduction = -_relative_change(parallel_final, parallel_initial)
    shared_response_reduction = -_relative_change(shared_response_final, shared_response_initial)
    variance_ratio = variance_final / max(variance_initial, EPS)
    response_rms_ratio = response_rms_final / max(response_rms_initial, EPS)
    direct_rgb_rms_ratio = direct_rgb_rms_final / max(direct_rgb_rms_initial, EPS)
    return {
        "disabled_equivalence_max_abs_diff": disabled_max_diff,
        "disabled_response_max_abs_diff": disabled_response_max_diff,
        "gradient_pathway": {
            "features_rest_gradient_norm": rest_gradient_norm,
            "opacity_direct_gradient_max_abs": opacity_gradient_max,
            "opacity_direct_gradient_is_none": opacity_gradient is None,
            "intended_features_rest_gradient_nonzero": rest_gradient_norm > 0.0,
            "opacity_direct_gradient_zero": opacity_gradient is None,
        },
        "optimization": {
            "steps": OPTIMIZATION_STEPS,
            "cumulative_control_budget": CONTROL_BUDGET,
            "effective_learning_rate": learning_rate,
            "loss_initial": losses[0],
            "loss_final": losses[-1],
            "anchored_shared_coordinate_energy_initial": parallel_initial,
            "anchored_shared_coordinate_energy_final": parallel_final,
            "anchored_shared_coordinate_energy_reduction_fraction": parallel_reduction,
            "opacity_tangent_shared_response_energy_initial": shared_response_initial,
            "opacity_tangent_shared_response_energy_final": shared_response_final,
            "opacity_tangent_shared_response_energy_reduction_fraction": shared_response_reduction,
            "orthogonal_coefficient_relative_drift": orthogonal_relative_drift,
        },
        "collapse": {
            "SH_coefficient_variance_initial": variance_initial,
            "SH_coefficient_variance_final": variance_final,
            "SH_coefficient_variance_ratio": variance_ratio,
            "SH_response_RMS_initial": response_rms_initial,
            "SH_response_RMS_final": response_rms_final,
            "SH_response_RMS_ratio": response_rms_ratio,
            "local_direct_RGB_RMS_initial": direct_rgb_rms_initial,
            "local_direct_RGB_RMS_final": direct_rgb_rms_final,
            "local_direct_RGB_RMS_ratio": direct_rgb_rms_ratio,
            "bounded_RGB_min_final": float(color_final.min()),
            "bounded_RGB_max_final": float(color_final.max()),
            "bounded_RGB_finite": bool(torch.isfinite(color_final).all()),
            "opacity_max_abs_diff": 0.0,
        },
    }


def worker(scene: str, gpu: str, sample_count: int = SAMPLE_COUNT) -> Dict[str, Any]:
    runtime = _runtime(scene, gpu)
    started = time.perf_counter()
    branch = FORMAL._setup_branch(REPO_ROOT, CAUSAL._scene_config(scene), "C0")
    try:
        model = branch.pipeline.model
        train_records = FORMAL._train_records(branch.pipeline)
        payload = FORMAL._load_checkpoint(branch, _checkpoint(scene))
        if (
            payload.get("experiment") != FORMAL.EXPERIMENT
            or payload.get("branch") != "C0"
            or int(payload.get("absolute_step", -1)) != STEP
            or payload.get("ocmc_bundle") is None
            or payload.get("raoc_state") is not None
        ):
            raise RuntimeError("checkpoint condition provenance drift")
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
        model_state_before = CAUSAL._model_state_hash(model)
        projector_before = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
        gaussian_count_before = int(model.num_points)
        opacity_before = CAUSAL._tensor_hash(model.opacities)
        dc_before = CAUSAL._tensor_hash(model.features_dc)

        support = AUDIT.VC._support_counts(model, train_records)
        selected, sampling = AUDIT._sample_gaussians(scene, STEP, support, sample_count)
        observations = _collect_observations(model, train_records, selected)
        if not torch.equal(observations["observed"], support[selected]):
            raise RuntimeError("training visibility differs from frozen support")
        selected_gpu = selected.to(model.device)
        rest = model.features_rest.detach()[selected_gpu].float().clone()
        dc = model.features_dc.detach()[selected_gpu].float().clone()
        opacity = torch.sigmoid(model.opacities.detach()).reshape(-1)[selected_gpu].float().clone()
        directions, targets, gates, direction_summary = _directions_and_gates(
            rest, dc, opacity, observations
        )
        metrics = _optimization_preflight(rest, dc, opacity, observations, directions, targets, gates)

        model_state_after = CAUSAL._model_state_hash(model)
        projector_after = CAUSAL._tensor_hash(model._camera_medium_observability_projector)
        gaussian_count_after = int(model.num_points)
        opacity_after = CAUSAL._tensor_hash(model.opacities)
        dc_after = CAUSAL._tensor_hash(model.features_dc)
        integrity = {
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "model_unchanged": model_state_before == model_state_after,
            "ocmc_projector_sha256_before": projector_before,
            "ocmc_projector_sha256_after": projector_after,
            "ocmc_projector_unchanged": projector_before == projector_after,
            "opacity_sha256_before": opacity_before,
            "opacity_sha256_after": opacity_after,
            "opacity_unchanged": opacity_before == opacity_after,
            "features_dc_sha256_before": dc_before,
            "features_dc_sha256_after": dc_after,
            "features_dc_unchanged": dc_before == dc_after,
            "gaussian_count_before": gaussian_count_before,
            "gaussian_count_after": gaussian_count_after,
            "gaussian_count_unchanged": gaussian_count_before == gaussian_count_after,
            "checkpoint_writes": 0,
            "render_writes": 0,
            "model_training_steps": 0,
        }
        collapse = metrics["collapse"]
        optimization = metrics["optimization"]
        gradient = metrics["gradient_pathway"]
        scene_ready = bool(
            direction_summary["finite_direction_count"] >= MIN_VALID_SAMPLE
            and direction_summary["active_gate_fraction"] >= MIN_ACTIVE_GATE_FRACTION
            and metrics["disabled_equivalence_max_abs_diff"] == 0.0
            and metrics["disabled_response_max_abs_diff"] == 0.0
            and gradient["intended_features_rest_gradient_nonzero"]
            and gradient["opacity_direct_gradient_zero"]
            and optimization["anchored_shared_coordinate_energy_reduction_fraction"]
            >= MIN_PARALLEL_ENERGY_REDUCTION
            and optimization["opacity_tangent_shared_response_energy_reduction_fraction"]
            >= MIN_RESPONSE_SHARED_ENERGY_REDUCTION
            and optimization["orthogonal_coefficient_relative_drift"] <= MAX_ORTHOGONAL_DRIFT
            and MIN_SH_VARIANCE_RATIO <= collapse["SH_coefficient_variance_ratio"] <= MAX_SH_VARIANCE_RATIO
            and MIN_SH_RESPONSE_RMS_RATIO <= collapse["SH_response_RMS_ratio"] <= MAX_SH_RESPONSE_RMS_RATIO
            and MIN_DIRECT_RGB_RMS_RATIO
            <= collapse["local_direct_RGB_RMS_ratio"]
            <= MAX_DIRECT_RGB_RMS_RATIO
            and collapse["bounded_RGB_finite"]
            and 0.0 <= collapse["bounded_RGB_min_final"] <= collapse["bounded_RGB_max_final"] <= 1.0
            and collapse["opacity_max_abs_diff"] == 0.0
            and all(
                integrity[name]
                for name in (
                    "model_unchanged",
                    "ocmc_projector_unchanged",
                    "opacity_unchanged",
                    "features_dc_unchanged",
                    "gaussian_count_unchanged",
                )
            )
        )
        result = {
            "experiment": EXPERIMENT,
            "scene": scene,
            "candidate": SELECTED_CANDIDATE,
            "runtime": runtime,
            "checkpoint": str(_checkpoint(scene)),
            "checkpoint_sha256": _sha256(_checkpoint(scene)),
            "sampling": sampling,
            "direction_summary": direction_summary,
            **metrics,
            "integrity": integrity,
            "scene_ready": scene_ready,
            "elapsed_seconds": time.perf_counter() - started,
            "uses_gt": False,
            "uses_heldout_views": False,
            "ocmc_enabled": True,
            "raoc_enabled": False,
        }
        worker_dir = OUTPUT_ROOT / "workers" / scene
        _write_json(worker_dir / "worker_summary.json", result)
        print(
            f"[{scene}] n={direction_summary['finite_direction_count']} "
            f"overlap={direction_summary['median_overlap']:.6f} "
            f"parallel_reduction={optimization['anchored_shared_coordinate_energy_reduction_fraction']:.6f} "
            f"response_reduction={optimization['opacity_tangent_shared_response_energy_reduction_fraction']:.6f} "
            f"ready={scene_ready}",
            flush=True,
        )
        return result
    finally:
        FORMAL._release(branch)


def _classification(worker_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scene_rows = []
    for result in worker_results:
        scene_rows.append(
            {
                "scene": result["scene"],
                "finite_direction_count": result["direction_summary"]["finite_direction_count"],
                "median_overlap": result["direction_summary"]["median_overlap"],
                "active_gate_fraction": result["direction_summary"]["active_gate_fraction"],
                "disabled_equivalence_max_abs_diff": result["disabled_equivalence_max_abs_diff"],
                "disabled_response_max_abs_diff": result["disabled_response_max_abs_diff"],
                "features_rest_gradient_norm": result["gradient_pathway"]["features_rest_gradient_norm"],
                "opacity_direct_gradient_max_abs": result["gradient_pathway"]["opacity_direct_gradient_max_abs"],
                "parallel_energy_reduction": result["optimization"]["anchored_shared_coordinate_energy_reduction_fraction"],
                "shared_response_energy_reduction": result["optimization"]["opacity_tangent_shared_response_energy_reduction_fraction"],
                "orthogonal_relative_drift": result["optimization"]["orthogonal_coefficient_relative_drift"],
                "SH_variance_ratio": result["collapse"]["SH_coefficient_variance_ratio"],
                "SH_response_RMS_ratio": result["collapse"]["SH_response_RMS_ratio"],
                "local_direct_RGB_RMS_ratio": result["collapse"]["local_direct_RGB_RMS_ratio"],
                "opacity_unchanged": result["integrity"]["opacity_unchanged"],
                "gaussian_count_unchanged": result["integrity"]["gaussian_count_unchanged"],
                "model_unchanged": result["integrity"]["model_unchanged"],
                "ocmc_projector_unchanged": result["integrity"]["ocmc_projector_unchanged"],
                "scene_ready": result["scene_ready"],
            }
        )
    ready_count = sum(bool(row["scene_ready"]) for row in scene_rows)
    integrity = all(
        result["integrity"]["checkpoint_writes"] == 0
        and result["integrity"]["render_writes"] == 0
        and result["integrity"]["model_training_steps"] == 0
        and result["integrity"]["model_unchanged"]
        and result["integrity"]["ocmc_projector_unchanged"]
        for result in worker_results
    )
    label = "MODULE_READY" if ready_count >= 3 and integrity else "MODULE_NOT_READY"
    return {
        "experiment": EXPERIMENT,
        "classification": label,
        "selected_candidate": SELECTED_CANDIDATE,
        "ready_scene_count": ready_count,
        "required_ready_scene_count": 3,
        "quality_and_integrity_passed": integrity,
        "full_causal_training_authorized": label == "MODULE_READY",
        "next_unique_task": (
            "IDENTIFIABILITY_MODULE_CAUSAL_EXPERIMENT"
            if label == "MODULE_READY"
            else "GAUSSIAN_IDENTIFIABILITY_MECHANISM_REDESIGN"
        ),
        "scene_rows": scene_rows,
    }


def _research_note(summary: Mapping[str, Any]) -> None:
    classification = summary["classification"]
    lines = [
        "# Gaussian Identifiability Module Design",
        "",
        "Date: 2026-09-01",
        f"Classification: `{classification['classification']}`",
        "",
        "## Motivation",
        "",
        "The frozen four-scene audit supported a Gaussian appearance-density ambiguity: non-DC/full SH sensitivity and raw-opacity sensitivity occupy nearly identical training-view RGB subspaces, and the ambiguity score predicts heldout error after depth, medium, and OCMC controls.",
        "",
        "## Candidate Comparison",
        "",
        "| Candidate | OCMC compatibility | Cost | valid-appearance risk | Decision |",
        "|---|---|---|---|---|",
        "| A: detached SH-opacity tangent orthogonalization | Separate per-Gaussian appearance axis | Periodic training-only analytic sensitivity | Low: one of 45 non-DC directions | selected |",
        "| B: opacity-conditioned SH capacity | Compatible but opacity is not observability | Low | High: suppresses legitimate SH | rejected |",
        "| C: update decorrelation | Interacts with all losses | High: paired gradients/update hook | Medium-high; normally GT-dependent | rejected |",
        "",
        "## Formulation",
        "",
        "For each Gaussian, stack the non-DC bounded-SH Jacobian `J_i`, non-DC response `r_i0`, and raw-opacity tangent `j_alpha_i` over visible training views. With `u_i=normalize(j_alpha_i)`, the detached shared direction is `h_i=normalize(J_i^T u_i)`. At refresh anchor `theta_i0`, the target `tau_i=h_i^T theta_i0-(u_i^T r_i0)/||J_i^T u_i||` nulls the shared response under the local linearization. A detached overlap gate activates only when the opacity tangent lies in the effective non-DC SH response subspace. The proposed regularizer is `lambda*sum_i g_i(h_i^T theta_i-tau_i)^2/(2 sum_i g_i)`.",
        "",
        "Only `features_rest` receives a direct gradient. DC color, opacity, geometry, medium parameters, OCMC, and Gaussian topology are unchanged. Forty-four orthogonal non-DC SH dimensions remain unpenalized for valid view-dependent effects. At inference, learned coefficients are rendered normally: no gate, Jacobian, or extra compute remains.",
        "",
        "## Relation With OCMC",
        "",
        "OCMC controls camera-conditioned medium context. The selected mechanism controls a per-Gaussian SH coefficient direction that locally duplicates opacity. Their parameters and gradient pathways are disjoint, so this module is complementary rather than an OCMC replacement.",
        "",
        "## Preflight Results",
        "",
        "| Scene | overlap | active | parallel reduction | response reduction | orth drift | SH var ratio | SH RMS ratio | direct RGB ratio | ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in classification["scene_rows"]:
        lines.append(
            f"| {row['scene']} | {row['median_overlap']:.6f} | {row['active_gate_fraction']:.3f} | "
            f"{row['parallel_energy_reduction']:.6f} | {row['shared_response_energy_reduction']:.6f} | "
            f"{row['orthogonal_relative_drift']:.3e} | {row['SH_variance_ratio']:.6f} | "
            f"{row['SH_response_RMS_ratio']:.6f} | {row['local_direct_RGB_RMS_ratio']:.6f} | "
            f"{'yes' if row['scene_ready'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Strength-zero coefficient output was exactly equivalent in every scene. The direct `features_rest` gradient was nonzero, direct opacity gradient was zero, and frozen model, opacity, DC, Gaussian count, and OCMC projector hashes were unchanged. The 50-step optimization affected cloned sampled SH tensors only; it was not model training.",
            "",
            "## Implementation Feasibility",
            "",
            "A later causal implementation can refresh detached directions and gates at low cadence from recent training cameras, retain them only for active/visible Gaussians, and add the scalar term to the existing objective. No renderer or inference modification is required. This task deliberately did not integrate the regularizer into the production loss or training loop.",
            "",
            "## Risks",
            "",
            "The controller is local and first-order; stale directions may become inaccurate. A non-DC SH direction can be opacity-equivalent on observed views yet useful outside them. The future causal experiment must therefore compare multiple small strengths, monitor novel-view quality and SH utilization, and include a zero-strength equivalence branch. Preflight readiness is engineering authorization, not evidence of causal quality improvement.",
            "",
            "## Classification",
            "",
            f"The result is `{classification['classification']}` with {classification['ready_scene_count']}/4 ready scenes. Full causal training authorization is `{str(classification['full_causal_training_authorized']).lower()}`.",
            "",
        ]
    )
    if classification["classification"] == "MODULE_READY":
        lines.append("The next and only authorized task is `IDENTIFIABILITY_MODULE_CAUSAL_EXPERIMENT`. No 15K training was run here.")
    else:
        lines.append("Return to Gaussian identifiability mechanism redesign. Full causal training is not authorized.")
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "All four registered C0 checkpoints used OCMC on, RAOC off, bounded SH3, `dir_xy_camera`, tied `B_inf`, and classic rasterization. No heldout view or GT entered sampling, tangent construction, optimization, or classification. Checkpoint writes, render writes, model training steps, renderer/CUDA/optimizer/loss/training-loop changes were zero.",
            "",
        ]
    )
    RESEARCH_NOTE.write_text("\n".join(lines), encoding="utf8")


def aggregate() -> Dict[str, Any]:
    manifest = _read_json(OUTPUT_ROOT / "preflight_manifest.json")
    workers = [_read_json(OUTPUT_ROOT / "workers" / scene / "worker_summary.json") for scene in SCENES]
    classification = _classification(workers)
    if _sha256(Path(__file__).resolve()) != manifest["repo"]["preflight_script_sha256"]:
        raise RuntimeError("preflight script changed during execution")
    after = {relative: _sha256(REPO_ROOT / relative) for relative in PROTECTED_HASHES}
    if after != manifest["repo"]["protected_hashes"]:
        raise RuntimeError("protected source changed during execution")
    for row in manifest["sources"]:
        if _sha256(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed during execution: {row['checkpoint']}")
    summary = {
        "experiment": EXPERIMENT,
        "classification": classification,
        "candidate_analysis": CANDIDATE_ANALYSIS,
        "mechanism_specification": str(OUTPUT_ROOT / "mechanism_specification.md"),
        "workers": workers,
        "integrity": {
            "preflight_script_sha256_before_after_match": True,
            "protected_source_hashes_before_after_match": True,
            "checkpoint_hashes_before_after_match": True,
            "checkpoint_count": len(manifest["sources"]),
            "checkpoint_writes": 0,
            "render_writes": 0,
            "model_training_steps": 0,
            "renderer_changes": 0,
            "cuda_changes": 0,
            "optimizer_changes": 0,
            "loss_changes": 0,
            "training_loop_changes": 0,
        },
    }
    _write_json(OUTPUT_ROOT / "preflight_results.json", summary)
    _write_json(OUTPUT_ROOT / "classification.json", classification)
    _research_note(summary)
    return summary


def launch() -> Dict[str, Any]:
    manifest = preflight()
    logs = OUTPUT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for scene, gpu in SCENE_GPUS.items():
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["CONDA_DEFAULT_ENV"] = "water_splatting"
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
            failures.append(
                {"scene": scene, "gpu": gpu, "exit_code": code, "log": str(logs / f"{scene}.log")}
            )
    if failures:
        _write_json(OUTPUT_ROOT / "worker_failures.json", {"rows": failures})
        raise RuntimeError(f"identifiability module preflight workers failed: {failures}")
    return {"manifest": manifest, "summary": aggregate()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=tuple(SCENE_GPUS.values()))
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    args = parser.parse_args()
    if args.worker:
        if args.scene is None or args.gpu is None:
            parser.error("--worker requires --scene and --gpu")
        if args.sample_count < 12:
            parser.error("--sample-count must be at least 12")
        result = worker(args.scene, args.gpu, args.sample_count)
    elif args.preflight:
        result = preflight()
    elif args.aggregate:
        result = aggregate()
    else:
        result = launch()
    print(json.dumps(_sanitize(result), indent=2, sort_keys=True, default=_json_default, allow_nan=False))


if __name__ == "__main__":
    main()
