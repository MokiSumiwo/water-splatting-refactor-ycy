#!/usr/bin/env python3
"""Validate the H1 RAOC CUDA sensitivity-only backend.

This validation is intentionally separate from the historical full-fused
audit.  H1 compares the reference and ``cuda_hybrid`` production paths while
keeping ``cuda_fused`` as a performance upper-bound reference.  The script
does not launch a formal 15K experiment.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import validate_raoc_cuda_training_equivalence as previous
from scripts.experiments import run_raoc_q50_q80_causal_scene as RAOC
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from water_splatting.raoc import (
    apply_modal_keep_gate,
    apply_standardized_projector,
    cuda_sensitivity_norm,
    local_keep_gates,
    modal_coefficients,
    ray_keep_gates,
)


OUTPUT_ROOT = REPO_ROOT / "outputs" / "raoc_hybrid_cuda_acceleration_20260831"
BACKENDS = ("reference", "cuda_hybrid", "cuda_fused")
Q50 = 0.50
Q80 = 0.80
EPS = 1e-12
OPERATOR_SCENE = "IUI3-RedSea"
START_STEP = 3000


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


def _finite(value: Any) -> bool:
    return bool(torch.isfinite(value).all().item()) if isinstance(value, Tensor) else True


def _diff(left: Tensor, right: Tensor) -> Dict[str, float]:
    a = left.detach().float().cpu().reshape(-1)
    b = right.detach().float().cpu().reshape(-1)
    if a.shape != b.shape:
        return {"max_abs": float("inf"), "mean_abs": float("inf"), "rms": float("inf"), "relative_l2": float("inf"), "cosine": float("nan")}
    if not a.numel():
        return {"max_abs": 0.0, "mean_abs": 0.0, "rms": 0.0, "relative_l2": 0.0, "cosine": 1.0}
    delta = (a - b).abs()
    denom = max(float(a.norm().item()), EPS)
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "rms": float(delta.square().mean().sqrt().item()),
        "relative_l2": float((a - b).norm().item() / denom),
        "cosine": float(torch.nn.functional.cosine_similarity(a[None], b[None], dim=1, eps=EPS).item()),
    }


def _distribution(value: Tensor) -> Dict[str, float]:
    flat = value.detach().float().cpu().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if not flat.numel():
        return {key: float("nan") for key in ("mean", "median", "p95", "p99", "p99.9", "max")}
    return {
        "mean": float(flat.mean().item()),
        "median": float(torch.quantile(flat, 0.50).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "p99.9": float(torch.quantile(flat, 0.999).item()),
        "max": float(flat.max().item()),
    }


def _runtime(gpu: str) -> Dict[str, Any]:
    return previous._runtime(gpu)


def _load_branch(scene: str, backend: str, step: int = START_STEP):
    previous.BACKENDS = BACKENDS
    return previous._load_raoc_branch(scene, backend, step)


def _hybrid_controls(model: Any, camera: Any, state: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    """Run H1 sensitivity CUDA and the unchanged reference modal path."""

    raw_full, raw_base, height, width = RAOC._raw_pair(model, camera)
    scale = state["standardization_scale"].detach().to(model.device, dtype=torch.float32).clamp_min(1e-6)
    basis = state["basis"].detach().to(model.device, dtype=torch.float32)
    raw = raw_full.reshape(-1, 9).float()
    delta_raw = (raw_full - raw_base).reshape(-1, 9).float()
    delta_std = delta_raw / scale.reshape(1, 9)
    geometry = previous._operator_geometry(model, camera, height, width)
    medium_rgb = torch.sigmoid(raw[:, :3])
    medium_bs = torch.nn.functional.softplus(raw[:, 3:6] + float(model.medium_density_bias))
    medium_attn = torch.nn.functional.softplus(raw[:, 6:9] + float(model.medium_density_bias))
    d_rgb = medium_rgb * (1.0 - medium_rgb)
    d_bs = torch.sigmoid(raw[:, 3:6] + float(model.medium_density_bias))
    d_attn = torch.sigmoid(raw[:, 6:9] + float(model.medium_density_bias))
    sensitivity = cuda_sensitivity_norm(
        raw_medium=raw,
        raw_directions=basis.T * scale.reshape(1, 9),
        medium_rgb=medium_rgb,
        medium_bs=medium_bs,
        medium_attn=medium_attn,
        d_rgb=d_rgb,
        d_bs=d_bs,
        d_attn=d_attn,
        xys=geometry["xys"],
        depths=geometry["depths"],
        radii=geometry["radii"],
        conics=geometry["conics"],
        colors=geometry["colors"],
        opacities=geometry["opacities"],
        gaussian_ids_sorted=geometry["ids"],
        tile_bins=geometry["tile_bins"],
        height=height,
        width=width,
        block_width=model.underwater_rasterizer.block_width,
        num_intersects=geometry["num_intersects"],
    )
    coefficients = modal_coefficients(delta_std.detach(), basis)
    evidence = coefficients.abs() * sensitivity
    local_gate = local_keep_gates(evidence, state["local_scale"].to(model.device), state["active"].to(model.device))
    keep_gate = ray_keep_gates(state["global_gate"].to(model.device), local_gate)
    global_projector = basis @ torch.diag(state["global_gate"].to(model.device).float()) @ basis.T
    delta_raoc_std = apply_modal_keep_gate(delta_std, basis, keep_gate)
    all_keep = (keep_gate == 1).all(dim=1)
    zero_local = (local_gate == 0).all(dim=1)
    delta_raoc_std = torch.where(all_keep[:, None], delta_std, delta_raoc_std)
    ocmc_delta_raw = apply_standardized_projector(delta_raw, global_projector, scale)
    delta_raoc_std = torch.where(zero_local[:, None], ocmc_delta_raw / scale.reshape(1, 9), delta_raoc_std)
    delta_raoc_raw = delta_raoc_std * scale.reshape(1, 9)
    delta_raoc_raw = torch.where(all_keep[:, None], delta_raw, delta_raoc_raw)
    delta_raoc_raw = torch.where(zero_local[:, None], ocmc_delta_raw, delta_raoc_raw)
    return {
        "delta_std": delta_std.detach(),
        "coefficients": coefficients.detach(),
        "sensitivity": sensitivity.detach(),
        "evidence": evidence.detach(),
        "local_gate": local_gate.detach(),
        "keep_gate": keep_gate.detach(),
        "delta_raoc_std": delta_raoc_std.detach(),
        "delta_raoc_raw": delta_raoc_raw.detach(),
    }


def _controls(model: Any, camera: Any, state: Mapping[str, Tensor], backend: str, flat: Tensor) -> Dict[str, Tensor]:
    model.config.camera_medium_raoc_backend = backend
    if backend == "reference" or backend == "cuda_fused":
        values = previous._operator_controls(model, camera, state, flat)
    else:
        values_all = _hybrid_controls(model, camera, state)
        index = flat.to(model.device)
        values = {key: value[index] for key, value in values_all.items()}
    return {key: value.detach().float().cpu() for key, value in values.items()}


def _run_operator_backend(scene: str, backend: str, state: Mapping[str, Tensor], samples: Mapping[str, Tensor], step: int):
    holder, checkpoint = _load_branch(scene, backend, step)
    try:
        model = holder.pipeline.model
        RAOC._install_condition(model, "C1", checkpoint.get("ocmc_bundle"), state)
        records = {view_id: (camera, batch) for _idx, view_id, camera, batch in previous._records(holder)}
        pooled: Dict[str, List[Tensor]] = {}
        controls: Dict[str, List[Tensor]] = {}
        first_loss = None
        model.eval()
        for view_id, flat in samples.items():
            camera, batch = records[view_id]
            with torch.no_grad():
                outputs = model.get_outputs(camera.to(model.device))
                gt = MIC._gt_for(model, batch, outputs["background"]).to(model.device)
            values = previous._output_values(outputs, gt)
            for key, value in values.items():
                if key == "gt":
                    continue
                pooled.setdefault(key, []).append(value.reshape(-1, value.shape[-1] if value.ndim > 1 else 1))
            current = _controls(model, camera, state, backend, flat)
            for key, value in current.items():
                controls.setdefault(key, []).append(value)
            total = sum(model.get_loss_dict(outputs, previous._batch_device(batch, model.device), {}).values())
            first_loss = float(total.detach().cpu().item()) if first_loss is None else first_loss
            del outputs, gt, values, current, total
            gc.collect()
            torch.cuda.empty_cache()
        return (
            {key: torch.cat(values, dim=0) for key, values in pooled.items()},
            {key: torch.cat(values, dim=0) for key, values in controls.items()},
            {"scene": scene, "backend": backend, "step": int(step), "loss": first_loss, "finite": True},
        )
    finally:
        previous._release(holder)


def _gradient(scene: str, backend: str, state: Mapping[str, Tensor], step: int) -> Tuple[Tensor, float]:
    holder, checkpoint = _load_branch(scene, backend, step)
    try:
        model = holder.pipeline.model
        RAOC._install_condition(model, "C1", checkpoint.get("ocmc_bundle"), state)
        model.config.camera_medium_raoc_backend = backend
        model.train()
        model.zero_grad(set_to_none=True)
        _idx, _view_id, camera, batch = previous._records(holder)[0]
        outputs = model.get_outputs(camera.to(model.device))
        total = sum(model.get_loss_dict(outputs, previous._batch_device(batch, model.device), {}).values())
        total.backward()
        return previous._gradient(model).clone(), float(total.detach().cpu().item())
    finally:
        previous._release(holder)


def _operator_phase(gpu: str) -> None:
    runtime = _runtime(gpu)
    out = OUTPUT_ROOT / "operator"
    probe, checkpoint = _load_branch(OPERATOR_SCENE, "reference", START_STEP)
    try:
        samples = previous._sample_bank(OPERATOR_SCENE, probe)
        state_q50 = {key: value.detach().cpu().clone() for key, value in checkpoint["raoc_state"].items()}
        state_q80 = previous._state_for_q80(probe.pipeline.model, probe, samples, state_q50)
    finally:
        previous._release(probe)

    _write_json(out / "q_states.json", {"scene": OPERATOR_SCENE, "q50": state_q50, "q80": state_q80, "q50_quantile": Q50, "q80_quantile": Q80})
    distribution: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for label, state in (("q50", state_q50), ("q80", state_q80)):
        results = {}
        for backend in BACKENDS:
            print(f"[operator] {label} {backend}", flush=True)
            results[backend] = _run_operator_backend(OPERATOR_SCENE, backend, state, samples, START_STEP)
        ref_outputs, ref_controls, ref_meta = results["reference"]
        comparisons: Dict[str, Any] = {}
        for backend in ("cuda_hybrid", "cuda_fused"):
            out_errors = {name: _diff(ref_outputs[name], results[backend][0][name]) for name in sorted(set(ref_outputs) & set(results[backend][0]))}
            control_errors = {name: _diff(ref_controls[name], results[backend][1][name]) for name in sorted(set(ref_controls) & set(results[backend][1]))}
            for name, stats in {**out_errors, **control_errors}.items():
                rows.append({"scope": label, "backend": backend, "quantity": name, **stats})
            for name in ("sensitivity", "delta_raoc_std"):
                if name in ref_controls and name in results[backend][1]:
                    distribution[f"{label}.{backend}.{name}"] = _distribution((ref_controls[name] - results[backend][1][name]).abs())
            grad_ref, loss_ref = _gradient(OPERATOR_SCENE, "reference", state, START_STEP)
            grad_backend, loss_backend = _gradient(OPERATOR_SCENE, backend, state, START_STEP)
            grad_stats = _diff(grad_ref, grad_backend)
            grad_stats["cosine"] = float(torch.nn.functional.cosine_similarity(grad_ref[None], grad_backend[None], dim=1, eps=EPS).item())
            comparisons[backend] = {"outputs": out_errors, "controls": control_errors, "gradient": grad_stats, "reference_loss": loss_ref, "backend_loss": loss_backend}
            rows.append({"scope": label, "backend": backend, "quantity": "medium_mlp_gradient", **grad_stats})
        summaries[label] = {"scene": OPERATOR_SCENE, "checkpoint_step": START_STEP, "reference": ref_meta, "comparisons": comparisons}
        _write_json(out / f"operator_equivalence_{label}.json", summaries[label])

    _write_csv(out / "operator_errors.csv", rows)
    _write_json(out / "operator_error_distribution.json", distribution)
    _write_json(out / "operator_summary.json", {"scene": OPERATOR_SCENE, "q50": summaries["q50"], "q80": summaries["q80"], "runtime": runtime})

    # H1's early gate is evaluated against the production reference path.
    gate: Dict[str, Any] = {}
    for label in ("q50", "q80"):
        comparison = summaries[label]["comparisons"]["cuda_hybrid"]
        output_errors = comparison["outputs"]
        control_errors = comparison["controls"]
        gradient = comparison["gradient"]
        limits = {
            "sensitivity": 5e-5,
            "keep_gate": 1e-5,
            "delta_raoc_std": 2.5e-4,
            "pred_image": 5e-4,
            "medium_mlp_gradient_relative_l2": 5e-4,
            "gradient_cosine": 0.99999,
        }
        measured = {
            "sensitivity": control_errors["sensitivity"]["max_abs"],
            "keep_gate": control_errors["keep_gate"]["max_abs"],
            "delta_raoc_std": control_errors["delta_raoc_std"]["max_abs"],
            "pred_image": output_errors["pred_image"]["max_abs"],
            "medium_mlp_gradient_relative_l2": gradient["relative_l2"],
            "gradient_cosine": gradient["cosine"],
        }
        passed = all(measured[key] <= limits[key] for key in limits if key != "gradient_cosine") and measured["gradient_cosine"] >= limits["gradient_cosine"]
        gate[label] = {"limits": limits, "measured": measured, "pass": bool(passed)}
    gate_pass = all(item["pass"] for item in gate.values())
    _write_json(OUTPUT_ROOT / "operator_gate.json", {"gate": gate, "pass": gate_pass, "stop_if_failed": True, "previous_delta_z_raoc_std_max": 2.0751953125})
    if not gate_pass:
        raise SystemExit(2)


def _repo_phase() -> None:
    runtime = {
        "python_path": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_device_count": int(torch.cuda.device_count()),
    }
    previous.BACKENDS = BACKENDS
    _write_json(OUTPUT_ROOT / "repo_state.json", {
        "branch": previous._git("branch", "--show-current"),
        "head": previous._git("rev-parse", "HEAD"),
        "status_short": previous._git("status", "--short"),
        "historical_gmvc": list(previous.HISTORICAL_GMVC),
        "protected_q50_q80": list(previous.PROTECTED_Q50_Q80),
        "allowed_physical_gpus": sorted(previous.ALLOWED_GPUS),
        "active_backends": list(BACKENDS),
    })
    _write_json(OUTPUT_ROOT / "environment.json", runtime)
    _write_json(OUTPUT_ROOT / "hybrid_backend_contract.json", {
        "reference": "camera_medium_raoc_backend='reference'",
        "cuda_hybrid": "camera_medium_raoc_backend='cuda_hybrid'",
        "cuda_fused": "camera_medium_raoc_backend='cuda_fused'",
        "cuda_output": "sensitivity_norm [N, 9] only",
        "cuda_excludes": ["a", "e", "g_local", "g_keep", "Delta_z_raoc_std", "modal_reconstruction"],
        "reference_path_retained": ["V^T Delta_z_std", "e", "g_local", "g_keep", "V(g_keep*a)", "unstandardization", "medium MLP residual"],
        "sensitivity_detached": True,
        "all_modes": 9,
        "external_quantiles": [Q50, Q80],
        "default_backend": "reference",
        "formal_15k_experiment": False,
    })
    previous_error = json.loads((REPO_ROOT / "outputs/raoc_cuda_training_equivalence_20260829/error_localization.json").read_text(encoding="utf8"))
    _write_json(OUTPUT_ROOT / "previous_error_localization.json", previous_error)


def _finalize_phase() -> None:
    """Close the experiment after the registered H1 early gate fails."""

    operator_root = OUTPUT_ROOT / "operator"
    gate = json.loads((OUTPUT_ROOT / "operator_gate.json").read_text(encoding="utf8"))
    summary = json.loads((operator_root / "operator_summary.json").read_text(encoding="utf8"))
    distributions = json.loads((operator_root / "operator_error_distribution.json").read_text(encoding="utf8"))
    q50 = json.loads((operator_root / "operator_equivalence_q50.json").read_text(encoding="utf8"))
    q80 = json.loads((operator_root / "operator_equivalence_q80.json").read_text(encoding="utf8"))

    # Promote the operator artifacts to stable names at the output root.  The
    # nested copies retain the complete per-backend diagnostics.
    _write_json(OUTPUT_ROOT / "operator_equivalence_q50.json", q50)
    _write_json(OUTPUT_ROOT / "operator_equivalence_q80.json", q80)
    _write_json(OUTPUT_ROOT / "operator_error_distribution.json", {
        "scope": "reference_vs_cuda_hybrid_and_cuda_fused",
        "registered_fields": ["mean", "median", "p95", "p99", "p99.9", "max"],
        "distributions": distributions,
    })
    _write_json(OUTPUT_ROOT / "performance_gate.json", {
        "status": "NOT_RUN_EARLY_STOP",
        "reason": "H1 operator gate failed; the protocol forbids performance and trajectory phases after a bad early operator result.",
        "reference_vs_hybrid_complete_forward_backward_speedup": None,
        "minimum_speedup_for_trajectory": 2.0,
        "cuda_fused_performance_reference": True,
    })

    not_run = {
        "status": "NOT_RUN_EARLY_STOP",
        "reason": "H1 operator gate failed before this phase.",
        "formal_15k_experiment": False,
    }
    for name in ("iui3_fixed_topology.json", "four_scene_normal_topology.json", "topology_divergence.json", "memory_reference_vs_hybrid.json", "performance_reference_vs_hybrid.json"):
        _write_json(OUTPUT_ROOT / name, not_run)
    for name in ("iui3_fixed_topology.csv", "four_scene_normal_topology.csv", "memory_reference_vs_hybrid.csv", "performance_reference_vs_hybrid.csv"):
        _write_csv(OUTPUT_ROOT / name, [])

    q_metrics = {
        label: gate["gate"][label]
        for label in ("q50", "q80")
    }
    engineering = {
        "decision": "RAOC_HYBRID_ACCELERATION_NOT_SUPPORTED",
        "stage": "H1_OPERATOR_EARLY_GATE",
        "operator_gate_pass": bool(gate["pass"]),
        "q50": q_metrics["q50"],
        "q80": q_metrics["q80"],
        "previous_full_fused_delta_z_raoc_std_max": 2.0751953125e-3,
        "interpretation": "CUDA sensitivity is numerically close, but its small FP32 error is amplified by the reference gate and modal reconstruction path. H1 does not reduce the dominant Delta_z_raoc_std error.",
        "thresholds_relaxed": False,
        "long_training_phases_run": False,
        "cuda_fused_retained_as_performance_reference": True,
    }
    research = {
        "decision": "CLOSE_RAOC_AND_LOCK_OCMC",
        "reason": "The exactness-preserving sensitivity-only hybrid failed the pre-registered operator gate, so no training-equivalent RAOC acceleration was established.",
        "formal_raoc_q50_q80_experiment_cancelled": True,
        "ocmc_interpretation": "OCMC remains the previously validated camera-conditioned medium capacity-control mechanism; closing RAOC acceleration is not an OCMC failure.",
        "next_direction": "Keep formal science on OCMC and begin a separate research direction only after this task.",
    }
    _write_json(OUTPUT_ROOT / "final_engineering_classification.json", engineering)
    _write_json(OUTPUT_ROOT / "final_research_line_decision.json", research)
    _write_json(OUTPUT_ROOT / "final_summary.json", {
        "experiment": "RAOC-HYBRID-CUDA-EXACTNESS-PRESERVING-ACCELERATION",
        "date": "2026-08-31",
        "repo": {"branch": previous._git("branch", "--show-current"), "head": previous._git("rev-parse", "HEAD"), "status_short": previous._git("status", "--short")},
        "previous_error_localization": json.loads((OUTPUT_ROOT / "previous_error_localization.json").read_text(encoding="utf8")),
        "backend_contract": json.loads((OUTPUT_ROOT / "hybrid_backend_contract.json").read_text(encoding="utf8")),
        "operator": {"summary": summary, "gate": gate, "distributions": distributions},
        "phase_status": {
            "repo": "COMPLETE",
            "operator_q50_q80": "COMPLETE",
            "performance": "NOT_RUN_EARLY_STOP",
            "iui3_fixed_topology": "NOT_RUN_EARLY_STOP",
            "four_scene_normal_topology": "NOT_RUN_EARLY_STOP",
            "memory": "NOT_RUN_EARLY_STOP",
        },
        "engineering_classification": engineering,
        "research_line_decision": research,
    })

    note = f"""# RAOC Hybrid CUDA Acceleration Validation

Date: 2026-08-31

## Scope

This was the final engineering feasibility attempt for RAOC. The new
`camera_medium_raoc_backend='cuda_hybrid'` backend computes only the nine
renderer-local directional sensitivities `||J_p v_i||_2` in CUDA. The modal
projection, evidence, local and global gates, modal reconstruction,
unstandardization, and differentiable medium residual path remain in the
reference PyTorch implementation. The existing `reference` and
`cuda_fused` backends remain available; `reference` remains the default.

No RAOC equation, calibration state, quantile definition, medium model, or
formal 15K experiment was changed or run.

## Previous error localization

The archived full-fused validation identified `Delta_z_raoc_std` as the
dominant discrepancy: approximately `2.075195e-3` for Q50 and
`2.074718e-3` for Q80, while sensitivity itself differed by only approximately
`2.7567e-7`. This made H1 a direct test of whether moving only sensitivity to
CUDA would remove the error amplification.

## Operator result

IUI3 was evaluated from the archived C1 checkpoint at step 3000 with the
same geometry, calibration state, camera samples, and external Q50/Q80
states. The registered H1 gate was evaluated before any performance or
training phase.

- Q50: sensitivity max `{q_metrics['q50']['measured']['sensitivity']:.9g}`, g_keep max `{q_metrics['q50']['measured']['keep_gate']:.9g}`, Delta_z max `{q_metrics['q50']['measured']['delta_raoc_std']:.9g}`, pred max `{q_metrics['q50']['measured']['pred_image']:.9g}`, medium-gradient relative L2 `{q_metrics['q50']['measured']['medium_mlp_gradient_relative_l2']:.9g}`, cosine `{q_metrics['q50']['measured']['gradient_cosine']:.9g}`.
- Q80: sensitivity max `{q_metrics['q80']['measured']['sensitivity']:.9g}`, g_keep max `{q_metrics['q80']['measured']['keep_gate']:.9g}`, Delta_z max `{q_metrics['q80']['measured']['delta_raoc_std']:.9g}`, pred max `{q_metrics['q80']['measured']['pred_image']:.9g}`, medium-gradient relative L2 `{q_metrics['q80']['measured']['medium_mlp_gradient_relative_l2']:.9g}`, cosine `{q_metrics['q80']['measured']['gradient_cosine']:.9g}`.

The sensitivity output is close to reference, and the prediction and
gradient checks are within their registered limits. However, the Q50 gate
and Delta_z limits fail, and Q80 also fails the Delta_z limit. The
sensitivity-only CUDA boundary therefore does not remove the dominant
reference-vs-fused reconstruction discrepancy. The error distribution,
including mean, median, p95, p99, p99.9, and max, is in
`operator_error_distribution.json`.

## Early stop

The H1 operator gate failed. In accordance with the frozen protocol,
synchronized performance, IUI3 fixed-topology 500-step training,
Panama replication, four-scene normal-topology training, and memory A/B
were not run. No threshold was relaxed, no additional CUDA fusion boundary
was attempted, and no Q50/Q80 formal 15K experiment was launched.

## Decision

Engineering classification: `RAOC_HYBRID_ACCELERATION_NOT_SUPPORTED`.

Research-line decision: `CLOSE_RAOC_AND_LOCK_OCMC`.

This closes the RAOC acceleration line, not the OCMC mechanism. The formal
research direction should remain on the previously validated OCMC
camera-conditioned medium capacity-control mechanism. RAOC results may be
retained as mechanistic evidence, but RAOC should not be used as the formal
accelerated backend on the basis of this attempt.
"""
    (REPO_ROOT / "research_notes" / "RAOC_HYBRID_CUDA_ACCELERATION_2026-08-31.md").write_text(note, encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("repo", "operator", "finalize"))
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()
    if args.phase == "repo":
        _repo_phase()
    elif args.phase == "finalize":
        _finalize_phase()
    else:
        if args.gpu is None:
            raise ValueError("--gpu is required for operator phase")
        _operator_phase(args.gpu)


if __name__ == "__main__":
    main()
