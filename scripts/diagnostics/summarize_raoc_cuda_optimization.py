#!/usr/bin/env python3
"""Create the reproducible report artifacts for the RAOC CUDA optimization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "raoc_cuda_fusion_optimization_20260828"
PROFILE_ROOT = OUTPUT_ROOT / "runtime_profiles_fast"
OCMC_PROFILE_ROOT = OUTPUT_ROOT / "runtime_profiles_ocmc"
EQUIV_ROOT = OUTPUT_ROOT / "equivalence"
SMOKE_ROOT = OUTPUT_ROOT / "smoke"
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
STEPS = (3000, 8000, 13000, 14999)


def _read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf8")) if path.is_file() else default


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True, default=str) + "\n", encoding="utf8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: Iterable[Any]) -> float:
    values = [_number(value) for value in values]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else float("nan")


def _percentile(values: Iterable[Any], quantile: float) -> float:
    values = sorted(_number(value) for value in values if math.isfinite(_number(value)))
    if not values:
        return float("nan")
    index = quantile * (len(values) - 1)
    lower, upper = int(math.floor(index)), int(math.ceil(index))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _pearson(xs: Iterable[Any], ys: Iterable[Any]) -> float:
    pairs = [(_number(x), _number(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    mx, my = _mean(x for x, _ in pairs), _mean(y for _, y in pairs)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs))
    return numerator / denominator if denominator else 0.0


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()
    except Exception as exc:
        return "unavailable: %s" % exc


def _load_profiles(profile_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        payload = _read(profile_root / ("profile_%s.json" % scene), {}) or {}
        rows.extend(payload.get("rows", []))
    return rows


def _runtime_artifacts(rows: Sequence[Mapping[str, Any]], ocmc_rows: Sequence[Mapping[str, Any]]) -> None:
    breakdown: List[Dict[str, Any]] = []
    micro: List[Dict[str, Any]] = []
    four_scene: List[Dict[str, Any]] = []
    memory: List[Dict[str, Any]] = []
    ocmc_baseline: List[Dict[str, Any]] = []
    for row in rows:
        total = _number(row.get("full_train_like_step_ms"))
        ref_control = _number(row.get("raoc_control_ms_median"))
        item = {
            "scene": row.get("scene"),
            "step": row.get("step"),
            "backend": row.get("backend"),
            "gaussian_count": row.get("gaussian_count"),
            "num_intersects": row.get("num_intersects"),
            "height": row.get("height"),
            "width": row.get("width"),
            "raoc_control_ms": ref_control,
            "renderer_forward_ms": row.get("renderer_forward_ms"),
            "backward_ms": row.get("backward_ms"),
            "forward_backward_ms": row.get("full_train_like_step_ms"),
            "raoc_control_fraction_of_measured_step": ref_control / total if math.isfinite(ref_control) and math.isfinite(total) and total else float("nan"),
            "peak_allocated_MB": row.get("peak_allocated_MB"),
            "peak_reserved_MB": row.get("peak_reserved_MB"),
            "allocated_before_MB": row.get("allocated_before_MB"),
            "allocated_after_render_MB": row.get("allocated_after_render_MB"),
            "allocated_after_backward_MB": row.get("allocated_after_backward_MB"),
            "loss": row.get("loss"),
        }
        micro.append(item)
        memory.append(item)
        if row.get("backend") == "reference":
            breakdown.append({
                **item,
                "ordinary_renderer_forward_not_isolated": True,
                "interpretation": "renderer_forward_ms includes the reference RAOC control path and final render; raoc_control_ms is separately measured with CUDA events",
            })
    by_key = {
        (row.get("scene"), row.get("step"), row.get("backend")): row
        for row in rows
    }
    ocmc_by_key = {
        (row.get("scene"), row.get("step")): row
        for row in ocmc_rows
        if row.get("backend") == "ocmc"
    }
    for row in ocmc_rows:
        ocmc_baseline.append({
            "scene": row.get("scene"),
            "step": row.get("step"),
            "backend": "ocmc",
            "gaussian_count": row.get("gaussian_count"),
            "num_intersects": row.get("num_intersects"),
            "renderer_forward_ms": row.get("renderer_forward_ms"),
            "backward_ms": row.get("backward_ms"),
            "forward_backward_ms": row.get("full_train_like_step_ms"),
            "peak_allocated_MB": row.get("peak_allocated_MB"),
            "peak_reserved_MB": row.get("peak_reserved_MB"),
            "allocated_before_MB": row.get("allocated_before_MB"),
            "allocated_after_render_MB": row.get("allocated_after_render_MB"),
            "allocated_after_backward_MB": row.get("allocated_after_backward_MB"),
            "loss": row.get("loss"),
            "raoc_control_not_applicable": True,
        })
    for scene in SCENES:
        for step in STEPS:
            ref = by_key.get((scene, step, "reference"), {})
            fused = by_key.get((scene, step, "cuda_fused"), {})
            ocmc = ocmc_by_key.get((scene, step), {})
            if not ref or not fused:
                continue
            ref_step, fused_step = _number(ref.get("full_train_like_step_ms")), _number(fused.get("full_train_like_step_ms"))
            ocmc_step = _number(ocmc.get("full_train_like_step_ms"))
            ref_peak, fused_peak = _number(ref.get("peak_allocated_MB")), _number(fused.get("peak_allocated_MB"))
            ref_res, fused_res = _number(ref.get("peak_reserved_MB")), _number(fused.get("peak_reserved_MB"))
            four_scene.append({
                "scene": scene,
                "step": step,
                "reference_raoc_control_ms": ref.get("raoc_control_ms_median"),
                "cuda_fused_raoc_control_ms": fused.get("raoc_control_ms_median"),
                "reference_forward_backward_ms": ref.get("full_train_like_step_ms"),
                "cuda_fused_forward_backward_ms": fused.get("full_train_like_step_ms"),
                "fused_over_reference_speedup": ref_step / fused_step if fused_step else float("nan"),
                "ocmc_forward_backward_ms": ocmc.get("full_train_like_step_ms"),
                "reference_over_ocmc_overhead_ms": ref_step - ocmc_step if math.isfinite(ocmc_step) else float("nan"),
                "reference_over_ocmc_overhead_fraction": ref_step / ocmc_step - 1.0 if math.isfinite(ocmc_step) and ocmc_step else float("nan"),
                "reference_over_ocmc_ratio": ref_step / ocmc_step if math.isfinite(ocmc_step) and ocmc_step else float("nan"),
                "cuda_fused_over_ocmc_overhead_ms": fused_step - ocmc_step if math.isfinite(ocmc_step) else float("nan"),
                "cuda_fused_over_ocmc_overhead_fraction": fused_step / ocmc_step - 1.0 if math.isfinite(ocmc_step) and ocmc_step else float("nan"),
                "cuda_fused_over_ocmc_ratio": fused_step / ocmc_step if math.isfinite(ocmc_step) and ocmc_step else float("nan"),
                "reference_peak_allocated_MB": ref_peak,
                "cuda_fused_peak_allocated_MB": fused_peak,
                "peak_allocated_reduction_MB": ref_peak - fused_peak,
                "reference_peak_reserved_MB": ref_res,
                "cuda_fused_peak_reserved_MB": fused_res,
                "peak_reserved_reduction_MB": ref_res - fused_res,
                "gaussian_count": ref.get("gaussian_count"),
                "num_intersects": ref.get("num_intersects"),
                "reference_loss": ref.get("loss"),
                "cuda_fused_loss": fused.get("loss"),
                "loss_abs_diff": abs(_number(ref.get("loss")) - _number(fused.get("loss"))),
            })
    _csv(OUTPUT_ROOT / "reference_runtime_breakdown.csv", breakdown)
    _write(OUTPUT_ROOT / "reference_runtime_breakdown.json", {"rows": breakdown, "scope": "read-only archived C1 checkpoint profile"})
    _csv(OUTPUT_ROOT / "microbenchmark.csv", micro)
    _write(OUTPUT_ROOT / "microbenchmark.json", {"rows": micro, "warmup": 3, "timed_repeats": 5, "timing": "torch.cuda.Event"})
    _csv(OUTPUT_ROOT / "four_scene_runtime.csv", four_scene)
    _write(OUTPUT_ROOT / "four_scene_runtime.json", {"rows": four_scene})
    _csv(OUTPUT_ROOT / "ocmc_baseline.csv", ocmc_baseline)
    _write(OUTPUT_ROOT / "ocmc_baseline.json", {
        "rows": ocmc_baseline,
        "scope": "read-only archived C1 checkpoint profile with the OCMC projector enabled and RAOC disabled",
        "raoc_control_timing": "not applicable; OCMC does not execute the per-ray RAOC control path",
    })
    _csv(OUTPUT_ROOT / "four_scene_memory.csv", four_scene)
    _write(OUTPUT_ROOT / "four_scene_memory.json", {"rows": four_scene})
    _csv(OUTPUT_ROOT / "reference_memory_trace.csv", memory)
    _write(OUTPUT_ROOT / "reference_memory_trace.json", {"rows": memory, "trace_source": "profile checkpoint states; no empty_cache during timed model step"})


def _smoke_artifacts() -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    jitter: List[Dict[str, Any]] = []
    for scene in SCENES:
        payload = _read(SMOKE_ROOT / ("smoke_%s.json" % scene), {}) or {}
        for backend_name in ("reference", "cuda_fused"):
            backend = payload.get(backend_name, {})
            rows = backend.get("rows", [])
            allocated = [row.get("allocated_MB") for row in rows]
            reserved = [row.get("reserved_MB") for row in rows]
            summaries.append({
                "scene": scene,
                "backend": backend_name,
                "steps": backend.get("steps", 0),
                "all_finite": backend.get("all_finite", False),
                "gaussian_count_constant": backend.get("gaussian_count_constant", False),
                "peak_allocated_MB": backend.get("peak_allocated_MB"),
                "peak_reserved_MB": backend.get("peak_reserved_MB"),
                "allocated_p50_MB": _percentile(allocated, 0.50),
                "allocated_p90_MB": _percentile(allocated, 0.90),
                "allocated_p95_MB": _percentile(allocated, 0.95),
                "allocated_p99_MB": _percentile(allocated, 0.99),
                "allocated_max_MB": _percentile(allocated, 1.0),
                "reserved_p50_MB": _percentile(reserved, 0.50),
                "reserved_p90_MB": _percentile(reserved, 0.90),
                "reserved_p95_MB": _percentile(reserved, 0.95),
                "reserved_p99_MB": _percentile(reserved, 0.99),
                "reserved_max_MB": _percentile(reserved, 1.0),
            })
        matched = payload.get("matched_rows", [])
        jitter.append({
            "scene": scene,
            "backend": "matched_20_step",
            "all_finite": payload.get("all_finite", False),
            "gaussian_count_match_all": payload.get("gaussian_count_match_all", False),
            "max_loss_abs_diff": payload.get("max_loss_abs_diff"),
            "steps": payload.get("steps"),
            "reference_allocated_p50_MB": _percentile((row.get("reference_allocated_MB") for row in matched), 0.50),
            "fused_allocated_p50_MB": _percentile((row.get("fused_allocated_MB") for row in matched), 0.50),
        })
    _csv(OUTPUT_ROOT / "four_scene_smoke.csv", summaries)
    _write(OUTPUT_ROOT / "four_scene_smoke.json", {"rows": summaries, "matched": jitter})
    _csv(OUTPUT_ROOT / "memory_benchmark.csv", summaries)
    _write(OUTPUT_ROOT / "memory_benchmark.json", {"rows": summaries, "jitter_rows": jitter, "sequence_length": 20})
    return {"rows": summaries, "jitter": jitter}


def _compatibility() -> Dict[str, Any]:
    contract = _read(EQUIV_ROOT / "contract_audit.json", {}) or {}
    legacy = _read(EQUIV_ROOT / "legacy_nonraoc.json", {}) or {}
    model = _read(EQUIV_ROOT / "model_iui3_step3000.json", {}) or {}
    direct_old = _read(REPO_ROOT / "outputs/raoc_ray_adaptive_observability_preflight_20260827/gradient_pathway.json", {}) or {}
    forward_files = [EQUIV_ROOT / ("%s-step3000-fast-final.json" % scene) for scene in SCENES]
    forward_payloads = [_read(path, {}) or {} for path in forward_files]
    rows: List[Dict[str, Any]] = []
    for payload in forward_payloads:
        for row in payload.get("forward_rows", []):
            rows.append({"scene": payload.get("scene"), **row})
    max_by_quantity: Dict[str, float] = {}
    for row in rows:
        max_by_quantity[row["quantity"]] = max(max_by_quantity.get(row["quantity"], 0.0), _number(row.get("max_abs_diff")))
    _csv(OUTPUT_ROOT / "forward_equivalence.csv", rows)
    forward = {
        "rows": rows,
        "max_abs_diff_by_quantity": max_by_quantity,
        "strict_target": 1e-6,
        "strict_pass": all(value <= 1e-6 for value in max_by_quantity.values()),
        "model_level": model.get("output_diffs", {}),
        "model_level_strict_pass": model.get("model_level_strict_forward_pass", False),
    }
    _write(OUTPUT_ROOT / "forward_equivalence.json", forward)
    backward = {
        "scene_audits": [{"scene": payload.get("scene"), "backward_max_abs_diff": payload.get("backward_max_abs_diff"), "backward_mean_abs_diff": payload.get("backward_mean_abs_diff"), "strict_pass": payload.get("backward_pass", False)} for payload in forward_payloads],
        "model_level": model.get("gradient_diffs", {}),
        "medium_mlp_relative_l2": model.get("gradient_diffs", {}).get("medium_mlp", {}).get("relative_l2"),
        "model_level_strict_pass": model.get("model_level_strict_backward_pass", False),
        "strict_target": 1e-6,
    }
    _write(OUTPUT_ROOT / "backward_equivalence.json", backward)
    direct = {
        "optimization_contract": {key: contract.get(key) for key in ("direct_medium_grad_nonzero", "direct_gaussian_grad_l2", "diagnostic_outputs_detached", "second_order_gate_graph_retained", "second_order_gate_operations")},
        "historical_reference": direct_old.get("direct_raoc_gradient_stats", {}),
        "medium_mlp_reference_grad_l2": direct_old.get("direct_raoc_gradient_stats", {}).get("medium_mlp", {}).get("grad_l2"),
        "gaussian_reference_grad_l2_sum": direct_old.get("direct_mechanism_gaussian_grad_l2_sum"),
        "pass": bool(contract.get("direct_medium_grad_nonzero")) and float(contract.get("direct_gaussian_grad_l2", 1.0)) == 0.0,
    }
    _write(OUTPUT_ROOT / "direct_gradient_equivalence.json", direct)
    checkpoint = {
        "old_pre_raoc_checkpoint": legacy.get("old_pre_raoc_checkpoint", {}),
        "old_calibrated_raoc_state_load_pass": legacy.get("old_calibrated_raoc_state_load_pass", False),
        "disabled_path": legacy.get("nonraoc", {}).get("disabled_path", {}),
        "ocmc_path_repeatability": legacy.get("nonraoc", {}).get("ocmc_path_repeatability", {}),
        "old_checkpoint_and_state_compatible": bool(legacy.get("old_pre_raoc_checkpoint", {}).get("pass")) and bool(legacy.get("old_calibrated_raoc_state_load_pass", False)),
    }
    _write(OUTPUT_ROOT / "checkpoint_compatibility.json", checkpoint)
    return {"contract": contract, "legacy": legacy, "model": model, "forward": forward, "backward": backward, "direct": direct, "checkpoint": checkpoint}


def _memory_cause(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    profiles = [row for row in rows if row.get("backend") == "reference"]
    by_scene: Dict[str, List[Mapping[str, Any]]] = {scene: [row for row in profiles if row.get("scene") == scene] for scene in SCENES}
    scene_corr = {}
    for scene, values in by_scene.items():
        scene_corr[scene] = {
            "gaussian_count_vs_peak_allocated": _pearson((row.get("gaussian_count") for row in values), (row.get("peak_allocated_MB") for row in values)),
            "intersection_vs_peak_allocated": _pearson((row.get("num_intersects") for row in values), (row.get("peak_allocated_MB") for row in values)),
        }
    return {
        "classification": "MIXED",
        "evidence": {
            "profile_state_count": len(profiles),
            "gaussian_count_range": [min((int(row.get("gaussian_count")) for row in profiles), default=0), max((int(row.get("gaussian_count")) for row in profiles), default=0)],
            "peak_allocated_range_MB": [min((_number(row.get("peak_allocated_MB")) for row in profiles), default=float("nan")), max((_number(row.get("peak_allocated_MB")) for row in profiles), default=float("nan"))],
            "gaussian_count_vs_peak_allocated_all_states": _pearson((row.get("gaussian_count") for row in profiles), (row.get("peak_allocated_MB") for row in profiles)),
            "intersection_vs_peak_allocated_all_states": _pearson((row.get("num_intersects") for row in profiles), (row.get("peak_allocated_MB") for row in profiles)),
            "per_scene_correlations": scene_corr,
        },
        "interpretation": "Gaussian topology establishes the persistent base/workload; visible/tile intersections and renderer workspace add view-dependent variation; RAOC temporary tensors contribute an avoidable peak; reserved memory also reflects the PyTorch caching allocator.",
        "empty_cache_policy": "not called inside normal timed model iterations; only cleanup between independent checkpoint runs",
    }


def _state_and_config(compat: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], smoke: Mapping[str, Any]) -> None:
    extension = REPO_ROOT / "water_splatting" / "csrc.so"
    environment = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "visible_device_count_in_summary_process": torch.cuda.device_count(),
        "allowed_physical_gpus": [6, 7, 8, 9],
        "build": {
            "production": "RAOC_PRECISE_MATH=0 CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH python setup.py build_ext --inplace --force",
            "precise_audit": "RAOC_PRECISE_MATH=1 CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH python setup.py build_ext --inplace",
            "production_flag": "--use_fast_math",
            "precise_flag": "--fmad=false --prec-div=true --prec-sqrt=true --ftz=false",
            "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest() if extension.is_file() else "missing",
        },
    }
    _write(OUTPUT_ROOT / "environment.json", environment)
    _write(OUTPUT_ROOT / "repo_state.json", {
        "starting_branch": "research/m1-bounded-intrinsic",
        "starting_head": "4cd40a6",
        "current_branch": _git("branch", "--show-current"),
        "current_head": _git("rev-parse", "HEAD"),
        "status_short": _git("status", "--short"),
        "git_diff_check": _git("diff", "--check"),
        "tracked_large_output_count": len([line for line in _git("ls-files", "outputs", "renders", "logs", "common_masks", "checkpoints").splitlines() if line.strip()]),
        "existing_formal_jobs_on_gpus_6_9": False,
        "existing_jobs_left_untouched": True,
        "historical_gmvc_untouched": True,
        "new_formal_15k_experiment": False,
    })
    _write(OUTPUT_ROOT / "current_raoc_semantics.json", {
        "equations_unchanged": True,
        "camera_residual": "delta_z_cam = z_full - z_base",
        "standardization": "delta_z_std = delta_z_cam / standardization_scale",
        "modal_coefficient": "a = V^T delta_z_std",
        "local_sensitivity": "s_i = ||J_p v_i||_2",
        "evidence": "e_i = abs(a_i) * s_i",
        "local_gate": "g_local = e^2 / (e^2 + q^2)",
        "global_gate": "g_obs = sigma^2 / (sigma^2 + median(sigma)^2)",
        "keep_gate": "g_keep = 1 - (1-g_obs)(1-g_local)",
        "reconstruction": "delta_raoc_std = V (g_keep * a)",
        "q_is_external_input": True,
        "q50_q80_supported": True,
        "all_nine_modes_generic": True,
        "detached_control": ["V", "sigma", "g_obs", "q", "a", "Jv", "e", "g_local", "g_keep"],
        "no_mode_skipping": True,
        "no_mixed_precision_change": True,
    })
    _write(OUTPUT_ROOT / "fused_backend_config.json", {
        "backend_name": "cuda_fused",
        "reference_backend_retained": True,
        "default_backend": "reference",
        "standalone_cuda_operator": True,
        "renderer_primary_pass_integrated": False,
        "fused_operations": ["compositor local directional sensitivity", "modal coefficient projection", "evidence", "local gate", "keep gate", "modal reconstruction"],
        "full_global_jacobian_materialized": False,
        "full_jv_saved_for_backward": False,
        "backward_state": ["detached basis", "detached keep_gate"],
        "diagnostics_materialized_currently": ["evidence", "local_gate", "keep_gate", "sensitivity"],
        "production_math": "fast-math build; strict audit still uses reference comparisons",
    })
    _write(OUTPUT_ROOT / "kernel_allocation_summary.json", {
        "reference_large_intermediate": "[N, 9, 3] analytic Jv action tensor, chunked then concatenated",
        "fused_large_intermediate": "no [N,9,3] Jv tensor; per-ray derivative primitives and 9-mode arrays remain in registers",
        "eliminated_estimated_bytes_per_ray": {"jv_9x3_fp32": 108, "full_jacobian_3x9_fp32_if_materialized": 108},
        "remaining_fused_diagnostic_bytes_per_ray": {"delta_out_evidence_local_gate_keep_gate_sensitivity": 5 * 9 * 4},
        "launch_reduction": "many PyTorch compositor/control kernels to one RAOC CUDA kernel plus existing renderer work",
        "allocation_summary_is_estimate": True,
    })
    _write(OUTPUT_ROOT / "memory_cause_analysis.json", _memory_cause(rows))


def run() -> Dict[str, Any]:
    rows = _load_profiles(PROFILE_ROOT)
    ocmc_rows = _load_profiles(OCMC_PROFILE_ROOT)
    _runtime_artifacts(rows, ocmc_rows)
    smoke = _smoke_artifacts()
    compat = _compatibility()
    _state_and_config(compat, rows, smoke)
    speedups = [_number(row.get("fused_over_reference_speedup")) for row in _read(OUTPUT_ROOT / "four_scene_runtime.json", {}).get("rows", [])]
    classification = {
        "classification": "RAOC_CUDA_OPTIMIZATION_NOT_READY",
        "ready": False,
        "reason": "strict forward and backward equivalence targets are not met on the production fast-math fused compositor; reference remains the recommended formal backend",
        "strict_forward_pass": compat["forward"].get("strict_pass", False) and compat["model"].get("model_level_strict_forward_pass", False),
        "strict_backward_pass": compat["model"].get("model_level_strict_backward_pass", False),
        "matched_smoke_all_finite": all(row.get("all_finite", False) for row in smoke.get("rows", [])),
        "matched_smoke_gaussian_count_match": all(row.get("gaussian_count_match_all", False) for row in smoke.get("jitter", [])),
        "mean_four_scene_speedup": _mean(speedups),
        "per_scene_speedup_at_14999": {
            scene: next((row.get("fused_over_reference_speedup") for row in _read(OUTPUT_ROOT / "four_scene_runtime.json", {}).get("rows", []) if row.get("scene") == scene and row.get("step") == 14999), None)
            for scene in SCENES
        },
        "forward_max_abs": compat["forward"].get("max_abs_diff_by_quantity", {}),
        "model_pred_image_max_abs": compat["model"].get("output_diffs", {}).get("pred_image", {}).get("max_abs"),
        "medium_mlp_gradient_relative_l2": compat["backward"].get("medium_mlp_relative_l2"),
        "ocmc_profile_rows": len(ocmc_rows),
        "ocmc_step_14999_full_train_like_ms": {
            scene: next((row.get("full_train_like_step_ms") for row in ocmc_rows if row.get("scene") == scene and row.get("step") == 14999), None)
            for scene in SCENES
        },
        "new_nan_inf": False,
        "remaining_bottleneck": "strict compositor floating-point equivalence and renderer/operator boundary; fused control is standalone rather than inside the primary renderer pass",
        "recommended_formal_backend": "reference",
    }
    _write(OUTPUT_ROOT / "final_classification.json", classification)
    _write(OUTPUT_ROOT / "final_summary.json", {
        "classification": classification,
        "compatibility": compat,
        "memory_jitter": smoke,
        "profile_rows": len(rows),
        "ocmc_profile_rows": len(ocmc_rows),
        "formal_training_started": False,
    })
    return classification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
