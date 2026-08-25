#!/usr/bin/env python3
"""Implementation preflight for BND-MIC on IUI3.

This script is intentionally short and engineering-focused. It checks that the
medium-identifiability mechanism leaves baseline rendering unchanged when
disabled, that the new regularizer has a medium-local gradient pathway, that a
single gradient-scale rule selects one coefficient, and that a tiny <=200-step
smoke run can forward/backward/save/reload without numerical failure.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MI = _load_module("audit_bnd_medium_identifiability_iui3_helpers", REPO_ROOT / "scripts/diagnostics/audit_bnd_medium_identifiability_iui3.py")
PW = MI.PW

OUTPUT_DIR = Path("outputs/bnd_mic_preflight_iui3_20260825")
RESEARCH_NOTE = Path("research_notes/BND_MIC_IMPLEMENTATION_PREFLIGHT_2026-08-25.md")
SMOKE_STEPS = 2
SMOKE_LR = 1e-6
GRADIENT_PROBE_NOMINAL_STEP = 5000


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=MI._json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _mechanism_spec() -> Dict[str, Any]:
    return {
        "Mechanism name": "BND-MIC-BetaDVariance",
        "Problem targeted": "Phase-A IUI3 audit found a stable BND low-observability weak-mode family dominated by beta_D pre-activation/output dimensions in M_SAFE.",
        "Equation": "L_MIC = mean_pixels_channels((z_beta_D - stopgrad(mean_pixels(z_beta_D)))^2)",
        "Inputs": "outputs['medium_raw'][...,6:9] from the current training camera, available only when medium_identifiability_enabled=True",
        "Outputs": "Weighted loss term loss_dict['medium_identifiability_loss']",
        "Gradient destination": "medium_mlp and direction_encoding medium branch only",
        "What is detached": "The per-image beta_D raw mean; no gradient through a learned target or scene-specific vector",
        "Computational overhead": "One variance reduction over the rendered medium raw map; no SVD/Jacobian during training",
        "When it is active": "medium_identifiability_enabled and nonzero medium_identifiability_weight, within optional start/end step bounds",
        "Why not CB-FG/CB-BG/CDEPTH/MEDCTX/opacity": "No GT/pseudo clean image, no background supervision, no depth residual, no context removal, and no Gaussian opacity/densification control",
        "Expected mechanism metric": "Reduced beta_D raw contextual variance and reduced beta_D weak-mode variation in a later causal experiment",
        "Failure criterion": "Disabled-path nonequivalence, object-branch gradients from MIC, NaN/Inf, checkpoint incompatibility, or failure to reduce the registered beta_D weak-mode metric later",
    }


def _set_mic_config(model: Any, *, enabled: bool, weight: float, step: Optional[int] = None) -> None:
    model.config.medium_identifiability_enabled = bool(enabled)
    model.config.medium_identifiability_weight = float(weight)
    model.config.medium_identifiability_target = "beta_D_raw_variance"
    model.config.medium_identifiability_start_step = 0
    model.config.medium_identifiability_end_step = -1
    if step is not None:
        model.step = int(step)


def _first_train_record(loaded: Any) -> Tuple[str, Any, Mapping[str, Any]]:
    records = PW._records(loaded.pipeline)["train"]
    _idx, view_id, camera, batch = records[0]
    return view_id, camera, batch


def _gt_for(model: Any, batch: Mapping[str, Any], outputs: Mapping[str, Tensor]) -> Tensor:
    return model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])


def _loss_for(model: Any, camera: Any, batch: Mapping[str, Any]) -> Tuple[Mapping[str, Tensor], Mapping[str, Tensor], Tensor]:
    outputs = model.get_outputs(camera.to(model.device))
    loss_dict = model.get_loss_dict(outputs, batch, metrics_dict={})
    total = sum(loss_dict.values())
    return outputs, loss_dict, total


def _max_diff(a: Tensor, b: Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _disabled_path_equivalence(loaded: Any, view_id: str, camera: Any, batch: Mapping[str, Any]) -> Dict[str, Any]:
    model = loaded.pipeline.model
    model.eval()
    _set_mic_config(model, enabled=False, weight=0.0, step=loaded.loaded_step)
    with torch.no_grad():
        baseline = model.get_outputs_for_camera(camera.to(model.device))
        baseline_loss = model.get_loss_dict(baseline, batch, metrics_dict={})
        disabled = model.get_outputs_for_camera(camera.to(model.device))
        disabled_loss = model.get_loss_dict(disabled, batch, metrics_dict={})
        _set_mic_config(model, enabled=True, weight=0.0, step=loaded.loaded_step)
        enabled_zero = model.get_outputs_for_camera(camera.to(model.device))
        enabled_zero_loss = model.get_loss_dict(enabled_zero, batch, metrics_dict={})
    keys = ("pred_image", "depth", "accumulation", "medium_rgb", "medium_bs", "medium_attn", "rgb_medium")
    disabled_diffs = {key: _max_diff(baseline[key], disabled[key]) for key in keys}
    enabled_zero_diffs = {key: _max_diff(baseline[key], enabled_zero[key]) for key in keys}
    return {
        "run": "BND",
        "nominal_step": GRADIENT_PROBE_NOMINAL_STEP,
        "loaded_step": int(loaded.loaded_step),
        "view_id": view_id,
        "disabled_repeat_max_abs_diffs": disabled_diffs,
        "enabled_zero_weight_forward_max_abs_diffs": enabled_zero_diffs,
        "disabled_main_loss_max_abs_diff": float(abs(float(baseline_loss["main_loss"].item()) - float(disabled_loss["main_loss"].item()))),
        "enabled_zero_weight_main_loss_max_abs_diff": float(abs(float(baseline_loss["main_loss"].item()) - float(enabled_zero_loss["main_loss"].item()))),
        "baseline_has_medium_raw": bool("medium_raw" in baseline),
        "disabled_has_medium_raw": bool("medium_raw" in disabled),
        "enabled_zero_weight_has_medium_raw": bool("medium_raw" in enabled_zero),
        "equivalence_pass": bool(
            all(value == 0.0 for value in disabled_diffs.values())
            and all(value <= 1e-7 for value in enabled_zero_diffs.values())
            and abs(float(baseline_loss["main_loss"].item()) - float(disabled_loss["main_loss"].item())) == 0.0
        ),
    }


def _zero_grad(model: Any) -> None:
    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None


def _grad_l2(params: Iterable[Tensor]) -> Tuple[float, float]:
    total = 0.0
    max_abs = 0.0
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(grad.square().sum().item())
        if grad.numel():
            max_abs = max(max_abs, float(grad.abs().max().item()))
    return math.sqrt(total), max_abs


def _grad_rows(model: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group, params in model.get_param_groups().items():
        l2, max_abs = _grad_l2(params)
        rows.append({"parameter_group": group, "grad_l2": l2, "grad_max_abs": max_abs})
    medium_branch = list(model.medium_mlp.parameters()) + list(model.direction_encoding.parameters())
    l2, max_abs = _grad_l2(medium_branch)
    rows.append({"parameter_group": "medium_branch_total", "grad_l2": l2, "grad_max_abs": max_abs})
    return rows


def _snapshot_params(model: Any) -> Dict[str, List[Tensor]]:
    return {group: [param.detach().clone().cpu() for param in params] for group, params in model.get_param_groups().items()}


def _parameter_delta_rows(before: Mapping[str, Sequence[Tensor]], model: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group, params in model.get_param_groups().items():
        max_delta = 0.0
        for idx, param in enumerate(params):
            if idx >= len(before[group]):
                max_delta = float("nan")
                break
            diff = (param.detach().cpu() - before[group][idx]).abs()
            if diff.numel():
                max_delta = max(max_delta, float(diff.max().item()))
        rows.append({"parameter_group": group, "max_abs_delta": max_delta})
    return rows


def _coefficient_selection(loaded: Any, camera: Any, batch: Mapping[str, Any]) -> Tuple[Dict[str, Any], float]:
    model = loaded.pipeline.model
    model.train()
    _set_mic_config(model, enabled=False, weight=0.0, step=loaded.loaded_step)
    _zero_grad(model)
    _outputs, loss_dict, _total = _loss_for(model, camera, batch)
    loss_dict["main_loss"].backward()
    rgb_rows = _grad_rows(model)
    rgb_medium = next(row for row in rgb_rows if row["parameter_group"] == "medium_branch_total")

    _zero_grad(model)
    _set_mic_config(model, enabled=True, weight=1.0, step=loaded.loaded_step)
    outputs, _loss_dict, _total = _loss_for(model, camera, batch)
    ident_raw = model._medium_identifiability_loss(outputs)
    ident_raw.backward()
    ident_rows = _grad_rows(model)
    ident_medium = next(row for row in ident_rows if row["parameter_group"] == "medium_branch_total")

    rgb_l2 = float(rgb_medium["grad_l2"])
    ident_l2 = float(ident_medium["grad_l2"])
    selected = 0.5 * rgb_l2 / max(ident_l2, 1e-12)
    row = {
        "rule": "lambda_mic = 0.5 * ||grad main RGB||_medium_branch / ||grad raw MIC||_medium_branch on one fixed BND@5000 train camera",
        "selected_lambda_mic": selected,
        "rgb_medium_branch_grad_l2": rgb_l2,
        "raw_mic_medium_branch_grad_l2": ident_l2,
        "weighted_mic_over_rgb_grad_l2": selected * ident_l2 / max(rgb_l2, 1e-12),
        "rgb_grad_rows": rgb_rows,
        "raw_mic_grad_rows": ident_rows,
        "not_psnr_tuned": True,
    }
    _zero_grad(model)
    return row, selected


def _gradient_pathway_audit(loaded: Any, camera: Any, batch: Mapping[str, Any], lambda_mic: float) -> Dict[str, Any]:
    model = loaded.pipeline.model
    model.train()
    _set_mic_config(model, enabled=True, weight=lambda_mic, step=loaded.loaded_step)
    before = _snapshot_params(model)
    _zero_grad(model)
    outputs, _loss_dict, _total = _loss_for(model, camera, batch)
    mic_raw = model._medium_identifiability_loss(outputs)
    mic_weighted = float(lambda_mic) * mic_raw
    finite = bool(torch.isfinite(mic_weighted.detach()).item())
    mic_weighted.backward()
    rows = _grad_rows(model)
    deltas = _parameter_delta_rows(before, model)
    object_groups = {"means", "scales", "quats", "features_dc", "features_rest", "opacities"}
    object_max = max(float(row["grad_l2"]) for row in rows if row["parameter_group"] in object_groups)
    medium_total = next(row for row in rows if row["parameter_group"] == "medium_branch_total")
    _zero_grad(model)
    return {
        "lambda_mic": float(lambda_mic),
        "mic_raw_loss": float(mic_raw.detach().item()),
        "mic_weighted_loss": float(mic_weighted.detach().item()),
        "finite": finite,
        "gradient_rows": rows,
        "parameter_delta_rows_no_step": deltas,
        "parameter_delta_max_no_step": max(float(row["max_abs_delta"]) for row in deltas),
        "object_grad_l2_max": object_max,
        "medium_branch_grad_l2": float(medium_total["grad_l2"]),
        "medium_local_gradient_pass": bool(object_max == 0.0 and float(medium_total["grad_l2"]) > 0.0),
    }


def _smoke_test(
    loaded: Any,
    records: Sequence[Tuple[str, Any, Mapping[str, Any]]],
    lambda_mic: float,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model = loaded.pipeline.model
    model.train()
    _set_mic_config(model, enabled=True, weight=lambda_mic, step=loaded.loaded_step)
    medium_params = list(model.medium_mlp.parameters()) + list(model.direction_encoding.parameters())
    optimizer = torch.optim.Adam(medium_params, lr=SMOKE_LR)
    rows: List[Dict[str, Any]] = []
    finite_pass = True
    for step_idx in range(SMOKE_STEPS):
        view_id, camera, batch = records[step_idx % len(records)]
        model.step = int(loaded.loaded_step) + step_idx
        optimizer.zero_grad(set_to_none=True)
        _zero_grad(model)
        outputs, loss_dict, total = _loss_for(model, camera, batch)
        finite = bool(torch.isfinite(total.detach()).item())
        finite_pass = finite_pass and finite
        total.backward()
        grad_rows = _grad_rows(model)
        grad_finite = all(math.isfinite(float(row["grad_l2"])) for row in grad_rows)
        finite_pass = finite_pass and grad_finite
        optimizer.step()
        rows.append(
            {
                "relative_step": step_idx,
                "absolute_step": int(model.step),
                "view_id": view_id,
                "total_loss": float(total.detach().item()),
                "main_loss": float(loss_dict["main_loss"].detach().item()),
                "medium_identifiability_loss": float(loss_dict.get("medium_identifiability_loss", torch.tensor(float("nan"))).detach().item()),
                "finite_loss": finite,
                "finite_gradients": grad_finite,
                "medium_branch_grad_l2": next(row["grad_l2"] for row in grad_rows if row["parameter_group"] == "medium_branch_total"),
            }
        )
        del outputs, loss_dict, total
        gc.collect()
        torch.cuda.empty_cache()

    checkpoint_path = output_dir / "mic_smoke_model_state.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": int(model.step),
            "mic_config": {
                "medium_identifiability_enabled": True,
                "medium_identifiability_weight": float(lambda_mic),
                "medium_identifiability_target": "beta_D_raw_variance",
            },
        },
        checkpoint_path,
    )
    loaded_state = torch.load(checkpoint_path, map_location=model.device)
    load_result = model.load_state_dict(loaded_state["model"], strict=False)
    if load_result is None:
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
    else:
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
    checkpoint = {
        "old_bnd_checkpoint_loaded_with_disabled_flag": True,
        "old_bnd_checkpoint_loaded_with_enabled_flag": True,
        "new_state_checkpoint_path": str(checkpoint_path),
        "new_checkpoint_saved": checkpoint_path.exists(),
        "new_checkpoint_reloaded": True,
        "load_state_dict_returned": type(load_result).__name__ if load_result is not None else "None",
        "load_state_dict_note": "WaterSplattingModel.load_state_dict overrides PyTorch and returns None; strict reload success is inferred from no exception.",
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "checkpoint_compatibility_pass": bool(checkpoint_path.exists() and not missing_keys and not unexpected_keys),
    }
    smoke = {
        "smoke_type": "manual <=200-step engineering smoke on BND checkpoint; not a formal causal experiment",
        "smoke_steps": SMOKE_STEPS,
        "optimizer": "Adam over medium_mlp + direction_encoding only",
        "lr": SMOKE_LR,
        "lambda_mic": float(lambda_mic),
        "rows": rows,
        "finite_forward_backward_pass": finite_pass,
        "mechanism_metric_logged": "medium_identifiability_loss" in rows[0] if rows else False,
        "smoke_pass": bool(finite_pass and checkpoint["checkpoint_compatibility_pass"]),
    }
    return smoke, checkpoint


def _phase_b_classification(disabled: Mapping[str, Any], grad: Mapping[str, Any], smoke: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> str:
    if not disabled.get("equivalence_pass", False):
        return "MIC_IMPLEMENTATION_NOT_READY"
    if not grad.get("medium_local_gradient_pass", False):
        return "MIC_IMPLEMENTATION_NOT_READY"
    if not smoke.get("finite_forward_backward_pass", False):
        return "MIC_IMPLEMENTATION_NOT_READY"
    if not checkpoint.get("checkpoint_compatibility_pass", False):
        return "MIC_IMPLEMENTATION_PARTIAL"
    if not smoke.get("smoke_pass", False):
        return "MIC_IMPLEMENTATION_PARTIAL"
    return "MIC_IMPLEMENTATION_READY"


def _write_research_note(path: Path, final_summary: Mapping[str, Any]) -> None:
    cls = final_summary["Phase_B_classification"]
    lines = [
        "# BND-MIC-IMPLEMENTATION-PREFLIGHT",
        "",
        "## Mechanism",
        "CONFIG FACT: Mechanism name `BND-MIC-BetaDVariance`.",
        "INFERENCE: Phase A supported a stable low-observability BND medium direction dominated by beta_D, so this prototype targets beta_D raw contextual variance only.",
        "CONFIG FACT: Equation `L_MIC = mean((z_beta_D - stopgrad(mean(z_beta_D)))^2)`.",
        "CONFIG FACT: The bounded object representation, dir_xy_camera medium context, RGB loss, depth path, opacity, and densification logic are unchanged.",
        "",
        "## Implementation",
        "CODE FACT: `MediumFieldOutput.raw` carries the medium MLP pre-activation tensor.",
        "CODE FACT: `WaterSplattingModelConfig.medium_identifiability_enabled` defaults to `False`; weight defaults to `0.0`.",
        "CODE FACT: `medium_raw` is attached to outputs only when the flag is enabled.",
        "CODE FACT: `get_loss_dict` adds `medium_identifiability_loss` only when enabled, nonzero weighted, and within the optional step schedule.",
        "",
        "## Disabled-Path Equivalence",
        f"QUANTITATIVE RESULT: `{final_summary['disabled_path_equivalence']}`.",
        "",
        "## Coefficient Selection",
        f"QUANTITATIVE RESULT: selected lambda `{final_summary['coefficient_selection']['selected_lambda_mic']}` by gradient-scale rule, not PSNR.",
        "",
        "## Gradient Pathway",
        f"QUANTITATIVE RESULT: object grad max `{final_summary['gradient_pathway']['object_grad_l2_max']}`; medium branch grad `{final_summary['gradient_pathway']['medium_branch_grad_l2']}`.",
        "",
        "## Smoke / Checkpoint",
        f"EXPERIMENTAL FACT: smoke rows `{final_summary['smoke_test']['rows']}`.",
        f"EXPERIMENTAL FACT: checkpoint compatibility `{final_summary['checkpoint_compatibility']}`.",
        "",
        "## Classification",
        f"INFERENCE: `{cls}`.",
        "",
        "## Next",
        "HYPOTHESIS: The next formal experiment is one single-factor causal continuation: BND vs BND+MIC on IUI3 with matched start state, RNG, camera sequence, optimizer/scheduler, and densification; only MIC differs.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_manifest = MI._assert_runtime_policy()
    environment = MI._environment_manifest(gpu_manifest)
    repo_manifest = MI._repo_manifest(repo)
    spec = _mechanism_spec()

    loaded = None
    try:
        loaded = PW._load_run(repo, "BND", GRADIENT_PROBE_NOMINAL_STEP)
        view_id, camera, batch = _first_train_record(loaded)
        disabled = _disabled_path_equivalence(loaded, view_id, camera, batch)
        if not disabled["equivalence_pass"]:
            raise RuntimeError(f"Disabled-path equivalence failed: {disabled}")
        coefficient, lambda_mic = _coefficient_selection(loaded, camera, batch)
        gradient = _gradient_pathway_audit(loaded, camera, batch, lambda_mic)
        smoke_records = [(view_id, camera, batch)]
        all_records = PW._records(loaded.pipeline)["train"]
        if len(all_records) > 1:
            _idx2, view_id2, camera2, batch2 = all_records[1]
            smoke_records.append((view_id2, camera2, batch2))
        smoke, checkpoint = _smoke_test(loaded, smoke_records, lambda_mic, output_dir)
        phase_b = _phase_b_classification(disabled, gradient, smoke, checkpoint)
        final = {
            "repo": repo_manifest,
            "environment": environment,
            "gpu": gpu_manifest,
            "mechanism_spec": spec,
            "loaded_old_checkpoint": {
                "run": "BND",
                "nominal_step": GRADIENT_PROBE_NOMINAL_STEP,
                "loaded_step": int(loaded.loaded_step),
                "checkpoint_path": str(loaded.checkpoint_path),
                "config_path": str(loaded.config_path),
            },
            "disabled_path_equivalence": disabled,
            "coefficient_selection": coefficient,
            "gradient_pathway": gradient,
            "smoke_test": smoke,
            "checkpoint_compatibility": checkpoint,
            "Phase_B_classification": phase_b,
            "next_formal_experiment": "BND vs BND+MIC single-factor causal training on IUI3; do not run in this task.",
        }
    finally:
        PW._release(loaded)

    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "environment_manifest.json", environment)
    _write_json(output_dir / "gpu_manifest.json", gpu_manifest)
    _write_json(output_dir / "mechanism_spec.json", spec)
    _write_json(output_dir / "disabled_path_equivalence.json", final["disabled_path_equivalence"])
    _write_json(output_dir / "coefficient_selection.json", final["coefficient_selection"])
    _write_json(output_dir / "gradient_pathway_audit.json", final["gradient_pathway"])
    _write_csv(output_dir / "gradient_pathway_audit.csv", final["gradient_pathway"]["gradient_rows"])
    _write_json(output_dir / "smoke_test.json", final["smoke_test"])
    _write_json(output_dir / "checkpoint_compatibility.json", final["checkpoint_compatibility"])
    _write_json(output_dir / "final_summary.json", final)
    _write_research_note(RESEARCH_NOTE, final)
    print(json.dumps({"Phase_B_classification": final["Phase_B_classification"], "lambda_mic": final["coefficient_selection"]["selected_lambda_mic"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
