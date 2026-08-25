#!/usr/bin/env python3
"""Phase-B OCMC preflight for camera-medium observability control on IUI3.

This is not a formal 15k causal experiment. It only checks engineering,
equivalence, gradient, and short-smoke behavior for a default-off prototype.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_bnd_medium_identifiability_iui3 as MI
from scripts.experiments import run_bnd_mic_causal_iui3 as MIC
from scripts.experiments import run_m1_camera_context_ablation_iui3 as CAM


OUTPUT_DIR = Path("outputs/camera_medium_observability_preflight_iui3_20260825")
PHASE_A_OUTPUT_DIR = Path("outputs/m1_camera_context_identifiability_iui3_20260825")
FINAL_STEP = 14999
SAMPLES_PER_VIEW = 1024
SAMPLE_SEED = 20260825
SMOKE_STEPS = 20
EPS = 1e-12


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


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


def _repo_path(path: Path, repo: Path) -> Path:
    return path if path.is_absolute() else repo / path


def _load_c1(repo: Path, phase_a_output_dir: Path, step: int) -> CAM.BranchState:
    branch = CAM._setup_branch(repo, "C1")
    CAM._load_snapshot(branch, phase_a_output_dir, int(step))
    branch.pipeline.eval()
    branch.pipeline.model.eval()
    return branch


def _first_train_record(branch: CAM.BranchState) -> Tuple[int, str, Any, Dict[str, Any]]:
    idx, view_id, camera, batch = CAM._train_records(branch.pipeline)[0]
    return idx, view_id, camera.to(branch.pipeline.model.device), CAM._batch_to_device(batch, branch.pipeline.model.device)


def _enable_raw_output(model: Any) -> None:
    model.config.medium_identifiability_enabled = True
    model.config.medium_identifiability_weight = 0.0


def _compute_outputs(model: Any, camera: Any, batch: Mapping[str, Any]) -> Tuple[Dict[str, Tensor], Tensor]:
    _enable_raw_output(model)
    outputs = model.get_outputs(camera)
    losses = model.get_loss_dict(outputs, batch, {})
    loss = sum(losses.values())
    return outputs, loss


def _max_diff_rows(
    left: Mapping[str, Tensor],
    right: Mapping[str, Tensor],
    left_loss: Tensor,
    right_loss: Tensor,
    label: str,
) -> List[Dict[str, Any]]:
    key_map = {
        "pred_image": "pred_image",
        "depth": "depth",
        "accumulation": "accumulation",
        "medium_rgb": "medium_rgb",
        "beta_B": "medium_bs",
        "beta_D": "medium_attn",
        "raw_z_med": "medium_raw",
    }
    rows: List[Dict[str, Any]] = []
    for label_name, key in key_map.items():
        diff = float((left[key].detach().float() - right[key].detach().float()).abs().max().cpu().item())
        rows.append({"check": label, "quantity": label_name, "max_abs_diff": diff, "pass": bool(diff == 0.0)})
    loss_diff = float(abs(float(left_loss.detach().cpu().item()) - float(right_loss.detach().cpu().item())))
    rows.append({"check": label, "quantity": "loss", "max_abs_diff": loss_diff, "pass": bool(loss_diff == 0.0)})
    return rows


def _estimate_projector(
    repo: Path,
    output_dir: Path,
    branch: CAM.BranchState,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
    samples, sampling_meta, sampling_rows = MI._build_samples(repo, output_dir, SAMPLES_PER_VIEW, SAMPLE_SEED)
    _write_json(output_dir / "ocmc_sampling_meta.json", sampling_meta)
    _write_csv(output_dir / "ocmc_sampling_rows.csv", sampling_rows)
    analyses, analysis_meta = CAM._analyse_loaded_branch(branch, samples)
    analysis = analyses["M_SAFE"]
    singular = analysis.singular_values_per_sqrt_ray.detach().float().cpu()
    eigvecs = analysis.eigvecs.detach().float().cpu()
    scale = analysis.scale.detach().float().cpu()
    sigma_ref = torch.median(singular).clamp_min(1e-12)
    gates = singular.square() / (singular.square() + sigma_ref.square())
    projector = eigvecs @ torch.diag(gates) @ eigvecs.T
    projector = 0.5 * (projector + projector.T)
    rows = []
    for idx, sigma in enumerate(singular):
        rows.append(
            {
                "mode": idx,
                "sigma_per_sqrt_ray": float(sigma.item()),
                "gate": float(gates[idx].item()),
                "rule": "g_i=sigma_i^2/(sigma_i^2+median(sigma)^2)",
            }
        )
    _write_csv(output_dir / "ocmc_projector_spectrum.csv", rows)
    meta = {
        "population": "M_SAFE",
        "step": int(branch.pipeline.model.step),
        "analysis_meta": analysis_meta,
        "sigma_ref_rule": "median singular value from detached aggregate structured Jacobian",
        "scale_rule": "S_j=max(std(z_med_j),1e-3); camera residual is projected in standardized coordinates and then unstandardized",
        "sigma_ref": float(sigma_ref.item()),
        "projector_finite": bool(torch.isfinite(projector).all().item()),
        "projector_trace": float(torch.trace(projector).item()),
        "projector_fro_norm": float(torch.linalg.norm(projector).item()),
        "min_gate": float(gates.min().item()),
        "max_gate": float(gates.max().item()),
    }
    _write_json(output_dir / "ocmc_projector.json", {"projector": projector, "spectrum": singular, "gates": gates, "scale": scale, **meta})
    return projector, singular, gates, scale, meta


def _mechanism_metrics(outputs: Mapping[str, Tensor], gates: Tensor) -> Dict[str, Any]:
    delta = outputs["camera_medium_delta_raw"].detach().float()
    projected = outputs["camera_medium_delta_projected_raw"].detach().float()
    suppressed = outputs["camera_medium_delta_suppressed_raw"].detach().float()
    delta_sq = float(delta.square().mean().cpu().item())
    projected_sq = float(projected.square().mean().cpu().item())
    suppressed_sq = float(suppressed.square().mean().cpu().item())
    return {
        "camera_residual_rms": math.sqrt(max(delta_sq, 0.0)),
        "projected_residual_rms": math.sqrt(max(projected_sq, 0.0)),
        "suppressed_residual_rms": math.sqrt(max(suppressed_sq, 0.0)),
        "projected_over_full_rms": math.sqrt(max(projected_sq, 0.0)) / max(math.sqrt(max(delta_sq, 0.0)), EPS),
        "suppressed_over_full_rms": math.sqrt(max(suppressed_sq, 0.0)) / max(math.sqrt(max(delta_sq, 0.0)), EPS),
        "gate_min": float(gates.min().item()),
        "gate_median": float(torch.median(gates).item()),
        "gate_max": float(gates.max().item()),
    }


def _param_vector(model: Any) -> Tensor:
    return torch.cat([param.detach().flatten().float().cpu() for param in model.parameters() if param.numel()])


def _gradient_preflight(model: Any, camera: Any, batch: Mapping[str, Any]) -> Dict[str, Any]:
    model.train()
    before = _param_vector(model)
    model.zero_grad(set_to_none=True)
    outputs, _loss = _compute_outputs(model, camera, batch)
    metric = outputs["camera_medium_delta_suppressed_raw"].float().square().mean()
    metric.backward()
    stats = MIC._param_group_grad_stats(model)
    after = _param_vector(model)
    model.zero_grad(set_to_none=True)
    model.eval()
    gaussian_groups = ("means", "scales", "quats", "features_dc", "features_rest", "opacities")
    gaussian_grad_l2 = sum(float(stats[group]["grad_l2"]) for group in gaussian_groups if group in stats)
    return {
        "mechanism_metric": float(metric.detach().cpu().item()),
        "parameter_delta_max_without_optimizer_step": float((after - before).abs().max().item()) if before.numel() else 0.0,
        "medium_mlp_grad_l2": float(stats["medium_mlp"]["grad_l2"]),
        "direction_encoding_grad_l2": float(stats["direction_encoding"]["grad_l2"]),
        "gaussian_grad_l2_sum": gaussian_grad_l2,
        "gaussian_grad_zero_for_direct_mechanism_metric": bool(gaussian_grad_l2 == 0.0),
        "grad_stats": stats,
    }


def _smoke_train(
    branch: CAM.BranchState,
    camera_sequence_path: Path,
    output_dir: Path,
    steps: int,
    gates: Tensor,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model = branch.pipeline.model
    dm = branch.pipeline.datamanager
    cached_train = dm.cached_train
    train_cameras = getattr(dm, "train_cameras", dm.train_dataset.cameras).to(model.device)
    sequence = CAM._read_csv(camera_sequence_path)
    rows: List[Dict[str, Any]] = []
    all_finite = True
    for offset in range(int(steps)):
        seq_row = sequence[offset % len(sequence)]
        camera_index = int(seq_row["camera_index"])
        camera_name = str(seq_row["camera_name"])
        abs_step = FINAL_STEP + 2 + offset
        branch.pipeline.train()
        model.train()
        MIC._run_before(model, branch.optimizers, abs_step)
        branch.optimizers.zero_grad_all()
        batch = CAM._batch_to_device(cached_train[camera_index].copy(), model.device)
        camera = train_cameras[camera_index : camera_index + 1]
        outputs, loss = _compute_outputs(model, camera, batch)
        finite = bool(torch.isfinite(loss).detach().cpu().item())
        all_finite = all_finite and finite
        if not finite:
            raise RuntimeError(f"Non-finite OCMC smoke loss at local step {offset}")
        loss.backward()
        grad_stats = MIC._param_group_grad_stats(model)
        branch.optimizers.optimizer_step_all()
        branch.optimizers.scheduler_step_all(abs_step)
        MIC._run_after(model, branch.optimizers, abs_step)
        metrics = _mechanism_metrics(outputs, gates.detach().cpu())
        rows.append(
            {
                "local_step": offset,
                "absolute_step": abs_step,
                "camera_index": camera_index,
                "camera_name": camera_name,
                "loss": float(loss.detach().cpu().item()),
                "finite": finite,
                "medium_mlp_grad_l2": float(grad_stats["medium_mlp"]["grad_l2"]),
                "gaussian_count": int(model.means.shape[0]),
                **metrics,
            }
        )
    branch.pipeline.eval()
    model.eval()
    ckpt_path = output_dir / "checkpoints" / "ocmc_smoke_step20.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": "CAMERA_MEDIUM_OBSERVABILITY_PREFLIGHT",
            "pipeline": branch.pipeline.state_dict(),
            "optimizers": {group: opt.state_dict() for group, opt in branch.optimizers.optimizers.items()},
            "local_smoke_steps": int(steps),
            "ocmc_projector_is_nonpersistent": True,
        },
        ckpt_path,
    )
    summary = {"smoke_steps": int(steps), "all_losses_finite": bool(all_finite), "checkpoint_path": str(ckpt_path)}
    return rows, summary


def run(repo: Path, output_dir: Path, phase_a_output_dir: Path, step: int, smoke_steps: int) -> Dict[str, Any]:
    gpu = CAM._assert_runtime_policy()
    repo = repo.resolve()
    output_dir = _repo_path(output_dir, repo)
    phase_a_output_dir = _repo_path(phase_a_output_dir, repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "gpu_manifest.json", gpu)
    _write_json(output_dir / "environment_manifest.json", CAM._environment_manifest(gpu))
    _write_json(output_dir / "repo_manifest.json", CAM._repo_manifest(repo))

    branch: Optional[CAM.BranchState] = None
    load_branch: Optional[CAM.BranchState] = None
    try:
        branch = _load_c1(repo, phase_a_output_dir, int(step))
        model = branch.pipeline.model
        idx, view_id, camera, batch = _first_train_record(branch)

        model.config.camera_medium_observability_enabled = False
        model.set_camera_medium_observability_projector(None)
        baseline_outputs, baseline_loss = _compute_outputs(model, camera, batch)
        model.set_camera_medium_observability_projector(torch.randn(9, 9, device=model.device))
        off_outputs, off_loss = _compute_outputs(model, camera, batch)
        disabled_rows = _max_diff_rows(baseline_outputs, off_outputs, baseline_loss, off_loss, "disabled_path")
        disabled_pass = all(row["pass"] for row in disabled_rows)

        model.set_camera_medium_observability_projector(None)
        model.config.camera_medium_observability_enabled = True
        neutral_outputs, neutral_loss = _compute_outputs(model, camera, batch)
        neutral_rows = _max_diff_rows(baseline_outputs, neutral_outputs, baseline_loss, neutral_loss, "enabled_no_projector")
        neutral_pass = all(row["pass"] for row in neutral_rows)
        _write_csv(output_dir / "equivalence_checks.csv", disabled_rows + neutral_rows)
        _write_json(output_dir / "equivalence_checks.json", {"rows": disabled_rows + neutral_rows})

        model.config.camera_medium_observability_enabled = False
        model.set_camera_medium_observability_projector(None)
        projector, singular, gates, scale, projector_meta = _estimate_projector(repo, output_dir, branch)
        model.set_camera_medium_observability_projector(projector, gates, scale)
        model.config.camera_medium_observability_enabled = True
        model.config.camera_medium_observability_strength = 1.0
        projected_outputs, projected_loss = _compute_outputs(model, camera, batch)
        mechanism_metrics = _mechanism_metrics(projected_outputs, gates)
        mechanism_metrics.update({"projected_loss": float(projected_loss.detach().cpu().item()), "view_id": view_id, "view_index": int(idx)})
        _write_json(output_dir / "mechanism_metrics.json", mechanism_metrics)

        grad = _gradient_preflight(model, camera, batch)
        _write_json(output_dir / "gradient_pathway.json", grad)

        smoke_rows, smoke_summary = _smoke_train(
            branch,
            phase_a_output_dir / "paired_camera_sequence.csv",
            output_dir,
            int(smoke_steps),
            gates,
        )
        _write_csv(output_dir / "smoke_training_metrics.csv", smoke_rows)
        _write_json(output_dir / "smoke_training_metrics.json", {"rows": smoke_rows, **smoke_summary})

        load_branch = _load_c1(repo, phase_a_output_dir, int(step))
        ckpt = torch.load(smoke_summary["checkpoint_path"], map_location="cpu")
        load_branch.pipeline.load_pipeline(ckpt["pipeline"], int(step) + int(smoke_steps))
        checkpoint_compat = {
            "checkpoint_load_pass": True,
            "loaded_smoke_checkpoint": smoke_summary["checkpoint_path"],
            "projector_nonpersistent": True,
            "loaded_without_projector_state": True,
        }
        _write_json(output_dir / "checkpoint_compatibility.json", checkpoint_compat)

        residual_not_collapsed = mechanism_metrics["camera_residual_rms"] > 0.0 and mechanism_metrics["projected_over_full_rms"] > 0.05
        ready = (
            disabled_pass
            and neutral_pass
            and bool(projector_meta["projector_finite"])
            and bool(grad["gaussian_grad_zero_for_direct_mechanism_metric"])
            and bool(smoke_summary["all_losses_finite"])
            and bool(checkpoint_compat["checkpoint_load_pass"])
            and bool(residual_not_collapsed)
        )
        classification = "CAMERA_OBSERVABILITY_MODULE_READY" if ready else "CAMERA_OBSERVABILITY_MODULE_PARTIAL"
        summary = {
            "phase_b_experiment": "CAMERA_MEDIUM_OBSERVABILITY_PREFLIGHT",
            "mechanism": "OCMC",
            "source_phase_a_output_dir": str(phase_a_output_dir),
            "checkpoint_step": int(step),
            "gpu": gpu,
            "disabled_path_equivalence_pass": bool(disabled_pass),
            "enabled_no_projector_equivalence_pass": bool(neutral_pass),
            "projector": projector_meta,
            "gradient_pathway_medium_local_for_direct_metric": bool(grad["gaussian_grad_zero_for_direct_mechanism_metric"]),
            "smoke_training": smoke_summary,
            "checkpoint_compatibility": checkpoint_compat,
            "residual_not_trivially_collapsed": bool(residual_not_collapsed),
            "phase_b_classification": classification,
            "no_15k_training_run": True,
        }
        _write_json(output_dir / "preflight_summary.json", summary)
        return summary
    finally:
        CAM._release(branch)
        CAM._release(load_branch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--phase-a-output-dir", type=Path, default=PHASE_A_OUTPUT_DIR)
    parser.add_argument("--step", type=int, default=FINAL_STEP)
    parser.add_argument("--smoke-steps", type=int, default=SMOKE_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        args.repo,
        args.output_dir,
        args.phase_a_output_dir,
        int(args.step),
        int(args.smoke_steps),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
