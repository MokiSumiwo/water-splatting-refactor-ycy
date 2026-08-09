#!/usr/bin/env python
"""Pre-training audits for the Panama BND-AOPT experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.cuda.amp import GradScaler

from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.scripts.train import _set_random_seed

from water_splatting.water_splatting import SHLogits2RGB
from water_splatting.water_splatting_config import water_splatting_method


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _tensor_hash(tensor: torch.Tensor) -> str:
    arr = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _param_hashes(model: Any) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    for name in ("means", "features_dc", "features_rest", "opacities", "scales", "quats"):
        rows[name] = _tensor_hash(getattr(model, name))
    medium_items = []
    for name, param in model.named_parameters():
        if name.startswith("medium_mlp") or name.startswith("direction_encoding"):
            medium_items.append((name, _tensor_hash(param)))
    rows["medium_parameters"] = hashlib.sha256(
        "|".join(f"{name}:{digest}" for name, digest in sorted(medium_items)).encode("utf8")
    ).hexdigest()
    return rows


def _grad_hashes(model: Any) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    for name in ("features_dc", "features_rest", "means", "scales", "quats", "opacities"):
        param = getattr(model, name)
        rows[name] = _tensor_hash(param.grad) if param.grad is not None else "NO_GRAD"
    return rows


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    if a.shape != b.shape:
        return {"max_abs": float("inf"), "mean_abs": float("inf")}
    diff = (a.detach().float() - b.detach().float()).abs()
    return {"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())}


def _parameter_diffs(model_a: Any, model_b: Any, *, use_grad: bool = False) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in ("means", "features_dc", "features_rest", "opacities", "scales", "quats"):
        item_a = getattr(model_a, name)
        item_b = getattr(model_b, name)
        if use_grad:
            item_a = item_a.grad
            item_b = item_b.grad
            if item_a is None and item_b is None:
                out[f"{name}_max_abs"] = 0.0
                out[f"{name}_mean_abs"] = 0.0
                continue
            if item_a is None or item_b is None:
                out[f"{name}_max_abs"] = float("inf")
                out[f"{name}_mean_abs"] = float("inf")
                continue
        stats = _tensor_diff(item_a, item_b)
        out[f"{name}_max_abs"] = stats["max_abs"]
        out[f"{name}_mean_abs"] = stats["mean_abs"]
    return out


def _make_config(repo: Path, data_path: Path, scale: float, output_dir: Path, experiment_name: str) -> Any:
    config = copy.deepcopy(water_splatting_method.config)
    config.output_dir = output_dir
    config.experiment_name = experiment_name
    config.timestamp = experiment_name
    config.vis = "tensorboard"
    config.machine.seed = 42
    config.max_num_iterations = 1
    config.steps_per_save = 1000
    config.save_only_latest_checkpoint = False
    config.pipeline.datamanager.dataparser = ColmapDataParserConfig(
        data=data_path,
        images_path=Path("images/ColorImage"),
        colmap_path=Path("sparse/0"),
        downscale_factor=1,
        load_3D_points=True,
        eval_mode="interval",
        eval_interval=8,
        train_split_fraction=0.9,
        orientation_method="up",
        scale_factor=1.0,
        scene_scale=1.0,
        max_2D_matches_per_3D_point=0,
    )
    config.pipeline.datamanager.train_cameras_sampling_seed = 42
    model_config = config.pipeline.model
    model_config.num_steps = 15000
    model_config.sh_degree = 3
    model_config.medium_context_mode = "dir_xy_camera"
    model_config.b_inf_mode = "tied"
    model_config.infinite_water_enabled = False
    model_config.intrinsic_color_parameterization = "bounded_sh3"
    model_config.bounded_sh_logit_eps = 1e-7
    model_config.appearance_lr_scale = scale
    model_config.appearance_audit_log_dir = None
    return config


def _setup(repo: Path, data_path: Path, scale: float, tag: str) -> Tuple[Any, Optimizers]:
    _set_random_seed(42)
    config = _make_config(repo, data_path, scale, repo / "outputs" / "_aopt_setup_audit_tmp", tag)
    grad_scaler = GradScaler(enabled=False)
    pipeline = config.pipeline.setup(
        device="cuda:0",
        test_mode="val",
        world_size=1,
        local_rank=0,
        grad_scaler=grad_scaler,
    )
    optimizers = Optimizers(config.optimizers, pipeline.get_param_groups())
    return pipeline, optimizers


def _optimizer_rows(optimizers: Optimizers) -> List[Dict[str, Any]]:
    rows = []
    for group, item in optimizers.config.items():
        opt_config = item["optimizer"]
        scheduler_config = item.get("scheduler")
        rows.append(
            {
                "group": group,
                "optimizer_type": type(opt_config).__name__,
                "optimizer_target": str(getattr(opt_config, "_target", "")),
                "lr": float(opt_config.lr),
                "eps": float(getattr(opt_config, "eps", float("nan"))),
                "max_norm": getattr(opt_config, "max_norm", None),
                "scheduler_type": type(scheduler_config).__name__ if scheduler_config else "None",
                "lr_final": getattr(scheduler_config, "lr_final", None) if scheduler_config else None,
                "max_steps": getattr(scheduler_config, "max_steps", None) if scheduler_config else None,
                "actual_optimizer_lr": float(optimizers.optimizers[group].param_groups[0]["lr"])
                if group in optimizers.optimizers
                else "",
                "scheduler_base_lr": float(optimizers.schedulers[group].base_lrs[0])
                if group in optimizers.schedulers
                else "",
            }
        )
    return rows


def _initialization_rows(pairs: Sequence[Tuple[str, Any]]) -> List[Dict[str, Any]]:
    hashes = {tag: _param_hashes(pipeline.model) for tag, pipeline in pairs}
    rows = []
    ref_tag, ref_pipeline = pairs[0]
    ref_hash = hashes[ref_tag]
    for tag, pipeline in pairs:
        model = pipeline.model
        seed_rgb = model.seed_points[1].to(model.features_dc.device).float() / 255.0
        bounded_rgb = SHLogits2RGB(model.features_dc.detach())
        err = torch.linalg.norm(bounded_rgb - seed_rgb, dim=-1)
        for name, digest in hashes[tag].items():
            rows.append(
                {
                    "run": tag,
                    "parameter": name,
                    "sha256": digest,
                    "matches_reference": digest == ref_hash[name],
                    "gaussian_count": int(model.num_points),
                    "init_rgb_error_mean": float(err.mean().item()),
                    "init_rgb_error_p95": float(torch.quantile(err, 0.95).item()),
                    "init_rgb_error_max": float(err.max().item()),
                    "bounded_rgb_min": float(bounded_rgb.min().item()),
                    "bounded_rgb_max": float(bounded_rgb.max().item()),
                }
            )
    return rows


def _one_step_equivalence(repo: Path, data_path: Path) -> Dict[str, Any]:
    baseline, baseline_opt = _setup(repo, data_path, 1.0, "aopt_k1_equiv_baseline")
    newpath, newpath_opt = _setup(repo, data_path, 1.0, "aopt_k1_equiv_newpath")
    newpath.model._apply_appearance_lr_scale(newpath_opt)

    row: Dict[str, Any] = {"appearance_lr_scale": 1.0}
    row["initial_hash_match"] = _param_hashes(baseline.model) == _param_hashes(newpath.model)
    row["features_dc_lr_match"] = math.isclose(
        baseline_opt.optimizers["features_dc"].param_groups[0]["lr"],
        newpath_opt.optimizers["features_dc"].param_groups[0]["lr"],
        rel_tol=0.0,
        abs_tol=0.0,
    )
    row["features_rest_lr_match"] = math.isclose(
        baseline_opt.optimizers["features_rest"].param_groups[0]["lr"],
        newpath_opt.optimizers["features_rest"].param_groups[0]["lr"],
        rel_tol=0.0,
        abs_tol=0.0,
    )

    losses = []
    for pipeline, optimizers in ((baseline, baseline_opt), (newpath, newpath_opt)):
        pipeline.train()
        optimizers.zero_grad_all()
        _, loss_dict, _ = pipeline.get_train_loss_dict(step=0)
        loss = sum(loss_dict.values())
        loss.backward()
        losses.append(float(loss.detach().item()))
    grad_diffs = _parameter_diffs(baseline.model, newpath.model, use_grad=True)
    for key, value in grad_diffs.items():
        row[f"grad_{key}"] = value

    for pipeline, optimizers in ((baseline, baseline_opt), (newpath, newpath_opt)):
        optimizers.optimizer_step_all()
    post_step_diffs = _parameter_diffs(baseline.model, newpath.model, use_grad=False)
    for key, value in post_step_diffs.items():
        row[f"post_step_{key}"] = value
    row["loss_baseline"] = losses[0]
    row["loss_newpath"] = losses[1]
    row["loss_abs_diff"] = abs(losses[0] - losses[1])
    row["gradient_max_abs_diff"] = max(value for key, value in grad_diffs.items() if key.endswith("_max_abs"))
    row["post_step_max_abs_diff"] = max(value for key, value in post_step_diffs.items() if key.endswith("_max_abs"))
    row["appearance_gradient_max_abs_diff"] = max(
        grad_diffs["features_dc_max_abs"],
        grad_diffs["features_rest_max_abs"],
    )
    row["appearance_post_step_max_abs_diff"] = max(
        post_step_diffs["features_dc_max_abs"],
        post_step_diffs["features_rest_max_abs"],
    )
    row["nonappearance_post_step_max_abs_diff"] = max(
        value
        for key, value in post_step_diffs.items()
        if key.endswith("_max_abs") and not key.startswith(("features_dc", "features_rest"))
    )
    row["gradient_numeric_match"] = row["gradient_max_abs_diff"] <= 1e-9
    row["post_step_numeric_match"] = row["post_step_max_abs_diff"] <= 1e-9
    row["appearance_gradient_numeric_match"] = row["appearance_gradient_max_abs_diff"] <= 1e-9
    row["appearance_post_step_numeric_match"] = row["appearance_post_step_max_abs_diff"] <= 1e-6
    row["K1_OPTIMIZER_EQUIVALENCE"] = bool(
        row["initial_hash_match"]
        and row["features_dc_lr_match"]
        and row["features_rest_lr_match"]
        and row["loss_abs_diff"] == 0.0
        and row["appearance_gradient_numeric_match"]
        and row["appearance_post_step_numeric_match"]
    )
    row["equivalence_scope"] = (
        "appearance optimizer groups; non-appearance post-step differences are recorded separately "
        "because independent CUDA/tcnn backward passes are not bit-identical"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_aopt_equivalence_panama_20260809"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    data_path = args.data_path or repo / "undistorted_data" / "undistorted_Panama"
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    k2, k2_opt = _setup(repo, data_path, 2.0, "aopt_init_k2")
    k4, k4_opt = _setup(repo, data_path, 4.0, "aopt_init_k4")

    optimizer_rows = _optimizer_rows(k2_opt)
    init_rows = _initialization_rows((("K2", k2), ("K4", k4)))
    equivalence = _one_step_equivalence(repo, data_path)

    _write_csv(output_dir / "aopt_optimizer_audit.csv", optimizer_rows)
    _write_json(output_dir / "aopt_optimizer_audit.json", optimizer_rows)
    _write_csv(output_dir / "aopt_initialization_audit.csv", init_rows)
    _write_json(output_dir / "aopt_initialization_audit.json", init_rows)
    _write_csv(output_dir / "aopt_k1_equivalence_audit.csv", [equivalence])
    _write_json(output_dir / "aopt_k1_equivalence_audit.json", equivalence)
    print(json.dumps({"k1_equivalence": equivalence, "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
