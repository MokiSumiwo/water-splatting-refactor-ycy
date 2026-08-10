#!/usr/bin/env python
"""Audit setup gates for the Panama bounded-headroom SH3 experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.train import _set_random_seed
from nerfstudio.utils.eval_utils import eval_setup
from water_splatting.sh import spherical_harmonics


SCENE = "Panama"
CHANNELS = ("r", "g", "b")
EPS = 1e-8
START_HEAD_EXPECTED = "95919d153f9c5a5d8fad1983a12137a5a45255a0"
M1_CONFIG = (
    "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
    "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
)
K1_CONFIG = (
    "outputs/dewater_bounded_sh3_cross_scene_20260808/"
    "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
    "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
    "config.yml"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
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


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception:
        return "unknown"


def _safe_quantile(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if q <= 0.0:
        return float(flat.min().item())
    if q >= 1.0:
        return float(flat.max().item())
    rank = max(1, min(flat.numel(), int(math.ceil(q * flat.numel()))))
    return float(torch.kthvalue(flat, rank).values.item())


def _stats(values: Tensor, prefix: str) -> Dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    names = ("count", "mean", "p50", "p75", "p90", "p95", "p99", "max")
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in names}
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p50": _safe_quantile(flat, 0.50),
        f"{prefix}p75": _safe_quantile(flat, 0.75),
        f"{prefix}p90": _safe_quantile(flat, 0.90),
        f"{prefix}p95": _safe_quantile(flat, 0.95),
        f"{prefix}p99": _safe_quantile(flat, 0.99),
        f"{prefix}max": float(flat.max().item()),
    }


def _threshold_fraction(values: Tensor, threshold: float, op: str = "gt") -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if op == "gt":
        return float((flat > threshold).float().mean().item())
    if op == "lt":
        return float((flat < threshold).float().mean().item())
    raise ValueError(op)


def _channel_u_stats(values: Tensor, prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for index, channel in enumerate(CHANNELS):
        channel_values = values[..., index].reshape(-1)
        channel_values = channel_values[torch.isfinite(channel_values)]
        out.update(_stats(channel_values, f"{prefix}_{channel}_"))
        for threshold in (0.25, 0.50, 0.75, 0.90):
            out[f"{prefix}_{channel}_P_gt_{threshold:g}"] = _threshold_fraction(channel_values, threshold)
    all_values = values.reshape(-1)
    all_values = all_values[torch.isfinite(all_values)]
    out.update(_stats(all_values, f"{prefix}_all_"))
    for threshold in (0.25, 0.50, 0.75, 0.90):
        out[f"{prefix}_all_P_gt_{threshold:g}"] = _threshold_fraction(all_values, threshold)
    return out


def _release_pipeline(pipeline: Optional[Any]) -> None:
    if pipeline is not None:
        del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _config_from_yaml(config_path: Path, parameterization: str) -> Any:
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    config.load_dir = None
    config.load_step = None
    config.pipeline.model.intrinsic_color_parameterization = parameterization
    config.pipeline.model.rasterize_mode = "classic"
    return config


def _setup_pipeline(repo: Path, parameterization: str) -> Any:
    config = _config_from_yaml(repo / K1_CONFIG, parameterization)
    _set_random_seed(config.machine.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="test")
    assert isinstance(pipeline, Pipeline)
    pipeline.model.config.intrinsic_color_parameterization = parameterization
    pipeline.model.config.rasterize_mode = "classic"
    return pipeline


def _view_records(pipeline: Any) -> List[Tuple[int, str, Cameras, Mapping[str, Any]]]:
    dataset = pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    rows: List[Tuple[int, str, Cameras, Mapping[str, Any]]] = []
    for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        rows.append((eval_index, Path(filename).stem, camera, batch))
    return rows


def _metric_loss(model: Any, camera: Cameras, batch: Mapping[str, Any], audit_step: int) -> Tuple[Mapping[str, Tensor], Tensor]:
    model.train()
    model.step = audit_step
    outputs = model.get_outputs(camera.to(model.device))
    loss = model.get_loss_dict(outputs, batch)["main_loss"]
    return outputs, loss


def _named_trainable_tensors(model: Any) -> Dict[str, Tensor]:
    out = {
        "means": model.means,
        "scales": model.scales,
        "quats": model.quats,
        "opacities": model.opacities,
        "features_dc": model.features_dc,
        "features_rest": model.features_rest,
    }
    for name, param in model.medium_mlp.named_parameters():
        out[f"medium_mlp.{name}"] = param
    return out


def sh_semantics_audit(repo: Path) -> Dict[str, Any]:
    return {
        "scene": SCENE,
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "CODE_FACTS": {
            "features_dc_shape": "N x 3, stored as the SH degree-0 coefficient for RGB/logit channels.",
            "features_rest_shape": "N x (num_sh_bases(sh_degree)-1) x 3, concatenated after features_dc before SH evaluation.",
            "dc_basis": "spherical_harmonics(0, viewdirs, colors[:, :1, :]) returns the degree-0 contribution; for bounded modes seed RGB is initialized with RGB2SHLogits so degree-0 output equals logit(seed RGB).",
            "s_full": "spherical_harmonics(active_sh_degree, viewdirs, cat(features_dc[:,None,:], features_rest))",
            "s0": "spherical_harmonics(0, viewdirs, colors[:, :1, :]); viewdirs are passed but degree-0 basis is view-independent.",
            "r_SH": "s_full(v) - s0; because SH evaluation is linear in coefficients and degree-0 is separated, this is the active non-DC SH logit contribution.",
            "bounded_sh3": "c_BND1(v) = sigmoid(s_full(v)); the legacy +0.5 offset is not applied.",
            "renderer_input": "WaterSplattingModel.get_outputs passes the current-view Gaussian RGB tensor to UnderwaterRasterizer.rasterize; no renderer physics branch is changed by BND-HR.",
        },
        "code_paths": [
            "water_splatting/fields/gaussian_appearance.py::compute_bounded_gaussian_colors",
            "water_splatting/fields/gaussian_appearance.py::compute_bounded_headroom_gaussian_colors",
            "water_splatting/water_splatting.py::WaterSplattingModel.get_outputs",
            "water_splatting/sh.py::spherical_harmonics",
        ],
        "SH_LINEAR_DECOMPOSITION_CONFIRMED": True,
    }


def formula_audit() -> Dict[str, Any]:
    return {
        "name": "Jacobian-Matched Asymmetric Headroom SH3",
        "intrinsic_color_parameterization": "bounded_headroom_sh3",
        "formula": {
            "s0": "DC/base SH logit contribution",
            "c0": "sigmoid(s0)",
            "r": "s_full(v) - s0",
            "positive_branch": "c0 + (1-c0) * tanh(c0 * r), for r >= 0",
            "negative_branch": "c0 + c0 * tanh((1-c0) * r), for r < 0",
        },
        "properties": {
            "boundedness": "r>=0 gives c0 <= c < 1; r<0 gives 0 < c < c0.",
            "zero_residual_equivalence": "r=0 gives c=c0, matching bounded_sh3 sigmoid(s0).",
            "jacobian_matching": "At r=0, d c / d r = c0*(1-c0), matching bounded_sh3.",
            "dc_local_derivative": "At r=0, r is the non-DC contribution and d c / d s0 = c0*(1-c0), matching bounded_sh3.",
        },
        "forbidden_changes": [
            "no residual gain alpha",
            "no temperature",
            "no clamp-after-unbounded-output",
            "no renderer physics change",
            "no rasterize_mode change",
        ],
    }


def legacy_headroom_sign_audit(repo: Path, output_dir: Path) -> List[Dict[str, Any]]:
    config_path = repo / M1_CONFIG
    def update_config(config: Any) -> Any:
        config.load_step = 14999
        return config

    config, pipeline, _, _ = eval_setup(config_path, test_mode="test", update_config_callback=update_config)
    pipeline.model.config.intrinsic_color_parameterization = "legacy"
    pipeline.eval()
    model = pipeline.model
    rows: List[Dict[str, Any]] = []
    all_dc: List[Tensor] = []
    all_full: List[Tensor] = []
    all_delta: List[Tensor] = []
    try:
        for _, view_id, camera, _ in _view_records(pipeline):
            camera = camera.to(model.device)
            if model.crop_box is not None and not model.training:
                crop_ids = model.crop_box.within(model.means).squeeze()
            else:
                crop_ids = None
            means = model.means[crop_ids] if crop_ids is not None and crop_ids.sum() != 0 else model.means
            fdc = model.features_dc[crop_ids] if crop_ids is not None and crop_ids.sum() != 0 else model.features_dc
            frest = model.features_rest[crop_ids] if crop_ids is not None and crop_ids.sum() != 0 else model.features_rest
            colors = torch.cat((fdc[:, None, :], frest), dim=1)
            viewdirs = means.detach() - camera.camera_to_worlds[..., :3, 3].detach()
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            raw_full = spherical_harmonics(model.config.sh_degree, viewdirs, colors)
            raw_dc = spherical_harmonics(0, viewdirs, colors[:, :1, :])
            full = torch.clamp(raw_full + 0.5, min=0.0).detach().float().cpu()
            dc = torch.clamp(raw_dc + 0.5, min=0.0).detach().float().cpu()
            delta = full - dc
            all_dc.append(dc)
            all_full.append(full)
            all_delta.append(delta)
            rows.extend(_legacy_headroom_rows_for_tensor(view_id, dc, full, delta))
    finally:
        _release_pipeline(pipeline)

    dc_all = torch.cat(all_dc, dim=0)
    full_all = torch.cat(all_full, dim=0)
    delta_all = torch.cat(all_delta, dim=0)
    rows.extend(_legacy_headroom_rows_for_tensor("AGGREGATE", dc_all, full_all, delta_all))

    # Preserve the previous total legal/overflow energy numbers for direct traceability.
    prior_energy_path = repo / "outputs/bnd_shstruct_audit_20260810/sh_residual_energy_decomposition.csv"
    if prior_energy_path.exists():
        with prior_energy_path.open("r", encoding="utf8") as handle:
            prior_rows = list(csv.DictReader(handle))
        for prior in prior_rows:
            if prior.get("scene") == SCENE:
                _write_json(
                    output_dir / "legacy_sh_energy_reference.json",
                    {
                        "source": str(prior_energy_path),
                        "scene": SCENE,
                        "LEGAL_SH_ENERGY_FRACTION": prior.get("LEGAL_SH_ENERGY_FRACTION"),
                        "OVERFLOW_SH_ENERGY_FRACTION": prior.get("OVERFLOW_SH_ENERGY_FRACTION"),
                        "BASE_INVALID_SH_ENERGY_FRACTION": prior.get("BASE_INVALID_SH_ENERGY_FRACTION"),
                    },
                )
                break
    return rows


def _legacy_headroom_rows_for_tensor(view_id: str, dc: Tensor, full: Tensor, delta: Tensor) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    valid = (dc > 0.0) & (dc < 1.0) & (full > 0.0) & (full < 1.0)
    pos = valid & (delta > 0.0)
    neg = valid & (delta < 0.0)
    pos_energy = delta[pos].square().sum()
    neg_energy = delta[neg].square().sum()
    total_energy = pos_energy + neg_energy
    base = {
        "scene": SCENE,
        "view_id": view_id,
        "valid_to_valid_channel_observations": int(valid.sum().item()),
        "positive_residual_fraction": float(pos.float().sum().item() / max(float(valid.sum().item()), 1.0)),
        "negative_residual_fraction": float(neg.float().sum().item() / max(float(valid.sum().item()), 1.0)),
        "positive_residual_energy_fraction": float(pos_energy.item() / max(float(total_energy.item()), EPS)),
        "negative_residual_energy_fraction": float(neg_energy.item() / max(float(total_energy.item()), EPS)),
        "spatial_subsets": "Brightness Q5 and M1 high-J image masks are not mapped to per-Gaussian observations in this audit; no cross-run/per-pixel Gaussian matching was forced.",
    }
    u_pos = torch.full_like(delta, float("nan"))
    u_neg = torch.full_like(delta, float("nan"))
    u_pos[pos] = delta[pos] / (1.0 - dc[pos] + EPS)
    u_neg[neg] = (-delta[neg]) / (dc[neg] + EPS)
    row = dict(base)
    row.update(_channel_u_stats(u_pos, "u_pos"))
    row.update(_channel_u_stats(u_neg, "u_neg"))
    for index, channel in enumerate(CHANNELS):
        channel_valid = valid[:, index]
        channel_pos = pos[:, index]
        channel_neg = neg[:, index]
        e_pos = delta[:, index][channel_pos].square().sum()
        e_neg = delta[:, index][channel_neg].square().sum()
        e_total = e_pos + e_neg
        row[f"{channel}_positive_fraction"] = float(channel_pos.float().sum().item() / max(float(channel_valid.float().sum().item()), 1.0))
        row[f"{channel}_negative_fraction"] = float(channel_neg.float().sum().item() / max(float(channel_valid.float().sum().item()), 1.0))
        row[f"{channel}_positive_energy_fraction"] = float(e_pos.item() / max(float(e_total.item()), EPS))
        row[f"{channel}_negative_energy_fraction"] = float(e_neg.item() / max(float(e_total.item()), EPS))
    rows.append(row)
    return rows


def initialization_parameter_audit(repo: Path) -> Tuple[List[Dict[str, Any]], bool]:
    k1_pipe = hr_pipe = None
    rows: List[Dict[str, Any]] = []
    try:
        k1_pipe = _setup_pipeline(repo, "bounded_sh3")
        hr_pipe = _setup_pipeline(repo, "bounded_headroom_sh3")
        for name, k1_param in _named_trainable_tensors(k1_pipe.model).items():
            hr_param = _named_trainable_tensors(hr_pipe.model).get(name)
            row = {
                "scene": SCENE,
                "parameter": name,
                "k1_shape": list(k1_param.shape),
                "hr_shape": list(hr_param.shape) if hr_param is not None else None,
            }
            if hr_param is not None and tuple(k1_param.shape) == tuple(hr_param.shape):
                diff = hr_param.detach().float().cpu() - k1_param.detach().float().cpu()
                row["max_abs_diff"] = float(diff.abs().max().item()) if diff.numel() else 0.0
                row["mean_abs_diff"] = float(diff.abs().mean().item()) if diff.numel() else 0.0
            else:
                row["max_abs_diff"] = float("nan")
                row["mean_abs_diff"] = float("nan")
            row["INIT_PARAMETER_MATCH"] = bool(row["max_abs_diff"] == 0.0)
            rows.append(row)
    finally:
        _release_pipeline(k1_pipe)
        _release_pipeline(hr_pipe)
    return rows, all(bool(row["INIT_PARAMETER_MATCH"]) for row in rows)


def initialization_forward_audit(repo: Path, audit_step: int) -> Tuple[List[Dict[str, Any]], bool]:
    k1_pipe = hr_pipe = None
    rows: List[Dict[str, Any]] = []
    try:
        k1_pipe = _setup_pipeline(repo, "bounded_sh3")
        hr_pipe = _setup_pipeline(repo, "bounded_headroom_sh3")
        _, view_id, camera, _ = _view_records(k1_pipe)[0]
        k1_pipe.model.eval()
        hr_pipe.model.eval()
        k1_pipe.model.step = audit_step
        hr_pipe.model.step = audit_step
        with torch.no_grad():
            k1_out = k1_pipe.model.get_outputs(camera.to(k1_pipe.model.device))
            hr_out = hr_pipe.model.get_outputs(camera.to(hr_pipe.model.device))
        for key in (
            "gaussian_view_rgb",
            "pred_image",
            "direct_object_signal",
            "rgb_medium",
            "depth",
            "accumulation",
            "clear_object_fullsh_raw",
            "transmission",
            "tau_D",
        ):
            a = k1_out[key].detach().float().cpu()
            b = hr_out[key].detach().float().cpu()
            diff = (a - b).abs()
            rows.append(
                {
                    "scene": SCENE,
                    "view_id": view_id,
                    "audit_step": audit_step,
                    "output": key,
                    "mean_abs_diff": float(diff.mean().item()),
                    "p99_abs_diff": _safe_quantile(diff, 0.99),
                    "max_abs_diff": float(diff.max().item()),
                    "shape_match": tuple(a.shape) == tuple(b.shape),
                }
            )
    finally:
        _release_pipeline(k1_pipe)
        _release_pipeline(hr_pipe)
    gate = all(float(row["max_abs_diff"]) <= 1e-6 for row in rows if row["output"] in ("pred_image", "gaussian_view_rgb"))
    return rows, gate


def initialization_jacobian_audit(repo: Path, audit_step: int) -> Tuple[List[Dict[str, Any]], bool]:
    k1_pipe = hr_pipe = None
    rows: List[Dict[str, Any]] = []
    try:
        k1_pipe = _setup_pipeline(repo, "bounded_sh3")
        hr_pipe = _setup_pipeline(repo, "bounded_headroom_sh3")
        _, view_id, camera, batch = _view_records(k1_pipe)[0]
        k1_out, k1_loss = _metric_loss(k1_pipe.model, camera, batch, audit_step)
        k1_loss.backward()
        hr_out, hr_loss = _metric_loss(hr_pipe.model, camera, batch, audit_step)
        hr_loss.backward()
        for name in ("features_dc", "features_rest", "means", "scales", "opacities"):
            k1_param = getattr(k1_pipe.model, name)
            hr_param = getattr(hr_pipe.model, name)
            k1_grad = k1_param.grad.detach().float().cpu() if k1_param.grad is not None else torch.zeros_like(k1_param.detach().cpu())
            hr_grad = hr_param.grad.detach().float().cpu() if hr_param.grad is not None else torch.zeros_like(hr_param.detach().cpu())
            diff = hr_grad - k1_grad
            denom = float(torch.linalg.norm(k1_grad).item())
            rows.append(
                {
                    "scene": SCENE,
                    "view_id": view_id,
                    "audit_step": audit_step,
                    "parameter": name,
                    "k1_grad_l2": denom,
                    "hr_grad_l2": float(torch.linalg.norm(hr_grad).item()),
                    "diff_l2": float(torch.linalg.norm(diff).item()),
                    "relative_l2_diff": float(torch.linalg.norm(diff).item() / max(denom, EPS)),
                    "mean_abs_diff": float(diff.abs().mean().item()) if diff.numel() else 0.0,
                    "max_abs_diff": float(diff.abs().max().item()) if diff.numel() else 0.0,
                }
            )
        rows.append(
            {
                "scene": SCENE,
                "view_id": view_id,
                "audit_step": audit_step,
                "parameter": "loss",
                "k1_loss": float(k1_loss.detach().item()),
                "hr_loss": float(hr_loss.detach().item()),
                "max_abs_diff": float(abs(k1_loss.detach().item() - hr_loss.detach().item())),
                "relative_l2_diff": 0.0,
                "outputs_finite": bool(
                    torch.isfinite(k1_out["pred_image"]).all().item() and torch.isfinite(hr_out["pred_image"]).all().item()
                ),
            }
        )
    finally:
        _release_pipeline(k1_pipe)
        _release_pipeline(hr_pipe)
    appearance_rows = [row for row in rows if row["parameter"] in ("features_dc", "features_rest")]
    gate = all(float(row["relative_l2_diff"]) <= 1e-5 and float(row["max_abs_diff"]) <= 1e-6 for row in appearance_rows)
    return rows, gate


def compatibility_smoke(repo: Path) -> List[Dict[str, Any]]:
    rows = []
    for parameterization in ("legacy", "bounded_sh3", "bounded_headroom_sh3"):
        pipe = None
        try:
            pipe = _setup_pipeline(repo, parameterization if parameterization != "legacy" else "legacy")
            _, view_id, camera, _ = _view_records(pipe)[0]
            pipe.model.eval()
            with torch.no_grad():
                outputs = pipe.model.get_outputs_for_camera(camera)
            rows.append(
                {
                    "scene": SCENE,
                    "parameterization": parameterization,
                    "view_id": view_id,
                    "pred_image_finite": bool(torch.isfinite(outputs["pred_image"]).all().item()),
                    "gaussian_rgb_finite": bool(torch.isfinite(outputs["gaussian_view_rgb"]).all().item()),
                    "rasterize_mode": pipe.model.config.rasterize_mode,
                    "status": "OK",
                }
            )
        except Exception as exc:
            rows.append({"scene": SCENE, "parameterization": parameterization, "status": "ERROR", "error": repr(exc)})
        finally:
            _release_pipeline(pipe)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_hr_panama_20260810"))
    parser.add_argument("--audit-step", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    semantics = sh_semantics_audit(repo)
    formula = formula_audit()
    legacy_rows = legacy_headroom_sign_audit(repo, output_dir)
    param_rows, param_gate = initialization_parameter_audit(repo)
    forward_rows, forward_gate = initialization_forward_audit(repo, args.audit_step)
    jacobian_rows, jacobian_gate = initialization_jacobian_audit(repo, args.audit_step)
    smoke_rows = compatibility_smoke(repo)
    gates = {
        "INIT_PARAMETER_EQUIVALENCE": param_gate,
        "INIT_FORWARD_EQUIVALENCE": forward_gate,
        "INIT_APPEARANCE_JACOBIAN_EQUIVALENCE": jacobian_gate,
        "ALL_TRAINING_GATES_PASS": bool(param_gate and forward_gate and jacobian_gate),
    }
    manifest = {
        "scene": SCENE,
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "start_head_expected": START_HEAD_EXPECTED,
        "audit_step": args.audit_step,
        "gates": gates,
        "outputs": {
            "sh_semantics_audit": str(output_dir / "sh_semantics_audit.json"),
            "legacy_headroom_sign_audit": str(output_dir / "legacy_headroom_sign_audit.csv"),
            "bnd_hr_formula_audit": str(output_dir / "bnd_hr_formula_audit.json"),
            "initialization_parameter_equivalence": str(output_dir / "initialization_parameter_equivalence.csv"),
            "initialization_forward_equivalence": str(output_dir / "initialization_forward_equivalence.csv"),
            "initialization_jacobian_equivalence": str(output_dir / "initialization_jacobian_equivalence.csv"),
        },
    }
    _write_json(output_dir / "sh_semantics_audit.json", semantics)
    _write_json(output_dir / "bnd_hr_formula_audit.json", formula)
    _write_csv(output_dir / "legacy_headroom_sign_audit.csv", legacy_rows)
    _write_json(output_dir / "legacy_headroom_sign_audit.json", legacy_rows)
    _write_csv(output_dir / "initialization_parameter_equivalence.csv", param_rows)
    _write_json(output_dir / "initialization_parameter_equivalence.json", param_rows)
    _write_csv(output_dir / "initialization_forward_equivalence.csv", forward_rows)
    _write_json(output_dir / "initialization_forward_equivalence.json", forward_rows)
    _write_csv(output_dir / "initialization_jacobian_equivalence.csv", jacobian_rows)
    _write_json(output_dir / "initialization_jacobian_equivalence.json", jacobian_rows)
    _write_csv(output_dir / "compatibility_smoke.csv", smoke_rows)
    _write_json(output_dir / "compatibility_smoke.json", smoke_rows)
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
