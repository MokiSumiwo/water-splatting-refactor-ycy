#!/usr/bin/env python
"""Single-batch gradient audit for Transmission-Balanced Appearance Preconditioning."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from nerfstudio.utils.eval_utils import eval_setup


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _zero_grads(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)


def _stats(values: torch.Tensor) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1).cpu()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
    }


def _safe_grad(param: torch.nn.Parameter) -> torch.Tensor:
    if param.grad is None:
        return torch.zeros_like(param.detach())
    return param.grad.detach().clone()


def _depth_bins(depths: torch.Tensor, visible: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    visible_depth = depths.detach().reshape(-1)[visible]
    if visible_depth.numel() == 0:
        empty = torch.zeros_like(visible, dtype=torch.bool)
        return [(name, empty) for name in ("q0_25", "q25_50", "q50_75", "q75_100")]
    q25, q50, q75 = torch.quantile(visible_depth.float(), torch.tensor([0.25, 0.50, 0.75], device=depths.device))
    d = depths.detach().reshape(-1)
    return [
        ("q0_25_nearest", visible & (d <= q25)),
        ("q25_50", visible & (d > q25) & (d <= q50)),
        ("q50_75", visible & (d > q50) & (d <= q75)),
        ("q75_100_farthest", visible & (d > q75)),
    ]


def _grad_summary(model: Any) -> Dict[str, Any]:
    dc_grad = _safe_grad(model.features_dc).reshape(model.features_dc.shape[0], -1)
    rest_grad = _safe_grad(model.features_rest).reshape(model.features_rest.shape[0], -1)
    visible = (model.radii.detach().reshape(-1) > 0) if getattr(model, "radii", None) is not None else torch.ones(
        model.features_dc.shape[0], device=model.features_dc.device, dtype=torch.bool
    )
    depths = getattr(model, "depths", torch.zeros(model.features_dc.shape[0], device=model.features_dc.device))
    bins = _depth_bins(depths, visible)

    def group(mask: torch.Tensor) -> Dict[str, Any]:
        if mask.sum() == 0:
            return {
                "count": 0,
                "dc_abs_r": 0.0,
                "dc_abs_g": 0.0,
                "dc_abs_b": 0.0,
                "dc_abs_mean": 0.0,
                "dc_norm_mean": 0.0,
                "rest_norm_mean": 0.0,
            }
        dc = dc_grad[mask]
        rest = rest_grad[mask]
        dc_abs = dc.abs()
        return {
            "count": int(mask.sum().item()),
            "dc_abs_r": float(dc_abs[:, 0].mean().item()),
            "dc_abs_g": float(dc_abs[:, 1].mean().item()),
            "dc_abs_b": float(dc_abs[:, 2].mean().item()),
            "dc_abs_mean": float(dc_abs.mean().item()),
            "dc_norm_mean": float(dc.norm(dim=-1).mean().item()),
            "rest_norm_mean": float(rest.norm(dim=-1).mean().item()) if rest.numel() else 0.0,
        }

    return {
        "visible_count": int(visible.sum().item()),
        "dc_total_norm": float(torch.linalg.vector_norm(dc_grad).item()),
        "rest_total_norm": float(torch.linalg.vector_norm(rest_grad).item()),
        "appearance_total_norm": float(
            torch.sqrt(torch.linalg.vector_norm(dc_grad).square() + torch.linalg.vector_norm(rest_grad).square()).item()
        ),
        "depth_bins": {name: group(mask) for name, mask in bins},
    }


def _ratio(new: float, base: float) -> float:
    return float(new) / max(float(base), 1e-20)


def _compare(main: Dict[str, Any], tbap: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "dc_total_norm_ratio": _ratio(tbap["dc_total_norm"], main["dc_total_norm"]),
        "rest_total_norm_ratio": _ratio(tbap["rest_total_norm"], main["rest_total_norm"]),
        "appearance_total_norm_ratio": _ratio(tbap["appearance_total_norm"], main["appearance_total_norm"]),
        "depth_bins": {},
    }
    for name, row in main["depth_bins"].items():
        tb = tbap["depth_bins"][name]
        base_rgb = torch.tensor([row["dc_abs_r"], row["dc_abs_g"], row["dc_abs_b"]], dtype=torch.float32)
        new_rgb = torch.tensor([tb["dc_abs_r"], tb["dc_abs_g"], tb["dc_abs_b"]], dtype=torch.float32)
        if base_rgb.mean() > 0 and new_rgb.mean() > 0:
            base_shape = base_rgb / base_rgb.mean().clamp_min(1e-20)
            new_shape = new_rgb / new_rgb.mean().clamp_min(1e-20)
            channel_ratio_change = torch.max(torch.abs(new_shape / base_shape.clamp_min(1e-20) - 1.0)).item()
        else:
            channel_ratio_change = 0.0
        out["depth_bins"][name] = {
            "dc_abs_r_ratio": _ratio(tb["dc_abs_r"], row["dc_abs_r"]),
            "dc_abs_g_ratio": _ratio(tb["dc_abs_g"], row["dc_abs_g"]),
            "dc_abs_b_ratio": _ratio(tb["dc_abs_b"], row["dc_abs_b"]),
            "dc_abs_mean_ratio": _ratio(tb["dc_abs_mean"], row["dc_abs_mean"]),
            "dc_norm_mean_ratio": _ratio(tb["dc_norm_mean"], row["dc_norm_mean"]),
            "rest_norm_mean_ratio": _ratio(tb["rest_norm_mean"], row["rest_norm_mean"]),
            "dc_channel_ratio_change_max": float(channel_ratio_change),
        }
    return out


def _pixel_depth_bins(depth: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    valid = torch.isfinite(depth.detach()) & (depth.detach() > 0)
    if not valid.any():
        empty = torch.zeros_like(depth, dtype=torch.bool)
        return [(name, empty) for name in ("q0_25_nearest", "q25_50", "q50_75", "q75_100_farthest")]
    valid_depth = depth.detach()[valid].float()
    q25, q50, q75 = torch.quantile(valid_depth, torch.tensor([0.25, 0.50, 0.75], device=depth.device))
    d = depth.detach()
    return [
        ("q0_25_nearest", valid & (d <= q25)),
        ("q25_50", valid & (d > q25) & (d <= q50)),
        ("q50_75", valid & (d > q50) & (d <= q75)),
        ("q75_100_farthest", valid & (d > q75)),
    ]


def _pixel_support_summary(
    outputs: Dict[str, torch.Tensor],
    support: torch.Tensor,
    weights: torch.Tensor,
    diag: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    depth = outputs["depth"].detach().squeeze(-1)
    support_s = support.detach().squeeze(-1)
    transmission = diag["transmission"].detach()
    conditioning_signal = diag.get("conditioning_signal", transmission).detach()
    raw_weight = diag["raw_weight"].detach()
    normalized_weight = weights.detach()

    def channel_means(value: torch.Tensor, mask: torch.Tensor) -> List[float]:
        if mask.sum() == 0:
            channels = value.shape[-1] if value.ndim >= 3 else 1
            return [0.0 for _ in range(channels)]
        selected = value[mask]
        if selected.ndim == 1:
            selected = selected[:, None]
        return [float(v) for v in selected.float().mean(dim=0).detach().cpu().tolist()]

    bins: Dict[str, Any] = {}
    for name, mask in _pixel_depth_bins(depth):
        bins[name] = {
            "count": int(mask.sum().item()),
            "support_mean": float(support_s[mask].float().mean().item()) if mask.any() else 0.0,
            "support_gt_0p25_fraction": float((support_s[mask] > 0.25).float().mean().item()) if mask.any() else 0.0,
            "transmission_mean_rgb": channel_means(transmission, mask),
            "conditioning_signal_mean": channel_means(conditioning_signal, mask),
            "raw_weight_mean_rgb": channel_means(raw_weight, mask),
            "normalized_weight_mean_rgb": channel_means(normalized_weight, mask),
        }
    return bins


def _configure_tbap(model: Any, args: argparse.Namespace, enabled: bool) -> None:
    model.config.tbap_enabled = bool(enabled)
    model.config.lambda_tbap = float(args.lambda_tbap if enabled else 0.0)
    model.config.tbap_start_step = int(args.step_override if args.step_override is not None else 0)
    model.config.tbap_ramp_steps = 0
    model.config.tbap_gamma = float(args.gamma)
    model.config.tbap_max_weight = float(args.max_weight)
    model.config.tbap_weight_mode = str(args.weight_mode)
    model.config.tbap_support_mode = str(args.support_mode)
    model.config.tbap_support_top_fraction = float(args.support_top_fraction)
    model.config.tbap_depth_weight_strength = float(args.depth_weight_strength)
    model.config.tbap_transmission_floor = float(args.transmission_floor)
    model.config.tbap_transmission_info_temp = float(args.transmission_info_temp)
    model.config.tbap_object_accum_mid = float(args.object_accum_mid)
    model.config.tbap_object_accum_temp = float(args.object_accum_temp)
    model.config.tbap_object_concentration_kappa = float(args.object_concentration_kappa)
    model.config.tbap_far_depth_mid = float(args.far_depth_mid)
    model.config.tbap_far_depth_temp = float(args.far_depth_temp)
    model.config.tbap_depth_normalize_mode = str(args.depth_normalize_mode)
    model.config.tbap_smooth_l1_beta = float(args.smooth_l1_beta)


def _run_case(
    model: Any,
    camera: Any,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    enabled: bool,
    loss_mode: str,
) -> Dict[str, Any]:
    _zero_grads(model)
    _configure_tbap(model, args, enabled)
    outputs = model.get_outputs(camera)
    metrics: Dict[str, torch.Tensor] = {}
    losses = model.get_loss_dict(outputs, batch, metrics)
    if loss_mode == "main":
        loss = losses["main_loss"]
    elif loss_mode == "tbap_only":
        if "tbap_loss" not in losses:
            raise RuntimeError("TBAP-only audit requested but tbap_loss was not produced")
        loss = losses["tbap_loss"]
    elif loss_mode == "main_plus_tbap":
        loss = losses["main_loss"] + losses.get("tbap_loss", losses["main_loss"].new_zeros(()))
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")
    loss.backward()
    grad = _grad_summary(model)
    support_diag = {}
    pixel_depth_bins: Optional[Dict[str, Any]] = None
    if enabled and "tbap_rgb_object_proxy" in outputs:
        support, weights, diag = model._tbap_support_and_weights(outputs)  # diagnostic-only reuse
        support_diag = {
            "support": _stats(support),
            "support_gt_0p25_fraction": float((support > 0.25).float().mean().item()),
            "transmission": _stats(diag["transmission"]),
            "raw_weight": _stats(diag["raw_weight"]),
            "normalized_weight": _stats(weights),
            "tbap_loss_weighted": float(losses.get("tbap_loss", loss.new_zeros(())).detach().item()),
            "tbap_proxy_abs_diff_rgb_object": _stats(outputs.get("tbap_proxy_abs_diff_rgb_object", support.new_zeros(1))),
        }
        pixel_depth_bins = _pixel_support_summary(outputs, support, weights, diag)
    result = {
        "enabled": bool(enabled),
        "loss_mode": loss_mode,
        "losses": {key: float(value.detach().item()) for key, value in losses.items()},
        "metrics": {key: float(value.detach().item()) for key, value in metrics.items()},
        "gradients": grad,
        "support_diag": support_diag,
        "pixel_depth_bins": pixel_depth_bins,
    }
    _zero_grads(model)
    return result


def _parse_indices(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip() != ""]


def _train_item(datamanager: Any, image_idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
    data = datamanager.cached_train[image_idx].copy()
    if "image" in data and hasattr(data["image"], "to"):
        data["image"] = data["image"].to(datamanager.device)
    camera = datamanager.train_cameras[image_idx : image_idx + 1].to(datamanager.device)
    if camera.metadata is None:
        camera.metadata = {}
    camera.metadata["cam_idx"] = image_idx
    return camera, data


def _selected_items(datamanager: Any, args: argparse.Namespace) -> List[Tuple[int, Any, Dict[str, torch.Tensor]]]:
    explicit = _parse_indices(args.image_indices)
    if args.split == "train":
        total = len(datamanager.train_dataset)
        if explicit is None:
            start = int(args.image_index)
            indices = list(range(start, min(total, start + int(args.max_images))))
        else:
            indices = explicit[: int(args.max_images)]
        return [(idx, *_train_item(datamanager, idx)) for idx in indices if 0 <= idx < total]

    rows: List[Tuple[int, Any, Dict[str, torch.Tensor]]] = []
    keep = set(explicit) if explicit is not None else None
    for image_idx, (camera, batch) in enumerate(datamanager.fixed_indices_eval_dataloader):
        if keep is None and image_idx < args.image_index:
            continue
        if keep is not None and image_idx not in keep:
            continue
        rows.append((image_idx, camera, batch))
        if len(rows) >= args.max_images:
            break
    return rows


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(args.load_config, update_config_callback=update_config)
    pipeline.eval()
    model = pipeline.model
    if args.step_override is not None:
        model.step = int(args.step_override)
    else:
        model.step = int(step)

    rows: List[Dict[str, Any]] = []
    selected = _selected_items(pipeline.datamanager, args)
    for image_idx, camera, batch in selected:
        main = _run_case(model, camera, batch, args, enabled=False, loss_mode="main")
        tbap_only = _run_case(model, camera, batch, args, enabled=True, loss_mode="tbap_only")
        tbap = _run_case(model, camera, batch, args, enabled=True, loss_mode="main_plus_tbap")
        rows.append(
            {
                "image_index": image_idx,
                "split": args.split,
                "main": main,
                "tbap_only": tbap_only,
                "main_plus_tbap": tbap,
                "tbap_only_to_main_ratio": _compare(main["gradients"], tbap_only["gradients"]),
                "ratio": _compare(main["gradients"], tbap["gradients"]),
            }
        )

    repo = Path(__file__).resolve().parents[2]
    result = {
        "experiment": "tbap_gradient_audit",
        "scene_name": args.scene_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "step_override": args.step_override,
        "split": args.split,
        "image_indices": [row["image_index"] for row in rows],
        "git_commit": _git_commit(repo),
        "lambda_tbap": args.lambda_tbap,
        "gamma": args.gamma,
        "max_weight": args.max_weight,
        "weight_mode": args.weight_mode,
        "support_mode": args.support_mode,
        "support_top_fraction": args.support_top_fraction,
        "depth_weight_strength": args.depth_weight_strength,
        "transmission_floor": args.transmission_floor,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "tbap_gradient_audit.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-name", type=str, default="unknown")
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--step-override", type=int, default=None)
    parser.add_argument("--split", type=str, choices=("train", "eval"), default="eval")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--image-indices", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--lambda-tbap", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float, default=3.0)
    parser.add_argument(
        "--weight-mode",
        type=str,
        choices=("channel_transmission", "depth", "scalar_transmission", "median_transmission", "luma_transmission"),
        default="channel_transmission",
    )
    parser.add_argument("--support-mode", type=str, choices=("legacy", "object_far"), default="legacy")
    parser.add_argument("--support-top-fraction", type=float, default=0.0)
    parser.add_argument("--depth-weight-strength", type=float, default=1.0)
    parser.add_argument("--transmission-floor", type=float, default=0.08)
    parser.add_argument("--transmission-info-temp", type=float, default=0.04)
    parser.add_argument("--object-accum-mid", type=float, default=0.35)
    parser.add_argument("--object-accum-temp", type=float, default=0.08)
    parser.add_argument("--object-concentration-kappa", type=float, default=0.25)
    parser.add_argument("--far-depth-mid", type=float, default=0.60)
    parser.add_argument("--far-depth-temp", type=float, default=0.15)
    parser.add_argument("--depth-normalize-mode", type=str, choices=("max", "p95"), default="p95")
    parser.add_argument("--smooth-l1-beta", type=float, default=0.01)
    args = parser.parse_args()

    result = diagnose(args)
    compact = []
    for row in result["rows"]:
        compact.append(
            {
                "image_index": row["image_index"],
                "appearance_total_norm_ratio": row["ratio"]["appearance_total_norm_ratio"],
                "tbap_only_appearance_to_main": row["tbap_only_to_main_ratio"]["appearance_total_norm_ratio"],
                "nearest_dc_ratio": row["ratio"]["depth_bins"]["q0_25_nearest"]["dc_abs_mean_ratio"],
                "farthest_dc_ratio": row["ratio"]["depth_bins"]["q75_100_farthest"]["dc_abs_mean_ratio"],
                "tbap_only_nearest_dc_to_main": row["tbap_only_to_main_ratio"]["depth_bins"]["q0_25_nearest"]["dc_abs_mean_ratio"],
                "tbap_only_farthest_dc_to_main": row["tbap_only_to_main_ratio"]["depth_bins"]["q75_100_farthest"]["dc_abs_mean_ratio"],
                "farthest_r_ratio": row["ratio"]["depth_bins"]["q75_100_farthest"]["dc_abs_r_ratio"],
                "farthest_g_ratio": row["ratio"]["depth_bins"]["q75_100_farthest"]["dc_abs_g_ratio"],
                "farthest_b_ratio": row["ratio"]["depth_bins"]["q75_100_farthest"]["dc_abs_b_ratio"],
                "farthest_channel_ratio_change": row["ratio"]["depth_bins"]["q75_100_farthest"][
                    "dc_channel_ratio_change_max"
                ],
                "support_mean": row["main_plus_tbap"]["support_diag"].get("support", {}).get("mean", 0.0),
            }
        )
    print(json.dumps(compact, indent=2))
    print(f"saved={args.output_dir / 'tbap_gradient_audit.json'}")


if __name__ == "__main__":
    main()
