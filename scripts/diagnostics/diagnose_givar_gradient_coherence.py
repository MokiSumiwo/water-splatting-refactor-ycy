#!/usr/bin/env python
"""Phase-0 GIVAR gradient-coherence diagnostic.

The diagnostic is read-only: it loads an existing checkpoint, evaluates fixed
train or eval views, computes standard M1 reconstruction gradients, and reports
whether Gaussian-ID appearance consensus is dense enough to justify training
GIVAR.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from nerfstudio.utils.eval_utils import eval_setup

from water_splatting.appearance import (
    build_givar_detail_residual,
    build_givar_gaussian_evidence,
    build_givar_reliability_map,
    compute_givar_dc_gate,
    pearson_corr,
)


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _stats(values: torch.Tensor) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
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


def _camera_items(pipeline: Any, split: str, max_images: int, device: torch.device) -> Iterator[Tuple[int, Any, Dict[str, Any]]]:
    max_count = max_images if max_images > 0 else 10**9
    if split == "eval":
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_count:
                break
            yield image_idx, camera.to(device) if hasattr(camera, "to") else camera, batch
        return

    dataset = pipeline.datamanager.train_dataset
    count = min(len(dataset.cameras), max_count)
    for image_idx in range(count):
        camera = dataset.cameras[image_idx : image_idx + 1]
        image = dataset[image_idx]["image"]
        yield image_idx, camera.to(device) if hasattr(camera, "to") else camera, {"image": image}


def _clear_grads(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)
    for name in ("xys", "xys_grad_abs", "xys_grad_abs_proxy", "xys_grad_abs_capacity"):
        value = getattr(model, name, None)
        if value is not None and getattr(value, "grad", None) is not None:
            value.grad = None


def _per_gaussian_norm(value: Optional[torch.Tensor], size: int, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.zeros(size, device=device)
    flat = value.detach().float().reshape(size, -1)
    return torch.linalg.vector_norm(flat, dim=-1)


def _normalized_direction(value: torch.Tensor) -> torch.Tensor:
    flat = value.detach().float().reshape(value.shape[0], -1)
    return flat / torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(1e-12)


def _coherence(direction_sum: torch.Tensor, weight_sum: torch.Tensor) -> torch.Tensor:
    return (torch.linalg.vector_norm(direction_sum.float(), dim=-1) / weight_sum.float().clamp_min(1e-6)).clamp(0.0, 1.0)


def _gate_from_buffers(
    *,
    view_count: torch.Tensor,
    direction_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    magnitude_sum: torch.Tensor,
    view_direction_sum: torch.Tensor,
    min_view_count: int,
    coherence_threshold: float,
    min_view_spread: float,
    magnitude_quantile: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    gate, stats = compute_givar_dc_gate(
        view_count=view_count,
        grad_direction_sum=direction_sum,
        grad_weight_sum=weight_sum,
        grad_magnitude_sum=magnitude_sum,
        view_direction_sum=view_direction_sum,
        min_view_count=min_view_count,
        coherence_threshold=coherence_threshold,
        min_view_spread=min_view_spread,
        magnitude_quantile=magnitude_quantile,
        dc_enabled=True,
    )
    return gate > 0, stats


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        update_config_callback=_update_config,
    )
    model = pipeline.model
    device = model.device
    model.train()
    model.step = int(step)
    model.config.givar_enabled = False
    model.config.mvgar_enabled = False
    model.config.mvgar_diagnostic_only = False
    model.config.mcgr_enabled = False
    model.config.mcgr_diagnostic_only = False

    n = int(model.num_points)
    sh_dim = int(model.features_rest.detach().reshape(n, -1).shape[1]) if model.features_rest.numel() else 0
    dc_direction_sum = torch.zeros(n, 3, device=device)
    dc_weight_sum = torch.zeros(n, device=device)
    dc_magnitude_sum = torch.zeros(n, device=device)
    sh_direction_sum = torch.zeros(n, sh_dim, device=device) if sh_dim > 0 else None
    sh_weight_sum = torch.zeros(n, device=device)
    sh_magnitude_sum = torch.zeros(n, device=device)
    view_count = torch.zeros(n, device=device)
    view_direction_sum = torch.zeros(n, 3, device=device)
    detail_sum = torch.zeros(n, device=device)
    reliability_sum = torch.zeros(n, device=device)
    texture_sum = torch.zeros(n, device=device)
    dc_corr_detail: List[torch.Tensor] = []
    dc_corr_grad: List[torch.Tensor] = []
    sh_corr_detail: List[torch.Tensor] = []
    sh_corr_grad: List[torch.Tensor] = []
    image_rows: List[Dict[str, Any]] = []

    for image_idx, camera, batch in _camera_items(pipeline, args.split, int(args.max_images), device):
        _clear_grads(model)
        outputs = model.get_outputs(camera)
        gt_img = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["pred_image"]
        if "mask" in batch:
            mask = model._downscale_if_required(batch["mask"]).to(device)
            gt_img = gt_img * mask
            pred_img = pred_img * mask
        loss_dict = model.get_loss_dict(outputs, {"image": batch["image"]}, {})
        main_loss = loss_dict["main_loss"]
        dc_grad, sh_grad = torch.autograd.grad(
            main_loss,
            [model.features_dc, model.features_rest],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        dc_mag = _per_gaussian_norm(dc_grad, n, device)
        sh_mag = _per_gaussian_norm(sh_grad, n, device)
        detail_map = build_givar_detail_residual(
            pred_img,
            gt_img,
            highpass_weight=float(args.highpass_weight),
        )
        reliability_map, reliability_components = build_givar_reliability_map(
            gt_img=gt_img,
            accumulation=outputs["accumulation"],
            depth_std_relative=outputs["depth_std_relative"],
            texture_mid=float(args.texture_mid),
            texture_temp=float(args.texture_temp),
            accumulation_mid=float(args.accumulation_mid),
            accumulation_temp=float(args.accumulation_temp),
            depth_std_kappa=float(args.depth_std_kappa),
        )
        evidence = build_givar_gaussian_evidence(
            detail_map=detail_map,
            reliability_map=reliability_map,
            texture_map=reliability_components["texture"],
            xys=model.xys,
            radii=model.radii,
            means=model.means,
            camera_position=camera.camera_to_worlds[0, :3, 3],
        )
        visible = (model.radii.detach().reshape(-1) > 0).to(device)
        keep_dc = visible & (evidence.weight > 0) & torch.isfinite(dc_mag) & (dc_mag > 1e-12)
        keep_sh = visible & (evidence.weight > 0) & torch.isfinite(sh_mag) & (sh_mag > 1e-12)
        if bool(keep_dc.any().item()):
            q95 = torch.quantile(dc_mag[keep_dc].float(), 0.95).clamp_min(1e-12)
            w = (evidence.weight[keep_dc] * (dc_mag[keep_dc] / q95).clamp(0.0, 1.0)).clamp_min(0.0)
            idx = torch.where(keep_dc)[0]
            dc_direction_sum[idx] += w[:, None] * _normalized_direction(dc_grad)[idx]
            dc_weight_sum[idx] += w
            dc_magnitude_sum[idx] += w * dc_mag[idx]
            view_count[idx] += 1.0
            view_direction_sum[idx] += evidence.view_direction[idx]
            detail_sum[idx] += w * evidence.detail[idx]
            reliability_sum[idx] += w * evidence.reliability[idx]
            texture_sum[idx] += w * evidence.texture[idx]
            dc_corr_detail.append(evidence.detail[keep_dc].detach().float().cpu())
            dc_corr_grad.append(dc_mag[keep_dc].detach().float().cpu())
        if sh_direction_sum is not None and sh_grad is not None and bool(keep_sh.any().item()):
            q95_sh = torch.quantile(sh_mag[keep_sh].float(), 0.95).clamp_min(1e-12)
            w_sh = (evidence.weight[keep_sh] * (sh_mag[keep_sh] / q95_sh).clamp(0.0, 1.0)).clamp_min(0.0)
            idx_sh = torch.where(keep_sh)[0]
            sh_direction_sum[idx_sh] += w_sh[:, None] * _normalized_direction(sh_grad)[idx_sh]
            sh_weight_sum[idx_sh] += w_sh
            sh_magnitude_sum[idx_sh] += w_sh * sh_mag[idx_sh]
            sh_corr_detail.append(evidence.detail[keep_sh].detach().float().cpu())
            sh_corr_grad.append(sh_mag[keep_sh].detach().float().cpu())
        image_rows.append(
            {
                "image_index": int(image_idx),
                "visible_gaussians": int(visible.sum().item()),
                "evidence_gaussians": int((evidence.weight > 0).sum().item()),
                "detail_mean": float(detail_map.float().mean().item()),
                "reliability_mean": float(reliability_map.float().mean().item()),
                "dc_grad_p95": float(torch.quantile(dc_mag[torch.isfinite(dc_mag)].float(), 0.95).item())
                if bool(torch.isfinite(dc_mag).any().item())
                else 0.0,
                "sh_grad_p95": float(torch.quantile(sh_mag[torch.isfinite(sh_mag)].float(), 0.95).item())
                if bool(torch.isfinite(sh_mag).any().item())
                else 0.0,
            }
        )
        _clear_grads(model)

    dc_eligible, dc_gate_stats = _gate_from_buffers(
        view_count=view_count,
        direction_sum=dc_direction_sum,
        weight_sum=dc_weight_sum,
        magnitude_sum=dc_magnitude_sum,
        view_direction_sum=view_direction_sum,
        min_view_count=int(args.min_view_count),
        coherence_threshold=float(args.dc_coherence_threshold),
        min_view_spread=float(args.min_view_spread),
        magnitude_quantile=float(args.gradient_magnitude_quantile),
    )
    sh_coherence = _coherence(sh_direction_sum, sh_weight_sum) if sh_direction_sum is not None else torch.zeros(n, device=device)
    mean_sh_mag = sh_magnitude_sum / sh_weight_sum.clamp_min(1e-6)
    view_mean = view_direction_sum / view_count.clamp_min(1.0)[:, None]
    view_spread = (1.0 - view_mean.norm(dim=-1)).clamp(0.0, 1.0)
    sh_pre = (
        (view_count >= int(args.min_view_count))
        & (sh_coherence >= float(args.sh_coherence_threshold))
        & (view_spread >= float(args.min_view_spread))
        & torch.isfinite(mean_sh_mag)
    )
    if bool(sh_pre.any().item()):
        sh_thr = torch.quantile(mean_sh_mag[sh_pre].float(), float(args.gradient_magnitude_quantile))
        sh_eligible = sh_pre & (mean_sh_mag >= sh_thr)
    else:
        sh_eligible = torch.zeros(n, dtype=torch.bool, device=device)

    mean_detail = detail_sum / dc_weight_sum.clamp_min(1e-6)
    mean_reliability = reliability_sum / dc_weight_sum.clamp_min(1e-6)
    mean_texture = texture_sum / dc_weight_sum.clamp_min(1e-6)
    dc_detail = torch.cat(dc_corr_detail) if dc_corr_detail else torch.zeros(0)
    dc_grad_values = torch.cat(dc_corr_grad) if dc_corr_grad else torch.zeros(0)
    sh_detail = torch.cat(sh_corr_detail) if sh_corr_detail else torch.zeros(0)
    sh_grad_values = torch.cat(sh_corr_grad) if sh_corr_grad else torch.zeros(0)
    open_water_like = dc_eligible & ((mean_reliability < 0.20) | (mean_texture < float(args.texture_mid)))
    result = {
        "scene_name": args.scene_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "split": args.split,
        "max_images": int(args.max_images),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "total_gaussians": int(n),
        "multi_view_visible_gaussian_fraction": float((view_count >= int(args.min_view_count)).float().mean().item()),
        "dc_gradient_magnitude": _stats(dc_magnitude_sum / dc_weight_sum.clamp_min(1e-6)),
        "sh_gradient_magnitude": _stats(mean_sh_mag),
        "dc_gradient_coherence": _stats(_coherence(dc_direction_sum, dc_weight_sum)[view_count > 0]),
        "sh_gradient_coherence": _stats(sh_coherence[view_count > 0]),
        "view_direction_spread": _stats(view_spread[view_count > 0]),
        "eligible_dc_fraction": float(dc_eligible.float().mean().item()) if n else 0.0,
        "eligible_dc_count": int(dc_eligible.sum().item()),
        "eligible_sh_fraction": float(sh_eligible.float().mean().item()) if n else 0.0,
        "eligible_sh_count": int(sh_eligible.sum().item()),
        "high_frequency_residual_vs_dc_gradient_correlation": pearson_corr(dc_detail, dc_grad_values),
        "high_frequency_residual_vs_sh_gradient_correlation": pearson_corr(sh_detail, sh_grad_values),
        "sh_to_dc_gradient_energy_ratio": float(sh_magnitude_sum.sum().item() / dc_magnitude_sum.sum().clamp_min(1e-12).item()),
        "eligible_open_water_like_fraction": float(open_water_like.float().sum().item() / max(int(dc_eligible.sum().item()), 1)),
        "eligible_detail": _stats(mean_detail[dc_eligible]),
        "eligible_reliability": _stats(mean_reliability[dc_eligible]),
        "eligible_texture": _stats(mean_texture[dc_eligible]),
        "dc_gate_stats": dc_gate_stats,
        "images": image_rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--scene-name", type=str, default="")
    parser.add_argument("--split", choices=["train", "eval"], default="train")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/givar_diagnostics"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--highpass-weight", type=float, default=0.35)
    parser.add_argument("--min-view-count", type=int, default=4)
    parser.add_argument("--dc-coherence-threshold", type=float, default=0.55)
    parser.add_argument("--sh-coherence-threshold", type=float, default=0.50)
    parser.add_argument("--min-view-spread", type=float, default=0.02)
    parser.add_argument("--gradient-magnitude-quantile", type=float, default=0.75)
    parser.add_argument("--accumulation-mid", type=float, default=0.40)
    parser.add_argument("--accumulation-temp", type=float, default=0.08)
    parser.add_argument("--depth-std-kappa", type=float, default=0.25)
    parser.add_argument("--texture-mid", type=float, default=0.10)
    parser.add_argument("--texture-temp", type=float, default=0.05)
    args = parser.parse_args()

    result = diagnose(args)
    output_json = args.output_json
    if output_json is None:
        scene = args.scene_name or "scene"
        output_json = args.output_dir / f"{scene}_givar_gradient_coherence.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    summary_keys = [
        "multi_view_visible_gaussian_fraction",
        "eligible_dc_fraction",
        "eligible_sh_fraction",
        "high_frequency_residual_vs_dc_gradient_correlation",
        "eligible_open_water_like_fraction",
        "sh_to_dc_gradient_energy_ratio",
    ]
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
