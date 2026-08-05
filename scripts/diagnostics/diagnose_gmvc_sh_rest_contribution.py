#!/usr/bin/env python
"""Audit Full-SH versus DC-only rendering for GMVC checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from nerfstudio.utils.eval_utils import eval_setup
from torch import Tensor


R_EDIT = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _setup_pipeline(config_path: Path, step: int, test_mode: str) -> Tuple[Any, Any, Path, int]:
    def _update_config(config: Any) -> Any:
        config.load_step = int(step)
        return config

    return eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode=test_mode,
        update_config_callback=_update_config,
    )


def _parse_label_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be LABEL=PATH")
    label, path_text = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("label cannot be empty")
    return label, Path(path_text)


def _parse_run_spec(value: str) -> Tuple[str, Path, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run spec must be LABEL=CONFIG:STEP")
    label, rest = value.split("=", 1)
    if ":" not in rest:
        raise argparse.ArgumentTypeError("run spec must be LABEL=CONFIG:STEP")
    config_text, step_text = rest.rsplit(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    try:
        step = int(step_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid step in {value}") from exc
    return label, Path(config_text), step


def _nearest_rank(values: Tensor, q: float) -> float:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0
    rank = max(1, min(int(values.numel()), math.ceil(float(q) * int(values.numel()))))
    return float(values.kthvalue(rank).values.item())


def _stats(values: Tensor) -> Dict[str, float]:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p50": _nearest_rank(values, 0.50),
        "p75": _nearest_rank(values, 0.75),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "max": float(values.max().item()),
    }


def _weighted_mean(values: Tensor, weights: Tensor, eps: float = 1e-8) -> float:
    if values.numel() == 0:
        return 0.0
    denom = weights.sum().clamp_min(float(eps))
    return float((values * weights).sum().detach().cpu().item() / denom.detach().cpu().item())


def _pearson(x: Tensor, y: Tensor) -> Optional[float]:
    x = x.detach().float().reshape(-1).cpu()
    y = y.detach().float().reshape(-1).cpu()
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return None
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.square().mean() * y.square().mean())
    if float(denom.item()) <= 1e-12:
        return None
    return float((x * y).mean().item() / denom.item())


def _luma(rgb: Tensor) -> Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _residual_metrics(pred: Tensor, gt: Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0)
    gt = gt.detach().float().clamp(0.0, 1.0)
    pred_luma = _luma(pred)
    gt_luma = _luma(gt)
    chroma_residual = (pred - pred_luma) - (gt - gt_luma)
    return {
        "rgb_l1": float((pred - gt).abs().mean().item()),
        "luminance_l1": float((pred_luma - gt_luma).abs().mean().item()),
        "chroma_l1": float(chroma_residual.abs().mean().item()),
    }


def _image_name(pipeline: Any, image_idx: int) -> str:
    dataset = pipeline.datamanager.eval_dataset
    try:
        filenames = dataset._dataparser_outputs.image_filenames
        return Path(filenames[int(image_idx)]).name
    except Exception:
        return f"eval_{int(image_idx):04d}"


def _save_hwc(path: Path, image: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().float().cpu()
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = image.expand(*image.shape[:2], 3)
    vutils.save_image(image.permute(2, 0, 1).clamp(0.0, 1.0), path)


def _normalize_map(value: Tensor, q_low: float = 0.01, q_high: float = 0.99) -> Tensor:
    value = value.detach().float()
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return torch.zeros_like(value)
    lo = torch.quantile(finite, float(q_low))
    hi = torch.quantile(finite, float(q_high))
    return ((value - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)


def _render_with_degree(model: Any, camera: Any, active_sh_degree: Optional[int]) -> Dict[str, Tensor]:
    if active_sh_degree is None:
        return model.get_outputs_for_camera(camera=camera)
    original = model._get_active_sh_degree
    try:
        model._get_active_sh_degree = lambda: int(active_sh_degree)
        return model.get_outputs_for_camera(camera=camera)
    finally:
        model._get_active_sh_degree = original


def _derived_medium(outputs: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    depth = outputs["depth"].detach().float()
    medium_attn = outputs["medium_attn"].detach().float()
    medium_bs = outputs["medium_bs"].detach().float()
    b_inf = outputs.get("b_inf", outputs["medium_rgb"]).detach().float()
    transmission = torch.exp(-(medium_attn * depth).clamp_min(0.0))
    backscatter_endpoint = b_inf * (1.0 - torch.exp(-(medium_bs * depth).clamp_min(0.0)))
    return {
        "depth": depth,
        "transmission": transmission,
        "backscatter_endpoint": backscatter_endpoint,
        "actual_rgb_medium": outputs["rgb_medium"].detach().float(),
    }


def _ray_directions(camera: Any, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x_full = xx * (float(width) / max(float(width - 1), 1.0))
    y_full = yy * (float(height) / max(float(height - 1), 1.0))
    p_view = torch.stack(
        [
            (x_full - float(camera.cx.item())) / float(camera.fx.item()),
            (y_full - float(camera.cy.item())) / float(camera.fy.item()),
            torch.ones_like(xx),
        ],
        dim=-1,
    )
    p_view = p_view / torch.linalg.norm(p_view, dim=-1, keepdim=True).clamp_min(1e-8)
    rotation = camera.camera_to_worlds[0, :3, :3].to(device=device, dtype=dtype) @ R_EDIT.to(device=device, dtype=dtype)
    directions = p_view @ rotation.T
    return directions / torch.linalg.norm(directions, dim=-1, keepdim=True).clamp_min(1e-8)


def _pixel_correlations(sh_mag: Tensor, residual_mag: Tensor, medium: Mapping[str, Tensor], camera: Any) -> Dict[str, Optional[float]]:
    device = sh_mag.device
    dtype = sh_mag.dtype
    height, width = sh_mag.shape[:2]
    dirs = _ray_directions(camera, height, width, device=device, dtype=dtype)
    depth = medium["depth"][..., 0]
    transmission = medium["transmission"].mean(dim=-1)
    backscatter = medium["backscatter_endpoint"].mean(dim=-1)
    return {
        "sh_abs_vs_depth": _pearson(sh_mag, depth),
        "sh_abs_vs_transmission": _pearson(sh_mag, transmission),
        "sh_abs_vs_backscatter": _pearson(sh_mag, backscatter),
        "sh_abs_vs_rgb_residual": _pearson(sh_mag, residual_mag),
        "sh_abs_vs_viewdir_x": _pearson(sh_mag, dirs[..., 0]),
        "sh_abs_vs_viewdir_y": _pearson(sh_mag, dirs[..., 1]),
        "sh_abs_vs_viewdir_z": _pearson(sh_mag, dirs[..., 2]),
    }


def _make_contact(
    gt: Tensor,
    full_rgb: Tensor,
    dc_rgb: Tensor,
    contribution: Tensor,
    medium: Mapping[str, Tensor],
) -> Tensor:
    full_residual = (full_rgb - gt).abs().mean(dim=-1, keepdim=True).expand_as(gt)
    dc_residual = (dc_rgb - gt).abs().mean(dim=-1, keepdim=True).expand_as(gt)
    residual_improvement = (0.5 + 4.0 * (dc_residual[..., :1] - full_residual[..., :1])).expand_as(gt)
    tiles = [
        gt.clamp(0.0, 1.0),
        full_rgb.clamp(0.0, 1.0),
        dc_rgb.clamp(0.0, 1.0),
        full_residual.clamp(0.0, 1.0),
        dc_residual.clamp(0.0, 1.0),
        (contribution.abs().mean(dim=-1, keepdim=True) * 6.0).expand_as(gt).clamp(0.0, 1.0),
        residual_improvement.clamp(0.0, 1.0),
        medium["transmission"].mean(dim=-1, keepdim=True).expand_as(gt).clamp(0.0, 1.0),
        medium["backscatter_endpoint"].mean(dim=-1, keepdim=True).expand_as(gt).clamp(0.0, 1.0),
        _normalize_map(medium["depth"]).expand_as(gt),
    ]
    separator = torch.ones((gt.shape[0], max(gt.shape[1] // 220, 4), 3), dtype=gt.dtype)
    row: List[Tensor] = []
    for idx, tile in enumerate(tiles):
        if idx:
            row.append(separator)
        row.append(tile.detach().float().cpu().clamp(0.0, 1.0))
    return torch.cat(row, dim=1)


def _sample_hwc(image: Tensor, xy: Tensor) -> Tensor:
    if xy.numel() == 0:
        return torch.empty((0, image.shape[-1]), dtype=image.dtype, device=image.device)
    h, w = image.shape[:2]
    xy = xy.to(device=image.device, dtype=image.dtype)
    grid_x = 2.0 * xy[:, 0] / max(float(w - 1), 1.0) - 1.0
    grid_y = 2.0 * xy[:, 1] / max(float(h - 1), 1.0) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    nchw = image.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(nchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled[0, :, :, 0].T.contiguous()


def _track_indices(obs: Mapping[str, Tensor], track_ids: Tensor) -> Tensor:
    starts = obs["track_starts"].long()
    lengths = obs["track_lengths"].long()
    chunks = []
    for track_id in track_ids.long().tolist():
        start = int(starts[track_id].item())
        length = int(lengths[track_id].item())
        if length > 0:
            chunks.append(torch.arange(start, start + length, dtype=torch.long))
    if not chunks:
        return torch.empty((0,), dtype=torch.long)
    return torch.cat(chunks, dim=0)


def _select_tracks(obs: Mapping[str, Tensor], max_tracks: int, seed: int) -> Tensor:
    track_ids = obs["track_ids"].long()
    if max_tracks > 0 and int(track_ids.numel()) > max_tracks:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        keep = torch.randperm(int(track_ids.numel()), generator=generator)[:max_tracks]
        track_ids = track_ids[keep]
    return track_ids.sort().values


def _camera_for_image(pipeline: Any, image_index: int) -> Any:
    camera = pipeline.datamanager.train_dataset.cameras[int(image_index) : int(image_index) + 1]
    return camera.to(pipeline.model.device) if hasattr(camera, "to") else camera


def _render_train_image_contribution(pipeline: Any, image_index: int) -> Dict[str, Tensor]:
    model = pipeline.model
    camera = _camera_for_image(pipeline, image_index)
    with torch.no_grad():
        full_outputs = _render_with_degree(model, camera, None)
        dc_outputs = _render_with_degree(model, camera, 0)
    medium = _derived_medium(full_outputs)
    full_rgb = full_outputs["pred_image"].detach().float().clamp(0.0, 1.0)
    dc_rgb = dc_outputs["pred_image"].detach().float().clamp(0.0, 1.0)
    return {
        "contribution": full_rgb - dc_rgb,
        "sh_abs": (full_rgb - dc_rgb).abs().mean(dim=-1, keepdim=True),
        "full_rgb": full_rgb,
        "transmission": medium["transmission"].mean(dim=-1, keepdim=True),
        "backscatter": medium["backscatter_endpoint"].mean(dim=-1, keepdim=True),
    }


def _track_stats(values: Tensor, local_track: Tensor, track_count: int, weights: Tensor) -> Tuple[Tensor, Tensor]:
    variances: List[Tensor] = []
    track_weights: List[Tensor] = []
    for track_id in range(track_count):
        rows = torch.nonzero(local_track == int(track_id), as_tuple=False).reshape(-1)
        if int(rows.numel()) < 2:
            continue
        vals = values[rows].float()
        variances.append(vals.var(dim=0, unbiased=False).mean().reshape(1))
        track_weights.append(weights[rows].float().mean().reshape(1))
    if not variances:
        return torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)
    return torch.cat(variances), torch.cat(track_weights)


def _evaluate_track_bank(
    pipeline: Any,
    bank_label: str,
    bank_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    bank = torch.load(bank_path, map_location="cpu")
    obs = bank["observations"]
    selected_tracks = _select_tracks(obs, args.max_tracks, args.seed)
    row_indices = _track_indices(obs, selected_tracks)
    old_to_local = torch.full((int(obs["track_ids"].max().item()) + 1,), -1, dtype=torch.long)
    old_to_local[selected_tracks.long()] = torch.arange(int(selected_tracks.numel()), dtype=torch.long)
    local_track = old_to_local[obs["track_id"][row_indices].long()]
    image_indices = obs["image_index"][row_indices].long()
    xy_all = obs["xy"][row_indices].float()
    weights = obs["weight"][row_indices].float()
    row_count = int(row_indices.numel())

    contribution = torch.empty((row_count, 3), dtype=torch.float32)
    sh_abs = torch.empty((row_count,), dtype=torch.float32)
    full_residual = torch.empty((row_count,), dtype=torch.float32)
    transmission = torch.empty((row_count,), dtype=torch.float32)
    backscatter = torch.empty((row_count,), dtype=torch.float32)
    cache: Dict[int, Dict[str, Tensor]] = {}
    model = pipeline.model
    for image_index in image_indices.unique(sorted=True).tolist():
        local = torch.nonzero(image_indices == int(image_index), as_tuple=False).reshape(-1)
        if int(local.numel()) == 0:
            continue
        cache[int(image_index)] = _render_train_image_contribution(pipeline, int(image_index))
        rendered = cache[int(image_index)]
        xy = xy_all[local].to(device=model.device)
        contribution[local] = _sample_hwc(rendered["contribution"], xy).detach().float().cpu()
        sh_abs[local] = _sample_hwc(rendered["sh_abs"], xy).detach().float().cpu().reshape(-1)
        full_rgb_sampled = _sample_hwc(rendered["full_rgb"], xy).detach().float().cpu()
        full_residual[local] = (full_rgb_sampled - obs["gt"][row_indices[local]].float()).abs().mean(dim=-1)
        transmission[local] = _sample_hwc(rendered["transmission"], xy).detach().float().cpu().reshape(-1)
        backscatter[local] = _sample_hwc(rendered["backscatter"], xy).detach().float().cpu().reshape(-1)
        del cache[int(image_index)]

    variances, track_weights = _track_stats(contribution, local_track, int(selected_tracks.numel()), weights)
    ray_direction = obs["ray_direction"][row_indices].float()
    fixed_depth = obs["fixed_depth"][row_indices].float()
    return {
        "bank_label": bank_label,
        "track_bank": str(bank_path),
        "bank_metadata": {
            "bank_type": bank["metadata"].get("bank_type", ""),
            "split": bank["metadata"].get("split", ""),
            "step": bank["metadata"].get("step", None),
            "v2_track_count": int(bank["metadata"].get("v2_track_count", 0)),
            "v2_observation_count": int(bank["metadata"].get("v2_observation_count", 0)),
        },
        "selected_track_count": int(selected_tracks.numel()),
        "row_count": row_count,
        "e_sh_var": {
            "weighted_mean": _weighted_mean(variances, track_weights) if variances.numel() else 0.0,
            "stats": _stats(variances),
        },
        "row_stats": {
            "sh_abs": _stats(sh_abs),
            "full_rgb_residual": _stats(full_residual),
            "transmission": _stats(transmission),
            "backscatter": _stats(backscatter),
            "fixed_depth": _stats(fixed_depth),
        },
        "correlations": {
            "sh_abs_vs_fixed_depth": _pearson(sh_abs, fixed_depth),
            "sh_abs_vs_transmission": _pearson(sh_abs, transmission),
            "sh_abs_vs_backscatter": _pearson(sh_abs, backscatter),
            "sh_abs_vs_full_rgb_residual": _pearson(sh_abs, full_residual),
            "sh_abs_vs_raydir_x": _pearson(sh_abs, ray_direction[:, 0]),
            "sh_abs_vs_raydir_y": _pearson(sh_abs, ray_direction[:, 1]),
            "sh_abs_vs_raydir_z": _pearson(sh_abs, ray_direction[:, 2]),
        },
    }


def _mean_rows(rows: List[Mapping[str, Any]]) -> Dict[str, float]:
    keys = [
        "full_psnr",
        "full_ssim",
        "full_lpips",
        "dc_psnr",
        "dc_ssim",
        "dc_lpips",
        "psnr_delta_full_minus_dc",
        "ssim_delta_full_minus_dc",
        "lpips_delta_full_minus_dc",
        "rgb_l1_gain_dc_minus_full",
        "luminance_l1_gain_dc_minus_full",
        "chroma_l1_gain_dc_minus_full",
        "sh_abs_mean",
        "full_rgb_l1",
        "dc_rgb_l1",
        "transmission_mean",
        "backscatter_endpoint_mean",
        "depth_mean",
    ]
    return {key: float(sum(float(row.get(key, 0.0)) for row in rows) / max(len(rows), 1)) for key in keys}


def _eval_views(label: str, pipeline: Any, step: int, args: argparse.Namespace) -> Dict[str, Any]:
    model = pipeline.model
    model.eval()
    rows: List[Dict[str, Any]] = []
    view_tensors: Dict[int, Dict[str, Tensor]] = {}
    data_loader = pipeline.datamanager.fixed_indices_eval_dataloader
    if args.max_images > 0:
        data_loader = data_loader[: int(args.max_images)]
    with torch.no_grad():
        for view_index, (camera, batch) in enumerate(data_loader):
            full_outputs = _render_with_degree(model, camera, None)
            full_metrics, full_images = model.get_image_metrics_and_images(full_outputs, batch)
            dc_outputs = _render_with_degree(model, camera, 0)
            dc_metrics, dc_images = model.get_image_metrics_and_images(dc_outputs, batch)
            gt = full_images["gt"].detach().float().clamp(0.0, 1.0).cpu()
            full_rgb = full_outputs["pred_image"].detach().float().clamp(0.0, 1.0).cpu()
            dc_rgb = dc_outputs["pred_image"].detach().float().clamp(0.0, 1.0).cpu()
            contribution = full_rgb - dc_rgb
            sh_mag = contribution.abs().mean(dim=-1)
            full_extra = _residual_metrics(full_rgb, gt)
            dc_extra = _residual_metrics(dc_rgb, gt)
            medium = {key: value.detach().float().cpu() for key, value in _derived_medium(full_outputs).items()}
            residual_mag = (full_rgb - gt).abs().mean(dim=-1)
            correlations = _pixel_correlations(sh_mag, residual_mag, medium, camera.to(model.device))
            image_idx_raw = batch.get("image_idx", view_index)
            image_idx = int(image_idx_raw.item() if torch.is_tensor(image_idx_raw) else image_idx_raw)
            row = {
                "view_index": int(view_index),
                "image_idx": image_idx,
                "image_name": _image_name(pipeline, image_idx),
                "full_psnr": float(full_metrics.get("psnr", 0.0)),
                "full_ssim": float(full_metrics.get("ssim", 0.0)),
                "full_lpips": float(full_metrics.get("lpips", 0.0)),
                "dc_psnr": float(dc_metrics.get("psnr", 0.0)),
                "dc_ssim": float(dc_metrics.get("ssim", 0.0)),
                "dc_lpips": float(dc_metrics.get("lpips", 0.0)),
                "psnr_delta_full_minus_dc": float(full_metrics.get("psnr", 0.0) - dc_metrics.get("psnr", 0.0)),
                "ssim_delta_full_minus_dc": float(full_metrics.get("ssim", 0.0) - dc_metrics.get("ssim", 0.0)),
                "lpips_delta_full_minus_dc": float(full_metrics.get("lpips", 0.0) - dc_metrics.get("lpips", 0.0)),
                "rgb_l1_gain_dc_minus_full": float(dc_extra["rgb_l1"] - full_extra["rgb_l1"]),
                "luminance_l1_gain_dc_minus_full": float(dc_extra["luminance_l1"] - full_extra["luminance_l1"]),
                "chroma_l1_gain_dc_minus_full": float(dc_extra["chroma_l1"] - full_extra["chroma_l1"]),
                "full_rgb_l1": full_extra["rgb_l1"],
                "dc_rgb_l1": dc_extra["rgb_l1"],
                "full_luminance_l1": full_extra["luminance_l1"],
                "dc_luminance_l1": dc_extra["luminance_l1"],
                "full_chroma_l1": full_extra["chroma_l1"],
                "dc_chroma_l1": dc_extra["chroma_l1"],
                "sh_abs_mean": float(sh_mag.mean().item()),
                "sh_abs_p95": _nearest_rank(sh_mag, 0.95),
                "transmission_mean": float(medium["transmission"].mean().item()),
                "backscatter_endpoint_mean": float(medium["backscatter_endpoint"].mean().item()),
                "actual_rgb_medium_mean": float(medium["actual_rgb_medium"].mean().item()),
                "depth_mean": float(medium["depth"].mean().item()),
                "correlations": correlations,
            }
            view_dir = args.output_dir / label / f"view_{int(view_index):04d}"
            if not args.no_images:
                _save_hwc(view_dir / "gt.png", gt)
                _save_hwc(view_dir / "full_rgb.png", full_rgb)
                _save_hwc(view_dir / "dc_only_rgb.png", dc_rgb)
                _save_hwc(view_dir / "full_abs_residual.png", (full_rgb - gt).abs())
                _save_hwc(view_dir / "dc_abs_residual.png", (dc_rgb - gt).abs())
                _save_hwc(view_dir / "sh_rest_contribution_signed.png", 0.5 + 2.0 * contribution)
                _save_hwc(view_dir / "sh_rest_contribution_abs.png", sh_mag[..., None] * 6.0)
                _save_hwc(view_dir / "residual_improvement_full_over_dc.png", 0.5 + 4.0 * ((dc_rgb - gt).abs().mean(dim=-1, keepdim=True) - (full_rgb - gt).abs().mean(dim=-1, keepdim=True)))
                _save_hwc(view_dir / "transmission.png", medium["transmission"].mean(dim=-1, keepdim=True))
                _save_hwc(view_dir / "backscatter_endpoint.png", medium["backscatter_endpoint"].mean(dim=-1, keepdim=True))
                _save_hwc(view_dir / "actual_rgb_medium.png", medium["actual_rgb_medium"].clamp(0.0, 1.0))
                _save_hwc(view_dir / "depth.png", _normalize_map(medium["depth"]))
                contact = _make_contact(gt, full_rgb, dc_rgb, contribution, medium)
                _save_hwc(view_dir / "contact_sheet.png", contact)
                row["outputs"] = {
                    "view_dir": str(view_dir),
                    "contact_sheet": str(view_dir / "contact_sheet.png"),
                }
            (view_dir / "sh_rest_view_metrics.json").parent.mkdir(parents=True, exist_ok=True)
            (view_dir / "sh_rest_view_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf8")
            rows.append(row)
            view_tensors[int(view_index)] = {
                "sh_abs": sh_mag.cpu(),
                "full_abs_residual": (full_rgb - gt).abs().mean(dim=-1).cpu(),
            }
    return {
        "step": int(step),
        "per_view": rows,
        "mean": _mean_rows(rows),
        "view_tensors": view_tensors,
    }


def _cross_run_eval_correlations(evaluated: Mapping[str, Mapping[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    labels = list(evaluated.keys())
    if "MHOLD_15K" not in evaluated:
        return {}
    reference = evaluated["MHOLD_15K"]["view_tensors"]
    out: Dict[str, Any] = {}
    for other in labels:
        if other == "MHOLD_15K":
            continue
        common = sorted(set(reference.keys()) & set(evaluated[other]["view_tensors"].keys()))
        rows = []
        for view_index in common:
            mhold_sh = reference[view_index]["sh_abs"]
            mhold_res = reference[view_index]["full_abs_residual"]
            other_res = evaluated[other]["view_tensors"][view_index]["full_abs_residual"]
            improvement = other_res - mhold_res
            rows.append(
                {
                    "view_index": int(view_index),
                    "mhold_sh_abs_vs_residual_improvement": _pearson(mhold_sh, improvement),
                    "mean_residual_improvement": float(improvement.mean().item()),
                }
            )
        out[f"MHOLD_15K_vs_{other}"] = {
            "per_view": rows,
            "mean_correlation": float(
                sum(float(row["mhold_sh_abs_vs_residual_improvement"] or 0.0) for row in rows) / max(len(rows), 1)
            ),
            "mean_residual_improvement": float(
                sum(float(row["mean_residual_improvement"]) for row in rows) / max(len(rows), 1)
            ),
        }
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_specs = [_parse_run_spec(item) for item in args.run]
    track_banks = [_parse_label_path(item) for item in args.track_bank]
    evaluated: Dict[str, Dict[str, Any]] = {}
    runs_meta: Dict[str, Dict[str, Any]] = {}
    for label, config_path, step in run_specs:
        config, pipeline, checkpoint_path, loaded_step = _setup_pipeline(config_path, step, args.test_mode)
        result = _eval_views(label, pipeline, int(loaded_step), args)
        track_results = []
        for bank_label, bank_path in track_banks:
            track_results.append(_evaluate_track_bank(pipeline, bank_label, bank_path, args))
        result["track_banks"] = track_results
        evaluated[label] = result
        runs_meta[label] = {
            "config": str(config_path),
            "requested_step": int(step),
            "step": int(loaded_step),
            "checkpoint": str(checkpoint_path),
            "experiment_name": getattr(config, "experiment_name", ""),
            "method_name": getattr(config, "method_name", ""),
            "mean": result["mean"],
            "track_banks": track_results,
        }
        (args.output_dir / label / "sh_rest_run_summary.json").parent.mkdir(parents=True, exist_ok=True)
        (args.output_dir / label / "sh_rest_run_summary.json").write_text(
            json.dumps({k: v for k, v in result.items() if k != "view_tensors"}, indent=2),
            encoding="utf8",
        )
        del pipeline
        del config
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary: Dict[str, Any] = {
        "diagnostic": "gmvc_sh_rest_contribution",
        "test_mode": args.test_mode,
        "max_images": int(args.max_images),
        "max_tracks": int(args.max_tracks),
        "seed": int(args.seed),
        "runs": runs_meta,
        "cross_run_eval_correlations": _cross_run_eval_correlations(evaluated, args),
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / "gmvc_sh_rest_contribution_summary.json")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="Run as LABEL=CONFIG:STEP")
    parser.add_argument("--track-bank", action="append", default=[], help="Track bank as LABEL=PATH")
    parser.add_argument("--test-mode", choices=["test", "val", "inference"], default="test")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--max-tracks", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run(args)
    compact = {
        "output": str(args.output_json or (args.output_dir / "gmvc_sh_rest_contribution_summary.json")),
        "runs": {
            label: {
                "mean": meta["mean"],
                "track_banks": {
                    bank["bank_label"]: bank["e_sh_var"]["weighted_mean"]
                    for bank in meta.get("track_banks", [])
                },
            }
            for label, meta in result["runs"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
