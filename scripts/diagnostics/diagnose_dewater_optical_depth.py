#!/usr/bin/env python
"""No-training dewatering and direct optical-depth audit."""

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
import torch.nn.functional as F
from PIL import Image, ImageDraw
from nerfstudio.utils.eval_utils import eval_setup
from torch import Tensor


CHANNELS = ("r", "g", "b")
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
STAT_Q = (0.10, 0.50, 0.90, 0.95, 0.99)
T_Q = (0.01, 0.05, 0.10, 0.50, 0.90)
RGB_RANGE = "[0,1]"
NO_TONE = "none; tensors are clamped only for PNG conversion unless a named diagnostic mapping is used"
COLOR_SPACE = "renderer/dataset RGB tensor space; saved as RGB PNG without color conversion"


SCENE_CONFIGS = {
    "Curasao": (
        "outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/"
        "cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml"
    ),
    "JapaneseGradens": (
        "outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/"
        "cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml"
    ),
    "IUI3": (
        "outputs/gmvc_v3_four_scene_iui3_m1_seed42_15000/water-splatting/"
        "gmvc_v3_four_scene_iui3_m1_seed42_15000_20260806_gmvc_four_scene_p30_mhold_15k_m1_bootstrap/config.yml"
    ),
    "Panama": (
        "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
        "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _to_hwc(value: Tensor, key: str) -> Tensor:
    out = value.detach().float()
    if out.ndim == 2:
        out = out[..., None]
    if out.ndim != 3:
        raise ValueError(f"{key} must be HxW or HxWxC, got {tuple(out.shape)}")
    if out.shape[-1] == 1:
        return out
    if out.shape[-1] == 3:
        return out
    return out.mean(dim=-1, keepdim=True)


def _rgb(value: Tensor) -> Tensor:
    image = _to_hwc(value, "image").clamp(0.0, 1.0)
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    return image


def _save_png(path: Path, value: Tensor) -> Tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (_rgb(value).detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return int(arr.shape[1]), int(arr.shape[0])


def _save_mapped_png(path: Path, value: Tensor, *, lo: float, hi: float) -> Tuple[int, int]:
    mapped = ((value.detach().float() - float(lo)) / max(float(hi) - float(lo), 1e-12)).clamp(0.0, 1.0)
    return _save_png(path, mapped)


def _finite_flat(value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    data = value.detach().float()
    if mask is not None:
        mask = mask.detach().bool()
        while mask.ndim < data.ndim:
            mask = mask.expand(*data.shape[:2], data.shape[-1] if data.ndim == 3 else 1)
        data = data[mask]
    else:
        data = data.reshape(-1)
    data = data.reshape(-1).cpu()
    return data[torch.isfinite(data)]


def _safe_quantile(values: Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    flat = values.detach().float().reshape(-1)
    rank = max(1, min(int(flat.numel()), int(math.ceil(float(q) * float(flat.numel())))))
    return float(flat.kthvalue(rank).values.item())


def _channel_stats(value: Tensor, mask: Optional[Tensor], quantiles: Sequence[float]) -> Dict[str, Any]:
    image = _to_hwc(value, "stats")
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    out: Dict[str, Any] = {}
    for index, channel in enumerate(CHANNELS):
        flat = _finite_flat(image[..., index], mask[..., 0] if mask is not None else None)
        item = {
            "count": int(flat.numel()),
            "mean": float(flat.mean().item()) if flat.numel() else 0.0,
            "min": float(flat.min().item()) if flat.numel() else 0.0,
            "max": float(flat.max().item()) if flat.numel() else 0.0,
        }
        for q in quantiles:
            item[f"p{int(round(q * 100)):02d}"] = _safe_quantile(flat, q)
        out[channel] = item
    return out


def _scalar_stats(value: Tensor, mask: Optional[Tensor], quantiles: Sequence[float]) -> Dict[str, Any]:
    flat = _finite_flat(_to_hwc(value, "scalar")[..., 0], mask[..., 0] if mask is not None else None)
    item = {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
    }
    for q in quantiles:
        item[f"p{int(round(q * 100)):02d}"] = _safe_quantile(flat, q)
    return item


def _threshold_fractions(value: Tensor, mask: Optional[Tensor], thresholds: Sequence[float], op: str) -> Dict[str, Any]:
    image = _to_hwc(value, "threshold")
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    out: Dict[str, Any] = {}
    for index, channel in enumerate(CHANNELS):
        flat = _finite_flat(image[..., index], mask[..., 0] if mask is not None else None)
        item = {}
        for threshold in thresholds:
            if flat.numel() == 0:
                fraction = 0.0
            elif op == "lt":
                fraction = float((flat < float(threshold)).float().mean().item())
            elif op == "gt":
                fraction = float((flat > float(threshold)).float().mean().item())
            else:
                raise ValueError(op)
            item[f"P({op}{threshold:g})"] = fraction
        out[channel] = item
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.nonzero(sorted_values[1:] != sorted_values[:-1])[0] + 1]
    ends = np.r_[starts[1:], values.shape[0]]
    for start, end in zip(starts, ends):
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks


def _pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 1e-20:
        return 0.0
    return float((x * y).sum() / denom)


def _correlations(tau: Tensor, transmission: Tensor, clear: Tensor, mask: Tensor, sample_cap: int) -> Dict[str, Any]:
    tau_img = _to_hwc(tau, "tau")
    t_img = _to_hwc(transmission, "transmission")
    clear_img = _to_hwc(clear, "clear")
    if tau_img.shape[-1] == 1:
        tau_img = tau_img.expand_as(clear_img)
    if t_img.shape[-1] == 1:
        t_img = t_img.expand_as(clear_img)
    mask2 = mask[..., 0].detach().bool().cpu().numpy()
    out: Dict[str, Any] = {}
    for index, channel in enumerate(CHANNELS):
        x_tau = tau_img[..., index].detach().float().cpu().numpy()[mask2]
        x_nlogt = (-torch.log(t_img[..., index].clamp_min(1e-8))).detach().float().cpu().numpy()[mask2]
        y = clear_img[..., index].detach().float().cpu().numpy()[mask2]
        finite = np.isfinite(x_tau) & np.isfinite(x_nlogt) & np.isfinite(y)
        x_tau = x_tau[finite]
        x_nlogt = x_nlogt[finite]
        y = y[finite]
        if sample_cap > 0 and y.size > sample_cap:
            indices = np.linspace(0, y.size - 1, num=sample_cap, dtype=np.int64)
            x_tau = x_tau[indices]
            x_nlogt = x_nlogt[indices]
            y = y[indices]
        out[channel] = {
            "sample_count": int(y.size),
            "pearson_tau_D_J": _pearson_np(x_tau, y),
            "pearson_neglogT_J": _pearson_np(x_nlogt, y),
            "spearman_tau_D_J": _pearson_np(_rankdata(x_tau), _rankdata(y)) if y.size >= 2 else 0.0,
            "spearman_neglogT_J": _pearson_np(_rankdata(x_nlogt), _rankdata(y)) if y.size >= 2 else 0.0,
        }
    return out


def _empty_correlations() -> Dict[str, Any]:
    return {
        channel: {
            "sample_count": 0,
            "pearson_tau_D_J": 0.0,
            "pearson_neglogT_J": 0.0,
            "spearman_tau_D_J": 0.0,
            "spearman_neglogT_J": 0.0,
        }
        for channel in CHANNELS
    }


def _image_name(pipeline: Any, image_idx: int) -> str:
    dataset = pipeline.datamanager.eval_dataset
    try:
        filenames = dataset._dataparser_outputs.image_filenames
        return Path(filenames[int(image_idx)]).name
    except Exception:
        return f"eval_{int(image_idx):04d}"


def _dataset_image_name(pipeline: Any, split: str, image_idx: int) -> str:
    if split == "train":
        dataset = pipeline.datamanager.train_dataset
        try:
            return Path(dataset.image_filenames[int(image_idx)]).name
        except Exception:
            return f"train_{int(image_idx):04d}"
    return _image_name(pipeline, image_idx)


def _camera_id(camera: Any, image_idx: int) -> int:
    if camera.metadata is not None and "cam_idx" in camera.metadata:
        value = camera.metadata["cam_idx"]
        if torch.is_tensor(value):
            return int(value.detach().cpu().reshape(-1)[0].item())
        return int(value)
    return int(image_idx)


def _camera_items(pipeline: Any, split: str, max_images: int, device: torch.device) -> Iterable[Tuple[int, Any, Dict[str, Any]]]:
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


def _load_region_mask(mask_dir: Optional[Path], camera_id: int, shape: Tuple[int, int], key: str) -> Optional[Tensor]:
    if mask_dir is None:
        return None
    path = mask_dir / f"view_{int(camera_id):04d}_regions.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or key not in payload:
        return None
    mask = payload[key].detach().float()
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.shape[:2] != shape:
        mask = F.interpolate(mask.permute(2, 0, 1)[None], size=shape, mode="nearest")[0].permute(1, 2, 0)
    return mask.clamp(0.0, 1.0)


def _tile(path: Path, label: str, width: int) -> Image.Image:
    with Image.open(path) as src:
        image = src.convert("RGB")
    if width > 0 and image.width > width:
        height = max(1, int(round(image.height * width / image.width)))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    pad = 22
    canvas = Image.new("RGB", (image.width, image.height + pad), "white")
    canvas.paste(image, (0, pad))
    ImageDraw.Draw(canvas).text((4, 4), label, fill="black")
    return canvas


def _sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Path]]], tile_width: int) -> Tuple[int, int]:
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_tile(tile_path, label, tile_width) for label, tile_path in row]
        width = sum(tile.width for tile in tiles)
        height = max(tile.height for tile in tiles)
        row_img = Image.new("RGB", (width, height), "white")
        x = 0
        for tile in tiles:
            row_img.paste(tile, (x, 0))
            x += tile.width
        rendered_rows.append(row_img)
    sheet_width = max(row.width for row in rendered_rows)
    sheet_height = sum(row.height for row in rendered_rows)
    out = Image.new("RGB", (sheet_width, sheet_height), "white")
    y = 0
    for row in rendered_rows:
        out.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return sheet_width, sheet_height


def _manifest_row(
    scene: str,
    view_id: str,
    camera_id: str,
    image_type: str,
    component: str,
    path: Path,
    width: int,
    height: int,
    checkpoint: str,
    step: int,
    value_range: str,
    normalization: str,
    source_tensor: str,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "scene": scene,
        "view_id": view_id,
        "camera_id": camera_id,
        "image_type": image_type,
        "component": component,
        "file_path": str(path),
        "width": int(width),
        "height": int(height),
        "checkpoint_step": int(step),
        "source_checkpoint": checkpoint,
        "value_range": value_range,
        "normalization": normalization,
        "tone_mapping": NO_TONE,
        "color_space": COLOR_SPACE,
        "source_tensor": source_tensor,
        "notes": notes,
    }


def _range_text(value: Tensor) -> str:
    flat = _finite_flat(value)
    if flat.numel() == 0:
        return "unavailable"
    return f"[{float(flat.min().item()):.8g},{float(flat.max().item()):.8g}]"


def _masked_rgb_mean(value: Tensor, mask: Tensor) -> Dict[str, float]:
    image = _to_hwc(value, "masked_rgb_mean")
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    mask2 = mask[..., 0].detach().bool()
    out: Dict[str, float] = {}
    for index, channel in enumerate(CHANNELS):
        flat = image[..., index][mask2].detach().float().reshape(-1)
        flat = flat[torch.isfinite(flat)]
        out[channel] = float(flat.mean().item()) if flat.numel() else 0.0
    return out


def _background_mask_stats(
    *,
    mask: Optional[Tensor],
    medium_rgb: Tensor,
    medium_bs: Tensor,
    beta_raw: Tensor,
    beta_effective: Tensor,
    gt: Tensor,
) -> Dict[str, Any]:
    if mask is None:
        return {"available": False}
    mask = _to_hwc(mask, "background_mask").detach().float().clamp(0.0, 1.0)
    if mask.shape[-1] != 1:
        mask = mask[..., :1]
    denom = mask.sum().clamp_min(1e-6)
    residual = torch.abs(_to_hwc(medium_rgb, "medium_rgb") - _to_hwc(gt, "gt"))
    weight = 1.0 / (_to_hwc(medium_rgb, "medium_rgb").detach() + 1e-3)
    return {
        "available": True,
        "coverage": float(mask.mean().item()),
        "pixel_count": int((mask > 0.5).sum().item()),
        "background_medium_l1": float((mask * residual).sum().item() / denom.item()),
        "weighted_background_medium_l1": float((mask * weight * residual).sum().item() / denom.item()),
        "medium_rgb_mean": _masked_rgb_mean(medium_rgb, mask),
        "medium_bs_mean": _masked_rgb_mean(medium_bs, mask),
        "medium_attn_raw_mean": _masked_rgb_mean(beta_raw, mask),
        "medium_attn_effective_mean": _masked_rgb_mean(beta_effective, mask),
    }


def _aggregate_background_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    bg_rows = [row.get("background_mask_stats", {}) for row in rows if row.get("background_mask_stats", {}).get("available")]
    if not bg_rows:
        return {"available": False}
    weights = [max(int(row.get("pixel_count", 0)), 0) for row in bg_rows]
    denom = max(sum(weights), 1)

    def weighted_mean(key: str) -> float:
        return float(sum(float(row.get(key, 0.0)) * weight for row, weight in zip(bg_rows, weights)) / denom)

    coverage_values = [float(row.get("coverage", 0.0)) for row in bg_rows]
    out: Dict[str, Any] = {
        "available": True,
        "coverage": {
            "mean": float(np.mean(coverage_values)),
            "p10": float(np.quantile(coverage_values, 0.10)),
            "p50": float(np.quantile(coverage_values, 0.50)),
            "p90": float(np.quantile(coverage_values, 0.90)),
        },
        "background_medium_l1": weighted_mean("background_medium_l1"),
        "weighted_background_medium_l1": weighted_mean("weighted_background_medium_l1"),
    }
    for key in ("medium_rgb_mean", "medium_bs_mean", "medium_attn_raw_mean", "medium_attn_effective_mean"):
        out[key] = {
            channel: float(
                sum(float(row.get(key, {}).get(channel, 0.0)) * weight for row, weight in zip(bg_rows, weights)) / denom
            )
            for channel in CHANNELS
        }
    return out


def _aggregate_channel_stats(items: Sequence[Dict[str, Any]], key: str, channel: str, stat: str) -> float:
    weighted = [(float(item[key][channel][stat]), int(item[key][channel]["count"])) for item in items]
    denom = sum(count for _, count in weighted)
    if denom <= 0:
        return 0.0
    return float(sum(value * count for value, count in weighted) / denom)


def _concat_channel_values(per_view: Sequence[Mapping[str, Tensor]], key: str, channel_index: int) -> Tensor:
    values = []
    for item in per_view:
        mask = item["object_support_mask"][..., 0].bool()
        tensor = _to_hwc(item[key], key)
        if tensor.shape[-1] == 1:
            tensor = tensor.expand(-1, -1, 3)
        values.append(tensor[..., channel_index][mask].detach().float().cpu())
    return torch.cat(values) if values else torch.empty(0)


def _aggregate_stats(rendered: Sequence[Mapping[str, Tensor]], per_view_stats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "support_mask_definition": "renderer object support: outputs['accumulation'] > 0.01",
        "view_count": len(per_view_stats),
        "support_coverage": {
            "mean": float(np.mean([item["support_coverage"] for item in per_view_stats])) if per_view_stats else 0.0,
            "p10": float(np.quantile([item["support_coverage"] for item in per_view_stats], 0.10)) if per_view_stats else 0.0,
            "p50": float(np.quantile([item["support_coverage"] for item in per_view_stats], 0.50)) if per_view_stats else 0.0,
            "p90": float(np.quantile([item["support_coverage"] for item in per_view_stats], 0.90)) if per_view_stats else 0.0,
        },
    }
    tensors = {
        "beta_D_raw": ("beta_D_raw", STAT_Q),
        "beta_D_effective": ("beta_D_effective", STAT_Q),
        "depth": ("depth", STAT_Q),
        "tau_D_raw": ("tau_D_raw", STAT_Q),
        "tau_D_effective": ("tau_D_effective", STAT_Q),
        "T_D_effective": ("T_D_effective", T_Q),
        "clear_object_fullsh_raw": ("clear_object_fullsh_raw", (0.90, 0.95, 0.99)),
    }
    for out_key, (tensor_key, quantiles) in tensors.items():
        if out_key == "depth":
            all_values = []
            for item in rendered:
                mask = item["object_support_mask"][..., 0].bool()
                all_values.append(_to_hwc(item[tensor_key], tensor_key)[..., 0][mask].detach().float().cpu())
            flat = torch.cat(all_values) if all_values else torch.empty(0)
            stats = {
                "count": int(flat.numel()),
                "mean": float(flat.mean().item()) if flat.numel() else 0.0,
                "min": float(flat.min().item()) if flat.numel() else 0.0,
                "max": float(flat.max().item()) if flat.numel() else 0.0,
            }
            for q in quantiles:
                stats[f"p{int(round(q * 100)):02d}"] = _safe_quantile(flat, q)
            out[out_key] = stats
            continue
        out[out_key] = {}
        for index, channel in enumerate(CHANNELS):
            flat = _concat_channel_values(rendered, tensor_key, index)
            stats = {
                "count": int(flat.numel()),
                "mean": float(flat.mean().item()) if flat.numel() else 0.0,
                "min": float(flat.min().item()) if flat.numel() else 0.0,
                "max": float(flat.max().item()) if flat.numel() else 0.0,
            }
            for q in quantiles:
                stats[f"p{int(round(q * 100)):02d}"] = _safe_quantile(flat, q)
            out[out_key][channel] = stats
    out["T_D_effective_thresholds"] = {}
    out["clear_object_fullsh_raw_thresholds"] = {}
    for index, channel in enumerate(CHANNELS):
        t_flat = _concat_channel_values(rendered, "T_D_effective", index)
        j_flat = _concat_channel_values(rendered, "clear_object_fullsh_raw", index)
        out["T_D_effective_thresholds"][channel] = {
            "P(T<0.3)": float((t_flat < 0.30).float().mean().item()) if t_flat.numel() else 0.0,
            "P(T<0.2)": float((t_flat < 0.20).float().mean().item()) if t_flat.numel() else 0.0,
            "P(T<0.1)": float((t_flat < 0.10).float().mean().item()) if t_flat.numel() else 0.0,
            "P(T<0.05)": float((t_flat < 0.05).float().mean().item()) if t_flat.numel() else 0.0,
        }
        out["clear_object_fullsh_raw_thresholds"][channel] = {
            "P(J<0)": float((j_flat < 0.0).float().mean().item()) if j_flat.numel() else 0.0,
            "P(J>1.0)": float((j_flat > 1.0).float().mean().item()) if j_flat.numel() else 0.0,
            "P(J>1.5)": float((j_flat > 1.5).float().mean().item()) if j_flat.numel() else 0.0,
            "P(J>2.0)": float((j_flat > 2.0).float().mean().item()) if j_flat.numel() else 0.0,
        }
    return out


def _write_csv(path: Path, per_view_rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]) -> None:
    fieldnames = [
        "scene",
        "view_id",
        "image_idx",
        "image_name",
        "camera_id",
        "support_coverage",
        "psnr",
        "ssim",
        "lpips",
    ]
    for prefix in ("beta_D_raw", "beta_D_effective", "tau_D_effective", "T_D_effective", "clear_object_fullsh_raw"):
        for channel in CHANNELS:
            fieldnames.extend([f"{prefix}_{channel}_mean", f"{prefix}_{channel}_p90", f"{prefix}_{channel}_p95", f"{prefix}_{channel}_p99"])
    for channel in CHANNELS:
        fieldnames.extend(
            [
                f"T_{channel}_lt_0p3",
                f"T_{channel}_lt_0p2",
                f"T_{channel}_lt_0p1",
                f"T_{channel}_lt_0p05",
                f"J_{channel}_lt_0",
                f"J_{channel}_gt_1",
                f"J_{channel}_gt_1p5",
                f"J_{channel}_gt_2",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_view_rows:
            flat = {
                "scene": row["scene"],
                "view_id": row["view_id"],
                "image_idx": row["image_idx"],
                "image_name": row["image_name"],
                "camera_id": row["camera_id"],
                "support_coverage": row["support_coverage"],
                "psnr": row["metrics"]["psnr"],
                "ssim": row["metrics"]["ssim"],
                "lpips": row["metrics"]["lpips"],
            }
            for prefix in ("beta_D_raw", "beta_D_effective", "tau_D_effective", "T_D_effective", "clear_object_fullsh_raw"):
                for channel in CHANNELS:
                    stats = row[prefix][channel]
                    for stat in ("mean", "p90", "p95", "p99"):
                        flat[f"{prefix}_{channel}_{stat}"] = stats.get(stat, "")
            for channel in CHANNELS:
                thresholds = row["T_D_effective_thresholds"][channel]
                j_thresholds = row["clear_object_fullsh_raw_thresholds"][channel]
                flat[f"T_{channel}_lt_0p3"] = thresholds["P(lt0.3)"]
                flat[f"T_{channel}_lt_0p2"] = thresholds["P(lt0.2)"]
                flat[f"T_{channel}_lt_0p1"] = thresholds["P(lt0.1)"]
                flat[f"T_{channel}_lt_0p05"] = thresholds["P(lt0.05)"]
                flat[f"J_{channel}_lt_0"] = j_thresholds["P(J<0)"]
                flat[f"J_{channel}_gt_1"] = j_thresholds["P(gt1)"]
                flat[f"J_{channel}_gt_1p5"] = j_thresholds["P(gt1.5)"]
                flat[f"J_{channel}_gt_2"] = j_thresholds["P(gt2)"]
            writer.writerow(flat)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo = _repo_root()
    config_path = args.load_config
    if config_path is None:
        if args.scene not in SCENE_CONFIGS:
            raise ValueError(f"Unknown scene {args.scene}; pass --load-config explicitly")
        config_path = repo / SCENE_CONFIGS[args.scene]
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config, pipeline, checkpoint_path, loaded_step = _setup_pipeline(config_path, args.load_step, args.test_mode)
    model = pipeline.model
    model.eval()
    gamma = float(getattr(model.config, "direct_optical_depth_scale", 1.0))
    active_sh_degree = int(model._get_active_sh_degree()) if hasattr(model, "_get_active_sh_degree") else None
    configured_sh_degree = int(getattr(model.config, "sh_degree", -1))

    rows: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    rendered_for_aggregate: List[Dict[str, Tensor]] = []
    view_image_paths: Dict[int, Dict[str, Path]] = {}

    with torch.no_grad():
        for view_id, (image_idx, camera, batch) in enumerate(
            _camera_items(pipeline, args.split, int(args.max_images), model.device)
        ):
            if args.split == "eval":
                outputs = model.get_outputs_for_camera(camera=camera)
            else:
                outputs = model.get_outputs(camera)
            if args.stats_only:
                gt_for_stats = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
                metrics = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
                images = {"gt": gt_for_stats}
            else:
                metrics, images = model.get_image_metrics_and_images(outputs, batch)
            camera_id = _camera_id(camera, image_idx)
            image_name = _dataset_image_name(pipeline, args.split, image_idx)

            underwater = _to_hwc(outputs["pred_image"], "pred_image").detach().float().cpu()
            direct = _to_hwc(outputs["rgb_object"], "rgb_object").detach().float().cpu()
            clear = _to_hwc(outputs["J_gaussian_raw"], "J_gaussian_raw").detach().float().cpu()
            clear_clamp = clear.clamp(0.0, 1.0)
            clear_ws = clear / (clear + 1.0).clamp_min(1e-6)
            gt = _to_hwc(images["gt"], "gt").detach().float().cpu()
            depth = _to_hwc(outputs["depth"], "depth").detach().float().cpu()
            accumulation = _to_hwc(outputs["accumulation"], "accumulation").detach().float().cpu()
            mask = (accumulation > float(args.object_support_accumulation_threshold)).float()
            beta_effective = _to_hwc(outputs["medium_attn"], "medium_attn").detach().float().cpu()
            beta_raw = _to_hwc(outputs.get("medium_attn_raw", outputs["medium_attn"]), "medium_attn_raw").detach().float().cpu()
            tau_raw = beta_raw * depth
            tau_effective = beta_effective * depth
            transmission = torch.exp(-tau_effective.clamp_min(0.0))
            medium_rgb = _to_hwc(outputs["medium_rgb"], "medium_rgb").detach().float().cpu()
            b_inf = _to_hwc(outputs.get("b_inf", outputs["medium_rgb"]), "b_inf").detach().float().cpu()
            medium_bs = _to_hwc(outputs["medium_bs"], "medium_bs").detach().float().cpu()
            backscatter = b_inf * (1.0 - torch.exp(-(medium_bs * depth).clamp_min(0.0)))
            bg_mask = _load_region_mask(args.background_mask_dir, camera_id, tuple(underwater.shape[:2]), args.background_mask_key)
            bg_stats = _background_mask_stats(
                mask=bg_mask,
                medium_rgb=medium_rgb,
                medium_bs=medium_bs,
                beta_raw=beta_raw,
                beta_effective=beta_effective,
                gt=gt,
            )
            tau_vis = tau_effective.mean(dim=-1, keepdim=True)
            paths: Dict[str, Path] = {}
            view_dir = args.output_dir / args.scene / "per_view" / f"view_{view_id:04d}"
            alpha_dir = args.output_dir / args.scene / "alpha_sweep" / f"view_{view_id:04d}"

            image_specs = [
                ("gt_underwater", "gt_underwater", gt, "images['gt']", RGB_RANGE, "clamp_to_[0,1]_then_uint8"),
                ("underwater_rgb", "underwater_rgb", underwater, "outputs['pred_image']", RGB_RANGE, "clamp_to_[0,1]_then_uint8"),
                ("direct_object_signal", "direct_object_signal", direct, "outputs['rgb_object']", _range_text(direct), "clamp_to_[0,1]_then_uint8"),
                (
                    "clear_object_fullsh_raw_display",
                    "clear_object_fullsh_raw",
                    clear,
                    "outputs['J_gaussian_raw']",
                    _range_text(clear),
                    "raw display PNG uses clamp_to_[0,1]_then_uint8; raw stats are in summary.json",
                ),
                (
                    "clear_object_fullsh_clamp01",
                    "clear_object_fullsh_clamp01",
                    clear_clamp,
                    "clamp(outputs['J_gaussian_raw'],0,1)",
                    RGB_RANGE,
                    "clamp_to_[0,1]_then_uint8",
                ),
                (
                    "clear_object_fullsh_ws_tonemap",
                    "clear_object_fullsh_ws_tonemap",
                    clear_ws,
                    "J_gaussian_raw / (J_gaussian_raw + 1)",
                    _range_text(clear_ws),
                    "watersplatting_tonemap_then_clamp_to_[0,1]_then_uint8",
                ),
                ("transmission", "transmission", transmission, "exp(-medium_attn_effective * depth)", _range_text(transmission), "clamp_to_[0,1]_then_uint8"),
                (
                    "tau_D_visualization",
                    "tau_D",
                    (tau_vis / float(args.tau_display_max)).clamp(0.0, 1.0),
                    "mean(tau_D_effective RGB) / tau_display_max",
                    _range_text(tau_effective),
                    f"fixed_linear_mapping_[0,{args.tau_display_max}]_to_[0,1]",
                ),
                ("medium_rgb", "medium_rgb", medium_rgb, "outputs['medium_rgb']", _range_text(medium_rgb), "clamp_to_[0,1]_then_uint8"),
                ("backscatter", "backscatter", backscatter, "b_inf * (1 - exp(-medium_bs * depth))", _range_text(backscatter), "clamp_to_[0,1]_then_uint8"),
            ]
            if "J_proxy_raw" in outputs:
                proxy = _to_hwc(outputs["J_proxy_raw"], "J_proxy_raw").detach().float().cpu()
                image_specs.append(
                    (
                        "gmvc_J_proxy_raw",
                        "gmvc_J_proxy_raw",
                        proxy,
                        "outputs['J_proxy_raw']",
                        _range_text(proxy),
                        "clamp_to_[0,1]_then_uint8",
                    )
                )

            if not args.stats_only:
                for filename, component, tensor, source, value_range, normalization in image_specs:
                    path = view_dir / f"{filename}.png"
                    width, height = _save_png(path, tensor)
                    paths[filename] = path
                    manifest.append(
                        _manifest_row(
                            args.scene,
                            str(view_id),
                            str(camera_id),
                            "dewater_optical_depth_audit",
                            component,
                            path,
                            width,
                            height,
                            str(checkpoint_path),
                            int(loaded_step),
                            value_range,
                            normalization,
                            source,
                            "gmvc_J_proxy_raw is recorded only if produced by the checkpoint without changing config"
                            if component == "gmvc_J_proxy_raw"
                            else "",
                        )
                    )
                if "J_proxy_raw" not in outputs:
                    paths["gmvc_J_proxy_raw"] = Path("unavailable")

                for alpha in ALPHAS:
                    alpha_tensor = clear * torch.pow(transmission.clamp_min(0.0), float(alpha))
                    label = f"alpha_{alpha:.2f}".replace(".", "p")
                    path = alpha_dir / f"partial_deattenuation_{label}.png"
                    width, height = _save_png(path, alpha_tensor)
                    paths[label] = path
                    manifest.append(
                        _manifest_row(
                            args.scene,
                            str(view_id),
                            str(camera_id),
                            "partial_deattenuation",
                            label,
                            path,
                            width,
                            height,
                            str(checkpoint_path),
                            int(loaded_step),
                            _range_text(alpha_tensor),
                            "clamp_to_[0,1]_then_uint8",
                            f"J_gaussian_raw * T_D_effective^{alpha:.2f}",
                        )
                    )
            elif "J_proxy_raw" not in outputs:
                paths["gmvc_J_proxy_raw"] = Path("unavailable")

            view_stats = {
                "scene": args.scene,
                "view_id": int(view_id),
                "image_idx": int(image_idx),
                "image_name": image_name,
                "camera_id": int(camera_id),
                "width": int(underwater.shape[1]),
                "height": int(underwater.shape[0]),
                "metrics": {key: float(metrics.get(key, 0.0)) for key in ("psnr", "ssim", "lpips")},
                "support_coverage": float(mask.mean().item()),
                "beta_D_raw": _channel_stats(beta_raw, mask, STAT_Q),
                "beta_D_effective": _channel_stats(beta_effective, mask, STAT_Q),
                "depth": _scalar_stats(depth, mask, STAT_Q),
                "tau_D_raw": _channel_stats(tau_raw, mask, STAT_Q),
                "tau_D_effective": _channel_stats(tau_effective, mask, STAT_Q),
                "T_D_effective": _channel_stats(transmission, mask, T_Q),
                "T_D_effective_thresholds": _threshold_fractions(transmission, mask, (0.30, 0.20, 0.10, 0.05), "lt"),
                "clear_object_fullsh_raw": _channel_stats(clear, mask, (0.90, 0.95, 0.99)),
                "clear_object_fullsh_raw_thresholds": {
                    channel: {
                        "P(J<0)": _threshold_fractions(clear, mask, (0.0,), "lt")[channel]["P(lt0)"],
                        **_threshold_fractions(clear, mask, (1.0, 1.5, 2.0), "gt")[channel],
                    }
                    for channel in CHANNELS
                },
                "correlations": _empty_correlations()
                if args.stats_only
                else _correlations(tau_effective, transmission, clear, mask, args.correlation_sample_cap),
                "background_mask_stats": bg_stats,
                "files": {key: str(value) for key, value in paths.items() if str(value) != "unavailable"},
                "unavailable": {"gmvc_J_proxy_raw": "not present in outputs"} if "J_proxy_raw" not in outputs else {},
            }
            rows.append(view_stats)
            rendered_for_aggregate.append(
                {
                    "object_support_mask": mask,
                    "beta_D_raw": beta_raw,
                    "beta_D_effective": beta_effective,
                    "depth": depth,
                    "tau_D_raw": tau_raw,
                    "tau_D_effective": tau_effective,
                    "T_D_effective": transmission,
                    "clear_object_fullsh_raw": clear,
                }
            )
            view_image_paths[int(view_id)] = paths

    representative_ids = _representative_ids(sorted(view_image_paths))
    contact_rows = (
        []
        if args.stats_only
        else _write_contact_sheets(args, view_image_paths, representative_ids, sorted(view_image_paths), checkpoint_path, loaded_step)
    )
    manifest.extend(contact_rows)
    aggregate = _aggregate_stats(rendered_for_aggregate, rows)
    aggregate["correlations"] = _aggregate_correlations(rows)
    summary = {
        "diagnostic": "dewater_optical_depth_audit",
        "scene": args.scene,
        "load_config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "requested_step": int(args.load_step),
        "loaded_step": int(loaded_step),
        "split": args.split,
        "stats_only": bool(args.stats_only),
        "test_mode": args.test_mode,
        "direct_optical_depth_scale": gamma,
        "configured_sh_degree": configured_sh_degree,
        "active_sh_degree": active_sh_degree,
        "definitions": {
            "clear_object_fullsh_raw": (
                "outputs['J_gaussian_raw']; alpha-composited clear-object render from the current active full-SH Gaussian "
                "appearance branch, without medium direct attenuation or backscatter."
            ),
            "direct_object_signal": "outputs['rgb_object']; direct object contribution after medium attenuation in the underwater rasterizer.",
            "gmvc_J_proxy_raw": "outputs['J_proxy_raw'] if naturally present; this script does not enable GMVC/proxy context to create it.",
            "water_splatting_tonemap": "clear_object_fullsh_raw / (clear_object_fullsh_raw + 1)",
            "partial_deattenuation": "J_alpha = clear_object_fullsh_raw * T_D_effective^alpha",
        },
        "mask": {
            "name": "object_support",
            "definition": "outputs['accumulation'] > object_support_accumulation_threshold",
            "object_support_accumulation_threshold": float(args.object_support_accumulation_threshold),
            "source": "renderer accumulation output",
        },
        "background_supervision_mask": {
            "mask_dir": str(args.background_mask_dir) if args.background_mask_dir is not None else None,
            "mask_key": args.background_mask_key,
            "definition": (
                "optional fixed detached background-water mask loaded from view_<camera_id>_regions.pt; "
                "background residuals use sum(M * abs(medium_rgb - GT)) / sum(M) and "
                "sum(M * abs(medium_rgb - GT) / (medium_rgb.detach() + 1e-3)) / sum(M)"
            ),
        },
        "aggregate": aggregate,
        "background_supervision": _aggregate_background_stats(rows),
        "per_view": rows,
        "manifest": manifest,
        "git_commit": _git_commit(repo),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    _write_csv(args.output_dir / "summary.csv", rows, aggregate)
    _write_manifest_csv(args.output_dir / "manifest.csv", manifest)

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _aggregate_correlations(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for channel in CHANNELS:
        out[channel] = {}
        total = sum(row["correlations"][channel]["sample_count"] for row in rows)
        for key in ("pearson_tau_D_J", "pearson_neglogT_J", "spearman_tau_D_J", "spearman_neglogT_J"):
            if total <= 0:
                out[channel][key] = 0.0
            else:
                out[channel][key] = float(
                    sum(row["correlations"][channel][key] * row["correlations"][channel]["sample_count"] for row in rows)
                    / total
                )
        out[channel]["sample_count"] = int(total)
    return out


def _representative_ids(view_ids: Sequence[int]) -> List[int]:
    ids = list(view_ids)
    if len(ids) <= 3:
        return ids
    return [ids[0], ids[len(ids) // 2], ids[-1]]


def _write_contact_sheets(
    args: argparse.Namespace,
    paths: Mapping[int, Mapping[str, Path]],
    representative_ids: Sequence[int],
    all_ids: Sequence[int],
    checkpoint_path: Path,
    loaded_step: int,
) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []

    def add(name: str, subset: Sequence[int], rows: Sequence[Sequence[Tuple[str, Path]]]) -> None:
        path = args.output_dir / args.scene / "contact_sheets" / f"{name}.png"
        width, height = _sheet(path, rows, args.contact_tile_width)
        manifest.append(
            _manifest_row(
                args.scene,
                "multiple",
                "multiple",
                "contact_sheet",
                name,
                path,
                width,
                height,
                str(checkpoint_path),
                int(loaded_step),
                RGB_RANGE,
                "tiles reuse corresponding per-view PNG normalization",
                f"views={list(subset)}",
                "labels identify tensors only; no visual-quality annotation",
            )
        )

    for prefix, subset in (("all", all_ids), ("representative", representative_ids)):
        add(
            f"{prefix}_dewater_audit",
            subset,
            [
                [
                    (f"view {view_id} underwater", paths[view_id]["underwater_rgb"]),
                    (f"view {view_id} direct", paths[view_id]["direct_object_signal"]),
                    (f"view {view_id} clear raw", paths[view_id]["clear_object_fullsh_raw_display"]),
                    (f"view {view_id} clear clamp", paths[view_id]["clear_object_fullsh_clamp01"]),
                    (f"view {view_id} WS tonemap", paths[view_id]["clear_object_fullsh_ws_tonemap"]),
                ]
                for view_id in subset
            ],
        )
        add(
            f"{prefix}_medium_optical_depth",
            subset,
            [
                [
                    (f"view {view_id} transmission", paths[view_id]["transmission"]),
                    (f"view {view_id} tau_D", paths[view_id]["tau_D_visualization"]),
                    (f"view {view_id} medium_rgb", paths[view_id]["medium_rgb"]),
                    (f"view {view_id} backscatter", paths[view_id]["backscatter"]),
                ]
                for view_id in subset
            ],
        )
        add(
            f"{prefix}_partial_deattenuation_alpha_sweep",
            subset,
            [
                [
                    (f"view {view_id} alpha 0", paths[view_id]["alpha_0p00"]),
                    (f"view {view_id} alpha .25", paths[view_id]["alpha_0p25"]),
                    (f"view {view_id} alpha .50", paths[view_id]["alpha_0p50"]),
                    (f"view {view_id} alpha .75", paths[view_id]["alpha_0p75"]),
                    (f"view {view_id} alpha 1", paths[view_id]["alpha_1p00"]),
                    (f"view {view_id} direct", paths[view_id]["direct_object_signal"]),
                ]
                for view_id in subset
            ],
        )
    return manifest


def _write_manifest_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "scene",
        "view_id",
        "camera_id",
        "image_type",
        "component",
        "file_path",
        "width",
        "height",
        "checkpoint_step",
        "source_checkpoint",
        "value_range",
        "normalization",
        "tone_mapping",
        "color_space",
        "source_tensor",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Curasao")
    parser.add_argument("--load-config", type=Path, default=None)
    parser.add_argument("--load-step", type=int, default=15000)
    parser.add_argument("--split", choices=("eval", "train"), default="eval")
    parser.add_argument("--test-mode", choices=("test", "val", "inference"), default="test")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--background-mask-dir", type=Path, default=None)
    parser.add_argument("--background-mask-key", default="water")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--object-support-accumulation-threshold", type=float, default=0.01)
    parser.add_argument("--tau-display-max", type=float, default=3.0)
    parser.add_argument("--correlation-sample-cap", type=int, default=300000)
    parser.add_argument("--contact-tile-width", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=Path("renders/dewater_optical_depth_20260807"))
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "csv": str(args.output_dir / "summary.csv"),
                "scene": summary["scene"],
                "split": summary["split"],
                "stats_only": summary["stats_only"],
                "views": len(summary["per_view"]),
                "checkpoint": summary["checkpoint"],
                "loaded_step": summary["loaded_step"],
                "direct_optical_depth_scale": summary["direct_optical_depth_scale"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
