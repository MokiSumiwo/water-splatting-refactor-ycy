#!/usr/bin/env python
"""Offline view-wise medium-parameter inversion diagnostics.

This script does not train. It loads a trained WaterSplatting checkpoint,
extracts per-view medium parameters, and evaluates simplified closed-form
dewatering variants D0-D16.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

from nerfstudio.utils.eval_utils import eval_setup


VARIANT_LABELS = {
    "D0_model": "D0 Model J",
    "D1_pixel": "D1 Pixel",
    "D2_A_mean": "D2 A Mean",
    "D3_bs_mean": "D3 BS Mean",
    "D4_attn_mean": "D4 Attn Mean",
    "D5_all_mean": "D5 All Mean",
    "D6_all_median": "D6 Median",
    "D7_open_mean": "D7 Open Mean",
    "D8_logbeta_mean": "D8 Log-Beta",
    "D9_smooth31": "D9 Smooth31",
    "D10_smooth61": "D10 Smooth61",
    "D11_bs_spectrum_mean": "D11 BS Spectrum",
    "D12_bs_strength_mean": "D12 BS Strength",
    "D13_bs_spectrum_shrink025": "D13 BS Spec 0.25",
    "D14_bs_spectrum_shrink050": "D14 BS Spec 0.50",
    "D15_bs_spectrum_shrink075": "D15 BS Spec 0.75",
    "D16_bs_full_shrink025": "D16 BS Full 0.25",
    "D16_bs_full_shrink050": "D16 BS Full 0.50",
    "D16_bs_full_shrink075": "D16 BS Full 0.75",
}


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _to_float_image(tensor: torch.Tensor) -> torch.Tensor:
    image = tensor.detach().float()
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = image.expand(*image.shape[:2], 3)
    return image


def _to_uint8(tensor: torch.Tensor, clamp: bool = True) -> np.ndarray:
    image = _to_float_image(tensor).detach().cpu()
    if clamp:
        image = image.clamp(0.0, 1.0)
    arr = image.numpy()
    return (arr * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def _save_png(path: Path, tensor: torch.Tensor, clamp: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(tensor, clamp=clamp)).save(path)


def _normalize_vis(tensor: torch.Tensor, mask: torch.Tensor | None = None, q_low: float = 0.01, q_high: float = 0.99) -> torch.Tensor:
    image = _to_float_image(tensor)
    values = image
    if mask is not None and mask.any():
        values = image[mask.squeeze(-1)]
    finite = values[torch.isfinite(values)]
    if finite.numel() < 10:
        return image.clamp(0.0, 1.0)
    lo = torch.quantile(finite, q_low)
    hi = torch.quantile(finite, q_high)
    if float((hi - lo).abs().item()) < 1e-8:
        return torch.zeros_like(image)
    return ((image - lo) / (hi - lo)).clamp(0.0, 1.0)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []
    names += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _make_sheet(
    path: Path,
    items: List[Tuple[str, torch.Tensor]],
    *,
    cols: int,
    thumb_w: int = 260,
    title: str = "",
) -> None:
    if not items:
        return
    font = _load_font(14)
    title_font = _load_font(18, bold=True)
    label_h = 24
    pad = 10
    gap = 8
    thumbs: List[Tuple[str, Image.Image]] = []
    for label, image in items:
        arr = _to_uint8(image)
        im = Image.fromarray(arr)
        h = max(1, round(im.height * (thumb_w / im.width)))
        im = im.resize((thumb_w, h), Image.Resampling.LANCZOS)
        im = ImageOps.expand(im, border=1, fill=(210, 210, 210))
        thumbs.append((label, im))
    rows = math.ceil(len(thumbs) / cols)
    max_h_by_row = []
    for row in range(rows):
        max_h_by_row.append(max(im.height for _, im in thumbs[row * cols : (row + 1) * cols]) + label_h)
    title_h = 30 if title else 0
    width = pad * 2 + cols * thumb_w + (cols - 1) * gap
    height = pad * 2 + title_h + sum(max_h_by_row) + (rows - 1) * gap
    sheet = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    y = pad
    if title:
        draw.text((pad, y), title, fill=(20, 20, 20), font=title_font)
        y += title_h
    for row in range(rows):
        x = pad
        row_items = thumbs[row * cols : (row + 1) * cols]
        for label, im in row_items:
            draw.text((x, y), label, fill=(20, 20, 20), font=font)
            sheet.paste(im, (x, y + label_h))
            x += thumb_w + gap
        y += max_h_by_row[row] + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _load_binary_png(path: Path, shape: Tuple[int, int], device: torch.device) -> torch.Tensor | None:
    if not path.exists():
        return None
    mask = Image.open(path).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.NEAREST)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy((arr > 0.5).astype(np.float32)).to(device=device)[..., None]


def _load_far_mask(mask_dir: Path | None, view_index: int, shape: Tuple[int, int], device: torch.device) -> torch.Tensor | None:
    if mask_dir is None:
        return None
    candidates = [
        mask_dir / f"view_{view_index:04d}_far.pt",
        mask_dir / f"view_{view_index:04d}_far.png",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu")
            mask = payload["mask"] if isinstance(payload, dict) and "mask" in payload else payload
            mask = mask.to(device=device)
            if mask.ndim == 2:
                mask = mask[..., None]
            if mask.shape[:2] != shape:
                raise ValueError(f"Far mask shape {tuple(mask.shape)} does not match {shape}: {path}")
            return mask.bool()
        loaded = _load_binary_png(path, shape, device)
        if loaded is not None:
            return loaded.bool()
    return None


def _load_region_masks(mask_dir: Path | None, view_index: int, shape: Tuple[int, int], device: torch.device) -> Dict[str, torch.Tensor | None]:
    masks: Dict[str, torch.Tensor | None] = {"water": None, "object": None, "boundary": None}
    if mask_dir is None:
        return masks
    for key in masks:
        masks[key] = _load_binary_png(mask_dir / f"view_{view_index:04d}_{key}.png", shape, device)
    return masks


def _gradient_l1(image: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros_like(image[..., :1])
    grad[:, 1:, :] += (image[:, 1:, :] - image[:, :-1, :]).abs().mean(dim=-1, keepdim=True)
    grad[1:, :, :] += (image[1:, :, :] - image[:-1, :, :]).abs().mean(dim=-1, keepdim=True)
    return grad


def _masked_values(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        value = value[..., None]
    return value[mask.squeeze(-1)]


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = _masked_values(value, mask)
    if vals.numel() == 0:
        return torch.zeros((value.shape[-1],), device=value.device, dtype=value.dtype)
    if vals.ndim == 1:
        return vals.mean()[None]
    return vals.mean(dim=0)


def _masked_median(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = _masked_values(value, mask)
    if vals.numel() == 0:
        return torch.zeros((value.shape[-1],), device=value.device, dtype=value.dtype)
    if vals.ndim == 1:
        return vals.median()[None]
    return vals.median(dim=0).values


def _masked_geomean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vals = _masked_values(value.clamp_min(eps), mask)
    if vals.numel() == 0:
        return torch.zeros((value.shape[-1],), device=value.device, dtype=value.dtype)
    if vals.ndim == 1:
        return vals.log().mean().exp()[None]
    return vals.log().mean(dim=0).exp()


def _expand_param(param: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    if param.ndim == 1:
        return param.view(1, 1, -1).expand(shape[0], shape[1], -1)
    return param


def _smooth(image: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return image
    pad = kernel // 2
    x = image.permute(2, 0, 1).unsqueeze(0)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = F.avg_pool2d(x, kernel_size=kernel, stride=1)
    return x.squeeze(0).permute(1, 2, 0)


def _channel_normalize(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize an RGB tensor into per-pixel channel proportions."""
    return value.clamp_min(0.0) / value.clamp_min(0.0).sum(dim=-1, keepdim=True).clamp_min(eps)


def _invert(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    A: torch.Tensor,
    beta_bs: torch.Tensor,
    beta_attn: torch.Tensor,
    transmission_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depth3 = depth.expand_as(rgb)
    t_bs = torch.exp(-(beta_bs.clamp_min(0.0) * depth3).clamp(max=80.0))
    t_attn = torch.exp(-(beta_attn.clamp_min(0.0) * depth3).clamp(max=80.0))
    j = (rgb - A * (1.0 - t_bs)) / t_attn.clamp_min(transmission_floor)
    return j, t_bs, t_attn


def _psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    diff = (a - b).float()
    if mask is not None and mask.any():
        diff = diff[mask.squeeze(-1)]
    mse = diff.pow(2).mean().clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).item())


def _mae(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    diff = (a - b).abs().float()
    if mask is not None and mask.any():
        diff = diff[mask.squeeze(-1)]
    return float(diff.mean().item()) if diff.numel() else 0.0


def _ssim_np(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    del mask  # SSIM is reported on the bounding image for stability.
    try:
        from skimage.metrics import structural_similarity

        arr_a = a.detach().cpu().float().clamp(0.0, 1.0).numpy()
        arr_b = b.detach().cpu().float().clamp(0.0, 1.0).numpy()
        return float(structural_similarity(arr_a, arr_b, channel_axis=-1, data_range=1.0))
    except Exception:
        return 0.0


def _corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    x = a.detach().float()
    y = b.detach().float()
    if x.ndim == 3 and x.shape[-1] == 3:
        x = x.mean(dim=-1, keepdim=True)
    if y.ndim == 3 and y.shape[-1] == 3:
        y = y.mean(dim=-1, keepdim=True)
    if mask is not None and mask.any():
        x = x[mask.squeeze(-1)]
        y = y[mask.squeeze(-1)]
    else:
        x = x.reshape(-1)
        y = y.reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    if int(finite.sum().item()) < 2:
        return 0.0
    x = x[finite] - x[finite].mean()
    y = y[finite] - y[finite].mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((x * y).sum().item() / denom.item())


def _ntv(value: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    image = value.detach().float()
    dx = (image[:, 1:, :] - image[:, :-1, :]).abs()
    dy = (image[1:, :, :] - image[:-1, :, :]).abs()
    if mask is not None and mask.any():
        mx = mask[:, 1:, :] & mask[:, :-1, :]
        my = mask[1:, :, :] & mask[:-1, :, :]
        dx_vals = dx[mx.squeeze(-1)]
        dy_vals = dy[my.squeeze(-1)]
        base = image[mask.squeeze(-1)].abs().mean().clamp_min(1e-8)
    else:
        dx_vals = dx.reshape(-1, image.shape[-1])
        dy_vals = dy.reshape(-1, image.shape[-1])
        base = image.abs().mean().clamp_min(1e-8)
    if dx_vals.numel() == 0 or dy_vals.numel() == 0:
        return 0.0
    return float(((dx_vals.mean() + dy_vals.mean()) / base).item())


def _channel_distribution(value: torch.Tensor, mask: torch.Tensor) -> List[Dict[str, float]]:
    vals = _masked_values(value, mask).detach().float()
    if vals.numel() == 0:
        return [{"mean": 0.0, "std": 0.0, "cv": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0} for _ in range(value.shape[-1])]
    if vals.ndim == 1:
        vals = vals[:, None]
    out: List[Dict[str, float]] = []
    for c in range(vals.shape[-1]):
        x = vals[:, c]
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            out.append({"mean": 0.0, "std": 0.0, "cv": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0})
            continue
        mean = x.mean()
        std = x.std(unbiased=False)
        out.append(
            {
                "mean": float(mean.item()),
                "std": float(std.item()),
                "cv": float((std / mean.abs().clamp_min(1e-8)).item()),
                "p05": float(torch.quantile(x, 0.05).item()),
                "p50": float(torch.quantile(x, 0.50).item()),
                "p95": float(torch.quantile(x, 0.95).item()),
                "min": float(x.min().item()),
                "max": float(x.max().item()),
            }
        )
    return out


def _bluegreen_score(image: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    vals = image[..., 1:2].add(image[..., 2:3]).mul(0.5).sub(image[..., 0:1])
    return float(vals[mask].mean().item())


def _chroma(image: torch.Tensor) -> torch.Tensor:
    return image - image.mean(dim=-1, keepdim=True)


def _variant_metrics(
    *,
    raw: torch.Tensor,
    d0: torch.Tensor,
    far_mask: torch.Tensor,
    near_mask: torch.Tensor,
    t_attn: torch.Tensor,
    transmission_floor: float,
) -> Dict[str, float]:
    clamped = raw.clamp(0.0, 1.0)
    far_bg = _bluegreen_score(clamped, far_mask)
    near_bg = _bluegreen_score(clamped, near_mask)
    near_mae = _mae(clamped, d0.clamp(0.0, 1.0), near_mask)
    near_chroma_shift = _mae(_chroma(clamped), _chroma(d0.clamp(0.0, 1.0)), near_mask)
    tfloor = (t_attn <= transmission_floor + 1e-6).float()
    if far_mask.any():
        tfloor_rate = float(tfloor[far_mask.expand_as(tfloor)].mean().item())
    else:
        tfloor_rate = float(tfloor.mean().item())
    return {
        "far_bg_score": far_bg,
        "near_bg_score": near_bg,
        "far_near_bg_gap": far_bg - near_bg,
        "abs_far_near_bg_gap": abs(far_bg - near_bg),
        "near_rgb_mae_vs_d0": near_mae,
        "near_chroma_shift_vs_d0": near_chroma_shift,
        "near_ssim_vs_d0": _ssim_np(clamped, d0.clamp(0.0, 1.0), near_mask),
        "raw_negative_rate": float((raw < 0.0).float().mean().item()),
        "raw_over_one_rate": float((raw > 1.0).float().mean().item()),
        "raw_clip_rate": float(((raw < 0.0) | (raw > 1.0)).float().mean().item()),
        "raw_negative_r": float((raw[..., 0] < 0.0).float().mean().item()),
        "raw_negative_g": float((raw[..., 1] < 0.0).float().mean().item()),
        "raw_negative_b": float((raw[..., 2] < 0.0).float().mean().item()),
        "raw_over_one_r": float((raw[..., 0] > 1.0).float().mean().item()),
        "raw_over_one_g": float((raw[..., 1] > 1.0).float().mean().item()),
        "raw_over_one_b": float((raw[..., 2] > 1.0).float().mean().item()),
        "transmission_floor_hit_rate": tfloor_rate,
    }


def _infer_open_mask(
    *,
    gt_or_rgb: torch.Tensor,
    valid_mask: torch.Tensor,
    far_mask: torch.Tensor,
    water_mask: torch.Tensor | None,
    minimum_pixels: int,
    min_coverage: float,
    lowgrad_fraction: float,
) -> Tuple[torch.Tensor, str, Dict[str, float]]:
    h, w = valid_mask.shape[:2]
    total = h * w
    if water_mask is not None:
        water = water_mask.bool() & valid_mask
        pixels = int(water.sum().item())
        coverage = pixels / max(1, total)
        if pixels >= minimum_pixels and coverage >= min_coverage:
            return water, "region_water", {"water_pixels": pixels, "water_coverage": coverage}

    grad = _gradient_l1(gt_or_rgb.detach().float())
    candidates = far_mask & valid_mask
    vals = grad[candidates.squeeze(-1)]
    if vals.numel() > 0:
        threshold = torch.quantile(vals, float(lowgrad_fraction))
        lowgrad = grad <= threshold
        open_mask = candidates & lowgrad
    else:
        open_mask = candidates
    pixels = int(open_mask.sum().item())
    return open_mask, "far_lowgrad_fallback", {"water_pixels": int(water_mask.sum().item()) if water_mask is not None else 0, "open_pixels": pixels}


def _ensure_mask(mask: torch.Tensor | None, fallback: torch.Tensor) -> torch.Tensor:
    if mask is not None and mask.any():
        return mask.bool()
    return fallback.bool()


def _process_view(
    *,
    outputs: Dict[str, torch.Tensor],
    gt_or_rgb: torch.Tensor,
    view_index: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rgb = outputs.get("rgb", outputs.get("pred_image")).detach().float().clamp(0.0, 1.0)
    j_model = outputs["J"].detach().float()
    j_model_raw = outputs.get("J_raw", outputs["J"]).detach().float()
    depth = outputs.get(args.depth_mode if args.depth_mode != "expected" else "depth", outputs["depth"]).detach().float()
    if depth.ndim == 2:
        depth = depth[..., None]
    accumulation = outputs["accumulation"].detach().float()
    if accumulation.ndim == 2:
        accumulation = accumulation[..., None]
    A = outputs["medium_rgb"].detach().float()
    beta_bs = outputs["medium_bs"].detach().float().clamp_min(0.0)
    beta_attn = outputs["medium_attn"].detach().float().clamp_min(0.0)
    h, w = rgb.shape[:2]
    device = rgb.device

    valid_mask = torch.isfinite(depth) & (depth > 0.0) & (accumulation > args.valid_accumulation_threshold)
    far_loaded = _load_far_mask(args.far_mask_dir, view_index, (h, w), device)
    if far_loaded is None:
        valid_depth = depth[valid_mask]
        if valid_depth.numel() > 0:
            cutoff = torch.quantile(valid_depth, 0.90)
            far_mask = valid_mask & (depth >= cutoff)
        else:
            far_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    else:
        far_mask = far_loaded.bool() & valid_mask
    depth_valid = depth[valid_mask]
    if depth_valid.numel() > 0:
        near_cutoff = torch.quantile(depth_valid, 0.50)
        near_mask = valid_mask & (depth <= near_cutoff)
    else:
        near_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

    region = _load_region_masks(args.region_mask_dir, view_index, (h, w), device)
    open_mask, open_source, open_stats = _infer_open_mask(
        gt_or_rgb=gt_or_rgb,
        valid_mask=valid_mask,
        far_mask=far_mask,
        water_mask=region["water"],
        minimum_pixels=args.minimum_mask_pixels,
        min_coverage=0.005,
        lowgrad_fraction=args.lowgrad_fraction,
    )
    if int(open_mask.sum().item()) < args.minimum_mask_pixels:
        open_mask = _ensure_mask(far_mask, valid_mask)
        open_source = f"{open_source}_insufficient_using_far"

    A_mean = _masked_mean(A, valid_mask)
    A_median = _masked_median(A, valid_mask)
    A_open = _masked_mean(A, open_mask)
    bs_mean = _masked_mean(beta_bs, valid_mask)
    bs_median = _masked_median(beta_bs, valid_mask)
    bs_open = _masked_mean(beta_bs, open_mask)
    bs_geo = _masked_geomean(beta_bs, valid_mask)
    attn_mean = _masked_mean(beta_attn, valid_mask)
    attn_median = _masked_median(beta_attn, valid_mask)
    attn_open = _masked_mean(beta_attn, open_mask)
    attn_geo = _masked_geomean(beta_attn, valid_mask)

    variants: Dict[str, Dict[str, torch.Tensor]] = {}
    variants["D0_model"] = {"raw": j_model_raw, "clamped": j_model.clamp(0.0, 1.0), "t_attn": torch.ones_like(rgb)}
    for key, Ap, Bp, Dp in [
        ("D1_pixel", A, beta_bs, beta_attn),
        ("D2_A_mean", _expand_param(A_mean, (h, w)), beta_bs, beta_attn),
        ("D3_bs_mean", A, _expand_param(bs_mean, (h, w)), beta_attn),
        ("D4_attn_mean", A, beta_bs, _expand_param(attn_mean, (h, w))),
        ("D5_all_mean", _expand_param(A_mean, (h, w)), _expand_param(bs_mean, (h, w)), _expand_param(attn_mean, (h, w))),
        ("D6_all_median", _expand_param(A_median, (h, w)), _expand_param(bs_median, (h, w)), _expand_param(attn_median, (h, w))),
        ("D7_open_mean", _expand_param(A_open, (h, w)), _expand_param(bs_open, (h, w)), _expand_param(attn_open, (h, w))),
        ("D8_logbeta_mean", _expand_param(A_mean, (h, w)), _expand_param(bs_geo, (h, w)), _expand_param(attn_geo, (h, w))),
    ]:
        raw, _tbs, tattn = _invert(rgb, depth, Ap, Bp, Dp, args.transmission_floor)
        variants[key] = {"raw": raw, "clamped": raw.clamp(0.0, 1.0), "t_attn": tattn}

    bs_strength = beta_bs.mean(dim=-1, keepdim=True)
    bs_spectrum = _channel_normalize(beta_bs)
    bs_spectrum_mean = _channel_normalize(_masked_mean(bs_spectrum, valid_mask).view(1, 1, 3)).expand(h, w, 3)
    bs_strength_mean = _masked_mean(bs_strength, valid_mask).view(1, 1, 1).expand(h, w, 1)

    bs_d11 = (3.0 * bs_strength * bs_spectrum_mean).clamp_min(0.0)
    raw, _tbs, tattn = _invert(rgb, depth, A, bs_d11, beta_attn, args.transmission_floor)
    variants["D11_bs_spectrum_mean"] = {"raw": raw, "clamped": raw.clamp(0.0, 1.0), "t_attn": tattn}

    bs_d12 = (3.0 * bs_strength_mean * bs_spectrum).clamp_min(0.0)
    raw, _tbs, tattn = _invert(rgb, depth, A, bs_d12, beta_attn, args.transmission_floor)
    variants["D12_bs_strength_mean"] = {"raw": raw, "clamped": raw.clamp(0.0, 1.0), "t_attn": tattn}

    for key, lam in [
        ("D13_bs_spectrum_shrink025", 0.25),
        ("D14_bs_spectrum_shrink050", 0.50),
        ("D15_bs_spectrum_shrink075", 0.75),
    ]:
        spectrum = _channel_normalize((1.0 - lam) * bs_spectrum + lam * bs_spectrum_mean)
        Bp = (3.0 * bs_strength * spectrum).clamp_min(0.0)
        raw, _tbs, tattn = _invert(rgb, depth, A, Bp, beta_attn, args.transmission_floor)
        variants[key] = {"raw": raw, "clamped": raw.clamp(0.0, 1.0), "t_attn": tattn}

    bs_mean_image = _expand_param(bs_mean, (h, w))
    for key, lam in [
        ("D16_bs_full_shrink025", 0.25),
        ("D16_bs_full_shrink050", 0.50),
        ("D16_bs_full_shrink075", 0.75),
    ]:
        Bp = ((1.0 - lam) * beta_bs + lam * bs_mean_image).clamp_min(0.0)
        raw, _tbs, tattn = _invert(rgb, depth, A, Bp, beta_attn, args.transmission_floor)
        variants[key] = {"raw": raw, "clamped": raw.clamp(0.0, 1.0), "t_attn": tattn}

    smooth_outputs: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for kernel in args.smooth_kernels:
        Ap = _smooth(A, kernel)
        Bp = _smooth(beta_bs, kernel).clamp_min(0.0)
        Dp = _smooth(beta_attn, kernel).clamp_min(0.0)
        raw, _tbs, tattn = _invert(rgb, depth, Ap, Bp, Dp, args.transmission_floor)
        smooth_outputs[kernel] = (raw, tattn, Ap)
    if 31 in smooth_outputs:
        raw, tattn, _Ap = smooth_outputs[31]
        variants["D9_smooth31"] = {"raw": raw, "clamped": raw.clamp(0.0, 1.0), "t_attn": tattn}
    if 61 in smooth_outputs:
        raw, tattn, _Ap = smooth_outputs[61]
        variants["D10_smooth61"] = {"raw": raw, "clamped": raw.clamp(0.0, 1.0), "t_attn": tattn}

    d1_raw, tbs_pixel, tattn_pixel = _invert(rgb, depth, A, beta_bs, beta_attn, args.transmission_floor)
    del d1_raw
    forward = j_model.clamp(0.0, 1.0) * tattn_pixel + A * (1.0 - tbs_pixel)
    forward_error = (forward - rgb).abs()
    forward_stats = {
        "mae": _mae(forward, rgb, valid_mask),
        "psnr": _psnr(forward, rgb, valid_mask),
        "ssim": _ssim_np(forward.clamp(0.0, 1.0), rgb.clamp(0.0, 1.0), valid_mask),
        "far_mae": _mae(forward, rgb, far_mask),
        "near_mae": _mae(forward, rgb, near_mask),
    }

    metrics: Dict[str, Any] = {
        "view_index": view_index,
        "shape": [h, w],
        "valid_pixels": int(valid_mask.sum().item()),
        "valid_coverage": float(valid_mask.float().mean().item()),
        "far_pixels": int(far_mask.sum().item()),
        "far_coverage": float(far_mask.float().mean().item()),
        "near_pixels": int(near_mask.sum().item()),
        "near_coverage": float(near_mask.float().mean().item()),
        "open_pixels": int(open_mask.sum().item()),
        "open_coverage": float(open_mask.float().mean().item()),
        "open_source": open_source,
        "open_stats": open_stats,
        "forward_recomposition": forward_stats,
        "parameter_stats": {},
        "variants": {},
    }

    for name, param in [("A", A), ("beta_bs", beta_bs), ("beta_attn", beta_attn)]:
        metrics["parameter_stats"][name] = {
            "valid_distribution": _channel_distribution(param, valid_mask),
            "far_distribution": _channel_distribution(param, far_mask),
            "near_distribution": _channel_distribution(param, near_mask),
            "open_distribution": _channel_distribution(param, open_mask),
            "ntv_valid": _ntv(param, valid_mask),
            "depth_corr_valid": _corr(param, depth, valid_mask),
            "far_minus_near_mean": [
                float(x.item()) for x in (_masked_mean(param, far_mask) - _masked_mean(param, near_mask)).reshape(-1)
            ],
        }

    for key, payload in variants.items():
        metrics["variants"][key] = _variant_metrics(
            raw=payload["raw"],
            d0=j_model,
            far_mask=far_mask,
            near_mask=near_mask,
            t_attn=payload["t_attn"],
            transmission_floor=args.transmission_floor,
        )

    view_dir = output_dir / f"view_{view_index:04d}"
    view_dir.mkdir(parents=True, exist_ok=True)
    if args.save_full_resolution:
        _save_png(view_dir / "rgb.png", rgb)
        _save_png(view_dir / "J_model.png", j_model)
        _save_png(view_dir / "J_model_raw_vis.png", j_model_raw.clamp(0.0, 1.0))
        _save_png(view_dir / "depth.png", _normalize_vis(depth, valid_mask))
        _save_png(view_dir / "accumulation.png", accumulation.clamp(0.0, 1.0))
        _save_png(view_dir / "medium_rgb.png", A)
        _save_png(view_dir / "medium_bs_channels.png", _normalize_vis(beta_bs, valid_mask))
        _save_png(view_dir / "medium_attn_channels.png", _normalize_vis(beta_attn, valid_mask))
        _save_png(view_dir / "transmission_bs.png", tbs_pixel.clamp(0.0, 1.0))
        _save_png(view_dir / "transmission_attn.png", tattn_pixel.clamp(0.0, 1.0))
        _save_png(view_dir / "forward_recomposition.png", forward.clamp(0.0, 1.0))
        _save_png(view_dir / "forward_error.png", (forward_error / 0.10).clamp(0.0, 1.0))
        _save_png(view_dir / "far_mask.png", far_mask.float())
        _save_png(view_dir / "open_mask.png", open_mask.float())
        for key, payload in variants.items():
            label = key.replace("_", "-")
            _save_png(view_dir / f"{label}_raw_vis.png", payload["raw"].clamp(0.0, 1.0))
            _save_png(view_dir / f"{label}_clamped.png", payload["clamped"])

    if args.save_raw_tensors:
        torch.save(
            {
                "rgb": rgb.cpu(),
                "j_model": j_model.cpu(),
                "j_model_raw": j_model_raw.cpu(),
                "depth": depth.cpu(),
                "accumulation": accumulation.cpu(),
                "A": A.cpu(),
                "beta_bs": beta_bs.cpu(),
                "beta_attn": beta_attn.cpu(),
                "valid_mask": valid_mask.cpu(),
                "far_mask": far_mask.cpu(),
                "near_mask": near_mask.cpu(),
                "open_mask": open_mask.cpu(),
                "variants_raw": {key: payload["raw"].cpu() for key, payload in variants.items()},
            },
            view_dir / "raw_outputs.pt",
        )

    if args.save_contact_sheet:
        _make_sheet(
            view_dir / "parameter_sheet.png",
            [
                ("RGB", rgb),
                ("Current J", j_model),
                ("Depth", _normalize_vis(depth, valid_mask)),
                ("Accum", accumulation),
                ("Medium RGB", A),
                ("Far/Open Mask", torch.maximum(far_mask.float() * 0.55, open_mask.float())),
                ("BS R", _normalize_vis(beta_bs[..., 0:1], valid_mask)),
                ("BS G", _normalize_vis(beta_bs[..., 1:2], valid_mask)),
                ("BS B", _normalize_vis(beta_bs[..., 2:3], valid_mask)),
                ("Attn R", _normalize_vis(beta_attn[..., 0:1], valid_mask)),
                ("Attn G", _normalize_vis(beta_attn[..., 1:2], valid_mask)),
                ("Attn B", _normalize_vis(beta_attn[..., 2:3], valid_mask)),
            ],
            cols=6,
            title=f"View {view_index:04d} Parameters",
        )
        inversion_items = [(VARIANT_LABELS[key], payload["clamped"]) for key, payload in variants.items()]
        far_blue = torch.zeros_like(rgb)
        far_blue[..., 0:1] = far_mask.float()
        far_blue[..., 1:2] = torch.relu(variants["D1_pixel"]["clamped"][..., 1:2] - variants["D5_all_mean"]["clamped"][..., 1:2])
        far_blue[..., 2:3] = torch.relu(variants["D1_pixel"]["clamped"][..., 2:3] - variants["D5_all_mean"]["clamped"][..., 2:3])
        inversion_items.append(("Far Blue Diff D1-D5", far_blue.clamp(0.0, 1.0)))
        _make_sheet(view_dir / "inversion_sheet.png", inversion_items, cols=6, title=f"View {view_index:04d} Inversions")

    if args.save_json:
        (view_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    return metrics


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {"num_views": len(results), "variants": {}, "forward_recomposition": {}, "parameter_stats": {}}
    if not results:
        return aggregate
    variant_keys = list(results[0]["variants"].keys())
    metric_keys = list(results[0]["variants"][variant_keys[0]].keys())
    for variant in variant_keys:
        aggregate["variants"][variant] = {}
        for key in metric_keys:
            vals = [float(item["variants"][variant][key]) for item in results]
            aggregate["variants"][variant][key] = float(np.mean(vals))
    for key in results[0]["forward_recomposition"]:
        aggregate["forward_recomposition"][key] = float(np.mean([item["forward_recomposition"][key] for item in results]))
    for param in ["A", "beta_bs", "beta_attn"]:
        aggregate["parameter_stats"][param] = {
            "ntv_valid": float(np.mean([item["parameter_stats"][param]["ntv_valid"] for item in results])),
            "depth_corr_valid": float(np.mean([item["parameter_stats"][param]["depth_corr_valid"] for item in results])),
            "far_minus_near_mean": [
                float(np.mean([item["parameter_stats"][param]["far_minus_near_mean"][c] for item in results]))
                for c in range(3)
            ],
        }
    for key in ["valid_coverage", "far_coverage", "near_coverage", "open_coverage"]:
        aggregate[key] = float(np.mean([item[key] for item in results]))
    return aggregate


def _write_summary_sheet(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    rows = []
    for view in results:
        view_idx = view["view_index"]
        view_dir = output_dir / f"view_{view_idx:04d}"
        sheet = view_dir / "inversion_sheet.png"
        if sheet.exists():
            rows.append((view_idx, sheet))
    if not rows:
        return
    images = []
    target_w = 1600
    for view_idx, path in rows:
        im = Image.open(path).convert("RGB")
        h = round(im.height * (target_w / im.width))
        im = im.resize((target_w, h), Image.Resampling.LANCZOS)
        images.append((view_idx, im))
    pad = 12
    label_h = 24
    font = _load_font(16, bold=True)
    width = target_w + 2 * pad
    height = pad + sum(im.height + label_h + pad for _, im in images)
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    y = pad
    for view_idx, im in images:
        draw.text((pad, y), f"view {view_idx:04d}", fill=(20, 20, 20), font=font)
        y += label_h
        canvas.paste(im, (pad, y))
        y += im.height + pad
    canvas.save(output_dir / "inversion_contact_sheet_all_views.png")


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        eval_num_rays_per_chunk=None,
        test_mode="inference",
        update_config_callback=_update_config,
    )
    del config
    pipeline.eval()
    device = pipeline.model.device

    if args.split != "eval":
        raise NotImplementedError("Only split=eval is implemented for this diagnostic.")

    results: List[Dict[str, Any]] = []
    max_count = args.max_images if args.max_images > 0 else 10**9
    with torch.no_grad():
        for view_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if view_index >= max_count:
                break
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            if "image" in batch:
                gt = pipeline.model.composite_with_background(
                    pipeline.model.get_gt_img(batch["image"].to(device) if hasattr(batch["image"], "to") else batch["image"]),
                    outputs["background"],
                )
            else:
                gt = outputs.get("rgb", outputs["pred_image"]).detach()
            metrics = _process_view(outputs=outputs, gt_or_rgb=gt, view_index=view_index, output_dir=args.output_dir, args=args)
            results.append(metrics)

    aggregate = _aggregate(results)
    result = {
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "split": args.split,
        "transmission_floor": args.transmission_floor,
        "smooth_kernels": args.smooth_kernels,
        "far_mask_dir": str(args.far_mask_dir) if args.far_mask_dir else "",
        "region_mask_dir": str(args.region_mask_dir) if args.region_mask_dir else "",
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "aggregate": aggregate,
        "views": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_json:
        (args.output_dir / "aggregate.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    if args.save_contact_sheet:
        _write_summary_sheet(args.output_dir, results)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--split", choices=["eval"], default="eval")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--far-mask-dir", type=Path, default=None)
    parser.add_argument("--region-mask-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transmission-floor", type=float, default=0.05)
    parser.add_argument("--smooth-kernels", type=int, nargs="*", default=[31, 61])
    parser.add_argument("--valid-accumulation-threshold", type=float, default=0.01)
    parser.add_argument("--minimum-mask-pixels", type=int, default=1000)
    parser.add_argument("--trim-ratio", type=float, default=0.10)
    parser.add_argument("--lowgrad-fraction", type=float, default=0.35)
    parser.add_argument("--depth-mode", choices=["expected", "first_depth", "last_depth"], default="expected")
    parser.add_argument("--save-raw-tensors", action="store_true")
    parser.add_argument("--save-full-resolution", action="store_true")
    parser.add_argument("--save-contact-sheet", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    result = diagnose(args)
    print(json.dumps({"step": result["step"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
