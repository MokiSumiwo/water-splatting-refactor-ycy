#!/usr/bin/env python
"""Frozen-checkpoint renderer color-bias audit.

This script is diagnostic-only.  It does not modify checkpoints or training
modules.  The goal is to separate far clear-color bias into measurable signals:
surface-vs-layer degradation mismatch, attenuation exposure collapse, tail
confusion, SH residuals, clamp amplification, and medium spectral ordering.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

from nerfstudio.utils.eval_utils import eval_setup


CHANNELS = ("r", "g", "b")
LUMA = (0.2126, 0.7152, 0.0722)


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor(LUMA)
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _chroma(rgb: torch.Tensor) -> torch.Tensor:
    return rgb - rgb.mean(dim=-1, keepdim=True)


def _medium_axis_projection(rgb: torch.Tensor, medium_rgb: torch.Tensor) -> torch.Tensor:
    rgb_chroma = _chroma(rgb.float())
    medium_chroma = _chroma(medium_rgb.float())
    medium_dir = medium_chroma / medium_chroma.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (rgb_chroma * medium_dir).sum(dim=-1, keepdim=True)


def _blue_green_minus_red(rgb: torch.Tensor) -> torch.Tensor:
    return 0.5 * (rgb[..., 1:2] + rgb[..., 2:3]) - rgb[..., 0:1]


def _stats(values: torch.Tensor) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1).cpu()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
        "min": float(flat.min().item()),
    }


def _mean(values: torch.Tensor) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    return float(flat.mean().item()) if flat.numel() else 0.0


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    keep = torch.isfinite(a) & torch.isfinite(b)
    a = a[keep]
    b = b[keep]
    if a.numel() < 4:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).mean() * (b * b).mean()).clamp_min(1e-12)
    return float(((a * b).mean() / denom).item())


def _masked(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[..., None]
    if image.ndim == 3 and image.shape[-1] != 1:
        return image[mask.squeeze(-1)]
    return image[mask.squeeze(-1)]


def _safe_mask(mask: Optional[torch.Tensor], like: torch.Tensor, default: bool = True) -> torch.Tensor:
    if mask is None:
        fill = 1 if default else 0
        return torch.full((*like.shape[:2], 1), fill, device=like.device, dtype=torch.bool)
    mask = mask.to(device=like.device, dtype=torch.bool)
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.shape[:2] != like.shape[:2]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match image shape {tuple(like.shape)}")
    return mask


def _load_far_mask(mask_dir: Optional[Path], image_idx: int, like: torch.Tensor) -> torch.Tensor:
    if mask_dir is None:
        depth = like
        valid = torch.isfinite(depth) & (depth > 0)
        if not valid.any():
            return torch.zeros_like(depth, dtype=torch.bool)
        cutoff = torch.quantile(depth[valid], 0.90)
        return valid & (depth >= cutoff)
    path = mask_dir / f"view_{image_idx:04d}_far.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing far mask: {path}")
    payload = torch.load(path, map_location="cpu")
    mask = payload["mask"] if isinstance(payload, Mapping) and "mask" in payload else payload
    return _safe_mask(mask, like)


def _load_region_masks(mask_dir: Optional[Path], image_idx: int, like: torch.Tensor) -> Dict[str, torch.Tensor]:
    if mask_dir is None:
        return {}
    path = mask_dir / f"view_{image_idx:04d}_regions.pt"
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu")
    out: Dict[str, torch.Tensor] = {}
    for key in ("water", "object", "boundary", "uncertain"):
        if key in payload:
            out[key] = _safe_mask(payload[key], like)
    return out


def _maybe_save_image(path: Path, image: torch.Tensor) -> None:
    try:
        from torchvision.utils import save_image
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = image.detach().float().cpu()
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = img.expand(*img.shape[:2], 3)
    save_image(img.clamp(0.0, 1.0).permute(2, 0, 1), path)


def _normalize_vis(image: torch.Tensor, *, clip: float = 1.0) -> torch.Tensor:
    image = image.detach().float()
    if clip > 0:
        return (image / clip).clamp(0.0, 1.0)
    finite = image[torch.isfinite(image)]
    if finite.numel() == 0:
        return torch.zeros_like(image)
    lo = torch.quantile(finite, 0.02)
    hi = torch.quantile(finite, 0.98)
    return ((image - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def _surface_rgb(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Approximate a surface-style degradation using exposed pixel summaries."""

    j_raw = outputs["J_gaussian_raw"].detach().float()
    alpha = outputs["accumulation"].detach().float().clamp(0.0, 1.0)
    depth = outputs["depth"].detach().float().clamp_min(0.0)
    medium_rgb = outputs.get("b_inf", outputs["medium_rgb"]).detach().float()
    medium_attn = outputs["medium_attn"].detach().float().clamp_min(0.0)
    medium_bs = outputs["medium_bs"].detach().float().clamp_min(0.0)
    trans_d = torch.exp(-(medium_attn * depth).clamp_min(0.0)).clamp(0.0, 1.0)
    trans_b = torch.exp(-(medium_bs * depth).clamp_min(0.0)).clamp(0.0, 1.0)
    surface_object = j_raw * trans_d + alpha * medium_rgb * (1.0 - trans_b)
    surface_tail = (1.0 - alpha) * medium_rgb
    return surface_object + surface_tail


def _surface_confidence(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    alpha = outputs["accumulation"].detach().float().clamp(0.0, 1.0)
    depth_std = outputs["depth_std_relative"].detach().float().clamp_min(0.0)
    final_t = outputs["final_transmittance"].detach().float().clamp(0.0, 1.0)
    q_acc = torch.sigmoid((alpha - 0.50) / 0.08)
    q_conc = torch.exp(-depth_std / 0.20).clamp(0.0, 1.0)
    q_closed = (1.0 - final_t).clamp(0.0, 1.0)
    return (q_acc * q_conc * q_closed).clamp(0.0, 1.0)


def _jensen_gap_approx(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    depth_var = outputs["depth_variance"].detach().float().clamp_min(0.0)
    attn = outputs["medium_attn"].detach().float().clamp_min(0.0)
    bs = outputs["medium_bs"].detach().float().clamp_min(0.0)
    direct = torch.exp((0.5 * attn.square() * depth_var).clamp(max=20.0)) - 1.0
    backscatter = torch.exp((0.5 * bs.square() * depth_var).clamp(max=20.0)) - 1.0
    return {"direct": direct, "backscatter": backscatter}


def _exposure_outputs(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    depth = outputs["depth"].detach().float().clamp_min(0.0)
    attn = outputs["medium_attn"].detach().float().clamp_min(0.0)
    trans = torch.exp(-(attn * depth).clamp_min(0.0)).clamp(1e-6, 1.0)
    gain = (1.0 / trans).clamp(max=1e6)
    j_raw = outputs["J_gaussian_raw"].detach().float().clamp_min(0.0)
    hidden_surface = j_raw * (1.0 - trans)
    hidden_observed = j_raw - outputs["rgb_object"].detach().float()
    return {"trans": trans, "gain": gain, "hidden_surface": hidden_surface, "hidden_observed": hidden_observed}


def _channel_dict(prefix: str, image: torch.Tensor, mask: torch.Tensor) -> Dict[str, Dict[str, float]]:
    return {f"{prefix}_{name}": _stats(_masked(image[..., i : i + 1], mask)) for i, name in enumerate(CHANNELS)}


def _spectral_order(outputs: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, float]:
    result: Dict[str, float] = {}
    checks = {
        "attn_r_ge_g_ge_b": outputs["medium_attn"].detach().float(),
        "bs_b_ge_g_ge_r": outputs["medium_bs"].detach().float(),
        "medium_b_ge_g_ge_r": outputs.get("b_inf", outputs["medium_rgb"]).detach().float(),
    }
    for name, tensor in checks.items():
        vals = _masked(tensor, mask)
        if vals.numel() == 0:
            result[f"{name}_violation_fraction"] = 0.0
            for channel in CHANNELS:
                result[f"{name}_{channel}_mean"] = 0.0
            continue
        for idx, channel in enumerate(CHANNELS):
            result[f"{name}_{channel}_mean"] = float(vals[:, idx].mean().item())
        if name == "attn_r_ge_g_ge_b":
            ok = (vals[:, 0] >= vals[:, 1]) & (vals[:, 1] >= vals[:, 2])
        else:
            ok = (vals[:, 2] >= vals[:, 1]) & (vals[:, 1] >= vals[:, 0])
        result[f"{name}_violation_fraction"] = float((~ok).float().mean().item())
    return result


def _region_summary(
    *,
    name: str,
    mask: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    gt: torch.Tensor,
    surface_rgb: torch.Tensor,
    hybrid_rgb: torch.Tensor,
    dc_raw: Optional[torch.Tensor],
    save_prefix: Optional[Path] = None,
) -> Dict[str, Any]:
    layer_rgb = outputs["rgb"].detach().float()
    j_raw = outputs["J_gaussian_raw"].detach().float()
    j_clamp = j_raw.clamp(0.0, 1.0)
    medium_rgb = outputs.get("b_inf", outputs["medium_rgb"]).detach().float()
    depth_std = outputs["depth_std_relative"].detach().float()
    final_t = outputs["final_transmittance"].detach().float().clamp(0.0, 1.0)
    tail = outputs["rgb_tail"].detach().float().clamp_min(0.0)
    rgb_total_luma = _luma(layer_rgb.abs()).clamp_min(1e-6)
    tail_ratio = (_luma(tail) / rgb_total_luma).clamp(0.0, 100.0)

    j_bias = _medium_axis_projection(j_clamp, medium_rgb)
    bg_bias = _blue_green_minus_red(j_clamp)
    exposure = _exposure_outputs(outputs)
    jensen = _jensen_gap_approx(outputs)
    layer_surface_abs = (layer_rgb - surface_rgb).abs()
    layer_surface_luma = _luma(layer_surface_abs)
    surface_conf = _surface_confidence(outputs)
    high_order = None if dc_raw is None else (j_raw - dc_raw.detach().float())
    high_order_proj = torch.zeros_like(j_bias) if high_order is None else _medium_axis_projection(high_order, medium_rgb)
    clamp_proj = _medium_axis_projection(j_clamp, medium_rgb) - _medium_axis_projection(j_raw.clamp_min(0.0), medium_rgb)

    hidden_bg = _blue_green_minus_red(exposure["hidden_surface"])
    gain_mean = exposure["gain"].mean(dim=-1, keepdim=True)
    direct_gap_mean = jensen["direct"].mean(dim=-1, keepdim=True)
    bs_gap_mean = jensen["backscatter"].mean(dim=-1, keepdim=True)

    # Tail/surface confusion probe: suppress tail only where the frozen render
    # has high surface confidence, then compare to GT.
    tail_suppressed = layer_rgb - surface_conf * tail

    summary: Dict[str, Any] = {
        "name": name,
        "pixels": int(mask.sum().item()),
        "layer_l1_vs_gt": _mean(_masked((layer_rgb - gt).abs(), mask)),
        "surface_l1_vs_gt": _mean(_masked((surface_rgb - gt).abs(), mask)),
        "hybrid_l1_vs_gt": _mean(_masked((hybrid_rgb - gt).abs(), mask)),
        "hit_tail_suppressed_l1_vs_gt": _mean(_masked((tail_suppressed - gt).abs(), mask)),
        "layer_surface_abs_luma": _stats(_masked(layer_surface_luma, mask)),
        "j_bias_medium_axis": _stats(_masked(j_bias, mask)),
        "j_bg_minus_red": _stats(_masked(bg_bias, mask)),
        "tail_ratio": _stats(_masked(tail_ratio, mask)),
        "surface_confidence": _stats(_masked(surface_conf, mask)),
        "final_transmittance": _stats(_masked(final_t, mask)),
        "depth_std_relative": _stats(_masked(depth_std, mask)),
        "exposure_gain_mean": _stats(_masked(gain_mean, mask)),
        "hidden_bg_surface": _stats(_masked(hidden_bg, mask)),
        "jensen_direct_gap_mean_approx": _stats(_masked(direct_gap_mean, mask)),
        "jensen_backscatter_gap_mean_approx": _stats(_masked(bs_gap_mean, mask)),
        "high_order_medium_axis": _stats(_masked(high_order_proj, mask)),
        "clamp_medium_axis_delta": _stats(_masked(clamp_proj, mask)),
        "corr": {
            "layer_surface_vs_j_bias": _corr(_masked(layer_surface_luma, mask), _masked(j_bias, mask)),
            "depth_std_vs_j_bias": _corr(_masked(depth_std, mask), _masked(j_bias, mask)),
            "exposure_gain_vs_j_bias": _corr(_masked(gain_mean, mask), _masked(j_bias, mask)),
            "hidden_bg_vs_j_bias": _corr(_masked(hidden_bg, mask), _masked(j_bias, mask)),
            "jensen_direct_vs_j_bias": _corr(_masked(direct_gap_mean, mask), _masked(j_bias, mask)),
            "jensen_backscatter_vs_j_bias": _corr(_masked(bs_gap_mean, mask), _masked(j_bias, mask)),
            "tail_ratio_vs_j_bias": _corr(_masked(tail_ratio, mask), _masked(j_bias, mask)),
            "high_order_vs_j_bias": _corr(_masked(high_order_proj, mask), _masked(j_bias, mask)),
            "clamp_delta_vs_j_bias": _corr(_masked(clamp_proj, mask), _masked(j_bias, mask)),
        },
        "spectral_order": _spectral_order(outputs, mask),
    }
    summary.update(_channel_dict("exposure_gain", exposure["gain"], mask))
    summary.update(_channel_dict("jensen_direct_gap_approx", jensen["direct"], mask))
    summary.update(_channel_dict("jensen_backscatter_gap_approx", jensen["backscatter"], mask))
    summary.update(_channel_dict("j_raw_gt_1", (j_raw > 1.0).float(), mask))
    if dc_raw is not None:
        dc_clamp = dc_raw.clamp(0.0, 1.0)
        summary["dc_j_bias_medium_axis"] = _stats(_masked(_medium_axis_projection(dc_clamp, medium_rgb), mask))
        summary["dc_bg_minus_red"] = _stats(_masked(_blue_green_minus_red(dc_clamp), mask))
        summary["high_order_abs_luma"] = _stats(_masked(_luma(high_order.abs()), mask))

    if save_prefix is not None and name == "far":
        _maybe_save_image(save_prefix / "rgb_layer.png", layer_rgb)
        _maybe_save_image(save_prefix / "rgb_surface_approx.png", surface_rgb)
        _maybe_save_image(save_prefix / "rgb_hybrid_approx.png", hybrid_rgb)
        _maybe_save_image(save_prefix / "layer_surface_absdiff.png", _normalize_vis(layer_surface_abs, clip=0.25))
        _maybe_save_image(save_prefix / "J_full_clamped.png", j_clamp)
        _maybe_save_image(save_prefix / "J_full_raw_soft.png", j_raw.clamp_min(0.0) / (1.0 + j_raw.clamp_min(0.0)))
        _maybe_save_image(save_prefix / "j_bias_medium_axis.png", _normalize_vis(j_bias, clip=0.15))
        _maybe_save_image(save_prefix / "depth_std_relative.png", _normalize_vis(depth_std, clip=1.0))
        _maybe_save_image(save_prefix / "exposure_gain_mean.png", _normalize_vis(torch.log1p(gain_mean), clip=math.log(20.0)))
        _maybe_save_image(save_prefix / "hidden_bg_surface.png", _normalize_vis(hidden_bg, clip=0.25))
        _maybe_save_image(save_prefix / "tail_ratio.png", _normalize_vis(tail_ratio, clip=1.0))
        _maybe_save_image(save_prefix / "surface_confidence.png", surface_conf)
        _maybe_save_image(save_prefix / "far_mask.png", mask.float())
        if dc_raw is not None:
            _maybe_save_image(save_prefix / "J_dc_clamped.png", dc_raw.clamp(0.0, 1.0))
            _maybe_save_image(save_prefix / "J_high_order_abs.png", _normalize_vis(high_order.abs(), clip=0.25))
    return summary


def _aggregate_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {}
    scalar_keys = [
        "layer_l1_vs_gt",
        "surface_l1_vs_gt",
        "hybrid_l1_vs_gt",
        "hit_tail_suppressed_l1_vs_gt",
    ]
    aggregate: Dict[str, Any] = {}
    for key in scalar_keys:
        vals = torch.tensor([float(row[key]) for row in rows], dtype=torch.float32)
        aggregate[key] = _stats(vals)
    for group in (
        "layer_surface_abs_luma",
        "j_bias_medium_axis",
        "j_bg_minus_red",
        "tail_ratio",
        "surface_confidence",
        "final_transmittance",
        "depth_std_relative",
        "exposure_gain_mean",
        "hidden_bg_surface",
        "jensen_direct_gap_mean_approx",
        "jensen_backscatter_gap_mean_approx",
        "high_order_medium_axis",
        "clamp_medium_axis_delta",
    ):
        vals = torch.tensor([float(row[group]["mean"]) for row in rows if group in row], dtype=torch.float32)
        aggregate[f"{group}_view_mean"] = _stats(vals)
    corr_keys = sorted(rows[0].get("corr", {}).keys())
    aggregate["corr_view_mean"] = {}
    for key in corr_keys:
        vals = torch.tensor([float(row.get("corr", {}).get(key, 0.0)) for row in rows], dtype=torch.float32)
        aggregate["corr_view_mean"][key] = _stats(vals)
    spectral_keys = sorted(rows[0].get("spectral_order", {}).keys())
    aggregate["spectral_order_view_mean"] = {}
    for key in spectral_keys:
        vals = torch.tensor([float(row.get("spectral_order", {}).get(key, 0.0)) for row in rows], dtype=torch.float32)
        aggregate["spectral_order_view_mean"][key] = _stats(vals)
    return aggregate


def diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, step = eval_setup(args.load_config)
    pipeline.eval()
    model = pipeline.model
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_clear_proxy = bool(getattr(model.config, "clear_proxy_enabled", False))
    original_dual = bool(getattr(model.config, "dual_color_enabled", False))
    original_luma_scale = float(getattr(model.config, "clear_sh_luminance_scale", 1.0))
    original_chroma_scale = float(getattr(model.config, "clear_sh_chroma_scale", 0.0))
    model.config.clear_proxy_enabled = True

    image_rows: List[Dict[str, Any]] = []
    region_buckets: Dict[str, List[Dict[str, Any]]] = {}
    proxy_diff_stats: List[Dict[str, float]] = []

    with torch.no_grad():
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= args.max_images:
                break

            model.config.dual_color_enabled = False
            outputs = model.get_outputs_for_camera(camera=camera)
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"]).detach().float()
            far_mask = _load_far_mask(args.far_mask_dir, image_idx, outputs["depth"])
            regions = _load_region_masks(args.region_mask_dir, image_idx, outputs["depth"])

            dc_raw: Optional[torch.Tensor] = None
            if args.compute_dc:
                model.config.dual_color_enabled = True
                model.config.clear_sh_luminance_scale = 0.0
                model.config.clear_sh_chroma_scale = 0.0
                dc_outputs = model.get_outputs_for_camera(camera=camera)
                dc_raw = dc_outputs.get("J_intrinsic_raw", dc_outputs["J_gaussian_raw"]).detach().float()
                model.config.dual_color_enabled = False

            surface_rgb = _surface_rgb(outputs)
            q_surface = _surface_confidence(outputs)
            hybrid_rgb = q_surface * surface_rgb + (1.0 - q_surface) * outputs["rgb"].detach().float()

            masks: Dict[str, torch.Tensor] = {"all": _safe_mask(None, outputs["depth"]), "far": far_mask}
            for key, value in regions.items():
                masks[key] = value
                masks[f"far_{key}"] = far_mask & value
            masks["high_depth_variance"] = outputs["depth_std_relative"].detach().float() > torch.quantile(
                outputs["depth_std_relative"].detach().float().reshape(-1), 0.75
            )
            masks["low_depth_variance"] = outputs["depth_std_relative"].detach().float() <= torch.quantile(
                outputs["depth_std_relative"].detach().float().reshape(-1), 0.25
            )
            masks["high_accumulation"] = outputs["accumulation"].detach().float() > 0.70
            masks["low_accumulation"] = outputs["accumulation"].detach().float() < 0.20

            image_summary: Dict[str, Any] = {
                "image_index": image_idx,
                "regions": {},
                "height": int(outputs["depth"].shape[0]),
                "width": int(outputs["depth"].shape[1]),
            }
            save_prefix = args.output_dir / f"view_{image_idx:04d}" if args.save_images else None
            for name, mask in sorted(masks.items()):
                if mask.sum().item() == 0:
                    continue
                row = _region_summary(
                    name=name,
                    mask=mask,
                    outputs=outputs,
                    gt=gt,
                    surface_rgb=surface_rgb,
                    hybrid_rgb=hybrid_rgb,
                    dc_raw=dc_raw,
                    save_prefix=save_prefix,
                )
                image_summary["regions"][name] = row
                region_buckets.setdefault(name, []).append(row)

            if "J_proxy_abs_diff_from_renderer_clear" in outputs:
                proxy_diff_stats.append(_stats(outputs["J_proxy_abs_diff_from_renderer_clear"].detach().float()))
            image_rows.append(image_summary)

    model.config.clear_proxy_enabled = original_clear_proxy
    model.config.dual_color_enabled = original_dual
    model.config.clear_sh_luminance_scale = original_luma_scale
    model.config.clear_sh_chroma_scale = original_chroma_scale

    aggregate = {name: _aggregate_rows(rows) for name, rows in sorted(region_buckets.items())}
    repo = Path(__file__).resolve().parents[2]
    result = {
        "experiment": "renderer_color_bias_audit",
        "scene_name": args.scene_name,
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "git_commit": _git_commit(repo),
        "max_images": int(args.max_images),
        "far_mask_dir": str(args.far_mask_dir) if args.far_mask_dir else None,
        "region_mask_dir": str(args.region_mask_dir) if args.region_mask_dir else None,
        "notes": {
            "a0_limitation": (
                "The CUDA API does not expose per-Gaussian pixel weights, so this audit does not perform "
                "a strict explicit per-Gaussian forward/backward equivalence check. It records the existing "
                "J_proxy-vs-renderer-clear equivalence and uses pixel-summary approximations for A1/A2."
            ),
            "surface_approximation": (
                "Surface RGB is approximated from J_gaussian_raw, expected depth, accumulation, and tied/implicit "
                "medium color; it is a mechanism probe, not a replacement renderer."
            ),
        },
        "proxy_clear_equivalence_absdiff": proxy_diff_stats,
        "aggregate": aggregate,
        "images": image_rows,
    }
    output_json = args.output_dir / "renderer_color_bias_audit_summary.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-name", type=str, default="unknown")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--far-mask-dir", type=Path, default=None)
    parser.add_argument("--region-mask-dir", type=Path, default=None)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--compute-dc", action="store_true")
    args = parser.parse_args()

    result = diagnose(args)
    compact = {
        "scene": result["scene_name"],
        "checkpoint": result["checkpoint"],
        "far": result["aggregate"].get("far", {}),
        "far_object": result["aggregate"].get("far_object", {}),
        "far_water": result["aggregate"].get("far_water", {}),
    }
    print(json.dumps(compact, indent=2))
    print(f"saved={args.output_dir / 'renderer_color_bias_audit_summary.json'}")


if __name__ == "__main__":
    main()
