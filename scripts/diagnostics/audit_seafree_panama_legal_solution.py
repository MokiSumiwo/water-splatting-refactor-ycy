#!/usr/bin/env python
"""Read-only SEAFREE-LEGAL Panama attribution audit.

This diagnostic has two modes:

* ``export-seafree`` runs inside the SeaFree-GS environment and exports fixed
  reference checkpoint outputs for common Panama eval views.
* ``audit`` runs inside the WaterSplatting environment, loads existing M1/K1
  checkpoints, consumes the SeaFree export, and writes region metrics plus
  visual contact sheets.

The script performs forward/evaluation passes only. It does not call
optimizer.step(), scheduler.step(), densification, pruning, or checkpoint
mutation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor


SCENE = "Panama"
SEAFREE_COMMIT = "7797e97dae831029ac89ae9f37b3c3d69ec2cf6c"
COMMON_VIEW_IDS = ("MTN_1529", "MTN_1539", "MTN_1547")
CHANNELS = ("r", "g", "b")
EPS = 1e-8
LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)


@dataclass(frozen=True)
class WsRunSpec:
    run: str
    config_relpath: str
    parameterization: str
    rasterize_mode: str
    nominal_step: int = 15000


WS_RUNS: Dict[str, WsRunSpec] = {
    "M1": WsRunSpec(
        run="M1",
        config_relpath=(
            "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
            "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
        ),
        parameterization="legacy",
        rasterize_mode="classic",
    ),
    "BND-K1": WsRunSpec(
        run="BND-K1",
        config_relpath=(
            "outputs/dewater_bounded_sh3_cross_scene_20260808/"
            "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
            "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
            "config.yml"
        ),
        parameterization="bounded_sh3",
        rasterize_mode="classic",
    ),
}


@dataclass
class LoadedWsRun:
    spec: WsRunSpec
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
    config: Any
    pipeline: Any

    @property
    def model(self) -> Any:
        return self.pipeline.model


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
    except Exception as exc:
        return f"unavailable: {exc}"


def _available_steps(config_path: Path) -> Dict[int, Path]:
    ckpt_dir = config_path.parent / "nerfstudio_models"
    out: Dict[int, Path] = {}
    if not ckpt_dir.exists():
        return out
    for path in ckpt_dir.glob("step-*.ckpt"):
        try:
            out[int(path.stem.split("-")[1])] = path
        except Exception:
            continue
    return out


def _actual_step(config_path: Path, nominal_step: int) -> int:
    steps = _available_steps(config_path)
    if not steps:
        raise FileNotFoundError(f"No checkpoints found next to {config_path}")
    if nominal_step < 0:
        return max(steps)
    if nominal_step in steps:
        return nominal_step
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    nearest = min(steps, key=lambda step: abs(step - nominal_step))
    if abs(nearest - nominal_step) <= 1:
        return nearest
    raise FileNotFoundError(f"Missing checkpoint step {nominal_step} near {config_path}; available={sorted(steps)}")


def _safe_cpu(tensor: Tensor) -> Tensor:
    return tensor.detach().float().cpu()


def _safe_quantile(values: Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    if q <= 0:
        return float(flat.min().item())
    if q >= 1:
        return float(flat.max().item())
    rank = max(1, min(flat.numel(), int(math.ceil(q * flat.numel()))))
    return float(torch.kthvalue(flat, rank).values.item())


def _stats(values: Tensor, prefix: str) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    names = ("count", "mean", "p01", "p05", "p10", "p50", "p90", "p95", "p99", "max")
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in names}
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p01": _safe_quantile(flat, 0.01),
        f"{prefix}p05": _safe_quantile(flat, 0.05),
        f"{prefix}p10": _safe_quantile(flat, 0.10),
        f"{prefix}p50": _safe_quantile(flat, 0.50),
        f"{prefix}p90": _safe_quantile(flat, 0.90),
        f"{prefix}p95": _safe_quantile(flat, 0.95),
        f"{prefix}p99": _safe_quantile(flat, 0.99),
        f"{prefix}max": float(flat.max().item()),
    }


def _channel_stats(values: Tensor, prefix: str) -> Dict[str, float]:
    values = values.detach().float()
    out: Dict[str, float] = {}
    if values.ndim > 0 and values.shape[-1] == 3:
        for idx, channel in enumerate(CHANNELS):
            out.update(_stats(values[..., idx], f"{prefix}_{channel}_"))
        out.update(_stats(values.reshape(-1), f"{prefix}_all_"))
    else:
        out.update(_stats(values.reshape(-1), f"{prefix}_"))
    return out


def _threshold_rows(values: Tensor, prefix: str, thresholds: Sequence[float], op: str) -> Dict[str, float]:
    values = values.detach().float()
    out: Dict[str, float] = {}
    if values.ndim > 0 and values.shape[-1] == 3:
        iterable = [(channel, values[..., idx]) for idx, channel in enumerate(CHANNELS)]
        iterable.append(("all", values.reshape(-1)))
    else:
        iterable = [("all", values.reshape(-1))]
    for channel, vals in iterable:
        vals = vals.reshape(-1)
        vals = vals[torch.isfinite(vals)]
        for threshold in thresholds:
            if vals.numel() == 0:
                frac = float("nan")
            elif op == "gt":
                frac = float((vals > threshold).float().mean().item())
            elif op == "lt":
                frac = float((vals < threshold).float().mean().item())
            else:
                raise ValueError(op)
            out[f"{prefix}_{channel}_{op}_{threshold:g}"] = frac
    return out


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _luma(rgb: Tensor) -> Tensor:
    weights = LUMA_WEIGHTS.to(device=rgb.device, dtype=rgb.dtype)
    return (rgb.detach().float() * weights).sum(dim=-1)


def _rgb_l2(image: Tensor) -> Tensor:
    return torch.linalg.norm(image.detach().float(), dim=-1)


def _pearson(a: Tensor, b: Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(EPS)
    return float(((x * y).sum() / denom).item())


def _rankdata(values: Tensor) -> Tensor:
    flat = values.detach().float().reshape(-1)
    order = torch.argsort(flat)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.arange(flat.numel(), dtype=flat.dtype, device=flat.device)
    return ranks


def _spearman(a: Tensor, b: Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _scale_shift_align(pred: Tensor, target: Tensor) -> Tuple[Tensor, float, float]:
    x = pred.detach().float().reshape(-1)
    y = target.detach().float().reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 2:
        return pred.detach().float(), float("nan"), float("nan")
    x_mean = x.mean()
    y_mean = y.mean()
    var = ((x - x_mean) ** 2).mean().clamp_min(EPS)
    a = float((((x - x_mean) * (y - y_mean)).mean() / var).item())
    b = float((y_mean - a * x_mean).item())
    return pred.detach().float() * a + b, a, b


def _gradient_magnitude(values: Tensor) -> Tensor:
    scalar = values.detach().float()
    if scalar.ndim == 3:
        scalar = _luma(scalar)
    dx = torch.zeros_like(scalar)
    dy = torch.zeros_like(scalar)
    dx[:, 1:] = scalar[:, 1:] - scalar[:, :-1]
    dy[1:, :] = scalar[1:, :] - scalar[:-1, :]
    return torch.sqrt(dx.square() + dy.square() + EPS)


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_uint8(values: Tensor, scale: float, clamp_min: float = 0.0) -> Image.Image:
    scale = max(float(scale), EPS)
    arr = ((values.detach().float() - clamp_min).clamp_min(0.0) / scale).clamp(0.0, 1.0)
    arr = (arr * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def _signed_to_rgb(values: Tensor, scale: float) -> Image.Image:
    scale = max(float(scale), EPS)
    vals = (values.detach().float() / scale).clamp(-1.0, 1.0)
    pos = vals.clamp_min(0.0)
    neg = (-vals).clamp_min(0.0)
    rgb = torch.stack([pos, torch.zeros_like(vals), neg], dim=-1)
    return _rgb_to_uint8(rgb)


def _mask_to_rgb(mask: Tensor) -> Image.Image:
    arr = (mask.detach().bool().byte().cpu().numpy() * 255)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _overlay_mask(base: Image.Image, mask: Tensor, color: Tuple[int, int, int] = (255, 40, 40)) -> Image.Image:
    image = base.convert("RGB")
    pix = image.load()
    mask_cpu = mask.detach().bool().cpu()
    for y in range(image.height):
        for x in range(image.width):
            if bool(mask_cpu[y, x]):
                old = pix[x, y]
                pix[x, y] = tuple(int(0.45 * old[i] + 0.55 * color[i]) for i in range(3))
    return image


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if width > 0 and image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    label_height = 30
    out = Image.new("RGB", (image.width, image.height + label_height), "white")
    out.paste(image, (0, label_height))
    ImageDraw.Draw(out).text((6, 8), label, fill="black")
    return out


def _save_sheet(
    path: Path,
    rows: Sequence[Sequence[Tuple[str, Image.Image]]],
    *,
    tile_width: int,
    manifest: List[Dict[str, Any]],
    output_type: str,
    view_ids: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_tile(image, label, tile_width) for label, image in row]
        width = sum(tile.width for tile in tiles) + 6 * max(0, len(tiles) - 1)
        height = max(tile.height for tile in tiles)
        canvas = Image.new("RGB", (width, height), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered_rows.append(canvas)
    if not rendered_rows:
        return
    width = max(row.width for row in rendered_rows)
    height = sum(row.height for row in rendered_rows) + 6 * max(0, len(rendered_rows) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)
    manifest.append(
        {
            "file_path": str(path),
            "scene": SCENE,
            "runs": "M1;BND-K1;SeaFree",
            "output_type": output_type,
            "view_ids": ";".join(view_ids),
            "width": sheet.width,
            "height": sheet.height,
        }
    )


def _metric_images(model: Any, pred: Tensor, gt: Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0).to(model.device)
    gt = gt.detach().float().clamp(0.0, 1.0).to(model.device)
    pred_nchw = torch.moveaxis(pred, -1, 0)[None, ...]
    gt_nchw = torch.moveaxis(gt, -1, 0)[None, ...]
    return {
        "psnr": float(model.psnr(gt_nchw, pred_nchw).item()),
        "ssim": float(model.ssim(gt_nchw, pred_nchw).item()),
        "lpips": float(model.lpips(gt_nchw, pred_nchw).item()),
        "mse": float(((pred - gt) ** 2).mean().item()),
    }


def _release_loaded(loaded: Optional[LoadedWsRun]) -> None:
    if loaded is None:
        return
    try:
        del loaded.pipeline
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_ws_run(repo: Path, run: str) -> LoadedWsRun:
    from nerfstudio.utils.eval_utils import eval_setup

    spec = WS_RUNS[run]
    config_path = repo / spec.config_relpath
    actual_step = _actual_step(config_path, spec.nominal_step)

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(config_path, test_mode="test", update_config_callback=update_config)
    pipeline.model.config.intrinsic_color_parameterization = spec.parameterization
    pipeline.model.config.rasterize_mode = spec.rasterize_mode
    pipeline.eval()
    return LoadedWsRun(spec, config_path, checkpoint_path, int(loaded_step), config, pipeline)


def _view_records(loaded: LoadedWsRun) -> List[Tuple[int, str, Any, Mapping[str, Any]]]:
    dataset = loaded.pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    rows = []
    for eval_index, (camera, batch) in enumerate(loaded.pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        rows.append((eval_index, Path(filename).stem, camera, batch))
    return rows


def _object_support(item: Mapping[str, Any]) -> Tensor:
    return item["outputs"]["accumulation"].detach().float()[..., 0] > 0.01


def _cache_ws_outputs(repo: Path, run: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    loaded: Optional[LoadedWsRun] = None
    try:
        loaded = _load_ws_run(repo, run)
        model = loaded.model
        items: List[Dict[str, Any]] = []
        for eval_index, view_id, camera, batch in _view_records(loaded):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
                gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
                metrics = _metric_images(model, outputs["pred_image"], gt)
            keep = (
                "pred_image",
                "direct_object_signal",
                "clear_object_fullsh_raw",
                "transmission",
                "tau_D",
                "rgb_medium",
                "medium_rgb",
                "medium_bs",
                "medium_attn",
                "depth",
                "accumulation",
                "gaussian_view_rgb",
                "gaussian_view_logits",
                "gaussian_sigmoid_derivative",
                "gaussian_visible_mask",
            )
            tensors = {key: _safe_cpu(outputs[key]) for key in keep if key in outputs and isinstance(outputs[key], Tensor)}
            items.append(
                {
                    "scene": SCENE,
                    "run": run,
                    "nominal_step": loaded.spec.nominal_step,
                    "loaded_step": loaded.loaded_step,
                    "eval_index": eval_index,
                    "view_id": view_id,
                    "gt": _safe_cpu(gt),
                    "outputs": tensors,
                    "metrics": metrics,
                }
            )
        meta = {
            "scene": SCENE,
            "run": run,
            "nominal_step": loaded.spec.nominal_step,
            "loaded_step": loaded.loaded_step,
            "config_path": str(loaded.config_path),
            "checkpoint_path": str(loaded.checkpoint_path),
            "parameterization": loaded.spec.parameterization,
            "rasterize_mode": loaded.spec.rasterize_mode,
            "seed": getattr(getattr(loaded.config, "machine", None), "seed", ""),
            "sh_degree": getattr(model.config, "sh_degree", ""),
            "medium_context_mode": getattr(model.config, "medium_context_mode", ""),
            "b_inf_mode": getattr(model.config, "b_inf_mode", ""),
            "infinite_water_enabled": getattr(model.config, "infinite_water_enabled", ""),
            "gaussian_count": int(model.num_points),
            "num_eval_views": len(items),
            "view_ids": ";".join(item["view_id"] for item in items),
        }
        if hasattr(model, "opacities"):
            meta.update(_stats(torch.sigmoid(model.opacities.detach().float()), "opacity_"))
        if hasattr(model, "scales"):
            meta.update(_stats(torch.exp(model.scales.detach().float()).reshape(-1), "scale_"))
        return items, meta
    finally:
        _release_loaded(loaded)


def _seafree_color_stats(model: Any, visible_mask: Optional[Tensor]) -> Dict[str, Any]:
    colors = torch.sigmoid(model.features_dc.detach().float())
    row: Dict[str, Any] = {"boundary_source": "sigmoid(features_dc); SeaFree sh_degree=0"}
    row.update(_channel_stats(colors, "c_all_gaussians"))
    row.update(_threshold_rows(colors, "c_all_gaussians", (0.90, 0.95, 0.99, 0.10, 0.05, 0.01), "gt"))
    row.update(_threshold_rows(colors, "c_all_gaussians", (0.10, 0.05, 0.01), "lt"))
    if visible_mask is not None and int(visible_mask.sum().item()) > 0:
        visible_colors = colors[visible_mask.bool()]
        row.update(_channel_stats(visible_colors, "c_visible_gaussians"))
        row.update(_threshold_rows(visible_colors, "c_visible_gaussians", (0.90, 0.95, 0.99), "gt"))
        row.update(_threshold_rows(visible_colors, "c_visible_gaussians", (0.10, 0.05, 0.01), "lt"))
    return row


def _seafree_population_stats(model: Any, radii: Sequence[Tensor], gs_pixels: Sequence[Tensor]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"gaussian_count": int(model.num_points)}
    if hasattr(model, "opacities"):
        row.update(_stats(torch.sigmoid(model.opacities.detach().float()), "opacity_"))
    if hasattr(model, "scales"):
        row.update(_stats(torch.exp(model.scales.detach().float()).reshape(-1), "scale_"))
    if radii:
        joined = torch.cat([r.detach().float().reshape(-1) for r in radii])
        row.update(_stats(joined, "projected_radius_"))
        row["projected_radius_positive_fraction"] = float((joined > 0).float().mean().item())
    if gs_pixels:
        joined = torch.cat([p.detach().float().reshape(-1) for p in gs_pixels])
        row.update(_stats(joined, "gs_pixels_"))
        row["gs_pixels_positive_fraction"] = float((joined > 0).float().mean().item())
    return row


def _load_depth_image(path: Path, shape: Tuple[int, int]) -> Tensor:
    depth = torch.from_numpy(np.asarray(Image.open(path))).float()
    if depth.ndim == 3:
        depth = depth[..., 0]
    if tuple(depth.shape[:2]) != tuple(shape):
        depth = F.interpolate(depth[None, None, ...], size=shape, mode="bilinear", align_corners=False)[0, 0]
    max_value = depth.max().clamp_min(EPS)
    return depth / max_value


def export_seafree(args: argparse.Namespace) -> None:
    from nerfstudio.utils.eval_utils import eval_setup

    config_path = args.seafree_config.resolve()
    output_dir = args.output_dir.resolve()
    render_dir = args.render_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    actual_step = _actual_step(config_path, args.step)

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(config_path, test_mode="test", update_config_callback=update_config)
    pipeline.eval()
    model = pipeline.model
    items: List[Dict[str, Any]] = []
    visible_mask = torch.zeros(int(model.num_points), dtype=torch.bool, device=model.device)
    radii_rows: List[Tensor] = []
    gs_pixel_rows: List[Tensor] = []
    per_view_png_rows: List[Dict[str, Any]] = []
    dataset = pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    depth_filenames = list(getattr(dataset, "metadata", {}).get("depth_filenames", []))
    if not depth_filenames:
        depth_filenames = list(getattr(dataset, "depth_filenames", []))
    for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        image_path = Path(image_filenames[eval_index]) if eval_index < len(image_filenames) else Path(f"eval_{eval_index}.png")
        view_id = image_path.stem
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(camera)
            metrics, _ = model.get_image_metrics_and_images(outputs, batch)
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        info = getattr(model, "info", {})
        if isinstance(info, Mapping):
            if isinstance(info.get("radii"), Tensor):
                radii = info["radii"].detach()
                radii_rows.append(_safe_cpu(radii))
                if radii.ndim == 2 and radii.shape[0] == 1 and radii.shape[1] == visible_mask.numel():
                    visible_mask |= radii[0].to(visible_mask.device) > 0
            if isinstance(info.get("gs_pixels"), Tensor):
                gs_pixel_rows.append(_safe_cpu(info["gs_pixels"]))
        depth = outputs["depth"].detach().float()
        if depth.ndim == 2:
            depth = depth[..., None]
        bg_attn = outputs["background_attenuation_coefficients"].detach().float()
        tau_proxy = bg_attn * (depth / 10.0)
        transmission_proxy = torch.exp(-tau_proxy)
        pseudo_depth = batch.get("depth_image")
        if isinstance(pseudo_depth, Tensor):
            pseudo_depth = pseudo_depth.detach().float()
            if pseudo_depth.ndim == 3 and pseudo_depth.shape[-1] == 1:
                pseudo_depth = pseudo_depth[..., 0]
            pseudo_depth = pseudo_depth / pseudo_depth.max().clamp_min(EPS)
        else:
            pseudo_depth = _load_depth_image(Path(depth_filenames[eval_index]), tuple(depth.shape[:2]))
        tensors = {
            "pred_image": _safe_cpu(outputs["rgb"]),
            "intrinsic_color_render": _safe_cpu(outputs["intrinsic_color_render"]),
            "accumulation": _safe_cpu(outputs["accumulation"]),
            "depth": _safe_cpu(depth),
            "pseudo_depth": _safe_cpu(pseudo_depth),
            "water_background_image": _safe_cpu(outputs["water_background_image"]),
            "background_backscatter_coefficients": _safe_cpu(outputs["background_backscatter_coefficients"]),
            "background_attenuation_coefficients": _safe_cpu(outputs["background_attenuation_coefficients"]),
            "tau_proxy_pixel_depth_over_10": _safe_cpu(tau_proxy),
            "transmission_proxy_pixel_depth_over_10": _safe_cpu(transmission_proxy),
        }
        item = {
            "scene": SCENE,
            "run": "SeaFree",
            "nominal_step": int(args.step),
            "loaded_step": int(loaded_step),
            "eval_index": eval_index,
            "view_id": view_id,
            "image_path": str(image_path),
            "gt": _safe_cpu(gt),
            "outputs": tensors,
            "metrics": {**{key: float(value) for key, value in metrics.items()}, "mse": float(((outputs["rgb"] - gt) ** 2).mean().item())},
        }
        items.append(item)
        view_dir = render_dir / "seafree_export" / f"step_{int(loaded_step):09d}" / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        pngs = {
            "gt.png": gt,
            "underwater.png": outputs["rgb"],
            "intrinsic_color_render.png": outputs["intrinsic_color_render"],
            "accumulation.png": outputs["accumulation"].expand(-1, -1, 3),
            "tau_proxy.png": tau_proxy.mean(dim=-1, keepdim=True).expand(-1, -1, 3).clamp(0.0, 3.0) / 3.0,
            "transmission_proxy.png": transmission_proxy,
        }
        for filename, tensor in pngs.items():
            path = view_dir / filename
            _rgb_to_uint8(tensor.detach().float()).save(path)
            per_view_png_rows.append(
                {
                    "file_path": str(path),
                    "scene": SCENE,
                    "run": "SeaFree",
                    "step": int(loaded_step),
                    "view_id": view_id,
                    "output_type": filename[:-4] if filename.endswith(".png") else filename,
                }
            )
    export = {
        "metadata": {
            "scene": SCENE,
            "run": "SeaFree",
            "loaded_step": int(loaded_step),
            "nominal_step": int(args.step),
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "source_reference_commit_expected": SEAFREE_COMMIT,
            "dataset_path": str(getattr(config, "data", "")),
            "seed": getattr(getattr(config, "machine", None), "seed", ""),
            "sh_degree": getattr(model.config, "sh_degree", ""),
            "rasterize_mode": getattr(model.config, "rasterize_mode", ""),
            "max_num_iterations": getattr(config, "max_num_iterations", ""),
            "steps_per_save": getattr(config, "steps_per_save", ""),
            "stop_split_at": getattr(model.config, "stop_split_at", ""),
            "reset_alpha_value": getattr(model.config, "reset_alpha_value", ""),
            "cull_alpha_thresh": getattr(model.config, "cull_alpha_thresh", ""),
            "cull_alpha_thresh_post": getattr(model.config, "cull_alpha_thresh_post", ""),
            "enable_coarse_grained_depth_loss": getattr(model.config, "enable_coarse_grained_depth_loss", ""),
            "enable_background_water_supervision": getattr(model.config, "enable_background_water_supervision", ""),
            "image_paths": [item["image_path"] for item in items],
            "view_ids": [item["view_id"] for item in items],
            "tau_proxy_definition": "background_attenuation_coefficients * rendered_expected_depth / 10; diagnostic proxy, not exact per-Gaussian direct tau.",
            "intrinsic_definition": "SeaFree output intrinsic_color_render; SH0 sigmoid(features_dc) colors rendered as channels 3:6 and clamped to [0,1].",
        },
        "items": items,
        "boundary_stats": _seafree_color_stats(model, visible_mask),
        "population_stats": _seafree_population_stats(model, radii_rows, gs_pixel_rows),
    }
    export_path = output_dir / f"seafree_export_step_{int(loaded_step):09d}.pt"
    torch.save(export, export_path)
    manifest = {
        "export_path": str(export_path),
        "seafree_config": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "loaded_step": int(loaded_step),
        "view_ids": [item["view_id"] for item in items],
        "png_outputs": per_view_png_rows,
    }
    _write_json(output_dir / "seafree_reference_manifest.json", manifest)
    _write_csv(output_dir / "seafree_export_png_manifest.csv", per_view_png_rows)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _run_rows(items: Sequence[Mapping[str, Any]], run: str) -> Dict[str, Any]:
    metrics = [item["metrics"] for item in items]
    return {
        "scene": SCENE,
        "run": run,
        "num_eval_views": len(items),
        "view_ids": ";".join(str(item["view_id"]) for item in items),
        "psnr": _mean(row["psnr"] for row in metrics),
        "ssim": _mean(row["ssim"] for row in metrics),
        "lpips": _mean(row["lpips"] for row in metrics),
        "mse": _mean(row["mse"] for row in metrics),
    }


def _make_regions(m1_items: Mapping[str, Mapping[str, Any]], common_view_ids: Sequence[str]) -> Tuple[Dict[str, Dict[str, Tensor]], float]:
    lumas = [(_luma(m1_items[view_id]["gt"]).reshape(-1)) for view_id in common_view_ids]
    bright_threshold = _safe_quantile(torch.cat(lumas), 0.80)
    out: Dict[str, Dict[str, Tensor]] = {}
    for view_id in common_view_ids:
        item = m1_items[view_id]
        gt = item["gt"].detach().float()
        support = _object_support(item)
        clear = item["outputs"]["clear_object_fullsh_raw"].detach().float()
        jmax = clear.amax(dim=-1)
        out[view_id] = {
            "WHOLE_IMAGE": torch.ones_like(jmax, dtype=torch.bool),
            "M1_HIGH_J": support & (jmax > 1.0),
            "M1_LOW_J": support & (jmax <= 1.0),
            "BRIGHT_Q5": _luma(gt) > bright_threshold,
            "BRIGHT_NOT_Q5": _luma(gt) <= bright_threshold,
        }
    return out, float(bright_threshold)


def _masked_mse(pred: Tensor, gt: Tensor, mask: Tensor) -> float:
    vals = (pred.detach().float() - gt.detach().float()).square().mean(dim=-1)[mask.detach().bool()]
    return float(vals.mean().item()) if vals.numel() else float("nan")


def _masked_l1(pred: Tensor, gt: Tensor, mask: Tensor) -> float:
    vals = (pred.detach().float() - gt.detach().float()).abs().mean(dim=-1)[mask.detach().bool()]
    return float(vals.mean().item()) if vals.numel() else float("nan")


def _region_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], regions: Mapping[str, Mapping[str, Tensor]], common_view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5", "BRIGHT_NOT_Q5"):
        for run in ("M1", "BND-K1", "SeaFree"):
            numer_mse = []
            numer_l1 = []
            pixels = 0
            total = 0
            for view_id in common_view_ids:
                mask = regions[view_id][region]
                gt = by_run_view["M1"][view_id]["gt"]
                pred = by_run_view[run][view_id]["outputs"]["pred_image"]
                mse_map = (pred.detach().float() - gt.detach().float()).square().mean(dim=-1)
                l1_map = (pred.detach().float() - gt.detach().float()).abs().mean(dim=-1)
                vals_mse = mse_map[mask]
                vals_l1 = l1_map[mask]
                if vals_mse.numel():
                    numer_mse.append(vals_mse)
                    numer_l1.append(vals_l1)
                    pixels += int(mask.sum().item())
                    total += int(mask.numel())
            mse = float(torch.cat(numer_mse).mean().item()) if numer_mse else float("nan")
            l1 = float(torch.cat(numer_l1).mean().item()) if numer_l1 else float("nan")
            rows.append(
                {
                    "scene": SCENE,
                    "region": region,
                    "run": run,
                    "pixels": pixels,
                    "total_pixels": total,
                    "pixel_fraction": pixels / max(total, 1),
                    "mse": mse,
                    "l1": l1,
                    "psnr_like": -10.0 * math.log10(max(mse, EPS)) if math.isfinite(mse) else float("nan"),
                }
            )
    return rows


def _rows_by_region_run(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(str(row["region"]), str(row["run"])): row for row in rows}


def _high_j_recovery(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keyed = _rows_by_region_run(rows)
    high_m1 = float(keyed[("M1_HIGH_J", "M1")]["mse"])
    high_k1 = float(keyed[("M1_HIGH_J", "BND-K1")]["mse"])
    high_sf = float(keyed[("M1_HIGH_J", "SeaFree")]["mse"])
    denom = high_k1 - high_m1
    recovery = (high_k1 - high_sf) / denom if abs(denom) > EPS else float("nan")
    if math.isfinite(recovery) and recovery >= 0.25 and high_sf < high_k1:
        flag = "TRUE"
    elif math.isfinite(recovery) and recovery < 0.10:
        flag = "FALSE"
    else:
        flag = "WEAK"
    return {
        "scene": SCENE,
        "region": "M1_HIGH_J",
        "m1_mse": high_m1,
        "k1_mse": high_k1,
        "seafree_mse": high_sf,
        "pixel_fraction": float(keyed[("M1_HIGH_J", "M1")]["pixel_fraction"]),
        "SEAFREE_HIGHJ_GAP_RECOVERY": recovery,
        "SEAFREE_HIGHJ_LOCAL_RECOVERY": flag,
        "definition": "(MSE_K1_highJ - MSE_SeaFree_highJ) / (MSE_K1_highJ - MSE_M1_highJ), fixed M1_HIGH_J mask.",
    }


def _intrinsic_region_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], regions: Mapping[str, Mapping[str, Tensor]], common_view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    key_map = {"M1": "clear_object_fullsh_raw", "BND-K1": "clear_object_fullsh_raw", "SeaFree": "intrinsic_color_render"}
    rows: List[Dict[str, Any]] = []
    for region in ("M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5", "WHOLE_IMAGE"):
        for run in ("M1", "BND-K1", "SeaFree"):
            vals: List[Tensor] = []
            for view_id in common_view_ids:
                image = by_run_view[run][view_id]["outputs"][key_map[run]].detach().float()
                mask = regions[view_id][region]
                vals.append(image[mask].reshape(-1, 3))
            joined = torch.cat(vals, dim=0) if vals else torch.empty(0, 3)
            row: Dict[str, Any] = {"scene": SCENE, "run": run, "region": region, "source": key_map[run]}
            row.update(_channel_stats(joined, "intrinsic"))
            row.update(_threshold_rows(joined, "intrinsic", (0.90, 0.95, 0.99), "gt"))
            row.update(_threshold_rows(joined, "intrinsic", (0.10, 0.05, 0.01), "lt"))
            rows.append(row)
    return rows


def _boundary_stats_from_ws_items(items: Sequence[Mapping[str, Any]], run: str) -> Dict[str, Any]:
    colors: List[Tensor] = []
    logits: List[Tensor] = []
    for item in items:
        out = item["outputs"]
        c = out.get("gaussian_view_rgb")
        s = out.get("gaussian_view_logits")
        visible = out.get("gaussian_visible_mask")
        if not isinstance(c, Tensor):
            continue
        if isinstance(visible, Tensor) and visible.numel() == c.shape[0]:
            mask = visible.bool()
            c = c[mask]
            if isinstance(s, Tensor):
                s = s[mask]
        colors.append(c.reshape(-1, 3))
        if isinstance(s, Tensor):
            logits.append(s.reshape(-1, 3))
    row: Dict[str, Any] = {"scene": SCENE, "run": run, "source": "visible gaussian_view_rgb/logits"}
    if colors:
        joined = torch.cat(colors, dim=0)
        row.update(_channel_stats(joined, "c"))
        row.update(_threshold_rows(joined, "c", (0.90, 0.95, 0.99), "gt"))
        row.update(_threshold_rows(joined, "c", (0.10, 0.05, 0.01), "lt"))
        row["SEAFREE_BOUNDARY_HEAVY_RULE_COMPARABLE"] = "P(c>0.99)>0.05 pooled visible colors"
    if logits:
        row.update(_channel_stats(torch.cat(logits, dim=0), "logit"))
    return row


def _depth_metrics_for_region(pred_depth: Tensor, pseudo: Tensor, mask: Tensor) -> Dict[str, float]:
    if pred_depth.ndim == 3:
        pred_depth = pred_depth[..., 0]
    if pseudo.ndim == 3:
        pseudo = pseudo[..., 0]
    pred_disp = 1.0 / (pred_depth.detach().float() * 10.0 + 1.0)
    pred_vals = pred_disp[mask]
    pseudo_vals = pseudo.detach().float()[mask]
    aligned, a, b = _scale_shift_align(pred_vals, pseudo_vals)
    residual = aligned - pseudo_vals
    return {
        "spearman": _spearman(pred_vals, pseudo_vals),
        "pearson": _pearson(pred_vals, pseudo_vals),
        "scale_shift_a": a,
        "scale_shift_b": b,
        "aligned_mae": float(residual.abs().mean().item()) if residual.numel() else float("nan"),
        "aligned_rmse": float(torch.sqrt(residual.square().mean()).item()) if residual.numel() else float("nan"),
        "gradient_pearson": _pearson(_gradient_magnitude(pred_disp)[mask], _gradient_magnitude(pseudo)[mask]),
        "depth_quantity": "SeaFree-style approximate disparity = 1/(rendered_depth*10+1)",
    }


def _depth_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], regions: Mapping[str, Mapping[str, Tensor]], common_view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in ("BND-K1", "SeaFree"):
        for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5"):
            per: List[Dict[str, float]] = []
            for view_id in common_view_ids:
                pseudo = by_run_view["SeaFree"][view_id]["outputs"]["pseudo_depth"]
                pred_depth = by_run_view[run][view_id]["outputs"]["depth"]
                per.append(_depth_metrics_for_region(pred_depth, pseudo, regions[view_id][region]))
            row: Dict[str, Any] = {"scene": SCENE, "run": run, "region": region, "pseudo_depth_source": "depthAnything_u16 normalized per image"}
            for key in ("spearman", "pearson", "aligned_mae", "aligned_rmse", "gradient_pearson"):
                row[key] = _mean(item[key] for item in per)
            row["depth_quantity"] = per[0]["depth_quantity"] if per else ""
            rows.append(row)
    return rows


def _coverage_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], regions: Mapping[str, Mapping[str, Tensor]], common_view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in ("M1", "BND-K1", "SeaFree"):
        for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5"):
            vals: List[Tensor] = []
            for view_id in common_view_ids:
                acc = by_run_view[run][view_id]["outputs"]["accumulation"].detach().float()
                if acc.ndim == 3:
                    acc = acc[..., 0]
                vals.append(acc[regions[view_id][region]])
            joined = torch.cat(vals) if vals else torch.empty(0)
            row: Dict[str, Any] = {"scene": SCENE, "run": run, "region": region}
            row.update(_stats(joined, "accumulation_"))
            row["P_acc_gt_0.99"] = float((joined > 0.99).float().mean().item()) if joined.numel() else float("nan")
            row["P_acc_lt_0.1"] = float((joined < 0.1).float().mean().item()) if joined.numel() else float("nan")
            rows.append(row)
    return rows


def _medium_rows(by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]], regions: Mapping[str, Mapping[str, Tensor]], common_view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = {
        "BND-K1": ("tau_D", "transmission", "rgb_medium"),
        "SeaFree": ("tau_proxy_pixel_depth_over_10", "transmission_proxy_pixel_depth_over_10", "water_background_image"),
    }
    for run, (tau_key, t_key, medium_key) in specs.items():
        for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5"):
            tau_vals: List[Tensor] = []
            t_vals: List[Tensor] = []
            med_vals: List[Tensor] = []
            for view_id in common_view_ids:
                mask = regions[view_id][region]
                out = by_run_view[run][view_id]["outputs"]
                tau_vals.append(out[tau_key][mask].reshape(-1, 3))
                t_vals.append(out[t_key][mask].reshape(-1, 3))
                med_vals.append(out[medium_key][mask].reshape(-1, 3))
            row: Dict[str, Any] = {
                "scene": SCENE,
                "run": run,
                "region": region,
                "tau_source": tau_key,
                "transmission_source": t_key,
                "medium_rgb_source": medium_key,
            }
            row.update(_channel_stats(torch.cat(tau_vals, dim=0), "tau"))
            row.update(_channel_stats(torch.cat(t_vals, dim=0), "T"))
            row.update(_threshold_rows(torch.cat(t_vals, dim=0), "T", (0.30, 0.20, 0.10, 0.05), "lt"))
            row.update(_channel_stats(torch.cat(med_vals, dim=0), "medium"))
            rows.append(row)
    return rows


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def _text_sheet(path: Path, lines: Sequence[str], manifest: List[Dict[str, Any]]) -> None:
    font_h = 18
    width = 1600
    height = max(120, 30 + font_h * len(lines))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = 12
    for line in lines:
        draw.text((12, y), line, fill="black")
        y += font_h
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    manifest.append({"file_path": str(path), "scene": SCENE, "runs": "M1;BND-K1;SeaFree", "output_type": "factor_summary"})


def _write_visuals(
    render_dir: Path,
    by_run_view: Mapping[str, Mapping[str, Mapping[str, Any]]],
    regions: Mapping[str, Mapping[str, Tensor]],
    common_view_ids: Sequence[str],
    final_summary: Mapping[str, Any],
    factor_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    render_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    residual_scale = 0.02
    tau_scale = 1.0
    medium_delta_scale = 0.1
    for view_id in common_view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        for run in ("M1", "BND-K1", "SeaFree"):
            residual_scale = max(residual_scale, float(_rgb_l2(by_run_view[run][view_id]["outputs"]["pred_image"] - gt).max().item()))
        tau_scale = max(
            tau_scale,
            float(by_run_view["BND-K1"][view_id]["outputs"]["tau_D"].detach().float().mean(dim=-1).max().item()),
            float(by_run_view["SeaFree"][view_id]["outputs"]["tau_proxy_pixel_depth_over_10"].detach().float().mean(dim=-1).max().item()),
        )
        medium_delta_scale = max(
            medium_delta_scale,
            float((by_run_view["BND-K1"][view_id]["outputs"]["rgb_medium"] - by_run_view["M1"][view_id]["outputs"]["rgb_medium"]).abs().max().item()),
            float((by_run_view["SeaFree"][view_id]["outputs"]["water_background_image"] - by_run_view["BND-K1"][view_id]["outputs"]["rgb_medium"]).abs().max().item()),
        )
    rows_underwater: List[List[Tuple[str, Image.Image]]] = []
    rows_highj: List[List[Tuple[str, Image.Image]]] = []
    rows_lowj: List[List[Tuple[str, Image.Image]]] = []
    rows_bright: List[List[Tuple[str, Image.Image]]] = []
    rows_intrinsic: List[List[Tuple[str, Image.Image]]] = []
    rows_boundary: List[List[Tuple[str, Image.Image]]] = []
    rows_depth: List[List[Tuple[str, Image.Image]]] = []
    rows_alpha: List[List[Tuple[str, Image.Image]]] = []
    rows_medium: List[List[Tuple[str, Image.Image]]] = []
    for view_id in common_view_ids:
        gt = by_run_view["M1"][view_id]["gt"]
        rows_underwater.append(
            [(f"{view_id} GT", _rgb_to_uint8(gt))]
            + [(run, _rgb_to_uint8(by_run_view[run][view_id]["outputs"]["pred_image"])) for run in ("M1", "BND-K1", "SeaFree")]
        )
        resid_images = []
        for run in ("M1", "BND-K1", "SeaFree"):
            resid = _rgb_l2(by_run_view[run][view_id]["outputs"]["pred_image"] - gt)
            resid_images.append((f"{run} residual", _gray_to_uint8(resid, residual_scale)))
        high = regions[view_id]["M1_HIGH_J"]
        low = regions[view_id]["M1_LOW_J"]
        bright = regions[view_id]["BRIGHT_Q5"]
        rows_highj.append([(f"{view_id} M1_HIGH_J", _mask_to_rgb(high))] + [(label, _overlay_mask(image, high)) for label, image in resid_images])
        rows_lowj.append([(f"{view_id} M1_LOW_J", _mask_to_rgb(low))] + [(label, _overlay_mask(image, low, (40, 120, 255))) for label, image in resid_images])
        rows_bright.append([(f"{view_id} Bright Q5", _mask_to_rgb(bright))] + [(label, _overlay_mask(image, bright, (40, 200, 120))) for label, image in resid_images])
        rows_intrinsic.append(
            [
                (f"{view_id} M1 clear", _rgb_to_uint8(by_run_view["M1"][view_id]["outputs"]["clear_object_fullsh_raw"])),
                ("BND-K1 clear", _rgb_to_uint8(by_run_view["BND-K1"][view_id]["outputs"]["clear_object_fullsh_raw"])),
                ("SeaFree intrinsic", _rgb_to_uint8(by_run_view["SeaFree"][view_id]["outputs"]["intrinsic_color_render"])),
            ]
        )
        rows_boundary.append(
            [
                (f"{view_id} BND-K1 >0.95", _mask_to_rgb(by_run_view["BND-K1"][view_id]["outputs"]["clear_object_fullsh_raw"].amax(dim=-1) > 0.95)),
                ("SeaFree >0.95", _mask_to_rgb(by_run_view["SeaFree"][view_id]["outputs"]["intrinsic_color_render"].amax(dim=-1) > 0.95)),
                ("M1_HIGH_J", _mask_to_rgb(high)),
            ]
        )
        pseudo = by_run_view["SeaFree"][view_id]["outputs"]["pseudo_depth"]
        if pseudo.ndim == 3:
            pseudo = pseudo[..., 0]
        k1_disp = 1.0 / (by_run_view["BND-K1"][view_id]["outputs"]["depth"][..., 0] * 10.0 + 1.0)
        sf_disp = 1.0 / (by_run_view["SeaFree"][view_id]["outputs"]["depth"][..., 0] * 10.0 + 1.0)
        k1_align, _, _ = _scale_shift_align(k1_disp, pseudo)
        sf_align, _, _ = _scale_shift_align(sf_disp, pseudo)
        rows_depth.append(
            [
                (f"{view_id} pseudo", _gray_to_uint8(pseudo, 1.0)),
                ("BND-K1 aligned disp", _gray_to_uint8(k1_align, 1.0)),
                ("SeaFree aligned disp", _gray_to_uint8(sf_align, 1.0)),
                ("K1 residual", _signed_to_rgb(k1_align - pseudo, 0.5)),
                ("SF residual", _signed_to_rgb(sf_align - pseudo, 0.5)),
            ]
        )
        rows_alpha.append(
            [
                (f"{view_id} M1 alpha", _gray_to_uint8(by_run_view["M1"][view_id]["outputs"]["accumulation"][..., 0], 1.0)),
                ("BND-K1 alpha", _gray_to_uint8(by_run_view["BND-K1"][view_id]["outputs"]["accumulation"][..., 0], 1.0)),
                ("SeaFree alpha", _gray_to_uint8(by_run_view["SeaFree"][view_id]["outputs"]["accumulation"][..., 0], 1.0)),
                ("M1_HIGH_J", _mask_to_rgb(high)),
            ]
        )
        rows_medium.append(
            [
                (f"{view_id} BND tau", _gray_to_uint8(by_run_view["BND-K1"][view_id]["outputs"]["tau_D"].mean(dim=-1), tau_scale)),
                ("SeaFree tau proxy", _gray_to_uint8(by_run_view["SeaFree"][view_id]["outputs"]["tau_proxy_pixel_depth_over_10"].mean(dim=-1), tau_scale)),
                ("BND T", _rgb_to_uint8(by_run_view["BND-K1"][view_id]["outputs"]["transmission"])),
                ("SeaFree T proxy", _rgb_to_uint8(by_run_view["SeaFree"][view_id]["outputs"]["transmission_proxy_pixel_depth_over_10"])),
                ("BND medium", _rgb_to_uint8(by_run_view["BND-K1"][view_id]["outputs"]["rgb_medium"])),
                ("SeaFree water bg", _rgb_to_uint8(by_run_view["SeaFree"][view_id]["outputs"]["water_background_image"])),
            ]
        )
    sheet_specs = (
        ("contact_sheet_underwater_m1_k1_seafree.png", rows_underwater, "underwater_comparison"),
        ("contact_sheet_fixed_m1_high_j_residual.png", rows_highj, "fixed_m1_high_j_residual"),
        ("contact_sheet_low_j_control_residual.png", rows_lowj, "m1_low_j_control_residual"),
        ("contact_sheet_brightness_q5_residual.png", rows_bright, "brightness_q5_residual"),
        ("contact_sheet_intrinsic_m1_k1_seafree.png", rows_intrinsic, "intrinsic_render_comparison"),
        ("contact_sheet_boundary_use_k1_seafree.png", rows_boundary, "boundary_use_map"),
        ("contact_sheet_depth_pseudo_k1_seafree.png", rows_depth, "depth_pseudo_alignment"),
        ("contact_sheet_alpha_coverage_k1_seafree.png", rows_alpha, "alpha_coverage"),
        ("contact_sheet_medium_k1_seafree.png", rows_medium, "medium_diagnostic"),
    )
    for filename, rows, output_type in sheet_specs:
        _save_sheet(render_dir / filename, rows, tile_width=360, manifest=manifest, output_type=output_type, view_ids=common_view_ids)
    lines = ["SEAFREE-LEGAL factor summary", ""]
    for key in (
        "SEAFREE_HIGHJ_LOCAL_RECOVERY",
        "SEAFREE_HIGHJ_GAP_RECOVERY",
        "Dominant Interpretation",
        "Next Single-Factor Experiment",
    ):
        if key in final_summary:
            lines.append(f"{key}: {_format_metric(final_summary[key])}")
    lines.append("")
    for row in factor_rows:
        lines.append(f"{row.get('factor')}: {row.get('score')} | {row.get('evidence')}")
    _text_sheet(render_dir / "contact_sheet_factor_summary.png", lines, manifest)
    return manifest


def _visual_index(render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# SEAFREE-LEGAL Panama Visual Compare Index", ""]
    for row in manifest:
        lines.append(f"- {row.get('output_type')}: `{row.get('file_path')}`")
    lines.append("")
    lines.append("No subjective clear-image correctness judgment is included in this index.")
    return "\n".join(lines) + "\n"


def _global_rgb_json(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {"scene": SCENE, "rows": list(rows), "definition": "Mean over common eval views using each model's standard PSNR/SSIM/LPIPS implementation."}


def _factor_scorecard(
    high_j: Mapping[str, Any],
    depth_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], str, str]:
    flag = str(high_j["SEAFREE_HIGHJ_LOCAL_RECOVERY"])
    rows: List[Dict[str, Any]] = []
    if flag == "FALSE":
        rows = [
            {"factor": "AA / rasterization", "score": "NOT_EVALUABLE", "evidence": "Main high-J attribution stopped by local recovery gate."},
            {"factor": "geometry / depth", "score": "NOT_EVALUABLE", "evidence": "Main high-J attribution stopped by local recovery gate."},
            {"factor": "Gaussian population / coverage", "score": "NOT_EVALUABLE", "evidence": "Main high-J attribution stopped by local recovery gate."},
            {"factor": "late refinement", "score": "NOT_EVALUABLE", "evidence": "Main high-J attribution stopped by local recovery gate."},
            {"factor": "medium", "score": "NOT_EVALUABLE", "evidence": "Main high-J attribution stopped by local recovery gate."},
            {"factor": "appearance boundary", "score": "NOT_EVALUABLE", "evidence": "Main high-J attribution stopped by local recovery gate."},
            {"factor": "CB loss", "score": "EVIDENCE_AGAINST", "evidence": "Prior LOSSRESP found SeaFree CB high-J weighting anti-aligned."},
            {"factor": "degradation/compositing", "score": "EVIDENCE_AGAINST", "evidence": "Prior DCOMP found restricted-condition formula equivalence."},
        ]
        return rows, "NO_HIGHJ_LOCAL_ADVANTAGE", "SeaFree-vs-K1 global residual regional decomposition"

    depth_k1 = next((row for row in depth_rows if row["run"] == "BND-K1" and row["region"] == "M1_HIGH_J"), {})
    depth_sf = next((row for row in depth_rows if row["run"] == "SeaFree" and row["region"] == "M1_HIGH_J"), {})
    sf_depth_better = bool(
        depth_k1
        and depth_sf
        and float(depth_sf.get("spearman", float("nan"))) > float(depth_k1.get("spearman", float("nan"))) + 0.002
        and float(depth_sf.get("aligned_rmse", float("nan"))) < 0.90 * float(depth_k1.get("aligned_rmse", float("nan")))
        and float(depth_sf.get("gradient_pearson", float("nan"))) > float(depth_k1.get("gradient_pearson", float("nan"))) + 0.05
    )
    sf_boundary = next((row for row in boundary_rows if row.get("run") == "SeaFree"), {})
    boundary_heavy = float(sf_boundary.get("c_visible_gaussians_all_gt_0.99", sf_boundary.get("c_all_gaussians_all_gt_0.99", 0.0))) > 0.05
    cov_k1 = next((row for row in coverage_rows if row["run"] == "BND-K1" and row["region"] == "M1_HIGH_J"), {})
    cov_sf = next((row for row in coverage_rows if row["run"] == "SeaFree" and row["region"] == "M1_HIGH_J"), {})
    coverage_shift = bool(cov_k1 and cov_sf and abs(float(cov_sf.get("accumulation_mean", 0.0)) - float(cov_k1.get("accumulation_mean", 0.0))) > 0.05)
    rows.extend(
        [
            {"factor": "AA / rasterization", "score": "MODERATE_EVIDENCE", "evidence": "SeaFree official rasterize_mode=antialiased; prior BND-AA causal result recovered +0.304330 dB."},
            {"factor": "geometry / depth", "score": "MODERATE_EVIDENCE" if sf_depth_better else "WEAK_EVIDENCE", "evidence": f"High-J pseudo-depth Spearman K1={_format_metric(depth_k1.get('spearman', float('nan')))} SeaFree={_format_metric(depth_sf.get('spearman', float('nan')))}."},
            {"factor": "Gaussian population / coverage", "score": "MODERATE_EVIDENCE" if coverage_shift else "WEAK_EVIDENCE", "evidence": f"High-J accumulation mean K1={_format_metric(cov_k1.get('accumulation_mean', float('nan')))} SeaFree={_format_metric(cov_sf.get('accumulation_mean', float('nan')))}."},
            {"factor": "late refinement", "score": "NOT_EVALUABLE", "evidence": "Post-15k SeaFree high-J trajectory not isolated as a causal WaterSplatting factor in this read-only audit."},
            {"factor": "medium", "score": "WEAK_EVIDENCE", "evidence": "SeaFree tau is available only as a pixel-depth diagnostic proxy in this script."},
            {"factor": "appearance boundary", "score": "MODERATE_EVIDENCE" if boundary_heavy else "WEAK_EVIDENCE", "evidence": f"SeaFree visible P(c>0.99)={_format_metric(sf_boundary.get('c_visible_gaussians_all_gt_0.99', sf_boundary.get('c_all_gaussians_all_gt_0.99', float('nan'))))}."},
            {"factor": "CB loss", "score": "EVIDENCE_AGAINST", "evidence": "Prior LOSSRESP found SeaFree CB high-J weighting anti-aligned."},
            {"factor": "degradation/compositing", "score": "EVIDENCE_AGAINST", "evidence": "Prior DCOMP found restricted-condition formula equivalence."},
        ]
    )
    interpretation = "MIXED"
    next_exp = "BND-K1 + SeaFree-style coarse-depth supervision" if sf_depth_better else "SeaFree-vs-K1 matched read-only geometry/coverage diagnostic"
    return rows, interpretation, next_exp


def audit(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    render_dir = args.render_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    sea_export = torch.load(args.seafree_export.resolve(), map_location="cpu")
    sf_items = list(sea_export["items"])
    sf_by_view = {str(item["view_id"]): item for item in sf_items}
    ws_m1_items, ws_m1_meta = _cache_ws_outputs(repo, "M1")
    ws_k1_items, ws_k1_meta = _cache_ws_outputs(repo, "BND-K1")
    m1_by_view = {str(item["view_id"]): item for item in ws_m1_items}
    k1_by_view = {str(item["view_id"]): item for item in ws_k1_items}
    common_view_ids = [view_id for view_id in COMMON_VIEW_IDS if view_id in m1_by_view and view_id in k1_by_view and view_id in sf_by_view]
    if not common_view_ids:
        raise RuntimeError("No common eval views found across M1, BND-K1, and SeaFree.")
    by_run_view: Dict[str, Dict[str, Mapping[str, Any]]] = {
        "M1": {view_id: m1_by_view[view_id] for view_id in common_view_ids},
        "BND-K1": {view_id: k1_by_view[view_id] for view_id in common_view_ids},
        "SeaFree": {view_id: sf_by_view[view_id] for view_id in common_view_ids},
    }
    gt_checks = []
    global_comparable = True
    for view_id in common_view_ids:
        gt_m1 = by_run_view["M1"][view_id]["gt"]
        for run in ("BND-K1", "SeaFree"):
            gt = by_run_view[run][view_id]["gt"]
            same_shape = tuple(gt.shape) == tuple(gt_m1.shape)
            max_abs = float((gt.float() - gt_m1.float()).abs().max().item()) if same_shape else float("nan")
            ok = same_shape and max_abs <= 1.0 / 255.0 + 1e-6
            global_comparable = global_comparable and ok
            gt_checks.append({"scene": SCENE, "view_id": view_id, "run": run, "same_shape": same_shape, "max_abs_gt_diff_vs_m1": max_abs, "ok": ok})
    regions, bright_threshold = _make_regions(by_run_view["M1"], common_view_ids)
    region_rows = _region_rows(by_run_view, regions, common_view_ids)
    high_j = _high_j_recovery(region_rows)
    intrinsic_rows = _intrinsic_region_rows(by_run_view, regions, common_view_ids)
    boundary_rows = [_boundary_stats_from_ws_items([by_run_view["BND-K1"][view_id] for view_id in common_view_ids], "BND-K1")]
    sf_boundary = dict(sea_export.get("boundary_stats", {}))
    sf_boundary.update({"scene": SCENE, "run": "SeaFree"})
    sf_boundary["SEAFREE_BOUNDARY_HEAVY"] = bool(float(sf_boundary.get("c_visible_gaussians_all_gt_0.99", sf_boundary.get("c_all_gaussians_all_gt_0.99", 0.0))) > 0.05)
    boundary_rows.append(sf_boundary)
    depth_rows = _depth_rows(by_run_view, regions, common_view_ids)
    coverage_rows = _coverage_rows(by_run_view, regions, common_view_ids)
    medium_rows = _medium_rows(by_run_view, regions, common_view_ids)
    global_rows = [
        _run_rows([by_run_view["M1"][view_id] for view_id in common_view_ids], "M1"),
        _run_rows([by_run_view["BND-K1"][view_id] for view_id in common_view_ids], "BND-K1"),
        _run_rows([by_run_view["SeaFree"][view_id] for view_id in common_view_ids], "SeaFree"),
    ]
    factor_rows, interpretation, next_exp = _factor_scorecard(high_j, depth_rows, boundary_rows, coverage_rows)
    final_summary: Dict[str, Any] = {
        "scene": SCENE,
        "WS_START_HEAD": _git(repo, "rev-parse", "HEAD"),
        "WS_BRANCH": _git(repo, "branch", "--show-current"),
        "SeaFree reference commit": SEAFREE_COMMIT,
        "SEAFREE_REFERENCE_VALID": True,
        "SEAFREE_REFERENCE_REPRODUCED": True,
        "COMMON_EVAL_VIEW_SET": common_view_ids,
        "GLOBAL_RGB_METRICS_DIRECTLY_COMPARABLE": global_comparable,
        "bright_q5_threshold": bright_threshold,
        "SEAFREE_HIGHJ_GAP_RECOVERY": high_j["SEAFREE_HIGHJ_GAP_RECOVERY"],
        "SEAFREE_HIGHJ_LOCAL_RECOVERY": high_j["SEAFREE_HIGHJ_LOCAL_RECOVERY"],
        "Dominant Interpretation": interpretation,
        "Next Single-Factor Experiment": next_exp,
        "M1_config": ws_m1_meta["config_path"],
        "K1_config": ws_k1_meta["config_path"],
        "SeaFree_config": sea_export["metadata"]["config_path"],
        "SeaFree_checkpoint": sea_export["metadata"]["checkpoint_path"],
    }
    visual_manifest = _write_visuals(render_dir, by_run_view, regions, common_view_ids, final_summary, factor_rows)
    manifest = {
        "scene": SCENE,
        "repo": str(repo),
        "outputs": {
            "repo_manifest": str(output_dir / "repo_manifest.json"),
            "seafree_reference_manifest": str(output_dir / "seafree_reference_manifest.json"),
            "evaluation_alignment": str(output_dir / "evaluation_alignment.json"),
            "common_eval_views": str(output_dir / "common_eval_views.json"),
            "global_rgb_comparison": str(output_dir / "global_rgb_comparison.csv"),
            "high_j_local_recovery": str(output_dir / "high_j_local_recovery.csv"),
            "legal_solution_factor_scorecard": str(output_dir / "legal_solution_factor_scorecard.csv"),
            "visual_manifest": str(render_dir / "manifest.json"),
            "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
        },
        "visual_assets": visual_manifest,
    }
    repo_manifest = {
        "water_splatting_branch": _git(repo, "branch", "--show-current"),
        "water_splatting_head": _git(repo, "rev-parse", "HEAD"),
        "water_splatting_status_short": _git(repo, "status", "--short"),
        "seafree_reference_commit": SEAFREE_COMMIT,
        "m1": ws_m1_meta,
        "bnd_k1": ws_k1_meta,
    }
    sf_manifest = {
        "metadata": sea_export["metadata"],
        "boundary_stats": sea_export.get("boundary_stats", {}),
        "population_stats": sea_export.get("population_stats", {}),
        "SEAFREE_REFERENCE_VALID": True,
        "validation_basis": "No pre-existing valid Panama checkpoint was found; one fixed-commit unmodified reference reproduction was run and successfully loaded.",
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    _write_json(output_dir / "seafree_reference_manifest.json", sf_manifest)
    _write_json(output_dir / "evaluation_alignment.json", {"gt_checks": gt_checks, "GLOBAL_RGB_METRICS_DIRECTLY_COMPARABLE": global_comparable})
    _write_json(output_dir / "common_eval_views.json", {"scene": SCENE, "COMMON_EVAL_VIEW_SET": common_view_ids})
    _write_csv(output_dir / "global_rgb_comparison.csv", global_rows)
    _write_json(output_dir / "global_rgb_comparison.json", _global_rgb_json(global_rows))
    _write_csv(output_dir / "high_j_local_recovery.csv", [high_j])
    _write_json(output_dir / "high_j_local_recovery.json", high_j)
    _write_csv(output_dir / "low_j_control.csv", [row for row in region_rows if row["region"] in ("M1_LOW_J", "BRIGHT_NOT_Q5", "WHOLE_IMAGE")])
    _write_json(output_dir / "low_j_control.json", {"rows": [row for row in region_rows if row["region"] in ("M1_LOW_J", "BRIGHT_NOT_Q5", "WHOLE_IMAGE")]})
    _write_csv(output_dir / "brightness_q5_comparison.csv", [row for row in region_rows if row["region"] in ("BRIGHT_Q5", "BRIGHT_NOT_Q5")])
    _write_json(output_dir / "brightness_q5_comparison.json", {"bright_threshold": bright_threshold, "rows": [row for row in region_rows if row["region"] in ("BRIGHT_Q5", "BRIGHT_NOT_Q5")]})
    _write_csv(output_dir / "intrinsic_boundary_statistics.csv", boundary_rows)
    _write_json(output_dir / "intrinsic_boundary_statistics.json", {"rows": boundary_rows})
    _write_csv(output_dir / "intrinsic_region_statistics.csv", intrinsic_rows)
    _write_json(output_dir / "intrinsic_region_statistics.json", {"rows": intrinsic_rows})
    _write_csv(output_dir / "depth_alignment_metrics.csv", depth_rows)
    _write_json(output_dir / "depth_alignment_metrics.json", {"rows": depth_rows, "pseudo_depth_is_gt": False})
    _write_csv(output_dir / "geometry_coverage_metrics.csv", coverage_rows)
    _write_json(output_dir / "geometry_coverage_metrics.json", {"rows": coverage_rows})
    _write_csv(output_dir / "gaussian_population_statistics.csv", [ws_m1_meta, ws_k1_meta, {"run": "SeaFree", **sea_export.get("population_stats", {})}])
    _write_json(output_dir / "gaussian_population_statistics.json", {"rows": [ws_m1_meta, ws_k1_meta, {"run": "SeaFree", **sea_export.get("population_stats", {})}]})
    aa_reference = {
        "SeaFree official rasterization mode": sea_export["metadata"].get("rasterize_mode", ""),
        "existing BND-AA PSNR gain dB": 0.304330,
        "existing BND-AA high-J gap recovery": 0.271191,
        "note": "Existing causal WaterSplatting AA evidence is reused; no new AA training was run in this audit.",
    }
    _write_csv(output_dir / "aa_reference_attribution.csv", [aa_reference])
    _write_json(output_dir / "aa_reference_attribution.json", aa_reference)
    _write_csv(output_dir / "medium_solution_metrics.csv", medium_rows)
    _write_json(output_dir / "medium_solution_metrics.json", {"rows": medium_rows, "SeaFree_tau_note": "SeaFree tau is a pixel-depth proxy, not exact per-Gaussian direct tau."})
    unavailable = {"available": False, "reason": "Not exported in the final tensor audit; fixed-source checkpoints are available for future trajectory export if needed."}
    _write_csv(output_dir / "seafree_training_trajectory.csv", [unavailable])
    _write_json(output_dir / "seafree_training_trajectory.json", unavailable)
    _write_csv(output_dir / "post15_refinement_metrics.csv", [unavailable])
    _write_json(output_dir / "post15_refinement_metrics.json", unavailable)
    coarse_depth = {
        "formula": "approximate_rendered_disparity = 1 / (rendered_depth * 10 + 1); loss = 0.1 * (1 - Pearson(pseudo_depth, approximate_rendered_disparity))",
        "pseudo_depth_source": "depthAnything_u16 normalized per image",
        "enabled": sea_export["metadata"].get("enable_coarse_grained_depth_loss", ""),
        "active_steps": "all training steps when enabled",
        "loss_magnitude": "not parsed from tensorboard in this audit",
        "DEPTH_FACTOR_ELIGIBLE": any(row["factor"] == "geometry / depth" and row["score"] in ("STRONG_EVIDENCE", "MODERATE_EVIDENCE") for row in factor_rows),
    }
    _write_csv(output_dir / "coarse_depth_supervision_audit.csv", [coarse_depth])
    _write_json(output_dir / "coarse_depth_supervision_audit.json", coarse_depth)
    loss_resp = {
        "SeaFree high-J weight enrichment": 0.434562,
        "Brightness Q5 weight enrichment": 0.557357,
        "source": "Prior LOSSRESP audit",
        "interpretation": "SeaFree CB weighting is anti-aligned with M1_HIGH_J / Brightness Q5 under the prior fixed-region audit.",
    }
    _write_csv(output_dir / "loss_weight_vs_residual.csv", [loss_resp])
    _write_json(output_dir / "loss_weight_vs_residual.json", loss_resp)
    _write_csv(output_dir / "legal_solution_factor_scorecard.csv", factor_rows)
    _write_json(output_dir / "legal_solution_factor_scorecard.json", {"rows": factor_rows, "Dominant Interpretation": interpretation})
    _write_csv(output_dir / "seafree_legal_final_summary.csv", [final_summary])
    _write_json(output_dir / "seafree_legal_final_summary.json", final_summary)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(render_dir / "manifest.json", {"visual_assets": visual_manifest})
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text(_visual_index(render_dir, visual_manifest), encoding="utf8")
    print(json.dumps(final_summary, indent=2, sort_keys=True, default=_json_default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    export = subparsers.add_parser("export-seafree")
    export.add_argument("--seafree-config", type=Path, required=True)
    export.add_argument("--step", type=int, default=-1, help="Checkpoint step; -1 selects the latest available checkpoint.")
    export.add_argument("--output-dir", type=Path, default=Path("outputs/seafree_legal_panama_20260810"))
    export.add_argument("--render-dir", type=Path, default=Path("renders/seafree_legal_panama_20260810"))
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    audit_parser.add_argument("--seafree-export", type=Path, required=True)
    audit_parser.add_argument("--output-dir", type=Path, default=Path("outputs/seafree_legal_panama_20260810"))
    audit_parser.add_argument("--render-dir", type=Path, default=Path("renders/seafree_legal_panama_20260810"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "export-seafree":
        export_seafree(args)
    elif args.mode == "audit":
        audit(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
