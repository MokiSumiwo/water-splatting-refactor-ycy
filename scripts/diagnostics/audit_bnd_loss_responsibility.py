#!/usr/bin/env python
"""Read-only loss-responsibility audit for Panama BND.

This diagnostic performs forward passes and no-step autograd only. It does not
call optimizer.step(), scheduler.step(), densification, pruning, or checkpoint
mutation. Output CSV/JSON/PNG files are intentionally written under ignored
outputs/renders directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw

from nerfstudio.utils.eval_utils import eval_setup


EPS = 1e-12
FINAL_NOMINAL_STEP = 15000
SEAFREE_COMMIT = "7797e97dae831029ac89ae9f37b3c3d69ec2cf6c"
PANAMA_FIXED_EVAL_VIEWS = ("MTN_1539", "MTN_1529", "MTN_1547")
LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722])

SCENE_CONFIGS: Dict[str, Dict[str, str]] = {
    "Curasao": {
        "M1": "outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml",
        "BND": "outputs/dewater_bounded_sh3_scratch_20260808/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000/water-splatting/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_bnd-scratch_g1p00/config.yml",
    },
    "IUI3": {
        "M1": "outputs/gmvc_v3_four_scene_iui3_m1_seed42_15000/water-splatting/gmvc_v3_four_scene_iui3_m1_seed42_15000_20260806_gmvc_four_scene_p30_mhold_15k_m1_bootstrap/config.yml",
        "BND": "outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_iui3_bnd_g1p00/config.yml",
    },
    "Panama": {
        "M1": "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml",
        "BND": "outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/config.yml",
    },
}

REGION_ORDER = (
    "M1_HIGH_J",
    "M1_LOW_J",
    "BRIGHT_Q5",
    "BRIGHT_NOT_Q5",
    "BOTTOM20",
    "EDGE_TOP20",
    "LOW_TRANSMISSION",
)

PARAM_GROUP_ORDER = (
    "features_dc",
    "features_rest",
    "means",
    "scales",
    "opacities",
    "quats",
    "medium_mlp",
    "direction_encoding",
)


@dataclass
class LoadedRun:
    scene: str
    run: str
    nominal_step: int
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
    if isinstance(value, torch.Tensor):
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


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def _safe_quantile(values: torch.Tensor, q: float) -> float:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return float("nan")
    return float(torch.quantile(flat, q).item())


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = LUMA_WEIGHTS.to(device=rgb.device, dtype=rgb.dtype)
    return (rgb.detach().float() * weights).sum(dim=-1)


def _gradient_magnitude_luma(image: torch.Tensor) -> torch.Tensor:
    lum = _luma(image).float()
    dx = torch.zeros_like(lum)
    dy = torch.zeros_like(lum)
    dx[:, 1:] = lum[:, 1:] - lum[:, :-1]
    dy[1:, :] = lum[1:, :] - lum[:-1, :]
    return torch.sqrt(dx.square() + dy.square() + EPS)


def _rgb_to_uint8(image: torch.Tensor) -> Image.Image:
    arr = (image.detach().float().clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _scalar_to_uint8(values: torch.Tensor, scale: float, clamp_min: float = 0.0) -> Image.Image:
    scale = max(float(scale), 1e-8)
    arr = ((values.detach().float() - clamp_min).clamp_min(0.0) / scale).clamp(0.0, 1.0)
    arr = (arr * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def _mask_to_rgb(mask: torch.Tensor) -> Image.Image:
    arr = (mask.detach().bool().byte().cpu().numpy() * 255)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _overlay_mask(base_scalar: torch.Tensor, mask: torch.Tensor, scale: float, color: Tuple[int, int, int]) -> Image.Image:
    image = _scalar_to_uint8(base_scalar, scale).convert("RGB")
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
    label_height = 28
    out = Image.new("RGB", (image.width, image.height + label_height), "white")
    out.paste(image, (0, label_height))
    ImageDraw.Draw(out).text((6, 7), label, fill="black")
    return out


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], tile_width: int) -> None:
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
    sheet_width = max(row.width for row in rendered_rows)
    sheet_height = sum(row.height for row in rendered_rows) + 6 * max(0, len(rendered_rows) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)


def _available_steps(config_path: Path) -> Dict[int, Path]:
    ckpt_dir = config_path.parent / "nerfstudio_models"
    out: Dict[int, Path] = {}
    if not ckpt_dir.exists():
        return out
    for path in ckpt_dir.glob("step-*.ckpt"):
        try:
            step = int(path.stem.split("-")[1])
        except Exception:
            continue
        out[step] = path
    return out


def _actual_step(config_path: Path, nominal_step: int) -> Optional[int]:
    steps = _available_steps(config_path)
    if nominal_step in steps:
        return nominal_step
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    return None


def _load_run(repo: Path, scene: str, run: str, nominal_step: int = FINAL_NOMINAL_STEP) -> LoadedRun:
    config_path = repo / SCENE_CONFIGS[scene][run]
    actual_step = _actual_step(config_path, nominal_step)
    if actual_step is None:
        raise FileNotFoundError(f"Missing checkpoint for {scene} {run} nominal step {nominal_step}: {config_path}")

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(
        config_path,
        test_mode="test",
        update_config_callback=update_config,
    )
    if run == "BND":
        pipeline.model.config.intrinsic_color_parameterization = "bounded_sh3"
    elif run == "M1":
        pipeline.model.config.intrinsic_color_parameterization = "legacy"
    pipeline.eval()
    return LoadedRun(scene, run, nominal_step, config_path, checkpoint_path, loaded_step, config, pipeline)


def _release_loaded(loaded: Optional[LoadedRun]) -> None:
    if loaded is None:
        return
    try:
        del loaded.pipeline
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _view_records(loaded: LoadedRun) -> List[Tuple[int, str, Any, Mapping[str, Any]]]:
    rows = []
    dataset = loaded.pipeline.datamanager.eval_dataset
    filenames = list(getattr(dataset, "image_filenames", []))
    for eval_index, (camera, batch) in enumerate(loaded.pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = filenames[eval_index] if eval_index < len(filenames) else Path(f"eval_{eval_index}")
        rows.append((eval_index, Path(filename).stem, camera, batch))
    return rows


def _metric_images(model: Any, pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().float().clamp(0.0, 1.0).to(model.device)
    gt = gt.detach().float().clamp(0.0, 1.0).to(model.device)
    pred_nchw = torch.moveaxis(pred, -1, 0)[None, ...]
    gt_nchw = torch.moveaxis(gt, -1, 0)[None, ...]
    mse = float(((pred - gt) ** 2).mean().item())
    return {
        "psnr": float(model.psnr(gt_nchw, pred_nchw).item()),
        "ssim": float(model.ssim(gt_nchw, pred_nchw).item()),
        "lpips": float(model.lpips(gt_nchw, pred_nchw).item()),
        "mse": mse,
    }


def _formal_gt_pred(model: Any, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
    pred = outputs["pred_image"]
    if "mask" in batch:
        mask = model._downscale_if_required(batch["mask"]).to(model.device)
        gt = gt * mask
        pred = pred * mask
    return gt, pred


def _formal_loss_terms(model: Any, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if model.config.main_loss == "l1":
        per_channel = torch.abs(gt - pred)
        recon = per_channel.mean()
    elif model.config.main_loss == "reg_l1":
        per_channel = torch.abs((gt - pred) / (pred.detach() + 1e-3))
        recon = per_channel.mean()
    else:
        per_channel = ((pred - gt) / (pred.detach() + 1e-3)).square()
        recon = per_channel.mean()

    if model.config.ssim_loss != "ssim":
        denom = pred.detach() + 1e-3
        simloss = 1 - model.ssim((gt / denom).permute(2, 0, 1)[None, ...], (pred / denom).permute(2, 0, 1)[None, ...])
    else:
        simloss = 1 - model.ssim(gt.permute(2, 0, 1)[None, ...], pred.permute(2, 0, 1)[None, ...])
    total = (1 - model.config.ssim_lambda) * recon + model.config.ssim_lambda * simloss
    return per_channel, recon, simloss, total


def _object_support(item: Mapping[str, Any]) -> torch.Tensor:
    return item["outputs"]["accumulation"].detach().float()[..., 0].cpu() > 0.01


def _eval_view_no_grad(model: Any, camera: Any, batch: Mapping[str, Any]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        outputs = model.get_outputs_for_camera(camera)
        gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
        metrics = _metric_images(model, outputs["pred_image"], gt)
    return outputs, gt, metrics


def _safe_cpu_outputs(outputs: Mapping[str, torch.Tensor], keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    return {key: outputs[key].detach().float().cpu() for key in keys if key in outputs and isinstance(outputs[key], torch.Tensor)}


def _make_regions(m1_item: Mapping[str, Any], gt: torch.Tensor, bright_threshold: float) -> Dict[str, torch.Tensor]:
    gt_cpu = gt.detach().float().cpu()
    support = _object_support(m1_item)
    clear = m1_item["outputs"]["clear_object_fullsh_raw"].detach().float().cpu()
    jmax = clear.amax(dim=-1)
    h, w = jmax.shape
    yy = torch.linspace(0.0, 1.0, h).reshape(h, 1).expand(h, w)
    edge = _gradient_magnitude_luma(gt_cpu)
    edge_threshold = _safe_quantile(edge, 0.80)
    transmission = m1_item["outputs"].get("transmission")
    if transmission is None:
        low_t = torch.zeros((h, w), dtype=torch.bool)
    else:
        t = transmission.detach().float().cpu()
        if t.ndim == 3 and t.shape[-1] == 3:
            low_t = support & (t.amin(dim=-1) < 0.1)
        else:
            low_t = support & (t.squeeze(-1) < 0.1)
    bright = _luma(gt_cpu) > bright_threshold
    return {
        "M1_HIGH_J": support & (jmax > 1.0),
        "M1_LOW_J": support & (jmax <= 1.0),
        "BRIGHT_Q5": bright,
        "BRIGHT_NOT_Q5": ~bright,
        "BOTTOM20": yy >= 0.8,
        "EDGE_TOP20": edge >= edge_threshold,
        "LOW_TRANSMISSION": low_t,
    }


def _region_acc_init() -> Dict[str, float]:
    return {"pixels": 0.0, "total_pixels": 0.0, "l1": 0.0, "mse": 0.0, "formal": 0.0, "grad": 0.0}


def _region_fraction_rows(
    scene: str,
    view_id: str,
    masks: Mapping[str, torch.Tensor],
    l1_map: torch.Tensor,
    mse_map: torch.Tensor,
    formal_map: torch.Tensor,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    l1_map = l1_map.detach().float().cpu()
    mse_map = mse_map.detach().float().cpu()
    formal_map = formal_map.detach().float().cpu()
    total_pixels = float(l1_map.numel())
    l1_total = float(l1_map.sum().item())
    mse_total = float(mse_map.sum().item())
    formal_total = float(formal_map.sum().item())
    pixel_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, float]] = {}
    for region in REGION_ORDER:
        mask = masks[region].detach().bool().cpu()
        pixel_count = float(mask.sum().item())
        l1_sum = float(l1_map[mask].sum().item()) if pixel_count else 0.0
        mse_sum = float(mse_map[mask].sum().item()) if pixel_count else 0.0
        formal_sum = float(formal_map[mask].sum().item()) if pixel_count else 0.0
        pixel_fraction = pixel_count / max(total_pixels, EPS)
        l1_fraction = l1_sum / max(l1_total, EPS)
        mse_fraction = mse_sum / max(mse_total, EPS)
        formal_fraction = formal_sum / max(formal_total, EPS)
        pixel_rows.append(
            {
                "scene": scene,
                "view_id": view_id,
                "region": region,
                "pixel_count": int(pixel_count),
                "total_pixels": int(total_pixels),
                "pixel_fraction": pixel_fraction,
                "error_l1_sum": l1_sum,
                "error_mse_sum": mse_sum,
                "error_l1_fraction": l1_fraction,
                "error_mse_fraction": mse_fraction,
                "error_enrichment_l1": l1_fraction / max(pixel_fraction, EPS),
                "error_enrichment_mse": mse_fraction / max(pixel_fraction, EPS),
            }
        )
        loss_rows.append(
            {
                "scene": scene,
                "view_id": view_id,
                "region": region,
                "formal_decomposable_term": "reg_l1",
                "formal_weight_in_total": 1.0 - 0.2,
                "loss_sum_unweighted": formal_sum,
                "loss_sum_weighted": formal_sum * (1.0 - 0.2),
                "loss_fraction": formal_fraction,
                "loss_enrichment": formal_fraction / max(pixel_fraction, EPS),
                "pixel_fraction": pixel_fraction,
            }
        )
        totals[region] = {
            "pixels": pixel_count,
            "total_pixels": total_pixels,
            "l1": l1_sum,
            "mse": mse_sum,
            "formal": formal_sum,
            "l1_total": l1_total,
            "mse_total": mse_total,
            "formal_total": formal_total,
        }
    return pixel_rows, loss_rows, totals


def _aggregate_region_rows(
    scene: str,
    accum: Mapping[str, Mapping[str, float]],
    kind: str,
    term: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_pixels = max(next(iter(accum.values()))["total_pixels"], EPS) if accum else EPS
    l1_total = sum(v.get("l1_total", 0.0) for v in accum.values()) / max(len(accum), 1)
    mse_total = sum(v.get("mse_total", 0.0) for v in accum.values()) / max(len(accum), 1)
    formal_total = sum(v.get("formal_total", 0.0) for v in accum.values()) / max(len(accum), 1)
    grad_total = sum(v.get("grad_total", 0.0) for v in accum.values()) / max(len(accum), 1)
    # For aggregate rows the per-region accumulators are already summed across views,
    # and per-view total fields were accumulated separately by the caller.
    for region in REGION_ORDER:
        value = accum[region]
        pixel_fraction = value["pixels"] / max(value["total_pixels"], EPS)
        row = {"scene": scene, "view_id": "ALL", "region": region, "pixel_fraction": pixel_fraction}
        if kind == "pixel_error":
            l1_fraction = value["l1"] / max(value["l1_total"], EPS)
            mse_fraction = value["mse"] / max(value["mse_total"], EPS)
            row.update(
                {
                    "pixel_count": int(value["pixels"]),
                    "total_pixels": int(value["total_pixels"]),
                    "error_l1_sum": value["l1"],
                    "error_mse_sum": value["mse"],
                    "error_l1_fraction": l1_fraction,
                    "error_mse_fraction": mse_fraction,
                    "error_enrichment_l1": l1_fraction / max(pixel_fraction, EPS),
                    "error_enrichment_mse": mse_fraction / max(pixel_fraction, EPS),
                }
            )
        elif kind == "formal_loss":
            frac = value["formal"] / max(value["formal_total"], EPS)
            row.update(
                {
                    "formal_decomposable_term": "reg_l1",
                    "formal_weight_in_total": 0.8,
                    "loss_sum_unweighted": value["formal"],
                    "loss_sum_weighted": 0.8 * value["formal"],
                    "loss_fraction": frac,
                    "loss_enrichment": frac / max(pixel_fraction, EPS),
                }
            )
        elif kind == "image_grad":
            frac = value["grad"] / max(value["grad_total"], EPS)
            row.update(
                {
                    "loss_term": term,
                    "grad_image_sum": value["grad"],
                    "grad_image_total": value["grad_total"],
                    "grad_image_fraction": frac,
                    "grad_image_enrichment": frac / max(pixel_fraction, EPS),
                }
            )
        rows.append(row)
    return rows


def _add_to_accum(accum: MutableMapping[str, Dict[str, float]], totals: Mapping[str, Mapping[str, float]]) -> None:
    for region in REGION_ORDER:
        if region not in accum:
            accum[region] = {
                "pixels": 0.0,
                "total_pixels": 0.0,
                "l1": 0.0,
                "mse": 0.0,
                "formal": 0.0,
                "l1_total": 0.0,
                "mse_total": 0.0,
                "formal_total": 0.0,
            }
        for key in ("pixels", "total_pixels", "l1", "mse", "formal", "l1_total", "mse_total", "formal_total"):
            accum[region][key] += totals[region].get(key, 0.0)


def _image_gradient_rows(
    scene: str,
    view_id: str,
    masks: Mapping[str, torch.Tensor],
    grad_maps: Mapping[str, torch.Tensor],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Dict[str, float]]]]:
    rows: List[Dict[str, Any]] = []
    accum_by_term: Dict[str, Dict[str, Dict[str, float]]] = {}
    for term, grad_map in grad_maps.items():
        grad = grad_map.detach().float().cpu()
        total_pixels = float(grad.numel())
        grad_total = float(grad.sum().item())
        accum_by_term[term] = {}
        for region in REGION_ORDER:
            mask = masks[region].detach().bool().cpu()
            pixel_count = float(mask.sum().item())
            grad_sum = float(grad[mask].sum().item()) if pixel_count else 0.0
            pixel_fraction = pixel_count / max(total_pixels, EPS)
            grad_fraction = grad_sum / max(grad_total, EPS)
            rows.append(
                {
                    "scene": scene,
                    "view_id": view_id,
                    "region": region,
                    "loss_term": term,
                    "gradient_metric": "RGB_L2_norm",
                    "pixel_fraction": pixel_fraction,
                    "grad_image_sum": grad_sum,
                    "grad_image_total": grad_total,
                    "grad_image_fraction": grad_fraction,
                    "grad_image_enrichment": grad_fraction / max(pixel_fraction, EPS),
                }
            )
            accum_by_term[term][region] = {
                "pixels": pixel_count,
                "total_pixels": total_pixels,
                "grad": grad_sum,
                "grad_total": grad_total,
            }
    return rows, accum_by_term


def _add_grad_accum(
    accum: MutableMapping[str, MutableMapping[str, Dict[str, float]]],
    by_term: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> None:
    for term, regions in by_term.items():
        if term not in accum:
            accum[term] = {}
        for region in REGION_ORDER:
            if region not in accum[term]:
                accum[term][region] = {"pixels": 0.0, "total_pixels": 0.0, "grad": 0.0, "grad_total": 0.0}
            for key in ("pixels", "total_pixels", "grad", "grad_total"):
                accum[term][region][key] += regions[region].get(key, 0.0)


def _zero_grad(model: Any) -> None:
    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None


def _grad_map_for_losses(model: Any, camera: Any, batch: Mapping[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    _zero_grad(model)
    outputs = model.get_outputs(camera.to(model.device))
    pred_original = outputs["pred_image"]
    gt, pred = _formal_gt_pred(model, outputs, batch)
    _, recon, simloss, total = _formal_loss_terms(model, pred, gt)
    grads: Dict[str, torch.Tensor] = {}
    for name, loss in (("formal_reg_l1", recon), ("formal_reg_ssim", simloss), ("formal_total", total)):
        grad = torch.autograd.grad(loss, pred_original, retain_graph=True, allow_unused=False)[0]
        grads[name] = torch.linalg.norm(grad.detach().float(), dim=-1).cpu()
    output_cpu = _safe_cpu_outputs(outputs, ("pred_image", "background"))
    del outputs, pred_original, pred, gt, recon, simloss, total
    _zero_grad(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return grads, output_cpu


def _seafree_foreground_mask_from_batch(model: Any, batch: Mapping[str, Any]) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    if "depth_image" not in batch:
        return None, {
            "status": "UNAVAILABLE",
            "reason": "WaterSplatting Panama batch does not contain depth_image; accumulation mask was not substituted.",
            "seafree_mask_alignment_valid": False,
        }
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception as exc:
        return None, {
            "status": "UNAVAILABLE",
            "reason": f"cv2/numpy foreground construction dependency unavailable: {exc}",
            "seafree_mask_alignment_valid": False,
        }
    pseudo_depth = model._downscale_if_required(batch["depth_image"]).to(model.device)
    pseudo_depth = pseudo_depth / pseudo_depth.max()
    pseudo_np = pseudo_depth.squeeze().detach().cpu().numpy()
    threshold = 1e-2
    mask_1e_2_copy = (pseudo_np < threshold).astype(np.uint8) * 255
    _, binary_image = cv2.threshold(mask_1e_2_copy, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    foreground_mask_array = np.zeros_like(binary_image)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(foreground_mask_array, [max_contour], -1, (255,), thickness=cv2.FILLED)
    foreground_mask_array = foreground_mask_array.astype("float32") / 255.0
    foreground_mask_array[foreground_mask_array < threshold] = 0
    foreground_mask_array[foreground_mask_array > 0] = 1
    mask = torch.from_numpy(foreground_mask_array).bool()
    coverage = float(mask.float().mean().item())
    return mask, {
        "status": "OK",
        "definition": "SeaFree pseudo-depth foreground largest-contour mask at normalized depth threshold 1e-2",
        "foreground_coverage": coverage,
        "background_coverage": 1.0 - coverage,
        "seafree_mask_alignment_valid": True,
    }


def _seafree_and_oracle_gradients(
    model: Any,
    camera: Any,
    batch: Mapping[str, Any],
    highj_mask_cpu: torch.Tensor,
    foreground_mask_cpu: Optional[torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    _zero_grad(model)
    outputs = model.get_outputs(camera.to(model.device))
    pred_original = outputs["pred_image"]
    gt, pred = _formal_gt_pred(model, outputs, batch)
    sf_weight = 1.0 / (pred.detach() + 1e-3)
    reference_mode = "FULL_SEAFREE_CB_REFERENCE" if foreground_mask_cpu is not None else "INTENSITY_ONLY_REFERENCE"
    if foreground_mask_cpu is not None:
        fg = foreground_mask_cpu.to(device=pred.device)
        sf_weight = torch.where(fg[..., None], sf_weight, torch.ones_like(sf_weight))
    sf_l1 = torch.abs((gt - pred) * sf_weight).mean()
    sf_weight_chw = sf_weight.permute(2, 0, 1)[None, ...]
    sf_gt = gt.permute(2, 0, 1)[None, ...] * sf_weight_chw
    sf_pred = pred.permute(2, 0, 1)[None, ...] * sf_weight_chw
    sf_dssim = 1 - model.ssim(sf_gt, sf_pred)
    sf_total = 0.8 * sf_l1 + 0.2 * sf_dssim

    highj = highj_mask_cpu.to(device=pred.device)
    oracle_weight = torch.where(highj[..., None], torch.full_like(pred, 2.0), torch.ones_like(pred))
    per_channel, _, simloss, _ = _formal_loss_terms(model, pred, gt)
    oracle_reg_l1 = (per_channel * oracle_weight).mean()
    oracle_total = 0.8 * oracle_reg_l1 + 0.2 * simloss

    sf_grad = torch.autograd.grad(sf_total, pred_original, retain_graph=True, allow_unused=False)[0]
    oracle_grad = torch.autograd.grad(oracle_total, pred_original, retain_graph=False, allow_unused=False)[0]
    maps = {
        "seafree_weight_scalar": sf_weight.detach().float().mean(dim=-1).cpu(),
        "seafree_total_grad": torch.linalg.norm(sf_grad.detach().float(), dim=-1).cpu(),
        "oracle_weight_scalar": oracle_weight.detach().float().mean(dim=-1).cpu(),
        "oracle_total_grad": torch.linalg.norm(oracle_grad.detach().float(), dim=-1).cpu(),
    }
    meta = {
        "reference_mode": reference_mode,
        "sf_l1": float(sf_l1.detach().item()),
        "sf_dssim": float(sf_dssim.detach().item()),
        "sf_total": float(sf_total.detach().item()),
        "oracle_reg_l1": float(oracle_reg_l1.detach().item()),
        "oracle_total": float(oracle_total.detach().item()),
    }
    del outputs, pred_original, gt, pred, sf_weight, sf_l1, sf_dssim, sf_total, oracle_total
    _zero_grad(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return maps, meta


def _stats(values: torch.Tensor) -> Dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "mean": float(flat.mean().item()),
        "median": _safe_quantile(flat, 0.50),
        "p10": _safe_quantile(flat, 0.10),
        "p90": _safe_quantile(flat, 0.90),
    }


def _weight_alignment_rows(
    scene: str,
    view_id: str,
    masks: Mapping[str, torch.Tensor],
    weight: torch.Tensor,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    all_stats = _stats(weight)
    for region in REGION_ORDER:
        mask = masks[region].bool()
        vals = weight[mask]
        st = _stats(vals)
        rows.append(
            {
                "scene": scene,
                "view_id": view_id,
                "region": region,
                "weight_name": "SeaFree_intensity_weight_scalar",
                "pixel_fraction": float(mask.float().mean().item()),
                "mean_weight": st["mean"],
                "median_weight": st["median"],
                "p10_weight": st["p10"],
                "p90_weight": st["p90"],
                "global_mean_weight": all_stats["mean"],
                "weight_enrichment": st["mean"] / max(all_stats["mean"], EPS),
            }
        )
    return rows


def _counterfactual_rows(
    scene: str,
    view_id: str,
    masks: Mapping[str, torch.Tensor],
    grad_map: torch.Tensor,
    name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grad = grad_map.detach().float().cpu()
    grad_total = float(grad.sum().item())
    total_pixels = float(grad.numel())
    for region in REGION_ORDER:
        mask = masks[region].bool()
        pix = float(mask.sum().item())
        grad_sum = float(grad[mask].sum().item()) if pix else 0.0
        pix_frac = pix / max(total_pixels, EPS)
        grad_frac = grad_sum / max(grad_total, EPS)
        rows.append(
            {
                "scene": scene,
                "view_id": view_id,
                "region": region,
                "counterfactual": name,
                "gradient_metric": "RGB_L2_norm",
                "pixel_fraction": pix_frac,
                "grad_image_fraction": grad_frac,
                "grad_image_enrichment": grad_frac / max(pix_frac, EPS),
            }
        )
    return rows


def _params_by_group(model: Any) -> Dict[str, List[torch.nn.Parameter]]:
    groups: Dict[str, List[torch.nn.Parameter]] = {}
    for name in ("features_dc", "features_rest", "means", "scales", "opacities", "quats"):
        try:
            param = model.gauss_params[name]
        except Exception:
            continue
        groups[name] = [param] if isinstance(param, torch.nn.Parameter) and param.requires_grad else []
    for name in ("medium_mlp", "direction_encoding"):
        params: List[torch.nn.Parameter] = []
        try:
            module = getattr(model, name)
            params = [p for p in module.parameters() if p.requires_grad]
        except Exception:
            params = []
        groups[name] = params
    return groups


def _flatten_grad(params: Sequence[torch.nn.Parameter], grads: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for param, grad in zip(params, grads):
        if grad is None:
            chunks.append(torch.zeros(param.numel(), dtype=torch.float32, device="cpu"))
        else:
            chunks.append(grad.detach().float().reshape(-1).cpu())
    if not chunks:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def _vector_stats(vector: torch.Tensor, total: Optional[torch.Tensor] = None) -> Dict[str, float]:
    if vector.numel() == 0:
        return {"grad_l2": 0.0, "grad_mean_abs": 0.0, "normalized_to_total": 0.0, "cosine_with_total": float("nan")}
    l2 = float(torch.linalg.norm(vector).item())
    out = {"grad_l2": l2, "grad_mean_abs": float(vector.abs().mean().item())}
    if total is not None and total.numel() == vector.numel():
        total_l2 = float(torch.linalg.norm(total).item())
        out["normalized_to_total"] = l2 / max(total_l2, EPS)
        cosine = float(torch.dot(vector, total).item() / max(l2 * total_l2, EPS))
        out["cosine_with_total"] = max(-1.0, min(1.0, cosine))
    else:
        out["normalized_to_total"] = float("nan")
        out["cosine_with_total"] = float("nan")
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return float("nan")
    na = float(torch.linalg.norm(a).item())
    nb = float(torch.linalg.norm(b).item())
    cosine = float(torch.dot(a, b).item() / max(na * nb, EPS))
    return max(-1.0, min(1.0, cosine))


def _snapshot_params(model: Any) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for name in ("features_dc", "features_rest", "means", "opacities"):
        param = model.gauss_params[name]
        snap[name] = param.detach().cpu().clone()
    medium_params: List[torch.Tensor] = []
    for p in model.medium_mlp.parameters():
        medium_params.append(p.detach().cpu().reshape(-1))
    snap["medium_mlp_flat"] = torch.cat(medium_params).clone() if medium_params else torch.empty(0)
    return snap


def _param_delta(model: Any, snap: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in ("features_dc", "features_rest", "means", "opacities"):
        current = model.gauss_params[name].detach().cpu()
        out[name] = float((current - snap[name]).abs().max().item())
    medium_params = [p.detach().cpu().reshape(-1) for p in model.medium_mlp.parameters()]
    current_medium = torch.cat(medium_params) if medium_params else torch.empty(0)
    if current_medium.numel() == snap["medium_mlp_flat"].numel() and current_medium.numel() > 0:
        out["medium_mlp_flat"] = float((current_medium - snap["medium_mlp_flat"]).abs().max().item())
    else:
        out["medium_mlp_flat"] = 0.0
    return out


def _param_grad_audit_for_view(
    model: Any,
    camera: Any,
    batch: Mapping[str, Any],
    masks: Mapping[str, torch.Tensor],
    scene: str,
    view_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    conflict_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    groups = _params_by_group(model)
    flat_params: List[torch.nn.Parameter] = []
    group_slices: Dict[str, Tuple[int, int, List[torch.nn.Parameter]]] = {}
    offset = 0
    for group in PARAM_GROUP_ORDER:
        params = groups.get(group, [])
        if not params:
            continue
        start = offset
        flat_params.extend(params)
        offset += len(params)
        group_slices[group] = (start, offset, params)

    def grads_for_loss(loss: torch.Tensor, retain_graph: bool) -> Dict[str, torch.Tensor]:
        grads = torch.autograd.grad(loss, flat_params, retain_graph=retain_graph, allow_unused=True)
        by_group: Dict[str, torch.Tensor] = {}
        for group, (start, end, params) in group_slices.items():
            by_group[group] = _flatten_grad(params, grads[start:end])
        return by_group

    _zero_grad(model)
    outputs = model.get_outputs(camera.to(model.device))
    gt, pred = _formal_gt_pred(model, outputs, batch)
    per_channel, _, simloss, _ = _formal_loss_terms(model, pred, gt)
    loss_map = per_channel.mean(dim=-1)
    total_loss = loss_map.mean()
    total_grads = grads_for_loss(total_loss, retain_graph=True)

    for region in REGION_ORDER:
        mask = masks[region].to(device=pred.device)
        if int(mask.sum().item()) == 0:
            continue
        natural_loss = (loss_map * mask.float()).sum() / loss_map.numel()
        equal_loss = loss_map[mask].mean()
        for mode, loss in (("natural_mass", natural_loss), ("equal_area_normalized", equal_loss)):
            region_grads = grads_for_loss(loss, retain_graph=True)
            for group in PARAM_GROUP_ORDER:
                vector = region_grads.get(group, torch.empty(0))
                total_vector = total_grads.get(group, torch.empty(0))
                st = _vector_stats(vector, total_vector)
                rows.append(
                    {
                        "scene": scene,
                        "view_id": view_id,
                        "region": region,
                        "parameter_group": group,
                        "loss_term": "formal_reg_l1",
                        "regionalization": mode,
                        **st,
                    }
                )

    conflict_pairs = (("M1_HIGH_J", "M1_LOW_J"), ("BRIGHT_Q5", "BRIGHT_NOT_Q5"))
    for left, right in conflict_pairs:
        left_mask = masks[left].to(device=pred.device)
        right_mask = masks[right].to(device=pred.device)
        if int(left_mask.sum().item()) == 0 or int(right_mask.sum().item()) == 0:
            continue
        left_loss = (loss_map * left_mask.float()).sum() / loss_map.numel()
        right_loss = (loss_map * right_mask.float()).sum() / loss_map.numel()
        left_grads = grads_for_loss(left_loss, retain_graph=True)
        right_grads = grads_for_loss(right_loss, retain_graph=True)
        for group in PARAM_GROUP_ORDER:
            left_vec = left_grads.get(group, torch.empty(0))
            right_vec = right_grads.get(group, torch.empty(0))
            total_vec = total_grads.get(group, torch.empty(0))
            left_norm = float(torch.linalg.norm(left_vec).item()) if left_vec.numel() else 0.0
            right_norm = float(torch.linalg.norm(right_vec).item()) if right_vec.numel() else 0.0
            total_norm = float(torch.linalg.norm(total_vec).item()) if total_vec.numel() else 0.0
            conflict_rows.append(
                {
                    "scene": scene,
                    "view_id": view_id,
                    "region_pair": f"{left}_vs_{right}",
                    "parameter_group": group,
                    "cosine": _cosine(left_vec, right_vec),
                    "left_norm_to_total": left_norm / max(total_norm, EPS),
                    "right_norm_to_total": right_norm / max(total_norm, EPS),
                }
            )

    highj = masks["M1_HIGH_J"].to(device=pred.device)
    oracle_weight = torch.where(highj[..., None], torch.full_like(pred, 2.0), torch.ones_like(pred))
    oracle_loss = 0.8 * (per_channel * oracle_weight).mean() + 0.2 * simloss
    formal_total = 0.8 * total_loss + 0.2 * simloss
    formal_total_grads = grads_for_loss(formal_total, retain_graph=True)
    oracle_grads = grads_for_loss(oracle_loss, retain_graph=False)
    for group in PARAM_GROUP_ORDER:
        formal_vec = formal_total_grads.get(group, torch.empty(0))
        oracle_vec = oracle_grads.get(group, torch.empty(0))
        formal_norm = float(torch.linalg.norm(formal_vec).item()) if formal_vec.numel() else 0.0
        oracle_norm = float(torch.linalg.norm(oracle_vec).item()) if oracle_vec.numel() else 0.0
        oracle_rows.append(
            {
                "scene": scene,
                "view_id": view_id,
                "parameter_group": group,
                "oracle": "M1_HIGH_J_2x",
                "formal_vs_oracle_cosine": _cosine(formal_vec, oracle_vec),
                "oracle_to_formal_magnitude_ratio": oracle_norm / max(formal_norm, EPS),
                "formal_grad_l2": formal_norm,
                "oracle_grad_l2": oracle_norm,
            }
        )
    del outputs, gt, pred, per_channel, simloss, loss_map, total_loss
    _zero_grad(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, conflict_rows, oracle_rows


def _binary_metrics(scores: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    s = scores.detach().float().reshape(-1)
    y = labels.detach().bool().reshape(-1)
    finite = torch.isfinite(s)
    s = s[finite]
    y = y[finite]
    n = y.numel()
    pos = int(y.sum().item())
    neg = int(n - pos)
    if n == 0 or pos == 0 or neg == 0:
        return {
            "base_rate": float(pos / max(n, 1)),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "top20_precision": float("nan"),
            "top20_recall": float("nan"),
            "top20_iou": float("nan"),
            "top20_enrichment": float("nan"),
        }
    order = torch.argsort(s, descending=True)
    y_sorted = y[order].float()
    tp = torch.cumsum(y_sorted, dim=0)
    fp = torch.cumsum(1.0 - y_sorted, dim=0)
    tpr = tp / max(pos, 1)
    fpr = fp / max(neg, 1)
    auroc = float(torch.trapz(torch.cat([torch.zeros(1), tpr]), torch.cat([torch.zeros(1), fpr])).item())
    precision = tp / torch.arange(1, n + 1, dtype=torch.float32)
    recall = tpr
    auprc = float((precision[y_sorted.bool()].sum() / max(pos, 1)).item())
    threshold = torch.quantile(s, 0.80)
    top = s >= threshold
    top_count = int(top.sum().item())
    true_pos = int((top & y).sum().item())
    precision20 = true_pos / max(top_count, 1)
    recall20 = true_pos / max(pos, 1)
    union = int((top | y).sum().item())
    base_rate = pos / max(n, 1)
    return {
        "base_rate": base_rate,
        "auroc": auroc,
        "auprc": auprc,
        "top20_precision": precision20,
        "top20_recall": recall20,
        "top20_iou": true_pos / max(union, 1),
        "top20_enrichment": precision20 / max(base_rate, EPS),
    }


def _proxy_rows(scene: str, items: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    labels: List[torch.Tensor] = []
    scores: Dict[str, List[torch.Tensor]] = {
        "gt_input_brightness": [],
        "current_prediction_brightness": [],
        "current_abs_residual": [],
        "gt_edge_magnitude": [],
    }
    for view_id, item in items.items():
        labels.append(item["regions"]["M1_HIGH_J"].reshape(-1))
        gt = item["gt"].detach().float().cpu()
        pred = item["bnd_outputs"]["pred_image"].detach().float().cpu()
        scores["gt_input_brightness"].append(_luma(gt).reshape(-1))
        scores["current_prediction_brightness"].append(_luma(pred).reshape(-1))
        scores["current_abs_residual"].append((pred - gt).abs().mean(dim=-1).reshape(-1))
        scores["gt_edge_magnitude"].append(_gradient_magnitude_luma(gt).reshape(-1))
    target = torch.cat(labels, dim=0)
    rows = []
    for proxy, parts in scores.items():
        metric = _binary_metrics(torch.cat(parts, dim=0), target)
        rows.append({"scene": scene, "target": "M1_HIGH_J", "proxy": proxy, "availability": "OK", **metric})
    rows.append(
        {
            "scene": scene,
            "target": "M1_HIGH_J",
            "proxy": "pseudo_depth_foreground",
            "availability": "UNAVAILABLE",
            "reason": "No WaterSplatting depth_image pseudo-depth available in Panama batch; accumulation mask was not substituted.",
        }
    )
    return rows


def _make_visuals(render_dir: Path, visual_items: Mapping[str, Mapping[str, Any]], tile_width: int) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []

    def scale_for(key: str) -> float:
        vals = []
        for item in visual_items.values():
            if key in item:
                vals.append(item[key].detach().float().reshape(-1))
        if not vals:
            return 1.0
        flat = torch.cat(vals)
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            return 1.0
        return max(float(torch.quantile(finite, 0.99).item()), 1e-8)

    scales = {
        "abs_residual": scale_for("abs_residual"),
        "squared_residual": scale_for("squared_residual"),
        "formal_loss_map": scale_for("formal_loss_map"),
        "formal_reg_l1": scale_for("formal_reg_l1"),
        "formal_reg_ssim": scale_for("formal_reg_ssim"),
        "formal_total": scale_for("formal_total"),
        "seafree_weight_scalar": scale_for("seafree_weight_scalar"),
        "oracle_weight_scalar": 2.0,
        "oracle_grad_delta": scale_for("oracle_grad_delta"),
        "gt_input_brightness": 1.0,
        "current_prediction_brightness": 1.0,
        "current_abs_residual": scale_for("current_abs_residual"),
        "gt_edge_magnitude": scale_for("gt_edge_magnitude"),
    }

    def save(name: str, rows: Sequence[Sequence[Tuple[str, Image.Image]]], output_type: str) -> None:
        path = render_dir / name
        _save_sheet(path, rows, tile_width)
        manifest.append(
            {
                "file_path": str(path),
                "output_type": output_type,
                "scene": "Panama",
                "view_ids": list(visual_items.keys()),
            }
        )

    rows_failure = []
    rows_loss = []
    rows_grad = []
    rows_overlay = []
    rows_sf = []
    rows_oracle = []
    rows_proxy = []
    for view_id, item in visual_items.items():
        rows_failure.append(
            [
                (f"{view_id} GT", _rgb_to_uint8(item["gt"])),
                ("BND-K1 pred", _rgb_to_uint8(item["bnd_outputs"]["pred_image"])),
                ("abs residual", _scalar_to_uint8(item["abs_residual"], scales["abs_residual"])),
                ("squared residual", _scalar_to_uint8(item["squared_residual"], scales["squared_residual"])),
                ("M1 high-J", _mask_to_rgb(item["regions"]["M1_HIGH_J"])),
                ("Bright Q5", _mask_to_rgb(item["regions"]["BRIGHT_Q5"])),
            ]
        )
        rows_loss.append([(f"{view_id} formal reg_l1 map", _scalar_to_uint8(item["formal_loss_map"], scales["formal_loss_map"]))])
        rows_grad.append(
            [
                (f"{view_id} |dL_l1/dI|", _scalar_to_uint8(item["formal_reg_l1"], scales["formal_reg_l1"])),
                ("|dL_ssim/dI|", _scalar_to_uint8(item["formal_reg_ssim"], scales["formal_reg_ssim"])),
                ("|dL_total/dI|", _scalar_to_uint8(item["formal_total"], scales["formal_total"])),
            ]
        )
        rows_overlay.append(
            [
                (f"{view_id} total grad + high-J", _overlay_mask(item["formal_total"], item["regions"]["M1_HIGH_J"], scales["formal_total"], (255, 40, 40))),
                ("total grad + Bright Q5", _overlay_mask(item["formal_total"], item["regions"]["BRIGHT_Q5"], scales["formal_total"], (40, 120, 255))),
            ]
        )
        rows_sf.append(
            [
                (f"{view_id} SeaFree W", _scalar_to_uint8(item["seafree_weight_scalar"], scales["seafree_weight_scalar"])),
                ("W + high-J", _overlay_mask(item["seafree_weight_scalar"], item["regions"]["M1_HIGH_J"], scales["seafree_weight_scalar"], (255, 40, 40))),
                ("W + Bright Q5", _overlay_mask(item["seafree_weight_scalar"], item["regions"]["BRIGHT_Q5"], scales["seafree_weight_scalar"], (40, 120, 255))),
            ]
        )
        rows_oracle.append(
            [
                (f"{view_id} Oracle 2x W", _scalar_to_uint8(item["oracle_weight_scalar"], scales["oracle_weight_scalar"])),
                ("oracle-formal grad abs delta", _scalar_to_uint8(item["oracle_grad_delta"], scales["oracle_grad_delta"])),
            ]
        )
        rows_proxy.append(
            [
                (f"{view_id} GT brightness", _scalar_to_uint8(item["gt_input_brightness"], scales["gt_input_brightness"])),
                ("prediction brightness", _scalar_to_uint8(item["current_prediction_brightness"], scales["current_prediction_brightness"])),
                ("abs residual", _scalar_to_uint8(item["current_abs_residual"], scales["current_abs_residual"])),
                ("GT edge", _scalar_to_uint8(item["gt_edge_magnitude"], scales["gt_edge_magnitude"])),
                ("high-J mask", _mask_to_rgb(item["regions"]["M1_HIGH_J"])),
            ]
        )
    save("contact_sheet_failure_localization.png", rows_failure, "failure_localization")
    save("contact_sheet_formal_loss_map.png", rows_loss, "formal_decomposable_loss_map")
    save("contact_sheet_formal_image_gradients.png", rows_grad, "formal_image_gradient")
    save("contact_sheet_responsibility_overlay.png", rows_overlay, "responsibility_overlay")
    save("contact_sheet_seafree_weight_alignment.png", rows_sf, "seafree_weight_alignment")
    save("contact_sheet_oracle_2x_diagnostic.png", rows_oracle, "oracle_2x_diagnostic")
    save("contact_sheet_proxy_alignment.png", rows_proxy, "proxy_alignment")
    return manifest


def _make_cross_scene_sheet(render_dir: Path, rows: Sequence[Mapping[str, Any]], tile_width: int, manifest: List[Dict[str, Any]]) -> None:
    scene_rows = []
    for row in rows:
        if row.get("view_id") != "ALL" or row.get("region") != "M1_HIGH_J":
            continue
        text = "\n".join(
            [
                f"{row['scene']} M1_HIGH_J",
                f"pixel {float(row.get('pixel_fraction', float('nan'))):.4f}",
                f"MSE enrich {float(row.get('error_enrichment_mse', float('nan'))):.4f}",
                f"grad enrich {float(row.get('grad_image_enrichment', float('nan'))):.4f}",
                f"ratio {float(row.get('responsibility_ratio', float('nan'))):.4f}",
            ]
        )
        img = Image.new("RGB", (420, 180), "white")
        ImageDraw.Draw(img).multiline_text((12, 12), text, fill="black", spacing=8)
        scene_rows.append([(str(row["scene"]), img)])
    path = render_dir / "contact_sheet_cross_scene_control.png"
    _save_sheet(path, scene_rows, tile_width)
    manifest.append({"file_path": str(path), "output_type": "cross_scene_control", "scenes": ["Panama", "Curasao", "IUI3"]})


def _source_semantics(repo: Path, seafree_repo: Path) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    formal = {
        "water_splatting_code_fact": {
            "file": "water_splatting/water_splatting.py",
            "lines": "1560-1595",
            "gt": "gt_img = composite_with_background(get_gt_img(batch['image']), outputs['background'])",
            "pred": "pred_img = outputs['pred_image']",
            "mask": "If batch['mask'] exists, gt and pred are multiplied by the mask before loss.",
            "main_loss_options": {
                "l1": "abs(gt - pred).mean()",
                "reg_l1": "abs((gt - pred) / (pred.detach() + 1e-3)).mean()",
                "reg_l2": "(((pred - gt) / (pred.detach() + 1e-3)) ** 2).mean()",
            },
            "ssim_loss_options": {
                "reg_ssim": "1 - SSIM(gt/(pred.detach()+1e-3), pred/(pred.detach()+1e-3))",
                "ssim": "1 - SSIM(gt, pred)",
            },
            "total": "(1 - ssim_lambda) * recon_loss + ssim_lambda * simloss",
            "default_config": "water_splatting/water_splatting_config.py sets main_loss=reg_l1, ssim_loss=reg_ssim, ssim_lambda=0.2.",
            "extra_regularizers": "No medium/depth/opacity regularizer is added by get_loss_dict.",
        },
        "formal_target": {
            "scene": "Panama",
            "run": "BND-K1",
            "main_loss": "reg_l1",
            "ssim_loss": "reg_ssim",
            "ssim_lambda": 0.2,
            "pixel_decomposable_term": "reg_l1 only",
            "nonlocal_term": "reg_ssim image/window SSIM term, attributed by image-space gradient only",
        },
    }
    seafree = {
        "seafree_reference_commit": SEAFREE_COMMIT,
        "source_file": "seafree_gs/seafree_model.py",
        "source_lines": "804-950 at fixed commit",
        "content_based_code_fact": {
            "pseudo_depth": "batch['depth_image'] is downscaled, moved to device, then normalized by pseudo_depth.max().",
            "foreground_mask": "normalized pseudo-depth < 1e-2 is thresholded, inverted with cv2.THRESH_BINARY_INV, largest external contour is filled, then binarized.",
            "weight": "foreground_aware_reconstruction_weight = 1 / (rendered_underwater_image.detach() + 1e-3)",
            "background_pixels": "foreground_mask < 0.5 pixels get weight 1.",
            "weighted_l1": "abs((GT_underwater - rendered_underwater) * weight).mean()",
            "weighted_dssim": "1 - SSIM(GT*weight, rendered*weight), with a four-block split when dimensions exceed 800.",
            "content_based_loss": "(1-ssim_lambda)*weighted_l1 + ssim_lambda*weighted_dssim + 0.01*background_water_supervision_loss",
            "coarse_depth_loss": "0.1 * (1 - pearson_corrcoef(pseudo_depth, 1/(rendered_depth*10+1))) when enabled.",
        },
        "counterfactual_scope_in_this_audit": {
            "included": "foreground/intensity Content-Based reconstruction weighting only",
            "not_included": ["background_water_supervision_loss", "coarse_grained_depth_loss", "scale_regularization_loss"],
        },
    }
    equation = """# Formal WaterSplatting Loss Equation

Code fact from `water_splatting/water_splatting.py:get_loss_dict`.

Let `I` be `outputs["pred_image"]` after any training mask, and `G` be the
background-composited underwater ground truth after the same mask.

For the formal M1/BND config:

```text
L_reg_l1 = mean_{x,c} |(G[x,c] - I[x,c]) / (stopgrad(I[x,c]) + 1e-3)|

L_reg_ssim = 1 - SSIM(
    G / (stopgrad(I) + 1e-3),
    I / (stopgrad(I) + 1e-3)
)

L_main = 0.8 * L_reg_l1 + 0.2 * L_reg_ssim
```

`L_reg_l1` is the only strictly pixel/channel-decomposable term used in this
audit. `L_reg_ssim` is not treated as independent per-pixel loss; spatial
responsibility is measured through `dL/dI_pred`.
"""
    return formal, seafree, equation


def _lookup(rows: Sequence[Mapping[str, Any]], scene: str, view_id: str, region: str, key: str, term: Optional[str] = None) -> float:
    for row in rows:
        if row.get("scene") == scene and row.get("view_id") == view_id and row.get("region") == region:
            if term is None or row.get("loss_term") == term:
                try:
                    return float(row[key])
                except Exception:
                    return float("nan")
    return float("nan")


def _classifications(
    pixel_rows: Sequence[Mapping[str, Any]],
    grad_rows: Sequence[Mapping[str, Any]],
    conflict_rows: Sequence[Mapping[str, Any]],
    seafree_weight_rows: Sequence[Mapping[str, Any]],
    sf_resp_rows: Sequence[Mapping[str, Any]],
    formal_resp_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    proxy_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    highj_mse_enrich = _lookup(pixel_rows, "Panama", "ALL", "M1_HIGH_J", "error_enrichment_mse")
    highj_error_share = _lookup(pixel_rows, "Panama", "ALL", "M1_HIGH_J", "error_mse_fraction")
    highj_grad_share = _lookup(grad_rows, "Panama", "ALL", "M1_HIGH_J", "grad_image_fraction", term="formal_total")
    highj_grad_enrich = _lookup(grad_rows, "Panama", "ALL", "M1_HIGH_J", "grad_image_enrichment", term="formal_total")
    resp_ratio = highj_grad_share / max(highj_error_share, EPS)
    error_localized = highj_mse_enrich >= 2.0
    under = error_localized and resp_ratio <= 0.60 and highj_grad_enrich < highj_mse_enrich

    conflict = False
    for row in conflict_rows:
        if row.get("scene") == "Panama" and row.get("region_pair") == "M1_HIGH_J_vs_M1_LOW_J" and row.get("parameter_group") in ("features_dc", "features_rest"):
            cos = float(row.get("cosine", float("nan")))
            left = float(row.get("left_norm_to_total", 0.0))
            right = float(row.get("right_norm_to_total", 0.0))
            if cos <= -0.20 and left >= 0.01 and right >= 0.01:
                conflict = True

    def weight_enrichment(region: str) -> float:
        vals = [float(r["weight_enrichment"]) for r in seafree_weight_rows if r.get("scene") == "Panama" and r.get("view_id") == "ALL" and r.get("region") == region]
        return vals[0] if vals else float("nan")

    sf_highj_enrich = weight_enrichment("M1_HIGH_J")
    sf_bright_enrich = weight_enrichment("BRIGHT_Q5")
    sf_highj_down = sf_highj_enrich <= 0.90
    sf_bright_down = sf_bright_enrich <= 0.90

    def grad_share(rows: Sequence[Mapping[str, Any]], region: str, counterfactual: Optional[str] = None) -> float:
        for row in rows:
            if row.get("scene") == "Panama" and row.get("view_id") == "ALL" and row.get("region") == region:
                if counterfactual is None or row.get("counterfactual") == counterfactual:
                    return float(row.get("grad_image_fraction", float("nan")))
        return float("nan")

    formal_highj = grad_share(formal_resp_rows, "M1_HIGH_J")
    formal_bright = grad_share(formal_resp_rows, "BRIGHT_Q5")
    sf_highj = grad_share(sf_resp_rows, "M1_HIGH_J", "SeaFree_CB_intensity_reference")
    sf_bright = grad_share(sf_resp_rows, "BRIGHT_Q5", "SeaFree_CB_intensity_reference")
    sf_highj_rel = sf_highj / max(formal_highj, EPS) - 1.0
    sf_bright_rel = sf_bright / max(formal_bright, EPS) - 1.0
    seafree_aligned = sf_highj_rel >= 0.25 or sf_bright_rel >= 0.25
    seafree_anti = (sf_highj_down and sf_highj < formal_highj) or (sf_bright_down and sf_bright < formal_bright)

    deployable = False
    for row in proxy_rows:
        if row.get("availability") == "OK" and row.get("proxy") != "current_abs_residual":
            try:
                if float(row["auprc"]) > float(row["base_rate"]) * 1.5 and float(row["top20_enrichment"]) >= 2.0:
                    deployable = True
            except Exception:
                pass

    oracle_direction_change = False
    for row in oracle_rows:
        if row.get("scene") == "Panama" and row.get("view_id") == "ALL" and row.get("parameter_group") in ("features_dc", "features_rest"):
            try:
                if float(row.get("formal_vs_oracle_cosine", 1.0)) < 0.98 or abs(float(row.get("oracle_to_formal_magnitude_ratio", 1.0)) - 1.0) > 0.10:
                    oracle_direction_change = True
            except Exception:
                pass

    if error_localized and under and not conflict and oracle_direction_change:
        overall = "SUPPORTED"
    elif error_localized and (under or highj_grad_enrich < highj_mse_enrich):
        overall = "PARTIALLY_SUPPORTED"
    elif error_localized and (resp_ratio >= 0.9 or conflict):
        overall = "NOT_SUPPORTED"
    else:
        overall = "UNRESOLVED"

    if seafree_anti:
        sf_class = "ANTI_ALIGNED"
    elif seafree_aligned:
        sf_class = "SUPPORTED"
    elif math.isfinite(sf_highj_enrich) or math.isfinite(sf_bright_enrich):
        sf_class = "NOT_SUPPORTED"
    else:
        sf_class = "UNRESOLVED"

    flags = {
        "ERROR_LOCALIZED": error_localized,
        "FAILURE_REGION_UNDER_EMPHASIZED": under,
        "HIGHJ_GRADIENT_CONFLICT": conflict,
        "SEAFREE_CB_ALIGNED": seafree_aligned,
        "SEAFREE_CB_ANTI_ALIGNED": seafree_anti,
        "SEAFREE_HIGHJ_ALIGNED": sf_highj_enrich >= 1.25,
        "SEAFREE_BRIGHT_ALIGNED": sf_bright_enrich >= 1.25,
        "SEAFREE_HIGHJ_DOWNWEIGHTS": sf_highj_down,
        "SEAFREE_BRIGHT_DOWNWEIGHTS": sf_bright_down,
        "DEPLOYABLE_PROXY_EXISTS": deployable,
        "OVERALL_HYPOTHESIS": overall,
        "SEAFREE_SPECIFIC_HYPOTHESIS": sf_class,
        "highj_error_enrichment_mse": highj_mse_enrich,
        "highj_error_share_mse": highj_error_share,
        "highj_total_grad_share": highj_grad_share,
        "highj_total_grad_enrichment": highj_grad_enrich,
        "highj_responsibility_ratio": resp_ratio,
        "seafree_highj_weight_enrichment": sf_highj_enrich,
        "seafree_bright_weight_enrichment": sf_bright_enrich,
        "seafree_highj_grad_share_relative_change": sf_highj_rel,
        "seafree_bright_grad_share_relative_change": sf_bright_rel,
        "oracle_direction_or_magnitude_change": oracle_direction_change,
    }
    summary_rows = [{"key": key, "value": value} for key, value in flags.items()]
    return summary_rows, flags


def _aggregate_weight_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = list(rows)
    for region in REGION_ORDER:
        region_rows = [r for r in rows if r.get("scene") == "Panama" and r.get("region") == region]
        if not region_rows:
            continue
        pix = [float(r["pixel_fraction"]) for r in region_rows]
        mean_w = [float(r["mean_weight"]) for r in region_rows]
        global_w = [float(r["global_mean_weight"]) for r in region_rows]
        out.append(
            {
                "scene": "Panama",
                "view_id": "ALL",
                "region": region,
                "weight_name": "SeaFree_intensity_weight_scalar",
                "pixel_fraction": sum(pix) / len(pix),
                "mean_weight": sum(mean_w) / len(mean_w),
                "median_weight": "",
                "p10_weight": "",
                "p90_weight": "",
                "global_mean_weight": sum(global_w) / len(global_w),
                "weight_enrichment": (sum(mean_w) / len(mean_w)) / max(sum(global_w) / len(global_w), EPS),
            }
        )
    return out


def _aggregate_counterfactual_rows(rows: Sequence[Mapping[str, Any]], name: str) -> List[Dict[str, Any]]:
    out = list(rows)
    for region in REGION_ORDER:
        selected = [r for r in rows if r.get("scene") == "Panama" and r.get("region") == region and r.get("counterfactual") == name]
        if not selected:
            continue
        pix = sum(float(r["pixel_fraction"]) for r in selected) / len(selected)
        grad = sum(float(r["grad_image_fraction"]) for r in selected) / len(selected)
        out.append(
            {
                "scene": "Panama",
                "view_id": "ALL",
                "region": region,
                "counterfactual": name,
                "gradient_metric": "RGB_L2_norm",
                "pixel_fraction": pix,
                "grad_image_fraction": grad,
                "grad_image_enrichment": grad / max(pix, EPS),
            }
        )
    return out


def _aggregate_oracle_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = list(rows)
    for group in PARAM_GROUP_ORDER:
        selected = [r for r in rows if r.get("scene") == "Panama" and r.get("parameter_group") == group]
        if not selected:
            continue
        out.append(
            {
                "scene": "Panama",
                "view_id": "ALL",
                "parameter_group": group,
                "oracle": "M1_HIGH_J_2x",
                "formal_vs_oracle_cosine": sum(float(r["formal_vs_oracle_cosine"]) for r in selected) / len(selected),
                "oracle_to_formal_magnitude_ratio": sum(float(r["oracle_to_formal_magnitude_ratio"]) for r in selected) / len(selected),
                "formal_grad_l2": sum(float(r["formal_grad_l2"]) for r in selected) / len(selected),
                "oracle_grad_l2": sum(float(r["oracle_grad_l2"]) for r in selected) / len(selected),
            }
        )
    return out


def _aggregate_conflict_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = list(rows)
    keys = sorted({(r.get("scene"), r.get("region_pair"), r.get("parameter_group")) for r in rows})
    for scene, pair, group in keys:
        selected = [r for r in rows if r.get("scene") == scene and r.get("region_pair") == pair and r.get("parameter_group") == group]
        vals = [float(r["cosine"]) for r in selected if math.isfinite(float(r["cosine"]))]
        if not selected or not vals:
            continue
        out.append(
            {
                "scene": scene,
                "view_id": "ALL",
                "region_pair": pair,
                "parameter_group": group,
                "cosine": sum(vals) / len(vals),
                "left_norm_to_total": sum(float(r["left_norm_to_total"]) for r in selected) / len(selected),
                "right_norm_to_total": sum(float(r["right_norm_to_total"]) for r in selected) / len(selected),
            }
        )
    return out


def _panama_cross_scene_rows(
    pixel_rows: Sequence[Mapping[str, Any]],
    image_grad_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for region in REGION_ORDER:
        prow = next(
            (
                r
                for r in pixel_rows
                if r.get("scene") == "Panama" and r.get("view_id") == "ALL" and r.get("region") == region
            ),
            {},
        )
        grow = next(
            (
                r
                for r in image_grad_rows
                if r.get("scene") == "Panama"
                and r.get("view_id") == "ALL"
                and r.get("region") == region
                and r.get("loss_term") == "formal_total"
            ),
            {},
        )
        if not prow or not grow:
            continue
        err_share = float(prow.get("error_mse_fraction", float("nan")))
        grad_share = float(grow.get("grad_image_fraction", float("nan")))
        rows.append(
            {
                "scene": "Panama",
                "view_id": "ALL",
                "region": region,
                "pixel_fraction": prow.get("pixel_fraction"),
                "error_mse_fraction": err_share,
                "error_enrichment_mse": prow.get("error_enrichment_mse"),
                "grad_image_fraction": grad_share,
                "grad_image_enrichment": grow.get("grad_image_enrichment"),
                "responsibility_ratio": grad_share / max(err_share, EPS),
            }
        )
    return rows


def _write_visual_index(render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# LOSSRESP Visual Compare Index", ""]
    for row in manifest:
        lines.append(f"- {row.get('output_type')}: `{row.get('file_path')}`")
    lines.append("")
    lines.append("No subjective visual-quality conclusion is made in this index.")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_research_note(path: Path, flags: Mapping[str, Any], repo_manifest: Mapping[str, Any], visual_manifest: Sequence[Mapping[str, Any]]) -> None:
    def v(key: str) -> Any:
        return flags.get(key, "NA")

    lines = [
        "# BND Loss Responsibility Alignment Audit - 2026-08-10",
        "",
        "## Motivation",
        "",
        "HYPOTHESIS: The remaining Panama BND RGB gap may be materially limited by spatial loss-responsibility allocation in localized legacy-high-J / bright regions.",
        "",
        "## Code Fact - Repository",
        "",
        f"- WaterSplatting branch: `{repo_manifest.get('branch')}`",
        f"- Start HEAD: `{repo_manifest.get('start_head')}`",
        f"- SeaFree reference commit: `{repo_manifest.get('seafree_reference_commit')}`",
        "",
        "## Code Fact - Formal WaterSplatting Loss",
        "",
        "- Formal BND/M1 loss uses `main_loss=reg_l1`, `ssim_loss=reg_ssim`, `ssim_lambda=0.2`.",
        "- `L_reg_l1 = mean(abs((GT - pred) / (stopgrad(pred) + 1e-3)))`.",
        "- `L_reg_ssim = 1 - SSIM(GT/(stopgrad(pred)+1e-3), pred/(stopgrad(pred)+1e-3))`.",
        "- `L_main = 0.8 * L_reg_l1 + 0.2 * L_reg_ssim`.",
        "- `reg_l1` is pixel/channel decomposable; `reg_ssim` is attributed by image-space gradients, not by a fake per-pixel SSIM map.",
        "",
        "## Code Fact - SeaFree CB Loss",
        "",
        "- SeaFree builds a foreground mask from normalized `depth_image` with threshold `1e-2`, inverse thresholding, largest contour fill, and binarization.",
        "- SeaFree content weighting is `1 / (rendered_underwater_image.detach() + 1e-3)` on foreground pixels and `1` on background pixels.",
        "- This audit includes only the Content-Based reconstruction weighting counterfactual. SeaFree background-water supervision and coarse depth loss are recorded but not mixed into the CB alignment result.",
        "- Panama WaterSplatting batches did not provide SeaFree-compatible `depth_image`; accumulation was not substituted. SeaFree maps are therefore marked intensity-only reference.",
        "",
        "## Experimental Fact - Region Definitions",
        "",
        "- `M1_HIGH_J`: formal M1 object support (`accumulation > 0.01`) and `clear_object_fullsh_raw.max_rgb > 1.0`.",
        "- `M1_LOW_J`: formal M1 object support and `clear_object_fullsh_raw.max_rgb <= 1.0`.",
        "- `BRIGHT_Q5`: top 20 percent GT luminance over Panama eval views, matching the prior Q5 convention.",
        "- `BOTTOM20`: image rows with normalized y >= 0.8.",
        "- `EDGE_TOP20`: top 20 percent GT luminance gradient magnitude.",
        "- `LOW_TRANSMISSION`: M1 object support and min-RGB transmission < 0.1.",
        "",
        "## Quantitative Result - Key Flags",
        "",
        f"- ERROR_LOCALIZED: `{v('ERROR_LOCALIZED')}`",
        f"- FAILURE_REGION_UNDER_EMPHASIZED: `{v('FAILURE_REGION_UNDER_EMPHASIZED')}`",
        f"- HIGHJ_GRADIENT_CONFLICT: `{v('HIGHJ_GRADIENT_CONFLICT')}`",
        f"- SEAFREE_CB_ALIGNED: `{v('SEAFREE_CB_ALIGNED')}`",
        f"- SEAFREE_CB_ANTI_ALIGNED: `{v('SEAFREE_CB_ANTI_ALIGNED')}`",
        f"- DEPLOYABLE_PROXY_EXISTS: `{v('DEPLOYABLE_PROXY_EXISTS')}`",
        "",
        "## Quantitative Result - M1_HIGH_J Responsibility",
        "",
        f"- MSE error enrichment: `{v('highj_error_enrichment_mse')}`",
        f"- MSE error share: `{v('highj_error_share_mse')}`",
        f"- formal total image-gradient share: `{v('highj_total_grad_share')}`",
        f"- formal total image-gradient enrichment: `{v('highj_total_grad_enrichment')}`",
        f"- responsibility ratio: `{v('highj_responsibility_ratio')}`",
        "",
        "## Quantitative Result - SeaFree Alignment",
        "",
        f"- M1_HIGH_J SeaFree intensity-weight enrichment: `{v('seafree_highj_weight_enrichment')}`",
        f"- BRIGHT_Q5 SeaFree intensity-weight enrichment: `{v('seafree_bright_weight_enrichment')}`",
        f"- M1_HIGH_J SeaFree gradient-share relative change: `{v('seafree_highj_grad_share_relative_change')}`",
        f"- BRIGHT_Q5 SeaFree gradient-share relative change: `{v('seafree_bright_grad_share_relative_change')}`",
        "",
        "## Quantitative Conclusion",
        "",
        f"- Overall loss-responsibility hypothesis classification: `{v('OVERALL_HYPOTHESIS')}`.",
        f"- SeaFree-specific CB spatial weighting classification: `{v('SEAFREE_SPECIFIC_HYPOTHESIS')}`.",
        "",
        "## Reasonable Inference",
        "",
        "- These classifications are based on error share vs formal loss contribution vs image/parameter gradient responsibility. They are not visual-quality judgments.",
        "",
        "## Proposed Controlled Experiment",
        "",
        "- Do not train from this note alone unless a follow-up explicitly selects one single-factor weighting candidate from the quantitative flags.",
        "",
        "## Visual Assets",
        "",
    ]
    for row in visual_manifest:
        lines.append(f"- {row.get('output_type')}: `{row.get('file_path')}`")
    lines.extend(
        [
            "",
            "Visual assets are ready for external/manual analysis.",
            "No subjective clear-image correctness judgment was made.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def run_panama_primary(repo: Path, output_dir: Path, render_dir: Path, tile_width: int, skip_param_gradients: bool) -> Dict[str, Any]:
    checkpoint_rows: List[Dict[str, Any]] = []
    pixel_rows: List[Dict[str, Any]] = []
    formal_rows: List[Dict[str, Any]] = []
    image_grad_rows: List[Dict[str, Any]] = []
    parameter_rows: List[Dict[str, Any]] = []
    conflict_rows: List[Dict[str, Any]] = []
    oracle_param_rows: List[Dict[str, Any]] = []
    seafree_semantic_rows: List[Dict[str, Any]] = []
    seafree_weight_rows: List[Dict[str, Any]] = []
    seafree_resp_rows: List[Dict[str, Any]] = []
    oracle_image_rows: List[Dict[str, Any]] = []
    responsibility_rows: List[Dict[str, Any]] = []
    pixel_accum: Dict[str, Dict[str, float]] = {}
    grad_accum: Dict[str, Dict[str, Dict[str, float]]] = {}
    sf_resp_base: List[Dict[str, Any]] = []
    oracle_resp_base: List[Dict[str, Any]] = []
    visual_items: Dict[str, Dict[str, Any]] = {}

    m1 = _load_run(repo, "Panama", "M1")
    m1_views = _view_records(m1)
    view_ids = [v for _, v, _, _ in m1_views]
    selected_view_ids = list(view_ids)
    missing_requested = [v for v in PANAMA_FIXED_EVAL_VIEWS if v not in view_ids]
    checkpoint_rows.append(
        {
            "scene": "Panama",
            "run": "M1",
            "nominal_step": m1.nominal_step,
            "loaded_step": m1.loaded_step,
            "config_path": str(m1.config_path),
            "checkpoint_path": str(m1.checkpoint_path),
            "intrinsic_color_parameterization": m1.model.config.intrinsic_color_parameterization,
            "eval_view_ids": ";".join(view_ids),
            "missing_fixed_requested_views": ";".join(missing_requested),
        }
    )
    m1_items: Dict[str, Dict[str, Any]] = {}
    lumas: List[torch.Tensor] = []
    for _, view_id, camera, batch in m1_views:
        outputs, gt, metrics = _eval_view_no_grad(m1.model, camera, batch)
        item = {
            "gt": gt.detach().float().cpu(),
            "metrics": metrics,
            "outputs": _safe_cpu_outputs(outputs, ("pred_image", "clear_object_fullsh_raw", "transmission", "accumulation")),
        }
        m1_items[view_id] = item
        lumas.append(_luma(item["gt"]).reshape(-1))
    bright_threshold = _safe_quantile(torch.cat(lumas, dim=0), 0.80)
    _release_loaded(m1)

    bnd = _load_run(repo, "Panama", "BND")
    bnd_views = _view_records(bnd)
    bnd_by_view = {view_id: (camera, batch) for _, view_id, camera, batch in bnd_views}
    checkpoint_rows.append(
        {
            "scene": "Panama",
            "run": "BND-K1",
            "nominal_step": bnd.nominal_step,
            "loaded_step": bnd.loaded_step,
            "config_path": str(bnd.config_path),
            "checkpoint_path": str(bnd.checkpoint_path),
            "intrinsic_color_parameterization": bnd.model.config.intrinsic_color_parameterization,
            "main_loss": bnd.model.config.main_loss,
            "ssim_loss": bnd.model.config.ssim_loss,
            "ssim_lambda": bnd.model.config.ssim_lambda,
            "eval_view_ids": ";".join([v for _, v, _, _ in bnd_views]),
        }
    )
    if [v for _, v, _, _ in bnd_views] != view_ids:
        raise RuntimeError("Panama M1/BND eval view order mismatch")

    delta_before: Optional[Dict[str, torch.Tensor]] = None
    if not skip_param_gradients:
        delta_before = _snapshot_params(bnd.model)

    seafree_mask_meta: Dict[str, Any] = {}
    for view_id in selected_view_ids:
        camera, batch = bnd_by_view[view_id]
        bnd_outputs, gt, metrics = _eval_view_no_grad(bnd.model, camera, batch)
        bnd_cpu = _safe_cpu_outputs(bnd_outputs, ("pred_image", "background"))
        regions = _make_regions(m1_items[view_id], gt.detach().cpu(), bright_threshold)
        pred = bnd_cpu["pred_image"]
        gt_cpu = gt.detach().float().cpu()
        residual = pred - gt_cpu
        l1_map = residual.abs().mean(dim=-1)
        mse_map = residual.square().mean(dim=-1)
        formal_map = ((gt_cpu - pred) / (pred.detach() + 1e-3)).abs().mean(dim=-1)
        p_rows, l_rows, totals = _region_fraction_rows("Panama", view_id, regions, l1_map, mse_map, formal_map)
        pixel_rows.extend(p_rows)
        formal_rows.extend(l_rows)
        _add_to_accum(pixel_accum, totals)

        grad_maps, _ = _grad_map_for_losses(bnd.model, camera, batch)
        g_rows, g_acc = _image_gradient_rows("Panama", view_id, regions, grad_maps)
        image_grad_rows.extend(g_rows)
        _add_grad_accum(grad_accum, g_acc)

        foreground_mask, sf_meta = _seafree_foreground_mask_from_batch(bnd.model, batch)
        seafree_mask_meta[view_id] = sf_meta
        sf_oracle_maps, sf_oracle_meta = _seafree_and_oracle_gradients(bnd.model, camera, batch, regions["M1_HIGH_J"], foreground_mask)
        seafree_semantic_rows.append({"scene": "Panama", "view_id": view_id, **sf_meta, **sf_oracle_meta})
        seafree_weight_rows.extend(_weight_alignment_rows("Panama", view_id, regions, sf_oracle_maps["seafree_weight_scalar"]))
        sf_rows = _counterfactual_rows("Panama", view_id, regions, sf_oracle_maps["seafree_total_grad"], "SeaFree_CB_intensity_reference")
        or_rows = _counterfactual_rows("Panama", view_id, regions, sf_oracle_maps["oracle_total_grad"], "Oracle_M1_HIGH_J_2x")
        seafree_resp_rows.extend(sf_rows)
        oracle_image_rows.extend(or_rows)
        sf_resp_base.extend(sf_rows)
        oracle_resp_base.extend(or_rows)

        if not skip_param_gradients:
            try:
                pgrad, cgrad, ograd = _param_grad_audit_for_view(bnd.model, camera, batch, regions, "Panama", view_id)
                parameter_rows.extend(pgrad)
                conflict_rows.extend(cgrad)
                oracle_param_rows.extend(ograd)
            except RuntimeError as exc:
                parameter_rows.append(
                    {
                        "scene": "Panama",
                        "view_id": view_id,
                        "status": "FAILED",
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=5),
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        visual_items[view_id] = {
            "gt": gt_cpu,
            "bnd_outputs": bnd_cpu,
            "regions": regions,
            "abs_residual": l1_map,
            "squared_residual": mse_map,
            "formal_loss_map": formal_map,
            "formal_reg_l1": grad_maps["formal_reg_l1"],
            "formal_reg_ssim": grad_maps["formal_reg_ssim"],
            "formal_total": grad_maps["formal_total"],
            "seafree_weight_scalar": sf_oracle_maps["seafree_weight_scalar"],
            "oracle_weight_scalar": sf_oracle_maps["oracle_weight_scalar"],
            "oracle_grad_delta": (sf_oracle_maps["oracle_total_grad"] - grad_maps["formal_total"]).abs(),
            "gt_input_brightness": _luma(gt_cpu),
            "current_prediction_brightness": _luma(pred),
            "current_abs_residual": l1_map,
            "gt_edge_magnitude": _gradient_magnitude_luma(gt_cpu),
        }

    pixel_rows.extend(_aggregate_region_rows("Panama", pixel_accum, "pixel_error"))
    formal_rows.extend(_aggregate_region_rows("Panama", pixel_accum, "formal_loss"))
    for term, term_accum in grad_accum.items():
        image_grad_rows.extend(_aggregate_region_rows("Panama", term_accum, "image_grad", term=term))
    seafree_weight_rows = _aggregate_weight_rows(seafree_weight_rows)
    seafree_resp_rows = _aggregate_counterfactual_rows(seafree_resp_rows, "SeaFree_CB_intensity_reference")
    oracle_image_rows = _aggregate_counterfactual_rows(oracle_image_rows, "Oracle_M1_HIGH_J_2x")
    oracle_param_rows = _aggregate_oracle_rows(oracle_param_rows)
    conflict_rows = _aggregate_conflict_rows(conflict_rows)

    if delta_before is not None:
        delta_after = _param_delta(bnd.model, delta_before)
        checkpoint_rows.append(
            {
                "scene": "Panama",
                "run": "BND-K1",
                "audit": "NO_PARAMETER_DELTA_AUDIT",
                **{f"max_abs_delta_{k}": v for k, v in delta_after.items()},
                "all_zero": all(v == 0.0 for v in delta_after.values()),
            }
        )

    proxy_rows = _proxy_rows("Panama", visual_items)
    visual_manifest = _make_visuals(render_dir, visual_items, tile_width)

    formal_total_rows = [r for r in image_grad_rows if r.get("loss_term") == "formal_total"]
    for row in formal_total_rows:
        if row.get("scene") == "Panama":
            err_share = _lookup(pixel_rows, "Panama", str(row.get("view_id")), str(row.get("region")), "error_mse_fraction")
            responsibility_rows.append(
                {
                    "scene": row.get("scene"),
                    "view_id": row.get("view_id"),
                    "region": row.get("region"),
                    "error_share_mse": err_share,
                    "grad_share_formal_total": row.get("grad_image_fraction"),
                    "responsibility_ratio": float(row.get("grad_image_fraction", 0.0)) / max(err_share, EPS),
                    "pixel_fraction": row.get("pixel_fraction"),
                }
            )

    _release_loaded(bnd)
    return {
        "checkpoint_rows": checkpoint_rows,
        "pixel_rows": pixel_rows,
        "formal_rows": formal_rows,
        "image_grad_rows": image_grad_rows,
        "parameter_rows": parameter_rows,
        "conflict_rows": conflict_rows,
        "oracle_param_rows": oracle_param_rows,
        "seafree_semantic_rows": seafree_semantic_rows,
        "seafree_weight_rows": seafree_weight_rows,
        "seafree_resp_rows": seafree_resp_rows,
        "oracle_image_rows": oracle_image_rows,
        "responsibility_rows": responsibility_rows,
        "proxy_rows": proxy_rows,
        "visual_manifest": visual_manifest,
        "view_ids": selected_view_ids,
        "bright_threshold": bright_threshold,
        "seafree_mask_meta": seafree_mask_meta,
    }


def run_cross_scene_controls(repo: Path, scenes: Sequence[str], panama_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in scenes:
        if scene == "Panama":
            continue
        m1: Optional[LoadedRun] = None
        bnd: Optional[LoadedRun] = None
        try:
            m1 = _load_run(repo, scene, "M1")
            m1_views = _view_records(m1)
            m1_items: Dict[str, Dict[str, Any]] = {}
            lumas: List[torch.Tensor] = []
            for _, view_id, camera, batch in m1_views:
                outputs, gt, _ = _eval_view_no_grad(m1.model, camera, batch)
                item = {
                    "gt": gt.detach().float().cpu(),
                    "outputs": _safe_cpu_outputs(outputs, ("pred_image", "clear_object_fullsh_raw", "transmission", "accumulation")),
                }
                m1_items[view_id] = item
                lumas.append(_luma(item["gt"]).reshape(-1))
            bright_threshold = _safe_quantile(torch.cat(lumas), 0.80)
            _release_loaded(m1)
            m1 = None

            bnd = _load_run(repo, scene, "BND")
            bnd_views = _view_records(bnd)
            pixel_accum: Dict[str, Dict[str, float]] = {}
            grad_accum: Dict[str, Dict[str, Dict[str, float]]] = {}
            for _, view_id, camera, batch in bnd_views:
                if view_id not in m1_items:
                    continue
                outputs, gt, _ = _eval_view_no_grad(bnd.model, camera, batch)
                pred = outputs["pred_image"].detach().float().cpu()
                gt_cpu = gt.detach().float().cpu()
                regions = _make_regions(m1_items[view_id], gt_cpu, bright_threshold)
                residual = pred - gt_cpu
                l1_map = residual.abs().mean(dim=-1)
                mse_map = residual.square().mean(dim=-1)
                formal_map = ((gt_cpu - pred) / (pred.detach() + 1e-3)).abs().mean(dim=-1)
                _, _, totals = _region_fraction_rows(scene, view_id, regions, l1_map, mse_map, formal_map)
                _add_to_accum(pixel_accum, totals)
                grad_maps, _ = _grad_map_for_losses(bnd.model, camera, batch)
                _, g_acc = _image_gradient_rows(scene, view_id, regions, {"formal_total": grad_maps["formal_total"]})
                _add_grad_accum(grad_accum, g_acc)
            pix_all = _aggregate_region_rows(scene, pixel_accum, "pixel_error")
            grad_all = _aggregate_region_rows(scene, grad_accum.get("formal_total", {}), "image_grad", "formal_total")
            for prow in pix_all:
                if prow.get("view_id") != "ALL":
                    continue
                grow = next((r for r in grad_all if r.get("region") == prow.get("region")), {})
                err_share = float(prow.get("error_mse_fraction", float("nan")))
                grad_share = float(grow.get("grad_image_fraction", float("nan")))
                rows.append(
                    {
                        "scene": scene,
                        "view_id": "ALL",
                        "region": prow.get("region"),
                        "pixel_fraction": prow.get("pixel_fraction"),
                        "error_mse_fraction": err_share,
                        "error_enrichment_mse": prow.get("error_enrichment_mse"),
                        "grad_image_fraction": grad_share,
                        "grad_image_enrichment": grow.get("grad_image_enrichment"),
                        "responsibility_ratio": grad_share / max(err_share, EPS),
                    }
                )
        finally:
            _release_loaded(m1)
            _release_loaded(bnd)
    # Add already computed Panama aggregate rows from caller.
    for row in panama_rows:
        if row.get("scene") == "Panama" and row.get("view_id") == "ALL":
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--seafree-repo", type=Path, default=Path("/mnt/new/home_old/ycy/reference_repos/SeaFree-GS"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lossresp_audit_20260810"))
    parser.add_argument("--render-dir", type=Path, default=Path("renders/lossresp_panama_20260810"))
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--skip-parameter-gradients", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    render_dir = (repo / args.render_dir).resolve() if not args.render_dir.is_absolute() else args.render_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "branch": _git(repo, "branch", "--show-current"),
        "start_head": _git(repo, "rev-parse", "HEAD"),
        "log_5": _git(repo, "log", "-5", "--oneline").splitlines(),
        "status_short_start": _git(repo, "status", "--short").splitlines(),
        "seafree_reference_repo": str(args.seafree_repo),
        "seafree_reference_commit": _git(args.seafree_repo, "rev-parse", "HEAD"),
        "seafree_status_short": _git(args.seafree_repo, "status", "--short").splitlines(),
        "fixed_seafree_commit_requested": SEAFREE_COMMIT,
        "read_only_rules": {
            "training": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "densification_or_pruning": False,
            "checkpoint_write": False,
        },
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    formal_semantics, seafree_semantics, equation = _source_semantics(repo, args.seafree_repo)
    _write_json(output_dir / "formal_loss_semantics.json", formal_semantics)
    _write_json(output_dir / "seafree_cb_loss_semantics.json", seafree_semantics)
    (output_dir / "formal_loss_equation.md").write_text(equation, encoding="utf8")

    primary = run_panama_primary(repo, output_dir, render_dir, args.tile_width, args.skip_parameter_gradients)
    panama_cross_rows = _panama_cross_scene_rows(primary["pixel_rows"], primary["image_grad_rows"])
    cross_scene_rows = run_cross_scene_controls(repo, ("Curasao", "IUI3", "Panama"), panama_cross_rows)

    visual_manifest = list(primary["visual_manifest"])
    _make_cross_scene_sheet(render_dir, cross_scene_rows, args.tile_width, visual_manifest)
    _write_visual_index(render_dir, visual_manifest)

    final_summary_rows, flags = _classifications(
        primary["pixel_rows"],
        primary["image_grad_rows"],
        primary["conflict_rows"],
        primary["seafree_weight_rows"],
        primary["seafree_resp_rows"],
        [r for r in primary["image_grad_rows"] if r.get("loss_term") == "formal_total"],
        primary["oracle_param_rows"],
        primary["proxy_rows"],
    )

    outputs: Dict[str, Sequence[Mapping[str, Any]]] = {
        "checkpoint_audit": primary["checkpoint_rows"],
        "failure_region_pixel_error": primary["pixel_rows"],
        "formal_loss_region_contribution": primary["formal_rows"],
        "image_gradient_responsibility": primary["image_grad_rows"],
        "parameter_gradient_responsibility": primary["parameter_rows"],
        "gradient_conflict": primary["conflict_rows"],
        "responsibility_ratio": primary["responsibility_rows"],
        "seafree_weight_alignment": primary["seafree_weight_rows"],
        "seafree_counterfactual_responsibility": primary["seafree_resp_rows"],
        "oracle_2x_gradient_diagnostic": primary["oracle_param_rows"] + primary["oracle_image_rows"],
        "proxy_alignment": primary["proxy_rows"],
        "cross_scene_control": cross_scene_rows,
        "lossresp_final_summary": final_summary_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
        _write_json(output_dir / f"{name}.json", list(rows))
    _write_json(output_dir / "seafree_mask_reference.json", primary["seafree_mask_meta"])

    manifest = {
        "repo_manifest": str(output_dir / "repo_manifest.json"),
        "formal_loss_semantics": str(output_dir / "formal_loss_semantics.json"),
        "formal_loss_equation": str(output_dir / "formal_loss_equation.md"),
        "seafree_cb_loss_semantics": str(output_dir / "seafree_cb_loss_semantics.json"),
        "view_ids": primary["view_ids"],
        "bright_q5_threshold": primary["bright_threshold"],
        "outputs": sorted(str(p) for p in output_dir.rglob("*") if p.is_file()),
        "renders": visual_manifest,
        "flags": flags,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(render_dir / "manifest.json", visual_manifest)
    _write_research_note(
        repo / "research_notes/BND_LOSS_RESPONSIBILITY_ALIGNMENT_AUDIT_2026-08-10.md",
        flags,
        repo_manifest,
        visual_manifest,
    )
    print(json.dumps({"output_dir": str(output_dir), "render_dir": str(render_dir), "flags": flags}, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
