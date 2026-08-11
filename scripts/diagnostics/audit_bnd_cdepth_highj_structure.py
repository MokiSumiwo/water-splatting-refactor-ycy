#!/usr/bin/env python
"""Read-only BND-CDEPTH high-J gain-vs-harm structural signature audit.

This diagnostic loads existing Panama M1, BND-K1, and BND-CDEPTH checkpoints,
reuses the formal M1_HIGH_J and RGB gain definitions, and measures whether
CDEPTH's successful high-J pixels have a stable Gaussian structural signature.
It does not train, mutate checkpoints, or call optimizer/scheduler updates.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import audit_bnd_cdepth_optimization_path as prev
from scripts.diagnostics import audit_seafree_panama_legal_solution as sea


SCENE = "Panama"
FINAL_STEP = 15000
OUTPUT_DIR = Path("outputs/bnd_cdepth_highj_structure_panama_20260811")
RENDER_DIR = Path("renders/bnd_cdepth_highj_structure_panama_20260811")
LOG_DIR = Path("logs/bnd_cdepth_highj_structure_panama_20260811")
RESEARCH_NOTE = Path("research_notes/BND_CDEPTH_HIGHJ_STRUCTURAL_SIGNATURE_2026-08-11.md")
PREVIOUS_OUTPUT_DIR = Path("outputs/bnd_cdepth_mitigation_path_panama_20260811")
WINDOW_RADIUS_PX = 16
EPS = 1e-8
REGIONS = ("M1_HIGH_J", "HJ_GAIN", "HJ_HARM", "HJ_STRONG_GAIN", "HJ_STRONG_HARM", "M1_LOW_J")
QUANTILES = ("Q1_MOST_HARMED", "Q2", "Q3", "Q4", "Q5_MOST_IMPROVED")


@dataclass
class ViewItem:
    eval_index: int
    view_id: str
    gt: Tensor
    outputs: Dict[str, Tensor]
    center_count_map: Optional[Tensor] = None
    local_density_map: Optional[Tensor] = None
    local_radius_map: Optional[Tensor] = None
    visible_radius_values: Optional[Tensor] = None
    local_density_semantics: str = ""
    local_radius_semantics: str = ""


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


def _safe_cpu(value: Tensor) -> Tensor:
    return value.detach().float().cpu()


def _finite_flat(values: Tensor) -> Tensor:
    flat = values.detach().float().reshape(-1)
    return flat[torch.isfinite(flat)]


def _q(values: Tensor, q: float) -> float:
    flat = _finite_flat(values)
    if flat.numel() == 0:
        return float("nan")
    if q <= 0:
        return float(flat.min().item())
    if q >= 1:
        return float(flat.max().item())
    rank = max(1, min(flat.numel(), int(math.ceil(q * flat.numel()))))
    return float(torch.kthvalue(flat, rank).values.item())


def _stats(values: Tensor, prefix: str = "") -> Dict[str, Any]:
    flat = _finite_flat(values)
    keys = ("count", "mean", "p01", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max")
    if flat.numel() == 0:
        return {f"{prefix}{key}": float("nan") for key in keys}
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p01": _q(flat, 0.01),
        f"{prefix}p10": _q(flat, 0.10),
        f"{prefix}p25": _q(flat, 0.25),
        f"{prefix}p50": _q(flat, 0.50),
        f"{prefix}p75": _q(flat, 0.75),
        f"{prefix}p90": _q(flat, 0.90),
        f"{prefix}p95": _q(flat, 0.95),
        f"{prefix}p99": _q(flat, 0.99),
        f"{prefix}max": float(flat.max().item()),
    }


def _masked(values: Tensor, mask: Tensor) -> Tensor:
    vals = values.detach().float()
    m = mask.detach().bool()
    while m.ndim < vals.ndim:
        m = m[..., None].expand(*vals.shape)
    return vals[m]


def _spearman(xs: Tensor, ys: Tensor) -> float:
    x = _finite_flat(xs)
    y = _finite_flat(ys)
    n = min(x.numel(), y.numel())
    if n < 3:
        return float("nan")
    x = x[:n]
    y = y[:n]
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 3:
        return float("nan")
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.linalg.norm(rx) * torch.linalg.norm(ry)
    if float(denom.item()) <= EPS:
        return float("nan")
    return float((rx * ry).sum().item() / float(denom.item()))


def _pearson(xs: Tensor, ys: Tensor) -> float:
    x = xs.detach().float().reshape(-1)
    y = ys.detach().float().reshape(-1)
    n = min(x.numel(), y.numel())
    if n < 3:
        return float("nan")
    x = x[:n]
    y = y[:n]
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if float(denom.item()) <= EPS:
        return float("nan")
    return float((x * y).sum().item() / float(denom.item()))


def _rgb_to_uint8(image: Tensor) -> Image.Image:
    arr = (torch.nan_to_num(image.detach().float(), nan=0.0).clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _gray_to_uint8(values: Tensor, scale: float, clamp_min: float = 0.0) -> Image.Image:
    vals = torch.nan_to_num(values.detach().float(), nan=0.0, posinf=scale, neginf=clamp_min)
    scale = max(float(scale), EPS)
    arr = ((vals - clamp_min).clamp_min(0.0) / scale).clamp(0.0, 1.0)
    arr = (arr * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr, mode="L").convert("RGB")


def _signed_to_rgb(values: Tensor, scale: float) -> Image.Image:
    vals = torch.nan_to_num(values.detach().float(), nan=0.0)
    scale = max(float(scale), EPS)
    vals = (vals / scale).clamp(-1.0, 1.0)
    pos = vals.clamp_min(0.0)
    neg = (-vals).clamp_min(0.0)
    rgb = torch.stack([pos, torch.zeros_like(vals), neg], dim=-1)
    return _rgb_to_uint8(rgb)


def _mask_to_rgb(mask: Tensor, color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    arr = mask.detach().bool().cpu().numpy()
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[arr] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(out, mode="RGB")


def _label_tile(image: Image.Image, label: str, tile_width: int = 250) -> Image.Image:
    ratio = tile_width / max(image.width, 1)
    resized = image.resize((tile_width, max(1, int(round(image.height * ratio)))), Image.BILINEAR)
    label_h = 28
    canvas = Image.new("RGB", (tile_width, resized.height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), label, fill=(0, 0, 0))
    canvas.paste(resized, (0, label_h))
    return canvas


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], manifest: List[Dict[str, Any]], output_type: str, view_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_label_tile(img, label) for label, img in row]
        canvas = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), max(t.height for t in tiles)), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered_rows.append(canvas)
    if not rendered_rows:
        return
    sheet = Image.new("RGB", (max(row.width for row in rendered_rows), sum(row.height for row in rendered_rows) + 6 * (len(rendered_rows) - 1)), "white")
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)
    manifest.append(
        {
            "scene": SCENE,
            "file_path": str(path),
            "output_type": output_type,
            "view_ids": ";".join(view_ids),
            "width": sheet.width,
            "height": sheet.height,
        }
    )


def _text_sheet(path: Path, title: str, lines: Sequence[str], manifest: List[Dict[str, Any]], output_type: str, view_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_lines = [title, ""] + list(lines)
    width = 1900
    height = max(180, 30 * len(all_lines) + 24)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for idx, line in enumerate(all_lines):
        draw.text((10, 12 + idx * 30), line, fill=(0, 0, 0))
    img.save(path)
    manifest.append({"scene": SCENE, "file_path": str(path), "output_type": output_type, "view_ids": ";".join(view_ids), "width": width, "height": height})


def _box_sum(image: Tensor, radius: int) -> Tensor:
    img = image.detach().float().cpu()
    h, w = img.shape
    integral = torch.zeros((h + 1, w + 1), dtype=img.dtype)
    integral[1:, 1:] = img.cumsum(dim=0).cumsum(dim=1)
    ys = torch.arange(h, dtype=torch.long)
    xs = torch.arange(w, dtype=torch.long)
    y0 = (ys - radius).clamp(0, h)
    y1 = (ys + radius + 1).clamp(0, h)
    x0 = (xs - radius).clamp(0, w)
    x1 = (xs + radius + 1).clamp(0, w)
    return integral[y1[:, None], x1[None, :]] - integral[y0[:, None], x1[None, :]] - integral[y1[:, None], x0[None, :]] + integral[y0[:, None], x0[None, :]]


def _projected_center_proxy_maps(model: Any, height: int, width: int, window_radius_px: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    xys = getattr(model, "xys", None)
    radii = getattr(model, "radii", None)
    if not isinstance(xys, Tensor) or not isinstance(radii, Tensor):
        raise RuntimeError("Projected xys/radii are not available after forward pass")
    xy = xys.detach().float().reshape(-1, 2).cpu()
    rad = radii.detach().float().reshape(-1).cpu()
    valid = (
        torch.isfinite(xy).all(dim=-1)
        & torch.isfinite(rad)
        & (rad > 0)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    xy = xy[valid]
    rad = rad[valid]
    count = torch.zeros((height, width), dtype=torch.float32)
    radius_sum = torch.zeros((height, width), dtype=torch.float32)
    if xy.numel():
        xi = xy[:, 0].round().long().clamp(0, width - 1)
        yi = xy[:, 1].round().long().clamp(0, height - 1)
        count.index_put_((yi, xi), torch.ones_like(rad), accumulate=True)
        radius_sum.index_put_((yi, xi), rad, accumulate=True)
    local_count = _box_sum(count, window_radius_px)
    local_radius_sum = _box_sum(radius_sum, window_radius_px)
    local_radius = torch.where(local_count > 0, local_radius_sum / local_count.clamp_min(1.0), torch.full_like(local_count, float("nan")))
    return count, local_count, local_radius, rad


def _load_final_run(repo: Path, run: str) -> Tuple[Dict[str, ViewItem], List[Dict[str, Any]], str]:
    loaded = None
    param_delta_rows: List[Dict[str, Any]] = []
    try:
        loaded = prev._load_run(repo, run, FINAL_STEP)
        model = loaded.pipeline.model
        model.eval()
        before = prev._parameter_snapshot(model)
        items: Dict[str, ViewItem] = {}
        for eval_index, view_id, camera, batch in prev._view_records(loaded):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
                gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
            pred = _safe_cpu(outputs["pred_image"])
            height, width = pred.shape[:2]
            center_count, local_density, local_radius, visible_radii = _projected_center_proxy_maps(model, height, width, WINDOW_RADIUS_PX)
            keep = (
                "pred_image",
                "direct_object_signal",
                "rgb_medium",
                "depth",
                "accumulation",
                "tau_D",
                "transmission",
                "clear_object_fullsh_raw",
            )
            items[view_id] = ViewItem(
                eval_index=eval_index,
                view_id=view_id,
                gt=_safe_cpu(gt),
                outputs={key: _safe_cpu(outputs[key]) for key in keep if key in outputs},
                center_count_map=center_count,
                local_density_map=local_density,
                local_radius_map=local_radius,
                visible_radius_values=visible_radii,
                local_density_semantics=f"projected Gaussian center count in a square window of radius {WINDOW_RADIUS_PX}px; proxy, not true contributor count",
                local_radius_semantics=f"mean projected Gaussian radius of centers inside a square window of radius {WINDOW_RADIUS_PX}px; proxy, not contribution-weighted footprint",
            )
        param_delta_rows = prev._parameter_delta(before, model, run, FINAL_STEP)
        return items, param_delta_rows, str(loaded.checkpoint_path)
    finally:
        prev._release(loaded)


def _support(item: ViewItem) -> Tensor:
    return item.outputs["accumulation"][..., 0] > 0.01


def _error(pred: Tensor, gt: Tensor) -> Tensor:
    return (pred.detach().float() - gt.detach().float()).square().mean(dim=-1)


def _region_size_rows(regions: Mapping[str, Mapping[str, Tensor]], view_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    totals = {name: 0 for name in list(REGIONS) + list(QUANTILES)}
    total_pixels = 0
    for view_id in view_ids:
        total = next(iter(regions[view_id].values())).numel()
        total_pixels += total
        for name, mask in regions[view_id].items():
            pixels = int(mask.sum().item())
            totals[name] = totals.get(name, 0) + pixels
            rows.append({"scene": SCENE, "view_id": view_id, "region": name, "pixels": pixels, "pixel_fraction": pixels / max(total, 1)})
    for name, pixels in totals.items():
        rows.append({"scene": SCENE, "view_id": "POOLED", "region": name, "pixels": pixels, "pixel_fraction": pixels / max(total_pixels, 1)})
    return rows


def _build_regions(
    m1_items: Mapping[str, ViewItem],
    k1_items: Mapping[str, ViewItem],
    cd_items: Mapping[str, ViewItem],
    view_ids: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Tensor], Dict[str, Any]]:
    high_gain_values: List[Tensor] = []
    high_positive: List[Tensor] = []
    high_negative_abs: List[Tensor] = []
    gain_maps: Dict[str, Tensor] = {}
    high_masks: Dict[str, Tensor] = {}
    low_masks: Dict[str, Tensor] = {}
    for view_id in view_ids:
        m1 = m1_items[view_id]
        gain = _error(k1_items[view_id].outputs["pred_image"], m1.gt) - _error(cd_items[view_id].outputs["pred_image"], m1.gt)
        gain_maps[view_id] = gain
        high = _support(m1) & (m1.outputs["clear_object_fullsh_raw"].amax(dim=-1) > 1.0)
        low = _support(m1) & ~high
        high_masks[view_id] = high
        low_masks[view_id] = low
        vals = gain[high]
        high_gain_values.append(vals)
        pos = vals[vals > 0]
        neg = -vals[vals < 0]
        if pos.numel():
            high_positive.append(pos)
        if neg.numel():
            high_negative_abs.append(neg)
    pooled_gain = torch.cat(high_gain_values) if high_gain_values else torch.empty(0)
    pos_values = torch.cat(high_positive) if high_positive else torch.empty(0)
    neg_values = torch.cat(high_negative_abs) if high_negative_abs else torch.empty(0)
    strong_gain_threshold = _q(pos_values, 0.75) if pos_values.numel() else float("inf")
    strong_harm_threshold = _q(neg_values, 0.75) if neg_values.numel() else float("inf")
    q20 = _q(pooled_gain, 0.20)
    q40 = _q(pooled_gain, 0.40)
    q60 = _q(pooled_gain, 0.60)
    q80 = _q(pooled_gain, 0.80)

    regions: Dict[str, Dict[str, Tensor]] = {}
    for view_id in view_ids:
        high = high_masks[view_id]
        low = low_masks[view_id]
        gain = gain_maps[view_id]
        regions[view_id] = {
            "M1_HIGH_J": high,
            "M1_LOW_J": low,
            "HJ_GAIN": high & (gain > 0),
            "HJ_HARM": high & (gain < 0),
            "HJ_STRONG_GAIN": high & (gain >= strong_gain_threshold),
            "HJ_STRONG_HARM": high & ((-gain) >= strong_harm_threshold),
            "Q1_MOST_HARMED": high & (gain <= q20),
            "Q2": high & (gain > q20) & (gain <= q40),
            "Q3": high & (gain > q40) & (gain <= q60),
            "Q4": high & (gain > q60) & (gain <= q80),
            "Q5_MOST_IMPROVED": high & (gain > q80),
        }
    meta = {
        "M1_HIGH_J_definition": "M1 object support (M1 accumulation > 0.01) and M1 clear_object_fullsh_raw max RGB channel > 1.0",
        "GAIN_definition": "mean_RGB((pred_K1-GT)^2) - mean_RGB((pred_CDEPTH-GT)^2); positive means CDEPTH improved RGB MSE",
        "HJ_GAIN_definition": "M1_HIGH_J and GAIN > 0",
        "HJ_HARM_definition": "M1_HIGH_J and GAIN < 0",
        "HJ_STRONG_GAIN_definition": "M1_HIGH_J positive GAIN values in the top 25 percent, using pooled threshold",
        "HJ_STRONG_HARM_definition": "M1_HIGH_J negative GAIN magnitude in the top 25 percent, using pooled threshold",
        "strong_gain_threshold": strong_gain_threshold,
        "strong_harm_threshold": strong_harm_threshold,
        "GAIN_highj_quantiles": {"q20": q20, "q40": q40, "q60": q60, "q80": q80},
        "GAIN_abs_quantiles_highj": {
            "p50": _q(pooled_gain.abs(), 0.50),
            "p75": _q(pooled_gain.abs(), 0.75),
            "p90": _q(pooled_gain.abs(), 0.90),
            "p95": _q(pooled_gain.abs(), 0.95),
            "p99": _q(pooled_gain.abs(), 0.99),
        },
    }
    return regions, gain_maps, meta


def _pooled_region_values(
    items: Mapping[str, ViewItem],
    regions: Mapping[str, Mapping[str, Tensor]],
    view_ids: Sequence[str],
    region: str,
    value_getter: Any,
) -> Tensor:
    vals = []
    for view_id in view_ids:
        mask = regions[view_id][region]
        value = value_getter(items[view_id], view_id)
        if value is None:
            continue
        selected = _masked(value, mask)
        if selected.numel():
            vals.append(selected.reshape(-1))
    return torch.cat(vals) if vals else torch.empty(0)


def _region_metric_rows(
    run_items: Mapping[str, Mapping[str, ViewItem]],
    regions: Mapping[str, Mapping[str, Tensor]],
    view_ids: Sequence[str],
    metric_name: str,
    value_getter: Any,
    regions_to_use: Sequence[str] = REGIONS,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run, items in run_items.items():
        for view_id in list(view_ids) + ["POOLED"]:
            for region in regions_to_use:
                if view_id == "POOLED":
                    vals = _pooled_region_values(items, regions, view_ids, region, value_getter)
                else:
                    vals = _masked(value_getter(items[view_id], view_id), regions[view_id][region])
                row = {"scene": SCENE, "run": run, "view_id": view_id, "region": region, "metric": metric_name}
                row.update(_stats(vals))
                rows.append(row)
    return rows


def _delta_region_rows(
    k1_items: Mapping[str, ViewItem],
    cd_items: Mapping[str, ViewItem],
    regions: Mapping[str, Mapping[str, Tensor]],
    view_ids: Sequence[str],
    metric_name: str,
    k1_getter: Any,
    cd_getter: Any,
    regions_to_use: Sequence[str] = REGIONS,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id in list(view_ids) + ["POOLED"]:
        for region in regions_to_use:
            vals: List[Tensor] = []
            loop_ids = view_ids if view_id == "POOLED" else [view_id]
            for vid in loop_ids:
                delta = cd_getter(cd_items[vid], vid) - k1_getter(k1_items[vid], vid)
                selected = _masked(delta, regions[vid][region])
                if selected.numel():
                    vals.append(selected.reshape(-1))
            joined = torch.cat(vals) if vals else torch.empty(0)
            row = {"scene": SCENE, "run": "CDEPTH_MINUS_K1", "view_id": view_id, "region": region, "metric": metric_name}
            row.update(_stats(joined))
            rows.append(row)
    return rows


def _ratio_summary(rows: Sequence[Mapping[str, Any]], metric_name: str, run: str) -> Dict[str, Any]:
    keyed = {(row["run"], row["view_id"], row["region"], row["metric"]): row for row in rows}
    def val(region: str, stat: str = "p50") -> float:
        row = keyed.get((run, "POOLED", region, metric_name), {})
        try:
            return float(row.get(stat, float("nan")))
        except Exception:
            return float("nan")
    gain = val("HJ_GAIN")
    harm = val("HJ_HARM")
    sgain = val("HJ_STRONG_GAIN")
    sharm = val("HJ_STRONG_HARM")
    return {
        f"{run}_{metric_name}_HJ_GAIN_median": gain,
        f"{run}_{metric_name}_HJ_HARM_median": harm,
        f"{run}_{metric_name}_HJ_GAIN_vs_HARM_RATIO": gain / max(abs(harm), EPS),
        f"{run}_{metric_name}_HJ_STRONG_GAIN_median": sgain,
        f"{run}_{metric_name}_HJ_STRONG_HARM_median": sharm,
        f"{run}_{metric_name}_HJ_STRONG_GAIN_vs_HARM_RATIO": sgain / max(abs(sharm), EPS),
    }


def _alpha_rows(
    k1_items: Mapping[str, ViewItem],
    cd_items: Mapping[str, ViewItem],
    regions: Mapping[str, Mapping[str, Tensor]],
    view_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    run_items = {"BND-K1": k1_items, "CDEPTH": cd_items}
    rows = _region_metric_rows(run_items, regions, view_ids, "alpha_accumulation", lambda item, _vid: item.outputs["accumulation"][..., 0])
    rows.extend(
        _delta_region_rows(
            k1_items,
            cd_items,
            regions,
            view_ids,
            "delta_alpha_accumulation",
            lambda item, _vid: item.outputs["accumulation"][..., 0],
            lambda item, _vid: item.outputs["accumulation"][..., 0],
        )
    )
    return rows


def _depth_tau_control_rows(
    k1_items: Mapping[str, ViewItem],
    cd_items: Mapping[str, ViewItem],
    regions: Mapping[str, Mapping[str, Tensor]],
    view_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run, items in (("BND-K1", k1_items), ("CDEPTH", cd_items)):
        for view_id in list(view_ids) + ["POOLED"]:
            loop_ids = view_ids if view_id == "POOLED" else [view_id]
            for region in REGIONS:
                depth_vals: List[Tensor] = []
                tau_vals: List[Tensor] = []
                t_vals: List[Tensor] = []
                new_low_vals: List[Tensor] = []
                for vid in loop_ids:
                    mask = regions[vid][region]
                    item = items[vid]
                    depth_vals.append(_masked(item.outputs["depth"][..., 0], mask).reshape(-1))
                    tau_vals.append(_masked(item.outputs["tau_D"].mean(dim=-1), mask).reshape(-1))
                    t_mean = item.outputs["transmission"].mean(dim=-1)
                    t_vals.append(_masked(t_mean, mask).reshape(-1))
                    if run == "CDEPTH":
                        k_t = k1_items[vid].outputs["transmission"].mean(dim=-1)
                        new_low = (t_mean < 0.1) & (k_t >= 0.1)
                    else:
                        new_low = t_mean < 0.1
                    new_low_vals.append(_masked(new_low.float(), mask).reshape(-1))
                depth = torch.cat([v for v in depth_vals if v.numel()]) if any(v.numel() for v in depth_vals) else torch.empty(0)
                tau = torch.cat([v for v in tau_vals if v.numel()]) if any(v.numel() for v in tau_vals) else torch.empty(0)
                t = torch.cat([v for v in t_vals if v.numel()]) if any(v.numel() for v in t_vals) else torch.empty(0)
                nl = torch.cat([v for v in new_low_vals if v.numel()]) if any(v.numel() for v in new_low_vals) else torch.empty(0)
                row = {
                    "scene": SCENE,
                    "run": run,
                    "view_id": view_id,
                    "region": region,
                    "depth_p50": _q(depth, 0.50),
                    "depth_p90": _q(depth, 0.90),
                    "tau_p50": _q(tau, 0.50),
                    "tau_p90": _q(tau, 0.90),
                    "tau_p99": _q(tau, 0.99),
                    "T_p01": _q(t, 0.01),
                    "T_p10": _q(t, 0.10),
                    "T_p50": _q(t, 0.50),
                    "P_T_lt_0.1_or_new_low_t": float(nl.mean().item()) if nl.numel() else float("nan"),
                }
                rows.append(row)
    return rows


def _direct_medium_rows(
    k1_items: Mapping[str, ViewItem],
    cd_items: Mapping[str, ViewItem],
    regions: Mapping[str, Mapping[str, Tensor]],
    gain_maps: Mapping[str, Tensor],
    view_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id in list(view_ids) + ["POOLED"]:
        loop_ids = view_ids if view_id == "POOLED" else [view_id]
        for region in REGIONS:
            direct_vals: List[Tensor] = []
            medium_vals: List[Tensor] = []
            pred_vals: List[Tensor] = []
            err_reduction: List[Tensor] = []
            for vid in loop_ids:
                mask = regions[vid][region]
                direct = (cd_items[vid].outputs["direct_object_signal"] - k1_items[vid].outputs["direct_object_signal"]).abs().mean(dim=-1)
                medium = (cd_items[vid].outputs["rgb_medium"] - k1_items[vid].outputs["rgb_medium"]).abs().mean(dim=-1)
                pred = (cd_items[vid].outputs["pred_image"] - k1_items[vid].outputs["pred_image"]).abs().mean(dim=-1)
                direct_vals.append(_masked(direct, mask).reshape(-1))
                medium_vals.append(_masked(medium, mask).reshape(-1))
                pred_vals.append(_masked(pred, mask).reshape(-1))
                err_reduction.append(_masked(gain_maps[vid], mask).reshape(-1))
            direct_j = torch.cat([v for v in direct_vals if v.numel()]) if any(v.numel() for v in direct_vals) else torch.empty(0)
            medium_j = torch.cat([v for v in medium_vals if v.numel()]) if any(v.numel() for v in medium_vals) else torch.empty(0)
            pred_j = torch.cat([v for v in pred_vals if v.numel()]) if any(v.numel() for v in pred_vals) else torch.empty(0)
            err_j = torch.cat([v for v in err_reduction if v.numel()]) if any(v.numel() for v in err_reduction) else torch.empty(0)
            rows.append(
                {
                    "scene": SCENE,
                    "view_id": view_id,
                    "region": region,
                    "direct_delta_abs_mean": float(direct_j.mean().item()) if direct_j.numel() else float("nan"),
                    "direct_delta_abs_p50": _q(direct_j, 0.50),
                    "direct_delta_abs_p90": _q(direct_j, 0.90),
                    "medium_delta_abs_mean": float(medium_j.mean().item()) if medium_j.numel() else float("nan"),
                    "medium_delta_abs_p50": _q(medium_j, 0.50),
                    "medium_delta_abs_p90": _q(medium_j, 0.90),
                    "pred_delta_abs_mean": float(pred_j.mean().item()) if pred_j.numel() else float("nan"),
                    "pred_delta_abs_p50": _q(pred_j, 0.50),
                    "pred_delta_abs_p90": _q(pred_j, 0.90),
                    "error_reduction_mean": float(err_j.mean().item()) if err_j.numel() else float("nan"),
                    "error_reduction_p50": _q(err_j, 0.50),
                    "direct_to_medium_delta_ratio": float(direct_j.mean().item() / max(float(medium_j.mean().item()), EPS)) if direct_j.numel() and medium_j.numel() else float("nan"),
                    "definition": "true renderer direct_object_signal and rgb_medium outputs; no J*T image approximation",
                }
            )
    return rows


def _gain_quantile_rows(
    k1_items: Mapping[str, ViewItem],
    cd_items: Mapping[str, ViewItem],
    regions: Mapping[str, Mapping[str, Tensor]],
    gain_maps: Mapping[str, Tensor],
    view_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for view_id in list(view_ids) + ["POOLED"]:
        loop_ids = view_ids if view_id == "POOLED" else [view_id]
        for quantile in QUANTILES:
            vals: Dict[str, List[Tensor]] = {
                "gain": [],
                "K1_radius": [],
                "CDEPTH_radius": [],
                "delta_radius": [],
                "K1_density": [],
                "CDEPTH_density": [],
                "delta_density": [],
                "K1_alpha": [],
                "CDEPTH_alpha": [],
                "delta_alpha": [],
                "K1_tau": [],
                "CDEPTH_tau": [],
                "delta_tau": [],
                "K1_T": [],
                "CDEPTH_T": [],
                "delta_T": [],
                "CDEPTH_depth": [],
            }
            for vid in loop_ids:
                mask = regions[vid][quantile]
                k = k1_items[vid]
                c = cd_items[vid]
                vals["gain"].append(_masked(gain_maps[vid], mask).reshape(-1))
                vals["K1_radius"].append(_masked(k.local_radius_map, mask).reshape(-1))
                vals["CDEPTH_radius"].append(_masked(c.local_radius_map, mask).reshape(-1))
                vals["delta_radius"].append(_masked(c.local_radius_map - k.local_radius_map, mask).reshape(-1))
                vals["K1_density"].append(_masked(k.local_density_map, mask).reshape(-1))
                vals["CDEPTH_density"].append(_masked(c.local_density_map, mask).reshape(-1))
                vals["delta_density"].append(_masked(c.local_density_map - k.local_density_map, mask).reshape(-1))
                ka = k.outputs["accumulation"][..., 0]
                ca = c.outputs["accumulation"][..., 0]
                vals["K1_alpha"].append(_masked(ka, mask).reshape(-1))
                vals["CDEPTH_alpha"].append(_masked(ca, mask).reshape(-1))
                vals["delta_alpha"].append(_masked(ca - ka, mask).reshape(-1))
                ktau = k.outputs["tau_D"].mean(dim=-1)
                ctau = c.outputs["tau_D"].mean(dim=-1)
                kt = k.outputs["transmission"].mean(dim=-1)
                ct = c.outputs["transmission"].mean(dim=-1)
                vals["K1_tau"].append(_masked(ktau, mask).reshape(-1))
                vals["CDEPTH_tau"].append(_masked(ctau, mask).reshape(-1))
                vals["delta_tau"].append(_masked(ctau - ktau, mask).reshape(-1))
                vals["K1_T"].append(_masked(kt, mask).reshape(-1))
                vals["CDEPTH_T"].append(_masked(ct, mask).reshape(-1))
                vals["delta_T"].append(_masked(ct - kt, mask).reshape(-1))
                vals["CDEPTH_depth"].append(_masked(c.outputs["depth"][..., 0], mask).reshape(-1))
            row = {"scene": SCENE, "view_id": view_id, "gain_quantile": quantile}
            for key, parts in vals.items():
                joined = torch.cat([v for v in parts if v.numel()]) if any(v.numel() for v in parts) else torch.empty(0)
                row[f"{key}_p50"] = _q(joined, 0.50)
                row[f"{key}_p75"] = _q(joined, 0.75)
                row[f"{key}_p90"] = _q(joined, 0.90)
                if key == "gain":
                    row[f"{key}_mean"] = float(joined.mean().item()) if joined.numel() else float("nan")
            rows.append(row)
    return rows


def _spatial_correlation_rows(
    k1_items: Mapping[str, ViewItem],
    cd_items: Mapping[str, ViewItem],
    regions: Mapping[str, Mapping[str, Tensor]],
    gain_maps: Mapping[str, Tensor],
    view_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metric_getters = {
        "delta_projected_center_window_radius": lambda k, c: c.local_radius_map - k.local_radius_map,
        "CDEPTH_projected_center_window_radius": lambda _k, c: c.local_radius_map,
        "delta_projected_center_window_density": lambda k, c: c.local_density_map - k.local_density_map,
        "CDEPTH_projected_center_window_density": lambda _k, c: c.local_density_map,
        "delta_alpha": lambda k, c: c.outputs["accumulation"][..., 0] - k.outputs["accumulation"][..., 0],
        "delta_tau": lambda k, c: c.outputs["tau_D"].mean(dim=-1) - k.outputs["tau_D"].mean(dim=-1),
        "delta_T": lambda k, c: c.outputs["transmission"].mean(dim=-1) - k.outputs["transmission"].mean(dim=-1),
    }
    for view_id in list(view_ids) + ["POOLED"]:
        loop_ids = view_ids if view_id == "POOLED" else [view_id]
        gain_vals: List[Tensor] = []
        metric_vals: Dict[str, List[Tensor]] = {key: [] for key in metric_getters}
        for vid in loop_ids:
            mask = regions[vid]["M1_HIGH_J"]
            gain_vals.append(_masked(gain_maps[vid], mask).reshape(-1))
            for key, getter in metric_getters.items():
                metric_vals[key].append(_masked(getter(k1_items[vid], cd_items[vid]), mask).reshape(-1))
        gain_joined = torch.cat([v for v in gain_vals if v.numel()]) if any(v.numel() for v in gain_vals) else torch.empty(0)
        for key, parts in metric_vals.items():
            joined = torch.cat([v for v in parts if v.numel()]) if any(v.numel() for v in parts) else torch.empty(0)
            rows.append(
                {
                    "scene": SCENE,
                    "view_id": view_id,
                    "region": "M1_HIGH_J",
                    "metric": key,
                    "spearman_vs_GAIN": _spearman(gain_joined, joined),
                    "pearson_vs_GAIN": _pearson(gain_joined, joined),
                    "note": "pixel correlation within M1_HIGH_J; pixels are not independent samples; diagnostic association only",
                }
            )
    return rows


def _quantile_monotonic_rows(quantile_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics = (
        "CDEPTH_radius_p50",
        "delta_radius_p50",
        "CDEPTH_density_p50",
        "delta_density_p50",
        "delta_alpha_p50",
        "delta_tau_p50",
        "delta_T_p50",
    )
    for view_id in sorted(set(str(row["view_id"]) for row in quantile_rows)):
        qrows = [row for row in quantile_rows if row["view_id"] == view_id]
        qrows = sorted(qrows, key=lambda row: QUANTILES.index(str(row["gain_quantile"])))
        if len(qrows) != 5:
            continue
        xs = torch.arange(1, 6, dtype=torch.float32)
        for metric in metrics:
            vals = torch.tensor([float(row.get(metric, float("nan"))) for row in qrows], dtype=torch.float32)
            diffs = vals[1:] - vals[:-1]
            pos = int((diffs > 0).sum().item())
            neg = int((diffs < 0).sum().item())
            rho = _spearman(xs, vals)
            if pos == 4:
                direction = "NONDECREASING"
                near = True
            elif neg == 4:
                direction = "NONINCREASING"
                near = True
            elif pos >= 3:
                direction = "NEAR_NONDECREASING"
                near = True
            elif neg >= 3:
                direction = "NEAR_NONINCREASING"
                near = True
            else:
                direction = "NON_MONOTONIC"
                near = False
            rows.append(
                {
                    "scene": SCENE,
                    "view_id": view_id,
                    "metric": metric,
                    "Q1": float(vals[0].item()),
                    "Q2": float(vals[1].item()),
                    "Q3": float(vals[2].item()),
                    "Q4": float(vals[3].item()),
                    "Q5": float(vals[4].item()),
                    "spearman_vs_gain_quantile": rho,
                    "monotonic_direction": direction,
                    "MONOTONIC_ASSOCIATION": bool(near and math.isfinite(rho) and abs(rho) >= 0.2),
                }
            )
    return rows


def _cross_view_rows(
    footprint_rows: Sequence[Mapping[str, Any]],
    density_rows: Sequence[Mapping[str, Any]],
    alpha_rows: Sequence[Mapping[str, Any]],
    view_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    all_rows = list(footprint_rows) + list(density_rows) + list(alpha_rows)
    keyed = {(row["run"], row["view_id"], row["region"], row["metric"]): row for row in all_rows}
    specs = (
        ("FINER_FOOTPRINT", "CDEPTH", "projected_center_window_mean_radius_px", -1),
        ("HIGHER_LOCAL_DENSITY", "CDEPTH", "projected_center_window_density_count", 1),
        ("COVERAGE_REBALANCING", "CDEPTH_MINUS_K1", "delta_alpha_accumulation", 0),
    )
    rows: List[Dict[str, Any]] = []
    for candidate, run, metric, expected_sign in specs:
        pooled_gain = float(keyed[(run, "POOLED", "HJ_GAIN", metric)]["p50"])
        pooled_harm = float(keyed[(run, "POOLED", "HJ_HARM", metric)]["p50"])
        pooled_effect = pooled_gain - pooled_harm
        view_signs: List[int] = []
        for view_id in view_ids:
            gain = float(keyed[(run, view_id, "HJ_GAIN", metric)]["p50"])
            harm = float(keyed[(run, view_id, "HJ_HARM", metric)]["p50"])
            effect = gain - harm
            sign = 1 if effect > 0 else (-1 if effect < 0 else 0)
            view_signs.append(sign)
            rows.append(
                {
                    "scene": SCENE,
                    "candidate": candidate,
                    "view_id": view_id,
                    "metric": metric,
                    "run": run,
                    "HJ_GAIN_p50": gain,
                    "HJ_HARM_p50": harm,
                    "effect_gain_minus_harm": effect,
                    "direction_sign": sign,
                    "expected_sign": expected_sign,
                }
            )
        pooled_sign = 1 if pooled_effect > 0 else (-1 if pooled_effect < 0 else 0)
        if expected_sign == 0:
            consistent = sum(1 for s in view_signs if s == pooled_sign and s != 0) >= 2 and pooled_sign != 0
        else:
            consistent = sum(1 for s in view_signs if s == expected_sign) >= 2 and pooled_sign == expected_sign
        rows.append(
            {
                "scene": SCENE,
                "candidate": candidate,
                "view_id": "POOLED",
                "metric": metric,
                "run": run,
                "HJ_GAIN_p50": pooled_gain,
                "HJ_HARM_p50": pooled_harm,
                "effect_gain_minus_harm": pooled_effect,
                "direction_sign": pooled_sign,
                "expected_sign": expected_sign,
                "CROSS_VIEW_CONSISTENT": bool(consistent),
                "view_direction_summary": ";".join(str(s) for s in view_signs),
            }
        )
    return rows


def _strong_subset_rows(
    footprint_rows: Sequence[Mapping[str, Any]],
    density_rows: Sequence[Mapping[str, Any]],
    alpha_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    all_rows = list(footprint_rows) + list(density_rows) + list(alpha_rows)
    keyed = {(row["run"], row["view_id"], row["region"], row["metric"]): row for row in all_rows}
    specs = (
        ("FINER_FOOTPRINT", "CDEPTH", "projected_center_window_mean_radius_px", -1),
        ("HIGHER_LOCAL_DENSITY", "CDEPTH", "projected_center_window_density_count", 1),
        ("COVERAGE_REBALANCING", "CDEPTH_MINUS_K1", "delta_alpha_accumulation", 0),
    )
    rows: List[Dict[str, Any]] = []
    for candidate, run, metric, expected_sign in specs:
        gain = float(keyed[(run, "POOLED", "HJ_GAIN", metric)]["p50"])
        harm = float(keyed[(run, "POOLED", "HJ_HARM", metric)]["p50"])
        sgain = float(keyed[(run, "POOLED", "HJ_STRONG_GAIN", metric)]["p50"])
        sharm = float(keyed[(run, "POOLED", "HJ_STRONG_HARM", metric)]["p50"])
        effect = gain - harm
        strong_effect = sgain - sharm
        sign = 1 if effect > 0 else (-1 if effect < 0 else 0)
        strong_sign = 1 if strong_effect > 0 else (-1 if strong_effect < 0 else 0)
        if expected_sign == 0:
            robust = sign != 0 and sign == strong_sign
        else:
            robust = sign == expected_sign and strong_sign == expected_sign
        rows.append(
            {
                "scene": SCENE,
                "candidate": candidate,
                "metric": metric,
                "run": run,
                "HJ_GAIN_p50": gain,
                "HJ_HARM_p50": harm,
                "gain_minus_harm": effect,
                "HJ_STRONG_GAIN_p50": sgain,
                "HJ_STRONG_HARM_p50": sharm,
                "strong_gain_minus_harm": strong_effect,
                "expected_sign": expected_sign,
                "STRONG_SUBSET_ROBUST": bool(robust),
            }
        )
    return rows


def _scorecard(
    cross_view: Sequence[Mapping[str, Any]],
    strong: Sequence[Mapping[str, Any]],
    monotonic: Sequence[Mapping[str, Any]],
    corr: Sequence[Mapping[str, Any]],
    footprint_rows: Sequence[Mapping[str, Any]],
    density_rows: Sequence[Mapping[str, Any]],
    alpha_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    keyed_cross = {(row["candidate"], row["view_id"]): row for row in cross_view}
    keyed_strong = {row["candidate"]: row for row in strong}
    keyed_corr = {(row["view_id"], row["metric"]): row for row in corr}
    keyed_mono = {(row["view_id"], row["metric"]): row for row in monotonic}
    all_metric_rows = list(footprint_rows) + list(density_rows) + list(alpha_rows)
    keyed = {(row["run"], row["view_id"], row["region"], row["metric"]): row for row in all_metric_rows}

    specs = {
        "FINER_FOOTPRINT": {
            "metric": "projected_center_window_mean_radius_px",
            "run": "CDEPTH",
            "corr_metric": "CDEPTH_projected_center_window_radius",
            "quantile_metric": "CDEPTH_radius_p50",
            "expected_sign": -1,
            "effect_threshold_rel": 0.05,
            "trajectory_compatible": True,
        },
        "HIGHER_LOCAL_DENSITY": {
            "metric": "projected_center_window_density_count",
            "run": "CDEPTH",
            "corr_metric": "CDEPTH_projected_center_window_density",
            "quantile_metric": "CDEPTH_density_p50",
            "expected_sign": 1,
            "effect_threshold_rel": 0.05,
            "trajectory_compatible": True,
        },
        "COVERAGE_REBALANCING": {
            "metric": "delta_alpha_accumulation",
            "run": "CDEPTH_MINUS_K1",
            "corr_metric": "delta_alpha",
            "quantile_metric": "delta_alpha_p50",
            "expected_sign": 0,
            "effect_threshold_abs": 0.002,
            "trajectory_compatible": True,
        },
    }
    rows: List[Dict[str, Any]] = []
    for candidate, spec in specs.items():
        metric = str(spec["metric"])
        run = str(spec["run"])
        gain = float(keyed[(run, "POOLED", "HJ_GAIN", metric)]["p50"])
        harm = float(keyed[(run, "POOLED", "HJ_HARM", metric)]["p50"])
        effect = gain - harm
        expected_sign = int(spec["expected_sign"])
        if "effect_threshold_rel" in spec:
            effect_size = abs(effect) / max(abs(harm), EPS)
            distribution_difference = effect_size >= float(spec["effect_threshold_rel"])
        else:
            effect_size = abs(effect)
            distribution_difference = effect_size >= float(spec["effect_threshold_abs"])
        if expected_sign == 0:
            direction_ok = effect != 0
        else:
            direction_ok = (effect > 0 and expected_sign > 0) or (effect < 0 and expected_sign < 0)
        pooled_cross = keyed_cross.get((candidate, "POOLED"), {})
        cross_consistent = bool(pooled_cross.get("CROSS_VIEW_CONSISTENT", False))
        strong_robust = bool(keyed_strong.get(candidate, {}).get("STRONG_SUBSET_ROBUST", False))
        mono = keyed_mono.get(("POOLED", spec["quantile_metric"]), {})
        monotonic_assoc = bool(mono.get("MONOTONIC_ASSOCIATION", False))
        rho = float(keyed_corr.get(("POOLED", spec["corr_metric"]), {}).get("spearman_vs_GAIN", float("nan")))
        tau_rho = abs(float(keyed_corr.get(("POOLED", "delta_tau"), {}).get("spearman_vs_GAIN", 0.0)))
        t_rho = abs(float(keyed_corr.get(("POOLED", "delta_T"), {}).get("spearman_vs_GAIN", 0.0)))
        control_not_replacement = math.isfinite(rho) and abs(rho) >= 0.2 and abs(rho) >= max(tau_rho, t_rho) - 0.05
        conditions = {
            "distribution_difference": bool(distribution_difference and direction_ok),
            "strong_subset_same_direction": bool(strong_robust),
            "gain_quantile_monotonic": bool(monotonic_assoc and (expected_sign == 0 or (rho > 0) == (expected_sign > 0))),
            "cross_view_consistent": bool(cross_consistent),
            "not_simple_tau_T_replacement": bool(control_not_replacement),
            "trajectory_compatible": bool(spec["trajectory_compatible"]),
        }
        score_count = sum(1 for value in conditions.values() if value)
        if not direction_ok and expected_sign != 0:
            score = "EVIDENCE_AGAINST"
        elif not conditions["distribution_difference"]:
            # Monotonic or cross-view patterns with near-zero region effect are
            # weak diagnostics only; they cannot define a structural signature.
            score = "WEAK" if score_count >= 2 else "EVIDENCE_AGAINST"
        elif not conditions["strong_subset_same_direction"]:
            score = "WEAK"
        elif score_count >= 5:
            score = "STRONG"
        elif score_count >= 3:
            score = "MODERATE"
        elif score_count >= 1:
            score = "WEAK"
        else:
            score = "EVIDENCE_AGAINST"
        rows.append(
            {
                "scene": SCENE,
                "candidate": candidate,
                "score": score,
                "metric": metric,
                "HJ_GAIN_p50": gain,
                "HJ_HARM_p50": harm,
                "gain_minus_harm": effect,
                "effect_size": effect_size,
                "spearman_vs_GAIN": rho,
                **conditions,
                "evidence_count": score_count,
            }
        )
    rows.append(
        {
            "scene": SCENE,
            "candidate": "MORE_DISTRIBUTED_CONTRIBUTORS",
            "score": "NOT_EVALUABLE",
            "metric": "true contributor count / N_eff",
            "CONTRIBUTOR_DIAGNOSTIC_AVAILABLE": False,
            "reason": "Existing renderer outputs do not expose per-pixel contributor IDs or normalized contribution weights.",
        }
    )
    rows.append(
        {
            "scene": SCENE,
            "candidate": "ANISOTROPIC_SUPPORT_REORGANIZATION",
            "score": "NOT_EVALUABLE",
            "metric": "contribution-weighted anisotropy",
            "REGION_SCALE_ATTRIBUTION_NOT_EVALUABLE": True,
            "reason": "Cross-run Gaussian matching is invalid and true contribution weights are unavailable; global anisotropy trajectory exists but region attribution is not evaluated.",
        }
    )
    evaluable = [row for row in rows if row["score"] in ("STRONG", "MODERATE")]
    if any(row["candidate"] == "FINER_FOOTPRINT" and row["score"] == "STRONG" for row in rows) and any(
        row["candidate"] == "HIGHER_LOCAL_DENSITY" and row["score"] in ("STRONG", "MODERATE") for row in rows
    ):
        signature = "FINER_DISTRIBUTED_SUPPORT"
        signature_mode = "COMPOSITE_SIGNATURE"
    elif evaluable:
        ordered = [row for row in rows if row["score"] == "STRONG"] or evaluable
        signature = str(ordered[0]["candidate"])
        signature_mode = "SINGLE_SIGNATURE"
    else:
        signature = "NO_CLEAR_HIGHJ_STRUCTURAL_SIGNATURE"
        signature_mode = "NONE"
    deployability = "UNRESOLVED" if evaluable else "FALSE"
    final = {
        "BENEFICIAL_STRUCTURAL_SIGNATURE": signature,
        "SIGNATURE_MODE": signature_mode,
        "DEPLOYABLE_PROXY_AVAILABLE": deployability,
        "DEPLOYABILITY_NOTE": "Structural observables are current-state computable proxies, but HJ_GAIN/HARM and M1_HIGH_J are diagnostic/oracle regions; no deployable hard-region selector was validated in this stage.",
        "CONTRIBUTOR_DIAGNOSTIC_AVAILABLE": False,
        "REGION_SCALE_ATTRIBUTION_NOT_EVALUABLE": True,
    }
    return rows, final


def _plot_distribution_bars(path: Path, rows: Sequence[Mapping[str, Any]], title: str, metric: str, run_filter: Sequence[str], manifest: List[Dict[str, Any]], output_type: str, view_ids: Sequence[str]) -> None:
    regions = ["HJ_GAIN", "HJ_HARM", "HJ_STRONG_GAIN", "HJ_STRONG_HARM", "M1_LOW_J"]
    plt.figure(figsize=(10, 4.8))
    width = 0.8 / max(len(run_filter), 1)
    xs = np.arange(len(regions))
    for idx, run in enumerate(run_filter):
        vals = []
        for region in regions:
            row = next((r for r in rows if r.get("run") == run and r.get("view_id") == "POOLED" and r.get("region") == region and r.get("metric") == metric), None)
            vals.append(float(row.get("p50", float("nan"))) if row else float("nan"))
        plt.bar(xs + idx * width, vals, width=width, label=run)
    plt.xticks(xs + width * (len(run_filter) - 1) / 2, regions, rotation=20)
    plt.ylabel("median")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    manifest.append({"scene": SCENE, "file_path": str(path), "output_type": output_type, "view_ids": ";".join(view_ids)})


def _quantile_color_map(regions: Mapping[str, Tensor]) -> Image.Image:
    h, w = regions["M1_HIGH_J"].shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {
        "Q1_MOST_HARMED": (180, 20, 120),
        "Q2": (80, 80, 220),
        "Q3": (160, 160, 160),
        "Q4": (80, 190, 120),
        "Q5_MOST_IMPROVED": (20, 170, 40),
    }
    for name, color in colors.items():
        mask = regions[name].detach().bool().cpu().numpy()
        arr[mask] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_visuals(
    render_dir: Path,
    m1_items: Mapping[str, ViewItem],
    k1_items: Mapping[str, ViewItem],
    cd_items: Mapping[str, ViewItem],
    regions: Mapping[str, Mapping[str, Tensor]],
    gain_maps: Mapping[str, Tensor],
    view_ids: Sequence[str],
    footprint_rows: Sequence[Mapping[str, Any]],
    density_rows: Sequence[Mapping[str, Any]],
    alpha_rows: Sequence[Mapping[str, Any]],
    quantile_rows: Sequence[Mapping[str, Any]],
    cross_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    final_summary: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    render_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    radius_vals = torch.cat(
        [
            item.local_radius_map[torch.isfinite(item.local_radius_map)].reshape(-1)
            for items in (k1_items, cd_items)
            for item in items.values()
            if item.local_radius_map is not None and torch.isfinite(item.local_radius_map).any()
        ]
    )
    radius_scale = _q(radius_vals, 0.99)
    density_vals = torch.cat([item.local_density_map.reshape(-1) for items in (k1_items, cd_items) for item in items.values()])
    density_scale = _q(density_vals, 0.99)
    gain_scale = _q(torch.cat([gain_maps[vid].abs().reshape(-1) for vid in view_ids]), 0.99)
    radius_delta_scale = _q(torch.cat([(cd_items[vid].local_radius_map - k1_items[vid].local_radius_map).abs().reshape(-1) for vid in view_ids]), 0.99)
    density_delta_scale = _q(torch.cat([(cd_items[vid].local_density_map - k1_items[vid].local_density_map).abs().reshape(-1) for vid in view_ids]), 0.99)
    alpha_delta_scale = _q(torch.cat([(cd_items[vid].outputs["accumulation"][..., 0] - k1_items[vid].outputs["accumulation"][..., 0]).abs().reshape(-1) for vid in view_ids]), 0.99)

    region_sheet_rows = []
    quantile_sheet_rows = []
    footprint_sheet_rows = []
    density_sheet_rows = []
    alpha_sheet_rows = []
    for view_id in view_ids:
        r = regions[view_id]
        region_sheet_rows.append(
            [
                (f"{view_id} M1_HIGH_J", _mask_to_rgb(r["M1_HIGH_J"])),
                ("HJ_GAIN", _mask_to_rgb(r["HJ_GAIN"], (40, 190, 80))),
                ("HJ_HARM", _mask_to_rgb(r["HJ_HARM"], (220, 60, 60))),
                ("strong gain", _mask_to_rgb(r["HJ_STRONG_GAIN"], (0, 160, 40))),
                ("strong harm", _mask_to_rgb(r["HJ_STRONG_HARM"], (180, 0, 80))),
            ]
        )
        quantile_sheet_rows.append(
            [
                (f"{view_id} gain signed", _signed_to_rgb(gain_maps[view_id], gain_scale)),
                ("high-J gain Q1-Q5", _quantile_color_map(r)),
                ("M1_HIGH_J", _mask_to_rgb(r["M1_HIGH_J"])),
            ]
        )
        kr = k1_items[view_id].local_radius_map
        cr = cd_items[view_id].local_radius_map
        kd = k1_items[view_id].local_density_map
        cd = cd_items[view_id].local_density_map
        ka = k1_items[view_id].outputs["accumulation"][..., 0]
        ca = cd_items[view_id].outputs["accumulation"][..., 0]
        footprint_sheet_rows.append(
            [
                (f"{view_id} K1 radius proxy", _gray_to_uint8(kr, radius_scale)),
                ("CDEPTH radius proxy", _gray_to_uint8(cr, radius_scale)),
                ("delta radius", _signed_to_rgb(cr - kr, radius_delta_scale)),
            ]
        )
        density_sheet_rows.append(
            [
                (f"{view_id} K1 density proxy", _gray_to_uint8(kd, density_scale)),
                ("CDEPTH density proxy", _gray_to_uint8(cd, density_scale)),
                ("delta density", _signed_to_rgb(cd - kd, density_delta_scale)),
            ]
        )
        alpha_sheet_rows.append(
            [
                (f"{view_id} K1 alpha", _gray_to_uint8(ka, 1.0)),
                ("CDEPTH alpha", _gray_to_uint8(ca, 1.0)),
                ("delta alpha", _signed_to_rgb(ca - ka, alpha_delta_scale)),
            ]
        )
    _save_sheet(render_dir / "contact_sheet_highj_gain_harm_regions.png", region_sheet_rows, manifest, "highj_gain_harm_regions", view_ids)
    _save_sheet(render_dir / "contact_sheet_gain_quantile_regions.png", quantile_sheet_rows, manifest, "gain_quantile_regions", view_ids)
    _save_sheet(render_dir / "contact_sheet_projected_footprint_proxy.png", footprint_sheet_rows, manifest, "projected_footprint_proxy", view_ids)
    _save_sheet(render_dir / "contact_sheet_local_density_proxy.png", density_sheet_rows, manifest, "local_density_proxy", view_ids)
    _save_sheet(render_dir / "contact_sheet_alpha_coverage.png", alpha_sheet_rows, manifest, "alpha_coverage", view_ids)

    _plot_distribution_bars(
        render_dir / "plot_footprint_region_distributions.png",
        footprint_rows,
        "Projected-Center Window Radius Proxy",
        "projected_center_window_mean_radius_px",
        ("BND-K1", "CDEPTH"),
        manifest,
        "footprint_distribution_plot",
        view_ids,
    )
    _plot_distribution_bars(
        render_dir / "plot_density_region_distributions.png",
        density_rows,
        "Projected-Center Window Density Proxy",
        "projected_center_window_density_count",
        ("BND-K1", "CDEPTH"),
        manifest,
        "density_distribution_plot",
        view_ids,
    )
    _plot_distribution_bars(
        render_dir / "plot_alpha_region_distributions.png",
        alpha_rows,
        "Alpha / Coverage Delta",
        "delta_alpha_accumulation",
        ("CDEPTH_MINUS_K1",),
        manifest,
        "alpha_distribution_plot",
        view_ids,
    )

    q_pooled = [row for row in quantile_rows if row["view_id"] == "POOLED"]
    q_pooled = sorted(q_pooled, key=lambda row: QUANTILES.index(str(row["gain_quantile"])))
    plt.figure(figsize=(9, 5))
    xs = np.arange(1, 6)
    for key, label in (
        ("CDEPTH_radius_p50", "CDEPTH radius proxy p50"),
        ("CDEPTH_density_p50", "CDEPTH density proxy p50"),
        ("delta_alpha_p50", "delta alpha p50"),
    ):
        vals = np.asarray([float(row.get(key, float("nan"))) for row in q_pooled], dtype=float)
        if np.nanmax(np.abs(vals)) > 0:
            vals = vals / np.nanmax(np.abs(vals))
        plt.plot(xs, vals, marker="o", label=label)
    plt.xticks(xs, ["Q1", "Q2", "Q3", "Q4", "Q5"])
    plt.title("Gain-Quantile Structural Summary (normalized curves)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    qplot = render_dir / "plot_gain_quantile_structural_summary.png"
    plt.savefig(qplot, dpi=150)
    plt.close()
    manifest.append({"scene": SCENE, "file_path": str(qplot), "output_type": "gain_quantile_structural_summary_plot", "view_ids": ";".join(view_ids)})

    cross_lines = []
    for row in cross_rows:
        if row["view_id"] == "POOLED":
            cross_lines.append(
                f"{row['candidate']}: effect={float(row['effect_gain_minus_harm']):.6g}, signs={row.get('view_direction_summary','')}, consistent={row.get('CROSS_VIEW_CONSISTENT','')}"
            )
    _text_sheet(render_dir / "contact_sheet_cross_view_signature_summary.png", "Cross-View Structural Signature Summary", cross_lines, manifest, "cross_view_signature_summary", view_ids)

    score_lines = [
        f"Beneficial signature: {final_summary.get('BENEFICIAL_STRUCTURAL_SIGNATURE')}",
        f"Signature mode: {final_summary.get('SIGNATURE_MODE')}",
        f"Deployable proxy: {final_summary.get('DEPLOYABLE_PROXY_AVAILABLE')}",
        f"Contributor diagnostic available: {final_summary.get('CONTRIBUTOR_DIAGNOSTIC_AVAILABLE')}",
    ]
    for row in score_rows:
        score_lines.append(f"{row['candidate']}: {row['score']}")
    _text_sheet(render_dir / "contact_sheet_final_structural_signature.png", "Final High-J Structural Signature", score_lines, manifest, "final_structural_signature_sheet", view_ids)

    return manifest


def _write_visual_index(render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# BND-CDEPTH High-J Structure Visual Compare Index", ""]
    for row in manifest:
        lines.append(f"- {row['output_type']}: `{row['file_path']}`")
    lines.append("")
    lines.append("Visual assets are ready for external/manual analysis.")
    lines.append("No subjective clear-image correctness judgment was made.")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_research_note(path: Path, summary: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]], outputs: Mapping[str, str]) -> None:
    lines = [
        "# BND-CDEPTH High-J Structural Signature Audit",
        "",
        "## Motivation",
        "",
        "**Code Fact**",
        "",
        "- This stage is a read-only analysis of existing Panama BND-K1 and BND-CDEPTH checkpoints.",
        "- No training, optimizer step, scheduler step, densification, pruning, or checkpoint modification was performed.",
        "- The audit asks why some fixed M1_HIGH_J pixels are recovered by CDEPTH while other pixels in the same diagnostic region are not.",
        "",
        "## CDEPTH As Partial Mitigation",
        "",
        "**Experimental Fact**",
        "",
        "- Prior CDEPTH final result: PSNR `31.753299` versus BND-K1 `31.498353`, delta `+0.254946 dB`.",
        "- Prior fixed M1_HIGH_J local recovery was about `38%`; SSIM and LPIPS were worse than BND-K1.",
        "- Prior pathway classification was `MIXED_GAUSSIAN_STRUCTURE`, with harmful pathway unresolved.",
        "",
        "## Formal Region Definitions",
        "",
        "**Code Fact**",
        "",
        "- `M1_HIGH_J`: M1 object support (`accumulation > 0.01`) and M1 `clear_object_fullsh_raw` max RGB channel `> 1.0`.",
        "- `GAIN(x) = mean_RGB((pred_K1-GT)^2) - mean_RGB((pred_CDEPTH-GT)^2)`.",
        "- `HJ_GAIN = M1_HIGH_J and GAIN > 0`.",
        "- `HJ_HARM = M1_HIGH_J and GAIN < 0`.",
        "- `HJ_STRONG_GAIN`: top 25 percent of positive GAIN inside pooled M1_HIGH_J.",
        "- `HJ_STRONG_HARM`: top 25 percent of negative GAIN magnitude inside pooled M1_HIGH_J.",
        "",
        "## Region Sizes",
        "",
        "**Quantitative Result**",
        "",
        f"- M1_HIGH_J pixel fraction: `{summary.get('M1_HIGH_J_pixel_fraction')}`.",
        f"- HJ_GAIN fraction: `{summary.get('HJ_GAIN_pixel_fraction')}`.",
        f"- HJ_HARM fraction: `{summary.get('HJ_HARM_pixel_fraction')}`.",
        f"- HJ_STRONG_GAIN fraction: `{summary.get('HJ_STRONG_GAIN_pixel_fraction')}`.",
        f"- HJ_STRONG_HARM fraction: `{summary.get('HJ_STRONG_HARM_pixel_fraction')}`.",
        "",
        "## Projected Footprint",
        "",
        "**Code Fact**",
        "",
        f"- Availability: `{summary.get('REGION_FOOTPRINT_AVAILABLE')}`.",
        f"- Semantics: `{summary.get('projected_footprint_semantics')}`.",
        "- This is a projected-center window proxy, not contribution-weighted effective footprint.",
        "",
        "**Quantitative Result**",
        "",
        f"- CDEPTH HJ_GAIN radius proxy median: `{summary.get('CDEPTH_projected_center_window_mean_radius_px_HJ_GAIN_median')}`.",
        f"- CDEPTH HJ_HARM radius proxy median: `{summary.get('CDEPTH_projected_center_window_mean_radius_px_HJ_HARM_median')}`.",
        f"- CDEPTH HJ_GAIN/HARM radius ratio: `{summary.get('CDEPTH_projected_center_window_mean_radius_px_HJ_GAIN_vs_HARM_RATIO')}`.",
        f"- CDEPTH strong gain/harm radius ratio: `{summary.get('CDEPTH_projected_center_window_mean_radius_px_HJ_STRONG_GAIN_vs_HARM_RATIO')}`.",
        "",
        "## Local Gaussian Density",
        "",
        "**Code Fact**",
        "",
        f"- Semantics: `{summary.get('local_density_semantics')}`.",
        "",
        "**Quantitative Result**",
        "",
        f"- CDEPTH HJ_GAIN density median: `{summary.get('CDEPTH_projected_center_window_density_count_HJ_GAIN_median')}`.",
        f"- CDEPTH HJ_HARM density median: `{summary.get('CDEPTH_projected_center_window_density_count_HJ_HARM_median')}`.",
        f"- CDEPTH HJ_GAIN/HARM density ratio: `{summary.get('CDEPTH_projected_center_window_density_count_HJ_GAIN_vs_HARM_RATIO')}`.",
        f"- CDEPTH strong gain/harm density ratio: `{summary.get('CDEPTH_projected_center_window_density_count_HJ_STRONG_GAIN_vs_HARM_RATIO')}`.",
        "",
        "## Effective Contributors",
        "",
        "**Code Fact**",
        "",
        f"- `CONTRIBUTOR_DIAGNOSTIC_AVAILABLE = {summary.get('CONTRIBUTOR_DIAGNOSTIC_AVAILABLE')}`.",
        "- Existing renderer outputs do not expose per-pixel contributor IDs or normalized contribution weights.",
        "- Raw contributor count and `N_eff = 1/sum_i p_i^2` are therefore not evaluated.",
        "",
        "## Scale / Anisotropy",
        "",
        "**Code Fact**",
        "",
        f"- `REGION_SCALE_ATTRIBUTION_NOT_EVALUABLE = {summary.get('REGION_SCALE_ATTRIBUTION_NOT_EVALUABLE')}`.",
        "- Cross-run Gaussian matching is invalid after densification/pruning, and true contribution weights are unavailable.",
        "- Region-conditioned physical scale / anisotropy attribution is not reported.",
        "",
        "## Alpha / Coverage",
        "",
        "**Quantitative Result**",
        "",
        f"- CDEPTH-minus-K1 HJ_GAIN alpha delta median: `{summary.get('CDEPTH_MINUS_K1_delta_alpha_accumulation_HJ_GAIN_median')}`.",
        f"- CDEPTH-minus-K1 HJ_HARM alpha delta median: `{summary.get('CDEPTH_MINUS_K1_delta_alpha_accumulation_HJ_HARM_median')}`.",
        f"- CDEPTH-minus-K1 strong gain/harm alpha delta ratio: `{summary.get('CDEPTH_MINUS_K1_delta_alpha_accumulation_HJ_STRONG_GAIN_vs_HARM_RATIO')}`.",
        "",
        "## Gain Quantiles And Correlation",
        "",
        "**Quantitative Result**",
        "",
        f"- Monotonic footprint association: `{summary.get('FINER_FOOTPRINT_MONOTONIC_ASSOCIATION')}`.",
        f"- Monotonic density association: `{summary.get('HIGHER_LOCAL_DENSITY_MONOTONIC_ASSOCIATION')}`.",
        f"- Monotonic alpha association: `{summary.get('COVERAGE_REBALANCING_MONOTONIC_ASSOCIATION')}`.",
        f"- Spearman(GAIN, CDEPTH radius proxy): `{summary.get('spearman_GAIN_CDEPTH_projected_center_window_radius')}`.",
        f"- Spearman(GAIN, CDEPTH density proxy): `{summary.get('spearman_GAIN_CDEPTH_projected_center_window_density')}`.",
        f"- Spearman(GAIN, delta alpha): `{summary.get('spearman_GAIN_delta_alpha')}`.",
        "",
        "## Cross-View Consistency",
        "",
        "**Quantitative Result**",
        "",
        f"- FINER_FOOTPRINT cross-view: `{summary.get('FINER_FOOTPRINT_CROSS_VIEW_CONSISTENT')}`.",
        f"- HIGHER_LOCAL_DENSITY cross-view: `{summary.get('HIGHER_LOCAL_DENSITY_CROSS_VIEW_CONSISTENT')}`.",
        f"- COVERAGE_REBALANCING cross-view: `{summary.get('COVERAGE_REBALANCING_CROSS_VIEW_CONSISTENT')}`.",
        "",
        "## Controls",
        "",
        "**Code Fact**",
        "",
        "- Depth, tau, T, NEW_LOW_T, direct signal, and medium signal are reported as controls.",
        "- Direct/medium control uses true renderer `direct_object_signal` and `rgb_medium`; it does not use `J*T` image approximation.",
        "",
        "**Quantitative Result**",
        "",
        f"- Spearman(GAIN, delta tau): `{summary.get('spearman_GAIN_delta_tau')}`.",
        f"- Spearman(GAIN, delta T): `{summary.get('spearman_GAIN_delta_T')}`.",
        f"- NEW_LOW_T fraction in HJ_GAIN: `{summary.get('NEW_LOW_T_fraction_HJ_GAIN')}`.",
        f"- NEW_LOW_T fraction in HJ_HARM: `{summary.get('NEW_LOW_T_fraction_HJ_HARM')}`.",
        "",
        "## Structural Signature Scorecard",
        "",
        "**Quantitative Result**",
        "",
    ]
    for row in score_rows:
        lines.append(f"- `{row['candidate']}`: `{row['score']}`.")
    lines.extend(
        [
            "",
            "## Beneficial Structural Signature",
            "",
            "**Quantitative Conclusion**",
            "",
            f"- Beneficial structural signature: `{summary.get('BENEFICIAL_STRUCTURAL_SIGNATURE')}`.",
            f"- Signature mode: `{summary.get('SIGNATURE_MODE')}`.",
            "",
            "**Inference**",
            "",
            "- The result is spatial association evidence, not a causal proof.",
            "- No statement is made about geometric or clear-image physical correctness.",
            "",
            "## Is It Merely Gaussian Count?",
            "",
            "**Quantitative Conclusion**",
            "",
            f"- `IS_IT_JUST_MORE_GAUSSIANS = {summary.get('IS_IT_JUST_MORE_GAUSSIANS')}`.",
            f"- Reason: {summary.get('IS_IT_JUST_MORE_GAUSSIANS_REASON')}",
            "",
            "## Deployability",
            "",
            "**Inference**",
            "",
            f"- `DEPLOYABLE_PROXY_AVAILABLE = {summary.get('DEPLOYABLE_PROXY_AVAILABLE')}`.",
            f"- {summary.get('DEPLOYABILITY_NOTE')}",
            "",
            "## Next Single-Factor Experiment",
            "",
            "**Hypothesis**",
            "",
            f"- Recommended next step: `{summary.get('NEXT_SINGLE_FACTOR_EXPERIMENT')}`.",
            "",
            "## Outputs",
            "",
            f"- Final summary: `{outputs['summary_json']}`.",
            f"- Output manifest: `{outputs['output_manifest']}`.",
            f"- Visual manifest: `{outputs['visual_manifest']}`.",
            f"- Visual index: `{outputs['visual_index']}`.",
            "",
            "No subjective clear-image correctness judgment was made.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _summary_from_rows(
    region_rows: Sequence[Mapping[str, Any]],
    footprint_rows: Sequence[Mapping[str, Any]],
    density_rows: Sequence[Mapping[str, Any]],
    alpha_rows: Sequence[Mapping[str, Any]],
    corr_rows: Sequence[Mapping[str, Any]],
    monotonic_rows: Sequence[Mapping[str, Any]],
    cross_rows: Sequence[Mapping[str, Any]],
    depth_rows: Sequence[Mapping[str, Any]],
    direct_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    final_flags: Mapping[str, Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = dict(final_flags)
    total = next(float(row["pixels"]) / max(float(row["pixel_fraction"]), EPS) for row in region_rows if row["view_id"] == "POOLED" and row["region"] == "M1_HIGH_J")
    for row in region_rows:
        if row["view_id"] == "POOLED":
            summary[f"{row['region']}_pixels"] = int(row["pixels"])
            summary[f"{row['region']}_pixel_fraction"] = float(row["pixel_fraction"])
    summary["total_pixels"] = int(round(total))
    summary["REGION_FOOTPRINT_AVAILABLE"] = "PROXY_PROJECTED_CENTER_WINDOW"
    summary["projected_footprint_semantics"] = f"mean projected Gaussian radius of centers inside a square window of radius {WINDOW_RADIUS_PX}px; proxy, not contribution-weighted footprint"
    summary["local_density_semantics"] = f"projected Gaussian center count in a square window of radius {WINDOW_RADIUS_PX}px; proxy, not true contributor count"

    for rows, metric, run in (
        (footprint_rows, "projected_center_window_mean_radius_px", "CDEPTH"),
        (density_rows, "projected_center_window_density_count", "CDEPTH"),
        (alpha_rows, "delta_alpha_accumulation", "CDEPTH_MINUS_K1"),
    ):
        summary.update(_ratio_summary(rows, metric, run))

    for row in corr_rows:
        if row["view_id"] == "POOLED":
            metric = row["metric"]
            summary[f"spearman_GAIN_{metric}"] = float(row["spearman_vs_GAIN"])
            summary[f"pearson_GAIN_{metric}"] = float(row["pearson_vs_GAIN"])

    mono_map = {
        "CDEPTH_radius_p50": "FINER_FOOTPRINT_MONOTONIC_ASSOCIATION",
        "CDEPTH_density_p50": "HIGHER_LOCAL_DENSITY_MONOTONIC_ASSOCIATION",
        "delta_alpha_p50": "COVERAGE_REBALANCING_MONOTONIC_ASSOCIATION",
    }
    for row in monotonic_rows:
        if row["view_id"] == "POOLED" and row["metric"] in mono_map:
            summary[mono_map[str(row["metric"])]] = bool(row["MONOTONIC_ASSOCIATION"])
            summary[f"{mono_map[str(row['metric'])]}_direction"] = row["monotonic_direction"]

    for row in cross_rows:
        if row["view_id"] == "POOLED":
            summary[f"{row['candidate']}_CROSS_VIEW_CONSISTENT"] = bool(row.get("CROSS_VIEW_CONSISTENT", False))
            summary[f"{row['candidate']}_cross_view_effect"] = float(row["effect_gain_minus_harm"])

    depth_keyed = {(row["run"], row["view_id"], row["region"]): row for row in depth_rows}
    for region in ("HJ_GAIN", "HJ_HARM", "HJ_STRONG_GAIN", "HJ_STRONG_HARM"):
        cd = depth_keyed.get(("CDEPTH", "POOLED", region), {})
        summary[f"CDEPTH_tau_p50_{region}"] = float(cd.get("tau_p50", float("nan")))
        summary[f"CDEPTH_T_p50_{region}"] = float(cd.get("T_p50", float("nan")))
        summary[f"NEW_LOW_T_fraction_{region}"] = float(cd.get("P_T_lt_0.1_or_new_low_t", float("nan")))
    dm_keyed = {(row["view_id"], row["region"]): row for row in direct_rows}
    for region in ("HJ_GAIN", "HJ_HARM", "HJ_STRONG_GAIN", "HJ_STRONG_HARM"):
        row = dm_keyed.get(("POOLED", region), {})
        summary[f"direct_delta_abs_mean_{region}"] = float(row.get("direct_delta_abs_mean", float("nan")))
        summary[f"medium_delta_abs_mean_{region}"] = float(row.get("medium_delta_abs_mean", float("nan")))
        summary[f"direct_to_medium_delta_ratio_{region}"] = float(row.get("direct_to_medium_delta_ratio", float("nan")))

    score_map = {row["candidate"]: row["score"] for row in score_rows}
    final_count_nearly_equal = True
    if summary.get("HIGHER_LOCAL_DENSITY") == "STRONG":
        just_more = "PARTIALLY"
    elif score_map.get("HIGHER_LOCAL_DENSITY") in ("STRONG", "MODERATE") and score_map.get("FINER_FOOTPRINT") not in ("STRONG", "MODERATE"):
        just_more = "PARTIALLY"
    elif score_map.get("HIGHER_LOCAL_DENSITY") in ("STRONG", "MODERATE") and score_map.get("FINER_FOOTPRINT") in ("STRONG", "MODERATE"):
        just_more = "NO"
    else:
        just_more = "UNRESOLVED"
    summary["IS_IT_JUST_MORE_GAUSSIANS"] = just_more
    summary["IS_IT_JUST_MORE_GAUSSIANS_REASON"] = (
        "Final global Gaussian counts are nearly equal in the previous trajectory, so any density signature must be local rather than simple final count. "
        "This audit still uses projected-center density proxy rather than true contributors."
    )
    if final_flags["BENEFICIAL_STRUCTURAL_SIGNATURE"] == "FINER_DISTRIBUTED_SUPPORT":
        next_experiment = "no-step deployable bounded-hard-region proxy design for bounded-aware Gaussian refinement"
    elif score_map.get("HIGHER_LOCAL_DENSITY") == "STRONG":
        next_experiment = "K1 densification-capacity single-factor control based on observed population divergence timing"
    elif final_flags["BENEFICIAL_STRUCTURAL_SIGNATURE"] == "NO_CLEAR_HIGHJ_STRUCTURAL_SIGNATURE":
        next_experiment = "training-dynamics trigger audit for densification candidate selection and CDEPTH eligibility changes"
    else:
        next_experiment = "region-conditioned read-only footprint/density proxy validation"
    summary["NEXT_SINGLE_FACTOR_EXPERIMENT"] = next_experiment
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--render-dir", type=Path, default=RENDER_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--research-note", type=Path, default=RESEARCH_NOTE)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    render_dir = (repo / args.render_dir).resolve() if not args.render_dir.is_absolute() else args.render_dir
    log_dir = (repo / args.log_dir).resolve() if not args.log_dir.is_absolute() else args.log_dir
    note_path = (repo / args.research_note).resolve() if not args.research_note.is_absolute() else args.research_note
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "start_head": _git(repo, "rev-parse", "HEAD"),
        "log_6": _git(repo, "log", "-6", "--oneline"),
        "status_short": _git(repo, "status", "--short"),
        "diff_check": _git(repo, "diff", "--check"),
        "tracked_output_files": _git(repo, "ls-files", "outputs", "renders", "logs", "common_masks", "checkpoints"),
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    previous_summary_path = repo / PREVIOUS_OUTPUT_DIR / "cdepth_mitigation_final_summary.json"
    previous_region_path = repo / PREVIOUS_OUTPUT_DIR / "region_definition.json"
    previous_summary = json.loads(previous_summary_path.read_text(encoding="utf8")) if previous_summary_path.exists() else {}
    previous_region = json.loads(previous_region_path.read_text(encoding="utf8")) if previous_region_path.exists() else {}
    input_manifest = {
        "scene": SCENE,
        "final_nominal_step": FINAL_STEP,
        "window_radius_px": WINDOW_RADIUS_PX,
        "previous_summary": str(previous_summary_path),
        "previous_region_definition": str(previous_region_path),
        "previous_summary_key_facts": {
            "BENEFICIAL_MECHANISM": previous_summary.get("BENEFICIAL_MECHANISM"),
            "HARMFUL_MECHANISM": previous_summary.get("HARMFUL_MECHANISM"),
            "PATHWAY_RELATION": previous_summary.get("PATHWAY_RELATION"),
            "HIGHJ_RECOVERY_ONSET_STEP": previous_summary.get("HIGHJ_RECOVERY_ONSET_STEP"),
            "GLOBAL_RECOVERY_ONSET_STEP": previous_summary.get("GLOBAL_RECOVERY_ONSET_STEP"),
        },
        "M1_HIGH_J_source": "recomputed from formal M1 final checkpoint using the previous definition; previous region metadata is referenced",
        "previous_M1_HIGH_J_definition": previous_region.get("M1_HIGH_J_definition"),
    }

    m1_items, m1_deltas, m1_ckpt = _load_final_run(repo, "M1")
    k1_items, k1_deltas, k1_ckpt = _load_final_run(repo, "BND-K1")
    cd_items, cd_deltas, cd_ckpt = _load_final_run(repo, "CDEPTH")
    view_ids = sorted(m1_items, key=lambda view_id: m1_items[view_id].eval_index)
    input_manifest.update(
        {
            "M1_checkpoint": m1_ckpt,
            "K1_checkpoint": k1_ckpt,
            "CDEPTH_checkpoint": cd_ckpt,
            "eval_views": view_ids,
        }
    )
    _write_json(output_dir / "input_manifest.json", input_manifest)

    parameter_rows = m1_deltas + k1_deltas + cd_deltas
    audit_parameter_safety = all(float(row["max_abs_delta"]) == 0.0 for row in parameter_rows)
    _write_csv(output_dir / "parameter_safety.csv", parameter_rows)
    _write_json(output_dir / "parameter_safety.json", {"AUDIT_PARAMETER_SAFETY": "PASS" if audit_parameter_safety else "FAIL", "rows": parameter_rows})

    regions, gain_maps, region_meta = _build_regions(m1_items, k1_items, cd_items, view_ids)
    _write_json(output_dir / "region_definitions.json", region_meta)
    region_rows = _region_size_rows(regions, view_ids)
    _write_csv(output_dir / "highj_region_sizes.csv", region_rows)
    _write_json(output_dir / "highj_region_sizes.json", {"rows": region_rows})

    validation_rows = []
    for view_id in view_ids:
        gain = gain_maps[view_id]
        high = regions[view_id]["M1_HIGH_J"]
        validation_rows.append(
            {
                "scene": SCENE,
                "view_id": view_id,
                "M1_HIGH_J_pixels": int(high.sum().item()),
                "GAIN_mean_highj": float(gain[high].mean().item()) if int(high.sum()) else float("nan"),
                "GAIN_abs_p50_highj": _q(gain[high].abs(), 0.50),
                "GAIN_abs_p90_highj": _q(gain[high].abs(), 0.90),
                "HJ_GAIN_pixels": int(regions[view_id]["HJ_GAIN"].sum().item()),
                "HJ_HARM_pixels": int(regions[view_id]["HJ_HARM"].sum().item()),
            }
        )
    _write_csv(output_dir / "gain_harm_validation.csv", validation_rows)
    _write_json(output_dir / "gain_harm_validation.json", {"rows": validation_rows, "definition": region_meta["GAIN_definition"]})

    run_items = {"BND-K1": k1_items, "CDEPTH": cd_items}
    footprint_rows = _region_metric_rows(run_items, regions, view_ids, "projected_center_window_mean_radius_px", lambda item, _vid: item.local_radius_map)
    density_rows = _region_metric_rows(run_items, regions, view_ids, "projected_center_window_density_count", lambda item, _vid: item.local_density_map)
    alpha_rows = _alpha_rows(k1_items, cd_items, regions, view_ids)
    contributor_rows = [
        {
            "scene": SCENE,
            "CONTRIBUTOR_DIAGNOSTIC_AVAILABLE": False,
            "raw_contributor_count": "unavailable",
            "N_eff": "unavailable",
            "reason": "Existing renderer outputs do not expose per-pixel contributor IDs or normalized contribution weights.",
        }
    ]
    scale_rows = [
        {
            "scene": SCENE,
            "REGION_SCALE_ATTRIBUTION_NOT_EVALUABLE": True,
            "reason": "True contribution weights unavailable and cross-run Gaussian matching is invalid after split/duplicate/prune.",
        }
    ]
    anis_rows = [
        {
            "scene": SCENE,
            "REGION_ANISOTROPY_ATTRIBUTION_NOT_EVALUABLE": True,
            "reason": "True contribution weights unavailable and cross-run Gaussian matching is invalid after split/duplicate/prune.",
        }
    ]
    _write_csv(output_dir / "footprint_region_metrics.csv", footprint_rows)
    _write_json(output_dir / "footprint_region_metrics.json", {"rows": footprint_rows, "availability": "PROXY_PROJECTED_CENTER_WINDOW", "window_radius_px": WINDOW_RADIUS_PX})
    _write_csv(output_dir / "local_density_region_metrics.csv", density_rows)
    _write_json(output_dir / "local_density_region_metrics.json", {"rows": density_rows, "availability": "PROXY_PROJECTED_CENTER_WINDOW", "window_radius_px": WINDOW_RADIUS_PX})
    _write_csv(output_dir / "contributor_region_metrics.csv", contributor_rows)
    _write_json(output_dir / "contributor_region_metrics.json", {"rows": contributor_rows})
    _write_csv(output_dir / "scale_region_metrics.csv", scale_rows)
    _write_json(output_dir / "scale_region_metrics.json", {"rows": scale_rows})
    _write_csv(output_dir / "anisotropy_region_metrics.csv", anis_rows)
    _write_json(output_dir / "anisotropy_region_metrics.json", {"rows": anis_rows})
    _write_csv(output_dir / "alpha_region_metrics.csv", alpha_rows)
    _write_json(output_dir / "alpha_region_metrics.json", {"rows": alpha_rows})

    depth_rows = _depth_tau_control_rows(k1_items, cd_items, regions, view_ids)
    direct_rows = _direct_medium_rows(k1_items, cd_items, regions, gain_maps, view_ids)
    quantile_rows = _gain_quantile_rows(k1_items, cd_items, regions, gain_maps, view_ids)
    corr_rows = _spatial_correlation_rows(k1_items, cd_items, regions, gain_maps, view_ids)
    monotonic_rows = _quantile_monotonic_rows(quantile_rows)
    cross_rows = _cross_view_rows(footprint_rows, density_rows, alpha_rows, view_ids)
    strong_rows = _strong_subset_rows(footprint_rows, density_rows, alpha_rows)
    score_rows, final_flags = _scorecard(cross_rows, strong_rows, monotonic_rows, corr_rows, footprint_rows, density_rows, alpha_rows)

    _write_csv(output_dir / "depth_tau_control_metrics.csv", depth_rows)
    _write_json(output_dir / "depth_tau_control_metrics.json", {"rows": depth_rows})
    _write_csv(output_dir / "direct_medium_control_metrics.csv", direct_rows)
    _write_json(output_dir / "direct_medium_control_metrics.json", {"rows": direct_rows})
    _write_csv(output_dir / "gain_quantile_structural_metrics.csv", quantile_rows)
    _write_json(output_dir / "gain_quantile_structural_metrics.json", {"rows": quantile_rows})
    _write_csv(output_dir / "spatial_correlation_metrics.csv", corr_rows)
    _write_json(output_dir / "spatial_correlation_metrics.json", {"rows": corr_rows})
    _write_csv(output_dir / "gain_quantile_monotonicity.csv", monotonic_rows)
    _write_json(output_dir / "gain_quantile_monotonicity.json", {"rows": monotonic_rows})
    _write_csv(output_dir / "cross_view_signature_metrics.csv", cross_rows)
    _write_json(output_dir / "cross_view_signature_metrics.json", {"rows": cross_rows})
    _write_csv(output_dir / "strong_subset_robustness.csv", strong_rows)
    _write_json(output_dir / "strong_subset_robustness.json", {"rows": strong_rows})
    _write_csv(output_dir / "structural_signature_scorecard.csv", score_rows)
    _write_json(output_dir / "structural_signature_scorecard.json", {"rows": score_rows})
    _write_json(output_dir / "deployability_audit.json", final_flags)

    final_summary = {
        "scene": SCENE,
        "branch": repo_manifest["branch"],
        "start_head": repo_manifest["start_head"],
        "final_nominal_step": FINAL_STEP,
        "actual_K1_checkpoint": k1_ckpt,
        "actual_CDEPTH_checkpoint": cd_ckpt,
        "view_ids": view_ids,
        "AUDIT_PARAMETER_SAFETY": "PASS" if audit_parameter_safety else "FAIL",
        **_summary_from_rows(region_rows, footprint_rows, density_rows, alpha_rows, corr_rows, monotonic_rows, cross_rows, depth_rows, direct_rows, score_rows, final_flags),
    }
    _write_csv(output_dir / "highj_structure_final_summary.csv", [final_summary])
    _write_json(output_dir / "highj_structure_final_summary.json", final_summary)

    visual_manifest = _make_visuals(
        render_dir,
        m1_items,
        k1_items,
        cd_items,
        regions,
        gain_maps,
        view_ids,
        footprint_rows,
        density_rows,
        alpha_rows,
        quantile_rows,
        cross_rows,
        score_rows,
        final_summary,
    )
    _write_json(render_dir / "manifest.json", visual_manifest)
    _write_csv(render_dir / "manifest.csv", visual_manifest)
    _write_visual_index(render_dir, visual_manifest)

    output_manifest = {
        "repo_manifest": str(output_dir / "repo_manifest.json"),
        "input_manifest": str(output_dir / "input_manifest.json"),
        "region_definitions": str(output_dir / "region_definitions.json"),
        "gain_harm_validation": str(output_dir / "gain_harm_validation.json"),
        "highj_region_sizes": str(output_dir / "highj_region_sizes.json"),
        "footprint_region_metrics": str(output_dir / "footprint_region_metrics.json"),
        "local_density_region_metrics": str(output_dir / "local_density_region_metrics.json"),
        "contributor_region_metrics": str(output_dir / "contributor_region_metrics.json"),
        "alpha_region_metrics": str(output_dir / "alpha_region_metrics.json"),
        "scale_region_metrics": str(output_dir / "scale_region_metrics.json"),
        "anisotropy_region_metrics": str(output_dir / "anisotropy_region_metrics.json"),
        "depth_tau_control_metrics": str(output_dir / "depth_tau_control_metrics.json"),
        "direct_medium_control_metrics": str(output_dir / "direct_medium_control_metrics.json"),
        "gain_quantile_structural_metrics": str(output_dir / "gain_quantile_structural_metrics.json"),
        "cross_view_signature_metrics": str(output_dir / "cross_view_signature_metrics.json"),
        "strong_subset_robustness": str(output_dir / "strong_subset_robustness.json"),
        "structural_signature_scorecard": str(output_dir / "structural_signature_scorecard.json"),
        "deployability_audit": str(output_dir / "deployability_audit.json"),
        "final_summary": str(output_dir / "highj_structure_final_summary.json"),
        "visual_manifest": str(render_dir / "manifest.json"),
        "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
    }
    _write_json(output_dir / "manifest.json", output_manifest)

    _write_research_note(
        note_path,
        final_summary,
        score_rows,
        {
            "summary_json": str(output_dir / "highj_structure_final_summary.json"),
            "output_manifest": str(output_dir / "manifest.json"),
            "visual_manifest": str(render_dir / "manifest.json"),
            "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
        },
    )
    print(json.dumps(final_summary, indent=2, sort_keys=True, default=_json_default))

    del m1_items, k1_items, cd_items
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
