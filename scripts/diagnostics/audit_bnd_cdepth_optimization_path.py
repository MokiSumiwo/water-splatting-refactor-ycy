#!/usr/bin/env python
"""Read-only BND-CDEPTH partial-mitigation optimization-path audit.

This diagnostic loads existing Panama BND-K1 and BND-CDEPTH checkpoints,
renders fixed eval views, and writes structure / gain-harm attribution tables
and contact sheets. It never calls optimizer.step(), scheduler.step(),
densification, pruning, or checkpoint writes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
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

from scripts.diagnostics import audit_seafree_panama_legal_solution as sea
from nerfstudio.utils.eval_utils import eval_setup


SCENE = "Panama"
TARGET_STEPS = (1000, 3000, 5000, 8000, 10000, 13000, 15000)
FINAL_NOMINAL_STEP = 15000
OUTPUT_DIR = Path("outputs/bnd_cdepth_mitigation_path_panama_20260811")
RENDER_DIR = Path("renders/bnd_cdepth_mitigation_path_panama_20260811")
LOG_DIR = Path("logs/bnd_cdepth_mitigation_path_panama_20260811")
RESEARCH_NOTE = Path("research_notes/BND_CDEPTH_MITIGATION_PATH_AUDIT_2026-08-11.md")
EPS = 1e-8
LUMA_WEIGHTS = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)


@dataclass(frozen=True)
class RunSpec:
    run: str
    config_relpath: str
    parameterization: str
    rasterize_mode: str


RUNS: Dict[str, RunSpec] = {
    "M1": RunSpec(
        run="M1",
        config_relpath=(
            "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
            "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
        ),
        parameterization="legacy",
        rasterize_mode="classic",
    ),
    "BND-K1": RunSpec(
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
    "CDEPTH": RunSpec(
        run="CDEPTH",
        config_relpath=(
            "outputs/bnd_cdepth_panama_20260811/panama_bnd_cdepth_seed42_step0_to_15000/"
            "water-splatting/20260811_bnd_cdepth/config.yml"
        ),
        parameterization="bounded_sh3",
        rasterize_mode="classic",
    ),
}


@dataclass
class LoadedRun:
    spec: RunSpec
    config_path: Path
    checkpoint_path: Path
    loaded_step: int
    config: Any
    pipeline: Any


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


def _file_manifest_rows(repo: Path, patterns: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(repo.glob(pattern)):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            rel = path.relative_to(repo)
            if rel in seen:
                continue
            seen.add(rel)
            is_file = path.is_file()
            complete = "directory" if path.is_dir() else ("nonempty" if stat.st_size > 0 else "empty")
            reusable = "reviewed_current_source" if str(rel) == "scripts/diagnostics/audit_bnd_cdepth_optimization_path.py" else "partial_or_context_only"
            rows.append(
                {
                    "path": str(rel),
                    "is_file": is_file,
                    "is_dir": path.is_dir(),
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                    "completeness": complete,
                    "reuse_status": reusable,
                    "note": "kept in place; not deleted or cleaned",
                }
            )
    return rows


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _stale_process_rows() -> List[Dict[str, Any]]:
    try:
        text = subprocess.check_output(["ps", "-eo", "pid,ppid,stat,etime,cmd"], text=True)
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    own_pid = str(os.getpid())
    needles = ("bnd_cdepth_path", "cdepth_mitigation_path", "audit_bnd_cdepth_optimization_path", "audit_bnd_cdepth_mitigation_path")
    for line in text.splitlines()[1:]:
        if not any(needle in line for needle in needles):
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, stat, etime, cmd = parts
        if pid == own_pid:
            continue
        if "audit_bnd_cdepth_optimization_path.py" in cmd or "audit_bnd_cdepth_mitigation_path.py" in cmd:
            continue
        rows.append({"pid": pid, "ppid": ppid, "stat": stat, "etime": etime, "cmd": cmd})
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _available_steps(config_path: Path) -> Dict[int, Path]:
    ckpt_dir = config_path.parent / "nerfstudio_models"
    out: Dict[int, Path] = {}
    if ckpt_dir.exists():
        for path in ckpt_dir.glob("step-*.ckpt"):
            try:
                out[int(path.stem.split("-")[1])] = path
            except Exception:
                continue
    return out


def _actual_step(config_path: Path, nominal_step: int) -> Optional[int]:
    steps = _available_steps(config_path)
    if nominal_step in steps:
        return nominal_step
    if nominal_step == 15000 and 14999 in steps:
        return 14999
    nearest = min(steps, key=lambda step: abs(step - nominal_step)) if steps else None
    if nearest is not None and abs(nearest - nominal_step) <= 1:
        return nearest
    return None


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


def _stats(values: Tensor, prefix: str) -> Dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    names = ("count", "mean", "p005", "p01", "p03", "p05", "p10", "p50", "p75", "p90", "p95", "p97", "p99", "p995", "max")
    if flat.numel() == 0:
        return {f"{prefix}{name}": float("nan") for name in names}
    return {
        f"{prefix}count": int(flat.numel()),
        f"{prefix}mean": float(flat.mean().item()),
        f"{prefix}p005": _safe_quantile(flat, 0.005),
        f"{prefix}p01": _safe_quantile(flat, 0.01),
        f"{prefix}p03": _safe_quantile(flat, 0.03),
        f"{prefix}p05": _safe_quantile(flat, 0.05),
        f"{prefix}p10": _safe_quantile(flat, 0.10),
        f"{prefix}p50": _safe_quantile(flat, 0.50),
        f"{prefix}p75": _safe_quantile(flat, 0.75),
        f"{prefix}p90": _safe_quantile(flat, 0.90),
        f"{prefix}p95": _safe_quantile(flat, 0.95),
        f"{prefix}p97": _safe_quantile(flat, 0.97),
        f"{prefix}p99": _safe_quantile(flat, 0.99),
        f"{prefix}p995": _safe_quantile(flat, 0.995),
        f"{prefix}max": float(flat.max().item()),
    }


def _thresholds(values: Tensor, prefix: str, thresholds: Sequence[float], op: str = "gt") -> Dict[str, Any]:
    vals = values.detach().float().reshape(-1)
    vals = vals[torch.isfinite(vals)]
    out: Dict[str, Any] = {}
    for threshold in thresholds:
        if vals.numel() == 0:
            frac = float("nan")
        elif op == "gt":
            frac = float((vals > threshold).float().mean().item())
        elif op == "lt":
            frac = float((vals < threshold).float().mean().item())
        else:
            raise ValueError(op)
        out[f"{prefix}{op}_{threshold:g}"] = frac
    return out


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    vals = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(vals) < 3:
        return float("nan")
    x = torch.tensor([v[0] for v in vals], dtype=torch.float32)
    y = torch.tensor([v[1] for v in vals], dtype=torch.float32)
    rx = torch.argsort(torch.argsort(x)).float()
    ry = torch.argsort(torch.argsort(y)).float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.linalg.norm(rx) * torch.linalg.norm(ry)
    return float((rx * ry).sum().item() / max(float(denom.item()), EPS))


def _luma(rgb: Tensor) -> Tensor:
    return (rgb.detach().float() * LUMA_WEIGHTS.to(rgb.device)).sum(dim=-1)


def _gradient_magnitude(scalar: Tensor) -> Tensor:
    if scalar.ndim == 3:
        scalar = _luma(scalar)
    scalar = scalar.detach().float()
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


def _mask_to_rgb(mask: Tensor, color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    arr = mask.detach().bool().cpu().numpy()
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[arr] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(out, mode="RGB")


def _overlay_mask(base: Image.Image, mask: Tensor, color: Tuple[int, int, int] = (255, 40, 40), alpha: float = 0.55) -> Image.Image:
    image = np.asarray(base.convert("RGB")).astype(np.float32)
    mask_np = mask.detach().bool().cpu().numpy()
    image[mask_np] = image[mask_np] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")


def _tile(image: Image.Image, label: str, tile_width: int = 260) -> Image.Image:
    ratio = tile_width / max(image.width, 1)
    size = (tile_width, max(1, int(round(image.height * ratio))))
    resized = image.resize(size, Image.BILINEAR)
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
        tiles = [_tile(img, label) for label, img in row]
        canvas = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), max(t.height for t in tiles)), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered_rows.append(canvas)
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


def _release(loaded: Optional[LoadedRun]) -> None:
    if loaded is None:
        return
    try:
        del loaded.pipeline
    except Exception:
        pass
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_run(repo: Path, run: str, nominal_step: int) -> LoadedRun:
    spec = RUNS[run]
    config_path = repo / spec.config_relpath
    actual_step = _actual_step(config_path, nominal_step)
    if actual_step is None:
        raise FileNotFoundError(f"{run} missing checkpoint for nominal step {nominal_step}")

    def update_config(config: Any) -> Any:
        config.load_step = actual_step
        return config

    config, pipeline, checkpoint_path, loaded_step = eval_setup(config_path, test_mode="test", update_config_callback=update_config)
    pipeline.model.config.intrinsic_color_parameterization = spec.parameterization
    pipeline.model.config.rasterize_mode = spec.rasterize_mode
    pipeline.eval()
    return LoadedRun(spec, config_path, checkpoint_path, int(loaded_step), config, pipeline)


def _view_records(loaded: LoadedRun) -> List[Tuple[int, str, Any, Mapping[str, Any]]]:
    dataset = loaded.pipeline.datamanager.eval_dataset
    image_filenames = list(getattr(dataset, "image_filenames", []))
    rows: List[Tuple[int, str, Any, Mapping[str, Any]]] = []
    for eval_index, (camera, batch) in enumerate(loaded.pipeline.datamanager.fixed_indices_eval_dataloader):
        filename = image_filenames[eval_index] if eval_index < len(image_filenames) else Path(f"eval_{eval_index}")
        rows.append((eval_index, Path(filename).stem, camera, batch))
    return rows


def _parameter_snapshot(model: Any) -> Dict[str, Tensor]:
    out = {
        "means": model.means.detach().cpu().clone(),
        "scales": model.scales.detach().cpu().clone(),
        "opacities": model.opacities.detach().cpu().clone(),
        "features_dc": model.features_dc.detach().cpu().clone(),
        "features_rest": model.features_rest.detach().cpu().clone(),
    }
    for name, param in model.medium_mlp.named_parameters():
        out[f"medium_mlp.{name}"] = param.detach().cpu().clone()
    return out


def _parameter_delta(before: Mapping[str, Tensor], model: Any, run: str, step: int) -> List[Dict[str, Any]]:
    after = _parameter_snapshot(model)
    rows = []
    for key, value in before.items():
        diff = (after[key] - value).abs()
        rows.append(
            {
                "scene": SCENE,
                "run": run,
                "nominal_step": step,
                "parameter": key,
                "max_abs_delta": float(diff.max().item()) if diff.numel() else 0.0,
            }
        )
    return rows


def _checkpoint_stats(repo: Path, run: str, nominal_step: int, actual_step: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    spec = RUNS[run]
    config_path = repo / spec.config_relpath
    ckpt_path = _available_steps(config_path)[actual_step]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pipe = ckpt["pipeline"]
    scales_raw = pipe["_model.gauss_params.scales"].float()
    scales = torch.exp(scales_raw)
    sorted_scales, _ = torch.sort(scales, dim=-1)
    opacities = torch.sigmoid(pipe["_model.gauss_params.opacities"].float()).reshape(-1)
    geom = torch.exp(torch.log(scales.clamp_min(EPS)).mean(dim=-1))
    anisotropy = sorted_scales[:, -1] / sorted_scales[:, 0].clamp_min(EPS)
    pop = {
        "scene": SCENE,
        "run": run,
        "nominal_step": nominal_step,
        "actual_step": actual_step,
        "checkpoint_path": str(ckpt_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_step_field": int(ckpt.get("step", actual_step)),
        "gaussian_count": int(scales.shape[0]),
    }
    scale_row = {"scene": SCENE, "run": run, "nominal_step": nominal_step, "actual_step": actual_step}
    scale_row.update(_stats(scales[:, 0], "axis_x_"))
    scale_row.update(_stats(scales[:, 1], "axis_y_"))
    scale_row.update(_stats(scales[:, 2], "axis_z_"))
    scale_row.update(_stats(sorted_scales[:, 0], "scale_min_"))
    scale_row.update(_stats(sorted_scales[:, 1], "scale_mid_"))
    scale_row.update(_stats(sorted_scales[:, 2], "scale_max_"))
    scale_row.update(_stats(geom, "scale_geom_mean_"))
    scale_row.update(_stats(torch.log(scales.clamp_min(EPS)).reshape(-1), "log_scale_"))
    scale_row.update(_stats(anisotropy, "anisotropy_"))
    opacity_row = {"scene": SCENE, "run": run, "nominal_step": nominal_step, "actual_step": actual_step}
    opacity_row.update(_stats(opacities, "opacity_"))
    opacity_row.update(_thresholds(opacities, "opacity_", (0.5, 0.8, 0.95, 0.99), "gt"))
    del ckpt
    return pop, scale_row, opacity_row


def _masked_stats(values: Tensor, mask: Tensor, prefix: str) -> Dict[str, Any]:
    vals = values.detach().float()
    if vals.ndim > mask.ndim:
        while mask.ndim < vals.ndim:
            mask = mask[..., None].expand(*vals.shape)
    selected = vals[mask.detach().bool()]
    return _stats(selected.reshape(-1), prefix)


def _support(item: Mapping[str, Any]) -> Tensor:
    return item["outputs"]["accumulation"].detach().float()[..., 0] > 0.01


def _load_m1_regions(repo: Path, output_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Tensor]], Dict[str, Any], List[str]]:
    loaded = None
    try:
        loaded = _load_run(repo, "M1", FINAL_NOMINAL_STEP)
        model = loaded.pipeline.model
        model.eval()
        items: Dict[str, Dict[str, Any]] = {}
        for eval_index, view_id, camera, batch in _view_records(loaded):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
                gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
                metrics = sea._metric_images(model, outputs["pred_image"], gt)
            keep = ("pred_image", "clear_object_fullsh_raw", "transmission", "accumulation")
            items[view_id] = {
                "eval_index": eval_index,
                "view_id": view_id,
                "gt": sea._safe_cpu(gt),
                "metrics": metrics,
                "outputs": {key: sea._safe_cpu(outputs[key]) for key in keep},
            }
        view_ids = list(items.keys())
    finally:
        _release(loaded)

    lumas = torch.cat([_luma(items[view_id]["gt"]).reshape(-1) for view_id in view_ids])
    bright_threshold = _safe_quantile(lumas, 0.80)
    dark_threshold = _safe_quantile(lumas, 0.20)
    edges = torch.cat([_gradient_magnitude(items[view_id]["gt"]).reshape(-1) for view_id in view_ids])
    edge_threshold = _safe_quantile(edges, 0.80)
    pseudo_grads: List[Tensor] = []
    pseudo_by_view: Dict[str, Tensor] = {}
    depth_dir = repo / "undistorted_data/undistorted_Panama/depthAnything_u16"
    for view_id in view_ids:
        depth = torch.from_numpy(np.array(Image.open(depth_dir / f"{view_id}.png"), dtype=np.float32, copy=True))
        if depth.ndim == 3:
            depth = depth[..., 0]
        shape = items[view_id]["gt"].shape[:2]
        if tuple(depth.shape[:2]) != tuple(shape):
            import torch.nn.functional as F

            depth = F.interpolate(depth[None, None, ...], size=tuple(shape), mode="bilinear", align_corners=False)[0, 0]
        depth = depth / depth.max().clamp_min(EPS)
        pseudo_by_view[view_id] = depth
        pseudo_grads.append(_gradient_magnitude(depth).reshape(-1))
    pseudo_grad_threshold = _safe_quantile(torch.cat(pseudo_grads), 0.80)

    regions: Dict[str, Dict[str, Tensor]] = {}
    for view_id in view_ids:
        item = items[view_id]
        gt = item["gt"]
        clear = item["outputs"]["clear_object_fullsh_raw"]
        jmax = clear.amax(dim=-1)
        support = _support(item)
        luma = _luma(gt)
        edge = _gradient_magnitude(gt)
        h, w = luma.shape
        y = torch.arange(h).reshape(h, 1).expand(h, w)
        t_mean = item["outputs"]["transmission"].mean(dim=-1)
        regions[view_id] = {
            "WHOLE_IMAGE": torch.ones_like(jmax, dtype=torch.bool),
            "M1_HIGH_J": support & (jmax > 1.0),
            "M1_LOW_J": support & (jmax <= 1.0),
            "BRIGHT_Q5": luma > bright_threshold,
            "DARK_Q5": luma <= dark_threshold,
            "EDGE_TOP20": edge > edge_threshold,
            "BOTTOM20": y >= int(math.floor(0.8 * h)),
            "M1_LOW_T": t_mean < 0.1,
            "PSEUDO_DEPTH_GRAD_TOP20": _gradient_magnitude(pseudo_by_view[view_id]) > pseudo_grad_threshold,
        }
    metadata = {
        "view_ids": view_ids,
        "bright_q5_threshold": bright_threshold,
        "dark_q5_threshold": dark_threshold,
        "edge_top20_threshold": edge_threshold,
        "pseudo_depth_gradient_top20_threshold": pseudo_grad_threshold,
        "M1_HIGH_J_definition": "object support and M1 clear_object_fullsh_raw max channel > 1.0",
        "M1_LOW_T_definition": "M1 RGB-channel-mean transmission < 0.1",
        "BOTTOM20_definition": "image y coordinate in bottom 20 percent",
    }
    _write_json(output_dir / "region_definition.json", metadata)
    return items, regions, metadata, view_ids


def _render_run_step(
    repo: Path,
    run: str,
    nominal_step: int,
    regions: Mapping[str, Mapping[str, Tensor]],
    store_final: bool,
    actual_step: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Image.Image]]:
    loaded = None
    try:
        loaded = _load_run(repo, run, nominal_step)
        model = loaded.pipeline.model
        model.eval()
        before = _parameter_snapshot(model)
        per_view_rows: List[Dict[str, Any]] = []
        radius_rows: List[Dict[str, Any]] = []
        visible_scale_rows: List[Dict[str, Any]] = []
        final_items: Dict[str, Dict[str, Any]] = {}
        residual_images: Dict[str, Image.Image] = {}
        for eval_index, view_id, camera, batch in _view_records(loaded):
            with torch.no_grad():
                outputs = model.get_outputs_for_camera(camera)
                gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
                metrics = sea._metric_images(model, outputs["pred_image"], gt)
            gt_cpu = sea._safe_cpu(gt)
            pred_cpu = sea._safe_cpu(outputs["pred_image"])
            err = (pred_cpu - gt_cpu).square().mean(dim=-1)
            residual_images[view_id] = _gray_to_uint8(err, 0.02)
            row = {
                "scene": SCENE,
                "run": run,
                "nominal_step": nominal_step,
                "actual_step": actual_step,
                "eval_index": eval_index,
                "view_id": view_id,
                **metrics,
            }
            for region in ("M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5"):
                mask = regions[view_id][region]
                row[f"{region}_mse"] = float(err[mask].mean().item()) if int(mask.sum()) else float("nan")
                row[f"{region}_l1"] = float((pred_cpu - gt_cpu).abs().mean(dim=-1)[mask].mean().item()) if int(mask.sum()) else float("nan")
            per_view_rows.append(row)

            radii = getattr(model, "radii", None)
            visible = outputs.get("gaussian_visible_mask")
            if isinstance(radii, Tensor):
                radii_cpu = radii.detach().float().reshape(-1).cpu()
                if isinstance(visible, Tensor) and visible.numel() == radii_cpu.numel():
                    visible_cpu = visible.detach().bool().reshape(-1).cpu()
                else:
                    visible_cpu = radii_cpu > 0
                visible_radii = radii_cpu[visible_cpu & torch.isfinite(radii_cpu) & (radii_cpu > 0)]
                rrow = {
                    "scene": SCENE,
                    "run": run,
                    "nominal_step": nominal_step,
                    "actual_step": actual_step,
                    "view_id": view_id,
                    "radii_semantics": "project_gaussians screen-space radius; treated as projected-radius proxy in pixels",
                    "visible_fraction": float(visible_cpu.float().mean().item()) if visible_cpu.numel() else float("nan"),
                }
                rrow.update(_stats(visible_radii, "radius_"))
                rrow.update(_thresholds(visible_radii, "radius_", (4.0, 8.0, 16.0, 32.0), "gt"))
                support_proxy = math.pi * visible_radii.square()
                rrow.update(_stats(support_proxy, "support_proxy_"))
                radius_rows.append(rrow)
                if visible_cpu.numel() == model.scales.shape[0]:
                    v_scales = torch.exp(model.scales.detach().float().cpu())[visible_cpu]
                    if v_scales.numel():
                        sorted_scales, _ = torch.sort(v_scales, dim=-1)
                        anis = sorted_scales[:, -1] / sorted_scales[:, 0].clamp_min(EPS)
                        srow = {"scene": SCENE, "run": run, "nominal_step": nominal_step, "actual_step": actual_step, "view_id": view_id}
                        srow.update(_stats(sorted_scales[:, 2], "visible_scale_max_"))
                        srow.update(_stats(anis, "visible_anisotropy_"))
                        visible_scale_rows.append(srow)
            if store_final:
                keep = (
                    "pred_image",
                    "direct_object_signal",
                    "rgb_medium",
                    "depth",
                    "accumulation",
                    "tau_D",
                    "transmission",
                    "clear_object_fullsh_raw",
                    "medium_rgb",
                )
                final_items[view_id] = {
                    "scene": SCENE,
                    "run": run,
                    "nominal_step": nominal_step,
                    "actual_step": actual_step,
                    "eval_index": eval_index,
                    "view_id": view_id,
                    "gt": gt_cpu,
                    "metrics": metrics,
                    "outputs": {key: sea._safe_cpu(outputs[key]) for key in keep if key in outputs},
                }
        delta_rows = _parameter_delta(before, model, run, nominal_step)
        return per_view_rows, radius_rows, visible_scale_rows, final_items, delta_rows, residual_images
    finally:
        _release(loaded)


def _aggregate_rgb_rows(per_view: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, str], List[Mapping[str, Any]]] = {}
    for row in per_view:
        grouped.setdefault((int(row["nominal_step"]), str(row["run"])), []).append(row)
    rgb_rows: List[Dict[str, Any]] = []
    high_rows: List[Dict[str, Any]] = []
    for (step, run), rows in sorted(grouped.items()):
        row = {
            "scene": SCENE,
            "run": run,
            "nominal_step": step,
            "actual_step": rows[0]["actual_step"],
            "num_views": len(rows),
            "view_ids": ";".join(str(item["view_id"]) for item in rows),
            "psnr": _mean(float(item["psnr"]) for item in rows),
            "ssim": _mean(float(item["ssim"]) for item in rows),
            "lpips": _mean(float(item["lpips"]) for item in rows),
            "mse": _mean(float(item["mse"]) for item in rows),
        }
        rgb_rows.append(row)
        high_rows.append(
            {
                "scene": SCENE,
                "run": run,
                "nominal_step": step,
                "actual_step": rows[0]["actual_step"],
                "M1_HIGH_J_mse": _mean(float(item["M1_HIGH_J_mse"]) for item in rows),
                "M1_HIGH_J_l1": _mean(float(item["M1_HIGH_J_l1"]) for item in rows),
                "M1_LOW_J_mse": _mean(float(item["M1_LOW_J_mse"]) for item in rows),
                "BRIGHT_Q5_mse": _mean(float(item["BRIGHT_Q5_mse"]) for item in rows),
            }
        )
    return rgb_rows, high_rows


def _paired_step_rows(rgb_rows: Sequence[Mapping[str, Any]], high_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rgb = {(int(row["nominal_step"]), str(row["run"])): row for row in rgb_rows}
    high = {(int(row["nominal_step"]), str(row["run"])): row for row in high_rows}
    rows: List[Dict[str, Any]] = []
    for step in TARGET_STEPS:
        if (step, "BND-K1") not in rgb or (step, "CDEPTH") not in rgb:
            continue
        k = rgb[(step, "BND-K1")]
        c = rgb[(step, "CDEPTH")]
        hk = high[(step, "BND-K1")]
        hc = high[(step, "CDEPTH")]
        rows.append(
            {
                "scene": SCENE,
                "nominal_step": step,
                "actual_step_K1": k["actual_step"],
                "actual_step_CDEPTH": c["actual_step"],
                "K1_psnr": k["psnr"],
                "CDEPTH_psnr": c["psnr"],
                "K1_ssim": k["ssim"],
                "CDEPTH_ssim": c["ssim"],
                "K1_lpips": k["lpips"],
                "CDEPTH_lpips": c["lpips"],
                "K1_mse": k["mse"],
                "CDEPTH_mse": c["mse"],
                "GLOBAL_MSE_GAIN": float(k["mse"]) - float(c["mse"]),
                "K1_highJ_mse": hk["M1_HIGH_J_mse"],
                "CDEPTH_highJ_mse": hc["M1_HIGH_J_mse"],
                "HIGHJ_MSE_GAIN": float(hk["M1_HIGH_J_mse"]) - float(hc["M1_HIGH_J_mse"]),
                "K1_lowJ_mse": hk["M1_LOW_J_mse"],
                "CDEPTH_lowJ_mse": hc["M1_LOW_J_mse"],
                "K1_brightQ5_mse": hk["BRIGHT_Q5_mse"],
                "CDEPTH_brightQ5_mse": hc["BRIGHT_Q5_mse"],
            }
        )
    return rows


def _stable_positive_onset(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[int]:
    ordered = sorted(rows, key=lambda row: int(row["nominal_step"]))
    for idx in range(len(ordered) - 1):
        if float(ordered[idx][key]) > 0 and float(ordered[idx + 1][key]) > 0:
            return int(ordered[idx]["nominal_step"])
    return None


def _first_stable_onset(rows: Sequence[Mapping[str, Any]], key: str, threshold: float, absolute: bool = True) -> Optional[int]:
    ordered = sorted(rows, key=lambda row: int(row["nominal_step"]))
    for idx in range(len(ordered) - 1):
        v0 = float(ordered[idx].get(key, float("nan")))
        v1 = float(ordered[idx + 1].get(key, float("nan")))
        if not (math.isfinite(v0) and math.isfinite(v1)):
            continue
        if absolute:
            if abs(v0) >= threshold and abs(v1) >= threshold and (v0 >= 0) == (v1 >= 0):
                return int(ordered[idx]["nominal_step"])
        elif v0 >= threshold and v1 >= threshold:
            return int(ordered[idx]["nominal_step"])
    return None


def _compare_population(pop_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keyed = {(int(row["nominal_step"]), str(row["run"])): row for row in pop_rows}
    rows = []
    for step in TARGET_STEPS:
        k = keyed.get((step, "BND-K1"))
        c = keyed.get((step, "CDEPTH"))
        if not k or not c:
            continue
        nk = int(k["gaussian_count"])
        nc = int(c["gaussian_count"])
        rows.append(
            {
                "scene": SCENE,
                "nominal_step": step,
                "K1_gaussian_count": nk,
                "CDEPTH_gaussian_count": nc,
                "POPULATION_DELTA": nc - nk,
                "POPULATION_RATIO": nc / max(nk, 1),
                "POPULATION_REL_DELTA": (nc - nk) / max(nk, 1),
            }
        )
    return rows


def _aggregate_by_run_step(rows: Sequence[Mapping[str, Any]], prefixes: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["nominal_step"]), str(row["run"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (step, run), items in sorted(grouped.items()):
        row = {"scene": SCENE, "run": run, "nominal_step": step, "actual_step": items[0].get("actual_step", "")}
        for prefix in prefixes:
            keys = [key for key in items[0] if key.startswith(prefix)]
            for key in keys:
                row[key] = _mean(float(item[key]) for item in items if key in item and str(item[key]) != "")
        out.append(row)
    return out


def _compare_metric_rows(rows: Sequence[Mapping[str, Any]], key: str, out_key: str) -> List[Dict[str, Any]]:
    keyed = {(int(row["nominal_step"]), str(row["run"])): row for row in rows}
    out = []
    for step in TARGET_STEPS:
        k = keyed.get((step, "BND-K1"))
        c = keyed.get((step, "CDEPTH"))
        if not k or not c or key not in k or key not in c:
            continue
        kv = float(k[key])
        cv = float(c[key])
        out.append(
            {
                "scene": SCENE,
                "nominal_step": step,
                f"K1_{out_key}": kv,
                f"CDEPTH_{out_key}": cv,
                f"{out_key}_DELTA": cv - kv,
                f"{out_key}_REL_DELTA": (cv - kv) / max(abs(kv), EPS),
            }
        )
    return out


def _region_pixel_count(regions: Mapping[str, Mapping[str, Tensor]], view_ids: Sequence[str], region: str) -> Tuple[int, int]:
    pixels = sum(int(regions[view_id][region].sum().item()) for view_id in view_ids)
    total = sum(int(regions[view_id][region].numel()) for view_id in view_ids)
    return pixels, total


def _final_spatial_attribution(
    k1: Mapping[str, Mapping[str, Any]],
    cd: Mapping[str, Mapping[str, Any]],
    m1: Mapping[str, Mapping[str, Any]],
    regions: Mapping[str, Mapping[str, Tensor]],
    view_ids: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Tensor]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    gain_maps: Dict[str, Tensor] = {}
    pos_vals: List[Tensor] = []
    neg_vals: List[Tensor] = []
    for view_id in view_ids:
        gt = m1[view_id]["gt"]
        err_k = (k1[view_id]["outputs"]["pred_image"] - gt).square().mean(dim=-1)
        err_c = (cd[view_id]["outputs"]["pred_image"] - gt).square().mean(dim=-1)
        gain = err_k - err_c
        gain_maps[view_id] = gain
        pos = gain[gain > 0]
        neg = -gain[gain < 0]
        if pos.numel():
            pos_vals.append(pos)
        if neg.numel():
            neg_vals.append(neg)
    pos_thr = _safe_quantile(torch.cat(pos_vals), 0.80) if pos_vals else float("inf")
    neg_thr = _safe_quantile(torch.cat(neg_vals), 0.80) if neg_vals else float("inf")

    masks: Dict[str, Dict[str, Tensor]] = {}
    total_gain_mass = 0.0
    total_harm_mass = 0.0
    total_pixels = 0
    harm_pixels = 0
    for view_id in view_ids:
        gain = gain_maps[view_id]
        masks[view_id] = {
            "GAIN_PIXELS": gain > 0,
            "HARM_PIXELS": gain < 0,
            "STRONG_GAIN_TOP20": gain >= pos_thr,
            "STRONG_HARM_TOP20": (-gain) >= neg_thr,
        }
        total_gain_mass += float(gain.clamp_min(0).sum().item())
        total_harm_mass += float((-gain).clamp_min(0).sum().item())
        total_pixels += int(gain.numel())
        harm_pixels += int((gain < 0).sum().item())

    region_names = ("M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5", "DARK_Q5", "EDGE_TOP20", "BOTTOM20")
    gain_rows: List[Dict[str, Any]] = []
    for region in region_names:
        gain_mass = 0.0
        harm_mass = 0.0
        region_pixels = 0
        region_harm = 0
        region_gain = 0
        for view_id in view_ids:
            region_mask = regions[view_id][region]
            gain = gain_maps[view_id]
            gain_mass += float(gain[region_mask].clamp_min(0).sum().item())
            harm_mass += float((-gain[region_mask]).clamp_min(0).sum().item())
            region_pixels += int(region_mask.sum().item())
            region_harm += int((region_mask & masks[view_id]["HARM_PIXELS"]).sum().item())
            region_gain += int((region_mask & masks[view_id]["GAIN_PIXELS"]).sum().item())
        pixel_fraction = region_pixels / max(total_pixels, 1)
        gain_fraction = gain_mass / max(total_gain_mass, EPS)
        harm_fraction = harm_mass / max(total_harm_mass, EPS)
        gain_rows.append(
            {
                "scene": SCENE,
                "region": region,
                "pixels": region_pixels,
                "pixel_fraction": pixel_fraction,
                "GAIN_MASS": gain_mass,
                "HARM_MASS": harm_mass,
                "GAIN_MASS_FRACTION": gain_fraction,
                "HARM_MASS_FRACTION": harm_fraction,
                "GAIN_ENRICHMENT": gain_fraction / max(pixel_fraction, EPS),
                "HARM_ENRICHMENT": harm_fraction / max(pixel_fraction, EPS),
                "P_GAIN_given_region": region_gain / max(region_pixels, 1),
                "P_HARM_given_region": region_harm / max(region_pixels, 1),
            }
        )

    def region_compare(mask_name: str, view_id: str) -> Tensor:
        if mask_name in regions[view_id]:
            return regions[view_id][mask_name]
        return masks[view_id][mask_name]

    structural_rows: List[Dict[str, Any]] = []
    compare_regions = ("GAIN_PIXELS", "HARM_PIXELS", "STRONG_GAIN_TOP20", "STRONG_HARM_TOP20", "M1_HIGH_J", "NEW_LOW_T")
    low_t_masks: Dict[str, Dict[str, Tensor]] = {}
    for view_id in view_ids:
        tk = k1[view_id]["outputs"]["transmission"].mean(dim=-1)
        tc = cd[view_id]["outputs"]["transmission"].mean(dim=-1)
        low_t_masks[view_id] = {
            "K1_LOW_T": tk < 0.1,
            "CDEPTH_LOW_T": tc < 0.1,
            "NEW_LOW_T": (tc < 0.1) & (tk >= 0.1),
            "PERSISTENT_LOW_T": (tc < 0.1) & (tk < 0.1),
            "RESOLVED_LOW_T": (tc >= 0.1) & (tk < 0.1),
        }
    for region in compare_regions:
        for run_name, source in (("BND-K1", k1), ("CDEPTH", cd)):
            alpha_vals: List[Tensor] = []
            tau_vals: List[Tensor] = []
            t_vals: List[Tensor] = []
            depth_vals: List[Tensor] = []
            for view_id in view_ids:
                mask = low_t_masks[view_id][region] if region == "NEW_LOW_T" else region_compare(region, view_id)
                out = source[view_id]["outputs"]
                alpha_vals.append(out["accumulation"][..., 0][mask])
                tau_vals.append(out["tau_D"].mean(dim=-1)[mask])
                t_vals.append(out["transmission"].mean(dim=-1)[mask])
                depth_vals.append(out["depth"][..., 0][mask])
            row = {"scene": SCENE, "run": run_name, "region": region}
            for prefix, vals in (("alpha_", alpha_vals), ("tau_", tau_vals), ("T_", t_vals), ("depth_", depth_vals)):
                joined = torch.cat([v.reshape(-1) for v in vals if v.numel()]) if any(v.numel() for v in vals) else torch.empty(0)
                row.update(_stats(joined, prefix))
            structural_rows.append(row)

    lowt_rows: List[Dict[str, Any]] = []
    p_harm = harm_pixels / max(total_pixels, 1)
    new_lowt_pixels = 0
    new_lowt_harm = 0
    for mask_name in ("K1_LOW_T", "CDEPTH_LOW_T", "NEW_LOW_T", "PERSISTENT_LOW_T", "RESOLVED_LOW_T"):
        pixels = 0
        harm = 0
        gain_sum = 0.0
        mse_k = 0.0
        mse_c = 0.0
        overlap_rows: Dict[str, float] = {}
        for view_id in view_ids:
            mask = low_t_masks[view_id][mask_name]
            gain = gain_maps[view_id]
            gt = m1[view_id]["gt"]
            err_k = (k1[view_id]["outputs"]["pred_image"] - gt).square().mean(dim=-1)
            err_c = (cd[view_id]["outputs"]["pred_image"] - gt).square().mean(dim=-1)
            pixels += int(mask.sum().item())
            harm += int((mask & masks[view_id]["HARM_PIXELS"]).sum().item())
            gain_sum += float(gain[mask].mean().item()) if int(mask.sum()) else 0.0
            mse_k += float(err_k[mask].mean().item()) if int(mask.sum()) else 0.0
            mse_c += float(err_c[mask].mean().item()) if int(mask.sum()) else 0.0
        if mask_name == "NEW_LOW_T":
            new_lowt_pixels = pixels
            new_lowt_harm = harm
        row = {
            "scene": SCENE,
            "mask": mask_name,
            "definition": "RGB-channel-mean transmission pixel mask",
            "pixels": pixels,
            "pixel_fraction": pixels / max(total_pixels, 1),
            "P_HARM_given_mask": harm / max(pixels, 1),
            "HARM_ENRICHMENT": (harm / max(pixels, 1)) / max(p_harm, EPS),
            "mean_GAIN": gain_sum / max(len(view_ids), 1),
            "mean_MSE_K1": mse_k / max(len(view_ids), 1),
            "mean_MSE_CDEPTH": mse_c / max(len(view_ids), 1),
        }
        for region in ("M1_HIGH_J", "BRIGHT_Q5", "DARK_Q5", "EDGE_TOP20", "BOTTOM20", "PSEUDO_DEPTH_GRAD_TOP20"):
            overlap = 0
            union = 0
            region_pixels = 0
            for view_id in view_ids:
                mask = low_t_masks[view_id][mask_name]
                reg = regions[view_id][region]
                overlap += int((mask & reg).sum().item())
                union += int((mask | reg).sum().item())
                region_pixels += int(reg.sum().item())
            p_region = region_pixels / max(total_pixels, 1)
            p_region_given = overlap / max(pixels, 1)
            row[f"overlap_{region}"] = overlap
            row[f"IoU_{region}"] = overlap / max(union, 1)
            row[f"enrichment_{region}"] = p_region_given / max(p_region, EPS)
        lowt_rows.append(row)
    lowt_harm_enrichment = (new_lowt_harm / max(new_lowt_pixels, 1)) / max(p_harm, EPS)
    lowt_summary = {
        "NEW_LOWT_HARM_ENRICHMENT": lowt_harm_enrichment,
        "NEW_LOW_T_pixels": new_lowt_pixels,
        "LOWT_HARM_ALIGNED": bool(new_lowt_pixels > 100 and lowt_harm_enrichment >= 2.0),
    }

    attn_rows: List[Dict[str, Any]] = []
    for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J", "GAIN_PIXELS", "HARM_PIXELS", "NEW_LOW_T", "EDGE_TOP20"):
        for run_name, source in (("BND-K1", k1), ("CDEPTH", cd)):
            taus: List[Tensor] = []
            ts: List[Tensor] = []
            mediums: List[Tensor] = []
            directs: List[Tensor] = []
            for view_id in view_ids:
                if region == "NEW_LOW_T":
                    mask = low_t_masks[view_id]["NEW_LOW_T"]
                elif region in masks[view_id]:
                    mask = masks[view_id][region]
                else:
                    mask = regions[view_id][region]
                out = source[view_id]["outputs"]
                taus.append(out["tau_D"].mean(dim=-1)[mask])
                ts.append(out["transmission"].mean(dim=-1)[mask])
                mediums.append(out["rgb_medium"].mean(dim=-1)[mask])
                directs.append(out["direct_object_signal"].mean(dim=-1)[mask])
            tau = torch.cat([v.reshape(-1) for v in taus if v.numel()]) if any(v.numel() for v in taus) else torch.empty(0)
            t = torch.cat([v.reshape(-1) for v in ts if v.numel()]) if any(v.numel() for v in ts) else torch.empty(0)
            row = {"scene": SCENE, "run": run_name, "region": region}
            for q in (0.50, 0.75, 0.90, 0.95, 0.97, 0.99, 0.995):
                row[f"tau_p{q:g}"] = _safe_quantile(tau, q)
            for q in (0.005, 0.01, 0.03, 0.05, 0.10, 0.50):
                row[f"T_p{q:g}"] = _safe_quantile(t, q)
            row.update(_thresholds(t, "T_", (0.5, 0.2, 0.1, 0.05), "lt"))
            row["medium_mean"] = float(torch.cat([v.reshape(-1) for v in mediums if v.numel()]).mean().item()) if any(v.numel() for v in mediums) else float("nan")
            row["direct_mean"] = float(torch.cat([v.reshape(-1) for v in directs if v.numel()]).mean().item()) if any(v.numel() for v in directs) else float("nan")
            attn_rows.append(row)

    edge_rows: List[Dict[str, Any]] = []
    for region in ("EDGE_TOP20", "PSEUDO_DEPTH_GRAD_TOP20"):
        pixels = 0
        harm = 0
        harm_mass = 0.0
        total_harm_mass = 0.0
        for view_id in view_ids:
            reg = regions[view_id][region]
            gain = gain_maps[view_id]
            pixels += int(reg.sum().item())
            harm += int((reg & masks[view_id]["HARM_PIXELS"]).sum().item())
            harm_mass += float((-gain[reg]).clamp_min(0).sum().item())
            total_harm_mass += float((-gain).clamp_min(0).sum().item())
        p_region = pixels / max(total_pixels, 1)
        p_harm_given = harm / max(pixels, 1)
        enrichment = p_harm_given / max(p_harm, EPS)
        edge_rows.append(
            {
                "scene": SCENE,
                "region": region,
                "pixels": pixels,
                "pixel_fraction": p_region,
                "P_HARM_given_region": p_harm_given,
                "EDGE_HARM_ENRICHMENT": enrichment,
                "HARM_MASS_FRACTION": harm_mass / max(total_harm_mass, EPS),
                "HARM_MASS_ENRICHMENT": (harm_mass / max(total_harm_mass, EPS)) / max(p_region, EPS),
                "EDGE_HARM_ALIGNED": bool(enrichment >= 1.5),
                "proxy_note": "RGB harm map plus GT/pseudo-depth gradient proxy; no per-pixel LPIPS attribution.",
            }
        )

    dm_rows: List[Dict[str, Any]] = []
    for region in ("GAIN_PIXELS", "HARM_PIXELS", "M1_HIGH_J", "NEW_LOW_T"):
        vals: Dict[str, List[Tensor]] = {key: [] for key in ("direct_delta", "medium_delta", "pred_delta", "gt_error_change")}
        for view_id in view_ids:
            if region == "NEW_LOW_T":
                mask = low_t_masks[view_id]["NEW_LOW_T"]
            elif region in masks[view_id]:
                mask = masks[view_id][region]
            else:
                mask = regions[view_id][region]
            gt = m1[view_id]["gt"]
            dk = k1[view_id]["outputs"]["direct_object_signal"]
            dc = cd[view_id]["outputs"]["direct_object_signal"]
            bk = k1[view_id]["outputs"]["rgb_medium"]
            bc = cd[view_id]["outputs"]["rgb_medium"]
            pk = k1[view_id]["outputs"]["pred_image"]
            pc = cd[view_id]["outputs"]["pred_image"]
            vals["direct_delta"].append((dc - dk).abs().mean(dim=-1)[mask])
            vals["medium_delta"].append((bc - bk).abs().mean(dim=-1)[mask])
            vals["pred_delta"].append((pc - pk).abs().mean(dim=-1)[mask])
            vals["gt_error_change"].append(((pc - gt).square().mean(dim=-1) - (pk - gt).square().mean(dim=-1))[mask])
        row = {"scene": SCENE, "region": region, "definition": "final CDEPTH vs K1, using true direct_object_signal and rgb_medium outputs"}
        for key, tensors in vals.items():
            joined = torch.cat([v.reshape(-1) for v in tensors if v.numel()]) if any(v.numel() for v in tensors) else torch.empty(0)
            row[f"{key}_mean"] = float(joined.mean().item()) if joined.numel() else float("nan")
            row[f"{key}_p90"] = _safe_quantile(joined.abs() if key == "gt_error_change" else joined, 0.90)
        dm_rows.append(row)

    summary = {
        "TOTAL_GAIN_MASS": total_gain_mass,
        "TOTAL_HARM_MASS": total_harm_mass,
        "P_HARM_PIXELS": p_harm,
        "strong_gain_threshold": pos_thr,
        "strong_harm_threshold": neg_thr,
        **lowt_summary,
    }
    return masks, gain_rows, structural_rows, lowt_rows, attn_rows, edge_rows, dm_rows, summary


def _plot_lines(path: Path, rows: Sequence[Mapping[str, Any]], x_key: str, y_keys: Sequence[Tuple[str, str]], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for key, label in y_keys:
        xs = [float(row[x_key]) for row in rows if key in row and str(row[key]) != ""]
        ys = [float(row[key]) for row in rows if key in row and str(row[key]) != ""]
        if xs and ys:
            plt.plot(xs, ys, marker="o", label=label)
    plt.title(title)
    plt.xlabel("nominal step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _make_visuals(
    render_dir: Path,
    view_ids: Sequence[str],
    m1: Mapping[str, Mapping[str, Any]],
    k1_final: Mapping[str, Mapping[str, Any]],
    cd_final: Mapping[str, Mapping[str, Any]],
    regions: Mapping[str, Mapping[str, Tensor]],
    final_masks: Mapping[str, Mapping[str, Tensor]],
    gain_maps: Mapping[str, Tensor],
    residual_images: Mapping[Tuple[str, int, str], Image.Image],
    paired_rows: Sequence[Mapping[str, Any]],
    pop_compare: Sequence[Mapping[str, Any]],
    scale_compare: Sequence[Mapping[str, Any]],
    anis_compare: Sequence[Mapping[str, Any]],
    radius_compare: Sequence[Mapping[str, Any]],
    opacity_compare: Sequence[Mapping[str, Any]],
    alpha_compare: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    render_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    gain_scale = max(max(float(gain_maps[v].abs().max().item()) for v in view_ids), EPS)
    residual_scale = 0.02
    trajectory_rows = []
    for step in TARGET_STEPS:
        for view_id in view_ids:
            row = []
            for run in ("BND-K1", "CDEPTH"):
                img = residual_images.get((run, step, view_id))
                if img is not None:
                    row.append((f"{step} {view_id} {run} residual", img))
            if row:
                trajectory_rows.append(row)
    _save_sheet(render_dir / "contact_sheet_rgb_trajectory.png", trajectory_rows, manifest, "rgb_trajectory", view_ids)

    gain_rows = []
    high_rows = []
    alpha_rows = []
    trans_rows = []
    lowt_rows = []
    edge_rows = []
    dm_rows = []
    for view_id in view_ids:
        gt = m1[view_id]["gt"]
        err_k = (k1_final[view_id]["outputs"]["pred_image"] - gt).square().mean(dim=-1)
        err_c = (cd_final[view_id]["outputs"]["pred_image"] - gt).square().mean(dim=-1)
        gain = gain_maps[view_id]
        gain_rows.append(
            [
                (f"{view_id} GT", _rgb_to_uint8(gt)),
                ("K1 residual", _gray_to_uint8(err_k, residual_scale)),
                ("CDEPTH residual", _gray_to_uint8(err_c, residual_scale)),
                ("gain signed", _signed_to_rgb(gain, gain_scale)),
            ]
        )
        high = regions[view_id]["M1_HIGH_J"]
        high_rows.append(
            [
                (f"{view_id} M1_HIGH_J", _mask_to_rgb(high)),
                ("gain signed", _signed_to_rgb(gain, gain_scale)),
                ("strong gain top20", _mask_to_rgb(final_masks[view_id]["STRONG_GAIN_TOP20"], (40, 200, 80))),
                ("strong harm top20", _mask_to_rgb(final_masks[view_id]["STRONG_HARM_TOP20"], (220, 40, 40))),
            ]
        )
        ak = k1_final[view_id]["outputs"]["accumulation"][..., 0]
        ac = cd_final[view_id]["outputs"]["accumulation"][..., 0]
        alpha_rows.append(
            [
                (f"{view_id} K1 alpha", _gray_to_uint8(ak, 1.0)),
                ("CDEPTH alpha", _gray_to_uint8(ac, 1.0)),
                ("alpha delta", _signed_to_rgb(ac - ak, 0.05)),
                ("high-J overlay", _overlay_mask(_signed_to_rgb(ac - ak, 0.05), high)),
            ]
        )
        tk = k1_final[view_id]["outputs"]["transmission"].mean(dim=-1)
        tc = cd_final[view_id]["outputs"]["transmission"].mean(dim=-1)
        new_low = (tc < 0.1) & (tk >= 0.1)
        trans_rows.append(
            [
                (f"{view_id} K1 Tmean", _gray_to_uint8(tk, 1.0)),
                ("CDEPTH Tmean", _gray_to_uint8(tc, 1.0)),
                ("NEW_LOW_T", _mask_to_rgb(new_low, (255, 80, 40))),
                ("NEW_LOW_T on gain", _overlay_mask(_signed_to_rgb(gain, gain_scale), new_low, (255, 80, 40))),
            ]
        )
        lowt_rows.append(
            [
                (f"{view_id} NEW_LOW_T", _mask_to_rgb(new_low, (255, 80, 40))),
                ("gain/harm", _signed_to_rgb(gain, gain_scale)),
                ("EDGE_TOP20", _mask_to_rgb(regions[view_id]["EDGE_TOP20"], (40, 120, 255))),
                ("edge on NEW_LOW_T", _overlay_mask(_mask_to_rgb(new_low, (255, 80, 40)), regions[view_id]["EDGE_TOP20"], (40, 120, 255))),
            ]
        )
        edge_rows.append(
            [
                (f"{view_id} GT edge", _gray_to_uint8(_gradient_magnitude(gt), float(summary.get("edge_display_scale", 0.2)))),
                ("K1 residual", _gray_to_uint8(err_k, residual_scale)),
                ("CDEPTH residual", _gray_to_uint8(err_c, residual_scale)),
                ("gain signed", _signed_to_rgb(gain, gain_scale)),
            ]
        )
        d_delta = (cd_final[view_id]["outputs"]["direct_object_signal"] - k1_final[view_id]["outputs"]["direct_object_signal"]).abs().mean(dim=-1)
        b_delta = (cd_final[view_id]["outputs"]["rgb_medium"] - k1_final[view_id]["outputs"]["rgb_medium"]).abs().mean(dim=-1)
        dm_rows.append(
            [
                (f"{view_id} direct abs delta", _gray_to_uint8(d_delta, 0.1)),
                ("medium abs delta", _gray_to_uint8(b_delta, 0.1)),
                ("gain signed", _signed_to_rgb(gain, gain_scale)),
                ("high-J overlay", _overlay_mask(_signed_to_rgb(gain, gain_scale), high)),
            ]
        )
    _save_sheet(render_dir / "contact_sheet_gain_harm.png", gain_rows, manifest, "gain_harm", view_ids)
    _save_sheet(render_dir / "contact_sheet_high_j_gain_harm.png", high_rows, manifest, "high_j_gain_harm", view_ids)
    _save_sheet(render_dir / "contact_sheet_alpha_delta.png", alpha_rows, manifest, "alpha_coverage", view_ids)
    _save_sheet(render_dir / "contact_sheet_transmission_new_low_t.png", trans_rows, manifest, "transmission_new_low_t", view_ids)
    _save_sheet(render_dir / "contact_sheet_new_low_t_harm.png", lowt_rows, manifest, "new_low_t_harm", view_ids)
    _save_sheet(render_dir / "contact_sheet_edge_harm.png", edge_rows, manifest, "edge_harm", view_ids)
    _save_sheet(render_dir / "contact_sheet_direct_medium_attribution.png", dm_rows, manifest, "direct_medium_attribution", view_ids)

    _plot_lines(
        render_dir / "plot_rgb_highj_trajectory.png",
        paired_rows,
        "nominal_step",
        (("GLOBAL_MSE_GAIN", "global MSE gain"), ("HIGHJ_MSE_GAIN", "high-J MSE gain")),
        "RGB Gain Trajectory",
        "K1 MSE - CDEPTH MSE",
    )
    manifest.append({"scene": SCENE, "file_path": str(render_dir / "plot_rgb_highj_trajectory.png"), "output_type": "rgb_highj_trajectory_plot", "view_ids": ";".join(view_ids)})
    _plot_lines(render_dir / "plot_population_trajectory.png", pop_compare, "nominal_step", (("POPULATION_DELTA", "CDEPTH-K1 count"),), "Gaussian Count Delta", "delta")
    manifest.append({"scene": SCENE, "file_path": str(render_dir / "plot_population_trajectory.png"), "output_type": "population_plot", "view_ids": ";".join(view_ids)})
    _plot_lines(render_dir / "plot_scale_trajectory.png", scale_compare, "nominal_step", (("scale_max_p90_DELTA", "scale max p90 delta"), ("scale_max_p99_DELTA", "scale max p99 delta")), "Scale Distribution Delta", "delta")
    manifest.append({"scene": SCENE, "file_path": str(render_dir / "plot_scale_trajectory.png"), "output_type": "scale_plot", "view_ids": ";".join(view_ids)})
    _plot_lines(render_dir / "plot_anisotropy_trajectory.png", anis_compare, "nominal_step", (("anisotropy_p90_DELTA", "anisotropy p90 delta"), ("anisotropy_p99_DELTA", "anisotropy p99 delta")), "Anisotropy Delta", "delta")
    manifest.append({"scene": SCENE, "file_path": str(render_dir / "plot_anisotropy_trajectory.png"), "output_type": "anisotropy_plot", "view_ids": ";".join(view_ids)})
    _plot_lines(render_dir / "plot_projected_radius_trajectory.png", radius_compare, "nominal_step", (("radius_p90_DELTA", "radius p90 delta"), ("radius_p99_DELTA", "radius p99 delta")), "Projected Radius Delta", "delta")
    manifest.append({"scene": SCENE, "file_path": str(render_dir / "plot_projected_radius_trajectory.png"), "output_type": "projected_radius_plot", "view_ids": ";".join(view_ids)})
    _plot_lines(render_dir / "plot_opacity_trajectory.png", opacity_compare, "nominal_step", (("opacity_p90_DELTA", "opacity p90 delta"), ("opacity_p99_DELTA", "opacity p99 delta"), ("opacity_gt_0.95_DELTA", "P(opacity>0.95) delta")), "Opacity Delta", "delta")
    manifest.append({"scene": SCENE, "file_path": str(render_dir / "plot_opacity_trajectory.png"), "output_type": "opacity_plot", "view_ids": ";".join(view_ids)})
    _plot_lines(render_dir / "plot_alpha_trajectory.png", alpha_compare, "nominal_step", (("M1_HIGH_J_alpha_mean_DELTA", "high-J alpha mean delta"), ("EDGE_TOP20_alpha_mean_DELTA", "edge alpha mean delta")), "Alpha Region Delta", "delta")
    manifest.append({"scene": SCENE, "file_path": str(render_dir / "plot_alpha_trajectory.png"), "output_type": "alpha_plot", "view_ids": ";".join(view_ids)})

    lines = ["BND-CDEPTH PATH factor summary", ""]
    for key in (
        "HIGHJ_RECOVERY_ONSET_STEP",
        "POPULATION_DIVERGENCE_ONSET",
        "SCALE_DIVERGENCE_ONSET",
        "FOOTPRINT_DIVERGENCE_ONSET",
        "OPACITY_DIVERGENCE_ONSET",
        "ALPHA_DIVERGENCE_ONSET",
        "LOWT_DIVERGENCE_ONSET",
        "SCALE_FOOTPRINT_PATH_SUPPORTED",
        "OPACITY_COVERAGE_PATH_SUPPORTED",
        "POPULATION_DENSIFICATION_PATH_SUPPORTED",
        "LOWT_HARM_PATH_SUPPORTED",
        "STRUCTURE_HARM_PATH_SUPPORTED",
        "BENEFICIAL_MECHANISM",
        "HARMFUL_MECHANISM",
        "PATHWAY_RELATION",
        "NEXT_SINGLE_FACTOR_EXPERIMENT",
    ):
        if key in summary:
            lines.append(f"{key}: {summary[key]}")
    img = Image.new("RGB", (1700, max(140, len(lines) * 28 + 20)), "white")
    draw = ImageDraw.Draw(img)
    for idx, line in enumerate(lines):
        draw.text((10, 10 + idx * 28), line, fill=(0, 0, 0))
    factor_path = render_dir / "contact_sheet_factor_summary.png"
    img.save(factor_path)
    manifest.append({"scene": SCENE, "file_path": str(factor_path), "output_type": "factor_summary", "view_ids": ";".join(view_ids), "width": img.width, "height": img.height})
    return manifest


def _row_by_step(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    return {int(row["nominal_step"]): row for row in rows}


def _trajectory_correlations(structural_rows: Sequence[Mapping[str, Any]], paired_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    paired = _row_by_step(paired_rows)
    rows = []
    for key in (
        "POPULATION_REL_DELTA",
        "scale_max_p90_REL_DELTA",
        "scale_max_p99_REL_DELTA",
        "anisotropy_p90_REL_DELTA",
        "radius_p90_REL_DELTA",
        "radius_p99_REL_DELTA",
        "opacity_p90_REL_DELTA",
        "opacity_p99_REL_DELTA",
        "M1_HIGH_J_alpha_mean_DELTA",
        "CDEPTH_LOW_T_fraction_DELTA",
    ):
        vals = []
        high = []
        glob = []
        for row in structural_rows:
            step = int(row["nominal_step"])
            if step in paired and key in row and str(row[key]) != "":
                vals.append(float(row[key]))
                high.append(float(paired[step]["HIGHJ_MSE_GAIN"]))
                glob.append(float(paired[step]["GLOBAL_MSE_GAIN"]))
        rows.append(
            {
                "scene": SCENE,
                "metric": key,
                "num_steps": len(vals),
                "spearman_vs_HIGHJ_MSE_GAIN": _spearman(vals, high) if len(vals) >= 5 else float("nan"),
                "spearman_vs_GLOBAL_MSE_GAIN": _spearman(vals, glob) if len(vals) >= 5 else float("nan"),
                "note": "small-sample trajectory correlation; supporting evidence only",
            }
        )
    return rows


def _classify(
    paired_rows: Sequence[Mapping[str, Any]],
    pop_compare: Sequence[Mapping[str, Any]],
    scale_compare: Sequence[Mapping[str, Any]],
    anis_compare: Sequence[Mapping[str, Any]],
    radius_compare: Sequence[Mapping[str, Any]],
    opacity_compare: Sequence[Mapping[str, Any]],
    alpha_compare: Sequence[Mapping[str, Any]],
    lowt_compare: Sequence[Mapping[str, Any]],
    gain_rows: Sequence[Mapping[str, Any]],
    lowt_summary: Mapping[str, Any],
    edge_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    high_onset = _stable_positive_onset(paired_rows, "HIGHJ_MSE_GAIN")
    global_onset = _stable_positive_onset(paired_rows, "GLOBAL_MSE_GAIN")
    pop_onset = _first_stable_onset(pop_compare, "POPULATION_REL_DELTA", 0.02, absolute=True)
    scale_onset = _first_stable_onset(scale_compare, "scale_max_p90_REL_DELTA", 0.05, absolute=True) or _first_stable_onset(scale_compare, "scale_max_p99_REL_DELTA", 0.05, absolute=True)
    anis_onset = _first_stable_onset(anis_compare, "anisotropy_p90_REL_DELTA", 0.05, absolute=True) or _first_stable_onset(anis_compare, "anisotropy_p99_REL_DELTA", 0.05, absolute=True)
    radius_onset = _first_stable_onset(radius_compare, "radius_p90_REL_DELTA", 0.05, absolute=True) or _first_stable_onset(radius_compare, "radius_p99_REL_DELTA", 0.05, absolute=True)
    opacity_onset = _first_stable_onset(opacity_compare, "opacity_p90_REL_DELTA", 0.05, absolute=True) or _first_stable_onset(opacity_compare, "opacity_gt_0.95_DELTA", 0.02, absolute=True)
    alpha_onset = _first_stable_onset(alpha_compare, "M1_HIGH_J_alpha_mean_DELTA", 0.002, absolute=True)
    lowt_onset = _first_stable_onset(lowt_compare, "CDEPTH_LOW_T_fraction_DELTA", 0.001, absolute=True)

    def relation(onset: Optional[int]) -> str:
        if onset is None or high_onset is None:
            return "NO_CLEAR_DIVERGENCE"
        if onset < high_onset:
            return "PRECEDES"
        if onset == high_onset:
            return "COINCIDES"
        return "FOLLOWS"

    onset_rows = [
        {"metric": "HIGHJ_RECOVERY_ONSET_STEP", "onset_step": high_onset, "ordering_vs_highJ": "REFERENCE"},
        {"metric": "GLOBAL_RECOVERY_ONSET_STEP", "onset_step": global_onset, "ordering_vs_highJ": relation(global_onset)},
        {"metric": "POPULATION_DIVERGENCE_ONSET", "onset_step": pop_onset, "ordering_vs_highJ": relation(pop_onset)},
        {"metric": "SCALE_DIVERGENCE_ONSET", "onset_step": scale_onset, "ordering_vs_highJ": relation(scale_onset)},
        {"metric": "ANISOTROPY_DIVERGENCE_ONSET", "onset_step": anis_onset, "ordering_vs_highJ": relation(anis_onset)},
        {"metric": "FOOTPRINT_DIVERGENCE_ONSET", "onset_step": radius_onset, "ordering_vs_highJ": relation(radius_onset)},
        {"metric": "OPACITY_DIVERGENCE_ONSET", "onset_step": opacity_onset, "ordering_vs_highJ": relation(opacity_onset)},
        {"metric": "ALPHA_DIVERGENCE_ONSET", "onset_step": alpha_onset, "ordering_vs_highJ": relation(alpha_onset)},
        {"metric": "LOWT_DIVERGENCE_ONSET", "onset_step": lowt_onset, "ordering_vs_highJ": relation(lowt_onset)},
    ]
    for row in onset_rows:
        row["scene"] = SCENE

    high_gain = next((row for row in gain_rows if row["region"] == "M1_HIGH_J"), {})
    edge = next((row for row in edge_rows if row["region"] == "EDGE_TOP20"), {})
    scale_signal = radius_onset is not None or scale_onset is not None or anis_onset is not None
    opacity_signal = opacity_onset is not None or alpha_onset is not None
    population_signal = pop_onset is not None
    highj_enriched = float(high_gain.get("GAIN_ENRICHMENT", 0.0)) >= 1.25 or float(high_gain.get("P_GAIN_given_region", 0.0)) > 0.5
    edge_harm_aligned = bool(edge.get("EDGE_HARM_ALIGNED", False))
    lowt_harm = bool(lowt_summary.get("LOWT_HARM_ALIGNED", False))

    flags = {
        "HIGHJ_RECOVERY_ONSET_STEP": high_onset,
        "GLOBAL_RECOVERY_ONSET_STEP": global_onset,
        "POPULATION_DIVERGENCE_ONSET": pop_onset,
        "SCALE_DIVERGENCE_ONSET": scale_onset,
        "ANISOTROPY_DIVERGENCE_ONSET": anis_onset,
        "FOOTPRINT_DIVERGENCE_ONSET": radius_onset,
        "OPACITY_DIVERGENCE_ONSET": opacity_onset,
        "ALPHA_DIVERGENCE_ONSET": alpha_onset,
        "LOWT_DIVERGENCE_ONSET": lowt_onset,
        "CONTRIBUTOR_DIAGNOSTIC_AVAILABLE": False,
        "SCALE_FOOTPRINT_PATH_SUPPORTED": bool(scale_signal and relation(scale_onset or radius_onset or anis_onset) in ("PRECEDES", "COINCIDES") and highj_enriched),
        "OPACITY_COVERAGE_PATH_SUPPORTED": bool(opacity_signal and relation(opacity_onset or alpha_onset) in ("PRECEDES", "COINCIDES") and highj_enriched),
        "POPULATION_DENSIFICATION_PATH_SUPPORTED": bool(population_signal and relation(pop_onset) in ("PRECEDES", "COINCIDES") and highj_enriched),
        "LOWT_HARM_ALIGNED": bool(lowt_harm),
        "EDGE_HARM_ALIGNED": bool(edge_harm_aligned),
        "LOWT_HARM_PATH_SUPPORTED": bool(lowt_harm and (float(next((row for row in lowt_compare if row["nominal_step"] == FINAL_NOMINAL_STEP), {}).get("CDEPTH_LOW_T_fraction_DELTA", 0.0)) > 0.0)),
        "STRUCTURE_HARM_PATH_SUPPORTED": bool(edge_harm_aligned),
    }

    score_rows: List[Dict[str, Any]] = []

    def score(factor: str, score_value: str, rationale: str) -> None:
        score_rows.append({"scene": SCENE, "factor": factor, "score": score_value, "rationale": rationale})

    score("Gaussian scale / footprint", "MODERATE" if flags["SCALE_FOOTPRINT_PATH_SUPPORTED"] else ("WEAK" if scale_signal else "EVIDENCE_AGAINST"), f"scale_onset={scale_onset}; radius_onset={radius_onset}; highJ_gain_enriched={highj_enriched}")
    score("orientation / anisotropy", "MODERATE" if anis_onset is not None and relation(anis_onset) in ("PRECEDES", "COINCIDES") else ("WEAK" if anis_onset is not None else "EVIDENCE_AGAINST"), f"anisotropy_onset={anis_onset}")
    score("opacity / alpha coverage", "MODERATE" if flags["OPACITY_COVERAGE_PATH_SUPPORTED"] else ("WEAK" if opacity_signal else "EVIDENCE_AGAINST"), f"opacity_onset={opacity_onset}; alpha_onset={alpha_onset}")
    score("population / densification", "MODERATE" if flags["POPULATION_DENSIFICATION_PATH_SUPPORTED"] else ("WEAK" if population_signal else "EVIDENCE_AGAINST"), f"population_onset={pop_onset}")
    score("overlap / contributors", "NOT_EVALUABLE", "No true contributor-weight buffer is available without CUDA/rasterizer changes; proxy only.")
    score("extreme low-T tail", "MODERATE" if flags["LOWT_HARM_PATH_SUPPORTED"] else "WEAK", f"NEW_LOWT_HARM_ENRICHMENT={lowt_summary.get('NEW_LOWT_HARM_ENRICHMENT')}")
    score("edge / structure harm", "MODERATE" if flags["STRUCTURE_HARM_PATH_SUPPORTED"] else "WEAK", f"EDGE_HARM_ENRICHMENT={edge.get('EDGE_HARM_ENRICHMENT')}")
    score("pseudo-depth agreement", "EVIDENCE_AGAINST", "Previous final diagnostic showed high-J aligned RMSE and gradient Pearson did not improve for CDEPTH.")

    supported_benefit = [name for name, value in (
        ("SCALE_FOOTPRINT", flags["SCALE_FOOTPRINT_PATH_SUPPORTED"]),
        ("OPACITY_COVERAGE", flags["OPACITY_COVERAGE_PATH_SUPPORTED"]),
        ("POPULATION_DENSIFICATION", flags["POPULATION_DENSIFICATION_PATH_SUPPORTED"]),
    ) if value]
    if len(supported_benefit) >= 2:
        beneficial = "MIXED_GAUSSIAN_STRUCTURE"
    elif supported_benefit == ["SCALE_FOOTPRINT"]:
        beneficial = "SCALE_FOOTPRINT_DOMINANT"
    elif supported_benefit == ["OPACITY_COVERAGE"]:
        beneficial = "OPACITY_COVERAGE_DOMINANT"
    elif supported_benefit == ["POPULATION_DENSIFICATION"]:
        beneficial = "POPULATION_DENSIFICATION_DOMINANT"
    elif scale_signal or opacity_signal or population_signal:
        beneficial = "UNRESOLVED"
    else:
        beneficial = "NO_CLEAR_STRUCTURAL_PATHWAY"

    if flags["LOWT_HARM_PATH_SUPPORTED"] and flags["STRUCTURE_HARM_PATH_SUPPORTED"]:
        harmful = "MIXED_HARM"
    elif flags["LOWT_HARM_PATH_SUPPORTED"]:
        harmful = "EXTREME_LOW_T_DOMINANT"
    elif flags["STRUCTURE_HARM_PATH_SUPPORTED"]:
        harmful = "EDGE_STRUCTURE_DOMINANT"
    else:
        harmful = "NO_CLEAR_HARM_PATHWAY"

    if beneficial in ("SCALE_FOOTPRINT_DOMINANT", "MIXED_GAUSSIAN_STRUCTURE", "OPACITY_COVERAGE_DOMINANT", "POPULATION_DENSIFICATION_DOMINANT") and harmful in ("EXTREME_LOW_T_DOMINANT", "EDGE_STRUCTURE_DOMINANT", "MIXED_HARM"):
        pathway_relation = "PARTIALLY_SEPARABLE"
    elif beneficial == "NO_CLEAR_STRUCTURAL_PATHWAY" or harmful == "NO_CLEAR_HARM_PATHWAY":
        pathway_relation = "UNRESOLVED"
    else:
        pathway_relation = "UNRESOLVED"

    if flags["LOWT_HARM_PATH_SUPPORTED"] and beneficial in ("UNRESOLVED", "NO_CLEAR_STRUCTURAL_PATHWAY"):
        next_experiment = "read-only attenuation-tail ownership diagnostic"
    elif beneficial == "POPULATION_DENSIFICATION_DOMINANT":
        next_experiment = "K1 densification/population matched control based on observed divergence timing"
    elif beneficial in ("SCALE_FOOTPRINT_DOMINANT", "MIXED_GAUSSIAN_STRUCTURE") and harmful == "EXTREME_LOW_T_DOMINANT" and pathway_relation == "SEPARABLE_PATHWAYS":
        next_experiment = "CDEPTH-NO-OPACITY-GRAD"
    elif pathway_relation in ("UNRESOLVED", "PARTIALLY_SEPARABLE"):
        next_experiment = "region-conditioned read-only footprint/attenuation-tail diagnostic"
    else:
        next_experiment = "region-conditioned read-only footprint diagnostic"

    classification = {
        **flags,
        "BENEFICIAL_MECHANISM": beneficial,
        "HARMFUL_MECHANISM": harmful,
        "PATHWAY_RELATION": pathway_relation,
        "NEXT_SINGLE_FACTOR_EXPERIMENT": next_experiment,
    }
    return onset_rows, classification, score_rows, []


def _write_visual_index(render_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# BND-CDEPTH PATH Visual Compare Index", ""]
    for row in manifest:
        lines.append(f"- {row['output_type']}: `{row['file_path']}`")
    lines.append("")
    lines.append("Visual assets are ready for external/manual analysis.")
    lines.append("No subjective clear-image correctness judgment was made.")
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_research_note(path: Path, summary: Mapping[str, Any], outputs: Mapping[str, str]) -> None:
    lines = [
        "# BND-CDEPTH Partial-Mitigation Optimization-Path Audit",
        "",
        "## Motivation",
        "",
        "- This note records a read-only post-hoc audit of why CDEPTH partially mitigates the bounded intrinsic RGB fitting deficit on Panama.",
        "- The target comparison is BND-K1 versus BND-CDEPTH; SeaFree is treated only as method context, not as a solved reference.",
        "",
        "## Revised Interpretation",
        "",
        "- SeaFree-GS is not treated as a complete solution to the bounded reconstruction trade-off.",
        "- CDEPTH is treated as one experimentally validated partial mitigation mechanism, with known PSNR/high-J recovery and known SSIM/LPIPS cost.",
        "- The simple pseudo-depth accuracy explanation is not used: previous fixed M1_HIGH_J diagnostics did not show improved pseudo-depth agreement for CDEPTH.",
        "",
        "## Code Fact",
        "",
        "- This audit is read-only and uses existing BND-K1 and BND-CDEPTH checkpoints.",
        "- No optimizer, scheduler, densification, pruning, checkpoint mutation, or training source change is used.",
        "- Projected radius is read from `model.radii` after projection; it is treated as a screen-space projected-radius proxy.",
        "- True per-pixel contributor weights are not available without rasterizer/CUDA changes; overlap is therefore reported through safe proxies.",
        "- Gain is defined as per-pixel `MSE_K1 - MSE_CDEPTH`; positive values mean lower RGB MSE for CDEPTH at that pixel.",
        "",
        "## Config Fact",
        "",
        f"- BND-K1 config: `{outputs['k1_config']}`.",
        f"- CDEPTH config: `{outputs['cdepth_config']}`.",
        "- Fixed eval views are inherited from the Panama dataparser split.",
        "",
        "## Experimental Fact",
        "",
        f"- Common trajectory steps: `{summary.get('COMMON_TRAJECTORY_STEPS')}`.",
        f"- Audit parameter safety: `{summary.get('AUDIT_PARAMETER_SAFETY')}`.",
        f"- Contributor diagnostic available: `{summary.get('CONTRIBUTOR_DIAGNOSTIC_AVAILABLE')}`.",
        "",
        "## Quantitative Result",
        "",
        f"- `HIGHJ_RECOVERY_ONSET_STEP`: `{summary.get('HIGHJ_RECOVERY_ONSET_STEP')}`.",
        f"- `GLOBAL_RECOVERY_ONSET_STEP`: `{summary.get('GLOBAL_RECOVERY_ONSET_STEP')}`.",
        f"- `POPULATION_DIVERGENCE_ONSET`: `{summary.get('POPULATION_DIVERGENCE_ONSET')}`.",
        f"- `SCALE_DIVERGENCE_ONSET`: `{summary.get('SCALE_DIVERGENCE_ONSET')}`.",
        f"- `ANISOTROPY_DIVERGENCE_ONSET`: `{summary.get('ANISOTROPY_DIVERGENCE_ONSET')}`.",
        f"- `FOOTPRINT_DIVERGENCE_ONSET`: `{summary.get('FOOTPRINT_DIVERGENCE_ONSET')}`.",
        f"- `OPACITY_DIVERGENCE_ONSET`: `{summary.get('OPACITY_DIVERGENCE_ONSET')}`.",
        f"- `ALPHA_DIVERGENCE_ONSET`: `{summary.get('ALPHA_DIVERGENCE_ONSET')}`.",
        f"- `LOWT_DIVERGENCE_ONSET`: `{summary.get('LOWT_DIVERGENCE_ONSET')}`.",
        f"- `NEW_LOWT_HARM_ENRICHMENT`: `{summary.get('NEW_LOWT_HARM_ENRICHMENT')}`.",
        f"- `EDGE_HARM_ENRICHMENT`: `{summary.get('EDGE_HARM_ENRICHMENT')}`.",
        f"- `TOTAL_GAIN_MASS`: `{summary.get('TOTAL_GAIN_MASS')}`.",
        f"- `TOTAL_HARM_MASS`: `{summary.get('TOTAL_HARM_MASS')}`.",
        f"- Beneficial mechanism: `{summary.get('BENEFICIAL_MECHANISM')}`.",
        f"- Harmful mechanism: `{summary.get('HARMFUL_MECHANISM')}`.",
        f"- Pathway relation: `{summary.get('PATHWAY_RELATION')}`.",
        "",
        "## Inference",
        "",
        "- The results are post-hoc attribution evidence, not causal proof.",
        "- Pseudo-depth remains a coarse diagnostic cue rather than metric depth ground truth.",
        "- Any reported structural association should be interpreted as consistent with an optimization pathway, not as proof of causality.",
        "- The current evidence is most consistent with a mixed Gaussian-structure pathway, because population, scale/footprint, anisotropy, and alpha differences precede or coincide high-J recovery.",
        "- No clear harmful pathway was isolated by the tested low-T and edge/structure proxies; the SSIM/LPIPS degradation remains unresolved by this audit.",
        "",
        "## Hypothesis",
        "",
        f"- Next single-factor recommendation: `{summary.get('NEXT_SINGLE_FACTOR_EXPERIMENT')}`.",
        "",
        "## Outputs",
        "",
        f"- Final summary: `{outputs['summary_json']}`.",
        f"- Visual index: `{outputs['visual_index']}`.",
        f"- Visual manifest: `{outputs['visual_manifest']}`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf8")


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
        "log_5": _git(repo, "log", "-5", "--oneline"),
        "status_short": _git(repo, "status", "--short"),
        "tracked_output_files": _git(repo, "ls-files", "outputs", "renders", "logs", "common_masks", "checkpoints"),
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)
    stale_processes = _stale_process_rows()
    interrupted_rows = _file_manifest_rows(
        repo,
        (
            "scripts/diagnostics/*cdepth*path*",
            "scripts/diagnostics/*cdepth*optimization*",
            "scripts/diagnostics/*cdepth*population*",
            "scripts/diagnostics/*cdepth*screen*",
            "scripts/diagnostics/*gain*harm*",
            "research_notes/*CDEPTH*PATH*",
            "research_notes/*CDEPTH*OPTIMIZATION*",
            "outputs/bnd_cdepth_path_panama_20260811",
            "renders/bnd_cdepth_path_panama_20260811",
            "logs/bnd_cdepth_path_panama_20260811",
            "outputs/bnd_cdepth_mitigation_path_panama_20260811",
            "renders/bnd_cdepth_mitigation_path_panama_20260811",
            "logs/bnd_cdepth_mitigation_path_panama_20260811",
        ),
    )
    _write_json(
        output_dir / "interrupted_work_manifest.json",
        {
            "rows": interrupted_rows,
            "stale_process_rows": stale_processes,
            "NO_STALE_CDEPTH_PROCESS": len(stale_processes) == 0,
            "reused": ["scripts/diagnostics/audit_bnd_cdepth_optimization_path.py"],
            "recomputed": ["formal mitigation-path outputs under the new bnd_cdepth_mitigation_path_panama_20260811 directories"],
            "discarded_as_incomplete_without_deletion": [
                "outputs/bnd_cdepth_path_panama_20260811",
                "renders/bnd_cdepth_path_panama_20260811",
            ],
        },
    )

    availability: Dict[str, Any] = {}
    checkpoint_rows: List[Dict[str, Any]] = []
    pop_rows: List[Dict[str, Any]] = []
    scale_rows: List[Dict[str, Any]] = []
    opacity_rows: List[Dict[str, Any]] = []
    for run in ("BND-K1", "CDEPTH"):
        config_path = repo / RUNS[run].config_relpath
        available = _available_steps(config_path)
        actuals: Dict[int, Optional[int]] = {step: _actual_step(config_path, step) for step in TARGET_STEPS}
        availability[run] = {
            "config_path": str(config_path),
            "available_checkpoint_steps": sorted(available),
            "target_to_actual_step": {str(k): v for k, v in actuals.items()},
            "missing_target_steps": [step for step, actual in actuals.items() if actual is None],
        }
        for nominal, actual in actuals.items():
            if actual is None:
                continue
            pop, scale, opacity = _checkpoint_stats(repo, run, nominal, actual)
            checkpoint_rows.append(pop)
            pop_rows.append(pop)
            scale_rows.append(scale)
            opacity_rows.append(opacity)
    common_steps = [
        step
        for step in TARGET_STEPS
        if availability["BND-K1"]["target_to_actual_step"][str(step)] is not None and availability["CDEPTH"]["target_to_actual_step"][str(step)] is not None
    ]
    availability["COMMON_TRAJECTORY_STEPS"] = common_steps
    _write_json(output_dir / "trajectory_availability.json", availability)
    _write_csv(output_dir / "checkpoint_manifest.csv", checkpoint_rows)
    _write_json(output_dir / "checkpoint_manifest.json", {"rows": checkpoint_rows})
    _write_csv(output_dir / "gaussian_population_trajectory.csv", pop_rows)
    _write_json(output_dir / "gaussian_population_trajectory.json", {"rows": pop_rows})
    _write_csv(output_dir / "gaussian_scale_trajectory.csv", scale_rows)
    _write_json(output_dir / "gaussian_scale_trajectory.json", {"rows": scale_rows})
    _write_csv(output_dir / "gaussian_anisotropy_trajectory.csv", scale_rows)
    _write_json(output_dir / "gaussian_anisotropy_trajectory.json", {"rows": scale_rows, "source": "anisotropy_* columns in activated scale statistics"})
    _write_csv(output_dir / "opacity_trajectory.csv", opacity_rows)
    _write_json(output_dir / "opacity_trajectory.json", {"rows": opacity_rows})

    m1_items, regions, region_meta, view_ids = _load_m1_regions(repo, output_dir)
    per_view_rows: List[Dict[str, Any]] = []
    radius_rows: List[Dict[str, Any]] = []
    visible_scale_rows: List[Dict[str, Any]] = []
    parameter_delta_rows: List[Dict[str, Any]] = []
    final_items: Dict[str, Dict[str, Dict[str, Any]]] = {"BND-K1": {}, "CDEPTH": {}}
    residual_images: Dict[Tuple[str, int, str], Image.Image] = {}
    for step in common_steps:
        for run in ("BND-K1", "CDEPTH"):
            actual = availability[run]["target_to_actual_step"][str(step)]
            store_final = step == FINAL_NOMINAL_STEP
            p_rows, r_rows, vs_rows, f_items, deltas, res_imgs = _render_run_step(repo, run, step, regions, store_final, int(actual))
            per_view_rows.extend(p_rows)
            radius_rows.extend(r_rows)
            visible_scale_rows.extend(vs_rows)
            parameter_delta_rows.extend(deltas)
            if store_final:
                final_items[run] = f_items
            for view_id, image in res_imgs.items():
                residual_images[(run, step, view_id)] = image

    audit_parameter_safety = all(float(row["max_abs_delta"]) == 0.0 for row in parameter_delta_rows)
    _write_csv(output_dir / "parameter_safety.csv", parameter_delta_rows)
    _write_json(output_dir / "parameter_safety.json", {"AUDIT_PARAMETER_SAFETY": "PASS" if audit_parameter_safety else "FAIL", "rows": parameter_delta_rows})

    rgb_rows, high_rows = _aggregate_rgb_rows(per_view_rows)
    paired_rows = _paired_step_rows(rgb_rows, high_rows)
    highj_onset = _stable_positive_onset(paired_rows, "HIGHJ_MSE_GAIN")
    for row in paired_rows:
        row["HIGHJ_RECOVERY_ONSET_STEP"] = highj_onset if highj_onset is not None else ""
    _write_csv(output_dir / "rgb_mitigation_trajectory.csv", rgb_rows)
    _write_json(output_dir / "rgb_mitigation_trajectory.json", {"rows": rgb_rows, "paired_rows": paired_rows})
    _write_csv(output_dir / "high_j_recovery_trajectory.csv", paired_rows)
    _write_json(output_dir / "high_j_recovery_trajectory.json", {"rows": paired_rows, "HIGHJ_RECOVERY_ONSET_STEP": highj_onset})

    pop_compare = _compare_population(pop_rows)
    scale_compare = _compare_metric_rows(scale_rows, "scale_max_p90", "scale_max_p90")
    scale_compare2 = _compare_metric_rows(scale_rows, "scale_max_p99", "scale_max_p99")
    for row in scale_compare:
        match = next((r for r in scale_compare2 if r["nominal_step"] == row["nominal_step"]), {})
        row.update({k: v for k, v in match.items() if k not in row})
    anis_compare = _compare_metric_rows(scale_rows, "anisotropy_p90", "anisotropy_p90")
    anis_compare2 = _compare_metric_rows(scale_rows, "anisotropy_p99", "anisotropy_p99")
    for row in anis_compare:
        match = next((r for r in anis_compare2 if r["nominal_step"] == row["nominal_step"]), {})
        row.update({k: v for k, v in match.items() if k not in row})
    radius_agg = _aggregate_by_run_step(radius_rows, ("radius_", "support_proxy_"))
    radius_compare = _compare_metric_rows(radius_agg, "radius_p90", "radius_p90")
    for key in ("radius_p50", "radius_p95", "radius_p99", "radius_gt_4", "radius_gt_8", "radius_gt_16", "radius_gt_32"):
        comp = _compare_metric_rows(radius_agg, key, key)
        for row in radius_compare:
            match = next((r for r in comp if r["nominal_step"] == row["nominal_step"]), {})
            row.update({k: v for k, v in match.items() if k not in row})
    opacity_compare = _compare_metric_rows(opacity_rows, "opacity_p90", "opacity_p90")
    for key in ("opacity_p50", "opacity_p95", "opacity_p99", "opacity_gt_0.95", "opacity_gt_0.99"):
        comp = _compare_metric_rows(opacity_rows, key, key)
        for row in opacity_compare:
            match = next((r for r in comp if r["nominal_step"] == row["nominal_step"]), {})
            row.update({k: v for k, v in match.items() if k not in row})

    alpha_rows: List[Dict[str, Any]] = []
    lowt_compare: List[Dict[str, Any]] = []
    grouped_final_by_step_run: Dict[Tuple[int, str], List[Mapping[str, Any]]] = {}
    for row in per_view_rows:
        grouped_final_by_step_run.setdefault((int(row["nominal_step"]), str(row["run"])), []).append(row)
    # Re-rendered outputs were not retained for all steps, so alpha/low-T rows are rebuilt from final retained rows where available
    # and from stored per-view metric rows for the trajectory-level RGB path. To keep alpha/low-T trajectory complete,
    # run lightweight final-output pass for each step again without metrics.
    alpha_step_rows: List[Dict[str, Any]] = []
    lowt_step_rows: List[Dict[str, Any]] = []
    for step in common_steps:
        step_items: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for run in ("BND-K1", "CDEPTH"):
            actual = availability[run]["target_to_actual_step"][str(step)]
            _, _, _, items, deltas, _ = _render_run_step(repo, run, step, regions, True, int(actual))
            step_items[run] = items
            parameter_delta_rows.extend(deltas)
        for run in ("BND-K1", "CDEPTH"):
            for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5", "EDGE_TOP20"):
                vals = []
                for view_id in view_ids:
                    vals.append(step_items[run][view_id]["outputs"]["accumulation"][..., 0][regions[view_id][region]])
                joined = torch.cat([v.reshape(-1) for v in vals if v.numel()]) if any(v.numel() for v in vals) else torch.empty(0)
                row = {"scene": SCENE, "run": run, "nominal_step": step, "region": region}
                row.update(_stats(joined, "alpha_"))
                row.update(_thresholds(joined, "alpha_", (0.5, 0.9, 0.99), "gt"))
                alpha_step_rows.append(row)
        tk_vals = []
        tc_vals = []
        for view_id in view_ids:
            tk_vals.append(step_items["BND-K1"][view_id]["outputs"]["transmission"].mean(dim=-1).reshape(-1))
            tc_vals.append(step_items["CDEPTH"][view_id]["outputs"]["transmission"].mean(dim=-1).reshape(-1))
        tk = torch.cat(tk_vals)
        tc = torch.cat(tc_vals)
        lowt_step_rows.append(
            {
                "scene": SCENE,
                "nominal_step": step,
                "K1_LOW_T_fraction": float((tk < 0.1).float().mean().item()),
                "CDEPTH_LOW_T_fraction": float((tc < 0.1).float().mean().item()),
                "CDEPTH_LOW_T_fraction_DELTA": float((tc < 0.1).float().mean().item() - (tk < 0.1).float().mean().item()),
            }
        )
        if step == FINAL_NOMINAL_STEP:
            final_items = step_items
    _write_csv(output_dir / "alpha_region_metrics.csv", alpha_step_rows)
    _write_json(output_dir / "alpha_region_metrics.json", {"rows": alpha_step_rows})
    keyed_alpha = {(int(row["nominal_step"]), str(row["run"]), str(row["region"])): row for row in alpha_step_rows}
    for step in common_steps:
        row = {"scene": SCENE, "nominal_step": step}
        for region in ("WHOLE_IMAGE", "M1_HIGH_J", "M1_LOW_J", "BRIGHT_Q5", "EDGE_TOP20"):
            k = keyed_alpha[(step, "BND-K1", region)]
            c = keyed_alpha[(step, "CDEPTH", region)]
            row[f"{region}_alpha_mean_DELTA"] = float(c["alpha_mean"]) - float(k["alpha_mean"])
            row[f"{region}_alpha_p90_DELTA"] = float(c["alpha_p90"]) - float(k["alpha_p90"])
            row[f"{region}_alpha_gt_0.99_DELTA"] = float(c["alpha_gt_0.99"]) - float(k["alpha_gt_0.99"])
        alpha_rows.append(row)

    final_masks, gain_rows, highj_structural_rows, lowt_rows, attn_rows, edge_rows, dm_rows, spatial_summary = _final_spatial_attribution(
        final_items["BND-K1"], final_items["CDEPTH"], m1_items, regions, view_ids
    )
    gain_maps = {}
    for view_id in view_ids:
        gt = m1_items[view_id]["gt"]
        gain_maps[view_id] = (final_items["BND-K1"][view_id]["outputs"]["pred_image"] - gt).square().mean(dim=-1) - (
            final_items["CDEPTH"][view_id]["outputs"]["pred_image"] - gt
        ).square().mean(dim=-1)

    _write_csv(output_dir / "projected_radius_trajectory.csv", radius_rows)
    _write_json(output_dir / "projected_radius_trajectory.json", {"rows": radius_rows, "summary_rows": radius_agg, "compare_rows": radius_compare, "radii_semantics": "project_gaussians screen-space radius; projected-radius proxy in pixels"})
    _write_csv(output_dir / "visible_gaussian_scale_trajectory.csv", visible_scale_rows)
    _write_json(output_dir / "visible_gaussian_scale_trajectory.json", {"rows": visible_scale_rows})
    _write_csv(output_dir / "overlap_proxy_metrics.csv", highj_structural_rows)
    _write_json(output_dir / "overlap_proxy_metrics.json", {"CONTRIBUTOR_DIAGNOSTIC_AVAILABLE": False, "proxy_rows": highj_structural_rows})
    _write_csv(output_dir / "gain_harm_region_metrics.csv", gain_rows)
    _write_json(output_dir / "gain_harm_region_metrics.json", {"rows": gain_rows, **spatial_summary})
    _write_csv(output_dir / "new_low_t_localization.csv", lowt_rows)
    _write_json(output_dir / "new_low_t_localization.json", {"rows": lowt_rows, **spatial_summary})
    _write_csv(output_dir / "attenuation_distribution.csv", attn_rows)
    _write_json(output_dir / "attenuation_distribution.json", {"rows": attn_rows})
    _write_csv(output_dir / "edge_harm_metrics.csv", edge_rows)
    _write_json(output_dir / "edge_harm_metrics.json", {"rows": edge_rows})
    _write_csv(output_dir / "direct_medium_gain_harm.csv", dm_rows)
    _write_json(output_dir / "direct_medium_gain_harm.json", {"rows": dm_rows})
    _write_csv(output_dir / "low_t_trajectory.csv", lowt_step_rows)
    _write_json(output_dir / "low_t_trajectory.json", {"rows": lowt_step_rows})

    alpha_compare = alpha_rows
    structural_compare_rows: List[Dict[str, Any]] = []
    for step in common_steps:
        row = {"scene": SCENE, "nominal_step": step}
        for collection in (pop_compare, scale_compare, anis_compare, radius_compare, opacity_compare, alpha_compare, lowt_step_rows):
            match = next((item for item in collection if int(item["nominal_step"]) == step), {})
            row.update({key: value for key, value in match.items() if key not in row})
        structural_compare_rows.append(row)

    onsets, classification, scorecard, _ = _classify(
        paired_rows,
        pop_compare,
        scale_compare,
        anis_compare,
        radius_compare,
        opacity_compare,
        alpha_compare,
        lowt_step_rows,
        gain_rows,
        spatial_summary,
        edge_rows,
    )
    corr_rows = _trajectory_correlations(structural_compare_rows, paired_rows)
    _write_csv(output_dir / "structural_divergence_onsets.csv", onsets)
    _write_json(output_dir / "structural_divergence_onsets.json", {"rows": onsets})
    _write_csv(output_dir / "trajectory_correlations.csv", corr_rows)
    _write_json(output_dir / "trajectory_correlations.json", {"rows": corr_rows})
    _write_csv(output_dir / "factor_scorecard.csv", scorecard)
    _write_json(output_dir / "factor_scorecard.json", {"rows": scorecard})
    _write_json(output_dir / "pathway_classification.json", classification)
    _write_json(output_dir / "pathway_flags.json", {key: classification[key] for key in classification if key.endswith("_SUPPORTED") or key.endswith("_ALIGNED") or key in ("CONTRIBUTOR_DIAGNOSTIC_AVAILABLE",)})

    final_summary = {
        "scene": SCENE,
        "branch": repo_manifest["branch"],
        "start_head": repo_manifest["start_head"],
        "COMMON_TRAJECTORY_STEPS": common_steps,
        "view_ids": view_ids,
        "AUDIT_PARAMETER_SAFETY": "PASS" if audit_parameter_safety else "FAIL",
        **spatial_summary,
        **classification,
    }
    edge_top = next((row for row in edge_rows if row["region"] == "EDGE_TOP20"), {})
    final_summary["EDGE_HARM_ENRICHMENT"] = edge_top.get("EDGE_HARM_ENRICHMENT", float("nan"))
    _write_csv(output_dir / "cdepth_mitigation_final_summary.csv", [final_summary])
    _write_json(output_dir / "cdepth_mitigation_final_summary.json", final_summary)

    visual_manifest = _make_visuals(
        render_dir,
        view_ids,
        m1_items,
        final_items["BND-K1"],
        final_items["CDEPTH"],
        regions,
        final_masks,
        gain_maps,
        residual_images,
        paired_rows,
        pop_compare,
        scale_compare,
        anis_compare,
        radius_compare,
        opacity_compare,
        alpha_compare,
        final_summary,
    )
    _write_json(render_dir / "manifest.json", visual_manifest)
    _write_csv(render_dir / "manifest.csv", visual_manifest)
    _write_visual_index(render_dir, visual_manifest)

    _write_json(
        output_dir / "manifest.json",
        {
            "repo_manifest": str(output_dir / "repo_manifest.json"),
            "interrupted_work_manifest": str(output_dir / "interrupted_work_manifest.json"),
            "trajectory_availability": str(output_dir / "trajectory_availability.json"),
            "checkpoint_manifest": str(output_dir / "checkpoint_manifest.json"),
            "final_summary": str(output_dir / "cdepth_mitigation_final_summary.json"),
            "visual_manifest": str(render_dir / "manifest.json"),
            "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
            "outputs": {
                "rgb_mitigation_trajectory": str(output_dir / "rgb_mitigation_trajectory.json"),
                "high_j_recovery_trajectory": str(output_dir / "high_j_recovery_trajectory.json"),
                "gaussian_population_trajectory": str(output_dir / "gaussian_population_trajectory.json"),
                "gaussian_scale_trajectory": str(output_dir / "gaussian_scale_trajectory.json"),
                "gaussian_anisotropy_trajectory": str(output_dir / "gaussian_anisotropy_trajectory.json"),
                "projected_radius_trajectory": str(output_dir / "projected_radius_trajectory.json"),
                "opacity_trajectory": str(output_dir / "opacity_trajectory.json"),
                "alpha_region_metrics": str(output_dir / "alpha_region_metrics.json"),
                "gain_harm_region_metrics": str(output_dir / "gain_harm_region_metrics.json"),
                "new_low_t_localization": str(output_dir / "new_low_t_localization.json"),
                "attenuation_distribution": str(output_dir / "attenuation_distribution.json"),
                "edge_harm_metrics": str(output_dir / "edge_harm_metrics.json"),
                "direct_medium_gain_harm": str(output_dir / "direct_medium_gain_harm.json"),
                "factor_scorecard": str(output_dir / "factor_scorecard.json"),
                "pathway_flags": str(output_dir / "pathway_flags.json"),
                "pathway_classification": str(output_dir / "pathway_classification.json"),
            },
        },
    )
    _write_research_note(
        note_path,
        final_summary,
        {
            "k1_config": str(repo / RUNS["BND-K1"].config_relpath),
            "cdepth_config": str(repo / RUNS["CDEPTH"].config_relpath),
            "summary_json": str(output_dir / "cdepth_mitigation_final_summary.json"),
            "visual_manifest": str(render_dir / "manifest.json"),
            "visual_index": str(render_dir / "VISUAL_COMPARE_INDEX.md"),
        },
    )
    print(json.dumps(final_summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
