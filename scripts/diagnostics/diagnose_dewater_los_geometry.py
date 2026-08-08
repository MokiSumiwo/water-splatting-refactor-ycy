#!/usr/bin/env python
"""Audit LOS/depth geometry used by the dewatering direct-attenuation path."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from nerfstudio.utils.eval_utils import eval_setup
from torch import Tensor


STAT_Q = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


DEFAULT_RUNS = {
    "D100-SCRATCH": {
        "nominal_step": 15000,
        "load_step": 14999,
        "config": (
            "outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/"
            "cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml"
        ),
    },
    "D010-SWITCH": {
        "nominal_step": 15000,
        "load_step": 15000,
        "config": (
            "outputs/dewater_d010_persistence_20260807/"
            "dewater_d010_persist_curasao_seed42_step13000_to_15000/water-splatting/"
            "dewater_d010_persist_curasao_seed42_step13000_to_15000_"
            "20260807_d010_persistence_d010_persist_g0p10/config.yml"
        ),
    },
    "D010-SCRATCH": {
        "nominal_step": 15000,
        "load_step": 14999,
        "config": (
            "outputs/dewater_d010_scratch_20260807/"
            "dewater_d010_scratch_curasao_seed42_step0_to_15000/water-splatting/"
            "dewater_d010_scratch_curasao_seed42_step0_to_15000_20260807_d010_scratch_g0p10/config.yml"
        ),
    },
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

    return eval_setup(config_path, eval_num_rays_per_chunk=None, test_mode=test_mode, update_config_callback=_update_config)


def _to_hwc(value: Tensor) -> Tensor:
    out = value.detach().float()
    if out.ndim == 2:
        out = out[..., None]
    if out.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC tensor, got {tuple(out.shape)}")
    return out


def _finite(value: Tensor) -> Tensor:
    flat = value.detach().float().reshape(-1).cpu()
    return flat[torch.isfinite(flat)]


def _stats(values: Tensor) -> Dict[str, float | int]:
    flat = _finite(values)
    out: Dict[str, float | int] = {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
    }
    for q in STAT_Q:
        key = f"p{int(round(q * 100)):02d}"
        if flat.numel():
            rank = max(1, min(int(flat.numel()), int(math.ceil(float(q) * float(flat.numel())))))
            out[key] = float(flat.kthvalue(rank).values.item())
        else:
            out[key] = 0.0
    return out


def _camera_items(pipeline: Any, max_images: int, device: torch.device) -> Iterable[Tuple[int, Any, Mapping[str, Any]]]:
    max_count = max_images if max_images > 0 else 10**9
    for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
        if image_idx >= max_count:
            break
        yield image_idx, camera.to(device) if hasattr(camera, "to") else camera, batch


def _save_depth_png(path: Path, value: Tensor, lo: float, hi: float) -> Tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mapped = ((_to_hwc(value)[..., :1] - float(lo)) / max(float(hi) - float(lo), 1e-8)).clamp(0.0, 1.0)
    rgb = mapped.expand(-1, -1, 3)
    arr = (rgb.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return int(arr.shape[1]), int(arr.shape[0])


def _tile(path: Path, label: str, width: int) -> Image.Image:
    with Image.open(path) as src:
        image = src.convert("RGB")
    if image.width > width:
        height = max(1, int(round(image.height * width / image.width)))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    pad = 24
    canvas = Image.new("RGB", (image.width, image.height + pad), "white")
    canvas.paste(image, (0, pad))
    ImageDraw.Draw(canvas).text((4, 5), label, fill="black")
    return canvas


def _write_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Path]]], width: int) -> None:
    rendered = []
    for row in rows:
        tiles = [_tile(tile_path, label, width) for label, tile_path in row]
        row_img = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
        x = 0
        for tile in tiles:
            row_img.paste(tile, (x, 0))
            x += tile.width
        rendered.append(row_img)
    sheet = Image.new("RGB", (max(row.width for row in rendered), sum(row.height for row in rendered)), "white")
    y = 0
    for row in rendered:
        sheet.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _run_one(
    *,
    repo: Path,
    run: str,
    config_path: Path,
    load_step: int,
    nominal_step: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    config, pipeline, checkpoint_path, loaded_step = _setup_pipeline(config_path, load_step, args.test_mode)
    model = pipeline.model
    model.eval()
    gamma = float(getattr(model.config, "direct_optical_depth_scale", 1.0))

    gaussian_depths: List[Tensor] = []
    image_depths: List[Tensor] = []
    beta_values: List[Tensor] = []
    tau_values: List[Tensor] = []
    tau_over_beta_values: List[Tensor] = []
    per_view: List[Dict[str, Any]] = []
    depth_images: Dict[int, Path] = {}
    last_depth_images: Dict[int, Path] = {}

    with torch.no_grad():
        for view_id, (image_idx, camera, _batch) in enumerate(_camera_items(pipeline, args.max_images, model.device)):
            outputs = model.get_outputs_for_camera(camera=camera)
            depth = _to_hwc(outputs["depth"]).cpu()
            last_depth = _to_hwc(outputs.get("last_depth", outputs["depth"])).cpu()
            accumulation = _to_hwc(outputs["accumulation"]).cpu()
            support = (accumulation[..., :1] > float(args.object_support_accumulation_threshold)).expand_as(
                _to_hwc(outputs["medium_attn"]).cpu()
            )
            beta_eff = _to_hwc(outputs["medium_attn"]).cpu()
            tau = beta_eff * depth
            tau_over_beta = tau / beta_eff.clamp_min(1e-8)
            visible = outputs.get("gaussian_visible_mask")
            projected_depths = outputs.get("projected_gaussian_depths")
            if visible is None or projected_depths is None:
                raise RuntimeError("Model outputs must include gaussian_visible_mask and projected_gaussian_depths")
            visible = visible.detach().bool().reshape(-1).cpu()
            projected_depths = projected_depths.detach().float().reshape(-1).cpu()
            gaussian_view_depths = projected_depths[visible & torch.isfinite(projected_depths) & (projected_depths > 0)]

            gaussian_depths.append(gaussian_view_depths)
            image_depths.append(depth[..., :1][support[..., :1]].reshape(-1))
            beta_values.append(beta_eff[support].reshape(-1))
            tau_values.append(tau[support].reshape(-1))
            tau_over_beta_values.append(tau_over_beta[support].reshape(-1))
            per_view.append(
                {
                    "view_id": int(view_id),
                    "image_idx": int(image_idx),
                    "support_coverage": float(accumulation[..., :1].gt(args.object_support_accumulation_threshold).float().mean().item()),
                    "visible_gaussian_count": int(gaussian_view_depths.numel()),
                    "gaussian_los_distance": _stats(gaussian_view_depths),
                    "image_expected_depth": _stats(depth[..., :1][support[..., :1]].reshape(-1)),
                    "beta_eff": _stats(beta_eff[support].reshape(-1)),
                    "tau": _stats(tau[support].reshape(-1)),
                    "tau_over_beta_eff": _stats(tau_over_beta[support].reshape(-1)),
                }
            )

            if view_id in {0, 1, 2}:
                depth_path = args.render_dir / "los_geometry_audit" / run / f"view_{view_id:04d}_expected_depth.png"
                last_depth_path = args.render_dir / "los_geometry_audit" / run / f"view_{view_id:04d}_last_depth.png"
                depth_images[view_id] = depth_path
                last_depth_images[view_id] = last_depth_path
                _save_depth_png(depth_path, depth, args.depth_display_min, args.depth_display_max)
                _save_depth_png(last_depth_path, last_depth, args.depth_display_min, args.depth_display_max)

    aggregate = {
        "gaussian_los_distance": _stats(torch.cat(gaussian_depths) if gaussian_depths else torch.empty(0)),
        "image_expected_depth": _stats(torch.cat(image_depths) if image_depths else torch.empty(0)),
        "beta_eff": _stats(torch.cat(beta_values) if beta_values else torch.empty(0)),
        "tau": _stats(torch.cat(tau_values) if tau_values else torch.empty(0)),
        "tau_over_beta_eff": _stats(torch.cat(tau_over_beta_values) if tau_over_beta_values else torch.empty(0)),
    }
    result = {
        "run": run,
        "nominal_step": int(nominal_step),
        "requested_load_step": int(load_step),
        "loaded_step": int(loaded_step),
        "load_config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "direct_optical_depth_scale": gamma,
        "gaussian_count": int(getattr(model, "num_points", 0)),
        "definitions": {
            "gaussian_los_distance": (
                "projected per-Gaussian view-space z depth from UnderwaterRasterizer.project; "
                "this is the depth value passed to the CUDA direct attenuation exp(-medium_attn_pix * depth)."
            ),
            "image_expected_depth": "outputs['depth']; alpha-normalized expected depth image used for image-space diagnostics.",
            "tau_over_beta_eff": "outputs['medium_attn'] * outputs['depth'] / outputs['medium_attn']; consistency check for image-space depth.",
        },
        "aggregate": aggregate,
        "per_view": per_view,
        "depth_images": {str(key): str(value) for key, value in depth_images.items()},
        "last_depth_images": {str(key): str(value) for key, value in last_depth_images.items()},
    }
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo = _repo_root()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.render_dir.mkdir(parents=True, exist_ok=True)

    run_specs: Dict[str, Dict[str, Any]] = dict(DEFAULT_RUNS)
    if args.run_json is not None:
        run_specs = json.loads(args.run_json.read_text(encoding="utf8"))

    results = []
    for run_name, spec in run_specs.items():
        config_path = Path(spec["config"])
        if not config_path.is_absolute():
            config_path = repo / config_path
        results.append(
            _run_one(
                repo=repo,
                run=run_name,
                config_path=config_path,
                load_step=int(spec["load_step"]),
                nominal_step=int(spec.get("nominal_step", spec["load_step"])),
                args=args,
            )
        )

    by_run = {item["run"]: item for item in results}
    d100 = by_run.get("D100-SCRATCH")
    scratch = by_run.get("D010-SCRATCH")
    switch = by_run.get("D010-SWITCH")
    d_p90_change_scratch = 0.0
    trigger = "UNAVAILABLE"
    if d100 is not None and scratch is not None:
        base = float(d100["aggregate"]["gaussian_los_distance"]["p90"])
        comp = float(scratch["aggregate"]["gaussian_los_distance"]["p90"])
        d_p90_change_scratch = comp / max(base, 1e-8) - 1.0
        if d_p90_change_scratch >= 0.20:
            trigger = "TRUE"
        elif d_p90_change_scratch < 0.10:
            trigger = "FALSE"
        else:
            trigger = "AMBIGUOUS"

    if results:
        rows = []
        view_ids = sorted({int(k) for item in results for k in item["depth_images"].keys()})
        for view_id in view_ids:
            row = []
            for item in results:
                key = str(view_id)
                if key in item["depth_images"]:
                    row.append((f"{item['run']} expected depth", Path(item["depth_images"][key])))
            if row:
                rows.append(row)
        if rows:
            _write_sheet(args.render_dir / "los_geometry_audit" / "contact_sheet_expected_depth.png", rows, args.contact_tile_width)

    summary = {
        "diagnostic": "dewater_los_geometry_audit",
        "git_commit": _git_commit(repo),
        "scene": args.scene,
        "test_mode": args.test_mode,
        "object_support_accumulation_threshold": float(args.object_support_accumulation_threshold),
        "depth_display_range": [float(args.depth_display_min), float(args.depth_display_max)],
        "runs": results,
        "comparison": {
            "distance_p90_change_scratch": d_p90_change_scratch,
            "DEPTH_COMPENSATION_TRIGGER": trigger,
            "d010_switch_distance_p90_change_vs_d100": (
                float(switch["aggregate"]["gaussian_los_distance"]["p90"])
                / max(float(d100["aggregate"]["gaussian_los_distance"]["p90"]), 1e-8)
                - 1.0
                if d100 is not None and switch is not None
                else None
            ),
        },
        "contact_sheets": {
            "expected_depth": str(args.render_dir / "los_geometry_audit" / "contact_sheet_expected_depth.png"),
        },
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "nominal_step",
        "loaded_step",
        "gamma_D",
        "gaussian_count",
        "distance_mean",
        "distance_p50",
        "distance_p90",
        "distance_p95",
        "distance_p99",
        "distance_max",
        "image_depth_p90",
        "beta_eff_mean",
        "tau_p90",
        "tau_over_beta_p90",
    ]
    with args.output_csv.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            agg = item["aggregate"]
            writer.writerow(
                {
                    "run": item["run"],
                    "nominal_step": item["nominal_step"],
                    "loaded_step": item["loaded_step"],
                    "gamma_D": item["direct_optical_depth_scale"],
                    "gaussian_count": item["gaussian_count"],
                    "distance_mean": agg["gaussian_los_distance"]["mean"],
                    "distance_p50": agg["gaussian_los_distance"]["p50"],
                    "distance_p90": agg["gaussian_los_distance"]["p90"],
                    "distance_p95": agg["gaussian_los_distance"]["p95"],
                    "distance_p99": agg["gaussian_los_distance"]["p99"],
                    "distance_max": agg["gaussian_los_distance"]["max"],
                    "image_depth_p90": agg["image_expected_depth"]["p90"],
                    "beta_eff_mean": agg["beta_eff"]["mean"],
                    "tau_p90": agg["tau"]["p90"],
                    "tau_over_beta_p90": agg["tau_over_beta_eff"]["p90"],
                }
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="Curasao")
    parser.add_argument("--test-mode", default="test")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--run-json", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/dewater_seafree_factor_20260808/los_geometry_audit.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/dewater_seafree_factor_20260808/los_geometry_audit.csv"),
    )
    parser.add_argument("--render-dir", type=Path, default=Path("renders/dewater_seafree_factor_20260808"))
    parser.add_argument("--object-support-accumulation-threshold", type=float, default=0.01)
    parser.add_argument("--depth-display-min", type=float, default=0.0)
    parser.add_argument("--depth-display-max", type=float, default=10.0)
    parser.add_argument("--contact-tile-width", type=int, default=320)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
