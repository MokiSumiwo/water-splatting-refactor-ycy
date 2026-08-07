#!/usr/bin/env python
"""Build detached renderer-derived background-water masks for dewatering B runs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from nerfstudio.utils.eval_utils import eval_setup

from water_splatting.losses.background_attribution import effective_background_mask


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _setup_pipeline(config_path: Path, step: int | None, test_mode: str) -> Tuple[Any, Any, Path, int]:
    def _update_config(config: Any) -> Any:
        if step is not None:
            config.load_step = int(step)
        return config

    return eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode=test_mode,
        update_config_callback=_update_config,
    )


def _to_hwc(value: torch.Tensor, key: str) -> torch.Tensor:
    tensor = value.detach().float()
    if tensor.ndim == 2:
        tensor = tensor[..., None]
    if tensor.ndim != 3:
        raise ValueError(f"{key} must be HxW or HxWxC, got {tuple(tensor.shape)}")
    return tensor


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    image = _to_hwc(rgb, "rgb")
    if image.shape[-1] == 1:
        return image
    weights = image.new_tensor([0.2126, 0.7152, 0.0722])
    return (image[..., :3] * weights).sum(dim=-1, keepdim=True)


def _to_nchw(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[..., None]
    return mask.float().permute(2, 0, 1)[None]


def _from_nchw(mask: torch.Tensor) -> torch.Tensor:
    return mask[0].permute(1, 2, 0)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    kernel = 2 * radius + 1
    pooled = F.max_pool2d(_to_nchw(mask), kernel_size=kernel, stride=1, padding=radius)
    return (_from_nchw(pooled) > 0.5).bool()


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    kernel = 2 * radius + 1
    eroded = -F.max_pool2d(-_to_nchw(mask), kernel_size=kernel, stride=1, padding=radius)
    return (_from_nchw(eroded) > 0.5).bool()


def _camera_id(camera: Any, outputs: Dict[str, torch.Tensor], fallback: int) -> int:
    if "camera_index" in outputs:
        return int(outputs["camera_index"].detach().cpu().reshape(-1)[0].item())
    if getattr(camera, "metadata", None) is not None and "cam_idx" in camera.metadata:
        value = camera.metadata["cam_idx"]
        if torch.is_tensor(value):
            return int(value.detach().cpu().reshape(-1)[0].item())
        return int(value)
    return int(fallback)


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


def _image_name(pipeline: Any, split: str, image_idx: int) -> str:
    dataset = pipeline.datamanager.train_dataset if split == "train" else pipeline.datamanager.eval_dataset
    try:
        return Path(dataset.image_filenames[int(image_idx)]).name
    except Exception:
        return f"{split}_{int(image_idx):04d}"


def _output_clear_luma(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    clear = outputs.get("J_gaussian_raw", outputs.get("J_object_raw", outputs.get("J_raw", outputs["J"])))
    return _luma(clear.detach().float().clamp(0.0, 1.0))


def _save_png(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _to_hwc(value, "png").detach().float().cpu()
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    arr = (image[..., :3].clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_overlay(rgb: torch.Tensor, water: torch.Tensor, boundary: torch.Tensor, path: Path) -> None:
    image = _to_hwc(rgb, "rgb").detach().float().cpu().clamp(0.0, 1.0)
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    if water.shape[:2] != image.shape[:2]:
        water = F.interpolate(_to_nchw(water), size=image.shape[:2], mode="nearest")[0].permute(1, 2, 0)
    if boundary.shape[:2] != image.shape[:2]:
        boundary = F.interpolate(_to_nchw(boundary), size=image.shape[:2], mode="nearest")[0].permute(1, 2, 0)
    water2 = water[..., 0].float()
    boundary2 = boundary[..., 0].float()
    overlay = image.clone()
    overlay[..., 1] = torch.maximum(overlay[..., 1], 0.90 * water2)
    overlay[..., 0] = torch.maximum(overlay[..., 0], 0.80 * boundary2)
    overlay[..., 2] = overlay[..., 2] * (1.0 - 0.25 * water2)
    _save_png(path, overlay)


def _coverage_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _save_contact_sheet(output_dir: Path, rows: List[Dict[str, Any]], max_views: int) -> str | None:
    selected = rows[: max(0, max_views)]
    tiles = []
    for row in selected:
        path = output_dir / f"view_{int(row['camera_id']):04d}_water_overlay.png"
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((320, 180))
        tile = Image.new("RGB", (320, 210), "white")
        tile.paste(img, (0, 0))
        draw = ImageDraw.Draw(tile)
        label = f"local {row['local_index']} cam {row['camera_id']} cov {row['water_coverage']:.4f}"
        draw.text((6, 186), label, fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return None
    cols = min(5, len(tiles))
    sheet_rows = int(math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 320, sheet_rows * 210), "white")
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 320, (idx // cols) * 210))
    path = output_dir / "background_water_mask_contact_sheet.jpg"
    sheet.save(path, quality=92)
    return str(path)


def build_masks(args: argparse.Namespace) -> Dict[str, Any]:
    _config, pipeline, checkpoint_path, loaded_step = _setup_pipeline(args.load_config, args.load_step, args.test_mode)
    pipeline.eval()
    model = pipeline.model
    device = model.device
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    camera_ids: set[int] = set()
    with torch.no_grad():
        for local_index, camera, batch in _camera_items(pipeline, args.split, args.max_images, device):
            outputs = model.get_outputs(camera)
            camera_id = _camera_id(camera, outputs, local_index)
            if camera_id in camera_ids:
                raise RuntimeError(f"Duplicate camera_id {camera_id} in split {args.split}")
            camera_ids.add(camera_id)

            depth = _to_hwc(outputs["depth"], "depth").detach().float()
            accumulation = _to_hwc(outputs["accumulation"], "accumulation").detach().float().clamp(0.0, 1.0)
            clear_luma = _output_clear_luma(outputs)
            valid_depth = torch.isfinite(depth) & (depth > 0)
            if valid_depth.any():
                far_cutoff = torch.quantile(depth[valid_depth], float(args.water_depth_quantile))
                far_mask = valid_depth & (depth >= far_cutoff)
            else:
                far_cutoff = depth.new_tensor(0.0)
                far_mask = torch.zeros_like(depth, dtype=torch.bool)

            water_seed = far_mask & (accumulation <= float(args.water_accum_max)) & (clear_luma <= float(args.water_j_luma_max))
            object_seed = (accumulation >= float(args.object_accum_min)) & (clear_luma >= float(args.object_j_luma_min))
            water_raw = _erode(water_seed.detach().cpu(), int(args.core_erode_radius))
            object_mask = _erode(object_seed.detach().cpu(), int(args.core_erode_radius))
            object_mask = object_mask & (~_dilate(water_raw, int(args.boundary_radius)))
            boundary_mask = _dilate(object_mask, int(args.boundary_radius)) & (~_erode(object_mask, int(args.boundary_radius)))
            boundary_mask = boundary_mask & (~water_raw)

            hit_confidence = outputs.get("hit_confidence")
            hit_cpu = hit_confidence.detach().float().cpu() if hit_confidence is not None else None
            water_final = effective_background_mask(
                water_mask=water_raw.float(),
                boundary_mask=boundary_mask.float() if args.exclude_boundary else None,
                hit_confidence=hit_cpu,
                hit_threshold=float(args.hit_exclusion_threshold),
            ).detach().cpu() > 0.5

            if args.save_png:
                _save_png(args.output_dir / f"view_{camera_id:04d}_water.png", water_final.float())
                _save_png(args.output_dir / f"view_{camera_id:04d}_water_raw.png", water_raw.float())
                _save_png(args.output_dir / f"view_{camera_id:04d}_object.png", object_mask.float())
                _save_png(args.output_dir / f"view_{camera_id:04d}_boundary.png", boundary_mask.float())
                gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"]).detach()
                _save_png(args.output_dir / f"view_{camera_id:04d}_rgb.png", gt)
                _save_overlay(gt, water_final.float(), boundary_mask.float(), args.output_dir / f"view_{camera_id:04d}_water_overlay.png")

            total_pixels = int(water_final.numel())
            row = {
                "local_index": int(local_index),
                "camera_id": int(camera_id),
                "image_name": _image_name(pipeline, args.split, local_index),
                "path": str(args.output_dir / f"view_{camera_id:04d}_regions.pt"),
                "height": int(water_final.shape[0]),
                "width": int(water_final.shape[1]),
                "water_pixels": int(water_final.sum().item()),
                "water_coverage": float(water_final.float().mean().item()),
                "water_raw_coverage": float(water_raw.float().mean().item()),
                "object_coverage": float(object_mask.float().mean().item()),
                "boundary_coverage": float(boundary_mask.float().mean().item()),
                "water_depth_cutoff": float(far_cutoff.detach().cpu().item()),
                "total_pixels": total_pixels,
            }
            payload = {
                "water": water_final.bool(),
                "effective_background": water_final.bool(),
                "water_raw": water_raw.bool(),
                "object": object_mask.bool(),
                "boundary": boundary_mask.bool(),
                "local_index": int(local_index),
                "camera_id": int(camera_id),
                "image_name": row["image_name"],
                "height": int(water_final.shape[0]),
                "width": int(water_final.shape[1]),
                "source_load_config": str(args.load_config),
                "source_checkpoint": str(checkpoint_path),
                "source_step": int(loaded_step),
                "split": args.split,
                "mask_definition": "far renderer depth + low accumulation + low full-SH clear luma, eroded, boundary/hit excluded",
                "water_depth_quantile": float(args.water_depth_quantile),
                "water_depth_cutoff": row["water_depth_cutoff"],
                "water_accum_max": float(args.water_accum_max),
                "water_j_luma_max": float(args.water_j_luma_max),
                "object_accum_min": float(args.object_accum_min),
                "object_j_luma_min": float(args.object_j_luma_min),
                "core_erode_radius": int(args.core_erode_radius),
                "boundary_radius": int(args.boundary_radius),
                "exclude_boundary": bool(args.exclude_boundary),
                "hit_exclusion_threshold": float(args.hit_exclusion_threshold),
            }
            torch.save(payload, args.output_dir / f"view_{camera_id:04d}_regions.pt")
            rows.append(row)

    coverage = _coverage_stats([row["water_coverage"] for row in rows])
    contact_sheet = _save_contact_sheet(args.output_dir, rows, int(args.contact_sheet_views)) if args.save_png else None
    gate_pass = bool(
        coverage["mean"] >= float(args.min_mean_coverage)
        and coverage["mean"] <= float(args.max_mean_coverage)
        and len(rows) > 0
    )
    repo = Path(__file__).resolve().parents[2]
    metadata: Dict[str, Any] = {
        "mask_type": "dewater_renderer_background_water",
        "usage": "training_only_detached_background_medium_supervision_candidate",
        "split": args.split,
        "load_config": str(args.load_config),
        "reference_checkpoint": str(checkpoint_path),
        "reference_step": int(loaded_step),
        "test_mode": args.test_mode,
        "output_dir": str(args.output_dir),
        "count": len(rows),
        "definition": {
            "water": "fixed effective background-water mask used by B training",
            "water_raw": "far renderer-depth seed with low accumulation and low full-SH clear-object luma before boundary/hit exclusion",
            "boundary": "renderer-derived object boundary exclusion from high accumulation/full-SH clear-luma object seed",
            "effective_background": "same bool tensor as water; produced by effective_background_mask",
        },
        "parameters": {
            "water_depth_quantile": float(args.water_depth_quantile),
            "water_accum_max": float(args.water_accum_max),
            "water_j_luma_max": float(args.water_j_luma_max),
            "object_accum_min": float(args.object_accum_min),
            "object_j_luma_min": float(args.object_j_luma_min),
            "core_erode_radius": int(args.core_erode_radius),
            "boundary_radius": int(args.boundary_radius),
            "exclude_boundary": bool(args.exclude_boundary),
            "hit_exclusion_threshold": float(args.hit_exclusion_threshold),
        },
        "coverage": coverage,
        "coverage_gate": {
            "min_mean_coverage": float(args.min_mean_coverage),
            "max_mean_coverage": float(args.max_mean_coverage),
            "pass": gate_pass,
            "criterion": "mean water coverage must be within [min_mean_coverage, max_mean_coverage]",
        },
        "contact_sheet": contact_sheet,
        "git_commit": _git_commit(repo),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "masks": rows,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    if args.fail_on_coverage_gate and not gate_pass:
        raise RuntimeError(f"Coverage gate failed: {coverage}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--test-mode", choices=("test", "val", "inference"), default="inference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--water-depth-quantile", type=float, default=0.90)
    parser.add_argument("--water-accum-max", type=float, default=0.25)
    parser.add_argument("--water-j-luma-max", type=float, default=0.12)
    parser.add_argument("--object-accum-min", type=float, default=0.55)
    parser.add_argument("--object-j-luma-min", type=float, default=0.03)
    parser.add_argument("--core-erode-radius", type=int, default=2)
    parser.add_argument("--boundary-radius", type=int, default=5)
    parser.set_defaults(exclude_boundary=True)
    parser.add_argument("--exclude-boundary", dest="exclude_boundary", action="store_true")
    parser.add_argument("--include-boundary", dest="exclude_boundary", action="store_false")
    parser.add_argument("--hit-exclusion-threshold", type=float, default=0.60)
    parser.add_argument("--min-mean-coverage", type=float, default=0.02)
    parser.add_argument("--max-mean-coverage", type=float, default=0.80)
    parser.add_argument("--fail-on-coverage-gate", action="store_true")
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--contact-sheet-views", type=int, default=25)
    args = parser.parse_args()
    metadata = build_masks(args)
    print(
        json.dumps(
            {
                "metadata": str(args.output_dir / "metadata.json"),
                "mask_type": metadata["mask_type"],
                "split": metadata["split"],
                "count": metadata["count"],
                "coverage": metadata["coverage"],
                "coverage_gate": metadata["coverage_gate"],
                "contact_sheet": metadata["contact_sheet"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
