#!/usr/bin/env python
"""Build high-precision open-water masks from pseudo-depth and image edges."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.utils import save_image


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _list_image_paths(data_dir: Path, images_path: str, split: str) -> List[Path]:
    list_name = {
        "train": "train_list.txt",
        "eval": "val_list.txt",
        "val": "val_list.txt",
        "test": "test_list.txt",
    }[split]
    list_path = data_dir / list_name
    if list_path.exists():
        names = [line.strip() for line in list_path.read_text(encoding="utf8").splitlines() if line.strip()]
        return [data_dir / images_path / name for name in names]
    return sorted((data_dir / images_path).glob("*.png"))


def _list_image_paths_from_config(load_config: Path, split: str) -> List[Path]:
    from nerfstudio.utils.eval_utils import eval_setup

    _config, pipeline, _checkpoint_path, _step = eval_setup(load_config)
    if split == "train":
        dataset = pipeline.datamanager.train_dataset
    else:
        dataset = pipeline.datamanager.eval_dataset
    return [Path(path) for path in dataset.image_filenames]


def _read_depth(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    finite = np.isfinite(arr)
    if not finite.any():
        return torch.zeros((*arr.shape, 1), dtype=torch.float32)
    valid = arr[finite]
    lo, hi = float(valid.min()), float(valid.max())
    depth = (arr - lo) / max(hi - lo, 1e-6)
    return torch.from_numpy(depth).float()[..., None].clamp(0.0, 1.0)


def _read_rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).float()


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    return 0.299 * rgb[..., :1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]


def _gradient_luma(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 3:
        x = _gray(x)
    dx = torch.zeros_like(x)
    dy = torch.zeros_like(x)
    dx[:, 1:] = torch.abs(x[:, 1:] - x[:, :-1])
    dy[1:, :] = torch.abs(x[1:, :] - x[:-1, :])
    return torch.maximum(dx, dy)


def _local_variance(gray: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return torch.zeros_like(gray)
    x = gray.permute(2, 0, 1)[None]
    kernel = 2 * radius + 1
    mean = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
    mean_sq = F.avg_pool2d(x.square(), kernel_size=kernel, stride=1, padding=radius)
    return (mean_sq - mean.square()).clamp_min(0.0)[0].permute(1, 2, 0)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.float().permute(2, 0, 1)[None]
    bad = F.max_pool2d(1.0 - x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (bad < 0.5)[0].permute(1, 2, 0)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.float().permute(2, 0, 1)[None]
    return (F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius) > 0.5)[0].permute(1, 2, 0)


def _remove_border(mask: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return mask
    out = mask.clone()
    out[:border] = False
    out[-border:] = False
    out[:, :border] = False
    out[:, -border:] = False
    return out


def _apply_water_y_limit(mask: torch.Tensor, max_y_fraction: float) -> torch.Tensor:
    if max_y_fraction <= 0.0 or max_y_fraction >= 1.0:
        return mask
    height = mask.shape[0]
    cutoff = max(1, min(height, int(round(height * max_y_fraction))))
    out = mask.clone()
    out[cutoff:] = False
    return out


def _largest_component(mask: torch.Tensor) -> torch.Tensor:
    try:
        from scipy import ndimage
    except Exception:
        return mask
    labels, count = ndimage.label(mask.squeeze(-1).cpu().numpy().astype(np.uint8))
    if count == 0:
        return mask
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    keep = int(sizes.argmax())
    return torch.from_numpy(labels == keep)[..., None]


def _connected_water_filter(mask: torch.Tensor, top_only: bool, side_min_area: float) -> torch.Tensor:
    if not top_only and side_min_area <= 0:
        return mask.bool()
    try:
        from scipy import ndimage
    except Exception:
        return mask.bool()

    arr = mask.squeeze(-1).cpu().numpy().astype(np.uint8)
    labels, count = ndimage.label(arr)
    if count == 0:
        return torch.zeros_like(mask, dtype=torch.bool)

    image_area = arr.shape[0] * arr.shape[1]
    min_side_pixels = int(side_min_area * image_area) if 0.0 < side_min_area < 1.0 else int(side_min_area)
    keep = np.zeros(count + 1, dtype=bool)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    for label in range(1, count + 1):
        touches_top = bool((labels[0, :] == label).any())
        touches_side = bool((labels[:, 0] == label).any() or (labels[:, -1] == label).any())
        if top_only and touches_top:
            keep[label] = True
        if min_side_pixels > 0 and touches_side and sizes[label] >= min_side_pixels:
            keep[label] = True
    return torch.from_numpy(keep[labels])[..., None]


def _load_optional_q_hit(q_hit_dir: Optional[Path], image_idx: int, image_name: str) -> Optional[torch.Tensor]:
    if q_hit_dir is None:
        return None
    candidates = [
        q_hit_dir / f"view_{image_idx:04d}_q_hit.pt",
        q_hit_dir / f"view_{image_idx:04d}_regions.pt",
        q_hit_dir / image_name,
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict):
                value = payload.get("q_hit")
                if value is None:
                    value = payload.get("hit_confidence")
            else:
                value = payload
            if value is not None:
                if value.ndim == 2:
                    value = value[..., None]
                return value.float().clamp(0.0, 1.0)
        else:
            arr = np.asarray(Image.open(path)).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            arr = arr / max(float(arr.max()), 1.0)
            return torch.from_numpy(arr).float()[..., None].clamp(0.0, 1.0)
    return None


def _save_mask(mask: torch.Tensor, path: Path) -> None:
    save_image(mask.float().permute(2, 0, 1), path)


def _save_overlay(rgb: torch.Tensor, water: torch.Tensor, boundary: torch.Tensor, uncertain: torch.Tensor, path: Path) -> None:
    overlay = rgb.clone()
    water2 = water[..., 0].float()
    boundary2 = boundary[..., 0].float()
    uncertain2 = uncertain[..., 0].float()
    overlay[..., 0] = torch.maximum(overlay[..., 0], 0.90 * water2 + 0.55 * boundary2)
    overlay[..., 1] = torch.maximum(overlay[..., 1], 0.75 * uncertain2)
    overlay[..., 2] = overlay[..., 2] * (1.0 - 0.35 * water2)
    save_image(overlay.permute(2, 0, 1).clamp(0.0, 1.0), path)


def _save_contact_sheet(output_dir: Path, count: int, max_views: int) -> Optional[str]:
    overlays = []
    for idx in range(min(count, max_views)):
        path = output_dir / f"view_{idx:04d}_water_overlay.png"
        if path.exists():
            overlays.append(path)
    if not overlays:
        return None
    thumbs = []
    for path in overlays:
        img = Image.open(path).convert("RGB")
        img.thumbnail((320, 180))
        tile = Image.new("RGB", (320, 205), "white")
        tile.paste(img, (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((6, 184), path.stem.replace("_water_overlay", ""), fill=(0, 0, 0))
        thumbs.append(tile)
    cols = min(5, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 320, rows * 205), "white")
    for idx, tile in enumerate(thumbs):
        sheet.paste(tile, ((idx % cols) * 320, (idx // cols) * 205))
    out = output_dir / "water_mask_contact_sheet.jpg"
    sheet.save(out, quality=92)
    return str(out)


def build_masks(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.load_config is not None:
        image_paths = _list_image_paths_from_config(args.load_config, args.split)
    else:
        image_paths = _list_image_paths(args.data, args.images_path, args.split)

    summaries: List[Dict[str, Any]] = []
    for image_idx, image_path in enumerate(image_paths):
        if args.max_images is not None and image_idx >= args.max_images:
            break
        depth_path = args.depth_dir / image_path.name
        if not depth_path.exists():
            continue

        rgb = _read_rgb(image_path)
        depth = _read_depth(depth_path)
        fg = depth > float(args.foreground_depth_threshold)
        if args.keep_largest_foreground:
            fg = _largest_component(fg)
        bg_candidate = ~fg

        rgb_edge = _gradient_luma(rgb) > float(args.rgb_grad_threshold)
        depth_edge = _gradient_luma(depth) > float(args.depth_grad_threshold)
        transition = _dilate(fg, args.transition_radius) & _dilate(bg_candidate, args.transition_radius)
        edge = _dilate(rgb_edge | depth_edge, args.edge_dilate_radius)
        texture = torch.zeros_like(fg)
        if args.local_texture_variance_threshold >= 0.0:
            texture = _local_variance(_gray(rgb), args.local_texture_radius) > float(args.local_texture_variance_threshold)

        q_hit = _load_optional_q_hit(args.q_hit_dir, image_idx, image_path.name)
        q_hit_excluded = torch.zeros_like(fg)
        if q_hit is not None and args.q_hit_exclusion_threshold >= 0.0:
            if q_hit.shape[:2] != fg.shape[:2]:
                q_hit = F.interpolate(q_hit.permute(2, 0, 1)[None], size=fg.shape[:2], mode="nearest")[0].permute(1, 2, 0)
            q_hit_excluded = q_hit > float(args.q_hit_exclusion_threshold)

        excluded = edge | transition | texture | q_hit_excluded
        water = bg_candidate & ~excluded
        water = _connected_water_filter(
            water,
            top_only=bool(args.top_connected_only),
            side_min_area=float(args.side_connected_min_area),
        )
        water = _apply_water_y_limit(water, float(args.water_max_y_fraction))
        water = _remove_border(water, args.border)
        water = _erode(water, args.erosion_radius)

        object_core = _erode(fg & ~edge & ~texture, max(args.transition_radius, 1))
        boundary = _dilate(fg, args.transition_radius) & ~object_core
        boundary = boundary | edge | transition
        water = water & ~object_core & ~boundary
        boundary = boundary & ~water & ~object_core
        uncertain = ~(water | object_core | boundary)

        payload = {
            "water": water.bool(),
            "object": object_core.bool(),
            "boundary": boundary.bool(),
            "uncertain": uncertain.bool(),
            "image_index": image_idx,
            "image_name": image_path.name,
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "depth_source": str(depth_path),
            "foreground_depth_threshold": float(args.foreground_depth_threshold),
            "rgb_grad_threshold": float(args.rgb_grad_threshold),
            "depth_grad_threshold": float(args.depth_grad_threshold),
            "transition_radius": int(args.transition_radius),
            "edge_dilate_radius": int(args.edge_dilate_radius),
            "erosion_radius": int(args.erosion_radius),
            "border": int(args.border),
            "top_connected_only": bool(args.top_connected_only),
            "water_max_y_fraction": float(args.water_max_y_fraction),
            "side_connected_min_area": float(args.side_connected_min_area),
            "local_texture_variance_threshold": float(args.local_texture_variance_threshold),
            "q_hit_exclusion_threshold": float(args.q_hit_exclusion_threshold),
        }
        torch.save(payload, args.output_dir / f"view_{image_idx:04d}_regions.pt")
        if args.save_png:
            save_image(rgb.permute(2, 0, 1), args.output_dir / f"view_{image_idx:04d}_rgb.png")
            _save_mask(depth, args.output_dir / f"view_{image_idx:04d}_pseudo_depth.png")
            _save_mask(water, args.output_dir / f"view_{image_idx:04d}_water.png")
            _save_mask(object_core, args.output_dir / f"view_{image_idx:04d}_object.png")
            _save_mask(boundary, args.output_dir / f"view_{image_idx:04d}_boundary.png")
            _save_mask(uncertain, args.output_dir / f"view_{image_idx:04d}_uncertain.png")
            _save_overlay(rgb, water, boundary, uncertain, args.output_dir / f"view_{image_idx:04d}_water_overlay.png")

        denom = float(rgb.shape[0] * rgb.shape[1])
        summaries.append(
            {
                "image_index": image_idx,
                "image_name": image_path.name,
                "water_coverage": float(water.float().sum().item() / denom),
                "object_coverage": float(object_core.float().sum().item() / denom),
                "boundary_coverage": float(boundary.float().sum().item() / denom),
                "uncertain_coverage": float(uncertain.float().sum().item() / denom),
            }
        )

    contact_sheet = _save_contact_sheet(args.output_dir, len(summaries), args.contact_sheet_views) if args.save_png else None
    coverage = {
        key: {
            "mean": float(np.mean([row[key] for row in summaries])) if summaries else 0.0,
            "min": float(np.min([row[key] for row in summaries])) if summaries else 0.0,
            "max": float(np.max([row[key] for row in summaries])) if summaries else 0.0,
        }
        for key in ("water_coverage", "object_coverage", "boundary_coverage", "uncertain_coverage")
    }
    repo = Path(__file__).resolve().parents[2]
    metadata = {
        "mask_type": "high_precision_open_water",
        "git_commit": _git_commit(repo),
        "data": str(args.data),
        "load_config": str(args.load_config) if args.load_config is not None else None,
        "split": args.split,
        "images_path": args.images_path,
        "depth_dir": str(args.depth_dir),
        "output_dir": str(args.output_dir),
        "count": len(summaries),
        "coverage": coverage,
        "contact_sheet": contact_sheet,
        "parameters": {
            "foreground_depth_threshold": float(args.foreground_depth_threshold),
            "rgb_grad_threshold": float(args.rgb_grad_threshold),
            "depth_grad_threshold": float(args.depth_grad_threshold),
            "transition_radius": int(args.transition_radius),
            "edge_dilate_radius": int(args.edge_dilate_radius),
            "erosion_radius": int(args.erosion_radius),
            "border": int(args.border),
            "top_connected_only": bool(args.top_connected_only),
            "water_max_y_fraction": float(args.water_max_y_fraction),
            "side_connected_min_area": float(args.side_connected_min_area),
            "local_texture_radius": int(args.local_texture_radius),
            "local_texture_variance_threshold": float(args.local_texture_variance_threshold),
            "q_hit_dir": str(args.q_hit_dir) if args.q_hit_dir is not None else None,
            "q_hit_exclusion_threshold": float(args.q_hit_exclusion_threshold),
            "keep_largest_foreground": bool(args.keep_largest_foreground),
        },
        "images": summaries,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("undistorted_data/undistorted_IUI3-RedSea"))
    parser.add_argument("--images-path", type=str, default="images/ColorImage")
    parser.add_argument("--split", type=str, choices=["train", "eval", "val", "test"], default="train")
    parser.add_argument("--load-config", type=Path, default=None)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--foreground-depth-threshold", type=float, default=0.50)
    parser.add_argument("--rgb-grad-threshold", type=float, default=0.05)
    parser.add_argument("--depth-grad-threshold", type=float, default=0.05)
    parser.add_argument("--transition-radius", type=int, default=7)
    parser.add_argument("--edge-dilate-radius", type=int, default=5)
    parser.add_argument("--erosion-radius", type=int, default=13)
    parser.add_argument("--border", type=int, default=0)
    parser.add_argument("--top-connected-only", action="store_true")
    parser.add_argument("--water-max-y-fraction", type=float, default=1.0)
    parser.add_argument("--side-connected-min-area", type=float, default=0.0)
    parser.add_argument("--local-texture-radius", type=int, default=4)
    parser.add_argument("--local-texture-variance-threshold", type=float, default=-1.0)
    parser.add_argument("--q-hit-dir", type=Path, default=None)
    parser.add_argument("--q-hit-exclusion-threshold", type=float, default=-1.0)
    parser.add_argument("--keep-largest-foreground", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--contact-sheet-views", type=int, default=25)
    args = parser.parse_args()

    metadata = build_masks(args)
    print(
        json.dumps(
            {
                "mask_type": metadata["mask_type"],
                "split": metadata["split"],
                "count": metadata["count"],
                "coverage": metadata["coverage"],
                "contact_sheet": metadata["contact_sheet"],
                "output_dir": metadata["output_dir"],
            },
            indent=2,
        )
    )
    print(f"saved={args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
