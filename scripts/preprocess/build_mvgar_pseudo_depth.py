#!/usr/bin/env python
"""Build MV-GAR aligned pseudo-depth payloads in Nerfstudio train-view order."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class ViewInfo:
    image_idx: int
    image_name: str
    image_path: Path
    depth_path: Path
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation: np.ndarray
    raw_inverse: np.ndarray
    aligned_depth: Optional[np.ndarray] = None
    alignment: Optional[Dict[str, Any]] = None
    neighbor_ids: Optional[List[int]] = None


def _read_u16_depth(path: Path, height: int, width: int) -> np.ndarray:
    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != (height, width):
        arr_img = Image.fromarray(arr)
        arr = np.asarray(arr_img.resize((width, height), resample=Image.BILINEAR)).astype(np.float32)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    if finite.any():
        vals = arr[finite]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    return out


def _read_rgb(path: Path, height: int, width: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _gradient_map(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3 and value.shape[-1] == 3:
        value = 0.299 * value[..., :1] + 0.587 * value[..., 1:2] + 0.114 * value[..., 2:3]
    if value.ndim == 2:
        value = value[..., None]
    x = value.permute(2, 0, 1)[None].float()
    kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)[0, 0]


def _project_points(view: ViewInfo, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xc = (points - view.translation[None, :]) @ view.rotation
    z = -xc[:, 2]
    valid_z = z > 1e-6
    u = xc[:, 0] / np.maximum(z, 1e-6) * view.fx + view.cx
    v = xc[:, 1] / np.maximum(z, 1e-6) * view.fy + view.cy
    valid = valid_z & (u >= 0.0) & (u <= view.width - 1) & (v >= 0.0) & (v <= view.height - 1)
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32), valid


def _fit_affine_inverse_depth(
    raw_samples: np.ndarray,
    target_inverse: np.ndarray,
    *,
    min_points: int,
    ransac_trials: int,
    residual_threshold: float,
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    finite = np.isfinite(raw_samples) & np.isfinite(target_inverse) & (target_inverse > 0.0)
    x = raw_samples[finite].astype(np.float64)
    y = target_inverse[finite].astype(np.float64)
    if x.size < min_points:
        return None

    best_inliers: Optional[np.ndarray] = None
    best_count = 0
    for _ in range(max(ransac_trials, 1)):
        idx = rng.choice(x.size, size=2, replace=False)
        denom = x[idx[1]] - x[idx[0]]
        if abs(float(denom)) < 1e-8:
            continue
        a = (y[idx[1]] - y[idx[0]]) / denom
        b = y[idx[0]] - a * x[idx[0]]
        if not np.isfinite(a) or a <= 0.0 or not np.isfinite(b):
            continue
        pred = a * x + b
        rel = np.abs(pred - y) / np.maximum(y, 1e-8)
        inliers = rel < residual_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_points:
        return None
    x_in = x[best_inliers]
    y_in = y[best_inliers]
    A = np.stack([x_in, np.ones_like(x_in)], axis=1)
    a, b = np.linalg.lstsq(A, y_in, rcond=None)[0]
    if not np.isfinite(a) or a <= 0.0 or not np.isfinite(b):
        return None
    pred = a * x + b
    rel = np.abs(pred - y) / np.maximum(y, 1e-8)
    inliers = rel < residual_threshold
    return {
        "scale": float(a),
        "bias": float(b),
        "inlier_ratio": float(inliers.mean()),
        "median_relative_error": float(np.median(rel[inliers])) if inliers.any() else float(np.median(rel)),
        "num_sparse_samples": int(x.size),
        "num_inliers": int(inliers.sum()),
    }


def _align_view(
    view: ViewInfo,
    points: np.ndarray,
    *,
    min_sparse_points: int,
    ransac_trials: int,
    residual_threshold: float,
    max_median_relative_error: float,
    rng: np.random.Generator,
) -> None:
    u, v, z, valid = _project_points(view, points)
    if int(valid.sum()) < min_sparse_points:
        view.alignment = {"ok": False, "reason": "too_few_visible_sparse_points", "visible_sparse_points": int(valid.sum())}
        return
    xi = np.rint(u[valid]).astype(np.int64).clip(0, view.width - 1)
    yi = np.rint(v[valid]).astype(np.int64).clip(0, view.height - 1)
    raw = view.raw_inverse[yi, xi]
    inv_target = 1.0 / np.maximum(z[valid], 1e-6)
    candidates = []
    for name, samples in [("raw", raw), ("inverted_raw", 1.0 - raw)]:
        fit = _fit_affine_inverse_depth(
            samples,
            inv_target,
            min_points=min_sparse_points,
            ransac_trials=ransac_trials,
            residual_threshold=residual_threshold,
            rng=rng,
        )
        if fit is not None:
            fit["orientation"] = name
            candidates.append(fit)
    if not candidates:
        view.alignment = {"ok": False, "reason": "affine_fit_failed", "visible_sparse_points": int(valid.sum())}
        return
    fit = min(candidates, key=lambda item: (item["median_relative_error"], -item["inlier_ratio"]))
    if fit["inlier_ratio"] < 0.4 or fit["median_relative_error"] > max_median_relative_error:
        fit = {**fit, "ok": False, "reason": "fit_quality_gate_failed"}
        view.alignment = fit
        return
    source = view.raw_inverse if fit["orientation"] == "raw" else (1.0 - view.raw_inverse)
    inv_depth = fit["scale"] * source + fit["bias"]
    valid_inv = np.isfinite(inv_depth) & (inv_depth > 1e-6)
    depth = np.zeros_like(inv_depth, dtype=np.float32)
    depth[valid_inv] = (1.0 / np.maximum(inv_depth[valid_inv], 1e-6)).astype(np.float32)
    hi = float(np.percentile(depth[valid_inv], 99.5)) if valid_inv.any() else 0.0
    if hi > 0.0:
        depth = np.clip(depth, 0.0, hi).astype(np.float32)
    view.aligned_depth = depth
    view.alignment = {**fit, "ok": True, "visible_sparse_points": int(valid.sum())}


def _neighbor_ids(views: List[ViewInfo], image_idx: int, count: int) -> List[int]:
    center = views[image_idx].translation
    distances = []
    for other in views:
        if other.image_idx == image_idx or other.aligned_depth is None:
            continue
        dist = float(np.linalg.norm(other.translation - center))
        distances.append((dist, other.image_idx))
    distances.sort()
    return [idx for _, idx in distances[: max(count, 0)]]


def _cross_view_confidence(
    view: ViewInfo,
    views: List[ViewInfo],
    *,
    tau_depth: float,
    occ_delta: float,
    grid_stride: int,
) -> np.ndarray:
    if view.aligned_depth is None or not view.neighbor_ids:
        return np.zeros((view.height, view.width), dtype=np.float32)
    yy, xx = np.mgrid[0 : view.height : grid_stride, 0 : view.width : grid_stride].astype(np.float32)
    depth = view.aligned_depth[::grid_stride, ::grid_stride].astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not valid.any():
        return np.zeros((view.height, view.width), dtype=np.float32)

    xcam = (xx - view.cx) / view.fx * depth
    ycam = (yy - view.cy) / view.fy * depth
    camera_points = np.stack([xcam, ycam, -depth], axis=-1).reshape(-1, 3)
    world = camera_points @ view.rotation.T + view.translation[None, :]
    scores = []
    for neighbor_id in view.neighbor_ids:
        neighbor = views[neighbor_id]
        if neighbor.aligned_depth is None:
            continue
        u, v, z, inside = _project_points(neighbor, world)
        xi = np.rint(u).astype(np.int64).clip(0, neighbor.width - 1)
        yi = np.rint(v).astype(np.int64).clip(0, neighbor.height - 1)
        nd = neighbor.aligned_depth[yi, xi]
        ok = inside & np.isfinite(nd) & (nd > 0.0)
        occluded = ok & (z > nd * (1.0 + occ_delta))
        ok = ok & ~occluded
        err = np.abs(np.log(np.maximum(z, 1e-6)) - np.log(np.maximum(nd, 1e-6)))
        score = np.zeros_like(z, dtype=np.float32)
        score[ok] = np.exp(-err[ok] / max(tau_depth, 1e-6)).astype(np.float32)
        scores.append(score.reshape(depth.shape))
    if not scores:
        return np.zeros((view.height, view.width), dtype=np.float32)
    small = np.median(np.stack(scores, axis=0), axis=0).astype(np.float32)
    small[~valid] = 0.0
    tensor = torch.from_numpy(small)[None, None]
    full = F.interpolate(tensor, size=(view.height, view.width), mode="bilinear", align_corners=False)[0, 0].numpy()
    return np.clip(full, 0.0, 1.0).astype(np.float32)


def _confidence_maps(
    view: ViewInfo,
    views: List[ViewInfo],
    *,
    structure_mid: float,
    structure_temp: float,
    boundary_tau: float,
    tau_depth: float,
    occ_delta: float,
    grid_stride: int,
    skip_cross_view: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert view.aligned_depth is not None and view.alignment is not None
    rgb = _read_rgb(view.image_path, view.height, view.width)
    rgb_grad = _gradient_map(rgb)
    depth_tensor = torch.from_numpy(np.log(np.maximum(view.aligned_depth, 1e-6))).float()
    depth_grad = _gradient_map(depth_tensor)
    depth_grad_norm = depth_grad / torch.quantile(depth_grad.reshape(-1), 0.95).clamp_min(1e-6)
    rgb_grad_norm = rgb_grad / torch.quantile(rgb_grad.reshape(-1), 0.95).clamp_min(1e-6)
    structure = torch.sigmoid((torch.maximum(rgb_grad_norm, 0.5 * depth_grad_norm) - structure_mid) / max(structure_temp, 1e-6))
    boundary_safe = torch.exp(-depth_grad_norm / max(boundary_tau, 1e-6)).clamp(0.0, 1.0)
    base_conf = float(view.alignment.get("inlier_ratio", 0.0))
    if skip_cross_view:
        pseudo_conf = np.full((view.height, view.width), base_conf, dtype=np.float32)
    else:
        cross = _cross_view_confidence(
            view,
            views,
            tau_depth=tau_depth,
            occ_delta=occ_delta,
            grid_stride=grid_stride,
        )
        pseudo_conf = np.clip(base_conf * np.maximum(cross, 0.25 * (cross > 0.0)), 0.0, 1.0).astype(np.float32)
    return pseudo_conf, structure.numpy().astype(np.float32), boundary_safe.numpy().astype(np.float32)


def _save_debug_png(array: np.ndarray, path: Path) -> None:
    finite = np.isfinite(array)
    out = np.zeros_like(array, dtype=np.float32)
    if finite.any():
        vals = array[finite]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        out = np.clip((array - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    Image.fromarray((out * 255.0).astype(np.uint8)).save(path)


def _load_views(args: argparse.Namespace) -> Tuple[List[ViewInfo], np.ndarray]:
    from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

    config = ColmapDataParserConfig(
        data=args.data,
        images_path=Path(args.images_path),
        colmap_path=Path(args.colmap_path),
        downscale_factor=1,
        load_3D_points=True,
    )
    outputs = config.setup().get_dataparser_outputs(split="train")
    points = outputs.metadata["points3D_xyz"].detach().cpu().numpy().astype(np.float32)
    views: List[ViewInfo] = []
    for image_idx, image_path in enumerate(outputs.image_filenames):
        if args.max_images is not None and image_idx >= args.max_images:
            break
        height = int(outputs.cameras.height[image_idx].reshape(-1)[0].item())
        width = int(outputs.cameras.width[image_idx].reshape(-1)[0].item())
        c2w = outputs.cameras.camera_to_worlds[image_idx].detach().cpu().numpy().astype(np.float32)
        rotation = c2w[:3, :3]
        translation = c2w[:3, 3]
        depth_path = args.depth_dir / Path(image_path).name
        if not depth_path.exists():
            raise FileNotFoundError(depth_path)
        raw_inverse = _read_u16_depth(depth_path, height, width)
        views.append(
            ViewInfo(
                image_idx=image_idx,
                image_name=Path(image_path).name,
                image_path=Path(image_path),
                depth_path=depth_path,
                height=height,
                width=width,
                fx=float(outputs.cameras.fx[image_idx].reshape(-1)[0].item()),
                fy=float(outputs.cameras.fy[image_idx].reshape(-1)[0].item()),
                cx=float(outputs.cameras.cx[image_idx].reshape(-1)[0].item()),
                cy=float(outputs.cameras.cy[image_idx].reshape(-1)[0].item()),
                rotation=rotation,
                translation=translation,
                raw_inverse=raw_inverse,
            )
        )
    return views, points


def build_mvgar_pseudo_depth(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    views, points = _load_views(args)
    summaries: List[Dict[str, Any]] = []
    for view in views:
        _align_view(
            view,
            points,
            min_sparse_points=args.min_sparse_points,
            ransac_trials=args.ransac_trials,
            residual_threshold=args.ransac_residual_threshold,
            max_median_relative_error=args.max_median_relative_error,
            rng=rng,
        )
    for view in views:
        if view.aligned_depth is not None:
            view.neighbor_ids = _neighbor_ids(views, view.image_idx, args.num_neighbors)

    for view in views:
        alignment = view.alignment or {"ok": False, "reason": "not_processed"}
        if view.aligned_depth is None or not bool(alignment.get("ok", False)):
            confidence = np.zeros((view.height, view.width), dtype=np.float32)
            structure = np.zeros_like(confidence)
            boundary_safe = np.zeros_like(confidence)
            depth = np.zeros_like(confidence)
        else:
            depth = view.aligned_depth.astype(np.float32)
            confidence, structure, boundary_safe = _confidence_maps(
                view,
                views,
                structure_mid=args.structure_mid,
                structure_temp=args.structure_temp,
                boundary_tau=args.boundary_tau,
                tau_depth=args.cross_view_tau_depth,
                occ_delta=args.cross_view_occ_delta,
                grid_stride=args.cross_view_grid_stride,
                skip_cross_view=args.skip_cross_view,
            )
        payload = {
            "depth": torch.from_numpy(depth).half(),
            "pseudo_confidence": torch.from_numpy(confidence).half(),
            "structure_confidence": torch.from_numpy(structure).half(),
            "boundary_safe": torch.from_numpy(boundary_safe).half(),
            "alignment_inlier_ratio": float(alignment.get("inlier_ratio", 0.0)),
            "alignment_median_error": float(alignment.get("median_relative_error", 0.0)),
            "alignment_ok": bool(alignment.get("ok", False)),
            "alignment_reason": str(alignment.get("reason", "")),
            "neighbor_ids": torch.tensor(view.neighbor_ids or [], dtype=torch.long),
            "image_index": int(view.image_idx),
            "image_name": view.image_name,
            "depth_source": str(view.depth_path),
        }
        torch.save(payload, args.output_dir / f"view_{view.image_idx:04d}_mvgar.pt")
        if args.save_png:
            _save_debug_png(depth, args.output_dir / f"view_{view.image_idx:04d}_depth.png")
            _save_debug_png(confidence, args.output_dir / f"view_{view.image_idx:04d}_confidence.png")
            _save_debug_png(structure, args.output_dir / f"view_{view.image_idx:04d}_structure.png")
        coverage = float((confidence >= args.report_confidence_threshold).mean())
        summaries.append(
            {
                "image_index": int(view.image_idx),
                "image_name": view.image_name,
                "alignment": alignment,
                "neighbor_ids": view.neighbor_ids or [],
                "pseudo_valid_coverage": float((depth > 0.0).mean()),
                "pseudo_confidence_coverage": coverage,
                "pseudo_confidence_mean": float(confidence.mean()),
                "structure_confidence_mean": float(structure.mean()),
                "boundary_safe_mean": float(boundary_safe.mean()),
            }
        )
    metadata = {
        "type": "mvgar_pseudo_depth_bank",
        "data": str(args.data),
        "depth_dir": str(args.depth_dir),
        "output_dir": str(args.output_dir),
        "count": len(summaries),
        "views": summaries,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images-path", type=str, default="images/ColorImage")
    parser.add_argument("--colmap-path", type=str, default="sparse/0")
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sparse-points", type=int, default=32)
    parser.add_argument("--ransac-trials", type=int, default=96)
    parser.add_argument("--ransac-residual-threshold", type=float, default=0.15)
    parser.add_argument("--max-median-relative-error", type=float, default=0.15)
    parser.add_argument("--num-neighbors", type=int, default=4)
    parser.add_argument("--cross-view-tau-depth", type=float, default=0.08)
    parser.add_argument("--cross-view-occ-delta", type=float, default=0.05)
    parser.add_argument("--cross-view-grid-stride", type=int, default=4)
    parser.add_argument("--skip-cross-view", action="store_true")
    parser.add_argument("--structure-mid", type=float, default=0.25)
    parser.add_argument("--structure-temp", type=float, default=0.10)
    parser.add_argument("--boundary-tau", type=float, default=0.50)
    parser.add_argument("--report-confidence-threshold", type=float, default=0.50)
    parser.add_argument("--save-png", action="store_true")
    args = parser.parse_args()

    metadata = build_mvgar_pseudo_depth(args)
    ok = sum(1 for view in metadata["views"] if view["alignment"].get("ok", False))
    mean_conf = float(np.mean([view["pseudo_confidence_mean"] for view in metadata["views"]])) if metadata["views"] else 0.0
    print(json.dumps({"count": metadata["count"], "aligned_ok": ok, "mean_confidence": mean_conf}, indent=2))
    print(f"saved={args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
