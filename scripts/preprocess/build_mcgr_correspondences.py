#!/usr/bin/env python
"""Build MCGR cross-view correspondence payloads in Nerfstudio train-view order."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess.build_mvgar_pseudo_depth import ViewInfo, _gradient_map, _load_views, _read_rgb


def _resize_np(value: np.ndarray, height: int, width: int, *, mode: str = "bilinear") -> np.ndarray:
    if value.shape == (height, width):
        return value.astype(np.float32)
    tensor = torch.from_numpy(value.astype(np.float32))[None, None]
    if mode == "nearest":
        out = F.interpolate(tensor, size=(height, width), mode="nearest")
    else:
        out = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return out[0, 0].numpy().astype(np.float32)


def _load_aligned_depths(views: List[ViewInfo], pseudo_depth_dir: Path) -> None:
    for view in views:
        path = pseudo_depth_dir / f"view_{view.image_idx:04d}_mvgar.pt"
        if not path.exists():
            view.aligned_depth = None
            view.alignment = {"ok": False, "reason": "missing_mvgar_pseudo_depth", "path": str(path)}
            continue
        payload = torch.load(path, map_location="cpu")
        depth = torch.as_tensor(payload.get("depth", torch.empty(0))).float().numpy()
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.size == 0:
            view.aligned_depth = None
            view.alignment = {"ok": False, "reason": "empty_mvgar_pseudo_depth", "path": str(path)}
            continue
        depth = _resize_np(depth, view.height, view.width)
        valid = np.isfinite(depth) & (depth > 0.0)
        view.aligned_depth = np.where(valid, depth, 0.0).astype(np.float32)
        view.alignment = {
            "ok": bool(payload.get("alignment_ok", valid.any())),
            "reason": str(payload.get("alignment_reason", "")),
            "alignment_inlier_ratio": float(payload.get("alignment_inlier_ratio", 0.0)),
            "alignment_median_error": float(payload.get("alignment_median_error", 0.0)),
        }


def _bilinear_sample(value: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = value.shape[:2]
    finite_uv = np.isfinite(u) & np.isfinite(v)
    inside = finite_uv & (u >= 0.0) & (u <= w - 1) & (v >= 0.0) & (v <= h - 1)
    u0 = np.floor(np.clip(u, 0.0, w - 1)).astype(np.int64)
    v0 = np.floor(np.clip(v, 0.0, h - 1)).astype(np.int64)
    u1 = np.clip(u0 + 1, 0, w - 1)
    v1 = np.clip(v0 + 1, 0, h - 1)
    wu = np.clip(u - u0.astype(np.float32), 0.0, 1.0).astype(np.float32)
    wv = np.clip(v - v0.astype(np.float32), 0.0, 1.0).astype(np.float32)
    top = value[v0, u0] * (1.0 - wu) + value[v0, u1] * wu
    bot = value[v1, u0] * (1.0 - wu) + value[v1, u1] * wu
    sampled = top * (1.0 - wv) + bot * wv
    sampled = sampled.astype(np.float32)
    sampled[~inside] = 0.0
    return sampled, inside


def _project_points(view: ViewInfo, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xc = (points - view.translation[None, :]) @ view.rotation
    z = -xc[:, 2]
    valid_z = z > 1e-6
    u = xc[:, 0] / np.maximum(z, 1e-6) * view.fx + view.cx
    v = xc[:, 1] / np.maximum(z, 1e-6) * view.fy + view.cy
    inside = valid_z & (u >= 0.0) & (u <= view.width - 1) & (v >= 0.0) & (v <= view.height - 1)
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32), inside


def _backproject(view: ViewInfo, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
    xcam = (u - view.cx) / view.fx * depth
    ycam = (v - view.cy) / view.fy * depth
    cam = np.stack([xcam, ycam, -depth], axis=-1).reshape(-1, 3)
    world = cam @ view.rotation.T + view.translation[None, :]
    return world.astype(np.float32)


def _robust01(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size < 8:
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.percentile(flat, 50.0))
    hi = float(np.percentile(flat, 95.0))
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def _sigmoid_np(value: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


def _load_structure_maps(views: List[ViewInfo]) -> Dict[int, np.ndarray]:
    maps: Dict[int, np.ndarray] = {}
    for view in views:
        rgb = _read_rgb(view.image_path, view.height, view.width)
        grad = _gradient_map(rgb).float().numpy()
        maps[view.image_idx] = _robust01(grad)
    return maps


def _read_colmap_tracks(data: Path, colmap_path: Path) -> Tuple[Dict[str, set[int]], Dict[int, np.ndarray]]:
    from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary, read_points3D_binary

    sparse_dir = data / colmap_path
    images = read_images_binary(sparse_dir / "images.bin")
    points = read_points3D_binary(sparse_dir / "points3D.bin")
    image_tracks: Dict[str, set[int]] = {}
    for image in images.values():
        ids = np.asarray(image.point3D_ids)
        valid = ids[ids >= 0].astype(np.int64)
        image_tracks[Path(image.name).name] = set(int(x) for x in valid.tolist())
    point_xyz = {int(pid): np.asarray(point.xyz, dtype=np.float32) for pid, point in points.items()}
    return image_tracks, point_xyz


def _median_baseline_angle(
    view_a: ViewInfo,
    view_b: ViewInfo,
    shared_ids: List[int],
    point_xyz: Dict[int, np.ndarray],
    *,
    max_points: int = 512,
) -> float:
    pts = [point_xyz[pid] for pid in shared_ids[:max_points] if pid in point_xyz]
    if len(pts) < 4:
        return 0.0
    xyz = np.stack(pts, axis=0)
    ra = xyz - view_a.translation[None, :]
    rb = xyz - view_b.translation[None, :]
    ra = ra / np.maximum(np.linalg.norm(ra, axis=-1, keepdims=True), 1e-8)
    rb = rb / np.maximum(np.linalg.norm(rb, axis=-1, keepdims=True), 1e-8)
    cos = np.sum(ra * rb, axis=-1).clip(-1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(cos))))


def _build_neighbor_graph(
    views: List[ViewInfo],
    *,
    data: Path,
    colmap_path: Path,
    neighbor_count: int,
    min_shared_tracks: int,
    min_baseline_angle: float,
    max_baseline_angle: float,
    target_baseline_angle: float,
    baseline_sigma: float,
) -> Dict[int, List[Dict[str, Any]]]:
    image_tracks, point_xyz = _read_colmap_tracks(data, colmap_path)
    graph: Dict[int, List[Dict[str, Any]]] = {}
    for view in views:
        tracks_i = image_tracks.get(Path(view.image_name).name, set())
        candidates: List[Dict[str, Any]] = []
        if view.aligned_depth is None or not tracks_i:
            graph[view.image_idx] = []
            continue
        for other in views:
            if other.image_idx == view.image_idx or other.aligned_depth is None:
                continue
            tracks_j = image_tracks.get(Path(other.image_name).name, set())
            shared = list(tracks_i & tracks_j)
            shared_count = len(shared)
            if shared_count < min_shared_tracks:
                continue
            angle = _median_baseline_angle(view, other, shared, point_xyz)
            if angle < min_baseline_angle or angle > max_baseline_angle:
                continue
            track_score = shared_count / max(math.sqrt(max(len(tracks_i), 1) * max(len(tracks_j), 1)), 1e-6)
            baseline_score = math.exp(-((angle - target_baseline_angle) ** 2) / (2.0 * max(baseline_sigma, 1e-6) ** 2))
            candidates.append(
                {
                    "image_idx": int(other.image_idx),
                    "image_name": other.image_name,
                    "shared_tracks": int(shared_count),
                    "baseline_angle_deg": float(angle),
                    "score": float(track_score * baseline_score),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        graph[view.image_idx] = candidates[: max(int(neighbor_count), 0)]
    return graph


def _top2_mean(conf: np.ndarray, valid: np.ndarray, min_valid_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    if conf.shape[0] == 0:
        h, w = conf.shape[1:]
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.uint8)
    valid_count = valid.sum(axis=0).astype(np.uint8)
    masked = np.where(valid, conf, -np.inf)
    k = min(2, masked.shape[0])
    top = np.partition(masked, kth=masked.shape[0] - k, axis=0)[-k:]
    top = np.where(np.isfinite(top), top, 0.0)
    cross = top.mean(axis=0).astype(np.float32)
    cross[valid_count < max(int(min_valid_neighbors), 1)] = 0.0
    return cross, valid_count


def _stats(values: np.ndarray) -> Dict[str, float]:
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(flat)),
        "p50": float(np.percentile(flat, 50.0)),
        "p90": float(np.percentile(flat, 90.0)),
        "p95": float(np.percentile(flat, 95.0)),
    }


def _save_debug_png(array: np.ndarray, path: Path) -> None:
    arr = _robust01(array.astype(np.float32))
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def _build_view_payload(
    view: ViewInfo,
    views_by_idx: Dict[int, ViewInfo],
    graph: Dict[int, List[Dict[str, Any]]],
    structure_maps: Dict[int, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    stride = max(int(args.correspondence_stride), 1)
    h_small = max(view.height // stride, 1)
    w_small = max(view.width // stride, 1)
    neighbor_items = graph.get(view.image_idx, [])
    neighbor_ids = [int(item["image_idx"]) for item in neighbor_items]
    yy, xx = np.mgrid[0:h_small, 0:w_small].astype(np.float32)
    u0 = (xx + 0.5) * stride - 0.5
    v0 = (yy + 0.5) * stride - 0.5
    if view.aligned_depth is None:
        depth_i = np.zeros((h_small, w_small), dtype=np.float32)
        valid_i = np.zeros_like(depth_i, dtype=bool)
    else:
        depth_i, inside_i = _bilinear_sample(view.aligned_depth, u0, v0)
        valid_i = inside_i & np.isfinite(depth_i) & (depth_i > 0.0)
    world = _backproject(view, u0, v0, depth_i)
    grad_i, _ = _bilinear_sample(structure_maps[view.image_idx], u0, v0)
    hf_confidence = _sigmoid_np((grad_i - float(args.hf_mid)) / max(float(args.hf_temp), 1e-6))

    corr_uv: List[np.ndarray] = []
    corr_conf: List[np.ndarray] = []
    corr_valid: List[np.ndarray] = []
    valid_log_errors: List[np.ndarray] = []
    valid_cycle_errors: List[np.ndarray] = []
    occlusion_reject = 0
    front_reject = 0
    total_considered = 0

    for neighbor_id in neighbor_ids:
        neighbor = views_by_idx[neighbor_id]
        u_j, v_j, z_j, inside_j = _project_points(neighbor, world.reshape(-1, 3))
        u_j = u_j.reshape(h_small, w_small)
        v_j = v_j.reshape(h_small, w_small)
        z_j = z_j.reshape(h_small, w_small)
        inside_j = inside_j.reshape(h_small, w_small)
        if neighbor.aligned_depth is None:
            depth_j = np.zeros_like(z_j, dtype=np.float32)
            inside_depth_j = np.zeros_like(z_j, dtype=bool)
        else:
            depth_j, inside_depth_j = _bilinear_sample(neighbor.aligned_depth, u_j, v_j)
        grad_j, _ = _bilinear_sample(structure_maps[neighbor_id], u_j, v_j)

        base_valid = (
            valid_i
            & inside_j
            & inside_depth_j
            & np.isfinite(z_j)
            & (z_j > 0.0)
            & np.isfinite(depth_j)
            & (depth_j > 0.0)
        )
        total_considered += int(base_valid.sum())
        behind = base_valid & (z_j > depth_j * (1.0 + float(args.occlusion_delta)))
        in_front = base_valid & (z_j < depth_j * (1.0 - float(args.front_delta)))
        occlusion_reject += int(behind.sum())
        front_reject += int(in_front.sum())
        depth_valid = base_valid & (~behind) & (~in_front)
        log_error = np.abs(np.log(np.maximum(z_j, 1e-6)) - np.log(np.maximum(depth_j, 1e-6))).astype(np.float32)
        depth_conf = np.exp(-log_error / max(float(args.depth_tau), 1e-6)).astype(np.float32)

        world_back = _backproject(neighbor, u_j, v_j, depth_j)
        u_back, v_back, _, inside_back = _project_points(view, world_back.reshape(-1, 3))
        u_back = u_back.reshape(h_small, w_small)
        v_back = v_back.reshape(h_small, w_small)
        inside_back = inside_back.reshape(h_small, w_small)
        cycle_error = np.sqrt((u_back - u0) ** 2 + (v_back - v0) ** 2).astype(np.float32)
        cycle_conf = np.exp(-cycle_error / max(float(args.cycle_tau), 1e-6)).astype(np.float32)
        cycle_valid = inside_back & (cycle_error <= float(args.cycle_hard_threshold))
        structure_conf = np.exp(-np.abs(grad_i - grad_j) / max(float(args.structure_tau), 1e-6)).astype(np.float32)
        valid = depth_valid & cycle_valid
        confidence = np.clip(depth_conf * cycle_conf * structure_conf, 0.0, 1.0).astype(np.float32)
        confidence[~valid] = 0.0
        uv_low = np.stack([(u_j + 0.5) / stride - 0.5, (v_j + 0.5) / stride - 0.5], axis=-1).astype(np.float32)
        corr_uv.append(uv_low)
        corr_conf.append(confidence)
        corr_valid.append(valid)
        if valid.any():
            valid_log_errors.append(log_error[valid])
            valid_cycle_errors.append(cycle_error[valid])

    if corr_conf:
        conf_arr = np.stack(corr_conf, axis=0).astype(np.float32)
        valid_arr = np.stack(corr_valid, axis=0).astype(bool)
        uv_arr = np.stack(corr_uv, axis=0).astype(np.float32)
    else:
        conf_arr = np.zeros((0, h_small, w_small), dtype=np.float32)
        valid_arr = np.zeros((0, h_small, w_small), dtype=bool)
        uv_arr = np.zeros((0, h_small, w_small, 2), dtype=np.float32)
    cross_view_confidence, valid_neighbor_count = _top2_mean(conf_arr, valid_arr, args.min_valid_neighbors)
    support = valid_neighbor_count >= max(int(args.min_valid_neighbors), 1)
    log_errors = np.concatenate(valid_log_errors) if valid_log_errors else np.zeros((0,), dtype=np.float32)
    cycle_errors = np.concatenate(valid_cycle_errors) if valid_cycle_errors else np.zeros((0,), dtype=np.float32)
    support_count = int(support.sum())
    open_water_support = support & (hf_confidence < float(args.open_water_hf_threshold))
    structure_support = support & (hf_confidence >= float(args.open_water_hf_threshold))
    summary = {
        "image_index": int(view.image_idx),
        "image_name": view.image_name,
        "neighbor_ids": neighbor_ids,
        "neighbors": neighbor_items,
        "two_neighbor_valid_coverage": float((valid_neighbor_count >= 2).mean()),
        "three_neighbor_valid_coverage": float((valid_neighbor_count >= 3).mean()),
        "cross_view_confidence": _stats(cross_view_confidence[support]) if support_count else _stats(cross_view_confidence),
        "log_depth_error": _stats(log_errors),
        "cycle_error": _stats(cycle_errors),
        "occlusion_rejection_ratio": float(occlusion_reject / max(total_considered, 1)),
        "front_rejection_ratio": float(front_reject / max(total_considered, 1)),
        "open_water_support_ratio": float(open_water_support.sum() / max(support_count, 1)),
        "structure_region_support_ratio": float(structure_support.sum() / max(support_count, 1)),
        "support_pixel_count": support_count,
    }
    payload = {
        "neighbor_ids": torch.tensor(neighbor_ids, dtype=torch.long),
        "corr_uv": torch.from_numpy(uv_arr).half(),
        "corr_confidence": torch.from_numpy(conf_arr).half(),
        "corr_valid": torch.from_numpy(valid_arr.astype(np.uint8)),
        "valid_neighbor_count": torch.from_numpy(valid_neighbor_count.astype(np.uint8)),
        "cross_view_confidence": torch.from_numpy(cross_view_confidence.astype(np.float32)).half(),
        "hf_confidence": torch.from_numpy(hf_confidence.astype(np.float32)).half(),
        "image_index": torch.tensor(int(view.image_idx), dtype=torch.long),
    }
    return payload, summary


def build_mcgr_correspondences(args: argparse.Namespace) -> Dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views, _ = _load_views(args)
    _load_aligned_depths(views, args.pseudo_depth_dir)
    graph = _build_neighbor_graph(
        views,
        data=args.data,
        colmap_path=Path(args.colmap_path),
        neighbor_count=args.neighbor_count,
        min_shared_tracks=args.min_shared_tracks,
        min_baseline_angle=args.min_baseline_angle,
        max_baseline_angle=args.max_baseline_angle,
        target_baseline_angle=args.target_baseline_angle,
        baseline_sigma=args.baseline_sigma,
    )
    structure_maps = _load_structure_maps(views)
    views_by_idx = {view.image_idx: view for view in views}
    summaries: List[Dict[str, Any]] = []
    for view in views:
        payload, summary = _build_view_payload(view, views_by_idx, graph, structure_maps, args)
        torch.save(payload, args.output_dir / f"view_{view.image_idx:04d}_mcgr.pt")
        summaries.append(summary)
        if args.save_png:
            _save_debug_png(
                payload["cross_view_confidence"].float().numpy(),
                args.output_dir / f"view_{view.image_idx:04d}_cross_conf.png",
            )
            _save_debug_png(
                payload["valid_neighbor_count"].float().numpy(),
                args.output_dir / f"view_{view.image_idx:04d}_valid_neighbors.png",
            )
            _save_debug_png(
                payload["hf_confidence"].float().numpy(),
                args.output_dir / f"view_{view.image_idx:04d}_hf_conf.png",
            )

    def mean_key(key: str) -> float:
        vals = [float(item[key]) for item in summaries if key in item]
        return float(np.mean(vals)) if vals else 0.0

    metadata = {
        "type": "mcgr_correspondence_bank",
        "data": str(args.data),
        "pseudo_depth_dir": str(args.pseudo_depth_dir),
        "output_dir": str(args.output_dir),
        "count": len(summaries),
        "summary": {
            "two_neighbor_valid_coverage_mean": mean_key("two_neighbor_valid_coverage"),
            "three_neighbor_valid_coverage_mean": mean_key("three_neighbor_valid_coverage"),
            "open_water_support_ratio_mean": mean_key("open_water_support_ratio"),
            "structure_region_support_ratio_mean": mean_key("structure_region_support_ratio"),
        },
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
    parser.add_argument("--depth-dir", type=Path, default=None)
    parser.add_argument("--pseudo-depth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--min-shared-tracks", type=int, default=30)
    parser.add_argument("--min-baseline-angle", type=float, default=2.0)
    parser.add_argument("--max-baseline-angle", type=float, default=30.0)
    parser.add_argument("--target-baseline-angle", type=float, default=10.0)
    parser.add_argument("--baseline-sigma", type=float, default=8.0)
    parser.add_argument("--depth-tau", type=float, default=0.08)
    parser.add_argument("--cycle-tau", type=float, default=1.5)
    parser.add_argument("--cycle-hard-threshold", type=float, default=3.0)
    parser.add_argument("--occlusion-delta", type=float, default=0.05)
    parser.add_argument("--front-delta", type=float, default=0.10)
    parser.add_argument("--structure-tau", type=float, default=0.25)
    parser.add_argument("--hf-mid", type=float, default=0.15)
    parser.add_argument("--hf-temp", type=float, default=0.05)
    parser.add_argument("--open-water-hf-threshold", type=float, default=0.20)
    parser.add_argument("--correspondence-stride", type=int, default=4)
    parser.add_argument("--min-valid-neighbors", type=int, default=2)
    parser.add_argument("--save-png", action="store_true")
    args = parser.parse_args()
    if args.depth_dir is None:
        args.depth_dir = args.data / "depthAnything_u16"

    metadata = build_mcgr_correspondences(args)
    print(
        json.dumps(
            {
                "count": metadata["count"],
                **metadata["summary"],
            },
            indent=2,
        )
    )
    print(f"saved={args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
