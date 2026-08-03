#!/usr/bin/env python
"""Render M1 medium parameter maps on eval cameras.

This diagnostic loads a WaterSplatting checkpoint, runs the eval/novel-view
cameras through ``model.get_outputs(camera)``, saves raw medium parameters, and
exports consistently normalized PNG maps. The visualization ranges are computed
globally across all rendered eval views rather than per image.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup


CHANNEL_NAMES = ("r", "g", "b")
VIS_KEYS = (
    "medium_bs",
    "medium_attn",
    "b_inf",
    "transmission",
    "backscatter_endpoint",
    "actual_rgb_medium",
)
RGB_FILENAMES = {
    "medium_bs": "sigma_bs_rgb.png",
    "medium_attn": "sigma_attn_rgb.png",
    "b_inf": "b_inf.png",
    "transmission": "transmission.png",
    "backscatter_endpoint": "backscatter_endpoint.png",
    "actual_rgb_medium": "actual_rgb_medium.png",
}
CHANNEL_PREFIXES = {
    "medium_bs": "sigma_bs",
    "medium_attn": "sigma_attn",
}


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _camera_index(camera: Any, outputs: Dict[str, torch.Tensor], fallback: int) -> int:
    if "camera_index" in outputs:
        value = outputs["camera_index"]
        if torch.is_tensor(value):
            return int(value.detach().cpu().reshape(-1)[0].item())
        return int(value)
    if getattr(camera, "metadata", None) is not None and "cam_idx" in camera.metadata:
        value = camera.metadata["cam_idx"]
        if torch.is_tensor(value):
            return int(value.detach().cpu().reshape(-1)[0].item())
        return int(value)
    return int(fallback)


def _as_hwc(tensor: torch.Tensor, name: str, channels: int | None = 3) -> torch.Tensor:
    value = tensor.detach().float()
    if value.ndim == 2:
        value = value[..., None]
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape HxW or HxWxC, got {tuple(value.shape)}")
    if channels is not None and value.shape[-1] != channels:
        raise ValueError(f"{name} must have {channels} channels, got shape {tuple(value.shape)}")
    return value


def _finite_flat(value: torch.Tensor) -> torch.Tensor:
    flat = value.detach().float().reshape(-1).cpu()
    return flat[torch.isfinite(flat)]


def _quantile_dict(value: torch.Tensor, quantiles: Iterable[float]) -> Dict[str, float]:
    flat = _finite_flat(value)
    if flat.numel() == 0:
        return {f"p{int(q * 100):02d}": 0.0 for q in quantiles}
    vals = _nearest_rank_quantiles(flat, quantiles)
    return {f"p{int(q * 100):02d}": float(v) for q, v in vals.items()}


def _nearest_rank_quantiles(flat: torch.Tensor, quantiles: Iterable[float]) -> Dict[float, float]:
    flat = _finite_flat(flat)
    if flat.numel() == 0:
        return {float(q): 0.0 for q in quantiles}
    n = flat.numel()
    out: Dict[float, float] = {}
    for q in quantiles:
        q_float = float(q)
        rank = max(1, min(n, math.ceil(q_float * n)))
        out[q_float] = float(flat.kthvalue(rank).values.item())
    return out


def _channel_stats(value: torch.Tensor) -> Dict[str, Dict[str, float]]:
    value = _as_hwc(value, "stats_value", channels=None)
    stats: Dict[str, Dict[str, float]] = {}
    for channel_idx in range(value.shape[-1]):
        name = CHANNEL_NAMES[channel_idx] if channel_idx < len(CHANNEL_NAMES) else f"c{channel_idx}"
        flat = _finite_flat(value[..., channel_idx])
        if flat.numel() == 0:
            stats[name] = {"mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
            continue
        quantiles = _nearest_rank_quantiles(flat, (0.05, 0.50, 0.95))
        stats[name] = {
            "mean": float(flat.mean().item()),
            "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
            "p05": quantiles[0.05],
            "p50": quantiles[0.50],
            "p95": quantiles[0.95],
        }
    return stats


def _mean_rgb(value: torch.Tensor) -> List[float]:
    value = _as_hwc(value, "mean_rgb", channels=None)
    means = []
    for channel_idx in range(value.shape[-1]):
        flat = _finite_flat(value[..., channel_idx])
        means.append(float(flat.mean().item()) if flat.numel() else 0.0)
    return means


def _sample_for_range(value: torch.Tensor, max_values: int) -> torch.Tensor:
    flat = _finite_flat(value)
    if max_values <= 0 or flat.numel() <= max_values:
        return flat
    # Deterministic evenly spaced sample. This keeps memory bounded while still
    # covering the whole image stack when exact quantiles are too large.
    indices = torch.linspace(0, flat.numel() - 1, steps=max_values).round().long()
    return flat[indices]


def _global_range(values: List[torch.Tensor]) -> Dict[str, float]:
    non_empty = [value for value in values if value.numel() > 0]
    if not non_empty:
        return {"p01": 0.0, "p50": 0.0, "p99": 1.0}
    flat = torch.cat(non_empty)
    quantiles = _nearest_rank_quantiles(flat, (0.01, 0.50, 0.99))
    p01, p50, p99 = quantiles[0.01], quantiles[0.50], quantiles[0.99]
    if p99 <= p01:
        p99 = p01 + 1e-6
    return {"p01": p01, "p50": p50, "p99": p99}


def _normalize_for_vis(value: torch.Tensor, vis_range: Dict[str, float]) -> torch.Tensor:
    lo = float(vis_range["p01"])
    hi = float(vis_range["p99"])
    denom = max(hi - lo, 1e-12)
    return ((value.detach().float() - lo) / denom).clamp(0.0, 1.0)


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if arr.ndim == 2:
        return (arr * 255.0 + 0.5).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return (arr[..., 0] * 255.0 + 0.5).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return (arr * 255.0 + 0.5).astype(np.uint8)
    raise ValueError(f"Cannot save image with shape {arr.shape}")


def _save_png(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(image)).save(path)


def _view_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    actual = payload["actual_rgb_medium"]
    stats = {
        "camera_index": int(payload["camera_index"]),
        "medium_bs": _channel_stats(payload["medium_bs"]),
        "medium_attn": _channel_stats(payload["medium_attn"]),
        "b_inf": _channel_stats(payload["b_inf"]),
        "transmission": _channel_stats(payload["transmission"]),
        "backscatter_endpoint": _channel_stats(payload["backscatter_endpoint"]),
        "actual_rgb_medium": _channel_stats(actual),
        "actual_backscatter_mean_rgb": _mean_rgb(actual),
        "actual_backscatter_mean": float(_finite_flat(actual).mean().item()),
        "depth": _channel_stats(payload["depth"]),
    }
    return stats


def _render_payload(model: Any, camera: Any, fallback_index: int) -> Dict[str, Any]:
    outputs = model.get_outputs(camera)
    sigma_bs = _as_hwc(outputs["medium_bs"], "medium_bs", channels=3)
    sigma_attn = _as_hwc(outputs["medium_attn"], "medium_attn", channels=3)
    b_inf = _as_hwc(outputs.get("b_inf", outputs["medium_rgb"]), "b_inf", channels=3)
    depth = _as_hwc(outputs["depth"], "depth", channels=None)
    if depth.shape[-1] != 1:
        depth = depth.mean(dim=-1, keepdim=True)
    actual_rgb_medium = _as_hwc(outputs["rgb_medium"], "actual_rgb_medium", channels=3)
    transmission = torch.exp(-sigma_attn * depth)
    backscatter_endpoint = b_inf * (1.0 - torch.exp(-sigma_bs * depth))
    return {
        "camera_index": _camera_index(camera, outputs, fallback_index),
        "medium_bs": sigma_bs.detach().cpu(),
        "medium_attn": sigma_attn.detach().cpu(),
        "b_inf": b_inf.detach().cpu(),
        "depth": depth.detach().cpu(),
        "transmission": transmission.detach().cpu(),
        "backscatter_endpoint": backscatter_endpoint.detach().cpu(),
        "actual_rgb_medium": actual_rgb_medium.detach().cpu(),
    }


def _write_view_pngs(view_dir: Path, payload: Dict[str, Any], ranges: Dict[str, Dict[str, float]]) -> None:
    for key in VIS_KEYS:
        _save_png(view_dir / RGB_FILENAMES[key], _normalize_for_vis(payload[key], ranges[key]))

    for key, prefix in CHANNEL_PREFIXES.items():
        for idx, channel_name in enumerate(CHANNEL_NAMES):
            _save_png(
                view_dir / f"{prefix}_{channel_name}.png",
                _normalize_for_vis(payload[key][..., idx], ranges[key]),
            )


def _load_payload(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def render_medium_maps(args: argparse.Namespace) -> Dict[str, Any]:
    def _update_config(config: Any) -> Any:
        if args.load_step is not None:
            config.load_step = int(args.load_step)
        return config

    config, pipeline, checkpoint_path, step = eval_setup(
        args.load_config,
        eval_num_rays_per_chunk=None,
        test_mode=args.test_mode,
        update_config_callback=_update_config,
    )
    pipeline.eval()
    model = pipeline.model
    device = model.device

    args.output_dir.mkdir(parents=True, exist_ok=True)
    range_values: Dict[str, List[torch.Tensor]] = {key: [] for key in VIS_KEYS}
    views: List[Dict[str, Any]] = []
    max_count = args.max_images if args.max_images > 0 else 10**9

    with torch.no_grad():
        for image_idx, (camera, _batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_count:
                break
            camera = camera.to(device) if hasattr(camera, "to") else camera
            view_dir = args.output_dir / f"view_{image_idx:04d}"
            view_dir.mkdir(parents=True, exist_ok=True)

            payload = _render_payload(model, camera, image_idx)
            pt_path = view_dir / "medium_parameters.pt"
            torch.save(payload, pt_path)

            stats = _view_stats(payload)
            stats_path = view_dir / "medium_parameter_stats.json"
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf8")

            for key in VIS_KEYS:
                range_values[key].append(_sample_for_range(payload[key], args.range_sample_max_values))

            views.append(
                {
                    "image_index": image_idx,
                    "camera_index": int(payload["camera_index"]),
                    "view_dir": str(view_dir),
                    "medium_parameters": str(pt_path),
                    "stats_json": str(stats_path),
                }
            )

    ranges = {key: _global_range(values) for key, values in range_values.items()}

    for view in views:
        view_dir = Path(view["view_dir"])
        payload = _load_payload(view_dir / "medium_parameters.pt")
        _write_view_pngs(view_dir, payload, ranges)

    aggregate_stats: Dict[str, Any] = {}
    for key in VIS_KEYS:
        values = range_values[key]
        aggregate_stats[key] = _quantile_dict(torch.cat(values) if values else torch.empty(0), (0.01, 0.50, 0.99))

    result = {
        "experiment_name": getattr(config, "experiment_name", ""),
        "method_name": getattr(config, "method_name", ""),
        "load_config": str(args.load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "test_mode": args.test_mode,
        "max_images": int(args.max_images),
        "view_count": len(views),
        "b_inf_mode_note": "For M1 b_inf_mode=tied, B_inf(r)=A(r)=medium_rgb(r).",
        "global_visualization_ranges": ranges,
        "global_quantiles": aggregate_stats,
        "range_sample_max_values": int(args.range_sample_max_values),
        "views": views,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    summary_path = args.output_dir / "medium_parameter_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--test-mode", default="inference")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all eval views.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--range-sample-max-values",
        type=int,
        default=0,
        help="0 uses exact global quantiles; positive values use a deterministic sample per view/map.",
    )
    args = parser.parse_args()

    result = render_medium_maps(args)
    print(
        json.dumps(
            {
                "checkpoint": result["checkpoint"],
                "step": result["step"],
                "view_count": result["view_count"],
                "output_dir": str(args.output_dir),
                "global_visualization_ranges": result["global_visualization_ranges"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
