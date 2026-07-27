#!/usr/bin/env python
"""Summarize saved eval render components under region masks.

This diagnostic is intentionally offline: it reads the PNGs written by ns-eval
and the high-precision eval masks, so it can be run across completed experiments
without loading checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image


RGB_COMPONENTS = (
    "gt",
    "rgb",
    "rgb_object",
    "rgb_medium_total",
    "rgb_medium_finite",
    "rgb_tail",
    "J",
    "J_raw",
    "J_gaussian_raw",
    "accumulation",
)
MASK_KEYS = ("water", "object", "boundary", "uncertain")


def _load_rgb(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if mask.shape != shape:
        mask_img = Image.open(path).convert("L").resize((shape[1], shape[0]), Image.NEAREST)
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    return (mask > 0.5).astype(np.float32)


def _stats(values: np.ndarray) -> Dict[str, float]:
    values = values.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _masked_values(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    keep = mask > 0.5
    if not np.any(keep):
        return image.reshape(0, image.shape[-1])
    return image[keep]


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _component_stats(image: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    vals = _masked_values(image, mask)
    if vals.size == 0:
        return {
            "rgb_mean": [0.0, 0.0, 0.0],
            "luma": _stats(np.asarray([], dtype=np.float32)),
            "blue_dominance_fraction": 0.0,
            "green_dominance_fraction": 0.0,
        }
    blue_dom = vals[:, 2] - np.maximum(vals[:, 0], vals[:, 1])
    green_dom = vals[:, 1] - np.maximum(vals[:, 0], vals[:, 2])
    return {
        "rgb_mean": [float(x) for x in vals.mean(axis=0)],
        "luma": _stats(_luma(vals)),
        "blue_dominance_fraction": float((blue_dom > 0.05).mean()),
        "green_dominance_fraction": float((green_dom > 0.05).mean()),
    }


def _masked_rgb_l1(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    vals = _masked_values(np.abs(pred - gt), mask)
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def _iter_view_indices(render_dir: Path) -> Iterable[int]:
    for path in sorted(render_dir.glob("eval_gt_*.png")):
        try:
            yield int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue


def diagnose(render_dir: Path, mask_dir: Path) -> Dict[str, object]:
    images: List[Dict[str, object]] = []
    aggregate_accumulator: Dict[str, List[float]] = {}

    for view_index in _iter_view_indices(render_dir):
        gt_path = render_dir / f"eval_gt_{view_index:04d}.png"
        if not gt_path.exists():
            continue
        gt = _load_rgb(gt_path)
        h, w = gt.shape[:2]

        masks = {
            key: _load_mask(mask_dir / f"view_{view_index:04d}_{key}.png", (h, w))
            for key in MASK_KEYS
            if (mask_dir / f"view_{view_index:04d}_{key}.png").exists()
        }
        components = {
            comp: _load_rgb(render_dir / f"eval_{comp}_{view_index:04d}.png")
            for comp in RGB_COMPONENTS
            if (render_dir / f"eval_{comp}_{view_index:04d}.png").exists()
        }
        image_summary: Dict[str, object] = {
            "view_index": view_index,
            "height": h,
            "width": w,
            "regions": {},
        }

        for region, mask in masks.items():
            region_summary: Dict[str, object] = {
                "coverage": float(mask.mean()),
                "components": {},
                "l1_vs_gt": {},
            }
            for comp, image in components.items():
                region_summary["components"][comp] = _component_stats(image, mask)
                if comp != "gt":
                    l1 = _masked_rgb_l1(image, gt, mask)
                    region_summary["l1_vs_gt"][comp] = l1
                    aggregate_accumulator.setdefault(f"{region}.l1_vs_gt.{comp}", []).append(l1)
            for comp in ("accumulation", "rgb_tail", "rgb_medium_total", "J", "J_gaussian_raw"):
                if comp in components:
                    luma_mean = region_summary["components"][comp]["luma"]["mean"]  # type: ignore[index]
                    aggregate_accumulator.setdefault(f"{region}.{comp}.luma_mean", []).append(float(luma_mean))
            if "J" in components:
                blue = region_summary["components"]["J"]["blue_dominance_fraction"]  # type: ignore[index]
                aggregate_accumulator.setdefault(f"{region}.J.blue_dominance_fraction", []).append(float(blue))
            image_summary["regions"][region] = region_summary
        images.append(image_summary)

    aggregate = {
        key: _stats(np.asarray(values, dtype=np.float32))
        for key, values in sorted(aggregate_accumulator.items())
    }
    return {
        "render_dir": str(render_dir),
        "mask_dir": str(mask_dir),
        "view_count": len(images),
        "aggregate": aggregate,
        "images": images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    result = diagnose(args.render_dir, args.mask_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
