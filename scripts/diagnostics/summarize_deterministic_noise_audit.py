#!/usr/bin/env python
"""Summarize metric and Gaussian-count ranges across nominally equivalent runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _load_metrics(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf8"))
    results = payload.get("results", payload)
    return {
        "psnr": float(results["psnr"]),
        "ssim": float(results["ssim"]),
        "lpips": float(results["lpips"]),
    }


def _load_gaussian_count(path: Optional[Path]) -> Optional[int]:
    if path is None:
        return None
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("pipeline", ckpt)
    for key, value in state.items():
        if key.endswith("gauss_params.means") or key.endswith("_model.gauss_params.means"):
            return int(value.shape[0])
    raise KeyError(f"Could not find gauss_params.means in {path}")


def _range(values: List[float]) -> float:
    return float(max(values) - min(values)) if values else 0.0


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoints = list(args.checkpoints or [])
    rows: List[Dict[str, Any]] = []
    for idx, metric_json in enumerate(args.metric_jsons):
        checkpoint = checkpoints[idx] if idx < len(checkpoints) else None
        metrics = _load_metrics(metric_json)
        gaussian_count = _load_gaussian_count(checkpoint)
        rows.append(
            {
                "label": args.labels[idx] if args.labels and idx < len(args.labels) else metric_json.parent.name,
                "metric_json": str(metric_json),
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "gaussian_count": gaussian_count,
                **metrics,
            }
        )

    psnr_values = [float(row["psnr"]) for row in rows]
    ssim_values = [float(row["ssim"]) for row in rows]
    lpips_values = [float(row["lpips"]) for row in rows]
    gaussian_values = [int(row["gaussian_count"]) for row in rows if row["gaussian_count"] is not None]
    gaussian_range = max(gaussian_values) - min(gaussian_values) if gaussian_values else 0
    gaussian_range_fraction = float(gaussian_range / max(min(gaussian_values), 1)) if gaussian_values else 0.0
    thresholds = {
        "psnr_range_max": float(args.psnr_range_max),
        "ssim_range_max": float(args.ssim_range_max),
        "lpips_range_max": float(args.lpips_range_max),
        "gaussian_count_range_fraction_max": float(args.gaussian_count_range_fraction_max),
    }
    aggregate = {
        "psnr_range": _range(psnr_values),
        "ssim_range": _range(ssim_values),
        "lpips_range": _range(lpips_values),
        "gaussian_count_range": int(gaussian_range),
        "gaussian_count_range_fraction": gaussian_range_fraction,
    }
    aggregate["passes_noise_gate"] = bool(
        aggregate["psnr_range"] <= thresholds["psnr_range_max"]
        and aggregate["ssim_range"] <= thresholds["ssim_range_max"]
        and aggregate["lpips_range"] <= thresholds["lpips_range_max"]
        and aggregate["gaussian_count_range_fraction"] <= thresholds["gaussian_count_range_fraction_max"]
    )
    return {"runs": rows, "aggregate": aggregate, "thresholds": thresholds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-jsons", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--labels", type=str, nargs="*", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--psnr-range-max", type=float, default=0.02)
    parser.add_argument("--ssim-range-max", type=float, default=0.0003)
    parser.add_argument("--lpips-range-max", type=float, default=0.0010)
    parser.add_argument("--gaussian-count-range-fraction-max", type=float, default=0.0020)
    args = parser.parse_args()

    result = summarize(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
