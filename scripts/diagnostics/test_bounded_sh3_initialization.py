#!/usr/bin/env python
"""Audit color-equivalent initialization for bounded SH3 intrinsic colors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from water_splatting.water_splatting import RGB2SH, RGB2SHLogits, SH2RGB, SHLogits2RGB


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float().reshape(-1), q).item())


def run_audit(seed: int, count: int, eps: float, output: Path | None) -> dict[str, float | bool | int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rgb = torch.rand(count, 3, generator=generator)
    endpoints = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0],
            [1.0, 0.5, 0.0],
        ],
        dtype=rgb.dtype,
    )
    rgb = torch.cat([rgb, endpoints], dim=0)

    legacy_rgb = SH2RGB(RGB2SH(rgb))
    bounded_rgb = SHLogits2RGB(RGB2SHLogits(rgb, eps=eps))
    abs_err = (legacy_rgb - bounded_rgb).abs()

    result: dict[str, float | bool | int] = {
        "seed": seed,
        "count": int(rgb.shape[0]),
        "eps": float(eps),
        "mean_abs_error": float(abs_err.mean().item()),
        "p95_abs_error": _quantile(abs_err, 0.95),
        "max_abs_error": float(abs_err.max().item()),
        "bounded_rgb_min": float(bounded_rgb.min().item()),
        "bounded_rgb_max": float(bounded_rgb.max().item()),
        "bounded_rgb_all_finite": bool(torch.isfinite(bounded_rgb).all().item()),
        "bounded_rgb_strictly_inside_0_1": bool(((bounded_rgb > 0.0) & (bounded_rgb < 1.0)).all().item()),
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_audit(seed=args.seed, count=args.count, eps=args.eps, output=args.output)


if __name__ == "__main__":
    main()
