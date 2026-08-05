#!/usr/bin/env python
"""Summarize GMVC persistence RGB and fixed-bank diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


RGB_KEYS = ["psnr", "ssim", "lpips"]
FIXED_KEYS = [
    "transfer_l1",
    "object_j_variance",
    "closure_signal_floor_l1",
    "object_target_l1",
    "dc_cross_view_variance",
    "dc_recomposition_l1",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _metric_delta(value: float, base: float) -> Dict[str, float]:
    pct = 0.0 if abs(base) < 1e-12 else 100.0 * (value - base) / base
    return {"abs": value - base, "pct": pct}


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def summarize(root: Path, variants: Iterable[str], steps: Iterable[int]) -> Dict[str, Any]:
    variants = list(variants)
    steps = [int(step) for step in steps]
    summary: Dict[str, Any] = {
        "diagnostic": "gmvc_curasao_profile_persistence_3k_summary",
        "root": str(root),
        "variants": variants,
        "steps": steps,
        "rgb": {},
        "fixed": {},
        "delta_vs_a0": {},
    }

    for step in steps:
        step_key = str(step)
        summary["rgb"][step_key] = {}
        summary["fixed"][step_key] = {}
        for variant in variants:
            slug = variant.lower()
            rgb_path = root / slug / f"step{step}" / "rgb" / "rgb_metrics.json"
            rgb_data = _read_json(rgb_path)
            rgb_metrics = rgb_data["results"]
            summary["rgb"][step_key][variant] = {key: float(rgb_metrics.get(key, 0.0)) for key in RGB_KEYS}

            summary["fixed"][step_key][variant] = {}
            for bank_name in ("evalf", "evalg"):
                fixed_path = root / slug / f"step{step}" / bank_name / "gmvc_fixed_bank_metrics.json"
                fixed_data = _read_json(fixed_path)
                fixed_metrics = fixed_data["metrics"]["heldout"]
                summary["fixed"][step_key][variant][bank_name.upper()] = {
                    key: float(fixed_metrics.get(key, 0.0)) for key in FIXED_KEYS
                }

        summary["delta_vs_a0"][step_key] = {}
        base_rgb = summary["rgb"][step_key]["A0"]
        base_fixed = summary["fixed"][step_key]["A0"]
        for variant in variants:
            if variant == "A0":
                continue
            summary["delta_vs_a0"][step_key][variant] = {
                "rgb": {key: _metric_delta(summary["rgb"][step_key][variant][key], base_rgb[key]) for key in RGB_KEYS},
                "fixed": {},
            }
            for bank_name in ("EVALF", "EVALG"):
                summary["delta_vs_a0"][step_key][variant]["fixed"][bank_name] = {
                    key: _metric_delta(
                        summary["fixed"][step_key][variant][bank_name][key],
                        base_fixed[bank_name][key],
                    )
                    for key in FIXED_KEYS
                }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variants", default="A0,P40,P35,P30")
    parser.add_argument("--steps", default="11000,12000,13000")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    variants = _parse_list(args.variants)
    steps = [int(item) for item in _parse_list(args.steps)]
    summary = summarize(args.root, variants, steps)
    output = args.output or (args.root / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({"output": str(output), "variants": variants, "steps": steps}, indent=2))


if __name__ == "__main__":
    main()
