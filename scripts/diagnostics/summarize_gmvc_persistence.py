#!/usr/bin/env python
"""Summarize GMVC persistence RGB and fixed-bank diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


RGB_KEYS = ["psnr", "ssim", "lpips"]
FIXED_KEYS = [
    "transfer_l1",
    "object_j_variance",
    "closure_signal_floor_l1",
    "consensus_j_reconstruction_l1",
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


def _read_variant_metrics(root: Path, variant: str, step: int) -> Dict[str, Any]:
    slug = variant.lower()
    rgb_path = root / slug / f"step{step}" / "rgb" / "rgb_metrics.json"
    rgb_data = _read_json(rgb_path)
    rgb_metrics = rgb_data["results"]
    fixed: Dict[str, Dict[str, float]] = {}
    for bank_name in ("evalf", "evalg"):
        fixed_path = root / slug / f"step{step}" / bank_name / "gmvc_fixed_bank_metrics.json"
        fixed_data = _read_json(fixed_path)
        fixed_metrics = fixed_data["metrics"]["heldout"]
        fixed[bank_name.upper()] = {key: float(fixed_metrics.get(key, 0.0)) for key in FIXED_KEYS}
    return {
        "rgb": {key: float(rgb_metrics.get(key, 0.0)) for key in RGB_KEYS},
        "fixed": fixed,
    }


def _delta_block(value: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rgb": {key: _metric_delta(value["rgb"][key], base["rgb"][key]) for key in RGB_KEYS},
        "fixed": {
            bank_name: {
                key: _metric_delta(value["fixed"][bank_name][key], base["fixed"][bank_name][key])
                for key in FIXED_KEYS
            }
            for bank_name in ("EVALF", "EVALG")
        },
    }


def summarize(
    root: Path,
    variants: Iterable[str],
    steps: Iterable[int],
    reference_variant: str = "",
    start_root: Optional[Path] = None,
    start_step: Optional[int] = None,
    start_variant: str = "",
) -> Dict[str, Any]:
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
    if reference_variant:
        summary["reference_variant"] = reference_variant
        summary["delta_vs_reference"] = {}
    start_metrics: Dict[str, Any] | None = None
    if start_root is not None and start_step is not None and start_variant:
        start_metrics = _read_variant_metrics(start_root, start_variant, int(start_step))
        summary["start_reference"] = {
            "root": str(start_root),
            "variant": start_variant,
            "step": int(start_step),
            "metrics": start_metrics,
        }
        summary["delta_vs_start"] = {}

    for step in steps:
        step_key = str(step)
        summary["rgb"][step_key] = {}
        summary["fixed"][step_key] = {}
        for variant in variants:
            metrics = _read_variant_metrics(root, variant, step)
            summary["rgb"][step_key][variant] = metrics["rgb"]
            summary["fixed"][step_key][variant] = metrics["fixed"]

        summary["delta_vs_a0"][step_key] = {}
        base_metrics = {"rgb": summary["rgb"][step_key]["A0"], "fixed": summary["fixed"][step_key]["A0"]}
        for variant in variants:
            if variant == "A0":
                continue
            value_metrics = {"rgb": summary["rgb"][step_key][variant], "fixed": summary["fixed"][step_key][variant]}
            summary["delta_vs_a0"][step_key][variant] = _delta_block(value_metrics, base_metrics)
        if reference_variant:
            summary["delta_vs_reference"][step_key] = {}
            reference_metrics = {
                "rgb": summary["rgb"][step_key][reference_variant],
                "fixed": summary["fixed"][step_key][reference_variant],
            }
            for variant in variants:
                if variant == reference_variant:
                    continue
                value_metrics = {
                    "rgb": summary["rgb"][step_key][variant],
                    "fixed": summary["fixed"][step_key][variant],
                }
                summary["delta_vs_reference"][step_key][variant] = _delta_block(value_metrics, reference_metrics)
        if start_metrics is not None:
            summary["delta_vs_start"][step_key] = {}
            for variant in variants:
                value_metrics = {
                    "rgb": summary["rgb"][step_key][variant],
                    "fixed": summary["fixed"][step_key][variant],
                }
                summary["delta_vs_start"][step_key][variant] = _delta_block(value_metrics, start_metrics)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variants", default="A0,P40,P35,P30")
    parser.add_argument("--steps", default="11000,12000,13000")
    parser.add_argument("--reference-variant", default="")
    parser.add_argument("--start-root", type=Path, default=None)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--start-variant", default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    variants = _parse_list(args.variants)
    steps = [int(item) for item in _parse_list(args.steps)]
    summary = summarize(
        args.root,
        variants,
        steps,
        reference_variant=args.reference_variant,
        start_root=args.start_root,
        start_step=args.start_step,
        start_variant=args.start_variant,
    )
    output = args.output or (args.root / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({"output": str(output), "variants": variants, "steps": steps}, indent=2))


if __name__ == "__main__":
    main()
