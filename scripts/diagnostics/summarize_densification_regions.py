#!/usr/bin/env python
"""Summarize training-time densification region JSONL diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    return sum(values) / max(len(values), 1)


def _nested_float(row: Dict[str, Any], *keys: str) -> float:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean_nested(rows: List[Dict[str, Any]], *keys: str) -> float:
    values = [_nested_float(row, *keys) for row in rows]
    return sum(values) / max(len(values), 1)


def summarize(input_jsonl: Path) -> Dict[str, Any]:
    rows = []
    with input_jsonl.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {"input_jsonl": str(input_jsonl), "count": 0}

    latest = rows[-1]
    water_latest = latest.get("regions", {}).get("water", {})
    return {
        "input_jsonl": str(input_jsonl),
        "count": len(rows),
        "first_step": int(rows[0].get("step", 0)),
        "last_step": int(latest.get("step", 0)),
        "background_gradient_fraction_mean": _mean(rows, "background_gradient_fraction"),
        "background_split_candidate_fraction_mean": _mean(rows, "background_split_candidate_fraction"),
        "background_duplicate_candidate_fraction_mean": _mean(rows, "background_duplicate_candidate_fraction"),
        "background_weighted_split_candidate_fraction_mean": _mean(rows, "background_weighted_split_candidate_fraction"),
        "background_weighted_duplicate_candidate_fraction_mean": _mean(rows, "background_weighted_duplicate_candidate_fraction"),
        "background_opacity_grad_abs_fraction_mean": _mean(rows, "background_opacity_grad_abs_fraction"),
        "background_opacity_decrease_pressure_fraction_mean": _mean(
            rows, "background_opacity_decrease_pressure_fraction"
        ),
        "background_opacity_increase_pressure_fraction_mean": _mean(
            rows, "background_opacity_increase_pressure_fraction"
        ),
        "background_accumulation_grad_abs_fraction_mean": _mean(rows, "background_accumulation_grad_abs_fraction"),
        "background_accumulation_decrease_pressure_fraction_mean": _mean(
            rows, "background_accumulation_decrease_pressure_fraction"
        ),
        "background_accumulation_increase_pressure_fraction_mean": _mean(
            rows, "background_accumulation_increase_pressure_fraction"
        ),
        "water_sampled_accumulation_mean": _mean_nested(rows, "regions", "water", "sampled_accumulation", "mean"),
        "water_sampled_accumulation_p95_mean": _mean_nested(rows, "regions", "water", "sampled_accumulation", "p95"),
        "water_opacity_decrease_pressure_mean": _mean_nested(
            rows, "regions", "water", "opacity_logit_grad", "decrease_pressure_mean"
        ),
        "water_opacity_increase_pressure_mean": _mean_nested(
            rows, "regions", "water", "opacity_logit_grad", "increase_pressure_mean"
        ),
        "water_scale_grad_norm_p95_mean": _mean_nested(rows, "regions", "water", "scale_grad_norm", "p95"),
        "latest": {
            "opacity_accumulation_diagnostic_enabled": bool(
                latest.get("opacity_accumulation_diagnostic_enabled", False)
            ),
            "opacity_grad_available": bool(latest.get("opacity_grad_available", False)),
            "scale_grad_available": bool(latest.get("scale_grad_available", False)),
            "accumulation_grad_available": bool(latest.get("accumulation_grad_available", False)),
            "background_gradient_fraction": float(latest.get("background_gradient_fraction", 0.0)),
            "background_split_candidate_fraction": float(latest.get("background_split_candidate_fraction", 0.0)),
            "background_duplicate_candidate_fraction": float(latest.get("background_duplicate_candidate_fraction", 0.0)),
            "background_weighted_split_candidate_fraction": float(
                latest.get("background_weighted_split_candidate_fraction", 0.0)
            ),
            "background_weighted_duplicate_candidate_fraction": float(
                latest.get("background_weighted_duplicate_candidate_fraction", 0.0)
            ),
            "water_visible_count": int(water_latest.get("visible_count", 0)),
            "water_raw_grad_mean": float(water_latest.get("raw_grad", {}).get("mean", 0.0)),
            "water_raw_grad_p95": float(water_latest.get("raw_grad", {}).get("p95", 0.0)),
            "water_split_candidate_count": int(water_latest.get("split_candidate_count", 0)),
            "water_duplicate_candidate_count": int(water_latest.get("duplicate_candidate_count", 0)),
            "water_sampled_accumulation_mean": float(water_latest.get("sampled_accumulation", {}).get("mean", 0.0)),
            "water_sampled_accumulation_p95": float(water_latest.get("sampled_accumulation", {}).get("p95", 0.0)),
            "water_opacity_decrease_pressure_mean": float(
                water_latest.get("opacity_logit_grad", {}).get("decrease_pressure_mean", 0.0)
            ),
            "water_opacity_increase_pressure_mean": float(
                water_latest.get("opacity_logit_grad", {}).get("increase_pressure_mean", 0.0)
            ),
            "water_scale_grad_norm_p95": float(water_latest.get("scale_grad_norm", {}).get("p95", 0.0)),
            "water_accumulation_opacity_grad_corr": float(water_latest.get("accumulation_opacity_grad_corr", 0.0)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = summarize(args.input_jsonl)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
