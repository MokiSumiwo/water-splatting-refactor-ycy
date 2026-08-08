#!/usr/bin/env python
"""Summarize SeaFree-factor dewatering diagnostics and gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


CHANNELS = ("r", "g", "b")


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _metric_mean(summary: Mapping[str, Any], key: str) -> float:
    rows = summary.get("per_view", [])
    values = [float(row.get("metrics", {}).get(key, 0.0)) for row in rows]
    return float(sum(values) / len(values)) if values else 0.0


def _channel_stat(summary: Mapping[str, Any], group: str, stat: str) -> float:
    values = []
    item = summary.get("aggregate", {}).get(group, {})
    for channel in CHANNELS:
        if channel in item and stat in item[channel]:
            values.append(float(item[channel][stat]))
    return float(sum(values) / len(values)) if values else 0.0


def _channel_threshold(summary: Mapping[str, Any], group: str, key: str, fallback_key: Optional[str] = None) -> float:
    values = []
    item = summary.get("aggregate", {}).get(group, {})
    for channel in CHANNELS:
        if channel not in item:
            continue
        if key in item[channel]:
            values.append(float(item[channel][key]))
        elif fallback_key is not None and fallback_key in item[channel]:
            values.append(float(item[channel][fallback_key]))
    return float(sum(values) / len(values)) if values else 0.0


def _run_row(run: str, path: Path, summary: Mapping[str, Any]) -> Dict[str, Any]:
    background = summary.get("background_supervision", {})
    return {
        "run": run,
        "summary_path": str(path),
        "step": int(summary.get("requested_step", summary.get("loaded_step", 0))),
        "loaded_step": int(summary.get("loaded_step", 0)),
        "gamma_D": float(summary.get("direct_optical_depth_scale", 1.0)),
        "PSNR": _metric_mean(summary, "psnr"),
        "SSIM": _metric_mean(summary, "ssim"),
        "LPIPS": _metric_mean(summary, "lpips"),
        "beta_raw": _channel_stat(summary, "beta_D_raw", "mean"),
        "beta_eff": _channel_stat(summary, "beta_D_effective", "mean"),
        "tau_p90": _channel_stat(summary, "tau_D_effective", "p90"),
        "tau_p99": _channel_stat(summary, "tau_D_effective", "p99"),
        "T_mean": _channel_stat(summary, "T_D_effective", "mean"),
        "P(T<0.1)": _channel_threshold(summary, "T_D_effective_thresholds", "P(T<0.1)"),
        "P(T<0.05)": _channel_threshold(summary, "T_D_effective_thresholds", "P(T<0.05)"),
        "P(J>1)": _channel_threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>1.0)"),
        "P(J>1.5)": _channel_threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>1.5)"),
        "P(J>2)": _channel_threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>2.0)"),
        "J_p95": _channel_stat(summary, "clear_object_fullsh_raw", "p95"),
        "J_p99": _channel_stat(summary, "clear_object_fullsh_raw", "p99"),
        "P(c>1)": _channel_threshold(summary, "gaussian_view_rgb_thresholds", "P(c>1.0)", "P(gt1)"),
        "P(c>1.5)": _channel_threshold(summary, "gaussian_view_rgb_thresholds", "P(c>1.5)", "P(gt1.5)"),
        "P(c>2)": _channel_threshold(summary, "gaussian_view_rgb_thresholds", "P(c>2.0)", "P(gt2)"),
        "c_p95": _channel_stat(summary, "gaussian_view_rgb", "p95"),
        "c_p99": _channel_stat(summary, "gaussian_view_rgb", "p99"),
        "Gaussian count": int(summary.get("model_state", {}).get("gaussian_count", 0)),
        "background_available": bool(background.get("available", False)),
        "background_medium_l1": float(background.get("background_medium_l1", 0.0)),
        "weighted_background_medium_l1": float(background.get("weighted_background_medium_l1", 0.0)),
        "background_integrated_medium_total_l1": float(
            background.get("background_integrated_medium_total_l1", 0.0)
        ),
        "background_integrated_medium_finite_l1": float(
            background.get("background_integrated_medium_finite_l1", 0.0)
        ),
    }


def _rel_reduction(base: float, cand: float) -> float:
    if abs(base) <= 1e-12:
        return 0.0
    return (base - cand) / base


def _deltas(base: Mapping[str, Any], cand: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "Delta PSNR": float(cand["PSNR"] - base["PSNR"]),
        "Delta SSIM": float(cand["SSIM"] - base["SSIM"]),
        "Delta LPIPS": float(cand["LPIPS"] - base["LPIPS"]),
        "tau_p90_relative_reduction": _rel_reduction(float(base["tau_p90"]), float(cand["tau_p90"])),
        "P(J>1)_relative_reduction": _rel_reduction(float(base["P(J>1)"]), float(cand["P(J>1)"])),
        "J_p99_relative_reduction": _rel_reduction(float(base["J_p99"]), float(cand["J_p99"])),
    }


def _gate(stage: str, base: Mapping[str, Any], cand: Mapping[str, Any]) -> Dict[str, Any]:
    d = _deltas(base, cand)
    if stage == "no_refine":
        rgb = d["Delta PSNR"] >= -0.15 and d["Delta SSIM"] >= -0.0015 and d["Delta LPIPS"] <= 0.003
        tau = d["tau_p90_relative_reduction"] >= 0.15
        j = d["P(J>1)_relative_reduction"] >= 0.20
        return {
            "stage": stage,
            "baseline": base["run"],
            "candidate": cand["run"],
            "deltas": d,
            "RGB_safety": rgb,
            "tau_gate": tau,
            "J_gate": j,
            "D010_RECALIBRATION_CAUSAL_SIGNAL": bool(rgb and tau and j),
            "STRUCTURE_COUPLED": bool(not (rgb and tau and j)),
        }
    if stage == "intrinsic_bound":
        rgb = d["Delta PSNR"] >= -0.10 and d["Delta SSIM"] >= -0.0010 and d["Delta LPIPS"] <= 0.002
        tau_ok = float(cand["tau_p90"]) <= float(base["tau_p90"]) * 1.05
        j1 = d["P(J>1)_relative_reduction"] >= 0.30
        jp99 = d["J_p99_relative_reduction"] >= 0.15
        strong = d["P(J>1)_relative_reduction"] >= 0.50 and d["Delta PSNR"] >= -0.05
        return {
            "stage": stage,
            "baseline": base["run"],
            "candidate": cand["run"],
            "deltas": d,
            "RGB_safety": rgb,
            "tau_not_worse_gt5pct": tau_ok,
            "J_P_gt1_gate": j1,
            "J_p99_gate": jp99,
            "BOUND_PASS": bool(rgb and tau_ok and j1 and jp99),
            "STRONG_BOUND_CANDIDATE": bool(rgb and tau_ok and j1 and jp99 and strong),
        }
    if stage == "foreground":
        rgb = d["Delta PSNR"] >= -0.10 and d["Delta SSIM"] >= -0.0010 and d["Delta LPIPS"] <= 0.002
        tau_ok = float(cand["tau_p90"]) <= float(base["tau_p90"]) * 1.10
        j_ok = d["P(J>1)_relative_reduction"] >= 0.15 or d["J_p99_relative_reduction"] >= 0.10
        return {
            "stage": stage,
            "baseline": base["run"],
            "candidate": cand["run"],
            "deltas": d,
            "RGB_safety": rgb,
            "tau_not_worse_gt10pct": tau_ok,
            "J_gate": j_ok,
            "FAW_PASS": bool(rgb and tau_ok and j_ok),
        }
    return {"stage": stage, "baseline": base["run"], "candidate": cand["run"], "deltas": d}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"--run must be NAME=PATH, got {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        rows.append(_run_row(name, path, _load(path)))

    by_name = {row["run"]: row for row in rows}
    gates = []
    for item in args.compare:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"--compare must be STAGE:BASE:CANDIDATE, got {item}")
        stage, base_name, cand_name = parts
        if base_name in by_name and cand_name in by_name:
            gates.append(_gate(stage, by_name[base_name], by_name[cand_name]))

    summary = {"summary": "dewater_seafree_factor_summary", "runs": rows, "gates": gates}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf8")

    if rows:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with args.output_csv.open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="Run summary as NAME=path/to/summary.json")
    parser.add_argument("--compare", action="append", default=[], help="Gate as STAGE:BASE:CANDIDATE")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/dewater_seafree_factor_20260808/final_candidate_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/dewater_seafree_factor_20260808/final_candidate_summary.csv"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
