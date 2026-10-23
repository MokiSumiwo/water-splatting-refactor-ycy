#!/usr/bin/env python3
"""Summarize the retained BRASH BND-ON screening artifacts.

The formal jobs use compact output, so this script intentionally reads only
the JSON/CSV artifacts that remain after checkpoint cleanup.  It never
re-runs training or assumes that a checkpoint or rendered image exists.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


VARIANTS = (
    "E0_bnd_fixed",
    "E1_brash_fixed_gate",
    "E2_brash_learned_gate",
    "E3_brash_reliability_prior",
)
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_metrics(path: Path) -> Dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no metric row in {path}")
    row = rows[-1]
    return {key: float(row[key]) for key in METRICS}


def _iter_results(root: Path, scenes: Iterable[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for scene in scenes:
        for variant in VARIANTS:
            directory = root / scene / variant
            summary = _read_json(directory / "final_summary.json")
            safety = _read_json(directory / "decomposition_safety.json")
            metrics = _read_metrics(directory / "evaluation_metrics.csv")
            result = {
                "scene": scene,
                "variant": variant,
                **metrics,
                "checkpoint_retained": bool(summary.get("checkpoint_retained", False)),
                "checkpoint_count": int(summary.get("checkpoint_count", 0)),
                "safety": bool(safety.get("all_P_J_gt_1_zero", False)),
                "brash_enabled": bool(summary.get("brash_enabled", False)),
                "gate_mode": summary.get("brash_gate_mode", ""),
                "prior_enabled": bool(summary.get("brash_gate_prior_enabled", False)),
                "prior_weight": float(summary.get("brash_gate_prior_weight", 0.0)),
                "reliability": summary.get("brash_reliability_stats", {"enabled": False}),
                "directory": str(directory),
            }
            results.append(result)
    return results


def _means(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {metric: mean(row[metric] for row in rows) for metric in METRICS}


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def _markdown(results: List[Dict[str, Any]], root: Path) -> str:
    baseline = {row["scene"]: row for row in results if row["variant"] == VARIANTS[0]}
    lines = [
        "# BRASH BND-ON Screening Summary",
        "",
        f"- Artifact root: `{root}`",
        "- Protocol: M1 + OCMC + CICA, Auxiliary OFF, BND ON, RAOC OFF, seed 42, step 14999.",
        "- Metrics are read from the repository-local evaluation CSVs.",
        "",
        "## Per-Scene Metrics",
        "",
        "| Scene | Variant | PSNR | SSIM | LPIPS | MSE | Delta PSNR | Delta SSIM | Delta LPIPS | Delta MSE | P(J > 1)=0 | Checkpoint retained |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in results:
        base = baseline[row["scene"]]
        deltas = {metric: row[metric] - base[metric] for metric in METRICS}
        lines.append(
            f"| {row['scene']} | {row['variant']} | {row['PSNR']:.6f} | {row['SSIM']:.6f} | "
            f"{row['LPIPS']:.6f} | {row['MSE']:.9f} | {deltas['PSNR']:+.6f} | "
            f"{deltas['SSIM']:+.6f} | {deltas['LPIPS']:+.6f} | {deltas['MSE']:+.9f} | "
            f"{'yes' if row['safety'] else 'no'} | {'yes' if row['checkpoint_retained'] else 'no'} |"
        )
    lines.extend(["", "## Two-Scene Means", "", "| Variant | PSNR | SSIM | LPIPS | MSE | Delta PSNR | Delta SSIM | Delta LPIPS | Delta MSE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    baseline_mean = _means([row for row in results if row["variant"] == VARIANTS[0]])
    for variant in VARIANTS:
        current = _means([row for row in results if row["variant"] == variant])
        delta = {metric: current[metric] - baseline_mean[metric] for metric in METRICS}
        lines.append(
            f"| {variant} | {current['PSNR']:.6f} | {current['SSIM']:.6f} | {current['LPIPS']:.6f} | {current['MSE']:.9f} | "
            f"{delta['PSNR']:+.6f} | {delta['SSIM']:+.6f} | {delta['LPIPS']:+.6f} | {delta['MSE']:+.9f} |"
        )
    lines.extend(["", "## Reliability and Retention", "", "| Scene | Variant | Confidence mean | Confidence p90 | High-confidence fraction | Support mean | View spread mean | Checkpoint retained |", "|---|---|---:|---:|---:|---:|---:|:---:|"])
    for row in results:
        reliability = row["reliability"]
        if not row["brash_enabled"] or not reliability.get("enabled", True):
            values = ("n/a",) * 5
        else:
            values = tuple(_format(reliability.get(key, "n/a")) for key in ("confidence_mean", "confidence_p90", "high_confidence_fraction", "support_mean", "view_spread_mean"))
        lines.append(f"| {row['scene']} | {row['variant']} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | {values[4]} | {'yes' if row['checkpoint_retained'] else 'no'} |")
    lines.extend(["", "## Interpretation", "", "- No candidate satisfies a clean all-metric non-regression rule against E0.", "- E1 has the strongest two-scene PSNR/SSIM mean, but it is not a strict all-metric win.", "- E3 improves Panama and two-scene SSIM/LPIPS, but lowers JapaneseGradens PSNR and worsens the two-scene MSE mean.", "- The E3 reliability prior is conservative: its high-confidence fraction is zero in both scenes and confidence is near zero for most Gaussians.", "- Checkpoint retention is reported explicitly; compact output removed the formal checkpoints, so this summary does not claim visual files exist.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/brash_bnd_on_screening_20260909"))
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    scenes = args.scenes or ["JapaneseGradens-RedSea", "Panama"]
    results = _iter_results(root, scenes)
    document = _markdown(results, root)
    if args.output:
        args.output.write_text(document + "\n", encoding="utf-8")
    else:
        print(document)


if __name__ == "__main__":
    main()
