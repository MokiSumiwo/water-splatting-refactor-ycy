#!/usr/bin/env python3
"""Summarize the Curasao continuation-seed sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping


METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")
LOWER_IS_BETTER = {"PSNR": False, "SSIM": False, "LPIPS": True, "MSE": True}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _sample(seed: int, source_dir: Path, reused: bool) -> Dict[str, Any]:
    metrics_path = source_dir / "evaluation_metrics.csv"
    manifest_path = source_dir / "experiment_manifest.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    rows = _read_csv(metrics_path)
    final_rows = [row for row in rows if row.get("split") == "eval" and int(row["absolute_step"]) == 14999]
    if len(final_rows) != 1:
        raise RuntimeError(f"expected one final eval row in {metrics_path}, found {len(final_rows)}")
    values = {metric: float(final_rows[0][metric]) for metric in METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"non-finite final metrics in {metrics_path}: {values}")
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    return {
        "seed": int(seed),
        "training_seed": manifest.get("training_seed", seed),
        "source_dir": str(source_dir),
        "reused_existing_seed42": bool(reused),
        "absolute_step": 14999,
        "view_count": int(final_rows[0].get("view_count", 0)),
        **values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--existing-seed42", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--nested-scene", default=None, help="Read samples from root/seed_<seed>/<scene>")
    args = parser.parse_args()
    root = args.root.resolve()
    samples: List[Dict[str, Any]] = []
    for seed in args.seed:
        source_dir = args.existing_seed42.resolve() if seed == 42 else root / f"seed_{seed}"
        if seed != 42 and args.nested_scene is not None:
            source_dir = source_dir / args.nested_scene
        samples.append(_sample(seed, source_dir, seed == 42))
    samples.sort(key=lambda row: int(row["seed"]))
    if len({int(row["seed"]) for row in samples}) != len(samples):
        raise RuntimeError("duplicate seeds in requested summary")

    summary: Dict[str, Any] = {
        "experiment": "M1_OCMC_CICA_AUXILIARY_OFF_LEGACY_SH3_CURASAO_CONTINUATION_SEED_SWEEP",
        "interpretation": "All samples continue from the same frozen OCMC C0@3000 checkpoint; only continuation RNG seed varies. This is not end-to-end seed sensitivity.",
        "sample_count": len(samples),
        "seeds": [int(row["seed"]) for row in samples],
        "metrics": {},
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in samples]
        best_index = min(range(len(samples)), key=lambda index: values[index] if LOWER_IS_BETTER[metric] else -values[index])
        worst_index = max(range(len(samples)), key=lambda index: values[index] if LOWER_IS_BETTER[metric] else -values[index])
        mean = statistics.fmean(values)
        population_std = statistics.pstdev(values)
        summary["metrics"][metric] = {
            "mean": mean,
            "std_population": population_std,
            "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "range_max_minus_min": max(values) - min(values),
            "coefficient_of_variation": population_std / abs(mean) if mean != 0.0 else None,
            "best_seed": int(samples[best_index]["seed"]),
            "worst_seed": int(samples[worst_index]["seed"]),
            "lower_is_better": LOWER_IS_BETTER[metric],
        }

    root.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "training_seed", "reused_existing_seed42", "absolute_step", "view_count", "PSNR", "SSIM", "LPIPS", "MSE", "source_dir"]
    with (root / "multiseed_summary.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in samples)
    summary["samples"] = samples
    (root / "multiseed_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
