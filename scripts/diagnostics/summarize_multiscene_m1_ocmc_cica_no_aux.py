#!/usr/bin/env python3
"""Summarize independent best metrics across multi-scene seed sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")
LOWER_IS_BETTER = {"PSNR": False, "SSIM": False, "LPIPS": True, "MSE": True}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _read_final_sample(scene: str, seed: int, source_dir: Path, reused: bool) -> Dict[str, Any]:
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
        raise RuntimeError(f"non-finite metrics in {metrics_path}: {values}")
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    return {
        "scene": scene,
        "seed": int(seed),
        "training_seed": manifest.get("training_seed", seed),
        "reused_existing_seed42": bool(reused),
        "absolute_step": 14999,
        "view_count": int(final_rows[0].get("view_count", 0)),
        "source_dir": str(source_dir),
        **values,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Root containing seed_<N>/<scene> results")
    parser.add_argument("--existing-seed42-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    existing_root = args.existing_seed42_root.resolve()
    scenes = list(dict.fromkeys(args.scene))
    seeds = list(dict.fromkeys(args.seed))
    if 42 not in seeds:
        raise RuntimeError("seed 42 must be included so the existing formal result can be reused")
    if len(scenes) == 0 or len(seeds) == 0:
        raise RuntimeError("scenes and seeds must be non-empty")

    samples: List[Dict[str, Any]] = []
    for scene in scenes:
        for seed in seeds:
            if seed == 42:
                source_dir = existing_root / scene
                reused = True
            else:
                source_dir = root / f"seed_{seed}" / scene
                reused = False
            sample = _read_final_sample(scene, seed, source_dir, reused)
            samples.append(sample)
            if int(sample["training_seed"]) != seed:
                raise RuntimeError(f"training seed mismatch for {scene}/seed_{seed}: {sample['training_seed']}")

    expected_count = len(scenes) * len(seeds)
    if len(samples) != expected_count:
        raise RuntimeError(f"expected {expected_count} samples, found {len(samples)}")

    best_rows: List[Dict[str, Any]] = []
    averages: Dict[str, Any] = {}
    for scene in scenes:
        scene_samples = [row for row in samples if row["scene"] == scene]
        if len(scene_samples) != len(seeds):
            raise RuntimeError(f"incomplete seed sweep for {scene}")
        best_row: Dict[str, Any] = {"scene": scene, "sample_count": len(scene_samples)}
        for metric in METRICS:
            if LOWER_IS_BETTER[metric]:
                chosen = min(scene_samples, key=lambda row: float(row[metric]))
            else:
                chosen = max(scene_samples, key=lambda row: float(row[metric]))
            best_row[metric] = float(chosen[metric])
            best_row[f"{metric}_best_seed"] = int(chosen["seed"])
        best_rows.append(best_row)

    for metric in METRICS:
        values = [float(row[metric]) for row in best_rows]
        averages[metric] = {
            "mean_of_scene_best": sum(values) / len(values),
            "scene_count": len(values),
            "values": values,
            "lower_is_better": LOWER_IS_BETTER[metric],
        }

    root.mkdir(parents=True, exist_ok=True)
    sample_fields = ["scene", "seed", "training_seed", "reused_existing_seed42", "absolute_step", "view_count", *METRICS, "source_dir"]
    best_fields = ["scene", "sample_count"]
    for metric in METRICS:
        best_fields.extend([metric, f"{metric}_best_seed"])
    _write_csv(root / "per_seed_metrics.csv", samples, sample_fields)
    _write_csv(root / "best_metrics_by_scene.csv", best_rows, best_fields)
    _write_csv(root / "final_average_of_scene_bests.csv", [{"metric": metric, **values} for metric, values in averages.items()], ["metric", "mean_of_scene_best", "scene_count", "lower_is_better"])

    summary = {
        "experiment": "M1_OCMC_CICA_AUXILIARY_OFF_LEGACY_SH3_MULTI_SCENE_CONTINUATION_SEED_SWEEP",
        "interpretation": "Each scene reuses its frozen OCMC C0@3000 checkpoint and varies only continuation RNG seed. For each metric, the best seed is selected independently within each scene; the final value is the arithmetic mean of those scene-wise best values.",
        "scenes": scenes,
        "seeds": seeds,
        "sample_count": len(samples),
        "samples": samples,
        "best_metrics_by_scene": best_rows,
        "final_average_of_scene_bests": averages,
    }
    (root / "multiscene_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
