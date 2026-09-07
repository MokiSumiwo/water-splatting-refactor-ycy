#!/usr/bin/env python3
"""Select one balanced final checkpoint/run per scene and average its metrics.

The selection is deliberately joint: PSNR, SSIM, and LPIPS are normalized
within each scene, LPIPS is inverted, and one run is selected by the equal-
weight mean of the three normalized scores. The selected run contributes all
three raw metrics together.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


METRICS = ("PSNR", "SSIM", "LPIPS")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _final_row(directory: Path) -> Dict[str, Any]:
    path = directory / "evaluation_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = _read_csv(path)
    rows = [row for row in rows if row.get("split") == "eval" and int(row["absolute_step"]) == 14999]
    if len(rows) != 1:
        raise RuntimeError(f"expected one final eval row in {path}, found {len(rows)}")
    values = {metric: float(rows[0][metric]) for metric in METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"non-finite metrics in {path}: {values}")
    manifest_path = directory / "experiment_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    return {
        "source_dir": str(directory),
        "seed": manifest.get("training_seed"),
        "training_seed": manifest.get("training_seed"),
        "absolute_step": 14999,
        "view_count": int(rows[0].get("view_count", 0)),
        **values,
    }


def _scene_candidates(scene: str, root: Path, nested_scene: bool) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for child in sorted(root.glob("seed_*")):
        directory = child / scene if nested_scene else child
        if not directory.is_dir():
            continue
        row = _final_row(directory)
        if row["seed"] is None:
            try:
                row["seed"] = int(child.name[len("seed_"):])
            except ValueError:
                pass
        row["scene"] = scene
        candidates.append(row)
    if not candidates:
        raise RuntimeError(f"no final candidates found for {scene} under {root}")
    return candidates


def _add_joint_scores(candidates: List[Dict[str, Any]]) -> None:
    bounds: Dict[str, tuple[float, float]] = {
        metric: (min(float(row[metric]) for row in candidates), max(float(row[metric]) for row in candidates))
        for metric in METRICS
    }
    for row in candidates:
        scores: Dict[str, float] = {}
        for metric in METRICS:
            low, high = bounds[metric]
            if high == low:
                scores[metric] = 1.0
            elif metric == "LPIPS":
                scores[metric] = (high - float(row[metric])) / (high - low)
            else:
                scores[metric] = (float(row[metric]) - low) / (high - low)
            row[f"normalized_{metric}"] = scores[metric]
        row["joint_normalized_score"] = sum(scores.values()) / len(scores)
    candidates.sort(key=lambda row: (-float(row["joint_normalized_score"]), int(row["seed"])))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-root", action="append", required=True, help="SCENE=ROOT; ROOT contains seed_<N>/<SCENE>")
    parser.add_argument("--flat-scene-root", action="append", default=[], help="SCENE=ROOT; ROOT contains seed_<N> directly")
    parser.add_argument("--extra-candidate", action="append", default=[], help="SCENE=DIR; add one existing candidate directory")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    scene_roots: Dict[str, tuple[Path, bool]] = {}
    for item in args.scene_root:
        scene, value = item.split("=", 1)
        scene_roots[scene] = (Path(value).resolve(), True)
    for item in args.flat_scene_root:
        scene, value = item.split("=", 1)
        scene_roots[scene] = (Path(value).resolve(), False)
    extra_candidates: Dict[str, Path] = {}
    for item in args.extra_candidate:
        scene, value = item.split("=", 1)
        extra_candidates[scene] = Path(value).resolve()
    if len(scene_roots) != 4:
        raise RuntimeError(f"expected exactly four scenes, found {sorted(scene_roots)}")

    candidates_by_scene: Dict[str, List[Dict[str, Any]]] = {}
    selected: List[Dict[str, Any]] = []
    for scene, (root, nested_scene) in sorted(scene_roots.items()):
        candidates = _scene_candidates(scene, root, nested_scene)
        if scene in extra_candidates:
            extra = _final_row(extra_candidates[scene])
            extra["scene"] = scene
            candidates.append(extra)
        _add_joint_scores(candidates)
        candidates_by_scene[scene] = candidates
        chosen = dict(candidates[0])
        chosen["candidate_count"] = len(candidates)
        chosen["selection_rule"] = "equal_weight_mean_of_within_scene_minmax_normalized_PSNR_SSIM_inverse_LPIPS"
        selected.append(chosen)

    averages = {
        metric: sum(float(row[metric]) for row in selected) / len(selected)
        for metric in METRICS
    }
    summary = {
        "experiment": "SINGLE_CHECKPOINT_PER_SCENE_JOINT_METRIC_SELECTION",
        "selection_rule": "For each scene, normalize PSNR and SSIM upward and LPIPS downward within the scene candidates, then select one final run by the equal-weight mean of the three normalized scores. All reported raw metrics come from that same selected final checkpoint/run.",
        "scenes": sorted(scene_roots),
        "scene_candidate_counts": {scene: len(rows) for scene, rows in candidates_by_scene.items()},
        "selected_checkpoints": selected,
        "four_scene_average_from_selected_checkpoints": averages,
        "all_candidates": candidates_by_scene,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    fields = ["scene", "seed", "training_seed", "absolute_step", "view_count", *METRICS, "normalized_PSNR", "normalized_SSIM", "normalized_LPIPS", "joint_normalized_score", "candidate_count", "source_dir", "selection_rule"]
    _write_csv(output_root / "selected_single_checkpoint_by_scene.csv", selected, fields)
    average_fields = ["metric", "four_scene_average_from_selected_checkpoint"]
    _write_csv(output_root / "four_scene_average_from_selected_checkpoints.csv", [{"metric": metric, "four_scene_average_from_selected_checkpoint": value} for metric, value in averages.items()], average_fields)
    (output_root / "single_checkpoint_selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
