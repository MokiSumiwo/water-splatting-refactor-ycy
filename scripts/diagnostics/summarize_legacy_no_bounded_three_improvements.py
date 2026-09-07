#!/usr/bin/env python3
"""Summarize the no-Bounded-SH3 Legacy -> OCMC -> Auxiliary -> CICA chain."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")
HIGHER_IS_BETTER = {"PSNR", "SSIM"}
FINAL_STEP = 14999
BASELINE_ROOT = "outputs/legacy_baseline_four_scene_20260905"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")


def _number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {value!r}")
    return result


def _metric_row(path: Path, arm: str | None = None) -> Dict[str, float]:
    rows = _read_csv(path)
    candidates = [row for row in rows if row.get("split") == "eval" and int(row.get("absolute_step", FINAL_STEP)) == FINAL_STEP]
    if arm is not None:
        candidates = [row for row in candidates if row.get("arm") == arm]
    if not candidates:
        # The standalone Legacy source uses split/final columns rather than an
        # arm column. Its final_metrics.csv has one eval row.
        candidates = [row for row in rows if row.get("split") == "eval"]
        if arm is not None:
            candidates = [row for row in candidates if row.get("arm", arm) == arm]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one final eval row in {path} for arm={arm!r}, got {len(candidates)}")
    return {metric: _number(candidates[0][metric]) for metric in METRICS}


def _metrics_for_scene(root: Path, scene: str) -> Dict[str, Dict[str, float]]:
    baseline = _metric_row(Path(__file__).resolve().parents[2] / BASELINE_ROOT / scene / "final_metrics.csv")
    source = _metric_row(root / scene / "final_metrics.csv")
    auxiliary = _metric_row(root / scene / "A0" / "evaluation_metrics.csv", "C1")
    cica = _metric_row(root / scene / "A2" / "evaluation_metrics.csv", "A2")
    return {"Legacy": baseline, "OCMC": source, "A0": auxiliary, "A2": cica}


def _delta(after: Mapping[str, float], before: Mapping[str, float]) -> Dict[str, float]:
    return {metric: float(after[metric] - before[metric]) for metric in METRICS}


def _relative(after: Mapping[str, float], before: Mapping[str, float]) -> Dict[str, float]:
    return {metric: float((after[metric] - before[metric]) / max(abs(before[metric]), 1e-12)) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows: List[Dict[str, Any]] = []
    cumulative_rows: List[Dict[str, Any]] = []
    scene_payload: Dict[str, Any] = {}
    for scene in SCENES:
        values = _metrics_for_scene(root, scene)
        direct_pairs = (
            ("OCMC_minus_Legacy", "OCMC", "Legacy"),
            ("Auxiliary_minus_OCMC", "A0", "OCMC"),
            ("CICA_minus_Auxiliary", "A2", "A0"),
        )
        scene_payload[scene] = {key: dict(value) for key, value in values.items()}
        for label, after_name, before_name in direct_pairs:
            delta = _delta(values[after_name], values[before_name])
            relative = _relative(values[after_name], values[before_name])
            row = {"scene": scene, "comparison": label}
            for metric in METRICS:
                row[f"{metric}_before"] = values[before_name][metric]
                row[f"{metric}_after"] = values[after_name][metric]
                row[f"{metric}_delta"] = delta[metric]
                row[f"{metric}_relative_delta"] = relative[metric]
                row[f"{metric}_favorable"] = (delta[metric] >= 0.0) if metric in HIGHER_IS_BETTER else (delta[metric] <= 0.0)
            rows.append(row)
        complete = _delta(values["A2"], values["Legacy"])
        cumulative_rows.append({"scene": scene, "comparison": "A2_minus_Legacy_complete_chain", **complete})
    for label in ("OCMC_minus_Legacy", "Auxiliary_minus_OCMC", "CICA_minus_Auxiliary"):
        selected = [row for row in rows if row["comparison"] == label]
        mean_row: Dict[str, Any] = {"scene": "MEAN", "comparison": label}
        for metric in METRICS:
            mean_row[f"{metric}_before"] = sum(float(row[f"{metric}_before"]) for row in selected) / len(selected)
            mean_row[f"{metric}_after"] = sum(float(row[f"{metric}_after"]) for row in selected) / len(selected)
            mean_row[f"{metric}_delta"] = sum(float(row[f"{metric}_delta"]) for row in selected) / len(selected)
            mean_row[f"{metric}_relative_delta"] = (mean_row[f"{metric}_after"] - mean_row[f"{metric}_before"]) / max(abs(mean_row[f"{metric}_before"]), 1e-12)
            mean_row[f"{metric}_favorable"] = (mean_row[f"{metric}_delta"] >= 0.0) if metric in HIGHER_IS_BETTER else (mean_row[f"{metric}_delta"] <= 0.0)
        rows.append(mean_row)
    complete_mean = {metric: sum(row[metric] for row in cumulative_rows) / len(cumulative_rows) for metric in METRICS}
    cumulative_rows.append({"scene": "MEAN", "comparison": "A2_minus_Legacy_complete_chain", **complete_mean})
    _write_csv(root / "direct_effects.csv", rows)
    _write_csv(root / "cumulative_vs_legacy.csv", cumulative_rows)
    summary = {
        "experiment": "LEGACY_NO_BOUNDED_THREE_IMPROVEMENTS_FOUR_SCENE",
        "root": str(root),
        "baseline_root": str(Path(__file__).resolve().parents[2] / BASELINE_ROOT),
        "parameterization": "legacy",
        "protocol": "seed 42, 15000 updates, dir_xy_camera, tied B_inf, OCMC on, RAOC off, MDRR excluded",
        "metric_direction": {metric: ("higher" if metric in HIGHER_IS_BETTER else "lower") for metric in METRICS},
        "direct_comparisons": {
            "OCMC_minus_Legacy": "Legacy + OCMC - Legacy",
            "Auxiliary_minus_OCMC": "Legacy + OCMC + auxiliary - Legacy + OCMC",
            "CICA_minus_Auxiliary": "Legacy + OCMC + auxiliary + CICA - Legacy + OCMC + auxiliary",
        },
        "values": scene_payload,
        "direct_effect_rows": rows,
        "complete_chain_mean_delta": complete_mean,
    }
    _write_json(root / "summary.json", summary)
    print(json.dumps({"root": str(root), "direct_effects": str(root / "direct_effects.csv"), "complete_chain_mean_delta": complete_mean}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
