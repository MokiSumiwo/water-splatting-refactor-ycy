#!/usr/bin/env python3
"""Summarize the corrected original-to-CICA four-state causal chain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")
HIGHER_IS_BETTER = {"PSNR": True, "SSIM": True, "LPIPS": False, "MSE": False}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _load_original(root: Path) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, float]] = {}
    for scene in SCENES:
        payload = json.loads((root / scene / "final_summary.json").read_text(encoding="utf8"))
        if payload.get("medium_context_mode") != "dir_only" or payload.get("ocmc_enabled") is not False:
            raise RuntimeError(f"not the original baseline: {scene}")
        completed_updates = payload.get("completed_updates", payload.get("training_runtime", {}).get("completed_updates"))
        if completed_updates != 15000:
            raise RuntimeError(f"incomplete original baseline: {scene}")
        values[scene] = {metric: float(payload["final_eval"][metric]) for metric in METRICS}
    return values


def _load_summary(root: Path) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, float]] = {}
    for scene in SCENES:
        payload = json.loads((root / scene / "final_summary.json").read_text(encoding="utf8"))
        values[scene] = {metric: float(payload["final_eval"][metric]) for metric in METRICS}
    return values


def _load_cica(root: Path) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, float]] = {}
    for scene in SCENES:
        rows = list(csv.DictReader((root / scene / "evaluation_metrics.csv").open(encoding="utf8")))
        row = next(item for item in rows if item["absolute_step"] == "14999" and item["split"] == "eval")
        values[scene] = {metric: float(row[metric]) for metric in METRICS}
    return values


def _load_cica_summary(root: Path) -> Dict[str, Dict[str, Any]]:
    """Read compact controller metadata for auditability without loading checkpoints."""

    values: Dict[str, Dict[str, Any]] = {}
    for scene in SCENES:
        payload = json.loads((root / scene / "final_summary.json").read_text(encoding="utf8"))
        if payload.get("auxiliary_appearance_enabled") is not False or payload.get("cica_enabled") is not True:
            raise RuntimeError(f"not the Auxiliary-OFF CICA arm: {scene}")
        values[scene] = {
            "completed_updates": int(payload["completed_updates"]),
            "auxiliary_appearance_enabled": bool(payload["auxiliary_appearance_enabled"]),
            "cica_enabled": bool(payload["cica_enabled"]),
            "ocmc_projector_unchanged": bool(payload["ocmc_projector_unchanged"]),
        }
    return values


def _wide_states(states: Mapping[str, Mapping[str, Mapping[str, float]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for state, values in states.items():
        for scene in (*SCENES, "MEAN"):
            row: Dict[str, Any] = {"state": state, "scene": scene}
            for metric in METRICS:
                row[metric] = _mean(values[item][metric] for item in SCENES) if scene == "MEAN" else values[scene][metric]
            rows.append(row)
    return rows


def _effects(
    baseline_name: str,
    baseline: Mapping[str, Mapping[str, float]],
    treatment_name: str,
    treatment: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in (*SCENES, "MEAN"):
        for metric in METRICS:
            left = _mean(baseline[item][metric] for item in SCENES) if scene == "MEAN" else baseline[scene][metric]
            right = _mean(treatment[item][metric] for item in SCENES) if scene == "MEAN" else treatment[scene][metric]
            delta = right - left
            rows.append(
                {
                    "effect": treatment_name,
                    "matched_baseline": baseline_name,
                    "scene": scene,
                    "metric": metric,
                    "baseline": left,
                    "treatment": right,
                    "delta": delta,
                    "improves_metric": delta > 0 if HIGHER_IS_BETTER[metric] else delta < 0,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (args.output_dir or repo / "outputs" / "original_m1_ocmc_cica_no_aux_comparison_20260905").resolve()

    original = _load_original(repo / "outputs" / "original_watersplatting_baseline_four_scene_20260905")
    m1 = _load_summary(repo / "outputs" / "legacy_baseline_four_scene_20260905")
    ocmc = _load_summary(repo / "outputs" / "legacy_no_bounded_three_improvements_20260905")
    cica = _load_cica(repo / "outputs" / "legacy_ocmc_cica_no_auxiliary_20260905")
    cica_audit = _load_cica_summary(repo / "outputs" / "legacy_ocmc_cica_no_auxiliary_20260905")
    states = {
        "Original dir_only baseline": original,
        "M1 dir_xy_camera baseline": m1,
        "M1 + OCMC": ocmc,
        "M1 + OCMC + CICA (Auxiliary OFF)": cica,
    }
    effects: List[Dict[str, Any]] = []
    effects += _effects("Original dir_only baseline", original, "M1 effect", m1)
    effects += _effects("M1 dir_xy_camera baseline", m1, "OCMC effect", ocmc)
    effects += _effects("M1 + OCMC", ocmc, "CICA effect (Auxiliary OFF)", cica)
    _write_json(
        output / "comparison.json",
        {
            "protocol": {
                "seed": 42,
                "updates": 15000,
                "scenes": list(SCENES),
                "original": "legacy SH3 + dir_only + OCMC OFF + Auxiliary OFF + CICA OFF + RAOC OFF + MDRR OFF",
                "m1": "legacy SH3 + dir_xy_camera + OCMC OFF + Auxiliary OFF + CICA OFF + RAOC OFF + MDRR OFF",
                "ocmc": "legacy SH3 + dir_xy_camera + OCMC ON + Auxiliary OFF + CICA OFF + RAOC OFF + MDRR OFF",
                "cica": "legacy SH3 + dir_xy_camera + OCMC ON + Auxiliary OFF + CICA ON + RAOC OFF + MDRR OFF",
            },
            "states": states,
            "effects": effects,
            "cica_audit": cica_audit,
        },
    )
    _write_csv(output / "state_metrics.csv", _wide_states(states))
    _write_csv(output / "causal_effects.csv", effects)
    print(json.dumps({"output_dir": str(output), "state_rows": len(_wide_states(states)), "effect_rows": len(effects)}, sort_keys=True))


if __name__ == "__main__":
    main()
