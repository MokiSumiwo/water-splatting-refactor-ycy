#!/usr/bin/env python3
"""Create the corrected three-configuration RGB ablation summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")


def _load_baseline(root: Path) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, float]] = {}
    for scene in SCENES:
        path = root / scene / "final_metrics.csv"
        rows = list(csv.DictReader(path.open(encoding="utf8")))
        row = next(item for item in rows if item["split"] == "eval")
        values[scene] = {metric: float(row[metric]) for metric in METRICS}
    return values


def _load_evaluator(root: Path) -> Dict[str, Dict[str, float]]:
    values: Dict[str, Dict[str, float]] = {}
    for scene in SCENES:
        payload = json.loads((root / f"{scene}.json").read_text(encoding="utf8"))
        aggregate = payload["aggregate"]
        if int(aggregate["num_valid_pairs"]) != int(aggregate["num_prediction_images"]):
            raise RuntimeError(f"incomplete RGB evaluation for {scene}: {aggregate}")
        values[scene] = {metric: float(aggregate[metric]) for metric in METRICS}
    return values


def _state_rows(
    configuration: str,
    values: Mapping[str, Mapping[str, float]],
    metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in (*SCENES, "MEAN"):
        row: Dict[str, Any] = {"configuration": configuration, "scene": scene}
        for key, value in metadata.items():
            row[key] = value
        for metric in METRICS:
            row[metric] = (
                _mean(values[item][metric] for item in SCENES)
                if scene == "MEAN"
                else values[scene][metric]
            )
        rows.append(row)
    return rows


def _delta_rows(
    name: str,
    baseline_name: str,
    baseline: Mapping[str, Mapping[str, float]],
    treatment_name: str,
    treatment: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in (*SCENES, "MEAN"):
        row: Dict[str, Any] = {
            "comparison": name,
            "baseline": baseline_name,
            "treatment": treatment_name,
            "scene": scene,
        }
        for metric in METRICS:
            left = _mean(baseline[item][metric] for item in SCENES) if scene == "MEAN" else baseline[scene][metric]
            right = _mean(treatment[item][metric] for item in SCENES) if scene == "MEAN" else treatment[scene][metric]
            row[f"baseline_{metric}"] = left
            row[f"treatment_{metric}"] = right
            row[f"delta_{metric}"] = right - left
        rows.append(row)
    return rows


def _markdown(state_rows: List[Mapping[str, Any]], delta_rows: List[Mapping[str, Any]]) -> str:
    lines = [
        "# Corrected RGB Ablation Summary",
        "",
        "Protocol: held-out underwater RGB reconstruction; exact filename pairing; no resize/crop/color conversion; project `metrics.py` loader; project PSNR/SSIM; local LPIPS-VGG v0.1; scene-level arithmetic mean.",
        "",
        "The original baseline was not retrained. Its values are archival values from `outputs/original_watersplatting_baseline_four_scene_20260905/*/final_metrics.csv`. The two treatment rows use the current evaluator and one final checkpoint per scene.",
        "",
        "## Metrics",
        "",
        "| Configuration | Scene | PSNR | SSIM | LPIPS | MSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in state_rows:
        lines.append(
            f"| {row['configuration']} | {row['scene']} | {row['PSNR']:.6f} | {row['SSIM']:.6f} | {row['LPIPS']:.6f} | {row['MSE']:.9f} |"
        )
    lines += [
        "",
        "## Deltas",
        "",
        "Positive PSNR/SSIM is favorable; negative LPIPS/MSE is favorable.",
        "",
        "| Comparison | Scene | dPSNR | dSSIM | dLPIPS | dMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in delta_rows:
        lines.append(
            f"| {row['comparison']} | {row['scene']} | {row['delta_PSNR']:+.6f} | {row['delta_SSIM']:+.6f} | {row['delta_LPIPS']:+.6f} | {row['delta_MSE']:+.9f} |"
        )
    lines += [
        "",
        "The M1+OCMC render source is `renders/ablation_m1_ocmc_recovered_20260907_v2`; the CICA render source is the existing Auxiliary-OFF checkpoint render set, rechecked under the same evaluator.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = (args.output_dir or repo / "outputs" / "ablation_rgb_metrics_20260907" / "final_corrected").resolve()

    baseline = _load_baseline(repo / "outputs" / "original_watersplatting_baseline_four_scene_20260905")
    m1 = _load_evaluator(repo / "outputs" / "ablation_rgb_metrics_20260907" / "M1_OCMC_recovered_v2")
    cica = _load_evaluator(repo / "outputs" / "ablation_rgb_metrics_20260907" / "M1_OCMC_CICA_recheck")

    states = []
    states += _state_rows("Original WaterSplatting baseline", baseline, {
        "seed": 42,
        "absolute_step": 14999,
        "view_count": 13 if False else "13 total",
        "data_provenance": "archival final_metrics.csv; baseline not retrained",
    })
    states += _state_rows("M1 + OCMC", m1, {
        "seed": 42,
        "absolute_step": 14999,
        "view_count": "13 total",
        "checkpoint": "outputs/legacy_no_bounded_three_improvements_20260905/<scene>/checkpoints/C0/step-000014999.ckpt",
        "render_provenance": "analysis-only render; legacy C0; OCMC ON; RAOC OFF",
    })
    states += _state_rows("M1 + OCMC + CICA (Auxiliary OFF)", cica, {
        "seed": 42,
        "absolute_step": 14999,
        "view_count": "13 total",
        "checkpoint": "outputs/legacy_ocmc_cica_no_auxiliary_20260905/<scene>/checkpoints/step-000014999.ckpt",
        "render_provenance": "existing Auxiliary-OFF CICA render; OCMC ON; RAOC OFF",
    })

    deltas = []
    deltas += _delta_rows("M1 + OCMC minus original baseline", "Original WaterSplatting baseline", baseline, "M1 + OCMC", m1)
    deltas += _delta_rows("M1 + OCMC + CICA minus M1 + OCMC", "M1 + OCMC", m1, "M1 + OCMC + CICA (Auxiliary OFF)", cica)

    _write_csv(out / "state_metrics.csv", states)
    _write_csv(out / "deltas.csv", deltas)
    (out / "summary.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(_markdown(states, deltas), encoding="utf8")
    _write_json(out / "summary.json", {
        "experiment": "CORRECTED_RGB_ABLATION_SUMMARY_20260907",
        "baseline_retrained": False,
        "metric_protocol": {
            "loader": "metrics.py::_load_rgb_tensor",
            "psnr": "utils.image_utils.psnr",
            "ssim": "utils.loss_utils.ssim",
            "lpips": "lpipsPyTorch.modules.lpips.LPIPS VGG v0.1",
            "pairing": "exact filename",
            "spatial_processing": "none",
            "aggregation": "scene-level arithmetic mean across four scene aggregates",
        },
        "states": states,
        "deltas": deltas,
    })
    print(json.dumps({"output_dir": str(out), "state_rows": len(states), "delta_rows": len(deltas)}, sort_keys=True))


if __name__ == "__main__":
    main()
