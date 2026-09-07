#!/usr/bin/env python3
"""Build the retained WaterSplatting improvement metric summary.

The summary keeps two comparisons separate:

* direct_effects: each treatment minus its matched causal baseline;
* cumulative_vs_legacy: each complete configuration minus the same-sequence
  legacy SH3 run.

The latter is descriptive for nested treatments and must not be summed.
"""

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


def _final_eval(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for scene in SCENES:
        payload = json.loads((path / scene / "final_summary.json").read_text(encoding="utf8"))
        out[scene] = {metric: float(payload["final_eval"][metric]) for metric in METRICS}
    return out


def _ocmc_on(path: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for scene in SCENES:
        rows = list(csv.DictReader((path / scene / "eval_metrics.csv").open(encoding="utf8")))
        row = next(
            item
            for item in rows
            if item["branch"] == "C0"
            and item["absolute_step"] == "14999"
            and item["split"] == "eval"
        )
        out[scene] = {metric: float(row[metric]) for metric in METRICS}
    return out


def _auxiliary(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in csv.DictReader((path / "final_metrics.csv").open(encoding="utf8")):
        if row["split"] != "heldout":
            continue
        out[row["scene"]] = {
            "C0": {metric: float(row[f"{metric}_C0"]) for metric in METRICS},
            "C1": {metric: float(row[f"{metric}_C1"]) for metric in METRICS},
        }
    return out


def _cica(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in csv.DictReader((path / "final_metrics.csv").open(encoding="utf8")):
        if row["scene"] not in SCENES or row["metric"] not in METRICS:
            continue
        out.setdefault(row["scene"], {})[row["metric"]] = {
            "A0": float(row["A0"]),
            "A2": float(row["A2"]),
        }
    return {
        scene: {
            arm: {metric: out[scene][metric][arm] for metric in METRICS}
            for arm in ("A0", "A2")
        }
        for scene in SCENES
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _direct_rows(
    method: str,
    baseline_name: str,
    baseline: Mapping[str, Mapping[str, float]],
    treatment_name: str,
    treatment: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scene in (*SCENES, "MEAN"):
        for metric in METRICS:
            left = (
                _mean(baseline[item][metric] for item in SCENES)
                if scene == "MEAN"
                else baseline[scene][metric]
            )
            right = (
                _mean(treatment[item][metric] for item in SCENES)
                if scene == "MEAN"
                else treatment[scene][metric]
            )
            delta = right - left
            rows.append(
                {
                    "method": method,
                    "matched_baseline": baseline_name,
                    "treatment": treatment_name,
                    "scene": scene,
                    "metric": metric,
                    "baseline_value": left,
                    "treatment_value": right,
                    "delta_treatment_minus_baseline": delta,
                    "direction_of_improvement": "higher" if HIGHER_IS_BETTER[metric] else "lower",
                    "improves_metric": delta > 0 if HIGHER_IS_BETTER[metric] else delta < 0,
                }
            )
    return rows


def _cumulative_rows(
    legacy: Mapping[str, Mapping[str, float]],
    levels: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, values in levels.items():
        for scene in (*SCENES, "MEAN"):
            for metric in METRICS:
                left = (
                    _mean(legacy[item][metric] for item in SCENES)
                    if scene == "MEAN"
                    else legacy[scene][metric]
                )
                right = (
                    _mean(values[item][metric] for item in SCENES)
                    if scene == "MEAN"
                    else values[scene][metric]
                )
                delta = right - left
                rows.append(
                    {
                        "complete_configuration": method,
                        "reference": "same-sequence legacy SH3 + dir_xy_camera + OCMC_OFF + RAOC_OFF",
                        "scene": scene,
                        "metric": metric,
                        "legacy_value": left,
                        "configuration_value": right,
                        "delta_configuration_minus_legacy": delta,
                        "direction_of_improvement": "higher" if HIGHER_IS_BETTER[metric] else "lower",
                        "improves_metric": delta > 0 if HIGHER_IS_BETTER[metric] else delta < 0,
                    }
                )
    return rows


def _wide(rows: Sequence[Mapping[str, Any]], key: str, delta_key: str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for group in sorted({(row[key], row["scene"]) for row in rows}):
        method, scene = group
        selected = [row for row in rows if row[key] == method and row["scene"] == scene]
        item: Dict[str, Any] = {key: method, "scene": scene}
        for row in selected:
            item[f"{row['metric']}_delta"] = row[delta_key]
        result.append(item)
    return result


def _markdown(
    direct_wide: Sequence[Mapping[str, Any]],
    cumulative_wide: Sequence[Mapping[str, Any]],
    paths: Mapping[str, str],
) -> str:
    lines = [
        "# WaterSplatting Improvement Metric Summary",
        "",
        "Date: 2026-09-05",
        "",
        "## Scope",
        "",
        "This summary uses the original WaterSplatting legacy SH3 implementation as the reference. The legacy and bounded runs were newly evaluated from scratch with seed 42, 15,000 updates, the same four-scene camera sequence, `dir_xy_camera`, tied `B_inf`, OCMC off, and RAOC off. This provides the missing same-sequence legacy MSE values.",
        "",
        "PSNR and SSIM improve upward. LPIPS and MSE improve downward. A positive signed delta is therefore not automatically an improvement for every metric.",
        "",
        "## Direct Matched Effects",
        "",
        "Each row is treatment minus its own matched causal baseline. These are the valid module-level effects and must not be added across rows.",
        "",
        "| Method | Scene | dPSNR | dSSIM | dLPIPS | dMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in direct_wide:
        lines.append(
            f"| {row['method']} | {row['scene']} | {row['PSNR_delta']:+.6f} | {row['SSIM_delta']:+.6f} | {row['LPIPS_delta']:+.6f} | {row['MSE_delta']:+.8f} |"
        )
    lines += [
        "",
        "## Complete Configurations Versus Legacy",
        "",
        "These rows compare complete trained configurations to the same-sequence legacy run. They are descriptive endpoint differences, not additive causal effects, because the configurations are nested and some use different continuation baselines.",
        "",
        "| Configuration | Scene | dPSNR | dSSIM | dLPIPS | dMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in cumulative_wide:
        lines.append(
            f"| {row['complete_configuration']} | {row['scene']} | {row['PSNR_delta']:+.6f} | {row['SSIM_delta']:+.6f} | {row['LPIPS_delta']:+.6f} | {row['MSE_delta']:+.8f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Bounded SH3 reduces decomposition tails and improves average MSE in this matched rerun, but its average PSNR decreases and its RGB effect remains scene-dependent.",
        "- OCMC is the retained first core mechanism. Its matched RGB effect is mixed: mean PSNR increases by the listed amount, while mean SSIM and MSE do not improve under this exact comparison.",
        "- Auxiliary appearance regularization has a small mean held-out PSNR gain, but its mechanism audit was not supported; it remains baseline-only.",
        "- CICA is the retained second mechanism. Relative to the auxiliary baseline A0, it improves mean PSNR, LPIPS, and MSE, with a very small SSIM increase. It must not be reported as the sum of all previous deltas.",
        "- RAOC, MDRR, and MDRR+CICA are excluded from the retained-positive table because the project decisions classify them as not retained or interfering.",
        "",
        "## Data Provenance",
        "",
        f"- Legacy: `{paths['legacy']}`",
        f"- Bounded SH3 / OCMC-off: `{paths['bounded']}`",
        f"- OCMC-on: `{paths['ocmc']}`",
        f"- Auxiliary causal result: `{paths['auxiliary']}`",
        f"- CICA causal result: `{paths['cica']}`",
        "",
        "There is no paired real clear-ground-truth dataset in this project. The metrics above are underwater/held-out RGB reconstruction metrics; they do not quantify true clear/dewatered rendering quality.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/improvement_metric_summary_20260905"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()

    legacy = _final_eval(root / "outputs/legacy_baseline_four_scene_20260905")
    bounded = _final_eval(root / "outputs/ocmc_bounded_baseline_four_scene_20260905_v3")
    ocmc = _ocmc_on(root / "outputs/m1_raoc_causal_four_scene_20260827")
    auxiliary = _auxiliary(root / "outputs/identifiability_module_causal_iui3_20260902")
    cica = _cica(root / "outputs/direct_mdrr_cica_four_scene_20260903")

    auxiliary_c0 = {scene: auxiliary[scene]["C0"] for scene in SCENES}
    auxiliary_c1 = {scene: auxiliary[scene]["C1"] for scene in SCENES}
    cica_a0 = {scene: cica[scene]["A0"] for scene in SCENES}
    cica_a2 = {scene: cica[scene]["A2"] for scene in SCENES}
    aux_cica_max_diff = max(
        abs(auxiliary_c1[scene][metric] - cica_a0[scene][metric])
        for scene in SCENES
        for metric in METRICS
    )

    direct_rows: List[Dict[str, Any]] = []
    direct_rows += _direct_rows("Bounded SH3", "Legacy SH3", legacy, "bounded_sh3", bounded)
    direct_rows += _direct_rows("OCMC", "bounded SH3, OCMC off", bounded, "bounded SH3, OCMC on", ocmc)
    direct_rows += _direct_rows("Auxiliary Appearance Regularization", "OCMC C0", auxiliary_c0, "OCMC+C1 auxiliary", auxiliary_c1)
    direct_rows += _direct_rows("CICA", "A0 auxiliary baseline", cica_a0, "A2 CICA", cica_a2)

    levels = {
        "Bounded SH3": bounded,
        "OCMC": ocmc,
        "OCMC + Auxiliary": auxiliary_c1,
        "OCMC + Auxiliary + CICA": cica_a2,
    }
    cumulative_rows = _cumulative_rows(legacy, levels)
    direct_wide = _wide(direct_rows, "method", "delta_treatment_minus_baseline")
    cumulative_wide = _wide(cumulative_rows, "complete_configuration", "delta_configuration_minus_legacy")

    paths = {
        "legacy": "outputs/legacy_baseline_four_scene_20260905/<scene>/final_summary.json",
        "bounded": "outputs/ocmc_bounded_baseline_four_scene_20260905_v3/<scene>/final_summary.json",
        "ocmc": "outputs/m1_raoc_causal_four_scene_20260827/<scene>/eval_metrics.csv",
        "auxiliary": "outputs/identifiability_module_causal_iui3_20260902/final_metrics.csv",
        "cica": "outputs/direct_mdrr_cica_four_scene_20260903/final_metrics.csv",
    }
    payload = {
        "experiment": "IMPROVEMENT_METRIC_SUMMARY",
        "date": "2026-09-05",
        "scenes": list(SCENES),
        "metrics": list(METRICS),
        "metric_direction": {metric: "higher" if HIGHER_IS_BETTER[metric] else "lower" for metric in METRICS},
        "protocol": {
            "seed": 42,
            "max_updates": 15000,
            "legacy_and_bounded_same_sequence": True,
            "legacy_and_bounded_ocmc": False,
            "raoc": False,
            "paired_clear_ground_truth": False,
        },
        "auxiliary_c1_equals_cica_a0_max_abs_diff": aux_cica_max_diff,
        "direct_effects": direct_rows,
        "cumulative_vs_legacy": cumulative_rows,
        "source_paths": paths,
        "decisions": {
            "ocmc": "KEEP_CORE_INNOVATION",
            "bounded_sh3": "RETAIN_AS_DECOMPOSITION_STABILIZER_RGB_MIXED",
            "auxiliary_appearance_regularization": "KEEP_AS_BASELINE_ONLY",
            "cica": "KEEP_SECOND_INNOVATION",
            "raoc": "CLOSED_NOT_RETAINED",
            "mdrr": "DROP",
            "mdrr_cica": "DROP_INTERFERING",
        },
    }
    _write_csv(out / "direct_effects.csv", direct_rows)
    _write_csv(out / "cumulative_vs_legacy.csv", cumulative_rows)
    _write_csv(out / "direct_effects_wide.csv", direct_wide)
    _write_csv(out / "cumulative_vs_legacy_wide.csv", cumulative_wide)
    _write_json(out / "summary.json", payload)
    (root / "research_notes/IMPROVEMENT_METRIC_SUMMARY_2026-09-05.md").write_text(
        _markdown(direct_wide, cumulative_wide, paths), encoding="utf8"
    )
    print(json.dumps({"output_dir": str(out), "aux_cica_max_abs_diff": aux_cica_max_diff}, indent=2))


if __name__ == "__main__":
    main()
