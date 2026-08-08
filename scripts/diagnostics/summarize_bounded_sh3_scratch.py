#!/usr/bin/env python
"""Summarize bounded-SH3 scratch diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CHANNELS = ("r", "g", "b")


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _load(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf8"))
    data["_summary_path"] = str(path)
    return data


def _parse_item(text: str) -> Tuple[str, int, Path]:
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"--summary must be RUN:STEP:PATH, got {text}")
    return parts[0], int(parts[1]), Path(parts[2])


def _metric(summary: Mapping[str, Any], key: str) -> float:
    return _mean(float(row.get("metrics", {}).get(key, 0.0)) for row in summary.get("per_view", []))


def _channel_stat(summary: Mapping[str, Any], group: str, stat: str) -> float:
    item = summary.get("aggregate", {}).get(group, {})
    return _mean(float(item.get(channel, {}).get(stat, 0.0)) for channel in CHANNELS if channel in item)


def _threshold(summary: Mapping[str, Any], group: str, *keys: str) -> float:
    item = summary.get("aggregate", {}).get(group, {})
    values = []
    for channel in CHANNELS:
        channel_item = item.get(channel, {})
        for key in keys:
            if key in channel_item:
                values.append(float(channel_item[key]))
                break
    return _mean(values)


def _row(run: str, step: int, summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "step": int(step),
        "run": run,
        "summary_path": summary.get("_summary_path", ""),
        "checkpoint": summary.get("checkpoint", ""),
        "loaded_step": int(summary.get("loaded_step", step)),
        "gamma_D": float(summary.get("direct_optical_depth_scale", 1.0)),
        "intrinsic_color_parameterization": summary.get("model_state", {}).get(
            "intrinsic_color_parameterization", "legacy"
        ),
        "PSNR": _metric(summary, "psnr"),
        "SSIM": _metric(summary, "ssim"),
        "LPIPS": _metric(summary, "lpips"),
        "beta_raw": _channel_stat(summary, "beta_D_raw", "mean"),
        "beta_eff": _channel_stat(summary, "beta_D_effective", "mean"),
        "tau_p50": _channel_stat(summary, "tau_D_effective", "p50"),
        "tau_p90": _channel_stat(summary, "tau_D_effective", "p90"),
        "tau_p99": _channel_stat(summary, "tau_D_effective", "p99"),
        "T_mean": _channel_stat(summary, "T_D_effective", "mean"),
        "P(T<0.3)": _threshold(summary, "T_D_effective_thresholds", "P(T<0.3)"),
        "P(T<0.2)": _threshold(summary, "T_D_effective_thresholds", "P(T<0.2)"),
        "P(T<0.1)": _threshold(summary, "T_D_effective_thresholds", "P(T<0.1)"),
        "P(T<0.05)": _threshold(summary, "T_D_effective_thresholds", "P(T<0.05)"),
        "J_p95": _channel_stat(summary, "clear_object_fullsh_raw", "p95"),
        "J_p99": _channel_stat(summary, "clear_object_fullsh_raw", "p99"),
        "P(J>0.95)": _threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>0.95)"),
        "P(J>0.99)": _threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>0.99)"),
        "P(J>1)": _threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>1.0)", "P(gt1)"),
        "c_p95": _channel_stat(summary, "gaussian_view_rgb", "p95"),
        "c_p99": _channel_stat(summary, "gaussian_view_rgb", "p99"),
        "P(c>0.95)": _threshold(summary, "gaussian_view_rgb_thresholds", "P(c>0.95)"),
        "P(c>0.99)": _threshold(summary, "gaussian_view_rgb_thresholds", "P(c>0.99)"),
        "P(c<0.01)": _threshold(summary, "gaussian_view_rgb_thresholds", "P(c<0.01)"),
        "SATURATION_MASS_001": _threshold(summary, "gaussian_view_rgb_thresholds", "SATURATION_MASS_001"),
        "logit_p99": _channel_stat(summary, "gaussian_view_logits", "p99"),
        "logit_max": _channel_stat(summary, "gaussian_view_logits", "max"),
        "P(|s|>5)": _threshold(summary, "gaussian_view_logits_thresholds", "P(|s|>5)"),
        "P(|s|>8)": _threshold(summary, "gaussian_view_logits_thresholds", "P(|s|>8)"),
        "sigmoid_derivative_mean": _channel_stat(summary, "gaussian_sigmoid_derivative", "mean"),
        "P(sigmoid_derivative<0.01)": _threshold(
            summary, "gaussian_sigmoid_derivative_thresholds", "P(sigmoid_derivative<0.01)"
        ),
        "beta_B": _channel_stat(summary, "medium_bs", "mean"),
        "medium_rgb_mean": _channel_stat(summary, "medium_rgb", "mean"),
        "B_inf_mean": _channel_stat(summary, "b_inf", "mean"),
        "backscatter_mean": _channel_stat(summary, "backscatter", "mean"),
        "Gaussian count": int(summary.get("model_state", {}).get("gaussian_count", 0)),
    }


def _rel_drop(base: float, cand: float) -> float:
    if abs(float(base)) <= 1e-12:
        return 0.0
    return 1.0 - float(cand) / float(base)


def _delta(base: Mapping[str, Any], cand: Mapping[str, Any], metric: str) -> float:
    return float(cand[metric]) - float(base[metric])


def _factor_effect(name: str, base: Mapping[str, Any], cand: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "comparison": name,
        "base": base["run"],
        "candidate": cand["run"],
        "Delta PSNR": _delta(base, cand, "PSNR"),
        "Delta SSIM": _delta(base, cand, "SSIM"),
        "Delta LPIPS": _delta(base, cand, "LPIPS"),
        "Delta tau p90": _delta(base, cand, "tau_p90"),
        "tau_p90_relative_drop": _rel_drop(float(base["tau_p90"]), float(cand["tau_p90"])),
        "Delta J p99": _delta(base, cand, "J_p99"),
        "J_p99_relative_drop": _rel_drop(float(base["J_p99"]), float(cand["J_p99"])),
        "Delta P(J>1)": _delta(base, cand, "P(J>1)"),
        "P(J>1)_relative_drop": _rel_drop(float(base["P(J>1)"]), float(cand["P(J>1)"])),
    }


def _rgb_safe(base: Mapping[str, Any], cand: Mapping[str, Any], psnr: float = -0.15) -> bool:
    return (
        _delta(base, cand, "PSNR") >= psnr
        and _delta(base, cand, "SSIM") >= -0.0015
        and _delta(base, cand, "LPIPS") <= 0.003
    )


def _classifications(final_rows: Mapping[str, Mapping[str, Any]], r_beta_15k: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    d100 = final_rows.get("D100")
    d010 = final_rows.get("D010")
    bnd = final_rows.get("BND")
    d010_bnd = final_rows.get("D010-BND")
    if d100 and bnd:
        bnd_main = _rgb_safe(d100, bnd) and (
            _rel_drop(float(d100["J_p99"]), float(bnd["J_p99"])) >= 0.15
            or _rel_drop(float(d100["P(J>1)"]), float(bnd["P(J>1)"])) >= 0.25
        )
        out["BOUND_MAIN_EFFECT"] = {
            "classification": bool(bnd_main),
            "numeric_basis": _factor_effect("BND - D100", d100, bnd),
        }
    if bnd and d010_bnd:
        r_beta = float(r_beta_15k.get("R_beta_BND_mean_rgb", 0.0)) if r_beta_15k else 0.0
        synergy = _rgb_safe(bnd, d010_bnd) and r_beta < 9.0 and float(d010_bnd["tau_p90"]) < float(bnd["tau_p90"])
        out["BOUND_SCALE_SYNERGY"] = {
            "classification": bool(synergy),
            "R_beta_BND_15k": r_beta,
            "numeric_basis": _factor_effect("D010-BND - BND", bnd, d010_bnd),
        }
        only = bool(out.get("BOUND_MAIN_EFFECT", {}).get("classification")) and not synergy
        out["BOUND_ONLY_SUFFICIENT"] = {
            "classification": bool(only),
            "numeric_basis": _factor_effect("D010-BND - BND", bnd, d010_bnd),
        }
        scale_comp = r_beta >= 9.0 and abs(_rel_drop(float(bnd["tau_p90"]), float(d010_bnd["tau_p90"]))) <= 0.10
        out["SCALE_STILL_FULLY_COMPENSATED"] = {
            "classification": bool(scale_comp),
            "R_beta_BND_15k": r_beta,
            "tau_p90_relative_difference_vs_BND": _rel_drop(float(bnd["tau_p90"]), float(d010_bnd["tau_p90"])),
        }
        boundary = (
            max(float(bnd["SATURATION_MASS_001"]), float(d010_bnd["SATURATION_MASS_001"])) > 0.05
            or max(float(bnd["P(|s|>5)"]), float(d010_bnd["P(|s|>5)"])) > 0.05
        )
        out["SIGMOID_BOUNDARY_ESCAPE"] = {
            "classification": bool(boundary),
            "BND_saturation_mass": float(bnd["SATURATION_MASS_001"]),
            "D010_BND_saturation_mass": float(d010_bnd["SATURATION_MASS_001"]),
            "BND_P_abs_logit_gt5": float(bnd["P(|s|>5)"]),
            "D010_BND_P_abs_logit_gt5": float(d010_bnd["P(|s|>5)"]),
        }
    if d100 and d010 and bnd and d010_bnd:
        failure = (float(bnd["PSNR"]) < float(d100["PSNR"]) - 0.20) and (
            float(d010_bnd["PSNR"]) < float(d010["PSNR"]) - 0.20
        )
        out["BOUNDED_PARAMETERIZATION_RGB_FAILURE"] = {
            "classification": bool(failure),
            "BND_delta_psnr_vs_D100": _delta(d100, bnd, "PSNR"),
            "D010_BND_delta_psnr_vs_D010": _delta(d010, d010_bnd, "PSNR"),
        }
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run(args: argparse.Namespace) -> Dict[str, Any]:
    summaries: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for item in args.summary:
        run_name, step, path = _parse_item(item)
        summaries[(run_name, step)] = _load(path)

    rows = [_row(run_name, step, summary) for (run_name, step), summary in sorted(summaries.items(), key=lambda x: (x[0][1], x[0][0]))]
    by_run_step = {(row["run"], int(row["step"])): row for row in rows}

    beta_rows: List[Dict[str, Any]] = []
    for step in sorted({int(row["step"]) for row in rows}):
        bnd = by_run_step.get(("BND", step))
        d010_bnd = by_run_step.get(("D010-BND", step))
        if not bnd or not d010_bnd:
            continue
        beta_rows.append(
            {
                "step": step,
                "R_beta_BND_mean_rgb": float(d010_bnd["beta_raw"]) / max(float(bnd["beta_raw"]), 1e-12),
                "expected_full_compensation": 10.0,
                "distance_to_full_compensation": 10.0 - float(d010_bnd["beta_raw"]) / max(float(bnd["beta_raw"]), 1e-12),
                "BND_beta_raw": bnd["beta_raw"],
                "D010_BND_beta_raw": d010_bnd["beta_raw"],
                "BND_beta_eff": bnd["beta_eff"],
                "D010_BND_beta_eff": d010_bnd["beta_eff"],
            }
        )

    final_names = {"D100": "D100", "D010": "D010", "BND": "BND", "D010-BND": "D010-BND"}
    final_rows = {alias: by_run_step[(run, args.final_step)] for alias, run in final_names.items() if (run, args.final_step) in by_run_step}
    factor_effects: List[Dict[str, Any]] = []
    if "D100" in final_rows and "BND" in final_rows:
        factor_effects.append(_factor_effect("Intrinsic-range main effect at gamma=1: BND - D100", final_rows["D100"], final_rows["BND"]))
    if "D010" in final_rows and "D010-BND" in final_rows:
        factor_effects.append(
            _factor_effect("Intrinsic-range effect at gamma=0.1: D010-BND - D010", final_rows["D010"], final_rows["D010-BND"])
        )
    if "D100" in final_rows and "D010" in final_rows:
        factor_effects.append(_factor_effect("LOS scaling effect under unbounded: D010 - D100", final_rows["D100"], final_rows["D010"]))
    if "BND" in final_rows and "D010-BND" in final_rows:
        factor_effects.append(_factor_effect("LOS scaling effect under bounded: D010-BND - BND", final_rows["BND"], final_rows["D010-BND"]))

    r_beta_15k = next((row for row in beta_rows if int(row["step"]) == int(args.final_step)), None)
    classifications = _classifications(final_rows, r_beta_15k)
    payload = {
        "diagnostic": "bounded_sh3_scratch_summary",
        "final_step": int(args.final_step),
        "trajectory": rows,
        "beta_compensation": beta_rows,
        "final_2x2": list(final_rows.values()),
        "factor_effects": factor_effects,
        "classifications": classifications,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bounded_sh3_trajectory.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    (args.output_dir / "bounded_sh3_final_2x2_summary.json").write_text(
        json.dumps({"rows": list(final_rows.values()), "factor_effects": factor_effects, "classifications": classifications}, indent=2),
        encoding="utf8",
    )
    (args.output_dir / "bounded_sh3_beta_compensation.json").write_text(json.dumps(beta_rows, indent=2), encoding="utf8")
    (args.output_dir / "bounded_sh3_saturation_summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "step": row["step"],
                        "run": row["run"],
                        "c_p99": row["c_p99"],
                        "P(c>0.99)": row["P(c>0.99)"],
                        "SATURATION_MASS_001": row["SATURATION_MASS_001"],
                        "logit_p99": row["logit_p99"],
                        "P(|s|>5)": row["P(|s|>5)"],
                        "sigmoid_derivative_mean": row["sigmoid_derivative_mean"],
                    }
                    for row in rows
                ]
            },
            indent=2,
        ),
        encoding="utf8",
    )
    _write_csv(args.output_dir / "bounded_sh3_trajectory.csv", rows)
    _write_csv(args.output_dir / "bounded_sh3_final_2x2_summary.csv", list(final_rows.values()))
    _write_csv(args.output_dir / "bounded_sh3_beta_compensation.csv", beta_rows)
    _write_csv(
        args.output_dir / "bounded_sh3_saturation_summary.csv",
        [
            {
                "step": row["step"],
                "run": row["run"],
                "c_p99": row["c_p99"],
                "P(c>0.99)": row["P(c>0.99)"],
                "SATURATION_MASS_001": row["SATURATION_MASS_001"],
                "logit_p99": row["logit_p99"],
                "P(|s|>5)": row["P(|s|>5)"],
                "sigmoid_derivative_mean": row["sigmoid_derivative_mean"],
            }
            for row in rows
        ],
    )
    (args.output_dir / "bounded_sh3_summary_all.json").write_text(json.dumps(payload, indent=2), encoding="utf8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, help="RUN:STEP:path/to/summary.json")
    parser.add_argument("--final-step", type=int, default=15000)
    parser.add_argument("--output-dir", type=Path, default=Path("renders/dewater_bounded_sh3_scratch_20260808"))
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "rows": len(payload["trajectory"]),
                "final_rows": len(payload["final_2x2"]),
                "beta_rows": len(payload["beta_compensation"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
