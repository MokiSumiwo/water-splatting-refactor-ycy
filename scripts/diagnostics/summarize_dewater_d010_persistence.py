#!/usr/bin/env python
"""Summarize D100/D010 persistence from 13k to 15k."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


CHANNELS = ("r", "g", "b")
RUNS = ("D100-PERSIST", "D010-PERSIST")


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf8"))
    data["_summary_path"] = str(path)
    return data


def _metric(summary: Mapping[str, Any], key: str) -> float:
    return _mean(row.get("metrics", {}).get(key, 0.0) for row in summary.get("per_view", []))


def _rgb_stat(summary: Mapping[str, Any], group: str, stat: str) -> Dict[str, float]:
    src = summary["aggregate"][group]
    return {channel: float(src[channel].get(stat, 0.0)) for channel in CHANNELS}


def _rgb_stat_mean(summary: Mapping[str, Any], group: str, stat: str) -> float:
    return _mean(_rgb_stat(summary, group, stat).values())


def _threshold_mean(summary: Mapping[str, Any], group: str, key: str) -> float:
    return _mean(float(summary["aggregate"][group][channel].get(key, 0.0)) for channel in CHANNELS)


def _safe_relative_drop(current: float, baseline: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return 0.0
    return 1.0 - float(current) / float(baseline)


def _safe_ratio(current: float, baseline: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return 0.0
    return float(current) / float(baseline)


def _summary_path(args: argparse.Namespace, run: str, step: int) -> Path:
    if int(step) == 13000:
        old_run = "D100" if run == "D100-PERSIST" else "D010"
        return args.a_root / old_run / "step_13000" / "summary.json"
    return args.persist_root / run / f"step_{int(step)}" / "summary.json"


def _row(run: str, step: int, summary: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    psnr = _metric(summary, "psnr")
    ssim = _metric(summary, "ssim")
    lpips = _metric(summary, "lpips")
    base_psnr = _metric(baseline, "psnr")
    base_ssim = _metric(baseline, "ssim")
    base_lpips = _metric(baseline, "lpips")
    dpsnr = psnr - base_psnr
    dssim = ssim - base_ssim
    dlpips = lpips - base_lpips
    rgb_safe = bool(dpsnr >= -0.15 and dssim >= -0.0015 and dlpips <= 0.003)

    beta_raw = _rgb_stat_mean(summary, "beta_D_raw", "mean")
    beta_eff = _rgb_stat_mean(summary, "beta_D_effective", "mean")
    base_beta_raw = _rgb_stat_mean(baseline, "beta_D_raw", "mean")
    tau_p90 = _rgb_stat_mean(summary, "tau_D_effective", "p90")
    base_tau_p90 = _rgb_stat_mean(baseline, "tau_D_effective", "p90")
    t_lt_01 = _threshold_mean(summary, "T_D_effective_thresholds", "P(T<0.1)")
    t_lt_005 = _threshold_mean(summary, "T_D_effective_thresholds", "P(T<0.05)")
    j_gt_1 = _threshold_mean(summary, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
    j_gt_15 = _threshold_mean(summary, "clear_object_fullsh_raw_thresholds", "P(J>1.5)")
    j_gt_2 = _threshold_mean(summary, "clear_object_fullsh_raw_thresholds", "P(J>2.0)")
    base_j_gt_1 = _threshold_mean(baseline, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
    j_p99 = _rgb_stat_mean(summary, "clear_object_fullsh_raw", "p99")

    return {
        "step": int(step),
        "run": run,
        "summary_path": summary["_summary_path"],
        "checkpoint": summary.get("checkpoint", ""),
        "loaded_step": int(summary.get("loaded_step", step)),
        "view_count": int(summary.get("aggregate", {}).get("view_count", 0)),
        "gamma_D": float(summary.get("direct_optical_depth_scale", 1.0)),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "delta_psnr_vs_D100_same_step": dpsnr,
        "delta_ssim_vs_D100_same_step": dssim,
        "delta_lpips_vs_D100_same_step": dlpips,
        "rgb_safety_pass": rgb_safe if run == "D010-PERSIST" else True,
        "beta_D_raw_mean_rgb": beta_raw,
        "beta_D_effective_mean_rgb": beta_eff,
        "R_beta_vs_D100": _safe_ratio(beta_raw, base_beta_raw),
        "ten_minus_R_beta": 10.0 - _safe_ratio(beta_raw, base_beta_raw),
        "tau_D_effective_p90_mean_rgb": tau_p90,
        "tau_D_effective_p90_reduction_vs_D100": _safe_relative_drop(tau_p90, base_tau_p90),
        "T_D_effective_lt_0p1_mean_rgb": t_lt_01,
        "T_D_effective_lt_0p05_mean_rgb": t_lt_005,
        "J_gt_1_mean_rgb": j_gt_1,
        "J_gt_1_reduction_vs_D100": _safe_relative_drop(j_gt_1, base_j_gt_1),
        "J_gt_1p5_mean_rgb": j_gt_15,
        "J_gt_2_mean_rgb": j_gt_2,
        "J_p99_mean_rgb": j_p99,
        "gaussian_count": int(summary.get("model_state", {}).get("gaussian_count", 0)),
        "beta_D_raw_mean_by_channel": _rgb_stat(summary, "beta_D_raw", "mean"),
        "beta_D_raw_p90_by_channel": _rgb_stat(summary, "beta_D_raw", "p90"),
        "beta_D_effective_mean_by_channel": _rgb_stat(summary, "beta_D_effective", "mean"),
        "tau_D_effective_p90_by_channel": _rgb_stat(summary, "tau_D_effective", "p90"),
        "T_D_effective_mean_by_channel": _rgb_stat(summary, "T_D_effective", "mean"),
        "J_gt_1_by_channel": {
            channel: float(summary["aggregate"]["clear_object_fullsh_raw_thresholds"][channel]["P(J>1.0)"])
            for channel in CHANNELS
        },
    }


def _classify(final_d010: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    r_beta = float(final_d010["R_beta_vs_D100"])
    tau_reduction = float(final_d010["tau_D_effective_p90_reduction_vs_D100"])
    j_reduction = float(final_d010["J_gt_1_reduction_vs_D100"])
    rgb_safe = bool(final_d010["rgb_safety_pass"])
    if rgb_safe and r_beta < 8.0 and tau_reduction >= 0.20 and j_reduction >= 0.20:
        label = "P-A"
        decision = "D010 changes the persistent optimization solution."
        trigger_scratch = True
    elif rgb_safe and 8.0 <= r_beta < 9.0 and tau_reduction >= 0.20 and j_reduction >= 0.20:
        label = "P-B"
        decision = "partial compensation catch-up, but effective dewatering signal remains."
        trigger_scratch = True
    else:
        label = "P-C"
        decision = "D010 primarily delays beta_D compensation or lacks the required 15k mechanism margin."
        trigger_scratch = False
    r_traj = [
        row["R_beta_vs_D100"]
        for row in rows
        if row["run"] == "D010-PERSIST"
    ]
    return {
        "classification": label,
        "decision": decision,
        "trigger_scratch": trigger_scratch,
        "R_beta_trajectory": r_traj,
        "R_beta_13k": r_traj[0] if len(r_traj) > 0 else None,
        "R_beta_14k": r_traj[1] if len(r_traj) > 1 else None,
        "R_beta_15k": r_traj[2] if len(r_traj) > 2 else None,
        "tau_p90_reduction_15k": tau_reduction,
        "J_gt_1_reduction_15k": j_reduction,
        "rgb_safety_15k": rgb_safe,
    }


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    steps = [int(step) for step in args.steps.split(",") if step.strip()]
    summaries: Dict[int, Dict[str, Dict[str, Any]]] = {}
    rows = []
    for step in steps:
        summaries[step] = {run: _load(_summary_path(args, run, step)) for run in RUNS}
        baseline = summaries[step]["D100-PERSIST"]
        for run in RUNS:
            rows.append(_row(run, step, summaries[step][run], baseline))
    final_d010 = [row for row in rows if row["run"] == "D010-PERSIST" and row["step"] == args.final_step][0]
    classification = _classify(final_d010, rows)
    result = {
        "experiment": "d010_persistence_13k_to_15k",
        "scene": args.scene,
        "steps": steps,
        "final_step": int(args.final_step),
        "definitions": {
            "R_beta": "mean_beta_D_raw(D010, step) / mean_beta_D_raw(D100, same step)",
            "RGB_safety": "D010 vs D100 at the same step: dPSNR >= -0.15, dSSIM >= -0.0015, dLPIPS <= +0.003",
            "P-A": "RGB safe, R_beta_15k < 8, tau p90 reduction >=20%, P(J>1) reduction >=20%",
            "P-B": "RGB safe, 8 <= R_beta_15k < 9, tau p90 reduction >=20%, P(J>1) reduction >=20%",
            "P-C": "does not satisfy P-A/P-B persistence criteria",
        },
        "rows": rows,
        "classification": classification,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    _write_csv(args.output_csv, rows)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "step",
        "run",
        "gamma_D",
        "loaded_step",
        "view_count",
        "psnr",
        "ssim",
        "lpips",
        "delta_psnr_vs_D100_same_step",
        "delta_ssim_vs_D100_same_step",
        "delta_lpips_vs_D100_same_step",
        "rgb_safety_pass",
        "beta_D_raw_mean_rgb",
        "beta_D_effective_mean_rgb",
        "R_beta_vs_D100",
        "ten_minus_R_beta",
        "tau_D_effective_p90_mean_rgb",
        "tau_D_effective_p90_reduction_vs_D100",
        "T_D_effective_lt_0p1_mean_rgb",
        "T_D_effective_lt_0p05_mean_rgb",
        "J_gt_1_mean_rgb",
        "J_gt_1_reduction_vs_D100",
        "J_gt_1p5_mean_rgb",
        "J_gt_2_mean_rgb",
        "J_p99_mean_rgb",
        "gaussian_count",
        "summary_path",
        "checkpoint",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Curasao")
    parser.add_argument("--a-root", type=Path, default=Path("renders/dewater_optical_depth_20260807/A"))
    parser.add_argument("--persist-root", type=Path, default=Path("renders/dewater_d010_persistence_20260807"))
    parser.add_argument("--steps", default="13000,14000,15000")
    parser.add_argument("--final-step", type=int, default=15000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/dewater_d010_persistence_20260807/d010_persistence_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/dewater_d010_persistence_20260807/d010_persistence_summary.csv"),
    )
    args = parser.parse_args()
    result = summarize(args)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "classification": result["classification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
