#!/usr/bin/env python
"""Summarize direct optical-depth scale diagnostics and gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CHANNELS = ("r", "g", "b")
RUN_GAMMA = {
    "D100": 1.00,
    "D050": 0.50,
    "D025": 0.25,
    "D010": 0.10,
}


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _metric(summary: Mapping[str, Any], key: str) -> float:
    rows = summary.get("per_view", [])
    return _mean(row.get("metrics", {}).get(key, 0.0) for row in rows)


def _rgb_stat(summary: Mapping[str, Any], group: str, stat: str) -> Dict[str, float]:
    src = summary["aggregate"][group]
    return {channel: float(src[channel].get(stat, 0.0)) for channel in CHANNELS}


def _rgb_stat_mean(summary: Mapping[str, Any], group: str, stat: str) -> float:
    return _mean(_rgb_stat(summary, group, stat).values())


def _threshold_mean(summary: Mapping[str, Any], group: str, key: str) -> float:
    src = summary["aggregate"][group]
    return _mean(float(src[channel].get(key, 0.0)) for channel in CHANNELS)


def _safe_relative_drop(current: float, baseline: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return 0.0
    return 1.0 - float(current) / float(baseline)


def _safe_ratio(current: float, baseline: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return 0.0
    return float(current) / float(baseline)


def _load_summaries(input_root: Path, steps: Sequence[int]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for run in RUN_GAMMA:
        out[run] = {}
        for step in steps:
            path = input_root / run / f"step_{step}" / "summary.json"
            if not path.exists():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text(encoding="utf8"))
            data["_summary_path"] = str(path)
            out[run][step] = data
    return out


def _summarize_run_step(run: str, step: int, summary: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    gamma = float(RUN_GAMMA[run])
    expected_comp = 1.0 / gamma if gamma > 0.0 else 0.0
    psnr = _metric(summary, "psnr")
    ssim = _metric(summary, "ssim")
    lpips = _metric(summary, "lpips")
    base_psnr = _metric(baseline, "psnr")
    base_ssim = _metric(baseline, "ssim")
    base_lpips = _metric(baseline, "lpips")

    beta_raw_mean = _rgb_stat_mean(summary, "beta_D_raw", "mean")
    beta_eff_mean = _rgb_stat_mean(summary, "beta_D_effective", "mean")
    base_beta_raw_mean = _rgb_stat_mean(baseline, "beta_D_raw", "mean")
    tau_p90 = _rgb_stat_mean(summary, "tau_D_effective", "p90")
    base_tau_p90 = _rgb_stat_mean(baseline, "tau_D_effective", "p90")
    t_lt_01 = _threshold_mean(summary, "T_D_effective_thresholds", "P(T<0.1)")
    base_t_lt_01 = _threshold_mean(baseline, "T_D_effective_thresholds", "P(T<0.1)")
    j_gt_1 = _threshold_mean(summary, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
    base_j_gt_1 = _threshold_mean(baseline, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
    j_p99 = _rgb_stat_mean(summary, "clear_object_fullsh_raw", "p99")
    base_j_p99 = _rgb_stat_mean(baseline, "clear_object_fullsh_raw", "p99")

    dpsnr = psnr - base_psnr
    dssim = ssim - base_ssim
    dlpips = lpips - base_lpips
    rgb_safe = bool(dpsnr >= -0.15 and dssim >= -0.0015 and dlpips <= 0.003)
    tau_p90_drop = _safe_relative_drop(tau_p90, base_tau_p90)
    t_lt_01_drop = _safe_relative_drop(t_lt_01, base_t_lt_01)
    j_gt_1_drop = _safe_relative_drop(j_gt_1, base_j_gt_1)
    j_p99_drop = _safe_relative_drop(j_p99, base_j_p99)
    a1 = bool(tau_p90_drop >= 0.20 or t_lt_01_drop >= 0.25)
    a2 = bool(j_gt_1_drop >= 0.25 or j_p99_drop >= 0.15)
    compensation_ratio = _safe_ratio(beta_raw_mean, base_beta_raw_mean)
    compensation_like = bool(
        run != "D100"
        and abs(compensation_ratio / expected_comp - 1.0) <= 0.25
        and abs(tau_p90_drop) <= 0.10
        and abs(j_gt_1_drop) <= 0.10
    )

    return {
        "run": run,
        "checkpoint_step": int(step),
        "gamma_D": gamma,
        "expected_full_compensation": expected_comp,
        "summary_path": summary["_summary_path"],
        "checkpoint": summary.get("checkpoint", ""),
        "view_count": int(summary["aggregate"].get("view_count", 0)),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "delta_psnr_vs_D100": dpsnr,
        "delta_ssim_vs_D100": dssim,
        "delta_lpips_vs_D100": dlpips,
        "rgb_safety_pass": rgb_safe,
        "beta_D_raw_mean_rgb": beta_raw_mean,
        "beta_D_effective_mean_rgb": beta_eff_mean,
        "beta_D_raw_compensation_ratio_vs_D100": compensation_ratio,
        "tau_D_effective_p90_mean_rgb": tau_p90,
        "tau_D_effective_p90_drop_vs_D100": tau_p90_drop,
        "T_D_effective_lt_0p1_mean_rgb": t_lt_01,
        "T_D_effective_lt_0p1_drop_vs_D100": t_lt_01_drop,
        "J_gt_1_mean_rgb": j_gt_1,
        "J_gt_1_drop_vs_D100": j_gt_1_drop,
        "J_p99_mean_rgb": j_p99,
        "J_p99_drop_vs_D100": j_p99_drop,
        "A1_effective_optical_depth_drop_pass": a1,
        "A2_J_saturation_drop_pass": a2,
        "A_gate_pass": bool(a1 and a2 and rgb_safe),
        "beta_compensation_absorbed_like": compensation_like,
        "beta_D_raw_mean_by_channel": _rgb_stat(summary, "beta_D_raw", "mean"),
        "beta_D_effective_mean_by_channel": _rgb_stat(summary, "beta_D_effective", "mean"),
        "tau_D_effective_p90_by_channel": _rgb_stat(summary, "tau_D_effective", "p90"),
        "T_D_effective_lt_0p3_by_channel": {
            channel: float(summary["aggregate"]["T_D_effective_thresholds"][channel]["P(T<0.3)"])
            for channel in CHANNELS
        },
        "T_D_effective_lt_0p2_by_channel": {
            channel: float(summary["aggregate"]["T_D_effective_thresholds"][channel]["P(T<0.2)"])
            for channel in CHANNELS
        },
        "T_D_effective_lt_0p1_by_channel": {
            channel: float(summary["aggregate"]["T_D_effective_thresholds"][channel]["P(T<0.1)"])
            for channel in CHANNELS
        },
        "T_D_effective_lt_0p05_by_channel": {
            channel: float(summary["aggregate"]["T_D_effective_thresholds"][channel]["P(T<0.05)"])
            for channel in CHANNELS
        },
        "J_gt_1_by_channel": {
            channel: float(summary["aggregate"]["clear_object_fullsh_raw_thresholds"][channel]["P(J>1.0)"])
            for channel in CHANNELS
        },
        "J_gt_1p5_by_channel": {
            channel: float(summary["aggregate"]["clear_object_fullsh_raw_thresholds"][channel]["P(J>1.5)"])
            for channel in CHANNELS
        },
        "J_gt_2_by_channel": {
            channel: float(summary["aggregate"]["clear_object_fullsh_raw_thresholds"][channel]["P(J>2.0)"])
            for channel in CHANNELS
        },
    }


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    steps = [int(step) for step in args.steps.split(",") if step.strip()]
    summaries = _load_summaries(args.input_root, steps)
    rows: List[Dict[str, Any]] = []
    for step in steps:
        baseline = summaries["D100"][step]
        for run in RUN_GAMMA:
            rows.append(_summarize_run_step(run, step, summaries[run][step], baseline))

    final_step = int(args.final_step)
    gate_rows = [row for row in rows if int(row["checkpoint_step"]) == final_step]
    best_gate = [row for row in gate_rows if row["A_gate_pass"]]
    if best_gate:
        best_gate = sorted(
            best_gate,
            key=lambda row: (
                row["J_gt_1_drop_vs_D100"],
                row["tau_D_effective_p90_drop_vs_D100"],
                row["delta_psnr_vs_D100"],
            ),
            reverse=True,
        )
    result = {
        "experiment": "direct_optical_depth_scale",
        "scene": args.scene,
        "input_root": str(args.input_root),
        "steps": steps,
        "final_step": final_step,
        "definitions": {
            "gamma_D": "direct_optical_depth_scale; only medium_attn is multiplied before direct attenuation",
            "beta_D_raw": "outputs['medium_attn_raw']; raw medium attenuation coefficient",
            "beta_D_effective": "outputs['medium_attn']; gamma_D * beta_D_raw",
            "tau_D_effective": "beta_D_effective * rendered depth",
            "T_D_effective": "exp(-tau_D_effective)",
            "gate_scope": "A gate is evaluated at the final checkpoint against matched-step D100 using RGB-channel means.",
        },
        "rows": rows,
        "final_gate_rows": gate_rows,
        "A_pass_runs": [row["run"] for row in best_gate],
        "recommended_A_star": best_gate[0]["run"] if best_gate else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    _write_csv(args.output_csv, rows)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "run",
        "checkpoint_step",
        "gamma_D",
        "expected_full_compensation",
        "view_count",
        "psnr",
        "ssim",
        "lpips",
        "delta_psnr_vs_D100",
        "delta_ssim_vs_D100",
        "delta_lpips_vs_D100",
        "rgb_safety_pass",
        "beta_D_raw_mean_rgb",
        "beta_D_effective_mean_rgb",
        "beta_D_raw_compensation_ratio_vs_D100",
        "tau_D_effective_p90_mean_rgb",
        "tau_D_effective_p90_drop_vs_D100",
        "T_D_effective_lt_0p1_mean_rgb",
        "T_D_effective_lt_0p1_drop_vs_D100",
        "J_gt_1_mean_rgb",
        "J_gt_1_drop_vs_D100",
        "J_p99_mean_rgb",
        "J_p99_drop_vs_D100",
        "A1_effective_optical_depth_drop_pass",
        "A2_J_saturation_drop_pass",
        "A_gate_pass",
        "beta_compensation_absorbed_like",
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
    parser.add_argument("--input-root", type=Path, default=Path("renders/dewater_optical_depth_20260807/A"))
    parser.add_argument("--steps", default="11000,12000,13000")
    parser.add_argument("--final-step", type=int, default=13000)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/dewater_optical_depth_20260807/direct_optical_depth_sweep_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/dewater_optical_depth_20260807/direct_optical_depth_sweep_summary.csv"))
    args = parser.parse_args()
    result = summarize(args)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "scene": result["scene"],
                "final_step": result["final_step"],
                "A_pass_runs": result["A_pass_runs"],
                "recommended_A_star": result["recommended_A_star"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
