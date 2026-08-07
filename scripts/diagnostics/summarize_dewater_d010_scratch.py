#!/usr/bin/env python
"""Summarize D010-from-scratch trajectory and three-path final comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


CHANNELS = ("r", "g", "b")


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf8"))
    data["_summary_path"] = str(path)
    return data


def _maybe_load(path: Path) -> Optional[Dict[str, Any]]:
    return _load(path) if path.exists() else None


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


def _row(run: str, step: int, summary: Mapping[str, Any], baseline: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    psnr = _metric(summary, "psnr")
    ssim = _metric(summary, "ssim")
    lpips = _metric(summary, "lpips")
    beta_raw = _rgb_stat_mean(summary, "beta_D_raw", "mean")
    beta_eff = _rgb_stat_mean(summary, "beta_D_effective", "mean")
    tau_p90 = _rgb_stat_mean(summary, "tau_D_effective", "p90")
    t_mean = _rgb_stat_mean(summary, "T_D_effective", "mean")
    t_lt_01 = _threshold_mean(summary, "T_D_effective_thresholds", "P(T<0.1)")
    t_lt_005 = _threshold_mean(summary, "T_D_effective_thresholds", "P(T<0.05)")
    j_gt_1 = _threshold_mean(summary, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
    j_gt_15 = _threshold_mean(summary, "clear_object_fullsh_raw_thresholds", "P(J>1.5)")
    j_gt_2 = _threshold_mean(summary, "clear_object_fullsh_raw_thresholds", "P(J>2.0)")
    j_p95 = _rgb_stat_mean(summary, "clear_object_fullsh_raw", "p95")
    j_p99 = _rgb_stat_mean(summary, "clear_object_fullsh_raw", "p99")
    out = {
        "step": int(step),
        "run": run,
        "summary_path": summary["_summary_path"],
        "checkpoint": summary.get("checkpoint", ""),
        "loaded_step": int(summary.get("loaded_step", step)),
        "gamma_D": float(summary.get("direct_optical_depth_scale", 1.0)),
        "view_count": int(summary.get("aggregate", {}).get("view_count", 0)),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "beta_D_raw_mean_rgb": beta_raw,
        "beta_D_effective_mean_rgb": beta_eff,
        "tau_D_effective_p50_mean_rgb": _rgb_stat_mean(summary, "tau_D_effective", "p50"),
        "tau_D_effective_p90_mean_rgb": tau_p90,
        "tau_D_effective_p99_mean_rgb": _rgb_stat_mean(summary, "tau_D_effective", "p99"),
        "T_D_effective_mean_rgb": t_mean,
        "T_D_effective_lt_0p1_mean_rgb": t_lt_01,
        "T_D_effective_lt_0p05_mean_rgb": t_lt_005,
        "J_p95_mean_rgb": j_p95,
        "J_p99_mean_rgb": j_p99,
        "J_gt_1_mean_rgb": j_gt_1,
        "J_gt_1p5_mean_rgb": j_gt_15,
        "J_gt_2_mean_rgb": j_gt_2,
        "gaussian_count": int(summary.get("model_state", {}).get("gaussian_count", 0)),
    }
    if baseline is not None:
        base_psnr = _metric(baseline, "psnr")
        base_ssim = _metric(baseline, "ssim")
        base_lpips = _metric(baseline, "lpips")
        dpsnr = psnr - base_psnr
        dssim = ssim - base_ssim
        dlpips = lpips - base_lpips
        out.update(
            {
                "d100_available": True,
                "delta_psnr_vs_D100_same_step": dpsnr,
                "delta_ssim_vs_D100_same_step": dssim,
                "delta_lpips_vs_D100_same_step": dlpips,
                "rgb_safety_pass_vs_D100": bool(dpsnr >= -0.15 and dssim >= -0.0015 and dlpips <= 0.003),
                "tau_D_effective_p90_reduction_vs_D100": _safe_relative_drop(
                    tau_p90, _rgb_stat_mean(baseline, "tau_D_effective", "p90")
                ),
                "J_gt_1_reduction_vs_D100": _safe_relative_drop(
                    j_gt_1, _threshold_mean(baseline, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
                ),
                "beta_raw_ratio_vs_D100": beta_raw / max(_rgb_stat_mean(baseline, "beta_D_raw", "mean"), 1e-12),
            }
        )
    else:
        out.update(
            {
                "d100_available": False,
                "delta_psnr_vs_D100_same_step": "",
                "delta_ssim_vs_D100_same_step": "",
                "delta_lpips_vs_D100_same_step": "",
                "rgb_safety_pass_vs_D100": "",
                "tau_D_effective_p90_reduction_vs_D100": "",
                "J_gt_1_reduction_vs_D100": "",
                "beta_raw_ratio_vs_D100": "",
            }
        )
    return out


def _parse_train_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "events": []}
    text = path.read_text(encoding="utf8", errors="ignore")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = " ".join(text.splitlines())
    densification_re = re.compile(
        r"Densification step=(?P<step>\d+) split=(?P<split>\d+) duplicate=(?P<duplicate>\d+) "
        r"split_children=(?P<split_children>\d+).*?"
        r"duplicate_children=(?P<duplicate_children>\d+) total_before=(?P<total_before>\d+) "
        r"total_after_append=(?P<total_after_append>\d+)",
        re.DOTALL,
    )
    cull_re = re.compile(
        r"Culled step=(?P<step>\d+) (?P<culled>\d+) gaussians "
        r"\((?P<below_alpha>\d+) below alpha thresh, (?P<too_big>\d+) too bigs, "
        r"(?P<remaining>\d+).*?remaining\)",
        re.DOTALL,
    )
    events: List[Dict[str, Any]] = []
    for match in densification_re.finditer(text):
        row = {key: int(value) for key, value in match.groupdict().items()}
        row["event"] = "densification"
        events.append(row)
    for match in cull_re.finditer(text):
        row = {key: int(value) for key, value in match.groupdict().items()}
        row["event"] = "cull"
        events.append(row)
    events.sort(key=lambda row: (row["step"], row["event"]))
    return {"available": True, "path": str(path), "events": events}


def _final_decision(final_rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    switch = final_rows["D010-SWITCH"]
    scratch = final_rows["D010-SCRATCH"]
    scratch_safe = bool(scratch.get("rgb_safety_pass_vs_D100", False))
    scratch_tau = float(scratch.get("tau_D_effective_p90_reduction_vs_D100", 0.0))
    scratch_j = float(scratch.get("J_gt_1_reduction_vs_D100", 0.0))
    psnr_gap_vs_switch = float(scratch["psnr"]) - float(switch["psnr"])
    if scratch_safe and scratch_tau >= 0.20 and scratch_j >= 0.20 and psnr_gap_vs_switch >= -0.10:
        label = "Better parameterization"
    elif bool(switch.get("rgb_safety_pass_vs_D100", False)) and (
        (not scratch_safe) or scratch_tau < 0.20 or scratch_j < 0.20 or psnr_gap_vs_switch < -0.10
    ):
        label = "Late-stage recalibration / basin switching"
    else:
        label = "Evidence remains mixed"
    return {
        "label": label,
        "scratch_rgb_safety": scratch_safe,
        "scratch_tau_p90_reduction_vs_D100": scratch_tau,
        "scratch_J_gt_1_reduction_vs_D100": scratch_j,
        "scratch_psnr_minus_switch": psnr_gap_vs_switch,
    }


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    steps = [int(step) for step in args.steps.split(",") if step.strip()]
    rows: List[Dict[str, Any]] = []
    for step in steps:
        scratch = _load(args.scratch_root / "D010-SCRATCH" / f"step_{step}" / "summary.json")
        baseline = _maybe_load(args.scratch_root / "D100-SCRATCH" / f"step_{step}" / "summary.json")
        rows.append(_row("D010-SCRATCH", step, scratch, baseline))
        if baseline is not None:
            rows.append(_row("D100-SCRATCH", step, baseline, baseline))

    persistence = _load(args.persistence_summary)
    p_rows = persistence["rows"]
    final_step = int(args.final_step)
    d100_final_summary = _load(args.scratch_root / "D100-SCRATCH" / f"step_{final_step}" / "summary.json")
    scratch_final_summary = _load(args.scratch_root / "D010-SCRATCH" / f"step_{final_step}" / "summary.json")
    switch_final_row = [
        row for row in p_rows if row["run"] == "D010-PERSIST" and int(row["step"]) == final_step
    ][0]
    switch_final_summary = _load(Path(switch_final_row["summary_path"]))
    d100_final_row = _row("D100-SCRATCH", final_step, d100_final_summary, d100_final_summary)
    scratch_final_row = _row("D010-SCRATCH", final_step, scratch_final_summary, d100_final_summary)
    switch_final = _row("D010-SWITCH", final_step, switch_final_summary, d100_final_summary)
    three_path = {
        "D100-SCRATCH": d100_final_row,
        "D010-SWITCH": switch_final,
        "D010-SCRATCH": scratch_final_row,
    }
    train_log = _parse_train_log(args.train_log)
    result = {
        "experiment": "d010_scratch_0_to_15k",
        "scene": args.scene,
        "steps": steps,
        "final_step": final_step,
        "trajectory_rows": rows,
        "three_path_final": three_path,
        "decision": _final_decision(three_path),
        "densification_log": train_log,
        "definitions": {
            "D100-SCRATCH": "gamma=1 baseline using existing Curasao M1-compatible checkpoints/continuation diagnostics",
            "D010-SWITCH": "gamma=1 from 0-10k, gamma=0.1 from 10-15k",
            "D010-SCRATCH": "gamma=0.1 from step 0 with default beta initialization",
        },
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
        "d100_available",
        "psnr",
        "ssim",
        "lpips",
        "delta_psnr_vs_D100_same_step",
        "delta_ssim_vs_D100_same_step",
        "delta_lpips_vs_D100_same_step",
        "rgb_safety_pass_vs_D100",
        "beta_D_raw_mean_rgb",
        "beta_D_effective_mean_rgb",
        "beta_raw_ratio_vs_D100",
        "tau_D_effective_p90_mean_rgb",
        "tau_D_effective_p90_reduction_vs_D100",
        "T_D_effective_mean_rgb",
        "T_D_effective_lt_0p1_mean_rgb",
        "T_D_effective_lt_0p05_mean_rgb",
        "J_p95_mean_rgb",
        "J_p99_mean_rgb",
        "J_gt_1_mean_rgb",
        "J_gt_1_reduction_vs_D100",
        "J_gt_1p5_mean_rgb",
        "J_gt_2_mean_rgb",
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
    parser.add_argument("--scratch-root", type=Path, default=Path("renders/dewater_d010_scratch_20260807"))
    parser.add_argument(
        "--persistence-summary",
        type=Path,
        default=Path("outputs/dewater_d010_persistence_20260807/d010_persistence_summary.json"),
    )
    parser.add_argument("--steps", default="1000,3000,5000,8000,10000,13000,15000")
    parser.add_argument("--final-step", type=int, default=15000)
    parser.add_argument(
        "--train-log",
        type=Path,
        default=Path(
            "logs/dewater_d010_scratch_20260807/dewater_d010_scratch_curasao_seed42_step0_to_15000_20260807_d010_scratch_g0p10/train.log"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/dewater_d010_scratch_20260807/d010_scratch_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/dewater_d010_scratch_20260807/d010_scratch_summary.csv"),
    )
    args = parser.parse_args()
    result = summarize(args)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
