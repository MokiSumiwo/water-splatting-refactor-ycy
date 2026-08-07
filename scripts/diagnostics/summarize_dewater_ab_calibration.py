#!/usr/bin/env python
"""Summarize the single D010+BG010 dewatering AB calibration run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


CHANNELS = ("r", "g", "b")
AB_RUN = "AB_D010_BG010"


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


def _background(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    bg = summary.get("background_supervision", {})
    if not bg.get("available", False):
        raise RuntimeError(f"Missing background supervision stats in {summary.get('_summary_path', '<unknown>')}")
    return bg


def _load_step_bundle(args: argparse.Namespace, step: int) -> Dict[str, Dict[str, Any]]:
    return {
        "ab_eval": _load(args.ab_root / AB_RUN / f"step_{step}" / "eval" / "summary.json"),
        "ab_train_bg": _load(args.ab_root / AB_RUN / f"step_{step}" / "train_background" / "summary.json"),
        "d100_eval": _load(args.a_root / "D100" / f"step_{step}" / "summary.json"),
        "d010_eval": _load(args.a_root / "D010" / f"step_{step}" / "summary.json"),
        "bg000_train_bg": _load(args.b_root / "BG000" / f"step_{step}" / "train_background" / "summary.json"),
        "bg010_train_bg": _load(args.b_root / "BG010" / f"step_{step}" / "train_background" / "summary.json"),
    }


def _summarize_step(step: int, bundle: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    ab_eval = bundle["ab_eval"]
    ab_train_bg = bundle["ab_train_bg"]
    d100_eval = bundle["d100_eval"]
    d010_eval = bundle["d010_eval"]
    bg000_train_bg = bundle["bg000_train_bg"]
    bg010_train_bg = bundle["bg010_train_bg"]

    psnr = _metric(ab_eval, "psnr")
    ssim = _metric(ab_eval, "ssim")
    lpips = _metric(ab_eval, "lpips")
    d100_psnr = _metric(d100_eval, "psnr")
    d100_ssim = _metric(d100_eval, "ssim")
    d100_lpips = _metric(d100_eval, "lpips")
    d010_psnr = _metric(d010_eval, "psnr")
    d010_ssim = _metric(d010_eval, "ssim")
    d010_lpips = _metric(d010_eval, "lpips")

    delta_psnr_vs_d100 = psnr - d100_psnr
    delta_ssim_vs_d100 = ssim - d100_ssim
    delta_lpips_vs_d100 = lpips - d100_lpips
    rgb_safe_vs_d100 = bool(delta_psnr_vs_d100 >= -0.15 and delta_ssim_vs_d100 >= -0.0015 and delta_lpips_vs_d100 <= 0.003)

    beta_raw_mean = _rgb_stat_mean(ab_eval, "beta_D_raw", "mean")
    beta_eff_mean = _rgb_stat_mean(ab_eval, "beta_D_effective", "mean")
    d100_beta_raw = _rgb_stat_mean(d100_eval, "beta_D_raw", "mean")

    tau_p90 = _rgb_stat_mean(ab_eval, "tau_D_effective", "p90")
    d100_tau_p90 = _rgb_stat_mean(d100_eval, "tau_D_effective", "p90")
    d010_tau_p90 = _rgb_stat_mean(d010_eval, "tau_D_effective", "p90")
    tau_drop_vs_d100 = _safe_relative_drop(tau_p90, d100_tau_p90)
    tau_drop_vs_d010 = _safe_relative_drop(tau_p90, d010_tau_p90)

    t_lt_01 = _threshold_mean(ab_eval, "T_D_effective_thresholds", "P(T<0.1)")
    d100_t_lt_01 = _threshold_mean(d100_eval, "T_D_effective_thresholds", "P(T<0.1)")
    t_lt_01_drop_vs_d100 = _safe_relative_drop(t_lt_01, d100_t_lt_01)

    j_gt_1 = _threshold_mean(ab_eval, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
    d100_j_gt_1 = _threshold_mean(d100_eval, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
    d010_j_gt_1 = _threshold_mean(d010_eval, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
    j_gt_1_drop_vs_d100 = _safe_relative_drop(j_gt_1, d100_j_gt_1)
    j_gt_1_drop_vs_d010 = _safe_relative_drop(j_gt_1, d010_j_gt_1)

    j_p99 = _rgb_stat_mean(ab_eval, "clear_object_fullsh_raw", "p99")
    d100_j_p99 = _rgb_stat_mean(d100_eval, "clear_object_fullsh_raw", "p99")
    d010_j_p99 = _rgb_stat_mean(d010_eval, "clear_object_fullsh_raw", "p99")
    j_p99_drop_vs_d100 = _safe_relative_drop(j_p99, d100_j_p99)
    j_p99_drop_vs_d010 = _safe_relative_drop(j_p99, d010_j_p99)

    bg = _background(ab_train_bg)
    bg000 = _background(bg000_train_bg)
    bg010 = _background(bg010_train_bg)
    background_l1 = float(bg.get("background_medium_l1", 0.0))
    weighted_background_l1 = float(bg.get("weighted_background_medium_l1", 0.0))
    background_l1_drop_vs_bg000 = _safe_relative_drop(background_l1, float(bg000.get("background_medium_l1", 0.0)))
    background_l1_drop_vs_bg010 = _safe_relative_drop(background_l1, float(bg010.get("background_medium_l1", 0.0)))

    a1 = bool(tau_drop_vs_d100 >= 0.20 or t_lt_01_drop_vs_d100 >= 0.25)
    a2 = bool(j_gt_1_drop_vs_d100 >= 0.25 or j_p99_drop_vs_d100 >= 0.15)
    residual_pass = bool(background_l1_drop_vs_bg000 >= 0.20)

    return {
        "run": AB_RUN,
        "checkpoint_step": int(step),
        "gamma_D": 0.10,
        "lambda_bg": 0.01,
        "expected_full_compensation": 10.0,
        "summary_path": ab_eval["_summary_path"],
        "train_background_summary_path": ab_train_bg["_summary_path"],
        "checkpoint": ab_eval.get("checkpoint", ""),
        "eval_view_count": int(ab_eval["aggregate"].get("view_count", 0)),
        "train_background_view_count": int(ab_train_bg["aggregate"].get("view_count", 0)),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "delta_psnr_vs_D100": delta_psnr_vs_d100,
        "delta_ssim_vs_D100": delta_ssim_vs_d100,
        "delta_lpips_vs_D100": delta_lpips_vs_d100,
        "delta_psnr_vs_D010": psnr - d010_psnr,
        "delta_ssim_vs_D010": ssim - d010_ssim,
        "delta_lpips_vs_D010": lpips - d010_lpips,
        "rgb_safety_pass_vs_D100": rgb_safe_vs_d100,
        "beta_D_raw_mean_rgb": beta_raw_mean,
        "beta_D_effective_mean_rgb": beta_eff_mean,
        "beta_D_raw_compensation_ratio_vs_D100": _safe_ratio(beta_raw_mean, d100_beta_raw),
        "tau_D_effective_p90_mean_rgb": tau_p90,
        "tau_D_effective_p90_drop_vs_D100": tau_drop_vs_d100,
        "tau_D_effective_p90_drop_vs_D010": tau_drop_vs_d010,
        "T_D_effective_lt_0p1_mean_rgb": t_lt_01,
        "T_D_effective_lt_0p1_drop_vs_D100": t_lt_01_drop_vs_d100,
        "J_gt_1_mean_rgb": j_gt_1,
        "J_gt_1_drop_vs_D100": j_gt_1_drop_vs_d100,
        "J_gt_1_drop_vs_D010": j_gt_1_drop_vs_d010,
        "J_p99_mean_rgb": j_p99,
        "J_p99_drop_vs_D100": j_p99_drop_vs_d100,
        "J_p99_drop_vs_D010": j_p99_drop_vs_d010,
        "background_medium_l1": background_l1,
        "weighted_background_medium_l1": weighted_background_l1,
        "background_medium_l1_drop_vs_BG000": background_l1_drop_vs_bg000,
        "background_medium_l1_drop_vs_BG010": background_l1_drop_vs_bg010,
        "A1_effective_optical_depth_drop_pass_vs_D100": a1,
        "A2_J_saturation_drop_pass_vs_D100": a2,
        "A_like_gate_pass_vs_D100": bool(a1 and a2 and rgb_safe_vs_d100),
        "background_residual_pass_vs_BG000": residual_pass,
        "beta_D_raw_mean_by_channel": _rgb_stat(ab_eval, "beta_D_raw", "mean"),
        "beta_D_effective_mean_by_channel": _rgb_stat(ab_eval, "beta_D_effective", "mean"),
        "tau_D_effective_p90_by_channel": _rgb_stat(ab_eval, "tau_D_effective", "p90"),
        "T_D_effective_lt_0p3_by_channel": {
            channel: float(ab_eval["aggregate"]["T_D_effective_thresholds"][channel]["P(T<0.3)"])
            for channel in CHANNELS
        },
        "T_D_effective_lt_0p2_by_channel": {
            channel: float(ab_eval["aggregate"]["T_D_effective_thresholds"][channel]["P(T<0.2)"])
            for channel in CHANNELS
        },
        "T_D_effective_lt_0p1_by_channel": {
            channel: float(ab_eval["aggregate"]["T_D_effective_thresholds"][channel]["P(T<0.1)"])
            for channel in CHANNELS
        },
        "T_D_effective_lt_0p05_by_channel": {
            channel: float(ab_eval["aggregate"]["T_D_effective_thresholds"][channel]["P(T<0.05)"])
            for channel in CHANNELS
        },
        "J_gt_1_by_channel": {
            channel: float(ab_eval["aggregate"]["clear_object_fullsh_raw_thresholds"][channel]["P(J>1.0)"])
            for channel in CHANNELS
        },
        "J_gt_1p5_by_channel": {
            channel: float(ab_eval["aggregate"]["clear_object_fullsh_raw_thresholds"][channel]["P(J>1.5)"])
            for channel in CHANNELS
        },
        "J_gt_2_by_channel": {
            channel: float(ab_eval["aggregate"]["clear_object_fullsh_raw_thresholds"][channel]["P(J>2.0)"])
            for channel in CHANNELS
        },
        "background_mask_coverage": bg.get("coverage", {}),
        "background_medium_rgb_mean_by_channel": bg.get("medium_rgb_mean", {}),
        "background_medium_bs_mean_by_channel": bg.get("medium_bs_mean", {}),
        "background_medium_attn_raw_mean_by_channel": bg.get("medium_attn_raw_mean", {}),
        "background_medium_attn_effective_mean_by_channel": bg.get("medium_attn_effective_mean", {}),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "run",
        "checkpoint_step",
        "gamma_D",
        "lambda_bg",
        "eval_view_count",
        "train_background_view_count",
        "psnr",
        "ssim",
        "lpips",
        "delta_psnr_vs_D100",
        "delta_ssim_vs_D100",
        "delta_lpips_vs_D100",
        "delta_psnr_vs_D010",
        "delta_ssim_vs_D010",
        "delta_lpips_vs_D010",
        "rgb_safety_pass_vs_D100",
        "beta_D_raw_mean_rgb",
        "beta_D_effective_mean_rgb",
        "beta_D_raw_compensation_ratio_vs_D100",
        "tau_D_effective_p90_mean_rgb",
        "tau_D_effective_p90_drop_vs_D100",
        "tau_D_effective_p90_drop_vs_D010",
        "T_D_effective_lt_0p1_mean_rgb",
        "T_D_effective_lt_0p1_drop_vs_D100",
        "J_gt_1_mean_rgb",
        "J_gt_1_drop_vs_D100",
        "J_gt_1_drop_vs_D010",
        "J_p99_mean_rgb",
        "J_p99_drop_vs_D100",
        "J_p99_drop_vs_D010",
        "background_medium_l1",
        "weighted_background_medium_l1",
        "background_medium_l1_drop_vs_BG000",
        "background_medium_l1_drop_vs_BG010",
        "A_like_gate_pass_vs_D100",
        "background_residual_pass_vs_BG000",
        "summary_path",
        "train_background_summary_path",
        "checkpoint",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    steps = [int(step) for step in args.steps.split(",") if step.strip()]
    rows = [_summarize_step(step, _load_step_bundle(args, step)) for step in steps]
    final_rows = [row for row in rows if int(row["checkpoint_step"]) == int(args.final_step)]
    result = {
        "experiment": "single_ab_direct_scale_plus_background_supervision",
        "scene": args.scene,
        "ab_run": AB_RUN,
        "steps": steps,
        "final_step": int(args.final_step),
        "definitions": {
            "AB": "D010 direct optical-depth scale gamma_D=0.10 plus BG010 medium-background supervision lambda_bg=0.01",
            "trigger_reason": "D010 passed the predefined A mechanism gate; BG010 did not pass the B gate, so only one AB combination was run.",
            "A_like_gate_vs_D100": "same A gate thresholds applied to AB relative to D100",
            "background_residual_pass_vs_BG000": "background_medium_l1 drop >=20% relative to BG000 train-background diagnostics",
        },
        "rows": rows,
        "final_gate_rows": final_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    _write_csv(args.output_csv, rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Curasao")
    parser.add_argument("--ab-root", type=Path, default=Path("renders/dewater_optical_depth_20260807/AB"))
    parser.add_argument("--a-root", type=Path, default=Path("renders/dewater_optical_depth_20260807/A"))
    parser.add_argument("--b-root", type=Path, default=Path("renders/dewater_optical_depth_20260807/B"))
    parser.add_argument("--steps", default="11000,12000,13000")
    parser.add_argument("--final-step", type=int, default=13000)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/dewater_optical_depth_20260807/ab_d010_bg010_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/dewater_optical_depth_20260807/ab_d010_bg010_summary.csv"))
    args = parser.parse_args()
    result = summarize(args)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "scene": result["scene"],
                "final_step": result["final_step"],
                "final_gate_rows": result["final_gate_rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
