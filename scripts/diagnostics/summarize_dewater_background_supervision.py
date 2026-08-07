#!/usr/bin/env python
"""Summarize medium-background supervision diagnostics and gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CHANNELS = ("r", "g", "b")
RUN_LAMBDAS = {
    "BG000": 0.00,
    "BG010": 0.01,
}


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


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


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf8"))
    data["_summary_path"] = str(path)
    return data


def _load_summaries(root: Path, steps: Sequence[int]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for run in RUN_LAMBDAS:
        out[run] = {}
        for step in steps:
            out[run][int(step)] = {
                "eval": _load(root / run / f"step_{int(step)}" / "eval" / "summary.json"),
                "train_background": _load(root / run / f"step_{int(step)}" / "train_background" / "summary.json"),
            }
    return out


def _background(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    bg = summary.get("background_supervision", {})
    if not bg.get("available", False):
        raise RuntimeError(f"Missing background supervision stats in {summary.get('_summary_path', '<unknown>')}")
    return bg


def _summarize_run_step(
    run: str,
    step: int,
    summaries: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    eval_summary = summaries["eval"]
    train_bg_summary = summaries["train_background"]
    base_eval = baseline["eval"]
    base_train_bg = baseline["train_background"]

    psnr = _metric(eval_summary, "psnr")
    ssim = _metric(eval_summary, "ssim")
    lpips = _metric(eval_summary, "lpips")
    base_psnr = _metric(base_eval, "psnr")
    base_ssim = _metric(base_eval, "ssim")
    base_lpips = _metric(base_eval, "lpips")
    dpsnr = psnr - base_psnr
    dssim = ssim - base_ssim
    dlpips = lpips - base_lpips
    rgb_safe = bool(dpsnr >= -0.15 and dssim >= -0.0015 and dlpips <= 0.003)

    bg = _background(train_bg_summary)
    base_bg = _background(base_train_bg)
    background_l1 = float(bg.get("background_medium_l1", 0.0))
    base_background_l1 = float(base_bg.get("background_medium_l1", 0.0))
    weighted_background_l1 = float(bg.get("weighted_background_medium_l1", 0.0))
    base_weighted_background_l1 = float(base_bg.get("weighted_background_medium_l1", 0.0))
    background_l1_drop = _safe_relative_drop(background_l1, base_background_l1)
    weighted_background_l1_drop = _safe_relative_drop(weighted_background_l1, base_weighted_background_l1)

    tau_p90 = _rgb_stat_mean(eval_summary, "tau_D_effective", "p90")
    base_tau_p90 = _rgb_stat_mean(base_eval, "tau_D_effective", "p90")
    j_gt_1 = _threshold_mean(eval_summary, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
    base_j_gt_1 = _threshold_mean(base_eval, "clear_object_fullsh_raw_thresholds", "P(J>1.0)")
    tau_p90_drop = _safe_relative_drop(tau_p90, base_tau_p90)
    j_gt_1_drop = _safe_relative_drop(j_gt_1, base_j_gt_1)
    residual_pass = bool(background_l1_drop >= 0.20)
    proxy_pass = bool(tau_p90_drop >= 0.10 or j_gt_1_drop >= 0.15)

    return {
        "run": run,
        "checkpoint_step": int(step),
        "lambda_bg": float(RUN_LAMBDAS[run]),
        "eval_summary_path": eval_summary["_summary_path"],
        "train_background_summary_path": train_bg_summary["_summary_path"],
        "eval_checkpoint": eval_summary.get("checkpoint", ""),
        "train_background_checkpoint": train_bg_summary.get("checkpoint", ""),
        "eval_view_count": int(eval_summary["aggregate"].get("view_count", 0)),
        "train_background_view_count": int(train_bg_summary["aggregate"].get("view_count", 0)),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "delta_psnr_vs_BG000": dpsnr,
        "delta_ssim_vs_BG000": dssim,
        "delta_lpips_vs_BG000": dlpips,
        "rgb_safety_pass": rgb_safe,
        "background_medium_l1": background_l1,
        "weighted_background_medium_l1": weighted_background_l1,
        "background_medium_l1_drop_vs_BG000": background_l1_drop,
        "weighted_background_medium_l1_drop_vs_BG000": weighted_background_l1_drop,
        "background_supervision_effective_pass": residual_pass,
        "tau_D_effective_p90_mean_rgb": tau_p90,
        "tau_D_effective_p90_drop_vs_BG000": tau_p90_drop,
        "J_gt_1_mean_rgb": j_gt_1,
        "J_gt_1_drop_vs_BG000": j_gt_1_drop,
        "decomposition_proxy_pass": proxy_pass,
        "B_gate_pass": bool(rgb_safe and residual_pass and proxy_pass),
        "background_mask_coverage": bg.get("coverage", {}),
        "background_medium_rgb_mean_by_channel": bg.get("medium_rgb_mean", {}),
        "background_medium_bs_mean_by_channel": bg.get("medium_bs_mean", {}),
        "background_medium_attn_raw_mean_by_channel": bg.get("medium_attn_raw_mean", {}),
        "background_medium_attn_effective_mean_by_channel": bg.get("medium_attn_effective_mean", {}),
        "beta_D_raw_mean_by_channel": _rgb_stat(eval_summary, "beta_D_raw", "mean"),
        "beta_D_effective_mean_by_channel": _rgb_stat(eval_summary, "beta_D_effective", "mean"),
        "tau_D_effective_p90_by_channel": _rgb_stat(eval_summary, "tau_D_effective", "p90"),
        "J_gt_1_by_channel": {
            channel: float(eval_summary["aggregate"]["clear_object_fullsh_raw_thresholds"][channel]["P(J>1.0)"])
            for channel in CHANNELS
        },
    }


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    steps = [int(step) for step in args.steps.split(",") if step.strip()]
    summaries = _load_summaries(args.input_root, steps)
    rows: List[Dict[str, Any]] = []
    for step in steps:
        baseline = summaries["BG000"][step]
        for run in RUN_LAMBDAS:
            rows.append(_summarize_run_step(run, step, summaries[run][step], baseline))

    final_step = int(args.final_step)
    gate_rows = [row for row in rows if int(row["checkpoint_step"]) == final_step]
    pass_rows = [row for row in gate_rows if row["B_gate_pass"]]
    result = {
        "experiment": "medium_background_direct_supervision",
        "scene": args.scene,
        "input_root": str(args.input_root),
        "steps": steps,
        "final_step": final_step,
        "definitions": {
            "L_bg": "sum(M * abs(medium_rgb - GT) / (medium_rgb.detach() + 1e-3)) / sum(M)",
            "M": "fixed detached renderer-derived background-water mask loaded on train cameras",
            "RGB_safety": "eval PSNR/SSIM/LPIPS compared to matched-step BG000",
            "B_gate": "RGB safe, background_medium_l1 drop >=20%, and tau p90 drop >=10% or P(J>1) drop >=15%",
        },
        "rows": rows,
        "final_gate_rows": gate_rows,
        "B_pass_runs": [row["run"] for row in pass_rows],
        "B_gate_pass": bool(pass_rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    _write_csv(args.output_csv, rows)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "run",
        "checkpoint_step",
        "lambda_bg",
        "eval_view_count",
        "train_background_view_count",
        "psnr",
        "ssim",
        "lpips",
        "delta_psnr_vs_BG000",
        "delta_ssim_vs_BG000",
        "delta_lpips_vs_BG000",
        "rgb_safety_pass",
        "background_medium_l1",
        "weighted_background_medium_l1",
        "background_medium_l1_drop_vs_BG000",
        "weighted_background_medium_l1_drop_vs_BG000",
        "background_supervision_effective_pass",
        "tau_D_effective_p90_mean_rgb",
        "tau_D_effective_p90_drop_vs_BG000",
        "J_gt_1_mean_rgb",
        "J_gt_1_drop_vs_BG000",
        "decomposition_proxy_pass",
        "B_gate_pass",
        "eval_summary_path",
        "train_background_summary_path",
        "eval_checkpoint",
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
    parser.add_argument("--input-root", type=Path, default=Path("renders/dewater_optical_depth_20260807/B"))
    parser.add_argument("--steps", default="11000,12000,13000")
    parser.add_argument("--final-step", type=int, default=13000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/dewater_optical_depth_20260807/medium_background_supervision_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/dewater_optical_depth_20260807/medium_background_supervision_summary.csv"),
    )
    args = parser.parse_args()
    result = summarize(args)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "scene": result["scene"],
                "final_step": result["final_step"],
                "B_pass_runs": result["B_pass_runs"],
                "B_gate_pass": result["B_gate_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
