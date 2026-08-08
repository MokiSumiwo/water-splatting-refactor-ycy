#!/usr/bin/env python
"""Summarize fixed BND-SCRATCH cross-scene diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image


CHANNELS = ("r", "g", "b")
FINAL_STEP = 15000


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _load(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf8"))
    data["_summary_path"] = str(path)
    return data


def _parse_summary_item(text: str) -> Tuple[str, str, int, Path]:
    parts = text.split(":", 3)
    if len(parts) != 4:
        raise ValueError(f"--summary must be SCENE:RUN:STEP:PATH, got {text}")
    return parts[0], parts[1], int(parts[2]), Path(parts[3])


def _parse_config_item(text: str) -> Tuple[str, Path]:
    parts = text.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"config item must be SCENE:PATH, got {text}")
    return parts[0], Path(parts[1])


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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


def _png_mean_abs_diff(summary_a: Mapping[str, Any], summary_b: Mapping[str, Any], component: str) -> float:
    rows_a = {int(row["view_id"]): row for row in summary_a.get("per_view", [])}
    rows_b = {int(row["view_id"]): row for row in summary_b.get("per_view", [])}
    diffs = []
    for view_id, row_a in rows_a.items():
        row_b = rows_b.get(view_id)
        if row_b is None:
            continue
        path_a = Path(row_a.get("files", {}).get(component, ""))
        path_b = Path(row_b.get("files", {}).get(component, ""))
        if not path_a.exists() or not path_b.exists():
            continue
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        if img_a.size != img_b.size:
            continue
        pixels_a = list(img_a.getdata())
        pixels_b = list(img_b.getdata())
        total = 0.0
        count = 0
        for pa, pb in zip(pixels_a, pixels_b):
            total += sum(abs(float(a) - float(b)) for a, b in zip(pa, pb)) / (3.0 * 255.0)
            count += 1
        if count:
            diffs.append(total / count)
    return _mean(diffs)


def _row(scene: str, run: str, step: int, summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "scene": scene,
        "run": run,
        "step": int(step),
        "summary_path": summary.get("_summary_path", ""),
        "load_config": summary.get("load_config", ""),
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
        "J_max": _channel_stat(summary, "clear_object_fullsh_raw", "max"),
        "P(J>0.95)": _threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>0.95)"),
        "P(J>0.99)": _threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>0.99)"),
        "P(J>1)": _threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>1.0)", "P(gt1)"),
        "P(J>1.5)": _threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>1.5)", "P(gt1.5)"),
        "P(J>2)": _threshold(summary, "clear_object_fullsh_raw_thresholds", "P(J>2.0)", "P(gt2)"),
        "c_p95": _channel_stat(summary, "gaussian_view_rgb", "p95"),
        "c_p99": _channel_stat(summary, "gaussian_view_rgb", "p99"),
        "c_max": _channel_stat(summary, "gaussian_view_rgb", "max"),
        "P(c>0.95)": _threshold(summary, "gaussian_view_rgb_thresholds", "P(c>0.95)"),
        "P(c>0.99)": _threshold(summary, "gaussian_view_rgb_thresholds", "P(c>0.99)"),
        "P(c>1)": _threshold(summary, "gaussian_view_rgb_thresholds", "P(c>1.0)"),
        "P(c>1.5)": _threshold(summary, "gaussian_view_rgb_thresholds", "P(c>1.5)"),
        "P(c>2)": _threshold(summary, "gaussian_view_rgb_thresholds", "P(c>2.0)"),
        "P(c<0.01)": _threshold(summary, "gaussian_view_rgb_thresholds", "P(c<0.01)"),
        "P(c<0.05)": _threshold(summary, "gaussian_view_rgb_thresholds", "P(c<0.05)"),
        "SATURATION_MASS_001": _threshold(summary, "gaussian_view_rgb_thresholds", "SATURATION_MASS_001"),
        "logit_p99": _channel_stat(summary, "gaussian_view_logits", "p99"),
        "logit_max": _channel_stat(summary, "gaussian_view_logits", "max"),
        "P(s>4.595)": _threshold(summary, "gaussian_view_logits_thresholds", "P(s>4.595)"),
        "P(s<-4.595)": _threshold(summary, "gaussian_view_logits_thresholds", "P(s<-4.595)"),
        "P(|s|>5)": _threshold(summary, "gaussian_view_logits_thresholds", "P(|s|>5)"),
        "P(|s|>8)": _threshold(summary, "gaussian_view_logits_thresholds", "P(|s|>8)"),
        "P(|s|>10)": _threshold(summary, "gaussian_view_logits_thresholds", "P(|s|>10)"),
        "sigmoid_derivative_mean": _channel_stat(summary, "gaussian_sigmoid_derivative", "mean"),
        "P(sigmoid_derivative<0.01)": _threshold(
            summary, "gaussian_sigmoid_derivative_thresholds", "P(sigmoid_derivative<0.01)"
        ),
        "direct_object_signal_mean": _channel_stat(summary, "direct_object_signal", "mean"),
        "direct_object_signal_p50": _channel_stat(summary, "direct_object_signal", "p50"),
        "direct_object_signal_p90": _channel_stat(summary, "direct_object_signal", "p90"),
        "beta_B": _channel_stat(summary, "medium_bs", "mean"),
        "beta_B_p50": _channel_stat(summary, "medium_bs", "p50"),
        "beta_B_p90": _channel_stat(summary, "medium_bs", "p90"),
        "medium_rgb_mean": _channel_stat(summary, "medium_rgb", "mean"),
        "medium_rgb_p50": _channel_stat(summary, "medium_rgb", "p50"),
        "medium_rgb_p90": _channel_stat(summary, "medium_rgb", "p90"),
        "B_inf_mean": _channel_stat(summary, "b_inf", "mean"),
        "backscatter_mean": _channel_stat(summary, "backscatter", "mean"),
        "backscatter_p50": _channel_stat(summary, "backscatter", "p50"),
        "backscatter_p90": _channel_stat(summary, "backscatter", "p90"),
        "Gaussian count": int(summary.get("model_state", {}).get("gaussian_count", 0)),
        "view_ids": ";".join(str(row.get("view_id")) for row in summary.get("per_view", [])),
        "camera_ids": ";".join(str(row.get("camera_id")) for row in summary.get("per_view", [])),
        "num_eval_views": len(summary.get("per_view", [])),
    }


def _delta(base: Mapping[str, Any], cand: Mapping[str, Any], metric: str) -> float:
    return float(cand.get(metric, 0.0)) - float(base.get(metric, 0.0))


def _rel_change(base: float, cand: float) -> float:
    if abs(float(base)) <= 1e-12:
        return 0.0
    return float(cand) / float(base) - 1.0


def _rel_drop(base: float, cand: float) -> float:
    return -_rel_change(base, cand)


def _rgb_safe(base: Mapping[str, Any], cand: Mapping[str, Any]) -> bool:
    return (
        _delta(base, cand, "PSNR") >= -0.15
        and _delta(base, cand, "SSIM") >= -0.0015
        and _delta(base, cand, "LPIPS") <= 0.003
    )


def _scene_delta(scene: str, base: Mapping[str, Any], cand: Mapping[str, Any], summaries: Mapping[Tuple[str, str, int], Mapping[str, Any]]) -> Dict[str, Any]:
    tau_drop = _rel_drop(float(base["tau_p90"]), float(cand["tau_p90"]))
    t01_drop = _rel_drop(float(base["P(T<0.1)"]), float(cand["P(T<0.1)"]))
    j_drop = _rel_drop(float(base["J_p99"]), float(cand["J_p99"]))
    low_tau_baseline = float(base["tau_p90"]) < 1.0 and float(base["P(T<0.1)"]) < 0.02
    rgb_safe = _rgb_safe(base, cand)
    boundary_escape = float(cand["P(c>0.99)"]) > 0.05 or float(cand["P(|s|>5)"]) > 0.05
    decomp_pass = (
        (tau_drop >= 0.15 and (float(cand["P(T<0.1)"]) <= 0.75 * float(base["P(T<0.1)"]) or j_drop >= 0.15))
        or (low_tau_baseline and j_drop >= 0.15)
    )
    bnd_summary = summaries.get((scene, "BND", FINAL_STEP))
    m1_summary = summaries.get((scene, "M1", FINAL_STEP))
    direct_display_l1 = _png_mean_abs_diff(m1_summary, bnd_summary, "direct_object_signal") if m1_summary and bnd_summary else 0.0
    backscatter_ratio = _rel_change(float(base["backscatter_mean"]), float(cand["backscatter_mean"]))
    beta_b_ratio = _rel_change(float(base["beta_B"]), float(cand["beta_B"]))
    medium_ratio = _rel_change(float(base["medium_rgb_mean"]), float(cand["medium_rgb_mean"]))
    redistributed = tau_drop > 0.15 and max(backscatter_ratio, beta_b_ratio, medium_ratio) > 0.50
    return {
        "scene": scene,
        "base_run": base["run"],
        "candidate_run": cand["run"],
        "Delta PSNR": _delta(base, cand, "PSNR"),
        "Delta SSIM": _delta(base, cand, "SSIM"),
        "Delta LPIPS": _delta(base, cand, "LPIPS"),
        "tau_p90_relative_change_BND_over_M1_minus_1": _rel_change(float(base["tau_p90"]), float(cand["tau_p90"])),
        "tau_p90_relative_reduction": tau_drop,
        "P(T<0.1)_relative_change_BND_over_M1_minus_1": _rel_change(float(base["P(T<0.1)"]), float(cand["P(T<0.1)"])),
        "P(T<0.1)_relative_reduction": t01_drop,
        "J_p99_relative_change_BND_over_M1_minus_1": _rel_change(float(base["J_p99"]), float(cand["J_p99"])),
        "J_p99_relative_reduction": j_drop,
        "P(J>1)_absolute_change": _delta(base, cand, "P(J>1)"),
        "Gaussian_count_change": int(cand["Gaussian count"]) - int(base["Gaussian count"]),
        "beta_B_relative_change": beta_b_ratio,
        "medium_rgb_relative_change": medium_ratio,
        "backscatter_relative_change": backscatter_ratio,
        "direct_object_signal_display_mean_abs_difference": direct_display_l1,
        "LOW_TAU_BASELINE": bool(low_tau_baseline),
        "RGB_SAFETY_PASS": bool(rgb_safe),
        "DECOMPOSITION_IMPROVEMENT_PASS": bool(decomp_pass),
        "BOUNDARY_ESCAPE": bool(boundary_escape),
        "COMPENSATION_REDISTRIBUTED_TO_BACKSCATTER": bool(redistributed),
        "SCENE_BND_PASS": bool(rgb_safe and decomp_pass and not boundary_escape),
    }


def _config_value(text: str, key: str, default: str = "") -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*(.*)$", text)
    return match.group(1).strip() if match else default


def _config_audit(scene: str, baseline_config: Optional[Path], bnd_config: Optional[Path]) -> Dict[str, Any]:
    def values(path: Optional[Path], intrinsic_default: str) -> Dict[str, Any]:
        if path is None or not path.exists():
            return {"path": str(path) if path else "", "exists": False}
        text = path.read_text(encoding="utf8", errors="ignore")
        return {
            "path": str(path),
            "exists": True,
            "seed": _config_value(text, "seed"),
            "max_num_iterations": _config_value(text, "max_num_iterations"),
            "steps_per_save": _config_value(text, "steps_per_save"),
            "sh_degree": _config_value(text, "sh_degree"),
            "medium_context_mode": _config_value(text, "medium_context_mode"),
            "b_inf_mode": _config_value(text, "b_inf_mode"),
            "infinite_water_enabled": _config_value(text, "infinite_water_enabled"),
            "direct_optical_depth_scale": _config_value(text, "direct_optical_depth_scale", "1.0"),
            "intrinsic_color_parameterization": _config_value(text, "intrinsic_color_parameterization", intrinsic_default),
            "disable_population_refinement": _config_value(text, "disable_population_refinement", "false"),
            "gmvc_enabled": _config_value(text, "gmvc_enabled", "false"),
            "train_split_fraction": _config_value(text, "train_split_fraction"),
            "eval_mode": _config_value(text, "eval_mode"),
            "downscale_factor": _config_value(text, "downscale_factor"),
            "densify_grad_thresh": _config_value(text, "densify_grad_thresh"),
            "refine_every": _config_value(text, "refine_every"),
            "reset_alpha_every": _config_value(text, "reset_alpha_every"),
        }

    base = values(baseline_config, "legacy")
    bnd = values(bnd_config, "sigmoid_sh")
    matched = bool(
        base.get("exists")
        and bnd.get("exists")
        and base.get("seed") == "42"
        and bnd.get("seed") == "42"
        and base.get("max_num_iterations") == "15000"
        and bnd.get("max_num_iterations") == "15000"
        and base.get("sh_degree") == "3"
        and bnd.get("sh_degree") == "3"
        and base.get("medium_context_mode") == "dir_xy_camera"
        and bnd.get("medium_context_mode") == "dir_xy_camera"
        and base.get("b_inf_mode") == "tied"
        and bnd.get("b_inf_mode") == "tied"
        and base.get("infinite_water_enabled") in ("false", "False")
        and bnd.get("infinite_water_enabled") in ("false", "False")
        and float(base.get("direct_optical_depth_scale", "1.0")) == 1.0
        and float(bnd.get("direct_optical_depth_scale", "1.0")) == 1.0
        and base.get("intrinsic_color_parameterization") == "legacy"
        and bnd.get("intrinsic_color_parameterization") == "sigmoid_sh"
    )
    return {"scene": scene, "baseline": base, "bnd": bnd, "matched": matched}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _flatten_audit(audits: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for audit in audits:
        row = {"scene": audit["scene"], "matched": audit["matched"]}
        for prefix in ("baseline", "bnd"):
            for key, value in audit[prefix].items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    summaries: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for item in args.summary:
        scene, run_name, step, path = _parse_summary_item(item)
        summaries[(scene, run_name, step)] = _load(path)
    rows = [
        _row(scene, run_name, step, summary)
        for (scene, run_name, step), summary in sorted(summaries.items(), key=lambda item: (item[0][0], item[0][2], item[0][1]))
    ]
    by_key = {(row["scene"], row["run"], int(row["step"])): row for row in rows}
    scenes = sorted({row["scene"] for row in rows})
    final_rows = [row for row in rows if int(row["step"]) == FINAL_STEP]
    delta_rows = []
    for scene in scenes:
        base = by_key.get((scene, "M1", FINAL_STEP))
        cand = by_key.get((scene, "BND", FINAL_STEP))
        if base and cand:
            delta_rows.append(_scene_delta(scene, base, cand, summaries))

    baseline_configs = dict(_parse_config_item(item) for item in args.baseline_config)
    bnd_configs = dict(_parse_config_item(item) for item in args.bnd_config)
    audits = [_config_audit(scene, baseline_configs.get(scene), bnd_configs.get(scene)) for scene in scenes]
    new_scenes = [scene for scene in scenes if scene != "Curasao"]
    new_deltas = [row for row in delta_rows if row["scene"] in new_scenes]
    pass_count = sum(1 for row in new_deltas if row["SCENE_BND_PASS"])
    rgb_fail_count = sum(1 for row in new_deltas if not row["RGB_SAFETY_PASS"])
    boundary_count = sum(1 for row in delta_rows if row["BOUNDARY_ESCAPE"])
    no_decomp_count = sum(1 for row in new_deltas if not row["DECOMPOSITION_IMPROVEMENT_PASS"])
    third_ok = all(row["RGB_SAFETY_PASS"] and not row["BOUNDARY_ESCAPE"] for row in new_deltas)
    classifications = {
        "CROSS_SCENE_BND_FULL": bool(len(new_deltas) == 3 and pass_count == 3),
        "CROSS_SCENE_BND_STRONG": bool(pass_count >= 2 and third_ok and any(row["scene"] == "Curasao" and row["SCENE_BND_PASS"] for row in delta_rows)),
        "RGB_SAFE_BUT_SCENE_DEPENDENT": bool(
            sum(1 for row in new_deltas if row["RGB_SAFETY_PASS"]) >= 2 and pass_count < len(new_deltas)
        ),
        "BOUNDARY_ESCAPE_CROSS_SCENE": bool(boundary_count >= 2),
        "BND_CROSS_SCENE_FAILURE": bool(rgb_fail_count >= 2 or no_decomp_count >= 2),
    }
    payload = {
        "diagnostic": "bounded_sh3_cross_scene_summary",
        "start_head": args.start_head,
        "water_splatting_commit": _git_commit(repo),
        "seafree_reference_commit": args.seafree_reference_commit,
        "scenes": scenes,
        "trajectory": rows,
        "final_metrics": final_rows,
        "deltas": delta_rows,
        "baseline_audit": audits,
        "classifications": classifications,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_scene_bnd_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf8")
    (args.output_dir / "cross_scene_bnd_trajectory.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    (args.output_dir / "cross_scene_bnd_final_metrics.json").write_text(json.dumps(final_rows, indent=2), encoding="utf8")
    (args.output_dir / "cross_scene_bnd_baseline_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf8")
    (args.output_dir / "cross_scene_bnd_saturation.json").write_text(
        json.dumps(
            [
                {
                    "scene": row["scene"],
                    "run": row["run"],
                    "step": row["step"],
                    "c_p99": row["c_p99"],
                    "P(c>0.99)": row["P(c>0.99)"],
                    "P(c<0.01)": row["P(c<0.01)"],
                    "SATURATION_MASS_001": row["SATURATION_MASS_001"],
                    "logit_p99": row["logit_p99"],
                    "P(|s|>5)": row["P(|s|>5)"],
                    "P(|s|>8)": row["P(|s|>8)"],
                }
                for row in rows
            ],
            indent=2,
        ),
        encoding="utf8",
    )
    (args.output_dir / "cross_scene_bnd_medium_redistribution.json").write_text(
        json.dumps(
            [
                {
                    "scene": row["scene"],
                    "beta_B_relative_change": row["beta_B_relative_change"],
                    "medium_rgb_relative_change": row["medium_rgb_relative_change"],
                    "backscatter_relative_change": row["backscatter_relative_change"],
                    "COMPENSATION_REDISTRIBUTED_TO_BACKSCATTER": row["COMPENSATION_REDISTRIBUTED_TO_BACKSCATTER"],
                }
                for row in delta_rows
            ],
            indent=2,
        ),
        encoding="utf8",
    )
    _write_csv(args.output_dir / "cross_scene_bnd_summary.csv", delta_rows)
    _write_csv(args.output_dir / "cross_scene_bnd_trajectory.csv", rows)
    _write_csv(args.output_dir / "cross_scene_bnd_final_metrics.csv", final_rows)
    _write_csv(args.output_dir / "cross_scene_bnd_baseline_audit.csv", _flatten_audit(audits))
    _write_csv(
        args.output_dir / "cross_scene_bnd_saturation.csv",
        [
            {
                "scene": row["scene"],
                "run": row["run"],
                "step": row["step"],
                "c_p99": row["c_p99"],
                "P(c>0.99)": row["P(c>0.99)"],
                "P(c<0.01)": row["P(c<0.01)"],
                "SATURATION_MASS_001": row["SATURATION_MASS_001"],
                "logit_p99": row["logit_p99"],
                "P(|s|>5)": row["P(|s|>5)"],
                "P(|s|>8)": row["P(|s|>8)"],
            }
            for row in rows
        ],
    )
    _write_csv(
        args.output_dir / "cross_scene_bnd_medium_redistribution.csv",
        [
            {
                "scene": row["scene"],
                "beta_B_relative_change": row["beta_B_relative_change"],
                "medium_rgb_relative_change": row["medium_rgb_relative_change"],
                "backscatter_relative_change": row["backscatter_relative_change"],
                "COMPENSATION_REDISTRIBUTED_TO_BACKSCATTER": row["COMPENSATION_REDISTRIBUTED_TO_BACKSCATTER"],
            }
            for row in delta_rows
        ],
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, help="SCENE:RUN:STEP:path/to/summary.json")
    parser.add_argument("--baseline-config", action="append", default=[], help="SCENE:path/to/M1/config.yml")
    parser.add_argument("--bnd-config", action="append", default=[], help="SCENE:path/to/BND/config.yml")
    parser.add_argument("--start-head", default="")
    parser.add_argument("--seafree-reference-commit", default="7797e97dae831029ac89ae9f37b3c3d69ec2cf6c")
    parser.add_argument("--output-dir", type=Path, default=Path("renders/dewater_bounded_sh3_cross_scene_20260808/four_scene_summary"))
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"output_dir": str(args.output_dir), "scenes": payload["scenes"], "classifications": payload["classifications"]}, indent=2))


if __name__ == "__main__":
    main()
