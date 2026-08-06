#!/usr/bin/env python
"""Summarize four-scene GMVC P30-MHOLD validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RGB_KEYS = ("psnr", "ssim", "lpips", "rgb_l1", "luminance_l1", "chroma_l1")
FIXED_KEYS = (
    "transfer_l1",
    "object_j_variance",
    "closure_signal_floor_l1",
    "consensus_j_reconstruction_l1",
    "object_target_l1",
    "dc_cross_view_variance",
    "dc_recomposition_l1",
)
HOLD_ZERO_KEYS = (
    "rgb_grad_norm_medium",
    "gmvc_profile_grad_norm_medium",
    "gmvc_medium_param_delta_mean_abs",
    "gmvc_medium_param_delta_max_abs",
    "gmvc_medium_attn_delta_l1_mean",
    "gmvc_medium_attn_delta_l1_p95",
    "gmvc_medium_bs_delta_l1_mean",
    "gmvc_medium_bs_delta_l1_p95",
    "gmvc_b_inf_delta_l1_mean",
    "gmvc_b_inf_delta_l1_p95",
    "gmvc_transmission_delta_l1_mean",
    "gmvc_transmission_delta_l1_p95",
)


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _relative_change(new: float, old: float) -> float:
    if abs(old) < 1e-12:
        return 0.0
    return (new - old) / old


def _parse_assignment(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SCENE=PATH")
    key, path = value.split("=", 1)
    return key.strip(), Path(path)


def _summarize_hold_log(path: Path, start_step: int, stop_step: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if not path or not path.exists():
        return {"path": str(path), "exists": False, "row_count": 0, "pass": False}
    for line in path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = int(_safe_float(row.get("global_step", row.get("step", -1)), -1))
        if start_step <= step <= stop_step:
            rows.append(row)
    maxima = {
        key: max((_safe_float(row.get(key, 0.0)) for row in rows), default=0.0)
        for key in HOLD_ZERO_KEYS
    }
    profile_effective_max = max((_safe_float(row.get("gmvc_profile_lambda_effective", 0.0)) for row in rows), default=0.0)
    active_fraction = (
        sum(1 for row in rows if bool(row.get("gmvc_medium_hold_active", False))) / max(len(rows), 1)
    )
    pass_flag = bool(rows) and profile_effective_max == 0.0 and active_fraction == 1.0 and all(value == 0.0 for value in maxima.values())
    return {
        "path": str(path),
        "exists": True,
        "row_count": len(rows),
        "start_step": int(start_step),
        "stop_step": int(stop_step),
        "profile_effective_max": profile_effective_max,
        "active_fraction": active_fraction,
        "maxima": maxima,
        "pass": pass_flag,
    }


def _rgb_summary(scene_dir: Path) -> Dict[str, Any]:
    summary_path = scene_dir / "visualization" / "visualization_summary.json"
    if not summary_path.exists():
        summary_path = scene_dir / "visualization_summary.json"
    if not summary_path.exists():
        return {"exists": False, "path": str(summary_path)}
    data = _load_json(summary_path)
    runs = data.get("runs", {})
    a0 = runs.get("A0", {}).get("mean", {})
    p30 = runs.get("P30-MHOLD", {}).get("mean", {})
    delta = data.get("mean_delta_vs_a0", {}).get("P30-MHOLD", {})
    gate = {
        "psnr": _safe_float(delta.get("psnr", -999.0)) >= -0.15,
        "ssim": _safe_float(delta.get("ssim", -999.0)) >= -0.0015,
        "lpips": _safe_float(delta.get("lpips", 999.0)) <= 0.003,
    }
    return {
        "exists": True,
        "path": str(summary_path),
        "a0": {key: _safe_float(a0.get(key, 0.0)) for key in RGB_KEYS},
        "p30_mhold": {key: _safe_float(p30.get(key, 0.0)) for key in RGB_KEYS},
        "delta": {key: _safe_float(delta.get(key, 0.0)) for key in RGB_KEYS},
        "gate": gate,
        "gate_pass": all(gate.values()),
        "runs": runs,
        "per_view": data.get("per_view", []),
        "contact_sheets": data.get("contact_sheets", {}),
        "selected_contact_views": data.get("selected_contact_views", []),
        "dewatered_definition": data.get("dewatered_definition", ""),
    }


def _fixed_bank_result(scene_dir: Path, bank: str, run: str) -> Dict[str, Any]:
    path = scene_dir / "fixed_bank" / bank / run / "gmvc_fixed_bank_metrics.json"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    data = _load_json(path)
    metrics = data.get("metrics", {}).get("heldout", {})
    return {
        "exists": True,
        "path": str(path),
        "checkpoint": data.get("checkpoint", ""),
        "step": data.get("step", None),
        "track_bank": data.get("track_bank", ""),
        "metrics": {key: _safe_float(metrics.get(key, 0.0)) for key in FIXED_KEYS},
        "selected": data.get("selected", {}),
    }


def _fixed_summary(scene_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for bank in ("evalf", "evalg"):
        a0 = _fixed_bank_result(scene_dir, bank, "a0")
        p30 = _fixed_bank_result(scene_dir, bank, "p30_mhold")
        if a0.get("exists") and p30.get("exists"):
            delta = {
                key: _safe_float(p30["metrics"].get(key, 0.0)) - _safe_float(a0["metrics"].get(key, 0.0))
                for key in FIXED_KEYS
            }
            relative = {
                key: _relative_change(_safe_float(p30["metrics"].get(key, 0.0)), _safe_float(a0["metrics"].get(key, 0.0)))
                for key in FIXED_KEYS
            }
        else:
            delta = {}
            relative = {}
        out[bank] = {"a0": a0, "p30_mhold": p30, "delta": delta, "relative_change": relative}
    return out


def _per_view_extremes(per_view: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(per_view)
    out: Dict[str, Any] = {"view_count": len(rows)}
    if not rows:
        return out
    for key in ("psnr", "ssim", "lpips", "rgb_l1", "luminance_l1", "chroma_l1"):
        values = [
            (row.get("view_index"), _safe_float(row.get("delta_vs_a0", {}).get("P30-MHOLD", {}).get(key, 0.0)))
            for row in rows
        ]
        out[f"{key}_min_delta"] = {"view_index": min(values, key=lambda item: item[1])[0], "delta": min(value for _, value in values)}
        out[f"{key}_max_delta"] = {"view_index": max(values, key=lambda item: item[1])[0], "delta": max(value for _, value in values)}
    return out


def _write_rgb_csv(path: Path, scenes: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "run", *RGB_KEYS, "delta_psnr", "delta_ssim", "delta_lpips", "gate_pass"])
        for scene, data in scenes.items():
            rgb = data.get("rgb", {})
            if not rgb.get("exists"):
                continue
            for run_key, run_name in (("a0", "A0"), ("p30_mhold", "P30-MHOLD")):
                values = rgb.get(run_key, {})
                delta = rgb.get("delta", {}) if run_key == "p30_mhold" else {}
                writer.writerow(
                    [
                        scene,
                        run_name,
                        *[values.get(key, 0.0) for key in RGB_KEYS],
                        delta.get("psnr", 0.0),
                        delta.get("ssim", 0.0),
                        delta.get("lpips", 0.0),
                        rgb.get("gate_pass", False) if run_key == "p30_mhold" else "",
                    ]
                )


def _write_fixed_csv(path: Path, scenes: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "eval", "metric", "a0", "p30_mhold", "delta", "relative_change"])
        for scene, data in scenes.items():
            fixed = data.get("fixed_bank", {})
            for bank in ("evalf", "evalg"):
                item = fixed.get(bank, {})
                if not (item.get("a0", {}).get("exists") and item.get("p30_mhold", {}).get("exists")):
                    continue
                for metric in FIXED_KEYS:
                    writer.writerow(
                        [
                            scene,
                            bank,
                            metric,
                            item["a0"]["metrics"].get(metric, 0.0),
                            item["p30_mhold"]["metrics"].get(metric, 0.0),
                            item["delta"].get(metric, 0.0),
                            item["relative_change"].get(metric, 0.0),
                        ]
                    )


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    hold_logs = dict(_parse_assignment(value) for value in args.mhold_log)
    scenes: Dict[str, Any] = {}
    for scene in args.scene:
        scene_dir = args.root / scene
        rgb = _rgb_summary(scene_dir)
        scenes[scene] = {
            "scene_dir": str(scene_dir),
            "rgb": rgb,
            "fixed_bank": _fixed_summary(scene_dir),
            "medium_hold_audit": _summarize_hold_log(hold_logs.get(scene, Path("")), args.hold_start_step, args.hold_stop_step),
            "per_view_extremes": _per_view_extremes(rgb.get("per_view", [])) if rgb.get("exists") else {},
        }
    summary = {
        "diagnostic": "gmvc_four_scene_p30_mhold_15k_summary",
        "root": str(args.root),
        "scenes": scenes,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
    }
    args.root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_json or (args.root / "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
    _write_rgb_csv(args.root / "summary_rgb.csv", scenes)
    _write_fixed_csv(args.root / "summary_fixed_bank.csv", scenes)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("renders/gmvc_four_scene_p30_mhold_15k"))
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene directory name under root. Defaults to the four formal scenes.",
    )
    parser.add_argument("--mhold-log", action="append", default=[], help="Medium hold log as SCENE=PATH")
    parser.add_argument("--hold-start-step", type=int, default=13001)
    parser.add_argument("--hold-stop-step", type=int, default=15000)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if not args.scene:
        args.scene = ["Curasao", "JapaneseGradens", "IUI3", "Panama"]
    result = summarize(args)
    print(json.dumps({"output": str(args.output_json or (args.root / "summary.json")), "scenes": list(result["scenes"])}, indent=2))


if __name__ == "__main__":
    main()
