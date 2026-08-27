#!/usr/bin/env python3
"""Aggregate completed per-scene RAOC causal worker summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
SNAPSHOT_STEPS = (3000, 5000, 8000, 10000, 13000, 14999)
BRANCHES = ("C0", "C1")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return float("nan")
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).lower() == "true"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _write_table(root: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(root / "aggregate" / f"{name}.csv", rows)
    _write_json(root / "aggregate" / f"{name}.json", {"rows": list(rows)})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _protocol_audit(root: Path, scene: str) -> Dict[str, Any]:
    scene_dir = root / scene
    expected_steps = set(SNAPSHOT_STEPS)
    checkpoint_steps = {
        branch: {
            int(path.stem[len("step-") :])
            for path in (scene_dir / "checkpoints" / branch).glob("step-*.ckpt")
        }
        for branch in BRANCHES
    }
    start = _read_json(scene_dir / "start_state_equivalence.json") if (scene_dir / "start_state_equivalence.json").is_file() else {}
    step0 = _read_json(scene_dir / "step0_basis_equivalence.json") if (scene_dir / "step0_basis_equivalence.json").is_file() else {}
    camera = _read_json(scene_dir / "camera_sequence.json") if (scene_dir / "camera_sequence.json").is_file() else {}
    decomp = _read_csv(scene_dir / "decomposition_safety.csv")
    rescue = _read_csv(scene_dir / "capacity_rescue.csv")
    context = _read_csv(scene_dir / "context_utility.csv")
    weak = _read_csv(scene_dir / "weak_capacity.csv")

    context_types = {row.get("row_type", "") for row in context}
    diagnostic_populations = {
        row.get("population", "")
        for rows in (rescue, context, weak)
        for row in rows
        if row.get("population")
    }
    expected_populations = {"GENERAL", "M_SAFE"} if scene == "IUI3-RedSea" else {"GENERAL"}
    weak_ratio_fields = (
        "ocmc_kept_full_residual_ratio",
        "raoc_kept_full_residual_ratio",
        "ocmc_suppressed_full_residual_ratio",
    )
    weak_ratios_finite = bool(weak) and all(
        math.isfinite(_number(row.get(field)))
        for row in weak
        for field in weak_ratio_fields
    )
    same_state_rows = [
        row
        for row in rescue
        if row.get("branch") == "C1"
        and row.get("population") == "GENERAL"
        and row.get("granularity") == "overall"
    ]
    same_state_steps = {int(row["absolute_step"]) for row in same_state_rows if row.get("absolute_step")}
    same_state_counterfactual = bool(same_state_rows) and all(
        row.get("counterfactual") == "C1_same_state_OCMC" for row in same_state_rows
    )
    decomposition_steps = {
        (row.get("branch"), int(row["absolute_step"]), row.get("split"))
        for row in decomp
        if row.get("absolute_step")
    }
    expected_decomposition = {
        (branch, step, split)
        for branch in BRANCHES
        for step in SNAPSHOT_STEPS
        for split in ("train", "eval")
    }
    artifact_names = (
        "weak_capacity",
        "capacity_rescue",
        "evidence_selectivity",
        "context_utility",
        "decomposition_safety",
        "scene_summary",
        "scene_classification",
    )
    artifact_pairs_complete = all(
        (scene_dir / f"{name}.csv").is_file() and (scene_dir / f"{name}.json").is_file()
        if name not in ("scene_summary", "scene_classification")
        else (scene_dir / f"{name}.json").is_file()
        for name in artifact_names
    )
    checks = {
        "C0_checkpoint_steps_exact": checkpoint_steps["C0"] == expected_steps,
        "C1_checkpoint_steps_exact": checkpoint_steps["C1"] == expected_steps,
        "start_state_equivalence": _bool(start.get("START_STATE_EQUIVALENCE", False)),
        "step0_basis_equivalence": _bool(step0.get("STEP0_BASIS_EQUIVALENCE", False)),
        "camera_sequence_match": _bool(camera.get("CAMERA_SEQUENCE_MATCH", False))
        and int(camera.get("mismatch_count", -1)) == 0
        and int(camera.get("length", -1)) == 15000,
        "decomposition_rows_complete": decomposition_steps == expected_decomposition,
        "decomposition_safety": bool(decomp)
        and all(math.isfinite(_number(row.get("P_J_gt_1"))) and _number(row.get("P_J_gt_1")) == 0.0 for row in decomp),
        "same_state_OCMC_counterfactual_complete": same_state_counterfactual and same_state_steps == expected_steps,
        "context_row_types_complete": {"pair", "aggregate", "causal_delta"}.issubset(context_types),
        "weak_capacity_ratios_finite": weak_ratios_finite,
        "diagnostic_populations_exact": diagnostic_populations == expected_populations,
        "required_artifacts_complete": artifact_pairs_complete,
    }
    return {
        "scene": scene,
        **checks,
        "protocol_complete": all(checks.values()),
        "C0_checkpoint_steps": sorted(checkpoint_steps["C0"]),
        "C1_checkpoint_steps": sorted(checkpoint_steps["C1"]),
        "decomposition_row_count": len(decomp),
        "context_row_types": sorted(context_types),
        "diagnostic_populations": sorted(diagnostic_populations),
    }


def run(root: Path) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for scene in SCENES:
        path = root / scene / "scene_summary.json"
        if path.exists():
            try:
                summary = _read_json(path)
                summary["_scene_dir"] = str(path.parent)
                summaries.append(summary)
            except Exception as exc:
                failures.append({"scene": scene, "reason": f"invalid scene_summary.json: {exc!r}"})
        else:
            failure_files = sorted(str(path.name) for path in (root / scene).glob("*_failure.json"))
            failures.append({"scene": scene, "reason": "missing scene_summary.json", "failure_files": failure_files})

    metrics_rows: List[Dict[str, Any]] = []
    utility_rows: List[Dict[str, Any]] = []
    mechanism_rows: List[Dict[str, Any]] = []
    rgb_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    safety_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    protocol_rows: List[Dict[str, Any]] = []
    for summary in summaries:
        scene = summary["scene"]
        delta = summary["final_eval_delta_C1_minus_C0"]
        rgb_rows.append({"scene": scene, "PSNR_delta": delta["PSNR"], "SSIM_delta": delta["SSIM"], "LPIPS_delta": delta["LPIPS"], "MSE_delta": delta["MSE"], "PSNR_C0": summary["final_eval_C0"]["PSNR"], "PSNR_C1": summary["final_eval_C1"]["PSNR"], "classification": summary["classification"]})
        utility_rows.append({"scene": scene, "U_C0": summary["final_correct_context_utility_C0"], "U_C1": summary["final_correct_context_utility_C1"], "U_delta_C1_minus_C0": summary["final_correct_context_utility_delta_C1_minus_C0"]})
        mechanism_rows.append({"scene": scene, "classification": summary["classification"], "high_evidence_selectivity": summary["high_evidence_selectivity"], "low_evidence_selectivity": summary["low_evidence_selectivity"], "selective": _number(summary["high_evidence_selectivity"]) > _number(summary["low_evidence_selectivity"]), "over_rescue": summary["classification"] == "RAOC_OVER_RESCUE"})
        safety_rows.append({"scene": scene, "decomposition_safety": _bool(summary.get("decomposition_safety", False)), "start_state_equivalence": _bool(summary.get("START_STATE_EQUIVALENCE", False)), "camera_sequence_match": _bool(summary.get("CAMERA_SEQUENCE_MATCH", False))})
        runtime = summary.get("runtime", {})
        branches = runtime.get("branches", {})
        c0_runtime, c1_runtime = branches.get("C0", {}), branches.get("C1", {})
        runtime_rows.append({"scene": scene, "physical_gpu_id": summary.get("runtime", {}).get("assigned_physical_gpu", runtime.get("physical_gpu_id", "")), "torch_logical_gpu_id": runtime.get("torch_logical_gpu_id", 0), "torch_visible_gpu_count": runtime.get("torch_visible_gpu_count", 1), "C0_training_wall_seconds": c0_runtime.get("training_wall_seconds", ""), "C1_training_wall_seconds": c1_runtime.get("training_wall_seconds", ""), "RAOC_minus_OCMC_training_wall_seconds": _number(c1_runtime.get("training_wall_seconds")) - _number(c0_runtime.get("training_wall_seconds")), "C0_refresh_seconds": c0_runtime.get("refresh_seconds", ""), "C1_refresh_seconds": c1_runtime.get("refresh_seconds", ""), "C0_peak_reserved_bytes": c0_runtime.get("peak_reserved_bytes", ""), "C1_peak_reserved_bytes": c1_runtime.get("peak_reserved_bytes", "")})
        metrics_rows.append({"scene": scene, "U_delta": summary["final_correct_context_utility_delta_C1_minus_C0"], "PSNR_delta": delta["PSNR"], "SSIM_delta": delta["SSIM"], "LPIPS_delta": delta["LPIPS"], "MSE_delta": delta["MSE"], "classification": summary["classification"]})
        per_view_path = path = root / scene / "per_view_eval_delta.json"
        if per_view_path.exists():
            for row in _read_json(per_view_path).get("rows", []):
                per_view_rows.append({"scene": scene, **row})
        protocol_rows.append(_protocol_audit(root, scene))

    u_deltas = [_number(row["U_delta_C1_minus_C0"]) for row in utility_rows]
    psnr = [_number(row["PSNR_delta"]) for row in rgb_rows]
    ssim = [_number(row["SSIM_delta"]) for row in rgb_rows]
    lpips = [_number(row["LPIPS_delta"]) for row in rgb_rows]
    mse = [_number(row["MSE_delta"]) for row in rgb_rows]
    positive_utility = sum(value > 0 for value in u_deltas)
    supported_mechanism = sum(row["classification"] in ("RAOC_CAPACITY_REALLOCATION_SUPPORTED", "RAOC_MECHANISM_SUPPORTED_RGB_MIXED") for row in mechanism_rows)
    selective = sum(bool(row["selective"]) for row in mechanism_rows)
    causal_valid = bool(summaries) and len(summaries) == 4 and all(row["start_state_equivalence"] and row["camera_sequence_match"] for row in safety_rows) and not failures
    safety = bool(summaries) and all(row["decomposition_safety"] for row in safety_rows)
    over_rescue = sum(bool(row["over_rescue"]) for row in mechanism_rows)
    mean_psnr = _mean(psnr)
    positive_psnr = sum(value > 0 for value in psnr)
    supported_criteria = {"all_four_scenes_completed": len(summaries) == 4 and not failures, "causal_valid_all_completed_scenes": causal_valid, "positive_final_utility_at_least_3": positive_utility >= 3, "supported_or_rgb_mixed_at_least_3": supported_mechanism >= 3, "high_evidence_rescue_greater_than_low_at_least_3": selective >= 3, "no_more_than_one_over_rescue_scene": over_rescue <= 1, "decomposition_safety_all_nonfailed": safety, "mean_scene_level_PSNR_delta_nonnegative": math.isfinite(mean_psnr) and mean_psnr >= 0, "at_least_two_positive_PSNR_scenes": positive_psnr >= 2}
    if all(supported_criteria.values()):
        aggregate_classification = "RAOC_MULTI_SCENE_SUPPORTED"
    elif positive_utility >= 2 and supported_mechanism >= 2 and selective >= 2 and safety:
        aggregate_classification = "RAOC_MULTI_SCENE_TENTATIVE"
    else:
        aggregate_classification = "RAOC_MULTI_SCENE_NOT_SUPPORTED"
    rgb_classification = "RGB_MULTI_SCENE_IMPROVED" if positive_psnr >= 3 and _mean(ssim) >= 0 and _mean(lpips) <= 0 and _mean(mse) <= 0 else "RGB_MULTI_SCENE_DEGRADED" if positive_psnr <= 1 else "RGB_MULTI_SCENE_MIXED"
    aggregate = {"experiment": "M1-RAOC-CAUSAL-FOUR-SCENE", "scenes_completed": [row["scene"] for row in summaries], "hard_failures": failures, "positive_final_utility_delta_scenes": positive_utility, "supported_or_mixed_mechanism_scenes": supported_mechanism, "selective_rescue_scenes": selective, "over_rescue_scenes": over_rescue, "causal_valid_all_completed_scenes": causal_valid, "decomposition_safety_all_completed": safety, "protocol_complete_all_scenes": len(protocol_rows) == 4 and all(row["protocol_complete"] for row in protocol_rows), "final_scene_level_PSNR_delta_mean": mean_psnr, "final_scene_level_PSNR_delta_median": _median(psnr), "final_scene_level_PSNR_positive_scenes": positive_psnr, "final_scene_level_SSIM_delta_mean": _mean(ssim), "final_scene_level_LPIPS_delta_mean": _mean(lpips), "final_scene_level_MSE_delta_mean": _mean(mse), "aggregate_classification": aggregate_classification, "rgb_classification": rgb_classification, "preregistered_supported_criteria": supported_criteria}
    for name, rows in (("four_scene_metrics", metrics_rows), ("four_scene_context_utility", utility_rows), ("four_scene_mechanism", mechanism_rows), ("four_scene_rgb", rgb_rows), ("four_scene_per_view", per_view_rows), ("four_scene_safety", safety_rows), ("four_scene_runtime", runtime_rows), ("four_scene_protocol_audit", protocol_rows)):
        _write_table(root, name, rows)
    _write_json(root / "aggregate" / "aggregate_classification.json", aggregate)
    _write_json(root / "aggregate" / "final_summary.json", {"aggregate": aggregate, "scene_summaries": summaries})
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root.resolve()), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
