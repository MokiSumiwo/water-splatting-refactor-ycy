#!/usr/bin/env python3
"""Aggregate the formal RAOC-Q50 versus RAOC-Q80 four-scene experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
SNAPSHOT_STEPS = (3000, 5000, 8000, 10000, 13000, 14999)
BRANCHES = ("C0", "C1")
QUANTILES = {"C0": 0.50, "C1": 0.80}
EPS = 1e-12
WEAK_CAPACITY_TOLERANCE = 1e-6


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True, default=str) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_table(root: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(root / "aggregate" / f"{name}.csv", rows)
    _write_json(root / "aggregate" / f"{name}.json", {"rows": list(rows)})


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite(value: Any) -> bool:
    return math.isfinite(_number(value))


def _mean(values: Iterable[Any]) -> float:
    numbers = [_number(value) for value in values]
    numbers = [value for value in numbers if math.isfinite(value)]
    return sum(numbers) / len(numbers) if numbers else float("nan")


def _median(values: Iterable[Any]) -> float:
    numbers = sorted(_number(value) for value in values if math.isfinite(_number(value)))
    if not numbers:
        return float("nan")
    middle = len(numbers) // 2
    return numbers[middle] if len(numbers) % 2 else (numbers[middle - 1] + numbers[middle]) / 2.0


def _bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _ratio(numerator: Any, denominator: Any) -> float:
    numerator_f, denominator_f = _number(numerator), _number(denominator)
    return numerator_f / denominator_f if math.isfinite(numerator_f) and math.isfinite(denominator_f) and denominator_f > EPS else float("nan")


def _protocol_audit(root: Path, scene: str) -> Dict[str, Any]:
    scene_dir = root / scene
    expected_steps = set(SNAPSHOT_STEPS)
    checkpoint_steps = {
        branch: {int(path.stem[len("step-") :]) for path in (scene_dir / "checkpoints" / branch).glob("step-*.ckpt")}
        for branch in BRANCHES
    }
    start = _read_json(scene_dir / "start_state_equivalence.json") if (scene_dir / "start_state_equivalence.json").is_file() else {}
    step0 = _read_json(scene_dir / "step0_basis_equivalence.json") if (scene_dir / "step0_basis_equivalence.json").is_file() else {}
    camera = _read_json(scene_dir / "camera_sequence.json") if (scene_dir / "camera_sequence.json").is_file() else {}
    config = _read_json(scene_dir / "scene_config.json") if (scene_dir / "scene_config.json").is_file() else {}
    decomp = _read_csv(scene_dir / "decomposition_safety.csv")
    rescue = _read_csv(scene_dir / "capacity_rescue.csv")
    context = _read_csv(scene_dir / "context_utility.csv")
    weak = _read_csv(scene_dir / "weak_capacity.csv")
    quantile_rows = _read_csv(scene_dir / "quantile_calibration.csv")
    populations = {row.get("population", "") for rows in (rescue, context, weak) for row in rows if row.get("population")}
    expected_populations = {"GENERAL", "M_SAFE"} if scene == "IUI3-RedSea" else {"GENERAL"}
    decomposition_keys = {(row.get("branch"), int(row["absolute_step"]), row.get("split")) for row in decomp if row.get("absolute_step")}
    expected_decomposition = {(branch, step, split) for branch in BRANCHES for step in SNAPSHOT_STEPS for split in ("train", "eval")}
    same_state_rows = [row for row in rescue if row.get("granularity") == "overall" and row.get("population") == "GENERAL"]
    same_state_complete = bool(same_state_rows) and {(row.get("branch"), int(row["absolute_step"])) for row in same_state_rows} == {(branch, step) for branch in BRANCHES for step in SNAPSHOT_STEPS} and all(row.get("counterfactual") == f"{row.get('branch')}_same_state_OCMC" for row in same_state_rows)
    q_config = config.get("arms", {})
    q_config_exact = all(_number(q_config.get(branch, {}).get("local_scale_quantile")) == QUANTILES[branch] for branch in BRANCHES)
    q_rows_exact = bool(quantile_rows) and all(_number(row.get("local_scale_quantile")) == QUANTILES.get(row.get("branch"), float("nan")) for row in quantile_rows)
    weak_finite = bool(weak) and all(_finite(row.get(field)) for row in weak for field in ("ocmc_suppressed_full_residual_ratio", "raoc_kept_full_residual_ratio", "weak_energy_fraction"))
    artifact_names = ("quantile_calibration", "refresh_state_summary", "train_metrics", "eval_metrics", "per_view_eval", "context_utility", "utility_retention", "capacity_rescue", "rescue_efficiency", "evidence_quintiles", "weak_capacity", "decomposition_safety", "depth_tau", "topology", "scene_summary", "scene_classification", "q50_reproducibility")
    artifacts_complete = all((scene_dir / f"{name}.json").is_file() and ((scene_dir / f"{name}.csv").is_file() if name not in ("scene_summary", "scene_classification", "q50_reproducibility") else True) for name in artifact_names)
    checks = {
        "C0_checkpoint_steps_exact": checkpoint_steps["C0"] == expected_steps,
        "C1_checkpoint_steps_exact": checkpoint_steps["C1"] == expected_steps,
        "start_state_equivalence": _bool(start.get("START_STATE_EQUIVALENCE", False)),
        "step0_basis_equivalence": _bool(step0.get("STEP0_BASIS_EQUIVALENCE", False)),
        "step0_q80_ge_q50": _bool(step0.get("q80_ge_q50_all_defined_modes", False)),
        "q_config_exact": q_config_exact,
        "q_rows_exact": q_rows_exact,
        "camera_sequence_match": _bool(camera.get("CAMERA_SEQUENCE_MATCH", False)) and int(camera.get("mismatch_count", -1)) == 0 and int(camera.get("length", -1)) == 15000,
        "decomposition_rows_complete": decomposition_keys == expected_decomposition,
        "decomposition_safety": bool(decomp) and all(_finite(row.get("P_J_gt_1")) and _number(row.get("P_J_gt_1")) == 0.0 for row in decomp),
        "same_state_OCMC_counterfactual_complete": same_state_complete,
        "context_row_types_complete": {"pair", "aggregate", "causal_delta"}.issubset({row.get("row_type") for row in context}),
        "weak_capacity_finite": weak_finite,
        "diagnostic_populations_exact": populations == expected_populations,
        "required_artifacts_complete": artifacts_complete,
    }
    return {"scene": scene, **checks, "protocol_complete": all(checks.values()), "C0_checkpoint_steps": sorted(checkpoint_steps["C0"]), "C1_checkpoint_steps": sorted(checkpoint_steps["C1"]), "decomposition_row_count": len(decomp), "diagnostic_populations": sorted(populations)}


def _scene_rows(root: Path, scene: str, summary: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    scene_dir = root / scene
    archived = summary.get("archived_OCMC_eval", {})
    q50 = summary.get("final_eval_C0", {})
    q80 = summary.get("final_eval_C1", {})
    metric_rows = [{"scene": scene, "arm": arm, "PSNR": metrics.get("PSNR", float("nan")), "SSIM": metrics.get("SSIM", float("nan")), "LPIPS": metrics.get("LPIPS", float("nan")), "MSE": metrics.get("MSE", float("nan")), "quantile": {"RAOC-Q50": 0.50, "RAOC-Q80": 0.80}.get(arm, "")} for arm, metrics in (("OCMC", archived), ("RAOC-Q50", q50), ("RAOC-Q80", q80))]
    utility_rows = [
        {"scene": scene, "arm": "OCMC", "U_context": summary.get("archived_OCMC_correct_context_utility", float("nan")), "utility_gain_over_OCMC": 0.0, "utility_gain_retention_vs_Q50": ""},
        {"scene": scene, "arm": "RAOC-Q50", "U_context": summary.get("final_correct_context_utility_C0", float("nan")), "utility_gain_over_OCMC": summary.get("utility_gain_Q50_over_archived_OCMC", float("nan")), "utility_gain_retention_vs_Q50": 1.0},
        {"scene": scene, "arm": "RAOC-Q80", "U_context": summary.get("final_correct_context_utility_C1", float("nan")), "utility_gain_over_OCMC": summary.get("utility_gain_Q80_over_archived_OCMC", float("nan")), "utility_gain_retention_vs_Q50": summary.get("utility_gain_retention_Q80_over_Q50", float("nan"))},
    ]
    capacity_rows = [
        {"scene": scene, "arm": "RAOC-Q50", "branch": "C0", "rescued_energy": summary.get("rescue_Q50", float("nan")), "rescue_reduction_vs_Q50": 0.0, "high_rescue_retention_vs_Q50": 1.0, "mid_rescue": summary.get("mid_rescue_Q50", float("nan"))},
        {"scene": scene, "arm": "RAOC-Q80", "branch": "C1", "rescued_energy": summary.get("rescue_Q80", float("nan")), "rescue_reduction_vs_Q50": summary.get("rescue_reduction_Q80_vs_Q50", float("nan")), "high_rescue_retention_vs_Q50": summary.get("high_rescue_retention_Q80_over_Q50", float("nan")), "mid_rescue": summary.get("mid_rescue_Q80", float("nan"))},
    ]
    efficiency_rows = [
        {"scene": scene, "arm": "RAOC-Q50", "eta_U": summary.get("eta_U_Q50", float("nan")), "eta_RGB": summary.get("eta_RGB_Q50", float("nan")), "rescued_energy": summary.get("rescue_Q50", float("nan"))},
        {"scene": scene, "arm": "RAOC-Q80", "eta_U": summary.get("eta_U_Q80", float("nan")), "eta_RGB": summary.get("eta_RGB_Q80", float("nan")), "rescued_energy": summary.get("rescue_Q80", float("nan"))},
    ]
    rgb_rows = [{"scene": scene, "arm": arm, **metrics} for arm, metrics in (("OCMC", archived), ("RAOC-Q50", q50), ("RAOC-Q80", q80))]
    per_view_rows = [{"scene": scene, **row} for row in (_read_json(scene_dir / "per_view_eval_delta.json").get("rows", []) if (scene_dir / "per_view_eval_delta.json").is_file() else [])]
    safety_rows = [{"scene": scene, "decomposition_safety": summary.get("decomposition_safety", False), "weak_capacity_raoc_kept_full_residual_ratio_Q50": summary.get("weak_capacity_raoc_kept_full_residual_ratio_Q50", float("nan")), "weak_capacity_raoc_kept_full_residual_ratio_Q80": summary.get("weak_capacity_raoc_kept_full_residual_ratio_Q80", float("nan")), "weak_capacity_suppression_retained_or_improved": summary.get("weak_capacity_suppression_retained_or_improved", False), "start_state_equivalence": summary.get("START_STATE_EQUIVALENCE", False), "camera_sequence_match": summary.get("CAMERA_SEQUENCE_MATCH", False), "q50_reproducible": summary.get("q50_reproducibility", {}).get("reproducibility_confirmed", False)}]
    return {"metrics": metric_rows, "utility": utility_rows, "capacity": capacity_rows, "efficiency": efficiency_rows, "rgb": rgb_rows, "per_view": per_view_rows, "safety": safety_rows}


def run(root: Path) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for scene in SCENES:
        path = root / scene / "scene_summary.json"
        if not path.is_file():
            failures.append({"scene": scene, "reason": "missing scene_summary.json"})
            continue
        summary = _read_json(path)
        summary["_scene_dir"] = str(path.parent)
        summaries.append(summary)
    grouped: Dict[str, List[Dict[str, Any]]] = {key: [] for key in ("metrics", "utility", "capacity", "efficiency", "rgb", "per_view", "safety")}
    protocol_rows = []
    for summary in summaries:
        scene_rows = _scene_rows(root, summary["scene"], summary)
        for key, values in scene_rows.items():
            grouped[key].extend(values)
        protocol_rows.append(_protocol_audit(root, summary["scene"]))
    reductions = [_number(summary.get("rescue_reduction_Q80_vs_Q50")) for summary in summaries]
    mid_reductions = [_number(summary.get("mid_rescue_reduction_Q80_vs_Q50")) for summary in summaries]
    high_reductions = [1.0 - _number(summary.get("high_rescue_retention_Q80_over_Q50")) for summary in summaries]
    utility_gains = [_number(summary.get("utility_gain_Q80_over_archived_OCMC")) for summary in summaries]
    eta_q50 = [_number(summary.get("eta_U_Q50")) for summary in summaries]
    eta_q80 = [_number(summary.get("eta_U_Q80")) for summary in summaries]
    psnr_delta = [_number(summary.get("final_eval_delta_C1_minus_C0", {}).get("PSNR")) for summary in summaries]
    weak_suppression = [
        _finite(summary.get("weak_capacity_raoc_kept_full_residual_ratio_Q50"))
        and _finite(summary.get("weak_capacity_raoc_kept_full_residual_ratio_Q80"))
        and _number(summary.get("weak_capacity_raoc_kept_full_residual_ratio_Q80"))
        <= _number(summary.get("weak_capacity_raoc_kept_full_residual_ratio_Q50")) + WEAK_CAPACITY_TOLERANCE
        and _bool(summary.get("weak_capacity_suppression_retained_or_improved", False))
        for summary in summaries
    ]
    q50_repro = [summary.get("q50_reproducibility", {}) for summary in summaries]
    reproducibility_confirmed = bool(summaries) and len(summaries) == 4 and all(_bool(row.get("reproducibility_confirmed", False)) for row in q50_repro)
    reduction_count = sum(value > 0.05 for value in reductions if math.isfinite(value))
    middle_selective_count = sum(mid > high for mid, high in zip(mid_reductions, high_reductions) if math.isfinite(mid) and math.isfinite(high))
    utility_positive_count = sum(value > 0 for value in utility_gains if math.isfinite(value))
    eta_improved_count = sum(q80 > q50 for q50, q80 in zip(eta_q50, eta_q80) if math.isfinite(q50) and math.isfinite(q80))
    psnr_positive_count = sum(value > 0 for value in psnr_delta if math.isfinite(value))
    protocol_complete = len(protocol_rows) == 4 and all(row["protocol_complete"] for row in protocol_rows)
    safety_complete = bool(summaries) and all(_bool(summary.get("decomposition_safety", False)) for summary in summaries)
    weak_suppression_complete = len(weak_suppression) == 4 and all(weak_suppression)
    over_conservative = sum(summary.get("classification") == "RAOC_Q80_OVER_CONSERVATIVE" for summary in summaries)
    causal_valid = protocol_complete and len(summaries) == 4 and all(
        _bool(summary.get("START_STATE_EQUIVALENCE", False))
        and _number(summary.get("camera_sequence_mismatch_count", -1)) == 0
        for summary in summaries
    )
    criteria = {
        "causal_valid_all_completed_scenes": causal_valid,
        "q50_reproducibility_confirmed": reproducibility_confirmed,
        "q80_reduces_total_rescue_at_least_3": reduction_count >= 3,
        "q80_reduces_middle_more_than_top_at_least_3": middle_selective_count >= 3,
        "q80_utility_gain_over_archived_ocmc_positive_at_least_3": utility_positive_count >= 3,
        "eta_U_improves_at_least_3": eta_improved_count >= 3,
        "weak_capacity_suppression_intact_all": weak_suppression_complete,
        "decomposition_safety_all": safety_complete,
        "q80_beats_q50_final_PSNR_at_least_3": psnr_positive_count >= 3,
        "mean_scene_level_PSNR_delta_positive": _mean(psnr_delta) > 0,
        "no_more_than_one_over_conservative": over_conservative <= 1,
    }
    if all(criteria.values()):
        aggregate_classification = "RAOC_Q80_MULTI_SCENE_SUPPORTED"
    elif causal_valid and reproducibility_confirmed and reduction_count >= 3 and middle_selective_count >= 3 and utility_positive_count >= 3 and eta_improved_count >= 3 and weak_suppression_complete and safety_complete and over_conservative <= 2:
        aggregate_classification = "RAOC_Q80_MULTI_SCENE_TENTATIVE"
    else:
        aggregate_classification = "RAOC_Q80_MULTI_SCENE_NOT_SUPPORTED"
    q80_ssim = _mean(row.get("SSIM") for row in grouped["rgb"] if row.get("arm") == "RAOC-Q80")
    q50_ssim = _mean(row.get("SSIM") for row in grouped["rgb"] if row.get("arm") == "RAOC-Q50")
    q80_lpips = _mean(row.get("LPIPS") for row in grouped["rgb"] if row.get("arm") == "RAOC-Q80")
    q50_lpips = _mean(row.get("LPIPS") for row in grouped["rgb"] if row.get("arm") == "RAOC-Q50")
    if psnr_positive_count >= 3 and q80_ssim >= q50_ssim and q80_lpips <= q50_lpips:
        rgb_classification = "RAOC_Q80_RGB_IMPROVED"
    elif psnr_positive_count <= 1:
        rgb_classification = "RAOC_Q80_RGB_DEGRADED"
    else:
        rgb_classification = "RAOC_Q80_RGB_MIXED"
    aggregate = {
        "experiment": "RAOC-Q50-VS-Q80-CAUSAL-FOUR-SCENE",
        "scenes_completed": [summary["scene"] for summary in summaries],
        "hard_failures": failures,
        "protocol_complete_all_scenes": protocol_complete,
        "causal_valid_all_completed_scenes": criteria["causal_valid_all_completed_scenes"],
        "q50_reproducibility_confirmed": reproducibility_confirmed,
        "q80_rescue_reduction_scenes": reduction_count,
        "q80_middle_selectivity_scenes": middle_selective_count,
        "q80_positive_utility_gain_over_ocmc_scenes": utility_positive_count,
        "eta_U_improved_scenes": eta_improved_count,
        "weak_capacity_suppression_retained_scenes": sum(weak_suppression),
        "weak_capacity_suppression_tolerance": WEAK_CAPACITY_TOLERANCE,
        "q80_positive_PSNR_scenes": psnr_positive_count,
        "q80_rescue_reduction_mean": _mean(reductions),
        "q80_mid_rescue_reduction_mean": _mean(mid_reductions),
        "q80_high_rescue_reduction_mean": _mean(high_reductions),
        "q80_scene_level_PSNR_delta_mean": _mean(psnr_delta),
        "q80_scene_level_PSNR_delta_median": _median(psnr_delta),
        "q80_scene_level_PSNR_positive_scenes": psnr_positive_count,
        "q80_utility_gain_over_ocmc_mean": _mean(utility_gains),
        "eta_U_Q50_mean": _mean(eta_q50),
        "eta_U_Q80_mean": _mean(eta_q80),
        "over_conservative_scenes": over_conservative,
        "preregistered_supported_criteria": criteria,
        "aggregate_classification": aggregate_classification,
        "rgb_classification": rgb_classification,
    }
    for name, rows in (("four_scene_q50_q80_metrics", grouped["metrics"]), ("four_scene_utility", grouped["utility"]), ("four_scene_capacity", grouped["capacity"]), ("four_scene_efficiency", grouped["efficiency"]), ("four_scene_rgb", grouped["rgb"]), ("four_scene_per_view", grouped["per_view"]), ("four_scene_safety", grouped["safety"]), ("four_scene_protocol_audit", protocol_rows)):
        _write_table(root, name, rows)
    _write_json(root / "aggregate" / "aggregate_classification.json", aggregate)
    _write_json(root / "aggregate" / "final_summary.json", {"aggregate": aggregate, "scene_summaries": summaries})
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root.resolve()), indent=2, sort_keys=True, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
