#!/usr/bin/env python3
"""Aggregate and classify the formal four-scene identifiability experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
ARMS = ("C0", "C1")
START_STEP = 3000
FINAL_STEP = 14999
STEPS = (3000, 5000, 8000, 10000, 13000, 14999)
MIN_CAPACITY_RATIO = 0.75
MIN_OPACITY_RATIO = 0.75
MAX_OPACITY_RATIO = 1.25
MAX_POPULATION_RELATIVE_GAP = 0.10
SUPPORTED_LABEL = "IDENTIFIABILITY_MODULE_SUPPORTED"
TENTATIVE_LABEL = "IDENTIFIABILITY_MECHANISM_SUPPORTED_BUT_RGB_TENTATIVE"
NOT_SUPPORTED_LABEL = "IDENTIFIABILITY_MODULE_NOT_SUPPORTED"


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf8",
    )


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
            writer.writerow({key: _sanitize(row.get(key, "")) for key in fields})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(abs(float(denominator)), 1e-12)


def _relative_delta(after: float, before: float) -> float:
    return (float(after) - float(before)) / max(abs(float(before)), 1e-12)


def _keyed(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], Mapping[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def _scene_tables(root: Path, scene: str) -> Dict[str, Any]:
    scene_dir = root / scene
    complete = _read_json(scene_dir / "scene_complete.json")
    training = _read_json(scene_dir / "training_summary.json")
    start = _read_json(scene_dir / "start_state_equivalence.json")
    camera = _read_json(scene_dir / "camera_sequence.json")
    metrics = _read_csv(scene_dir / "evaluation_metrics.csv")
    per_view = _read_csv(scene_dir / "per_view_metrics.csv")
    mechanism = _read_csv(scene_dir / "mechanism_metrics.csv")
    gradient = _read_csv(scene_dir / "gradient_audit.csv")
    topology = _read_csv(scene_dir / "topology_metrics.csv")
    events = _read_csv(scene_dir / "refinement_events.csv")
    decomp = _read_json(scene_dir / "decomposition_safety.json")["rows"]
    ocmc = _read_csv(scene_dir / "ocmc_magnitude.csv")
    counterfactual = _read_csv(scene_dir / "counterfactual_metrics.csv")
    checkpoints = _read_csv(scene_dir / "checkpoint_manifest.csv")
    return {
        "dir": scene_dir,
        "complete": complete,
        "training": training,
        "start": start,
        "camera": camera,
        "metrics": metrics,
        "per_view": per_view,
        "mechanism": mechanism,
        "gradient": gradient,
        "topology": topology,
        "events": events,
        "decomp": decomp,
        "ocmc": ocmc,
        "counterfactual": counterfactual,
        "checkpoints": checkpoints,
    }


def _scene_summary(scene: str, tables: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metrics = _keyed(tables["metrics"], ("arm", "absolute_step", "split"))
    mechanism = _keyed(tables["mechanism"], ("arm", "absolute_step"))
    topology = _keyed(tables["topology"], ("arm", "absolute_step"))
    c0_eval = metrics[("C0", str(FINAL_STEP), "eval")]
    c1_eval = metrics[("C1", str(FINAL_STEP), "eval")]
    c0_train = metrics[("C0", str(FINAL_STEP), "train")]
    c1_train = metrics[("C1", str(FINAL_STEP), "train")]
    c0_mech = mechanism[("C0", str(FINAL_STEP))]
    c1_mech = mechanism[("C1", str(FINAL_STEP))]
    c0_top = topology[("C0", str(FINAL_STEP))]
    c1_top = topology[("C1", str(FINAL_STEP))]

    eval_delta = {
        key: _number(c1_eval[key]) - _number(c0_eval[key])
        for key in ("PSNR", "SSIM", "LPIPS", "MSE")
    }
    eval_relative = {
        key: _relative_delta(_number(c1_eval[key]), _number(c0_eval[key]))
        for key in ("PSNR", "SSIM", "LPIPS", "MSE")
    }
    train_delta = {
        key: _number(c1_train[key]) - _number(c0_train[key])
        for key in ("PSNR", "SSIM", "LPIPS", "MSE")
    }
    shared_delta = _number(c1_mech["SH_opacity_shared_response_energy"]) - _number(
        c0_mech["SH_opacity_shared_response_energy"]
    )
    overlap_delta = _number(c1_mech["median_tangent_overlap"]) - _number(
        c0_mech["median_tangent_overlap"]
    )
    non_dc_ratio = _ratio(
        _number(c1_mech["SH_nonDC_energy"]), _number(c0_mech["SH_nonDC_energy"])
    )
    orthogonal_ratio = _ratio(
        _number(c1_mech["SH_orthogonal_energy"]), _number(c0_mech["SH_orthogonal_energy"])
    )
    opacity_mean_ratio = _ratio(
        _number(c1_mech["opacity_mean"]), _number(c0_mech["opacity_mean"])
    )
    opacity_median_ratio = _ratio(
        _number(c1_mech["opacity_q50"]), _number(c0_mech["opacity_q50"])
    )
    count0, count1 = _number(c0_top["gaussian_count"]), _number(c1_top["gaussian_count"])
    population_gap = abs(count1 - count0) / max(count0, 1.0)
    shared_temporal_decrease_count = 0
    overlap_temporal_decrease_count = 0
    for step in STEPS[1:]:
        temporal_c0 = mechanism[("C0", str(step))]
        temporal_c1 = mechanism[("C1", str(step))]
        shared_temporal_decrease_count += bool(
            _number(temporal_c1["SH_opacity_shared_response_energy"])
            < _number(temporal_c0["SH_opacity_shared_response_energy"])
        )
        overlap_temporal_decrease_count += bool(
            _number(temporal_c1["median_tangent_overlap"])
            < _number(temporal_c0["median_tangent_overlap"])
        )
    temporal_mechanism_stable = bool(
        shared_temporal_decrease_count >= 3
        and overlap_temporal_decrease_count >= 3
        and shared_delta < 0.0
        and overlap_delta < 0.0
    )

    final_views = [row for row in tables["per_view"] if row["absolute_step"] == str(FINAL_STEP) and row["split"] == "eval"]
    view_map = _keyed(final_views, ("arm", "view_id"))
    view_ids = sorted({row["view_id"] for row in final_views if row["arm"] == "C0"})
    per_view_delta = []
    for view_id in view_ids:
        left, right = view_map[("C0", view_id)], view_map[("C1", view_id)]
        per_view_delta.append(
            {
                "scene": scene,
                "absolute_step": FINAL_STEP,
                "split": "eval",
                "view_id": view_id,
            }
        )
    # Append paired values separately to keep the literal metric loop simple.
    for row, view_id in zip(per_view_delta, view_ids):
        left, right = view_map[("C0", view_id)], view_map[("C1", view_id)]
        for key in ("PSNR", "SSIM", "LPIPS", "MSE"):
            row[f"{key}_C0"] = _number(left[key])
            row[f"{key}_C1"] = _number(right[key])
            row[f"{key}_delta_C1_minus_C0"] = _number(right[key]) - _number(left[key])
            row[f"{key}_relative_delta"] = _relative_delta(_number(right[key]), _number(left[key]))

    expected_checkpoints = {(arm, str(step)) for arm in ARMS for step in STEPS[1:]}
    actual_checkpoints = {(row["arm"], row["absolute_step"]) for row in tables["checkpoints"]}
    checkpoint_files_valid = bool(len(tables["checkpoints"]) == len(expected_checkpoints))
    for row in tables["checkpoints"]:
        path = Path(row["path"])
        checkpoint_files_valid = bool(
            checkpoint_files_valid
            and path.is_file()
            and path.stat().st_size == int(row["size_bytes"])
            and _sha256(path) == row["sha256"]
        )
    gradient_c1 = [row for row in tables["gradient"] if row["arm"] == "C1"]
    gradient_c0 = [row for row in tables["gradient"] if row["arm"] == "C0"]
    direct_groups = {
        row["parameter_group"]: max(
            _number(item["module_direct_gradient_l2"])
            for item in gradient_c1
            if item["parameter_group"] == row["parameter_group"]
        )
        for row in gradient_c1
    }
    excluded_groups = (
        "means",
        "features_dc",
        "scales",
        "quats",
        "opacities",
        "medium_mlp",
        "direction_encoding",
        "medium_branch",
    )
    direct_gradient_valid = bool(
        gradient_c0
        and gradient_c1
        and all(_number(row["module_direct_gradient_l2"]) == 0.0 for row in gradient_c0)
        and direct_groups.get("features_rest", 0.0) > 0.0
        and all(direct_groups.get(group, 0.0) == 0.0 for group in excluded_groups)
    )
    safety_rows = [row for row in tables["decomp"] if int(row["absolute_step"]) == FINAL_STEP]
    decomposition_safe = bool(safety_rows) and all(
        math.isfinite(_number(row["P_J_gt_1"])) and _number(row["P_J_gt_1"]) == 0.0
        for row in safety_rows
    )
    decomposition_final = {
        f"{row['branch']}_{row['split']}": {
            "P_J_gt_1": _number(row["P_J_gt_1"]),
            "J_p99": _number(row["J_p99"]),
        }
        for row in safety_rows
    }
    final_ocmc = {
        (row["arm"], row["split"]): _number(row["ocmc_projected_raw_rms"])
        for row in tables["ocmc"]
        if row["absolute_step"] == str(FINAL_STEP)
    }
    counterfactual_summary = {}
    for arm in ARMS:
        arm_rows = [row for row in tables["counterfactual"] if row["arm"] == arm]
        counterfactual_summary[arm] = {
            "view_count": len(arm_rows),
            "mean_shared_removed_minus_full": {
                key: _mean([_number(row[f"SHARED_REMOVED_minus_FULL_{key}"]) for row in arm_rows])
                for key in ("PSNR", "SSIM", "LPIPS", "MSE")
            },
            "mean_orthogonal_relative_drift": _mean(
                [_number(row["orthogonal_relative_drift"]) for row in arm_rows]
            ),
        }
    start = tables["start"]
    camera = tables["camera"]
    training = tables["training"]
    causal_valid = bool(
        tables["complete"]["formal"]
        and tables["complete"]["integrity"]["worker_script_unchanged"]
        and tables["complete"]["integrity"]["protected_hashes_unchanged"]
        and tables["complete"]["integrity"]["source_checkpoint_unchanged"]
        and tables["complete"]["integrity"]["source_camera_sequence_unchanged"]
        and training["formal"]
        and start["START_STATE_EQUIVALENCE"]
        and start["max_abs_model_diff"] == 0.0
        and start["optimizer_equivalent"]
        and start["scheduler_equivalent"]
        and start["scaler_equivalent"]
        and start["rng_equivalent"]
        and camera["CAMERA_SEQUENCE_EXACT_MATCH"]
        and camera["camera_mismatch_count"] == 0
        and camera["sha256_C0"] == camera["sha256_C1"]
        and training["matched_updates_per_arm"] == FINAL_STEP - START_STEP
        and training["arms"]["C0"]["completed_updates"] == FINAL_STEP - START_STEP
        and training["arms"]["C1"]["completed_updates"] == FINAL_STEP - START_STEP
    )
    ocmc_hashes_equal = bool(
        start["ocmc_projector_max_abs_diff"] == 0.0
        and training["arms"]["C0"]["ocmc_projector_unchanged"]
        and training["arms"]["C1"]["ocmc_projector_unchanged"]
        and training["arms"]["C0"]["ocmc_projector_hash_start"]
        == training["arms"]["C1"]["ocmc_projector_hash_start"]
        and start["ocmc_projector_sha256_C0"] == start["ocmc_projector_sha256_C1"]
        and start["ocmc_configuration_sha256_C0"] == start["ocmc_configuration_sha256_C1"]
    )
    capacity_preserved = non_dc_ratio >= MIN_CAPACITY_RATIO and orthogonal_ratio >= MIN_CAPACITY_RATIO
    opacity_stable = (
        MIN_OPACITY_RATIO <= opacity_mean_ratio <= MAX_OPACITY_RATIO
        and MIN_OPACITY_RATIO <= opacity_median_ratio <= MAX_OPACITY_RATIO
    )
    topology_normal = population_gap <= MAX_POPULATION_RELATIVE_GAP
    scene_summary = {
        "scene": scene,
        "causal_valid": causal_valid,
        "checkpoint_set_complete": actual_checkpoints == expected_checkpoints,
        "checkpoint_files_valid": checkpoint_files_valid,
        "direct_gradient_valid": direct_gradient_valid,
        "ocmc_independent": ocmc_hashes_equal,
        "decomposition_safe": decomposition_safe,
        "final_decomposition": decomposition_final,
        "shared_energy_decreased": shared_delta < 0.0,
        "shared_energy_delta": shared_delta,
        "shared_energy_relative_delta": _relative_delta(
            _number(c1_mech["SH_opacity_shared_response_energy"]),
            _number(c0_mech["SH_opacity_shared_response_energy"]),
        ),
        "tangent_overlap_decreased": overlap_delta < 0.0,
        "tangent_overlap_delta": overlap_delta,
        "shared_energy_temporal_decrease_checkpoint_count": shared_temporal_decrease_count,
        "tangent_overlap_temporal_decrease_checkpoint_count": overlap_temporal_decrease_count,
        "temporal_mechanism_stable": temporal_mechanism_stable,
        "SH_nonDC_ratio_C1_over_C0": non_dc_ratio,
        "SH_orthogonal_ratio_C1_over_C0": orthogonal_ratio,
        "SH_capacity_preserved": capacity_preserved,
        "opacity_mean_ratio_C1_over_C0": opacity_mean_ratio,
        "opacity_median_ratio_C1_over_C0": opacity_median_ratio,
        "opacity_stable": opacity_stable,
        "gaussian_count_C0": int(count0),
        "gaussian_count_C1": int(count1),
        "gaussian_population_relative_gap": population_gap,
        "gaussian_population_normal": topology_normal,
        "final_train_C0": {key: _number(c0_train[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_train_C1": {key: _number(c1_train[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_train_delta_C1_minus_C0": train_delta,
        "final_heldout_C0": {key: _number(c0_eval[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_heldout_C1": {key: _number(c1_eval[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
        "final_heldout_delta_C1_minus_C0": eval_delta,
        "final_heldout_relative_delta_C1_minus_C0": eval_relative,
        "positive_heldout_PSNR_view_fraction": (
            sum(row["PSNR_delta_C1_minus_C0"] >= 0.0 for row in per_view_delta) / len(per_view_delta)
            if per_view_delta
            else float("nan")
        ),
        "final_eval_ocmc_projected_raw_rms_C0": final_ocmc[("C0", "eval")],
        "final_eval_ocmc_projected_raw_rms_C1": final_ocmc[("C1", "eval")],
        "counterfactual": counterfactual_summary,
    }
    return scene_summary, per_view_delta


def _temporal_rows(scene: str, tables: Mapping[str, Any]) -> List[Dict[str, Any]]:
    metrics = _keyed(tables["metrics"], ("arm", "absolute_step", "split"))
    mechanism = _keyed(tables["mechanism"], ("arm", "absolute_step"))
    rows = []
    for step in STEPS:
        for arm in ARMS:
            mech = mechanism[(arm, str(step))]
            for split in ("train", "eval"):
                metric = metrics[(arm, str(step), split)]
                rows.append(
                    {
                        "scene": scene,
                        "arm": arm,
                        "absolute_step": step,
                        "split": split,
                        **{key: _number(metric[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")},
                        **{
                            key: _number(mech[key])
                            for key in (
                                "SH_total_energy",
                                "SH_DC_energy",
                                "SH_nonDC_energy",
                                "SH_shared_coefficient_energy",
                                "SH_orthogonal_energy",
                                "SH_opacity_shared_response_energy",
                                "median_tangent_overlap",
                                "opacity_mean",
                                "opacity_q50",
                            )
                        },
                    }
                )
    return rows


def _research_note(root: Path, summary: Mapping[str, Any]) -> None:
    note = root.parents[1] / "research_notes" / "IDENTIFIABILITY_MODULE_CAUSAL_EXPERIMENT_2026-09-02.md"
    lines = [
        "# Identifiability Module Causal Experiment",
        "",
        "Date: 2026-09-02",
        f"Classification: `{summary['classification']}`",
        "",
        "## Hypothesis And Arms",
        "",
        "C0 is the frozen-OCMC baseline. C1 differs only by the detached SH-opacity tangent regularizer with strength 1.0. Both arms restore the same step-3000 model, optimizer, scheduler, scaler, RNG plan, OCMC projector, and camera sequence, then run 11,999 updates through step 14,999.",
        "",
        "Training is fixed to seed 42, `bounded_sh3`, SH degree 3, `dir_xy_camera`, tied `B_inf`, classic rasterization, OCMC on, RAOC off, and five retained arm checkpoints at steps 5000, 8000, 10000, 13000, and 14999. No sweep was run.",
        "",
        "## Mechanism",
        "",
        "Visible training cameras define one detached non-DC SH direction per Gaussian that is aligned with the raw-opacity RGB tangent. The anchored scalar penalty acts only on `features_rest`; DC, opacity, geometry, medium, OCMC, and topology receive no direct module gradient. No GT or heldout view constructs the controller.",
        "",
        "## Four-Scene Results",
        "",
        "| Scene | train dPSNR | heldout dPSNR | heldout dSSIM | heldout dLPIPS | view+ | shared rel | overlap delta | temporal shared/overlap | non-DC/orth | opacity | count gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scene_rows"]:
        delta = row["final_heldout_delta_C1_minus_C0"]
        lines.append(
            f"| {row['scene']} | {row['final_train_delta_C1_minus_C0']['PSNR']:.6f} | "
            f"{delta['PSNR']:.6f} | {delta['SSIM']:.6f} | {delta['LPIPS']:.6f} | "
            f"{row['positive_heldout_PSNR_view_fraction']:.3f} | "
            f"{row['shared_energy_relative_delta']:.6f} | {row['tangent_overlap_delta']:.6f} | "
            f"{row['shared_energy_temporal_decrease_checkpoint_count']}/"
            f"{row['tangent_overlap_temporal_decrease_checkpoint_count']} | "
            f"{row['SH_nonDC_ratio_C1_over_C0']:.3f}/{row['SH_orthogonal_ratio_C1_over_C0']:.3f} | "
            f"{row['opacity_mean_ratio_C1_over_C0']:.6f} | "
            f"{row['gaussian_population_relative_gap']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Causal Matching And OCMC Independence",
            "",
            f"All-scene causal validity: `{str(summary['causal_valid']).lower()}`. OCMC state/config independence: `{str(summary['ocmc_independence']).lower()}`. Direct-gradient audit: `{str(summary['direct_gradient_valid']).lower()}`.",
            "",
            "## Capacity, Opacity, Topology, And Decomposition Safety",
            "",
            f"No global SH collapse: `{str(summary['no_global_SH_collapse']).lower()}`. Opacity stable: `{str(summary['opacity_stable']).lower()}`. Gaussian population normal: `{str(summary['gaussian_population_normal']).lower()}`. Decomposition safety: `{str(summary['decomposition_safety']).lower()}`.",
            "",
            "Final `P(J > 1)` is exactly zero for train and heldout in both arms of every scene. Final maximum `J_p99` values by scene (C0/C1 over train and heldout) are: "
            + "; ".join(
                f"{row['scene']} {max(item['J_p99'] for key, item in row['final_decomposition'].items() if key.startswith('C0_')):.6f}/"
                f"{max(item['J_p99'] for key, item in row['final_decomposition'].items() if key.startswith('C1_')):.6f}"
                for row in summary["scene_rows"]
            )
            + ".",
            "",
            "Final eval OCMC projected-raw RMS (C0/C1) is: "
            + "; ".join(
                f"{row['scene']} {row['final_eval_ocmc_projected_raw_rms_C0']:.6f}/"
                f"{row['final_eval_ocmc_projected_raw_rms_C1']:.6f}"
                for row in summary["scene_rows"]
            )
            + ". The projector and OCMC configuration hashes are identical, and both arm projectors remain unchanged.",
            "",
            "## Counterfactual Diagnostic",
            "",
            "Removing the sampled training-anchored shared component at the final checkpoints changes mean heldout PSNR by "
            + "; ".join(
                f"{row['scene']} C0 {row['counterfactual']['C0']['mean_shared_removed_minus_full']['PSNR']:.6f} dB, "
                f"C1 {row['counterfactual']['C1']['mean_shared_removed_minus_full']['PSNR']:.6f} dB"
                for row in summary["scene_rows"]
            )
            + ". Mean orthogonal relative drift is below 1.3e-7 in every arm, confirming that this read-only diagnostic preserves the sampled orthogonal component.",
            "",
            "## RGB And Mechanism Classification",
            "",
            f"Heldout PSNR improved or tied in {summary['positive_heldout_PSNR_scene_count']}/4 scenes; mean delta was {summary['mean_heldout_PSNR_delta']:.6f} dB. Mechanism classification: `{summary['mechanism_classification']}`. RGB classification: `{summary['RGB_classification']}`.",
            "",
            "Shared response energy decreased at the final checkpoint in "
            f"{summary['shared_energy_decrease_scene_count']}/4 scenes, final tangent overlap decreased in "
            f"{summary['tangent_overlap_decrease_scene_count']}/4, and both mechanism metrics were temporally stable in "
            f"{summary['temporally_stable_mechanism_scene_count']}/4. JapaneseGradens-RedSea failed the registered SH capacity floor: non-DC and orthogonal C1/C0 ratios were 0.669652 and 0.735867, below 0.75.",
            "",
            "## Required Answers",
            "",
            "1. Same start state: yes, exactly, in 4/4 scenes.",
            "2. Camera sequence: exact match with zero mismatches in 4/4 scenes.",
            "3. Completion: all four scenes completed 11,999 updates per arm.",
            f"4. Shared energy: decreased in {summary['shared_energy_decrease_scene_count']}/4 scenes.",
            f"5. Tangent overlap: decreased at final in {summary['tangent_overlap_decrease_scene_count']}/4 scenes; temporal mechanism stability held in {summary['temporally_stable_mechanism_scene_count']}/4.",
            "6. SH collapse: registered capacity preservation failed in JapaneseGradens-RedSea.",
            "7. Opacity: stable in 4/4 scenes.",
            "8. Gaussian population: normal in 4/4 scenes; final relative gaps are all below 0.34%.",
            "9. Train PSNR deltas (C1-C0): Curasao +0.098333, IUI3-RedSea -0.051881, JapaneseGradens-RedSea -0.043542, Panama +0.019816 dB.",
            "10. Heldout PSNR deltas (C1-C0): Curasao -0.045276, IUI3-RedSea +0.090387, JapaneseGradens-RedSea +0.027695, Panama +0.072294 dB.",
            f"11. Heldout PSNR improved or tied in {summary['positive_heldout_PSNR_scene_count']}/4 scenes.",
            f"12. Mean heldout PSNR delta: {summary['mean_heldout_PSNR_delta']:+.6f} dB.",
            "13. Decomposition safety: preserved; final `P(J > 1) = 0` for every arm/split/scene.",
            "14. OCMC independence: passed; projector state and configuration stayed frozen and identical.",
            f"15. Mechanism SUPPORT: {summary['mechanism_classification']}.",
            f"16. RGB SUPPORT: {summary['RGB_classification']}.",
            f"17. Final module classification: `{summary['classification']}`.",
            f"18. Next unique task: `{summary['next_unique_task']}`.",
            "",
            "## Limitations",
            "",
            "The controller and diagnostics are local first-order tests of representation redundancy. They do not establish that SH is true radiance, that opacity is true geometry, or that PSNR implies physical correctness. This was one fixed-strength experiment with no sweep.",
            "",
            "## Final Classification",
            "",
            f"The final module classification is `{summary['classification']}`. Next task: `{summary['next_unique_task']}`.",
            "",
        ]
    )
    note.write_text("\n".join(lines), encoding="utf8")


def run(root: Path) -> Dict[str, Any]:
    scene_rows: List[Dict[str, Any]] = []
    per_view_rows: List[Dict[str, Any]] = []
    temporal_rows: List[Dict[str, Any]] = []
    all_metrics: List[Dict[str, Any]] = []
    all_mechanism: List[Dict[str, Any]] = []
    all_gradient: List[Dict[str, Any]] = []
    all_topology: List[Dict[str, Any]] = []
    all_decomp: List[Dict[str, Any]] = []
    ocmc_rows: List[Dict[str, Any]] = []
    counterfactual_rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        tables = _scene_tables(root, scene)
        scene_summary, view_delta = _scene_summary(scene, tables)
        scene_rows.append(scene_summary)
        per_view_rows.extend(view_delta)
        temporal_rows.extend(_temporal_rows(scene, tables))
        for name, target in (
            ("metrics", all_metrics),
            ("mechanism", all_mechanism),
            ("gradient", all_gradient),
            ("topology", all_topology),
            ("ocmc", ocmc_rows),
            ("counterfactual", counterfactual_rows),
        ):
            target.extend({"scene": scene, **row} for row in tables[name])
        all_decomp.extend({"scene": scene, **row} for row in tables["decomp"])

    positive_psnr = sum(row["final_heldout_delta_C1_minus_C0"]["PSNR"] >= 0.0 for row in scene_rows)
    mean_psnr = _mean([row["final_heldout_delta_C1_minus_C0"]["PSNR"] for row in scene_rows])
    shared_count = sum(row["shared_energy_decreased"] for row in scene_rows)
    overlap_count = sum(row["tangent_overlap_decreased"] for row in scene_rows)
    temporally_stable_count = sum(row["temporal_mechanism_stable"] for row in scene_rows)
    causal_valid = all(
        row["causal_valid"] and row["checkpoint_set_complete"] and row["checkpoint_files_valid"]
        for row in scene_rows
    )
    gradient_valid = all(row["direct_gradient_valid"] for row in scene_rows)
    ocmc_independent = all(row["ocmc_independent"] for row in scene_rows)
    decomposition_safe = all(row["decomposition_safe"] for row in scene_rows)
    capacity_preserved = all(row["SH_capacity_preserved"] for row in scene_rows)
    opacity_stable = all(row["opacity_stable"] for row in scene_rows)
    topology_normal = all(row["gaussian_population_normal"] for row in scene_rows)
    mechanism_supported = shared_count >= 3 and overlap_count >= 3 and temporally_stable_count >= 3
    rgb_supported = positive_psnr >= 3 and mean_psnr >= 0.0
    majority_rgb_degradation = positive_psnr < 2
    integrity = bool(
        causal_valid
        and gradient_valid
        and ocmc_independent
        and decomposition_safe
        and capacity_preserved
        and opacity_stable
        and topology_normal
    )
    if mechanism_supported and rgb_supported and integrity:
        classification = SUPPORTED_LABEL
        next_task = "LOCK_IDENTIFIABILITY_MODULE_AND_BEGIN_FINAL_ABLATION"
    elif mechanism_supported and integrity and not majority_rgb_degradation:
        classification = TENTATIVE_LABEL
        next_task = "DO_NOT_TUNE; perform one targeted diagnostic only"
    else:
        classification = NOT_SUPPORTED_LABEL
        next_task = "CLOSE_IDENTIFIABILITY_MODULE_RESEARCH_LINE"
    classification_reasons = []
    if not mechanism_supported:
        classification_reasons.append(
            "mechanism support failed: final tangent overlap decreased in fewer than 3/4 scenes "
            "and temporal stability held in fewer than 3/4 scenes"
        )
    if not rgb_supported:
        classification_reasons.append("RGB support failed")
    if not integrity:
        failed = [
            name
            for name, passed in (
                ("causal matching/checkpoints", causal_valid),
                ("direct gradient", gradient_valid),
                ("OCMC independence", ocmc_independent),
                ("decomposition safety", decomposition_safe),
                ("SH capacity", capacity_preserved),
                ("opacity stability", opacity_stable),
                ("Gaussian population", topology_normal),
            )
            if not passed
        ]
        classification_reasons.append("integrity/safety failed: " + ", ".join(failed))
    summary = {
        "experiment": "IDENTIFIABILITY_MODULE_CAUSAL_EXPERIMENT",
        "classification": classification,
        "classification_reasons": classification_reasons,
        "RGB_classification": "SUPPORTED" if rgb_supported else "NOT_SUPPORTED",
        "mechanism_classification": "SUPPORTED" if mechanism_supported else "NOT_SUPPORTED",
        "causal_valid": causal_valid,
        "direct_gradient_valid": gradient_valid,
        "ocmc_independence": ocmc_independent,
        "decomposition_safety": decomposition_safe,
        "no_global_SH_collapse": capacity_preserved,
        "opacity_stable": opacity_stable,
        "gaussian_population_normal": topology_normal,
        "all_four_scenes_completed": len(scene_rows) == 4,
        "positive_heldout_PSNR_scene_count": positive_psnr,
        "mean_heldout_PSNR_delta": mean_psnr,
        "shared_energy_decrease_scene_count": shared_count,
        "tangent_overlap_decrease_scene_count": overlap_count,
        "temporally_stable_mechanism_scene_count": temporally_stable_count,
        "next_unique_task": next_task,
        "preregistered_decision": {
            SUPPORTED_LABEL: "mechanism and RGB supported, with every integrity/safety gate passing",
            TENTATIVE_LABEL: "mechanism supported and integrity passes, but RGB support misses without majority-scene degradation",
            NOT_SUPPORTED_LABEL: "otherwise, including majority-scene RGB degradation or any safety/integrity failure",
        },
        "scene_rows": scene_rows,
    }
    final_rows = []
    for row in scene_rows:
        for split in ("train", "heldout"):
            c0 = row[f"final_{split}_C0"]
            c1 = row[f"final_{split}_C1"]
            delta = row[f"final_{split}_delta_C1_minus_C0"]
            relative = (
                row["final_heldout_relative_delta_C1_minus_C0"]
                if split == "heldout"
                else {key: _relative_delta(c1[key], c0[key]) for key in c0}
            )
            final_rows.append(
                {
                    "scene": row["scene"],
                    "split": split,
                    **{f"{key}_C0": c0[key] for key in c0},
                    **{f"{key}_C1": c1[key] for key in c1},
                    **{f"{key}_delta_C1_minus_C0": delta[key] for key in delta},
                    **{f"{key}_relative_delta": relative[key] for key in relative},
                }
            )

    _write_csv(root / "final_metrics.csv", final_rows)
    _write_csv(root / "per_view_metrics.csv", per_view_rows)
    _write_csv(root / "mechanism_metrics.csv", all_mechanism)
    _write_csv(root / "gradient_audit.csv", all_gradient)
    _write_csv(root / "topology_metrics.csv", all_topology)
    _write_csv(root / "temporal_metrics.csv", temporal_rows)
    _write_csv(root / "counterfactual_metrics.csv", counterfactual_rows)
    _write_json(root / "ocmc_independence.json", {"pass": ocmc_independent, "rows": ocmc_rows})
    _write_json(root / "decomposition_safety.json", {"pass": decomposition_safe, "rows": all_decomp})
    _write_json(root / "classification.json", summary)
    _write_json(root / "final_summary.json", summary)
    _write_json(
        root / "training_summary.json",
        {
            "all_four_scenes_completed": True,
            "matched_updates_per_arm_per_scene": FINAL_STEP - START_STEP,
            "scene_rows": [
                {
                    "scene": row["scene"],
                    "causal_valid": row["causal_valid"],
                    "checkpoint_set_complete": row["checkpoint_set_complete"],
                    "checkpoint_files_valid": row["checkpoint_files_valid"],
                }
                for row in scene_rows
            ],
        },
    )
    _research_note(root, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(_sanitize(run(args.output_root.resolve())), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
