#!/usr/bin/env python
"""Audit SeaFree-GS vs WaterSplatting degradation/compositing semantics.

This is a read-only diagnostic. It does not train, step optimizers, edit
checkpoints, or modify renderer code. The synthetic cases use formula emulators
derived from the audited source code paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


EPS = 1e-9
CHANNELS = ("r", "g", "b")
SEAFREE_COMMIT = "7797e97dae831029ac89ae9f37b3c3d69ec2cf6c"


@dataclass(frozen=True)
class Medium:
    A: np.ndarray
    beta_d: np.ndarray
    beta_b: np.ndarray


@dataclass(frozen=True)
class Gaussian:
    c: np.ndarray
    alpha: float
    d: float
    medium: Optional[Medium] = None


@dataclass(frozen=True)
class MicroCase:
    case_id: str
    family: str
    description: str
    gaussians: Tuple[Gaussian, ...]
    pixel_medium: Medium
    ordering_pair_id: str = ""
    ordering_label: str = ""


def _arr(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"unavailable: {exc}"


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _quantile(values: Sequence[float], q: float) -> float:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def _l1(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _relative_l1(a: np.ndarray, b: np.ndarray) -> float:
    return _l1(a - b) / (_l1(b) + EPS)


def _alpha_weights(gaussians: Sequence[Gaussian]) -> Tuple[np.ndarray, float, List[float]]:
    weights: List[float] = []
    t_alpha = 1.0
    before: List[float] = []
    for gaussian in gaussians:
        before.append(t_alpha)
        alpha = min(0.999, max(0.0, float(gaussian.alpha)))
        weights.append(alpha * t_alpha)
        t_alpha *= 1.0 - alpha
    return np.asarray(weights, dtype=np.float64), float(t_alpha), before


def seafree_formula_reference(
    gaussians: Sequence[Gaussian],
    *,
    pixel_medium: Medium,
    distance_divisor: float,
    final_clamp: bool = False,
) -> Dict[str, Any]:
    """Formula emulator for SeaFree's audited per-Gaussian degradation path."""

    weights, t_final, _ = _alpha_weights(gaussians)
    direct = np.zeros(3, dtype=np.float64)
    backscatter = np.zeros(3, dtype=np.float64)
    per_gaussian: List[Dict[str, Any]] = []
    for weight, gaussian in zip(weights, gaussians):
        medium = gaussian.medium or pixel_medium
        d_eff = float(gaussian.d) / float(distance_divisor)
        t_d = np.exp(-medium.beta_d * d_eff)
        b_i = medium.A * (1.0 - np.exp(-medium.beta_b * d_eff))
        d_i = gaussian.c * t_d
        direct += weight * d_i
        backscatter += weight * b_i
        per_gaussian.append(
            {
                "weight": float(weight),
                "d_effective": d_eff,
                "T_D": t_d,
                "D_i": d_i,
                "B_i": b_i,
                "A": medium.A,
                "beta_D": medium.beta_d,
                "beta_B": medium.beta_b,
            }
        )
    background = t_final * pixel_medium.A
    rgb = direct + backscatter + background
    if final_clamp:
        rgb = np.clip(rgb, 0.0, 1.0)
    return {
        "D": direct,
        "B": backscatter,
        "BG": background,
        "I": rgb,
        "clear": np.sum(weights[:, None] * np.stack([g.c for g in gaussians], axis=0), axis=0)
        if gaussians
        else np.zeros(3, dtype=np.float64),
        "alpha": 1.0 - t_final,
        "T_alpha_final": t_final,
        "weights": weights,
        "per_gaussian": per_gaussian,
        "closure_error": float(np.max(np.abs(rgb - (direct + backscatter + background)))),
    }


def watersplatting_formula_reference(gaussians: Sequence[Gaussian], *, pixel_medium: Medium) -> Dict[str, Any]:
    """Formula emulator for WaterSplatting's audited CUDA path at one pixel."""

    weights, t_final, before = _alpha_weights(gaussians)
    direct = np.zeros(3, dtype=np.float64)
    clear = np.zeros(3, dtype=np.float64)
    medium_finite = np.zeros(3, dtype=np.float64)
    depth_weighted = 0.0
    prev_depth = 0.0
    per_gaussian: List[Dict[str, Any]] = []
    for weight, t_before, gaussian in zip(weights, before, gaussians):
        depth = float(gaussian.d)
        t_d = np.exp(-pixel_medium.beta_d * depth)
        direct += weight * gaussian.c * t_d
        clear += weight * gaussian.c
        depth_weighted += weight * depth
        exp_bs_segment = np.exp(-pixel_medium.beta_b * prev_depth) - np.exp(-pixel_medium.beta_b * depth)
        segment = t_before * exp_bs_segment * pixel_medium.A
        medium_finite += segment
        per_gaussian.append(
            {
                "weight": float(weight),
                "T_alpha_before": float(t_before),
                "depth": depth,
                "T_D": t_d,
                "D_i": gaussian.c * t_d,
                "B_segment": segment,
                "segment_depth_start": prev_depth,
                "segment_depth_end": depth,
            }
        )
        prev_depth = depth
    tail = t_final * np.exp(-pixel_medium.beta_b * prev_depth) * pixel_medium.A
    medium_total = medium_finite + tail
    rgb = direct + medium_total
    alpha = 1.0 - t_final
    expected_depth = depth_weighted / max(alpha, EPS) if gaussians else 0.0
    transmission_image = np.exp(-pixel_medium.beta_d * expected_depth)
    return {
        "D": direct,
        "B": medium_total,
        "B_finite_segments": medium_finite,
        "BG": tail,
        "I": rgb,
        "clear": clear,
        "alpha": alpha,
        "T_alpha_final": t_final,
        "expected_depth": expected_depth,
        "transmission_image": transmission_image,
        "weights": weights,
        "per_gaussian": per_gaussian,
        "closure_error": float(np.max(np.abs(rgb - (direct + medium_total)))),
    }


def composite_then_degrade_reference(gaussians: Sequence[Gaussian], *, pixel_medium: Medium) -> Dict[str, Any]:
    ws = watersplatting_formula_reference(gaussians, pixel_medium=pixel_medium)
    clear = ws["clear"]
    depth = ws["expected_depth"]
    direct = clear * np.exp(-pixel_medium.beta_d * depth)
    # This medium term intentionally mirrors a simple image-space depth model,
    # not either audited implementation.
    medium = pixel_medium.A * (1.0 - np.exp(-pixel_medium.beta_b * depth))
    rgb = direct + medium
    return {
        "D": direct,
        "B": medium,
        "BG": np.zeros(3, dtype=np.float64),
        "I": rgb,
        "clear": clear,
        "expected_depth": depth,
    }


def _default_medium() -> Medium:
    return Medium(A=_arr([0.22, 0.46, 0.72]), beta_d=_arr([0.70, 0.50, 0.35]), beta_b=_arr([0.36, 0.24, 0.16]))


def _medium_variant(index: int) -> Medium:
    variants = (
        Medium(A=_arr([0.20, 0.45, 0.72]), beta_d=_arr([0.62, 0.46, 0.34]), beta_b=_arr([0.30, 0.22, 0.16])),
        Medium(A=_arr([0.26, 0.50, 0.68]), beta_d=_arr([0.86, 0.58, 0.38]), beta_b=_arr([0.46, 0.31, 0.19])),
        Medium(A=_arr([0.18, 0.42, 0.77]), beta_d=_arr([0.52, 0.43, 0.31]), beta_b=_arr([0.26, 0.21, 0.14])),
    )
    return variants[index % len(variants)]


def _rgb(v: float) -> np.ndarray:
    return _arr([v, v, v])


def build_microcases() -> Tuple[List[MicroCase], List[MicroCase], List[MicroCase]]:
    medium = _default_medium()
    single: List[MicroCase] = []
    for opacity_name, opacity in (("low", 0.1), ("mid", 0.5), ("high", 0.9)):
        for depth_name, depth in (("near", 0.5), ("mid", 2.0), ("far", 5.0)):
            for color_name, color in (("dark", 0.1), ("mid", 0.5), ("bright", 0.95)):
                single.append(
                    MicroCase(
                        case_id=f"single_{opacity_name}_{depth_name}_{color_name}",
                        family="single",
                        description=f"1 Gaussian, opacity={opacity}, depth={depth}, intrinsic={color}",
                        gaussians=(Gaussian(c=_rgb(color), alpha=opacity, d=depth),),
                        pixel_medium=medium,
                    )
                )

    two_specs: List[Tuple[str, str, Sequence[float], Sequence[Sequence[float]], Sequence[float], str, str]] = [
        ("same_depth_same_color", "same depth and same color", (2.0, 2.0), (_rgb(0.5), _rgb(0.5)), (0.5, 0.5), "", ""),
        ("small_depth_separation", "small depth separation", (2.0, 2.3), (_rgb(0.5), _rgb(0.5)), (0.5, 0.5), "", ""),
        ("large_depth_separation", "large depth separation", (1.0, 5.0), (_rgb(0.5), _rgb(0.5)), (0.5, 0.5), "", ""),
        ("front_bright", "near Gaussian bright, far Gaussian mid", (1.0, 5.0), (_rgb(0.95), _rgb(0.5)), (0.5, 0.5), "bright_mid_order", "bright_front"),
        ("back_bright", "near Gaussian mid, far Gaussian bright", (1.0, 5.0), (_rgb(0.5), _rgb(0.95)), (0.5, 0.5), "bright_mid_order", "bright_back"),
        ("high_opacity_overlap", "two high-opacity Gaussians", (1.0, 5.0), (_rgb(0.5), _rgb(0.75)), (0.9, 0.9), "", ""),
        ("low_opacity_overlap", "two low-opacity Gaussians", (1.0, 5.0), (_rgb(0.5), _rgb(0.75)), (0.1, 0.1), "", ""),
        ("front_low_back_high", "front low opacity, back high opacity", (1.0, 5.0), (_rgb(0.5), _rgb(0.95)), (0.1, 0.9), "", ""),
        ("front_high_back_low", "front high opacity, back low opacity", (1.0, 5.0), (_rgb(0.95), _rgb(0.5)), (0.9, 0.1), "", ""),
    ]
    two: List[MicroCase] = []
    for case_id, desc, depths, colors, alphas, pair_id, pair_label in two_specs:
        two.append(
            MicroCase(
                case_id=f"two_{case_id}",
                family="two",
                description=desc,
                gaussians=tuple(
                    Gaussian(c=np.asarray(color, dtype=np.float64), alpha=float(alpha), d=float(depth))
                    for color, alpha, depth in zip(colors, alphas, depths)
                ),
                pixel_medium=medium,
                ordering_pair_id=pair_id,
                ordering_label=pair_label,
            )
        )

    three: List[MicroCase] = [
        MicroCase(
            case_id="three_uniform_mid_bright_dark",
            family="three",
            description="near/mid/far, uniform opacity, mid/bright/dark colors",
            gaussians=(
                Gaussian(c=_rgb(0.5), alpha=0.45, d=0.8),
                Gaussian(c=_rgb(0.95), alpha=0.45, d=2.2),
                Gaussian(c=_rgb(0.1), alpha=0.45, d=5.0),
            ),
            pixel_medium=medium,
        ),
        MicroCase(
            case_id="three_mixed_opacity_bright_far",
            family="three",
            description="near/mid/far, mixed opacity, far bright",
            gaussians=(
                Gaussian(c=_rgb(0.35), alpha=0.2, d=0.8),
                Gaussian(c=_rgb(0.55), alpha=0.55, d=2.2),
                Gaussian(c=_rgb(0.95), alpha=0.85, d=5.0),
            ),
            pixel_medium=medium,
        ),
        MicroCase(
            case_id="three_high_opacity_bright_near",
            family="three",
            description="near bright with high-opacity overlap behind it",
            gaussians=(
                Gaussian(c=_rgb(0.95), alpha=0.85, d=0.8),
                Gaussian(c=_rgb(0.5), alpha=0.75, d=2.2),
                Gaussian(c=_rgb(0.2), alpha=0.65, d=5.0),
            ),
            pixel_medium=medium,
        ),
    ]
    return single, two, three


def _with_per_gaussian_medium(case: MicroCase) -> MicroCase:
    return MicroCase(
        case_id=case.case_id,
        family=case.family,
        description=case.description + " with controlled per-Gaussian medium mismatch",
        gaussians=tuple(
            Gaussian(c=g.c, alpha=g.alpha, d=g.d, medium=_medium_variant(i)) for i, g in enumerate(case.gaussians)
        ),
        pixel_medium=case.pixel_medium,
        ordering_pair_id=case.ordering_pair_id,
        ordering_label=case.ordering_label,
    )


def _case_summary(case: MicroCase) -> Dict[str, Any]:
    colors = np.stack([g.c for g in case.gaussians], axis=0)
    depths = np.asarray([g.d for g in case.gaussians], dtype=np.float64)
    alphas = np.asarray([g.alpha for g in case.gaussians], dtype=np.float64)
    return {
        "case_id": case.case_id,
        "family": case.family,
        "description": case.description,
        "n_gaussians": len(case.gaussians),
        "max_intrinsic": float(colors.max()),
        "mean_intrinsic": float(colors.mean()),
        "depth_min": float(depths.min()),
        "depth_max": float(depths.max()),
        "depth_span": float(depths.max() - depths.min()),
        "opacity_mean": float(alphas.mean()),
        "opacity_max": float(alphas.max()),
        "ordering_pair_id": case.ordering_pair_id,
        "ordering_label": case.ordering_label,
    }


def _evaluate_case(case: MicroCase, mode: str, distance_divisor: float, per_gaussian_medium: bool) -> Dict[str, Any]:
    actual = _with_per_gaussian_medium(case) if per_gaussian_medium else case
    sf = seafree_formula_reference(
        actual.gaussians,
        pixel_medium=actual.pixel_medium,
        distance_divisor=distance_divisor,
        final_clamp=False,
    )
    ws = watersplatting_formula_reference(actual.gaussians, pixel_medium=actual.pixel_medium)
    post = composite_then_degrade_reference(actual.gaussians, pixel_medium=actual.pixel_medium)
    row: Dict[str, Any] = {
        **_case_summary(case),
        "mode": mode,
        "seafree_distance_divisor": distance_divisor,
        "per_gaussian_medium_mismatch": bool(per_gaussian_medium),
        "SF_I": sf["I"].tolist(),
        "WS_I": ws["I"].tolist(),
        "SF_D": sf["D"].tolist(),
        "WS_D": ws["D"].tolist(),
        "SF_B_without_BG": sf["B"].tolist(),
        "SF_MEDIUM_TOTAL": (sf["B"] + sf["BG"]).tolist(),
        "WS_B": ws["B"].tolist(),
        "SF_BG": sf["BG"].tolist(),
        "WS_BG_tail": ws["BG"].tolist(),
        "DELTA_I": (ws["I"] - sf["I"]).tolist(),
        "DELTA_D": (ws["D"] - sf["D"]).tolist(),
        "DELTA_B_WITHOUT_BG": (ws["B"] - sf["B"]).tolist(),
        "DELTA_MEDIUM_TOTAL": (ws["B"] - (sf["B"] + sf["BG"])).tolist(),
        "COMPOSITING_DISAGREEMENT": _relative_l1(ws["I"], sf["I"]),
        "DIRECT_DISAGREEMENT": _relative_l1(ws["D"], sf["D"]),
        "MEDIUM_TOTAL_DISAGREEMENT": _relative_l1(ws["B"], sf["B"] + sf["BG"]),
        "DELTA_I_L1": _l1(ws["I"] - sf["I"]),
        "DELTA_D_L1": _l1(ws["D"] - sf["D"]),
        "DELTA_MEDIUM_TOTAL_L1": _l1(ws["B"] - (sf["B"] + sf["BG"])),
        "J_times_T_direct_L1": _l1(post["D"] - ws["D"]),
        "J_times_T_direct_relative": _relative_l1(post["D"], ws["D"]),
        "closure_SF_max": sf["closure_error"],
        "closure_WS_max": ws["closure_error"],
        "WS_expected_depth": float(ws["expected_depth"]),
        "WS_image_transmission": ws["transmission_image"].tolist(),
        "SF_alpha": float(sf["alpha"]),
        "WS_alpha": float(ws["alpha"]),
    }
    return row


def run_microcases(output_dir: Path, render_dir: Path, tile_width: int) -> Dict[str, Any]:
    single, two, three = build_microcases()
    all_cases = single + two + three
    rows: List[Dict[str, Any]] = []
    for case in single:
        rows.append(_evaluate_case(case, "scale_aligned_constant_medium", 1.0, False))
        rows.append(_evaluate_case(case, "source_native_distance_div10", 10.0, False))
    for case in two + three:
        rows.append(_evaluate_case(case, "scale_aligned_constant_medium", 1.0, False))
        rows.append(_evaluate_case(case, "source_native_distance_div10", 10.0, False))
        rows.append(_evaluate_case(case, "scale_aligned_per_gaussian_medium_mismatch", 1.0, True))

    single_rows = [r for r in rows if r["family"] == "single"]
    two_rows = [r for r in rows if r["family"] == "two"]
    three_rows = [r for r in rows if r["family"] == "three"]
    _write_json(output_dir / "single_gaussian_microcases.json", single_rows)
    _write_csv(output_dir / "single_gaussian_microcases.csv", single_rows)
    _write_json(output_dir / "two_gaussian_microcases.json", two_rows)
    _write_csv(output_dir / "two_gaussian_microcases.csv", two_rows)
    _write_json(output_dir / "three_gaussian_microcases.json", three_rows)
    _write_csv(output_dir / "three_gaussian_microcases.csv", three_rows)

    sensitivity = compute_sensitivity(rows)
    for name, data in sensitivity.items():
        _write_json(output_dir / f"{name}.json", data)
        _write_csv(output_dir / f"{name}.csv", data["rows"])

    ordering_rows = compute_ordering_sensitivity(rows)
    _write_json(output_dir / "ordering_sensitivity.json", ordering_rows)
    _write_csv(output_dir / "ordering_sensitivity.csv", ordering_rows)

    aa_rows = run_aa_opacity_interaction()
    _write_json(output_dir / "aa_opacity_interaction.json", aa_rows)
    _write_csv(output_dir / "aa_opacity_interaction.csv", aa_rows)

    jt_rows, counterexample = run_jt_invalidity_audit(all_cases)
    _write_json(output_dir / "jt_invalidity_audit.json", {"rows": jt_rows, "minimal_counterexample": counterexample})
    _write_csv(output_dir / "jt_invalidity_audit.csv", jt_rows)
    _write_json(output_dir / "non_commutativity_counterexample.json", counterexample)

    closure_rows = run_closure_audit(rows)
    _write_json(output_dir / "closure_audit.json", closure_rows)
    _write_csv(output_dir / "closure_audit.csv", closure_rows)

    visual_paths = render_microcase_sheets(render_dir, tile_width)

    return {
        "rows": rows,
        "sensitivity": sensitivity,
        "ordering": ordering_rows,
        "aa": aa_rows,
        "jt": jt_rows,
        "counterexample": counterexample,
        "closure": closure_rows,
        "visual_paths": visual_paths,
    }


def _select_rows(rows: Sequence[Mapping[str, Any]], *, mode: str) -> List[Mapping[str, Any]]:
    return [r for r in rows if r["mode"] == mode and float(r["COMPOSITING_DISAGREEMENT"]) >= 1e-12]


def _amplification_rows(rows: Sequence[Mapping[str, Any]], mode: str) -> List[Dict[str, Any]]:
    selected = _select_rows(rows, mode=mode)
    bright = [float(r["COMPOSITING_DISAGREEMENT"]) for r in selected if float(r["max_intrinsic"]) >= 0.9]
    normal = [float(r["COMPOSITING_DISAGREEMENT"]) for r in selected if float(r["max_intrinsic"]) <= 0.5]
    large_depth = [float(r["COMPOSITING_DISAGREEMENT"]) for r in selected if float(r["depth_span"]) >= 2.0]
    small_depth = [float(r["COMPOSITING_DISAGREEMENT"]) for r in selected if float(r["depth_span"]) <= 0.3]
    high_opacity = [float(r["COMPOSITING_DISAGREEMENT"]) for r in selected if float(r["opacity_mean"]) >= 0.75]
    low_opacity = [float(r["COMPOSITING_DISAGREEMENT"]) for r in selected if float(r["opacity_mean"]) <= 0.25]
    return [
        {
            "mode": mode,
            "metric": "BRIGHT_AMPLIFICATION",
            "numerator_mean": _mean(bright),
            "denominator_mean": _mean(normal),
            "amplification": _mean(bright) / (_mean(normal) + EPS),
            "numerator_count": len(bright),
            "denominator_count": len(normal),
        },
        {
            "mode": mode,
            "metric": "DEPTH_SEPARATION_AMPLIFICATION",
            "numerator_mean": _mean(large_depth),
            "denominator_mean": _mean(small_depth),
            "amplification": _mean(large_depth) / (_mean(small_depth) + EPS),
            "numerator_count": len(large_depth),
            "denominator_count": len(small_depth),
        },
        {
            "mode": mode,
            "metric": "OPACITY_AMPLIFICATION",
            "numerator_mean": _mean(high_opacity),
            "denominator_mean": _mean(low_opacity),
            "amplification": _mean(high_opacity) / (_mean(low_opacity) + EPS),
            "numerator_count": len(high_opacity),
            "denominator_count": len(low_opacity),
        },
    ]


def compute_sensitivity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    modes = sorted({str(r["mode"]) for r in rows})
    all_amp_rows: List[Dict[str, Any]] = []
    for mode in modes:
        all_amp_rows.extend(_amplification_rows(rows, mode))
    return {
        "brightness_sensitivity": {"rows": [r for r in all_amp_rows if r["metric"] == "BRIGHT_AMPLIFICATION"]},
        "depth_sensitivity": {"rows": [r for r in all_amp_rows if r["metric"] == "DEPTH_SEPARATION_AMPLIFICATION"]},
        "opacity_sensitivity": {"rows": [r for r in all_amp_rows if r["metric"] == "OPACITY_AMPLIFICATION"]},
    }


def compute_ordering_sensitivity(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str], Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        pair_id = str(row.get("ordering_pair_id") or "")
        label = str(row.get("ordering_label") or "")
        if not pair_id or not label:
            continue
        by_key.setdefault((str(row["mode"]), pair_id), {})[label] = row
    out: List[Dict[str, Any]] = []
    for (mode, pair_id), values in sorted(by_key.items()):
        if len(values) < 2:
            continue
        labels = sorted(values)
        a = values[labels[0]]
        b = values[labels[1]]
        sf_a = np.asarray(a["SF_I"], dtype=np.float64)
        sf_b = np.asarray(b["SF_I"], dtype=np.float64)
        ws_a = np.asarray(a["WS_I"], dtype=np.float64)
        ws_b = np.asarray(b["WS_I"], dtype=np.float64)
        sf_sens = _l1(sf_a - sf_b)
        ws_sens = _l1(ws_a - ws_b)
        out.append(
            {
                "mode": mode,
                "ordering_pair_id": pair_id,
                "label_a": labels[0],
                "label_b": labels[1],
                "ORDER_SENSITIVITY_SF": sf_sens,
                "ORDER_SENSITIVITY_WS": ws_sens,
                "ORDERING_SEMANTICS_GAP": abs(ws_sens - sf_sens),
            }
        )
    return out


def run_aa_opacity_interaction() -> List[Dict[str, Any]]:
    _, two, _ = build_microcases()
    selected = [case for case in two if case.case_id in {"two_front_bright", "two_back_bright", "two_high_opacity_overlap", "two_low_opacity_overlap"}]
    rows: List[Dict[str, Any]] = []
    for case in selected:
        for comp_factor in (1.0, 0.75, 0.50):
            adjusted = MicroCase(
                case_id=case.case_id,
                family=case.family,
                description=case.description,
                gaussians=tuple(
                    Gaussian(c=g.c, alpha=min(0.999, g.alpha * comp_factor), d=g.d, medium=g.medium) for g in case.gaussians
                ),
                pixel_medium=case.pixel_medium,
                ordering_pair_id=case.ordering_pair_id,
                ordering_label=case.ordering_label,
            )
            row = _evaluate_case(adjusted, "scale_aligned_per_gaussian_medium_mismatch", 1.0, True)
            rows.append(
                {
                    "case_id": case.case_id,
                    "controlled_opacity_compensation_factor": comp_factor,
                    "effective_opacity_mean": row["opacity_mean"],
                    "COMPOSITING_DISAGREEMENT": row["COMPOSITING_DISAGREEMENT"],
                    "DELTA_I_L1": row["DELTA_I_L1"],
                    "note": "controlled sensitivity test; not an actual AA renderer call",
                }
            )
    return rows


def run_jt_invalidity_audit(cases: Sequence[MicroCase]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        ws = watersplatting_formula_reference(case.gaussians, pixel_medium=case.pixel_medium)
        post = composite_then_degrade_reference(case.gaussians, pixel_medium=case.pixel_medium)
        rows.append(
            {
                **_case_summary(case),
                "direct_object_signal": ws["D"].tolist(),
                "clear_times_image_transmission": post["D"].tolist(),
                "mean_abs_discrepancy": _l1(post["D"] - ws["D"]),
                "max_abs_discrepancy": float(np.max(np.abs(post["D"] - ws["D"]))),
                "root_cause": "alpha-weighted per-Gaussian T_D differs from applying one T_D at alpha-weighted expected depth",
            }
        )

    medium = Medium(A=_arr([0.0, 0.0, 0.0]), beta_d=_arr([0.8, 0.8, 0.8]), beta_b=_arr([0.0, 0.0, 0.0]))
    counter_case = MicroCase(
        case_id="minimal_two_gaussian_jt_counterexample",
        family="counterexample",
        description="Two Gaussians with different depths and colors; no medium term.",
        gaussians=(
            Gaussian(c=_rgb(1.0), alpha=0.5, d=1.0),
            Gaussian(c=_rgb(0.2), alpha=0.5, d=5.0),
        ),
        pixel_medium=medium,
    )
    ws = watersplatting_formula_reference(counter_case.gaussians, pixel_medium=medium)
    post = composite_then_degrade_reference(counter_case.gaussians, pixel_medium=medium)
    counterexample = {
        "case": {
            "alpha1": 0.5,
            "alpha2": 0.5,
            "c1": [1.0, 1.0, 1.0],
            "c2": [0.2, 0.2, 0.2],
            "d1": 1.0,
            "d2": 5.0,
            "beta_D": [0.8, 0.8, 0.8],
            "A": [0.0, 0.0, 0.0],
        },
        "degrade_then_composite_direct": ws["D"].tolist(),
        "composite_then_degrade_direct": post["D"].tolist(),
        "mean_abs_difference": _l1(ws["D"] - post["D"]),
        "operation_commutes": bool(np.allclose(ws["D"], post["D"], atol=1e-8)),
    }
    return rows, counterexample


def run_closure_audit(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["mode"]), []).append(row)
    out: List[Dict[str, Any]] = []
    for mode, mode_rows in sorted(grouped.items()):
        for implementation, key in (("SeaFree formula reference", "closure_SF_max"), ("WaterSplatting formula reference", "closure_WS_max")):
            vals = [float(r[key]) for r in mode_rows]
            out.append(
                {
                    "mode": mode,
                    "implementation": implementation,
                    "mean_error": _mean(vals),
                    "p99_error": _quantile(vals, 0.99),
                    "max_error": max(vals) if vals else float("nan"),
                }
            )
    return out


def _rgb_swatch(rgb: Sequence[float], width: int, height: int) -> Image.Image:
    arr = np.asarray(rgb, dtype=np.float64)
    arr = np.clip(arr, 0.0, 1.0)
    color = tuple(int(round(v * 255.0)) for v in arr)
    return Image.new("RGB", (width, height), color)


def _abs_delta_swatch(delta: Sequence[float], width: int, height: int, scale: float = 0.5) -> Image.Image:
    arr = np.abs(np.asarray(delta, dtype=np.float64)) / max(scale, EPS)
    arr = np.clip(arr, 0.0, 1.0)
    color = tuple(int(round(v * 255.0)) for v in arr)
    return Image.new("RGB", (width, height), color)


def _signed_delta_swatch(delta: Sequence[float], width: int, height: int, scale: float = 0.5) -> Image.Image:
    arr = np.asarray(delta, dtype=np.float64)
    red = float(np.mean(np.clip(arr, 0.0, scale) / scale))
    blue = float(np.mean(np.clip(-arr, 0.0, scale) / scale))
    color_arr = np.asarray([red, 0.0, blue], dtype=np.float64)
    color = tuple(int(round(float(v) * 255.0)) for v in np.clip(color_arr, 0.0, 1.0))
    return Image.new("RGB", (width, height), color)


def _tile(image: Image.Image, label: str, width: int) -> Image.Image:
    if image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.NEAREST)
    label_h = 28
    canvas = Image.new("RGB", (image.width, image.height + label_h), "white")
    canvas.paste(image, (0, label_h))
    ImageDraw.Draw(canvas).text((6, 7), label, fill="black")
    return canvas


def _save_sheet(path: Path, rows: Sequence[Sequence[Tuple[str, Image.Image]]], tile_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered_rows: List[Image.Image] = []
    for row in rows:
        tiles = [_tile(img, label, tile_width) for label, img in row]
        w = sum(tile.width for tile in tiles) + 6 * (len(tiles) - 1)
        h = max(tile.height for tile in tiles)
        canvas = Image.new("RGB", (w, h), "white")
        x = 0
        for tile in tiles:
            canvas.paste(tile, (x, 0))
            x += tile.width + 6
        rendered_rows.append(canvas)
    w = max(row.width for row in rendered_rows)
    h = sum(row.height for row in rendered_rows) + 6 * (len(rendered_rows) - 1)
    sheet = Image.new("RGB", (w, h), "white")
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(path)


def _microcase_visual_row(case: MicroCase, mode: str, divisor: float, mismatch: bool) -> List[Tuple[str, Image.Image]]:
    row = _evaluate_case(case, mode, divisor, mismatch)
    sf_i = np.asarray(row["SF_I"], dtype=np.float64)
    ws_i = np.asarray(row["WS_I"], dtype=np.float64)
    delta = ws_i - sf_i
    h = 100
    return [
        (f"{case.case_id} SF", _rgb_swatch(sf_i, 160, h)),
        ("WS", _rgb_swatch(ws_i, 160, h)),
        ("abs delta scale0.5", _abs_delta_swatch(delta, 160, h, 0.5)),
        ("signed delta +/-0.5", _signed_delta_swatch(delta, 160, h, 0.5)),
    ]


def render_microcase_sheets(render_dir: Path, tile_width: int) -> List[str]:
    single, two, _ = build_microcases()
    lookup = {case.case_id: case for case in single + two}
    visual_specs = [
        ("single_gaussian_comparison.png", ["single_mid_mid_bright", "single_low_near_dark", "single_high_far_mid"]),
        ("two_gaussian_depth_ordering.png", ["two_same_depth_same_color", "two_large_depth_separation"]),
        ("front_bright.png", ["two_front_bright"]),
        ("back_bright.png", ["two_back_bright"]),
        ("high_opacity_overlap.png", ["two_high_opacity_overlap"]),
        ("low_opacity_overlap.png", ["two_low_opacity_overlap"]),
    ]
    paths: List[str] = []
    for filename, case_ids in visual_specs:
        rows = []
        for case_id in case_ids:
            case = lookup[case_id]
            rows.append(_microcase_visual_row(case, "source_native_distance_div10", 10.0, False))
            rows.append(_microcase_visual_row(case, "scale_aligned_per_gaussian_medium_mismatch", 1.0, True))
        path = render_dir / filename
        _save_sheet(path, rows, tile_width)
        paths.append(str(path))
    return paths


def source_semantics() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    seafree = {
        "source_commit": SEAFREE_COMMIT,
        "code_facts": [
            {
                "fact": "The registered SeaFree-GS method sets rasterize_mode='antialiased' and sh_degree=0.",
                "source": "seafree_gs/seafree_config.py:32-36",
            },
            {
                "fact": "With sh_degree=0, Gaussian intrinsic color is sigmoid(features_dc) before water degradation.",
                "source": "seafree_gs/seafree_model.py:595-599",
            },
            {
                "fact": "Gaussian LOS distance is Euclidean norm from camera center to Gaussian mean, detached, then divided by 10.",
                "source": "seafree_gs/seafree_model.py:601-608,652",
            },
            {
                "fact": "Water predictor outputs ambient A via sigmoid and beta_B/beta_D via softplus.",
                "source": "seafree_gs/seafree_model.py:638-650",
            },
            {
                "fact": "Degraded Gaussian color is c_i exp(-beta_D,i d_i/10) + A_i(1-exp(-beta_B,i d_i/10)) before rasterization.",
                "source": "seafree_gs/seafree_model.py:652-656",
            },
            {
                "fact": "Degraded and intrinsic colors are concatenated and rasterized together as six channels.",
                "source": "seafree_gs/seafree_model.py:656,663-683,694-695",
            },
            {
                "fact": "Final underwater RGB adds (1-alpha_image) * pixel_ambient_light_colors after rasterization, then clamps to [0,1].",
                "source": "seafree_gs/seafree_model.py:658-660,691-692",
            },
            {
                "fact": "gsplat antialiasing multiplies opacities by projection compensations before rasterize_to_pixels.",
                "source": "third_party/gsplat/gsplat/rendering.py:323-361",
            },
            {
                "fact": "gsplat rasterization uses alpha=min(0.999, opacity*exp(-sigma)), w_i=alpha_i*T_alpha_before_i, and stores final alpha=1-T_alpha.",
                "source": "third_party/gsplat/gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu:148-184",
            },
        ],
        "inferred_formula": {
            "weights": "w_i = alpha_i prod_{j<i}(1-alpha_j)",
            "D_SF": "sum_i w_i c_i exp(-beta_D,i d_i/10)",
            "B_SF": "sum_i w_i A_i(1-exp(-beta_B,i d_i/10))",
            "BG_SF": "T_alpha_final A_pixel",
            "I_SF": "D_SF + B_SF + BG_SF, followed by clamp in code",
        },
    }
    ws = {
        "code_facts": [
            {
                "fact": "Current-view Gaussian color is computed before rasterization; legacy SH uses clamp(SH+0.5,min=0), bounded modes use sigmoid/headroom helpers.",
                "source": "water_splatting/water_splatting.py:1379-1411; water_splatting/fields/gaussian_appearance.py:33-139",
            },
            {
                "fact": "Medium field predicts per-pixel medium_rgb via sigmoid and medium_bs/medium_attn via softplus(raw + density_bias); tied b_inf equals medium_rgb.",
                "source": "water_splatting/fields/medium_field.py:122-149",
            },
            {
                "fact": "Projected depth is camera-space z from project_gaussians, not SeaFree's Euclidean LOS norm.",
                "source": "water_splatting/cuda/csrc/forward.cu:40-96",
            },
            {
                "fact": "Classic opacity is sigmoid(opacity); antialiased opacity multiplies by projection compensation before underwater rasterization.",
                "source": "water_splatting/water_splatting.py:1415-1422",
            },
            {
                "fact": "Direct object signal is accumulated inside CUDA as sum_i alpha_i T_alpha_before_i c_i exp(-medium_attn_pixel depth_i).",
                "source": "water_splatting/cuda/csrc/forward.cu:453-466",
            },
            {
                "fact": "Clear object raw is accumulated as sum_i alpha_i T_alpha_before_i c_i.",
                "source": "water_splatting/cuda/csrc/forward.cu:460-466",
            },
            {
                "fact": "Backscatter/medium is accumulated over ray segments using T_alpha_before_i * A_pixel * (exp(-beta_B prev_depth)-exp(-beta_B depth_i)), then a tail T_alpha_final*A_pixel*exp(-beta_B last_depth).",
                "source": "water_splatting/cuda/csrc/forward.cu:473-508",
            },
            {
                "fact": "UnderwaterRasterizer returns rgb = rgb_object + rgb_medium; WaterSplatting exposes pred_image, direct_object_signal, rgb_medium, clear_object_fullsh_raw, transmission and tau_D.",
                "source": "water_splatting/rendering/underwater_rasterizer.py:124-158; water_splatting/water_splatting.py:1459-1495",
            },
        ],
        "inferred_formula": {
            "weights": "w_i = alpha_i prod_{j<i}(1-alpha_j)",
            "D_WS": "sum_i w_i c_i exp(-beta_D,pixel d_i)",
            "B_WS": "sum_i T_alpha_before_i A_pixel(exp(-beta_B,pixel d_{i-1})-exp(-beta_B,pixel d_i)) + T_alpha_final A_pixel exp(-beta_B,pixel d_N)",
            "I_WS": "D_WS + B_WS",
        },
    }
    equivalence = {
        "ARE_SEAFREE_AND_WS_DEGRADATION_COMPOSITING_EQUIVALENT": "EQUIVALENT_UNDER_RESTRICTED_CONDITIONS",
        "conditions": [
            "same alpha weights and sorted front-to-back order",
            "SeaFree d_i/10 and WaterSplatting d_i are numerically aligned, or beta values compensate the distance scale exactly",
            "SeaFree per-Gaussian A_i, beta_D,i, beta_B,i equal WaterSplatting per-pixel A, beta_D, beta_B for all contributors to the pixel",
            "same intrinsic c_i after activation/SH evaluation",
            "no final clamp saturation difference",
        ],
        "non_equivalence_causes_in_actual_code": [
            "SeaFree queries water properties per Gaussian LOS direction and per pixel; WaterSplatting uses per-pixel medium for all Gaussian contributors.",
            "SeaFree uses Euclidean LOS distance divided by 10; WaterSplatting uses projected camera z-depth in the underwater CUDA path.",
            "SeaFree clamps final RGB to [0,1]; WaterSplatting exposes rgb_object + rgb_medium without an equivalent final clamp in get_outputs.",
            "The code organization differs: SeaFree pre-degrades Gaussian colors, while WaterSplatting accumulates direct and medium terms inside the rasterizer.",
        ],
        "restricted_equivalence_note": "For aligned constant medium, the WaterSplatting segment-plus-tail medium expression simplifies to A*(1-sum_i w_i exp(-beta_B d_i)), matching SeaFree's per-Gaussian backscatter plus residual background.",
    }
    return seafree, ws, equivalence


def write_symbolic_outputs(output_dir: Path) -> None:
    single_md = """# Symbolic Single-Gaussian Equations

Let `T_alpha_final = 1-alpha1`, `T_D1 = exp(-beta_D d1)`, and `E_B1 = exp(-beta_B d1)`.

## SeaFree Formula Reference

`D_SF = alpha1 * c1 * T_D1`

`B_SF = alpha1 * A * (1 - E_B1)`

`BG_SF = (1 - alpha1) * A`

`I_SF = D_SF + B_SF + BG_SF`

## WaterSplatting Formula Reference

`D_WS = alpha1 * c1 * T_D1`

`B_WS = A * (1 - E_B1) + (1 - alpha1) * A * E_B1`

`I_WS = D_WS + B_WS`

Under constant aligned medium, `B_WS = A * (1 - alpha1 * E_B1)`, and `B_SF + BG_SF` has the same value.
"""
    two_md = """# Symbolic Two-Gaussian Equations

Let `w1 = alpha1`, `w2 = (1-alpha1) alpha2`, and `T_alpha_final = (1-alpha1)(1-alpha2)`.

## SeaFree Formula Reference

`D_SF = w1 c1 exp(-beta_D,1 d1) + w2 c2 exp(-beta_D,2 d2)`

`B_SF = w1 A1(1-exp(-beta_B,1 d1)) + w2 A2(1-exp(-beta_B,2 d2))`

`BG_SF = T_alpha_final A_pixel`

## WaterSplatting Formula Reference

`D_WS = w1 c1 exp(-beta_D,pixel d1) + w2 c2 exp(-beta_D,pixel d2)`

`B_WS = A_pixel(1-exp(-beta_B,pixel d1)) + (1-alpha1)A_pixel(exp(-beta_B,pixel d1)-exp(-beta_B,pixel d2)) + T_alpha_final A_pixel exp(-beta_B,pixel d2)`

For constant aligned `A` and `beta_B`, `B_WS = A(1 - w1 exp(-beta_B d1) - w2 exp(-beta_B d2))`, which equals `B_SF + BG_SF`.
"""
    eq_md = """# Equivalence Conditions

`I_SF = I_WS` is exact only under restricted conditions:

- identical alpha weights and front-to-back order;
- SeaFree normalized LOS distance and WaterSplatting depth are numerically aligned;
- per-Gaussian SeaFree medium values equal WaterSplatting's per-pixel medium values for every contributor;
- intrinsic Gaussian colors are already matched;
- final clamp does not change the SeaFree result.

Outside those conditions, actual source-code semantics are structurally different because the medium query domain, distance definition, distance scale, final clamp, and code organization differ.
"""
    (output_dir / "symbolic_single_gaussian.md").write_text(single_md, encoding="utf8")
    (output_dir / "symbolic_two_gaussian.md").write_text(two_md, encoding="utf8")
    (output_dir / "equivalence_conditions.md").write_text(eq_md, encoding="utf8")
    _write_json(
        output_dir / "symbolic_single_gaussian.json",
        {"equations": single_md, "source": "formula emulator derived from audited code"},
    )
    _write_json(
        output_dir / "symbolic_two_gaussian.json",
        {"equations": two_md, "source": "formula emulator derived from audited code"},
    )


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def run_panama_region_alignment(repo: Path, output_dir: Path, render_dir: Path, tile_width: int, skip: bool) -> Dict[str, Any]:
    if skip:
        result = {"status": "SKIPPED", "reason": "disabled by --skip-panama-forward", "rows": []}
        _write_json(output_dir / "panama_region_alignment.json", result)
        _write_csv(output_dir / "panama_region_alignment.csv", [])
        return result
    if not _torch_available():
        result = {"status": "SKIPPED", "reason": "torch is unavailable in this Python environment", "rows": []}
        _write_json(output_dir / "panama_region_alignment.json", result)
        _write_csv(output_dir / "panama_region_alignment.csv", [])
        return result

    try:
        return _run_panama_region_alignment_impl(repo, output_dir, render_dir, tile_width)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        result = {"status": "SKIPPED", "reason": f"Panama forward audit failed: {exc}", "rows": []}
        _write_json(output_dir / "panama_region_alignment.json", result)
        _write_csv(output_dir / "panama_region_alignment.csv", [])
        return result


def _run_panama_region_alignment_impl(repo: Path, output_dir: Path, render_dir: Path, tile_width: int) -> Dict[str, Any]:
    import gc

    import torch
    import torch.nn.functional as F
    from nerfstudio.utils.eval_utils import eval_setup

    runs = {
        "M1": {
            "config": repo
            / "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml",
            "parameterization": "legacy",
            "rasterize_mode": "classic",
        },
        "K1": {
            "config": repo
            / "outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/config.yml",
            "parameterization": "bounded_sh3",
            "rasterize_mode": "classic",
        },
        "AA": {
            "config": repo / "outputs/bnd_aa_panama_20260810/panama_bnd_aa_seed42_step0_to_15000/water-splatting/20260810_bnd_aa/config.yml",
            "parameterization": "bounded_sh3",
            "rasterize_mode": "antialiased",
        },
    }

    def actual_step(config_path: Path, nominal: int) -> int:
        ckpt_dir = config_path.parent / "nerfstudio_models"
        steps = []
        for path in ckpt_dir.glob("step-*.ckpt"):
            try:
                steps.append(int(path.stem.split("-")[1]))
            except Exception:
                continue
        if nominal in steps:
            return nominal
        if nominal == 15000 and 14999 in steps:
            return 14999
        raise FileNotFoundError(f"missing step {nominal} checkpoint for {config_path}")

    def load_run(name: str) -> Tuple[Any, Any, Path, int]:
        spec = runs[name]
        config_path = spec["config"]
        step = actual_step(config_path, 15000)

        def update_config(config: Any) -> Any:
            config.load_step = step
            return config

        config, pipeline, checkpoint_path, loaded_step = eval_setup(config_path, test_mode="test", update_config_callback=update_config)
        pipeline.model.config.intrinsic_color_parameterization = spec["parameterization"]
        pipeline.model.config.rasterize_mode = spec["rasterize_mode"]
        pipeline.eval()
        return pipeline, checkpoint_path, loaded_step, step

    def view_records(pipeline: Any) -> List[Tuple[str, Any, Mapping[str, Any]]]:
        dataset = pipeline.datamanager.eval_dataset
        filenames = list(getattr(dataset, "image_filenames", []))
        rows = []
        for eval_index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            view_id = Path(filenames[eval_index]).stem if eval_index < len(filenames) else f"eval_{eval_index:04d}"
            if view_id in {"MTN_1539", "MTN_1529", "MTN_1547"}:
                rows.append((view_id, camera, batch))
        return rows

    def gt_img(model: Any, batch: Mapping[str, Any], background: Any) -> Any:
        return model.composite_with_background(model.get_gt_img(batch["image"]), background)

    def safe_cpu(tensor: Any) -> Any:
        return tensor.detach().float().cpu()

    def luma(rgb: Any) -> Any:
        weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device)
        return (rgb * weights).sum(dim=-1)

    def edge_mag(luma_tensor: Any) -> Any:
        dx = F.pad((luma_tensor[:, 1:] - luma_tensor[:, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((luma_tensor[1:, :] - luma_tensor[:-1, :]).abs(), (0, 0, 0, 1))
        return dx + dy

    def scalar_to_image(values: Any, scale: float) -> Image.Image:
        arr = (values.detach().float().clamp_min(0.0) / max(float(scale), 1e-8)).clamp(0.0, 1.0)
        return Image.fromarray((arr.cpu().numpy() * 255.0).round().astype(np.uint8), mode="L").convert("RGB")

    def mask_image(mask: Any) -> Image.Image:
        arr = mask.detach().bool().cpu().numpy().astype(np.uint8) * 255
        return Image.fromarray(arr, mode="L").convert("RGB")

    def rgb_to_image(rgb: Any) -> Image.Image:
        arr = (rgb.detach().float().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    m1_cache: Dict[str, Dict[str, Any]] = {}
    pipeline = None
    try:
        pipeline, checkpoint_path, loaded_step, actual = load_run("M1")
        model = pipeline.model
        with torch.no_grad():
            for view_id, camera, batch in view_records(pipeline):
                outputs = model.get_outputs_for_camera(camera)
                gt = gt_img(model, batch, outputs["background"])
                gt_luma = luma(gt)
                high_j = outputs["clear_object_fullsh_raw"].amax(dim=-1) > 1.0
                brightness_q5 = gt_luma >= torch.quantile(gt_luma.reshape(-1), 0.80)
                depth_std = outputs.get("depth_std_relative", torch.zeros_like(gt_luma[..., None]))[..., 0]
                alpha = outputs.get("accumulation", torch.zeros_like(gt_luma[..., None]))[..., 0]
                edge = edge_mag(gt_luma)
                overlap = alpha * depth_std
                m1_cache[view_id] = {
                    "gt": safe_cpu(gt),
                    "gt_luma": safe_cpu(gt_luma),
                    "high_j": high_j.detach().bool().cpu(),
                    "brightness_q5": brightness_q5.detach().bool().cpu(),
                    "alpha": safe_cpu(alpha),
                    "depth_std_relative": safe_cpu(depth_std),
                    "edge": safe_cpu(edge),
                    "overlap_proxy": safe_cpu(overlap),
                }
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    residual_maps: Dict[str, Dict[str, Any]] = {"K1": {}, "AA": {}}
    for run_name in ("K1", "AA"):
        pipeline = None
        try:
            pipeline, checkpoint_path, loaded_step, actual = load_run(run_name)
            model = pipeline.model
            with torch.no_grad():
                for view_id, camera, batch in view_records(pipeline):
                    outputs = model.get_outputs_for_camera(camera)
                    gt = m1_cache[view_id]["gt"].to(outputs["pred_image"].device)
                    residual = torch.linalg.norm(outputs["pred_image"].float() - gt.float(), dim=-1)
                    residual_maps[run_name][view_id] = safe_cpu(residual)
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

    rows: List[Dict[str, Any]] = []
    masks_by_name = {
        "M1_J_gt_1": "high_j",
        "GT_brightness_Q5": "brightness_q5",
        "J_or_brightness": "union",
    }
    proxies = {
        "alpha": "alpha",
        "depth_variation": "depth_std_relative",
        "edge": "edge",
        "overlap_proxy": "overlap_proxy",
    }
    for view_id, data in m1_cache.items():
        union = data["high_j"] | data["brightness_q5"]
        data["union"] = union
        for mask_name, mask_key in masks_by_name.items():
            mask = data[mask_key].bool()
            comp = ~mask
            for proxy_name, proxy_key in proxies.items():
                values = data[proxy_key].float()
                region_mean = float(values[mask].mean().item()) if mask.any() else float("nan")
                control_mean = float(values[comp].mean().item()) if comp.any() else float("nan")
                rows.append(
                    {
                        "scene": "Panama",
                        "view_id": view_id,
                        "mask": mask_name,
                        "proxy": proxy_name,
                        "mask_fraction": float(mask.float().mean().item()),
                        "region_mean": region_mean,
                        "control_mean": control_mean,
                        "enrichment": region_mean / (control_mean + 1e-8) if math.isfinite(region_mean) and math.isfinite(control_mean) else float("nan"),
                    }
                )
            for run_name in ("K1", "AA"):
                residual = residual_maps[run_name][view_id].float()
                region_mean = float(residual[mask].mean().item()) if mask.any() else float("nan")
                control_mean = float(residual[comp].mean().item()) if comp.any() else float("nan")
                rows.append(
                    {
                        "scene": "Panama",
                        "view_id": view_id,
                        "mask": mask_name,
                        "proxy": f"{run_name}_residual",
                        "mask_fraction": float(mask.float().mean().item()),
                        "region_mean": region_mean,
                        "control_mean": control_mean,
                        "enrichment": region_mean / (control_mean + 1e-8) if math.isfinite(region_mean) and math.isfinite(control_mean) else float("nan"),
                    }
                )

    aggregate_rows: List[Dict[str, Any]] = []
    keys = sorted({(r["mask"], r["proxy"]) for r in rows})
    for mask, proxy in keys:
        selected = [r for r in rows if r["mask"] == mask and r["proxy"] == proxy]
        aggregate_rows.append(
            {
                "scene": "Panama",
                "view_id": "AGGREGATE",
                "mask": mask,
                "proxy": proxy,
                "mask_fraction": _mean(r["mask_fraction"] for r in selected),
                "region_mean": _mean(r["region_mean"] for r in selected),
                "control_mean": _mean(r["control_mean"] for r in selected),
                "enrichment": _mean(r["enrichment"] for r in selected),
            }
        )
    rows.extend(aggregate_rows)

    proxy_names = {"alpha", "depth_variation", "edge", "overlap_proxy"}
    aligned_candidates = []
    for mask in ("M1_J_gt_1", "GT_brightness_Q5", "J_or_brightness"):
        selected = [r for r in aggregate_rows if r["mask"] == mask and r["proxy"] in proxy_names]
        count = sum(1 for r in selected if float(r["enrichment"]) >= 1.25)
        aligned_candidates.append({"mask": mask, "enriched_proxy_count_ge_1p25": count})
    panama_aligned = any(item["enriched_proxy_count_ge_1p25"] >= 2 for item in aligned_candidates)

    max_resid = max(float(v.max().item()) for by_view in residual_maps.values() for v in by_view.values())
    max_depth = max(float(data["depth_std_relative"].max().item()) for data in m1_cache.values())
    max_edge = max(float(data["edge"].max().item()) for data in m1_cache.values())
    max_overlap = max(float(data["overlap_proxy"].max().item()) for data in m1_cache.values())
    sheet_rows = []
    for view_id, data in sorted(m1_cache.items()):
        sheet_rows.append(
            [
                (f"{view_id} GT", rgb_to_image(data["gt"])),
                ("M1 J>1 mask", mask_image(data["high_j"])),
                ("GT bright Q5", mask_image(data["brightness_q5"])),
                ("overlap proxy", scalar_to_image(data["overlap_proxy"], max_overlap)),
                ("depth std rel", scalar_to_image(data["depth_std_relative"], max_depth)),
                ("edge proxy", scalar_to_image(data["edge"], max_edge)),
                ("K1 residual", scalar_to_image(residual_maps["K1"][view_id], max_resid)),
                ("AA residual", scalar_to_image(residual_maps["AA"][view_id], max_resid)),
            ]
        )
    visual_path = render_dir / "panama_region_alignment.png"
    _save_sheet(visual_path, sheet_rows, tile_width)

    result = {
        "status": "OK",
        "rows": rows,
        "PANAMA_FAILURE_REGION_ALIGNED": bool(panama_aligned),
        "alignment_proxy_counts": aligned_candidates,
        "mask_definitions": {
            "M1_J_gt_1": "M1 clear_object_fullsh_raw max RGB channel > 1.0",
            "GT_brightness_Q5": "top 20 percent GT luminance within each eval view",
            "J_or_brightness": "union of M1_J_gt_1 and GT_brightness_Q5",
            "overlap_proxy": "M1 accumulation * M1 depth_std_relative; proxy only, not contributor count",
        },
        "visual_path": str(visual_path),
    }
    _write_json(output_dir / "panama_region_alignment.json", result)
    _write_csv(output_dir / "panama_region_alignment.csv", rows)
    return result


def write_visual_index(render_dir: Path, output_dir: Path, micro_visuals: Sequence[str], panama_result: Mapping[str, Any]) -> None:
    lines = ["# DCOMP Audit Visual Compare Index", ""]
    lines.extend(["## Micro-Case Visuals", ""])
    for path in micro_visuals:
        lines.append(path)
    lines.extend(["", "## Panama Region Alignment", ""])
    if panama_result.get("status") == "OK":
        lines.append(str(panama_result.get("visual_path")))
    else:
        lines.append(f"SKIPPED: {panama_result.get('reason', 'unknown')}")
    lines.extend(["", "## Output Tables", "", str(output_dir)])
    (render_dir / "VISUAL_COMPARE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def final_classification(micro: Mapping[str, Any], panama: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(micro["rows"])
    sensitivity_rows: List[Mapping[str, Any]] = []
    for item in micro["sensitivity"].values():
        sensitivity_rows.extend(item["rows"])

    def amp(metric: str, mode: str) -> float:
        for row in sensitivity_rows:
            if row["metric"] == metric and row["mode"] == mode:
                return float(row["amplification"])
        return float("nan")

    def non_negligible(mode: str) -> bool:
        vals = [float(r["COMPOSITING_DISAGREEMENT"]) for r in rows if r["mode"] == mode]
        return max(vals) >= 1e-4 if vals else False

    aligned_vals = [float(r["COMPOSITING_DISAGREEMENT"]) for r in rows if r["mode"] == "scale_aligned_constant_medium"]
    aligned_max = max(aligned_vals) if aligned_vals else float("nan")
    source_native_nonzero = non_negligible("source_native_distance_div10")
    medium_mismatch_nonzero = non_negligible("scale_aligned_per_gaussian_medium_mismatch")
    bright_amp = max(
        amp("BRIGHT_AMPLIFICATION", "source_native_distance_div10"),
        amp("BRIGHT_AMPLIFICATION", "scale_aligned_per_gaussian_medium_mismatch"),
    )
    depth_amp = max(
        amp("DEPTH_SEPARATION_AMPLIFICATION", "source_native_distance_div10"),
        amp("DEPTH_SEPARATION_AMPLIFICATION", "scale_aligned_per_gaussian_medium_mismatch"),
    )
    opacity_amp = max(
        amp("OPACITY_AMPLIFICATION", "source_native_distance_div10"),
        amp("OPACITY_AMPLIFICATION", "scale_aligned_per_gaussian_medium_mismatch"),
    )

    bright_flag = bool(bright_amp >= 1.5 and (source_native_nonzero or medium_mismatch_nonzero))
    depth_flag = bool(depth_amp >= 1.5 and (source_native_nonzero or medium_mismatch_nonzero))
    opacity_flag = bool(opacity_amp >= 1.5 and (source_native_nonzero or medium_mismatch_nonzero))
    panama_aligned = bool(panama.get("PANAMA_FAILURE_REGION_ALIGNED", False))
    sensitivity_count = sum((bright_flag, depth_flag, opacity_flag))

    if aligned_max <= 1e-8 and (source_native_nonzero or medium_mismatch_nonzero):
        equivalence = "EQUIVALENT_UNDER_RESTRICTED_CONDITIONS"
    elif source_native_nonzero or medium_mismatch_nonzero:
        equivalence = "STRUCTURALLY_DIFFERENT"
    else:
        equivalence = "EXACTLY_EQUIVALENT"

    if sensitivity_count >= 2 and panama_aligned:
        hypothesis = "SUPPORTED"
    elif (source_native_nonzero or medium_mismatch_nonzero) and sensitivity_count >= 1:
        hypothesis = "PARTIALLY_SUPPORTED"
    elif not (source_native_nonzero or medium_mismatch_nonzero):
        hypothesis = "NOT_SUPPORTED"
    else:
        hypothesis = "UNRESOLVED"

    return {
        "ARE_SEAFREE_AND_WS_DEGRADATION_COMPOSITING_EQUIVALENT": equivalence,
        "HYPOTHESIS_ASSESSMENT": hypothesis,
        "FORWARD_ORGANIZATION_DIFFERENT": True,
        "MULTI_GAUSSIAN_NON_EQUIVALENCE": bool(aligned_max > 1e-8),
        "BRIGHT_SENSITIVE_GAP": bright_flag,
        "DEPTH_SENSITIVE_GAP": depth_flag,
        "OPACITY_SENSITIVE_GAP": opacity_flag,
        "PANAMA_FAILURE_REGION_ALIGNED": panama_aligned,
        "aligned_constant_medium_max_disagreement": aligned_max,
        "source_native_distance_non_negligible": source_native_nonzero,
        "per_gaussian_medium_mismatch_non_negligible": medium_mismatch_nonzero,
        "BRIGHT_AMPLIFICATION": bright_amp,
        "DEPTH_SEPARATION_AMPLIFICATION": depth_amp,
        "OPACITY_AMPLIFICATION": opacity_amp,
        "NEXT_SINGLE_FACTOR_EXPERIMENT": (
            "Read-only fixed-checkpoint SeaFree-order counterfactual alignment diagnostic"
            if hypothesis == "PARTIALLY_SUPPORTED"
            else "Panama BND-K1 plus SeaFree-style degradation/compositing ordering from scratch"
            if hypothesis == "SUPPORTED"
            else "SeaFree-style content-based loss responsibility audit/test"
            if hypothesis == "NOT_SUPPORTED"
            else "Resolve semantic alignment needed for fixed-state counterfactual"
        ),
        "fixed_state_counterfactual": {
            "status": "SKIPPED",
            "COUNTERFACTUAL_ALIGNMENT_VALID": False,
            "reason": "A reliable fixed-state renderer would need exact alignment of per-Gaussian medium query, LOS distance, projection footprint, opacity compensation, and background semantics; this audit did not modify renderers to add that path.",
        },
        "gradient_sensitivity": {
            "status": "NOT_REQUIRED",
            "reason": "The optional gradient audit requires a valid fixed-state counterfactual, which was not established.",
        },
    }


def write_manifests(repo: Path, output_dir: Path, render_dir: Path, final_summary: Mapping[str, Any]) -> None:
    manifest = {
        "repo": str(repo),
        "output_dir": str(output_dir),
        "render_dir": str(render_dir),
        "outputs": sorted(str(p) for p in output_dir.rglob("*") if p.is_file()),
        "renders": sorted(str(p) for p in render_dir.rglob("*") if p.is_file()),
        "final_summary": final_summary,
    }
    _write_json(output_dir / "manifest.json", manifest)
    rows = []
    for kind, root in (("output", output_dir), ("render", render_dir)):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append({"kind": kind, "file_path": str(path), "bytes": path.stat().st_size})
    _write_csv(output_dir / "manifest.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--seafree-repo", type=Path, default=Path("/mnt/new/home_old/ycy/reference_repos/SeaFree-GS"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dcomp_audit_20260810"))
    parser.add_argument("--render-dir", type=Path, default=Path("renders/dcomp_audit_20260810"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/dcomp_audit_20260810"))
    parser.add_argument("--tile-width", type=int, default=240)
    parser.add_argument("--skip-panama-forward", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    seafree_repo = args.seafree_repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    render_dir = (repo / args.render_dir).resolve() if not args.render_dir.is_absolute() else args.render_dir
    logs_dir = (repo / args.logs_dir).resolve() if not args.logs_dir.is_absolute() else args.logs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    repo_manifest = {
        "water_splatting_repo": str(repo),
        "water_splatting_branch": _git(repo, "branch", "--show-current"),
        "water_splatting_head": _git(repo, "rev-parse", "HEAD"),
        "water_splatting_status_short": _git(repo, "status", "--short"),
        "seafree_repo": str(seafree_repo),
        "seafree_reference_commit_expected": SEAFREE_COMMIT,
        "seafree_head": _git(seafree_repo, "rev-parse", "HEAD"),
        "seafree_status_short": _git(seafree_repo, "status", "--short"),
    }
    _write_json(output_dir / "repo_manifest.json", repo_manifest)

    seafree, ws, equivalence = source_semantics()
    _write_json(output_dir / "seafree_forward_semantics.json", seafree)
    _write_json(output_dir / "watersplatting_forward_semantics.json", ws)
    _write_json(output_dir / "equivalence_conditions.json", equivalence)
    write_symbolic_outputs(output_dir)

    micro = run_microcases(output_dir, render_dir, args.tile_width)
    panama = run_panama_region_alignment(repo, output_dir, render_dir, args.tile_width, args.skip_panama_forward)
    classification = final_classification(micro, panama)
    _write_json(output_dir / "fixed_state_counterfactual_metrics.json", classification["fixed_state_counterfactual"])
    _write_csv(output_dir / "fixed_state_counterfactual_metrics.csv", [])
    _write_json(output_dir / "gradient_sensitivity.json", classification["gradient_sensitivity"])
    _write_csv(output_dir / "gradient_sensitivity.csv", [])
    _write_json(output_dir / "dcomp_final_summary.json", classification)
    _write_csv(output_dir / "dcomp_final_summary.csv", [classification])
    write_visual_index(render_dir, output_dir, micro["visual_paths"], panama)
    write_manifests(repo, output_dir, render_dir, classification)

    print(json.dumps({"output_dir": str(output_dir), "render_dir": str(render_dir), **classification}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
