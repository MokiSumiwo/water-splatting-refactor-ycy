#!/usr/bin/env python
"""Summarize objective image evidence for the four-scene GMVC visual audit.

The script intentionally leaves subjective review fields empty unless they are
provided in an annotations JSON. It operates on the already-rendered PNGs and
does not load checkpoints or change renderer behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
from PIL import Image


SCENES = ("Curasao", "JapaneseGradens", "IUI3", "Panama")
RGB_METRICS = ("psnr", "ssim", "lpips", "rgb_l1", "luminance_l1", "chroma_l1")
MEDIUM_COMPONENTS = (
    "transmission",
    "backscatter_endpoint",
    "rgb_medium",
    "b_inf",
    "medium_attn",
    "medium_bs",
)
MANUAL_FIELDS = (
    "underwater_rgb_trend",
    "underwater_luminance_trend",
    "underwater_chroma_trend",
    "dewatered_color_consistency",
    "dewatered_overcorrection",
    "texture_change",
    "medium_component_anomaly",
    "metric_visual_agreement",
    "review_notes",
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _luma(image: np.ndarray) -> np.ndarray:
    return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114


def _edge_energy(image: np.ndarray) -> float:
    luma = _luma(image)
    dx = np.abs(np.diff(luma, axis=1)).mean() if luma.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(luma, axis=0)).mean() if luma.shape[0] > 1 else 0.0
    return float(dx + dy)


def _image_stats(image: np.ndarray) -> Dict[str, Any]:
    luma = _luma(image)
    saturation = image.max(axis=-1) - image.min(axis=-1)
    return {
        "mean_rgb": [float(value) for value in image.mean(axis=(0, 1))],
        "luminance_mean": float(luma.mean()),
        "luminance_std": float(luma.std()),
        "saturation_mean": float(saturation.mean()),
        "lowlight_fraction": float((luma <= (5.0 / 255.0)).mean()),
        "highlight_fraction": float((luma >= (250.0 / 255.0)).mean()),
        "channel_clip_fraction": float(((image <= 0.0) | (image >= 1.0)).mean()),
        "edge_energy": _edge_energy(image),
    }


def _subtract_stats(candidate: Mapping[str, Any], base: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in candidate:
        if key == "mean_rgb":
            out[key] = [float(new - old) for new, old in zip(candidate[key], base[key])]
        else:
            out[key] = float(candidate[key]) - float(base[key])
    return out


def _pair_stats(base: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    if base.shape != candidate.shape:
        raise ValueError(f"Image shape mismatch: {base.shape} versus {candidate.shape}")
    difference = candidate - base
    absolute = np.abs(difference)
    pixel_l1 = absolute.mean(axis=-1)
    return {
        "signed_mean_rgb": [float(value) for value in difference.mean(axis=(0, 1))],
        "absolute_l1_mean": float(absolute.mean()),
        "absolute_l1_p95": float(np.quantile(pixel_l1, 0.95)),
        "changed_pixel_fraction_2_255": float((pixel_l1 > (2.0 / 255.0)).mean()),
        "candidate_minus_base": _subtract_stats(_image_stats(candidate), _image_stats(base)),
    }


def _residual_stats(gt: np.ndarray, a0: np.ndarray, p30: np.ndarray) -> Dict[str, float]:
    if not (gt.shape == a0.shape == p30.shape):
        raise ValueError(f"Residual image shape mismatch: {gt.shape}, {a0.shape}, {p30.shape}")
    a0_residual = np.abs(a0 - gt).mean(axis=-1)
    p30_residual = np.abs(p30 - gt).mean(axis=-1)
    difference = p30_residual - a0_residual
    return {
        "a0_mean": float(a0_residual.mean()),
        "p30_mhold_mean": float(p30_residual.mean()),
        "p30_minus_a0_mean": float(difference.mean()),
        "improved_pixel_fraction": float((difference < 0.0).mean()),
        "worsened_pixel_fraction": float((difference > 0.0).mean()),
        "unchanged_pixel_fraction": float((difference == 0.0).mean()),
    }


def _config_signature(path: Path) -> str:
    lines = path.read_text(encoding="utf8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("  datamanager:"))
    stop = next(
        index
        for index, line in enumerate(lines[start + 1 :], start + 1)
        if line.startswith("  model:")
    )
    payload = "\n".join(lines[start:stop]).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def _annotation_for(
    annotations: Mapping[str, Any], scene: str, view_index: int
) -> Dict[str, str]:
    scene_data = annotations.get(scene, {})
    view_data = scene_data.get(str(view_index), scene_data.get(f"view_{view_index:04d}", {}))
    return {key: str(view_data.get(key, "")) for key in MANUAL_FIELDS}


def _component_stats(view_dir: Path) -> Dict[str, Any]:
    components: Dict[str, Any] = {}
    for key in MEDIUM_COMPONENTS:
        a0_path = view_dir / "components" / f"a0_{key}.png"
        p30_path = view_dir / "components" / f"p30-mhold_{key}.png"
        a0 = _load_rgb(a0_path)
        p30 = _load_rgb(p30_path)
        pair = _pair_stats(a0, p30)
        pair["a0_spatial_edge_energy"] = _edge_energy(a0)
        pair["p30_mhold_spatial_edge_energy"] = _edge_energy(p30)
        pair["spatial_edge_energy_delta"] = (
            pair["p30_mhold_spatial_edge_energy"] - pair["a0_spatial_edge_energy"]
        )
        components[key] = pair
    return components


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(len(values), 1))


def _scene_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_deltas = {
        key: [float(row["metric_delta_vs_a0"][key]) for row in rows] for key in RGB_METRICS
    }
    return {
        "view_count": len(rows),
        "per_view_metric_sign_counts": {
            key: {
                "positive": sum(value > 0.0 for value in values),
                "negative": sum(value < 0.0 for value in values),
                "zero": sum(value == 0.0 for value in values),
            }
            for key, values in metric_deltas.items()
        },
        "underwater_residual_improved_pixel_fraction_mean": _mean(
            row["underwater"]["residual"]["improved_pixel_fraction"] for row in rows
        ),
        "underwater_render_absolute_l1_mean": _mean(
            row["underwater"]["p30_mhold_vs_a0"]["absolute_l1_mean"] for row in rows
        ),
        "dewatered_proxy_absolute_l1_mean": _mean(
            row["dewatered_proxy"]["p30_mhold_vs_a0"]["absolute_l1_mean"] for row in rows
        ),
        "medium_component_absolute_l1_mean": {
            key: _mean(row["medium_components"][key]["absolute_l1_mean"] for row in rows)
            for key in MEDIUM_COMPONENTS
        },
        "medium_component_spatial_edge_energy_delta_mean": {
            key: _mean(row["medium_components"][key]["spatial_edge_energy_delta"] for row in rows)
            for key in MEDIUM_COMPONENTS
        },
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "scene",
        "view_index",
        "image_idx",
        "image_name",
        *[f"delta_{key}" for key in RGB_METRICS],
        "underwater_abs_l1",
        "underwater_luminance_delta",
        "underwater_chroma_delta",
        "underwater_contrast_delta",
        "underwater_saturation_delta",
        "underwater_edge_energy_delta",
        "underwater_residual_improved_pixel_fraction",
        "underwater_residual_l1_delta",
        "dewatered_abs_l1",
        "dewatered_luminance_delta",
        "dewatered_contrast_delta",
        "dewatered_saturation_delta",
        "dewatered_edge_energy_delta",
        *[f"{key}_abs_l1" for key in MEDIUM_COMPONENTS],
        *[f"{key}_edge_energy_delta" for key in MEDIUM_COMPONENTS],
        *MANUAL_FIELDS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            underwater_delta = row["underwater"]["p30_mhold_vs_a0"]["candidate_minus_base"]
            dewatered_delta = row["dewatered_proxy"]["p30_mhold_vs_a0"]["candidate_minus_base"]
            flat: Dict[str, Any] = {
                "scene": row["scene"],
                "view_index": row["view_index"],
                "image_idx": row["image_idx"],
                "image_name": row["image_name"],
                **{f"delta_{key}": row["metric_delta_vs_a0"][key] for key in RGB_METRICS},
                "underwater_abs_l1": row["underwater"]["p30_mhold_vs_a0"]["absolute_l1_mean"],
                "underwater_luminance_delta": underwater_delta["luminance_mean"],
                "underwater_chroma_delta": row["metric_delta_vs_a0"]["chroma_l1"],
                "underwater_contrast_delta": underwater_delta["luminance_std"],
                "underwater_saturation_delta": underwater_delta["saturation_mean"],
                "underwater_edge_energy_delta": underwater_delta["edge_energy"],
                "underwater_residual_improved_pixel_fraction": row["underwater"]["residual"]["improved_pixel_fraction"],
                "underwater_residual_l1_delta": row["underwater"]["residual"]["p30_minus_a0_mean"],
                "dewatered_abs_l1": row["dewatered_proxy"]["p30_mhold_vs_a0"]["absolute_l1_mean"],
                "dewatered_luminance_delta": dewatered_delta["luminance_mean"],
                "dewatered_contrast_delta": dewatered_delta["luminance_std"],
                "dewatered_saturation_delta": dewatered_delta["saturation_mean"],
                "dewatered_edge_energy_delta": dewatered_delta["edge_energy"],
                **row["manual_review"],
            }
            for key in MEDIUM_COMPONENTS:
                flat[f"{key}_abs_l1"] = row["medium_components"][key]["absolute_l1_mean"]
                flat[f"{key}_edge_energy_delta"] = row["medium_components"][key]["spatial_edge_energy_delta"]
            writer.writerow(flat)


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    annotations = _load_json(args.annotations) if args.annotations else {}
    all_rows: List[Dict[str, Any]] = []
    scenes: Dict[str, Any] = {}
    for scene in args.scene:
        scene_dir = args.root / scene
        vis_dir = scene_dir / "visualization"
        summary_path = vis_dir / "visualization_summary.json"
        source = _load_json(summary_path)
        runs = source["runs"]
        a0_config = Path(runs["A0"]["config"])
        p30_config = Path(runs["P30-MHOLD"]["config"])
        a0_signature = _config_signature(a0_config)
        p30_signature = _config_signature(p30_config)
        scene_rows: List[Dict[str, Any]] = []
        for source_row in source["per_view"]:
            view_index = int(source_row["view_index"])
            view_dir = vis_dir / f"view_{view_index:04d}"
            gt = _load_rgb(view_dir / "underwater" / "gt.png")
            a0_underwater = _load_rgb(view_dir / "underwater" / "a0_underwater.png")
            p30_underwater = _load_rgb(view_dir / "underwater" / "p30-mhold_underwater.png")
            a0_dewatered = _load_rgb(view_dir / "dewatered" / "a0_j_proxy_or_clear.png")
            p30_dewatered = _load_rgb(view_dir / "dewatered" / "p30-mhold_j_proxy_or_clear.png")
            row = {
                "scene": scene,
                "view_index": view_index,
                "image_idx": int(source_row["image_idx"]),
                "image_name": source_row["image_name"],
                "metric_delta_vs_a0": {
                    key: float(source_row["delta_vs_a0"]["P30-MHOLD"][key])
                    for key in RGB_METRICS
                },
                "underwater": {
                    "definition": "Normal eval outputs['pred_image'], clamped to [0,1] before PNG save.",
                    "a0": _image_stats(a0_underwater),
                    "p30_mhold": _image_stats(p30_underwater),
                    "p30_mhold_vs_a0": _pair_stats(a0_underwater, p30_underwater),
                    "residual": _residual_stats(gt, a0_underwater, p30_underwater),
                },
                "dewatered_proxy": {
                    "definition": source["dewatered_definition"],
                    "a0": _image_stats(a0_dewatered),
                    "p30_mhold": _image_stats(p30_dewatered),
                    "p30_mhold_vs_a0": _pair_stats(a0_dewatered, p30_dewatered),
                    "clear_image_gt_available": False,
                },
                "medium_components": _component_stats(view_dir),
                "manual_review": _annotation_for(annotations, scene, view_index),
            }
            scene_rows.append(row)
            all_rows.append(row)
        scenes[scene] = {
            "source": str(summary_path),
            "runs": runs,
            "selected_contact_views": source["selected_contact_views"],
            "contact_sheets": source["contact_sheets"],
            "component_visualization_ranges": source["component_visualization_ranges"],
            "comparability": {
                "same_requested_step": runs["A0"]["requested_step"] == runs["P30-MHOLD"]["requested_step"],
                "same_loaded_step": runs["A0"]["step"] == runs["P30-MHOLD"]["step"],
                "same_datamanager_config": a0_signature == p30_signature,
                "a0_datamanager_sha256": a0_signature,
                "p30_mhold_datamanager_sha256": p30_signature,
                "same_png_range": "[0,1] clamp in the shared diagnostic save path",
                "same_tone_mapping": "No per-run tone mapping in the diagnostic save path",
            },
            "summary": _scene_summary(scene_rows),
            "per_view": scene_rows,
        }
    result = {
        "diagnostic": "gmvc_four_scene_visual_audit",
        "root": str(args.root),
        "scope": {
            "view_selection": "All eval views listed in each visualization_summary.json",
            "underwater_definition": "Normal eval outputs['pred_image'], clamped to [0,1] before PNG save.",
            "dewatered_definition": (
                "J_proxy_raw rendered with a diagnostic DC-proxy context when available; "
                "fallback is J_gaussian_raw/J_raw. This is a model intrinsic proxy, not clear-image GT."
            ),
            "component_definition": {
                "transmission": "exp(-medium_attn * depth)",
                "backscatter_endpoint": "B_inf * (1 - exp(-medium_bs * depth))",
                "rgb_medium": "renderer-integrated medium contribution",
                "b_inf": "predicted asymptotic medium color, or medium_rgb fallback",
                "medium_attn": "direct-signal attenuation coefficient",
                "medium_bs": "backscatter growth coefficient",
            },
            "subjective_fields": "Empty unless supplied through --annotations; no subjective labels are inferred.",
        },
        "scenes": scenes,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    _write_csv(args.output_csv, all_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("renders/gmvc_four_scene_p30_mhold_15k"))
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()
    if not args.scene:
        args.scene = list(SCENES)
    if args.output_json is None:
        args.output_json = args.root / "visual_audit_summary.json"
    if args.output_csv is None:
        args.output_csv = args.root / "visual_audit_summary.csv"
    result = summarize(args)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "scenes": list(result["scenes"]),
                "views": sum(scene["summary"]["view_count"] for scene in result["scenes"].values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
