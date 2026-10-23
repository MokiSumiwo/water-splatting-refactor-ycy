#!/usr/bin/env python3
"""Summarize the retained BVAM BND-ON screening runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


SCENES = ("JapaneseGradens-RedSea", "Panama")
VARIANTS = (
    "E0_bnd_fixed",
    "E2_brash_fixed_gate",
    "E3_bvam_k2",
    "E4_bvam_pseudogt",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _variant_root(root: Path, variant: str, scene: str) -> Path:
    if variant in ("E3_bvam_k2", "E4_bvam_pseudogt"):
        return root / ".." / "bvam_bnd_on_screening_20260909_retry2" / scene / variant
    return root / scene / variant


def _float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else 0.0


def _metric_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        for scene in SCENES:
            directory = _variant_root(root, variant, scene).resolve()
            metrics = _read_csv(directory / "evaluation_metrics.csv")[0]
            safety = _read_json(directory / "decomposition_safety.json")["rows"][0]
            final_summary = _read_json(directory / "final_summary.json")
            rows.append(
                {
                    "variant": variant,
                    "scene": scene,
                    "PSNR": _float(metrics, "PSNR"),
                    "SSIM": _float(metrics, "SSIM"),
                    "LPIPS": _float(metrics, "LPIPS"),
                    "MSE": _float(metrics, "MSE"),
                    "P_J_gt_1": _float(safety, "P_J_gt_1"),
                    "J_p99": _float(safety, "J_p99"),
                    "tau_p90": _float(safety, "tau_p90"),
                    "checkpoint": final_summary["checkpoint_rows"][-1]["path"],
                    "render_manifest": str(directory / "render_manifest.csv"),
                    "gradient_audit": str(directory / "gradient_routing_audit.json"),
                    "checkpoint_retained": bool(final_summary.get("checkpoint_retained", False)),
                }
            )
    return rows


def _means(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    values = list(rows)
    return {
        key: sum(float(row[key]) for row in values) / max(len(values), 1)
        for key in ("PSNR", "SSIM", "LPIPS", "MSE", "P_J_gt_1", "J_p99", "tau_p90")
    }


def _diagnostics(root: Path, variant: str, scene: str) -> Dict[str, Any]:
    directory = _variant_root(root, variant, scene).resolve()
    rows = _read_csv(directory / "module_statistics.csv")
    logs = [row for row in rows if row.get("event") == "training_log"]
    bvam_refresh = [row for row in rows if row.get("module") == "BVAM" and row.get("event") == "pseudo_target_refresh"]
    final = logs[-1] if logs else {}
    prior = bvam_refresh[-1] if bvam_refresh else (bvam_refresh[0] if bvam_refresh else {})
    return {
        "expert_usage_1": _float(final, "BVAM_expert_usage_1"),
        "expert_usage_2": _float(final, "BVAM_expert_usage_2"),
        "expert_entropy": _float(final, "BVAM_expert_entropy"),
        "expert_collapse_fraction": _float(final, "BVAM_expert_collapse_fraction"),
        "delta_norm_p90": _float(final, "BVAM_delta_norm_p90"),
        "support_mean_initial": _float(bvam_refresh[0], "support_mean") if bvam_refresh else 0.0,
        "support_p50_initial": _float(bvam_refresh[0], "support_p50") if bvam_refresh else 0.0,
        "valid_gaussian_fraction_initial": _float(bvam_refresh[0], "valid_gaussian_fraction") if bvam_refresh else 0.0,
        "pseudo_confidence_mean_initial": _float(bvam_refresh[0], "pseudo_confidence_mean") if bvam_refresh else 0.0,
        "pseudo_valid_fraction_final": _float(final, "BVAM_pseudo_valid_fraction"),
        "gradient_audit_pass": bool(_read_json(directory / "gradient_routing_audit.json").get("GRADIENT_ROUTING_AUDIT", False)),
        "evaluation_restore_audit": str(directory / "evaluation_restore_audit.json") if (directory / "evaluation_restore_audit.json").is_file() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/bvam_bnd_on_screening_20260909"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root).resolve()
    rows = _metric_rows(root)
    base = {row["scene"]: row for row in rows if row["variant"] == "E0_bnd_fixed"}
    for row in rows:
        row["delta_vs_E0"] = {key: row[key] - base[row["scene"]][key] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}
        row["diagnostics"] = _diagnostics(root, row["variant"], row["scene"]) if row["variant"].startswith("E3") or row["variant"].startswith("E4") else {}
    means = {variant: _means(row for row in rows if row["variant"] == variant) for variant in VARIANTS}
    mean_deltas = {
        variant: {key: means[variant][key] - means["E0_bnd_fixed"][key] for key in ("PSNR", "SSIM", "LPIPS", "MSE")}
        for variant in VARIANTS
    }
    payload = {
        "experiment": "BVAM_BND_ON_BOUNDED_APPEARANCE",
        "protocol": {
            "scenes": list(SCENES),
            "variants": list(VARIANTS),
            "seed": 42,
            "start_step": 3000,
            "final_step": 14999,
            "M1": True,
            "OCMC": True,
            "CICA": True,
            "auxiliary": False,
            "BND": True,
            "RAOC": False,
            "evaluation": "local metrics protocol; RGB in [0,1], no resize/crop/color conversion",
        },
        "rows": rows,
        "means": means,
        "mean_deltas_vs_E0": mean_deltas,
        "four_scene_validation": {"run": False, "reason": "No BVAM variant met the two-scene retention criterion."},
        "equivalence_audit": "outputs/bvam_audit_20260909/JapaneseGradens-RedSea.json",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "BVAM_SCREENING_SUMMARY.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")
    with (output_dir / "BVAM_SCREENING_METRICS.csv").open("w", newline="", encoding="utf8") as handle:
        fields = ["variant", "scene", "PSNR", "SSIM", "LPIPS", "MSE", "P_J_gt_1", "J_p99", "tau_p90", "checkpoint_retained", "checkpoint", "gradient_audit"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    print(json.dumps({"means": means, "mean_deltas_vs_E0": mean_deltas}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
