#!/usr/bin/env python
"""Audit whether Panama BND residuals justify an antialiased-rasterization run."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


SCENES = ("Curasao", "JapaneseGradens", "IUI3", "Panama")
CONTROL_SCENES = ("Curasao", "IUI3")
EPS = 1e-12


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf8")


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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception:
        return "unknown"


def _file_excerpt(path: Path, start: int, end: int) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf8", errors="replace").splitlines()
    return [f"{idx + 1}: {lines[idx]}" for idx in range(max(0, start - 1), min(end, len(lines)))]


def _scene_rows(freq_rows: Sequence[Mapping[str, str]], edge_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    freq_by_scene = {
        (row["scene"], float(row["sigma_px"])): row
        for row in freq_rows
        if row.get("view_id") == "AGGREGATE"
    }
    edge_by_scene = {row["scene"]: row for row in edge_rows if row.get("view_id") == "AGGREGATE"}
    rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        f3 = freq_by_scene[(scene, 3.0)]
        f9 = freq_by_scene[(scene, 9.0)]
        edge = edge_by_scene[scene]
        edge_fraction = float(edge["residual_energy_fraction_top20_edge"])
        edge_area = float(edge["top20_edge_pixel_fraction"])
        rows.append(
            {
                "scene": scene,
                "LOW_FREQ_ENERGY_FRACTION_sigma3": float(f3["LOW_FREQ_ENERGY_FRACTION"]),
                "HIGH_FREQ_ENERGY_FRACTION_sigma3": float(f3["HIGH_FREQ_ENERGY_FRACTION"]),
                "low_energy_sigma3": float(f3["low_energy"]),
                "high_energy_sigma3": float(f3["high_energy"]),
                "LOW_FREQ_ENERGY_FRACTION_sigma9": float(f9["LOW_FREQ_ENERGY_FRACTION"]),
                "HIGH_FREQ_ENERGY_FRACTION_sigma9": float(f9["HIGH_FREQ_ENERGY_FRACTION"]),
                "low_energy_sigma9": float(f9["low_energy"]),
                "high_energy_sigma9": float(f9["high_energy"]),
                "EDGE_RESIDUAL_CORRELATION": float(edge["residual_edge_pearson"]),
                "edge_pixel_fraction": edge_area,
                "TOP20_EDGE_RESIDUAL_ENERGY_FRACTION": edge_fraction,
                "NON_EDGE_RESIDUAL_ENERGY_FRACTION": float(edge["residual_energy_fraction_non_edge"]),
                "EDGE_ENRICHMENT": edge_fraction / max(edge_area, EPS),
                "frequency_definition": "BND residual = GT - I_BND; low = GaussianBlur(residual, sigma); high = residual - low; energy fractions use squared RGB residual energy.",
                "edge_definition": edge.get("edge_definition") or "top20 percent GT luminance gradient magnitude",
            }
        )
    return rows


def _eligibility(scene_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_scene = {row["scene"]: row for row in scene_rows}
    hf_control_sigma3 = sum(float(by_scene[s]["HIGH_FREQ_ENERGY_FRACTION_sigma3"]) for s in CONTROL_SCENES) / len(CONTROL_SCENES)
    hf_control_sigma9 = sum(float(by_scene[s]["HIGH_FREQ_ENERGY_FRACTION_sigma9"]) for s in CONTROL_SCENES) / len(CONTROL_SCENES)
    edge_control = sum(float(by_scene[s]["EDGE_ENRICHMENT"]) for s in CONTROL_SCENES) / len(CONTROL_SCENES)
    panama = by_scene["Panama"]
    panama_hf_ratio_sigma3 = float(panama["HIGH_FREQ_ENERGY_FRACTION_sigma3"]) / max(hf_control_sigma3, EPS)
    panama_hf_ratio_sigma9 = float(panama["HIGH_FREQ_ENERGY_FRACTION_sigma9"]) / max(hf_control_sigma9, EPS)
    panama_edge_ratio = float(panama["EDGE_ENRICHMENT"]) / max(edge_control, EPS)
    condition_a = float(panama["EDGE_ENRICHMENT"]) >= 1.50 and panama_edge_ratio >= 1.15
    condition_b_sigma3 = panama_hf_ratio_sigma3 >= 1.15 and float(panama["HIGH_FREQ_ENERGY_FRACTION_sigma3"]) >= 0.10
    condition_b_sigma9 = panama_hf_ratio_sigma9 >= 1.15 and float(panama["HIGH_FREQ_ENERGY_FRACTION_sigma9"]) >= 0.10
    low_freq_rejection = (
        float(panama["LOW_FREQ_ENERGY_FRACTION_sigma3"]) >= 0.80
        and float(panama["LOW_FREQ_ENERGY_FRACTION_sigma9"]) >= 0.80
        and not condition_a
    )
    eligible = (condition_a or condition_b_sigma3 or condition_b_sigma9) and not low_freq_rejection
    reasons = []
    if condition_a:
        reasons.append("Condition A edge-structured residual passed")
    if condition_b_sigma3:
        reasons.append("Condition B sigma3 high-frequency excess passed")
    if condition_b_sigma9:
        reasons.append("Condition B sigma9 high-frequency excess passed")
    if low_freq_rejection:
        reasons.append("Both low-frequency fractions >=0.80 without Condition A")
    if not reasons:
        reasons.append("No edge/high-frequency gate condition passed")
    return {
        "HF_CONTROL_MEAN_sigma3": hf_control_sigma3,
        "HF_CONTROL_MEAN_sigma9": hf_control_sigma9,
        "PANAMA_HF_RATIO_sigma3": panama_hf_ratio_sigma3,
        "PANAMA_HF_RATIO_sigma9": panama_hf_ratio_sigma9,
        "EDGE_CONTROL_MEAN": edge_control,
        "PANAMA_EDGE_RATIO": panama_edge_ratio,
        "Condition_A_edge_structured": condition_a,
        "Condition_B_sigma3_high_frequency": condition_b_sigma3,
        "Condition_B_sigma9_high_frequency": condition_b_sigma9,
        "low_frequency_rejection": low_freq_rejection,
        "AA_ELIGIBLE": eligible,
        "AA_REJECTION_REASON": "" if eligible else "; ".join(reasons),
        "AA_ELIGIBILITY_REASON": "; ".join(reasons),
    }


def current_aa_code_audit(repo: Path) -> Dict[str, Any]:
    model_path = repo / "water_splatting/water_splatting.py"
    return {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "CODE_FACTS": {
            "rasterize_mode_config": "WaterSplattingModelConfig.rasterize_mode supports classic and antialiased; default is classic.",
            "classic_opacity": "opacities = torch.sigmoid(opacities_crop)",
            "antialiased_opacity": "opacities = torch.sigmoid(opacities_crop) * comp[:, None]",
            "comp_source": "comp is returned by underwater_rasterizer.project along with projected Gaussian screen-space quantities.",
            "scope": "The switch changes screen-space effective opacity before underwater_rasterizer.rasterize; it does not directly redefine Gaussian color, bounded J, beta_D, beta_B, medium RGB, or tied B_inf.",
        },
        "line_excerpts": _file_excerpt(model_path, 171, 179) + _file_excerpt(model_path, 1404, 1411),
        "AA_SEMANTICS_CURRENT": "screen-space opacity compensation",
    }


def seafree_aa_semantics_audit(reference_repo: Path) -> Dict[str, Any]:
    if not reference_repo.exists():
        return {
            "reference_repo": str(reference_repo),
            "available": False,
            "reason": "reference repo not found; no clone attempted",
            "AA_SEMANTICS_MATCH": "UNAVAILABLE",
        }
    model_path = reference_repo / "seafree_gs/seafree_model.py"
    config_path = reference_repo / "seafree_gs/seafree_config.py"
    return {
        "reference_repo": str(reference_repo),
        "available": True,
        "git_status_short": _git(reference_repo, "status", "--short"),
        "reference_commit": _git(reference_repo, "rev-parse", "HEAD"),
        "CODE_FACTS": {
            "default_config": "seafree_gs/seafree_config.py sets rasterize_mode='antialiased' in the main model config.",
            "rasterization_call": "seafree_gs/seafree_model.py passes rasterize_mode=self.config.rasterize_mode to gsplat.rendering.rasterization.",
            "opacity_input": "SeaFree passes torch.sigmoid(opacities_crop).squeeze(-1) as opacities to gsplat rasterization; gsplat handles antialiased compensation internally for rasterize_mode='antialiased'.",
            "water_splatting_relation": "WaterSplatting uses its existing project() compensation factor and multiplies opacity before its underwater rasterizer. Both are screen-space antialias opacity-compensation mechanisms, not new water physics.",
        },
        "line_excerpts": _file_excerpt(config_path, 25, 42) + _file_excerpt(model_path, 150, 170) + _file_excerpt(model_path, 663, 681),
        "AA_SEMANTICS_MATCH": "STRUCTURALLY_SIMILAR",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/mnt/new/home_old/ycy/water-splatting-refactor"))
    parser.add_argument("--reference-repo", type=Path, default=Path("/mnt/new/home_old/ycy/reference_repos/SeaFree-GS"))
    parser.add_argument("--recomposition-dir", type=Path, default=Path("outputs/bnd_object_medium_recomposition_20260810"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_aa_panama_20260810"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    recomposition_dir = (repo / args.recomposition_dir).resolve() if not args.recomposition_dir.is_absolute() else args.recomposition_dir
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    freq_rows = _read_csv(recomposition_dir / "frequency_residual_analysis.csv")
    edge_rows = _read_csv(recomposition_dir / "edge_alignment_analysis.csv")
    scene_rows = _scene_rows(freq_rows, edge_rows)
    eligibility = _eligibility(scene_rows)
    audit_rows = []
    for row in scene_rows:
        out = dict(row)
        if row["scene"] == "Panama":
            out.update(eligibility)
        audit_rows.append(out)
    current_aa = current_aa_code_audit(repo)
    seafree = seafree_aa_semantics_audit(args.reference_repo.resolve())
    manifest = {
        "repo": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "head": _git(repo, "rev-parse", "HEAD"),
        "recomposition_source": str(recomposition_dir),
        "aa_eligibility_audit": str(output_dir / "aa_eligibility_audit.json"),
        "current_aa_code_audit": str(output_dir / "current_aa_code_audit.json"),
        "seafree_aa_semantics_audit": str(output_dir / "seafree_aa_semantics_audit.json"),
        "AA_ELIGIBLE": eligibility["AA_ELIGIBLE"],
        "AA_ELIGIBILITY_REASON": eligibility["AA_ELIGIBILITY_REASON"],
    }
    _write_csv(output_dir / "aa_eligibility_audit.csv", audit_rows)
    _write_json(output_dir / "aa_eligibility_audit.json", {"scene_rows": scene_rows, "gate": eligibility})
    _write_json(output_dir / "current_aa_code_audit.json", current_aa)
    _write_json(output_dir / "seafree_aa_semantics_audit.json", seafree)
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
