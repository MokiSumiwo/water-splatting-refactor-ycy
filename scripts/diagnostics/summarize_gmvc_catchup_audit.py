#!/usr/bin/env python
"""Summarize the GMVC A0 catch-up audit from existing fixed-bank JSON files."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


METRIC_KEYS = [
    "transfer_l1",
    "object_j_variance",
    "closure_signal_floor_l1",
    "consensus_j_reconstruction_l1",
    "object_target_l1",
    "dc_cross_view_variance",
    "dc_recomposition_l1",
]
BANKS = ["evalf", "evalg"]


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_fixed_json(root: Path, variant: str, step: int, bank: str) -> Dict[str, Any]:
    path = root / variant.lower() / f"step{int(step)}" / bank.lower() / "gmvc_fixed_bank_metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf8"))


def _load_rgb_json(root: Path, variant: str, step: int) -> Dict[str, float]:
    path = root / variant.lower() / f"step{int(step)}" / "rgb" / "rgb_metrics.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf8"))
    results = payload.get("results", payload)
    return {
        "psnr": float(results.get("psnr", 0.0)),
        "ssim": float(results.get("ssim", 0.0)),
        "lpips": float(results.get("lpips", 0.0)),
    }


def _heldout_metrics(payload: Mapping[str, Any]) -> Dict[str, float]:
    heldout = payload["metrics"]["heldout"]
    return {key: float(heldout.get(key, 0.0)) for key in METRIC_KEYS}


def _read_variant(
    root: Path,
    variant: str,
    step: int,
    banks: Iterable[str],
) -> Dict[str, Any]:
    fixed: Dict[str, Dict[str, float]] = {}
    source_jsons: Dict[str, str] = {}
    for bank in banks:
        payload = _load_fixed_json(root, variant, step, bank)
        fixed[bank.upper()] = _heldout_metrics(payload)
        source_jsons[bank.upper()] = str(root / variant.lower() / f"step{int(step)}" / bank.lower() / "gmvc_fixed_bank_metrics.json")
    return {
        "variant": variant.upper(),
        "step": int(step),
        "rgb": _load_rgb_json(root, variant, step),
        "fixed": fixed,
        "source_jsons": source_jsons,
    }


def _signed_delta(value: float, base: float) -> Dict[str, float]:
    abs_delta = float(value - base)
    pct_delta = 100.0 * abs_delta / abs(base) if abs(base) > 1e-12 else 0.0
    return {"abs": abs_delta, "pct": float(pct_delta)}


def _metric_delta(value: Mapping[str, float], base: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    return {key: _signed_delta(float(value[key]), float(base[key])) for key in METRIC_KEYS}


def _improvement_positive(value: float, base: float, lower_is_better: bool = True) -> float:
    return float(base - value) if lower_is_better else float(value - base)


def _format_float(value: float) -> str:
    return f"{value:.8f}"


def _markdown_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    out: List[str] = []
    for idx, row in enumerate(rows):
        out.append("| " + " | ".join(str(cell).ljust(widths[col]) for col, cell in enumerate(row)) + " |")
        if idx == 0:
            out.append("| " + " | ".join("-" * widths[col] for col in range(len(row))) + " |")
    return "\n".join(out)


def _build_markdown(summary: Mapping[str, Any]) -> str:
    lines: List[str] = ["# GMVC A0 Catch-up Audit", ""]
    for bank in ["EVALF", "EVALG"]:
        lines.append(f"## {bank} Absolute Metrics")
        rows = [["run", *METRIC_KEYS]]
        for label in summary["run_order"]:
            fixed = summary["runs"][label]["fixed"][bank]
            rows.append([label, *[_format_float(fixed[key]) for key in METRIC_KEYS]])
        lines.append(_markdown_table(rows))
        lines.append("")
        lines.append(f"## {bank} Relative Shrinkage")
        rows = [["metric", "P30_13K-A0_13K", "MHOLD_15K-A0_15K", "delta_relative", "A0_13K_to_15K_improvement"]]
        for key in METRIC_KEYS:
            item = summary["relative_decomposition"][bank][key]
            rows.append(
                [
                    key,
                    _format_float(item["p30_13k_advantage_vs_a0_13k"]["abs"]),
                    _format_float(item["mhold_15k_advantage_vs_a0_15k"]["abs"]),
                    _format_float(item["delta_relative"]["abs"]),
                    _format_float(item["a0_13k_to_15k_improvement_positive"]),
                ]
            )
        lines.append(_markdown_table(rows))
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    specs: List[Tuple[str, Path, str, int]] = [
        ("A0_13000", args.persistence_root, "a0", 13000),
        ("A0_14000", args.release_root, "a0", 14000),
        ("A0_15000", args.release_root, "a0", 15000),
        ("P30_13000", args.persistence_root, "p30", 13000),
        ("MHOLD_13500", args.release_root, "mhold", 13500),
        ("MHOLD_14000", args.release_root, "mhold", 14000),
        ("MHOLD_15000", args.release_root, "mhold", 15000),
    ]
    runs = {
        label: _read_variant(root=root, variant=variant, step=step, banks=args.banks)
        for label, root, variant, step in specs
    }
    relative_decomposition: Dict[str, Dict[str, Any]] = {}
    for bank in [bank.upper() for bank in args.banks]:
        relative_decomposition[bank] = {}
        a0_13 = runs["A0_13000"]["fixed"][bank]
        a0_15 = runs["A0_15000"]["fixed"][bank]
        p30_13 = runs["P30_13000"]["fixed"][bank]
        mhold_15 = runs["MHOLD_15000"]["fixed"][bank]
        p30_adv = _metric_delta(p30_13, a0_13)
        mhold_adv = _metric_delta(mhold_15, a0_15)
        mhold_vs_p30 = _metric_delta(mhold_15, p30_13)
        a0_delta = _metric_delta(a0_15, a0_13)
        for key in METRIC_KEYS:
            delta_rel_abs = mhold_adv[key]["abs"] - p30_adv[key]["abs"]
            relative_decomposition[bank][key] = {
                "p30_13k_advantage_vs_a0_13k": p30_adv[key],
                "mhold_15k_advantage_vs_a0_15k": mhold_adv[key],
                "delta_relative": {
                    "abs": float(delta_rel_abs),
                    "pct_points": float(mhold_adv[key]["pct"] - p30_adv[key]["pct"]),
                },
                "mhold_15k_delta_vs_p30_13k": mhold_vs_p30[key],
                "a0_15k_delta_vs_a0_13k": a0_delta[key],
                "a0_13k_to_15k_improvement_positive": _improvement_positive(a0_15[key], a0_13[key]),
            }
    summary: Dict[str, Any] = {
        "diagnostic": "gmvc_catchup_audit",
        "persistence_root": str(args.persistence_root),
        "release_root": str(args.release_root),
        "banks": [bank.upper() for bank in args.banks],
        "metrics": METRIC_KEYS,
        "run_order": [label for label, *_ in specs],
        "runs": runs,
        "relative_decomposition": relative_decomposition,
        "notes": {
            "signed_delta": "value - base; fixed-bank metrics are lower-is-better.",
            "delta_relative": "(MHOLD_15K - A0_15K) - (P30_13K - A0_13K). Positive means the relative advantage shrank.",
            "a0_13k_to_15k_improvement_positive": "A0_13K - A0_15K for lower-is-better metrics.",
        },
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (args.output_dir / "gmvc_catchup_audit_summary.json")
    output_md = args.output_md or (args.output_dir / "gmvc_catchup_audit_summary.md")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf8")
    output_md.write_text(_build_markdown(summary), encoding="utf8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persistence-root",
        type=Path,
        default=Path("renders/gmvc_fixed_bank_diag_20260805/curasao_r500_profile_persistence_3k"),
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k"),
    )
    parser.add_argument("--banks", nargs="+", default=BANKS, choices=BANKS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    summary = run(args)
    compact = {
        "output": str(args.output_json or (args.output_dir / "gmvc_catchup_audit_summary.json")),
        "banks": summary["banks"],
        "runs": summary["run_order"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
