#!/usr/bin/env python3
"""Collect registered GRAC metrics, reliability diagnostics and comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_grac_bnd_on import DEFAULT_OUT, PRESETS
from scripts.experiments.run_taba_bnd_on import check_environment, write_json
from scripts.experiments.summarize_taba_bnd_on import STATS, summarize
from scripts.experiments.summarize_taba_dcu_bnd_on import four_scene_sheet
from water_splatting.grac import GRAC_STATS, GRAC_QUANTITIES


def collect(root):
    result = summarize(root, variants=tuple(PRESETS), candidates=("E1", "E2", "E3"),
                       stats=STATS + GRAC_STATS, experiment="GRAC, BND ON")
    lines = ["", "## Gaussian reliability", "",
        "All GRAC statistics below are detached and use visible Gaussians (radii > 0).",
        "The gate reads prior history; history updates once after the formal training iteration.",
        "Evaluation and controller/audit renders do not update history. Counts include repeated training views, not distinct cameras.",
        "Scene quantiles are means of per-view quantiles. E0/E2 have no GRAC history: their reliability values are neutral placeholders, not measured evidence.",
        "E3 uses g_grac on the orthogonal residual and clamp(g_grac*g_taba, floor, 1) on the parallel residual.",
        "Cold/invalid GRAC history preserves TABA. Raw TABA water-axis direction can change mean logits.",
        "The reused runner retains taba_result.json/taba_training.jsonl alongside GRAC-named results, logs and snapshot CSVs."]
    for quantity in GRAC_QUANTITIES:
        keys = ["grac_" + quantity + "_" + label for label in ("mean", "p10", "p50", "p90")]
        lines += ["", "### " + quantity, "", "| Scene | Arm | mean | p10 | p50 | p90 |", "|---|---|---:|---:|---:|---:|"]
        for r in result["diagnostics"]:
            if r["step"] == 14999:
                lines.append("| " + r["scene"] + " | " + r["variant"] + " | " + " | ".join(f"{r[k]:.9g}" for k in keys) + " |")
    keys = GRAC_STATS[-6:]
    lines += ["", "### State coverage", "", "| Scene | Arm | " + " | ".join(keys) + " |", "|---|---|" + "---:|" * len(keys)]
    for r in result["diagnostics"]:
        if r["step"] == 14999:
            lines.append("| " + r["scene"] + " | " + r["variant"] + " | " + " | ".join(f"{r[k]:.9g}" for k in keys) + " |")
    if result["four_scene_complete"]:
        lines += ["", "## Four-scene comparisons", ""]
        for kind in ("underwater", "clear", "clear_raw_display_clamp01"):
            sheet = four_scene_sheet(root, result["selected_variant"], kind)
            result["contact_sheets"].append(sheet)
            lines.append("- " + sheet["path"])
        write_json(root / "summary.json", result)
    report = root / "report.md"
    report.write_text(report.read_text() + "\n".join(lines) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    check_environment()
    result = collect(args.output_root.resolve())
    print(json.dumps({"complete_runs":len(result["final_metrics"]), "selected_variant":result["selected_variant"],
                      "four_scene_complete":result["four_scene_complete"]}), flush=True)


if __name__ == "__main__":
    main()
