#!/usr/bin/env python3
"""Collect fixed GCRC experiments, metrics and contamination diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_gcrc_bnd_on import DEFAULT_OUT, PRESETS
from scripts.experiments.run_taba_bnd_on import check_environment, write_json
from scripts.experiments.summarize_taba_bnd_on import STATS, summarize
from scripts.experiments.summarize_taba_dcu_bnd_on import four_scene_sheet
from water_splatting.gcrc import GCRC_STATS, GCRC_QUANTITIES


def collect(root):
    extra = ("transmission_mean", "transmission_std", "transmission_p10", "transmission_p50", "transmission_p90")
    result = summarize(root, variants=tuple(PRESETS), candidates=("E1", "E2", "E3"),
        stats=STATS+extra+GCRC_STATS, experiment="GCRC, BND ON", exact_fallback_variants=("E2", "E3"))
    lines = ["", "## Contamination diagnostics", "",
        "GCRC factors and history are detached. Statistics use visible Gaussians (radii > 0).",
        "Current gates use prior history; history updates once after the formal training iteration.",
        "Evaluation/controller/audit forwards do not update history. Counts include repeated cameras and inherited history.",
        "Scene quantiles are means of per-view quantiles. E0/E1 have no GCRC history; their GCRC values are neutral placeholders.",
        "gcrc_transmission is the rounded-center Gaussian prepass transmission; transmission_* summarizes valid samples via the existing TABA helper.",
        "E3 applies GCRC then existing directional TABA, using both controllers from the ungated BND prepass.",
        "The raw RGB water axis is not recentered, so directional correction can change mean logits.",
        "alpha=0 preserves baseline sigmoid bits (or TABA exactly in E3); corrected samples use the unchanged safe_sigmoid.",
        "Original sigmoid endpoints are permitted only in exact zero-correction fallback; all RGB/J must remain in [0,1].",
        "The first JSONL step5000 row is a source-state gradient audit, not a trained snapshot; formal logging starts at step3500.",
        "Legacy taba_result/log/snapshot companions are retained alongside GCRC files."]
    for quantity in GCRC_QUANTITIES:
        keys = ["gcrc_"+quantity+"_"+label for label in ("mean", "std", "p10", "p50", "p90")]
        lines += ["", "### "+quantity, "", "| Scene | Arm | mean | std | p10 | p50 | p90 |", "|---|---|---:|---:|---:|---:|---:|"]
        for r in result["diagnostics"]:
            if r["step"]==14999:
                lines.append("| "+r["scene"]+" | "+r["variant"]+" | "+" | ".join(f"{r[k]:.9g}" for k in keys)+" |")
    keys=GCRC_STATS[len(GCRC_QUANTITIES)*5:]
    lines += ["", "### State coverage and bounds", "", "| Scene | Arm | "+" | ".join(keys)+" |", "|---|---|"+"---:|"*len(keys)]
    for r in result["diagnostics"]:
        if r["step"]==14999:
            lines.append("| "+r["scene"]+" | "+r["variant"]+" | "+" | ".join(f"{r[k]:.9g}" for k in keys)+" |")
    if result["four_scene_complete"]:
        for kind in ("underwater", "clear", "clear_raw_display_clamp01"):
            sheet=four_scene_sheet(root,result["selected_variant"],kind)
            result["contact_sheets"].append(sheet)
            lines.append("- "+sheet["path"])
        write_json(root/"summary.json",result)
    report=root/"report.md"
    report.write_text(report.read_text()+"\n".join(lines)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    check_environment()
    result=collect(args.output_root.resolve())
    print(json.dumps({"complete_runs":len(result["final_metrics"]),"selected_variant":result["selected_variant"],"four_scene_complete":result["four_scene_complete"]}),flush=True)


if __name__=="__main__":
    main()
