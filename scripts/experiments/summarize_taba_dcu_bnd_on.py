#!/usr/bin/env python3
"""Collect the complete five-arm DCU screen and its fixed four-scene extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_taba_bnd_on import check_environment
from scripts.experiments.run_taba_dcu_bnd_on import DEFAULT_OUT, PRESETS
from scripts.experiments.summarize_taba_bnd_on import SCENES, STATS, summarize, write_json

DCU_STATS = ("transmission_mean", "transmission_std", "transmission_p10", "transmission_p50", "transmission_p90",
             "transmission_channel_std", "transmission_cv", "q_T_mean", "q_T_std", "q_T_p10", "q_T_p50", "q_T_p90",
             "water_axis_strength", "directional_fallback_ratio", "delta_parallel_magnitude",
             "delta_orthogonal_magnitude", "luminance_logit_change_abs")


def four_scene_sheet(root, variant, kind):
    width, height, label = 480, 320, 48
    sheet = Image.new("RGB", (2 * width, len(SCENES) * (height + label)), "#171717")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    sources = []
    for row, scene in enumerate(SCENES):
        view = sorted((root / "renders/E0" / scene / "step_14999").iterdir())[0].name
        for col, arm in enumerate(("E0", variant)):
            path = root / "renders" / arm / scene / "step_14999" / view / "A2" / (kind + ".png")
            with Image.open(path) as original:
                tile = original.convert("RGB")
                tile.thumbnail((width, height), Image.Resampling.LANCZOS)
            x, y = col * width, row * (height + label)
            draw.text((x + 8, y + 4), scene + " / " + arm + "\n" + view, fill="white", font=font)
            sheet.paste(tile, (x + (width - tile.width) // 2, y + label + (height - tile.height) // 2))
            sources.append(str(path))
    target = root / "contact_sheets" / ("four_scene_E0_vs_" + variant + "_" + kind + ".jpg")
    sheet.save(target, quality=95, subsampling=0)
    return {"path": str(target), "sources": sources, "visualization_only_resize": True,
            "view_rule": "first lexicographic test view in each scene; full-view sheets are also retained"}


def collect(root):
    result = summarize(root, variants=tuple(PRESETS), candidates=("E1", "E2", "E3", "E4"),
                       stats=STATS + DCU_STATS, experiment="TABA-DCU, BND ON")
    lines = ["", "## DCU statistics", "",
             "Transmission summary is across valid visible Gaussian RGB means. Channel std is the mean RGB-channel population std.",
             "q_T/gate/axis statistics include visible invalid samples with documented fallback. Quantiles are means of per-view quantiles.",
             "Parallel and orthogonal magnitudes describe the ungated decomposition; chromatic residual magnitude describes applied logits.",
             "Water axis uses the ungated BND prepass. Its raw RGB direction is not recentered and may change mean logits."]
    for title, columns in (("Transmission", DCU_STATS[:7]), ("Reliability", DCU_STATS[7:12]), ("Directional decomposition", DCU_STATS[12:])):
        lines += ["", "### " + title, "", "| Scene | Arm | " + " | ".join(columns) + " |", "|---|---|" + "---:|" * len(columns)]
        for row in result["diagnostics"]:
            if row["step"] == 14999:
                lines.append("| " + row["scene"] + " | " + row["variant"] + " | " + " | ".join(f"{row[key]:.8g}" for key in columns) + " |")
    if result["four_scene_complete"]:
        lines += ["", "## Four-scene contact sheets", ""]
        for kind in ("underwater", "clear", "clear_raw_display_clamp01"):
            sheet = four_scene_sheet(root, result["selected_variant"], kind)
            result["contact_sheets"].append(sheet)
            lines.append("- " + sheet["path"])
        write_json(root / "summary.json", result)
    report = root / "report.md"
    report.write_text(report.read_text() + "\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    check_environment()
    result = collect(args.output_root.resolve())
    print(json.dumps({"complete_runs": len(result["final_metrics"]), "selected_variant": result["selected_variant"],
                      "four_scene_complete": result["four_scene_complete"]}), flush=True)


if __name__ == "__main__":
    main()
