#!/usr/bin/env python3
"""Collect complete TABA artifacts and matched underwater/clear contact sheets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/taba_chromatic_bnd_on_20260910"
SCENES = ("JapaneseGradens-RedSea", "Panama", "Curasao", "IUI3-RedSea")
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")
STATS = ("mean_transmission", "low_transmission_fraction", "middle_transmission_fraction", "high_transmission_fraction",
         "g_chroma_mean", "g_chroma_p10", "g_chroma_p50", "g_chroma_p90", "invalid_gate_fraction",
         "chromatic_residual_magnitude", "luminance_residual_magnitude", "rgb_object_residual", "rgb_medium_residual",
         "clear_rgb_min", "clear_rgb_max", "intrinsic_rgb_min", "intrinsic_rgb_max", "P_J_gt_1",
         "P_intrinsic_rgb_gt_1", "J_p99", "tau_D_p90")


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")


def read_csv(path):
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(root, scene, variants, kind):
    views = sorted(path.name for path in (root / "renders" / variants[0] / scene / "step_14999").iterdir() if path.is_dir())
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    tile_width, tile_height, label_height = 480, 320, 34
    sheet = Image.new("RGB", (tile_width * len(variants), (tile_height + label_height) * len(views)), "#171717")
    draw = ImageDraw.Draw(sheet)
    sources = []
    for row, view in enumerate(views):
        for col, variant in enumerate(variants):
            path = root / "renders" / variant / scene / "step_14999" / view / "A2" / (kind + ".png")
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
            x, y = col * tile_width, row * (tile_height + label_height)
            draw.text((x + 10, y + 7), variant + " / " + view, fill="white", font=font)
            sheet.paste(image, (x + (tile_width - image.width) // 2, y + label_height + (tile_height - image.height) // 2))
            sources.append(str(path))
    target = root / "contact_sheets" / (scene + "_" + "_vs_".join(variants) + "_" + kind + ".jpg")
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=95, subsampling=0)
    return {"path": str(target), "sources": sources, "visualization_only_resize": True}


def summarize(root, *, variants=("E0", "E1", "E2"), candidates=("E1", "E2"),
              stats=STATS, experiment="chromatic-only TABA, BND ON", exact_fallback_variants=(),
              selector=None, selection_rule=None):
    results, diagnostics, states = [], [], []
    for scene in SCENES:
        for variant in variants:
            directory = root / scene / variant
            if not directory.exists():
                continue
            state = {"scene": scene, "variant": variant, "path": str(directory),
                     "checkpoint_paths": [str(path) for path in sorted((directory / "checkpoints").glob("*.ckpt"))]}
            metrics_path = directory / "evaluation_metrics.csv"
            state["complete"] = metrics_path.is_file() and (directory / "taba_result.json").is_file()
            states.append(state)
            if not state["complete"]:
                continue
            snapshots = {path.name for path in (directory / "checkpoints").glob("*.ckpt")}
            assert all(f"step-{step:09d}.ckpt" in snapshots for step in (5000, 10000, 14999))
            for row in read_csv(metrics_path):
                results.append({"scene": scene, "variant": variant, "step": int(row["absolute_step"]),
                                **{key: float(row[key]) for key in METRICS}})
            per_view = read_csv(directory / "per_view_metrics.csv")
            for step in (5000, 10000, 14999):
                rows = [row for row in per_view if int(row["absolute_step"]) == step]
                assert rows
                diagnostics.append({"scene": scene, "variant": variant, "step": step,
                    **{key: (min if key.endswith("_min") else max if key.endswith("_max") else statistics.mean)(
                        float(row[key]) for row in rows) for key in stats}})
            state["gradient_audit_passed"] = read_json(directory / "gradient_routing_audit.json")["GRADIENT_ROUTING_AUDIT"]
            training = read_json(directory / "training_summary.json")
            state["ocmc_projector_unchanged"] = training["ocmc_projector_unchanged"]
            training_rows = read_csv(directory / "training_summary_A2.csv")
            state["cica_escape_max"] = max(float(row["cica_escape_grad_l2"]) for row in training_rows)
            logged = [json.loads(line) for line in (directory / "taba_training.jsonl").read_text().splitlines()]
            state["logged_max_P_J_gt_1"] = max(row["P_J_gt_1"] for row in logged)
            state["logged_max_P_intrinsic_rgb_gt_1"] = max(row["P_intrinsic_rgb_gt_1"] for row in logged)
            state["integrity_passed"] = (state["gradient_audit_passed"] and state["ocmc_projector_unchanged"]
                                         and state["cica_escape_max"] == 0
                                         and state["logged_max_P_J_gt_1"] == 0
                                         and state["logged_max_P_intrinsic_rgb_gt_1"] == 0)
    final = [row for row in results if row["step"] == 14999]
    pairings = []
    for scene in SCENES:
        scene_variants = [row["variant"] for row in final if row["scene"] == scene]
        if "E0" not in scene_variants or len(scene_variants) < 2:
            continue
        records = {variant: read_json(root / scene / variant / "start_state_equivalence.json") for variant in scene_variants}
        sequences = [read_json(root / scene / variant / "camera_sequence_hashes.json")["sha256"] for variant in scene_variants]
        banks = [read_json(root / scene / variant / "cica_camera_bank.json")["bank_sha256"] for variant in scene_variants]
        keys = ("source_checkpoint_sha256", "checkpoint_model_hash", "optimizer_hash", "scheduler_hash", "ocmc_projector_sha256")
        checks = {key: len({record[key] for record in records.values()}) == 1 for key in keys}
        checks["camera_sequence"] = len(set(sequences)) == 1
        checks["cica_camera_bank"] = len(set(banks)) == 1
        checks["training_rng"] = all(record["training_rng_manifest"] == records["E0"]["training_rng_manifest"] for record in records.values())
        checks["start_state"] = all(record["START_STATE_EQUIVALENCE"] for record in records.values())
        pairings.append({"scene": scene, "variants": scene_variants, "checks": checks, "passed": all(checks.values())})
    means = []
    for label, scenes in (("two_scene_mean", SCENES[:2]), ("four_scene_mean", SCENES)):
        for variant in variants:
            selected = [row for row in final if row["variant"] == variant and row["scene"] in scenes]
            if len(selected) == len(scenes):
                means.append({"scene": label, "variant": variant, "step": 14999,
                              **{key: statistics.mean(row[key] for row in selected) for key in METRICS}})
    sheets = []
    for scene in SCENES[:2]:
        if all(any(row["scene"] == scene and row["variant"] == variant for row in final) for variant in variants):
            for kind in ("underwater", "clear", "clear_raw_display_clamp01"):
                sheets.append(contact_sheet(root, scene, variants, kind))
    stable_variants = []
    for variant in candidates:
        stable = all(
            any(state["scene"] == scene and state["variant"] == variant and state.get("integrity_passed") for state in states)
            and any(pair["scene"] == scene and pair["passed"] for pair in pairings)
            and all(row["P_J_gt_1"] == 0 and row["P_intrinsic_rgb_gt_1"] == 0
                    and ((row["intrinsic_rgb_min"] >= 0 and row["intrinsic_rgb_max"] <= 1
                          and row["gcrc_corrected_intrinsic_min"] > 0 and row["gcrc_corrected_intrinsic_max"] < 1)
                         if variant in exact_fallback_variants else
                         (row["intrinsic_rgb_min"] > 0 and row["intrinsic_rgb_max"] < 1))
                    for row in diagnostics if row["scene"] == scene and row["variant"] == variant)
            for scene in SCENES[:2]
        )
        if stable:
            stable_variants.append(variant)
    eligible = [row for row in means if row["scene"] == "two_scene_mean" and row["variant"] in stable_variants]
    all_finished = all(any(row["scene"] == "two_scene_mean" and row["variant"] == variant for row in means) for variant in variants)
    selected_variant = max(eligible, key=lambda row: row["PSNR"])["variant"] if eligible and all_finished else None
    if selector is not None and eligible and all_finished:
        selected_variant = selector(eligible)
        assert selected_variant in stable_variants
    if selected_variant:
        for scene in SCENES:
            if all(any(row["scene"] == scene and row["variant"] == variant for row in final) for variant in ("E0", selected_variant)):
                for kind in ("underwater", "clear", "clear_raw_display_clamp01"):
                    sheets.append(contact_sheet(root, scene, ("E0", selected_variant), kind))
    summary = {"experiment": experiment, "root": str(root), "states": states,
               "final_metrics": final, "means": means, "diagnostics": diagnostics, "pairings": pairings,
               "selected_variant": selected_variant, "stable_variants": stable_variants,
               "selection_rule": selection_rule or ("highest two-scene mean final PSNR among stable " + "/".join(candidates) + " after all arms finish; exploratory selection on these two scenes"),
               "four_scene_complete": any(row["scene"] == "four_scene_mean" and row["variant"] != "E0" for row in means),
               "contact_sheets": sheets,
               "quantile_aggregation": "scene diagnostics are unweighted means of per-eval-view statistics, including quantiles",
               "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
               "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)}
    write_json(root / "summary.json", summary)
    write_csv(root / "metrics_all_steps.csv", results + means)
    write_csv(root / "diagnostics_all_steps.csv", diagnostics)
    write_csv(root / "metrics_final.csv", final + means)
    lines = ["# " + experiment + " training report", "", "BND ON; M1 + OCMC + CICA; Auxiliary/RAOC/BRASH/BVAM OFF; seed 42.", "",
             "Metrics use project-local metrics.py on native-resolution, exactly paired RGB tensors in [0,1].",
             "No metric resize, crop or extra color conversion. Means weight scenes equally.", "",
             "| Scene | Variant | PSNR | SSIM | LPIPS | MSE |", "|---|---|---:|---:|---:|---:|"]
    for row in final + means:
        lines.append(f"| {row['scene']} | {row['variant']} | {row['PSNR']:.6f} | {row['SSIM']:.6f} | {row['LPIPS']:.6f} | {row['MSE']:.9f} |")
    lines += ["", "Four-scene mean: " + ("complete" if summary["four_scene_complete"] else "not available; incomplete experiments are not imputed"), "",
              "## Bounded diagnostics", "", "| Scene | Variant | P(J>1) | P(intrinsic>1) | J p99 | tau_D p90 | gate mean |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in diagnostics:
        if row["step"] == 14999:
            lines.append(f"| {row['scene']} | {row['variant']} | {row['P_J_gt_1']:.6g} | {row['P_intrinsic_rgb_gt_1']:.6g} | {row['J_p99']:.6f} | {row['tau_D_p90']:.6f} | {row['g_chroma_mean']:.6f} |")
    lines += ["", "J is the raw, unclamped compositor output. Quantiles above are means of per-view quantiles.",
              "Native clear.png uses the existing J/(1+J) display mapping. Additional clear_raw_display_clamp01.png shows raw J.",
              "Transmission fractions are over image pixels: low <0.33, middle [0.33,0.66), high >=0.66.",
              "Gate and SH residual statistics are over visible Gaussians; invalid samples use gate=1.",
              "rgb_object residual = mean |rgb_object - GT|; rgb_medium residual = mean |rgb_medium - (GT - rgb_object)|.",
              "Neither residual is an error against a separately observed object/medium ground truth."]
    for title, columns in (
        ("Transmission", ("mean_transmission", "low_transmission_fraction", "middle_transmission_fraction", "high_transmission_fraction")),
        ("Chromatic gate", ("g_chroma_mean", "g_chroma_p10", "g_chroma_p50", "g_chroma_p90", "invalid_gate_fraction")),
        ("Residuals", ("chromatic_residual_magnitude", "luminance_residual_magnitude", "rgb_object_residual", "rgb_medium_residual")),
        ("RGB bounds", ("clear_rgb_min", "clear_rgb_max", "intrinsic_rgb_min", "intrinsic_rgb_max")),
    ):
        lines += ["", "## " + title, "", "| Scene | Variant | " + " | ".join(columns) + " |",
                  "|---|---|" + "---:|" * len(columns)]
        for row in diagnostics:
            if row["step"] == 14999:
                values = " | ".join(f"{row[key]:.9g}" for key in columns)
                lines.append(f"| {row['scene']} | {row['variant']} | {values} |")
    lines += ["", "## Pairing and integrity", "",
              "Selected extension variant: " + str(selected_variant) + ". " + summary["selection_rule"] + ".",
              "", "| Scene | Variants | Pairing passed |", "|---|---|---|"]
    for pair in pairings:
        lines.append(f"| {pair['scene']} | {', '.join(pair['variants'])} | {pair['passed']} |")
    lines += ["", "| Scene | Variant | Complete | Gradient audit | Integrity |", "|---|---|---|---|---|"]
    for state in states:
        lines.append(f"| {state['scene']} | {state['variant']} | {state['complete']} | {state.get('gradient_audit_passed', False)} | {state.get('integrity_passed', False)} |")
    lines += ["", "## Artifacts", "", "Full checkpoints: <root>/<scene>/<variant>/checkpoints/step-{000005000,000010000,000014999}.ckpt",
              "Renders: <root>/renders/<variant>/<scene>/step_<step>/<view>/A2/{underwater,clear,clear_raw_display_clamp01,gt}.png",
              "Metrics and audits: <root>/<scene>/<variant>/{evaluation_metrics.csv,per_view_metrics.csv,gradient_routing_audit.json}",
              "Training diagnostics: taba_training.jsonl and taba_statistics_step_<step>.csv in each experiment directory.", ""]
    lines.extend("- " + sheet["path"] for sheet in sheets)
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = summarize(args.output_root.resolve())
    print(json.dumps({"complete_runs": len(result["final_metrics"]), "selected_variant": result["selected_variant"],
                      "report": str(args.output_root / "report.md")}), flush=True)


if __name__ == "__main__":
    main()
