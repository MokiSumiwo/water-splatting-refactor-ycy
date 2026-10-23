#!/usr/bin/env python3
"""Aggregate independently reloaded D1/FSB/BND metrics and all-view contact sheets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
METHODS = ("BND_CICA", "FSB", "D1")
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def image_row(comp, scene, view, clear=False, crop=False, width=360,
              candidate_method="D1", candidate_label="D1 (1, .75, .5)"):
    kind = "clear" if clear else "underwater"
    columns = [("BND_CICA", "BND ON + CICA", kind), ("FSB", "FSB", kind),
               (candidate_method, candidate_label, kind), (candidate_method, f"abs({candidate_method} - FSB) x4", kind+"_diff_fsb_x4")]
    if not clear:
        columns.insert(0, ("BND_CICA", "Underwater GT", "gt"))
    tiles = []
    for method, title, typ in columns:
        with Image.open(comp / scene / method / view / (typ+".png")) as im:
            im = im.convert("RGB")
            if crop:
                w, h = im.size
                im = im.crop((int(.3*w), int(.3*h), int(.7*w), int(.7*h)))
            im = im.resize((width, round(im.height*width/im.width)), Image.Resampling.LANCZOS)
            tiles.append((title, im))
    height = max(im.height for _, im in tiles)
    canvas = Image.new("RGB", (width*len(tiles), height+62), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 14)
    subtitle = "clear: no GT; unchanged display" if clear else "native test camera; identical display"
    if crop:
        subtitle += "; fixed center 40% crop"
    draw.text((8, 4), f"{scene} / {view} | {subtitle}", font=font, fill="#111111")
    for i, (title, im) in enumerate(tiles):
        draw.text((i*width+8, 29), title, font=font, fill="#111111")
        canvas.paste(im, (i*width, 62))
    return canvas


def sheet(rows, target):
    width = max(im.width for im in rows)
    canvas = Image.new("RGB", (width, sum(im.height for im in rows)), "white")
    y = 0
    for im in rows:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(target, quality=95, subsampling=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    a = p.parse_args()
    candidate = a.candidate.resolve()
    comp = candidate / "comparison"
    out = comp / "summary"
    out.mkdir(exist_ok=False)
    all_rows, means, degree, audits, inputs = [], [], [], {}, []
    visual_rows = {False: [], True: []}
    for scene in SCENES:
        status = json.loads((comp/scene/"status.json").read_text())
        assert status["complete"]
        pairing = json.loads((comp/scene/"protocol_pairing.json").read_text())
        assert all(pairing.values())
        rows = csv_rows(comp/scene/"per_view_metrics.csv")
        degree.extend(dict(scene=scene, **row) for row in csv_rows(comp/scene/"degree_diagnostics.csv"))
        audits[scene] = dict(pairing=pairing, math=json.loads((comp/scene/"math_audit.json").read_text()))
        for row in csv_rows(comp/scene/"render_manifest.csv"):
            path = Path(row["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
            inputs.append(row)
        all_rows.extend(rows)
        for method in METHODS:
            group = [r for r in rows if r["method"] == method]
            assert len(group) == len(status["evaluation_views"])
            keys = METRICS + ("clear_black_fraction", "accumulation_mean", "coverage_gt_001", "render_seconds_median", "peak_render_allocated_mb")
            means.append(dict(scene=scene, method=method, view_count=len(group), **{k: mean(group, k) for k in keys}))
        for clear in (False, True):
            kind = "clear" if clear else "underwater"
            scene_rows = []
            crop_rows = []
            for view in status["evaluation_views"]:
                row = image_row(comp, scene, view, clear=clear)
                row.save(out/f"{scene}_{view}_{kind}.jpg", quality=95, subsampling=0)
                scene_rows.append(row)
                crop_rows.append(image_row(comp, scene, view, clear=clear, crop=True))
            sheet(scene_rows, out/f"{scene}_{kind}_all_views.jpg")
            sheet(crop_rows, out/f"{scene}_{kind}_center_crops.jpg")
            visual_rows[clear].extend(scene_rows)
    for clear, rows in visual_rows.items():
        sheet(rows, out/("clear_all_views.jpg" if clear else "underwater_all_views.jpg"))
    for method in METHODS:
        group = [r for r in means if r["method"] == method]
        keys = METRICS + ("clear_black_fraction", "accumulation_mean", "coverage_gt_001", "render_seconds_median", "peak_render_allocated_mb")
        means.append(dict(scene="MACRO", method=method, view_count=sum(r["view_count"] for r in group), **{k: mean(group, k) for k in keys}))
    deltas = []
    for scene in (*SCENES, "MACRO"):
        d = next(r for r in means if r["scene"] == scene and r["method"] == "D1")
        for reference in ("BND_CICA", "FSB"):
            b = next(r for r in means if r["scene"] == scene and r["method"] == reference)
            deltas.append(dict(scene=scene, reference=reference, **{k: d[k]-b[k] for k in METRICS}))
    view_deltas = []
    for d in all_rows:
        if d["method"] != "D1":
            continue
        for ref in ("BND_CICA", "FSB"):
            b = next(r for r in all_rows if r["scene"] == d["scene"] and r["view_id"] == d["view_id"] and r["method"] == ref)
            view_deltas.append(dict(scene=d["scene"], view_id=d["view_id"], reference=ref, **{k: float(d[k])-float(b[k]) for k in METRICS}))
    write_csv(out/"scene_metrics.csv", means)
    write_csv(out/"metric_deltas.csv", deltas)
    write_csv(out/"per_view_metrics.csv", all_rows)
    write_csv(out/"per_view_deltas.csv", view_deltas)
    write_csv(out/"degree_diagnostics.csv", degree)
    write_csv(out/"input_render_manifest.csv", inputs)
    (out/"audit.json").write_text(json.dumps(audits, indent=2)+"\n")
    lines = ["# Degree-wise FSB D1: Strict Four-Scene Results", "",
             "D1 = fixed SH-degree coefficient scales (1, 0.75, 0.50) before the FSB headroom map.",
             "All checkpoints are step 14999. Scenes receive equal weight; each scene averages its test views.",
             "No test view selects coefficients, checkpoint, or display settings. D1 was a prespecified milder candidate; 5k smoke was engineering validation only.", "",
             "| Scene | BND+CICA PSNR | FSB PSNR | D1 PSNR | D1-FSB PSNR | D1-FSB SSIM | D1-FSB LPIPS | D1-FSB MSE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scene in (*SCENES, "MACRO"):
        vals = [next(r for r in means if r["scene"] == scene and r["method"] == m) for m in METHODS]
        d = next(r for r in deltas if r["scene"] == scene and r["reference"] == "FSB")
        lines.append(f"| {scene} | {vals[0]['PSNR']:.6f} | {vals[1]['PSNR']:.6f} | {vals[2]['PSNR']:.6f} | {d['PSNR']:+.6f} | {d['SSIM']:+.6f} | {d['LPIPS']:+.6f} | {d['MSE']:+.9f} |")
    lines += ["", "## Macro Metrics", "", "| Method | PSNR | SSIM | LPIPS | MSE |", "|---|---:|---:|---:|---:|"]
    for r in means:
        if r["scene"] == "MACRO":
            lines.append(f"| {r['method']} | {r['PSNR']:.6f} | {r['SSIM']:.6f} | {r['LPIPS']:.6f} | {r['MSE']:.9f} |")
    lines += ["", "## Evidence Boundaries", "",
              "- The three methods share source model/optimizer/scheduler/projector hashes, continuation RNG, and camera sequence. All start and gradient routing audits passed.",
              "- Checkpoints were independently reloaded using each experiment manifest, and all 13 test views reproduced stored PSNR/SSIM/LPIPS/MSE within the evaluator tolerances. Original checkpoints and images were retained.",
              "- Older checkpoints do not embed fixed degree weights; the manifest is required. Reload audits record the exact checkpoint/manifest hashes and effective coefficients.",
              "- SH linearity makes this implementation equivalent to multiplying each trainable SH coefficient by its fixed degree scale. Nonzero scales can be absorbed into trainable coefficients. This changes initialization and optimization, not the representable color set. It is not a per-degree convex blend between BND and FSB.",
              "- Forward/VJP equivalence and inverse coefficient rescaling were tested on 4096 sampled Gaussians per scene. Per-degree effective coefficient energy and cumulative headroom utilization are measured on three deterministic training cameras only, with at most 65536 visible Gaussians per camera; they are not per-ray contribution statistics.",
              "- Normal joint training can alter geometry, opacity, and medium indirectly. These parameters were not frozen; no new geometry/medium loss or renderer changes were added.",
              "- Clear has no paired GT. Black fraction below mean RGB 0.02, coverage A>0.01, and pixel differences are descriptive proxies only. No exposure adjustment or auto-normalization is applied.",
              "- Single-seed results do not establish statistical significance or SOTA. D2 has only 5k smoke, not a negative full-training result.", "",
              "## Visual Comparisons", "",
              "[All underwater test views](underwater_all_views.jpg): GT, BND+CICA, FSB, D1, absolute D1-FSB difference x4.", "",
              "[All clear test views](clear_all_views.jpg): BND+CICA, FSB, D1, absolute D1-FSB difference x4. No clear GT.", ""]
    for scene in SCENES:
        lines.append(f"- {scene}: [underwater]({scene}_underwater_all_views.jpg), [clear]({scene}_clear_all_views.jpg), [underwater center crops]({scene}_underwater_center_crops.jpg), [clear center crops]({scene}_clear_center_crops.jpg)")
    lines += ["", "## Warm Render Measurements", "",
              "Full get_outputs_for_camera, one warmup + median of three calls per view; scene means, no metric computation or PNG saving included. Peak allocated CUDA memory includes model and outputs, not reserved memory. This is a small diagnostic sample, not a controlled throughput benchmark.", "",
              "| Scene | Method | Mean render ms | Peak allocated MB (view mean) | Clear black fraction | Mean accumulation | Coverage A>0.01 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in means:
        lines.append(f"| {r['scene']} | {r['method']} | {r['render_seconds_median']*1000:.3f} | {r['peak_render_allocated_mb']:.1f} | {r['clear_black_fraction']:.6f} | {r['accumulation_mean']:.6f} | {r['coverage_gt_001']:.6f} |")
    lines += ["", "## Decision", "",
              "D1 loses PSNR to strict FSB in all four scenes; its small SSIM improvement does not satisfy the joint PSNR/SSIM objective. Do not promote D1 or launch D2 15k merely because smoke passed. Retain strict FSB as the current candidate.",
              "Next evaluate a prespecified step-dependent FSB curriculum with training-only zero-step transition/Jacobian and short-run diagnostics. Establish a genuine validation protocol before choosing schedules; repeated existing test-set comparisons remain exploratory. Fixed positive degree scaling alone is insufficient evidence for a new bounded-capacity method.", ""]
    (out/"REPORT.md").write_text("\n".join(lines))
    print(out/"REPORT.md")
    print(json.dumps([r for r in means if r["scene"] == "MACRO"], indent=2))


if __name__ == "__main__":
    main()
