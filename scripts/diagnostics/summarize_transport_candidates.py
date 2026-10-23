#!/usr/bin/env python3
"""Verify retained candidate artifacts and create a reproducible visual index."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.experiments.run_taba_bnd_on import check_environment, read_json, write_json

VIEWS = {
    "JapaneseGradens-RedSea": ("MTN_1090", "MTN_1098", "MTN_1106"),
    "Panama": ("MTN_1539", "MTN_1529", "MTN_1547"),
    "Curasao": ("MTN_1288", "MTN_1296", "MTN_1304"),
    "IUI3-RedSea": ("MTN_5903", "MTN_5894", "MTN_5911", "MTN_5928"),
}
REAL = ("02_world_medium", "04_forward_scattering", "06_radiometric_domain", "09_clear_feasible_family")
PANELS = {
    REAL[0]: (("Original underwater", "underwater_original.png"), ("Ray MLP", "ray_mlp.png"),
              ("World grid", "world_grid.png"), ("Grid difference x10", "world_grid_difference_x10.png")),
    REAL[1]: (("Original underwater", "underwater_original.png"), ("Camera PSF", "camera_psf.png"),
              ("Water PSF", "water_psf.png"), ("Water difference x20", "water_psf_difference_x20.png")),
    REAL[2]: (("Original underwater", "underwater_original.png"), ("Encoded fit", "encoded_underwater.png"),
              ("Linear-sRGB fit", "linear_srgb_underwater.png"), ("Original clear", "clear_original.png"),
              ("Linear-sRGB clear", "linear_srgb_clear.png"), ("Clear difference x10", "linear_srgb_clear_difference_x10.png")),
    REAL[3]: (("Original underwater", "underwater_original.png"), ("Lower underwater", "lower_underwater.png"),
              ("Higher underwater", "higher_underwater.png"), ("Original clear", "clear_original.png"),
              ("Lower clear", "lower_clear.png"), ("Higher clear", "higher_clear.png"),
              ("Finite-family range x4", "restricted_clear_range_x4.png")),
}


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sheet(base, experiment, scene, output):
    panels = PANELS[experiment]
    width, height, gap = 280, 186, 12
    canvas = Image.new("RGB", (len(panels)*(width+gap)+gap, 50+len(VIEWS[scene])*(height+55)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((gap, 10), experiment+" / "+scene+" / all heldout views", font=font, fill="black")
    for row, view in enumerate(VIEWS[scene]):
        y = 50+row*(height+55)
        for column, (label, name) in enumerate(panels):
            path = base/experiment/scene/"renders"/view/name
            assert path.exists(), path
            with Image.open(path) as image:
                panel = ImageOps.pad(image.convert("RGB"), (width, height), color="#ededed")
            x = gap+column*(width+gap)
            draw.text((x, y), view+" / "+label, font=font, fill="black")
            canvas.paste(panel, (x, y+25))
    target = output/(experiment+"_"+scene+".png")
    canvas.save(target)
    return target.name


def score_plot(base, output):
    entries = [read_json(base/name/"summary.json") for name in REAL[:3]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    specifications = (
        ("World medium grid", entries[0]["comparisons"], "grid_minus_mlp_psnr", "grid_minus_original_psnr"),
        ("Distance-dependent scattering", entries[1]["deltas"], "psnr_water_minus_camera", "psnr_water_minus_original"),
        ("Assumed linear-sRGB transport", entries[2]["deltas"], "psnr_linear_minus_encoded", "psnr_linear_minus_original"),
    )
    x = np.arange(4)
    for axis, (title, rows, control_key, original_key) in zip(axes, specifications):
        axis.bar(x-.18, [row[control_key] for row in rows], .36, label="Versus matched control", color="#2c7b70")
        axis.bar(x+.18, [row[original_key] for row in rows], .36, label="Versus original E0", color="#b35454")
        axis.axhline(0, color="#333333", linewidth=.8)
        axis.set_xticks(x)
        axis.set_xticklabels(["Japanese", "Panama", "Curasao", "IUI3"], rotation=20)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("Mean per-view PSNR difference (dB)")
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle("Frozen-Gaussian screens: all three promotion gates failed; axes have different scales", fontsize=12)
    fig.savefig(output/"paired_psnr.png", dpi=160)
    plt.close(fig)


def run(base, output):
    output.mkdir(parents=True, exist_ok=False)
    expected = {}
    first = base/"01_optical_path/verified_v2"
    assert read_json(first/"completion_verification.json")["passed"]
    for name, key in ((first/"registration.json", "frozen_manifest"), (base/REAL[0]/"registration.json", "manifest")):
        for record in read_json(name)[key]:
            path = Path(record["path"])
            path = path if path.is_absolute() else ROOT/path
            expected[str(path)] = record["sha256"]
    native = []
    for scene in VIEWS:
        for arm in ("E0", "E2"):
            audit = read_json(first/scene/arm/"audit.json")
            assert audit["complete"] and audit["model_state_unchanged"]
            assert set(r["view_id"] for r in audit["rows"]) == set(VIEWS[scene])
            expected[audit["checkpoint"]] = audit["checkpoint_sha256"]
            native.extend(audit["rows"])
    for row in native:
        assert max(row["baseline_equivalence"].values()) < 2e-6
        assert row["controls"]["center_integrated"]["changed_gt_1_over_255_fraction"] == 0
    runs = []
    certificates = []
    raw_checks = []
    for experiment in REAL:
        assert read_json(base/experiment/"summary.json")["complete"]
        assert all(status == 0 for status in read_json(base/experiment/"queue_status.json").values())
        for scene, views in VIEWS.items():
            directory = base/experiment/scene
            result = read_json(directory/"result.json")
            assert result["complete"]
            expected[result["checkpoint"]] = result["checkpoint_sha256"]
            assert set(row["view_id"] for row in result["rows"]) == set(views)
            for row in result["rows"]:
                for value in row.values():
                    if isinstance(value, float):
                        assert np.isfinite(value)
            for path in (directory/"renders").rglob("*.png"):
                with Image.open(path) as image:
                    image.verify()
            runs.append(dict(experiment=experiment, scene=scene, views=len(views), rows=len(result["rows"])))
            if experiment != REAL[3]:
                continue
            cert = read_json(directory/"training_certification.json")
            train_views = set(row["view"] for row in cert["rows"])
            assert len(train_views) == {"JapaneseGradens-RedSea": 17, "Panama": 15, "Curasao": 18, "IUI3-RedSea": 25}[scene]
            assert not train_views.intersection(views)
            for direction, scale in cert["selected"].items():
                rows = [r for r in cert["rows"] if r["direction"] == direction and r["scale"] == scale]
                maximum = max(r["underwater_rmse"] for r in rows)
                assert len(rows) == len(train_views) and maximum <= cert["epsilon"]
                certificates.append(dict(scene=scene, direction=direction, scale=scale, train_views=len(rows),
                    max_full_training_view_rmse=maximum, max_training_pixel_error=max(r["underwater_max_abs"] for r in rows)))
            for view in views:
                with np.load(directory/"renders"/view/"raw_alternatives.npz") as arrays:
                    assert all(value.dtype == np.float32 and np.isfinite(value).all() for value in arrays.values())
                    stack = np.stack([arrays[key] for key in ("original_clear", "lower_clear", "higher_clear")])
                    spread = stack.max(0)-stack.min(0)
                    raw_checks.append(dict(scene=scene, view=view, mean_restricted_clear_range=float(spread.mean()),
                        max_restricted_clear_range=float(spread.max()), shape=list(spread.shape)))
    synthetic = []
    for candidate in (3, 5, 6, 7, 8, 9, 10):
        result = read_json(base/(f"{candidate:02d}_synthetic_calibrated")/"result.json")
        assert result["complete"] and result["synthetic_only"] and not result["real_four_scene_validation"]
        synthetic.append(dict(candidate=candidate, synthetic_only=True))
    hashes = []
    for path, expected_hash in expected.items():
        actual = checksum(Path(path))
        assert actual == expected_hash, path
        hashes.append(dict(path=path, sha256=actual, unchanged=True))
    images = {experiment: [sheet(base, experiment, scene, output) for scene in VIEWS] for experiment in REAL}
    score_plot(base, output)
    write_json(output/"verification.json", dict(passed=True, native_views=len(native), real_runs=runs,
        source_and_registered_hashes=hashes, ambiguity_full_training_certificates=certificates,
        raw_ambiguity_checks=raw_checks, synthetic_experiments=synthetic, real_training_models=28,
        real_training_steps_per_model=5000, ambiguity_searches=8, ambiguity_steps_per_search=1500,
        full_joint_gaussian_training_completed=False, sota_claim=False, clear_accuracy_claim=False))
    lines = ["# Optical Transport Candidate Experiment Index", "",
        "All currently feasible screens finished. None of the three real-data paired training screens passed its promotion gate.",
        "28 frozen-Gaussian model fits x 5000 updates; 8 restricted ambiguity searches x 1500 updates; 26 thickness diagnostic views.",
        "Candidates 3, 5, 7, 8 and 10 have synthetic evidence only. No full joint Gaussian training or SOTA result is claimed.", "",
        "[Research protocol and limitations](../../../research_notes/TRANSPORT_CANDIDATES_SEQUENTIAL_EXPERIMENT_2026-09-11.md)",
        "[Verification manifest](verification.json)", "", "![Paired local PSNR](paired_psnr.png)", "",
        "## Complete Test-View Sheets", "",
        "Each sheet includes every heldout view for that scene. PNG previews are clamped to [0,1]. Difference panels use the labeled gain.",
        "Clear brightness, black fractions and ambiguity ranges do not measure clear reconstruction accuracy.", ""]
    for experiment, sheets in images.items():
        lines.extend(["### "+experiment, "", f"[Raw summary](../{experiment}/summary.json)", ""])
        lines.extend(f"- [{scene}]({filename})" for scene, filename in zip(VIEWS, sheets))
        lines.append("")
    lines.extend(["## Restricted Clear Ambiguity", "", "Every selected alternative satisfies RGB RMSE <= 1/255 on every full training view.",
        "This is an average image constraint and permits larger local errors. Test views did not select alternatives.",
        "The range is from a finite global post-transform family, not a posterior or global uncertainty bound.", "",
        "| Scene | Mean clear range | Largest training pixel error |", "| --- | ---: | ---: |"])
    for scene in VIEWS:
        width = statistics.mean(r["mean_restricted_clear_range"] for r in raw_checks if r["scene"] == scene)
        maximum = max(r["max_training_pixel_error"] for r in certificates if r["scene"] == scene)
        lines.append(f"| {scene} | {width:.6f} | {maximum:.6f} |")
    lines.extend(["", "## Reproduction", "", "Use the existing conda water_splatting environment and physical GPUs 6-9.",
        "Pass a new --output-root to any runner; existing experiment directories are intentionally rejected.",
        "The standalone experimental CUDA extensions do not replace water_splatting/csrc.so.", "",
        "Runtime and memory scopes vary between screens and are not an FPS benchmark. Candidates 4 and 6 did not record per-arm training peak memory.",
        "No new dependencies, historical output replacement, commits or pushes were performed.", ""])
    (output/"report.md").write_text("\n".join(lines))
    print(json.dumps(dict(passed=True, report=str(output/"report.md"), registered_hashes=len(hashes), sheets=sum(map(len, images.values())))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT/"outputs/transport_candidates_20260911")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check_environment()
    run(args.base, args.output or args.base/"review")
