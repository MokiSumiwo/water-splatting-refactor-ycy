#!/usr/bin/env python3
"""Check complete transport artifacts and report resource and image statistics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from scripts.diagnostics.audit_dvs_frozen_visibility import SCENES, write_csv
from scripts.experiments.run_taba_bnd_on import check_environment, read_json, sha256, write_json


def verify(experiment):
    rows, training, images, scenes = [], [], [], []
    registration = read_json(experiment / "registration.json")
    expected_seeds = registration["seeds"]
    assert expected_seeds == [42, 123, 3407] and registration["steps"] == 5000
    for name, digest in registration["code_sha256"].items():
        assert sha256(ROOT / name) == digest, "Registered code changed: " + name
    for scene in SCENES:
        directory = experiment / scene
        result = read_json(directory / "result.json")
        assert result["complete"] and result["formal"] and result["source_unchanged"]
        assert result["steps"] == 5000 and result["seeds"] == expected_seeds
        assert sha256(Path(result["checkpoint"])) == result["checkpoint_sha256"]
        pixels = read_json(directory / "training_pixels.json")
        for view in pixels["rows"]:
            fit, diagnostic = set(view["fit_pixels"]), set(view["diagnostic_pixels"])
            assert len(fit) == len(diagnostic) == 4096 and not fit & diagnostic
            assert max(view[k] for k in ("native_field_error", "native_fast_error",
                "native_prediction_error", "native_clear_error")) <= 2e-6
        checks = read_json(directory / "render_preservation.json")
        views = [r["view_id"] for r in result["rows"] if r["arm"] == "A0"]
        assert len(views) == len(set(views)) == (4 if scene == "IUI3-RedSea" else 3)
        assert not set(views) & {r["view"] for r in pixels["rows"]}
        assert len(result["rows"]) == 10 * len(views)
        assert len(checks) == len(views) and all(r["original_png_identical"] and
            r["all_native_comparisons_bit_identical"] for r in checks)
        assert len(result["training"]) == 9
        for seed in expected_seeds:
            group = [r for r in result["training"] if r["seed"] == seed]
            assert len(group) == 3 and len({r["sampling_sha256"] for r in group}) == 1
            assert len({r["initial_medium_hash"] for r in group}) == 1
            for entry in group:
                assert entry["complete"] and entry["steps"] == 5000
                assert (directory / ("seed_%d" % seed) / entry["arm"] / "checkpoint.pt").is_file()
                training.append(dict(scene=scene, seed=seed, arm=entry["arm"],
                    wall_seconds=entry["wall_seconds"], peak_allocated_bytes=entry["peak_allocated_bytes"],
                    incremental_peak_bytes=entry["incremental_peak_bytes"], scalar=entry["scalar"],
                    initial_diagnostic_psnr=entry["initial_diagnostic"]["psnr"],
                    final_diagnostic_psnr=entry["final_diagnostic"]["psnr"],
                    final_fit_psnr=entry["final_fit"]["psnr"]))
        for view in views:
            render = directory / "renders" / view
            with np.load(render / "original_float.npz") as data:
                original, clear, accum = data["underwater"], data["clear"], data["accumulation"]
                assert all(np.isfinite(data[k]).all() for k in data.files)
            expected_shape = original.shape
            assert len(expected_shape) == 3 and expected_shape[-1] == 3
            for seed in expected_seeds:
                for arm in ("A1", "A2", "A3"):
                    prefix = "seed_%d_%s" % (seed, arm)
                    with np.load(render / (prefix+"_float.npz")) as data:
                        pred = data["underwater"]
                    assert pred.shape == expected_shape and np.isfinite(pred).all()
                    assert sha256(render / (prefix+"_clear.png")) == sha256(render / "A0_clear.png")
                    delta = np.abs(pred-original)
                    row = next(r for r in result["rows"] if r["view_id"] == view and
                               r["seed"] == seed and r["arm"] == arm)
                    assert row["clear_max_error"] == row["accumulation_max_error"] == 0.
                    assert row["fast_native_max_error"] <= 2e-6
                    rows.append(dict(scene=scene, view_id=view, seed=seed, arm=arm,
                        underwater_mae_vs_A0=float(delta.mean()), underwater_max_change=float(delta.max()),
                        underwater_pixel_fraction_changed_gt_1_255=float((delta.max(-1)>1/255).mean()),
                        underwater_channel_fraction_gt_1=float((pred>1).mean()),
                        underwater_min=float(pred.min()), underwater_max=float(pred.max()),
                        clear_black_fraction=float((clear.max(-1)<=3/255).mean()),
                        accumulation_mean=float(accum.mean()),
                        clear_change=0., coverage_change=0., **{k: row[k] for k in (
                            "native_render_seconds", "peak_allocated_bytes", "incremental_peak_bytes")}))
                    for suffix in ("_underwater.png", "_difference_x10.png", "_gt_error_x5.png"):
                        with Image.open(render / (prefix+suffix)) as image:
                            assert image.size == (expected_shape[1], expected_shape[0])
            for name in ("underwater_contact_sheet.jpg", "clear_contact_sheet.jpg", "differences_contact_sheet.jpg",
                         "underwater_with_bnd_off.jpg", "clear_with_bnd_off.jpg"):
                path = directory / name
                if not any(r["path"] == str(path.relative_to(experiment)) for r in images):
                    with Image.open(path) as image:
                        array = np.asarray(image)
                        assert array.std() > 5 and array.max() > 100
                        images.append(dict(path=str(path.relative_to(experiment)), width=image.width,
                            height=image.height, sha256=sha256(path), channel_std=float(array.std())))
        scenes.append(dict(scene=scene, training_views=len(pixels["rows"]), test_views=len(views),
            native_cache_max_error=pixels["native_max_error"], clear_bit_identical=True,
            original_archived_png_identical=True))
    assert len(rows) == 117 and len(training) == 36 and len(images) == 20
    resources = []
    for arm in ("A1", "A2", "A3"):
        chosen = [r for r in rows if r["arm"] == arm]
        fitted = [r for r in training if r["arm"] == arm]
        resources.append(dict(arm=arm, fits=len(fitted), training_seconds_sum=sum(r["wall_seconds"] for r in fitted),
            training_seconds_median=statistics.median(r["wall_seconds"] for r in fitted),
            evaluation_ms_median=1000*statistics.median(r["native_render_seconds"] for r in chosen),
            evaluation_incremental_peak_mib_median=statistics.median(r["incremental_peak_bytes"] for r in chosen)/2**20,
            total_peak_mib=max(r["peak_allocated_bytes"] for r in chosen+fitted)/2**20))
    write_csv(experiment / "image_diagnostics.csv", rows)
    write_csv(experiment / "training_diagnostics.csv", training)
    write_csv(experiment / "resource_summary.csv", resources)
    write_json(experiment / "artifact_verification.json", dict(complete=True, formal_fits=36,
        optimizer_updates=180000, test_views=13, candidate_view_evaluations=117,
        scenes=scenes, contact_sheets=images, resources=resources, registered_code_unchanged=True,
        comparison_timing_scope="medium evaluation plus rasterization plus equality checks; projection excluded",
        continuation_seeds_not_independent_full_training=True))
    print(json.dumps(dict(event="verification", scenes=scenes, resources=resources)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    check_environment()
    verify(args.experiment.resolve())
