#!/usr/bin/env python3
"""Verify saved DVS comparisons without loading a model or rendering again."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/dvs_frozen_visibility_20260911/verified_v2"
SCENES = {"JapaneseGradens-RedSea": 3, "Panama": 3, "Curasao": 3, "IUI3-RedSea": 4}
ZERO_FIELDS = ("clear_max_abs_difference", "clear_mean_abs_difference", "changed_pixel_fraction",
               "accumulation_max_abs_difference", "new_coverage_fraction", "recovered_black_fraction",
               "newly_black_fraction", "effective_cutoff_attenuation_max")


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixels(path):
    with Image.open(path) as image:
        return np.array(image)


def rgb8(array):
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def all_true(mapping):
    return all(all_true(value) if isinstance(value, dict) else value is True
               for value in mapping.values())


def verify(output):
    assert os.environ.get("CONDA_DEFAULT_ENV") == "water_splatting"
    assert set(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) <= {"6", "7", "8", "9"}
    destination = output / "final_verification.json"
    assert not destination.exists(), destination
    registration = read_json(output / "registration.json")
    summary = read_json(output / "summary.json")
    assert summary["complete"] and summary["all_clear_and_coverage_exact"]
    assert summary["hypothesis_supported"] is False
    manifest = registration["code_and_binary_manifest"]
    assert all(sha256(ROOT / row["path"]) == row["sha256"] for row in manifest)
    queue = read_json(output / "queue_status.json")
    assert not queue["running"] and set(queue["completed"]) == set(SCENES)
    assert all(code == 0 for code in queue["completed"].values())
    with (output / "per_view.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == summary["view_count"] == 26
    assert all(float(row[key]) == 0 for row in csv_rows for key in ZERO_FIELDS)
    original_files, checked_views, checkpoints = {}, [], []
    tables = []
    for scene, count in SCENES.items():
        synthetic = read_json(output / scene / "synthetic_checks.json")
        assert synthetic["passed"] and all_true(synthetic["checks"])
        for variant in ("E0", "E2"):
            directory = output / scene / variant
            audit = read_json(directory / "audit.json")
            assert audit["complete"] and audit["all_existing_test_views"]
            assert audit["model_and_ema_frozen"] and audit["checkpoint_and_original_pngs_unchanged"]
            assert audit["optimizer_steps"] == audit["backward_calls"] == 0
            assert audit["model_state_before"] == audit["model_state_after"]
            assert audit["runtime"]["physical_gpu"] in ("6", "7", "8", "9")
            assert "/envs/water_splatting/" in audit["runtime"]["python"]
            checkpoint = Path(audit["checkpoint"])
            assert sha256(checkpoint) == audit["checkpoint_sha256"]
            checkpoints.append(str(checkpoint))
            assert audit["view_count"] == len(audit["per_view"]) == count
            source_views = set()
            with (checkpoint.parent.parent / "per_view_metrics.csv").open() as handle:
                for row in csv.DictReader(handle):
                    if int(row["absolute_step"]) == 14999:
                        source_views.add(row["view_id"])
            assert source_views == {row["view_id"] for row in audit["per_view"]}
            for row in audit["original_files"]:
                original_files[row["path"]] = row["sha256"]
            for row in audit["per_view"]:
                target = directory / row["view_id"]
                diagnostic = read_json(target / "diagnostic.json")
                assert diagnostic["statistics"] == row
                controls = diagnostic["controls"]
                for key in ("baseline_raster_replay_exact", "zero_medium_clear_fields_exact",
                            "controls", "native_original_png_pixels_exact"):
                    assert all_true(controls[key]), (target, key)
                for key in ("geometry_and_colors_unchanged", "original_underwater_before_after_exact",
                            "sorted_gaussian_ids_and_bins_exact"):
                    assert controls[key] is True
                hashes = diagnostic["hashes"]
                assert hashes["original_clear_sha256"] == hashes["zero_medium_clear_sha256"]
                assert all(row[key] == 0 for key in ZERO_FIELDS)
                height, width = row["height"], row["width"]
                with np.load(target / "native_float32.npz", allow_pickle=False) as data:
                    assert set(data.files) == {"clear_original", "clear_zero_medium",
                                               "accumulation_original", "accumulation_zero_medium"}
                    for key in data.files:
                        array = data[key]
                        assert array.dtype == np.float32 and np.isfinite(array).all()
                        assert array.shape == (height, width, 3 if key.startswith("clear") else 1)
                        assert np.array_equal(pixels(target / (key + ".png")), rgb8(array))
                    for field in ("clear", "accumulation"):
                        assert np.array_equal(data[field + "_original"].view(np.uint32),
                                              data[field + "_zero_medium"].view(np.uint32))
                    clear = data["clear_original"]
                    alpha = data["accumulation_original"][..., 0]
                    assert abs(float((clear.max(-1) <= 3 / 255).mean()) - row["original_near_black_fraction"]) < 1e-7
                    assert abs(float((alpha == 0).mean()) - row["original_zero_coverage_fraction"]) < 1e-7
                    assert float(clear.min()) == row["original_clear_min"]
                    assert float(clear.max()) == row["original_clear_max"]
                for name in ("difference_x20.png", "coverage_difference_x20.png", "recovered_black_mask.png"):
                    assert not pixels(target / name).any()
                assert np.array_equal(pixels(target / "clear_original_tonemapped.png"),
                                      pixels(target / "clear_zero_medium_tonemapped.png"))
                for name in ("underwater_original.png", "gt.png"):
                    assert (target / name).is_symlink() and (target / name).resolve().is_file()
                assets = [dict(path=str(path.relative_to(output)), sha256=sha256(path))
                          for path in sorted(target.iterdir()) if path.is_file()]
                checked_views.append(dict(scene=scene, variant=variant, view=row["view_id"],
                                          shape=[height, width], assets=assets))
            rows = audit["per_view"]
            means = next(r for r in summary["scene_means"] if r["scene"] == scene and r["variant"] == variant)
            tables.append(dict(scene=scene, variant=variant, view_count=count,
                clear_range=[min(r["original_clear_min"] for r in rows), max(r["original_clear_max"] for r in rows)],
                color_range=[min(r["intrinsic_rgb_min"] for r in rows), max(r["intrinsic_rgb_max"] for r in rows)],
                attenuation_range=[min(r["medium_attn_min"] for r in rows), max(r["medium_attn_max"] for r in rows)],
                black_percent=100 * means["original_near_black_fraction"],
                zero_coverage_percent=100 * means["original_zero_coverage_fraction"],
                black_higher_coverage_percent=100 * means["near_black_with_higher_coverage_fraction"],
                original_wall_ms=means["original_raster_wall_ms"], zero_wall_ms=means["zero_medium_raster_wall_ms"],
                original_incremental_mib=means["original_incremental_peak_bytes"] / 2**20,
                zero_incremental_mib=means["zero_medium_incremental_peak_bytes"] / 2**20))
            print(json.dumps(dict(scene=scene, variant=variant, verified_views=count)), flush=True)
    assert all(sha256(Path(path)) == digest for path, digest in original_files.items())
    assert len(original_files) == 104 and len(checkpoints) == 8 and len(checked_views) == 26
    counts = dict(native_npz=len(list(output.glob("*/*/*/native_float32.npz"))),
                  new_png=len([p for p in output.glob("*/*/*/*.png") if not p.is_symlink()]),
                  historical_png_symlinks=len([p for p in output.glob("*/*/*/*.png") if p.is_symlink()]),
                  contact_sheets=len(list(output.glob("*/*/*_contact_sheet.jpg"))))
    assert counts == dict(native_npz=26, new_png=234, historical_png_symlinks=52, contact_sheets=16)
    process_query = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
    gpu_query = subprocess.check_output([
        "nvidia-smi", "--id=6,7,8,9", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    uuids = {line.split(",")[1].strip() for line in gpu_query.splitlines()}
    assigned_processes = [line for line in process_query.splitlines() if line.split(",")[0].strip() in uuids]
    assert not assigned_processes, assigned_processes
    result = dict(complete=True, verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  no_new_render_or_training=True, frozen_manifest_files=len(manifest),
                  core_and_binary_hashes_unchanged=True, checkpoint_count=len(checkpoints),
                  historical_pngs_unchanged=len(original_files), all_test_views_verified=True,
                  view_count=len(checked_views), native_float32_bit_exact=True, finite_arrays=True,
                  all_difference_and_recovery_images_zero=True, asset_counts=counts,
                  gpu_6_through_9_processes=assigned_processes, tables=tables, views=checked_views)
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "views"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    verify(parser.parse_args().output_root)
