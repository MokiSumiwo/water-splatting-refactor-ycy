#!/usr/bin/env python3
"""Verify the complete registered TABA experiment without loading a GPU model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_taba_bnd_on import DEFAULT_OUT, check_environment, sha256, write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def read_csv(path):
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def require_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), value
    elif isinstance(value, dict):
        for child in value.values():
            require_finite(child)
    elif isinstance(value, list):
        for child in value:
            require_finite(child)


def audit(root, *, variants=("E0", "E1", "E2"), exact_fallback_variants=()):
    summary = read_json(root / "summary.json")
    selection = read_json(root / "four_scene_selection.json")
    assert summary["four_scene_complete"]
    assert summary["selected_variant"] == selection["selected_variant"]
    selected = selection["selected_variant"]
    assert selected in variants and selected != "E0"
    expected = {(scene, arm) for scene in ("JapaneseGradens-RedSea", "Panama") for arm in variants}
    expected |= {(scene, arm) for scene in ("Curasao", "IUI3-RedSea") for arm in ("E0", selected)}
    assert {(row["scene"], row["variant"]) for row in summary["states"]} == expected
    assert len(summary["final_metrics"]) == len(expected)
    assert len(summary["pairings"]) == 4 and all(row["passed"] for row in summary["pairings"])
    require_finite(summary["final_metrics"])
    require_finite(summary["diagnostics"])
    checkpoint_rows, render_rows, runs = [], [], []
    gt_hashes = {}
    steps = (5000, 10000, 14999)
    for state in summary["states"]:
        assert state["complete"] and state["integrity_passed"], state
        scene, arm = state["scene"], state["variant"]
        directory = root / scene / arm
        training = read_json(directory / "training_summary.json")
        assert training["completed_updates"] == 11999
        assert training["checkpoint_count"] == 3
        assert {row["absolute_step"] for row in training["checkpoint_rows"]} == set(steps)
        for row in training["checkpoint_rows"]:
            path = directory / "checkpoints" / f"step-{row['absolute_step']:09d}.ckpt"
            assert path.stat().st_size == row["size_bytes"], path
            assert sha256(path) == row["sha256"], path
            checkpoint_rows.append({"scene": scene, "variant": arm, **row})
        metrics = read_csv(directory / "evaluation_metrics.csv")
        assert len(metrics) == 3 and {int(row["absolute_step"]) for row in metrics} == set(steps)
        views = read_csv(directory / "per_view_metrics.csv")
        expected_count = 4 if scene == "IUI3-RedSea" else 3
        assert len(views) == expected_count * len(steps)
        view_sets = [{row["view_id"] for row in views if int(row["absolute_step"]) == step} for step in steps]
        assert all(len(ids) == expected_count and ids == view_sets[0] for ids in view_sets)
        for row in views:
            step, view = int(row["absolute_step"]), row["view_id"]
            require_finite([float(row[key]) for key in ("PSNR", "SSIM", "LPIPS", "MSE")])
            assert float(row["P_J_gt_1"]) == float(row["P_intrinsic_rgb_gt_1"]) == 0
            if arm in exact_fallback_variants:
                assert 0 <= float(row["intrinsic_rgb_min"]) <= float(row["intrinsic_rgb_max"]) <= 1
                assert 0 < float(row["gcrc_corrected_intrinsic_min"]) <= float(row["gcrc_corrected_intrinsic_max"]) < 1
            elif arm != "E0":
                assert 0 < float(row["intrinsic_rgb_min"]) <= float(row["intrinsic_rgb_max"]) < 1
            original = ROOT / "undistorted_data" / ("undistorted_" + scene) / "images/ColorImage" / (view + ".png")
            with Image.open(original) as source:
                expected_size = source.size
                source_hash = hashlib.sha256(source.convert("RGB").tobytes()).hexdigest()
            for kind in ("underwater", "clear", "clear_raw_display_clamp01", "gt"):
                path = root / "renders" / arm / scene / f"step_{step:05d}" / view / "A2" / (kind + ".png")
                with Image.open(path) as image:
                    assert image.size == expected_size and image.mode == "RGB", path
                    image.load()
                    if kind == "gt":
                        digest = hashlib.sha256(image.tobytes()).hexdigest()
                        assert digest == source_hash, path
                        key = (scene, view)
                        assert gt_hashes.setdefault(key, digest) == digest, path
                render_rows.append(str(path))
        logged = [json.loads(line) for line in (directory / "taba_training.jsonl").read_text().splitlines()]
        require_finite(logged)
        assert logged[-1]["step"] == 14999
        assert all(row["P_J_gt_1"] == row["P_intrinsic_rgb_gt_1"] == 0 for row in logged)
        for step in steps:
            snapshot = read_csv(directory / f"taba_statistics_step_{step:05d}.csv")
            assert snapshot
            assert all(float(row["P_J_gt_1"]) == 0 for row in snapshot)
            assert all(float(row["P_intrinsic_rgb_gt_1"]) == 0 for row in snapshot if row["bucket"] == "all")
        runs.append({"scene": scene, "variant": arm, "passed": True, "view_count": expected_count})
    for sheet in summary["contact_sheets"]:
        with Image.open(sheet["path"]) as image:
            image.verify()
    assert len(checkpoint_rows) == len(expected) * 3
    assert len(render_rows) == sum(4 if scene == "IUI3-RedSea" else 3 for scene, arm in expected) * 12
    result = {"passed": True, "runs": runs, "checkpoint_count": len(checkpoint_rows),
              "render_count": len(render_rows), "contact_sheet_count": len(summary["contact_sheets"]),
              "paired_gt_count": len(gt_hashes), "checkpoint_hashes_verified": checkpoint_rows,
              "render_paths": render_rows, "all_pairings_passed": True,
              "native_resolution_verified": True, "gt_original_pixels_exact": True,
              "gt_identical_across_arms_and_steps": True}
    write_json(root / "artifact_integrity_audit.json", result)
    return {key: value for key, value in result.items() if key not in ("checkpoint_hashes_verified", "render_paths")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    check_environment()
    print(json.dumps(audit(args.output_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
