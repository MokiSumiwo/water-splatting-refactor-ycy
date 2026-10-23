#!/usr/bin/env python3
"""Frozen native-view audit of clear visibility under a zero-medium replay."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.experiments.run_crcc_bnd_on import DEFAULT_OUT as CRCC_ROOT, configure_worker
from scripts.experiments.run_taba_bnd_on import check_environment, gpu_processes, read_json, sha256, source_assets, write_json
from water_splatting.rendering.underwater_rasterizer import UnderwaterRasterizer

OUT = ROOT / "outputs/dvs_frozen_visibility_20260911"
SCENES = ("JapaneseGradens-RedSea", "Panama", "Curasao", "IUI3-RedSea")
VARIANTS = ("E0", "E2")
STEP = 14999
GEOMETRY = ("xys", "depths", "radii", "conics", "num_tiles_hit", "colors", "opacities")
CLEAR_FIELDS = ("j_raw", "rgb_clear", "accumulation", "depth", "depth_second_moment",
                "depth_variance", "first_depth", "last_depth", "final_transmittance")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def origin(scene, variant):
    if variant == "E2" and scene in ("Curasao", "IUI3-RedSea"):
        return CRCC_ROOT / "crcc_only_four_scene_extension"
    return CRCC_ROOT


def rgb8(tensor):
    return (tensor.detach().float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def save_rgb(path, tensor):
    if tensor.shape[-1] == 1:
        tensor = tensor.expand(-1, -1, 3)
    Image.fromarray(rgb8(tensor)).save(path)


def fraction(mask):
    return float(mask.float().mean())


def timed_render(call, kwargs, repeats=5):
    warm = call(**kwargs)
    del warm
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    walls, events = [], []
    result = None
    for _ in range(repeats):
        if result is not None:
            del result
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall = time.perf_counter()
        start.record()
        result = call(**kwargs)
        end.record()
        torch.cuda.synchronize()
        walls.append(1000 * (time.perf_counter() - wall))
        events.append(start.elapsed_time(end))
    return result, dict(repeats=repeats, wall_ms_median=statistics.median(walls),
                        cuda_ms_median=statistics.median(events), wall_ms_samples=walls,
                        cuda_ms_samples=events, allocated_before_bytes=allocated,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        incremental_peak_bytes=torch.cuda.max_memory_allocated() - allocated,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())


@torch.no_grad()
def synthetic_checks():
    renderer = UnderwaterRasterizer()
    device = "cuda:0"
    zeros = torch.zeros(16, 16, 3, device=device)
    args = dict(xys=torch.tensor([[8., 8.], [8., 8.]], device=device),
                xys_grad_abs=torch.zeros(2, 2, device=device),
                depths=torch.tensor([2., 3.], device=device),
                radii=torch.tensor([3, 3], device=device, dtype=torch.int32),
                conics=torch.tensor([[1., 0., 1.], [1., 0., 1.]], device=device),
                num_tiles_hit=torch.ones(2, device=device, dtype=torch.int32),
                colors=torch.tensor([[.4, .2, .8], [.1, .7, .3]], device=device),
                opacities=torch.tensor([[.1], [.1]], device=device),
                medium_rgb=zeros, medium_bs=zeros, medium_attn=zeros,
                height=16, width=16, background=torch.zeros(3, device=device),
                step=STEP, force_white_background=False)
    clear = renderer.rasterize(**args)
    attenuated = renderer.rasterize(**dict(args, medium_attn=torch.full_like(zeros, 100.)))
    checks = {"positive_high_attenuation_clear_exact": torch.equal(clear.j_raw, attenuated.j_raw),
              "positive_high_attenuation_coverage_exact": torch.equal(clear.accumulation, attenuated.accumulation),
              "positive_high_attenuation_object_zero": float(attenuated.rgb_object.max()) == 0,
              "nonzero_clear_despite_attenuation": float(attenuated.j_raw[8, 8].max()) > .05}
    low = renderer.rasterize(**dict(args, opacities=torch.full_like(args["opacities"], .001)))
    checks["alpha_below_1_over_255_excluded"] = float(low.j_raw.max()) == float(low.accumulation.max()) == 0
    stop = renderer.rasterize(**dict(args, opacities=torch.full_like(args["opacities"], .999)))
    expected = args["colors"][0] * .999
    checks["termination_excludes_threshold_crossing_gaussian"] = bool(torch.allclose(stop.j_raw[8, 8], expected, atol=1e-7, rtol=0))
    checks["termination_accumulation_first_only"] = abs(float(stop.accumulation[8, 8]) - .999) < 1e-7
    assert all(checks.values()), checks
    return dict(passed=True, checks=checks, high_attenuation_clear_center=attenuated.j_raw[8, 8].tolist())


def image_statistics(base, zero):
    original, independent = base.j_raw, zero.j_raw
    alpha0, alpha1 = base.accumulation[..., 0], zero.accumulation[..., 0]
    error = (independent - original).abs()
    black0 = original.max(-1).values <= 3 / 255
    black1 = independent.max(-1).values <= 3 / 255
    cyan = lambda x: ((x[..., 1] + x[..., 2]) * .5 - x[..., 0] > .1) & (x[..., 1] > .1) & (x[..., 2] > .1)
    recovered = black0 & ~black1
    return dict(clear_max_abs_difference=float(error.max()), clear_mean_abs_difference=float(error.mean()),
                changed_pixel_fraction=fraction(error.max(-1).values > 0),
                changed_gt_1_over_255_fraction=fraction(error.max(-1).values > 1 / 255),
                original_clear_min=float(original.min()), original_clear_max=float(original.max()),
                zero_clear_min=float(independent.min()), zero_clear_max=float(independent.max()),
                zero_clear_over_one_fraction=fraction(independent > 1),
                original_near_black_fraction=fraction(black0), zero_near_black_fraction=fraction(black1),
                recovered_black_fraction=fraction(recovered), newly_black_fraction=fraction(~black0 & black1),
                original_cyan_like_fraction=fraction(cyan(original)), zero_cyan_like_fraction=fraction(cyan(independent)),
                original_dark_fraction=fraction(original.mean(-1) < .03), zero_dark_fraction=fraction(independent.mean(-1) < .03),
                original_accumulation_mean=float(alpha0.mean()), zero_accumulation_mean=float(alpha1.mean()),
                accumulation_max_abs_difference=float((alpha1 - alpha0).abs().max()),
                original_zero_coverage_fraction=fraction(alpha0 == 0), zero_zero_coverage_fraction=fraction(alpha1 == 0),
                original_low_coverage_fraction=fraction(alpha0 < .1), zero_low_coverage_fraction=fraction(alpha1 < .1),
                new_coverage_fraction=fraction((alpha0 == 0) & (alpha1 > 0)),
                near_black_with_zero_coverage_fraction=fraction(black0 & (alpha0 == 0)),
                near_black_with_low_nonzero_coverage_fraction=fraction(black0 & (alpha0 > 0) & (alpha0 < .1)),
                near_black_with_higher_coverage_fraction=fraction(black0 & (alpha0 >= .1)))


def sheets(directory, rows):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    result = []
    for label, columns in (
        ("clear", ("underwater_original.png", "clear_original.png", "clear_zero_medium.png", "difference_x20.png")),
        ("coverage", ("accumulation_original.png", "accumulation_zero_medium.png", "coverage_difference_x20.png", "recovered_black_mask.png")),
    ):
        canvas = Image.new("RGB", (4 * 450, len(rows) * 332), "#171717")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(rows):
            for j, name in enumerate(columns):
                with Image.open(directory / row["view_id"] / name) as source:
                    tile = source.convert("RGB")
                    tile.thumbnail((450, 300), Image.Resampling.LANCZOS)
                draw.text((j * 450 + 5, i * 332 + 5), row["view_id"] + " / " + name.replace(".png", ""), font=font, fill="white")
                canvas.paste(tile, (j * 450 + (450 - tile.width) // 2, i * 332 + 32))
        path = directory / (label + "_contact_sheet.jpg")
        canvas.save(path, quality=95, subsampling=0)
        result.append(str(path))
    return result


@torch.no_grad()
def run_variant(scene, variant, gpu, output):
    directory = output / scene / variant
    directory.mkdir(parents=True, exist_ok=False)
    source = origin(scene, variant)
    run = source / scene / variant
    checkpoint = run / "checkpoints/step-000014999.ckpt"
    checkpoint_hash = sha256(checkpoint)
    recorded = read_json(run / "crcc_result.json")
    assert next(row["sha256"] for row in recorded["checkpoint_rows"] if row["absolute_step"] == STEP) == checkpoint_hash
    worker = configure_worker(variant)
    source_root, config, _ = source_assets(scene)
    worker.ACTIVE_SOURCE_ROOT, worker.ACTIVE_SOURCE_CONFIG = source_root, config
    runtime = worker._runtime(scene, gpu)
    worker._seed_all(42)
    branch = worker._new_branch(scene)
    try:
        payload = worker._load_snapshot(branch, scene, checkpoint)
        model = branch.pipeline.model
        assert not model.training and not torch.is_grad_enabled()
        state_before = worker._model_state_hash(model)
        assert state_before == worker._hash_object(payload["model"])
        del payload
        optimizer_before = worker._optimizer_hash(branch.optimizers)
        records = worker._eval_records(branch)
        with (run / "per_view_metrics.csv").open() as handle:
            old_metrics = {row["view_id"]: row for row in csv.DictReader(handle) if int(row["absolute_step"]) == STEP}
        assert {r[1] for r in records} == set(old_metrics)
        rows, controls, hashes = [], [], []
        native_files = []
        rasterize = model.underwater_rasterizer.rasterize
        import water_splatting.rasterize as raster_module
        binning = raster_module.bin_and_sort_gaussians
        for index, view, camera, batch in records:
            target = directory / view
            target.mkdir()
            capture = {}
            def intercept(**kwargs):
                result = rasterize(**kwargs)
                capture.update(kwargs=kwargs, result=result)
                return result
            with patch.object(model.underwater_rasterizer, "rasterize", side_effect=intercept):
                outputs = model.get_outputs_for_camera(camera.to(model.device))
            kwargs, original = capture["kwargs"], capture["result"]
            assert kwargs["colors"].shape[-1] == 3
            assert int(kwargs["num_tiles_hit"].sum()) > 0
            assert all(bool(torch.isfinite(value).all()) for value in kwargs.values() if isinstance(value, torch.Tensor))
            assert float(kwargs["medium_attn"].min()) >= 0
            native = source / "renders" / variant / scene / f"step_{STEP}" / view / "A2"
            gt = worker._gt(model, worker._batch_to_device(batch.copy(), model.device), outputs["background"])
            originals = {"underwater.png": outputs["pred_image"], "clear_raw_display_clamp01.png": original.j_raw,
                         "clear.png": original.rgb_clear, "gt.png": gt}
            paired = {}
            for name, value in originals.items():
                path = native / name
                with Image.open(path) as source_image:
                    paired[name] = np.array_equal(np.asarray(source_image), rgb8(value))
                native_files.append(dict(path=str(path), sha256=sha256(path)))
            assert all(paired.values()), (scene, variant, view, paired)
            (target / "underwater_original.png").symlink_to(native / "underwater.png")
            (target / "gt.png").symlink_to(native / "gt.png")
            geometry_hash = worker._hash_object({k: kwargs[k] for k in GEOMETRY})
            underwater_hash = worker._hash_tensor(outputs["pred_image"])
            sort_hashes = []
            def record_sort(*args, **kw):
                result = binning(*args, **kw)
                sort_hashes.append(worker._hash_object({"ids": result[3], "bins": result[4], "keys": result[2]}))
                return result
            zeros = torch.zeros_like(kwargs["medium_attn"])
            zero_args = dict(kwargs, medium_rgb=zeros, medium_bs=zeros, medium_attn=zeros)
            with patch.object(raster_module, "bin_and_sort_gaussians", side_effect=record_sort):
                replay = rasterize(**kwargs)
                independent = rasterize(**zero_args)
            assert len(set(sort_hashes)) == 1
            baseline_exact = {field.name: torch.equal(getattr(original, field.name), getattr(replay, field.name))
                              for field in dataclasses.fields(original)}
            assert all(baseline_exact.values())
            clear_equal = {key: torch.equal(getattr(original, key), getattr(independent, key)) for key in CLEAR_FIELDS}
            cases = {
                "zero_attenuation_only": dict(kwargs, medium_attn=zeros),
                "zero_rgb_bs_only": dict(kwargs, medium_rgb=zeros, medium_bs=zeros),
                "zero_medium_black_background": dict(zero_args, background=zeros.new_zeros(3), force_white_background=False),
                "zero_medium_white_background": dict(zero_args, background=zeros.new_ones(3), force_white_background=False),
            }
            case_checks = {}
            for name, case in cases.items():
                control = rasterize(**case)
                case_checks[name] = {key: torch.equal(getattr(independent, key), getattr(control, key)) for key in CLEAR_FIELDS}
                del control
            row = dict(scene=scene, variant=variant, step=STEP, view_id=view, camera_index=index,
                       width=kwargs["width"], height=kwargs["height"],
                       medium_attn_min=float(kwargs["medium_attn"].min()), medium_attn_max=float(kwargs["medium_attn"].max()),
                       effective_cutoff_attenuation_max=float(torch.minimum(kwargs["medium_attn"].min(-1).values, zeros[..., 0]).max()),
                       intrinsic_rgb_min=float(kwargs["colors"].min()), intrinsic_rgb_max=float(kwargs["colors"].max()),
                       **image_statistics(original, independent))
            raw_error = (independent.j_raw - original.j_raw).abs()
            alpha_error = (independent.accumulation - original.accumulation).abs()
            recovered = ((original.j_raw.max(-1).values <= 3 / 255) & (independent.j_raw.max(-1).values > 3 / 255))[..., None]
            for name, image in (
                ("clear_original.png", original.j_raw), ("clear_zero_medium.png", independent.j_raw),
                ("clear_original_tonemapped.png", original.rgb_clear), ("clear_zero_medium_tonemapped.png", independent.rgb_clear),
                ("difference_x20.png", (20 * raw_error).clamp(0, 1)),
                ("accumulation_original.png", original.accumulation), ("accumulation_zero_medium.png", independent.accumulation),
                ("coverage_difference_x20.png", (20 * alpha_error).clamp(0, 1)), ("recovered_black_mask.png", recovered.float()),
            ):
                save_rgb(target / name, image)
            np.savez_compressed(target / "native_float32.npz",
                                clear_original=original.j_raw.cpu().numpy(), clear_zero_medium=independent.j_raw.cpu().numpy(),
                                accumulation_original=original.accumulation.cpu().numpy(),
                                accumulation_zero_medium=independent.accumulation.cpu().numpy())
            del replay
            measured, baseline_timing = timed_render(rasterize, kwargs)
            assert torch.equal(measured.j_raw, original.j_raw)
            del measured
            measured, zero_timing = timed_render(rasterize, zero_args)
            assert torch.equal(measured.j_raw, independent.j_raw)
            del measured
            repeated = model.get_outputs_for_camera(camera.to(model.device))
            assert worker._hash_tensor(repeated["pred_image"]) == underwater_hash
            assert worker._hash_object({k: kwargs[k] for k in GEOMETRY}) == geometry_hash
            assert worker._hash_tensor(outputs["pred_image"]) == underwater_hash
            row.update(original_raster_wall_ms=baseline_timing["wall_ms_median"],
                       zero_medium_raster_wall_ms=zero_timing["wall_ms_median"],
                       original_raster_cuda_ms=baseline_timing["cuda_ms_median"],
                       zero_medium_raster_cuda_ms=zero_timing["cuda_ms_median"],
                       original_incremental_peak_bytes=baseline_timing["incremental_peak_bytes"],
                       zero_medium_incremental_peak_bytes=zero_timing["incremental_peak_bytes"])
            controls.append(dict(view_id=view, native_original_png_pixels_exact=paired,
                baseline_raster_replay_exact=baseline_exact, zero_medium_clear_fields_exact=clear_equal,
                controls=case_checks, sorted_gaussian_ids_and_bins_exact=True,
                original_underwater_before_after_exact=True, geometry_and_colors_unchanged=True,
                timing=dict(original=baseline_timing, zero_medium=zero_timing)))
            hashes.append(dict(view_id=view, camera_sha256=worker._hash_object(camera.to_json(camera_idx=0)),
                               geometry_and_colors_sha256=geometry_hash, sorted_ids_bins_sha256=sort_hashes[0],
                               original_underwater_sha256=underwater_hash,
                               original_clear_sha256=worker._hash_tensor(original.j_raw),
                               zero_medium_clear_sha256=worker._hash_tensor(independent.j_raw)))
            rows.append(row)
            write_json(target / "diagnostic.json", dict(statistics=row, controls=controls[-1], hashes=hashes[-1]))
            print(json.dumps(row), flush=True)
            del outputs, original, independent, kwargs, zero_args, capture, repeated, zeros, raw_error, alpha_error, recovered, originals, gt, cases, case
        state_after = worker._model_state_hash(model)
        assert state_before == state_after
        assert worker._optimizer_hash(branch.optimizers) == optimizer_before
        assert sha256(checkpoint) == checkpoint_hash
        assert all(sha256(Path(row["path"])) == row["sha256"] for row in native_files)
        write_csv(directory / "per_view.csv", rows)
        result = dict(complete=True, scene=scene, variant=variant, runtime=runtime,
                      checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                      model_state_before=state_before, model_state_after=state_after,
                      model_and_ema_frozen=True, optimizer_steps=0, backward_calls=0,
                      checkpoint_and_original_pngs_unchanged=True, original_files=native_files,
                      view_count=len(rows), all_existing_test_views=True, per_view=rows, controls=controls,
                      contact_sheets=sheets(directory, rows), hashes=hashes)
        write_json(directory / "audit.json", result)
        return result
    finally:
        worker._release(branch)


def launch(output):
    output.mkdir(parents=True, exist_ok=False)
    assert not any(gpu_processes().values()), "Assigned GPUs must be free before launch"
    files = list(ROOT.glob("water_splatting/**/*.py"))
    files += list(ROOT.glob("water_splatting/cuda/csrc/*.cu"))
    files += list(ROOT.glob("water_splatting/cuda/csrc/*.cuh"))
    files += list(ROOT.glob("water_splatting/cuda/csrc/*.cpp"))
    files += [ROOT / "water_splatting/csrc.so", Path(__file__).resolve()]
    manifest = [dict(path=str(p.relative_to(ROOT)), sha256=sha256(p)) for p in sorted(set(files))]
    write_json(output / "registration.json", dict(
        experiment="DVS frozen visibility diagnosis", scenes=SCENES, variants=VARIANTS, step=STEP,
        expected_views=26, hypothesis="Nonnegative attenuation does not change current clear support because min(0,min(beta)) is zero.",
        controls=["original replay", "zero medium", "zero attenuation only", "zero rgb/bs only", "black/white background"],
        no_training=True, no_parameter_updates=True, no_threshold_changes=True,
        code_and_binary_manifest=manifest,
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)))
    processes=[]
    for scene, gpu in zip(SCENES, ("6", "7", "8", "9")):
        handle=(output / (scene + ".log")).open("x")
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4")
        command=[sys.executable, str(Path(__file__).resolve()), "--scene", scene, "--gpu", gpu, "--output-root", str(output)]
        processes.append((scene, subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
    status={}
    while len(status)<len(processes):
        for scene, proc, handle in processes:
            if scene not in status and proc.poll() is not None:
                status[scene]=proc.returncode
                handle.close()
                print(json.dumps(dict(scene=scene, exit_code=proc.returncode)), flush=True)
        write_json(output / "queue_status.json", dict(completed=status, running=[s for s,p,_ in processes if p.poll() is None]))
        if len(status)<len(processes):
            time.sleep(5)
    assert all(v==0 for v in status.values()), status
    assert all(sha256(ROOT / r["path"])==r["sha256"] for r in manifest)
    audits=[read_json(output / scene / variant / "audit.json") for scene in SCENES for variant in VARIANTS]
    rows=[row for audit in audits for row in audit["per_view"]]
    assert len(rows)==26
    write_csv(output / "per_view.csv", rows)
    means=[]
    metric_keys=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in ("step","camera_index","width","height")]
    for scene in SCENES:
        for variant in VARIANTS:
            chosen=[r for r in rows if r["scene"]==scene and r["variant"]==variant]
            means.append(dict(scene=scene,variant=variant,view_count=len(chosen),
                              **{key:statistics.mean(r[key] for r in chosen) for key in metric_keys}))
    write_csv(output / "scene_means.csv", means)
    final_gpus=gpu_processes()
    assert not any(final_gpus.values())
    exact=all(all(c["zero_medium_clear_fields_exact"].values()) for a in audits for c in a["controls"])
    result=dict(complete=True, view_count=len(rows), unique_test_cameras=13, checkpoint_count=8,
                hypothesis_supported=not exact, all_clear_and_coverage_exact=exact,
                all_model_state_unchanged=True, all_original_underwater_unchanged=True,
                code_and_binary_unchanged=True, gpu_processes_after=final_gpus,
                scene_means=means, timing_note="Five warmed repetitions per view; wrapper includes bin/sort and output construction. CUDA event interval also spans CPU launch gaps. Four concurrent workers, one per GPU.",
                decision="Reject attenuation-induced clear support loss for these checkpoints." if exact else "Inspect nonzero changes before drawing conclusions.")
    write_json(output / "summary.json", result)
    print(json.dumps(dict(complete=True,views=len(rows),clear_exact=exact,gpus=final_gpus)), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=("6","7","8","9"))
    parser.add_argument("--output-root", type=Path, default=OUT)
    args=parser.parse_args()
    check_environment()
    if args.scene:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == args.gpu
        assert not gpu_processes()[args.gpu]
        write_json(args.output_root / args.scene / "synthetic_checks.json", synthetic_checks())
        for variant in VARIANTS:
            run_variant(args.scene, variant, args.gpu, args.output_root)
    else:
        launch(args.output_root)
