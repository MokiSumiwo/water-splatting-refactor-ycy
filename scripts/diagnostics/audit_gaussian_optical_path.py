#!/usr/bin/env python3
"""Frozen four-scene screening of Gaussian conditional optical-depth integration."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.cpp_extension import load

from scripts.diagnostics.audit_dvs_frozen_visibility import SCENES, origin, save_rgb, write_csv
from scripts.diagnostics.audit_multiview_degradation_responsibility import _geometry_bins
from scripts.experiments.run_crcc_bnd_on import configure_worker
from scripts.experiments.run_taba_bnd_on import check_environment, gpu_processes, read_json, sha256, source_assets, write_json
from water_splatting.rendering.optical_path import conditional_depth_parameters, positive_normal_transmission

OUT = ROOT / "outputs/transport_candidates_20260911/01_optical_path"
MODES = {"conditional_mean": 3, "conditional_integrated": 6, "center_integrated": 9}


def extension():
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    return load(name="optical_path_precision_precise", sources=[str(Path(__file__).parent / "csrc/optical_path_audit.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"], verbose=False)


@torch.no_grad()
def synthetic_check(ext):
    device = "cuda:0"
    ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    bins = torch.tensor([[0, 2]], dtype=torch.int32, device=device)
    xys = torch.tensor([[8., 8.], [9., 8.]], device=device)
    conics = torch.tensor([[1., 0., 1.], [1., 0., 1.]], device=device)
    colors = torch.tensor([[.2, .6, .8], [.5, .3, .7]], device=device)
    opacity = torch.tensor([.3, .6], device=device)
    beta = torch.tensor([.1, .5, 3.], device=device).expand(16, 16, 3).contiguous()
    depths = torch.tensor([2., 3.], device=device)
    conditional = torch.tensor([[.2, -.1, .25], [-.1, .3, 1.]], device=device)
    actual = ext.forward(ids, bins, xys, conics, colors, opacity, beta, depths, conditional, 16, 16)
    reference = torch.zeros_like(actual, dtype=torch.float64)
    yy, xx = torch.meshgrid(torch.arange(16, device=device), torch.arange(16, device=device), indexing="ij")
    transmission = torch.ones(16, 16, device=device, dtype=torch.float64)
    for k in range(2):
        dx, dy = xys[k, 0].double() - xx, xys[k, 1].double() - yy
        alpha = (opacity[k].double() * (-.5 * (dx.square() + dy.square())).exp()).clamp_max(.999)
        alpha = torch.where(alpha >= 1/255, alpha, 0.)
        weight = alpha * transmission
        mean = depths[k].double() - conditional[k, 0].double() * dx - conditional[k, 1].double() * dy
        b = beta.double()
        variance = conditional[k, 2].double().expand(16, 16, 1)
        contribution = weight[..., None] * colors[k].double()
        reference[..., :3] += contribution * (-b * depths[k].double()).exp()
        reference[..., 3:6] += contribution * (-b * mean[..., None].clamp_min(0)).exp()
        reference[..., 6:9] += contribution * positive_normal_transmission(mean[..., None], variance, b)
        reference[..., 9:12] += contribution * positive_normal_transmission(depths[k].double(), variance, b)
        reference[..., 12:15] += contribution
        reference[..., 15] += weight
        transmission *= 1-alpha
    error = float((actual.double()-reference).abs().max())
    assert error < 2e-6, error
    zero = ext.forward(ids, bins, xys, conics, colors, opacity, torch.zeros_like(beta), depths, conditional, 16, 16)
    assert all(torch.equal(zero[..., :3], zero[..., start:start+3]) for start in (3, 6, 9, 12))
    return dict(passed=True, reference_max_error=error, zero_medium_exact=True)


def contact_sheet(directory, rows):
    columns = ("underwater_original.png", "conditional_mean.png", "conditional_integrated.png", "center_integrated_difference_x20.png")
    canvas = Image.new("RGB", (1600, 290 * len(rows)), "#161616")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        for j, name in enumerate(columns):
            with Image.open(directory / row["view_id"] / name) as source:
                tile = source.convert("RGB")
                tile.thumbnail((400, 260), Image.Resampling.LANCZOS)
            draw.text((j*400+4, i*290+3), row["view_id"]+" "+name.removesuffix(".png") if hasattr(name, "removesuffix") else row["view_id"]+" "+name, fill="white")
            canvas.paste(tile, (j*400+(400-tile.width)//2, i*290+25))
    canvas.save(directory / "contact_sheet.jpg", quality=95)


@torch.no_grad()
def run_scene(scene, gpu, output):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpu
    ext = extension()
    write_json(output / scene / "synthetic_checks.json", synthetic_check(ext))
    for arm in ("E0", "E2"):
        target = output / scene / arm
        target.mkdir(parents=True, exist_ok=False)
        worker = configure_worker(arm)
        worker.ACTIVE_SOURCE_ROOT, worker.ACTIVE_SOURCE_CONFIG, _ = source_assets(scene)
        runtime = worker._runtime(scene, gpu)
        branch = worker._new_branch(scene)
        checkpoint = origin(scene, arm) / scene / arm / "checkpoints/step-000014999.ckpt"
        source_hash = sha256(checkpoint)
        try:
            worker._load_snapshot(branch, scene, checkpoint)
            model = branch.pipeline.model
            before = worker._model_state_hash(model)
            records = worker._eval_records(branch)
            rows = []
            for index, view, camera, batch in records:
                capture = {}
                original_project = model.underwater_rasterizer.project
                original_raster = model.underwater_rasterizer.rasterize
                def project(**kwargs):
                    result = original_project(**kwargs)
                    capture.update(project=kwargs, cov3d=result[-1])
                    return result
                def raster(**kwargs):
                    result = original_raster(**kwargs)
                    capture.update(raster=kwargs, native=result)
                    return result
                with patch.object(model.underwater_rasterizer, "project", side_effect=project), patch.object(model.underwater_rasterizer, "rasterize", side_effect=raster):
                    outputs = model.get_outputs_for_camera(camera.to(model.device))
                k, p, native = capture["raster"], capture["project"], capture["native"]
                state = conditional_depth_parameters(p["means"], capture["cov3d"], p["viewmat"], k["conics"], p["fx"], p["fy"], p["width"], p["height"]).float().contiguous()
                assert bool(torch.isfinite(state).all()) and float(k["medium_attn"].min()) >= 0
                ids, bins, _ = _geometry_bins(k)
                arguments = (ids, bins, k["xys"], k["conics"], k["colors"], k["opacities"].reshape(-1), k["medium_attn"], k["depths"], state, k["height"], k["width"])
                result = ext.forward(*arguments)
                torch.cuda.synchronize()
                assert bool(torch.isfinite(result).all())
                equivalence = dict(direct=float((result[..., :3]-native.rgb_object).abs().max()),
                    clear=float((result[..., 12:15]-native.j_raw).abs().max()),
                    accumulation=float((result[..., 15:16]-native.accumulation).abs().max()))
                assert max(equivalence.values()) < 2e-6, equivalence
                gt = worker._gt(model, worker._batch_to_device(batch.copy(), model.device), outputs["background"]).float().clamp(0, 1)
                original = outputs["pred_image"].float().clamp(0, 1)
                old_metrics = worker._taba_rgb_metrics(original, gt)
                directory = target / view
                directory.mkdir()
                native_path = origin(scene, arm) / "renders" / arm / scene / "step_14999" / view / "A2"
                for name, source in (("underwater_original.png", "underwater.png"), ("clear_original.png", "clear_raw_display_clamp01.png"), ("gt.png", "gt.png")):
                    (directory / name).symlink_to(native_path / source)
                with Image.open(native_path / "underwater.png") as image:
                    assert np.array_equal(np.asarray(image), (original.cpu().numpy()*255).round().astype(np.uint8))
                row = dict(scene=scene, arm=arm, view_id=view, camera_index=index, width=k["width"], height=k["height"],
                    base_metrics=old_metrics, baseline_equivalence=equivalence, controls={})
                arrays = {"direct_center": result[..., :3].cpu().numpy()}
                for mode, start in MODES.items():
                    delta = result[..., start:start+3] - result[..., :3]
                    changed = (outputs["pred_image"]+delta).clamp(0, 1)
                    error = delta.abs()
                    metrics = worker._taba_rgb_metrics(changed, gt)
                    row["controls"][mode] = dict(mean_abs_difference=float(error.mean()), max_abs_difference=float(error.max()),
                        changed_gt_1_over_255_fraction=float((error.max(-1).values > 1/255).float().mean()),
                        metrics=metrics, delta_psnr=metrics["PSNR"]-old_metrics["PSNR"])
                    arrays[mode+"_delta"] = delta.cpu().numpy()
                    save_rgb(directory / (mode+".png"), changed)
                    save_rgb(directory / (mode+"_difference_x20.png"), (20*error).clamp(0, 1))
                np.savez_compressed(directory / "direct_float32.npz", **arrays)
                samples = []
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    temporary = ext.forward(*arguments)
                    torch.cuda.synchronize()
                    samples.append(1000*(time.perf_counter()-started))
                    del temporary
                row["kernel_wall_ms_median"] = statistics.median(samples)
                row["kernel_incremental_peak_bytes"] = torch.cuda.max_memory_allocated()-allocated
                write_json(directory / "diagnostic.json", row)
                rows.append(row)
                print(json.dumps(dict(scene=scene, arm=arm, view=view, controls=row["controls"])), flush=True)
                del outputs, capture, k, p, native, state, result, arguments, original, gt, arrays, delta, changed, error
            assert before == worker._model_state_hash(model)
            assert source_hash == sha256(checkpoint)
            contact_sheet(target, rows)
            write_json(target / "audit.json", dict(complete=True, checkpoint=str(checkpoint), checkpoint_sha256=source_hash,
                runtime=runtime, source_unchanged=True, model_state_unchanged=True, training_steps=0, rows=rows))
        finally:
            worker._release(branch)


def launch(output):
    output.mkdir(parents=True, exist_ok=False)
    assert not any(gpu_processes().values())
    frozen = [Path(__file__), Path(__file__).parent / "csrc/optical_path_audit.cu",
              ROOT / "water_splatting/rendering/optical_path.py", ROOT / "water_splatting/csrc.so",
              ROOT / "water_splatting/water_splatting.py"]
    manifest = [dict(path=str(path), sha256=sha256(path)) for path in frozen]
    write_json(output / "registration.json", dict(candidate=1, frozen_manifest=manifest, scenes=SCENES, arms=["E0", "E2"],
        screen=dict(min_mean_abs=1e-4, min_changed_fraction=.01, min_scenes=2), exact_overlap_volume=False))
    extension()
    processes = []
    for scene, gpu in zip(SCENES, ("6", "7", "8", "9")):
        handle = (output / (scene+".log")).open("x")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4")
        proc = subprocess.Popen([sys.executable, str(Path(__file__)), "--scene", scene, "--gpu", gpu, "--output-root", str(output)],
                                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((scene, proc, handle))
    while any(proc.poll() is None for _, proc, _ in processes):
        write_json(output / "queue_status.json", {scene: proc.poll() for scene, proc, _ in processes})
        time.sleep(5)
    for scene, proc, handle in processes:
        handle.close()
    status = {scene: proc.returncode for scene, proc, _ in processes}
    write_json(output / "queue_status.json", status)
    assert all(code == 0 for code in status.values()), status
    rows = [row for scene in SCENES for arm in ("E0", "E2")
            for row in read_json(output / scene / arm / "audit.json")["rows"]]
    means = []
    for scene in SCENES:
        for arm in ("E0", "E2"):
            chosen = [row for row in rows if row["scene"] == scene and row["arm"] == arm]
            for mode in MODES:
                values = [row["controls"][mode] for row in chosen]
                means.append(dict(scene=scene, arm=arm, mode=mode, view_count=len(chosen),
                    **{key:statistics.mean(value[key] for value in values) for key in ("mean_abs_difference", "changed_gt_1_over_255_fraction", "delta_psnr")}))
    supported = [scene for scene in SCENES if any(row["scene"] == scene and row["mode"] == "center_integrated"
                and row["mean_abs_difference"] >= 1e-4 and row["changed_gt_1_over_255_fraction"] >= .01 for row in means)]
    assert all(sha256(Path(row["path"])) == row["sha256"] for row in manifest)
    write_csv(output / "scene_means.csv", means)
    write_json(output / "summary.json", dict(complete=True, comparisons=len(rows), means=means,
        thickness_effect_scenes=supported, thickness_screen_pass=len(supported)>=2,
        no_training=True, frozen_files_unchanged=True, original_assets_preserved=True,
        interpretation="Frozen direct-attenuation screen; does not establish training benefit or exact volume transport."))
    print(json.dumps(dict(complete=True, comparisons=len(rows), thickness_effect_scenes=supported)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=("6", "7", "8", "9"))
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    check_environment()
    if args.scene:
        run_scene(args.scene, args.gpu, args.output_root)
    else:
        launch(args.output_root)
