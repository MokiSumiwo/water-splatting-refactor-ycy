#!/usr/bin/env python3
"""Search a restricted clear family subject to full training-view constraints."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.diagnostics.audit_dvs_frozen_visibility import SCENES, origin, save_rgb, write_csv
from scripts.diagnostics.audit_multiview_degradation_responsibility import _geometry_bins
from scripts.experiments.run_crcc_bnd_on import configure_worker
from scripts.experiments.run_radiometric_domain_screen import cache, prepare
from scripts.experiments.run_taba_bnd_on import check_environment, gpu_processes, read_json, sha256, source_assets, write_json
from scripts.experiments.run_world_medium_screen import capture_view, extension
from water_splatting.rendering.radiometric_transport import FrozenRadiometricTransport

OUT = ROOT / "outputs/transport_candidates_20260911/09_clear_feasible_family"
EPSILON = 1 / 255
SCALES = (1., .5, .25, .125, 0.)


def train_direction(direction, data, directory):
    model = FrozenRadiometricTransport(appearance_shift=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    generator = torch.Generator().manual_seed(42)
    count = data["base"].shape[0] // 4096
    assert count * 4096 == data["base"].shape[0]
    offsets = torch.arange(count)[:, None] * 4096
    sign = 1 if direction == "higher" else -1
    rows = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, 1501):
        indices = (offsets + torch.randint(4096, (count, 32), generator=generator)).flatten().cuda()
        batch = {key: value[indices] for key, value in data.items()}
        optimizer.zero_grad(set_to_none=True)
        pred, clear = model(batch)
        mse = (pred - batch["base"]).square().reshape(count, -1).mean(-1)
        displacement = (clear - batch["clear_base"]).mean()
        loss = 4 * torch.relu(mse.max() / EPSILON**2 - .7).square() - sign * displacement / .05
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.raw.grad).all()
        optimizer.step()
        with torch.no_grad():
            model.raw.clamp_(-3, 3)
        if step == 1 or step % 250 == 0:
            row = dict(step=step, loss=float(loss.detach()), sampled_max_view_rmse=float(mse.max().sqrt().detach()),
                       clear_mean_displacement=float(displacement.detach()))
            rows.append(row)
            print(json.dumps(dict(event="search", direction=direction, **row)), flush=True)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), directory / (direction + "_search.pt"))
    write_json(directory / (direction + "_search.json"), dict(rows=rows, steps=1500, raw=model.raw.detach().tolist(),
        wall_seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    return model.eval()


def scaled_models(models):
    variants = {}
    for direction, source in models.items():
        for index, scale in enumerate(SCALES):
            model = FrozenRadiometricTransport(appearance_shift=True).cuda().eval()
            with torch.no_grad():
                model.raw.copy_(source.raw * scale)
            variants[(direction, index)] = model
    return variants


@torch.no_grad()
def certify_training(worker, branch, ext, models, directory):
    variants = scaled_models(models)
    rows = []
    for _, view, camera, batch in worker._train_records(branch):
        k, native, outputs, _ = capture_view(worker, branch.pipeline.model, camera, batch)
        assert k["colors"].min() >= 0 and k["colors"].max() <= 1
        bins = _geometry_bins(k)
        total = k["width"] * k["height"]
        stats = {key: torch.zeros(3, dtype=torch.float64, device="cuda") for key in variants}
        started = time.perf_counter()
        for start in range(0, total, 8192):
            pixels = torch.arange(start, min(start+8192, total), device="cuda", dtype=torch.int64)
            data = prepare(ext, k, bins, pixels, outputs)
            base_clear = native.j_raw.reshape(-1, 3)[pixels]
            for key, model in variants.items():
                pred, clear = model(data)
                error = pred - data["base"]
                stats[key][0] += error.double().square().sum()
                stats[key][1] = torch.maximum(stats[key][1], error.abs().max().double())
                stats[key][2] += (clear-base_clear).double().sum()
        for (direction, index), value in stats.items():
            assert torch.isfinite(value).all()
            if SCALES[index] == 0:
                assert float(value[1]) < 2e-6, (view, direction, float(value[1]))
            rows.append(dict(view=view, direction=direction, scale=SCALES[index],
                underwater_rmse=float((value[0]/(3*total)).sqrt()), underwater_max_abs=float(value[1]),
                clear_mean_displacement=float(value[2]/(3*total))))
        print(json.dumps(dict(event="certify_training", view=view, seconds=time.perf_counter()-started)), flush=True)
    selected = {}
    for direction in models:
        for index, scale in enumerate(SCALES):
            matches = [r for r in rows if r["direction"] == direction and r["scale"] == scale]
            if max(r["underwater_rmse"] for r in matches) <= EPSILON:
                selected[direction] = variants[(direction, index)]
                torch.save(dict(model=selected[direction].state_dict(), scale=scale, epsilon=EPSILON),
                           directory / (direction + "_feasible.pt"))
                break
        assert direction in selected
    write_json(directory / "training_certification.json", dict(rows=rows, constraint="Every full training image RGB RMSE <= 1/255 from original prediction",
        epsilon=EPSILON, scales=list(SCALES), all_pixels=True, selection_uses_test_views=False,
        selected={direction: read_selected_scale(directory, direction) for direction in selected}))
    return selected


def read_selected_scale(directory, direction):
    return float(torch.load(directory / (direction + "_feasible.pt"), map_location="cpu")["scale"])


@torch.no_grad()
def evaluate(worker, branch, ext, models, directory):
    rows = []
    for _, view, camera, batch in worker._eval_records(branch):
        k, native, outputs, _ = capture_view(worker, branch.pipeline.model, camera, batch)
        assert k["colors"].min() >= 0 and k["colors"].max() <= 1
        total = k["width"] * k["height"]
        bins = _geometry_bins(k)
        pred = {arm: torch.empty(total, 3, device="cuda") for arm in models}
        clear = {arm: torch.empty(total, 3, device="cuda") for arm in models}
        identity = FrozenRadiometricTransport(appearance_shift=True).cuda()
        identity_error = 0.
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, total, 8192):
            pixels = torch.arange(start, min(start+8192, total), device="cuda", dtype=torch.int64)
            data = prepare(ext, k, bins, pixels, outputs)
            ref, j = identity(data)
            identity_error = max(identity_error, float((ref-data["base"]).abs().max()),
                float((j-native.j_raw.reshape(-1, 3)[pixels]).abs().max()))
            for arm, model in models.items():
                pred[arm][pixels], clear[arm][pixels] = model(data)
        assert identity_error < 2e-6, (view, identity_error)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        original = outputs["pred_image"]
        base_clear = native.j_raw
        target = directory / "renders" / view
        target.mkdir(parents=True)
        source = origin(directory.name, "E0") / "renders/E0" / directory.name / "step_14999" / view / "A2"
        for name, source_name in (("underwater_original.png", "underwater.png"), ("clear_original.png", "clear_raw_display_clamp01.png")):
            (target/name).symlink_to(source/source_name)
        arrays = dict(original_underwater=original.cpu().numpy(), original_clear=base_clear.cpu().numpy())
        stack = [base_clear]
        for arm in models:
            image, j = pred[arm].reshape_as(original), clear[arm].reshape_as(original)
            assert torch.isfinite(image).all() and torch.isfinite(j).all()
            error = image-original
            row = dict(scene=directory.name, view_id=view, arm=arm,
                underwater_rmse=float(error.square().mean().sqrt()), underwater_max_abs=float(error.abs().max()),
                clear_mae=float((j-base_clear).abs().mean()), clear_mean_displacement=float((j-base_clear).mean()),
                clear_min=float(j.min()), clear_max=float(j.max()),
                original_near_black_fraction=float((base_clear.max(-1).values <= 3/255).float().mean()),
                near_black_fraction=float((j.max(-1).values <= 3/255).float().mean()),
                coverage_change=0., identity_max_error=identity_error, workflow_seconds=elapsed,
                workflow_peak_allocated_bytes=peak)
            rows.append(row)
            stack.append(j)
            save_rgb(target/(arm+"_clear.png"), j)
            save_rgb(target/(arm+"_underwater.png"), image)
            save_rgb(target/(arm+"_underwater_difference_x20.png"), 20*error.abs())
            save_rgb(target/(arm+"_clear_difference_x4.png"), 4*(j-base_clear).abs())
            arrays[arm+"_underwater"] = image.cpu().numpy()
            arrays[arm+"_clear"] = j.cpu().numpy()
            print(json.dumps(dict(event="eval", **row)), flush=True)
        stack = torch.stack(stack)
        width = stack.max(0).values-stack.min(0).values
        save_rgb(target/"restricted_clear_range_x4.png", 4*width)
        np.savez_compressed(target/"raw_alternatives.npz", **arrays)
    write_csv(directory/"per_view.csv", rows)
    return rows


def run_scene(args):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == args.gpu
    directory = args.output_root / args.scene
    directory.mkdir(parents=True, exist_ok=False)
    worker = configure_worker("E0")
    worker.ACTIVE_SOURCE_ROOT, worker.ACTIVE_SOURCE_CONFIG, _ = source_assets(args.scene)
    worker._runtime(args.scene, args.gpu)
    branch = worker._new_branch(args.scene)
    checkpoint = origin(args.scene, "E0") / args.scene / "E0/checkpoints/step-000014999.ckpt"
    digest = sha256(checkpoint)
    started = time.perf_counter()
    try:
        worker._load_snapshot(branch, args.scene, checkpoint)
        before = worker._model_state_hash(branch.pipeline.model)
        ext = extension()
        data = cache(worker, branch, ext, directory)
        del data["gt"]
        assert data["colors"].min() >= 0 and data["colors"].max() <= 1
        models = {direction: train_direction(direction, data, directory) for direction in ("lower", "higher")}
        del data
        feasible = certify_training(worker, branch, ext, models, directory)
        rows = evaluate(worker, branch, ext, feasible, directory)
        assert before == worker._model_state_hash(branch.pipeline.model) and digest == sha256(checkpoint)
        write_json(directory/"result.json", dict(complete=True, rows=rows, checkpoint=str(checkpoint), checkpoint_sha256=digest,
            source_unchanged=True, geometry_frozen=True, source_colors_unmodified=True,
            selected={direction: read_selected_scale(directory, direction) for direction in feasible},
            wall_seconds=time.perf_counter()-started, clear_accuracy_claim=False, global_bound_claim=False))
    finally:
        worker._release(branch)


def launch(args):
    args.output_root.mkdir(parents=True, exist_ok=False)
    assert not any(gpu_processes().values())
    write_json(args.output_root/"registration.json", dict(candidate=9, steps_per_direction=1500, lr=.003, seed=42,
        train_pixels_per_camera=4096, batch_pixels_per_camera=32, directions=["lower", "higher"], parameters=12,
        family="Global RGB logit shift +/-0.5; betaD/betaB log scale +/-1; ambient logit shift +/-1; raw bounded +/-3",
        objective="4*relu(max_sampled_view_MSE/epsilon^2-0.7)^2 - direction*mean_clear_displacement/0.05",
        epsilon=EPSILON, post_search_scales=list(SCALES), certification="All pixels of every training view; largest passing predefined scale",
        selection_uses_test_views=False, loss_uses_gt=False, original_predictions_preserved=True,
        limitations=["Restricted augmented post-transform family, not original MLP/SH parameter space",
            "Not a posterior, calibrated uncertainty interval, or proof of global bounds",
            "Mean RGB displacement is an ambiguity probe, not an image-quality objective",
            "Per-image RMSE constraint permits larger local errors; maximum errors are reported"]))
    jobs = []
    for scene, gpu in zip(SCENES, ("6", "7", "8", "9")):
        handle = (args.output_root/(scene+".log")).open("x")
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--scene", scene, "--gpu", gpu,
            "--output-root", str(args.output_root)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4"),
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((scene, process, handle))
    while any(process.poll() is None for _, process, _ in jobs):
        write_json(args.output_root/"queue_status.json", {scene: process.poll() for scene, process, _ in jobs})
        time.sleep(5)
    status = {scene: process.returncode for scene, process, _ in jobs}
    for _, _, handle in jobs:
        handle.close()
    write_json(args.output_root/"queue_status.json", status)
    assert not any(status.values()), status
    means = []
    for scene in SCENES:
        result = read_json(args.output_root/scene/"result.json")
        for arm in ("lower", "higher"):
            rows = [r for r in result["rows"] if r["arm"] == arm]
            means.append(dict(scene=scene, arm=arm, scale=result["selected"][arm],
                **{key: statistics.mean(r[key] for r in rows) for key in ("underwater_rmse", "clear_mae", "clear_mean_displacement")},
                max_test_pixel_error=max(r["underwater_max_abs"] for r in rows)))
    write_csv(args.output_root/"scene_means.csv", means)
    write_json(args.output_root/"summary.json", dict(complete=True, means=means, test_views=13,
        training_feasibility_verified_on_all_pixels=True, all_source_checkpoints_unchanged=True,
        clear_accuracy_claim=False, posterior_claim=False, sota_claim=False))
    print(json.dumps(dict(complete=True, means=means)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=("6", "7", "8", "9"))
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    check_environment()
    run_scene(args) if args.scene else launch(args)
