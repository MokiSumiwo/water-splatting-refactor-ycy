#!/usr/bin/env python3
"""Frozen direct-light gain fits, shared-region diagnosis and native evaluation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from scripts.diagnostics.audit_dvs_frozen_visibility import SCENES, origin, save_rgb, write_csv
from scripts.diagnostics.audit_multiview_degradation_responsibility import _geometry_bins
from scripts.experiments.run_crcc_bnd_on import configure_worker
from scripts.experiments.run_taba_bnd_on import check_environment, gpu_processes, read_json, sha256, source_assets, write_json
from scripts.experiments.run_world_medium_screen import capture_view, extension

ARCHIVE = ROOT / "outputs/bnd_transport_recovery_20260912_formal"
HISTORY = ROOT / "outputs/legacy_ocmc_cica_no_auxiliary_20260905"
PROTOCOL = ROOT / "research_notes/BND_DIRECT_GAIN_PROTOCOL_2026-09-14.md"
ARMS = ("scalar", "spatial")
SEEDS = (42, 123, 3407)
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")
GATED = False


def mean(values):
    return statistics.mean(values)


def digest_tensor(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def basis(points, center, extent):
    x, y, z = ((points-center)/extent).clamp(-1, 1).unbind(-1)
    return torch.stack((torch.ones_like(x), x, y, z, x*y, x*z, y*z, x*x, y*y, z*z), -1)


class DirectGain(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.coefficients = nn.Parameter(torch.zeros(1 if arm == "scalar" else 10))

    def gains(self, features):
        logits = features[..., :self.coefficients.numel()] @ self.coefficients
        return (.35*logits.tanh()).exp()

    def forward(self, data, features):
        gains = self.gains(features)
        return data["base"] + (data["contribution"]*(gains[data["ids"]]-1)[..., None]).sum(1)


@torch.no_grad()
def fixed_data(ext, k, native, outputs, gt, pixels, features):
    bins = _geometry_bins(k)
    capacity = 64
    while True:
        ids, weights, counts = ext.forward(pixels.contiguous(), bins[0], bins[1], k["xys"], k["conics"],
            k["opacities"].reshape(-1), k["height"], k["width"], capacity)
        maximum = int(counts.max())
        if maximum <= capacity:
            break
        capacity = 1 << (maximum-1).bit_length()
        if capacity > 8192:
            raise RuntimeError("No truncated ray supports are permitted")
    size = max(1, maximum)
    ids, weights = ids[:, :size].long(), weights[:, :size]
    depth = torch.where(weights > 0, k["depths"].reshape(-1)[ids], 0.)
    beta = k["medium_attn"].reshape(-1, 3)[pixels]
    contribution = weights[..., None]*k["colors"][ids]*(-beta[:, None]*depth[..., None]).exp()
    error = float((contribution.sum(1)-native.rgb_object.reshape(-1, 3)[pixels]).abs().max())
    if error > 2e-6:
        raise RuntimeError("Fixed native support mismatch: %g" % error)
    assert features.shape[0] == k["colors"].shape[0]
    return dict(ids=ids, weights=weights, contribution=contribution, base=outputs["pred_image"].reshape(-1, 3)[pixels],
                gt=gt.reshape(-1, 3)[pixels]), dict(max_contributors=maximum, direct_error=error)


@torch.no_grad()
def native_gain(renderer, k, native, original, adapter, features):
    gain = adapter.gains(features)
    other = renderer.rasterize(**dict(k, colors=(k["colors"]*gain[:, None]).contiguous()))
    for field in ("rgb_medium", "accumulation", "depth", "depth_second_moment", "final_transmittance"):
        if not torch.equal(getattr(other, field), getattr(native, field)):
            raise RuntimeError("Frozen field changed: " + field)
    # Preserve the model's original tail/background recomposition and clear path.
    return original + (other.rgb_object-native.rgb_object), native.j_raw


def regions(gt, on_acc, off_acc, transmission, off_clear):
    edge = torch.zeros(gt.shape[:2], device=gt.device, dtype=torch.bool)
    edge[1:] |= (gt[1:]-gt[:-1]).abs().amax(-1) >= .03
    edge[:, 1:] |= (gt[:, 1:]-gt[:, :-1]).abs().amax(-1) >= .03
    edge = F.max_pool2d(edge.float()[None, None], 5, 1, 2)[0, 0] > 0
    support = (on_acc.squeeze() >= .5) & (off_acc.squeeze() >= .5)
    bright = gt.mean(-1) >= .65
    trans = transmission.mean(-1)
    overflow = off_clear.amax(-1) > 1.
    return dict(all=torch.ones_like(support), object_proxy=support, boundary=support & edge,
        interior=support & ~edge, bright=support & bright, ordinary=support & ~bright,
        low_trans=support & (trans < .35), high_trans=support & (trans >= .7),
        mid_trans=support & (trans >= .35) & (trans < .7), overflow=overflow,
        nonoverflow=~overflow, outside_support=~support)


def regional_metrics(pred, gt, masks):
    error = (pred.clamp(0, 1)-gt).double().square().mean(-1)
    total = float(error.sum())
    rows = []
    for name, mask in masks.items():
        count = int(mask.sum())
        mse = float(error[mask].mean()) if count else None
        rows.append(dict(region=name, pixels=count, fraction=count/mask.numel(), MSE=mse,
            PSNR=-10*math.log10(max(mse, 1e-30)) if count else None,
            total_error_share=float(error[mask].sum())/max(total, 1e-30)))
    return rows


@torch.no_grad()
def diagnose(worker, branch, directory, smoke=False):
    rows, decompositions = [], []
    records = worker._eval_records(branch)
    if smoke:
        records = records[:1]
    for _, view, camera, batch in records:
        k, native, outputs, gt = capture_view(worker, branch.pipeline.model, camera, batch)
        target = directory/"renders"/view
        target.mkdir(parents=True, exist_ok=False)
        previous = ARCHIVE/directory.name/"renders"/view
        with np.load(previous/"original_float.npz") as old:
            for name, value in (("underwater", outputs["pred_image"]), ("clear", native.j_raw), ("gt", gt)):
                assert np.array_equal(old[name], value.cpu().numpy()), (view, name)
        with np.load(ARCHIVE/"historical_bnd_off"/directory.name/view/"float.npz") as old:
            off = {key: torch.from_numpy(old[key]).cuda() for key in old.files}
        masks = regions(gt, native.accumulation, off["accumulation"], outputs["transmission"], off["clear"])
        np.savez_compressed(target/"regions.npz", **{n: v.cpu().numpy() for n, v in masks.items()})
        for name, pred in (("B0", outputs["pred_image"]), ("OFF", off["underwater"])):
            rows.extend(dict(scene=directory.name, view=view, arm=name, **r) for r in regional_metrics(pred, gt, masks))
            residual = (pred.clamp(0, 1)-gt).double()
            h, w = residual.shape[:2]
            crop = residual[:h//16*16, :w//16*16].reshape(h//16, 16, w//16, 16, 3)
            energy = float(residual.square().mean())
            cropped_energy = float(crop.square().mean())
            block = float(crop.mean((1, 3)).square().mean())
            decompositions.append(dict(scene=directory.name, view=view, arm=name, mse=energy,
                achromatic_energy=float(residual.mean(-1).square().mean()),
                cropped_mse=cropped_energy, block_mean_energy=block, detail_energy=cropped_energy-block))
        for filename in ("original_float.npz", "A0_underwater.png", "A0_clear.png", "gt.png"):
            (target/filename).symlink_to(previous/filename)
        off_dir = ARCHIVE/"historical_bnd_off"/directory.name/view
        (target/"OFF_underwater.png").symlink_to(off_dir/"underwater.png")
        (target/"OFF_clear.png").symlink_to(off_dir/"clear_clamp01.png")
        diagnostic = torch.stack((masks["boundary"], masks["low_trans"], masks["overflow"]), -1).float()
        save_rgb(target/"region_overlay.png", .6*gt + .4*diagnostic)
        save_rgb(target/"B0_error_x5.png", (outputs["pred_image"].clamp(0, 1)-gt).abs()*5)
        save_rgb(target/"OFF_error_x5.png", (off["underwater"].clamp(0, 1)-gt).abs()*5)
        print(json.dumps(dict(event="diagnosis", scene=directory.name, view=view)), flush=True)
    write_json(directory/"diagnosis.json", dict(rows=rows, decompositions=decompositions, archived_floats_identical=True))
    write_csv(directory/"region_metrics.csv", rows)


@torch.no_grad()
def cache(worker, branch, ext, directory, features, smoke=False):
    records = sorted(worker._train_records(branch), key=lambda r: r[1])
    if smoke:
        records = records[:3]
    chunks, manifest = [], []
    for rank, (index, view, camera, batch) in enumerate(records):
        validation = rank % 5 == 2
        count = (16384 if validation else 8192) if not smoke else 1024
        k, native, outputs, gt = capture_view(worker, branch.pipeline.model, camera, batch)
        pixels = torch.randperm(k["height"]*k["width"], generator=torch.Generator().manual_seed(91400+index))[:count].cuda()
        data, checks = fixed_data(ext, k, native, outputs, gt, pixels, features)
        data["camera"] = torch.full((count,), rank, device="cuda", dtype=torch.long)
        data["validation"] = torch.full((count,), validation, device="cuda", dtype=torch.bool)
        # Restrict fitting to pixels with reliable direct support.  This gate
        # is detached and shared by both adapter architectures; validation
        # metrics still cover every sampled validation pixel.
        accumulation = data["weights"].sum(-1)
        transmission = outputs["transmission"].reshape(-1, 3)[pixels].mean(-1)
        direct = data["contribution"].sum(1).amax(-1)
        eligible = (accumulation >= 0.5) & (transmission >= 0.35) & (transmission <= 0.85) & (direct >= 1e-3)
        data["eligible"] = eligible if GATED else torch.ones_like(eligible)
        for arm in ARMS:
            identity = DirectGain(arm).cuda()
            assert torch.equal(identity(data, features), data["base"])
        probe = DirectGain("spatial").cuda()
        probe.coefficients.copy_(torch.linspace(-.15, .15, 10, device="cuda"))
        new, clear = native_gain(branch.pipeline.model.underwater_rasterizer, k, native, outputs["pred_image"], probe, features)
        native_error = float((probe(data, features)-new.reshape(-1, 3)[pixels]).abs().max())
        assert native_error <= 2e-6 and torch.equal(clear, native.j_raw)
        chunks.append({key: value.cpu() for key, value in data.items()})
        manifest.append(dict(rank=rank, index=index, view=view, validation=validation, eligible_pixels=int(eligible.sum()),
            eligible_fraction=float(eligible.float().mean()), pixels=pixels.cpu().tolist(),
            native_gain_error=native_error, **checks))
        print(json.dumps(dict(event="cache", view=view, validation=validation, native_gain_error=native_error, **checks)), flush=True)
    maximum = max(c["ids"].shape[1] for c in chunks)
    for chunk in chunks:
        padding = maximum-chunk["ids"].shape[1]
        chunk["ids"] = F.pad(chunk["ids"], (0, padding))
        chunk["weights"] = F.pad(chunk["weights"], (0, padding))
        chunk["contribution"] = F.pad(chunk["contribution"], (0, 0, 0, padding))
    data = {key: torch.cat([c[key] for c in chunks]) for key in chunks[0]}
    torch.save(data, directory/"ray_cache.pt")
    write_json(directory/"camera_pixels.json", manifest)
    return {key: value.cuda() for key, value in data.items()}


@torch.no_grad()
def sampled_metrics(adapter, data, features, validation):
    cameras = data["camera"][data["validation"] == validation].unique()
    values = []
    for camera in cameras:
        indices = (data["camera"] == camera).nonzero().flatten()
        total = 0.
        for local in indices.split(2048):
            batch = {k: v[local] for k, v in data.items()}
            total += float((adapter(batch, features).clamp(0, 1)-batch["gt"]).double().square().sum())
        values.append(-10*math.log10(max(total/(3*indices.numel()), 1e-30)))
    return mean(values)


def fit(data, features, arm, seed, directory, steps, teacher=False):
    label = arm+"_teacher" if teacher else arm
    target = directory/("seed_%d" % seed)/label
    target.mkdir(parents=True, exist_ok=False)
    adapter = DirectGain(arm).cuda()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.01)
    fit_indices = ((~data["validation"]) & data["eligible"]).nonzero().flatten()
    if fit_indices.numel() < 1024:
        raise RuntimeError("Transmission gate left too few fitting pixels: %d" % fit_indices.numel())
    generator = torch.Generator().manual_seed(seed)
    sequence = hashlib.sha256()
    initial = sampled_metrics(adapter, data, features, True)
    best = initial
    selected = dict(step=0, model=copy.deepcopy(adapter.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()))
    curve = [dict(step=0, val_psnr=initial)]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, steps+1):
        selection = torch.randint(fit_indices.numel(), (1024,), generator=generator)
        sequence.update(selection.numpy().tobytes())
        ids = fit_indices[selection.cuda()]
        batch = {k: v[ids] for k, v in data.items()}
        optimizer.zero_grad(set_to_none=True)
        pred = adapter(batch, features)
        loss = (pred-batch["gt"]).square().mean()
        if teacher:
            loss = loss + .1*(pred-batch["teacher"]).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(adapter.coefficients.grad).all()
        optimizer.step()
        if step % 250 == 0 or step == steps:
            val = sampled_metrics(adapter, data, features, True)
            curve.append(dict(step=step, val_psnr=val, train_batch_loss=float(loss.detach())))
            if val > best:
                best = val
                selected = dict(step=step, model=copy.deepcopy(adapter.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()))
            print(json.dumps(dict(event="fit", scene=directory.name, arm=label, seed=seed, step=step, val_gain=val-initial, best_gain=best-initial)), flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    torch.save(dict(model=adapter.state_dict(), optimizer=optimizer.state_dict(), step=steps), target/"last.pt")
    torch.save(selected, target/"selected.pt")
    adapter.load_state_dict(selected["model"])
    result = dict(arm=label, architecture=arm, seed=seed, steps=steps, selected_step=selected["step"],
        initial_val_psnr=initial, selected_val_psnr=best, val_gain=best-initial, curve=curve,
        selected_fit_psnr=sampled_metrics(adapter, data, features, False), wall_seconds=seconds,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), sampling_sha256=sequence.hexdigest(),
        coefficients=adapter.coefficients.detach().cpu().tolist(), parameter_count=adapter.coefficients.numel())
    write_json(target/"training.json", result)
    return adapter


def load_models(directory, labels):
    models = {}
    for seed in SEEDS:
        for label in labels:
            model = DirectGain(label.replace("_teacher", "")).cuda()
            model.load_state_dict(torch.load(directory/("seed_%d" % seed)/label/"selected.pt", weights_only=False)["model"])
            models[(seed, label)] = model
    return models


@torch.no_grad()
def evaluate(worker, branch, ext, directory, features, models, validation, smoke=False):
    if validation:
        names = {r["view"] for r in read_json(directory/"camera_pixels.json") if r["validation"]}
        records = [r for r in worker._train_records(branch) if r[1] in names]
    else:
        records = worker._eval_records(branch)
        if smoke:
            records = records[:1]
    rows, region_rows = [], []
    for _, view, camera, batch in records:
        torch.cuda.synchronize()
        started = time.perf_counter()
        k, native, outputs, gt = capture_view(worker, branch.pipeline.model, camera, batch)
        torch.cuda.synchronize()
        baseline_seconds = time.perf_counter()-started
        original = outputs["pred_image"]
        base_metrics = worker._taba_rgb_metrics(original.clamp(0, 1), gt)
        rows.append(dict(view=view, arm="B0", seed=0, **base_metrics, render_seconds=baseline_seconds))
        pixels = torch.arange(0, k["height"]*k["width"], max(1, k["height"]*k["width"]//2048), device="cuda")[:2048]
        data, _ = fixed_data(ext, k, native, outputs, gt, pixels, features)
        if not validation:
            target = directory/"renders"/view
            with np.load(target/"regions.npz") as stored:
                masks = {n: torch.from_numpy(stored[n]).cuda() for n in stored.files}
            with np.load(target/"original_float.npz") as stored:
                assert np.array_equal(stored["clear"], native.j_raw.cpu().numpy())
                assert np.array_equal(stored["underwater"], original.cpu().numpy())
        for (seed, label), adapter in models.items():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            pred, clear = native_gain(branch.pipeline.model.underwater_rasterizer, k, native, original, adapter, features)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            peak = torch.cuda.max_memory_allocated()
            error = float((adapter(data, features)-pred.reshape(-1, 3)[pixels]).abs().max())
            assert error <= 2e-6 and torch.equal(clear, native.j_raw)
            metrics = worker._taba_rgb_metrics(pred.clamp(0, 1), gt)
            gain = adapter.gains(features)
            row = dict(view=view, arm=label, seed=seed, **metrics, delta_psnr=metrics["PSNR"]-base_metrics["PSNR"],
                adapter_render_seconds=elapsed, total_render_seconds=baseline_seconds+elapsed,
                incremental_peak_bytes=peak-allocated, peak_allocated_bytes=peak, fixed_native_error=error,
                clear_max_error=0., clear_sha256=digest_tensor(clear), gain_min=float(gain.min()), gain_max=float(gain.max()),
                underwater_min=float(pred.min()), underwater_max=float(pred.max()))
            rows.append(row)
            if not validation:
                prefix = "seed_%d_%s" % (seed, label)
                save_rgb(target/(prefix+"_underwater.png"), pred)
                save_rgb(target/(prefix+"_difference_x10.png"), (pred-original).abs()*10)
                save_rgb(target/(prefix+"_error_x5.png"), (pred.clamp(0, 1)-gt).abs()*5)
                (target/(prefix+"_clear.png")).symlink_to("A0_clear.png")
                np.savez_compressed(target/(prefix+"_float.npz"), underwater=pred.cpu().numpy())
                region_rows.extend(dict(view=view, arm=label, seed=seed, **r) for r in regional_metrics(pred, gt, masks))
            print(json.dumps(dict(event="validation" if validation else "test", scene=directory.name, **row)), flush=True)
    return dict(rows=rows, regions=region_rows)


def setup(scene, gpu):
    worker = configure_worker("E0")
    worker.ACTIVE_SOURCE_ROOT, worker.ACTIVE_SOURCE_CONFIG, _ = source_assets(scene)
    worker._runtime(scene, gpu)
    branch = worker._new_branch(scene)
    checkpoint = origin(scene, "E0")/scene/"E0/checkpoints/step-000014999.ckpt"
    worker._load_snapshot(branch, scene, checkpoint)
    branch.pipeline.model.requires_grad_(False)
    return worker, branch, checkpoint


@torch.no_grad()
def teacher_cache(scene, gpu, directory, data):
    worker = configure_worker("E0")
    worker.ACTIVE_PARAMETERIZATION = "legacy"
    worker.ACTIVE_SOURCE_CONFIG = HISTORY/scene/"source_config.yml"
    worker.ACTIVE_SOURCE_ROOT = HISTORY
    worker._runtime(scene, gpu)
    branch = worker._new_branch(scene)
    checkpoint = HISTORY/scene/"checkpoints/step-000014999.ckpt"
    worker._load_snapshot(branch, scene, checkpoint)
    model = branch.pipeline.model.requires_grad_(False)
    before = worker._model_state_hash(model)
    predictions = data["base"].clone()
    manifest = read_json(directory/"camera_pixels.json")
    records = {r[1]: r for r in worker._train_records(branch)}
    rows = []
    start = time.perf_counter()
    try:
        for row in manifest:
            if row["validation"]:
                continue
            _, view, camera, batch = records[row["view"]]
            _, _, outputs, gt = capture_view(worker, model, camera, batch)
            pixels = torch.tensor(row["pixels"], device="cuda")
            indices = (data["camera"] == row["rank"]).nonzero().flatten()
            assert torch.equal(gt.reshape(-1, 3)[pixels], data["gt"][indices])
            pred = outputs["pred_image"].clamp(0, 1).reshape(-1, 3)[pixels]
            predictions[indices] = pred
            rows.append(dict(view=view, teacher_mse=float((pred-data["gt"][indices]).square().mean()),
                source_mse=float((data["base"][indices].clamp(0, 1)-data["gt"][indices]).square().mean())))
        assert before == worker._model_state_hash(model)
        write_json(directory/"teacher_provenance.json", dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint),
            source_unchanged=True, historical_not_paired=True, fit_views_only=True,
            wall_seconds=time.perf_counter()-start, rows=rows))
    finally:
        worker._release(branch)
    data["teacher"] = predictions


def scene_run(args):
    directory = args.output/args.scene
    if args.phase == "fit":
        directory.mkdir(parents=True, exist_ok=False)
    if args.phase == "final":
        gate = read_json(args.output/"validation_gate.json")
        teacher_arm = gate["selected_architecture"] if gate["passed"] else None
        if teacher_arm:
            data = {k: v.cuda() for k, v in torch.load(directory/"ray_cache.pt", weights_only=False).items()}
            teacher_cache(args.scene, args.gpu, directory, data)
            geometry = torch.load(directory/"gain_coordinates.pt", weights_only=False)
            features = geometry["features"].cuda()
            for seed in SEEDS:
                fit(data, features, teacher_arm, seed, directory, args.steps, teacher=True)
            del data, features
    worker, branch, checkpoint = setup(args.scene, args.gpu)
    source = branch.pipeline.model
    before, source_hash = worker._model_state_hash(source), sha256(checkpoint)
    try:
        ext = extension()
        if args.phase == "fit":
            with torch.no_grad():
                quantiles = torch.quantile(source.means, torch.tensor([.02, .98], device="cuda"), dim=0)
                center = quantiles.mean(0)
                extent = ((quantiles[1]-quantiles[0])*.5).clamp_min(1e-4)
                features = basis(source.means, center, extent).detach()
            torch.save(dict(center=center.cpu(), extent=extent.cpu(), features=features.cpu()), directory/"gain_coordinates.pt")
            diagnose(worker, branch, directory, args.smoke)
            data = cache(worker, branch, ext, directory, features, args.smoke)
            models = {(seed, arm): fit(data, features, arm, seed, directory, args.steps)
                      for seed in SEEDS for arm in ARMS}
            del data
            validations = evaluate(worker, branch, ext, directory, features, models, True, args.smoke)
            write_json(directory/"validation.json", validations)
        else:
            geometry = torch.load(directory/"gain_coordinates.pt", weights_only=False)
            features = geometry["features"].cuda()
            assert torch.equal(features, basis(source.means, geometry["center"].cuda(), geometry["extent"].cuda()))
            labels = list(ARMS)+([teacher_arm+"_teacher"] if teacher_arm else [])
            models = load_models(directory, labels)
            if teacher_arm:
                validation = evaluate(worker, branch, ext, directory, features,
                    {k: v for k, v in models.items() if k[1].endswith("_teacher")}, True)
                write_json(directory/"teacher_validation.json", validation)
            result = evaluate(worker, branch, ext, directory, features, models, False, args.smoke)
            write_json(directory/"test.json", result)
            sheets(directory, result["rows"], labels)
        assert before == worker._model_state_hash(source) and source_hash == sha256(checkpoint)
        write_json(directory/(args.phase+"_preservation.json"), dict(model_sha256=before, checkpoint_sha256=source_hash,
            source_unchanged=True, clear_unchanged=True, checkpoint=str(checkpoint), complete=True))
    finally:
        worker._release(branch)


def aggregate(results):
    scene_rows = []
    for scene, result in results.items():
        labels = sorted({r["arm"] for r in result["rows"]})
        for label in labels:
            selected = [r for r in result["rows"] if r["arm"] == label]
            per_seed = [{m: mean(r[m] for r in selected if r["seed"] == seed) for m in METRICS}
                        for seed in sorted({r["seed"] for r in selected})]
            scene_rows.append(dict(scene=scene, arm=label, **{m: mean(r[m] for r in per_seed) for m in METRICS},
                psnr_seed_std=statistics.stdev(r["PSNR"] for r in per_seed) if len(per_seed)>1 else 0.))
    macro = {label: {m: mean(r[m] for r in scene_rows if r["arm"] == label) for m in METRICS}
             for label in sorted({r["arm"] for r in scene_rows})}
    return scene_rows, macro


def validation_gate(output):
    results = {scene: read_json(output/scene/"validation.json") for scene in SCENES}
    rows, macro = aggregate(results)
    arm = max(ARMS, key=lambda a: macro[a]["PSNR"])
    scene_deltas = {s: next(r["PSNR"] for r in rows if r["scene"] == s and r["arm"] == arm)
        - next(r["PSNR"] for r in rows if r["scene"] == s and r["arm"] == "B0") for s in SCENES}
    seed_deltas = {seed: mean(mean(r["PSNR"] for r in result["rows"] if r["arm"] == arm and r["seed"] == seed)
        - mean(r["PSNR"] for r in result["rows"] if r["arm"] == "B0") for result in results.values()) for seed in SEEDS}
    checks = dict(macro_gain_003=macro[arm]["PSNR"]-macro["B0"]["PSNR"] >= .03,
        three_scenes_gain_001=sum(d >= .01 for d in scene_deltas.values()) >= 3,
        scene_safety=min(scene_deltas.values()) >= -.05, all_seeds_positive=min(seed_deltas.values()) > 0,
        lpips_nonworse=macro[arm]["LPIPS"] <= macro["B0"]["LPIPS"],
        preservation=all(read_json(output/s/"fit_preservation.json")["source_unchanged"] for s in SCENES))
    result = dict(selected_architecture=arm, passed=all(checks.values()), checks=checks, scene_deltas=scene_deltas,
        seed_deltas=seed_deltas, macro=macro, scene_means=rows, test_used=False)
    write_json(output/"validation_gate.json", result)
    print(json.dumps(dict(event="validation_gate", **result)), flush=True)


def sheets(directory, rows, labels):
    views = [r["view"] for r in rows if r["arm"] == "B0"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for kind, columns in (
        ("underwater", [("GT", "gt.png"), ("OFF historical", "OFF_underwater.png"), ("BND ON", "A0_underwater.png")]
         + [(a, "seed_42_"+a+"_underwater.png") for a in labels]),
        ("clear", [("OFF historical", "OFF_clear.png"), ("BND ON", "A0_clear.png")]
         + [(a+" preserved", "seed_42_"+a+"_clear.png") for a in labels]),
        ("errors", [("BND error x5", "B0_error_x5.png"), ("OFF error x5", "OFF_error_x5.png")]
         + [(a+" error x5", "seed_42_"+a+"_error_x5.png") for a in labels]),
        ("changes", [("GT", "gt.png"), ("R:boundary G:lowT B:overflow", "region_overlay.png")]
         + [(a+" change x10", "seed_42_"+a+"_difference_x10.png") for a in labels])):
        tw, th = 460, 380
        canvas = Image.new("RGB", (tw*len(columns), 60+th*len(views)), "#181818")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), directory.name+" | "+kind+" | fixed continuation seed 42", fill="white", font=font)
        draw.text((10, 32), "Common clamp [0,1]; no exposure adjustment. OFF is not a paired training ablation.", fill="#cccccc", font=font)
        for i, view in enumerate(views):
            for j, (label, filename) in enumerate(columns):
                x, y = j*tw, 60+i*th
                draw.text((x+8, y+4), view+" | "+label, fill="white", font=font)
                if kind == "underwater" and label != "GT":
                    if label == "OFF historical":
                        old = read_json(ARCHIVE/"historical_bnd_off"/directory.name/"result.json")["rows"]
                        metric = next(r for r in old if r["view_id"] == view)
                    else:
                        arm = "B0" if label == "BND ON" else label
                        metric = next(r for r in rows if r["view"] == view and r["arm"] == arm and r["seed"] in (0, 42))
                    draw.text((x+8, y+28), "PSNR %.4f | LPIPS %.4f" % (metric["PSNR"], metric["LPIPS"]), fill="#cccccc", font=font)
                with Image.open(directory/"renders"/view/filename) as source:
                    image = source.convert("RGB")
                    image.thumbnail((tw-12, th-66), Image.Resampling.LANCZOS)
                canvas.paste(image, (x+(tw-image.width)//2, y+58))
        canvas.save(directory/(kind+"_comparison.jpg"), quality=96)


def summarize(output):
    results = {s: read_json(output/s/"test.json") for s in SCENES}
    rows, macro = aggregate(results)
    gate = read_json(output/"validation_gate.json")
    write_csv(output/"scene_means.csv", rows)
    write_csv(output/"per_view.csv", [dict(scene=s, **r) for s, v in results.items() for r in v["rows"]])
    write_csv(output/"region_metrics.csv", [r for s in SCENES for r in read_json(output/s/"diagnosis.json")["rows"]])
    write_csv(output/"decomposition.csv", [r for s in SCENES for r in read_json(output/s/"diagnosis.json")["decompositions"]])
    registration = read_json(output/"registration.json")
    unchanged = all(sha256(ROOT/p) == h for p, h in registration["code_sha256"].items())
    assert unchanged
    result = dict(complete=True, scene_means=rows, macro=macro, validation_gate=gate,
        teacher_stage="completed" if gate["passed"] else "not_triggered", test_views=13,
        all_source_and_clear_preserved=True, registered_files_unchanged=unchanged)
    write_json(output/"summary.json", result)
    print(json.dumps(dict(event="summary", **result)), flush=True)


def launch(args):
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), PROTOCOL, ROOT/"scripts/diagnostics/csrc/extract_fixed_rays.cu",
             ROOT/"water_splatting/rendering/underwater_rasterizer.py", ROOT/"water_splatting/water_splatting.py"]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    for path in files:
        target = args.output/"snapshots"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(args.output/"registration.json", dict(code_sha256=hashes, steps=args.steps, seeds=SEEDS,
        protocol=str(PROTOCOL), smoke=args.smoke, gated=GATED,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    if any(gpu_processes().values()):
        raise RuntimeError("Authorized GPUs occupied")
    for phase in ("fit", "final"):
        jobs = []
        try:
            for scene, gpu in zip(SCENES, ("6", "7", "8", "9")):
                log = (args.output/(scene+"_"+phase+".log")).open("x")
                command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                    "--scene", scene, "--gpu", gpu, "--phase", phase, "--steps", str(args.steps)]
                if args.smoke:
                    command += ["--smoke"]
                if args.gated:
                    command += ["--gated"]
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
                jobs.append((scene, process, log))
            while any(p.poll() is None for _, p, _ in jobs):
                write_json(args.output/"queue_status.json", dict(phase=phase, statuses={s: p.poll() for s, p, _ in jobs}))
                time.sleep(5)
            if any(p.returncode for _, p, _ in jobs):
                raise RuntimeError("Phase failed: " + repr({s: p.returncode for s, p, _ in jobs}))
        finally:
            for _, process, log in jobs:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                log.close()
        if phase == "fit":
            validation_gate(args.output)
    summarize(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/bnd_direct_gain_20260914")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=("6", "7", "8", "9"), default="6")
    parser.add_argument("--phase", choices=("fit", "final"), default="fit")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gated", action="store_true")
    args = parser.parse_args()
    GATED = bool(args.gated)
    check_environment()
    if args.scene:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == args.gpu
        scene_run(args)
    else:
        launch(args)
