#!/usr/bin/env python3
"""Registered BND-preserving medium refits and complete native-view evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from scripts.diagnostics.audit_dvs_frozen_visibility import SCENES, origin, rgb8, save_rgb, write_csv
from scripts.diagnostics.audit_multiview_degradation_responsibility import _geometry_bins
from scripts.experiments.run_crcc_bnd_on import configure_worker
from scripts.experiments.run_taba_bnd_on import check_environment, gpu_processes, read_json, sha256, source_assets, write_json
from scripts.experiments.run_world_medium_screen import capture_view, extension, extract
from water_splatting.raoc import apply_standardized_projector
from water_splatting.rendering.optical_thickness_distribution import composite_fixed_transport

OUT = ROOT / "outputs/bnd_transport_recovery_20260912"
ARMS = ("A1", "A2", "A3")
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")


class ContinuedMedium(nn.Module):
    def __init__(self, source, arm):
        super().__init__()
        self.mlp = copy.deepcopy(source.medium_mlp).requires_grad_(True)
        self.register_buffer("projector", source._camera_medium_observability_projector.detach().clone())
        scale = source._camera_medium_observability_scale
        self.register_buffer("scale", None if scale is None else scale.detach().clone())
        self.density_bias = float(source.medium_density_bias)
        self.arm = arm
        if arm != "A1":
            self.scalar = nn.Parameter(torch.zeros((), device=source.device))

    def fields(self, full_input, neutral_input):
        full, neutral = self.mlp(full_input), self.mlp(neutral_input)
        difference = full - neutral
        if self.scale is None:
            projected = (difference.float() @ self.projector.float().T).to(difference.dtype)
        else:
            projected = apply_standardized_projector(difference, self.projector, self.scale)
        raw = neutral + projected
        return (raw[..., :3].sigmoid().float(),
                F.softplus(raw[..., 3:6] + self.density_bias).float(),
                F.softplus(raw[..., 6:9] + self.density_bias).float())

    def forward(self, data):
        ambient, bs, beta = self.fields(data["full_input"], data["neutral_input"])
        return composite_fixed_transport(data["depths"], data["weights"], data["colors"], ambient, bs, beta,
            eta=self.scalar if self.arm == "A3" else None,
            optical_scale=self.scalar.exp() if self.arm == "A2" else 1.)

    @torch.no_grad()
    def project(self):
        if self.arm == "A2":
            self.scalar.clamp_(math.log(.1), math.log(10.))
        elif self.arm == "A3":
            self.scalar.clamp_(0., 1.)

    def scalar_value(self):
        if self.arm == "A1":
            return 0.
        return float((self.scalar.exp() if self.arm == "A2" else self.scalar).detach())


@torch.no_grad()
def capture(worker, model, camera, batch):
    inputs = []
    handle = model.medium_mlp.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach()))
    try:
        k, native, outputs, gt = capture_view(worker, model, camera, batch)
    finally:
        handle.remove()
    if len(inputs) != 2:
        raise RuntimeError("Expected exactly full and neutral OCMC medium evaluations, got %d" % len(inputs))
    assert k["colors"].min() >= 0 and k["colors"].max() <= 1
    return k, native, outputs, gt, inputs


def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def cache_training(worker, branch, ext, directory, count, smoke=False):
    model = branch.pipeline.model
    identity = ContinuedMedium(model, "A1").eval()
    chunks, manifest = [], []
    records = worker._train_records(branch)
    if smoke:
        records = records[:2]
    max_error = 0.
    for index, view, camera, batch in records:
        k, native, outputs, gt, inputs = capture(worker, model, camera, batch)
        total = k["height"] * k["width"]
        generator = torch.Generator().manual_seed(91200 + int(index))
        pixels = torch.randperm(total, generator=generator)[:count*2].cuda()
        assert pixels.numel() == count*2
        data, maximum = extract(ext, k, _geometry_bins(k), pixels)
        data.update(full_input=inputs[0][pixels].float(), neutral_input=inputs[1][pixels].float(),
                    gt=gt.reshape(-1, 3)[pixels], base=outputs["pred_image"].reshape(-1, 3)[pixels])
        data["diagnostic"] = torch.arange(count*2, device="cuda") >= count
        native_parameters = tuple(k[name].reshape(-1, 3)[pixels] for name in ("medium_rgb", "medium_bs", "medium_attn"))
        actual_parameters = identity.fields(data["full_input"], data["neutral_input"])
        field_error = max(float((a-b).abs().max()) for a, b in zip(actual_parameters, native_parameters))
        reconstructed = composite_fixed_transport(data["depths"], data["weights"], data["colors"], *native_parameters)
        fast_error = float((reconstructed-data["base"]).abs().max())
        prediction_error = float((identity(data)-data["base"]).abs().max())
        clear = (data["weights"][..., None]*data["colors"]).sum(1)
        clear_error = float((clear-native.j_raw.reshape(-1, 3)[pixels]).abs().max())
        max_error = max(max_error, prediction_error)
        if max(field_error, fast_error, prediction_error, clear_error) > 2e-6:
            raise RuntimeError("Native cache mismatch %s: %s" % (view, [field_error, fast_error, prediction_error, clear_error]))
        chunks.append({key: value.detach().cpu() for key, value in data.items()})
        manifest.append(dict(view=view, camera_index=index, fit_pixels=pixels[:count].cpu().tolist(),
            diagnostic_pixels=pixels[count:].cpu().tolist(), max_contributors=maximum, native_field_error=field_error,
            native_fast_error=fast_error, native_prediction_error=prediction_error, native_clear_error=clear_error))
        print(json.dumps(dict(event="cache", view=view, max_contributors=maximum, native_max_error=prediction_error)), flush=True)
    maximum = max(chunk["weights"].shape[1] for chunk in chunks)
    for chunk in chunks:
        padding = maximum-chunk["weights"].shape[1]
        for key in ("weights", "depths"):
            chunk[key] = F.pad(chunk[key], (0, padding))
        chunk["colors"] = F.pad(chunk["colors"], (0, 0, 0, padding))
    data = {key: torch.cat([chunk[key] for chunk in chunks]).cuda() for key in chunks[0]}
    write_json(directory/"training_pixels.json", dict(rows=manifest, max_contributors=maximum,
        fit_pixels_per_view=count, diagnostic_pixels_per_view=count, native_max_error=max_error,
        diagnostic_pixels_are_from_pretrained_training_images=True, independent_new_views=False))
    del identity
    return data


@torch.no_grad()
def cache_metrics(model, data, indices):
    error = 0.
    for start in range(0, indices.numel(), 2048):
        batch = {key: value[indices[start:start+2048]] for key, value in data.items() if key != "diagnostic"}
        error += float((model(batch)-batch["gt"]).double().square().sum())
    mse = error/(indices.numel()*3)
    return dict(mse=mse, psnr=-10*math.log10(max(mse, 1e-30)))


def train_arm(source, payload, data, arm, seed, steps, directory):
    target = directory/("seed_%d" % seed)/arm
    target.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = ContinuedMedium(source, arm)
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=.00015, eps=1e-15)
    optimizer.load_state_dict(copy.deepcopy(payload["optimizers"]["medium_mlp"]))
    scalar_optimizer = torch.optim.Adam([model.scalar], lr=.005, eps=1e-8) if arm != "A1" else None
    fit = (~data["diagnostic"]).nonzero().flatten()
    diagnostic = data["diagnostic"].nonzero().flatten()
    initial = cache_metrics(model, data, diagnostic)
    initial_hash = tensor_digest(torch.cat([p.detach().reshape(-1) for p in model.mlp.parameters()]))
    generator = torch.Generator().manual_seed(seed)
    sequence = hashlib.sha256()
    rows = []
    started = time.perf_counter()
    initial_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps+1):
        selection = torch.randint(fit.numel(), (1024,), generator=generator)
        sequence.update(selection.numpy().tobytes())
        indices = fit[selection.cuda()]
        batch = {key: value[indices] for key, value in data.items() if key != "diagnostic"}
        optimizer.zero_grad(set_to_none=True)
        if scalar_optimizer is not None:
            scalar_optimizer.zero_grad(set_to_none=True)
        loss = (model(batch)-batch["gt"]).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite or missing training gradient")
        scalar_gradient = float(model.scalar.grad) if scalar_optimizer else 0.
        optimizer.step()
        if scalar_optimizer:
            scalar_optimizer.step()
            model.project()
        if step == 1 or step % 1000 == 0 or step == 2500 or step == steps:
            result = cache_metrics(model, data, diagnostic)
            row = dict(step=step, minibatch_mse=float(loss.detach()), diagnostic_mse=result["mse"],
                scalar=model.scalar_value(), scalar_gradient=scalar_gradient, elapsed_seconds=time.perf_counter()-started)
            rows.append(row)
            print(json.dumps(dict(event="train", seed=seed, arm=arm, **row)), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()
    fit_metrics = cache_metrics(model, data, fit)
    final_metrics = cache_metrics(model, data, diagnostic)
    model.eval()
    torch.save(dict(model=model.state_dict(), medium_optimizer=optimizer.state_dict(),
        scalar_optimizer=scalar_optimizer.state_dict() if scalar_optimizer else None,
        arm=arm, seed=seed, updates=steps, density_bias=model.density_bias, delta=.5,
        source_checkpoint=str(origin(directory.name, "E0")/directory.name/"E0/checkpoints/step-000014999.ckpt")),
        target/"checkpoint.pt")
    write_json(target/"training.json", dict(complete=True, steps=steps, arm=arm, seed=seed, rows=rows,
        initial_diagnostic=initial, final_diagnostic=final_metrics, final_fit=fit_metrics,
        initial_medium_hash=initial_hash, sampling_sha256=sequence.hexdigest(), scalar=model.scalar_value(),
        parameter_count=sum(p.numel() for p in model.parameters()), lr=optimizer.param_groups[0]["lr"],
        wall_seconds=elapsed, peak_allocated_bytes=peak, incremental_peak_bytes=peak-initial_allocated,
        optimizer_state_restored=True, loss="sampled native RGB MSE"))
    return model


@torch.no_grad()
def native_prediction(renderer, k, model, inputs, reference):
    ambient, bs, beta = model.fields(*inputs)
    shape = (k["height"], k["width"], 3)
    new = dict(k, medium_rgb=ambient.reshape(shape), medium_bs=bs.reshape(shape), medium_attn=beta.reshape(shape))
    if model.arm == "A2":
        scale = model.scalar.exp()
        new["medium_bs"] = new["medium_bs"] * scale
        new["medium_attn"] = new["medium_attn"] * scale
    result = renderer.rasterize(**new)
    outputs = [result]
    pred = result.rgb
    if model.arm == "A3":
        for scale in (.5, 1.5):
            outputs.append(renderer.rasterize(**dict(new, medium_bs=new["medium_bs"]*scale,
                                                      medium_attn=new["medium_attn"]*scale)))
        pred = pred + model.scalar*(.5*(outputs[1].rgb+outputs[2].rgb)-pred)
    for output in outputs:
        for field in ("j_raw", "accumulation", "depth", "depth_second_moment", "final_transmittance"):
            if not torch.equal(getattr(output, field), getattr(reference, field)):
                raise RuntimeError("Frozen field changed: " + field)
    return pred


def sync_timed_prediction(renderer, k, model, inputs, reference):
    torch.cuda.synchronize()
    initial_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pred = native_prediction(renderer, k, model, inputs, reference)
    torch.cuda.synchronize()
    return pred, dict(native_render_seconds=time.perf_counter()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        incremental_peak_bytes=torch.cuda.max_memory_allocated()-initial_allocated)


@torch.no_grad()
def evaluate(worker, branch, ext, models, directory, smoke=False):
    records = worker._eval_records(branch)
    if smoke:
        records = records[:1]
    rows, checks = [], []
    renderer = branch.pipeline.model.underwater_rasterizer
    for _, view, camera, batch in records:
        k, native, outputs, gt, inputs = capture(worker, branch.pipeline.model, camera, batch)
        original = outputs["pred_image"].float()
        base = worker._taba_rgb_metrics(original.clamp(0, 1), gt)
        target = directory/"renders"/view
        target.mkdir(parents=True, exist_ok=False)
        save_rgb(target/"gt.png", gt)
        save_rgb(target/"A0_underwater.png", original)
        save_rgb(target/"A0_clear.png", native.j_raw)
        save_rgb(target/"A0_gt_error_x5.png", (original-gt).abs()*5)
        np.savez_compressed(target/"original_float.npz", underwater=original.cpu().numpy(),
            clear=native.j_raw.cpu().numpy(), accumulation=native.accumulation.cpu().numpy(), gt=gt.cpu().numpy())
        archived = origin(directory.name, "E0")/"renders/E0"/directory.name/"step_14999"/view/"A2/underwater.png"
        if not archived.is_file():
            raise FileNotFoundError("Required original source PNG: " + str(archived))
        archived_equal = bool(np.array_equal(rgb8(original), np.asarray(Image.open(archived))))
        if not archived_equal:
            raise RuntimeError("Original source PNG mismatch: " + str(archived))
        pixels = torch.arange(0, k["height"]*k["width"], max(1, k["height"]*k["width"]//2048), device="cuda")[:2048]
        data, _ = extract(ext, k, _geometry_bins(k), pixels)
        data.update(full_input=inputs[0][pixels], neutral_input=inputs[1][pixels])
        rows.append(dict(scene=directory.name, view_id=view, seed=0, arm="A0", **base))
        for (seed, arm), medium in models.items():
            pred, timing = sync_timed_prediction(renderer, k, medium, inputs, native)
            error = float((medium(data)-pred.reshape(-1, 3)[pixels]).abs().max())
            if error > 2e-6:
                raise RuntimeError("Final fixed/native mismatch: %s %s %s" % (view, arm, error))
            metrics = worker._taba_rgb_metrics(pred.clamp(0, 1), gt)
            row = dict(scene=directory.name, view_id=view, seed=seed, arm=arm, **metrics,
                delta_psnr_vs_A0=metrics["PSNR"]-base["PSNR"], clear_max_error=0., accumulation_max_error=0.,
                fast_native_max_error=error, scalar=medium.scalar_value(),
                near_black_clear_fraction=float((native.j_raw.max(-1).values<=3/255).float().mean()),
                clear_min=float(native.j_raw.min()), clear_max=float(native.j_raw.max()),
                underwater_min=float(pred.min()), underwater_max=float(pred.max()), **timing)
            rows.append(row)
            prefix = "seed_%d_%s" % (seed, arm)
            save_rgb(target/(prefix+"_underwater.png"), pred)
            save_rgb(target/(prefix+"_difference_x10.png"), (pred-original).abs()*10)
            save_rgb(target/(prefix+"_gt_error_x5.png"), (pred-gt).abs()*5)
            (target/(prefix+"_clear.png")).symlink_to("A0_clear.png")
            np.savez_compressed(target/(prefix+"_float.npz"), underwater=pred.cpu().numpy())
            print(json.dumps(dict(event="eval", **row)), flush=True)
        checks.append(dict(view=view, original_png_identical=archived_equal, clear_sha256=tensor_digest(native.j_raw),
            accumulation_sha256=tensor_digest(native.accumulation), all_native_comparisons_bit_identical=True))
        write_json(target/"metrics.json", [r for r in rows if r["view_id"] == view])
    write_csv(directory/"per_view.csv", rows)
    write_json(directory/"render_preservation.json", checks)
    return rows


def contact_sheets(directory, rows, seed=42):
    views = [r["view_id"] for r in rows if r["arm"] == "A0"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    metrics = {(r["view_id"], r["arm"]): r for r in rows if r["seed"] in (0, seed)}
    for kind, names in (
        ("underwater", [("GT", "gt.png"), ("A0 BND", "A0_underwater.png")]
         + [(arm, "seed_%d_%s_underwater.png" % (seed, arm)) for arm in ARMS]),
        ("clear", [("A0 BND clear", "A0_clear.png")]
         + [(arm+" clear (identical)", "seed_%d_%s_clear.png" % (seed, arm)) for arm in ARMS]),
        ("differences", [("A0 error vs GT x5", "A0_gt_error_x5.png")]
         + [(arm+" change vs A0 x10", "seed_%d_%s_difference_x10.png" % (seed, arm)) for arm in ARMS]),
    ):
        tile_w, tile_h = 460, 370
        canvas = Image.new("RGB", (len(names)*tile_w, 55+len(views)*tile_h), "#181818")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), directory.name+" | "+kind+" | fixed seed "+str(seed), font=font, fill="white")
        for row_index, view in enumerate(views):
            for column, (label, filename) in enumerate(names):
                left, top = column*tile_w, 55+row_index*tile_h
                draw.text((left+8, top+3), view+"  "+label, font=font, fill="white")
                if kind == "underwater" and label != "GT":
                    arm = "A0" if label == "A0 BND" else label
                    m = metrics[(view, arm)]
                    draw.text((left+8, top+27), "PSNR %.4f  LPIPS %.4f" % (m["PSNR"], m["LPIPS"]), font=small, fill="#c7c7c7")
                with Image.open(directory/"renders"/view/filename) as source:
                    image = source.convert("RGB")
                    image.thumbnail((tile_w-12, tile_h-65), Image.Resampling.LANCZOS)
                canvas.paste(image, (left+(tile_w-image.width)//2, top+55))
        canvas.save(directory/(kind+"_contact_sheet.jpg"), quality=96)


def run_scene(args):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == args.gpu
    directory = args.output_root/args.scene
    directory.mkdir(parents=True, exist_ok=False)
    worker = configure_worker("E0")
    worker.ACTIVE_SOURCE_ROOT, worker.ACTIVE_SOURCE_CONFIG, _ = source_assets(args.scene)
    runtime = worker._runtime(args.scene, args.gpu)
    branch = worker._new_branch(args.scene)
    checkpoint = origin(args.scene, "E0")/args.scene/"E0/checkpoints/step-000014999.ckpt"
    digest = sha256(checkpoint)
    try:
        payload = worker._load_snapshot(branch, args.scene, checkpoint)
        source = branch.pipeline.model
        assert source.config.intrinsic_color_parameterization == "bounded_sh3"
        assert source.config.camera_medium_observability_enabled
        assert float(source.config.camera_medium_observability_strength) == 1.
        assert not source.config.taba_enabled
        before = worker._model_state_hash(source)
        source.requires_grad_(False)
        ext = extension()
        data = cache_training(worker, branch, ext, directory, args.pixels_per_view, args.smoke)
        models = {}
        for seed in args.seeds:
            for arm in ARMS:
                models[(seed, arm)] = train_arm(source, payload, data, arm, seed, args.steps, directory)
        training = [read_json(directory/("seed_%d" % seed)/arm/"training.json") for seed in args.seeds for arm in ARMS]
        for seed in args.seeds:
            selected = [r for r in training if r["seed"] == seed]
            assert len({r["sampling_sha256"] for r in selected}) == 1
            assert len({r["initial_medium_hash"] for r in selected}) == 1
        del data
        rows = evaluate(worker, branch, ext, models, directory, args.smoke)
        contact_sheets(directory, rows, args.seeds[0])
        after = worker._model_state_hash(source)
        assert before == after and digest == sha256(checkpoint)
        write_json(directory/"result.json", dict(complete=True, scene=args.scene, steps=args.steps, seeds=args.seeds,
            runtime=runtime, rows=rows, training=training, checkpoint=str(checkpoint), checkpoint_sha256=digest,
            model_sha256=before, source_unchanged=True, gaussians_frozen=True, all_clear_bit_identical=True,
            formal=not args.smoke))
    except Exception as exc:
        write_json(directory/"failure.json", dict(error=repr(exc)))
        raise
    finally:
        worker._release(branch)


def summarize(directory):
    results = [read_json(directory/scene/"result.json") for scene in SCENES]
    rows = [r for result in results for r in result["rows"]]
    seeds = results[0]["seeds"]
    means = []
    for scene in SCENES:
        for seed, arm in [(0, "A0")]+[(s, a) for s in seeds for a in ARMS]:
            chosen = [r for r in rows if r["scene"] == scene and r["seed"] == seed and r["arm"] == arm]
            means.append(dict(scene=scene, seed=seed, arm=arm, views=len(chosen),
                **{metric: statistics.mean(r[metric] for r in chosen) for metric in METRICS}))
    aggregated = []
    for scene in SCENES:
        for arm in ("A0",)+ARMS:
            chosen = [r for r in means if r["scene"] == scene and r["arm"] == arm]
            aggregated.append(dict(scene=scene, arm=arm,
                **{m: statistics.mean(r[m] for r in chosen) for m in METRICS},
                psnr_seed_std=statistics.stdev(r["PSNR"] for r in chosen) if len(chosen)>1 else 0.))
    macro = {arm: {m: statistics.mean(r[m] for r in aggregated if r["arm"] == arm) for m in METRICS}
             for arm in ("A0",)+ARMS}
    per_seed = []
    for seed in seeds:
        values = {arm: statistics.mean(r["PSNR"] for r in means if r["arm"] == arm and r["seed"] == seed) for arm in ARMS}
        per_seed.append(dict(seed=seed, A3_minus_A1=values["A3"]-values["A1"], A3_minus_A2=values["A3"]-values["A2"]))
    minimum = min(next(r["PSNR"] for r in aggregated if r["scene"] == s and r["arm"] == "A3")
                  - next(r["PSNR"] for r in aggregated if r["scene"] == s and r["arm"] == "A0") for s in SCENES)
    gate = dict(mean_A3_minus_A1_at_least_003=macro["A3"]["PSNR"]-macro["A1"]["PSNR"]>=.03,
        mean_A3_minus_A2_at_least_003=macro["A3"]["PSNR"]-macro["A2"]["PSNR"]>=.03,
        positive_each_seed=all(r["A3_minus_A1"]>0 and r["A3_minus_A2"]>0 for r in per_seed),
        scene_safety=minimum>=-.05,
        lpips_nonworse=all(macro["A3"]["LPIPS"]<=macro[a]["LPIPS"] for a in ("A0", "A1", "A2")),
        clear_preserved=all(r["all_clear_bit_identical"] and r["source_unchanged"] for r in results))
    passed = all(gate.values())
    write_csv(directory/"per_view.csv", rows)
    write_csv(directory/"scene_seed_means.csv", means)
    write_csv(directory/"scene_means.csv", aggregated)
    write_json(directory/"summary.json", dict(complete=True, macro=macro, scene_means=aggregated,
        seed_deltas=per_seed, gates=gate, promotion_gate=passed,
        stage2="eligible" if passed else "not_triggered", stage3="pending_stage2" if passed else "not_triggered"))
    lines = ["# BND-Preserving Transport Recovery Results", "", "Native renderer evaluation; all existing test views; common source per scene.",
             "", "| Scene | Arm | PSNR | SSIM | LPIPS | MSE | PSNR seed SD |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in aggregated:
        lines.append("| {scene} | {arm} | {PSNR:.6f} | {SSIM:.6f} | {LPIPS:.6f} | {MSE:.9f} | {psnr_seed_std:.6f} |".format(**row))
    lines += ["", "| Mean | PSNR | SSIM | LPIPS | MSE |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm, m in macro.items():
        lines.append("| %s | %.6f | %.6f | %.6f | %.9f |" % (arm, m["PSNR"], m["SSIM"], m["LPIPS"], m["MSE"]))
    lines += ["", "Promotion gate: **%s**." % ("PASS" if passed else "FAIL"), "", "```json", json.dumps(gate, indent=2), "```", "",
              "All native clear, accumulation, depth and final-transmittance checks were bit-identical. No clear accuracy claim is made.",
              "", "Seeds vary medium continuation minibatches, not source Gaussian training. Training uses sampled MSE, not the original full-image loss.",
              "", "## Visual Comparisons", ""]
    for scene in SCENES:
        lines.append("- %s: [underwater](%s/underwater_contact_sheet.jpg), [clear](%s/clear_contact_sheet.jpg), [differences](%s/differences_contact_sheet.jpg)." % (scene, scene, scene, scene))
    (directory/"report.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    print(json.dumps(dict(event="summary", macro=macro, gate=passed, seed_deltas=per_seed)), flush=True)


def launch(args):
    args.output_root.mkdir(parents=True, exist_ok=False)
    active = gpu_processes()
    if any(active.values()):
        raise RuntimeError("Authorized GPUs occupied: " + repr(active))
    artifacts = [Path(__file__).resolve(), ROOT/"water_splatting/rendering/optical_thickness_distribution.py",
                 ROOT/"research_notes/BND_PRESERVING_TRANSPORT_RECOVERY_2026-09-12.md"]
    write_json(args.output_root/"registration.json", dict(steps=args.steps, seeds=args.seeds,
        pixels_per_training_camera=args.pixels_per_view, diagnostic_pixels_per_training_camera=args.pixels_per_view,
        batch=1024, source_step=14999, source_training_seed=42, delta=.5, original_renderer_unchanged=True,
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in artifacts},
        medium_optimizer="source Adam state and fixed checkpoint learning rate", scalar_lr=.005,
        loss="sampled RGB MSE", test_used_for_optimization_or_checkpoint_selection=False))
    jobs = []
    for scene, gpu in zip(SCENES, ("6", "7", "8", "9")):
        log = (args.output_root/(scene+".log")).open("x")
        command = [sys.executable, str(Path(__file__).resolve()), "--scene", scene, "--gpu", gpu,
            "--output-root", str(args.output_root), "--steps", str(args.steps),
            "--pixels-per-view", str(args.pixels_per_view), "--seeds"] + [str(s) for s in args.seeds]
        process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((scene, process, log))
    try:
        while any(process.poll() is None for _, process, _ in jobs):
            write_json(args.output_root/"queue_status.json", {s: p.poll() for s, p, _ in jobs})
            time.sleep(5)
        statuses = {s: p.returncode for s, p, _ in jobs}
        write_json(args.output_root/"queue_status.json", statuses)
        if any(statuses.values()):
            raise RuntimeError("Scene failure: " + repr(statuses))
    finally:
        for _, process, log in jobs:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
    summarize(args.output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--gpu", choices=("6", "7", "8", "9"))
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--pixels-per-view", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 3407])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    check_environment()
    if args.summarize:
        summarize(args.output_root)
    elif args.scene:
        run_scene(args)
    else:
        launch(args)
