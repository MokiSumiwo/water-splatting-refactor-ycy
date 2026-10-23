#!/usr/bin/env python3
"""Re-render historical BND OFF references without fitting or changing assets."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.diagnostics.audit_dvs_frozen_visibility import SCENES, rgb8, save_rgb, write_csv
from scripts.experiments.run_crcc_bnd_on import configure_worker
from scripts.experiments.run_taba_bnd_on import check_environment, gpu_processes, read_json, sha256, write_json
from scripts.experiments.run_world_medium_screen import capture_view

HISTORY = ROOT / "outputs/legacy_ocmc_cica_no_auxiliary_20260905"
RENDERS = ROOT / "renders/legacy_ocmc_cica_no_auxiliary_20260905"
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")


@torch.no_grad()
def render_scene(scene, gpu, experiment):
    target = experiment / "historical_bnd_off" / scene
    target.mkdir(parents=True, exist_ok=False)
    worker = configure_worker("E0")
    worker.ACTIVE_PARAMETERIZATION = "legacy"
    worker.ACTIVE_SOURCE_CONFIG = HISTORY / scene / "source_config.yml"
    worker.ACTIVE_SOURCE_ROOT = HISTORY
    worker._runtime(scene, gpu)
    branch = worker._new_branch(scene)
    checkpoint = HISTORY / scene / "checkpoints/step-000014999.ckpt"
    digest = sha256(checkpoint)
    rows = []
    try:
        worker._load_snapshot(branch, scene, checkpoint)
        model = branch.pipeline.model.requires_grad_(False)
        before = worker._model_state_hash(model)
        assert model.config.intrinsic_color_parameterization == "legacy"
        for _, view, camera, batch in worker._eval_records(branch):
            torch.cuda.synchronize()
            started = time.perf_counter()
            _, native, outputs, gt = capture_view(worker, model, camera, batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            pred = outputs["pred_image"].float()
            original = RENDERS / scene / "step_14999" / view / "A2/underwater.png"
            if not original.is_file():
                raise FileNotFoundError(original)
            with Image.open(original) as image:
                assert np.array_equal(rgb8(pred), np.asarray(image)), str(original)
            with np.load(experiment / scene / "renders" / view / "original_float.npz") as baseline:
                assert np.array_equal(gt.cpu().numpy(), baseline["gt"]), "Unpaired reference GT"
            directory = target / view
            directory.mkdir(exist_ok=False)
            save_rgb(directory / "underwater.png", pred)
            save_rgb(directory / "clear_clamp01.png", native.j_raw)
            np.savez_compressed(directory / "float.npz", underwater=pred.cpu().numpy(),
                clear=native.j_raw.cpu().numpy(), accumulation=native.accumulation.cpu().numpy())
            row = dict(scene=scene, view_id=view, **worker._taba_rgb_metrics(pred.clamp(0, 1), gt),
                native_render_seconds=elapsed, original_png_identical=True,
                clear_min=float(native.j_raw.min()), clear_max=float(native.j_raw.max()),
                clear_channel_above_one_fraction=float((native.j_raw > 1).float().mean()),
                checkpoint_sha256=digest, historical_reference_not_paired_training=True)
            rows.append(row)
            print(json.dumps(dict(event="historical_reference", **row)), flush=True)
        assert before == worker._model_state_hash(model) and digest == sha256(checkpoint)
        write_json(target / "result.json", dict(complete=True, source_unchanged=True,
            checkpoint=str(checkpoint), sha256=digest, rows=rows,
            model_sha256=before, historical_reference_not_paired_training=True))
        write_csv(target / "per_view.csv", rows)
    finally:
        worker._release(branch)


def sheet(experiment, scene, reference, seed=42):
    result = read_json(experiment / scene / "result.json")
    views = [r["view_id"] for r in result["rows"] if r["arm"] == "A0"]
    metrics = {(r["view_id"], r["arm"]): r for r in result["rows"] if r["seed"] in (0, seed)}
    refs = {r["view_id"]: r for r in reference}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    tile_w, tile_h, header = 440, 370, 80
    for kind in ("underwater", "clear"):
        labels = ["GT", "BND OFF historical", "A0 BND", "A1 medium refit", "A2 optical scale", "A3 thickness mix"]
        if kind == "clear":
            labels = labels[1:]
        canvas = Image.new("RGB", (tile_w * len(labels), header + len(views)*tile_h), "#181818")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), scene+" | "+kind+" | continuation seed "+str(seed), fill="white", font=font)
        draw.text((12, 36), "BND OFF is a historical reference, not matched training. All clear displays use clamp [0,1].",
                  fill="#dddddd", font=small)
        for row_index, view in enumerate(views):
            local = experiment / scene / "renders" / view
            ref = experiment / "historical_bnd_off" / scene / view
            if kind == "underwater":
                paths = [local/"gt.png", ref/"underwater.png", local/"A0_underwater.png"]
                paths += [local/("seed_%d_%s_underwater.png" % (seed, a)) for a in ("A1", "A2", "A3")]
            else:
                paths = [ref/"clear_clamp01.png", local/"A0_clear.png"]
                paths += [local/("seed_%d_%s_clear.png" % (seed, a)) for a in ("A1", "A2", "A3")]
            for column, (label, path) in enumerate(zip(labels, paths)):
                x, y = column*tile_w, header+row_index*tile_h
                draw.text((x+8, y+3), view+" | "+label, fill="white", font=font)
                if kind == "underwater" and label != "GT":
                    m = refs[view] if label.startswith("BND OFF") else metrics[(view, label[:2])]
                    draw.text((x+8, y+29), "PSNR %.4f | LPIPS %.4f" % (m["PSNR"], m["LPIPS"]), fill="#cccccc", font=small)
                elif kind == "clear" and not label.startswith("BND OFF"):
                    draw.text((x+8, y+29), "Exactly equal to A0 clear", fill="#cccccc", font=small)
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((tile_w-12, tile_h-65), Image.Resampling.LANCZOS)
                canvas.paste(image, (x+(tile_w-image.width)//2, y+56))
        destination = experiment / scene / (kind+"_with_bnd_off.jpg")
        if destination.exists():
            raise FileExistsError(destination)
        canvas.save(destination, quality=96)


def summarize(experiment):
    rows, means = [], []
    for scene in SCENES:
        result = read_json(experiment / "historical_bnd_off" / scene / "result.json")
        rows += result["rows"]
        means.append(dict(scene=scene, **{m: statistics.mean(r[m] for r in result["rows"]) for m in METRICS}))
        sheet(experiment, scene, result["rows"])
    write_csv(experiment / "historical_bnd_off/per_view.csv", rows)
    write_csv(experiment / "historical_bnd_off/scene_means.csv", means)
    macro = {m: statistics.mean(r[m] for r in means) for m in METRICS}
    write_json(experiment / "historical_bnd_off/summary.json", dict(complete=True, macro=macro,
        scene_means=means, test_views=len(rows), historical_reference_not_paired_training=True,
        source="legacy_ocmc_cica_no_auxiliary_20260905", local_metrics=True,
        clear_display="raw clamp [0,1], no exposure or white-balance change"))
    print(json.dumps(dict(event="historical_summary", macro=macro)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--gpu", choices=("6", "7", "8", "9"), default="6")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    check_environment()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == args.gpu
    if args.summarize:
        summarize(args.experiment.resolve())
    else:
        if gpu_processes()[args.gpu]:
            raise RuntimeError("Requested GPU occupied")
        for scene in (args.scene,) if args.scene else SCENES:
            render_scene(scene, args.gpu, args.experiment.resolve())
        if not args.scene:
            summarize(args.experiment.resolve())
