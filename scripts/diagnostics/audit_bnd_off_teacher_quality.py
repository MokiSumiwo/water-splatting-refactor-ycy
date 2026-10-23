#!/usr/bin/env python3
"""Read-only audit of historical BND-OFF teacher quality versus current BND-ON."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("JapaneseGradens-RedSea", "Panama", "Curasao", "IUI3-RedSea")


def psnr(mse):
    return -10.0 * math.log10(max(float(mse), 1e-30))


def rgb8(x):
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def load_scene(experiment, scene):
    result = json.loads((experiment / scene / "result.json").read_text())
    teacher_result = json.loads((experiment / "historical_bnd_off" / scene / "result.json").read_text())
    teacher_rows = {r["view_id"]: r for r in teacher_result["rows"]}
    rows = []
    for view in [r["view_id"] for r in result["rows"] if r["arm"] == "A0"]:
        root = experiment / scene / "renders" / view
        teacher_root = experiment / "historical_bnd_off" / scene / view
        with np.load(root / "original_float.npz") as data:
            on, gt = data["underwater"].astype(np.float64), data["gt"].astype(np.float64)
        with np.load(teacher_root / "float.npz") as data:
            off = data["underwater"].astype(np.float64)
        assert on.shape == off.shape == gt.shape
        e_on, e_off = on - gt, off - gt
        mse_on, mse_off = np.mean(e_on * e_on), np.mean(e_off * e_off)
        # Oracle convex blend is an upper bound that uses GT only for diagnosis;
        # it is not a proposed training result or a valid test-time selection rule.
        direction = off - on
        denom = np.sum(direction * direction, axis=-1, keepdims=True)
        alpha_px = np.divide(np.sum((gt - on) * direction, axis=-1, keepdims=True), denom,
                             out=np.zeros_like(denom), where=denom > 1e-20)
        alpha_px = np.clip(alpha_px, 0, 1)
        oracle = on + alpha_px * direction
        mse_oracle_px = np.mean((oracle - gt) ** 2)
        numerator = np.sum((gt - on) * direction)
        denominator = np.sum(direction * direction)
        alpha_global = float(np.clip(numerator / max(denominator, 1e-20), 0, 1))
        blend = on + alpha_global * direction
        mse_blend = np.mean((blend - gt) ** 2)
        rows.append(dict(scene=scene, view_id=view, mse_bnd_on=float(mse_on), psnr_bnd_on=psnr(mse_on),
            mse_bnd_off=float(mse_off), psnr_bnd_off=psnr(mse_off), teacher_better=mse_off < mse_on,
            teacher_delta_psnr=psnr(mse_off)-psnr(mse_on), teacher_abs_rgb_mae=float(np.mean(np.abs(off-on))),
            oracle_pixel_blend_psnr=psnr(mse_oracle_px), oracle_global_alpha=alpha_global,
            oracle_global_blend_psnr=psnr(mse_blend), oracle_gain_vs_on=psnr(mse_blend)-psnr(mse_on),
            oracle_pixel_gain_vs_on=psnr(mse_oracle_px)-psnr(mse_on),
            on_above_one_fraction=float(np.mean(on > 1)), off_above_one_fraction=float(np.mean(off > 1))))
        np.savez_compressed(root / "teacher_diagnostic_float.npz", bnd_on=on, bnd_off=off, gt=gt, oracle=oracle)
    return rows


def write_csv(path, rows):
    keys = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def sheet(experiment, rows):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    by = {(r["scene"], r["view_id"]): r for r in rows}
    for scene in SCENES:
        views = [r["view_id"] for r in rows if r["scene"] == scene]
        labels = ("GT", "BND ON", "BND OFF teacher", "oracle global blend")
        tw, th, header = 460, 370, 62
        canvas = Image.new("RGB", (tw*len(labels), header+th*len(views)), "#181818")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), scene+" | teacher quality audit", fill="white", font=font)
        draw.text((10, 35), "Oracle uses GT only as an upper-bound diagnostic; no claim of usable test-time fusion.", fill="#cccccc", font=small)
        for i, view in enumerate(views):
            root = experiment / scene / "renders" / view
            paths = [root/"gt.png", root/"A0_underwater.png", experiment/"historical_bnd_off"/scene/view/"underwater.png"]
            with np.load(root / "teacher_diagnostic_float.npz") as d:
                oracle = d["bnd_on"] + by[(scene,view)]["oracle_global_alpha"]*(d["bnd_off"]-d["bnd_on"])
            oracle_path = root / "oracle_global_blend.png"; Image.fromarray(rgb8(oracle)).save(oracle_path)
            paths.append(oracle_path)
            for j, (label, path) in enumerate(zip(labels, paths)):
                x, y = j*tw, header+i*th
                m = by[(scene,view)]
                metric = "" if label == "GT" else ("PSNR %.3f" % (m["psnr_bnd_on"] if label == "BND ON" else m["psnr_bnd_off"] if label.startswith("BND OFF") else m["oracle_global_blend_psnr"]))
                draw.text((x+8,y+3), view+" | "+label, fill="white", font=font)
                draw.text((x+8,y+27), metric, fill="#cccccc", font=small)
                with Image.open(path) as src:
                    im=src.convert("RGB"); im.thumbnail((tw-12, th-65), Image.Resampling.LANCZOS)
                canvas.paste(im,(x+(tw-im.width)//2,y+54))
        canvas.save(experiment/scene/"teacher_quality_contact_sheet.jpg", quality=96)


def main(experiment):
    rows = [r for s in SCENES for r in load_scene(experiment, s)]
    write_csv(experiment/"teacher_quality_per_view.csv", rows)
    scene_rows=[]
    for s in SCENES:
        chosen=[r for r in rows if r["scene"]==s]
        scene_rows.append(dict(scene=s, views=len(chosen),
            bnd_on_psnr=float(statistics.mean(r["psnr_bnd_on"] for r in chosen)),
            bnd_off_psnr=float(statistics.mean(r["psnr_bnd_off"] for r in chosen)),
            teacher_better_fraction=float(np.mean([bool(r["teacher_better"]) for r in chosen])),
            oracle_global_blend_psnr=float(statistics.mean(r["oracle_global_blend_psnr"] for r in chosen)),
            oracle_gain_vs_on=float(statistics.mean(r["oracle_gain_vs_on"] for r in chosen)),
            oracle_pixel_blend_psnr=float(statistics.mean(r["oracle_pixel_blend_psnr"] for r in chosen)),
            oracle_pixel_gain_vs_on=float(statistics.mean(r["oracle_pixel_gain_vs_on"] for r in chosen))))
    macro={k:float(statistics.mean(r[k] for r in scene_rows)) for k in scene_rows[0] if k not in ("scene","views")}
    payload=dict(complete=True, test_views=len(rows), scene_rows=scene_rows, macro=macro,
        teacher_is_historical_reference=True, strict_paired_training_available=False,
        oracle_blends_use_gt_for_diagnosis_only=True,
        decision="teacher_is_not_a_uniform_oracle" if macro["teacher_better_fraction"] < .75 else "teacher_merits_trust_region_screen")
    (experiment/"teacher_quality_summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    sheet(experiment, rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--experiment", type=Path, required=True)
    main(p.parse_args().experiment.resolve())
