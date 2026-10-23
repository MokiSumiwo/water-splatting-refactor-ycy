#!/usr/bin/env python3
"""Audit completed paired runs and retain all-view comparisons without rerendering."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
ARMS = ("R0C0", "R1C0", "R0C1", "R1C1")
METRICS = ("PSNR", "SSIM", "LPIPS")
ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def image(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def sheet(rows, dest, tile=(420, 280)):
    width, height = tile
    canvas = Image.new("RGB", (len(rows[0]) * width, len(rows) * (height + 50)), "#17191b")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    for i, row in enumerate(rows):
        for j, (im, label) in enumerate(row):
            fitted = ImageOps.contain(im, tile, method=Image.Resampling.LANCZOS)
            x, y = j * width, i * (height + 50)
            canvas.paste(fitted, (x + (width - fitted.width) // 2, y + 50 + (height - fitted.height) // 2))
            draw.multiline_text((x + 8, y + 5), label, fill="white", font=font, spacing=3)
    canvas.save(dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    vis = out / "visualizations"
    vis.mkdir()
    metrics, per_view, effects, checks, drift, resource, clear_rows = [], [], [], [], [], [], []
    provenance, source_hashes = {}, {}
    overview_rgb, overview_clear, gallery = [], [], []

    for scene_index, scene in enumerate(SCENES):
        paths = {a: root / scene / a for a in ARMS}
        starts = {a: read_json(p / "start_state_equivalence.json") for a, p in paths.items()}
        manifests = {a: read_json(p / "experiment_manifest.json") for a, p in paths.items()}
        views, renders, scene_metrics, training = {}, {}, {}, {}
        assertions = {}
        assertions["all_exit_zero"] = read_json(root / scene / "status.json") == dict.fromkeys(ARMS, 0)
        assertions["identical_start_state_except_arm"] = all(
            {k: v for k, v in s.items() if k != "arm"} ==
            {k: v for k, v in starts["R0C0"].items() if k != "arm"} for s in starts.values())
        for name in ("camera_sequence.csv", "cica_camera_bank.json", "mdrr_partner_mapping.json", "source_config.yml"):
            assertions["identical_" + name] = len({sha(p / name) for p in paths.values()}) == 1
        for arm, path in paths.items():
            start, manifest = starts[arm], manifests[arm]
            final = read_json(path / "final_summary.json")
            route = read_json(path / "gradient_routing_audit.json")
            decomp = read_json(path / "decomposition_safety.json")
            expected_r, expected_c = ("fixed" if arm[1] == "1" else "legacy"), ("exact_fsb" if arm[3] == "1" else "legacy")
            assert manifest["renderer_backward_mode"] == expected_r
            assert manifest["module_schedule"]["CICA"]["derivative_mode"] == expected_c
            assert manifest["renderer_backend"]["forward_backend"] == "water_splatting.csrc"
            assert manifest["runtime"]["physical_gpu"] == str(6 + scene_index)
            assert manifest["runtime"]["cuda_visible_devices"] == str(6 + scene_index)
            assert manifest["runtime"]["python"] == "/opt/anaconda3/envs/water_splatting/bin/python"
            assert start["START_STATE_EQUIVALENCE"] and route["GRADIENT_ROUTING_AUDIT"]
            assert final["completed_updates"] == 11999 and final["first_step"] == 3001 and final["final_step"] == 14999
            assert final["ocmc_projector_unchanged"]
            source = Path(manifest["source_checkpoint"])
            if str(source) not in source_hashes:
                source_hashes[str(source)] = sha(source)
            assert source_hashes[str(source)] == manifest["source_checkpoint_sha256"]
            rows = read_csv(path / "evaluation_metrics.csv")
            assert len(rows) == 1 and rows[0]["absolute_step"] == "14999"
            scene_metrics[arm] = {m: float(rows[0][m]) for m in METRICS}
            views[arm] = {v["view_id"]: v for v in read_csv(path / "per_view_metrics.csv")}
            assert len(views[arm]) == int(rows[0]["view_count"])
            assert len(views[arm]) == (4 if scene == "IUI3-RedSea" else 3)
            for m in METRICS:
                assert all(math.isfinite(float(v[m])) for v in views[arm].values())
                assert abs(mean(float(v[m]) for v in views[arm].values()) - scene_metrics[arm][m]) < 1e-9
            training[arm] = {int(r["absolute_step"]): r for r in read_csv(path / f"training_summary_{arm}.csv")}
            renders[arm] = {}
            for row in read_csv(path / "render_manifest.csv"):
                render_path = Path(row["path"])
                assert render_path.is_file()
                renders[arm].setdefault(row["view_id"], {})[row["output_type"]] = render_path
                provenance[str(render_path)] = sha(render_path)
            assert set(renders[arm]) == set(views[arm])
            modules = read_csv(path / "module_statistics.csv")
            assert [int(r["absolute_step"]) for r in modules if r["module"] == "CICA" and r["event"] == "refresh"] == [10000, 12000, 14000]
            assert len([r for r in modules if r["event"] == "calibration"]) == 1
            resource.append(dict(scene=scene, arm=arm, training_seconds=final["training_wall_seconds"],
                                 final_gaussians=final["final_population"], cica_lambda=final["resolved_lambda_CICA"],
                                 render_ms=1000 * mean(float(v["render_seconds"]) for v in views[arm].values()),
                                 render_peak_gib=max(float(v["render_peak_cuda_bytes"]) for v in views[arm].values()) / 2**30,
                                 P_J_gt_1=decomp["rows"][0]["P_J_gt_1"], checkpoint_retained=final["checkpoint_retained"]))
            for name in ("experiment_manifest.json", "final_summary.json", "evaluation_metrics.csv", "per_view_metrics.csv", "start_state_equivalence.json", "gradient_routing_audit.json", "module_statistics.csv"):
                provenance[str(path / name)] = sha(path / name)
        assertions["identical_test_views"] = all(set(v) == set(views["R0C0"]) for v in views.values())
        checks.append(dict(scene=scene, **assertions))
        assert all(assertions.values())
        for arm in ARMS:
            metrics.append(dict(scene=scene, arm=arm, **scene_metrics[arm],
                                **{"delta_" + m: scene_metrics[arm][m] - scene_metrics["R0C0"][m] for m in METRICS}))
        for m in METRICS:
            a, b, c, d = (scene_metrics[arm][m] for arm in ARMS)
            effects.append(dict(scene=scene, metric=m, renderer_at_C0=b-a, renderer_at_C1=d-c,
                                cica_at_R0=c-a, cica_at_R1=d-b, renderer_main=((b-a)+(d-c))/2,
                                cica_main=((c-a)+(d-b))/2, interaction=d-b-c+a, combined=d-a))
        for a, b in (("R0C0", "R0C1"), ("R1C0", "R1C1")):
            for step in sorted(set(training[a]) & set(training[b])):
                if step >= 10000:
                    continue
                x, y = training[a][step], training[b][step]
                assert x["camera_name"] == y["camera_name"]
                drift.append(dict(scene=scene, pair=a + "/" + b, step=step,
                                  training_PSNR_delta=float(y["PSNR"]) - float(x["PSNR"]),
                                  gaussian_count_delta=int(y["gaussian_count"]) - int(x["gaussian_count"])))

        rgb_rows, clear_sheets, diff_rows = [], [], []
        for view in sorted(views["R0C0"]):
            rgb_row, clear_row, diff_row = [], [], []
            gt_paths = [renders[a][view]["underwater"].parent / "gt.png" for a in ARMS]
            assert len({sha(p) for p in gt_paths}) == 1
            gt = image(gt_paths[0])
            rgb_row.append((gt, scene + " / " + view + "\nUnderwater GT"))
            base_clear = np.asarray(image(renders["R0C0"][view]["clear_native"]), dtype=np.float32) / 255
            base_rgb = np.asarray(image(renders["R0C0"][view]["underwater"]), dtype=np.float32) / 255
            for arm in ARMS:
                entry = views[arm][view]
                rgb = image(renders[arm][view]["underwater"])
                clear = image(renders[arm][view]["clear_native"])
                assert rgb.size == gt.size == clear.size
                rgb_row.append((rgb, f'{arm} / {view}\nPSNR {float(entry["PSNR"]):.4f} | LPIPS {float(entry["LPIPS"]):.5f}'))
                clear_row.append((clear, scene + " / " + view + "\n" + arm + " clear (no GT)"))
                clear_array = np.asarray(clear, dtype=np.float32) / 255
                clear_rows.append(dict(scene=scene, view_id=view, arm=arm,
                                       clear_png_mean=float(clear_array.mean()),
                                       clear_png_dark_maxrgb_lt_0p02=float((clear_array.max(-1) < .02).mean()),
                                       clear_png_MAE_to_R0C0=float(np.abs(clear_array-base_clear).mean()),
                                       **{k: float(v) for k, v in entry.items() if k.startswith("clear_raw_")}))
                per_view.append(dict(scene=scene, view_id=view, arm=arm,
                                     **{m: float(entry[m]) for m in METRICS},
                                     **{"delta_" + m: float(entry[m])-float(views["R0C0"][view][m]) for m in METRICS}))
                if arm != "R0C0":
                    diff = np.abs(np.asarray(rgb, dtype=np.float32)/255 - base_rgb)
                    diff_row.append((Image.fromarray(np.round(np.clip(diff * 10, 0, 1) * 255).astype("uint8")),
                                     scene + " / " + view + "\nabs(" + arm + " - R0C0) x10"))
            rgb_rows.append(rgb_row)
            clear_sheets.append(clear_row)
            diff_rows.append(diff_row)
        overview_rgb.append(rgb_rows[0])
        overview_clear.append(clear_sheets[0])
        for kind, rows in (("underwater", rgb_rows), ("clear", clear_sheets), ("underwater_difference_x10", diff_rows)):
            filename = scene + "_" + kind + ".png"
            sheet(rows, vis / filename)
            gallery.append((scene + " / " + kind + " / all test views", filename))

    for arm in ARMS:
        rows = [r for r in metrics if r["arm"] == arm]
        metrics.append(dict(scene="MACRO_MEAN", arm=arm, **{k: mean(r[k] for r in rows) for k in rows[0] if k not in ("scene", "arm")}))
    for m in METRICS:
        rows = [r for r in effects if r["metric"] == m]
        effects.append(dict(scene="MACRO_MEAN", metric=m, **{k: mean(r[k] for r in rows) for k in rows[0] if k not in ("scene", "metric")}))
    for filename, rows in (("metrics.csv", metrics), ("per_view_metrics.csv", per_view), ("factorial_effects.csv", effects),
                           ("resources.csv", resource), ("clear_diagnostics.csv", clear_rows), ("pre_cica_trajectory_drift.csv", drift)):
        write_csv(out / filename, rows)
    (out / "pairing_audit.json").write_text(json.dumps({"checks": checks, "all_setup_checks_pass": True,
        "bitwise_training_reproducibility": False, "source_hashes": source_hashes,
        "endpoint_checkpoints_retained": False,
        "scope": "setup matched; trajectory differences exist before CICA; single-seed outcomes are descriptive"}, indent=2) + "\n")
    provenance[str(Path(__file__).resolve())] = sha(Path(__file__))
    (out / "provenance_sha256.json").write_text(json.dumps(provenance, indent=2) + "\n")
    sheet(overview_rgb, vis / "four_scene_underwater.png")
    sheet(overview_clear, vis / "four_scene_clear.png")
    html_rows = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>FSB Four-Way Comparison</title>",
                 "<style>body{font:16px sans-serif;margin:24px;background:#181a1c;color:#eee}img{max-width:100%;height:auto}a{color:#8bd7ea}section{margin:32px 0}</style>",
                 "<h1>FSB Renderer / CICA Four-Way Comparison</h1><p>R0: historical backward; R1: alpha-cap corrected backward. C0: historical CICA; C1: exact FSB CICA derivative.</p>",
                 "<p>All 13 test views. Fixed image scale; no exposure or color adjustment. Clear has no ground truth. Difference images use the same 10x multiplier.</p>"]
    for label, filename in gallery:
        safe = html.escape(filename)
        html_rows.append(f"<section><h2>{html.escape(label)}</h2><a href='visualizations/{safe}'><img loading='lazy' src='visualizations/{safe}' alt='{html.escape(label)}'></a></section>")
    html_rows.append("</html>")
    (out / "index.html").write_text("\n".join(html_rows))
    report = ["# FSB Renderer / CICA Four-Way Results, 2026-09-16", "",
        "All 16 v4 runs completed before this post-run audit. This report reuses their original float-tensor metrics and PNG renders; no training was repeated or endpoint selected by test score.", "",
        "## Protocol Recovered From Run Manifests", "",
        "- R0C0: historical backward + historical CICA; R1C0: corrected backward + historical CICA; R0C1: historical backward + exact FSB CICA; R1C1: both corrections.",
        "- Four scenes on physical GPUs 6, 7, 8, 9 respectively; conda water_splatting only.",
        "- FSB alpha=1, SH3, M1+OCMC+CICA, auxiliary appearance OFF. Identical step-3000 source, optimizer/scheduler state, seed 42 and camera sequence within each scene. Continue 11999 updates to step 14999.",
        "- CICA starts at 10000 and refreshes at 10000/12000/14000. Each arm retains the historical training-only 10% DC-gradient calibration policy, so resolved lambda is not held numerically equal. Effects include this calibration response.",
        "- All 13 held-out views evaluated in each arm (3/4/3/3); no scene/view removed. Macro results weight scenes equally. PSNR/SSIM higher is better; LPIPS lower is better.", "",
        "## Metric Results", "", "| Scene | Arm | PSNR | SSIM | LPIPS | Delta PSNR | Delta SSIM | Delta LPIPS |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in metrics:
        report.append(f'| {r["scene"]} | {r["arm"]} | {r["PSNR"]:.6f} | {r["SSIM"]:.6f} | {r["LPIPS"]:.6f} | {r["delta_PSNR"]:+.6f} | {r["delta_SSIM"]:+.6f} | {r["delta_LPIPS"]:+.6f} |')
    report += ["", "## Factorial Contrasts", "",
        "Renderer main effect = ((R1C0-R0C0)+(R1C1-R0C1))/2; CICA main effect = ((R0C1-R0C0)+(R1C1-R1C0))/2; interaction = R1C1-R1C0-R0C1+R0C0. These are descriptive contrasts, not significance estimates.", "",
        "| Metric, four-scene mean | Renderer main | CICA main | Interaction | Combined vs R0C0 |", "|---|---:|---:|---:|---:|"]
    for r in effects:
        if r["scene"] == "MACRO_MEAN":
            report.append(f'| {r["metric"]} | {r["renderer_main"]:+.6f} | {r["cica_main"]:+.6f} | {r["interaction"]:+.6f} | {r["combined"]:+.6f} |')
    report += ["", "## Pairing And Evidence Limits", "",
        "Initial model, optimizer, scheduler, RNG, OCMC projector and forward-probe hashes match across all four arms in each scene. Camera sequences, camera banks, partner mappings, config files and test-view GT images match. All start and gradient routing assertions passed. Source checkpoint hashes were rechecked after training.", "",
        "Nevertheless, same-renderer C0/C1 runs already diverge before CICA activates. The initial step-3001 PSNR and Gaussian count match, but later logged training states do not. This proves that matched setup is not bitwise matched trajectory; CUDA reduction/densification nondeterminism is a plausible explanation, not independently isolated here. See pre_cica_trajectory_drift.csv. Single-seed differences must not be claimed as a stable causal benefit.", "",
        "The training-time scripts were not archived with immutable source hashes in v4. Run manifests establish the configured routes; today's source/binary checks cannot retroactively prove every executed line. The supplemental backward-only audit is explicitly post-run and uses retained historical checkpoints.", "",
        "v4 used compact-output mode and deleted its 16 temporary endpoint checkpoints after evaluation. Original metrics, logs, PNGs and recorded checkpoint hashes remain. Endpoint float renders/model state cannot now be independently rerendered or resumed. No deletion was performed by this summary script.", "",
        "## Interpretation", "",
        "No corrected arm improves mean PSNR over R0C0. R1C1 is closest in mean PSNR and improves mean SSIM/LPIPS slightly, but its per-scene PSNR signs are mixed. This is not evidence that the corrections close the BND ON/OFF gap or achieve SOTA. Mathematical gradient correctness and generalization gains are separate claims.", "",
        "All recorded P_J_gt_1 values are zero. This preserves the measured BND bound, but it does not establish accurate dewatering. Clear PNG changes, darkness and raw-channel statistics are diagnostic only; no clear GT exists.", "",
        "Use the existing FSB result as the performance reference. Keep the derivative corrections available for controlled research, without promoting them as an accuracy improvement. For further CICA attribution, fork C0/C1 from one serialized pre-CICA state per renderer, including actual RNG and refinement state, retain endpoints, and add repeated identical-arm controls before interpreting small effects. Do not tune new gates or choose scene-specific switches from these test results.", "",
        "## Visuals And Resource Records", "",
        "[All-view comparison gallery](index.html) contains all four arms, underwater GT, clear images and 10x absolute underwater differences. The overview uses the lexicographically first test view of each scene; per-scene sheets include every view. Aspect ratios are preserved and there is no per-image normalization.", "",
        "Clear black fractions are measured on display-clamped 8-bit PNGs (max RGB < 0.02), not raw float tensors, and are not reconstruction metrics. resources.csv reports original wall times and per-view render peak allocated memory; timings are single observations, not a controlled performance benchmark.", "",
        "Input run: `" + str(root) + "`. All original assets are retained. No dependency installation, commit or push."]
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(out), "all_16_runs_complete": True,
                      "all_setup_checks_pass": True, "macro": [r for r in metrics if r["scene"] == "MACRO_MEAN"]}, indent=2))


if __name__ == "__main__":
    main()
