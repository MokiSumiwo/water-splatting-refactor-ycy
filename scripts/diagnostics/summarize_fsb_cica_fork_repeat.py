#!/usr/bin/env python3
"""Summarize shared-prefix repeats with execution-noise controls and all views."""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.summarize_four_way_fsb import read_json, read_csv, sha, write_csv, image, sheet

SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
ARMS = ("C0a", "C1a", "C1b", "C0b")
METRICS = ("PSNR", "SSIM", "LPIPS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    vis = out / "visualizations"
    vis.mkdir()
    status = read_json(root / "status.json")
    assert not status["active"] and not status["failed"]
    assert all(len(v) == 10 and all(c == 0 for c in v.values()) for v in status["completed"].values())
    metrics, contrasts, audits, resources, per_view, clear_diagnostics = [], [], [], [], [], []
    gallery, gt_hashes = [], {}
    retained, source_hashes, artifact_hashes = [], {}, {}
    for source, expected in read_json(root / "provenance/sha256.json").items():
        assert sha(Path(source)) == expected, f"source/binary changed: {source}"
    for scene_index, scene in enumerate(SCENES):
        for renderer in ("R0", "R1"):
            base = root / scene / renderer
            prefix = base / "PREFIX/checkpoints/step-000009999.ckpt"
            prefix_summary = read_json(base / "PREFIX/final_summary.json")
            prefix_hash = sha(prefix)
            assert prefix_summary["completed_updates"] == 6999 and prefix_summary["final_step"] == 9999
            retained.append(dict(scene=scene, renderer=renderer, arm="PREFIX", path=str(prefix), sha256=prefix_hash, bytes=prefix.stat().st_size))
            values, view_rows, renders, state_hashes = {}, {}, {}, []
            config_hashes, sequence_hashes, bank_hashes = [], [], []
            for arm in ARMS:
                path = base / arm
                audit = read_json(path / "resume_state_audit.json")
                assert audit["all_pass"] and audit["absolute_step"] == 9999 and audit["checkpoint_sha256"] == prefix_hash
                assert all(audit["optimizer_parameter_binding"].values())
                state_hashes.append(audit["hashes"])
                config_hashes.append(sha(path / "source_config.yml"))
                sequence_hashes.append(sha(path / "camera_sequence.csv"))
                bank_hashes.append(sha(path / "cica_camera_bank.json"))
                final = read_json(path / "final_summary.json")
                assert final["first_step"] == 10000 and final["final_step"] == 14999 and final["completed_updates"] == 5000
                assert final["checkpoint_retained"] and final["ocmc_projector_unchanged"]
                ckpt = path / "checkpoints/step-000014999.ckpt"
                digest = sha(ckpt)
                assert digest == final["checkpoint_rows"][0]["sha256"]
                retained.append(dict(scene=scene, renderer=renderer, arm=arm, path=str(ckpt), sha256=digest, bytes=ckpt.stat().st_size))
                manifest = read_json(path / "experiment_manifest.json")
                assert manifest["renderer_backward_mode"] == ("legacy" if renderer == "R0" else "fixed")
                assert manifest["module_schedule"]["CICA"]["derivative_mode"] == ("legacy" if arm.startswith("C0") else "exact_fsb")
                assert manifest["renderer_backend"]["forward_backend"] == "water_splatting.csrc"
                assert manifest["runtime"]["physical_gpu"] == str(6 + scene_index)
                assert manifest["runtime"]["cuda_visible_devices"] == str(6 + scene_index)
                assert manifest["runtime"]["python"] == "/opt/anaconda3/envs/water_splatting/bin/python"
                source = Path(manifest["source_checkpoint"])
                if str(source) not in source_hashes:
                    source_hashes[str(source)] = sha(source)
                assert source_hashes[str(source)] == manifest["source_checkpoint_sha256"]
                assert read_json(path / "gradient_routing_audit.json")["GRADIENT_ROUTING_AUDIT"]
                value = read_csv(path / "evaluation_metrics.csv")
                assert len(value) == 1 and int(value[0]["view_count"]) == (4 if scene == "IUI3-RedSea" else 3)
                assert int(value[0]["absolute_step"]) == 14999
                values[arm] = {m: float(value[0][m]) for m in METRICS}
                metrics.append(dict(scene=scene, renderer=renderer, arm=arm, **values[arm]))
                view_rows[arm] = {r["view_id"]: r for r in read_csv(path / "per_view_metrics.csv")}
                assert len(view_rows[arm]) == int(value[0]["view_count"])
                for metric in METRICS:
                    per_view_values = [float(v[metric]) for v in view_rows[arm].values()]
                    assert all(math.isfinite(v) for v in per_view_values)
                    assert abs(mean(per_view_values) - values[arm][metric]) < 1e-9
                renders[arm] = {}
                for row in read_csv(path / "render_manifest.csv"):
                    render_path = Path(row["path"])
                    artifact_hashes[str(render_path)] = sha(render_path)
                    renders[arm].setdefault(row["view_id"], {})[row["output_type"]] = render_path
                assert set(view_rows[arm]) == set(renders[arm])
                for view, row in view_rows[arm].items():
                    per_view.append(dict(row, scene=scene, renderer=renderer, arm=arm))
                module_rows = read_csv(path / "module_statistics.csv")
                assert [int(r["absolute_step"]) for r in module_rows if r["module"] == "CICA" and r["event"] == "refresh"] == [10000, 12000, 14000]
                assert len([r for r in module_rows if r["event"] == "calibration"]) == 1
                for name in ("evaluation_metrics.csv", "per_view_metrics.csv", "experiment_manifest.json",
                             "final_summary.json", "resume_state_audit.json", "module_statistics.csv"):
                    artifact_hashes[str(path / name)] = sha(path / name)
                decomp = read_json(path / "decomposition_safety.json")
                resources.append(dict(scene=scene, renderer=renderer, arm=arm,
                                      training_seconds=final["training_wall_seconds"], gaussians=final["final_population"],
                                      lambda_CICA=final["resolved_lambda_CICA"], P_J_gt_1=decomp["rows"][0]["P_J_gt_1"],
                                      render_ms=mean(float(v["render_seconds"])*1000 for v in view_rows[arm].values()),
                                      peak_render_gib=max(float(v["render_peak_cuda_bytes"])/2**30 for v in view_rows[arm].values())))
            assert all(s == state_hashes[0] for s in state_hashes)
            assert len(set(config_hashes)) == len(set(sequence_hashes)) == len(set(bank_hashes)) == 1
            assert all(set(v) == set(view_rows["C0a"]) for v in view_rows.values())
            audits.append(dict(scene=scene, renderer=renderer, restored_state_identical=True,
                               prefix_sha256=prefix_hash, hashes=state_hashes[0]))
            for m in METRICS:
                a, b = values["C0a"][m], values["C0b"][m]
                c, d = values["C1a"][m], values["C1b"][m]
                contrasts.append(dict(scene=scene, renderer=renderer, metric=m, C0_mean=(a+b)/2,
                                      C1_mean=(c+d)/2, delta=(c+d-a-b)/2,
                                      effect_a=c-a, effect_b=d-b, C0_repeat_abs=abs(b-a),
                                      C1_repeat_abs=abs(d-c), noise_reference=(abs(b-a)+abs(d-c))/2))
            rgb_rows, clear_rows, difference_rows, clear_difference_rows = [], [], [], []
            for view in sorted(view_rows["C0a"]):
                gt = renders["C0a"][view]["underwater"].parent / "gt.png"
                key = (scene, view)
                gt_hashes.setdefault(key, sha(gt))
                rgb_row = [(image(gt), f"{scene} / {view}\nUnderwater GT")]
                clear_row, arrays = [], {}
                for arm in ARMS:
                    gt_arm = renders[arm][view]["underwater"].parent / "gt.png"
                    assert sha(gt_arm) == gt_hashes[key]
                    rgb_row.append((image(renders[arm][view]["underwater"]),
                                    f'{renderer}/{arm} | {view}\nPSNR {float(view_rows[arm][view]["PSNR"]):.4f}'))
                    clear_row.append((image(renders[arm][view]["clear_native"]),
                                      f"{scene} / {view}\n{renderer}/{arm} clear (no GT)"))
                    assert rgb_row[-1][0].size == rgb_row[0][0].size == clear_row[-1][0].size
                    arrays[arm] = {"underwater": np.asarray(rgb_row[-1][0], dtype=np.float32) / 255,
                                   "clear": np.asarray(clear_row[-1][0], dtype=np.float32) / 255}
                    clear = arrays[arm]["clear"]
                    clear_diagnostics.append(dict(scene=scene, renderer=renderer, arm=arm, view_id=view,
                        png_mean=float(clear.mean()), png_dark_maxrgb_lt_0p02=float((clear.max(-1) < .02).mean()),
                        png_MAE_to_C0a=float(np.abs(clear - arrays["C0a"]["clear"]).mean()),
                        **{k: float(v) for k, v in view_rows[arm][view].items() if k.startswith("clear_raw_")}))
                rgb_rows.append(rgb_row)
                clear_rows.append(clear_row)
                for kind, rows in (("underwater", difference_rows), ("clear", clear_difference_rows)):
                    row = []
                    for a, b in (("C1a", "C0a"), ("C1b", "C0b"), ("C0b", "C0a"), ("C1b", "C1a")):
                        diff = np.abs(arrays[a][kind] - arrays[b][kind])
                        pixels = np.round(np.clip(diff * 10, 0, 1) * 255).astype("uint8")
                        row.append((Image.fromarray(pixels), f"{scene} / {view}\n{renderer} abs({a}-{b}) x10"))
                    rows.append(row)
            for kind, rows in (("underwater", rgb_rows), ("clear", clear_rows),
                               ("underwater_difference_x10", difference_rows),
                               ("clear_difference_x10", clear_difference_rows)):
                filename = f"{scene}_{renderer}_{kind}.png"
                sheet(rows, vis / filename)
                gallery.append((f"{scene} / {renderer} / {kind} / all test views", filename))
    for renderer in ("R0", "R1"):
        for m in METRICS:
            rows = [r for r in contrasts if r["renderer"] == renderer and r["metric"] == m]
            contrasts.append(dict(scene="MACRO_MEAN", renderer=renderer, metric=m,
                                  **{k: mean(r[k] for r in rows) for k in rows[0] if k not in ("scene", "renderer", "metric")}))
    gates = []
    for renderer in ("R0", "R1"):
        rows = [r for r in contrasts if r["renderer"] == renderer and r["metric"] == "PSNR" and r["scene"] != "MACRO_MEAN"]
        macro = next(r for r in contrasts if r["renderer"] == renderer and r["metric"] == "PSNR" and r["scene"] == "MACRO_MEAN")
        checks = dict(positive_macro=macro["delta"] > 0,
                      three_positive_scenes=sum(r["delta"] > 0 for r in rows) >= 3,
                      no_loss_beyond_0p05=all(r["delta"] >= -.05 for r in rows),
                      gain_larger_than_repeat_gap=macro["delta"] > macro["noise_reference"])
        gates.append(dict(renderer=renderer, passed=all(checks.values()), **checks))
    for name, rows in (("metrics.csv", metrics), ("contrasts.csv", contrasts), ("resources.csv", resources),
                       ("per_view_metrics.csv", per_view), ("retained_checkpoints.csv", retained),
                       ("clear_diagnostics.csv", clear_diagnostics)):
        write_csv(out / name, rows)
    (out / "pairing_audit.json").write_text(json.dumps(dict(all_pass=True, blocks=audits), indent=2)+"\n")
    (out / "decision.json").write_text(json.dumps(dict(gates=gates, promote=all(r["passed"] for r in gates),
        limit="two same-state execution repeats, not independent seeds or confidence intervals"), indent=2)+"\n")
    artifact_hashes[str(Path(__file__).resolve())] = sha(Path(__file__))
    (out / "provenance_sha256.json").write_text(json.dumps(dict(sources=source_hashes, artifacts=artifact_hashes), indent=2)+"\n")
    page = ["<!doctype html><html><meta charset='utf-8'><title>FSB CICA Shared-Prefix Repeats</title>",
            "<style>body{background:#191b1d;color:#eee;font:16px sans-serif;margin:24px}img{max-width:100%;height:auto}a{color:#8cddeb}section{margin:32px 0}</style>",
            "<h1>FSB CICA Shared-Prefix Repeats</h1><p>C0: historical derivative. C1: exact FSB derivative. a/b: identical-state repeats. R0/R1: historical/corrected backward.</p>",
            "<p>All test views; identical display scaling; no exposure or color adjustment. Clear has no ground truth.</p>",
            "<p>Difference sheets use a fixed 10x multiplier. Columns 1-2 compare CICA derivatives; columns 3-4 compare identical-state execution repeats.</p>"]
    for label, filename in gallery:
        page.append(f"<section><h2>{html.escape(label)}</h2><a href='visualizations/{filename}'><img loading='lazy' src='visualizations/{filename}'></a></section>")
    (out / "index.html").write_text("\n".join(page)+"</html>\n")
    report = ["# FSB CICA Shared-Prefix Repeat Results", "",
              "All 8 prefixes and 32 continuations completed. Within each scene/renderer, every C0/C1 repetition restored the exact same model, optimizer, scheduler, RNG and refinement-state hashes. Optimizer bindings match the live Gaussian Parameters. All 40 checkpoints are retained and checksummed.", "",
              "Two repeats estimate observed execution variation conditional on one prefix; they do not establish statistical significance or between-seed stability. CICA's historical gradient calibration policy was kept, so effects include lambda recalibration.", "",
              "| Scene | Renderer | Metric | C0 mean | C1 mean | C1-C0 | C0 abs(a-b) | C1 abs(a-b) |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in contrasts:
        report.append(f'| {r["scene"]} | {r["renderer"]} | {r["metric"]} | {r["C0_mean"]:.6f} | {r["C1_mean"]:.6f} | {r["delta"]:+.6f} | {r["C0_repeat_abs"]:.6f} | {r["C1_repeat_abs"]:.6f} |')
    report += ["", "## Registered Screening Rule", "",
               "Require positive four-scene mean under both renderers, >=3/4 positive scenes per renderer, no scene loss below -0.05 dB, and macro PSNR gain larger than mean within-condition absolute repeat difference. This is an operational gate, not a confidence interval.", "",
               "```json", json.dumps(gates, indent=2), "```", "",
               "## Artifacts And Limits", "",
               "[All-view visual comparisons](index.html). C0a/C1a/C1b/C0b are all shown; no best repetition or scene-specific combination is selected. All original 13 test views are included per arm. Metrics were computed from float render tensors, not comparison PNGs.", "",
               "Shared prefixes remove pre-CICA drift within a renderer. R0/R1 prefixes themselves were learned separately, so renderer comparisons include their training history. The result does not constitute a new SOTA comparison or a proof of accurate dewatering. Clear has no GT.", "",
               "Difference sheets compare both C1-minus-C0 pairs alongside the C0 and C1 repeat controls, at the same fixed 10x multiplier. These displayed PNG differences are not float-render reconstruction metrics. Clear brightness and dark fractions are diagnostics, not accuracy measures.", "",
               "See metrics.csv for every run, contrasts.csv for repeat gaps and effects, resources.csv for wall times/lambda/boundedness, clear_diagnostics.csv for display and raw-color diagnostics, pairing_audit.json for start hashes and retained_checkpoints.csv for every checkpoint. Source/binary archive is ../provenance/; provenance_sha256.json records report inputs."]
    (out / "REPORT.md").write_text("\n".join(report)+"\n")
    print(json.dumps({"gates": gates, "macro": [r for r in contrasts if r["scene"] == "MACRO_MEAN"],
                      "retained_checkpoints": len(retained), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
