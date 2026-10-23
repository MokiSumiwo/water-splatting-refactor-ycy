#!/usr/bin/env python3
"""Audit completed FSB four-way runs and make full-view comparisons without retraining."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
ARMS = ("R0C0", "R1C0", "R0C1", "R1C1")
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")
PAIRS = (("R1C0", "R0C0"), ("R0C1", "R0C0"), ("R1C1", "R0C0"),
         ("R1C1", "R1C0"), ("R1C1", "R0C1"))
LABELS = {"R0C0": "Original FSB", "R1C0": "Renderer fix", "R0C1": "Exact CICA", "R1C1": "Both fixes"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def image_array(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def contact_sheet(path, rows, title, width=340, height=232):
    """Letterbox full images; crops must be explicitly passed by the caller."""
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
    cols = max(len(row) for row in rows)
    canvas = Image.new("RGB", (cols * width, 46 + len(rows) * (height + 38)), "#141820")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, font=font, fill="white")
    for y, row in enumerate(rows):
        for x, (array, label) in enumerate(row):
            left, top = x * width, 46 + y * (height + 38)
            draw.text((left + 7, top + 5), label, font=font, fill="white")
            image = Image.fromarray(array)
            image = ImageOps.contain(image, (width - 8, height - 8), Image.Resampling.LANCZOS)
            canvas.paste(image, (left + (width - image.width) // 2,
                                top + 34 + (height - image.height) // 2))
    canvas.save(path, quality=94, subsampling=0)


def crop(array):
    h, w = array.shape[:2]
    return array[h // 4:3 * h // 4, w // 4:3 * w // 4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    vis = out / "visualizations"
    vis.mkdir()
    render_root = ROOT / "renders" / run.name
    rows, per_views, audits, drift, cica, visual, artifacts = [], [], [], [], [], [], []
    all_images = {kind: [] for kind in ("underwater", "clear", "clear_raw", "difference")}
    for scene, gpu in zip(SCENES, ("6", "7", "8", "9")):
        starts, manifests, sequences, banks, logs = {}, {}, {}, {}, {}
        expected_views = [Path(x).stem for x in (ROOT / "undistorted_data" /
                          ("undistorted_" + scene) / "test_list.txt").read_text().split()]
        scene_rows = {kind: [] for kind in all_images}
        for arm in ARMS:
            folder = run / scene / arm
            manifest = read_json(folder / "experiment_manifest.json")
            start = read_json(folder / "start_state_equivalence.json")
            final = read_json(folder / "final_summary.json")
            sequence = read_json(folder / "camera_sequence_hashes.json")
            routing = read_json(folder / "gradient_routing_audit.json")
            safety = read_json(folder / "decomposition_safety.json")
            views = read_csv(folder / "per_view_metrics.csv")
            endpoint = read_csv(folder / "evaluation_metrics.csv")
            assert len(endpoint) == 1 and int(endpoint[0]["absolute_step"]) == 14999
            assert final["final_step"] == 14999 and final["completed_updates"] == 11999
            assert final["snapshot_steps"] == [14999]
            assert final["ocmc_projector_unchanged"] and not final["auxiliary_appearance_enabled"]
            assert not final["mdrr_enabled"] and final["cica_enabled"]
            assert start["START_STATE_EQUIVALENCE"] and routing["GRADIENT_ROUTING_AUDIT"]
            assert sequence["CAMERA_SEQUENCE_EXACT_MATCH"] and sequence["length"] == 11999
            assert manifest["training_seed"] == 42 and manifest["runtime"]["physical_gpu"] == gpu
            assert manifest["renderer_backward_mode"] == ("fixed" if arm[1] == "1" else "legacy")
            assert manifest["module_schedule"]["CICA"]["derivative_mode"] == ("exact_fsb" if arm[3] == "1" else "legacy")
            assert manifest["module_schedule"]["CICA"]["start_step"] == 10000
            assert manifest["renderer_backend"]["forward_backend"] == "water_splatting.csrc"
            assert sorted(v["view_id"] for v in views) == sorted(expected_views)
            assert safety["all_P_J_gt_1_zero"]
            starts[arm], manifests[arm], sequences[arm] = start, manifest, sequence
            banks[arm] = read_json(folder / "cica_camera_bank.json")
            logs[arm] = {int(r["absolute_step"]): r for r in read_csv(folder / f"training_summary_{arm}.csv")}
            row = {"scene": scene, "arm": arm, "views": len(views), **{m: float(endpoint[0][m]) for m in METRICS}}
            for m in METRICS:
                assert abs(row[m] - statistics.mean(float(v[m]) for v in views)) < 1e-10
            row.update(training_minutes=final["training_wall_seconds"] / 60,
                       render_ms_mean=1000 * statistics.mean(float(v["render_seconds"]) for v in views),
                       render_peak_allocated_GiB=max(float(v["render_peak_cuda_bytes"]) for v in views) / 2 ** 30,
                       final_population=final["final_population"],
                       **{k: statistics.mean(float(v[k]) for v in views) for k in views[0] if k.startswith("clear_raw_")})
            rows.append(row)
            per_views.extend({"scene": scene, **v} for v in views)
            modules = read_csv(folder / "module_statistics.csv")
            calibration = [v for v in modules if v["event"] == "calibration"]
            assert len(calibration) == 1 and calibration[0]["absolute_step"] == "10000"
            c = calibration[0]
            last = logs[arm][14999]
            cica.append({"scene": scene, "arm": arm, "lambda": final["resolved_lambda_CICA"],
                         "calibration_photometric_dc_norm": float(c["photometric_features_dc_grad_l2"]),
                         "calibration_cica_raw_norm": float(c["raw_CICA_features_dc_grad_l2"]),
                         "calibrated_ratio": final["resolved_lambda_CICA"] * float(c["raw_CICA_features_dc_grad_l2"]) /
                            float(c["photometric_features_dc_grad_l2"]),
                         "refresh_count": sum(x["event"] in ("refresh", "topology_refresh") for x in modules),
                         **{k: float(last[k]) for k in ("CICA_valid_fraction", "CICA_median_abs_delta_chi",
                            "CICA_delta_R_over_G", "CICA_delta_B_over_G", "cica_grad_l2")},
                         "target_hash_final": [x["target_hash"] for x in modules if x.get("target_hash")][-1]})
            for f in ("experiment_manifest.json", "start_state_equivalence.json", "evaluation_metrics.csv",
                      "per_view_metrics.csv", "training_summary.json", "final_summary.json"):
                artifacts.append({"path": str(folder / f), "sha256": sha(folder / f)})
        keys = ("loaded_model_hash", "optimizer_hash", "scheduler_hash", "scaler_hash", "training_rng_manifest",
                "ocmc_projector_sha256", "forward_probe", "locked_config", "source_checkpoint_sha256")
        checks = {k: len({json.dumps(starts[a][k], sort_keys=True) for a in ARMS}) == 1 for k in keys}
        checks.update(camera_sequence=len({sequences[a]["sha256"] for a in ARMS}) == 1,
                      camera_csv=len({sha(run / scene / a / "camera_sequence.csv") for a in ARMS}) == 1,
                      camera_bank=len({json.dumps(banks[a], sort_keys=True) for a in ARMS}) == 1,
                      source_config=len({manifests[a]["source_config_sha256"] for a in ARMS}) == 1,
                      source_hash_verified=sha(Path(manifests["R0C0"]["source_checkpoint"])) == starts["R0C0"]["source_checkpoint_sha256"],
                      all_arm_exit_zero=read_json(run / scene / "status.json") == dict.fromkeys(ARMS, 0))
        assert all(checks.values()), (scene, checks)
        audits.append({"scene": scene, "gpu": gpu, "expected_test_views": expected_views, "checks": checks})
        for a, b in (("R0C1", "R0C0"), ("R1C1", "R1C0")):
            for step in sorted(logs[a]):
                if step >= 10000:
                    continue
                x, y = logs[a][step], logs[b][step]
                assert x["CICA_enabled"] == y["CICA_enabled"] == "False"
                drift.append({"scene": scene, "contrast": f"{a}-{b}", "step": step,
                              "delta_train_loss": float(x["L_RGB"]) - float(y["L_RGB"]),
                              "delta_train_PSNR": float(x["PSNR"]) - float(y["PSNR"]),
                              "delta_population": int(x["gaussian_count"]) - int(y["gaussian_count"])})
        for view in expected_views:
            data = {}
            for arm in ARMS:
                base = render_root / arm / scene / "step_14999" / view / arm
                data[arm] = {kind: image_array(base / (kind + ".png")) for kind in
                             ("gt", "underwater", "clear", "clear_raw_display_clamp01")}
                for kind in data[arm]:
                    f = base / (kind + ".png")
                    artifacts.append({"path": str(f), "sha256": sha(f)})
                assert np.array_equal(data[arm]["gt"], data["R0C0"]["gt"])
            uw, clear, raw, diffs = [(data["R0C0"]["gt"], f"{view} | underwater GT")], [], [], []
            for arm in ARMS:
                uw.append((data[arm]["underwater"], f"{arm} {LABELS[arm]}"))
                clear.append((data[arm]["clear"], f"{view} | {arm}"))
                raw.append((data[arm]["clear_raw_display_clamp01"], f"{view} | {arm}"))
                for kind in ("underwater", "clear", "clear_raw_display_clamp01"):
                    a = data[arm][kind].astype(np.float32) / 255
                    b = data["R0C0"][kind].astype(np.float32) / 255
                    delta = np.abs(a - b)
                    lum = a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
                    visual.append({"scene": scene, "view": view, "arm": arm, "export": kind,
                                   "mae_to_R0C0_png": float(delta.mean()), "p99_abs_difference_png": float(np.quantile(delta, .99)),
                                   "black_luma_lt_0p02_fraction_png": float((lum < .02).mean()),
                                   "near_white_all_gt_0p98_fraction_png": float((a.min(-1) > .98).mean()),
                                   "min_png": float(a.min()), "max_png": float(a.max()),
                                   "mean_r_png": float(a[..., 0].mean()), "mean_g_png": float(a[..., 1].mean()),
                                   "mean_b_png": float(a[..., 2].mean())})
            for kind in ("underwater", "clear"):
                for arm in ARMS[1:]:
                    delta = np.abs(data[arm][kind].astype(np.float32) - data["R0C0"][kind].astype(np.float32))
                    diffs.append((np.clip(delta * 8, 0, 255).astype(np.uint8), f"{kind} | {arm}-R0C0 x8"))
            for kind, cells in (("underwater", uw), ("clear", clear), ("clear_raw", raw), ("difference", diffs)):
                scene_rows[kind].append(cells)
                all_images[kind].append(cells)
            contact_sheet(vis / f"{scene}_{view}_underwater.jpg", [uw], scene + " | full view; identical display scale", width=420, height=290)
            contact_sheet(vis / f"{scene}_{view}_clear.jpg", [clear], scene + " | clear visualization; no clear GT", width=420, height=290)
            contact_sheet(vis / f"{scene}_{view}_differences.jpg", [diffs[:3], diffs[3:]],
                          scene + " / " + view + " | absolute PNG differences x8; fixed scale", width=420, height=290)
        for kind, cells in scene_rows.items():
            contact_sheet(vis / f"{scene}_{kind}_all_views.jpg", cells, scene + " | " + kind + " | all test views")
        for kind in ("underwater", "clear"):
            contact_sheet(vis / f"{scene}_{kind}_center_crops.jpg",
                          [[(crop(a), label) for a, label in cells] for cells in scene_rows[kind]],
                          scene + " | center 50% crop; same crop in every arm")

    for kind, cells in all_images.items():
        contact_sheet(vis / f"{kind}_all_views.jpg", cells, "Four scenes, all 13 test views | " + kind)
    by = {(r["scene"], r["arm"]): r for r in rows}
    macro = [{"scene": "Macro", "arm": a, "scene_count": 4,
              **{m: statistics.mean(by[(s, a)][m] for s in SCENES) for m in METRICS}} for a in ARMS]
    for r in macro:
        by[("Macro", r["arm"])] = r
    deltas = [{"scene": s, "contrast": f"{a}-{b}",
               **{"delta_" + m: by[(s, a)][m] - by[(s, b)][m] for m in METRICS}}
              for s in (*SCENES, "Macro") for a, b in PAIRS]
    interactions = [{"scene": s, **{m: by[(s, "R1C1")][m] - by[(s, "R1C0")][m] -
                                     by[(s, "R0C1")][m] + by[(s, "R0C0")][m] for m in METRICS}}
                    for s in (*SCENES, "Macro")]
    view_map = {(r["scene"], r["arm"], r["view_id"]): r for r in per_views}
    view_deltas = []
    for (s, arm, view), r in view_map.items():
        if arm == "R0C0":
            continue
        view_deltas.append({"scene": s, "arm": arm, "view": view,
                            **{"delta_" + m: float(r[m]) - float(view_map[(s, "R0C0", view)][m]) for m in METRICS}})
    write_csv(out / "scene_metrics.csv", rows)
    write_csv(out / "macro_metrics.csv", macro)
    write_csv(out / "paired_deltas.csv", deltas)
    write_csv(out / "factorial_interactions_descriptive.csv", interactions)
    write_csv(out / "per_view_metrics.csv", per_views)
    write_csv(out / "per_view_deltas.csv", view_deltas)
    write_csv(out / "pre_cica_trajectory_drift.csv", drift)
    write_csv(out / "cica_statistics.csv", cica)
    write_csv(out / "visual_diagnostics_png.csv", visual)
    write_csv(out / "input_hashes.csv", artifacts)
    production_hash = sha(ROOT / "water_splatting/csrc.so")
    production_unchanged = production_hash == read_json(run / "provenance/hashes.json")["water_splatting/csrc.so"]
    assert production_unchanged
    audit = dict(run=str(run), complete_arms=16, total_model_test_view_pairs=len(per_views),
                 paired_start_and_protocol_pass=True, scenes=audits,
                 bitwise_training_trajectory_equal=False,
                 pre_cica_drift_observed=any(r["delta_population"] != 0 for r in drift),
                 no_clear_ground_truth=True, production_binary_unchanged=production_unchanged,
                 production_binary_sha256=production_hash,
                 checkpoint_policy="New temporary checkpoints deleted only after successful endpoint reload/evaluation; hashes retained",
                 endpoint_vjp_residual_measured=False)
    write_json(out / "audit.json", audit)

    # Portable browseable gallery. Every test view is shown, never selected by metrics.
    links = []
    for s in SCENES:
        for kind in ("underwater", "clear", "difference", "clear_raw"):
            f = f"visualizations/{s}_{kind}_all_views.jpg"
            links.append(f'<h2>{html.escape(s)} — {kind}</h2><a href="{f}"><img loading="lazy" src="{f}" alt="{s} {kind}"></a>')
        links.append(f'<p><a href="visualizations/{s}_clear_center_crops.jpg">Fixed center crops: clear</a> · '
                     f'<a href="visualizations/{s}_underwater_center_crops.jpg">Fixed center crops: underwater</a></p>')
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>FSB four-way comparison</title>'
        '<style>body{font:16px system-ui;margin:24px;background:#141820;color:#eee}img{width:100%;height:auto}'
        'a{color:#8fd1ff}h2{margin-top:40px}</style><h1>FSB: renderer / CICA four-way comparison</h1>'
        '<p>R0C0 original; R1C0 renderer fix; R0C1 exact CICA; R1C1 both. All 13 test views. '
        'Images use the saved display mapping. Clear has no ground truth. Differences use a fixed ×8 scale.</p>' + ''.join(links))

    macro_table = md_table(["配置", "PSNR ↑", "ΔPSNR", "SSIM ↑", "ΔSSIM", "LPIPS ↓", "ΔLPIPS", "MSE ↓"],
        [[r["arm"], f'{r["PSNR"]:.6f}', f'{r["PSNR"]-macro[0]["PSNR"]:+.6f}', f'{r["SSIM"]:.6f}',
          f'{r["SSIM"]-macro[0]["SSIM"]:+.6f}', f'{r["LPIPS"]:.6f}',
          f'{r["LPIPS"]-macro[0]["LPIPS"]:+.6f}', f'{r["MSE"]:.9f}'] for r in macro])
    scene_table = md_table(["场景", *ARMS, "联合修正 ΔPSNR"],
        [[s, *[f'{by[(s,a)]["PSNR"]:.6f}' for a in ARMS],
          f'{by[(s,"R1C1")]["PSNR"]-by[(s,"R0C0")]["PSNR"]:+.6f}'] for s in SCENES])
    full_table = md_table(["场景", "配置", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "MSE ↓"],
        [[r["scene"], r["arm"], f'{r["PSNR"]:.6f}', f'{r["SSIM"]:.6f}',
          f'{r["LPIPS"]:.6f}', f'{r["MSE"]:.9f}'] for r in rows])
    contrast_table = md_table(["四场景宏平均对比", "ΔPSNR", "ΔSSIM", "ΔLPIPS", "ΔMSE"],
        [[r["contrast"], *[f'{r["delta_"+m]:+.9f}' for m in METRICS]] for r in deltas if r["scene"] == "Macro"])
    drift_table = md_table(["场景", "对比（CICA 尚未启用）", "9500 步 Gaussian 数量差", "训练 loss 差"],
        [[r["scene"], r["contrast"], r["delta_population"], f'{r["delta_train_loss"]:+.7f}']
         for r in drift if r["step"] == 9500])
    runtime_table = md_table(["场景", "配置", "训练分钟", "单次 render 均值 ms", "render 峰值 allocated GiB"],
        [[r["scene"], r["arm"], f'{r["training_minutes"]:.2f}', f'{r["render_ms_mean"]:.2f}',
          f'{r["render_peak_allocated_GiB"]:.3f}'] for r in rows])
    text = f'''# FSB renderer × CICA 四路配对结果（2026-09-16）

16/16 次训练、16/16 次终点评估已完成，包含全部 13 个测试相机，共 52 个 model-view 评估。联合修正相对 R0C0 的宏平均 PSNR 为 −0.015612 dB、SSIM +0.000331、LPIPS −0.000607、MSE +0.000010150。未获得稳定的带水 PSNR 改善；本轮不晋级为新的 SOTA 或 FSB 优化版本。

## 配置与已执行协议

- R0C0：历史 renderer backward + 历史 CICA 导数。
- R1C0：修正 renderer backward + 历史 CICA 导数。
- R0C1：历史 renderer backward + FSB 精确 CICA DC 导数。
- R1C1：修正 renderer backward + FSB 精确 CICA DC 导数。
- GPU 6=Curasao，7=IUI3-RedSea，8=JapaneseGradens-RedSea，9=Panama。同场景四 arm 顺序执行，四场景并发（Curasao 提前约 2.6 分钟启动）。仅 conda water_splatting。
- 同场景同一个 step-3000 checkpoint，seed=42；参数/optimizer/scheduler/scaler、RNG 起点、OCMC projector、camera sequence 和 bank 全部配对。训练 3001–14999 共 11999 次更新，固定终点，没有 best checkpoint 选择。
- BND ON，FSB full headroom α=1，degreewise=(1,1,1)，constant schedule；M1+OCMC、CICA ON，Auxiliary/MDRR/RAOC OFF。CICA step=10000 启用，保留原校准规则（初次 DC 梯度占比 0.10）及 topology refresh。
- 全部 forward、SH、投影继续使用历史 water_splatting.csrc；仅 rasterize_backward/nd_rasterize_backward 路由到独立 JIT 扩展。四路初始 forward tensor hashes 完全一致。历史 csrc.so SHA256 未变。
- 新终点 checkpoint 先保存在 /dev/shm，独立加载评估成功后按磁盘限制删除；保留 SHA256、manifest、逐视图指标、PNG 和日志。未删除历史资产。不能从这些已删除的新 checkpoint 再做端点 VJP 诊断。

运行目录：`{run}`。本报告从已有训练资产汇总，不重新训练。

## 指标

指标来自模型原生浮点张量评估，非 PNG 重算。每个场景先平均自己的全部测试视图，再对四场景等权平均。PSNR 也是逐视图均值，不等于对宏平均 MSE 取对数。PSNR/SSIM 越大越好，LPIPS/MSE 越小越好。未使用外部 clear GT。

{macro_table}

{scene_table}

{full_table}

{contrast_table}

完整 16 行四指标见 [scene_metrics.csv](scene_metrics.csv)，逐视图见 [per_view_metrics.csv](per_view_metrics.csv)。因子交互见 [factorial_interactions_descriptive.csv](factorial_interactions_descriptive.csv)，仅为单次观测差分，不能据此证明协同。

## 严格配对的边界

[audit.json](audit.json) 验证了四场景全部起点/协议匹配，以及全部测试视图覆盖。实际训练并非逐 bit 确定：同 renderer 的 C0/C1 在 CICA 启用前已经分叉。初步排查与 CUDA 浮点归约及 refinement 的敏感性相符，未隔离证明唯一原因。

{drift_table}

因此可以报告这四路在同起点同协议下的终点表现，但不能把每个 0.01–0.1 dB 差值当成精确的独立因果效应。只有一个 seed、没有同配置重复实验，尚无跨 seed 方差或统计显著性证据。CICA λ 也按相同规则重新校准而非固定数值；该对比评估的是完整 CICA 兼容性切换。见 [cica_statistics.csv](cica_statistics.csv)。

## 可视化与去水边界

- [浏览全部对比](index.html)
- [全部带水拼接图](visualizations/underwater_all_views.jpg)
- [全部去水拼接图](visualizations/clear_all_views.jpg)
- [全部 raw-clear 显示图](visualizations/clear_raw_all_views.jpg)
- [全部差异图，统一 ×8](visualizations/difference_all_views.jpg)

每个场景的全部测试视图均输出 full-view 和固定中心 50% crop，无按指标挑图、曝光调整或颜色增强。带水拼接包含 GT；clear 无 GT。差异图来自保存的 8-bit PNG（存在量化），统一放大 8 倍，不用于重算论文指标。

四路所有场景的 `P_J_gt_1=0`，说明记录到的可见 Gaussian 色值上界得到保留，不等于证明去水质量不退化。原始 clear 分布统计保存在 [scene_metrics.csv](scene_metrics.csv)；PNG 黑像素比例、色彩范围和逐像素差异在 [visual_diagnostics_png.csv](visual_diagnostics_png.csv)。黑区少、颜色亮或 raw 色差小均不是恢复准确度证据。

## 耗时与显存

{runtime_table}

render 耗时为独立端点评估时每视图一次同步计时，未做充分 warm-up/重复性能基准；显存是 render 调用期间 allocated 峰值，包含已加载模型/缓存，不是完整训练峰值。训练分钟来自 training_summary，未包含初始审计及独立评估开销。

## 结论与下一步

1. 导数一致性修正不等于新视角指标改善。R1C0 的四场景 PSNR 均下降；R0C1 仅 Panama 小幅上升。R1C1 只在 Curasao 提高 PSNR，其他三场景下降，无法替换当前候选。
2. 联合修正的 SSIM/LPIPS 有小幅宏平均改善，PSNR/MSE 则变差，是混合结果。不能宣称已弥合 BND ON/OFF 差距；本轮也没有新的 BND OFF 对照训练。
3. renderer alpha-cap 的合成/冻结 VJP 正确性证据仍成立，FSB 的 CICA DC 导数适配仍有数学依据；它们应按正确性修正评估，而非包装为创新收益。此前冻结 VJP 诊断使用的是历史 checkpoint，不能移作本轮已删除终点 checkpoint 的 VJP 测量。
4. 若继续做机制归因，先在每个 renderer 条件下从同一个 step-9999 完整训练状态分叉 C0/C1，恢复 optimizer、scheduler、RNG、refinement 和全部非参数状态；加同配置重复，先测噪声再解释微小收益。此为后续建议，本轮未新增该实验或调参。

## 过程记录

早期 v1–v3 启动被中止，未纳入指标。先前把 32 个同命令子进程解释为 DataLoader 递归复制的说法不成立；已核查 Torch Inductor 的 `AsyncCompile.warm_pool()` 会创建最多 32 个空闲编译进程。有效 v4 四场景正常完成，场景 status 全部为 0。为使后续启动与实际四卡协议一致，汇总阶段恢复 launcher 的四场景并发逻辑；不改变已完成的训练或其 provenance 快照。

汇总输入哈希见 [input_hashes.csv](input_hashes.csv)，训练时源码和二进制哈希保存在运行目录的 `provenance/`。当前生产二进制 SHA256：`{production_hash}`。
'''
    (out / "REPORT.md").write_text(text)
    print(json.dumps({"output": str(out), "complete_arms": 16, "test_pairs": len(per_views),
                      "macro": macro, "paired_start_pass": True}, indent=2))


if __name__ == "__main__":
    main()
