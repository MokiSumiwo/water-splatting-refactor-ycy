#!/usr/bin/env python3
"""Aggregate frozen derivative diagnostics without rerendering or training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
LABELS = ("Curasao", "IUI3", "JapaneseGradens", "Panama")


def write_csv(path, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    out = root/"summary"
    out.mkdir(exist_ok=False)
    summaries, views, metrics, probes, controller_rows = [], [], [], [], []
    for scene in SCENES:
        status = json.loads((root/scene/"status.json").read_text())
        assert status["complete"] and status["forward_reference_pass"]
        for method in ("BND_CICA", "FSB"):
            summary = json.loads((root/scene/f"{method}_summary.json").read_text())
            records = json.loads((root/scene/f"{method}_views.json").read_text())
            assert summary["forward_reference_pass"] and all(summary["preservation"].values())
            assert len(records) == 6 and not summary["test_views_used"] and summary["optimizer_steps"] == 0
            summaries.append(summary)
            views.extend(records)
            c = summary["controllers"]
            g = c["gradients"]["corrected"]
            metrics.append(dict(scene=scene, method=method,
                jacobian_relative_l2=summary["weighted_jacobian_relative_l2"],
                information_mass_error_gt_10=summary["information_mass_error_gt_10"],
                information_mass_error_gt_25=summary["information_mass_error_gt_25"],
                weighted_target_abs_delta=c["weighted_target_abs_delta"],
                target_abs_delta_p99=c["target_abs_delta"]["p99"],
                information_weight_relative_l2=c["weight_relative_l2"],
                cica_gradient_relative_change=g["gradient_relative_change"],
                cica_gradient_cosine=g["cosine_to_old"],
                cica_gradient_norm_ratio=g["raw_gradient_norm"]/c["gradients"]["old"]["raw_gradient_norm"],
                target_only_gradient_relative_change=c["gradients"]["target_only"]["gradient_relative_change"],
                weight_only_gradient_relative_change=c["gradients"]["weight_only"]["gradient_relative_change"],
                correction_sign_flip_fraction=c["correction_sign_flip_fraction"],
                support_changed=c["support_changed"], gate_changed=c["gate_changed"],
                old_valid=c["old_valid"], new_valid=c["corrected_valid"],
                elapsed_seconds=summary["elapsed_seconds"], peak_allocated_mb=summary["peak_allocated_mb"]))
            for arm, grad in c["gradients"].items():
                controller_rows.append(dict(scene=scene, method=method, arm=arm, **grad))
            for view in records:
                for native, forward in zip(view["native_probes"], view["forward_probes"]):
                    probes.append(dict(scene=scene, method=method, view_id=view["view_id"],
                        cotangent=native["cotangent"], native_backward_relative_l2=native["corrected_relative_l2"],
                        native_backward_cosine=native["corrected_cosine"],
                        forward_adjoint_relative_error=forward["corrected_relative_error"],
                        disagreement_direction_backward_relative_error=forward["native_backward_relative_error"]))
    write_csv(out/"scene_metrics.csv", metrics)
    write_csv(out/"native_forward_backward_probes.csv", probes)
    write_csv(out/"controller_gradients.csv", controller_rows)
    per_view = []
    for row in views:
        s = row["statistics"]
        per_view.append(dict(scene=row["scene"], method=row["method"], view_id=row["view_id"],
            derivative_max_abs=row["local_autograd"]["derivative_max_abs"],
            old_factor_n_relative_l2=row["old_factor_n_relative_l2"],
            old_factor_i_relative_l2=row["old_factor_i_relative_l2"],
            jacobian_relative_l2=s["weighted_jacobian_relative_l2"],
            information_mass_error_gt_10=s["information_mass_error_gt_10"],
            information_mass_error_gt_25=s["information_mass_error_gt_25"],
            local_delta_relative_l2=row["local_delta_relative_l2"],
            forward_reference_pass=row["forward_reference_pass"],
            native_backward_pass=row["native_backward_pass"]))
    write_csv(out/"per_view_metrics.csv", per_view)
    fsb = [r for r in metrics if r["method"] == "FSB"]
    x = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    axes[0, 0].bar(x, [100*r["jacobian_relative_l2"] for r in fsb], color="#3976af")
    axes[0, 0].set(title="FSB: information-weighted DC Jacobian error", ylabel="Relative L2 (%)", ylim=(0, 1.7))
    for xi, row in zip(x, fsb):
        axes[0, 0].text(xi, 100*row["jacobian_relative_l2"]+.025, f"{100*row['jacobian_relative_l2']:.2f}%", ha="center")
    for offset, key, label, color in ((-.25, "cica_gradient_relative_change", "Target + weight + gate", "#3976af"),
                                    (0, "target_only_gradient_relative_change", "Target only", "#e29841"),
                                    (.25, "weight_only_gradient_relative_change", "Weight only", "#66a681")):
        axes[0, 1].bar(x+offset, [100*r[key] for r in fsb], width=.24, label=label, color=color)
    axes[0, 1].set(title="FSB: CICA gradient changes at frozen refresh", ylabel="Relative L2 change (%)")
    axes[0, 1].legend(fontsize=8)
    for i, method in enumerate(("BND_CICA", "FSB")):
        values = []
        for scene in SCENES:
            group = [p["native_backward_relative_l2"] for p in probes if p["scene"] == scene and p["method"] == method and p["cotangent"] == "gt_minus_pred"]
            values.append(100*np.mean(group))
        axes[1, 0].bar(x+(-.16 if i == 0 else .16), values, width=.3, label=method, color=("#e29841", "#3976af")[i])
    axes[1, 0].set(title="Native backward discrepancy also exists in BND", ylabel="Residual VJP relative L2 (%)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].bar(x, [r["weighted_target_abs_delta"] for r in fsb], color="#66a681")
    axes[1, 1].set(title="FSB: information-weighted log-chroma target shift", ylabel="Mean absolute log-ratio change")
    for ax in axes.flat:
        ax.set_xticks(x, LABELS, rotation=12)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Frozen step-14999 checkpoints | 6 training-bank views per scene | No optimizer updates", fontsize=12)
    fig.savefig(out/"diagnostic_overview.png", dpi=170)
    fig.savefig(out/"diagnostic_overview.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    edges = ["0-.25", ".25-.5", ".5-.75", ".75-.9", ".9-.99", ".99-1.01", "1.01-1.1", "1.1-1.25", "1.25-1.5", "1.5-2", "2-4", ">=4"]
    for ax, scene, label in zip(axes.flat, SCENES, LABELS):
        hist = np.sum([v["statistics"]["ratio_histogram_information"] for v in views if v["scene"] == scene and v["method"] == "FSB"], axis=0)
        ax.bar(np.arange(len(hist)), 100*hist/hist.sum(), color="#3976af")
        ax.set_xticks(np.arange(len(hist)), edges, rotation=65, fontsize=8)
        ax.set(title=label, ylabel="Exact information mass (%)", xlabel="Assumed derivative / actual derivative")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(out/"derivative_ratio_information.png", dpi=170)
    plt.close(fig)
    maxima = dict(local_color_derivative_max_abs=max(r["derivative_max_abs"] for r in per_view),
                  old_factorization_relative_l2=max(max(r["old_factor_n_relative_l2"], r["old_factor_i_relative_l2"]) for r in per_view),
                  forward_adjoint_relative_error=max(p["forward_adjoint_relative_error"] for p in probes),
                  bnd_control_gradient_relative_change=max(r["cica_gradient_relative_change"] for r in metrics if r["method"] == "BND_CICA"))
    decision = dict(diagnostic_complete=True, all_forward_reference_checks_pass=True,
                    native_backward_checks_pass=False, color_derivative_mismatch_confirmed=True,
                    priority="Audit renderer forward/backward consistency, then isolate CICA DC-derivative compatibility; no new FSB variant yet",
                    training_started=False, psnr_improvement_measured=False, maxima=maxima)
    (out/"decision.json").write_text(json.dumps(decision, indent=2)+"\n")
    lines = ["# FSB–CICA 导数一致性：四场景冻结诊断", "",
        "诊断已完成：FSB 的 CICA DC 导数不匹配得到确认；同时发现普通 BND 也存在的 renderer forward/backward 不一致。原生 forward 参考核验通过，原生 backward 核验失败必须保留。", "",
        "固定 step 14999 的 BND ON+CICA 和严格 FSB checkpoint，各使用历史记录的 6 个 CICA 训练相机。共 8 个模型、48 个模型-视图组合；24 个相机只来自训练集。模型、配置、checkpoint 文件及首个训练相机的原生带水/去水/coverage/depth 哈希不变。没有 optimizer step，也没有测试视图评估。标准数据管理器初始化会缓存测试集，但诊断不访问其图像或指标。", "",
        "## FSB 内部：仅更换 CICA 的颜色导数", "",
        "| 场景 | 合成加权 Jacobian 相对 L2 误差 | 导数误差 >10% 的信息量占比 | CICA 梯度向量相对变化 | 梯度余弦 | 加权目标绝对变化 |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in fsb:
        lines.append(f"| {row['scene']} | {100*row['jacobian_relative_l2']:.4f}% | {100*row['information_mass_error_gt_10']:.4f}% | {100*row['cica_gradient_relative_change']:.4f}% | {row['cica_gradient_cosine']:.6f} | {row['weighted_target_abs_delta']:.7f} |")
    lines += ["", "梯度变化定义为 `||g_corrected-g_old||/||g_old||`，不是梯度范数提升百分比，更不是训练收益。合成误差用真实 alpha/transmission 合成权重与正确 DC Jacobian 的平方和归一化；信息量占比不是像素或 Gaussian 数量占比。目标变化是在同一个冻结模型上重新构建的 DC log-chroma 差值。", "",
        "| 场景 | 只改 target 的梯度变化 | 只改 weight 的梯度变化 | 完整改动的梯度范数比 | gate 变化数量 | 校正方向翻转比例 |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in fsb:
        lines.append(f"| {row['scene']} | {100*row['target_only_gradient_relative_change']:.4f}% | {100*row['weight_only_gradient_relative_change']:.4f}% | {row['cica_gradient_norm_ratio']:.6f} | {row['gate_changed']} | {100*row['correction_sign_flip_fraction']:.4f}% |")
    lines += ["", "只改 target/weight 的诊断保留旧 gate；它们不是可线性相加的分解。方向翻转仅统计两种校正幅度均超过 1e-4 的共同有效 log-chroma 通道。", "",
        "## 参考核验", "",
        f"- 颜色解析导数与实际 CUDA SH/color autograd 最大绝对误差：{maxima['local_color_derivative_max_abs']:.3g}。",
        f"- 旧 CICA 累加器与分解重算的最大相对 L2 误差：{maxima['old_factorization_relative_l2']:.3g}。",
        f"- 正确导数累加器与原生 forward 方向导数的最大相对误差：{maxima['forward_adjoint_relative_error']:.3g}。",
        f"- 普通 BND 的解析颜色导数差异为零；重算 target 带来的最大 CICA 梯度相对变化：{100*maxima['bnd_control_gradient_relative_change']:.4f}%，反映浮点原子累加及中值 target 的数值敏感性。", "",
        "这些核验保留 1e-3 的相对误差阈值。原生 backward 失败后新增 forward-adjoint 对照，保留原注册核验的失败标记，未放宽阈值。", "",
        "## 独立发现：原生 renderer 的反向不一致", "",
        "| 场景 | BND residual VJP 平均误差 | FSB residual VJP 平均误差 |",
        "|---|---:|---:|"]
    for scene in SCENES:
        vals = [100*np.mean([p["native_backward_relative_l2"] for p in probes if p["scene"] == scene and p["method"] == m and p["cotangent"] == "gt_minus_pred"]) for m in ("BND_CICA", "FSB")]
        lines.append(f"| {scene} | {vals[0]:.4f}% | {vals[1]:.4f}% |")
    lines += ["", "表中使用三个预定训练相机的 `GT-pred` 固定余向量；完整 CSV 也保留固定带符号余向量的结果。此误差与上表颜色导数误差有不同定义，不应相减归因。", "",
        "RGB forward 的 alpha cap 是 0.999，backward 是 0.99。单像素、单 Gaussian 测试在 opacity=0.995 时，真实 forward 颜色导数为 0.995，原生 backward 为约 0.495；opacity=0.999 时为 0.999 对约 0.099。0.5 和 0.99 两个对照一致。这个缺陷能独立复现，但尚未证明它解释所有真实场景的 backward 偏差。", "",
        "## 解释和下一步", "",
        "FSB 的 sigmoid 近似误差在当前 checkpoint 的总信息量上不大，但 CICA target/加权中值过程使其造成可测的梯度差异。当前证据支持修正兼容性，不支持宣称已找到 BND ON/OFF 指标差距的主要原因。", "",
        "优先核验和修复 renderer 的 forward/backward 一致性，再让 CICA 显式接收当前颜色映射的 DC feature 导数。冻结核验通过后，在 FSB 内以 renderer 修正与 CICA 修正做独立的 2×2 配对实验；保留旧实现对照，避免把两项变化混成新方法。暂不设计新的 gate、schedule 或颜色正则。", "",
        "本诊断的 target 是在 14999 重新构建的假想刷新，不是历史 14000 刷新 target 的复原；没有加载/更新历史 Adam 动量，没有重校准 lambda。现有历史指标仍有效，但不能把 FSB 收益完全归因于独立的颜色机制。冻结图像保持一致，无 PSNR/clear 质量提升声明。", "",
        "## 文件", "",
        "- [诊断总览图](diagnostic_overview.png) / [PDF](diagnostic_overview.pdf)",
        "- [导数比例的信息量分布](derivative_ratio_information.png)",
        "- [场景指标](scene_metrics.csv)",
        "- [逐视图记录](per_view_metrics.csv)",
        "- [CICA 梯度对照](controller_gradients.csv)",
        "- [原生 forward/backward 核验](native_forward_backward_probes.csv)",
        "- [结论与核验上界](decision.json)", "",
        "各场景目录保存原始 JSON、确定性 target 样本和单 Gaussian 复现。GPU 仅使用物理 6–9；未安装依赖、修改生产代码或覆盖历史资产，未 commit/push。"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    provenance = root/"provenance"
    provenance.mkdir(exist_ok=False)
    sources = ["scripts/diagnostics/audit_fsb_cica_derivative.py", "scripts/diagnostics/summarize_fsb_cica_derivative.py",
               "scripts/diagnostics/evaluate_fsb_degreewise_scene.py", "scripts/experiments/run_direct_mdrr_cica_scene.py",
               "water_splatting/fields/gaussian_appearance.py", "water_splatting/cuda/csrc/forward.cu", "water_splatting/cuda/csrc/backward.cu",
               "research_notes/FSB_CICA_DERIVATIVE_PROTOCOL_2026-09-15.md", "research_notes/FSB_CICA_DERIVATIVE_FORWARD_ADDENDUM_2026-09-15.md"]
    manifest = []
    for source in sources:
        src = ROOT/source
        dest = provenance/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        manifest.append(dict(source=source, sha256=hashlib.sha256(src.read_bytes()).hexdigest(), snapshot=str(dest)))
    (provenance/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(decision, indent=2))
    print(out/"REPORT.md")


if __name__ == "__main__":
    main()
