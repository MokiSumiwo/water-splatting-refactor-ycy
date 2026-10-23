# BND ON：M1 + OCMC + CICA 单因素归因结果（2026-09-14）

本轮完成了 BND ON 条件下 CICA 的严格单因素对照。实验从相同的 step-3000 OCMC C0 checkpoint 继续训练到 step 14999，使用相同的 `bounded_sh3`、M1 `dir_xy_camera`、OCMC 强度 1.0、seed 42、相机序列、优化器和 refinement 逻辑。仅 JapaneseGradens-RedSea 与 Panama 具有可核验的共同 step-3000 起点，因此本结果不外推到 Curasao 或 IUI3-RedSea。

## 实验分支

| 条件 | BND | M1 | OCMC | CICA | Auxiliary / RAOC |
|---|---|---|---|---|---|
| CICA OFF | ON | `dir_xy_camera` | ON | OFF | OFF / OFF |
| CICA ON | ON | `dir_xy_camera` | ON | ON | OFF / OFF |

两组均完成 11999 次 continuation update，最终步为 14999，保留 5000、10000、14999 checkpoint。梯度路由审计均通过，CICA ON 的 `cica_gradient_escape_l2=0`，OCMC projector 起止 hash 相同（训练期间未被意外修改）。

## 最终评估结果

所有指标均严格筛选 `absolute_step == 14999`、`split == eval`，每个场景 3 个测试视图。PSNR/SSIM 越高越好，LPIPS/MSE 越低越好。

### 场景均值

| 场景 | 条件 | PSNR | SSIM | LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| JapaneseGradens-RedSea | CICA OFF | 24.735882 | 0.895452 | 0.117185 | 0.005362990 |
| JapaneseGradens-RedSea | CICA ON | 24.796497 | 0.895493 | 0.117406 | 0.005193337 |
| Panama | CICA OFF | 31.426553 | 0.947028 | 0.077559 | 0.000726216 |
| Panama | CICA ON | 31.501145 | 0.947138 | 0.077803 | 0.000713910 |
| **两场景宏平均** | **CICA OFF** | **28.081218** | **0.921240** | **0.097372** | **0.003044603** |
| **两场景宏平均** | **CICA ON** | **28.148821** | **0.921315** | **0.097604** | **0.002953623** |
| **ON − OFF** |  | **+0.067603 dB** | **+0.000076** | **+0.000233** | **−0.000090980** |

CICA 在两个严格配对场景的 PSNR 都上升：JapaneseGradens `+0.060614 dB`，Panama `+0.074592 dB`。SSIM 和 MSE 同方向改善，但 LPIPS 分别增加 `+0.000221` 与 `+0.000244`，说明像素误差略降并不对应感知距离改善。两场景宏平均的 PSNR 增益为 `+0.067603 dB`，小于 BND OFF 历史主线中 CICA 相对 M1+OCMC 的 `+0.115898 dB`，约为其 58%。

## 与 BND OFF 的关系

BND OFF 的四场景固定主线 `M1 + OCMC + CICA` 历史均值为 PSNR `30.031764`、SSIM `0.927403`、LPIPS `0.137838`、MSE `0.002015746`。本轮 BND ON 两场景宏平均为 PSNR `28.148821`，但数据集和场景集合不同，不能作为严格四场景差值。它只能说明：即使在相同 M1/OCMC/CICA 组合下，BND ON 仍存在明显的指标容量差距；CICA 的 ON/OFF 增益在 BND ON 中仍为正，但不足以弥合该差距。

## 可视化检查

由于原始训练脚本的两个分支曾共享同一 `ACTIVE_RENDER_ROOT`，首次生成的差异图被后运行分支覆盖，不能用于比较。随后从各自保留的 step-14999 checkpoint 进行只读重新渲染，使用独立目录：

- [JapaneseGradens 全部测试视图拼接图](../outputs/bnd_on_cica_ablation_20260914/visualizations/JapaneseGradens-RedSea_cica_all_views_montage.jpg)
- [Panama 全部测试视图拼接图](../outputs/bnd_on_cica_ablation_20260914/visualizations/Panama_cica_all_views_montage.jpg)

每个视图包含 underwater：CICA OFF、CICA ON、放大差异图，以及 clear raw display（仅作 intrinsic radiance 诊断）：CICA OFF、CICA ON、放大差异图。clear 没有真实 clear GT，因此不能用“更亮、蓝色更少或黑区更少”证明恢复更准确。差异图显示 CICA ON 确实改变了较大范围的颜色响应，尤其 Panama 的 clear raw 变化明显；这与 LPIPS 小幅变差相一致，不能把它解释为单纯的视觉提升。

## 结论与下一步

1. 在已有严格配对的两个 BND ON 场景中，CICA 对 underwater PSNR 有稳定但较小的正向作用（宏平均 `+0.067603 dB`）。因此 CICA 不是 BND ON 下完全无效的模块。
2. CICA 的 BND ON 增益小于 BND OFF 历史增益，且 LPIPS 轻微退化；当前证据不足以声称它改善了感知质量或去水体可视化质量。
3. 不能据此宣称四场景结论。若需要完整归因，下一步应为 Curasao 与 IUI3-RedSea 从共同 BND ON step-3000 起点重新生成 CICA OFF/ON 配对，并保持本轮所有设置不变。
4. 在补齐四场景前，不应通过增点、改颜色或曝光调节放大当前小幅 PSNR 增益，也不应把 clear 代理指标当作真实清晰图像监督。

## 复现实验资产

- [汇总 JSON](../outputs/bnd_on_cica_ablation_20260914/summary.json)
- [最终指标 CSV](../outputs/bnd_on_cica_ablation_20260914/final_metrics_summary.csv)
- [JapaneseGradens CICA ON 训练摘要](../outputs/bnd_on_cica_ablation_20260914/JapaneseGradens-RedSea/cica_on/final_summary.json)
- [Panama CICA ON 训练摘要](../outputs/bnd_on_cica_ablation_20260914/Panama/cica_on/final_summary.json)
- 重新渲染目录：`renders/bnd_on_cica_ablation_20260914_rerender/`
