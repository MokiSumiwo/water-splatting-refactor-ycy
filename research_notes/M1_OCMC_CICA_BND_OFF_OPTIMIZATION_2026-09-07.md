# M1 + OCMC / CICA: BND-OFF 优化实验记录

日期：2026-09-07  
仓库：`water-splatting-refactor`  
分支：`research/m1-bounded-intrinsic`  
实验协议：Legacy SH3、M1 (`dir_xy_camera`)、OCMC ON、CICA ON、Auxiliary OFF、RAOC OFF、BND OFF、seed 42、从 `step-3000` 继续训练到 `step-14999`。

## 结论摘要

1. 当前固定强度的 `M1 + OCMC + CICA` 仍然是 BND-OFF 主线中最好的已验证方案。本轮没有发现能够在四场景整体均值上超过它的调度优化。
2. OCMC 从 `0.75` 线性升到 `1.0` 的 schedule 在 JapaneseGradens-RedSea 上明显不利，说明过早放松 OCMC 投影会重新暴露 camera-conditioned medium residual 的不稳定性。该 schedule 不保留为默认配置。
3. CICA 从 `0` ramp 到校准强度、并把最小多视图支持从 3 提高到 4，没有带来稳定收益；CICA 5% 梯度占比的额外四场景实验同样没有收益。该方向不继续扩大搜索。
4. `combined_ramp` 的保留 checkpoint 版本在本地 `metrics.py` 协议下相对当前主线的四场景均值变化为：PSNR `-0.021718`、SSIM `-0.000249`、LPIPS `+0.000830`、MSE `+0.000106902`。因此它只能作为诊断实验，不应替换当前主线。
5. Panama 没有被本轮 combined schedule 明显拖垮：本地协议下 PSNR `+0.013962`、SSIM `+0.000034`、LPIPS `-0.000206`、MSE `-0.000000998`。但整体均值被 JapaneseGradens 的下降抵消，Panama 仍是需要单独监控的硬场景。

## 两条主线的实际机理

### M1 + OCMC

M1 将 camera center 经过场景 AABB 中心和尺度归一化后，作为 medium MLP 的额外 camera context，与 ray direction、图像坐标等输入共同预测介质输出。它允许介质表示随相机位置变化，从而表达水下成像中与视点相关的 attenuation/backscatter 变化；它不是简单把所有水体强行压成一个全局常量。

OCMC 的作用是控制 M1 中最难由多视图观测约束的 camera-conditioned residual。对同一方向特征，代码额外计算去除 camera context 的 neutral output，并定义：

```text
camera_delta = medium(camera_context) - medium(neutral_context)
```

训练前由固定校准状态得到一个 detached 的 9x9 observability projector。OCMC 将 residual 投影到可观测子空间，并在 residual space 中混合：

```text
effective = neutral
          + (1 - strength) * camera_delta
          + strength * projected_camera_delta
```

投影矩阵、谱信息和尺度都不参与反向传播，OCMC 也与 RAOC 互斥。这样不是删除 M1 的 camera context，而是减少弱可观测 residual 把颜色、介质和视角变化互相泄漏的自由度。

本轮为 OCMC 增加了配置化强度调度：

- `camera_medium_observability_strength`
- `camera_medium_observability_strength_start`
- `camera_medium_observability_strength_ramp_start_step`
- `camera_medium_observability_strength_ramp_steps`

默认值保持原行为：起始强度 `1.0`、目标强度 `1.0`、ramp steps `0`。因此没有改变现有主线默认结果，也没有修改 BND 或 RAOC 开关。

### CICA

CICA 只作用于 Gaussian 的 `features_dc`，不直接更新 geometry、medium MLP 或 SH residual。它在固定 camera bank 上进行以下操作：

1. 对每个 bank view 计算当前预测与 GT 的 residual。
2. 使用 direct compositing Jacobian 估计 Gaussian 的颜色修正证据和信息量。
3. 将 degree-zero intrinsic color 转换为两个 log-chromaticity 分量：`log(R/G)` 和 `log(B/G)`。
4. 按 Jacobian information 做跨视图加权 robust median，形成 detached 的 chromaticity target。
5. 只有在足够多视图支持的 Gaussian 上施加 Huber loss，并通过校准把 CICA 的 `features_dc` 梯度控制在 photometric gradient 的固定比例内。

本轮为 CICA 增加了三个配置项：

- `CICA_MIN_SUPPORT`：最小有效视图支持数，默认 `3`。
- `CICA_RAMP_STEPS`：CICA 激活后的线性强度 ramp，默认 `0`。
- `CICA_ENABLED`：显式开关，默认 `True`。

CICA 的梯度路由审计在所有完成的实验中通过：CICA 有非零 `features_dc` 梯度，向其他参数组的逃逸梯度为 `0.0`。这些修改服务于 CICA 的跨视图锚定稳定性，没有新增第三个创新点。

## 实验设计

所有候选均从同一四场景 OCMC `C0@3000` checkpoint 开始，使用相同数据划分、相同训练步数和相同 seed。正式实验使用 10 张卡并行完成；后续为了保存可复现资产，又对最接近主线的 `combined_ramp` 做了四场景 non-compact 复跑。

候选 schedule：

| 变体 | OCMC | CICA | 目的 |
|---|---|---|---|
| `ocmc_ramp` | `0.75 -> 1.0`，2000 steps | OFF | 检查 OCMC 早期强度是否过强 |
| `cica_ramp` | 固定 `1.0` | `0 -> calibrated`，2000 steps；minimum support 4 | 检查 CICA 激活是否过急、弱支持是否污染 target |
| `combined_ramp` | `0.75 -> 1.0`，2000 steps | `0 -> calibrated`，2000 steps；minimum support 4 | 联合检查两条 schedule |
| `cica_half_strength` | 固定 `1.0` | 固定 schedule；gradient fraction `5%` | 检查 CICA 校准强度是否过大 |

其中 `cica_half_strength` 已完成四场景实验，但结果不支持保留，因此没有进入实验 wrapper 的可选变体。

## 带水体 RGB 指标

### 当前主线与保留版 combined 的最终协议

最终指标以项目专用 `scripts/diagnostics/evaluate_underwater_rgb_metrics.py` 为准。该脚本调用 `metrics.py` 的 RGB loader、PSNR、SSIM 和 `lpipsPyTorch` VGG v0.1，按精确文件名配对，不 resize、不 crop，逐视图平均。13/13 个视图配对成功。

基准是当前已验证的 `M1 + OCMC + CICA, Auxiliary OFF, BND OFF` 渲染结果；候选是 `combined_ramp` 的保留 checkpoint 复跑。

| 场景 | 基准 PSNR | combined PSNR | dPSNR | 基准 SSIM | combined SSIM | dSSIM | 基准 LPIPS | combined LPIPS | dLPIPS | 基准 MSE | combined MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | 32.209305 | 32.328136 | +0.118832 | 0.955404 | 0.955575 | +0.000171 | 0.123685 | 0.124469 | +0.000784 | 0.000862807 | 0.000820341 |
| IUI3-RedSea | 31.107620 | 31.073818 | -0.033802 | 0.911909 | 0.911945 | +0.000036 | 0.194955 | 0.194816 | -0.000139 | 0.001414615 | 0.001415081 |
| JapaneseGradens-RedSea | 24.636551 | 24.450689 | -0.185862 | 0.894535 | 0.893296 | -0.001239 | 0.133297 | 0.136178 | +0.002881 | 0.005172848 | 0.005643453 |
| Panama | 32.173580 | 32.187542 | +0.013962 | 0.947764 | 0.947798 | +0.000034 | 0.099413 | 0.099207 | -0.000206 | 0.000612713 | 0.000611715 |
| **四场景 mean** | **30.031764** | **30.010046** | **-0.021718** | **0.927403** | **0.927154** | **-0.000249** | **0.137838** | **0.138668** | **+0.000830** | **0.002015746** | **0.002122648** |

### Compact schedule 的诊断结果

下面的结果来自 worker 内部评估，仅用于比较 schedule 趋势；特别是 LPIPS 使用了 Nerfstudio model 内部实例，与项目本地 `metrics.py` 的 LPIPS 数值不一致，所以不能和上表混用。

| 变体 | PSNR mean | SSIM mean | LPIPS mean（内部协议） | MSE mean |
|---|---:|---:|---:|---:|
| 当前固定 OCMC+CICA（本地协议） | 30.031764 | 0.927403 | 0.137838 | 0.002015746 |
| `ocmc_ramp` | 29.913139 | 0.927444 | 0.120317 | 0.002117030 |
| `cica_ramp` | 29.980573 | 0.927525 | 0.119445 | 0.002078435 |
| `combined_ramp` | 30.001814 | 0.927765 | 0.119065 | 0.002094779 |
| `cica_half_strength` | 29.953604 | 0.927613 | 0.119123 | 0.002112668 |

PSNR/MSE 的趋势已经足以表明这些 schedule 没有整体胜出；最终 LPIPS 判定使用上面的本地复评，而不是该表的内部 LPIPS。

## Panama 与视觉检查

Panama 在 `combined_ramp` 本地协议下没有明显下降，四项变化分别为：PSNR `+0.013962`、SSIM `+0.000034`、LPIPS `-0.000206`、MSE `-0.000000998`。因此本轮没有证据表明 combined schedule 专门破坏 Panama，但也没有证据表明它解决了 Panama 的主要风险。

JapaneseGradens 是本轮 combined schedule 的主要失败场景：PSNR 下降 `0.185862`，SSIM 下降 `0.001239`，LPIPS 上升 `0.002881`，MSE 上升 `0.000470605`。这与 `ocmc_ramp` 在该场景上的下降方向一致，说明降低 OCMC 早期投影强度会使该场景重新受到弱可观测 camera residual 的影响。

保留版 combined 的四场景去水体对比图：

![combined ramp dewatering comparison](../renders/m1_ocmc_cica_optimization_20260907_combined_comparison/current_dewatering_comparison.png)

原当前主线的去水体对比图：

![current mainline dewatering comparison](../renders/current_dewatering_cica_no_auxiliary_20260905/current_dewatering_comparison.png)

原始 PNG 资产中，每个场景有 `underwater.png`、`clear.png` 和 `clear_raw_display_clamp01.png`。去水体图不是额外的后处理增强，而是 native WaterSplatting renderer 的输出；保存时只做 `[0,1]` clamp 和 PNG 编码。`clear_raw_display_clamp01.png` 只是 raw clear object 输出的显示版本，不是新的训练信号。

## 代码与输出位置

代码改动：

- [water_splatting/water_splatting.py](../water_splatting/water_splatting.py)：OCMC strength schedule 配置和 forward 混合。
- [scripts/experiments/run_direct_mdrr_cica_scene.py](../scripts/experiments/run_direct_mdrr_cica_scene.py)：CICA 支持阈值、ramp、开关，以及 OCMC schedule 传递和日志记录。
- [scripts/experiments/run_m1_ocmc_cica_optimization.py](../scripts/experiments/run_m1_ocmc_cica_optimization.py)：BND-OFF、Auxiliary-OFF 的三种 schedule 实验 wrapper。

候选 `combined_ramp` 的正式 checkpoint：

```text
outputs/m1_ocmc_cica_optimization_20260907_combined_preserved/<scene>/combined_ramp/checkpoints/step-000014999.ckpt
```

四场景渲染图：

```text
renders/m1_ocmc_cica_optimization_20260907/<scene>/step_14999/<view_id>/A2/{underwater.png,clear.png,clear_raw_display_clamp01.png}
```

本地协议复评：

```text
outputs/m1_ocmc_cica_optimization_20260907_combined_preserved/<scene>/combined_ramp/rgb_metrics/{per_view.csv,summary.json}
```

本轮 compact 实验的完整审计输出位于：

```text
outputs/m1_ocmc_cica_optimization_20260907_retry2/
```

生成的 checkpoint、PNG、logs 和 outputs 都被仓库 `.gitignore` 忽略，没有将大文件加入代码提交。

## 最终决策

- 保留：OCMC/CICA 的配置化 strength、ramp、support 和 enabled 开关，因为它们是两条现有创新线的可复现实验能力，默认值不改变当前主线。
- 不采用：`ocmc_ramp`、`cica_ramp`、`combined_ramp`、CICA 5% strength 作为默认训练策略。
- 不新增：第三个创新点、RAOC 主线或 BND 主线。
- 后续重点：如果继续优化，应优先针对 BND-OFF 下 JapaneseGradens 的 camera residual 稳定性，并同时验证是否能提升去水体自然度；Panama 继续作为硬场景保护指标。任何新 schedule 都必须用同一 `metrics.py` 协议复评，尤其不能直接用内部 LPIPS 替代最终 LPIPS。
