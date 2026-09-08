# M1 + OCMC / CICA 单因素归因实验

日期：2026-09-08  
仓库：`water-splatting-refactor`  
分支：`research/m1-bounded-intrinsic`  

## 结论

本轮严格按照“单因素归因 + 轻量修正”完成了四场景实验。结论是：当前固定强度的 `M1 + OCMC + CICA` 仍然是 BND-OFF 主线的最佳已验证配置。`OCMC-only schedule` 和 `CICA-only schedule` 都没有超过固定版，之前的 `combined_ramp` 也没有超过固定版，因此本轮没有新的默认训练策略需要保留。

主要风险仍然是 `JapaneseGradens-RedSea`。OCMC schedule 使该场景 PSNR 下降 `0.146399`，CICA schedule 下降 `0.199743`；combined schedule 下降 `0.185862`。`Panama` 在三种 schedule 下均基本稳定，变化范围小于 `0.014` dB PSNR，说明本轮没有为了保护 JapaneseGradens 而明显牺牲 Panama，也没有真正解决 JapaneseGradens 的问题。

因此后续不应继续做联合 schedule 搜索。若继续优化，应针对 JapaneseGradens 的 camera-conditioned medium residual 做更窄的稳定化诊断，并保留固定版作为门槛。

## 固定实验协议

- 四场景：`Curasao`、`IUI3-RedSea`、`JapaneseGradens-RedSea`、`Panama`
- Legacy SH3；`medium_context_mode=dir_xy_camera`，即 M1
- OCMC ON；RAOC OFF；BND OFF；Auxiliary OFF
- seed `42`
- 从各场景相同的 OCMC `C0@3000` checkpoint 继续训练到 step `14999`
- 相同数据划分、相同 camera sequence、相同训练步数和相同评估视图
- 带水体指标使用项目专用 `evaluate_underwater_rgb_metrics.py`
- RGB 协议：`metrics.py` loader、项目 PSNR/SSIM、`lpipsPyTorch` VGG v0.1，精确文件名配对，不 resize、不 crop、不做颜色变换
- 13 个测试视图总数：Curasao 3、IUI3-RedSea 4、JapaneseGradens-RedSea 3、Panama 3

## 四组配置

| 配置 | OCMC | CICA | 说明 |
|---|---|---|---|
| Fixed mainline | 固定 `1.0` | 固定强度，minimum support `3` | 当前主线；复用 `legacy_ocmc_cica_no_auxiliary_20260905` |
| OCMC-only schedule | `0.75 -> 1.0`，2000 steps | 固定强度，minimum support `3` | 只改变 OCMC 调度，CICA 保持主线开启 |
| CICA-only schedule | 固定 `1.0` | `0 -> calibrated`，2000 steps；minimum support `4` | 只改变 CICA 激活/支持策略 |
| Combined schedule | `0.75 -> 1.0`，2000 steps | `0 -> calibrated`，2000 steps；minimum support `4` | 复用上一轮完整保存的联合 schedule |

这里的 `OCMC-only` 是修正后的严格单因素实验。上一轮名为 `ocmc_ramp` 的 compact 结果曾关闭 CICA，因此不能用于 OCMC 的单独归因；本轮没有复用那组错误配置。

## 带水体 RGB 指标

正向方向为 PSNR/SSIM 越高越好，LPIPS/MSE 越低越好。固定主线与新实验均采用同一项目本地评估协议。

### 四场景均值

| 配置 | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| Fixed mainline | 30.031764 | 0.927403 | 0.137838 | 0.002015746 |
| OCMC-only schedule | 29.978931 | 0.927186 | 0.138708 | 0.002125325 |
| CICA-only schedule | 29.899368 | 0.926935 | 0.138729 | 0.002147853 |
| Combined schedule | 30.010046 | 0.927154 | 0.138668 | 0.002122648 |

相对 Fixed mainline：

| 配置 | dPSNR | dSSIM | dLPIPS | dMSE |
|---|---:|---:|---:|---:|
| OCMC-only schedule | -0.052833 | -0.000217 | +0.000870 | +0.000109579 |
| CICA-only schedule | -0.132396 | -0.000468 | +0.000892 | +0.000132108 |
| Combined schedule | -0.021718 | -0.000249 | +0.000830 | +0.000106902 |

### 每个场景

| 场景 | 配置 | PSNR | SSIM | LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| Curasao | Fixed mainline | 32.209305 | 0.955404 | 0.123685 | 0.000862807 |
| Curasao | OCMC-only schedule | 32.247868 | 0.955167 | 0.124321 | 0.000859785 |
| Curasao | CICA-only schedule | 32.020023 | 0.955096 | 0.124670 | 0.000966091 |
| Curasao | Combined schedule | 32.328136 | 0.955575 | 0.124469 | 0.000820341 |
| IUI3-RedSea | Fixed mainline | 31.107620 | 0.911909 | 0.194955 | 0.001414615 |
| IUI3-RedSea | OCMC-only schedule | 31.005108 | 0.912004 | 0.194347 | 0.001452905 |
| IUI3-RedSea | CICA-only schedule | 30.967688 | 0.911577 | 0.194663 | 0.001476967 |
| IUI3-RedSea | Combined schedule | 31.073818 | 0.911945 | 0.194816 | 0.001415081 |
| JapaneseGradens-RedSea | Fixed mainline | 24.636551 | 0.894535 | 0.133297 | 0.005172848 |
| JapaneseGradens-RedSea | OCMC-only schedule | 24.490152 | 0.893999 | 0.135986 | 0.005575368 |
| JapaneseGradens-RedSea | CICA-only schedule | 24.436808 | 0.893347 | 0.135921 | 0.005534304 |
| JapaneseGradens-RedSea | Combined schedule | 24.450689 | 0.893296 | 0.136178 | 0.005643453 |
| Panama | Fixed mainline | 32.173580 | 0.947764 | 0.099413 | 0.000612713 |
| Panama | OCMC-only schedule | 32.172596 | 0.947576 | 0.100177 | 0.000613242 |
| Panama | CICA-only schedule | 32.172954 | 0.947722 | 0.099664 | 0.000614051 |
| Panama | Combined schedule | 32.187542 | 0.947798 | 0.099207 | 0.000611715 |

## 场景诊断

### JapaneseGradens-RedSea

这是本轮唯一明确的主要退化场景。相对固定主线：

- OCMC-only：PSNR `-0.146399`，SSIM `-0.000536`，LPIPS `+0.002689`，MSE `+0.000402520`
- CICA-only：PSNR `-0.199743`，SSIM `-0.001188`，LPIPS `+0.002624`，MSE `+0.000361457`
- Combined：PSNR `-0.185862`，SSIM `-0.001239`，LPIPS `+0.002881`，MSE `+0.000470605`

OCMC 降低早期投影强度会使弱可观测 camera residual 更早参与主任务；CICA 延迟激活和提高 minimum support 也没有缓解该场景，反而使整体结果略差。这支持“问题在 BND-OFF 下的 camera-conditioned residual 稳定性”这一诊断，但还不能说明应该继续降低 OCMC 强度。

### Panama

Panama 没有明显被拖垮：

- OCMC-only：PSNR `-0.000984`，SSIM `-0.000188`，LPIPS `+0.000764`
- CICA-only：PSNR `-0.000625`，SSIM `-0.000042`，LPIPS `+0.000250`
- Combined：PSNR `+0.013962`，SSIM `+0.000034`，LPIPS `-0.000206`

Panama 是保护指标而不是本轮唯一优化目标。它的稳定只说明 schedule 没有造成明显的硬场景崩溃，不能证明 schedule 有整体收益。

## 梯度路由与完整性

8 个新实验都通过了训练前梯度路由审计：

- `GRADIENT_ROUTING_AUDIT = true`
- CICA gradient escape `= 0.0`
- Auxiliary gradient escape `= 0.0`
- 所有新任务均完成 `11999` 次 continuation update
- 每个新场景均保留一个 step `14999` checkpoint
- 每个新场景均保留全部测试视图的 native render
- 没有 `failure.json`

## 可视化结果

本轮为 JapaneseGradens 和 Panama 各保留了固定主线、OCMC-only、CICA-only、combined 四列的带水体和去水体对比图。

![JapaneseGradens underwater comparison](../renders/m1_ocmc_cica_single_factor_20260908/comparison/JapaneseGradens-RedSea_MTN_1098_underwater_comparison.png)

![JapaneseGradens clear comparison](../renders/m1_ocmc_cica_single_factor_20260908/comparison/JapaneseGradens-RedSea_MTN_1098_clear_comparison.png)

![Panama underwater comparison](../renders/m1_ocmc_cica_single_factor_20260908/comparison/Panama_MTN_1539_underwater_comparison.png)

![Panama clear comparison](../renders/m1_ocmc_cica_single_factor_20260908/comparison/Panama_MTN_1539_clear_comparison.png)

视觉上，各 schedule 的带水体图与固定主线接近，没有观察到稳定的新伪影。去水体图在 JapaneseGradens 的上方区域仍有明显黑色/低能量残留；本轮 schedule 没有带来稳定改善，因此不能以肉眼局部差异为理由替换当前指标最佳主线。

## 资产路径

### 新实验 checkpoint

```text
outputs/m1_ocmc_cica_single_factor_20260908/ocmc_only_schedule/<scene>/checkpoints/step-000014999.ckpt
outputs/m1_ocmc_cica_single_factor_20260908/cica_only_schedule/<scene>/checkpoints/step-000014999.ckpt
```

### 新实验 native render

```text
renders/m1_ocmc_cica_single_factor_20260908/<variant>/<scene>/step_14999/<view_id>/A2/{underwater.png,clear.png,clear_raw_display_clamp01.png}
```

### 新实验最终 RGB 评估

```text
outputs/m1_ocmc_cica_single_factor_20260908/<variant>/<scene>/rgb_metrics_final/{per_view.csv,summary.json}
```

### 固定主线与 combined

```text
outputs/legacy_ocmc_cica_no_auxiliary_20260905/<scene>/checkpoints/step-000014999.ckpt
outputs/m1_ocmc_cica_optimization_20260907_combined_preserved/<scene>/combined_ramp/checkpoints/step-000014999.ckpt
```

## 最终决策

- 保留当前 OCMC/CICA 的配置化开关和 schedule 能力，便于后续可复现实验。
- 不采用 `OCMC-only schedule` 作为默认策略。
- 不采用 `CICA-only schedule` 作为默认策略。
- 不采用 `combined schedule` 作为默认策略。
- 当前 BND-OFF 默认主线继续使用固定强度 `M1 + OCMC + CICA, Auxiliary OFF`。
- 后续若继续工作，优先对 JapaneseGradens 做窄范围 residual 稳定性诊断；Panama 继续作为保护指标。
