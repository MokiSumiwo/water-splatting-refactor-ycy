# C1/C2 四场景扩展结果

> 2026-10-26 更正：组合前审计发现，历史 C2 的占据项反向传播会向 renderer 的
> `xys_grad_abs` 累加额外统计，从而间接改变后续增密。下文的 checkpoint 和指标
> 仍是真实测量，但 C2 不能解释为仅改变 medium/opacity 梯度的干净消融，
> “占据项带来更稳定收益”的机制归因暂不成立。旧入口保留以复现历史结果；
> 修正入口隔离该统计副作用，并重新进行四场景单模块检验。
> 详见 [AC_VALIDATION_20261026.md](AC_VALIDATION_20261026.md)。

评估日期：2026-10-25。C1/C2 在 IUI3-RedSea 的结果沿用已完成的
`research_runs/C_open_water_20261024/formal/IUI3-RedSea/`；本轮新增
Curasao、JapaneseGradens-RedSea、Panama 三个场景，每个场景各运行 C1、C2
一条 seed 42、15,000 步的正式训练。

## 结论

六条新增正式任务均完成 15,000 步，exit code 为 0；六条 smoke 任务均通过。
四个场景的测试视角指标先在场景内取平均，再对四个场景等权平均。最终宏平均如下：

| 分支 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MSE ↓ |
|---|---:|---:|---:|---:|
| B0 | 29.898367 | 0.928811 | 0.139581 | 0.002221705 |
| A(0.1) | 29.999254 | 0.929202 | 0.139924 | 0.002038364 |
| C1 | 29.992036 | 0.928943 | 0.140295 | 0.002128582 |
| C2 | **30.036043** | **0.929403** | **0.139923** | **0.002091762** |

C1 相对配对 B0 的变化为 PSNR `+0.093669 dB`、SSIM `+0.000132`、LPIPS
`+0.000714`、MSE `-0.000093123`。C2 的变化为 PSNR `+0.137676 dB`、SSIM
`+0.000592`、LPIPS `+0.000342`、MSE `-0.000129943`。因此在四场景带水测试指标上，
C2 是本轮两个分支中更好的选择；它的 PSNR 高于 A(0.1) `0.036789 dB`，SSIM 高
`0.000201`，LPIPS 几乎相同（低 `5.3×10^-7`），但 MSE 仍高
`0.000053398`。PSNR 与逐视角 MSE 的场景平均不是可互相代换的量，因此同时报告两者。

表中的 B0、C1、C2 取各自正式运行的 `evaluation_metrics.csv` 15,000 步记录；
A(0.1) 取已冻结的 `geometry_gradient_A010_20261022_v2/three_arm_metrics.json`。
两者都遵循先场景内测试视角平均、再四场景等权平均的口径。

## 逐场景结果

以下均为 15,000 步、原分辨率 float32 测试指标；`Δ` 表示 C 分支减去同场景 B0。

| 场景 | B0 PSNR | C1 PSNR | C1 Δ | C2 PSNR | C2 Δ |
|---|---:|---:|---:|---:|---:|
| Curasao | 32.822168 | 32.900867 | +0.078700 | 32.816741 | -0.005426 |
| IUI3-RedSea | 30.551130 | 30.865563 | +0.314433 | 30.946317 | +0.395187 |
| JapaneseGradens-RedSea | 24.642320 | 24.685232 | +0.042912 | 24.734905 | +0.092585 |
| Panama | 31.577850 | 31.516483 | -0.061367 | 31.646210 | +0.068359 |

C1 的收益主要来自 IUI3-RedSea，Panama 有小幅下降。C2 在三个场景提升，
Curasao 仅下降 `0.0054 dB`，说明占据项在本轮宏平均中带来更稳定的 PSNR/SSIM
收益。LPIPS 并非每个场景都同步改善：JapaneseGradens-RedSea 的 C1/C2 分别比
B0 高 `0.001878/0.000946`，而 Panama 的 C2 比 B0 低 `0.000341`。

## 实验与掩码范围

- C1 在冻结的开放水域先验内，对原生 tied `B_inf == medium_rgb` 的 masked
  Charbonnier 残差施加 medium 梯度。
- C2 在 C1 基础上，对同一 mask 内的 `-log(final_transmittance)` 施加 opacity
  梯度。
- 两组附加梯度每四步施加，step 504–9,996 有记录；对应参数组的附加梯度上限为
  原生梯度范数的 5%，10,000 步之后关闭，最后 5,000 步为原生训练。
- Curasao mask 为 18 张训练图中的 8 张，每个有效图占 `0.3460%` 像素；
  JapaneseGradens-RedSea 为 17 张中的 11 张，每个有效图占 `1.8522%`；Panama
  为 15 张中的 3 张，每个有效图占 `0.3529%`。未选训练图使用显式空 mask，保留
  未知区域并跳过 C 附加梯度；测试图没有建立 C mask。

正式日志中的 C 触发统计为：

| 场景 | C1 medium 非零 | C2 medium 非零 | C2 opacity 非零 |
|---|---:|---:|---:|
| Curasao | 1061 | 1061 | 44 |
| IUI3-RedSea | 2374 | 2374 | 299 |
| JapaneseGradens-RedSea | 1534 | 1534 | 507 |
| Panama | 496 | 496 | 473 |

每个分支均有 2,374 条计划触发记录，范围为 step 504–9,996；空 mask 只跳过
辅助项，不跳过原生 RGB 更新。所有 checkpoint 均能重载，模型哈希匹配，测试指标
有限，诊断脚本与训练导出的指标逐视角一致（LPIPS 的复算差异在既定容差内）。

## 解释边界

这是四场景、单 seed 的探索性结果。C 的监督只作用于人工审查后冻结的小型训练图
开放水域区域，mask 外仍是未知 U，没有独立 clear ground truth 或 O/F 稠密标签。
因此，宏平均带水指标的提升不能直接解释为全图水体清除，也不能据此计算远景残留
减少比例。Panama 只有 3/15 张训练图有有效区域，结论尤其受稀疏覆盖限制。C2 的
改进更适合作为继续做独立 seed、扩大可信开放水域证据和逐视角稳定性复核的依据，
而不是最终的全场景去水证明。

## 复现与结果文件

- 执行方案：[OPEN_WATER_C_FOUR_SCENE_EXECUTION_20261025.md](OPEN_WATER_C_FOUR_SCENE_EXECUTION_20261025.md)
- 新增正式训练：[C_four_scene_20261025/formal](../research_runs/C_four_scene_20261025/formal)
- 新增终点评估：[C_four_scene_20261025/diagnostics](../research_runs/C_four_scene_20261025/diagnostics)
- IUI3 正式训练：[C_open_water_20261024/formal/IUI3-RedSea](../research_runs/C_open_water_20261024/formal/IUI3-RedSea)
- A(0.1) 参考：[three_arm_metrics.json](../research_runs/geometry_gradient_A010_20261022_v2/three_arm_metrics.json)
