# CDRC / LSPR / MST 四场景直接训练结果

日期：2026-09-17。结论：本轮三个候选均已实现并训练完毕，但没有一个在本轮设置下同时改善带水新视角指标和远景去水表现。保留原 FSB+CICA 基线；不将这些候选升级为默认配置，不宣称达到 SOTA。

## 范围与配对方式

- 按用户要求直接进入训练，没有以先前诊断未通过作为停止条件。
- 四场景分别使用物理 GPU 6、7、8、9；Python 来自 `conda water_splatting`。未安装依赖、commit/push 或覆盖历史资产。
- 每场景从同一 R0/C0a FSB+CICA step-14999 checkpoint 分叉，恢复模型、优化器、调度器、RNG、refinement 和 OCMC 状态；共享训练相机顺序，保留 BND。
- 独立训练 `CDRC`、`LSPR`、`MST`，另有同配置续训对照 `BASE_A`、`BASE_B`。三个候选没有相互叠加。
- 每路续训 2,000 次更新至 step 16999。共 20 路、40,000 次更新；每路场景评估全部原始测试视图，合计 65 次终态视图评估，覆盖 13 个不同测试视图。
- 这是已有模型的 2k 续训比较，不是从头训练、完整训练预算比较或多种子显著性检验。测试视图不用于辅助监督、挑选 checkpoint 或调参。

LSPR、MST 的完整原始定义未在可用上下文和仓库中找到。本轮已在训练前声明以下工作定义；结果仅适用于这些实现，不代表原始缩写定义已被恢复，也不代表方法新颖性已经成立。

## 实现

| 候选 | 本轮机制 | 实际训练 |
|---|---|---|
| CDRC | 已验证 train/train 跨距离对应，使用透射率加权有效颜色 `J_eff=D/max(S,0.05)`，对两个方向的水下观测做前向 Huber 拟合 | 每 4 步一次，共 500 次非零辅助梯度；辅助更新仅作用于 appearance/medium |
| LSPR | Low-Signal Surface Prior Recovery：以局部高信息 Gaussian 的冻结 DC log-chroma 约束弱观测 Gaussian | 每 4 步一次，共 500 次非零辅助梯度；实际作用于 DC，不提供亮度标签 |
| MST | Medium Scattering Transport：对原生可见 direct-light 近/中/远层施加正值、逐通道的深度相关 PSF | 与原场景联合训练 2,000 步；新增 3 个衰减强度及 2 个核混合参数 |

CDRC/LSPR 的辅助损失权重为 0.05，附加梯度范数最多为对应原梯度的 0.1 倍。CDRC 的 `J_eff` 是有效混合颜色，不等于已识别的真实表面 clear。原 clear buffer 的 CUDA backward 不支持所需上游梯度，因此 CDRC 使用可微的 object RGB 和单位颜色 RGB。

MST 使用 sigma=1、3 原生像素的 Gaussian PSF，散射混合比例上限为 0.3。固定场景参数时 clear 前向公式不变，但联合训练会改变场景参数，终态 clear 因而不保证不变。MST 的 CICA target refresh 仍使用原生 renderer，这是未消除的近似。MST 是可见层卷积，不是完整多重散射；与历史冻结 PSF 实验存在重叠。

实现见 [CDRC](../water_splatting/far_field_recovery.py)、[Panama 低显存重计算](../water_splatting/far_field_recovery_lowmem.py)、[LSPR/MST](../water_splatting/far_field_recovery_next.py)。配置见 [CDRC 协议](FAR_FIELD_DIRECT_TRAINING_PROTOCOL_2026-09-17.md) 和 [LSPR/MST 协议](LSPR_MST_DIRECT_TRAINING_PROTOCOL_2026-09-17.md)。

## 带水新视角指标

以下为四场景等权均值。增量相对 `mean(BASE_A, BASE_B)`；PSNR/SSIM 越高越好，LPIPS 越低越好。全图指标来自原生浮点输出。

| 方法 | PSNR | SSIM | LPIPS | Delta PSNR | Delta SSIM | Delta LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| 初始 step 14999 | 29.842257 | 0.927681 | 0.120738 | 不作为模块收益对照 | | |
| 两次续训对照均值 | 29.812863 | 0.927183 | 0.120810 | 0 | 0 | 0 |
| CDRC | 29.814409 | 0.927169 | 0.120802 | +0.001546 | -0.000014 | -0.000008 |
| LSPR | 29.806196 | 0.927176 | 0.120850 | -0.006667 | -0.000008 | +0.000040 |
| MST | 29.810163 | 0.927228 | 0.120712 | -0.002700 | +0.000045 | -0.000098 |

| 场景 | CDRC Delta PSNR | LSPR Delta PSNR | MST Delta PSNR | BASE_A/B PSNR 绝对差 |
|---|---:|---:|---:|---:|
| Curasao | -0.002123 | -0.002181 | -0.002081 | 0.002961 |
| IUI3-RedSea | -0.009387 | -0.023979 | -0.005064 | 0.002685 |
| JapaneseGradens-RedSea | +0.009353 | +0.003522 | +0.002378 | 0.000123 |
| Panama | +0.008342 | -0.004028 | -0.006031 | 0.000593 |

两次对照的逐场景绝对 PSNR 差均值为 0.001590 dB。CDRC 的平均收益与该量级相当，但不能据此把所有单场景变化都归于噪声：JapaneseGradens/Panama 有正向变化，IUI3 有明显反向变化，抵消后的均值很小。两次重复不足以给出置信区间。

普通续训本身较初始 checkpoint 平均下降约 0.0294 dB。模块收益必须对照同样续训的 BASE_A/B；不能把 INITIAL 到终态的所有变化归因于新增模块。

## 远景与去水图像

远景区域由初始模型固定：高 accumulation、低深度分散、处于训练深度分段的远景范围。以下使用 PNG 的 stride-4 采样，包含 8-bit 量化，不等同于原生全图浮点指标，也不涵盖全部低覆盖黑区。

| 候选 | 近景 Delta PSNR | 中景 Delta PSNR | 远景 Delta PSNR | 远景 clear MAE vs BASE_A |
|---|---:|---:|---:|---:|
| CDRC | +0.011590 | +0.008975 | -0.016959 | 0.000851 |
| LSPR | -0.013669 | -0.014967 | -0.009193 | 0.000603 |
| MST | -0.003387 | -0.001674 | -0.008258 | 0.000566 |

这些数值先在场景内对视图取均值，再对四场景等权平均。对照 BASE_B 相对 BASE_A 的远景 clear MAE 为 0.000209；颜色范围为 [0,1]。三种候选确实改变了图像，但远景平均变化不到一个 8-bit 码值，不能据此宣称视觉效果得到改善。

已检查四场景固定远景拼接，以及 IUI3、JapaneseGradens 的完整视图和固定 x10 差异图。图中前景原有去水效果仍在，远景偏色、模糊和上方黑区没有明显的整体改善；CDRC 的全图微小正收益也没有转化为远景收益。拼接显示 INITIAL、两个续训对照和三个候选，避免将普通续训变化误认为模块效果。

没有真实 clear GT。clear MAE、黑区比例、蓝红差和边缘强度只是变化描述，不是恢复精度。图中不进行曝光或颜色补偿，clear 仅做 [0,1] 显示裁剪。

## 各候选的发现

**CDRC：**跨距离互拟合产生了真实辅助更新，但在当前对应、共享有效颜色近似、权重和预算下，没有稳定消除介质与颜色的补偿。尤其 Curasao 远景 PSNR 下降约 0.05289 dB，IUI3 全图下降约 0.00939 dB。不能把另外两场景的正向变化推广为普遍有效。

**LSPR：**满足局部半径和平面条件的目标支持非常有限。

| 场景 | 弱观测候选 Gaussian | 接受的先验 Gaussian |
|---|---:|---:|
| Curasao | 201504 | 121 |
| IUI3-RedSea | 40327 | 1994 |
| JapaneseGradens-RedSea | 20132 | 39 |
| Panama | 22251 | 36 |

增加损失并没有增加真实观测。IUI3 的先验覆盖最多，也出现最大的全图退化，说明“附近高信息颜色可作为弱观测表面真实颜色”在本轮没有得到支持；不能仅靠扩大传播范围来宣称解决远景问题。

**MST：**四场景 LPIPS 都略有下降，SSIM 都略有上升，但三场景 PSNR 下降，四场景远景均值也下降。最终 sigma=1 核的混合权重为 0.962 至 0.995，表明本轮拟合主要使用窄核；这只是学习参数的事实，不能直接归因于真实介质散射。该结果更接近非常小的感知指标权衡，没有证据表明去水细节恢复或物体/介质解耦已改善。

## 完整性与资产

- 20 路训练全部完成；同场景恢复状态和相机序列一致，所有输入 checkpoint 哈希保持不变。BND 范围检查通过，评估 `P_J_gt_1=0`。
- 四个 MST 终态重新加载模型及 `transport_module.pt` 后，13 视图的 underwater/clear 共 26 张 PNG 与原终态逐像素一致。此处是 PNG 一致，不宣称未保存浮点张量的逐位一致。复现 MST 必须同时加载 sidecar，普通模型 checkpoint 单独加载会漏掉传输模块。
- CDRC 初次执行遭遇记录 wrapper 保留计算图导致的 OOM，修正后重新运行。Panama 后续遇到共享 GPU 显存竞争，改用 activation recomputation 并从原分叉状态重新运行；未改目标函数或接续失败状态。失败日志与成功终态均保留。
- 时间和 PyTorch 峰值显存见完整报告。共享 GPU 负载及分时执行使这些记录不适合作为严格速度排名。

统一资产目录：`outputs/far_field_three_candidates_20260917_v1`，通过符号链接指向保留的原始成功运行目录。

- [完整指标与资源报告](../outputs/far_field_three_candidates_20260917_v1/summary/REPORT.md)
- [全部 13 视图的 26 张全图/远景对比拼接](../outputs/far_field_three_candidates_20260917_v1/summary/index.html)
- [四场景远景总拼接](../outputs/far_field_three_candidates_20260917_v1/summary/visualizations/four_scene_far_crops.png)
- [收益与对照重复差图](../outputs/far_field_three_candidates_20260917_v1/summary/psnr_gain_vs_repeat.png)
- [完成状态](../outputs/far_field_three_candidates_20260917_v1/completion.json) / [配对状态与哈希](../outputs/far_field_three_candidates_20260917_v1/summary/verification.json) / [MST 重新加载记录](../outputs/far_field_three_candidates_20260917_v1/endpoint_replay)

本轮不支持用这三个候选替换当前 FSB+CICA 基线，也没有证明进一步缩小了 BND ON 与 OFF 的指标差距。结论限于上述工作定义和 2k 续训设置；不把结果外推为所有跨距离约束、局部先验或散射模型均无效。
