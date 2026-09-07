# WaterSplatting / Underwater 3DGS 项目完整交接提示词

> 请将下面的内容作为一个完整的项目交接上下文。你还会得到这个项目的 GitHub 代码。你的第一步不是重新设计方法，而是阅读代码、研究笔记和正式实验产物，确认下面的叙述与仓库一致。除非我明确提出新的研究问题，否则请严格遵守本文的研究边界，不要把已经关闭的方向重新包装成创新点。

## 0. 你的角色和首要任务

你现在接手的是一个正在进行中的正式水下 3D Gaussian Splatting 研究项目。请以一名严谨的计算机视觉研究员和代码审计员的身份工作，完成以下任务：

1. 理解原始 WaterSplatting 渲染模型、当前 `research/m1-bounded-intrinsic` 分支以及所有正式因果实验。
2. 区分代码事实、配置事实、实验事实、统计结论、机制归因和研究假设。
3. 正确识别当前保留的方法、当前 baseline 中的辅助项、已经否定的创新候选以及仍然只是 tentative 的诊断结果。
4. 如果继续做研究，必须从现有结论出发，不能因为某个历史输出目录或某段代码仍然存在，就认为该方向仍然有效。
5. 不要仅凭图像“更黄”“更少蓝色”就声称完成了物理去水或真实颜色恢复。

请把 `outputs/` 和 `renders/` 看成大型、可能不完整且未提交到 Git 的实验缓存；把 `research_notes/`、正式 JSON/CSV、训练脚本和模型代码结合起来解释。若 GitHub 代码中看不到 outputs，不要据此否定已经完成的实验。

## 1. 仓库、分支和当前状态

项目本地路径：

```text
/mnt/new/home_old/ycy/water-splatting-refactor
```

当前分支和 HEAD：

```text
branch: research/m1-bounded-intrinsic
HEAD:   f32a7a5
remote: origin/research/m1-bounded-intrinsic
```

最近的重要提交：

```text
e1d139d  Implement direct MDRR and CICA training
4d5b8d5  Fix MDRR partner training-resolution state
f32a7a5  Complete four-scene MDRR and CICA experiment
```

当前没有正式训练进程。GPU 6--9 是正式四场景实验使用的 GPU，目前不应自行启动新的长时间训练。

工作区有几份未跟踪的历史/辅助脚本，包括：

```text
scripts/diagnostics/analyze_raoc_q50_q80_causal_four_scene.py
scripts/diagnostics/render_gmvc_curasao_contact_sheet.py
scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py
scripts/experiments/run_raoc_q50_q80_causal_four_scene.py
scripts/experiments/run_raoc_q50_q80_causal_scene.py
```

这些未跟踪脚本不代表最终方法，也不要未经确认就删除、提交或重新运行。

不要执行以下操作：

```text
git clean
git reset --hard
git checkout -- .
```

不要删除历史正式实验 outputs，不要把 checkpoints、outputs 或 renders 提交到 Git，不要擅自修改 OCMC、RAOC、基础 optimizer、学习率、topology schedule 或正式协议。

## 2. 原始研究问题

原始项目是 WaterSplatting：使用显式 3D Gaussian 表示场景几何与物体外观，使用一个按像素查询的 medium field 表示水体/雾介质，从而同时进行水下重建和介质去除后的 clear/intrinsic rendering。

水下图像的简化物理结构可以理解为：

```text
I(x) = T_D(x) * I_object(x) + I_backscatter(x)
```

其中 `T_D` 是物体光经过介质衰减后的 transmission，`I_backscatter` 是有限深度介质贡献和远端 tail 介质贡献。当前 medium MLP 的 9 个 raw 输出按通道分成：

```text
z[0:3] -> B_inf / medium_rgb = sigmoid(z[0:3])
z[3:6] -> beta_B = softplus(z[3:6] + medium_density_bias)
z[6:9] -> beta_D = softplus(z[6:9] + medium_density_bias)
```

在 `b_inf_mode=tied` 下，`B_inf` 与 `medium_rgb` 绑定。direct object RGB 使用类似 `exp(-beta_D * depth)` 的衰减；finite medium 和 tail 使用 `medium_rgb`、`beta_B` 以及 renderer 的 alpha/compositing 语义。具体实现仍以代码为准，主要入口是：

```text
water_splatting/fields/medium_field.py
water_splatting/water_splatting.py
water_splatting/fields/gaussian_appearance.py
```

研究核心问题不是简单地“让图像少一点蓝色”，而是：

```text
在水下退化、多视角观测不足以及 object appearance / medium 参数耦合的情况下，
如何让 medium branch 和 Gaussian intrinsic appearance 更可观测、更稳定，
同时不破坏 novel-view underwater reconstruction 和 clear rendering？
```

必须把以下概念分开：

- underwater reconstruction：仍包含水体介质的重建质量。
- native clear/intrinsic rendering：关闭或重组介质后的模型输出。
- true clear ground truth：真实、配对、同一相机的无水图像。
- fake air / derived clear rendering：从数据或模型派生出的结果，不能自动当作真实 clear GT。

当前项目没有发现真实 paired clear ground truth。因此 clear/dewatered 只能作定性或分布性讨论，不能报告配对 clear PSNR/SSIM/LPIPS，也不能把颜色变化解释为物理正确性。

## 3. 当前最终研究结论，一句话版

当前推荐方法是：

```text
OCMC + Auxiliary Appearance Regularization + CICA
```

正式矩阵中的名称是：

```text
A2 = BASE + CICA
```

其中：

- OCMC 是已经锁定的第一核心创新/基础机制。
- CICA 是目前推荐的第二核心创新。
- Auxiliary Appearance Regularization 只保留在 baseline 中，不能宣称为第二创新。
- RAOC 已关闭，不属于当前最终方法。
- MDRR 已关闭，不属于当前最终方法。
- `A3 = A0 + MDRR + CICA` 不是 complementary 组合，而是 interfering 组合。

论文或报告中最稳妥的命名是：

```text
OCMC + CICA, with the detached SH-opacity auxiliary appearance regularizer retained in the baseline.
```

不要写成：

```text
OCMC + RAOC + MDRR + CICA
```

也不要把辅助 SH 正则称为 `SH-opacity Identifiability Module`，因为正式机制归因没有支持这一点。

## 4. 当前代码和模型配置

正式 OCMC/CICA/MDRR 实验使用的受控配置是：

```text
intrinsic_color_parameterization = bounded_sh3
SH degree = 3
medium_context_mode = dir_xy_camera
b_inf_mode = tied
infinite_water_enabled = false
rasterization = classic
seed = 42
OCMC = ON
RAOC = OFF
```

`dir_xy_camera` 的 medium 输入包括：

1. 方向编码，代码中是 16-D direction encoding；
2. 图像平面 `x/y/r` context，3-D；
3. scene-normalized camera-center context，3-D。

因此 formal medium input 为 22-D。camera context 不是 learned latent，而是由 camera center、scene-box center 和 scene-box diagonal 计算得到：

```text
camera_context = (camera_center - scene_center) / (scene_scale + 1e-6)
```

Gaussian intrinsic appearance 使用 bounded SH3 时，完整 SH 输出被解释为 RGB logits：

```text
s_i(v) = spherical_harmonics(active_sh_degree, viewdir,
                             [features_dc, features_rest])
c_i(v) = sigmoid(s_i(v))
```

初始化使用颜色等价的 logit：

```text
features_dc = RGB2SHLogits(seed_rgb) = logit(seed_rgb) / C0
features_rest = 0
C0 = 0.28209479177387814
```

bounded SH3 的作用是让每个 view-dependent Gaussian RGB 保持在 `(0, 1)`，消除 unbounded legacy `SH + 0.5` 路径的明显越界/分解风险。它是当前受控 representation baseline，不应在交接时误写成 OCMC 或 CICA。

相关实现：

```text
water_splatting/fields/gaussian_appearance.py
water_splatting/water_splatting.py
water_splatting/sh.py
```

## 5. 第一核心机制：OCMC

### 5.1 全称和目标

OCMC 的全称是：

```text
Observability-Controlled Medium Context
```

它针对的是 camera-conditioned medium context 的可观测性问题。medium MLP 可以根据 camera position 改变 medium raw 输出，但不同相机下的 medium 变化可能包含训练视图支持不足的弱方向。于是 medium branch 可能在 camera context、方向/空间 context 和 Gaussian appearance 之间产生不可观测的补偿。

OCMC 的目标不是删除 camera context，也不是对 beta_D 做全局方差惩罚，而是：

```text
保留 camera-conditioned medium response 中有证据支持的方向，
压制由当前训练观测估计出的低可观测 raw-medium 方向。
```

### 5.2 精确机理

对相同的方向和图像空间输入，medium MLP 做两次前向：

```text
z_full = f(direction, xy, camera_context)
z_base = f(direction, xy, zero_camera_context)
Delta_z_cam = z_full - z_base
```

由训练数据中的 structured RGB Jacobian 或相关诊断构造 detached 的 9x9 projector `P_obs` 和尺度 `S`。标准化后：

```text
Delta_z_std  = Delta_z_cam / S
Delta_z_proj = S * P_obs * Delta_z_std
```

实际代码采用等价的 residual blend：

```text
z_effective = z_base
              + (1 - strength) * Delta_z_cam
              + strength * Delta_z_proj
```

formal OCMC 使用 `strength=1.0`，因此主要路径是：

```text
z_effective = z_base + Delta_z_proj
```

projector、SVD/Jacobian 估计和 projector state 都是 detached 的；OCMC 不给 Gaussian geometry、opacity 或 intrinsic appearance 直接加 loss，也不改变 rasterizer 的 forward compositing 规则。

### 5.3 OCMC 的边界

OCMC：

- 只作用于 medium raw output 的 camera-conditioned residual；
- 不直接修改 Gaussian position、scale、rotation、opacity 或 SH；
- 不直接把 medium 变成真实物理参数；
- 不提供 true medium、true color 或 true geometry 证明；
- projector 不是普通 checkpoint 的持久模型参数，必须由 driver 根据注册协议估计并显式安装；
- 与 RAOC 互斥，不能同时叠加。

代码入口：

```text
water_splatting/fields/medium_field.py
    DirectionConditionedMediumField.__call__
    camera residual / projector path

water_splatting/water_splatting.py
    WaterSplattingModel._predict_medium
    set_camera_medium_observability_projector
```

### 5.4 OCMC 证据和当前地位

camera-context 预审先发现了 expressiveness/ambiguity trade-off，随后完成了 IUI3 的 matched causal OCMC 实验：

```text
final eval C1-C0 PSNR = +0.130078 dB
RGB safety classification = RGB_IMPROVED
start-state equivalence = True
camera sequence match = True
decomposition P(J > 1) = 0
```

机制诊断显示低可观测 component 被减少，但 `correct_context_utility_preserved` 并不总是成立，所以最严谨的早期分类是：

```text
OCMC_PARTIALLY_ACTIONABLE
CAMERA_CONTEXT_CAPACITY_ALLOCATION_TENTATIVE
```

后来 OCMC 被锁定为当前基础机制，所有 MDRR/CICA 正式四场景实验都从各场景 OCMC C0@3000 checkpoint 开始。不要把 OCMC 的 IUI3 causal result 扩写成已经完成的四场景独立 OCMC causal validation。

OCMC temporal mismatch audit 也没有支持“stale projector 就是唯一失败原因”：identity、stale、fresh 之间的 utility 差异很小，fresh-vs-stale gap 约为 `0.000003`，所以不要自行宣称已经解决 projector temporal mismatch。

优先阅读：

```text
research_notes/M1_OCMC_CAUSAL_IUI3_2026-08-25.md
research_notes/CAMERA_MEDIUM_OBSERVABILITY_PREFLIGHT_2026-08-25.md
research_notes/OCMC_PROJECTOR_TEMPORAL_MISMATCH_IUI3_2026-08-25.md
```

## 6. 当前 baseline 中保留的辅助项：Auxiliary Appearance Regularization

### 6.1 正确命名和地位

当前 A0 baseline 是：

```text
OCMC + A_DETACHED_SH_OPACITY_TANGENT_ORTHOGONALIZATION
```

但在论文叙述中只能把后者称为：

```text
Auxiliary Appearance Regularization
```

它不是当前推荐的第二创新。正式 four-scene causal experiment 发现它可能带来小的 RGB 增益，但 mechanism attribution 更支持 generic regularization / optimization trajectory effect，而不是已经被证明的 identifiability correction。

### 6.2 它试图解决什么

Gaussian appearance 的非 DC SH 方向可能在训练视图上产生与 opacity tangent 相近的 RGB response。Gaussian 可能用 view-dependent SH appearance 去补偿 opacity/visibility 变化，从而出现 representation redundancy。

预审阶段用局部一阶响应构造：

- 非 DC bounded-SH Jacobian `J_i`；
- 非 DC response `r_i0`；
- raw opacity tangent `j_alpha_i`；
- `u_i = normalize(j_alpha_i)`；
- `h_i = normalize(J_i^T u_i)`。

在 refresh anchor `theta_i0` 附近，构造使共享方向的一阶响应被抵消的 target：

```text
tau_i = h_i^T theta_i0
        - (u_i^T r_i0) / ||J_i^T u_i||
```

然后使用 detached gate 和标量惩罚：

```text
L_aux = lambda * sum_i g_i * (h_i^T theta_i - tau_i)^2
                         / (2 * sum_i g_i)
```

在当前实际训练实现中，辅助项的直接梯度只进入 `features_rest`，不直接进入 `features_dc`、opacity、geometry、medium MLP、OCMC 或 topology。有效的正交非 DC 方向不被整体删除。

### 6.3 为什么不能把它称为已验证的 identifiability module

Gaussian parameter identifiability audit 发现：

- SH/opacity 局部 observation subspace overlap 在四场景都很高；
- overlap 最大 pair 是 SH-Opacity；
- ambiguity score 与 heldout error 有一定关联，四场景通过了预审支持门槛。

但是，正式 causal training 的 A_DETACHED intervention 只得到：

```text
heldout mean PSNR delta = +0.036275 dB
heldout PSNR improved or tied = 3/4 scenes
```

更深入的 attribution audit 显示：

- shared response energy 在 4/4 场景下降；
- formal tangent overlap 只在 2/4 场景下降；
- non-DC SH energy 在 4/4 场景下降超过 5%；
- view-dependent SH response RMS 在 4/4 场景低于 C0 的 75%；
- registered orthogonal/non-DC capacity gate 在 JapaneseGradens-RedSea 失败；
- 高 ambiguity 区域比低 ambiguity 区域改善更强只在 2/4 场景成立；
- 没有任何场景通过固定的 Gaussian identity-quality gate；
- medium MLP 最终 state 仍然发生约 15%--27% relative L2 drift，虽然 direct module gradient 不进入 medium branch。

因此最合理结论是：

```text
RGB effect = causally observed
claimed identifiability mechanism = not supported
generic regularization / changed optimization trajectory = more likely
```

不要继续 sweep strength、不要做 rescue、不要把它写成第二创新。当前正确处理是保留在 A0/A2 baseline 中，作为一个已固定的 auxiliary appearance regularizer。

相关文件：

```text
research_notes/GAUSSIAN_PARAMETER_IDENTIFIABILITY_AUDIT_2026-09-01.md
research_notes/GAUSSIAN_IDENTIFIABILITY_MODULE_DESIGN_2026-09-01.md
research_notes/IDENTIFIABILITY_MODULE_CAUSAL_EXPERIMENT_2026-09-02.md
research_notes/IDENTIFIABILITY_GAIN_MECHANISM_AUDIT_2026-09-02.md
scripts/experiments/run_identifiability_module_causal_scene.py
```

## 7. 已关闭的 medium 方向：MIC

MIC 的全称/工程名是 medium identifiability / BND-MIC。它试图直接惩罚 camera-conditioned medium 的 raw `beta_D` contextual variance：

```text
L_MIC = mean((z_beta_D - stopgrad(mean(z_beta_D)))^2)
```

实现位于 `WaterSplattingModelConfig.medium_identifiability_enabled` 和 `WaterSplattingModel._medium_identifiability_loss()`。它只在显式打开 flag 且 weight 非零时生效，默认关闭。

IUI3 causal result：

```text
final eval PSNR delta = -0.229170 dB
M_SAFE raw beta_D contextual variance ratio C1/C0 = 4.106
aggregate conditioning improved at some checkpoints
weak-mode ambiguity reduction = not stable
classification = MIC_NOT_ACTIONABLE
```

这说明简单地压低 beta_D 的 raw within-camera variance 会压错自由度：它可能降低局部空间变化，却增加 across-camera variation，并且最终破坏 RGB。不能从 MIC 结果推出“medium identifiability 方向值得继续做”。

正式结论：关闭 beta_D variance regularization line，不 sweep lambda，不设计第二个 beta_D rescue。

相关文件：

```text
research_notes/BND_MIC_CAUSAL_IUI3_2026-08-25.md
research_notes/BND_MEDIUM_IDENTIFIABILITY_PREFLIGHT_IUI3_2026-08-25.md
```

## 8. 已关闭的 RAOC 方向

RAOC 的全称是：

```text
Ray-Adaptive Observability / medium correction
```

它的思想是对每条 ray、每个 medium mode，根据训练数据估计 evidence 和 local scale，选择性保留或压低 medium camera-conditioned modes。形式上包含：

1. 9 个 raw-medium mode 的 basis/spectrum；
2. train-only per-mode median scale；
3. ray-level evidence，例如 `e_i,p = |a_i,p| * s_i,p`；
4. global observability gate、local gate；
5. 类似 `g_keep = 1 - (1 - g_obs) * (1 - g_local)` 的组合；
6. reference、CUDA hybrid 和 CUDA fused runtime backend。

RAOC 与 OCMC 是互斥的。RAOC 不是当前最终方法，当前 formal configuration 一律 `OCMC=ON, RAOC=OFF`。

四场景 causal RAOC 结果：

```text
Curasao:             PSNR -0.312269 dB, utility recovery not supported
IUI3-RedSea:         PSNR +0.139137 dB, capacity reallocation supported
JapaneseGradens:     PSNR +0.170146 dB, capacity reallocation supported
Panama:              PSNR -0.206036 dB, utility recovery not supported
mean PSNR delta:     -0.052255 dB
aggregate:           RAOC_MULTI_SCENE_TENTATIVE
strong-support gate: failed
```

虽然 RAOC 在四场景都恢复了某种正的 correct-context utility，且高 evidence rescue 大于低 evidence rescue，但 RGB safety 和预注册的 multi-scene strong-support criteria 没有通过。之后的 CUDA fusion/training equivalence 主要是工程可行性与数值一致性工作，不能把 runtime equivalence 当成科学有效性。

最终决定：

```text
RAOC = DROP / ARCHIVED
```

不要继续做 Q50/Q80 sweep、lambda rescue、CUDA backend rescue 或把 RAOC 写进最终方法。

相关文件：

```text
research_notes/M1_RAOC_CAUSAL_FOUR_SCENE_2026-08-27.md
research_notes/RAOC_CUDA_FUSION_OPTIMIZATION_2026-08-28.md
research_notes/RAOC_CUDA_TRAINING_EQUIVALENCE_2026-08-29.md
research_notes/RAOC_HYBRID_CUDA_ACCELERATION_2026-08-31.md
```

## 9. 已关闭的 MDRR 方向

MDRR 的全称是：

```text
Multi-view Degradation Responsibility Routing
```

它的目标是把跨视角的 degradation residual responsibility 分给 appearance route 或 medium route。

### 9.1 具体算法

对每个 primary training camera，固定匹配一个 partner training camera。partner mapping 在 A1/A3 中完全一致。每个 active update 都从当前模型 state 重新计算 partner response。

对一对视角，分别计算：

- primary/partner 的 RGB residual；
- full medium response，包括 direct attenuation、finite medium contribution 和 tail medium contribution；
- Gaussian-level responsibility statistics；
- 两视角 responsibility 差异 `Delta_e`；
- 两视角 full degradation response 差异 `Delta_d`。

使用 detached positive cosine：

```text
q_i = max(0, cosine(Delta_e_i, Delta_d_i))
```

通过 classic renderer 的 compositing responsibility，把 Gaussian-level `q_i` 聚合成 pixel responsibility map `g_p`。然后路由梯度：

```text
appearance route = (1 - g_p)
medium route     = g_p
geometry/opacity = original base gradient
```

也就是说，MDRR 不改变 forward RGB；它改变的是交给 optimizer 的梯度。MDRR 从 step 5000 开始。partner 必须使用与 primary 相同的 training resolution schedule；此前 partner full-resolution 与 primary half-resolution 的 bug 已由 `4d5b8d5` 修复。

### 9.2 routing audit 通过不等于科学有效

正式 gradient routing audit 通过了：

- surface route 主要影响 appearance；
- medium route 主要影响 medium branch；
- geometry 不被 MDRR route 直接替换；
- OCMC projector 不变；
- forward finite；
- checkpoint reload 正常；
- `P(J > 1) = 0`。

但四场景最终效果非常差：

```text
Curasao:             -3.651619 dB
IUI3-RedSea:         -1.145608 dB
JapaneseGradens:    -18.071080 dB
Panama:              -4.796481 dB
mean:                -6.916197 dB
improved scenes:      0/4
```

MDRR 还在没有改变 topology schedule 的情况下显著改变 Gaussian population：Curasao、IUI3、Panama 最终数量分别约增加 56%、65%、56%，JapaneseGradens 约减少 24%。这说明 gradient routing 足以改变 split/duplicate/prune 的后续轨迹，并且这种改变与 RGB degradation 同时出现。

最终决定：

```text
MDRR = DROP
```

不要继续 MDRR rescue、threshold sweep、lambda sweep 或新 seed。

实现和正式结果：

```text
scripts/experiments/run_direct_mdrr_cica_scene.py
research_notes/FULL_MEDIUM_DEGRADATION_RESPONSIBILITY_DIAGNOSTIC_2026-09-03.md
research_notes/MULTIVIEW_DEGRADATION_RESPONSIBILITY_PREFLIGHT_2026-09-03.md
```

## 10. 当前第二核心机制：CICA

### 10.1 全称和研究假设

CICA 的全称是：

```text
Cross-view Intrinsic Chromaticity Anchoring
```

它针对的不是笼统的蓝色，而是一个更精确的假设：

```text
Gaussian intrinsic appearance，尤其是远距离 Gaussian 的 DC component，
可能吸收跨相机一致的 residual underwater color cast，
从而把介质造成的 chromatic residual 固化到 intrinsic appearance 中。
```

CICA 的目标是用多个训练视图估计一个 detached canonical intrinsic chromaticity target，再约束 Gaussian 的 intrinsic DC color 朝这个 target 对齐，同时不允许 target 本身被 heldout GT 或手工 blue prior 污染。

### 10.2 CICA 的数据和 camera 选择

CICA 从 step 10000 开始，refresh steps 固定为：

```text
10000, 12000, 14000
```

每次 refresh 最多选择 6 个 camera，规则固定为：

- 只使用 training cameras；
- 按 camera ID 排序；
- 均匀采样；
- 不使用 heldout cameras；
- 不根据 RGB、PSNR、LPIPS、残差大小或“蓝色程度”选 camera。

每次 refresh 都基于当时的 Gaussian population 重建 controller。Gaussian 必须至少在 3 个有效 training views 中有信息，才会得到 target。

### 10.3 从真实 renderer 语义构造 local evidence

CICA 不使用简单的 transmission inversion，例如：

```text
(I - medium) / transmission
```

因为这种公式不能正确处理 Gaussian overlap、alpha compositing、attenuation、finite medium 和 tail response。

CICA 使用实际 classic renderer 的 DC intrinsic local Jacobian：

1. 以当前 frozen-at-refresh 模型对每个 selected training view 做实际 forward；
2. 使用当前 renderer 的 Gaussian projection、depth、conic、opacity、medium attenuation 和 compositing 顺序；
3. 通过 read-only CUDA accumulator 累积每个 Gaussian 对像素 RGB 的 DC-logit Jacobian normal equation；
4. 信息量由 visibility、attenuation、compositing responsibility 和 conditioning 共同决定；
5. 得到每个 view 每个 Gaussian 的 DC correction evidence。

这一步是 read-only/detached controller construction，不是训练 forward 中的额外 learned branch。

### 10.4 log-chroma target

将 corrected intrinsic DC RGB 转成 log-chroma：

```text
u = log((R + eps) / (G + eps))
v = log((B + eps) / (G + eps))
```

在至少 3 个有效 training views 上，使用 information-weighted robust aggregation，当前实现为 weighted median，得到 detached canonical target：

```text
target_log_chroma_i = weighted_median_view(
    corrected_log_chroma_i,
    information_weight_i
)
```

当前 Gaussian 的 DC chromaticity 与 target 的差异为：

```text
Delta_chi_i = current_log_chroma_i - target_log_chroma_i
```

### 10.5 loss、梯度和安全约束

CICA 使用 Huber loss：

```text
L_CICA = weighted_mean(Huber(Delta_chi_i), gate_i * information_i)
```

它只约束 `features_dc`，不直接约束：

- medium MLP；
- camera context；
- OCMC projector；
- opacity；
- geometry；
- `features_rest`。

第一次 activation 时，将 CICA 的 `features_dc` gradient norm calibrate 到该步 photometric `features_dc` gradient norm 的 10%，随后 lambda 固定：

```text
lambda_CICA = 0.10 * ||grad_photometric(features_dc)||
              / ||grad_raw_CICA(features_dc)||
```

不做 lambda sweep，不依据 heldout result 调整 lambda。

方向审计必须允许双向修正：

- `Delta(R/G)` 可以正也可以负；
- `Delta(B/G)` 可以正也可以负；
- 不能把 CICA 变成“所有 Gaussian 都减少蓝色”的单向 color prior。

正式日志要求无 color-prior-collapse warning，且 CICA gradient escape 为 0。

### 10.6 CICA 的代码入口

主要实现都在：

```text
scripts/experiments/run_direct_mdrr_cica_scene.py
    _cica_local_evidence()
    _build_cica_controller()
    _cica_loss()
    _combine_gradients()
```

其中 `_cica_local_evidence()` 调用 read-only CUDA accumulator；`_build_cica_controller()` 生成 detached target；`_cica_loss()` 只对 `features_dc` 产生 loss gradient。

### 10.7 CICA 的正式结果和正确表述

在四场景 matched matrix 中，A2 相对 A0：

```text
Curasao:             +0.135233 dB
IUI3-RedSea:         +0.001946 dB
JapaneseGradens:     +0.029947 dB
Panama:              -0.000969 dB
mean:                +0.041539 dB
improved scenes:      3/4
```

A2 不超过预注册 `-0.05 dB` worsening tolerance，且没有新的明显 underwater reconstruction 失真。四场景 mean metrics：

```text
PSNR  = 29.892863
SSIM  = 0.927376
LPIPS = 0.120011
MSE   = 0.002130
```

相对于 A0，A2 的优势是：

- 小幅、跨场景、可复现的 RGB safety/quality improvement；
- 3/4 场景 PSNR 提升；
- underwater reconstruction 基本保持；
- Gaussian population 与 A0 在每个场景都保持在约 1% 内；
- 所有 final configurations 的 `P(J > 1) = 0`；
- CICA 和 auxiliary gradient escape 均为 0；
- correction directions 不是单向 collapse。

但正确限制是：

```text
CICA 的 native clear blue/cyan 变化是 mild distributional shift，
不是有真实 clear GT 支持的决定性 dewatering 证明。
```

不能写成“CICA 已证明恢复真实无水颜色”。当前安全的科学表述是：

```text
CICA is a training-view-only, renderer-consistent, information-weighted
cross-view chromaticity anchoring mechanism that preserves underwater RGB
and yields a small mean novel-view gain in the registered four-scene matrix.
```

## 11. A3 组合机制为什么失败

A3 定义为：

```text
A3 = A0 + MDRR + CICA
```

也就是 step 5000 后启用 MDRR，step 10000 后启用 CICA。

正式结果：

```text
mean PSNR(A3)       = 26.498558 dB
A3 - A0             = -3.352766 dB
A3 - A2             = -3.394305 dB
```

scene-level A3-A0：

```text
Curasao             -5.922593 dB
IUI3-RedSea         -1.211169 dB
JapaneseGradens     -4.664769 dB
Panama              -1.612532 dB
```

A3 没有保留 A2 的 CICA RGB 优势，特别是 JapaneseGradens 出现明显退化。正确交互分类是：

```text
INTERFERING
```

不能因为“两个模块理论上分别有用”就把 A3 写成 complementary。这里 MDRR 改变了 Gaussian population 和 optimization trajectory，破坏了 CICA 单独成立时的收益。

## 12. 正式四场景矩阵

正式实验目录：

```text
outputs/direct_mdrr_cica_four_scene_20260903/
```

四个 arm：

```text
A0 = historical CURRENT_BASE
     OCMC ON + RAOC OFF + auxiliary appearance regularization

A1 = A0 + MDRR
A2 = A0 + CICA
A3 = A0 + MDRR + CICA
```

四个场景：

```text
Curasao
IUI3-RedSea
JapaneseGradens-RedSea
Panama
```

协议：

```text
每个 arm 从该场景 OCMC C0@3000 checkpoint 开始
seed = 42
final step = 14999
matched updates = 11999
snapshot steps = 5000, 8000, 10000, 13000, 14999
OCMC = ON
RAOC = OFF
```

审计通过：

- 12/12 formal runs completed；
- 所有 worker status 为 0；
- start-state equivalence = true；
- primary camera sequence exact match = true；
- A1/A3 partner mapping match = true；
- A2/A3 CICA bank match = true；
- routing audits 全部通过；
- CICA refresh schedule 审计通过；
- decomposition safety 全部通过；
- 每个 final configuration `P(J > 1) = 0`。

### 12.1 四个指标的 scene mean

| Arm | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| A0 | 29.851324 | 0.927347 | 0.120760 | 0.002151 |
| A1 | 22.935127 | 0.828966 | 0.271514 | 0.066391 |
| A2 | 29.892863 | 0.927376 | 0.120011 | 0.002130 |
| A3 | 26.498558 | 0.912252 | 0.154336 | 0.005287 |

四个 mean metrics 的最优配置都是 A2。

### 12.2 PSNR scene table

| Scene | A0 | A1 | A2 | A3 | A1-A0 | A2-A0 | A3-A0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Curasao | 32.415435 | 28.763816 | 32.550667 | 26.492842 | -3.651619 | +0.135233 | -5.922593 |
| IUI3-RedSea | 30.748682 | 29.603074 | 30.750628 | 29.537513 | -1.145608 | +0.001946 | -1.211169 |
| JapaneseGradens-RedSea | 24.627346 | 6.556266 | 24.657293 | 19.962578 | -18.071080 | +0.029947 | -4.664769 |
| Panama | 31.613832 | 26.817352 | 31.612864 | 30.001301 | -4.796481 | -0.000969 | -1.612532 |
| MEAN | 29.851324 | 22.935127 | 29.892863 | 26.498558 | -6.916197 | +0.041539 | -3.352766 |

### 12.3 final classification

```text
MDRR:       DROP
CICA:       KEEP
A3:         DROP
interaction: INTERFERING
best arm:   A2 / BASE+CICA
```

## 13. 历史候选方向的状态

以下方向曾经被诊断、预审或实现过，但不能自动列入最终方法。

### 13.1 Low-support Gaussian intervention

假设：训练视图支持少的 Gaussian 更可能在 novel view 出错，因此尝试 detached 地压低 low-support Gaussian opacity。

结果：四场景 primary `alpha=0` low-support suppression 的 heldout PSNR 全部下降：

```text
Curasao             -0.006661 dB
IUI3-RedSea         -0.973782 dB
JapaneseGradens     -0.660028 dB
Panama              -0.004952 dB
```

low-support 没有在 0/4 场景改善 novel-view PSNR，也没有通过 matched random/high-support controls。结论：

```text
LOW_SUPPORT_NOT_SUPPORTED
```

关闭 support-aware refinement、reliability weighting 和 uncertainty module 设计。

### 13.2 Observation-aware optimization

该预审对 T1/T2/middle/high-support Gaussian 做 detached opacity、color、scale sensitivity perturbation，尝试判断低观测支持是否更 under-constrained。

结果：只有 2/4 场景出现 T1 sensitivity 高于 high-support，single-factor controls 也不一致，正式 scene pass 为 0/4。

结论：

```text
OBSERVATION_UNDERCONSTRAINED_NOT_SUPPORTED
```

不能据此设计 observation-aware optimization module。

### 13.3 Appearance-medium entanglement

该审计观察跨视角 SH contribution 与 medium contribution 是否反向共变，并控制 depth、tau、transmission、OCMC 等因素。

结果：四场景没有完整、稳定、跨场景的 compensation -> heldout error 机制链；temporal recurrence 为 0/5 checkpoint，AUROC 接近 chance。

结论：

```text
APPEARANCE_MEDIUM_ENTANGLEMENT = NOT_SUPPORTED
```

不要把相关性 proxy 写成 causal disentanglement evidence。

### 13.4 OCMC 后 residual candidate 审计

OCMC residual audit 检查过：

- cross-view intrinsic inconsistency：0/4 supported，关闭；
- geometry-medium coupling：2/4，tentative；
- view-dependent residual appearance：2/4，tentative；
- spatially structured residual：1/4，tentative；
- late-stage Gaussian allocation：0/4，关闭；
- depth/observability conditioned residual：2/4，tentative。

总体结论是：

```text
NO_SINGLE_DOMINANT_FAILURE_MODE
```

其中 Candidate C 曾是最高优先级的 tentative diagnostic，但它不是已验证机制，更不是已实现的第三创新。data-split feasibility audit 只说明未来可以用更大的 geometry-aware heldout split 做诊断，不能当作 Candidate C 已获得支持。

### 13.5 早期 BND、headroom、GMVC、CDEPTH、SeaFree-inspired 方向

项目早期还审计过 bounded SH3、headroom SH3、antialiased rasterization、GMVC、coarse depth、CB-FG/CB-BG、BAP、UNORM、LOSSRESP、CDEPTH、OMVC、pure-water anchor 等方向。它们是研究演化中的 controlled baselines、diagnostics 或失败候选。

当前正式结论不应把这些历史命名全部堆到最终 method 中。特别是：

- bounded SH3 是当前受控 intrinsic representation，主要负责有界 RGB 和分解安全；
- headroom SH3 没有被选为第二核心创新；
- antialiased rasterizer 不是当前 formal classic protocol；
- SeaFree 风格的额外监督不是当前方法；
- `CDEPTH`、`GMVC` 等历史输出不能代替 OCMC/CICA 的正式 causal evidence。

如果需要研究某个历史方向，必须先阅读对应 `research_notes/*.md`，明确其 classification、是否 causal、是否正式、是否已关闭。

## 14. 当前推荐阅读顺序

请按以下顺序阅读，而不是只看文件名猜测方向：

1. `research_notes/DIRECT_MDRR_CICA_FOUR_SCENE_EXPERIMENT_2026-09-03.md`
2. `outputs/direct_mdrr_cica_four_scene_20260903/final_classification.json`
3. `outputs/direct_mdrr_cica_four_scene_20260903/final_metrics.csv`
4. `outputs/direct_mdrr_cica_four_scene_20260903/final_summary.json`
5. `outputs/direct_mdrr_cica_four_scene_20260903/interaction_analysis.json`
6. `outputs/direct_mdrr_cica_four_scene_20260903/decomposition_safety.json`
7. `outputs/direct_mdrr_cica_four_scene_20260903/topology_statistics.csv`
8. `scripts/experiments/run_direct_mdrr_cica_scene.py`
9. `scripts/diagnostics/analyze_direct_mdrr_cica_four_scene.py`
10. `research_notes/M1_OCMC_CAUSAL_IUI3_2026-08-25.md`
11. `research_notes/CAMERA_MEDIUM_OBSERVABILITY_PREFLIGHT_2026-08-25.md`
12. `research_notes/M1_RAOC_CAUSAL_FOUR_SCENE_2026-08-27.md`
13. `research_notes/GAUSSIAN_PARAMETER_IDENTIFIABILITY_AUDIT_2026-09-01.md`
14. `research_notes/GAUSSIAN_IDENTIFIABILITY_MODULE_DESIGN_2026-09-01.md`
15. `research_notes/IDENTIFIABILITY_MODULE_CAUSAL_EXPERIMENT_2026-09-02.md`
16. `research_notes/IDENTIFIABILITY_GAIN_MECHANISM_AUDIT_2026-09-02.md`
17. `research_notes/FULL_MEDIUM_DEGRADATION_RESPONSIBILITY_DIAGNOSTIC_2026-09-03.md`
18. `research_notes/MULTIVIEW_DEGRADATION_RESPONSIBILITY_PREFLIGHT_2026-09-03.md`

然后再阅读：

```text
research_notes/LOW_SUPPORT_CAUSAL_INTERVENTION_PREFLIGHT_2026-09-01.md
research_notes/OBSERVATION_AWARE_OPTIMIZATION_PREFLIGHT_2026-09-01.md
research_notes/APPEARANCE_MEDIUM_ENTANGLEMENT_AUDIT_2026-09-01.md
research_notes/BND_MIC_CAUSAL_IUI3_2026-08-25.md
research_notes/OCMC_RESIDUAL_FAILURE_MODE_AUDIT_2026-08-31.md
research_notes/DATA_SPLIT_FEASIBILITY_AUDIT_2026-08-31.md
```

## 15. 当前项目的因果证据层级

请用下面的层级报告结论：

### Level 1：代码/协议事实

例如：

- OCMC projector 是 detached 9-D raw-medium projector；
- CICA 只作用于 `features_dc`；
- auxiliary 只对 `features_rest` 有 direct module gradient；
- MDRR route 改 optimizer gradient 而不是 forward RGB；
- RAOC 与 OCMC 互斥。

### Level 2：匹配实验事实

例如：

- 同一 start checkpoint；
- 同一 RNG、optimizer/scheduler/scaler state；
- 同一 primary camera sequence；
- partner mapping/CICA bank matched；
- 12/12 runs 完成。

这些支持 A1/A2/A3 的 causal arm-level RGB comparison。

### Level 3：机制诊断

例如：

- routing audit 说明梯度进入哪些 parameter group；
- CICA direction audit 说明是否发生单向 color collapse；
- topology statistics 说明模块如何改变 split/prune trajectory；
- identifiability audit 的 subspace overlap 说明局部 response ambiguity。

这些不能自动等价于物理解释。

### Level 4：物理和真实世界结论

当前没有真实 clear GT，也没有真实 medium parameter GT。因此不能从现有结果推断：

- CICA 恢复了真实 clear color；
- OCMC 输出是真实 beta_D；
- 某个 Gaussian 就是真实物体表面；
- 蓝色减少就是物理正确的去水。

## 16. 未来继续工作的硬边界

如果我让你继续做实验，请先说明属于以下哪一类：

1. 复核已有正式结果；
2. 代码/运行时工程验证；
3. OCMC 或 CICA 的受控复现；
4. 新的、预注册的科学假设诊断；
5. 论文整理或图表生成。

在没有明确新问题时，默认边界是：

- 不修改 OCMC 方程；
- 不开启 RAOC；
- 不启用 MDRR；
- 不重新把 auxiliary regularizer 命名为 identifiability module；
- 不搜索第三个正式创新点；
- 不做 MDRR/CICA 的 lambda/threshold/seed rescue sweep；
- 不用 heldout RGB/PSNR 选择 camera、target、quantile 或 loss weight；
- 不把 fake-air 派生图当成 paired clear GT；
- 不在历史 tentative signal 上直接设计训练模块；
- 不通过修改 topology schedule、optimizer 或学习率来挽救负结果；
- 不将工程 CUDA speed/equivalence 结果冒充科学 efficacy。

如果未来需要重新研究 identifiability，最低限度需要 immutable Gaussian parent/descendant IDs，覆盖 split、duplicate、prune lineage，并报告 ambiguity-stratified tangent/orthogonal updates 和 additive heldout contribution deltas。现有 checkpoint 没有完整 persistent lineage，不能靠 nearest-neighbor post-hoc 分析补回这一缺失。

## 17. 你应如何解释当前论文贡献

当前最稳妥的论文叙事是：

1. Underwater 3DGS 中，camera-conditioned medium context 存在可观测性/容量分配问题。
2. OCMC 在 detached raw-medium residual 上做 observation-controlled projection，减少低可观测 camera-conditioned medium component；在 IUI3 matched causal experiment 中具有正的 novel-view RGB evidence，因此作为第一核心机制锁定。
3. Gaussian intrinsic appearance 可能吸收跨视角水下色偏；CICA 使用真实 renderer compositing semantics 下的信息加权局部 DC evidence，通过多 training views 的 robust log-chroma target 约束 intrinsic DC chromaticity。
4. CICA 在四场景 matched matrix 中保持 underwater reconstruction，并带来小幅 mean RGB gain；其 native clear 变化只能被称为 mild distributional shift，因为没有 paired clear GT。
5. 多个看似合理的替代方向被因果/诊断实验关闭，包括 RAOC、MDRR、low-support suppression、MIC beta_D variance regularization、generic observation-aware optimization 和 SH-medium entanglement module。

不要把论文贡献写成“用了很多模块所以效果更强”。正式结果反而表明 MDRR 和 CICA 组合会互相干扰；该负结果应被如实报告。

## 18. 接手后的回答格式

当你读完代码和实验产物后，请先给出一份审计式总结，至少包含：

```text
1. 当前 branch/HEAD 是否与本文一致
2. 当前最终方法是什么，哪些模块关闭
3. OCMC 的输入、投影位置、梯度/参数边界
4. auxiliary appearance regularization 的实际作用和不能声称的内容
5. CICA 的 target、evidence、loss、梯度路径和安全审计
6. MDRR、RAOC、MIC、low-support 等为什么关闭
7. A0/A1/A2/A3 的协议和正式数字
8. clear rendering 结论的 ground-truth 限制
9. 你准备执行的下一步是否违反本文边界
```

对于任何新的实验建议，请明确写出：

```text
Hypothesis
Intervention
Controls
Heldout usage
Primary metric
Mechanism metric
Stop/close criterion
Why it is not a post-hoc rescue
```

如果某个方向只有相关性、预审或工程可行性证据，请明确写出 `TENTATIVE` 或 `NOT_SUPPORTED`，不要用 `validated innovation`。

## 19. 最终状态摘要

```text
CURRENT_BASE / A0:
    bounded SH3 representation
    + OCMC ON
    + RAOC OFF
    + Auxiliary Appearance Regularization ON

RECOMMENDED FINAL / A2:
    A0 + CICA

FIRST CORE INNOVATION:
    OCMC

SECOND CORE INNOVATION:
    CICA

BASELINE-ONLY AUXILIARY:
    A_DETACHED_SH_OPACITY_TANGENT_ORTHOGONALIZATION

CLOSED:
    RAOC
    MDRR
    A3 = MDRR + CICA
    MIC beta_D variance regularization
    low-support suppression
    observation-aware optimization module
    SH-medium entanglement module

FINAL FOUR-SCENE MEAN PSNR:
    A0 29.851324
    A1 22.935127
    A2 29.892863
    A3 26.498558

FINAL SCIENTIFIC CLAIM:
    OCMC + CICA is the current best registered method.
    CICA provides a small RGB gain and preserves underwater reconstruction.
    Clear/dewatered improvement is only qualitative/distributional.
```

请基于以上状态理解和分析 GitHub 代码。不要重新发明项目历史，不要把诊断代码当成最终算法，不要把小幅 RGB gain 夸大成物理去水证明，也不要在没有新注册假设的情况下继续已关闭方向。
