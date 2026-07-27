# WaterSplatting 下一阶段实验方案
## Rendered-Medium Background Attribution and Background-Excluded Densification

## 1. 实验背景

当前项目已经完成 Backscatter-Consistent $B_\infty$ 第一阶段实验，主要验证了：

- 显式 $B_\infty=A$；
- 背景水体颜色监督；
- foreground transmission-aware reconstruction；
- bounded-residual $B_\infty$；
- independent $B_\infty$。

现有结果说明：

1. 显式 $B_\infty=A$ 与 M1 原有 implicit tail 在数值上基本等价；
2. 背景颜色监督能够改善 $J$ 的蓝色偏差和部分远区 clear leakage；
3. 仅监督 $B_\infty$ 颜色不能阻止开放水体区域继续驱动 Gaussian 生长；
4. foreground transmission-aware loss 能恢复甚至提高 underwater RGB 指标，但会增加远区 accumulation；
5. bounded-residual $B_\infty$ 能提高带水体重建指标，但也增加了颜色与表示自由度；
6. 当前远处蓝色 Gaussian 残留的主要问题已经从“背景颜色定义不合理”转变为“背景区域的表示归属没有被约束”。

因此，下一阶段不再继续搜索 $B_\infty$ residual scale 或 background-color loss 的更多权重，而是围绕以下三项进行实验：

$$
\boxed{
\text{Rendered-Medium Background Supervision}
+
\text{Background Clear-Gaussian Suppression}
+
\text{Background-Excluded Densification}
}
$$

本阶段暂定名称：

## Background-Attributed Gaussian Growth Control

中文名称：

## 背景归属约束的 Gaussian 生长控制

---

# 2. 实验目标

本阶段希望回答以下问题。

## 目标 1：修正背景监督目标

当前背景损失约束：

$$
B_\infty \approx I_{\mathrm{gt}},
$$

但实际进入最终图像的是：

$$
I_{\mathrm{tail}}=W_{\mathrm{tail}}B_\infty.
$$

因此，需要验证直接监督实际 medium contribution 是否更合理。

## 目标 2：直接抑制开放水体区域中的 clear Gaussian contribution

远处蓝色残留主要在去水体输出 $J$ 中可见，因此应直接约束高置信开放水体区域中的：

$$
J_{\mathrm{gaussian}}.
$$

## 目标 3：限制开放水体区域驱动 Gaussian densification

即使 $B_\infty$ 已经能够拟合背景颜色，开放水体像素仍然通过 RGB reconstruction gradient 驱动 Gaussian split 和 duplicate。需要诊断并限制这部分 densification pressure。

## 目标 4：保持带水体重建质量

最终候选应尽量保持或超过 M1：

```text
PSNR  = 31.1314
SSIM  = 0.9120
LPIPS = 0.1750
```

同时以 A3 作为当前最佳单次带水体重建参考：

```text
PSNR  = 31.2954
SSIM  = 0.9144
LPIPS = 0.1753
```

---

# 3. 基础配置

本阶段统一使用：

```text
medium_context_mode = dir_xy_camera
b_inf_mode = tied
B_inf = medium_rgb = A
infinite_water_enabled = False
capacity loss = off
ownership = off
foreground transmission-aware loss = off
hard pruning = off
opacity decay = off
seed = 42
max_iterations = 15000
```

保留以下对照：

```text
M1:
    正式基线

E2/B2:
    tied B_inf
    lambda_background_water_color = 0.005
    当前最佳 leakage reference

A3:
    bounded residual s=0.02
    lambda_background_water_color = 0.005
    当前最佳 reconstruction reference
```

新实验主线不使用 A3 的 bounded residual，避免额外自由度干扰背景归属实验。

---

# 4. Phase A：高精度背景 Mask 重建

## 4.1 当前 Mask 问题

当前背景 mask 平均覆盖约 49%，覆盖范围过宽。远处弱纹理礁石、海床和低对比物体可能被误标为开放水体。

下一轮背景 mask 应优先保证 precision，而不是 coverage。

建议目标：

```text
Mean background coverage: 15%–30%
```

## 4.2 Mask 构建规则

### Pseudo-depth foreground

当前：

$$
M_{\mathrm{fg}}^0=\mathbb I[D_{\mathrm{pseudo}}>0.55].
$$

建议测试：

```text
foreground depth threshold ∈ {0.45, 0.50}
```

降低 threshold 会扩大 foreground，减少 object 被误划为 water。

### Edge exclusion

建议：

```text
RGB gradient threshold   ∈ {0.04, 0.05}
Depth gradient threshold ∈ {0.04, 0.05}
Edge dilation radius     ∈ {5, 7}
```

### Background erosion

建议：

```text
erosion radius ∈ {13, 17}
```

### Top-connected water prior

IUI3-RedSea 当前场景中的开放水体主要从图像上方延伸。建议增加：

```text
保留与图像顶部或两侧边界连通的低 disparity 区域
```

可选实现：

1. 对候选 background mask 做 connected-component analysis；
2. 只保留与图像顶部接触的连通分量；
3. 可保留与左右边界接触且面积较大的分量；
4. 删除内部孤立低 disparity 区域。

### 纹理排除

计算局部亮度方差：

$$
V(p)=\operatorname{Var}_{\mathcal N(p)}[I_{\mathrm{gray}}].
$$

对高纹理区域排除：

```text
local variance window = 7×7 或 11×11
```

### Hit-confidence 排除

若已有可用 `q_hit`：

```text
q_hit > 0.6 的区域不进入 background teacher
```

该条件只用于提升 mask precision，不用于 capacity loss。

## 4.3 Mask 候选

| Mask ID | Depth threshold | Erosion | Edge dilation | Top-connected | q_hit exclusion |
|---|---:|---:|---:|---:|---:|
| M0 | 0.55 | 9 | 3 | 否 | 否 |
| M1 | 0.50 | 13 | 5 | 是 | 否 |
| M2 | 0.50 | 17 | 7 | 是 | 否 |
| M3 | 0.45 | 13 | 5 | 是 | 是 |
| M4 | 0.45 | 17 | 7 | 是 | 是 |

## 4.4 人工检查要求

至少检查：

```text
20 个训练视角
4 个 eval 视角
```

每个视角输出：

```text
RGB
pseudo-depth
water mask
object mask
boundary mask
uncertain mask
water overlay
```

选择标准：

1. 水体 mask 中不应覆盖明显礁石和海床；
2. 允许遗漏部分开放水体；
3. 远处低对比物体优先归入 uncertain；
4. mask 顶部背景区域应保持连续；
5. 最终背景覆盖率建议控制在 15%–30%。

---

# 5. Phase B：Renderer-Consistent Background Supervision

## 5.1 当前问题

当前损失：

$$
\mathcal L_{\mathrm{bg-color}}=|B_\infty-I_{\mathrm{gt}}|
$$

监督的是未乘 tail weight 的 $B_\infty$，但实际渲染尾部为：

$$
I_{\mathrm{tail}}=W_{\mathrm{tail}}B_\infty.
$$

下一阶段应监督真实进入渲染结果的 medium contribution。

## 5.2 Medium-only Background Loss

定义：

$$
I_{\mathrm{medium}}=I_{\mathrm{med}}^{\mathrm{finite}}+I_{\mathrm{tail}}.
$$

其中：

$$
I_{\mathrm{tail}}=W_{\mathrm{tail}}B_\infty.
$$

背景 medium loss：

$$
\mathcal L_{\mathrm{bg-med}}
=
\frac{
\sum_pM_{\mathrm{bg}}(p)
\left\|I_{\mathrm{medium}}(p)-I_{\mathrm{gt}}(p)\right\|_1
}{
3\sum_pM_{\mathrm{bg}}(p)+\epsilon
}.
$$

该损失直接要求：

```text
高置信开放水体区域由 medium contribution 完成重建
```

## 5.3 Tail-only Background Loss

作为辅助消融：

$$
\mathcal L_{\mathrm{bg-tail}}
=
\frac{
\sum_pM_{\mathrm{bg}}(p)
\left\|I_{\mathrm{tail}}(p)-I_{\mathrm{gt}}(p)\right\|_1
}{
3\sum_pM_{\mathrm{bg}}(p)+\epsilon
}.
$$

该形式更强，但可能忽略 finite medium contribution，因此只作为对照。

## 5.4 实验矩阵

固定 tied $B_\infty=A$，使用最佳高精度 mask。

| 编号 | Raw $B_\infty$ loss | Medium-only loss | Tail-only loss |
|---|---:|---:|---:|
| R0 | 0 | 0 | 0 |
| R1 | 0.005 | 0 | 0 |
| R2 | 0 | 0.001 | 0 |
| R3 | 0 | 0.005 | 0 |
| R4 | 0 | 0 | 0.001 |
| R5 | 0.001 | 0.005 | 0 |

优先关注：

```text
R3
R5
```

---

# 6. Phase C：Background Clear-Gaussian Suppression

## 6.1 设计动机

远处 Gaussian 在 underwater RGB 中可能已经被 attenuation 和 backscatter 遮蔽，但在去水体 $J$ 中会被完全显露。

因此，直接约束：

$$
J_{\mathrm{gaussian}}
$$

比约束 accumulation 更接近最终问题。

## 6.2 背景 Clear Loss

定义：

$$
\mathcal L_{\mathrm{bg-J}}
=
\frac{
\sum_pM_{\mathrm{bg}}(p)
\left\|J_{\mathrm{gaussian}}(p)\right\|_1
}{
3\sum_pM_{\mathrm{bg}}(p)+\epsilon
}.
$$

建议使用：

```text
J_gaussian_raw
```

而不是 clamp 后的 $J$，避免高值区域梯度被截断。

## 6.3 延迟与 Ramp

```text
start_step = 3000
ramp_steps = 3000
```

即：

```text
0–3000:
    bg-J off

3000–6000:
    linear ramp

6000–15000:
    full weight
```

## 6.4 权重搜索

在最佳 renderer-consistent background loss 上：

| 编号 | $\lambda_{\mathrm{bg-J}}$ |
|---|---:|
| J0 | 0 |
| J1 | 0.0001 |
| J2 | 0.0005 |
| J3 | 0.001 |
| J4 | 0.002 |

优先：

```text
J1
J2
J3
```

不建议第一轮超过 0.002。

## 6.5 物体保护

```text
M_bg_effective =
    M_bg
    × (1 - M_boundary)
    × (1 - M_uncertain)
```

若使用 `q_hit`：

```text
M_bg_effective =
    M_bg_effective
    × 1[q_hit < 0.4]
```

`q_hit` 只作为 hard exclusion，不参与 loss 强度连续调节。

---

# 7. Phase D：Densification Pressure 诊断

## 7.1 Gaussian 区域归属

对每个可见 Gaussian，将其投影中心采样到：

```text
water
object
boundary
uncertain
```

四类区域。

## 7.2 统计内容

每 500 step 记录：

```text
visible Gaussian count
mean xys gradient
median xys gradient
P90/P95 gradient
gradient > densify_grad_thresh ratio
split candidate count
duplicate candidate count
mean depth
mean opacity
mean scale
mean visibility count
```

按区域分别统计。

## 7.3 关键比例

$$
R_{\mathrm{bg-grad}}
=
\frac{
\sum_{i\in\mathrm{bg}}g_i
}{
\sum_i g_i+\epsilon
}.
$$

$$
R_{\mathrm{bg-candidate}}
=
\frac{
N_{\mathrm{split/dup,bg}}
}{
N_{\mathrm{split/dup,total}}+\epsilon
}.
$$

若：

```text
R_bg_candidate >= 10%
```

则进入 Phase E。

## 7.4 诊断实验

| 编号 | 配置 |
|---|---|
| D0 | M1 |
| D1 | E2/B2 |
| D2 | 最佳 renderer-consistent loss |
| D3 | 最佳 renderer-consistent + bg-J |

---

# 8. Phase E：Background-Excluded Densification

## 8.1 原则

只修改 densification gradient accumulation，不修改：

```text
RGB reconstruction
medium supervision
Gaussian opacity
Gaussian culling
existing Gaussian parameters
```

## 8.2 区域权重

$$
w_{\mathrm{densify}}(p)
=
\begin{cases}
w_{\mathrm{bg}}, & p\in M_{\mathrm{bg}},\\
1.0, & p\in M_{\mathrm{fg}},\\
1.0, & p\in M_{\mathrm{boundary}},\\
0.5, & p\in M_{\mathrm{uncertain}}.
\end{cases}
$$

对每个 Gaussian 投影中心采样：

$$
w_i=w_{\mathrm{densify}}(\pi_i).
$$

修改梯度累计：

$$
g_i^{\mathrm{effective}}=w_i g_i.
$$

## 8.3 实验矩阵

| 编号 | $w_{\mathrm{bg}}$ |
|---|---:|
| F0 | 1.00 |
| F1 | 0.25 |
| F2 | 0.10 |
| F3 | 0.00 |

优先测试：

```text
F1
F2
```

F3 只在背景 mask precision 很高时运行。

## 8.4 不建议立即修改的项目

第一阶段禁止同时改变：

```text
densify_grad_thresh
stop_split_at
cull_alpha_thresh
opacity reset
Gaussian prune
```

---

# 9. 推荐最小实验矩阵

## 9.1 第一批：Mask 与 Renderer-Consistent Loss

| Run | Mask | BG raw | BG medium | BG J | BG densify weight |
|---|---|---:|---:|---:|---:|
| N0 | 当前 M0 | 0.005 | 0 | 0 | 1.0 |
| N1 | 精细 mask | 0.005 | 0 | 0 | 1.0 |
| N2 | 精细 mask | 0 | 0.001 | 0 | 1.0 |
| N3 | 精细 mask | 0 | 0.005 | 0 | 1.0 |
| N4 | 精细 mask | 0.001 | 0.005 | 0 | 1.0 |

## 9.2 第二批：Clear-Gaussian Suppression

在 N3/N4 最佳配置上：

| Run | BG J weight |
|---|---:|
| J0 | 0 |
| J1 | 0.0001 |
| J2 | 0.0005 |
| J3 | 0.001 |

## 9.3 第三批：Densification Gate

在最佳 J 配置上：

| Run | BG densify weight |
|---|---:|
| F0 | 1.00 |
| F1 | 0.25 |
| F2 | 0.10 |
| F3 | 0.00 |

---

# 10. 训练课程

## Stage 1：基础结构建立

```text
0–3000 step
background medium loss = on
background J loss = off
densification gate = off
```

## Stage 2：背景表示分工

```text
3000–6000 step
background J loss linear ramp
densification gate linear ramp from 1.0 to target
```

## Stage 3：联合优化

```text
6000–10000 step
background J full weight
densification gate full weight
normal densification schedule
```

## Stage 4：稳定收敛

```text
10000–15000 step
stop Gaussian split as existing config
continue appearance/medium optimization
```

---

# 11. 指标体系

## 11.1 Underwater Reconstruction

```text
PSNR ↑
SSIM ↑
LPIPS ↓
```

## 11.2 Far-Water Residual

```text
Common Far Accum ↓
Common Far Clear ↓
Water-region J ↓
J Blue Dominance ↓
J Green Dominance ↓
```

## 11.3 Background Attribution

```text
BG medium-only RGB error ↓
BG tail RGB error ↓
BG J Gaussian luma ↓
BG accumulation ↓
```

## 11.4 Object Retention

```text
Object Accumulation Retention ↑
Object J Retention ↑
Boundary Gradient Retention ↑
```

## 11.5 Densification Diagnostics

```text
BG gradient fraction
BG split candidate fraction
BG duplicate candidate fraction
Gaussian count by depth
Gaussian count by mask region
```

---

# 12. 成功标准

相对 M1：

## 带水体重建

```text
PSNR drop <= 0.05 dB
SSIM drop <= 0.001
LPIPS increase <= 0.001
```

理想状态：

```text
PSNR >= 31.20
SSIM >= 0.9135
LPIPS <= 0.1755
```

## 远区残留

```text
Far Clear 降低至少 25%
Water J 降低至少 25%
J Blue Dominance 降低至少 20%
```

## 物体保留

```text
Object J Retention >= 0.97
Boundary Retention >= 0.95
```

## Densification

```text
BG split/duplicate candidate fraction 明显下降
Total Gaussian count 不显著膨胀
```

---

# 13. 代码修改建议

## 13.1 Config

新增：

```python
lambda_background_medium_render: float = 0.0
lambda_background_tail_render: float = 0.0
lambda_background_clear_gaussian: float = 0.0

background_clear_loss_start_step: int = 3000
background_clear_loss_ramp_steps: int = 3000

background_densification_enabled: bool = False
background_densification_weight: float = 1.0
uncertain_densification_weight: float = 0.5
background_densification_start_step: int = 3000
background_densification_ramp_steps: int = 3000
```

## 13.2 Outputs

确保输出：

```text
rgb_medium_finite
rgb_tail
rgb_medium_total
J_gaussian_raw
tail_weight_last
background_region_mask
densification_region_weight
```

其中：

```python
rgb_medium_total = rgb_medium_finite + rgb_tail
```

## 13.3 Loss

```python
bg_medium_loss = (
    bg_mask * torch.abs(rgb_medium_total - gt_rgb)
).sum() / (bg_mask.sum().clamp_min(1e-6) * 3.0)

bg_j_loss = (
    bg_mask * torch.abs(J_gaussian_raw)
).sum() / (bg_mask.sum().clamp_min(1e-6) * 3.0)
```

## 13.4 Densification

在 `after_train()` 中：

1. 采样每个 Gaussian 投影中心的 region weight；
2. 将 `grads` 乘对应 weight；
3. 再累计到 `xys_grad_norm`；
4. 同时记录加权前后的 gradient diagnostics。

```python
weighted_grads = grads * region_weight
```

第一版使用 projection-center sampling 即可，不必立即实现完整 pixel-to-Gaussian contribution aggregation。

## 13.5 新增脚本

```text
scripts/diagnostics/build_high_precision_water_masks.py
scripts/diagnostics/diagnose_densification_regions.py
scripts/experiments/bg_attribution_medium_loss_iui3.sh
scripts/experiments/bg_attribution_clear_gaussian_iui3.sh
scripts/experiments/bg_excluded_densification_iui3.sh
```

---

# 14. 结果表模板

| Run | PSNR | SSIM | LPIPS | Far Accum | Far Clear | Water J | J Blue | BG Medium Err | BG J | Obj Ret | Boundary Ret | BG Candidate % | Gaussian Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| E2 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| N3 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| J2 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| F1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| F2 |  |  |  |  |  |  |  |  |  |  |  |  |  |

---

# 15. 决策路径

## 情况 A：Renderer-consistent loss 已显著改善

若 N3/N4 显著优于 E2，说明原 background $B_\infty$ supervision 与实际 renderer 不一致是主要问题。

最终保留：

```text
B_inf = A
+
rendered-medium background supervision
```

## 情况 B：BG J loss 才明显改善

若 J1–J3 才显著减少残留，说明远处 Gaussian 在 underwater RGB 中贡献弱，但在 clear branch 中贡献强。

最终应保留 small-weight background clear suppression。

## 情况 C：Densification gate 才明显改善

若 F1/F2 显著减少远区残留，说明开放水体区域确实持续驱动 Gaussian 生长。

最终模块应包括：

```text
background-attributed densification control
```

## 情况 D：Densification gate 仍无效

需要进一步检查：

```text
残留是否主要来自初始化 Gaussian
残留 Gaussian 是否早于 mask/gate 生效
残留是否来自 existing opacity optimization 而非 densification
```

此时再考虑：

```text
从 step 0 启用 gate
background opacity gradient gate
高置信背景 Gaussian 的 soft opacity regularization
```

暂不直接 hard prune。

---

# 16. 最终预期方法

若实验成功，最终模块可定义为：

## Background-Attributed Infinite-Water Representation

由以下三部分组成：

$$
\boxed{B_\infty=A}
$$

$$
\boxed{
\mathcal L_{\mathrm{bg-med}}
+
\mathcal L_{\mathrm{bg-J}}
}
$$

$$
\boxed{
\text{Background-Excluded Gaussian Densification}
}
$$

其核心含义是：

```text
开放水体颜色由 medium / B_inf 解释
开放水体中的 clear Gaussian contribution 被直接限制
开放水体不再持续驱动 Gaussian split 和 duplicate
物体和边界区域仍保留完整 Gaussian 表达能力
```

该方案不再依赖旧 M2 的 ownership 与 capacity suppression，而是从渲染监督和 Gaussian 生长路径两端共同建立 scene-medium attribution。
