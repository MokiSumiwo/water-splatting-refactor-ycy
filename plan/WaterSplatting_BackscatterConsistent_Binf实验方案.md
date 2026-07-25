# WaterSplatting 下一阶段实验方案
## Backscatter-Consistent Infinite-Water Closure

## 1. 实验背景

当前项目已经完成 M1、M2、M2 Phase 2/2.5/3、Closed-Tail 以及 Dual-Color Appearance 等多轮实验。

现阶段的主要结论为：

1. M1 的 `dir_xy_camera` medium predictor 能够明显提升带水体新视角重建质量；
2. 旧 M2 中的 ownership、capacity regularization 与 `rgb_mix` 能够减少部分远处蓝色 Gaussian 残留，但会损害 PSNR、LPIPS、物体保留或训练稳定性；
3. Closed-tail 的数值拆分是正确的，但直接与旧 M2 capacity 组合后没有形成稳定正收益；
4. Post-hoc DC/SH 分解不能改善 clear appearance，说明蓝绿色残留并非主要由高阶 SH 色度造成；
5. SeaFree-GS 的关键并不是学习两套彼此独立的外观，而是将 Gaussian color 视为 intrinsic appearance，通过 attenuation 与 backscatter 生成 underwater appearance，并使用同一个 ambient/backscatter color 同时表示有限距离后向散射颜色和无穷远水体背景。

因此，下一阶段不恢复旧 M2 的 ownership、capacity loss 或完整 RGB 替换，而是在 M1 基础上重新设计：

## Backscatter-Consistent Infinite-Water Closure

中文名称：

## 后向散射一致的无限水体闭合

核心假设为：

\[
\boxed{B_\infty(\mathbf p)=A(\mathbf p)}
\]

其中 \(A\) 为 medium predictor 输出的后向散射渐近颜色。有限距离后向散射与无穷远水体尾部共享同一个颜色，只由后向散射系数控制其随距离接近渐近颜色的速率。

---

# 2. 研究目标

本阶段希望回答以下问题：

1. 显式引入与后向散射颜色一致的 \(B_\infty\)，是否能够减少开放水体区域被 Gaussian 表示的问题？
2. 仅将 \(B_\infty=A\) 显式化是否足够，还是必须增加开放水体背景监督？
3. M1 的 `dir_xy_camera` 整体 MLP 是否过于自由，是否需要拆分为低自由度 base medium 与 bounded context residual？
4. 远处蓝色 Gaussian 的剩余部分是否主要由开放水体区域持续驱动 densification 产生？
5. 上述机制能否在不使用旧 M2 capacity suppression 的情况下，同时改善 underwater reconstruction、far Gaussian residual、clear appearance 以及 object/boundary retention？

---

# 3. 物理模型定义

## 3.1 Medium 参数

Medium predictor 输出：

\[
A(\mathbf p)\in[0,1]^3,
\]

\[
\beta^B(\mathbf p)>0,
\]

\[
\beta^D(\mathbf p)>0.
\]

其中：

- \(A\)：后向散射渐近颜色；
- \(\beta^B\)：后向散射增长系数；
- \(\beta^D\)：直接透射衰减系数。

## 3.2 Gaussian 退化

对 Gaussian \(i\)：

\[
c_i^{\mathrm{uw}}
=
c_i\odot\exp(-\beta_i^D d_i)
+
A_i\odot\left[1-\exp(-\beta_i^B d_i)\right].
\]

## 3.3 无限水体闭合

定义：

\[
\boxed{B_\infty(\mathbf p)=A(\mathbf p)}
\]

尾部贡献为：

\[
I_{\mathrm{tail}}=W_{\mathrm{tail}}(\mathbf p)B_\infty(\mathbf p).
\]

最终渲染：

\[
I=I_{\mathrm{obj}}+I_{\mathrm{med}}^{\mathrm{finite}}+W_{\mathrm{tail}}B_\infty.
\]

本阶段不采用：

```text
rgb_mix
m_inf
m_inf_eff
alpha-depth ownership
accumulation-zero loss
near-zero loss
hard pruning
```

---

# 4. 模块设计

## 4.1 Backscatter-Consistent \(B_\infty\)

### Tied 模式

\[
B_\infty=A.
\]

这是默认主方案，不额外增加 MLP 输出通道。

### Bounded Residual 模式

为验证完全绑定是否过强，可定义：

\[
B_\infty
=
\sigma\left[\operatorname{logit}(A)+s_\infty\tanh(\Delta B)\right].
\]

其中：

```text
s_inf ∈ {0.02, 0.05}
```

并加入：

\[
\mathcal L_{\mathrm{tie}}=\|B_\infty-A\|_1.
\]

不建议将 independent \(B_\infty\) 作为主方法，只保留为负面对照。

## 4.2 Background-Water Supervision

使用 pseudo-depth 构建高置信开放水体区域：

\[
M_{\mathrm{bg}}.
\]

在该区域直接约束：

\[
\mathcal L_{\mathrm{bg-color}}
=
\frac{
\sum_pM_{\mathrm{bg}}(p)\|B_\infty(p)-I_{\mathrm{gt}}(p)\|_1
}{
3\sum_pM_{\mathrm{bg}}(p)+\epsilon
}.
\]

由于 \(B_\infty=A\)，该损失也等价于直接监督后向散射渐近颜色。

建议第一轮：

```text
lambda_bg_color = 0.005
```

局部搜索：

```text
lambda_bg_color ∈ {0.001, 0.005, 0.01}
```

## 4.3 Foreground Transmission-Aware Reconstruction

对前景像素定义：

\[
T_c(p)=\exp[-\beta_c^D(p)d(p)].
\]

构造：

\[
w_c(p)=1+\lambda_T M_{\mathrm{fg}}(p)[1-T_c(p)]^\gamma.
\]

约束：

\[
w_c\le4.
\]

建议：

```text
lambda_T ∈ {0.5, 1.0}
gamma = 1.0
detach weight = True
```

重建损失：

\[
\mathcal L_{\mathrm{fg-rec}}
=
\operatorname{mean}\left[w_c|I_{\mathrm{pred},c}-I_{\mathrm{gt},c}|\right].
\]

该损失用于增强远处强衰减物体，尤其是红通道的有效梯度。

## 4.4 Base–Residual Medium Predictor

### Base 分支

只输入世界坐标 LOS direction：

\[
\Theta_{\mathrm{base}}(\mathbf v)=f_{\mathrm{base}}(\mathbf v).
\]

输出：

\[
A_{\mathrm{base}},\quad
\beta_{\mathrm{base}}^B,\quad
\beta_{\mathrm{base}}^D.
\]

### Context 分支

输入：

\[
(\mathbf v,x,y,r,\mathbf o),
\]

可选加入深度上下文 \(d_{\mathrm{ctx}}\)。输出：

\[
\Delta A,\quad\Delta\beta^B,\quad\Delta\beta^D.
\]

最终：

\[
\operatorname{logit}(A)
=
\operatorname{logit}(A_{\mathrm{base}})
+s_A\tanh(\Delta A),
\]

\[
\log\beta^B
=
\log\beta_{\mathrm{base}}^B
+s_B\tanh(\Delta\beta^B),
\]

\[
\log\beta^D
=
\log\beta_{\mathrm{base}}^D
+s_D\tanh(\Delta\beta^D).
\]

初始建议：

```text
s_A = 0.05
s_B = 0.10
s_D = 0.10
```

可选约束：

\[
\mathcal L_{\mathrm{ctx}}
=
\|\Delta A\|_1+\|\Delta\beta^B\|_1+\|\Delta\beta^D\|_1.
\]

建议：

```text
lambda_ctx = 1e-4
```

第一轮优先只使用幅度限制，不立即加入该损失。

## 4.5 Background-Excluded Densification

若背景监督后仍有远处 Gaussian，可进一步控制其生成过程。

定义 densification 权重：

\[
w_{\mathrm{densify}}(p)=M_{\mathrm{fg}}(p)+\lambda_bM_{\mathrm{boundary}}(p).
\]

开放水体区域不参与：

```text
split gradient accumulation
duplicate gradient accumulation
```

但仍参与：

```text
RGB reconstruction
medium supervision
```

第一阶段不使用 opacity decay、extra culling 或 hard prune。

---

# 5. Pseudo-Depth 与背景 Mask

## 5.1 深度输入

建议使用 DepthAnything 生成 relative disparity-like map：

```text
near → large
far  → small
```

每张图归一化至 \([0,1]\)。

## 5.2 前景提取

初步前景：

\[
M_{\mathrm{fg}}^0=\mathbb I[D_{\mathrm{pseudo}}>\tau_d].
\]

再执行：

```text
largest connected component
hole filling
small component removal
```

## 5.3 开放水体背景

定义：

\[
M_{\mathrm{bg}}^0=1-M_{\mathrm{fg}}^0.
\]

然后排除：

```text
RGB edge
pseudo-depth edge
image border
uncertain transition region
```

最后腐蚀：

```text
erosion radius ∈ {5, 9, 13}
```

目标是高 precision，而不是高 coverage。

## 5.4 Mask 诊断

正式训练前必须可视化：

```text
underwater RGB
pseudo-depth
foreground mask
background mask
boundary mask
background overlay
```

至少人工检查全部训练图像或均匀抽样 20 张。

---

# 6. 实验分阶段设计

## Phase A：\(B_\infty\) 关系验证

固定：

```text
medium_context_mode = dir_xy_camera
SH = 3
capacity losses = off
ownership = off
seed = 42
max_iterations = 15000
```

| 编号 | \(B_\infty\) 定义 | Background supervision | 目的 |
|---|---|---:|---|
| A0 | 当前 M1 implicit tail | 关闭 | M1 基线 |
| A1 | 显式 \(B_\infty=A\) | 关闭 | 数值重构检查 |
| A2 | \(B_\infty=A\) | 开启 | 主候选 |
| A3 | bounded residual 0.02 | 开启 | 小自由度 |
| A4 | bounded residual 0.05 | 开启 | 中等自由度 |
| A5 | independent \(B_\infty\) | 开启 | 负面对照 |

### A0/A1 数值要求

```text
mean absolute RGB difference < 1e-6
max absolute RGB difference < 1e-5
```

若不满足，优先检查 tail 定义和现有 renderer 中 `medium_rgb` 的具体物理角色。

## Phase B：Background Supervision 权重

以 A2 为基础：

| 编号 | \(\lambda_{\mathrm{bg-color}}\) |
|---|---:|
| B0 | 0 |
| B1 | 0.001 |
| B2 | 0.005 |
| B3 | 0.01 |

筛选重点：

```text
far accumulation
water J luma
far clear luma
PSNR / SSIM / LPIPS
background-mask RGB error
```

## Phase C：Medium Predictor 结构

在最佳 B 配置下：

| 编号 | Predictor |
|---|---|
| C0 | `dir_only` |
| C1 | `dir_xy_camera` |
| C2 | `dir_xy_depth_camera` |
| C3 | direction base + bounded `xy_camera` residual |
| C4 | direction base + bounded `xy_depth_camera` residual |

重点比较：

```text
C1 vs C3
C2 vs C4
```

若 C3/C4 在 RGB 指标相近时显著减少远区 Gaussian，则采用 base–residual 结构。

## Phase D：Foreground Transmission-Aware Loss

在最佳 C 配置下：

| 编号 | \(\lambda_T\) | Max weight |
|---|---:|---:|
| D0 | 0 | 1 |
| D1 | 0.5 | 4 |
| D2 | 1.0 | 4 |

建议主候选：

```text
D1
```

若 D2 改善远区颜色但明显损害 LPIPS，则不继续提高权重。

## Phase E：Pseudo-Depth Geometry Constraint

增加：

\[
\mathcal L_{\mathrm{depth}}
=
1-ho\left(D_{\mathrm{pseudo}},\frac{1}{d_{\mathrm{render}}+\epsilon}\right).
\]

测试：

| 编号 | \(\lambda_{\mathrm{depth}}\) |
|---|---:|
| E0 | 0 |
| E1 | 0.05 |
| E2 | 0.10 |

该损失只约束相对顺序，不使用绝对尺度。

## Phase F：Background-Excluded Densification

只在 Phase A–E 后仍有明显远区 Gaussian 时测试：

| 编号 | 背景 densification |
|---|---|
| F0 | 正常 |
| F1 | 背景区域不累计 split gradient |
| F2 | 背景区域不累计 split/duplicate gradient |

暂不额外改变 opacity reset、culling threshold 或 hard prune。

---

# 7. 训练课程

建议从头训练，而不是在 M1 checkpoint 上后处理。

## Stage 1：Scene–Medium 分工建立

```text
0–3000 steps
SH = 0
B_inf = A
background supervision = on
base medium only
context residual = off
```

## Stage 2：释放上下文水体变化

```text
3000–8000 steps
SH = 1
context residual scale: 0 → target
background supervision = on
foreground transmission weighting = on
```

## Stage 3：恢复视角相关外观

```text
8000–15000 steps
SH gradually increases to 3
bounded context residual remains active
low-rate joint optimization
```

---

# 8. 最小优先实验集

| 编号 | 配置 |
|---|---|
| E0 | 当前 M1 |
| E1 | 显式 \(B_\infty=A\)，无新增监督 |
| E2 | E1 + background water supervision |
| E3 | E2 + foreground transmission-aware loss |
| E4 | E3 + base–residual medium predictor |
| E5 | E4 + pseudo-depth loss |
| E6 | E5 + background-excluded densification |

当前阶段暂不将多 seed 稳定性作为主要筛选条件，统一使用：

```text
seed = 42
```

最终候选补一个额外 seed 做基本确认即可。

---

# 9. 指标体系

## 9.1 Underwater Reconstruction

```text
PSNR ↑
SSIM ↑
LPIPS ↓
```

## 9.2 Far-Water Gaussian Residual

```text
common far accumulation ↓
common far clear luma ↓
water-region J luma ↓
background Gaussian count ↓
```

## 9.3 Clear Appearance

```text
J Blue Dominance ↓
J Green Dominance ↓
Object J Retention ↑
Boundary Retention ↑
```

## 9.4 Medium Diagnostics

```text
A / B_inf visualization
beta_bs visualization
beta_attn visualization
background-mask B_inf error
B_inf - A residual
context residual magnitude
```

## 9.5 FUNA

有条件时增加：

```text
Clear PSNR ↑
Clear SSIM ↑
Clear LPIPS ↓
CIEDE2000 ↓
Far-depth color error ↓
```

---

# 10. 成功标准

相对 M1，最终候选至少满足：

## 重建质量

```text
PSNR 不低于 M1 超过 0.05 dB
SSIM 不下降超过 0.001
LPIPS 不增加超过 0.001
```

## 远区残留

```text
far clear luma 降低至少 20%
water J luma 降低至少 20%
```

## 物体保留

```text
Object J Retention >= 0.97
Boundary Retention >= 0.95
```

## 水体参数

```text
背景区域 B_inf 与 observed water color 更一致
context residual 不成为主导分支
B_inf 与 A 的差异受到限制
```

---

# 11. 代码修改建议

## 11.1 `water_splatting/fields/medium_field.py`

建议新增输出：

```python
@dataclass
class MediumFieldOutput:
    rgb: Tensor
    bs: Tensor
    attn: Tensor
    b_inf: Tensor
    base_rgb: Optional[Tensor]
    base_bs: Optional[Tensor]
    base_attn: Optional[Tensor]
    context_residual: Optional[Tensor]
```

新增配置：

```python
b_inf_mode: Literal[
    "implicit",
    "tied",
    "bounded_residual",
    "independent",
] = "tied"

b_inf_residual_scale: float = 0.02
```

## 11.2 `water_splatting/water_splatting.py`

新增：

```python
lambda_background_water_color: float = 0.0
lambda_foreground_transmission_reconstruction: float = 0.0
lambda_pseudo_depth: float = 0.0
lambda_medium_context_residual: float = 0.0
medium_predictor_mode: Literal[
    "single",
    "base_residual",
] = "single"
```

## 11.3 Datamanager

加载：

```text
pseudo_depth
foreground_mask
background_mask
boundary_mask
```

建议离线生成并保存，不在训练中重复计算。

## 11.4 Rasterizer / Output

确保显式输出：

```text
rgb_object
rgb_medium_finite
tail_weight
B_inf
rgb_tail
rgb_final
```

## 11.5 Densification

新增可选 pixel-to-Gaussian gradient gate：

```text
densification_support_mask
```

背景区域不累计 densification 统计。

---

# 12. 实验记录模板

| Experiment | PSNR | SSIM | LPIPS | Far Accum | Far Clear | Water J | J Blue | Obj Ret | Boundary Ret | BG B_inf Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 |  |  |  |  |  |  |  |  |  |  |
| E1 |  |  |  |  |  |  |  |  |  |  |
| E2 |  |  |  |  |  |  |  |  |  |  |
| E3 |  |  |  |  |  |  |  |  |  |  |
| E4 |  |  |  |  |  |  |  |  |  |  |
| E5 |  |  |  |  |  |  |  |  |  |  |
| E6 |  |  |  |  |  |  |  |  |  |  |

---

# 13. 决策路径

## 情况 A：E2 已显著改善

最终模块可简化为：

```text
M1
+
B_inf = A
+
Background Water Supervision
```

## 情况 B：E4 才显著改善

说明 `dir_xy_camera` 的整体 MLP 过于自由，需要：

```text
direction base
+
bounded context residual
```

## 情况 C：E6 才减少远处 Gaussian

说明剩余残留主要由开放水体区域持续驱动 densification。

## 情况 D：所有配置均无明显收益

重新审查：

```text
pseudo-depth mask accuracy
current finite-medium formulation
tail weight definition
Gaussian geometry initialization
input white balance
```

不再继续增加 ownership 或 capacity suppression。

---

# 14. 最终预期方法

理想的下一阶段方法为：

\[
\boxed{\text{Backscatter-Consistent Infinite-Water Closure}}
\]

由以下三部分构成：

\[
\boxed{B_\infty=A}
\]

\[
\boxed{\text{Background-Water Supervision}}
\]

\[
\boxed{\text{Base–Residual Context-Aware Medium Prediction}}
\]

可选增加：

\[
\boxed{\text{Background-Excluded Gaussian Densification}}
\]

该方案不再通过训练后抑制 Gaussian accumulation 来清理水体，而是从训练早期明确规定：

```text
开放水体颜色由 B_inf / backscatter field 解释
真实物体由 Gaussian 表示
上下文 MLP 只能做受限修正
背景水体不驱动 Gaussian 生长
```

这比旧 M2 更接近 SeaFree-GS 中有效的归纳偏置，同时保留 M1 在 medium context 建模方面的重建优势。
