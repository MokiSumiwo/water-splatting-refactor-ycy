# WaterSplatting M2 与去水体偏色联合优化实验方案

## 1. 实验背景

当前项目基于原始 WaterSplatting 进行模块化重构，并已完成 M1 至 M4 的独立消融。现阶段实验结论较为明确：

- **M1: Context-Aware Medium Modeling** 能够稳定提升带水体新视角重建质量，同时减少部分由 Gaussian 分支错误吸收的远处水体表征，应作为后续优化的基础模块。
- **M2: Infinite-Water Ownership** 能够进一步削弱甚至接近清除远处纯水体区域中的 Gaussian 残留，但当前实现会轻微降低带水体新视角重建指标，需要继续优化。
- **M3: Contribution-Aware Gaussian Cleanup** 在当前 IUI3-RedSea 场景中没有发现足够可靠的硬清理候选，暂不作为主线。
- **M4: Constrained View-Dependent Appearance** 的原始设计动机与早期错误的去水体图像解释有关，当前不继续沿用。

当前 IUI3-RedSea 的主要结果如下：

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | J Blue Dominance ↓ |
|---|---:|---:|---:|---:|
| Baseline | 29.8790 | 0.9105 | 0.1810 | 0.4174 |
| M1 | 31.1314 | 0.9120 | 0.1750 | 0.1691 |
| M1 + M2 | 31.0696 | 0.9129 | 0.1771 | 0.0514 |

结果表明，M2 已经显著改善去水体图像中远处蓝绿色残留，但相较 M1，PSNR 和 LPIPS 略有下降。因此，后续目标不是继续增强 M2 的删除强度，而是重新设计 M2 的归属、合成与容量控制方式，使其同时满足以下要求：

1. 减少远处纯水体 Gaussian 残留；
2. 不降低带水体新视角重建质量；
3. 不误伤真实远景物体和弱纹理区域；
4. 为后续去水体固有颜色恢复提供稳定的场景支撑。

除 M2 外，当前去水体结果还存在明显的远处偏蓝、偏绿问题。SeaFree-GS 的结果表明，将 Gaussian 固有颜色与水下退化颜色显式分离，有助于获得更加稳定的真实外观。因此，本方案将 M2 优化与固有颜色建模联合考虑，但在实验上分阶段验证，避免一次性引入过多变量。

---

## 2. 当前问题诊断

### 2.1 当前 M2 的核心机制

当前 M2 使用 accumulation 与归一化深度构造无限远水体归属：

\[
M_{\infty}
=
(1-A)^{\gamma}
\cdot
\sigma
\left(
\frac{\tilde d-\mu_d}{\tau_d}
\right),
\]

其中：

- \(A\) 为 Gaussian accumulation；
- \(\tilde d\) 为当前视图内归一化深度；
- \(\gamma\) 为低 accumulation 证据的幂指数；
- \(\mu_d\) 与 \(\tau_d\) 控制远深度 sigmoid。

在 occupancy-limited 模式下，又会额外乘以一次 \(1-A\)：

\[
M_{\infty}^{\mathrm{eff}}
=
M_{\infty}(1-A).
\]

当前最终图像采用：

\[
I
=
(1-M_{\infty}^{\mathrm{eff}})I_{\mathrm{near}}
+
M_{\infty}^{\mathrm{eff}}B_{\infty}.
\]

这种实现能够在低 accumulation、远深度区域中用 \(B_{\infty}\) 替换原始渲染结果，因此可以显著减弱远处 Gaussian 残留。但是，它存在三个主要问题。

### 2.2 完整图像混合会误伤真实远景物体

当前 M2 用 \(B_{\infty}\) 对完整的 `render.rgb` 做图像空间凸组合。对于远处真实物体，只要其 accumulation 较低，就可能被归入无限远水体区域。这会同时削弱：

- 真实 Gaussian 物体贡献；
- 有限距离介质贡献；
- 远景轮廓和弱纹理结构。

因此，当前 M2 的指标下降很可能主要来源于合成公式，而不是 \(B_{\infty}\) 本身无效。

### 2.3 `near_zero_loss` 的优化目标不够准确

当前损失将：

```python
near_rgb = rgb_object + rgb_medium
```

整体压向零。但纯水体区域中，Gaussian 物体贡献应当接近零，有限距离 medium contribution 并不必然为零。将二者同时抑制，会迫使 \(B_{\infty}\) 单独解释整个远区，使原有 WaterSplatting 的有限介质积分失去作用，进而影响带水体重建。

### 2.4 ownership 证据缺乏边界保护与外部几何支持

当前 ownership 主要依赖模型自身的 accumulation 和 depth，且默认对证据执行 detach。它没有显式考虑：

- 多视角几何支持；
- pseudo-depth 前景与背景分区；
- 物体边界保护；
- 图像梯度与深度梯度；
- 真实远景物体的低 accumulation 特性。

因此，当前 `alpha_depth` 更像一个软启发式，而不是高精度的 scene-medium attribution。

---

## 3. 去水体偏色问题诊断

### 3.1 当前 clear color 缺乏直接监督

当前去水体输出：

\[
J=\operatorname{clamp}(J_{\mathrm{raw}},0,1)
\]

由 Gaussian clear render 直接得到。Gaussian SH 颜色仅通过带水体 RGB 监督优化，没有真实无水图像监督。因此，远距离物体的固有颜色与水体衰减、后向散射之间存在明显的不可辨识性。

水下图像近似满足：

\[
I
=
J\exp(-\beta^D d)
+
B\left(1-\exp(-\beta^B d)\right).
\]

当深度较大时：

\[
T=\exp(-\beta^D d)\rightarrow 0,
\qquad
\frac{\partial I}{\partial J}=T\rightarrow 0.
\]

这意味着远处固有颜色 \(J\) 几乎得不到有效梯度。模型可以通过调整 \(J\)、\(\beta^D\) 和 \(B\) 的组合重建带水体 RGB，但不一定恢复正确的固有颜色。由于输入图像的远处区域通常以蓝绿通道占主导，Gaussian 固有颜色容易吸收蓝绿色偏差。

### 3.2 SeaFree-GS 的可借鉴机制

SeaFree-GS 使用 Degradation-Aware Dual-Color Modeling，将每个 Gaussian 的固有颜色与水下退化颜色显式区分：

\[
c_i^{\mathrm{uw}}
=
c_i^{\mathrm{int}}
\exp(-\beta_i^D z_i)
+
A_i
\left[
1-\exp(-\beta_i^B z_i)
\right].
\]

其中：

- \(c_i^{\mathrm{int}}\) 为 Gaussian 固有颜色；
- \(A_i\) 为环境水光；
- \(\beta_i^D\) 和 \(\beta_i^B\) 为衰减与后向散射系数；
- \(z_i\) 为 Gaussian 到相机的视线距离。

其关键优势不在于简单增加一个损失，而在于：

1. 固有颜色和水下颜色使用同一组 Gaussian 几何与 opacity；
2. 水下颜色由固有颜色经过物理退化获得；
3. intrinsic render 与 underwater render 共享完全一致的可见性；
4. 背景水体区域用于直接监督 ambient light；
5. pseudo-depth 用于区分前景与纯水体背景；
6. 公开配置采用 SH degree 0，使固有颜色更稳定。

本项目不应直接复制 SH degree 0，因为当前目标仍希望保留 SH=3 对视角依赖反射和外观变化的表达能力。更合理的做法是分离稳定固有颜色与受约束 SH 残差。

---

## 4. 总体优化路线

后续框架建议形成四个相互衔接的部分：

```text
M1 Context-Regularized Medium Field
        ↓
Intrinsic-Degraded Dual-Color Gaussians
        ↓
Closed-Tail Infinite-Water Attribution
        ↓
Confidence-Gated Gaussian Capacity Control
```

其中：

- M1 保留当前有效的 context-aware medium modeling；
- Dual-Color Gaussian 负责解决去水体偏色；
- Closed-Tail M2 负责无限远水体闭合与归属；
- Capacity Control 仅抑制高置信纯水体 Gaussian 的增长，不直接激进删除。

---

# 5. 阶段一：低成本排查当前 M2 的指标下降来源

## 5.1 实验 E1：移除 `near_zero_loss`

### 目的

验证带水体指标下降是否主要由 `near_zero_loss` 导致。

### 修改

设置：

```bash
NEAR_ZERO_WEIGHT=0
```

其余保持当前 M2 配置不变：

```text
medium_context_mode = dir_xy_camera
ownership_mode = alpha_depth
occupancy_limited = True
B_inf RGB loss = 0.01
accumulation loss = 0.004
```

### 对比配置

| 配置 | near-zero | 其他 M2 机制 |
|---|---:|---|
| M1 | 关闭 | 全部关闭 |
| M2-original | 0.001 | 当前配置 |
| M2-no-near | 0 | 保持不变 |

### 判断标准

若 `M2-no-near` 相比 `M2-original`：

- PSNR 提升；
- LPIPS 降低；
- 远区残留变化不大；

则说明 `near_zero_loss` 的优化方向确实破坏了有限介质贡献，应永久删除。

---

## 5.2 实验 E2：降低 accumulation-zero 权重

### 目的

验证当前 accumulation 约束是否过强。

### 配置

固定：

```text
lambda_binf_rgb = 0.005
lambda_near_zero = 0
```

搜索：

```text
lambda_accumulation_zero ∈ {0, 0.0005, 0.001, 0.002, 0.004}
```

### 评价

同时比较：

- underwater PSNR / SSIM / LPIPS；
- far-region accumulation；
- far-region object luma；
- `J Blue Dominance`；
- 远景物体结构完整性。

### 预期

较弱的 accumulation 约束可能保留 M2 对远处残留的抑制作用，同时降低对真实远景物体的损伤。

---

## 5.3 实验 E3：延迟 M2 启用时间

### 目的

避免 M2 在几何与 medium 尚未稳定时过早抢占远区解释权。

### 搜索参数

```text
loss_start_step ∈ {1000, 3000, 5000, 7000}
loss_ramp_steps ∈ {3000, 5000}
```

### 推荐起点

```text
loss_start_step = 5000
loss_ramp_steps = 5000
```

### 判断

若延迟启用能够恢复带水体指标，同时保持远区残留较低，则说明当前 M2 的主要问题之一是优化时序，而不是结构完全错误。

---

# 6. 阶段二：将 M2 从完整图像混合改为尾部水体闭合

## 6.1 实验 E4：取消完整 `render.rgb` alpha mix

### 当前形式

```python
rgb = (1 - m_inf_eff) * render.rgb + m_inf_eff * b_inf
```

### 快速近似形式

在不修改 CUDA 的情况下，先只替换低 occupancy 尾部背景：

```python
tail_gate = (1.0 - render.accumulation).detach()

rgb = (
    render.rgb
    + m_inf * tail_gate * (b_inf - medium_rgb)
)
```

### 解释

该式不再直接削弱完整的 object + finite-medium render，而是将低 occupancy 区域原有背景 medium color 向 \(B_{\infty}\) 校正。它仍是近似形式，但可以快速判断当前指标下降是否由完整图像混合导致。

### 对比

| 配置 | 合成方式 |
|---|---|
| M2-original | 完整 RGB alpha mix |
| M2-tail-approx | 仅背景尾部替换 |
| M1 | 不使用 \(B_{\infty}\) |

### 成功标准

`M2-tail-approx` 应满足：

- PSNR 不低于 M1 超过 0.05 dB；
- LPIPS 不高于 M1 超过 0.001；
- 远区 Gaussian leakage 明显低于 M1；
- 远景物体轮廓优于 M2-original。

---

## 6.2 实验 E5：CUDA 输出 finite medium 与 tail weight

### 目标

形成物理上闭合的无穷远水体渲染。

### 建议新增输出

CUDA rasterizer 返回：

```text
rgb_object
rgb_medium_finite
tail_weight
depth
accumulation
hit_depth
```

最终合成为：

\[
I
=
I_{\mathrm{obj}}
+
I_{\mathrm{med}}^{\mathrm{finite}}
+
W_{\mathrm{tail}}B_{\infty}.
\]

其中：

\[
W_{\mathrm{tail}}
=
T_{\mathrm{close}}^{\mathrm{obj}}
\exp(-\beta^B d_{\mathrm{close}}).
\]

### `d_close` 建议

\[
d_{\mathrm{close}}
=
w_{\mathrm{hit}}d_{\mathrm{hit}}
+
(1-w_{\mathrm{hit}})d_{\mathrm{far}}.
\]

其中：

- 有可靠命中时使用 hit-aware depth；
- 无可靠命中时使用稳定的 far depth；
- 不再直接依赖最后一个 Gaussian 的深度。

### 预期收益

- M2 不再遮挡真实远景物体；
- \(B_{\infty}\) 只解释无穷远介质尾部；
- Gaussian 数量变化不再显著改变背景介质闭合；
- 带水体 RGB 与去水体结构更加一致。

---

# 7. 阶段三：重新设计 ownership 证据

## 7.1 ownership 拆分

不再使用单一 `m_inf` 同时承担所有职责，拆分为：

```text
M_sup       高精度水体监督区域
M_render    尾部背景合成区域
M_capacity  Gaussian 容量抑制区域
```

### `M_sup`

用于监督 \(B_{\infty}\)，强调高精度：

\[
M_{\mathrm{sup}}
=
M_{\mathrm{bg}}^{\mathrm{teacher}}
\cdot
M_{\mathrm{low\text{-}acc}}
\cdot
M_{\mathrm{far}}
\cdot
M_{\mathrm{boundary}}.
\]

### `M_render`

用于尾部背景合成，可稍宽松但仍需边界保护。

### `M_capacity`

用于抑制 Gaussian densification，应最严格，避免误伤真实物体。

---

## 7.2 实验 E6：引入 pseudo-depth 背景水体支持

### 数据

参考 SeaFree-GS，使用 DepthAnything 输出的相对 disparity-like depth。

### 背景区域构建

1. 对 pseudo-depth 归一化；
2. 根据远小、近大的语义提取远背景候选；
3. 保留最大连续前景轮廓；
4. 将轮廓外区域视为背景候选；
5. 对背景候选执行 5、7、11 像素腐蚀；
6. 排除 RGB 强梯度与深度强梯度区域。

### 消融

| 配置 | Teacher mask | 边界腐蚀 |
|---|---|---|
| E6-a | pseudo-depth | 5 px |
| E6-b | pseudo-depth | 7 px |
| E6-c | pseudo-depth | 11 px |
| E6-d | 无 teacher | 无 |

### 目标

提高 ownership precision，而不是最大化背景覆盖率。

### 评价

建议人工标注少量验证图像中的：

- 纯水体区域；
- 真实远景物体；
- 物体与水体边界。

计算：

```text
Water Precision
Water Recall
Object False Positive Rate
Boundary False Positive Rate
```

优先优化 Water Precision 和 Object False Positive Rate。

---

## 7.3 实验 E7：增加边界保护

定义：

\[
M_{\mathrm{boundary}}
=
\exp
\left[
-\lambda_g
\left(
\|\nabla I\|
+
\lambda_d\|\nabla d\|
\right)
\right].
\]

或者使用二值方式：

```python
safe_mask = (
    rgb_grad < rgb_grad_threshold
) & (
    depth_grad < depth_grad_threshold
)
```

### 搜索参数

```text
rgb_grad_threshold ∈ {0.02, 0.04, 0.06}
depth_grad_threshold ∈ {0.02, 0.05, 0.10}
```

### 预期

保护珊瑚轮廓、海床边缘、礁石顶部和远处真实场景结构。

---

# 8. 阶段四：用容量控制替代激进删除

## 8.1 设计原则

当前 far-water residual 已经很小，硬删除的边际收益有限。后续应优先防止错误 Gaussian 持续增长，而不是删除已有结构。

## 8.2 Gaussian 水体归属 EMA

对 Gaussian \(i\)，维护：

\[
q_i^{t}
=
\rho q_i^{t-1}
+
(1-\rho)
\frac{
\sum_p w_{ip}M_{\mathrm{capacity}}(p)
}{
\sum_p w_{ip}+\epsilon
}.
\]

建议：

```text
rho = 0.9 or 0.95
```

同时维护：

```text
object_support_i
water_support_i
observation_count_i
```

## 8.3 分级动作

### 高置信纯水体 Gaussian

满足：

```text
q_i > 0.8
object_support_i < 0.1
observation_count_i >= 5
```

执行：

```text
densification block
禁止 duplicate / split
降低 densification score
缓慢 opacity decay
```

### 混合边界 Gaussian

满足：

```text
water_support 高
object_support 也非零
```

不删除，只进行：

```text
轻度 scale shrink
禁止继续 densify
```

### Object-dominant Gaussian

正常保留。

## 8.4 实验 E8：分级容量控制

| 配置 | Densification block | Opacity decay | Hard prune |
|---|---|---|---|
| E8-a | 开启 | 关闭 | 关闭 |
| E8-b | 开启 | 开启 | 关闭 |
| E8-c | 开启 | 开启 | 延迟开启 |
| E8-d | 关闭 | 关闭 | 关闭 |

### 推荐初始设置

```text
capacity_start_step = 8000
opacity_decay_interval = 500
opacity_decay_rate = 0.02
hard_prune_start_step = 14000
minimum_consecutive_hits = 4
maximum_processed_fraction = 0.5%
```

---

# 9. 阶段五：固有颜色与退化颜色双分支建模

## 9.1 目标

解决远处去水体结果偏蓝、偏绿问题，同时保留 SH=3 对视角依赖反射和外观变化的表达能力。

## 9.2 固有颜色

定义稳定固有颜色：

\[
c_i^{\mathrm{int}}
=
\sigma(\theta_i^{\mathrm{DC}}).
\]

该颜色作为主要去水体颜色，不再直接由完整 SH 输出决定。

## 9.3 SH 残差分解

计算：

\[
\Delta c_i^{\mathrm{SH}}(\mathbf v).
\]

将其分解为亮度残差与色度残差：

\[
\Delta l_i
=
\operatorname{mean}_c
\left(
\Delta c_i^{\mathrm{SH}}
\right),
\]

\[
\Delta c_i^{\perp}
=
\Delta c_i^{\mathrm{SH}}
-
\Delta l_i.
\]

定义受约束视角残差：

\[
\Delta c_i^{\mathrm{bounded}}
=
\Delta l_i
+
\eta_c\Delta c_i^{\perp}.
\]

其中：

```text
eta_c ∈ {0, 0.05, 0.1, 0.2}
```

### 带水体基础颜色

\[
c_i^{\mathrm{uw\text{-}base}}
=
c_i^{\mathrm{int}}
+
\eta_{\mathrm{uw}}
\Delta c_i^{\mathrm{bounded}},
\qquad
\eta_{\mathrm{uw}}=1.
\]

### 去水体颜色

\[
c_i^{\mathrm{clear}}
=
c_i^{\mathrm{int}}
+
\eta_{\mathrm{clear}}
\Delta c_i^{\mathrm{bounded}}.
\]

搜索：

```text
eta_clear ∈ {0, 0.05, 0.1, 0.2}
```

### 预期

- underwater RGB 继续使用 SH=3；
- clear render 主要依赖稳定 intrinsic DC color；
- SH 仍可表达亮度、高光和方向性反射；
- SH 不再自由改变远处物体色相。

---

## 9.4 实验 E9：Dual-Color Gaussian 消融

| 配置 | Underwater SH | Clear SH | 色度缩放 |
|---|---|---|---|
| E9-a | SH=3 | SH=0 | 0 |
| E9-b | SH=3 | 0.05 residual | 0.05 |
| E9-c | SH=3 | 0.10 residual | 0.10 |
| E9-d | SH=3 | 0.20 residual | 0.20 |
| E9-e | 原始共享 SH=3 | 原始共享 SH=3 | 1.0 |

### 评价

除 underwater PSNR / SSIM / LPIPS 外，增加：

```text
J Blue Dominance
J Green Dominance
J Chroma Variance
J Inter-view Color Consistency
```

若具备合成数据真值，则进一步计算：

```text
Clear PSNR
Clear SSIM
Clear LPIPS
CIEDE2000
Delta E in Lab
```

---

## 9.5 CUDA 实现建议

当前 rasterizer 已同时输出 underwater 与 clear 结果，因此建议将颜色输入扩展为：

```text
colors_underwater
colors_intrinsic
```

两套颜色共享：

- Gaussian 投影；
- front-to-back 排序；
- opacity；
- transmittance；
- visibility；
- depth。

CUDA 前向过程分别累积：

\[
I_{\mathrm{obj}}
=
\sum_i
T_i
\alpha_i
c_i^{\mathrm{underwater}},
\]

\[
J
=
\sum_i
T_i
\alpha_i
c_i^{\mathrm{intrinsic}}.
\]

这样可确保 underwater render 与 clear render 在几何和遮挡上完全一致。

---

# 10. 阶段六：增强远景固有颜色监督

## 10.1 Transmission-Aware Foreground Loss

定义平均透射率：

\[
\bar T(p)
=
\frac{1}{3}
\sum_{c\in\{R,G,B\}}
\exp
\left[
-\beta_c^D(p)d_{\mathrm{hit}}(p)
\right].
\]

构造权重：

\[
w(p)
=
1+
\lambda_T
M_{\mathrm{obj}}(p)
\left[
1-\bar T(p)
\right]^\gamma.
\]

重建损失：

\[
\mathcal L_{\mathrm{rec}}
=
\frac{
\sum_p w(p)
\left|
I(p)-\hat I(p)
\right|
}{
\sum_p w(p)+\epsilon
}
+
\lambda_s\mathcal L_{\mathrm{SSIM}}.
\]

### 初始参数

```text
lambda_T = 1.0
gamma = 1.0
w_max = 4.0
detach_weight = True
```

### 原理

远处低透射率物体对固有颜色的梯度很弱。该损失不直接监督 clear color，但能够增强这部分物体对 underwater reconstruction 的贡献，从而改善 intrinsic color 的可辨识性。

---

## 10.2 实验 E10：Foreground Loss 对比

| 配置 | 权重方式 |
|---|---|
| E10-a | 原始 reconstruction loss |
| E10-b | SeaFree-GS 式 inverse-brightness |
| E10-c | transmission-aware |
| E10-d | transmission-aware + pseudo-depth object mask |

### 判断

重点观察：

- 远处海床和珊瑚颜色；
- 弱纹理区域细节；
- underwater PSNR；
- clear color consistency；
- 是否放大噪声或产生过度锐化。

---

# 11. 阶段七：限制 M1 的过高自由度

## 11.1 当前风险

M1 使用：

```text
ray direction
image coordinates
normalized camera center
```

能够显著提高指标，但也可能将 medium MLP 变成接近逐视角颜色校正器。

## 11.2 全局主体加小幅残差

定义：

\[
\beta^D
=
\operatorname{softplus}
\left(
\beta_0^D
+
s_{\beta}\Delta\beta^D
\right),
\]

\[
B
=
\sigma
\left(
B_0+s_B\Delta B
\right).
\]

推荐：

```text
s_beta ∈ {0.05, 0.1, 0.2}
s_B ∈ {0.05, 0.1, 0.2}
camera_context_dropout ∈ {0, 0.1, 0.2}
```

### 目标

保留 M1 的适应性，但减少 medium 分支任意吸收场景颜色变化的能力。

---

# 12. 评价体系

## 12.1 带水体新视角重建

```text
PSNR ↑
SSIM ↑
LPIPS ↓
```

以 M1 为主要基准，而不是仅与原始 Baseline 比较。

## 12.2 远区 Gaussian 残留

定义远区 mask \(M_{\mathrm{far}}\)，统计：

\[
E_{\mathrm{acc}}
=
\frac{
\sum_p M_{\mathrm{far}}(p)A(p)
}{
\sum_p M_{\mathrm{far}}(p)+\epsilon
}.
\]

\[
E_{\mathrm{obj}}
=
\frac{
\sum_p
M_{\mathrm{far}}(p)
\operatorname{luma}
(I_{\mathrm{obj}}(p))
}{
\sum_p M_{\mathrm{far}}(p)+\epsilon
}.
\]

同时记录：

```text
far accumulation mean
far object luma mean
far object luma > 0.03 ratio
far Gaussian count
```

## 12.3 真实物体保留

定义 object-support mask \(M_{\mathrm{obj}}\)：

```text
Object accumulation retention
Object edge PSNR
Boundary LPIPS
Foreground depth consistency
```

## 12.4 去水体颜色

无真实 clear GT 时，使用诊断指标：

```text
J Blue Dominance Ratio
J Green Dominance Ratio
J Red Dominance Ratio
J Saturation Ratio
J White Ratio
Inter-view Color Consistency
Temporal Color Consistency on camera path
```

有 FUNA 合成数据真值时，必须报告：

```text
Clear PSNR
Clear SSIM
Clear LPIPS
CIEDE2000
Lab Delta E
Per-depth-bin color error
```

建议按深度区间统计：

```text
near
middle
far
```

以验证偏色是否主要发生在低透射率远区。

## 12.5 Pareto 评价

绘制二维 Pareto 图：

- 横轴：underwater LPIPS 或 PSNR loss；
- 纵轴：far Gaussian leakage；
- 点颜色：clear color error 或 J Blue Dominance。

目标不是单一指标最优，而是找到：

```text
低 leakage
低 underwater quality loss
低 clear color bias
```

的 Pareto 最优配置。

---

# 13. 推荐实验顺序

## 第一轮：快速排查

1. E1：关闭 `near_zero_loss`；
2. E2：降低 accumulation-zero 权重；
3. E3：延迟 M2 启用；
4. E4：将完整 RGB alpha mix 改为 tail replacement。

这一轮不修改 CUDA，目标是快速确定当前 M2 指标下降的主要来源。

## 第二轮：高精度 ownership

5. E6：pseudo-depth 水体 teacher；
6. E7：边界保护；
7. E8：densification blocking 与软 opacity decay。

目标是在不删除真实远景结构的前提下继续减少错误容量。

## 第三轮：颜色优化

8. E9：Dual-Color Gaussian；
9. E10：transmission-aware foreground loss；
10. M1 residual capacity 限制。

目标是解决远处蓝绿偏色，并保留 SH=3 的视角依赖表达。

## 第四轮：完整物理闭合

11. E5：CUDA 输出 finite medium 与 tail weight；
12. 将所有最优配置组合；
13. 在 SeaThru-NeRF 与 FUNA 多场景上验证泛化。

---

# 14. 建议的最小实验矩阵

| 编号 | M1 | M2 合成 | Ownership | Capacity | Dual Color | Foreground Loss |
|---|---|---|---|---|---|---|
| A0 | 开 | 关 | 无 | 无 | 关 | 原始 |
| A1 | 开 | 原始 mix | alpha-depth | 无 | 关 | 原始 |
| A2 | 开 | 原始 mix | alpha-depth | 无 | 关 | 原始，near-zero=0 |
| A3 | 开 | tail approx | alpha-depth | 无 | 关 | 原始 |
| A4 | 开 | tail approx | teacher + boundary | 无 | 关 | 原始 |
| A5 | 开 | tail approx | teacher + boundary | densify block | 关 | 原始 |
| A6 | 开 | tail approx | teacher + boundary | densify block | 开 | 原始 |
| A7 | 开 | tail approx | teacher + boundary | densify block | 开 | transmission-aware |
| A8 | 开 | closed tail | teacher + boundary | densify block | 开 | transmission-aware |

其中：

- A0 为当前可靠 M1；
- A1 为当前 M2；
- A2 至 A5 定位并修复 M2；
- A6 至 A7 解决颜色；
- A8 为最终完整模型。

---

# 15. 代码修改建议

## 15.1 `water_splatting/ownership/infinite_water_ownership.py`

新增：

```text
support_mask
render_mask
capacity_mask
boundary_evidence
teacher_evidence
```

将输出结构改为：

```python
@dataclass
class InfiniteWaterOwnershipOutput:
    m_support: Tensor
    m_render: Tensor
    m_capacity: Tensor
    alpha_evidence: Tensor
    depth_evidence: Tensor
    teacher_evidence: Tensor
    boundary_evidence: Tensor
```

## 15.2 `water_splatting/water_splatting.py`

删除完整图像混合：

```python
rgb = (1 - m_inf_eff) * render.rgb + m_inf_eff * b_inf
```

第一阶段替换为 tail approximation，后续替换为 closed-tail composition。

删除：

```python
infinite_water_near_zero_loss
```

增加：

```text
transmission-aware foreground loss
capacity EMA cache
densification blocking
dual-color outputs
```

## 15.3 `water_splatting/fields/gaussian_appearance.py`

新增：

```python
compute_dual_gaussian_colors()
```

输出：

```text
intrinsic_rgb
underwater_base_rgb
clear_rgb
sh_luminance_residual
sh_chroma_residual
```

## 15.4 `water_splatting/rendering/underwater_rasterizer.py`

短期：

```text
支持两套 Gaussian color
```

长期：

```text
输出 finite medium
输出 tail transmittance
输出 hit-aware closure depth
```

## 15.5 CUDA rasterizer

新增输入：

```text
colors_underwater
colors_intrinsic
```

新增输出：

```text
rgb_object_underwater
rgb_object_intrinsic
rgb_medium_finite
tail_weight
```

---

# 16. 最终成功标准

最终模型相较 M1 应同时满足：

## 带水体重建

```text
PSNR drop ≤ 0.05 dB
LPIPS increase ≤ 0.001
SSIM 不下降
```

理想情况下应超过 M1。

## Gaussian 残留

```text
far accumulation 至少下降 30%
far object luma 至少下降 30%
远区可见残留明显少于 M1
```

## 真实远景保护

```text
Object false positive rate < 2%
Boundary false positive rate < 5%
远景轮廓无明显缺失
```

## 去水体颜色

在 FUNA 合成数据上：

```text
far-region CIEDE2000 明显下降
clear LPIPS 优于 M1
per-depth-bin color error 随深度增长更缓慢
```

在真实 SeaThru-NeRF 场景上：

```text
J Blue/Green Dominance 降低
跨视角颜色一致性提高
远景海床与珊瑚色相更稳定
```

---

# 17. 预期最终方法表述

最终方法可以归纳为三个主模块：

## 1. Geometry-Anchored Pre-Medium Gaussian Representation

通过稳定的 Gaussian 几何和 intrinsic-degraded dual-color 表示，建立水体作用前的场景外观基础。

## 2. Physics-Guided Closed Scene-Medium Attribution

利用 context-regularized medium field、hit-aware closure 与 infinite-water tail，完成有限场景、有限介质和无限远水体之间的闭合归属。

## 3. Misattribution-Aware Gaussian Capacity Control

利用高置信水体区域限制 Gaussian densification，并通过软 opacity decay 抑制错误容量，而不是直接破坏真实远景结构。

最终核心目标为：

\[
\boxed{
\text{保留 SH=3 的新视角表达能力}
+
\text{稳定恢复 Gaussian 固有颜色}
+
\text{减少远处水体型 Gaussian 容量}
+
\text{不损害带水体新视角重建}
}
\]

---

# 18. 参考代码与材料

- 当前重构项目：`MokiSumiwo/water-splatting-refactor-ycy`
- 原始 WaterSplatting：`water-splatting/water-splatting`
- SeaFree-GS：`deng-ai-lab/SeaFree-GS`
- 当前实验记录：`research_notes/ALL_EXPERIMENT_RESULTS_MEETING_2026-07-23.md`
- 当前主对比图：`renders/baseline_m1_m2_rgb_J_long_20260724.png`
