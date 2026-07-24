# WaterSplatting 框架重构与消融实验方案

## 1. 研究背景与当前判断

当前研究项目是在原始 WaterSplatting 基础上逐步扩展形成的。原始 WaterSplatting 的方法逻辑较为清晰，其核心结构是：

```text
Gaussian SH view-dependent appearance
+ direction-conditioned medium MLP
+ underwater attenuation/backscatter rasterization
→ underwater novel-view rendering
```

经过多阶段机制探索后，现有研究代码已经进一步加入：

```text
context-aware medium field
+ infinite-water B_inf branch
+ pure-water ownership estimation
+ alpha-mix composition
+ occupancy-limited B_inf
+ contribution-aware Gaussian cleanup
+ residual-SH decomposition
+ low-transmission DC gauge
+ pseudo-depth supervision
+ canonical intrinsic experiments
+ several diagnostic and legacy branches
```

这些机制中，一部分已经表现出稳定价值，但当前代码同时保存了大量历史实验开关、Stage 编号、重复监督项和诊断分支，导致模型前向过程、损失计算、高斯增密剪枝和实验配置相互耦合，难以继续扩展，也不利于形成清晰的论文叙事。

因此，后续不建议继续在当前代码中叠加新的 Stage，而应重新 clone 原始 WaterSplatting，在干净基线上逐步实现四个具有明确语义的核心模块，并将每个模块设计为可独立开启、关闭和消融的组件。

---

## 2. 重构路线的总体原则

新的实现应满足以下原则。

### 2.1 原始基线必须可复现

四个新增模块全部关闭时，新代码应尽可能恢复原始 WaterSplatting 的行为，包括：

- underwater RGB
- clear Gaussian rendering
- depth
- accumulation
- reconstruction loss
- Gaussian densification and pruning
- Gaussian count evolution

允许 CUDA 非确定性引起微小数值差异，但不能出现明显的性能偏移。

### 2.2 模块命名采用方法语义

不再使用 Stage2AF、Stage2AG、Stage9、Stage10B 等历史实验编号作为主代码名称，而应采用可以直接对应论文方法章节的命名。

建议保留四个模块：

```text
M1 Context-Aware Medium Modeling
M2 Infinite-Water Ownership
M3 Contribution-Aware Gaussian Cleanup
M4 Constrained View-Dependent Appearance
```

### 2.3 每个模块必须具有独立的配置入口

例如：

```yaml
medium:
  context_enabled: false

infinite_water:
  enabled: false

cleanup:
  enabled: false

appearance:
  constrained_sh_enabled: false
```

模块关闭后，不应继续创建无效网络、无效参数或无效 optimizer。

### 2.4 训练机制与诊断机制分离

训练必需的输出与仅用于可视化、分析的输出应分开维护。主 forward 不应持续返回几十个历史诊断量。

---

## 3. 四个核心模块

# 3.1 M1: Context-Aware Medium Modeling

## 3.1.1 原始问题

原始 WaterSplatting 的 medium MLP 主要根据 ray direction 预测：

- medium RGB
- backscatter coefficient
- attenuation coefficient

这种表示假设相同方向上的水体退化具有较强一致性，但真实水下图像中的退化还可能受到以下因素影响：

- 像素位于成像平面的不同位置
- 镜头边缘和中心的成像差异
- 相机在场景中的位置
- 局部光照与拍摄位置变化

如果 medium field 的表达能力不足，部分水体相关变化可能被 Gaussian SH 外观或背景分支错误吸收。

## 3.1.2 建议输入

建议将 medium feature 定义为：

\[
\mathbf f_{\mathrm{med}}
=
[
E(\mathbf d),
x_n,
y_n,
r_n,
\tilde{\mathbf t}_{cam}
],
\]

其中：

- \(E(\mathbf d)\) 为 ray direction encoding
- \(x_n,y_n\) 为归一化图像坐标
- \(r_n=\sqrt{x_n^2+y_n^2}\) 为像素到图像中心的归一化距离
- \(\tilde{\mathbf t}_{cam}\) 为归一化相机中心

相机中心不建议直接使用：

```python
torch.tanh(camera_center)
```

而应基于训练相机或场景边界进行归一化：

\[
\tilde{\mathbf t}_{cam}
=
\frac{\mathbf t_{cam}-\boldsymbol\mu_{cam}}
{\boldsymbol\sigma_{cam}+\epsilon}.
\]

这样可以减弱不同场景坐标尺度带来的影响。

## 3.1.3 模块输出

在只启用 M1 时，输出维持原始 9 维：

```text
medium_rgb: 3
beta_bs:    3
beta_attn:  3
```

M1 不应改变最终合成公式、Gaussian ownership 或 Gaussian cleanup。

## 3.1.4 需要回答的问题

M1 的独立消融主要回答：

> 在不修改场景表示和最终合成方式的情况下，加入有限上下文的 medium field 是否能稳定提升带水体新视角合成质量？

## 3.1.5 风险与补充消融

需要防止相机位置被 medium MLP 当作 camera ID 使用，从而记忆训练视角。建议增加：

- direction only
- direction + image coordinate
- direction + image coordinate + normalized camera center
- camera-context dropout

---

# 3.2 M2: Infinite-Water Ownership

## 3.2.1 原始问题

原始 WaterSplatting 没有独立的无限远纯水体表示。远处平滑纯水区域可能同时被以下分支解释：

- 低纹理、低 opacity Gaussian
- finite-distance backscatter
- Gaussian SH color

这会导致远处纯水区域残留 Gaussian，使去水体结果中出现蓝色、绿色或灰白色残留。

## 3.2.2 独立的无限远水体分支

建议将 medium MLP 输出扩展为：

```text
medium_rgb: 3
beta_bs:    3
beta_attn:  3
B_inf:      3
```

其中 \(B_{\infty}\) 表示无限远纯水体背景颜色。

可以首先采用低容量形式：

\[
B_{\infty}
=
\sigma
\left(
\operatorname{logit}(B_{\mathrm{base}})
+
\Delta B
\right),
\]

但在首轮重构中建议先使用 `B_inf base`，再单独消融 residual field，避免一开始引入过高背景容量。

## 3.2.3 统一的 ownership 表示

模块对外只提供两个主要量：

```text
M_inf: infinite-water ownership
M_obj = 1 - M_inf
```

最终合成统一写为：

\[
\mathbf I
=
M_{\mathrm{obj}}\mathbf I_{\mathrm{near}}
+
M_{\infty}\mathbf B_{\infty},
\]

\[
\mathbf J_{\mathrm{dry}}
=
M_{\mathrm{obj}}\mathbf J_{\mathrm{gaussian}}.
\]

其中：

- \(\mathbf I_{\mathrm{near}}\) 为 Gaussian object 与有限距离 medium 的联合渲染
- \(\mathbf J_{\mathrm{gaussian}}\) 为去除水体退化后的 Gaussian clear rendering
- \(\mathbf B_{\infty}\) 为无限远水体背景

## 3.2.4 Ownership evidence

建议用一套统一、可解释的 evidence 构建 \(M_{\infty}\)：

```text
far-depth evidence
low-gradient evidence
B_inf color similarity
low object accumulation evidence
Gaussian residual-water evidence
object-boundary protection
```

不建议把 DepthAnything pseudo-depth 继续作为主 ownership label。

## 3.2.5 Occupancy-limited composition

仅使用 alpha-mix 可能让 \(B_{\infty}\) 在图像层面遮住错误 Gaussian，但不真正消除 Gaussian accumulation。

因此建议保留 occupancy-limited composition：

\[
M_{\infty}^{\mathrm{eff}}
=
M_{\infty}\cdot g(1-A),
\]

其中 \(A\) 是 Gaussian accumulation，\(g(\cdot)\) 为连续抑制函数。高 accumulation 区域不能被 \(B_{\infty}\) 无条件覆盖。

该设计建立了如下结构压力：

```text
B_inf wants to explain pure water
→ Gaussian accumulation must decrease
→ pure-water Gaussian ownership is transferred
```

## 3.2.6 需要回答的问题

M2 的消融主要回答：

1. 独立 \(B_{\infty}\) 是否能提高纯水背景表达能力？
2. 显式 ownership 是否能减少远水 Gaussian 残留？
3. occupancy limit 是否能防止 \(B_{\infty}\) 仅成为视觉遮罩？

---

# 3.3 M3: Contribution-Aware Gaussian Cleanup

## 3.3.1 模块目标

M2 解决像素级归属，M3 进一步解决结构级错误高斯。其目标是识别主要贡献于纯水区域、而不是物体区域的 Gaussian，并对其执行：

- opacity decay
- scale shrink
- directional relocation
- delayed pruning
- densification blocking

## 3.3.2 True contribution 定义

对 Gaussian \(g\)，定义其对纯水区域和物体区域的真实前向贡献：

\[
C_g^{water}
=
\sum_p
M_{\infty}(p)
T_{\mathrm{before}}(g,p)
\alpha_g(p),
\]

\[
C_g^{object}
=
\sum_p
M_{\mathrm{obj}}(p)
T_{\mathrm{before}}(g,p)
\alpha_g(p).
\]

该定义考虑 front-to-back rendering order，比投影中心、footprint overlap 或简单 opacity 更能反映 Gaussian 的真实作用。

进一步定义：

\[
r_g
=
\frac{C_g^{water}}
{C_g^{water}+C_g^{object}+\epsilon}.
\]

## 3.3.3 Gaussian 分类与动作

### Pure-water dominant Gaussian

满足：

```text
r_g high
C_object low
observation count sufficient
```

建议执行：

```text
opacity decay
→ densification block
→ delayed prune
```

### Mixed boundary Gaussian

满足：

```text
C_water high
C_object also non-negligible
```

这类 Gaussian 不能直接删除，否则容易损伤礁石、船体或其他边界。建议执行：

```text
scale shrink
or directional shrink
or slight relocation toward object-support region
```

### Object-dominant Gaussian

满足：

```text
C_object high
```

应予以保护，不进行 cleanup。

## 3.3.4 Controller 独立化

该模块不应继续直接写入 `get_outputs()`。建议拆分为：

```text
ContributionAttributor
GaussianCleanupState
GaussianCleanupController
```

其中：

- `ContributionAttributor` 计算每个 Gaussian 的 water/object contribution
- `GaussianCleanupState` 维护 EMA、observation count 和 action history
- `GaussianCleanupController` 在 refinement callback 中执行动作

## 3.3.5 与普通 densification 的关系

M3 应能够：

- 阻止纯水 Gaussian 被重复 densify
- 在 split/duplicate 后扩展 state buffer
- 在 prune 后同步裁剪 state buffer
- 正确清除 opacity、scale 和 mean optimizer state
- 避免 cleanup 与原始 opacity reset 相互抵消

## 3.3.6 需要回答的问题

M3 的消融应证明：

1. true contribution 优于 projected footprint
2. cleanup 减少了真实 pure-water Gaussian，而不只是改变最终 RGB
3. boundary protection 可以降低物体边缘损伤
4. cleanup 带来的计算成本是可接受的

---

# 3.4 M4: Constrained View-Dependent Appearance

## 3.4.1 当前矛盾

SH=0 或 canonical/DC-only appearance 通常更容易得到稳定的去水体颜色，但会失去视角相关反射与外观变化。

SH=3 能提高带水体 RGB 指标，但其容量可能过于自由，导致：

- intrinsic color 过亮
- 去水体图像偏白
- 红色通道过强
- 固定颜色偏移被 non-DC SH 长期承载
- medium effect 被吸收到 Gaussian appearance

因此，最终方向不应是简单地将 SH 改为 0，而应限制 SH 的职责。

## 3.4.2 Residual-SH decomposition

将 Gaussian 颜色写为：

\[
\mathbf c_g(\mathbf v)
=
\mathbf c_g^{DC}
+
\Delta\mathbf c_g^{SH}(\mathbf v).
\]

其中：

- \(\mathbf c_g^{DC}\) 表示主要 intrinsic appearance anchor
- \(\Delta\mathbf c_g^{SH}(\mathbf v)\) 只表示必要的视角相关残差

## 3.4.3 Residual mean anchor

对每个 Gaussian 的观测视角 residual 建立 EMA：

\[
\bar{\Delta \mathbf c}_g
\approx
\mathbb E_{\mathbf v}
[
\Delta\mathbf c_g^{SH}(\mathbf v)
].
\]

增加：

\[
\mathcal L_{\mathrm{SH\text{-}mean}}
=
\frac{1}{|\mathcal G|}
\sum_g
\left\|
\bar{\Delta \mathbf c}_g
\right\|_1.
\]

该损失不禁止 SH 表达 view-dependent variation，而是防止 non-DC residual 长期承担固定颜色偏移。

## 3.4.4 Low-transmission DC softclip

对低 transmission Gaussian，intrinsic DC 颜色的可观测性较弱，训练容易通过增大 intrinsic color 抵消 attenuation。

定义 Gaussian 权重：

\[
w_g^{low\text{-}trans}
=
\phi(T_g),
\]

其中 transmission 越低，权重越高。

增加 DC soft upper-bound：

\[
\mathcal L_{\mathrm{DC}}
=
\sum_g
w_g^{low\text{-}trans}
\operatorname{softplus}
\left(
\frac{\mathbf c_g^{DC}-\tau}{\beta}
\right)^2.
\]

该约束只作用于 DC anchor，不直接 clamp 最终 SH rendering。

## 3.4.5 暂不迁入主线的外观机制

首轮重构不建议迁入：

```text
independent canonical intrinsic branch
canonical re-degradation branch
Stage4H hotspot contributor loss
Stage5A medium gauge
hard RGB clamp
global SH-rest L2
spectral attenuation order
```

这些机制可以在主框架稳定后再作为第二阶段研究。

## 3.4.6 需要回答的问题

M4 的消融主要回答：

1. SH=3 是否确实优于 SH=0 的 underwater RGB？
2. residual mean anchor 是否能减少固定颜色漂移？
3. DC softclip 是否能降低去水体过曝和偏红？
4. 二者联合是否能保留 view-dependent appearance，同时改善 intrinsic restoration？

---

## 4. 推荐的新代码结构

```text
water_splatting/
├── model.py
├── water_splatting_config.py
├── configs/
│   ├── schema.py
│   └── ablation_presets.py
├── fields/
│   ├── context_medium_field.py
│   └── gaussian_appearance.py
├── rendering/
│   ├── rasterizer.py
│   ├── compositor.py
│   └── outputs.py
├── ownership/
│   ├── infinite_water_ownership.py
│   └── ownership_losses.py
├── cleanup/
│   ├── contribution_attribution.py
│   ├── cleanup_state.py
│   └── cleanup_controller.py
├── losses/
│   ├── reconstruction.py
│   ├── appearance_losses.py
│   └── separation_losses.py
└── diagnostics/
    ├── dewater_metrics.py
    └── gaussian_statistics.py
```

## 4.1 Model 只负责编排

建议主 forward 近似为：

```python
medium = self.medium_field(camera)

appearance = self.appearance_field(
    gaussians=self.gaussians,
    camera=camera,
)

render = self.renderer(
    gaussians=self.gaussians,
    appearance=appearance,
    medium=medium,
    camera=camera,
)

ownership = self.ownership_estimator(
    accumulation=render.accumulation.detach(),
    depth=render.depth.detach(),
    clear_rgb=render.clear_rgb.detach(),
    b_inf=medium.b_inf.detach(),
)

outputs = self.compositor(
    render=render,
    medium=medium,
    ownership=ownership,
)
```

cleanup contribution 可以在 forward 中收集，但实际动作应在 refinement callback 中执行。

---

## 5. 新仓库的建立方式

建议重新 clone 原始项目：

```bash
git clone https://github.com/water-splatting/water-splatting.git water-splatting-refactor
cd water-splatting-refactor

git remote rename origin upstream
git remote add origin <YOUR_NEW_REPOSITORY_URL>

git checkout -b refactor/core-framework
git tag baseline-original-watersplatting
```

现有研究仓库保留为机制探索档案，不再进行大规模重构。

建议现有仓库创建归档 tag：

```bash
git tag archive-stage10-framework
git push origin archive-stage10-framework
```

---

## 6. 推荐的开发顺序

```text
Step 0: reproduce original WaterSplatting
Step 1: refactor without numerical change
Step 2: add M1 Context-Aware Medium
Step 3: add M2 Infinite-Water Ownership
Step 4: add M3 Contribution-Aware Cleanup
Step 5: add M4 Constrained SH Appearance
Step 6: run main ablations
Step 7: run interaction and mechanism ablations
Step 8: run full multi-scene evaluation
```

## 6.1 Step 0: 原始基线复现

必须固定：

- dataset version
- train/eval split
- image preprocessing
- COLMAP poses
- random seed
- training iterations
- evaluation script
- dependency versions
- CUDA implementation

保存：

```text
baseline config
Git commit SHA
PSNR / SSIM / LPIPS
training time
FPS
peak VRAM
final Gaussian count
```

## 6.2 Step 1: 无数值变化重构

先把原始代码拆分为：

```text
MediumField
GaussianAppearance
UnderwaterRasterizer
ReconstructionLoss
```

此时不加入任何新机制。

要求：

```text
old model output ≈ refactored model output
```

## 6.3 Step 2 至 Step 5

每加入一个模块：

1. 建立独立 branch
2. 添加单元测试
3. 完成单场景短训练
4. 比较打开与关闭模块的结果
5. 合并到主重构分支

---

## 7. 配置与 CLI 命名

建议注册以下方法：

```text
water-splatting-original
water-splatting-context-medium
water-splatting-infinite-water
water-splatting-cleanup
water-splatting-constrained-sh
water-splatting-full
```

建议配置结构：

```yaml
model:
  sh_degree: 3

medium:
  context_enabled: false
  image_context_enabled: true
  camera_context_enabled: true
  camera_context_dropout: 0.0

infinite_water:
  enabled: false
  residual_field_enabled: false
  occupancy_limited: true

ownership:
  enabled: false
  far_depth_weight: 1.0
  low_gradient_weight: 1.0
  color_similarity_weight: 1.0
  accumulation_weight: 1.0
  boundary_protection_weight: 1.0

cleanup:
  enabled: false
  attribution_mode: true_contribution
  opacity_decay_enabled: true
  scale_shrink_enabled: true
  directional_action_enabled: true
  delayed_prune_enabled: true
  densification_block_enabled: true

appearance:
  residual_sh_enabled: false
  mean_anchor_weight: 0.0
  dc_softclip_enabled: false
  dc_softclip_weight: 0.0
```

---

## 8. 主消融实验

# 8.1 逐步累加消融

| ID | M1 Context Medium | M2 Infinite Ownership | M3 Cleanup | M4 Constrained SH | 目的 |
|---|---:|---:|---:|---:|---|
| A0 |  |  |  |  | Original WaterSplatting |
| A1 | ✓ |  |  |  | 验证 context medium |
| A2 | ✓ | ✓ |  |  | 验证 B_inf 和 ownership |
| A3 | ✓ | ✓ | ✓ |  | 验证结构 cleanup |
| A4 | ✓ | ✓ | ✓ | ✓ | 完整模型 |

这一组用于展示从原始框架到完整方法的演进。

# 8.2 关键独立与交互消融

| ID | 配置 | 研究问题 |
|---|---|---|
| B0 | A0 + M2 | ownership 是否必须依赖 context medium |
| B1 | A0 + M4 | SH 约束是否能够独立改善 intrinsic appearance |
| B2 | A2 + M4 | M4 在没有 cleanup 时是否仍然有效 |
| B3 | A2 + M3 | cleanup 的直接贡献 |
| B4 | A4 without occupancy limit | B_inf 是否仅在视觉上遮挡 Gaussian |
| B5 | A4 with footprint cleanup | true contribution 是否优于 footprint |
| B6 | A4 without boundary protection | boundary protection 是否必要 |

无需穷举全部 \(2^4=16\) 种组合，但必须覆盖核心主效应和关键交互。

---

## 9. M1 细化消融

| ID | Medium input |
|---|---|
| M1-0 | direction |
| M1-1 | direction + xy + radius |
| M1-2 | direction + xy + radius + normalized camera center |
| M1-3 | M1-2 + camera-context dropout |

需要同时观察：

- train PSNR
- eval PSNR
- train-eval gap
- novel trajectory stability
- medium parameter variation across nearby views

若加入 camera center 后训练指标提高但测试指标下降，应警惕 camera memorization。

---

## 10. M2 细化消融

| ID | B_inf | Ownership | Occupancy limit | Residual field |
|---|---:|---:|---:|---:|
| M2-0 |  |  |  |  |
| M2-1 | ✓ |  |  |  |
| M2-2 | ✓ | ✓ |  |  |
| M2-3 | ✓ | ✓ | ✓ |  |
| M2-4 | ✓ | ✓ | ✓ | ✓ |

主要比较：

- underwater RGB
- pure-water accumulation
- B_inf object leakage
- J_dry residual-water energy
- water/object boundary quality

如果 residual field 提高 RGB，但增加物体颜色泄漏，则不应作为默认模块。

---

## 11. M3 细化消融

| ID | Attribution | Opacity decay | Scale shrink | Directional action | Prune |
|---|---|---:|---:|---:|---:|
| M3-0 | none |  |  |  |  |
| M3-1 | footprint | ✓ |  |  | ✓ |
| M3-2 | true contribution | ✓ |  |  | ✓ |
| M3-3 | true contribution | ✓ | ✓ |  | ✓ |
| M3-4 | true contribution | ✓ | ✓ | ✓ | ✓ |

重点对照：

```text
projected footprint
vs.
front-to-back true contribution
```

主要指标：

- water-dominant Gaussian count
- pure-water accumulation mean
- pure-water accumulation p95
- object-boundary leakage
- final Gaussian count
- cleanup overhead

---

## 12. M4 细化消融

| ID | Appearance configuration |
|---|---|
| S0 | SH=0 |
| S1 | vanilla SH=3 |
| S2 | SH=3 + residual mean anchor |
| S3 | SH=3 + DC softclip |
| S4 | SH=3 + residual mean anchor + DC softclip |

主要比较：

- underwater RGB PSNR/SSIM/LPIPS
- clean restoration PSNR/SSIM/LPIPS
- saturated pixel ratio
- white clipping ratio
- red dominance ratio
- cross-view appearance consistency

SH=0 应作为诊断下界，而不建议直接作为最终模型。

---

## 13. 伪深度补充实验

伪深度不再作为主线，只保留补充对照：

| ID | Pseudo-depth configuration |
|---|---|
| D0 | no pseudo-depth |
| D1 | weak masked Pearson |
| D2 | full-image Pearson, lambda=0.1 |

需要分别报告：

- underwater RGB
- clean restoration
- saturation
- color balance
- pure-water accumulation

如果 D2 提高带水体 RGB 但降低去水体质量，应将其报告为负向结果或局限性，而不是主贡献。

---

## 14. 评价指标体系

# 14.1 带水体新视角合成

```text
PSNR
SSIM
LPIPS
```

# 14.2 去水体恢复质量

在具有 clean ground truth 的 FUNA 数据上计算：

```text
Clean PSNR
Clean SSIM
Clean LPIPS
Delta E 2000
saturated-pixel ratio
white-clipping ratio
red-dominance ratio
```

建议定义：

\[
R_{\mathrm{sat}}
=
\frac{1}{HW}
\sum_p
\mathbb I
[
\max_c J(p,c) > \tau_{\mathrm{sat}}
].
\]

红色优势比例可以定义为：

\[
R_{\mathrm{red}}
=
\frac{1}{HW}
\sum_p
\mathbb I
[
J_R(p)-\max(J_G(p),J_B(p))>\delta
].
\]

# 14.3 几何与水体分离

建议建立独立于训练 ownership mask 的 evaluation water mask。

计算：

```text
water-region accumulation mean
water-region accumulation p95
water-region opacity mass
water-dominant Gaussian count
object-boundary leakage
water/object contribution ratio
```

定义：

\[
r_g
=
\frac{C_g^{water}}
{C_g^{water}+C_g^{object}+\epsilon}.
\]

可将满足以下条件的 Gaussian 视为 water-dominant：

\[
r_g > 0.8,
\qquad
C_g^{water} > \tau_c.
\]

# 14.4 效率

```text
training time
rendering FPS
peak VRAM
final Gaussian count
cleanup CUDA overhead
additional parameter count
```

---

## 15. 实验数据与阶段安排

# 15.1 第一阶段：单场景快速筛选

选择一个具有以下特点的场景：

- 大面积远处纯水
- 前景具有复杂物体边界
- J_dry 中容易出现蓝绿残留
- SH=3 容易出现过曝或红偏

训练设置：

```text
5k or 7.5k iterations
one random seed
fixed initialization
fixed train/eval split
```

目标是快速排除明显无效配置。

# 15.2 第二阶段：完整实验

对保留下来的配置执行：

```text
15k iterations
all selected scenes
3 random seeds
same initialization protocol
same evaluation code
```

报告：

```text
mean ± standard deviation
```

# 15.3 推荐数据分工

```text
SeaThru-NeRF:
underwater novel-view rendering
real-scene qualitative restoration
runtime comparison

FUNA:
paired underwater/clean restoration
depth and normal analysis
water-condition generalization
geometry-medium separation evaluation
```

---

## 16. 单元测试与回归测试

至少需要以下测试。

### 16.1 原始模式数值回归

```python
assert torch.allclose(
    original_outputs["rgb"],
    refactored_outputs["rgb"],
    atol=1e-5,
    rtol=1e-4,
)
```

同时测试：

```text
depth
accumulation
rgb_object
rgb_medium
rgb_clear
loss
```

### 16.2 Compositor 测试

```text
M_obj = 1 → output = near branch
M_obj = 0 → output = B_inf
high accumulation → occupancy-limited B_inf cannot fully take over
```

### 16.3 Contribution 测试

在小尺寸场景中实现 PyTorch reference，并与 CUDA true-contribution 对比。

### 16.4 Cleanup state 测试

```text
split → state buffer extends correctly
duplicate → state buffer extends correctly
prune → state buffer is cropped correctly
optimizer state → reset only for selected Gaussian
```

### 16.5 Module-disabled 测试

关闭模块时：

```text
no unused parameters
no unused optimizer group
no additional checkpoint state
no change to forward behavior
```

---

## 17. 每次实验必须保存的信息

```text
experiment name
full YAML config
Git commit SHA
dataset version
train/eval split
random seed
dependency versions
CUDA version
training time
peak VRAM
rendering FPS
final Gaussian count
underwater RGB metrics
clean restoration metrics
water-region accumulation
water-dominant Gaussian count
J_dry saturation
J_dry red dominance
qualitative output paths
```

建议输出：

```text
config.yaml
metrics.json
metrics.csv
environment.txt
git_commit.txt
renders/
diagnostics/
```

---

## 18. 推荐的 Git 分支结构

```text
main
├── baseline/original
├── refactor/core
├── feature/context-medium
├── feature/infinite-water
├── feature/contribution-cleanup
├── feature/constrained-sh
├── experiment/main-ablation
└── experiment/full-model
```

每个 feature 分支合并前必须通过：

```text
unit tests
baseline regression
one-scene smoke test
```

---

## 19. 预期论文方法逻辑

最终方法可以形成如下闭环：

```text
Context-aware medium field
        ↓
more appropriate water degradation modeling
        ↓
explicit infinite-water background
        ↓
pixel-level water/object ownership
        ↓
occupancy-limited closed composition
        ↓
true-contribution Gaussian attribution
        ↓
structure-level pure-water Gaussian cleanup
        ↓
residual-SH and DC gauge
        ↓
stable view-dependent rendering and cleaner intrinsic appearance
```

可概括为：

> The framework explicitly transfers infinite-water appearance from finite Gaussian geometry to a closed medium-background branch, then removes residual water-dominant Gaussian capacity through render-order contribution attribution, while constraining residual spherical harmonics to preserve necessary view-dependent appearance without corrupting intrinsic scene color.

---

## 20. 当前最推荐的执行计划

### 阶段一：准备

- 冻结并归档当前 Stage10 研究仓库
- 新建原始 WaterSplatting 重构仓库
- 固定数据、环境和评估协议
- 复现原始基线

### 阶段二：代码重构

- 拆分原始 medium、appearance、renderer 和 loss
- 验证数值一致性
- 建立统一配置系统
- 建立实验日志与指标保存工具

### 阶段三：四模块实现

1. M1 Context-Aware Medium Modeling
2. M2 Infinite-Water Ownership
3. M3 Contribution-Aware Gaussian Cleanup
4. M4 Constrained View-Dependent Appearance

### 阶段四：快速消融

- 单场景
- 5k 至 7.5k iterations
- 单 seed
- 主消融与关键交互消融

### 阶段五：正式实验

- 全场景
- 15k iterations
- 3 seeds
- underwater RGB、clean restoration、separation 和 efficiency 全指标

---

## 21. 最终结论

重新 clone 原始 WaterSplatting，并在其上重新实现四个语义清晰的模块，是当前最合理且风险最低的技术路线。

这一路线有三项直接收益：

1. 可以保留可信、可复现的原始 WaterSplatting 基线。
2. 每个模块都能形成清晰的因果消融和论文贡献。
3. 可以避免当前代码中 Stage 历史、重复监督、诊断分支和 cleanup state 的持续耦合。

后续的研究主线建议集中于：

```text
M1: context-aware water degradation
M2: infinite-water ownership transfer
M3: render-order Gaussian cleanup
M4: constrained view-dependent intrinsic appearance
```

其中：

- M2 与 M3 是解决远处纯水 Gaussian 错误归属的核心结构创新。
- M4 解决 SH=3 的视角表达能力与去水体颜色稳定性之间的矛盾。
- M1 提供更合理的 medium variation 表达，并减少水体变化被 Gaussian SH 吸收。

伪深度、canonical branch、attenuation spectral order 等机制暂时保留为补充实验或后续研究，不应继续占据主框架。
