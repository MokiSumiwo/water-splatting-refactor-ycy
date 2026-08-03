# GMVC：几何锚定的多视图水体参数校准实验报告

**英文名称：** Geometry-Anchored Multi-View Medium Calibration for Water Splatting  
**建议缩写：** GMVC  
**日期：** 2026-08-03  
**基线：** WaterSplatting M1，`medium_context_mode=dir_xy_camera`，`b_inf_mode=tied`  
**建议分支：** `research/gmvc-medium-calibration`  
**建议起点：** 从纯 M1 正式分支或对应 M1 commit 建分支，不从 MCMC-WS 实验分支继续叠加。

---

## 1. 实验背景

现有 M1 通过在 medium field 中引入相机视角、图像平面位置和相机相关信息，提高了水体参数对不同视角的适应能力。该设计能够改善最终新视角 RGB 重建，但水下成像模型仍然存在明显的参数不可辨识性：

- Gaussian 固有颜色未知；
- 直接信号衰减系数未知；
- 后向散射系数未知；
- 无穷远水体颜色未知；
- 多组不同的参数组合可以生成相近的最终 RGB。

因此，M1 当前优化目标更容易确定的是最终组合量：

\[
\mathbf I \approx \mathbf J \odot \mathbf T + \mathbf B,
\]

而不是分别唯一确定：

\[
\mathbf J,\quad \boldsymbol\beta^D,\quad
\boldsymbol\beta^B,\quad \mathbf B_\infty.
\]

本实验不再从 Gaussian densification、refinement 或 capacity control 入手，而是利用同一三维表面点在不同训练视角下具有不同传播距离这一事实，为水体参数和 Gaussian 固有颜色的联合分解增加显式的多视图物理约束。

---

## 2. M1 水体参数诊断结果

已从四个场景的 M1 15k checkpoint 渲染：

- `medium_bs`
- `medium_attn`
- `B_inf / medium_rgb`
- `transmission`
- `backscatter_endpoint`
- `actual_rgb_medium`
- depth

所有视角均使用场景级统一的 `p01/p50/p99` 显示范围，没有逐图 min-max 拉伸。

### 2.1 场景级均值

| Scene | bs mean RGB | attn mean RGB | B_inf mean RGB | transmission mean RGB | actual backscatter mean |
|---|---:|---:|---:|---:|---:|
| JapaneseGradens | `[0.5238, 0.4846, 0.5261]` | `[0.3438, 0.3404, 0.3627]` | `[0.2123, 0.2649, 0.3076]` | `[0.4119, 0.4137, 0.3911]` | `0.1900` |
| IUI3 | `[0.4974, 0.3857, 0.3164]` | `[0.1410, 0.1626, 0.1674]` | `[0.2629, 0.3347, 0.4276]` | `[0.4901, 0.4318, 0.4252]` | `0.2647` |
| Curasao | `[0.3895, 0.3736, 0.3726]` | `[0.5103, 0.5353, 0.5322]` | `[0.1394, 0.1677, 0.2099]` | `[0.2412, 0.2277, 0.2319]` | `0.1265` |
| Panama | `[0.3656, 0.2619, 0.2630]` | `[0.3530, 0.4093, 0.4285]` | `[0.1715, 0.2158, 0.2605]` | `[0.4253, 0.3722, 0.3559]` | `0.1195` |

### 2.2 关键诊断结论

1. **最终 RGB 拟合良好，但内部参数存在明显跨视角补偿。** 例如 Curasao 相邻 eval views 的平均 `medium_attn` 从约 `[0.412, 0.424, 0.414]` 增加到 `[0.564, 0.596, 0.594]`，同时 \(B_\infty\) 从 `[0.170, 0.217, 0.282]` 降低到 `[0.127, 0.145, 0.176]`。衰减变强和水体颜色变暗相互补偿，但最终 RGB 仍保持良好。

2. **水体参数图整体平滑，没有直接复制场景纹理。** 这说明 medium field 没有退化成显式的纹理记忆网络。但参数图存在大范围水平、垂直和对角渐变，且相邻视角之间变化幅度较大，表明相机条件具有较强自由度。

3. **原始系数的可解释性弱于组合量。** Transmission 和实际 backscatter contribution 明显受深度控制，具有合理的场景层次；而 \(\beta^D\)、\(\beta^B\) 和 \(B_\infty\) 的独立变化更自由。

4. **简化后向散射公式具有可用性。** 多数视角中：

\[
\mathbf B_{\mathrm{endpoint}}
=
\mathbf B_\infty
\odot
\left(1-\exp(-\boldsymbol\beta^B d)\right)
\]

与 renderer 输出的 `actual_rgb_medium` 接近。后续校准可先基于该显式公式，不必第一版就完整复现逐 Gaussian 分段积分。

5. **存在明显的无效区域。** Depth、transmission 和 medium contribution 图中存在低 alpha、空背景、错误深度、遮挡边界和弧形/矩形缺失区域。未来多视图约束必须只使用高置信度表面轨迹。

---

## 3. 核心研究假设

对同一个三维表面点 \(p\)，在视角 \(i\) 下的水下退化观测可以写为：

\[
\mathbf I_{i,p}
=
\mathbf J_p
\odot
\exp(-\boldsymbol\beta^D_{i,p}d_{i,p})
+
\mathbf B_{\infty,i,p}
\odot
\left[
1-\exp(-\boldsymbol\beta^B_{i,p}d_{i,p})
\right],
\]

其中：

- \(\mathbf I_{i,p}\)：训练图像中的观测 RGB；
- \(\mathbf J_p\)：该三维表面点的固有辐射颜色；
- \(d_{i,p}\)：相机到表面点的传播距离；
- \(\boldsymbol\beta^D_{i,p}\)：直接信号衰减系数；
- \(\boldsymbol\beta^B_{i,p}\)：后向散射增长系数；
- \(\mathbf B_{\infty,i,p}\)：无穷远水体颜色。

若几何、对应和水体参数合理，则不同视角反演出的固有颜色：

\[
\widehat{\mathbf J}_{i,p}
=
\frac{
\mathbf I_{i,p}
-
\mathbf B_{\infty,i,p}
\odot
\left[
1-\exp(-\boldsymbol\beta^B_{i,p}d_{i,p})
\right]
}{
\exp(-\boldsymbol\beta^D_{i,p}d_{i,p})+\epsilon
}
\]

应该满足：

\[
\widehat{\mathbf J}_{i,p}
\approx
\widehat{\mathbf J}_{j,p}.
\]

本实验的核心不是要求不同视角的水体参数完全相同，而是要求：

> 同一个三维表面点经过不同传播距离和不同视角的水体参数反演后，恢复出一致的固有颜色。

同时假设同一场景中的水体参数处于有限变化范围内：

\[
\boldsymbol\beta^D_{i,p}
=
\overline{\boldsymbol\beta}^{D}
+
\Delta\boldsymbol\beta^D_{i,p},
\]

\[
\boldsymbol\beta^B_{i,p}
=
\overline{\boldsymbol\beta}^{B}
+
\Delta\boldsymbol\beta^B_{i,p},
\]

\[
\mathbf B_{\infty,i,p}
=
\overline{\mathbf B}_{\infty}
+
\Delta\mathbf B_{\infty,i,p},
\]

其中局部残差允许存在，但不应在相邻视角之间任意漂移。

---

## 4. 实验目标

### 4.1 主要目标

1. 降低同一三维表面点的跨视角反演固有颜色方差；
2. 降低 `medium_attn`、`medium_bs` 和 \(B_\infty\) 的无约束跨视角漂移；
3. 改善水体参数与 Gaussian 固有颜色之间的可辨识性；
4. 保持 M1 的新视角 RGB 重建质量，不要求必须显著提升。

### 4.2 次要目标

1. 提高水体参数图的跨视角连续性；
2. 减少参数补偿，例如“衰减变强、\(B_\infty\) 同时变暗”；
3. 得到更稳定、更可解释的 clear color / intrinsic radiance；
4. 为论文提供独立于 PSNR 的物理一致性证据。

### 4.3 本轮允许的结果

本实验不以“必须提高 PSNR/SSIM/LPIPS”为唯一成功条件。只要新视角指标没有明显下降，同时物理一致性显著改善，即可认为方向成立。

---

## 5. 方法设计：GMVC

### 5.1 几何锚定的多视图表面轨迹

不依赖人工选择纯白物体，而是从几何稳定后的 M1 中自动构造高置信度三维表面轨迹。

对训练视角 \(i\) 中的像素 \(\mathbf u_i\)：

1. 使用渲染深度 \(D_i(\mathbf u_i)\) 反投影到三维：

\[
\mathbf X_p
=
\Pi_i^{-1}
\left(
\mathbf u_i,D_i(\mathbf u_i)
\right).
\]

2. 将 \(\mathbf X_p\) 投影到其他训练视角 \(j\)：

\[
\mathbf u_j
=
\Pi_j(\mathbf X_p).
\]

3. 使用目标视角渲染深度验证可见性：

\[
e^d_{ij,p}
=
\frac{
|D_j(\mathbf u_j)-d_{j,p}^{\mathrm{proj}}|
}{
D_j(\mathbf u_j)+\epsilon
}.
\]

4. 仅当 \(e^d_{ij,p}<\tau_d\)、alpha 足够高、深度方差足够低且不在图像边界时，将两个像素视为同一表面点。

建议轨迹长度：

\[
|\mathcal V_p|\ge 3.
\]

两个视角虽然可以形成颜色差异，但不足以稳健分离多种水体参数。

### 5.2 轨迹可靠性筛选

每个轨迹需要满足：

- alpha \(\ge 0.95\)；
- 相对深度一致性误差 \(\le 0.02\)；
- 轨迹长度 \(\ge 3\)；
- 像素不在图像边缘；
- 目标视角坐标位于有效成像区域；
- transmission 不低于 0.10，避免反演除法放大噪声；
- 避开高深度方差和多层混合区域；
- 传播距离跨度足够大：

\[
\Delta d_p
=
\max_i d_{i,p}-\min_i d_{i,p}
\ge \tau_{\Delta d}.
\]

建议 \(\tau_{\Delta d}\) 不直接使用固定场景单位，而采用相对阈值：

\[
\frac{\Delta d_p}{\operatorname{median}_i(d_{i,p})}
\ge 0.05.
\]

如果有效轨迹过少，可降到 0.03；低于 0.02 的轨迹对水体参数可辨识性贡献很弱。

### 5.3 反演固有颜色一致性

对轨迹 \(p\) 的每个视角 \(i\)，计算：

\[
\mathbf T_{i,p}
=
\exp(-\boldsymbol\beta^D_{i,p}d_{i,p}),
\]

\[
\mathbf B_{i,p}
=
\mathbf B_{\infty,i,p}
\odot
\left[
1-\exp(-\boldsymbol\beta^B_{i,p}d_{i,p})
\right],
\]

\[
\widehat{\mathbf J}_{i,p}
=
\frac{
\mathbf I_{i,p}-\mathbf B_{i,p}
}{
\mathbf T_{i,p}+\epsilon
}.
\]

轨迹共识颜色：

\[
\overline{\mathbf J}_p
=
\frac{
\sum_{i\in\mathcal V_p}
w_{i,p}\widehat{\mathbf J}_{i,p}
}{
\sum_{i\in\mathcal V_p}w_{i,p}+\epsilon
}.
\]

建议对共识颜色使用 stop-gradient：

\[
\mathcal L_{\mathrm{J-cons}}
=
\frac{
\sum_p\sum_{i\in\mathcal V_p}
w_{i,p}
\rho
\left(
\widehat{\mathbf J}_{i,p}
-
\operatorname{sg}(\overline{\mathbf J}_p)
\right)
}{
\sum_p\sum_iw_{i,p}+\epsilon
}.
\]

其中 \(\rho\) 使用 Charbonnier 或 Huber loss，避免少量错误对应破坏训练。

### 5.4 可靠性权重

建议：

\[
w_{i,p}
=
w^\alpha_{i,p}
w^d_{i,p}
w^T_{i,p}
w^{\Delta d}_p.
\]

各项可定义为：

\[
w^\alpha_{i,p}
=
\operatorname{clamp}
\left(
\frac{\alpha_{i,p}-\tau_\alpha}{1-\tau_\alpha},
0,1
\right),
\]

\[
w^d_{i,p}
=
\exp
\left(
-\frac{e^d_{i,p}}{\sigma_d}
\right),
\]

\[
w^T_{i,p}
=
\operatorname{clamp}
\left(
\frac{T_{i,p}-T_{\min}}{1-T_{\min}},
0,1
\right),
\]

\[
w^{\Delta d}_p
=
\operatorname{clamp}
\left(
\frac{\Delta d_p}{\tau_{\Delta d,\mathrm{high}}},
0,1
\right).
\]

不建议第一版加入语义类别、目标检测、白色物体识别或人工 mask。

### 5.5 场景级水体参数范围约束

不能直接强制所有视角参数相同。建议对正值系数在 log-domain 中限制其偏离场景中心：

\[
\boldsymbol\mu_D
=
\operatorname{sg}
\left[
\operatorname{median}
\left(
\log\boldsymbol\beta^D
\right)
\right],
\]

\[
\boldsymbol\mu_B
=
\operatorname{sg}
\left[
\operatorname{median}
\left(
\log\boldsymbol\beta^B
\right)
\right].
\]

\[
\mathcal L_{\mathrm{range}}
=
\rho
\left(
\frac{\log\boldsymbol\beta^D-\boldsymbol\mu_D}{s_D}
\right)
+
\rho
\left(
\frac{\log\boldsymbol\beta^B-\boldsymbol\mu_B}{s_B}
\right).
\]

对 \(B_\infty\) 使用较弱约束：

\[
\mathcal L_{B_\infty}
=
\rho
\left(
\mathbf B_\infty-\overline{\mathbf B}_\infty
\right).
\]

第一轮建议只对高置信度轨迹上的参数计算范围约束，不对整张图施加全局收缩。

### 5.6 总损失

保留原 M1 RGB reconstruction loss：

\[
\mathcal L_{\mathrm{M1}}.
\]

GMVC 总损失：

\[
\mathcal L
=
\mathcal L_{\mathrm{M1}}
+
\lambda_J\mathcal L_{\mathrm{J-cons}}
+
\lambda_R\mathcal L_{\mathrm{range}}
+
\lambda_\infty\mathcal L_{B_\infty}.
\]

第一版不加入：

- pseudo depth loss；
- correspondence network；
- Gaussian densification modification；
- opacity suppression；
- water ownership；
- clear-image supervision；
- \(J\) 图真值；
- 单独的白色物体先验。

---

## 6. 优化阶段

GMVC 应在几何和 densification 基本稳定后启用。

### Stage 0：M1 几何与外观学习

```text
step 0–10000
```

- 完全使用原 M1；
- GMVC 关闭；
- 保留现有 densification；
- 训练到 `stop_split_at=10000`。

### Stage 1：水体参数校准

```text
step 10000–11500
```

- 几何参数停止接收 GMVC 梯度；
- 建议冻结：means、scales、quats、opacities；
- 只更新 medium field；
- 保留原 RGB reconstruction loss；
- \(\lambda_J\) 从 0 线性 ramp 到目标值。

### Stage 2：水体参数与固有颜色联合调整

```text
step 11500–14000
```

更新：

- medium field；
- Gaussian `features_dc`；
- 可选低学习率更新 `features_rest`。

仍冻结：

- means；
- scales；
- quats；
- opacities。

这一阶段允许模型根据校准后的水体参数逐渐修正 Gaussian 固有颜色。

### Stage 3：稳定收尾

```text
step 14000–15000
```

- \(\lambda_J\) 保持或下降到目标值的 50%；
- \(\lambda_R\) 保持；
- 继续原 RGB loss；
- 不重新开启 densification；
- 不重新开启 opacity reset。

---

## 7. 第一轮实验设计

### 7.1 Phase A：只做可辨识性诊断

使用现有 M1 15k checkpoint，不训练。

输出：

- 有效三维轨迹数量；
- track length 分布；
- depth span 分布；
- alpha 分布；
- transmission 分布；
- 反演 \(\widehat{\mathbf J}\) 的跨视角方差；
- 参数在轨迹内的跨视角方差；
- `medium_attn` 与 \(B_\infty\) 的补偿相关性；
- 每个场景满足不同阈值后的轨迹保留率。

继续条件：

```text
有效轨迹 >= 10000
长度 >= 3 的轨迹比例 >= 30%
相对 depth span >= 0.05 的轨迹 >= 5000
T > 0.10 的有效观测比例 >= 80%
```

如果一个场景不满足，先调整轨迹构建，不进入损失训练。

### 7.2 Phase B：M1 checkpoint continuation

建议使用 M1 step-9999 checkpoint 共享恢复到 15k。

| Variant | 设置 |
|---|---|
| C0 | M1 原始 continuation |
| C1 | GMVC diagnostic-only，计算轨迹和损失但不反传 |
| C2 | 仅 \(\mathcal L_{\mathrm{J-cons}}\)，medium-only |
| C3 | \(\mathcal L_{\mathrm{J-cons}}+\mathcal L_{\mathrm{range}}\)，medium-only |
| C4 | C3 + `features_dc` 联合更新 |
| C5 | C4 + 弱 \(B_\infty\) 场景中心约束，仅在 C4 稳定时运行 |

C1 必须与 C0 保持一致。如果 C1 对训练轨迹产生变化，则优先修复 diagnostic path 的 side effect。

### 7.3 推荐场景顺序

1. JapaneseGradens：主要机制实验；
2. IUI3：高后向散射场景安全性；
3. Curasao：强衰减场景；
4. Panama：明显跨视角 \(B_\infty\) 漂移场景。

---

## 8. 初始超参数

### 8.1 轨迹参数

```python
gmvc_enabled = False
gmvc_start_step = 10000
gmvc_stop_step = 15000
gmvc_ramp_steps = 500

gmvc_track_min_views = 3
gmvc_alpha_threshold = 0.95
gmvc_depth_rel_threshold = 0.02
gmvc_relative_depth_span = 0.05
gmvc_transmission_min = 0.10
gmvc_max_tracks_per_step = 4096
gmvc_track_refresh_interval = 500
```

### 8.2 损失权重

不建议只依据 loss 数值选择 \(\lambda\)，应记录 auxiliary loss 对 medium 参数的梯度范数，使其初始梯度强度约为 RGB loss 对 medium 梯度的 5%–10%。

初始候选：

```python
lambda_gmvc_j = 0.002
lambda_gmvc_range = 0.0002
lambda_gmvc_binf = 0.00005
```

若 gradient ratio 小于 2%，可将 `lambda_gmvc_j` 提高至 0.005；若超过 15%，应降低。

### 8.3 数值稳定

```python
gmvc_eps = 1e-4
gmvc_j_clamp_min = -0.25
gmvc_j_clamp_max = 1.25
gmvc_huber_delta = 0.03
```

超出合理范围的 \(\widehat{\mathbf J}\) 不应直接硬裁剪后继续赋予高权重，而应记录为 invalid 或降低权重。

---

## 9. 评价指标

### 9.1 新视角重建指标

继续使用：

- PSNR；
- SSIM；
- LPIPS。

### 9.2 物理一致性指标

#### A. 反演固有颜色方差

\[
E_J
=
\frac{1}{|\mathcal P|}
\sum_p
\frac{
\sum_{i\in\mathcal V_p}
w_{i,p}
\|\widehat{\mathbf J}_{i,p}-\overline{\mathbf J}_p\|_1
}{
\sum_iw_{i,p}+\epsilon
}.
\]

这是本实验最主要的物理指标。

#### B. 视角间参数变异系数

对每个场景计算：

\[
CV(\beta^D_c)
=
\frac{
\operatorname{std}_{view}
\left[
\operatorname{mean}_{pixel}(\beta^D_c)
\right]
}{
\operatorname{mean}_{view,pixel}(\beta^D_c)+\epsilon
}.
\]

同样计算：

- \(CV(\beta^B_c)\)；
- \(CV(B_{\infty,c})\)。

#### C. 轨迹内参数变化

同一三维轨迹上的：

\[
E_{\beta^D},\quad
E_{\beta^B},\quad
E_{B_\infty}.
\]

该指标不要求趋近于零，只要求明显减少无约束漂移。

#### D. 参数补偿相关性

统计跨视角：

\[
\Delta\beta^D
\quad\text{与}\quad
\Delta B_\infty
\]

以及：

\[
\Delta\beta^B
\quad\text{与}\quad
\Delta B_\infty
\]

的负相关程度。过强负相关通常说明参数在互相补偿。

#### E. 简化公式一致性

\[
E_B
=
\|\mathbf B_{\mathrm{endpoint}}-\mathbf B_{\mathrm{actual}}\|_1.
\]

确认 GMVC 没有使显式近似与实际 renderer 分解严重偏离。

### 9.3 可视化

每个场景至少输出：

- GT；
- M1 baseline render；
- GMVC render；
- absolute RGB error；
- `medium_bs`；
- `medium_attn`；
- \(B_\infty\)；
- transmission；
- actual backscatter；
- \(\widehat{\mathbf J}\)；
- 跨视角 \(\widehat{\mathbf J}\) variance map；
- 多视图参数差异图。

所有参数图继续使用场景级统一显示范围。

---

## 10. 成功标准

本实验不要求 RGB 指标必须提升。

### 10.1 质量保持 gate

相对于同 checkpoint、同训练步数的 C0：

```text
平均 PSNR 下降 <= 0.15 dB
任何单场景 PSNR 下降 <= 0.25 dB

平均 SSIM 下降 <= 0.0015
任何单场景 SSIM 下降 <= 0.0030

平均 LPIPS 增加 <= 0.0030
任何单场景 LPIPS 增加 <= 0.0050
```

### 10.2 物理一致性 gate

至少满足以下四项中的两项：

```text
反演固有颜色误差 E_J 降低 >= 20%
B_inf 跨视角 CV 降低 >= 15%
medium_attn 或 medium_bs 跨视角 CV 降低 >= 15%
参数补偿相关性绝对值降低 >= 20%
```

### 10.3 强成功

若同时满足：

```text
PSNR / SSIM / LPIPS 不劣于 M1
E_J 降低 >= 20%
至少两类水体参数跨视角变化下降 >= 15%
```

则可以进入完整四场景 15k 与论文方法主线。

### 10.4 可接受成功

若 RGB 指标存在轻微下降但在质量保持 gate 内，同时物理一致性明显提升，则仍可认为方法成立。论文中应将其定位为：

> 以极小的图像质量代价换取更可辨识、更跨视图一致的水体—场景分解。

---

## 11. 风险与解释

### 11.1 几何误差污染参数

错误深度会直接影响指数衰减。若低置信度区域占比高，GMVC 可能把几何误差解释为水体变化。

处理：严格 visibility/depth check、geometry detach、仅使用低深度方差区域、错误轨迹不得反传。

### 11.2 非朗伯反射和照明变化

同一表面点的无水颜色不一定严格相同，原因包括高光、水面焦散、阴影、人工光源、自动曝光和白平衡。

处理：使用 robust loss、剔除颜色离群视角、不强制所有轨迹完全一致；后续可考虑只约束低频色度，但第一版先不加入。

### 11.3 Transmission 太低导致反演爆炸

当 \(T\rightarrow0\) 时：

\[
\widehat{\mathbf J}
=(\mathbf I-\mathbf B)/T
\]

会极不稳定。

处理：`transmission_min=0.10`、权重随 \(T\) 降低、记录 invalid ratio、不对低 T 像素强行优化。

### 11.4 参数范围约束过强

若直接强制所有水体参数接近统一值，会将误差转移到 Gaussian 颜色，并可能降低新视角指标。

处理：log-domain 软约束、使用 robust scene center、保留局部残差、\(\lambda_R\) 明显小于 \(\lambda_J\)。

### 11.5 指标轻微下降

这不一定说明方向错误。M1 当前可能依赖视角特定参数补偿提高 RGB 拟合；GMVC 减少该自由度后，可能牺牲少量 PSNR，但得到更一致的物理分解。

---

## 12. 需要记录的 JSONL 诊断

每个 GMVC update 记录：

```text
step
camera_ids
track_count
track_length_mean / p50 / p95
depth_span_mean / p50 / p95
valid_observation_ratio
alpha_mean
depth_consistency_error
transmission_mean / p05
J_consistency_loss
range_loss
B_inf_loss
RGB_loss
aux_to_rgb_gradient_ratio

J_variance_before / after
medium_attn_track_variance
medium_bs_track_variance
B_inf_track_variance

invalid_low_T_count
invalid_depth_count
invalid_alpha_count
invalid_out_of_bounds_count
```

每次 eval 记录：

```text
PSNR / SSIM / LPIPS
scene-level parameter means
scene-level parameter CV
inverse-radiance consistency
endpoint-vs-actual-backscatter error
```

---

## 13. 工程约束

GMVC 必须满足：

- 不修改 underwater renderer；
- 不增加推理时网络；
- 不依赖冻结外部模型；
- 不使用 clear-image GT；
- 不使用人工语义标注；
- 不改变 M1 的基础水下成像公式；
- 不修改 densification；
- 不引入 pseudo depth；
- 几何仅用于提供对应和传播距离；
- 推理阶段不需要保存轨迹或 correspondence。

建议目录：

```text
water_splatting/
└── medium_calibration/
    ├── __init__.py
    ├── gmvc_tracks.py
    ├── gmvc_losses.py
    ├── gmvc_diagnostics.py
    └── gmvc_types.py

scripts/
├── diagnostics/
│   ├── build_gmvc_tracks.py
│   └── diagnose_gmvc_identifiability.py
└── experiments/
    ├── gmvc_common.sh
    ├── gmvc_c0_m1_*.sh
    ├── gmvc_c1_diag_*.sh
    ├── gmvc_c2_jcons_*.sh
    ├── gmvc_c3_jcons_range_*.sh
    └── gmvc_c4_joint_dc_*.sh
```

---

## 14. 预期结论与论文价值

若实验成立，可以形成以下核心结论：

1. 基于 RGB reconstruction 的水下 3DGS 可以获得准确的新视角图像，但内部水体参数和场景固有颜色仍存在明显的非唯一分解；
2. 相邻视角的 medium 参数可能通过互相补偿保持最终 RGB，而不对应稳定的水体属性；
3. 几何稳定后的三维表面轨迹提供了跨视图传播距离变化；
4. 利用该变化约束反演固有辐射一致性，可以显式减少水体参数与 Gaussian 颜色之间的歧义；
5. 该约束不要求水体参数严格全局恒定，而是允许其围绕场景级中心有限变化；
6. 即使 PSNR 仅保持或轻微下降，只要物理分解显著稳定，也具有独立的方法贡献。

建议的方法表述：

> We introduce a geometry-anchored multi-view medium calibration mechanism that converts depth variation along cross-view surface tracks into explicit constraints on underwater image formation. Rather than forcing medium parameters to be globally identical, the proposed calibration requires different degraded observations of the same 3D surface point to recover consistent intrinsic radiance, thereby reducing the ambiguity between Gaussian appearance, direct-signal attenuation, and backscatter.

---

## 15. 最终执行顺序

```text
从纯 M1 建立 GMVC 分支
        ↓
离线构建高置信度多视图表面轨迹
        ↓
完成 identifiability diagnostic
        ↓
确认 track 数量、长度和 depth span
        ↓
共享 step-9999 checkpoint
        ↓
C0 / C1 exact-control gate
        ↓
C2 medium-only J consistency
        ↓
C3 + parameter range
        ↓
C4 + Gaussian DC joint refinement
        ↓
JapaneseGradens quality-preservation gate
        ↓
IUI3 / Curasao / Panama
        ↓
四场景物理一致性与 RGB 综合评估
```

---

## 16. 本轮决策原则

本实验的首要问题不是：

> 能否再提升 0.1 dB PSNR？

而是：

> 能否在几乎不损失新视角质量的前提下，使同一三维物体在不同传播距离下恢复出更加一致的固有颜色，并使水体参数不再进行任意的跨视角补偿？

因此，实验应同时报告：

- 新视角合成质量；
- 水体参数跨视角稳定性；
- 反演固有颜色一致性；
- 参数补偿程度；
- 失败区域与对应置信度。

只有同时观察 RGB 和物理分解，才能判断 GMVC 是否真正解决了 M1 当前的核心欠约束。

---

## 17. Codex Implementation Pass：M1 基线重启与 Phase A 诊断

本轮已按报告建议从纯 M1 代码起点重启，而不是继续叠加 MCMC-WS、GDADC、IGAF 等后续实验分支。

### 17.1 分支与代码起点

- 新分支：`research/gmvc-medium-calibration`
- 起点提交：`0de407b Restore M1 code baseline`
- 当前目标：只实现 Phase A identifiability diagnostic，不先进入训练 loss。

新增代码：

```text
water_splatting/medium_calibration/__init__.py
water_splatting/medium_calibration/gmvc_types.py
water_splatting/medium_calibration/gmvc_losses.py
water_splatting/medium_calibration/gmvc_tracks.py
water_splatting/medium_calibration/gmvc_diagnostics.py
scripts/diagnostics/diagnose_gmvc_identifiability.py
scripts/experiments/gmvc_phase_a_identifiability_all_scenes.sh
```

核心实现：

- 从 M1 checkpoint 渲染 train/eval views；
- 使用 M1 rendered depth 反投影训练视角高置信像素到三维；
- 重投影到其他训练视角；
- 使用 depth consistency、alpha、depth std、edge、transmission 做 track 筛选；
- 对每条 track 计算 simplified image formation 下的反演固有颜色 \(\widehat J\)；
- 汇总 `E_J`、track length、depth span、T 有效比例、参数 track variance、endpoint-vs-actual backscatter error、参数补偿相关性。

### 17.2 验证命令

静态检查：

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/medium_calibration/*.py \
  scripts/diagnostics/diagnose_gmvc_identifiability.py

git diff --check
```

JapaneseGradens smoke：

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_gmvc_identifiability.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --split train \
  --max-images 3 \
  --samples-per-view 128 \
  --output-dir renders/gmvc_phase_a_smoke_japanesegradens_20260803
```

四场景 Phase A：

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_identifiability.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --split train --samples-per-view 4096 \
  --output-dir renders/gmvc_phase_a_identifiability_20260803/japanesegradens

CUDA_VISIBLE_DEVICES=7 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_identifiability.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --split train --samples-per-view 4096 \
  --output-dir renders/gmvc_phase_a_identifiability_20260803/iui3

CUDA_VISIBLE_DEVICES=8 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_identifiability.py \
  --load-config outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml \
  --split train --samples-per-view 4096 \
  --output-dir renders/gmvc_phase_a_identifiability_20260803/curasao

CUDA_VISIBLE_DEVICES=9 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_identifiability.py \
  --load-config outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml \
  --split train --samples-per-view 4096 \
  --output-dir renders/gmvc_phase_a_identifiability_20260803/panama
```

### 17.3 Phase A 结果

| Scene | Train views | Final tracks | Len≥3 ratio | T valid ratio | \(E_J\) mean | corr \(\Delta\beta^D,\Delta B_\infty\) | corr \(\Delta\beta^B,\Delta B_\infty\) | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| JapaneseGradens | 17 | 53,209 | 0.8372 | 0.9984 | 0.0544 | -0.2052 | -0.1509 | Pass |
| IUI3 | 25 | 91,114 | 0.9060 | 1.0000 | 0.0337 | -0.3929 | -0.0602 | Pass |
| Curasao | 18 | 59,507 | 0.8775 | 0.9411 | 0.0545 | -0.7130 | 0.6436 | Pass |
| Panama | 15 | 48,079 | 0.8262 | 0.9984 | 0.0378 | 0.1892 | 0.8461 | Pass |

Output JSONs:

```text
renders/gmvc_phase_a_identifiability_20260803/japanesegradens/gmvc_identifiability.json
renders/gmvc_phase_a_identifiability_20260803/iui3/gmvc_identifiability.json
renders/gmvc_phase_a_identifiability_20260803/curasao/gmvc_identifiability.json
renders/gmvc_phase_a_identifiability_20260803/panama/gmvc_identifiability.json
```

### 17.4 Phase A 结论

四个场景均满足报告中的继续条件：

- 有效 track 数均超过 10,000；
- 长度 ≥ 3 的 track 比例均显著高于 30%；
- 相对 depth span ≥ 0.05 的 track 数均超过 5,000；
- \(T>0.10\) 有效观测比例均高于 80%。

这说明 GMVC 的几何锚定多视图信号在当前 M1 checkpoint 上足够密集，值得进入 Phase B 的 continuation 训练实验。Curasao 的 low-T invalid 观测明显更多，后续训练中应保留 transmission gate，且不应降低 `gmvc_transmission_min`。

---

## 18. Codex Implementation Pass：Phase B JapaneseGradens 10k→15k

### 18.1 新增训练实现

本轮在 M1 基线代码上新增 GMVC continuation 训练路径：

```text
scripts/diagnostics/build_gmvc_tracks.py
water_splatting/medium_calibration/gmvc_training.py
scripts/experiments/gmvc_phase_b_common.sh
scripts/experiments/gmvc_c0_m1_japanesegradens_10k_to_15k.sh
scripts/experiments/gmvc_c1_diag_japanesegradens_10k_to_15k.sh
scripts/experiments/gmvc_c1_diag_freeze_japanesegradens_10k_to_15k.sh
scripts/experiments/gmvc_c2_jcons_japanesegradens_10k_to_15k.sh
scripts/experiments/gmvc_c3_jcons_range_japanesegradens_10k_to_15k.sh
```

新增 `WaterSplattingModelConfig` flags：

```python
gmvc_enabled: bool = False
gmvc_diagnostic_only: bool = False
gmvc_track_bank_path: Optional[str] = None
gmvc_start_step: int = 10000
gmvc_stop_step: int = 15000
gmvc_ramp_steps: int = 500
lambda_gmvc_j: float = 0.0
lambda_gmvc_range: float = 0.0
lambda_gmvc_binf: float = 0.0
gmvc_max_tracks_per_step: int = 4096
gmvc_eps: float = 1e-4
gmvc_charbonnier_eps: float = 1e-6
gmvc_j_clamp_min: float = -0.25
gmvc_j_clamp_max: float = 1.25
gmvc_range_log_scale: float = 0.25
gmvc_detach_depth: bool = True
gmvc_seed: int = 42
gmvc_freeze_geometry: bool = False
gmvc_train_features_dc: bool = False
gmvc_train_features_rest: bool = False
```

实现方式：

- 先从 M1 step 10000 checkpoint 构建 detached training track bank；
- 训练时按当前 camera index 采样对应轨迹观测；
- 使用当前 `medium_attn`、`medium_bs`、`b_inf` 和 GT RGB 反演 \(\widehat J\)；
- C2 对 \(\widehat J\) 施加离线 track consensus 监督；
- C3 额外对 `medium_attn` / `medium_bs` 的 log-domain track center 施加弱 range loss；
- C2/C3 中冻结 `means/scales/quats/opacities/features_dc/features_rest`，只让 medium branch 接收 GMVC loss 与 RGB loss。

### 18.2 验证与 smoke

静态检查通过：

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/medium_calibration/*.py \
  scripts/diagnostics/diagnose_gmvc_identifiability.py \
  scripts/diagnostics/build_gmvc_tracks.py \
  scripts/diagnostics/render_m1_medium_parameter_maps.py

bash -n scripts/experiments/gmvc_phase_b_common.sh
bash -n scripts/experiments/gmvc_c0_m1_japanesegradens_10k_to_15k.sh
bash -n scripts/experiments/gmvc_c1_diag_japanesegradens_10k_to_15k.sh
bash -n scripts/experiments/gmvc_c2_jcons_japanesegradens_10k_to_15k.sh
bash -n scripts/experiments/gmvc_c3_jcons_range_japanesegradens_10k_to_15k.sh
git diff --check
```

C2 two-step smoke 通过：

```bash
GPU=6 CONTINUATION_STEPS=2 RUN_EVAL=0 RUN_CLOSURE_DIAG=0 STEPS_PER_SAVE=1 \
STAMP=20260803_gmvc_c2_smoke \
GMVC_TRACK_BANK_PATH=/mnt/new/home_old/ycy/water-splatting-refactor/renders/gmvc_track_banks_smoke/japanesegradens_m1_step10000/gmvc_track_bank.pt \
scripts/experiments/gmvc_c2_jcons_japanesegradens_10k_to_15k.sh
```

结果：训练可启动，track bank 成功加载，无 CUDA/autograd 错误，checkpoint 正常写入。

### 18.3 JapaneseGradens training bank

构建命令：

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/build_gmvc_tracks.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --load-step 10000 \
  --split train \
  --samples-per-view 4096 \
  --max-observations-per-camera 20000 \
  --output-path renders/gmvc_track_banks/japanesegradens_redsea_m1_step10000_train_s4096/gmvc_track_bank.pt
```

Bank summary：

| Metric | Value |
|---|---:|
| train views | 17 |
| sampled source tracks | 69,632 |
| accepted tracks | 53,397 |
| accepted observations | 580,275 |
| per-camera cap | 20,000 |
| bank size | 20 MB |

### 18.4 Phase B 质量结果

所有实验均从 JapaneseGradens M1 step 10000 checkpoint 继续到 step 15000，并运行 `ns-eval`。

| Variant | PSNR | SSIM | LPIPS | ΔPSNR vs C0 | ΔSSIM vs C0 | ΔLPIPS vs C0 | J blue | J sat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 M1 continuation | 24.7551 | 0.8995 | 0.1200 | +0.0000 | +0.0000 | +0.0000 | 0.1413 | 0.0511 |
| C1 diagnostic-only | 24.7610 | 0.8995 | 0.1201 | +0.0059 | +0.0000 | +0.0001 | 0.1398 | 0.0510 |
| C1 diagnostic-only + geometry freeze | 24.8648 | 0.8963 | 0.1251 | +0.1098 | -0.0032 | +0.0050 | 0.1414 | 0.0574 |
| C2 J-consistency, λ=0.0001 | 24.8639 | 0.8963 | 0.1251 | +0.1088 | -0.0031 | +0.0050 | 0.1414 | 0.0574 |
| C2 J-consistency, λ=0.0005 | 24.8649 | 0.8963 | 0.1250 | +0.1098 | -0.0031 | +0.0050 | 0.1414 | 0.0574 |
| C2 J-consistency, λ=0.002 | 24.8655 | 0.8963 | 0.1250 | +0.1105 | -0.0031 | +0.0050 | 0.1414 | 0.0574 |
| C2 J-consistency, λ=0.020 | 24.8670 | 0.8963 | 0.1250 | +0.1120 | -0.0031 | +0.0050 | 0.1414 | 0.0574 |
| C3 J-consistency + range | 24.8653 | 0.8963 | 0.1251 | +0.1103 | -0.0031 | +0.0050 | 0.1414 | 0.0574 |

Key observations：

- C1 diagnostic-only matches C0 within noise, so diagnostic path has no material side effect.
- C1 freeze-only nearly exactly matches C2/C3; therefore the C2/C3 RGB change is mainly caused by freezing Gaussian geometry/culling behavior after 10k, not by GMVC calibration.
- C2/C3 increase PSNR by about 0.11 dB but reduce SSIM by about 0.0031 and increase LPIPS by about 0.0050.
- Under the single-scene quality gate, C2/C3 are at or just beyond the allowed SSIM/LPIPS degradation boundary and should not advance on RGB metrics alone.

Output JSONs：

```text
renders/gmvc_c0_m1_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_5k/output.json
renders/gmvc_c1_diag_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_5k/output.json
renders/gmvc_c1_diag_freeze_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_c1_freeze/output.json
renders/gmvc_c2_jcons_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_5k/output.json
renders/gmvc_c2_jcons_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_c2_lam0005/output.json
renders/gmvc_c2_jcons_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_c2_lam0001/output.json
renders/gmvc_c2_jcons_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_c2_lam002/output.json
renders/gmvc_c3_jcons_range_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_5k/output.json
```

### 18.5 Phase B identifiability 结果

| Variant | Final tracks | T valid ratio | \(E_J\) mean | Δ\(E_J\) vs C0 | corr \(\Delta\beta^D,\Delta B_\infty\) | corr \(\Delta\beta^B,\Delta B_\infty\) |
|---|---:|---:|---:|---:|---:|---:|
| C0 M1 continuation | 53,204 | 0.9981 | 0.05434 | +0.0% | -0.2054 | -0.1500 |
| C1 diagnostic-only | 53,051 | 0.9983 | 0.05442 | +0.1% | -0.1953 | -0.1479 |
| C1 diagnostic-only + geometry freeze | 53,401 | 0.9832 | 0.06065 | +11.6% | -0.2908 | -0.2493 |
| C2 J-consistency, λ=0.0001 | 53,252 | 0.9834 | 0.06079 | +11.9% | -0.2915 | -0.2517 |
| C2 J-consistency, λ=0.0005 | 53,247 | 0.9830 | 0.06058 | +11.5% | -0.2922 | -0.2529 |
| C2 J-consistency, λ=0.002 | 53,304 | 0.9835 | 0.06061 | +11.5% | -0.2896 | -0.2499 |
| C2 J-consistency, λ=0.020 | 53,379 | 0.9838 | 0.06081 | +11.9% | -0.2910 | -0.2636 |
| C3 J-consistency + range | 53,267 | 0.9835 | 0.06077 | +11.8% | -0.2837 | -0.2480 |

Physical gate result：

- \(E_J\) does not decrease; it worsens by about 11.5%–11.9% for all freeze/C2/C3 variants.
- Compensation correlation absolute value increases, especially for \(\Delta\beta^B\) vs \(B_\infty\).
- Valid transmission observation ratio drops from about 0.998 to about 0.983 after geometry freeze / C2 / C3.
- No tested GMVC variant satisfies the physical consistency gate.

Diagnostic JSONs：

```text
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c0/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c1/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c1_freeze/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c2/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c2_lam0001/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c2_lam0005/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c2_lam002/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c3/gmvc_identifiability.json
```

### 18.6 Phase B 结论

当前 GMVC medium-only loss 不应进入 IUI3、Curasao、Panama 或四场景 15k。

原因：

1. C2/C3 的 RGB 变化几乎完全由 `gmvc_freeze_geometry=True` 解释，而不是由 GMVC loss 本身解释；
2. C2/C3 虽然提高 PSNR，但 SSIM 与 LPIPS 明显变差；
3. 物理一致性主指标 \(E_J\) 反而恶化；
4. 参数补偿相关性绝对值上升，说明 medium 参数不可辨识性没有被缓解；
5. `lambda_gmvc_j` 从 0.0001 到 0.020 的 sweep 不能改变上述结论。

因此，本轮应停止当前 C2/C3 medium-only formulation。C4 不应直接运行，因为当前 loss 主要通过 GT RGB 与 medium 参数反演 \(\widehat J\)，不会对 `features_dc` 形成明确的 GMVC 梯度；仅打开 `gmvc_train_features_dc=True` 会把 C4 变成“RGB loss 下的 feature unfreeze”，而不是真正的 GMVC joint refinement。

下一步如果继续 GMVC，应先修改机制，而不是扩展场景：

- 方案 A：改成 current-batch / current-bank 的 online track variance loss，让同一 track 的当前多视角 \(\widehat J\) 互相一致，而不是只贴近 M1 step-10000 的 detached consensus；
- 方案 B：新增 renderer-consistent intrinsic appearance term，将 `outputs["J_gaussian_raw"]` 或 clear-proxy Gaussian intrinsic render 与 track consensus 对齐，使 C4 对 `features_dc` 有真实 GMVC 梯度；
- 方案 C：保留 geometry freeze 作为独立后处理消融，但不要把其 PSNR 增益归因于 GMVC。

### 18.7 C4 renderer-consistent intrinsic DC refinement 复核

为确认 C4 是否只是 `features_dc` unfreeze，本轮按上节方案 B 做了最小机制修改：在 `compute_gmvc_training_terms()` 中新增 `gmvc_intrinsic_loss`，从 `outputs["J_gaussian_raw"]`、`outputs["J_raw"]` 或 `outputs["J"]` 采样当前 intrinsic Gaussian render，并与 track bank 中 detached `j_consensus` 做 Charbonnier 对齐。该项由新增默认关闭参数 `lambda_gmvc_intrinsic` 控制，使用与 GMVC 其他项一致的 start/stop/ramp。

代码改动：

```text
water_splatting/water_splatting.py
  lambda_gmvc_intrinsic: float = 0.0

water_splatting/medium_calibration/gmvc_training.py
  gmvc_intrinsic_loss = Charbonnier(sample(J_gaussian_raw/J_raw/J) - detach(j_consensus))

scripts/experiments/gmvc_phase_b_common.sh
scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh
  透传 LAMBDA_GMVC_INTRINSIC / --pipeline.model.lambda-gmvc-intrinsic

scripts/experiments/gmvc_c4_intrinsic_dc_japanesegradens_10k_to_15k.sh
  GMVC enabled, geometry frozen, features_dc trainable, features_rest frozen
```

Smoke：

```bash
GPU=6 \
STAMP=20260803_gmvc_c4_intrinsic_smoke \
MAX_NUM_ITERATIONS=2 \
RUN_EVAL=0 \
RUN_CLOSURE_DIAG=0 \
GMVC_TRACK_BANK_PATH=/mnt/new/home_old/ycy/water-splatting-refactor/renders/gmvc_track_banks_smoke/japanesegradens_m1_step10000/gmvc_track_bank.pt \
scripts/experiments/gmvc_c4_intrinsic_dc_japanesegradens_10k_to_15k.sh
```

结果：训练启动正常，track bank 加载正常，`lambda_gmvc_intrinsic=0.01` 配置写入正常，`features_dc=True` / `features_rest=False` 路径无 CUDA 或 autograd 错误。

正式命令：

```bash
GPU=6 STAMP=20260803_gmvc_phase_b_jg_c1_freeze_dc RUN_EVAL=1 RUN_CLOSURE_DIAG=0 \
GMVC_VARIANT=c1_diag_freeze_dc GMVC_ENABLED=False GMVC_DIAGNOSTIC_ONLY=True \
GMVC_FREEZE_GEOMETRY=True GMVC_TRAIN_FEATURES_DC=True GMVC_TRAIN_FEATURES_REST=False \
LAMBDA_GMVC_INTRINSIC=0.0 \
scripts/experiments/gmvc_phase_b_common.sh

GPU=6 STAMP=20260803_gmvc_phase_b_jg_c4_intrinsic001 RUN_EVAL=1 RUN_CLOSURE_DIAG=0 \
LAMBDA_GMVC_INTRINSIC=0.01 \
scripts/experiments/gmvc_c4_intrinsic_dc_japanesegradens_10k_to_15k.sh

GPU=6 STAMP=20260803_gmvc_phase_b_jg_c4_intrinsic010 RUN_EVAL=1 RUN_CLOSURE_DIAG=0 \
LAMBDA_GMVC_INTRINSIC=0.10 \
scripts/experiments/gmvc_c4_intrinsic_dc_japanesegradens_10k_to_15k.sh
```

质量与物理诊断结果：

| Variant | PSNR | SSIM | LPIPS | Final tracks | T valid | \(E_J\) | corr \(\Delta\beta^D,\Delta B_\infty\) | corr \(\Delta\beta^B,\Delta B_\infty\) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 M1 continuation | 24.7551 | 0.8995 | 0.1200 | 53,204 | 0.9981 | 0.05434 | -0.2054 | -0.1500 |
| C1 freeze | 24.8648 | 0.8963 | 0.1251 | 53,401 | 0.9832 | 0.06065 | -0.2908 | -0.2493 |
| C1 freeze + DC unfreeze | 24.8637 | 0.8972 | 0.1240 | 53,647 | 0.9870 | 0.05892 | -0.2487 | -0.1496 |
| C4 intrinsic, \(\lambda=0.01\) | 24.8648 | 0.8972 | 0.1240 | 53,554 | 0.9873 | 0.05898 | -0.2572 | -0.1471 |
| C4 intrinsic, \(\lambda=0.10\) | 24.8649 | 0.8972 | 0.1240 | 53,536 | 0.9877 | 0.05895 | -0.2474 | -0.1452 |

C4 output JSONs：

```text
renders/gmvc_c1_diag_freeze_dc_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_c1_freeze_dc/output.json
renders/gmvc_c4_intrinsic_dc_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_c4_intrinsic001/output.json
renders/gmvc_c4_intrinsic_dc_japanesegradens_redsea_seed42_step10000_to_15000_20260803_gmvc_phase_b_jg_c4_intrinsic010/output.json
```

C4 identifiability JSONs：

```text
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c1_freeze_dc/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c4_intrinsic001/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_c4_intrinsic010/gmvc_identifiability.json
```

结论：

- `features_dc` unfreeze 可以相对 freeze-only 改善 SSIM/LPIPS，但仍明显差于 C0 M1 continuation。
- C4 \(\lambda=0.01\) 和 \(\lambda=0.10\) 与 C1 freeze + DC unfreeze control 几乎完全重合，说明 intrinsic GMVC loss 没有形成可观测的额外收益。
- C4 的 \(E_J\) 约为 0.05895，仍比 C0 的 0.05434 差约 8.5%，不满足 GMVC 的物理一致性 gate。
- 因为 JapaneseGradens 主场景已经失败，本轮不进入 IUI3 / Curasao / Panama，也不进入四场景 15k。

修正判断：旧 C4 未能有效测试 intrinsic DC calibration，因为默认 target `J_gaussian_raw` 对应 CUDA wrapper 中未使用的 `out_clr` backward 路径；它只能说明“直接解冻 DC + RGB loss”不够，不能说明 active intrinsic DC proxy 无效。下一步已按 GPT 复盘建议先修复梯度路径，再做短程验证。

### 18.8 Active DC proxy intrinsic path 复核

本轮将 C4 intrinsic source 从旧 `J_gaussian_raw` 改为默认 `J_proxy_raw`。`J_proxy_raw` 来自零 medium、黑背景的 clear proxy render，走主 `out_img` backward 路径，因此可以把颜色梯度传回 Gaussian appearance。为避免完整 SH clear render 混入 view-dependent residual，本轮 GMVC 触发的 proxy 默认使用 DC-only color：

```text
gmvc_intrinsic_source = J_proxy_raw
gmvc_intrinsic_use_dc_proxy = True
clear proxy geometry gradient = 0
clear proxy opacity gradient = 0
clear proxy color gradient = 1
```

新增默认关闭/可配置项：

```text
lambda_gmvc_intrinsic: float = 0.0
gmvc_intrinsic_source: Literal["J_proxy_raw", "J_gaussian_raw", "J_raw", "J"] = "J_proxy_raw"
gmvc_intrinsic_use_dc_proxy: bool = True
gmvc_grad_log_path: Optional[str] = None
gmvc_grad_log_every: int = 100
```

代码改动：

```text
water_splatting/water_splatting.py
  - GMVC intrinsic active 时自动触发 clear proxy render；
  - GMVC proxy 默认使用 DC-only colors；
  - GMVC proxy 对 geometry/opacities/medium 梯度隔离；
  - 可选记录 intrinsic/RGB DC grad ratio JSONL。

water_splatting/medium_calibration/gmvc_training.py
  - intrinsic source 改为配置项，默认 J_proxy_raw；
  - 记录 gmvc_intrinsic_source_available。

scripts/diagnostics/diagnose_gmvc_intrinsic_gradient_paths.py
  - 对同一 checkpoint 和 track bank 比较 J_gaussian_raw vs J_proxy_raw 的 intrinsic loss 梯度。

scripts/experiments/gmvc_p0_nofreeze_japanesegradens_10k_to_10200.sh
scripts/experiments/gmvc_p1_active_proxy003_japanesegradens_10k_to_10200.sh
scripts/experiments/gmvc_p2_active_proxy005_japanesegradens_10k_to_10200.sh
  - P0/P1/P2 no-freeze short-run 筛选脚本。
```

#### 18.8.1 梯度路径诊断

命令：

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_gmvc_intrinsic_gradient_paths.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --load-step 10000 \
  --probe-step 10001 \
  --gmvc-track-bank renders/gmvc_track_banks/japanesegradens_redsea_m1_step10000_train_s4096/gmvc_track_bank.pt \
  --lambda-gmvc-intrinsic 1.0 \
  --max-tracks 2048 \
  --max-images 17 \
  --use-dc-proxy \
  --output-json renders/gmvc_phase_b_gradients_20260803/japanesegradens_intrinsic_paths.json
```

结果：

| Source | intrinsic raw | DC grad norm | RGB DC grad norm | ratio | geometry | opacity | medium |
|---|---:|---:|---:|---:|---:|---:|---:|
| `J_gaussian_raw` | 0.04224 | 0.000000 | 0.000623 | 0.0000 | 0.0 | 0.0 | 0.0 |
| `J_proxy_raw` DC-only | 0.04724 | 0.001910 | 0.000623 | 3.0671 | 0.0 | 0.0 | 0.0 |

结论：

- 旧 C4 的 `J_gaussian_raw` intrinsic loss 确认是 dead-gradient，不能更新 `features_dc`。
- 新 `J_proxy_raw` DC-only path 确认可以更新 `features_dc`，同时 geometry / opacity / medium 梯度为 0。
- 因为 λ=1.0 的 ratio 约 3.07，短程训练选用 λ=0.03 和 λ=0.05，使实际 ratio 落在约 5%–10%。

#### 18.8.2 Active proxy short-run 结果

所有实验均从 JapaneseGradens M1 step 10000 checkpoint 继续，不冻结 geometry，不改变原始 densification / culling 路径。

200-step 命令：

```bash
GPU=6 STAMP=20260803_gmvc_p0_nofreeze_short RUN_EVAL=1 RUN_CLOSURE_DIAG=0 \
  scripts/experiments/gmvc_p0_nofreeze_japanesegradens_10k_to_10200.sh

GPU=7 STAMP=20260803_gmvc_p1_active_proxy003_short RUN_EVAL=1 RUN_CLOSURE_DIAG=0 \
  scripts/experiments/gmvc_p1_active_proxy003_japanesegradens_10k_to_10200.sh

GPU=8 STAMP=20260803_gmvc_p2_active_proxy005_short RUN_EVAL=1 RUN_CLOSURE_DIAG=0 \
  scripts/experiments/gmvc_p2_active_proxy005_japanesegradens_10k_to_10200.sh
```

500-step 命令：

```bash
GPU=6 EXPERIMENT_NAME=gmvc_p0_nofreeze_japanesegradens_redsea_seed42_step10000_to_10500 \
  STAMP=20260803_gmvc_p0_nofreeze_500 TARGET_FINAL_STEP=10500 MODEL_NUM_STEPS=10500 \
  MAX_NUM_ITERATIONS=500 STEPS_PER_SAVE=500 RUN_EVAL=1 RUN_CLOSURE_DIAG=0 \
  scripts/experiments/gmvc_p0_nofreeze_japanesegradens_10k_to_10200.sh

GPU=7 EXPERIMENT_NAME=gmvc_p1_active_proxy003_japanesegradens_redsea_seed42_step10000_to_10500 \
  STAMP=20260803_gmvc_p1_active_proxy003_500 TARGET_FINAL_STEP=10500 MODEL_NUM_STEPS=10500 \
  MAX_NUM_ITERATIONS=500 STEPS_PER_SAVE=500 GMVC_GRAD_LOG_EVERY=100 RUN_EVAL=1 RUN_CLOSURE_DIAG=0 \
  scripts/experiments/gmvc_p1_active_proxy003_japanesegradens_10k_to_10200.sh

GPU=8 EXPERIMENT_NAME=gmvc_p2_active_proxy005_japanesegradens_redsea_seed42_step10000_to_10500 \
  STAMP=20260803_gmvc_p2_active_proxy005_500 TARGET_FINAL_STEP=10500 MODEL_NUM_STEPS=10500 \
  MAX_NUM_ITERATIONS=500 STEPS_PER_SAVE=500 GMVC_GRAD_LOG_EVERY=100 RUN_EVAL=1 RUN_CLOSURE_DIAG=0 \
  scripts/experiments/gmvc_p2_active_proxy005_japanesegradens_10k_to_10200.sh
```

质量结果：

| Run | PSNR | SSIM | LPIPS | J blue | J sat |
|---|---:|---:|---:|---:|---:|
| 200 P0 no-freeze | 24.9172 | 0.8981 | 0.1220 | 0.1451 | 0.0573 |
| 200 P1 proxy λ=0.03 | 24.9212 | 0.8980 | 0.1220 | 0.1443 | 0.0574 |
| 200 P2 proxy λ=0.05 | 24.9277 | 0.8980 | 0.1220 | 0.1441 | 0.0575 |
| 500 P0 no-freeze | 24.9521 | 0.9005 | 0.1197 | 0.1449 | 0.0563 |
| 500 P1 proxy λ=0.03 | 24.9612 | 0.9006 | 0.1199 | 0.1445 | 0.0565 |
| 500 P2 proxy λ=0.05 | 24.9633 | 0.9006 | 0.1201 | 0.1441 | 0.0566 |

梯度日志：

| Run | entries | raw first→last | DC grad ratio mean | ratio min/max | max non-DC grad |
|---|---:|---:|---:|---:|---:|
| 200 P1 λ=0.03 | 4 | 0.03958→0.04466 | 0.0478 | 0.0389/0.0603 | 0.0e+00 |
| 200 P2 λ=0.05 | 4 | 0.03913→0.04348 | 0.0793 | 0.0646/0.0999 | 0.0e+00 |
| 500 P1 λ=0.03 | 5 | 0.03884→0.05157 | 0.0503 | 0.0403/0.0602 | 0.0e+00 |
| 500 P2 λ=0.05 | 5 | 0.03806→0.04972 | 0.0833 | 0.0666/0.0999 | 0.0e+00 |

#### 18.8.3 500-step identifiability

| Variant | Final tracks | T valid ratio | \(E_J\) mean | corr \(\Delta\beta^D,\Delta B_\infty\) | corr \(\Delta\beta^B,\Delta B_\infty\) |
|---|---:|---:|---:|---:|---:|
| P0 no-freeze 10k→10.5k | 53,441 | 0.9875 | 0.05943 | -0.3127 | -0.2147 |
| P1 active proxy λ=0.03 | 53,413 | 0.9869 | 0.05981 | -0.3193 | -0.2074 |
| P2 active proxy λ=0.05 | 53,308 | 0.9862 | 0.05981 | -0.3165 | -0.2009 |

Diagnostic JSONs：

```text
renders/gmvc_phase_b_gradients_20260803/japanesegradens_intrinsic_paths.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_p0_nofreeze_10500/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_p1_active_proxy003_10500/gmvc_identifiability.json
renders/gmvc_phase_b_identifiability_20260803/japanesegradens_p2_active_proxy005_10500/gmvc_identifiability.json
```

#### 18.8.4 Active proxy 结论

Active `J_proxy_raw` 修复了旧 C4 的梯度路径问题，并且 λ=0.03/0.05 产生了可控的 DC-only auxiliary gradient。但是，短程结果显示：

- RGB 质量只有很小的 PSNR 剂量响应，SSIM 基本持平；
- P2 的 LPIPS 相对 P0 略差；
- \(E_J\) 没有改善，P1/P2 反而比 P0 更高；
- compensation correlation 没有朝更稳定方向移动；
- intrinsic raw loss 在训练日志中没有下降，说明 detached M1 consensus 不是有效的新观测目标。

因此，结论不是“GMVC 核心思路失败”，而是：

> offline detached M1 consensus formulation 即使使用 active DC proxy，也主要是旧 M1 分解的自蒸馏，不能改善 medium/Gaussian 不可辨识性。

本轮不应继续把 active-proxy offline intrinsic target 拉到 15k 或扩展四场景。下一步若继续 GMVC，应转向 GPT 建议的 online multi-view degradation closure / scene-level bounded medium variation，或者先做低维物理 oracle 验证场景级 medium 参数假设是否成立。
