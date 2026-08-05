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

---

## 19. Phase O：低维物理 Oracle

### 19.1 目的

根据最新分析，本阶段不直接编写 GMVC-V2 训练模块，而是先验证一个更低风险的问题：

> 在固定 M1 geometry/depth 和多视图对应关系后，全局水体参数或“全局中心 + 每相机有界残差”是否能解释 track 上的 GT RGB，并改善 held-out cross-view transfer。

该 oracle 不使用旧 M1 detached intrinsic consensus 作为监督目标。它只使用：

- track 的 GT RGB；
- M1 渲染深度与几何投影对应；
- alpha、depth error、depth std、transmission 等 track 可靠性权重；
- 每条 track 的 latent intrinsic color \(J_p\)；
- 低维水体参数。

### 19.2 新增代码

新增：

```text
scripts/diagnostics/fit_gmvc_lowdim_oracle.py
scripts/experiments/gmvc_phase_o_lowdim_oracle.sh
```

`fit_gmvc_lowdim_oracle.py` 支持：

- `--load-config`
- `--load-step`
- `--test-mode`
- `--split`
- `--max-images`
- `--samples-per-view`
- `--target-neighbor-window`
- `--max-tracks`
- `--models O0,O1`
- `--output-dir`

实现的 oracle：

- **O0:** 全场景统一 \( \beta^D, \beta^B, B_\infty \)，共 9 个水体参数，加每条 track 的 \(J_p\)。
- **O1:** 场景中心 + 每相机 bounded residual，其中 `log_beta` residual scale 为 `0.15`，`B_inf` logit residual scale 为 `0.10`。

暂未实现 O2。原因是 O2 需要把 residual 从 per-camera 扩展到 `dir_xy_camera` 函数形式，并引入 ray/xy 条件网络；当前 Phase O 的目标是先验证低维 hypothesis，而不是提前写训练框架。

### 19.3 指标定义

主要指标：

- **held-out transfer L1:** 从视角 \(i\) 反演出的 \(J_i\) 通过视角 \(j\) 的 fitted water/depth 预测 \(I_j\)。
- **held-out normalized closure L1:** 使用无除法闭合式 \((I_i-B_i)T_j - (I_j-B_j)T_i\) 的归一化残差。
- **held-out object \(J\) variance:** 同一 track 在不同视角反演出的 \(J_i\) 方差。
- **consensus-J recon L1:** 固定 fitted water 后，用 held-out track 自身的 online \(J\)-center 重建各 view。
- **residual saturation:** O1 中 `abs(tanh(delta)) > 0.95` 的比例，用于判断有界残差是否碰到上限。

### 19.4 实验命令

复现实验脚本：

```bash
SCENE=all scripts/experiments/gmvc_phase_o_lowdim_oracle.sh
```

本轮实际运行配置：

```text
SAMPLES_PER_VIEW=4096
MAX_TRACKS=30000
ITERS=500
LR=0.03
MODELS=O0,O1
TRAIN_FRACTION=0.80
```

场景与 checkpoint：

| Scene | Config | Step |
|---|---|---:|
| JapaneseGradens | `outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/.../config.yml` | 10000 |
| IUI3 | `outputs/m1_dir_xy_camera_iui3_redsea_15000/.../config.yml` | 14999 |
| Curasao | `outputs/cross_scene_curasao_m1_seed42_15000/.../config.yml` | 10000 |
| Panama | `outputs/cross_scene_panama_m1_seed42_15000/.../config.yml` | 10000 |

IUI3 的该 M1 run 不存在 step-10000 checkpoint，实际使用已有 `step-000014999.ckpt`。

### 19.5 输出文件

```text
renders/gmvc_oracle_phase_o_20260803/japanesegradens_m1_step10000_train_s4096/gmvc_lowdim_oracle.json
renders/gmvc_oracle_phase_o_20260803/iui3_m1_step14999_train_s4096/gmvc_lowdim_oracle.json
renders/gmvc_oracle_phase_o_20260803/curasao_m1_step10000_train_s4096/gmvc_lowdim_oracle.json
renders/gmvc_oracle_phase_o_20260803/panama_m1_step10000_train_s4096/gmvc_lowdim_oracle.json
```

### 19.6 四场景结果

| Scene | Tracks | O0 transfer | O1 transfer | Δ | O0 J-var | O1 J-var | Δ | O0 closure-norm | O1 closure-norm | Δ | O1 saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JapaneseGradens | 30,000 | 0.04593 | 0.04336 | -5.6% | 0.00830 | 0.00718 | -13.4% | 0.26634 | 0.30414 | +14.2% | 28.1% |
| IUI3 | 30,000 | 0.03435 | 0.03182 | -7.4% | 0.00650 | 0.00469 | -27.8% | 0.10845 | 0.15379 | +41.8% | 28.0% |
| Curasao | 30,000 | 0.03184 | 0.02134 | -33.0% | 0.00411 | 0.00209 | -49.1% | 0.27770 | 0.23782 | -14.4% | 40.7% |
| Panama | 30,000 | 0.02731 | 0.02535 | -7.2% | 0.00350 | 0.00273 | -22.0% | 0.28238 | 0.22880 | -19.0% | 8.9% |

Consensus-J reconstruction:

| Scene | O0 held-out consensus recon | O1 held-out consensus recon | Δ |
|---|---:|---:|---:|
| JapaneseGradens | 0.03206 | 0.02986 | -6.9% |
| IUI3 | 0.02576 | 0.02369 | -8.0% |
| Curasao | 0.02261 | 0.01504 | -33.5% |
| Panama | 0.01733 | 0.01624 | -6.3% |

### 19.7 Phase O 结论

1. **GMVC 不应在当前结果处停止。** O1 在四个场景中都降低了 held-out transfer error 和 object \(J\) variance，说明“场景公共水体中心 + 有限视角变化”确实有可用信号。

2. **O1 不是无条件胜利。** JapaneseGradens 和 IUI3 的 normalized closure 变差，且 O1 residual saturation 约 28%；Curasao transfer 改善最大，但 saturation 达 40.7%。这说明 per-camera residual 在 oracle 中很容易碰到边界，未来训练版必须加入 residual budget 监控和更稳的 closure 权重。

3. **Panama 是最干净的正例。** O1 同时改善 transfer、closure、J variance，且 saturation 只有 8.9%，表明 bounded residual 假设在部分场景中可以稳定工作。

4. **JapaneseGradens 的结论更谨慎。** O1 transfer 下降 5.6%，J variance 下降 13.4%，consensus-J recon 下降 6.9%，但 closure-norm 上升 14.2%。这支持继续做 GMVC-V2，但不支持直接把 O1 形式无正则地塞进训练。

5. **下一步应进入 V1，而不是回到 offline consensus。** 建议先实现 medium-only online cross-view closure，附带 scene-centered bounded medium parameterization 和 residual saturation 日志；只有 V1 能降低 held-out closure/transfer 后，再接回 active DC proxy 做 V2 alternating object-medium calibration。

---

## 20. Phase O-Pareto：centered residual、regularization、closure sweep

### 20.1 动机

Phase O 证明 O1 中的 per-camera bounded residual 有可优化信号，但原始 O1 只优化 RGB reconstruction，没有直接约束 transfer、closure 或 residual budget。因此本轮补充 Oracle-Pareto，目标不是追求最低 reconstruction，而是找到：

```text
transfer 降低
object-J variance 降低
robust closure 不明显变差，最好降低
residual saturation <= 10%–15%
```

同时修复 Phase O 的两个诊断边界：

1. O1 residual 默认按相机观测权重去均值，降低 center-residual gauge 自由度；
2. 新增 signal-floor normalized closure，分母使用 `max(|L|+|R|, tau_signal)`，默认 `tau_signal=0.03`，避免暗通道/低信号 pair 主导旧 normalized closure。

### 20.2 代码改动

更新：

```text
scripts/diagnostics/fit_gmvc_lowdim_oracle.py
scripts/experiments/gmvc_phase_o_pareto_oracle.sh
```

新增能力：

- `--o1-variants`：一次运行多个 O1 variant，格式为 `name:beta_scale:binf_scale:lambda_res:lambda_sat:lambda_closure`；
- centered residual：默认启用，使用相机权重对 residual 去均值；
- residual L2 regularization；
- saturation softplus penalty；
- farthest-depth pair robust closure loss；
- held-out raw closure、旧 normalized closure、signal-floor closure；
- 按 transmission 和 signal strength 分桶的诊断；
- saturation 按参数和 RGB channel 拆分记录。

### 20.3 实验设置

复现命令：

```bash
SCENE=japanesegradens scripts/experiments/gmvc_phase_o_pareto_oracle.sh
SCENE=panama scripts/experiments/gmvc_phase_o_pareto_oracle.sh
SCENE=curasao scripts/experiments/gmvc_phase_o_pareto_oracle.sh
SCENE=iui3 scripts/experiments/gmvc_phase_o_pareto_oracle.sh
```

统一配置：

```text
SAMPLES_PER_VIEW=4096
MAX_TRACKS=30000
ITERS=500
LR=0.03
CLOSURE_SIGNAL_FLOOR=0.03
```

Variant 矩阵：

| Variant | beta scale | B-inf scale | lambda_res | lambda_sat | lambda_closure |
|---|---:|---:|---:|---:|---:|
| O1_S1 | 0.05 | 0.05 | 0 | 0 | 0 |
| O1_S2 | 0.10 | 0.075 | 0 | 0 | 0 |
| O1_S3 | 0.15 | 0.10 | 0 | 0 | 0 |
| O1_S4 | 0.20 | 0.15 | 0 | 0 | 0 |
| O1_R1 | 0.15 | 0.10 | 0.0001 | 0.0001 | 0 |
| O1_R2 | 0.15 | 0.10 | 0.0005 | 0.0001 | 0 |
| O1_R3 | 0.15 | 0.10 | 0.0010 | 0.0005 | 0 |
| O1_C1 | 0.15 | 0.10 | 0.0005 | 0.0001 | 0.01 |
| O1_C2 | 0.15 | 0.10 | 0.0005 | 0.0001 | 0.05 |
| O1_C3 | 0.15 | 0.10 | 0.0005 | 0.0001 | 0.10 |

Output JSONs：

```text
renders/gmvc_oracle_pareto_20260803/japanesegradens_m1_step10000_train_s4096/gmvc_lowdim_oracle.json
renders/gmvc_oracle_pareto_20260803/iui3_m1_step14999_train_s4096/gmvc_lowdim_oracle.json
renders/gmvc_oracle_pareto_20260803/curasao_m1_step10000_train_s4096/gmvc_lowdim_oracle.json
renders/gmvc_oracle_pareto_20260803/panama_m1_step10000_train_s4096/gmvc_lowdim_oracle.json
```

### 20.4 关键结果

表中 Δ 均相对 O0 held-out 指标。

| Scene | Candidate | transfer Δ | J-var Δ | robust closure Δ | raw closure Δ | saturation | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| JapaneseGradens | O1_R3 | -5.0% | -4.9% | +8.7% | -7.8% | 0.0% | transfer 勉强过线，但 J-var/robust closure 不足 |
| JapaneseGradens | O1_C1 | -3.8% | -27.0% | -39.2% | +7.0% | 0.0% | closure/J-var 强，但 transfer 不足 |
| IUI3 | O1_R3 | -6.6% | -27.3% | -2.8% | +0.1% | 0.0% | 通过 Pareto gate |
| IUI3 | O1_C1 | -6.6% | -32.5% | -11.5% | +3.0% | 0.0% | 更强 closure/J-var，通过 |
| Curasao | O1_S4 | -35.0% | -52.5% | -21.7% | -40.4% | 13.0% | range sweep 最佳，saturation 可接受 |
| Curasao | O1_R1 | -29.8% | -45.2% | -16.0% | -34.8% | 6.8% | 更保守，强候选 |
| Curasao | O1_C1 | -25.8% | -37.0% | -24.3% | -33.0% | 4.3% | closure 更强，强候选 |
| Panama | O1_R1 | -7.2% | -17.9% | -10.3% | -1.3% | 0.0% | 通过 Pareto gate |
| Panama | O1_C1 | -5.1% | -23.0% | -34.7% | +4.6% | 0.0% | closure 强，通过 |

### 20.5 场景结论

1. **Curasao 解释最清楚。** Phase O 中 40.7% saturation 主要是 residual range/regularization 选择问题，不是 GMVC 信号失败。O1_S4 将 saturation 降到 13.0%，同时 transfer、J-var、robust closure、raw closure 全部改善；O1_R1/O1_C1 在 saturation 更低时仍保留大部分收益。

2. **Panama 是稳定正例。** O1_R1 与 O1_C1 都满足 transfer、J-var、closure、saturation gate。后续 V1 可以优先用 Panama 做 medium-only online closure 的正向 sanity check。

3. **IUI3 可以被正则化修复。** 无正则 range sweep 中 robust closure 会随自由度扩大而变差，但 R3/C1 在 0% saturation 下同时改善 transfer、J-var 和 robust closure。

4. **JapaneseGradens 仍是压力测试。** 没有单个 variant 同时满足所有 gate。R3 保留 transfer 改善并消除 saturation，但 J-var 和 robust closure 不够；C1 显著改善 closure/J-var，但 transfer 只有 -3.8%。这说明 JapaneseGradens 的在线训练不能只优化单一 closure 或单一 transfer，需要 V1 中分阶段/分权重测试。

5. **closure loss 存在明确 trade-off。** C2/C3 通常继续降低 robust closure 和 J-var，但会牺牲 transfer 或 raw closure。第一版 V1 不应使用过大的 `lambda_C`，更合理的起点是接近 C1：`lambda_C≈0.01`，配合 residual/saturation regularization。

### 20.6 下一步 V1 建议

进入 medium-only online closure，但先只做 500-step continuation，不接 active DC proxy。

建议矩阵：

| Run | Setting |
|---|---|
| V0 | M1 continuation |
| V1 | closure only, original M1 medium parameterization |
| V2 | centered bounded residual, no closure |
| V3 | centered bounded residual + residual/saturation regularization |
| V4 | V3 + weak closure, `lambda_C≈0.01` |

首轮场景：

- Panama：正向 sanity check；
- JapaneseGradens：压力测试。

进入下一阶段的条件：

```text
PSNR drop <= 0.15 dB
LPIPS increase <= 0.003
held-out transfer降低 >= 5%
object-J variance降低 >= 10%
robust closure不变差超过 5%，最好降低
residual saturation <= 15%
```

---

## 21. Closure denominator audit and online V1

### 21.1 代码更新

本轮在 `research/gmvc-medium-calibration` 上实现了最小在线 V1，不替换 M1 medium forward 参数化，只在训练期加入 medium-only 约束：

- `scripts/diagnostics/build_gmvc_tracks.py`
  - 为每个 GMVC observation 增加 nearest/farthest closure partner；
  - 保存 `closure_partner_gt/depth/medium_attn/medium_bs/b_inf`；
  - 保存 M1 baseline 固定分母 `closure_denom_fixed` 和 `closure_weight`。
- `scripts/diagnostics/fit_gmvc_lowdim_oracle.py`
  - 增加 `closure_denominator={current,detach,fixed}`；
  - 修复 `--o1-variants` 解包；
  - residual regularization 改为约束 `tanh(delta)`；
  - 增加 actual weighted mean residual 统计。
- `water_splatting/medium_calibration/gmvc_training.py`
  - 增加在线 scene-center residual budget；
  - 增加 fixed-denominator cross-view closure；
  - 记录 residual、closure、transmission、backscatter 诊断。
- `water_splatting/water_splatting.py`
  - 新增 V1 config flags；
  - 将 `_gmvc_online_state` 传入训练项；
  - 扩展 GMVC gradient JSONL，记录 residual/closure 相对 RGB 的 medium-gradient ratio。
- `scripts/diagnostics/diagnose_gmvc_checkpoint_tracks.py`
  - 新增 checkpoint-level GMVC cross-view diagnostic；
  - 直接评估 checkpoint 当前 medium 输出的 held-out transfer、raw closure、fixed-normalized closure 和 object-J variance，不拟合 oracle。
- `scripts/experiments/gmvc_v1_online_500.sh`
  - 新增 Panama/JapaneseGradens 的 500-step V1 continuation runner；
  - 默认从 M1 step-10000 checkpoint 继续，旧 GMVC/TBAP/TACMD/TMICA/J/cleanup/capacity 分支全部关闭。

### 21.2 新增 config flags

```text
lambda_gmvc_residual_budget: float = 0.0
lambda_gmvc_fixed_closure: float = 0.0
gmvc_residual_beta_log_scale: float = 0.15
gmvc_residual_binf_logit_scale: float = 0.10
gmvc_residual_ema_momentum: float = 0.99
gmvc_closure_signal_floor: float = 0.03
```

这些 flag 默认关闭。V1 loss 仅在 `gmvc_enabled=True` 且 `gmvc_diagnostic_only=False` 时进入 loss dict。

### 21.3 Closure denominator audit

运行：

```bash
# 代表命令；四个场景分别运行
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/fit_gmvc_lowdim_oracle.py \
  --load-config <M1_CONFIG> \
  --load-step <M1_STEP> \
  --test-mode inference \
  --split train \
  --samples-per-view 4096 \
  --max-tracks 30000 \
  --models O0,O1 \
  --o1-variants "C1_current:0.15:0.10:0.0005:0.0001:0.01:current;C1_detach:0.15:0.10:0.0005:0.0001:0.01:detach;C1_fixed:0.15:0.10:0.0005:0.0001:0.01:fixed" \
  --output-dir renders/gmvc_oracle_denominator_audit_20260804/<scene>
```

输出：

```text
renders/gmvc_oracle_denominator_audit_20260804/curasao_m1_step10000_train_s4096/gmvc_lowdim_oracle.json
renders/gmvc_oracle_denominator_audit_20260804/iui3_m1_step14999_train_s4096/gmvc_lowdim_oracle.json
renders/gmvc_oracle_denominator_audit_20260804/japanesegradens_m1_step10000_train_s4096/gmvc_lowdim_oracle.json
renders/gmvc_oracle_denominator_audit_20260804/panama_m1_step10000_train_s4096/gmvc_lowdim_oracle.json
```

相对 O0 held-out：

| Scene | Variant | transfer Δ | J-var Δ | raw closure Δ | fixed closure Δ | old norm closure Δ | saturation |
|---|---|---:|---:|---:|---:|---:|---:|
| Curasao | C1_current | -29.9% | -43.9% | -36.3% | -27.7% | -17.1% | 9.3% |
| Curasao | C1_fixed | -31.0% | -38.3% | -41.3% | -22.1% | -11.9% | 9.3% |
| IUI3 | C1_current | -6.6% | -33.5% | +3.7% | -8.9% | -11.8% | 1.3% |
| IUI3 | C1_fixed | -6.7% | -7.6% | -10.2% | +18.7% | +38.7% | 5.8% |
| JapaneseGradens | C1_current | -4.0% | -28.7% | +7.7% | -39.7% | -55.9% | 5.2% |
| JapaneseGradens | C1_fixed | -5.6% | +13.0% | -16.0% | +14.1% | +17.9% | 1.3% |
| Panama | C1_current | -4.9% | -28.2% | +9.2% | -40.5% | -55.8% | 3.7% |
| Panama | C1_fixed | -7.3% | +7.5% | -15.0% | -13.7% | -6.6% | 0.7% |

结论：

- `current` denominator 的 normalized closure 改善包含明显分母投机，JapaneseGradens/Panama 的 raw closure 反而变差。
- `fixed` denominator 能保留 transfer/raw-closure 信号，特别是 Curasao、Panama，但会让 JapaneseGradens/Panama 的 held-out object-J variance 变差。
- 正式 V1 只能使用 fixed denominator；不能再用 current differentiable denominator 作为成功证据。

### 21.4 GMVC V1 track banks

训练 bank 使用 M1 step-10000 的 train views 构建：

```text
renders/gmvc_v1_track_banks/japanesegradens_redsea_m1_step10000_train_s4096/gmvc_track_bank.pt
renders/gmvc_v1_track_banks/panama_m1_step10000_train_s4096/gmvc_track_bank.pt
```

统计：

| Scene | accepted tracks | accepted observations | cap |
|---|---:|---:|---:|
| JapaneseGradens | 53,397 | 580,275 | 20,000 per camera |
| Panama | 47,918 | 400,203 | 20,000 per camera |

### 21.5 Online V1 500-step matrix

所有 run 从 M1 step-10000 checkpoint 继续到 step-10500：

| Run | Setting |
|---|---|
| V0 | M1 continuation |
| V1 | residual budget only, `lambda_gmvc_residual_budget=0.001` |
| V2 | fixed closure only, JG `0.0005`, Panama `0.0010` |
| V3 | budget `0.001` + low closure |
| V4 | budget `0.001` + mid closure, JG `0.0015`, Panama `0.0025` |
| T1 | tuned budget `0.003` + closure, JG `0.020`, Panama `0.050` |
| T2 | tuned budget `0.003` + closure, JG `0.050`, Panama `0.100` |

代表命令：

```bash
SCENE=japanesegradens VARIANT=V3 GPU=6 \
  TARGET_FINAL_STEP=10500 \
  STAMP=20260804_gmvc_v1_online_500 \
  bash scripts/experiments/gmvc_v1_online_500.sh

SCENE=panama VARIANT=V3 GPU=9 \
  TARGET_FINAL_STEP=10500 \
  STAMP=20260804_gmvc_v1_online_500_tuned \
  LAMBDA_GMVC_RESIDUAL_BUDGET=0.0030 \
  LAMBDA_GMVC_FIXED_CLOSURE=0.1000 \
  GMVC_GRAD_LOG_PATH=logs/gmvc_v1_t2_budget003_closure010_panama_seed42_step10000_to_10500/gmvc_grad.jsonl \
  GMVC_GRAD_LOG_EVERY=100 \
  bash scripts/experiments/gmvc_v1_online_500.sh
```

Eval JSONs：

```text
renders/gmvc_v1_*_20260804_gmvc_v1_online_500/output.json
renders/gmvc_v1_*_20260804_gmvc_v1_online_500_tuned/output.json
```

Checkpoint diagnostics：

```text
renders/gmvc_v1_checkpoint_diag_20260804/*/gmvc_checkpoint_tracks.json
renders/gmvc_v1_checkpoint_diag_20260804_tuned/*/gmvc_checkpoint_tracks.json
```

### 21.6 Image metrics

相对同场景 V0：

| Scene | Run | PSNR Δ | SSIM Δ | LPIPS Δ | Image gate |
|---|---|---:|---:|---:|---|
| JapaneseGradens | V1 | +0.0052 | +0.00001 | -0.00013 | pass |
| JapaneseGradens | V2 | -0.0039 | -0.00013 | +0.00012 | pass |
| JapaneseGradens | V3 | -0.0014 | -0.00008 | +0.00037 | pass |
| JapaneseGradens | V4 | +0.0014 | +0.00003 | -0.00004 | pass |
| JapaneseGradens | T1 | +0.0105 | -0.00005 | +0.00023 | pass |
| JapaneseGradens | T2 | +0.0122 | -0.00003 | +0.00043 | pass |
| Panama | V1 | -0.0191 | +0.00001 | -0.00005 | pass |
| Panama | V2 | -0.0171 | -0.00003 | +0.00005 | pass |
| Panama | V3 | +0.0059 | +0.00003 | +0.00002 | pass |
| Panama | V4 | -0.0127 | +0.00002 | -0.00013 | pass |
| Panama | T1 | -0.0171 | +0.00009 | -0.00025 | pass |
| Panama | T2 | +0.0448 | +0.00022 | -0.00008 | pass |

所有 V1/Tuned run 都满足 image safety gate：

```text
PSNR drop <= 0.15 dB
SSIM drop <= 0.0015
LPIPS increase <= 0.0030
```

### 21.7 Checkpoint-level decoupling metrics

相对同场景 V0 held-out cross-view metrics：

| Scene | Run | transfer Δ | object-J var Δ | raw closure Δ | fixed closure Δ | old norm closure Δ | Decoupling gate |
|---|---|---:|---:|---:|---:|---:|---|
| JapaneseGradens | V1 | +1.10% | +3.12% | +0.22% | +1.41% | +1.31% | fail |
| JapaneseGradens | V2 | +0.07% | +1.91% | -0.05% | +2.02% | +2.64% | fail |
| JapaneseGradens | V3 | -1.90% | -0.20% | -2.53% | +1.91% | +4.14% | fail |
| JapaneseGradens | V4 | -1.15% | +4.02% | -1.89% | +2.42% | +4.45% | fail |
| JapaneseGradens | T1 | +1.69% | +3.71% | -0.08% | +0.88% | +4.30% | fail |
| JapaneseGradens | T2 | -2.88% | +2.57% | -5.74% | +2.22% | +4.57% | fail |
| Panama | V1 | -4.26% | -3.69% | -4.94% | -1.15% | -0.81% | fail |
| Panama | V2 | -1.32% | +0.43% | -1.44% | -2.63% | -2.68% | fail |
| Panama | V3 | +1.79% | +6.44% | +1.09% | -0.02% | +0.39% | fail |
| Panama | V4 | -3.71% | -1.70% | -4.47% | -2.67% | -2.92% | fail |
| Panama | T1 | -0.84% | +5.26% | -2.91% | +0.76% | +0.89% | fail |
| Panama | T2 | +0.36% | +8.13% | -2.68% | +0.79% | +2.16% | fail |

Decoupling gate 使用：

```text
held-out transfer降低 >= 5%
object-J variance降低 >= 10%
raw closure不恶化超过 5%
fixed-normalized closure降低，或不恶化超过 5%
residual saturation <= 15%
```

没有任何在线 V1 run 同时满足 decoupling gate。

### 21.8 Gradient calibration

旧低权重 V3 的 20-step smoke：

```text
JapaneseGradens V3, lambda_R=0.001, lambda_C=0.0005
step 10020: residual/RGB-medium grad ratio = 0.0030
step 10020: closure/RGB-medium grad ratio  = 0.000006
```

closure 梯度几乎为零，解释了 V0-V4 的 online decoupling 变化很弱。

tuned probe：

| Scene | Setting | step | residual/RGB medium grad | closure/RGB medium grad |
|---|---|---:|---:|---:|
| JapaneseGradens | R=0.003, C=0.050 | 10020 | 0.0138 | 0.0111 |
| Panama | R=0.003, C=0.100 | 10020 | 0.0041 | 0.0062 |
| JapaneseGradens T1 | R=0.003, C=0.020 | 10400 | 0.0104 | 0.0306 |
| JapaneseGradens T2 | R=0.003, C=0.050 | 10400 | 0.0108 | 0.0753 |
| Panama T1 | R=0.003, C=0.050 | 10400 | 0.0361 | 0.0214 |
| Panama T2 | R=0.003, C=0.100 | 10400 | 0.0492 | 0.0647 |

tuned 权重已经进入或超过建议梯度比例区间，但仍未改善 held-out object-J variance。这说明失败不是单纯由于 `lambda_C` 太小。

### 21.9 结论

1. **Oracle-Pareto 的跨视图信号不是假的，但当前 online V1 没有复现 oracle 解耦收益。** fixed-denominator audit 证明 Curasao/Panama 仍有真实 transfer/raw-closure 信号，但在线训练中这些信号没有稳定转化为 held-out object-J variance 改善。

2. **当前 V1 可以安全训练，但不是成功模块。** 所有 V1/Tuned run 都通过 image safety gate；T2 甚至使 Panama PSNR 增加 `+0.0448 dB`，JapaneseGradens 增加 `+0.0122 dB`。但这些提升不伴随 decoupling gate 改善，不能作为 GMVC 成功证据。

3. **fixed closure 的在线梯度方向可能过于局部。** 当前训练只对当前 camera 的 medium 输出反向传播，partner observation 使用 track bank 中的 M1 detached partner medium。这能避免重渲染 partner view，但也可能把 closure 变成单视角对 M1 partner 的局部校正，难以推动真正的跨视角 scene-center 分解。

4. **object-J variance 是主要失败项。** tuned run 提高了 closure 梯度比例，但 JapaneseGradens/Panama 的 object-J variance 都没有达到 `-10%`，多数还变差。因此不应进入 15k，也不应接 active DC proxy。

### 21.10 下一步建议

停止当前 online V1 作为正式方法进入 15k。保留代码和诊断作为可复现实验基础。

如果继续 GMVC，应优先测试下面两个方向，而不是继续盲目加大 `lambda_C`：

1. **Symmetric pair minibatch closure.** 每步同时渲染 source 与 partner camera，closure 两端都使用当前模型 medium 输出，仍固定 denominator，并只让 medium 分支吃辅助梯度。

2. **Scene-center residual parameterization.** 不再只对当前 M1 output 做 budget loss，而是显式把 medium field 写成 scene center + bounded camera residual，使 optimizer 不能靠原 M1 MLP 自由度绕开 residual center。

当前 gate 结论：

```text
进入 15k: No
最佳 image candidate: Panama T2, JapaneseGradens T2
最佳 decoupling candidate: None
是否确认 GMVC online V1 成功: No
是否确认 oracle 信号完全失败: No
```

## 22. GMVC-V2: symmetric profiled-radiance calibration

### 22.1 Motivation

本轮根据新的分析回到 M1 step-10000 continuation，目标是修复 V1 与 Oracle 不一致的问题。V1 使用当前 view 的 medium 去适配旧 bank partner；V2 改为对同一 track 的所有 observation 都查询当前 medium network，并用解析消元的共享固有辐射 \(J_p^\star\) 恢复 Oracle 中最关键的 latent object color。

V2 不修改 underwater renderer，不监督去水体颜色，不重新启用旧 J/TBAP/TMICA/TACMD/cleanup/capacity 分支。

### 22.2 Code changes

- `water_splatting/fields/medium_field.py`
  - 新增 `DirectionConditionedMediumField.query_points(...)`；
  - 新增 `_append_point_context(...)`，复用 `dir_xy_camera` 的 direction、image xy、camera center context；
  - point query 不 rasterize partner image，只对 bank 中的 ray context 做 medium MLP 查询。
- `scripts/diagnostics/build_gmvc_tracks.py`
  - track bank 新增 V2 flat `observations` 表；
  - 保存 `track_id/camera_index/image_index/xy/image_xy_norm/ray_direction/camera_center/gt/fixed_depth/weight`；
  - 保留 legacy `per_camera` bank，保证 V1 诊断兼容；
  - 同时保存 M1 bank medium 参数，仅用于固定分母/诊断，不作为 V2 当前 medium 的梯度端。
- `water_splatting/medium_calibration/gmvc_training.py`
  - 新增 `_compute_gmvc_v2_terms(...)`；
  - 实现 profiled radiance:

```text
J_p* = sum_i w_i T_i (I_i - B_i) / (sum_i w_i T_i^2 + eps)
I_hat_i = J_p* T_i + B_i
```

  - 默认 `J_p*` detach，medium 仍通过 `I_hat_i = sg(J_p*) T_i + B_i` 接收梯度；
  - 保留 weak symmetric closure，使用每条 sampled track 的 near/far depth-span pair。
- `water_splatting/water_splatting.py`
  - 新增 `_query_gmvc_medium_points(...)`；
  - 将 V2 losses 接入 `compute_gmvc_training_terms(...)`；
  - 新增 bounded medium projection 试验路径；
  - 扩展 GMVC grad JSONL，记录 profile / symmetric closure 对 RGB medium gradient ratio。
- `scripts/experiments/gmvc_v2_symmetric_profile_500.sh`
  - 新增 S0-S6 500-step continuation wrapper；
  - S6 是不带 bounded 的 `profile + weak closure`，用于避开 S3/S5 中已经失败的 bounded projection；
  - Curasao/Panama 的默认 profile 权重调为 `0.5`，保持 profile/RGB-medium gradient ratio 在更保守区间。

### 22.3 New config flags

```text
gmvc_v2_enabled: bool = False
lambda_gmvc_profile: float = 0.0
lambda_gmvc_symmetric_closure: float = 0.0
gmvc_profile_detach_j_star: bool = True
gmvc_v2_max_tracks_per_step: int = 512
gmvc_v2_min_observations_per_track: int = 2
gmvc_bounded_medium_enabled: bool = False
gmvc_bounded_medium_start_step: int = 10000
gmvc_bounded_medium_projection_steps: int = 500
gmvc_bounded_beta_log_scale: float = 0.15
gmvc_bounded_binf_logit_scale: float = 0.10
gmvc_bounded_init_from_first_batch: bool = True
```

所有新增训练项默认关闭。

### 22.4 Validation and smoke

Point-query diagnostic:

```text
script: scripts/diagnostics/diagnose_gmvc_point_query.py
scene: Panama M1 step-10000
output: renders/gmvc_v2_point_query_smoke_panama.json
mean_abs_error: 1.44e-5
max_abs_error: 4.88e-4
```

结论：point-query 与 full-map bilinear sample 已经没有坐标系级别错配；`max_abs` 未达原始 `1e-5` 理想阈值，主要集中在 tcnn/采样精度量级，作为后续风险记录。

Smoke tests:

```text
S2 profile 10-step Panama: pass
S1 symmetric closure 2-step Panama: pass
S3 bounded-only 2-step Panama: pass after wrapper fixed GMVC bank requirement
```

Gradient calibration:

```text
profile lambda=1.0: profile/RGB-medium grad ratio about 3.8% at first probe, later can peak above 10%
symmetric closure lambda=0.01: closure/RGB-medium grad ratio about 0.4%
wrapper default closure: 0.02 for Curasao/Panama
```

### 22.5 Track banks

```text
Curasao:
renders/gmvc_v2_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt
tracks: 60,168
observations: 701,330

Panama:
renders/gmvc_v2_track_banks/panama_m1_step10000_train_s4096/gmvc_track_bank.pt
tracks: 47,918
observations: 400,203
```

### 22.6 Experiment commands

Main 500-step matrix:

```bash
SCENE=curasao VARIANT=S0 GPU=6 LOAD_STEP=10000 TARGET_FINAL_STEP=10500 RUN_EVAL=1 STAMP=20260804_gmvc_v2_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
SCENE=curasao VARIANT=S1 GPU=7 LOAD_STEP=10000 TARGET_FINAL_STEP=10500 RUN_EVAL=1 STAMP=20260804_gmvc_v2_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
SCENE=curasao VARIANT=S2 GPU=8 LOAD_STEP=10000 TARGET_FINAL_STEP=10500 RUN_EVAL=1 STAMP=20260804_gmvc_v2_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
SCENE=curasao VARIANT=S3 GPU=9 LOAD_STEP=10000 TARGET_FINAL_STEP=10500 RUN_EVAL=1 STAMP=20260804_gmvc_v2_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh

SCENE=panama VARIANT=S0 GPU=6 LOAD_STEP=10000 TARGET_FINAL_STEP=10500 RUN_EVAL=1 STAMP=20260804_gmvc_v2_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
SCENE=panama VARIANT=S1 GPU=7 LOAD_STEP=10000 TARGET_FINAL_STEP=10500 RUN_EVAL=1 STAMP=20260804_gmvc_v2_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
SCENE=panama VARIANT=S2 GPU=8 LOAD_STEP=10000 TARGET_FINAL_STEP=10500 RUN_EVAL=1 STAMP=20260804_gmvc_v2_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
SCENE=panama VARIANT=S3 GPU=9 LOAD_STEP=10000 TARGET_FINAL_STEP=10500 RUN_EVAL=1 STAMP=20260804_gmvc_v2_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
```

Low-profile follow-up:

```bash
SCENE=curasao VARIANT=S2 LAMBDA_GMVC_PROFILE=0.5 EXPERIMENT_NAME=gmvc_v2_s2_profile05_curasao_seed42_step10000_to_10500 STAMP=20260804_gmvc_v2_profile05_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
SCENE=curasao VARIANT=S6 LAMBDA_GMVC_PROFILE=0.5 EXPERIMENT_NAME=gmvc_v2_s6_profile05_closure_curasao_seed42_step10000_to_10500 STAMP=20260804_gmvc_v2_profile05_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
SCENE=panama VARIANT=S2 LAMBDA_GMVC_PROFILE=0.5 EXPERIMENT_NAME=gmvc_v2_s2_profile05_panama_seed42_step10000_to_10500 STAMP=20260804_gmvc_v2_profile05_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
SCENE=panama VARIANT=S6 LAMBDA_GMVC_PROFILE=0.5 EXPERIMENT_NAME=gmvc_v2_s6_profile05_closure_panama_seed42_step10000_to_10500 STAMP=20260804_gmvc_v2_profile05_500 bash scripts/experiments/gmvc_v2_symmetric_profile_500.sh
```

Checkpoint decoupling diagnostic:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_checkpoint_tracks.py \
  --load-config <RUN_CONFIG> \
  --load-step 10500 \
  --test-mode inference \
  --split train \
  --samples-per-view 4096 \
  --max-tracks 30000 \
  --output-dir renders/gmvc_v2_checkpoint_diag_20260804_gmvc_v2_500/<scene_variant>
```

### 22.7 Image metrics

Relative to each scene S0 M1 continuation:

| Scene | Run | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Image gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Curasao | S1 symmetric closure | 32.7090 | +0.1149 | 0.957952 | +0.000273 | 0.107950 | -0.000162 | pass |
| Curasao | S2 profile, lambda 1.0 | 32.3168 | -0.2773 | 0.956500 | -0.001179 | 0.108387 | +0.000275 | fail PSNR |
| Curasao | S2 profile, lambda 0.5 | 32.4308 | -0.1633 | 0.957052 | -0.000627 | 0.108219 | +0.000107 | fail PSNR |
| Curasao | S3 bounded only | 9.9653 | -22.6288 | 0.666375 | -0.291304 | 0.412123 | +0.304011 | fail |
| Curasao | S6 profile 0.5 + closure | 32.4344 | -0.1597 | 0.957193 | -0.000486 | 0.108138 | +0.000026 | fail PSNR |
| Panama | S1 symmetric closure | 32.3406 | +0.0268 | 0.950163 | +0.000111 | 0.073914 | +0.000034 | pass |
| Panama | S2 profile, lambda 1.0 | 32.3912 | +0.0773 | 0.950053 | +0.000001 | 0.073774 | -0.000105 | pass |
| Panama | S2 profile, lambda 0.5 | 32.3273 | +0.0134 | 0.950071 | +0.000019 | 0.073922 | +0.000043 | pass |
| Panama | S3 bounded only | 26.3908 | -5.9230 | 0.937792 | -0.012259 | 0.097067 | +0.023187 | fail |
| Panama | S6 profile 0.5 + closure | 32.3337 | +0.0198 | 0.950135 | +0.000083 | 0.073710 | -0.000170 | pass |

Image gate:

```text
PSNR drop <= 0.15 dB
SSIM drop <= 0.0015
LPIPS increase <= 0.003
```

### 22.8 Checkpoint-level decoupling metrics

Relative to each scene S0 M1 continuation:

| Scene | Run | transfer Δ | object-J var Δ | raw closure Δ | fixed closure Δ | old norm closure Δ | Decoupling gate |
|---|---|---:|---:|---:|---:|---:|---|
| Curasao | S1 symmetric closure | -1.01% | -3.36% | -2.01% | -0.67% | -1.77% | fail |
| Curasao | S2 profile, lambda 1.0 | -3.38% | -5.80% | -3.52% | -4.73% | -5.69% | fail |
| Curasao | S2 profile, lambda 0.5 | -2.89% | -6.37% | -2.70% | -2.73% | -3.76% | fail |
| Curasao | S3 bounded only | +37.11% | -34.74% | +119.22% | +75.28% | +58.16% | fail |
| Curasao | S6 profile 0.5 + closure | -2.77% | -4.23% | -3.72% | -3.12% | -0.22% | fail |
| Panama | S1 symmetric closure | +0.55% | -3.34% | -0.91% | -0.38% | +0.23% | fail |
| Panama | S2 profile, lambda 1.0 | -2.40% | -6.43% | -2.23% | -1.23% | -1.43% | fail |
| Panama | S2 profile, lambda 0.5 | +2.18% | +0.24% | +2.13% | +1.26% | +0.97% | fail |
| Panama | S3 bounded only | +28.65% | +52.47% | +17.14% | +32.69% | +31.19% | fail |
| Panama | S6 profile 0.5 + closure | -3.37% | -5.85% | -4.36% | -0.89% | -0.27% | fail |

Decoupling gate:

```text
held-out transfer降低 >= 5%
object-J variance降低 >= 10%
raw closure不恶化超过 5%
fixed closure不恶化超过 5%
```

### 22.9 Gradient diagnostics

| Scene | Run | profile/RGB medium grad mean | profile max | closure/RGB medium grad mean | closure max |
|---|---|---:|---:|---:|---:|
| Curasao | S1 closure | 0.000 | 0.000 | 0.0185 | 0.0376 |
| Curasao | S2 profile 1.0 | 0.0752 | 0.1362 | 0.000 | 0.000 |
| Curasao | S2 profile 0.5 | 0.0406 | 0.0711 | 0.000 | 0.000 |
| Curasao | S6 profile 0.5 + closure | 0.0431 | 0.0821 | 0.0184 | 0.0312 |
| Panama | S1 closure | 0.000 | 0.000 | 0.0195 | 0.0538 |
| Panama | S2 profile 1.0 | 0.0441 | 0.1261 | 0.000 | 0.000 |
| Panama | S2 profile 0.5 | 0.0207 | 0.0650 | 0.000 | 0.000 |
| Panama | S6 profile 0.5 + closure | 0.0191 | 0.0396 | 0.0177 | 0.0336 |

`lambda=0.5` 让 Curasao profile 梯度落入建议区间，但 RGB PSNR 仍略低于 image gate，说明 Curasao 的失败不是单纯梯度过强。Panama 对 profile 更稳定，`lambda=1.0` 是当前最好的 image candidate。

### 22.10 Conclusions

1. **V2 profile 有真实解耦信号，但未达到正式 gate。** Curasao S2 profile 1.0 同时降低 transfer、object-J variance、raw/fixed closure，但 PSNR 下降 `0.277 dB`；profile 0.5 仍差 `0.163 dB`，刚好超过 image gate。Panama profile 1.0 同时改善 image metrics 和部分 decoupling metrics，但 transfer 只降低 `2.4%`，object-J variance 只降低 `6.4%`，未达 `5%/10%` decoupling gate。

2. **Symmetric closure 是安全但弱的辅助项。** Curasao/Panama S1 都通过 image gate，Curasao RGB 还提升 `+0.115 dB`，但 decoupling 改善幅度只有约 `1%–3%`，不能作为进入 15k 的核心方法。

3. **Explicit bounded medium projection 当前不可用。** S3 在 Curasao 和 Panama 都造成大幅 RGB 崩坏。Curasao 虽然 object-J variance 下降 `34.7%`，但 transfer、closure 和图像指标严重恶化；Panama 则所有 decoupling 指标也恶化。因此 S4/S5 bounded combinations 不应继续运行。

4. **不进入 JapaneseGradens/IUI3 或 15k。** 按原计划，只有 Curasao/Panama 500-step 同时通过 image + decoupling gate 才扩展压力场景。当前没有任何 V2 run 同时满足两个 gate。

当前 gate 结论：

```text
进入 15k: No
进入 JapaneseGradens/IUI3: No
最佳 image candidate: Panama S2 profile lambda=1.0
最佳 Curasao image candidate: S1 symmetric closure
最佳 decoupling candidate: none
是否确认 V2 核心信号存在: Yes, profile loss can reduce held-out decoupling metrics
是否确认 V2 已成为成功模块: No
下一步: 不继续 bounded；若继续 GMVC，应重新设计低扰动 profile schedule 或 medium-only freeze/ramp，再考虑 object-step active DC proxy
```

## 23. GMVC-V3 alternating medium-object calibration

日期：2026-08-04

### 23.1 Motivation

V2 结果说明 profiled radiance 有真实解耦信号，但旧实现仍有三个限制：

1. track bank 仍由旧 M1 medium 反演出的 `J`、旧 M1 transmission 和 `j_valid` 筛选，容易只保留旧 M1 已经自洽的轨迹；
2. 解析 `J*` 用 L2 解，外层却用 Charbonnier，且 `J*` detach，目标函数与梯度不严格匹配；
3. medium 被校准后，Gaussian `features_dc` 没有用新的在线 `J*` 重新对齐，因此可能出现解耦指标改善但 RGB 下降。

V3 因此实现三项修改：

```text
geometry-only V2/V3 track bank
robust IRLS profiled medium objective + track-balanced averaging
4 medium steps : 1 object DC-only calibration step
```

### 23.2 Code changes

Changed files:

```text
water_splatting/medium_calibration/gmvc_types.py
water_splatting/medium_calibration/gmvc_tracks.py
scripts/diagnostics/build_gmvc_tracks.py
water_splatting/medium_calibration/gmvc_training.py
water_splatting/water_splatting.py
scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh
scripts/experiments/gmvc_phase_b_common.sh
scripts/experiments/gmvc_v3_alternating_1000.sh
```

New/default-off model flags:

```text
gmvc_profile_loss_mode: charbonnier | irls_l2
gmvc_profile_track_balanced
gmvc_profile_irls_delta
gmvc_profile_irls_max_weight
gmvc_profile_min_hessian
gmvc_profile_min_transmission_span
gmvc_profile_min_depth_span_rel
gmvc_v3_enabled
lambda_gmvc_object
gmvc_v3_medium_steps
gmvc_v3_object_steps
gmvc_v3_object_source
gmvc_object_track_balanced
gmvc_object_j_clamp_min / max
gmvc_object_min_hessian
gmvc_object_min_depth_span_rel
```

Track-bank flags:

```text
--geometry-only-v2-bank
--signal-min
--signal-max
--signal-softness
```

V3 routing:

```text
medium phase: profile / symmetric closure -> medium MLP only through normal optimizer ownership
object phase: online detached J* -> J_proxy_raw, intended to update features_dc
bounded S3/S4/S5 path remains disabled
```

The 1000-step wrapper now defaults to calibrated V3 scales:

```text
Curasao profile default = 40
Panama profile default = 80
GMVC_GRAD_LOG_EVERY = 49
```

`49` avoids phase-locking the gradient diagnostic to the 4+1 alternating cycle boundary.

### 23.3 Smoke tests

Static checks:

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/medium_calibration/gmvc_training.py \
  water_splatting/medium_calibration/gmvc_types.py \
  water_splatting/medium_calibration/gmvc_tracks.py \
  scripts/diagnostics/build_gmvc_tracks.py \
  water_splatting/water_splatting.py

bash -n scripts/experiments/gmvc_phase_b_common.sh
bash -n scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh
bash -n scripts/experiments/gmvc_v3_alternating_1000.sh
git diff --check
```

Geometry-only bank smoke:

```text
scene: Curasao
max train images: 4
samples/view: 128
accepted tracks: 270
accepted observations: 1021
output: renders/gmvc_v3_smoke_bank/curasao/gmvc_track_bank.pt
```

Alternating object smoke:

```text
run: Curasao A2, 5 steps
grad log: logs/gmvc_v3_grad_a2_alternating_object_curasao_20260804_gmvc_v3_a2_grad_smoke5.jsonl
object phase step: 10004
object_to_rgb_dc_grad_ratio: 0.00126 during early ramp
object geometry grad: 0
object opacity grad: 0
object medium grad: 0
```

Profile scale smoke:

```text
Curasao profile=50, full ramp smoke:
profile/RGB-medium grad ratio: 0.0114-0.0125 on active steps

Curasao profile=100, 1000-step:
profile/RGB-medium grad mean: 0.0254-0.0268
result: strong decoupling but RGB unsafe
```

### 23.4 Track banks

| Scene | Bank | Tracks | Observations | Train views | Per-camera cap |
|---|---|---:|---:|---:|---:|
| Curasao | `renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt` | 64,810 | 760,864 | 18 | 20,000 |
| Panama | `renders/gmvc_v3_geometry_track_banks/panama_m1_step10000_train_s4096/gmvc_track_bank.pt` | 49,216 | 422,324 | 15 | 20,000 |

Both banks are geometry-only and do not use old M1 `J`, old M1 transmission gating, or old `j_valid` filtering.

### 23.5 Commands

Curasao tuned runs:

```bash
SCENE=curasao VARIANT=A1 LAMBDA_GMVC_PROFILE=40 STAMP=20260804_gmvc_v3_curasao_profile40_1000 bash scripts/experiments/gmvc_v3_alternating_1000.sh
SCENE=curasao VARIANT=A2 LAMBDA_GMVC_PROFILE=40 STAMP=20260804_gmvc_v3_curasao_profile40_1000 bash scripts/experiments/gmvc_v3_alternating_1000.sh
SCENE=curasao VARIANT=A3 LAMBDA_GMVC_PROFILE=40 STAMP=20260804_gmvc_v3_curasao_profile40_1000 bash scripts/experiments/gmvc_v3_alternating_1000.sh
```

Panama runs:

```bash
SCENE=panama VARIANT=A0 STAMP=20260804_gmvc_v3_panama_profile80_1000 bash scripts/experiments/gmvc_v3_alternating_1000.sh
SCENE=panama VARIANT=A1 STAMP=20260804_gmvc_v3_panama_profile80_1000 bash scripts/experiments/gmvc_v3_alternating_1000.sh
SCENE=panama VARIANT=A2 STAMP=20260804_gmvc_v3_panama_profile80_1000 bash scripts/experiments/gmvc_v3_alternating_1000.sh
SCENE=panama VARIANT=A3 STAMP=20260804_gmvc_v3_panama_profile80_1000 bash scripts/experiments/gmvc_v3_alternating_1000.sh
```

Panama higher-profile follow-up:

```bash
SCENE=panama VARIANT=A1 LAMBDA_GMVC_PROFILE=120 STAMP=20260804_gmvc_v3_panama_profile120_1000 bash scripts/experiments/gmvc_v3_alternating_1000.sh
SCENE=panama VARIANT=A2 LAMBDA_GMVC_PROFILE=120 STAMP=20260804_gmvc_v3_panama_profile120_1000 bash scripts/experiments/gmvc_v3_alternating_1000.sh
SCENE=panama VARIANT=A3 LAMBDA_GMVC_PROFILE=120 STAMP=20260804_gmvc_v3_panama_profile120_1000 bash scripts/experiments/gmvc_v3_alternating_1000.sh
```

Checkpoint diagnostics:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_checkpoint_tracks.py \
  --load-config <RUN_CONFIG> \
  --split train \
  --samples-per-view 4096 \
  --max-tracks 30000 \
  --output-dir renders/gmvc_v3_checkpoint_diag_20260804_<scene>/<variant>
```

### 23.6 Curasao 1000-step results

Baseline: A0 M1 continuation at step 11000.

| Run | Profile | Closure | Object | ΔPSNR | ΔSSIM | ΔLPIPS | Transfer Δ | Object-J var Δ | Fixed closure Δ | Consensus recon Δ | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 profile | 40 | 0 | 0 | -0.1733 | -0.000862 | +0.000488 | -1.36% | -7.61% | -2.02% | -1.37% | image fail |
| A2 alternating object | 40 | 0 | 0.004 | -0.1330 | -0.000696 | +0.000341 | -1.20% | -5.03% | -0.93% | -1.27% | partial pass |
| A3 object + closure | 40 | 0.005 | 0.004 | -0.1299 | -0.000638 | +0.000283 | -1.36% | -7.77% | -1.95% | -1.55% | partial pass |
| A1 profile | 100 | 0 | 0 | -0.4929 | -0.001850 | +0.001044 | -4.65% | -15.71% | -3.79% | -4.46% | image fail |
| A2 alternating object | 100 | 0 | 0.004 | -0.4061 | -0.001559 | +0.000858 | -4.45% | -16.74% | -2.63% | -4.41% | image fail |
| A3 object + closure | 100 | 0.005 | 0.004 | -0.4024 | -0.001522 | +0.000867 | -3.99% | -15.03% | -2.57% | -3.87% | image fail |

Interpretation:

```text
profile=100 proves the geometry-only IRLS profile objective can strongly reduce Curasao transfer and object-J variance,
but RGB degradation is too large.

profile=40 is the best current Curasao candidate:
it stays inside PSNR and LPIPS safety limits for A2/A3 and gives consistent but sub-gate decoupling gains.
```

### 23.7 Panama 1000-step results

Baseline: A0 M1 continuation at step 11000.

| Run | Profile | Closure | Object | ΔPSNR | ΔSSIM | ΔLPIPS | Transfer Δ | Object-J var Δ | Fixed closure Δ | Consensus recon Δ | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 profile | 80 | 0 | 0 | -0.0243 | -0.000065 | -0.000163 | +1.98% | +0.98% | -2.10% | +1.54% | image pass, decoupling fail |
| A2 alternating object | 80 | 0 | 0.006 | -0.0087 | -0.000033 | -0.000141 | -0.36% | +0.07% | -2.25% | -0.40% | image pass, decoupling fail |
| A3 object + closure | 80 | 0.0075 | 0.006 | -0.0106 | -0.000004 | -0.000117 | +0.72% | +3.99% | -2.96% | +0.04% | image pass, decoupling fail |
| A1 profile | 120 | 0 | 0 | -0.0087 | -0.000046 | -0.000090 | +1.60% | +2.87% | -2.09% | +0.85% | image pass, decoupling fail |
| A2 alternating object | 120 | 0 | 0.006 | -0.0601 | -0.000204 | +0.001073 | +0.88% | +3.17% | -0.89% | +0.81% | image pass, decoupling fail |
| A3 object + closure | 120 | 0.0075 | 0.006 | -0.0710 | -0.000180 | +0.000968 | +0.44% | +5.57% | -1.32% | +0.68% | image pass, decoupling fail |

Panama gradient diagnostics:

| Profile | Run | profile/RGB-medium mean | closure/RGB-medium mean | object/RGB-DC mean | Object route |
|---:|---|---:|---:|---:|---|
| 80 | A1 | 0.0359 | 0.0000 | 0.0000 | n/a |
| 80 | A2 | 0.0294 | 0.0000 | 0.0067 | DC only; geometry/opacity/medium zero |
| 80 | A3 | 0.0340 | 0.0078 | 0.0067 | DC only; geometry/opacity/medium zero |
| 120 | A1 | 0.0418 | 0.0000 | 0.0000 | n/a |
| 120 | A2 | 0.0415 | 0.0000 | 0.0066 | DC only; geometry/opacity/medium zero |
| 120 | A3 | 0.0398 | 0.0062 | 0.0067 | DC only; geometry/opacity/medium zero |

Interpretation:

```text
Panama V3 is RGB-safe and object-gradient routing works,
but profile80/profile120 no longer reproduce the V2 Panama S2 transfer/J-var improvement.
The strongest consistent Panama signal is only closure-floor reduction.
```

### 23.8 Gate decision

V3 final gate:

```text
transfer下降 >= 5%
J-var下降 >= 10%
PSNR下降 <= 0.15 dB
LPIPS增加 <= 0.003
raw/fixed closure不恶化
```

Gate outcome:

| Scene | Best safe run | Image safety | Transfer | J-var | Closure | Decision |
|---|---|---|---:|---:|---:|---|
| Curasao | A3 profile40 | pass | -1.36% | -7.77% | -1.95% | partial success, below full gate |
| Panama | A2 profile80 | pass | -0.36% | +0.07% | -2.25% | fail decoupling gate |
| Panama | A1 profile120 | pass | +1.60% | +2.87% | -2.09% | fail decoupling gate |

Current decision:

```text
进入 15k: No
进入 JapaneseGradens/IUI3: No
继续 bounded path: No
最佳 Curasao V3 candidate: A3 profile40
最佳 Panama V3 candidate: none; A2 profile80 is image-safe but not a decoupling win
```

### 23.9 Conclusion

GMVC-V3 should be recorded as a partial but insufficient success:

1. Geometry-only banks and robust IRLS profile remove the old M1 filtering bias and can produce stronger Curasao decoupling than low-weight V2.
2. Alternating object calibration is correctly routed: object loss reaches `features_dc` and does not update geometry, opacity, or medium in object phase.
3. Object calibration helps recover RGB on Curasao: at profile40, A2/A3 recover about `0.040-0.043 dB` versus A1 and bring PSNR back inside the safety gate.
4. The full V3 gate is not met. Curasao is still below `5%/10%` transfer/J-var improvement, and Panama does not show consistent transfer/J-var improvement at profile80 or profile120.
5. V3 should not proceed to 15k or pressure scenes yet. If GMVC continues, the next change should target why geometry-only IRLS profile loses the V2 Panama transfer/J-var signal, not simply increase `lambda_gmvc_profile`.

Recommended next experiment, if continuing GMVC:

```text
Compare V2 filtered bank vs V3 geometry-only bank under the same IRLS profile objective,
and separately compare Charbonnier profile vs IRLS-L2 on the same bank.
This isolates whether Panama regression comes from bank construction or objective scaling.
```

## 24. Fixed-bank evaluation and phase-matched V3 controls

日期：2026-08-04

### 24.1 Motivation

Section 23 used `diagnose_gmvc_checkpoint_tracks.py`, which rebuilds tracks for every evaluated checkpoint. That means A0 and A3 can be evaluated on different 3D points, different observation rows, different weights, and a different held-out split. For 1-3% transfer changes this is not reliable enough.

This round fixes the evaluation surface first:

```text
Eval-F = fixed M1-filtered bank
Eval-G = fixed geometry-only bank
```

Both are generated once from the M1 step-10000 checkpoint and then reused for all checkpoints. The fixed diagnostic never rebuilds correspondences from the evaluated model.

### 24.2 Code changes

New diagnostic:

```text
scripts/diagnostics/diagnose_gmvc_fixed_bank.py
```

It loads:

```text
--load-config <checkpoint config>
--track-bank <fixed bank .pt>
```

and evaluates all checkpoints on fixed:

```text
track_id
observation rows
GT RGB
fixed propagation depth
camera context
geometry confidence weights
held-out split
```

New fixed-bank metrics:

```text
transfer_l1
object_j_variance
closure_signal_floor_l1
consensus_j_reconstruction_l1
object_target_l1
dc_cross_view_variance
dc_recomposition_l1
track_profile_residual p50/p75/p90/p95
IRLS effective weight ratio
J* outside [-0.1, 1.1] ratio
valid hessian / transmission-span / depth-span ratios
```

`object_target_l1`, `dc_cross_view_variance`, and `dc_recomposition_l1` use rendered `J_proxy_raw`, so they actually measure the Gaussian DC/object branch. This fixes the Section 23 issue where `object_j_variance` only measured GT inverted through current medium and did not include Gaussian appearance.

Additional training flags:

```text
gmvc_v3_freeze_medium_on_object_phase: bool = False
gmvc_v3_target_current_camera_tracks: bool = False
```

`gmvc_v3_freeze_medium_on_object_phase=True` detaches medium outputs during object phase. This approximates strict block-coordinate alternation where object phase does not update the medium through the RGB loss.

`gmvc_v3_target_current_camera_tracks=True` samples object-phase tracks from tracks that contain the current training camera. This reduces wasted object-loss samples after the current-camera mask.

Wrapper update:

```text
VARIANT=A1C
```

means:

```text
4 profile steps + 1 auxiliary-off step
object loss = 0
```

This is the phase-matched control for A2.

### 24.3 Fixed-bank commands

Curasao A0-A3 fixed re-evaluation:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_fixed_bank.py \
  --load-config <Curasao A0/A1/A2/A3 config> \
  --track-bank renders/gmvc_v2_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt \
  --max-tracks 30000 \
  --output-dir renders/gmvc_fixed_bank_diag_20260804/curasao_profile40/<run>/evalf

/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_fixed_bank.py \
  --load-config <Curasao A0/A1/A2/A3 config> \
  --track-bank renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt \
  --max-tracks 30000 \
  --output-dir renders/gmvc_fixed_bank_diag_20260804/curasao_profile40/<run>/evalg
```

Panama fixed re-evaluation uses the analogous Panama Eval-F and Eval-G banks.

### 24.4 Fixed-bank Curasao profile40 results

Relative to fixed-bank A0 M1 continuation:

| Bank | Run | Transfer Δ | J-var Δ | Closure Δ | Obj-fit Δ | DC-var Δ | Recomp Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| Eval-F | A1 profile | -2.19% | -6.97% | -1.46% | -2.50% | -1.64% | -1.89% |
| Eval-F | A2 object | -1.83% | -6.04% | -1.29% | -2.43% | -1.14% | -1.80% |
| Eval-F | A3 object+closure | -1.77% | -5.31% | -1.31% | -1.72% | -0.68% | -1.54% |
| Eval-G | A1 profile | -2.15% | -7.38% | -1.06% | -2.63% | -1.84% | -1.58% |
| Eval-G | A2 object | -1.82% | -6.49% | -0.95% | -3.01% | -1.56% | -1.68% |
| Eval-G | A3 object+closure | -1.78% | -5.74% | -1.04% | -2.25% | -0.91% | -1.39% |

Interpretation:

```text
Fixed Eval-F/Eval-G confirms Curasao profile40 is a real partial success.
The original checkpoint-rebuild diagnostic under-estimated A1/A2 transfer and J-var gains.
Object metrics are now meaningful: A2 improves object_target_l1 and dc_recomposition_l1 on both fixed banks.
```

### 24.5 Fixed-bank Panama profile80 results

Relative to fixed-bank A0 M1 continuation:

| Bank | Run | Transfer Δ | J-var Δ | Closure Δ | Obj-fit Δ | DC-var Δ | Recomp Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| Eval-F | A1 profile | -0.73% | -1.98% | -1.63% | +0.07% | -1.01% | -0.19% |
| Eval-F | A2 object | -0.63% | -1.49% | -1.52% | +0.55% | -0.79% | +0.04% |
| Eval-F | A3 object+closure | -0.73% | +0.01% | -0.59% | +0.70% | +0.83% | -0.35% |
| Eval-G | A1 profile | -0.94% | -1.58% | -1.76% | -0.06% | -0.97% | -0.51% |
| Eval-G | A2 object | -0.82% | -1.15% | -1.64% | +0.18% | -0.59% | -0.39% |
| Eval-G | A3 object+closure | -0.88% | +0.75% | -1.01% | +0.10% | +1.03% | -0.88% |

Interpretation:

```text
Fixed evaluation changes the Panama conclusion.
Profile80 is not directionless; A1/A2 reduce transfer and J-var slightly on both fixed banks.
However, the magnitude remains far below the full 5%/10% gate.
A3 closure hurts Panama J-var and DC-var, so closure should not be preferred here.
```

### 24.6 Curasao phase-matched object controls

Runs:

| Label | Run | Schedule |
|---|---|---|
| C0 | A1C | 4 profile + 1 auxiliary-off |
| C1 | Existing A2 | 4 profile + 1 object |
| C2 | Targeted A2 | 4 profile + 1 object, current-camera track sampling |
| C3 | Targeted strict A2 | C2 + medium detached during object phase |

Image metrics versus A0:

| Run | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---:|---:|---:|
| C0 A1C | -0.1345 | -0.000699 | +0.000381 |
| C1 existing A2 | -0.1330 | -0.000696 | +0.000341 |
| C2 targeted A2 | -0.1282 | -0.000697 | +0.000359 |
| C3 targeted strict A2 | -0.1522 | -0.000633 | +0.000271 |

Gradient diagnostics:

| Run | profile/RGB-medium mean | object/RGB-DC mean | object route |
|---|---:|---:|---|
| C0 A1C | 0.0077 | 0.0000 | no object loss |
| C2 targeted A2 | 0.0077 | 0.0040 | DC only; geometry/opacity/medium zero |
| C3 targeted strict A2 | 0.0068 | 0.0039 | DC only; geometry/opacity/medium zero |

Fixed-bank object-control metrics versus A0:

| Bank | Run | Transfer Δ | J-var Δ | Closure Δ | Obj-fit Δ | DC-var Δ | Recomp Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| Eval-F | C0 A1C | -1.83% | -5.85% | -1.26% | -2.06% | -1.40% | -1.57% |
| Eval-F | C2 targeted | -1.84% | -6.04% | -1.32% | -2.49% | -1.05% | -1.85% |
| Eval-F | C3 targeted strict | -2.00% | -6.52% | -1.39% | -2.02% | -2.70% | -1.30% |
| Eval-G | C0 A1C | -1.82% | -6.25% | -0.94% | -2.18% | -1.48% | -1.33% |
| Eval-G | C2 targeted | -1.82% | -6.48% | -0.97% | -3.12% | -1.39% | -1.73% |
| Eval-G | C3 targeted strict | -2.02% | -7.50% | -1.02% | -2.97% | -2.72% | -1.19% |

Interpretation:

```text
C2 is the best object calibration variant so far.
Against phase-matched C0, C2 slightly improves PSNR, object-fit, and recomposition.
Targeted sampling raises object/RGB-DC gradient visibility from logging-zero to about 0.4% mean.
C3 strict medium freeze improves transfer/J-var and DC-var, but PSNR falls just outside the -0.15 dB safety line.
Strict alternation is promising for decoupling but needs lower strength or slower schedule.
```

### 24.7 Panama P0-P4 single-factor matrix

All runs:

```text
start checkpoint: Panama M1 step 10000
length: 500 steps
object loss: off
closure: off
profile gates: hessian=0, transmission-span=0, depth-span=0
```

Matrix:

| Run | Bank | Profile loss | Averaging |
|---|---|---|---|
| P0 | filtered | Charbonnier | observation-balanced |
| P1 | geometry-only | Charbonnier | observation-balanced |
| P2 | filtered | IRLS-L2 | observation-balanced |
| P3 | filtered | Charbonnier | track-balanced |
| P4 | geometry-only | IRLS-L2 | track-balanced |

Image metrics versus A0:

| Run | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---:|---:|---:|
| P0 | -0.0327 | -0.000237 | +0.000828 |
| P1 | -0.0621 | -0.000223 | +0.000974 |
| P2 | -0.1137 | -0.000260 | +0.000410 |
| P3 | -0.1152 | -0.000277 | +0.000733 |
| P4 | -0.0988 | -0.000298 | +0.000664 |

Fixed-bank metrics versus A0:

| Bank | Run | Transfer Δ | J-var Δ | Closure Δ | Obj-fit Δ | DC-var Δ | Recomp Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| Eval-F | P0 | -0.38% | -0.10% | -0.36% | -3.51% | +1.92% | -2.31% |
| Eval-F | P1 | -0.62% | +0.85% | -0.41% | -0.38% | +1.32% | -0.57% |
| Eval-F | P2 | -0.71% | +0.42% | +0.72% | -2.61% | +1.71% | -2.05% |
| Eval-F | P3 | -0.52% | +0.46% | -0.70% | +0.19% | +0.81% | -0.06% |
| Eval-F | P4 | -0.52% | +0.75% | +0.09% | -0.22% | +1.34% | -0.56% |
| Eval-G | P0 | -0.07% | +0.73% | -0.03% | -4.61% | +1.00% | -2.60% |
| Eval-G | P1 | -0.72% | +0.67% | -0.69% | -1.82% | +0.38% | -1.24% |
| Eval-G | P2 | -0.48% | +1.16% | +0.64% | -4.45% | +1.07% | -2.72% |
| Eval-G | P3 | -0.34% | +0.55% | -0.49% | -1.39% | -0.17% | -0.59% |
| Eval-G | P4 | -0.56% | +1.23% | -0.23% | -1.10% | +0.50% | -1.03% |

Important note:

```text
P0-P4 accidentally reused the same default gradient JSONL path because VARIANT=A1 and STAMP were identical.
The final P4 run overwrote the P0-P3 gradient logs.
Image and fixed-bank JSON metrics are unaffected.
Future factor-matrix runs must use unique STAMP or GMVC_GRAD_LOG_PATH per run.
```

Interpretation:

```text
The Panama V2-to-V3 regression is not caused only by geometry-only banks.
P0, which should be closest to V2 filtered Charbonnier observation-balanced, no longer reproduces the old V2 S2 -2.40% transfer / -6.43% J-var signal under this fixed 500-step/gates-off setup.
Geometry-only bank alone improves transfer more than P0 but worsens J-var.
IRLS-L2 and track-balanced variants also do not recover J-var.
The most stable Panama gain across P0-P4 is object/recomposition proxy improvement, not medium-object J-var decoupling.
```

### 24.8 Historical V2 S2 under fixed-bank evaluation

To check whether P0 failed because it differed from the historical V2 implementation, the existing historical Panama V2 S2 checkpoints were also evaluated on fixed Eval-F/Eval-G:

| Historical run | Bank | Transfer Δ | J-var Δ | Closure Δ | Obj-fit Δ | Recomp Δ |
|---|---|---:|---:|---:|---:|---:|
| V2 S2 profile 1.0 | Eval-F | -0.41% | -0.13% | -0.33% | -3.54% | -2.33% |
| V2 S2 profile 1.0 | Eval-G | -0.10% | +0.70% | -0.01% | -4.81% | -2.71% |
| V2 S2 profile 0.5 | Eval-F | -0.32% | +2.25% | +0.06% | +1.14% | +0.26% |
| V2 S2 profile 0.5 | Eval-G | -0.12% | +2.78% | +0.03% | +0.22% | -0.05% |

This changes the interpretation of the old V2 result:

```text
The previously reported Panama V2 S2 -2.40% transfer / -6.43% J-var improvement does not survive fixed-bank evaluation.
The old checkpoint-rebuild diagnostic likely mixed true model change with evaluation-set drift.
Therefore Panama should not be treated as a lost strong-positive V2 case.
Under fixed evaluation, Panama profile losses mostly give small transfer/closure or object-proxy gains, not robust J-var gains.
```

### 24.9 Updated decision

Current gate status:

```text
进入 15k: No
进入 JapaneseGradens/IUI3: No
继续 Curasao V3 object line: Yes, but only as 1000-step refinement
继续 Panama V3 line: No 15k; fixed evaluation shows no strong V2/V3 decoupling signal
```

Best candidates:

```text
Curasao decoupling: C3 targeted strict object, but PSNR just fails safety
Curasao balanced: C2 targeted object
Panama image-safe: profile80 A2 or P0, but decoupling is too weak
```

Next concrete step:

```text
Run Curasao C2/C3 at lower object/profile strength or slower ramp.
For Panama, do not continue profile-weight escalation; only revisit if a new fixed-bank objective directly improves J-var.
```

### 24.10 Curasao object-phase medium gradient-scale sweep

Motivation:

```text
The C2/C3 comparison suggested that object-phase RGB gradients into the medium branch may be too strong.
This sweep tests whether an internal Pareto point exists between object target stability and RGB medium adaptability.
Only object-phase RGB gradients through medium outputs are scaled.
Medium output values are unchanged.
Geometry, opacity, SH-rest, and the object auxiliary loss route are not scaled.
```

Implementation:

```text
Added WaterSplattingModelConfig.gmvc_v3_object_phase_medium_grad_scale, default 1.0.
During GMVC-V3 object phase, medium.rgb, medium.bs, medium.attn, b_inf, and b_inf_residual use a straight-through gradient scale:
value.detach() + scale * (value - value.detach()).
gmvc_v3_freeze_medium_on_object_phase still forces scale 0.0.
directions are not detached or scaled.
```

Sweep wrapper:

```text
scripts/experiments/gmvc_v3_curasao_medium_grad_sweep_1000.sh

Common setup:
scene=curasao
start=M1 step 10000
target step=11000
seed=42
medium_context_mode=dir_xy_camera
b_inf_mode=tied
lambda_gmvc_profile=40
lambda_gmvc_object=0.004
gmvc_v3_target_current_camera_tracks=True
gmvc_grad_log_every=49
```

Runs:

| Run | Object-phase medium RGB grad scale |
|---|---:|
| G100 | 1.00 |
| G075 | 0.75 |
| G050 | 0.50 |
| G025 | 0.25 |
| G000 | 0.00 |

Smoke:

```text
G050 5-step smoke passed.
Config recorded gmvc_v3_object_phase_medium_grad_scale=0.50 and targeted current-camera tracks.
No CUDA/autograd error.
Gradient log was written to logs/gmvc_v3_grad_a2_alternating_object_curasao_20260804_gmvc_v3_gradscale_smoke_g050.jsonl.
```

Image metrics:

| Run | PSNR | dPSNR vs A0 | dPSNR vs C0 | SSIM | dSSIM vs A0 | LPIPS | dLPIPS vs A0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 32.9690 | +0.0000 | +0.1345 | 0.957966 | +0.000000 | 0.106611 | +0.000000 |
| C0 | 32.8345 | -0.1345 | +0.0000 | 0.957266 | -0.000699 | 0.106992 | +0.000381 |
| C2 | 32.8408 | -0.1282 | +0.0063 | 0.957269 | -0.000697 | 0.106971 | +0.000359 |
| C3 | 32.8169 | -0.1522 | -0.0177 | 0.957333 | -0.000633 | 0.106882 | +0.000271 |
| G100 | 32.8370 | -0.1320 | +0.0025 | 0.957277 | -0.000689 | 0.106910 | +0.000298 |
| G075 | 32.8350 | -0.1340 | +0.0005 | 0.957265 | -0.000700 | 0.106915 | +0.000303 |
| G050 | 32.8427 | -0.1264 | +0.0081 | 0.957270 | -0.000696 | 0.106937 | +0.000326 |
| G025 | 32.8406 | -0.1285 | +0.0061 | 0.957256 | -0.000710 | 0.106928 | +0.000316 |
| G000 | 32.8183 | -0.1507 | -0.0162 | 0.957343 | -0.000623 | 0.106901 | +0.000289 |

Fixed-bank metrics, percent change versus phase-matched C0:

Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| C2 | -0.00% | -0.20% | -0.06% | -0.44% | +0.36% | -0.28% |
| C3 | -0.17% | -0.71% | -0.12% | +0.04% | -1.32% | +0.28% |
| G100 | +0.00% | -0.17% | -0.04% | -0.28% | +0.21% | -0.18% |
| G075 | +0.00% | -0.20% | -0.05% | -0.38% | +0.28% | -0.23% |
| G050 | -0.01% | -0.27% | -0.02% | -0.53% | +0.25% | -0.31% |
| G025 | -0.02% | -0.25% | -0.00% | -0.67% | +0.29% | -0.41% |
| G000 | -0.17% | -0.73% | -0.14% | +0.01% | -1.42% | +0.27% |

Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| C2 | -0.00% | -0.24% | -0.03% | -0.96% | +0.09% | -0.40% |
| C3 | -0.21% | -1.34% | -0.09% | -0.80% | -1.27% | +0.14% |
| G100 | +0.00% | -0.20% | -0.02% | -0.80% | +0.11% | -0.31% |
| G075 | +0.00% | -0.24% | -0.02% | -0.85% | +0.22% | -0.33% |
| G050 | -0.01% | -0.31% | +0.01% | -1.06% | +0.13% | -0.45% |
| G025 | -0.02% | -0.33% | +0.02% | -1.24% | +0.26% | -0.56% |
| G000 | -0.21% | -1.36% | -0.10% | -0.78% | -1.37% | +0.16% |

Gradient route audit:

| Run | Object log rows | RGB medium grad mean in object rows | Object aux DC grad mean | Object aux / RGB DC ratio mean | Object aux medium max | Object aux geometry max | Object aux opacity max |
|---|---:|---:|---:|---:|---:|---:|---:|
| G100 | 4 | 0.455006 | 0.000013030290 | 0.020167 | 0.000000 | 0.000000 | 0.000000 |
| G075 | 4 | 0.341295 | 0.000013027328 | 0.020251 | 0.000000 | 0.000000 | 0.000000 |
| G050 | 4 | 0.227001 | 0.000013028543 | 0.020260 | 0.000000 | 0.000000 | 0.000000 |
| G025 | 4 | 0.112840 | 0.000013016824 | 0.020266 | 0.000000 | 0.000000 | 0.000000 |
| G000 | 4 | 0.000000 | 0.000013025069 | 0.019436 | 0.000000 | 0.000000 | 0.000000 |

Control reproduction:

```text
G100 approximately reproduces C2 targeted object.
G000 approximately reproduces C3 strict medium-freeze behavior.
The implementation therefore passes the first control check.
The object auxiliary gradient remains DC-only; medium, geometry, and opacity object-aux gradients stay zero in logged object rows.
The object-phase RGB medium gradient decreases monotonically with the requested scale.
```

Interpretation:

```text
No clean internal Pareto point was found.
G050 is the best image-safe run in this sweep and improves PSNR versus both C0 and C2, but it does not inherit the C3/G000 DC-var improvement.
G025 gives the strongest object-fit and recomposition gains, but DC-var remains worse than C0 and RGB is not better than G050.
G000 keeps the transfer/J-var/DC-var direction of C3, but still carries the same PSNR safety problem.
The main tradeoff therefore still looks like direct competition between medium adaptability and strict object-medium decoupling, not a simple medium-gradient-scale sweet spot.
```

Updated decision:

```text
进入 15k: No
扩展 JapaneseGradens/IUI3/Panama: No
Best image-safe candidate: G050
Best decoupling candidate: G000/C3, but PSNR unsafe
Next single-factor experiment, if continuing GMVC-V3: adjust ramp or object lambda rather than fine-grained medium gradient scale.
```

## 26. G000 ramp sweep with optimizer-level medium detachment

The gradient-scale sweep above left one unresolved question: G000/C3 gave the strongest decoupling direction, but its PSNR drop was just outside the safety line. This section tests the smallest timing change: keep object-phase medium RGB gradient scale at 0.00, keep all Curasao V3 settings fixed, and sweep only `gmvc_ramp_steps`.

Implementation changes:

```text
water_splatting/water_splatting.py
- _scale_gradient(value, scale) now returns value.detach() when scale <= 0.0.
- This makes G000 a true detach path for the object-phase RGB medium tensors, instead of sending a zero current gradient through medium parameters.
- GMVC JSONL logging now includes RGB gradients for features_rest, geometry, opacity, the current phase flag, lambda values, sampled/valid track counts, J* drift, and medium parameter deltas.

water_splatting/medium_calibration/gmvc_training.py
- _compute_gmvc_v2_terms() now receives the persistent model state.
- The state caches previous per-track J* and previous per-observation medium_attn, medium_bs, b_inf, and transmission values.
- Logged diagnostics include J* drift and medium parameter delta mean/p95/count.

scripts/experiments/gmvc_v3_curasao_g000_ramp_sweep_1000.sh
- New wrapper for R100/R300/R500.
- It reuses gmvc_v3_curasao_medium_grad_sweep_1000.sh with VARIANT=G000 and only changes GMVC_RAMP_STEPS.
```

Experiment setup:

```text
scene=curasao
start=M1 step 10000
target step=11000
seed=42
medium_context_mode=dir_xy_camera
b_inf_mode=tied
lambda_gmvc_profile=40
lambda_gmvc_object=0.004
gmvc_v3_target_current_camera_tracks=True
gmvc_v3_object_phase_medium_grad_scale=0.00
gmvc_grad_log_every=49
fixed Eval-F bank=renders/gmvc_v2_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt
fixed Eval-G bank=renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt
```

Commands:

```text
CUDA_VISIBLE_DEVICES=8 VARIANT=R100 GPU=8 STAMP=20260804_gmvc_v3_curasao_g000_ramp_sweep_1000_log49 GMVC_GRAD_LOG_EVERY=49 RUN_EVAL=1 scripts/experiments/gmvc_v3_curasao_g000_ramp_sweep_1000.sh
CUDA_VISIBLE_DEVICES=8 VARIANT=R300 GPU=8 STAMP=20260804_gmvc_v3_curasao_g000_ramp_sweep_1000_log49 GMVC_GRAD_LOG_EVERY=49 RUN_EVAL=1 scripts/experiments/gmvc_v3_curasao_g000_ramp_sweep_1000.sh
CUDA_VISIBLE_DEVICES=8 VARIANT=R500 GPU=8 STAMP=20260804_gmvc_v3_curasao_g000_ramp_sweep_1000_log49 GMVC_GRAD_LOG_EVERY=49 RUN_EVAL=1 scripts/experiments/gmvc_v3_curasao_g000_ramp_sweep_1000.sh

/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_fixed_bank.py \
  --load-config <run config.yml> \
  --load-step 11000 \
  --track-bank renders/gmvc_v2_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt \
  --max-tracks 30000 \
  --output-dir renders/gmvc_fixed_bank_diag_20260804/curasao_g000_ramp_log49/<run>/evalf

/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_fixed_bank.py \
  --load-config <run config.yml> \
  --load-step 11000 \
  --track-bank renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt \
  --max-tracks 30000 \
  --output-dir renders/gmvc_fixed_bank_diag_20260804/curasao_g000_ramp_log49/<run>/evalg
```

Notes:

```text
The first R300 run before cleanup wrote a corrupt checkpoint because the filesystem was full.
After deleting outdated outputs/renders, R300 was rerun successfully and produced a valid step-11000 checkpoint and eval JSON.
Summary JSON: renders/gmvc_fixed_bank_diag_20260804/curasao_g000_ramp_log49/summary.json
```

RGB metrics:

| Run | Ramp | PSNR | dPSNR vs A0 | SSIM | dSSIM vs A0 | LPIPS | dLPIPS vs A0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | n/a | 32.9690 | +0.0000 | 0.957966 | +0.000000 | 0.106611 | +0.000000 |
| prior G000 | 100 | 32.8183 | -0.1507 | 0.957343 | -0.000623 | 0.106901 | +0.000289 |
| R100 | 100 | 32.8147 | -0.1543 | 0.957334 | -0.000632 | 0.106908 | +0.000296 |
| R300 | 300 | 32.8349 | -0.1342 | 0.957400 | -0.000566 | 0.106868 | +0.000257 |
| R500 | 500 | 32.8627 | -0.1063 | 0.957492 | -0.000474 | 0.106813 | +0.000202 |

Fixed-bank metrics, percent change versus A0:

Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| prior G000 | -2.00% | -6.54% | -1.40% | -2.05% | -2.80% | -1.30% |
| R100 | -2.00% | -6.51% | -1.38% | -1.98% | -2.61% | -1.27% |
| R300 | -1.87% | -6.27% | -1.24% | -2.23% | -2.86% | -1.35% |
| R500 | -1.70% | -5.87% | -1.05% | -2.35% | -2.63% | -1.34% |

Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| prior G000 | -2.02% | -7.53% | -1.04% | -2.95% | -2.82% | -1.18% |
| R100 | -2.02% | -7.51% | -1.02% | -2.93% | -2.89% | -1.17% |
| R300 | -1.89% | -7.27% | -0.89% | -3.13% | -2.85% | -1.22% |
| R500 | -1.73% | -6.94% | -0.73% | -3.30% | -2.67% | -1.26% |

Gradient route and drift diagnostics:

| Run | Log rows | Object rows | Object RGB medium grad mean | Object RGB geometry grad mean | Object RGB SH-rest grad mean | Object aux DC grad mean | Last J* drift mean | Last attn delta | Last bs delta | Last transmission delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R100 | 20 | 4 | 0.000000 | 0.418124 | 0.00259657 | 0.0000130146 | 0.016570 | 0.006972 | 0.004551 | 0.004177 |
| R300 | 20 | 4 | 0.000000 | 0.418279 | 0.00258426 | 0.0000109282 | 0.016517 | 0.006984 | 0.004495 | 0.004186 |
| R500 | 20 | 4 | 0.000000 | 0.418534 | 0.00257224 | 0.0000094229 | 0.016477 | 0.006969 | 0.004416 | 0.004176 |

Interpretation:

```text
The implementation now passes the stricter G000 route audit:
- object-phase RGB medium gradient is exactly 0.0 in logged object rows;
- object-phase RGB still reaches geometry, opacity, and SH-rest;
- object auxiliary loss still only produces DC gradient and does not update medium, geometry, opacity, or SH-rest.

Longer ramp improves RGB safety monotonically in this small sweep.
R500 recovers the PSNR safety margin relative to the previous G000 endpoint: dPSNR improves from -0.1507 dB to -0.1063 dB versus A0.
However, the decoupling signal weakens slightly as ramp length increases: Eval-F transfer/J-var improvements move from -2.00%/-6.54% to -1.70%/-5.87%; Eval-G moves from -2.02%/-7.53% to -1.73%/-6.94%.
This is a real internal Pareto direction, but not a free improvement.
```

Updated decision:

```text
进入 15k: No.
扩展场景: No.
Best short-run candidate: R500, because it keeps meaningful fixed-bank transfer/J-var/DC-var gains while restoring RGB safety.
Mechanism conclusion: the bottleneck is partly timing/target instability, not only steady-state direct competition. A slower ramp lets RGB medium adaptation recover part of the lost PSNR while preserving most of the decoupling signal.
Next minimal experiment, if continuing GMVC-V3: object-lambda sweep around R500, not more gradient-scale tuning.
Suggested matrix: R500-L002, R500-L003, R500-L004 current, R500-L006, same fixed Eval-F/G and RGB eval.
```

## 27. R500 object-lambda sweep

This section follows the R500 ramp result with the smallest object-loss strength test. It keeps the successful R500 timing and freezes the object-phase RGB medium path exactly as before. The only intended training change is `lambda_gmvc_object`.

Implementation note:

```text
water_splatting/water_splatting.py
- GMVC gradient JSONL now also records:
  gmvc_profile_j_star_mean
  gmvc_profile_j_star_p05
  gmvc_profile_j_star_p95

scripts/experiments/gmvc_v3_curasao_r500_object_lambda_sweep_1000.sh
- New wrapper for O004/O003/O002.
- It reuses the R500 G000 ramp wrapper and only changes LAMBDA_GMVC_OBJECT.
```

Experiment setup:

```text
scene=curasao
start=M1 step 10000
target step=11000
seed=42
medium_context_mode=dir_xy_camera
b_inf_mode=tied
lambda_gmvc_profile=40
gmvc_ramp_steps=500
gmvc_v3_object_phase_medium_grad_scale=0.00
gmvc_v3_target_current_camera_tracks=True
gmvc_grad_log_every=49
fixed Eval-F bank=renders/gmvc_v2_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt
fixed Eval-G bank=renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt
```

Runs:

| Run | Object lambda |
|---|---:|
| O004 | 0.004 |
| O003 | 0.003 |
| O002 | 0.002 |

Commands:

```text
CUDA_VISIBLE_DEVICES=7 VARIANT=O004 GPU=7 STAMP=20260804_gmvc_v3_curasao_r500_object_lambda_sweep_1000_jstar_log49 GMVC_GRAD_LOG_EVERY=49 RUN_EVAL=1 scripts/experiments/gmvc_v3_curasao_r500_object_lambda_sweep_1000.sh
CUDA_VISIBLE_DEVICES=8 VARIANT=O003 GPU=8 STAMP=20260804_gmvc_v3_curasao_r500_object_lambda_sweep_1000_jstar_log49 GMVC_GRAD_LOG_EVERY=49 RUN_EVAL=1 scripts/experiments/gmvc_v3_curasao_r500_object_lambda_sweep_1000.sh
CUDA_VISIBLE_DEVICES=9 VARIANT=O002 GPU=9 STAMP=20260804_gmvc_v3_curasao_r500_object_lambda_sweep_1000_jstar_log49 GMVC_GRAD_LOG_EVERY=49 RUN_EVAL=1 scripts/experiments/gmvc_v3_curasao_r500_object_lambda_sweep_1000.sh
```

Summary JSON:

```text
renders/gmvc_fixed_bank_diag_20260804/curasao_r500_object_lambda_jstar_log49/summary.json
```

RGB metrics:

| Run | Object lambda | PSNR | dPSNR vs A0 | dPSNR vs O004 | SSIM | dSSIM vs A0 | LPIPS | dLPIPS vs A0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | n/a | 32.9690 | +0.0000 | +0.1076 | 0.957966 | +0.000000 | 0.106611 | +0.000000 |
| O004 | 0.004 | 32.8614 | -0.1076 | +0.0000 | 0.957483 | -0.000482 | 0.106860 | +0.000248 |
| O003 | 0.003 | 32.8630 | -0.1061 | +0.0016 | 0.957490 | -0.000476 | 0.106853 | +0.000242 |
| O002 | 0.002 | 32.8587 | -0.1103 | -0.0027 | 0.957471 | -0.000495 | 0.106789 | +0.000178 |

Fixed-bank metrics, percent change versus A0:

Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| O004 | -1.69% | -5.87% | -1.06% | -2.33% | -2.70% | -1.34% |
| O003 | -1.70% | -5.95% | -1.04% | -2.43% | -2.67% | -1.38% |
| O002 | -1.70% | -5.91% | -1.05% | -2.39% | -2.68% | -1.37% |

Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| O004 | -1.72% | -6.93% | -0.74% | -3.26% | -2.66% | -1.24% |
| O003 | -1.73% | -7.05% | -0.70% | -3.30% | -2.80% | -1.27% |
| O002 | -1.72% | -6.98% | -0.72% | -3.19% | -2.75% | -1.22% |

Fixed-bank metrics, percent change versus O004:

Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| O003 | -0.00% | -0.09% | +0.02% | -0.10% | +0.04% | -0.05% |
| O002 | -0.01% | -0.04% | +0.01% | -0.07% | +0.03% | -0.04% |

Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| O003 | -0.00% | -0.14% | +0.03% | -0.05% | -0.14% | -0.04% |
| O002 | -0.00% | -0.06% | +0.01% | +0.07% | -0.09% | +0.01% |

Mechanism diagnostics:

| Run | Rows | Object rows | Object RGB medium grad | Object aux DC grad | Object/RGB DC ratio mean | Object/RGB DC ratio max | Drift early | Drift middle | Drift late | Last J* mean | Last J* p05 | Last J* p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O004 | 20 | 4 | 0.000000 | 0.000009425 | 0.014129 | 0.019907 | 0.021896 | 0.022126 | 0.020860 | 0.485173 | 0.048079 | 1.585536 |
| O003 | 20 | 4 | 0.000000 | 0.000007065 | 0.010599 | 0.014975 | 0.021910 | 0.022228 | 0.020911 | 0.484760 | 0.048016 | 1.581720 |
| O002 | 20 | 4 | 0.000000 | 0.000004709 | 0.007055 | 0.009968 | 0.021886 | 0.022193 | 0.020879 | 0.484952 | 0.048004 | 1.584424 |

Interpretation:

```text
The object lambda implementation behaves correctly:
- object-phase RGB medium gradient stays exactly 0.0;
- object auxiliary DC gradient and object/RGB-DC ratio decrease roughly proportionally with lambda;
- valid object track coverage is stable across runs.

The experiment does not find a meaningful object-lambda Pareto improvement.
O003 is numerically the best PSNR point, but the gain over O004 is only +0.0016 dB, far below the +0.01 dB local gate.
O002 improves LPIPS slightly but loses PSNR and SSIM.
Fixed-bank metrics are effectively flat across O004/O003/O002, with all relative changes versus O004 within about 0.15%.
J* drift and J* distribution are also effectively unchanged.

Lowering object lambda therefore does not explain the remaining RGB cost under R500.
The remaining difference from A0 is more likely tied to profile medium calibration or to the shared profile/object target construction than to object auxiliary strength.
```

Updated decision:

```text
进入 15k: No.
扩展场景: No.
Best candidate remains R500/O004 or O003-equivalent; O003 is not a statistically meaningful upgrade.
Do not continue lowering object lambda.
Next single-factor experiment, if continuing GMVC-V3: reduce profile lambda from 40 to 35/30 under the same R500/O004 setup, because object-lambda strength is not the active bottleneck.
```

## 28. R500 auxiliary-off control and profile-lambda sweep

This section first adds the missing causal control requested after the object-lambda sweep: keep the R500 object phase, 4:1 schedule, strict object-phase medium freeze, and profile objective unchanged, but set `lambda_gmvc_object=0.000`. This tests whether R500's fixed-bank gains require the explicit object auxiliary loss, rather than only the phase schedule and medium freeze.

Implementation changes:

```text
scripts/experiments/gmvc_v3_curasao_r500_object_lambda_sweep_1000.sh
- Added O000 with LAMBDA_GMVC_OBJECT=0.000.

scripts/experiments/gmvc_v3_curasao_r500_profile_lambda_sweep_1000.sh
- New wrapper for P40/P35/P30.
- It reuses the R500 ramp wrapper, fixes object lambda at 0.004, and changes only LAMBDA_GMVC_PROFILE.
```

Auxiliary-off command:

```text
CUDA_VISIBLE_DEVICES=9 VARIANT=O000 GPU=9 STAMP=20260804_gmvc_v3_curasao_r500_object_lambda_sweep_1000_jstar_log49 GMVC_GRAD_LOG_EVERY=49 RUN_EVAL=1 scripts/experiments/gmvc_v3_curasao_r500_object_lambda_sweep_1000.sh
```

Profile sweep commands:

```text
CUDA_VISIBLE_DEVICES=8 VARIANT=P35 GPU=8 STAMP=20260804_gmvc_v3_curasao_r500_profile_lambda_sweep_1000_log49 GMVC_GRAD_LOG_EVERY=49 RUN_EVAL=1 scripts/experiments/gmvc_v3_curasao_r500_profile_lambda_sweep_1000.sh
CUDA_VISIBLE_DEVICES=9 VARIANT=P30 GPU=9 STAMP=20260804_gmvc_v3_curasao_r500_profile_lambda_sweep_1000_log49 GMVC_GRAD_LOG_EVERY=49 RUN_EVAL=1 scripts/experiments/gmvc_v3_curasao_r500_profile_lambda_sweep_1000.sh
```

Summary JSON:

```text
renders/gmvc_fixed_bank_diag_20260804/curasao_r500_object_lambda_jstar_log49/summary_with_o000.json
renders/gmvc_fixed_bank_diag_20260804/curasao_r500_profile_lambda_log49/summary.json
```

### 28.1 R500-O000 auxiliary-off control

RGB metrics:

| Run | Object lambda | PSNR | dPSNR vs A0 | dPSNR vs O004 | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| O004 | 0.004 | 32.8614 | -0.1076 | +0.0000 | 0.957483 | 0.106860 |
| O000 | 0.000 | 32.8614 | -0.1077 | -0.0001 | 0.957492 | 0.106874 |

Fixed-bank metrics, percent change versus A0:

Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| O004 | -1.69% | -5.87% | -1.06% | -2.33% | -2.70% | -1.34% |
| O000 | -1.69% | -5.76% | -0.97% | -2.04% | -2.85% | -1.15% |

Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| O004 | -1.72% | -6.93% | -0.74% | -3.26% | -2.66% | -1.24% |
| O000 | -1.72% | -6.86% | -0.66% | -2.56% | -2.98% | -0.98% |

O004 versus O000, percent change:

| Eval | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| Eval-F | +0.00% | -0.12% | -0.09% | -0.29% | +0.15% | -0.19% |
| Eval-G | +0.00% | -0.07% | -0.08% | -0.72% | +0.33% | -0.26% |

Mechanism diagnostics:

| Run | Object rows | Object RGB medium grad | Object aux DC grad | Object/RGB DC ratio mean | Drift early | Drift middle | Drift late | Last J* mean | Last J* p05 | Last J* p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O004 | 4 | 0.000000 | 0.000009425 | 0.014129 | 0.021896 | 0.022126 | 0.020860 | 0.485173 | 0.048079 | 1.585536 |
| O000 | 4 | 0.000000 | 0.000000000 | 0.000000 | 0.021859 | 0.022121 | 0.020941 | 0.484612 | 0.046806 | 1.583098 |

Interpretation:

```text
O000 confirms that the object phase and medium freeze are still present while explicit object auxiliary DC gradients are zero.
O004 and O000 have indistinguishable RGB metrics.
The explicit object auxiliary contributes weakly but detectably to object_target_l1 and dc_recomposition_l1, especially on Eval-G, while J-var/closure improvements are very small and DC-var slightly worsens versus O000.
Therefore R500's main fixed-bank signal is not caused primarily by the explicit object auxiliary. It comes mostly from the R500 profile/phase/medium-freeze setup, with object auxiliary acting as a small object-fit/recomposition stabilizer.
For the profile sweep, keep O004 as the cleaner continuation because it improves object-fit/recomposition without RGB cost, but do not claim the object auxiliary is the main source of the GMVC-V3 effect.
```

### 28.2 R500 profile-lambda sweep

RGB metrics:

| Run | Profile lambda | PSNR | dPSNR vs A0 | dPSNR vs P40 | SSIM | dSSIM vs A0 | LPIPS | dLPIPS vs A0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P40 | 40 | 32.8614 | -0.1076 | +0.0000 | 0.957483 | -0.000482 | 0.106860 | +0.000248 |
| P35 | 35 | 32.8925 | -0.0765 | +0.0311 | 0.957576 | -0.000390 | 0.106773 | +0.000161 |
| P30 | 30 | 32.9243 | -0.0448 | +0.0628 | 0.957664 | -0.000302 | 0.106695 | +0.000084 |

Fixed-bank metrics, percent change versus A0:

Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| P40 | -1.69% | -5.87% | -1.06% | -2.33% | -2.70% | -1.34% |
| P35 | -1.51% | -5.45% | -0.87% | -2.53% | -2.58% | -1.38% |
| P30 | -1.32% | -4.98% | -0.68% | -2.72% | -2.57% | -1.44% |

Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| P40 | -1.72% | -6.93% | -0.74% | -3.26% | -2.66% | -1.24% |
| P35 | -1.55% | -6.56% | -0.58% | -3.52% | -2.64% | -1.35% |
| P30 | -1.37% | -6.12% | -0.41% | -3.70% | -2.69% | -1.40% |

Fixed-bank metrics, percent change versus P40:

Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| P35 | +0.19% | +0.45% | +0.19% | -0.20% | +0.13% | -0.04% |
| P30 | +0.38% | +0.95% | +0.39% | -0.40% | +0.14% | -0.10% |

Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| P35 | +0.18% | +0.39% | +0.16% | -0.27% | +0.02% | -0.11% |
| P30 | +0.36% | +0.87% | +0.32% | -0.45% | -0.03% | -0.17% |

Mechanism diagnostics:

| Run | Profile lambda | Last profile grad norm | Last profile/RGB medium ratio | Drift early | Drift middle | Drift late | Last J* mean | Last J* p05 | Last J* p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P40 | 40 | 0.002655 | 0.014331 | 0.021896 | 0.022126 | 0.020860 | 0.485173 | 0.048079 | 1.585536 |
| P35 | 35 | 0.002374 | 0.012565 | 0.021906 | 0.022137 | 0.020922 | 0.485274 | 0.047985 | 1.583848 |
| P30 | 30 | 0.002085 | 0.010804 | 0.021906 | 0.022175 | 0.020925 | 0.485454 | 0.047666 | 1.584946 |

Interpretation:

```text
Profile lambda is the active Pareto control that object lambda was not.
Lowering profile lambda monotonically improves PSNR, SSIM, and LPIPS while retaining positive fixed-bank transfer/J-var/DC-var signals.
The RGB recovery is material: P30 improves PSNR by +0.0628 dB versus P40 and reduces the A0 PSNR gap to -0.0448 dB.
The cost is a controlled weakening of transfer/J-var/closure improvements, while object-fit and recomposition improve.
P30 retains around 78%-80% of the P40 transfer gain and more than 85% of the J-var gain, depending on Eval-F/G. P35 is the conservative retention point; P30 is the stronger RGB recovery point.
J* drift and J* distribution remain essentially unchanged across P40/P35/P30, so the profile-lambda effect is not explained by target-distribution drift.
```

Updated decision:

```text
进入 15k: No.
扩展场景: No.
Best short-run RGB candidate: P30.
Best conservative decoupling candidate: P35.
Current recommended next step: Curasao-only 3k persistence check for P30 and P35 against P40/A0, still no cross-scene expansion.
If only one candidate can be carried forward, prefer P30 because it restores most RGB loss while keeping fixed-bank metrics positive.
```

## 29. Curasao R500 profile 3k persistence

Date: 2026-08-05

Goal:

```text
Run the minimal Curasao-only persistence check before any 15k or cross-scene expansion.
All runs start from the same Curasao M1 step-10000 checkpoint and continue to step-13000.
Evaluate every run at step-11000, step-12000, and step-13000 against the same-step A0 continuation.
```

Code additions:

```text
scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_3000.sh
scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_eval.sh
scripts/diagnostics/evaluate_checkpoint_metrics.py
scripts/diagnostics/summarize_gmvc_persistence.py
```

Training matrix:

| Run | Profile lambda | GMVC setup |
|---|---:|---|
| A0 | 0 | M1 continuation, GMVC off |
| P40 | 40 | R500, O004, object-phase medium grad scale 0 |
| P35 | 35 | R500, O004, object-phase medium grad scale 0 |
| P30 | 30 | R500, O004, object-phase medium grad scale 0 |

Fixed conditions:

```text
scene=Curasao
start_checkpoint=outputs/cross_scene_curasao_m1_seed42_15000/.../step-000010000.ckpt
target_final_step=13000
max_num_iterations=3000
steps_per_save=1000
save_only_latest_checkpoint=False
object_lambda=0.004
ramp_steps=500
object_phase_medium_grad_scale=0.00
cycle=4 medium : 1 object
target_current_camera_tracks=True
train_bank=renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt
Eval-F bank=renders/gmvc_v2_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt
Eval-G bank=renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt
```

Commands:

```bash
GPU=6 VARIANT=A0 scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_3000.sh
GPU=7 VARIANT=P40 scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_3000.sh
GPU=8 VARIANT=P35 scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_3000.sh
GPU=9 VARIANT=P30 scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_3000.sh

GPU=6 VARIANTS=A0 RUN_SUMMARY=0 scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_eval.sh
GPU=7 VARIANTS=P40 RUN_SUMMARY=0 scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_eval.sh
GPU=8 VARIANTS=P35 RUN_SUMMARY=0 scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_eval.sh
GPU=9 VARIANTS=P30 RUN_SUMMARY=0 scripts/experiments/gmvc_v3_curasao_r500_profile_persistence_eval.sh

/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/summarize_gmvc_persistence.py \
  --root renders/gmvc_fixed_bank_diag_20260805/curasao_r500_profile_persistence_3k \
  --variants A0,P40,P35,P30 \
  --steps 11000,12000,13000 \
  --output renders/gmvc_fixed_bank_diag_20260805/curasao_r500_profile_persistence_3k/summary.json
```

Outputs:

```text
outputs/gmvc_v3_a0_profile_persistence3k_curasao_seed42_step10000_to_13000
outputs/gmvc_v3_r500_p40_profile_persistence3k_curasao_seed42_step10000_to_13000
outputs/gmvc_v3_r500_p35_profile_persistence3k_curasao_seed42_step10000_to_13000
outputs/gmvc_v3_r500_p30_profile_persistence3k_curasao_seed42_step10000_to_13000
renders/gmvc_fixed_bank_diag_20260805/curasao_r500_profile_persistence_3k/summary.json
```

All four runs saved step-11000, step-12000, and step-13000 checkpoints.

RGB metrics:

| Step | Run | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 11000 | A0 | 32.9731 | +0.0000 | 0.957970 | +0.000000 | 0.106627 | +0.000000 |
| 11000 | P40 | 32.8651 | -0.1080 | 0.957492 | -0.000478 | 0.106849 | +0.000222 |
| 11000 | P35 | 32.8911 | -0.0819 | 0.957577 | -0.000393 | 0.106827 | +0.000199 |
| 11000 | P30 | 32.9229 | -0.0502 | 0.957666 | -0.000304 | 0.106736 | +0.000109 |
| 12000 | A0 | 32.1031 | +0.0000 | 0.957241 | +0.000000 | 0.108162 | +0.000000 |
| 12000 | P40 | 32.0497 | -0.0534 | 0.956766 | -0.000475 | 0.107706 | -0.000456 |
| 12000 | P35 | 32.0667 | -0.0365 | 0.956841 | -0.000399 | 0.107632 | -0.000530 |
| 12000 | P30 | 32.0832 | -0.0199 | 0.956913 | -0.000327 | 0.107652 | -0.000510 |
| 13000 | A0 | 32.2263 | +0.0000 | 0.956923 | +0.000000 | 0.107986 | +0.000000 |
| 13000 | P40 | 32.2513 | +0.0250 | 0.955700 | -0.001223 | 0.108110 | +0.000124 |
| 13000 | P35 | 32.2522 | +0.0259 | 0.955864 | -0.001059 | 0.108055 | +0.000069 |
| 13000 | P30 | 32.2673 | +0.0410 | 0.956042 | -0.000880 | 0.107933 | -0.000054 |

Fixed-bank percent change versus same-step A0. Lower is better for all listed fixed-bank metrics.

Step 11000 Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| P40 | -1.70% | -5.85% | -1.08% | -2.26% | -2.58% | -1.32% |
| P35 | -1.52% | -5.42% | -0.89% | -2.42% | -2.49% | -1.33% |
| P30 | -1.32% | -4.90% | -0.71% | -2.54% | -2.43% | -1.34% |

Step 11000 Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| P40 | -1.73% | -6.91% | -0.75% | -3.07% | -2.62% | -1.12% |
| P35 | -1.55% | -6.52% | -0.59% | -3.31% | -2.40% | -1.21% |
| P30 | -1.37% | -6.02% | -0.45% | -3.44% | -2.35% | -1.26% |

Step 12000 Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| P40 | -2.16% | -8.78% | -1.62% | -3.05% | -2.62% | -1.58% |
| P35 | -1.95% | -8.23% | -1.46% | -2.90% | -2.54% | -1.42% |
| P30 | -1.73% | -7.57% | -1.31% | -2.67% | -2.43% | -1.23% |

Step 12000 Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| P40 | -2.15% | -10.68% | -0.95% | -4.86% | -3.24% | -1.96% |
| P35 | -1.92% | -9.94% | -0.82% | -4.57% | -3.13% | -1.80% |
| P30 | -1.67% | -9.11% | -0.70% | -4.29% | -3.07% | -1.62% |

Step 13000 Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| P40 | -2.77% | -13.44% | -1.06% | -13.13% | -3.32% | -7.84% |
| P35 | -2.54% | -12.39% | -1.13% | -11.54% | -2.99% | -6.84% |
| P30 | -2.25% | -11.48% | -1.07% | -10.44% | -2.71% | -6.11% |

Step 13000 Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| P40 | -2.80% | -15.13% | -0.38% | -15.17% | -4.10% | -8.43% |
| P35 | -2.54% | -13.80% | -0.48% | -13.48% | -3.77% | -7.41% |
| P30 | -2.23% | -12.68% | -0.45% | -12.40% | -3.58% | -6.68% |

Step-13000 absolute fixed-bank metrics:

Eval-F:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.021120 | 0.006574 | 0.101228 | 0.073453 | 0.001890 | 0.022649 |
| P40 | 0.020535 | 0.005691 | 0.100153 | 0.063808 | 0.001827 | 0.020874 |
| P35 | 0.020582 | 0.005760 | 0.100085 | 0.064978 | 0.001833 | 0.021100 |
| P30 | 0.020644 | 0.005820 | 0.100142 | 0.065783 | 0.001839 | 0.021264 |

Eval-G:

| Run | Transfer | J-var | Closure | Obj-fit | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.020381 | 0.012368 | 0.088419 | 0.095129 | 0.002026 | 0.024620 |
| P40 | 0.019810 | 0.010497 | 0.088081 | 0.080702 | 0.001943 | 0.022544 |
| P35 | 0.019863 | 0.010661 | 0.087994 | 0.082304 | 0.001950 | 0.022796 |
| P30 | 0.019927 | 0.010800 | 0.088019 | 0.083338 | 0.001954 | 0.022975 |

Gradient log facts:

| Run | Rows | Last logged step | Last profile lambda | Last profile grad norm | Last profile/RGB medium ratio | Object grad DC mean | Object/RGB DC mean | Late J* drift mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P40 | 61 | 12985 | 40 | 0.003829 | 0.004783 | 0.0000119 | 0.016854 | 0.017188 |
| P35 | 61 | 12985 | 35 | 0.003315 | 0.004153 | 0.0000119 | 0.016848 | 0.017338 |
| P30 | 61 | 12985 | 30 | 0.002843 | 0.003586 | 0.0000119 | 0.016882 | 0.017325 |

Interpretation:

```text
The 3k persistence result supports the profile signal as persistent on Curasao.
Fixed Eval-F and Eval-G improvements do not decay after step-11000. They become stronger by step-13000 for transfer, J-var, object-fit, DC-var, and recomposition.
P40 remains the strongest decoupling configuration, but it has worse RGB tradeoff.
P30 is the best RGB/Pareto point at step-13000: +0.0410 dB PSNR versus A0 and slightly better LPIPS, while retaining positive fixed-bank improvements on both banks.
The remaining problem is SSIM. P30 still drops -0.000880 SSIM versus same-step A0 at step-13000, while P35 and P40 drop even more.
Therefore the persistence test passes the decoupling-signal requirement, but it is not a clean RGB safety pass.
```

Gate decision:

```text
进入 15k: No.
扩展场景: No.
Best current candidate: P30.
Reason: P30 has the best RGB tradeoff and retains persistent fixed-bank transfer/J-var/DC-var/object-fit/recomposition gains.
Blocking issue: same-step SSIM drop remains too large for a clean safety claim.
Next step should stay Curasao-only and single-factor. Do not tune profile lambda further yet. First identify whether the step-13000 SSIM drop is from eval-view texture/edge degradation, Gaussian appearance drift, or medium over-calibration.
Recommended next diagnostic: generate Curasao step-13000 contact sheets/residual maps for A0 and P30 on the three eval views, plus per-view RGB metrics, before deciding any 15k run.
```

## 30. Curasao P30 per-view diagnosis and 13k to 15k continuation

Date: 2026-08-05

Correction to the previous gate:

```text
The step-13000 P30 SSIM delta is -0.000880 versus same-step A0.
This is below the predefined safety limit of -0.0015, so it should not block a Curasao-only 15k continuation.
The correct 13k decision is: run the per-view residual diagnosis, then continue only A0 and P30 from the matched step-13000 checkpoints to step-15000.
No cross-scene expansion is allowed before the 15k Curasao RGB and fixed-bank gates are checked.
```

Code additions:

```text
scripts/diagnostics/diagnose_gmvc_per_view_residuals.py
scripts/experiments/gmvc_v3_curasao_p30_13k_to_15k.sh
scripts/experiments/gmvc_v3_curasao_p30_15k_eval.sh
```

The continuation wrapper uses Nerfstudio `--load-checkpoint` through `LOAD_CHECKPOINT`, not model-only loading. Nerfstudio `Trainer._load_checkpoint()` restores the pipeline, optimizers, schedulers when enabled, and grad scaler. The P30 wrapper also pins the train bank to:

```text
renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt
```

This avoids accidentally constructing or loading a `curasao_m1_step13000` bank when `LOAD_STEP=13000`.

13k per-view diagnostic command:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_per_view_residuals.py \
  --a0-config outputs/gmvc_v3_a0_profile_persistence3k_curasao_seed42_step10000_to_13000/water-splatting/gmvc_v3_a0_profile_persistence3k_curasao_seed42_step10000_to_13000_20260805_gmvc_v3_curasao_r500_profile_persistence_3k_a0/config.yml \
  --a0-step 13000 \
  --p30-config outputs/gmvc_v3_r500_p30_profile_persistence3k_curasao_seed42_step10000_to_13000/water-splatting/gmvc_v3_r500_p30_profile_persistence3k_curasao_seed42_step10000_to_13000_20260805_gmvc_v3_curasao_r500_profile_persistence_3k_p30_p30_r500_g000/config.yml \
  --p30-step 13000 \
  --test-mode test \
  --output-dir renders/gmvc_per_view_residuals_20260805/curasao_a0_vs_p30_step13000
```

13k per-view summary:

| View | Image | dPSNR | dSSIM | dLPIPS | dRGB L1 | dLuma L1 | dChroma L1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | MTN_1288.png | -0.6093 | -0.003634 | +0.001219 | +0.003249 | +0.003817 | -0.000293 |
| 1 | MTN_1296.png | +0.4687 | +0.000574 | -0.000732 | -0.000690 | -0.000971 | +0.000175 |
| 2 | MTN_1304.png | +0.2635 | +0.000419 | -0.000648 | -0.000426 | -0.000628 | -0.000248 |
| Mean | all | +0.0410 | -0.000880 | -0.000054 | +0.000711 | +0.000739 | -0.000122 |

13k interpretation:

```text
The 13k SSIM drop is dominated by view 0, while views 1 and 2 improve in PSNR, SSIM, and LPIPS.
This supports the updated decision that P30 was safe enough for Curasao-only 15k.
The diagnostic output is:
renders/gmvc_per_view_residuals_20260805/curasao_a0_vs_p30_step13000/per_view_residual_summary.json
```

15k continuation commands:

```bash
GPU=6 VARIANT=A0 scripts/experiments/gmvc_v3_curasao_p30_13k_to_15k.sh
GPU=9 VARIANT=P30 scripts/experiments/gmvc_v3_curasao_p30_13k_to_15k.sh

GPU=6 VARIANTS=A0 RUN_SUMMARY=0 scripts/experiments/gmvc_v3_curasao_p30_15k_eval.sh
GPU=9 VARIANTS=P30 RUN_SUMMARY=0 scripts/experiments/gmvc_v3_curasao_p30_15k_eval.sh

/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/summarize_gmvc_persistence.py \
  --root renders/gmvc_fixed_bank_diag_20260805/curasao_p30_15k \
  --variants A0,P30 \
  --steps 14000,15000 \
  --output renders/gmvc_fixed_bank_diag_20260805/curasao_p30_15k/summary.json
```

Continuation outputs:

```text
outputs/gmvc_v3_a0_15k_curasao_seed42_step13000_to_15000
outputs/gmvc_v3_r500_p30_15k_curasao_seed42_step13000_to_15000
renders/gmvc_fixed_bank_diag_20260805/curasao_p30_15k/summary.json
```

Both runs saved:

```text
step-000014000.ckpt
step-000015000.ckpt
```

RGB metrics:

| Step | Run | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS |
|---:|---|---:|---:|---:|---:|---:|---:|
| 14000 | A0 | 32.3610 | +0.0000 | 0.955977 | +0.000000 | 0.108265 | +0.000000 |
| 14000 | P30 | 32.0915 | -0.2695 | 0.955008 | -0.000969 | 0.108525 | +0.000260 |
| 15000 | A0 | 32.1818 | +0.0000 | 0.955928 | +0.000000 | 0.107999 | +0.000000 |
| 15000 | P30 | 31.9670 | -0.2148 | 0.955133 | -0.000795 | 0.108302 | +0.000303 |

Fixed-bank percent change versus same-step A0. Lower is better for all listed fixed-bank metrics.

Step 14000:

| Eval | Transfer | J-var | Closure | Obj-target | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| Eval-F | -2.64% | -10.19% | -0.94% | -2.77% | -4.20% | -1.54% |
| Eval-G | -2.80% | -11.92% | -0.51% | -4.81% | -4.67% | -1.56% |

Step 15000:

| Eval | Transfer | J-var | Closure | Obj-target | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|
| Eval-F | -2.23% | -12.16% | -0.78% | -7.07% | -5.30% | -3.40% |
| Eval-G | -2.61% | -15.80% | -0.06% | -9.87% | -6.27% | -4.08% |

15k per-view diagnostic command:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_per_view_residuals.py \
  --a0-config outputs/gmvc_v3_a0_15k_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_a0_15k_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_curasao_p30_13k_to_15k_a0/config.yml \
  --a0-step 15000 \
  --p30-config outputs/gmvc_v3_r500_p30_15k_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_r500_p30_15k_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_curasao_p30_13k_to_15k_p30_p30_r500_g000/config.yml \
  --p30-step 15000 \
  --test-mode test \
  --output-dir renders/gmvc_per_view_residuals_20260805/curasao_a0_vs_p30_step15000
```

15k per-view summary:

| View | Image | dPSNR | dSSIM | dLPIPS | dRGB L1 | dLuma L1 | dChroma L1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | MTN_1288.png | -0.3133 | -0.001990 | +0.001005 | +0.002250 | +0.002643 | +0.000118 |
| 1 | MTN_1296.png | -0.2558 | -0.000211 | -0.000230 | +0.000371 | +0.000382 | -0.000021 |
| 2 | MTN_1304.png | -0.0752 | -0.000183 | +0.000134 | +0.000223 | +0.000216 | -0.000015 |
| Mean | all | -0.2148 | -0.000795 | +0.000303 | +0.000948 | +0.001080 | +0.000027 |

15k interpretation:

```text
P30 preserves the fixed-bank decoupling signal through step-15000.
J-var improves strongly on both Eval-F and Eval-G, and transfer remains positive but limited to roughly 2-3%.
The signal does not collapse from 14k to 15k; however, the RGB penalty persists.
The 15k P30 run fails the RGB safety gate because dPSNR=-0.2148 dB, which is worse than the -0.15 dB limit.
SSIM and LPIPS remain within the predefined safety limits, but PSNR is enough to block cross-scene expansion.
The 15k PSNR drop is no longer a single-view issue: all three eval views lose PSNR, with view 0 still the largest SSIM contributor.
The increase in RGB L1 is mostly luminance L1; chroma L1 is nearly unchanged.
```

Gate decision:

```text
进入四场景或跨场景扩展: No.
15k success claim: No.
Best Curasao candidate: P30 at step-13000, not step-15000.
Reason: step-13000 passes RGB safety and has persistent fixed-bank gains; step-15000 keeps decoupling gains but fails PSNR safety.
Current conclusion: GMVC profile calibration has a real Curasao medium-decoupling signal, but the current 15k schedule over-trades RGB fidelity for calibration.
Next action should remain Curasao-only and single-factor. Prefer testing profile stop/decay or best-checkpoint selection around 13k before any scene expansion. Do not increase profile lambda.
```

## 31. Curasao P30 profile release sweep from 13k to 15k

Date: 2026-08-05

### Code facts

Implemented a single-factor profile release schedule for GMVC-V3:

```text
gmvc_v3_profile_schedule: constant | stop | linear_decay
gmvc_v3_profile_decay_start_step: 13000
gmvc_v3_profile_decay_end_step: 14000
gmvc_v3_profile_decay_final_scale: 0.0
```

The schedule only scales the effective profile loss weight. It does not change medium forward values, renderer equations, densification, refinement, track thresholds, IRLS delta, object lambda, object ramp, or the 4:1 medium/object phase.

Additional JSONL audit fields:

```text
global_step
gmvc_phase
gmvc_profile_lambda_configured
gmvc_profile_lambda_scheduled
gmvc_profile_lambda_effective
gmvc_object_ramp_factor
gmvc_object_phase_medium_grad_scale
learning_rate
learning_rate_medium_mlp
learning_rate_direction_encoding
learning_rate_features_dc
grad_scaler_scale
```

The per-view residual diagnostic now supports multiple runs through:

```text
--run LABEL=config:step
--reference-run C30
```

New scripts:

```text
scripts/experiments/gmvc_v3_curasao_p30_profile_release_13k_to_15k.sh
scripts/experiments/gmvc_v3_curasao_p30_profile_release_eval.sh
```

Updated scripts:

```text
water_splatting/water_splatting.py
water_splatting/medium_calibration/gmvc_training.py
scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh
scripts/experiments/gmvc_phase_b_common.sh
scripts/experiments/gmvc_v3_alternating_1000.sh
scripts/diagnostics/diagnose_gmvc_per_view_residuals.py
scripts/diagnostics/summarize_gmvc_persistence.py
```

### Experiment facts

All four runs were rerun under the new code. A0 was rerun because common training code changed.

Training commands:

```bash
GPU=6 VARIANT=A0 scripts/experiments/gmvc_v3_curasao_p30_profile_release_13k_to_15k.sh
GPU=7 VARIANT=C30 scripts/experiments/gmvc_v3_curasao_p30_profile_release_13k_to_15k.sh
GPU=8 VARIANT=STOP scripts/experiments/gmvc_v3_curasao_p30_profile_release_13k_to_15k.sh
GPU=9 VARIANT=DECAY scripts/experiments/gmvc_v3_curasao_p30_profile_release_13k_to_15k.sh
```

Evaluation command:

```bash
GPU=6 scripts/experiments/gmvc_v3_curasao_p30_profile_release_eval.sh
```

Per-view command:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_gmvc_per_view_residuals.py \
  --a0-config outputs/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_a0/config.yml \
  --a0-step 15000 \
  --run C30=outputs/gmvc_v3_p30_release_c30_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_c30_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_c30/config.yml:15000 \
  --run STOP=outputs/gmvc_v3_p30_release_stop_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_stop_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_stop/config.yml:15000 \
  --run DECAY=outputs/gmvc_v3_p30_release_decay_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_decay_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_decay/config.yml:15000 \
  --reference-run C30 \
  --test-mode test \
  --output-dir renders/gmvc_per_view_residuals_20260805/curasao_p30_profile_release_step15000
```

Main outputs:

```text
renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k/summary.json
renders/gmvc_per_view_residuals_20260805/curasao_p30_profile_release_step15000/per_view_residual_summary.json
```

Each run saved:

```text
step-000014000.ckpt
step-000015000.ckpt
```

### Resume audit

Nerfstudio resume state:

```text
Trainer._load_checkpoint sets _start_step = checkpoint step + 1.
All release runs restored from step-000013000.ckpt and started training at global step 13001.
Optimizer, scheduler, and scaler state are restored by Nerfstudio load-checkpoint.
```

Forced JSONL audit rows:

| Run | Step | Phase | Schedule | Config lambda | Scheduled lambda | Effective lambda | Object lambda | Object ramp | Medium grad scale | medium LR | DC LR | scaler |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C30 | 13001 | medium | constant | 30.0 | 30.00 | 30.00 | 0.000 | 1.0 | 0.0 | 0.0001931 | 0.0025 | 1.0 |
| C30 | 13004 | object | constant | 30.0 | 30.00 | 0.00 | 0.004 | 1.0 | 0.0 | 0.0001931 | 0.0025 | 1.0 |
| STOP | 13001 | medium | stop | 30.0 | 0.00 | 0.00 | 0.000 | 1.0 | 0.0 | 0.0001931 | 0.0025 | 1.0 |
| STOP | 13004 | object | stop | 30.0 | 0.00 | 0.00 | 0.004 | 1.0 | 0.0 | 0.0001931 | 0.0025 | 1.0 |
| DECAY | 13001 | medium | linear_decay | 30.0 | 29.97 | 29.97 | 0.000 | 1.0 | 0.0 | 0.0001931 | 0.0025 | 1.0 |
| DECAY | 13004 | object | linear_decay | 30.0 | 29.88 | 0.00 | 0.004 | 1.0 | 0.0 | 0.0001931 | 0.0025 | 1.0 |
| DECAY | 14000 | medium | linear_decay | 30.0 | 0.00 | 0.00 | 0.000 | 1.0 | 0.0 | 0.0001702 | 0.0025 | 1.0 |

Audit conclusion:

```text
No local-step ramp reset was observed.
The 4:1 phase resumed from global step arithmetic: 13001 is medium phase, 13004 is object phase.
Object ramp factor is already 1.0 after resume.
Object phase medium RGB gradient scale is 0.0.
STOP removes profile gradient from the first resumed training step.
DECAY reaches zero effective profile weight at global step 14000.
```

### Config matrix

All runs use Curasao only, seed 42, the same geometry-only train bank, the same Eval-F/Eval-G banks, `object lambda=0.004`, 4:1 phase, `target_current_camera_tracks=True`, and `object-phase medium grad scale=0` for GMVC runs.

| Run | Source checkpoint | Profile schedule | Profile base lambda |
|---|---|---|---:|
| A0 | A0 step-13000 | GMVC off | 0 |
| C30 | P30 step-13000 | constant | 30 |
| STOP | P30 step-13000 | stop at 13000 | 30 |
| DECAY | P30 step-13000 | linear 13000 to 14000, final 0 | 30 |

### RGB metrics

| Step | Run | PSNR | dPSNR vs A0 | dPSNR vs C30 | dPSNR vs P30-13k | SSIM | dSSIM vs A0 | LPIPS | dLPIPS vs A0 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 14000 | A0 | 32.3633 | +0.0000 | +0.2683 | +0.0960 | 0.955980 | +0.000000 | 0.108264 | +0.000000 |
| 14000 | C30 | 32.0949 | -0.2683 | +0.0000 | -0.1723 | 0.955015 | -0.000965 | 0.108530 | +0.000266 |
| 14000 | STOP | 32.1442 | -0.2191 | +0.0492 | -0.1231 | 0.955490 | -0.000490 | 0.108237 | -0.000027 |
| 14000 | DECAY | 32.1120 | -0.2512 | +0.0171 | -0.1552 | 0.955209 | -0.000771 | 0.108437 | +0.000173 |
| 15000 | A0 | 32.1800 | +0.0000 | +0.2106 | -0.0873 | 0.955931 | +0.000000 | 0.108039 | +0.000000 |
| 15000 | C30 | 31.9695 | -0.2106 | +0.0000 | -0.2978 | 0.955143 | -0.000787 | 0.108258 | +0.000218 |
| 15000 | STOP | 32.0353 | -0.1447 | +0.0659 | -0.2319 | 0.955601 | -0.000330 | 0.108005 | -0.000034 |
| 15000 | DECAY | 32.0261 | -0.1539 | +0.0566 | -0.2412 | 0.955511 | -0.000420 | 0.108028 | -0.000011 |

RGB gate:

```text
C30 fails PSNR.
STOP passes PSNR/SSIM/LPIPS.
DECAY misses PSNR gate by about 0.0039 dB, while SSIM/LPIPS pass.
```

### Fixed-bank metrics versus same-step A0

Percent change. Lower is better.

Step 14000:

| Run | Eval | Transfer | J-var | Closure | Consensus-J | Obj-target | DC-var | Recomp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C30 | F | -2.64% | -10.17% | -0.95% | -2.69% | -2.75% | -4.24% | -1.52% |
| C30 | G | -2.80% | -11.89% | -0.52% | -3.02% | -4.78% | -4.66% | -1.56% |
| STOP | F | -1.78% | -7.58% | -0.58% | -1.79% | -0.93% | -3.76% | -0.22% |
| STOP | G | -1.80% | -8.04% | -0.28% | -1.90% | -2.60% | -4.25% | -0.31% |
| DECAY | F | -2.25% | -8.97% | -0.77% | -2.27% | -2.08% | -3.95% | -1.01% |
| DECAY | G | -2.29% | -9.83% | -0.40% | -2.41% | -3.85% | -4.42% | -1.05% |

Step 15000:

| Run | Eval | Transfer | J-var | Closure | Consensus-J | Obj-target | DC-var | Recomp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C30 | F | -2.23% | -12.15% | -0.78% | -2.26% | -7.10% | -5.35% | -3.42% |
| C30 | G | -2.61% | -15.78% | -0.06% | -2.88% | -9.89% | -6.21% | -4.11% |
| STOP | F | -1.15% | -9.40% | -0.24% | -1.15% | -5.80% | -5.05% | -2.25% |
| STOP | G | -1.33% | -11.70% | +0.39% | -1.44% | -8.41% | -5.61% | -3.04% |
| DECAY | F | -1.40% | -10.28% | -0.34% | -1.40% | -6.31% | -5.29% | -2.58% |
| DECAY | G | -1.61% | -12.83% | +0.34% | -1.74% | -8.97% | -5.95% | -3.38% |

Relative to C30 at step 15000:

| Run | Eval | Transfer | J-var | Closure | Consensus-J | Obj-target | DC-var | Recomp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| STOP | F | +1.10% | +3.12% | +0.54% | +1.14% | +1.41% | +0.32% | +1.22% |
| STOP | G | +1.31% | +4.84% | +0.45% | +1.48% | +1.65% | +0.65% | +1.11% |
| DECAY | F | +0.84% | +2.12% | +0.44% | +0.88% | +0.85% | +0.06% | +0.87% |
| DECAY | G | +1.02% | +3.50% | +0.40% | +1.17% | +1.03% | +0.29% | +0.76% |

Relative to P30 step-13000 at step 15000:

| Run | Eval | Transfer | J-var | Closure | Consensus-J | Obj-target | DC-var | Recomp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C30 | F | -1.20% | -3.46% | +0.20% | -1.14% | -2.49% | -0.44% | -1.78% |
| C30 | G | -1.34% | -5.33% | +0.40% | -1.31% | -2.14% | +0.26% | -1.16% |
| STOP | F | -0.11% | -0.45% | +0.75% | -0.01% | -1.12% | -0.13% | -0.58% |
| STOP | G | -0.05% | -0.75% | +0.86% | +0.14% | -0.53% | +0.91% | -0.06% |
| DECAY | F | -0.36% | -1.42% | +0.65% | -0.27% | -1.66% | -0.38% | -0.92% |
| DECAY | G | -0.33% | -2.02% | +0.80% | -0.16% | -1.14% | +0.55% | -0.40% |

### Gradient and medium-change diagnostics

Each GMVC run wrote 45 JSONL rows. C30, STOP, and DECAY all logged 35 medium-phase and 10 object-phase rows.

Mean JSONL values:

| Run | Interval | Effective profile lambda | Profile medium grad | Profile/RGB medium grad | RGB medium grad | J* drift | attn delta | bs delta | B_inf delta | transmission delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C30 | 13001-13999 | 23.18 | 0.002166 | 0.00730 | 0.3112 | 0.01455 | 0.00596 | 0.00410 | 0.00167 | 0.00361 |
| C30 | 14000-14999 | 23.18 | 0.002016 | 0.00689 | 0.3492 | 0.01899 | 0.00823 | 0.00472 | 0.00195 | 0.00507 |
| STOP | 13001-13999 | 0.00 | 0.000000 | 0.00000 | 0.3125 | 0.01453 | 0.00599 | 0.00404 | 0.00165 | 0.00362 |
| STOP | 14000-14999 | 0.00 | 0.000000 | 0.00000 | 0.3554 | 0.01879 | 0.00813 | 0.00457 | 0.00190 | 0.00498 |
| DECAY | 13001-13999 | 11.75 | 0.001153 | 0.00414 | 0.3110 | 0.01454 | 0.00597 | 0.00407 | 0.00166 | 0.00361 |
| DECAY | 14000-14999 | 0.00 | 0.000000 | 0.00000 | 0.3552 | 0.01883 | 0.00814 | 0.00462 | 0.00192 | 0.00500 |

The profile schedule affected the profile-to-medium gradient as intended. STOP makes it zero from step 13001. DECAY has a smaller profile gradient before 14k and zero after 14k. Medium delta statistics remain close across runs, so the RGB recovery is not explained by a large gross reduction in medium parameter motion.

### Per-view residual diagnosis at 15k

Mean deltas versus A0:

| Run | dPSNR | dSSIM | dLPIPS | dRGB L1 | dLuma L1 | dChroma L1 |
|---|---:|---:|---:|---:|---:|---:|
| C30 | -0.2106 | -0.000787 | +0.000218 | +0.000943 | +0.001079 | +0.000023 |
| STOP | -0.1447 | -0.000330 | -0.000034 | +0.000440 | +0.000488 | +0.000033 |
| DECAY | -0.1539 | -0.000420 | -0.000011 | +0.000530 | +0.000597 | +0.000032 |

STOP and DECAY versus C30:

| Run | dPSNR | dSSIM | dLPIPS | dRGB L1 | dLuma L1 | dChroma L1 |
|---|---:|---:|---:|---:|---:|---:|
| STOP | +0.0659 | +0.000457 | -0.000253 | -0.000503 | -0.000591 | +0.000009 |
| DECAY | +0.0566 | +0.000368 | -0.000229 | -0.000412 | -0.000482 | +0.000009 |

Per-view STOP versus C30:

| View | Image | dPSNR | dSSIM | dLPIPS | dRGB L1 | dLuma L1 | dChroma L1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | MTN_1288.png | +0.1746 | +0.001254 | -0.000720 | -0.001399 | -0.001617 | +0.000014 |
| 1 | MTN_1296.png | +0.0047 | +0.000041 | +0.000082 | -0.000035 | -0.000067 | +0.000018 |
| 2 | MTN_1304.png | +0.0184 | +0.000076 | -0.000120 | -0.000075 | -0.000088 | -0.000004 |

Per-view interpretation:

```text
STOP improves C30 on all three eval views, but most of the gain comes from view 0.
The improvement is mainly luminance residual reduction.
Chroma residual is nearly unchanged and slightly worse on average.
There is no evidence that STOP introduces a new single-view structural collapse.
```

### Gate decision

15k gate:

| Run | RGB gate | Transfer threshold | J-var threshold | DC/recomp/object | Closure | Decision |
|---|---|---|---|---|---|---|
| C30 | Fail PSNR | Pass | Pass | Pass | Mostly pass | Not usable due RGB |
| STOP | Pass | Fail, F=-1.15%, G=-1.33% | Pass | Pass | Eval-G worsens +0.39% | Not a final candidate |
| DECAY | Fail PSNR by 0.0039 dB | Fail, F=-1.40%, G=-1.61% | Pass | Pass | Eval-G worsens +0.34% | Not a final candidate |

### Reasonable inference

This sweep supports a narrower conclusion:

```text
Continued profile pressure after 13k is a real contributor to the 15k RGB penalty.
Turning it off recovers enough RGB for STOP to pass the image gate.
```

But the same sweep does not support zero-floor calibrate-then-release as the final method:

```text
STOP and DECAY lose too much transfer improvement.
Both fall below the 1.8% Eval-F/G transfer retention threshold.
Both also weaken closure on Eval-G.
```

The result is closest to case three from the plan:

```text
STOP restores RGB, but the main transfer component of decoupling is not sufficiently retained.
The calibrated medium decomposition is not fully self-maintaining after profile supervision is removed.
```

DECAY is not better than STOP in this zero-floor form. STOP has better RGB; DECAY retains slightly stronger transfer/J-var but still misses both the PSNR and transfer gates.

### Unverified hypotheses

The current evidence does not yet show whether a nonzero profile floor can keep transfer while preserving most RGB recovery. It also does not separate whether the transfer loss after STOP comes from medium drift, Gaussian compensation, or the 4:1 phase/object auxiliary continuing without profile support.

### Next decision

Do not enter cross-scene expansion, 15k success claim, longer training, or JapaneseGradens/IUI3/Panama.

The next single-factor experiment, if GMVC continues, should be:

```text
P30 from 13k to 15k with profile linear decay to a nonzero floor.
Suggested floors: 5 and 10.
Keep all other settings unchanged.
```

Do not increase the profile base lambda and do not tune object lambda, ramp, cycle, bank, thresholds, or renderer.

## 32. Curasao P30 H500-STOP profile timing test

Date: 2026-08-05

### Code facts

No renderer, densification, refinement, loss routing, object auxiliary, or GMVC core equation was changed in this round. The existing `gmvc_v3_profile_schedule=stop` implementation was reused with a later global stop step:

```text
gmvc_v3_profile_schedule=stop
gmvc_v3_profile_decay_start_step=13501
gmvc_v3_profile_decay_final_scale=0.0
```

This gives the intended H500-STOP behavior:

```text
step 13001-13500: effective profile lambda = 30
step 13501-15000: effective profile lambda = 0
```

The experiment wrappers now accept:

```text
VARIANT=H500
```

The H500 run used `STEPS_PER_SAVE=500`, so the saved checkpoints are:

```text
step-000013500.ckpt
step-000014000.ckpt
step-000014500.ckpt
step-000015000.ckpt
```

### Experiment facts

Training command:

```bash
GPU=6 VARIANT=H500 STEPS_PER_SAVE=500 \
  scripts/experiments/gmvc_v3_curasao_p30_profile_release_13k_to_15k.sh
```

Evaluation command:

```bash
GPU=6 VARIANTS=H500 STEPS=13500,14000,15000 RUN_SUMMARY=0 \
  scripts/experiments/gmvc_v3_curasao_p30_profile_release_eval.sh
```

Summary command:

```bash
/opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/summarize_gmvc_persistence.py \
  --root renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k \
  --variants A0,C30,STOP,DECAY,H500 \
  --steps 14000,15000 \
  --reference-variant DECAY \
  --start-root renders/gmvc_fixed_bank_diag_20260805/curasao_r500_profile_persistence_3k \
  --start-step 13000 \
  --start-variant P30 \
  --output renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k/summary_with_h500.json
```

Per-view command:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_gmvc_per_view_residuals.py \
  --a0-config outputs/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_a0/config.yml \
  --a0-step 15000 \
  --run C30=outputs/gmvc_v3_p30_release_c30_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_c30_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_c30/config.yml:15000 \
  --run STOP=outputs/gmvc_v3_p30_release_stop_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_stop_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_stop/config.yml:15000 \
  --run DECAY=outputs/gmvc_v3_p30_release_decay_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_decay_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_decay/config.yml:15000 \
  --run H500=outputs/gmvc_v3_p30_release_h500_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_h500_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_h500/config.yml:15000 \
  --reference-run DECAY \
  --test-mode test \
  --output-dir renders/gmvc_per_view_residuals_20260805/curasao_p30_profile_h500_step15000
```

Main outputs:

```text
renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k/summary_with_h500.json
renders/gmvc_per_view_residuals_20260805/curasao_p30_profile_h500_step15000/per_view_residual_summary.json
logs/gmvc_v3_p30_release_h500_20260805_gmvc_v3_p30_profile_release_13k_to_15k.jsonl
```

### H500 resume and schedule audit

Forced JSONL rows:

| Step | Phase | Config lambda | Scheduled lambda | Effective lambda | Schedule scale | Object lambda | Object ramp | Medium grad scale | medium LR | DC LR | scaler |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13001 | medium | 30.0 | 30.0 | 30.0 | 1.0 | 0.000 | 1.0 | 0.0 | 0.0001931 | 0.0025 | 1.0 |
| 13004 | object | 30.0 | 30.0 | 0.0 | 1.0 | 0.004 | 1.0 | 0.0 | 0.0001931 | 0.0025 | 1.0 |
| 13500 | medium | 30.0 | 30.0 | 30.0 | 1.0 | 0.000 | 1.0 | 0.0 | 0.0001813 | 0.0025 | 1.0 |
| 13501 | medium | 30.0 | 0.0 | 0.0 | 0.0 | 0.000 | 1.0 | 0.0 | 0.0001813 | 0.0025 | 1.0 |
| 14000 | medium | 30.0 | 0.0 | 0.0 | 0.0 | 0.000 | 1.0 | 0.0 | 0.0001702 | 0.0025 | 1.0 |
| 15000 | medium | 30.0 | 0.0 | 0.0 | 0.0 | 0.000 | 0.0 | 0.0 | 0.0001500 | 0.0025 | 1.0 |

Audit conclusion:

```text
H500 keeps profile active through step 13500 and removes profile gradient from step 13501 onward.
The run keeps the same global phase arithmetic, optimizer/scheduler/scaler restoration, object ramp, object lambda, and object-phase medium gradient scale as the previous release sweep.
```

### H500 13500 checkpoint

There are no A0/C30/STOP/DECAY step-13500 checkpoints from the prior matched-control sweep because those controls were saved at 1000-step intervals. To preserve the single-new-experiment constraint, controls were not rerun only to add 13500 checkpoints. Therefore step 13500 is used as an H500 internal transition diagnostic; formal matched comparisons remain at 14000 and 15000.

H500 absolute metrics at step 13500:

| Eval | PSNR | SSIM | LPIPS | Transfer | J-var | Closure | Consensus-J | Obj-target | DC-var | Recomp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RGB/F/G | 32.4999 | 0.956194 | 0.107571 | F 0.0206047 / G 0.0198591 | F 0.0058084 / G 0.0106750 | F 0.1010187 / G 0.0886265 | F 0.0141219 / G 0.0137341 | F 0.0622242 / G 0.0786250 | F 0.0018676 / G 0.0019827 | F 0.0205290 / G 0.0221621 |

### RGB metrics

| Step | Run | PSNR | dPSNR vs A0 | dPSNR vs DECAY | SSIM | dSSIM vs A0 | LPIPS | dLPIPS vs A0 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 14000 | A0 | 32.3633 | +0.0000 | +0.2512 | 0.955980 | +0.000000 | 0.108264 | +0.000000 |
| 14000 | C30 | 32.0949 | -0.2683 | -0.0171 | 0.955015 | -0.000965 | 0.108530 | +0.000266 |
| 14000 | STOP | 32.1442 | -0.2191 | +0.0321 | 0.955490 | -0.000490 | 0.108237 | -0.000027 |
| 14000 | DECAY | 32.1120 | -0.2512 | +0.0000 | 0.955209 | -0.000771 | 0.108437 | +0.000173 |
| 14000 | H500 | 32.1146 | -0.2486 | +0.0026 | 0.955224 | -0.000756 | 0.108418 | +0.000154 |
| 15000 | A0 | 32.1800 | +0.0000 | +0.1539 | 0.955931 | +0.000000 | 0.108039 | +0.000000 |
| 15000 | C30 | 31.9695 | -0.2106 | -0.0566 | 0.955143 | -0.000787 | 0.108258 | +0.000218 |
| 15000 | STOP | 32.0353 | -0.1447 | +0.0093 | 0.955601 | -0.000330 | 0.108005 | -0.000034 |
| 15000 | DECAY | 32.0261 | -0.1539 | +0.0000 | 0.955511 | -0.000420 | 0.108028 | -0.000011 |
| 15000 | H500 | 32.0229 | -0.1571 | -0.0031 | 0.955513 | -0.000418 | 0.108041 | +0.000002 |

RGB gate:

```text
H500 fails PSNR by about 0.0071 dB at 15k.
H500 passes SSIM and LPIPS.
H500 is almost identical to DECAY at 15k: -0.0031 dB PSNR, +0.0000019 SSIM, +0.000013 LPIPS versus DECAY.
```

### Fixed-bank metrics versus same-step A0

Percent change. Lower is better.

Step 14000:

| Run | Eval | Transfer | J-var | Closure | Consensus-J | Obj-target | DC-var | Recomp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C30 | F | -2.64% | -10.17% | -0.95% | -2.69% | -2.75% | -4.24% | -1.52% |
| C30 | G | -2.80% | -11.89% | -0.52% | -3.02% | -4.78% | -4.66% | -1.56% |
| STOP | F | -1.78% | -7.58% | -0.58% | -1.79% | -0.93% | -3.76% | -0.22% |
| STOP | G | -1.80% | -8.04% | -0.28% | -1.90% | -2.60% | -4.25% | -0.31% |
| DECAY | F | -2.25% | -8.97% | -0.77% | -2.27% | -2.08% | -3.95% | -1.01% |
| DECAY | G | -2.29% | -9.83% | -0.40% | -2.41% | -3.85% | -4.42% | -1.05% |
| H500 | F | -2.22% | -9.00% | -0.75% | -2.23% | -2.07% | -4.14% | -0.98% |
| H500 | G | -2.24% | -9.76% | -0.36% | -2.36% | -3.79% | -4.38% | -1.01% |

Step 15000:

| Run | Eval | Transfer | J-var | Closure | Consensus-J | Obj-target | DC-var | Recomp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C30 | F | -2.23% | -12.15% | -0.78% | -2.26% | -7.10% | -5.35% | -3.42% |
| C30 | G | -2.61% | -15.78% | -0.06% | -2.88% | -9.89% | -6.21% | -4.11% |
| STOP | F | -1.15% | -9.40% | -0.24% | -1.15% | -5.80% | -5.05% | -2.25% |
| STOP | G | -1.33% | -11.70% | +0.39% | -1.44% | -8.41% | -5.61% | -3.04% |
| DECAY | F | -1.40% | -10.28% | -0.34% | -1.40% | -6.31% | -5.29% | -2.58% |
| DECAY | G | -1.61% | -12.83% | +0.34% | -1.74% | -8.97% | -5.95% | -3.38% |
| H500 | F | -1.41% | -10.33% | -0.30% | -1.41% | -6.35% | -5.33% | -2.61% |
| H500 | G | -1.62% | -12.88% | +0.37% | -1.74% | -9.03% | -6.04% | -3.40% |

Relative to DECAY at step 15000:

| Run | Eval | Transfer | J-var | Closure | Consensus-J | Obj-target | DC-var | Recomp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H500 | F | -0.0039% | -0.0455% | +0.0338% | -0.0038% | -0.0459% | -0.0445% | -0.0306% |
| H500 | G | -0.0053% | -0.0595% | +0.0306% | +0.0001% | -0.0668% | -0.1021% | -0.0234% |

### Gradient and medium-change diagnostics

H500 JSONL interval means:

| Interval | Rows | Effective profile lambda | Profile medium grad | Profile/RGB medium grad | RGB medium grad | J* drift | attn delta | bs delta | B_inf delta | transmission delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13001-13500 | 13 | 23.08 | 0.002169 | 0.00889 | 0.2394 | 0.01257 | 0.00499 | 0.00362 | 0.00149 | 0.00303 |
| 13501-13999 | 11 | 0.00 | 0.000000 | 0.00000 | 0.3924 | 0.01637 | 0.00686 | 0.00445 | 0.00185 | 0.00414 |
| 14000-14999 | 22 | 0.00 | 0.000000 | 0.00000 | 0.3552 | 0.01883 | 0.00815 | 0.00463 | 0.00192 | 0.00501 |

This confirms the intended temporal redistribution: H500 uses full profile pressure until 13500 and then behaves like a zero-profile release run. After release, medium/RGB gradients and medium delta magnitudes are close to the prior DECAY/STOP release phase.

### Per-view residual diagnosis at 15k

Mean deltas versus A0:

| Run | dPSNR | dSSIM | dLPIPS | dRGB L1 | dLuma L1 | dChroma L1 |
|---|---:|---:|---:|---:|---:|---:|
| C30 | -0.2106 | -0.000787 | +0.000218 | +0.000943 | +0.001079 | +0.000023 |
| STOP | -0.1447 | -0.000330 | -0.000034 | +0.000440 | +0.000488 | +0.000033 |
| DECAY | -0.1539 | -0.000420 | -0.000011 | +0.000530 | +0.000597 | +0.000032 |
| H500 | -0.1571 | -0.000418 | +0.000002 | +0.000526 | +0.000593 | +0.000031 |

H500 versus DECAY:

| View | Image | dPSNR | dSSIM | dLPIPS | dRGB L1 | dLuma L1 | dChroma L1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | MTN_1288.png | +0.0075 | +0.000008 | +0.000024 | -0.000031 | -0.000030 | -0.000003 |
| 1 | MTN_1296.png | -0.0121 | -0.000004 | +0.000051 | +0.000011 | +0.000011 | -0.000001 |
| 2 | MTN_1304.png | -0.0047 | +0.000002 | -0.000035 | +0.000007 | +0.000008 | +0.000001 |

Per-view interpretation:

```text
H500 is not separated from DECAY by a single catastrophic view.
The average H500-vs-DECAY difference is tiny and mixed across views.
The main residual pattern remains luminance-dominated, as in the previous release sweep.
```

### Gate decision

15k gate:

| Run | RGB gate | Transfer threshold | J-var threshold | DC/recomp/object | Closure | Decision |
|---|---|---|---|---|---|---|
| STOP | Pass | Fail, F=-1.15%, G=-1.33% | Pass | Pass | Eval-G worsens +0.39% | Not final |
| DECAY | Fail PSNR by 0.0039 dB | Fail, F=-1.40%, G=-1.61% | Pass | Pass | Eval-G worsens +0.34% | Not final |
| H500 | Fail PSNR by 0.0071 dB | Fail, F=-1.41%, G=-1.62% | Pass | Pass | Eval-G worsens +0.37% | Not final |

### Reasonable inference

The H500 result does not support the hypothesis that concentrating the same profile budget into the first 500 resumed steps is better than linear decay:

```text
H500 is nearly identical to DECAY at both 14k and 15k.
At 15k, H500 slightly improves transfer/J-var over DECAY by only about 0.004%-0.060%, while losing about 0.0031 dB PSNR.
```

Thus, for the current Curasao P30 setting, the profile effect appears dominated more by cumulative post-13k profile budget than by this coarse 500-step-vs-linear time distribution. The proposed "calibrate, hold briefly, then release" mechanism is not validated by H500.

### Unverified hypotheses

This does not prove that all timing schedules are irrelevant. It only rejects this specific same-budget H500-STOP schedule as a better Pareto candidate. A different budget, a smoother two-stage schedule, or an earlier stop could still behave differently, but these would be new single-factor experiments.

### Next decision

Do not enter cross-scene expansion, longer-than-15k training, or final-candidate reporting from H500.

The current best practical checkpoint remains:

```text
P30 step-13000
```

The best completed 15k RGB-safe run remains:

```text
STOP
```

but STOP does not retain enough transfer to be a GMVC final candidate.

If GMVC continues, the next minimal test should not be H750-STOP yet, because H500 did not show transfer advantage over DECAY. A more informative next single-factor choice is either:

```text
DECAY with slightly earlier full release, e.g. 13000-13750 then zero
```

or:

```text
explicit best-checkpoint strategy around 13k-14k, with matched 13.5k controls if that checkpoint-selection route is acceptable.
```

Do not increase profile lambda, do not tune object lambda, and do not expand scenes until a Curasao 15k strategy passes both RGB and transfer gates.

## 33. Curasao P30-MHOLD calibrated-medium hold test

Date: 2026-08-05

### Code facts

Implemented a strict GMVC medium-hold switch:

```text
gmvc_medium_hold_enabled: bool = False
gmvc_medium_hold_start_step: int = 13001
gmvc_medium_hold_stop_step: int = 15000
```

When active, all medium-owned parameters are set to `requires_grad=False` before the training forward/backward pass and any existing `.grad` is set to `None`. This covers:

```text
medium_mlp
direction_encoding
gmvc_bounded_log_attn_center
gmvc_bounded_log_bs_center
gmvc_bounded_binf_logit_center
```

This is an optimizer-level freeze in the sense relevant for Adam: no gradient tensor exists for the medium parameters, so optimizer momentum does not advance those parameters. The renderer output is not detached as a tensor-level shortcut.

Additional GMVC JSONL audit fields:

```text
gmvc_medium_hold_enabled
gmvc_medium_hold_active
gmvc_medium_hold_start_step
gmvc_medium_hold_stop_step
gmvc_medium_hold_reference_step
gmvc_medium_param_delta_mean_abs
gmvc_medium_param_delta_max_abs
gmvc_medium_param_delta_l2
gmvc_medium_param_delta_shape_mismatch
gmvc_mhold_features_dc_delta_l2
gmvc_mhold_features_rest_delta_l2
gmvc_mhold_features_rest_to_dc_delta_ratio
gmvc_mhold_opacity_delta_l2
gmvc_mhold_geometry_delta_l2
```

The Gaussian reference snapshot is synchronized with post-13k culling, so the adaptation deltas remain shape-aligned after transparent Gaussian removal.

### Experiment facts

The new run:

```text
P30-MHOLD
```

starts from the existing P30 step-13000 checkpoint and keeps:

```text
scene = Curasao
object lambda = 0.004
object ramp factor = 1.0
4 medium : 1 object schedule
target_current_camera_tracks = True
same geometry-only train bank
same Eval-F / Eval-G
```

The only intended difference relative to STOP is:

```text
STOP:
  profile=0 after resume; medium remains trainable under RGB.

MHOLD:
  profile=0 after resume; medium is frozen in all phases.
```

Training command:

```bash
GPU=6 VARIANT=MHOLD \
  scripts/experiments/gmvc_v3_curasao_p30_profile_release_13k_to_15k.sh
```

Evaluation command:

```bash
GPU=6 VARIANTS=MHOLD STEPS=13500,14000,15000 RUN_SUMMARY=0 \
  scripts/experiments/gmvc_v3_curasao_p30_profile_release_eval.sh
```

Summary command:

```bash
/opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/summarize_gmvc_persistence.py \
  --root renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k \
  --variants A0,C30,STOP,DECAY,H500,MHOLD \
  --steps 14000,15000 \
  --reference-variant STOP \
  --start-root renders/gmvc_fixed_bank_diag_20260805/curasao_r500_profile_persistence_3k \
  --start-step 13000 \
  --start-variant P30 \
  --output renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k/summary_with_mhold.json
```

Per-view command:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_gmvc_per_view_residuals.py \
  --a0-config outputs/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_a0/config.yml \
  --a0-step 15000 \
  --run C30=outputs/gmvc_v3_p30_release_c30_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_c30_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_c30/config.yml:15000 \
  --run STOP=outputs/gmvc_v3_p30_release_stop_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_stop_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_stop/config.yml:15000 \
  --run DECAY=outputs/gmvc_v3_p30_release_decay_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_decay_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_decay/config.yml:15000 \
  --run H500=outputs/gmvc_v3_p30_release_h500_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_h500_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_h500/config.yml:15000 \
  --run MHOLD=outputs/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_mhold/config.yml:15000 \
  --reference-run STOP \
  --test-mode test \
  --output-dir renders/gmvc_per_view_residuals_20260805/curasao_p30_profile_mhold_step15000
```

Main outputs:

```text
renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k/summary_with_mhold.json
renders/gmvc_per_view_residuals_20260805/curasao_p30_profile_mhold_step15000/per_view_residual_summary.json
logs/gmvc_v3_p30_release_mhold_20260805_gmvc_v3_p30_profile_release_13k_to_15k.jsonl
```

### Freeze audit

Forced JSONL rows:

| Step | Phase | Hold active | Profile lambda | Object lambda | Profile->medium grad | RGB->medium grad | Medium param mean delta | Medium param max delta |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 13001 | medium | True | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 |
| 13004 | object | True | 0.0 | 0.004 | 0.0 | 0.0 | 0.0 | 0.0 |
| 13500 | medium | True | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 |
| 13501 | medium | True | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 |
| 14000 | medium | True | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 |
| 15000 | medium | True | 0.0 | 0.000 | 0.0 | 0.0 | 0.0 | 0.0 |

Across all 47 JSONL rows:

```text
max(gmvc_medium_param_delta_max_abs) = 0.0
max(gmvc_medium_param_delta_mean_abs) = 0.0
max(gmvc_medium_param_delta_shape_mismatch) = 0.0
max(gmvc_mhold_gaussian_delta_shape_mismatch) = 0.0
```

The sampled fixed-row medium deltas are also zero throughout:

```text
medium_attn delta = 0.0
medium_bs delta = 0.0
B_inf delta = 0.0
transmission delta = 0.0
```

Audit conclusion:

```text
MHOLD medium freezing is strict. The result is not confounded by Adam momentum or hidden medium parameter updates.
```

### Gaussian adaptation diagnostics

Selected JSONL rows:

| Step | RGB->DC grad | RGB->SH-rest grad | Object->DC grad | DC delta L2 | SH-rest delta L2 | Rest/DC delta ratio | Opacity delta L2 | Geometry delta L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 13001 | 0.000681 | 0.002639 | 0.000000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 13004 | 0.000570 | 0.002209 | 0.0000129 | 2.438 | 0.476 | 0.195 | 27.542 | 5.088 |
| 13500 | 0.000681 | 0.002636 | 0.000000 | 47.063 | 18.770 | 0.399 | 300.108 | 141.619 |
| 14000 | 0.000613 | 0.002373 | 0.000000 | 78.959 | 35.475 | 0.449 | 485.050 | 262.042 |
| 15000 | 0.000866 | 0.003352 | 0.000000 | 134.031 | 66.432 | 0.496 | 803.015 | 469.069 |

Interval means:

| Interval | Rows | RGB->DC grad | RGB->SH-rest grad | Object->DC grad | Object/RGB-DC ratio |
|---|---:|---:|---:|---:|---:|
| 13001-13999 | 24 | 0.000715 | 0.002768 | 0.00000264 | 0.00364 |
| 14000-14999 | 22 | 0.000674 | 0.002611 | 0.00000298 | 0.00469 |

The RGB adaptation is not DC-only. SH-rest changes grow to about half the DC delta norm by 15k, so any "success" interpretation must include the possibility that SH-rest absorbs some residual radiometric compensation.

### RGB metrics

| Step | Run | PSNR | dPSNR vs A0 | dPSNR vs STOP | SSIM | dSSIM vs A0 | LPIPS | dLPIPS vs A0 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 14000 | A0 | 32.3633 | +0.0000 | +0.2191 | 0.955980 | +0.000000 | 0.108264 | +0.000000 |
| 14000 | STOP | 32.1442 | -0.2191 | +0.0000 | 0.955490 | -0.000490 | 0.108237 | -0.000027 |
| 14000 | MHOLD | 32.3094 | -0.0539 | +0.1652 | 0.955918 | -0.000061 | 0.107920 | -0.000344 |
| 15000 | A0 | 32.1800 | +0.0000 | +0.1447 | 0.955931 | +0.000000 | 0.108039 | +0.000000 |
| 15000 | STOP | 32.0353 | -0.1447 | +0.0000 | 0.955601 | -0.000330 | 0.108005 | -0.000034 |
| 15000 | MHOLD | 32.2156 | +0.0356 | +0.1803 | 0.955745 | -0.000186 | 0.108008 | -0.000032 |

RGB gate:

```text
MHOLD passes PSNR, SSIM, and LPIPS.
At 15k, MHOLD is +0.0356 dB over same-step A0 and +0.1803 dB over STOP.
```

### Fixed-bank metrics versus same-step A0

Percent change. Lower is better.

Step 14000:

| Run | Eval | Transfer | J-var | Closure | Consensus-J | Obj-target | DC-var | Recomp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| STOP | F | -1.78% | -7.58% | -0.58% | -1.79% | -0.93% | -3.76% | -0.22% |
| STOP | G | -1.80% | -8.04% | -0.28% | -1.90% | -2.60% | -4.25% | -0.31% |
| MHOLD | F | -1.42% | -8.16% | -1.56% | -1.49% | -1.02% | -3.96% | +0.10% |
| MHOLD | G | -1.76% | -10.81% | -1.07% | -2.03% | -4.14% | -4.51% | -0.73% |

Step 15000:

| Run | Eval | Transfer | J-var | Closure | Consensus-J | Obj-target | DC-var | Recomp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| STOP | F | -1.15% | -9.40% | -0.24% | -1.15% | -5.80% | -5.05% | -2.25% |
| STOP | G | -1.33% | -11.70% | +0.39% | -1.44% | -8.41% | -5.61% | -3.04% |
| MHOLD | F | -1.04% | -9.00% | -0.98% | -1.14% | -5.93% | -4.71% | -2.37% |
| MHOLD | G | -1.29% | -11.03% | -0.46% | -1.59% | -9.40% | -5.57% | -3.72% |

Formal fixed-bank gate at 15k:

```text
Transfer threshold not met: F=-1.04%, G=-1.29%, required >=2.0%.
J-var threshold partly fails: F=-9.00% is below 10%, G=-11.03% passes.
DC-var, object-target, recomposition, and closure remain positive versus A0.
```

### Absolute retention versus P30 step-13000

The key medium-only metrics are exactly retained:

| Step | Eval | Transfer | J-var | Closure | Consensus-J |
|---:|---|---:|---:|---:|---:|
| 14000 | F | 0.0% vs P30-13k | 0.0% | 0.0% | 0.0% |
| 14000 | G | 0.0% vs P30-13k | 0.0% | 0.0% | 0.0% |
| 15000 | F | 0.0% vs P30-13k | 0.0% | 0.0% | 0.0% |
| 15000 | G | 0.0% vs P30-13k | 0.0% | 0.0% | 0.0% |

This resolves an apparent contradiction:

```text
The calibrated medium is preserved exactly in absolute fixed-bank terms.
The same-step relative improvement shrinks because the A0 control also changes between 13k and 15k.
```

Therefore the same-step transfer gate is not failed by medium drift. It is failed because the frozen P30-13k medium is not sufficiently better than the 15k A0 medium under the current fixed-bank relative scoring.

### Per-view residual diagnosis at 15k

Mean deltas versus A0:

| Run | dPSNR | dSSIM | dLPIPS | dRGB L1 | dLuma L1 | dChroma L1 |
|---|---:|---:|---:|---:|---:|---:|
| STOP | -0.1447 | -0.000330 | -0.000034 | +0.000440 | +0.000488 | +0.000033 |
| MHOLD | +0.0356 | -0.000186 | -0.000032 | +0.000224 | +0.000318 | -0.000044 |

MHOLD versus STOP:

| View | Image | dPSNR | dSSIM | dLPIPS | dRGB L1 | dLuma L1 | dChroma L1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | MTN_1288.png | +0.0175 | +0.000077 | -0.000188 | +0.000336 | +0.000387 | -0.000309 |
| 1 | MTN_1296.png | +0.3874 | +0.000266 | -0.000310 | -0.000584 | -0.000501 | -0.000059 |
| 2 | MTN_1304.png | +0.1359 | +0.000090 | +0.000505 | -0.000399 | -0.000397 | +0.000139 |

Per-view interpretation:

```text
MHOLD improves PSNR versus STOP on all three eval views.
The strongest gain is view 1.
MHOLD is not a single-view artifact.
Residual changes are mixed: view 0 has worse RGB/luma L1 but better PSNR and chroma; views 1 and 2 improve RGB/luma L1.
```

### Gate decision

| Criterion | MHOLD result | Decision |
|---|---|---|
| RGB PSNR | +0.0356 dB vs A0 | Pass |
| RGB SSIM | -0.000186 vs A0 | Pass |
| RGB LPIPS | -0.000032 vs A0 | Pass |
| Medium parameter freeze | max delta 0.0 | Pass |
| Absolute medium metric retention | exact vs P30-13k | Pass |
| Same-step Eval-F/G transfer | -1.04% / -1.29% | Fail threshold |
| Same-step Eval-F/G J-var | -9.00% / -11.03% | F fails, G passes |
| DC-var/object/recomp/closure | positive vs A0 | Pass |

### Reasonable inference

MHOLD strongly supports the core mechanism that STOP could not isolate:

```text
The P30 step-13000 medium calibration is compatible with RGB recovery when the medium is prevented from drifting.
Gaussian and scene parameters can adapt to the frozen calibrated medium.
```

It also shows that the RGB penalty in C30/DECAY/H500 is not an unavoidable consequence of the calibrated medium state itself. The penalty comes from continued medium movement or from the joint medium/Gaussian optimization path after 13k.

However, MHOLD does not satisfy the predeclared same-step transfer gate. Since absolute medium metrics are exactly retained, this gate failure should be interpreted carefully:

```text
The fixed medium is preserved.
The 15k A0 baseline is stronger under the relative fixed-bank transfer score than the earlier 13k A0 baseline.
```

This means MHOLD is a genuine RGB breakthrough and a strong mechanism result, but it is not yet a clean final GMVC method claim under the current relative gate.

### Unverified hypotheses

The current result does not determine whether the Gaussian adaptation is physically clean. SH-rest delta grows to about half the DC delta by 15k, so part of the RGB recovery may be carried by high-order appearance compensation. This needs a targeted appearance diagnostic before declaring MHOLD a final method.

### Next decision

Do not expand to other scenes yet and do not train beyond 15k yet.

The next minimal check should be diagnostic, not another schedule sweep:

```text
Evaluate MHOLD 15k for SH-rest compensation and DC-only recomposition.
```

Recommended single diagnostic:

```text
Compare A0 / P30-13k / STOP-15k / MHOLD-15k:
  full RGB metrics
  DC-only render metrics if available
  full-SH minus DC-only residual
  object-safe residual maps
  fixed-bank metrics with and without forced DC proxy
```

If MHOLD's RGB gain remains under DC-only or mostly DC-supported rendering, then MHOLD is a strong candidate for the formal GMVC continuation rule. If the gain is mostly SH-rest compensation, then the next module should constrain Gaussian appearance adaptation after medium freeze rather than changing the medium schedule.

## 34. Curasao MHOLD catch-up and SH-rest no-training audits

Date: 2026-08-05

Commit before this section:

```text
3a90929 Add GMVC medium hold continuation experiment
```

New diagnostic scripts:

```text
scripts/diagnostics/summarize_gmvc_catchup_audit.py
scripts/diagnostics/diagnose_gmvc_sh_rest_contribution.py
```

Outputs:

```text
renders/gmvc_fixed_bank_diag_20260805/curasao_mhold_catchup_audit/gmvc_catchup_audit_summary.json
renders/gmvc_fixed_bank_diag_20260805/curasao_mhold_catchup_audit/gmvc_catchup_audit_summary.md
renders/gmvc_sh_rest_audit_20260805/curasao_full/gmvc_sh_rest_contribution_summary.json
```

Commands:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/summarize_gmvc_catchup_audit.py \
  --output-dir renders/gmvc_fixed_bank_diag_20260805/curasao_mhold_catchup_audit
```

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_sh_rest_contribution.py \
  --run P30_13K=outputs/gmvc_v3_r500_p30_profile_persistence3k_curasao_seed42_step10000_to_13000/water-splatting/gmvc_v3_r500_p30_profile_persistence3k_curasao_seed42_step10000_to_13000_20260805_gmvc_v3_curasao_r500_profile_persistence_3k_p30_p30_r500_g000/config.yml:13000 \
  --run MHOLD_15K=outputs/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_mhold/config.yml:15000 \
  --run A0_15K=outputs/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_a0/config.yml:15000 \
  --track-bank EVALF=renders/gmvc_v2_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt \
  --track-bank EVALG=renders/gmvc_v3_geometry_track_banks/curasao_m1_step10000_train_s4096/gmvc_track_bank.pt \
  --test-mode test --max-images -1 --max-tracks 30000 \
  --output-dir renders/gmvc_sh_rest_audit_20260805/curasao_full
```

### 34.1 A0 catch-up audit

The audit used only existing fixed-bank JSON/checkpoints. Lower is better for all fixed-bank metrics.

Eval-F absolute values:

| Run | transfer | J-var | closure | consensus-J | obj-target | DC-var | recomp |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0-13k | 0.02111963 | 0.00657435 | 0.10122827 | 0.01447284 | 0.07345321 | 0.00188982 | 0.02264931 |
| A0-14k | 0.02094264 | 0.00633670 | 0.10173087 | 0.01436557 | 0.06510883 | 0.00189892 | 0.02099976 |
| A0-15k | 0.02086218 | 0.00639501 | 0.10113202 | 0.01431394 | 0.06904572 | 0.00193389 | 0.02162616 |
| P30-13k | 0.02064424 | 0.00581974 | 0.10014221 | 0.01415097 | 0.06578295 | 0.00183853 | 0.02126446 |
| MHOLD-15k | 0.02064424 | 0.00581974 | 0.10014221 | 0.01415097 | 0.06494885 | 0.00184273 | 0.02111380 |

Eval-G absolute values:

| Run | transfer | J-var | closure | consensus-J | obj-target | DC-var | recomp |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0-13k | 0.02038126 | 0.01236850 | 0.08841901 | 0.01410909 | 0.09512940 | 0.00202648 | 0.02462047 |
| A0-14k | 0.02028283 | 0.01210805 | 0.08896773 | 0.01407697 | 0.08530914 | 0.00204093 | 0.02291383 |
| A0-15k | 0.02018684 | 0.01213915 | 0.08842448 | 0.01401308 | 0.09050732 | 0.00208890 | 0.02368240 |
| P30-13k | 0.01992675 | 0.01079966 | 0.08801936 | 0.01379083 | 0.08333792 | 0.00195394 | 0.02297537 |
| MHOLD-15k | 0.01992675 | 0.01079966 | 0.08801936 | 0.01379083 | 0.08199851 | 0.00197246 | 0.02280215 |

Relative shrinkage is:

```text
delta_relative = (MHOLD_15K - A0_15K) - (P30_13K - A0_13K)
```

Positive values mean the relative advantage shrank.

| Eval | Metric | P30-13k advantage | MHOLD-15k advantage | delta_relative | A0 13k->15k improvement |
|---|---|---:|---:|---:|---:|
| F | transfer | -0.00047539 | -0.00021794 | +0.00025745 | +0.00025745 |
| F | J-var | -0.00075461 | -0.00057527 | +0.00017934 | +0.00017934 |
| F | closure | -0.00108606 | -0.00098982 | +0.00009624 | +0.00009624 |
| F | consensus-J | -0.00032188 | -0.00016297 | +0.00015890 | +0.00015890 |
| F | obj-target | -0.00767026 | -0.00409687 | +0.00357340 | +0.00440749 |
| F | DC-var | -0.00005129 | -0.00009116 | -0.00003987 | -0.00004407 |
| F | recomp | -0.00138486 | -0.00051236 | +0.00087250 | +0.00102316 |
| G | transfer | -0.00045450 | -0.00026008 | +0.00019442 | +0.00019442 |
| G | J-var | -0.00156883 | -0.00133949 | +0.00022935 | +0.00022935 |
| G | closure | -0.00039965 | -0.00040512 | -0.00000546 | -0.00000546 |
| G | consensus-J | -0.00031825 | -0.00022224 | +0.00009601 | +0.00009601 |
| G | obj-target | -0.01179148 | -0.00850881 | +0.00328267 | +0.00462208 |
| G | DC-var | -0.00007254 | -0.00011643 | -0.00004389 | -0.00006242 |
| G | recomp | -0.00164510 | -0.00088025 | +0.00076485 | +0.00093807 |

Catch-up conclusion:

```text
For transfer, J-var, closure, and consensus-J, MHOLD-15k exactly retains P30-13k.
The relative advantage shrinkage is therefore almost exactly A0's own 13k-to-15k movement.
GMVC currently looks more like a medium calibration accelerator than a proven better asymptotic decomposition.
```

Object/DC metrics remain more favorable for MHOLD than A0 in several places, especially object-target and DC-var. This leaves a weaker possibility that GMVC changes responsibility allocation, but the strongest fixed-medium metrics no longer show a large same-step advantage.

### 34.2 SH-rest contribution audit

The SH audit renders every Curasao test eval view twice from each checkpoint:

```text
Full-SH: normal model._get_active_sh_degree()
DC-only: temporary diagnostic override model._get_active_sh_degree() = 0
```

The override is used only during forward rendering. Checkpoints are not modified.

Mean eval metrics:

| Run | Full PSNR | DC-only PSNR | dPSNR Full-DC | Full SSIM | DC SSIM | dLPIPS Full-DC | RGB L1 gain | Luma gain | Chroma gain | SH abs mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P30-13k | 32.2673 | 29.8909 | +2.3764 | 0.956042 | 0.939543 | -0.029750 | +0.003621 | +0.003961 | +0.000457 | 0.010570 |
| MHOLD-15k | 32.2156 | 29.9081 | +2.3075 | 0.955745 | 0.939329 | -0.030218 | +0.003327 | +0.003642 | +0.000420 | 0.010986 |
| A0-15k | 32.1800 | 29.7598 | +2.4202 | 0.955931 | 0.939225 | -0.030393 | +0.003559 | +0.003952 | +0.000453 | 0.011241 |

Track-level SH variance:

| Run | Bank | E_SH-var | corr depth | corr T | corr B | corr residual | corr ray-z |
|---|---|---:|---:|---:|---:|---:|---:|
| P30-13k | Eval-F | 0.00005163 | -0.3161 | +0.3099 | -0.2743 | +0.2250 | -0.1754 |
| P30-13k | Eval-G | 0.00004860 | -0.3406 | +0.2757 | -0.3630 | +0.3513 | -0.3245 |
| MHOLD-15k | Eval-F | 0.00005423 | -0.3087 | +0.3086 | -0.2664 | +0.2222 | -0.1630 |
| MHOLD-15k | Eval-G | 0.00005127 | -0.3374 | +0.2746 | -0.3591 | +0.3441 | -0.3177 |
| A0-15k | Eval-F | 0.00005273 | -0.3014 | +0.2967 | -0.2616 | +0.2210 | -0.1646 |
| A0-15k | Eval-G | 0.00005055 | -0.3273 | +0.2534 | -0.3486 | +0.3473 | -0.3171 |

Per-view summary:

| Run | View | dPSNR Full-DC | SH abs mean | corr depth | corr T | corr B | corr residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| P30-13k | 0 | -0.6176 | 0.009528 | -0.4432 | +0.2427 | -0.4637 | -0.0365 |
| P30-13k | 1 | +5.4106 | 0.013433 | -0.3956 | +0.3289 | -0.4514 | +0.4913 |
| P30-13k | 2 | +2.3361 | 0.008749 | -0.3857 | +0.3808 | -0.3815 | +0.2943 |
| MHOLD-15k | 0 | -0.7851 | 0.010472 | -0.4232 | +0.2104 | -0.4586 | -0.0141 |
| MHOLD-15k | 1 | +5.4236 | 0.013594 | -0.3909 | +0.3197 | -0.4455 | +0.4953 |
| MHOLD-15k | 2 | +2.2842 | 0.008893 | -0.3789 | +0.3753 | -0.3741 | +0.2960 |
| A0-15k | 0 | -0.9447 | 0.010761 | -0.4250 | +0.2236 | -0.4414 | +0.0537 |
| A0-15k | 1 | +5.8294 | 0.013931 | -0.3855 | +0.2870 | -0.4366 | +0.4847 |
| A0-15k | 2 | +2.3760 | 0.009031 | -0.3653 | +0.3603 | -0.3580 | +0.3005 |

Cross-run residual relation using MHOLD SH magnitude:

| Comparison | mean corr with residual improvement | mean residual improvement |
|---|---:|---:|
| MHOLD-15k vs P30-13k | -0.0525 | -0.00028753 |
| MHOLD-15k vs A0-15k | +0.1070 | -0.00022399 |

SH audit interpretation:

```text
Full-SH is important for all three checkpoints.
MHOLD is not more dependent on SH-rest than A0-15k or P30-13k.
The water-parameter correlations of SH contribution are similar across all three checkpoints.
The residual-improvement correlation with MHOLD SH magnitude is weak and not a clean sign of water-error reabsorption.
```

The strongest evidence against an immediate RESTFREEZE run is that A0-15k has a larger Full-SH over DC-only PSNR gain than MHOLD-15k:

```text
A0-15k:    +2.4202 dB
P30-13k:   +2.3764 dB
MHOLD-15k: +2.3075 dB
```

MHOLD's track-level E_SH-var is slightly higher than A0/P30, but the increase is small:

```text
Eval-F: MHOLD 0.00005423 vs A0 0.00005273
Eval-G: MHOLD 0.00005127 vs A0 0.00005055
```

This is not enough to claim that MHOLD's RGB recovery mainly comes from a new SH-rest water-compensation path.

### 34.3 RESTFREEZE gate

RESTFREEZE was not run.

Reason:

```text
The trigger condition was not met.
The audit shows generic SH-rest dependence already present in A0 and P30, not a distinct MHOLD-specific recovery channel.
The correlations with propagation depth, transmission, and backscatter are systematic but shared across runs.
Running RESTFREEZE now would test a broad "freeze all high-order appearance" question rather than the intended conditional control.
```

Updated conclusion:

```text
MHOLD remains the strongest GMVC mechanism result.
It proves calibrated-medium hold is compatible with RGB recovery.
The same-step medium advantage shrinks mainly because A0 catches up between 13k and 15k.
The SH audit does not show that MHOLD succeeds by uniquely hiding water residuals in high-order SH.
No further profile scheduling, cross-scene expansion, or >15k training is justified from this branch yet.
```

## 35. Curasao A0-MHOLD generic medium-freeze control

Date: 2026-08-05

Purpose:

```text
Test whether P30-MHOLD succeeds because P30 found a better calibrated medium,
or because freezing any 13k medium is already beneficial for RGB recovery.
```

Code change:

```text
scripts/experiments/gmvc_v3_curasao_p30_profile_release_13k_to_15k.sh
scripts/experiments/gmvc_v3_curasao_p30_profile_release_eval.sh
```

Both scripts now accept:

```text
VARIANT=A0_MHOLD
```

The variant starts from the existing A0 step-13000 checkpoint, enables GMVC phase logic and object auxiliary, sets profile lambda to zero, and activates the same medium hold mechanism used by P30-MHOLD.

Training command:

```bash
VARIANT=A0_MHOLD GPU=6 RUN_EVAL=0 RUN_CLOSURE_DIAG=0 \
SAVE_ONLY_LATEST_CHECKPOINT=False STEPS_PER_SAVE=500 \
/bin/bash scripts/experiments/gmvc_v3_curasao_p30_profile_release_13k_to_15k.sh
```

Evaluation commands:

```bash
VARIANTS=A0_MHOLD STEPS=14000,15000 GPU=6 RUN_SUMMARY=0 \
/bin/bash scripts/experiments/gmvc_v3_curasao_p30_profile_release_eval.sh
```

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/summarize_gmvc_persistence.py \
  --root renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k \
  --variants A0,A0_MHOLD,MHOLD \
  --steps 14000,15000 \
  --reference-variant MHOLD \
  --start-root renders/gmvc_fixed_bank_diag_20260805/curasao_r500_profile_persistence_3k \
  --start-step 13000 \
  --start-variant P30 \
  --output renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k/summary_with_a0_mhold.json
```

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gmvc_per_view_residuals.py \
  --a0-config outputs/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_a0_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_a0/config.yml \
  --a0-step 15000 \
  --run A0_MHOLD=outputs/gmvc_v3_p30_release_a0_mhold_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_a0_mhold_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_a0_mhold/config.yml:15000 \
  --run P30_MHOLD=outputs/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000/water-splatting/gmvc_v3_p30_release_mhold_curasao_seed42_step13000_to_15000_20260805_gmvc_v3_p30_profile_release_13k_to_15k_mhold/config.yml:15000 \
  --reference-run A0 \
  --test-mode test --max-images -1 \
  --output-dir renders/gmvc_per_view_residuals_20260805/curasao_a0_mhold_step15000
```

Main outputs:

```text
outputs/gmvc_v3_p30_release_a0_mhold_curasao_seed42_step13000_to_15000/...
logs/gmvc_v3_p30_release_a0_mhold_20260805_gmvc_v3_p30_profile_release_13k_to_15k.jsonl
renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k/a0_mhold/
renders/gmvc_fixed_bank_diag_20260805/curasao_p30_profile_release_15k/summary_with_a0_mhold.json
renders/gmvc_per_view_residuals_20260805/curasao_a0_mhold_step15000/per_view_residual_summary.json
```

### 35.1 Freeze audit

Across all 47 JSONL rows:

```text
max(gmvc_medium_param_delta_max_abs) = 0.0
max(gmvc_medium_param_delta_mean_abs) = 0.0
max(gmvc_medium_param_delta_shape_mismatch) = 0.0
max(gmvc_medium_attn_delta_l1_mean) = 0.0
max(gmvc_medium_bs_delta_l1_mean) = 0.0
max(gmvc_b_inf_delta_l1_mean) = 0.0
max(gmvc_transmission_delta_l1_mean) = 0.0
max(rgb_grad_norm_medium) = 0.0
```

Selected rows:

| Step | Phase | Profile lambda | Object lambda | Medium hold | RGB->medium grad | Medium param max delta | Attn delta | BS delta | B_inf delta | T delta | DC delta L2 | SH-rest delta L2 |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 13001 | medium | 0.0 | 0.000 | True | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.000 | 0.000 |
| 13004 | object | 0.0 | 0.004 | True | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.446 | 0.478 |
| 13500 | medium | 0.0 | 0.000 | True | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 60.244 | 20.226 |
| 14000 | medium | 0.0 | 0.000 | True | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 94.649 | 37.375 |
| 15000 | medium | 0.0 | 0.000 | True | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 151.045 | 68.999 |

Conclusion:

```text
A0-MHOLD is a valid generic medium-freeze control.
The A0 medium state is held exactly from step 13001 to 15000.
```

### 35.2 RGB metrics

| Step | Run | PSNR | dPSNR vs A0 | dPSNR vs P30-MHOLD | SSIM | dSSIM vs A0 | LPIPS | dLPIPS vs A0 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 14000 | A0 | 32.3633 | +0.0000 | +0.0539 | 0.955980 | +0.000000 | 0.108264 | +0.000000 |
| 14000 | A0-MHOLD | 32.5347 | +0.1715 | +0.2254 | 0.956992 | +0.001012 | 0.107449 | -0.000815 |
| 14000 | P30-MHOLD | 32.3094 | -0.0539 | +0.0000 | 0.955918 | -0.000061 | 0.107920 | -0.000344 |
| 15000 | A0 | 32.1800 | +0.0000 | -0.0356 | 0.955931 | +0.000000 | 0.108039 | +0.000000 |
| 15000 | A0-MHOLD | 32.4629 | +0.2829 | +0.2473 | 0.956852 | +0.000922 | 0.107474 | -0.000566 |
| 15000 | P30-MHOLD | 32.2156 | +0.0356 | +0.0000 | 0.955745 | -0.000186 | 0.108008 | -0.000032 |

RGB recovery from each 13k start:

| Run | Start | End | dPSNR | dSSIM | dLPIPS |
|---|---|---|---:|---:|---:|
| A0-MHOLD | A0-13k | A0-MHOLD-15k | +0.2366 | -0.000070 | -0.000513 |
| P30-MHOLD | P30-13k | P30-MHOLD-15k | -0.0517 | -0.000298 | +0.000075 |

RGB conclusion:

```text
Freezing the ordinary A0-13k medium is strongly beneficial for RGB.
A0-MHOLD is +0.2829 dB over normal A0-15k and +0.2473 dB over P30-MHOLD-15k.
This fails the proposed "P30-MHOLD RGB not below A0-MHOLD by more than 0.05 dB" condition.
```

### 35.3 Fixed-bank absolute comparison at 15k

Eval-F:

| Run | transfer | J-var | closure | consensus-J | obj-target | DC-var | recomp |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.02086218 | 0.00639501 | 0.10113202 | 0.01431394 | 0.06904572 | 0.00193389 | 0.02162616 |
| A0-MHOLD | 0.02111963 | 0.00657435 | 0.10122827 | 0.01447284 | 0.06984047 | 0.00194658 | 0.02187263 |
| P30-MHOLD | 0.02064424 | 0.00581974 | 0.10014221 | 0.01415097 | 0.06494885 | 0.00184273 | 0.02111380 |

Eval-G:

| Run | transfer | J-var | closure | consensus-J | obj-target | DC-var | recomp |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.02018684 | 0.01213915 | 0.08842448 | 0.01401308 | 0.09050732 | 0.00208890 | 0.02368240 |
| A0-MHOLD | 0.02038126 | 0.01236850 | 0.08841901 | 0.01410909 | 0.08973609 | 0.00209275 | 0.02372395 |
| P30-MHOLD | 0.01992675 | 0.01079966 | 0.08801936 | 0.01379083 | 0.08199851 | 0.00197246 | 0.02280215 |

P30-MHOLD versus A0-MHOLD at 15k:

| Eval | transfer | J-var | closure | consensus-J | obj-target | DC-var | recomp |
|---|---:|---:|---:|---:|---:|---:|---:|
| F abs | -0.00047539 | -0.00075461 | -0.00108606 | -0.00032188 | -0.00489161 | -0.00010385 | -0.00075883 |
| F pct | -2.25% | -11.48% | -1.07% | -2.22% | -7.00% | -5.34% | -3.47% |
| G abs | -0.00045450 | -0.00156883 | -0.00039965 | -0.00031825 | -0.00773758 | -0.00012029 | -0.00092179 |
| G pct | -2.23% | -12.68% | -0.45% | -2.26% | -8.62% | -5.75% | -3.89% |

Fixed-bank conclusion:

```text
P30-MHOLD clearly has the better medium/object decomposition metrics.
A0-MHOLD keeps the A0-13k medium-only metrics unchanged, so its worse transfer/J-var are not an evaluation artifact.
```

The A0-MHOLD medium-only metrics match A0-13k exactly for transfer, J-var, closure, and consensus-J. Its object-target/recomposition metrics move because those include the current Gaussian DC proxy, which is allowed to adapt during the hold phase.

### 35.4 Per-view RGB residuals at 15k

Mean deltas versus normal A0-15k:

| Run | dPSNR | dSSIM | dLPIPS | dRGB L1 | dLuma L1 | dChroma L1 |
|---|---:|---:|---:|---:|---:|---:|
| A0-MHOLD | +0.2829 | +0.000922 | -0.000566 | -0.000919 | -0.000985 | -0.000011 |
| P30-MHOLD | +0.0356 | -0.000186 | -0.000032 | +0.000224 | +0.000318 | -0.000044 |

Per-view deltas versus normal A0-15k:

| View | Image | A0-MHOLD dPSNR | P30-MHOLD dPSNR | A0-MHOLD dRGB L1 | P30-MHOLD dRGB L1 |
|---:|---|---:|---:|---:|---:|
| 0 | MTN_1288.png | +0.4248 | -0.1168 | -0.001817 | +0.001176 |
| 1 | MTN_1296.png | +0.2277 | +0.1384 | -0.000424 | -0.000248 |
| 2 | MTN_1304.png | +0.1961 | +0.0852 | -0.000516 | -0.000256 |

Per-view conclusion:

```text
A0-MHOLD improves PSNR on all three eval views relative to A0-15k.
P30-MHOLD improves two of three views but loses view 0.
The A0-MHOLD RGB advantage is not a single-view artifact.
```

### 35.5 Gate decision

This run lands in the mixed case:

```text
A0-MHOLD has better RGB.
P30-MHOLD has better fixed-bank medium/object metrics.
A0-MHOLD is also clearly better than normal A0-15k.
```

Therefore:

```text
The P30 profile calibration does provide an independent decomposition advantage over generic medium freeze.
However, it does not dominate the generic freeze control in RGB.
The strongest new result is that medium hold itself is a powerful training mechanism on Curasao.
```

Current interpretation:

```text
GMVC should be split into two factors:
1. medium early stopping / hold, which strongly improves RGB on Curasao;
2. profile calibration, which improves transfer, J-var, object-target, DC-var, and recomposition but introduces an RGB Pareto cost relative to A0-MHOLD.
```

This means P30-MHOLD should not yet enter cross-scene as a final candidate. The next step should not be another profile schedule sweep. A better next control would isolate whether A0 medium hold alone, without GMVC object auxiliary, produces the same RGB gain, because A0-MHOLD currently differs from normal A0 by both medium freezing and the GMVC object auxiliary/phase structure.
