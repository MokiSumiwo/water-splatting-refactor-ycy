# WaterSplatting 基于 M1 的新模块实验方案
## 从 M2 容量抑制转向固有色与水下外观解耦

## 1. 研究背景与阶段判断

当前项目已经完成 M2 的多阶段实验，包括 infinite-water ownership、capacity regularization、hit-aware depth、object-retention capacity floor、monotonic budget 以及 closed-tail composition。现有结果表明，M2 能降低远处 Gaussian accumulation 与 clear leakage，但没有任何配置能够在多随机种子下同时满足以下目标：

- 带水体新视角重建指标不低于 M1；
- LPIPS 不恶化；
- 远处蓝绿色 Gaussian 残留显著降低；
- 真实远景物体和边界保持完整；
- 实验结果具有足够稳定性。

机制拆解进一步说明，真正减少远水残留的是 capacity regularization，而不是 \(B_\infty\) composition；但原始、单调和预算式 capacity loss 都会不同程度损害重建质量或感知质量。Closed-tail 的解析拆分虽然数值正确，正式训练仍未形成稳定正收益。因此，当前 M2 不再作为主方法继续扩展，而是保留为 diagnostics 与 appendix。

本阶段从稳定的 M1 基线重新设计：

## Intrinsic–Underwater Dual-Color Gaussian Appearance

中文名称：

## 固有色与水下外观解耦的双颜色 Gaussian 表示

核心目标是：

\[
\boxed{
\text{保留对 underwater reconstruction 有用的 Gaussian 表达能力，}
\quad
\text{同时阻止水体型蓝绿色外观进入 clear representation}
}
\]

---

## 2. 新模块研究假设

当前 Gaussian 的同一套 SH 外观同时承担：

1. 带水体 RGB 新视角重建；
2. 去水体固有颜色输出。

远处物体透射率较低，clear color 获得的有效梯度很弱。模型可能将水体偏色、视角相关反射、弱纹理远景补偿以及 medium field 未建模残差共同吸收到 Gaussian SH 或 DC 中。因此：

\[
\text{带水体重建有用的外观}
\neq
\text{去水体结果应保留的固有外观}.
\]

建议将 Gaussian 外观分解为：

\[
c_i^{\mathrm{uw}}(\mathbf v)
=
c_i^{\mathrm{int}}
+
\Delta c_i^{\mathrm{view}}(\mathbf v),
\]

其中：

- \(c_i^{\mathrm{int}}\) 为稳定 intrinsic color；
- \(\Delta c_i^{\mathrm{view}}\) 为水下条件下的视角相关残差。

带水体分支使用完整颜色：

\[
c_i^{\mathrm{uw}}(\mathbf v),
\]

去水体分支主要使用：

\[
c_i^J \approx c_i^{\mathrm{int}}.
\]

---

## 3. 总体实验路线

```text
M1 checkpoint
    ↓
Post-hoc DC/SH diagnosis
    ↓
Dual-color representation
    ↓
Frozen-geometry fine-tuning
    ↓
Intrinsic anchoring and residual regularization
    ↓
Low-learning-rate joint refinement
    ↓
FUNA clear-GT verification
    ↓
Optional multi-view surface-support module
```

---

## 4. Phase 0：冻结 M1 与终止旧 M2 主线

### 4.1 固定 M1 基线

建议固定：

```text
medium_context_mode = dir_xy_camera
infinite_water_enabled = False
constrained_appearance_enabled = False
sh_degree = 3
max_iterations = 15000
```

记录：

```text
checkpoint
Git commit
seed
PSNR / SSIM / LPIPS
Gaussian count
J Blue / Green Dominance
water/object/boundary diagnostics
```

### 4.2 M2 降级处理

不再进入新主线：

```text
capacity loss
dynamic hit protection
capacity floor
rgb_mix
tail_approx
closed-tail training
hard pruning
```

保留为诊断：

```text
expected depth
depth variance
first / last depth
final transmittance
q_hit
common far mask
water/object/boundary masks
```

---

## 5. Phase 1：Post-hoc DC/SH 外观诊断

### 5.1 实验目的

在不重新训练的情况下判断远处蓝绿色残留主要来自：

- 高阶 SH；
- DC 固有颜色；
- Gaussian 几何与 opacity。

### 5.2 SH 分解

当前颜色写为：

\[
c_i^{\mathrm{SH}}(\mathbf v)
=
c_i^{\mathrm{DC}}
+
\Delta c_i^{\mathrm{SH}}(\mathbf v).
\]

将 SH residual 分解为亮度与色度：

\[
\Delta l_i(\mathbf v)
=
\operatorname{mean}_c
\left[
\Delta c_i^{\mathrm{SH}}(\mathbf v)
\right],
\]

\[
\Delta c_i^{\perp}(\mathbf v)
=
\Delta c_i^{\mathrm{SH}}(\mathbf v)
-
\Delta l_i(\mathbf v).
\]

定义 clear appearance：

\[
c_i^J
=
c_i^{\mathrm{DC}}
+
\eta_l\Delta l_i(\mathbf v)\mathbf 1
+
\eta_c\Delta c_i^{\perp}(\mathbf v).
\]

### 5.3 无训练诊断矩阵

| 编号 | Underwater RGB | Clear \(J\) |
|---|---|---|
| A0 | Full SH=3 | Full SH=3 |
| A1 | Full SH=3 | DC only |
| A2 | Full SH=3 | DC + luminance-only SH |
| A3 | Full SH=3 | DC + luminance + 0.05 chroma |
| A4 | Full SH=3 | DC + luminance + 0.10 chroma |

对应：

```text
A1: eta_l=0,   eta_c=0
A2: eta_l=1.0, eta_c=0
A3: eta_l=1.0, eta_c=0.05
A4: eta_l=1.0, eta_c=0.10
```

### 5.4 输出

```text
Underwater RGB
A0–A4 clear renders
A0-A1 difference
SH luminance residual
SH chroma residual
common far mask overlay
water/object/boundary overlays
```

### 5.5 决策规则

若 A1/A2 显著减少蓝绿色残留且结构完整，说明问题主要来自高阶 SH，进入 Dual-Color 模块。

若 A1 仍有明显偏色但结构合理，说明颜色污染已进入 DC，进入 Dual-Color 并加强 intrinsic anchoring。

若 A1 中仍有明显漂浮蓝色结构，说明问题还包含 Gaussian geometry/opacity 错误，后续再启动多视角表面支持模块。

---

## 6. Phase 2：Dual-Color Gaussian 基础实现

### 6.1 外观职责

保留现有参数但重新定义：

```text
features_dc   → intrinsic color
features_rest → underwater view-dependent residual
```

### 6.2 双分支颜色

带水体颜色：

\[
c_i^{\mathrm{uw}}(\mathbf v)
=
c_i^{\mathrm{int}}
+
\Delta c_i^{\mathrm{view}}(\mathbf v).
\]

去水体颜色：

\[
c_i^J(\mathbf v)
=
c_i^{\mathrm{int}}
+
\eta_l\Delta l_i(\mathbf v)\mathbf 1
+
\eta_c\Delta c_i^\perp(\mathbf v).
\]

推荐默认：

```text
eta_l = 1.0
eta_c = 0.0
```

这样 clear branch 保留视角相关亮度，但不允许 SH 自由改变色相。

### 6.3 Rasterizer 原型

优先让同一次 rasterization 接收：

```text
colors_underwater
colors_intrinsic
```

两套颜色共享：

```text
projection
visibility
sorting
opacity
transmittance
depth
```

输出：

```text
rgb_object_underwater
rgb_object_intrinsic
rgb_medium
rgb_final
J_intrinsic
```

若暂时不修改 CUDA，可先使用完全相同的几何和 opacity 运行两次 rasterization 完成原型验证。

---

## 7. Phase 3：冻结几何的双颜色微调

### 7.1 初始化

从稳定 M1 checkpoint 初始化：

```text
intrinsic color ← current DC
view residual   ← current higher-order SH
medium MLP      ← M1
geometry        ← M1
opacity         ← M1
```

### 7.2 冻结参数

第一轮冻结：

```text
means
scales
quats
opacities
medium MLP
```

只训练：

```text
features_dc
features_rest
dual-color control parameters
```

建议训练：

```text
2000 steps
```

并补充：

```text
1000 / 3000 steps
```

作为训练长度消融。

### 7.3 基础矩阵

| 编号 | Clear appearance |
|---|---|
| B0 | Full SH |
| B1 | DC only |
| B2 | DC + luminance SH |
| B3 | DC + luminance + 0.05 chroma |
| B4 | DC + luminance + 0.10 chroma |

---

## 8. Phase 4：Intrinsic Anchoring 与残差约束

### 8.1 Near-Transmission Anchor

M1 预测的平均透射率：

\[
\bar T(p)
=
\frac{1}{3}
\sum_c
\exp[-\beta_c^D(p)d(p)].
\]

定义高透射权重：

\[
w_{\mathrm{near}}(p)
=
\sigma
\left(
\frac{\bar T(p)-\tau_T}{t_T}
\right).
\]

在高透射区域约束 view residual：

\[
\mathcal L_{\mathrm{near}}
=
\frac{
\sum_p
w_{\mathrm{near}}(p)
\left\|
\Delta c^{\mathrm{view}}(p)
\right\|_1
}{
\sum_p w_{\mathrm{near}}(p)+\epsilon
}.
\]

推荐：

```text
tau_T = 0.70
t_T = 0.10
```

### 8.2 Residual Mean Anchor

\[
\mathcal L_{\mathrm{mean}}
=
\left\|
\mathbb E_{\mathbf v}
\left[
\Delta c_i^{\mathrm{view}}(\mathbf v)
\right]
\right\|_1.
\]

其作用是让稳定颜色进入 intrinsic branch，而不是长期存储于 view residual。

### 8.3 Chroma Constraint

\[
\mathcal L_{\mathrm{chroma}}
=
\left\|
\Delta c_i^\perp(\mathbf v)
\right\|_1.
\]

该损失仅约束 residual 的色度，不对真实物体 DC 施加红蓝通道偏好。

### 8.4 总损失

\[
\mathcal L
=
\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{near}}\mathcal L_{\mathrm{near}}
+
\lambda_{\mathrm{mean}}\mathcal L_{\mathrm{mean}}
+
\lambda_{\mathrm{chroma}}\mathcal L_{\mathrm{chroma}}.
\]

首轮建议：

```text
lambda_near   ∈ {0, 1e-4, 5e-4}
lambda_mean   ∈ {0, 1e-4}
lambda_chroma ∈ {0, 1e-4, 5e-4}
```

按逐项消融进行，不做完整笛卡尔积。

---

## 9. Phase 5：逐项消融矩阵

| 编号 | Clear branch | Near anchor | Mean anchor | Chroma constraint |
|---|---|---:|---:|---:|
| C0 | Full SH | 0 | 0 | 0 |
| C1 | DC only | 0 | 0 | 0 |
| C2 | DC + luminance | 0 | 0 | 0 |
| C3 | DC + luminance | 开 | 0 | 0 |
| C4 | DC + luminance | 开 | 开 | 0 |
| C5 | DC + luminance | 开 | 开 | 开 |
| C6 | DC + luminance + 0.05 chroma | 开 | 开 | 开 |

选择优先级：

1. underwater RGB 指标不低于 M1；
2. clear GT 或真实颜色诊断改善；
3. water-region J leakage 不增加；
4. object J retention 与边界保留不下降；
5. 多 seed 稳定。

---

## 10. Phase 6：低学习率联合微调

从 Phase 5 选择两个最佳配置。

### 10.1 第一轮解冻

解冻：

```text
features_dc
features_rest
medium MLP
```

保持冻结：

```text
means
scales
quats
opacities
```

训练：

```text
1000–3000 steps
```

medium MLP 学习率使用 M1 原学习率的：

```text
0.1×
```

### 10.2 可选全局微调

只有当颜色稳定改善但 underwater RGB 略低时，才解冻 geometry 与 opacity：

```text
500–1000 steps
very low learning rate
disable densification
disable split/duplicate
```

---

## 11. Phase 7：FUNA Clear-GT 验证

### 11.1 训练输入

仅使用：

```text
underwater RGB
camera poses
```

clean GT 只用于评价，不参与训练，除非明确标注为 oracle ablation。

### 11.2 指标

整体报告：

```text
Clear PSNR
Clear SSIM
Clear LPIPS
Underwater PSNR
Underwater SSIM
Underwater LPIPS
CIEDE2000
```

按深度分箱：

```text
Near
Middle
Far
```

统计：

```text
Clear PSNR by depth
Lab ΔE by depth
Blue/green channel bias by depth
```

按水体条件统计：

```text
Blue water level 1–4
Green water level 1–4
```

---

## 12. Phase 8：真实数据验证

在 IUI3-RedSea 与 SeaThru-NeRF 上报告：

```text
Underwater PSNR / SSIM / LPIPS
J Blue Dominance
J Green Dominance
Water J leakage
Object J retention
Boundary retention
Cross-view J consistency
Camera-path temporal color consistency
```

定性对比：

```text
M1 Full SH
M1 DC only
Best Dual-Color
SeaFree-GS visual reference
```

---

## 13. 多随机种子验证

第一轮统一使用：

```text
seed = 42
```

满足以下门槛的配置补：

```text
seed = 123
seed = 3407
```

门槛：

```text
Underwater PSNR drop <= 0.05 dB
LPIPS increase <= 0.001
J Blue/Green Dominance 明显下降
FUNA clear LPIPS 改善
Object/Boundary retention 不下降
```

最终报告：

```text
mean ± std
```

---

## 14. 备选模块：Multi-View Surface-Support Gaussian Routing

仅当 A1/A2 中仍存在明显蓝色漂浮结构时启动。

### 14.1 几何支持

\[
s_i^{\mathrm{geo}}
=
s_i^{\mathrm{visibility}}
\cdot
s_i^{\mathrm{depth}}
\cdot
s_i^{\mathrm{reprojection}}.
\]

包括：

```text
多视角可见次数
投影深度一致性
局部重投影一致性
开放水体区域曝光比例
```

### 14.2 第一阶段只限制 densification

对满足：

```text
low multi-view support
high water-region exposure
low object-region support
```

的 Gaussian：

```text
禁止 split
禁止 duplicate
```

不执行 opacity decay 或 hard prune。

若该模块只减少 Gaussian count，而不改善 clear residual 或 RGB 指标，则停止开发。

---

## 15. 代码修改建议

### 15.1 `water_splatting/fields/gaussian_appearance.py`

新增：

```python
@dataclass
class DualColorOutput:
    intrinsic_rgb: Tensor
    underwater_rgb: Tensor
    view_residual: Tensor
    luminance_residual: Tensor
    chroma_residual: Tensor
```

新增：

```python
compute_dual_gaussian_colors(...)
```

### 15.2 `water_splatting/rendering/underwater_rasterizer.py`

增加输入：

```text
colors_underwater
colors_intrinsic
```

增加输出：

```text
rgb_object_underwater
rgb_object_intrinsic
J_intrinsic
```

### 15.3 `water_splatting/water_splatting.py`

新增：

```python
dual_color_enabled: bool = False
clear_sh_luminance_scale: float = 1.0
clear_sh_chroma_scale: float = 0.0
lambda_intrinsic_near_anchor: float = 0.0
lambda_view_residual_mean: float = 0.0
lambda_clear_chroma: float = 0.0
dual_color_freeze_geometry: bool = True
dual_color_freeze_medium: bool = True
```

### 15.4 新增脚本

```text
scripts/diagnostics/diagnose_dc_sh_clear_appearance.py
scripts/experiments/dual_color_stage1_frozen_geometry_iui3.sh
scripts/experiments/dual_color_stage2_joint_refine_iui3.sh
scripts/experiments/dual_color_funa_clear_gt.sh
```

---

## 16. 推荐执行顺序

### 第一批：最低成本诊断

1. A0 Full SH；
2. A1 DC only；
3. A2 DC + luminance；
4. A3 DC + 0.05 chroma；
5. A4 DC + 0.10 chroma。

### 第二批：冻结几何训练

6. C1；
7. C2；
8. C3；
9. C4；
10. C5；
11. C6。

### 第三批：联合微调

12. 最佳配置解冻 medium MLP；
13. 第二候选解冻 medium MLP；
14. 必要时低学习率全局微调。

### 第四批：FUNA 验证

15. Blue water；
16. Green water；
17. 深度分箱颜色误差；
18. 多 seed。

### 第五批：几何备选

19. 仅在 A1/A2 失败时启动 surface-support diagnostics；
20. densification blocking；
21. 决定是否继续。

---

## 17. 最终决策路径

### 情况 A：Dual-Color 成功

最终方法：

```text
M1 Context-Aware Medium Modeling
+
Intrinsic–Underwater Dual-Color Gaussian Appearance
```

### 情况 B：颜色改善但仍有蓝色漂浮结构

最终候选：

```text
M1
+
Dual-Color Appearance
+
Lightweight Multi-View Surface Support
```

### 情况 C：Dual-Color 对 clear GT 无改善

重新审查：

```text
medium field parameterization
attenuation/backscatter identifiability
training supervision
synthetic-to-real gap
```

不再继续围绕 Gaussian 清理做局部优化。

---

## 18. 成功标准

### 带水体重建

相对 M1：

```text
PSNR drop <= 0.05 dB
SSIM 不下降超过 0.001
LPIPS increase <= 0.001
```

### 真实数据去水体效果

```text
J Blue Dominance 明显下降
J Green Dominance 明显下降
Water J leakage 不增加
Object J retention >= 0.97
Boundary retention >= 0.95
```

### FUNA clear GT

```text
Clear PSNR 提升
Clear LPIPS 降低
Far-region CIEDE2000 降低
不同水色和等级下均有一致收益
```

### 稳定性

```text
PSNR std <= 0.10 dB
LPIPS std <= 0.0015
```

---

## 19. 最终建议

当前 M2 已完成充分的负结果探索，应停止继续增加 ownership、capacity 与 hit-protection 机制。下一阶段应以 M1 为可靠基础，将研究重点从：

```text
删除或压制远处 Gaussian
```

转向：

```text
分离 Gaussian 的固有颜色与水下外观职责
```

这条路线更有可能同时改善去水体颜色、远处蓝绿色残留和带水体新视角重建指标。
