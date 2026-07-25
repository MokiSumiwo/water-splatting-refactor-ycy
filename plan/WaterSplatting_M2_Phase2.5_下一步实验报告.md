# WaterSplatting M2 下一阶段实验报告
## Phase 2.5：Object-Retention-Calibrated Capacity Attribution

## 1. 实验背景

当前项目已完成 M2 第二阶段主体闭环，包括固定公共远区 mask、显式随机种子、多 seed 稳定性实验、accumulation 权重局部搜索、ownership 模式消融、depth evidence 软化、CUDA hit-aware depth 统计、`m_support`、`m_render`、`m_capacity` 输出，以及第一轮 hit-aware capacity-support 实验。

当前代码分支为：

```text
refactor/core-framework
```

当前第二阶段主要提交为：

```text
781a656a31bafb2f8774d59f9b16b71335020fd1
```

对应实验记录为：

```text
research_notes/M2_PHASE2_HITAWARE_DEPTH_2026-07-24.md
```

第二阶段实验已证明：

1. 固定公共 mask 后，第一阶段 E2 对远区残留的改善仍然成立，但其效果没有按模型自身 depth-q90 统计时那么显著。
2. `lambda_accumulation_zero=0.002` 相比 `0.0005` 更适合容量抑制，但多 seed 稳定性仍然不足。
3. `alpha_depth` 对远区 leakage 的控制强于 `alpha_only`，说明 depth evidence 仍然必要。
4. D2，即 `depth_mid=0.75, depth_temp=0.15`，是当前重建质量较好的候选，但并非远区残留控制最强的候选。
5. hit confidence 在可视化上能够区分开放水体和可见场景表面，但直接使用 `1-q_hit` 减弱 accumulation pressure 会造成 leakage rebound。
6. 当前 hit-aware capacity 失败的主要原因不是 hit-aware depth 完全无效，而是动态 hit confidence 与 Gaussian accumulation 之间形成了自保护反馈。
7. 现阶段尚不能直接进入 pseudo-depth teacher 训练、Gaussian hard pruning 或最终 closed-tail rendering。

因此，下一阶段应围绕以下核心问题展开：

\[
\boxed{
\text{如何利用 hit-aware depth 保护真实物体，
但不让错误水体 Gaussian 获得自我保护}
}
\]

---

# 2. 第二阶段结果总结

## 2.1 固定公共 mask 修正了第一阶段结论

在固定 M1 depth-q90 mask 下：

| 配置 | PSNR | SSIM | LPIPS | Common Far Accum | Common Far Clear |
|---|---:|---:|---:|---:|---:|
| M1 | 31.1314 | 0.9120 | 0.1750 | 0.407096 | 0.083962 |
| Old M2 | 31.0696 | 0.9129 | 0.1771 | 0.119960 | 0.054452 |
| E2 old unseeded | 31.2206 | 0.9140 | 0.1765 | 0.185775 | 0.069223 |

这说明 E2 的真实效果是：

- 相比 M1，远区容量和 clear leakage 均有所下降；
- 相比 Old M2，重建指标更好，但 leakage 控制较弱；
- 不能再将 E2 描述为“几乎完全清除了远水 Gaussian”。

## 2.2 多 seed 稳定性不足

| 配置 | PSNR mean ± std | LPIPS mean ± std | Common Far Accum | Common Far Clear |
|---|---:|---:|---:|---:|
| R05 `accum=0.0005` | 30.9883 ± 0.2023 | 0.1752 ± 0.0005 | 0.3049 ± 0.0219 | 0.0872 ± 0.0084 |
| R20 `accum=0.0020` | 30.9378 ± 0.1106 | 0.1757 ± 0.0007 | 0.2107 ± 0.0276 | 0.0663 ± 0.0096 |

结论：

- R20 的 leakage 控制优于 R05；
- 两组 PSNR 方差均高于预设稳定性门槛；
- 单次最高结果不能作为最终方法依据；
- 后续所有关键候选必须至少补充 3 个 seeds。

## 2.3 Depth evidence 仍然必要

ownership 消融表明：

| Ownership | PSNR | LPIPS | Common Far Accum | Common Far Clear |
|---|---:|---:|---:|---:|
| `alpha_only` | 31.0093 | 0.1748 | 0.231339 | 0.084730 |
| `alpha_depth` | 30.9408 | 0.1791 | 0.147092 | 0.053148 |
| `alpha_depth_color` | 31.1162 | 0.1760 | 0.230851 | 0.070352 |

`alpha_only` 无法充分区分低 accumulation 的真实远景与开放水体；`alpha_depth_color` 在当前场景中对蓝绿色物体存在潜在误判风险。因此，后续仍以 `alpha_depth` 为基础。

## 2.4 D2 是质量候选，不是最终容量候选

| 配置 | depth_mid | depth_temp | PSNR | LPIPS | Common Far Accum | Common Far Clear |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.75 | 0.10 | 31.1190 | 0.1792 | 0.146410 | 0.052388 |
| D2 | 0.75 | 0.15 | 31.2212 | 0.1758 | 0.199784 | 0.070869 |

D2 通过软化 depth sigmoid 改善了重建质量，但同时扩大了远区 residual。下一阶段将 D2 作为质量优先基线，将 D0 作为 leakage-control 参考。

## 2.5 当前 Hit-Aware Capacity 未通过验证

| Support Mode | PSNR | LPIPS | Common Far Accum | Common Far Clear |
|---|---:|---:|---:|---:|
| H0 `m_inf` | 31.0151 | 0.1752 | 0.200738 | 0.064587 |
| H1 `m_inf(1-q_alpha)` | 30.9388 | 0.1755 | 0.315076 | 0.095148 |
| H2 `m_inf(1-q_hit)` | 30.9496 | 0.1759 | 0.252780 | 0.070761 |
| H3 `m_inf(1-q_hit)^2` | 31.1083 | 0.1768 | 0.246008 | 0.073644 |

H2/H3 相比 H0 均出现 leakage rebound，说明直接以 `1-q_hit` 减弱 capacity pressure 不可行。

---

# 3. 当前机制问题分析

## 3.1 动态 Hit Confidence 存在自保护反馈

当前定义：

\[
q_{\alpha}
=
\sigma
\left(
\frac{A-\tau_A}{t_A}
\right),
\]

\[
q_{\mathrm{conc}}
=
\exp
\left(
-\frac{r_d}{\kappa}
\right),
\]

\[
q_{\mathrm{hit}}
=
q_{\alpha}q_{\mathrm{conc}}.
\]

当前 capacity mask 为：

\[
M_{\mathrm{capacity}}
=
M_{\infty}(1-q_{\mathrm{hit}}).
\]

这会形成：

\[
A\uparrow
\Rightarrow
q_{\mathrm{hit}}\uparrow
\Rightarrow
M_{\mathrm{capacity}}\downarrow
\Rightarrow
\mathcal L_{\mathrm{acc}}\downarrow
\Rightarrow
A\text{继续保留}.
\]

因此，一部分错误远水 Gaussian 只要形成较集中的深度贡献，就可能被判定为可靠 hit，从而减少自身受到的 accumulation penalty。

## 3.2 Depth concentration 不等于真实物体支持

相对深度方差较小只能说明贡献集中在某一深度，并不能证明该贡献来自稳定场景表面。单个漂浮 Gaussian 或小簇错误 Gaussian 同样可能产生：

\[
r_d\approx 0,
\qquad
q_{\mathrm{conc}}\approx 1.
\]

因此，`q_hit` 适合成为潜在物体证据，但不能直接作为取消 capacity pressure 的充分条件。

## 3.3 目前缺少显式 Object-Retention 评价

当前固定 common far mask 只能衡量远区总 residual，不能回答：

- H2 是否保护了真实海床；
- H2 增加的 accumulation 是物体恢复还是水体残留；
- 边界区域是否得到改善；
- 高 `q_hit` 区域是否真正对应场景表面。

因此，下一阶段必须建立 water、object 和 boundary 三类固定评价区域。

---

# 4. 下一阶段总体目标

下一阶段命名为：

## Phase 2.5：Object-Retention-Calibrated Capacity Attribution

主要目标包括：

### 目标 1：验证 D2 的多 seed 稳定性

确认 D2 的高 PSNR 是稳定收益还是单次训练波动。

### 目标 2：建立真实物体保留指标

区分：

```text
Water leakage reduction
Object retention
Boundary preservation
```

### 目标 3：校准 hit confidence

确定 `q_hit` 在真实物体和开放水体区域中的分布，寻找高 precision 的物体保护阈值。

### 目标 4：引入 capacity floor

使 hit-aware protection 只能减弱容量约束，不能完全取消容量约束。

### 目标 5：比较 dynamic hit 与 frozen hit

判断 H2 失败是否主要来自动态自保护反馈。

### 目标 6：决定 hit-aware depth 的最终用途

根据实验结果决定：

```text
保留在 capacity control 中
或
仅用于 closed-tail depth / scene-medium routing
```

---

# 5. Phase A：D2 多 Seed 稳定性验证

## 5.1 固定配置

```text
medium_context_mode = dir_xy_camera
ownership_mode = alpha_depth
compose_mode = rgb_mix
occupancy_limited = True
lambda_binf_rgb = 0.005
lambda_accumulation_zero = 0.002
lambda_near_zero = 0
loss_start_step = 1000
loss_ramp_steps = 3000
depth_mid = 0.75
depth_temp = 0.15
max_iterations = 15000
```

## 5.2 实验矩阵

| 编号 | depth_mid | depth_temp | Seed |
|---|---:|---:|---:|
| D2-S42 | 0.75 | 0.15 | 42 |
| D2-S123 | 0.75 | 0.15 | 123 |
| D2-S3407 | 0.75 | 0.15 | 3407 |

D0 作为参考：

| 编号 | depth_mid | depth_temp | Seed |
|---|---:|---:|---:|
| D0-S42 | 0.75 | 0.10 | 42 |
| D0-S123 | 0.75 | 0.10 | 123 |
| D0-S3407 | 0.75 | 0.10 | 3407 |

## 5.3 统计指标

```text
PSNR mean ± std
SSIM mean ± std
LPIPS mean ± std
J blue dominance mean ± std
common far accumulation mean ± std
common far clear leakage mean ± std
Gaussian count mean ± std
```

## 5.4 决策规则

若 D2 满足：

```text
PSNR mean > D0 mean
LPIPS mean <= D0 mean
PSNR std <= 0.10 dB
Common far clear 不高于 D0 的 1.4 倍
```

则使用 D2 作为 Phase 2.5 主基线。

否则使用 D0 或 D0/D2 的折中配置：

```text
depth_temp = 0.125
```

补充一次局部实验。

---

# 6. Phase B：建立 Object/Water/Boundary 固定评价区域

## 6.1 评价 mask

对 Eval view 0000 至 0003 建立三类 mask。

### 开放水体区域

\[
M_{\mathrm{water}}
\]

只标注明确没有场景物体的水体区域。

### 真实物体区域

\[
M_{\mathrm{object}}
\]

包括：

```text
海床
珊瑚
礁石
颜色板
其他明确场景表面
```

### 物体边界区域

\[
M_{\mathrm{boundary}}
\]

从 object mask 边界膨胀 3 至 7 像素获得。

第一轮采用人工高置信标注，只用于评价，不用于训练。

## 6.2 新增指标

### 水体容量泄漏

\[
E_{\mathrm{water}}^{A}
=
\frac{
\sum_pM_{\mathrm{water}}(p)A(p)
}{
\sum_pM_{\mathrm{water}}(p)+\epsilon
}.
\]

### 水体去水体图像泄漏

\[
E_{\mathrm{water}}^{J}
=
\frac{
\sum_p
M_{\mathrm{water}}(p)
\operatorname{luma}(J(p))
}{
\sum_pM_{\mathrm{water}}(p)+\epsilon
}.
\]

### 物体 accumulation 保留率

\[
R_{\mathrm{obj}}^{A}
=
\frac{
\operatorname{mean}_{M_{\mathrm{object}}}
A_{\mathrm{candidate}}
}{
\operatorname{mean}_{M_{\mathrm{object}}}
A_{\mathrm{M1}}
+\epsilon
}.
\]

### 物体 clear-luma 保留率

\[
R_{\mathrm{obj}}^{J}
=
\frac{
\operatorname{mean}_{M_{\mathrm{object}}}
\operatorname{luma}(J_{\mathrm{candidate}})
}{
\operatorname{mean}_{M_{\mathrm{object}}}
\operatorname{luma}(J_{\mathrm{M1}})
+\epsilon
}.
\]

### 边界梯度保留

\[
R_{\mathrm{boundary}}^{\nabla J}
=
\frac{
\operatorname{mean}_{M_{\mathrm{boundary}}}
\|\nabla J_{\mathrm{candidate}}\|
}{
\operatorname{mean}_{M_{\mathrm{boundary}}}
\|\nabla J_{\mathrm{M1}}\|
+\epsilon
}.
\]

### 水体误保护率

统计：

```text
water pixels with q_hit > threshold
water pixels with q_alpha > threshold
water pixels with low relative depth std
```

## 6.3 输出图

每个视角输出：

```text
Water mask
Object mask
Boundary mask
Accumulation overlay
J leakage overlay
q_hit overlay
m_capacity overlay
```

---

# 7. Phase C：Hit Confidence 校准

## 7.1 统计分布

在固定 masks 中统计：

```text
q_alpha distribution in water
q_alpha distribution in object
q_conc distribution in water
q_conc distribution in object
q_hit distribution in water
q_hit distribution in object
```

输出：

```text
histogram
precision-recall curve
ROC curve
threshold table
```

## 7.2 高精度物体保护

定义：

\[
P_{\mathrm{obj}}
=
\sigma
\left(
\frac{q_{\mathrm{hit}}-\tau_h}{t_h}
\right).
\]

建议：

```text
tau_h ∈ {0.60, 0.70, 0.80}
t_h = 0.05
```

选择原则：

```text
Object precision 优先
Water false positive rate < 5%
不追求最高 object recall
```

若 `tau_h=0.80` 仍无法保证高 precision，则不再将动态 `q_hit` 用于 capacity control。

---

# 8. Phase D：Conservative Hit-Aware Capacity Floor

## 8.1 新容量公式

不再使用：

\[
M_{\mathrm{capacity}}
=
M_{\infty}(1-q_{\mathrm{hit}}).
\]

改为：

\[
M_{\mathrm{capacity}}
=
M_{\mathrm{support}}
\left[
1-(1-f_{\min})P_{\mathrm{obj}}
\right].
\]

其中：

- \(P_{\mathrm{obj}}\) 是高精度物体保护概率；
- \(f_{\min}\) 是容量约束下限；
- 即使高置信物体区域也保留一定 accumulation pressure。

## 8.2 实验矩阵

| 编号 | \(\tau_h\) | \(f_{\min}\) | 说明 |
|---|---:|---:|---|
| C0 | 不启用 | 1.00 | H0 基线 |
| C1 | 0.60 | 0.50 | 中等保护 |
| C2 | 0.80 | 0.50 | 高精度保护 |
| C3 | 0.80 | 0.75 | 保守保护 |
| C4 | 0.80 | 0.25 | 强保护，仅诊断 |

首先统一使用 seed 42。

只有优于 C0 的配置补充：

```text
seed 123
seed 3407
```

## 8.3 建议配置字段

新增：

```python
infinite_water_hit_protection_enabled: bool = False
infinite_water_hit_protection_threshold: float = 0.80
infinite_water_hit_protection_temp: float = 0.05
infinite_water_capacity_floor: float = 0.50
```

## 8.4 成功标准

相对 C0：

```text
PSNR drop <= 0.02 dB
LPIPS increase <= 0.0005
Water clear leakage increase <= 5%
Object accumulation retention 提升
Boundary gradient retention 提升
```

---

# 9. Phase E：Dynamic Hit 与 Frozen Hit 对照

## 9.1 实验动机

动态 hit confidence 来自当前正在训练的模型，因此错误 Gaussian 可以通过提高 accumulation 获得保护。Frozen hit 使用 M1 或稳定预训练模型生成固定 reference，不随当前 M2 变化。

## 9.2 Frozen Hit 数据生成

使用 M1 checkpoint 为每个训练视图生成：

```text
q_hit_ref
q_alpha_ref
q_conc_ref
depth_expected_ref
object_support_ref
```

保存格式建议：

```text
hit_reference/train/view_xxxx.pt
```

需要保存：

```text
source checkpoint
source commit
camera filename
resolution
normalization information
```

## 9.3 接入方式

通过 datamanager 将 reference map 加载为：

```python
batch["hit_confidence_ref"]
```

禁止在模型中使用绝对路径直接读取。

## 9.4 实验矩阵

| 编号 | Hit Source | \(\tau_h\) | \(f_{\min}\) |
|---|---|---:|---:|
| F0 | Dynamic | 0.80 | 0.50 |
| F1 | Frozen M1 | 0.80 | 0.50 |
| F2 | Frozen M1 | 0.80 | 0.75 |
| F3 | Frozen D2 | 0.80 | 0.50 |

## 9.5 判断

若 Frozen Hit 明显优于 Dynamic Hit，则说明：

```text
hit-aware depth 机制有效
动态自保护是主要失败原因
```

若 Frozen Hit 仍然没有改善 object retention，则说明当前 `q_hit` 缺少多视角几何真实性，不适合 capacity control。

---

# 10. Phase F：延迟启用 Hit Protection

第一阶段 E3 证明不应延迟整个 M2，但可以延迟物体保护门。

定义：

```python
if step < protection_start_step:
    m_capacity = m_support
else:
    m_capacity = floor_gated_support
```

测试：

```text
protection_start_step ∈ {3000, 5000, 7000}
```

推荐优先：

```text
5000
```

目的：

- 训练早期保持完整 accumulation pressure；
- 防止错误远水 Gaussian 先建立高 accumulation；
- 中后期再保护稳定物体表面。

---

# 11. Phase G：决定 Hit-Aware Depth 的最终用途

## 11.1 保留在 Capacity Control 中的条件

只有当最佳 hit-aware capacity 配置满足：

```text
多 seed 稳定
Water leakage 不高于 C0
Object retention 明显优于 C0
RGB 指标不低于 C0
```

才保留为 M2 capacity 模块。

## 11.2 转入 Closed-Tail Routing 的条件

若经过：

```text
capacity floor
high-threshold protection
frozen hit
delayed protection
```

后仍无法通过门槛，则停止使用 hit-aware depth 调节 accumulation loss。

此时将 hit-aware depth 仅用于：

```text
scene-medium routing
closure depth
tail background attribution
```

即：

\[
d_{\mathrm{close}}
=
q_{\mathrm{hit}}d_{\mathrm{exp}}
+
(1-q_{\mathrm{hit}})d_{\mathrm{far}}.
\]

---

# 12. Closed-Tail Rendering 的进入条件

只有完成 Phase A 至 Phase G 后，才进入 closed-tail CUDA 修改。

首先验证数值等价性：

\[
I_{\mathrm{old}}
\approx
I_{\mathrm{obj}}
+
I_{\mathrm{med}}^{\mathrm{finite}}
+
W_{\mathrm{tail}}B.
\]

必须显式输出：

```text
rgb_object
rgb_medium_finite
tail_weight
tail_medium_original
last_depth
expected_depth
hit_aware_close_depth
```

数值等价验证通过后，再测试：

\[
B_{\mathrm{tail}}
=
(1-M_{\mathrm{render}})B
+
M_{\mathrm{render}}B_{\infty},
\]

\[
I
=
I_{\mathrm{obj}}
+
I_{\mathrm{med}}^{\mathrm{finite}}
+
W_{\mathrm{tail}}B_{\mathrm{tail}}.
\]

---

# 13. 推荐实验顺序

## 第一批：必须完成

1. D2 seeds 123、3407；
2. D0 seeds 123、3407；
3. 建立 water/object/boundary masks；
4. 增加 object-retention diagnostics；
5. 统计 q-hit 在 water/object 区域中的分布。

## 第二批：Capacity Floor

6. C1；
7. C2；
8. C3；
9. 选择最优配置；
10. 对最优配置补充 seeds。

## 第三批：Frozen Hit

11. 生成 M1 frozen hit maps；
12. F1；
13. F2；
14. 与 dynamic hit 比较。

## 第四批：Protection Schedule

15. protection start 3000；
16. protection start 5000；
17. protection start 7000。

## 第五批：机制决策

18. 决定 hit-aware capacity 是否保留；
19. 若不保留，则将 hit-aware depth 转入 closed-tail routing；
20. 开始 closed-tail 数值等价改造。

---

# 14. 最小实验矩阵

| 编号 | 配置 | Seed | 目的 |
|---|---|---:|---|
| D2-S123 | D2 | 123 | 稳定性 |
| D2-S3407 | D2 | 3407 | 稳定性 |
| C0 | 无 hit protection | 42 | 对照 |
| C1 | \(\tau_h=0.6, f_{\min}=0.5\) | 42 | 中等保护 |
| C2 | \(\tau_h=0.8, f_{\min}=0.5\) | 42 | 高精度保护 |
| C3 | \(\tau_h=0.8, f_{\min}=0.75\) | 42 | 保守保护 |
| F1 | Frozen M1, C2 | 42 | 消除动态反馈 |
| F2 | Frozen M1, C3 | 42 | 保守冻结保护 |
| G5 | 最佳配置，start=5000 | 42 | 延迟保护 |
| BEST-123 | 最佳配置 | 123 | 复验 |
| BEST-3407 | 最佳配置 | 3407 | 复验 |

---

# 15. 代码修改位置

## 15.1 `water_splatting/water_splatting.py`

修改：

```python
_infinite_water_capacity_support()
```

新增：

```text
hit protection threshold
hit protection temperature
capacity floor
protection start step
hit source mode
```

推荐实现：

```python
def build_object_protection(q_hit, threshold, temperature):
    return torch.sigmoid(
        (q_hit - threshold) / max(temperature, 1e-6)
    )


def build_capacity_support(
    m_support,
    object_protection,
    capacity_floor,
):
    gate = 1.0 - (1.0 - capacity_floor) * object_protection
    return m_support * gate
```

## 15.2 Datamanager

新增可选字段：

```text
hit_confidence_ref
water_eval_mask
object_eval_mask
boundary_eval_mask
```

训练阶段只需要：

```text
hit_confidence_ref
```

人工评价 masks 不参与训练。

## 15.3 Diagnostic Script

扩展：

```text
scripts/diagnostics/diagnose_far_water_residual.py
```

增加：

```text
water accumulation
water clear leakage
object accumulation retention
object clear retention
boundary gradient retention
q_hit water false positive rate
q_hit object precision/recall
```

## 15.4 可视化

输出：

```text
RGB
J
Accum
m_support
m_capacity
q_hit
object protection
water mask
object mask
boundary mask
```

---

# 16. 结果表模板

| Experiment | Seed | PSNR | SSIM | LPIPS | Water Accum | Water Clear | Object Accum Ret. | Object J Ret. | Boundary Ret. | J Blue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 42 |  |  |  |  |  |  |  |  |  |
| C1 | 42 |  |  |  |  |  |  |  |  |  |
| C2 | 42 |  |  |  |  |  |  |  |  |  |
| C3 | 42 |  |  |  |  |  |  |  |  |  |
| F1 | 42 |  |  |  |  |  |  |  |  |  |
| F2 | 42 |  |  |  |  |  |  |  |  |  |

---

# 17. 停止条件

若 hit-aware capacity 在以下全部实验后仍未通过：

```text
高阈值保护
capacity floor
frozen reference
延迟 protection
多 seed 复验
```

则停止在 accumulation loss 中使用 hit-aware gate。

最终结论应写为：

\[
\boxed{
\text{Hit-aware depth 能够提供场景命中诊断，
但不适合直接取消 Gaussian 容量压力}
}
\]

随后将其用于：

```text
closed-tail closure depth
scene-medium attribution
background tail routing
```

---

# 18. 当前推荐临时主线

在 Phase 2.5 完成前，建议保留两套基线。

## 质量基线

```text
M1 dir_xy_camera
M2 alpha_depth
depth_mid = 0.75
depth_temp = 0.15
accum = 0.002
rgb_mix
occupancy_limited = True
```

## Leakage 基线

```text
M1 dir_xy_camera
M2 alpha_depth
depth_mid = 0.75
depth_temp = 0.10
accum = 0.002
rgb_mix
occupancy_limited = True
```

下一阶段不应只追求单一最高 PSNR，而应寻找：

\[
\boxed{
\text{低水体泄漏}
+
\text{高物体保留}
+
\text{稳定新视角指标}
}
\]

的 Pareto 最优配置。

---

# 19. 最终预期

若 Phase 2.5 成功，M2 将从：

```text
Low accumulation + far depth
→ global capacity suppression
```

升级为：

```text
Scene-medium support
→ calibrated object protection
→ conservative capacity control
```

最终机制应表达为：

\[
M_{\mathrm{capacity}}
=
M_{\mathrm{support}}
\left[
1-(1-f_{\min})P_{\mathrm{object}}
\right],
\]

其中 \(P_{\mathrm{object}}\) 必须具有高 precision，且 capacity pressure 始终保留非零下限。

该设计的目标不是简单清除更多 Gaussian，而是在不牺牲真实远景物体和新视角重建质量的情况下，抑制真正属于开放水体的错误 Gaussian 容量。
