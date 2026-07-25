# WaterSplatting M2 简化与闭合归属下一阶段实验方案

## 1. 实验背景

当前项目已经完成 M2 第一阶段、第二阶段以及 Phase 2.5 的连续实验。Phase 2.5 已完成 D0/D2 多随机种子稳定性验证、固定 water/object/boundary 评价区域、`q_hit` 校准、conservative capacity floor，以及 object retention、boundary retention 与 water leakage 诊断。

现阶段结论较为明确：

1. M2 的远水 Gaussian 抑制方向仍然有效，不宜直接舍弃。
2. D2 更偏向重建质量，D0 更偏向远区 leakage 控制，但二者都缺乏充分稳定的多 seed 优势。
3. `q_hit` 作为高精度物体证据有效，但动态用于容量保护时容易引入 self-protection。
4. Capacity floor 只能在重建质量、物体保留和远区清除之间移动 Pareto 点，没有形成全面稳定优于基线的配置。
5. 当前 M2 的机制复杂度增长已经快于性能收益。

因此，本阶段的核心目标是：

\[
\boxed{
\text{验证 M2 的必要组成}
+
\text{将容量损失改为单调预算}
+
\text{将 }B_\infty\text{ 限定到介质尾部}
}
\]

---

## 2. 当前 M2 的主要问题

### 2.1 当前 Capacity Loss 不是单调容量惩罚

当前 ownership 近似为：

\[
M_{\infty}=(1-A)^\gamma M_d,
\]

而 accumulation loss 为：

\[
\mathcal L_{\mathrm{acc}}
=
\frac{
\sum_p M_{\infty}(p)A(p)
}{
\sum_p M_{\infty}(p)+\epsilon
}.
\]

在默认 \(\gamma=1\) 时，其核心像素项为：

\[
M_dA(1-A).
\]

该项在 \(A\rightarrow0\) 和 \(A\rightarrow1\) 时都趋近于零，只在中间 accumulation 区间产生较强压力。因此，当错误远水 Gaussian 已形成较高 accumulation 时，它可能逐渐逃离容量惩罚。

### 2.2 Dynamic Hit Protection 容易产生自保护

当前 hit-aware capacity 依赖：

\[
q_{\mathrm{hit}}=q_{\alpha}q_{\mathrm{conc}},
\]

并通过 \(1-q_{\mathrm{hit}}\) 或 capacity floor 减小容量损失。由于 \(q_{\alpha}\) 本身依赖当前 accumulation，可能形成：

\[
A\uparrow
\Rightarrow
q_{\mathrm{hit}}\uparrow
\Rightarrow
\text{capacity pressure}\downarrow.
\]

因此，错误 Gaussian 有机会利用当前模型状态获得保护。

### 2.3 当前 RGB Composition 仍直接削弱完整渲染结果

当前 `rgb_mix` 采用：

\[
I=(1-M_{\mathrm{render}})I_{\mathrm{near}}+M_{\mathrm{render}}B_{\infty}.
\]

该公式会同时削弱 Gaussian object contribution、有限距离介质贡献和真实远景物体边界。

---

## 3. 本阶段实验目标

本阶段分为四个主任务：

1. 拆解 M2 的实际收益来源，分别验证 \(B_\infty\)、RGB composition 与 capacity loss。
2. 将 capacity loss 改为单调预算，避免高 accumulation 错误 Gaussian 逃离约束。
3. 缩减 hit-aware capacity 的角色，仅保留 hit-aware depth 用于诊断、scene-medium routing 与 closure depth。
4. 实现真正的 closed-tail rendering，使 \(B_\infty\) 只替换无限远介质尾部。

---

## 4. Phase 3A：M2 机制拆解实验

### 4.1 新增 `compose_mode=none`

建议在当前：

```text
rgb_mix
tail_approx
```

之外新增：

```text
none
```

在 `none` 模式下：

```python
rgb = render.rgb
rgb_clear = render.rgb_clear
j_object_raw = render.j_raw
```

但仍保留 ownership、\(B_\infty\) prediction、capacity loss 与 diagnostics。

### 4.2 2×2 机制矩阵

固定：

```text
medium_context_mode = dir_xy_camera
ownership_mode = alpha_depth
depth_mid = 0.75
depth_temp = 0.10
occupancy_limited = True
lambda_near_zero = 0
loss_start_step = 1000
loss_ramp_steps = 3000
max_iterations = 15000
```

| 编号 | \(B_\infty\) RGB loss | RGB composition | Capacity loss | 目的 |
|---|---:|---|---:|---|
| S0 | 关闭 | none | 关闭 | M1 对照 |
| S1 | 开启 | rgb_mix | 关闭 | 仅测试 \(B_\infty\) 合成 |
| S2 | 关闭 | none | 开启 | 仅测试容量抑制 |
| S3 | 开启 | rgb_mix | 开启 | 当前简化 M2 |
| S4 | 开启 | none | 开启 | 学习 \(B_\infty\)，但不参与 RGB 合成 |

建议参数：

```text
lambda_binf_rgb = 0.005
lambda_capacity = 0.002
```

### 4.3 关键判断

若 S1 改善 RGB，但不降低 `J_gaussian` residual，说明 \(B_\infty\) 主要解决输出合成，不能解决 Gaussian 表示污染。

若 S2 降低 clear leakage 且不损害 RGB，说明 capacity loss 可独立保留，RGB composition 不是必要条件。

若 S3 不优于 S1 或 S2，则 M2 不应继续将 \(B_\infty\) composition 与 capacity regularization 强绑定。

---

## 5. Phase 3B：单调 Capacity Budget

### 5.1 L0：无容量约束

\[
\mathcal L_{\mathrm{cap}}^{L0}=0.
\]

### 5.2 L1：当前形式

\[
\mathcal L_{\mathrm{cap}}^{L1}
=
\frac{
\sum_p M_d(p)(1-A(p))A(p)
}{
\sum_pM_d(p)(1-A(p))+\epsilon
}.
\]

### 5.3 L2：Depth-Only Monotonic Capacity

\[
\mathcal L_{\mathrm{cap}}^{L2}
=
\frac{
\sum_p M_d(p)A(p)
}{
\sum_pM_d(p)+\epsilon
}.
\]

该形式满足 accumulation 越高，惩罚越强，不会在 \(A\rightarrow1\) 时自动消失。

### 5.4 L3/L4：容量预算

\[
\mathcal L_{\mathrm{cap}}^{\mathrm{budget}}
=
\frac{
\sum_p
M_d(p)
\operatorname{ReLU}[A(p)-A_0]
}{
\sum_pM_d(p)+\epsilon
}.
\]

测试：

```text
L3: A0 = 0.05
L4: A0 = 0.10
```

该设计允许远区保留少量真实物体容量，只对超过预算的 accumulation 施加压力。

### 5.5 可选 L5：Softplus Budget

若 L3/L4 在阈值附近表现不稳定，可补充：

\[
\mathcal L_{\mathrm{cap}}^{L5}
=
\frac{
\sum_p
M_d(p)
\operatorname{softplus}
\left(
\frac{A(p)-A_0}{t_A}
\right)t_A
}{
\sum_pM_d(p)+\epsilon
}.
\]

建议：

```text
A0 = 0.05
t_A = 0.02
```

### 5.6 实验矩阵

| 编号 | Capacity loss | Capacity threshold |
|---|---|---:|
| L0 | none | n/a |
| L1 | current alpha-depth | n/a |
| L2 | depth-only monotonic | 0 |
| L3 | ReLU budget | 0.05 |
| L4 | ReLU budget | 0.10 |
| L5 | softplus budget | 0.05 |

第一轮统一使用：

```text
seed = 42
compose_mode = none
lambda_binf_rgb = 0
lambda_capacity = 0.002
```

只有进入 Pareto 前沿的配置再补：

```text
seed = 123
seed = 3407
```

---

## 6. Phase 3C：Capacity Weight 局部搜索

对 Phase 3B 最优的两种 loss 形式进行：

```text
lambda_capacity ∈ {0.0005, 0.001, 0.002}
```

不建议继续使用 0.004，因为第一阶段已表现出较强的重建损伤趋势。

---

## 7. Phase 3D：Hit-Aware 简化决策

### 7.1 主方法中关闭 Dynamic Protection

默认：

```text
infinite_water_hit_protection_enabled = False
infinite_water_capacity_support_mode = m_inf
```

### 7.2 保留 Hit-Aware Outputs

继续输出：

```text
depth_expected
depth_variance
depth_std_relative
first_depth
last_depth
final_transmittance
q_alpha
q_conc
q_hit
```

用于诊断、物体/水体路由和 closed-tail depth。

### 7.3 Frozen Hit 只做一次诊断实验

| 编号 | Capacity | Hit source |
|---|---|---|
| F0 | C2 dynamic floor | dynamic |
| F1 | C2 frozen reference | M1 frozen |
| F2 | 最佳简化 budget | none |

只有当 F1 显著优于 F0 且显著优于 F2 时，才考虑 frozen reference；否则保留为负结果分析。

---

## 8. Phase 3E：评价体系

### 8.1 新视角重建

```text
PSNR ↑
SSIM ↑
LPIPS ↓
```

### 8.2 Water Leakage

\[
E_{\mathrm{water}}^{A}
=
\operatorname{mean}_{M_{\mathrm{water}}}A,
\]

\[
E_{\mathrm{water}}^{J}
=
\operatorname{mean}_{M_{\mathrm{water}}}
\operatorname{luma}(J).
\]

### 8.3 Object Retention

\[
R_{\mathrm{obj}}^{A}
=
\frac{
\operatorname{mean}_{M_{\mathrm{object}}}A_{\mathrm{candidate}}
}{
\operatorname{mean}_{M_{\mathrm{object}}}A_{\mathrm{M1}}+\epsilon
},
\]

\[
R_{\mathrm{obj}}^{J}
=
\frac{
\operatorname{mean}_{M_{\mathrm{object}}}
\operatorname{luma}(J_{\mathrm{candidate}})
}{
\operatorname{mean}_{M_{\mathrm{object}}}
\operatorname{luma}(J_{\mathrm{M1}})+\epsilon
}.
\]

### 8.4 Boundary Retention

\[
R_{\mathrm{boundary}}
=
\frac{
\operatorname{mean}_{M_{\mathrm{boundary}}}
\|\nabla J_{\mathrm{candidate}}\|
}{
\operatorname{mean}_{M_{\mathrm{boundary}}}
\|\nabla J_{\mathrm{M1}}\|+\epsilon
}.
\]

### 8.5 其他指标

继续报告：

```text
Common Far Accum
Common Far Clear
J Blue Dominance
Gaussian Count
mean ± std across seeds
```

稳定性门槛：

```text
PSNR std <= 0.10 dB
LPIPS std <= 0.0015
```

---

## 9. Phase 3F：候选筛选规则

### 重建质量

相对 M1：

```text
PSNR drop <= 0.05 dB
SSIM 不下降超过 0.001
LPIPS increase <= 0.001
```

### 水体泄漏

```text
Water Accum reduction >= 50%
Water J reduction >= 50%
```

### 物体保留

```text
Object Acc Ret >= 0.85
Object J Ret >= 0.95
Boundary Ret >= 0.95
```

若没有配置全部通过，则使用 Pareto 选择，不再继续增加复杂模块。

---

## 10. Phase 4：Closed-Tail Rendering

只有完成 Phase 3 的简化 capacity 选择后，才进入 Closed Tail。

### 10.1 CUDA 输出拆分

新增：

```text
rgb_medium_finite
tail_weight_last
tail_medium_original
```

当前介质项应验证为：

\[
I_{\mathrm{med}}^{\mathrm{old}}
=
I_{\mathrm{med}}^{\mathrm{finite}}
+
W_{\mathrm{tail}}^{\mathrm{last}}B.
\]

### 10.2 数值等价性实验

| 编号 | 实现 |
|---|---|
| T0 | 当前 renderer |
| T1 | finite medium + original tail 重组 |

要求：

```text
max absolute difference < 1e-5
mean absolute difference < 1e-6
```

未通过前禁止修改 tail color。

### 10.3 Closed \(B_\infty\) Tail

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

该公式不会直接对完整 object render 做 alpha mix。

### 10.4 Closure Depth 消融

| 编号 | Closure depth |
|---|---|
| T2 | last depth |
| T3 | expected depth |
| T4 | hit-aware depth |

Hit-aware closure：

\[
d_{\mathrm{close}}
=
q_{\mathrm{hit}}d_{\mathrm{exp}}
+
(1-q_{\mathrm{hit}})d_{\mathrm{far}}.
\]

在这一阶段，`q_hit` 不再作用于 capacity loss，只影响介质尾部闭合深度。

---

## 11. 推荐执行顺序

### 第一批：机制拆解

1. 新增 `compose_mode=none`；
2. 完成 S0 至 S4。

### 第二批：简化 Capacity

3. 完成 L0 至 L4；
4. 必要时补 L5。

### 第三批：权重搜索

5. 对最优两种 loss 搜索 0.0005、0.001、0.002；
6. 选择 Pareto 前沿；
7. 补 seeds 123、3407。

### 第四批：Frozen Hit 诊断

8. Dynamic C2；
9. Frozen-M1 C2；
10. 与最佳简化 budget 对比。

### 第五批：Closed Tail

11. 输出 finite medium 与 tail weight；
12. T0/T1 数值等价验证；
13. T2；
14. T3；
15. T4。

---

## 12. 最小实验矩阵

| 编号 | Compose | \(B_\infty\) loss | Capacity | Seed | 优先级 |
|---|---|---:|---|---:|---:|
| S0 | none | 0 | none | 42 | 最高 |
| S1 | rgb_mix | 0.005 | none | 42 | 最高 |
| S2 | none | 0 | current | 42 | 最高 |
| S3 | rgb_mix | 0.005 | current | 42 | 最高 |
| L2 | none | 0 | depth monotonic | 42 | 最高 |
| L3 | none | 0 | budget 0.05 | 42 | 最高 |
| L4 | none | 0 | budget 0.10 | 42 | 高 |
| F1 | none/rgb_mix | 0.005 | frozen C2 | 42 | 中 |
| T1 | closed-tail recompose | 0 | none | eval | 高 |
| T4 | closed-tail | 0.005 | best simple budget | 42 | 高 |

---

## 13. 代码修改建议

### 13.1 `water_splatting/water_splatting.py`

新增：

```python
infinite_water_compose_mode: Literal[
    "none",
    "rgb_mix",
    "tail_approx",
    "closed_tail",
]
```

新增：

```python
infinite_water_capacity_loss_mode: Literal[
    "none",
    "current",
    "depth_monotonic",
    "relu_budget",
    "softplus_budget",
]
```

新增：

```python
infinite_water_capacity_budget: float = 0.05
infinite_water_capacity_budget_temp: float = 0.02
```

### 13.2 Capacity Support

将 ownership 与 capacity weight 分开：

```python
m_support = alpha_evidence * depth_evidence
m_render = m_support * occupancy_gate
depth_support = depth_evidence.detach()
```

简化 capacity loss 中不要再次将 `(1 - accumulation)` 乘入 support。

### 13.3 Loss 实现

```python
if mode == "depth_monotonic":
    penalty = accumulation

elif mode == "relu_budget":
    penalty = torch.relu(accumulation - budget)

elif mode == "softplus_budget":
    penalty = F.softplus(
        (accumulation - budget) / temp
    ) * temp
```

最终：

```python
loss = weight * (
    depth_support * penalty
).sum() / depth_support.sum().clamp_min(1e-6)
```

### 13.4 实验脚本

新增：

```text
scripts/experiments/m2_phase3_mechanism_decomposition_iui3_redsea.sh
scripts/experiments/m2_phase3_capacity_budget_iui3_redsea.sh
scripts/experiments/m2_phase4_closed_tail_iui3_redsea.sh
```

---

## 14. 最终决策路径

### 情况 A：简化 Capacity 成功

最终 M2 为：

```text
Alpha-depth far support
+ monotonic capacity budget
+ closed B_inf tail
```

### 情况 B：Capacity 不稳定，但 Closed Tail 成功

M2 简化为：

```text
Alpha-depth scene-medium routing
+ closed B_inf tail
```

Capacity loss 降级为辅助正则或从主方法移除。

### 情况 C：Closed Tail 也无稳定收益

保留 M1，舍弃 M2 作为独立创新，仅将：

```text
B_inf
hit-aware diagnostics
far-water analysis
```

保留为实验分析或附录内容。

---

## 15. 预期最终方法

理想的简化 M2 为：

\[
\boxed{
\text{Far-Water Support}
+
\text{Monotonic Capacity Budget}
+
\text{Closed Medium Tail}
}
\]

其中：

- far-water support 只使用低复杂度的 depth/occupancy evidence；
- capacity budget 对远区 accumulation 施加单调约束；
- \(B_\infty\) 只进入介质尾部；
- hit-aware depth 只用于 closure 和诊断；
- 不再引入动态 hit protection、复杂 capacity floor 和 hard pruning。

本阶段的核心原则是：

\[
\boxed{
\text{先证明每个组件必要，再保留最小充分结构}
}
\]
