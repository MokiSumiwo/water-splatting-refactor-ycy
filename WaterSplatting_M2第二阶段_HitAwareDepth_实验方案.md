# WaterSplatting M2 第二阶段实验方案：稳定性验证、Hit-Aware Depth 与闭合尾部归属

## 1. 方案依据与代码快照

本方案基于以下代码与实验状态制定：

- 仓库：`MokiSumiwo/water-splatting-refactor-ycy`
- 分支：`refactor/core-framework`
- 当前审计提交：`c6260624b6c7f9a355092a07d7d46c20fc3ec8d5`
- 第一阶段记录：`research_notes/M2_REFINEMENT_E1_E4_2026-07-24.md`
- 当前主线：M1 `medium_context_mode=dir_xy_camera`
- 当前第一阶段候选：E2 `lambda_accumulation_zero=0.002`

第一阶段已经完成 E1 至 E4，主要结论如下：

1. M1 仍然是稳定主线。
2. M2 的方向成立，能够显著减少远水区域中的 Gaussian 容量和可见残留。
3. `lambda_accumulation_zero` 是当前最敏感的控制变量。
4. E2 `accum=0.0005` 的 LPIPS 最好，E2 `accum=0.002` 的 PSNR、SSIM 和远区抑制更均衡。
5. 延迟到 5000 step 才启用 M2 会降低重建质量，因此不再继续搜索晚启动策略。
6. 当前 `tail_approx` 不能代表真正的 closed-tail rendering，不应根据 E4 结果否定闭合尾部机制。
7. 下一阶段的核心不再是扩大 loss 权重搜索，而是提高 ownership 的准确性，尤其要区分纯水体射线与具有可靠远景物体命中的射线。

---

## 2. 当前代码审计结论

### 2.1 当前 ownership 只有一套像素级 mask

文件：

```text
water_splatting/ownership/infinite_water_ownership.py
```

当前输出为：

```python
m_inf
m_inf_eff
alpha_evidence
depth_evidence
color_evidence
```

其中：

\[
M_{\infty}^{\alpha}
=
(1-A)^\gamma,
\]

\[
M_{\infty}^{d}
=
\sigma
\left(
\frac{\tilde d-\mu_d}{\tau_d}
\right),
\]

`alpha_depth` 模式为：

\[
M_{\infty}
=
M_{\infty}^{\alpha}
M_{\infty}^{d}.
\]

若开启 `infinite_water_occupancy_limited`，合成 mask 为：

\[
M_{\infty}^{\mathrm{eff}}
=
M_{\infty}(1-A).
\]

当前单一 `m_inf` 同时承担以下任务：

- \(B_\infty\) RGB 监督区域；
- accumulation-zero loss 的支持区域；
- near-zero loss 的支持区域；
- RGB composition 的归属基础；
- 后续 Gaussian cleanup 的像素证据。

这些任务所需的 precision、recall 和边界保护强度并不相同，因此单一 mask 会造成相互牵制。

### 2.2 occupancy-limited 当前只影响合成，不影响 M2 辅助损失

文件：

```text
water_splatting/water_splatting.py
```

当前三个 M2 辅助损失统一使用：

```python
support = outputs["m_inf"].detach()
```

而不是 `m_inf_eff`。因此：

```text
infinite_water_occupancy_limited=True/False
```

只会改变 RGB composition 和 cleanup ownership，不会改变：

- `infinite_water_binf_rgb_loss`
- `infinite_water_accumulation_zero_loss`
- `infinite_water_near_zero_loss`

后续 ownership 消融必须将“loss support”和“render support”分开记录，否则 occupancy gate 的实验解释会不完整。

### 2.3 当前 depth 是 expected depth，但无命中区域被填为当前视图最大深度

CUDA 中计算：

\[
m_1(p)
=
\sum_i
T_i(p)\alpha_i(p)z_i.
\]

Python wrapper 中使用：

\[
d_{\mathrm{exp}}(p)
=
\frac{m_1(p)}{A(p)},
\qquad
A(p)=\sum_iT_i\alpha_i.
\]

对于 \(A=0\) 的像素，当前代码使用该视图的最大深度进行填充。因此：

- 纯水体无命中区域会自然获得最大深度；
- 低 accumulation 的伪命中区域也可能获得很大的 expected depth；
- 真实远处薄结构和纯水体容易被 `alpha_depth` 混淆；
- 当前 depth evidence 不能表达“是否真的命中了一个稳定表面”。

这正是引入 hit-aware depth 的直接理由。

### 2.4 当前 `tail_approx` 减去的不是实际介质尾部

当前 `tail_approx` 为：

```python
tail_gate = (1.0 - render.accumulation).detach()
rgb = render.rgb + m_inf * tail_gate * (b_inf - medium_rgb)
```

但 CUDA 中总介质贡献实际为：

\[
I_{\mathrm{med}}
=
I_{\mathrm{med}}^{\mathrm{finite}}
+
T_{\mathrm{end}}
\exp(-\beta^{B}d_{\mathrm{last}})
B,
\]

其中：

- `pix_medium` 是各 Gaussian 深度区间之间积累的有限介质贡献；
- `T` 是最终 object transmittance；
- `prev_depth` 是最后一个有效 Gaussian 的深度；
- 当前 `out_med` 已经将 finite medium 与 tail medium 合并。

因此，`medium_rgb` 只是水体颜色参数，不等于实际 tail contribution。当前 E4 只能说明这种 Python 近似不足，不能说明 closed-tail rendering 无效。

### 2.5 当前 far-water diagnostic 的跨模型可比性不足

文件：

```text
scripts/diagnostics/diagnose_far_water_residual.py
```

当前每个模型分别根据自身输出 depth 的 90% 分位数构建：

```python
far_mask = depth >= quantile(depth, 0.90)
```

这会造成不同模型使用不同的 far pixels。由于 M2 会改变 accumulation 和 depth，以下数值不能被视为严格的同像素比较：

- far accumulation mean；
- far object luma；
- far alpha threshold fraction；
- far object threshold fraction。

此外，当前 `far_rgb_object_luma` 使用的是带衰减的 `rgb_object`，而不是去水体 `J_gaussian` 或 `J_object`。它适合衡量带水体可见物体贡献，但不能完整衡量 clear Gaussian 残留。

因此，下一阶段的第一项工作必须是修正诊断体系。

---

# 3. 第二阶段总体目标

第二阶段不直接追求更激进的清除，而是完成三个层次的验证：

## 目标 A：确认第一阶段候选具有统计稳定性

回答：

> E2 `accum=0.002` 的优势是真实机制收益，还是单次非凸训练波动？

## 目标 B：建立可靠的 hit-aware object protection

回答：

> 是否可以只在没有可靠物体命中的区域施加 accumulation pressure，从而兼顾远水清除和远景物体保留？

## 目标 C：将单一 ownership 拆成监督、渲染和容量三类归属

回答：

> \(B_\infty\) 监督、背景合成与 Gaussian 容量控制是否应使用不同的 mask？

闭合尾部 CUDA 改造放在上述三项完成后进行。去水体颜色优化继续作为独立分支，不与早期 hit-aware 实验同时叠加。

---

# 4. Phase 0：先修正可重复性与诊断体系

## 4.1 P0-1：增加显式随机种子

当前实验脚本没有显式记录或传递训练 seed。应在：

```text
scripts/experiments/m2_infinite_water_iui3_redsea.sh
```

增加：

```bash
SEED="${SEED:-42}"
```

在 `run_manifest.txt` 中记录：

```bash
echo "seed=${SEED}"
```

确认当前 Nerfstudio CLI 支持的 seed 参数后，将其传给 `ns-train`。通常应检查：

```bash
ns-train water-splatting --help | grep -i seed
```

若暴露 `--machine.seed`，则增加：

```bash
--machine.seed "${SEED}"
```

必须同时记录：

```text
Git commit
CUDA_VISIBLE_DEVICES
seed
PyTorch version
CUDA version
dataset path
all M1/M2 flags
```

## 4.2 P0-2：建立固定公共远区 mask

新增：

```text
scripts/diagnostics/build_common_far_masks.py
```

第一轮使用 M1 checkpoint 生成固定 mask：

\[
M_{\mathrm{far}}^{\mathrm{common}}
=
\mathbf 1
\left[
d_{\mathrm{M1}}
\ge
Q_{0.90}(d_{\mathrm{M1}})
\right].
\]

保存：

```text
common_masks/view_0000_far.pt
common_masks/view_0001_far.pt
...
```

所有候选模型使用同一套 mask 进行诊断。

同时保留两套 mask：

1. `far_quantile_mask`：M1 expected depth 的 90% 分位区域；
2. `water_background_roi`：人工或 pseudo-depth 构建的高置信纯水体区域。

第一轮可以先完成 `far_quantile_mask`，teacher mask 放到后续阶段。

## 4.3 P0-3：扩展 leakage 指标

修改：

```text
scripts/diagnostics/diagnose_far_water_residual.py
```

增加固定 mask 参数：

```bash
--mask-dir common_masks/m1_q90
```

增加以下指标：

### 结构容量泄漏

\[
E_{\mathrm{capacity}}
=
\operatorname{mean}_{p\in M_{\mathrm{far}}}
A(p).
\]

### 带水体可见物体泄漏

\[
E_{\mathrm{uw\text{-}obj}}
=
\operatorname{mean}_{p\in M_{\mathrm{far}}}
\operatorname{luma}
\left(
I_{\mathrm{obj}}(p)
\right).
\]

### 去水体可见泄漏

\[
E_{\mathrm{clear}}
=
\operatorname{mean}_{p\in M_{\mathrm{far}}}
\operatorname{luma}
\left(
J_{\mathrm{gaussian}}(p)
\right).
\]

### Ownership 覆盖

```text
m_inf mean
m_inf_eff mean
m_inf > 0.5 fraction
m_inf_eff > 0.5 fraction
```

### 真实物体保留

在公共 object mask 中统计：

```text
object accumulation mean
object clear luma mean
object edge retention
```

## 4.4 P0-4：生成统一诊断图

每个模型输出：

```text
RGB
J_gaussian
J_object
depth
accumulation
m_inf
m_inf_eff
common far mask
far accumulation overlay
far J leakage overlay
```

要求同一视图、同一 crop、同一显示动态范围，禁止每个模型独立归一化。

## 4.5 Phase 0 验收标准

完成后必须确认：

- M1 与所有 M2 模型使用相同像素 mask；
- `far accumulation` 与 `far clear leakage` 分开报告；
- 可复现实验具有显式 seed；
- 诊断 JSON 包含 checkpoint SHA、代码 commit 与 mask 来源。

---

# 5. Phase 1：E2 候选稳定性与局部搜索

## 5.1 固定公共配置

以下参数保持不变：

```text
medium_context_mode = dir_xy_camera
infinite_water_enabled = True
ownership_mode = alpha_depth
compose_mode = rgb_mix
occupancy_limited = True
lambda_binf_rgb = 0.005
lambda_near_zero = 0
loss_start_step = 1000
loss_ramp_steps = 3000
max_iterations = 15000
```

## 5.2 P1-1：重复候选

优先重复：

| 编号 | accumulation weight | Seeds |
|---|---:|---|
| R05 | 0.0005 | 42, 123, 3407 |
| R20 | 0.0020 | 42, 123, 3407 |

目的：

- `0.0005` 是感知质量优先候选；
- `0.0020` 是容量控制优先候选；
- 先判断两者的均值与方差，再进行更密集搜索。

## 5.3 P1-2：窄范围搜索

在相同 seeds 下运行：

| 编号 | accumulation weight |
|---|---:|
| R10 | 0.0010 |
| R15 | 0.0015 |
| R20 | 0.0020 |
| R25 | 0.0025 |

如果算力受限，先使用 seed 42 完成四点搜索，再对 Pareto 最优的两点补齐另外两个 seeds。

## 5.4 P1-3：统计报告

每组报告：

```text
PSNR mean ± std
SSIM mean ± std
LPIPS mean ± std
common-mask far accumulation mean ± std
common-mask far clear leakage mean ± std
J blue dominance mean ± std
Gaussian count mean ± std
```

## 5.5 Phase 1 候选选择规则

不再只选择 PSNR 最大值，使用 Pareto 规则。

### 重建门槛

相对 M1：

```text
mean PSNR >= M1 mean - 0.02 dB
mean SSIM >= M1 mean
mean LPIPS <= M1 mean + 0.001
```

### 远水控制门槛

相对 M1：

```text
far capacity leakage reduction >= 90%
far clear leakage reduction >= 90%
```

### 稳定性门槛

```text
PSNR std <= 0.08 dB
LPIPS std <= 0.002
```

如果 `0.0015` 与 `0.0020` 性能接近，优先选择较小权重，减少真实远景误伤风险。

---

# 6. Phase 2：低成本 ownership 消融

在 Phase 1 确定的 accumulation weight 上，先不修改 CUDA，完成以下 ownership 结构测试。

## 6.1 P2-1：alpha evidence 与 depth evidence

| 编号 | Ownership mode | 目的 |
|---|---|---|
| O-A | `alpha_only` | 判断 depth evidence 是否真正必要 |
| O-AD | `alpha_depth` | 当前对照 |
| O-ADC | `alpha_depth_color` | 判断颜色相似度是否能提高纯水体 precision |

`alpha_depth_color` 必须重点观察真实蓝色、绿色物体是否被误判为水体。

## 6.2 P2-2：occupancy gate

比较：

```text
occupancy_limited=True
occupancy_limited=False
```

但必须同时记录：

```text
m_inf
m_inf_eff
```

由于当前 loss 使用 `m_inf`，该实验主要改变 RGB composition，不改变 accumulation loss support。报告中必须明确这一点。

## 6.3 P2-3：depth evidence 软化

固定 `alpha_depth`，测试：

| 编号 | depth_mid | depth_temp |
|---|---:|---:|
| D0 | 0.75 | 0.10 |
| D1 | 0.80 | 0.10 |
| D2 | 0.75 | 0.15 |
| D3 | 0.80 | 0.15 |

目标是降低远景边界附近的硬切换。

## 6.4 P2-4：alpha power

只在上述实验表明 alpha evidence 过于宽松时测试：

```text
alpha_power ∈ {1.0, 1.5, 2.0}
```

不建议一开始同时搜索 `alpha_power × depth_mid × depth_temp`，避免实验规模失控。

## 6.5 Phase 2 输出

必须输出：

```text
alpha_evidence
depth_evidence
color_evidence
m_inf
m_inf_eff
```

并对以下区域分别统计：

- 高置信纯水体；
- 远景海床；
- 珊瑚边界；
- 低纹理真实物体。

Phase 2 结束后选择一套基础 ownership，用于 hit-aware depth 实验。

---

# 7. Phase 3：Hit-Aware Depth 诊断实现

## 7.1 设计目标

当前 expected depth 只能描述贡献的平均距离，不能描述该深度是否来自稳定表面。Hit-aware depth 应回答：

1. 当前射线是否存在足够 object accumulation？
2. 深度贡献是否集中在一个表面附近？
3. 当前射线是否只是由稀疏、分散的远处 Gaussian 形成伪命中？

## 7.2 CUDA 新增统计量

修改文件：

```text
water_splatting/cuda/csrc/forward.cuh
water_splatting/cuda/csrc/forward.cu
water_splatting/cuda/csrc/bindings.h
water_splatting/cuda/csrc/bindings.cu
water_splatting/rasterize.py
water_splatting/rendering/underwater_rasterizer.py
```

当前 kernel 已经维护：

```text
T
prev_depth
pix_depth = sum(vis * depth)
```

新增：

\[
m_2(p)
=
\sum_i
T_i\alpha_i z_i^2.
\]

CUDA 中增加：

```cpp
float pix_depth2 = 0.f;
pix_depth2 += vis * depth * depth;
```

同时建议输出：

```text
final_transmittance = T
first_depth
last_depth = prev_depth
depth_first_moment
depth_second_moment
```

### 推荐输出字段

```python
UnderwaterRenderOutput(
    ...
    depth_expected,
    depth_variance,
    depth_std_relative,
    first_depth,
    last_depth,
    final_transmittance,
    hit_confidence,
)
```

## 7.3 Expected depth 与方差

Python 中计算：

\[
A=1-T_{\mathrm{end}},
\]

\[
d_{\mathrm{exp}}
=
\frac{m_1}{A+\epsilon},
\]

\[
\sigma_d^2
=
\max
\left(
\frac{m_2}{A+\epsilon}
-
d_{\mathrm{exp}}^2,
0
\right).
\]

定义相对深度离散度：

\[
r_d
=
\frac{\sqrt{\sigma_d^2}}
{d_{\mathrm{exp}}+\epsilon}.
\]

## 7.4 Hit confidence

第一版使用简单、可解释的形式：

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

初始参数：

```text
tau_A ∈ {0.10, 0.20, 0.30}
t_A = 0.05
kappa ∈ {0.10, 0.20, 0.30}
```

第一轮不训练这些参数，只进行离线可视化和阈值诊断。

## 7.5 P3-1：只输出诊断，不改变训练

新增可视化：

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

重点检查：

- 纯水体区的 \(q_{\mathrm{hit}}\) 是否接近 0；
- 海床与珊瑚表面的 \(q_{\mathrm{hit}}\) 是否较高；
- 远景边界是否出现低置信保护区；
- 漂浮 Gaussian 区是否表现为高相对方差。

## 7.6 Phase 3 诊断验收

在四个 eval views 中：

```text
纯水体区域 q_hit 均值低
真实海床区域 q_hit 均值高
边界处 q_hit 连续变化
不存在大面积真实物体被判为无命中
```

只有达到上述条件后，才允许将 `q_hit` 接入 loss。

---

# 8. Phase 4：Hit-Aware Capacity Protection

## 8.1 第一轮只修改 accumulation loss support

当前：

\[
\mathcal L_{\mathrm{acc}}
=
\frac{
\sum_p M_{\infty}(p)A(p)
}{
\sum_pM_{\infty}(p)+\epsilon
}.
\]

修改为：

\[
M_{\mathrm{capacity}}
=
M_{\infty}
\left(
1-q_{\mathrm{hit}}
\right),
\]

\[
\mathcal L_{\mathrm{acc}}^{\mathrm{hit}}
=
\frac{
\sum_p
M_{\mathrm{capacity}}(p)A(p)
}{
\sum_p
M_{\mathrm{capacity}}(p)+\epsilon
}.
\]

`q_hit` 默认 detach，不让模型通过篡改 hit confidence 规避约束。

## 8.2 P4 实验矩阵

| 编号 | Accumulation support | RGB render mask |
|---|---|---|
| H0 | `m_inf` | 当前 `m_inf_eff` |
| H1 | `m_inf * (1-q_alpha)` | 当前 `m_inf_eff` |
| H2 | `m_inf * (1-q_hit)` | 当前 `m_inf_eff` |
| H3 | `m_inf * (1-q_hit)^2` | 当前 `m_inf_eff` |

H2 为主候选。H3 用于判断是否需要更强的 object protection。

## 8.3 P4 成功标准

相对 Phase 1 候选：

```text
PSNR 不下降超过 0.02 dB
LPIPS 改善或不恶化
far clear leakage 不上升超过 10%
object-mask accumulation 明显恢复
远景轮廓和纹理完整性改善
```

若 hit-aware gating 仅改善 RGB 指标，却导致远水残留明显反弹，则需要进一步拆分 capacity mask，而不是直接减小 accumulation weight。

---

# 9. Phase 5：拆分 Support、Render 与 Capacity Ownership

## 9.1 数据结构修改

将：

```python
InfiniteWaterOwnershipOutput
```

扩展为：

```python
@dataclass
class InfiniteWaterOwnershipOutput:
    m_support: Tensor
    m_render: Tensor
    m_capacity: Tensor

    alpha_evidence: Tensor
    depth_evidence: Tensor
    color_evidence: Tensor
    hit_evidence: Tensor
```

为兼容旧实验，可暂时保留：

```python
m_inf = m_support
m_inf_eff = m_render
```

但新代码和新日志应使用显式命名。

## 9.2 三类 ownership 定义

### B-infinity 监督区域

\[
M_{\mathrm{support}}
=
M_{\alpha}
M_d.
\]

要求：

- 覆盖纯水体；
- 可比 render mask 稍宽；
- 初期仍保持 detach。

### 渲染归属

\[
M_{\mathrm{render}}
=
M_{\mathrm{support}}
(1-A).
\]

用于：

```text
rgb_mix 或未来 closed-tail composition
J_object 可视化
```

### 容量控制归属

\[
M_{\mathrm{capacity}}
=
M_{\mathrm{support}}
(1-q_{\mathrm{hit}}).
\]

用于：

```text
accumulation-zero loss
后续 densification blocking
opacity decay
```

## 9.3 损失对应关系

```text
B_inf RGB loss       使用 m_support
RGB composition      使用 m_render
Accumulation loss    使用 m_capacity
Near-zero loss       保持关闭
```

## 9.4 P5 消融

| 编号 | support | render | capacity |
|---|---|---|---|
| S0 | 单一 `m_inf` | `m_inf_eff` | `m_inf` |
| S1 | `m_support` | `m_render` | `m_support` |
| S2 | `m_support` | `m_render` | `m_capacity` |
| S3 | `m_support` | `m_render * (1-q_hit)` | `m_capacity` |

优先测试 S2。S3 只有在 RGB composition 仍明显误伤远景时才启用。

---

# 10. Phase 6：Pseudo-Depth Teacher 先诊断后训练

## 10.1 原则

当前项目 datamanager 使用标准 FullImageDatamanager，并未在 M2 路径中加载 pseudo-depth。不要直接将 DepthAnything 路径写死到模型中。

第一步只实现离线 mask 工具：

```text
scripts/diagnostics/build_pseudo_depth_water_masks.py
```

输入：

```text
image path
pseudo-depth path
camera/eval filename mapping
```

输出：

```text
foreground mask
background-water candidate
eroded water mask
RGB boundary mask
depth boundary mask
```

## 10.2 Mask 构建

建议：

1. 归一化 pseudo-depth；
2. 提取远背景候选；
3. 保留主要前景连通区域；
4. 对水体候选执行腐蚀；
5. 排除 RGB 强边缘；
6. 排除 pseudo-depth 强边缘；
7. 与低 \(q_{\mathrm{hit}}\) 区域取交集。

定义：

\[
M_{\mathrm{teacher}}
=
M_{\mathrm{pseudo\text{-}water}}
M_{\mathrm{eroded}}
M_{\mathrm{low\text{-}edge}}
(1-q_{\mathrm{hit}}).
\]

## 10.3 腐蚀消融

```text
erosion radius ∈ {5, 7, 11}
```

先生成 overlay，不接入训练。

## 10.4 接入训练时的方式

当 teacher precision 达标后，将 mask 作为 batch 的可选字段：

```python
batch["water_teacher_mask"]
```

不要在 `water_splatting.py` 中根据绝对路径读取文件。

推荐：

\[
M_{\mathrm{support}}
=
M_{\infty}
\left[
\lambda_t M_{\mathrm{teacher}}
+
(1-\lambda_t)
\right].
\]

第一轮：

```text
lambda_teacher ∈ {0.25, 0.5}
```

避免将 pseudo-depth 当作硬标签。

---

# 11. Phase 7：真正的 Closed-Tail Rendering

## 11.1 先输出当前尾部，再改变公式

当前 CUDA 已经计算：

```text
pix_medium
T
prev_depth
medium_bs
medium_rgb
```

增加：

\[
W_{\mathrm{tail}}^{\mathrm{last}}
=
T
\exp
\left(
-\beta^B d_{\mathrm{last}}
\right).
\]

并分别输出：

```text
medium_finite = pix_medium
tail_weight_last
tail_medium_original = tail_weight_last * medium_rgb
```

验证：

\[
I_{\mathrm{med}}^{\mathrm{old}}
=
I_{\mathrm{med}}^{\mathrm{finite}}
+
W_{\mathrm{tail}}^{\mathrm{last}}B.
\]

必须先确认拆分后的 `rgb_object + medium_finite + tail_medium_original` 与现有输出数值一致。

## 11.2 Closed-tail composition

定义：

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

该公式只替换水体尾部颜色，不会直接乘法削弱 object render。

## 11.3 Closure depth 消融

### Last-depth closure

\[
d_{\mathrm{close}}=d_{\mathrm{last}}.
\]

用于验证与当前渲染器的一致性。

### Expected-depth closure

\[
d_{\mathrm{close}}=d_{\mathrm{exp}}.
\]

### Hit-aware closure

\[
d_{\mathrm{close}}
=
q_{\mathrm{hit}}d_{\mathrm{exp}}
+
(1-q_{\mathrm{hit}})d_{\mathrm{far}}.
\]

其中 `d_far` 应是稳定的场景级或训练集级尺度，不要使用每个像素最后一个稀疏 Gaussian 的深度。

## 11.4 P7 实验矩阵

| 编号 | Tail composition | Closure depth |
|---|---|---|
| T0 | 当前 renderer | current last depth |
| T1 | 分离后重组，使用 medium_rgb | last depth |
| T2 | \(B_\infty\) closed tail | last depth |
| T3 | \(B_\infty\) closed tail | expected depth |
| T4 | \(B_\infty\) closed tail | hit-aware depth |

T1 必须与 T0 数值近似一致，否则不能继续 T2 至 T4。

---

# 12. Phase 8：Gaussian Capacity Control

只有在 `m_capacity` 稳定后，才进入 Gaussian 级控制。

## 12.1 第一优先级：Densification Blocking

为每个 Gaussian 累积：

```text
water support EMA
object support EMA
hit confidence EMA
observation count
```

高置信水体型 Gaussian：

\[
q_i^{\mathrm{water}}
=
\frac{
C_i^{\mathrm{water}}
}{
C_i^{\mathrm{water}}
+
C_i^{\mathrm{object}}
+\epsilon
}.
\]

满足：

```text
q_water > 0.8
object_support < 0.1
observation_count >= 5
```

时：

```text
禁止 duplicate
禁止 split
```

不直接 prune。

## 12.2 第二优先级：Opacity Decay

只对连续多个周期满足条件的 Gaussian 执行：

```text
opacity decay
```

建议：

```text
start_step = 8000
interval = 500
minimum_consecutive_hits = 4
maximum_processed_fraction = 0.5%
```

## 12.3 暂不启用硬删除

第一阶段结果已经表明远区残留可以通过 loss 显著降低。硬删除容易破坏真实远景结构，因此仅作为最终附加消融，不作为主方法默认配置。

---

# 13. 去水体颜色分支的并行安排

偏蓝、偏绿问题不应与 hit-aware M2 同时开发。建议建立独立分支：

```text
M1 only
+ intrinsic DC color
+ constrained SH residual
+ dual-color rasterization
```

先测试：

| 编号 | Underwater color | Clear color |
|---|---|---|
| C0 | SH=3 | SH=3 |
| C1 | SH=3 | DC only |
| C2 | SH=3 | DC + 0.05 SH residual |
| C3 | SH=3 | DC + 0.10 SH residual |
| C4 | SH=3 | DC + luminance-only SH residual |

只有颜色分支独立证明有效后，再与 Phase 5 或 Phase 7 的稳定 M2 组合。

---

# 14. 推荐的实际执行顺序

## 第一批：立即执行

1. 增加 seed 记录与控制；
2. 建立 M1 固定公共 far mask；
3. 修正 diagnostic，增加 clear leakage；
4. 重复 `accum=0.0005` 与 `accum=0.0020`；
5. 搜索 `0.0010 / 0.0015 / 0.0020 / 0.0025`。

## 第二批：低成本 ownership

6. `alpha_only / alpha_depth / alpha_depth_color`；
7. occupancy gate 开关；
8. `depth_mid / depth_temp` 小范围消融。

## 第三批：Hit-Aware Depth

9. CUDA 输出 depth second moment、first depth、last depth 和 final transmittance；
10. 仅生成 `q_hit` 诊断图；
11. 将 `q_hit` 只接入 accumulation loss support；
12. 选择 hit-aware capacity candidate。

## 第四批：Ownership Split

13. 实现 `m_support / m_render / m_capacity`；
14. 分别绑定监督、合成和容量损失；
15. 验证远景保护。

## 第五批：Closed Tail

16. 拆分 finite medium 与 tail medium；
17. 数值复现当前 renderer；
18. 测试 last、expected 与 hit-aware closure；
19. 用 closed-tail 替代当前 `rgb_mix`。

## 第六批：Teacher 与 Capacity

20. pseudo-depth teacher 只做 mask diagnostics；
21. 软接入 `m_support`；
22. densification blocking；
23. 软 opacity decay。

## 第七批：颜色恢复

24. dual-color Gaussian；
25. SH luminance/chroma 分离；
26. transmission-aware foreground loss；
27. 与最终 M2 组合。

---

# 15. 下一阶段最小实验矩阵

| 编号 | 主要改动 | CUDA 修改 | 训练成本 | 优先级 |
|---|---|---:|---:|---:|
| P0 | 固定 mask 与新诊断 | 否 | 无 | 最高 |
| R05 | accum=0.0005 多 seed | 否 | 3 runs | 最高 |
| R20 | accum=0.0020 多 seed | 否 | 3 runs | 最高 |
| R15 | accum=0.0015 | 否 | 1 至 3 runs | 高 |
| O-A | alpha_only | 否 | 1 run | 高 |
| O-AD | alpha_depth | 否 | 对照 | 高 |
| O-ADC | alpha_depth_color | 否 | 1 run | 中 |
| H-DIAG | hit-aware diagnostics | 是 | 无训练或短评估 | 最高 |
| H2 | capacity × (1-q_hit) | 是 | 1 至 3 runs | 最高 |
| S2 | ownership split | 少量 Python | 1 至 3 runs | 高 |
| T1 | tail 拆分数值复现 | 是 | 评估 | 高 |
| T4 | hit-aware closed tail | 是 | 1 至 3 runs | 高 |
| C1-C4 | dual color | 是 | 独立分支 | 后续 |

---

# 16. 结果记录模板

每个实验必须记录：

```text
Experiment name
Git commit
Seed
Checkpoint
Dataset
M1 context mode
Ownership mode
Support mask definition
Render mask definition
Capacity mask definition
B_inf loss weight
Accumulation loss weight
Compose mode
Closure depth mode
Hit confidence parameters
PSNR / SSIM / LPIPS
Common-mask capacity leakage
Common-mask underwater object leakage
Common-mask clear leakage
Object-region retention
J color diagnostics
Gaussian count
Training time
Peak GPU memory
```

所有对比图必须至少包含：

```text
RGB
J_gaussian
J_object
accumulation
depth
m_support
m_render
m_capacity
q_hit
common far mask
```

---

# 17. 阶段性决策规则

## 保留某个 ownership 配置

必须同时满足：

```text
underwater metrics 不低于 M1 门槛
common-mask clear leakage 显著下降
object retention 不下降
多 seed 方差可接受
```

## 保留 hit-aware depth

必须证明：

```text
q_hit 能区分纯水体与真实远景物体
hit-aware capacity 比单纯 alpha-depth 更保护远景
far leakage 不发生明显反弹
```

## 保留 closed-tail rendering

必须先通过：

```text
T1 与当前 renderer 数值一致
```

随后证明：

```text
T2-T4 不再像 rgb_mix 一样直接削弱 object render
underwater PSNR/LPIPS 优于当前 tail_approx
```

## 进入颜色组合实验

只有当 M2 的结构分支在多 seed 上稳定后，才允许与 dual-color 分支组合，避免无法判断收益来源。

---

# 18. 当前推荐主线

在下一轮代码修改完成前，当前临时主线仍为：

```text
M1 dir_xy_camera
+ M2 alpha_depth
+ rgb_mix
+ occupancy_limited=True
+ lambda_binf_rgb=0.005
+ lambda_accumulation_zero=0.002
+ lambda_near_zero=0
+ loss_start_step=1000
+ loss_ramp_steps=3000
```

但该配置仅是第一阶段候选，不应写成最终方法。

下一阶段最关键的创新验证应集中在：

\[
\boxed{
\text{Hit-Aware Object Protection}
+
\text{Support/Render/Capacity Ownership Split}
+
\text{Closed Infinite-Water Tail}
}
\]

其核心目的不是继续扩大水体区域，而是让 Gaussian 容量抑制只发生在缺乏可靠场景命中的射线上，从而同时保留远水清除能力与新视角重建质量。
