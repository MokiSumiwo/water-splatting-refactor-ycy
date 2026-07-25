# WaterSplatting M2 Phase 3 简化与 Closed Tail 实验记录

日期：2026-07-24
分支：`refactor/core-framework`
起始提交：`a9a98578537a29a3880bc2a8f1c1d45c6ff6b417`

## 1. 目标

本轮对应 `plan/WaterSplatting_M2简化与ClosedTail_下一阶段实验方案.md`，核心目标是：

1. 拆解 M2 的收益来源：`B_inf` RGB loss、RGB composition、capacity loss。
2. 将 capacity loss 显式化为可消融的 loss mode，包括 current、depth-monotonic、ReLU budget、softplus budget。
3. 关闭 dynamic hit protection，仅保留 hit-aware outputs 作为诊断。
4. 在不改 CUDA 的前提下，验证 closed-tail 的 T1 数值重组等价性，并加入可回退 `closed_tail` compose mode。

## 2. 代码改动

### 2.1 Compose mode

新增：

```python
infinite_water_compose_mode: Literal["none", "rgb_mix", "tail_approx", "closed_tail"]
```

- `none`：不使用 `B_inf` 改写 RGB，仍保留 ownership、`B_inf`、capacity loss 和 diagnostics。
- `closed_tail`：只替换解析得到的 infinite medium tail，不对完整 `render.rgb` 做 alpha mix。

### 2.2 Capacity loss mode

新增：

```python
infinite_water_capacity_loss_mode: Literal[
    "none",
    "current",
    "depth_monotonic",
    "relu_budget",
    "softplus_budget",
]
infinite_water_capacity_budget: float = 0.05
infinite_water_capacity_budget_temp: float = 0.02
```

实现方式：

```text
current:          m_capacity * accumulation
depth_monotonic:  depth_evidence * accumulation
relu_budget:      depth_evidence * relu(accumulation - budget)
softplus_budget:  depth_evidence * softplus((accumulation - budget) / temp) * temp
```

`current` 保留旧行为；budget 系列不再把 `(1 - accumulation)` 再乘入 capacity support。

### 2.3 Closed-tail diagnostic outputs

新增派生输出：

```text
rgb_medium_finite
tail_weight_last
tail_medium_original
```

其中：

```text
tail_weight_last = final_transmittance * exp(-medium_bs * last_depth)
tail_medium_original = tail_weight_last * medium_rgb
rgb_medium_finite = rgb_medium - tail_medium_original
```

### 2.4 新脚本

- `scripts/experiments/m2_phase3_mechanism_decomposition_iui3_redsea.sh`
- `scripts/experiments/m2_phase3_capacity_budget_iui3_redsea.sh`
- `scripts/diagnostics/diagnose_closed_tail_recompose.py`

统一训练脚本 `scripts/experiments/m2_infinite_water_iui3_redsea.sh` 已接入 capacity loss mode 和 budget flags。

## 3. Phase 3A：机制拆解

固定配置：

```text
medium_context_mode=dir_xy_camera
ownership_mode=alpha_depth
depth_mid=0.75
depth_temp=0.10
occupancy_limited=True
lambda_near_zero=0
loss_start_step=1000
loss_ramp_steps=3000
seed=42
max_iterations=15000
```

命令模板：

```bash
GPU=<6-9> MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=42 \
  MECHANISM_GRID='<single item>' \
  EXPERIMENT_NAME_PREFIX=m2_p3_mech \
  STAMP=20260724_142500_phase3A \
  bash scripts/experiments/m2_phase3_mechanism_decomposition_iui3_redsea.sh
```

| Run | Mechanism | PSNR | SSIM | LPIPS | J Blue | Common Far Accum | Common Far Clear | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | none / no B_inf / no cap | 30.9818 | 0.9111 | 0.1777 | 0.1663 | 0.497513 | 0.075597 | 0.281239 | 0.001122 | 0.9902 | 1.0003 | 0.9887 |
| S1 | rgb_mix / B_inf / no cap | 30.8137 | 0.9119 | 0.1751 | 0.2483 | 0.652263 | 0.122220 | 0.250791 | 0.007290 | 1.0152 | 1.0441 | 0.9973 |
| S2 / L1 | none / current cap 0.002 | 31.0759 | 0.9127 | 0.1769 | 0.0310 | 0.192367 | 0.067679 | 0.001609 | 0.000561 | 0.8680 | 0.9816 | 0.9868 |
| S3 | rgb_mix / B_inf / current cap 0.002 | 31.0109 | 0.9139 | 0.1769 | 0.0567 | 0.167254 | 0.047874 | 0.001265 | 0.000129 | 0.8561 | 0.9712 | 0.9719 |
| S4 | none / B_inf / current cap 0.002 | 31.0746 | 0.9145 | 0.1765 | 0.0524 | 0.161351 | 0.048120 | 0.000348 | 0.000090 | 0.8595 | 0.9838 | 0.9320 |

### 3A 结论

- S0/S1 说明没有 capacity 时，远水 Gaussian leakage 非常强；`rgb_mix` + `B_inf` 本身不能清理 Gaussian 表示污染。
- S2 说明 capacity loss 可以独立降低 leakage，不需要 RGB composition 强绑定。
- S3/S4 相比 S2 可进一步降低 far clear / water J，但 S4 boundary retention 下降到 0.9320，说明 `B_inf` loss 即使不参与 RGB，也会改变优化轨迹。

## 4. Phase 3B：capacity loss 形式

第一轮固定：

```text
compose_mode=none
lambda_binf_rgb=0
lambda_capacity=0.002
seed=42
```

| Run | Capacity | PSNR | SSIM | LPIPS | J Blue | Common Far Accum | Common Far Clear | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 | depth_monotonic | 31.2946 | 0.9141 | 0.1792 | 0.0265 | 0.162004 | 0.061395 | 0.028083 | 0.000711 | 0.8501 | 0.9472 | 0.9526 |
| L3 | relu_budget 0.05 | 31.1984 | 0.9148 | 0.1781 | 0.0402 | 0.133902 | 0.050968 | 0.004459 | 0.000287 | 0.8491 | 0.9721 | 0.9500 |
| L4 | relu_budget 0.10 | 31.0784 | 0.9129 | 0.1776 | 0.0332 | 0.135855 | 0.054475 | 0.001081 | 0.000349 | 0.8512 | 0.9492 | 0.9416 |
| L5 | softplus_budget 0.05 | 30.9548 | 0.9130 | 0.1787 | 0.0536 | 0.158078 | 0.059154 | 0.007007 | 0.000439 | 0.8577 | 0.9715 | 0.9769 |

### 3B 结论

- L2/L3 PSNR 高，但 LPIPS 变差明显，说明强单调 depth support 会牺牲 perceptual quality。
- L3 是 budget loss 中最强 leakage-control 点，但 object accumulation retention 为 0.8491，略低于 0.85 门槛。
- L4/L5 没有形成明显更优 Pareto。

## 5. Phase 3C：capacity weight 搜索

对 `current` 和 `relu_budget0.05` 搜索：

```text
lambda_capacity in {0.0005, 0.001, 0.002}
compose_mode=none
lambda_binf_rgb=0
seed=42
```

| Run | PSNR | SSIM | LPIPS | J Blue | Common Far Accum | Common Far Clear | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current 0.0005 | 31.0960 | 0.9126 | 0.1774 | 0.0619 | 0.236210 | 0.068889 | 0.002916 | 0.000619 | 0.9185 | 0.9765 | 0.9752 |
| current 0.001 | 31.2739 | 0.9128 | 0.1769 | 0.0557 | 0.193654 | 0.059862 | 0.000766 | 0.000426 | 0.8894 | 0.9903 | 0.9754 |
| current 0.002 | 31.0759 | 0.9127 | 0.1769 | 0.0310 | 0.192367 | 0.067679 | 0.001609 | 0.000561 | 0.8680 | 0.9816 | 0.9868 |
| relu_budget 0.0005 | 31.0755 | 0.9127 | 0.1756 | 0.0572 | 0.224522 | 0.070734 | 0.001815 | 0.000479 | 0.9076 | 0.9749 | 0.9665 |
| relu_budget 0.001 | 31.0927 | 0.9130 | 0.1773 | 0.0690 | 0.183173 | 0.052289 | 0.005291 | 0.000072 | 0.8653 | 0.9448 | 0.9224 |
| relu_budget 0.002 | 31.1984 | 0.9148 | 0.1781 | 0.0402 | 0.133902 | 0.050968 | 0.004459 | 0.000287 | 0.8491 | 0.9721 | 0.9500 |

Pareto candidates selected for seed completion:

- `current@0.001`: best seed42 PSNR and strong object retention.
- `relu_budget0.05@0.0005`: best LPIPS among capacity candidates and simplified monotonic-budget form.

## 6. 三 seed 稳定性

| Config | PSNR mean ± std | SSIM mean ± std | LPIPS mean ± std | Far Accum mean ± std | Far Clear mean ± std | Water Accum mean ± std | Obj Acc Ret mean ± std | Obj J Ret mean ± std | Boundary Ret mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current@0.001 | 31.0991 ± 0.1610 | 0.9128 ± 0.0007 | 0.1767 ± 0.0003 | 0.213891 ± 0.018160 | 0.063818 ± 0.003573 | 0.002052 ± 0.001467 | 0.8948 ± 0.0079 | 0.9707 ± 0.0175 | 0.9766 ± 0.0239 |
| relu_budget0.05@0.0005 | 31.0373 ± 0.0642 | 0.9130 ± 0.0004 | 0.1763 ± 0.0008 | 0.227207 ± 0.005541 | 0.068886 ± 0.002730 | 0.007537 ± 0.006648 | 0.9027 ± 0.0074 | 0.9650 ± 0.0095 | 0.9581 ± 0.0076 |

M1 reference:

```text
PSNR=31.1314
SSIM=0.9120
LPIPS=0.1750
Common Far Accum=0.407096
Common Far Clear=0.083962
Water Accum=0.032564
Water J=0.000928
```

### 6.1 稳定性判断

相对 M1：

- `current@0.001` PSNR drop 为 0.0323 dB，SSIM 持平略升，但 LPIPS 增加 0.0017，且 PSNR std=0.1610，未满足稳定性门槛。
- `relu_budget0.05@0.0005` PSNR drop 为 0.0941 dB，LPIPS 增加 0.0013，但 PSNR std=0.0642、LPIPS std=0.0008，稳定性较好。
- 两者均显著降低 water accumulation 和 common far accumulation，但都没有同时满足重建质量、LPIPS、稳定性三类门槛。

因此，本轮没有配置可以升级为稳定最终 M2 candidate。

## 7. Closed Tail T1 诊断

本轮不改 CUDA，而是基于已有 `final_transmittance` 和 `last_depth` 派生：

```text
tail_weight_last = final_transmittance * exp(-medium_bs * last_depth)
tail_medium_original = tail_weight_last * medium_rgb
rgb_medium_finite = rgb_medium - tail_medium_original
```

T1 diagnostic command：

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_closed_tail_recompose.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --output-dir renders/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/closed_tail_recompose \
  --max-images 4
```

Result：

| Metric | Mean abs | Max abs | Pass threshold |
|---|---:|---:|---|
| Medium recompose | 6.42e-11 | 5.96e-08 | Yes |
| RGB recompose | 6.43e-11 | 1.19e-07 | Yes |

T1 threshold:

```text
max absolute difference < 1e-5
mean absolute difference < 1e-6
```

结论：基于已有 CUDA 输出的解析 tail / finite split 在 eval 视角上数值等价通过，可作为后续 closed-tail 的实现基础。

## 8. Closed Tail smoke

Smoke command：

```bash
GPU=7 MAX_NUM_ITERATIONS=10 RUN_EVAL=0 SEED=42 \
  CAPACITY_GRID='T4smoke:relu_budget:0.05' \
  ACCUM_ZERO_WEIGHT=0.0005 \
  INFINITE_WATER_COMPOSE_MODE=closed_tail \
  BINF_RGB_WEIGHT=0.005 \
  EXPERIMENT_NAME_PREFIX=sanity_m2_p4_closed_tail \
  bash scripts/experiments/m2_phase3_capacity_budget_iui3_redsea.sh
```

Result：10-step smoke training completed without runtime error.

## 9. Closed Tail seed42 主实验

由于 T1 通过，本轮继续补跑 CT0/CT1/CT2 最小 closed-tail 矩阵：

```bash
GPU=6 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=42 \
  CAPACITY_GRID='CT0:none:0' \
  ACCUM_ZERO_WEIGHT=0 \
  BINF_RGB_WEIGHT=0.005 \
  INFINITE_WATER_COMPOSE_MODE=closed_tail \
  EXPERIMENT_NAME_PREFIX=m2_p4_closed_tail \
  STAMP=20260724_155500_phase4CT \
  bash scripts/experiments/m2_phase3_capacity_budget_iui3_redsea.sh

GPU=7 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=42 \
  CAPACITY_GRID='CT1:current:0' \
  ACCUM_ZERO_WEIGHT=0.001 \
  BINF_RGB_WEIGHT=0.005 \
  INFINITE_WATER_COMPOSE_MODE=closed_tail \
  EXPERIMENT_NAME_PREFIX=m2_p4_closed_tail \
  STAMP=20260724_155500_phase4CT \
  bash scripts/experiments/m2_phase3_capacity_budget_iui3_redsea.sh

GPU=8 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=42 \
  CAPACITY_GRID='CT2:relu_budget:0.05' \
  ACCUM_ZERO_WEIGHT=0.0005 \
  BINF_RGB_WEIGHT=0.005 \
  INFINITE_WATER_COMPOSE_MODE=closed_tail \
  EXPERIMENT_NAME_PREFIX=m2_p4_closed_tail \
  STAMP=20260724_155500_phase4CT \
  bash scripts/experiments/m2_phase3_capacity_budget_iui3_redsea.sh
```

| Run | Capacity | PSNR | SSIM | LPIPS | J Blue | Common Far Accum | Common Far Clear | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CT0 | none | 31.0320 | 0.9144 | 0.1743 | 0.1548 | 0.569974 | 0.078978 | 0.377473 | 0.000987 | 0.9994 | 1.0293 | 0.9834 |
| CT1 | current@0.001 | 30.8521 | 0.9106 | 0.1769 | 0.0531 | 0.186073 | 0.062119 | 0.002218 | 0.000524 | 0.8744 | 0.9899 | 1.0051 |
| CT2 | relu_budget0.05@0.0005 | 31.0948 | 0.9128 | 0.1790 | 0.0539 | 0.173726 | 0.055323 | 0.000671 | 0.000414 | 0.8647 | 0.9723 | 0.9407 |

### 9.1 Closed Tail 判断

- CT0 的 LPIPS=0.1743 优于 M1，但 water accumulation 反弹到 0.377473，远高于 M1 的 0.032564，不可作为主方法。
- CT1 降低 water leakage，但 PSNR=30.8521、SSIM=0.9106，重建质量明显不合格。
- CT2 的 PSNR=31.0948 接近 M1 门槛，far clear/water leakage 均较低，但 LPIPS=0.1790 且 boundary retention=0.9407，不合格。
- 因此 seed42 下 closed-tail 尚未形成稳定正收益；无需继续补 CT seeds。

## 10. 结论

1. M2 的主要有效项仍是 capacity regularization，而不是 `B_inf` RGB composition。
2. `rgb_mix` 不应作为默认主线继续强化；S1 表明它不能清理 Gaussian 表示污染。
3. 单调/budget capacity 能降低 leakage，但当前权重下 LPIPS 和/或 PSNR 仍无法同时通过 M1-relative 门槛。
4. 三 seed 后没有稳定最终 M2 capacity candidate；closed-tail seed42 最小矩阵也没有超过 M1。
5. Closed-tail T1 数值等价已经通过，但正式 closed-tail 训练未形成收益；当前应保留 M1 作为可靠主线，M2 降级为 diagnostic / appendix，除非后续有新的 capacity 或 supervision 信号。

## 11. 文件位置

- Full summary:
  - `logs/phase3_jobs/phase3_full_summary.json`
  - `logs/phase3_jobs/phase3_seed_summary.json`
  - `logs/phase3_jobs/phase4_closed_tail_summary.json`
- T1 diagnostic:
  - `renders/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/closed_tail_recompose/closed_tail_recompose_diagnostic.json`
- Every run contains:
  - `output.json`
  - `common_far_m1_q90/far_water_residual_diagnostic.json`
  - `eval_regions/eval_region_diagnostic.json`
  - `eval_regions/heatmaps/`

## 12. Recommended next action

当前建议进入情况 C 的判断路径：

```text
Closed Tail 也无稳定收益
=> 保留 M1，M2 作为 diagnostic / appendix
```

后续如果继续探索 M2，不建议再增加 dynamic hit protection、capacity floor 或 hard prune；更值得测试的是外部 teacher / pseudo-depth 或更明确的 water/object supervision。
