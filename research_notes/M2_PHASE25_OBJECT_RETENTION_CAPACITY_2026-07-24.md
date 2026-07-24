# WaterSplatting M2 Phase 2.5 实验记录

日期：2026-07-24
分支：`refactor/core-framework`
起始提交：`781a656a31bafb2f8774d59f9b16b71335020fd1`

## 1. 目标

本轮对应 `WaterSplatting_M2_Phase2.5_下一步实验报告.md`，目标是验证 D2 多 seed 稳定性，建立固定 water/object/boundary 评价区域，校准 `q_hit`，并测试 conservative hit-aware capacity floor。

本轮不做 CUDA 改动、不做 frozen-hit 训练接入、不做 Gaussian hard prune。

## 2. 代码改动

### 2.1 Conservative capacity floor

新增配置项，默认保持关闭，不改变既有 M1/M2 行为：

```python
infinite_water_hit_protection_enabled: bool = False
infinite_water_hit_protection_threshold: float = 0.80
infinite_water_hit_protection_temp: float = 0.05
infinite_water_capacity_floor: float = 0.50
infinite_water_hit_protection_start_step: int = 0
```

启用后：

```text
P_obj = sigmoid((q_hit - threshold) / temp)
M_capacity = M_capacity_base * [1 - (1 - capacity_floor) * P_obj]
```

其中 `M_capacity_base` 仍由 `infinite_water_capacity_support_mode` 决定，Phase 2.5 主实验使用 `m_inf`。

### 2.2 脚本

- `scripts/experiments/m2_infinite_water_iui3_redsea.sh`
  - 接入 capacity floor 五个新 flag；
  - run manifest 记录新参数。
- `scripts/experiments/m2_phase25_capacity_floor_iui3_redsea.sh`
  - C0-C4 grid 脚本，默认 15000 iter + eval。
- `scripts/diagnostics/build_eval_region_masks.py`
  - 基于 M1 自动生成 high-confidence eval-only water/object/boundary masks。
- `scripts/diagnostics/diagnose_eval_regions.py`
  - 输出 water leakage、object retention、boundary gradient retention、q_hit calibration table。

## 3. 固定评价区域

命令：

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/build_eval_region_masks.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --output-dir common_masks/m1_auto_eval_regions_iui3_redsea_20260724 \
  --max-images 4 \
  --save-png
```

说明：当前为 M1-derived 自动高置信评价 mask，只用于诊断，不用于训练。后续如有人工 mask，应替换本轮自动 mask。

| Eval view | Water pixels | Object pixels | Boundary pixels |
|---|---:|---:|---:|
| 0000 | 100,424 | 643,528 | 94,108 |
| 0001 | 5,869 | 941,382 | 186,073 |
| 0002 | 29,488 | 1,018,775 | 115,663 |
| 0003 | 83,490 | 866,870 | 70,813 |

## 4. Phase A：D0/D2 多 seed 稳定性

配置固定：

```text
medium_context_mode=dir_xy_camera
ownership_mode=alpha_depth
compose_mode=rgb_mix
occupancy_limited=True
lambda_binf_rgb=0.005
lambda_accumulation_zero=0.002
lambda_near_zero=0
loss_start_step=1000
loss_ramp_steps=3000
max_iterations=15000
```

新增主实验命令使用：

```bash
GPU=6 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=123 DEPTH_GRID='D2:0.75:0.15' EXPERIMENT_NAME_PREFIX=m2_p25_depth STAMP=20260724_121000_phaseA bash scripts/experiments/m2_phase2_depth_softening_iui3_redsea.sh
GPU=7 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=3407 DEPTH_GRID='D2:0.75:0.15' EXPERIMENT_NAME_PREFIX=m2_p25_depth STAMP=20260724_121000_phaseA bash scripts/experiments/m2_phase2_depth_softening_iui3_redsea.sh
GPU=8 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=123 DEPTH_GRID='D0:0.75:0.10' EXPERIMENT_NAME_PREFIX=m2_p25_depth STAMP=20260724_121000_phaseA bash scripts/experiments/m2_phase2_depth_softening_iui3_redsea.sh
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=3407 DEPTH_GRID='D0:0.75:0.10' EXPERIMENT_NAME_PREFIX=m2_p25_depth STAMP=20260724_121000_phaseA bash scripts/experiments/m2_phase2_depth_softening_iui3_redsea.sh
```

Seed42 复用 Phase 2 已有 D0/D2 checkpoint。

### 4.1 单 run 结果

| Run | PSNR | SSIM | LPIPS | J Blue | Common Far Accum | Common Far Clear | Object Acc Ret | Object J Ret | Boundary Grad Ret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0-S42 | 31.1190 | 0.9147 | 0.1792 | 0.0606 | 0.146410 | 0.052388 | 0.8466 | 0.9767 | 0.9323 |
| D0-S123 | 31.0010 | 0.9141 | 0.1766 | 0.0665 | 0.221345 | 0.068293 | 0.8661 | 0.9760 | 0.9810 |
| D0-S3407 | 30.9045 | 0.9115 | 0.1787 | 0.0677 | 0.182885 | 0.058829 | 0.8693 | 0.9994 | 0.9658 |
| D2-S42 | 31.2212 | 0.9145 | 0.1758 | 0.0551 | 0.199784 | 0.070869 | 0.8617 | 0.9866 | 1.0084 |
| D2-S123 | 30.8869 | 0.9135 | 0.1768 | 0.0732 | 0.209748 | 0.061482 | 0.8661 | 0.9807 | 0.9701 |
| D2-S3407 | 30.9815 | 0.9115 | 0.1754 | 0.0838 | 0.211001 | 0.067887 | 0.8681 | 0.9641 | 0.9802 |

### 4.2 多 seed 汇总

| Config | PSNR mean ± std | SSIM mean ± std | LPIPS mean ± std | J Blue mean ± std | Far Accum mean ± std | Far Clear mean ± std | Gaussian count mean ± std |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | 31.0082 ± 0.1074 | 0.9134 ± 0.0017 | 0.1782 ± 0.0014 | 0.0649 ± 0.0038 | 0.183547 ± 0.037472 | 0.059837 ± 0.008001 | 805,323 ± 5,627 |
| D2 | 31.0299 ± 0.1723 | 0.9132 ± 0.0015 | 0.1760 ± 0.0007 | 0.0707 ± 0.0145 | 0.206844 ± 0.006146 | 0.066746 ± 0.004796 | 807,714 ± 1,065 |

### 4.3 Phase A 判断

- D2 的 PSNR mean 高于 D0，LPIPS 明显优于 D0。
- D2 的 common far clear 是 D0 的 1.115x，低于方案中 1.4x 上限。
- 但 D2 的 PSNR std=0.1723 dB，未满足 `<=0.10 dB` 稳定性门槛。
- 因此 D2 可作为质量倾向候选，但不能作为稳定最终配置。

按方案补跑折中：

```bash
GPU=6 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=42 DEPTH_GRID='D125:0.75:0.125' EXPERIMENT_NAME_PREFIX=m2_p25_depth STAMP=20260724_124500_phaseA_mid bash scripts/experiments/m2_phase2_depth_softening_iui3_redsea.sh
```

| Run | PSNR | SSIM | LPIPS | J Blue | Common Far Accum | Common Far Clear | Water Accum | Water J | Object Acc Ret | Object J Ret | Boundary Grad Ret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D125-S42 | 30.9316 | 0.9130 | 0.1774 | 0.0384 | 0.176898 | 0.054272 | 0.000724 | 0.000174 | 0.8531 | 0.9717 | 0.9731 |

折中配置降低了 J blue 和 leakage，但 PSNR/LPIPS 不优于 D2-S42，也不优于 D0-S42。

## 5. Phase C：q_hit 校准

以 D2-S42 为代表：

| Threshold | Object precision vs water | Water FPR | Object recall |
|---:|---:|---:|---:|
| 0.60 | 0.999937 | 0.000538 | 0.5384 |
| 0.80 | 0.999976 | 0.000141 | 0.3744 |

区域分布：

| Region | q_hit mean |
|---|---:|
| Water | 0.020743 |
| Object | 0.565855 |

结论：

- 在当前自动评价 mask 上，`q_hit` 的 high-threshold precision 很高；
- `q_hit > 0.8` 的 water false positive 很低，但 object recall 只有约 37%；
- 这支持“高精度、低覆盖”的 object-protection 用法，但不支持用 `1-q_hit` 完全取消 capacity pressure。

## 6. Phase D：Conservative capacity floor

命令：

```bash
GPU=6 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=42 CAPACITY_FLOOR_GRID='C0:False:0.80:1.00' EXPERIMENT_NAME_PREFIX=m2_p25_capacity_floor STAMP=20260724_122500_phaseD bash scripts/experiments/m2_phase25_capacity_floor_iui3_redsea.sh
GPU=7 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=42 CAPACITY_FLOOR_GRID='C1:True:0.60:0.50' EXPERIMENT_NAME_PREFIX=m2_p25_capacity_floor STAMP=20260724_122500_phaseD bash scripts/experiments/m2_phase25_capacity_floor_iui3_redsea.sh
GPU=8 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=42 CAPACITY_FLOOR_GRID='C2:True:0.80:0.50' EXPERIMENT_NAME_PREFIX=m2_p25_capacity_floor STAMP=20260724_122500_phaseD bash scripts/experiments/m2_phase25_capacity_floor_iui3_redsea.sh
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 SEED=42 CAPACITY_FLOOR_GRID='C3:True:0.80:0.75' EXPERIMENT_NAME_PREFIX=m2_p25_capacity_floor STAMP=20260724_122500_phaseD bash scripts/experiments/m2_phase25_capacity_floor_iui3_redsea.sh
```

| Run | tau | floor | PSNR | SSIM | LPIPS | J Blue | Common Far Accum | Common Far Clear | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Grad Ret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | off | 1.00 | 30.9901 | 0.9139 | 0.1764 | 0.0899 | 0.191040 | 0.068041 | 0.002484 | 0.000242 | 0.8604 | 0.9729 | 0.9525 |
| C1 | 0.60 | 0.50 | 30.9169 | 0.9144 | 0.1758 | 0.0523 | 0.183619 | 0.057658 | 0.001434 | 0.000175 | 0.8624 | 0.9653 | 0.9425 |
| C2 | 0.80 | 0.50 | 31.1051 | 0.9138 | 0.1763 | 0.0669 | 0.192273 | 0.072526 | 0.001713 | 0.000789 | 0.8656 | 0.9774 | 1.0029 |
| C3 | 0.80 | 0.75 | 30.9757 | 0.9116 | 0.1772 | 0.0757 | 0.130217 | 0.054456 | 0.002168 | 0.000512 | 0.8402 | 0.9490 | 0.9465 |

### 6.1 Phase D 判断

相对 C0：

- C1 降低 far clear、water accumulation、J blue，LPIPS 略好，但 PSNR drop=0.0733 dB，未满足 `<=0.02 dB`。
- C2 PSNR 提升 0.1150 dB，LPIPS 略好，object accumulation retention 和 boundary gradient retention 均提升；但 common far clear 增加约 6.6%，略超 `<=5%` 成功标准。
- C3 显著降低 common far accumulation/clear，但 object retention 和 LPIPS 变差，不是合格 object-retention 配置。

因此，capacity floor 没有出现完全通过成功标准的配置。C2 是当前最好的质量/边界平衡候选，但不能直接升级为稳定 M2 candidate；C1 是 leakage-control 候选但重建质量不足。

## 7. 文件位置

- Phase A 新结果：
  - `renders/m2_p25_depth_D0_mid0p75_temp0p10_accum0p002_seed123_dir_xy_camera_iui3_redsea_15000_20260724_121000_phaseA`
  - `renders/m2_p25_depth_D0_mid0p75_temp0p10_accum0p002_seed3407_dir_xy_camera_iui3_redsea_15000_20260724_121000_phaseA`
  - `renders/m2_p25_depth_D2_mid0p75_temp0p15_accum0p002_seed123_dir_xy_camera_iui3_redsea_15000_20260724_121000_phaseA`
  - `renders/m2_p25_depth_D2_mid0p75_temp0p15_accum0p002_seed3407_dir_xy_camera_iui3_redsea_15000_20260724_121000_phaseA`
- Phase A 折中：
  - `renders/m2_p25_depth_D125_mid0p75_temp0p125_accum0p002_seed42_dir_xy_camera_iui3_redsea_15000_20260724_124500_phaseA_mid`
- Phase D：
  - `renders/m2_p25_capacity_floor_C0_tau0p80_floor1p00_seed42_dir_xy_camera_iui3_redsea_15000_20260724_122500_phaseD`
  - `renders/m2_p25_capacity_floor_C1_tau0p60_floor0p50_seed42_dir_xy_camera_iui3_redsea_15000_20260724_122500_phaseD`
  - `renders/m2_p25_capacity_floor_C2_tau0p80_floor0p50_seed42_dir_xy_camera_iui3_redsea_15000_20260724_122500_phaseD`
  - `renders/m2_p25_capacity_floor_C3_tau0p80_floor0p75_seed42_dir_xy_camera_iui3_redsea_15000_20260724_122500_phaseD`
- 汇总 JSON：
  - `logs/phase25_jobs/phase25_summary.json`

每个 render 目录下均包含：

```text
output.json
common_far_m1_q90/far_water_residual_diagnostic.json
eval_regions/eval_region_diagnostic.json
eval_regions/heatmaps/
```

## 8. 结论与下一步

1. D2 仍是质量倾向最强的 depth softening 配置，但多 seed PSNR 方差过大，不能作为稳定最终配置。
2. 自动评价 mask 能把 `q_hit` 的 water false positive 压到很低，说明 `q_hit` 可作为高精度 object evidence。
3. 直接用 dynamic `q_hit` 做 capacity floor 仍不稳定：C2 改善质量/边界但增加 common far clear，C1 改善 leakage 但损失 PSNR。
4. 暂不建议进入 hard prune 或 closed-tail；下一步应优先做 frozen-hit reference map 的离线生成和诊断，验证 dynamic self-protection 是否是主要失败来源。
5. 若继续 capacity floor，建议只围绕 C2 做延迟保护 `hit_protection_start_step=5000` 的单 seed 验证，再决定是否补 seed。
