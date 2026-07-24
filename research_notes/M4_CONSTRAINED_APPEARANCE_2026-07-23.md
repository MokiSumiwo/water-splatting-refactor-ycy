# M4 Constrained View-Dependent Appearance - 2026-07-23

## Purpose

M4 targets dewatered-image quality while preserving underwater reconstruction quality. The main failure mode is SH/DC appearance absorbing water effects, which can make the dewatered image over-white, overexposed, or red/blue biased.

## Output Semantics

- `J`: the dewatered image requested here, defined as pure Gaussian-only clear rendering:

```text
J = clamp(raw Gaussian-only clear render, 0, 1)
```

- `J_raw`: unclamped raw Gaussian-only clear render.
- `J_gaussian`: alias of `J` for explicit Gaussian-only naming.
- `J_object`: M2 ownership-masked diagnostic image, `clamp((1 - m_inf_eff) * J_raw, 0, 1)`.
- `rgb_clear`: legacy WaterSplatting compressed clear image, `raw / (raw + 1)`. This is kept for backward compatibility but is not treated as the dewatered image.

## Implementation

- Added active SH scheduling controlled by:
  - `constrained_appearance_enabled`
  - `appearance_sh_delay_enabled`
  - `appearance_sh_delay_start_step`
  - `appearance_sh_delay_interval`
- Added ramped M4 losses:
  - `lambda_sh_residual_mean`: penalizes view-dependent SH residual color offsets on visible Gaussians.
  - `lambda_dc_softclip`: soft upper bound on intrinsic DC RGB, optionally weighted in low-transmission regions.
  - `lambda_dc_channel_balance`: optional soft penalty on strong red/blue DC dominance.
  - `lambda_medium_attenuation_order`: optional soft prior for medium attenuation order `red >= green >= blue`.
- All M4 flags default to original-equivalent behavior unless explicitly enabled by the experiment script.

## Main M4 Experiment

- Experiment: `m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000`
- Mechanism: M1 `dir_xy_camera` + M2 `alpha_depth` + delayed SH + SH residual mean + DC softclip.
- Command:

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 \
  EXPERIMENT_NAME=m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m4_constrained_appearance_iui3_redsea.sh
```

- Checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_095026/nerfstudio_models/step-000014999.ckpt`
- Pure-J eval after output semantic correction: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_095026_pureJ_eval/output.json`
- Visual contact sheet: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m4_vs_m2_pureJ_contact_sheet_20260723.png`

```text
PSNR  = 31.092180252075195
SSIM  = 0.9157193899154663
LPIPS = 0.17794030904769897

J_white_ratio          = 0.014490095898509026
J_saturation_ratio     = 0.016798991709947586
J_red_dominance_ratio  = 0.052886899560689926
J_blue_dominance_ratio = 0.035804372280836105
```

Relative to M2 pure-J eval:

```text
PSNR  = +0.022617340087890625
SSIM  = +0.0027877092361450195
LPIPS = +0.0008031129837036133

J_white_ratio          = -0.006525726988911629
J_saturation_ratio     = -0.009747259318828583
J_red_dominance_ratio  = +0.010292842984199524
J_blue_dominance_ratio = -0.015636198222637177
```

Visual judgment: M4 is darker and less over-white/over-saturated than M2. It also reduces blue dominance, but increases red dominance in some bright coral/rock regions.

## M4b DC Balance Ablation

- Experiment: `m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000`
- Mechanism: M4 main plus `lambda_dc_channel_balance=0.001`; medium attenuation order remains off.
- Command:

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 \
  EXPERIMENT_NAME=m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  DC_CHANNEL_BALANCE_WEIGHT=0.001 MEDIUM_ATTENUATION_ORDER_WEIGHT=0.0 \
  bash scripts/experiments/m4_constrained_appearance_iui3_redsea.sh
```

Equivalent wrapper:

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 \
  bash scripts/experiments/m4b_dc_balance_iui3_redsea.sh
```

- Checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_101241/nerfstudio_models/step-000014999.ckpt`
- Eval renders/metrics: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_101241/output.json`
- Visual contact sheet: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m2_m4_m4b_pureJ_contact_sheet_20260723.png`

```text
PSNR  = 31.014209747314453
SSIM  = 0.9132192134857178
LPIPS = 0.18013009428977966

J_white_ratio          = 0.014512766152620316
J_saturation_ratio     = 0.017113082110881805
J_red_dominance_ratio  = 0.04683053493499756
J_blue_dominance_ratio = 0.020334944128990173
```

Relative to M4 main:

```text
PSNR  = -0.07797050476074219
SSIM  = -0.002500176429748535
LPIPS = +0.0021897852420806885

J_white_ratio          = +0.000022670254111289978
J_saturation_ratio     = +0.00031409040093421936
J_red_dominance_ratio  = -0.0060563646256923676
J_blue_dominance_ratio = -0.015469428151845932
```

Visual judgment: M4b does reduce red/blue dominance, but it introduces visible streaking/light-spike artifacts near the water boundary and hurts underwater PSNR/SSIM/LPIPS. It is useful as an ablation but should not replace M4 main on IUI3-RedSea.

## M4c Weak DC Balance Ablation

- Experiment: `m4c_dc_balance0003_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000`
- Mechanism: M4 main plus weaker `lambda_dc_channel_balance=0.0003`; medium attenuation order remains off.
- Command:

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 \
  EXPERIMENT_NAME=m4c_dc_balance0003_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  DC_CHANNEL_BALANCE_WEIGHT=0.0003 MEDIUM_ATTENUATION_ORDER_WEIGHT=0.0 \
  bash scripts/experiments/m4_constrained_appearance_iui3_redsea.sh
```

Equivalent wrapper:

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 \
  bash scripts/experiments/m4c_dc_balance0003_iui3_redsea.sh
```

- Checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m4c_dc_balance0003_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m4c_dc_balance0003_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_103115/nerfstudio_models/step-000014999.ckpt`
- Eval renders/metrics: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m4c_dc_balance0003_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_103115/output.json`
- Visual contact sheet: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m2_m4_m4b_m4c_pureJ_contact_sheet_20260723.png`

```text
PSNR  = 31.117921829223633
SSIM  = 0.9132004976272583
LPIPS = 0.17735272645950317

J_white_ratio          = 0.013658018782734871
J_saturation_ratio     = 0.01621161587536335
J_red_dominance_ratio  = 0.03941240534186363
J_blue_dominance_ratio = 0.024587390944361687
```

Relative to M4 main:

```text
PSNR  = +0.0257415771484375
SSIM  = -0.002518892288208008
LPIPS = -0.0005875825881958008

J_white_ratio          = -0.0008320771157741547
J_saturation_ratio     = -0.0005873758345842361
J_red_dominance_ratio  = -0.013474494218826294
J_blue_dominance_ratio = -0.011216981336474419
```

Relative to M2 pure-J eval:

```text
PSNR  = +0.048358917236328125
SSIM  = +0.0002688169479370117
LPIPS = +0.0002155303955078125

J_white_ratio          = -0.007357804104685783
J_saturation_ratio     = -0.010334635153412819
J_red_dominance_ratio  = -0.00318165123462677
J_blue_dominance_ratio = -0.026853179559111595
```

Visual judgment: M4c removes the strong M4b boundary streaking while improving over-white, saturation, red dominance, and blue dominance versus M2. It gives the best PSNR among M2/M4/M4b/M4c and the best `J` diagnostic balance in this set.

## Current Recommendation

- Use M4c as the current M4 candidate on IUI3-RedSea.
- Treat strong `lambda_dc_channel_balance=0.001` as an ablation only; it creates water-boundary streaking.
- Do not enable `lambda_medium_attenuation_order` in a main run until a smaller weight or later-ramp ablation is tested; it is more likely to constrain the medium branch than directly fix `J`.
- If further M4 tuning is needed, test later-ramped weak DC balance or a still smaller value such as `DC_CHANNEL_BALANCE_WEIGHT=0.0001`.
