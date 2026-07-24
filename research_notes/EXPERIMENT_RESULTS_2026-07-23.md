# Experiment Results - 2026-07-23

## Baseline: Original WaterSplatting

- Experiment: `baseline_original_watersplatting_iui3_redsea`
- Mechanism: original WaterSplatting, `medium_context_mode=dir_only`
- Checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/baseline_original_watersplatting_iui3_redsea/water-splatting/orig_watersplatting_iui3_redsea_15000_20260723_063201/nerfstudio_models/step-000014999.ckpt`
- Eval: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/baseline_original_watersplatting_iui3_redsea/eval_metrics.json`

```text
PSNR  = 29.879043579101562
SSIM  = 0.9104903936386108
LPIPS = 0.18103083968162537
```

## M1: Context-Aware Medium

- Experiment: `m1_dir_xy_camera_iui3_redsea_15000`
- Mechanism: M1 context-aware medium, `medium_context_mode=dir_xy_camera`
- Command:

```bash
GPU=6 MEDIUM_CONTEXT_MODE=dir_xy_camera MAX_NUM_ITERATIONS=15000 \
  EXPERIMENT_NAME=m1_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m1_context_medium_iui3_redsea.sh
```

- Checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/nerfstudio_models/step-000014999.ckpt`
- Eval renders/metrics: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/output.json`

```text
PSNR  = 31.13144874572754
SSIM  = 0.9120101928710938
LPIPS = 0.17501820623874664
```

Relative to baseline:

```text
PSNR  = +1.2524051666259766
SSIM  = +0.0015197992324829102
LPIPS = -0.006012633442878723
```

## M2: Infinite-Water Ownership

- Experiment: `m2_alpha_depth_dir_xy_camera_iui3_redsea_15000`
- Mechanism: M1 `dir_xy_camera` plus M2 infinite-water ownership, `ownership_mode=alpha_depth`
- Command:

```bash
GPU=7 OWNERSHIP_MODE=alpha_depth MEDIUM_CONTEXT_MODE=dir_xy_camera MAX_NUM_ITERATIONS=15000 \
  EXPERIMENT_NAME=m2_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m2_infinite_water_iui3_redsea.sh
```

- Checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_080936/nerfstudio_models/step-000014999.ckpt`
- Eval renders/metrics: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_080936/output.json`

```text
PSNR  = 31.069562911987305
SSIM  = 0.9129316806793213
LPIPS = 0.17713719606399536
```

Relative to original baseline:

```text
PSNR  = +1.1905193328857422
SSIM  = +0.002441287040710449
LPIPS = -0.003893643617630005
```

Relative to M1:

```text
PSNR  = -0.061885833740234375
SSIM  = +0.0009214878082275391
LPIPS = +0.0021189898252487183
```

## M3: Contribution Cleanup Diagnostic

- Experiment: `m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000`
- Mechanism: M1 + M2 plus M3 contribution-aware cleanup diagnostics, `gaussian_cleanup_dry_run=True`
- Command:

```bash
GPU=8 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 CLEANUP_DRY_RUN=True \
  CLEANUP_START_STEP=12000 CLEANUP_INTERVAL=500 \
  EXPERIMENT_NAME=m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m3_cleanup_diagnostic_iui3_redsea.sh
```

- Checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_083104/nerfstudio_models/step-000014999.ckpt`
- Eval renders/metrics: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_083104/output.json`

```text
PSNR  = 31.17935562133789
SSIM  = 0.9149407148361206
LPIPS = 0.17747271060943604
```

Relative to original baseline:

```text
PSNR  = +1.3003120422363281
SSIM  = +0.004450321197509766
LPIPS = -0.003558129072189331
```

Relative to M2:

```text
PSNR  = +0.10979270935058594
SSIM  = +0.0020090341567993164
LPIPS = +0.00033551454544067383
```

M3 dry-run candidate summary:

```text
steps logged: 12000, 12500, 13000, 13500, 14000, 14500
candidates:  0 at every logged step
reason:      ownership and alpha gates are very sparse at Gaussian projected centers
```

## M4: Constrained View-Dependent Appearance

- Experiment: `m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000`
- Mechanism: M1 `dir_xy_camera` + M2 `alpha_depth` + delayed SH activation + SH residual mean loss + DC softclip loss.
- Dewatered image definition: `J = clamp(raw Gaussian-only clear render, 0, 1)`. The legacy `rgb_clear = raw / (raw + 1)` is no longer treated as the dewatered image.
- Command:

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 \
  EXPERIMENT_NAME=m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m4_constrained_appearance_iui3_redsea.sh
```

- Checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_095026/nerfstudio_models/step-000014999.ckpt`
- Pure-J eval renders/metrics: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m4_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_095026_pureJ_eval/output.json`

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

## M4b: DC Channel Balance Ablation

- Experiment: `m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000`
- Mechanism: M4 main plus `lambda_dc_channel_balance=0.001`; medium attenuation order remains off.
- Command:

```bash
GPU=9 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 \
  EXPERIMENT_NAME=m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  DC_CHANNEL_BALANCE_WEIGHT=0.001 MEDIUM_ATTENUATION_ORDER_WEIGHT=0.0 \
  bash scripts/experiments/m4_constrained_appearance_iui3_redsea.sh
```

- Checkpoint: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_101241/nerfstudio_models/step-000014999.ckpt`
- Eval renders/metrics: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/m4b_dc_balance_constrained_appearance_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_101241/output.json`

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

## M4c: Weak DC Channel Balance Ablation

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

## Judgment

M1 did not drop the underwater rendering metrics. It substantially improves PSNR and LPIPS on IUI3-RedSea while preserving the original render/composition path.

M2 `alpha_depth` remains materially better than the original baseline. Compared with M1, it trades a very small PSNR/LPIPS drop for a modest SSIM gain and adds explicit infinite-water maps (`m_inf`, `m_inf_eff`, `b_inf`) for visual/ownership diagnosis.

M3 diagnostic mode is implemented and stable, but the current conservative gates produce zero cleanup candidates. A direct far-water residual check on the M2 checkpoint also shows the far region is already effectively object-free: far-depth `accumulation` mean is `0.00048`, far-depth `rgb_object` luma mean is `0.0000249`, and only `0.0234%` of far pixels exceed object-luma `0.03`.

M4 main improves underwater SSIM versus M2 and materially reduces `J` over-white and saturation ratios under the corrected pure-Gaussian `J` definition. It also reduces blue dominance but increases red dominance in some bright regions.

M4b reduces both red and blue dominance, but it lowers underwater metrics and introduces visible water-boundary streak/light-spike artifacts in the contact sheet. Keep M4b as an ablation only.

M4c weak DC balance is the current recommended M4 candidate: it improves PSNR versus M2/M4, reduces `J` over-white and saturation, and lowers both red and blue dominance without the obvious M4b boundary streaking. Next recommended step: if further M4 tuning is needed, test later-ramped weak DC balance or `DC_CHANNEL_BALANCE_WEIGHT=0.0001` before trying the medium attenuation order prior.
