# M1 Context-Aware Medium Modeling - 2026-07-23

## Implementation

Added `medium_context_mode` to `WaterSplattingModelConfig`:

- `dir_only`: original WaterSplatting direction-only medium input; default.
- `dir_xy`: direction encoding + normalized image `x`, `y`, and radius.
- `dir_xy_depth`: `dir_xy` + rendered expected depth context.
- `dir_xy_camera`: `dir_xy` + scene-box-normalized camera center.
- `dir_xy_depth_camera`: `dir_xy_depth` + scene-box-normalized camera center.

Depth context is explicitly defined as rendered expected depth from a first medium/raster pass, normalized per view by p95 by default, and detached by default before the second medium prediction.

Camera context uses:

```text
(camera_center - scene_box_center) / scene_box_diagonal
```

This avoids the older exploratory `tanh(camera_center)` behavior.

## Default Compatibility

Default mode is `dir_only`, so the medium MLP input dimensionality and checkpoint keys remain equivalent to original WaterSplatting.

Regression check after adding M1 flags:

```text
rgb: 0.0
depth: 0.0
accumulation: 0.0
rgb_object: 0.0
rgb_clear: 0.0
rgb_clear_clamp: 0.0
rgb_medium: 0.0
pred_image: 0.0
medium_rgb: 0.0
medium_bs: 0.0
medium_attn: 0.0
```

Result: `default_dir_only_equivalence_ok`.

## Context Mode Smoke

CUDA forward smoke passed for:

```text
dir_xy
dir_xy_depth
dir_xy_camera
dir_xy_depth_camera
```

## Experiment Script

Main run:

```bash
GPU=6 MEDIUM_CONTEXT_MODE=dir_xy_camera MAX_NUM_ITERATIONS=15000 \
  bash scripts/experiments/m1_context_medium_iui3_redsea.sh
```

Smoke run:

```bash
GPU=6 MEDIUM_CONTEXT_MODE=dir_xy_camera MAX_NUM_ITERATIONS=5 RUN_EVAL=0 \
  EXPERIMENT_NAME=m1_smoke_dir_xy_camera_iui3_redsea \
  bash scripts/experiments/m1_context_medium_iui3_redsea.sh
```

## CLI Smoke Results

Completed:

```text
m1_smoke_dir_xy_camera_iui3_redsea_20260723_072056
m1_smoke_dir_xy_depth_camera_iui3_redsea_20260723_072121
```

Both smoke runs completed from scratch and wrote checkpoints under ignored `outputs/`.

## Baseline Checkpoint Compatibility

After adding M1, the original baseline checkpoint/config still evaluates to the recorded metrics:

```text
step1_refactor_sanity_iui3_redsea_20260723_072224
PSNR  = 29.879043579101562
SSIM  = 0.9104903936386108
LPIPS = 0.18103083968162537
```

## Main 15000-Iter Result

Run:

```text
m1_dir_xy_camera_iui3_redsea_15000_20260723_072412
```

Checkpoint:

```text
/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/nerfstudio_models/step-000014999.ckpt
```

Eval:

```text
PSNR  = 31.13144874572754
SSIM  = 0.9120101928710938
LPIPS = 0.17501820623874664
```

Delta vs original baseline:

```text
PSNR  = +1.2524051666259766
SSIM  = +0.0015197992324829102
LPIPS = -0.006012633442878723
```

Initial judgment: M1 is positive on IUI3-RedSea and is safe to carry into M2.
