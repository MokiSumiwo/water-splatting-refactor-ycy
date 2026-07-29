# Shrink-Only Capacity Scale Control - 2026-07-29

## Scope

This note records the S-series follow-up after the R-series showed that global
footprint/position scaling does not dominate P3.

The hypothesis was:

```text
P3's useful capacity geometry signal may be a signed footprint-shrink signal.
Keeping only log-scale gradients that shrink Gaussians might preserve far-water
cleanup while avoiding object-damaging geometry movement.
```

## Code Changes

Commit:

```text
13b9321 Add shrink-only capacity scale control
```

Added default-off config flags:

```text
capacity_control_scale_shrink_only: bool = False
capacity_control_scale_shrink_clip_quantile: float = -1.0
capacity_control_scale_shrink_clip_value: float = 0.0
```

Implementation details:

- Capacity-control can now use a separate branch-local projection when scale
  sign control is active.
- `means` and `depth` capacity gradients can be set to `0.0`.
- `quats` are detached in shrink-only capacity projection, so S1/S2 test
  opacity + log-scale only rather than generic footprint/conic/quaternion
  pressure.
- A hook on branch-local log-scale tensors keeps only positive scale gradients
  when `capacity_control_scale_shrink_only=True`.
- Positive shrink gradients can be clipped by quantile or absolute value.
- Main render, clear proxy, and inference composition are unchanged.

Added diagnostics and scripts:

```text
scripts/diagnostics/diagnose_capacity_signed_region_gradients.py
scripts/experiments/medium_attr_s1_capacity_scale_shrink_only_iui3.sh
scripts/experiments/medium_attr_s2_capacity_scale_shrink_p90_iui3.sh
```

## Smoke Tests

Both 20-step smoke tests completed:

```bash
GPU=7 MAX_NUM_ITERATIONS=20 MODEL_NUM_STEPS=20 \
  RUN_EVAL=0 RUN_CLOSURE_DIAG=0 RUN_FAR_DIAG=0 RUN_REGION_DIAG=0 \
  EXPERIMENT_NAME=medium_attr_s1_capacity_scale_shrink_only_iui3_smoke20 \
  STAMP=20260729_s1_shrink_smoke20 \
  bash scripts/experiments/medium_attr_s1_capacity_scale_shrink_only_iui3.sh

GPU=8 MAX_NUM_ITERATIONS=20 MODEL_NUM_STEPS=20 \
  RUN_EVAL=0 RUN_CLOSURE_DIAG=0 RUN_FAR_DIAG=0 RUN_REGION_DIAG=0 \
  EXPERIMENT_NAME=medium_attr_s2_capacity_scale_shrink_p90_iui3_smoke20 \
  STAMP=20260729_s2_shrink_p90_smoke20 \
  bash scripts/experiments/medium_attr_s2_capacity_scale_shrink_p90_iui3.sh
```

Smoke output directories were deleted after verification.

## S0 Signed Region Gradient Audit

P3 checkpoints available locally:

```text
step-000005000.ckpt
step-000010000.ckpt
step-000014999.ckpt
```

The requested 8000 and 12000 diagnostics were not run because those checkpoints
do not exist for P3.

Commands:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_capacity_signed_region_gradients.py \
  --load-config outputs/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000/water-splatting/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000_20260729_p3_b02_proxy_geom000_opacity050/config.yml \
  --load-step 5000 --force-step 5000 \
  --split train --max-images 12 \
  --mask-dir common_masks/high_precision_water_m1_core_y025_thr050_er13_edge5_top_iui3_redsea_20260726 \
  --output-json logs/s0_signed_region_p3_step5000_train12.json

CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_capacity_signed_region_gradients.py \
  --load-config outputs/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000/water-splatting/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000_20260729_p3_b02_proxy_geom000_opacity050/config.yml \
  --load-step 10000 --force-step 10000 \
  --split train --max-images 12 \
  --mask-dir common_masks/high_precision_water_m1_core_y025_thr050_er13_edge5_top_iui3_redsea_20260726 \
  --output-json logs/s0_signed_region_p3_step10000_train12.json

CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_capacity_signed_region_gradients.py \
  --load-config outputs/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000/water-splatting/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000_20260729_p3_b02_proxy_geom000_opacity050/config.yml \
  --load-step 14999 --force-step 14999 \
  --split train --max-images 12 \
  --mask-dir common_masks/high_precision_water_m1_core_y025_thr050_er13_edge5_top_iui3_redsea_20260726 \
  --output-json logs/s0_signed_region_p3_step14999_train12.json
```

Key aggregate observations:

| Step | scale shrink mass | scale grow mass | effective opacity update | effective scale update | effective means update | mean shrink persistence | frac persistence >= 0.70 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 5.48e-05 | 2.49e-40 | 1.34e-07 | 2.08e-07 | 1.24e-08 | 0.9726 | 0.9948 |
| 10000 | 3.61e-05 | 2.53e-30 | 5.48e-08 | 1.42e-07 | 6.08e-09 | 0.9678 | 0.9905 |
| 14999 | 5.73e-05 | 1.38e-25 | 2.70e-07 | 2.22e-07 | 6.95e-09 | 0.9690 | 0.9921 |

Interpretation:

- P3's capacity scale gradients are already essentially shrink-only.
- There is almost no scale-grow capacity mass to remove.
- Shrink persistence is too broad: nearly all active visible Gaussians have
  persistent shrink pressure.
- Therefore, multi-view shrink persistence alone is unlikely to discriminate
  bad far-water Gaussians from useful object/boundary Gaussians.
- Effective scale update mass is comparable to or above opacity update mass;
  effective means update mass is smaller but still nonzero and spatially
  structured.

## Experiment Matrix

Shared base:

```text
medium_context_mode = dir_xy_camera
b_inf_mode = tied
lambda_medium_explainability = 0.005
lambda_budgeted_capacity = 0.0002
budgeted_capacity_value = 0.05
lambda_background_clear_chroma = 0.0015
clear_proxy_geometry_gradient_scale = 0.0
clear_proxy_opacity_gradient_scale = 0.50
clear_proxy_color_gradient_scale = 1.0
capacity_control_enabled = True
capacity_control_position_gradient_scale = 0.0
capacity_control_depth_gradient_scale = 0.0
capacity_control_footprint_gradient_scale = 1.0
capacity_control_opacity_gradient_scale = 1.0
```

| Run | Scale direction | Scale clamp |
|---|---|---|
| S1 | shrink-only | none |
| S2 | shrink-only | positive p90 |

Formal commands:

```bash
GPU=8 bash scripts/experiments/medium_attr_s1_capacity_scale_shrink_only_iui3.sh
GPU=9 bash scripts/experiments/medium_attr_s2_capacity_scale_shrink_p90_iui3.sh
```

## Results

| Run | PSNR | SSIM | LPIPS | FarAccum | FarClear | FarBGFrac | FarBGLCCmax | WaterAccum | WaterJ | ObjAccRet | ObjJRet | BoundaryRet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3 | 31.2235 | 0.913678 | 0.174772 | 0.294079 | 0.061725 | 0.145686 | 0.097096 | 0.021233 | 0.000502 | 0.975145 | 0.970946 | 0.994212 |
| Q1 | 30.9785 | 0.913084 | 0.175349 | 0.397450 | 0.076567 | 0.169030 | 0.112062 | 0.019425 | 0.000732 | 0.975205 | 1.012164 | 0.974045 |
| R4 | 30.9914 | 0.912165 | 0.175812 | 0.298163 | 0.063580 | 0.116519 | 0.103924 | 0.015636 | 0.000480 | 0.950579 | 0.976050 | 0.957869 |
| S1 | 31.0766 | 0.911743 | 0.174171 | 0.321556 | 0.074872 | 0.216630 | 0.184266 | 0.015654 | 0.000799 | 0.959936 | 0.980067 | 0.959735 |
| S2 | 30.9101 | 0.911210 | 0.175631 | 0.361367 | 0.084520 | 0.285015 | 0.401271 | 0.034344 | 0.003529 | 0.968421 | 0.974031 | 0.926852 |

Output paths:

```text
outputs/medium_attr_s1_capacity_scale_shrink_only_iui3_15000
renders/medium_attr_s1_capacity_scale_shrink_only_iui3_15000_20260729_s1_capacity_scale_shrink_only
logs/medium_attr_s1_capacity_scale_shrink_only_iui3_15000_20260729_s1_capacity_scale_shrink_only

outputs/medium_attr_s2_capacity_scale_shrink_p90_iui3_15000
renders/medium_attr_s2_capacity_scale_shrink_p90_iui3_15000_20260729_s2_capacity_scale_shrink_p90
logs/medium_attr_s2_capacity_scale_shrink_p90_iui3_15000_20260729_s2_capacity_scale_shrink_p90
```

## Interpretation

S1 does not recover P3:

- PSNR is `0.1469 dB` below P3.
- FarAccum worsens from `0.2941` to `0.3216`.
- FarClear worsens from `0.0617` to `0.0749`.
- FarBGFrac worsens from `0.1457` to `0.2166`.
- FarBGLCCmax worsens from `0.0971` to `0.1843`.
- Object Acc Ret falls from `0.9751` to `0.9599`.

S2 is worse than S1:

- P90 clipping removes too much useful shrink pressure.
- FarBGFrac and FarBGLCCmax increase sharply.
- Boundary retention drops to `0.9269`.

S0 explains why:

- The scale component of P3 was already shrink-only, so S1 did not remove a
  meaningful harmful scale-grow component.
- Persistence is not selective because shrink pressure is persistent for almost
  all active visible Gaussians.
- The useful P3 behavior appears to require the full projected footprint path,
  including conic/quaternion and possibly means/depth interactions, not just
  log-scale shrink.

## Decision

Do not run S3 as originally proposed.

Reason:

```text
S3's persistence gate is unlikely to separate far residuals from object/boundary
Gaussians because S0 shows shrink persistence is already high for nearly all
active visible Gaussians.
```

Current best candidate remains P3.

The next diagnostic should move away from Gaussian-level shrink persistence and
toward pixel-footprint attribution:

```text
For each Gaussian, estimate how much of its projected footprint overlaps:
  support_core
  support_halo
  far-bg residual pixels
  object mask
  boundary mask

Then gate projected footprint/conic gradients by footprint-overlap ratios rather
than by global scale sign or Gaussian center sampling.
```

This is more consistent with the visual failure mode: large projected far
Gaussians and object-adjacent footprints leaking into open water.
