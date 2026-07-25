# Backscatter-Consistent B_inf Experiments

Date: 2026-07-25  
Scene: IUI3-RedSea  
Main baseline: M1 `dir_xy_camera`

## Objective

Test a cleaner replacement for old M2 ownership/capacity/rgb-mix:

```text
B_inf(p) = A(p) = medium_rgb(p)
I = rgb_object + rgb_medium_finite + tail_weight * B_inf
```

No CUDA change was used. Old M2 ownership, accumulation-zero, near-zero, rgb-mix, capacity suppression, opacity decay, and hard pruning were kept off for this experiment set.

## Code Changes

- Added `b_inf_mode={implicit,tied,bounded_residual,independent}` and `b_inf_residual_scale`.
- Added explicit closure outputs: `b_inf`, `rgb_tail`, `rgb_implicit_tail`, `b_inf_minus_A_abs`.
- Added background-water color loss using `view_XXXX_regions.pt` masks.
- Added foreground transmission-aware reconstruction loss.
- Added pseudo-depth mask builder:
  - `scripts/diagnostics/build_pseudo_depth_bg_masks.py`
- Added closure diagnostic:
  - `scripts/diagnostics/diagnose_backscatter_closure.py`
- Added experiment scripts under `scripts/experiments/`.

Defaults preserve the previous behavior:

```text
b_inf_mode=implicit
lambda_background_water_color=0
lambda_foreground_transmission_reconstruction=0
medium_predictor_mode=single
lambda_pseudo_depth=0
lambda_medium_context_residual=0
```

## Mask Build

Pseudo-depth masks were generated from:

```text
undistorted_data/undistorted_IUI3-RedSea/depthAnything_u16
```

Command:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/build_pseudo_depth_bg_masks.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --data undistorted_data/undistorted_IUI3-RedSea \
  --depth-dir undistorted_data/undistorted_IUI3-RedSea/depthAnything_u16 \
  --output-dir common_masks/pseudo_depth_bg_iui3_redsea_20260725 \
  --foreground-depth-threshold 0.55 \
  --rgb-grad-threshold 0.06 \
  --depth-grad-threshold 0.06 \
  --erosion-radius 9 \
  --save-png
```

Mask coverage over 25 train views:

| Mask | Mean | Min | Max |
|---|---:|---:|---:|
| water/background | 0.489076 | 0.227793 | 0.752359 |
| object/foreground | 0.161094 | 0.050884 | 0.333675 |
| boundary | 0.016692 | 0.010695 | 0.020200 |

Visual spot check: `view_0000_background_overlay.png` covers most open water and excludes the main reef/foreground core, but coverage is broad. Treat as a first-pass high-coverage proxy, not a final high-precision teacher.

## A0/A1 Closure Equivalence

Diagnostic command:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_backscatter_closure.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --output-dir renders/backscatter_closure_a0_m1_diag_20260725 \
  --max-images 4
```

Result:

| Check | Mean Abs | Max Abs | Pass |
|---|---:|---:|---|
| explicit tied RGB vs implicit M1 tail | 7.69e-09 | 1.19e-07 | yes |

This confirms the explicit Python recomposition matches the current implicit tail within the required threshold.

## Experiments

Baseline references:

| Run | PSNR | SSIM | LPIPS | J Blue | Far Accum | Far Clear | Water J |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 `dir_xy_camera` | 31.1314 | 0.9120 | 0.1750 | 0.1691 | 0.407096 | 0.083962 | 0.000928 |
| old M2 `alpha_depth` | 31.0696 | 0.9129 | 0.1771 | n/a | n/a | n/a | n/a |

Main 15k experiments:

| Experiment | Mechanism | PSNR | SSIM | LPIPS | J Blue | Far Accum | Far Clear | Water J | Obj Ret | Boundary Ret | Binf-A | Closure Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | tied, no bg loss | 31.027380 | 0.913569 | 0.174922 | 0.146856 | 0.462219 | 0.077600 | 0.000740 | 0.985757 | 0.954140 | 0.000000 | 0.000000 |
| B1 | tied, bg=0.001 | 31.014822 | 0.911847 | 0.177040 | 0.164838 | 0.657745 | 0.081046 | 0.000626 | 0.982392 | 0.947498 | 0.000000 | 0.000000 |
| E2/B2 | tied, bg=0.005 | 30.964132 | 0.912216 | 0.177284 | 0.114581 | 0.368988 | 0.068289 | 0.000952 | 0.984439 | 0.968839 | 0.000000 | 0.000000 |
| B3 | tied, bg=0.010 | 31.024960 | 0.912472 | 0.176255 | 0.128324 | 0.470489 | 0.076724 | 0.001363 | 1.005474 | 0.984235 | 0.000000 | 0.000000 |
| E3/D1 | tied, bg=0.005, fg-T=0.5 | 31.211056 | 0.913841 | 0.175490 | 0.131678 | 0.579333 | 0.077692 | 0.000744 | 0.995942 | 0.973079 | 0.000000 | 0.000000 |
| A3 | bounded residual s=0.02, bg=0.005 | 31.295404 | 0.914427 | 0.175311 | 0.121400 | 0.457915 | 0.081729 | 0.001490 | 0.976828 | 0.969459 | 0.003186 | 0.000010 |
| A4 | bounded residual s=0.05, bg=0.005 | 30.903370 | 0.912483 | 0.175186 | 0.178621 | 0.582083 | 0.083771 | 0.002321 | 0.998494 | 1.003716 | 0.002429 | 0.000005 |
| A5 | independent B_inf, bg=0.005 | 30.889668 | 0.912584 | 0.174999 | 0.140472 | 0.380476 | 0.087367 | 0.003103 | 1.014258 | 1.005681 | 0.014978 | 0.000194 |

## Commands

Each experiment was run from an independent wrapper:

```bash
GPU=6 STAMP=20260725_binf_e1 scripts/experiments/backscatter_e1_tied_no_bg_iui3_redsea.sh
GPU=6 STAMP=20260725_binf_b1 scripts/experiments/backscatter_b1_tied_bg001_iui3_redsea.sh
GPU=7 STAMP=20260725_binf_e2 scripts/experiments/backscatter_e2_tied_bg005_iui3_redsea.sh
GPU=7 STAMP=20260725_binf_b3 scripts/experiments/backscatter_b3_tied_bg010_iui3_redsea.sh
GPU=8 STAMP=20260725_binf_e3 scripts/experiments/backscatter_e3_tied_bg005_fg05_iui3_redsea.sh
GPU=9 STAMP=20260725_binf_a3 scripts/experiments/backscatter_a3_bounded_residual_bg005_iui3_redsea.sh
GPU=8 STAMP=20260725_binf_a4 scripts/experiments/backscatter_a4_bounded_residual_bg005_iui3_redsea.sh
GPU=9 STAMP=20260725_binf_a5 scripts/experiments/backscatter_a5_independent_bg005_iui3_redsea.sh
```

All scripts use:

```text
/opt/anaconda3/envs/water_splatting
DATA_PATH=/mnt/new/home_old/ycy/water-splatting-refactor/undistorted_data/undistorted_IUI3-RedSea
OUTPUT_DIR=/mnt/new/home_old/ycy/water-splatting-refactor/outputs
RENDER_ROOT=/mnt/new/home_old/ycy/water-splatting-refactor/renders
LOG_ROOT=/mnt/new/home_old/ycy/water-splatting-refactor/logs
```

## Checkpoints

| Experiment | Checkpoint |
|---|---|
| E1 | `outputs/binf_e1_tied_no_bg_dir_xy_camera_iui3_redsea_15000/water-splatting/binf_e1_tied_no_bg_dir_xy_camera_iui3_redsea_15000_20260725_binf_e1/nerfstudio_models/step-000014999.ckpt` |
| B1 | `outputs/binf_b1_tied_bg001_dir_xy_camera_iui3_redsea_15000/water-splatting/binf_b1_tied_bg001_dir_xy_camera_iui3_redsea_15000_20260725_binf_b1/nerfstudio_models/step-000014999.ckpt` |
| E2/B2 | `outputs/binf_e2_tied_bg005_dir_xy_camera_iui3_redsea_15000/water-splatting/binf_e2_tied_bg005_dir_xy_camera_iui3_redsea_15000_20260725_binf_e2/nerfstudio_models/step-000014999.ckpt` |
| B3 | `outputs/binf_b3_tied_bg010_dir_xy_camera_iui3_redsea_15000/water-splatting/binf_b3_tied_bg010_dir_xy_camera_iui3_redsea_15000_20260725_binf_b3/nerfstudio_models/step-000014999.ckpt` |
| E3/D1 | `outputs/binf_e3_tied_bg005_fg05_dir_xy_camera_iui3_redsea_15000/water-splatting/binf_e3_tied_bg005_fg05_dir_xy_camera_iui3_redsea_15000_20260725_binf_e3/nerfstudio_models/step-000014999.ckpt` |
| A3 | `outputs/binf_a3_bounded_residual_s002_bg005_dir_xy_camera_iui3_redsea_15000/water-splatting/binf_a3_bounded_residual_s002_bg005_dir_xy_camera_iui3_redsea_15000_20260725_binf_a3/nerfstudio_models/step-000014999.ckpt` |
| A4 | `outputs/binf_a4_bounded_residual_s005_bg005_dir_xy_camera_iui3_redsea_15000/water-splatting/binf_a4_bounded_residual_s005_bg005_dir_xy_camera_iui3_redsea_15000_20260725_binf_a4/nerfstudio_models/step-000014999.ckpt` |
| A5 | `outputs/binf_a5_independent_bg005_dir_xy_camera_iui3_redsea_15000/water-splatting/binf_a5_independent_bg005_dir_xy_camera_iui3_redsea_15000_20260725_binf_a5/nerfstudio_models/step-000014999.ckpt` |

## Interpretation

1. Explicit `B_inf=A` is numerically valid. The closure diagnostic passes by a large margin.
2. Tied background supervision at `lambda=0.005` is the best leakage reducer in this sweep:
   - far accumulation: `0.368988` vs M1 `0.407096`
   - far clear luma: `0.068289` vs M1 `0.083962`
   - J blue dominance: `0.114581` vs M1 `0.1691`
   - but PSNR and LPIPS fail the success criteria.
3. Foreground transmission-aware loss (`E3/D1`) recovers and improves RGB metrics:
   - PSNR `31.2111`, SSIM `0.9138`, LPIPS `0.1755`
   - but far accumulation worsens to `0.579333`, so it is not a leakage solution.
4. Bounded residual `s=0.02` (`A3`) gives the strongest PSNR/SSIM and acceptable LPIPS delta:
   - PSNR `31.2954`
   - SSIM `0.9144`
   - LPIPS `0.1753`
   - but far accumulation and water J are worse than M1/E2.
5. Larger residual (`A4`) and independent `B_inf` (`A5`) are not useful:
   - A4 worsens PSNR and J blue.
   - A5 decouples too much from `A` (`Binf-A mean=0.014978`) and increases far clear/water J.

## Decision

No tested configuration should replace M1 as the new reliable mainline yet.

Recommended labels:

```text
Best reconstruction candidate: A3 bounded_residual s=0.02 bg=0.005
Best leakage candidate: E2/B2 tied bg=0.005
Best balanced-but-not-leakage-safe candidate: E3/D1 tied bg=0.005 fg-T=0.5
Do not continue: A4 s=0.05, A5 independent
```

## Next Step

The main blocker is not explicit closure. It is that background supervision alone changes medium color but does not reliably stop open-water pixels from driving Gaussian growth.

Recommended next experiment:

1. Improve pseudo-depth masks for precision:
   - lower water coverage target;
   - inspect 20 train overlays;
   - try erosion radius `13` and stricter RGB/depth edge exclusion.
2. Implement diagnostic-only background-excluded densification first:
   - count gradient samples landing in `M_bg`, `M_fg`, `M_boundary`;
   - do not prune or decay opacity yet.
3. If diagnostic confirms open-water densification pressure, test `F1`:
   - background pixels excluded from split gradient accumulation;
   - RGB/medium losses still use all pixels.
4. Keep `E2/B2` as leakage reference and `A3` as reconstruction reference for the next comparison.
