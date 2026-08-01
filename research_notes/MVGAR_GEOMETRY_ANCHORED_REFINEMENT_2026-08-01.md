# MV-GAR Geometry-Anchored Refinement

Date: 2026-08-01

## Objective

Implement and test MV-GAR, a training-only module for M1 that uses aligned pseudo depth to softly anchor reliable front-surface Gaussian means. The module does not change the underwater renderer or inference outputs.

M1 baseline configuration remains:

- `medium_context_mode=dir_xy_camera`
- `b_inf_mode=tied`
- `B_inf = medium_rgb = A`
- Historical J/TBAP/TMICA/TACMD/capacity/cleanup directions remain disabled in the experiment scripts.

## Code Changes

- Added `water_splatting/geometry/mvgar.py`
  - Loads `view_XXXX_mvgar.pt` pseudo-depth payloads.
  - Builds detached Sobel + high-pass detail maps.
  - Samples pseudo depth, confidence, structure support, render reliability, and front-surface gates at Gaussian projection centers.
  - Computes weighted log-depth Huber surface anchor loss.
  - Selects conservative MV-GAR densification candidates from multi-view evidence buffers.
- Integrated MV-GAR into `water_splatting/water_splatting.py`
  - Added default-off `WaterSplattingModelConfig` flags.
  - Exposes differentiable per-Gaussian camera depths as `outputs["gaussian_depths"]`.
  - Adds `mvgar_surface_loss` only during the configured training window.
  - Maintains per-Gaussian MV-GAR buffers with split/duplicate/cull synchronization.
  - Logs surface/refinement diagnostics to JSONL.
- Added `scripts/preprocess/build_mvgar_pseudo_depth.py`
  - Uses Nerfstudio `ColmapDataParser` train-view order, so `view_XXXX_mvgar.pt` matches training `camera_index`.
  - Aligns DepthAnything u16 relative depth to COLMAP sparse inverse depth with robust affine fitting.
  - Saves `depth`, `pseudo_confidence`, `structure_confidence`, and `boundary_safe`.
- Updated `scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh`
  - Passes MV-GAR flags.
  - Explicitly keeps cleanup/pruning and old auxiliary branches off.
- Added 5k wrapper scripts:
  - `scripts/experiments/mvgar_5k_common.sh`
  - `scripts/experiments/mvgar_g0_m1_japanesegradens_5k.sh`
  - `scripts/experiments/mvgar_g1_surface001_japanesegradens_5k.sh`
  - `scripts/experiments/mvgar_g2_surface002_japanesegradens_5k.sh`
  - `scripts/experiments/mvgar_g3_surface001_dens_japanesegradens_5k.sh`
  - `scripts/experiments/mvgar_g0_m1_iui3_5k.sh`
  - `scripts/experiments/mvgar_g1_surface001_iui3_5k.sh`
  - `scripts/experiments/mvgar_g2_surface002_iui3_5k.sh`
  - `scripts/experiments/mvgar_g3_surface001_dens_iui3_5k.sh`

## New Config Flags

```python
mvgar_enabled: bool = False
mvgar_diagnostic_only: bool = False
mvgar_pseudo_depth_dir: Optional[str] = None
mvgar_camera_graph_path: Optional[str] = None
mvgar_log_path: Optional[str] = None
mvgar_start_step: int = 1500
mvgar_ramp_steps: int = 1500
mvgar_stop_step: int = 10000
lambda_mvgar_surface: float = 0.01
mvgar_surface_huber_delta: float = 0.05
mvgar_accumulation_mid: float = 0.45
mvgar_accumulation_temp: float = 0.08
mvgar_depth_std_kappa: float = 0.20
mvgar_front_depth_log_tau: float = 0.08
mvgar_min_pseudo_confidence: float = 0.50
mvgar_densification_enabled: bool = False
mvgar_min_view_count: int = 3
mvgar_min_mean_weight: float = 0.20
mvgar_detail_quantile: float = 0.85
mvgar_depth_variance_threshold: float = 0.02
mvgar_max_extra_ratio_to_base: float = 0.25
mvgar_max_extra_fraction_per_refine: float = 0.002
mvgar_concentration_enabled: bool = False
lambda_mvgar_concentration: float = 0.002
mvgar_concentration_target: float = 0.12
```

## Pseudo-Depth Preprocessing

Commands:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/preprocess/build_mvgar_pseudo_depth.py \
  --data undistorted_data/undistorted_JapaneseGradens-RedSea \
  --depth-dir undistorted_data/undistorted_JapaneseGradens-RedSea/depthAnything_u16 \
  --output-dir outputs/mvgar_pseudo_depth/japanesegradens_train \
  --skip-cross-view \
  --ransac-residual-threshold 0.25 \
  --max-median-relative-error 0.25

/opt/anaconda3/envs/water_splatting/bin/python scripts/preprocess/build_mvgar_pseudo_depth.py \
  --data undistorted_data/undistorted_IUI3-RedSea \
  --depth-dir undistorted_data/undistorted_IUI3-RedSea/depthAnything_u16 \
  --output-dir outputs/mvgar_pseudo_depth/iui3_train \
  --skip-cross-view \
  --ransac-residual-threshold 0.25 \
  --max-median-relative-error 0.25
```

Results:

| Scene | Train Views | Aligned OK | Mean Confidence |
| --- | ---: | ---: | ---: |
| JapaneseGradens | 17 | 17 | 0.7580 |
| IUI3 | 25 | 24 | 0.5545 |

The initial cross-view confidence variant was too sparse for V0. The 5k wrappers therefore default to sparse-alignment confidence with structural support. Multi-view evidence still accumulates over repeated train views for candidate diagnostics.

## Smoke Tests

Checks:

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/water_splatting.py \
  water_splatting/geometry/mvgar.py \
  scripts/preprocess/build_mvgar_pseudo_depth.py

bash -n scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh
bash -n scripts/experiments/mvgar_5k_common.sh
git diff --check
```

Smoke training:

```bash
GPU=6 SCENE_SLUG=japanesegradens \
DATA_PATH=/mnt/new/home_old/ycy/water-splatting-refactor/undistorted_data/undistorted_JapaneseGradens-RedSea \
EXPERIMENT_TAG=smoke_dens_buffer2 MAX_NUM_ITERATIONS=800 RUN_EVAL=0 RUN_CLOSURE_DIAG=0 \
BUILD_MVGAR_DEPTH=0 \
MVGAR_PSEUDO_DEPTH_DIR=/mnt/new/home_old/ycy/water-splatting-refactor/outputs/mvgar_pseudo_depth/japanesegradens_train \
MVGAR_ENABLED=True LAMBDA_MVGAR_SURFACE=0.01 MVGAR_DENSIFICATION_ENABLED=True \
MVGAR_START_STEP=0 MVGAR_RAMP_STEPS=0 MVGAR_STOP_STEP=900 \
MVGAR_MIN_PSEUDO_CONFIDENCE=0.35 MVGAR_ACCUMULATION_MID=0.20 MVGAR_FRONT_DEPTH_LOG_TAU=0.20 \
scripts/experiments/mvgar_5k_common.sh
```

Smoke result:

- Training completed without CUDA/autograd errors.
- MV-GAR surface JSONL wrote normally.
- Split/duplicate/cull buffer synchronization worked after fixing sync to use pre-append masks.
- At step 700, supported buffer count was 21,140 and no extra candidates were selected because log-depth variance was much higher than the conservative default threshold.

## 5k Commands

```bash
STAMP=20260801_mvgar5k_round1 GPU=6 scripts/experiments/mvgar_g0_m1_japanesegradens_5k.sh
STAMP=20260801_mvgar5k_round1 GPU=7 MVGAR_PSEUDO_DEPTH_DIR=outputs/mvgar_pseudo_depth/japanesegradens_train scripts/experiments/mvgar_g1_surface001_japanesegradens_5k.sh
STAMP=20260801_mvgar5k_round1 GPU=8 MVGAR_PSEUDO_DEPTH_DIR=outputs/mvgar_pseudo_depth/japanesegradens_train scripts/experiments/mvgar_g2_surface002_japanesegradens_5k.sh

STAMP=20260801_mvgar5k_round1 GPU=6 scripts/experiments/mvgar_g0_m1_iui3_5k.sh
STAMP=20260801_mvgar5k_round1 GPU=7 MVGAR_PSEUDO_DEPTH_DIR=outputs/mvgar_pseudo_depth/iui3_train scripts/experiments/mvgar_g1_surface001_iui3_5k.sh
```

Tuned run:

```bash
STAMP=20260801_mvgar5k_round1 GPU=6 SCENE_SLUG=japanesegradens \
DATA_PATH=/mnt/new/home_old/ycy/water-splatting-refactor/undistorted_data/undistorted_JapaneseGradens-RedSea \
EXPERIMENT_TAG=g1b_surface0005 LAMBDA_MVGAR_SURFACE=0.005 \
MVGAR_ENABLED=True MVGAR_DENSIFICATION_ENABLED=False \
MVGAR_PSEUDO_DEPTH_DIR=outputs/mvgar_pseudo_depth/japanesegradens_train \
scripts/experiments/mvgar_5k_common.sh

STAMP=20260801_mvgar5k_round1 GPU=7 SCENE_SLUG=iui3 \
DATA_PATH=/mnt/new/home_old/ycy/water-splatting-refactor/undistorted_data/undistorted_IUI3-RedSea \
EXPERIMENT_TAG=g1b_surface0005 LAMBDA_MVGAR_SURFACE=0.005 \
MVGAR_ENABLED=True MVGAR_DENSIFICATION_ENABLED=False \
MVGAR_PSEUDO_DEPTH_DIR=outputs/mvgar_pseudo_depth/iui3_train \
scripts/experiments/mvgar_5k_common.sh
```

## 5k Results

JapaneseGradens:

| Experiment | PSNR | Delta | SSIM | Delta | LPIPS | Delta | Gaussians | Growth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 M1 | 22.8341 | 0.0000 | 0.8011 | 0.0000 | 0.2394 | 0.0000 | 718,272 | 0.00% |
| G1 surface 0.01 | 22.9683 | +0.1342 | 0.8016 | +0.0005 | 0.2400 | +0.0006 | 717,102 | -0.16% |
| G1b surface 0.005 | 22.9843 | +0.1502 | 0.8029 | +0.0018 | 0.2440 | +0.0046 | 715,746 | -0.35% |
| G2 surface 0.02 | 22.7875 | -0.0466 | 0.8012 | +0.0001 | 0.2397 | +0.0003 | 716,981 | -0.18% |

IUI3 safety:

| Experiment | PSNR | Delta | SSIM | Delta | LPIPS | Delta | Gaussians | Growth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 M1 | 25.8273 | 0.0000 | 0.7652 | 0.0000 | 0.3043 | 0.0000 | 646,264 | 0.00% |
| G1 surface 0.01 | 25.6428 | -0.1845 | 0.7658 | +0.0006 | 0.3045 | +0.0002 | 641,735 | -0.70% |
| G1b surface 0.005 | 25.7536 | -0.0737 | 0.7651 | -0.0001 | 0.3069 | +0.0026 | 640,921 | -0.83% |

## Gate Decision

5k gate requires JapaneseGradens PSNR improvement, LPIPS decrease, SSIM non-degradation, and IUI3 safety.

- G1 surface 0.01 improves JapaneseGradens PSNR and SSIM, but LPIPS worsens and IUI3 PSNR drops by 0.1845 dB.
- G1b surface 0.005 improves JapaneseGradens PSNR/SSIM more, but LPIPS worsens further and IUI3 still fails safety.
- G2 surface 0.02 does not improve JapaneseGradens PSNR and also worsens LPIPS.
- Conservative densification was not advanced to formal G3 5k because surface-anchor-only failed the LPIPS and IUI3 safety gates. The 800-step densification smoke showed the buffers are functional but default pseudo-depth error variance is too high for candidate eligibility.

Decision: do not enter 15k with MV-GAR V0.

## Interpretation

MV-GAR surface anchoring produces a real PSNR/SSIM signal on JapaneseGradens without increasing Gaussian count, but the effect is not safe. The IUI3 PSNR drop and LPIPS degradation suggest the current pseudo-depth anchor is over-constraining geometry or aligning to a depth prior that is not reliable enough in all underwater scenes.

This does not confirm that M1's remaining loss is primarily Gaussian detail/refinement capacity. It suggests a narrower statement: some JapaneseGradens error is sensitive to Gaussian surface depth, but the current pseudo-depth supervision is not robust enough to be a main contribution.

## Next Steps

- Add a stricter pseudo-depth reliability diagnostic before any further MV-GAR training.
- Revisit true cross-view confidence rather than sparse-alignment-only confidence.
- Consider using pseudo depth only as candidate evidence, not a direct surface loss, if IUI3 remains sensitive.
- Stop before 15k until JapaneseGradens LPIPS and IUI3 PSNR safety both pass at 5k.
