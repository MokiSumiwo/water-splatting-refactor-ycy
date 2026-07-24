# M3 Contribution-Aware Gaussian Cleanup - 2026-07-23

## Implementation

Added a default-off M3 cleanup path:

- `gaussian_cleanup_enabled=False` preserves original behavior.
- `gaussian_cleanup_dry_run=True` logs candidates without deleting Gaussians.
- Cleanup runs during `refinement_after` after `gaussian_cleanup_start_step` and every `gaussian_cleanup_interval`.
- Candidate evidence uses internal training signals only:
  - projected gradient contribution proxy
  - opacity
  - projected object accumulation sampled at Gaussian screen centers
  - projected M2 infinite-water ownership sampled at Gaussian screen centers
  - optional average projected depth gate

Added helper module:

```text
water_splatting/cleanup/contribution_cleanup.py
```

The helper is tensor-only and side-effect-free; model deletion is guarded by `gaussian_cleanup_dry_run`.

## Config Flags

```text
gaussian_cleanup_enabled
gaussian_cleanup_dry_run
gaussian_cleanup_start_step
gaussian_cleanup_interval
gaussian_cleanup_contribution_threshold
gaussian_cleanup_opacity_threshold
gaussian_cleanup_visibility_min_count
gaussian_cleanup_alpha_threshold
gaussian_cleanup_depth_threshold
gaussian_cleanup_ownership_threshold
gaussian_cleanup_ownership_source
gaussian_cleanup_require_alpha_gate
gaussian_cleanup_require_depth_gate
gaussian_cleanup_require_ownership_gate
```

## Validation

Static checks:

```bash
/opt/anaconda3/bin/conda run -n water_splatting python -m compileall -q water_splatting
git diff --check
```

Smoke:

```bash
GPU=8 MAX_NUM_ITERATIONS=620 RUN_EVAL=0 CLEANUP_START_STEP=600 CLEANUP_INTERVAL=100 \
  EXPERIMENT_NAME=m3_smoke_cleanup_diag_iui3_redsea \
  bash scripts/experiments/m3_cleanup_diagnostic_iui3_redsea.sh
```

Smoke result:

```text
M3 cleanup dry-run step=600 candidates=0/21907
low_contrib=2397
opacity_gate=7197
alpha_gate=0
ownership_gate=0
mean_alpha=0.979414
mean_ownership=0.000321
```

Ownership-source CLI parse smoke:

```bash
GPU=9 MAX_NUM_ITERATIONS=1 RUN_EVAL=0 CLEANUP_OWNERSHIP_SOURCE=m_inf \
  EXPERIMENT_NAME=m3_cli_ownership_source_smoke_iui3_redsea \
  bash scripts/experiments/m3_cleanup_diagnostic_iui3_redsea.sh
```

Result: config parsed successfully with `gaussian_cleanup_ownership_source='m_inf'`.

## Main Diagnostic Run

Command:

```bash
GPU=8 MAX_NUM_ITERATIONS=15000 RUN_EVAL=1 CLEANUP_DRY_RUN=True \
  CLEANUP_START_STEP=12000 CLEANUP_INTERVAL=500 \
  EXPERIMENT_NAME=m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m3_cleanup_diagnostic_iui3_redsea.sh
```

Checkpoint:

```text
/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_083104/nerfstudio_models/step-000014999.ckpt
```

Eval:

```text
/mnt/new/home_old/ycy/water-splatting-refactor/renders/m3_cleanup_diag_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_083104/output.json
```

Metrics:

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

Dry-run cleanup statistics:

```text
step=12000 candidates=0/815378 low_contrib=62055 opacity_gate=27 alpha_gate=13 ownership_gate=19 mean_alpha=0.996021 mean_ownership=0.000134
step=12500 candidates=0/813218 low_contrib=55895 opacity_gate=36 alpha_gate=12 ownership_gate=17 mean_alpha=0.996143 mean_ownership=0.000130
step=13000 candidates=0/811610 low_contrib=56269 opacity_gate=27 alpha_gate=7  ownership_gate=20 mean_alpha=0.996251 mean_ownership=0.000126
step=13500 candidates=0/809914 low_contrib=52990
step=14000 candidates=0/808211 low_contrib=50590
step=14500 candidates=0/806855 low_contrib=53676 opacity_gate=8 alpha_gate=7 ownership_gate=12 mean_alpha=0.996380 mean_ownership=0.000112
```

## Interpretation

- Dry-run cleanup did not delete Gaussians and produced zero candidates under the conservative alpha+ownership gates.
- The sampled ownership signal is extremely sparse at projected Gaussian centers, so active pruning is not justified yet.
- Metrics remain above the original baseline; the M3 dry-run metric changes should be treated as fresh-run variance/diagnostic overhead, not a pruning effect.
- Follow-up far-water residual diagnosis on the M2 checkpoint shows the far water region is already effectively object-free, so M3 should be deprioritized for IUI3-RedSea unless new scenes show visible residual Gaussians.
- If M3 is revisited, keep it diagnostic-only first:
  - disable ownership gate to measure low-contribution/opacity overlap
  - relax alpha threshold to measure pure-water support sensitivity
  - optionally use `m_inf` rather than `m_inf_eff` for Gaussian-center ownership sampling

## Far-Water Residual Check on M2

Added a checkpoint-level diagnostic script:

```text
scripts/diagnostics/diagnose_far_water_residual.py
```

Command:

```bash
CUDA_VISIBLE_DEVICES=9 /opt/anaconda3/bin/conda run -n water_splatting python \
  scripts/diagnostics/diagnose_far_water_residual.py \
  --load-config outputs/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_080936/config.yml \
  --output-dir logs/diagnostics/m2_alpha_depth_far_water_residual_20260723 \
  --far-depth-quantile 0.90 \
  --alpha-threshold 0.05 \
  --object-threshold 0.03 \
  --save-heatmaps
```

Result JSON:

```text
logs/diagnostics/m2_alpha_depth_far_water_residual_20260723/far_water_residual_diagnostic.json
```

Aggregate far-depth-region statistics:

```text
far_pixels = 1,067,972
far_accumulation_mean = 0.00048002792755141854
far_accumulation_p95 = 0.0
far_accumulation_max = 0.9978291988372803
far_rgb_object_luma_mean = 0.000024907762053771876
far_rgb_object_luma_p95 = 0.0
far_rgb_object_luma_max = 0.25764328241348267
far_m_inf_eff_mean = 0.9235095381736755
far_alpha_gt_0.05_fraction = 0.001198533340357244
far_object_gt_0.03_fraction = 0.00023408854031004012
```

Per-image far residual fractions:

```text
image 0: alpha>0.05 0.000473, object>0.03 0.000293
image 1: alpha>0.05 0.003040, object>0.03 0.000192
image 2: alpha>0.05 0.002420, object>0.03 0.000342
image 3: alpha>0.05 0.000160, object>0.03 0.000057
```

Interpretation:

- Most far-depth pixels have exactly zero object accumulation and zero object luma through the 95th percentile.
- M2 assigns strong infinite-water ownership in the same region (`m_inf_eff_mean ~= 0.924`).
- The remaining high-alpha/high-object pixels are extremely sparse and likely edge/foreground-depth contamination rather than broad far-water Gaussian residuals.
- For IUI3-RedSea, active M3 cleanup is not currently justified; effort should shift to M4 constrained appearance / dewatered-image color quality.
