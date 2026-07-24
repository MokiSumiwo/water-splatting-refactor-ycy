# M2 Infinite-Water Ownership - 2026-07-23

## Implementation

Added a default-off M2 path:

- `infinite_water_enabled=False` preserves original behavior and checkpoint state.
- When enabled, medium MLP output expands from 9 to 12 channels:
  - `medium_rgb`: 3
  - `medium_bs`: 3
  - `medium_attn`: 3
  - `b_inf`: 3
- Ownership modes:
  - `alpha_only`
  - `alpha_depth`
  - `alpha_depth_color`
- Evidence is internal-only:
  - low object accumulation
  - rendered expected depth, normalized per view
  - optional B_inf color similarity
- No pseudo-depth label is used.
- Occupancy limit is enabled by default when M2 is enabled:

```text
m_inf_eff = m_inf * (1 - accumulation)
```

Final composition:

```text
rgb = (1 - m_inf_eff) * rgb_near + m_inf_eff * b_inf
rgb_clear = (1 - m_inf_eff) * rgb_clear_gaussian
```

## Losses

All M2 auxiliary losses are independently weighted and ramped:

- `lambda_infinite_water_binf_rgb`
- `lambda_infinite_water_accumulation_zero`
- `lambda_infinite_water_near_zero`

Default weights are all `0.0`.

## Validation

Default-off equivalence:

```text
default_m2_off_equivalence_ok
```

M2 enabled forward/backward smoke:

```text
keys: b_inf, m_inf, m_inf_eff
main_loss: ok
infinite_water_binf_rgb_loss: ok
infinite_water_accumulation_zero_loss: ok
infinite_water_near_zero_loss: ok
backward_ok
```

CLI smoke completed:

```text
m2_smoke_alpha_depth_iui3_redsea_20260723_075927
checkpoint: /mnt/new/home_old/ycy/water-splatting-refactor/outputs/m2_smoke_alpha_depth_iui3_redsea/water-splatting/m2_smoke_alpha_depth_iui3_redsea_20260723_075927/nerfstudio_models/step-000000004.ckpt
```

## Experiment Script

Smoke:

```bash
GPU=7 MAX_NUM_ITERATIONS=5 RUN_EVAL=0 \
  EXPERIMENT_NAME=m2_smoke_alpha_depth_iui3_redsea \
  bash scripts/experiments/m2_infinite_water_iui3_redsea.sh
```

Candidate main run:

```bash
GPU=7 OWNERSHIP_MODE=alpha_depth MAX_NUM_ITERATIONS=15000 \
  EXPERIMENT_NAME=m2_alpha_depth_dir_xy_camera_iui3_redsea_15000 \
  bash scripts/experiments/m2_infinite_water_iui3_redsea.sh
```

## Main Run Result

Completed:

```text
experiment: m2_alpha_depth_dir_xy_camera_iui3_redsea_15000
timestamp:  m2_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_080936
mode:       alpha_depth
context:    dir_xy_camera
```

Checkpoint:

```text
/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000/water-splatting/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_080936/nerfstudio_models/step-000014999.ckpt
```

Eval:

```text
/mnt/new/home_old/ycy/water-splatting-refactor/renders/m2_alpha_depth_dir_xy_camera_iui3_redsea_15000_20260723_080936/output.json
```

Metrics:

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

Rendered diagnostic maps were saved alongside eval RGB/depth outputs:

```text
eval_m_inf_*.png
eval_m_inf_eff_*.png
eval_b_inf_*.png
```

Interpretation:

- M2 does not collapse the underwater metric baseline and remains clearly above the original model.
- The small PSNR/LPIPS regression relative to M1 suggests the current `alpha_depth` losses should stay conservative.
- This run is suitable as the input branch for M3 diagnostic cleanup, but not yet evidence for enabling destructive Gaussian pruning.
