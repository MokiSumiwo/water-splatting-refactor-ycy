# Step 1 Equivalent Refactor - 2026-07-23

## Scope

This refactor keeps the default WaterSplatting behavior numerically equivalent while separating the main model into small helpers:

- `water_splatting/fields/medium_field.py`
  - Wraps original direction-conditioned medium MLP logic.
  - Does not own trainable modules; checkpoint keys remain `direction_encoding.*` and `medium_mlp.*`.
- `water_splatting/fields/gaussian_appearance.py`
  - Wraps original SH / sigmoid Gaussian color computation.
- `water_splatting/rendering/underwater_rasterizer.py`
  - Wraps original projection, underwater rasterization, RGB composition, clear RGB transform, alpha, and depth normalization.
- `water_splatting/losses/reconstruction.py`
  - Wraps original relative-L1/L2 and SSIM reconstruction objective.

No new research mechanism is enabled in Step 1.

## Compatibility

- Baseline checkpoint path:
  `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/baseline_original_watersplatting_iui3_redsea/water-splatting/orig_watersplatting_iui3_redsea_15000_20260723_063201/nerfstudio_models/step-000014999.ckpt`
- Baseline eval:
  `PSNR=29.8790, SSIM=0.9105, LPIPS=0.1810`
- The helper objects are intentionally not `nn.Module` subclasses, so they do not add checkpoint state.
- Verified current editable install points to this repository.
- Verified CUDA backend imports through `water_splatting.cuda._backend`.

## Regression Check

Command run:

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m compileall -q water_splatting
```

Dynamic old-vs-new CUDA render regression:

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

Result: `equivalence_regression_ok`.

## Sanity Eval Script

Use:

```bash
GPU=6 bash scripts/experiments/step1_refactor_sanity_eval.sh
```

The script writes eval renders under `renders/` and logs under `logs/`, both ignored by git.

## Sanity Eval Result

Run:

```text
step1_refactor_sanity_iui3_redsea_20260723_071025
```

Output:

```text
PSNR  = 29.879043579101562
SSIM  = 0.9104903936386108
LPIPS = 0.18103083968162537
```

This matches the recorded original baseline metrics.

Post-M1 compatibility rerun:

```text
step1_refactor_sanity_iui3_redsea_20260723_072224
PSNR  = 29.879043579101562
SSIM  = 0.9104903936386108
LPIPS = 0.18103083968162537
```
