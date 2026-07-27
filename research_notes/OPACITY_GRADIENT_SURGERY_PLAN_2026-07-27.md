# Opacity Gradient Surgery Plan

Date: 2026-07-27

Repository: `/mnt/new/home_old/ycy/water-splatting-refactor`

Branch: `refactor/core-framework`

Starting commit: `36f51812966cd76d1e7c0e1391ce31d50f20aed8`

## Current Conclusion

The main blocker is no longer `B_inf` definition or mask precision. The current
problem is attribution through the trainable Gaussian path:

- `J_gaussian_raw` and `final_transmittance` do not provide useful Gaussian
  gradients through the current CUDA backward path.
- `rgb_tail` affects the medium side, not Gaussian opacity/scale/color.
- `J_proxy_raw` is numerically equivalent to `J_gaussian_raw` in audited views
  and reaches Gaussian parameters through the supported `out_img` backward path.
- Proxy chroma suppression is causal and monotonic within a shared 10k resume
  branch, but C5/C6 are not formal replacements for M1 because the resume
  control underperforms the uninterrupted N1/M1 references.

Therefore the next mainline is:

1. fix resume-control mismatch;
2. generate train-view contribution candidates only from train masks/views;
3. apply conservative object-protected opacity-gradient modulation to those
   candidates.

## Corrections Applied Before New Runs

- Gate clear-proxy rasterization by
  `background_clear_chroma_start_step` during training, unless
  `clear_proxy_enabled=True`.
- Use a separate `xys_grad_abs_proxy` buffer for clear-proxy renders so proxy
  loss does not contaminate the main densification gradient accumulator.
- Reject non-1D custom background tensors in `rasterize_gaussians()` instead of
  silently taking the first pixel.
- Warn when `lambda_background_clear_gaussian > 0` because that loss targets the
  dead-gradient `J_gaussian_raw` path.
- Extend contribution sensitivity diagnostics with:
  - `--split train|eval`;
  - `features_rest` gradient sensitivity;
  - active-Gaussian top-fraction statistics;
  - fixed top-50/100/500 statistics;
  - cumulative 50/80/90% sensitivity counts;
  - bounded, train-view candidate mask export.

## Priority 0: Resume Control

Run an uninterrupted N1 control with checkpoint retention:

```bash
GPU=6 \
  scripts/experiments/bg_attr_n1_uninterrupted_control_iui3.sh
```

Key config:

```text
MAX_NUM_ITERATIONS=15000
MODEL_NUM_STEPS=15000
STEPS_PER_SAVE=5000
SAVE_ONLY_LATEST_CHECKPOINT=False
medium_context_mode=dir_xy_camera
b_inf_mode=tied
lambda_background_water_color=0.005
lambda_background_clear_chroma=0.0
seed=42
```

Expected checkpoint:

```text
outputs/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control/water-splatting/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control_20260727_n1_uninterrupted_control/nerfstudio_models/step-000010000.ckpt
```

Then run R0 from that exact step-10000 checkpoint:

```bash
GPU=7 \
  scripts/experiments/bg_attr_r0_resume10k_control_iui3.sh
```

Pass criteria versus uninterrupted final:

| metric | max allowed difference |
| --- | ---: |
| PSNR | 0.02 dB |
| SSIM | 0.0003 |
| LPIPS | 0.0003 |
| Far Accum | 3% |
| J Blue | 3% |

If R0 fails, inspect dataloader RNG state, optimizer/scheduler state, callback
state, resume step offset, opacity reset, and culling/refinement state before
promoting any C/G run.

## Priority 1: Train-View Candidate Mask

Generate candidates from train views only:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_gaussian_region_sensitivity.py \
  --load-config outputs/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control/water-splatting/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control_20260727_n1_uninterrupted_control/config.yml \
  --load-step 10000 \
  --mask-dir common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726 \
  --output-json renders/gradient_surgery_20260727/train_region_sensitivity_step10000.json \
  --candidate-output-prefix renders/gradient_surgery_20260727/candidate_mask_step10000 \
  --split train \
  --max-images 25 \
  --enable-clear-proxy \
  --candidate-min-view-count 5 \
  --candidate-water-quantile 0.995 \
  --candidate-proxy-quantile 0.95 \
  --candidate-object-ratio-max 0.10 \
  --candidate-boundary-ratio-max 0.10 \
  --candidate-require-proxy \
  --top-k 100
```

Candidate output:

```text
renders/gradient_surgery_20260727/candidate_mask_step10000.pt
renders/gradient_surgery_20260727/candidate_mask_step10000.json
```

Selection rule:

```text
train view support >= 5
water accumulation opacity sensitivity >= q99.5
water proxy-bluegreen opacity sensitivity >= q95
object / water sensitivity <= 0.10
boundary / water sensitivity <= 0.10
```

Eval views may be used only for validation, not for training candidate indices.

## Priority 2: Opacity Gradient Surgery

Run from the shared train-view step-10000 checkpoint and candidate mask.

G1:

```bash
GPU=8 \
  scripts/experiments/bg_attr_g1_opacity_surgery_x2_iui3.sh
```

G2:

```bash
GPU=9 \
  scripts/experiments/bg_attr_g2_opacity_surgery_x4_iui3.sh
```

Shared config:

```text
lambda_background_clear_chroma=0.002
background_gradient_surgery_enabled=True
background_candidate_mask_path=renders/gradient_surgery_20260727/candidate_mask_step10000.pt
background_opacity_decrease_multiplier={2.0,4.0}
background_opacity_increase_multiplier=1.0
background_gradient_surgery_start_step=10000
background_gradient_surgery_min_view_count=5
```

Interpretation: positive opacity-logit gradients are the existing optimizer
signal to reduce opacity under gradient descent. The surgery only amplifies that
existing signal for object-protected train-view candidates; it does not inject a
constant opacity decay.

## Success Criteria

Relative to the matched C5 branch:

```text
Far Accum: down another 8%-12%
Water Accum: down another 8%-12%
Far Clear: down at least 10%
Water J: down at least 15%
PSNR drop: <= 0.03 dB
Object accumulation retention: >= 0.985
```

Relative to formal M1, the model still needs:

```text
PSNR drop <= 0.05 dB
SSIM drop <= 0.001
LPIPS increase <= 0.001
Far Clear down >= 25%
Water J down >= 25%
J Blue down >= 20%
Object J Retention >= 0.97
Boundary Retention >= 0.95
```

## Stop Conditions

Stop or revise before running stronger interventions if:

- R0 does not match uninterrupted N1 within tolerance;
- train-view candidate count is zero or dominated by boundary/object overlap;
- G1/G2 lowers opacity but increases scale or footprint enough to preserve far
  accumulation;
- object accumulation retention drops below `0.985` relative to matched C5.

Do not re-enable accumulation-zero loss, hard pruning, opacity decay, capacity
floor, or old M2 ownership as the mainline.
