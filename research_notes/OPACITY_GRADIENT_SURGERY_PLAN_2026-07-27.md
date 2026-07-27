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
background_gradient_surgery_start_step=10001
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

## Execution Update: 2026-07-27

### N1 Uninterrupted Control

Completed:

```bash
GPU=6 scripts/experiments/bg_attr_n1_uninterrupted_control_iui3.sh
```

Output:

```text
outputs/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control/water-splatting/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control_20260727_n1_uninterrupted_control
renders/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control_20260727_n1_uninterrupted_control
```

Checkpoints:

```text
step-000005000.ckpt
step-000010000.ckpt
step-000014999.ckpt
```

Metrics:

| run | PSNR | SSIM | LPIPS | J Blue | Far Accum | Far Clear | Water Accum | Water J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N1 uninterrupted | 31.1573 | 0.9142 | 0.1752 | 0.1213 | 0.3798 | 0.0693 | 0.0748 | 0.000699 |

### Resume Control Result

Two resume controls were tested from `step-000010000.ckpt`.

R0 to step 15000:

```bash
GPU=7 scripts/experiments/bg_attr_r0_resume10k_control_iui3.sh
```

R0b to step 14999:

```bash
GPU=7 EXPERIMENT_NAME=bg_attr_r0b_resume10k_control_iui3_14999 \
  STAMP=20260727_r0b_resume10k_control_14999 \
  MAX_NUM_ITERATIONS=4999 STEPS_PER_SAVE=4999 \
  scripts/experiments/bg_attr_r0_resume10k_control_iui3.sh
```

Result:

| run | PSNR | SSIM | LPIPS | J Blue | Far Accum | Far Clear | Water Accum | Water J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N1 uninterrupted | 31.1573 | 0.9142 | 0.1752 | 0.1213 | 0.3798 | 0.0693 | 0.0748 | 0.000699 |
| R0 step 15000 | 31.0520 | 0.9143 | 0.1752 | 0.1333 | 0.3912 | 0.0698 | 0.0941 | 0.000644 |
| R0b step 14999 | 31.0410 | 0.9141 | 0.1754 | 0.1332 | 0.3823 | 0.0701 | 0.0852 | 0.000647 |

Conclusion:

- Resume control fails the planned gate: PSNR differs by roughly `0.11 dB`,
  and J Blue differs by roughly `9.8%`.
- The step-15000 cull explained part of the Far Accum difference, but not the
  PSNR/J Blue mismatch.
- Inspection of `FullImageDatamanager` shows train camera order is driven by a
  Python `random.Random(train_cameras_sampling_seed)` object. Its state is not
  serialized in the Nerfstudio checkpoint, so resumed training repeats a
  different 10k-15k image order than the uninterrupted run.

Decision:

- Do not promote resume-10k C/G experiments as formal candidates.
- Use uninterrupted full 15k runs for C5/G1/G2 so the 0-10k prefix is
  deterministic and comparable.

### Train-View Candidate Mask

Completed:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_gaussian_region_sensitivity.py \
  --load-config outputs/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control/water-splatting/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control_20260727_n1_uninterrupted_control/config.yml \
  --load-step 10000 \
  --mask-dir common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726 \
  --output-json renders/gradient_surgery_20260727/train_region_sensitivity_step10000.json \
  --candidate-output-prefix renders/gradient_surgery_20260727/candidate_mask_step10000 \
  --split train --max-images 25 --enable-clear-proxy \
  --candidate-min-view-count 5 \
  --candidate-water-quantile 0.995 \
  --candidate-proxy-quantile 0.95 \
  --candidate-object-ratio-max 0.10 \
  --candidate-boundary-ratio-max 0.10 \
  --candidate-require-proxy \
  --top-k 100
```

Candidate result:

```text
candidate_count = 28
candidate_fraction = 0.00331%
active_gaussian_count = 840,531
water_threshold = 0.001715
proxy_threshold = 0.00000386
selected_water_score_sum = 0.114294
selected_proxy_score_sum = 0.027808
```

Candidate files:

```text
renders/gradient_surgery_20260727/candidate_mask_step10000.pt
renders/gradient_surgery_20260727/candidate_mask_step10000.json
```

### Revised Formal Runs

Run matched full 15k experiments:

```bash
GPU=7 scripts/experiments/bg_attr_c5_proxy_chroma002_full_iui3.sh
GPU=8 scripts/experiments/bg_attr_g1_opacity_surgery_x2_full_iui3.sh
GPU=9 scripts/experiments/bg_attr_g2_opacity_surgery_x4_full_iui3.sh
```

These runs avoid resume RNG mismatch. The candidate mask is fixed from the N1
step-10000 train-view attribution diagnostic, and intervention starts at step
10001.

## Execution Update: 2026-07-27 Later Pass

### Candidate-Mask Lifecycle Fix

Initial full-from-scratch G1/G2 attempts failed at step 10000 because the fixed
candidate mask was generated from the N1 `step-000010000.ckpt`, while the
full-run Gaussian set diverged slightly before the intervention step:

```text
G1 full current Gaussian count: 847606
G2 full current Gaussian count: 845585
candidate mask length: 845552
```

Matched resume runs from the N1 checkpoint then failed after the first post-10k
cull because the candidate mask length was not updated when Gaussians were
deleted:

```text
candidate mask length 845552 did not match current Gaussian count 838831
```

Code fix:

- `water_splatting/water_splatting.py`
  - added candidate-mask synchronization for densification append;
  - added candidate-mask synchronization for culling;
  - split and duplicate children inherit the parent candidate flag;
  - culled Gaussians are removed from the in-memory candidate mask with the same
    cull mask used for Gaussian parameters.

Smoke test:

```bash
GPU=6 MAX_NUM_ITERATIONS=300 STEPS_PER_SAVE=300 \
  EXPERIMENT_NAME=bg_attr_g1_hook_cullsync_smoke_iui3 \
  STAMP=20260727_g1_hook_cullsync_smoke \
  RUN_EVAL=0 RUN_CLOSURE_DIAG=0 RUN_FAR_DIAG=0 RUN_REGION_DIAG=0 \
  scripts/experiments/bg_attr_g1_opacity_surgery_x2_iui3.sh
```

Result:

```text
passed through step 10200/10300 culls
checkpoint: outputs/bg_attr_g1_hook_cullsync_smoke_iui3/.../step-000010300.ckpt
```

### Matched Resume Experiments

All matched runs below resume from:

```text
outputs/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control/water-splatting/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control_20260727_n1_uninterrupted_control/nerfstudio_models/step-000010000.ckpt
```

Shared settings:

```text
MAX_NUM_ITERATIONS=5000
MODEL_NUM_STEPS=15000
STEPS_PER_SAVE=5000
lambda_background_clear_chroma=0.002 for C5/G runs
background_clear_chroma_start_step=10001
```

Commands:

```bash
GPU=7 MAX_NUM_ITERATIONS=5000 STEPS_PER_SAVE=5000 \
  EXPERIMENT_NAME=bg_attr_c5_proxy_chroma002_iui3_resume10k_ckpt_cullsync_15000 \
  STAMP=20260727_c5_chroma002_resume10k_ckpt_cullsync \
  LOAD_CHECKPOINT=/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control/water-splatting/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control_20260727_n1_uninterrupted_control/nerfstudio_models/step-000010000.ckpt \
  BACKGROUND_CLEAR_CHROMA_START_STEP=10001 \
  RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 \
  scripts/experiments/bg_attr_c5_proxy_chroma002_full_iui3.sh

GPU=8 MAX_NUM_ITERATIONS=5000 STEPS_PER_SAVE=5000 \
  EXPERIMENT_NAME=bg_attr_g1_c5_opacity_surgery_x2_iui3_resume10k_cullsync_15000 \
  STAMP=20260727_g1_c5_opacity_x2_resume10k_cullsync \
  RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 \
  scripts/experiments/bg_attr_g1_opacity_surgery_x2_iui3.sh

GPU=9 MAX_NUM_ITERATIONS=5000 STEPS_PER_SAVE=5000 \
  EXPERIMENT_NAME=bg_attr_g2_c5_opacity_surgery_x4_iui3_resume10k_cullsync_15000 \
  STAMP=20260727_g2_c5_opacity_x4_resume10k_cullsync \
  RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 \
  scripts/experiments/bg_attr_g2_opacity_surgery_x4_iui3.sh
```

Matched results:

| run | PSNR | SSIM | LPIPS | J Blue | Far Accum | Far Clear | Water Accum | Water J | Object Ret | bg split mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 resume | 31.0520 | 0.9143 | 0.1752 | 0.1333 | 0.3912 | 0.0698 | 0.0941 | 0.000644 | 0.9834 | 0.001818 |
| C5 matched | 31.0321 | 0.9143 | 0.1752 | 0.1016 | 0.3762 | 0.0673 | 0.0813 | 0.000645 | 0.9681 | 0.001792 |
| G1 x2, 28 cand | 31.0242 | 0.9143 | 0.1753 | 0.1028 | 0.3599 | 0.0677 | 0.0747 | 0.000639 | 0.9634 | 0.001782 |
| G2 x4, 28 cand | 31.0347 | 0.9143 | 0.1751 | 0.1030 | 0.3634 | 0.0676 | 0.0742 | 0.000645 | 0.9643 | 0.001769 |

Relative to matched C5:

| run | dPSNR | J Blue | Far Accum | Far Clear | Water Accum | Water J | Object Ret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 x2 | -0.0078 | +1.1% | -4.3% | +0.6% | -8.1% | -1.0% | -0.48% |
| G2 x4 | +0.0026 | +1.4% | -3.4% | +0.4% | -8.7% | +0.0% | -0.39% |

Interpretation:

- C5 proxy chroma remains the main source of J Blue reduction.
- Opacity gradient surgery provides a real accumulation signal: Water Accum
  drops by roughly `8%`, and Far Accum drops by `3%-4%` relative to matched C5.
- It does not yet reduce Far Clear or J Blue beyond C5.
- Object retention is already below the formal `0.97` target for matched C5 and
  drops further under opacity surgery.
- The original candidate set is too sparse: only `28` Gaussians, and logs show
  it is effectively eliminated by early post-10k culls:

```text
G1 x2 candidates at step 10500: 6
G1 x2 candidates at step 11000+: 0
G2 x4 candidates at step 10500+: 0
```

### Wider Candidate Test

Generated a less sparse train-view candidate mask:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_gaussian_region_sensitivity.py \
  --load-config outputs/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control/water-splatting/bg_attr_n1_precise_raw_binf_iui3_uninterrupted_control_20260727_n1_uninterrupted_control/config.yml \
  --load-step 10000 \
  --mask-dir common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726 \
  --output-json renders/gradient_surgery_20260727/train_region_sensitivity_step10000_q990_p90.json \
  --candidate-output-prefix renders/gradient_surgery_20260727/candidate_mask_step10000_q990_p90 \
  --split train --max-images 25 --enable-clear-proxy \
  --candidate-min-view-count 5 \
  --candidate-water-quantile 0.99 \
  --candidate-proxy-quantile 0.90 \
  --candidate-object-ratio-max 0.10 \
  --candidate-boundary-ratio-max 0.10 \
  --candidate-require-proxy \
  --top-k 100
```

Candidate summary:

| mask | candidate count | selected water score | selected proxy score | min view count |
| --- | ---: | ---: | ---: | ---: |
| q99.5 / p95 | 28 | 0.1143 | 0.0278 | 8 |
| q99.0 / p90 | 51 | 0.1373 | 0.0330 | 5 |

Ran G1w with the wider mask:

```bash
GPU=8 MAX_NUM_ITERATIONS=5000 STEPS_PER_SAVE=5000 \
  EXPERIMENT_NAME=bg_attr_g1w_c5_opacity_surgery_x2_q990_p90_iui3_resume10k_cullsync_15000 \
  STAMP=20260727_g1w_x2_q990_p90_resume10k_cullsync \
  BACKGROUND_CANDIDATE_MASK_PATH=/mnt/new/home_old/ycy/water-splatting-refactor/renders/gradient_surgery_20260727/candidate_mask_step10000_q990_p90.pt \
  RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 \
  scripts/experiments/bg_attr_g1_opacity_surgery_x2_iui3.sh
```

Result:

| run | PSNR | SSIM | LPIPS | J Blue | Far Accum | Far Clear | Water Accum | Water J | Object Ret | bg split mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1w x2, 51 cand | 31.0369 | 0.9142 | 0.1752 | 0.1001 | 0.3736 | 0.0677 | 0.0776 | 0.000625 | 0.9655 | 0.001767 |

Relative to matched C5:

| run | dPSNR | J Blue | Far Accum | Far Clear | Water Accum | Water J | Object Ret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1w x2 | +0.0048 | -1.5% | -0.7% | +0.5% | -4.5% | -3.0% | -0.26% |

Interpretation:

- Wider candidates improve J Blue and Water J slightly versus C5, with better
  PSNR than the 28-candidate G1.
- The accumulation reduction is weaker than the 28-candidate G1/G2.
- Candidates still mostly disappear by step 12000:

```text
step 10500 candidates: 18
step 11000 candidates: 1
step 12000+: 0
```

## Current Conclusion After Opacity Surgery

The opacity-gradient signal is real but not yet sufficient as a mainline
solution:

- It can reduce high-precision Water Accum by `4%-9%` and common Far Accum by
  up to `4%` relative to matched C5.
- It does not reduce Far Clear enough, and narrow surgery can slightly worsen
  J Blue.
- Candidate indices are fragile because most selected Gaussians are removed by
  normal post-10k culling shortly after surgery starts.
- Since C5 already falls below the formal object-retention target in this
  matched resume branch, stronger opacity multipliers are not justified.

Next experimental direction:

1. keep `J_proxy_raw` and proxy chroma as the main differentiable clear
   interface;
2. stop increasing opacity-decrease multipliers for this fixed candidate set;
3. test a candidate objective that targets blue/green clear chroma contributors
   while explicitly monitoring scale/SH-rest transfer, rather than only
   amplifying opacity decrease;
4. for any future index-based intervention, either select candidates online
   after the step-10200/10500 culls or persist Gaussian lineage IDs through
   split/dup/cull.
