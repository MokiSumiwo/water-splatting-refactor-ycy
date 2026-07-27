# Next Experiment Plan: Gradient-Path Attribution

Date: 2026-07-27

Repository: `/mnt/new/home_old/ycy/water-splatting-refactor`

Branch: `refactor/core-framework`

Current anchor commit: `17a3b9e Add background attribution diagnostics`

## High-Level Diagnosis

The current primary issue is no longer the definition of `B_inf`, nor primarily
background-mask precision. The unresolved problem is that training objective,
differentiable path, and per-Gaussian attribution are not yet closed.

The current `bg-J` and background-tail experiments are therefore not fully
causal until we prove which renderer outputs actually send useful gradients to
Gaussian parameters.

Key code facts to verify and build on:

- `rasterize_gaussians()` currently ignores the passed `background` and forces a
  white background.
- `_RasterizeGaussians.backward()` receives `v_out_clr`, `v_final_Ts`, and
  depth-related gradients, but currently only forwards `v_out_img`,
  `v_out_medium`, and `v_out_alpha` into CUDA backward.
- `background_clear_gaussian_loss` supervises `J_gaussian_raw`, so it may not
  provide a valid Gaussian-parameter gradient if `v_out_clr` is unused.
- `rgb_tail` depends on `final_transmittance` and `last_depth`, but if those
  gradients are not consumed by CUDA backward, tail losses primarily affect
  `B_inf` / medium parameters rather than Gaussian opacity or scale.

## Current Empirical State

Baseline:

| run | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| M1 dir_xy_camera | 31.1314 | 0.9120 | 0.1750 |

Important references:

| run | role | PSNR | SSIM | LPIPS | FarAccum | FarClear | J Blue |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2/B2 tied B_inf bg=0.005 | leakage reference | 30.9641 | 0.9122 | 0.1773 | 0.3690 | 0.0683 | 0.1146 |
| A3 bounded residual s=0.02 bg=0.005 | reconstruction reference | 31.2954 | 0.9144 | 0.1753 | 0.4579 | 0.0817 | 0.1214 |
| N1 precise raw B_inf=0.005 | high-precision mask baseline | 31.1045 | 0.9129 | 0.1767 | 0.5106 | 0.0681 | 0.0901 |
| N1_J004 | reconstruction-safe bg-J ablation | 31.1529 | 0.9137 | 0.1751 | 0.4476 | 0.0805 | 0.1593 |
| N1_J005 | leakage-oriented bg-J ablation | 31.0783 | 0.9126 | 0.1763 | 0.6457 | 0.0624 | 0.0886 |

Diagnostic conclusion from the latest opacity/accumulation run:

- High-precision water sampled Gaussian accumulation approaches saturation late
  in training.
- At step 14999, sampled water accumulation mean is `0.994672` and p95 is
  `0.999897`.
- The 10k-15k window has low water tail luma (`0.000536`) while water `J` luma
  remains non-zero (`0.091462`).
- Background split/duplicate pressure is low under the high-precision mask
  (`bg split fraction mean=0.001468`, duplicate fraction zero).
- The native opacity-gradient signal is weak and inconsistent.

Interpretation:

> The current model can suppress tail while retaining or increasing Gaussian
> occupancy in far water. We need to determine whether this is caused by missing
> gradient paths, weak but correct opacity gradients, or distributed
> representation leakage.

## Do Not Restore As Mainline

Do not bring back these old M2 mechanisms as the primary route:

- alpha-depth ownership
- `m_inf` / `m_inf_eff` RGB mixing
- accumulation-zero capacity loss
- near-zero loss
- dynamic hit protection
- capacity floor
- hard pruning
- opacity decay

Do not continue fine-grained `lambda_background_clear_gaussian` sweeps until the
gradient path audit is complete.

## Phase P0: Output Gradient Path Audit

### Objective

Determine which model outputs actually send gradients to Gaussian parameters and
which only affect medium-side parameters or diagnostics.

### New Script

`scripts/diagnostics/diagnose_output_gradient_paths.py`

### Outputs To Test

For each selected eval camera, separately backpropagate scalar probes:

- `mean(rgb)`
- `mean(rgb_object)`
- `mean(rgb_medium)`
- `mean(rgb_medium_total)`
- `mean(J_gaussian_raw)`
- `mean(accumulation)`
- `mean(final_transmittance)`
- `mean(rgb_tail)`

### Parameters To Record

Record gradient norm, max absolute gradient, and nonzero ratio for:

- `means`
- `scales`
- `quats`
- `features_dc`
- `features_rest`
- `opacities`
- medium MLP parameters
- direction encoding parameters

Also record available non-parameter screen-space gradients where useful:

- `xys.grad`
- `xys_grad_abs`

### Expected Result

Expected path table:

| output scalar | expected Gaussian gradient |
| --- | --- |
| `rgb` | non-zero |
| `rgb_object` | non-zero |
| `rgb_medium` | likely medium-side dominant |
| `J_gaussian_raw` | likely zero or incomplete |
| `accumulation` | non-zero via `v_out_alpha` |
| `final_transmittance` | likely zero/incomplete |
| `rgb_tail` | likely medium/B_inf-side only |

### Gate

If `J_gaussian_raw -> Gaussian` is zero, all previous `bg-J` runs must be
treated as non-causal or indirect retraining effects. Do not keep tuning bg-J.

## Phase P1: Clear Proxy Render Without CUDA Backward Changes

### Objective

Create a clear-render proxy that uses the already-supported `out_img` backward
path, so clear/chroma losses can actually train Gaussian parameters.

### Minimal Code Changes

In `water_splatting/rasterize.py`:

- Add a backwards-compatible option such as
  `force_white_background: bool = True`.
- Preserve current behavior by default.
- When `force_white_background=False`, use the passed background tensor.

In `water_splatting/rendering/underwater_rasterizer.py`:

- Add `rasterize_clear_proxy(...)`.
- Call `rasterize_gaussians()` with:
  - zero `medium_rgb`
  - zero `medium_bs`
  - zero `medium_attn`
  - black background
  - `force_white_background=False`

### Diagnostic

Add or extend gradient-path script to compare:

- `J_proxy_raw`
- existing `J_gaussian_raw`
- `abs(J_proxy_raw - J_gaussian_raw)`
- gradient path from `J_proxy_raw` to Gaussian parameters

### Pass Criteria

- `J_proxy_raw` must have non-zero gradients to opacity/color/geometry.
- Forward difference from current clear output should be understood and small
  enough to justify using it as a proxy. If white-background legacy makes exact
  equality impossible, document the expected difference.

## Phase P2: Contribution-Aware Diagnostic

### Objective

Replace projection-center region classification with differentiable
per-Gaussian sensitivity attribution.

### New Diagnostic

Compute per-Gaussian sensitivity from scalar masked probes:

- `mean(water_mask * accumulation)`
- `mean(object_mask * accumulation)`
- `mean(boundary_mask * accumulation)`
- later: `mean(water_mask * clear_proxy_chroma)`

For each probe, record per-Gaussian gradient magnitudes for:

- opacity
- scale
- color/DC
- SH-rest
- means

Aggregate across views with EMA or simple sums.

### Candidate Score

A candidate background contributor should satisfy:

- high water sensitivity
- low object sensitivity
- low boundary sensitivity
- multi-view support >= 3
- high sampled or contribution-aware accumulation
- non-zero clear/J/chroma contribution

### Reports

For top 1%, 5%, and 10% candidates, report:

- water sensitivity share
- object sensitivity share
- boundary sensitivity share
- opacity distribution
- scale distribution
- depth distribution
- projected radius
- color blue/green dominance
- view count

### Comparisons

Run for:

- M1
- N1
- N1_J005
- N1 opacity/accum diagnostic run if available

## Phase P3: Post-Densification Gradient Surgery

Only run after P2 confirms useful high-water / low-object Gaussian candidates.

### Mechanism

Optimizer-side, sign-preserving opacity gradient modulation:

- multiply existing positive opacity-logit gradients by 2 or 4
- leave negative opacity-logit gradients unchanged
- no constant opacity decay
- start after step 10000
- require contribution-aware candidate score

### Matrix

Use one shared N1 step-10000 checkpoint if possible, then continue to 15000:

| run | opacity decrease multiplier | opacity increase multiplier | scale surgery |
| --- | ---: | ---: | --- |
| G0 | 1.0 | 1.0 | off |
| G1 | 2.0 | 1.0 | off |
| G2 | 4.0 | 1.0 | off |

## Phase P4: Accumulation-Gated Chroma Suppression

Only run after P1 validates `J_proxy_raw` as an effective trainable proxy.

### Mechanism

Use water-mask and detached accumulation gate. Penalize blue/green medium-like
chroma in clear proxy rather than forcing all clear radiance to zero.

### Matrix

| run | chroma weight | accumulation max |
| --- | ---: | ---: |
| C1 | low | 0.65 |
| C2 | 2x low | 0.70 |

## Success Criteria

Underwater:

- PSNR >= 31.0814
- SSIM >= 0.9110
- LPIPS <= 0.1760

Residual:

- J Blue relative to M1 decreases by at least 20%
- Far Clear relative to M1 decreases by at least 25%
- Water J relative to M1 decreases by at least 25%
- high-precision water accumulation should not increase like N1_J005

Retention:

- Object J retention >= 0.97
- Boundary retention >= 0.95

## Execution Order

1. Implement and run P0 gradient path audit.
2. If P0 confirms `J_gaussian_raw` / tail gradients are incomplete, stop bg-J
   weight tuning.
3. Implement P1 clear proxy and verify forward / backward behavior.
4. Implement P2 contribution-aware diagnostic.
5. Only then run P3 gradient surgery or P4 chroma suppression.

## Execution Log: 2026-07-27

### Code Added

- Added `scripts/diagnostics/diagnose_output_gradient_paths.py`
  - Audits scalar-output gradients to Gaussian and medium parameters.
  - Avoids Nerfstudio `get_outputs_for_camera()` because it is decorated with
    `torch.no_grad()`.
- Added `scripts/diagnostics/diagnose_gaussian_region_sensitivity.py`
  - Computes per-Gaussian sensitivity from masked scalar probes instead of
    projection-center region labels.
  - Saves aggregate and top-candidate summaries only, not full per-Gaussian
    arrays.
- Added legacy-safe clear proxy path:
  - `water_splatting/rasterize.py`
    - new `force_white_background` argument, default `True`
  - `water_splatting/rendering/underwater_rasterizer.py`
    - new `rasterize_clear_proxy()`
  - `water_splatting/water_splatting.py`
    - new `clear_proxy_enabled: bool = False`
    - optional outputs: `J_proxy_raw`, `J_proxy`,
      `J_proxy_abs_diff_from_renderer_clear`, `J_proxy_rgb_object`,
      `J_proxy_accumulation`

All new behavior is default-off and should preserve previous M1/N1 behavior
unless explicitly enabled.

### P0 Gradient Path Audit

Commands:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_output_gradient_paths.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --output-json renders/gradient_path_audit_20260727/m1_gradient_paths.json \
  --max-images 2

CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_output_gradient_paths.py \
  --load-config outputs/bg_attr_n1_precise_raw_binf_iui3_15000/water-splatting/bg_attr_n1_precise_raw_binf_iui3_15000_20260726_n1/config.yml \
  --output-json renders/gradient_path_audit_20260727/n1_gradient_paths_v2.json \
  --max-images 2

CUDA_VISIBLE_DEVICES=7 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_output_gradient_paths.py \
  --load-config outputs/bg_attr_n1_j005_clear00005_iui3_15000/water-splatting/bg_attr_n1_j005_clear00005_iui3_15000_20260727_n1j005/config.yml \
  --output-json renders/gradient_path_audit_20260727/n1_j005_gradient_paths.json \
  --max-images 2
```

Key result:

| run | probe | Gaussian grad norm mean | opacity grad norm mean | medium grad norm mean |
| --- | --- | ---: | ---: | ---: |
| M1 | `rgb` | 0.029326 | 0.0001877 | 0.358332 |
| M1 | `J_gaussian_raw` | 0.000000 | 0.0000000 | 0.000000 |
| M1 | `accumulation` | 0.105884 | 0.0102489 | 0.000000 |
| M1 | `final_transmittance` | 0.000000 | 0.0000000 | 0.000000 |
| M1 | `rgb_tail` | 0.000000 | 0.0000000 | 0.010310 |
| N1 | `rgb` | 0.027809 | 0.0000566 | 0.363526 |
| N1 | `J_gaussian_raw` | 0.000000 | 0.0000000 | 0.000000 |
| N1 | `accumulation` | 0.133491 | 0.0014672 | 0.000000 |
| N1 | `final_transmittance` | 0.000000 | 0.0000000 | 0.000000 |
| N1 | `rgb_tail` | 0.000000 | 0.0000000 | 0.023598 |
| N1_J005 | `rgb` | 0.028248 | 0.0001180 | 0.366128 |
| N1_J005 | `J_gaussian_raw` | 0.000000 | 0.0000000 | 0.000000 |
| N1_J005 | `accumulation` | 0.183504 | 0.0011975 | 0.000000 |
| N1_J005 | `final_transmittance` | 0.000000 | 0.0000000 | 0.000000 |
| N1_J005 | `rgb_tail` | 0.000000 | 0.0000000 | 0.000000 |

Conclusion:

- `J_gaussian_raw` has `requires_grad=True` but does not send gradients to
  Gaussian or medium parameters.
- `final_transmittance` also does not send gradients to Gaussian parameters.
- `rgb_tail` does not send gradients to Gaussian parameters; in M1/N1 it can
  affect medium-side parameters, but not opacity/scale/means/color.
- `accumulation` is the currently valid scalar path for Gaussian opacity/scale
  pressure through `v_out_alpha`.
- Previous N1_J runs should not be treated as causal evidence that bg-J
  directly cleaned Gaussian residual.

### P1 Clear Proxy Audit

Command:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_output_gradient_paths.py \
  --load-config outputs/bg_attr_n1_precise_raw_binf_iui3_15000/water-splatting/bg_attr_n1_precise_raw_binf_iui3_15000_20260726_n1/config.yml \
  --output-json renders/gradient_path_audit_20260727/n1_clear_proxy_gradient_paths.json \
  --max-images 2 \
  --enable-clear-proxy
```

Result:

| probe | Gaussian grad norm mean | opacity grad norm mean | scale grad norm mean | color/DC grad norm mean |
| --- | ---: | ---: | ---: | ---: |
| `J_gaussian_raw` | 0.000000 | 0.0000000 | 0.000000 | 0.000000 |
| `J_proxy_raw` | 0.067887 | 0.0003652 | 0.005126 | 0.001714 |
| `J_proxy` | 0.062611 | 0.0003597 | 0.004635 | 0.001675 |

Forward alignment:

- `mean(abs(J_proxy_raw - J_gaussian_raw)) = 0.0` on both audited eval views.

Conclusion:

- The zero-medium black-background clear proxy is numerically aligned with
  current `J_gaussian_raw` for the audited views.
- Unlike `J_gaussian_raw`, it has a valid Gaussian gradient path through
  `out_img`.
- Future clear/chroma losses should use `J_proxy_raw`, not
  `J_gaussian_raw`, unless CUDA backward is extended to consume `v_out_clr`.

### P2 Contribution-Aware Sensitivity Diagnostic

Commands:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_region_sensitivity.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --mask-dir common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726_evalcheck \
  --output-json renders/gradient_path_audit_20260727/m1_region_sensitivity.json \
  --max-images 4 \
  --enable-clear-proxy \
  --top-k 50

CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_region_sensitivity.py \
  --load-config outputs/bg_attr_n1_precise_raw_binf_iui3_15000/water-splatting/bg_attr_n1_precise_raw_binf_iui3_15000_20260726_n1/config.yml \
  --mask-dir common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726_evalcheck \
  --output-json renders/gradient_path_audit_20260727/n1_region_sensitivity.json \
  --max-images 4 \
  --enable-clear-proxy \
  --top-k 50

CUDA_VISIBLE_DEVICES=7 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_region_sensitivity.py \
  --load-config outputs/bg_attr_n1_j005_clear00005_iui3_15000/water-splatting/bg_attr_n1_j005_clear00005_iui3_15000_20260727_n1j005/config.yml \
  --mask-dir common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726_evalcheck \
  --output-json renders/gradient_path_audit_20260727/n1_j005_region_sensitivity.json \
  --max-images 4 \
  --enable-clear-proxy \
  --top-k 50
```

Summary:

| run | total water opacity sensitivity | total water proxy-bluegreen opacity sensitivity | top 1% water sensitivity share | top 1% object share | top 1% boundary share |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.389558 | 0.071494 | ~1.000000 | 0.017704 | 0.145577 |
| N1 | 0.072312 | 0.027591 | ~1.000000 | 0.078751 | 0.099211 |
| N1_J005 | 0.069828 | 0.029382 | 1.000000 | 0.201110 | 0.137342 |

Interpretation:

- Water accumulation sensitivity is highly concentrated: top 1% Gaussians cover
  effectively all measured water opacity sensitivity under this 4-view eval
  diagnostic.
- N1 has top water-sensitive candidates with low object and boundary
  sensitivity, so contribution-aware targeting is plausible.
- N1_J005 shows materially higher object sensitivity overlap in the top 1%
  (`0.2011` vs N1 `0.0788`), consistent with its boundary/object retention
  problems.
- Top candidates have moderate opacity, moderate scale, and positive
  blue/green-minus-red DC color, making them plausible far-water residual
  contributors rather than random low-impact points.

Example N1 top candidates:

| index | water score | object score | boundary score | views | opacity | max scale | bluegreen-red |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 664379 | 0.002452 | 0.0 | 0.0 | 3 | 0.258800 | 0.235666 | 0.129308 |
| 670287 | 0.002196 | 0.0 | 0.0 | 4 | 0.124976 | 0.324114 | 0.223474 |
| 649583 | 0.001646 | 0.0 | 0.0 | 3 | 0.578962 | 0.185424 | 0.202665 |
| 367688 | 0.001107 | 0.0 | 0.0 | 3 | 0.265534 | 0.289056 | 0.243984 |
| 495922 | 0.001029 | 0.0 | 0.0 | 3 | 0.313497 | 0.339687 | 0.229258 |

### Decision After P0-P2

- Stop using current `background_clear_gaussian_loss` on `J_gaussian_raw` as an
  active mechanism; its Gaussian gradient path is dead.
- Keep `J_proxy_raw` as the next clear/chroma training target because it is
  numerically aligned and differentiable to Gaussian parameters.
- Contribution-aware targeting is justified: there are concentrated
  high-water / low-object candidates.
- Do not run G1/G2 yet from the existing final checkpoints:
  - current saved N1 checkpoint set only contains `step-000014999.ckpt`;
  - the planned G1/G2 comparison requires a shared N1 step-10000 checkpoint and
    preserved optimizer state;
  - continuing from 15k to 20k would answer a different question and would not
    test the intended post-densification 10k-15k window.

### Next Immediate Training Step

Run a controlled N1 10k checkpoint generation with non-latest checkpoint saving,
then branch:

1. `G0`: resume N1 10k to 15k without intervention.
2. `C1`: resume N1 10k to 15k with low-weight accumulation-gated
   `J_proxy_raw` water-chroma suppression.
3. Only after C1 is evaluated, implement G1/G2 gradient surgery or combine
   weak C1 with candidate-gated opacity-gradient modulation.

## Execution Update: 2026-07-27 Resume-10k Proxy Chroma Sweep

### Code Changes Made

- Added a differentiable clear proxy render path:
  - `water_splatting/rasterize.py`
    - added `force_white_background`, defaulting to legacy white-background
      behavior;
  - `water_splatting/rendering/underwater_rasterizer.py`
    - added `rasterize_clear_proxy()` using zero medium and black background;
  - `water_splatting/water_splatting.py`
    - added `clear_proxy_enabled`;
    - added `J_proxy_raw`, `J_proxy`, `J_proxy_rgb_object`,
      `J_proxy_accumulation`, and
      `J_proxy_abs_diff_from_renderer_clear`.
- Added accumulation-gated proxy chroma suppression:
  - `lambda_background_clear_chroma`
  - `background_clear_chroma_start_step`
  - `background_clear_chroma_ramp_steps`
  - `background_clear_chroma_accumulation_max`
  - `background_clear_chroma_accumulation_temperature`
  - `background_clear_chroma_margin`
  - `background_clear_chroma_medium_detach`
- Updated `scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh`:
  - passes the proxy-chroma flags;
  - supports `LOAD_DIR` / `LOAD_CHECKPOINT`;
  - supports `STEPS_PER_SAVE` / `SAVE_ONLY_LATEST_CHECKPOINT`;
  - added `MODEL_NUM_STEPS` so resumed runs can use trainer iterations as
    "additional steps" while preserving the model's 15k final-step logic.
- Added diagnostics:
  - `scripts/diagnostics/diagnose_output_gradient_paths.py`
  - `scripts/diagnostics/diagnose_gaussian_region_sensitivity.py`
- Added reproducible experiment wrappers:
  - `scripts/experiments/bg_attr_g0_resume10k_control_iui3.sh`
  - `scripts/experiments/bg_attr_c1_proxy_chroma00001_resume10k_iui3.sh`
  - `scripts/experiments/bg_attr_c2_proxy_chroma0001_resume10k_iui3.sh`
  - `scripts/experiments/bg_attr_c3_proxy_chroma0005_resume10k_iui3.sh`
  - `scripts/experiments/bg_attr_c4_proxy_chroma001_resume10k_iui3.sh`
  - `scripts/experiments/bg_attr_c5_proxy_chroma002_resume10k_iui3.sh`
  - `scripts/experiments/bg_attr_c6_proxy_chroma005_resume10k_iui3.sh`
  - `scripts/experiments/bg_attr_c7_proxy_chroma01_resume10k_iui3.sh`

### Validation

Static checks:

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/rasterize.py \
  water_splatting/rendering/underwater_rasterizer.py \
  water_splatting/water_splatting.py \
  scripts/diagnostics/diagnose_output_gradient_paths.py \
  scripts/diagnostics/diagnose_gaussian_region_sensitivity.py

bash -n scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh \
  scripts/experiments/bg_attr_g0_resume10k_control_iui3.sh \
  scripts/experiments/bg_attr_c1_proxy_chroma00001_resume10k_iui3.sh \
  scripts/experiments/bg_attr_c2_proxy_chroma0001_resume10k_iui3.sh \
  scripts/experiments/bg_attr_c3_proxy_chroma0005_resume10k_iui3.sh \
  scripts/experiments/bg_attr_c4_proxy_chroma001_resume10k_iui3.sh \
  scripts/experiments/bg_attr_c5_proxy_chroma002_resume10k_iui3.sh \
  scripts/experiments/bg_attr_c6_proxy_chroma005_resume10k_iui3.sh \
  scripts/experiments/bg_attr_c7_proxy_chroma01_resume10k_iui3.sh
```

Smoke tests:

- 500-step C1 smoke passed.
- TensorBoard contained `Train Loss Dict/background_clear_chroma_loss`, confirming
  the loss was active.
- Resume semantics were verified:
  - Nerfstudio trainer interprets `--max-num-iterations` as additional
    iterations after resume;
  - correct 10k to 15k resume command uses
    `MAX_NUM_ITERATIONS=5001 MODEL_NUM_STEPS=15000`.
- Smoke output directories were removed after validation to avoid retaining
  unnecessary checkpoints.

### 10k Branch Checkpoint

Generated a shared N1 branch checkpoint:

```bash
GPU=6 MAX_NUM_ITERATIONS=10000 \
  EXPERIMENT_NAME=bg_attr_n1_precise_raw_binf_iui3_10000_branch \
  STAMP=20260727_n1_10k_branch \
  RUN_EVAL=0 RUN_CLOSURE_DIAG=0 RUN_FAR_DIAG=0 RUN_REGION_DIAG=0 \
  SAVE_ONLY_LATEST_CHECKPOINT=True STEPS_PER_SAVE=1000 \
  scripts/experiments/bg_attr_n1_precise_raw_binf_iui3.sh
```

Checkpoint:

```text
outputs/bg_attr_n1_precise_raw_binf_iui3_10000_branch/water-splatting/bg_attr_n1_precise_raw_binf_iui3_10000_branch_20260727_n1_10k_branch/nerfstudio_models/step-000009999.ckpt
```

### Formal Resume-10k Sweep

All runs resumed from the same step-9999 N1 branch checkpoint and trained to
step 15000 with:

```text
medium_context_mode=dir_xy_camera
b_inf_mode=tied
B_inf=A
infinite_water_enabled=False
lambda_background_water_color=0.005
lambda_background_medium_render=0.0
lambda_background_tail_render=0.0
lambda_background_clear_gaussian=0.0
background mask=common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726
seed=42
MODEL_NUM_STEPS=15000
MAX_NUM_ITERATIONS=5001
```

Formal commands are reproducible through the wrapper scripts listed above. The
actual completed experiment names and timestamps:

| run | weight | output config |
| --- | ---: | --- |
| G0 control | 0 | `outputs/bg_attr_g0_resume10k_control_iui3_15000/water-splatting/bg_attr_g0_resume10k_control_iui3_15000_20260727_g0_control_resume10k/config.yml` |
| C1 | 0.00001 | `outputs/bg_attr_c1_proxy_chroma_iui3_resume10k_15000/water-splatting/bg_attr_c1_proxy_chroma_iui3_resume10k_15000_20260727_c1_chroma_resume10k/config.yml` |
| C2 | 0.0001 | `outputs/bg_attr_c2_proxy_chroma0001_iui3_resume10k_15000/water-splatting/bg_attr_c2_proxy_chroma0001_iui3_resume10k_15000_20260727_c2_chroma0001_resume10k/config.yml` |
| C3 | 0.0005 | `outputs/bg_attr_c3_proxy_chroma0005_iui3_resume10k_15000/water-splatting/bg_attr_c3_proxy_chroma0005_iui3_resume10k_15000_20260727_c3_chroma0005_resume10k/config.yml` |
| C4 | 0.001 | `outputs/bg_attr_c4_proxy_chroma001_iui3_resume10k_15000/water-splatting/bg_attr_c4_proxy_chroma001_iui3_resume10k_15000_20260727_c4_chroma001_resume10k/config.yml` |
| C5 | 0.002 | `outputs/bg_attr_c5_proxy_chroma002_iui3_resume10k_15000/water-splatting/bg_attr_c5_proxy_chroma002_iui3_resume10k_15000_20260727_c5_chroma002_resume10k/config.yml` |
| C6 | 0.005 | `outputs/bg_attr_c6_proxy_chroma005_iui3_resume10k_15000/water-splatting/bg_attr_c6_proxy_chroma005_iui3_resume10k_15000_20260727_c6_chroma005_resume10k/config.yml` |
| C7 | 0.010 | `outputs/bg_attr_c7_proxy_chroma01_iui3_resume10k_15000/water-splatting/bg_attr_c7_proxy_chroma01_iui3_resume10k_15000_20260727_c7_chroma01_resume10k/config.yml` |

### Results

| run | PSNR | SSIM | LPIPS | J blue | Far Accum | Far Clear | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret | BG split frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 30.9901 | 0.9137 | 0.1750 | 0.1845 | 0.5733 | 0.0889 | 0.2776 | 0.00239 | 1.0036 | 1.0049 | 1.0107 | 0.00155 |
| C1 1e-5 | 30.9957 | 0.9137 | 0.1751 | 0.1812 | 0.5738 | 0.0901 | 0.2788 | 0.00251 | 1.0035 | 1.0064 | 1.0107 | 0.00156 |
| C2 1e-4 | 31.0089 | 0.9137 | 0.1750 | 0.1772 | 0.5673 | 0.0902 | 0.2815 | 0.00259 | 1.0023 | 1.0061 | 1.0102 | 0.00155 |
| C3 5e-4 | 30.9982 | 0.9136 | 0.1751 | 0.1564 | 0.5483 | 0.0881 | 0.2724 | 0.00218 | 0.9990 | 1.0049 | 1.0100 | 0.00156 |
| C4 1e-3 | 30.9987 | 0.9137 | 0.1751 | 0.1398 | 0.5412 | 0.0862 | 0.2767 | 0.00183 | 0.9970 | 1.0021 | 1.0079 | 0.00158 |
| C5 2e-3 | 31.0013 | 0.9137 | 0.1752 | 0.1059 | 0.5260 | 0.0845 | 0.2702 | 0.00162 | 0.9909 | 1.0024 | 1.0081 | 0.00161 |
| C6 5e-3 | 30.9926 | 0.9136 | 0.1751 | 0.0831 | 0.5165 | 0.0831 | 0.2657 | 0.00153 | 0.9796 | 0.9991 | 1.0069 | 0.00157 |
| C7 1e-2 | 30.9970 | 0.9136 | 0.1751 | 0.0738 | 0.5004 | 0.0829 | 0.2557 | 0.00158 | 0.9700 | 0.9963 | 1.0048 | 0.00153 |

Relative to G0:

| run | Far Accum delta | J blue delta | Water J delta | retention note |
| --- | ---: | ---: | ---: | --- |
| C3 | -4.35% | -15.20% | -8.53% | safe |
| C4 | -5.60% | -24.19% | -23.17% | safe |
| C5 | -8.25% | -42.58% | -32.11% | safe but object accumulation retention down to 0.9909 |
| C6 | -9.90% | -54.92% | -35.87% | near retention edge: object accumulation retention 0.9796 |
| C7 | -12.72% | -59.98% | -33.78% | fails/near-fails retention: object accumulation retention 0.9700, below strict `>=0.97` after rounding margin |

### Interpretation

- `J_proxy_raw` is the first verified clear-target path that both matches the
  current clear output numerically and reaches Gaussian parameters through
  `out_img` backward.
- The proxy chroma loss produces a monotonic reduction in global blue dominance
  and a meaningful reduction in far accumulation as the weight increases.
- The loss is not only suppressing tail: `far_accumulation`, `water_accumulation`,
  and `far_hit_confidence` decline together at higher weights.
- The mechanism still does not materially reduce background split candidate
  fraction; BG split stays around `0.0015`, so it is not solving densification
  pressure directly.
- C7 gives the strongest residual cleanup but pushes object accumulation
  retention to the threshold edge; treat it as a negative / upper-bound run, not
  the best candidate.

### Current Candidates

- Best reconstruction-safe candidate: **C5 (`lambda_background_clear_chroma=0.002`)**
  - PSNR `31.0013`, SSIM `0.9137`, LPIPS `0.1752`.
  - Far Accum `0.5260`, Far Clear `0.0845`, J blue `0.1059`.
  - Object accumulation retention `0.9909`, object J retention `1.0024`.
- Best leakage-pressure candidate: **C6 (`lambda_background_clear_chroma=0.005`)**
  - Far Accum `0.5165`, Far Clear `0.0831`, J blue `0.0831`.
  - Object accumulation retention `0.9796`, still above the `0.97` retention
    criterion but with little margin.
- Rejected upper bound: **C7 (`lambda_background_clear_chroma=0.01`)**
  - Far Accum `0.5004`, J blue `0.0738`.
  - Object accumulation retention `0.9700`, effectively at the stop line; do
    not push higher without an object-protection term.

### Contribution Sensitivity Follow-Up

Additional contribution-aware diagnostics were run:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_region_sensitivity.py \
  --load-config outputs/bg_attr_g0_resume10k_control_iui3_15000/water-splatting/bg_attr_g0_resume10k_control_iui3_15000_20260727_g0_control_resume10k/config.yml \
  --mask-dir common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726_evalcheck \
  --output-json renders/gradient_path_audit_20260727/g0_region_sensitivity.json \
  --max-images 4 --enable-clear-proxy --top-k 50

CUDA_VISIBLE_DEVICES=7 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_gaussian_region_sensitivity.py \
  --load-config outputs/bg_attr_c5_proxy_chroma002_iui3_resume10k_15000/water-splatting/bg_attr_c5_proxy_chroma002_iui3_resume10k_15000_20260727_c5_chroma002_resume10k/config.yml \
  --mask-dir common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726_evalcheck \
  --output-json renders/gradient_path_audit_20260727/c5_region_sensitivity.json \
  --max-images 4 --enable-clear-proxy --top-k 50
```

Top-1% water-sensitivity comparison:

| run | top 1% water share | top 1% object share | top 1% boundary share | mean water score |
| --- | ---: | ---: | ---: | ---: |
| G0 | ~1.0000 | 0.0372 | 0.1767 | 2.766e-5 |
| C5 | ~1.0000 | 0.0296 | 0.1691 | 2.941e-5 |

Interpretation:

- C5 reduces rendered residuals and far accumulation, but top water-sensitive
  candidates remain highly concentrated.
- C5 lowers object overlap in the top 1% candidates, but does not eliminate the
  need for a contribution-aware opacity/scale intervention.

### Decision

- Keep proxy chroma suppression as a valid, default-off mechanism.
- Use C5 as the current reconstruction-safe checkpoint and C6 as the leakage
  pressure reference.
- Do not use the old dead-gradient `background_clear_gaussian_loss` for future
  active training.
- Next mechanism should be object-protected and contribution-aware, because pure
  chroma suppression alone does not reach the 25% far-accumulation target.
- Recommended next experiment:
  - branch from the same 10k checkpoint;
  - use C5 or C6 proxy chroma;
  - add a conservative object-protected accumulation/proxy-luma pressure or
    optimizer-side opacity/scale gradient modulation only for high water /
    low object sensitivity candidates;
  - keep hard pruning, opacity decay, capacity floor, and old M2 ownership losses
    disabled.
