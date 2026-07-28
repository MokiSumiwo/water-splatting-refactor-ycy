# Core-Halo Capacity Support Experiments

Date: 2026-07-28
Branch: `refactor/core-framework`
Baseline commit before this phase: `4dabd4b Add medium-explainable attribution experiments`

## Current Interpretation

`C2-B02` is not a final model. It should be treated as:

> High-confidence open-water core Gaussian cleanup is strong, but far transition / seafloor halo regions still contain continuous medium-like Gaussian residual.

The remaining visible residuals are concentrated in:

- the transition band between seafloor and black open water;
- the middle/right blue-green wedge;
- the thin cyan layer above the distant lower-right slope.

This is spatially continuous residual occupancy, not only a color-only artifact.

## Prior Reference

`C2-B02`:

- render path: `renders/medium_attr_c2_b02_capacity_proxy_chroma015_iui3_15000_20260728_c2_b02_capacity_proxy_chroma015_full`
- config path: `outputs/medium_attr_c2_b02_capacity_proxy_chroma015_iui3_15000/water-splatting/medium_attr_c2_b02_capacity_proxy_chroma015_iui3_15000_20260728_c2_b02_capacity_proxy_chroma015_full/config.yml`

Metrics:

```text
PSNR      31.1139
SSIM       0.914619
LPIPS      0.176461
J Blue     0.0525108
FarAccum   0.237373
FarClear   0.0594533
WaterAccum 0.00679978
WaterJ     0.000762279
ObjJRet    0.975337
BoundaryRet 0.956130
ObjAccumRet 0.931326
```

## Mechanism

The current core capacity support is:

```text
S_core = S_flat^2 * S_med^2 * S_far
```

This is precise on open-water cores, but it weakens near seafloor / water transitions because gradients and texture suppress `S_flat`, and squaring amplifies that suppression.

This phase adds:

```text
S_broad = S_flat * S_med * S_far
S_halo_base = relu(S_broad - S_core)
```

Then gates the halo with detached clear-proxy residual:

```text
P_bg = dot(C(J_proxy_raw), C(B_inf) / (norm(C(B_inf)) + eps))
G_C = sigmoid((P_bg - 0.015) / 0.01)
G_Y = sigmoid((Y(J_proxy_raw) - 0.02) / 0.01)
S_halo = stopgrad(S_halo_base * G_C * G_Y)
```

Capacity loss becomes:

```text
L_capacity =
    lambda_core * budget(A, S_core, A_core=0.05)
  + lambda_halo * budget(A, S_halo, A_halo=0.03)
```

The proxy chroma branch is also made appearance-only:

```text
xys, depths, radii, conics, opacities = detached
colors = live
```

This keeps proxy chroma focused on `features_dc` / `features_rest`, while core/halo capacity handles occupancy.

## New Config Flags

```text
clear_proxy_appearance_only
halo_capacity_enabled
lambda_halo_capacity
halo_capacity_value
halo_capacity_temperature
halo_capacity_start_step
halo_capacity_ramp_steps
halo_capacity_post_scale
halo_chroma_margin
halo_chroma_temperature
halo_luma_min
halo_luma_temperature
```

All default to disabled / zero-preserving behavior.

## Diagnostics

`scripts/diagnostics/diagnose_far_water_residual.py` now records:

```text
far_bg_residual_fraction
far_bg_residual_luma
far_bg_residual_accumulation
far_bg_residual_projection
far_bg_largest_component_fraction_sum
far_bg_largest_component_fraction_max
```

The residual mask is:

```text
far_mask & (P_bg > 0.015) & (Y(J_proxy_raw) > 0.02)
```

Initial C2-B02 diagnostic check:

```text
far_bg_residual_pixels              60,589 / 507,289
far_bg_residual_fraction            0.1194368
far_bg_residual_luma_mean           0.1656362
far_bg_residual_accumulation_mean   0.9038834
far_bg_residual_projection_mean     0.0490406
far_bg_largest_component_fraction_sum 0.0722941
far_bg_largest_component_fraction_max 0.1119671
```

Diagnostic output:

```text
renders/medium_attr_c2_b02_capacity_proxy_chroma015_iui3_15000_20260728_c2_b02_capacity_proxy_chroma015_full/diagnostics/far_water_bg_residual_check/far_water_residual_diagnostic.json
```

## Experiment Matrix

E0 is the prior `C2-B02`.

| ID | Core capacity | Halo capacity | Proxy |
| --- | ---: | ---: | --- |
| E1 | 0.0002 | off | appearance-only chroma 0.0015 |
| E2 | 0.0002 | 0.00002 | appearance-only chroma 0.0015 |
| E3 | 0.0002 | 0.00004 | appearance-only chroma 0.0015 |

E1 tests whether object accumulation loss in C2-B02 mainly came from full proxy geometry/opacity gradients. E2 tests light halo pressure. E3 is only useful if E2 improves directionally but leaves visible halo residual.

## Commands

```bash
GPU=6 bash scripts/experiments/medium_attr_e1_b02_app_proxy_chroma015_iui3.sh
GPU=7 bash scripts/experiments/medium_attr_e2_b02_halo002_app_proxy_iui3.sh
GPU=8 bash scripts/experiments/medium_attr_e3_b02_halo004_app_proxy_iui3.sh
```

Smoke-test form:

```bash
MAX_NUM_ITERATIONS=200 MODEL_NUM_STEPS=200 RUN_EVAL=0 RUN_CLOSURE_DIAG=0 RUN_FAR_DIAG=0 RUN_REGION_DIAG=0 STAMP=smoke_core_halo GPU=6 bash scripts/experiments/medium_attr_e2_b02_halo002_app_proxy_iui3.sh
```

## Success Criteria

Relative to C2-B02:

```text
far_bg_residual_fraction down >= 40%
far_bg_largest_component down >= 50%
```

Absolute targets:

```text
Far Accum <= 0.18
Far Clear <= 0.055
J Blue <= 0.050
Object Acc Ret >= 0.95
Object J Ret >= 0.975
Boundary Ret >= 0.95
PSNR >= 31.08
```

## Guardrails

Do not re-enable:

- inference RGB mix / alpha-depth ownership;
- q-hit or dynamic hit protection;
- near-zero loss;
- hard pruning;
- opacity decay;
- capacity floor;
- fixed Gaussian candidate surgery as the primary mechanism.

Do not increase global/core capacity above `lambda_budgeted_capacity=0.0002` in this phase.
