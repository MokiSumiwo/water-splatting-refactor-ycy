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
| E4 | 0.0002 | 0.00002, start 10000 | appearance-only chroma 0.0015 |

E1 tests whether object accumulation loss in C2-B02 mainly came from full proxy geometry/opacity gradients. E2 tests light halo pressure. E3 is only useful if E2 improves directionally but leaves visible halo residual.
E4 was added after E2 failed directionally; it keeps the halo pressure but delays it until the 10k residual-refinement stage.

## Commands

```bash
GPU=6 bash scripts/experiments/medium_attr_e1_b02_app_proxy_chroma015_iui3.sh
GPU=7 bash scripts/experiments/medium_attr_e2_b02_halo002_app_proxy_iui3.sh
GPU=8 bash scripts/experiments/medium_attr_e3_b02_halo004_app_proxy_iui3.sh
GPU=8 bash scripts/experiments/medium_attr_e4_b02_halo002_late_app_proxy_iui3.sh
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

## E1 / E2 Results

| Metric | C2-B02 | E1 app-only | E2 halo 0.00002 |
| --- | ---: | ---: | ---: |
| PSNR | 31.1139 | 31.0844 | 30.8886 |
| SSIM | 0.914619 | 0.913836 | 0.911628 |
| LPIPS | 0.176461 | 0.175482 | 0.175025 |
| Far Accum | 0.237373 | 0.316544 | 0.300064 |
| Far Clear | 0.059453 | 0.068810 | 0.079266 |
| Far BG Residual Fraction | 0.119437 | 0.188311 | 0.196444 |
| Far BG Largest Component Max | 0.111967 | 0.118322 | 0.246124 |
| Water Accum | 0.006800 | 0.023967 | 0.011167 |
| Water J | 0.000762 | 0.000300 | 0.001668 |
| Object Acc Ret | 0.931326 | 0.973084 | 0.941777 |
| Object J Ret | 0.975337 | 1.004191 | 0.994637 |
| Boundary Ret | 0.956130 | 0.961212 | 0.992662 |

Immediate interpretation:

- E1 confirms full proxy gradients were contributing to occupancy cleanup; making proxy appearance-only recovers object accumulation retention but loses far accumulation and far-clear cleanup.
- E2 fails directionally. It does not reduce the continuous far blue-green residual; it increases far residual fraction and largest connected component while dropping PSNR and object accumulation retention.
- E3 should not be run under the original schedule because stronger early halo pressure is likely to amplify the same failure.

Support diagnostics:

```text
E1 final support:
S_halo mean              0.01848
water S_halo mean        0.00407
object S_halo mean       0.02294
boundary S_halo mean     0.02144

E2 final support:
S_halo mean              0.01790
water S_halo mean        0.03364
object S_halo mean       0.02198
boundary S_halo mean     0.01771
```

C2-B02 post-hoc threshold sweep shows the default halo support does respond to the newly defined far-BG residual mask:

```text
default gate on C2:
far-BG S_halo mean       0.10685
object S_halo mean       0.01925
boundary S_halo mean     0.01197
object / far-BG          0.18
boundary / far-BG        0.11
```

This suggests the residual-gated halo signal is meaningful late in training, but using it from step 4000 changes the training trajectory too early. E4 therefore delays halo capacity until step 10000 and keeps the same low halo weight.

## E4 Result

E4 uses the same halo weight as E2 (`lambda_halo_capacity=0.00002`) but delays halo capacity to step 10000.

| Metric | C2-B02 | E4 late halo 0.00002 |
| --- | ---: | ---: |
| PSNR | 31.1139 | 31.2759 |
| SSIM | 0.914619 | 0.914072 |
| LPIPS | 0.176461 | 0.175213 |
| Far Accum | 0.237373 | 0.216155 |
| Far Clear | 0.059453 | 0.064486 |
| Far BG Residual Fraction | 0.119437 | 0.101745 |
| Far BG LCC Sum | 0.072294 | 0.064241 |
| Far BG LCC Max | 0.111967 | 0.108419 |
| Water Accum | 0.006800 | 0.004964 |
| Water J | 0.000762 | 0.000337 |
| Object Acc Ret | 0.931326 | 0.910958 |
| Object J Ret | 0.975337 | 0.970405 |
| Boundary Ret | 0.956130 | 0.951009 |

Interpretation:

- Delaying halo pressure fixes the worst E2 behavior and improves Far Accum, Water Accum, Water J, and Far BG residual fraction versus C2-B02.
- The improvement is not enough for the residual-area target, and object retention is still too low.
- The next test should not increase halo weight. It should reduce late-halo weight to find whether the far-BG improvement can be retained while recovering object retention.

## E5 / E6 Plan

| ID | Core capacity | Halo capacity | Halo start | Proxy |
| --- | ---: | ---: | ---: | --- |
| E5 | 0.0002 | 0.000005 | 10000 | appearance-only chroma 0.0015 |
| E6 | 0.0002 | 0.000010 | 10000 | appearance-only chroma 0.0015 |

Commands:

```bash
GPU=6 bash scripts/experiments/medium_attr_e5_b02_halo0005_late_app_proxy_iui3.sh
GPU=7 bash scripts/experiments/medium_attr_e6_b02_halo001_late_app_proxy_iui3.sh
```

Decision rule:

- If E5/E6 recover object retention but lose all Far BG benefit, this branch is likely not sufficient.
- If E6 keeps Far BG residual below E4 while improving object retention, use E6 as the next visual candidate.
- If neither beats C2-B02 on both Far BG connected residual and retention, revert to full-proxy C2-B02 as the main branch and investigate a geometry-aware but object-safe proxy path instead of halo capacity.
