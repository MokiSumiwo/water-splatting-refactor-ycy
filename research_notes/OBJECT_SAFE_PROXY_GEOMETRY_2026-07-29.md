# Object-Safe Proxy Geometry Experiments

Date: 2026-07-29

## Motivation

C2-B02 remains the best current mechanism for core open-water cleanup, but it misses the far transition / seafloor halo. The halo-capacity branch did not solve this: appearance-only proxy recovered object retention but weakened cleanup, while full proxy plus late halo worsened far residuals.

The next hypothesis is that the useful C2 signal comes from clear-proxy geometry/opacity gradients, but full-strength proxy backward is too aggressive for object and boundary regions. Instead of turning those gradients fully on or fully off, this experiment scales them while keeping forward proxy values identical.

## Code Change

Added clear-proxy gradient scaling flags:

```text
clear_proxy_geometry_gradient_scale: float = 1.0
clear_proxy_opacity_gradient_scale: float = 1.0
clear_proxy_color_gradient_scale: float = 1.0
```

Implementation uses:

```text
x_proxy = detach(x) + scale * (x - detach(x))
```

Thus the clear-proxy forward render is unchanged, while the backward contribution can be attenuated per parameter group. Defaults are historical behavior. `clear_proxy_appearance_only=True` still forces geometry and opacity scale to zero.

## Fixed Base

All P-series runs use the C2-B02 base without halo capacity:

```text
medium_context_mode = dir_xy_camera
b_inf_mode = tied
lambda_medium_explainability = 0.005
lambda_budgeted_capacity = 0.0002
budgeted_capacity_value = 0.05
budgeted_capacity_post_scale = 0.5
lambda_background_clear_chroma = 0.0015
background_clear_chroma_use_medium_support = True
halo_capacity_enabled = False
lambda_proxy_clear_luma = 0
seed = 42
max_iterations = 15000
```

## Experiment Matrix

| ID | Geometry Grad | Opacity Grad | Color Grad | Purpose |
| --- | ---: | ---: | ---: | --- |
| C2 current-code | 1.00 | 1.00 | 1.00 | Full-proxy reference |
| E1 | 0.00 | 0.00 | 1.00 | Appearance-only endpoint |
| P1 | 0.25 | 0.25 | 1.00 | Conservative middle point |
| P2 | 0.50 | 0.50 | 1.00 | Stronger middle point |
| P3 | 0.00 | 0.50 | 1.00 | Test opacity-only cleanup without footprint gradients |

## Scripts

```bash
GPU=6 bash scripts/experiments/medium_attr_p1_b02_proxy_geom025_opacity025_iui3.sh
GPU=7 bash scripts/experiments/medium_attr_p2_b02_proxy_geom050_opacity050_iui3.sh
GPU=8 bash scripts/experiments/medium_attr_p3_b02_proxy_geom000_opacity050_iui3.sh
```

## Decision Criteria

Relative to C2 current-code repro:

```text
Object Acc Ret improves toward >= 0.97
Object J Ret stays >= 0.975
Boundary Ret stays >= 0.95
Far Accum does not exceed C2 repro
Far BG LCC Max decreases or stays near C2 repro
PSNR remains >= 31.08 if possible
```

Interpretation:

- If P1/P2 keep far cleanup while improving object retention, scaled proxy gradients replace full proxy.
- If P3 works best, footprint gradients are the main object-risk source and opacity-only proxy should be developed.
- If all P runs regress like E1, C2's full geometry/opacity gradient is necessary and the next step must be a stronger object/boundary support exclusion rather than gradient scaling.

## Results

| Run | Geometry Grad | Opacity Grad | PSNR | SSIM | LPIPS | Far Accum | Far Clear | Far BG Frac | Far BG LCC Max | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 current-code | 1.00 | 1.00 | 31.0738 | 0.911893 | 0.175782 | 0.291455 | 0.077338 | 0.230165 | 0.118401 | 0.006519 | 0.000761 | 0.947713 | 0.992589 | 0.963250 |
| E1 app-only | 0.00 | 0.00 | 31.0844 | 0.913836 | 0.175482 | 0.316544 | 0.068810 | 0.188311 | 0.118322 | 0.023967 | 0.000300 | 0.973084 | 1.004191 | 0.961212 |
| P1 | 0.25 | 0.25 | 31.0099 | 0.910496 | 0.175409 | 0.237284 | 0.076160 | 0.107448 | 0.108096 | 0.002519 | 0.000733 | 0.933842 | 0.998412 | 1.015901 |
| P2 | 0.50 | 0.50 | 30.9625 | 0.912780 | 0.177377 | 0.274733 | 0.068266 | 0.177518 | 0.113812 | 0.011727 | 0.000740 | 0.948894 | 0.989123 | 0.995391 |
| P3 | 0.00 | 0.50 | 31.2235 | 0.913678 | 0.174772 | 0.294079 | 0.061725 | 0.145686 | 0.097096 | 0.021233 | 0.000502 | 0.975145 | 0.970946 | 0.994212 |

## Interpretation After P1-P3

- P1 gives the best Water Accum and Far BG fraction, but PSNR and Object Acc Ret fail. Even 0.25 geometry gradient appears too risky.
- P2 is dominated by P1/P3 and should not be continued.
- P3 is the most promising new branch: it improves PSNR, LPIPS, Object Acc Ret, Far Clear, Far BG fraction, and connected residual versus C2 current-code repro, while keeping Boundary Ret strong. Its weakness is higher Water Accum than C2 and Object J Ret below the 0.975 target.
- The next sweep should keep geometry gradient disabled and vary opacity gradient only.

## P4-P6 Plan

| ID | Geometry Grad | Opacity Grad | Color Grad | Purpose |
| --- | ---: | ---: | ---: | --- |
| P4 | 0.00 | 0.25 | 1.00 | Test whether lighter opacity pressure recovers Object J Ret |
| P5 | 0.00 | 0.75 | 1.00 | Test whether stronger opacity pressure reduces Water Accum/Far BG further |
| P6 | 0.00 | 1.00 | 1.00 | Test opacity-only upper bound without footprint gradients |

Commands:

```bash
GPU=6 bash scripts/experiments/medium_attr_p4_b02_proxy_geom000_opacity025_iui3.sh
GPU=7 bash scripts/experiments/medium_attr_p5_b02_proxy_geom000_opacity075_iui3.sh
GPU=8 bash scripts/experiments/medium_attr_p6_b02_proxy_geom000_opacity100_iui3.sh
```

## P4-P6 Results

| Run | PSNR | SSIM | LPIPS | Far Accum | Far Clear | Far BG Frac | Far BG LCC Max | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 current-code | 31.0738 | 0.911893 | 0.175782 | 0.291455 | 0.077338 | 0.230165 | 0.118401 | 0.006519 | 0.000761 | 0.947713 | 0.992589 | 0.963250 |
| E1 app-only | 31.0844 | 0.913836 | 0.175482 | 0.316544 | 0.068810 | 0.188311 | 0.118322 | 0.023967 | 0.000300 | 0.973084 | 1.004191 | 0.961212 |
| P3 opacity 0.50 | 31.2235 | 0.913678 | 0.174772 | 0.294079 | 0.061725 | 0.145686 | 0.097096 | 0.021233 | 0.000502 | 0.975145 | 0.970946 | 0.994212 |
| P4 opacity 0.25 | 31.1005 | 0.913308 | 0.175376 | 0.249231 | 0.067418 | 0.135538 | 0.097813 | 0.002444 | 0.000427 | 0.932011 | 0.980311 | 1.000126 |
| P5 opacity 0.75 | 31.0631 | 0.912810 | 0.178201 | 0.239775 | 0.068273 | 0.141543 | 0.101425 | 0.004272 | 0.000388 | 0.937746 | 0.958617 | 0.962253 |
| P6 opacity 1.00 | 31.0880 | 0.911970 | 0.178141 | 0.290109 | 0.074354 | 0.119766 | 0.086640 | 0.010774 | 0.000621 | 0.939125 | 0.982407 | 0.972912 |

Contact sheet:

```text
renders/contact_sheets/proxy_opacity_sweep_rgb_j_accum_20260729.jpg
```

Interpretation:

- Geometry gradients are the most obvious risk: P1/P2 both missed either reconstruction or object retention, while all opacity-only runs improved far connected residual metrics versus C2 current-code.
- P3 is the best overall candidate: highest PSNR, best LPIPS, Object Acc Ret above 0.97, strong Boundary Ret, Far Clear improved from 0.077338 to 0.061725, and Far BG LCC Max improved from 0.118401 to 0.097096.
- P4/P5 clean core Water Accum better than P3, but both substantially damage Object Acc Ret. P6 gives the best Far BG fraction and connected component, but still misses Object Acc Ret and has weaker Far Clear than P3.
- Opacity-only strength is not monotonic. This suggests interaction with proxy chroma weight / margin and the shared medium-support mask, not simply insufficient opacity pressure.

Current best candidate:

```text
P3: geometry grad = 0.0, opacity grad = 0.5, color grad = 1.0, chroma weight = 0.0015
```

P3 still misses `Object J Ret >= 0.975` by a small margin and has higher Water Accum than C2. The next sweep should preserve P3's geometry-disabled / opacity-half structure and adjust proxy chroma strength or margin.

## P7-P9 Plan

| ID | Geometry Grad | Opacity Grad | Chroma Weight | Chroma Margin | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| P7 | 0.00 | 0.50 | 0.0010 | 0.020 | Recover object J by lowering chroma pressure |
| P8 | 0.00 | 0.50 | 0.00125 | 0.020 | Interpolate between P7 and P3 |
| P9 | 0.00 | 0.50 | 0.0015 | 0.030 | Keep weight but increase tolerated chroma margin |

Decision rule:

- Prefer the first run that keeps `Obj Acc Ret >= 0.97`, `Obj J Ret >= 0.975`, `Boundary Ret >= 0.95`, `PSNR >= 31.08`, and `Far BG LCC Max <= 0.10`.
- If all P7-P9 lose far cleanup, retain P3 as the best mechanism candidate and investigate stronger object/boundary exclusion in support construction rather than further proxy-weight tuning.

## P7-P9 Results

| Run | PSNR | SSIM | LPIPS | Far Accum | Far Clear | Far BG Frac | Far BG LCC Max | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 current-code | 31.0738 | 0.911893 | 0.175782 | 0.291455 | 0.077338 | 0.230165 | 0.118401 | 0.006519 | 0.000761 | 0.947713 | 0.992589 | 0.963250 |
| E1 app-only | 31.0844 | 0.913836 | 0.175482 | 0.316544 | 0.068810 | 0.188311 | 0.118322 | 0.023967 | 0.000300 | 0.973084 | 1.004191 | 0.961212 |
| P3 opacity 0.50 / chroma 0.0015 | 31.2235 | 0.913678 | 0.174772 | 0.294079 | 0.061725 | 0.145686 | 0.097096 | 0.021233 | 0.000502 | 0.975145 | 0.970946 | 0.994212 |
| P7 opacity 0.50 / chroma 0.0010 | 31.1065 | 0.913121 | 0.175454 | 0.282326 | 0.071197 | 0.156516 | 0.117581 | 0.014844 | 0.000467 | 0.953923 | 0.977999 | 1.018762 |
| P8 opacity 0.50 / chroma 0.00125 | 30.9175 | 0.912783 | 0.176663 | 0.296909 | 0.075979 | 0.243607 | 0.202386 | 0.012316 | 0.001743 | 0.956427 | 1.005013 | 0.971542 |
| P9 opacity 0.50 / margin 0.030 | 31.1912 | 0.913403 | 0.178260 | 0.227394 | 0.064931 | 0.096367 | 0.113812 | 0.014574 | 0.000670 | 0.920260 | 0.977208 | 0.967358 |

Support diagnostic on P3/P4/P9:

| Run | S_cap Mean | Water S_cap | Object S_cap | Boundary S_cap | Object / Water | Boundary / Water | Corr(S_cap, Accum) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P3 | 0.03100 | 0.08929 | 0.01284 | 0.02451 | 0.1430 | 0.2698 | -0.6761 |
| P4 | 0.03026 | 0.09964 | 0.01091 | 0.02080 | 0.1119 | 0.2068 | -0.7344 |
| P9 | 0.03053 | 0.09190 | 0.01320 | 0.02393 | 0.1420 | 0.2553 | -0.7398 |

Interpretation:

- P7 fixes Object J Ret but loses Object Acc Ret and connected residual improvement, so lowering chroma weight alone is not enough.
- P8 is dominated and should be dropped.
- P9 gives the strongest Far Accum / Far BG fraction but damages Object Acc Ret, indicating that good far cleanup is still entangled with object-capacity loss.
- Mean support ratios are not extreme, but boundary/object tails remain nonzero. Since `S_cap` is directly reused for both budgeted capacity and proxy chroma, the next test should remove low-confidence support tails before losses rather than increasing or decreasing global loss weights.

## P10-P12 Plan

New default-off flags:

```text
medium_support_capacity_threshold: float = 0.0
medium_support_capacity_power: float = 1.0
```

The thresholded support is:

```text
S_cap' = clamp((S_cap - threshold) / (1 - threshold), 0, 1) ** power
```

Experiment matrix:

| ID | Base | Chroma Margin | Support Threshold | Purpose |
| --- | --- | ---: | ---: | --- |
| P10 | P9 | 0.030 | 0.02 | Try to recover object capacity while keeping P9 far cleanup |
| P11 | P9 | 0.030 | 0.04 | Stronger support-tail removal |
| P12 | P3 | 0.020 | 0.02 | Test whether P3's Object J miss improves with the same support cut |

Commands:

```bash
GPU=6 bash scripts/experiments/medium_attr_p10_b02_proxy_geom000_opacity050_margin003_capthr002_iui3.sh
GPU=7 bash scripts/experiments/medium_attr_p11_b02_proxy_geom000_opacity050_margin003_capthr004_iui3.sh
GPU=8 bash scripts/experiments/medium_attr_p12_b02_proxy_geom000_opacity050_capthr002_iui3.sh
```

## P10-P12 Results

| Run | PSNR | SSIM | LPIPS | Far Accum | Far Clear | Far BG Frac | Far BG LCC Max | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 current-code | 31.0738 | 0.911893 | 0.175782 | 0.291455 | 0.077338 | 0.230165 | 0.118401 | 0.006519 | 0.000761 | 0.947713 | 0.992589 | 0.963250 |
| P3 | 31.2235 | 0.913678 | 0.174772 | 0.294079 | 0.061725 | 0.145686 | 0.097096 | 0.021233 | 0.000502 | 0.975145 | 0.970946 | 0.994212 |
| P9 | 31.1912 | 0.913403 | 0.178260 | 0.227394 | 0.064931 | 0.096367 | 0.113812 | 0.014574 | 0.000670 | 0.920260 | 0.977208 | 0.967358 |
| P10 P9 + threshold 0.02 | 31.0864 | 0.914303 | 0.176892 | 0.264315 | 0.074018 | 0.163288 | 0.106589 | 0.006921 | 0.000919 | 0.949545 | 0.975125 | 0.924177 |
| P11 P9 + threshold 0.04 | 31.1917 | 0.913542 | 0.177140 | 0.302929 | 0.081297 | 0.212510 | 0.115429 | 0.008820 | 0.000701 | 0.970120 | 0.988361 | 0.963415 |
| P12 P3 + threshold 0.02 | 30.9602 | 0.913443 | 0.177699 | 0.239073 | 0.067689 | 0.137320 | 0.118535 | 0.003547 | 0.001001 | 0.921146 | 0.957111 | 0.962421 |

Interpretation:

- Thresholding support can recover Object Acc Ret only at the cost of losing far-water cleanup. P11 is object-safe but Far Accum/Far Clear/Far BG revert close to or worse than C2 current-code.
- P10 and P12 are not viable: both miss retention or reconstruction while failing to keep P9-level far cleanup.
- Pure support threshold is therefore not the right object-protection mechanism.

## P13-P15 Plan

New default-off flags:

```text
medium_support_region_exclusion_enabled: bool = False
medium_support_exclude_object: bool = True
medium_support_exclude_boundary: bool = False
```

These flags use train-time high-precision region masks only to zero the support that enters capacity/proxy losses. They do not change inference composition, do not use q_hit, and do not use RGB mix.

Mask source:

```text
common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726
```

Experiment matrix:

| ID | Base | Exclude Object | Exclude Boundary | Purpose |
| --- | --- | ---: | ---: | --- |
| P13 | P9 | yes | no | Recover object capacity while preserving boundary/halo cleanup |
| P14 | P9 | yes | yes | Stronger object/boundary protection |
| P15 | P3 | yes | no | Test whether P3 object-J issue is helped by object support exclusion |

Commands:

```bash
GPU=6 bash scripts/experiments/medium_attr_p13_b02_proxy_geom000_opacity050_margin003_objexclude_iui3.sh
GPU=7 bash scripts/experiments/medium_attr_p14_b02_proxy_geom000_opacity050_margin003_objboundaryexclude_iui3.sh
GPU=8 bash scripts/experiments/medium_attr_p15_b02_proxy_geom000_opacity050_objexclude_iui3.sh
```
