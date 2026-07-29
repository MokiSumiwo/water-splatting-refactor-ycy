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

Pending.
