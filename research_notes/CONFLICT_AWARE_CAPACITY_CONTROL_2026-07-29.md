# Conflict-Aware Dense Capacity Control - 2026-07-29

## Scope

This note records the Q-series follow-up after stopping the P-series proxy sweep.

The goal was to test whether P3's remaining object/far cleanup tradeoff comes from
`budgeted_capacity_loss` gradients, rather than from clear-proxy chroma gradients.

Outputs/renders were cleaned before the run. After cleanup, only M1, C2-B02, P3,
P9, P19, and P20 were retained. Q-series outputs were then added.

## Code Changes

- Added controlled capacity rendering in `water_splatting/water_splatting.py`.
- Added config flags:
  - `capacity_control_enabled`
  - `capacity_control_geometry_gradient_scale`
  - `capacity_control_opacity_gradient_scale`
  - `capacity_conflict_gate_enabled`
  - `capacity_conflict_rho`
- `capacity_control_accumulation` is generated through an auxiliary clear-proxy-style rasterization branch.
- Q1 uses `capacity_control_geometry_gradient_scale=0.0` and `capacity_control_opacity_gradient_scale=1.0`.
- Q2/Q3 additionally register an opacity-gradient hook on the capacity-control opacity tensor.
- Added densification-stat synchronization when Gaussian count changes unexpectedly between refinement callbacks.
- Added prepass correction for conflict-gate `autograd.grad(main_loss, opacities)` so its screen-space gradient side effect is subtracted from densification statistics.
- Added complete empty-Gaussian output keys so extreme failure models can still run eval/diagnostics.
- Added Q0 diagnostic script:
  - `scripts/diagnostics/diagnose_capacity_gradient_conflict.py`
- Added experiment scripts:
  - `scripts/experiments/medium_attr_q1_capacity_opacity_only_iui3.sh`
  - `scripts/experiments/medium_attr_q2_capacity_conflict025_iui3.sh`
  - `scripts/experiments/medium_attr_q3_capacity_conflict000_iui3.sh`

All new mechanisms are default-off.

## Q0 Gradient Audit

P3 checkpoints audited:

- `step-000005000.ckpt`
- `step-000010000.ckpt`
- `step-000014999.ckpt`

Commands:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_capacity_gradient_conflict.py \
  --load-config outputs/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000/water-splatting/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000_20260729_p3_b02_proxy_geom000_opacity050/config.yml \
  --load-step 5000 --max-images 4 \
  --output-json logs/q0_capacity_conflict_step5000.json

CUDA_VISIBLE_DEVICES=7 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_capacity_gradient_conflict.py \
  --load-config outputs/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000/water-splatting/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000_20260729_p3_b02_proxy_geom000_opacity050/config.yml \
  --load-step 10000 --max-images 4 \
  --output-json logs/q0_capacity_conflict_step10000.json

CUDA_VISIBLE_DEVICES=8 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_capacity_gradient_conflict.py \
  --load-config outputs/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000/water-splatting/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000_20260729_p3_b02_proxy_geom000_opacity050/config.yml \
  --load-step 14999 --max-images 4 \
  --output-json logs/q0_capacity_conflict_step14999.json
```

| Step | cap opacity norm | cap scale norm | cap mean norm | scale / opacity mass | mean / opacity mass | conflict fraction of cap-positive opacity | conflicting cap mass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 7.34e-07 | 7.41e-06 | 1.08e-05 | 12.51 | 30.09 | 0.5120 | 0.4033 |
| 10000 | 1.96e-07 | 4.80e-06 | 3.99e-06 | 18.02 | 38.91 | 0.4812 | 0.5298 |
| 14999 | 1.96e-06 | 8.02e-06 | 6.55e-06 | 6.93 | 8.88 | 0.4860 | 0.4100 |

Interpretation:

- P3 capacity gradients are not opacity-only. Scale and mean gradients carry much larger mass than opacity.
- Roughly half of cap-positive opacity gradients conflict with reconstruction wanting higher opacity.
- The hypothesis that capacity/reconstruction conflict is real is supported.

## Q-Series Experiment Matrix

Shared base:

```text
medium_context_mode = dir_xy_camera
b_inf_mode = tied
lambda_medium_explainability = 0.005
lambda_budgeted_capacity = 0.0002
budgeted_capacity_value = 0.05
lambda_background_clear_chroma = 0.0015
background_clear_chroma_margin = 0.02
clear_proxy_geometry_gradient_scale = 0.0
clear_proxy_opacity_gradient_scale = 0.5
clear_proxy_color_gradient_scale = 1.0
```

| ID | Capacity geometry grad | Capacity opacity grad | Conflict gate |
|---|---:|---:|---|
| Q1 | 0.0 | 1.0 | off |
| Q2 | 0.0 | 1.0 | rho = 0.25 |
| Q3 | 0.0 | 1.0 | rho = 0.0 |

Formal commands:

```bash
GPU=6 bash scripts/experiments/medium_attr_q1_capacity_opacity_only_iui3.sh
GPU=7 bash scripts/experiments/medium_attr_q2_capacity_conflict025_iui3.sh
GPU=9 bash scripts/experiments/medium_attr_q3_capacity_conflict000_iui3.sh
```

## Results

| Run | PSNR | SSIM | LPIPS | FarAccum | FarClear | FarBGFrac | FarBGLCCmax | WaterAccum | WaterJ | ObjAccRet | ObjJRet | BoundaryRet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3 | 31.2235 | 0.913678 | 0.174772 | 0.294079 | 0.061725 | 0.145686 | 0.097096 | 0.021233 | 0.000502 | 0.975145 | 0.970946 | 0.994212 |
| P9 | 31.1912 | 0.913403 | 0.178260 | 0.227394 | 0.064931 | 0.096367 | 0.113812 | 0.014574 | 0.000670 | 0.920260 | 0.977208 | 0.967358 |
| Q1 | 30.9785 | 0.913084 | 0.175349 | 0.397450 | 0.076567 | 0.169030 | 0.112062 | 0.019425 | 0.000732 | 0.975205 | 1.012164 | 0.974045 |
| Q2 | 20.2185 | 0.652023 | 0.607043 | 0.007350 | 0.003663 | 0.001516 | 0.003572 | 0.000170 | 0.000065 | 0.069925 | 0.100809 | 0.114629 |
| Q3 | 18.6832 | 0.611842 | 0.659933 | 0.000084 | 0.000088 | 0.000000 | 0.000000 | 0.000002 | 0.000001 | 0.000147 | 0.000271 | 0.012467 |

Diagnostics and renders:

- Q1 render root: `renders/medium_attr_q1_capacity_opacity_only_iui3_15000_20260729_q1_capacity_opacity_only`
- Q2 render root: `renders/medium_attr_q2_capacity_conflict025_iui3_15000_20260729_q2_capacity_conflict025`
- Q3 render root: `renders/medium_attr_q3_capacity_conflict000_iui3_15000_20260729_q3_capacity_conflict000`
- Q0 diagnostics:
  - `logs/q0_capacity_conflict_step5000.json`
  - `logs/q0_capacity_conflict_step10000.json`
  - `logs/q0_capacity_conflict_step14999.json`

## Interpretation

Q0 was useful and confirms the diagnosis: current dense capacity pressure has large scale/mean components and substantial opacity conflict with reconstruction.

Q1 answers the opacity-only question:

- Object accumulation retention is preserved (`0.975205`), but far cleanup regresses badly.
- FarAccum worsens from P3 `0.294079` to Q1 `0.397450`.
- FarClear worsens from P3 `0.061725` to Q1 `0.076567`.
- Therefore, P3's far cleanup depends strongly on capacity gradients through geometry/scale/footprint, not only opacity.

Q2/Q3 are negative but informative:

- The current opacity conflict gate is not a deployable protection mechanism.
- It suppresses/redirects opacity gradients so aggressively that the model collapses object and boundary representation.
- Q2/Q3 prove that naive sign-based opacity conflict gating is insufficient; it does not solve the object/far tradeoff.

## Engineering Notes

Two implementation hazards were found and fixed:

1. `autograd.grad(main_loss, opacities)` triggers the custom rasterizer backward and writes to `xys_grad_abs`. This must not be allowed to inflate densification gradients. The implemented fix records the prepass contribution and subtracts it in `after_train()`.
2. Extreme capacity runs can leave eval views with no visible Gaussian. Empty-render outputs now include all diagnostic keys required by eval, closure, far-water, and region diagnostics.

## Current Decision

Do not continue Q2/Q3 as-is.

Keep:

- Q0 capacity gradient audit script.
- `capacity_control_enabled` branch for controlled ablations.
- Q1 result as evidence that opacity-only capacity is object-safe but far-cleanup insufficient.

Do not promote:

- Q1, because reconstruction and far metrics regress versus P3.
- Q2/Q3, because they collapse object and boundary representation.

Current best candidate remains P3. P9 remains a non-deployable far-cleanup reference.

## Next Step

The next viable direction should not be naive opacity sign gating. The evidence suggests:

- far cleanup needs geometry/scale pressure;
- object safety needs a protection signal that operates before dense capacity reaches object-adjacent footprints;
- any future conflict-aware version must avoid per-step `autograd.grad` through the main rasterizer or isolate it in a separate diagnostic-only forward/backward path.

Recommended next experiment:

- keep P3 as base;
- test a bounded scale/geometry capacity gradient scale, not zero:
  - `capacity_control_geometry_gradient_scale = 0.10`
  - `capacity_control_opacity_gradient_scale = 1.0`
  - no conflict gate
- compare to Q1 and P3 to see whether a small amount of footprint pressure recovers far cleanup without P9-level object damage.

## Correction: Branch-Local Capacity Isolation

After reviewing commit `7f367e4`, Q2/Q3 could not be treated as reliable
evidence that sign-based conflict gating itself failed.
The invalid Q2/Q3 output, render, and log directories were deleted after this
correction; their metrics remain above only to document the implementation
artifact.

Issue found:

- `capacity_control_opacities = _scale_aux_grad(opacities, 1.0)` returned the
  original `opacities` tensor when scale was exactly `1.0`.
- The conflict hook was therefore registered on a tensor shared with the main
  render path.
- In Q2/Q3, that hook could scale main reconstruction opacity gradients, not
  only capacity gradients.

Fix:

- Capacity-control tensors now use a branch-local helper even when scale is
  exactly `1.0`:

```python
detached = value.detach()
branch_value = detached + scale * (value - detached)
```

- The capacity conflict hook now checks the incoming capacity gradient:

```python
conflict = (cap_grad > 0.0) & (rec_grad < -rec_threshold)
```

- Added split controls:
  - `capacity_control_position_gradient_scale`
  - `capacity_control_depth_gradient_scale`
  - `capacity_control_footprint_gradient_scale`
  - `capacity_conflict_rec_grad_threshold`

## Gradient-Isolation Verification

Diagnostic:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/verify_capacity_gradient_isolation.py \
  --load-config outputs/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000/water-splatting/medium_attr_p3_b02_proxy_geom000_opacity050_iui3_15000_20260729_p3_b02_proxy_geom000_opacity050/config.yml \
  --load-step 14999 \
  --force-step 14999 \
  --image-index 1 \
  --rho 0.25 \
  --geometry-scale 0.0 \
  --opacity-scale 1.0 \
  --output-json logs/verify_capacity_gradient_isolation_p3_step14999_img1.json
```

Result:

```text
capacity_control_opacities is main_render_opacities: False
capacity_control_opacities data_ptr equals main: False
capacity_control_opacities grad_fn: AddBackward0
main_render_opacities grad_fn: SigmoidBackward0

main_only_before_vs_after_hook:
    max_abs       = 2.91e-11
    relative_norm = 5.42e-07

gated_total_vs_expected:
    max_abs       = 1.58e-11
    relative_norm = 5.22e-07
```

This passes the intended isolation gate. The hook no longer changes main-only
opacity gradients.

## R-Series Experiments

Scripts:

- `scripts/experiments/medium_attr_r1_capacity_conflict025_isolated_iui3.sh`
- `scripts/experiments/medium_attr_r2_capacity_conflict000_isolated_iui3.sh`
- `scripts/experiments/medium_attr_r3_capacity_footprint010_iui3.sh`
- `scripts/experiments/medium_attr_r4_capacity_footprint025_iui3.sh`
- `scripts/experiments/medium_attr_r5_capacity_position005_footprint010_iui3.sh`

Commands:

```bash
GPU=7 bash scripts/experiments/medium_attr_r1_capacity_conflict025_isolated_iui3.sh
GPU=8 bash scripts/experiments/medium_attr_r2_capacity_conflict000_isolated_iui3.sh
GPU=9 bash scripts/experiments/medium_attr_r3_capacity_footprint010_iui3.sh
GPU=7 bash scripts/experiments/medium_attr_r4_capacity_footprint025_iui3.sh
GPU=8 bash scripts/experiments/medium_attr_r5_capacity_position005_footprint010_iui3.sh
```

| Run | PSNR | SSIM | LPIPS | FarAccum | FarClear | FarBGFrac | FarBGLCCmax | WaterAccum | WaterJ | ObjAccRet | ObjJRet | BoundaryRet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3 | 31.2235 | 0.913678 | 0.174772 | 0.294079 | 0.061725 | 0.145686 | 0.097096 | 0.021233 | 0.000502 | 0.975145 | 0.970946 | 0.994212 |
| Q1 | 30.9785 | 0.913084 | 0.175349 | 0.397450 | 0.076567 | 0.169030 | 0.112062 | 0.019425 | 0.000732 | 0.975205 | 1.012164 | 0.974045 |
| R1 | 30.9975 | 0.913815 | 0.175137 | 0.400138 | 0.079664 | 0.258594 | 0.159238 | 0.085502 | 0.001365 | 0.984699 | 0.976120 | 0.960178 |
| R2 | 30.7300 | 0.910564 | 0.175192 | 0.275255 | 0.068640 | 0.162580 | 0.105825 | 0.009350 | 0.000692 | 0.947264 | 0.980547 | 0.992124 |
| R3 | 30.8260 | 0.911688 | 0.178227 | 0.388411 | 0.068179 | 0.150741 | 0.106739 | 0.038065 | 0.000240 | 0.973791 | 0.977543 | 0.963894 |
| R4 | 30.9914 | 0.912165 | 0.175812 | 0.298163 | 0.063580 | 0.116519 | 0.103924 | 0.015636 | 0.000480 | 0.950579 | 0.976050 | 0.957869 |
| R5 | 31.0063 | 0.913000 | 0.174983 | 0.321917 | 0.073812 | 0.185691 | 0.109199 | 0.018447 | 0.000841 | 0.961491 | 0.969699 | 0.959937 |

## Updated Interpretation

R1/R2 no longer collapse globally after branch-local isolation. Therefore, the
old Q2/Q3 result should be marked as an implementation artifact rather than
a valid rejection of sign-based conflict gating.

However, the corrected R-series still does not produce a better candidate than
P3:

- R1 (`rho=0.25`) is object-safe but makes far residual much worse.
- R2 (`rho=0.0`) improves FarAccum versus P3 but fails PSNR and Object Acc Ret.
- R3 footprint-only `0.10` improves WaterJ but not far cleanup or reconstruction.
- R4 footprint-only `0.25` improves FarBGFrac but damages Object Acc Ret and still
  misses P3 reconstruction.
- R5 small position/depth plus footprint does not recover far cleanup and fails
  Object Acc Ret.

Current decision:

- P3 remains the best current candidate.
- Q1 remains useful evidence that opacity-only capacity is object-safe but
  insufficient for far cleanup.
- Corrected conflict gating is not rejected as a principle, but the simple
  opacity sign gate does not dominate P3.
- Small footprint/position scaling, as tested in R3-R5, does not solve the
  remaining far-water residual without object/reconstruction damage.

Recommended next direction:

- Do not continue scalar footprint/position sweeps.
- Investigate where P3's successful footprint/geometry gradient acts spatially,
  preferably with a region-aware gradient audit over support core, halo, object,
  and boundary.
- If continuing capacity control, gate footprint gradients with pixel/region
  support rather than a global scalar.
