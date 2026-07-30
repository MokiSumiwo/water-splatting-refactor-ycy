# M2-Core Release T-Series

Date: 2026-07-30

## Motivation

The P/Q/R/S series showed that late Gaussian parameter-level interventions
(`scale sign`, gradient clipping, persistence gates, and fixed candidate
surgery) do not reproduce the original M2 open-water cleanup. The next
experiment line returns to representation formation during training:

1. identify stable open-water support without using Gaussian accumulation;
2. release Gaussian RGB reconstruction responsibility in that support during
   training only;
3. apply dense high-confidence core capacity pressure before densification has
   finished;
4. add weak halo and object-radiance refinements for transition-band residuals;
5. keep inference/render composition identical to the physical renderer.

This does not re-enable old M2 inference RGB mix, ownership-masked J, near-zero
loss, dynamic hit protection, capacity floor, hard pruning, opacity decay, or
q-hit gating.

## Code Changes

- Added boundary-connected open-water support in
  `water_splatting/attribution/medium_explainability.py`.
  - `medium_support_connected_enabled`
  - `medium_support_connected_threshold`
  - `medium_support_connected_top_only`
  - `medium_support_connected_border`
- Added core zero-target capacity pressure.
  - `core_zero_capacity_enabled`
  - `lambda_core_zero_capacity`
  - `core_zero_capacity_start_step`
  - `core_zero_capacity_ramp_steps`
  - `core_zero_capacity_post_scale`
- Added detached accumulation clearance amplifier.
  - `core_clearance_amplifier_enabled`
  - `core_clearance_amplifier_min`
  - `core_clearance_amplifier_threshold`
  - `core_clearance_amplifier_temperature`
- Added object-radiance luma budget on `rgb_object`.
  - `object_radiance_budget_enabled`
  - `lambda_object_radiance_budget`
  - `object_radiance_budget_value`
  - `object_radiance_budget_temperature`
  - `object_radiance_budget_start_step`
  - `object_radiance_budget_ramp_steps`
- Extended `scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh`
  so all new flags are recorded in `run_manifest.txt` and passed through to
  `ns-train`.

## Experiment Scripts

- `scripts/experiments/medium_attr_t1_early_budget_iui3.sh`
  - P3 with budgeted capacity moved to step 1000, ramp 3000, no post-10000
    scale-down.
- `scripts/experiments/medium_attr_t2_corezero_halo_iui3.sh`
  - boundary-connected support, medium-independent support (`flat * far`),
    core zero capacity, weak residual-gated halo budget, no routing.
- `scripts/experiments/medium_attr_t3_corezero_halo_route_iui3.sh`
  - T2 plus training-only responsibility release with
    `gradient_routing_min_scene_weight=0.70`.
- `scripts/experiments/medium_attr_t4_corezero_halo_route_objrad_iui3.sh`
  - T3 plus object-radiance luma budget.
- `scripts/experiments/medium_attr_t5_corezero_halo_route_objrad_amp_iui3.sh`
  - T4 plus detached clearance amplifier.

## Smoke Tests

Smoke tests were run with 5 iterations and all new start/ramp steps overridden
to trigger the branches immediately. Outputs/logs were deleted after checking
to avoid clutter.

| Smoke | Branches exercised | Result |
| --- | --- | --- |
| T2 smoke | connected support, core-zero, halo support/loss | Passed |
| T4 smoke | T2 + training routing + object-radiance budget | Passed |
| T5 smoke | T4 + clearance amplifier | Passed |

Syntax checks:

```text
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/attribution/medium_explainability.py \
  water_splatting/attribution/__init__.py \
  water_splatting/water_splatting.py

bash -n scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh \
  scripts/experiments/medium_attr_t1_early_budget_iui3.sh \
  scripts/experiments/medium_attr_t2_corezero_halo_iui3.sh \
  scripts/experiments/medium_attr_t3_corezero_halo_route_iui3.sh \
  scripts/experiments/medium_attr_t4_corezero_halo_route_objrad_iui3.sh \
  scripts/experiments/medium_attr_t5_corezero_halo_route_objrad_amp_iui3.sh
```

## Formal Run Order

Run first:

```text
T1 -> T2 -> T3
```

Decision gate:

- If T2 is unstable or object/boundary retention collapses, reduce
  `lambda_core_zero_capacity` before running T4/T5.
- If T3 improves Far Accum / Far Clear without large PSNR or retention loss,
  run T4.
- Run T5 only if T4 is stable and still leaves visible connected residuals.

Primary metrics:

- Underwater PSNR / SSIM / LPIPS
- Common Far Accum
- Common Far Clear
- Far blue-green residual fraction and largest component
- Water Accum / Water J / J Blue
- Object Accum Retention / Object J Retention / Boundary Retention

## Open Questions

- Whether medium-independent boundary-connected support is too broad near
  seafloor transitions.
- Whether core zero with `lambda=0.0002` is too strong compared with P3 budget.
- Whether training-only responsibility release helps reduce Gaussian formation
  or mainly weakens reconstruction.
- Whether object-radiance budget can remove visible thin layers without
  damaging object retention.

## Formal T1/T2/T3 Results

Commit: `ceb8a2f9e9848f89969c202505ded261e730f2e8`

Formal runs completed:

```text
scripts/experiments/medium_attr_t1_early_budget_iui3.sh
scripts/experiments/medium_attr_t2_corezero_halo_iui3.sh
scripts/experiments/medium_attr_t3_corezero_halo_route_iui3.sh
```

Primary comparison:

| Exp | PSNR | SSIM | LPIPS | J Blue | Far Accum | Far Clear | Far BG Frac | Far BG LCC | Water Accum | Water J | Obj Acc Ret | Obj J Ret | Boundary Ret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P3 | 31.2235 | 0.9137 | 0.1748 | 0.0727 | 0.2941 | 0.0617 | 0.1457 | 0.0971 | 0.0212 | 0.000502 | 0.9751 | 0.9709 | 0.9942 |
| T1 | 30.9697 | 0.9132 | 0.1764 | 0.0535 | 0.2431 | 0.0702 | 0.1083 | 0.1078 | 0.0019 | 0.000548 | 0.9143 | 0.9782 | 0.9864 |
| T2 | 31.1774 | 0.9134 | 0.1735 | 0.1245 | 0.3428 | 0.0766 | 0.2502 | 0.2071 | 0.0377 | 0.001003 | 0.9736 | 0.9814 | 0.9726 |
| T3 | 31.0052 | 0.9124 | 0.1757 | 0.0751 | 0.2726 | 0.0724 | 0.2339 | 0.2345 | 0.0146 | 0.001255 | 0.9409 | 1.0025 | 1.0083 |

Artifacts:

```text
renders/medium_attr_t1_early_budget_iui3_15000_20260730_t1_early_budget/output.json
renders/medium_attr_t1_early_budget_iui3_15000_20260730_t1_early_budget/diagnostics/far_water/far_water_residual_diagnostic.json
renders/medium_attr_t1_early_budget_iui3_15000_20260730_t1_early_budget/diagnostics/eval_regions/eval_region_diagnostic.json

renders/medium_attr_t2_corezero_halo_iui3_15000_20260730_t2_corezero_halo/output.json
renders/medium_attr_t2_corezero_halo_iui3_15000_20260730_t2_corezero_halo/diagnostics/far_water/far_water_residual_diagnostic.json
renders/medium_attr_t2_corezero_halo_iui3_15000_20260730_t2_corezero_halo/diagnostics/eval_regions/eval_region_diagnostic.json

renders/medium_attr_t3_corezero_halo_route_iui3_15000_20260730_t3_corezero_halo_route/output.json
renders/medium_attr_t3_corezero_halo_route_iui3_15000_20260730_t3_corezero_halo_route/diagnostics/far_water/far_water_residual_diagnostic.json
renders/medium_attr_t3_corezero_halo_route_iui3_15000_20260730_t3_corezero_halo_route/diagnostics/eval_regions/eval_region_diagnostic.json
```

Checkpoints:

```text
outputs/medium_attr_t1_early_budget_iui3_15000/water-splatting/medium_attr_t1_early_budget_iui3_15000_20260730_t1_early_budget/nerfstudio_models/step-000014999.ckpt
outputs/medium_attr_t2_corezero_halo_iui3_15000/water-splatting/medium_attr_t2_corezero_halo_iui3_15000_20260730_t2_corezero_halo/nerfstudio_models/step-000014999.ckpt
outputs/medium_attr_t3_corezero_halo_route_iui3_15000/water-splatting/medium_attr_t3_corezero_halo_route_iui3_15000_20260730_t3_corezero_halo_route/nerfstudio_models/step-000014999.ckpt
```

## Formal Interpretation

T1 confirms that moving the P3 budgeted capacity earlier is powerful but not
safe. It reduces J Blue, Far Accum, Far BG Frac, and Water Accum, but it drops
PSNR by 0.2538 dB versus P3 and collapses object accumulation retention to
0.9143. Far Clear and Far BG largest component also worsen, so early dense
pressure is deleting or distorting useful scene capacity without removing the
dominant visible residual component.

T2 shows that the current boundary-connected, medium-independent core-zero
support is too broad or misaligned. It keeps PSNR relatively high, but it
worsens J Blue, Far Accum, Far Clear, Far BG Frac/LCC, Water Accum, and Water J
relative to P3. Core-zero plus halo budget did not recreate original M2 cleanup.

T3 shows that training-only responsibility release is not safe with this
support. It lowers Far Accum relative to P3, but PSNR falls to 31.0052, object
accumulation retention falls to 0.9409, and Far Clear / Far BG Frac / Far BG LCC
all worsen. The routing appears to weaken scene reconstruction without producing
the desired clear residual reduction.

Decision: do not run T4/T5 from this branch. The planned gate required T3 to be
stable before adding object-radiance budget or clearance amplifier; T3 fails
PSNR, object retention, Far Clear, Water J, and Far BG connected residual goals.

## Next Direction

The useful signal in T1 is that early dense pressure can strongly reduce core
water accumulation. The failure mode is support safety and object capacity
damage, not lack of capacity strength. The next experiment should not add more
loss terms on top of T3. Instead:

1. build an offline visualization/contact sheet of `medium_support_connected`,
   `medium_support_core`, and `medium_support_halo` for T2/T3 views;
2. tune support precision before changing weights:
   - raise `medium_support_connected_threshold`;
   - reintroduce medium/color evidence carefully, or add GT chroma color seed;
   - add side/top connected area constraints;
   - consider object/boundary exclusion only for support diagnostics first;
3. rerun a lighter T2 variant only after support coverage is visibly correct;
4. keep P3 as the current best formal candidate until a precision-safe early
   support is demonstrated.
