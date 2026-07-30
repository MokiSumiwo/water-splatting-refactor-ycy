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
