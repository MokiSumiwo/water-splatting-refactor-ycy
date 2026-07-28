# Medium-Explainable Scene-Medium Attribution

Date: 2026-07-28

Branch: `refactor/core-framework`

Pre-change pushed commit: `8ceb54a026feb5bea5c3e8675caee7c8d0361c60` (`Add deterministic resume and turnover diagnostics`)

## Goal

The objective of this round was to move away from fixed Gaussian candidate interventions and test a dense, pixel-space attribution mechanism:

1. Build medium-explainable support from image flatness, detached medium color explainability, and detached depth.
2. Optionally route training reconstruction gradients toward the medium branch in supported pixels.
3. Optionally apply dense budgeted Gaussian accumulation suppression on that support.
4. Optionally refine remaining clear/dewatered chroma using the differentiable `J_proxy_raw` path.

Inference remains unchanged: `outputs["pred_image"]` is the physical renderer output. The routed prediction is used only for training loss.

## Code Changes

Added `water_splatting/attribution/medium_explainability.py`:

- `compute_image_structure_support`: low-gradient, low-variance image support.
- `compute_medium_explainability`: detached chroma/luma explainability of `B_inf` or `medium_rgb`.
- `compute_far_depth_support`: weak detached depth support with a floor.
- `build_route_capacity_support`: creates `S_route`, `S_cap`, and bootstrap support without using accumulation.
- `build_training_routed_prediction`: training-only blend between physical RGB and medium RGB.
- `budgeted_capacity_loss`: dense softplus budget on Gaussian accumulation.
- `clear_proxy_chroma_loss`: support-weighted medium-direction chroma penalty on `J_proxy_raw`.
- `clear_proxy_luma_budget_loss`: optional support-weighted clear-proxy luma budget.

Modified `water_splatting/water_splatting.py`:

- Added default-off config flags for medium support, routing, budgeted capacity, and optional proxy luma.
- Builds medium support in `get_loss_dict()` only when a related mechanism is enabled.
- Routes only the training reconstruction loss when `training_gradient_routing_enabled=True`.
- Adds medium explainability, budgeted capacity, medium-support proxy chroma, and optional proxy luma losses.
- Does not change evaluation or inference composition.

Modified `scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh`:

- Exposes all new flags in the manifest and `ns-train` model args.
- Keeps all new mechanisms default-off.

Added `scripts/diagnostics/diagnose_medium_explainability_support.py`:

- Saves `S_flat`, `S_med`, `S_far`, `S_route`, `S_cap`, accumulation, proxy blue/green, GT, RGB, and `B_inf`.
- Reports support coverage, water/object/boundary support means, and correlations with accumulation, gradient, and medium error.

Added experiment scripts:

- `scripts/experiments/medium_attr_a1_explain_iui3.sh`
- `scripts/experiments/medium_attr_a2_route_iui3.sh`
- `scripts/experiments/medium_attr_b2_capacity005_iui3.sh`
- `scripts/experiments/medium_attr_c2_capacity_proxy_chroma_iui3.sh`
- `scripts/experiments/medium_attr_a1_proxy_chroma015_iui3.sh`
- `scripts/experiments/medium_attr_b2_noroute_capacity005_iui3.sh`
- `scripts/experiments/medium_attr_b1_noroute_capacity008_iui3.sh`
- `scripts/experiments/medium_attr_b05_noroute_capacity005_iui3.sh`
- `scripts/experiments/medium_attr_b05_noroute_capacity005_start8000_iui3.sh`
- `scripts/experiments/medium_attr_b01_noroute_capacity005_iui3.sh`
- `scripts/experiments/medium_attr_b02_noroute_capacity005_iui3.sh`
- `scripts/experiments/medium_attr_b03_noroute_capacity005_iui3.sh`
- `scripts/experiments/medium_attr_c1_b02_capacity_proxy_chroma001_iui3.sh`
- `scripts/experiments/medium_attr_c2_b02_capacity_proxy_chroma015_iui3.sh`

## New Config Flags

```text
background_clear_chroma_use_medium_support
medium_explainability_enabled
medium_explainability_start_step
medium_explainability_ramp_steps
lambda_medium_explainability
training_gradient_routing_enabled
gradient_routing_start_step
gradient_routing_ramp_steps
gradient_routing_min_scene_weight
budgeted_capacity_enabled
budgeted_capacity_start_step
budgeted_capacity_ramp_steps
budgeted_capacity_value
budgeted_capacity_temperature
lambda_budgeted_capacity
budgeted_capacity_post_scale
medium_support_gradient_tau
medium_support_variance_tau
medium_support_color_tau
medium_support_luma_weight
medium_support_far_floor
medium_support_depth_mid
medium_support_depth_temperature
medium_support_use_flatness
medium_support_use_medium
medium_support_use_far
lambda_proxy_clear_luma
proxy_clear_luma_budget
proxy_clear_luma_temperature
```

## Smoke Tests

Completed before full runs:

- `py_compile` passed for the new attribution module, model, and support diagnostic script.
- `bash -n` passed for the modified base launcher and medium attribution experiment scripts.
- `git diff --check` passed.
- `medium_attr_c2_smoke_iui3_20` completed 20 training steps.
- `medium_attr_c2_smoke_iui3_200` completed 200 training steps.
- 200-step support diagnostic sanity:
  - `support_route_mean`: `0.0770`
  - `support_capacity_mean`: `0.0120`
  - `water_support_capacity_mean`: `0.0196`
  - `object_support_capacity_mean`: `0.000806`
  - `boundary_support_capacity_mean`: `0.000214`
  - `object_over_water_support`: `0.078`
  - `boundary_over_water_support`: `0.019`

## Experiment Commands

All runs used:

```text
DATA_PATH=/mnt/new/home_old/ycy/water-splatting-refactor/undistorted_data/undistorted_IUI3-RedSea
OUTPUT_DIR=/mnt/new/home_old/ycy/water-splatting-refactor/outputs
RENDER_ROOT=/mnt/new/home_old/ycy/water-splatting-refactor/renders
LOG_ROOT=/mnt/new/home_old/ycy/water-splatting-refactor/logs
seed=42
max_iterations=15000
medium_context_mode=dir_xy_camera
b_inf_mode=tied
infinite_water_enabled=False
old bg-J / old M2 mechanisms off
```

Full commands:

```bash
GPU=6 STAMP=20260728_a1_explain_full scripts/experiments/medium_attr_a1_explain_iui3.sh
GPU=7 STAMP=20260728_a2_route_full scripts/experiments/medium_attr_a2_route_iui3.sh
GPU=8 STAMP=20260728_b2_capacity005_full scripts/experiments/medium_attr_b2_capacity005_iui3.sh
GPU=9 STAMP=20260728_c2_capacity_proxy_chroma_full scripts/experiments/medium_attr_c2_capacity_proxy_chroma_iui3.sh
GPU=6 STAMP=20260728_a1_proxy_chroma015_full scripts/experiments/medium_attr_a1_proxy_chroma015_iui3.sh
GPU=7 STAMP=20260728_b2_noroute_capacity005_full scripts/experiments/medium_attr_b2_noroute_capacity005_iui3.sh
GPU=8 STAMP=20260728_b1_noroute_capacity008_full scripts/experiments/medium_attr_b1_noroute_capacity008_iui3.sh
GPU=9 STAMP=20260728_b05_noroute_capacity005_full scripts/experiments/medium_attr_b05_noroute_capacity005_iui3.sh
GPU=8 STAMP=20260728_b01_noroute_capacity005_full scripts/experiments/medium_attr_b01_noroute_capacity005_iui3.sh
GPU=9 STAMP=20260728_b05_noroute_capacity005_start8000_full scripts/experiments/medium_attr_b05_noroute_capacity005_start8000_iui3.sh
GPU=8 STAMP=20260728_b02_noroute_capacity005_full scripts/experiments/medium_attr_b02_noroute_capacity005_iui3.sh
GPU=9 STAMP=20260728_b03_noroute_capacity005_full scripts/experiments/medium_attr_b03_noroute_capacity005_iui3.sh
GPU=8 STAMP=20260728_c1_b02_capacity_proxy_chroma001_full scripts/experiments/medium_attr_c1_b02_capacity_proxy_chroma001_iui3.sh
GPU=9 STAMP=20260728_c2_b02_capacity_proxy_chroma015_full scripts/experiments/medium_attr_c2_b02_capacity_proxy_chroma015_iui3.sh
```

Support diagnostics:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_medium_explainability_support.py \
  --load-config outputs/<experiment>/water-splatting/<timestamp>/config.yml \
  --split eval --max-images 4 \
  --mask-dir common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726 \
  --output-dir renders/<render_dir>/support_diag \
  --output-json renders/<render_dir>/support_diag/support_diagnostic.json \
  --enable-clear-proxy
```

## Final Main Results

Reference is deterministic `R0_resume`.

| Run | PSNR | SSIM | LPIPS | J Blue | Far Accum | Far Clear | Water Accum | Water J | Obj J Ret | Boundary Ret | Obj Acc Ret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 resume | 30.8838 | 0.9131 | 0.1781 | 0.1166 | 0.8008 | 0.0812 | 0.7952 | 0.02636 | 0.9961 | 0.9832 | 1.0002 |
| A1 explain | 31.0934 | 0.9123 | 0.1774 | 0.1250 | 0.6835 | 0.0741 | 0.4120 | 0.000930 | 0.9667 | 0.9826 | 1.0085 |
| A1 + proxy 0.0015 | 31.1514 | 0.9136 | 0.1764 | 0.1440 | 0.3747 | 0.0608 | 0.1373 | 0.001159 | 0.9863 | 0.9666 | 0.9752 |
| B01 no-route cap 0.0001 | 30.9064 | 0.9115 | 0.1749 | 0.1572 | 0.3459 | 0.0875 | 0.0194 | 0.001713 | 0.9973 | 1.0192 | 0.9761 |
| B02 no-route cap 0.0002 | 31.0180 | 0.9118 | 0.1772 | 0.0574 | 0.2942 | 0.0725 | 0.0183 | 0.000733 | 0.9737 | 0.9713 | 0.9503 |
| C1 B02 + proxy 0.0010 | 31.1046 | 0.9141 | 0.1749 | 0.0891 | 0.3751 | 0.0719 | 0.0228 | 0.000279 | 0.9704 | 0.9407 | 0.9778 |
| **C2 B02 + proxy 0.0015** | **31.1139** | **0.9146** | **0.1765** | **0.0525** | **0.2374** | **0.0595** | **0.0068** | **0.000762** | **0.9753** | **0.9561** | **0.9313** |
| B03 no-route cap 0.0003 | 30.9785 | 0.9124 | 0.1758 | 0.1511 | 0.2928 | 0.0760 | 0.0203 | 0.000678 | 0.9963 | 1.0370 | 0.9537 |
| B05 no-route cap 0.0005 | 31.1412 | 0.9122 | 0.1793 | 0.0484 | 0.1936 | 0.0603 | 0.0033 | 0.000672 | 0.9910 | 0.9370 | 0.8906 |
| B05 start8000 | 31.0053 | 0.9127 | 0.1765 | 0.0599 | 0.2485 | 0.0759 | 0.0072 | 0.001066 | 0.9886 | 1.0068 | 0.9232 |
| B1 budget 0.08 | 31.1513 | 0.9145 | 0.1775 | 0.0458 | 0.1994 | 0.0601 | 0.0015 | 0.000598 | 0.9584 | 0.9359 | 0.8867 |
| B2 no-route cap 0.001 | 31.0583 | 0.9123 | 0.1773 | 0.0464 | 0.2014 | 0.0654 | 0.0043 | 0.000580 | 0.9621 | 0.9785 | 0.8839 |
| A2 route | 31.1350 | 0.9134 | 0.1748 | 0.3530 | 0.9996 | 0.1051 | 0.9997 | 0.007992 | 1.0056 | 0.9333 | 1.0234 |
| B2 route + capacity | 30.8964 | 0.9129 | 0.1757 | 0.0677 | 0.8836 | 0.1447 | 0.8025 | 0.05193 | 1.0167 | 0.8399 | 1.0132 |
| C2 route + capacity + proxy | 30.8511 | 0.9130 | 0.1768 | 0.0437 | 0.6974 | 0.1600 | 0.4858 | 0.04465 | 1.0677 | 0.9803 | 1.0007 |

Relative to `R0_resume`:

| Run | dPSNR | dSSIM | dLPIPS | J Blue | Far Accum | Far Clear | Water Accum | Water J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 explain | +0.2096 | -0.00077 | -0.00075 | +7.2% | -14.7% | -8.8% | -48.2% | -96.5% |
| A1 + proxy 0.0015 | +0.2676 | +0.00049 | -0.00177 | +23.6% | -53.2% | -25.1% | -82.7% | -95.6% |
| B01 no-route cap 0.0001 | +0.0226 | -0.00166 | -0.00319 | +34.9% | -56.8% | +7.7% | -97.6% | -93.5% |
| B02 no-route cap 0.0002 | +0.1342 | -0.00131 | -0.00091 | -50.7% | -63.3% | -10.8% | -97.7% | -97.2% |
| C1 B02 + proxy 0.0010 | +0.2208 | +0.00098 | -0.00323 | -23.5% | -53.2% | -11.4% | -97.1% | -98.9% |
| **C2 B02 + proxy 0.0015** | **+0.2301** | **+0.00150** | **-0.00166** | **-55.0%** | **-70.4%** | **-26.8%** | **-99.1%** | **-97.1%** |
| B03 no-route cap 0.0003 | +0.0947 | -0.00076 | -0.00236 | +29.7% | -63.4% | -6.4% | -97.4% | -97.4% |
| B05 no-route cap 0.0005 | +0.2575 | -0.00094 | +0.00118 | -58.5% | -75.8% | -25.7% | -99.6% | -97.5% |
| B05 start8000 | +0.1215 | -0.00043 | -0.00165 | -48.6% | -69.0% | -6.6% | -99.1% | -96.0% |
| B1 budget 0.08 | +0.2675 | +0.00134 | -0.00060 | -60.7% | -75.1% | -26.1% | -99.8% | -97.7% |
| B2 no-route cap 0.001 | +0.1745 | -0.00085 | -0.00087 | -60.2% | -74.8% | -19.4% | -99.5% | -97.8% |
| A2 route | +0.2512 | +0.00027 | -0.00337 | +202.9% | +24.8% | +29.4% | +25.7% | -69.7% |
| B2 route + capacity | +0.0126 | -0.00021 | -0.00247 | -41.9% | +10.4% | +78.1% | +0.9% | +97.0% |
| C2 route + capacity + proxy | -0.0327 | -0.00016 | -0.00130 | -62.5% | -12.9% | +97.0% | -38.9% | +69.4% |

## Support Diagnostics

| Run | Water S_cap | Object/Water S_cap | Boundary/Water S_cap | Corr(S_cap, Accum) |
|---|---:|---:|---:|---:|
| A1 explain | 0.07169 | 0.0055 | 0.0052 | -0.438 |
| A1 + proxy 0.0015 | 0.07791 | 0.0079 | 0.0048 | -0.650 |
| B01 no-route cap 0.0001 | 0.07599 | 0.0147 | 0.0076 | -0.686 |
| B02 no-route cap 0.0002 | 0.07453 | 0.0076 | 0.0053 | -0.713 |
| C1 B02 + proxy 0.0010 | 0.07561 | 0.0107 | 0.0061 | -0.691 |
| C2 B02 + proxy 0.0015 | 0.08103 | 0.0093 | 0.0049 | -0.733 |
| B03 no-route cap 0.0003 | 0.07043 | 0.0369 | 0.0100 | -0.697 |
| B05 no-route cap 0.0005 | 0.07727 | 0.0577 | 0.0094 | -0.743 |
| A2 route | 0.0000058 | 0.0826 | 0.2017 | -0.001 |
| B2 route + capacity | 0.000642 | 17.77 | 2.08 | -0.285 |
| B2 no-route capacity | 0.07886 | 0.0122 | 0.0068 | -0.758 |
| C2 route + capacity + proxy | 0.000536 | 35.56 | 4.26 | -0.258 |

The route-enabled path caused support collapse or support inversion by the end of training. `B2 route + capacity` and `C2 route + capacity + proxy` no longer target water at eval time; their support is dominated by object regions. This explains their high Far Clear / Water J failures despite lower global J Blue.

The no-route capacity branch preserved water-targeted support but was too aggressive for object accumulation: it nearly eliminated water accumulation, but object accumulation retention fell to `0.8839`.

## Contact Sheet

Generated comparison sheet:

```text
renders/contact_sheets/medium_explainable_attr_20260728_long.png
renders/contact_sheets/medium_explainable_attr_final_candidates_20260728_long.png
```

The final sheet includes GT, R0, A1, B02, C2-B02, C1-B02, B05, A1+proxy, and A2 route for eval RGB and `J`.

## Conclusions

1. `medium_explainability` alone is a strong positive result. It sharply reduces Water J and Water Accum while improving PSNR, but J Blue remains slightly worse than R0 and Object J retention is slightly below the old formal 0.97 target.
2. The current training-only routing formulation is not safe. It improves underwater metrics but drives J Blue / Far Accum / Water Accum in the wrong direction and collapses `S_cap` in water.
3. Dense budgeted capacity is a real accumulation lever when routing is disabled. High weights (`0.0005` to `0.001`) remove water accumulation very strongly but over-suppress object accumulation and/or boundary retention.
4. `B02 no-route capacity` (`A_budget=0.05`, `lambda_capacity=0.0002`) is the best object-safe capacity base: J Blue `-50.7%`, Far Accum `-63.3%`, Water Accum `-97.7%`, Water J `-97.2%`, Object J retention `0.9737`, Boundary retention `0.9713`.
5. `C2 B02 + proxy chroma 0.0015` is the current best leakage candidate and first configuration in this round to satisfy the main leakage/retention gates: PSNR `+0.2301 dB`, J Blue `-55.0%`, Far Accum `-70.4%`, Far Clear `-26.8%`, Water Accum `-99.1%`, Water J `-97.1%`, Object J retention `0.9753`, Boundary retention `0.9561`.
6. `C1 B02 + proxy chroma 0.0010` preserves object accumulation better than C2-B02 (`0.9778` vs `0.9313`) but fails boundary retention (`0.9407`) and does not reach Far Clear / J Blue improvements of C2-B02.
7. Route-enabled B/C experiments should not be promoted. The failure mode is now diagnosed as support inversion, not merely hyperparameter strength.

## Next Experimental Direction

Priority should shift to validating and stress-testing C2-B02:

1. Keep `medium_explainability_enabled=True`.
2. Keep `training_gradient_routing_enabled=False`.
3. Use the C2-B02 core:
   - `budgeted_capacity_value=0.05`
   - `lambda_budgeted_capacity=0.0002`
   - `background_clear_chroma_use_medium_support=True`
   - `lambda_background_clear_chroma=0.0015`
4. Run multi-seed full validation (`42`, `123`, `3407`) before calling it the formal replacement.
5. Add visual/object diagnostics around boundary regions because C2-B02 passes boundary retention but with limited margin (`0.9561`).
6. Do not run more route-enabled experiments until routing support is redesigned.

Stop conditions:

- If C2-B02 fails retention on another seed, test `lambda_background_clear_chroma=0.00125` or reduce `lambda_capacity` slightly before changing support.
- If Far Clear remains seed-unstable, test optional proxy luma only at a very small weight; do not increase capacity weight first.
- Do not re-enable old M2 ownership, near-zero, hard prune, opacity decay, q-hit protection, or fixed Gaussian candidate surgery as the main line.

## Generated Artifacts

Code/notes intended for Git:

- `water_splatting/attribution/`
- `water_splatting/water_splatting.py`
- `scripts/diagnostics/diagnose_medium_explainability_support.py`
- `scripts/experiments/medium_attr_*.sh`
- this research note

Generated artifacts intentionally not tracked:

- `outputs/medium_attr_*`
- `renders/medium_attr_*`
- `renders/contact_sheets/medium_explainable_attr_20260728_long.png`
- `renders/contact_sheets/medium_explainable_attr_final_candidates_20260728_long.png`
- `logs/medium_attr_*`
