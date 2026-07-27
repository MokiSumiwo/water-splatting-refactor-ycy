# Background Attribution and Densification Experiments - 2026-07-26

## Scope

Continue from commit `91c1006b5ee44d0e779217166777f1688f9040e1` in
`/mnt/new/home_old/ycy/water-splatting-refactor`.

The goal is to test high-precision open-water masks, renderer-consistent
background medium losses, background clear-Gaussian suppression, and
densification region diagnostics without changing the CUDA renderer and without
reviving the retired M2 mechanisms.

## Code Changes

- Added high-precision open-water mask builder:
  - `scripts/diagnostics/build_high_precision_water_masks.py`
- Added renderer-consistent background attribution losses:
  - `water_splatting/losses/background_attribution.py`
  - wired through `water_splatting/losses/__init__.py`
- Updated `water_splatting/water_splatting.py`:
  - emits `rgb_medium_finite`, `rgb_tail`, `rgb_medium_total`
  - emits `J_raw`, `J_gaussian_raw`, `accumulation`
  - adds renderer-consistent background medium/tail losses
  - adds background clear-Gaussian suppression loss
  - adds region-sampled densification diagnostics and optional gradient weighting
- Added densification summary script:
  - `scripts/diagnostics/summarize_densification_regions.py`
- Extended experiment wrapper:
  - `scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh`
- Added experiment scripts:
  - `scripts/experiments/bg_attr_n1_precise_raw_binf_iui3.sh`
  - `scripts/experiments/bg_attr_n2_medium001_iui3.sh`
  - `scripts/experiments/bg_attr_n3_medium005_iui3.sh`
  - `scripts/experiments/bg_attr_n4_raw001_medium005_iui3.sh`
  - `scripts/experiments/bg_attr_j1_clear0001_iui3.sh`
  - `scripts/experiments/bg_attr_j2_clear0005_iui3.sh`
  - `scripts/experiments/bg_attr_j3_clear001_iui3.sh`
  - `scripts/experiments/bg_attr_f1_densify025_iui3.sh`
  - `scripts/experiments/bg_attr_f2_densify010_iui3.sh`

## Config Flags

All new flags default to disabled / no behavior change.

- Renderer-consistent background losses:
  - `lambda_background_medium_render: 0.0`
  - `lambda_background_tail_render: 0.0`
  - `background_render_loss_start_step: 0`
  - `background_render_loss_ramp_steps: 0`
- Background clear-Gaussian suppression:
  - `lambda_background_clear_gaussian: 0.0`
  - `background_clear_loss_start_step: 3000`
  - `background_clear_loss_ramp_steps: 3000`
  - `background_clear_use_raw_j: True`
  - `background_clear_exclude_boundary: True`
  - `background_clear_hit_exclusion_threshold: -1.0`
- Densification diagnostics / optional gate:
  - `background_densification_enabled: False`
  - `background_densification_weight: 1.0`
  - `uncertain_densification_weight: 0.5`
  - `background_densification_start_step: 3000`
  - `background_densification_ramp_steps: 3000`
  - `background_densification_diagnostic_only: True`
  - `densification_region_log_path: None`

## Mask

Selected high-precision mask:

`common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726`

Parameters:

- `foreground_depth_threshold=0.50`
- `erosion_radius=13`
- `edge_dilate_radius=5`
- `transition_radius=7`
- `top_connected_only=True`
- `water_max_y_fraction=0.25`
- `load_config=outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml`

Coverage:

| split | views | water mean | water min | water max |
| --- | ---: | ---: | ---: | ---: |
| train | 25 | 0.227825 | 0.198573 | 0.235551 |
| eval check | 4 | 0.223865 | n/a | n/a |

Contact sheet:

`common_masks/high_precision_water_m1_core_y025_nsorder_iui3_redsea_20260726/water_mask_contact_sheet.jpg`

Visual check: conservative top-open-water core, no obvious reef/seafloor coverage;
the mask has a deliberate hard top-region cutoff and is used only as a
high-precision background supervision / diagnostics mask.

## Smoke Tests

- Python compile passed for modified/new Python files.
- `bash -n` passed for experiment scripts.
- 100-step N3 smoke passed:
  - `logs/smoke_bg_attr_n3_medium005_nsorder_iui3_100_20260726_smoke_n3_nsorder`
- 100-step active J2 smoke passed with clear-Gaussian loss active from step 0.
- 100-step active F1 smoke passed with densification gate active from step 0:
  - `background_densification_effective_weight=0.25`
  - water weighted gradients lower than raw gradients.

## Baselines

| run | PSNR | SSIM | LPIPS | J blue | Far accum | Far clear | Water J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 dir_xy_camera | 31.131449 | 0.912010 | 0.175018 | 0.169100 | 0.407096 | 0.083962 | 0.000928 |
| E2/B2 tied bg=0.005 | 30.964132 | 0.912216 | 0.177284 | 0.114581 | 0.368988 | 0.068289 | 0.000952 |
| A3 bounded residual s=0.02 bg=0.005 | 31.295404 | 0.914427 | 0.175311 | 0.121400 | 0.457915 | 0.081729 | 0.001490 |

## Formal Batch 1: Precise Mask + Renderer-Consistent Loss

Commands:

```bash
GPU=6 STAMP=20260726_n1 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_n1_precise_raw_binf_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_n1_precise_raw_binf_iui3.sh
GPU=7 STAMP=20260726_n2 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_n2_medium001_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_n2_medium001_iui3.sh
GPU=8 STAMP=20260726_n3 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_n3_medium005_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_n3_medium005_iui3.sh
GPU=9 STAMP=20260726_n4 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_n4_raw001_medium005_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_n4_raw001_medium005_iui3.sh
```

Results:

| run | config | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | J blue | Far accum | Far clear | Water J | Obj J ret | Boundary ret | bg split mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N1 | precise mask + raw Binf 0.005 | 31.104534 | -0.026866 | 0.912893 | +0.000893 | 0.176719 | +0.001719 | 0.090082 | 0.510643 | 0.068129 | 0.000180 | 0.990953 | 0.949793 | 0.000747 |
| N2 | precise mask + bg_medium 0.001 | 30.903131 | -0.228269 | 0.913173 | +0.001173 | 0.177067 | +0.002067 | 0.111984 | 0.503369 | 0.073885 | 0.000728 | 0.993476 | 0.958437 | 0.001126 |
| N3 | precise mask + bg_medium 0.005 | 30.961487 | -0.169913 | 0.913801 | +0.001801 | 0.175698 | +0.000698 | 0.088499 | 0.603775 | 0.076201 | 0.000659 | 0.984586 | 1.027562 | 0.001157 |
| N4 | precise mask + raw Binf 0.001 + bg_medium 0.005 | 31.036030 | -0.095370 | 0.912867 | +0.000867 | 0.173550 | -0.001450 | 0.202921 | 0.533816 | 0.078434 | 0.001953 | 0.984194 | 0.981740 | 0.001504 |

Checkpoints:

- N1: `outputs/bg_attr_n1_precise_raw_binf_iui3_15000/water-splatting/bg_attr_n1_precise_raw_binf_iui3_15000_20260726_n1/nerfstudio_models/step-000014999.ckpt`
- N2: `outputs/bg_attr_n2_medium001_iui3_15000/water-splatting/bg_attr_n2_medium001_iui3_15000_20260726_n2/nerfstudio_models/step-000014999.ckpt`
- N3: `outputs/bg_attr_n3_medium005_iui3_15000/water-splatting/bg_attr_n3_medium005_iui3_15000_20260726_n3/nerfstudio_models/step-000014999.ckpt`
- N4: `outputs/bg_attr_n4_raw001_medium005_iui3_15000/water-splatting/bg_attr_n4_raw001_medium005_iui3_15000_20260726_n4/nerfstudio_models/step-000014999.ckpt`

Diagnostics:

- Each run saved `output.json`.
- Each run saved `diagnostics/backscatter_closure_diagnostic.json`.
- Each run saved `diagnostics/far_water/far_water_residual_diagnostic.json`.
- Each run saved `diagnostics/eval_regions/eval_region_diagnostic.json`.
- Each run saved `diagnostics/densification_regions_summary.json`.

Interpretation:

- N1 is the best reconstruction-retention point from batch 1, but LPIPS is
  outside the strict M1-relative threshold by about `0.0007`, and boundary
  retention is just under `0.95`.
- N3 has the best J-blue among N1-N4 and acceptable LPIPS delta, but PSNR drops
  by `0.17 dB` and far accumulation increases strongly.
- N4 has the best LPIPS but fails the leakage objective because J-blue rises to
  `0.202921`.
- Renderer-consistent medium loss did not improve far-clear leakage over E2/B2.
- Region diagnostics do not support the Phase E gate yet: background split
  fractions are around `0.07%-0.15%`, far below the `10%` threshold.

## Formal Batch 2: Background Clear-Gaussian Suppression

Base config: N3/R3 (`bg_medium=0.005`, raw Binf supervision off).

Commands launched:

```bash
GPU=6 STAMP=20260726_j1 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_j1_clear0001_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_j1_clear0001_iui3.sh
GPU=7 STAMP=20260726_j2 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_j2_clear0005_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_j2_clear0005_iui3.sh
GPU=8 STAMP=20260726_j3 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_j3_clear001_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_j3_clear001_iui3.sh
```

Results:

| run | config | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | J blue | Far accum | Far clear | Water J | Obj J ret | Boundary ret | bg split mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| J1 | N3 + bg-J 0.0001 | 30.934458 | -0.196991 | 0.911791 | -0.000219 | 0.175576 | +0.000558 | 0.104155 | 0.606246 | 0.071120 | 0.000603 | 0.980137 | 1.012828 | 0.001404 |
| J2 | N3 + bg-J 0.0005 | 30.937271 | -0.194178 | 0.911938 | -0.000072 | 0.178288 | +0.003270 | 0.139275 | 0.694319 | 0.076642 | 0.000904 | 0.984786 | 0.993145 | 0.001495 |
| J3 | N3 + bg-J 0.0010 | 31.066254 | -0.065195 | 0.913114 | +0.001104 | 0.174829 | -0.000189 | 0.154116 | 0.558039 | 0.074122 | 0.000954 | 1.004415 | 1.014294 | 0.001438 |

Checkpoints:

- J1: `outputs/bg_attr_j1_clear0001_iui3_15000/water-splatting/bg_attr_j1_clear0001_iui3_15000_20260726_j1/nerfstudio_models/step-000014999.ckpt`
- J2: `outputs/bg_attr_j2_clear0005_iui3_15000/water-splatting/bg_attr_j2_clear0005_iui3_15000_20260726_j2/nerfstudio_models/step-000014999.ckpt`
- J3: `outputs/bg_attr_j3_clear001_iui3_15000/water-splatting/bg_attr_j3_clear001_iui3_15000_20260726_j3/nerfstudio_models/step-000014999.ckpt`

Interpretation:

- J3 is the best BG-J reconstruction point and passes SSIM/LPIPS retention, but
  PSNR is still `0.065 dB` below M1 and leakage is worse than N1/E2.
- J1 gives the best J-batch far-clear value (`0.071120`) but PSNR drops by
  `0.197 dB` and SSIM falls below M1.
- J2 is dominated by J1/J3 and should not be retained.
- BG-J did not confirm the intended clear-Gaussian suppression mechanism in this
  configuration: far accumulation remains high and J-blue worsens versus N3.
- Densification region fractions remain far below the Phase E gate:
  background split mean is only `0.140%-0.150%`, duplicate fraction is zero.

## Contact Sheets

Per-run eval component contact sheets:

- `renders/bg_attr_d0_m1_densdiag_iui3_15000_20260726_d0/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_d1_e2_b2_densdiag_iui3_15000_20260726_d1/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_n1_precise_raw_binf_iui3_15000_20260726_n1/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_n2_medium001_iui3_15000_20260726_n2/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_n3_medium005_iui3_15000_20260726_n3/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_n4_raw001_medium005_iui3_15000_20260726_n4/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_j1_clear0001_iui3_15000_20260726_j1/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_j2_clear0005_iui3_15000_20260726_j2/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_j3_clear001_iui3_15000_20260726_j3/diagnostics/contact_sheet_eval_components.jpg`

Each sheet includes eval `gt`, `rgb`, `J`, `J_raw`,
`J_gaussian_raw`, `rgb_medium_total`, `rgb_medium_finite`, `rgb_tail`, and
`accumulation` for all four eval views.

## Formal Densification Diagnostics

Additional diagnostic-only controls were run after N/J to complete the D matrix:

```bash
GPU=6 STAMP=20260726_d0 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_d0_m1_densdiag_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_d0_m1_densdiag_iui3.sh
GPU=7 STAMP=20260726_d1 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_d1_e2_b2_densdiag_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_d1_e2_b2_densdiag_iui3.sh
```

Notes:

- D0 is an M1-like diagnostic rerun with all new losses off and high-precision
  mask attached only for region logging. It should not replace the official M1
  reconstruction baseline because this wrapper path produced a lower PSNR
  single rerun.
- D1 reproduces the historical E2/B2 style (`tied`, bg=0.005) with the original
  `pseudo_depth_bg_iui3_redsea_20260725` mask so its region fractions are not
  directly comparable to the high-precision mask fractions.

Results:

| run | mask | PSNR | dPSNR | SSIM | LPIPS | J blue | Far accum | Far clear | Water J | bg grad mean | bg split mean | bg dup mean | bg split latest |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 M1 diag | high-precision | 30.681572 | -0.449877 | 0.911833 | 0.175759 | 0.184853 | 0.516757 | 0.088591 | 0.002721 | 0.001180 | 0.001404 | 0.000000 | 0.001530 |
| D1 E2/B2 diag | historical pseudo-depth | 31.100227 | -0.031221 | 0.912410 | 0.177276 | 0.099278 | 0.333923 | 0.070053 | 0.001810 | 0.067302 | 0.074591 | 0.009351 | 0.072334 |
| N1 E2-like | high-precision | 31.104534 | -0.026915 | 0.912893 | 0.176719 | 0.090082 | 0.510643 | 0.068129 | 0.000180 | 0.000711 | 0.000747 | 0.000000 | 0.000251 |
| N3 best R | high-precision | 30.961487 | -0.169962 | 0.913801 | 0.175698 | 0.088499 | 0.603775 | 0.076201 | 0.000659 | 0.001010 | 0.001157 | 0.000000 | 0.000768 |
| J3 best J | high-precision | 31.066254 | -0.065195 | 0.913114 | 0.174829 | 0.154116 | 0.558039 | 0.074122 | 0.000954 | 0.001130 | 0.001438 | 0.000000 | 0.001033 |

Gate decision:

- High-precision true-open-water mask: bg split fractions stay around
  `0.07%-0.15%`, duplicate fraction is zero. This does not support Phase E.
- Historical pseudo-depth E2/B2 mask: bg split mean is `7.46%`, latest `7.23%`,
  and duplicate mean is `0.94%`. This is elevated but still below the explicit
  `10%` gate, and the mask is lower precision / broader than the selected mask.
- Final D decision: do not run F1/F2 in this round. The measured failure mode is
  not high-precision open-water-driven split/duplicate; it is more likely
  broader mask attribution contamination plus persistent far accumulation.

## Current Decisions

- Best reconstruction candidate in this round: N1 if prioritizing PSNR, J3 if
  prioritizing LPIPS/retention. Neither is a strict full pass; N1 misses LPIPS,
  while J3 misses PSNR and leakage targets.
- Best leakage candidate in this round: N1 by far-clear / Water-J / J-blue
  balance. It improves J-blue versus E2 but raises far accumulation.
- Open-water densification driving is not confirmed in batch 1 under the
  selected high-precision mask because background split/duplicate fractions
  remain far below the Phase E threshold in both N and J batches.
- Historical E2/B2's broader pseudo-depth background mask does show elevated
  split pressure (`7.46%` mean), but it does not cross the pre-set `10%` gate and
  is not high-precision enough to justify suppressing densification.
- Do not run F1/F2 yet under the original gate rule. The current data does not
  justify background-excluded densification because the measured background
  split/duplicate candidate fraction is not the active failure mode.

## Negative Results To Preserve

- Pure renderer-consistent medium supervision (`bg_medium=0.001/0.005`) does not
  improve far-clear leakage relative to E2/B2 and can raise far accumulation.
- Combining raw Binf 0.001 with bg_medium 0.005 worsens J-blue strongly.
- Current precise mask is high precision but conservative; it cannot by itself
  stop far Gaussian accumulation.
- BG-J on top of N3 does not improve the overall leakage/reconstruction tradeoff.
  Larger bg-J weights worsen J-blue and/or far accumulation in this setup.

## Pending

- See the 2026-07-27 continuation below. Low bg-J was applied to N1 and bracketed.
- If future diagnostics show background split/duplicate fractions above the gate,
  run F1/F2 and optionally F3 only if mask precision remains acceptable.

## Continuation: N1 Low BG-J Bracket - 2026-07-27

Rationale:

- N1 was the best leakage/reconstruction balance from batch 1.
- N3-based BG-J did not work, but N1 was closer to the M1 reconstruction gate and
  had much better far-clear / J-blue behavior.
- Therefore the next conservative step was to apply very low bg-J to N1, without
  renderer-consistent bg-medium loss and without densification gate.

Additional code:

- Added offline render-component diagnostic:
  - `scripts/diagnostics/diagnose_render_components_on_masks.py`
- Added N1 low bg-J scripts:
  - `scripts/experiments/bg_attr_n1_j003_clear00003_iui3.sh`
  - `scripts/experiments/bg_attr_n1_j004_clear00004_iui3.sh`
  - `scripts/experiments/bg_attr_n1_j0045_clear000045_iui3.sh`
  - `scripts/experiments/bg_attr_n1_j005_clear00005_iui3.sh`
  - `scripts/experiments/bg_attr_n1_j01_clear0001_iui3.sh`

Smoke:

```bash
GPU=8 STAMP=20260727_smoke_n1j005_active MAX_NUM_ITERATIONS=100 EXPERIMENT_NAME=smoke_bg_attr_n1_j005_clear00005_active_iui3_100 RUN_EVAL=0 RUN_CLOSURE_DIAG=0 RUN_FAR_DIAG=0 RUN_REGION_DIAG=0 BACKGROUND_CLEAR_LOSS_START_STEP=0 BACKGROUND_CLEAR_LOSS_RAMP_STEPS=0 bash scripts/experiments/bg_attr_n1_j005_clear00005_iui3.sh
```

Result: passed. Densification summary was written and bg-J path was active from
step 0.

Formal commands:

```bash
GPU=8 STAMP=20260727_n1j003 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_n1_j003_clear00003_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_n1_j003_clear00003_iui3.sh
GPU=9 STAMP=20260727_n1j004 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_n1_j004_clear00004_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_n1_j004_clear00004_iui3.sh
GPU=6 STAMP=20260727_n1j0045 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_n1_j0045_clear000045_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_n1_j0045_clear000045_iui3.sh
GPU=6 STAMP=20260727_n1j005 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_n1_j005_clear00005_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_n1_j005_clear00005_iui3.sh
GPU=7 STAMP=20260727_n1j01 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_n1_j01_clear0001_iui3_15000 RUN_EVAL=1 RUN_CLOSURE_DIAG=1 RUN_FAR_DIAG=1 RUN_REGION_DIAG=1 bash scripts/experiments/bg_attr_n1_j01_clear0001_iui3.sh
```

Results:

| run | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | J blue | Far accum | Far clear | Water J | Obj J ret | Boundary ret | bg split mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N1 | 31.104534 | -0.026915 | 0.912893 | +0.000883 | 0.176719 | +0.001701 | 0.090082 | 0.510643 | 0.068129 | 0.000180 | 0.990953 | 0.949793 | 0.000747 |
| N1_J003 | 31.195288 | +0.063839 | 0.913601 | +0.001591 | 0.179384 | +0.004366 | 0.185291 | 0.723111 | 0.031264 | 0.000462 | 0.980943 | 0.843338 | 0.001240 |
| N1_J004 | 31.152946 | +0.021498 | 0.913691 | +0.001681 | 0.175071 | +0.000054 | 0.159346 | 0.447571 | 0.080543 | 0.000742 | 1.034253 | 0.994835 | 0.001299 |
| N1_J0045 | 31.030876 | -0.100573 | 0.913326 | +0.001316 | 0.174442 | -0.000575 | 0.211334 | 0.682689 | 0.097705 | 0.003162 | 1.005061 | 0.957249 | 0.001593 |
| N1_J005 | 31.078276 | -0.053173 | 0.912617 | +0.000607 | 0.176258 | +0.001240 | 0.088616 | 0.645665 | 0.062392 | 0.000473 | 0.998354 | 0.944613 | 0.001083 |
| N1_J01 | 31.010143 | -0.121305 | 0.913114 | +0.001104 | 0.176799 | +0.001781 | 0.195286 | 0.608083 | 0.073142 | 0.001555 | 0.989998 | 0.978742 | 0.001360 |

High-precision eval-water component diagnostic:

| run | hp water accum | hp tail | hp J luma | hp J blue frac |
| --- | ---: | ---: | ---: | ---: |
| N1 | 0.408968 | 0.039152 | 0.018819 | 0.111561 |
| N1_J003 | 0.782233 | 0.000351 | 0.019716 | 0.483712 |
| N1_J004 | 0.567396 | 0.001472 | 0.023799 | 0.242371 |
| N1_J0045 | 0.623199 | 0.000673 | 0.036740 | 0.483491 |
| N1_J005 | 0.716507 | 0.000160 | 0.017383 | 0.146805 |
| N1_J01 | 0.670776 | 0.007044 | 0.022376 | 0.281619 |

Contact sheets:

- `renders/bg_attr_n1_j003_clear00003_iui3_15000_20260727_n1j003/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_n1_j004_clear00004_iui3_15000_20260727_n1j004/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_n1_j0045_clear000045_iui3_15000_20260727_n1j0045/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_n1_j005_clear00005_iui3_15000_20260727_n1j005/diagnostics/contact_sheet_eval_components.jpg`
- `renders/bg_attr_n1_j01_clear0001_iui3_15000_20260727_n1j01/diagnostics/contact_sheet_eval_components.jpg`

Checkpoint highlights:

- N1_J004: `outputs/bg_attr_n1_j004_clear00004_iui3_15000/water-splatting/bg_attr_n1_j004_clear00004_iui3_15000_20260727_n1j004/nerfstudio_models/step-000014999.ckpt`
- N1_J005: `outputs/bg_attr_n1_j005_clear00005_iui3_15000/water-splatting/bg_attr_n1_j005_clear00005_iui3_15000_20260727_n1j005/nerfstudio_models/step-000014999.ckpt`

Interpretation:

- N1_J004 is the best reconstruction candidate in the continuation:
  - passes PSNR / SSIM / LPIPS thresholds vs M1
  - passes object retention and boundary retention
  - but does not meet leakage goals: Far Clear only improves `4.1%` vs M1 and
    J-blue only improves `5.8%`.
- N1_J005 is the best leakage candidate:
  - Far Clear improves `25.7%` vs M1
  - Water J improves `49.0%` vs M1
  - J-blue improves `47.6%` vs M1
  - but PSNR misses by `0.003 dB`, LPIPS misses by `0.00024`, and boundary
    retention is `0.9446`.
- N1_J003 and N1_J0045 are negative / unstable:
  - both over-suppress tail, raise high-precision water accumulation, and worsen
    J-blue.
- Low bg-J on N1 has a sharp non-monotonic response. It can strongly reduce
  far-clear residual, but the mechanism still transfers explanation into
  Gaussian accumulation rather than solving attribution.

Updated decision:

- Do not promote any N1_J variant as final.
- Retain N1_J004 as the reconstruction-safe ablation.
- Retain N1_J005 as the leakage reference ablation.
- Do not proceed to densification gate from these high-precision-mask runs:
  background split mean remains around `0.1%-0.16%` and duplicate fraction is
  effectively zero.
- The next scientifically justified step is not more bg-J weight tuning; it is
  an opacity/accumulation-gradient diagnostic or a different attribution signal
  that discourages Gaussian accumulation without hard pruning.

## 2026-07-27 opacity / accumulation-gradient diagnostic

Cleanup:

- Removed obsolete outputs/renders for old M2/M3/M4/dual-color branches, smoke
  runs, and negative bg-attr variants.
- Preserved current references/candidates: M1, E2/B2, A3, N1, N1_J004,
  N1_J005, and D1.
- Disk usage changed from `outputs=67G`, `renders=5.3G` to `outputs=10G`,
  `renders=935M`.
- Cleanup manifests were written outside the repo:
  - `/tmp/water_splatting_cleanup_outputs_20260727.txt`
  - `/tmp/water_splatting_cleanup_renders_20260727.txt`

Code additions:

- Added config flag:
  - `opacity_accumulation_diagnostic_enabled: bool = False`
- When enabled, densification JSONL now logs, per region:
  - opacity logit gradient signed stats
  - opacity alpha-gradient proxy signed stats
  - scale gradient norm stats
  - sampled accumulation
  - sampled final transmittance
  - sampled `J_gaussian_raw` luma
  - sampled `rgb_tail` luma
  - attempted sampled accumulation-output gradient
  - sampled accumulation vs opacity-gradient correlation
- Added global background fractions:
  - `background_opacity_grad_abs_fraction`
  - `background_opacity_decrease_pressure_fraction`
  - `background_opacity_increase_pressure_fraction`
  - `background_accumulation_grad_abs_fraction`
  - `background_accumulation_decrease_pressure_fraction`
  - `background_accumulation_increase_pressure_fraction`
- Short-run diagnostics now log both every 500 steps and at final step.
- Updated `scripts/diagnostics/summarize_densification_regions.py` to surface
  the new fields.
- Added script:
  - `scripts/experiments/bg_attr_n1_opacity_accumdiag_iui3.sh`

Smoke command:

```bash
GPU=6 MAX_NUM_ITERATIONS=500 EXPERIMENT_NAME=smoke_bg_attr_n1_opacity_accumdiag_iui3_500 STAMP=20260727_opacc_smoke2 RUN_EVAL=0 RUN_CLOSURE_DIAG=0 RUN_FAR_DIAG=0 RUN_REGION_DIAG=0 scripts/experiments/bg_attr_n1_opacity_accumdiag_iui3.sh
```

Smoke result:

- Passed 500-step train.
- Summary:
  - JSONL rows: `2` (`step=0`, `step=499`)
  - opacity gradients available: `true`
  - scale gradients available: `true`
  - accumulation output gradients available: `false`
- The false accumulation-gradient result is expected: `accumulation` is a
  renderer sibling output and is not directly consumed by the active loss path.

Formal diagnostic command:

```bash
GPU=6 MAX_NUM_ITERATIONS=15000 EXPERIMENT_NAME=bg_attr_n1_opacity_accumdiag_iui3_15000 STAMP=20260727_opacc_diag RUN_EVAL=0 RUN_CLOSURE_DIAG=0 RUN_FAR_DIAG=0 RUN_REGION_DIAG=0 scripts/experiments/bg_attr_n1_opacity_accumdiag_iui3.sh
```

Diagnostic artifacts:

- JSONL:
  - `logs/bg_attr_n1_opacity_accumdiag_iui3_15000_20260727_opacc_diag/densification_regions.jsonl`
- Summary:
  - `renders/bg_attr_n1_opacity_accumdiag_iui3_15000_20260727_opacc_diag/diagnostics/densification_regions_summary.json`
- Checkpoint:
  - `outputs/bg_attr_n1_opacity_accumdiag_iui3_15000/water-splatting/bg_attr_n1_opacity_accumdiag_iui3_15000_20260727_opacc_diag/nerfstudio_models/step-000014999.ckpt`

15k diagnostic summary:

| window | bg grad frac | bg split frac | water accum mean | water accum p95 | water opacity dec | water opacity inc | inc/dec | water tail luma | water J luma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.001232 | 0.001468 | 0.862864 | 0.985534 | 0.0000630 | 0.0001230 | 1.95x | 0.001577 | 0.149492 |
| 0-3000 | 0.002022 | 0.001932 | 0.797778 | 0.936366 | 0.0002787 | 0.0005441 | 1.95x | 0.001365 | 0.146892 |
| 3000-6000 | 0.000415 | 0.000673 | 0.828834 | 0.999855 | 0.000000187 | 0.000000211 | 1.12x | 0.003634 | 0.163134 |
| 6000-10000 | 0.000682 | 0.001090 | 0.864395 | 0.999871 | 0.000000118 | 0.000000105 | 0.89x | 0.001521 | 0.214072 |
| 10000-15000 | 0.001609 | 0.001923 | 0.927616 | 0.999890 | 0.0000000339 | 0.0000000888 | 2.62x | 0.000536 | 0.091462 |

Key interpretation:

- The high-precision water region reaches near-saturated sampled Gaussian
  accumulation by the end (`mean=0.994672`, `p95=0.999897` at step 14999).
- Tail contribution in the same sampled region is already very low in the late
  window (`tail luma mean=0.000536`), while water `J` luma remains non-zero.
  This supports the current conclusion that bg-J / bg-medium style objectives
  can suppress tail while moving explanation into Gaussian accumulation.
- Background split/duplicate pressure remains small under the high-precision
  mask (`bg split fraction mean=0.001468`, duplicate fraction zero), so a
  densification gate is not the primary next lever for this mask.
- Native opacity-gradient signal is weak and inconsistent:
  - early training pushes water opacity upward more than downward (`inc/dec`
    about `1.95x`)
  - 6000-10000 briefly has slightly more decrease pressure
  - after 10000 the signal is tiny and again biased toward increase
- Direct accumulation-output gradient is not available from the current graph
  without adding an explicit differentiable loss consumer of accumulation.

Decision:

- Do not add an accumulation-zero loss or opacity decay. Those would revive
  previously rejected mechanisms.
- Do not promote densification gate yet for the high-precision mask; split and
  duplicate fractions are too small to justify it as the main intervention.
- The most defensible next signal is diagnostic-only for now:
  high sampled water accumulation plus low tail luma plus non-zero J luma marks
  Gaussian-over-explanation after tail suppression.
- If we test an active intervention later, prefer an optimizer-side,
  mask-gated opacity-gradient experiment that only amplifies existing positive
  opacity gradients in high-accumulation water pixels. This is distinct from
  opacity decay because it does not inject a constant opacity-down force when
  the reconstruction gradient wants opacity up.
