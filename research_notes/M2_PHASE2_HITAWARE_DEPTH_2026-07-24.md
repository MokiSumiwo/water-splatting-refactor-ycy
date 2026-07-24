# M2 Phase 2 Hit-Aware Depth Results - 2026-07-24

## Scope

Second-stage M2 experiments on IUI3-RedSea based on commit `c6260624b6c7f9a355092a07d7d46c20fc3ec8d5` and first-stage candidate E2 `lambda_accumulation_zero=0.002`. This phase implements fixed common-mask diagnostics, explicit seeds, low-cost ownership/depth ablations, CUDA hit-aware depth diagnostics, and a first hit-aware capacity-support experiment.

## Code Changes

- Added explicit `SEED` handling in `scripts/experiments/m2_infinite_water_iui3_redsea.sh` and pass-through to `--machine.seed`.
- Added manifest logging for seed, PyTorch/CUDA versions, and expanded M2/hit-aware config flags.
- Added fixed far-mask builder: `scripts/diagnostics/build_common_far_masks.py`.
- Extended `scripts/diagnostics/diagnose_far_water_residual.py` to support `--mask-dir`, common-mask clear leakage, J-object leakage, ownership coverage, hit confidence, and heatmaps.
- Added experiment scripts:
  - `scripts/experiments/m2_phase2_accum_seed_grid_iui3_redsea.sh`
  - `scripts/experiments/m2_phase2_ownership_ablation_iui3_redsea.sh`
  - `scripts/experiments/m2_phase2_depth_softening_iui3_redsea.sh`
  - `scripts/experiments/m2_phase2_hit_aware_capacity_iui3_redsea.sh`
- Added CUDA hit-aware depth outputs for RGB rasterization:
  - second depth moment
  - depth variance
  - relative depth std
  - first depth
  - last depth
  - final transmittance
- Added model outputs and diagnostics:
  - `hit_q_alpha`
  - `hit_q_conc`
  - `hit_confidence`
  - `m_support`
  - `m_render`
  - `m_capacity`
- Added `infinite_water_capacity_support_mode={m_inf,hit_alpha,hit,hit_squared}`. Default is `m_inf`, preserving first-stage behavior.

## Validation

- CUDA extension rebuilt successfully with `/opt/anaconda3/envs/water_splatting/bin/pip install -e . --no-build-isolation`.
- CUDA import passed after rebuild.
- `ns-train water-splatting --help` exposes hit-aware config flags.
- Python compile passed for edited Python files.
- `bash -n` passed for new experiment scripts.
- 10-step sanity passed for seeded P1 and H2 hit-aware capacity.

## Common Far Mask

Built from M1 `dir_xy_camera` expected-depth q90:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/build_common_far_masks.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --output-dir common_masks/m1_q90_iui3_redsea_20260724 \
  --far-depth-quantile 0.90 \
  --save-png
```

Common-mask diagnostic baseline:

| Experiment | PSNR | SSIM | LPIPS | common far accum | common far clear luma |
|---|---:|---:|---:|---:|---:|
| M1 `dir_xy_camera` | 31.1314 | 0.9120 | 0.1750 | 0.407096 | 0.083962 |
| Old M2 `alpha_depth` | 31.0696 | 0.9129 | 0.1771 | 0.119960 | 0.054452 |
| E2 old unseeded `accum=0.002` | 31.2206 | 0.9140 | 0.1765 | 0.185775 | 0.069223 |

Important correction: the first-stage per-model far diagnostics were too optimistic because each model used its own depth q90 mask. Under fixed M1-q90 pixels, E2 still reduces capacity leakage vs M1 but does not achieve the earlier apparent near-zero far leakage.

## Phase 1: Seed Stability And Local Accumulation Search

Fixed config unless noted:

```text
medium_context_mode=dir_xy_camera
ownership_mode=alpha_depth
compose_mode=rgb_mix
occupancy_limited=True
lambda_binf_rgb=0.005
lambda_near_zero=0
loss_start_step=1000
loss_ramp_steps=3000
max_iterations=15000
```

### Multi-seed stability

| Group | Seeds | PSNR mean +/- std | SSIM mean +/- std | LPIPS mean +/- std | J blue mean +/- std | common far accum mean +/- std | common far clear mean +/- std |
|---|---|---:|---:|---:|---:|---:|---:|
| R05 `accum=0.0005` | 42, 123, 3407 | 30.9883 +/- 0.2023 | 0.9121 +/- 0.0015 | 0.1752 +/- 0.0005 | 0.0832 +/- 0.0093 | 0.3049 +/- 0.0219 | 0.0872 +/- 0.0084 |
| R20 `accum=0.0020` | 42, 123, 3407 | 30.9378 +/- 0.1106 | 0.9133 +/- 0.0017 | 0.1757 +/- 0.0007 | 0.0721 +/- 0.0035 | 0.2107 +/- 0.0276 | 0.0663 +/- 0.0096 |

Decision: neither R05 nor R20 meets the proposed stability gate (`PSNR std <= 0.08 dB`). R20 is more effective for common-mask leakage; R05 is not a good capacity-control candidate under fixed mask.

### Seed-42 narrow search

| Experiment | PSNR | SSIM | LPIPS | J blue | common far accum | common far clear |
|---|---:|---:|---:|---:|---:|---:|
| `accum=0.0010` | 30.8618 | 0.9119 | 0.1755 | 0.0916 | 0.261948 | 0.075348 |
| `accum=0.0015` | 30.8945 | 0.9135 | 0.1773 | 0.0572 | 0.159890 | 0.053131 |
| `accum=0.0025` | 30.9945 | 0.9143 | 0.1763 | 0.0441 | 0.174141 | 0.064058 |

Decision: `accum=0.0015` has strongest common clear leakage control among seed-42 narrow points, but LPIPS is worse. `accum=0.0025` gives better PSNR/SSIM and J-blue, with moderate leakage. Neither clears the M1-relative full gate.

## Phase 2: Ownership Ablation

Seed 42, `accum=0.002`.

| Ownership | PSNR | SSIM | LPIPS | J blue | common far accum | common far clear |
|---|---:|---:|---:|---:|---:|---:|
| `alpha_only` | 31.0093 | 0.9128 | 0.1748 | 0.0628 | 0.231339 | 0.084730 |
| `alpha_depth` | 30.9408 | 0.9127 | 0.1791 | 0.0538 | 0.147092 | 0.053148 |
| `alpha_depth_color` | 31.1162 | 0.9143 | 0.1760 | 0.0867 | 0.230851 | 0.070352 |

Decision: `alpha_depth` gives strongest leakage control but worse LPIPS. `alpha_depth_color` improves PSNR/SSIM vs this seeded `alpha_depth` run but gives weaker leakage and higher J-blue dominance. Keep `alpha_depth` as default ownership for capacity experiments; do not promote `alpha_depth_color` yet.

## Phase 2: Occupancy Gate

| Experiment | PSNR | SSIM | LPIPS | J blue | common far accum | common far clear |
|---|---:|---:|---:|---:|---:|---:|
| `occupancy_limited=False` | 31.0025 | 0.9124 | 0.1760 | 0.0462 | 0.245343 | 0.060265 |

Decision: disabling occupancy-limited changes RGB composition but not loss support; common far accumulation rises vs the seeded `alpha_depth` ownership run. Keep `occupancy_limited=True`.

## Phase 2: Depth Evidence Softening

Seed 42, `accum=0.002`.

| ID | depth_mid | depth_temp | PSNR | SSIM | LPIPS | J blue | common far accum | common far clear |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0.75 | 0.10 | 31.1190 | 0.9147 | 0.1792 | 0.0606 | 0.146410 | 0.052388 |
| D1 | 0.80 | 0.10 | 30.9951 | 0.9138 | 0.1764 | 0.0922 | 0.224650 | 0.064281 |
| D2 | 0.75 | 0.15 | 31.2212 | 0.9145 | 0.1758 | 0.0551 | 0.199784 | 0.070869 |
| D3 | 0.80 | 0.15 | 30.8895 | 0.9131 | 0.1773 | 0.0350 | 0.199242 | 0.065207 |

Decision: D2 is the best reconstruction-quality point and close to first-stage E2 PSNR, but it does not improve common-mask leakage. D0 is strongest for common far clear leakage among depth-softening runs, but LPIPS is too weak. D2 is the best practical depth-softening candidate for follow-up repeat; D0 is useful as leakage-control reference.

## Phase 3: Hit-Aware Depth Diagnostics

Hit confidence definition:

```text
q_alpha = sigmoid((A - tau_A) / t_A)
q_conc = exp(-relative_depth_std / kappa)
q_hit = q_alpha * q_conc
```

Default parameters used:

```text
tau_A=0.20
t_A=0.05
kappa=0.20
```

Diagnostics generated at:

```text
logs/diagnostics/common_m1_q90/m2_e2_accum0p002_hit_diag
renders/m2_phase2_hit_diag_view0000_contact_20260724.png
renders/m2_phase2_hitaware_summary_view0000_20260724.png
```

Visual judgment: `q_hit` is high on visible sea-floor/object surfaces and low over open water/low-accumulation regions. It is not degenerate, so the diagnostic implementation is usable. However, it is still strongly tied to existing accumulation, so capacity gating can easily relax pressure exactly where common-mask far leakage is already high.

## Phase 4: Hit-Aware Capacity Support

Seed 42, `accum=0.002`.

| Support mode | PSNR | SSIM | LPIPS | J blue | common far accum | common far clear | far hit mean | far capacity mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 `m_inf` | 31.0151 | 0.9123 | 0.1752 | 0.0623 | 0.200738 | 0.064587 | 0.188325 | 0.708708 |
| H1 `m_inf*(1-q_alpha)` | 30.9388 | 0.9127 | 0.1755 | 0.1392 | 0.315076 | 0.095148 | 0.193637 | 0.512700 |
| H2 `m_inf*(1-q_hit)` | 30.9496 | 0.9129 | 0.1759 | 0.0799 | 0.252780 | 0.070761 | 0.248513 | 0.455317 |
| H3 `m_inf*(1-q_hit)^2` | 31.1083 | 0.9123 | 0.1768 | 0.0639 | 0.246008 | 0.073644 | 0.201906 | 0.546166 |

Decision: H2/H3 do not pass the Phase 4 success rule. Compared with H0, hit-aware gating relaxes capacity pressure and common far leakage rebounds. H3 recovers PSNR but worsens LPIPS and leakage. Do not promote hit-aware capacity support yet.

## Current Best Interpretation

1. The fixed common-mask diagnostic changes the conclusion materially: first-stage E2 is still useful, but its far-water cleanup is not as strong as per-model q90 diagnostics implied.
2. Seed stability is weaker than expected. `accum=0.002` reduces leakage more than `0.0005`, but both have PSNR variance above the target.
3. Depth evidence is still useful; `alpha_only` does not control far leakage enough.
4. `alpha_depth_color` is not a safe default because it gives weaker leakage and higher J-blue dominance in this scene.
5. Hit confidence itself is visually meaningful, but directly using `1-q_hit` to reduce accumulation pressure causes leakage rebound. The next useful version should not simply relax pressure wherever `q_hit` is high; it needs either an object-mask retention metric, a more conservative hit threshold, or a split support/capacity design with an explicit minimum capacity floor.

## Recommended Next Step

Do not proceed to closed-tail rendering or pseudo-depth teacher training yet. First stabilize the Phase 2 base:

1. Repeat D2 (`depth_mid=0.75`, `depth_temp=0.15`, `accum=0.002`) for seeds 123 and 3407.
2. Repeat D0 (`depth_mid=0.75`, `depth_temp=0.10`, `accum=0.002`) for seeds 123 and 3407 only if leakage control is prioritized over LPIPS.
3. Add a conservative hit-aware capacity floor before retrying H2:

```text
m_capacity = m_inf * max(capacity_floor, 1 - q_hit)
capacity_floor in {0.25, 0.50}
```

4. Add object-retention diagnostics before any further hit-aware training decision. Current common far mask is insufficient to prove true object protection.
