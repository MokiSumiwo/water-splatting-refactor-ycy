# M1 + Bounded SH3 Clean Branch Audit

Date: 2026-08-08

## Code Fact: Snapshot Preservation

Original historical research branch:

- `research/gmvc-medium-calibration`

Original research HEAD at snapshot time:

- `dd1a63e56ff00b1c6d3309cc02d1505c6a6393c1`
- Commit title: `Record bounded SH3 cross-scene conclusion`

Archive snapshot:

- Branch: `archive/gmvc-dewatering-full-20260808`
- Tag: `research-snapshot-20260808-gmvc-dewatering-full`
- Target commit: `dd1a63e56ff00b1c6d3309cc02d1505c6a6393c1`
- Remote push verified for archive branch and tag.

The full GMVC, D010, SeaFree-inspired, background-supervision, and bounded-SH3 research implementation is preserved on the archive branch and the original historical research branch.

## Code Fact: M1 Base Identification

Selected M1 base commit:

- `0de407bb0210aa6bfd356cbbab6978a2fc94e58d`
- Commit title: `Restore M1 code baseline`

Candidate notes:

- `d752f89` / `69abf5f` record and run the formal cross-scene M1/P3 experiments, including the M1 settings later used as the formal baseline.
- `0de407b` is the last explicit restore point before the GMVC/D010/SeaFree/BND sequence and is the best behavior anchor for rebuilding the active branch from the formal M1 code path.
- `0de407b` still contained many default-off historical experiment fields and scripts. The clean branch therefore starts from `0de407b` for behavior compatibility, then actively removes default-off historical residue.

Formal M1 settings preserved by the clean branch runners:

- `medium_context_mode=dir_xy_camera`
- `b_inf_mode=tied`
- `infinite_water_enabled=False`
- `sh_degree=3`
- Original WaterSplatting RGB reconstruction loss.
- Original direct attenuation form: `T_D = exp(-beta_D * d)`.
- Original backscatter exponent and rasterizer composition.
- Original optimizer, densification, pruning, opacity reset, and SH schedule.

## Code Fact: Active Clean Model Definition

Active branch:

- `research/m1-bounded-intrinsic`
- Start commit: `0de407bb0210aa6bfd356cbbab6978a2fc94e58d`

Active model:

- M1 Gaussian geometry.
- M1 direction-conditioned medium field.
- M1 underwater rasterizer physics.
- M1 RGB/SSIM reconstruction objective.
- M1 optimizer, densification, pruning, and opacity behavior.
- Full SH degree 3 appearance capacity.

Only new active mechanism:

```text
c_i(v) = sigmoid(s_i(v))
```

where `s_i(v)` is the current-view full active SH3 evaluation interpreted as RGB logits. This bounded color is used in the underwater forward render, the direct object signal, and the clear-object render.

Legacy M1 intrinsic remains available as an ablation:

```text
intrinsic_color_parameterization=legacy
```

Bounded SH3 is selected with:

```text
intrinsic_color_parameterization=bounded_sh3
```

## Code Fact: Excluded Historical Modules

The clean branch active training graph excludes:

- GMVC profile calibration and GMVC object auxiliary losses.
- GMVC alternating / medium-hold schedules.
- D010 direct optical-depth scale and gamma sweeps.
- Background-supervision / BGI / BG010 losses.
- Foreground-aware weighting.
- Pseudo-depth / DepthAnything / coarse depth losses.
- Soft intrinsic bound losses.
- SeaFree-factor training logic.
- Infinite-water ownership branches.
- Dual-color intrinsic experiments.

`infinite_water_enabled` is kept only as a config-compatibility field and raises at runtime if enabled.

## Code Fact: Clean Branch Files

Modified core files:

- `water_splatting/water_splatting.py`: minimized config, BND initialization, legacy/BND appearance dispatch, M1 medium prediction, core diagnostics.
- `water_splatting/fields/gaussian_appearance.py`: legacy SH3 color helper plus bounded full-SH logits helper.
- `water_splatting/fields/medium_field.py`: restricted M1 medium context and tied/implicit `B_inf` support.
- `water_splatting/fields/__init__.py`: exports minimal field helpers.
- `water_splatting/rendering/underwater_rasterizer.py`: keeps original rasterizer wrapper and removes historical clear-proxy helper.

Added minimal scripts:

- `scripts/experiments/m1_legacy_15k.sh`
- `scripts/experiments/m1_bounded_intrinsic_15k.sh`
- `scripts/diagnostics/test_bounded_sh3_initialization.py`

Kept BND research notes:

- `research_notes/BOUNDED_SH3_SCRATCH_EXPERIMENT_2026-08-08.md`
- `research_notes/BOUNDED_SH3_CROSS_SCENE_VALIDATION_2026-08-08.md`
- `research_notes/M1_BND_CLEAN_BRANCH_AUDIT_2026-08-08.md`

## Experimental Fact: Legacy Forward Equivalence

Checkpoint:

```text
outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000014999.ckpt
```

Config:

```text
outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml
```

View:

- Curasao first eval view.
- `image_idx=0`.

Compared:

- Archive branch `archive/gmvc-dewatering-full-20260808`.
- Clean branch `research/m1-bounded-intrinsic`.
- Same checkpoint, camera, resolution, and eval batch.

Common tensor differences:

| Output | Mean abs diff | P95 abs diff | Max abs diff |
| --- | ---: | ---: | ---: |
| `J_gaussian_raw` | 0 | 0 | 0 |
| `accumulation` | 0 | 0 | 0 |
| `depth` | 0 | 0 | 0 |
| `medium_attn` | 0 | 0 | 0 |
| `medium_bs` | 0 | 0 | 0 |
| `medium_rgb` | 0 | 0 | 0 |
| `pred_image` | 4.82295536841e-09 | 2.98023223877e-08 | 1.19209289551e-07 |
| `rgb` | 4.82295536841e-09 | 2.98023223877e-08 | 1.19209289551e-07 |
| `rgb_clear` | 0 | 0 | 0 |
| `rgb_clear_clamp` | 0 | 0 | 0 |
| `rgb_medium` | 2.18723483769e-10 | 0 | 2.98023223877e-08 |
| `rgb_object` | 0 | 0 | 0 |

RGB metrics:

| Branch | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| archive | 27.181760787963867 | 0.9376756548881531 | 0.15667593479156494 |
| clean | 27.181760787963867 | 0.9376756548881531 | 0.15667590498924255 |

Legacy equivalence result: PASS.

## Experimental Fact: Bounded SH3 Forward Equivalence

Checkpoint:

```text
outputs/dewater_bounded_sh3_scratch_20260808/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000/water-splatting/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_bnd-scratch_g1p00/nerfstudio_models/step-000014999.ckpt
```

Config:

```text
outputs/dewater_bounded_sh3_scratch_20260808/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000/water-splatting/dewater_bounded_sh3_curasao_seed42_bnd-scratch_step0_to_15000_20260808_bounded_sh3_scratch_full_bnd-scratch_g1p00/config.yml
```

View:

- Curasao first eval view.
- `image_idx=0`.

Schema note:

- Historical branch parameter name/value: `intrinsic_color_parameterization=sigmoid_sh`.
- Clean branch parameter name/value: `intrinsic_color_parameterization=bounded_sh3`.
- The audit process mapped only this config value in memory. No `sigmoid_sh` alias was reintroduced into clean active config.

Common tensor differences:

| Output | Mean abs diff | P95 abs diff | Max abs diff |
| --- | ---: | ---: | ---: |
| `J_gaussian` | 0 | 0 | 0 |
| `J_gaussian_raw` | 0 | 0 | 0 |
| `accumulation` | 0 | 0 | 0 |
| `depth` | 0 | 0 | 0 |
| `gaussian_sigmoid_derivative` | 0 | 0 | 0 |
| `gaussian_view_logits` | 0 | 0 | 0 |
| `gaussian_view_rgb` | 0 | 0 | 0 |
| `medium_attn` | 0 | 0 | 0 |
| `medium_bs` | 0 | 0 | 0 |
| `medium_rgb` | 0 | 0 | 0 |
| `pred_image` | 4.48517445406e-09 | 2.98023223877e-08 | 5.96046447754e-08 |
| `rgb` | 4.48517445406e-09 | 2.98023223877e-08 | 5.96046447754e-08 |
| `rgb_clear` | 0 | 0 | 0 |
| `rgb_clear_clamp` | 0 | 0 | 0 |
| `rgb_medium` | 2.31669364248e-10 | 0 | 2.98023223877e-08 |
| `rgb_object` | 0 | 0 | 0 |

RGB metrics:

| Branch | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| archive | 27.997447967529297 | 0.9450036287307739 | 0.15617376565933228 |
| clean | 27.997447967529297 | 0.9450036287307739 | 0.15617406368255615 |

Bounded SH3 forward equivalence result: PASS.

## Experimental Fact: Bounded Initialization Equivalence

Command:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/test_bounded_sh3_initialization.py --seed 42 --count 100000 --eps 1e-7
```

Result:

| Statistic | Value |
| --- | ---: |
| mean_abs_error | 1.1623210838251907e-08 |
| p95_abs_error | 5.960464477539063e-08 |
| max_abs_error | 1.1920928955078125e-07 |
| bounded_rgb_min | 1.000000082740371e-07 |
| bounded_rgb_max | 0.9999998807907104 |
| all finite | true |
| strictly inside `(0,1)` | true |

Bounded initialization equivalence result: PASS.

## Code Fact: Active-Code Residue Audit

Search scope:

- `water_splatting`
- `scripts/experiments/m1_legacy_15k.sh`
- `scripts/experiments/m1_bounded_intrinsic_15k.sh`
- `scripts/diagnostics/test_bounded_sh3_initialization.py`

Patterns checked:

- `gmvc|GMVC`
- `direct_optical_depth_scale|D010|gamma_D`
- `BGI|BG010|background_supervision`
- `FAW|foreground.*weight`
- `intrinsic_bound|IB-G`
- `DepthAnything|coarse.*depth|pseudo_depth|pseudo-depth`

Active-code residue result:

- No matches in the active clean Python/shell files listed above.

Allowed compatibility residues:

- `infinite_water_enabled` exists only as a boolean compatibility field and raises if enabled.
- Generic words such as `bounded`, `bound`, and `depth` remain in normal geometry/rasterizer terminology and BND naming; they are not historical loss branches.

Untracked historical scripts still present in the shared worktree:

- `scripts/diagnostics/render_gmvc_curasao_contact_sheet.py`
- `scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py`

These two files are not modified and are not part of the clean branch commit.

## Code Fact: Output Policy

No training outputs, renders, logs, masks, checkpoints, PNG/JPG files, CSV/JSON experimental outputs, model weights, videos, or large binary artifacts are tracked in the clean branch.

Tracked large-output check:

```text
git ls-files outputs renders logs common_masks checkpoints | wc -l
0
```

## Validation Status

Validation commands run on the clean branch:

- Python compilation:
  - `water_splatting/water_splatting.py`
  - `water_splatting/fields/gaussian_appearance.py`
  - `water_splatting/fields/medium_field.py`
  - `water_splatting/fields/__init__.py`
  - `water_splatting/rendering/underwater_rasterizer.py`
  - `scripts/diagnostics/test_bounded_sh3_initialization.py`
- Shell syntax:
  - `scripts/experiments/m1_legacy_15k.sh`
  - `scripts/experiments/m1_bounded_intrinsic_15k.sh`
- `git diff --check`
- tracked output check:
  - `git ls-files outputs renders logs common_masks checkpoints | wc -l`

Validation result:

- Python compilation: PASS.
- Shell syntax: PASS.
- `git diff --check`: PASS.
- tracked output check: PASS, value `0`.

Smoke check:

- Loaded the existing Curasao M1 15k config/checkpoint on the clean branch.
- `intrinsic_color_parameterization=legacy`.
- `get_param_groups()` returned the expected groups: `direction_encoding`, `features_dc`, `features_rest`, `means`, `medium_mlp`, `opacities`, `quats`, `scales`.
- First eval view returned required diagnostic tensors:
  - `pred_image`: `(1188, 1794, 3)`
  - `direct_object_signal`: `(1188, 1794, 3)`
  - `clear_object_fullsh_raw`: `(1188, 1794, 3)`
  - `transmission`: `(1188, 1794, 3)`
  - `tau_D`: `(1188, 1794, 3)`

## Recommended Baseline

Active research baseline:

```text
M1 + Bounded SH3
```

Ablation baseline:

```text
M1 + Legacy SH3
```

Historical experimental implementation:

```text
archive/gmvc-dewatering-full-20260808
research/gmvc-medium-calibration
```
