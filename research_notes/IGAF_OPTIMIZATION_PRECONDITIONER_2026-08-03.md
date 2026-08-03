# IGAF Optimization Preconditioner Follow-up

Date: 2026-08-03  
Branch: `refactor/core-framework`  
Base commit at start: `3504174 Add IGAF intra-Gaussian frequency field`

## Objective

This follow-up tests whether IGAF's positive 5k signal is a direct final representation gain or a training-trajectory / densification preconditioner effect. The key experiments are:

1. Full-vs-off counterfactual eval on trained IGAF 5k checkpoints.
2. Shared step-2500 replay from a single M1 checkpoint.
3. Inference-off replay to test an IGAF-P style temporary training branch.

## Code Changes

- Added IGAF replay and counterfactual tooling:
  - `scripts/diagnostics/eval_igaf_full_vs_off.py`
  - `scripts/diagnostics/sanitize_adam_checkpoint.py`
  - `scripts/experiments/igaf_phasea_full_vs_off_existing_5k.sh`
  - `scripts/experiments/igaf_phaseb_seed2500.sh`
  - `scripts/experiments/igaf_phaseb_replay_variant.sh`
  - `scripts/experiments/igaf_g7_variance_mip_locked_japanesegradens_5k.sh`
  - `scripts/experiments/igaf_g7_variance_mip_locked_iui3_5k.sh`
- Added IGAF controls in `WaterSplattingModelConfig`:
  - `igaf_stop_step`
  - `igaf_ramp_down_steps`
  - `igaf_inference_enabled`
  - `igaf_axis_mode`
  - `igaf_mip_mode`
  - `igaf_reset_split_coeffs`
- Fixed split/cull compatibility:
  - split children reset `igaf_coeffs` when `igaf_reset_split_coeffs=True`
  - duplicate children inherit `igaf_coeffs`
  - `igaf_axis_order` is synced through split/duplicate/cull
- Added corrected Mip option:
  - `igaf_mip_mode=variance`
  - per-basis attenuation uses pixel variance factor `1/12`
  - CUDA gate shape changed from scalar to `(N, 5)`: output gate + four basis gates
- Added diagnostics:
  - `axis_permutation_change_fraction`
  - `tangent_scale_ratio`
  - `base_gate`, `output_gate`, `mip_gate`
  - `effective_local_rgb_abs`
- Added checkpoint sanitizer:
  - removes empty Adam states for inactive param groups, mainly `igaf_coeffs`
  - preserves model, optimizer, scheduler, and scaler states for replay loading

## Commands

Validation:

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/water_splatting.py \
  water_splatting/rasterize.py \
  scripts/diagnostics/eval_igaf_full_vs_off.py \
  scripts/diagnostics/sanitize_adam_checkpoint.py

bash -n \
  scripts/experiments/igaf_phasea_full_vs_off_existing_5k.sh \
  scripts/experiments/igaf_phaseb_seed2500.sh \
  scripts/experiments/igaf_phaseb_replay_variant.sh \
  scripts/experiments/igaf_g7_variance_mip_locked_japanesegradens_5k.sh \
  scripts/experiments/igaf_g7_variance_mip_locked_iui3_5k.sh

git diff --check
git ls-files outputs renders logs common_masks | wc -l
```

Phase A:

```bash
GPU=6 STAMP=20260803_phasea_full_vs_off \
  scripts/experiments/igaf_phasea_full_vs_off_existing_5k.sh
```

Phase B seed:

```bash
GPU=6 SCENE_SLUG=japanesegradens STAMP=20260803_phaseb_seed2500_jg \
  scripts/experiments/igaf_phaseb_seed2500.sh
```

Phase B replay:

```bash
LOAD_DIR=outputs/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500/water-splatting/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500_20260803_phaseb_seed2500_jg/nerfstudio_models \
GPU=6 SCENE_SLUG=japanesegradens VARIANT=r0_m1 STAMP=20260803_phaseb_jg_r0_fixed \
MAX_NUM_ITERATIONS=2500 RUN_EVAL=1 scripts/experiments/igaf_phaseb_replay_variant.sh

LOAD_DIR=outputs/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500/water-splatting/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500_20260803_phaseb_seed2500_jg/nerfstudio_models \
GPU=7 SCENE_SLUG=japanesegradens VARIANT=r1_amp0 STAMP=20260803_phaseb_jg_r1_fixed \
MAX_NUM_ITERATIONS=2500 RUN_EVAL=1 scripts/experiments/igaf_phaseb_replay_variant.sh

LOAD_DIR=outputs/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500/water-splatting/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500_20260803_phaseb_seed2500_jg/nerfstudio_models \
GPU=8 SCENE_SLUG=japanesegradens VARIANT=r2_nomip STAMP=20260803_phaseb_jg_r2_fixed \
MAX_NUM_ITERATIONS=2500 RUN_EVAL=1 scripts/experiments/igaf_phaseb_replay_variant.sh

LOAD_DIR=outputs/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500/water-splatting/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500_20260803_phaseb_seed2500_jg/nerfstudio_models \
GPU=9 SCENE_SLUG=japanesegradens VARIANT=r3_variance_mip STAMP=20260803_phaseb_jg_r3_fixed \
MAX_NUM_ITERATIONS=2500 RUN_EVAL=1 scripts/experiments/igaf_phaseb_replay_variant.sh
```

IGAF-P inference-off replay:

```bash
LOAD_DIR=outputs/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500/water-splatting/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500_20260803_phaseb_seed2500_jg/nerfstudio_models \
GPU=8 SCENE_SLUG=japanesegradens VARIANT=r2_nomip STAMP=20260803_phaseb_jg_p1_inferoff \
MAX_NUM_ITERATIONS=2500 RUN_EVAL=1 IGAF_INFERENCE_ENABLED=False \
EXPERIMENT_TAG=phaseb_p1_nomip_inferoff scripts/experiments/igaf_phaseb_replay_variant.sh

LOAD_DIR=outputs/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500/water-splatting/igaf_phaseb_seed2500_m1_japanesegradens_seed42_2500_20260803_phaseb_seed2500_jg/nerfstudio_models \
GPU=9 SCENE_SLUG=japanesegradens VARIANT=r3_variance_mip STAMP=20260803_phaseb_jg_p2_inferoff \
MAX_NUM_ITERATIONS=2500 RUN_EVAL=1 IGAF_INFERENCE_ENABLED=False \
EXPERIMENT_TAG=phaseb_p2_variance_mip_inferoff scripts/experiments/igaf_phaseb_replay_variant.sh
```

## Phase A Results

Full-vs-off eval shows most of the apparent IGAF gain remains when IGAF is disabled at eval time.

| Scene | Checkpoint | Eval | PSNR | SSIM | LPIPS | dPSNR vs G0 | dSSIM vs G0 | dLPIPS vs G0 | PSNR retention | LPIPS retention |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| JapaneseGradens | G1 no-Mip | Full | 22.9792 | 0.80218 | 0.23490 | +0.0333 | +0.00038 | -0.00334 | 0.966 | 0.959 |
| JapaneseGradens | G1 no-Mip | Off | 22.9781 | 0.80208 | 0.23503 | +0.0321 | +0.00029 | -0.00320 | 0.966 | 0.959 |
| JapaneseGradens | G2 legacy Mip | Full | 23.0397 | 0.80230 | 0.23687 | +0.0938 | +0.00051 | -0.00137 | 0.989 | 0.890 |
| JapaneseGradens | G2 legacy Mip | Off | 23.0387 | 0.80222 | 0.23702 | +0.0927 | +0.00043 | -0.00122 | 0.989 | 0.890 |
| IUI3 | G1 no-Mip | Full | 25.8799 | 0.76579 | 0.30376 | +0.2063 | +0.00161 | -0.00419 | 1.000 | 0.950 |
| IUI3 | G1 no-Mip | Off | 25.8800 | 0.76584 | 0.30397 | +0.2064 | +0.00166 | -0.00398 | 1.000 | 0.950 |
| IUI3 | G2 legacy Mip | Full | 25.8613 | 0.76541 | 0.30368 | +0.1877 | +0.00123 | -0.00427 | 0.994 | 0.945 |
| IUI3 | G2 legacy Mip | Off | 25.8603 | 0.76533 | 0.30391 | +0.1866 | +0.00116 | -0.00404 | 0.994 | 0.945 |

Interpretation: IGAF is not acting mainly as final local texture capacity. The stronger hypothesis is a training trajectory / optimization preconditioner effect.

## Phase B Results: Shared Step-2500 Replay

All runs load the same M1 step-2499 checkpoint and continue to step 4999 on JapaneseGradens.

| Run | Setting | PSNR | SSIM | LPIPS | dPSNR vs R0 | dSSIM vs R0 | dLPIPS vs R0 | Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | M1 continuation | 23.1069 | 0.80174 | 0.24411 | 0.0000 | 0.00000 | 0.00000 | 713,622 |
| R1 | IGAF path, amplitude 0 | 23.0554 | 0.80195 | 0.24410 | -0.0515 | +0.00021 | -0.00001 | 712,560 |
| R2 | no-Mip IGAF | 23.0771 | 0.80165 | 0.24480 | -0.0298 | -0.00009 | +0.00070 | 712,441 |
| R3 | corrected variance Mip + locked axis | 23.0708 | 0.80242 | 0.24289 | -0.0361 | +0.00068 | -0.00122 | 713,222 |

R0/R1 deterministic audit:

- R0 and R1 match at step 2500 for loss, gradient stats, and refinement.
- First divergence occurs at step 2600 loss:
  - R0 `main_loss=0.1437698305`
  - R1 `main_loss=0.1438111365`
- By step 4900, refinement masks differ:
  - R0 high-grad hash `c8427b907246c2abff80226806b17ef3c5827267`
  - R1 high-grad hash `084f22981c73f38d261258e184051561871d78a0`

This means the amplitude-zero IGAF CUDA path is not a deterministic no-op. R2/R3 cannot be interpreted as cleanly isolated IGAF representation effects without accounting for this codepath perturbation.

## IGAF-P Inference-Off Replay

These runs train with IGAF active from the same shared checkpoint but disable IGAF at eval.

| Run | Setting | PSNR | SSIM | LPIPS | dPSNR vs R0 | dSSIM vs R0 | dLPIPS vs R0 | Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | no-Mip train, inference off | 23.0487 | 0.80156 | 0.24345 | -0.0581 | -0.00018 | -0.00065 | 714,241 |
| P2 | corrected Mip train, inference off | 22.9576 | 0.80100 | 0.24171 | -0.1493 | -0.00075 | -0.00240 | 713,994 |

P2 improves LPIPS but has a large PSNR penalty and lower SSIM. This does not pass the JapaneseGradens pilot gate.

## IGAF Diagnostics

Latest logged IGAF stats at step 4500:

| Run | Gate Mean | Mip Mean | Effective Local RGB Abs Mean | Axis Change Fraction |
| --- | ---: | ---: | ---: | ---: |
| R2 no-Mip | 0.6715 | 1.0000 | 0.000363 | 0.00073 |
| R3 variance Mip | 0.2102 | 0.2486 | 0.000356 | 0.19082 |
| P1 no-Mip inference-off | 0.6711 | 1.0000 | 0.000362 | 0.00079 |
| P2 variance Mip inference-off | 0.2097 | 0.2478 | 0.000357 | 0.19088 |

The corrected Mip is much less destructive than the legacy Mip gate near 0.01, but it still materially attenuates the branch. Locked axis diagnostics reveal a high fraction of dynamic axis-order disagreement, so axis stability matters and may itself alter trajectory.

## Decision

Do not enter JapaneseGradens 15k pilot from this IGAF configuration.

Reasons:

- Phase A supports an optimization trajectory effect, not direct final representation capacity.
- Shared-checkpoint Phase B did not reproduce the original from-scratch 5k JapaneseGradens PSNR gain.
- R1 amplitude-zero branch diverges from R0 at step 2600, so codepath perturbation is not negligible.
- R3 improves LPIPS by 0.00122 vs R0 and SSIM by 0.00068, but PSNR drops by 0.036 dB.
- P2 inference-off improves LPIPS by 0.00240, but PSNR drops by 0.149 dB and SSIM drops by 0.00075.
- The JapaneseGradens pilot gate required roughly `dPSNR >= +0.08 dB` and `dLPIPS <= -0.002`; no shared-checkpoint replay satisfies both.

IUI3 Phase B replay was not run because the JapaneseGradens gate failed. This follows the planned sequence: JapaneseGradens mechanism first, then IUI3 safety only after a positive JapaneseGradens shared-checkpoint result.

## Next Recommendation

Stop IGAF 15k escalation for now.

If IGAF is revisited, the next useful experiment is not another amplitude/frequency sweep. It should isolate densification explicitly:

- add a base-render densification-gradient path for IGAF-P
- use IGAF render for RGB loss
- use base render for `xys_grad_norm`, split, duplicate, and cull candidate statistics

That P3 experiment is a larger model-logic change because current densification is driven directly by `self.xys.grad` from the main render. It was not added in this round to avoid mixing a new densification implementation with the replay attribution experiment.

If the project needs a stronger next module, shift away from small IGAF-style appearance plugins and evaluate a stronger 3DGS backbone change such as Pixel-GS, full Mip-Splatting, or 3DGS-MCMC.
