# MCMC-WS Backbone Experiments

Date: 2026-08-03

Branch: `backbone/mcmc-water-splatting`

## Motivation

GDADC is stopped at the current result. It changed split/clone allocation and final Gaussian count but did not produce stable metric gains on JapaneseGradens. This suggests the next test should not be another candidate-score tweak inside the original ADC framework.

MCMC-WS keeps the M1 rendering setup unchanged:

- `medium_context_mode=dir_xy_camera`
- `b_inf_mode=tied`
- medium field, underwater attenuation/backscatter renderer, SH appearance, RGB+SSIM loss

MCMC-WS replaces ADC split/duplicate/cull/opacity reset during the enabled window with MCMC-style Gaussian relocation, controlled birth, optional opacity/scale regularization, and optional SGLD position noise.

## Code Changes

- Added `water_splatting/density_control/mcmc_relocation.py`.
- Added `water_splatting/density_control/mcmc_diagnostics.py`.
- Added `scripts/diagnostics/test_mcmc_relocation.py`.
- Added MCMC config flags to `WaterSplattingModelConfig`.
- Added MCMC state transition in `WaterSplattingModel.refinement_after()`.
- Added optional SGLD in `WaterSplattingModel.after_train()`.
- Added MCMC log writing to JSONL.
- Added MCMC flag plumbing to `scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh` and `scripts/experiments/igaf_5k_common.sh`.
- Added JapaneseGradens seed-500 replay scripts:
  - `scripts/experiments/mcmc_ws_m0_adc_japanesegradens_5k.sh`
  - `scripts/experiments/mcmc_ws_m1_reloc_birth_japanesegradens_5k.sh`
  - `scripts/experiments/mcmc_ws_m2_reg_japanesegradens_5k.sh`
  - `scripts/experiments/mcmc_ws_m3_sgld_japanesegradens_5k.sh`

## Config Flags

```python
mcmc_enabled: bool = False
mcmc_cap_max: int = -1
mcmc_start_step: int = 500
mcmc_stop_step: int = 10000
mcmc_interval: int = 100
mcmc_dead_opacity_threshold: float = 0.005
mcmc_growth_rate: float = 0.05
mcmc_sgld_enabled: bool = True
mcmc_noise_scale: float = 1.0
mcmc_noise_lr: float = 1.6e-4
mcmc_noise_opacity_mid: float = 0.995
mcmc_noise_opacity_temperature: float = 0.01
lambda_mcmc_opacity: float = 0.0
lambda_mcmc_scale: float = 0.0
mcmc_log_path: Optional[str] = None
```

## Relocation Formula

The pure-Torch implementation ports the official 3DGS-MCMC CUDA relocation formula.

For a parent with original opacity `alpha` assigned `N - 1` children:

```text
alpha_new = 1 - (1 - alpha) ** (1 / N)
```

The scale coefficient uses the official denominator:

```text
denom = sum_{i=1..N} sum_{k=0..i-1} C(i-1,k) * (-1)^k * alpha_new^(k+1) / sqrt(k+1)
scale_new = scale_old * alpha / denom
```

This preserves the 1D projected alpha mass used by the official kernel, not the full pointwise footprint.

## Validation

Commands:

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/water_splatting.py \
  water_splatting/density_control/mcmc_relocation.py \
  water_splatting/density_control/mcmc_diagnostics.py \
  scripts/diagnostics/test_mcmc_relocation.py

/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/test_mcmc_relocation.py

git diff --check
```

Results:

- `py_compile`: pass
- relocation mass preservation cases: 84
- max mean absolute mass error: `6.56e-07`
- max absolute mass error: `6.56e-07`
- Adam row reset check: pass
- `git diff --check`: pass

## Smoke Tests

Source checkpoint:

```text
outputs/gdadc_gdadc_seed500_m1_japanesegradens_seed42_500/water-splatting/gdadc_gdadc_seed500_m1_japanesegradens_seed42_500_20260803_gdadc_seed500_jg/nerfstudio_models_adam_sanitized/step-000000499.ckpt
```

### M1 relocation/birth smoke

Command:

```bash
GPU=6 RUN_EVAL=0 RUN_CLOSURE_DIAG=0 MAX_NUM_ITERATIONS=120 MODEL_NUM_STEPS=620 \
STEPS_PER_SAVE=120 SAVE_ONLY_LATEST_CHECKPOINT=True \
EXPERIMENT_NAME=mcmc_ws_smoke_m1_transition_japanesegradens_seed42_620 \
STAMP=20260803_smoke_transition MCMC_INTERVAL=100 MCMC_STOP_STEP=700 \
scripts/experiments/mcmc_ws_m1_reloc_birth_japanesegradens_5k.sh
```

Result:

- Training started from shared step-499 checkpoint and finished without CUDA/autograd/optimizer errors.
- MCMC log wrote one transition at step 600.
- Gaussian count: `21140 -> 22197`
- Dead relocated: `246`
- Newborn: `1057`
- Parent unique count: `1236`

### M3 SGLD smoke

Command:

```bash
GPU=6 RUN_EVAL=0 RUN_CLOSURE_DIAG=0 MAX_NUM_ITERATIONS=120 MODEL_NUM_STEPS=620 \
STEPS_PER_SAVE=120 SAVE_ONLY_LATEST_CHECKPOINT=True \
EXPERIMENT_NAME=mcmc_ws_smoke_m3_sgld_japanesegradens_seed42_620 \
STAMP=20260803_smoke_m3 MCMC_INTERVAL=100 MCMC_STOP_STEP=700 \
scripts/experiments/mcmc_ws_m3_sgld_japanesegradens_5k.sh
```

Result:

- Training finished without CUDA/autograd/optimizer errors.
- MCMC log wrote SGLD events at steps 500 and 600, plus one transition at step 600.
- Step-600 Gaussian count: `21140 -> 22197`
- Dead relocated: `251`
- Newborn: `1057`
- SGLD noise ratio p95 at step 600: `5.68e-05`

## 5k Experiment Matrix

JapaneseGradens first. IUI3 and 15k are gated on JapaneseGradens.

| Variant | Description | Status | PSNR | SSIM | LPIPS | Gaussian Count | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| M0 | M1 control with original ADC | pending | | | | | |
| M1 | MCMC relocation + controlled birth, no reg, no SGLD | pending | | | | | |
| M2 | M1 + opacity/scale reg at 0.001 | pending | | | | | |
| M3 | M2 + SGLD | pending | | | | | |

Gate:

- JapaneseGradens vs same-step M0: PSNR +0.08 dB, LPIPS -0.0015, SSIM non-decrease.
- Gaussian count increase <= 15%.
- Training time increase <= 15%.
- No obvious floaters.

## Current Decision

Implementation smoke is passed. Proceed to JapaneseGradens M0-M3 5k screening from shared step-500 checkpoint.
