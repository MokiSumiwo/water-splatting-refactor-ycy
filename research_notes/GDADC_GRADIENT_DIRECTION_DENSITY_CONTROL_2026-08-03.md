# GDADC Gradient-Direction-Aware Density Control

Date: 2026-08-03

## Summary

This note closes the IGAF mainline with a small amplitude-zero hygiene fix and records the first GDADC experiment. GDADC was implemented as a densification-only candidate selector. It does not add any loss, residual supervision, pseudo depth, correspondence, renderer change, pruning rule, or CUDA change.

JapaneseGradens 5k gate failed. D2 split-score worsened PSNR/SSIM while only slightly improving LPIPS. D3 split+clone gave a small PSNR lift but materially hurt SSIM and LPIPS. Because JapaneseGradens failed the gate, IUI3 safety and 15k runs were not launched.

## Code Changes

- Archived the previous IGAF preconditioner state with tag `archive/igaf-preconditioner-c15d26f`.
- Fixed IGAF amplitude-zero routing in `water_splatting/rasterize.py`: `igaf_amplitude_max == 0` now forces the base rasterizer path rather than `rasterize_forward_igaf`.
- Added GDADC flags and buffers in `water_splatting/water_splatting.py`.
- Added GDADC signed/absolute gradient accumulation in `after_train()`.
- Modified `refinement_after()` so GDADC can replace only split candidates, or both split and clone candidates.
- Added JSONL diagnostics with gradient consistency, GDADC weights, candidate counts, threshold suggestions, scale/aspect/radius/depth stats, and applied split/duplicate counts.
- Added experiment wrappers:
  - `scripts/experiments/gdadc_phase_seed500.sh`
  - `scripts/experiments/gdadc_phase_replay_variant.sh`
- Extended M1 experiment wrappers with GDADC config plumbing while keeping MV-GAR, MCGR, GIVAR, IGAF, pseudo depth, cleanup, and capacity modules off by default.

## Config Flags

```python
gdadc_enabled: bool = False
gdadc_diagnostic_only: bool = True
gdadc_weight_base: float = 0.8
gdadc_weight_scale: float = 25.0
gdadc_weight_power: float = 15.0
gdadc_split_enabled: bool = True
gdadc_clone_enabled: bool = True
gdadc_split_grad_thresh: float = 0.0
gdadc_clone_grad_thresh: float = 0.0
gdadc_log_path: Optional[str] = None
```

For the 5k D2/D3 JapaneseGradens runs, `gdadc_split_grad_thresh=0.000305` was used. This value came from the first diagnostic refinement window and matched the baseline split count much better than the raw M1 `densify_grad_thresh=0.0008`.

## Mechanism

GDADC accumulates two screen-space gradient magnitudes per Gaussian over a refinement window:

- `signed_mean`: the usual signed screen-space gradient magnitude from `self.xys.grad`.
- `abs_mean`: the AbsGS-style absolute screen-space gradient magnitude from `self.xys_grad_abs`.

It then estimates direction consistency:

```text
consistency = clamp((signed_mean + eps) / (abs_mean + eps), 0, 1)
weight = 0.8 + 25.0 * (1 - consistency) ** 15
split_score = signed_mean * weight
clone_score = signed_mean / weight
```

Large Gaussians use `split_score`; small Gaussians use `clone_score`. Low-consistency large Gaussians are therefore more likely to split, while low-consistency small Gaussians are less likely to clone.

## Validation

Static checks:

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/water_splatting.py \
  water_splatting/rasterize.py \
  water_splatting/rendering/underwater_rasterizer.py \
  scripts/diagnostics/sanitize_adam_checkpoint.py

bash -n \
  scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh \
  scripts/experiments/igaf_5k_common.sh \
  scripts/experiments/gdadc_phase_seed500.sh \
  scripts/experiments/gdadc_phase_replay_variant.sh

git diff --check
git ls-files outputs renders logs common_masks | wc -l
```

Result: all checks passed, and tracked generated-artifact count was `0`.

IGAF amplitude-zero route test:

```text
amplitude_zero_calls = ["base"]
amplitude_nonzero_calls = ["igaf", "base"]
```

The nonzero case records both entries because the IGAF test stub internally reused the base stub for output construction.

## Smoke Runs

Shared JapaneseGradens step-500 seed:

```bash
GPU=6 SCENE_SLUG=japanesegradens STAMP=20260803_gdadc_seed500_jg RUN_EVAL=0 \
  bash scripts/experiments/gdadc_phase_seed500.sh
```

Checkpoint:

```text
outputs/gdadc_gdadc_seed500_m1_japanesegradens_seed42_500/water-splatting/gdadc_gdadc_seed500_m1_japanesegradens_seed42_500_20260803_gdadc_seed500_jg/nerfstudio_models/step-000000499.ckpt
```

Diagnostic-only smoke:

```bash
LOAD_DIR=/mnt/new/home_old/ycy/water-splatting-refactor/outputs/gdadc_gdadc_seed500_m1_japanesegradens_seed42_500/water-splatting/gdadc_gdadc_seed500_m1_japanesegradens_seed42_500_20260803_gdadc_seed500_jg/nerfstudio_models \
GPU=6 SCENE_SLUG=japanesegradens VARIANT=d1_diag STAMP=20260803_gdadc_d1_thresh_smoke \
MAX_NUM_ITERATIONS=300 RUN_EVAL=0 \
  bash scripts/experiments/gdadc_phase_replay_variant.sh
```

Smoke result at step 700:

| Metric | Value |
| --- | ---: |
| Large low-consistency ratio | 80.34% |
| Baseline split count | 16,963 |
| GDADC split candidates at 0.0008 | 14,626 |
| Suggested split threshold for baseline count | 0.000305 |
| Baseline clone count | 0 |

Note: D0 and D1 short replays were not bitwise identical, but repeated D0 was also not bitwise identical. The continuation path therefore has small non-GDADC replay variation. D1 was treated as a diagnostic behavior check, not as a module result.

## 5k Commands

All 5k runs used the shared step-500 sanitized checkpoint:

```text
outputs/gdadc_gdadc_seed500_m1_japanesegradens_seed42_500/water-splatting/gdadc_gdadc_seed500_m1_japanesegradens_seed42_500_20260803_gdadc_seed500_jg/nerfstudio_models_adam_sanitized
```

Commands:

```bash
LOAD_DIR=.../nerfstudio_models_adam_sanitized SANITIZE_LOAD_DIR=0 GPU=6 \
SCENE_SLUG=japanesegradens VARIANT=d0_m1 STAMP=20260803_gdadc_jg5k_v1 \
MAX_NUM_ITERATIONS=4500 RUN_EVAL=1 \
  bash scripts/experiments/gdadc_phase_replay_variant.sh

LOAD_DIR=.../nerfstudio_models_adam_sanitized SANITIZE_LOAD_DIR=0 GPU=7 \
SCENE_SLUG=japanesegradens VARIANT=d1_diag STAMP=20260803_gdadc_jg5k_v1 \
MAX_NUM_ITERATIONS=4500 RUN_EVAL=1 \
  bash scripts/experiments/gdadc_phase_replay_variant.sh

LOAD_DIR=.../nerfstudio_models_adam_sanitized SANITIZE_LOAD_DIR=0 GPU=8 \
SCENE_SLUG=japanesegradens VARIANT=d2_split STAMP=20260803_gdadc_jg5k_v1 \
MAX_NUM_ITERATIONS=4500 RUN_EVAL=1 \
  bash scripts/experiments/gdadc_phase_replay_variant.sh

LOAD_DIR=.../nerfstudio_models_adam_sanitized SANITIZE_LOAD_DIR=0 GPU=9 \
SCENE_SLUG=japanesegradens VARIANT=d3_split_clone STAMP=20260803_gdadc_jg5k_v1 \
MAX_NUM_ITERATIONS=4500 RUN_EVAL=1 \
  bash scripts/experiments/gdadc_phase_replay_variant.sh
```

## JapaneseGradens 5k Results

Gate criteria vs D0: `dPSNR >= +0.10`, `dSSIM >= +0.0003`, `dLPIPS <= -0.0020`, Gaussian count `<= 1.10x`.

| Run | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | Gaussians | Growth vs D0 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D0 M1 | 22.8876 | +0.0000 | 0.8016 | +0.0000 | 0.2397 | +0.0000 | 657,221 | +0.00% | control |
| D1 diagnostic-only | 22.9059 | +0.0183 | 0.8028 | +0.0013 | 0.2369 | -0.0028 | 654,702 | -0.38% | not a module result |
| D2 split-score | 22.8521 | -0.0355 | 0.8010 | -0.0006 | 0.2380 | -0.0017 | 687,890 | +4.67% | fail |
| D3 split+clone-score | 22.9216 | +0.0339 | 0.7996 | -0.0020 | 0.2422 | +0.0025 | 632,571 | -3.75% | fail |

Checkpoints:

```text
D0: outputs/gdadc_gdadc_d0_m1_japanesegradens_seed42_4500/water-splatting/gdadc_gdadc_d0_m1_japanesegradens_seed42_4500_20260803_gdadc_jg5k_v1/nerfstudio_models/step-000004999.ckpt
D1: outputs/gdadc_gdadc_d1_diag_japanesegradens_seed42_4500/water-splatting/gdadc_gdadc_d1_diag_japanesegradens_seed42_4500_20260803_gdadc_jg5k_v1/nerfstudio_models/step-000004999.ckpt
D2: outputs/gdadc_gdadc_d2_split_japanesegradens_seed42_4500/water-splatting/gdadc_gdadc_d2_split_japanesegradens_seed42_4500_20260803_gdadc_jg5k_v1/nerfstudio_models/step-000004999.ckpt
D3: outputs/gdadc_gdadc_d3_split_clone_japanesegradens_seed42_4500/water-splatting/gdadc_gdadc_d3_split_clone_japanesegradens_seed42_4500_20260803_gdadc_jg5k_v1/nerfstudio_models/step-000004999.ckpt
```

## GDADC Diagnostics

Diagnostic-only D1 showed that gradient-direction conflict is real:

- Step 700 large low-consistency ratio was about 80%.
- The ratio fell to about 6% by step 4900 as the model densified.
- With default threshold `0.0008`, GDADC split candidates were far below baseline split candidates.
- Thresholds matching baseline split counts stayed around `0.00030` to `0.00036` across the run.
- Clone-score logic would suppress about 28% of late baseline clone candidates.

D2 and D3 confirmed the mechanism changed density allocation:

| Run | Step | Base Split | GDADC Split | Base Clone | GDADC Clone | Applied Split | Applied Dup | Gaussians After Cull |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D2 | 700 | 16,975 | 16,951 | 0 | 0 | 16,951 | 0 | 24,385 |
| D2 | 4900 | 53,993 | 63,094 | 116,446 | 84,102 | 63,094 | 116,446 | 687,890 |
| D3 | 700 | 16,975 | 16,960 | 0 | 0 | 16,960 | 0 | 24,375 |
| D3 | 4900 | 56,094 | 65,003 | 115,568 | 83,926 | 65,003 | 83,926 | 632,571 |

D2 increased final Gaussian count by 4.67%; D3 reduced final Gaussian count by 3.75%. Both were within the Gaussian-count gate, but neither passed the metric gate.

## Decision

GDADC should not proceed to IUI3 safety or 15k formal runs in this configuration. The JapaneseGradens 5k gate failed because:

- D2 did not improve PSNR or SSIM.
- D3 improved PSNR by only `+0.0339 dB`, below the `+0.10 dB` gate.
- D3 reduced Gaussian count, but SSIM dropped by `-0.0020` and LPIPS worsened by `+0.0025`.

The diagnostic signal remains useful: M1/AbsGS has substantial early gradient direction conflict, but simply reweighting split/clone decisions does not translate into a robust three-metric improvement on JapaneseGradens.

## Next Conclusion

This result weakens the hypothesis that JapaneseGradens 5k failure is mainly a density-control split/clone candidate ranking problem. If continuing this family, the next experiment should not be another residual or pseudo-depth module. More plausible next checks are:

- why continuation replays are not bitwise stable even for D0;
- whether the early 80% low-consistency ratio is caused by rasterizer gradient noise rather than scene detail demand;
- whether M1 camera sampling, scale normalization, or medium/context learning rate coupling is the dominant source of the JapaneseGradens gap.

