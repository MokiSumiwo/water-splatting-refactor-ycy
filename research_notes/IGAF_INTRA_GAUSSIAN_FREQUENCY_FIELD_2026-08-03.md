# IGAF Intra-Gaussian Adaptive Frequency Field

Date: 2026-08-03

## Summary

IGAF tests whether the remaining M1 high-frequency gap comes from the per-Gaussian appearance parameterization rather than geometry, densification, pseudo-depth, or medium ownership. The module adds a small bounded local harmonic residual inside each Gaussian footprint before underwater attenuation/backscatter.

The first-pass result is mixed but mostly negative for 15k promotion:

- Frozen-M1 oracle produced only tiny eval gains, far below the gate.
- 5k from-scratch IGAF improved both JapaneseGradens and IUI3 relative to same-step M1, but no single canonical configuration passed the full 5k gate.
- The best JapaneseGradens PSNR candidate was G2 Mip, with +0.094 dB PSNR and +0.00055 SSIM, but LPIPS improved only -0.00125.
- The best JapaneseGradens LPIPS candidate was G1 no-Mip, with -0.00329 LPIPS, but PSNR improved only +0.034 dB.
- IUI3 was safe and improved for G1/G2, so IGAF is not obviously harmful, but the 15k gate is not met.

Conclusion: do not promote IGAF V0 to 15k/four-scene formal runs yet. The evidence suggests intra-Gaussian texture has some early-training benefit, but the frozen mature-M1 oracle is too weak to support the claim that M1's 15k JapaneseGradens deficit is mainly missing per-Gaussian local texture capacity.

## Mechanism

IGAF adds `igaf_coeffs` with shape `(N, 4, 3)` to every Gaussian. The four bases are:

```text
cos(wu) - exp(-w^2 / 2)
sin(wu)
cos(wv) - exp(-w^2 / 2)
sin(wv)
```

The local residual is:

```text
delta_rgb = amplitude_max * tanh(sum_k coeff[k] * basis[k])
rgb_local = rgb_sh + gate * delta_rgb
```

The residual is added to intrinsic Gaussian radiance before the existing underwater attenuation/backscatter compositor. It does not add J supervision, pseudo-depth, extra densification, medium losses, pruning, cleanup, or renderer-physics changes beyond this local intrinsic radiance term.

The canonical local coordinate frame projects the two largest Gaussian ellipsoid axes into the current camera and applies a detached stable pseudo-inverse to map pixel offset to `(u, v)`. A screen-space coordinate mode is included only as O3 control.

The detached gate is:

```text
gate = ramp * planarity_gate * projection_condition_gate * optional_mip_gate
```

All coordinate/gate paths are detached. Only `igaf_coeffs` receive gradients through the local texture path.

## Code Changes

- Added CUDA RGB rasterizer variants with IGAF support:
  - `water_splatting/cuda/csrc/forward.cu`
  - `water_splatting/cuda/csrc/backward.cu`
  - `water_splatting/cuda/csrc/bindings.cu`
  - `water_splatting/cuda/csrc/ext.cpp`
- Added Python autograd/wrapper support:
  - `water_splatting/rasterize.py`
  - `water_splatting/rendering/underwater_rasterizer.py`
- Added model config/state/render integration:
  - `water_splatting/water_splatting.py`
  - `water_splatting/water_splatting_config.py`
- Added experiment wrappers:
  - `scripts/experiments/igaf_5k_common.sh`
  - `scripts/experiments/igaf_oracle_common.sh`
  - `scripts/experiments/igaf_g0_m1_*_5k.sh`
  - `scripts/experiments/igaf_g1_nomip_*_5k.sh`
  - `scripts/experiments/igaf_g2_mip_*_5k.sh`
  - `scripts/experiments/igaf_g3_screen_*_5k.sh`
  - `scripts/experiments/igaf_o1_nomip_*.sh`
  - `scripts/experiments/igaf_o2_mip_*.sh`
  - `scripts/experiments/igaf_o3_screen_*.sh`

## Config Flags

```text
igaf_enabled: bool = False
igaf_start_step: int = 10000
igaf_ramp_steps: int = 1000
igaf_frequency: float = 1.5
igaf_amplitude_max: float = 0.10
igaf_coordinate_mode: Literal["canonical", "screen"] = "canonical"
igaf_planarity_ratio: float = 2.0
igaf_planarity_temperature: float = 0.25
igaf_condition_threshold: float = 5.0
igaf_condition_temperature: float = 5.0
igaf_mip_enabled: bool = True
igaf_coordinate_clamp: float = 3.0
lambda_igaf_amplitude: float = 0.0
igaf_log_path: Optional[str] = None
igaf_freeze_base_gaussians: bool = False
igaf_freeze_medium: bool = False
```

`igaf_freeze_base_gaussians` and `igaf_freeze_medium` are only for the frozen-M1 coefficient oracle.

## Verification

Static and CUDA checks:

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/water_splatting.py \
  water_splatting/rasterize.py \
  water_splatting/rendering/underwater_rasterizer.py \
  water_splatting/water_splatting_config.py

bash -n scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh scripts/experiments/igaf_*.sh
git diff --check
git ls-files outputs renders logs common_masks | wc -l
```

Results:

- `py_compile`: passed
- `bash -n`: passed
- `git diff --check`: passed
- tracked generated-output files: `0`
- CUDA symbol check: `rasterize_forward_igaf=True`, `rasterize_backward_igaf=True`
- CUDA equivalence smoke: zero coeffs produced `max_abs_diff_zero=0.0`
- CUDA gradient smoke: IGAF coeff gradient was nonzero
- 20-step IUI3 smoke: passed
- 720-step JapaneseGradens smoke: passed through split/duplicate/cull; `igaf_coeffs` optimizer/state shape stayed valid
- `LOAD_DIR` oracle smoke: passed when using continuation semantics correctly

Note: `--load-checkpoint` behaves like weight initialization for this trainer path and can continue for a full new run. Frozen oracle/continuation should use `--load-dir`; `--max-num-iterations` is treated as additional training iterations.

## Commands

5k JapaneseGradens:

```bash
STAMP=20260803_igaf5k_jg RUN_EVAL=1 RUN_CLOSURE_DIAG=0 scripts/experiments/igaf_g0_m1_japanesegradens_5k.sh
STAMP=20260803_igaf5k_jg RUN_EVAL=1 RUN_CLOSURE_DIAG=0 scripts/experiments/igaf_g1_nomip_japanesegradens_5k.sh
STAMP=20260803_igaf5k_jg RUN_EVAL=1 RUN_CLOSURE_DIAG=0 scripts/experiments/igaf_g2_mip_japanesegradens_5k.sh
STAMP=20260803_igaf5k_jg RUN_EVAL=1 RUN_CLOSURE_DIAG=0 scripts/experiments/igaf_g3_screen_japanesegradens_5k.sh
```

5k IUI3:

```bash
STAMP=20260803_igaf5k_iui3 RUN_EVAL=1 RUN_CLOSURE_DIAG=0 scripts/experiments/igaf_g0_m1_iui3_5k.sh
STAMP=20260803_igaf5k_iui3 RUN_EVAL=1 RUN_CLOSURE_DIAG=0 scripts/experiments/igaf_g1_nomip_iui3_5k.sh
STAMP=20260803_igaf5k_iui3 RUN_EVAL=1 RUN_CLOSURE_DIAG=0 scripts/experiments/igaf_g2_mip_iui3_5k.sh
STAMP=20260803_igaf5k_iui3 RUN_EVAL=1 RUN_CLOSURE_DIAG=0 scripts/experiments/igaf_g3_screen_iui3_5k.sh
```

Frozen M1 oracle:

```bash
JG_LOAD=outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/nerfstudio_models
IUI3_LOAD=outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/nerfstudio_models

STAMP=20260803_igaf_oracle_jg LOAD_DIR="$JG_LOAD" RUN_EVAL=1 MAX_NUM_ITERATIONS=1000 MODEL_NUM_STEPS=16000 scripts/experiments/igaf_o1_nomip_japanesegradens.sh
STAMP=20260803_igaf_oracle_jg LOAD_DIR="$JG_LOAD" RUN_EVAL=1 MAX_NUM_ITERATIONS=1000 MODEL_NUM_STEPS=16000 scripts/experiments/igaf_o2_mip_japanesegradens.sh
STAMP=20260803_igaf_oracle_jg LOAD_DIR="$JG_LOAD" RUN_EVAL=1 MAX_NUM_ITERATIONS=1000 MODEL_NUM_STEPS=16000 scripts/experiments/igaf_o3_screen_japanesegradens.sh

STAMP=20260803_igaf_oracle_iui3 LOAD_DIR="$IUI3_LOAD" RUN_EVAL=1 MAX_NUM_ITERATIONS=1000 MODEL_NUM_STEPS=16000 scripts/experiments/igaf_o1_nomip_iui3.sh
STAMP=20260803_igaf_oracle_iui3 LOAD_DIR="$IUI3_LOAD" RUN_EVAL=1 MAX_NUM_ITERATIONS=1000 MODEL_NUM_STEPS=16000 scripts/experiments/igaf_o2_mip_iui3.sh
STAMP=20260803_igaf_oracle_iui3 LOAD_DIR="$IUI3_LOAD" RUN_EVAL=1 MAX_NUM_ITERATIONS=1000 MODEL_NUM_STEPS=16000 scripts/experiments/igaf_o3_screen_iui3.sh
```

JapaneseGradens tuning:

```bash
# G4: no-Mip, amplitude 0.05
# G5: no-Mip, frequency 1.0
# G6: Mip, frequency 0.75
```

## Frozen M1 Oracle Results

15k M1 baselines:

| Scene | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| JapaneseGradens | 24.7565 | 0.8995 | 0.1204 |
| IUI3 | 31.1314 | 0.9120 | 0.1750 |

Oracle deltas after 1000 coefficient-only steps:

| Scene | Run | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | Gaussians | Active frac | Gate mean | Coeff p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JapaneseGradens | O1 no-Mip | 24.7584 | +0.0020 | 0.8997 | +0.0002 | 0.1202 | -0.0001 | 861,508 | 0.3693 | 0.2735 | 0.0726 |
| JapaneseGradens | O2 Mip | 24.7573 | +0.0009 | 0.8996 | +0.0001 | 0.1203 | -0.0000 | 861,508 | 0.2844 | 0.0763 | 0.0698 |
| JapaneseGradens | O3 screen | 24.7561 | -0.0004 | 0.8997 | +0.0002 | 0.1202 | -0.0002 | 861,508 | 0.4036 | 0.3246 | 0.0811 |
| IUI3 | O1 no-Mip | 31.1368 | +0.0053 | 0.9122 | +0.0002 | 0.1748 | -0.0002 | 807,502 | 0.4196 | 0.3088 | 0.0529 |
| IUI3 | O2 Mip | 31.1319 | +0.0005 | 0.9121 | +0.0001 | 0.1749 | -0.0001 | 807,502 | 0.3220 | 0.0710 | 0.0511 |
| IUI3 | O3 screen | 31.1292 | -0.0022 | 0.9122 | +0.0002 | 0.1748 | -0.0002 | 807,502 | 0.4506 | 0.3520 | 0.0576 |

Oracle gate result: fail. The improvements are orders of magnitude below the requested +0.10 dB PSNR and -0.002 LPIPS gate.

## 5k Results

Same-step 5k baselines:

| Scene | Run | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | Gaussians | Growth | Active frac | Gate mean | Coeff p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JapaneseGradens | G0 M1 | 22.9459 | +0.0000 | 0.8018 | +0.0000 | 0.2382 | +0.0000 | 717,723 | +0.00% | | | |
| JapaneseGradens | G1 no-Mip | 22.9801 | +0.0342 | 0.8023 | +0.0005 | 0.2349 | -0.0033 | 723,259 | +0.77% | 0.7358 | 0.2975 | 0.1311 |
| JapaneseGradens | G2 Mip | 23.0399 | +0.0940 | 0.8023 | +0.0005 | 0.2370 | -0.0013 | 709,692 | -1.12% | 0.0899 | 0.0102 | 0.1367 |
| JapaneseGradens | G3 screen | 22.9818 | +0.0358 | 0.8021 | +0.0003 | 0.2381 | -0.0001 | 718,689 | +0.13% | 0.7669 | 0.4596 | 0.1216 |
| IUI3 | G0 M1 | 25.6736 | +0.0000 | 0.7642 | +0.0000 | 0.3079 | +0.0000 | 642,110 | +0.00% | | | |
| IUI3 | G1 no-Mip | 25.8810 | +0.2073 | 0.7659 | +0.0017 | 0.3037 | -0.0042 | 645,771 | +0.57% | 0.4038 | 0.1632 | 0.1327 |
| IUI3 | G2 Mip | 25.8612 | +0.1875 | 0.7654 | +0.0012 | 0.3038 | -0.0041 | 647,998 | +0.92% | 0.0424 | 0.0060 | 0.1193 |
| IUI3 | G3 screen | 25.6458 | -0.0278 | 0.7650 | +0.0008 | 0.3056 | -0.0024 | 644,148 | +0.32% | 0.4240 | 0.2695 | 0.1257 |

5k gate result: fail, but with useful signal.

- G2 is closest on JapaneseGradens PSNR but misses the LPIPS gate.
- G1 passes JapaneseGradens LPIPS directionally but misses the PSNR gate.
- G1/G2 are safe on IUI3 and actually improve IUI3.
- Gaussian count stayed within the limit.

## 5k Tuning Results

JapaneseGradens only:

| Run | Change | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | Growth |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G4 | no-Mip, amplitude 0.05 | 22.9415 | -0.0044 | 0.8013 | -0.0005 | 0.2384 | +0.0002 | -0.25% |
| G5 | no-Mip, frequency 1.0 | 22.9868 | +0.0409 | 0.8027 | +0.0009 | 0.2423 | +0.0040 | -0.12% |
| G6 | Mip, frequency 0.75 | 23.0032 | +0.0573 | 0.8013 | -0.0005 | 0.2402 | +0.0020 | +0.65% |

Tuning result: no improvement over G1/G2. Do not expand these variants.

## Interpretation

The coefficient norms stayed far from saturation, so V0 did not fail because the amplitude bound was too low. Instead, the frozen 15k oracle shows that mature M1 Gaussians cannot extract meaningful novel-view gains from this local harmonic field when geometry and base SH are fixed.

The 5k gains likely come from interaction with early training and densification dynamics rather than pure final representation capacity. This is useful but not enough for the intended claim because the module was meant to improve representation without changing densification or relying on early training instability.

The Mip version is safer for PSNR on JapaneseGradens but gates most texture away by 5k (`gate_mean=0.0102`). The no-Mip version improves LPIPS but does not deliver enough PSNR, and lower frequency/amplitude tuning does not fix the tradeoff.

## Decision

Do not enter 15k formal continuation for IGAF V0.

Best 5k candidates:

- JapaneseGradens PSNR: G2 Mip
- JapaneseGradens LPIPS: G1 no-Mip
- IUI3 safety: G1/G2 both safe

Current conclusion:

IGAF does not confirm that M1's 15k JapaneseGradens loss mainly comes from missing intra-Gaussian local texture. It remains an interesting optional early-training regular appearance extension, but not a strong next mainline module.

## Checkpoints and Outputs

Primary 5k checkpoints:

```text
outputs/igaf_g1_nomip_japanesegradens_seed42_5000/water-splatting/igaf_g1_nomip_japanesegradens_seed42_5000_20260803_igaf5k_jg/nerfstudio_models/step-000004999.ckpt
outputs/igaf_g2_mip_japanesegradens_seed42_5000/water-splatting/igaf_g2_mip_japanesegradens_seed42_5000_20260803_igaf5k_jg/nerfstudio_models/step-000004999.ckpt
outputs/igaf_g1_nomip_iui3_seed42_5000/water-splatting/igaf_g1_nomip_iui3_seed42_5000_20260803_igaf5k_iui3/nerfstudio_models/step-000004999.ckpt
outputs/igaf_g2_mip_iui3_seed42_5000/water-splatting/igaf_g2_mip_iui3_seed42_5000_20260803_igaf5k_iui3/nerfstudio_models/step-000004999.ckpt
```

Primary metric JSONs:

```text
renders/igaf_g1_nomip_japanesegradens_seed42_5000_20260803_igaf5k_jg/output.json
renders/igaf_g2_mip_japanesegradens_seed42_5000_20260803_igaf5k_jg/output.json
renders/igaf_g1_nomip_iui3_seed42_5000_20260803_igaf5k_iui3/output.json
renders/igaf_g2_mip_iui3_seed42_5000_20260803_igaf5k_iui3/output.json
```

IGAF logs:

```text
logs/igaf_g1_nomip_japanesegradens_seed42_5000_20260803_igaf5k_jg/igaf.jsonl
logs/igaf_g2_mip_japanesegradens_seed42_5000_20260803_igaf5k_jg/igaf.jsonl
logs/igaf_g1_nomip_iui3_seed42_5000_20260803_igaf5k_iui3/igaf.jsonl
logs/igaf_g2_mip_iui3_seed42_5000_20260803_igaf5k_iui3/igaf.jsonl
```

No generated outputs, renders, logs, common masks, checkpoints, images, videos, numpy files, or torch binaries should be committed.
