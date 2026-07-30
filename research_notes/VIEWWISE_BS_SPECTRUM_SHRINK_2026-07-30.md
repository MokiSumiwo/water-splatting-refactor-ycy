# Viewwise BS Spectrum Shrink

Date: 2026-07-30
Branch: `refactor/core-framework`

## Goal

This round extends the offline view-wise medium inversion diagnostic with D11-D16 to separate:

```text
beta_bs total strength variation
vs
beta_bs RGB spectral-proportion variation
```

No training was run for this diagnostic. The main question is whether far blue/green residuals are better explained by backscatter spectral-ratio drift than by total backscatter strength drift.

## Code Changes

Updated:

```text
scripts/diagnostics/diagnose_viewwise_medium_inversion.py
scripts/experiments/run_viewwise_medium_inversion_all_scenes.sh
```

Added variants:

| Variant | Meaning |
|---|---|
| D11 | Uniform per-view BS spectral proportion, preserve per-pixel BS strength |
| D12 | Uniform per-view BS strength, preserve per-pixel BS spectral proportion |
| D13 | BS spectral proportion shrink to view mean, lambda=0.25 |
| D14 | BS spectral proportion shrink to view mean, lambda=0.50 |
| D15 | BS spectral proportion shrink to view mean, lambda=0.75 |
| D16 | Full BS coefficient shrink to view mean, lambda=0.25/0.50/0.75 |

Also added `abs_far_near_bg_gap` to avoid treating negative far-near blue/green gap as automatically better.

## Commands

Smoke:

```bash
STAMP=20260730_medium_inversion_d11_smoke CASE_FILTER=panama/m1 MAX_IMAGES=1 GPU_DEFAULT=6 \
  bash scripts/experiments/run_viewwise_medium_inversion_all_scenes.sh
```

Full M1-only diagnostic, run in parallel:

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/diagnostics/diagnose_viewwise_medium_inversion.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --split eval --max-images -1 \
  --far-mask-dir common_masks/m1_q90_iui3_redsea_20260724 \
  --region-mask-dir common_masks/m1_auto_eval_regions_iui3_redsea_20260724 \
  --output-dir renders/viewwise_medium_inversion_20260730_d11_d16_m1/iui3/m1 \
  --transmission-floor 0.05 --smooth-kernels 31 61 --minimum-mask-pixels 1000 \
  --save-full-resolution --save-contact-sheet --save-json
```

Equivalent commands were run for Curasao, JapaneseGradens, and Panama using GPUs 7, 8, and 9.

## Outputs

```text
renders/viewwise_medium_inversion_20260730_d11_d16_m1/
logs/viewwise_medium_inversion_20260730_d11_d16_m1/
```

Summary files:

```text
renders/viewwise_medium_inversion_20260730_d11_d16_m1/summary/summary.json
renders/viewwise_medium_inversion_20260730_d11_d16_m1/summary/summary.md
renders/viewwise_medium_inversion_20260730_d11_d16_m1/summary/summary_rows.csv
renders/viewwise_medium_inversion_20260730_d11_d16_m1/summary/summary_macro.csv
```

## Macro M1 Results

| Variant | Abs Gap | Improvement vs D1 | Near RGB MAE | Near Chroma MAE | Clip Rate | Scenes Improved |
|---|---:|---:|---:|---:|---:|---:|
| D1 pixel | 0.1825 | +0.0% | 0.0082 | 0.0021 | 0.0429 | 0/4 |
| D3 BS mean | 0.1164 | +40.7% | 0.0269 | 0.0112 | 0.0554 | 4/4 |
| D11 BS spectrum mean | 0.1298 | +28.8% | 0.0161 | 0.0110 | 0.0475 | 4/4 |
| D12 BS strength mean | 0.1651 | +11.2% | 0.0252 | 0.0043 | 0.0552 | 4/4 |
| D13 spectrum shrink 0.25 | 0.1685 | +8.5% | 0.0097 | 0.0040 | 0.0441 | 4/4 |
| D14 spectrum shrink 0.50 | 0.1554 | +15.5% | 0.0117 | 0.0063 | 0.0451 | 4/4 |
| D15 spectrum shrink 0.75 | 0.1426 | +22.2% | 0.0139 | 0.0086 | 0.0463 | 4/4 |
| D16 full BS shrink 0.75 | 0.1306 | +31.8% | 0.0219 | 0.0089 | 0.0516 | 4/4 |

## Interpretation

D11 is the cleanest spectrum-only candidate. It reduces macro absolute far-near blue/green gap by `28.8%`, improves all four scenes, and stays within the offline safety thresholds:

```text
Near RGB MAE    = 0.0161 <= 0.025
Near Chroma MAE = 0.0110 <= 0.020
Clip Rate       = 0.0475 <= 0.070
```

D12 is much weaker, with only `11.2%` macro absolute-gap improvement. This supports the hypothesis that the problematic signal is more related to BS RGB spectral-proportion drift than to total BS strength alone.

D3 remains the strongest absolute-gap reducer, but it misses the near RGB safety threshold at macro level (`0.0269`) and has larger per-scene near damage, especially on IUI3 and JapaneseGradens. It is useful as a mechanism upper bound, not as the preferred intervention.

D16 full BS shrink 0.75 is competitive on macro gap, but it mixes spectral and total-strength shrink. Since D11 achieves nearly the same macro abs gap with lower near RGB MAE, D11 is the preferred next candidate for renderer-native intervention.

## Decision

Proceed to renderer-native no-training intervention with:

```text
original BS
D3 BS mean
D11 BS spectrum mean
D14 BS spectrum shrink 0.50
```

D11 is the primary candidate. D14 is included as a conservative partial-shrink fallback. D3 is included as the full-BS-mean upper-bound reference.

## Renderer-Native Intervention

Added script:

```text
scripts/diagnostics/diagnose_native_bs_intervention.py
```

This is a no-training native rasterizer intervention. It leaves Gaussian geometry, opacity, SH color, medium RGB, and medium attenuation unchanged, then monkeypatches the Python rasterizer wrapper to replace only `medium_bs` before the CUDA rasterizer call.

Variants:

```text
original
D3_bs_mean
D11_bs_spectrum_mean
D14_bs_spectrum_shrink050
```

Outputs:

```text
renders/native_bs_intervention_20260730_d11_d14_m1/
logs/native_bs_intervention_20260730_d11_d14_m1/
renders/native_bs_intervention_20260730_d11_d14_m1/summary/summary.json
renders/native_bs_intervention_20260730_d11_d14_m1/summary/summary_rows.csv
renders/native_bs_intervention_20260730_d11_d14_m1/summary/summary_macro.csv
```

Macro reconstruction results:

| Variant | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | Safety Pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 30.0905 | +0.0000 | 0.9293 | +0.00000 | 0.1194 | +0.00000 | 4/4 |
| D3 BS mean | 29.0164 | -1.0741 | 0.9282 | -0.00108 | 0.1224 | +0.00298 | 0/4 |
| D11 BS spectrum mean | 29.7454 | -0.3451 | 0.9290 | -0.00022 | 0.1215 | +0.00214 | 0/4 |
| D14 BS spectrum shrink 0.50 | 29.9940 | -0.0965 | 0.9292 | -0.00005 | 0.1201 | +0.00069 | 0/4 |

Per-scene highlights:

| Scene | Variant | dPSNR | dSSIM | dLPIPS |
|---|---|---:|---:|---:|
| IUI3 | D11 | -1.0625 | -0.00049 | +0.00467 |
| IUI3 | D14 | -0.2924 | -0.00011 | +0.00124 |
| JapaneseGradens | D11 | -0.0338 | -0.00011 | +0.00192 |
| JapaneseGradens | D14 | -0.0083 | -0.00003 | +0.00086 |
| Panama | D11 | -0.2219 | -0.00008 | +0.00025 |
| Panama | D14 | -0.0789 | +0.00000 | +0.00004 |
| Curasao | D11 | -0.0620 | -0.00020 | +0.00174 |
| Curasao | D14 | -0.0063 | -0.00006 | +0.00062 |

Safety gate:

```text
PSNR drop <= 0.03 dB
SSIM drop <= 0.0005
LPIPS increase <= 0.0005
```

No native intervention variant passes the full gate. D14 is close on SSIM and often small on PSNR, but it still fails macro PSNR and LPIPS, and IUI3 remains clearly unsafe. D11 is too destructive in native underwater RGB, despite being the cleanest offline spectrum-only inversion candidate.

Important detail: native `medium_bs` intervention does not change `J` metrics in this diagnostic, because renderer clear `J` is produced by the Gaussian clear branch. This test only answers whether BS spectrum replacement can preserve underwater RGB under the native rasterizer.

## Final Decision

Do not proceed directly to the training-side `View-Consistent Backscatter Spectrum` module yet.

The current evidence is:

```text
closed-form inversion: D11 supports BS spectral drift as a real mechanism
native rasterizer RGB: direct BS spectrum replacement is not reconstruction-safe
```

This matches case D in the plan: offline-effective but native-RGB unsafe. The next step should not be a BS spectrum training loss. Instead, inspect why native RGB is sensitive to even conservative BS spectral replacement, especially on IUI3:

```text
Gaussian clear color / medium coupling
native multi-layer BS integration vs single-depth closed-form approximation
depth / final transmittance mismatch
tail and finite-medium contribution split
```

