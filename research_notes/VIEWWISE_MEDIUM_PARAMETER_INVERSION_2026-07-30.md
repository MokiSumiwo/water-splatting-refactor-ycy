# Viewwise Medium Parameter Inversion

Date: 2026-07-30
Branch: `refactor/core-framework`

## Goal

This is an offline, no-training diagnostic. The question is whether far blue/green clear residuals are mainly caused by excessive same-view spatial variation in:

```text
A(x,y) = medium_rgb
beta_bs(x,y) = medium_bs
beta_attn(x,y) = medium_attn
```

The diagnostic uses the simplified model:

```text
I = J * exp(-beta_attn * depth) + A * (1 - exp(-beta_bs * depth))
```

This is an approximation, because WaterSplatting composes Gaussian and medium contributions through the rasterizer rather than a single-depth closed form.

## Files

```text
scripts/diagnostics/diagnose_viewwise_medium_inversion.py
scripts/diagnostics/viewwise_medium_inversion_cases.tsv
scripts/experiments/run_viewwise_medium_inversion_all_scenes.sh
```

## Variants

```text
D0  model J
D1  pixel A, pixel beta_bs, pixel beta_attn
D2  view-mean A only
D3  view-mean beta_bs only
D4  view-mean beta_attn only
D5  all view-mean
D6  all view-median
D7  water/open-region mean
D8  mean A + geometric-mean beta
D9  smooth-31 parameters
D10 smooth-61 parameters
```

## Command

```bash
STAMP=20260730_medium_inversion \
GPU_IUI3=6 \
GPU_CURASAO=7 \
GPU_JGRADENS=8 \
GPU_PANAMA=9 \
bash scripts/experiments/run_viewwise_medium_inversion_all_scenes.sh
```

## Results

Output root:

```text
renders/viewwise_medium_inversion_20260730_medium_inversion/
logs/viewwise_medium_inversion_20260730_medium_inversion/
```

Smoke:

```text
STAMP=20260730_medium_inversion_smoke CASE_FILTER=panama/m1 MAX_IMAGES=1 GPU_DEFAULT=6 \
bash scripts/experiments/run_viewwise_medium_inversion_all_scenes.sh
```

Smoke confirmed that the Panama `water` mask is invalid and correctly rejected:

```text
water_pixels = 1
open_source = far_lowgrad_fallback
open_coverage = 3.73%
```

Full run:

```text
STAMP=20260730_medium_inversion \
GPU_IUI3=6 \
GPU_CURASAO=7 \
GPU_JGRADENS=8 \
GPU_PANAMA=9 \
bash scripts/experiments/run_viewwise_medium_inversion_all_scenes.sh
```

Summary artifacts:

```text
renders/viewwise_medium_inversion_20260730_medium_inversion/summary/summary.json
renders/viewwise_medium_inversion_20260730_medium_inversion/summary/summary.md
renders/viewwise_medium_inversion_20260730_medium_inversion/summary/summary_rows.csv
```

Each scene/method also has:

```text
aggregate.json
view_000*/metrics.json
view_000*/parameter_sheet.png
view_000*/inversion_sheet.png
inversion_contact_sheet_all_views.png
```

## Forward Approximation Check

| Scene | Method | Forward PSNR | Forward MAE | Interpretation |
|---|---|---:|---:|---|
| IUI3 | Baseline | 30.58 | 0.0113 | usable |
| IUI3 | M1 | 37.14 | 0.0057 | strong |
| Curasao | Baseline | 25.32 | 0.0191 | qualitative only |
| Curasao | M1 | 26.36 | 0.0171 | qualitative only |
| JapaneseGradens | Baseline | 29.56 | 0.0136 | usable but not strict |
| JapaneseGradens | M1 | 30.69 | 0.0123 | usable |
| Panama | Baseline | 33.19 | 0.0049 | strong |
| Panama | M1 | 30.74 | 0.0071 | usable |

The closed-form approximation is credible on IUI3 M1, JapaneseGradens M1, Panama baseline, and Panama M1. Curasao should be treated as qualitative because forward PSNR is only `25-26 dB`.

## M1 Variant Results

The table reports far-near blue/green gap, relative to D1 pixel inversion. Lower gap is better.

| Scene | Variant | Gap | Gap Delta vs D1 | Near MAE | Clip Rate |
|---|---|---:|---:|---:|---:|
| IUI3 | D1 Pixel | 0.0287 | +0.0000 | 0.0057 | 0.0328 |
| IUI3 | D2 A Mean | 0.2037 | +0.1750 | 0.0459 | 0.2705 |
| IUI3 | D3 BS Mean | -0.0231 | -0.0518 | 0.0389 | 0.0518 |
| IUI3 | D4 Attn Mean | 0.0543 | +0.0256 | 0.0350 | 0.0294 |
| IUI3 | D5 All Mean | 0.1979 | +0.1693 | 0.0671 | 0.2777 |
| IUI3 | D7 Open Mean | 0.0925 | +0.0638 | 0.0802 | 0.1336 |
| Curasao | D1 Pixel | 0.2345 | +0.0000 | 0.0185 | 0.0444 |
| Curasao | D2 A Mean | 0.7001 | +0.4656 | 0.0703 | 0.1609 |
| Curasao | D3 BS Mean | 0.1062 | -0.1283 | 0.0253 | 0.0505 |
| Curasao | D4 Attn Mean | 0.3899 | +0.1553 | 0.0797 | 0.0766 |
| Curasao | D5 All Mean | 0.5926 | +0.3581 | 0.1096 | 0.2022 |
| Curasao | D7 Open Mean | 0.1664 | -0.0682 | 0.1776 | 0.1710 |
| JapaneseGradens | D1 Pixel | 0.1411 | +0.0000 | 0.0048 | 0.0519 |
| JapaneseGradens | D2 A Mean | 0.5557 | +0.4146 | 0.0258 | 0.1372 |
| JapaneseGradens | D3 BS Mean | 0.0520 | -0.0890 | 0.0322 | 0.0806 |
| JapaneseGradens | D4 Attn Mean | 0.1820 | +0.0409 | 0.0201 | 0.0477 |
| JapaneseGradens | D5 All Mean | 0.5652 | +0.4242 | 0.0413 | 0.1411 |
| JapaneseGradens | D7 Open Mean | 0.1765 | +0.0355 | 0.0586 | 0.1425 |
| Panama | D1 Pixel | 0.3046 | +0.0000 | 0.0039 | 0.0425 |
| Panama | D2 A Mean | 0.4352 | +0.1306 | 0.0280 | 0.1032 |
| Panama | D3 BS Mean | 0.2749 | -0.0297 | 0.0112 | 0.0385 |
| Panama | D4 Attn Mean | 0.4292 | +0.1246 | 0.0334 | 0.0832 |
| Panama | D5 All Mean | 0.3975 | +0.0929 | 0.0474 | 0.1301 |
| Panama | D7 Open Mean | 0.3672 | +0.0626 | 0.0977 | 0.1088 |

Macro averages over M1 scenes:

| Variant | Macro Gap | Gap Delta vs D1 | Near MAE | Clip Rate |
|---|---:|---:|---:|---:|
| D0 Model | 0.1192 | -0.0580 | 0.0000 | 0.0378 |
| D1 Pixel | 0.1772 | +0.0000 | 0.0082 | 0.0429 |
| D2 A Mean | 0.4737 | +0.2964 | 0.0425 | 0.1679 |
| D3 BS Mean | 0.1025 | -0.0747 | 0.0269 | 0.0554 |
| D4 Attn Mean | 0.2638 | +0.0866 | 0.0421 | 0.0593 |
| D5 All Mean | 0.4383 | +0.2611 | 0.0663 | 0.1878 |
| D7 Open Mean | 0.2007 | +0.0234 | 0.1035 | 0.1390 |
| D9 Smooth31 | 0.1773 | +0.0001 | 0.0083 | 0.0500 |
| D10 Smooth61 | 0.1777 | +0.0005 | 0.0083 | 0.0533 |

## Parameter Statistics

| Scene | Method | A NTV | BS NTV | Attn NTV | corr(A,d) | corr(BS,d) | corr(Attn,d) |
|---|---|---:|---:|---:|---:|---:|---:|
| IUI3 | Baseline | 0.000620 | 0.001333 | 0.001671 | -0.460 | -0.386 | +0.424 |
| IUI3 | M1 | 0.000878 | 0.001498 | 0.001896 | -0.132 | +0.454 | -0.568 |
| Curasao | Baseline | 0.001147 | 0.002233 | 0.000661 | -0.817 | +0.759 | -0.660 |
| Curasao | M1 | 0.001050 | 0.000699 | 0.000972 | +0.763 | -0.112 | -0.657 |
| JapaneseGradens | Baseline | 0.000659 | 0.001108 | 0.001276 | -0.620 | -0.435 | +0.403 |
| JapaneseGradens | M1 | 0.000796 | 0.001348 | 0.001110 | +0.522 | -0.423 | -0.314 |
| Panama | Baseline | 0.000702 | 0.000735 | 0.000614 | +0.530 | +0.451 | +0.060 |
| Panama | M1 | 0.000798 | 0.000643 | 0.000919 | +0.870 | -0.217 | -0.383 |

## Interpretation

The broad hypothesis is not supported in its strongest form. Full view-level unification of all medium parameters, D5, worsens M1 macro far-near blue/green gap from `0.1772` to `0.4383`, increases near MAE to `0.0663`, and raises clipping to `18.8%`.

The results do not support `A(x,y)` as the primary source of residual blue/green bias. D2 worsens every M1 scene and has the highest macro gap among the tested variants.

The results also do not support `beta_attn(x,y)` as the primary issue. D4 worsens all M1 gaps and often exceeds the near-MAE safety target.

High-frequency parameter instability is not the main issue. D9/D10 are nearly identical to D1 on the M1 macro gap, so large-kernel smoothing does not materially improve the blue/green residual.

The only consistent signal is `beta_bs`. D3 improves the M1 macro gap from `0.1772` to `0.1025`, with macro near MAE `0.0269` and modest clip increase. It improves IUI3, Curasao, and JapaneseGradens strongly, and improves Panama slightly. However, IUI3 and JapaneseGradens slightly miss the strict near-MAE target, so this should be treated as a mechanism lead rather than a training-ready loss.

The current model J, D0, remains better than the simplified inversion D1 on macro gap. This means the simplified inversion is useful for mechanism diagnosis, but should not replace renderer-native J or be interpreted as a strict causal rendering model.

## Conclusion

Do not add a blanket view-level parameter unification loss. It is too destructive.

The next plausible training-side idea is a conservative, low-weight regularizer or architecture constraint on `beta_bs` only:

```text
view-level beta_bs base + small bounded spatial residual
or low-frequency beta_bs branch
or weak beta_bs depth-decorrelation / TV prior
```

Avoid constraining `medium_rgb` or `medium_attn` globally based on this experiment.
