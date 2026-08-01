# Hierarchical Attenuation Spectrum Diagnostic

Date: 2026-07-31

Commit at diagnostic time: `527789fc2001d557489b84b94cfa51b5f64602a6`

## Purpose

This note records the low-cost offline decomposition requested after stopping the
TBAP gradient preconditioning route. The goal is to decide whether bounded
attenuation spectrum should become a trainable module, or whether the observed
far clear color bias is dominated by a higher-level view/scene spectrum term.

The diagnostic is intentionally read-only. It does not change CUDA, model
training, checkpoints, or renderer composition.

## Added Diagnostic

New script:

```text
scripts/diagnostics/diagnose_attn_spectrum_decomposition.py
```

For each evaluated view, it decomposes the learned attenuation spectrum:

```text
z(p) = clr(log beta_D(p))
s_v  = mean_p z(p)
r(p) = z(p) - s_v
```

It reports:

- `view_spectrum_violation_rate`
- `full_violation_rate_valid`
- `view_unresolved_violation_rate_valid`
- `residual_induced_violation_rate_valid`
- `residual_corrected_view_violation_rate_valid`
- view/pixel spectral variance fractions
- correlation between view spectrum and far J bias
- correlation between pixel residual magnitude and pixel J bias

The ordering violation used here is the centered attenuation spectrum condition:

```text
not (z_R >= z_G >= z_B)
```

This is diagnostic only, not a hard training target.

## Commands

IUI3 M1:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_attn_spectrum_decomposition.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --load-step 14999 \
  --test-mode inference \
  --max-images 4 \
  --scene-name IUI3-RedSea \
  --output-dir renders/attn_spectrum_decomposition_20260731/iui3_m1
```

JapaneseGradens M1:

```bash
CUDA_VISIBLE_DEVICES=7 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_attn_spectrum_decomposition.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --load-step 10000 \
  --test-mode inference \
  --max-images 3 \
  --scene-name JapaneseGradens-RedSea \
  --output-dir renders/attn_spectrum_decomposition_20260731/japanese_m1_step10000
```

Outputs:

```text
renders/attn_spectrum_decomposition_20260731/iui3_m1/attn_spectrum_decomposition.json
renders/attn_spectrum_decomposition_20260731/japanese_m1_step10000/attn_spectrum_decomposition.json
```

## Results

| Metric | IUI3 M1 step 14999 | JapaneseGradens M1 step 10000 |
|---|---:|---:|
| view spectrum violation rate | 1.0000 | 1.0000 |
| full violation rate valid | 0.9170 | 0.9022 |
| view-unresolved violation rate valid | 0.9170 | 0.9022 |
| residual-induced violation rate valid | 0.0000 | 0.0000 |
| residual-corrected view violation rate valid | 0.0830 | 0.0978 |
| pixel residual abs valid mean | 0.0627 | 0.0355 |
| pixel residual abs far mean | 0.0707 | 0.0415 |
| corr(view spectrum BG-R, far J medium axis) | -0.6756 | 0.8169 |
| corr(view spectrum BG-R, far J BG-R) | -0.6083 | 0.8150 |
| corr(pixel residual abs, far J medium axis) | -0.1952 | -0.1996 |
| view spectrum variance fraction | 0.1230 | 0.2522 |
| pixel residual variance fraction | 0.8770 | 0.7478 |

## Interpretation

The strongest conclusion is that attenuation spectral violations are already
present at the view-spectrum level in both tested scenes:

```text
view_spectrum_violation_rate = 1.0
residual_induced_violation_rate_valid = 0.0
```

This means the pixel residual is not the source that turns an otherwise valid
view spectrum into an invalid full spectrum. In fact, the residual sometimes
corrects the view-level violation:

```text
residual_corrected_view_violation_rate_valid
IUI3              0.0830
JapaneseGradens   0.0978
```

Pixel residual still carries most total spectral variance, especially on IUI3,
but this is a variance statement, not a causality statement for the violation.
The actual violation source is view-unresolved.

The view-spectrum correlation with far J color bias is scene-dependent:

- JapaneseGradens has strong positive correlation with far J BG-vs-red bias.
- IUI3 has negative correlation under this metric.

Therefore, a naive pixel-only bounded residual module is unlikely to be a stable
general fix. It may regularize redundant attenuation freedom, but it does not
target the dominant violation source observed here.

## Decision

Stop TBAP gradient preconditioning as an active route.

Do not immediately run a new full 15k attenuation module. The correct next gate
is a 1000-step frozen-structure pilot, and the trainable version should include
scene/view/pixel hierarchy rather than only bounded pixel residuals.

Recommended pilot matrix:

| ID | Attenuation | A / BS | Gaussian appearance | Purpose |
|---|---|---|---|---|
| A0 | original M1 | frozen | DC-only | frozen-structure control |
| A1 | bounded pixel residual, rho_p=0.20 | frozen | DC-only | clean pixel-bound causality test |
| A2 | bounded pixel residual, rho_p=0.30 | frozen | DC-only | weaker bound / RGB safety check |
| A3 | bounded pixel residual, rho_p=0.20 | trainable | DC-only | check A/BS compensation |
| A4 | hierarchical bounded, rho_p=0.20, rho_v=0.05 | frozen | DC-only | test scene/view/pixel parameterization |

Success criteria for this pilot:

- RGB metrics remain paired-control safe.
- Structure remains frozen: Far Accum, Water Accum, Gaussian count, object and
  boundary accumulation retention unchanged.
- Pixel residual mean and p95 decrease.
- View residual does not saturate at tanh bounds.
- Far-near CLR distance and far red deficit improve in raw and clamped J.
- No increase in G/B-only clipping or near chroma error.

If A1 reduces residuals but does not improve J color, then attenuation pixel
residual is a redundant parameter rather than a causal color-bias handle. If A4
outperforms A1, the next implementation should be hierarchical
scene/view/pixel attenuation spectrum, not a pixel-only bound.

## Current Status

No trainable attenuation module was added in this step.

No generated outputs are tracked by Git.
