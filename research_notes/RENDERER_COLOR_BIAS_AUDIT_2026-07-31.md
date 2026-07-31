# Renderer Color Bias Audit

Date: 2026-07-31
Branch: `refactor/core-framework`

## Goal

This audit pauses new 15k module tuning and investigates why far-distance `J` remains blue/green biased in frozen M1 checkpoints.

The tested hypotheses were:

1. Far clear-color supervision is hidden by underwater attenuation.
2. Per-Gaussian layer degradation differs materially from a surface-style degradation model.
3. Remaining tail transmittance is confused with incomplete surface coverage.
4. SH residuals or hard clamp amplify the visible clear-image color cast.
5. Learned medium spectra violate expected water-like channel ordering.

No training modules were changed in this phase.

## Implementation

Added:

```text
scripts/diagnostics/diagnose_renderer_color_bias.py
scripts/experiments/run_renderer_color_bias_audit.sh
```

The diagnostic loads frozen M1 checkpoints and records:

```text
layer RGB vs GT
surface-style approximate RGB vs GT
hybrid layer/surface approximate RGB vs GT
hit-conditioned tail-suppressed RGB vs GT
J medium-axis bias
blue/green-minus-red bias
depth variance
exposure gain
hidden blue/green clear energy
approximate Jensen direct/backscatter gaps
DC-only J
high-order SH residual
hard-clamp amplification
medium attn / bs / B_inf spectral ordering
```

Important limitation:

```text
The CUDA rasterizer does not expose per-Gaussian per-pixel weights.
Therefore A0 strict explicit per-Gaussian forward/backward equivalence and true Jensen gaps are not available in this script.
This audit uses exposed pixel summaries: J_raw, expected depth, depth variance, accumulation, final transmittance, medium attn/bs/rgb, and J_proxy equivalence.
```

## Commands

```bash
GPU=6 MAX_IMAGES=4 SAVE_IMAGES=1 COMPUTE_DC=1 \
  scripts/experiments/run_renderer_color_bias_audit.sh
```

Outputs:

```text
renders/renderer_color_bias_audit_20260731_renderer_color_bias/iui3_redsea_m1/renderer_color_bias_audit_summary.json
renders/renderer_color_bias_audit_20260731_renderer_color_bias/japanesegradens_redsea_m1/renderer_color_bias_audit_summary.json
renders/renderer_color_bias_audit_20260731_renderer_color_bias/renderer_color_bias_contact_sheet.png
logs/renderer_color_bias_audit_20260731_renderer_color_bias/
```

Checkpoints:

```text
IUI3:
outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/nerfstudio_models/step-000014999.ckpt

JapaneseGradens:
outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/nerfstudio_models/step-000014999.ckpt
```

## Key Metrics

### IUI3 M1

| Region | Layer L1 | Surface L1 | Hybrid L1 | Tail Supp L1 | J Bias | BG-Red | Tail Ratio | Exposure Gain | Hidden BG | Layer-Surface | SH Axis | Clamp Axis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Far | 0.01134 | 0.01139 | 0.01133 | 0.01118 | 0.01874 | 0.02199 | 0.2920 | 4.5892 | 0.00212 | 0.000565 | -0.01425 | 0.01372 |
| Far object | 0.01381 | 0.01389 | 0.01381 | 0.01372 | 0.04175 | 0.05638 | 0.0026 | 5.0409 | 0.02029 | 0.001168 | -0.02654 | 0.02261 |
| Far water | 0.00698 | 0.00704 | 0.00698 | 0.00697 | 0.01304 | 0.01132 | 0.4225 | 4.2888 | 0.00926 | 0.000076 | 0.00081 | 0.00000 |

Main far-region correlations with `J` medium-axis bias:

| Predictor | Far | Far object | Far water |
|---|---:|---:|---:|
| Hidden BG clear energy | 0.9067 | 0.8886 | 0.9839 |
| Layer-surface diff | 0.2469 | 0.0631 | 0.4918 |
| Exposure gain | 0.0036 | -0.3295 | -0.3328 |
| Tail ratio | -0.2429 | 0.2058 | -0.2286 |
| High-order SH | -0.1817 | -0.0030 | -0.1458 |
| Clamp delta | -0.0525 | -0.0446 | 0.0000 |

Spectral means:

| Region | Attn R/G/B | Attn Violation | BS R/G/B | BS Violation | B_inf R/G/B | B_inf Violation |
|---|---|---:|---|---:|---|---:|
| Far | 0.114 / 0.144 / 0.143 | 0.927 | 0.691 / 0.477 / 0.332 | 1.000 | 0.211 / 0.304 / 0.451 | 0.003 |
| Far object | 0.143 / 0.152 / 0.149 | 0.828 | 0.458 / 0.349 / 0.269 | 1.000 | 0.243 / 0.336 / 0.469 | 0.002 |
| Far water | 0.099 / 0.140 / 0.141 | 0.995 | 0.828 / 0.551 / 0.368 | 1.000 | 0.191 / 0.284 / 0.436 | 0.000 |

### JapaneseGradens M1

| Region | Layer L1 | Surface L1 | Hybrid L1 | Tail Supp L1 | J Bias | BG-Red | Tail Ratio | Exposure Gain | Hidden BG | Layer-Surface | SH Axis | Clamp Axis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Far | 0.00713 | 0.00711 | 0.00712 | 0.00764 | 0.03465 | 0.03170 | 0.0985 | 11.2643 | 0.02657 | 0.000170 | 0.00044 | 0.00000 |
| Far object | 0.00925 | 0.00929 | 0.00927 | 0.00942 | 0.07064 | 0.06774 | 0.0147 | 7.8482 | 0.05698 | 0.000433 | -0.00385 | 0.00000 |
| Far water | 0.00668 | 0.00668 | 0.00668 | 0.00670 | 0.01123 | 0.00965 | 0.1442 | 7.8688 | 0.00773 | 0.000027 | 0.00055 | 0.00000 |

Main far-region correlations with `J` medium-axis bias:

| Predictor | Far | Far object | Far water |
|---|---:|---:|---:|
| Hidden BG clear energy | 0.9831 | 0.9833 | 0.9780 |
| Layer-surface diff | 0.1258 | -0.1711 | -0.0607 |
| Exposure gain | 0.0165 | 0.1249 | 0.1497 |
| Tail ratio | -0.3012 | -0.1831 | -0.0874 |
| High-order SH | 0.4089 | 0.5293 | 0.8057 |
| Clamp delta | 0.0000 | 0.0000 | 0.0000 |

Spectral means:

| Region | Attn R/G/B | Attn Violation | BS R/G/B | BS Violation | B_inf R/G/B | B_inf Violation |
|---|---|---:|---|---:|---|---:|
| Far | 0.367 / 0.324 / 0.344 | 0.946 | 0.509 / 0.434 / 0.397 | 1.000 | 0.188 / 0.284 / 0.393 | 0.000 |
| Far object | 0.325 / 0.305 / 0.328 | 0.980 | 0.455 / 0.403 / 0.377 | 1.000 | 0.192 / 0.284 / 0.378 | 0.000 |
| Far water | 0.348 / 0.305 / 0.329 | 1.000 | 0.507 / 0.426 / 0.385 | 1.000 | 0.188 / 0.288 / 0.404 | 0.000 |

## Interpretation

### H1: Hidden clear-color energy is the strongest supported cause

Both scenes show very high correlation between hidden blue/green clear energy and `J` medium-axis bias:

```text
IUI3 far:              0.9067
IUI3 far object:       0.8886
IUI3 far water:        0.9839
Japanese far:          0.9831
Japanese far object:   0.9833
Japanese far water:    0.9780
```

Exposure gain magnitude is also large:

```text
IUI3 far exposure gain:        4.59x
Japanese far exposure gain:   11.26x
```

However, exposure gain alone is not spatially correlated enough to be the whole explanation. The better statement is:

```text
Far attenuation creates a large hidden clear-color space.
The pixels that actually become visibly biased are the ones where hidden blue/green clear energy is already present.
```

This supports a next module that corrects far clear appearance gradients or parameterization, not another stronger image-space blue/green penalty.

### H2: Layer-vs-surface degradation is not the primary current cause

The surface-style approximation changes RGB very little:

```text
IUI3 far layer-surface luma diff:        0.000565
Japanese far layer-surface luma diff:    0.000170
```

Surface/hybrid RGB does not materially improve GT error:

```text
IUI3 far: layer 0.01134, surface 0.01139, hybrid 0.01133
Japanese far: layer 0.00713, surface 0.00711, hybrid 0.00712
```

The phase-B gate from the plan was:

```text
far-object corr(layer-surface difference, color bias) >= 0.30
and surface/hybrid lower GT RGB error
```

This gate does not pass:

```text
IUI3 far object corr:        0.063
Japanese far object corr:   -0.171
```

Conclusion: do not prioritize a renderer rewrite to surface-conditioned degradation yet.

### H3: Tail confusion is secondary and scene-dependent

Tail ratio is high in open-water masks, but far-object tail ratio is small:

```text
IUI3 far object tail ratio:        0.0026
Japanese far object tail ratio:    0.0147
```

Hit-conditioned tail suppression slightly improves IUI3 far/far-object L1, but worsens Japanese:

```text
IUI3 far:        0.01134 -> 0.01118
IUI3 far object: 0.01381 -> 0.01372
Japanese far:    0.00713 -> 0.00764
Japanese object: 0.00925 -> 0.00942
```

Conclusion: tail/surface closure is not a shared root cause. It may help IUI3 locally but should not be the next global mechanism.

### H4: SH/clamp effects are real but scene-specific

IUI3 shows clamp amplification in far object:

```text
IUI3 far object clamp medium-axis delta: 0.0226
```

JapaneseGradens has no clamp amplification, but high-order SH correlates with bias:

```text
Japanese far SH corr:        0.4089
Japanese far object corr:   0.5293
Japanese far water corr:    0.8057
```

Conclusion:

```text
IUI3: hard clamp contributes to visible bias.
JapaneseGradens: high-order SH directionality contributes to spatial bias.
```

These are refinements, not the shared first cause.

### H5: Medium spectra are underconstrained

`B_inf` / medium RGB follows expected blue-green ordering, but attenuation and backscatter order are mostly inconsistent with simple water-like priors:

```text
IUI3 far attn violation:        92.7%
IUI3 far BS violation:         100.0%
Japanese far attn violation:   94.6%
Japanese far BS violation:     100.0%
```

The learned BS means are red-dominant in both scenes, for example:

```text
IUI3 far BS R/G/B:       0.691 / 0.477 / 0.332
Japanese far BS R/G/B:   0.509 / 0.434 / 0.397
```

Because real underwater optics can vary by scene and preprocessing, this should not immediately become a hard spectral loss. It does show that medium parameters remain highly underconstrained and can compensate for clear-color bias.

## Current Conclusion

The strongest shared cause of far-distance color cast is:

```text
attenuated underwater RGB leaves far clear appearance weakly supervised,
and medium-aligned hidden blue/green clear energy survives in J.
```

The current evidence does not support immediately changing to a surface-conditioned renderer. It also does not support another broad TMICA weight sweep. The next controlled experiment should isolate appearance and medium while freezing geometry/opacity:

```text
B0: frozen geometry/opacity M1 resume control
B2: layer renderer + transmission-balanced color-only loss
B4: low-weight TMICA under frozen geometry/opacity as a control
```

I would skip B1/B3 surface/hybrid for the next immediate run unless a future stricter per-Gaussian audit contradicts the surface-approx result.

## Next Recommended Step

Implement a short controlled optimization branch from M1 step 10000 to 14999:

```text
freeze means / scales / quats / opacities
disable densification
train Gaussian DC/SH + medium only
add transmission-balanced color-only auxiliary loss only to Gaussian appearance
```

First test:

```text
TB0: frozen control
TB1: gamma=0.5, w_max=3.0, far-object/support detached
TB2: gamma=0.75, w_max=4.0 only if TB1 improves J without PSNR loss
```

Do not implement renderer replacement or hard spectral ordering yet.

## Verification

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile scripts/diagnostics/diagnose_renderer_color_bias.py
bash -n scripts/experiments/run_renderer_color_bias_audit.sh
GPU=6 MAX_IMAGES=4 SAVE_IMAGES=1 COMPUTE_DC=1 scripts/experiments/run_renderer_color_bias_audit.sh
```

Generated artifact size:

```text
renders/renderer_color_bias_audit_20260731_renderer_color_bias: 111M
logs/renderer_color_bias_audit_20260731_renderer_color_bias: 60K
```

No generated outputs, renders, logs, masks, or checkpoints are tracked by Git.
