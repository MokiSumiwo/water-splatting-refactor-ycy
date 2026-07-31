# Scalar TBAP Gradient Audit

Date: 2026-07-31
Branch: `refactor/core-framework`
Base commit: `c8db128 Add TBAP gradient audit`

## Goal

This pass followed the GPT-proposed correction after per-channel TBAP failed.

The previous audit showed that using learned per-channel medium transmission as an inverse-transmission preconditioner can amplify the wrong color direction. The new question was narrower:

```text
If RGB channels share one scalar preconditioner, can we selectively increase far Gaussian DC appearance gradients without changing color direction?
```

The scalar audit intentionally does not use:

```text
TMICA tail anchor
BS sensitivity
J Blue
water-color axis
per-channel learned transmission as RGB weights
```

## Implementation

Extended the default-off TBAP module in `water_splatting/water_splatting.py`:

```text
tbap_weight_mode:
  channel_transmission
  depth
  scalar_transmission
  median_transmission
  luma_transmission

tbap_support_mode:
  legacy
  object_far

tbap_support_top_fraction
tbap_depth_weight_strength
```

The scalar modes share one weight across RGB:

```text
depth:
  w = 1 + depth_weight_strength * q_far

scalar_transmission:
  T = exp(-depth * mean(beta_attn))

median_transmission:
  T = median(T_R, T_G, T_B)

luma_transmission:
  T_Y = Y(rgb_object) / Y(J_raw)
```

All scalar signals are detached. The forward render remains unchanged.

Extended `scripts/diagnostics/diagnose_tbap_gradient_audit.py`:

```text
--split train|eval
--image-indices
--weight-mode
--support-mode
--support-top-fraction
--depth-weight-strength
```

The diagnostic now also reports max RGB channel-ratio drift per Gaussian depth bin.

## Checkpoints

The plan requested M1 `step=10000` for both scenes. JapaneseGradens has that checkpoint. The formal IUI3 M1 run only retained `step-000014999.ckpt`, so IUI3 was audited at step 14999 and this limitation should be kept in mind.

```text
IUI3:
outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml
load-step 14999

JapaneseGradens:
outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml
load-step 10000
```

## Views

Used 4 fixed train views per scene:

```text
IUI3 train indices: 0, 6, 13, 20
JapaneseGradens train indices: 0, 5, 10, 15
```

Support was standardized as top 15% of:

```text
q_object * q_far
```

## Commands

S1 depth scalar:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_tbap_gradient_audit.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --output-dir renders/scalar_tbap_gradient_audit_20260731/iui3_s1_depth_l1 \
  --scene-name IUI3-RedSea \
  --load-step 14999 \
  --step-override 14999 \
  --split train \
  --image-indices 0,6,13,20 \
  --max-images 4 \
  --lambda-tbap 1.0 \
  --weight-mode depth \
  --support-mode object_far \
  --support-top-fraction 0.15 \
  --max-weight 2.0 \
  --far-depth-mid 0.65 \
  --far-depth-temp 0.15 \
  --depth-weight-strength 1.0
```

The same command pattern was run for:

```text
iui3_s2_scalarT_l1
iui3_s3_lumaT_l1
japanese_s1_depth_l1
japanese_s2_scalarT_l1
japanese_s3_lumaT_l1
```

Outputs:

```text
renders/scalar_tbap_gradient_audit_20260731/
```

## Gate

The revised scalar gate was:

```text
farthest 25% DC gradient: 1.5x to 2.5x
nearest 25% DC gradient: <10% change
total appearance gradient: <=1.5x
RGB channel-ratio drift: <5%
no NaN/Inf
```

## Results

| Case | Scene | Views | Mean appearance | Max appearance | Mean farthest DC | Max farthest DC | Max nearest DC | Mean channel drift | Max channel drift | Mean support | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| iui3_s1_depth_l1 | IUI3 | 4 | 1.441 | 1.607 | 1.159 | 1.271 | 1.000 | 0.099 | 0.201 | 0.127 | fail |
| iui3_s2_scalarT_l1 | IUI3 | 4 | 1.432 | 1.594 | 1.160 | 1.271 | 1.000 | 0.099 | 0.200 | 0.127 | fail |
| iui3_s3_lumaT_l1 | IUI3 | 4 | 1.442 | 1.600 | 1.160 | 1.271 | 1.000 | 0.099 | 0.200 | 0.127 | fail |
| japanese_s1_depth_l1 | JapaneseGradens | 4 | 1.275 | 1.455 | 1.143 | 1.200 | 1.000 | 0.089 | 0.109 | 0.114 | fail |
| japanese_s2_scalarT_l1 | JapaneseGradens | 4 | 1.274 | 1.454 | 1.145 | 1.200 | 1.000 | 0.090 | 0.108 | 0.114 | fail |
| japanese_s3_lumaT_l1 | JapaneseGradens | 4 | 1.274 | 1.454 | 1.145 | 1.200 | 1.000 | 0.090 | 0.108 | 0.114 | fail |

## Interpretation

All S1-S3 scalar modes fail before training.

The strongest IUI3 farthest-DC response is only:

```text
1.271x
```

but its total appearance gradient is already:

```text
1.594x to 1.607x
```

which exceeds the `<=1.5x` gate. Increasing loss weight would further violate total-gradient safety before reaching the desired far-DC range.

JapaneseGradens is safer on total appearance gradient:

```text
max appearance = 1.454x to 1.455x
```

but farthest DC remains too weak:

```text
max farthest DC = 1.200x
```

The scalar modes also do not fully preserve channel proportions in the resulting Gaussian DC gradients. Even with RGB-shared weights, the branch residual and visibility structure induce channel-ratio drift:

```text
IUI3 max drift:        ~20%
Japanese max drift:    ~11%
```

This means the assumption "scalar weighting preserves RGB gradient ratio" is not true at the Gaussian-parameter level, even though the pixel loss weights are scalar.

## Decision

Do not run C0-C4 500/1000-step pilots.

Reason:

```text
No scalar TBAP candidate passes the multi-view gradient gate.
```

This is a stronger negative result than the per-channel TBAP audit. It says:

```text
The current single-color Gaussian appearance path cannot selectively strengthen
far DC supervision enough without excessive total appearance gradients or
unintended channel-ratio drift.
```

## Conclusion

The scalar audit does not support TBAP-style gradient preconditioning as the next training module.

The likely remaining issue is not just missing far-gradient magnitude. It is the coupled parameterization:

```text
one Gaussian color variable must serve both underwater RGB fitting and exposed clear J,
while the medium field can absorb spectrum errors.
```

The recommended next step is not a stronger TBAP, TMICA, opacity/capacity control, or renderer rewrite. It is the plan's fourth step:

```text
view-level medium spectrum
+ bounded pixel spectral residual
```

Start with attenuation only:

```text
log beta_D,c(p) = m_D(p) + s_D,v,c + rho_D * tanh(r_D,c(p))
sum_c s_D,v,c = 0
sum_c r_D,c(p) = 0
rho_D in {0.10, 0.20, 0.30}
```

Before training, run the proposed offline approximation on frozen M1 predictions and require:

```text
RGB PSNR drop <= 0.03 dB
pixel spectral residual clearly shrinks
```

## Checks

```bash
/opt/anaconda3/envs/water_splatting/bin/python -m py_compile \
  water_splatting/water_splatting.py \
  scripts/diagnostics/diagnose_tbap_gradient_audit.py

bash -n scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh
```

Both passed before the full scalar audit runs.

## Follow-up: Offline Bounded Attenuation Spectrum

Because scalar TBAP failed, the plan's next step was to test whether a view-level attenuation spectrum plus bounded pixel residual can approximate the current M1 medium field without damaging underwater RGB.

Added:

```text
scripts/diagnostics/diagnose_native_attn_spectrum_bound.py
```

This is a no-training renderer-native intervention. It monkeypatches only `medium_attn` inside the Python rasterizer wrapper:

```text
log beta_D,c(p) = m_D(p) + s_D,v,c + rho_D * tanh(delta_c(p) / rho_D)
sum_c s_D,v,c = 0
sum_c delta_c(p) = 0
```

It leaves:

```text
Gaussian geometry
opacity
SH/DC color
medium RGB
medium BS
```

unchanged.

Important limitation:

```text
This intervention changes underwater RGB rendering only.
J is still the clear Gaussian output and therefore its dominance metrics remain unchanged.
The diagnostic is only a safety/identifiability check for a future trainable medium parameterization.
```

### Commands

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_native_attn_spectrum_bound.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --load-step 14999 \
  --test-mode inference \
  --max-images 4 \
  --output-dir renders/native_attn_spectrum_bound_20260731/iui3_m1 \
  --rhos 0.10 0.20 0.30

CUDA_VISIBLE_DEVICES=7 /opt/anaconda3/envs/water_splatting/bin/python \
  scripts/diagnostics/diagnose_native_attn_spectrum_bound.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --load-step 10000 \
  --test-mode inference \
  --max-images 3 \
  --output-dir renders/native_attn_spectrum_bound_20260731/japanese_m1_step10000 \
  --rhos 0.10 0.20 0.30
```

Outputs:

```text
renders/native_attn_spectrum_bound_20260731/iui3_m1/native_attn_spectrum_bound.json
renders/native_attn_spectrum_bound_20260731/japanese_m1_step10000/native_attn_spectrum_bound.json
```

### Offline Results

| Scene | rho | PSNR delta | LPIPS delta | RGB L1 vs original | Residual mean before -> after | Residual p95 before -> after | Approx violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| IUI3 | 0.10 | -0.0590 | +0.000210 | 0.000870 | 0.0627 -> 0.0432 | 0.1869 -> 0.0939 | 0.954 |
| IUI3 | 0.20 | -0.0098 | -0.000004 | 0.000303 | 0.0627 -> 0.0529 | 0.1869 -> 0.1249 | 0.944 |
| IUI3 | 0.30 | -0.0029 | -0.000011 | 0.000148 | 0.0627 -> 0.0566 | 0.1869 -> 0.1454 | 0.932 |
| JapaneseGradens | 0.10 | +0.0001 | +0.000353 | 0.000411 | 0.0355 -> 0.0310 | 0.0928 -> 0.0744 | 0.918 |
| JapaneseGradens | 0.20 | +0.0003 | +0.000072 | 0.000126 | 0.0355 -> 0.0341 | 0.0928 -> 0.0868 | 0.907 |
| JapaneseGradens | 0.30 | +0.0002 | +0.000029 | 0.000059 | 0.0355 -> 0.0348 | 0.0928 -> 0.0900 | 0.905 |

### Offline Interpretation

The bounded attenuation spectrum check passes the RGB-safety requirement for `rho=0.20` and `rho=0.30` on both audited scenes:

```text
IUI3 rho=0.20: PSNR delta -0.0098 dB
IUI3 rho=0.30: PSNR delta -0.0029 dB
Japanese rho=0.20: PSNR delta +0.0003 dB
Japanese rho=0.30: PSNR delta +0.0002 dB
```

`rho=0.10` is too restrictive for IUI3:

```text
PSNR delta -0.0590 dB
```

The residual does shrink, especially in IUI3:

```text
IUI3 rho=0.20 residual mean: 0.0627 -> 0.0529
IUI3 rho=0.20 residual p95:  0.1869 -> 0.1249
```

The spectral violation rate remains high because this parameterization intentionally does not impose hard RGB channel ordering. It only separates:

```text
pixel scalar attenuation strength
view-level spectrum
bounded per-pixel spectral residual
```

### Updated Decision

The scalar TBAP route should stop.

The bounded attenuation parameterization is viable enough to become the next trainable medium experiment, with `rho_D=0.20` as the first candidate and `rho_D=0.30` as the safer fallback.

Do not modify BS at the same time. The next module should only restructure attenuation:

```text
medium_attn = exp(pixel_strength + view_spectrum + bounded_pixel_residual)
```

and then evaluate whether constraining medium spectral degrees of freedom reduces the compensation loop with Gaussian clear color.
