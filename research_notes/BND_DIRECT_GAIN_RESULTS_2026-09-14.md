# Frozen BND Direct-Light Gain Results

The registered protocol is in
`research_notes/BND_DIRECT_GAIN_PROTOCOL_2026-09-14.md`. The run used the
frozen CRCC E0 BND-ON checkpoints at step 14999, three continuation seeds, all
four scenes, and all 13 official test views. No source asset was overwritten.

## Engineering checks

The zero-gain path reproduced the original underwater output. Fixed accepted
Gaussian contributions agreed with the native renderer within `6e-7` RGB in
the sampled checks. Native clear output, accumulation, depth,
final-transmittance, source checkpoint hash and frozen model hash were
preserved. The auxiliary modified-color call was never used as clear output.
The selected adapter added roughly 0.17--0.28 GiB over the frozen native
render in the recorded views. These checks establish implementation fidelity,
not clear-image correctness.

## Step 1: error diagnosis

The output contains per-view proxy masks and regional metrics in
`outputs/bnd_direct_gain_20260914/region_metrics.csv`, plus residual
decompositions in `decomposition.csv`. The masks use common clamped GT,
accumulation/transmission thresholds and raw historical OFF overflow; they are
not semantic object labels. They show that error is often concentrated in
object-proxy boundaries and block-mean/achromatic components, but this is a
descriptive decomposition and does not prove a lighting cause.

## Step 2: frozen gains

The validation gate selected the spatial quadratic gain as the best candidate,
but rejected promotion because only Curasao and IUI3 met the per-scene gain
threshold. Validation macro PSNR was 36.63257 versus 36.53487 for B0
(+0.09770 dB); all three seeds improved and LPIPS did not worsen. Japanese
Gradens and Panama selected the identity checkpoint. This is adapter
generalization over cameras already seen by the base model, not an unseen-base
test.

Official test results use the registered selection and were not used to select
anything:

| Scene | B0 PSNR | Selected gain PSNR | Delta | Selected arm |
| --- | ---: | ---: | ---: | --- |
| JapaneseGradens-RedSea | 24.738506 | 24.738506 | +0.000000 | identity |
| Panama | 31.466801 | 31.466801 | +0.000000 | identity |
| Curasao | 32.596074 | 32.887641 | +0.291567 | spatial |
| IUI3-RedSea | 30.613711 | 30.613711 | +0.000000 | identity |
| Macro | 29.853773 | 29.926665 | +0.072893 | per-scene selected |

The table above is per-scene selection. The raw arm macro values are B0
29.853773, scalar 29.894727 and spatial 29.925047 dB; they must not be
reported as a single deployable global model because the validation policy
selected identity for JapaneseGradens, Panama and IUI3. Exact per-view and
macro values are in `outputs/bnd_direct_gain_20260914/summary.json` and
`scene_means.csv`.

The fixed-seed comparison sheets are available for each scene as underwater,
clear, error and change JPEGs. Clear sheets are expected to be identical;
brighter or darker clear pixels are not evidence of improved clear recovery.

## Step 3: teacher

The GT-plus-historical-OFF teacher comparison was not triggered. The promotion
gate failed before teacher fitting because the gain did not improve at least
three scenes by 0.01 dB. This avoids spending teacher cost after a capacity
diagnostic that did not generalize across scenes. Historical OFF remains a
non-paired reference and cannot establish a causal BND-ON/BND-OFF gap.

## Interpretation and next experiment

The direct-light mismatch hypothesis has a limited positive signal: low-
capacity spatial gains help Curasao validation and its official test views,
while preserving clear rendering. It does not explain JapaneseGradens or
Panama, and a global spatial adapter is not a stable four-scene solution.
The next experiment should therefore use a **scene-calibrated, transmission-
gated gain** with the same frozen renderer and a strict identity fallback:
fit only pixels with non-saturated direct support and low regional ambiguity,
bound the gain more tightly, and select per-scene adapters from validation.
This should be registered as a new diagnostic, with no teacher, geometry,
color, exposure or densification changes. If its validation gate again fails,
freeze direct-gain refinement and focus on adding independent clear
supervision or accepting BND-ON/BND-OFF as different objectives.
