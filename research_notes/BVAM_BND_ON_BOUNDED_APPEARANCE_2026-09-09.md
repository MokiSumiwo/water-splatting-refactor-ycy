# BVAM on the BND-ON Mainline

Date: 2026-09-09  
Branch: `research/bvam-bnd-on`  
Environment: `water_splatting`

## Scope

This experiment evaluates only **BVAM: Bounded View-Adaptive Mixture of Appearance Experts**. No HCC, BARR, BGTR, BHTC, Water-Type MoE, or hard-view curriculum was implemented.

The fixed protocol was:

```text
M1 + OCMC + CICA
Auxiliary OFF
BND ON
RAOC OFF
seed = 42
source = OCMC C0 checkpoint at step 3000
matched continuation updates = 11999
final step = 14999
evaluation = local RGB metrics protocol
```

The two-scene screening used `JapaneseGradens-RedSea` and `Panama`. The source checkpoint and training camera split were shared across variants. The final metrics use the local protocol with RGB in `[0, 1]`, exact prediction/ground-truth pairing, no resize, crop, or color conversion.

## Mechanism

For Gaussian `g` and view direction `v`, the existing bounded SH path produces shared logits:

```text
s_shared(g, v) = SH3(features_dc_g, features_rest_g, v)
```

BVAM adds two bounded expert residual heads:

```text
delta_k(g, v) = delta_scale * tanh(SH1(bvam_delta_k_g, v))
s_k(g, v) = s_shared(g, v) + signed(delta_k(g, v))
c_k(g, v) = sigmoid(s_k(g, v))
```

The expert gate is a degree-1 view-direction SH function:

```text
pi(g, v) = softmax(SH1(bvam_gate_g, v) / temperature)
```

The rendered Gaussian color is the convex mixture:

```text
c_bvam(g, v) = sum_k pi_k(g, v) * c_k(g, v)
```

Each expert is strictly bounded by the sigmoid, and the mixture weights are nonnegative and sum to one. Therefore the BVAM Gaussian color remains in `(0, 1)` without a render-time clamp and does not restore the legacy `SH + 0.5` parameterization.

Initialization is continuous with BND: both residual heads start at zero and the gate starts at uniform `0.5 / 0.5`. A detached support mask allows low-support Gaussians to use zero residual and uniform mixing. The mask is propagated through Gaussian split, duplicate, and cull operations and is rebuilt after checkpoint reload from the training-camera bank.

The optional pseudo-target uses only detached training-camera statistics: visibility, transmission, photometric residual, renderer sensitivity, view spread, and a two-quantile `q` pattern. It does not backpropagate into geometry, medium, CICA, OCMC, or BVAM parameters. The pseudo-target loss is restricted to the BVAM gate and is not a new physical supervision signal.

## Variants

| Variant | Description |
| --- | --- |
| E0 | BND fixed baseline |
| E2 | BRASH fixed gate control |
| E3 | BVAM-K2, zero residual initialization, no pseudo-target loss |
| E4 | BVAM-K2 plus detached pseudo-target gate consistency |

All variants used the same `A2` continuation arm, so MDRR was off in this screening. E2 is included as the registered BRASH control requested by the experiment protocol; it is not part of the BVAM mechanism.

## Metrics

Metrics below are from the corrected final evaluation. BVAM E3/E4 were re-evaluated after restoring the detached BVAM support mask from the same training-camera bank.

### Per scene

| Scene | Variant | PSNR | SSIM | LPIPS | MSE | dPSNR vs E0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| JapaneseGradens-RedSea | E0 BND | 24.577732 | 0.891628 | 0.117865 | 0.005672582 | 0 |
| JapaneseGradens-RedSea | E2 BRASH | 24.664141 | 0.892106 | 0.117512 | 0.005458406 | +0.086409 |
| JapaneseGradens-RedSea | E3 BVAM | 24.566895 | 0.892494 | 0.117616 | 0.005570154 | -0.011070 |
| JapaneseGradens-RedSea | E4 BVAM + pseudo-target | 24.607768 | 0.892857 | 0.117183 | 0.005519434 | +0.030036 |
| Panama | E0 BND | 31.618036 | 0.948297 | 0.076780 | 0.000694134 | 0 |
| Panama | E2 BRASH | 31.584509 | 0.948470 | 0.076931 | 0.000700468 | -0.033527 |
| Panama | E3 BVAM | 31.591046 | 0.948504 | 0.077161 | 0.000699723 | -0.026990 |
| Panama | E4 BVAM + pseudo-target | 31.607717 | 0.947992 | 0.077156 | 0.000696597 | -0.010319 |

### Two-scene mean

| Variant | PSNR | SSIM | LPIPS | MSE | dPSNR vs E0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 BND | 28.097884 | 0.919963 | 0.097322 | 0.003183358 | 0 |
| E2 BRASH | 28.124325 | 0.920288 | 0.097222 | 0.003079437 | +0.026441 |
| E3 BVAM | 28.078971 | 0.920499 | 0.097389 | 0.003134939 | -0.018913 |
| E4 BVAM + pseudo-target | 28.107743 | 0.920424 | 0.097169 | 0.003108015 | +0.009859 |

E4 improves JapaneseGradens-RedSea by about `0.030 dB`, but decreases Panama by about `0.010 dB`. Its two-scene PSNR improvement is only `0.0099 dB`, below the `0.03 dB` screening criterion. E3 is below E0 in mean PSNR and has worse mean LPIPS.

## Boundedness and Safety

All completed variants satisfy `P(J > 1) = 0`.

| Scene | Variant | P(J > 1) | J p99 | tau_D p90 |
| --- | --- | ---: | ---: | ---: |
| JapaneseGradens-RedSea | E0 | 0 | 0.968501 | 0.927769 |
| JapaneseGradens-RedSea | E2 | 0 | 0.968183 | 0.912141 |
| JapaneseGradens-RedSea | E3 | 0 | 0.969153 | 0.936103 |
| JapaneseGradens-RedSea | E4 | 0 | 0.965290 | 0.902014 |
| Panama | E0 | 0 | 0.819504 | 0.911632 |
| Panama | E2 | 0 | 0.820928 | 0.916478 |
| Panama | E3 | 0 | 0.819434 | 0.914568 |
| Panama | E4 | 0 | 0.818885 | 0.903881 |

There were no NaN/Inf failures in the completed BVAM runs. The gradient routing audit passed for all BVAM runs: CICA remained restricted to `features_dc`, BVAM auxiliary gradients stayed in `bvam_delta`/`bvam_gate`, and no BVAM/CICA route escaped to geometry, medium, or the OCMC projector.

## BVAM Diagnostics

Initial detached pseudo-target statistics:

| Scene | support mean | support p50 | valid Gaussian fraction | pseudo confidence mean |
| --- | ---: | ---: | ---: | ---: |
| JapaneseGradens-RedSea | 3.862 | 4 | 0.545 | 0.171 |
| Panama | 3.620 | 3 | 0.576 | 0.144 |

Final training diagnostics:

| Scene | Variant | expert usage 1/2 | entropy | collapse fraction | delta norm p90 |
| --- | --- | ---: | ---: | ---: | ---: |
| JapaneseGradens-RedSea | E3 | 0.500 / 0.500 | 0.693147 | 1.000 | 0.06734 |
| JapaneseGradens-RedSea | E4 | 0.501 / 0.499 | 0.692627 | 0.613 | 0.06727 |
| Panama | E3 | 0.500 / 0.500 | 0.693147 | 1.000 | 0.07899 |
| Panama | E4 | 0.501 / 0.499 | 0.692832 | 0.716 | 0.08067 |

The gate remains almost exactly uniform in both BVAM variants. E3 therefore does not demonstrate useful expert routing; its two experts are effectively collapsed according to the recorded collapse statistic. E4 reduces the recorded collapse fraction but still has nearly uniform usage and does not yield a consistent RGB gain. The low pseudo-target confidence and only roughly half valid Gaussian population also make the pseudo-target an unreliable basis for a stronger claim.

## Equivalence Audit

The real CUDA audit is recorded in [bvam audit](../outputs/bvam_audit_20260909/JapaneseGradens-RedSea.json).

Results:

```text
BRASH E1 render equivalence: true
BVAM zero-initialized render equivalence: true
finite BND/BRASH/BVAM outputs: true
BVAM gate at initialization: [0.5, 0.5]
```

The BND-to-BRASH-E1 maximum absolute differences were at floating-point roundoff (`5.96e-8` for the principal RGB tensors). After the final strict open-interval sigmoid safeguard was added, the zero-initialized BVAM path remains numerically equivalent within the audit tolerance. Its largest reported difference was `4.17e-7` on `pred_image` and `clear_object_fullsh_raw`; the difference comes only from protecting saturated float32 sigmoid values from representing exactly `0` or `1`. E1 should therefore be interpreted as an optimization/topology control, not as a new representation-capacity gain.

## Visual Outputs

The contact sheets contain columns E0, E2, E3, E4 and rows underwater, native clear, and clear-raw display. They use one representative evaluation view per scene:

![JapaneseGradens-RedSea BVAM comparison](../renders/bvam_bnd_on_screening_20260909/contact_sheets/JapaneseGradens-RedSea_MTN_1098_E0-E4.png)

![Panama BVAM comparison](../renders/bvam_bnd_on_screening_20260909/contact_sheets/Panama_MTN_1529_E0-E4.png)

The retained individual render paths are listed in each variant's `render_manifest.csv`. The representative paths are:

```text
renders/bvam_bnd_on_screening_20260909/E0_bnd_fixed/JapaneseGradens-RedSea/step_14999/MTN_1098/A2/
renders/bvam_bnd_on_screening_20260909/E2_brash_fixed_gate/JapaneseGradens-RedSea/step_14999/MTN_1098/A2/
renders/bvam_bnd_on_screening_20260909/E3_bvam_k2/JapaneseGradens-RedSea/step_14999/MTN_1098/A2/
renders/bvam_bnd_on_screening_20260909/E4_bvam_pseudogt/JapaneseGradens-RedSea/step_14999/MTN_1098/A2/
renders/bvam_bnd_on_screening_20260909/E0_bnd_fixed/Panama/step_14999/MTN_1529/A2/
renders/bvam_bnd_on_screening_20260909/E2_brash_fixed_gate/Panama/step_14999/MTN_1529/A2/
renders/bvam_bnd_on_screening_20260909/E3_bvam_k2/Panama/step_14999/MTN_1529/A2/
renders/bvam_bnd_on_screening_20260909/E4_bvam_pseudogt/Panama/step_14999/MTN_1529/A2/
```

## Code and Experiment Assets

Implementation files:

- `water_splatting/bvam.py`
- `water_splatting/water_splatting.py`
- `scripts/experiments/run_direct_mdrr_cica_scene.py`
- `scripts/experiments/run_bvam_bnd_on.py`
- `scripts/experiments/re_evaluate_bvam_bnd_on.py`
- `scripts/experiments/summarize_bvam_bnd_on.py`
- `scripts/diagnostics/audit_bvam_equivalence.py`
- `tests/test_bvam.py`

Summary files:

- [screening metrics CSV](../outputs/bvam_bnd_on_screening_20260909/BVAM_SCREENING_METRICS.csv)
- [screening summary JSON](../outputs/bvam_bnd_on_screening_20260909/BVAM_SCREENING_SUMMARY.json)
- [BVAM equivalence audit](../outputs/bvam_audit_20260909/JapaneseGradens-RedSea.json)

Final checkpoint roots:

```text
outputs/bvam_bnd_on_screening_20260909/<scene>/E0_bnd_fixed/checkpoints/step-000014999.ckpt
outputs/bvam_bnd_on_screening_20260909/<scene>/E2_brash_fixed_gate/checkpoints/step-000014999.ckpt
outputs/bvam_bnd_on_screening_20260909_retry2/<scene>/E3_bvam_k2/checkpoints/step-000014999.ckpt
outputs/bvam_bnd_on_screening_20260909_retry2/<scene>/E4_bvam_pseudogt/checkpoints/step-000014999.ckpt
```

The corrected BVAM evaluations are recorded in each BVAM directory as `evaluation_restore_audit.json`, `evaluation_metrics.csv`, `per_view_metrics.csv`, `render_manifest.csv`, and `decomposition_safety.json`.

## Conclusion

BVAM preserved the BND boundedness guarantee and passed the intended gradient-routing and zero-state equivalence checks. However, this two-scene screening does **not** support retaining BVAM as the next mainline mechanism:

1. E3 has lower mean PSNR than the BND baseline and worse mean LPIPS.
2. E4 has only a small mean PSNR gain, below the screening threshold, and still decreases Panama PSNR/SSIM.
3. The gate remains nearly uniform, while E3 records complete expert collapse and E4 only partial recovery.
4. The pseudo-target has low confidence and does not produce consistent cross-scene gains.
5. The contact sheets show no clear, stable visual improvement over BND.

Therefore no four-scene BVAM validation was run. BVAM should remain an isolated, documented negative/diagnostic result. The current evidence points to insufficiently useful expert specialization and unreliable pseudo-target routing, rather than a failure of boundedness or a CICA/OCMC gradient leak. No additional third mechanism is proposed in this record.
