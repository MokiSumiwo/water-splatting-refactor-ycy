# BND Antialiased Rasterization Eligibility and Panama Test

Date: 2026-08-10

Branch: `research/m1-bounded-intrinsic`

Start HEAD: `c6063c679f7ef46f076d8f5d9656ce27b989f7e2`

Stage A diagnostic:

- `scripts/diagnostics/audit_bnd_aa_eligibility.py`

Stage B runner and diagnostic:

- `scripts/experiments/bnd_aa_panama_15k.sh`
- `scripts/diagnostics/summarize_bnd_aa_panama.py`

Outputs:

- Metrics: `outputs/bnd_aa_panama_20260810/`
- Visuals: `renders/bnd_aa_panama_20260810/`
- Visual index: `renders/bnd_aa_panama_20260810/VISUAL_COMPARE_INDEX.md`
- Geometry controls: `outputs/bnd_aa_panama_20260810/aa_geometry_metrics.csv`

No subjective clear-image correctness judgment was made in this experiment.

## 1. Motivation

HYPOTHESIS:

Panama's remaining BND-K1 RGB gap may have a screen-space / high-frequency / edge-aligned residual component. If so, switching only the existing rasterizer mode from `classic` to `antialiased` may recover part of the RGB gap while preserving bounded intrinsic appearance and low optical-depth decomposition proxies.

QUESTION:

Does antialiased rasterization recover Panama RGB by reducing high-frequency / edge-aligned residuals while preserving the bounded decomposition?

## 2. Current BND Evidence

EXPERIMENTAL FACT:

The current clean research baseline is `M1 + BND-K1`, not AOPT or staged medium-hold variants.

EXPERIMENTAL FACT:

Panama M1 and BND-K1 at nominal step 15000:

| Run | PSNR | SSIM | LPIPS | tau p90 | J p99 | P(J>1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 | 1.769849 | 1.311801 | 0.037758 |
| BND-K1 | 31.498353 | 0.948783 | 0.075521 | 0.999069 | 0.838911 | 0.000000 |

QUANTITATIVE CONCLUSION:

BND-K1 preserves the bounded intrinsic mechanism on Panama but leaves a `-0.810558 dB` PSNR gap relative to M1.

## 3. Why Staged Optimization Was Closed

EXPERIMENTAL FACT:

The previous Panama temporary medium-hold staged optimization test did not recover the gap relative to its matched restart control:

- `K1-RST PSNR = 31.495648`
- `STAGE PSNR = 31.427324`
- `STAGE vs K1-RST = -0.068324 dB`

QUANTITATIVE CONCLUSION:

Temporary object-medium decoupling was not supported as the next active direction, so this experiment does not continue medium hold or any GMVC-style alternating schedule.

## 4. SeaFree-GS Antialias Inspiration

CODE FACT:

Reference repository:

- Path: `/mnt/new/home_old/ycy/reference_repos/SeaFree-GS`
- Commit: `7797e97dae831029ac89ae9f37b3c3d69ec2cf6c`
- `git status --short`: clean

CODE FACT:

SeaFree-GS sets `rasterize_mode="antialiased"` in its main config and passes that mode to gsplat rasterization. SeaFree passes sigmoid opacity into gsplat and gsplat handles antialiased compensation internally for `rasterize_mode="antialiased"`.

## 5. Distinction From SeaFree Full Method

CODE FACT:

This experiment does not use SeaFree's SH0 setting, foreground-aware content loss, background-water supervision, coarse depth loss, distance scaling, pseudo-depth, or opacity schedule changes.

CODE FACT:

The only new Stage B variable is WaterSplatting's existing `rasterize_mode=antialiased`.

## 6. Existing Residual Frequency Audit

CODE FACT:

Stage A reused existing read-only outputs from:

- `outputs/bnd_object_medium_recomposition_20260810/frequency_residual_analysis.csv`

Frequency definition:

`BND residual = GT - I_BND`; low frequency is Gaussian blur of the residual at fixed sigma; high frequency is `residual - low`; energy fractions use squared RGB residual energy.

QUANTITATIVE RESULT:

| Scene | LF sigma3 | HF sigma3 | LF sigma9 | HF sigma9 |
| --- | ---: | ---: | ---: | ---: |
| Curasao | 0.817686 | 0.182314 | 0.688230 | 0.311770 |
| JapaneseGradens | 0.836102 | 0.163898 | 0.666037 | 0.333963 |
| IUI3 | 0.533552 | 0.466448 | 0.378375 | 0.621625 |
| Panama | 0.617868 | 0.382132 | 0.371264 | 0.628736 |

QUANTITATIVE RESULT:

- `HF_CONTROL_MEAN_sigma3 = 0.324381`
- `PANAMA_HF_RATIO_sigma3 = 1.178034`
- `HF_CONTROL_MEAN_sigma9 = 0.466697`
- `PANAMA_HF_RATIO_sigma9 = 1.347203`

## 7. Existing Edge-Alignment Audit

CODE FACT:

Stage A reused existing read-only outputs from:

- `outputs/bnd_object_medium_recomposition_20260810/edge_alignment_analysis.csv`

Edge definition:

Top 20 percent GT luminance gradient magnitude.

QUANTITATIVE RESULT:

| Scene | Edge enrichment | Edge correlation |
| --- | ---: | ---: |
| Curasao | 1.269140 | 0.086160 |
| JapaneseGradens | 2.292459 | 0.352795 |
| IUI3 | 2.134996 | 0.335710 |
| Panama | 2.561469 | 0.329556 |

QUANTITATIVE RESULT:

- `EDGE_CONTROL_MEAN = 1.702068`
- `PANAMA_EDGE_RATIO = 1.504916`

## 8. AA Eligibility Gate

CODE FACT:

The pre-registered Stage A gate was evaluated before Stage B training.

QUANTITATIVE RESULT:

`AA_ELIGIBLE = TRUE`

Reasons:

- Condition A edge-structured residual passed.
- Condition B sigma3 high-frequency excess passed.
- Condition B sigma9 high-frequency excess passed.
- Low-frequency rejection did not trigger.

QUANTITATIVE CONCLUSION:

Stage B was allowed by the pre-registered eligibility gate.

## 9. Current WaterSplatting AA Semantics

CODE FACT:

WaterSplatting supports `rasterize_mode = classic / antialiased`.

CODE FACT:

Classic opacity:

`opacities = torch.sigmoid(opacities_crop)`

Antialiased opacity:

`opacities = torch.sigmoid(opacities_crop) * comp[:, None]`

CODE FACT:

`comp` is returned by `underwater_rasterizer.project`. The AA switch changes screen-space effective opacity before `underwater_rasterizer.rasterize`; it does not directly redefine Gaussian color, bounded `J`, `beta_D`, `beta_B`, `medium_rgb`, or tied `B_inf`.

## 10. SeaFree vs WaterSplatting AA Comparison

CODE FACT:

`AA_SEMANTICS_MATCH = STRUCTURALLY_SIMILAR`

INFERENCE:

Both implementations use screen-space opacity compensation for antialiased rasterization. This is a structural comparison only; no line-by-line equivalence is claimed.

## 11. Stage B Experiment Definition

EXPERIMENTAL FACT:

Runs:

| Run | Role | Source | Intrinsic mode | Rasterize mode | Seed | SH | Loaded step |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| M1 | reference | reused | legacy | classic | 42 | 3 | 14999 |
| BND-K1 | bounded classic control | reused | bounded_sh3 | classic | 42 | 3 | 14999 |
| BND-AA | bounded antialiased candidate | new 0->15k | bounded_sh3 | antialiased | 42 | 3 | 14999 |

EXPERIMENTAL FACT:

All runs use Panama eval views:

`MTN_1539; MTN_1529; MTN_1547`

EXPERIMENTAL FACT:

All Stage B runs use:

- `medium_context_mode = dir_xy_camera`
- `b_inf_mode = tied`
- `infinite_water_enabled = False`
- `sh_degree = 3`

## 12. Initialization Audit

QUANTITATIVE RESULT:

Classic and antialiased initialized from the same Panama BND-AA config with seed 42 have exact parameter equality:

| Parameter | Max abs diff | Mean abs diff | Match |
| --- | ---: | ---: | --- |
| features_dc | 0.0 | 0.0 | TRUE |
| features_rest | 0.0 | 0.0 | TRUE |
| means | 0.0 | 0.0 | TRUE |
| scales | 0.0 | 0.0 | TRUE |
| quats | 0.0 | 0.0 | TRUE |
| opacities | 0.0 | 0.0 | TRUE |
| medium_mlp.tcnn_encoding.params | 0.0 | 0.0 | TRUE |

QUANTITATIVE CONCLUSION:

The BND-AA run starts from the same initialized Gaussian and medium parameter values as the classic variant under the same seed.

## 13. Forward Smoke Audit

QUANTITATIVE RESULT:

On initialized parameters and the same camera (`MTN_1539`), toggling classic vs AA produced finite outputs with no NaN/Inf in the audited tensors.

QUANTITATIVE RESULT:

Selected initial forward differences, AA minus classic:

| Output | Max abs diff | Mean abs diff |
| --- | ---: | ---: |
| pred_image | 0.002531 | 0.000032 |
| accumulation | 0.002037 | 0.000086 |
| depth | 0.026132 | 0.001333 |
| direct_object_signal | 0.002733 | 0.000041 |
| rgb_medium | 0.000737 | 0.000023 |
| clear_object_fullsh_raw | 0.003855 | 0.000067 |
| transmission | 0.002437 | 0.000107 |
| tau_D | 0.030486 | 0.000838 |

QUANTITATIVE CONCLUSION:

AA has a finite forward effect at initialization, as expected for a render intervention.

## 14. RGB Trajectory

QUANTITATIVE RESULT:

| Run | Step | PSNR | SSIM | LPIPS | MSE | Gaussian count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K1 | 1000 | 22.563314 | 0.688374 | 0.436505 | 0.005638 | 60112 |
| K1 | 3000 | 20.869651 | 0.597971 | 0.440031 | 0.008590 | 566332 |
| K1 | 5000 | 26.459814 | 0.822805 | 0.252951 | 0.002286 | 998668 |
| K1 | 8000 | 31.008972 | 0.943347 | 0.086439 | 0.000801 | 1223420 |
| K1 | 10000 | 31.154767 | 0.945652 | 0.082064 | 0.000776 | 1219898 |
| K1 | 13000 | 31.540089 | 0.949397 | 0.074719 | 0.000712 | 1183679 |
| K1 | 15000 | 31.498353 | 0.948783 | 0.075521 | 0.000714 | 1177886 |
| AA | 1000 | 23.078799 | 0.673193 | 0.396010 | 0.004963 | 74261 |
| AA | 3000 | 21.285377 | 0.507916 | 0.345449 | 0.007550 | 1379877 |
| AA | 5000 | 27.585084 | 0.827434 | 0.219774 | 0.001764 | 1929213 |
| AA | 8000 | 31.494855 | 0.946661 | 0.088079 | 0.000712 | 1969439 |
| AA | 10000 | 31.687296 | 0.949237 | 0.081904 | 0.000684 | 1988592 |
| AA | 13000 | 31.822493 | 0.948979 | 0.080088 | 0.000664 | 1972549 |
| AA | 15000 | 31.802683 | 0.948218 | 0.080714 | 0.000663 | 1969585 |

QUANTITATIVE RESULT:

Final RGB:

| Run | PSNR | SSIM | LPIPS | MSE |
| --- | ---: | ---: | ---: | ---: |
| M1 | 32.308910 | 0.949487 | 0.073979 | 0.000595 |
| K1 | 31.498353 | 0.948783 | 0.075521 | 0.000714 |
| AA | 31.802683 | 0.948218 | 0.080714 | 0.000663 |

QUANTITATIVE RESULT:

- `AA_PSNR_GAIN = +0.304330 dB` versus K1.
- `GLOBAL_MSE_GAP_RECOVERY = 0.427705`.
- `RGB_SAFETY = FALSE` versus M1 because SSIM and LPIPS fail the pre-registered M1-relative thresholds.

## 15. Decomposition Retention

QUANTITATIVE RESULT:

Final canonical decomposition metrics:

| Run | tau p90 | J p99 | P(J>1) | P(T<0.1) | beta_D mean | T mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1.769849 | 1.311801 | 0.037758 | 0.007454 | 0.396922 | 0.384434 |
| K1 | 0.999069 | 0.838911 | 0.000000 | 0.000477 | 0.143104 | 0.685939 |
| AA | 0.652180 | 0.813754 | 0.000000 | 0.000000 | 0.118056 | 0.736802 |

QUANTITATIVE RESULT:

`TAU_BENEFIT_RETENTION = 1.450050`

QUANTITATIVE CONCLUSION:

AA does not regress the canonical tau/J decomposition proxies; it further lowers tau p90 and keeps `P(J>1)=0` under the bounded parameterization.

## 16. Boundary Safety

QUANTITATIVE RESULT:

Final bounded-color boundary metrics:

| Run | c p99 | P(c>0.99) | P(|s|>5) | P(|s|>8) | sigmoid derivative p10 | sigmoid derivative p50 | BOUNDARY_ESCAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| K1 | 1.000000 | 0.017517 | 0.017149 | 0.015563 | 0.137411 | 0.210403 | FALSE |
| AA | 1.000000 | 0.064198 | 0.062464 | 0.056722 | 0.103075 | 0.211398 | TRUE |

QUANTITATIVE CONCLUSION:

AA triggers the pre-registered `BOUNDARY_ESCAPE` flag because both `P(c>0.99)` and `P(|s|>5)` exceed 0.05.

## 17. Frequency / Edge Residual

QUANTITATIVE RESULT:

Final aggregate frequency metrics:

| Run | HF sigma3 | HF energy sigma3 | HF sigma9 | HF energy sigma9 |
| --- | ---: | ---: | ---: | ---: |
| K1 | 0.382132 | 1449.377889 | 0.628736 | 2613.198079 |
| AA | 0.390054 | 1395.052775 | 0.622940 | 2418.624430 |

QUANTITATIVE RESULT:

- `HF_RESIDUAL_REDUCTION_sigma3 = 0.037482`
- `HF_RESIDUAL_REDUCTION_sigma9 = 0.074458`

QUANTITATIVE RESULT:

Final aggregate edge metrics:

| Run | Edge enrichment | Edge correlation | Edge residual energy |
| --- | ---: | ---: | ---: |
| K1 | 2.561470 | 0.329556 | 2336.312785 |
| AA | 2.497786 | 0.322856 | 2124.738037 |

QUANTITATIVE RESULT:

`EDGE_RESIDUAL_REDUCTION = 0.090559`

QUANTITATIVE CONCLUSION:

AA reduces absolute high-frequency and edge residual energies, but the reductions are below the 20 percent mechanism threshold.

## 18. High-J Region

CODE FACT:

The high-J mask is fixed from M1: object support and `max_channel(J_M1) > 1`.

QUANTITATIVE RESULT:

M1-defined `J>1` aggregate:

| Run | MSE | Mean abs residual L1 |
| --- | ---: | ---: |
| M1 | 0.002618 | 0.092595 |
| K1 | 0.004818 | 0.127099 |
| AA | 0.004222 | 0.119812 |

QUANTITATIVE RESULT:

`HIGH_J_MSE_GAP_RECOVERY = 0.271191`

QUANTITATIVE CONCLUSION:

AA recovers part of the K1 high-J-region MSE gap under the fixed M1 high-J mask.

## 19. Control Regions

QUANTITATIVE RESULT:

M1-defined `J<=1` aggregate:

| Run | MSE | Mean abs residual L1 |
| --- | ---: | ---: |
| M1 | 0.000493 | 0.036839 |
| K1 | 0.000498 | 0.037324 |
| AA | 0.000472 | 0.037574 |

QUANTITATIVE RESULT:

`LOW_J_DAMAGE = -0.000026`

QUANTITATIVE RESULT:

GT brightness Q5 aggregate:

| Run | MSE | Mean abs residual L1 |
| --- | ---: | ---: |
| M1 | 0.001682 | 0.067424 |
| K1 | 0.002243 | 0.077978 |
| AA | 0.002011 | 0.075068 |

QUANTITATIVE RESULT:

Bottom20 image-y aggregate:

| Run | MSE | Mean abs residual L1 |
| --- | ---: | ---: |
| M1 | 0.001263 | 0.057980 |
| K1 | 0.001604 | 0.064290 |
| AA | 0.001516 | 0.063817 |

## 20. Gaussian Population

QUANTITATIVE RESULT:

| Step | K1 Gaussian count | AA Gaussian count |
| ---: | ---: | ---: |
| 1000 | 60112 | 74261 |
| 3000 | 566332 | 1379877 |
| 5000 | 998668 | 1929213 |
| 8000 | 1223420 | 1969439 |
| 10000 | 1219898 | 1988592 |
| 13000 | 1183679 | 1972549 |
| 15000 | 1177886 | 1969585 |

INFERENCE:

AA substantially changes the Gaussian population trajectory under the unchanged densification rules. This is a downstream effect of the single rasterization-mode intervention, not a separate experimental variable.

## 21. Opacity Compensation

QUANTITATIVE RESULT:

Final opacity statistics averaged over the three eval views:

| Run | Raw opacity mean | Raw opacity p50 | Raw opacity p99 | AA comp mean | AA comp p50 | AA comp p99 | Effective opacity mean | Effective opacity p50 | Effective opacity p99 | Collapse | Extreme |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| K1 | 0.728173 | 0.780767 | 0.999751 | 0.655232 | 0.718755 | 0.987132 | 0.728173 | 0.780767 | 0.999751 | FALSE | TRUE |
| AA | 0.882713 | 0.970437 | 0.999497 | 0.446566 | 0.403271 | 0.979339 | 0.382015 | 0.332418 | 0.955666 | FALSE | FALSE |
| M1 | 0.716369 | 0.766334 | 0.999639 | 0.656540 | 0.718775 | 0.986562 | 0.716369 | 0.766334 | 0.999639 | FALSE | TRUE |

EXPERIMENTAL FACT:

For `classic` runs, the compensation factor is audited but not applied to effective opacity. For the AA run, effective opacity is raw sigmoid opacity multiplied by the compensation factor.

## 22. Geometry Controls

CODE FACT:

Gaussian scale statistics use `exp(model.scales)`, matching the scale value passed to rasterization.

QUANTITATIVE RESULT:

Final geometry controls:

| Run | Depth mean | Depth p50 | Depth p90 | Depth p99 | Scale mean | Scale p50 | Scale p90 | Scale p99 | Gaussian means finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M1 | 2.699353 | 2.281872 | 4.691591 | 6.151975 | 0.001453 | 0.000795 | 0.002825 | 0.011846 | TRUE |
| K1 | 2.766760 | 2.299029 | 4.873890 | 7.022541 | 0.001543 | 0.000793 | 0.002932 | 0.013251 | TRUE |
| AA | 2.694103 | 2.307909 | 4.509810 | 6.338000 | 0.001324 | 0.000512 | 0.002585 | 0.013066 | TRUE |

## 23. Per-View Metrics

QUANTITATIVE RESULT:

AA vs K1 per-view RGB deltas:

| View | Delta PSNR | Delta SSIM | Delta LPIPS |
| --- | ---: | ---: | ---: |
| MTN_1539 | +0.372797 | -0.000997 | +0.004331 |
| MTN_1529 | +0.087440 | -0.000977 | +0.006190 |
| MTN_1547 | +0.452753 | +0.000278 | +0.005059 |

QUANTITATIVE RESULT:

- Improved views by PSNR: 3
- Degraded views by PSNR: 0

## 24. Recomposition Control

QUANTITATIVE RESULT:

Final M1-referenced recomposition metrics:

| Run | C_direct | C_medium | C_cross | Recomp efficiency | Mean abs DeltaI L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| K1 | 0.004331 | 0.004166 | -0.008379 | 0.928829 | 0.026861 |
| AA | 0.005910 | 0.005603 | -0.011445 | 0.936001 | 0.027986 |

QUANTITATIVE RESULT:

MSE attribution closure errors are below `1e-9` for both K1 and AA aggregate rows.

QUANTITATIVE CONCLUSION:

The direct/medium additive attribution remains numerically closed for the final comparisons.

## 25. Final Classification

QUANTITATIVE RESULT:

| Flag | Value |
| --- | --- |
| STRONG_AA_RECOVERY | FALSE |
| PARTIAL_AA_RECOVERY | FALSE |
| NO_AA_RECOVERY | FALSE |
| AA_RGB_ONLY_GAIN | FALSE |
| AA_DECOMPOSITION_REGRESSION | TRUE |
| AA_HARMFUL | TRUE |

QUANTITATIVE RESULT:

`MECHANISM_SUPPORT = NOT_SUPPORTED`

QUANTITATIVE CONCLUSION:

AA provides a measurable RGB gain over K1 and lowers tau/J proxies, but it does not satisfy the pre-registered success mechanism because RGB safety fails versus M1, boundary escape is triggered, and HF/edge residual reductions are below 20 percent.

INFERENCE:

Under the current gate definitions, the result is not evidence that antialiased rasterization cleanly resolves Panama's residual by reducing high-frequency / edge-aligned residuals while preserving all bounded safety properties.

## 26. Visual Assets

EXPERIMENTAL FACT:

Visual contact sheets were generated at:

- Underwater: `renders/bnd_aa_panama_20260810/contact_sheet_underwater_m1_k1_aa.png`
- Clear raw display clamp01: `renders/bnd_aa_panama_20260810/contact_sheet_clear_raw_m1_k1_aa.png`
- Residual and excess: `renders/bnd_aa_panama_20260810/contact_sheet_residual_m1_k1_aa.png`
- High-frequency residual: `renders/bnd_aa_panama_20260810/contact_sheet_high_frequency_residual_k1_aa.png`
- Edge residual overlay: `renders/bnd_aa_panama_20260810/contact_sheet_edge_residual_k1_aa.png`
- Fixed M1 high-J mask overlay: `renders/bnd_aa_panama_20260810/contact_sheet_high_j_mask_k1_aa.png`
- Direct object delta: `renders/bnd_aa_panama_20260810/contact_sheet_direct_k1_aa_delta.png`
- Medium delta: `renders/bnd_aa_panama_20260810/contact_sheet_medium_k1_aa_delta.png`

EXPERIMENTAL FACT:

Visual manifest and index:

- `renders/bnd_aa_panama_20260810/manifest.json`
- `renders/bnd_aa_panama_20260810/manifest.csv`
- `renders/bnd_aa_panama_20260810/VISUAL_COMPARE_INDEX.md`

## 27. Next Single-Factor Recommendation

RECOMMENDATION:

Run a BND-v2 appearance representation diagnostic/test as the next single-factor experiment.

RATIONALE:

AA produced a PSNR gain but did not satisfy RGB safety or the screen-space HF/edge mechanism gate, and it triggered bounded-color boundary escape. This points more directly to bounded appearance representation pressure than to a clean screen-space rasterization fix.

UNVERIFIED HYPOTHESIS:

A revised bounded appearance representation may reduce Panama's localized fitting deficit without relying on larger Gaussian population changes or boundary saturation.
