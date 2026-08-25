# BND-MIC-CAUSAL-IUI3

## CODE FACT
The MIC prototype is implemented behind `WaterSplattingModelConfig.medium_identifiability_enabled`; the default remains `False` and `medium_identifiability_weight=0.0`.

`WaterSplattingModel._attach_medium_identifiability_outputs()` attaches `outputs["medium_raw"]` only when MIC is enabled and the medium field exposes raw pre-activation output. `WaterSplattingModel._medium_identifiability_loss()` reads `outputs["medium_raw"][..., 6:9]`, the raw pre-softplus beta_D channels, and penalizes variance around a stop-gradient per-channel mean:

```text
L_MIC = mean((z_beta_D - stopgrad(mean(z_beta_D)))^2)
```

`get_loss_dict()` adds `medium_identifiability_weight * L_MIC` only when the MIC flag is active, the weight is nonzero, and the optional step bounds include the current step. The causal driver is `scripts/experiments/run_bnd_mic_causal_iui3.py`; it restores pipeline, optimizer, scheduler, scaler, RNG state, and an explicit camera sequence from the common BND@3000 checkpoint.

## CONFIG FACT
Both arms use `bounded_sh3`, SH degree 3, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, and `infinite_water_enabled=False`.
C0 has MIC disabled. C1 uses `lambda_mic=1.5118506741538569` with target `beta_D_raw_variance` for every continuation update from 3001 through the final step.
No CB-FG, CB-BG, BAP, UNORM, LOSSRESP, CDEPTH, OMVC, depth prior, depth residual, depth-aware alpha, or medium-context removal is enabled.

The fixed coefficient was selected before this causal run by the registered rule:

```text
lambda_mic = 0.5 * ||grad RGB||_medium / ||grad raw MIC||_medium
```

with the preflight values `||grad RGB||_medium=0.585562`, `||grad raw MIC||_medium=0.193658`, giving `lambda_mic=1.5118506741538569`.

## EXPERIMENTAL FACT
Scene is `IUI3-RedSea`. Both arms start from:

```text
outputs/dewater_bounded_sh3_cross_scene_20260808/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000/water-splatting/dewater_bnd_cross_scene_iui3_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_iui3_bnd_g1p00/nerfstudio_models/step-000003000.ckpt
```

The stored checkpoint step is 3000, the start Gaussian count is 297659, and the formal final checkpoint is 14999 for nominal 15000. `START_STATE_EQUIVALENCE=True`, with zero max forward difference across rendered RGB, depth, accumulation, medium_rgb, medium_bs, medium_attn, b_inf, beta_B, beta_D, raw z_med, and main RGB loss. Model parameters, optimizer state, scheduler state, scaler state, and RNG state also matched.

The paired camera sequence covers 11999 continuation updates from absolute steps 3001 through 14999. `CAMERA_SEQUENCE_MATCH=True`, mismatch count is 0, and the sequence SHA256 is `c44cf6552eff45bbcb00b1873e281ef2876ee3edec5a81f7fb2026ee3d62ce7b`.

Runtime environment in the formal manifest:

```text
CONDA_ENV=water_splatting
PYTHON_PATH=/opt/anaconda3/envs/water_splatting/bin/python
TORCH_VERSION=2.1.2+cu118
CUDA_VISIBLE_DEVICES=6
physical GPU=6
torch logical GPU=0
GPU=NVIDIA GeForce RTX 3080
```

Outputs are written under `outputs/bnd_mic_causal_iui3_20260825/` and logs under `logs/bnd_mic_causal_iui3_20260825/`. They are intentionally not committed.

## QUANTITATIVE RESULT
Final eval delta C1-C0: PSNR `-0.229170` dB, SSIM `-0.000753`, LPIPS `-0.000535`, MSE `-0.00000477`.
Raw beta_D contextual variance reduced on M_SAFE: `False`; steps `[5000]`.
Aggregate identifiability improved on M_SAFE: `True`; steps `[5000, 10000, 14999]`.
Weak-mode natural variation reduced on M_SAFE: `False`; steps `[10000]`.
Decomposition safety intact: `True`.

### RGB Summary

| split | step | C0 PSNR | C1 PSNR | dPSNR | C0 SSIM | C1 SSIM | dSSIM | C0 LPIPS | C1 LPIPS | dLPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 3000 | 21.390 | 21.390 | +0.000 | 0.634845 | 0.634845 | +0.000000 | 0.384129 | 0.384129 | +0.000000 |
| train | 5000 | 27.302 | 27.418 | +0.116 | 0.796904 | 0.796656 | -0.000248 | 0.294745 | 0.295965 | +0.001220 |
| train | 8000 | 34.733 | 34.235 | -0.499 | 0.939134 | 0.938339 | -0.000795 | 0.172704 | 0.173548 | +0.000844 |
| train | 10000 | 35.081 | 34.484 | -0.597 | 0.940721 | 0.939996 | -0.000725 | 0.168703 | 0.169289 | +0.000585 |
| train | 13000 | 36.457 | 35.775 | -0.682 | 0.949590 | 0.949095 | -0.000495 | 0.155853 | 0.156276 | +0.000423 |
| train | 14999 | 36.679 | 36.101 | -0.578 | 0.950800 | 0.950439 | -0.000361 | 0.153977 | 0.154144 | +0.000167 |
| eval | 3000 | 20.621 | 20.621 | +0.000 | 0.595353 | 0.595353 | +0.000000 | 0.403602 | 0.403602 | +0.000000 |
| eval | 5000 | 25.633 | 25.806 | +0.174 | 0.764244 | 0.764088 | -0.000156 | 0.306835 | 0.309845 | +0.003010 |
| eval | 8000 | 30.294 | 30.139 | -0.155 | 0.901631 | 0.900971 | -0.000660 | 0.192056 | 0.192754 | +0.000698 |
| eval | 10000 | 30.432 | 30.250 | -0.182 | 0.901314 | 0.900144 | -0.001170 | 0.189102 | 0.189758 | +0.000656 |
| eval | 13000 | 30.855 | 30.752 | -0.102 | 0.910841 | 0.910054 | -0.000787 | 0.179112 | 0.179008 | -0.000104 |
| eval | 14999 | 30.961 | 30.732 | -0.229 | 0.911200 | 0.910447 | -0.000753 | 0.178439 | 0.177904 | -0.000535 |

Final eval PSNR improved on 2/4 held-out views: `MTN_5894` (+0.217 dB) and `MTN_5928` (+0.058 dB). It degraded on `MTN_5903` (-0.245 dB) and `MTN_5911` (-0.947 dB). The registered RGB safety rule failed because final eval dPSNR was below -0.05 dB.

### Primary MIC Mechanism Metric

M_SAFE raw beta_D contextual variance:

| step | C0 raw var | C1 raw var | delta | ratio |
|---:|---:|---:|---:|---:|
| 3000 | 0.107037 | 0.107037 | +0.000000 | 1.000 |
| 5000 | 0.064597 | 0.060930 | -0.003666 | 0.943 |
| 8000 | 0.054282 | 0.125591 | +0.071309 | 2.314 |
| 10000 | 0.053631 | 0.171323 | +0.117692 | 3.194 |
| 13000 | 0.056855 | 0.220714 | +0.163859 | 3.882 |
| 14999 | 0.060102 | 0.246790 | +0.186688 | 4.106 |

The direct target was therefore not stably reduced. Activated beta_D variance was reduced at 5k but became larger than C0 from 8k onward.

### Aggregate Identifiability

M_SAFE structured aggregate Jacobian:

| step | C0 sigma_min/sigma_max | C1 sigma_min/sigma_max | ratio | C0 condition | C1 condition | condition ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 0.000291 | 0.000511 | 1.754 | 3434.393 | 1958.133 | 0.570 |
| 10000 | 0.001604 | 0.001865 | 1.163 | 623.538 | 536.163 | 0.860 |
| 14999 | 0.001938 | 0.004245 | 2.191 | 516.105 | 235.573 | 0.456 |

The weakest mode remains `WEAK_MODE_BETAD` with beta_D energy above 0.999996 at all audited C1 M_SAFE checkpoints.

### Counterfactual Weak-Mode Sensitivity

M_SAFE counterfactual epsilon is fixed at 0.25:

| step | C0 vmin/vmax RGB change | C1 vmin/vmax RGB change | ratio | C0 vmin beta_D RMS | C1 vmin beta_D RMS | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 0.000251 | 0.000371 | 1.479 | 0.004507 | 0.003559 | 0.790 |
| 10000 | 0.000970 | 0.001189 | 1.225 | 0.003216 | 0.005091 | 1.583 |
| 14999 | 0.003300 | 0.007292 | 2.210 | 0.002841 | 0.004860 | 1.711 |

This does not support reduced weak-mode ambiguity. The weak direction remained extremely weak in RGB terms, but C1 did not remove the available variation along it.

### Camera-Conditioned Expressiveness

Final M_SAFE C1/C0 ratios at 14999:

| channel | raw across-camera var | raw within-camera var | activated across-camera var | activated within-camera var |
|---|---:|---:|---:|---:|
| beta_D r | 9.072 | 0.00448 | 3.865 | 0.00332 |
| beta_D g | 9.875 | 0.00573 | 2.635 | 0.00245 |
| beta_D b | 9.880 | 0.00367 | 2.420 | 0.00156 |

MIC did not collapse camera-conditioned variation globally. Instead it nearly eliminated within-camera spatial beta_D variance while increasing across-camera beta_D variance, producing very large between/within ratios.

Camera-context swap sensitivity was still present at 14999: median raw beta_D RMS delta was 0.2106 for C0 and 0.3330 for C1; median activated beta_D RMS delta was 0.02132 for C0 and 0.02153 for C1.

### Far / High-Tau Analysis

At final M_SAFE, raw beta_D variance was still higher in C1 than C0 in all depth and tau thirds. For depth far/high, C1/C0 raw beta_D variance ratio was 1.188 and sigma_min/sigma_max ratio improved by 5.029. For tau high, raw variance ratio was 1.759 and sigma_min/sigma_max ratio improved by 4.264. The strongest conditioning effect appears in far/high-tau strata, but the direct variance-suppression target was not achieved there.

### Medium Output and Responsibility

Final M_SAFE activated beta_D means:

```text
C0: r=0.088513, g=0.101304, b=0.087000
C1: r=0.063697, g=0.057784, b=0.047581
```

Final M_SAFE tau means:

```text
C0: r=0.803825, g=0.921520, b=0.786633
C1: r=0.630930, g=0.571432, b=0.471670
```

Final M_SAFE responsibility context:

```text
C0 accumulation_mean=0.045736, direct_object_signal_l2_mean=0.007012, rgb_medium_l2_mean=0.614032
C1 accumulation_mean=0.037701, direct_object_signal_l2_mean=0.010471, rgb_medium_l2_mean=0.609286
```

### Decomposition Safety

Final BND structural gate stayed intact:

| split | branch | J p99 | P(J>1) | tau p90 | tau p99 | P(T<0.1) | P(c>0.99) | P(|s_full|>5) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | C0 | 0.856854 | 0.000000 | 1.139075 | 1.730906 | 0.000000 | 0.009130 | 0.008701 |
| train | C1 | 0.857649 | 0.000000 | 0.795742 | 1.098770 | 0.000000 | 0.008518 | 0.008233 |
| eval | C0 | 0.874474 | 0.000000 | 1.184582 | 1.872856 | 0.000000 | 0.010771 | 0.010612 |
| eval | C1 | 0.873161 | 0.000000 | 0.827594 | 1.187921 | 0.000000 | 0.010432 | 0.010268 |

### Topology Context

Gaussian population trajectory:

| step | C0 count | C1 count | delta |
|---:|---:|---:|---:|
| 3000 | 297659 | 297659 | 0 |
| 5000 | 580221 | 579117 | -1104 |
| 8000 | 766643 | 763278 | -3365 |
| 10000 | 780734 | 780353 | -381 |
| 13000 | 755461 | 754901 | -560 |
| 14999 | 750897 | 750195 | -702 |

Both arms used the standard topology schedule. Divergence is reported as a mechanism context, not forced to match.

## INFERENCE
MIC actionability classification: `MIC_NOT_ACTIONABLE`.
Simple variance-probe classification: `BETAD_VARIANCE_PROBE_FAILED`.

The direct target did not remain suppressed: M_SAFE raw beta_D contextual variance was 4.106x C0 at 14999. Aggregate M_SAFE conditioning did improve, and the effect was visible in far/high-tau strata, but this improvement did not translate into stable weak-mode ambiguity reduction or final RGB safety. The simple all-beta_D raw variance penalty is therefore not an actionable mechanism under this controlled BND setting.

This experiment does not prove true attenuation, true colors, or true geometry; it tests whether suppressing a measured beta_D-dominated low-observability freedom is useful under BND. The correct scientific statement is that a stable residual beta_D ambiguity exists within the camera-conditioned medium representation, and this crude MIC probe did not provide a safe or reliable control mechanism.

## HYPOTHESIS
Next single experiment: Close the betaD identifiability regularization line; do not sweep lambda. Do not implement a second beta_D regularizer as a rescue. A later observability-aware contextual medium mechanism would require a different positive causal signal than this simple variance probe produced.
