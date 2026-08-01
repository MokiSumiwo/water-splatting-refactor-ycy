# MFRS Medium Frequency Responsibility Audit

Date: 2026-08-01

## Objective

After MPDR, MV-GAR, MCGR, and GIVAR produced negative or unsafe evidence for Gaussian-side fixes, this audit tests whether M1's remaining error is instead caused by medium/Gaussian frequency responsibility mixing.

The proposed next module was BLMF, a band-limited medium field. This note records the required precondition checks before implementing BLMF. Because the audit does not provide positive evidence, BLMF was not implemented or trained in this round.

M1 baseline remains:

- `medium_context_mode=dir_xy_camera`
- `b_inf_mode=tied`
- `B_inf = medium_rgb = A`
- No pseudo-depth, MPDR, MV-GAR, MCGR, GIVAR, J/TBAP/TMICA/TACMD/capacity/cleanup/pruning losses are enabled for the audit.

## Code Changes

- Added `scripts/diagnostics/diagnose_medium_frequency_responsibility.py`
  - Loads frozen checkpoints with `eval_setup`.
  - Computes branch high-pass energy for `rgb_object`, `rgb_medium_total`, `medium_rgb`, `medium_bs`, `medium_attn`, final RGB, and GT.
  - Computes region-wise statistics for whole image, high-accumulation object region, low-accumulation open-water region, and accumulation-boundary region.
  - Computes high-frequency residual correlation against medium-render and Gaussian-render high-frequency energy.
  - Runs counterfactual low-pass evaluation for F2/F4/F8, FA, and FC by re-rendering with low-passed medium maps and unchanged Gaussians.
  - Tests camera-context sensitivity by replacing or perturbing the camera-center context while keeping directions, image coordinates, and Gaussian render fixed.
- Added `scripts/diagnostics/summarize_deterministic_noise_audit.py`
  - Summarizes PSNR/SSIM/LPIPS/Gaussian-count ranges across nominally equivalent runs.
  - Used here to quantify the observed A0 vs A1 diagnostic-only discrepancy from the prior GIVAR screen.

No BLMF model flags or medium-field architecture changes were added because the Phase 0 gate did not support training a new medium module.

## Phase -1 Noise Audit

The exact requested Phase -1 requires resuming N0/N1/N2 from the same M1 step-3000 checkpoint with model, optimizer, datamanager sampler, Python, NumPy, Torch CPU, and Torch CUDA RNG states restored.

No `step-000003000.ckpt` is currently available in `outputs/`, so the exact resume audit could not be run yet. As a lower-bound noise check, I summarized the already completed JapaneseGradens A0 M1 and A1 GIVAR diagnostic-only 5k runs.

Command:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/summarize_deterministic_noise_audit.py \
  --labels A0_M1 A1_GIVAR_diag \
  --metric-jsons \
    renders/givar_a0_m1_japanesegradens_seed42_5000_20260801_givar5k_a0/output.json \
    renders/givar_a1_diag_japanesegradens_seed42_5000_20260801_givar5k_a1/output.json \
  --checkpoints \
    outputs/givar_a0_m1_japanesegradens_seed42_5000/water-splatting/givar_a0_m1_japanesegradens_seed42_5000_20260801_givar5k_a0/nerfstudio_models/step-000004999.ckpt \
    outputs/givar_a1_diag_japanesegradens_seed42_5000/water-splatting/givar_a1_diag_japanesegradens_seed42_5000_20260801_givar5k_a1/nerfstudio_models/step-000004999.ckpt \
  --output-json outputs/mfrs_phase_minus1_20260801/givar_a0_a1_noise_summary.json
```

Result:

| Comparison | PSNR Range | SSIM Range | LPIPS Range | Gaussian Count Range | Gaussian Count Range % | Noise Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A0 M1 vs A1 diagnostic-only | 0.00047 | 0.00102 | 0.00507 | 3,024 | 0.421% | Fail |

The observed LPIPS and Gaussian-count range exceed the proposed noise thresholds. This confirms that 5k gates must be interpreted cautiously until a true resume-based deterministic audit is run.

## Phase 0 Frequency Responsibility Audit

Commands:

```bash
CUDA_VISIBLE_DEVICES=6 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_medium_frequency_responsibility.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --scene-name JapaneseGradens \
  --split eval \
  --max-images 0 \
  --output-dir outputs/mfrs_phase0_20260801

CUDA_VISIBLE_DEVICES=7 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_medium_frequency_responsibility.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --scene-name IUI3 \
  --split eval \
  --max-images 0 \
  --output-dir outputs/mfrs_phase0_20260801

CUDA_VISIBLE_DEVICES=8 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_medium_frequency_responsibility.py \
  --load-config outputs/cross_scene_curasao_m1_seed42_15000/water-splatting/cross_scene_curasao_m1_seed42_15000_20260730_cross_scene/config.yml \
  --scene-name Curasao \
  --split eval \
  --max-images 0 \
  --output-dir outputs/mfrs_phase0_20260801

CUDA_VISIBLE_DEVICES=9 /opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_medium_frequency_responsibility.py \
  --load-config outputs/cross_scene_panama_m1_seed42_15000/water-splatting/cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml \
  --scene-name Panama \
  --split eval \
  --max-images 0 \
  --output-dir outputs/mfrs_phase0_20260801
```

## Branch Frequency Results

Whole-image aggregate:

| Scene | Medium HF / Final HF | Gaussian HF / Final HF | Medium HF vs Residual Corr | Gaussian HF vs Residual Corr |
| --- | ---: | ---: | ---: | ---: |
| JapaneseGradens | 7.39% | 102.00% | 0.2991 | 0.4483 |
| IUI3 | 9.62% | 102.56% | 0.2556 | 0.4006 |
| Curasao | 5.08% | 101.82% | 0.0964 | 0.4095 |
| Panama | 7.37% | 102.95% | 0.1456 | 0.4094 |

The medium branch has measurable high-frequency energy, but it is below the proposed 10% threshold in all four scenes. The final high-frequency residual correlates more strongly with Gaussian-render high-frequency energy than with medium-render high-frequency energy in every scene.

## Counterfactual Low-Pass Results

All values are deltas vs original M1 F0 on the same frozen checkpoint.

| Scene | F4 dPSNR | F4 dSSIM | F4 dLPIPS | F8 dPSNR | F8 dSSIM | F8 dLPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JapaneseGradens | +0.000060 | +0.000004 | +0.000055 | +0.000167 | +0.000005 | +0.000098 |
| IUI3 | +0.000221 | +0.000008 | +0.000109 | +0.000081 | +0.000009 | +0.000325 |
| Curasao | +0.000209 | +0.000004 | +0.000045 | +0.000906 | +0.000004 | +0.000089 |
| Panama | +0.000031 | +0.000003 | +0.000007 | +0.000048 | +0.000004 | +0.000025 |

Medium low-pass has near-zero PSNR/SSIM impact and consistently worsens LPIPS, although only by a small amount. It does not satisfy the proposed evidence condition of LPIPS improvement >= 0.001 with PSNR decline <= 0.05 dB.

Branch-only downscale=4:

| Scene | FA dLPIPS | FC dLPIPS | Interpretation |
| --- | ---: | ---: | --- |
| JapaneseGradens | +0.000054 | +0.000002 | Neither A-only nor coefficient-only low-pass helps |
| IUI3 | +0.000088 | +0.000017 | Neither helps |
| Curasao | +0.000036 | +0.000007 | Neither helps |
| Panama | +0.000005 | +0.000002 | Neither helps |

There is no evidence that the issue is specifically medium color `A` or specifically `sigma_bs` / `sigma_attn` spatial variation.

## Camera-Context Sensitivity

A small camera-center perturbation changes medium RGB by a very small absolute amount:

| Scene | Mean Abs Medium RGB Delta | HF Energy of Delta |
| --- | ---: | ---: |
| JapaneseGradens | 0.000103 | 0.9482 |
| IUI3 | 0.000375 | 0.3039 |
| Curasao | 0.000199 | 0.5363 |
| Panama | 0.000259 | 0.4530 |

The delta can be high-pass dominated because the absolute delta is tiny, but the magnitude is too small to support the claim that camera context is materially injecting object-like structure into the medium maps.

## Gate Decision

Proposed Phase 0 evidence conditions:

- Medium HF / final HF >= 10%.
- Medium HF vs final HF residual correlation >= 0.15.
- F4/F8 LPIPS improvement >= 0.001.
- F4/F8 PSNR decline <= 0.05 dB.
- Camera-context perturbation creates clear structured medium change.

Assessment:

- JapaneseGradens fails medium HF >= 10%.
- JapaneseGradens passes medium correlation >= 0.15, but Gaussian correlation is higher.
- F4/F8 do not improve LPIPS in any scene.
- F4/F8 preserve PSNR, but this is only because the effect size is nearly zero.
- Camera perturbation absolute medium changes are too small to be actionable.
- Curasao and Panama do not show similar trend; IUI3 has correlation but also no LPIPS benefit.

Decision: do not implement or train BLMF from this evidence.

## Interpretation

The audit does not support the hypothesis that M1 is losing JapaneseGradens primarily because the medium branch is absorbing harmful spatial high-frequency content. The medium field is not completely smooth, but counterfactual low-pass evaluation shows that removing its high frequencies is essentially a no-op for PSNR/SSIM and slightly negative for LPIPS.

Combined with GIVAR, MCGR, MV-GAR, and MPDR, the current evidence says the next route should not be another Gaussian refinement or medium band-limit module. The next candidate should be camera exposure / white-balance nuisance factorization, or a datamanager/camera consistency audit.

## Remaining Work

The exact deterministic resume audit still requires creating or locating a shared step-3000 checkpoint. Future run:

1. Train M1 to step 3000 with `save_only_latest_checkpoint=False` or `steps_per_save=3000`.
2. Resume N0/N1/N2 from the same checkpoint to step 5000.
3. Compare metrics and Gaussian counts with `summarize_deterministic_noise_audit.py`.

