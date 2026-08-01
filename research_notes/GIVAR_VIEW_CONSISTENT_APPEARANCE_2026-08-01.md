# GIVAR View-Consistent Appearance Refinement

Date: 2026-08-01

## Objective

Implement and test GIVAR, a training-only M1 appearance refinement module. GIVAR stops the pseudo-depth route and uses Gaussian identity as the cross-view correspondence: the same Gaussian index can accumulate appearance-gradient evidence across train views without depth warping, feature matching, or correspondence banks.

M1 remains:

- `medium_context_mode=dir_xy_camera`
- `b_inf_mode=tied`
- `B_inf = medium_rgb = A`
- Old pseudo-depth, MV-GAR, MCGR, MPDR, J/TBAP/TMICA/TACMD/capacity/cleanup/pruning directions remain disabled in GIVAR experiment scripts.

## Mechanism

GIVAR V0 tests whether M1's remaining JapaneseGradens deficit comes from stable object appearance errors that do not sufficiently enter Gaussian DC features.

For each train view:

- Render the standard M1 underwater prediction.
- Build a detached high-frequency residual map from Sobel RGB error plus 5x5 high-pass RGB error.
- Build detached reliability from GT texture, rendered accumulation, and `depth_std_relative`.
- Sample residual and reliability at projected Gaussian centers.
- After the standard M1 backward pass, read per-Gaussian `features_dc` gradients.
- Accumulate DC gradient direction, magnitude, reliability, detail, view count, and view-direction spread per Gaussian.
- Compute a DC consensus gate from multi-view support, gradient-direction coherence, view spread, and gradient magnitude.
- If active, run an auxiliary underwater render whose geometry, opacity, medium, and background are detached, and whose forward color equals the current render while only gated Gaussian DC features receive the high-frequency Charbonnier auxiliary gradient.

GIVAR V0 does not change geometry, opacity, densification, medium, renderer CUDA code, or inference outputs. SH consensus is reserved and disabled in this version.

## Code Changes

- Added `water_splatting/appearance/givar.py`
  - Detached Sobel and HP5 residual helpers.
  - Detached reliability map from texture, accumulation, and depth concentration.
  - Gaussian-center evidence sampling.
  - DC auxiliary color construction with gated DC gradient exposure only.
  - Weighted HP5 Charbonnier loss.
  - DC gradient-consensus gate and diagnostic statistics.
- Added `water_splatting/appearance/__init__.py`.
- Integrated GIVAR into `water_splatting/water_splatting.py`
  - Added default-off config flags.
  - Added per-Gaussian GIVAR buffers and split/duplicate/cull synchronization.
  - Added load-time buffer rebuilding for MV-GAR, MCGR, and GIVAR when checkpoint Gaussian counts differ.
  - Added detached evidence construction in `get_loss_dict`.
  - Added auxiliary appearance-only render path.
  - Added JSONL logging for window/gate/loss diagnostics.
- Added `scripts/diagnostics/diagnose_givar_gradient_coherence.py`.
- Added 5k experiment wrappers:
  - `scripts/experiments/givar_5k_common.sh`
  - `scripts/experiments/givar_a0_m1_japanesegradens_5k.sh`
  - `scripts/experiments/givar_a1_diag_japanesegradens_5k.sh`
  - `scripts/experiments/givar_a2_dc001_japanesegradens_5k.sh`
  - `scripts/experiments/givar_a3_dc002_japanesegradens_5k.sh`
  - `scripts/experiments/givar_a0_m1_iui3_5k.sh`
  - `scripts/experiments/givar_a1_diag_iui3_5k.sh`
  - `scripts/experiments/givar_a2_dc001_iui3_5k.sh`
  - `scripts/experiments/givar_a3_dc002_iui3_5k.sh`
- Updated `scripts/experiments/backscatter_consistent_binf_iui3_redsea.sh` to pass and record GIVAR flags.

## Config Flags

```python
givar_enabled: bool = False
givar_diagnostic_only: bool = True
givar_dc_enabled: bool = True
givar_sh_enabled: bool = False
givar_start_step: int = 3000
givar_ramp_steps: int = 1000
givar_stop_step: int = 15000
lambda_givar: float = 0.01
givar_highpass_weight: float = 0.35
givar_charbonnier_epsilon: float = 1e-3
givar_min_view_count: int = 4
givar_dc_coherence_threshold: float = 0.55
givar_sh_coherence_threshold: float = 0.50
givar_min_view_spread: float = 0.02
givar_gradient_magnitude_quantile: float = 0.75
givar_accumulation_mid: float = 0.40
givar_accumulation_temp: float = 0.08
givar_depth_std_kappa: float = 0.25
givar_texture_mid: float = 0.10
givar_texture_temp: float = 0.05
givar_stats_window: int = 500
givar_log_path: Optional[str] = None
```

## Phase 0 Gradient-Coherence Diagnostic

The diagnostic loads frozen M1 15k checkpoints, evaluates train views without optimizer steps, computes standard M1 reconstruction gradients, and tests whether Gaussian-ID appearance consensus is dense and aligned enough to justify GIVAR training.

Representative commands:

```bash
/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_givar_gradient_coherence.py \
  --load-config outputs/cross_scene_japanesegradens_redsea_m1_seed42_15000/water-splatting/cross_scene_japanesegradens_redsea_m1_seed42_15000_20260730_cross_scene/config.yml \
  --scene-name JapaneseGradens \
  --split train \
  --max-images 16 \
  --output-dir outputs/givar_phase0_20260801

/opt/anaconda3/envs/water_splatting/bin/python scripts/diagnostics/diagnose_givar_gradient_coherence.py \
  --load-config outputs/m1_dir_xy_camera_iui3_redsea_15000/water-splatting/m1_dir_xy_camera_iui3_redsea_15000_20260723_072412/config.yml \
  --scene-name IUI3 \
  --split train \
  --max-images 16 \
  --output-dir outputs/givar_phase0_20260801
```

Results:

| Scene / Views | Multi-View Visible Fraction | Eligible DC Fraction | Eligible SH Fraction | HF Residual vs DC Grad Corr | Open-Water-Like Eligible Fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| JapaneseGradens / train8 | 62.30% | 0.022% | 0.025% | -0.0485 | 6.84% |
| JapaneseGradens / train16 | 69.92% | 1.693% | 1.933% | -0.0475 | 12.09% |
| IUI3 / train8 | 40.75% | 0.000% | 0.000% | -0.0492 | 0.00% |
| IUI3 / train16 | 57.38% | 0.000% | 0.000% | -0.0404 | 0.00% |
| Curasao / train8 | 83.03% | 0.000% | 0.000% | -0.0611 | 0.00% |
| Panama / train8 | 67.97% | 0.345% | 0.398% | -0.0798 | 35.56% |

Phase 0 was weak to negative. JapaneseGradens has enough DC-eligible support only with 16 views, but high-frequency residual and DC gradient magnitude are negatively correlated. IUI3, Curasao, and Panama have little or no eligible DC support under the V0 gate. This already argues against a strong appearance-consensus signal.

## Smoke Tests

650-step diagnostic-only smoke:

```bash
STAMP=20260801_110713 GPU=6 MAX_NUM_ITERATIONS=650 MODEL_NUM_STEPS=650 STEPS_PER_SAVE=650 RUN_EVAL=0 \
scripts/experiments/givar_a1_diag_japanesegradens_5k.sh
```

Result:

- Training completed without CUDA/autograd errors.
- `logs/givar_smoke_a1_diag_japanesegradens_650_20260801_110713/givar.jsonl` was written.
- Gate/window diagnostics were emitted by step 500.
- No auxiliary loss was active.

800-step active smoke:

```bash
STAMP=20260801_110713 GPU=7 MAX_NUM_ITERATIONS=800 MODEL_NUM_STEPS=800 STEPS_PER_SAVE=800 RUN_EVAL=0 \
EXPERIMENT_NAME=givar_smoke_a2_dc001_japanesegradens_800 \
GIVAR_START_STEP=200 GIVAR_RAMP_STEPS=200 \
scripts/experiments/givar_a2_dc001_japanesegradens_5k.sh
```

Result:

- Training completed without CUDA/autograd errors.
- Step 500 log: `givar_weight=0.01`, raw loss `0.03367`, gate fraction `0.00605`.
- Buffer synchronization survived split/duplicate/cull lifecycle in smoke training.

Gradient isolation check:

- With a manual all-positive DC gate, `givar_loss` produced zero gradient norm for means, scales, quats, opacities, `features_rest`, and medium parameters.
- `features_dc` received nonzero gradient, approximately `1.99e-06`.
- This confirms that the active GIVAR V0 auxiliary loss is appearance-only and DC-only.

## 5k JapaneseGradens Gate

Commands:

```bash
STAMP=20260801_givar5k_a0 GPU=6 scripts/experiments/givar_a0_m1_japanesegradens_5k.sh
STAMP=20260801_givar5k_a1 GPU=7 scripts/experiments/givar_a1_diag_japanesegradens_5k.sh
STAMP=20260801_givar5k_a2 GPU=8 scripts/experiments/givar_a2_dc001_japanesegradens_5k.sh
STAMP=20260801_givar5k_a3 GPU=9 scripts/experiments/givar_a3_dc002_japanesegradens_5k.sh
```

Gate thresholds for JapaneseGradens:

- PSNR >= +0.10 dB vs A0.
- SSIM >= +0.0003 vs A0.
- LPIPS <= -0.0010 vs A0.

Results:

| Run | Setting | PSNR | dPSNR | SSIM | dSSIM | LPIPS | dLPIPS | Gaussians | Growth vs A0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | M1 control | 22.895731 | 0.000000 | 0.800414 | 0.000000 | 0.237683 | 0.000000 | 721,558 | 0.00% |
| A1 | GIVAR diagnostic-only | 22.896202 | +0.000471 | 0.801436 | +0.001021 | 0.232611 | -0.005072 | 718,534 | -0.42% |
| A2 | DC consensus, lambda=0.01 | 22.976023 | +0.080292 | 0.801352 | +0.000938 | 0.238323 | +0.000640 | 721,097 | -0.06% |
| A3 | DC consensus, lambda=0.02 | 22.795620 | -0.100111 | 0.802689 | +0.002274 | 0.238030 | +0.000347 | 721,205 | -0.05% |

GIVAR diagnostic logs:

- A2 step 4500: gate fraction `0.00461`, raw loss `0.01236`, active weight `0.01`.
- A3 step 4500: gate fraction `0.00470`, raw loss `0.01283`, active weight `0.02`.
- A1 step 4500: diagnostic-only gate fraction `0.00470`, active weight `0.0`.

Decision:

- A2 improved PSNR by +0.080 dB and SSIM by +0.00094, but missed the +0.10 dB PSNR gate and worsened LPIPS by +0.00064.
- A3 worsened PSNR and worsened LPIPS.
- A1 diagnostic-only improved LPIPS, but because it applies no GIVAR loss, it is treated as trajectory noise / training nondeterminism rather than evidence for the module.
- No active GIVAR configuration passes the JapaneseGradens 5k gate.
- IUI3 safety and 15k formal runs were not launched because the primary scene gate failed.

## Checkpoint and Eval Artifacts

Local artifacts were kept under untracked output locations:

- `outputs/givar_a0_m1_japanesegradens_seed42_5000/.../step-000004999.ckpt`
- `outputs/givar_a1_diag_japanesegradens_seed42_5000/.../step-000004999.ckpt`
- `outputs/givar_a2_dc001_japanesegradens_seed42_5000/.../step-000004999.ckpt`
- `outputs/givar_a3_dc002_japanesegradens_seed42_5000/.../step-000004999.ckpt`
- `renders/givar_a0_m1_japanesegradens_seed42_5000_20260801_givar5k_a0/output.json`
- `renders/givar_a1_diag_japanesegradens_seed42_5000_20260801_givar5k_a1/output.json`
- `renders/givar_a2_dc001_japanesegradens_seed42_5000_20260801_givar5k_a2/output.json`
- `renders/givar_a3_dc002_japanesegradens_seed42_5000_20260801_givar5k_a3/output.json`
- `logs/givar_a1_diag_japanesegradens_seed42_5000_20260801_givar5k_a1/givar.jsonl`
- `logs/givar_a2_dc001_japanesegradens_seed42_5000_20260801_givar5k_a2/givar.jsonl`
- `logs/givar_a3_dc002_japanesegradens_seed42_5000_20260801_givar5k_a3/givar.jsonl`

These output, render, log, and checkpoint artifacts must remain uncommitted.

## Interpretation

GIVAR V0 does not confirm that M1's JapaneseGradens loss comes from insufficient Gaussian DC appearance refinement.

The negative evidence is consistent across diagnostics and training:

- Phase 0 found weak or absent eligible DC support in most scenes.
- HF residual is negatively correlated with DC gradient magnitude in all checked scenes.
- Active GIVAR gates only about 0.46% to 0.47% of Gaussians late in 5k training.
- Lambda 0.01 gives a modest PSNR gain but worsens LPIPS.
- Lambda 0.02 degrades PSNR and still worsens LPIPS.
- Gaussian counts remain essentially unchanged, so the result is not a capacity-growth effect.

The current evidence says JapaneseGradens is unlikely to be solved by the tested Gaussian appearance-consensus route. Combined with MPDR, MV-GAR, and MCGR, Gaussian refinement has repeatedly failed to produce a safe PSNR/SSIM/LPIPS improvement profile.

## Next Step

Stop GIVAR V0 before IUI3/15k unless there is a deliberate decision to override the gate. The next useful direction is outside pseudo-depth and Gaussian refinement: camera/exposure consistency, datamanager/camera-scale audit, medium/Gaussian frequency separation, or medium and SH learning-rate coupling.

