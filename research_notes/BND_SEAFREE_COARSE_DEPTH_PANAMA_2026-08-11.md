# BND-CDEPTH Panama Coarse-Depth Supervision

## Code Fact

- WaterSplatting repo branch: `research/m1-bounded-intrinsic`.
- WaterSplatting start HEAD: `e6ddb7c0b8a06adc0da3b61d6817ae62a35be604`.
- SeaFree reference commit: `7797e97dae831029ac89ae9f37b3c3d69ec2cf6c`.
- SeaFree coarse-depth source uses `0.1 * (1 - pearson_corrcoef(pseudo_depth, 1/(10*rendered_depth+1)))`.
- Pseudo-depth is a coarse geometric cue, not metric depth GT.
- BND-CDEPTH adds only a disabled-by-default coarse-depth term to the BND-K1 objective.

## Config Fact

- Formal BND-K1 controls retained: SH degree 3, classic rasterization, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=False`, relative RGB objective.
- New defaults: `coarse_depth_supervision_enabled=False`, `coarse_depth_supervision_weight=0.1`, `load_depths=False`.
- BND-CDEPTH training enables depth loading from `depthAnything_u16` and enables coarse-depth supervision.

## Experimental Fact

- `PSEUDO_DEPTH_ALIGNMENT_PASS`: `True`.
- `CDEPTH_SEMANTIC_ALIGNMENT_VALID`: `True`.
- `DEFAULT_K1_COMPATIBILITY`: `PASS`.
- Forward finite: `True`.
- Gradient finite: `True`.
- `AUDIT_PARAMETER_SAFETY`: `PASS`.
- `INIT_PARAMETER_EQUIVALENCE`: `PASS`.
- `INIT_FORWARD_EQUIVALENCE`: `PASS`.
- `BND_CDEPTH_TRAINING_ELIGIBLE`: `True`.

## Quantitative Result

- Pre-training audit tables are stored under `outputs/bnd_cdepth_panama_20260811/`.

## Inference

- No training result is inferred from the setup audit. The setup audit only decides whether the one-factor training run is eligible.

## Hypothesis

- The causal hypothesis remains untested until the eligible BND-CDEPTH run is trained and summarized.

## BND-CDEPTH Final Summary

### Code Fact

- SeaFree reference repo: `/mnt/new/home_old/ycy/reference_repos/SeaFree-GS`.
- SeaFree fixed commit: `7797e97dae831029ac89ae9f37b3c3d69ec2cf6c`.
- Source files audited: `seafree_gs/seafree_model.py`, `seafree_gs/seafree_dataparser.py`, `seafree_gs/seafree_datamanager.py`.
- SeaFree pseudo-depth input is `batch["depth_image"]`; the dataparser maps `depths_path/<image_stem>.png`.
- SeaFree loss normalizes pseudo-depth per image by `pseudo_depth.max()`.
- SeaFree rendered depth is gsplat `RGB+ED` expected depth with no-support pixels filled by q95 valid depth before loss.
- SeaFree transform is `approximate_rendered_disparity = 1 / (rendered_depth * 10 + 1)`.
- SeaFree coarse-depth term is `0.1 * (1 - pearson_corrcoef(pseudo_depth.flatten(), approximate_rendered_disparity.flatten()))`.
- There is no coarse-depth step cutoff in the fixed source; generic batch masks are multiplied if present, but foreground/background masks are not applied to coarse-depth.
- WaterSplatting BND-CDEPTH implements this as a disabled-by-default term using `outputs["depth"]`, with q95 no-support filling inside the loss input only; renderer physics is unchanged.
- Added datamanager depth loading is controlled by `load_depths=False` by default.

### Experimental Fact

- CDEPTH run config: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_panama_20260811/panama_bnd_cdepth_seed42_step0_to_15000/water-splatting/20260811_bnd_cdepth/config.yml`.
- Summary outputs: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_panama_20260811/bnd_cdepth_final_summary.json`.
- Visual manifest: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/manifest.json`.
- Pseudo-depth source: `/mnt/new/home_old/ycy/water-splatting-refactor/undistorted_data/undistorted_Panama/depthAnything_u16`.
- Pseudo-depth files: 18; RGB files: 18; eval views: `MTN_1529`, `MTN_1539`, `MTN_1547`.
- Validated train views: `MTN_1532`, `MTN_1533`, `MTN_1535`, `MTN_1536`, `MTN_1538`, `MTN_1540`, `MTN_1542`, `MTN_1543`, `MTN_1545`, `MTN_1548`.
- `PSEUDO_DEPTH_ALIGNMENT_PASS=True`.
- `CDEPTH_SEMANTIC_ALIGNMENT_VALID=True`.
- `DEFAULT_K1_COMPATIBILITY=PASS`.
- `AUDIT_PARAMETER_SAFETY=PASS`.
- `INIT_PARAMETER_EQUIVALENCE=PASS`.
- `INIT_FORWARD_EQUIVALENCE=PASS`.
- `BND_CDEPTH_TRAINING_ELIGIBLE=True`.
- One new training run was executed: Panama CDEPTH, seed 42, scratch 0 to 15000, actual final checkpoint `step-000014999.ckpt`.
- Fixed controls retained: SH degree 3, classic rasterization, `bounded_sh3`, `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=False`, formal BND-K1 relative RGB objective, original optimizer/scheduler/densification/pruning.
- TensorBoard scalar extraction found 1500 loss rows and no NaN/Inf logged scalars. Final logged step was 14990.

### Quantitative Result

- Fixed-state depth gradient routing:
  - `means`: RGB norm `1.1827837309`, weighted depth norm `0.0190356804`, ratio `0.0160939654`, cosine `-0.0047889999`.
  - `scales`: RGB norm `0.0018043096`, weighted depth norm `0.0008510921`, ratio `0.4716995695`, cosine `0.0679139632`.
  - `quats`: RGB norm `0.0041357641`, weighted depth norm `0.0017433664`, ratio `0.4215343037`, cosine `0.1142944564`.
  - `opacities`: RGB norm `0.0004180734`, weighted depth norm `0.0002114064`, ratio `0.5056680621`, cosine `0.0424380588`.
  - `features_dc`, `features_rest`, and `medium`: weighted depth norm `0.0`.
- Final RGB:
  - M1: PSNR `32.308910`, SSIM `0.949487`, LPIPS `0.073979`, MSE `0.000595238`.
  - BND-K1: PSNR `31.498353`, SSIM `0.948783`, LPIPS `0.075521`, MSE `0.000714139`.
  - CDEPTH: PSNR `31.753299`, SSIM `0.946292`, LPIPS `0.080931`, MSE `0.000672041`.
  - SeaFree context: PSNR `31.725087`, SSIM `0.944203`, LPIPS `0.088957`, MSE `0.000676841`.
- CDEPTH vs K1: PSNR gain `+0.254946`, SSIM delta `-0.002492`, LPIPS delta `+0.005410`, global MSE gap recovery `0.354060`, RGB safety `False`.
- Per-view CDEPTH vs K1 delta PSNR: mean `+0.254946`, median `+0.232901`, min `+0.138977`, max `+0.392962`; improved views `3`, degraded views `0`.
- Fixed M1_HIGH_J: pixel fraction `0.050461`; MSE M1 `0.002520954`, K1 `0.004780798`, CDEPTH `0.003913123`, SeaFree `0.003383297`; high-J MSE gap recovery `0.383953`.
- M1_LOW_J control: K1 MSE `0.000498052`, CDEPTH MSE `0.000499830`, LOW_J_DAMAGE `0.000001778`.
- Brightness Q5: K1 MSE `0.002150722`, CDEPTH MSE `0.001933033`, gap recovery `0.358741`.
- High-J pseudo-depth diagnostics:
  - BND-K1: Spearman `0.985837`, Pearson `0.985610`, aligned RMSE `0.040947`, gradient Pearson `0.261865`.
  - CDEPTH: Spearman `0.985512`, Pearson `0.984041`, aligned RMSE `0.043154`, gradient Pearson `0.261786`.
  - SeaFree context: Spearman `0.990782`, Pearson `0.990268`, aligned RMSE `0.033772`, gradient Pearson `0.401077`.
  - `HIGHJ_DEPTH_RMSE_IMPROVEMENT=-0.053886`, `HIGHJ_DEPTH_GRAD_GAIN=-0.000079`, `HIGHJ_SPEARMAN_GAIN=-0.000325`, `GEOMETRY_TARGET_IMPROVED=False`.
- Canonical decomposition:
  - M1: tau p90 `1.769849`, J p99 `1.311801`, P(J>1) `0.037758`, P(T<0.1) `0.007454`.
  - BND-K1: tau p90 `0.999069`, J p99 `0.838911`, P(J>1) `0.0`, P(T<0.1) `0.000477`.
  - CDEPTH: tau p90 `0.846655`, J p99 `0.786507`, P(J>1) `0.0`, P(T<0.1) `0.006307`.
  - `TAU_BENEFIT_RETENTION=1.197740`.
- Boundary audit:
  - BND-K1 P(c>0.99) `0.017517`, P(|s|>5) `0.0`.
  - CDEPTH P(c>0.99) `0.012096`, P(|s|>5) `0.0`.
  - `BOUNDARY_ESCAPE=False`.
- Gaussian count trajectory:
  - BND-K1: 1k `60112`, 3k `566332`, 5k `998668`, 8k `1223420`, 10k `1219898`, 13k `1183679`, 15k actual 14999 `1177886`.
  - CDEPTH: 1k `43198`, 3k `453172`, 5k `941413`, 8k `1213251`, 10k `1219880`, 13k `1184186`, 15k actual 14999 `1178513`.
- Forward closure `pred_image - (direct_object_signal + rgb_medium)` was exactly zero for M1, BND-K1, and CDEPTH in recorded outputs.
- Final classification flags: `STRONG_CDEPTH_RECOVERY=False`, `PARTIAL_CDEPTH_RECOVERY=False`, `GEOMETRY_ONLY_POSITIVE=False`, `RGB_ONLY_POSITIVE=True`, `NO_CDEPTH_RECOVERY=False`, `CDEPTH_HARMFUL=True`, `CDEPTH_DECOMPOSITION_REGRESSION=False`, `PANAMA_PARETO_CLOSED=False`.
- Formal hypothesis label: `PERFORMANCE_EFFECT_WITHOUT_GEOMETRY_EVIDENCE`.

### Inference

- The pre-registered gate identifies a performance effect without matching high-J geometry evidence: CDEPTH improved PSNR/global MSE and fixed high-J MSE, but did not improve the high-J pseudo-depth geometry diagnostics.
- The RGB safety gate did not pass because SSIM decreased by `0.002492` and LPIPS worsened by `0.005410` relative to BND-K1.
- The decomposition gate did not fail: P(J>1) remained `0`, tau benefit retention exceeded `0.75`, and boundary escape was false.
- Pseudo-depth remains a coarse diagnostic cue, not metric depth GT.

### Hypothesis

- Causal hypothesis assessment: `PERFORMANCE_EFFECT_WITHOUT_GEOMETRY_EVIDENCE`.
- Next single-factor recommendation: depth-regularization optimization-path diagnostic.

### Visual Assets

- Underwater RGB: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_underwater_rgb.png`.
- Fixed M1_HIGH_J residual: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_fixed_m1_high_j_residual.png`.
- M1_LOW_J control: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_m1_low_j_control.png`.
- Brightness Q5: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_brightness_q5.png`.
- Pseudo-depth diagnostic: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_pseudo_depth_diagnostic.png`.
- Depth residual: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_depth_residual.png`.
- Depth-gradient structure: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_depth_gradient_structure.png`.
- Clear raw: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_clear_raw.png`.
- Boundary usage: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_boundary_usage.png`.
- Direct / medium: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_direct_medium.png`.
- Alpha / coverage: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_alpha_coverage.png`.
- Training trajectory compact summary: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/contact_sheet_training_trajectory_summary.png`.
- Visual index: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_panama_20260811/VISUAL_COMPARE_INDEX.md`.
