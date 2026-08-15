# BND-CBFG Alignment Audit - 2026-08-15

## Repo
EXPERIMENTAL FACT: Branch `research/m1-bounded-intrinsic`, HEAD `13679185e6362ec354a6bc0676887d8697f028d4`.

## Environment
EXPERIMENTAL FACT: CONDA_ENV `water_splatting`, PYTHON_PATH `/opt/anaconda3/envs/water_splatting/bin/python`, TORCH_VERSION `2.1.2+cu118`.
EXPERIMENTAL FACT: CUDA_VISIBLE_DEVICES `6` maps torch logical cuda:0 to physical GPU `6`.

## OMVC Trajectory Closure
QUANTITATIVE RESULT: closure `OMVC_NONPERSISTENT_AFTER_DEACTIVATION`; final registered O1-C0 `0.0006584823131561279`; final clear-J O1-C0 `0.0008721264700094836`.
INFERENCE: BND-OMVC-DIRECT_OBJECT_SIGNAL remains CLOSED. Pure intrinsic-J OMVC remains untested due current backward limitation.

## Historical Recovery
QUANTITATIVE RESULT: LOSSRESP SeaFree-specific hypothesis `NOT_SUPPORTED`; high-J MSE share `0.33781035921730446`; high-J gradient share `0.020002139310155452`.
QUANTITATIVE RESULT: UNORM PSNR gain `-0.04113515218099195`, HIGH_J_MSE_GAP_RECOVERY `0.511933552865014`, RGB_SAFETY `False`.

## SeaFree CB-FG Semantics
CODE FACT: This audit reproduces only foreground-aware reconstruction weighting: normalized pseudo-depth, threshold 1e-2, THRESH_BINARY_INV, largest-contour foreground, W=1/(rendered_underwater_rgb.detach()+1e-3) on foreground and W=1 on background.
CONFIG FACT: CB-BG, coarse-depth loss, OMVC, CDEPTH, depth residuals, depth-aware alpha, and training are excluded.

## Pseudo-Depth / Foreground
EXPERIMENTAL FACT: locked pseudo-depth source `undistorted_data/undistorted_Panama/depthAnything_u16`; availability `True`.

## FAW Alignment
QUANTITATIVE RESULT: train valid-foreground Spearman(FAW, positive delta_e_BND) `-0.20756709769595966`; Spearman(raw darkness, positive delta_e_BND) `-0.21077238831837597`.
QUANTITATIVE RESULT: FAW top-10 positive-regression enrichment `0.875625914813275`; positive-excess concentration `0.05879214031474578`.

## Hard Regions
QUANTITATIVE RESULT: M1_HIGH_J coverage `0.051671340454998764`, FAW enrichment vs foreground `0.4639924481550805`, region in FAW top10 `0.0`, delta_e_BND mean `0.0007192081538960338`.
QUANTITATIVE RESULT: PERSISTENT_BND_HARD coverage `0.06497456771365778`, FAW enrichment vs foreground `0.6083253486699367`, region in FAW top10 `0.004639771740359471`, delta_e_BND mean `0.0006498343427665532`.
QUANTITATIVE RESULT: BND_HARD_CORE coverage `0.023231135245991328`, FAW enrichment vs foreground `0.4040706072998672`, region in FAW top10 `0.0`, delta_e_BND mean `0.0016218775417655706`.

## Gradient Responsibility
QUANTITATIVE RESULT: object CBFG/BASE ratio `1.001081815501511`, medium CBFG/BASE ratio `0.9935402671616025`, CBFG medium/object ratio `0.29250148619912136`, no parameter update `True`.
INFERENCE: Gradient magnitude alone is not interpreted as disentanglement improvement.

## Classification
INFERENCE: Final classification `CBFG_NOT_SUPPORTED`.
INFERENCE: Do not launch CB-FG training in this task. Next single experiment is BND + CB-FG-only only if classification is CBFG_READY; otherwise run one read-only alternative responsibility-signal audit.

## Next Single Experiment
RECOMMENDATION: If the classification is not CBFG_READY, do not run CB-FG training. The next single experiment should be one read-only alternative responsibility-signal audit; if future evidence reaches CBFG_READY, the only next training experiment should be BND + CB-FG-only.

## Outputs
- `outputs/bnd_cbfg_alignment_audit_20260815/`
