# BND-RESPONSIBILITY-DRIFT-AUDIT-IUI3

## Scope
CONFIG FACT: This is a read-only, zero-training diagnostic. No optimizer, parameter update, new loss, new module, threshold sweep, CUDA edit, checkpoint write, or training intervention is used.

## Repo
EXPERIMENTAL FACT: Branch `research/m1-bounded-intrinsic`, HEAD `89bf1f0aab28cd952f329f7ed5faa7018bbdbdb9`.
EXPERIMENTAL FACT: script-run pre-staging status was `?? research_notes/BND_RESPONSIBILITY_DRIFT_AUDIT_IUI3_2026-08-25.md
?? scripts/diagnostics/audit_bnd_responsibility_drift_iui3.py
?? scripts/diagnostics/render_gmvc_curasao_contact_sheet.py
?? scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py`.

## Environment
EXPERIMENTAL FACT: `CONDA_ENV=water_splatting`, `PYTHON_PATH=/opt/anaconda3/envs/water_splatting/bin/python`, `PYTHON_VERSION=3.8.20`, `TORCH_VERSION=2.1.2+cu118`.
EXPERIMENTAL FACT: `CUDA_VISIBLE_DEVICES=6` maps torch logical cuda:0 to physical GPU `6` (`NVIDIA GeForce RTX 3080`).

## Locked Candidate Semantics
CONFIG FACT: `M_SAFE = erode_5px(M_SF & (BND@3000 accumulation <= 0.01))`; `M_SF` is the pseudo-depth background candidate from per-image max-normalized `depthAnything_u16`, threshold `1e-2`, largest filled foreground component, and complement background.
CONFIG FACT: The candidate mask is fixed across all later checkpoints and is not treated as true water.

## Accumulation Semantics
CODE FACT: `outputs['accumulation']` is per-pixel `1 - final_Ts`, the screen-space transmittance complement from Gaussian alpha compositing.
CODE FACT: It is not Gaussian opacity alpha_i and not exact per-Gaussian contribution; medium does not itself add alpha.

## Gaussian Topology / Lineage
CODE FACT: Lineage classification `GAUSSIAN_LINEAGE_PARTIAL`.
INFERENCE: Late-birth counterfactual `NOT_RUN` because exact row-wise identity is not recoverable.

## Quantitative Results
QUANTITATIVE RESULT: 10k->13k pooled mean delta_A `0.017192916944622993`, crossing fraction `0.14186608116221308`.
QUANTITATIVE RESULT: 13k->15k pooled mean delta_A `0.003041217103600502`, crossing fraction `0.059028438113554245`.
QUANTITATIVE RESULT: 10k->15k pooled Spearman(delta_A, delta_E) `-0.02832792169455418`.
QUANTITATIVE RESULT: top-10% delta_A RGB worsening enrichment `0.969559673749798`, positive delta_E share `0.09452624563182654`.
QUANTITATIVE RESULT: late contaminated delta RGB MSE `-4.525914846453816e-06`; late clean delta RGB MSE `-1.0855583241209388e-06`.
QUANTITATIVE RESULT: Spearman(delta_A, delta_BINF_L1) `-0.027208738231485282`; Spearman(delta_A, delta_tau) `-0.30169787434179446`.
QUANTITATIVE RESULT: M1/BND pooled comparison `{'M1': {'run': 'M1', 'view_id': 'ALL', 'accumulation_10k': 0.21646784245967865, 'accumulation_15k': 0.2739681899547577, 'delta_A_10k_to_15k': 0.05750015005469322, 'fraction_accumulation_gt_0p01_10k': 0.4454549681080938, 'fraction_accumulation_gt_0p01_15k': 0.8227407715510957, 'rgb_mse_10k': 9.424053132534027e-05, 'rgb_mse_15k': 9.075683919945732e-05, 'delta_E_10k_to_15k': -3.4836357372114435e-06}, 'BND': {'run': 'BND', 'view_id': 'ALL', 'accumulation_10k': 0.010211782529950142, 'accumulation_15k': 0.030445925891399384, 'delta_A_10k_to_15k': 0.0202341265976429, 'fraction_accumulation_gt_0p01_10k': 0.06721215875813465, 'fraction_accumulation_gt_0p01_15k': 0.26742699720933866, 'rgb_mse_10k': 9.088312071980909e-05, 'rgb_mse_15k': 8.887751027941704e-05, 'delta_E_10k_to_15k': -2.0056106677657226e-06}, 'BND_minus_M1_delta_A_10k_to_15k': -0.037266023457050323, 'BND_minus_M1_final_fraction_acc_gt_0p01': -0.5553137743417571}`.
QUANTITATIVE RESULT: origin proxy summary `{'total_gaussian_count_delta_10k_to_15k': -40646.0, 'center_in_safe_count_delta_10k_to_15k': 29.0, 'center_in_safe_opacity_mass_delta_10k_to_15k': -34.11391136050224, 'outside_footprint_count_delta_10k_to_15k': 1489.0, 'outside_footprint_opacity_footprint_mass_delta_10k_to_15k': 347965384.0, 'any_footprint_projected_radius_mean_delta_10k_to_15k': 12.180899060043913, 'lineage_exact': False}`.

## Hard-Region Context
EXPERIMENTAL FACT: `NOT_AVAILABLE_FOR_IUI3_COMPATIBLE_LABELS`. Existing IUI3 sidecars expose J1/J95/TAU90/TLOW/COMP, not the registered formal labels; this audit does not redefine them.

## Classifications
INFERENCE: Harmfulness `RESPONSIBILITY_DRIFT_NOT_SUPPORTED`.
INFERENCE: Origin `DRIFT_ORIGIN_UNRESOLVED`.
INFERENCE: Responsibility preservation `RESPONSIBILITY_PRESERVATION_NOT_SUPPORTED`.

## Scientific Interpretation
INFERENCE: The audit evaluates late candidate-region object occupation as a hypothesis, not as confirmed misattribution. It does not claim true color, true geometry, or exact per-Gaussian responsibility.

## Next Single Experiment
RECOMMENDATION: `BND-MEDIUM-IDENTIFIABILITY-PREFLIGHT`.
