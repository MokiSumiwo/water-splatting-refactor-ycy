# BND-PW-AUDIT-IUI3

## Scope
CONFIG FACT: This is a read-only, zero-training audit. No optimizer step, checkpoint write, new loss training, threshold sweep, CDEPTH, OMVC, depth-aware alpha, CB-FG, or CB-BG training is performed.

## Repo
EXPERIMENTAL FACT: Branch `research/m1-bounded-intrinsic`, HEAD `33847b4805d28d891aeca7f004750469a5869a81`.

## Environment
EXPERIMENTAL FACT: `CONDA_ENV=water_splatting`, `PYTHON_PATH=/opt/anaconda3/envs/water_splatting/bin/python`, `TORCH_VERSION=2.1.2+cu118`.
EXPERIMENTAL FACT: `CUDA_VISIBLE_DEVICES=6` maps torch logical cuda:0 to physical GPU `6`.

## Recovered Locked PW-Audit Semantics
CODE FACT: `M_SF` is the SeaFree-style pseudo-depth background candidate using per-image max normalization, threshold `1e-2`, largest filled foreground component, and complement background.
CONFIG FACT: `M_LOW_SUPPORT = BND@3000 accumulation <= 0.01`; `M_INTERSECT = M_SF & M_LOW_SUPPORT`; `M_SAFE = BinaryErode(M_INTERSECT, radius=5 px)`.
CONFIG FACT: These thresholds are reused unchanged from the Panama locked BND-PW-AUDIT.

## WaterSplatting B_inf Semantics
CODE FACT: With `b_inf_mode=tied`, `B_inf = medium_rgb = sigmoid(medium_mlp[...,0:3])`.
CODE FACT: The 22-D medium input is 16-D direction encoding plus 3-D XY/r context plus 3-D normalized camera-center context.
CODE FACT: Tied tail recomposition uses `tail_weight * b_inf`; this differs from SeaFree CB-BG's `water_background_image` semantics.

## IUI3 Data / Pseudo-Depth Availability
EXPERIMENTAL FACT: image path `/mnt/new/home_old/ycy/water-splatting-refactor/undistorted_data/undistorted_IUI3-RedSea/images/ColorImage`.
EXPERIMENTAL FACT: pseudo-depth path `/mnt/new/home_old/ycy/water-splatting-refactor/undistorted_data/undistorted_IUI3-RedSea/depthAnything_u16`.
EXPERIMENTAL FACT: pseudo-depth available/required `29/29`.
EXPERIMENTAL FACT: train views `['MTN_5906', 'MTN_5898', 'MTN_5905', 'MTN_5900', 'MTN_5899', 'MTN_5901', 'MTN_5904', 'MTN_5896', 'MTN_5895', 'MTN_5902', 'MTN_5907', 'MTN_5908', 'MTN_5910', 'MTN_5916', 'MTN_5917', 'MTN_5909', 'MTN_5915', 'MTN_5927', 'MTN_5914', 'MTN_5913', 'MTN_5912', 'MTN_5929', 'MTN_5930', 'MTN_5931', 'MTN_5933']`.
EXPERIMENTAL FACT: eval views `['MTN_5903', 'MTN_5894', 'MTN_5911', 'MTN_5928']`.

## Available IUI3 Checkpoints
EXPERIMENTAL FACT: BND available requested-to-actual map `{'3000': 3000, '5000': 5000, '8000': 8000, '10000': 10000, '13000': 13000, '15000': 14999}`.
EXPERIMENTAL FACT: M1 available requested-to-actual map `{'5000': 5000, '10000': 10000, '15000': 14999}`.

## Pure-Water Candidate Coverage
QUANTITATIVE RESULT: Train M_SAFE pooled coverage `0.329706523598991`; train views >=1 percent `25`.
QUANTITATIVE RESULT: Eval M_SAFE pooled coverage `0.2715545756975771`.
EXPERIMENTAL FACT: Per-view coverage and spatial descriptors are stored in `pure_water_candidate_coverage.csv/json` and `spatial_candidate_stability.csv/json`.

## Temporal Object Contamination
QUANTITATIVE RESULT: Final BND train mean fraction accumulation>0.01 across views `0.3336633026599884`.
QUANTITATIVE CONCLUSION: `LOW_OBJECT_CONTAMINATION_LOCKED_RULE = False`.

## Temporal Medium / B_inf Headroom
QUANTITATIVE RESULT: Final BND train BINF_L1 view mean `0.006945778522640467`; R_anchor view mean `0.003816798587873637`.
QUANTITATIVE CONCLUSION: `BACKGROUND_HEADROOM_LOCKED_RULE = False`.
EXPERIMENTAL FACT: Temporal headroom rows are stored in `binf_medium_headroom.csv/json`.

## Contamination-Headroom Relationship
EXPERIMENTAL FACT: Spearman associations across views/checkpoints are stored in `contamination_headroom_relationship.csv/json`.

## B_inf Saturation
EXPERIMENTAL FACT: B_inf saturation and recovered pre-sigmoid logit statistics are stored in `binf_saturation.csv/json`.

## Medium-Only Gradient Pathway
QUANTITATIVE RESULT: medium_mlp grad L2 `0.0957169182238159`, direction_encoding grad L2 `0.0`, max object grad L2 `0.0`.
EXPERIMENTAL FACT: Head/trunk parameter split status `PARAMETER_SUBGROUP_NOT_EXPOSED_FOR_CURRENT_MEDIUM_MLP`.
QUANTITATIVE CONCLUSION: `MEDIUM_DOMINANT_GRADIENT_ROUTE = True`; parameter delta max `0.0`.

## M1 vs BND Comparison
EXPERIMENTAL FACT: Matched M1/BND comparison rows are stored in `m1_bnd_background_comparison.csv/json` for 5k, 10k, and final.

## View / Direction Stability
QUANTITATIVE CONCLUSION: `VIEW_STABILITY_RULE = True`; channel sign consistency `{'R': 0.6, 'G': 0.8, 'B': 0.88}`.

## Early-Mid-Late Observability
QUANTITATIVE RESULT: early summary `{'stage': 'early', 'steps': '3000;5000', 'candidate_coverage_fixed_train_M_SAFE': 0.329706523598991, 'views_ge_1pct_locked_nontrivial': 25, 'fraction_accumulation_gt_0p01_view_checkpoint_mean': 0.01848019815981388, 'mean_accumulation_view_checkpoint_mean': 0.0010654572179191746, 'BINF_L1_view_checkpoint_mean': 0.008697864571586252, 'R_anchor_view_checkpoint_mean': 0.0012135821468341754, 'BINF_L1_view_checkpoint_std': 0.0015587180639441355, 'contamination_view_checkpoint_std': 0.027871536450833657, 'BINF_L1_cv': 0.17920698248580205, 'contamination_cv': 1.5081838522403788, 'R_sign_consistency': 0.64, 'G_sign_consistency': 0.94, 'B_sign_consistency': 0.92, 'stage_candidate_reliability_high_locked_rule': True, 'stage_headroom_nontrivial_locked_rule': False}`.
QUANTITATIVE RESULT: mid summary `{'stage': 'mid', 'steps': '8000;10000', 'candidate_coverage_fixed_train_M_SAFE': 0.329706523598991, 'views_ge_1pct_locked_nontrivial': 25, 'fraction_accumulation_gt_0p01_view_checkpoint_mean': 0.06716504835523665, 'mean_accumulation_view_checkpoint_mean': 0.009464865289628506, 'BINF_L1_view_checkpoint_mean': 0.007054399345070124, 'R_anchor_view_checkpoint_mean': 0.002975253044119424, 'BINF_L1_view_checkpoint_std': 0.0007688922726524519, 'contamination_view_checkpoint_std': 0.0509160766326622, 'BINF_L1_cv': 0.10899471876224053, 'contamination_cv': 0.7580739965132837, 'R_sign_consistency': 0.52, 'G_sign_consistency': 0.82, 'B_sign_consistency': 0.72, 'stage_candidate_reliability_high_locked_rule': True, 'stage_headroom_nontrivial_locked_rule': False}`.
QUANTITATIVE RESULT: late summary `{'stage': 'late', 'steps': '13000;15000', 'candidate_coverage_fixed_train_M_SAFE': 0.329706523598991, 'views_ge_1pct_locked_nontrivial': 25, 'fraction_accumulation_gt_0p01_view_checkpoint_mean': 0.3002371853590012, 'mean_accumulation_view_checkpoint_mean': 0.0362676421739161, 'BINF_L1_view_checkpoint_mean': 0.006833206713199615, 'R_anchor_view_checkpoint_mean': 0.005922927846160723, 'BINF_L1_view_checkpoint_std': 0.0008691741591292618, 'contamination_view_checkpoint_std': 0.1736656899715916, 'BINF_L1_cv': 0.1271985753702269, 'contamination_cv': 0.5784283174781802, 'R_sign_consistency': 0.56, 'G_sign_consistency': 0.62, 'B_sign_consistency': 0.66, 'stage_candidate_reliability_high_locked_rule': False, 'stage_headroom_nontrivial_locked_rule': False}`.

## Panama-Curasao-IUI3 Comparison
EXPERIMENTAL FACT: Cross-scene comparison uses only formalized Panama/Curasao values plus this IUI3 audit and is stored in `panama_curasao_iui3_comparison.csv/json`.

## Decomposition Context
EXPERIMENTAL FACT: BND decomposition context rows are stored in `decomposition_context.csv/json`.

## BG-Anchor Classification
INFERENCE: `BG_ANCHOR_WEAK`.

## Observability-Routing Classification
INFERENCE: `OBSERVABILITY_ROUTING_TENTATIVE`.
INFERENCE: reliability_degrades `True`, headroom_diminishes `True`, early_mid_usable_locked_rule `False`.

## Next Single Experiment
RECOMMENDATION: `Read-only design/preflight for observability-guided medium calibration`.

## Required Question Answers
INFERENCE: Q1-Q14 are answered in the final report using the output tables named above.
