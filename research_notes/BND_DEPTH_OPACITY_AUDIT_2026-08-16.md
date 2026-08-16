# BND Depth-Opacity Audit - 2026-08-16

## Scope
CONFIG FACT: This is a read-only, zero-training mechanism audit. No optimizer step, checkpoint write, CDEPTH/OMVC/CB-FG/BAP training, depth residual loss, synthetic epipolar depth, or depth-aware alpha implementation is performed.

## Repo
EXPERIMENTAL FACT: Branch `research/m1-bounded-intrinsic`, HEAD `17d5cde133db009b37844899264986c2d99425ae`.

## Environment
EXPERIMENTAL FACT: `CONDA_ENV=water_splatting`, `PYTHON_PATH=/opt/anaconda3/envs/water_splatting/bin/python`, `TORCH_VERSION=2.1.2+cu118`.
EXPERIMENTAL FACT: `CUDA_VISIBLE_DEVICES=6` maps torch logical `cuda:0` to physical GPU `6`.

## Historical CDEPTH Semantics
CODE FACT: CDEPTH supervision source `batch['depth_image'] from DepthDataset; dataparser sets depth_path from depths_path/image_name.with_suffix('.png').`.
CODE FACT: Loss `coarse_grained_depth_loss = 0.1 * (1 - pearson_corrcoef(pseudo_depth.flatten(), approximate_rendered_disparity.flatten()))`.
CODE FACT: Gradient pathway `rendered_depth branch, therefore rasterization geometry/opacity paths according to autograd; not a direct color loss.`.
INFERENCE: Formal CDEPTH conclusion `PERFORMANCE_EFFECT_WITHOUT_GEOMETRY_EVIDENCE`.
INFERENCE: A future depth-aware-alpha-only mechanism would differ from CDEPTH by operating through opacity modulation/pruning rather than a pseudo-depth rendered-depth loss.

## Available Matched Checkpoints
EXPERIMENTAL FACT: Requested nominal checkpoints were `[3000, 5000, 8000, 10000, 15000]`.
EXPERIMENTAL FACT: BND-K1 checkpoints were available at requested nominal `[3000, 5000, 8000, 10000, 15000]`, with nominal 15000 loaded as actual 14999.
EXPERIMENTAL FACT: M1 checkpoints were available only at nominal `[5000, 10000, 15000]`, with nominal 15000 loaded as actual 14999.
EXPERIMENTAL FACT: Matched M1/BND analysis therefore used nominal `[5000, 10000, 15000]` and actual `[5000, 10000, 14999]`; no checkpoint was trained or synthesized to fill missing 3000/8000 M1 states.

## Depth-Inconsistency Definition
CODE FACT: `RZ_ABS(i,v)=abs(D_v(x_i,v)-z_i,v)` and `RZ_REL=RZ_ABS/(D_v(x_i,v)+epsilon)` with `epsilon=1e-6`.
CODE FACT: `D_v` is normal rendered alpha-blended expected depth; this is pseudo/structure consistency evidence, not ground-truth geometry.

## Opacity / Contribution Availability
CODE FACT: `IMPLEMENTATION_BLOCKED`; safe proxies are `opacity alpha_i=sigmoid(opacity_logit), screen-space radius/footprint, opacity_footprint=alpha_i*pi*r^2, joint_depth_opacity=RZ_REL*alpha_i, and deterministic nearest-pixel sums.`.

## M1 vs BND Depth-Opacity Distributions
QUANTITATIVE RESULT: Final train M1 RZ_REL mean `0.048840705305337906`, fraction RZ_REL>1 `0.00043731896960699996`.
QUANTITATIVE RESULT: Final train BND RZ_REL mean `0.05118415877223015`, fraction RZ_REL>1 `0.001649322481909976`, opacity mean `0.7259640097618103`, joint mean `0.039814338088035583`.
QUANTITATIVE RESULT: BND-M1 train fraction RZ_REL>1 deltas were `+0.0003881093218537193` at 5k, `+0.0012054312898341156` at 10k, and `+0.0012120035123029762` at final.
QUANTITATIVE RESULT: BND-M1 eval fraction RZ_REL>1 deltas were `+0.000816644897369194` at 5k, `+0.003485887044432937` at 10k, and `+0.003565687124995367` at final.
INFERENCE: BND shows a small distributional increase in depth-inconsistent projected Gaussian pairs, but the absolute RZ_REL>1 train population remains below 0.17 percent.

## BND Regression Alignment
QUANTITATIVE RESULT: Final train Spearman(pixel_joint_depth_opacity_sum, positive_delta_e_BND) `0.027404585011140577`.
QUANTITATIVE RESULT: Train Spearman(pixel_joint_depth_opacity_sum, positive_delta_e_BND) was `0.007714595905129062` at 5k, `0.048076310711427414` at 10k, and `0.027404585011140577` at final.
QUANTITATIVE RESULT: Final train top-10 joint-depth-opacity population had positive-regression enrichment `1.0177204034504628`.
INFERENCE: The BND-specific RGB regression alignment is weak and does not satisfy causal actionability.

## Hard-Region Alignment
QUANTITATIVE RESULT: Final train M1_HIGH_J joint enrichment `1.3294648777718427`, top10 overlap `0.19193178146739545`.
QUANTITATIVE RESULT: Final train PERSISTENT_BND_HARD joint enrichment `1.5652489059524115`, top10 overlap `0.19126319866624003`.
QUANTITATIVE RESULT: Final train BND_HARD_CORE joint enrichment `1.523502487701078`, top10 overlap `0.22087305474961377`.
INFERENCE: Registered hard regions are enriched for the proxy burden, but prior BAP evidence established that hard-region observability alone is not causal actionability.

## Medium / Direction Alignment
QUANTITATIVE RESULT: Final train Spearman(joint_depth_opacity,tau) `0.11867766079108623`; Spearman(joint_depth_opacity,transmission) `-0.11895944374450973`.
QUANTITATIVE RESULT: Train Spearman(joint_depth_opacity,tau) was `0.09600992039506841` at 5k, `0.14185610913136734` at 10k, and `0.11867766079108623` at final.
INFERENCE: There is a reproducible but modest medium-load association; it is not enough to distinguish a strong medium-induced opacity-contamination mechanism from generic projected-geometry dispersion.

## Temporal Emergence
QUANTITATIVE RESULT: BND train fraction RZ_REL>1 was `0.0005997026255566389` at 5k, `0.0016267642049796185` at 10k, and `0.001649322481909976` at final.
QUANTITATIVE RESULT: BND train joint_depth_opacity_mean was `0.037488725036382675` at 5k, `0.03919284790754318` at 10k, and `0.039814338088035583` at final.
INFERENCE: The signal is visible by 5k/10k, but the actionable high-RZ_REL tail remains very small.

## Decomposition Context
QUANTITATIVE RESULT: Final train BND `J_p99=0.8506316596269596`, `P_J_gt_1=0.0`, `tau_p90=1.1410107612609863`, `tau_p99=1.877296932935713`, `P_T_lt_0p1=0.0`, `P_c_gt_0p99=0.017516765429387328`, `P_abs_s_full_gt_5=0.017123359197282455`.

## Classification
INFERENCE: `DEPTH_OPACITY_NOT_SUPPORTED`.
RECOMMENDATION: `DO NOT train depth-aware alpha; close this alpha mechanism line`.

## Required Question Answers
INFERENCE: Q1: WaterSplatting+BND contains a measurable but not meaningful large depth-inconsistent population; final train RZ_REL>1 is `0.001649322481909976`.
INFERENCE: Q2: The projected pairs have high raw opacity on average, but exact rendering contribution is implementation-blocked; proxy opacity alone is insufficient.
INFERENCE: Q3: The burden is mildly stronger under BND than M1, but absolute deltas are small.
INFERENCE: Q4: Alignment with BND-specific RGB regression is weak, with maximum train joint Spearman `0.048076310711427414`.
INFERENCE: Q5: Alignment with registered hard regions exists, with final train joint enrichment from `1.3294648777718427` to `1.5652489059524115`, but this is not sufficient support.
INFERENCE: Q6: Medium association exists but is modest; final train joint-vs-tau Spearman is `0.11867766079108623`.
INFERENCE: Q7: The signal appears by 5k/10k but remains too small and weakly aligned to justify an early preventive alpha intervention.
CODE FACT: Q8: Historical CDEPTH is pseudo-depth supervision on rendered depth/disparity through a Pearson correlation loss; a future alpha-only mechanism would modulate/prune opacity and would not optimize rendered depth to pseudo-depth.
INFERENCE: Q9: Final classification is `DEPTH_OPACITY_NOT_SUPPORTED`.
RECOMMENDATION: Q10: The one next experiment is no experiment on this line: do not train depth-aware alpha and close the alpha mechanism line.

## Outputs
- `outputs/bnd_depth_opacity_audit_20260816/`
