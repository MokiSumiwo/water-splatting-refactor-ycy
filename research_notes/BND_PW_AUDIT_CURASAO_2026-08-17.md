# BND-PW-AUDIT-CURASAO

## Scope
CONFIG FACT: This is a read-only, zero-training audit. No optimizer step, checkpoint write, new loss training, CDEPTH, OMVC, depth-aware alpha, CB-FG, or CB-BG training is performed.

## Repo
EXPERIMENTAL FACT: Branch `research/m1-bounded-intrinsic`, HEAD `96ec08a0057e7a8f356e7576085b2783cab76a99`.

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

## Available Curasao Checkpoints
EXPERIMENTAL FACT: BND available requested-to-actual map `{'3000': 3000, '5000': 5000, '8000': 8000, '10000': 10000, '13000': 13000, '15000': 14999}`.
EXPERIMENTAL FACT: M1 available requested-to-actual map `{'5000': 5000, '10000': 10000, '15000': 14999}`.

## Pure-Water Candidate Coverage
QUANTITATIVE RESULT: Train M_SAFE pooled coverage `0.04173292860893505`; train views >=1 percent `18`.
QUANTITATIVE RESULT: Eval M_SAFE pooled coverage `0.09178556280005555`.

## Late Object Contamination
QUANTITATIVE RESULT: Final BND train mean fraction accumulation>0.01 across views `0.27303126868274474`.
QUANTITATIVE CONCLUSION: `LOW_OBJECT_CONTAMINATION_LOCKED_RULE = False`.

## B_inf / Medium Headroom
QUANTITATIVE RESULT: Final BND train BINF_L1 view mean `0.0052376006367719835`; R_anchor view mean `0.012957839604056259`.
QUANTITATIVE CONCLUSION: `BACKGROUND_HEADROOM_LOCKED_RULE = False`.

## B_inf Saturation
EXPERIMENTAL FACT: B_inf saturation and recovered pre-sigmoid logit statistics are stored in `binf_saturation.csv/json`.

## Medium-Only Gradient Pathway
QUANTITATIVE RESULT: medium_mlp grad L2 `0.021889758692652417`, direction_encoding grad L2 `0.0`, max object grad L2 `0.0`.
EXPERIMENTAL FACT: Head/trunk parameter split status `PARAMETER_SUBGROUP_NOT_EXPOSED_FOR_CURRENT_MEDIUM_MLP`.
QUANTITATIVE CONCLUSION: `MEDIUM_DOMINANT_GRADIENT_ROUTE = True`; parameter delta max `0.0`.

## M1 vs BND Comparison
EXPERIMENTAL FACT: Matched M1/BND comparison rows are stored in `m1_bnd_background_comparison.csv/json` for 5k, 10k, and final.

## View / Direction Stability
QUANTITATIVE CONCLUSION: `VIEW_STABILITY_RULE = False`; channel sign consistency `{'R': 0.5, 'G': 0.6111111111111112, 'B': 0.6111111111111112}`.

## Decomposition Context
EXPERIMENTAL FACT: BND decomposition context rows are stored in `decomposition_context.csv/json`.

## Classification
INFERENCE: `BG_ANCHOR_WEAK`.

## Next Single Experiment
RECOMMENDATION: `Apply the same locked BND-PW-AUDIT to IUI3; do not train CB-BG on Curasao`.

## Required Question Answers
INFERENCE: Q1 is answered by comparing `train_M_SAFE_coverage` against Panama's `0.0014541265330494672` in the final report.
INFERENCE: Q2 depends on train views >=1 percent and pooled coverage under the locked rule.
INFERENCE: Q3 depends on final BND accumulation on M_SAFE.
INFERENCE: Q4 depends on final BND BINF_L1 and R_anchor.
INFERENCE: Q5 depends on the no-step B_inf probe gradient rows.
INFERENCE: Q6 depends on view stability rows.
INFERENCE: Q7 is reported in the M1/BND comparison table.
INFERENCE: Q8 final classification is `BG_ANCHOR_WEAK`.
RECOMMENDATION: Q9 one next experiment is `Apply the same locked BND-PW-AUDIT to IUI3; do not train CB-BG on Curasao`.
