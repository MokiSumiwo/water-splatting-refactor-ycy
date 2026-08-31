# OCMC Candidate-C Resplit Replication (2026-08-31)

## 1. Motivation

HYPOTHESIS: this preregistered replication asks whether held-out camera residual structure survives a larger outcome-blind split after locked OCMC. The split change is not a causal RGB arm.

## 2. Previous C_DATA_LIMITED Result

EXPERIMENTAL FACT: the old split had only 3/4/3/3 held-out cameras. Candidate C remained data-limited despite a positive descriptive unseen-fraction direction in all four scenes.

## 3. Pre-Registered Split Provenance

CONFIG FACT: all four scene hashes and global hash `8615b61e3d4d7f3355a196e41708d2774110d6f025fdfa61d0f41e5b426ad465` matched the feasibility artifact before training. Output-local list files and read-only source-data symlinks preserved the official split files.

| Scene | Train | Heldout | Locked split SHA-256 |
| --- | ---: | ---: | --- |
| Curasao | 15 | 6 | `9ab62359f5a860886ca86367d8e868adef4a4121ebe32355927227da63dfb735` |
| IUI3-RedSea | 24 | 5 | `caf4ad6ec74edb145093d8f0cee423c0b2f8a038d2fb4b23b3fc08f0bd3438a3` |
| JapaneseGradens-RedSea | 14 | 6 | `afb39101ed59294ec818d6f4962b10f224c387d56555a8c8842c0f8d8a053bf2` |
| Panama | 13 | 5 | `8b7584daaad06464b1d2a47fa8e279b9d7970416e57d32d5cc86df452a4dde1a` |

## 4. Formal Training Protocol

CONFIG FACT: each scene used one fresh seed-42, 15K C0/OCMC run with bounded SH3, SH degree 3, classic rasterization, `dir_xy_camera`, tied B_inf, refreshes at 0/5000/10000, formal refinement through stop_split_at=10000, and RAOC disabled. Held-out IDs were absent from optimization sequences and OCMC banks.

PROTOCOL FACT: all four construction preflights passed before training. Formal workers reproduced the locked split, config, seed-42 start-state, 15K camera sequence, and train-only OCMC bank hashes. All four final checkpoints identify this experiment, branch C0, and step 14999. Frozen evaluation used no optimizer step or backward call.

## 5. New OCMC Training Sanity

| Scene | Steps | Train s | Eval s | Final Gaussians | Peak reserved GiB | PSNR | SSIM | LPIPS | MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Curasao | 15000 | 3563.70 | 16.25 | 1042041 | 5.55 | 30.989 | 0.9434 | 0.1172 | 0.00152201 |
| IUI3-RedSea | 15000 | 2999.23 | 12.47 | 758482 | 3.78 | 31.823 | 0.9087 | 0.1855 | 0.0014015 |
| JapaneseGradens-RedSea | 15000 | 3074.15 | 11.39 | 823209 | 3.84 | 25.748 | 0.9162 | 0.1061 | 0.00358877 |
| Panama | 15000 | 3555.04 | 13.89 | 1184969 | 5.76 | 32.165 | 0.9350 | 0.0954 | 0.000819089 |

RUNTIME FACT: total training cost was 3.664 GPU-hours and parallel worker wall-clock was 0.999 hours. This is 2.3% from the preregistered 3.75 GPU-hour estimate and is classified close to estimate. No NaN, Inf, OOM, or missing heldout render occurred.

## 6. Heldout Camera Population

QUANTITATIVE RESULT: all 22 preregistered heldout cameras rendered. `E_cam` is mean per-pixel squared RGB residual.

| Scene | N | E_cam min | median | mean | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Curasao | 6 | 0.000281782 | 0.000766678 | 0.00152201 | 0.00351484 | 0.00581455 |
| IUI3-RedSea | 5 | 0.000192917 | 0.000414982 | 0.0014015 | 0.00337748 | 0.00448638 |
| JapaneseGradens-RedSea | 6 | 0.00125675 | 0.0024284 | 0.00358877 | 0.00695031 | 0.0103743 |
| Panama | 5 | 0.000122257 | 0.000820806 | 0.000819089 | 0.001393 | 0.0016785 |

## 7. Primary GT-Free Predictor

| Scene | unseen-fraction rho | Kendall tau | Controls survive |
| --- | ---: | ---: | --- |
| Curasao | 0.541 | 0.430 | True |
| IUI3-RedSea | 0.707 | 0.632 | True |
| JapaneseGradens-RedSea | 0.928 | 0.828 | True |
| Panama | -0.400 | -0.400 | False |

INFERENCE: the primary rho reached +0.4 in 3/4 scenes and survived major controls in 3/4. Its decision is `UNSEEN_FRACTION_SUPPORTED`. This is association evidence only, not a causal support-error claim.

## 8. Camera-Center/Context Novelty

| Scene | Center nearest | Center 3NN | Context nearest | Context 3NN |
| --- | ---: | ---: | ---: | ---: |
| Curasao | 0.086 | 0.086 | 0.086 | 0.086 |
| IUI3-RedSea | -0.400 | 0.300 | -0.400 | 0.300 |
| JapaneseGradens-RedSea | 0.314 | 0.371 | 0.314 | 0.371 |
| Panama | 0.600 | 0.300 | 0.600 | 0.300 |

INFERENCE: center and exact OCMC-context ranks match here because `dir_xy_camera` is derived from scene-normalized camera center. Neither distance family replicates at the +0.4 threshold in three scenes.

## 9. View-Direction Novelty

| Scene | Nearest angle rho | 3NN angle rho |
| --- | ---: | ---: |
| Curasao | 0.486 | 0.314 |
| IUI3-RedSea | -0.700 | -0.200 |
| JapaneseGradens-RedSea | -0.086 | 0.314 |
| Panama | -1.000 | -0.900 |

INFERENCE: view-direction novelty is inconsistent and strongly opposite in IUI3 nearest-angle and both Panama angle summaries.

## 10. Training-View Support

CODE FACT: visible means final `radii > 0`, exactly as in the focused Candidate-C audit; equality with `gaussian_visible_mask` was asserted for every support/eval render. Train support is the number of preregistered train cameras in which that Gaussian is visible; low support means count <=1. Definitions were locked before outcomes.

| Scene | Mean support rho | Median support rho | Low-support fraction rho |
| --- | ---: | ---: | ---: |
| Curasao | -0.600 | -0.372 | 0.406 |
| IUI3-RedSea | -0.400 | -0.359 | 0.821 |
| JapaneseGradens-RedSea | -0.600 | -0.676 | 0.754 |
| Panama | -0.700 | -0.821 | 0.800 |

INFERENCE: `fraction_visible_low_support` is the strongest cross-scene preregistered predictor: expected-direction |rho| >=0.4 in 4/4 scenes, with median expected-signed rho 0.777.

## 11. Confounder Controls

QUANTITATIVE RESULT: one-control-at-a-time rank residualization used the same seven factors in every scene. No multivariable regression was fit.

| Scene | Raw | Depth | Tau | Transmission | Accumulation | Footprint | Visible count | Mean support | Survives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Curasao | 0.541 | 0.600 | 0.600 | 0.600 | 0.543 | 0.371 | 0.371 | 0.029 | True |
| IUI3-RedSea | 0.707 | 0.667 | 0.700 | 0.700 | 0.100 | 0.900 | 0.500 | 0.500 | True |
| JapaneseGradens-RedSea | 0.928 | 0.943 | 0.943 | 0.886 | 0.943 | 0.943 | 0.943 | 0.943 | True |
| Panama | -0.400 | 0.000 | -0.300 | -0.400 | -0.100 | 0.000 | 0.000 | 0.000 | False |

INFERENCE: all seven controlled directions remain positive in Curasao, IUI3, and JapaneseGradens; none is positive in Panama. No single registered control met the full-explanation rule.

## 12. Camera-Neighbor Analysis

| Scene | Center LOO rho | Center rank MAE | Direction LOO rho | Direction rank MAE |
| --- | ---: | ---: | ---: | ---: |
| Curasao | -0.657 | 0.444 | -0.543 | 0.444 |
| IUI3-RedSea | -0.700 | 0.480 | -0.700 | 0.480 |
| JapaneseGradens-RedSea | -0.429 | 0.444 | -0.429 | 0.444 |
| Panama | -0.600 | 0.480 | -0.600 | 0.480 |

QUANTITATIVE RESULT: the fixed inverse-distance LOO score is negative in 0/4 positive center scenes and 0/4 positive direction scenes. Self-neighbor use was programmatically false. This blocks `C_SUPPORTED_AND_ACTIONABLE`.

## 13. Permutation Analysis

| Scene | Space | Observed | Null median | Null p95 | Percentile | Exact N! |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | center | -0.657 | -0.600 | -0.086 | 0.449 | 720 |
| Curasao | view_direction | -0.543 | -0.600 | -0.086 | 0.558 | 720 |
| IUI3-RedSea | center | -0.700 | -0.700 | -0.100 | 0.625 | 120 |
| IUI3-RedSea | view_direction | -0.700 | -0.700 | -0.395 | 0.767 | 120 |
| JapaneseGradens-RedSea | center | -0.429 | -0.600 | 0.031 | 0.669 | 720 |
| JapaneseGradens-RedSea | view_direction | -0.429 | -0.600 | -0.086 | 0.696 | 720 |
| Panama | center | -0.600 | -0.800 | -0.290 | 0.775 | 120 |
| Panama | view_direction | -0.600 | -0.800 | -0.095 | 0.750 | 120 |

QUANTITATIVE RESULT: no observed neighbor score exceeded its exact permutation null p95.

## 14. Pair-Distance Analysis

| Scene | Center rho | View-angle rho | Center pair direction |
| --- | ---: | ---: | --- |
| Curasao | -0.168 | 0.321 | False |
| IUI3-RedSea | 0.018 | 0.709 | True |
| JapaneseGradens-RedSea | 0.179 | -0.014 | True |
| Panama | 0.091 | -0.394 | True |

INFERENCE: center pair-distance direction is positive in IUI3, JapaneseGradens, and Panama, but all three effects are small. Camera pairs are descriptive; scene remains the replication unit.

## 15. Old-vs-New Candidate-C Replication

| Scene | Old N | Old unseen rho | New N | New unseen rho | Direction replicated |
| --- | ---: | ---: | ---: | ---: | --- |
| Curasao | 3 | 0.866 | 6 | 0.541 | True |
| IUI3-RedSea | 4 | 0.632 | 5 | 0.707 | True |
| JapaneseGradens-RedSea | 3 | 0.866 | 6 | 0.928 | True |
| Panama | 3 | 0.500 | 5 | -0.400 | False |

CONFIG FACT: only old/new directions and sample counts were compared. Absolute PSNR or other reconstruction differences were not interpreted as a causal Candidate-C effect.

## 16. OCMC-Independence Analysis

| Scene | Unseen vs OCMC magnitude | OCMC magnitude vs E_cam | Unseen vs E_cam controlled |
| --- | ---: | ---: | ---: |
| Curasao | 0.270 | -0.257 | 0.771 |
| IUI3-RedSea | 0.354 | 0.800 | 0.800 |
| JapaneseGradens-RedSea | 0.812 | 0.714 | 0.657 |
| Panama | -0.100 | 0.000 | -0.400 |

INFERENCE: distinguishable from OCMC observability is `True` because controlled direction remains positive in 3/4 scenes. OCMC g_obs remains global/mode-level; Candidate C predictors are per-view coverage statistics.

DECOMPOSITION FACT: new-split OCMC remained safe on every scene.

| Scene | P(J>1) | J p99 | Tau p90/p99 | P(T<0.1) | B_inf mean | beta_B mean | beta_D mean | OCMC modes <0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Curasao | 0 | 0.952 | 1.631/2.178 | 0.0004 | 0.189 | 0.123 | 0.229 | 4 |
| IUI3-RedSea | 0 | 0.877 | 1.928/2.763 | 0.0475 | 0.329 | 0.299 | 0.131 | 4 |
| JapaneseGradens-RedSea | 0 | 0.958 | 1.003/1.835 | 0.0014 | 0.248 | 0.157 | 0.087 | 4 |
| Panama | 0 | 0.797 | 1.324/1.824 | 0.0000 | 0.220 | 0.150 | 0.159 | 4 |

MECHANISM SANITY: OCMC was active, RAOC was off, all nine global gates were finite and bounded, and four modes were below 0.5 in every scene. This is a sanity check, not a new baseline comparison.

## 17. Per-Scene Classifications

- Curasao: `WEAK_CAMERA_RESIDUAL_REPLICATION`; primary controls survive=True, center neighbor positive=False, center pair positive=False.
- IUI3-RedSea: `STRONG_CAMERA_RESIDUAL_REPLICATION`; primary controls survive=True, center neighbor positive=False, center pair positive=True.
- JapaneseGradens-RedSea: `STRONG_CAMERA_RESIDUAL_REPLICATION`; primary controls survive=True, center neighbor positive=False, center pair positive=True.
- Panama: `WEAK_CAMERA_RESIDUAL_REPLICATION`; primary controls survive=False, center neighbor positive=False, center pair positive=True.

## 18. Cross-Scene Decision

INFERENCE: `C_SUPPORTED_BUT_NOT_ACTIONABLE`. Residual structure appears in 3/4 scenes and controls survive in 3/4. `fraction_visible_low_support` reaches the registered direction/threshold in 4/4, but center-neighbor structure is positive in only 0/4. The support association is replicated but is not actionable enough to justify a module.

## 19. Candidate-C Research-Line Decision

INFERENCE: `C_SCIENTIFICALLY_SUPPORTED_BUT_DEFER_MODULE`. No second split, seed, k-fold rescue, predictor sweep, or module design was performed. The largest remaining uncertainty is whether the replicated low-support association is a stable, inference-time measurable signal given absent LOO neighbor support and tied or near-zero unseen fractions.

## 20. ONE Next Task

HYPOTHESIS: `ISOLATE-FRACTION-VISIBLE-LOW-SUPPORT-PROXY`. Isolate the measurement stability and inference-time computability of the preregistered low-support-visible-Gaussian fraction without training a module or changing the split.
