# Data-Split Feasibility Audit (2026-08-31)

## 1. Motivation

INFERENCE: Candidate C is data-limited, so this audit asks whether one larger geometry-aware split and four fresh locked-OCMC runs are worth approving. This is feasibility evidence, not Candidate-C support. It does not train a model or apply a split.

## 2. Why Candidate C Is Data-Limited

EXPERIMENTAL FACT: the current genuine held-out counts are Curasao 3, IUI3-RedSea 4, JapaneseGradens-RedSea 3, and Panama 3. Every scene is below the five-camera reliability threshold for camera-neighbor, permutation, and controlled rank analyses.

## 3. Camera Inventory

DATA FACT: independent COLMAP, source RGB, dataparser, and formal-list verification recovered 88 calibrated RGB cameras with no unused calibrated GT cameras.

| Scene | Formal train | Formal eval | Total | RGB and calibration complete |
| --- | ---: | ---: | ---: | --- |
| Curasao | 18 | 3 | 21 | True |
| IUI3-RedSea | 25 | 4 | 29 | True |
| JapaneseGradens-RedSea | 17 | 3 | 20 | True |
| Panama | 15 | 3 | 18 | True |

## 4. Split-Design Constraints

CONFIG FACT: candidate IDs were constructed only from transformed camera centers and geometrically verified numeric acquisition order. No old MSE, PSNR, LPIPS, residual ranking, or support ranking entered ID construction. After IDs were locked and hashed, final candidate-family ranking used view-direction coverage and GT-free support-coverage retention, as permitted by the protocol.

CONFIG FACT: the normalized camera context exactly follows `(camera_center - scene_box_center) / (scene_box_diagonal + 1e-6)`. Center and direction analyses remain separate; no learned or outcome-tuned combined metric is used.

## 5. Scene Geometry

| Scene | Frame-center rho | Frame-direction rho | Adjacent/all center ratio | Trajectory eligible |
| --- | ---: | ---: | ---: | --- |
| Curasao | 0.766 | 0.596 | 0.320 | True |
| IUI3-RedSea | 0.965 | 0.515 | 0.131 | True |
| JapaneseGradens-RedSea | 0.996 | 0.962 | 0.165 | True |
| Panama | 0.929 | 0.653 | 0.174 | True |

INFERENCE: the strong frame-center rank association and short adjacent steps verify numeric filename order as acquisition-trajectory evidence in all four scenes; filename order was not accepted without this geometry check.

## 6. Candidate Split Families

CODE FACT: exactly three proposals were audited per scene: A geometry-stratified novelty (N=5), B center farthest-point (N=7/8/6/6), and C trajectory-interleaved (N=6). This respects the three-proposal cap. All manifests were hashed before frozen support analysis.

## 7. Curasao Feasibility

| Rank | Family | Train | Held out | Classification | Center max/original | Direction max/original | Hull retention | Visibility retention |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | C | 15 | 6 | SPLIT_FEASIBLE | 1.000 | 1.348 | 0.997 | 0.992 |
| 2 | A | 16 | 5 | SPLIT_FEASIBLE_BUT_TIGHT | 0.546 | 1.152 | 0.598 | 0.970 |
| 3 | B | 14 | 7 | SPLIT_NOT_FEASIBLE | 0.842 | 1.152 | 0.484 | 0.971 |

INFERENCE: audited feasible held-out sizes are [5, 6]; recommend `Curasao:C:N6` with 15 train and 6 held-out cameras.

## 8. IUI3-RedSea Feasibility

| Rank | Family | Train | Held out | Classification | Center max/original | Direction max/original | Hull retention | Visibility retention |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | A | 24 | 5 | SPLIT_FEASIBLE | 0.762 | 1.000 | 0.898 | 1.000 |
| 2 | C | 23 | 6 | SPLIT_FEASIBLE | 1.122 | 1.094 | 0.925 | 1.000 |
| 3 | B | 21 | 8 | SPLIT_FEASIBLE | 1.122 | 1.094 | 0.881 | 0.999 |

INFERENCE: audited feasible held-out sizes are [5, 6, 8]; recommend `IUI3-RedSea:A:N5` with 24 train and 5 held-out cameras.

## 9. JapaneseGradens-RedSea Feasibility

| Rank | Family | Train | Held out | Classification | Center max/original | Direction max/original | Hull retention | Visibility retention |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | C | 14 | 6 | SPLIT_FEASIBLE | 2.021 | 1.276 | 0.961 | 0.999 |
| 2 | A | 15 | 5 | SPLIT_FEASIBLE_BUT_TIGHT | 1.782 | 1.276 | 0.661 | 1.000 |
| 3 | B | 14 | 6 | SPLIT_FEASIBLE_BUT_TIGHT | 1.921 | 0.888 | 0.679 | 0.974 |

INFERENCE: audited feasible held-out sizes are [5, 6]; recommend `JapaneseGradens-RedSea:C:N6` with 14 train and 6 held-out cameras.

## 10. Panama Feasibility

| Rank | Family | Train | Held out | Classification | Center max/original | Direction max/original | Hull retention | Visibility retention |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | A | 13 | 5 | SPLIT_FEASIBLE | 0.921 | 0.929 | 0.890 | 0.996 |
| 2 | B | 12 | 6 | SPLIT_NOT_FEASIBLE | 1.224 | 1.059 | 0.901 | 0.921 |
| 3 | C | 12 | 6 | SPLIT_NOT_FEASIBLE | 1.000 | 1.140 | 0.959 | 0.921 |

INFERENCE: audited feasible held-out sizes are [5]; recommend `Panama:A:N5` with 13 train and 5 held-out cameras.

## 11. Retained Training Coverage

| Scene | Center p90 ratio | Center max ratio | Direction p90 ratio | Direction max ratio | PCA hull retention | Severe hole |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Curasao | 1.619 | 1.000 | 1.000 | 1.348 | 0.997 | False |
| IUI3-RedSea | 0.978 | 0.762 | 0.891 | 1.000 | 0.898 | False |
| JapaneseGradens-RedSea | 1.384 | 2.021 | 0.865 | 1.276 | 0.961 | False |
| Panama | 1.014 | 0.921 | 0.999 | 0.929 | 0.890 | False |

INFERENCE: all recommended splits retain acceptable center and orientation coverage and none creates a severe train-manifold hole under the fixed thresholds in the audit code.

## 12. Held-Out Novelty Coverage

| Scene | Held out | Center novelty min/median/max | Direction novelty min/median/max (deg) | LOW/MID/HIGH |
| --- | ---: | --- | --- | --- |
| Curasao | 6 | 0.072/0.092/0.185 | 0.630/1.451/4.544 | True |
| IUI3-RedSea | 5 | 0.020/0.085/0.172 | 1.274/1.898/4.159 | True |
| JapaneseGradens-RedSea | 6 | 0.088/0.108/0.132 | 0.527/2.189/3.899 | True |
| Panama | 5 | 0.034/0.161/0.311 | 1.191/2.260/3.836 | True |

QUANTITATIVE RESULT: all recommendations span low/mid/high center-novelty ranks and pass the independent meaningful direction-novelty threshold.

## 13. Support Coverage

CONFIG FACT: only after split manifests were frozen and hashed, old C0 checkpoints supplied GT-free `gaussian_visible_mask` values. The framework datamanager may cache image batches during setup, but the audit logic never reads target image tensors or residuals. These support values are coverage-potential proxies because fresh-resplit Gaussians may differ.

| Scene | Visibility retention | Normalized mean-support retention | Unseen range | Low-support range | Mean-support range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Curasao | 0.992 | 0.991 | 0.0078 | 0.0340 | 3.620 |
| IUI3-RedSea | 1.000 | 1.016 | 0.0005 | 0.1153 | 9.632 |
| JapaneseGradens-RedSea | 0.999 | 1.016 | 0.0011 | 0.0810 | 4.794 |
| Panama | 0.996 | 0.928 | 0.0036 | 0.1049 | 5.293 |

| Scene | Support fraction 0/1/2/>=3 | Support mean/median |
| --- | --- | --- |
| Curasao | 0.0084/0.0647/0.0514/0.8755 | 9.880/11.0 |
| IUI3-RedSea | 0.0004/0.1097/0.1557/0.7341 | 10.560/9.0 |
| JapaneseGradens-RedSea | 0.0014/0.1064/0.1179/0.7743 | 7.930/6.0 |
| Panama | 0.0037/0.1880/0.0531/0.7553 | 6.313/6.0 |

INFERENCE: all four retain acceptable visibility/Gaussian support and span meaningful GT-free held-out support variation. The old-split representation makes this a bounded feasibility proxy, not a prediction of fresh-run support.

## 14. Statistical Power / Feasibility

| Scene | Held out | Independent pairs | LOO neighbor | Permutation | Single control | Multivariable regression |
| --- | ---: | ---: | --- | --- | --- | --- |
| Curasao | 6 | 15 | True | True | True | False |
| IUI3-RedSea | 5 | 10 | True | True | True | False |
| JapaneseGradens-RedSea | 6 | 15 | True | True | True | False |
| Panama | 5 | 10 | True | True | True | False |

INFERENCE: all four recommendations reach N>=5; two reach N>=6. Leave-one-view-out and 1,000 fixed-seed camera-label permutations become more meaningful than the original split, but exact permutation resolution remains coarse. Only one-control-at-a-time rank residualization is approved; multivariable regression remains inappropriate.

## 15. Compute-Cost Estimate

| Scene | Historical C0 wall time (s) | GPU-hours | Peak reserved (GiB) | GPU |
| --- | ---: | ---: | ---: | --- |
| Curasao | 3408.67 | 0.947 | 5.51 | NVIDIA GeForce RTX 3080 |
| IUI3-RedSea | 3019.83 | 0.839 | 3.78 | NVIDIA GeForce RTX 3080 |
| JapaneseGradens-RedSea | 3135.91 | 0.871 | 3.85 | NVIDIA GeForce RTX 3080 |
| Panama | 3907.99 | 1.086 | 5.60 | NVIDIA GeForce RTX 3080 |

QUANTITATIVE RESULT: four fresh 15K runs require an estimated 3.742 training GPU-hours, 3.752 GPU-hours including frozen diagnostics, 1.088 hours with four GPUs in parallel, and 15.46 GB. Cost is `LOW_COST` relative to prior formal experiments.

## 16. Pre-Registered Future Candidate-C Criteria

HYPOTHESIS: the camera is the unit and `E_cam = MSE` is the target. Fixed GT-free predictors are: `fraction_visible_unseen_train`, `camera_center_nearest_train`, `camera_center_knn3_mean`, `camera_context_nearest_train`, `camera_context_knn3_mean`, `view_direction_nearest_angle`, `view_direction_knn3_angle`, `mean_train_visibility_support`, `median_train_visibility_support`, `fraction_visible_low_support`. `fraction_visible_unseen_train` is preregistered from the prior audit before any new-split outcome exists.

C_SUPPORTED_AND_ACTIONABLE requires all of: same-direction Candidate-C replication in at least 3/4 scenes; at least 3/4 scenes have at least five genuine held-out views; a pre-registered GT-free predictor has absolute Spearman rho >= 0.4 in at least 3/4 scenes; predictor direction survives major single-factor controls; positive camera-neighbor structure in at least three adequate-N scenes; signal is not reducible to OCMC observability; no scene has a strong opposite-direction contradiction.

C_NOT_SUPPORTED applies if any of: Candidate C replicates in at most 2/4 scenes; pre-registered GT-free predictors are weak or inconsistent; major controls explain the effect. Its consequence is to close Candidate C without more split variants.

## 17. Recommended Split Proposal

### Curasao

TRAIN (15): `MTN_1288`, `MTN_1290`, `MTN_1291`, `MTN_1292`, `MTN_1294`, `MTN_1295`, `MTN_1297`, `MTN_1298`, `MTN_1299`, `MTN_1301`, `MTN_1302`, `MTN_1304`, `MTN_1305`, `MTN_1306`, `MTN_1308`.

HELDOUT (6): `MTN_1289`, `MTN_1293`, `MTN_1296`, `MTN_1300`, `MTN_1303`, `MTN_1307`.

Hash: `9ab62359f5a860886ca86367d8e868adef4a4121ebe32355927227da63dfb735`.

### IUI3-RedSea

TRAIN (24): `MTN_5895`, `MTN_5896`, `MTN_5899`, `MTN_5900`, `MTN_5901`, `MTN_5902`, `MTN_5904`, `MTN_5905`, `MTN_5906`, `MTN_5907`, `MTN_5908`, `MTN_5909`, `MTN_5910`, `MTN_5911`, `MTN_5912`, `MTN_5913`, `MTN_5914`, `MTN_5915`, `MTN_5917`, `MTN_5927`, `MTN_5928`, `MTN_5929`, `MTN_5930`, `MTN_5931`.

HELDOUT (5): `MTN_5894`, `MTN_5898`, `MTN_5903`, `MTN_5916`, `MTN_5933`.

Hash: `caf4ad6ec74edb145093d8f0cee423c0b2f8a038d2fb4b23b3fc08f0bd3438a3`.

### JapaneseGradens-RedSea

TRAIN (14): `MTN_1090`, `MTN_1092`, `MTN_1093`, `MTN_1095`, `MTN_1096`, `MTN_1097`, `MTN_1099`, `MTN_1100`, `MTN_1102`, `MTN_1103`, `MTN_1104`, `MTN_1106`, `MTN_1107`, `MTN_1109`.

HELDOUT (6): `MTN_1091`, `MTN_1094`, `MTN_1098`, `MTN_1101`, `MTN_1105`, `MTN_1108`.

Hash: `afb39101ed59294ec818d6f4962b10f224c387d56555a8c8842c0f8d8a053bf2`.

### Panama

TRAIN (13): `MTN_1532`, `MTN_1534`, `MTN_1535`, `MTN_1537`, `MTN_1538`, `MTN_1540`, `MTN_1541`, `MTN_1542`, `MTN_1544`, `MTN_1545`, `MTN_1546`, `MTN_1547`, `MTN_1548`.

HELDOUT (5): `MTN_1529`, `MTN_1533`, `MTN_1536`, `MTN_1539`, `MTN_1543`.

Hash: `8b7584daaad06464b1d2a47fa8e279b9d7970416e57d32d5cc86df452a4dde1a`.

Global manifest hash: `8615b61e3d4d7f3355a196e41708d2774110d6f025fdfa61d0f41e5b426ad465`.

## 18. GO / NO-GO Decision

INFERENCE: `C_SPLIT_RETRAIN_GO` and `CONTINUE_C_WITH_NEW_SPLIT`. Four of four recommended splits are `SPLIT_FEASIBLE`; expected information gain is sufficient relative to the low cost. This does not alter the original OCMC result and does not establish Candidate C.

Largest scientific risk: a geometry-selected split may change the reconstruction problem enough that residual differences reflect resplit-induced representation change rather than stable Candidate-C structure.

Largest compute/engineering risk: future split plumbing or checkpoint provenance drift could invalidate four otherwise inexpensive 15K runs.

## 19. ONE Next Task

HYPOTHESIS: `OCMC-CANDIDATE-C-RESPLIT-CAUSAL-REPLICATION`. Use exactly one preregistered split and one fresh locked-OCMC run per scene; do not use k-fold by default.
