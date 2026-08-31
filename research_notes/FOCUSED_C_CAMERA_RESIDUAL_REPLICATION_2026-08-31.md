# Focused Candidate-C Camera Residual Replication (2026-08-31)

## 1. Motivation

This frozen-state audit tests whether camera/view novelty or training-view support predicts genuine held-out camera error after locked OCMC. It does not train or design an intervention.

## 2. Previous Candidate-C Evidence

EXPERIMENTAL FACT: the preceding audit classified Candidate C as tentative: camera-neighborhood structure was descriptive in IUI3-RedSea and JapaneseGradens-RedSea, but only 13 formal eval cameras existed across four scenes.

## 3. Held-out Camera Coverage Audit

CODE FACT: COLMAP `images.bin`, source `ColorImage` files, formal split lists, and the dataparser-loaded train/eval IDs were cross-checked. Counts are Curasao 18/3 train/eval; IUI3-RedSea 25/4 train/eval; JapaneseGradens-RedSea 17/3 train/eval; Panama 15/3 train/eval.

## 4. Additional Unused-Camera Discovery

DATA FACT: no `UNUSED_CALIBRATED_GT` camera exists in any scene. Every calibrated source camera is already in formal train or formal eval, and every source RGB has calibration.

## 5. Frozen OCMC Rendering Protocol

CONFIG FACT: only formal C0 step-14999 checkpoints were loaded. Rendering used classic rasterization, bounded SH3, `dir_xy_camera`, OCMC on, and RAOC off. Formal train views supplied only GT-free camera/support geometry; their residuals were never held-out labels. No backward or optimizer step occurred.

## 6. Camera-Level Error Statistics

QUANTITATIVE RESULT: E_cam mean/std/range are Curasao 0.000737931/0.000549872/[0.000327597, 0.00151516]; IUI3-RedSea 0.00153721/0.00169046/[0.000186541, 0.00442875]; JapaneseGradens-RedSea 0.00555827/0.00523457/[0.00141577, 0.0129428]; Panama 0.000703838/9.65397e-05/[0.00056911, 0.00079034].

## 7. Camera-Center/Context Novelty

QUANTITATIVE RESULT: nearest-center Spearman rho with E_cam is Curasao 1.000; IUI3-RedSea -0.400; JapaneseGradens-RedSea 0.500; Panama -0.500. Exact OCMC-context nearest-distance rho is Curasao 1.000; IUI3-RedSea -0.400; JapaneseGradens-RedSea 0.500; Panama -0.500. All are descriptive because N=3/4/3/3. Center and context ranks can coincide because the OCMC context is an affine scene normalization of camera center.

## 8. View-Direction Novelty

QUANTITATIVE RESULT: nearest angular-novelty rho with E_cam is Curasao 1.000; IUI3-RedSea -0.400; JapaneseGradens-RedSea -1.000; Panama -0.500. Minimum and fixed 3-NN angular novelty were evaluated independently, without a tuned center-angle score.

## 9. Training-View Support

CODE FACT: each held-out visible Gaussian was assigned its exact formal-training visibility count. The fraction-visible-with-zero-training-support rho with E_cam is Curasao 0.866; IUI3-RedSea 0.632; JapaneseGradens-RedSea 0.866; Panama 0.500. Mean/median support and zero/one-view support fractions are GT-free; contribution-weighted support was not claimed.

## 10. Optical/Geometric Controls

QUANTITATIVE RESULT: depth, tau, transmission, accumulation, projected footprint, visible count, and visibility support were examined by one-control-at-a-time within-scene rank residualization. With fewer than five cameras, these are descriptive controls rather than a reliable multivariate adjustment. Independence from these confounders and from OCMC projected camera-residual magnitude is not established.

## 11. Leave-One-View-Out Neighbor Analysis

QUANTITATIVE RESULT: center-space leave-one-view-out scores are Curasao -1.000 (INSUFFICIENT_CAMERA_COUNT); IUI3-RedSea -1.000 (INSUFFICIENT_CAMERA_COUNT); JapaneseGradens-RedSea -1.000 (INSUFFICIENT_CAMERA_COUNT); Panama -1.000 (INSUFFICIENT_CAMERA_COUNT). All four scenes are `INSUFFICIENT_CAMERA_COUNT`; no camera-label permutation test was run or interpreted. Center-distance versus absolute E_cam-difference rho is Curasao -0.500; IUI3-RedSea 0.029; JapaneseGradens-RedSea 0.500; Panama -0.500, also descriptive only.

## 12. Formal-Eval vs Additional-Heldout Comparison

DATA FACT: the additional-heldout population is empty, so combined genuine-heldout results equal formal-eval-only results and no independent expansion comparison exists.

## 13. Cross-Scene Replication

INFERENCE: no scene has adequate coverage for the protocol's reliable camera-neighbor test. The previous IUI3 and JapaneseGradens center-neighbor difference ratios reproduce numerically on the same formal-eval cameras, but this is not an expanded replication and cannot be upgraded to replicated evidence. Curasao and Panama do not newly establish Candidate C.

## 14. GT-Free Actionability

INFERENCE: no GT-free predictor is actionable under this audit. `fraction_visible_unseen_train` is only the strongest small-N directional summary, not a validated predictor.

## 15. Final Candidate-C Decision

INFERENCE: `C_DATA_LIMITED`. This does not support or refute Candidate C; it records that every scene remains below five genuine held-out cameras and no additional cameras exist.

## 16. Closed / Remaining Uncertainties

RAOC remains closed and OCMC remains frozen. Candidate C remains unresolved rather than supported. Camera-neighbor effects, optical-confounder independence, and OCMC independence cannot be reliably adjudicated with current held-out coverage.

## 17. ONE Next Task

HYPOTHESIS: `DATA-SPLIT-FEASIBILITY-AUDIT`. Audit whether moving enough currently trained cameras to reach at least five (preferably eight) held-out views per scene, followed by four fresh locked-OCMC retrains, is scientifically worth the cost. Do not retrain during that audit.
