# OCMC Residual Failure-Mode Audit (2026-08-31)

## 1. Motivation

This frozen-state audit asks which measurable failure remains after observability-controlled camera-conditioned medium context (OCMC). It does not design or train a new method.

## 2. Locked OCMC Status

CONFIG FACT: every audited state used bounded SH3 only as a controlled representation, `dir_xy_camera` scene-normalized camera-position context, OCMC enabled, and RAOC disabled.

## 3. Why RAOC Was Closed

EXPERIMENTAL FACT: the final hybrid CUDA feasibility attempt preserved tiny sensitivity error but failed the full modal reconstruction gate. RAOC remains scientifically archived and the formal line remains closed.

## 4. Audit Protocol

CODE FACT: all formal train and held-out eval cameras were rendered once at C0@14999. Deterministic per-view ray banks, projected Gaussian-center samples, normalized residual patches, and archived six-step topology trajectories were reused across candidates. No optimization, backward pass, or checkpoint mutation occurred.

The exact audited train/eval camera counts were Curasao 18/3, IUI3-RedSea 25/4, JapaneseGradens-RedSea 17/3, and Panama 15/3. IUI3 reused the exact locked train-view M_SAFE indices; no eval-derived mask was defined.

## 5. Candidate A: Cross-View Intrinsic Inconsistency

QUANTITATIVE RESULT: `RESIDUAL_FAILURE_NOT_SUPPORTED` (0/4 held-out scene support). Held-out angle/depth/radius-controlled Spearman values were Curasao 0.132, IUI3-RedSea 0.071, JapaneseGradens-RedSea 0.119, Panama 0.057.

INFERENCE: intrinsic variation was strongly associated with SH magnitude, but it did not consistently enrich held-out RGB error. Only JapaneseGradens-RedSea had a raw held-out rho above 0.20, while the controlled effect remained 0.119. Candidate A is closed.

## 6. Candidate B: Geometry-Medium Coupling

QUANTITATIVE RESULT: `RESIDUAL_FAILURE_TENTATIVE` (2/4). IUI3 beta_D-depth coupling was rho 0.787 with beta_D/error rho -0.517; JapaneseGradens medium-contribution/accumulation coupling was rho -0.508 with error rho -0.272.

INFERENCE: strong coupling exists in every scene for at least one pair, but only IUI3 and JapaneseGradens replicated the same pair across train/eval and also linked it to error. Correlation does not establish harmful coupling.

## 7. Candidate C: View-Dependent Residual Appearance

QUANTITATIVE RESULT: `RESIDUAL_FAILURE_TENTATIVE` (2/4). Eval-view MSE coefficients of variation were Curasao 0.745, IUI3-RedSea 1.100, JapaneseGradens-RedSea 0.942, Panama 0.137; nearest-camera/all-pair error-difference ratios were Curasao 1.018, IUI3-RedSea 0.698, JapaneseGradens-RedSea 0.578, Panama 1.087.

INFERENCE: difficult views were late-stage persistent, but camera-neighbor structure replicated only in IUI3 and JapaneseGradens. Three or four eval cameras per scene leave substantial small-population uncertainty.

## 8. Candidate D: Spatially Structured Medium / RGB Residual

QUANTITATIVE RESULT: `RESIDUAL_FAILURE_TENTATIVE` (1/4 held-out predictive support). Eval RGB residual Moran-like values ranged from 0.887 to 0.946, but nearest-train patch scores predicted held-out patch error only in IUI3.

INFERENCE: individual residual maps and medium contributions are spatially structured, as expected, but similar structure was not consistently predictive of held-out error across views. This prevents a supported classification.

## 9. Candidate E: Late-Stage Gaussian Representation Allocation

QUANTITATIVE RESULT: `RESIDUAL_FAILURE_NOT_SUPPORTED` (0/4). From 10k to 14999, Gaussian populations changed by Curasao -2.38%, IUI3-RedSea -3.86%, JapaneseGradens-RedSea -3.77%, Panama -2.50%.

INFERENCE: every scene pruned 2.38-3.86% of its Gaussians instead of continuing large topology growth. The registered failure signature is absent, so Candidate E is closed.

## 10. Candidate F: Depth / Observability Conditioned Residual

QUANTITATIVE RESULT: `RESIDUAL_FAILURE_TENTATIVE` (2/4). IUI3 low-tau score achieved rho 0.492, AUROC 0.815, and top/bottom error ratio 34.61; JapaneseGradens low-depth achieved rho 0.486, AUROC 0.752, and ratio 7.83.

INFERENCE: the effect is strong where present but the selected regime differs by scene and reverses or disappears in Curasao and Panama. It is also closest to OCMC observability, reducing independence.

## 11. Cross-Scene Comparison

QUANTITATIVE RESULT: A=RESIDUAL_FAILURE_NOT_SUPPORTED (0/4); B=RESIDUAL_FAILURE_TENTATIVE (2/4); C=RESIDUAL_FAILURE_TENTATIVE (2/4); D=RESIDUAL_FAILURE_TENTATIVE (1/4); E=RESIDUAL_FAILURE_NOT_SUPPORTED (0/4); F=RESIDUAL_FAILURE_TENTATIVE (2/4).

## 12. Priority Matrix

| Candidate | Persistence | Cross-scene | Held-out | Clarity | OCMC independence | Total |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 0 | 0 | 3 | 3 | 7 |
| B | 2 | 2 | 2 | 2 | 2 | 10 |
| C | 2 | 2 | 2 | 2 | 3 | 11 |
| D | 3 | 1 | 1 | 2 | 3 | 10 |
| E | 0 | 0 | 0 | 3 | 3 | 6 |
| F | 2 | 2 | 2 | 3 | 1 | 10 |

## 13. Primary Remaining Failure Mode

INFERENCE: `NO_SINGLE_DOMINANT_FAILURE_MODE`. No primary failure is selected because no candidate reached supported status in at least 3/4 scenes with held-out relevance. Candidate C is the highest-priority tentative result at 11/15, not a supported mechanism.

## 14. Closed Candidates

Candidates A and E are formally closed as `RESIDUAL_FAILURE_NOT_SUPPORTED`. B, C, D, and F remain tentative only; none motivates module design yet.

## 15. What Should Be Tested Next

HYPOTHESIS: the one next task is `FOCUSED-C-CAMERA-RESIDUAL-REPLICATION-DIAGNOSTIC`: expand Candidate C coverage using deterministic leave-one-train-camera pseudo-held-out views plus all formal eval views, then retest difficult-view persistence and camera-neighbor structure while controlling depth, tau, accumulation, visibility, and footprint. This remains a focused diagnostic, not a module or training experiment.

The audit stops before module design.
