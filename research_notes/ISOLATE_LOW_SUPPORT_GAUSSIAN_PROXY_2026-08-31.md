# Isolation Audit of Low-Support Gaussian Failure Proxy (2026-08-31)

## 1. Motivation

HYPOTHESIS: Gaussians constrained by few distinct training views may be associated with heldout reconstruction error after locked OCMC. This audit is diagnostic and does not establish causality.

## 2. Why Camera-Neighborhood Hypothesis Was Rejected

LOCKED RESULT: center-space leave-one-out correlations were negative in all four prior scenes. Camera-neighborhood structure is not reused as the mechanism here.

## 3. Current Low-Support Hypothesis

The candidate signal is representation support across distinct preregistered training cameras, not camera-space proximity.

## 4. Frozen OCMC States

CONFIG FACT: all 5K/8K/10K/13K/14999 C0 checkpoints use bounded_sh3, SH degree 3, classic rasterization, dir_xy_camera, OCMC on, RAOC off, and seed 42. The 3K state is descriptive only. No optimization or backward pass was run.

EXPERIMENTAL FACT: four frozen workers used physical GPUs 6/7/8/9 as logical cuda:0 for Curasao/IUI3-RedSea/JapaneseGradens-RedSea/Panama. All 20 formal checkpoints and four descriptive 3K checkpoints were present and hash-verified.

## 5. Support Definition

CODE FACT: visibility is model.radii > 0; exact equality with gaussian_visible_mask was asserted. s_i is the number of distinct preregistered training cameras in which Gaussian i is visible at that frozen checkpoint. Heldout cameras, duplicate pixels, and future views are excluded.

Fixed thresholds are T0-T3 (s_i <= 0,1,2,3); groups are G0 (s=0), G1 (s=1), G2 (s=2), and G3+ (s>=3).

## 6. Threshold Stability

| Threshold | scenes rho >= 0.4 | median rho |
| --- | ---: | ---: |
| T0 | 3/4 | 0.624 |
| T1 | 4/4 | 0.777 |
| T2 | 4/4 | 0.727 |
| T3 | 4/4 | 0.700 |

QUANTITATIVE RESULT: LOW_SUPPORT_THRESHOLD_ROBUST. Adjacent replicating pairs: ['T0/T1', 'T1/T2', 'T2/T3'].

| Scene | T0 | T1 | T2 | T3 |
| --- | ---: | ---: | ---: | ---: |
| Curasao | 0.541 | 0.406 | 0.638 | 0.638 |
| IUI3-RedSea | 0.707 | 0.821 | 0.821 | 0.700 |
| JapaneseGradens-RedSea | 0.928 | 0.754 | 0.754 | 0.714 |
| Panama | -0.400 | 0.800 | 0.700 | 0.700 |

## 7. Temporal Stability

QUANTITATIVE RESULT: LOW_SUPPORT_TEMPORALLY_STABLE. Comparisons are distributional because split/prune operations invalidate identity continuity.

| Scene | first positive T1 | rho at 5K | rho at 14999 | positive persists |
| --- | ---: | ---: | ---: | --- |
| Curasao | 5000 | 0.516 | 0.406 | yes |
| IUI3-RedSea | 5000 | 0.894 | 0.821 | yes |
| JapaneseGradens-RedSea | 5000 | 0.395 | 0.754 | yes |
| Panama | 5000 | 0.564 | 0.800 | yes |

T1 is positive by 5K in every scene and remains positive through 14999. JapaneseGradens-RedSea first exceeds rho >= 0.4 at 8K; its 5K rho is 0.395. Panama T0 reverses after 5K, but T1-T3 persist.

## 8. Camera-Level Replication

QUANTITATIVE RESULT: final replication covers 22 preregistered heldout cameras. E_cam is heldout MSE; PSNR, SSIM, LPIPS, and MAE are descriptive.

## 9. Gaussian-Level Localization

| Scene | s=0 | s=1 | s=2 | s>=3 |
| --- | ---: | ---: | ---: | ---: |
| Curasao | 0.039% | 6.536% | 5.272% | 88.153% |
| IUI3-RedSea | 0.050% | 11.046% | 15.927% | 72.977% |
| JapaneseGradens-RedSea | 0.043% | 12.769% | 11.736% | 75.451% |
| Panama | 0.042% | 17.930% | 4.913% | 77.116% |

QUANTITATIVE RESULT: low-support high-error enrichment is supported in 3/4 scenes. Approximate support-order monotonicity appears in 3/4.

## 10. Contribution Weighting

CODE FACT: each group is rendered as a standard 3-channel indicator under the original projected geometry, depth order, opacity, and alpha/transmittance compositor. Four group maps sum to formal accumulation under 2e-6 tolerance. The unstable ND path was not used.

QUANTITATIVE RESULT: contribution weighting weakens the cross-scene association: at preregistered T1 the median rho changes from 0.777 to 0.685 (delta -0.092). The strongest descriptive proxies are T1 and CW_T2.

## 11. High-Residual Enrichment

| Scene | top-20% low-support fraction | remaining fraction | enrichment | cameras enriched | scene criterion |
| --- | ---: | ---: | ---: | ---: | --- |
| Curasao | 0.017256 | 0.003299 | 5.230 | 1/6 | no |
| IUI3-RedSea | 0.075441 | 0.022878 | 3.298 | 3/5 | yes |
| JapaneseGradens-RedSea | 0.069830 | 0.012257 | 5.697 | 3/6 | yes |
| Panama | 0.113998 | 0.019465 | 5.857 | 4/5 | yes |

GT is used only to define diagnostic pixel regions; enrichment is not an online training variable. Curasao has aggregate enrichment but only 1/6 cameras are enriched, so it does not pass the registered scene criterion; the other three scenes do.

## 12. Confounders

QUANTITATIVE RESULT: preregistered T1 remains positive after every single-factor camera control in 3/4 scenes. Controls are depth, tau, transmission, accumulation, footprint, opacity, scale, and visible count; no multivariable regression was fit. Passing scenes: Curasao, IUI3-RedSea, JapaneseGradens-RedSea. Panama fails only because its scale control is constant, making the rank residual undefined.

QUANTITATIVE RESULT: fixed-tertile stratified localization survives all four Gaussian factors in 3/4 scenes: IUI3-RedSea, JapaneseGradens-RedSea, Panama. The registered confounder criterion requires each control family to replicate in >=3 scenes; it does not require the same three scenes. A unique physical view angle is unavailable because no registered surface normal exists for an anisotropic Gaussian.

## 13. OCMC Independence

INFERENCE: LOW_SUPPORT_DISTINCT_FROM_OCMC. OCMC controls medium mode capacity while support records scene-Gaussian distinct-view evidence. T1 remains positive after OCMC-residual control in 4/4.

## 14. Online Computability

CODE FACT: the current forward exposes radii > 0. Heldout GT, future views, Jacobians, and an extra render are unnecessary. Existing vis_counts counts optimization observations rather than distinct cameras. Exact deduplication needs camera-identity state such as a per-Gaussian bitset. Such a bitset is GT-free and feasible at LOW update cost, but it is not yet a reliable production statistic because its topology lifecycle is unresolved.

## 15. Topology Lifecycle Considerations

CODE FACT: split/duplicate append state, split parents are culled, and pruning masks state. Locked checkpoints contain no reliable age or lineage. Child inheritance can overstate independent evidence while reset can understate inherited evidence; no production policy is selected.

LIMITATION: Gaussian age/newborn status cannot be controlled reliably from these checkpoints.

## 16. Memory / Runtime Cost

A scalar uint8/uint16/int32 counter costs 0.75-1.20 / 1.50-2.40 / 3.00-4.80 decimal MB for 0.75M-1.2M Gaussians. Exact distinct-camera bitsets are reported separately at actual camera counts. Estimated per-iteration cost is LOW: one visible-mask bit update, no rerender.

## 17. Final Proxy Classification

FINAL DECISION: LOW_SUPPORT_PROXY_SUPPORTED_BUT_NOT_ACTIONABLE.

RESEARCH-LINE DECISION: DEFER_LOW_SUPPORT_MODULE.

INFERENCE: The scientific signal passes, but exact distinct-camera state is not yet reliable across split/duplicate/prune lifecycle.

## 18. ONE Next Task

RESOLVE-LOW-SUPPORT-STATE-LIFECYCLE-PREFLIGHT

No support-aware loss, pruning, refinement, counter, or other module is implemented in this task.

## 19. Disk Cleanup Summary

One reviewed excluded OOM attempt was deleted: outputs/m1_raoc_causal_four_scene_20260827_attempt1_oom (14,073,146,279 bytes; 13,743,756 KiB allocated). No render path was deleted. Every current resplit checkpoint was preserved.
