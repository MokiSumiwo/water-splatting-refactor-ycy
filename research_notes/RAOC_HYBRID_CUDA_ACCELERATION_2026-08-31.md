# RAOC Hybrid CUDA Acceleration Validation

Date: 2026-08-31

## Scope

This was the final engineering feasibility attempt for RAOC. The new
`camera_medium_raoc_backend='cuda_hybrid'` backend computes only the nine
renderer-local directional sensitivities `||J_p v_i||_2` in CUDA. The modal
projection, evidence, local and global gates, modal reconstruction,
unstandardization, and differentiable medium residual path remain in the
reference PyTorch implementation. The existing `reference` and
`cuda_fused` backends remain available; `reference` remains the default.

No RAOC equation, calibration state, quantile definition, medium model, or
formal 15K experiment was changed or run.

## Previous error localization

The archived full-fused validation identified `Delta_z_raoc_std` as the
dominant discrepancy: approximately `2.075195e-3` for Q50 and
`2.074718e-3` for Q80, while sensitivity itself differed by only approximately
`2.7567e-7`. This made H1 a direct test of whether moving only sensitivity to
CUDA would remove the error amplification.

## Operator result

IUI3 was evaluated from the archived C1 checkpoint at step 3000 with the
same geometry, calibration state, camera samples, and external Q50/Q80
states. The registered H1 gate was evaluated before any performance or
training phase.

- Q50: sensitivity max `1.86264515e-07`, g_keep max `9.91225243e-05`, Delta_z max `0.00207519531`, pred max `0.00043284893`, medium-gradient relative L2 `1.03489309e-05`, cosine `1.00000012`.
- Q80: sensitivity max `1.86264515e-07`, g_keep max `6.85453415e-06`, Delta_z max `0.0020699501`, pred max `0.000368177891`, medium-gradient relative L2 `0`, cosine `1.00000012`.

The sensitivity output is close to reference, and the prediction and
gradient checks are within their registered limits. However, the Q50 gate
and Delta_z limits fail, and Q80 also fails the Delta_z limit. The
sensitivity-only CUDA boundary therefore does not remove the dominant
reference-vs-fused reconstruction discrepancy. The error distribution,
including mean, median, p95, p99, p99.9, and max, is in
`operator_error_distribution.json`.

## Early stop

The H1 operator gate failed. In accordance with the frozen protocol,
synchronized performance, IUI3 fixed-topology 500-step training,
Panama replication, four-scene normal-topology training, and memory A/B
were not run. No threshold was relaxed, no additional CUDA fusion boundary
was attempted, and no Q50/Q80 formal 15K experiment was launched.

## Decision

Engineering classification: `RAOC_HYBRID_ACCELERATION_NOT_SUPPORTED`.

Research-line decision: `CLOSE_RAOC_AND_LOCK_OCMC`.

This closes the RAOC acceleration line, not the OCMC mechanism. The formal
research direction should remain on the previously validated OCMC
camera-conditioned medium capacity-control mechanism. RAOC results may be
retained as mechanistic evidence, but RAOC should not be used as the formal
accelerated backend on the basis of this attempt.
