# Identifiability Gain Mechanism Attribution Audit

Date: 2026-09-02
Classification: `GENERIC_EFFECT_MORE_LIKELY`

## Objective And Boundary

This read-only audit asks whether the completed C1 RGB gain comes from the claimed SH-opacity identifiability correction or from a generic regularization/optimization side effect. It uses only the completed matched C0/C1 checkpoints and saved trajectories. OCMC stays on, RAOC stays off, and no model training, backward pass, optimizer step, checkpoint write, render write, strength sweep, or module change is performed.

## Evidence Levels

Arm-level RGB and saved distribution metrics are causal because C0/C1 used matched starts, cameras, and updates. Ambiguity-conditioned RGB localization freezes C0 training-only ambiguity, sample, and projected heldout boxes before heldout GT, but overlapping projected boxes are associative and not additive per-Gaussian render decomposition. Parameter differences after topology divergence use reciprocal nearest geometry only; they are explicitly proxy evidence because persistent split/prune lineage was not saved.

## Scene Results

| Scene | dPSNR | shared rel | overlap d | nonDC | orth | mutual/strict match | high-low RGB MSE improvement | rho(A, improvement) | SH shrink |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | -0.045276 | -0.983627 | -0.000690 | 0.779 | 0.869 | 0.514/0.150 | +2.934e-05 | +0.078 | 0.864 |
| IUI3-RedSea | +0.090387 | -0.998155 | +0.000240 | 1.027 | 1.274 | 0.469/0.105 | +1.810e-04 | +0.011 | 0.767 |
| JapaneseGradens-RedSea | +0.027695 | -0.974517 | -0.000109 | 0.670 | 0.736 | 0.465/0.133 | -1.331e-04 | -0.040 | 0.799 |
| Panama | +0.072294 | -0.958631 | +0.000161 | 1.160 | 0.941 | 0.473/0.129 | -3.327e-05 | +0.013 | 0.864 |

## Attribution Result

The causal RGB effect remains supported. Ambiguity/improvement rank direction is positive in 3/4 scenes but significant in 0/4; high-ambiguity regions improve more than low-ambiguity regions in only 2/4. Shared SH-opacity response energy decreases in 4/4 scenes, while formal tangent overlap decreases in only 2/4 and orthogonal capacity is preserved in 3/4.

No scene passes the fixed Gaussian identity-quality gate. Non-DC SH energy falls by more than 5% in 4/4 scenes, and arm-local view-dependent SH response RMS falls below 75% of C0 in 4/4. Final medium MLP states also drift by 15%-27% relative L2 despite identical OCMC projector bundles. This systematic capacity suppression together with absent selective localization makes generic regularization/trajectory effects more likely than the claimed identifiability mechanism. Level C does not identify one unique generic cause; it rejects H1 as the reasonable attribution for the gain. The defensible result is `GENERIC_EFFECT_MORE_LIKELY`.

## Cross-Scene Interpretation

Across the four scenes, Spearman rho between heldout dPSNR and final median C0 ambiguity is +0.800; rho with high-minus-low localized improvement is +0.400; rho with non-DC energy ratio is -0.400. With n=4 these are descriptive only. Curasao's negative dPSNR cannot be uniquely explained by ambiguity, Gaussian count, view coverage, tangent overlap, or SH energy from this sample; no post-hoc scene-specific mechanism is selected.

## Required Answers

1. **yes at direct-gradient and population-distribution level, not at Gaussian identity level** features_rest has the largest standardized population shift in 4/4 scenes, and it is the only direct module-gradient target; however split/prune lineage was not retained.
2. **partially** shared response energy decreases in 4/4, while formal tangent overlap decreases in only 2/4 scenes.
3. **not consistently** registered orthogonal/non-DC capacity gate passes in 3/4 scenes.
4. **not identifiable from these checkpoints** the high-ambiguity nearest-match action proxy is interpretable in 0/4 scenes; reliable lineage quality passes in 0/4.
5. **not consistently** frozen-C0 high-ambiguity regions improve more than low ambiguity in 2/4 scenes.
6. **yes, generic SH regularization is more likely than the claimed selective mechanism** non-DC energy falls by more than 5% in 4/4 scenes, view-dependent response RMS is below 75% of C0 in 4/4, while tangent overlap is directionally favorable in only 2/4.
7. **yes, optimization trajectory is a credible co-explanation** the direct gradient is routed only to features_rest, but saved total gradients, Adam states, base-loss trajectories, topology, and non-target parameters diverge downstream.
8. **not reliably** nearest-match action/RGB Spearman direction is positive in 3/4 scenes, but no scene passes the pre-fixed identity-quality gate.
9. **no** heldout PSNR improves in 3/4 scenes and Curasao is negative.
10. **no** ambiguity-conditioned RGB localization is directionally stable in 2/4 scenes.
11. **only at intervention and projector-state level, not final learned-state level** OCMC projector bundles are equal and direct module gradients exclude medium/OCMC, but medium_mlp follows the changed optimization trajectory (final exact relative L2 drift 0.155-0.271).
12. **no** the +0.036275 dB mean causal gain is real, but the ambiguity -> selective correction -> representation change -> novel-view improvement chain does not close.
13. **selective ambiguity-conditioned correction and causal RGB localization** persistent lineage is absent, tangent overlap is inconsistent, and frozen projected boxes are associative rather than additive per-Gaussian render attribution.
14. **no, not as an identifiability module** classification is GENERIC_EFFECT_MORE_LIKELY; archive the intervention's small positive RGB effect, but reject and do not tune or present the identifiability attribution.
15. **no** the task explicitly forbids a third module; resolve attribution before new mechanism search.

## Minimum Next Diagnostic

Do not tune or retain this intervention as an identifiability module, and do not search for a third module. Archive the small RGB effect as generic/unattributed. If H1 must be revisited despite this classification, the minimum decisive diagnostic is one exact protocol replication that records immutable Gaussian parent/descendant IDs through every split, duplicate, and prune, then reports ambiguity-stratified tangent/orthogonal updates and additive heldout contribution deltas for those lineages. Without lineage, further nearest-neighbor checkpoint analysis cannot recover the missing causal link.

## Integrity

The audit read 40 hashed causal checkpoints and produced 6144 frozen-region rows. All workers report zero backward, JVP, VJP, optimizer, training, checkpoint-write, and render-write calls. OCMC projector equality and protected-source hashes pass in every scene.
