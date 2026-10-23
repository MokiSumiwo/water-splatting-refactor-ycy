# SARG: Surface-Anchored Gaussian Responsibility

Date: 2026-10-14

## 1. Problem statement

The current renderer has one Gaussian branch for object radiance and a separate, camera-conditioned medium branch. A Gaussian that represents a blue water patch is therefore rendered as an object in both the underwater image and `J_gaussian_raw`; removing water from the latter does not reassign its radiance to the medium. The current SRG prior does not identify this role: it treats `radii > 0` as support, measures depth and scale relative to the current cloud, and penalizes SH-rest energy. Its opacity extension is weak and does not establish that the medium can explain a candidate's image contribution. The surface-path loss constrains medium attenuation and does not classify Gaussians.

The latest 15k experiments confirm that these objectives did not improve the general reconstruction baseline: the mean underwater PSNR over Panama and IUI3 was 31.131 for SRG-appearance, 31.035 for SRG-opacity, and 30.888 for surface-path, versus 31.219 for BND ON. This does not quantify clear-image quality, but it is enough to reject the current proxies as a promoted solution.

## 2. Proposed mechanism

SARG assigns each Gaussian one of four evidence states: `protected_object`, `water_explainable`, `uncertain`, or `unobserved`. Only `water_explainable` Gaussians are eligible for suppression. The state is an evidence-derived controller, not a semantic claim that every unanchored Gaussian is water.

SARG requires two independent keys before suppression:

1. **No reliable surface anchor.** Project COLMAP observations and verified multi-view feature tracks into each Gaussian's 3D covariance. A track is accepted only with mutual feature matching, adequate triangulation angle, bidirectional reprojection/depth agreement, and local patch correlation. Independent source cameras, not repeated training draws, determine support. Reliable positive evidence protects a Gaussian; missing evidence alone only makes it uncertain.
2. **The water explanation passes a counterfactual test.** For the candidate's matched training patches, rerender with that Gaussian (or a small, spatially isolated candidate group) omitted from the object branch while leaving the medium and all other parameters fixed. Compare the full underwater RGB loss in multiple source views. A candidate is water-explainable only when omission does not worsen the robust photometric loss beyond a calibrated tolerance, and improves or preserves it in repeated views. Low texture, blue color, distance, or current view visibility are never sufficient by themselves.

Use image-space footprint integration for the counterfactual rather than a projected-center sample. When dense one-Gaussian-at-a-time tests are too costly, rank candidates using detached raster contributions, then run exact chunked leave-one-out renders on the bounded candidate set. Candidate selection and role-state updates are detached; the formal photometric and medium gradients remain the renderer's normal gradients.

Let `A_g = sigmoid(opacity_logit_g)`. Let `G_g` be calibrated geometric object support and `D_g` be the cross-view loss increase caused by dropping Gaussian `g` (`D_g = L_drop - L_full`). Define a conservative water-explainability confidence:

```text
q_g = 1, only if:
    G_g < tau_geometry
    and median_view(D_g) <= tau_drop
    and enough independent views pass the counterfactual test
    and the candidate is not protected by a held-out training track
q_g = 0 otherwise
```

The first training objective is deliberately small and direct:

```text
L_SARG = lambda_SARG * sum_g(q_g * A_g) / max(1, sum_g(q_g))
L_total = L_existing + L_SARG
```

There is no SH-color shrinkage. The role evidence is refreshed only at fixed refinement boundaries and after topology changes; split/duplicate children inherit the parent's state as `uncertain` until independently revalidated. Suppression is ramped over several refresh windows. A confirmed candidate can be pruned only after its opacity crosses the existing post-refinement threshold and the protection checks remain satisfied. This avoids interpreting a transient low opacity as proof of water.

This first version removes only Gaussian energy that the existing medium and remaining scene can already explain. It does not claim to represent localized suspended particles as a physical volume. If real localized water scattering cannot be reconstructed after suppression, a separate spatial medium-source representation would be needed; changing a Gaussian's label alone cannot provide that missing radiance model.

## 3. Protection and rollback

- Any Gaussian associated with a reliable multi-view track is protected from SARG opacity pressure and SARG pruning.
- A candidate without enough independent views, a reliable track, or a stable counterfactual result remains `uncertain` and receives no SARG penalty.
- Reevaluate role states after densification and opacity resets. Do not transfer a parent's `water_explainable` state blindly to newly split geometry.
- Compare every checkpoint against a paired BND ON run with the same initialization, camera order, seed, corrected backward, BND, M1, OCMC, and CICA settings.
- Stop or roll back if protected-track reprojection coverage, held-out underwater quality, or stable object-edge structure crosses a predeclared tolerance. Reduced clear brightness, opacity, or Gaussian count is not a success criterion by itself.

## 4. Evaluation

There is no clear-water ground truth in the current datasets, so SARG cannot claim true dehazing from clear RGB metrics. Report these separate outcomes:

- **Underwater fidelity:** held-out PSNR, SSIM, LPIPS, and per-view deltas against paired BND ON.
- **Object preservation:** fraction of accepted COLMAP/feature tracks still supported by an object Gaussian, reprojection error, and local patch consistency.
- **Water-candidate reduction:** count and integrated rendered contribution of `water_explainable` Gaussians in distance bins, alongside the same values for protected and uncertain Gaussians.
- **Clear-render stability:** clear coverage, black/near-black fraction, luminance and chroma changes, plus fixed-view and novel-view contact sheets. These are regression indicators, not ground-truth quality metrics.
- **Counterfactual validity:** per-view `L_drop - L_full` distributions before suppression and after the final checkpoint, including held-out training cameras never used to form the role state.

The primary success condition is fewer counterfactually removable far-field Gaussians while preserving accepted object tracks and remaining within the preregistered underwater-quality tolerance. Clear output color shift is secondary and must be reviewed visually.

## 5. Minimal experiment sequence

1. **Frozen audit, no optimizer:** on the BND ON 15k Panama and IUI3 checkpoints, compute Gaussian-to-track association, independent-view support, and exact leave-one-out tests for a bounded far-field candidate set. Produce per-Gaussian records and contact sheets. This establishes whether reliable object anchors and stable counterfactual removability are available at all.
2. **Admission gate:** continue only if candidate groups exist that have weak geometric support, stable non-harmful omission across multiple training views, and no protected-track loss. Lock thresholds and candidate budget from training views only.
3. **Small controlled training:** one scene at a time, paired baseline and SARG from the same fresh COLMAP initialization for 15k updates. Keep all existing modules fixed. Begin with conservative opacity pressure; do not add a new learnable role network, semantic color prior, or extra medium capacity in this comparison.
4. **Promotion decision:** require the joint success criteria above across Panama and IUI3 before considering a four-scene run or integrating the state into the shared renderer.

## 6. Why this is materially different from SRG

SRG asks whether a far Gaussian has low approximate support and then shrinks its appearance coefficients. SARG asks whether independent image geometry fails to anchor it **and** whether removing its actual compositing contribution is safe across views, then suppresses opacity so the Gaussian exits the object branch. This makes the responsibility decision observable, falsifiable, and protective of real distant surfaces.

