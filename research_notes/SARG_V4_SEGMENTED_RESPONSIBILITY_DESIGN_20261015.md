# SARG-v4: segmented transport and Gaussian responsibility residual

## Motivation

SARG-v3 validates a candidate mostly with a whole-ROI replacement ratio. A
mixed Gaussian can therefore pass when a near-surface contribution hides a
far-field responsibility error. SARG-v4 evaluates the replacement along the
optical path.

## Mechanism

For each candidate group and each independent training view, the verified ROI
is divided into four depth quantiles. For segment `s`, the audit records:

* removed object contribution: `(rgb_object_full - rgb_object_drop)+`;
* medium replacement contribution: `(rgb_medium_drop - rgb_medium_full)+`;
* underwater reconstruction residual after omission;
* clear-signal change and mean depth.

The segment responsibility is

`r_s = clamp(medium_replacement_s / removed_object_s, 0, 1)`.

The group score is a pixel-weighted average of `r_s`, with the farthest
segment weighted twice. A candidate must still pass the existing cross-view
photometric and joint compositing tests, and its median far-segment
responsibility must exceed `0.08`. Surface anchors remain protected.

## Training effect

The validated binary water role is retained for topology handling, but the
opacity regularizer is scaled by `0.5 + 0.5*r_group`. This preserves a weak
pressure for mixed candidates while applying full pressure only to candidates
whose far-field contribution is explained by the medium. The detached depth
field additionally provides the existing distance gain.

No blue/green threshold is used as a label, and the segmented audit is run
under the frozen medium and isolated RNG context. It does not change native
forward/backward or the renderer.

## Parameters

`transport_segments=4`, `segment_far_weight=2.0`,
`minimum_segment_pixels=32`, `minimum_segment_responsibility=0.08`.

These are deliberately conservative starting values. Formal Panama and IUI3
15k runs should compare both endpoint metrics and the far-segment residual
distribution before increasing the suppression strength.
