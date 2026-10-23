# SBR BND-ON protocol (2026-09-14)

Selective Bounded Relaxation (SBR) is evaluated as a single-factor continuation
from the registered OCMC C0 checkpoint at step 3000. CICA, M1, OCMC, camera
sequence, topology callbacks, seed and optimizer settings are unchanged;
Auxiliary appearance is disabled. The only changed operation is the Gaussian
appearance map: ordinary bounded sigmoid colors are smoothly mixed with the
existing headroom map when the per-channel sigmoid derivative is below 0.055.

The gate is differentiable and keeps RGB in [0,1]. It does not alter the
underwater renderer, medium field, geometry, opacity, background, or clear
recomposition semantics. A 5k smoke continuation was completed during
implementation; the strict four-scene 15k run is recorded under
`outputs/sbr_bnd_on_four_scene_20260914_v4/` and is only promoted after all
four final summaries and validation metrics are present.

No clear ground truth is available. Clear images are therefore diagnostic only.
