# Transmission-Gated Scene-Calibrated Gain

This is a follow-up diagnostic after the frozen direct-light gain run. It uses
the same CRCC E0 BND-ON checkpoints, cameras and native renderer. All Gaussian
parameters, medium parameters, clear rendering and topology remain frozen.

The adapter is the previously tested scalar or ten-term spatial quadratic gain,
initialized at identity and bounded to `exp(±0.35)`. During fitting, a detached
pixel gate requires accepted-ray accumulation >= 0.5, mean BND transmission in
[0.35, 0.85], and direct RGB support >= 1e-3. Pixels outside the gate retain
the exact BND output at inference. The gate is a support criterion, not a
semantic object mask or physical visibility estimate.

Use the same sorted-camera split (rank modulo 5 == 2 validation), 8192 sampled
pixels on fitting cameras and 16384 on validation cameras, three continuation
seeds (42, 123, 3407), Adam lr .01 and 2500 updates. Validation metrics cover
all sampled validation pixels so the gate cannot hide regressions. Identity is
an explicit candidate. Select each scene's architecture/step from validation;
official test views are evaluated only after selection.

Promotion requires macro validation gain >=0.03 dB, at least three scenes gain
>=0.01 dB, no scene loss >0.05 dB, positive macro gain for every seed, no LPIPS
worsening, and exact native/clear/source preservation. If the gate fails,
teacher, geometry, color, exposure and densification are not attempted.

The experiment is an effective capacity and support diagnostic. It cannot
establish physical intrinsic radiance recovery, and no clear-image accuracy
claim is made without independent clear ground truth.
