# LSPR / MST operational definitions and direct training, 2026-09-17

The repository and available conversation state did not retain full definitions of
these two acronyms. The user was asked for clarification while CDRC proceeded.
To follow the direct-implementation request, the following explicit working
hypotheses were announced. These are **this experiment's definitions**, not a
claim that the original missing descriptions have been recovered.

No diagnostic admission gate. No expansion of color bounds or test-view tuning.
Both independent candidates restore the same R0/C0a step-14999 model, optimizer,
scheduler, runtime, RNG and OCMC state as BASE_A/CDRC/BASE_B. Same 2000 updates,
camera sequence and CICA settings as FAR_FIELD_DIRECT_TRAINING_PROTOCOL_2026-09-17.
They do not stack on CDRC. Four scenes on physical GPUs 6–9 only.

## LSPR: Low-Signal Surface Prior Recovery

This is a prior for regions with weak native color sensitivity. Use the previously
saved per-Gaussian information ratio relative to zero medium at the same checkpoint.
Strong anchors require >=3 contributing views, weight>=.1, min-channel information
ratio>=.35. Weak candidates require >=2 views and min-channel ratio<.2. No external
teacher or clear GT is introduced.

For each weak Gaussian find 8 spatially nearest strong anchors. Require >=4 within
6 times local Gaussian scale, local PCA smallest/total eigenvalue<.1, and distance
to the neighbor plane <=2 local scales. Targets are frozen inverse-distance and
information-weighted DC log-chroma. No brightness pseudo-label or exposure change.
Strength decreases with information ratio. Optimize DC log-chroma Huber at most
32768 valid Gaussians per auxiliary step; the same .05 loss weight, every-4-update
schedule and .1 gradient-norm cap as CDRC. Current positions are rematched to source
identities after pruning, with a scale-based displacement check. Geometry and
medium receive only their original training gradients.

The nearest anchor graph is not proof of a shared material. It may oversmooth real
color boundaries or propagate biased anchor colors. This is a testable prior and
adds no independent observation. Its support and actual nonzero updates are logged.

## MST: Medium Scattering Transport

Jointly train positive, depth-dependent lateral scattering of visible direct light.
Reuse the original native Gaussian ordering/alpha/transmission, partition Gaussian
camera-z by the existing train-derived scene depth boundaries, and render near/mid
direct color layers. The far layer is the native direct output minus the first two;
linearity in color preserves the original support and makes their sum exact.

For each layer apply a convex mixture of separable Gaussian PSFs (sigma 1 and 3
native pixels, reflective image boundary). Blend scattered and direct images with
`q_layer,c=.3*(1-exp(-softplus(tau_c)*relative_layer_depth))`. Three channel optical
strengths and two mixture logits are learned: 5 parameters. Initialize optical
strength .01 (small but nonzero, **not exact identity**), optimize with Adam .01.
The ordinary underwater loss jointly updates all original parameters and the new
five parameters. No auxiliary clear loss. Store the transport state and optimizer
beside each endpoint model; reproducing MST underwater images requires both files.

Clear forward rendering of any fixed Gaussian state is unchanged; joint training
can change that state, so final clear images are explicitly compared. CICA target
refresh uses the native original renderer in every arm, not the PSF-modified
outputs; this preserves the existing controller but is not an exact Jacobian of
MST. A positive diagnostic result would require studying this approximation.

This is visible-layer convolution, not full multiple scattering or transport of
occluded surfaces. Constant layer representative depths approximate depth variation.
A prior frozen two-parameter PSF experiment exists (`run_forward_scattering_screen.py`);
this experiment differs by jointly training the current FSB scene and wavelength
strength. That difference does not itself establish novelty.

## Reporting

All 13 original test views: floating-point full-image PSNR/SSIM/LPIPS; equal scene
means; each candidate minus mean of continued BASE_A/B, and BASE_A/B repeat gap.
Fixed initial near/mid/far support masks; clear raw [0,1] images, native crops and
fixed x10 differences without recoloring. Initial step14999 is displayed separately.
Clear proxies do not establish restoration accuracy. No SOTA claim is warranted
solely from this short continuation experiment, and all endpoints are retained.
