# Frozen FSB/CICA DC Derivative Diagnostic

Registered before inspecting real-checkpoint diagnostic results. This is a
read-only diagnostic, not a training experiment or a new appearance variant.

Use the strict FSB and paired BND ON+CICA step-14999 checkpoints in all four
scenes. For each checkpoint, use its six recorded CICA training-bank cameras.
Do not inspect test images or use test metrics. Compare the existing sigmoid
DC derivative with the actual derivative of the checkpoint's appearance map
within the SAME frozen model. BND is the negative control for the diagnostic.

Let q be native accepted alpha-compositing visibility times RGB transmission
times SH_C0. The existing CICA accumulator in legacy mode returns sum(q*e)
and sum(q*q), without a color derivative. A Gaussian's color derivative d is
constant across pixels in a camera, so its exact DC-feature normal-equation
terms are d*sum(q*e) and d*d*sum(q*q). Reuse this existing kernel; do not
modify the native renderer or install/compile a new diagnostic extension.

For BND, d=c*(1-c). For FSB, with c0=sigmoid(s0), r=full_logit-s0:

```text
r >= 0: d = c0*(1-c0)*[1-tanh(c0*r)+(1-c0)*r*(1-tanh(c0*r)^2)]
r <  0: d = c0*(1-c0)*[1+tanh((1-c0)*r)-c0*r*(1-tanh((1-c0)*r)^2)]
```

Verify the analytic derivative against CUDA SH/color autograd on at most
8192 deterministically sampled visible Gaussians in every bank view. Compare
the factorized old rule against the unmodified CICA accumulator. On bank
ordinals 0, 3, 5, compare corrected normal-equation numerators against native
renderer DC VJPs for both the detached GT-pred residual and a fixed signed
cotangent. Use relative L2 <= 1e-3 for floating-point kernel agreement and
absolute color-derivative error <= 2e-6. Report rather than conceal failures.

Report exact information-weighted Jacobian relative L2 error; derivative
ratio distributions; information mass above 10%/25% derivative error;
changes in per-view DC corrections; target log-chroma, support gates, and
information weights; and CICA DC-gradient norms, cosine and relative changes.
Rebuild targets using the existing controller and loss code with original and
corrected evidence. Also isolate target-only and weight-only substitutions.
These are hypothetical refreshes at 14999, not reconstructions of the saved
run's last target refresh at 14000. Do not recalibrate lambda or update Adam.

Check model hashes, source checkpoint hashes, Python configuration, and the
first bank-view native underwater/clear/accumulation/depth outputs before and
after the diagnostic. Preserve old assets. Save aggregate/per-view records,
deterministic samples and diagnostic plots without full checkpoint copies.
There is no expected image improvement with frozen parameters and no clear
GT; do not claim PSNR or dewatering gains from derivative consistency alone.

Use conda water_splatting and physical GPUs 6-9 only, no dependency install,
commit/push, training, or schedule search. If errors are material, prioritize
an isolated CICA compatibility correction and paired experiment design;
otherwise do not assume this explains the FSB/BND performance difference.
