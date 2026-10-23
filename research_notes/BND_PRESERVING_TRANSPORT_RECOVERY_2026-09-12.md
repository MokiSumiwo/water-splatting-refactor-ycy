# BND-Preserving Underwater Transport Recovery

Registered 2026-09-12 before fitting or inspecting candidate test results.

## Objective and Scope

Improve underwater novel-view RGB while retaining the existing BND clear
rendering. Stage 1 freezes all Gaussian geometry, opacity, DC/SH and topology.
The original CRCC E0 BND ON, OCMC ON, CICA-trained, Auxiliary OFF checkpoint
at step 14999 is the scene-specific source. No source files are overwritten.

Only conda water_splatting and physical GPUs 6-9 are permitted. No dependency
installation or commit/push. Existing dirty worktree changes are preserved.

## Registered Stage 1

- A0: frozen original model.
- A1: continue original medium MLP with frozen OCMC projector.
- A2: A1 plus one shared log optical-depth scale, affecting direct and
  backscatter, initialized to zero and projected to log([0.1, 10]).
- A3: A1 plus a shared optical-thickness distribution coefficient eta,
  initialized to zero and projected to [0,1]. Thickness scales are
  1, 0.5, 1.5, with probabilities 1-eta, eta/2, eta/2.

The distribution preserves mean thickness and changes its variance. It is an
effective transport approximation, not an exact multiple-scattering model.
Its RGB result is equivalent to a convex mixture of three complete native
transport calls with shared Gaussian state and ambient radiance.

Each arm receives 5000 medium optimization updates. Seeds 42, 123 and 3407
vary continuation minibatches only; the common source was trained with seed
42. These are not three independent end-to-end 3DGS training seeds.

Use 4096 fixed fitting pixels and 4096 disjoint diagnostic pixels per training
image, selected without test data. Batch size 1024, uniform sampling from
equal-size camera caches. The loss is sampled native RGB MSE. This is a
medium-only refit, not continuation of the original full-image normalized
L1/SSIM objective. All arms use the same loss and minibatches for each seed.

Clone the original medium MLP and its Adam state. Keep its checkpoint learning
rate fixed for the 5000 updates. New scalar Adam uses lr=0.005 and eps=1e-8.
OCMC raw input, neutral input, scale, projector, activation precision and
source encoding are retained. CICA needs no update while colors are frozen.

Exact accepted Gaussian contribution lists are extracted with the existing
verified native-support helper. No top-K truncation, pseudo-depth, background
substitution, exposure change, color modification or densification is used.
The fast fixed-ray compositor is checked against complete native calls on
real inputs. Final metrics and images use the native renderer, including
finite medium, tail and original early termination.

Training diagnostics at steps 0/1000/2500/5000 are recorded. Only step 5000
is a reported treatment endpoint; no checkpoint selection uses test views.
Disjoint training-image pixels are not independent novel-view validation,
since the source Gaussians were trained on those images.

## Gates and Further Stages

All 13 existing test views across all four scenes are evaluated, using the
project-local PSNR/SSIM/LPIPS/MSE implementation and common display mapping.
Sources and original outputs are hashed and preserved. Clear and accumulation
must remain bit-identical in every native comparison.

Proceed to stage 2 only if A3 beats both A1 and A2 by at least 0.03 dB in
equal-scene mean PSNR, its gain over both is positive in each continuation
seed, no scene loses more than 0.05 dB versus A0, mean LPIPS does not worsen
versus A0/A1/A2, and every engineering preservation check passes. These are
screening thresholds, not a statistical significance or SOTA claim.

Stage 2, if triggered, tests constrained color refinement with an explicit
training-view clear-deviation budget; geometry and topology remain fixed.
Stage 3, if needed after a successful stage 2, tests underwater-only BND OFF
teacher distillation after a training-internal teacher-quality check. Neither
stage is run to conceal a failed stage-1 transport hypothesis. If the gate
fails, both are recorded as not triggered, not as failed training experiments.

## Required Artifacts

Save per-seed medium checkpoints, optimizer states, sampling manifests,
learning curves, per-view and scene metrics, source hashes, renderer equality
checks, timings, peak GPU memory and native float images. Produce all-view
underwater and clear contact sheets plus fixed-scale difference maps. No
clear accuracy claim is made without independent clear GT.

Results will be appended after the registered runs finish.

## Completed Results (2026-09-14)

The registered run completed on all four scenes. It used 36 frozen-Gaussian
fits (three arms × three continuation seeds × four scenes), 180,000 optimizer
updates in total, and all 13 existing test views. The native renderer was used
for final metrics and image export.

| Arm | PSNR | SSIM | LPIPS | MSE |
| --- | ---: | ---: | ---: | ---: |
| A0 frozen BND | 29.853773 | 0.927374 | 0.140762 | 0.002118488 |
| A1 medium refit | 29.396507 | 0.924276 | 0.141811 | 0.002392699 |
| A2 optical scale | 29.394938 | 0.924278 | 0.141802 | 0.002391962 |
| A3 mean-preserving thickness mix | 29.397197 | 0.924277 | 0.141803 | 0.002391917 |

A3 was only 0.000690 dB above A1 and 0.002259 dB above A2, far below the
registered 0.03 dB promotion threshold. The gain over A1/A2 was positive for
each continuation seed, but scene safety and LPIPS gates failed. Relative to
A0, A3 changed PSNR by -0.0451 dB on JapaneseGradens, +0.1863 dB on Panama,
-1.9300 dB on Curasao, and -0.0375 dB on IUI3-RedSea. Curasao MTN_1288 was
the dominant failure (approximately -5.19 dB across seeds). Training-image
diagnostic PSNR could improve while this held-out view degraded, showing that
the medium-only refit does not generalize reliably across camera viewpoints.

All clear images, accumulation, depth, and final-transmittance tensors were
bit-identical to A0. The source checkpoint PNGs were also byte-equivalent after
re-rendering. No clear-accuracy claim is made because no independent clear GT
exists. A3 required about 24.95 ms per native evaluation versus 11.54 ms for
A1 and a median incremental peak of 475.5 MiB; its three native transport calls
therefore cost roughly 2.2× the memory and time of a single call.

The promotion gate is **FAIL**. Conditional color refinement and BND-OFF
teacher distillation were not triggered. This result rejects the tested
mean-preserving transport-recovery hypothesis as a stable route to BND-OFF
underwater metrics under the current observations. The historical BND-OFF
reference was re-rendered only for visual context; it is not a matched training
comparator.

Artifacts: [formal report](../outputs/bnd_transport_recovery_20260912_formal/report.md),
[summary JSON](../outputs/bnd_transport_recovery_20260912_formal/summary.json),
[artifact verification](../outputs/bnd_transport_recovery_20260912_formal/artifact_verification.json),
[resource summary](../outputs/bnd_transport_recovery_20260912_formal/resource_summary.csv).
Each scene directory contains underwater, clear, difference, and BND-OFF
reference contact sheets.
