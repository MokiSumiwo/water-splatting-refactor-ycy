# B: independent center-to-supported-plane experiment

2026-10-23. User requested publishing the current framework and then proceeding with the registered plan. A010 snapshot `ec63f9e` was pushed before this work. This stage implements only B center support, independently on B0; A, normal-thickness loss, free-space pressure, and density redistribution are OFF.

## Frozen protocol

Four scenes, seed 42, raw COLMAP plus random medium initialization, 15,000 updates each (60,000 formal updates). Same BND/M1/OCMC/CICA, corrected_v1 and projection correction as paired B0. CICA starts at 10k. B applies every fourth step; ramp 500–2000, hold to 8000, decay to zero at 10000. Final 5000 steps are native updates. No checkpoint selection or test-driven tuning.

Support comes from COLMAP tracks observed by >=3 non-review training cameras, reprojection error <=1px. Fit each 16-neighbor patch only if all vertices share >=3 eligible cameras; require normal/middle covariance eigenvalue <=.05 and middle/largest >=.15. The patch domain is a disk inscribed in the tangent convex hull. A Gaussian's entire conservative 3-sigma sphere projection must fit that disk; its center must be within 3 uncertainty scales of the plane. Current camera must observe every patch vertex and the Gaussian must be renderer-visible. This is a sparse planar interpolation prior, not measured dense footprint truth or a water label. COLMAP reconstruction itself is inherited from the existing dataset; training-only filtering does not imply retriangulation without all historical cameras.

Normalize signed center distance by max(plane RMS, .05 local 4NN spacing), apply Huber(delta=1), average over eligible samples. Calibrate weight once at first nonzero support to at most 5% of native supported-center gradient norm, then cap each Gaussian's auxiliary gradient at 5% times schedule of its native center gradient. Only means receives the added gradient. No additional RGB backward or densification-buffer changes. Detached support selection, no RNG calls, Adam moments accumulate normally. Actual center displacement, normalized displacement, moments, hook cost, and allocated peak CUDA memory are logged.

Support patch counts Curasao/IUI3/Japanese/Panama: 126/7/77/267. Zero-update baseline 10k footprint eligibility upper bounds: .6753%/.1299%/.5945%/1.4156%, before current-camera gating. This is a low-coverage local test and cannot establish global residual removal. Initial coverage is almost zero because initial primitives are larger than patches; it may rise after ordinary densification. Do not relax filters to manufacture coverage.

Finite-difference Huber gradient and schedule checks passed. Four disposable 20-update smoke runs precede formal training; smoke checkpoints never initialize formal runs. Existing B0 is reused only after unchanged camera sequence/initialization and disabled-path routing checks. Dataset/splits, fixed training review regions, and B0 forward remain frozen. Float endpoint metrics, fixed regions, camera videos and contribution diagnostics use the same exporter as A. U labels remain U; no 20% residual success claim is possible without independent O/F evidence.

C requires verified open-water masks, which current frozen regions do not provide. It must not be launched on U labels. Combinations and extra seeds remain conditional on independent evidence, as specified in the plan.

## Preflight outcome

All four 20-step smokes completed (80 total), with finite reload render and exact checkpoint model hash recovery. The first three scenes had no eligible visible footprint in the short initial window, as expected from the coverage audit. Panama step 8 exercised a nonzero B gradient on two Gaussians (global gradient ratio 0.00002998), with actual Adam step/moment logging. This validates the route but is not evidence of scientific effectiveness. No extra smoke updates were added to force coverage. Pure-function finite differences, per-Gaussian cap and optimizer hook checks passed. All four raw initializations and complete camera sequences match historical B0 exactly.

## Formal monitoring correction

The actual optimizer post-hook displacement files are empty in this environment, including Panama's nonzero-gradient smoke. The earlier wording “actual Adam step/moment logging” described the intended hook, not successful output; it is superseded here. The pure optimizer hook check passed in isolation, but that is insufficient for the real worker. Formal training and finite gradients continue normally; do not infer actual step sizes/moments from the logged gradient norms. Preserve this run without an in-flight instrumentation change or hidden restart, and report missing displacement/moment telemetry as a delivery limitation. The frozen training code and native optimizer route remain unchanged.

Formal four-scene training started on GPUs 6–9 after exact raw initialization and complete camera-sequence equality checks. Supervisor writes `research_runs/surface_support_B_20261023/run_manifest.json`; failed child processes are recorded, and endpoint export runs only after successful training. Checkpoints at 0/10000/15000; same `metrics.py` exporter and fixed B0 assets. Support binary assets can be reconstructed using the committed preparation script and hashed dataset.
