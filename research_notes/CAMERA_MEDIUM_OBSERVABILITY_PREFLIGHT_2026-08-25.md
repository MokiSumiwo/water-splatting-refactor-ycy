# CAMERA MEDIUM OBSERVABILITY PREFLIGHT

## CODE FACT

The prototype engineering name is `OCMC`: Observability-Controlled Medium Context.

The current M1 camera context is not a learned latent. It is the scene-normalized camera center appended to the medium MLP input.

`OCMC` keeps the existing M1 medium MLP and defines, for the same direction/spatial input:

`z_full = f(direction, spatial, camera_context)`

`z_base = f(direction, spatial, zero_camera_context)`

`Delta_z_cam = z_full - z_base`

When enabled and supplied with a detached 9x9 projector `P_obs` and diagnostic scale `S`, the camera residual is projected in standardized coordinates:

`Delta_z_std = Delta_z_cam / S`

`Delta_z_proj = S * P_obs Delta_z_std`

`z_med = z_base + Delta_z_cam + strength * (Delta_z_proj - Delta_z_cam)`

Default disabled behavior does not evaluate `z_base`, does not apply a projector, and is intended to be baseline-equivalent.

The projector is not persistent checkpoint state. It must be estimated by a diagnostic/training driver from the current scene/checkpoint/batch and installed explicitly with `set_camera_medium_observability_projector`.

## CONFIG FACT

New flags:

`camera_medium_observability_enabled = False`

`camera_medium_observability_strength = 1.0`

Default `False` preserves the current camera-conditioned M1/BND model.

The first preflight projector uses the already validated aggregate structured Jacobian semantics:

`J_struct = d stacked RGB / d standardized shared 9-D raw medium-output perturbation`

The pre-registered scale-free gate is:

`g_i = sigma_i^2 / (sigma_i^2 + median(sigma)^2)`

`P_obs = V diag(g_i) V^T`

`V`, `sigma`, and `P_obs` are detached. No gradient flows through Jacobian construction, SVD/eigendecomposition, or projector estimation.

## HYPOTHESIS

Camera-conditioned medium residuals contain both useful supported components and low-observability components. Projecting only the residual, rather than globally penalizing beta_D variance, should preserve useful camera-conditioned variation while suppressing empirically weak contextual freedom.

## GRADIENT DESTINATION

There is no new RGB target, clean-image supervision, depth supervision, or auxiliary scalar training loss in this prototype. The projection changes only the medium raw field before rasterization.

Direct mechanism metrics based on `Delta_z_cam` should backpropagate only into the medium branch (`direction_encoding`, `medium_mlp`). Gaussian/object parameters should receive no gradient from those direct mechanism metrics.

Normal RGB reconstruction gradients still follow the existing renderer coupling and are not interpreted as new mechanism-only Gaussian gradients.

## WHY THIS DIFFERS FROM FAILED MIC

MIC penalized beta_D raw within-camera variance and collapsed the wrong degree of freedom while increasing across-camera beta_D variance.

`OCMC` does not add a beta_D variance penalty, does not suppress all beta_D variation, and does not hard-code beta_D red or any IUI3-specific weak vector. It projects only the camera-induced residual using a detached observability basis estimated from image-formation sensitivity.

## PREFLIGHT REQUIREMENTS

The prototype is `READY` only if:

disabled-path equivalence passes;

enabled neutral initialization is controlled;

the detached projector is finite and nontrivial;

direct mechanism gradients are medium-local;

short smoke optimization is finite;

the camera residual is measurable and not trivially collapsed;

checkpoint save/load remains compatible.

No 15k training claim is made in this preflight.

## QUANTITATIVE RESULT

Phase-A gate was satisfied: `CAMERA_CONDITIONED_IDENTIFIABILITY_TRADEOFF_SUPPORTED`.

Preflight output directory: `outputs/camera_medium_observability_preflight_iui3_20260825`.

Checkpoint used: C1 step `14999`.

GPU: physical `6`, PyTorch logical `0`, `NVIDIA GeForce RTX 3080`.

Disabled-path equivalence: `true`.

Enabled with no projector equivalence: `true`.

Detached projector: finite `true`, trace `4.1189775466918945`, Frobenius norm `1.7889589071273804`, min gate `5.048208186053671e-05`, max gate `0.9472894668579102`, sigma reference `0.018684396520256996`.

Direct mechanism metric gradient locality: Gaussian gradient sum `0`; medium-local check `true`.

Smoke training: `20` optimizer steps, all losses finite `true`.

Checkpoint compatibility: save/load pass `true`; projector is non-persistent state.

Residual collapse check: residual not trivially collapsed `true`.

## INFERENCE

Phase-B classification: `CAMERA_OBSERVABILITY_MODULE_READY`.

This only establishes engineering readiness for one OCMC prototype. It does not establish PSNR benefit or decomposition improvement for OCMC, because the requested Phase-B scope forbids a formal 15k training run in this task.
