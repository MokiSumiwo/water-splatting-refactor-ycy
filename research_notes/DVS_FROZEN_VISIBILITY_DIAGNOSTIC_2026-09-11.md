# DVS Frozen Visibility Diagnostic

Date: 2026-09-11. Status: complete; reject the proposed attenuation-culling
explanation for the audited renderer and checkpoints. No training was started.
DVS is only the candidate's working name, not a validated method.

## Finding

All 26 frozen test-view comparisons across four scenes and eight final
checkpoints produced bit-identical original and zero-medium clear RGB and
accumulation. Depth, depth moments, first/last depth and final transmittance
also compare exactly. No black pixels recovered, no coverage appeared, and
no additional colors or artifacts were exposed. Original underwater output
was retained and reproduced exactly, including historical PNG pixels.

The proposed causal premise omitted a decisive line in the renderer:

```cpp
min_medium_attn_pix = std::min(beta.x, std::min(beta.y, beta.z));
min_medium_attn_pix = std::min(0.f, min_medium_attn_pix);
if (sigma < 0.f || alpha * __expf(-min_medium_attn_pix * depth) < 1.f / 255.f)
    continue;
```

`medium_attn` uses Softplus, and every actual captured input was finite and
nonnegative. Thus `min(0, min(beta)) = 0`, `exp(-0 * depth) = 1`, and the
contribution cutoff is simply `alpha < 1/255`. Positive medium attenuation
does not remove clear contributions. Any earlier reading that inferred
attenuation-based rejection from the conditional alone is corrected here.

This is a negative result about a specific discrete support mechanism. It
does not establish the true cause of all dark regions or show that current
Gaussian geometry, colors, or underwater decomposition are accurate.

## Repository And Frozen Scope

Root confirmed using `git rev-parse --show-toplevel`:
`/mnt/new/home_old/ycy/water-splatting-refactor`.
Branch: `research/bvam-bnd-on`; HEAD: `3a0df36`, with pre-existing local changes.
Those changes were retained. Core code and the installed CUDA binary were
hashed before and after the successful run; all 38 registered files match.

Only conda `water_splatting` was used, with Python 3.8.20 and Torch
2.1.2+cu118. Rendering used physical GPU 6/7/8/9, one scene per GPU, with
logical `cuda:0` inside each worker. Runtime/device details are in each
`audit.json`. No dependency installation, CUDA rebuild, optimizer step,
backward call, point insertion, pruning, color change, exposure adjustment,
commit or push occurred.

The checkpoints are the recent CRCC experiment's final step 14999: E0 is
the paired BND baseline; E2 is CRCC with TABA off. E0/E2 are compared with
their own frozen zero-medium replay, not with each other as a DVS treatment.

| Scene | Physical GPU | Native W x H | Every existing test view in these runs | Checkpoints |
| --- | ---: | --- | --- | --- |
| JapaneseGradens-RedSea | 6 | 1384 x 918 | MTN_1090, MTN_1098, MTN_1106 | E0, E2 |
| Panama | 7 | 1795 x 1188 | MTN_1539, MTN_1529, MTN_1547 | E0, E2 |
| Curasao | 8 | 1794 x 1188 | MTN_1288, MTN_1296, MTN_1304 | E0, E2 |
| IUI3-RedSea | 9 | 1383 x 917 | MTN_5903, MTN_5894, MTN_5911, MTN_5928 | E0, E2 |

There are 13 unique test cameras and 26 checkpoint/view comparisons. The
camera sets were checked against each source run's final per-view metrics.
This scope covers all formal test views of these selected checkpoints; it
is not a replay of every historical checkpoint or alternative data split.

Let `C = outputs/crcc_bnd_on_20260911`. Source checkpoint paths are:

- Default: `C/<scene>/<E0|E2>/checkpoints/step-000014999.ckpt`.
- Curasao/E2 and IUI3/E2: `C/crcc_only_four_scene_extension/<scene>/E2/checkpoints/step-000014999.ckpt`.
- Native original images: `<source>/renders/<arm>/<scene>/step_14999/<view>/A2/`.

Full checkpoint, camera, geometry/color, sorting, original output and file
hashes are retained in the per-checkpoint audits and per-view diagnostics.

## Renderer Path Audit

Line references describe the unchanged source at this audit.

| Path | Filtering and termination semantics | Use in this diagnosis |
| --- | --- | --- |
| Model preprocessing, `water_splatting.py:2391,2439,2570` | Optional evaluation crop; camera projection; sigmoid opacity, optionally multiplied by antialias compensation | Exact final inputs captured and reused; no support or opacity change |
| CUDA projection, `forward.cu:44,76,85`; `helpers.cuh:43,229` | Near-plane clip (config default 0.01), zero determinant rejection, 3-sigma projected tile bounds, zero tile-area rejection | Unchanged; medium is not an input to projection |
| Binning/sorting, `utils.py:179` | Sort tile/depth intersection keys front to back; no medium input | Original and replay sorted keys, Gaussian IDs and tile bins hash identically |
| Final model raster call, `water_splatting.py:2758` | `UnderwaterRasterizer.rasterize` to `rasterize_gaussians`; N x 3 colors select `_C.rasterize_forward` | Actual path for all 26 views |
| RGB CUDA, `forward.cu:390,439,441,446` | Effective attenuation zero; reject sigma < 0 or alpha < 1/255; alpha capped at 0.999; stop at next_T <= 1e-4 | Same discrete support for original and all zero-medium controls |
| Tile termination, `forward.cu:408` | Stop loading batches when all threads in tile are done; outside-image threads start done | Unchanged |
| TABA depth prepass, `water_splatting.py:2669` | Same RGB rasterizer, detached inputs | Off in these E0/E2 arms; final-call interception avoids confusing a prepass with the output |
| Model empty crop/zero radii, `water_splatting.py:2391,2458` | Return medium RGB, zero clear and zero accumulation without rasterizing | Not encountered in any audited view |
| Python whole-frame no-intersections branch, `rasterize.py:151` | Object image becomes supplied background, clear zero, final_Ts zero (hence returned accumulation one) | Legacy edge case differs from ordinary empty pixels; not encountered, not modified |
| ND CUDA, `forward.cu:175,270,275`; `rasterize.py:187,220` | Alpha-only cutoff; same stop threshold; half accumulation and different medium/background composition; wrapper output contract differs/stubs stats | Unused; not an equivalent replacement for the RGB underwater renderer |
| Torch reference, `_torch_impl.py:401` | Alpha cutoff 1/255, stop 1e-4, residual T times background; no matching underwater composition | Unused; not an automatic fallback for this run |
| RGB backward, `backward.cu:222,281` | Same min-with-zero attenuation clamp and cutoff | Source audit only; no backward executed |

The early-stop test occurs BEFORE accumulation and before updating T. The
Gaussian that would cross `next_T <= 1e-4` is excluded. Although the nearby
comment suggests rendering the last contributor, the executed code omits
this threshold-crossing Gaussian. Both compared calls retain that behavior.
Changing this common alpha rule would be a different experiment.

The active RGB kernel uses `vis = alpha * T` and accumulates
`J = sum(vis * color)` separately from attenuated object RGB. It does not
add `T * background` to clear RGB. Uncovered pixels therefore remain black,
regardless of the supplied black/white background. Underwater medium RGB
includes finite intervals and a residual tail. The model can subsequently
recompose its tied B_inf tail; the saved original `pred_image` is taken
after the original model path and is never replaced by the replay.

`rasterize.py` defaults to a white-background override. This is not evidence
that clear uses a white residual background: the active RGB kernel ignores
that argument for clear. Explicit black/white controls with the override
disabled confirmed equality on every view.

`UnderwaterRasterizer` returns raw `j_raw`, clamp01 display, and a distinct
`rgb_clear = J/(1+J)` display. Comparisons use raw float32 J first, with both
display versions saved separately. They do not compare a tone-mapped image
against an untransformed one.

Finally, `rasterize.py:270` accepts but does not forward the clear/depth/moment
output gradients into CUDA, and returns no depth-input gradient. A future
training proposal cannot assume this independent forward control supplies
the desired backward semantics. No such training proposal is justified here.

## History And Related Methods

- Commit `8284f18` (2024-08-16, original code) already includes the
  min-with-zero line and the separate clear accumulation. This behavior is
  not newly introduced by CRCC or the recent BND experiments.
- Commit `36f5181` (2026-07-27, "Add proxy clear chroma diagnostics and
  experiments") adds `UnderwaterRasterizer.rasterize_clear_proxy`: same
  projected Gaussians/colors/opacities, zero RGB/bs/attn, black background,
  `force_white_background=False`. The independent zero-medium mechanism
  therefore already exists in repository history.
- Commit `62294d6` (2026-08-08) removes that helper in the clean M1 branch;
  see [clean branch audit](M1_BND_CLEAN_BRANCH_AUDIT_2026-08-08.md). Removal
  of the convenience wrapper does not imply loss of clear visibility.
- [Low-support Gaussian proxy audit](ISOLATE_LOW_SUPPORT_GAUSSIAN_PROXY_2026-08-31.md),
  section 10, uses standard three-channel indicator rendering with zero
  medium. Four contribution groups reconstruct formal accumulation within
  2e-6. The implementation is
  `scripts/diagnostics/isolate_low_support_gaussian_proxy.py:208`.
- `rendering/medium_jacobian.py:114` already mirrors the min-with-zero rule.
  [SeaFree/WaterSplatting compositing audit](SEAFREE_WS_DEGRADATION_COMPOSITING_AUDIT_2026-08-10.md)
  also documents separate intrinsic/degraded compositing and why alternate
  renderers require semantic checks.
- [WaterSplatting, arXiv:2408.08206](https://arxiv.org/abs/2408.08206)
  already describes restoration by removing the learned medium.
  [SeaThru-NeRF, arXiv:2304.07743](https://arxiv.org/abs/2304.07743) and its
  [project page](https://sea-thru-nerf.github.io/) likewise describe medium
  removal for clear views. These pages were checked; this is a targeted
  related-method check, not an exhaustive literature novelty review.

The user's MV-GAR/MCGR results motivate avoiding another geometry constraint;
their quoted Windows-checkout paths are not used as evidence about this
Linux branch's current renderer. The local source and prior zero-medium
experiments are sufficient to show overlap and reject the present premise.

## Frozen Comparison Protocol

1. Load each existing snapshot in eval mode under `torch.no_grad()`.
   Compare loaded model hash to snapshot, retaining all model/EMA state.
2. Run the complete original model call. Capture its final rasterizer
   arguments after current-camera appearance and medium computation.
3. Replay those exact projected positions, depths, radii, conics, tile
   counts, RGB colors, opacities, dimensions and step. Independently set
   `medium_rgb`, `medium_bs`, `medium_attn` to zero only in the replay.
4. Keep the original underwater result. Check repeat-original equality for
   every raster output and full-model underwater output, including source
   PNG equality. Hash-check sorting and unchanged geometry/colors.
5. Run zero-attenuation-only, zero-RGB/bs-only and explicit black/white
   background controls. Every clear/support/depth comparison is exact.
6. Save native float32 arrays, raw/tone-mapped clear pairs, coverage pairs,
   absolute differences at x20 and recovered-black masks. Preserve original
   underwater/GT through symlinks and hash-check original files.
7. Time five warmed repeats per view/variant using synchronization and CUDA
   events. Check model, EMA, optimizer and source asset hashes afterwards.

The full-model `zero_medium` configuration is not toggled. It is upstream
of optional appearance logic and would not isolate the final rasterizer's
contribution selection. Colors in this control are the exact N x 3 tensors
already supplied to the original rendering call.

Synthetic actual-CUDA checks passed on all four assigned GPUs: beta=100
suppresses underwater object signal to zero while clear remains nonzero
and unchanged; alpha=0.001 is excluded; two center opacities 0.999 show that
the threshold-crossing second Gaussian is excluded. These tests check the
installed binary as well as the source-level argument.

## Quantitative Results

All 26 views have raw maximum and mean clear difference 0, changed-pixel
fraction 0, maximum accumulation difference 0, new coverage 0, recovered
black fraction 0 and newly black fraction 0. Clear/depth controls also pass
for every view. Native arrays were independently reloaded and compared by
their uint32 bit representations; this is stronger than PNG equality.

Near-black means `max(J_R,J_G,J_B) <= 3/255` on raw float32 J. All fractions
below are percentages of the full image, averaged equally over test views.
The old CRCC report used an 8-bit display proxy, so its fractions can differ
slightly. Each number below is unchanged in the zero-medium replay.

| Scene | Arm | Near-black % | A = 0 % | Near-black with A >= 0.1 % | Recovered black % |
| --- | --- | ---: | ---: | ---: | ---: |
| JapaneseGradens | E0 | 4.296877 | 1.656130 | 0 | 0 |
| JapaneseGradens | E2 | 2.384708 | 1.178081 | 0.000367 | 0 |
| Panama | E0 | 0.001251 | 0 | 0 | 0 |
| Panama | E2 | 0.005768 | 0 | 0 | 0 |
| Curasao | E0 | 9.160445 | 7.003032 | 0.000016 | 0 |
| Curasao | E2 | 7.687772 | 6.092324 | 0 | 0 |
| IUI3 | E0 | 7.965669 | 2.082954 | 2.609976 | 0 |
| IUI3 | E2 | 13.404452 | 7.092274 | 1.305993 | 0 |

Most near-black pixels outside IUI3 occur with A < 0.1. IUI3 additionally
has measurable dark pixels with A >= 0.1. These are descriptive partitions,
not object masks: A=0 can reflect no projected support, the common alpha
cutoff or other unchanged screening; it does not prove missing geometry.
Dark covered pixels do not by themselves establish erroneous learned color.

Range values below are true extrema across all pixels/channels and views
of each arm, not means of per-view extrema. Original and replay ranges are
identical. No raw clear values are negative, above one, NaN or infinite.

| Scene | Arm | Raw clear J range | Fixed Gaussian RGB range | Actual attenuation range |
| --- | --- | --- | --- | --- |
| JapaneseGradens | E0 | [0, 0.999898851] | [0.000359660, 1] | [0.000325203, 0.455322266] |
| JapaneseGradens | E2 | [0, 0.999898612] | [0.000862107, 0.999999881] | [0.000456572, 0.465087891] |
| Panama | E0 | [0.006679244, 0.999900103] | [0.007216293, 1] | [0.026702881, 0.325927734] |
| Panama | E2 | [0.001024887, 0.999899924] | [0.004877942, 0.999999881] | [0.026550293, 0.317382812] |
| Curasao | E0 | [0, 0.999900103] | [5.35442e-9, 1] | [0.133056641, 0.468261719] |
| Curasao | E2 | [0, 0.999900103] | [1.42362e-8, 0.999999881] | [0.109924316, 0.488525391] |
| IUI3 | E0 | [0, 0.999900043] | [4.21205e-9, 1] | [0.004657745, 0.323486328] |
| IUI3 | E2 | [0, 0.999899983] | [4.36607e-9, 0.999999881] | [0.004585266, 0.297851562] |

There is no newly supported pixel subset on which to test whether colors
are sufficiently constrained: recovered original contributions = 0;
newly exposed colors = 0; newly exposed floaters/streaks = 0. This result
cannot be reframed as successful repair or accurate clear reconstruction.

## Runtime And Memory

Entries are scene means of per-view warmed median wall times, and scene
means of allocator incremental peaks. Each timer covers the rasterizer
wrapper, including bin/sort and output construction, but not the entire
model pipeline. Per-view CUDA event times, five individual samples,
allocated baselines, absolute peaks and reserved peaks are in diagnostics.

| Scene | Arm | Original ms | Independent zero-medium ms | Original incremental MiB | Zero-medium incremental MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| JapaneseGradens | E0 | 5.501 | 5.482 | 148.484 | 148.541 |
| JapaneseGradens | E2 | 5.478 | 5.467 | 148.210 | 148.210 |
| Panama | E0 | 7.849 | 7.812 | 246.457 | 246.457 |
| Panama | E2 | 7.845 | 7.803 | 244.863 | 244.863 |
| Curasao | E0 | 7.371 | 7.304 | 245.186 | 245.186 |
| Curasao | E2 | 7.374 | 7.364 | 244.223 | 244.223 |
| IUI3 | E0 | 4.655 | 4.635 | 145.040 | 143.242 |
| IUI3 | E2 | 4.611 | 4.595 | 146.429 | 146.429 |

Keeping the original underwater call and adding a separate clear call costs
approximately another rasterization, with no visual benefit here. The small
differences between original and zero-medium timings are not a speedup for
the combined workflow. This is not a full-pipeline FPS or training benchmark.

Original and independent outputs are retained during both timing series.
Incremental peaks exclude that retained baseline, the model, and the
already allocated shared zero tensor (H x W x 3 x 4 bytes, about 14.5-24.4
MiB). They include temporary sort buffers and newly created outputs. These
values are not total end-to-end VRAM costs. Allocator history and fixed
timing order can explain small peak differences. Workers ran concurrently
on separate GPUs, with no other processes on the assigned GPUs.

## Visual Review And Interpretation

All eight clear contact sheets and all eight coverage contact sheets were
visually inspected, covering every view in both arms. The x20 difference
columns and recovered-black masks are entirely zero. The pre-existing
JapaneseGradens upper dark regions and boundary stretching, Panama cyan
upper background and streaks, Curasao dark top/right and colored/gray
streaks, and IUI3 dark regions and radiating streaks persist unchanged.
The E0/E2 differences are historical model differences, not DVS effects.

There is no true clear-image GT. Near-black/cyan/dark fractions describe
the images and cannot establish physical accuracy, object ownership, or
repair quality. We do not claim underwater PSNR improvement: underwater
was the original call throughout, so any recomputed deterministic metric
would have the same input pixels as the retained historical evaluation.

## Artifacts And Verification

Authoritative successful root:
`outputs/dvs_frozen_visibility_20260911/verified_v2/`.

- [Summary](../outputs/dvs_frozen_visibility_20260911/verified_v2/summary.json),
  [per-view CSV](../outputs/dvs_frozen_visibility_20260911/verified_v2/per_view.csv),
  [scene means](../outputs/dvs_frozen_visibility_20260911/verified_v2/scene_means.csv).
- [Final independent artifact verification](../outputs/dvs_frozen_visibility_20260911/verified_v2/final_verification.json)
  and [visual review](../outputs/dvs_frozen_visibility_20260911/verified_v2/visual_review.json).
- `<scene>/<arm>/audit.json`: source/checkpoint/state hashes and all controls.
- `<scene>/<arm>/<view>/native_float32.npz`: clear and accumulation pairs.
- Same view folder: original/zero-medium raw and tone-mapped clear PNGs,
  accumulation PNGs, `difference_x20.png`, `coverage_difference_x20.png`,
  `recovered_black_mask.png`, historical underwater/GT symlinks, diagnostic.
- `<scene>/<arm>/clear_contact_sheet.jpg` and `coverage_contact_sheet.jpg`.
- `registration.json`, four scene logs, four actual-CUDA synthetic checks,
  and `queue_status.json` with all four workers exited successfully.

Example complete all-view E2 sheets:
[JapaneseGradens](../outputs/dvs_frozen_visibility_20260911/verified_v2/JapaneseGradens-RedSea/E2/clear_contact_sheet.jpg),
[Panama](../outputs/dvs_frozen_visibility_20260911/verified_v2/Panama/E2/clear_contact_sheet.jpg),
[Curasao](../outputs/dvs_frozen_visibility_20260911/verified_v2/Curasao/E2/clear_contact_sheet.jpg),
[IUI3](../outputs/dvs_frozen_visibility_20260911/verified_v2/IUI3-RedSea/E2/clear_contact_sheet.jpg).

Final verification reloaded all 26 NPZs, checked finite float32 native
dimensions and bit equality, matched raw PNG encoding, checked zero
difference/recovery images, and rehashed all 8 source checkpoints and 104
source PNGs. Counts: 234 new PNGs, 52 valid historical-image symlinks, 26
NPZs, 16 contact sheets. Core and binary hashes remained unchanged; physical
GPUs 6-9 were empty after completion. Both diagnostic scripts compile and
the scoped diff whitespace check passes.

The first attempt remains preserved in the parent output directory. It
stopped during first-view camera-hash logging because `Cameras.to_json`
requires `camera_idx`. Only the new diagnostic's call was corrected to
`to_json(camera_idx=0)`, and the complete audit reran into `verified_v2`.
The failed attempt is not pooled into the authoritative results, and no
historical assets were overwritten.

Implementation: `scripts/diagnostics/audit_dvs_frozen_visibility.py`.
Independent saved-artifact check: `scripts/diagnostics/verify_dvs_frozen_artifacts.py`.
Both refuse to overwrite their authoritative output destinations. Any
future replay must use a new output root and the already installed conda
environment; source hashes specify the audited version.

## Decision

Stop DVS as an attenuation-culling repair direction for the current system.
The existing renderer already keeps positive medium attenuation out of the
discrete clear support rule. A second zero-medium call duplicates existing
clear output, adds cost, and overlaps both historical code and conventional
medium-removal rendering.

There is no evidence here to justify training with new dual support,
adding clear supervision, or claiming an independent methodological
contribution. In particular, changing `min(0, ...)` into `max(0, ...)` would
introduce real attenuation-dependent culling and change underwater behavior;
it must not be presented as a fix for the rejected hypothesis. No points,
colors or exposure were adjusted to rescue this negative result.
