# SeaFree-GS / WaterSplatting Implementation Audit - 2026-08-08

This note records source-code facts only. No SeaFree-inspired training experiment was started during this audit, and no file in the SeaFree-GS reference repository was modified.

## Repository State

### WaterSplatting

- Repository: `/mnt/new/home_old/ycy/water-splatting-refactor`
- Branch at audit start: `research/gmvc-medium-calibration`
- HEAD at audit start: `794419943d7ec3700d3929a83afaf98e750541a9`
- Latest commits at audit start:
  - `7944199 Validate D010 persistence and scratch training`
  - `dfbf1ef Add direct optical-depth dewatering audit`
  - `234dd62 Add GMVC four-scene visualization export`
- Untracked historical files present and left untouched:
  - `scripts/diagnostics/render_gmvc_curasao_contact_sheet.py`
  - `scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py`

### SeaFree-GS Reference

- Repository: `/mnt/new/home_old/ycy/reference_repos/SeaFree-GS`
- Reference commit: `7797e97dae831029ac89ae9f37b3c3d69ec2cf6c`
- `git status --short`: clean
- Read/search only. No edits, commits, or pushes were made in the reference repository.

## Scope

The audit covers six implementation areas requested for future controlled experiments:

1. LOS distance `/10` complete call path.
2. Intrinsic color boundedness.
3. Foreground-aware content loss.
4. Background-water supervision.
5. Coarse depth loss.
6. SH capacity.

For each area, the note separates:

- SeaFree code fact
- WaterSplatting code fact
- Structural difference
- Reasonable inference
- Proposed controlled experiment

## 1. LOS Distance `/10` Complete Call Path

### SeaFree Code Fact

- In `seafree_gs/seafree_model.py`, Gaussian line-of-sight vectors are computed as `means_crop - camera_position` from the inverse camera view matrix and are detached before distance/direction use (`seafree_model.py:602-608`).
- `gaussian_line_of_sight_distances = norm(gaussian_line_of_sight_vectors)` and `gaussian_line_of_sight_directions = vector / distance` (`seafree_model.py:607-608`).
- The water property predictor is queried for both Gaussian LOS directions and pixel LOS directions in a single concatenated batch (`seafree_model.py:611-650`).
- Water property activations:
  - Ambient light `A`: sigmoid on raw channels `0:3` (`seafree_model.py:640`).
  - Backscatter coefficients: softplus on raw channels `3:6` (`seafree_model.py:641`).
  - Direct attenuation coefficients: softplus on raw channels `6:9` (`seafree_model.py:642`).
- The Gaussian LOS distance is divided by 10 at `normalized_gaussian_line_of_sight_distances = gaussian_line_of_sight_distances / 10` (`seafree_model.py:652`).
- The same normalized Gaussian distance is used for both:
  - Direct attenuation: `colors_crop * exp(-attenuation_coefficients * distance / 10)`.
  - Gaussian-level backscatter: `A * (1 - exp(-backscatter_coefficients * distance / 10))`.
  These are composed in `degraded_gaussian_colors` (`seafree_model.py:653-655`).
- The degraded underwater Gaussian colors and intrinsic Gaussian colors are concatenated and rasterized together (`seafree_model.py:656-678`).
- Pixel background output is `water_background_image = pixel_ambient_light_colors`; pixel backscatter and attenuation coefficients are produced and returned, but the rendered background color is ambient `A` rather than a finite-distance background integral (`seafree_model.py:658-660`, `seafree_model.py:691-692`).
- The final RGB is `render[..., :3] + (1 - alpha) * water_background_image`, clamped to `[0,1]` (`seafree_model.py:691-692`).
- The same `get_outputs` implementation is used for training and evaluation; no separate inference-time water formula was found in the audited path.
- Scene scale is applied before this LOS computation. `seafree_gs/seafree_dataparser.py` auto-orients and centers poses, then scales translations by `1 / max(abs(poses translation)) * scale_factor`; COLMAP points are transformed and multiplied by the same scale (`seafree_dataparser.py:164-176`, `seafree_dataparser.py:277-288`). The GS strategy is initialized with `scene_scale=1.0` (`seafree_model.py:334`).

### WaterSplatting Code Fact

- Current baseline config has `direct_optical_depth_scale: float = 1.0`, documented as applying only to direct attenuation before underwater rasterization (`water_splatting/water_splatting.py:289-290`).
- In the current branch, `medium_attn_raw = medium.attn`, then `medium_attn = medium_attn_raw * direct_optical_depth_scale`; negative scale is rejected (`water_splatting.py:3181-3187`).
- If the depth-context medium pass is used, the same raw/effective recomputation is repeated after the second medium query (`water_splatting.py:3362-3365`).
- The underwater rasterizer receives `medium_bs=medium_bs` and `medium_attn=medium_attn` separately (`water_splatting.py:3367-3379`). Thus the implemented D010 scale affects direct attenuation only and does not scale backscatter.
- `UnderwaterRasterizer.rasterize` calls the CUDA rasterization path and returns `rgb_object`, `rgb_clear_raw`, `rgb_medium`, depth, alpha, and transmission-related depth statistics. Final underwater RGB is `rgb_object + rgb_medium` (`water_splatting/rendering/underwater_rasterizer.py:92-126`).
- Current renderer outputs include both `medium_attn` and `medium_attn_raw`, so diagnostics can distinguish effective and raw direct coefficients (`water_splatting.py:3733-3768`).

### Structural Difference

- SeaFree divides the LOS distance by 10 before applying both direct attenuation and Gaussian-level backscatter.
- WaterSplatting's current D010 experiment applies `gamma_D` only to direct attenuation coefficients before the underwater rasterizer. It leaves `medium_bs`, finite/tail backscatter, `medium_rgb`, and `B_inf` unchanged.
- SeaFree's `/10` is a geometric distance normalization inside the degradation formula; WaterSplatting's D010 is a coefficient-side scale on the direct attenuation branch only.
- Because the direct coefficient is learnable in both systems, a distance or coefficient scale can be partially or fully absorbed by increasing learned attenuation coefficients.

### Reasonable Inference

- The previous WaterSplatting D010 results are not a full reproduction of SeaFree's LOS scaling because SeaFree also changes the effective backscatter optical depth on foreground Gaussians.
- A SeaFree-style `/10` cannot be interpreted as a guaranteed 10x attenuation reduction unless the learned coefficients are audited. The relevant variables are raw beta, effective beta, tau, and transmission after optimization.

### Proposed Controlled Experiment

- Do not replace the existing D010 result. Add a separate diagnostic flag only if needed, for example `los_distance_scale_for_direct_and_gaussian_backscatter`, default `1.0`.
- In one controlled run, scale both direct and Gaussian-level backscatter optical depths by the same factor while keeping all other WaterSplatting settings unchanged.
- Track `beta_D_raw`, `beta_D_effective`, `beta_B_raw`, `beta_B_effective`, `tau_D`, foreground backscatter optical depth, `T_D`, underwater RGB metrics, and full-SH clear-object saturation.
- Stop if raw coefficients compensate to recover the baseline effective optical depths.

## 2. Intrinsic Color Boundedness

### SeaFree Code Fact

- `SeaFreeGsModelConfig` has `sh_degree=3` as a class default (`seafree_model.py:151`), but the public `SeaFreeGsMethod` overrides the model config to `sh_degree=0` (`seafree_gs/seafree_config.py:32-36`).
- When seeded points are available and `sh_degree == 0`, SeaFree initializes `features_dc` with `logit(seed_rgb / 255)` (`seafree_model.py:260-266`).
- The `colors` property returns `sigmoid(features_dc)` when `sh_degree == 0` (`seafree_model.py:339-344`).
- In `get_outputs`, when `sh_degree == 0`, `colors_crop = sigmoid(colors_crop).squeeze(1)` and `sh_degree_to_use = None` before underwater degradation (`seafree_model.py:595-599`).
- The bounded sigmoid color enters the underwater degradation formula directly and is also concatenated as the intrinsic color render channel (`seafree_model.py:653-656`, `seafree_model.py:694-695`).
- The gradient path from RGB/content losses to intrinsic color parameters passes through the sigmoid activation. No hard clamp was found in the training path for the SH0 intrinsic colors. The returned `intrinsic_color_render` is clamped for output (`seafree_model.py:694-718`).

### WaterSplatting Code Fact

- Current WaterSplatting baseline uses `sh_degree=3` (`water_splatting.py:183`). The active color for rendering is computed with `compute_gaussian_colors(..., sh_degree=self.config.sh_degree, active_sh_degree=n)` (`water_splatting.py:3283-3292`).
- For seeded SH>0 initialization, DC features are initialized through `RGB2SH(seed_rgb / 255)` (`water_splatting.py:866-873`).
- If WaterSplatting is configured with `sh_degree == 0`, the `colors` property returns `sigmoid(features_dc)` (`water_splatting.py:942-947`), but the `shs_0` property itself returns raw `features_dc` (`water_splatting.py:949-951`). The current baseline and D010 branch do not use SH0.
- Current full-SH clear-object render is `J_gaussian_raw`, from the underwater rasterizer's `rgb_clear_raw` (`water_splatting/rendering/underwater_rasterizer.py:92-126`, `water_splatting.py:3405-3410`, `water_splatting.py:3733-3755`).
- Display variants are:
  - `rgb_clear_clamp`: `clamp(J_gaussian_raw, 0, 1)`.
  - `rgb_clear`: `J_gaussian_raw / (J_gaussian_raw + 1)`.
  These are display/diagnostic mappings, not training-time hard bounds in the baseline (`underwater_rasterizer.py:124-126`).

### Structural Difference

- SeaFree's public method uses SH0 with a sigmoid-bounded intrinsic color parameterization from the start.
- WaterSplatting's current research branch intentionally keeps SH3 active. Its full-SH view-dependent colors are not bounded by sigmoid before the underwater compositor.
- WaterSplatting display clamp and WaterSplatting-style `J/(J+1)` tonemap do not constrain the trained intrinsic Gaussian colors.

### Reasonable Inference

- SeaFree's bounded SH0 color can reduce intrinsic color blow-up by construction, but it also reduces appearance capacity relative to SH3.
- Prior WaterSplatting experiments observed an RGB drop when switching to SH0, so SH0/sigmoid boundedness is not an immediate replacement for optical-depth calibration.
- A soft full-SH bound would test color boundedness while preserving SH3, and is structurally closer to the current WaterSplatting branch than switching directly to SH0.

### Proposed Controlled Experiment

- Keep SH0 deferred unless optical-depth calibration, background supervision, and weak SH3 soft bounds fail.
- If color boundedness is tested, first run a no-training audit of current-view full-SH colors `c_i(view)` to estimate raw bound loss scale.
- Then test a single weak SH3 soft bound on current-view visible Gaussian colors, not only `features_dc`, with `lambda * L_bound` initialized near 1% of the main RGB loss.
- Do not hard clamp or sigmoid the renderer output in this branch.

## 3. Foreground-Aware Content Loss

### SeaFree Code Fact

- SeaFree loss starts from underwater GT and rendered underwater RGB (`seafree_model.py:812-813`).
- `pseudo_depth = batch["depth_image"]`, downscaled if needed, moved to device, and normalized by `pseudo_depth.max()` (`seafree_model.py:816-818`).
- If a dataset `mask` is present, GT, rendered RGB, pseudo depth, and rendered depth are multiplied by the mask before loss calculations (`seafree_model.py:823-831`).
- Foreground mask construction uses the normalized pseudo-depth map:
  - `background_depth_threshold = 1e-2`.
  - `mask_1e_2_copy = (pseudo_depth < 1e-2) * 255`.
  - OpenCV threshold with `THRESH_BINARY_INV`.
  - External contours are found.
  - The largest contour is drawn filled as foreground.
  - The result is binarized and cached by image index/downscale (`seafree_model.py:833-866`).
- Background pixel ratio is computed from the complement of that foreground mask (`seafree_model.py:857-865`).
- Foreground-aware reconstruction weight is `1 / (rendered_underwater_image.detach() + 1e-3)`, but it is replaced by 1 on background pixels (`seafree_model.py:868-873`).
- L1 term: `abs((gt_underwater_image - rendered_underwater_image) * weight).mean()` (`seafree_model.py:874-876`).
- SSIM term: GT and rendered images are both multiplied by the same weight before SSIM; large images are split into four blocks (`seafree_model.py:878-893`).
- Content-based reconstruction loss combines foreground-weighted L1, foreground-weighted DSSIM, and background water supervision:
  `content = (1 - ssim_lambda) * foreground_weighted_l1 + ssim_lambda * foreground_weighted_dssim + 0.01 * background_water_supervision_loss` (`seafree_model.py:933-938`).
- `ssim_lambda` defaults to `0.2` (`seafree_model.py:147`).

### WaterSplatting Code Fact

- Current baseline main loss is `reconstruction_loss(gt_img, pred_img_for_loss, main_loss, ssim_loss, ssim_lambda, ssim_metric)` (`water_splatting.py:4114-4122`).
- WaterSplatting does not currently use SeaFree's pseudo-depth-derived largest-contour foreground mask in the baseline.
- A separate optional foreground transmission reconstruction term exists, default off. It loads a region mask from `backscatter_region_mask_dir`, uses key `foreground_water_mask_key`, builds weights from `1 - transmission`, optionally detaches weights, and applies an L1 residual term (`water_splatting.py:275-282`, `water_splatting.py:4493-4512`).
- Region masks are loaded from `view_XXXX_regions.pt` using `_load_backscatter_region_mask`; no pseudo-depth threshold/largest-contour foreground construction is used in this loader (`water_splatting.py:451-456`, `water_splatting.py:2871-2910`).

### Structural Difference

- SeaFree's foreground-aware content loss is pseudo-depth-driven and directly reweights both L1 and SSIM using inverse predicted intensity on foreground pixels.
- WaterSplatting's existing optional foreground weighting is region-mask-driven, transmission-driven, L1-only, and disabled by default.
- SeaFree's foreground/background split is derived automatically from `depth_image`; WaterSplatting's current masks come from precomputed region files.

### Reasonable Inference

- SeaFree's content loss may change the optimization pressure on dark or highly attenuated foreground pixels independently of the physical degradation formula.
- If reproduced without separating it from LOS scaling and background/depth supervision, it would be impossible to identify which component affects intrinsic saturation.

### Proposed Controlled Experiment

- First implement a diagnostic-only pseudo-depth foreground mask builder matching SeaFree semantics and report mask coverage/contact sheets.
- If coverage is technically valid, run one foreground-aware content-loss experiment with:
  - The same D100 baseline physics.
  - The same pseudo-depth foreground mask rule.
  - SeaFree's inverse rendered-intensity weight formula.
  - L1 and SSIM weighting separated in metrics.
- Do not combine this loss with D010, background supervision, or coarse depth in the first controlled run.

## 4. Background-Water Supervision

### SeaFree Code Fact

- SeaFree uses the complement of the pseudo-depth-derived foreground mask as background support (`seafree_model.py:833-866`, `seafree_model.py:921-929`).
- Background supervision is enabled when `enable_background_water_supervision` is true, `self.step < 15000`, and `background_pixel_ratio > 0.05` (`seafree_model.py:921-923`).
- The target compares `water_background_image` to GT underwater pixels on background support (`seafree_model.py:924-929`).
- In SeaFree's render path, `water_background_image = pixel_ambient_light_colors`; it is ambient light `A`, not a finite-distance rendered medium/backscatter contribution (`seafree_model.py:648-660`, `seafree_model.py:691-692`).
- The weight is `1 / (background_ambient_light_pixels.detach() + 1e-3)` (`seafree_model.py:926`).
- The raw background supervision loss is an L1 mean and enters the content loss with coefficient `0.01` (`seafree_model.py:927-938`).
- The public method config leaves `enable_background_water_supervision=True` by default through the model config default (`seafree_model.py:191`, `seafree_gs/seafree_config.py:32-41`).
- No upper coverage gate was found in the audited code; the explicit coverage condition is only `background_pixel_ratio > 0.05`.

### WaterSplatting Code Fact

- Current branch has default-off `medium_background_supervision_enabled=False` and `medium_background_supervision_lambda=0.0` (`water_splatting.py:291-294`).
- When enabled, WaterSplatting loads a precomputed region mask from `backscatter_region_mask_dir`, key `background_water_mask_key`, and optionally excludes boundary pixels and high hit-confidence pixels using `effective_background_mask` (`water_splatting.py:451-456`, `water_splatting.py:4346-4365`, `water_splatting/losses/background_attribution.py:21-45`).
- The WaterSplatting target is `outputs["medium_rgb"]` versus GT underwater RGB on the detached effective background mask (`water_splatting.py:4366-4371`).
- The weight is `1 / (outputs["medium_rgb"].detach() + 1e-3)` (`water_splatting.py:4366-4371`).
- Metrics record `background_medium_l1`, `weighted_background_medium_l1`, mask coverage, and lambda when active (`water_splatting.py:4372-4379`).
- Other optional WaterSplatting background losses can target `b_inf`, `rgb_medium_total`, or `rgb_tail`, but they are separate optional losses and are default off in the M1/D010 baseline (`water_splatting.py:4308-4344`).

### Structural Difference

- SeaFree supervises ambient `A` on pseudo-depth background pixels.
- WaterSplatting's first medium background supervision implementation supervises `medium_rgb` on precomputed water-region masks.
- SeaFree's background support is derived from pseudo depth and largest connected foreground extraction; WaterSplatting's support is loaded from region-mask files with optional boundary/hit exclusions.
- SeaFree applies background supervision as an internal fixed `0.01` term inside content loss; WaterSplatting exposes an explicit lambda.

### Reasonable Inference

- SeaFree's background supervision is an independent medium-color anchor, not a direct supervision of finite-distance backscatter.
- WaterSplatting's `medium_rgb` target is structurally close to SeaFree's ambient `A` target when `b_inf_mode=tied`, but mask semantics differ.
- Mask semantics can dominate this experiment; a pseudo-depth background audit is required before interpreting training results.

### Proposed Controlled Experiment

- Before new training, build a SeaFree-style pseudo-depth foreground/background mask audit in WaterSplatting and compare coverage with existing region masks.
- If coverage is valid, run one background-supervision-only experiment:
  - `gamma_D = 1.0`.
  - SH3 unchanged.
  - No D010, foreground content loss, or coarse depth loss.
  - Target either current `medium_rgb` or explicit ambient-equivalent output, documented before running.
- Record background residual, beta/tau/T, full-SH clear saturation, and RGB safety. Do not interpret lower tau as physical correctness without medium GT.

## 5. Coarse Depth Loss

### SeaFree Code Fact

- SeaFree's README states that the public release assumes depth supervision and expects 16-bit grayscale depth maps with relative disparity-like semantics: near large, far small. The example folder is `depthAnything_u16/` (`README.md:129-139`).
- The datamanager caches `depth_image` to GPU when available (`seafree_gs/seafree_datamanager.py:91-99`).
- The public method config sets `output_depth_during_training=True` (`seafree_gs/seafree_config.py:32-35`).
- In `get_outputs`, `render_mode = "RGB+ED"` if `output_depth_during_training` is true or if not training (`seafree_model.py:590-593`).
- Rendered depth is the expected-depth channel from gsplat. For no-alpha pixels, SeaFree replaces depth with the 0.95 quantile of valid depth (`seafree_model.py:702-706`).
- In loss, `pseudo_depth` is normalized by max and flattened; rendered depth is flattened (`seafree_model.py:816-818`, `seafree_model.py:912-914`).
- If `enable_coarse_grained_depth_loss` is true, SeaFree computes:
  `approximate_rendered_disparity = 1 / (rendered_depth_flattened * 10 + 1)`,
  then `coarse_grained_depth_loss = 1 - pearson_corrcoef(pseudo_depth_flattened, approximate_rendered_disparity)` (`seafree_model.py:915-917`).
- The loss enters with coefficient `0.1` (`seafree_model.py:940`).
- `enable_coarse_grained_depth_loss=True` in the model config default (`seafree_model.py:189`). No step-gating beyond the config flag was found.
- Dataset masks, if present, multiply both pseudo depth and rendered depth before flattening (`seafree_model.py:823-831`).

### WaterSplatting Code Fact

- WaterSplatting currently has `lambda_pseudo_depth: float = 0.0`, documented as a reserved pseudo-depth rank-consistency loss weight and off by default (`water_splatting.py:283-284`).
- No active SeaFree-style coarse depth Pearson loss was found in the current baseline path.
- WaterSplatting renderer already outputs expected depth and related depth statistics (`water_splatting/rendering/underwater_rasterizer.py:128-158`, `water_splatting.py:3733-3741`).
- Existing diagnostics contain Pearson/Spearman helper code for audits, but that is not an active training loss in the baseline.

### Structural Difference

- SeaFree requires pseudo depth as a dataset input and trains with a coarse depth correlation loss by default.
- WaterSplatting current M1/D010 experiments do not use pseudo-depth supervision in training.
- SeaFree's depth target is disparity-like pseudo depth, while rendered depth is converted to approximate disparity using `1 / (10 * depth + 1)`.

### Reasonable Inference

- SeaFree's coarse depth loss can affect Gaussian geometry and depth distribution, which can indirectly affect optical depth `tau_D = beta_D * depth`.
- Adding this loss to WaterSplatting would not be a pure direct-attenuation calibration experiment; it changes geometry/depth optimization pressure.

### Proposed Controlled Experiment

- First run a diagnostic-only pseudo-depth/rendered-depth correlation audit on existing WaterSplatting checkpoints:
  - Normalize pseudo depth using the SeaFree rule.
  - Compute rendered approximate disparity `1 / (10 * rendered_depth + 1)`.
  - Report Pearson correlation with and without masks.
- Only if the diagnostic correlation is meaningful and depth inputs are already available locally, run one coefficient-matched coarse-depth experiment at `lambda_pseudo_depth=0.1`.
- Do not combine coarse depth with D010, background supervision, or foreground-aware content loss in the first run.

## 6. SH Capacity

### SeaFree Code Fact

- `SeaFreeGsModelConfig` has a class default `sh_degree=3`, but the registered public `SeaFreeGsMethod` overrides it to `sh_degree=0` (`seafree_model.py:151`, `seafree_gs/seafree_config.py:32-36`).
- With `sh_degree=0`, SeaFree uses only DC/color parameters for each Gaussian. In the render path, `colors_crop` is sigmoid-activated and `sh_degree_to_use=None`, so no view-dependent SH bases are evaluated by rasterization (`seafree_model.py:595-599`, `seafree_model.py:663-678`).
- `features_rest` is still present as an empty SH-rest tensor when `dim_sh - 1 == 0`; it does not provide higher-order view-dependent color capacity under the public SH0 method (`seafree_model.py:251-270`).

### WaterSplatting Code Fact

- Current WaterSplatting M1/D010 baseline uses `sh_degree=3` and active SH degree scheduling through `_get_active_sh_degree()` (`water_splatting.py:183`, `water_splatting.py:3002-3008`, `water_splatting.py:3283-3292`).
- The current-view Gaussian color is computed by evaluating `features_dc + features_rest` with spherical harmonics and then applying `torch.clamp(rgbs + 0.5, min=0.0)`; there is no upper bound (`water_splatting/fields/gaussian_appearance.py:24-36`).
- The image-space clear-object tensor `J_gaussian_raw` is the alpha-composited render of those current-view full-SH colors without medium attenuation/backscatter (`water_splatting/rendering/underwater_rasterizer.py:92-126`, `water_splatting.py:3405-3410`).

### Structural Difference

- SeaFree reduces intrinsic appearance capacity by using SH0 in the public method and bounds color through sigmoid.
- WaterSplatting preserves SH3 view-dependent capacity and only lower-bounds evaluated colors at zero; high positive values remain available as a compensation channel.
- Therefore, SeaFree's stable intrinsic appearance may be caused by bounded color, reduced SH capacity, or both. These cannot be separated by switching WaterSplatting directly to SH0.

### Reasonable Inference

- If WaterSplatting full-SH evaluated colors `c_i(v)` have substantial mass above 1.0, then the remaining pure-J issue after D010 can originate at Gaussian-level appearance rather than only image-space alpha accumulation.
- SH0 is a confounded intervention because it changes both boundedness and view-dependent capacity.

### Proposed Controlled Experiment

- Do not run SH0 in the current round.
- Preserve SH3 and test only a weak current-view full-SH color bound:
  `mean(ReLU(c_i(v)-1)^2 + ReLU(-c_i(v))^2)` on visible Gaussians.
- Select the bound weight by no-update gradient matching against RGB gradients on `features_dc` and `features_rest`.
- Report both Gaussian-level `c_i(v)` statistics and image-space `J_gaussian_raw` statistics to distinguish appearance-parameter overflow from alpha-composition effects.

## Implementation Boundary for Next Work

- No SeaFree-GS code should be copied into WaterSplatting.
- Any future implementation should be a minimal WaterSplatting-native experiment behind default-off flags.
- Suggested order remains:
  1. Pseudo-depth mask/depth audit only.
  2. Single-factor background supervision or foreground content loss.
  3. Single-factor SeaFree-style LOS scaling, clearly separated from the already implemented direct-only D010.
  4. Depth loss only after pseudo-depth semantics are verified.
  5. SH0/sigmoid boundedness remains a deferred hypothesis because prior WaterSplatting experiments observed RGB degradation when lowering appearance capacity.

## Audit Status

- Source audit completed.
- WaterSplatting-native controlled training experiments are recorded separately in `research_notes/DEWATERING_SEAFREE_FACTOR_AUDIT_2026-08-08.md`.
- SeaFree reference repository: read-only, clean.
- SeaFree-GS code was not modified or copied into WaterSplatting.
