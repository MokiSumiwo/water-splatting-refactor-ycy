# SeaFree-GS vs WaterSplatting Degradation-Compositing Audit

Date: 2026-08-10

## Motivation

### Code Fact

This stage was a read-only / forward-only diagnostic. No optimizer, scheduler, densification, pruning, opacity reset, checkpoint finetuning, renderer physics change, loss change, GMVC, D010, medium supervision, SH0, soft bound, or new training run was executed.

Diagnostic source:

```text
scripts/diagnostics/audit_seafree_ws_degradation_compositing.py
```

Outputs:

```text
outputs/dcomp_audit_20260810/
renders/dcomp_audit_20260810/
logs/dcomp_audit_20260810/
```

## Repository State

### Experimental Fact

WaterSplatting:

```text
branch = research/m1-bounded-intrinsic
START_HEAD = 1f8fa1c743458c88cf8043dea3fe05190102cc85
START_COMMIT = Test bounded headroom SH3 appearance
```

SeaFree-GS reference:

```text
repo = /mnt/new/home_old/ycy/reference_repos/SeaFree-GS
reference commit = 7797e97dae831029ac89ae9f37b3c3d69ec2cf6c
status --short = clean
```

Historical untracked GMVC scripts were present and were not modified:

```text
scripts/diagnostics/render_gmvc_curasao_contact_sheet.py
scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py
```

## SeaFree Source Semantics

### Code Fact

Key source paths:

```text
/mnt/new/home_old/ycy/reference_repos/SeaFree-GS/seafree_gs/seafree_config.py
/mnt/new/home_old/ycy/reference_repos/SeaFree-GS/seafree_gs/seafree_model.py
/mnt/new/home_old/ycy/reference_repos/SeaFree-GS/third_party/gsplat/gsplat/rendering.py
/mnt/new/home_old/ycy/reference_repos/SeaFree-GS/third_party/gsplat/gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu
```

The registered SeaFree method sets:

```text
rasterize_mode = antialiased
sh_degree = 0
```

With `sh_degree=0`, intrinsic Gaussian color is:

```text
c_i = sigmoid(features_dc_i)
```

before underwater degradation.

SeaFree computes Gaussian LOS vector from camera center to Gaussian mean, detaches it, uses Euclidean distance, and applies:

```text
d_i' = ||mean_i - camera_center|| / 10
```

The water predictor returns:

```text
A       = sigmoid(raw[:3])
beta_B  = softplus(raw[3:6])
beta_D  = softplus(raw[6:9])
```

The code constructs degraded Gaussian colors before rasterization:

```text
c_i^deg =
    c_i * exp(-beta_D,i * d_i')
    +
    A_i * (1 - exp(-beta_B,i * d_i'))
```

Then it rasterizes degraded RGB and intrinsic RGB together as six channels. After rasterization:

```text
I_SF = render[..., :3] + (1 - alpha_image) * A_pixel
I_SF = clamp(I_SF, 0, 1)
J_SF = render[..., 3:6]
J_SF_clamped = clamp(J_SF, 0, 1)
```

gsplat uses front-to-back alpha compositing:

```text
alpha_i = min(0.999, opacity_i * exp(-sigma_i))
w_i = alpha_i * T_alpha_before_i
T_alpha_after_i = T_alpha_before_i * (1 - alpha_i)
```

Antialiased mode multiplies opacity by projection compensation before the CUDA rasterization.

## WaterSplatting Source Semantics

### Code Fact

Key source paths:

```text
water_splatting/water_splatting.py
water_splatting/rendering/underwater_rasterizer.py
water_splatting/fields/gaussian_appearance.py
water_splatting/fields/medium_field.py
water_splatting/cuda/csrc/forward.cu
```

WaterSplatting computes current-view Gaussian color before rasterization. In the current branch:

```text
legacy SH3:
  c_i = clamp(SH_i(view) + 0.5, min=0)

bounded_sh3:
  c_i = sigmoid(SH_logit_i(view))

bounded_headroom_sh3:
  c_i = bounded headroom mapping around sigmoid(DC logit)
```

The medium field is per-pixel:

```text
medium_rgb  = sigmoid(raw[:3])
medium_bs   = softplus(raw[3:6] + density_bias)
medium_attn = softplus(raw[6:9] + density_bias)
```

With `b_inf_mode=tied`, `b_inf = medium_rgb`.

Projected Gaussian depth is camera-space `z` from `project_gaussians`, not SeaFree's Euclidean LOS distance. WaterSplatting does not divide this depth by 10 inside the underwater CUDA path.

Classic opacity is:

```text
opacity_eff = sigmoid(opacity)
```

Antialiased opacity is:

```text
opacity_eff = sigmoid(opacity) * projection_compensation
```

The underwater CUDA kernel accumulates:

```text
D_WS = sum_i w_i c_i exp(-beta_D,pixel * d_i)
J_WS = sum_i w_i c_i
```

Backscatter / water medium is accumulated over ray segments:

```text
B_segment_i =
    T_alpha_before_i
    *
    A_pixel
    *
    (exp(-beta_B,pixel * d_{i-1}) - exp(-beta_B,pixel * d_i))

B_tail =
    T_alpha_final
    *
    A_pixel
    *
    exp(-beta_B,pixel * d_N)
```

The exposed prediction is:

```text
I_WS = direct_object_signal + rgb_medium
```

For an empty ray or no contributing Gaussian coverage, the medium/background value is `medium_rgb`.

## Unified Equations

### Inferred Formula

Use:

```text
c_i        = Gaussian current-view intrinsic RGB
alpha_i    = effective screen-space alpha
d_i        = scalar depth/distance after each method's own definition
T_D,i      = exp(-beta_D,i d_i)
T_alpha    = Gaussian alpha-compositing transmittance
w_i        = alpha_i prod_{j<i}(1-alpha_j)
```

SeaFree formula reference:

```text
D_SF  = sum_i w_i c_i exp(-beta_D,i d_i')
B_SF  = sum_i w_i A_i (1 - exp(-beta_B,i d_i'))
BG_SF = T_alpha_final A_pixel
I_SF  = D_SF + B_SF + BG_SF
```

WaterSplatting formula reference:

```text
D_WS = sum_i w_i c_i exp(-beta_D,pixel d_i)

B_WS =
    sum_i T_alpha_before_i A_pixel
    (exp(-beta_B,pixel d_{i-1}) - exp(-beta_B,pixel d_i))
    +
    T_alpha_final A_pixel exp(-beta_B,pixel d_N)

I_WS = D_WS + B_WS
```

Detailed symbolic files:

```text
outputs/dcomp_audit_20260810/symbolic_single_gaussian.md
outputs/dcomp_audit_20260810/symbolic_two_gaussian.md
outputs/dcomp_audit_20260810/equivalence_conditions.md
```

## Equivalence

### Quantitative Result

Final classification:

```text
ARE_SEAFREE_AND_WS_DEGRADATION_COMPOSITING_EQUIVALENT =
EQUIVALENT_UNDER_RESTRICTED_CONDITIONS
```

Exact equality holds in the formula emulator when:

```text
same alpha weights and sorted order
SeaFree d_i/10 and WaterSplatting d_i are numerically aligned
SeaFree per-Gaussian A/beta values equal WaterSplatting per-pixel A/beta values
same intrinsic c_i
SeaFree final clamp is inactive
```

Under those restrictions:

```text
max COMPOSITING_DISAGREEMENT = 1.4852072621515682e-16
max DIRECT_DISAGREEMENT      = 1.1709900946621996e-16
max MEDIUM_TOTAL_DISAGREEMENT = 1.2334378693694748e-16
```

### Inference

The pure alpha/backscatter ordering is not, by itself, a multi-depth non-equivalence under constant aligned medium. The WaterSplatting segment-plus-tail expression simplifies to the same total water term as SeaFree's per-Gaussian backscatter plus residual background:

```text
A * (1 - sum_i w_i exp(-beta_B d_i))
```

Actual code is still structurally different because SeaFree and WaterSplatting use different medium query domains, distance definitions, distance scaling, code organization, and final clamp behavior.

## J*T Audit

### Quantitative Result

The image-space operation:

```text
clear_object_fullsh_raw * transmission
```

does not reconstruct:

```text
direct_object_signal
```

in general.

Across micro-cases:

```text
mean direct discrepancy = 0.00821707315073128
p99 direct discrepancy  = 0.10843805605566893
max direct discrepancy  = 0.10843805605566893
```

Minimal counterexample:

```text
alpha1 = 0.5
alpha2 = 0.5
c1 = [1, 1, 1]
c2 = [0.2, 0.2, 0.2]
d1 = 1.0
d2 = 5.0
beta_D = [0.8, 0.8, 0.8]
A = [0, 0, 0]

degrade-then-composite direct = [0.22558026400304748] * 3
composite-then-degrade direct = [0.08505104550209014] * 3
mean abs difference = 0.14052921850095734
```

### Inference

Root cause classification:

```text
E. multiple causes
```

The direct signal uses alpha-weighted per-Gaussian `T_D(d_i)`. The image-space transmission uses a composited expected depth, and:

```text
sum_i w_i c_i exp(-beta_D d_i)
!=
(sum_i w_i c_i) exp(-beta_D * expected_depth)
```

unless restricted cases hold, such as one contributor, equal depths, beta near zero, or matching colors/depths that make the nonlinear average collapse.

## Micro-Case Results

### Quantitative Result

Single-Gaussian cases:

| Mode | n | mean disagreement | max disagreement |
| --- | ---: | ---: | ---: |
| scale_aligned_constant_medium | 27 | 4.774952519442779e-17 | 1.4852072621515682e-16 |
| source_native_distance_div10 | 27 | 0.18808622046666895 | 1.042904508295438 |

Two-Gaussian cases:

| Mode | n | mean disagreement | max disagreement |
| --- | ---: | ---: | ---: |
| scale_aligned_constant_medium | 9 | 3.7645237447894763e-17 | 1.3530055912156527e-16 |
| scale_aligned_per_gaussian_medium_mismatch | 9 | 0.02238364933255904 | 0.05428039034989093 |
| source_native_distance_div10 | 9 | 0.22355334718620512 | 0.4511612856574565 |

Three-Gaussian cases:

| Mode | n | mean disagreement | max disagreement |
| --- | ---: | ---: | ---: |
| scale_aligned_constant_medium | 3 | 6.724653257448383e-17 | 1.0154786906321955e-16 |
| scale_aligned_per_gaussian_medium_mismatch | 3 | 0.015447927794852434 | 0.019657101565241 |
| source_native_distance_div10 | 3 | 0.25775251250627956 | 0.34853627994050307 |

## Sensitivity

### Quantitative Result

| Metric | Mode | Amplification |
| --- | --- | ---: |
| BRIGHT_AMPLIFICATION | scale_aligned_per_gaussian_medium_mismatch | 1.4461739762570527 |
| BRIGHT_AMPLIFICATION | source_native_distance_div10 | 1.3466040448179417 |
| DEPTH_SEPARATION_AMPLIFICATION | scale_aligned_per_gaussian_medium_mismatch | 1.8684191282442368 |
| DEPTH_SEPARATION_AMPLIFICATION | source_native_distance_div10 | 1.2647298658371906 |
| OPACITY_AMPLIFICATION | scale_aligned_per_gaussian_medium_mismatch | 2.5722899997904154 |
| OPACITY_AMPLIFICATION | source_native_distance_div10 | 9.068060441546832 |

Flags:

```text
BRIGHT_SENSITIVE_GAP = FALSE
DEPTH_SENSITIVE_GAP = TRUE
OPACITY_SENSITIVE_GAP = TRUE
```

Ordering sensitivity:

```text
scale_aligned_constant_medium ORDERING_SEMANTICS_GAP = 1.3877787807814457e-17
scale_aligned_per_gaussian_medium_mismatch ORDERING_SEMANTICS_GAP = 0.008035673718312222
source_native_distance_div10 ORDERING_SEMANTICS_GAP = 0.0017234867557605754
```

AA interaction was a controlled opacity-compensation test, not an actual AA renderer call. It is recorded in:

```text
outputs/dcomp_audit_20260810/aa_opacity_interaction.csv
```

## Panama Region Alignment

### Experimental Fact

Existing checkpoints were loaded read-only at nominal step 15000, actual step 14999:

```text
Panama M1 classic
Panama BND-K1 classic
Panama BND-AA antialiased
```

Eval views:

```text
MTN_1539
MTN_1529
MTN_1547
```

Mask definitions:

```text
M1_J_gt_1       = M1 clear_object_fullsh_raw max RGB channel > 1.0
GT_brightness_Q5 = top 20 percent GT luminance within each eval view
J_or_brightness = union of the two masks
overlap_proxy = M1 accumulation * M1 depth_std_relative
```

`overlap_proxy` is a proxy only, not a contributor count. In this run `depth_std_relative` was zero in the audited outputs, so the depth-variation and overlap-proxy enrichment rows are non-informative.

### Quantitative Result

Aggregate enrichments:

| Mask | Proxy | Enrichment |
| --- | --- | ---: |
| M1_J_gt_1 | alpha | 1.0327101402888543 |
| M1_J_gt_1 | edge | 3.3735248948225887 |
| M1_J_gt_1 | depth_variation | 0.0 |
| M1_J_gt_1 | overlap_proxy | 0.0 |
| GT_brightness_Q5 | alpha | 0.9674601545992759 |
| GT_brightness_Q5 | edge | 3.111406557056645 |
| GT_brightness_Q5 | depth_variation | 0.0 |
| GT_brightness_Q5 | overlap_proxy | 0.0 |
| J_or_brightness | alpha | 0.9678361793588195 |
| J_or_brightness | edge | 3.096329762911468 |
| J_or_brightness | depth_variation | 0.0 |
| J_or_brightness | overlap_proxy | 0.0 |

Residual enrichments:

| Mask | K1 residual enrichment | AA residual enrichment |
| --- | ---: | ---: |
| M1_J_gt_1 | 3.3716475249718836 | 3.1898943860709035 |
| GT_brightness_Q5 | 2.5352029205931905 | 2.4060955239793196 |
| J_or_brightness | 2.5545429354780915 | 2.4234489433168154 |

Alignment flag:

```text
PANAMA_FAILURE_REGION_ALIGNED = FALSE
```

Reason: each tested Panama mask had only one enriched structural proxy above 1.25 (`edge`). The predefined rule required at least two of overlap/depth/opacity/edge-style proxies.

## Fixed-State Counterfactual

### Experimental Fact

```text
fixed_state_counterfactual = SKIPPED
COUNTERFACTUAL_ALIGNMENT_VALID = FALSE
```

Reason: a reliable fixed-state SeaFree-order counterfactual would require exact alignment of per-Gaussian medium query, LOS distance, projection footprint, opacity compensation, and background semantics. This audit did not modify renderer code to add that path.

## Gradient Sensitivity

### Experimental Fact

```text
gradient_sensitivity = NOT_REQUIRED
```

Reason: the optional gradient audit required a valid fixed-state counterfactual, which was not established.

## Closure

### Quantitative Result

Formula closure:

```text
SeaFree formula reference mean/p99/max closure error = 0
WaterSplatting formula reference mean/p99/max closure error = 0
```

Detailed closure table:

```text
outputs/dcomp_audit_20260810/closure_audit.csv
```

## Final Flags

### Quantitative Result

```text
FORWARD_ORGANIZATION_DIFFERENT = TRUE
MULTI_GAUSSIAN_NON_EQUIVALENCE = FALSE
BRIGHT_SENSITIVE_GAP = FALSE
DEPTH_SENSITIVE_GAP = TRUE
OPACITY_SENSITIVE_GAP = TRUE
PANAMA_FAILURE_REGION_ALIGNED = FALSE
```

## Hypothesis Assessment

### Quantitative Conclusion

Hypothesis:

```text
Differences in water-degradation and Gaussian alpha-compositing semantics contribute materially to the localized Panama RGB fitting deficit after bounding intrinsic appearance.
```

Assessment:

```text
PARTIALLY_SUPPORTED
```

Basis:

```text
Code organization and actual method semantics differ.
Aligned constant-medium compositing is equivalent to numerical precision.
Controlled non-aligned source-native distance and per-Gaussian medium cases produce non-negligible gaps.
The gap is depth-sensitive and opacity-sensitive under the predefined thresholds.
Brightness sensitivity did not reach the predefined 1.5 threshold.
Panama region alignment did not meet the predefined two-proxy rule.
Fixed-state counterfactual was not valid in this stage.
```

### Reasonable Inference

The current evidence does not support a claim that pure degradation/compositing ordering alone explains the Panama bounded-intrinsic RGB deficit. It does support a narrower mechanism: actual SeaFree and WaterSplatting semantics differ through distance scaling/definition and per-Gaussian-vs-per-pixel medium ownership, and those controlled differences can be depth/opacity sensitive.

### Unverified Hypothesis

Whether a fixed WaterSplatting checkpoint would move closer to or farther from GT under a precisely aligned SeaFree-order forward counterfactual remains unresolved.

## Next Single-Factor Recommendation

### Proposed Controlled Experiment

Only one next step is recommended:

```text
Read-only fixed-checkpoint SeaFree-order counterfactual alignment diagnostic
```

This should remain a diagnostic before any new training. It should first establish exact alignment for:

```text
same Gaussian opacity
same projected footprint
same distance convention
same A/beta query
same background semantics
```

If alignment cannot be made exact without changing renderer semantics, the counterfactual should remain skipped.

## Visual Assets

### Experimental Fact

Micro-case sheets:

```text
renders/dcomp_audit_20260810/single_gaussian_comparison.png
renders/dcomp_audit_20260810/two_gaussian_depth_ordering.png
renders/dcomp_audit_20260810/front_bright.png
renders/dcomp_audit_20260810/back_bright.png
renders/dcomp_audit_20260810/high_opacity_overlap.png
renders/dcomp_audit_20260810/low_opacity_overlap.png
```

Panama region alignment:

```text
renders/dcomp_audit_20260810/panama_region_alignment.png
```

Index:

```text
renders/dcomp_audit_20260810/VISUAL_COMPARE_INDEX.md
```

Manifests:

```text
outputs/dcomp_audit_20260810/manifest.json
outputs/dcomp_audit_20260810/manifest.csv
```

Visual assets are ready for external/manual analysis.
No subjective clear-image correctness judgment was made.
