# TABA-DCU BND-ON Directional Chromatic Experiment

Date: 2026-09-10. Repository: `/mnt/new/home_old/ycy/water-splatting-refactor`.
Branch: `research/bvam-bnd-on`; existing uncommitted research assets retained.
Output root: `outputs/taba_dcu_bnd_on_20260910`.

## Motivation

Previous scalar-chroma TABA E2 improved four-scene mean PSNR by 0.029097 dB
but worsened mean LPIPS and MSE, with mild losses on JapaneseGradens and
Panama. DCU tests whether attenuating only a renderer-derived water direction
preserves more useful appearance capacity. The previous experiment and all
its checkpoints, logs and negative findings are preserved.

## Registered Implementation

For every projected Gaussian, RGB transmission and the object/medium RGB
components are sampled at its rounded screen center from the existing
detached BND rasterizer prepass. Its alpha-normalized expected depth is the
same color-independent depth used in the final render. Existing
`sample_taba_transmission` computes `tau=clamp(medium_attn*depth,min=0)` and
`t=exp(-tau)` once; DCU reuses these sampled tensors. The real-checkpoint
audit verifies bitwise equality with final-render transmission samples.
There is no previous-frame cache or learned proxy. The water axis uses the
ungated BND prepass, making its source deterministic without circular
dependence on the adjusted appearance. No rasterizer physics is changed.

Actual integration point is after bounded-SH color evaluation and medium
activation, before the final underwater rasterizer call. For N Gaussians,
`features_dc` is N x 3 and degree-3 `features_rest` is N x 15 x 3; full/base
logits, sampled RGB transmission and water axes are N x 3. Gate, validity,
reliability and axis strength are N x 1. The renderer's depth/accumulation
are H x W x 1, and attenuation/object/medium RGB are H x W x 3. Object RGB
is the attenuated direct contribution; medium RGB is the integrated water
contribution including the existing tail. They are not intrinsic-color
ground truths.

```text
delta = s_full - s_base
chroma = delta - mean_RGB(delta)
T_mean = mean_RGB(t)
T_std = sqrt(var_RGB(t, correction=0))
T_cv = clamp(T_std / (T_mean + 1e-8), 0, 1)
q_T = clamp(T_mean * (1 - uncertainty_lambda*T_cv), 0, 1)
w_raw = detach(rgb_medium - rgb_object)
w_strength = norm(w_raw)
w = w_raw / (w_strength + 1e-8)
parallel = dot(chroma, w) * w
orthogonal = chroma - parallel
gate = floor + (1-floor)*sigmoid((q_T-mid)/temperature)
s_adjusted = s_full + (detach(gate)-1)*parallel
rgb = existing_safe_sigmoid(s_adjusted)
```

The population variance reduces the three RGB channels at each Gaussian
sample; it is channel dispersion, not an estimated posterior uncertainty.
Directional mode without uncertainty uses `q_T=T_mean`. Chromatic mode uses
the original full-chroma allocation. An invalid or weak water axis
(`w_strength<1e-4`) falls back to the original chromatic gate based on
`T_mean`, without the uncertainty correction. Invalid depth, attenuation,
transmission, projection or empty visibility uses `q_T=gate=1`; negative
depth/attenuation are invalid. Nonfinite auxiliary data is sanitized.

The requested axis `rgb_medium-rgb_object` is not necessarily zero-mean.
Following the supplied formula exactly can therefore change mean logits,
even though the input residual is chromatic. The implementation does not
silently recenter the axis, and logs `luminance_logit_change_abs`. Normalizing
with norm+epsilon makes the parallel/orthogonal relation accurate up to the
specified epsilon, not exact unit-vector orthogonality. Both properties are
covered by tests and must be considered when interpreting results.
The real-checkpoint audit measures maximum mean-logit changes of approximately
0.0087-0.0130 in JapaneseGradens and 0.0199-0.0303 in Panama for directional
presets at the audited source state. Chromatic mode differs only by about
6.4e-7 roundoff. Evidence is in `engineering/directional_luminance_audit.json`.

Only `rgb` and `s_adjusted` retain appearance gradients. All other helper
outputs, including residual decomposition diagnostics, are detached. The
main photometric renderer still has its existing medium/geometry gradients;
the auxiliary gate/axis does not add any. CICA targets, calibration, losses,
optimizer groups and the OCMC projector remain unchanged.

`taba_enabled=False` or `taba_mode=disabled` uses the original BND path.
Defaults are false/disabled. The old runner explicitly selects chromatic
mode to preserve reproduction of the previous training presets.
The existing safe sigmoid clips floating-point endpoints for active modes.
Invalid samples preserve original logits and safe-BND RGB; ordinary logits
match the original sigmoid exactly, while extreme rounded 0/1 endpoints are
kept inside (0,1). Disabled baseline retains its original rounded endpoints.

## Registered Matrix

| Arm | Mode | Mid | Temperature | Floor | Uncertainty lambda |
|---|---|---:|---:|---:|---:|
| E0 | disabled | .50 | .15 | .20 | .50 |
| E1 | chromatic | .50 | .15 | .20 | .50 (unused) |
| E2 | directional | .50 | .15 | .20 | 0 |
| E3 | directional_uncertainty | .50 | .15 | .20 | .50 |
| E4 | directional_uncertainty | .40 | .20 | .30 | .50 |

The request calls E1 the previous E2 but explicitly gives .50/.15/.20;
the explicit matrix governs. It numerically matches previous TABA-A/E1,
not previous TABA-B/E2 (.40/.20/.30).

All arms: M1+OCMC+CICA, BND ON, Auxiliary/RAOC/BRASH/BVAM/BARR/BHTC/BGTR OFF,
seed 42, registered source at step 3000, final step 14999. CICA starts at
10000 with original refresh at 10000/12000/14000. Checkpoints and all native
evaluation PNGs at 5000/10000/14999 are retained. Only the existing
`water_splatting` conda environment and physical GPUs 6,7,8,9 are used,
with no more than one training worker per GPU.

JapaneseGradens and Panama use the strict-v4 registered BND-ON sources.
The sources load strictly without regeneration. Source audit and original
BND ON/OFF pairing evidence are saved in `source_asset_audit.json`.
All 10 two-scene arms are trained from these original sources, not from
earlier TABA final states.

Selection, registered before results: choose the stable enabled E1-E4 with
the highest two-scene mean final PSNR, after all 10 runs finish. Report all
metrics and retain all negative results. Apply one selected configuration
to both additional scenes, alongside new E0 runs; original two-scene runs
form the other half of the four-scene validation. Selection on test scenes
is exploratory and does not establish an independently validated optimum.

## Engineering Checks Before Training

- 39 pytest tests pass, including directional projection, uncertainty,
  missing/invalid axes, invalid transmission, negative depth/attenuation,
  empty samples, strict active RGB bounds and detached diagnostic gradients.
- Both original scenes pass real-checkpoint audits for E1-E4: strict source
  loading; historical disabled forward hash; exact disabled/diagnostic-only
  forward and loss; DC/rest gradient flow; no auxiliary gradient escape;
  raw J and intrinsic bounds; CICA/OCMC routing; unchanged optimizer groups.
- Sampled prepass transmission exactly matches final-render samples.
- Explicit `enabled=True, mode=disabled` also matches baseline forward and
  loss exactly in both real scenes. Evidence is retained separately in
  `engineering/explicit_disabled_mode/<scene>/audit.json`.
- Existing CUDA audit has unchanged pre-existing strict tolerance failures;
  current forward/backward errors equal the committed reference exactly.
- `py_compile` and `git diff --check` pass before formal training.

Evidence: `<out>/engineering/<scene>/audit.json`,
`<out>/engineering/cuda_nonregression.json`, and `<out>/logs/`.

### Repeat-run limitation

The new E0 shares exactly the old experiment's source model, optimizer,
scheduler, projector, initial RNG manifest, first camera and first-step loss
in both original scenes. Subsequent training logs and Gaussian counts differ
by step 3500. The existing CUDA rasterizer uses floating-point atomic gradient
accumulation (`water_splatting/cuda/csrc/backward.cu`), so a fixed seed and
identical initial state do not guarantee bitwise repeatable optimization.
The new detached diagnostics also change allocations/timing. The precise
contribution of each cause is not isolated here. This does not violate the
verified disabled single-forward/loss equivalence, but prevents treating
cross-experiment metric differences as causal DCU gains. Comparisons use the
newly trained E0 controls, and small single-seed differences remain exploratory.

The existing pre-training gradient audit performs a synthetic step-5000
forward at the source state, which writes the first `taba_training.jsonl`
record. Formal continuation records then start at step 3500 (first 500-step
logging boundary after step 3001). The initial audit record is not a trained
5000-step checkpoint. Snapshot CSVs, evaluation metrics and PNGs use the
actual saved checkpoint states; none are computed from the synthetic audit.

## Results

### Complete two-scene screen

Final checkpoint: step 14999. PSNR and SSIM are higher-is-better; LPIPS and
MSE are lower-is-better. Scene means are unweighted means of test-view
metrics; aggregate means weight scenes equally. PSNR is averaged in dB,
so its ranking need not equal the ranking of arithmetic mean MSE.

| Scene | Arm | PSNR | SSIM | LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| JapaneseGradens-RedSea | E0 | 24.894173 | 0.895967 | 0.133397 | 0.005024503 |
| JapaneseGradens-RedSea | E1 | 24.876078 | 0.895863 | 0.133574 | 0.005159377 |
| JapaneseGradens-RedSea | E2 | 24.929468 | 0.895643 | 0.133711 | 0.004923892 |
| JapaneseGradens-RedSea | E3 | 24.833061 | 0.895839 | 0.133906 | 0.005090833 |
| JapaneseGradens-RedSea | E4 | 24.850044 | 0.896194 | 0.133561 | 0.005124603 |
| Panama | E0 | 31.539353 | 0.947331 | 0.103464 | 0.000706837 |
| Panama | E1 | 31.525684 | 0.947391 | 0.103775 | 0.000709049 |
| Panama | E2 | 31.540917 | 0.947421 | 0.103598 | 0.000706889 |
| Panama | E3 | 31.490575 | 0.947130 | 0.104035 | 0.000714380 |
| Panama | E4 | 31.522141 | 0.947353 | 0.103858 | 0.000709746 |
| Two-scene mean | E0 | 28.216763 | 0.921649 | 0.118430 | 0.002865670 |
| Two-scene mean | E1 | 28.200881 | 0.921627 | 0.118675 | 0.002934213 |
| Two-scene mean | E2 | 28.235192 | 0.921532 | 0.118654 | 0.002815390 |
| Two-scene mean | E3 | 28.161818 | 0.921485 | 0.118970 | 0.002902607 |
| Two-scene mean | E4 | 28.186092 | 0.921773 | 0.118709 | 0.002917175 |

All E1-E4 pass the registered engineering screen. E2 wins the registered
mean-PSNR criterion and was frozen before launching either additional scene
in `four_scene_selection.json`. It uses directional allocation with
mid=.50, temperature=.15, floor=.20, lambda=0, axis epsilon=1e-4. The full
uncertainty versions E3/E4 do not win this screen.

Relative to the newly trained E0, E2 changes mean PSNR by +0.018429 dB,
SSIM by -0.000117, LPIPS by +0.000224 and MSE by -0.000050280. JapaneseGradens
gains +0.035295 dB and improves MSE, but worsens SSIM and LPIPS. Panama gains
only +0.001564 dB, with slightly higher SSIM and slightly worse MSE/LPIPS.
This is a small mixed result, not an established broad improvement. E3/E4
reduce mean PSNR by 0.054945/0.030670 dB respectively. Every negative result
and checkpoint is retained.

Per-view E2-minus-E0 PSNR explains the limits of the means: JapaneseGradens
MTN_1090 gains 0.116402 dB, while MTN_1098/MTN_1106 lose 0.009851/0.000666 dB.
Panama MTN_1529/MTN_1539 gain 0.028782/0.032406 dB, while MTN_1547 loses
0.056496 dB. IUI3-RedSea MTN_5903/MTN_5911 gain 0.041691/0.028364 dB, while
MTN_5894/MTN_5928 lose 0.012953/0.015717 dB. Local improvements do not extend
uniformly over the test cameras; per-view SSIM/LPIPS sometimes disagree
with PSNR as well. Exact values are retained in each `per_view_metrics.csv`.

The frozen complete screen, including all requested statistics, is in
`<out>/two_scene_report.md` and `<out>/two_scene_summary.json`.

### Four-scene extension

All 14 runs completed, each with 11999 updates from step 3000 to 14999.
Curasao and IUI3-RedSea E0/E2 used the same registered per-scene step-3000
sources. JapaneseGradens and Panama reuse the completed screen's E0/E2
runs; no scene chooses a different configuration.

| Scene | Arm | PSNR | SSIM | LPIPS | MSE |
|---|---|---:|---:|---:|---:|
| Curasao | E0 | 32.453103 | 0.959035 | 0.125685 | 0.000730089 |
| Curasao | E2 | 32.508619 | 0.959008 | 0.125743 | 0.000713908 |
| IUI3-RedSea | E0 | 30.710550 | 0.908051 | 0.199617 | 0.001662743 |
| IUI3-RedSea | E2 | 30.720896 | 0.908462 | 0.199614 | 0.001664953 |
| JapaneseGradens-RedSea | E0 | 24.894173 | 0.895967 | 0.133397 | 0.005024503 |
| JapaneseGradens-RedSea | E2 | 24.929468 | 0.895643 | 0.133711 | 0.004923892 |
| Panama | E0 | 31.539353 | 0.947331 | 0.103464 | 0.000706837 |
| Panama | E2 | 31.540917 | 0.947421 | 0.103598 | 0.000706889 |
| Four-scene mean | E0 | 29.899295 | 0.927596 | 0.140541 | 0.002031043 |
| Four-scene mean | E2 | 29.924975 | 0.927634 | 0.140667 | 0.002002410 |

E2-minus-E0 four-scene changes are PSNR +0.025680 dB, SSIM +0.000038,
LPIPS +0.000126 (worse), and MSE -0.000028633 (better). Curasao's PSNR
gain is +0.055515 dB, with lower MSE and slightly worse SSIM/LPIPS. Its
MTN_1288/MTN_1296/MTN_1304 PSNR changes are +0.137028/+0.012794/+0.016724 dB.
IUI3-RedSea gains +0.010346 dB and SSIM +0.000412; LPIPS changes by only
-0.000002522, and MSE increases by 0.000002209.

Thus the positive PSNR direction is retained on Curasao and IUI3-RedSea,
but not the previous chromatic E2's larger +0.097521/+0.084582 dB gains.
These magnitudes come from separate matched experiments with different
trajectories; the difference is descriptive, not a controlled ablation
between the old and new E2. The new E2 is directional, whereas the old E2
was full-chroma gating with different thresholds.

### Diagnostics and bounds

All requested per-scene/per-arm statistics, including quantiles, are in
`<out>/report.md` and `diagnostics_all_steps.csv`. Selected E2 at step 14999:

| Scene | T mean | T std | q_T mean | Gate mean | Axis strength | Fallback ratio | J p99 | tau_D p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JapaneseGradens-RedSea | 0.855401 | 0.119642 | 0.857253 | 0.900185 | 0.394081 | 0 | 0.849517 | 0.863842 |
| Panama | 0.738470 | 0.111967 | 0.744969 | 0.845248 | 0.397076 | 0 | 0.823375 | 0.868652 |
| Curasao | 0.538862 | 0.141157 | 0.549036 | 0.653366 | 0.195613 | 0 | 0.742575 | 1.806389 |
| IUI3-RedSea | 0.709840 | 0.171269 | 0.719641 | 0.806982 | 0.334584 | 0 | 0.868758 | 1.985698 |

T summaries exclude invalid visible samples; q_T includes their unit
fallback, explaining why their means differ even when lambda=0. The
directional fallback ratio measures weak/invalid water axes only; invalid
transmission is separately counted. E2 invalid gate fractions are
1.8734%, 2.6195%, 2.2160%, 3.1781% in table order. No weak-axis fallback was
needed on these final evaluation views, but unit tests exercise that path.

| Scene | Applied chroma norm | Parallel norm | Orthogonal norm | Object residual | Medium residual | Clear RGB range |
|---|---:|---:|---:|---:|---:|---|
| JapaneseGradens-RedSea | 0.092410 | 0.009061 | 0.091572 | 0.111704 | 0.034291 | [0, 0.499974936] |
| Panama | 0.130649 | 0.011091 | 0.129963 | 0.058892 | 0.013746 | [0.005217694, 0.499974996] |
| Curasao | 0.133356 | 0.036792 | 0.130451 | 0.091880 | 0.016211 | [0, 0.499974996] |
| IUI3-RedSea | 0.119030 | 0.016692 | 0.116954 | 0.241208 | 0.018675 | [0, 0.499974996] |

Parallel/orthogonal norms describe the ungated residual; applied chroma is
recentered from the adjusted residual for diagnostics. Object residual is
mean |rgb_object-GT|, and medium residual is mean |rgb_medium-(GT-rgb_object)|.
They do not use separately observed component ground truths. The small
parallel component in JapaneseGradens/Panama helps explain the modest
mechanical effect, but it is not proof of causal performance attribution.

Every logged training record and all three evaluation snapshots for all
14 runs have `P(J>1)=P(intrinsic_rgb>1)=0`. Across evaluated active arms,
intrinsic RGB spans [4.5043098e-9, 0.9999998808], strictly inside (0,1).
Raw J is audited before any display clamp; it is nonnegative and bounded
above by one, with zero permitted in uncovered pixels. Native clear uses
the existing J/(1+J) mapping. E0 preserves the original sigmoid endpoint
rounding and can reach exactly one; this is baseline equivalence, not a
strict-open-interval claim about E0.

### BND-OFF reference and recommendation

| Four-scene quantity | PSNR | SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| User-supplied BND OFF | 30.031764 | 0.927403 | 0.137838 | 0.002015746 |
| New E2 | 29.924975 | 0.927634 | 0.140667 | 0.002002410 |
| E2 minus reference | -0.106789 | +0.000231 | +0.002829 | -0.000013336 |

The PSNR gap remains 0.106789 dB, and LPIPS is worse. The slightly higher
SSIM and lower MSE do not establish superiority. The supplied reference is
not a freshly paired control and its LPIPS protocol may differ; only this
run's E0 comparison supports direct within-experiment conclusions.

Keep TABA-DCU as an optional research implementation and retain all E0-E4
assets, including negative uncertainty results. Do not promote it to the
default mainline configuration. The strongest tested variant is directional
E2 without uncertainty, and even that produces small mixed single-seed
gains without a consistent underwater or dewatered visual improvement.
The full uncertainty correction has no demonstrated benefit here. Defaults
remain `taba_enabled=False`, `taba_mode="disabled"`. A future claim of
robust gains would need independent scenes and repeated seeds, but those
experiments are outside this completed registered matrix.

### Completion audit

The automatic coordinator exited successfully. All 14 runs pass gradient
routing/integrity checks, all four scenes pass initial-state/camera-sequence
pairing, every OCMC projector is unchanged, and maximum recorded CICA
gradient escape is zero. No task was stopped for bad metrics.

`artifact_integrity_audit.json` verifies 42 checkpoint hashes, 528 native
RGB PNGs, 21 contact sheets, and 13 distinct exactly paired GT views. GT
pixels match the source images exactly across arms and steps; PNG sizes
match the source resolution. All 39 tests passed before training. The
historical CUDA strict-equivalence failures remain disclosed above, with
identical errors in the committed reference and current implementation.
The six frozen training-source hashes were rechecked after all runs and
match `code_snapshot/manifest.json`.
Final `py_compile` of the 14 relevant Python files and `git diff --check`
pass. Branch remains `research/bvam-bnd-on`; the worktree contains both
the preserved earlier uncommitted work and the DCU changes listed below.
The final worktree listing is also captured in `<out>/summary.json`.
All experiment/coordinator processes have exited and physical GPUs 6-9
are idle at completion.

## Runtime and Visual Inspection Protocol

The median of logged iteration times on the completed two-scene E2 runs is
about 0.1174 s versus E0 0.1060 s in JapaneseGradens, and 0.1694 s versus
0.1544 s in Panama. The approximately 10% overhead is a sparse training-log
comparison across runs/devices, not a controlled benchmark. The extra
detached rasterizer prepass and diagnostics have real cost.

Underwater, native clear and raw-J contact sheets are inspected together.
Native clear uses the unchanged `J/(1+J)` display transform; raw-J display
is saved separately. Only contact-sheet thumbnails are resized. Native PNGs
and the exactly paired local `metrics.py` evaluation inputs are not resized,
cropped, or recolored. There is no clear-image ground truth.
Metrics are computed before PNG quantization, directly on native floating
RGB tensors via local `metrics.py` PSNR/SSIM/VGG-LPIPS; MSE is the same paired
RGB squared error. Re-evaluating the 8-bit PNGs can differ slightly from the
reported float-tensor metrics.

The five-arm JapaneseGradens comparison shows mostly subtle differences and
retained foreground texture. Black background coverage gaps and blue/cyan
distance artifacts persist across all arms, including E2's modest PSNR gain.
No consistent visual removal of water color is established by these sheets.

Panama's five-arm underwater renders also differ only subtly. The native
clear and raw-J views preserve much of the foreground texture but retain
cyan/blue distant regions and black patches near the upper background.
The tiny E2 PSNR change does not establish a visible underwater or dewatered
improvement. These are clear/dewatered renderings, not recovered true colors.

Curasao's underwater E0/E2 sheets are nearly indistinguishable at the
comparison scale, retaining similar foreground detail. Both dewatered
variants have large black upper-background regions; E2 additionally shows
a conspicuous pale vertical streak at the far right of MTN_1288 and a
green/cyan streak near the upper left. Its PSNR gain does not resolve these
clear-rendering defects. IUI3-RedSea likewise retains similar foreground
texture, blue distant regions and black coverage gaps. No consistent
visual dewatering improvement is supported in either extension scene.

Inspected all-view E0/E2 sheets for both extension scenes in underwater,
native clear and raw-J display, both five-arm screen comparisons, and
the three four-scene overview sheets. Overviews use the first lexicographic
test view in each scene; full-view sheets remain available to prevent a
single selected view from hiding the failures above.

## Code and Artifact Index

Files changed for DCU (earlier BRASH/BVAM/TABA work in the dirty tree is
retained independently):

- `water_splatting/taba.py`: RGB sampling, directional helper and diagnostics.
- `water_splatting/water_splatting.py`: default-off configuration and integration.
- `scripts/experiments/run_direct_mdrr_cica_scene.py`: config and statistics.
- `scripts/experiments/run_taba_bnd_on.py`: explicit legacy chromatic mode and queue reuse.
- `scripts/experiments/run_taba_dcu_bnd_on.py`: registered E0-E4 workers.
- `scripts/experiments/extend_taba_dcu_when_ready.py`: frozen selection and extension.
- `scripts/experiments/summarize_taba_bnd_on.py`: parameterized collection.
- `scripts/experiments/summarize_taba_dcu_bnd_on.py`: DCU tables and four-scene sheets.
- `scripts/diagnostics/audit_taba_training.py`: reusable directional audit.
- `scripts/diagnostics/audit_taba_dcu_training.py`: E1-E4 real-checkpoint audits.
- `scripts/diagnostics/audit_taba_artifacts.py`: parameterized artifact validation.
- `scripts/diagnostics/audit_taba_dcu_artifacts.py`: complete DCU artifact validation.
- `tests/test_taba_dcu.py`: numerical, fallback and gradient tests.
- This research note.

The existing `water_splatting/fields/gaussian_appearance.py` provides the
reused bounded SH decomposition, allocation and safe sigmoid. Its frozen
content is included in the code snapshot; no new sigmoid or appearance
expert was added. Losses and renderer physics remain unchanged.

Checkpoints retain the existing research-runner dictionary structure:
`model`, `optimizers`, `schedulers`, `scalers`, `ocmc_bundle`, `raoc_state`,
`rng_manifest`, experiment/scene/arm/step and metadata. TABA adds no model
parameter or optimizer group; its configuration/diagnostics are metadata.
The existing `rng_manifest` is a digest record, not the full random state,
so these checkpoints are loadable/evaluable but do not promise bitwise
identical mid-run resume. The registered runs execute uninterrupted from
their shared step-3000 source with the paired RNG and camera sequence.

`<out>` denotes `outputs/taba_dcu_bnd_on_20260910`. `<scene>/<arm>` ranges
over E0-E4 for JapaneseGradens-RedSea/Panama and E0/E2 for Curasao/IUI3-RedSea.

| Asset | Path relative to `<out>` |
|---|---|
| Full checkpoints | `<scene>/<arm>/checkpoints/step-{000005000,000010000,000014999}.ckpt` |
| Native RGB PNGs | `renders/<arm>/<scene>/step_{05000,10000,14999}/<view>/A2/{underwater,clear,clear_raw_display_clamp01,gt}.png` |
| RGB metrics | `metrics_final.csv`, `metrics_all_steps.csv`, `<scene>/<arm>/evaluation_metrics.csv`, `<scene>/<arm>/per_view_metrics.csv` |
| Gradient audits | `<scene>/<arm>/gradient_routing_audit.json` |
| Training statistics | `<scene>/<arm>/taba_training.jsonl`, `<scene>/<arm>/taba_statistics_step_{05000,10000,14999}.csv` |
| All evaluation diagnostics | `diagnostics_all_steps.csv` |
| Contact sheets | `contact_sheets/` |
| Frozen training code | `code_snapshot/manifest.json` and its referenced files |
| Source/pairing evidence | `source_asset_audit.json`, `four_scene_source_audit.json`, `five_arm_start_pairing.json`, `summary.json` |
| Test evidence | `engineering/test_results.json`, `engineering/<scene>/audit.json`, `engineering/cuda_nonregression.json` |
| Complete report | `report.md`, `summary.json` |
| Complete artifact audit | `artifact_integrity_audit.json` |

Training and evaluation run only in the existing conda `water_splatting`
environment. Each worker sets one physical GPU from 6,7,8,9. A single-run
reproduction, using a fresh output directory and an available allowed GPU:

```bash
export PATH=/opt/anaconda3/bin:$PATH
export CUDA_VISIBLE_DEVICES=6,7,8,9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate water_splatting
python scripts/experiments/run_taba_dcu_bnd_on.py \
  --scene JapaneseGradens-RedSea --variant E2 --gpu 6 \
  --output-root outputs/taba_dcu_reproduction
```

For completed registered outputs, reporting and artifact checks are:

```bash
python scripts/experiments/summarize_taba_dcu_bnd_on.py
python scripts/diagnostics/audit_taba_dcu_artifacts.py
```

No install, commit, push, compact-output operation, checkpoint deletion,
image deletion or rollback of pre-existing work was performed.
