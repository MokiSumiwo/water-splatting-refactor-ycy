# BND Pure-Water Background Anchor Readiness

## Motivation
INFERENCE: This read-only audit asks whether Panama contains enough reliable background/water-only candidate observations to anchor the existing asymptotic water prediction in BND-K1.

## Why BND-Aware Refinement Is Not The Next Primary Direction
EXPERIMENTAL FACT: BND-AWARE-REFINE was formally INCONCLUSIVE. RH showed a weak +0.021 dB signal over R0, but the final population mismatch exceeded the preregistered 2 percent tolerance.
INFERENCE: The main research direction therefore returns to object-medium identifiability rather than further spatial refinement heuristics.

## Why Medium Identifiability Is Revisited
HYPOTHESIS: A directly observable background/asymptotic water channel could reduce object-medium compensation freedom without changing renderer physics.

## SeaFree Background-Water Supervision Mechanism
CODE FACT: SeaFree-GS reference commit `7797e97dae831029ac89ae9f37b3c3d69ec2cf6c` with status ``.
CODE FACT: SeaFree background supervision compares pixel ambient-light `water_background_image` with GT underwater image on pseudo-depth background pixels, using inverse ambient-light weights and coefficient `0.01`.
CODE FACT: SeaFree's pseudo-depth mask is cached per image/downscale and does not carry gradient to pseudo-depth.

## WaterSplatting Asymptotic-Water Semantics
CODE FACT: In current WaterSplatting, `b_inf_mode=tied` makes `B_inf = medium_rgb`; `medium_rgb` is sigmoid-activated channels 0:3 from the shared 9-channel medium MLP.
CODE FACT: Formal K1 uses `medium_context_mode=dir_xy_camera`, `b_inf_mode=tied`, and `infinite_water_enabled=False`; the clean branch rejects enabling infinite-water.
CODE FACT: The tied recomposition replaces the renderer default tail with `tail_weight * b_inf` where `tail_weight = final_transmittance * exp(-medium_bs * last_depth)`.

## B_inf / Medium_Rgb / Infinite-Water Source Audit
CODE FACT: `medium_rgb`, `medium_bs`, and `medium_attn` are channel slices of one 9-channel medium MLP output. Only `medium_rgb` is directly equivalent to the tied `B_inf` tensor.
CODE FACT: Current native infinite-water diagnostic is NOT_EVALUABLE because the active clean branch raises on `infinite_water_enabled=True`.

## WaterSplatting-SeaFree Semantic Mapping
CODE FACT: SeaFree `A` maps most closely to WaterSplatting `b_inf / medium_rgb`, but this is NOT EXACTLY EQUIVALENT because WaterSplatting uses custom finite-medium integration plus tied tail recomposition.
CODE FACT: SeaFree `beta_D` maps to WaterSplatting `medium_attn`; SeaFree `beta_B` maps to `medium_bs`.

## Pseudo-Depth Source
CONFIG FACT: Pseudo-depth source is `undistorted_data/undistorted_Panama/depthAnything_u16`, the same cache used in the formal Panama CDEPTH diagnostics.
CONFIG FACT: Pseudo-depth is used offline only for mask construction. `PSEUDO_DEPTH_GRADIENT_TO_GEOMETRY = FALSE`.

## SeaFree Background-Mask Semantics
CODE FACT: SeaFree normalizes pseudo-depth by per-image max, thresholds at `1e-2`, inverts the thresholded image to choose `pseudo_depth >= 1e-2`, fills the largest external contour as foreground, and uses its complement as background.

## Candidate-Mask Definitions
CONFIG FACT: `M_SF` is the SeaFree-style pseudo-depth background candidate; `M_LOW_SUPPORT = accumulation_3k <= 0.01`; `M_INTERSECT = M_SF & M_LOW_SUPPORT`; `M_SAFE = BinaryErode(M_INTERSECT, radius=5 px)`.
CONFIG FACT: The definition was locked before held-out processing in `locked_pure_water_candidate_definition.json`.

## Training / Held-Out Split
CONFIG FACT: Training-development views are MTN_1538, MTN_1541, MTN_1540, MTN_1534, MTN_1535, MTN_1536, MTN_1533, MTN_1542, MTN_1537, MTN_1532, MTN_1546, MTN_1543, MTN_1544, MTN_1545, MTN_1548.
CONFIG FACT: Held-out views are MTN_1529, MTN_1539, MTN_1547.

## Leakage Controls
EXPERIMENTAL FACT: `locked_pure_water_candidate_definition.json` was written before held-out metrics were interpreted. `HELD_OUT_MASK_SELECTION_LEAKAGE = FALSE`.

## Mask Coverage
QUANTITATIVE RESULT: training pooled M_SAFE coverage = `0.0014541265330494672`; views >=1% = `0`.
QUANTITATIVE RESULT: held-out pooled M_SAFE coverage = `0.0030137337472527816`; views >=1% = `0`.
QUANTITATIVE CONCLUSION: `SAFE_MASK_COVERAGE_ADEQUATE = False`.

## Mask Agreement
EXPERIMENTAL FACT: Cross-mask agreement rows are stored in `mask_agreement.csv/json`; high agreement is not treated as proof of true water.

## Temporal Support Stability
QUANTITATIVE RESULT: final pooled train late contamination fraction = `1.0`.
QUANTITATIVE CONCLUSION: `SUPPORT_STABLE_SAFE_MASK = False`.

## Late Object-Contamination Proxy
QUANTITATIVE CONCLUSION: `LOW_OBJECT_SUPPORT_CONFIRMED = False` and `OBJECT_CONTAMINATION_WARNING = True`.

## K1 B_inf Extraction
CODE FACT: Current K1 B_inf is extracted from `outputs['b_inf']` at final checkpoint step 14999, with BND-K1@3k used only for locked low-support mask construction.

## B_inf Error
QUANTITATIVE RESULT: train M_SAFE E_BINF = `0.010516314767301083`, E_full = `0.004834328778088093`, R_anchor = `0.5403020083499479`.
QUANTITATIVE RESULT: held-out M_SAFE E_BINF = `0.010305261239409447`, E_full = `0.007992291823029518`, R_anchor = `0.22444549076879836`.

## Full-Render vs B_inf Error
INFERENCE: Positive `R_anchor` is compensation-compatible evidence, but it does not identify the responsible component and does not make the mask a pure-water ground truth.
QUANTITATIVE CONCLUSION: `BACKGROUND_ANCHOR_HEADROOM = False` and `HELDOUT_BG_HEADROOM_CONSISTENT = False`.

## Anchor-Headroom Gap
EXPERIMENTAL FACT: Headroom tables are stored in `background_anchor_headroom.csv/json`, `binf_statistics.csv/json`, and `full_render_vs_binf.csv/json`.

## Object Contribution On Candidate Water Pixels
EXPERIMENTAL FACT: `object_contamination_audit.csv/json` records accumulation, direct-object signal magnitude, and medium signal magnitude on M_SAFE.

## Brightness Confound
EXPERIMENTAL FACT: Brightness/context diagnostics and matched-control rows are stored in `brightness_matched_control.csv/json`; these are diagnostics only and not method masks.

## Held-Out Validation
EXPERIMENTAL FACT: Held-out metrics are stored in `heldout_water_candidate_metrics.csv/json` and `heldout_binf_headroom.csv/json`.

## Virtual Background-Loss Gradient Route
QUANTITATIVE RESULT: medium_mlp grad L2 = `0.15965156587578883`; max object/appearance grad L2 = `0.0`.
QUANTITATIVE CONCLUSION: `MEDIUM_ONLY_GRADIENT_ROUTE = True`.
CODE FACT: Output-level gradient rows show nonzero `dL/dA_b_inf` and zero `dL/dbeta_B`, `dL/dbeta_D` for the virtual `|B_inf-GT|` loss.

## Native Infinite-Water Diagnostic
EXPERIMENTAL FACT: Native infinite-water diagnostic is `NOT_EVALUABLE` for this branch because the code path is disabled and not activated in K1.

## Water-Candidate Classification
QUANTITATIVE CONCLUSION: Water candidate classification = `WATER_CANDIDATE_WEAK`.

## Background-Anchor Readiness Classification
QUANTITATIVE CONCLUSION: Background anchor classification = `BG_ANCHOR_NOT_SUPPORTED`.

## Scientific Interpretation
INFERENCE: The audit treats masks only as conservative background/water-only candidates, not ground-truth pure-water labels.
INFERENCE: Under the locked definition, M_SAFE has nontrivial B_inf disagreement but insufficient stable low-object support and insufficient coverage for a training-ready anchor.

## Next Single-Factor Decision
INFERENCE: Next single-factor decision = `close background-anchor direction for this mask definition`.

## Safety
EXPERIMENTAL FACT: AUDIT_PARAMETER_SAFETY = `PASS`; CHECKPOINT_SAFETY = `PASS`.

## Outputs
EXPERIMENTAL FACT: Output directory `outputs/bnd_pure_water_audit_panama_20260812`.
EXPERIMENTAL FACT: Render directory `renders/bnd_pure_water_audit_panama_20260812`.
