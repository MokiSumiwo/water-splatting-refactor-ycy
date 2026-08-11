# BND Hard-Region Observability Audit

## Motivation

CODE FACT: This is a read-only diagnostic audit. No training, optimizer step, scheduler step, checkpoint mutation, densification, split, duplicate, prune, opacity reset, new loss, or new model was executed.

INFERENCE: CDEPTH is closed for the current study. The active question is bounded representation capacity under `bounded_sh3`, not another CDEPTH variant.

## Repository State

- Branch: `research/m1-bounded-intrinsic`
- Start HEAD: `91b6d56266bdd25cab3ea16324cd937b80a9f016`
- Initial status: `?? scripts/diagnostics/audit_bnd_hardness_observability.py
?? scripts/diagnostics/render_gmvc_curasao_contact_sheet.py
?? scripts/diagnostics/summarize_gmvc_four_scene_visual_audit.py`

## Checkpoints And Camera Split

CONFIG FACT: BND-K1 checkpoints were audited at nominal 1k, 3k, 5k, 8k, 10k, 13k, and 15k. M1 final was used only for offline labels/context.

| Run | Nominal | Actual | Exists |
| --- | ---: | ---: | --- |
| M1 | 15000 | 14999 | True |
| BND-K1 | 1000 | 1000 | True |
| BND-K1 | 3000 | 3000 | True |
| BND-K1 | 5000 | 5000 | True |
| BND-K1 | 8000 | 8000 | True |
| BND-K1 | 10000 | 10000 | True |
| BND-K1 | 13000 | 13000 | True |
| BND-K1 | 15000 | 14999 | True |

CONFIG FACT: Training-development views: `MTN_1538;MTN_1541;MTN_1540;MTN_1534;MTN_1535;MTN_1536;MTN_1533;MTN_1542;MTN_1537;MTN_1532;MTN_1546;MTN_1543;MTN_1544;MTN_1545;MTN_1548`.
CONFIG FACT: Held-out eval views: `MTN_1529;MTN_1539;MTN_1547`.
CONFIG FACT: `HELD_OUT_SELECTION_LEAKAGE = FALSE`; signal selection and proxy locking used training views only.

## Offline Labels

- `M1_HIGH_J`: M1 final accumulation > 0.01 and max RGB of M1 final `clear_object_fullsh_raw` > 1.0. Oracle diagnostic label only.
- `PERSISTENT_BND_HARD`: K1 late residual top 10% inside final K1 object support for at least 75% of available late checkpoints. Future-outcome diagnostic label only.
- `BND_HARD_CORE`: `PERSISTENT_BND_HARD AND M1_HIGH_J`. Oracle plus future diagnostic label only.

QUANTITATIVE RESULT: train pooled prevalence M1_HIGH_J `0.051671`, PERSISTENT_BND_HARD `0.064975`, BND_HARD_CORE `0.023231`.
QUANTITATIVE RESULT: held-out pooled prevalence M1_HIGH_J `0.050461`, PERSISTENT_BND_HARD `0.083991`, BND_HARD_CORE `0.021428`.

## Signal Availability

| Signal | Semantics | Availability | Deployability | Compute cost |
| --- | --- | --- | --- | --- |
| S_RES_CURRENT | current K1 RGB residual MSE rank inside final K1 object support | AVAILABLE | ONLINE_TRAINING_DEPLOYABLE | LOW_COST_ONLINE |
| S_RES_PERSIST | mean of current and past per-view residual percentile ranks | AVAILABLE | ONLINE_TRAINING_DEPLOYABLE | MODERATE_ONLINE |
| S_SH | mean RGB abs difference between bounded full-SH clear render and bounded DC-only clear render | AVAILABLE | TRAINING_DEPLOYABLE | MODERATE_ONLINE |
| S_BOUND | max RGB channel of current bounded full-SH clear_object_fullsh_raw | AVAILABLE | ONLINE_TRAINING_DEPLOYABLE | LOW_COST_ONLINE |
| S_RESP | L2 RGB norm of d formal K1 RGB loss / d pred_image | AVAILABLE | ONLINE_TRAINING_DEPLOYABLE | MODERATE_ONLINE |
| S_REFINE | projected-center 16px-window proxy from fixed 15-view RGB-only densification trigger score | AVAILABLE | TRAINING_BANK_DEPLOYABLE | PERIODIC_TRAIN_BANK |

## Training-Bank Prediction Results

| Signal | Score | 3k ENRICH@10 | 3k AP_LIFT | 5k ENRICH@10 | Early predictive |
| --- | --- | ---: | ---: | ---: | --- |
| S_RES_CURRENT | STRONG_TRAIN | 4.675482726521482 | 5.125478426618128 | 4.896409560400884 | True |
| S_RES_PERSIST | STRONG_TRAIN | 4.791926892100266 | 5.378548556003807 | 4.998553818533688 | True |
| S_SH | WEAK_TRAIN | 1.1245635828025442 | 1.0523681969755445 | 2.4206442587164014 | False |
| S_BOUND | STRONG_TRAIN | 3.0263839115080113 | 3.0602895098454947 | 4.309805416328204 | True |
| S_RESP | EVIDENCE_AGAINST | 0.35068935201890894 | 0.6497037106347558 | 0.34132127671698376 | False |
| S_REFINE | EVIDENCE_AGAINST | 0.34320259024705707 | 0.8156298589891867 | 0.5362369061371114 | False |

## Locked Proxy

CONFIG FACT: Locked proxy definition: `{'COMPOSITE_AVAILABLE': True, 'proxy_type': 'COMPOSITE', 'signal_a': 'S_RES_PERSIST', 'signal_b': 'S_BOUND', 'formula': '0.5 * percentile_rank(S_RES_PERSIST) + 0.5 * percentile_rank(S_BOUND)', 'selection_stage': 'training views only, K1@3k, PERSISTENT_BND_HARD target', 'selection_used_heldout': False}`.
CONFIG FACT: Held-out views were not used for signal selection, formula choice, direction choice, or threshold tuning.

## Held-Out Evaluation

QUANTITATIVE RESULT: `HELDOUT_CROSS_VIEW_CONSISTENT = True`.
QUANTITATIVE RESULT: held-out views with ENRICH@10 > 1: `3`; >=1.5: `3`; >=2: `3`.

For the locked proxy at K1@3k:

| View | Label | ENRICH@10 | Precision@10 | Recall@10 | AP_LIFT |
| --- | --- | ---: | ---: | ---: | ---: |
| MTN_1529 | PERSISTENT_BND_HARD | 3.343 | 0.281 | 0.334 | 3.179 |
| MTN_1539 | PERSISTENT_BND_HARD | 2.742 | 0.223 | 0.274 | 2.508 |
| MTN_1547 | PERSISTENT_BND_HARD | 4.796 | 0.416 | 0.480 | 5.190 |
| ALL | PERSISTENT_BND_HARD | 3.650 | 0.307 | 0.365 | 3.531 |
| ALL | M1_HIGH_J | 7.024 | 0.354 | 0.702 | 9.267 |
| ALL | BND_HARD_CORE | 7.733 | 0.166 | 0.773 | 17.492 |

## Deployability

CONFIG FACT: `DEPLOYABLE_PROXY_AVAILABLE = True`.
CONFIG FACT: compute cost class `MODERATE_ONLINE`.
CONFIG FACT: uses M1 `False`, uses future K1 `False`, uses eval GT for training trigger `False`, uses CDEPTH `False`.
CONFIG FACT: selected proxy requires training GT `True`, residual-rank history `True`, backward `False`, extra render `False`, full training-bank pass `False`.
QUANTITATIVE RESULT: `PROXY_SPECIFICITY_WEAK = True` from the brightness-control comparison.
QUANTITATIVE RESULT: brightness control at K1@3k for PERSISTENT_BND_HARD had train ENRICH@10 `5.671` / AP_LIFT `6.998`, versus locked proxy `4.369` / `4.566`.
QUANTITATIVE RESULT: brightness control at K1@3k for held-out PERSISTENT_BND_HARD had ENRICH@10 `4.258` / AP_LIFT `4.622`, versus locked proxy `3.650` / `3.531`.
INFERENCE: The locked proxy passes the formal deployable-proxy gates, but the simple brightness control is a strong confound. This does not automatically invalidate the proxy; it limits the specificity claim and should be controlled in the next stage.

## Formal Classification

QUANTITATIVE RESULT: `DEPLOYABLE_HARDNESS_PROXY_STRONG`.
QUANTITATIVE RESULT: semantic alignment `BOUND_SPECIFIC_HARDNESS_PROXY`.

## Scientific Interpretation

INFERENCE: A deployable signal is only considered suitable for the next refinement stage if it predicts late persistent K1 error on held-out views and does not depend on M1 oracle labels or future K1 labels.
QUANTITATIVE RESULT: At 3k, the strongest individual training signals for PERSISTENT_BND_HARD were `S_RES_PERSIST` (ENRICH@10 `4.792`, AP_LIFT `5.379`), `S_RES_CURRENT` (`4.675`, `5.125`), and `S_BOUND` (`3.026`, `3.060`).
QUANTITATIVE RESULT: `S_RESP` and `S_REFINE` were evidence-against under the training scorecard at 3k, with ENRICH@10 `0.351` and `0.343` respectively.
QUANTITATIVE RESULT: `S_RES_CURRENT` and `S_RES_PERSIST` were strongly redundant at 3k (Spearman rho `0.839`); `S_RES_PERSIST` and `S_BOUND` were not strongly redundant (rho `0.189`), so the equal-weight composite was allowed.
QUANTITATIVE RESULT: BND_HARD_CORE was not sparse in pooled train (`0.023231`) or held-out (`0.021428`) domains, and no per-view BND_HARD_CORE prevalence was below the 0.5% sparse threshold.
QUANTITATIVE RESULT: spatial-bias audit flagged 4/15 training views and 2/3 held-out views. The locked proxy top10 was entirely inside the final object-support domain by construction; mean bright-top20 fraction was `0.730` on train and `0.737` on held-out.
INFERENCE: Early bounded-model current-state observables can predict late persistent bounded-hard regions, but part of the predictiveness overlaps with image brightness. The deployable proxy should therefore be treated as a strong hardness proxy with a brightness-specificity warning, not as proof of a brightness-independent bounded-capacity mechanism.

## Next Single-Factor Decision

PROPOSED NEXT STEP: BND-AWARE-REFINE single-factor proxy-guided refinement causal test.
CONFIG FACT: The next experiment must use only the locked deployable proxy or an explicitly brightness-controlled variant defined before training. It must not use `M1_HIGH_J`, `PERSISTENT_BND_HARD`, `BND_HARD_CORE`, held-out GT, CDEPTH, AA, or any reopened unbounded intrinsic appearance as a training signal.

## Safety And Artifacts

CODE FACT: `AUDIT_PARAMETER_SAFETY = True`; all tracked model parameters had zero max absolute delta after diagnostic forwards/backwards and temporary DC-only rendering restoration.
CODE FACT: `CHECKPOINT_SAFETY = True`; the audited checkpoint paths had unchanged size and mtime fingerprints after the audit.
CODE FACT: PNG visual assets were generated under `renders/bnd_hardness_panama_20260811/` and verified openable.
CODE FACT: Output metrics and manifests were generated under `outputs/bnd_hardness_panama_20260811/`.

Primary visual assets:

- `renders/bnd_hardness_panama_20260811/plot_label_prevalence_summary.png`
- `renders/bnd_hardness_panama_20260811/contact_sheet_1k_signal_maps.png`
- `renders/bnd_hardness_panama_20260811/contact_sheet_3k_signal_maps.png`
- `renders/bnd_hardness_panama_20260811/contact_sheet_5k_signal_maps.png`
- `renders/bnd_hardness_panama_20260811/contact_sheet_persistent_hard_label_maps.png`
- `renders/bnd_hardness_panama_20260811/contact_sheet_bnd_hard_core_maps.png`
- `renders/bnd_hardness_panama_20260811/plot_signal_enrich10_trajectories.png`
- `renders/bnd_hardness_panama_20260811/plot_signal_aplift_trajectories.png`
- `renders/bnd_hardness_panama_20260811/training_view_scorecard_sheet.png`
- `renders/bnd_hardness_panama_20260811/plot_signal_spearman_matrix.png`
- `renders/bnd_hardness_panama_20260811/contact_sheet_locked_proxy_maps.png`
- `renders/bnd_hardness_panama_20260811/heldout_MTN_1529_comparison.png`
- `renders/bnd_hardness_panama_20260811/heldout_MTN_1539_comparison.png`
- `renders/bnd_hardness_panama_20260811/heldout_MTN_1547_comparison.png`
- `renders/bnd_hardness_panama_20260811/contact_sheet_selected_proxy_top10_vs_labels.png`
- `renders/bnd_hardness_panama_20260811/plot_brightness_control_comparison.png`
- `renders/bnd_hardness_panama_20260811/temporal_prediction_summary_sheet.png`
- `renders/bnd_hardness_panama_20260811/final_deployability_next_step_summary_sheet.png`

Primary output files:

- `outputs/bnd_hardness_panama_20260811/bnd_hardness_final_summary.json`
- `outputs/bnd_hardness_panama_20260811/hardness_proxy_classification.json`
- `outputs/bnd_hardness_panama_20260811/proxy_specificity_audit.json`
- `outputs/bnd_hardness_panama_20260811/training_view_scorecard.csv`
- `outputs/bnd_hardness_panama_20260811/heldout_proxy_metrics.csv`
- `outputs/bnd_hardness_panama_20260811/brightness_control.csv`
- `outputs/bnd_hardness_panama_20260811/spatial_bias_audit.csv`
- `outputs/bnd_hardness_panama_20260811/checkpoint_safety.json`
- `outputs/bnd_hardness_panama_20260811/parameter_safety.json`

Visual assets are ready for external/manual analysis.
No subjective clear-image correctness judgment was made.
