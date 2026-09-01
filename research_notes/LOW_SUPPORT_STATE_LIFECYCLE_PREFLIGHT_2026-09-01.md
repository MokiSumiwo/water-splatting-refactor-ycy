# Low-Support State Lifecycle Preflight (2026-09-01)

## Objective

HYPOTHESIS: a distinct-camera support state may lose a unique meaning when Gaussian identity changes through split, duplicate, and prune.

This preflight does not re-test the existence of the low-support association and does not design a module or loss.

## Frozen Inputs

CONFIG FACT: only the four C0 branches under outputs/m1_raoc_causal_four_scene_20260827 were used at 5K, 8K, 10K, 13K, and 14999. C0 has OCMC on and RAOC off.

EXPERIMENTAL FACT: every operation was checkpoint loading, frozen rendering, or offline CSV/JSON analysis. No optimizer step, backward call, checkpoint write, or training was performed.

## Lineage Availability

CODE FACT: all 20 checkpoint state_dict objects omit lineage, birth iteration, and parent/source identifiers. The 576 C0 refinement records contain event counts but no Gaussian indices. Model loading also explicitly discards a legacy gaussian_lineage_ids key.

QUANTITATIVE RESULT: GAUSSIAN_LINEAGE_UNAVAILABLE. Parent-child matching by array index or geometry proximity was not attempted.

## Population Evolution

| Scene | step | Gaussian count | fraction s<=1 | mean s | median s | baseline T1 rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Curasao | 5000 | 812907 | 0.044081 | 12.148 | 14.0 | 0.000 |
| Curasao | 8000 | 1045867 | 0.050959 | 12.136 | 14.0 | 1.000 |
| Curasao | 10000 | 1031481 | 0.054664 | 11.952 | 14.0 | 1.000 |
| Curasao | 13000 | 1009656 | 0.054713 | 11.953 | 14.0 | 0.500 |
| Curasao | 14999 | 1006921 | 0.054584 | 11.957 | 14.0 | 0.500 |
| IUI3-RedSea | 5000 | 575248 | 0.074881 | 11.737 | 10.0 | 0.800 |
| IUI3-RedSea | 8000 | 770478 | 0.106015 | 10.890 | 8.0 | 0.800 |
| IUI3-RedSea | 10000 | 782874 | 0.111063 | 10.697 | 8.0 | 0.800 |
| IUI3-RedSea | 13000 | 757644 | 0.110925 | 10.803 | 8.0 | 0.800 |
| IUI3-RedSea | 14999 | 752689 | 0.110673 | 10.826 | 8.0 | 0.800 |
| JapaneseGradens-RedSea | 5000 | 644887 | 0.084959 | 9.624 | 8.0 | 0.866 |
| JapaneseGradens-RedSea | 8000 | 840041 | 0.097483 | 9.532 | 8.0 | 0.866 |
| JapaneseGradens-RedSea | 10000 | 838661 | 0.101117 | 9.366 | 7.0 | 0.866 |
| JapaneseGradens-RedSea | 13000 | 811855 | 0.100343 | 9.450 | 7.0 | 0.866 |
| JapaneseGradens-RedSea | 14999 | 807042 | 0.099994 | 9.472 | 7.0 | 0.866 |
| Panama | 5000 | 915076 | 0.072763 | 8.235 | 8.0 | -1.000 |
| Panama | 8000 | 1107459 | 0.101118 | 7.862 | 8.0 | -1.000 |
| Panama | 10000 | 1124580 | 0.102145 | 7.808 | 8.0 | -0.500 |
| Panama | 13000 | 1100290 | 0.102198 | 7.841 | 8.0 | -0.500 |
| Panama | 14999 | 1096443 | 0.102107 | 7.847 | 8.0 | -0.500 |

Baseline rho is reported only as frozen-checkpoint provenance. It is not an inheritance-strategy result. These C0 checkpoints use their original 2026-08-27 split with 3/4/3/3 heldout cameras for Curasao/IUI3-RedSea/JapaneseGradens-RedSea/Panama, so the small-N rho values must not be mixed with the later 2026-08-31 resplit evidence.

## Topology Events

CODE FACT: split appends sampled children and then culls original split parents; duplicate appends parameter copies while retaining sources; prune masks all Gaussian parameter arrays.

EXPERIMENTAL FACT: split and duplicate stop after 10K in all four recorded C0 branches, while pruning continues. The last child addition is at step 9900 in every scene, so every final survivor is at least 5099 iterations past its latest possible creation. Event support transitions remain NOT_AVAILABLE because selected/pruned identities were not persisted.

## Strategy Sensitivity

| Strategy | alpha | median final reference low fraction | median whole-population reference rho | actual child result |
| --- | ---: | ---: | ---: | --- |
| A_INHERIT | - | 0.101051 | 0.650 | NOT_IDENTIFIABLE_WITHOUT_PARENT_CHILD_LINEAGE |
| B_RESET | - | 1.000000 | undefined | NOT_IDENTIFIABLE_WITHOUT_PARENT_CHILD_LINEAGE |
| C_FRACTIONAL | 0.25 | 0.315983 | 0.750 | NOT_IDENTIFIABLE_WITHOUT_PARENT_CHILD_LINEAGE |
| C_FRACTIONAL | 0.5 | 0.209145 | 0.750 | NOT_IDENTIFIABLE_WITHOUT_PARENT_CHILD_LINEAGE |
| C_FRACTIONAL | 0.75 | 0.101051 | 0.650 | NOT_IDENTIFIABLE_WITHOUT_PARENT_CHILD_LINEAGE |

QUANTITATIVE RESULT: the fractions and reference rho values apply each rule to the complete checkpoint population as transparent limiting cases. They are not estimates of actual split-child outcomes because the selected parent population is unknown. Reset makes every visible Gaussian low-support, so its reference predictor is constant and rho is undefined. No alpha was selected.

INFERENCE: inherit preserves a parent's camera set but can overstate coverage of a displaced, smaller split child. Reset preserves literal post-birth observation history but discards inherited parameter evidence. Fractional inheritance is not an integer distinct-camera count for many parent supports and silently redefines the statistic.

## Temporal Identity

QUANTITATIVE RESULT: persistent-low, late-created-low, and gradually-growing identity fractions are NOT_AVAILABLE. Independent checkpoint distributions cannot establish Gaussian identity continuity.

EXPERIMENTAL FACT: low-support populations remain measurable through the 10K-to-14999 culling-only phase. This rules out only a post-10K newborn explanation; it establishes neither persistent identity nor whether final low-support Gaussians came from pre-10K split/duplicate.

## Age Confounding

QUANTITATIVE RESULT: AGE_CONTROL_NOT_AVAILABLE. Creation iteration and lineage metadata are absent, so age confounding cannot be excluded.

## Lifecycle Semantics

INFERENCE: prune can preserve state semantics for surviving indices through exact masking. Split and duplicate do not have one empirically validated state rule in the locked artifacts; each candidate answers a different question about inherited versus post-birth evidence.

## Final Classification

FINAL DECISION: LOW_SUPPORT_STATE_LIFECYCLE_AMBIGUOUS.

MODULE DESIGN AUTHORIZED: FALSE.

RECOMMENDATION: DO_NOT_DESIGN_LOW_SUPPORT_AWARE_MODULE; next run INSTRUMENT-GAUSSIAN-LINEAGE-SIDECAR-SMOKE-VALIDATION.

The low-support association remains a diagnostic finding, not a claim that Gaussian geometry is wrong and not a causal mechanism.
