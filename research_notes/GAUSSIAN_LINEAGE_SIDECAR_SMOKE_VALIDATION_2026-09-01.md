# Gaussian Lineage Sidecar Smoke Validation (2026-09-01)

## Scope

CONFIG FACT: this was one 1000-step IUI3-RedSea smoke run (steps 0-999), using the existing C0 configuration with OCMC on and RAOC off. It was not a formal 15K experiment.

CODE FACT: lineage is an external CPU sidecar. It observes existing split/duplicate/prune masks and never enters the model state_dict, renderer, loss, optimizer, gradients, or refinement selection.

## Topology Coverage

QUANTITATIVE RESULT: initial=21907, ever-created=138763, current=47215. The smoke recorded 58386 split-parent events in 3 batches, 84 duplicate events in 2 batches, and 3 non-empty prune calls.

EXPERIMENTAL FACT: each child has a unique ID, birth iteration, parent ID, event type, and generation depth. Pruned IDs remain in the registry while current slot IDs are masked with the exact prune mask.

## Reload Compatibility

EXPERIMENTAL FACT: checkpoint/sidecar reload passed: TRUE. Gaussian count, current IDs, parent relations, birth iterations, generation depths, and topology events matched exactly. The frozen prediction max-absolute difference was 0.

EXPERIMENTAL FACT: installing the sidecar wrappers changed neither the model state_dict hash nor a frozen prediction; installation-time prediction max-absolute difference was 0.

## Frozen Support By Lineage

The table uses the existing proxy: the number of distinct frozen training cameras where model.radii > 0 at step 999. It does not use heldout cameras to compute support.

| Lineage group | current N | mean support | low-support fraction | share of all low-support | rho vs heldout residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial | 5 | 18.600 | 0.000000 | 0.000000 | undefined |
| child_generation_1 | 300 | 17.317 | 0.000000 | 0.000000 | undefined |
| child_generation_2_plus | 46910 | 16.230 | 0.000938 | 1.000000 | 0.775 |

QUANTITATIVE RESULT: children are 0.999894 of the final population and 1.000000 of the final low-support population. Children born within the last 200 steps are 0.994472 of the population and 1.000000 of low-support Gaussians.

INFERENCE: the recent-child representation ratio in low support is only 1.0056x because recent children already dominate the whole smoke population. The 100% child share therefore does not provide a discriminating lineage enrichment test.

## Scientific Limits

INFERENCE: the smoke can describe whether newborn groups are enriched for low support at step 999, but it cannot establish what survives to 15K. The protocol forbids a new formal 15K run.

QUANTITATIVE RESULT: lineage-controlled residual correlation is NOT_ESTIMABLE_WITH_4_HELDOUT_CAMERAS_AND_3_LINEAGE_GROUPS. Per-group rho values are small-N diagnostics only.

## Decision

FINAL CLASSIFICATION: LINEAGE_INCONCLUSIVE.

MODULE DESIGN AUTHORIZED: FALSE.

RECOMMENDATION: use the validated sidecar in a bounded longer diagnostic with adequate heldout support before module design. Do not modify OCMC and do not reopen RAOC.
