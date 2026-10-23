# Transmission-Gated Gain Support/Benefit Audit

This analysis uses the completed v2 gated run and performs no training. It
compares the detached gate's training coverage with regional error reduction on
official test views. Results are in
`outputs/bnd_transmission_gated_gain_20260914_v2/gain_overlap_audit.csv` and
`gain_overlap_summary.json`.

Training eligible-pixel fractions averaged 0.2706 for JapaneseGradens, 0.7947
for Panama, 0.7372 for Curasao and 0.3668 for IUI3. Thus the negative result in
Panama and identity fallback in IUI3 cannot be attributed simply to too few
eligible samples.

For the selected spatial test outputs, JapaneseGradens and IUI3 are exact BND
identity, so their mean pixel improvement is zero. Panama has only two of nine
view/seed records with positive global MSE change and mean improvement
`-1.32e-6`; its boundary and overflow regions are negative on average even
though its gate covers most training pixels. Curasao has positive mean global
improvement `2.11e-5` and six of nine positive records. Its boundary region
improves on average `3.68e-5`, high-transmission `1.06e-4`, and overflow
`3.11e-4`, while low-transmission and interior regions worsen slightly.

The evidence suggests that transmission gating is correlated with where a
correction can help in Curasao, but is not a sufficient criterion for selecting
correctable pixels across scenes. In particular, high support and high
transmission can still represent view-dependent or overflow errors, as seen in
Panama. The gate should therefore not be expanded or used to justify teacher
distillation. A next diagnostic, if pursued, should estimate per-region
cross-view agreement of the required gain before fitting any new parameters;
otherwise direct-gain refinement should be frozen.
