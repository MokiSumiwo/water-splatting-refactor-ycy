# BND-CDEPTH High-J Structural Signature Audit

## Motivation

**Code Fact**

- This stage is a read-only analysis of existing Panama BND-K1 and BND-CDEPTH checkpoints.
- No training, optimizer step, scheduler step, densification, pruning, or checkpoint modification was performed.
- The audit asks why some fixed M1_HIGH_J pixels are recovered by CDEPTH while other pixels in the same diagnostic region are not.

## CDEPTH As Partial Mitigation

**Experimental Fact**

- Prior CDEPTH final result: PSNR `31.753299` versus BND-K1 `31.498353`, delta `+0.254946 dB`.
- Prior fixed M1_HIGH_J local recovery was about `38%`; SSIM and LPIPS were worse than BND-K1.
- Prior pathway classification was `MIXED_GAUSSIAN_STRUCTURE`, with harmful pathway unresolved.

## Formal Region Definitions

**Code Fact**

- `M1_HIGH_J`: M1 object support (`accumulation > 0.01`) and M1 `clear_object_fullsh_raw` max RGB channel `> 1.0`.
- `GAIN(x) = mean_RGB((pred_K1-GT)^2) - mean_RGB((pred_CDEPTH-GT)^2)`.
- `HJ_GAIN = M1_HIGH_J and GAIN > 0`.
- `HJ_HARM = M1_HIGH_J and GAIN < 0`.
- `HJ_STRONG_GAIN`: top 25 percent of positive GAIN inside pooled M1_HIGH_J.
- `HJ_STRONG_HARM`: top 25 percent of negative GAIN magnitude inside pooled M1_HIGH_J.

## Region Sizes

**Quantitative Result**

- M1_HIGH_J pixel fraction: `0.050460969959577204`.
- HJ_GAIN fraction: `0.028471968211986783`.
- HJ_HARM fraction: `0.02198900174759042`.
- HJ_STRONG_GAIN fraction: `0.007118070209992215`.
- HJ_STRONG_HARM fraction: `0.005497406750888645`.

## Projected Footprint

**Code Fact**

- Availability: `PROXY_PROJECTED_CENTER_WINDOW`.
- Semantics: `mean projected Gaussian radius of centers inside a square window of radius 16px; proxy, not contribution-weighted footprint`.
- This is a projected-center window proxy, not contribution-weighted effective footprint.

**Quantitative Result**

- CDEPTH HJ_GAIN radius proxy median: `7.890625`.
- CDEPTH HJ_HARM radius proxy median: `7.755364894866943`.
- CDEPTH HJ_GAIN/HARM radius ratio: `1.0174408434634175`.
- CDEPTH strong gain/harm radius ratio: `1.097909083203512`.

## Local Gaussian Density

**Code Fact**

- Semantics: `projected Gaussian center count in a square window of radius 16px; proxy, not true contributor count`.

**Quantitative Result**

- CDEPTH HJ_GAIN density median: `285.0`.
- CDEPTH HJ_HARM density median: `300.0`.
- CDEPTH HJ_GAIN/HARM density ratio: `0.95`.
- CDEPTH strong gain/harm density ratio: `0.812`.

## Effective Contributors

**Code Fact**

- `CONTRIBUTOR_DIAGNOSTIC_AVAILABLE = False`.
- Existing renderer outputs do not expose per-pixel contributor IDs or normalized contribution weights.
- Raw contributor count and `N_eff = 1/sum_i p_i^2` are therefore not evaluated.

## Scale / Anisotropy

**Code Fact**

- `REGION_SCALE_ATTRIBUTION_NOT_EVALUABLE = True`.
- Cross-run Gaussian matching is invalid after densification/pruning, and true contribution weights are unavailable.
- Region-conditioned physical scale / anisotropy attribution is not reported.

## Alpha / Coverage

**Quantitative Result**

- CDEPTH-minus-K1 HJ_GAIN alpha delta median: `5.340576171875e-05`.
- CDEPTH-minus-K1 HJ_HARM alpha delta median: `4.5299530029296875e-05`.
- CDEPTH-minus-K1 strong gain/harm alpha delta ratio: `0.7959542656112577`.

## Gain Quantiles And Correlation

**Quantitative Result**

- Monotonic footprint association: `False`.
- Monotonic density association: `True`.
- Monotonic alpha association: `True`.
- Spearman(GAIN, CDEPTH radius proxy): `0.08798965458152523`.
- Spearman(GAIN, CDEPTH density proxy): `-0.07436819028745845`.
- Spearman(GAIN, delta alpha): `0.022894411252994653`.

## Cross-View Consistency

**Quantitative Result**

- FINER_FOOTPRINT cross-view: `False`.
- HIGHER_LOCAL_DENSITY cross-view: `False`.
- COVERAGE_REBALANCING cross-view: `True`.

## Controls

**Code Fact**

- Depth, tau, T, NEW_LOW_T, direct signal, and medium signal are reported as controls.
- Direct/medium control uses true renderer `direct_object_signal` and `rgb_medium`; it does not use `J*T` image approximation.

**Quantitative Result**

- Spearman(GAIN, delta tau): `0.0419213963528766`.
- Spearman(GAIN, delta T): `-0.01934942974563727`.
- NEW_LOW_T fraction in HJ_GAIN: `0.0025693674106150866`.
- NEW_LOW_T fraction in HJ_HARM: `0.0005260463804006577`.

## Structural Signature Scorecard

**Quantitative Result**

- `FINER_FOOTPRINT`: `EVIDENCE_AGAINST`.
- `HIGHER_LOCAL_DENSITY`: `EVIDENCE_AGAINST`.
- `COVERAGE_REBALANCING`: `WEAK`.
- `MORE_DISTRIBUTED_CONTRIBUTORS`: `NOT_EVALUABLE`.
- `ANISOTROPIC_SUPPORT_REORGANIZATION`: `NOT_EVALUABLE`.

## Beneficial Structural Signature

**Quantitative Conclusion**

- Beneficial structural signature: `NO_CLEAR_HIGHJ_STRUCTURAL_SIGNATURE`.
- Signature mode: `NONE`.

**Inference**

- The result is spatial association evidence, not a causal proof.
- No statement is made about geometric or clear-image physical correctness.

## Is It Merely Gaussian Count?

**Quantitative Conclusion**

- `IS_IT_JUST_MORE_GAUSSIANS = UNRESOLVED`.
- Reason: Final global Gaussian counts are nearly equal in the previous trajectory, so any density signature must be local rather than simple final count. This audit still uses projected-center density proxy rather than true contributors.

## Deployability

**Inference**

- `DEPLOYABLE_PROXY_AVAILABLE = FALSE`.
- Structural observables are current-state computable proxies, but HJ_GAIN/HARM and M1_HIGH_J are diagnostic/oracle regions; no deployable hard-region selector was validated in this stage.

## Next Single-Factor Experiment

**Hypothesis**

- Recommended next step: `training-dynamics trigger audit for densification candidate selection and CDEPTH eligibility changes`.

## Outputs

- Final summary: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_highj_structure_panama_20260811/highj_structure_final_summary.json`.
- Output manifest: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/bnd_cdepth_highj_structure_panama_20260811/manifest.json`.
- Visual manifest: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_highj_structure_panama_20260811/manifest.json`.
- Visual index: `/mnt/new/home_old/ycy/water-splatting-refactor/renders/bnd_cdepth_highj_structure_panama_20260811/VISUAL_COMPARE_INDEX.md`.

No subjective clear-image correctness judgment was made.
