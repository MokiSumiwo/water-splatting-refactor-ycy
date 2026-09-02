# Identifiability Module Causal Experiment

Date: 2026-09-02
Classification: `IDENTIFIABILITY_MODULE_NOT_SUPPORTED`

## Hypothesis And Arms

C0 is the frozen-OCMC baseline. C1 differs only by the detached SH-opacity tangent regularizer with strength 1.0. Both arms restore the same step-3000 model, optimizer, scheduler, scaler, RNG plan, OCMC projector, and camera sequence, then run 11,999 updates through step 14,999.

Training is fixed to seed 42, `bounded_sh3`, SH degree 3, `dir_xy_camera`, tied `B_inf`, classic rasterization, OCMC on, RAOC off, and five retained arm checkpoints at steps 5000, 8000, 10000, 13000, and 14999. No sweep was run.

## Mechanism

Visible training cameras define one detached non-DC SH direction per Gaussian that is aligned with the raw-opacity RGB tangent. The anchored scalar penalty acts only on `features_rest`; DC, opacity, geometry, medium, OCMC, and topology receive no direct module gradient. No GT or heldout view constructs the controller.

## Four-Scene Results

| Scene | train dPSNR | heldout dPSNR | heldout dSSIM | heldout dLPIPS | view+ | shared rel | overlap delta | temporal shared/overlap | non-DC/orth | opacity | count gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Curasao | 0.098333 | -0.045276 | -0.000012 | 0.000199 | 0.333 | -0.983627 | -0.000690 | 5/5 | 0.779/0.869 | 1.002937 | 0.000929 |
| IUI3-RedSea | -0.051881 | 0.090387 | 0.000146 | -0.000192 | 0.750 | -0.998155 | 0.000240 | 5/4 | 1.027/1.274 | 1.001862 | 0.002416 |
| JapaneseGradens-RedSea | -0.043542 | 0.027695 | 0.000018 | 0.000473 | 0.667 | -0.974517 | -0.000109 | 5/2 | 0.670/0.736 | 1.005392 | 0.003311 |
| Panama | 0.019816 | 0.072294 | 0.000673 | 0.000332 | 1.000 | -0.958631 | 0.000161 | 5/2 | 1.160/0.941 | 1.003369 | 0.001956 |

## Causal Matching And OCMC Independence

All-scene causal validity: `true`. OCMC state/config independence: `true`. Direct-gradient audit: `true`.

## Capacity, Opacity, Topology, And Decomposition Safety

No global SH collapse: `false`. Opacity stable: `true`. Gaussian population normal: `true`. Decomposition safety: `true`.

Final `P(J > 1)` is exactly zero for train and heldout in both arms of every scene. Final maximum `J_p99` values by scene (C0/C1 over train and heldout) are: Curasao 0.979036/0.977841; IUI3-RedSea 0.876031/0.871583; JapaneseGradens-RedSea 0.966258/0.969045; Panama 0.822406/0.817901.

Final eval OCMC projected-raw RMS (C0/C1) is: Curasao 0.066574/0.072381; IUI3-RedSea 0.028510/0.030618; JapaneseGradens-RedSea 0.016670/0.016967; Panama 0.068928/0.068138. The projector and OCMC configuration hashes are identical, and both arm projectors remain unchanged.

## Counterfactual Diagnostic

Removing the sampled training-anchored shared component at the final checkpoints changes mean heldout PSNR by Curasao C0 -0.000247 dB, C1 -0.000012 dB; IUI3-RedSea C0 -0.000257 dB, C1 0.000001 dB; JapaneseGradens-RedSea C0 -0.000085 dB, C1 -0.000012 dB; Panama C0 -0.000156 dB, C1 0.000040 dB. Mean orthogonal relative drift is below 1.3e-7 in every arm, confirming that this read-only diagnostic preserves the sampled orthogonal component.

## RGB And Mechanism Classification

Heldout PSNR improved or tied in 3/4 scenes; mean delta was 0.036275 dB. Mechanism classification: `NOT_SUPPORTED`. RGB classification: `SUPPORTED`.

Shared response energy decreased at the final checkpoint in 4/4 scenes, final tangent overlap decreased in 2/4, and both mechanism metrics were temporally stable in 1/4. JapaneseGradens-RedSea failed the registered SH capacity floor: non-DC and orthogonal C1/C0 ratios were 0.669652 and 0.735867, below 0.75.

## Required Answers

1. Same start state: yes, exactly, in 4/4 scenes.
2. Camera sequence: exact match with zero mismatches in 4/4 scenes.
3. Completion: all four scenes completed 11,999 updates per arm.
4. Shared energy: decreased in 4/4 scenes.
5. Tangent overlap: decreased at final in 2/4 scenes; temporal mechanism stability held in 1/4.
6. SH collapse: registered capacity preservation failed in JapaneseGradens-RedSea.
7. Opacity: stable in 4/4 scenes.
8. Gaussian population: normal in 4/4 scenes; final relative gaps are all below 0.34%.
9. Train PSNR deltas (C1-C0): Curasao +0.098333, IUI3-RedSea -0.051881, JapaneseGradens-RedSea -0.043542, Panama +0.019816 dB.
10. Heldout PSNR deltas (C1-C0): Curasao -0.045276, IUI3-RedSea +0.090387, JapaneseGradens-RedSea +0.027695, Panama +0.072294 dB.
11. Heldout PSNR improved or tied in 3/4 scenes.
12. Mean heldout PSNR delta: +0.036275 dB.
13. Decomposition safety: preserved; final `P(J > 1) = 0` for every arm/split/scene.
14. OCMC independence: passed; projector state and configuration stayed frozen and identical.
15. Mechanism SUPPORT: NOT_SUPPORTED.
16. RGB SUPPORT: SUPPORTED.
17. Final module classification: `IDENTIFIABILITY_MODULE_NOT_SUPPORTED`.
18. Next unique task: `CLOSE_IDENTIFIABILITY_MODULE_RESEARCH_LINE`.

## Limitations

The controller and diagnostics are local first-order tests of representation redundancy. They do not establish that SH is true radiance, that opacity is true geometry, or that PSNR implies physical correctness. This was one fixed-strength experiment with no sweep.

## Final Classification

The final module classification is `IDENTIFIABILITY_MODULE_NOT_SUPPORTED`. Next task: `CLOSE_IDENTIFIABILITY_MODULE_RESEARCH_LINE`.
