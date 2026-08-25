# M1-CAMERA-CONTEXT-IDENTIFIABILITY-AUDIT

## CODE FACT
The current 3-D camera context is `(camera.camera_to_worlds[0,:3,3] - scene_center) / (scene_scale + 1e-6) * medium_camera_context_scale`.
It is not a learned latent, is not trainable, and is constructed from each camera center at train/eval time.
C0 keeps the 22-D medium MLP input but sets only the final camera-context feature to zero.

## CONFIG FACT
Both arms use BND as a controlled bounded intrinsic-color parameterization: `bounded_sh3`, SH degree 3, `dir_xy_camera`, `b_inf_mode=tied`, `infinite_water_enabled=False`.
Only intervention: C0 neutral camera context vs C1 formal scene-normalized camera-center context.

## EXPERIMENTAL FACT
Parameter/optimizer/RNG equivalence: `True`, `True`, `True`.
Camera sequence match: `True`.
Outputs: `/mnt/new/home_old/ycy/water-splatting-refactor/outputs/m1_camera_context_identifiability_iui3_20260825`.

## QUANTITATIVE RESULT
Final eval C1-C0: PSNR `0.186090` dB, SSIM `0.003313`, LPIPS `-0.000750`, MSE `-0.00006704`.
Final eval PSNR positive views: `2/4`.
Correct-context swap utility final fraction positive: `0.66913818359375`.
M_SAFE weak projection over 1/9 random reference by step: `{5000: 2.1984930367001265, 10000: 2.8677584071252467, 14999: 2.8107161276314896}`.
M_SAFE weak/orth RGB sensitivity by step: `{5000: 0.010703520627108451, 10000: 0.016512079023460247, 14999: 0.03028191776181891}`.
M_SAFE weak-removal delta_E by step: `{5000: 6.574706847395361e-08, 10000: 1.41253056327173e-07, 14999: 8.134895669975605e-08}`.

## INFERENCE
Expressiveness classification: `CAMERA_CONTEXT_EXPRESSIVENESS_SUPPORTED`.
Ambiguity classification: `CAMERA_CONTEXT_AMBIGUITY_SUPPORTED`.
Combined tradeoff classification: `CAMERA_CONDITIONED_IDENTIFIABILITY_TRADEOFF_SUPPORTED`.
Phase B entered: `True`; Phase-B classification: `CAMERA_OBSERVABILITY_MODULE_READY`.
No true-medium, true-color, or true-geometry claim is made.

## HYPOTHESIS
Next single formal experiment: formal matched causal training: M1 camera-conditioned baseline vs M1 camera-conditioned + OCMC on IUI3

## RECOVERED HISTORICAL M1 EVIDENCE
{'medium_field_append_context': '72927e7 Refactor WaterSplatting and add M1-M4 ablations', 'formal_clean_bnd_baseline': '62294d6 Build clean M1 bounded-intrinsic baseline', 'previous_medctx_audit': '4f9cffa Audit bounded medium context utilization'}
