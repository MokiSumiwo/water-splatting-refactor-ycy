# Transmission-Gated Gain Results

The follow-up used a detached support gate during adapter fitting and kept the
same frozen CRCC E0 BND-ON source. The complete artifact is
`outputs/bnd_transmission_gated_gain_20260914_v2/summary.json`; the earlier
`..._20260914` directory is retained as an invalid orchestration record because
its child jobs did not receive the `--gated` flag.

The gate required accumulation >= 0.5, mean transmission in [0.35, 0.85], and
direct RGB support >= 1e-3. Pixels outside it used exact BND output. Native
support agreement remained below `6e-7`, clear hashes and frozen source hashes
were preserved, and all 13 official test views were rendered.

The validation gate failed again. Spatial gain validation macro PSNR was
36.594592 versus 36.534873 for B0 (+0.059719 dB), with LPIPS slightly lower,
but only Curasao exceeded +0.01 dB by the registered scene criterion; Panama's
gain was +0.0038 dB and JapaneseGradens/IUI3 selected identity. Consequently
teacher fitting was not triggered.

Official test behavior was mixed. Curasao spatial gain reached 32.917689 dB
(+0.321615 dB over B0), while JapaneseGradens stayed at 24.738506 dB and IUI3
at 30.613711 dB under identity fallback. Panama spatial was 31.460987 dB
(-0.005814 dB), so it also correctly fell back to identity for deployment.
Per-view metrics and visual sheets are under the output directory.

This confirms that transmission gating reduces unwanted edits in some scenes but
does not add information that makes the direct-light correction consistently
identifiable across scenes. The next step should be an analysis-only study of
why Curasao's gate aligns with its error while the other scenes do not, using
regional error/support overlap and gain histograms. Do not increase gain
capacity, add a teacher, alter exposure, or change geometry before that audit.
