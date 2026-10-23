# Frozen A(0.10) reproduction

This snapshot preserves the completed four-scene seed-42 experiment (15,000 updates per scene). The reference equal-scene underwater PSNR is **29.999253749847412**. This is a single-seed observation, not a clear-image quality guarantee or a bitwise reproducibility promise. GPU atomic operations can be nondeterministic.

## Dependencies and data

Clone with submodules (`git clone --recurse-submodules`). The original README describes the Python 3.8 / PyTorch / Nerfstudio installation. `pip-freeze.txt`, `conda-explicit.txt`, and `runtime.json` record the actual experiment environment; editable local package entries must be installed from this checkout instead of their old absolute paths. Use CUDA 11.8 and make `/usr/local/cuda-11.8/bin` available in PATH (adjust for your machine). Install ffmpeg for exported real-camera videos. LPIPS uses the existing local LPIPS-VGG implementation and its pretrained VGG dependency.

Place the original preprocessed data in `undistorted_data/undistorted_<scene>/`, including images, COLMAP `sparse/0`, and `train_list.txt` / `test_list.txt`. `data_sha256.json` identifies the inputs; the dataset and trained checkpoints are not uploaded. Recomputing COLMAP or using another split is a different experiment.

The corrected backward explicitly checks the native forward binary SHA-256. The small **Linux x86-64 / original PyTorch-CUDA environment** binary is archived in `native/csrc.so`. After installing the project, copy it to `water_splatting/csrc.so` to use the validated forward. It is not a portable binary; a different platform/toolchain requires rebuilding and validating the gradient audit before comparison. Do not remove the identity check to bypass this limitation. Corrected rasterizer source and experiment-local projection source are committed and JIT-built.

## Run

From the repository root in the recorded environment:

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
cp reproducibility/a010/native/csrc.so water_splatting/csrc.so
python -m scripts.experiments.reproduce_geometry_gradient --arm A010 --gpu 0 --output research_runs/a010_reproduction --export
```

The launcher refuses an existing output directory, uses the active Python interpreter, and runs the four scenes sequentially. `--scenes Curasao` selects a single scene. Use `--arm B0` or `--arm A025` with different fresh output paths for paired references. No historical checkpoint is loaded. Do not use the old campaign launchers for reproduction: their output paths are historical and intentionally fixed.

Preserved dependencies include source configs under `research_runs/bnd_on_baseline_scratch15k_20261021_v2`, frozen region/camera JSON under `research_runs/geometry_gradient_A_20261021/regions`, and projection correction sources under that same campaign. Absolute paths in region JSON are provenance only. The training worker resolves the dataset relative to the checkout.

The experiment uses M1 + OCMC + CICA + BND ON, corrected_v1 plus projection backward correction. A changes only supported center gradients, peaks at eta=.10, and is zero after step 10,000. CICA starts at 10,000. Extra VJPs restore densification buffers. B0 uses the same renderer repairs.

Outputs include step-15,000 checkpoints and float evaluation. `--export` adds fixed training views, test floats, native camera videos, and contribution diagnostics. Metrics use `metrics.py`; report scene means and their equal-weight macro mean. Compare against the committed `three_arm_report.md`, not historical worker LPIPS with another implementation. The 13 test views consist of 3/4/3/3 views across Curasao/IUI3/Japanese/Panama.

Residual ROIs remain U (unknown mixture). Neither A arm established the registered 20% residual-removal gate. B/C and additional seeds are separate experiments.
