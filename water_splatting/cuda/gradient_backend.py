"""Opt-in corrected RGB backward, preserving the installed native forward.

Set WATER_SPLATTING_GRADIENT_BACKEND=corrected_v1 before importing the renderer.
Historical rendering remains on the installed csrc.so; no binary is overwritten.
"""
import hashlib
from pathlib import Path
from torch.utils.cpp_extension import load


class CorrectedGradientBackend:
    gradient_abi_version = 1

    def __init__(self, native, corrected):
        self.native = native
        self.corrected = corrected

    def __getattr__(self, name):
        if name == "rasterize_backward":
            return self.corrected.rasterize_backward
        return getattr(self.native, name)


def build_corrected_backend(native):
    # The corrected VJP is validated against this precise forward binary.
    expected = "681978131413f306b8cee5d4235b98ea316768765f652f5db6d8700180f88c5c"
    actual = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"corrected_v1 forward identity mismatch: {actual}; revalidate before using this backend")
    root = Path(__file__).resolve().parent
    source = root / "gradient_fix"
    corrected = load(
        name="water_splatting_corrected_gradient_v1",
        sources=sorted(str(p) for p in source.iterdir() if p.suffix in (".cu", ".cpp")),
        extra_include_paths=[str(root / "csrc"), str(root / "csrc/third_party/glm")],
        extra_cflags=["-O3"], extra_cuda_cflags=["-O3"], verbose=False,
    )
    return CorrectedGradientBackend(native, corrected)
