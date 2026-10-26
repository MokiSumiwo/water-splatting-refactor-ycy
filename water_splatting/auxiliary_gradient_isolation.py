"""Isolate renderer scratch side effects of auxiliary VJPs.

The rasterizer saves xys_grad_abs for its CUDA backward, which mutates it.
Keep that original as graph scratch and rebind the model-visible statistics
to a private native clone. Legacy A copy_ restoration then touches only the
clone, not a tensor saved by the still-live graph.
"""
from contextlib import contextmanager

@contextmanager
def isolate_densification(model):
    original_buffer = model.xys_grad_abs
    native = original_buffer.detach().clone()
    xygrad = None if model.xys.grad is None else model.xys.grad.detach().clone()
    model.xys_grad_abs = native
    try:
        yield
    finally:
        # Rebind instead of copy_: later auxiliary VJPs may still read the
        # original saved buffer. CUDA changes that scratch buffer, not native.
        model.xys_grad_abs = native
        model.xys.grad = xygrad
