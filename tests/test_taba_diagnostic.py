import torch

from water_splatting.fields.gaussian_appearance import compute_taba_allocation


def test_taba_gates_are_detached_and_bounded_monotone():
    t = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)
    support = torch.tensor([3.0, 4.0, 6.0], requires_grad=True)
    gl, gc, rel = compute_taba_allocation(t, support)
    assert not gl.requires_grad and not gc.requires_grad and not rel.requires_grad
    assert bool(((gl >= 0) & (gl <= 1)).all())
    assert bool(((gc >= 0) & (gc <= 1)).all())
    assert bool((gl[1:] >= gl[:-1]).all())
    assert bool((gc[1:] >= gc[:-1]).all())


def test_taba_support_fallback_has_no_suppression():
    gl, gc, rel = compute_taba_allocation(torch.tensor([0.2]), torch.tensor([0.0]))
    assert torch.allclose(gl, torch.tensor([0.6]))
    assert torch.allclose(gc, torch.tensor([0.2]))
    assert torch.allclose(rel, torch.zeros(1))


def test_luminance_chroma_decomposition():
    delta = torch.tensor([[1.0, 2.0, 3.0]])
    luma = delta.mean(dim=-1, keepdim=True)
    chroma = delta - luma
    assert torch.allclose(luma, torch.tensor([[2.0]]))
    assert torch.allclose(chroma, torch.tensor([[-1.0, 0.0, 1.0]]))
