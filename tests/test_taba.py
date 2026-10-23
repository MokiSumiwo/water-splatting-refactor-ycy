"""Chromatic-only TABA allocation, bounds and gradient boundaries."""

import pytest
import torch

from water_splatting.fields.gaussian_appearance import (
    GaussianColorOutput, apply_taba_to_bounded_color, compute_taba_allocation, safe_sigmoid,
)
from water_splatting.taba import sample_taba_transmission


def color_from_logits(base, residual):
    logits = base + residual
    return GaussianColorOutput(rgb=torch.sigmoid(logits), logits=logits,
                               dc_rgb=torch.sigmoid(base), dc_logits=base)


def test_chromatic_allocation_preserves_luminance_and_routes_appearance_gradients():
    dc = torch.tensor([[0.2, -0.5, 0.1]], requires_grad=True)
    rest = torch.tensor([[1.0, -0.3, 0.2]], requires_grad=True)
    transmission = torch.tensor([[0.25]], requires_grad=True)
    gl, gate, diagnostic = compute_taba_allocation(transmission)
    out = apply_taba_to_bounded_color(color_from_logits(dc, rest), gate)
    residual = out.logits - dc
    assert torch.allclose(residual.mean(-1), rest.mean(-1))
    assert torch.allclose(residual - residual.mean(-1, keepdim=True),
                          gate * (rest - rest.mean(-1, keepdim=True)))
    grads = torch.autograd.grad(out.rgb.square().sum(), (dc, rest, transmission), allow_unused=True)
    assert grads[0].abs().sum() > 0 and grads[1].abs().sum() > 0 and grads[2] is None
    assert all(not value.requires_grad for value in (gl, gate, diagnostic))
    assert torch.equal(gl, torch.ones_like(gl))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_extreme_logits_are_strictly_bounded(dtype):
    logits = torch.tensor([-1e4, -100.0, 0.0, 100.0, 1e4], dtype=dtype)
    rgb = safe_sigmoid(logits)
    assert torch.isfinite(rgb).all() and (rgb > 0).all() and (rgb < 1).all()


def test_invalid_transmission_and_depth_fall_back_to_bnd():
    transmission = torch.tensor([[float("nan")], [float("inf")], [-0.1], [1.1], [0.2]])
    _, gate, _ = compute_taba_allocation(transmission, valid_mask=torch.tensor([[1], [1], [1], [1], [0]]))
    assert torch.equal(gate, torch.ones_like(gate))
    base, rest = torch.randn(5, 3), torch.randn(5, 3)
    bnd = color_from_logits(base, rest)
    out = apply_taba_to_bounded_color(bnd, gate)
    assert torch.equal(out.logits, bnd.logits) and torch.equal(out.rgb, bnd.rgb)


def test_proxy_uses_expected_depth_and_mean_rgb_transmission_without_gradients():
    attn = torch.tensor([[[0.2, 0.5, 1.0], [0.1, 0.1, 0.1]],
                         [[0.4, 0.4, 0.4], [0.1, 0.1, 0.1]]], requires_grad=True)
    depth = torch.tensor([[[2.0], [0.0]], [[float("nan")], [3.0]]], requires_grad=True)
    alpha = torch.tensor([[[1.0], [1.0]], [[1.0], [0.0]]], requires_grad=True)
    xy = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [-1., 0.], [float("nan"), 0.]], requires_grad=True)
    proxy = sample_taba_transmission(attn, depth, alpha, xy, torch.ones(6))
    assert torch.allclose(proxy["transmission"][0], torch.exp(-attn[0, 0].detach() * 2).mean())
    assert proxy["valid"].flatten().tolist() == [True, False, False, False, False, False]
    assert all(not value.requires_grad and torch.isfinite(value).all() for value in proxy.values())
    _, gate, _ = compute_taba_allocation(proxy["transmission"], valid_mask=proxy["valid"])
    assert torch.equal(gate[1:], torch.ones_like(gate[1:]))


@pytest.mark.parametrize("mid,temp,floor", [(0.5, 0.15, 0.2), (0.4, 0.2, 0.3)])
def test_registered_gate_is_deterministic_monotone(mid, temp, floor):
    t = torch.linspace(0, 1, 101)[:, None]
    _, gate, _ = compute_taba_allocation(t, transmission_mid=mid, transmission_temperature=temp, chroma_floor=floor)
    expected = floor + (1 - floor) * torch.sigmoid((t - mid) / temp)
    assert torch.equal(gate, expected)
    assert (gate[1:] >= gate[:-1]).all() and (gate >= floor).all() and (gate <= 1).all()


@pytest.mark.parametrize("kwargs", [{"transmission_temperature": 0}, {"transmission_temperature": float("nan")},
                                    {"transmission_mid": 2}, {"chroma_floor": -0.1}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        compute_taba_allocation(torch.ones(1, 1), **kwargs)


def test_channelwise_gate_is_rejected():
    with pytest.raises(ValueError):
        apply_taba_to_bounded_color(color_from_logits(torch.zeros(2, 3), torch.ones(2, 3)), torch.ones(2, 3))
