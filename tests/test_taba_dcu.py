"""Directional allocation, invalid auxiliaries and detached gradient boundaries."""

import pytest
import torch

from water_splatting.fields.gaussian_appearance import safe_sigmoid
from water_splatting.taba import compute_taba_dcu, sample_taba_transmission


def test_directional_decomposition_and_orthogonal_preservation():
    full = torch.tensor([[1., -.5, -.5], [.4, -.7, .3]], dtype=torch.float64)
    base = torch.zeros_like(full)
    axis = torch.tensor([[1., -1., 0.]], dtype=torch.float64).expand_as(full)
    out = compute_taba_dcu(full, base, torch.ones_like(full), base, axis, mode="directional")
    w = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    parallel = (full * w).sum(-1, keepdim=True) * w
    assert torch.allclose(out['delta_parallel'], parallel, atol=1e-12)
    assert torch.allclose(out['delta_parallel'] + out['delta_orthogonal'], full)
    assert ((out['delta_orthogonal'] * w).sum(-1).abs() < 3e-8).all()
    assert torch.allclose(out['s_adjusted'], full + (out['gate'] - 1) * parallel)


def test_rgb_axis_is_not_silently_recentered():
    full = torch.tensor([[1., -1., 0.]])
    out = compute_taba_dcu(full, torch.zeros_like(full), torch.ones_like(full),
                           torch.zeros_like(full), torch.tensor([[1., 0., 0.]]))
    assert out['water_axis'][0, 1:].abs().sum() == 0
    assert out['luminance_logit_change'].abs().sum() > 0


def test_uncertainty_uses_channel_population_variance():
    t = torch.tensor([[.1, .4, .7], [.5, .5, .5]])
    full = torch.tensor([[1., -1., .1]]).expand(2, 3)
    axis = torch.tensor([[0., 1., 1.]]).expand(2, 3)
    out = compute_taba_dcu(full, torch.zeros_like(full), -t.log(), torch.zeros_like(full), axis, transmission=t)
    cv = (t.std(-1, keepdim=True, correction=0) / (t.mean(-1, keepdim=True) + 1e-8)).clamp(0, 1)
    assert torch.allclose(out['q_T'], t.mean(-1, keepdim=True) * (1 - .5 * cv))
    assert out['T_std'][1] == 0 and out['T_cv'][1] == 0


@pytest.mark.parametrize('axis', [torch.zeros(2, 3), torch.full((2, 3), float('nan')), torch.full((2, 3), float('inf'))])
def test_bad_water_axis_uses_original_chromatic_gate(axis):
    full, base, tau = torch.randn(2, 3), torch.zeros(2, 3), torch.ones(2, 3)
    a = compute_taba_dcu(full, base, tau, base, axis)
    b = compute_taba_dcu(full, base, tau, mode='chromatic')
    assert a['fallback_mask'].all()
    assert torch.equal(a['s_adjusted'], b['s_adjusted'])
    assert all(torch.isfinite(v).all() for v in a.values())


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.])
def test_invalid_transmission_reverts_appearance(bad):
    full, base = torch.randn(2, 3), torch.zeros(2, 3)
    out = compute_taba_dcu(full, base, torch.full_like(full, bad), base, torch.ones_like(full))
    assert torch.equal(out['gate'], torch.ones(2, 1))
    assert torch.equal(out['q_T'], torch.ones(2, 1))
    assert torch.equal(out['s_adjusted'], full)
    assert torch.equal(out['rgb'], safe_sigmoid(full))
    assert all(torch.isfinite(v).all() for v in out.values())


def test_negative_depth_and_attenuation_are_invalid_samples():
    attn = torch.tensor([[[.1, .2, .3], [-.1, .2, .3]]])
    depth = torch.tensor([[[-1.], [1.]]])
    proxy = sample_taba_transmission(attn, depth, torch.ones_like(depth), torch.tensor([[0., 0.], [1., 0.]]), torch.ones(2))
    assert not proxy['valid'].any()


def test_no_valid_samples_and_empty_gaussians():
    full = torch.empty(0, 3)
    out = compute_taba_dcu(full, full, full)
    assert out['rgb'].shape == (0, 3)
    assert out['gate'].shape == (0, 1)
    full = torch.randn(2, 3)
    out = compute_taba_dcu(full, torch.zeros_like(full), torch.ones_like(full), valid_mask=torch.zeros(2, 1, dtype=torch.bool))
    assert torch.equal(out['s_adjusted'], full)


@pytest.mark.parametrize('mode', ['chromatic', 'directional', 'directional_uncertainty'])
def test_auxiliary_and_diagnostics_are_detached(mode):
    dc = torch.randn(2, 3, requires_grad=True)
    rest = torch.randn(2, 3, requires_grad=True)
    tau = torch.ones(2, 3, requires_grad=True)
    obj = torch.randn(2, 3, requires_grad=True)
    med = torch.randn(2, 3, requires_grad=True)
    out = compute_taba_dcu(dc + rest, dc, tau, obj, med, mode=mode)
    grads = torch.autograd.grad(out['rgb'].square().sum(), (dc, rest, tau, obj, med), allow_unused=True)
    assert grads[0].abs().sum() > 0 and grads[1].abs().sum() > 0
    assert grads[2:] == (None, None, None)
    assert all(not value.requires_grad for key, value in out.items() if key not in ('rgb', 's_adjusted'))


def test_disabled_exact_and_extreme_active_bounds():
    full = torch.tensor([[-1e4, 0., 1e4]])
    base = torch.zeros_like(full)
    disabled = compute_taba_dcu(full, base, torch.ones_like(full), mode='disabled')
    assert torch.equal(disabled['rgb'], full.sigmoid())
    assert torch.equal(disabled['s_adjusted'], full)
    active = compute_taba_dcu(full, base, torch.ones_like(full))
    assert (active['rgb'] > 0).all() and (active['rgb'] < 1).all()
