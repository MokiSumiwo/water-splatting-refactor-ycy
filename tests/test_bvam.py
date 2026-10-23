"""Tensor-level BVAM invariants."""

import torch

import water_splatting.bvam as bvam
from water_splatting.bvam import compute_bvam_gaussian_colors


def _fake_sh(_degree, _viewdirs, coeffs):
    return coeffs[:, 0, :]


def test_bvam_is_bounded_and_mixes_valid_weights(monkeypatch) -> None:
    monkeypatch.setattr(bvam, "spherical_harmonics", _fake_sh)
    output = compute_bvam_gaussian_colors(
        means=torch.randn(5, 3),
        features_dc=torch.randn(5, 3) * 8.0,
        features_rest=torch.randn(5, 3, 3) * 8.0,
        bvam_delta=torch.randn(5, 2, 4, 3) * 8.0,
        bvam_gate=torch.randn(5, 2, 4) * 8.0,
        camera_position=torch.zeros(1, 3),
        sh_degree=3,
        active_sh_degree=1,
    )
    assert bool((output.rgb > 0.0).all())
    assert bool((output.rgb < 1.0).all())
    assert bool((output.expert_rgb > 0.0).all())
    assert bool((output.expert_rgb < 1.0).all())
    assert torch.all(output.gate_probs >= 0.0)
    assert torch.allclose(output.gate_probs.sum(dim=1), torch.ones(5), atol=1e-6)


def test_bvam_zero_delta_uniform_gate_matches_bounded_shared_path(monkeypatch) -> None:
    monkeypatch.setattr(bvam, "spherical_harmonics", _fake_sh)
    means = torch.randn(4, 3)
    dc = torch.randn(4, 3)
    rest = torch.randn(4, 3, 3)
    output = compute_bvam_gaussian_colors(
        means=means,
        features_dc=dc,
        features_rest=rest,
        bvam_delta=torch.zeros(4, 2, 4, 3),
        bvam_gate=torch.zeros(4, 2, 4),
        camera_position=torch.zeros(1, 3),
        sh_degree=3,
        active_sh_degree=1,
    )
    expected = torch.sigmoid(_fake_sh(1, None, torch.cat((dc[:, None, :], rest), dim=1)))
    assert torch.allclose(output.rgb, expected)
    assert torch.allclose(output.gate_probs, torch.full((4, 2), 0.5))


def test_bvam_invalid_support_falls_back_to_uniform_and_zero_delta(monkeypatch) -> None:
    monkeypatch.setattr(bvam, "spherical_harmonics", _fake_sh)
    output = compute_bvam_gaussian_colors(
        means=torch.randn(3, 3),
        features_dc=torch.randn(3, 3),
        features_rest=torch.randn(3, 3, 3),
        bvam_delta=torch.ones(3, 2, 4, 3),
        bvam_gate=torch.randn(3, 2, 4),
        camera_position=torch.zeros(1, 3),
        sh_degree=3,
        active_sh_degree=1,
        valid_mask=torch.tensor([True, False, True]),
    )
    assert torch.allclose(output.gate_probs[1], torch.full((2,), 0.5))
    assert torch.allclose(output.deltas[1], torch.zeros_like(output.deltas[1]))
