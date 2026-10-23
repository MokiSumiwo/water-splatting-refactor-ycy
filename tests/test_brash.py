"""Tensor-level BRASH and reliability invariants."""

import torch

from water_splatting.brash_reliability import build_reliability_prior
from water_splatting.fields.gaussian_appearance import (
    compute_brash_gaussian_colors,
    compute_bounded_gaussian_colors,
)


def test_brash_outputs_are_strictly_bounded() -> None:
    torch.manual_seed(4)
    means = torch.randn(5, 3)
    dc = torch.randn(5, 3) * 8.0
    rest = torch.randn(5, 3, 3) * 8.0
    gate = torch.randn(5, 1)
    result = compute_brash_gaussian_colors(
        means=means,
        features_dc=dc,
        features_rest=rest,
        brash_gate_logit=gate,
        camera_position=torch.zeros(1, 3),
        sh_degree=0,
        active_sh_degree=0,
    )
    assert bool((result.rgb > 0.0).all())
    assert bool((result.rgb < 1.0).all())


def test_brash_fixed_gate_matches_bounded_full_sh_degree_zero() -> None:
    means = torch.randn(4, 3)
    dc = torch.randn(4, 3)
    rest = torch.zeros(4, 3, 3)
    gate = torch.zeros(4, 1)
    bounded = compute_bounded_gaussian_colors(
        means=means,
        features_dc=dc,
        features_rest=rest,
        camera_position=torch.zeros(1, 3),
        sh_degree=0,
        active_sh_degree=0,
    )
    brash = compute_brash_gaussian_colors(
        means=means,
        features_dc=dc,
        features_rest=rest,
        brash_gate_logit=gate,
        camera_position=torch.zeros(1, 3),
        sh_degree=0,
        active_sh_degree=0,
        gate_mode="fixed_one",
    )
    assert torch.allclose(brash.rgb, bounded.rgb)


def test_reliability_is_detached_and_finite() -> None:
    confidence, components = build_reliability_prior(
        support=torch.tensor([1.0, 3.0, 6.0], requires_grad=True),
        depth_variance=torch.tensor([0.0, 0.25, 2.0], requires_grad=True),
        transmission=torch.tensor([0.2, 0.7, 1.0], requires_grad=True),
        view_spread=torch.tensor([0.0, 0.5, 1.0], requires_grad=True),
        bank_size=6,
        minimum_support=3,
        transmission_floor=0.0,
        depth_temperature=1.0,
        view_spread_scale=1.0,
    )
    assert not confidence.requires_grad
    assert all(not value.requires_grad for value in components.values())
    assert bool(torch.isfinite(confidence).all())
    assert bool(((confidence >= 0.0) & (confidence <= 1.0)).all())
