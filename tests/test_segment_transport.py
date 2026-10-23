import torch

from water_splatting.rendering.segment_transport import (
    ConstantMedium, RayMediumMLP, WorldMediumGrid, compose_segments,
    composite_fixed_rays, integrate_medium,
)


def test_segment_composition_associative():
    torch.manual_seed(4)
    segments = [(torch.rand(4, 3), torch.rand(4, 3)) for _ in range(3)]
    first = compose_segments(*compose_segments(*segments[0], *segments[1]), *segments[2])
    second = compose_segments(*segments[0], *compose_segments(*segments[1], *segments[2]))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)


def test_homogeneous_medium_matches_analytic_at_bin_edges():
    raw = torch.tensor([.2, .3, -.1, -1., -2., -3., -.4, -.6, -.7], dtype=torch.float64)
    depths = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    weights = torch.tensor([[.2, .3, .1]], dtype=torch.float64)
    colors = torch.tensor([[[.2, .7, .8], [.5, .1, .4], [.9, .5, .7]]], dtype=torch.float64)
    ambient = raw[:3].sigmoid()
    bd, bb = torch.nn.functional.softplus(raw[6:]), torch.nn.functional.softplus(raw[3:6])
    expected = (weights[..., None] * ((-bd * depths[..., None]).exp() * colors
                + (1 - (-bb * depths[..., None]).exp()) * ambient)).sum(1) + (1-weights.sum(1))[:, None]*ambient
    actual = composite_fixed_rays(depths, weights, colors, raw.expand(1, 4, 9), torch.tensor([4.]), torch.ones(1))
    torch.testing.assert_close(actual, expected)


def test_empty_ray_returns_integrated_medium_and_tail():
    raw = torch.randn(2, 8, 9, dtype=torch.float64)
    far, norms = torch.tensor([2., 3.], dtype=torch.float64), torch.tensor([1., 1.2], dtype=torch.float64)
    _, scatter, total_tau = integrate_medium(raw, (far*norms/8)[:, None].expand(2, 8))
    expected = scatter[:, -1]+(-total_tau).exp()*raw[:, -1, :3].sigmoid()
    actual = composite_fixed_rays(torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 1, 3), raw, far, norms)
    torch.testing.assert_close(actual, expected)


def test_medium_gradient_matches_finite_difference():
    raw = torch.randn(1, 5, 9, dtype=torch.float64, requires_grad=True)
    def render(value):
        return composite_fixed_rays(torch.tensor([[.3, 1.7]], dtype=torch.float64),
            torch.tensor([[.4, .3]], dtype=torch.float64), torch.ones(1, 2, 3, dtype=torch.float64)*.5,
            value, torch.tensor([2.], dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    assert torch.autograd.gradcheck(render, (raw,))


def test_models_start_at_common_homogeneous_medium():
    base = torch.randn(9)
    positions = torch.rand(7, 16, 3)
    context = torch.rand(7, 9)
    models = [ConstantMedium(base), WorldMediumGrid(base, torch.zeros(3), torch.ones(3)), RayMediumMLP(base)]
    for model in models:
        torch.testing.assert_close(model(positions, context), base.expand(7, 16, 9), rtol=0, atol=0)
    sizes = [sum(p.numel() for p in model.parameters()) for model in models]
    assert abs(sizes[1]-sizes[2])/sizes[1] < .02


def test_world_grid_same_point_independent_of_camera():
    model = WorldMediumGrid(torch.randn(9), torch.zeros(3), torch.ones(3))
    with torch.no_grad():
        model.delta.normal_()
    positions = torch.rand(3, 8, 3)
    torch.testing.assert_close(model(positions, torch.zeros(3, 9)), model(positions, torch.ones(3, 9)), rtol=0, atol=0)
