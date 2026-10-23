import pytest
import torch

from water_splatting.rendering.optical_thickness_distribution import (
    composite_fixed_transport, distributed_transmission,
)


def rays():
    return dict(
        depths=torch.tensor([[0.3, 1.4, 4.2], [0., 0., 0.]], dtype=torch.double),
        weights=torch.tensor([[0.2, 0.35, 0.1], [0., 0., 0.]], dtype=torch.double),
        colors=torch.tensor([[[.2, .6, .9], [.8, .3, .1], [.4, .5, .2]],
                             [[.1, .2, .3]] * 3], dtype=torch.double),
        ambient=torch.tensor([[.1, .5, .7], [.3, .6, .8]], dtype=torch.double),
        backscatter=torch.tensor([[.5, .3, .2], [.3, .4, .5]], dtype=torch.double),
        attenuation=torch.tensor([[.9, .5, .3], [.3, .4, .5]], dtype=torch.double),
    )


def test_identity_endpoint_has_nonzero_gradient():
    tau = torch.tensor([.2, 1., 5.], dtype=torch.double)
    eta = torch.tensor(0., dtype=torch.double, requires_grad=True)
    result = distributed_transmission(tau, eta)
    assert torch.equal(result, torch.exp(-tau))
    result.sum().backward()
    assert eta.grad > 0


@pytest.mark.parametrize("eta", [0., .3, 1.])
def test_bounds_and_monotonicity(eta):
    tau = torch.linspace(0, 100, 1001, dtype=torch.double)
    value = distributed_transmission(tau, torch.tensor(eta))
    assert value[0] == 1 and torch.all((value >= 0) & (value <= 1))
    assert torch.all(value[1:] <= value[:-1])
    assert torch.all(value >= torch.exp(-tau) - 1e-15)


def test_matches_three_complete_transports_including_empty_ray():
    data = rays()
    eta = torch.tensor(.4, dtype=torch.double)
    baseline = composite_fixed_transport(**data)
    reference = (1-eta)*baseline + eta*.5*(
        composite_fixed_transport(**data, optical_scale=.5)
        + composite_fixed_transport(**data, optical_scale=1.5))
    actual = composite_fixed_transport(**data, eta=eta)
    torch.testing.assert_close(actual, reference, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(actual[1], data["ambient"][1])


def test_gradient_agrees_with_finite_differences():
    data = rays()
    eta = torch.tensor(.4, dtype=torch.double, requires_grad=True)
    attenuation = data.pop("attenuation").requires_grad_()
    bs = data.pop("backscatter").requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a, b, e: composite_fixed_transport(**data, attenuation=a, backscatter=b, eta=e),
        (attenuation, bs, eta), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_zero_medium_reconstructs_original_clear():
    data = rays()
    for key in ("ambient", "backscatter", "attenuation"):
        data[key] = torch.zeros_like(data[key])
    clear = (data["weights"][..., None]*data["colors"]).sum(1)
    torch.testing.assert_close(composite_fixed_transport(**data, eta=torch.tensor(.7)), clear)
