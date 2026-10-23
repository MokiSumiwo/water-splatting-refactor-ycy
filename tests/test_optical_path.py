import numpy as np
import pytest
import torch
from scipy.integrate import quad

from water_splatting.rendering.optical_path import (
    conditional_depth_parameters,
    positive_normal_transmission,
)


@pytest.mark.parametrize("mean,sigma,beta", [
    (2., .01, .1), (2., .7, .4), (.2, 1., 2.), (-1., .5, .2),
    (4., .3, 0.), (2., 3., 12.), (1., 100., .5),
])
def test_transmission_against_numerical_quadrature(mean, sigma, beta):
    lower = max(0., mean - 12 * sigma)
    upper = max(0., mean) + 12 * sigma
    log_mode = -0.5 * (min(mean, 0.) / sigma) ** 2
    density = lambda z: np.exp(-0.5 * ((z - mean) / sigma) ** 2 - log_mode)
    denominator = quad(density, lower, upper, epsabs=1e-12, epsrel=1e-12)[0]
    reference = quad(lambda z: density(z) * np.exp(-beta * z), lower, upper,
                     epsabs=1e-12, epsrel=1e-12)[0] / denominator
    actual = positive_normal_transmission(*(torch.tensor(x, dtype=torch.float64)
                                          for x in (mean, sigma ** 2, beta)))
    assert float(actual) == pytest.approx(reference, rel=1e-9, abs=1e-12)


def test_point_limit_and_zero_medium():
    means = torch.tensor([.1, 2., 4.], dtype=torch.float64)
    beta = torch.tensor([.2, .4, .6], dtype=torch.float64)
    assert torch.equal(positive_normal_transmission(means, torch.zeros_like(means), beta), (-means * beta).exp())
    assert torch.equal(positive_normal_transmission(means, torch.ones_like(means), torch.zeros_like(beta)), torch.ones_like(means))


def test_transmission_gradients():
    args = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True)
                 for value in ([.5, 3.], [.4, .1], [.2, .8]))
    assert torch.autograd.gradcheck(positive_normal_transmission, args)


def test_conditional_depth_matches_schur_complement():
    mean = torch.tensor([[.4, -.3, 2.]], dtype=torch.float64)
    covariance = torch.tensor([[.04, .005, .008], [.005, .02, -.003], [.008, -.003, .09]], dtype=torch.float64)
    packed = covariance[torch.tensor([0, 0, 0, 1, 1, 2]), torch.tensor([0, 1, 2, 1, 2, 2])][None]
    jacobian = torch.tensor([[50., 0., -10.], [0., 45., 6.75]], dtype=torch.float64)
    xy_covariance = jacobian @ covariance @ jacobian.T + .3 * torch.eye(2, dtype=torch.float64)
    inverse = xy_covariance.inverse()
    conic = inverse[torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])][None]
    cross = covariance[2] @ jacobian.T
    slope = cross @ inverse
    expected_variance = covariance[2, 2] - slope @ cross
    actual = conditional_depth_parameters(mean, packed, torch.eye(4, dtype=torch.float64), conic, 100., 90., 100, 80)
    torch.testing.assert_close(actual[0], torch.cat((slope, expected_variance[None])))
