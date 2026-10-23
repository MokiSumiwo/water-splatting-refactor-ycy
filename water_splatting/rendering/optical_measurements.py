"""Physical measurement operators for explicitly calibrated experiments."""

from __future__ import annotations

import torch
from torch import Tensor


def trace_flat_port(air_directions: Tensor, air_distance: Tensor,
                    glass_thickness: Tensor, water_index: Tensor, glass_index: float = 1.5):
    """Parallel air/glass/water interfaces normal to +z; air index is one.

    Returned water-ray origins include both interface intersections.
    Directions must point forward. Interface distances use the same metric
    scale as calibration points. This is not a pinhole image undistorter.
    """
    directions = air_directions / air_directions.norm(dim=-1, keepdim=True)
    tangent = directions[..., :2]
    glass_xy = tangent / glass_index
    water_xy = tangent / water_index
    glass_z = (1 - glass_xy.square().sum(-1, keepdim=True)).sqrt()
    water_z = (1 - water_xy.square().sum(-1, keepdim=True)).sqrt()
    first_xy = tangent * air_distance / directions[..., 2:3]
    final_xy = first_xy + glass_xy * glass_thickness / glass_z
    origin_z = torch.ones_like(glass_z) * (air_distance + glass_thickness)
    return torch.cat((final_xy, origin_z), -1), torch.cat((water_xy, water_z), -1)


def intersect_z_plane(origins: Tensor, directions: Tensor, depth: Tensor):
    distance = (depth[..., None] - origins[..., 2:3]) / directions[..., 2:3]
    return origins + distance * directions


def dual_path_direct(reflectance: Tensor, illumination: Tensor, beta: Tensor,
                     incident_distance: Tensor, outgoing_distance: Tensor):
    return reflectance * illumination * (-beta * (incident_distance + outgoing_distance)[..., None]).exp()


def linear_to_srgb(value: Tensor):
    value = value.clamp_min(0)
    return torch.where(value <= .0031308, 12.92 * value, 1.055 * value.clamp_min(.0031308).pow(1/2.4) - .055)


def srgb_to_linear(value: Tensor):
    return torch.where(value <= .04045, value / 12.92, ((value + .055) / 1.055).clamp_min(0).pow(2.4))


def spectral_direct(spectra: Tensor, beta: Tensor, distance: Tensor, sensor: Tensor):
    """Known K-wavelength nonnegative spectra and a 3xK sensor response."""
    return (spectra * (-beta * distance[..., None]).exp()) @ sensor.T


def polarization_measurement(stokes: Tensor, angles: Tensor):
    """Ideal linear polarizer measurements; last Stokes axis is (I,Q,U)."""
    return .5 * (stokes[..., 0, None] + stokes[..., 1, None] * (2*angles).cos()
                 + stokes[..., 2, None] * (2*angles).sin())


def recover_linear_stokes(measurements: Tensor, angles: Tensor):
    design = .5 * torch.stack((torch.ones_like(angles), (2*angles).cos(), (2*angles).sin()), -1)
    return measurements @ torch.linalg.pinv(design).T


def homogeneous_observation(clear: Tensor, beta: Tensor, background: Tensor, distance: Tensor):
    transmission = (-beta * distance[..., None]).exp()
    return transmission * clear + (1-transmission) * background


def admissible_clear_grid(observed: Tensor, distance: float, beta_grid: Tensor,
                          background_grid: Tensor, tolerance: float):
    """Explicit feasible scalar solutions, not posterior confidence bounds."""
    beta, background = torch.meshgrid(beta_grid, background_grid, indexing="ij")
    transmission = (-beta * distance).exp()
    clear = (observed - (1-transmission)*background)/transmission
    predicted = transmission*clear.clamp(0,1)+(1-transmission)*background
    valid = (clear>=0)&(clear<=1)&((predicted-observed).abs()<=tolerance)
    return clear[valid], beta[valid], background[valid]


def observation_jacobian(clear: Tensor, beta: Tensor, background: Tensor, distances: Tensor):
    """Scalar-channel Jacobian in (clear,beta,background) coordinates."""
    t = (-beta*distances).exp()
    return torch.stack((t, -distances*t*(clear-background), 1-t), -1)


def select_informative_distances(jacobian: Tensor, count: int, noise_variance: float = 1e-4):
    """Greedy D-optimal selection under an explicit local parameter model."""
    information = torch.eye(jacobian.shape[-1], device=jacobian.device, dtype=jacobian.dtype)*1e-8
    selected=[]
    candidates=jacobian[...,None]*jacobian[:,None,:]/noise_variance
    for _ in range(count):
        values=torch.linalg.slogdet(information[None]+candidates).logabsdet
        if selected: values[selected]=-torch.inf
        index=int(values.argmax())
        selected.append(index)
        information=information+candidates[index]
    return selected, information
