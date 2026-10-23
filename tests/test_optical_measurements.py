import math

import torch

from water_splatting.rendering.optical_measurements import (
    admissible_clear_grid, dual_path_direct, homogeneous_observation, intersect_z_plane,
    linear_to_srgb, observation_jacobian, polarization_measurement,
    recover_linear_stokes, select_informative_distances, spectral_direct,
    srgb_to_linear, trace_flat_port,
)


def test_normal_incidence_and_snell_law():
    directions=torch.tensor([[0.,0.,1.],[.5,0.,1.]],dtype=torch.float64)
    origin,water=trace_flat_port(directions,torch.tensor(.1),torch.tensor(.01),torch.tensor(1.333,dtype=torch.float64))
    torch.testing.assert_close(water[0],torch.tensor([0.,0.,1.],dtype=torch.float64))
    normalized=directions/directions.norm(dim=-1,keepdim=True)
    torch.testing.assert_close(normalized[:,0],1.333*water[:,0])
    assert origin[1,0]>0
    assert (intersect_z_plane(origin,water,torch.tensor([1.,2.]))[:,2]-torch.tensor([1.,2.])).abs().max()<1e-12


def test_refractive_calibration_gradients():
    directions=torch.tensor([[.4,.1,1.]],dtype=torch.float64)
    args=tuple(torch.tensor(v,dtype=torch.float64,requires_grad=True) for v in (.1,.01,1.33))
    assert torch.autograd.gradcheck(lambda a,g,n:torch.cat(trace_flat_port(directions,a,g,n),-1),args)


def test_radiometric_roundtrip():
    values=torch.linspace(0,1,200,dtype=torch.float64)
    torch.testing.assert_close(srgb_to_linear(linear_to_srgb(values)),values,atol=1e-12,rtol=1e-10)


def test_dual_path_zero_water_and_path_additivity():
    rho=torch.tensor([.2,.4,.8])
    illumination=torch.tensor([.8,.7,.6])
    zero=torch.zeros(3)
    torch.testing.assert_close(dual_path_direct(rho,illumination,zero,torch.tensor(2.),torch.tensor(3.)),rho*illumination)
    beta=torch.tensor([.1,.2,.3])
    torch.testing.assert_close(dual_path_direct(rho,illumination,beta,torch.tensor(2.),torch.tensor(3.)),rho*illumination*(-beta*5).exp())


def test_spectral_zero_medium_sensor_integral():
    spectra=torch.rand(7,6)
    sensor=torch.rand(3,6)
    torch.testing.assert_close(spectral_direct(spectra,torch.zeros(6),torch.arange(7.),sensor),spectra@sensor.T)


def test_polarization_stokes_roundtrip():
    stokes=torch.tensor([[.6,.1,-.2],[.4,-.05,.1]],dtype=torch.float64)
    angles=torch.arange(4,dtype=torch.float64)*math.pi/4
    measured=polarization_measurement(stokes,angles)
    torch.testing.assert_close(recover_linear_stokes(measured,angles),stokes,atol=1e-12,rtol=1e-10)
    assert (measured>=0).all()


def test_admissible_solutions_reproduce_observation():
    clear,beta,background=admissible_clear_grid(torch.tensor(.3),2.,torch.linspace(.01,1,30),torch.linspace(0,1,30),1e-6)
    prediction=homogeneous_observation(clear[:,None],beta[:,None],background[:,None],torch.full_like(clear,2.))[:,0]
    torch.testing.assert_close(prediction,torch.full_like(prediction,.3),atol=1e-6,rtol=0)
    assert clear.max()-clear.min()>.5


def test_design_uses_distinct_measurements_and_adds_information():
    distance=torch.linspace(.5,6,30,dtype=torch.float64)
    j=observation_jacobian(torch.tensor(.7),torch.tensor(.3),torch.tensor(.2),distance)
    chosen,information=select_informative_distances(j,5)
    assert len(set(chosen))==5
    assert torch.linalg.eigvalsh(information).min()>0
