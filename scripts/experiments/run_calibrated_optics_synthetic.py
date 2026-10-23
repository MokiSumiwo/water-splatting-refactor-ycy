#!/usr/bin/env python3
"""Truth-controlled studies for candidates requiring unavailable measurements."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares

from scripts.experiments.run_taba_bnd_on import check_environment,write_json
from water_splatting.rendering.optical_measurements import (
    admissible_clear_grid,dual_path_direct,homogeneous_observation,intersect_z_plane,
    linear_to_srgb,observation_jacobian,polarization_measurement,
    recover_linear_stokes,select_informative_distances,spectral_direct,trace_flat_port,
)


def fit(parameter,prediction,target,steps=2500,lr=.02):
    optimizer=torch.optim.Adam([parameter],lr=lr)
    log=[]
    for step in range(steps):
        optimizer.zero_grad()
        loss=(prediction(parameter)-target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if step%250==0 or step==steps-1:
            log.append(dict(step=step+1,mse=float(loss.detach())))
    return log


def refraction(output):
    direction=torch.cat((torch.rand(120,2)*1.2-.6,torch.ones(120,1)),1)
    z=torch.linspace(.4,3.,120)
    truth=torch.tensor([.08,.012,1.333])
    origins,rays=trace_flat_port(direction,*truth)
    points=intersect_z_plane(origins,rays,z)
    train=torch.arange(120)%4!=0
    parameter=torch.nn.Parameter(torch.tensor([-2.,.1]))
    def render(p):
        origins,rays=trace_flat_port(direction,F.softplus(p[0]),truth[1],1+F.softplus(p[1]))
        return intersect_z_plane(origins,rays,z)
    log=fit(parameter,lambda p:render(p)[train],points[train])
    full=render(parameter).detach()
    baseline=direction/direction[:,2:3]*z[:,None]
    result=dict(candidate=3,known_glass_thickness=float(truth[1]),true_air_distance=float(truth[0]),true_water_index=float(truth[2]),
        fitted_air_distance=float(F.softplus(parameter[0])),fitted_water_index=float(1+F.softplus(parameter[1])),
        refractive_test_rmse=float((full[~train]-points[~train]).square().mean().sqrt()),
        pinhole_test_rmse=float((baseline[~train]-points[~train]).square().mean().sqrt()),training=log,
        real_data_status="Missing housing calibration; no real image reprojection or four-scene benefit claim")
    np.savez_compressed(output/"synthetic_observations.npz",directions=direction.numpy(),depth=z.numpy(),points=points.numpy(),prediction=full.numpy())
    torch.save(parameter.detach(),output/"fitted_parameters.pt")
    return result


def dual_path(output):
    incident=.2+3*torch.rand(200)
    outgoing=.2+4*torch.rand(200)
    light=.5+.5*torch.rand(200,3)
    rho=torch.tensor([.3,.6,.8]); beta=torch.tensor([.25,.12,.07])
    target=dual_path_direct(rho,light,beta,incident,outgoing)
    train=torch.arange(200)%4!=0
    results={}
    for mode in ("outgoing_only","dual_path"):
        parameter=torch.nn.Parameter(torch.tensor([0.,0.,0.,-2.,-2.,-2.]))
        def render(p):
            incoming=incident if mode=="dual_path" else torch.zeros_like(incident)
            return dual_path_direct(p[:3].sigmoid(),light,F.softplus(p[3:]),incoming,outgoing)
        log=fit(parameter,lambda p:render(p)[train],target[train])
        results[mode]=dict(test_mse=float((render(parameter)[~train]-target[~train]).square().mean()),
            fitted_reflectance=parameter[:3].sigmoid().detach().tolist(),fitted_beta=F.softplus(parameter[3:]).detach().tolist(),training=log)
        torch.save(parameter.detach(),output/(mode+".pt"))
    return dict(candidate=5,models=results,real_data_status="Missing known incident illumination/path observations; synthetic known-light test only")


def radiometry(output):
    clear=.05+.8*torch.rand(240,3)
    distance=.1+5*torch.rand(240)
    beta=torch.tensor([.4,.2,.1]); ambient=torch.tensor([.1,.3,.4])
    target=linear_to_srgb(homogeneous_observation(clear,beta,ambient,distance))
    train=torch.arange(240)%4!=0
    result={}
    for mode in ("encoded_domain","linear_then_srgb"):
        parameter=torch.nn.Parameter(torch.tensor([-2.,-2.,-2.,-1.,-1.,-1.]))
        def render(p):
            c=linear_to_srgb(clear) if mode=="encoded_domain" else clear
            value=homogeneous_observation(c,F.softplus(p[:3]),p[3:].sigmoid(),distance)
            return value if mode=="encoded_domain" else linear_to_srgb(value)
        log=fit(parameter,lambda p:render(p)[train],target[train])
        result[mode]=dict(test_mse=float((render(parameter)[~train]-target[~train]).square().mean()),training=log)
        torch.save(parameter.detach(),output/(mode+".pt"))
    return dict(candidate=6,models=result,real_data_status="No RAW/EXIF/calibrated response in 88 source PNGs; known synthetic sRGB is not verified real camera response")


def spectral(output):
    spectra=.05+.9*torch.rand(240,6)
    sensor=torch.tensor([[.45,.35,.15,.04,.01,0.],[.02,.12,.4,.35,.1,.01],[0.,.01,.04,.15,.35,.45]])
    beta=torch.tensor([.7,.45,.25,.15,.10,.08])
    distance=torch.linspace(.1,6.,240)
    target=spectral_direct(spectra,beta,distance,sensor)
    train=distance<4.
    result={}
    for mode in ("rgb_exponential","spectral_integral"):
        parameter=torch.nn.Parameter(torch.ones(3 if mode=="rgb_exponential" else 6)*-1.5)
        def render(p):
            if mode=="rgb_exponential": return (spectra@sensor.T)*(-F.softplus(p)*distance[:,None]).exp()
            return spectral_direct(spectra,F.softplus(p),distance,sensor)
        log=fit(parameter,lambda p:render(p)[train],target[train])
        result[mode]=dict(extrapolation_test_mse=float((render(parameter)[~train]-target[~train]).square().mean()),
            fitted_beta=F.softplus(parameter).detach().tolist(),training=log)
        torch.save(parameter.detach(),output/(mode+".pt"))
    return dict(candidate=7,models=result,real_data_status="Known synthetic spectra and sensor only; real sensor response and spectral constraints absent")


def polarimetry(output):
    angles=torch.arange(4)*math.pi/4
    object_signature=torch.tensor([1.,.08,-.04])
    medium_signature=torch.tensor([1.,.45,.2])
    design=torch.stack((polarization_measurement(object_signature,angles),polarization_measurement(medium_signature,angles)),-1)
    intensities=.1+.6*torch.rand(200,2)
    measurements=intensities@design.T+.001*torch.randn(200,4)
    recovered=measurements@torch.linalg.pinv(design).T
    stokes=recover_linear_stokes(measurements,angles)
    np.savez_compressed(output/"polarized_observations.npz",angles=angles.numpy(),measurements=measurements.numpy(),truth=intensities.numpy(),estimated=recovered.numpy())
    return dict(candidate=8,known_signature_design_rank=int(torch.linalg.matrix_rank(design)),
        ordinary_intensity_design_rank=1,decomposition_rmse=float((recovered-intensities).square().mean().sqrt()),
        stokes_rmse=float((stokes-(intensities[:,0,None]*object_signature+intensities[:,1,None]*medium_signature)).square().mean().sqrt()),
        limitation="Object and medium polarization signatures were known. Unknown signatures remain ambiguous; four polarizer angles alone do not guarantee separation.",
        real_data_status="No polarized source images; real training cannot be completed with available RGB")


def ambiguity(output):
    true_clear,true_beta,true_background=.72,.32,.18
    distances=torch.tensor([.5,1.,2.,4.],dtype=torch.float64)
    observed=true_background+(true_clear-true_background)*(-true_beta*distances).exp()
    values,beta,background=admissible_clear_grid(observed[2],2.,torch.linspace(.01,1.,600,dtype=torch.float64),
        torch.linspace(0,1,600,dtype=torch.float64),1e-8)
    prediction=background[:,None]+(values-background)[:,None]*(-beta[:,None]*distances).exp()
    valid=(prediction-observed).abs().max(-1).values<=.002
    assert valid.any()
    np.savez_compressed(output/"feasible_solutions.npz",clear=values.numpy(),beta=beta.numpy(),background=background.numpy(),multiview_valid=valid.numpy())
    return dict(candidate=9,single_view_clear_range=[float(values.min()),float(values.max())],
        four_view_clear_range=[float(values[valid].min()),float(values[valid].max())],true_clear=true_clear,
        feasible_grid_points_single=values.numel(),feasible_grid_points_four=int(valid.sum()),
        real_data_status="Scalar homogeneous synthetic identifiability demonstration only; no calibrated full-Gaussian uncertainty claim",
        limitation="Finite feasible grid within specified parameter bounds, not a posterior or proof of global extrema")


def acquisition(output):
    distances=torch.linspace(.3,6.,32,dtype=torch.float64)
    nominal=torch.tensor([.7,.3,.2],dtype=torch.float64)
    jacobian=observation_jacobian(*nominal,distances)
    selected,information=select_informative_distances(jacobian,6)
    uniform=np.linspace(0,31,6).round().astype(int).tolist()
    rng=np.random.default_rng(42)
    results={}
    for name,indices in (("uniform",uniform),("d_optimal",selected)):
        d=distances[indices].numpy()
        truth=nominal.numpy()
        clean=truth[2]+(truth[0]-truth[2])*np.exp(-truth[1]*d)
        errors=[]
        noise=np.random.default_rng(7).normal(0,.01,(100,len(d)))
        for sample in noise:
            result=least_squares(lambda p:p[2]+(p[0]-p[2])*np.exp(-p[1]*d)-(clean+sample),
                x0=[.6,.2,.3],bounds=([0,.001,0],[1,2,1]),max_nfev=500)
            errors.append(result.x-truth)
        errors=np.asarray(errors)
        results[name]=dict(indices=indices,distances=d.tolist(),clear_rmse=float(np.sqrt(np.mean(errors[:,0]**2))),
                           parameter_rmse=np.sqrt(np.mean(errors**2,axis=0)).tolist())
    return dict(candidate=10,designs=results,noise_sigma=.01,repetitions=100,
        real_data_status="No acquisition hardware/new calibrated views; simulated equal-budget acquisition only",
        limitation="Design uses local nominal parameters and known homogeneous model; robustness to real illumination and model mismatch is untested")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",type=int,choices=(3,5,6,7,8,9,10),required=True)
    parser.add_argument("--output-root",type=Path,default=ROOT/"outputs/transport_candidates_20260911")
    args=parser.parse_args()
    check_environment()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    output=args.output_root/(f"{args.candidate:02d}_synthetic_calibrated")
    output.mkdir(parents=True,exist_ok=False)
    function={3:refraction,5:dual_path,6:radiometry,7:spectral,8:polarimetry,9:ambiguity,10:acquisition}[args.candidate]
    result=function(output)
    result.update(complete=True,synthetic_only=True,real_four_scene_validation=False,sota_claim=False,
                  environment=os.environ["CONDA_DEFAULT_ENV"],seed=42)
    write_json(output/"result.json",result)
    print(json.dumps(result),flush=True)
