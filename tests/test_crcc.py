"""CRCC numerical, state-lineage and appearance-gradient contracts."""

import io
from types import SimpleNamespace

import pytest
import torch

from water_splatting.crcc import ClearRadianceHistory, CRCC_FIELDS, clear_consistency, loss_coefficient
from water_splatting.fields.gaussian_appearance import safe_sigmoid
from water_splatting.taba import sample_taba_transmission


def history(n=2):
    return ClearRadianceHistory(torch.zeros(n, 3, dtype=torch.float64))


def observe(state, chroma, step):
    mask = torch.ones(chroma.shape[0], 1, dtype=torch.bool)
    state.update(chroma, mask, mask, step=step)


def loss(rgb, state, **kwargs):
    n = rgb.shape[0]
    return clear_consistency(rgb, state.values(), torch.full_like(rgb, .4),
                             torch.ones(n, dtype=torch.bool), torch.ones(n, 1, dtype=torch.bool), **kwargs)


def test_first_and_following_observations():
    state = history()
    a = torch.tensor([[.2,-.2,0.],[.1,.2,-.3]], dtype=torch.float64)
    observe(state, a, 1)
    assert torch.equal(state.crcc_chroma_ema, a)
    assert state.crcc_initialized.all() and (state.crcc_observation_count == 1).all()
    observe(state, -a, 2)
    torch.testing.assert_close(state.crcc_chroma_ema, .9*a)
    assert (state.crcc_observation_count == 2).all()
    observe(state, a, 2)
    assert (state.crcc_observation_count == 2).all()


@pytest.mark.parametrize("invalid", [float('nan'), float('inf'), -float('inf')])
def test_invalid_radiance_never_updates(invalid):
    state = history()
    observe(state, torch.zeros(2,3), 1)
    observe(state, torch.full((2,3),invalid), 2)
    assert (state.crcc_observation_count == 1).all()
    assert not state.crcc_chroma_ema.any()
    rgb = torch.full((2,3), invalid, requires_grad=True)
    out = loss(rgb,state)
    assert out['loss'].item() == 0
    out['loss'].backward()
    assert torch.isfinite(rgb.grad).all() and not rgb.grad.any()


def test_invisible_and_invalid_renderer_observations_are_excluded():
    state = history(3)
    state.update(torch.zeros(3,3), torch.tensor([True,False,True]), torch.tensor([True,True,False]), step=1)
    assert state.crcc_observation_count.flatten().tolist() == [1,0,0]
    assert state.crcc_initialized.flatten().tolist() == [True,False,False]


@pytest.mark.parametrize("count", [0,1,2,8])
def test_prior_observation_threshold_and_support(count):
    state = history()
    for step in range(count):
        observe(state, torch.zeros(2,3), step)
    rgb = torch.tensor([[.2,.3,.6],[.8,.1,.2]],requires_grad=True)
    out = loss(rgb,state)
    torch.testing.assert_close(out['support'], torch.full_like(out['support'], 1-torch.exp(torch.tensor(-count/4.)).item()))
    assert bool(out['active'].all()) == (count >= 2)
    assert bool(out['loss'] > 0) == (count >= 2)


def test_charbonnier_weighted_normalization_and_single_medium_floor():
    state = history()
    observe(state, torch.zeros(2,3), 1)
    observe(state, torch.zeros(2,3), 2)
    rgb = torch.tensor([[.2,.3,.7],[.8,.4,.1]], dtype=torch.float64, requires_grad=True)
    transmission = torch.tensor([[0.,0.,0.],[1.,1.,1.]],dtype=torch.float64)
    out = clear_consistency(rgb,state.values(),transmission,torch.ones(2,dtype=torch.bool),torch.ones(2,1,dtype=torch.bool))
    assert out['medium_weight'].flatten().tolist() == [1.,.25]
    residual = rgb-rgb.mean(-1,keepdim=True)
    rho = (residual.square()+1e-6).sqrt().mean(-1,keepdim=True)
    expected = (out['weight']*rho).sum()/(out['weight'].sum()+1e-8)
    torch.testing.assert_close(out['loss'],expected)
    out['loss'].backward()
    torch.testing.assert_close(rgb.grad.sum(-1),torch.zeros(2,dtype=rgb.dtype),atol=1e-12,rtol=0)


def test_auxiliary_inputs_and_ema_do_not_receive_gradient():
    state = history()
    observe(state,torch.zeros(2,3),1)
    observe(state,torch.zeros(2,3),2)
    state.crcc_chroma_ema.requires_grad_(True)
    dc = torch.tensor([[.2,.1,-.4],[.3,-.2,.8]],requires_grad=True)
    rest = torch.tensor([[.5,-.4,.7],[-.8,.4,.1]],requires_grad=True)
    t = torch.full_like(dc,.3,requires_grad=True)
    alpha = torch.ones(2,1,requires_grad=True)
    out = clear_consistency(safe_sigmoid(dc+rest),state.values(),t,torch.ones(2,dtype=torch.bool),torch.ones(2,1,dtype=torch.bool),alpha=alpha)
    gradients = torch.autograd.grad(out['loss'],(dc,rest,t,alpha,state.crcc_chroma_ema),allow_unused=True)
    assert gradients[0].abs().sum()>0 and gradients[1].abs().sum()>0
    assert gradients[2:] == (None,None,None)
    assert all(not value.requires_grad for key,value in out.items() if key != 'loss')


@pytest.mark.parametrize("reason", ['empty','invisible','invalid_transmission','invalid_history','mismatch','alpha_zero'])
def test_zero_weight_fallback_has_finite_zero_gradient(reason):
    n = 0 if reason == 'empty' else 2
    state = history(n)
    for step in (1,2):
        observe(state,torch.zeros(n,3),step)
    rgb = torch.full((n,3),.5,requires_grad=True)
    t = torch.full((n,3),.3)
    visible = torch.ones(n,dtype=torch.bool)
    alpha = None
    if reason == 'invisible': visible.zero_()
    if reason == 'invalid_transmission': t.fill_(float('nan'))
    if reason == 'invalid_history': state.crcc_chroma_ema.fill_(float('inf'))
    if reason == 'mismatch': state = history(n+1)
    if reason == 'alpha_zero': alpha = torch.zeros(n,1)
    out = clear_consistency(rgb,state.values(),t,visible,torch.ones(n,1,dtype=torch.bool),alpha=alpha)
    assert out['loss'].item() == 0 and torch.isfinite(out['loss'])
    out['loss'].backward()
    assert torch.isfinite(rgb.grad).all() and not rgb.grad.any()


@pytest.mark.parametrize("step,expected",[(0,0),(2999,0),(3000,0),(3500,.005),(4000,.01),(5000,.02),(14999,.02)])
def test_schedule(step,expected):
    assert loss_coefficient(step) == expected


def test_no_ramp_schedule_and_sigmoid_bounds():
    assert loss_coefficient(2999,ramp_steps=0)==0
    assert loss_coefficient(3000,ramp_steps=0)==.02
    rgb = safe_sigmoid(torch.tensor([[-1e6,0.,1e6]]))
    assert ((rgb>0)&(rgb<1)).all()


def test_transmission_sampling_reuses_supplied_renderer_tensors():
    attn = torch.ones(2,2,3,requires_grad=True)
    depth = torch.ones(2,2,1,requires_grad=True)
    tau = torch.full((2,2,3),2.,requires_grad=True)
    t = torch.full((2,2,3),.2,requires_grad=True)
    out = sample_taba_transmission(attn,depth,torch.ones_like(depth),torch.tensor([[1.,1.]]),torch.ones(1),tau_D=tau,transmission_rgb=t)
    assert torch.equal(out['transmission_rgb'],t[1,1][None])
    assert torch.equal(out['tau_D'],tau[1,1][None])
    assert all(not v.requires_grad for v in out.values())


def test_state_lineage_roundtrip_and_device_dtype():
    state = history(3)
    c = torch.tensor([[.1,-.1,0.],[.2,-.2,0.],[.3,-.3,0.]],dtype=torch.float64)
    observe(state,c,1)
    original = {k:v.clone() for k,v in state.values().items()}
    state.densify(torch.tensor([True,False,True]),2,torch.tensor([False,True,False]))
    lineage = torch.tensor([0,1,2,0,2,0,2,1])
    remove = torch.tensor([True,False,True,False,False,False,False,False])
    state.prune(remove)
    for key in CRCC_FIELDS:
        assert torch.equal(getattr(state,key),original[key][lineage[~remove]])
    buffer = io.BytesIO()
    torch.save(state.state_dict(),buffer)
    buffer.seek(0)
    restored = history(6)
    restored.load_state_dict(torch.load(buffer,weights_only=True))
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in state.state_dict().items())
    restored.to(dtype=torch.float32)
    assert restored.crcc_chroma_ema.dtype==torch.float32
    assert restored.crcc_observation_count.dtype==torch.int64
    assert restored.crcc_initialized.dtype==torch.bool
    assert not list(restored.parameters())


def test_real_refinement_preserves_crcc_lineage_and_adam(monkeypatch):
    from water_splatting.water_splatting import WaterSplattingModel
    from test_grac import test_model_refinement_keeps_history_and_adam_aligned
    original_refine = WaterSplattingModel.refinement_after
    original_cull = WaterSplattingModel.cull_gaussians
    saved = {}
    def refine(model,optimizers,step):
        model._crcc_state = ClearRadianceHistory(model.features_dc)
        observe(model._crcc_state,torch.tensor([[.1,-.1,0.],[.2,-.2,0.],[.3,-.3,0.]]),1)
        saved.update({key:value.clone() for key,value in model._crcc_state.values().items()})
        original_refine(model,optimizers,step)
        for key in CRCC_FIELDS:
            assert torch.equal(getattr(model._crcc_state,key),saved[key][torch.tensor([1,2,0,0,1])])
    def cull(model,extra_cull_mask=None):
        before = {key:value.clone() for key,value in model._crcc_state.values().items()}
        removed = original_cull(model,extra_cull_mask)
        for key in CRCC_FIELDS:
            assert torch.equal(getattr(model._crcc_state,key),before[key][~removed])
        return removed
    monkeypatch.setattr(WaterSplattingModel,'refinement_after',refine)
    monkeypatch.setattr(WaterSplattingModel,'cull_gaussians',cull)
    test_model_refinement_keeps_history_and_adam_aligned()


def test_training_gradient_addition_preserves_other_routes():
    from scripts.experiments.run_direct_mdrr_cica_scene import _add_crcc_gradients
    dc = torch.nn.Parameter(torch.ones(3))
    rest = torch.nn.Parameter(torch.ones(3))
    medium = torch.nn.Parameter(torch.ones(3))
    params = [dc,rest,medium]
    for p in params: p.grad=torch.full_like(p,7.)
    groups = dict(features_dc=[0],features_rest=[1],medium_mlp=[2])
    _add_crcc_gradients(dc.sum()+2*rest.sum(),params,groups)
    assert torch.equal(dc.grad,torch.full_like(dc,8.))
    assert torch.equal(rest.grad,torch.full_like(rest,9.))
    assert torch.equal(medium.grad,torch.full_like(medium,7.))
    with pytest.raises(RuntimeError,match='leaked'):
        _add_crcc_gradients(medium.sum(),params,groups)
