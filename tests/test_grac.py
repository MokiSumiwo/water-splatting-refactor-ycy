"""Reliability history, appearance routing and topology contracts."""

import io
from types import SimpleNamespace

import pytest
import torch

from water_splatting.grac import GaussianReliabilityState, GRAC_FIELDS, apply_grac, compute_grac_reliability
from water_splatting.taba import compute_taba_dcu


def history(n=3, dtype=torch.float64):
    return GaussianReliabilityState(torch.zeros(n, 3, dtype=dtype))


def observe(state, d, start=1, count=5):
    for step in range(start, start+count):
        state.update(d, torch.ones(d.shape[0], dtype=torch.bool), step=step)


def test_direction_ema_uses_normalized_previous_prototype():
    s = history(1)
    d1 = torch.tensor([[1., -1., 0.]], dtype=torch.float64)
    d2 = torch.tensor([[0., 1., -1.]], dtype=torch.float64)
    observe(s, d1, count=1)
    old = s.direction_prototype.clone()
    s.update(d2, torch.tensor([True]), step=2)
    expected = .95*old + .05*d2/(d2.norm(dim=-1, keepdim=True)+1e-8)
    expected /= expected.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(s.direction_prototype, expected)
    assert s.reliability_initialized_mask.all()


@pytest.mark.parametrize("bad", [0., float("nan"), float("inf")])
def test_zero_or_invalid_residual_does_not_update_prototype(bad):
    s = history(1)
    observe(s, torch.tensor([[1., -1., 0.]], dtype=torch.float64), count=1)
    before = s.direction_prototype.clone()
    s.update(torch.full((1, 3), bad), torch.tensor([True]), step=2)
    assert torch.equal(before, s.direction_prototype)
    assert s.visibility_count.item() == 2
    assert all(torch.isfinite(v).all() for v in s.buffers())


def test_magnitude_first_sample_and_ema_variance():
    s = history(1)
    d = torch.tensor([[3., -3., 0.]], dtype=torch.float64)
    observe(s, d, count=1)
    a = d.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(s.magnitude_ema, a)
    s.update(2*d, torch.tensor([True]), step=2)
    torch.testing.assert_close(s.magnitude_ema, 1.05*a)
    torch.testing.assert_close(s.magnitude_sq_ema, 1.15*a.square())
    r = compute_grac_reliability(d, s.values(), step=3001)
    assert (r['variance'] >= 0).all()
    assert (r['magnitude_stability'] < 1).all()


def test_visibility_uses_supplied_mask_once_per_step():
    s = history()
    d = torch.ones(3, 3)
    for _ in range(2):
        s.update(d, torch.tensor([True, False, True]), step=3001)
    assert s.visibility_count.flatten().tolist() == [1, 0, 1]
    assert s.magnitude_count.flatten().tolist() == [1, 0, 1]


def test_cold_start_and_step_fallback():
    s = history(1)
    d = torch.tensor([[1., -1., 0.]], dtype=torch.float64)
    observe(s, d, count=3)
    r = compute_grac_reliability(-d, s.values(), step=3001)
    assert r['gate'].item() == 1 and not r['active'].item()
    observe(s, d, start=4, count=2)
    assert compute_grac_reliability(-d, s.values(), step=2999)['gate'].item() == 1
    r = compute_grac_reliability(-d, s.values(), step=3000)
    assert .35 <= r['gate'].item() < 1 and r['active'].item()
    assert r['direction_consistency'].item() < 1e-4


def test_uninitialized_history_does_not_penalize_direction_or_magnitude():
    s = history(1)
    r = compute_grac_reliability(torch.zeros(1, 3), s.values(), step=3000)
    assert r['direction_consistency'].item() == r['magnitude_stability'].item() == r['gate'].item() == 1


def test_reliability_formula_ranges_and_detach():
    s = history(2)
    d = torch.tensor([[1., -1., 0.], [0., 1., -1.]], dtype=torch.float64, requires_grad=True)
    observe(s, d)
    r = compute_grac_reliability(d, s.values(), step=3000)
    torch.testing.assert_close(r['pseudo_reliability'], .5*r['direction_consistency']+.3*r['magnitude_stability']+.2*r['visibility_confidence'])
    torch.testing.assert_close(r['visibility_confidence'], -torch.expm1(-s.visibility_count.to(d.dtype)/8.))
    for key in ('pseudo_reliability', 'direction_consistency', 'magnitude_stability', 'gate'):
        assert ((r[key] >= 0) & (r[key] <= 1)).all()
    assert all(not value.requires_grad for value in r.values())
    assert not list(s.parameters())


@pytest.mark.parametrize("key", ["direction_prototype", "magnitude_ema", "magnitude_sq_ema"])
def test_invalid_statistics_fall_back(key):
    s = history(1)
    d = torch.tensor([[1., -1., 0.]], dtype=torch.float64)
    observe(s, d)
    getattr(s, key).fill_(float('nan'))
    r = compute_grac_reliability(d, s.values(), step=3000)
    assert r['gate'].item() == 1
    assert all(torch.isfinite(x).all() for x in r.values())


def test_split_clone_prune_preserve_identity_and_roundtrip():
    s = history()
    observe(s, torch.tensor([[1., -1., 0.], [3., -3., 0.], [4., -4., 0.]], dtype=torch.float64))
    original = {key: v.clone() for key, v in s.values().items()}
    s.densify(torch.tensor([True, False, True]), 2, torch.tensor([False, True, False]))
    lineage = torch.tensor([0, 1, 2, 0, 2, 0, 2, 1])
    for key in GRAC_FIELDS:
        assert torch.equal(getattr(s, key), original[key][lineage])
    removed = torch.tensor([True, False, True, False, False, False, False, False])
    s.prune(removed)
    for key in GRAC_FIELDS:
        assert torch.equal(getattr(s, key), original[key][lineage[~removed]])
    buffer = io.BytesIO()
    torch.save(s.state_dict(), buffer)
    buffer.seek(0)
    restored = history(6)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert all(torch.equal(v, restored.state_dict()[key]) for key, v in s.state_dict().items())
    restored.to(dtype=torch.float32)
    assert restored.direction_prototype.dtype == torch.float32
    assert restored.visibility_count.dtype == torch.int64
    assert restored.reliability_initialized_mask.dtype == torch.bool


def test_appearance_gradients_luminance_and_bounds():
    dc = torch.tensor([[.3, -.1, .4]], requires_grad=True)
    rest = torch.tensor([[2., -1., .5]], requires_grad=True)
    gate = torch.tensor([[.6]], requires_grad=True)
    out = apply_grac(dc+rest, dc, {'gate':gate})
    torch.testing.assert_close((out['s_adjusted']-dc).mean(-1), rest.mean(-1))
    gradients = torch.autograd.grad(out['rgb'].square().sum(), (dc, rest, gate), allow_unused=True)
    assert gradients[0].abs().sum() > 0 and gradients[1].abs().sum() > 0 and gradients[2] is None
    extreme = torch.tensor([[-1e4, 0., 1e4]])
    rgb = apply_grac(extreme, torch.zeros_like(extreme), {'gate':torch.ones(1,1)})['rgb']
    assert ((rgb > 0) & (rgb < 1)).all()


def test_combined_gate_formula_and_cold_start_matches_taba():
    full = torch.tensor([[1., -1., .2]])
    base = torch.zeros_like(full)
    taba = compute_taba_dcu(full, base, torch.ones_like(full), base, torch.tensor([[1., 0., 0.]]), mode='directional')
    cold = apply_grac(full, base, {'gate':torch.ones(1,1), 'active':torch.zeros(1,1,dtype=torch.bool)}, taba=taba)
    assert torch.equal(cold['s_adjusted'], taba['s_adjusted'])
    r = {'gate':torch.tensor([[.6]]), 'active':torch.ones(1,1,dtype=torch.bool)}
    out = apply_grac(full, base, r, taba=taba)
    final = (taba['gate']*.6).clamp(.35,1)
    chroma = full-full.mean(-1,keepdim=True)
    expected = full + (.6-1)*chroma + (final-.6)*taba['delta_parallel']
    torch.testing.assert_close(out['s_adjusted'], expected)


def test_empty_state_and_shape_mismatch():
    s = history(0)
    s.update(torch.empty(0,3),torch.empty(0,dtype=torch.bool),step=1)
    assert compute_grac_reliability(torch.empty(0,3),s.values(),step=3000)['gate'].shape == (0,1)
    with pytest.raises(RuntimeError,match='mismatch'):
        s.check_shape(1)


def test_model_refinement_keeps_history_and_adam_aligned():
    from water_splatting.water_splatting import WaterSplattingModel
    model = WaterSplattingModel.__new__(WaterSplattingModel)
    torch.nn.Module.__init__(model)
    model.device_indicator_param = torch.nn.Parameter(torch.empty(0))
    model.config = SimpleNamespace(warmup_length=0, refine_every=1, reset_alpha_every=100,
        stop_split_at=100, densify_grad_thresh=.1, densify_size_thresh=.1, stop_screen_size_at=0,
        n_split_samples=2, cull_alpha_thresh=.01, cull_alpha_thresh_post=.01, cull_scale_thresh=10.,
        cull_screen_size=10., refinement_priority_mode='baseline')
    model.step, model.num_train_data = 10, 1
    model.gauss_params = torch.nn.ParameterDict({
        'means':torch.nn.Parameter(torch.arange(9.).reshape(3,3)),
        'features_dc':torch.nn.Parameter(torch.arange(9.).reshape(3,3)),
        'features_rest':torch.nn.Parameter(torch.zeros(3,15,3)),
        'scales':torch.nn.Parameter(torch.tensor([[.2,.2,.2],[.01,.01,.01],[.01,.01,.01]]).log()),
        'quats':torch.nn.Parameter(torch.tensor([[1.,0.,0.,0.]]).repeat(3,1)),
        'opacities':torch.nn.Parameter(torch.zeros(3,1)),
    })
    model._grac_state = history(3,dtype=torch.float32)
    observe(model._grac_state, torch.tensor([[1.,-1.,0.],[2.,-2.,0.],[3.,-3.,0.]]))
    saved = {k:v.clone() for k,v in model._grac_state.values().items()}
    model._bvam_valid_mask = model._refinement_guidance_hardness = model._refinement_guidance_brightness = None
    model.xys_grad_norm = torch.tensor([1.,1.,0.])
    model.vis_counts = torch.ones(3)
    model.max_2Dsize = torch.zeros(3)
    model.last_size = (10,10)
    optimizers = {}
    for key,p in model.gauss_params.items():
        optimizer = torch.optim.Adam([p])
        optimizer.state[p] = {'step':torch.tensor(1.),'exp_avg':torch.full_like(p,7.),'exp_avg_sq':torch.full_like(p,49.)}
        optimizers[key] = optimizer
    for key in ('medium_mlp', 'direction_encoding'):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter])
        optimizer.state[parameter] = {'step':torch.tensor(1.), 'exp_avg':torch.ones(1), 'exp_avg_sq':torch.ones(1)}
        optimizers[key] = optimizer
    model.refinement_after(SimpleNamespace(optimizers=optimizers),step=10)
    lineage = torch.tensor([1,2,0,0,1])
    assert model.num_points == 5
    for key in GRAC_FIELDS:
        assert torch.equal(getattr(model._grac_state,key),saved[key][lineage])
    for key,p in model.gauss_params.items():
        state = optimizers[key].state[p]
        assert state['exp_avg'].shape == p.shape
        assert (state['exp_avg'][:2] == 7).all() and (state['exp_avg'][2:] == 0).all()
    removed = model.cull_gaussians(torch.tensor([False,True,False,False,False]))
    model.remove_from_all_optim(SimpleNamespace(optimizers=optimizers),removed)
    assert model.num_points == 4
    for key in GRAC_FIELDS:
        assert torch.equal(getattr(model._grac_state,key),saved[key][lineage[~removed]])
