"""GCRC contamination factors, history identity and appearance boundaries."""

import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from water_splatting.gcrc import GaussianContaminationState, GCRC_FIELDS, apply_gcrc, compute_gcrc_contamination, validate_config
from water_splatting.taba import compute_taba_dcu


def example(n=1):
    d = torch.tensor([[1., -1., 0.]], dtype=torch.float64).repeat(n, 1)
    w = torch.tensor([[.2, -.1, .1]], dtype=torch.float64).repeat(n, 1)
    return d, w, GaussianContaminationState(d)


def observe(s, d, w, count=5):
    for step in range(1, count+1):
        s.update(d, w, torch.ones(len(d), dtype=torch.bool), torch.ones(len(d), 1, dtype=torch.bool), step=step)


def score(s, d, w, **kwargs):
    return compute_gcrc_contamination(d, w, torch.full_like(d, .2), torch.ones(len(d), 1, dtype=torch.bool), s.values(), step=kwargs.pop('step', 3000), **kwargs)


@pytest.mark.parametrize('prefix', ['water', 'residual'])
def test_direction_ema(prefix):
    d, w, s = example()
    observe(s, d, w, count=1)
    old = getattr(s, prefix+'_direction_prototype').clone()
    d2, w2 = d.roll(1, -1), w.roll(1, -1)
    s.update(d2, w2, torch.tensor([True]), torch.tensor([[True]]), step=2)
    raw = w2 if prefix=='water' else d2
    expected = .95*old+.05*raw/(raw.norm(dim=-1, keepdim=True)+1e-8)
    expected /= expected.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(getattr(s, prefix+'_direction_prototype'), expected)
    assert getattr(s, prefix+'_direction_initialized').all()


@pytest.mark.parametrize('bad', [0., float('nan'), float('inf')])
@pytest.mark.parametrize('prefix', ['water', 'residual'])
def test_invalid_observation_preserves_prototype(prefix, bad):
    d, w, s = example()
    observe(s, d, w, count=1)
    old = getattr(s, prefix+'_direction_prototype').clone()
    if prefix=='water': w.fill_(bad)
    else: d.fill_(bad)
    s.update(d, w, torch.tensor([True]), torch.tensor([[True]]), step=2)
    assert torch.equal(getattr(s, prefix+'_direction_prototype'), old)
    assert all(torch.isfinite(v).all() for v in s.buffers())


def test_visibility_mask_and_once_per_step():
    d, w, s = example(3)
    for _ in range(2):
        s.update(d, w, torch.tensor([True, False, True]), torch.tensor([[True], [True], [False]]), step=1)
    assert s.visibility_count.flatten().tolist()==[1, 0, 1]
    assert s.water_direction_initialized.flatten().tolist()==[True, False, False]
    assert s.residual_direction_initialized.flatten().tolist()==[True, False, True]


@pytest.mark.parametrize('reason', ['step', 'count', 'water', 'residual', 'transmission', 'water_nan', 'zero_residual', 'corrupt_water'])
def test_fallback_is_exact(reason):
    d, w, s = example()
    observe(s, d, w)
    t = torch.full_like(d, .1)
    current = -d
    step = 3000
    if reason=='step': step=2999
    if reason=='count': s.visibility_count.fill_(3)
    if reason=='water': s.water_direction_initialized.zero_()
    if reason=='residual': s.residual_direction_initialized.zero_()
    if reason=='transmission': t.fill_(float('nan'))
    if reason=='water_nan': w.fill_(float('nan'))
    if reason=='zero_residual': current=torch.zeros_like(d)
    if reason=='corrupt_water': s.water_direction_prototype.fill_(float('nan'))
    r = compute_gcrc_contamination(current, w, t, torch.tensor([[True]]), s.values(), step=step)
    assert r['alpha'].item()==0 and not r['active'].item()
    assert all(torch.isfinite(v).all() for v in r.values())
    out = apply_gcrc(current, torch.zeros_like(d), r)
    assert torch.equal(out['s_adjusted'], current)
    assert torch.equal(out['rgb'], torch.sigmoid(current))


def test_six_factor_target_ranges_and_detach():
    d, w, s = example(4)
    observe(s, d, w)
    r = score(s, -d.requires_grad_(), w.requires_grad_())
    keys = ('low_transmission','water_strength','water_axis_consistency','water_alignment','residual_instability','visibility_support')
    expected = torch.ones_like(r['alpha'])
    for key in keys:
        assert ((r[key]>=0) & (r[key]<=1)).all()
        expected *= r[key]
    torch.testing.assert_close(r['contamination'], expected)
    torch.testing.assert_close(r['alpha'], .5*expected)
    assert r['alpha'].min()>0 and r['alpha'].max()<=.5
    assert all(not v.requires_grad for v in r.values())
    assert not list(s.parameters())


def test_any_zero_factor_prevents_correction():
    d, w, s = example()
    observe(s, d, w)
    # Perfectly matching residual direction has only epsilon-sized instability.
    consistent = score(s, d, w)
    assert consistent['alpha'].item()<1e-8
    high_t = compute_gcrc_contamination(-d, w, torch.ones_like(d), torch.tensor([[True]]), s.values(), step=3000)
    assert high_t['alpha'].item()==0
    orthogonal = torch.linalg.cross(s.water_direction_prototype, d)
    assert score(s, orthogonal, w)['water_alignment'].item()<1e-12


def test_parallel_projection_and_gradient_contract():
    d, w, s = example()
    observe(s, d, w)
    dc = torch.tensor([[.4, -.2, .1]], dtype=torch.float64, requires_grad=True)
    rest = (-d).clone().requires_grad_()
    r = score(s, rest, w)
    r['alpha'].requires_grad_()
    r['water_dir'].requires_grad_()
    out = apply_gcrc(dc+rest, dc, r)
    axis = r['water_dir'].detach()
    parallel = (rest*axis).sum(-1, keepdim=True)*axis
    torch.testing.assert_close(out['s_adjusted'], dc+rest-r['alpha'].detach()*parallel)
    correction = out['s_adjusted']-(dc+rest)
    torch.testing.assert_close(correction-(correction*axis).sum(-1, keepdim=True)*axis, torch.zeros_like(correction), atol=1e-12, rtol=0)
    assert out['decomposition_error'].max()<1e-12
    grads = torch.autograd.grad(out['rgb'].square().sum(), (dc, rest, r['alpha'], r['water_dir']), allow_unused=True)
    assert grads[0].abs().sum()>0 and grads[1].abs().sum()>0 and grads[2] is None and grads[3] is None


def test_raw_rgb_axis_mean_shift_is_reported_not_hidden():
    d, w, s = example()
    observe(s, d, w)
    r = score(s, -d, w)
    out = apply_gcrc(-d, torch.zeros_like(d), r)
    assert out['luminance_logit_change'].abs().item()>0
    torch.testing.assert_close(out['luminance_logit_change'], (out['s_adjusted']+d).mean(-1,keepdim=True))


def test_zero_alpha_preserves_rounded_endpoints_and_active_is_strict():
    full = torch.tensor([[-1e4, 0., 1e4]])
    axis = torch.tensor([[0., 1., 0.]])
    cold = apply_gcrc(full, torch.zeros_like(full), {'alpha':torch.zeros(1,1), 'water_dir':axis})
    assert torch.equal(cold['rgb'], full.sigmoid())
    active = apply_gcrc(full, torch.zeros_like(full), {'alpha':torch.full((1,1),.5), 'water_dir':axis})
    assert ((active['rgb']>0) & (active['rgb']<1)).all()


def test_combination_is_gcrc_then_existing_taba_and_zero_is_exact():
    d, w, s = example()
    observe(s, d, w)
    full, base, tau = -d, torch.zeros_like(d), torch.ones_like(d)
    taba = compute_taba_dcu(full, base, tau, base, w, mode='directional')
    r = score(s, full, w)
    corrected = apply_gcrc(full, base, r)
    expected = compute_taba_dcu(corrected['s_adjusted'], base, tau, base, w, mode='directional')
    combined = apply_gcrc(full, base, r, taba=taba)
    torch.testing.assert_close(combined['s_adjusted'], expected['s_adjusted'])
    torch.testing.assert_close(combined['rgb'], expected['rgb'])
    r['alpha'].zero_()
    cold = apply_gcrc(full, base, r, taba=taba)
    assert torch.equal(cold['s_adjusted'], taba['s_adjusted']) and torch.equal(cold['rgb'], taba['rgb'])


def test_history_topology_and_checkpoint_roundtrip():
    d, w, s = example(3)
    w = w*torch.tensor([[1.],[-1.],[2.]])
    observe(s, d, w)
    s.visibility_count.copy_(torch.tensor([[2],[4],[5]]))
    saved = {k:v.clone() for k,v in s.values().items()}
    s.densify(torch.tensor([True,False,True]), 2, torch.tensor([False,True,False]))
    lineage = torch.tensor([0,1,2,0,2,0,2,1])
    removed = torch.tensor([True,False,True,False,False,False,False,False])
    s.prune(removed)
    for key in GCRC_FIELDS:
        assert torch.equal(getattr(s,key), saved[key][lineage[~removed]])
    buffer = io.BytesIO()
    torch.save(s.state_dict(), buffer)
    buffer.seek(0)
    restored = GaussianContaminationState(torch.zeros(6,3,dtype=torch.float64))
    restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in s.state_dict().items())
    restored.float()
    assert restored.water_direction_prototype.dtype==torch.float32
    assert restored.visibility_count.dtype==torch.int64 and restored.water_direction_initialized.dtype==torch.bool


def test_actual_model_refinement_aligns_gcrc_history_and_adam():
    import test_grac
    from water_splatting.water_splatting import WaterSplattingModel
    original = WaterSplattingModel.refinement_after
    captured = {}
    def refinement(model, *args, **kwargs):
        d,w,s = example(3)
        s=s.float()
        observe(s,d.float(),w.float())
        s.visibility_count.copy_(torch.tensor([[1],[2],[3]]))
        model._gcrc_state=s
        captured['model']=model
        captured['before']={k:v.clone() for k,v in s.values().items()}
        return original(model,*args,**kwargs)
    with patch.object(WaterSplattingModel,'refinement_after',refinement):
        test_grac.test_model_refinement_keeps_history_and_adam_aligned()
    model=captured['model']
    lineage=torch.tensor([1,0,0,1])
    for key in GCRC_FIELDS:
        assert torch.equal(getattr(model._gcrc_state,key),captured['before'][key][lineage])


def test_empty_history_and_configuration():
    d,w,s=example(0)
    observe(s,d,w)
    assert score(s,d,w)['alpha'].shape==(0,1)
    validate_config(SimpleNamespace())
    with pytest.raises(ValueError): validate_config(SimpleNamespace(gcrc_water_strength_scale=1e-5))
    with pytest.raises(RuntimeError): s.check_shape(1)
