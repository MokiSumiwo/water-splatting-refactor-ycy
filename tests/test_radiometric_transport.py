import torch

from water_splatting.rendering.radiometric_transport import FrozenRadiometricTransport


def data():
    return dict(beta=torch.rand(5,3),bs=torch.rand(5,3),ambient=torch.rand(5,3),
                colors=torch.rand(5,4,3),depths=torch.rand(5,4)*4,weights=torch.rand(5,4)*.2)


def test_identity_matches_discrete_medium_formula():
    d=data();model=FrozenRadiometricTransport()
    actual,clear=model(d)
    w=d["weights"][...,None];z=d["depths"][...,None]
    expected=(w*(d["colors"]*(-d["beta"][:,None]*z).exp()+d["ambient"][:,None]*(1-(-d["bs"][:,None]*z).exp()))).sum(1)+(1-w.sum(1))*d["ambient"]
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(clear,(w*d["colors"]).sum(1))


def test_zero_scattering_does_not_modify_clear():
    d=data();d["beta"].zero_();d["bs"].zero_();d["ambient"].zero_()
    model=FrozenRadiometricTransport()
    value,clear=model(d)
    torch.testing.assert_close(value,clear,atol=1e-7,rtol=1e-6)


def test_global_calibration_gradients_finite():
    for linear,appearance in ((False,False),(True,False),(False,True)):
        model=FrozenRadiometricTransport(linear,appearance)
        prediction,clear=model(data())
        (prediction.square().mean()+clear.mean()).backward()
        assert torch.isfinite(model.raw.grad).all()
        assert (model.raw.grad.abs()>0).all()


def test_feasible_family_identity_preserves_bounded_color_endpoints():
    d=data()
    d["colors"][0,0]=torch.tensor([0.,1.,.5])
    reference=FrozenRadiometricTransport()(d)
    candidate=FrozenRadiometricTransport(appearance_shift=True)(d)
    for actual,expected in zip(candidate,reference):
        torch.testing.assert_close(actual,expected,atol=2e-7,rtol=1e-6)


def test_feasible_family_cannot_create_clear_coverage():
    d=data()
    d["weights"].zero_()
    model=FrozenRadiometricTransport(appearance_shift=True)
    with torch.no_grad():
        model.raw.fill_(3)
    _,clear=model(d)
    assert torch.equal(clear,torch.zeros_like(clear))
