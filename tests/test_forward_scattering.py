import torch

from water_splatting.rendering.forward_scattering import ScatteringPSF,blur_layers


def test_psf_flux_and_symmetry_for_interior_impulse():
    image=torch.zeros(1,3,65,65)
    image[:,:,32,32]=1
    blurred=blur_layers(image,4)
    torch.testing.assert_close(blurred.sum((-1,-2)),torch.ones(1,3))
    torch.testing.assert_close(blurred,blurred.flip(-1))
    torch.testing.assert_close(blurred,blurred.flip(-2))


def test_constant_depth_reduces_to_camera_psf():
    depths=torch.ones(3)
    a,b=ScatteringPSF(depths,True),ScatteringPSF(depths,False)
    torch.testing.assert_close(a.kernel_weights(),b.kernel_weights(),rtol=0,atol=0)
    assert torch.all(a.kernel_weights()>=0)
    torch.testing.assert_close(a.kernel_weights().sum(-1),torch.ones(3))


def test_depth_orders_expected_psf_width():
    model=ScatteringPSF(torch.tensor([.5,1.,2.]),True)
    weights=model.kernel_weights()
    expected=weights@torch.tensor([0.,1.,2.,4.,8.])
    assert (expected[1:]>expected[:-1]).all()
    basis=torch.rand(100,3,5,3)
    model(basis).sum().backward()
    assert torch.isfinite(model.raw.grad).all() and (model.raw.grad.abs()>0).all()
