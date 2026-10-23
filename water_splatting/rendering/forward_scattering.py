"""Visible-layer forward-scattering approximation, independent of clear RGB."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor,nn

SIGMAS=(0.,1.,2.,4.,8.)


def blur_layers(layers: Tensor, sigma: float):
    """Separable normalized Gaussian PSF with zero image-boundary padding."""
    if sigma==0: return layers
    radius=int(3*sigma+.5)
    coordinate=torch.arange(-radius,radius+1,device=layers.device,dtype=layers.dtype)
    kernel=(-.5*(coordinate/sigma).square()).exp()
    kernel=kernel/kernel.sum()
    channels=layers.shape[0]*layers.shape[1]
    image=layers.reshape(1,channels,*layers.shape[-2:])
    vertical=kernel.reshape(1,1,-1,1).expand(channels,1,-1,1)
    horizontal=kernel.reshape(1,1,1,-1).expand(channels,1,1,-1)
    return F.conv2d(F.conv2d(image,vertical,padding=(radius,0),groups=channels),
                    horizontal,padding=(0,radius),groups=channels).reshape_as(layers)


class ScatteringPSF(nn.Module):
    """Two parameters shared across all cameras: scattered fraction and width."""

    def __init__(self,relative_depths: Tensor,depth_dependent: bool):
        super().__init__()
        self.raw=nn.Parameter(torch.tensor([-4.,-1.],device=relative_depths.device))
        self.register_buffer("relative_depths",relative_depths)
        self.depth_dependent=depth_dependent

    def kernel_weights(self):
        sigma=8*self.raw[1].sigmoid()
        scale=self.relative_depths if self.depth_dependent else torch.ones_like(self.relative_depths)
        width=(sigma*scale).clamp(0,8)
        nodes=width.new_tensor(SIGMAS)
        left=torch.searchsorted(nodes,width,right=True).sub(1).clamp(0,len(SIGMAS)-2)
        right=left+1
        fraction=(width-nodes[left])/(nodes[right]-nodes[left])
        weights=width.new_zeros(width.numel(),len(SIGMAS))
        weights=weights.scatter_add(1,left[:,None],(1-fraction)[:,None])
        weights=weights.scatter_add(1,right[:,None],fraction[:,None])
        strength=self.raw[0].sigmoid()
        identity=torch.zeros_like(weights);identity[:,0]=1
        return (1-strength)*identity+strength*weights

    def forward(self,basis: Tensor):
        """basis is rays x layers x PSF widths x RGB."""
        return (basis*self.kernel_weights()[None,:,:,None]).sum((1,2))
