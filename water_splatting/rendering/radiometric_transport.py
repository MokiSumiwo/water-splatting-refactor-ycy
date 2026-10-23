"""Global response ablations on exact fixed discrete Gaussian contributions."""

from __future__ import annotations

import torch
from torch import Tensor,nn

from water_splatting.rendering.optical_measurements import linear_to_srgb,srgb_to_linear


class FrozenRadiometricTransport(nn.Module):
    def __init__(self,linear_srgb: bool=False,appearance_shift: bool=False):
        super().__init__()
        self.linear_srgb=linear_srgb
        self.appearance_shift=appearance_shift
        self.raw=nn.Parameter(torch.zeros(12 if appearance_shift else 9))

    def forward(self,data):
        delta=self.raw.tanh()
        beta=data["beta"]*delta[:3].exp()
        bs=data["bs"]*delta[3:6].exp()
        ambient=(torch.logit(data["ambient"].clamp(1e-7,1-1e-7))+delta[6:9]).sigmoid()
        colors=data["colors"]
        if self.appearance_shift:
            colors=(torch.logit(colors.clamp(1e-7,1-1e-7))+.5*delta[9:12]).sigmoid()
        if self.linear_srgb:
            colors=srgb_to_linear(colors)
            ambient=srgb_to_linear(ambient)
        weight=data["weights"][...,None]
        depth=data["depths"][...,None]
        direct=(weight*colors*(-beta[:,None]*depth).exp()).sum(1)
        medium=ambient*(1-(weight*(-bs[:,None]*depth).exp()).sum(1))
        clear=(weight*colors).sum(1)
        if self.linear_srgb:
            return linear_to_srgb(direct+medium),linear_to_srgb(clear)
        return direct+medium,clear
