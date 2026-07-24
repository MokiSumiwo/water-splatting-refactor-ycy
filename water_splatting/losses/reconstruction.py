"""Original WaterSplatting reconstruction loss."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module


def reconstruction_loss(
    *,
    gt_img: Tensor,
    pred_img: Tensor,
    main_loss: Literal["l1", "reg_l1", "reg_l2"],
    ssim_loss: Literal["reg_ssim", "ssim"],
    ssim_lambda: float,
    ssim_metric: Module,
) -> Tensor:
    """Compute the original combined reconstruction + SSIM loss."""

    if main_loss == "l1":
        recon_loss = torch.abs(gt_img - pred_img).mean()
    elif main_loss == "reg_l1":
        recon_loss = torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
    else:
        recon_loss = (((pred_img - gt_img) / (pred_img.detach() + 1e-3)) ** 2).mean()

    if ssim_loss != "ssim":
        simloss = 1 - ssim_metric(
            (gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
            (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
        )
    else:
        simloss = 1 - ssim_metric(
            gt_img.permute(2, 0, 1)[None, ...],
            pred_img.permute(2, 0, 1)[None, ...],
        )

    return (1 - ssim_lambda) * recon_loss + ssim_lambda * simloss
