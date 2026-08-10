# ruff: noqa: E741
# Copyright 2024 Huapeng Li, Wenxuan Song, Tianao Xu, Alexandre Elsig and Jonas KulhanekS. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Python package for combining 3DGS with volume rendering to enable water/fog modeling.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from water_splatting._torch_impl import quat_to_rotmat
from water_splatting.fields import (
    DirectionConditionedMediumField,
    compute_bounded_gaussian_colors,
    compute_bounded_headroom_gaussian_colors,
    compute_gaussian_colors,
    get_medium_context_extra_dim,
)
from water_splatting.rendering import UnderwaterRasterizer
from water_splatting.sh import num_sh_bases
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.encodings import SHEncoding


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def RGB2SHLogits(rgb, eps: float = 1e-7):
    """Map seed RGB to DC SH coefficients whose degree-0 output is logit(RGB)."""

    C0 = 0.28209479177387814
    return torch.logit(rgb, eps=float(eps)) / C0


def SHLogits2RGB(sh):
    """Map bounded-SH DC coefficients to the corresponding sigmoid RGB."""

    C0 = 0.28209479177387814
    return torch.sigmoid(sh * C0)


@dataclass
class WaterSplattingModelConfig(ModelConfig):
    """Water Splatting Model Config"""

    _target: Type = field(default_factory=lambda: WaterSplattingModel)
    num_steps: int = 15000
    """Number of steps to train the model"""
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.5
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_alpha_thresh_post: float = 0.1
    """threshold of opacity for post culling gaussians"""
    reset_alpha_thresh: float = 0.5
    """threshold of opacity for resetting alpha"""
    cull_scale_thresh: float = 10.
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    zero_medium: bool = False
    """If True, zero out the medium field"""
    reset_alpha_every: int = 5
    """Every this many refinement steps, reset the alpha"""
    abs_grad_densification: bool = True
    """If True, use absolute gradient for densification"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians (0.0004, 0.0008)"""
    densify_size_thresh: float = 0.001
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    clip_thresh: float = 0.01
    """minimum depth threshold"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 0
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    main_loss: Literal["l1", "reg_l1", "reg_l2"] = "reg_l1"
    """main loss to use"""
    ssim_loss: Literal["reg_ssim", "ssim"] = "reg_ssim"
    """ssim loss to use"""
    photometric_normalization_mode: Literal["relative_pred_detached", "absolute"] = "relative_pred_detached"
    """Photometric normalization for the main RGB objective. The default preserves historical reg_l1/reg_ssim behavior."""
    stop_split_at: int = 10000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    num_layers_medium: int = 2
    """Number of hidden layers for medium MLP."""
    hidden_dim_medium: int = 128
    """Dimension of hidden layers for medium MLP."""
    medium_density_bias: float = 0.0
    """Bias for medium density (sigma_bs and sigma_attn)."""
    mlp_type: Literal["tcnn", "torch"] = "tcnn"
    """Type of MLP to use for medium MLP."""
    medium_context_mode: Literal["dir_only", "dir_xy", "dir_xy_camera"] = "dir_only"
    """M1 medium input mode. dir_xy_camera is the formal M1 setting."""
    medium_camera_context_scale: float = 1.0
    """Multiplier applied after scene-box camera-center normalization."""
    medium_camera_context_dropout: float = 0.0
    """Dropout applied to the 3D camera context feature during training."""
    infinite_water_enabled: bool = False
    """Kept for M1 config compatibility; clean branch does not implement infinite-water ownership."""
    b_inf_mode: Literal["implicit", "tied"] = "implicit"
    """Backscatter tail mode. M1+BND uses tied, where B_inf equals medium_rgb."""
    intrinsic_color_parameterization: Literal["legacy", "bounded_sh3", "bounded_headroom_sh3"] = "legacy"
    """Gaussian intrinsic color mapping."""
    bounded_sh_logit_eps: float = 1e-7
    """Epsilon used only for RGB-equivalent bounded-SH seed color logit initialization."""
    appearance_lr_scale: float = 1.0
    """Scale applied only to features_dc and features_rest optimizer LR trajectories."""
    appearance_audit_log_dir: Optional[str] = None
    """Optional directory for AOPT LR/update JSONL diagnostics."""
    appearance_lr_audit_steps: str = "0,1,1000,3000,5000,8000,10000,13000,14999"
    """Comma-separated training steps where AOPT LR diagnostics are written."""
    appearance_update_audit_steps: str = "100,1000,5000,10000,14999"
    """Comma-separated training steps where AOPT one-step update diagnostics are written."""
    medium_hold_start_step: int = -1
    """Checkpoint boundary after which medium parameters are temporarily held. Disabled when negative."""
    medium_hold_end_step: int = -1
    """Last training update whose medium optimizer update is skipped. Disabled when <= start."""
    medium_hold_audit_log_dir: Optional[str] = None
    """Optional directory for medium-hold schedule and update JSONL diagnostics."""
    medium_hold_audit_steps: str = "10001,10500,11000,11500,12000,12500,12501,13000,14000,14999"
    """Comma-separated training steps where medium-hold diagnostics are written."""


class WaterSplattingModel(Model):
    """
    Args:
        config: Water Splatting configuration to instantiate model
    """

    config: WaterSplattingModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        if self.config.infinite_water_enabled:
            raise ValueError("research/m1-bounded-intrinsic only supports infinite_water_enabled=False")
        # initialize the medium MLP
        self.direction_encoding = SHEncoding(levels=4, implementation="tcnn")
        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()
        # medium MLP
        num_layers_medium=self.config.num_layers_medium,
        hidden_dim_medium=self.config.hidden_dim_medium,
        self.medium_density_bias=self.config.medium_density_bias,
        # if type is tuple, then [0]
        num_layers_medium = num_layers_medium if isinstance(num_layers_medium, int) else num_layers_medium[0]
        hidden_dim_medium = hidden_dim_medium if isinstance(hidden_dim_medium, int) else hidden_dim_medium[0]
        self.medium_density_bias = self.medium_density_bias if isinstance(self.medium_density_bias, float) else self.medium_density_bias[0]
        # ------------------------Medium network------------------------
        # Medium MLP
        medium_context_mode = getattr(self.config, "medium_context_mode", "dir_only")
        medium_input_dim = self.direction_encoding.get_out_dim() + get_medium_context_extra_dim(medium_context_mode)
        if num_layers_medium > 1:
            self.medium_mlp = MLP(
                in_dim=medium_input_dim,
                num_layers=num_layers_medium,
                layer_width=hidden_dim_medium,
                out_dim=9,
                activation=nn.Sigmoid(),
                out_activation=None,
                implementation=self.config.mlp_type,
            )
        else:
            self.medium_mlp = nn.Linear(medium_input_dim, 9)
            self.config.mlp_type = "torch"
        self.medium_field = DirectionConditionedMediumField(
            direction_encoding=self.direction_encoding,
            medium_mlp=self.medium_mlp,
            colour_activation=self.colour_activation,
            sigma_activation=self.sigma_activation,
        )
        self.underwater_rasterizer = UnderwaterRasterizer()

        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                seed_rgb = self.seed_points[1] / 255
                if self.config.intrinsic_color_parameterization in ("bounded_sh3", "bounded_headroom_sh3"):
                    CONSOLE.log("use bounded SH3 intrinsic color parameterization")
                    shs[:, 0, :3] = RGB2SHLogits(seed_rgb, eps=self.config.bounded_sh_logit_eps)
                else:
                    shs[:, 0, :3] = RGB2SH(seed_rgb)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        self._appearance_lr_scale_applied = False
        self._appearance_update_pre: Optional[Dict[str, Union[int, torch.Tensor]]] = None
        self._medium_hold_update_pre: Optional[Dict[str, Any]] = None
        self._medium_hold_last_active: Optional[bool] = None

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            if self.config.intrinsic_color_parameterization in ("bounded_sh3", "bounded_headroom_sh3"):
                return SHLogits2RGB(self.features_dc)
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]
    
    @property
    def medium_mlp(self):
        return self.gauss_params["medium_mlp"]
    
    @property
    def direction_encoding(self):
        return self.gauss_params["direction_encoding"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = self.config.num_steps
        dict = dict.copy()
        dict.pop("gaussian_lineage_ids", None)
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        # if self.step >= self.config.stop_split_at:
        #     return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            if self.config.abs_grad_densification:
                assert self.xys_grad_abs is not None
                grads = self.xys_grad_abs.detach().norm(dim=-1)
            else:
                assert self.xys.grad is not None
                grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.depths_accum = self.depths
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]
                self.depths_accum[visible_mask] = self.depths[visible_mask] + self.depths_accum[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and (self.step % reset_interval > self.num_train_data + self.config.refine_every)
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])

                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()

                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads

                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads

                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # if self.step < self.config.stop_screen_size_at:
                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )                
                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None
    
            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

                # reset the exp of optimizer
                for key in ["medium_mlp", "direction_encoding"]:
                    optim = optimizers.optimizers[key]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    if "exp_avg" in param_state:
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

                
            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:                
                # Reset value is set to be reset_alpha_thresh
                reset_value = self.config.reset_alpha_thresh
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            
            self.xys_grad_norm = None
            self.vis_counts = None
            self.depths_accum = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        if self.step < self.config.stop_split_at:
            cull_alpha_thresh = self.config.cull_alpha_thresh
        else:
            cull_alpha_thresh = self.config.cull_alpha_thresh_post
        culls = (torch.sigmoid(self.opacities) < cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.aopt_before_train_iteration,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.medium_hold_before_train_iteration,
                args=[training_callback_attributes.optimizers],
            )
        )
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.aopt_after_train_iteration,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.medium_hold_after_train_iteration,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def _parse_aopt_steps(self, value: str) -> set[int]:
        steps: set[int] = set()
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            steps.add(int(item))
        return steps

    def _aopt_log_jsonl(self, filename: str, row: Dict[str, Union[str, int, float, bool]]) -> None:
        if not self.config.appearance_audit_log_dir:
            return
        path = Path(os.path.expanduser(self.config.appearance_audit_log_dir)) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _apply_appearance_lr_scale(self, optimizers: Optimizers) -> None:
        scale = float(self.config.appearance_lr_scale)
        if scale <= 0.0:
            raise ValueError(f"appearance_lr_scale must be positive, got {scale}")
        if self._appearance_lr_scale_applied:
            return

        rows = []
        for group in ("features_dc", "features_rest"):
            if group not in optimizers.optimizers:
                raise RuntimeError(f"Missing optimizer group for appearance LR scale: {group}")
            base_lr = float(optimizers.config[group]["optimizer"].lr)
            target_base_lr = base_lr * scale
            optimizer = optimizers.optimizers[group]
            current_base_lr = float(optimizer.param_groups[0].get("initial_lr", base_lr))
            if not math.isclose(current_base_lr, target_base_lr, rel_tol=1e-9, abs_tol=1e-12):
                ratio = target_base_lr / max(current_base_lr, 1e-30)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = float(param_group["lr"]) * ratio
                    param_group["initial_lr"] = target_base_lr

            scheduler = optimizers.schedulers.get(group)
            if scheduler is not None:
                base_lrs = []
                for base in scheduler.base_lrs:
                    base = float(base)
                    if math.isclose(base, target_base_lr, rel_tol=1e-9, abs_tol=1e-12):
                        base_lrs.append(base)
                    else:
                        base_lrs.append(target_base_lr)
                scheduler.base_lrs = base_lrs

            rows.append(
                {
                    "group": group,
                    "base_lr": base_lr,
                    "target_base_lr": target_base_lr,
                    "actual_lr": float(optimizer.param_groups[0]["lr"]),
                    "scale": scale,
                    "scheduler_base_lr": float(optimizers.schedulers[group].base_lrs[0])
                    if group in optimizers.schedulers
                    else float("nan"),
                }
            )

        self._appearance_lr_scale_applied = True
        self._aopt_log_jsonl(
            "aopt_lr_scale_application.jsonl",
            {
                "step": int(self.step),
                "scale": scale,
                "features_dc_actual_lr": rows[0]["actual_lr"],
                "features_rest_actual_lr": rows[1]["actual_lr"],
                "features_dc_scheduler_base_lr": rows[0]["scheduler_base_lr"],
                "features_rest_scheduler_base_lr": rows[1]["scheduler_base_lr"],
            },
        )
        if scale != 1.0:
            CONSOLE.log(f"Applied appearance_lr_scale={scale} to features_dc/features_rest only")

    def _aopt_lr_row(self, optimizers: Optimizers, step: int) -> Dict[str, Union[str, int, float]]:
        row: Dict[str, Union[str, int, float]] = {
            "step": int(step),
            "appearance_lr_scale": float(self.config.appearance_lr_scale),
        }
        for group in ("features_dc", "features_rest", "means", "scales", "quats", "opacities", "medium_mlp"):
            if group in optimizers.optimizers:
                row[f"{group}_lr"] = float(optimizers.optimizers[group].param_groups[0]["lr"])
        return row

    def aopt_before_train_iteration(self, optimizers: Optimizers, step: int) -> None:
        self._apply_appearance_lr_scale(optimizers)
        if step in self._parse_aopt_steps(self.config.appearance_lr_audit_steps):
            self._aopt_log_jsonl("aopt_lr_trajectory.jsonl", self._aopt_lr_row(optimizers, step))
        if step in self._parse_aopt_steps(self.config.appearance_update_audit_steps):
            self._appearance_update_pre = {
                "step": int(step),
                "features_dc": self.features_dc.detach().clone(),
                "features_rest": self.features_rest.detach().clone(),
            }

    def aopt_after_train_iteration(self, step: int) -> None:
        if self._appearance_update_pre is None:
            return
        if int(self._appearance_update_pre["step"]) != int(step):
            return
        row: Dict[str, Union[str, int, float]] = {
            "step": int(step),
            "appearance_lr_scale": float(self.config.appearance_lr_scale),
        }
        for name, param in (("features_dc", self.features_dc), ("features_rest", self.features_rest)):
            before = self._appearance_update_pre[name]
            if not isinstance(before, torch.Tensor) or before.shape != param.shape:
                row[f"{name}_status"] = "shape_changed"
                continue
            current = param.detach()
            delta = current - before.to(device=current.device, dtype=current.dtype)
            grad = param.grad.detach().float() if param.grad is not None else None
            theta_norm = float(torch.linalg.norm(before.float()).item())
            update_norm = float(torch.linalg.norm(delta.float()).item())
            row[f"{name}_status"] = "ok"
            row[f"{name}_update_l2"] = update_norm
            row[f"{name}_theta_l2_before"] = theta_norm
            row[f"{name}_normalized_update"] = update_norm / max(theta_norm, 1e-12)
            row[f"{name}_grad_l2"] = float(torch.linalg.norm(grad).item()) if grad is not None else 0.0
            row[f"{name}_grad_mean_abs"] = float(grad.abs().mean().item()) if grad is not None else 0.0
        self._aopt_log_jsonl("aopt_parameter_updates.jsonl", row)
        self._appearance_update_pre = None

    def _medium_hold_enabled(self) -> bool:
        start = int(self.config.medium_hold_start_step)
        end = int(self.config.medium_hold_end_step)
        return start >= 0 and end > start

    def _medium_hold_active(self, step: int) -> bool:
        """Return True for training updates whose medium optimizer step is skipped.

        The schedule is defined by checkpoint boundaries: after loading a checkpoint
        saved at ``medium_hold_start_step``, the first held update is
        ``start + 1`` and the last held update is ``medium_hold_end_step``.
        This gives exactly 2500 held updates for a 10000->12500 schedule.
        """

        if not self._medium_hold_enabled():
            return False
        start = int(self.config.medium_hold_start_step)
        end = int(self.config.medium_hold_end_step)
        return start < int(step) <= end

    def _medium_hold_phase(self, step: int) -> str:
        if not self._medium_hold_enabled():
            return "DISABLED"
        if self._medium_hold_active(step):
            return "MEDIUM_HOLD"
        if int(step) > int(self.config.medium_hold_end_step):
            return "JOINT"
        return "PRE_HOLD"

    def _medium_hold_groups(self) -> Tuple[str, str]:
        return ("medium_mlp", "direction_encoding")

    def _medium_hold_log_jsonl(self, filename: str, row: Dict[str, Any]) -> None:
        if not self.config.medium_hold_audit_log_dir:
            return
        path = Path(os.path.expanduser(self.config.medium_hold_audit_log_dir)) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _parse_medium_hold_audit_steps(self) -> set[int]:
        steps = self._parse_aopt_steps(self.config.medium_hold_audit_steps)
        if self._medium_hold_enabled():
            start = int(self.config.medium_hold_start_step)
            end = int(self.config.medium_hold_end_step)
            steps.update({start + 1, end, end + 1})
        return steps

    def _set_medium_requires_grad(self, enabled: bool) -> None:
        for group in self._medium_hold_groups():
            params = getattr(self, group).parameters()
            for param in params:
                param.requires_grad_(enabled)

    def _snapshot_parameters(self, optimizers: Optimizers, groups: Tuple[str, ...]) -> Dict[str, List[torch.Tensor]]:
        snapshots: Dict[str, List[torch.Tensor]] = {}
        for group in groups:
            tensors = []
            for param in optimizers.parameters.get(group, []):
                tensors.append(param.detach().clone())
            snapshots[group] = tensors
        return snapshots

    def _snapshot_optimizer_state(self, optimizers: Optimizers, groups: Tuple[str, ...]) -> Dict[str, List[Dict[str, Any]]]:
        snapshots: Dict[str, List[Dict[str, Any]]] = {}
        for group in groups:
            optimizer = optimizers.optimizers.get(group)
            if optimizer is None:
                snapshots[group] = []
                continue
            group_rows: List[Dict[str, Any]] = []
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    state = optimizer.state.get(param, {})
                    row: Dict[str, Any] = {}
                    for key in ("exp_avg", "exp_avg_sq"):
                        value = state.get(key)
                        row[key] = value.detach().clone() if isinstance(value, torch.Tensor) else None
                    step_value = state.get("step")
                    if isinstance(step_value, torch.Tensor):
                        row["step"] = step_value.detach().clone()
                    elif step_value is None:
                        row["step"] = None
                    else:
                        row["step"] = float(step_value)
                    group_rows.append(row)
            snapshots[group] = group_rows
        return snapshots

    def _write_param_delta(
        self,
        row: Dict[str, Any],
        prefix: str,
        before: Dict[str, List[torch.Tensor]],
        optimizers: Optimizers,
        groups: Tuple[str, ...],
    ) -> None:
        for group in groups:
            current = list(optimizers.parameters.get(group, []))
            old = before.get(group, [])
            if len(current) != len(old) or any(param.shape != old_param.shape for param, old_param in zip(current, old)):
                row[f"{prefix}_{group}_status"] = "shape_changed"
                continue
            max_abs = 0.0
            l2_sq = 0.0
            theta_sq = 0.0
            for param, old_param in zip(current, old):
                delta = param.detach().float() - old_param.to(device=param.device, dtype=torch.float32)
                max_abs = max(max_abs, float(delta.abs().max().item()) if delta.numel() else 0.0)
                l2_sq += float(delta.square().sum().item())
                theta_sq += float(old_param.float().square().sum().item())
            row[f"{prefix}_{group}_status"] = "ok"
            row[f"{prefix}_{group}_max_abs_delta"] = max_abs
            row[f"{prefix}_{group}_l2_delta"] = math.sqrt(l2_sq)
            row[f"{prefix}_{group}_normalized_l2_delta"] = math.sqrt(l2_sq) / max(math.sqrt(theta_sq), 1e-12)

    def _write_optimizer_state_delta(
        self,
        row: Dict[str, Any],
        before: Dict[str, List[Dict[str, Any]]],
        optimizers: Optimizers,
        groups: Tuple[str, ...],
    ) -> None:
        after = self._snapshot_optimizer_state(optimizers, groups)
        for group in groups:
            old_rows = before.get(group, [])
            new_rows = after.get(group, [])
            if len(old_rows) != len(new_rows):
                row[f"{group}_optimizer_state_status"] = "shape_changed"
                continue
            row[f"{group}_optimizer_state_status"] = "ok"
            for state_key in ("exp_avg", "exp_avg_sq"):
                max_abs = 0.0
                l2_sq = 0.0
                valid = False
                for old_state, new_state in zip(old_rows, new_rows):
                    old_tensor = old_state.get(state_key)
                    new_tensor = new_state.get(state_key)
                    if not isinstance(old_tensor, torch.Tensor) or not isinstance(new_tensor, torch.Tensor):
                        continue
                    valid = True
                    delta = new_tensor.float() - old_tensor.to(device=new_tensor.device, dtype=torch.float32)
                    max_abs = max(max_abs, float(delta.abs().max().item()) if delta.numel() else 0.0)
                    l2_sq += float(delta.square().sum().item())
                row[f"{group}_{state_key}_max_abs_delta"] = max_abs if valid else 0.0
                row[f"{group}_{state_key}_l2_delta"] = math.sqrt(l2_sq) if valid else 0.0

            step_delta = 0.0
            valid_step = False
            for old_state, new_state in zip(old_rows, new_rows):
                old_step = old_state.get("step")
                new_step = new_state.get("step")
                if isinstance(old_step, torch.Tensor):
                    old_value = float(old_step.detach().cpu().reshape(-1)[0].item())
                elif old_step is None:
                    old_value = 0.0
                else:
                    old_value = float(old_step)
                if isinstance(new_step, torch.Tensor):
                    new_value = float(new_step.detach().cpu().reshape(-1)[0].item())
                elif new_step is None:
                    new_value = 0.0
                else:
                    new_value = float(new_step)
                step_delta = max(step_delta, abs(new_value - old_value))
                valid_step = True
            row[f"{group}_optimizer_step_max_delta"] = step_delta if valid_step else 0.0

    def _write_grad_summary(self, row: Dict[str, Any], optimizers: Optimizers, groups: Tuple[str, ...]) -> None:
        for group in groups:
            grad_l2_sq = 0.0
            grad_max_abs = 0.0
            grad_param_count = 0
            for param in optimizers.parameters.get(group, []):
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                grad_l2_sq += float(grad.square().sum().item())
                grad_max_abs = max(grad_max_abs, float(grad.abs().max().item()) if grad.numel() else 0.0)
                grad_param_count += 1
            row[f"{group}_grad_l2"] = math.sqrt(grad_l2_sq)
            row[f"{group}_grad_max_abs"] = grad_max_abs
            row[f"{group}_grad_param_count"] = grad_param_count

    def medium_hold_before_train_iteration(self, optimizers: Optimizers, step: int) -> None:
        active = self._medium_hold_active(step)
        if self._medium_hold_enabled():
            self._set_medium_requires_grad(not active)
        self._medium_hold_last_active = active

        if not self._medium_hold_enabled():
            return
        audit_steps = self._parse_medium_hold_audit_steps()
        if int(step) not in audit_steps:
            return

        medium_groups = self._medium_hold_groups()
        object_groups = ("features_dc", "features_rest", "means", "scales", "opacities")
        self._medium_hold_update_pre = {
            "step": int(step),
            "medium_params": self._snapshot_parameters(optimizers, medium_groups),
            "object_params": self._snapshot_parameters(optimizers, object_groups),
            "medium_optimizer_state": self._snapshot_optimizer_state(optimizers, medium_groups),
        }

    def medium_hold_after_train_iteration(self, optimizers: Optimizers, step: int) -> None:
        if self._medium_hold_update_pre is None:
            return
        if int(self._medium_hold_update_pre["step"]) != int(step):
            return

        active = self._medium_hold_active(step)
        row: Dict[str, Any] = {
            "step": int(step),
            "phase": self._medium_hold_phase(step),
            "medium_hold_start_step": int(self.config.medium_hold_start_step),
            "medium_hold_end_step": int(self.config.medium_hold_end_step),
            "medium_requires_grad": not active,
            "gaussian_count": int(self.num_points),
        }
        for group in self._medium_hold_groups():
            if group in optimizers.optimizers:
                row[f"{group}_lr"] = float(optimizers.optimizers[group].param_groups[0]["lr"])
            if group in optimizers.schedulers:
                row[f"{group}_scheduler_last_lr"] = float(optimizers.schedulers[group].get_last_lr()[0])

        self._write_param_delta(
            row,
            "medium_param",
            self._medium_hold_update_pre["medium_params"],
            optimizers,
            self._medium_hold_groups(),
        )
        self._write_optimizer_state_delta(
            row,
            self._medium_hold_update_pre["medium_optimizer_state"],
            optimizers,
            self._medium_hold_groups(),
        )
        self._write_param_delta(
            row,
            "object_param",
            self._medium_hold_update_pre["object_params"],
            optimizers,
            ("features_dc", "features_rest", "means", "scales", "opacities"),
        )
        self._write_grad_summary(
            row,
            optimizers,
            self._medium_hold_groups() + ("features_dc", "features_rest", "means", "opacities"),
        )
        self._medium_hold_log_jsonl("stage_transition_audit.jsonl", row)
        self._medium_hold_log_jsonl(
            "lr_scheduler_audit.jsonl",
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "step",
                    "phase",
                    "medium_mlp_lr",
                    "direction_encoding_lr",
                    "medium_mlp_scheduler_last_lr",
                    "direction_encoding_scheduler_last_lr",
                }
            },
        )
        self._medium_hold_update_pre = None

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        gps["medium_mlp"] = list(self.medium_mlp.parameters())
        gps["direction_encoding"] = list(self.direction_encoding.parameters())
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image

    def _get_scene_normalization(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        aabb = self.scene_box.aabb.to(device=device, dtype=dtype)
        center = (aabb[0] + aabb[1]) * 0.5
        scale = torch.linalg.norm(aabb[1] - aabb[0]).clamp_min(1e-6)
        return center, scale

    def _effective_b_inf_mode(self) -> str:
        mode = getattr(self.config, "b_inf_mode", "implicit")
        if mode not in {"implicit", "tied"}:
            raise ValueError(f"Unsupported b_inf_mode on clean M1+BND branch: {mode}")
        return mode

    def _predict_medium(
        self,
        *,
        camera: Cameras,
        rotation_world_from_camera: torch.Tensor,
        height: int,
        width: int,
        cx: float,
        cy: float,
    ):
        scene_center, scene_scale = self._get_scene_normalization(
            dtype=rotation_world_from_camera.dtype,
            device=rotation_world_from_camera.device,
        )
        return self.medium_field(
            camera=camera,
            rotation_world_from_camera=rotation_world_from_camera,
            height=height,
            width=width,
            cx=cx,
            cy=cy,
            density_bias=self.medium_density_bias,
            mlp_type=self.config.mlp_type,
            zero_medium=self.config.zero_medium,
            context_mode=getattr(self.config, "medium_context_mode", "dir_only"),
            camera_center=camera.camera_to_worlds[0, :3, 3],
            scene_center=scene_center,
            scene_scale=scene_scale,
            camera_context_scale=getattr(self.config, "medium_camera_context_scale", 1.0),
            camera_context_dropout=getattr(self.config, "medium_camera_context_dropout", 0.0),
            training=self.training,
            b_inf_mode=self._effective_b_inf_mode(),
        )

    def get_outputs(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        self.last_fx = camera.fx.item()
        self.last_fy = camera.fy.item()

        medium = self._predict_medium(
            camera=camera,
            rotation_world_from_camera=R,
            height=H,
            width=W,
            cx=cx,
            cy=cy,
        )
        medium_rgb = medium.rgb
        medium_bs = medium.bs
        medium_attn = medium.attn

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = medium_rgb
                depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
                clear = torch.zeros_like(rgb)
                tau_d = medium_attn * depth
                transmission = torch.exp(-tau_d.clamp_min(0.0)).clamp(0.0, 1.0)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb,
                        "rgb_object": torch.zeros_like(rgb), "rgb_clear": torch.zeros_like(rgb),
                        "rgb_clear_clamp": clear, "clear_object_fullsh_raw": clear,
                        "J_gaussian_raw": clear, "J_gaussian": clear, "rgb_medium": medium_rgb,
                        "pred_image": rgb, "medium_rgb": medium_rgb, "medium_bs": medium_bs,
                        "medium_attn": medium_attn, "b_inf": medium.b_inf,
                        "direct_object_signal": clear, "transmission": transmission, "tau_D": tau_d}
        else:
            crop_ids = None

        if crop_ids is not None and crop_ids.sum() != 0:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = self.underwater_rasterizer.project(  # type: ignore
            means=means_crop,
            scales=scales_crop,
            quats=quats_crop,
            viewmat=viewmat,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=cx,
            cy=cy,
            height=H,
            width=W,
            clip_thresh=self.config.clip_thresh,
        )  # type: ignore

        self.depths = depths.detach()

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = medium_rgb
            depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
            clear = torch.zeros_like(rgb)
            tau_d = medium_attn * depth
            transmission = torch.exp(-tau_d.clamp_min(0.0)).clamp(0.0, 1.0)
            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb,
                    "rgb_object": clear, "rgb_clear": clear, "rgb_clear_clamp": clear,
                    "clear_object_fullsh_raw": clear, "J_gaussian_raw": clear, "J_gaussian": clear,
                    "rgb_medium": medium_rgb, "pred_image": rgb, "medium_rgb": medium_rgb,
                    "medium_bs": medium_bs, "medium_attn": medium_attn, "b_inf": medium.b_inf,
                    "direct_object_signal": clear, "transmission": transmission, "tau_D": tau_d}

        if self.training:
            self.xys.retain_grad()

        n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        bounded_color = None
        if self.config.intrinsic_color_parameterization == "legacy":
            rgbs = compute_gaussian_colors(
                means=means_crop,
                features_dc=features_dc_crop,
                features_rest=features_rest_crop,
                camera_position=camera.camera_to_worlds[..., :3, 3],
                sh_degree=self.config.sh_degree,
                active_sh_degree=n,
            )
        elif self.config.intrinsic_color_parameterization == "bounded_sh3":
            bounded_color = compute_bounded_gaussian_colors(
                means=means_crop,
                features_dc=features_dc_crop,
                features_rest=features_rest_crop,
                camera_position=camera.camera_to_worlds[..., :3, 3],
                sh_degree=self.config.sh_degree,
                active_sh_degree=n,
            )
            rgbs = bounded_color.rgb
        elif self.config.intrinsic_color_parameterization == "bounded_headroom_sh3":
            bounded_color = compute_bounded_headroom_gaussian_colors(
                means=means_crop,
                features_dc=features_dc_crop,
                features_rest=features_rest_crop,
                camera_position=camera.camera_to_worlds[..., :3, 3],
                sh_degree=self.config.sh_degree,
                active_sh_degree=n,
            )
            rgbs = bounded_color.rgb
        else:
            raise ValueError(f"Unknown intrinsic_color_parameterization: {self.config.intrinsic_color_parameterization}")

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)
        
        self.xys_grad_abs = torch.zeros_like(self.xys)

        render = self.underwater_rasterizer.rasterize(  # type: ignore
            xys=self.xys,
            xys_grad_abs=self.xys_grad_abs,
            depths=depths,
            radii=self.radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,  # type: ignore
            colors=rgbs,
            opacities=opacities,
            medium_rgb=medium_rgb,
            medium_bs=medium_bs,
            medium_attn=medium_attn,
            height=H,
            width=W,
            background=medium_rgb,
            step=self.step,
        )  # type: ignore

        rgb = render.rgb
        rgb_medium = render.rgb_medium
        rgb_tail = torch.zeros_like(rgb_medium)
        rgb_medium_finite = rgb_medium
        b_inf = medium.b_inf
        if self._effective_b_inf_mode() == "tied":
            if b_inf is None:
                raise RuntimeError("b_inf_mode='tied' requires a B_inf tensor")
            tail_weight = render.final_transmittance * torch.exp(-medium_bs * render.last_depth)
            rgb_tail_original = tail_weight * medium_rgb
            rgb_medium_finite = rgb_medium - rgb_tail_original
            rgb_tail = tail_weight * b_inf
            rgb_medium = rgb_medium_finite + rgb_tail
            rgb = render.rgb_object + rgb_medium

        tau_d = medium_attn * render.depth
        transmission = torch.exp(-tau_d.clamp_min(0.0)).clamp(0.0, 1.0)

        outputs = {
            "rgb": rgb,
            "depth": render.depth,
            "accumulation": render.accumulation,
            "background": medium_rgb,
            "rgb_object": render.rgb_object,
            "direct_object_signal": render.rgb_object,
            "rgb_clear": render.rgb_clear,
            "rgb_clear_clamp": render.rgb_clear_clamp,
            "clear_object_fullsh_raw": render.j_raw,
            "J_gaussian_raw": render.j_raw,
            "J_gaussian": render.j_gaussian,
            "rgb_medium": rgb_medium,
            "rgb_medium_finite": rgb_medium_finite,
            "rgb_tail": rgb_tail,
            "pred_image": rgb,
            "medium_rgb": medium_rgb,
            "medium_bs": medium_bs,
            "medium_attn": medium_attn,
            "b_inf": b_inf,
            "transmission": transmission,
            "tau_D": tau_d,
            "appearance_active_sh_degree": rgbs.new_tensor(float(n)),
            "intrinsic_color_parameterization": rgbs.new_tensor(
                {
                    "legacy": 0.0,
                    "bounded_sh3": 1.0,
                    "bounded_headroom_sh3": 2.0,
                }[self.config.intrinsic_color_parameterization]
            ),
            "gaussian_view_rgb": rgbs.detach(),
            "gaussian_visible_mask": (self.radii > 0).reshape(-1).detach(),
            "projected_gaussian_depths": depths.detach(),
        }
        if bounded_color is not None:
            if bounded_color.logits is not None:
                outputs["gaussian_view_logits"] = bounded_color.logits
            if bounded_color.sigmoid_derivative is not None:
                outputs["gaussian_sigmoid_derivative"] = bounded_color.sigmoid_derivative
            if bounded_color.dc_rgb is not None:
                outputs["gaussian_view_dc_rgb"] = bounded_color.dc_rgb
            if bounded_color.dc_logits is not None:
                outputs["gaussian_view_dc_logits"] = bounded_color.dc_logits
            if bounded_color.sh_residual is not None:
                outputs["gaussian_sh_residual"] = bounded_color.sh_residual
            if bounded_color.color_residual is not None:
                outputs["gaussian_color_residual"] = bounded_color.color_residual
            if bounded_color.positive_utilization is not None:
                outputs["gaussian_headroom_u_pos"] = bounded_color.positive_utilization
            if bounded_color.negative_utilization is not None:
                outputs["gaussian_headroom_u_neg"] = bounded_color.negative_utilization
        return outputs  # type: ignore
        
    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            # alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return image[..., :3]
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        for i in range(3):
            # 3 channels
            metrics_dict[f"medium_attn_{i}"] = outputs["medium_attn"][:, :, i].mean()
            metrics_dict[f"medium_bs_{i}"] = outputs["medium_bs"][:, :, i].mean()
            metrics_dict[f"medium_rgb_{i}"] = outputs["medium_rgb"][:, :, i].mean()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["pred_image"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        if self.config.photometric_normalization_mode == "absolute":
            recon_loss = torch.abs(gt_img - pred_img).mean()
            simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        elif self.config.photometric_normalization_mode == "relative_pred_detached":
            if self.config.main_loss == "l1":
                recon_loss = torch.abs(gt_img - pred_img).mean()
            elif self.config.main_loss == "reg_l1":
                recon_loss = torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
            else:
                recon_loss = (((pred_img - gt_img) / (pred_img.detach() + 1e-3)) ** 2).mean()

            if self.config.ssim_loss != "ssim":
                simloss = 1 - self.ssim((gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...], (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...])
            else:
                simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        else:
            raise ValueError(f"Unknown photometric_normalization_mode: {self.config.photometric_normalization_mode}")

        return {
            "main_loss": (1 - self.config.ssim_lambda) * recon_loss + self.config.ssim_lambda * simloss,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device), obb_box=obb_box)
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])

        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)

        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(predicted_rgb.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = predicted_rgb

        output_gt_rgb = gt_rgb.cpu()

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "gt": output_gt_rgb,
            "rgb_medium": outputs["rgb_medium"],
            "rgb_object": outputs["rgb_object"],
            "direct_object_signal": outputs["direct_object_signal"],
            "clear_object_fullsh_raw": outputs["clear_object_fullsh_raw"],
            "rgb_clear_clamp": outputs["rgb_clear_clamp"],
            "rgb_clear": outputs["rgb_clear"],
            "transmission": outputs["transmission"],
            "tau_D": outputs["tau_D"],
            "depth": outputs["depth"],
            "rgb": outputs["rgb"],
        }
        return metrics_dict, images_dict
