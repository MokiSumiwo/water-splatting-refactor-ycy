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

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from water_splatting.cleanup import build_cleanup_candidate_mask, format_cleanup_stats, sample_pixel_map_at_gaussians
from water_splatting.fields import (
    DirectionConditionedMediumField,
    compute_gaussian_colors,
    compute_gaussian_sh_residual,
    get_medium_context_extra_dim,
)
from water_splatting.losses import (
    dc_channel_balance_loss,
    dc_softclip_loss,
    low_transmission_weights,
    medium_attenuation_order_loss,
    reconstruction_loss,
    sh_residual_mean_anchor_loss,
)
from water_splatting.ownership import compute_infinite_water_ownership
from water_splatting.rendering import UnderwaterRasterizer
from water_splatting._torch_impl import quat_to_rotmat
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
    medium_context_mode: Literal["dir_only", "dir_xy", "dir_xy_depth", "dir_xy_camera", "dir_xy_depth_camera"] = "dir_only"
    """M1 medium input mode. dir_only preserves original WaterSplatting behavior."""
    medium_camera_context_scale: float = 1.0
    """Multiplier applied after scene-box camera-center normalization."""
    medium_camera_context_dropout: float = 0.0
    """Dropout applied to the 3D camera context feature during training."""
    medium_depth_context_detach: bool = True
    """If True, M1 depth context is detached before the second medium pass."""
    medium_depth_context_normalize: bool = True
    """If True, normalize M1 depth context per rendered view."""
    medium_depth_context_normalize_mode: Literal["max", "p95"] = "p95"
    """Depth normalization statistic for M1 depth context."""
    infinite_water_enabled: bool = False
    """M2: enable infinite-water B_inf branch and ownership composition."""
    infinite_water_ownership_mode: Literal["off", "alpha_only", "alpha_depth", "alpha_depth_color"] = "alpha_depth"
    """M2 ownership evidence mode."""
    infinite_water_detach_evidence: bool = True
    """Detach internal render evidence before constructing M_inf."""
    infinite_water_occupancy_limited: bool = True
    """If True, B_inf can only take over low-accumulation pixels."""
    infinite_water_compose_mode: Literal["none", "rgb_mix", "tail_approx", "closed_tail"] = "rgb_mix"
    """M2 composition mode. rgb_mix preserves current behavior; none disables B_inf RGB composition."""
    infinite_water_alpha_power: float = 1.0
    """Power applied to low-accumulation evidence."""
    infinite_water_depth_mid: float = 0.75
    """Normalized depth midpoint for far-depth evidence."""
    infinite_water_depth_temp: float = 0.10
    """Temperature for far-depth evidence sigmoid."""
    infinite_water_color_temp: float = 0.20
    """Temperature for B_inf color-similarity evidence."""
    infinite_water_depth_normalize_mode: Literal["max", "p95"] = "p95"
    """Depth normalization statistic for M2 ownership evidence."""
    infinite_water_hit_alpha_threshold: float = 0.20
    """Object accumulation threshold for hit-aware confidence."""
    infinite_water_hit_alpha_temp: float = 0.05
    """Temperature for hit-aware accumulation confidence."""
    infinite_water_hit_concentration_kappa: float = 0.20
    """Relative-depth-dispersion scale for hit-aware concentration confidence."""
    infinite_water_capacity_support_mode: Literal["m_inf", "hit_alpha", "hit", "hit_squared"] = "m_inf"
    """Support used by accumulation-zero loss. m_inf preserves first-stage M2 behavior."""
    infinite_water_capacity_loss_mode: Literal["none", "current", "depth_monotonic", "relu_budget", "softplus_budget"] = "current"
    """Capacity loss form. current preserves first-stage M2; budget modes use depth-only far support."""
    infinite_water_capacity_budget: float = 0.05
    """Allowed accumulation budget for capacity budget losses."""
    infinite_water_capacity_budget_temp: float = 0.02
    """Softplus temperature for softplus capacity budget loss."""
    infinite_water_hit_protection_enabled: bool = False
    """If True, attenuate capacity support on high-confidence hit regions with a nonzero floor."""
    infinite_water_hit_protection_threshold: float = 0.80
    """Hit-confidence threshold for conservative object protection."""
    infinite_water_hit_protection_temp: float = 0.05
    """Temperature for the hit-confidence object-protection sigmoid."""
    infinite_water_capacity_floor: float = 0.50
    """Minimum capacity pressure kept on high-confidence object-protection regions."""
    infinite_water_hit_protection_start_step: int = 0
    """First training step where hit-protection attenuation is active."""
    infinite_water_loss_start_step: int = 1000
    """First step where M2 auxiliary losses are active."""
    infinite_water_loss_ramp_steps: int = 3000
    """Linear ramp length for M2 auxiliary loss weights."""
    lambda_infinite_water_binf_rgb: float = 0.0
    """M2 supervised B_inf RGB fit on soft infinite-water support."""
    lambda_infinite_water_accumulation_zero: float = 0.0
    """M2 low-accumulation pressure on soft infinite-water support."""
    lambda_infinite_water_near_zero: float = 0.0
    """M2 near-branch RGB suppression on soft infinite-water support."""
    gaussian_cleanup_enabled: bool = False
    """M3: enable contribution-aware Gaussian cleanup diagnostics/pruning."""
    gaussian_cleanup_dry_run: bool = True
    """If True, log M3 cleanup candidates without deleting Gaussians."""
    gaussian_cleanup_start_step: int = 12000
    """First training step where M3 cleanup diagnostics can run."""
    gaussian_cleanup_interval: int = 500
    """Run M3 cleanup every this many steps."""
    gaussian_cleanup_contribution_threshold: float = 1e-4
    """Maximum projected gradient contribution proxy for M3 cleanup candidates."""
    gaussian_cleanup_opacity_threshold: float = 0.08
    """Maximum opacity for M3 cleanup candidates."""
    gaussian_cleanup_visibility_min_count: int = 2
    """Minimum accumulated visibility samples before a Gaussian can be considered."""
    gaussian_cleanup_alpha_threshold: float = 0.25
    """Maximum sampled object accumulation for the M3 alpha gate."""
    gaussian_cleanup_depth_threshold: float = 0.0
    """Minimum average projected depth for the M3 depth gate; <=0 disables by default."""
    gaussian_cleanup_ownership_threshold: float = 0.35
    """Minimum sampled M2 infinite-water ownership for the M3 ownership gate."""
    gaussian_cleanup_ownership_source: Literal["m_inf", "m_inf_eff"] = "m_inf_eff"
    """M2 ownership map sampled for M3 cleanup diagnostics."""
    gaussian_cleanup_require_alpha_gate: bool = True
    """Require low object accumulation at the projected Gaussian center."""
    gaussian_cleanup_require_depth_gate: bool = False
    """Require average Gaussian depth above gaussian_cleanup_depth_threshold."""
    gaussian_cleanup_require_ownership_gate: bool = True
    """Require M2 ownership support at the projected Gaussian center."""
    constrained_appearance_enabled: bool = False
    """M4: enable constrained view-dependent appearance losses/scheduling."""
    appearance_sh_delay_enabled: bool = False
    """If True, delay active SH degree schedule to reduce early color absorption."""
    appearance_sh_delay_start_step: int = 3000
    """First step where delayed SH can increase above DC-only."""
    appearance_sh_delay_interval: int = 2000
    """Interval between delayed SH degree increments."""
    appearance_loss_start_step: int = 1000
    """First step where M4 auxiliary losses are active."""
    appearance_loss_ramp_steps: int = 3000
    """Linear ramp length for M4 auxiliary loss weights."""
    lambda_sh_residual_mean: float = 0.0
    """M4 residual-SH mean anchor loss weight."""
    lambda_dc_softclip: float = 0.0
    """M4 low-transmission DC softclip loss weight."""
    dc_softclip_threshold: float = 0.95
    """Soft upper bound for DC intrinsic RGB."""
    dc_softclip_beta: float = 0.05
    """Softclip temperature for DC intrinsic RGB."""
    dc_softclip_use_low_transmission_weight: bool = True
    """Weight DC softclip more in low estimated transmission regions."""
    lambda_dc_channel_balance: float = 0.0
    """M4 DC color-balance loss weight for suppressing red/blue dominance."""
    dc_channel_balance_margin: float = 0.05
    """Allowed DC red/blue dominance before applying the balance penalty."""
    dc_channel_balance_beta: float = 0.05
    """Softplus temperature for DC channel-balance loss."""
    dc_channel_balance_use_low_transmission_weight: bool = True
    """Weight DC channel-balance loss more in low estimated transmission regions."""
    lambda_medium_attenuation_order: float = 0.0
    """M4 medium attenuation order loss weight for red >= green >= blue."""
    medium_attenuation_order_margin: float = 0.0
    """Margin for the medium attenuation channel-order prior."""
    medium_attenuation_order_beta: float = 0.05
    """Softplus temperature for the medium attenuation channel-order prior."""
    medium_attenuation_order_use_low_transmission_weight: bool = True
    """Weight attenuation-order loss more in low estimated transmission regions."""
    low_transmission_threshold: float = 0.35
    """Transmission midpoint for low-transmission weighting."""
    low_transmission_temperature: float = 0.10
    """Temperature for low-transmission weighting."""


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
        medium_out_dim = 12 if getattr(self.config, "infinite_water_enabled", False) else 9
        if num_layers_medium > 1:
            self.medium_mlp = MLP(
                in_dim=medium_input_dim,
                num_layers=num_layers_medium,
                layer_width=hidden_dim_medium,
                out_dim=medium_out_dim,
                activation=nn.Sigmoid(),
                out_activation=None,
                implementation=self.config.mlp_type,
            )
        else:
            self.medium_mlp = nn.Linear(medium_input_dim, medium_out_dim)
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
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
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
        self.cleanup_alpha_accum = None
        self.cleanup_ownership_accum = None
        self.cleanup_sample_counts = None
        self.cleanup_current_alpha = None
        self.cleanup_current_ownership = None
        self.cleanup_last_stats = None
        self.last_active_sh_degree = 0

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
            self._accumulate_cleanup_evidence(visible_mask)

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
            cleanup_cull_mask = self._compute_cleanup_candidate_mask()
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
                        splits if cleanup_cull_mask is None else (splits | cleanup_cull_mask),
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )                
                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians(cleanup_cull_mask)
            elif cleanup_cull_mask is not None:
                deleted_mask = self.cull_gaussians(cleanup_cull_mask)
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
            self._reset_cleanup_accumulators()

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
        # The order of these matters
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

    def _predict_medium(
        self,
        *,
        camera: Cameras,
        rotation_world_from_camera: torch.Tensor,
        height: int,
        width: int,
        cx: float,
        cy: float,
        depth_context: Optional[torch.Tensor] = None,
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
            depth_context=depth_context,
            enable_b_inf=getattr(self.config, "infinite_water_enabled", False),
        )

    def _uses_medium_depth_context(self) -> bool:
        return "depth" in getattr(self.config, "medium_context_mode", "dir_only")

    def _normalize_medium_depth_context(self, depth: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "medium_depth_context_detach", True):
            depth = depth.detach()

        if not getattr(self.config, "medium_depth_context_normalize", True):
            return depth

        valid = torch.isfinite(depth) & (depth > 0)
        if valid.any():
            valid_depth = depth[valid]
            mode = getattr(self.config, "medium_depth_context_normalize_mode", "p95")
            if mode == "p95":
                scale = torch.quantile(valid_depth, 0.95)
            elif mode == "max":
                scale = valid_depth.max()
            else:
                raise ValueError(f"Unknown medium_depth_context_normalize_mode: {mode}")
            scale = scale.clamp_min(1e-6)
            return (depth / scale).clamp(0.0, 2.0)

        return torch.zeros_like(depth)

    def _m2_ramp_weight(self, weight: float) -> float:
        if weight <= 0.0:
            return 0.0
        start = getattr(self.config, "infinite_water_loss_start_step", 1000)
        ramp = max(getattr(self.config, "infinite_water_loss_ramp_steps", 3000), 1)
        if self.step < start:
            return 0.0
        return float(weight) * min((self.step - start) / ramp, 1.0)

    def _infinite_water_capacity_support(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        support = outputs["m_inf"].detach()
        mode = getattr(self.config, "infinite_water_capacity_support_mode", "m_inf")
        if mode == "m_inf":
            capacity_support = support
        elif mode == "hit_alpha":
            capacity_support = support * (1.0 - outputs["hit_q_alpha"].detach()).clamp(0.0, 1.0)
        elif mode == "hit":
            capacity_support = support * (1.0 - outputs["hit_confidence"].detach()).clamp(0.0, 1.0)
        elif mode == "hit_squared":
            gate = (1.0 - outputs["hit_confidence"].detach()).clamp(0.0, 1.0)
            capacity_support = support * gate.square()
        else:
            raise ValueError(f"Unknown infinite_water_capacity_support_mode: {mode}")

        if bool(getattr(self.config, "infinite_water_hit_protection_enabled", False)):
            protection = self._infinite_water_hit_object_protection(outputs).detach()
            capacity_floor = float(getattr(self.config, "infinite_water_capacity_floor", 0.50))
            capacity_floor = min(max(capacity_floor, 0.0), 1.0)
            floor_gate = 1.0 - (1.0 - capacity_floor) * protection
            capacity_support = capacity_support * floor_gate.clamp(capacity_floor, 1.0)
        return capacity_support

    def _infinite_water_hit_object_protection(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        hit_confidence = outputs["hit_confidence"].detach()
        if not bool(getattr(self.config, "infinite_water_hit_protection_enabled", False)):
            return torch.zeros_like(hit_confidence)
        start = int(getattr(self.config, "infinite_water_hit_protection_start_step", 0))
        if self.step < start:
            return torch.zeros_like(hit_confidence)
        threshold = float(getattr(self.config, "infinite_water_hit_protection_threshold", 0.80))
        temp = max(float(getattr(self.config, "infinite_water_hit_protection_temp", 0.05)), 1e-6)
        return torch.sigmoid((hit_confidence - threshold) / temp).clamp(0.0, 1.0)

    def _infinite_water_capacity_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        mode = getattr(self.config, "infinite_water_capacity_loss_mode", "current")
        accumulation = outputs["accumulation"]
        if mode == "none":
            return accumulation.new_zeros(())
        if mode == "current":
            support = outputs.get("m_capacity", outputs["m_inf"]).detach()
            return (support * accumulation).sum() / support.sum().clamp_min(1e-6)

        depth_support = outputs["m_inf_depth_evidence"].detach()
        support_norm = depth_support.sum().clamp_min(1e-6)
        if mode == "depth_monotonic":
            penalty = accumulation
        elif mode == "relu_budget":
            budget = float(getattr(self.config, "infinite_water_capacity_budget", 0.05))
            penalty = torch.relu(accumulation - budget)
        elif mode == "softplus_budget":
            budget = float(getattr(self.config, "infinite_water_capacity_budget", 0.05))
            temp = max(float(getattr(self.config, "infinite_water_capacity_budget_temp", 0.02)), 1e-6)
            penalty = F.softplus((accumulation - budget) / temp) * temp
        else:
            raise ValueError(f"Unknown infinite_water_capacity_loss_mode: {mode}")
        return (depth_support * penalty).sum() / support_norm

    def _appearance_enabled(self) -> bool:
        return bool(getattr(self.config, "constrained_appearance_enabled", False))

    def _appearance_ramp_weight(self, weight: float) -> float:
        if weight <= 0.0:
            return 0.0
        start = getattr(self.config, "appearance_loss_start_step", 1000)
        ramp = max(getattr(self.config, "appearance_loss_ramp_steps", 3000), 1)
        if self.step < start:
            return 0.0
        return float(weight) * min((self.step - start) / ramp, 1.0)

    def _get_active_sh_degree(self) -> int:
        if not (
            self._appearance_enabled()
            and getattr(self.config, "appearance_sh_delay_enabled", False)
        ):
            return min(self.step // self.config.sh_degree_interval, self.config.sh_degree)

        start = getattr(self.config, "appearance_sh_delay_start_step", 3000)
        interval = max(getattr(self.config, "appearance_sh_delay_interval", 2000), 1)
        if self.step < start:
            return 0
        return min((self.step - start) // interval + 1, self.config.sh_degree)

    def _cleanup_enabled(self) -> bool:
        return bool(getattr(self.config, "gaussian_cleanup_enabled", False))

    def _reset_cleanup_accumulators(self) -> None:
        self.cleanup_alpha_accum = None
        self.cleanup_ownership_accum = None
        self.cleanup_sample_counts = None
        self.cleanup_current_alpha = None
        self.cleanup_current_ownership = None

    def _cache_cleanup_pixel_evidence(
        self,
        *,
        accumulation: torch.Tensor,
        ownership: Optional[torch.Tensor],
        height: int,
        width: int,
    ) -> None:
        if not self._cleanup_enabled() or not self.training:
            return

        self.cleanup_current_alpha = sample_pixel_map_at_gaussians(
            accumulation.detach(),
            self.xys.detach(),
            self.radii.detach(),
            height,
            width,
        )
        if ownership is None:
            self.cleanup_current_ownership = torch.zeros_like(self.cleanup_current_alpha)
        else:
            self.cleanup_current_ownership = sample_pixel_map_at_gaussians(
                ownership.detach(),
                self.xys.detach(),
                self.radii.detach(),
                height,
                width,
            )

    def _accumulate_cleanup_evidence(self, visible_mask: torch.Tensor) -> None:
        if not self._cleanup_enabled() or self.cleanup_current_alpha is None:
            return
        if self.cleanup_alpha_accum is None or self.cleanup_alpha_accum.shape[0] != self.num_points:
            self.cleanup_alpha_accum = torch.zeros_like(self.radii, dtype=torch.float32)
            self.cleanup_ownership_accum = torch.zeros_like(self.radii, dtype=torch.float32)
            self.cleanup_sample_counts = torch.zeros_like(self.radii, dtype=torch.float32)

        assert self.cleanup_ownership_accum is not None and self.cleanup_sample_counts is not None
        visible_mask = visible_mask.reshape(-1)
        self.cleanup_alpha_accum[visible_mask] += self.cleanup_current_alpha[visible_mask].float()
        self.cleanup_ownership_accum[visible_mask] += self.cleanup_current_ownership[visible_mask].float()
        self.cleanup_sample_counts[visible_mask] += 1.0

    def _should_run_cleanup(self) -> bool:
        if not self._cleanup_enabled():
            return False
        if self.step < getattr(self.config, "gaussian_cleanup_start_step", 12000):
            return False
        interval = max(getattr(self.config, "gaussian_cleanup_interval", 500), 1)
        return self.step % interval == 0

    def _compute_cleanup_candidate_mask(self) -> Optional[torch.Tensor]:
        if not self._should_run_cleanup():
            return None
        if self.xys_grad_norm is None or self.vis_counts is None or self.depths_accum is None:
            return None

        visibility = self.vis_counts.clamp_min(1)
        contribution = (self.xys_grad_norm / visibility) * 0.5 * max(self.last_size[0], self.last_size[1])
        avg_depth = self.depths_accum / visibility

        sampled_alpha = None
        sampled_ownership = None
        if self.cleanup_alpha_accum is not None and self.cleanup_sample_counts is not None:
            counts = self.cleanup_sample_counts.clamp_min(1.0)
            sampled_alpha = self.cleanup_alpha_accum / counts
            if self.cleanup_ownership_accum is not None:
                sampled_ownership = self.cleanup_ownership_accum / counts

        require_depth = (
            bool(getattr(self.config, "gaussian_cleanup_require_depth_gate", False))
            and getattr(self.config, "gaussian_cleanup_depth_threshold", 0.0) > 0.0
        )
        dry_run = bool(getattr(self.config, "gaussian_cleanup_dry_run", True))
        cleanup_mask, stats = build_cleanup_candidate_mask(
            step=self.step,
            opacities=self.opacities,
            contribution=contribution,
            visibility=visibility,
            avg_depth=avg_depth,
            sampled_alpha=sampled_alpha,
            sampled_ownership=sampled_ownership,
            min_visibility=getattr(self.config, "gaussian_cleanup_visibility_min_count", 2),
            contribution_threshold=getattr(self.config, "gaussian_cleanup_contribution_threshold", 1e-4),
            opacity_threshold=getattr(self.config, "gaussian_cleanup_opacity_threshold", 0.08),
            alpha_threshold=getattr(self.config, "gaussian_cleanup_alpha_threshold", 0.25),
            depth_threshold=getattr(self.config, "gaussian_cleanup_depth_threshold", 0.0),
            ownership_threshold=getattr(self.config, "gaussian_cleanup_ownership_threshold", 0.35),
            require_alpha_gate=getattr(self.config, "gaussian_cleanup_require_alpha_gate", True),
            require_depth_gate=require_depth,
            require_ownership_gate=getattr(self.config, "gaussian_cleanup_require_ownership_gate", True),
            dry_run=dry_run,
        )
        self.cleanup_last_stats = stats
        CONSOLE.log(format_cleanup_stats(stats))
        if dry_run or stats.candidate_count == 0:
            return None
        return cleanup_mask

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
        if self._cleanup_enabled() and self.training:
            self.cleanup_current_alpha = None
            self.cleanup_current_ownership = None
        
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
                j_empty = torch.zeros_like(rgb)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb, 
                        "rgb_object": torch.zeros_like(rgb), "J": j_empty, "J_raw": j_empty,
                        "J_gaussian": j_empty, "J_gaussian_raw": j_empty,
                        "J_object": j_empty, "J_object_raw": j_empty,
                        "rgb_clear": j_empty, "rgb_clear_clamp": j_empty, "rgb_medium": medium_rgb, "pred_image": rgb,
                        "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}
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
            j_empty = torch.zeros_like(rgb)
            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb, 
                    "rgb_object": torch.zeros_like(rgb), "J": j_empty, "J_raw": j_empty,
                    "J_gaussian": j_empty, "J_gaussian_raw": j_empty,
                    "J_object": j_empty, "J_object_raw": j_empty,
                    "rgb_clear": j_empty, "rgb_clear_clamp": j_empty, "rgb_medium": medium_rgb, "pred_image": rgb,
                    "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}

        if self.training:
            self.xys.retain_grad()

        n = self._get_active_sh_degree()
        self.last_active_sh_degree = int(n)
        rgbs = compute_gaussian_colors(
            means=means_crop,
            features_dc=features_dc_crop,
            features_rest=features_rest_crop,
            camera_position=camera.camera_to_worlds[..., :3, 3],
            sh_degree=self.config.sh_degree,
            active_sh_degree=n,
        )
        sh_residual = None
        dc_rgb = None
        visible_mask = (self.radii > 0).reshape(-1)
        if self._appearance_enabled():
            sh_residual = compute_gaussian_sh_residual(
                means=means_crop,
                features_dc=features_dc_crop,
                features_rest=features_rest_crop,
                camera_position=camera.camera_to_worlds[..., :3, 3],
                sh_degree=self.config.sh_degree,
                active_sh_degree=n,
            )
            dc_rgb = SH2RGB(features_dc_crop) if self.config.sh_degree > 0 else torch.sigmoid(features_dc_crop)

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

        if self._uses_medium_depth_context():
            depth_seed_render = self.underwater_rasterizer.rasterize(  # type: ignore
                xys=self.xys,
                xys_grad_abs=self.xys_grad_abs,
                depths=depths,
                radii=self.radii,
                conics=conics,
                num_tiles_hit=num_tiles_hit,
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
            depth_context = self._normalize_medium_depth_context(depth_seed_render.depth)
            medium = self._predict_medium(
                camera=camera,
                rotation_world_from_camera=R,
                height=H,
                width=W,
                cx=cx,
                cy=cy,
                depth_context=depth_context,
            )
            medium_rgb = medium.rgb
            medium_bs = medium.bs
            medium_attn = medium.attn

        render = self.underwater_rasterizer.rasterize(  # type: ignore
            xys=self.xys,
            xys_grad_abs=self.xys_grad_abs,
            depths=depths,
            radii=self.radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
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
        rgb_clear = render.rgb_clear
        j_gaussian_raw = render.j_raw
        j_gaussian = render.j_gaussian
        j_object_raw = render.j_raw
        j_object = j_gaussian
        tail_weight_last = render.final_transmittance * torch.exp(-medium_bs * render.last_depth)
        tail_medium_original = tail_weight_last * medium_rgb
        rgb_medium_finite = render.rgb_medium - tail_medium_original
        hit_q_alpha = torch.sigmoid(
            (render.accumulation - self.config.infinite_water_hit_alpha_threshold)
            / max(self.config.infinite_water_hit_alpha_temp, 1e-6)
        ).clamp(0.0, 1.0)
        hit_q_conc = torch.exp(
            -render.depth_std_relative.detach() / max(self.config.infinite_water_hit_concentration_kappa, 1e-6)
        ).clamp(0.0, 1.0)
        hit_confidence = (hit_q_alpha * hit_q_conc).clamp(0.0, 1.0)
        ownership = None
        if getattr(self.config, "infinite_water_enabled", False):
            if medium.b_inf is None:
                raise RuntimeError("infinite_water_enabled=True requires medium.b_inf output")
            ownership = compute_infinite_water_ownership(
                accumulation=render.accumulation,
                depth=render.depth,
                rgb_near=render.rgb,
                b_inf=medium.b_inf,
                mode=self.config.infinite_water_ownership_mode,
                detach_evidence=self.config.infinite_water_detach_evidence,
                alpha_power=self.config.infinite_water_alpha_power,
                depth_mid=self.config.infinite_water_depth_mid,
                depth_temp=self.config.infinite_water_depth_temp,
                color_temp=self.config.infinite_water_color_temp,
                depth_normalize_mode=self.config.infinite_water_depth_normalize_mode,
                occupancy_limited=self.config.infinite_water_occupancy_limited,
            )
            compose_mode = getattr(self.config, "infinite_water_compose_mode", "rgb_mix")
            if compose_mode == "none":
                rgb = render.rgb
                rgb_clear = render.rgb_clear
                j_object_raw = render.j_raw
            elif compose_mode == "rgb_mix":
                m_obj_eff = 1.0 - ownership.m_inf_eff
                rgb = m_obj_eff * render.rgb + ownership.m_inf_eff * medium.b_inf
                rgb_clear = m_obj_eff * render.rgb_clear
                j_object_raw = m_obj_eff * render.j_raw
            elif compose_mode == "tail_approx":
                if getattr(self.config, "infinite_water_occupancy_limited", True):
                    tail_gate = (1.0 - render.accumulation).detach().clamp(0.0, 1.0)
                else:
                    tail_gate = torch.ones_like(render.accumulation)
                rgb = render.rgb + ownership.m_inf * tail_gate * (medium.b_inf - medium_rgb)
                j_object_raw = (1.0 - ownership.m_inf_eff) * render.j_raw
            elif compose_mode == "closed_tail":
                tail_color = (1.0 - ownership.m_inf_eff) * medium_rgb + ownership.m_inf_eff * medium.b_inf
                rgb = render.rgb_object + rgb_medium_finite + tail_weight_last * tail_color
                rgb_clear = render.rgb_clear
                j_object_raw = render.j_raw
            else:
                raise ValueError(f"Unknown infinite_water_compose_mode: {compose_mode}")
            j_object = torch.clamp(j_object_raw, 0.0, 1.0)

        cleanup_ownership = None
        if ownership is not None:
            ownership_source = getattr(self.config, "gaussian_cleanup_ownership_source", "m_inf_eff")
            if ownership_source == "m_inf":
                cleanup_ownership = ownership.m_inf
            elif ownership_source == "m_inf_eff":
                cleanup_ownership = ownership.m_inf_eff
            else:
                raise ValueError(f"Unknown gaussian_cleanup_ownership_source: {ownership_source}")

        self._cache_cleanup_pixel_evidence(
            accumulation=render.accumulation,
            ownership=cleanup_ownership,
            height=H,
            width=W,
        )

        low_trans_weight = None
        pixel_low_trans_weight = None
        if self._appearance_enabled():
            sampled_attn = sample_pixel_map_at_gaussians(
                medium_attn.mean(dim=-1, keepdim=True).detach(),
                self.xys.detach(),
                self.radii.detach(),
                H,
                W,
            )
            low_trans_weight = low_transmission_weights(
                sampled_attn=sampled_attn,
                depths=depths.detach(),
                threshold=getattr(self.config, "low_transmission_threshold", 0.35),
                temperature=getattr(self.config, "low_transmission_temperature", 0.10),
            )
            pixel_low_trans_weight = low_transmission_weights(
                sampled_attn=medium_attn.mean(dim=-1, keepdim=True).detach(),
                depths=render.depth.detach(),
                threshold=getattr(self.config, "low_transmission_threshold", 0.35),
                temperature=getattr(self.config, "low_transmission_temperature", 0.10),
            ).reshape(H, W, 1)

        outputs = {
            "rgb": rgb,
            "depth": render.depth,
            "depth_second_moment": render.depth_second_moment,
            "depth_variance": render.depth_variance,
            "depth_std_relative": render.depth_std_relative,
            "first_depth": render.first_depth,
            "last_depth": render.last_depth,
            "final_transmittance": render.final_transmittance,
            "hit_q_alpha": hit_q_alpha,
            "hit_q_conc": hit_q_conc,
            "hit_confidence": hit_confidence,
            "accumulation": render.accumulation,
            "background": medium_rgb,
            "rgb_object": render.rgb_object,
            "J": j_gaussian,
            "J_raw": j_gaussian_raw,
            "J_gaussian": j_gaussian,
            "J_gaussian_raw": j_gaussian_raw,
            "J_object": j_object,
            "J_object_raw": j_object_raw,
            "rgb_clear": rgb_clear,
            "rgb_clear_clamp": j_gaussian,
            "rgb_medium": render.rgb_medium,
            "rgb_medium_finite": rgb_medium_finite,
            "tail_weight_last": tail_weight_last,
            "tail_medium_original": tail_medium_original,
            "pred_image": rgb,
            "medium_rgb": medium_rgb,
            "medium_bs": medium_bs,
            "medium_attn": medium_attn,
        }
        if self._appearance_enabled():
            outputs["appearance_active_sh_degree"] = torch.tensor(float(n), device=self.device)
            outputs["appearance_visible_mask"] = visible_mask.detach()
            outputs["appearance_sh_residual"] = sh_residual
            outputs["appearance_dc_rgb"] = dc_rgb
            outputs["appearance_low_trans_weight"] = low_trans_weight
            outputs["appearance_pixel_low_trans_weight"] = pixel_low_trans_weight
        if getattr(self.config, "infinite_water_enabled", False):
            outputs["b_inf"] = medium.b_inf
            outputs["m_inf"] = ownership.m_inf
            outputs["m_inf_eff"] = ownership.m_inf_eff
            outputs["m_support"] = ownership.m_inf
            outputs["m_render"] = ownership.m_inf_eff
            outputs["hit_object_protection"] = self._infinite_water_hit_object_protection(outputs)
            outputs["m_capacity"] = self._infinite_water_capacity_support(outputs)
            outputs["m_inf_alpha_evidence"] = ownership.alpha_evidence
            outputs["m_inf_depth_evidence"] = ownership.depth_evidence
            outputs["m_inf_color_evidence"] = ownership.color_evidence
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
            if "b_inf" in outputs:
                metrics_dict[f"b_inf_{i}"] = outputs["b_inf"][:, :, i].mean()
        if "m_inf" in outputs:
            metrics_dict["m_inf_mean"] = outputs["m_inf"].mean()
            metrics_dict["m_inf_eff_mean"] = outputs["m_inf_eff"].mean()
            metrics_dict["m_capacity_mean"] = outputs["m_capacity"].mean()
        if "hit_confidence" in outputs:
            metrics_dict["hit_q_alpha_mean"] = outputs["hit_q_alpha"].mean()
            metrics_dict["hit_q_conc_mean"] = outputs["hit_q_conc"].mean()
            metrics_dict["hit_confidence_mean"] = outputs["hit_confidence"].mean()
            metrics_dict["depth_std_relative_mean"] = outputs["depth_std_relative"].mean()
        if "hit_object_protection" in outputs:
            metrics_dict["hit_object_protection_mean"] = outputs["hit_object_protection"].mean()
        j_metric = torch.clamp(outputs["J"], 0.0, 1.0)
        metrics_dict["J_white_ratio"] = (j_metric > 0.95).all(dim=-1).float().mean()
        metrics_dict["J_saturation_ratio"] = (j_metric > 0.98).float().mean()
        j_red_dominance = j_metric[..., 0] - torch.maximum(j_metric[..., 1], j_metric[..., 2])
        metrics_dict["J_red_dominance_ratio"] = (j_red_dominance > 0.05).float().mean()
        j_blue_dominance = j_metric[..., 2] - torch.maximum(j_metric[..., 0], j_metric[..., 1])
        metrics_dict["J_blue_dominance_ratio"] = (j_blue_dominance > 0.05).float().mean()
        rgb_clear_metric = torch.clamp(outputs["rgb_clear"], 0.0, 1.0)
        metrics_dict["rgb_clear_legacy_white_ratio"] = (rgb_clear_metric > 0.95).all(dim=-1).float().mean()
        metrics_dict["rgb_clear_legacy_saturation_ratio"] = (rgb_clear_metric > 0.98).float().mean()
        if "appearance_active_sh_degree" in outputs:
            metrics_dict["appearance_active_sh_degree"] = outputs["appearance_active_sh_degree"]
        if "appearance_sh_residual" in outputs:
            metrics_dict["appearance_sh_residual_abs_mean"] = outputs["appearance_sh_residual"].abs().mean()
        if "appearance_dc_rgb" in outputs:
            metrics_dict["appearance_dc_rgb_mean"] = outputs["appearance_dc_rgb"].mean()
            metrics_dict["appearance_dc_rgb_gt_1_ratio"] = (outputs["appearance_dc_rgb"] > 1.0).float().mean()
            dc_rgb = outputs["appearance_dc_rgb"]
            dc_red_dom = dc_rgb[..., 0] - torch.maximum(dc_rgb[..., 1], dc_rgb[..., 2])
            dc_blue_dom = dc_rgb[..., 2] - torch.maximum(dc_rgb[..., 0], dc_rgb[..., 1])
            metrics_dict["appearance_dc_red_dominance_ratio"] = (dc_red_dom > 0.05).float().mean()
            metrics_dict["appearance_dc_blue_dominance_ratio"] = (dc_blue_dom > 0.05).float().mean()
        if self.cleanup_last_stats is not None:
            stats = self.cleanup_last_stats
            metrics_dict["gaussian_cleanup_candidate_count"] = stats.candidate_count
            metrics_dict["gaussian_cleanup_candidate_fraction"] = stats.candidate_fraction
            metrics_dict["gaussian_cleanup_mean_contribution"] = stats.mean_contribution
            metrics_dict["gaussian_cleanup_mean_ownership"] = stats.mean_sampled_ownership
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

        loss_dict = {
            "main_loss": reconstruction_loss(
                gt_img=gt_img,
                pred_img=pred_img,
                main_loss=self.config.main_loss,
                ssim_loss=self.config.ssim_loss,
                ssim_lambda=self.config.ssim_lambda,
                ssim_metric=self.ssim,
            ),
        }

        if getattr(self.config, "infinite_water_enabled", False) and "m_inf" in outputs:
            support = outputs["m_inf"].detach()
            support_norm = support.sum().clamp_min(1e-6)

            binf_weight = self._m2_ramp_weight(self.config.lambda_infinite_water_binf_rgb)
            if binf_weight > 0.0:
                loss_dict["infinite_water_binf_rgb_loss"] = binf_weight * (
                    support * torch.abs(outputs["b_inf"] - gt_img)
                ).sum() / (support_norm * 3.0)

            accum_weight = self._m2_ramp_weight(self.config.lambda_infinite_water_accumulation_zero)
            if accum_weight > 0.0 and getattr(self.config, "infinite_water_capacity_loss_mode", "current") != "none":
                loss_dict["infinite_water_accumulation_zero_loss"] = accum_weight * self._infinite_water_capacity_loss(outputs)

            near_weight = self._m2_ramp_weight(self.config.lambda_infinite_water_near_zero)
            if near_weight > 0.0:
                near_rgb = outputs["rgb_object"] + outputs["rgb_medium"]
                loss_dict["infinite_water_near_zero_loss"] = near_weight * (
                    support * torch.abs(near_rgb)
                ).sum() / (support_norm * 3.0)

        if self._appearance_enabled() and "appearance_visible_mask" in outputs:
            sh_weight = self._appearance_ramp_weight(self.config.lambda_sh_residual_mean)
            if sh_weight > 0.0 and "appearance_sh_residual" in outputs:
                loss_dict["appearance_sh_residual_mean_loss"] = sh_weight * sh_residual_mean_anchor_loss(
                    outputs["appearance_sh_residual"],
                    outputs["appearance_visible_mask"],
                )

            dc_weight = self._appearance_ramp_weight(self.config.lambda_dc_softclip)
            if dc_weight > 0.0 and "appearance_dc_rgb" in outputs:
                low_trans_weight = (
                    outputs.get("appearance_low_trans_weight")
                    if getattr(self.config, "dc_softclip_use_low_transmission_weight", True)
                    else None
                )
                loss_dict["appearance_dc_softclip_loss"] = dc_weight * dc_softclip_loss(
                    dc_rgb=outputs["appearance_dc_rgb"],
                    visible_mask=outputs["appearance_visible_mask"],
                    low_transmission_weight=low_trans_weight,
                    threshold=self.config.dc_softclip_threshold,
                    beta=self.config.dc_softclip_beta,
                )

            dc_balance_weight = self._appearance_ramp_weight(self.config.lambda_dc_channel_balance)
            if dc_balance_weight > 0.0 and "appearance_dc_rgb" in outputs:
                low_trans_weight = (
                    outputs.get("appearance_low_trans_weight")
                    if getattr(self.config, "dc_channel_balance_use_low_transmission_weight", True)
                    else None
                )
                loss_dict["appearance_dc_channel_balance_loss"] = dc_balance_weight * dc_channel_balance_loss(
                    dc_rgb=outputs["appearance_dc_rgb"],
                    visible_mask=outputs["appearance_visible_mask"],
                    low_transmission_weight=low_trans_weight,
                    margin=self.config.dc_channel_balance_margin,
                    beta=self.config.dc_channel_balance_beta,
                )

            attn_order_weight = self._appearance_ramp_weight(self.config.lambda_medium_attenuation_order)
            if attn_order_weight > 0.0:
                pixel_low_trans_weight = (
                    outputs.get("appearance_pixel_low_trans_weight")
                    if getattr(self.config, "medium_attenuation_order_use_low_transmission_weight", True)
                    else None
                )
                loss_dict["appearance_medium_attenuation_order_loss"] = (
                    attn_order_weight
                    * medium_attenuation_order_loss(
                        medium_attn=outputs["medium_attn"],
                        low_transmission_weight=pixel_low_trans_weight,
                        margin=self.config.medium_attenuation_order_margin,
                        beta=self.config.medium_attenuation_order_beta,
                    )
                )

        return loss_dict

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

        j_metric = torch.clamp(outputs["J"], 0.0, 1.0)
        metrics_dict["J_white_ratio"] = float((j_metric > 0.95).all(dim=-1).float().mean().item())
        metrics_dict["J_saturation_ratio"] = float((j_metric > 0.98).float().mean().item())
        j_red_dominance = j_metric[..., 0] - torch.maximum(j_metric[..., 1], j_metric[..., 2])
        metrics_dict["J_red_dominance_ratio"] = float((j_red_dominance > 0.05).float().mean().item())
        j_blue_dominance = j_metric[..., 2] - torch.maximum(j_metric[..., 0], j_metric[..., 1])
        metrics_dict["J_blue_dominance_ratio"] = float((j_blue_dominance > 0.05).float().mean().item())

        images_dict = {
            "gt": output_gt_rgb,
            "rgb_medium": outputs["rgb_medium"],
            "rgb_object": outputs["rgb_object"],
            "depth": outputs["depth"],
            "rgb": outputs["rgb"],
            "J": outputs["J"],
            "J_gaussian": outputs["J_gaussian"],
            "J_object": outputs["J_object"],
            "rgb_clear_legacy": outputs["rgb_clear"],
        }
        if "b_inf" in outputs:
            images_dict["b_inf"] = outputs["b_inf"]
            images_dict["m_inf"] = outputs["m_inf"].expand_as(outputs["rgb"])
            images_dict["m_inf_eff"] = outputs["m_inf_eff"].expand_as(outputs["rgb"])
        return metrics_dict, images_dict
