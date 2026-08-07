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
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from water_splatting.attribution import (
    accumulation_clearance_amplifier,
    budgeted_capacity_loss,
    build_residual_gated_halo_support,
    build_route_capacity_support,
    build_training_routed_prediction,
    clear_proxy_chroma_loss,
    clear_proxy_luma_budget_loss,
    core_zero_capacity_loss,
    rgb_luma_budget_loss,
    support_coverage_stats,
    weighted_rgb_l1,
)
from water_splatting.cleanup import build_cleanup_candidate_mask, format_cleanup_stats, sample_pixel_map_at_gaussians
from water_splatting.fields import (
    DirectionConditionedMediumField,
    MediumFieldOutput,
    compute_dual_gaussian_colors,
    compute_gaussian_colors,
    compute_gaussian_sh_residual,
    get_medium_context_extra_dim,
)
from water_splatting.losses import (
    dc_channel_balance_loss,
    dc_softclip_loss,
    effective_background_mask,
    low_transmission_weights,
    masked_rgb_l1_loss,
    medium_attenuation_order_loss,
    reconstruction_loss,
    sh_residual_mean_anchor_loss,
)
from water_splatting.medium_calibration import compute_gmvc_training_terms, load_gmvc_training_bank
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
    b_inf_mode: Literal["implicit", "tied", "bounded_residual", "independent"] = "implicit"
    """Backscatter closure B_inf mode. implicit preserves M1; tied uses B_inf=A without an extra head."""
    b_inf_residual_scale: float = 0.02
    """Bounded-residual scale for B_inf around medium_rgb/A."""
    lambda_background_water_color: float = 0.0
    """Backscatter closure background-water color supervision weight."""
    lambda_foreground_transmission_reconstruction: float = 0.0
    """Extra foreground reconstruction weight for low-transmission pixels."""
    foreground_transmission_gamma: float = 1.0
    """Exponent used by foreground transmission-aware reconstruction."""
    foreground_transmission_max_weight: float = 4.0
    """Maximum per-channel foreground reconstruction multiplier."""
    foreground_transmission_detach_weight: bool = True
    """Detach foreground transmission weights from the weighted reconstruction loss."""
    lambda_pseudo_depth: float = 0.0
    """Reserved pseudo-depth rank-consistency loss weight. Off by default."""
    lambda_medium_context_residual: float = 0.0
    """Reserved base-residual medium context regularization weight. Off by default."""
    medium_predictor_mode: Literal["single", "base_residual"] = "single"
    """Medium predictor structure flag. single preserves the current M1/M2 predictor."""
    direct_optical_depth_scale: float = 1.0
    """Scale applied only to direct attenuation beta_D before underwater rasterization."""
    medium_background_supervision_enabled: bool = False
    """Enable training-only direct medium_rgb supervision on detached background-water masks."""
    medium_background_supervision_lambda: float = 0.0
    """Weight for detached background-water medium_rgb supervision."""
    medium_background_supervision_exclude_boundary: bool = True
    """Exclude boundary pixels when building the effective background supervision mask."""
    medium_background_supervision_hit_exclusion_threshold: float = -1.0
    """If >=0, exclude background pixels whose detached hit_confidence is above this threshold."""
    gmvc_enabled: bool = False
    """Enable Geometry-Anchored Multi-View Medium Calibration training terms."""
    gmvc_diagnostic_only: bool = False
    """If True, compute GMVC diagnostics but do not add GMVC losses to training."""
    gmvc_track_bank_path: Optional[str] = None
    """Path to a detached GMVC training track bank built from an M1 checkpoint."""
    gmvc_start_step: int = 10000
    """First step where GMVC losses can ramp in."""
    gmvc_stop_step: int = 15000
    """First step where GMVC losses are disabled."""
    gmvc_ramp_steps: int = 500
    """Linear ramp length for GMVC auxiliary losses."""
    lambda_gmvc_j: float = 0.0
    """Weight for inverse-radiance consistency against detached track consensus."""
    lambda_gmvc_range: float = 0.0
    """Weight for log-domain medium attenuation/backscatter range consistency."""
    lambda_gmvc_binf: float = 0.0
    """Weight for weak B_inf/medium_rgb scene-center consistency."""
    lambda_gmvc_intrinsic: float = 0.0
    """Weight for renderer intrinsic Gaussian color consistency against detached track consensus."""
    lambda_gmvc_residual_budget: float = 0.0
    """Weight for online scene-center residual budget on medium attenuation/backscatter/B_inf."""
    lambda_gmvc_fixed_closure: float = 0.0
    """Weight for online fixed-denominator cross-view medium closure."""
    gmvc_v2_enabled: bool = False
    """Enable symmetric current-medium GMVC-V2 diagnostics/losses."""
    lambda_gmvc_profile: float = 0.0
    """Weight for GMVC-V2 profiled shared intrinsic radiance loss."""
    lambda_gmvc_symmetric_closure: float = 0.0
    """Weight for weak GMVC-V2 symmetric current-medium closure."""
    gmvc_profile_detach_j_star: bool = True
    """Detach analytic per-track J* inside the GMVC-V2 profiled-radiance loss."""
    gmvc_profile_loss_mode: Literal["charbonnier", "irls_l2"] = "charbonnier"
    """Profile objective form. irls_l2 aligns the analytic J* solve and the outer profile loss."""
    gmvc_profile_track_balanced: bool = False
    """Average profiled-radiance loss per track before averaging across tracks."""
    gmvc_profile_irls_delta: float = 0.03
    """Robust residual scale for GMVC IRLS profile."""
    gmvc_profile_irls_max_weight: float = 1.0
    """Maximum normalized IRLS observation weight."""
    gmvc_profile_min_hessian: float = 0.0
    """Minimum per-track current-medium profile Hessian for GMVC profile loss."""
    gmvc_profile_min_transmission_span: float = 0.0
    """Minimum per-track current-medium transmission span for GMVC profile loss."""
    gmvc_profile_min_depth_span_rel: float = 0.0
    """Minimum relative depth span for GMVC profile loss."""
    gmvc_profile_observability_weight_enabled: bool = False
    """Use detached track observability to reweight GMVC profile tracks."""
    gmvc_profile_observability_weight_mode: Literal["dtb", "dtb_view", "dtb_view_geom"] = "dtb_view"
    """Track observability score inputs: depth/transmission/backscatter, optional view span and geometry."""
    gmvc_profile_observability_weight_power: float = 1.0
    """Exponent applied to the normalized observability track score."""
    gmvc_profile_observability_weight_min: float = 0.25
    """Minimum normalized observability track weight."""
    gmvc_profile_observability_weight_max: float = 4.0
    """Maximum normalized observability track weight."""
    gmvc_v3_enabled: bool = False
    """Enable GMVC-V3 alternating medium/object calibration."""
    gmvc_v3_profile_schedule: Literal["constant", "stop", "linear_decay"] = "constant"
    """Schedule for the effective GMVC-V3 profile loss weight."""
    gmvc_v3_profile_decay_start_step: int = 13000
    """Global step where stop/linear_decay profile release begins."""
    gmvc_v3_profile_decay_end_step: int = 14000
    """Global step where linear_decay reaches the final profile scale."""
    gmvc_v3_profile_decay_final_scale: float = 0.0
    """Final profile scale for stop/linear_decay schedules."""
    gmvc_medium_hold_enabled: bool = False
    """Freeze all medium-owned parameters during a GMVC calibrated-medium hold interval."""
    gmvc_medium_hold_start_step: int = 13001
    """First global step where GMVC medium-hold freezing is active."""
    gmvc_medium_hold_stop_step: int = 15000
    """Last global step where GMVC medium-hold freezing is active."""
    lambda_gmvc_object: float = 0.0
    """Weight for online DC-only object calibration against profiled J*."""
    gmvc_v3_medium_steps: int = 4
    """Number of medium calibration steps in each GMVC-V3 alternation cycle."""
    gmvc_v3_object_steps: int = 1
    """Number of object DC calibration steps in each GMVC-V3 alternation cycle."""
    gmvc_v3_freeze_medium_on_object_phase: bool = False
    """Detach medium outputs during GMVC-V3 object phase for strict block-coordinate alternation."""
    gmvc_v3_object_phase_medium_grad_scale: float = 1.0
    """Gradient multiplier for medium outputs during GMVC-V3 object phase. Values are unchanged."""
    gmvc_v3_target_current_camera_tracks: bool = False
    """In object phase, sample GMVC tracks that contain the current training camera."""
    gmvc_v3_object_source: Literal["J_proxy_raw"] = "J_proxy_raw"
    """Rendered clear proxy used for GMVC-V3 object calibration."""
    gmvc_object_track_balanced: bool = True
    """Average GMVC-V3 object loss per track before averaging across tracks."""
    gmvc_object_j_clamp_min: float = -0.1
    """Minimum profiled J* value accepted as a GMVC-V3 object target."""
    gmvc_object_j_clamp_max: float = 1.1
    """Maximum profiled J* value accepted as a GMVC-V3 object target."""
    gmvc_object_min_hessian: float = 0.0
    """Minimum per-track profile Hessian for GMVC-V3 object targets."""
    gmvc_object_min_depth_span_rel: float = 0.05
    """Minimum relative depth span for GMVC-V3 object targets."""
    gmvc_v2_max_tracks_per_step: int = 512
    """Maximum full tracks sampled per step for GMVC-V2."""
    gmvc_v2_min_observations_per_track: int = 2
    """Minimum valid observations in a sampled track for GMVC-V2 losses."""
    gmvc_bounded_medium_enabled: bool = False
    """Apply explicit scene-centered bounded medium projection to renderer and GMVC point queries."""
    gmvc_bounded_medium_start_step: int = 10000
    """First step for ramping in bounded medium projection."""
    gmvc_bounded_medium_projection_steps: int = 500
    """Ramp length for bounded medium projection alpha."""
    gmvc_bounded_beta_log_scale: float = 0.15
    """Tanh residual scale for log betaD and log betaB in bounded medium mode."""
    gmvc_bounded_binf_logit_scale: float = 0.10
    """Tanh residual scale for logit B_inf/medium_rgb in bounded medium mode."""
    gmvc_bounded_init_from_first_batch: bool = True
    """Initialize bounded scene centers from the first rendered/query medium batch."""
    gmvc_residual_beta_log_scale: float = 0.15
    """Log residual budget scale for beta_D and beta_B."""
    gmvc_residual_binf_logit_scale: float = 0.10
    """Logit residual budget scale for tied B_inf/medium_rgb."""
    gmvc_residual_ema_momentum: float = 0.99
    """EMA momentum for online GMVC scene centers."""
    gmvc_closure_signal_floor: float = 0.03
    """Fixed-denominator closure signal floor."""
    gmvc_intrinsic_source: Literal["J_proxy_raw", "J_gaussian_raw", "J_raw", "J"] = "J_proxy_raw"
    """Rendered intrinsic output used by lambda_gmvc_intrinsic. J_proxy_raw has an active color backward path."""
    gmvc_intrinsic_use_dc_proxy: bool = True
    """Use DC-only Gaussian colors for the GMVC-triggered clear proxy render."""
    gmvc_grad_log_path: Optional[str] = None
    """Optional JSONL path for GMVC intrinsic-vs-RGB DC gradient diagnostics."""
    gmvc_grad_log_every: int = 100
    """Training-step interval for GMVC gradient JSONL diagnostics when gmvc_grad_log_path is set."""
    gmvc_grad_log_force_steps: Optional[str] = None
    """Comma-separated global steps that should always be written to the GMVC gradient JSONL."""
    gmvc_max_tracks_per_step: int = 4096
    """Maximum detached track observations sampled for the current camera per step."""
    gmvc_eps: float = 1e-4
    """Numerical epsilon for simplified inverse-radiance calculations."""
    gmvc_charbonnier_eps: float = 1e-6
    """Charbonnier epsilon for GMVC robust losses."""
    gmvc_j_clamp_min: float = -0.25
    """Minimum valid inverse-radiance value before an observation is ignored."""
    gmvc_j_clamp_max: float = 1.25
    """Maximum valid inverse-radiance value before an observation is ignored."""
    gmvc_range_log_scale: float = 0.25
    """Scale used to normalize log betaD/betaB deviations from track centers."""
    gmvc_detach_depth: bool = True
    """Detach rendered depth inside GMVC losses so geometry is only a correspondence source."""
    gmvc_seed: int = 42
    """Seed for deterministic per-step GMVC track subsampling."""
    gmvc_freeze_geometry: bool = False
    """Freeze means/scales/quats/opacities for GMVC continuation variants."""
    gmvc_train_features_dc: bool = False
    """Allow features_dc optimization while GMVC geometry freeze is active."""
    gmvc_train_features_rest: bool = False
    """Allow features_rest optimization while GMVC geometry freeze is active."""
    backscatter_region_mask_dir: Optional[str] = None
    """Directory containing view_XXXX_regions.pt masks with water/object/boundary keys."""
    background_water_mask_key: str = "water"
    """Mask key used for background-water B_inf supervision."""
    foreground_water_mask_key: str = "object"
    """Mask key used for foreground transmission-aware reconstruction."""
    backscatter_loss_start_step: int = 0
    """First step where backscatter-closure auxiliary losses are active."""
    backscatter_loss_ramp_steps: int = 0
    """Linear ramp length for backscatter-closure auxiliary losses."""
    lambda_background_medium_render: float = 0.0
    """Renderer-consistent background loss on rgb_medium_finite + rgb_tail."""
    lambda_background_tail_render: float = 0.0
    """Renderer-consistent background loss on rgb_tail only."""
    background_render_loss_start_step: int = 0
    """First step where renderer-consistent background losses are active."""
    background_render_loss_ramp_steps: int = 0
    """Linear ramp length for renderer-consistent background losses."""
    lambda_background_clear_gaussian: float = 0.0
    """Background clear-Gaussian suppression loss weight."""
    background_clear_loss_start_step: int = 3000
    """First step where background clear-Gaussian loss is active."""
    background_clear_loss_ramp_steps: int = 3000
    """Linear ramp length for background clear-Gaussian loss."""
    background_clear_use_raw_j: bool = True
    """Use raw unclamped J_gaussian for background clear-Gaussian suppression."""
    background_clear_exclude_boundary: bool = True
    """Exclude boundary pixels from background clear-Gaussian suppression."""
    background_clear_hit_exclusion_threshold: float = -1.0
    """If >=0, exclude background pixels whose hit_confidence is above this threshold."""
    lambda_background_clear_chroma: float = 0.0
    """Accumulation-gated water-chroma suppression on differentiable clear proxy."""
    background_clear_chroma_start_step: int = 10000
    """First step where proxy clear-chroma loss is active."""
    background_clear_chroma_ramp_steps: int = 1000
    """Linear ramp length for proxy clear-chroma loss."""
    background_clear_chroma_accumulation_max: float = 0.65
    """Detached accumulation gate midpoint for proxy clear-chroma loss."""
    background_clear_chroma_accumulation_temperature: float = 0.05
    """Detached accumulation gate temperature for proxy clear-chroma loss."""
    background_clear_chroma_margin: float = 0.02
    """Allowed clear-proxy chroma projection margin before penalty."""
    background_clear_chroma_medium_detach: bool = True
    """Detach medium chroma direction in proxy clear-chroma loss."""
    background_clear_chroma_use_medium_support: bool = False
    """Use medium-explainable capacity support instead of region masks for proxy chroma."""
    medium_explainability_enabled: bool = False
    """Enable training-only medium-explainable support and auxiliary losses."""
    medium_explainability_start_step: int = 2000
    """First step where medium explainability bootstrap loss may ramp."""
    medium_explainability_ramp_steps: int = 2000
    """Ramp length for medium explainability bootstrap loss."""
    lambda_medium_explainability: float = 0.0
    """Weight for medium color explainability supervision."""
    training_gradient_routing_enabled: bool = False
    """Use medium support to route reconstruction gradients during training only."""
    gradient_routing_start_step: int = 4000
    """First step where training-only gradient routing may ramp."""
    gradient_routing_ramp_steps: int = 1000
    """Ramp length for training-only gradient routing."""
    gradient_routing_min_scene_weight: float = 0.30
    """Minimum physical-renderer reconstruction weight in supported water pixels."""
    budgeted_capacity_enabled: bool = False
    """Enable dense support-weighted Gaussian accumulation budget loss."""
    budgeted_capacity_start_step: int = 4000
    """First step where budgeted capacity loss may ramp."""
    budgeted_capacity_ramp_steps: int = 1000
    """Ramp length for budgeted capacity loss."""
    budgeted_capacity_value: float = 0.05
    """Allowed Gaussian accumulation budget on medium-explainable support."""
    budgeted_capacity_temperature: float = 0.02
    """Softplus temperature for accumulation budget loss."""
    lambda_budgeted_capacity: float = 0.0
    """Weight for budgeted Gaussian capacity suppression."""
    budgeted_capacity_post_scale: float = 0.5
    """Capacity weight multiplier after proxy refinement begins."""
    core_zero_capacity_enabled: bool = False
    """Enable zero-target accumulation pressure on high-confidence open-water core."""
    lambda_core_zero_capacity: float = 0.0
    """Weight for core zero-target Gaussian capacity release."""
    core_zero_capacity_start_step: int = 1000
    """First step where core zero capacity may ramp."""
    core_zero_capacity_ramp_steps: int = 3000
    """Ramp length for core zero capacity."""
    core_zero_capacity_post_scale: float = 1.0
    """Core zero capacity multiplier after proxy refinement begins."""
    core_clearance_amplifier_enabled: bool = False
    """Amplify core-zero capacity as detached accumulation clears, with a nonzero floor."""
    core_clearance_amplifier_min: float = 0.30
    """Minimum clearance-amplifier pressure on high-accumulation core pixels."""
    core_clearance_amplifier_threshold: float = 0.20
    """Detached accumulation threshold where the clearance amplifier turns on."""
    core_clearance_amplifier_temperature: float = 0.05
    """Clearance-amplifier sigmoid temperature."""
    capacity_control_enabled: bool = False
    """Use an auxiliary accumulation render with controlled capacity gradients."""
    capacity_control_geometry_gradient_scale: float = 1.0
    """Scale budgeted-capacity gradients to projected geometry/depth/conics."""
    capacity_control_position_gradient_scale: float = -1.0
    """Override capacity xys gradient scale; negative values inherit geometry scale."""
    capacity_control_depth_gradient_scale: float = -1.0
    """Override capacity depth gradient scale; negative values inherit geometry scale."""
    capacity_control_footprint_gradient_scale: float = -1.0
    """Override capacity radii/conics gradient scale; negative values inherit geometry scale."""
    capacity_control_opacity_gradient_scale: float = 1.0
    """Scale budgeted-capacity gradients to Gaussian opacity."""
    capacity_control_scale_shrink_only: bool = False
    """Only keep budgeted-capacity log-scale gradients that shrink Gaussian footprints."""
    capacity_control_scale_shrink_clip_quantile: float = -1.0
    """Optional positive shrink-gradient quantile clamp; <=0 disables clipping."""
    capacity_control_scale_shrink_clip_value: float = 0.0
    """Optional absolute positive shrink-gradient clamp; <=0 disables this clamp."""
    capacity_conflict_gate_enabled: bool = False
    """Attenuate capacity opacity gradients when reconstruction wants opacity increased."""
    capacity_conflict_rho: float = 1.0
    """Residual capacity opacity-gradient multiplier on reconstruction-conflicting Gaussians."""
    capacity_conflict_rec_grad_threshold: float = 1e-10
    """Negative reconstruction opacity-gradient threshold used by capacity conflict gating."""
    halo_capacity_enabled: bool = False
    """Enable residual-gated halo capacity pressure around medium-explainable core water."""
    lambda_halo_capacity: float = 0.0
    """Weight for residual-gated halo capacity suppression."""
    halo_capacity_value: float = 0.03
    """Allowed Gaussian accumulation budget on residual-gated halo support."""
    halo_capacity_temperature: float = 0.02
    """Softplus temperature for halo capacity budget loss."""
    halo_capacity_start_step: int = 4000
    """First step where halo capacity loss may ramp."""
    halo_capacity_ramp_steps: int = 1000
    """Ramp length for halo capacity loss."""
    halo_capacity_post_scale: float = 0.5
    """Halo capacity multiplier after proxy refinement begins."""
    halo_chroma_margin: float = 0.015
    """Medium-direction clear-proxy chroma residual threshold for halo support."""
    halo_chroma_temperature: float = 0.01
    """Sigmoid temperature for halo chroma gate."""
    halo_luma_min: float = 0.02
    """Minimum clear-proxy luma threshold for halo support."""
    halo_luma_temperature: float = 0.01
    """Sigmoid temperature for halo luma gate."""
    medium_support_gradient_tau: float = 0.05
    """Image-gradient temperature for flat-water support."""
    medium_support_variance_tau: float = 0.02
    """Local-variance temperature for flat-water support."""
    medium_support_color_tau: float = 0.08
    """Medium explainability color/luma error temperature."""
    medium_support_luma_weight: float = 0.25
    """Relative luma weight in medium explainability error."""
    medium_support_far_floor: float = 0.50
    """Minimum weak far-depth multiplier for capacity support."""
    medium_support_depth_mid: float = 0.75
    """Normalized detached-depth midpoint for weak far support."""
    medium_support_depth_temperature: float = 0.15
    """Detached-depth sigmoid temperature for weak far support."""
    medium_support_use_flatness: bool = True
    """Use image flatness in medium-explainable support."""
    medium_support_use_medium: bool = True
    """Use detached medium color explainability in support."""
    medium_support_use_far: bool = True
    """Use weak detached far-depth modulation in capacity support."""
    medium_support_connected_enabled: bool = False
    """Keep only medium support connected to the image boundary/top edge."""
    medium_support_connected_threshold: float = 0.25
    """Binary threshold for boundary-connected support flood fill."""
    medium_support_connected_top_only: bool = True
    """If True, seed connected support only from the top image border."""
    medium_support_connected_border: int = 2
    """Border width in pixels used to seed connected support."""
    medium_support_capacity_threshold: float = 0.0
    """Optional lower threshold applied to medium capacity support before capacity/proxy losses."""
    medium_support_capacity_power: float = 1.0
    """Optional exponent applied to thresholded medium capacity support before capacity/proxy losses."""
    medium_support_region_exclusion_enabled: bool = False
    """Use training-only object/boundary region masks to exclude selected support-loss pixels."""
    medium_support_exclude_object: bool = True
    """Exclude object-mask pixels from selected medium supports when region exclusion is enabled."""
    medium_support_exclude_boundary: bool = False
    """Exclude boundary-mask pixels from selected medium supports when region exclusion is enabled."""
    medium_support_region_exclusion_apply_capacity: bool = True
    """Apply region exclusion to budgeted/halo capacity support."""
    medium_support_region_exclusion_apply_chroma: bool = True
    """Apply region exclusion to clear-proxy chroma support."""
    lambda_proxy_clear_luma: float = 0.0
    """Optional support-weighted clear-proxy luma budget loss."""
    proxy_clear_luma_budget: float = 0.03
    """Clear-proxy luma budget for optional luma refinement."""
    proxy_clear_luma_temperature: float = 0.01
    """Softplus temperature for optional clear-proxy luma budget."""
    object_radiance_budget_enabled: bool = False
    """Enable support-weighted luma budget on underwater Gaussian rgb_object."""
    lambda_object_radiance_budget: float = 0.0
    """Weight for object-radiance budget on supported open-water core."""
    object_radiance_budget_value: float = 0.015
    """Allowed rgb_object luma budget on supported open-water core."""
    object_radiance_budget_temperature: float = 0.01
    """Softplus temperature for object-radiance luma budget."""
    object_radiance_budget_start_step: int = 10000
    """First step where object-radiance budget may ramp."""
    object_radiance_budget_ramp_steps: int = 1000
    """Ramp length for object-radiance budget."""
    background_densification_enabled: bool = False
    """Enable background region weighting for densification gradient accumulation."""
    background_densification_weight: float = 1.0
    """Target densification gradient weight for high-precision background-water pixels."""
    uncertain_densification_weight: float = 0.5
    """Densification gradient weight for uncertain pixels."""
    background_densification_start_step: int = 3000
    """First step where background densification weighting can ramp."""
    background_densification_ramp_steps: int = 3000
    """Linear ramp length for background densification weighting."""
    background_densification_diagnostic_only: bool = True
    """If True, log region diagnostics but do not change densification gradients."""
    densification_region_log_path: Optional[str] = None
    """Optional JSONL path for per-region densification diagnostics."""
    opacity_accumulation_diagnostic_enabled: bool = False
    """Retain/log opacity, scale, and sampled accumulation gradient diagnostics."""
    clear_proxy_enabled: bool = False
    """Enable an auxiliary zero-medium black-background clear proxy render."""
    clear_proxy_appearance_only: bool = False
    """Detach clear-proxy geometry and opacity so chroma loss updates only Gaussian appearance."""
    clear_proxy_geometry_gradient_scale: float = 1.0
    """Scale clear-proxy gradients to screen-space geometry/depth/conic tensors; 1 keeps historical behavior."""
    clear_proxy_opacity_gradient_scale: float = 1.0
    """Scale clear-proxy gradients to Gaussian opacity; 1 keeps historical behavior."""
    clear_proxy_color_gradient_scale: float = 1.0
    """Scale clear-proxy gradients to Gaussian SH color; 1 keeps historical behavior."""
    background_gradient_surgery_enabled: bool = False
    """Enable candidate-mask opacity-gradient modulation for open-water contributors."""
    background_candidate_mask_path: Optional[str] = None
    """Path to a train-view contribution candidate .pt mask."""
    background_opacity_decrease_multiplier: float = 1.0
    """Multiplier for positive opacity-logit gradients on background candidates."""
    background_opacity_increase_multiplier: float = 1.0
    """Multiplier for negative opacity-logit gradients on background candidates."""
    background_gradient_surgery_start_step: int = 10001
    """First step where background gradient surgery may modify opacity gradients."""
    background_gradient_surgery_min_view_count: int = 5
    """Minimum candidate train-view support when loading candidate masks with view_count."""
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
    dual_color_enabled: bool = False
    """Enable intrinsic-underwater dual-color Gaussian appearance."""
    clear_sh_luminance_scale: float = 1.0
    """Scale applied to SH luminance residual in the clear/intrinsic branch."""
    clear_sh_chroma_scale: float = 0.0
    """Scale applied to SH chroma residual in the clear/intrinsic branch."""
    lambda_intrinsic_near_anchor: float = 0.0
    """DualColor near-transmission anchor loss weight."""
    lambda_view_residual_mean: float = 0.0
    """DualColor visible residual mean anchor loss weight."""
    lambda_clear_chroma: float = 0.0
    """DualColor visible SH chroma residual loss weight."""
    dual_color_loss_start_step: int = 0
    """First step where DualColor auxiliary losses are active."""
    dual_color_loss_ramp_steps: int = 0
    """Linear ramp length for DualColor auxiliary losses."""
    dual_color_near_transmission_threshold: float = 0.70
    """Transmission midpoint for near-transmission intrinsic anchoring."""
    dual_color_near_transmission_temp: float = 0.10
    """Transmission temperature for near-transmission intrinsic anchoring."""
    dual_color_freeze_geometry: bool = True
    """When DualColor is enabled, keep means/scales/quats/opacities fixed."""
    dual_color_freeze_medium: bool = True
    """When DualColor is enabled, keep medium MLP and direction encoding fixed."""


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
        medium_out_dim = 12 if self._b_inf_requires_head() else 9
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
        self.gmvc_bounded_log_attn_center = nn.Parameter(torch.full((3,), math.log(0.05)))
        self.gmvc_bounded_log_bs_center = nn.Parameter(torch.full((3,), math.log(0.05)))
        self.gmvc_bounded_binf_logit_center = nn.Parameter(torch.zeros(3))
        self._gmvc_bounded_initialized = False
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
        self.register_buffer(
            "gaussian_lineage_ids",
            torch.arange(num_points, dtype=torch.long, device=means.device),
            persistent=True,
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
        self._backscatter_mask_cache: Dict[Tuple[str, int, str], Optional[torch.Tensor]] = {}
        self.current_densification_region_weight = None
        self.current_densification_region_samples = None
        self.current_densification_accumulation_map = None
        self.last_densification_region_stats = None
        self.xys_grad_abs_proxy = None
        self._background_candidate_mask = None
        self._background_candidate_path = None
        self._background_candidate_num_points = 0
        self._background_gradient_hook_handle = None
        self._background_gradient_hook_param_id = None
        self._background_gradient_surgery_last_log_step = -1
        self._warned_background_clear_gaussian_dead_grad = False
        self._gmvc_training_bank = None
        self._gmvc_training_bank_path = None
        self._gmvc_warned_missing_bank = False
        self._gmvc_online_state: Dict[str, torch.Tensor] = {}
        self._gmvc_latest_learning_rates: Dict[str, float] = {}
        self._gmvc_latest_grad_scaler_scale = 0.0
        self._gmvc_medium_hold_reference_step = -1
        self._gmvc_medium_hold_reference: Optional[Dict[str, torch.Tensor]] = None
        self._gmvc_gaussian_hold_reference: Optional[Dict[str, torch.Tensor]] = None

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
        if "gaussian_lineage_ids" not in dict:
            dict["gaussian_lineage_ids"] = torch.arange(newp, dtype=torch.long)
        if tuple(self.gaussian_lineage_ids.shape) != (newp,):
            self.gaussian_lineage_ids.data = torch.zeros(newp, device=self.device, dtype=torch.long)
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            if tuple(old_shape) != tuple(new_shape):
                # Keep the Parameter object identity stable: Nerfstudio builds optimizers
                # before loading checkpoints, so replacing Parameter objects here leaves
                # optimizer param groups attached to stale tensors.
                param.data = torch.zeros(new_shape, device=self.device, dtype=param.dtype)
        for key in [
            "gmvc_bounded_log_attn_center",
            "gmvc_bounded_log_bs_center",
            "gmvc_bounded_binf_logit_center",
        ]:
            if key not in dict:
                dict[key] = getattr(self, key).detach().clone()
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
        if getattr(self.config, "dual_color_enabled", False) and getattr(self.config, "dual_color_freeze_geometry", True):
            return
        # to save some training time, we no longer need to update those stats post refinement
        # if self.step >= self.config.stop_split_at:
        #     return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            if self.config.abs_grad_densification:
                assert self.xys_grad_abs is not None
                xys_grad_abs_for_stats = self.xys_grad_abs.detach()
                prepass_grad_abs = getattr(self, "_capacity_conflict_xys_grad_abs_prepass", None)
                if prepass_grad_abs is not None:
                    if tuple(prepass_grad_abs.shape) == tuple(xys_grad_abs_for_stats.shape):
                        xys_grad_abs_for_stats = (xys_grad_abs_for_stats - prepass_grad_abs).clamp_min(0.0)
                    self._capacity_conflict_xys_grad_abs_prepass = None
                grads = xys_grad_abs_for_stats.norm(dim=-1)
            else:
                assert self.xys.grad is not None
                grads = self.xys.grad.detach().norm(dim=-1)
            weighted_grads = grads
            if (
                getattr(self.config, "background_densification_enabled", False)
                and not getattr(self.config, "background_densification_diagnostic_only", True)
                and self.current_densification_region_weight is not None
                and self.current_densification_region_weight.shape[0] == grads.shape[0]
            ):
                weighted_grads = grads * self.current_densification_region_weight.to(
                    device=grads.device,
                    dtype=grads.dtype,
                )
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            stats_shape_mismatch = (
                self.xys_grad_norm is not None
                and (
                    self.xys_grad_norm.shape[0] != weighted_grads.shape[0]
                    or self.vis_counts is None
                    or self.vis_counts.shape[0] != weighted_grads.shape[0]
                    or self.depths_accum is None
                    or self.depths_accum.shape[0] != weighted_grads.shape[0]
                )
            )
            if self.xys_grad_norm is None or stats_shape_mismatch:
                self.xys_grad_norm = weighted_grads
                self.depths_accum = self.depths
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = weighted_grads[visible_mask] + self.xys_grad_norm[visible_mask]
                self.depths_accum[visible_mask] = self.depths[visible_mask] + self.depths_accum[visible_mask]
            self._record_densification_region_diagnostics(
                visible_mask=visible_mask,
                raw_grads=grads,
                weighted_grads=weighted_grads,
            )

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None or self.max_2Dsize.shape[0] != self.radii.shape[0]:
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
        if getattr(self.config, "dual_color_enabled", False) and getattr(self.config, "dual_color_freeze_geometry", True):
            return
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
                n_before_densification = self.num_points
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])

                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()

                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads

                nsamps = self.config.n_split_samples
                split_count = int(splits.sum().item())
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                duplicate_count = int(dups.sum().item())

                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )
                self._sync_gaussian_lineage_ids_for_densification(splits, dups, nsamps)
                self._sync_background_candidate_mask_for_densification(splits, dups, nsamps)

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
                CONSOLE.log(
                    "Densification step="
                    f"{self.step} split={split_count} duplicate={duplicate_count} "
                    f"split_children={int(nsamps) * split_count} duplicate_children={duplicate_count} "
                    f"total_before={n_before_densification} total_after_append={self.num_points}"
                )

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
        self._sync_gaussian_lineage_ids_for_cull(culls)
        self._sync_background_candidate_mask_for_cull(culls)
        self._sync_gmvc_hold_gaussian_reference_for_cull(culls)

        CONSOLE.log(
            f"Culled step={self.step} {n_bef - self.num_points} gaussians "
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
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers, training_callback_attributes.grad_scaler],
            )
        )
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

    def step_cb(self, optimizers: Optional[Optimizers] = None, grad_scaler=None, step: int = 0):
        self.step = step
        if optimizers is not None:
            self._gmvc_latest_learning_rates = {
                str(name): float(optimizer.param_groups[0].get("lr", 0.0))
                for name, optimizer in optimizers.optimizers.items()
                if getattr(optimizer, "param_groups", None)
            }
        if grad_scaler is not None:
            try:
                self._gmvc_latest_grad_scaler_scale = float(grad_scaler.get_scale())
            except Exception:
                self._gmvc_latest_grad_scaler_scale = 0.0
        self._update_gmvc_medium_hold_freeze()
        self._ensure_background_gradient_surgery_hook()

    def _gmvc_medium_hold_active(self, step: Optional[int] = None) -> bool:
        if not bool(getattr(self.config, "gmvc_medium_hold_enabled", False)):
            return False
        current_step = int(self.step if step is None else step)
        start = int(getattr(self.config, "gmvc_medium_hold_start_step", 13001))
        stop = int(getattr(self.config, "gmvc_medium_hold_stop_step", 15000))
        return start <= current_step <= stop

    def _named_medium_parameters_for_hold(self) -> List[Tuple[str, Parameter]]:
        params: List[Tuple[str, Parameter]] = []
        params.extend((f"medium_mlp.{name}", param) for name, param in self.medium_mlp.named_parameters())
        params.extend(
            (f"direction_encoding.{name}", param) for name, param in self.direction_encoding.named_parameters()
        )
        params.extend(
            [
                ("gmvc_bounded_log_attn_center", self.gmvc_bounded_log_attn_center),
                ("gmvc_bounded_log_bs_center", self.gmvc_bounded_log_bs_center),
                ("gmvc_bounded_binf_logit_center", self.gmvc_bounded_binf_logit_center),
            ]
        )
        return params

    @staticmethod
    def _snapshot_parameters_cpu(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: param.detach().float().cpu().clone() for name, param in params.items()}

    def _ensure_gmvc_medium_hold_references(self) -> None:
        if self._gmvc_medium_hold_reference is not None:
            return
        self._gmvc_medium_hold_reference_step = int(self.step)
        self._gmvc_medium_hold_reference = self._snapshot_parameters_cpu(
            {name: param for name, param in self._named_medium_parameters_for_hold()}
        )
        self._gmvc_gaussian_hold_reference = self._snapshot_parameters_cpu(
            {
                "features_dc": self.features_dc,
                "features_rest": self.features_rest,
                "opacities": self.opacities,
                "means": self.means,
                "scales": self.scales,
                "quats": self.quats,
            }
        )

    def _update_gmvc_medium_hold_freeze(self) -> None:
        dual_color_freeze_medium = bool(getattr(self.config, "dual_color_enabled", False)) and bool(
            getattr(self.config, "dual_color_freeze_medium", True)
        )
        hold_active = self._gmvc_medium_hold_active()
        medium_trainable = not (dual_color_freeze_medium or hold_active)
        bounded_param_ids = {
            id(self.gmvc_bounded_log_attn_center),
            id(self.gmvc_bounded_log_bs_center),
            id(self.gmvc_bounded_binf_logit_center),
        }
        for _, param in self._named_medium_parameters_for_hold():
            is_bounded_center = id(param) in bounded_param_ids
            trainable = medium_trainable
            if is_bounded_center:
                trainable = medium_trainable and bool(getattr(self.config, "gmvc_bounded_medium_enabled", False))
            param.requires_grad_(trainable)
            if not trainable:
                param.grad = None
        if hold_active:
            self._ensure_gmvc_medium_hold_references()

    @staticmethod
    def _delta_stats_from_reference(
        params: Dict[str, torch.Tensor],
        reference: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        if reference is None:
            return {"mean_abs": 0.0, "max_abs": 0.0, "l2": 0.0, "count": 0.0, "shape_mismatch": 0.0}
        total_abs = 0.0
        total_sq = 0.0
        total_count = 0
        max_abs = 0.0
        shape_mismatch = False
        for name, param in params.items():
            ref = reference.get(name)
            if ref is None:
                shape_mismatch = True
                continue
            current = param.detach().float().cpu()
            if tuple(current.shape) != tuple(ref.shape):
                shape_mismatch = True
                continue
            delta = current - ref
            abs_delta = delta.abs()
            total_abs += float(abs_delta.sum().item())
            total_sq += float(delta.square().sum().item())
            total_count += int(delta.numel())
            if delta.numel() > 0:
                max_abs = max(max_abs, float(abs_delta.max().item()))
        mean_abs = total_abs / max(total_count, 1)
        return {
            "mean_abs": mean_abs,
            "max_abs": max_abs,
            "l2": math.sqrt(total_sq),
            "count": float(total_count),
            "shape_mismatch": float(shape_mismatch),
        }

    def _gmvc_hold_delta_stats(self) -> Dict[str, float]:
        medium_stats = self._delta_stats_from_reference(
            {name: param for name, param in self._named_medium_parameters_for_hold()},
            self._gmvc_medium_hold_reference,
        )
        gaussian_ref = self._gmvc_gaussian_hold_reference
        dc_stats = self._delta_stats_from_reference({"features_dc": self.features_dc}, gaussian_ref)
        rest_stats = self._delta_stats_from_reference({"features_rest": self.features_rest}, gaussian_ref)
        opacity_stats = self._delta_stats_from_reference({"opacities": self.opacities}, gaussian_ref)
        geometry_stats = self._delta_stats_from_reference(
            {"means": self.means, "scales": self.scales, "quats": self.quats},
            gaussian_ref,
        )
        return {
            "gmvc_medium_hold_reference_step": float(self._gmvc_medium_hold_reference_step),
            "gmvc_medium_param_delta_mean_abs": medium_stats["mean_abs"],
            "gmvc_medium_param_delta_max_abs": medium_stats["max_abs"],
            "gmvc_medium_param_delta_l2": medium_stats["l2"],
            "gmvc_medium_param_delta_count": medium_stats["count"],
            "gmvc_medium_param_delta_shape_mismatch": medium_stats["shape_mismatch"],
            "gmvc_mhold_features_dc_delta_l2": dc_stats["l2"],
            "gmvc_mhold_features_dc_delta_mean_abs": dc_stats["mean_abs"],
            "gmvc_mhold_features_rest_delta_l2": rest_stats["l2"],
            "gmvc_mhold_features_rest_delta_mean_abs": rest_stats["mean_abs"],
            "gmvc_mhold_features_rest_to_dc_delta_ratio": rest_stats["l2"] / (dc_stats["l2"] + 1e-12),
            "gmvc_mhold_opacity_delta_l2": opacity_stats["l2"],
            "gmvc_mhold_opacity_delta_mean_abs": opacity_stats["mean_abs"],
            "gmvc_mhold_geometry_delta_l2": geometry_stats["l2"],
            "gmvc_mhold_geometry_delta_mean_abs": geometry_stats["mean_abs"],
            "gmvc_mhold_gaussian_delta_shape_mismatch": max(
                dc_stats["shape_mismatch"],
                rest_stats["shape_mismatch"],
                opacity_stats["shape_mismatch"],
                geometry_stats["shape_mismatch"],
            ),
        }

    def _sync_gmvc_hold_gaussian_reference_for_cull(self, culls: torch.Tensor) -> None:
        if self._gmvc_gaussian_hold_reference is None:
            return
        keep = (~culls.detach().cpu()).bool()
        for name, ref in list(self._gmvc_gaussian_hold_reference.items()):
            if ref.ndim > 0 and int(ref.shape[0]) == int(keep.shape[0]):
                self._gmvc_gaussian_hold_reference[name] = ref[keep].clone()

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        names = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        if self._gmvc_requested() and getattr(self.config, "gmvc_freeze_geometry", False):
            for name in ["means", "scales", "quats", "opacities"]:
                self.gauss_params[name].requires_grad_(False)
            self.gauss_params["features_dc"].requires_grad_(bool(getattr(self.config, "gmvc_train_features_dc", False)))
            self.gauss_params["features_rest"].requires_grad_(bool(getattr(self.config, "gmvc_train_features_rest", False)))
        elif getattr(self.config, "dual_color_enabled", False) and getattr(self.config, "dual_color_freeze_geometry", True):
            for name in ["means", "scales", "quats", "opacities"]:
                self.gauss_params[name].requires_grad_(False)
            for name in ["features_dc", "features_rest"]:
                self.gauss_params[name].requires_grad_(True)
        else:
            for name in names:
                self.gauss_params[name].requires_grad_(True)
        return {name: [self.gauss_params[name]] for name in names}

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        freeze_medium = getattr(self.config, "dual_color_enabled", False) and getattr(
            self.config, "dual_color_freeze_medium", True
        )
        for param in self.medium_mlp.parameters():
            param.requires_grad_(not freeze_medium)
        for param in self.direction_encoding.parameters():
            param.requires_grad_(not freeze_medium)
        gps["medium_mlp"] = list(self.medium_mlp.parameters())
        gps["direction_encoding"] = list(self.direction_encoding.parameters())
        bounded_trainable = bool(getattr(self.config, "gmvc_bounded_medium_enabled", False)) and not freeze_medium
        for param in [
            self.gmvc_bounded_log_attn_center,
            self.gmvc_bounded_log_bs_center,
            self.gmvc_bounded_binf_logit_center,
        ]:
            param.requires_grad_(bounded_trainable)
        gps["gmvc_bounded_medium"] = [
            self.gmvc_bounded_log_attn_center,
            self.gmvc_bounded_log_bs_center,
            self.gmvc_bounded_binf_logit_center,
        ]
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
        if mode == "implicit" and getattr(self.config, "infinite_water_enabled", False):
            return "independent"
        return mode

    def _b_inf_requires_head(self) -> bool:
        return self._effective_b_inf_mode() in {"bounded_residual", "independent"}

    def _gmvc_requested(self) -> bool:
        return bool(
            getattr(self.config, "gmvc_enabled", False)
            or getattr(self.config, "gmvc_diagnostic_only", False)
            or getattr(self.config, "gmvc_v2_enabled", False)
            or getattr(self.config, "gmvc_v3_enabled", False)
            or float(getattr(self.config, "lambda_gmvc_object", 0.0)) > 0.0
        )

    def _gmvc_intrinsic_active(self) -> bool:
        if not (self.training and self._gmvc_requested()):
            return False
        weight = max(
            float(getattr(self.config, "lambda_gmvc_intrinsic", 0.0)),
            float(getattr(self.config, "lambda_gmvc_object", 0.0)),
        )
        if weight <= 0.0:
            return False
        start = int(getattr(self.config, "gmvc_start_step", 10000))
        stop = int(getattr(self.config, "gmvc_stop_step", 15000))
        return start <= int(self.step) < stop

    def _gmvc_intrinsic_uses_proxy(self) -> bool:
        return bool(
            str(getattr(self.config, "gmvc_intrinsic_source", "J_proxy_raw")) == "J_proxy_raw"
            or (
                float(getattr(self.config, "lambda_gmvc_object", 0.0)) > 0.0
                and str(getattr(self.config, "gmvc_v3_object_source", "J_proxy_raw")) == "J_proxy_raw"
            )
        )

    def _gmvc_v3_object_phase_active(self) -> bool:
        if not (self.training and bool(getattr(self.config, "gmvc_v3_enabled", False))):
            return False
        start = int(getattr(self.config, "gmvc_start_step", 10000))
        stop = int(getattr(self.config, "gmvc_stop_step", 15000))
        if int(self.step) < start or int(self.step) >= stop:
            return False
        medium_steps = max(int(getattr(self.config, "gmvc_v3_medium_steps", 4)), 0)
        object_steps = max(int(getattr(self.config, "gmvc_v3_object_steps", 1)), 0)
        cycle = medium_steps + object_steps
        if cycle <= 0 or object_steps <= 0:
            return False
        return ((int(self.step) - start) % cycle) >= medium_steps

    @staticmethod
    def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
        if scale <= 0.0:
            return value.detach()
        if scale >= 1.0:
            return value
        return value.detach() + float(scale) * (value - value.detach())

    def _maybe_scale_gmvc_v3_object_medium_grad(self, medium: MediumFieldOutput) -> MediumFieldOutput:
        if not self._gmvc_v3_object_phase_active():
            return medium
        if bool(getattr(self.config, "gmvc_v3_freeze_medium_on_object_phase", False)):
            scale = 0.0
        else:
            scale = float(getattr(self.config, "gmvc_v3_object_phase_medium_grad_scale", 1.0))
        scale = max(0.0, min(scale, 1.0))
        if scale >= 1.0:
            return medium
        return MediumFieldOutput(
            rgb=self._scale_gradient(medium.rgb, scale),
            bs=self._scale_gradient(medium.bs, scale),
            attn=self._scale_gradient(medium.attn, scale),
            directions=medium.directions,
            b_inf=self._scale_gradient(medium.b_inf, scale) if medium.b_inf is not None else None,
            b_inf_residual=self._scale_gradient(medium.b_inf_residual, scale)
            if medium.b_inf_residual is not None
            else None,
        )

    @staticmethod
    def _grad_norm(grads: Tuple[Optional[torch.Tensor], ...]) -> float:
        total = 0.0
        for grad in grads:
            if grad is not None:
                total += float(grad.detach().float().square().sum().item())
        return math.sqrt(total)

    def _maybe_log_gmvc_grad_stats(
        self,
        loss_dict: Dict[str, torch.Tensor],
        metrics_dict: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        path = getattr(self.config, "gmvc_grad_log_path", None)
        if not (self.training and path):
            return
        every = max(int(getattr(self.config, "gmvc_grad_log_every", 100)), 1)
        force_steps_raw = getattr(self.config, "gmvc_grad_log_force_steps", None)
        force_steps = set()
        if force_steps_raw:
            for item in str(force_steps_raw).split(","):
                item = item.strip()
                if item:
                    try:
                        force_steps.add(int(item))
                    except ValueError:
                        pass
        if int(self.step) % every != 0 and int(self.step) not in force_steps:
            return
        main_loss = loss_dict.get("main_loss")
        intrinsic_loss = loss_dict.get("gmvc_intrinsic_loss")
        residual_loss = loss_dict.get("gmvc_residual_budget_loss")
        closure_loss = loss_dict.get("gmvc_fixed_closure_loss")
        profile_loss = loss_dict.get("gmvc_profile_loss")
        symmetric_closure_loss = loss_dict.get("gmvc_symmetric_closure_loss")
        object_loss = loss_dict.get("gmvc_object_loss")
        gmvc_losses = [intrinsic_loss, residual_loss, closure_loss, profile_loss, symmetric_closure_loss, object_loss]
        mhold_active = self._gmvc_medium_hold_active()
        if not any(loss is not None and bool(getattr(loss, "requires_grad", False)) for loss in gmvc_losses) and not (
            mhold_active and main_loss is not None and bool(getattr(main_loss, "requires_grad", False))
        ):
            return

        def _safe_grad(loss: Optional[torch.Tensor], params: List[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
            if loss is None or not bool(getattr(loss, "requires_grad", False)):
                return tuple(None for _ in params)
            active_params = [param for param in params if bool(getattr(param, "requires_grad", False))]
            if not active_params:
                return tuple(None for _ in params)
            try:
                active_grads = torch.autograd.grad(
                    loss,
                    active_params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_iter = iter(active_grads)
                return tuple(next(grad_iter) if bool(getattr(param, "requires_grad", False)) else None for param in params)
            except RuntimeError as exc:
                CONSOLE.log(f"[yellow]GMVC grad diagnostic skipped at step {self.step}: {exc}[/yellow]")
                return tuple(None for _ in params)

        xys_grad_abs_before = self.xys_grad_abs.detach().clone() if self.xys_grad_abs is not None else None
        xys_grad_abs_proxy_before = (
            self.xys_grad_abs_proxy.detach().clone() if self.xys_grad_abs_proxy is not None else None
        )

        dc_params = [self.gauss_params["features_dc"]]
        rest_params = [self.gauss_params["features_rest"]]
        geom_params = [
            self.gauss_params["means"],
            self.gauss_params["scales"],
            self.gauss_params["quats"],
        ]
        opacity_params = [self.gauss_params["opacities"]]
        medium_params = (
            list(self.medium_mlp.parameters())
            + list(self.direction_encoding.parameters())
            + [
                self.gmvc_bounded_log_attn_center,
                self.gmvc_bounded_log_bs_center,
                self.gmvc_bounded_binf_logit_center,
            ]
        )

        intrinsic_dc = _safe_grad(intrinsic_loss, dc_params)
        rgb_dc = _safe_grad(main_loss, dc_params)
        rgb_rest = _safe_grad(main_loss, rest_params)
        intrinsic_geom = _safe_grad(intrinsic_loss, geom_params)
        rgb_geom = _safe_grad(main_loss, geom_params)
        intrinsic_opacity = _safe_grad(intrinsic_loss, opacity_params)
        rgb_opacity = _safe_grad(main_loss, opacity_params)
        intrinsic_medium = _safe_grad(intrinsic_loss, medium_params)
        object_dc = _safe_grad(object_loss, dc_params)
        object_geom = _safe_grad(object_loss, geom_params)
        object_opacity = _safe_grad(object_loss, opacity_params)
        object_medium = _safe_grad(object_loss, medium_params)
        residual_medium = _safe_grad(residual_loss, medium_params)
        closure_medium = _safe_grad(closure_loss, medium_params)
        profile_medium = _safe_grad(profile_loss, medium_params)
        symmetric_closure_medium = _safe_grad(symmetric_closure_loss, medium_params)
        rgb_medium = _safe_grad(main_loss, medium_params)

        if xys_grad_abs_before is not None and self.xys_grad_abs is not None:
            self.xys_grad_abs = xys_grad_abs_before
        if xys_grad_abs_proxy_before is not None and self.xys_grad_abs_proxy is not None:
            self.xys_grad_abs_proxy = xys_grad_abs_proxy_before

        intrinsic_dc_norm = self._grad_norm(intrinsic_dc)
        object_dc_norm = self._grad_norm(object_dc)
        rgb_dc_norm = self._grad_norm(rgb_dc)
        rgb_rest_norm = self._grad_norm(rgb_rest)
        rgb_geom_norm = self._grad_norm(rgb_geom)
        rgb_opacity_norm = self._grad_norm(rgb_opacity)
        intrinsic_medium_norm = self._grad_norm(intrinsic_medium)
        object_medium_norm = self._grad_norm(object_medium)
        residual_medium_norm = self._grad_norm(residual_medium)
        closure_medium_norm = self._grad_norm(closure_medium)
        profile_medium_norm = self._grad_norm(profile_medium)
        symmetric_closure_medium_norm = self._grad_norm(symmetric_closure_medium)
        rgb_medium_norm = self._grad_norm(rgb_medium)
        hold_delta_stats = self._gmvc_hold_delta_stats() if mhold_active else {}

        def _metric_float(name: str, default: float = 0.0) -> float:
            if metrics_dict is None or name not in metrics_dict:
                return float(default)
            value = metrics_dict[name]
            if not torch.is_tensor(value):
                return float(value)
            return float(value.detach().float().reshape(-1)[0].item())

        row = {
            "step": int(self.step),
            "global_step": int(self.step),
            "gmvc_intrinsic_raw": float(
                metrics_dict.get(
                    "gmvc_intrinsic_raw",
                    intrinsic_loss.detach()
                    if intrinsic_loss is not None
                    else torch.zeros((), device=self.device),
                )
                .detach()
                .float()
                .item()
                if metrics_dict is not None
                else (
                    intrinsic_loss.detach().float().item()
                    if intrinsic_loss is not None
                    else 0.0
                )
            ),
            "gmvc_residual_budget_raw": float(
                metrics_dict.get(
                    "gmvc_residual_budget_raw",
                    residual_loss.detach()
                    if residual_loss is not None
                    else torch.zeros((), device=self.device),
                )
                .detach()
                .float()
                .item()
                if metrics_dict is not None
                else (
                    residual_loss.detach().float().item()
                    if residual_loss is not None
                    else 0.0
                )
            ),
            "gmvc_fixed_closure_raw": float(
                metrics_dict.get(
                    "gmvc_fixed_closure_raw",
                    closure_loss.detach()
                    if closure_loss is not None
                    else torch.zeros((), device=self.device),
                )
                .detach()
                .float()
                .item()
                if metrics_dict is not None
                else (
                    closure_loss.detach().float().item()
                    if closure_loss is not None
                    else 0.0
                )
            ),
            "gmvc_intrinsic_weighted": float(
                intrinsic_loss.detach().float().item() if intrinsic_loss is not None else 0.0
            ),
            "gmvc_residual_budget_weighted": float(
                residual_loss.detach().float().item() if residual_loss is not None else 0.0
            ),
            "gmvc_fixed_closure_weighted": float(
                closure_loss.detach().float().item() if closure_loss is not None else 0.0
            ),
            "gmvc_profile_raw": float(
                metrics_dict.get(
                    "gmvc_profile_raw",
                    profile_loss.detach()
                    if profile_loss is not None
                    else torch.zeros((), device=self.device),
                )
                .detach()
                .float()
                .item()
                if metrics_dict is not None
                else (
                    profile_loss.detach().float().item()
                    if profile_loss is not None
                    else 0.0
                )
            ),
            "gmvc_symmetric_closure_raw": float(
                metrics_dict.get(
                    "gmvc_symmetric_closure_raw",
                    symmetric_closure_loss.detach()
                    if symmetric_closure_loss is not None
                    else torch.zeros((), device=self.device),
                )
                .detach()
                .float()
                .item()
                if metrics_dict is not None
                else (
                    symmetric_closure_loss.detach().float().item()
                    if symmetric_closure_loss is not None
                    else 0.0
                )
            ),
            "gmvc_object_raw": float(
                metrics_dict.get(
                    "gmvc_object_raw",
                    object_loss.detach()
                    if object_loss is not None
                    else torch.zeros((), device=self.device),
                )
                .detach()
                .float()
                .item()
                if metrics_dict is not None
                else (
                    object_loss.detach().float().item()
                    if object_loss is not None
                    else 0.0
                )
            ),
            "gmvc_profile_weighted": float(
                profile_loss.detach().float().item() if profile_loss is not None else 0.0
            ),
            "gmvc_symmetric_closure_weighted": float(
                symmetric_closure_loss.detach().float().item() if symmetric_closure_loss is not None else 0.0
            ),
            "gmvc_object_weighted": float(
                object_loss.detach().float().item() if object_loss is not None else 0.0
            ),
            "gmvc_intrinsic_grad_norm_dc": intrinsic_dc_norm,
            "gmvc_object_grad_norm_dc": object_dc_norm,
            "rgb_grad_norm_dc": rgb_dc_norm,
            "rgb_grad_norm_features_rest": rgb_rest_norm,
            "rgb_grad_norm_geometry": rgb_geom_norm,
            "rgb_grad_norm_opacity": rgb_opacity_norm,
            "intrinsic_to_rgb_dc_grad_ratio": intrinsic_dc_norm / (rgb_dc_norm + 1e-12),
            "object_to_rgb_dc_grad_ratio": object_dc_norm / (rgb_dc_norm + 1e-12),
            "gmvc_intrinsic_grad_norm_geometry": self._grad_norm(intrinsic_geom),
            "gmvc_object_grad_norm_geometry": self._grad_norm(object_geom),
            "gmvc_intrinsic_grad_norm_opacity": self._grad_norm(intrinsic_opacity),
            "gmvc_object_grad_norm_opacity": self._grad_norm(object_opacity),
            "gmvc_intrinsic_grad_norm_medium": intrinsic_medium_norm,
            "gmvc_object_grad_norm_medium": object_medium_norm,
            "gmvc_residual_grad_norm_medium": residual_medium_norm,
            "gmvc_closure_grad_norm_medium": closure_medium_norm,
            "gmvc_profile_grad_norm_medium": profile_medium_norm,
            "gmvc_symmetric_closure_grad_norm_medium": symmetric_closure_medium_norm,
            "rgb_grad_norm_medium": rgb_medium_norm,
            "gmvc_intrinsic_to_rgb_medium_grad_ratio": intrinsic_medium_norm / (rgb_medium_norm + 1e-12),
            "gmvc_object_to_rgb_medium_grad_ratio": object_medium_norm / (rgb_medium_norm + 1e-12),
            "gmvc_residual_to_rgb_medium_grad_ratio": residual_medium_norm / (rgb_medium_norm + 1e-12),
            "gmvc_closure_to_rgb_medium_grad_ratio": closure_medium_norm / (rgb_medium_norm + 1e-12),
            "gmvc_profile_to_rgb_medium_grad_ratio": profile_medium_norm / (rgb_medium_norm + 1e-12),
            "gmvc_symmetric_closure_to_rgb_medium_grad_ratio": symmetric_closure_medium_norm
            / (rgb_medium_norm + 1e-12),
            "gmvc_intrinsic_source": str(getattr(self.config, "gmvc_intrinsic_source", "J_proxy_raw")),
            "gmvc_intrinsic_use_dc_proxy": bool(getattr(self.config, "gmvc_intrinsic_use_dc_proxy", True)),
            "gmvc_v3_profile_schedule": str(getattr(self.config, "gmvc_v3_profile_schedule", "constant")),
            "gmvc_v3_profile_decay_start_step": int(
                getattr(self.config, "gmvc_v3_profile_decay_start_step", 13000)
            ),
            "gmvc_v3_profile_decay_end_step": int(getattr(self.config, "gmvc_v3_profile_decay_end_step", 14000)),
            "gmvc_v3_profile_decay_final_scale": float(
                getattr(self.config, "gmvc_v3_profile_decay_final_scale", 0.0)
            ),
            "gmvc_medium_hold_enabled": bool(getattr(self.config, "gmvc_medium_hold_enabled", False)),
            "gmvc_medium_hold_active": bool(mhold_active),
            "gmvc_medium_hold_start_step": int(getattr(self.config, "gmvc_medium_hold_start_step", 13001)),
            "gmvc_medium_hold_stop_step": int(getattr(self.config, "gmvc_medium_hold_stop_step", 15000)),
            "learning_rate": dict(self._gmvc_latest_learning_rates),
            "learning_rate_medium_mlp": float(self._gmvc_latest_learning_rates.get("medium_mlp", 0.0)),
            "learning_rate_direction_encoding": float(
                self._gmvc_latest_learning_rates.get("direction_encoding", 0.0)
            ),
            "learning_rate_features_dc": float(self._gmvc_latest_learning_rates.get("features_dc", 0.0)),
            "grad_scaler_scale": float(self._gmvc_latest_grad_scaler_scale),
        }
        for key in [
            "gmvc_global_step",
            "gmvc_phase",
            "gmvc_profile_lambda_configured",
            "gmvc_profile_lambda_scheduled",
            "gmvc_profile_lambda_effective",
            "gmvc_profile_schedule_scale",
            "gmvc_object_ramp_factor",
            "gmvc_object_phase_medium_grad_scale",
            "gmvc_profile_bank_transmission_span_p50",
            "gmvc_profile_bank_backscatter_span_p50",
            "gmvc_profile_view_angle_span_p50",
            "gmvc_profile_observability_weight_enabled",
            "gmvc_profile_observability_weight_mean",
            "gmvc_profile_observability_weight_p10",
            "gmvc_profile_observability_weight_p50",
            "gmvc_profile_observability_weight_p90",
            "gmvc_v3_object_phase",
            "gmvc_lambda_profile",
            "gmvc_lambda_object",
            "gmvc_v2_sampled_tracks",
            "gmvc_v2_sampled_observations",
            "gmvc_v2_valid_tracks",
            "gmvc_v2_valid_observation_fraction",
            "gmvc_object_valid_tracks",
            "gmvc_object_valid_observation_fraction",
            "gmvc_object_weight_sum",
            "gmvc_profile_j_star_mean",
            "gmvc_profile_j_star_p05",
            "gmvc_profile_j_star_p95",
            "gmvc_profile_j_star_drift_l1_mean",
            "gmvc_profile_j_star_drift_l1_p95",
            "gmvc_profile_j_star_drift_count",
            "gmvc_medium_attn_delta_l1_mean",
            "gmvc_medium_attn_delta_l1_p95",
            "gmvc_medium_bs_delta_l1_mean",
            "gmvc_medium_bs_delta_l1_p95",
            "gmvc_b_inf_delta_l1_mean",
            "gmvc_b_inf_delta_l1_p95",
            "gmvc_transmission_delta_l1_mean",
            "gmvc_transmission_delta_l1_p95",
            "gmvc_medium_delta_count",
        ]:
            row[key] = _metric_float(key)
        row.update(hold_delta_stats)
        row["gmvc_phase"] = "object" if row.get("gmvc_v3_object_phase", 0.0) >= 0.5 else "medium"
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")

    def _load_gmvc_training_bank(self):
        path = getattr(self.config, "gmvc_track_bank_path", None)
        if not path:
            if self._gmvc_requested() and not self._gmvc_warned_missing_bank:
                CONSOLE.log("[yellow]GMVC requested but gmvc_track_bank_path is empty; skipping GMVC terms.[/yellow]")
                self._gmvc_warned_missing_bank = True
            return None
        if self._gmvc_training_bank is not None and self._gmvc_training_bank_path == path:
            return self._gmvc_training_bank
        self._gmvc_training_bank = load_gmvc_training_bank(path)
        self._gmvc_training_bank_path = path
        self._gmvc_online_state = {}
        metadata = self._gmvc_training_bank.get("metadata", {})
        counts = metadata.get("per_camera_counts", {})
        CONSOLE.log(
            f"[cyan]Loaded GMVC track bank[/cyan] path={path} "
            f"cameras={len(counts)} observations={sum(int(v) for v in counts.values())}"
        )
        return self._gmvc_training_bank

    def _backscatter_ramp_weight(self, weight: float) -> float:
        if weight <= 0.0:
            return 0.0
        start = int(getattr(self.config, "backscatter_loss_start_step", 0))
        ramp = int(getattr(self.config, "backscatter_loss_ramp_steps", 0))
        if self.step < start:
            return 0.0
        if ramp <= 0:
            return float(weight)
        return float(weight) * min((self.step - start) / max(float(ramp), 1.0), 1.0)

    def _ramped_weight(self, weight: float, start: int, ramp: int) -> float:
        if weight <= 0.0:
            return 0.0
        if self.step < start:
            return 0.0
        if ramp <= 0:
            return float(weight)
        return float(weight) * min((self.step - start) / max(float(ramp), 1.0), 1.0)

    def _background_render_ramp_weight(self, weight: float) -> float:
        return self._ramped_weight(
            weight,
            int(getattr(self.config, "background_render_loss_start_step", 0)),
            int(getattr(self.config, "background_render_loss_ramp_steps", 0)),
        )

    def _background_clear_ramp_weight(self, weight: float) -> float:
        return self._ramped_weight(
            weight,
            int(getattr(self.config, "background_clear_loss_start_step", 3000)),
            int(getattr(self.config, "background_clear_loss_ramp_steps", 3000)),
        )

    def _background_densification_effective_weight(self) -> float:
        target = float(getattr(self.config, "background_densification_weight", 1.0))
        start = int(getattr(self.config, "background_densification_start_step", 3000))
        ramp = int(getattr(self.config, "background_densification_ramp_steps", 3000))
        if self.step < start:
            return 1.0
        if ramp <= 0:
            return target
        progress = min((self.step - start) / max(float(ramp), 1.0), 1.0)
        return 1.0 + (target - 1.0) * progress

    def _sync_background_candidate_mask_for_densification(
        self,
        splits: torch.Tensor,
        dups: torch.Tensor,
        nsamps: int,
    ) -> None:
        """Keep candidate flags aligned when densification appends children."""
        if self._background_candidate_mask is None:
            return
        mask = self._background_candidate_mask.reshape(-1).to(device=self.device)
        splits = splits.reshape(-1).to(device=self.device)
        dups = dups.reshape(-1).to(device=self.device)
        if mask.numel() != splits.numel() or mask.numel() != dups.numel():
            raise ValueError(
                "background candidate mask cannot be synchronized with densification: "
                f"mask={mask.numel()} splits={splits.numel()} dups={dups.numel()}"
            )
        split_children = mask[splits].repeat(int(nsamps))
        dup_children = mask[dups]
        self._background_candidate_mask = torch.cat([mask, split_children, dup_children], dim=0).bool()
        self._background_candidate_num_points = int(self._background_candidate_mask.numel())

    def _sync_gaussian_lineage_ids_for_densification(
        self,
        splits: torch.Tensor,
        dups: torch.Tensor,
        nsamps: int,
    ) -> None:
        ids = self.gaussian_lineage_ids.reshape(-1).to(device=self.device)
        splits = splits.reshape(-1).to(device=self.device)
        dups = dups.reshape(-1).to(device=self.device)
        if ids.numel() != splits.numel() or ids.numel() != dups.numel():
            raise ValueError(
                "gaussian lineage ids cannot be synchronized with densification: "
                f"ids={ids.numel()} splits={splits.numel()} dups={dups.numel()}"
            )
        split_children = ids[splits].repeat(int(nsamps))
        dup_children = ids[dups]
        self.gaussian_lineage_ids = torch.cat([ids, split_children, dup_children], dim=0).detach().long()

    def _sync_gaussian_lineage_ids_for_cull(self, culls: torch.Tensor) -> None:
        ids = self.gaussian_lineage_ids.reshape(-1).to(device=self.device)
        culls = culls.reshape(-1).to(device=self.device)
        if ids.numel() != culls.numel():
            raise ValueError(
                "gaussian lineage ids cannot be synchronized with culling: "
                f"ids={ids.numel()} culls={culls.numel()}"
            )
        self.gaussian_lineage_ids = ids[~culls].detach().long()

    def _sync_background_candidate_mask_for_cull(self, culls: torch.Tensor) -> None:
        """Keep candidate flags aligned when Gaussian parameters are culled."""
        if self._background_candidate_mask is None:
            return
        mask = self._background_candidate_mask.reshape(-1).to(device=self.device)
        culls = culls.reshape(-1).to(device=self.device)
        if mask.numel() != culls.numel():
            raise ValueError(
                "background candidate mask cannot be synchronized with culling: "
                f"mask={mask.numel()} culls={culls.numel()}"
            )
        self._background_candidate_mask = mask[~culls].detach().bool()
        self._background_candidate_num_points = int(self._background_candidate_mask.numel())

    def _load_background_candidate_mask(self) -> torch.Tensor:
        path_text = getattr(self.config, "background_candidate_mask_path", None)
        if not path_text:
            raise RuntimeError(
                "background_gradient_surgery_enabled=True requires "
                "background_candidate_mask_path"
            )
        path = Path(path_text)
        num_points = int(self.num_points)
        if (
            self._background_candidate_mask is not None
            and self._background_candidate_path == str(path)
            and self._background_candidate_num_points == num_points
        ):
            return self._background_candidate_mask

        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            mask = payload.get("candidate_mask")
            if mask is None:
                mask = payload.get("mask")
            view_count = payload.get("view_count")
        else:
            mask = payload
            view_count = None
        if mask is None:
            raise KeyError(f"{path} does not contain candidate_mask or mask")
        mask = mask.reshape(-1).bool()
        if mask.numel() != num_points:
            raise ValueError(
                f"candidate mask length {mask.numel()} does not match current "
                f"Gaussian count {num_points}"
            )
        if view_count is not None:
            view_count_t = torch.as_tensor(view_count).reshape(-1)
            if view_count_t.numel() != num_points:
                raise ValueError(
                    f"candidate view_count length {view_count_t.numel()} does not "
                    f"match current Gaussian count {num_points}"
                )
            min_view_count = int(getattr(self.config, "background_gradient_surgery_min_view_count", 5))
            if min_view_count > 0:
                mask &= view_count_t >= min_view_count
        mask = mask.to(device=self.device)
        self._background_candidate_mask = mask
        self._background_candidate_path = str(path)
        self._background_candidate_num_points = num_points
        CONSOLE.log(
            f"Loaded background candidate mask from {path}: "
            f"{int(mask.sum().item())}/{num_points} candidates"
        )
        return mask

    def _background_gradient_surgery_active(self) -> bool:
        if not getattr(self.config, "background_gradient_surgery_enabled", False):
            return False
        start = int(getattr(self.config, "background_gradient_surgery_start_step", 10001))
        return self.step >= start

    def _background_gradient_surgery_hook(self, grad: torch.Tensor) -> torch.Tensor:
        if not self._background_gradient_surgery_active():
            return grad
        mask = self._load_background_candidate_mask().reshape(-1, 1).to(device=grad.device)
        if mask.numel() != grad.numel():
            raise ValueError(
                f"candidate mask shape {tuple(mask.shape)} does not match "
                f"opacity gradient shape {tuple(grad.shape)}"
            )
        decrease_mult = float(getattr(self.config, "background_opacity_decrease_multiplier", 1.0))
        increase_mult = float(getattr(self.config, "background_opacity_increase_multiplier", 1.0))
        positive = grad > 0.0
        negative = grad < 0.0
        multiplier = torch.ones_like(grad)
        multiplier = torch.where(mask & positive, torch.full_like(multiplier, decrease_mult), multiplier)
        multiplier = torch.where(mask & negative, torch.full_like(multiplier, increase_mult), multiplier)

        if self.step % 500 == 0 and self._background_gradient_surgery_last_log_step != self.step:
            adjusted = grad * multiplier
            candidate_grad = adjusted[mask].detach().reshape(-1)
            CONSOLE.log(
                "Background gradient surgery step="
                f"{self.step} candidates={int(mask.sum().item())} "
                f"opacity_decrease_x={decrease_mult:.3g} "
                f"opacity_increase_x={increase_mult:.3g} "
                f"candidate_grad_abs_mean={float(candidate_grad.abs().mean().item()) if candidate_grad.numel() else 0.0:.3e}"
            )
            self._background_gradient_surgery_last_log_step = int(self.step)
        return grad * multiplier

    def _ensure_background_gradient_surgery_hook(self) -> None:
        if not getattr(self.config, "background_gradient_surgery_enabled", False):
            return
        param_id = id(self.opacities)
        if self._background_gradient_hook_handle is not None and self._background_gradient_hook_param_id == param_id:
            return
        if self._background_gradient_hook_handle is not None:
            self._background_gradient_hook_handle.remove()
        self._background_gradient_hook_handle = self.opacities.register_hook(self._background_gradient_surgery_hook)
        self._background_gradient_hook_param_id = param_id

    def _should_prepare_densification_regions(self) -> bool:
        if not getattr(self.config, "backscatter_region_mask_dir", None):
            return False
        return bool(
            getattr(self.config, "background_densification_enabled", False)
            or getattr(self.config, "background_densification_diagnostic_only", True)
        )

    def _load_region_mask_or_zeros(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        key: str,
        target: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._load_backscatter_region_mask(outputs=outputs, key=key, target=target)
        if mask is None:
            return torch.zeros(*target.shape[:2], 1, device=target.device, dtype=target.dtype)
        return mask.to(device=target.device, dtype=target.dtype).clamp(0.0, 1.0)

    def _prepare_densification_region_state(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> None:
        self.current_densification_region_weight = None
        self.current_densification_region_samples = None
        self.current_densification_accumulation_map = None
        if not self.training or not self._should_prepare_densification_regions():
            return
        if "camera_index" not in outputs:
            return

        target = outputs["rgb"]
        water_key = getattr(self.config, "background_water_mask_key", "water")
        foreground_key = getattr(self.config, "foreground_water_mask_key", "object")
        water = self._load_region_mask_or_zeros(outputs=outputs, key=water_key, target=target)
        foreground = self._load_region_mask_or_zeros(outputs=outputs, key=foreground_key, target=target)
        boundary = self._load_region_mask_or_zeros(outputs=outputs, key="boundary", target=target)
        uncertain = self._load_region_mask_or_zeros(outputs=outputs, key="uncertain", target=target)

        bg_weight = self._background_densification_effective_weight()
        uncertain_weight = float(getattr(self.config, "uncertain_densification_weight", 0.5))
        weight_map = torch.ones_like(water)
        weight_map = torch.where(water > 0.5, torch.full_like(weight_map, bg_weight), weight_map)
        weight_map = torch.where(uncertain > 0.5, torch.full_like(weight_map, uncertain_weight), weight_map)
        weight_map = torch.where((foreground > 0.5) | (boundary > 0.5), torch.ones_like(weight_map), weight_map)

        self.current_densification_region_weight = sample_pixel_map_at_gaussians(
            weight_map.detach(),
            self.xys.detach(),
            self.radii.detach(),
            height,
            width,
        )
        self.current_densification_region_samples = {
            "water": sample_pixel_map_at_gaussians(water.detach(), self.xys.detach(), self.radii.detach(), height, width),
            "object": sample_pixel_map_at_gaussians(
                foreground.detach(), self.xys.detach(), self.radii.detach(), height, width
            ),
            "boundary": sample_pixel_map_at_gaussians(
                boundary.detach(), self.xys.detach(), self.radii.detach(), height, width
            ),
            "uncertain": sample_pixel_map_at_gaussians(
                uncertain.detach(), self.xys.detach(), self.radii.detach(), height, width
            ),
            "weight": self.current_densification_region_weight,
        }
        if getattr(self.config, "opacity_accumulation_diagnostic_enabled", False):
            if "accumulation" in outputs:
                accumulation = outputs["accumulation"]
                self.current_densification_region_samples["sampled_accumulation"] = sample_pixel_map_at_gaussians(
                    accumulation.detach(), self.xys.detach(), self.radii.detach(), height, width
                )
                if accumulation.requires_grad:
                    accumulation.retain_grad()
                    self.current_densification_accumulation_map = accumulation
            if "final_transmittance" in outputs:
                self.current_densification_region_samples["sampled_final_transmittance"] = (
                    sample_pixel_map_at_gaussians(
                        outputs["final_transmittance"].detach(),
                        self.xys.detach(),
                        self.radii.detach(),
                        height,
                        width,
                    )
                )
            if "J_gaussian_raw" in outputs:
                self.current_densification_region_samples["sampled_j_gaussian_raw_luma"] = (
                    sample_pixel_map_at_gaussians(
                        outputs["J_gaussian_raw"].detach().mean(dim=-1, keepdim=True),
                        self.xys.detach(),
                        self.radii.detach(),
                        height,
                        width,
                    )
                )
            if "rgb_tail" in outputs:
                self.current_densification_region_samples["sampled_rgb_tail_luma"] = sample_pixel_map_at_gaussians(
                    outputs["rgb_tail"].detach().mean(dim=-1, keepdim=True),
                    self.xys.detach(),
                    self.radii.detach(),
                    height,
                    width,
                )
        outputs["background_region_mask"] = water
        outputs["densification_region_weight"] = weight_map

    def _tensor_region_stats(self, values: torch.Tensor) -> Dict[str, float]:
        values = values.detach().float().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}
        return {
            "mean": float(values.mean().item()),
            "median": float(torch.quantile(values, 0.50).item()),
            "p90": float(torch.quantile(values, 0.90).item()),
            "p95": float(torch.quantile(values, 0.95).item()),
        }

    def _signed_tensor_region_stats(self, values: torch.Tensor) -> Dict[str, float]:
        values = values.detach().float().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return {
                "mean": 0.0,
                "abs_mean": 0.0,
                "abs_median": 0.0,
                "abs_p90": 0.0,
                "abs_p95": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "decrease_pressure_mean": 0.0,
                "increase_pressure_mean": 0.0,
            }
        abs_values = values.abs()
        positive = values > 0
        negative = values < 0
        return {
            "mean": float(values.mean().item()),
            "abs_mean": float(abs_values.mean().item()),
            "abs_median": float(torch.quantile(abs_values, 0.50).item()),
            "abs_p90": float(torch.quantile(abs_values, 0.90).item()),
            "abs_p95": float(torch.quantile(abs_values, 0.95).item()),
            "positive_ratio": float(positive.float().mean().item()),
            "negative_ratio": float(negative.float().mean().item()),
            "decrease_pressure_mean": float(torch.relu(values).mean().item()),
            "increase_pressure_mean": float(torch.relu(-values).mean().item()),
        }

    def _tensor_correlation(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.detach().float().reshape(-1)
        b = b.detach().float().reshape(-1)
        finite = torch.isfinite(a) & torch.isfinite(b)
        if finite.sum() < 2:
            return 0.0
        a = a[finite]
        b = b[finite]
        a = a - a.mean()
        b = b - b.mean()
        denom = torch.sqrt((a * a).mean() * (b * b).mean()).clamp_min(1e-12)
        return float(((a * b).mean() / denom).item())

    def _record_densification_region_diagnostics(
        self,
        *,
        visible_mask: torch.Tensor,
        raw_grads: torch.Tensor,
        weighted_grads: torch.Tensor,
    ) -> None:
        if self.current_densification_region_samples is None:
            self.last_densification_region_stats = None
            return
        final_step = max(int(getattr(self.config, "num_steps", 0)) - 1, 0)
        if self.step % 500 != 0 and self.step != final_step:
            return

        visible = visible_mask.reshape(-1)
        if visible.numel() != raw_grads.reshape(-1).numel():
            return
        scale_factor = 0.5 * float(max(self.last_size[0], self.last_size[1]))
        raw_scaled = raw_grads.reshape(-1).detach().float() * scale_factor
        weighted_scaled = weighted_grads.reshape(-1).detach().float() * scale_factor
        opacity = torch.sigmoid(self.opacities.detach()).reshape(-1).float()
        depth = self.depths.detach().reshape(-1).float()
        scale = self.scales.detach().exp().max(dim=-1).values.reshape(-1).float()
        opacity_grad_available = self.opacities.grad is not None and self.opacities.grad.numel() == opacity.numel()
        if opacity_grad_available:
            opacity_logit_grad = self.opacities.grad.detach().reshape(-1).float()
        else:
            opacity_logit_grad = torch.zeros_like(opacity)
        opacity_alpha_grad = opacity_logit_grad / (opacity * (1.0 - opacity)).clamp_min(1e-6)
        scale_grad_available = self.scales.grad is not None and self.scales.grad.shape[0] == raw_scaled.shape[0]
        if scale_grad_available:
            scale_grad_norm = self.scales.grad.detach().reshape(raw_scaled.shape[0], -1).float().norm(dim=-1)
        else:
            scale_grad_norm = torch.zeros_like(raw_scaled)
        visibility_count = (
            self.vis_counts.detach().reshape(-1).float()
            if self.vis_counts is not None
            else torch.ones_like(raw_scaled)
        )
        high_raw = raw_scaled > float(getattr(self.config, "densify_grad_thresh", 0.0008))
        high_weighted = weighted_scaled > float(getattr(self.config, "densify_grad_thresh", 0.0008))
        split_size = scale > float(getattr(self.config, "densify_size_thresh", 0.001))
        split_raw = high_raw & split_size
        dup_raw = high_raw & ~split_size
        split_weighted = high_weighted & split_size
        dup_weighted = high_weighted & ~split_size

        samples = self.current_densification_region_samples
        water = samples["water"].reshape(-1) > 0.5
        obj = samples["object"].reshape(-1) > 0.5
        boundary = samples["boundary"].reshape(-1) > 0.5
        uncertain = samples["uncertain"].reshape(-1) > 0.5
        regions = {
            "water": water,
            "object": obj,
            "boundary": boundary,
            "uncertain": uncertain,
            "other": ~(water | obj | boundary | uncertain),
        }
        sampled_accumulation = samples.get("sampled_accumulation", torch.zeros_like(raw_scaled)).reshape(-1).float()
        sampled_final_transmittance = samples.get(
            "sampled_final_transmittance",
            torch.zeros_like(raw_scaled),
        ).reshape(-1).float()
        sampled_j_luma = samples.get("sampled_j_gaussian_raw_luma", torch.zeros_like(raw_scaled)).reshape(-1).float()
        sampled_tail_luma = samples.get("sampled_rgb_tail_luma", torch.zeros_like(raw_scaled)).reshape(-1).float()
        accumulation_grad_available = False
        sampled_accumulation_grad = torch.zeros_like(raw_scaled)
        accumulation_map = self.current_densification_accumulation_map
        if accumulation_map is not None and accumulation_map.grad is not None:
            grad_map = accumulation_map.grad.detach()
            sampled_accumulation_grad = sample_pixel_map_at_gaussians(
                grad_map,
                self.xys.detach(),
                self.radii.detach(),
                int(grad_map.shape[0]),
                int(grad_map.shape[1]),
            ).reshape(-1).float()
            accumulation_grad_available = True

        region_payload: Dict[str, Dict[str, Union[int, float, Dict[str, float]]]] = {}
        for name, region in regions.items():
            mask = visible & region
            count = int(mask.sum().item())
            denom = max(count, 1)
            region_payload[name] = {
                "visible_count": count,
                "raw_grad": self._tensor_region_stats(raw_scaled[mask]),
                "weighted_grad": self._tensor_region_stats(weighted_scaled[mask]),
                "grad_gt_densify_thresh_ratio": float((high_raw & mask).sum().item() / denom),
                "weighted_grad_gt_densify_thresh_ratio": float((high_weighted & mask).sum().item() / denom),
                "split_candidate_count": int((split_raw & mask).sum().item()),
                "duplicate_candidate_count": int((dup_raw & mask).sum().item()),
                "weighted_split_candidate_count": int((split_weighted & mask).sum().item()),
                "weighted_duplicate_candidate_count": int((dup_weighted & mask).sum().item()),
                "mean_opacity": float(opacity[mask].mean().item()) if count else 0.0,
                "mean_depth": float(depth[mask].mean().item()) if count else 0.0,
                "mean_scale": float(scale[mask].mean().item()) if count else 0.0,
                "mean_visibility_count": float(visibility_count[mask].mean().item()) if count else 0.0,
                "opacity_logit_grad": self._signed_tensor_region_stats(opacity_logit_grad[mask]),
                "opacity_alpha_grad": self._signed_tensor_region_stats(opacity_alpha_grad[mask]),
                "scale_grad_norm": self._tensor_region_stats(scale_grad_norm[mask]),
                "sampled_accumulation": self._tensor_region_stats(sampled_accumulation[mask]),
                "sampled_final_transmittance": self._tensor_region_stats(sampled_final_transmittance[mask]),
                "sampled_j_gaussian_raw_luma": self._tensor_region_stats(sampled_j_luma[mask]),
                "sampled_rgb_tail_luma": self._tensor_region_stats(sampled_tail_luma[mask]),
                "accumulation_grad": self._signed_tensor_region_stats(sampled_accumulation_grad[mask]),
                "accumulation_opacity_grad_corr": self._tensor_correlation(
                    sampled_accumulation[mask],
                    opacity_logit_grad[mask],
                )
                if count
                else 0.0,
            }

        visible_grad_sum = raw_scaled[visible].sum().clamp_min(1e-12)
        bg_mask = visible & water
        total_split = int((split_raw & visible).sum().item())
        total_dup = int((dup_raw & visible).sum().item())
        total_split_weighted = int((split_weighted & visible).sum().item())
        total_dup_weighted = int((dup_weighted & visible).sum().item())
        visible_opacity_abs = opacity_logit_grad[visible].abs().sum().clamp_min(1e-12)
        visible_opacity_decrease = torch.relu(opacity_logit_grad[visible]).sum().clamp_min(1e-12)
        visible_opacity_increase = torch.relu(-opacity_logit_grad[visible]).sum().clamp_min(1e-12)
        visible_accum_abs = sampled_accumulation_grad[visible].abs().sum().clamp_min(1e-12)
        visible_accum_decrease = torch.relu(sampled_accumulation_grad[visible]).sum().clamp_min(1e-12)
        visible_accum_increase = torch.relu(-sampled_accumulation_grad[visible]).sum().clamp_min(1e-12)
        payload: Dict[str, Union[int, float, str, bool, Dict[str, object]]] = {
            "step": int(self.step),
            "total_gaussians": int(self.num_points),
            "visible_gaussians": int(visible.sum().item()),
            "background_densification_enabled": bool(getattr(self.config, "background_densification_enabled", False)),
            "background_densification_diagnostic_only": bool(
                getattr(self.config, "background_densification_diagnostic_only", True)
            ),
            "background_densification_effective_weight": float(self._background_densification_effective_weight()),
            "opacity_accumulation_diagnostic_enabled": bool(
                getattr(self.config, "opacity_accumulation_diagnostic_enabled", False)
            ),
            "opacity_grad_available": bool(opacity_grad_available),
            "scale_grad_available": bool(scale_grad_available),
            "accumulation_grad_available": bool(accumulation_grad_available),
            "regions": region_payload,
            "background_gradient_fraction": float(raw_scaled[bg_mask].sum().item() / visible_grad_sum.item())
            if bg_mask.any()
            else 0.0,
            "background_split_candidate_fraction": float((split_raw & bg_mask).sum().item() / max(total_split, 1)),
            "background_duplicate_candidate_fraction": float((dup_raw & bg_mask).sum().item() / max(total_dup, 1)),
            "background_weighted_split_candidate_fraction": float(
                (split_weighted & bg_mask).sum().item() / max(total_split_weighted, 1)
            ),
            "background_weighted_duplicate_candidate_fraction": float(
                (dup_weighted & bg_mask).sum().item() / max(total_dup_weighted, 1)
            ),
            "background_opacity_grad_abs_fraction": float(
                opacity_logit_grad[bg_mask].abs().sum().item() / visible_opacity_abs.item()
            )
            if bg_mask.any() and opacity_grad_available
            else 0.0,
            "background_opacity_decrease_pressure_fraction": float(
                torch.relu(opacity_logit_grad[bg_mask]).sum().item() / visible_opacity_decrease.item()
            )
            if bg_mask.any() and opacity_grad_available
            else 0.0,
            "background_opacity_increase_pressure_fraction": float(
                torch.relu(-opacity_logit_grad[bg_mask]).sum().item() / visible_opacity_increase.item()
            )
            if bg_mask.any() and opacity_grad_available
            else 0.0,
            "background_accumulation_grad_abs_fraction": float(
                sampled_accumulation_grad[bg_mask].abs().sum().item() / visible_accum_abs.item()
            )
            if bg_mask.any() and accumulation_grad_available
            else 0.0,
            "background_accumulation_decrease_pressure_fraction": float(
                torch.relu(sampled_accumulation_grad[bg_mask]).sum().item() / visible_accum_decrease.item()
            )
            if bg_mask.any() and accumulation_grad_available
            else 0.0,
            "background_accumulation_increase_pressure_fraction": float(
                torch.relu(-sampled_accumulation_grad[bg_mask]).sum().item() / visible_accum_increase.item()
            )
            if bg_mask.any() and accumulation_grad_available
            else 0.0,
        }
        self.last_densification_region_stats = payload
        CONSOLE.log(
            "Densification region step="
            f"{self.step} bg_grad_frac={payload['background_gradient_fraction']:.4f} "
            f"bg_split_frac={payload['background_split_candidate_fraction']:.4f} "
            f"bg_dup_frac={payload['background_duplicate_candidate_fraction']:.4f}"
        )

        log_path = getattr(self.config, "densification_region_log_path", None)
        if log_path:
            try:
                path = Path(log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf8") as f:
                    f.write(json.dumps(payload) + "\n")
            except Exception as exc:
                CONSOLE.log(f"[yellow]Failed to write densification region log: {exc}[/yellow]")

    def _gmvc_bounded_projection_alpha(self) -> float:
        if not bool(getattr(self.config, "gmvc_bounded_medium_enabled", False)):
            return 0.0
        start = int(getattr(self.config, "gmvc_bounded_medium_start_step", getattr(self.config, "gmvc_start_step", 10000)))
        ramp = int(getattr(self.config, "gmvc_bounded_medium_projection_steps", 500))
        if int(self.step) < start:
            return 0.0
        if ramp <= 0:
            return 1.0
        return min(max((int(self.step) - start) / max(float(ramp), 1.0), 0.0), 1.0)

    @staticmethod
    def _logit_unit(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        clipped = value.clamp(float(eps), 1.0 - float(eps))
        return torch.log(clipped) - torch.log1p(-clipped)

    def _maybe_initialize_gmvc_bounded_centers(
        self,
        *,
        medium_attn: torch.Tensor,
        medium_bs: torch.Tensor,
        b_inf: torch.Tensor,
    ) -> None:
        if self._gmvc_bounded_initialized:
            return
        if not bool(getattr(self.config, "gmvc_bounded_init_from_first_batch", True)):
            self._gmvc_bounded_initialized = True
            return
        eps = float(getattr(self.config, "gmvc_eps", 1e-4))
        with torch.no_grad():
            log_attn = torch.log(medium_attn.detach().clamp_min(eps)).reshape(-1, 3)
            log_bs = torch.log(medium_bs.detach().clamp_min(eps)).reshape(-1, 3)
            binf_logit = self._logit_unit(b_inf.detach(), eps=eps).reshape(-1, 3)
            if log_attn.numel() > 0:
                self.gmvc_bounded_log_attn_center.data.copy_(log_attn.median(dim=0).values)
                self.gmvc_bounded_log_bs_center.data.copy_(log_bs.median(dim=0).values)
                self.gmvc_bounded_binf_logit_center.data.copy_(binf_logit.median(dim=0).values)
        self._gmvc_bounded_initialized = True

    def _apply_gmvc_bounded_medium(self, medium: MediumFieldOutput) -> MediumFieldOutput:
        alpha = self._gmvc_bounded_projection_alpha()
        if alpha <= 0.0 or self.config.zero_medium:
            return medium

        eps = float(getattr(self.config, "gmvc_eps", 1e-4))
        b_inf_in = medium.b_inf if medium.b_inf is not None else medium.rgb
        self._maybe_initialize_gmvc_bounded_centers(
            medium_attn=medium.attn,
            medium_bs=medium.bs,
            b_inf=b_inf_in,
        )

        beta_scale = max(float(getattr(self.config, "gmvc_bounded_beta_log_scale", 0.15)), eps)
        binf_scale = max(float(getattr(self.config, "gmvc_bounded_binf_logit_scale", 0.10)), eps)

        def bounded_q(q: torch.Tensor, center: torch.Tensor, scale: float) -> torch.Tensor:
            center = center.to(device=q.device, dtype=q.dtype).view(*([1] * (q.ndim - 1)), 3)
            projected = center + float(scale) * torch.tanh((q - center) / float(scale))
            return (1.0 - alpha) * q + alpha * projected

        log_attn = torch.log(medium.attn.clamp_min(eps))
        log_bs = torch.log(medium.bs.clamp_min(eps))
        binf_logit = self._logit_unit(b_inf_in, eps=eps)
        bounded_attn = torch.exp(bounded_q(log_attn, self.gmvc_bounded_log_attn_center, beta_scale))
        bounded_bs = torch.exp(bounded_q(log_bs, self.gmvc_bounded_log_bs_center, beta_scale))
        bounded_binf = torch.sigmoid(bounded_q(binf_logit, self.gmvc_bounded_binf_logit_center, binf_scale))
        mode = self._effective_b_inf_mode()
        bounded_rgb = bounded_binf if mode == "tied" else medium.rgb
        bounded_b_inf = bounded_binf if medium.b_inf is not None or mode == "tied" else None
        return MediumFieldOutput(
            rgb=bounded_rgb,
            bs=bounded_bs,
            attn=bounded_attn,
            directions=medium.directions,
            b_inf=bounded_b_inf,
            b_inf_residual=medium.b_inf_residual,
        )

    def _query_gmvc_medium_points(
        self,
        directions: torch.Tensor,
        image_xy: torch.Tensor,
        camera_centers: torch.Tensor,
        depth_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        scene_center, scene_scale = self._get_scene_normalization(dtype=directions.dtype, device=directions.device)
        medium = self.medium_field.query_points(
            directions=directions,
            image_xy=image_xy,
            camera_centers=camera_centers,
            density_bias=self.medium_density_bias,
            mlp_type=self.config.mlp_type,
            zero_medium=self.config.zero_medium,
            context_mode=getattr(self.config, "medium_context_mode", "dir_only"),
            scene_center=scene_center,
            scene_scale=scene_scale,
            camera_context_scale=getattr(self.config, "medium_camera_context_scale", 1.0),
            camera_context_dropout=getattr(self.config, "medium_camera_context_dropout", 0.0),
            training=self.training,
            depth_context=depth_context,
            enable_b_inf=self._b_inf_requires_head(),
            b_inf_mode=self._effective_b_inf_mode(),
            b_inf_residual_scale=getattr(self.config, "b_inf_residual_scale", 0.02),
        )
        medium = self._apply_gmvc_bounded_medium(medium)
        return {
            "medium_rgb": medium.rgb,
            "medium_bs": medium.bs,
            "medium_attn": medium.attn,
            "b_inf": medium.b_inf if medium.b_inf is not None else medium.rgb,
        }

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
        medium = self.medium_field(
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
            enable_b_inf=self._b_inf_requires_head(),
            b_inf_mode=self._effective_b_inf_mode(),
            b_inf_residual_scale=getattr(self.config, "b_inf_residual_scale", 0.02),
        )
        medium = self._apply_gmvc_bounded_medium(medium)
        return self._maybe_scale_gmvc_v3_object_medium_grad(medium)

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

    def _camera_index_from_outputs(self, outputs: Dict[str, torch.Tensor]) -> Optional[int]:
        camera_index = outputs.get("camera_index")
        if camera_index is None:
            return None
        return int(camera_index.detach().cpu().item())

    def _load_backscatter_region_mask(
        self,
        *,
        outputs: Dict[str, torch.Tensor],
        key: str,
        target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        mask_dir = getattr(self.config, "backscatter_region_mask_dir", None)
        if not mask_dir:
            return None
        image_idx = self._camera_index_from_outputs(outputs)
        if image_idx is None:
            return None

        cache_key = (str(mask_dir), image_idx, key)
        if cache_key not in self._backscatter_mask_cache:
            path = Path(mask_dir) / f"view_{image_idx:04d}_regions.pt"
            if not path.exists():
                self._backscatter_mask_cache[cache_key] = None
            else:
                payload = torch.load(path, map_location="cpu")
                mask = payload.get(key) if isinstance(payload, dict) else None
                if mask is None:
                    self._backscatter_mask_cache[cache_key] = None
                else:
                    if mask.ndim == 2:
                        mask = mask[..., None]
                    self._backscatter_mask_cache[cache_key] = mask.bool().cpu()

        mask_cpu = self._backscatter_mask_cache[cache_key]
        if mask_cpu is None:
            return None
        mask = mask_cpu.to(device=target.device, dtype=target.dtype)
        if mask.shape[:2] != target.shape[:2]:
            mask = F.interpolate(
                mask.permute(2, 0, 1)[None],
                size=target.shape[:2],
                mode="nearest",
            )[0].permute(1, 2, 0)
        return mask.clamp(0.0, 1.0)

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

    def _dual_color_ramp_weight(self, weight: float) -> float:
        if weight <= 0.0:
            return 0.0
        start = int(getattr(self.config, "dual_color_loss_start_step", 0))
        ramp = int(getattr(self.config, "dual_color_loss_ramp_steps", 0))
        if self.step < start:
            return 0.0
        if ramp <= 0:
            return float(weight)
        return float(weight) * min((self.step - start) / max(float(ramp), 1.0), 1.0)

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
        camera_index: Optional[int] = None
        if camera.metadata is not None and "cam_idx" in camera.metadata:
            camera_index_value = camera.metadata["cam_idx"]
            if torch.is_tensor(camera_index_value):
                camera_index = int(camera_index_value.detach().cpu().reshape(-1)[0].item())
            else:
                camera_index = int(camera_index_value)
        if self._cleanup_enabled() and self.training:
            self.cleanup_current_alpha = None
            self.cleanup_current_ownership = None
        self.current_densification_region_weight = None
        self.current_densification_region_samples = None
        
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
        medium_attn_raw = medium.attn
        direct_optical_depth_scale = float(getattr(self.config, "direct_optical_depth_scale", 1.0))
        if direct_optical_depth_scale < 0.0:
            raise ValueError("direct_optical_depth_scale must be non-negative")
        medium_attn = medium_attn_raw * direct_optical_depth_scale

        def _empty_gaussian_outputs(rgb: torch.Tensor, depth_value: float = 10.0) -> Dict[str, torch.Tensor]:
            depth = rgb.new_ones(*rgb.shape[:2], 1) * float(depth_value)
            accumulation = rgb.new_zeros(*rgb.shape[:2], 1)
            j_empty = torch.zeros_like(rgb)
            final_transmittance = rgb.new_ones(*rgb.shape[:2], 1)
            out = {
                "rgb": rgb,
                "depth": depth,
                "depth_second_moment": depth.square(),
                "depth_variance": accumulation,
                "depth_std_relative": rgb.new_ones(*rgb.shape[:2], 1),
                "first_depth": depth,
                "last_depth": depth,
                "final_transmittance": final_transmittance,
                "hit_q_alpha": accumulation,
                "hit_q_conc": accumulation,
                "hit_confidence": accumulation,
                "accumulation": accumulation,
                "background": medium_rgb,
                "rgb_object": torch.zeros_like(rgb),
                "J": j_empty,
                "J_raw": j_empty,
                "J_gaussian": j_empty,
                "J_gaussian_raw": j_empty,
                "J_object": j_empty,
                "J_object_raw": j_empty,
                "rgb_clear": j_empty,
                "rgb_clear_clamp": j_empty,
                "rgb_medium": rgb,
                "rgb_medium_finite": rgb,
                "rgb_medium_total": rgb,
                "tail_weight_last": accumulation,
                "tail_medium_original": j_empty,
                "rgb_tail": j_empty,
                "rgb_implicit_tail": rgb,
                "pred_image": rgb,
                "medium_rgb": medium_rgb,
                "medium_bs": medium_bs,
                "medium_attn": medium_attn,
                "medium_attn_raw": medium_attn_raw,
                "direct_optical_depth_scale": medium_attn.new_tensor(direct_optical_depth_scale),
            }
            if medium.b_inf is not None:
                out["b_inf"] = medium.b_inf
                out["b_inf_minus_A_abs"] = torch.abs(medium.b_inf - medium_rgb)
            return out

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return _empty_gaussian_outputs(medium_rgb)
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
            return _empty_gaussian_outputs(medium_rgb)

        if self.training and self.xys.requires_grad:
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
        dual_color = None
        if getattr(self.config, "dual_color_enabled", False):
            dual_color = compute_dual_gaussian_colors(
                means=means_crop,
                features_dc=features_dc_crop,
                features_rest=features_rest_crop,
                camera_position=camera.camera_to_worlds[..., :3, 3],
                sh_degree=self.config.sh_degree,
                active_sh_degree=n,
                luminance_scale=getattr(self.config, "clear_sh_luminance_scale", 1.0),
                chroma_scale=getattr(self.config, "clear_sh_chroma_scale", 0.0),
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
        self.xys_grad_abs_proxy = None
        self.xys_grad_abs_capacity = None

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
            medium_attn_raw = medium.attn
            medium_attn = medium_attn_raw * direct_optical_depth_scale

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

        dual_render = None
        if dual_color is not None:
            dual_render = self.underwater_rasterizer.rasterize(  # type: ignore
                xys=self.xys,
                xys_grad_abs=self.xys_grad_abs,
                depths=depths,
                radii=self.radii,
                conics=conics,
                num_tiles_hit=num_tiles_hit,
                colors=dual_color.intrinsic_rgb,
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
        if dual_render is not None:
            rgb_clear = dual_render.rgb_clear
            j_gaussian_raw = dual_render.j_raw
            j_gaussian = dual_render.j_gaussian
            j_object_raw = dual_render.j_raw
            j_object = dual_render.j_gaussian

        def _scale_aux_grad(value: torch.Tensor, scale: float) -> torch.Tensor:
            scale = max(float(scale), 0.0)
            if scale <= 0.0:
                return value.detach()
            if (not value.is_floating_point()) or (not value.requires_grad):
                return value
            if abs(scale - 1.0) < 1e-8:
                return value
            return value.detach() + scale * (value - value.detach())

        def _branch_scaled_grad(value: torch.Tensor, scale: float) -> torch.Tensor:
            scale = max(float(scale), 0.0)
            if (not value.is_floating_point()) or (not value.requires_grad):
                return value
            detached = value.detach()
            return detached + scale * (value - detached)

        def _capacity_grad_scale(name: str, geometry_scale: float) -> float:
            value = float(getattr(self.config, name, -1.0))
            return geometry_scale if value < 0.0 else value

        clear_proxy_render = None
        capacity_control_render = None
        chroma_weight_config = float(getattr(self.config, "lambda_background_clear_chroma", 0.0))
        chroma_active_by_step = (
            chroma_weight_config > 0.0
            and (not self.training or self.step >= int(getattr(self.config, "background_clear_chroma_start_step", 10000)))
        )
        capacity_weight_config = float(getattr(self.config, "lambda_budgeted_capacity", 0.0))
        capacity_active_by_step = (
            bool(getattr(self.config, "budgeted_capacity_enabled", False))
            and capacity_weight_config > 0.0
            and (not self.training or self.step >= int(getattr(self.config, "budgeted_capacity_start_step", 4000)))
        )
        halo_weight_config = float(getattr(self.config, "lambda_halo_capacity", 0.0))
        halo_active_by_step = (
            bool(getattr(self.config, "halo_capacity_enabled", False))
            and halo_weight_config > 0.0
            and (not self.training or self.step >= int(getattr(self.config, "halo_capacity_start_step", 4000)))
        )
        core_zero_weight_config = float(getattr(self.config, "lambda_core_zero_capacity", 0.0))
        core_zero_active_by_step = (
            bool(getattr(self.config, "core_zero_capacity_enabled", False))
            and core_zero_weight_config > 0.0
            and (not self.training or self.step >= int(getattr(self.config, "core_zero_capacity_start_step", 1000)))
        )
        gmvc_intrinsic_proxy_active = self._gmvc_intrinsic_active() and self._gmvc_intrinsic_uses_proxy()
        clear_proxy_required = bool(
            getattr(self.config, "clear_proxy_enabled", False)
            or chroma_active_by_step
            or halo_active_by_step
            or gmvc_intrinsic_proxy_active
        )
        capacity_control_required = bool(
            getattr(self.config, "capacity_control_enabled", False)
            and (capacity_active_by_step or halo_active_by_step or core_zero_active_by_step)
        )
        if capacity_control_required:
            self.xys_grad_abs_capacity = torch.zeros_like(self.xys)
            capacity_geometry_grad_scale = float(
                getattr(self.config, "capacity_control_geometry_gradient_scale", 1.0)
            )
            capacity_position_grad_scale = _capacity_grad_scale(
                "capacity_control_position_gradient_scale",
                capacity_geometry_grad_scale,
            )
            capacity_depth_grad_scale = _capacity_grad_scale(
                "capacity_control_depth_gradient_scale",
                capacity_geometry_grad_scale,
            )
            capacity_footprint_grad_scale = _capacity_grad_scale(
                "capacity_control_footprint_gradient_scale",
                capacity_geometry_grad_scale,
            )
            capacity_opacity_grad_scale = float(
                getattr(self.config, "capacity_control_opacity_gradient_scale", 1.0)
            )
            capacity_scale_shrink_only = bool(
                getattr(self.config, "capacity_control_scale_shrink_only", False)
            )
            capacity_scale_clip_quantile = float(
                getattr(self.config, "capacity_control_scale_shrink_clip_quantile", -1.0)
            )
            capacity_scale_clip_value = float(
                getattr(self.config, "capacity_control_scale_shrink_clip_value", 0.0)
            )
            capacity_scale_control_required = (
                capacity_scale_shrink_only
                or 0.0 < capacity_scale_clip_quantile <= 1.0
                or capacity_scale_clip_value > 0.0
            )
            capacity_control_opacities = _branch_scaled_grad(opacities, capacity_opacity_grad_scale)
            capacity_control_scales = None
            if capacity_scale_control_required:
                capacity_control_scales = _branch_scaled_grad(scales_crop, capacity_footprint_grad_scale)
                if capacity_control_scales.requires_grad:

                    def _capacity_scale_shrink_hook(
                        grad: torch.Tensor,
                        shrink_only: bool = capacity_scale_shrink_only,
                        clip_quantile: float = capacity_scale_clip_quantile,
                        clip_value: float = capacity_scale_clip_value,
                    ) -> torch.Tensor:
                        controlled = grad
                        if shrink_only:
                            # scales are log-scales, so positive gradients shrink the footprint
                            # under gradient descent; negative gradients grow it.
                            controlled = controlled.clamp_min(0.0)
                        if clip_value > 0.0:
                            controlled = controlled.clamp_max(float(clip_value))
                        if 0.0 < clip_quantile <= 1.0:
                            positive = controlled.detach()[torch.isfinite(controlled.detach()) & (controlled.detach() > 0.0)]
                            if positive.numel() > 0:
                                threshold = torch.quantile(positive.float(), float(clip_quantile)).to(
                                    device=controlled.device,
                                    dtype=controlled.dtype,
                                )
                                controlled = torch.minimum(controlled, threshold)
                        return controlled

                    capacity_control_scales.register_hook(_capacity_scale_shrink_hook)
                capacity_xys, capacity_depths, capacity_radii, capacity_conics, _, capacity_num_tiles_hit, _ = (
                    self.underwater_rasterizer.project(  # type: ignore
                        means=_branch_scaled_grad(means_crop, capacity_position_grad_scale),
                        scales=capacity_control_scales,
                        quats=quats_crop.detach(),
                        viewmat=viewmat,
                        fx=camera.fx.item(),
                        fy=camera.fy.item(),
                        cx=cx,
                        cy=cy,
                        height=H,
                        width=W,
                        clip_thresh=self.config.clip_thresh,
                    )
                )
                capacity_control_render = self.underwater_rasterizer.rasterize_clear_proxy(  # type: ignore
                    xys=capacity_xys,
                    xys_grad_abs=self.xys_grad_abs_capacity,
                    depths=_branch_scaled_grad(capacity_depths, capacity_depth_grad_scale),
                    radii=capacity_radii,
                    conics=capacity_conics,
                    num_tiles_hit=capacity_num_tiles_hit,
                    colors=rgbs.detach(),
                    opacities=capacity_control_opacities,
                    height=H,
                    width=W,
                    step=self.step,
                )
            else:
                capacity_control_render = self.underwater_rasterizer.rasterize_clear_proxy(  # type: ignore
                    xys=_branch_scaled_grad(self.xys, capacity_position_grad_scale),
                    xys_grad_abs=self.xys_grad_abs_capacity,
                    depths=_branch_scaled_grad(depths, capacity_depth_grad_scale),
                    radii=_branch_scaled_grad(self.radii, capacity_footprint_grad_scale),
                    conics=_branch_scaled_grad(conics, capacity_footprint_grad_scale),
                    num_tiles_hit=num_tiles_hit,
                    colors=rgbs.detach(),
                    opacities=capacity_control_opacities,
                    height=H,
                    width=W,
                    step=self.step,
                )
            if capacity_control_opacities.requires_grad:
                capacity_control_opacities.retain_grad()
        if clear_proxy_required:
            self.xys_grad_abs_proxy = torch.zeros_like(self.xys)

            proxy_xys = self.xys
            proxy_depths = depths
            proxy_radii = self.radii
            proxy_conics = conics
            proxy_colors = rgbs
            proxy_opacities = opacities
            proxy_uses_dc_colors = False
            if gmvc_intrinsic_proxy_active and bool(getattr(self.config, "gmvc_intrinsic_use_dc_proxy", True)):
                proxy_colors = SH2RGB(features_dc_crop) if self.config.sh_degree > 0 else torch.sigmoid(features_dc_crop)
                proxy_uses_dc_colors = True
            if gmvc_intrinsic_proxy_active or bool(getattr(self.config, "clear_proxy_appearance_only", False)):
                proxy_geometry_grad_scale = 0.0
                proxy_opacity_grad_scale = 0.0
            else:
                proxy_geometry_grad_scale = float(
                    getattr(self.config, "clear_proxy_geometry_gradient_scale", 1.0)
                )
                proxy_opacity_grad_scale = float(
                    getattr(self.config, "clear_proxy_opacity_gradient_scale", 1.0)
                )
            proxy_color_grad_scale = (
                1.0 if gmvc_intrinsic_proxy_active else float(getattr(self.config, "clear_proxy_color_gradient_scale", 1.0))
            )
            proxy_xys = _scale_aux_grad(proxy_xys, proxy_geometry_grad_scale)
            proxy_depths = _scale_aux_grad(proxy_depths, proxy_geometry_grad_scale)
            proxy_radii = _scale_aux_grad(proxy_radii, proxy_geometry_grad_scale)
            proxy_conics = _scale_aux_grad(proxy_conics, proxy_geometry_grad_scale)
            proxy_opacities = _scale_aux_grad(proxy_opacities, proxy_opacity_grad_scale)
            proxy_colors = _scale_aux_grad(proxy_colors, proxy_color_grad_scale)
            clear_proxy_render = self.underwater_rasterizer.rasterize_clear_proxy(  # type: ignore
                xys=proxy_xys,
                xys_grad_abs=self.xys_grad_abs_proxy,
                depths=proxy_depths,
                radii=proxy_radii,
                conics=proxy_conics,
                num_tiles_hit=num_tiles_hit,
                colors=proxy_colors,
                opacities=proxy_opacities,
                height=H,
                width=W,
                step=self.step,
            )
        else:
            proxy_uses_dc_colors = False
        tail_weight_last = render.final_transmittance * torch.exp(-medium_bs * render.last_depth)
        tail_medium_original = tail_weight_last * medium_rgb
        rgb_medium_finite = render.rgb_medium - tail_medium_original
        b_inf_mode = self._effective_b_inf_mode()
        b_inf = medium.b_inf
        rgb_tail = tail_medium_original
        if b_inf_mode != "implicit":
            if b_inf is None:
                raise RuntimeError(f"b_inf_mode='{b_inf_mode}' requires a B_inf output")
            rgb_tail = tail_weight_last * b_inf
            if not getattr(self.config, "infinite_water_enabled", False):
                rgb = render.rgb_object + rgb_medium_finite + rgb_tail
        rgb_medium_total = rgb_medium_finite + rgb_tail
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
            if b_inf is None:
                raise RuntimeError("infinite_water_enabled=True requires medium.b_inf output")
            ownership = compute_infinite_water_ownership(
                accumulation=render.accumulation,
                depth=render.depth,
                rgb_near=render.rgb,
                b_inf=b_inf,
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
                j_object_raw = j_gaussian_raw
            elif compose_mode == "rgb_mix":
                m_obj_eff = 1.0 - ownership.m_inf_eff
                rgb = m_obj_eff * render.rgb + ownership.m_inf_eff * b_inf
                rgb_clear = m_obj_eff * rgb_clear
                j_object_raw = m_obj_eff * j_gaussian_raw
            elif compose_mode == "tail_approx":
                if getattr(self.config, "infinite_water_occupancy_limited", True):
                    tail_gate = (1.0 - render.accumulation).detach().clamp(0.0, 1.0)
                else:
                    tail_gate = torch.ones_like(render.accumulation)
                rgb = render.rgb + ownership.m_inf * tail_gate * (b_inf - medium_rgb)
                j_object_raw = (1.0 - ownership.m_inf_eff) * j_gaussian_raw
            elif compose_mode == "closed_tail":
                tail_color = (1.0 - ownership.m_inf_eff) * medium_rgb + ownership.m_inf_eff * b_inf
                rgb = render.rgb_object + rgb_medium_finite + tail_weight_last * tail_color
                j_object_raw = j_gaussian_raw
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
            "rgb_medium_total": rgb_medium_total,
            "tail_weight_last": tail_weight_last,
            "tail_medium_original": tail_medium_original,
            "rgb_tail": rgb_tail,
            "rgb_implicit_tail": render.rgb,
            "pred_image": rgb,
            "medium_rgb": medium_rgb,
            "medium_bs": medium_bs,
            "medium_attn": medium_attn,
            "medium_attn_raw": medium_attn_raw,
            "direct_optical_depth_scale": medium_attn.new_tensor(direct_optical_depth_scale),
        }
        if clear_proxy_render is not None:
            j_proxy_raw = clear_proxy_render.rgb
            outputs["J_proxy_raw"] = j_proxy_raw
            outputs["J_proxy"] = torch.clamp(j_proxy_raw, 0.0, 1.0)
            outputs["J_proxy_uses_dc_colors"] = j_proxy_raw.new_tensor(float(proxy_uses_dc_colors))
            outputs["J_proxy_abs_diff_from_renderer_clear"] = torch.abs(
                j_proxy_raw.detach() - j_gaussian_raw.detach()
            )
            outputs["J_proxy_rgb_object"] = clear_proxy_render.rgb_object
            outputs["J_proxy_accumulation"] = clear_proxy_render.accumulation
        if capacity_control_render is not None:
            outputs["capacity_control_accumulation"] = capacity_control_render.accumulation
            outputs["capacity_control_opacities"] = capacity_control_opacities
            outputs["main_render_opacities"] = opacities
            if capacity_control_scales is not None:
                outputs["capacity_control_scales"] = capacity_control_scales
        if camera_index is not None:
            outputs["camera_index"] = torch.tensor(float(camera_index), device=self.device)
        if b_inf is not None:
            outputs["b_inf"] = b_inf
            outputs["b_inf_minus_A_abs"] = torch.abs(b_inf - medium_rgb)
            if medium.b_inf_residual is not None:
                outputs["b_inf_residual"] = medium.b_inf_residual
        if dual_color is not None and dual_render is not None:
            outputs["dual_color_active_sh_degree"] = torch.tensor(float(n), device=self.device)
            outputs["dual_color_visible_mask"] = visible_mask.detach()
            outputs["dual_color_underwater_rgb"] = dual_color.underwater_rgb
            outputs["dual_color_intrinsic_rgb"] = dual_color.intrinsic_rgb
            outputs["dual_color_view_residual"] = dual_color.view_residual
            outputs["dual_color_luminance_residual"] = dual_color.luminance_residual
            outputs["dual_color_chroma_residual"] = dual_color.chroma_residual
            outputs["dual_color_j_residual_raw"] = render.j_raw - dual_render.j_raw
            outputs["rgb_object_intrinsic"] = dual_render.rgb_object
            outputs["J_intrinsic"] = dual_render.j_gaussian
            outputs["J_intrinsic_raw"] = dual_render.j_raw
        if self._appearance_enabled():
            outputs["appearance_active_sh_degree"] = torch.tensor(float(n), device=self.device)
            outputs["appearance_visible_mask"] = visible_mask.detach()
            outputs["appearance_sh_residual"] = sh_residual
            outputs["appearance_dc_rgb"] = dc_rgb
            outputs["appearance_low_trans_weight"] = low_trans_weight
            outputs["appearance_pixel_low_trans_weight"] = pixel_low_trans_weight
        if getattr(self.config, "infinite_water_enabled", False):
            outputs["m_inf"] = ownership.m_inf
            outputs["m_inf_eff"] = ownership.m_inf_eff
            outputs["m_support"] = ownership.m_inf
            outputs["m_render"] = ownership.m_inf_eff
            outputs["hit_object_protection"] = self._infinite_water_hit_object_protection(outputs)
            outputs["m_capacity"] = self._infinite_water_capacity_support(outputs)
            outputs["m_inf_alpha_evidence"] = ownership.alpha_evidence
            outputs["m_inf_depth_evidence"] = ownership.depth_evidence
            outputs["m_inf_color_evidence"] = ownership.color_evidence
        self._prepare_densification_region_state(outputs=outputs, height=H, width=W)
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
            if "medium_attn_raw" in outputs:
                metrics_dict[f"medium_attn_raw_{i}"] = outputs["medium_attn_raw"][:, :, i].mean()
                metrics_dict[f"medium_attn_effective_{i}"] = outputs["medium_attn"][:, :, i].mean()
            metrics_dict[f"medium_bs_{i}"] = outputs["medium_bs"][:, :, i].mean()
            metrics_dict[f"medium_rgb_{i}"] = outputs["medium_rgb"][:, :, i].mean()
            if "b_inf" in outputs:
                metrics_dict[f"b_inf_{i}"] = outputs["b_inf"][:, :, i].mean()
        if "direct_optical_depth_scale" in outputs:
            metrics_dict["direct_optical_depth_scale"] = outputs["direct_optical_depth_scale"]
        if "b_inf_minus_A_abs" in outputs:
            metrics_dict["b_inf_minus_A_abs_mean"] = outputs["b_inf_minus_A_abs"].mean()
            metrics_dict["b_inf_minus_A_abs_max"] = outputs["b_inf_minus_A_abs"].max()
        if "rgb_implicit_tail" in outputs:
            rgb_abs = torch.abs(outputs["pred_image"] - outputs["rgb_implicit_tail"])
            metrics_dict["backscatter_closure_rgb_abs_mean"] = rgb_abs.mean()
            metrics_dict["backscatter_closure_rgb_abs_max"] = rgb_abs.max()
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
        j_green_dominance = j_metric[..., 1] - torch.maximum(j_metric[..., 0], j_metric[..., 2])
        metrics_dict["J_green_dominance_ratio"] = (j_green_dominance > 0.05).float().mean()
        j_blue_dominance = j_metric[..., 2] - torch.maximum(j_metric[..., 0], j_metric[..., 1])
        metrics_dict["J_blue_dominance_ratio"] = (j_blue_dominance > 0.05).float().mean()
        rgb_clear_metric = torch.clamp(outputs["rgb_clear"], 0.0, 1.0)
        metrics_dict["rgb_clear_legacy_white_ratio"] = (rgb_clear_metric > 0.95).all(dim=-1).float().mean()
        metrics_dict["rgb_clear_legacy_saturation_ratio"] = (rgb_clear_metric > 0.98).float().mean()
        if "appearance_active_sh_degree" in outputs:
            metrics_dict["appearance_active_sh_degree"] = outputs["appearance_active_sh_degree"]
        if "dual_color_active_sh_degree" in outputs:
            metrics_dict["dual_color_active_sh_degree"] = outputs["dual_color_active_sh_degree"]
        if "dual_color_j_residual_raw" in outputs:
            metrics_dict["dual_color_j_residual_abs_mean"] = outputs["dual_color_j_residual_raw"].abs().mean()
        if "dual_color_chroma_residual" in outputs:
            metrics_dict["dual_color_chroma_residual_abs_mean"] = outputs["dual_color_chroma_residual"].abs().mean()
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
        if self.last_densification_region_stats is not None:
            stats = self.last_densification_region_stats
            metrics_dict["background_gradient_fraction"] = torch.tensor(
                float(stats.get("background_gradient_fraction", 0.0)),
                device=self.device,
            )
            metrics_dict["background_split_candidate_fraction"] = torch.tensor(
                float(stats.get("background_split_candidate_fraction", 0.0)),
                device=self.device,
            )
            metrics_dict["background_duplicate_candidate_fraction"] = torch.tensor(
                float(stats.get("background_duplicate_candidate_fraction", 0.0)),
                device=self.device,
            )
            metrics_dict["background_weighted_split_candidate_fraction"] = torch.tensor(
                float(stats.get("background_weighted_split_candidate_fraction", 0.0)),
                device=self.device,
            )
            metrics_dict["background_weighted_duplicate_candidate_fraction"] = torch.tensor(
                float(stats.get("background_weighted_duplicate_candidate_fraction", 0.0)),
                device=self.device,
            )
            metrics_dict["background_opacity_grad_abs_fraction"] = torch.tensor(
                float(stats.get("background_opacity_grad_abs_fraction", 0.0)),
                device=self.device,
            )
            metrics_dict["background_opacity_decrease_pressure_fraction"] = torch.tensor(
                float(stats.get("background_opacity_decrease_pressure_fraction", 0.0)),
                device=self.device,
            )
            metrics_dict["background_accumulation_grad_abs_fraction"] = torch.tensor(
                float(stats.get("background_accumulation_grad_abs_fraction", 0.0)),
                device=self.device,
            )
            metrics_dict["background_accumulation_decrease_pressure_fraction"] = torch.tensor(
                float(stats.get("background_accumulation_decrease_pressure_fraction", 0.0)),
                device=self.device,
            )
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
        image_mask = None

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask
            image_mask = mask.to(device=self.device, dtype=gt_img.dtype).clamp(0.0, 1.0)

        medium_supports = None
        support_route = None
        support_broad = None
        support_capacity = None
        support_chroma = None
        support_halo_base = None
        support_bootstrap = None
        needs_medium_support = any(
            [
                bool(getattr(self.config, "medium_explainability_enabled", False)),
                bool(getattr(self.config, "training_gradient_routing_enabled", False)),
                bool(getattr(self.config, "budgeted_capacity_enabled", False)),
                bool(getattr(self.config, "core_zero_capacity_enabled", False)),
                bool(getattr(self.config, "halo_capacity_enabled", False)),
                bool(getattr(self.config, "object_radiance_budget_enabled", False)),
                bool(getattr(self.config, "background_clear_chroma_use_medium_support", False)),
                float(getattr(self.config, "lambda_proxy_clear_luma", 0.0)) > 0.0,
            ]
        )
        if needs_medium_support:
            medium_explainer = outputs.get("b_inf", outputs["medium_rgb"])
            medium_supports = build_route_capacity_support(
                gt_img=gt_img,
                medium_rgb=medium_explainer,
                depth=outputs["depth"],
                gradient_tau=float(getattr(self.config, "medium_support_gradient_tau", 0.05)),
                variance_tau=float(getattr(self.config, "medium_support_variance_tau", 0.02)),
                color_tau=float(getattr(self.config, "medium_support_color_tau", 0.08)),
                luma_weight=float(getattr(self.config, "medium_support_luma_weight", 0.25)),
                far_floor=float(getattr(self.config, "medium_support_far_floor", 0.50)),
                depth_mid=float(getattr(self.config, "medium_support_depth_mid", 0.75)),
                depth_temperature=float(getattr(self.config, "medium_support_depth_temperature", 0.15)),
                use_flatness=bool(getattr(self.config, "medium_support_use_flatness", True)),
                use_medium=bool(getattr(self.config, "medium_support_use_medium", True)),
                use_far=bool(getattr(self.config, "medium_support_use_far", True)),
                use_connected=bool(getattr(self.config, "medium_support_connected_enabled", False)),
                connected_threshold=float(getattr(self.config, "medium_support_connected_threshold", 0.25)),
                connected_top_only=bool(getattr(self.config, "medium_support_connected_top_only", True)),
                connected_border=int(getattr(self.config, "medium_support_connected_border", 2)),
            )
            support_route = medium_supports.route
            support_broad = medium_supports.broad
            support_capacity_raw = medium_supports.core
            support_capacity = support_capacity_raw
            capacity_threshold = float(getattr(self.config, "medium_support_capacity_threshold", 0.0))
            if capacity_threshold > 0.0:
                threshold = min(max(capacity_threshold, 0.0), 0.999999)
                support_capacity = ((support_capacity - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
            capacity_power = float(getattr(self.config, "medium_support_capacity_power", 1.0))
            if abs(capacity_power - 1.0) > 1e-8:
                support_capacity = support_capacity.clamp_min(0.0).pow(max(capacity_power, 1e-6))
            support_capacity = support_capacity.detach()
            support_chroma = support_capacity
            support_halo_base = medium_supports.halo_base
            support_bootstrap = medium_supports.bootstrap
            if image_mask is not None:
                support_route = support_route * image_mask
                support_broad = support_broad * image_mask
                support_capacity = support_capacity * image_mask
                support_chroma = support_chroma * image_mask
                support_halo_base = support_halo_base * image_mask
                support_bootstrap = support_bootstrap * image_mask
            if bool(getattr(self.config, "medium_support_region_exclusion_enabled", False)):
                exclusion = torch.zeros_like(support_capacity)
                if bool(getattr(self.config, "medium_support_exclude_object", True)):
                    object_mask = self._load_backscatter_region_mask(outputs=outputs, key="object", target=gt_img)
                    if object_mask is not None:
                        exclusion = torch.maximum(exclusion, object_mask.to(support_capacity).clamp(0.0, 1.0))
                if bool(getattr(self.config, "medium_support_exclude_boundary", False)):
                    boundary_mask = self._load_backscatter_region_mask(outputs=outputs, key="boundary", target=gt_img)
                    if boundary_mask is not None:
                        exclusion = torch.maximum(exclusion, boundary_mask.to(support_capacity).clamp(0.0, 1.0))
                keep = (1.0 - exclusion).clamp(0.0, 1.0)
                if bool(getattr(self.config, "medium_support_region_exclusion_apply_capacity", True)):
                    support_capacity = support_capacity * keep
                if bool(getattr(self.config, "medium_support_region_exclusion_apply_chroma", True)):
                    support_chroma = support_chroma * keep
                outputs["medium_support_region_exclusion"] = exclusion.detach()
            outputs["medium_support_flat"] = medium_supports.flat
            outputs["medium_support_med"] = medium_supports.medium
            outputs["medium_support_far"] = medium_supports.far
            outputs["medium_support_connected"] = medium_supports.connected
            outputs["medium_support_route"] = support_route
            outputs["medium_support_broad"] = support_broad
            outputs["medium_support_core_raw"] = support_capacity_raw
            outputs["medium_support_core"] = support_capacity
            outputs["medium_support_capacity"] = support_capacity
            outputs["medium_support_chroma"] = support_chroma
            outputs["medium_support_halo_base"] = support_halo_base
            outputs["medium_support_bootstrap"] = support_bootstrap
            outputs["medium_support_error"] = medium_supports.medium_error
            if metrics_dict is not None:
                for stat_name, stat_value in support_coverage_stats(medium_supports).items():
                    metrics_dict[f"medium_support_{stat_name}"] = stat_value

        pred_img_for_loss = pred_img
        route_progress = self._ramped_weight(
            1.0,
            int(getattr(self.config, "gradient_routing_start_step", 4000)),
            int(getattr(self.config, "gradient_routing_ramp_steps", 1000)),
        )
        if (
            bool(getattr(self.config, "training_gradient_routing_enabled", False))
            and support_route is not None
            and route_progress > 0.0
        ):
            target_min_scene = float(getattr(self.config, "gradient_routing_min_scene_weight", 0.30))
            effective_min_scene = 1.0 + (target_min_scene - 1.0) * route_progress
            pred_img_for_loss = build_training_routed_prediction(
                pred_img=pred_img,
                medium_rgb=outputs.get("b_inf", outputs["medium_rgb"]),
                route_support=support_route,
                min_scene_weight=effective_min_scene,
            )
            outputs["medium_training_routed_rgb"] = pred_img_for_loss
            if metrics_dict is not None:
                metrics_dict["medium_gradient_routing_progress"] = torch.tensor(route_progress, device=self.device)
                metrics_dict["medium_gradient_routing_min_scene_weight"] = torch.tensor(
                    effective_min_scene,
                    device=self.device,
                )

        loss_dict = {
            "main_loss": reconstruction_loss(
                gt_img=gt_img,
                pred_img=pred_img_for_loss,
                main_loss=self.config.main_loss,
                ssim_loss=self.config.ssim_loss,
                ssim_lambda=self.config.ssim_lambda,
                ssim_metric=self.ssim,
            ),
        }

        if self._gmvc_requested():
            gmvc_bank = self._load_gmvc_training_bank()
            if gmvc_bank is not None:
                gmvc_losses, gmvc_metrics = compute_gmvc_training_terms(
                    outputs=outputs,
                    gt_img=gt_img,
                    bank=gmvc_bank,
                    step=int(self.step),
                    config=self.config,
                    state=self._gmvc_online_state,
                    medium_query_fn=self._query_gmvc_medium_points,
                )
                if metrics_dict is not None:
                    for name, value in gmvc_metrics.items():
                        metrics_dict[name] = value
                if bool(getattr(self.config, "gmvc_enabled", False)) and not bool(
                    getattr(self.config, "gmvc_diagnostic_only", False)
                ):
                    for name, value in gmvc_losses.items():
                        if value.requires_grad or float(value.detach().item()) != 0.0:
                            loss_dict[name] = value

        medium_weight = self._ramped_weight(
            float(getattr(self.config, "lambda_medium_explainability", 0.0)),
            int(getattr(self.config, "medium_explainability_start_step", 2000)),
            int(getattr(self.config, "medium_explainability_ramp_steps", 2000)),
        )
        if (
            bool(getattr(self.config, "medium_explainability_enabled", False))
            and medium_weight > 0.0
            and support_bootstrap is not None
            and support_route is not None
        ):
            route_blend = self._ramped_weight(
                1.0,
                int(getattr(self.config, "gradient_routing_start_step", 4000)),
                int(getattr(self.config, "gradient_routing_ramp_steps", 1000)),
            )
            medium_supervision_support = ((1.0 - route_blend) * support_bootstrap + route_blend * support_route).detach()
            loss_dict["medium_explainability_loss"] = medium_weight * weighted_rgb_l1(
                outputs.get("b_inf", outputs["medium_rgb"]),
                gt_img,
                medium_supervision_support,
            )

        core_weight = self._ramped_weight(
            float(getattr(self.config, "lambda_core_zero_capacity", 0.0)),
            int(getattr(self.config, "core_zero_capacity_start_step", 1000)),
            int(getattr(self.config, "core_zero_capacity_ramp_steps", 3000)),
        )
        if (
            bool(getattr(self.config, "core_zero_capacity_enabled", False))
            and core_weight > 0.0
            and support_capacity is not None
        ):
            post_start = int(getattr(self.config, "background_clear_chroma_start_step", 10000))
            if self.step >= post_start:
                core_weight *= float(getattr(self.config, "core_zero_capacity_post_scale", 1.0))
            capacity_accumulation = outputs.get("capacity_control_accumulation", outputs["accumulation"])
            clearance_weight = None
            if bool(getattr(self.config, "core_clearance_amplifier_enabled", False)):
                clearance_weight = accumulation_clearance_amplifier(
                    accumulation=capacity_accumulation,
                    min_weight=float(getattr(self.config, "core_clearance_amplifier_min", 0.30)),
                    threshold=float(getattr(self.config, "core_clearance_amplifier_threshold", 0.20)),
                    temperature=float(getattr(self.config, "core_clearance_amplifier_temperature", 0.05)),
                )
                outputs["core_clearance_amplifier"] = clearance_weight
                if metrics_dict is not None:
                    metrics_dict["core_clearance_amplifier_mean"] = clearance_weight.mean()
            loss_dict["core_zero_capacity_loss"] = core_weight * core_zero_capacity_loss(
                accumulation=capacity_accumulation,
                support=support_capacity,
                clearance_weight=clearance_weight,
            )
            if metrics_dict is not None:
                metrics_dict["core_zero_capacity_weight"] = torch.tensor(core_weight, device=self.device)

        cap_weight = self._ramped_weight(
            float(getattr(self.config, "lambda_budgeted_capacity", 0.0)),
            int(getattr(self.config, "budgeted_capacity_start_step", 4000)),
            int(getattr(self.config, "budgeted_capacity_ramp_steps", 1000)),
        )
        if (
            bool(getattr(self.config, "budgeted_capacity_enabled", False))
            and cap_weight > 0.0
            and support_capacity is not None
        ):
            post_start = int(getattr(self.config, "background_clear_chroma_start_step", 10000))
            if self.step >= post_start:
                cap_weight *= float(getattr(self.config, "budgeted_capacity_post_scale", 0.5))
            capacity_accumulation = outputs.get("capacity_control_accumulation", outputs["accumulation"])
            capacity_opacities = outputs.get("capacity_control_opacities")
            if (
                bool(getattr(self.config, "capacity_conflict_gate_enabled", False))
                and capacity_opacities is not None
                and getattr(capacity_opacities, "requires_grad", False)
                and loss_dict["main_loss"].requires_grad
            ):
                # The gsplat backward writes screen-space absolute gradients as
                # a side effect. This extra diagnostic grad is only used to
                # choose the opacity gate, so record the prepass contribution
                # and subtract it later from densification statistics.
                xys_grad_abs_before = (
                    self.xys_grad_abs.detach().clone()
                    if self.xys_grad_abs is not None
                    else None
                )
                rec_opacity_grad = torch.autograd.grad(
                    loss_dict["main_loss"],
                    self.opacities,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if xys_grad_abs_before is not None and self.xys_grad_abs is not None:
                    self._capacity_conflict_xys_grad_abs_prepass = (
                        self.xys_grad_abs.detach() - xys_grad_abs_before
                    ).clamp_min(0.0)
                if rec_opacity_grad is not None and tuple(rec_opacity_grad.shape) == tuple(capacity_opacities.shape):
                    rho = min(max(float(getattr(self.config, "capacity_conflict_rho", 1.0)), 0.0), 1.0)
                    rec_threshold = max(float(getattr(self.config, "capacity_conflict_rec_grad_threshold", 1e-10)), 0.0)
                    rec_conflict = (rec_opacity_grad.detach() < -rec_threshold)

                    def _capacity_conflict_hook(
                        grad: torch.Tensor,
                        rec_conflict: torch.Tensor = rec_conflict,
                        rho: float = rho,
                    ) -> torch.Tensor:
                        conflict = (grad > 0.0) & rec_conflict.to(device=grad.device)
                        return torch.where(conflict, rho * grad, grad)

                    capacity_opacities.register_hook(_capacity_conflict_hook)
                    outputs["capacity_conflict_rec_mask"] = rec_conflict.detach().to(dtype=capacity_opacities.dtype)
                    if metrics_dict is not None:
                        metrics_dict["capacity_conflict_rec_fraction"] = rec_conflict.float().mean()
                        metrics_dict["capacity_conflict_rho"] = torch.tensor(rho, device=self.device)
            loss_dict["budgeted_capacity_loss"] = cap_weight * budgeted_capacity_loss(
                accumulation=capacity_accumulation,
                support=support_capacity,
                budget=float(getattr(self.config, "budgeted_capacity_value", 0.05)),
                temperature=float(getattr(self.config, "budgeted_capacity_temperature", 0.02)),
            )

        halo_weight = self._ramped_weight(
            float(getattr(self.config, "lambda_halo_capacity", 0.0)),
            int(getattr(self.config, "halo_capacity_start_step", 4000)),
            int(getattr(self.config, "halo_capacity_ramp_steps", 1000)),
        )
        if (
            bool(getattr(self.config, "halo_capacity_enabled", False))
            and halo_weight > 0.0
            and support_broad is not None
            and support_capacity is not None
            and "J_proxy_raw" in outputs
        ):
            halo_support = build_residual_gated_halo_support(
                j_proxy=outputs["J_proxy_raw"],
                medium_rgb=outputs.get("b_inf", outputs["medium_rgb"]),
                broad_support=support_broad,
                core_support=support_capacity,
                chroma_margin=float(getattr(self.config, "halo_chroma_margin", 0.015)),
                chroma_temperature=float(getattr(self.config, "halo_chroma_temperature", 0.01)),
                luma_min=float(getattr(self.config, "halo_luma_min", 0.02)),
                luma_temperature=float(getattr(self.config, "halo_luma_temperature", 0.01)),
            )
            if image_mask is not None:
                halo_support = halo_support * image_mask
            post_start = int(getattr(self.config, "background_clear_chroma_start_step", 10000))
            if self.step >= post_start:
                halo_weight *= float(getattr(self.config, "halo_capacity_post_scale", 0.5))
            outputs["medium_support_halo"] = halo_support
            capacity_accumulation = outputs.get("capacity_control_accumulation", outputs["accumulation"])
            loss_dict["halo_capacity_loss"] = halo_weight * budgeted_capacity_loss(
                accumulation=capacity_accumulation,
                support=halo_support,
                budget=float(getattr(self.config, "halo_capacity_value", 0.03)),
                temperature=float(getattr(self.config, "halo_capacity_temperature", 0.02)),
            )
            if metrics_dict is not None:
                metrics_dict["medium_support_halo_mean"] = halo_support.mean()
                metrics_dict["medium_support_halo_gt_0p25_fraction"] = (halo_support > 0.25).float().mean()

        bg_weight = self._backscatter_ramp_weight(getattr(self.config, "lambda_background_water_color", 0.0))
        if bg_weight > 0.0 and "b_inf" in outputs:
            bg_mask = self._load_backscatter_region_mask(
                outputs=outputs,
                key=getattr(self.config, "background_water_mask_key", "water"),
                target=gt_img,
            )
            if bg_mask is not None and bg_mask.sum() > 0:
                loss_dict["background_water_color_loss"] = bg_weight * (
                    bg_mask * torch.abs(outputs["b_inf"] - gt_img)
                ).sum() / (bg_mask.sum().clamp_min(1e-6) * 3.0)

        bg_medium_weight = self._background_render_ramp_weight(
            getattr(self.config, "lambda_background_medium_render", 0.0)
        )
        bg_tail_weight = self._background_render_ramp_weight(
            getattr(self.config, "lambda_background_tail_render", 0.0)
        )
        if bg_medium_weight > 0.0 or bg_tail_weight > 0.0:
            bg_mask = self._load_backscatter_region_mask(
                outputs=outputs,
                key=getattr(self.config, "background_water_mask_key", "water"),
                target=gt_img,
            )
            if bg_mask is not None and bg_mask.sum() > 0:
                if bg_medium_weight > 0.0:
                    loss_dict["background_medium_render_loss"] = bg_medium_weight * masked_rgb_l1_loss(
                        outputs["rgb_medium_total"],
                        gt_img,
                        bg_mask,
                    )
                if bg_tail_weight > 0.0:
                    loss_dict["background_tail_render_loss"] = bg_tail_weight * masked_rgb_l1_loss(
                        outputs["rgb_tail"],
                        gt_img,
                        bg_mask,
                    )

        medium_bg_weight = float(getattr(self.config, "medium_background_supervision_lambda", 0.0))
        if bool(getattr(self.config, "medium_background_supervision_enabled", False)) and medium_bg_weight > 0.0:
            bg_mask = self._load_backscatter_region_mask(
                outputs=outputs,
                key=getattr(self.config, "background_water_mask_key", "water"),
                target=gt_img,
            )
            if bg_mask is not None and bg_mask.sum() > 0:
                boundary_mask = None
                if bool(getattr(self.config, "medium_background_supervision_exclude_boundary", True)):
                    boundary_mask = self._load_backscatter_region_mask(outputs=outputs, key="boundary", target=gt_img)
                bg_medium_mask = effective_background_mask(
                    water_mask=bg_mask,
                    boundary_mask=boundary_mask,
                    hit_confidence=outputs.get("hit_confidence"),
                    hit_threshold=float(getattr(self.config, "medium_background_supervision_hit_exclusion_threshold", -1.0)),
                ).detach()
                if image_mask is not None:
                    bg_medium_mask = (bg_medium_mask * image_mask.detach()).clamp(0.0, 1.0)
                if bg_medium_mask.sum() > 0:
                    residual = torch.abs(outputs["medium_rgb"] - gt_img)
                    channel_weight = 1.0 / (outputs["medium_rgb"].detach() + 1e-3)
                    denom = bg_medium_mask.sum().clamp_min(1e-6)
                    background_medium_l1 = (bg_medium_mask * residual).sum() / denom
                    weighted_background_medium_l1 = (bg_medium_mask * channel_weight * residual).sum() / denom
                    loss_dict["medium_background_supervision_loss"] = medium_bg_weight * weighted_background_medium_l1
                    if metrics_dict is not None:
                        metrics_dict["background_medium_l1"] = background_medium_l1.detach()
                        metrics_dict["weighted_background_medium_l1"] = weighted_background_medium_l1.detach()
                        metrics_dict["medium_background_supervision_mask_coverage"] = bg_medium_mask.detach().mean()
                        metrics_dict["medium_background_supervision_lambda"] = torch.tensor(
                            medium_bg_weight,
                            device=self.device,
                        )

        bg_clear_weight = self._background_clear_ramp_weight(
            getattr(self.config, "lambda_background_clear_gaussian", 0.0)
        )
        if bg_clear_weight > 0.0:
            if not self._warned_background_clear_gaussian_dead_grad:
                CONSOLE.log(
                    "[yellow]lambda_background_clear_gaussian targets "
                    "J_gaussian_raw, whose Gaussian backward path is inactive "
                    "in the current CUDA wrapper. Prefer "
                    "lambda_background_clear_chroma with J_proxy_raw for active "
                    "clear/dewatered optimization.[/yellow]"
                )
                self._warned_background_clear_gaussian_dead_grad = True
            bg_mask = self._load_backscatter_region_mask(
                outputs=outputs,
                key=getattr(self.config, "background_water_mask_key", "water"),
                target=gt_img,
            )
            if bg_mask is not None and bg_mask.sum() > 0:
                boundary_mask = None
                if getattr(self.config, "background_clear_exclude_boundary", True):
                    boundary_mask = self._load_backscatter_region_mask(outputs=outputs, key="boundary", target=gt_img)
                bg_clear_mask = effective_background_mask(
                    water_mask=bg_mask,
                    boundary_mask=boundary_mask,
                    hit_confidence=outputs.get("hit_confidence"),
                    hit_threshold=float(getattr(self.config, "background_clear_hit_exclusion_threshold", -1.0)),
                )
                if bg_clear_mask.sum() > 0:
                    j_key = "J_gaussian_raw" if getattr(self.config, "background_clear_use_raw_j", True) else "J_gaussian"
                    loss_dict["background_clear_gaussian_loss"] = bg_clear_weight * masked_rgb_l1_loss(
                        outputs[j_key],
                        torch.zeros_like(outputs[j_key]),
                        bg_clear_mask,
                    )

        bg_chroma_weight = self._ramped_weight(
            float(getattr(self.config, "lambda_background_clear_chroma", 0.0)),
            int(getattr(self.config, "background_clear_chroma_start_step", 10000)),
            int(getattr(self.config, "background_clear_chroma_ramp_steps", 1000)),
        )
        if bg_chroma_weight > 0.0 and "J_proxy_raw" in outputs:
            if (
                bool(getattr(self.config, "background_clear_chroma_use_medium_support", False))
                and support_chroma is not None
                and support_chroma.sum() > 0
            ):
                loss_dict["background_clear_chroma_loss"] = bg_chroma_weight * clear_proxy_chroma_loss(
                    j_proxy=outputs["J_proxy_raw"],
                    medium_rgb=outputs.get("b_inf", outputs["medium_rgb"]),
                    support=support_chroma,
                    margin=float(getattr(self.config, "background_clear_chroma_margin", 0.02)),
                    detach_medium=bool(getattr(self.config, "background_clear_chroma_medium_detach", True)),
                )
            else:
                bg_mask = self._load_backscatter_region_mask(
                    outputs=outputs,
                    key=getattr(self.config, "background_water_mask_key", "water"),
                    target=gt_img,
                )
                if bg_mask is not None and bg_mask.sum() > 0:
                    acc_max = float(getattr(self.config, "background_clear_chroma_accumulation_max", 0.65))
                    acc_temp = max(float(getattr(self.config, "background_clear_chroma_accumulation_temperature", 0.05)), 1e-6)
                    margin = float(getattr(self.config, "background_clear_chroma_margin", 0.02))
                    acc_gate = torch.sigmoid((acc_max - outputs["accumulation"].detach()) / acc_temp).clamp(0.0, 1.0)
                    j_proxy = outputs["J_proxy_raw"]
                    medium_chroma_source = outputs["medium_rgb"]
                    if getattr(self.config, "background_clear_chroma_medium_detach", True):
                        medium_chroma_source = medium_chroma_source.detach()
                    j_chroma = j_proxy - j_proxy.mean(dim=-1, keepdim=True)
                    medium_chroma = medium_chroma_source - medium_chroma_source.mean(dim=-1, keepdim=True)
                    medium_dir = medium_chroma / medium_chroma.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    projection = (j_chroma * medium_dir).sum(dim=-1, keepdim=True)
                    penalty = F.relu(projection - margin)
                    bg_chroma_mask = bg_mask * acc_gate
                    if bg_chroma_mask.sum() > 0:
                        loss_dict["background_clear_chroma_loss"] = bg_chroma_weight * (
                            bg_chroma_mask * penalty
                        ).sum() / bg_chroma_mask.sum().clamp_min(1e-6)

        proxy_luma_weight = self._ramped_weight(
            float(getattr(self.config, "lambda_proxy_clear_luma", 0.0)),
            int(getattr(self.config, "background_clear_chroma_start_step", 10000)),
            int(getattr(self.config, "background_clear_chroma_ramp_steps", 1000)),
        )
        if proxy_luma_weight > 0.0 and "J_proxy_raw" in outputs and support_capacity is not None:
            loss_dict["proxy_clear_luma_budget_loss"] = proxy_luma_weight * clear_proxy_luma_budget_loss(
                j_proxy=outputs["J_proxy_raw"],
                support=support_capacity,
                budget=float(getattr(self.config, "proxy_clear_luma_budget", 0.03)),
                temperature=float(getattr(self.config, "proxy_clear_luma_temperature", 0.01)),
            )

        object_radiance_weight = self._ramped_weight(
            float(getattr(self.config, "lambda_object_radiance_budget", 0.0)),
            int(getattr(self.config, "object_radiance_budget_start_step", 10000)),
            int(getattr(self.config, "object_radiance_budget_ramp_steps", 1000)),
        )
        if (
            bool(getattr(self.config, "object_radiance_budget_enabled", False))
            and object_radiance_weight > 0.0
            and support_capacity is not None
        ):
            loss_dict["object_radiance_budget_loss"] = object_radiance_weight * rgb_luma_budget_loss(
                rgb=outputs["rgb_object"],
                support=support_capacity,
                budget=float(getattr(self.config, "object_radiance_budget_value", 0.015)),
                temperature=float(getattr(self.config, "object_radiance_budget_temperature", 0.01)),
            )
            if metrics_dict is not None:
                metrics_dict["object_radiance_budget_weight"] = torch.tensor(object_radiance_weight, device=self.device)

        fg_weight = self._backscatter_ramp_weight(
            getattr(self.config, "lambda_foreground_transmission_reconstruction", 0.0)
        )
        if fg_weight > 0.0:
            fg_mask = self._load_backscatter_region_mask(
                outputs=outputs,
                key=getattr(self.config, "foreground_water_mask_key", "object"),
                target=gt_img,
            )
            if fg_mask is not None and fg_mask.sum() > 0:
                transmission = torch.exp(-(outputs["medium_attn"] * outputs["depth"]).clamp_min(0.0)).clamp(0.0, 1.0)
                gamma = float(getattr(self.config, "foreground_transmission_gamma", 1.0))
                max_weight = float(getattr(self.config, "foreground_transmission_max_weight", 4.0))
                weights = 1.0 + fg_weight * fg_mask * torch.pow((1.0 - transmission).clamp(0.0, 1.0), gamma)
                weights = weights.clamp(1.0, max_weight)
                if getattr(self.config, "foreground_transmission_detach_weight", True):
                    weights = weights.detach()
                loss_dict["foreground_transmission_reconstruction_loss"] = (
                    (weights - 1.0) * torch.abs(pred_img - gt_img)
                ).mean()

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

        if getattr(self.config, "dual_color_enabled", False) and "dual_color_visible_mask" in outputs:
            near_weight = self._dual_color_ramp_weight(self.config.lambda_intrinsic_near_anchor)
            if near_weight > 0.0 and "dual_color_j_residual_raw" in outputs:
                mean_attn = outputs["medium_attn"].mean(dim=-1, keepdim=True)
                transmission = torch.exp(-(mean_attn * outputs["depth"]).clamp_min(0.0)).clamp(0.0, 1.0)
                threshold = float(getattr(self.config, "dual_color_near_transmission_threshold", 0.70))
                temp = max(float(getattr(self.config, "dual_color_near_transmission_temp", 0.10)), 1e-6)
                near_gate = torch.sigmoid((transmission - threshold) / temp).clamp(0.0, 1.0)
                loss_dict["dual_color_intrinsic_near_anchor_loss"] = near_weight * (
                    near_gate * outputs["dual_color_j_residual_raw"].abs()
                ).sum() / (near_gate.sum().clamp_min(1e-6) * 3.0)

            visible_mask = outputs["dual_color_visible_mask"].reshape(-1)
            mean_weight = self._dual_color_ramp_weight(self.config.lambda_view_residual_mean)
            if mean_weight > 0.0 and "dual_color_view_residual" in outputs and visible_mask.any():
                residual = outputs["dual_color_view_residual"][visible_mask]
                loss_dict["dual_color_view_residual_mean_loss"] = mean_weight * residual.mean(dim=0).abs().mean()

            chroma_weight = self._dual_color_ramp_weight(self.config.lambda_clear_chroma)
            if chroma_weight > 0.0 and "dual_color_chroma_residual" in outputs and visible_mask.any():
                chroma = outputs["dual_color_chroma_residual"][visible_mask]
                loss_dict["dual_color_clear_chroma_loss"] = chroma_weight * chroma.abs().mean()

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

        self._maybe_log_gmvc_grad_stats(loss_dict, metrics_dict)

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
        j_green_dominance = j_metric[..., 1] - torch.maximum(j_metric[..., 0], j_metric[..., 2])
        metrics_dict["J_green_dominance_ratio"] = float((j_green_dominance > 0.05).float().mean().item())
        j_blue_dominance = j_metric[..., 2] - torch.maximum(j_metric[..., 0], j_metric[..., 1])
        metrics_dict["J_blue_dominance_ratio"] = float((j_blue_dominance > 0.05).float().mean().item())

        images_dict = {
            "gt": output_gt_rgb,
            "rgb_medium": outputs["rgb_medium"],
            "rgb_medium_finite": outputs["rgb_medium_finite"].clamp(0.0, 1.0),
            "rgb_medium_total": outputs["rgb_medium_total"].clamp(0.0, 1.0),
            "rgb_tail": outputs["rgb_tail"].clamp(0.0, 1.0),
            "rgb_object": outputs["rgb_object"],
            "depth": outputs["depth"],
            "accumulation": outputs["accumulation"].expand_as(outputs["rgb"]),
            "rgb": outputs["rgb"],
            "J": outputs["J"],
            "J_raw": outputs["J_raw"].clamp(0.0, 1.0),
            "J_gaussian": outputs["J_gaussian"],
            "J_gaussian_raw": outputs["J_gaussian_raw"].clamp(0.0, 1.0),
            "J_object": outputs["J_object"],
            "rgb_clear_legacy": outputs["rgb_clear"],
        }
        if "J_proxy_raw" in outputs:
            images_dict["J_proxy"] = outputs["J_proxy"]
            images_dict["J_proxy_raw"] = outputs["J_proxy_raw"].clamp(0.0, 1.0)
            images_dict["J_proxy_abs_diff_from_renderer_clear"] = outputs[
                "J_proxy_abs_diff_from_renderer_clear"
            ].clamp(0.0, 1.0)
        if "background_region_mask" in outputs:
            images_dict["background_region_mask"] = outputs["background_region_mask"].expand_as(outputs["rgb"])
        if "densification_region_weight" in outputs:
            images_dict["densification_region_weight"] = outputs["densification_region_weight"].expand_as(outputs["rgb"])
        if "J_intrinsic" in outputs:
            images_dict["J_intrinsic"] = outputs["J_intrinsic"]
            images_dict["rgb_object_intrinsic"] = outputs["rgb_object_intrinsic"]
            images_dict["dual_color_j_residual_abs"] = outputs["dual_color_j_residual_raw"].abs().clamp(0.0, 1.0)
        if "b_inf" in outputs:
            images_dict["b_inf"] = outputs["b_inf"]
            images_dict["b_inf_minus_A_abs"] = outputs["b_inf_minus_A_abs"].clamp(0.0, 1.0)
        if "m_inf" in outputs:
            images_dict["m_inf"] = outputs["m_inf"].expand_as(outputs["rgb"])
            images_dict["m_inf_eff"] = outputs["m_inf_eff"].expand_as(outputs["rgb"])
        return metrics_dict, images_dict
