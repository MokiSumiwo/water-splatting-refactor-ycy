"""Geometry-anchored track construction for GMVC diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Tuple

import torch
import torch.nn.functional as F

from nerfstudio.cameras.cameras import Cameras

from .gmvc_losses import invert_intrinsic_radiance
from .gmvc_types import GMVCTrackConfig, GMVCView


R_EDIT = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))


def _as_hwc(value: torch.Tensor, key: str, channels: int | None = None) -> torch.Tensor:
    tensor = value.detach().float().cpu()
    if tensor.ndim == 2:
        tensor = tensor[..., None]
    if tensor.ndim != 3:
        raise ValueError(f"{key} must be HxW or HxWxC, got {tuple(tensor.shape)}")
    if channels is not None and tensor.shape[-1] != channels:
        raise ValueError(f"{key} must have {channels} channels, got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _camera_index(camera: Cameras, outputs: Dict[str, torch.Tensor], fallback: int) -> int:
    if "camera_index" in outputs:
        return int(outputs["camera_index"].detach().cpu().reshape(-1)[0].item())
    if camera.metadata is not None and "cam_idx" in camera.metadata:
        value = camera.metadata["cam_idx"]
        if torch.is_tensor(value):
            return int(value.detach().cpu().reshape(-1)[0].item())
        return int(value)
    return int(fallback)


def _camera_items(pipeline: Any, split: str, max_images: int, device: torch.device) -> Iterator[Tuple[int, Any, Dict]]:
    max_count = max_images if max_images > 0 else 10**9
    if split == "eval":
        for image_idx, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):
            if image_idx >= max_count:
                break
            yield image_idx, camera.to(device) if hasattr(camera, "to") else camera, batch
        return

    dataset = pipeline.datamanager.train_dataset
    count = min(len(dataset.cameras), max_count)
    for image_idx in range(count):
        camera = dataset.cameras[image_idx : image_idx + 1]
        image = dataset[image_idx]["image"]
        yield image_idx, camera.to(device) if hasattr(camera, "to") else camera, {"image": image}


def render_gmvc_views(pipeline: Any, split: str, max_images: int) -> List[GMVCView]:
    """Render train/eval cameras and collect the tensors used by GMVC diagnostics."""

    pipeline.eval()
    model = pipeline.model
    device = model.device
    views: List[GMVCView] = []
    with torch.no_grad():
        for image_idx, camera, batch in _camera_items(pipeline, split, max_images, device):
            outputs = model.get_outputs(camera)
            gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
            depth = _as_hwc(outputs["depth"], "depth")
            if depth.shape[-1] != 1:
                depth = depth.mean(dim=-1, keepdim=True)
            medium_attn = _as_hwc(outputs["medium_attn"], "medium_attn", channels=3)
            medium_bs = _as_hwc(outputs["medium_bs"], "medium_bs", channels=3)
            b_inf = _as_hwc(outputs.get("b_inf", outputs["medium_rgb"]), "b_inf", channels=3)
            transmission = torch.exp(-(medium_attn * depth).clamp_min(0.0))
            backscatter_endpoint = b_inf * (1.0 - torch.exp(-(medium_bs * depth).clamp_min(0.0)))
            views.append(
                GMVCView(
                    image_index=image_idx,
                    camera_index=_camera_index(camera, outputs, image_idx),
                    camera_to_world=camera.camera_to_worlds[0].detach().float().cpu(),
                    fx=float(camera.fx.item()),
                    fy=float(camera.fy.item()),
                    cx=float(camera.cx.item()),
                    cy=float(camera.cy.item()),
                    width=int(camera.width.item()),
                    height=int(camera.height.item()),
                    gt=_as_hwc(gt, "gt", channels=3),
                    depth=depth,
                    accumulation=_as_hwc(outputs["accumulation"], "accumulation"),
                    depth_std_relative=_as_hwc(outputs["depth_std_relative"], "depth_std_relative"),
                    medium_bs=medium_bs,
                    medium_attn=medium_attn,
                    b_inf=b_inf,
                    transmission=transmission.contiguous(),
                    backscatter_endpoint=backscatter_endpoint.contiguous(),
                    actual_rgb_medium=_as_hwc(outputs["rgb_medium"], "actual_rgb_medium", channels=3),
                )
            )
    return views


def _view_to_world_rotation(view: GMVCView) -> torch.Tensor:
    return view.camera_to_world[:3, :3] @ R_EDIT.to(view.camera_to_world)


def unproject_pixels(view: GMVCView, xy: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    depth = depth.reshape(-1)
    p_view = torch.stack(
        [
            (xy[:, 0] - view.cx) / view.fx * depth,
            (xy[:, 1] - view.cy) / view.fy * depth,
            depth,
        ],
        dim=-1,
    )
    rotation = _view_to_world_rotation(view)
    translation = view.camera_to_world[:3, 3]
    return p_view @ rotation.T + translation


def project_world(view: GMVCView, points_world: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    rotation = _view_to_world_rotation(view)
    translation = view.camera_to_world[:3, 3]
    p_view = (points_world - translation) @ rotation
    z = p_view[:, 2].clamp_min(float(eps))
    xy = torch.stack(
        [
            view.fx * p_view[:, 0] / z + view.cx,
            view.fy * p_view[:, 1] / z + view.cy,
        ],
        dim=-1,
    )
    return xy, p_view[:, 2]


def _sample_hwc(image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    if xy.numel() == 0:
        return torch.empty((0, image.shape[-1]), dtype=image.dtype)
    h, w = image.shape[:2]
    grid_x = 2.0 * xy[:, 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * xy[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    nchw = image.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(nchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled[0, :, :, 0].T.contiguous()


def _valid_source_mask(view: GMVCView, cfg: GMVCTrackConfig) -> torch.Tensor:
    h, w = view.height, view.width
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    edge = (
        (xx >= cfg.edge_margin)
        & (xx < w - cfg.edge_margin)
        & (yy >= cfg.edge_margin)
        & (yy < h - cfg.edge_margin)
    )
    t_mean = view.transmission.mean(dim=-1, keepdim=True)
    valid = (
        torch.isfinite(view.depth)
        & (view.depth > 0)
        & (view.accumulation >= cfg.alpha_threshold)
        & (view.depth_std_relative <= cfg.depth_std_rel_threshold)
        & (t_mean >= cfg.transmission_min)
        & edge[..., None]
    )
    return valid[..., 0]


def _sample_source_pixels(view: GMVCView, cfg: GMVCTrackConfig, source_offset: int) -> Tuple[torch.Tensor, int]:
    valid = _valid_source_mask(view, cfg)
    coords_yx = torch.nonzero(valid, as_tuple=False)
    total = int(coords_yx.shape[0])
    if total == 0:
        return torch.empty((0, 2), dtype=torch.float32), 0
    max_samples = int(cfg.samples_per_view)
    if max_samples > 0 and total > max_samples:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(cfg.seed) + int(source_offset) * 1009)
        keep = torch.randperm(total, generator=generator)[:max_samples]
        coords_yx = coords_yx[keep]
    xy = torch.stack([coords_yx[:, 1].float(), coords_yx[:, 0].float()], dim=-1)
    return xy, total


def _sample_observations(view: GMVCView, xy: torch.Tensor, depth_error: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "gt": _sample_hwc(view.gt, xy),
        "depth": _sample_hwc(view.depth, xy),
        "alpha": _sample_hwc(view.accumulation, xy),
        "depth_std_relative": _sample_hwc(view.depth_std_relative, xy),
        "medium_bs": _sample_hwc(view.medium_bs, xy),
        "medium_attn": _sample_hwc(view.medium_attn, xy),
        "b_inf": _sample_hwc(view.b_inf, xy),
        "transmission": _sample_hwc(view.transmission, xy),
        "backscatter_endpoint": _sample_hwc(view.backscatter_endpoint, xy),
        "actual_rgb_medium": _sample_hwc(view.actual_rgb_medium, xy),
        "depth_rel_error": depth_error.reshape(-1, 1).float(),
    }


def _append_observation(obs_lists: List[List[Dict[str, torch.Tensor]]], local_idx: int, obs: Dict[str, torch.Tensor], row_idx: int) -> None:
    obs_lists[local_idx].append({key: value[row_idx].detach().cpu() for key, value in obs.items()})


def _selected_target_indices(source_idx: int, view_count: int, cfg: GMVCTrackConfig) -> List[int]:
    window = int(cfg.target_neighbor_window)
    if window <= 0:
        return [idx for idx in range(view_count) if idx != source_idx]
    lo = max(0, source_idx - window)
    hi = min(view_count, source_idx + window + 1)
    return [idx for idx in range(lo, hi) if idx != source_idx]


def _weighted_center(values: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    return (values * weights[:, None]).sum(dim=0) / (weights.sum() + float(eps))


def _weighted_l1(values: torch.Tensor, center: torch.Tensor, weights: torch.Tensor, eps: float) -> float:
    residual = (values - center[None]).abs().mean(dim=-1)
    return float((residual * weights).sum().item() / (weights.sum().item() + float(eps)))


def _track_row(observations: List[Dict[str, torch.Tensor]], cfg: GMVCTrackConfig) -> Dict[str, Any]:
    depth = torch.stack([obs["depth"].reshape(()) for obs in observations]).float()
    alpha = torch.stack([obs["alpha"].reshape(()) for obs in observations]).float()
    depth_err = torch.stack([obs["depth_rel_error"].reshape(()) for obs in observations]).float()
    transmission = torch.stack([obs["transmission"].float() for obs in observations])
    t_scalar = transmission.mean(dim=-1)
    gt = torch.stack([obs["gt"].float() for obs in observations])
    medium_attn = torch.stack([obs["medium_attn"].float() for obs in observations])
    medium_bs = torch.stack([obs["medium_bs"].float() for obs in observations])
    b_inf = torch.stack([obs["b_inf"].float() for obs in observations])
    endpoint = torch.stack([obs["backscatter_endpoint"].float() for obs in observations])
    actual = torch.stack([obs["actual_rgb_medium"].float() for obs in observations])

    j_hat = invert_intrinsic_radiance(gt, depth[:, None], medium_attn, medium_bs, b_inf, eps=cfg.eps)
    j_valid = (
        torch.isfinite(j_hat).all(dim=-1)
        & (j_hat >= cfg.j_clamp_min).all(dim=-1)
        & (j_hat <= cfg.j_clamp_max).all(dim=-1)
    )
    span = float((depth.max() - depth.min()).item())
    median_depth = depth.median().clamp_min(float(cfg.eps))
    relative_span = float((span / median_depth).item())

    w_alpha = ((alpha - cfg.alpha_threshold) / max(1.0 - cfg.alpha_threshold, cfg.eps)).clamp(0.0, 1.0)
    w_depth = torch.exp(-depth_err / max(cfg.depth_error_sigma, cfg.eps)).clamp(0.0, 1.0)
    w_t = ((t_scalar - cfg.transmission_min) / max(1.0 - cfg.transmission_min, cfg.eps)).clamp(0.0, 1.0)
    w_span = min(max(relative_span / max(cfg.span_weight_high, cfg.eps), 0.0), 1.0)
    weights = (w_alpha * w_depth * w_t * w_span).float()
    weights = torch.where(j_valid, weights, torch.zeros_like(weights))

    if int((weights > 0).sum().item()) == 0:
        j_consistency = 0.0
        attn_l1 = 0.0
        bs_l1 = 0.0
        b_inf_l1 = 0.0
    else:
        j_center = _weighted_center(j_hat, weights, cfg.eps)
        attn_center = _weighted_center(medium_attn, weights, cfg.eps)
        bs_center = _weighted_center(medium_bs, weights, cfg.eps)
        b_inf_center = _weighted_center(b_inf, weights, cfg.eps)
        j_consistency = _weighted_l1(j_hat, j_center, weights, cfg.eps)
        attn_l1 = _weighted_l1(medium_attn, attn_center, weights, cfg.eps)
        bs_l1 = _weighted_l1(medium_bs, bs_center, weights, cfg.eps)
        b_inf_l1 = _weighted_l1(b_inf, b_inf_center, weights, cfg.eps)

    attn_scalar = medium_attn.mean(dim=-1)
    bs_scalar = medium_bs.mean(dim=-1)
    b_inf_scalar = b_inf.mean(dim=-1)
    return {
        "track_length": int(len(observations)),
        "valid_j_observation_count": int((weights > 0).sum().item()),
        "relative_depth_span": relative_span,
        "alpha_mean": float(alpha.mean().item()),
        "depth_consistency_error_mean": float(depth_err.mean().item()),
        "transmission_mean": float(t_scalar.mean().item()),
        "transmission_p05": float(t_scalar.kthvalue(max(1, int(0.05 * t_scalar.numel()))).values.item()),
        "j_consistency_l1": j_consistency,
        "medium_attn_track_l1": attn_l1,
        "medium_bs_track_l1": bs_l1,
        "b_inf_track_l1": b_inf_l1,
        "endpoint_actual_l1": float((endpoint - actual).abs().mean().item()),
        "invalid_j_ratio": float((~j_valid).float().mean().item()),
        "attn_delta_scalar": (attn_scalar - attn_scalar.mean()).tolist(),
        "bs_delta_scalar": (bs_scalar - bs_scalar.mean()).tolist(),
        "b_inf_delta_scalar": (b_inf_scalar - b_inf_scalar.mean()).tolist(),
    }


def _view_parameter_row(view: GMVCView, cfg: GMVCTrackConfig) -> Dict[str, Any]:
    mask = _valid_source_mask(view, cfg)
    if not mask.any():
        mask = torch.ones((view.height, view.width), dtype=torch.bool)

    def mean_rgb(tensor: torch.Tensor) -> List[float]:
        vals = tensor[mask]
        return [float(vals[:, idx].mean().item()) for idx in range(vals.shape[-1])]

    return {
        "image_index": view.image_index,
        "camera_index": view.camera_index,
        "valid_pixel_fraction": float(mask.float().mean().item()),
        "medium_attn_mean_rgb": mean_rgb(view.medium_attn),
        "medium_bs_mean_rgb": mean_rgb(view.medium_bs),
        "b_inf_mean_rgb": mean_rgb(view.b_inf),
    }


def build_gmvc_track_metrics(views: List[GMVCView], cfg: GMVCTrackConfig) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]]]:
    """Build GMVC track metrics from rendered views."""

    counters: Dict[str, int] = {
        "source_valid_pixels_total": 0,
        "sampled_source_tracks": 0,
        "target_projection_attempts": 0,
        "invalid_out_of_bounds_count": 0,
        "invalid_depth_count": 0,
        "invalid_alpha_count": 0,
        "invalid_depth_std_count": 0,
        "invalid_low_T_count": 0,
        "target_valid_after_depth_std": 0,
        "target_valid_after_low_T": 0,
    }
    rows: List[Dict[str, Any]] = []
    for source_idx, source_view in enumerate(views):
        source_xy, source_valid_count = _sample_source_pixels(source_view, cfg, source_idx)
        counters["source_valid_pixels_total"] += source_valid_count
        counters["sampled_source_tracks"] += int(source_xy.shape[0])
        if source_xy.numel() == 0:
            continue
        source_depth = _sample_hwc(source_view.depth, source_xy).reshape(-1)
        points_world = unproject_pixels(source_view, source_xy, source_depth)
        source_obs = _sample_observations(source_view, source_xy, torch.zeros_like(source_depth))
        obs_lists: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(source_xy.shape[0])]
        for local_idx in range(source_xy.shape[0]):
            _append_observation(obs_lists, local_idx, source_obs, local_idx)

        for target_idx in _selected_target_indices(source_idx, len(views), cfg):
            target_view = views[target_idx]
            xy_target, z_projected = project_world(target_view, points_world, eps=cfg.eps)
            in_bounds = (
                (z_projected > cfg.eps)
                & (xy_target[:, 0] >= cfg.edge_margin)
                & (xy_target[:, 0] < target_view.width - cfg.edge_margin)
                & (xy_target[:, 1] >= cfg.edge_margin)
                & (xy_target[:, 1] < target_view.height - cfg.edge_margin)
            )
            counters["target_projection_attempts"] += int(xy_target.shape[0])
            counters["invalid_out_of_bounds_count"] += int((~in_bounds).sum().item())
            if not in_bounds.any():
                continue

            local_indices = torch.nonzero(in_bounds, as_tuple=False).reshape(-1)
            target_xy = xy_target[local_indices]
            projected_depth = z_projected[local_indices]
            target_depth = _sample_hwc(target_view.depth, target_xy).reshape(-1)
            depth_rel_error = (target_depth - projected_depth).abs() / target_depth.clamp_min(cfg.eps)
            target_obs = _sample_observations(target_view, target_xy, depth_rel_error)
            alpha = target_obs["alpha"].reshape(-1)
            depth_std = target_obs["depth_std_relative"].reshape(-1)
            t_mean = target_obs["transmission"].mean(dim=-1)
            depth_valid = torch.isfinite(target_depth) & (target_depth > 0) & (depth_rel_error <= cfg.depth_rel_threshold)
            alpha_valid = alpha >= cfg.alpha_threshold
            depth_std_valid = depth_std <= cfg.depth_std_rel_threshold
            t_valid = t_mean >= cfg.transmission_min

            counters["invalid_depth_count"] += int((~depth_valid).sum().item())
            after_depth = depth_valid
            counters["invalid_alpha_count"] += int((after_depth & ~alpha_valid).sum().item())
            after_alpha = after_depth & alpha_valid
            counters["invalid_depth_std_count"] += int((after_alpha & ~depth_std_valid).sum().item())
            after_depth_std = after_alpha & depth_std_valid
            counters["target_valid_after_depth_std"] += int(after_depth_std.sum().item())
            counters["invalid_low_T_count"] += int((after_depth_std & ~t_valid).sum().item())
            final_valid = after_depth_std & t_valid
            counters["target_valid_after_low_T"] += int(final_valid.sum().item())

            valid_rows = torch.nonzero(final_valid, as_tuple=False).reshape(-1)
            for row_idx in valid_rows.tolist():
                _append_observation(obs_lists, int(local_indices[row_idx].item()), target_obs, int(row_idx))

        for observations in obs_lists:
            if len(observations) >= 2:
                rows.append(_track_row(observations, cfg))

    view_rows = [_view_parameter_row(view, cfg) for view in views]
    return rows, counters, view_rows
