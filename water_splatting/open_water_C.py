"""Conservative, train-only open-water supervision for registered C arms."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


def schedule(step: int) -> float:
    """C uses the preregistered 15k A schedule and runs every fourth step."""
    if step <= 500 or step >= 10000:
        return 0.0
    if step <= 2000:
        return (step - 500) / 1500.0
    if step <= 8000:
        return 1.0
    return (10000 - step) / 2000.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(grads: Sequence[Optional[Tensor]]) -> float:
    return math.sqrt(sum(float(g.detach().float().square().sum().item()) for g in grads if g is not None))


class OpenWaterC:
    """Add separate capped gradients only to M1 medium and opacity groups."""

    def __init__(self, worker: Any, branch: Any, output: Path, mask_dir: Path,
                 arm: str, smoke: bool = False, scene: str = "IUI3-RedSea") -> None:
        if arm not in ("C1", "C2"):
            raise ValueError(f"unsupported open-water arm: {arm}")
        self.worker = worker
        self.branch = branch
        self.model = branch.pipeline.model
        self.output = Path(output)
        self.mask_dir = Path(mask_dir)
        self.arm = arm
        self.scene = scene
        self.smoke = bool(smoke)
        self.manifest = json.loads((self.mask_dir / "manifest.json").read_text(encoding="utf8"))
        if (self.manifest.get("scene") != scene or not self.manifest.get("train_only")
                or self.manifest.get("test_masks_created") is not False):
            raise RuntimeError("C masks must be frozen training-only assets for the requested scene")
        self.rows = {str(row["camera"]): row for row in self.manifest["rows"]}
        records = worker._train_records(branch)
        record_names = {Path(str(row[1])).stem for row in records}
        if record_names != set(self.rows):
            raise RuntimeError(f"C mask/train camera mismatch: {sorted(record_names ^ set(self.rows))}")
        self._mask_cpu: Dict[str, Tensor] = {}
        for name, row in self.rows.items():
            path = self.mask_dir / f"{name}.png"
            if _sha256(path) != row["mask_sha256"]:
                raise RuntimeError(f"C mask digest mismatch: {path}")
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (int(row["width"]), int(row["height"])):
                    raise RuntimeError(f"C mask shape/mode mismatch: {path}")
                array = np.asarray(image, dtype=np.uint8).copy()
            if set(np.unique(array)) - {0, 255}:
                raise RuntimeError(f"C mask is not binary: {path}")
            if int((array > 0).sum()) != int(row["pixels"]):
                raise RuntimeError(f"C mask pixel count mismatch: {path}")
            if not array.any() and row.get("status") != "UNKNOWN_NO_SUPERVISION":
                raise RuntimeError(f"empty C mask must explicitly declare unknown support: {path}")
            self._mask_cpu[name] = torch.from_numpy((array > 0).astype(np.uint8))
        self._resized: Dict[tuple[str, int, int, str], Tensor] = {}
        self._medium_groups = ("medium_mlp", "direction_encoding")
        self._opacity_groups = ("opacities",)
        self.log_path = self.output / "open_water_C.jsonl"
        self.log = self.log_path.open("a", buffering=1, encoding="utf8")
        config = {
            "arm": arm,
            "scene": scene,
            "mask_manifest_sha256": _sha256(self.mask_dir / "manifest.json"),
            "mask_views": len(self.rows),
            "nonempty_mask_views": sum(int(row["pixels"]) > 0 for row in self.rows.values()),
            "empty_mask_policy": "skip auxiliary gradients and retain all native updates",
            "target": "native tied B_inf == medium_rgb, the existing no-Gaussian infinite-water output",
            "C1_loss": "masked pixel-mean Charbonnier(B_inf - observed RGB), delta=0.01",
            "C2_loss": "C1 plus masked mean -log(final_transmittance), floor=1e-4",
            "C1_parameter_groups": list(self._medium_groups),
            "C2_parameter_groups": list(self._opacity_groups),
            "per_group_gradient_cap_fraction": 0.05,
            "formal_schedule": "every 4 steps; 0 through 500; ramp 501-2000; hold 2000-8000; decay 8001-10000; off thereafter",
            "smoke_schedule": "steps 4,8,12,16 at full strength; smoke is not a formal initialization",
            "smoke": self.smoke,
            "unknown_outside_mask": True,
            "forward_change": "none; uses existing renderer outputs and terminal transmittance",
        }
        (self.output / "open_water_C_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf8")

    def _schedule(self, step: int) -> float:
        if self.smoke:
            return 1.0 if 4 <= step <= 16 else 0.0
        return schedule(step)

    def _mask(self, name: str, height: int, width: int, device: torch.device) -> Tensor:
        name = Path(name).stem
        if name not in self._mask_cpu:
            raise RuntimeError(f"C has no frozen O mask for training camera {name}")
        key = (name, int(height), int(width), str(device))
        if key not in self._resized:
            source = self._mask_cpu[name].to(device=device, dtype=torch.float32)[None, None]
            resized = F.interpolate(source, size=(height, width), mode="nearest")[0, 0] > 0.5
            if int(self.rows[name]["pixels"]) > 0 and not bool(resized.any()):
                raise RuntimeError(f"C open-water mask vanished at active resolution: {name}/{height}x{width}")
            self._resized[key] = resized
        return self._resized[key]

    def _apply_group(self, loss: Tensor, params: Sequence[Tensor], group_indices: Mapping[str, Sequence[int]],
                     names: Sequence[str], strength: float) -> Dict[str, Any]:
        indices = sorted({int(index) for name in names for index in group_indices.get(name, ())})
        if not indices:
            raise RuntimeError(f"registered C parameter groups are absent: {names}")
        selected = [params[index] for index in indices]
        native = [parameter.grad for parameter in selected]
        raw = torch.autograd.grad(loss, selected, retain_graph=True, allow_unused=True)
        native_norm = _norm(native)
        raw_norm = _norm(raw)
        cap = 0.05 * strength * native_norm
        scale = min(1.0, cap / max(raw_norm, 1e-30)) if cap > 0.0 and raw_norm > 0.0 else 0.0
        added = []
        with torch.no_grad():
            for parameter, original, gradient in zip(selected, native, raw):
                if gradient is None or scale == 0.0:
                    added.append(None)
                    continue
                value = gradient.detach() * scale
                if original is None:
                    parameter.grad = value.clone()
                else:
                    parameter.grad.add_(value)
                added.append(value)
        added_norm = _norm(added)
        if added_norm > cap + max(1e-12, cap * 1e-5):
            raise RuntimeError(f"C group gradient cap exceeded for {names}: {added_norm} > {cap}")
        return {
            "native_grad_l2": native_norm,
            "raw_aux_grad_l2": raw_norm,
            "added_grad_l2": added_norm,
            "cap_grad_l2": cap,
            "scale": scale,
            "nonzero_raw_grad_tensors": sum(g is not None and bool(torch.count_nonzero(g).item()) for g in raw),
        }

    def apply(self, branch: Any, outputs: Mapping[str, Tensor], batch: Mapping[str, Any], base_loss: Tensor,
              step: int, output_dir: Path, params: Sequence[Tensor], group_indices: Mapping[str, Sequence[int]],
              camera_index: int, sequence_row: Mapping[str, Any]) -> Dict[str, Any]:
        if branch is not self.branch:
            raise RuntimeError("C hook attached to a different branch")
        strength = self._schedule(int(step))
        if strength <= 0.0 or int(step) % 4:
            return {"C_arm": self.arm, "C_active": False, "C_strength": strength,
                    "C_step_seconds": 0.0, "C_medium_grad_added_l2": 0.0, "C_opacity_grad_added_l2": 0.0}
        import time
        started = time.perf_counter()
        model = branch.pipeline.model
        predicted = outputs["pred_image"]
        camera_name = Path(str(sequence_row["camera_name"])).stem
        mask = self._mask(camera_name, int(predicted.shape[0]), int(predicted.shape[1]), predicted.device)
        count = int(mask.sum().item())
        if count == 0:
            self.log.write(json.dumps({"arm": self.arm, "step": int(step),
                "camera_index": int(camera_index), "camera_name": camera_name,
                "strength": float(strength), "open_pixels": 0,
                "skipped": "UNKNOWN_NO_SUPERVISION"}) + "\n")
            return {"C_arm": self.arm, "C_active": False, "C_strength": strength,
                    "C_open_pixels": 0, "C_step_seconds": time.perf_counter() - started,
                    "C_medium_grad_added_l2": 0.0, "C_opacity_grad_added_l2": 0.0}
        gt = self.worker._gt(model, batch, outputs["background"]).detach().float()
        if gt.shape[:2] != predicted.shape[:2]:
            raise RuntimeError(f"C GT/output resolution mismatch: {tuple(gt.shape)} vs {tuple(predicted.shape)}")
        medium = outputs.get("b_inf")
        if not isinstance(medium, Tensor):
            raise RuntimeError("C requires the native tied B_inf output")
        medium_rgb = outputs["medium_rgb"]
        if medium.shape != medium_rgb.shape or not torch.equal(medium.detach(), medium_rgb.detach()):
            raise RuntimeError("C only accepts the native tied B_inf == medium_rgb path")
        residual = medium.float() - gt
        pixel_charbonnier = torch.sqrt(residual.square() + 1e-4).mean(dim=-1)
        medium_loss = pixel_charbonnier[mask].mean()
        if not bool(torch.isfinite(medium_loss).item()):
            raise RuntimeError(f"non-finite C1 loss at step {step}")
        medium_info: Dict[str, Any] = {"native_grad_l2": 0.0, "raw_aux_grad_l2": 0.0,
                                       "added_grad_l2": 0.0, "cap_grad_l2": 0.0,
                                       "scale": 0.0, "nonzero_raw_grad_tensors": 0}
        medium_info = self._apply_group(medium_loss, params, group_indices, self._medium_groups, strength)

        transmittance = outputs.get("final_transmittance")
        if not isinstance(transmittance, Tensor):
            raise RuntimeError("C2 requires the native terminal-transmittance output")
        if transmittance.shape != outputs["accumulation"].shape or not torch.allclose(
                transmittance.detach(), 1.0 - outputs["accumulation"].detach(), atol=2e-6, rtol=1e-6):
            raise RuntimeError("native final transmittance and accumulation are inconsistent")
        opacity_info: Dict[str, Any] = {"native_grad_l2": 0.0, "raw_aux_grad_l2": 0.0,
                                        "added_grad_l2": 0.0, "cap_grad_l2": 0.0,
                                        "scale": 0.0, "nonzero_raw_grad_tensors": 0}
        occupancy_loss = transmittance.new_zeros(())
        if self.arm == "C2":
            occupancy_loss = -torch.log(transmittance.float().clamp(min=1e-4, max=1.0))[mask].mean()
            if not bool(torch.isfinite(occupancy_loss).item()):
                raise RuntimeError(f"non-finite C2 occupancy loss at step {step}")
            opacity_info = self._apply_group(occupancy_loss, params, group_indices, self._opacity_groups, strength)

        row = {
            "arm": self.arm,
            "step": int(step),
            "camera_index": int(camera_index),
            "camera_name": camera_name,
            "strength": float(strength),
            "open_pixels": count,
            "open_fraction": float(mask.float().mean().item()),
            "L_C1_medium_only": float(medium_loss.detach().item()),
            "L_C2_occupancy": float(occupancy_loss.detach().item()),
            "T_end_mean": float(transmittance.detach().float()[mask].mean().item()),
            "T_end_p05": float(torch.quantile(transmittance.detach().float()[mask].reshape(-1), 0.05).item()),
            "medium_group": medium_info,
            "opacity_group": opacity_info,
            "medium_equals_b_inf_max_abs": float((medium.detach() - medium_rgb.detach()).abs().max().item()),
        }
        self.log.write(json.dumps(row, allow_nan=False) + "\n")
        return {
            "C_arm": self.arm,
            "C_active": True,
            "C_strength": float(strength),
            "C_open_pixels": count,
            "C_L_medium_only": row["L_C1_medium_only"],
            "C_L_occupancy": row["L_C2_occupancy"],
            "C_medium_grad_raw_l2": medium_info["raw_aux_grad_l2"],
            "C_medium_grad_added_l2": medium_info["added_grad_l2"],
            "C_medium_grad_native_l2": medium_info["native_grad_l2"],
            "C_opacity_grad_raw_l2": opacity_info["raw_aux_grad_l2"],
            "C_opacity_grad_added_l2": opacity_info["added_grad_l2"],
            "C_opacity_grad_native_l2": opacity_info["native_grad_l2"],
            "C_T_end_mean": row["T_end_mean"],
            "C_step_seconds": time.perf_counter() - started,
        }
