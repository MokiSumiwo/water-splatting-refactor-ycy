#!/usr/bin/env python3
"""Run one arm of the strict FSB renderer/CICA four-way pairing."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# Keep any optional DataLoader synchronous.  The actual full-image pipeline
# uses cached image lists.  The 32 idle child processes observed during the
# run come from Torch Inductor's compile pool, not recursive data loading.
import torch
from torch.utils.data import DataLoader
_dataloader_init = DataLoader.__init__
def _single_process_dataloader(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs["persistent_workers"] = False
    kwargs.pop("prefetch_factor", None)
    return _dataloader_init(self, *args, **kwargs)
DataLoader.__init__ = _single_process_dataloader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments import run_direct_mdrr_cica_scene as W

ARMS = {
    "R0C0": ("legacy", "legacy"),
    "R1C0": ("fixed", "legacy"),
    "R0C1": ("legacy", "exact_fsb"),
    "R1C1": ("fixed", "exact_fsb"),
}
SOURCE_MAP = {
    "Curasao": ROOT / "outputs/bnd_on_cica_source_extra_20260914",
    "IUI3-RedSea": ROOT / "outputs/bnd_on_cica_source_extra_20260914",
    "JapaneseGradens-RedSea": ROOT / "outputs/taba_paired_source_20260910_v2/bnd_on",
    "Panama": ROOT / "outputs/taba_paired_source_20260910_v2/bnd_on",
}
CONFIG_MAP = {
    "Curasao": ROOT / "outputs/taba_chromatic_bnd_on_20260910/Curasao/E0/source_config.yml",
    "IUI3-RedSea": ROOT / "outputs/taba_chromatic_bnd_on_20260910/IUI3-RedSea/E0/source_config.yml",
    "JapaneseGradens-RedSea": ROOT / "outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml",
    "Panama": ROOT / "outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-step", type=int, default=14999)
    p.add_argument("--source-root", type=Path, default=None)
    p.add_argument("--source-config", type=Path, default=None)
    p.add_argument("--compact-output", action="store_true")
    args = p.parse_args()
    renderer_mode, cica_mode = ARMS[args.arm]
    W.EXPERIMENT = "FSB_RENDERER_CICA_STRICT_FOUR_WAY_20260916"
    W.ARMS = tuple(ARMS)
    W.AUXILIARY_APPEARANCE_ENABLED = False
    W.AUXILIARY_APPEARANCE_STRENGTH = 0.0
    W.OCMC_ENABLED = True
    W.OCMC_STRENGTH = 1.0
    W.OCMC_STRENGTH_START = 1.0
    W.OCMC_STRENGTH_RAMP_STEPS = 0
    W.CICA_ENABLED = True
    W.ACTIVE_CICA_OVERRIDE = True
    W.ACTIVE_MDRR_OVERRIDE = False
    W.CICA_DERIVATIVE_MODE = cica_mode
    W.RENDERER_BACKWARD_MODE = renderer_mode
    W.BRASH_ENABLED = False
    W.BVAM_ENABLED = False
    W.ENFORCE_SCENE_GPU_ASSIGNMENT = False
    W.START_STEP = 3000
    W.FINAL_STEP = 14999
    # Retain only the endpoint.  The endpoint is evaluated before the worker
    # removes its temporary checkpoint under --compact-output.
    W.ACTIVE_SNAPSHOT_STEPS = (14999,)
    output_dir = args.output_dir.resolve()
    run_name = output_dir.parent.parent.name
    W.ACTIVE_RENDER_ROOT = ROOT / "renders" / run_name / args.arm
    W.ACTIVE_SAVE_COMPACT_RENDERS = True
    storage_tag = hashlib.sha256(str(output_dir).encode()).hexdigest()[:16]
    temporary_root = Path("/dev/shm") / ("water_splatting_four_way_" + storage_tag)
    W.ACTIVE_CHECKPOINT_DIRECTORY = temporary_root / "checkpoints" if args.compact_output else None
    if args.max_step == W.FINAL_STEP and (W.ACTIVE_RENDER_ROOT / args.scene).exists():
        raise FileExistsError(W.ACTIVE_RENDER_ROOT / args.scene)
    source_root = (args.source_root or SOURCE_MAP[args.scene]).resolve()
    source_config = (args.source_config or CONFIG_MAP[args.scene]).resolve()
    W.run(
        args.scene,
        args.arm,
        args.gpu,
        output_dir,
        int(args.max_step),
        source_root=source_root,
        source_config=source_config,
        parameterization="bounded_headroom_sh3",
        training_seed=42,
        compact_output=bool(args.compact_output),
    )
    if args.compact_output:
        removed = [{"name": path.name, "bytes": path.stat().st_size}
                   for path in W.ACTIVE_CHECKPOINT_DIRECTORY.glob("*.ckpt")]
        shutil.rmtree(temporary_root)
        (output_dir / "checkpoints").unlink()
        (output_dir / "checkpoint_storage.json").write_text(json.dumps({
            "storage": str(temporary_root), "ephemeral": True,
            "removed_after_successful_evaluation_or_smoke": True,
            "additional_smoke_cleanup": removed,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
