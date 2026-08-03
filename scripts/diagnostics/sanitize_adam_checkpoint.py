#!/usr/bin/env python
"""Sanitize Nerfstudio Adam optimizer states for replay experiments.

Some phase-replay checkpoints contain Adam param entries that are present but
empty because a param group was configured but inactive for the seed run. Torch
2.1 expects the first Adam state entry to contain ``step`` during
``load_state_dict``. This script preserves the model/scheduler/scaler state and
removes those empty optimizer entries so Adam can lazily initialize them when
the parameter first receives gradients.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable

import torch


def _checkpoint_paths(load_dir: Path, load_step: int | None) -> Iterable[Path]:
    if load_step is not None:
        path = load_dir / f"step-{load_step:09d}.ckpt"
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]
    paths = sorted(load_dir.glob("step-*.ckpt"))
    if not paths:
        raise FileNotFoundError(f"No step-*.ckpt files found in {load_dir}")
    return paths


def _sanitize_optimizer_state(opt_state: Dict[str, Any], checkpoint_step: int) -> Dict[str, int]:
    removed_empty = 0
    added_step = 0
    state = opt_state.get("state", {})
    if not isinstance(state, dict):
        return {"removed_empty": removed_empty, "added_step": added_step}

    for key in list(state.keys()):
        value = state[key]
        if isinstance(value, dict) and not value:
            del state[key]
            removed_empty += 1
            continue
        if isinstance(value, dict) and ("exp_avg" in value or "exp_avg_sq" in value) and "step" not in value:
            ref = value.get("exp_avg")
            if torch.is_tensor(ref):
                value["step"] = torch.tensor(float(checkpoint_step), dtype=torch.float32)
            else:
                value["step"] = checkpoint_step
            added_step += 1
    return {"removed_empty": removed_empty, "added_step": added_step}


def sanitize_checkpoint(input_path: Path, output_path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(input_path, map_location="cpu")
    checkpoint_step = int(checkpoint.get("step", 0))
    summary: Dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "step": checkpoint_step,
        "optimizers": {},
    }
    for name, opt_state in checkpoint.get("optimizers", {}).items():
        if isinstance(opt_state, dict):
            summary["optimizers"][name] = _sanitize_optimizer_state(opt_state, checkpoint_step)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-dir", type=Path, required=True, help="Input Nerfstudio checkpoint directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for sanitized checkpoint copies.")
    parser.add_argument("--load-step", type=int, default=None, help="Optional single step to sanitize.")
    parser.add_argument("--copy-non-ckpt", action="store_true", help="Copy non-checkpoint files from load dir.")
    args = parser.parse_args()

    if args.output_dir.resolve() == args.load_dir.resolve():
        raise ValueError("output-dir must differ from load-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for ckpt_path in _checkpoint_paths(args.load_dir, args.load_step):
        summaries.append(sanitize_checkpoint(ckpt_path, args.output_dir / ckpt_path.name))

    if args.copy_non_ckpt:
        for item in args.load_dir.iterdir():
            if item.name.startswith("step-") and item.suffix == ".ckpt":
                continue
            target = args.output_dir / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
