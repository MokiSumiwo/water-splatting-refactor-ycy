#!/usr/bin/env python3
"""Compare the existing CUDA audit against the pre-TABA committed model."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_taba_bnd_on import (
    DEFAULT_OUT, GPUS, check_environment, gpu_processes, source_assets, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=GPUS, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT / "engineering")
    args = parser.parse_args()
    check_environment()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --gpu exactly")
    if gpu_processes()[args.gpu]:
        raise RuntimeError(f"GPU {args.gpu} is occupied")
    import water_splatting.water_splatting as production
    from scripts.diagnostics import audit_raoc_cuda_fused_equivalence as existing

    root, config, _ = source_assets("JapaneseGradens-RedSea")
    existing.ARCHIVE_ROOT = root
    existing.RAOC.SCENES["JapaneseGradens-RedSea"]["source_config"] = str(config)
    current = existing.audit("JapaneseGradens-RedSea", args.gpu, 3000, args.output_root / "cuda_equivalence.json")
    baseline = types.ModuleType("water_splatting._taba_head_audit")
    sys.modules[baseline.__name__] = baseline
    source = subprocess.check_output(["git", "show", "HEAD:water_splatting/water_splatting.py"], cwd=ROOT, text=True)
    exec(compile(source, "HEAD:water_splatting/water_splatting.py", "exec"), baseline.__dict__)
    with patch.object(production, "WaterSplattingModel", baseline.WaterSplattingModel):
        reference = existing.audit("JapaneseGradens-RedSea", args.gpu, 3000, args.output_root / "cuda_equivalence_head.json")
    current.pop("second_order_graph")
    reference.pop("second_order_graph")
    result = {"no_regression": current == reference, "strict_forward_pass": current["forward_pass"],
              "strict_backward_pass": current["backward_pass"], "current": current, "head_reference": reference}
    write_json(args.output_root / "cuda_nonregression.json", result)
    print(result, flush=True)
    if not result["no_regression"]:
        raise RuntimeError("CUDA equivalence regression")


if __name__ == "__main__":
    main()
