#!/usr/bin/env python3
"""Analysis-only wrapper for M1-OCMC-CAUSAL-IUI3 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments import run_m1_ocmc_causal_iui3 as OCMC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OCMC.OUTPUT_DIR)
    parser.add_argument("--phase-a-output-dir", type=Path, default=OCMC.PHASE_A_OUTPUT_DIR)
    parser.add_argument("--final-step", type=int, default=OCMC.FINAL_ACTUAL_STEP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = OCMC.run_analysis_only(args.repo, args.output_dir, int(args.final_step), args.phase_a_output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, default=OCMC._json_default), flush=True)


if __name__ == "__main__":
    main()
