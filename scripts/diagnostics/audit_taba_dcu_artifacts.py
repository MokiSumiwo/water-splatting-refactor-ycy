#!/usr/bin/env python3
"""Check all 14 DCU runs, their checkpoints and exact native image pairing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.audit_taba_artifacts import audit
from scripts.experiments.run_taba_bnd_on import check_environment
from scripts.experiments.run_taba_dcu_bnd_on import DEFAULT_OUT, PRESETS


if __name__ == "__main__":
    check_environment()
    print(json.dumps(audit(DEFAULT_OUT, variants=tuple(PRESETS)), indent=2))
