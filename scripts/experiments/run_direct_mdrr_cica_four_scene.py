#!/usr/bin/env python3
"""Run the registered A1/A2/A3 four-scene MDRR/CICA matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "direct_mdrr_cica_four_scene_20260903"
WORKER = REPO_ROOT / "scripts" / "experiments" / "run_direct_mdrr_cica_scene.py"
ANALYZER = REPO_ROOT / "scripts" / "diagnostics" / "analyze_direct_mdrr_cica_four_scene.py"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENE_GPUS = {
    "Curasao": "6",
    "IUI3-RedSea": "7",
    "JapaneseGradens-RedSea": "8",
    "Panama": "9",
}
PHASES = ("A1", "A2", "A3")
FINAL_STEP = 14999
SNAPSHOT_STEPS = (5000, 8000, 10000, 13000, 14999)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def _completed_run(scene_dir: Path, scene: str, arm: str) -> bool:
    required = (
        "training_summary.json",
        "final_summary.json",
        "evaluation_metrics.csv",
        "per_view_metrics.csv",
        "decomposition_safety.json",
        "render_manifest.csv",
        "start_state_equivalence.json",
        "gradient_routing_audit.json",
    )
    if not all((scene_dir / name).is_file() for name in required):
        return False
    if not all((scene_dir / "checkpoints" / f"step-{step:09d}.ckpt").is_file() for step in SNAPSHOT_STEPS):
        return False
    training = json.loads((scene_dir / "training_summary.json").read_text(encoding="utf8"))
    start = json.loads((scene_dir / "start_state_equivalence.json").read_text(encoding="utf8"))
    routing = json.loads((scene_dir / "gradient_routing_audit.json").read_text(encoding="utf8"))
    return bool(
        training.get("scene") == scene
        and training.get("arm") == arm
        and int(training.get("final_step", -1)) == FINAL_STEP
        and int(training.get("completed_updates", -1)) == FINAL_STEP - 3000
        and start.get("START_STATE_EQUIVALENCE") is True
        and routing.get("GRADIENT_ROUTING_AUDIT") is True
    )


def _launch_phase(repo: Path, root: Path, phase: str, logs: Path, resume: bool) -> Dict[str, int]:
    processes: Dict[str, subprocess.Popen[str]] = {}
    handles: Dict[str, Any] = {}
    statuses: Dict[str, int] = {}
    pending: List[tuple[str, str, Path]] = []
    for scene, gpu in SCENE_GPUS.items():
        scene_dir = root / scene / phase
        if scene_dir.exists() and any(scene_dir.iterdir()):
            if resume and _completed_run(scene_dir, scene, phase):
                print(f"skipping completed {phase}/{scene}", flush=True)
                statuses[scene] = 0
                continue
            raise RuntimeError(f"partial or invalid formal output directory: {scene_dir}")
        pending.append((scene, gpu, scene_dir))
    try:
        for scene, gpu, scene_dir in pending:
            scene_dir.mkdir(parents=True, exist_ok=True)
            handle = (logs / f"{phase}_{scene}.log").open("w", encoding="utf8")
            handles[scene] = handle
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["CONDA_DEFAULT_ENV"] = "water_splatting"
            command = [
                str(PYTHON),
                str(WORKER),
                "--scene",
                scene,
                "--arm",
                phase,
                "--gpu",
                gpu,
                "--output-dir",
                str(scene_dir),
            ]
            print(f"launching {phase}/{scene} on physical GPU {gpu}", flush=True)
            processes[scene] = subprocess.Popen(
                command,
                cwd=str(repo),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        for scene, process in processes.items():
            statuses[scene] = int(process.wait())
        _write(root / f"{phase}_worker_status.json", statuses)
        return statuses
    finally:
        for handle in handles.values():
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip only formally complete scene/arm directories")
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise RuntimeError(f"non-empty formal output root: {root}")
    if args.resume and root.exists() and any(root.iterdir()):
        manifest_path = root / "launcher_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"resume root has no launcher manifest: {root}")
        existing = json.loads(manifest_path.read_text(encoding="utf8"))
        if existing.get("experiment") != "DIRECT_TRAINING_MDRR_CICA_AND_COMBINED_FOUR_SCENE":
            raise RuntimeError(f"resume root belongs to another experiment: {root}")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "DIRECT_TRAINING_MDRR_CICA_AND_COMBINED_FOUR_SCENE",
        "scene_gpu": SCENE_GPUS,
        "phases": list(PHASES),
        "phase_order": "A1 then A2 then A3; four workers in parallel within each phase",
        "worker": str(WORKER),
        "analyzer": str(ANALYZER),
        "python": str(PYTHON),
        "cuda_policy": "only physical GPUs 6,7,8,9; one visible GPU per worker",
        "formal_training": {"start_step": 3000, "final_step": 14999, "matched_updates": 11999},
        "no_sweep": True,
        "resume_policy": "skip only runs passing final artifacts, checkpoint, start-state, and routing audits",
    }
    _write(root / "launcher_manifest.json", manifest)
    phase_status: Dict[str, Dict[str, int]] = {}
    for phase in PHASES:
        statuses = _launch_phase(repo, root, phase, logs, args.resume)
        phase_status[phase] = statuses
        _write(root / "phase_status.json", phase_status)
        if any(code != 0 for code in statuses.values()):
            raise SystemExit(1)
    if not args.skip_analysis:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "6"
        environment["CONDA_DEFAULT_ENV"] = "water_splatting"
        with (logs / "aggregate.log").open("w", encoding="utf8") as handle:
            result = subprocess.run(
                [str(PYTHON), str(ANALYZER), "--output-root", str(root), "--render-gpu", "6"],
                cwd=str(repo),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        _write(root / "aggregate_status.json", {"returncode": int(result.returncode)})
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    _write(root / "worker_status.json", phase_status)
    print(json.dumps({"output_root": str(root), "phase_status": phase_status}, sort_keys=True))


if __name__ == "__main__":
    main()
