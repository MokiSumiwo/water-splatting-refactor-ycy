#!/usr/bin/env python3
"""Launch the no-Bounded-SH3 three-improvement four-scene experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SOURCE_WORKER = REPO_ROOT / "scripts/experiments/run_legacy_ocmc_scene.py"
AUX_WORKER = REPO_ROOT / "scripts/experiments/run_legacy_aux_scene.py"
CICA_WORKER = REPO_ROOT / "scripts/experiments/run_legacy_aux_cica_scene.py"
ROOT_DEFAULT = REPO_ROOT / "outputs/legacy_no_bounded_three_improvements_20260905"
CONFIG_DEFAULT = REPO_ROOT / "outputs/m1_ocmc_causal_iui3_20260825/source_bnd_config.yml"
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
SOURCE_GPUS = {scene: str(index) for index, scene in enumerate(SCENES)}
AUX_GPUS = {scene: str(index) for index, scene in enumerate(SCENES)}
CICA_GPUS = {scene: str(index + 4) for index, scene in enumerate(SCENES)}
FINAL_STEP = 14999


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf8")


def _env(gpu: str) -> Dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["CONDA_DEFAULT_ENV"] = "water_splatting"
    return environment


def _launch(
    phase: str,
    worker: Path,
    assignments: Mapping[str, str],
    root: Path,
    config: Path,
    source_root: Path | None = None,
    resume: bool = False,
) -> Dict[str, int]:
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: Dict[str, subprocess.Popen[str]] = {}
    handles: Dict[str, Any] = {}
    statuses: Dict[str, int] = {}
    try:
        for scene in SCENES:
            output_dir = root / scene if phase == "source" else root / scene / phase
            if output_dir.exists() and any(output_dir.iterdir()):
                if resume and _complete(phase, output_dir):
                    statuses[scene] = 0
                    continue
                raise RuntimeError(f"partial or invalid output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            handle = (logs / f"{phase}_{scene}.log").open("w", encoding="utf8")
            handles[scene] = handle
            if phase == "source":
                command = [
                    str(PYTHON), str(worker), "--scene", scene, "--gpu", assignments[scene],
                    "--output-dir", str(output_dir), "--final-step", str(FINAL_STEP),
                ]
            elif phase == "A0":
                command = [
                    str(PYTHON), str(worker), "--scene", scene, "--gpu", assignments[scene],
                    "--output-dir", str(output_dir), "--final-step", str(FINAL_STEP),
                    "--source-root", str(source_root), "--source-config", str(config),
                ]
            else:
                command = [
                    str(PYTHON), str(worker), "--scene", scene, "--arm", "A2", "--gpu", assignments[scene],
                    "--output-dir", str(output_dir), "--max-step", str(FINAL_STEP),
                    "--source-root", str(source_root), "--source-config", str(config),
                ]
            processes[scene] = subprocess.Popen(
                command, cwd=str(REPO_ROOT), env=_env(assignments[scene]),
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
        for scene, process in processes.items():
            statuses[scene] = int(process.wait())
        _write(root / f"{phase}_worker_status.json", statuses)
        if any(code != 0 for code in statuses.values()):
            raise SystemExit(1)
        return statuses
    finally:
        for handle in handles.values():
            handle.close()


def _complete(phase: str, output_dir: Path) -> bool:
    if phase == "source":
        return all(
            (output_dir / name).is_file()
            for name in ("final_summary.json", "final_metrics.csv", "camera_sequence.json")
        ) and all(
            (output_dir / "checkpoints/C0" / f"step-{step:09d}.ckpt").is_file()
            for step in (3000, FINAL_STEP)
        )
    if phase == "A0":
        return all((output_dir / name).is_file() for name in ("scene_complete.json", "training_summary.json", "evaluation_metrics.csv"))
    return all((output_dir / name).is_file() for name in ("final_summary.json", "training_summary_A2.csv", "evaluation_metrics.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--source-config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    config = args.source_config.resolve()
    if not PYTHON.is_file() or not config.is_file():
        raise FileNotFoundError(f"missing Python or source config: {PYTHON}, {config}")
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise RuntimeError(f"non-empty output root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "LEGACY_NO_BOUNDED_THREE_IMPROVEMENTS_FOUR_SCENE",
        "parameterization": "legacy",
        "seed": 42,
        "final_step": FINAL_STEP,
        "chain": {
            "legacy": "Legacy SH3 + OCMC OFF",
            "source": "Legacy SH3 + OCMC ON",
            "A0": "Legacy SH3 + OCMC ON + auxiliary appearance regularization",
            "A2": "Legacy SH3 + OCMC ON + auxiliary appearance regularization + CICA",
        },
        "source_gpus": SOURCE_GPUS,
        "auxiliary_gpus": AUX_GPUS,
        "cica_gpus": CICA_GPUS,
        "source_config": str(config),
        "workers": {"source": str(SOURCE_WORKER), "A0": str(AUX_WORKER), "A2": str(CICA_WORKER)},
        "protocol": "dir_xy_camera, tied B_inf, RAOC off, MDRR excluded, 15,000 updates, held-out RGB metrics",
    }
    _write(root / "launcher_manifest.json", manifest)
    source_status = _launch("source", SOURCE_WORKER, SOURCE_GPUS, root, config, resume=args.resume)
    source_root = root
    # A0 and A2 both depend only on the completed source checkpoint, so run
    # the two four-scene matrices concurrently on disjoint GPU sets.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        auxiliary_future = executor.submit(_launch, "A0", AUX_WORKER, AUX_GPUS, root, config, source_root, args.resume)
        cica_future = executor.submit(_launch, "A2", CICA_WORKER, CICA_GPUS, root, config, source_root, args.resume)
        aux_status = auxiliary_future.result()
        cica_status = cica_future.result()
    status = {"source": source_status, "A0": aux_status, "A2": cica_status}
    _write(root / "worker_status.json", status)
    if not args.skip_summary:
        summarizer = REPO_ROOT / "scripts/diagnostics/summarize_legacy_no_bounded_three_improvements.py"
        result = subprocess.run(
            [str(PYTHON), str(summarizer), "--root", str(root)],
            cwd=str(REPO_ROOT), env=_env("0"), check=False,
        )
        _write(root / "aggregate_status.json", {"returncode": int(result.returncode)})
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    print(json.dumps({"output_root": str(root), "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
