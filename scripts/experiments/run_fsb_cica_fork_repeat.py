#!/usr/bin/env python3
"""Four GPU lanes, each with two prefixes and eight retained continuations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/opt/anaconda3/envs/water_splatting/bin/python"
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
ORDER = ("C0a", "C1a", "C1b", "C0b")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if shutil.disk_usage(out.parent).free < 45 * 2**30:
        raise RuntimeError("need 45 GiB free to retain prefixes, endpoints and comparisons")
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    archive = out / "provenance"
    archive.mkdir()
    sources = [Path(__file__).resolve(), ROOT / "scripts/experiments/run_fsb_cica_fork_worker.py",
               ROOT / "scripts/experiments/run_direct_mdrr_cica_scene.py",
               ROOT / "scripts/experiments/run_four_way_fsb_renderer_cica_worker.py",
               ROOT / "scripts/experiments/run_m1_raoc_causal_scene.py",
               ROOT / "water_splatting/water_splatting.py",
               ROOT / "water_splatting/fields/gaussian_appearance.py",
               ROOT / "water_splatting/cuda/csrc/backward.cu",
               ROOT / "water_splatting/cuda/csrc/forward.cu",
               ROOT / "research_notes/FSB_CICA_SHARED_PREFIX_PROTOCOL_2026-09-16.md"]
    hashes = {}
    for source in sources:
        dest = archive / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[str(source)] = sha(source)
    binaries = [ROOT / "water_splatting/csrc.so",
                Path.home() / ".cache/torch_extensions/py38_cu118/water_splatting_cuda_fsb_fix_20260915/water_splatting_cuda_fsb_fix_20260915.so"]
    for binary in binaries:
        if not binary.exists():
            raise FileNotFoundError(binary)
        dest = archive / "binaries" / binary.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(binary, dest)
        hashes[str(binary)] = sha(binary)
    (archive / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    queues, running, statuses = {}, {}, {}
    for i, scene in enumerate(SCENES):
        queue = []
        for renderer, mode in (("R0", "legacy"), ("R1", "fixed")):
            base = out / scene / renderer
            prefix = base / "PREFIX/checkpoints/step-000009999.ckpt"
            for arm in ("PREFIX",) + ORDER:
                cmd = [PYTHON, str(ROOT / "scripts/experiments/run_fsb_cica_fork_worker.py"),
                       "--scene", scene, "--gpu", str(6+i), "--renderer", mode,
                       "--arm", arm, "--output", str(base / arm)]
                if arm != "PREFIX":
                    cmd.extend(["--fork", str(prefix)])
                queue.append((renderer + "/" + arm, cmd))
        queues[scene] = queue
        statuses[scene] = {}
    failed = False
    try:
        while any(queues.values()) or running:
            for i, scene in enumerate(SCENES):
                if scene in running:
                    name, process, log = running[scene]
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    statuses[scene][name] = code
                    del running[scene]
                    print(json.dumps({"scene": scene, "job": name, "exit": code}), flush=True)
                    if code:
                        queues[scene].clear()
                        failed = True
                if queues[scene]:
                    # Refuse to silently mix source versions across workers.
                    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
                        raise RuntimeError("source/binary changed after protocol snapshot")
                    name, cmd = queues[scene].pop(0)
                    log = (out / "logs" / (scene + "_" + name.replace("/", "_") + ".log")).open("x")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(6+i), CONDA_DEFAULT_ENV="water_splatting",
                               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", MAX_JOBS="1", TORCHINDUCTOR_COMPILE_THREADS="1")
                    process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    running[scene] = (name, process, log)
                    print(json.dumps({"scene": scene, "job": name, "pid": process.pid, "gpu": 6+i}), flush=True)
            state = {"completed": statuses, "active": {s: {"job": x[0], "pid": x[1].pid} for s, x in running.items()},
                     "pending": {s: [x[0] for x in q] for s, q in queues.items()}, "failed": failed}
            (out / "status.json").write_text(json.dumps(state, indent=2) + "\n")
            if running:
                time.sleep(5)
    finally:
        for name, process, log in running.values():
            process.terminate()
            process.wait()
            log.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
