"""Supervise six 15k C runs and retain exit codes, code snapshot and telemetry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCENES = ["Curasao", "JapaneseGradens-RedSea", "Panama"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", nargs=6, default=["1", "2", "3", "4", "5", "6"])
    a = p.parse_args()
    out = a.output.resolve()
    if (out / "formal").exists():
        raise FileExistsError("Refusing to overwrite existing formal runs")
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError("Need at least 15 GiB free for this campaign")
    smoke = json.loads((out/"smoke_audit.json").read_text())
    assert len(smoke) == 6 and all(r["updates"] == 20 and r["reload"] and
        r["initial_model_match"] and r["sequence_match"] for r in smoke)
    (out/"formal").mkdir()
    paths = ["water_splatting/open_water_C.py", "water_splatting/water_splatting.py",
             "scripts/experiments/run_open_water_C_worker.py",
             "scripts/experiments/run_open_water_C_extension.py",
             "scripts/experiments/prepare_open_water_C_extension.py",
             "scripts/experiments/run_direct_mdrr_cica_scene.py",
             "scripts/experiments/geometry_projection_backend.py"]
    hashes = {}
    for name in paths:
        src = ROOT/name
        dst = out/"code"/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    manifest = dict(status="TRAINING", seed=42, steps_per_arm=15000,
                    new_formal_updates=90000, smoke_updates=120, code_sha256=hashes,
                    IUI3_reuse="research_runs/C_open_water_20261024/formal/IUI3-RedSea",
                    masks_sha256={s: hashlib.sha256((out/"assets"/s/"masks/manifest.json").read_bytes()).hexdigest() for s in SCENES},
                    started_unix=time.time(), jobs=[])
    def save():
        tmp = out/"run_manifest.tmp"
        tmp.write_text(json.dumps(manifest, indent=2)+"\n")
        tmp.replace(out/"run_manifest.json")
    processes = []
    for gpu, (scene, arm) in zip(a.gpus, ((s,c) for s in SCENES for c in ["C1", "C2"])):
        cmd = [sys.executable, "-m", "scripts.experiments.run_open_water_C_worker",
               "--scene", scene, "--arm", arm, "--gpu", gpu,
               "--output", str(out/"formal"/scene/arm),
               "--mask-dir", str(out/"assets"/scene/"masks")]
        log = (out/"logs"/f"{scene}_{arm}_formal.log").open("w")
        process = subprocess.Popen(cmd, cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CONDA_DEFAULT_ENV="water_splatting"),
            stdout=log, stderr=subprocess.STDOUT)
        row = dict(scene=scene, arm=arm, gpu=gpu, pid=process.pid, command=cmd,
                   status="RUNNING", started_unix=time.time(), sampled_gpu_memory_mib_max=0)
        manifest["jobs"].append(row)
        processes.append((process, log, row))
    save()
    with (out/"gpu_process_samples.jsonl").open("w", buffering=1) as telemetry:
        while any(process.poll() is None for process, _, _ in processes):
            sample = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            mem = {}
            for line in sample.stdout.splitlines():
                fields = line.split(",")
                if len(fields) == 2 and fields[0].strip().isdigit() and fields[1].strip().isdigit():
                    mem[int(fields[0])] = int(fields[1])
            for process, log, row in processes:
                row["sampled_gpu_memory_mib_max"] = max(row["sampled_gpu_memory_mib_max"], mem.get(process.pid, 0))
                if process.poll() is not None and row["status"] == "RUNNING":
                    row.update(exit_code=process.returncode, status="COMPLETE" if process.returncode == 0 else "FAILED", finished_unix=time.time())
                    log.close()
            telemetry.write(json.dumps(dict(unix=time.time(), memory_mib=mem))+"\n")
            save()
            time.sleep(15)
    for process, log, row in processes:
        code = process.wait()
        if row["status"] == "RUNNING":
            row.update(exit_code=code, status="COMPLETE" if code == 0 else "FAILED", finished_unix=time.time())
        log.close()
    manifest.update(status="TRAINING_COMPLETE" if all(r["exit_code"] == 0 for r in manifest["jobs"]) else "FAILED", finished_unix=time.time())
    save()


if __name__ == "__main__":
    main()
