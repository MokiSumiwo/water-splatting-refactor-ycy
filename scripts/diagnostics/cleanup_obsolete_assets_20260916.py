#!/usr/bin/env python3
"""One-off, manifest-based cleanup of retired outputs and renders."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import stat
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
WHOLE_OUTPUTS = {
    "bnd_on_cica_source_extra_20260914", "taba_paired_source_20260910_v2",
    "m1_ocmc_causal_iui3_20260825", "m1_raoc_causal_four_scene_20260827",
    "bnd_off_teacher_strict_20260914", "reproduce_best_single_checkpoints_20260907",
    "four_way_fsb_renderer_cica_strict_20260916_v4",
    "fsb_cica_fork_repeat_20260916_v1",
}
CORE_RENDERS = {
    "fsb_bnd_on_four_scene_strict_20260915_v1",
    "four_way_fsb_renderer_cica_strict_20260916_v4",
    "legacy_ocmc_cica_no_auxiliary_20260905",
    "legacy_no_bounded_three_improvements_20260905",
    "bnd_on_cica_ablation_20260914_rerender", "bnd_on_cica_ablation_extra3_20260914",
    "reproduce_best_single_checkpoints_20260907",
    "current_dewatering_cica_no_auxiliary_20260905",
    "ablation_m1_ocmc_recovered_20260907_v2",
    "m1_ocmc_cica_optimization_20260907_combined_comparison",
}
CORE_OUTPUT_IMAGES = CORE_RENDERS | {
    "bnd_on_cica_ablation_20260914", "sbr_bnd_on_four_scene_strict_20260915",
    "dvs_frozen_visibility_20260911",
}
ENDPOINTS = set()
for scene in SCENES:
    for family in ("fsb_bnd_on_four_scene_strict_20260915_v1",
                   "legacy_ocmc_cica_no_auxiliary_20260905",
                   "sbr_bnd_on_four_scene_strict_20260915"):
        ENDPOINTS.add(f"outputs/{family}/{scene}/checkpoints/step-000014999.ckpt")
    ENDPOINTS.add(f"outputs/direct_mdrr_cica_four_scene_20260903/{scene}/A2/checkpoints/step-000014999.ckpt")
    for step in (3000, 14999):
        ENDPOINTS.add(f"outputs/legacy_no_bounded_three_improvements_20260905/{scene}/checkpoints/C0/step-{step:09d}.ckpt")
    family = ("bnd_on_cica_ablation_extra3_20260914" if scene in SCENES[:2]
              else "bnd_on_cica_ablation_20260914")
    for arm in ("cica_on", "cica_off"):
        ENDPOINTS.add(f"outputs/{family}/{scene}/{arm}/checkpoints/step-000014999.ckpt")


def policy(relative):
    p = Path(relative)
    scope, family = p.parts[:2]
    if scope == "outputs" and family in WHOLE_OUTPUTS:
        return "keep", "active source or canonical experiment"
    if scope == "renders" and family in CORE_RENDERS:
        return "keep", "core visual comparison"
    if relative in ENDPOINTS:
        return "keep", "canonical checkpoint endpoint or paired source"
    if p.suffix.lower() == ".ckpt":
        return "delete", "retired checkpoint or intermediate snapshot"
    if p.suffix.lower() in {".pt", ".pth", ".npz", ".npy"}:
        return "delete", "retired raw tensor or diagnostic cache"
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif"}:
        if scope == "outputs" and family in CORE_OUTPUT_IMAGES:
            return "keep", "core rendered evidence"
        markers = ("contact", "montage", "sheet", "comparison", "summary", "plot")
        if any(m in p.name.lower() for m in markers) or any(
                x.lower().startswith(("contact", "montage", "summary", "visualization"))
                for x in p.parts[2:-1]):
            return "keep", "summary figure or contact sheet"
        return "delete", "retired individual render or duplicate visual cache"
    return "keep", "metrics configuration logs report or other evidence"


def inventory():
    files, links = [], []
    for scope in ("outputs", "renders"):
        base = ROOT / scope
        if base.is_symlink() or base.resolve() != base:
            raise RuntimeError(f"cleanup root must be a real directory: {base}")
        for directory, dirs, names in os.walk(base, followlinks=False):
            for name in list(dirs):
                path = Path(directory) / name
                if path.is_symlink():
                    links.append(path)
                    dirs.remove(name)
            for name in names:
                path = Path(directory) / name
                if path.is_symlink():
                    links.append(path)
                elif path.is_file():
                    files.append(path)
    return sorted(files), sorted(links)


def fingerprint(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise RuntimeError(f"not a regular file: {path}")
    return dict(bytes=s.st_size, allocated_bytes=s.st_blocks * 512,
                device=s.st_dev, inode=s.st_ino, mtime_ns=s.st_mtime_ns)


def write_csv(path, rows):
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def snapshot(rows):
    values = {}
    for scope in ("outputs", "renders"):
        subset = [r for r in rows if r["path"].startswith(scope + "/")]
        values[scope] = {"files": len(subset), "bytes": sum(r["bytes"] for r in subset),
                         "allocated_bytes": sum(r["allocated_bytes"] for r in subset)}
    values["filesystem_free_bytes"] = shutil.disk_usage(ROOT).free
    return values


def validate_endpoints():
    for name in ENDPOINTS:
        fingerprint(ROOT / name)
    link = ROOT / "outputs/fsb_cica_fork_repeat_20260916_v1"
    if not link.is_symlink() or not link.is_dir():
        raise RuntimeError("active experiment symlink missing")


def plan(audit):
    audit.mkdir(parents=True, exist_ok=False)
    validate_endpoints()
    files, links = inventory()
    # Core render aliases can point into a different output family.
    protected_targets = set()
    for path in links:
        relative = str(path.relative_to(ROOT))
        if policy(relative)[0] == "keep":
            target = path.resolve()
            if any(base in target.parents for base in (ROOT / "outputs", ROOT / "renders")):
                protected_targets.add(target)
    rows = []
    for path in files:
        relative = str(path.relative_to(ROOT))
        action, reason = policy(relative)
        if path in protected_targets or any(p in protected_targets for p in path.parents):
            action, reason = "keep", "target of retained render alias"
        rows.append(dict(path=relative, action=action, reason=reason, **fingerprint(path)))
    write_csv(audit / "inventory.csv", rows)
    write_csv(audit / "delete_manifest.csv", [r for r in rows if r["action"] == "delete"])
    write_csv(audit / "retained_checkpoints.csv", [r for r in rows if r["action"] == "keep" and r["path"].endswith(".ckpt")])
    link_rows = [dict(path=str(p.relative_to(ROOT)), target=os.readlink(p),
                      existed_before=p.exists()) for p in links]
    write_csv(audit / "untouched_symlinks.csv", link_rows)
    planned = [r for r in rows if r["action"] == "delete"]
    info = dict(root=str(ROOT), before=snapshot(rows), planned_files=len(planned),
                planned_bytes=sum(r["bytes"] for r in planned),
                planned_allocated_bytes=sum(r["allocated_bytes"] for r in planned),
                reasons=dict(Counter(r["reason"] for r in planned)),
                retained_checkpoints=sum(r["action"] == "keep" and r["path"].endswith(".ckpt") for r in rows),
                symlinks_untouched=len(links))
    (audit / "plan.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def execute(audit):
    info = json.loads((audit / "plan.json").read_text())
    if info["root"] != str(ROOT):
        raise RuntimeError("manifest root mismatch")
    rows = read_csv(audit / "delete_manifest.csv")
    inventory_rows = read_csv(audit / "inventory.csv")
    validate_endpoints()
    for row in inventory_rows:
        p = Path(row["path"])
        if p.is_absolute() or ".." in p.parts or p.parts[0] not in ("outputs", "renders"):
            raise RuntimeError(f"out of scope: {p}")
        path = ROOT / p
        if path.resolve() != path:
            raise RuntimeError(f"symlink traversal: {path}")
        if any(int(row[k]) != v for k, v in fingerprint(path).items()):
            raise RuntimeError(f"file changed since inventory: {path}")
    expected = {r["path"] for r in inventory_rows if r["action"] == "delete"}
    if {r["path"] for r in rows} != expected or len(rows) != len(expected):
        raise RuntimeError("deletion manifest does not match inventory")
    freed, removed = 0, 0
    with (audit / "deletion_log.jsonl").open("x", buffering=1) as log:
        for row in rows:
            parts = Path(row["path"]).parts
            # Walk parents with O_NOFOLLOW so a changed symlink cannot redirect unlink.
            fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                for part in parts[:-1]:
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd)
                    fd = child
                s = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
                if not stat.S_ISREG(s.st_mode) or (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) != tuple(
                        int(row[k]) for k in ("device", "inode", "bytes", "mtime_ns")):
                    raise RuntimeError(f"file changed before unlink: {row['path']}")
                os.unlink(parts[-1], dir_fd=fd)
            finally:
                os.close(fd)
            freed += int(row["allocated_bytes"])
            removed += 1
            log.write(json.dumps(dict(path=row["path"], bytes=int(row["bytes"]), result="deleted")) + "\n")
            if removed % 500 == 0:
                print(json.dumps(dict(deleted=removed, allocated_gib=freed / 2**30)), flush=True)
        os.fsync(log.fileno())
    kept = [r for r in inventory_rows if r["action"] == "keep"]
    for row in kept:
        if any(int(row[k]) != v for k, v in fingerprint(ROOT / row["path"]).items()):
            raise RuntimeError(f"retained file changed: {row['path']}")
    validate_endpoints()
    files, links = inventory()
    remaining = [dict(path=str(p.relative_to(ROOT)), **fingerprint(p)) for p in files]
    result = dict(deleted_files=removed, deleted_allocated_bytes=freed,
                  deleted_logical_bytes=sum(int(r["bytes"]) for r in rows),
                  after=snapshot(remaining), retained_files_verified=len(kept),
                  retained_checkpoints=info["retained_checkpoints"], symlinks_untouched=len(links),
                  missing_deleted_paths=all(not os.path.lexists(ROOT / r["path"]) for r in rows))
    (audit / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    if any(base == audit or base in audit.parents for base in (ROOT / "outputs", ROOT / "renders")):
        raise RuntimeError("audit must be outside cleanup roots")
    (execute if args.execute else plan)(audit)


if __name__ == "__main__":
    main()
