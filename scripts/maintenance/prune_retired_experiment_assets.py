#!/usr/bin/env python3
"""Delete only explicitly retired artifacts, preserving research records and sources."""
from __future__ import annotations

import argparse
import json
import os
import stat
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Historical controls remain in newer strict ablations; retain their metrics.
RETIRED_CHECKPOINT_ROOTS = {
    "crcc_bnd_on_20260911": "CRCC eliminated; keep metrics, reports and comparison sheets",
    "sbr_bnd_on_four_scene_strict_20260915": "SBR eliminated in favor of FSB; keep evaluation records",
    "direct_mdrr_cica_four_scene_20260903": "old CICA reference superseded by retained September-14 strict ablations",
}
KEEP_RENDER_ROOTS = {
    "fsb_bnd_on_four_scene_strict_20260915_v1",
    "four_way_fsb_renderer_cica_strict_20260916_v4",
    "bnd_on_cica_ablation_20260914", "bnd_on_cica_ablation_20260914_rerender",
    "bnd_on_cica_ablation_extra3_20260914",
    "legacy_no_bounded_three_improvements_20260905",
    "legacy_ocmc_cica_no_auxiliary_20260905",
    "reproduce_best_single_checkpoints_20260907",
    "current_dewatering_cica_no_auxiliary_20260905",
    "ablation_m1_ocmc_recovered_20260907_v2",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".exr", ".tiff"}
PROTECTED_OUTPUT_ROOTS = {
    "bnd_on_cica_source_extra_20260914", "taba_paired_source_20260910_v2",
    "fsb_bnd_on_four_scene_strict_20260915_v1",
    "four_way_fsb_renderer_cica_strict_20260916_v4", "fsb_cica_fork_repeat_20260916_v1",
    "bnd_on_cica_ablation_20260914", "bnd_on_cica_ablation_extra3_20260914",
    "bnd_off_teacher_strict_20260914", "legacy_ocmc_cica_no_auxiliary_20260905",
    "legacy_no_bounded_three_improvements_20260905", "reproduce_best_single_checkpoints_20260907",
    "m1_raoc_causal_four_scene_20260827",
}


def regular_files(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
        for name in files:
            p = Path(directory) / name
            if stat.S_ISREG(p.lstat().st_mode):
                yield p


def reason_for(path):
    rel = path.relative_to(ROOT)
    kind, top = rel.parts[:2]
    if kind == "outputs":
        if top in PROTECTED_OUTPUT_ROOTS:
            return None
        if top in RETIRED_CHECKPOINT_ROOTS and path.suffix in {".ckpt", ".pt", ".pth"}:
            return RETIRED_CHECKPOINT_ROOTS[top]
        if top == "m1_ocmc_causal_iui3_20260825" and path.suffix == ".ckpt":
            if path.name not in {"step-000003000.ckpt", "step-000014999.ckpt"}:
                return "old OCMC intermediate snapshot; preserve source, endpoints and all numeric records"
        if top == "crcc_bnd_on_20260911" and "renders" in rel.parts and path.suffix in IMAGE_SUFFIXES:
            return "retired CRCC per-view image; comparison contact sheets preserved"
        if top == "dvs_frozen_visibility_20260911" and "verified_v2" not in rel.parts and path.suffix in IMAGE_SUFFIXES:
            return "superseded DVS first-pass rendering; verified_v2 retained"
    elif kind == "renders" and top not in KEEP_RENDER_ROOTS and path.suffix in IMAGE_SUFFIXES:
        return "obsolete candidate/repeated rendering; baseline, best FSB and latest comparisons retained"
    return None


def disk_free():
    s = os.statvfs(ROOT)
    return s.f_bavail * s.f_frsize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    out = args.manifest_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = disk_free()
    rows, kept = [], []
    bytes_before, files_before = Counter(), Counter()
    for kind in ("outputs", "renders"):
        base = ROOT / kind
        assert not base.is_symlink()
        for path in regular_files(base):
            s = path.lstat()
            bytes_before[kind] += s.st_size
            files_before[kind] += 1
            reason = reason_for(path)
            row = dict(path=str(path.relative_to(ROOT)), bytes=s.st_size, blocks_bytes=s.st_blocks*512,
                       inode=s.st_ino, device=s.st_dev, mtime_ns=s.st_mtime_ns)
            if reason:
                rows.append(dict(row, reason=reason))
            elif path.suffix == ".ckpt":
                kept.append(row)
    plan = dict(created_unix=time.time(), apply=args.apply, root=str(ROOT), files=rows,
                file_count=len(rows), bytes=sum(r["bytes"] for r in rows),
                before_bytes=dict(bytes_before), before_file_count=dict(files_before),
                kept_checkpoint_files=kept, protected_output_roots=sorted(PROTECTED_OUTPUT_ROOTS),
                kept_render_roots=sorted(KEEP_RENDER_ROOTS), never_follow_symlinks=True,
                permission="User explicitly requested removal of most retired/unimportant files under outputs and renders.")
    (out / "deletion_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    if not args.apply:
        print(json.dumps({k: plan[k] for k in ("file_count", "bytes", "before_bytes")}, indent=2))
        return
    affected = set()
    with (out / "deleted_files.jsonl").open("x") as log:
        for row in rows:
            path = ROOT / row["path"]
            assert path.relative_to(ROOT).parts[0] in ("outputs", "renders")
            assert not any(p.is_symlink() for p in [path] + list(path.parents) if p != ROOT.parent)
            current = path.lstat()
            if (current.st_ino, current.st_dev, current.st_size, current.st_mtime_ns) != (row["inode"], row["device"], row["bytes"], row["mtime_ns"]):
                raise RuntimeError(f"file changed after plan; stopped before deleting {path}")
            path.unlink()
            log.write(json.dumps(row)+"\n")
            log.flush()
            affected.add(ROOT / Path(row["path"]).parts[0] / Path(row["path"]).parts[1])
    for top in affected:
        notice = top / "ASSET_CLEANUP_2026-09-16.json"
        if notice.exists():
            raise FileExistsError(notice)
        subset = [r for r in rows if str(ROOT / r["path"]).startswith(str(top)+"/")]
        notice.write_text(json.dumps(dict(reason="User requested pruning retired assets; original metrics are historical records.",
            removed_files=len(subset), removed_bytes=sum(r["bytes"] for r in subset),
            deletion_manifest=str(out / "deletion_plan.json"),
            caveat="Original checkpoint_retained fields describe time of evaluation; consult this cleanup record for current availability."), indent=2)+"\n")
    # Verify protected checkpoint objects were neither removed nor changed.
    for row in kept:
        s = (ROOT / row["path"]).stat()
        assert (s.st_ino, s.st_size, s.st_mtime_ns) == (row["inode"], row["bytes"], row["mtime_ns"])
    result = dict(deleted_count=len(rows), deleted_bytes=sum(r["bytes"] for r in rows),
                  checkpoints_deleted=sum(Path(r["path"]).suffix == ".ckpt" for r in rows),
                  checkpoints_preserved=len(kept), preserved_checkpoint_stat_checks=True,
                  free_bytes_before=before, free_bytes_after=disk_free(),
                  before_bytes=dict(bytes_before),
                  after_bytes={k: sum(p.stat().st_size for p in regular_files(ROOT/k)) for k in ("outputs", "renders")})
    (out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
