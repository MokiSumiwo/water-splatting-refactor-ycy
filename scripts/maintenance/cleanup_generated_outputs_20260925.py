#!/usr/bin/env python3
"""Manifest-based cleanup of generated experiment caches.

The script only removes regular files under outputs/, renders/, and research_runs/.
It never follows symlinks, never touches source/data directories, and aborts if a
file changes between inventory and deletion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html.parser
import importlib.util
import json
import os
import re
import shutil
import stat
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "research_runs"
AUDIT_DEFAULT = ROOT / "research_notes" / "storage_cleanup_20260925"


def load_old_policy():
    path = ROOT / "scripts/diagnostics/cleanup_obsolete_assets_20260916.py"
    spec = importlib.util.spec_from_file_location("old_cleanup_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


OLD_POLICY = load_old_policy()


def fingerprint(path: Path):
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise RuntimeError(f"not a regular file: {path}")
    return {
        "bytes": st.st_size,
        "allocated_bytes": st.st_blocks * 512,
        "device": st.st_dev,
        "inode": st.st_ino,
        "mtime_ns": st.st_mtime_ns,
    }


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_links():
    """Return regular files referenced by local Markdown/HTML links."""
    protected = set()

    class Parser(html.parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self.values = []

        def handle_starttag(self, tag, attrs):
            self.values.extend(value for key, value in attrs if key in ("href", "src") and value)

    sources = list(RUNS.rglob("*.md")) + list(RUNS.rglob("*.html"))
    for source in sources:
        try:
            text = source.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        values = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
        parser = Parser()
        parser.feed(text)
        values.extend(parser.values)
        for raw in values:
            target = raw.strip().split("#", 1)[0].split("?", 1)[0]
            parsed = urlparse(target)
            if parsed.scheme or not parsed.path:
                continue
            path = (source.parent / unquote(parsed.path)).resolve()
            if path.is_file() and RUNS in path.parents:
                protected.add(path)
    for directory in RUNS.rglob("gallery"):
        if directory.is_dir():
            protected.update(path.resolve() for path in directory.rglob("*") if path.is_file())
    return protected


def research_action(path: Path, linked: set[Path]):
    rel = path.relative_to(ROOT)
    lower = str(rel).lower()
    suffix = path.suffix.lower()
    if suffix in {".ckpt", ".pt", ".pth"}:
        return "delete", "experiment checkpoint or adapter state; source checkpoints are external"
    if suffix in {".npz", ".npy"}:
        keep_tokens = (
            "/preflight/",
            "/basis/",
            "/support/",
            "sample_banks",
            "target_residual",
            "candidate_trace_strict_zero",
            "/final_native_audit/",
        )
        if any(token in lower for token in keep_tokens) and "/failed_" not in lower:
            return "keep", "locked support or final diagnostic array"
        if "initial_gaussians.npz" in lower and "/failed_" not in lower:
            return "keep", "canonical source Gaussian array"
        return "delete", "rebuildable raw tensor/render cache"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov"}:
        if path.resolve() in linked:
            return "keep", "referenced by report or gallery"
        markers = ("contact", "montage", "sheet", "summary", "plot", "figure")
        if any(marker in path.name.lower() for marker in markers):
            return "keep", "summary visual"
        return "delete", "unreferenced per-view image or video cache"
    return "keep", "report, metric, configuration, code, or audit evidence"


def inventory():
    linked = local_links()
    rows = []
    symlinks = []
    roots = [ROOT / "outputs", ROOT / "renders", RUNS]
    for base in roots:
        if not base.exists():
            continue
        for directory, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
            for name in names:
                path = Path(directory) / name
                if path.is_symlink():
                    symlinks.append(str(path.relative_to(ROOT)))
                    continue
                if not path.is_file():
                    continue
                rel = str(path.relative_to(ROOT))
                if rel.startswith("outputs/") or rel.startswith("renders/"):
                    action, reason = OLD_POLICY.policy(rel)
                    if path.resolve() in linked and action == "delete":
                        action, reason = "keep", "referenced by report or gallery"
                else:
                    action, reason = research_action(path, linked)
                rows.append({"path": rel, "action": action, "reason": reason, **fingerprint(path)})
    return rows, symlinks


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "action", "reason", "bytes", "allocated_bytes", "device", "inode", "mtime_ns"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plan(audit: Path):
    audit.mkdir(parents=True, exist_ok=False)
    rows, symlinks = inventory()
    deleted = [row for row in rows if row["action"] == "delete"]
    kept = [row for row in rows if row["action"] == "keep"]
    write_csv(audit / "inventory.csv", rows)
    write_csv(audit / "delete_manifest.csv", deleted)
    write_csv(audit / "retained_files.csv", kept)
    (audit / "untouched_symlinks.json").write_text(json.dumps(symlinks, indent=2) + "\n")
    summary = {
        "root": str(ROOT),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": len(rows),
        "delete_files": len(deleted),
        "delete_logical_bytes": sum(row["bytes"] for row in deleted),
        "delete_allocated_bytes": sum(row["allocated_bytes"] for row in deleted),
        "delete_gib_logical": sum(row["bytes"] for row in deleted) / 2**30,
        "delete_gib_allocated": sum(row["allocated_bytes"] for row in deleted) / 2**30,
        "reasons": dict(Counter(row["reason"] for row in deleted)),
        "untouched_symlinks": len(symlinks),
        "free_before": shutil.disk_usage(ROOT).free,
    }
    (audit / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def execute(audit: Path):
    summary = json.loads((audit / "plan.json").read_text())
    rows = list(csv.DictReader((audit / "inventory.csv").open(newline="")))
    deleted = list(csv.DictReader((audit / "delete_manifest.csv").open(newline="")))
    if summary["root"] != str(ROOT):
        raise RuntimeError("manifest root mismatch")
    expected = {row["path"] for row in rows if row["action"] == "delete"}
    actual = {row["path"] for row in deleted}
    if expected != actual or len(expected) != len(actual):
        raise RuntimeError("delete manifest mismatch")
    for row in rows:
        path = ROOT / row["path"]
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"path changed or became symlink: {path}")
        current = fingerprint(path)
        if any(int(row[key]) != current[key] for key in current):
            raise RuntimeError(f"file changed since inventory: {path}")
    deleted_bytes = 0
    with (audit / "deletion_log.jsonl").open("w") as log:
        for row in deleted:
            path = ROOT / row["path"]
            path.unlink()
            deleted_bytes += int(row["allocated_bytes"])
            log.write(json.dumps({"path": row["path"], "bytes": int(row["bytes"]), "result": "deleted"}) + "\n")
    after_files, after_links = inventory()
    missing = [row["path"] for row in deleted if (ROOT / row["path"]).exists()]
    result = {
        "deleted_files": len(deleted),
        "deleted_allocated_bytes": deleted_bytes,
        "missing_deleted_paths": len(missing) == 0,
        "missing_deleted_path_count": len(missing),
        "retained_files_rechecked": len(after_files),
        "symlinks_untouched": len(after_links),
        "free_after": shutil.disk_usage(ROOT).free,
        "free_delta": shutil.disk_usage(ROOT).free - int(summary["free_before"]),
        "sha256_delete_manifest": digest(audit / "delete_manifest.csv"),
    }
    (audit / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DEFAULT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    if args.execute:
        execute(audit)
    else:
        plan(audit)


if __name__ == "__main__":
    main()
