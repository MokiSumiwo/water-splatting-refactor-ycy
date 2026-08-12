#!/usr/bin/env python
"""Offline summaries and supplemental plots for BND-AWARE-REFINE.

This script reads the completed Panama BND-AWARE-REFINE outputs and final
continuation checkpoints. It does not train or modify checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics import run_bnd_aware_refine_panama as base


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _line_plot(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    label_filter: str | None = None,
) -> None:
    plt.figure(figsize=(8.5, 5.0))
    for branch in base.BRANCHES:
        selected = [
            row
            for row in rows
            if row.get("branch") == branch and (label_filter is None or row.get("label") == label_filter)
        ]
        selected = sorted(selected, key=lambda row: int(row["absolute_step"]))
        plt.plot(
            [int(row["absolute_step"]) for row in selected],
            [float(row[metric]) for row in selected],
            marker="o",
            label=branch,
        )
    plt.title(title)
    plt.xlabel("absolute step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _bar_plot(path: Path, labels: Sequence[str], values: Sequence[float], *, title: str, ylabel: str) -> None:
    plt.figure(figsize=(8.0, 4.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _final_rel(output_dir: Path) -> int:
    summary = json.loads((output_dir / "bnd_aware_refine_final_summary.json").read_text())
    return int(summary["final_relative_step"])


def _per_view_rgb_metrics(repo: Path, output_dir: Path, final_rel: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for branch_name in base.BRANCHES:
        branch = base._load_branch(repo, branch_name)
        try:
            base._load_snapshot(branch, output_dir, final_rel)
            records = base._eval_records(branch.pipeline)
            maps = base._render_records(branch.pipeline, records)
            for _idx, view_id, _camera, _batch in records:
                metrics = base._metric_images(branch.pipeline.model, maps[view_id]["pred"], maps[view_id]["gt"])
                rows.append(
                    {
                        "branch": branch_name,
                        "relative_step": final_rel,
                        "absolute_step": base.START_NOMINAL_STEP + final_rel,
                        "view_id": view_id,
                        **metrics,
                    }
                )
        finally:
            base._release(branch)
    return rows


def _selection_summary(output_dir: Path) -> List[Dict[str, Any]]:
    rows = _read_csv(output_dir / "selection_priority_statistics.csv")
    out: List[Dict[str, Any]] = []
    for branch in ("RH", "RB"):
        for kind in ("split", "duplicate"):
            selected = [
                row
                for row in rows
                if row.get("branch") == branch and row.get("kind") == kind and row.get("mean_selected_guidance")
            ]
            if not selected:
                continue
            sel = sum(float(row["mean_selected_guidance"]) for row in selected) / len(selected)
            cand = sum(float(row["mean_candidate_guidance"]) for row in selected) / len(selected)
            base_sel = sum(float(row["mean_selected_base_score"]) for row in selected) / len(selected)
            base_cand = sum(float(row["mean_candidate_base_score"]) for row in selected) / len(selected)
            out.append(
                {
                    "branch": branch,
                    "kind": kind,
                    "event_count": len(selected),
                    "mean_selected_guidance": sel,
                    "mean_candidate_guidance": cand,
                    "guidance_lift": sel - cand,
                    "mean_selected_base_score": base_sel,
                    "mean_candidate_base_score": base_cand,
                    "base_score_lift": base_sel - base_cand,
                    "selected_total": sum(int(float(row["selected_count"])) for row in selected),
                    "candidate_total": sum(int(float(row["candidate_count"])) for row in selected),
                }
            )
    return out


def _final_region_by_view(output_dir: Path, final_rel: int, label: str) -> List[Dict[str, str]]:
    rows = _read_csv(output_dir / "per_view_metrics.csv")
    return [row for row in rows if int(row["relative_step"]) == final_rel and row["label"] == label]


def _plot_per_view_rgb(render_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = render_dir / "plot_per_view_final_psnr.png"
    views = list(dict.fromkeys(str(row["view_id"]) for row in rows))
    x = range(len(views))
    width = 0.24
    plt.figure(figsize=(9.0, 4.8))
    for offset, branch in enumerate(base.BRANCHES):
        branch_rows = {str(row["view_id"]): row for row in rows if row["branch"] == branch}
        plt.bar(
            [i + (offset - 1) * width for i in x],
            [float(branch_rows[view]["PSNR"]) for view in views],
            width=width,
            label=branch,
        )
    plt.xticks(list(x), views)
    plt.ylabel("PSNR")
    plt.title("Final per-view RGB PSNR")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_per_view_region(render_dir: Path, rows: Sequence[Mapping[str, str]], label: str, filename: str) -> Path:
    path = render_dir / filename
    views = list(dict.fromkeys(row["view_id"] for row in rows))
    x = range(len(views))
    width = 0.24
    plt.figure(figsize=(9.0, 4.8))
    for offset, branch in enumerate(base.BRANCHES):
        branch_rows = {row["view_id"]: row for row in rows if row["branch"] == branch}
        plt.bar(
            [i + (offset - 1) * width for i in x],
            [float(branch_rows[view]["MSE"]) for view in views],
            width=width,
            label=branch,
        )
    plt.xticks(list(x), views)
    plt.ylabel("MSE")
    plt.title(f"Final per-view {label} MSE")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _update_visual_index(render_dir: Path, new_paths: Iterable[Path]) -> None:
    index_path = render_dir / "VISUAL_COMPARE_INDEX.md"
    existing: List[str] = []
    if index_path.exists():
        existing = index_path.read_text(encoding="utf8").splitlines()
    paths = []
    for line in existing:
        if line.startswith("- `") and line.endswith("`"):
            paths.append(line[3:-1])
    for path in new_paths:
        text = str(path)
        if text not in paths:
            paths.append(text)
    lines = ["# BND-AWARE-REFINE Visual Index", "", *[f"- `{path}`" for path in paths]]
    index_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    _write_json(render_dir / "manifest.json", {"rows": [{"file_path": path} for path in paths]})


def summarize(repo: Path) -> Dict[str, Any]:
    output_dir = repo / base.OUTPUT_DIR
    render_dir = repo / base.RENDER_DIR
    final_rel = _final_rel(output_dir)

    per_view_rows = _per_view_rgb_metrics(repo, output_dir, final_rel)
    _write_csv(output_dir / "per_view_rgb_metrics.csv", per_view_rows)
    _write_json(output_dir / "per_view_rgb_metrics.json", {"rows": per_view_rows})

    selection_rows = _selection_summary(output_dir)
    _write_csv(output_dir / "selection_guidance_lift_summary.csv", selection_rows)
    _write_json(output_dir / "selection_guidance_lift_summary.json", {"rows": selection_rows})

    render_paths: List[Path] = []
    render_paths.append(_plot_per_view_rgb(render_dir, per_view_rows))
    render_paths.append(
        _plot_per_view_region(
            render_dir,
            _final_region_by_view(output_dir, final_rel, "PERSISTENT_BND_HARD"),
            "PERSISTENT_BND_HARD",
            "plot_per_view_persistent_hard_mse_final.png",
        )
    )
    render_paths.append(
        _plot_per_view_region(
            render_dir,
            _final_region_by_view(output_dir, final_rel, "BND_HARD_CORE"),
            "BND_HARD_CORE",
            "plot_per_view_bnd_hard_core_mse_final.png",
        )
    )

    labels = [f"{row['branch']} {row['kind']}" for row in selection_rows]
    render_paths.append(
        render_dir / "plot_selection_guidance_lift.png"
    )
    _bar_plot(
        render_paths[-1],
        labels,
        [float(row["guidance_lift"]) for row in selection_rows],
        title="Mean selected-minus-candidate guidance lift",
        ylabel="guidance lift",
    )
    render_paths.append(
        render_dir / "plot_selection_base_score_lift.png"
    )
    _bar_plot(
        render_paths[-1],
        labels,
        [float(row["base_score_lift"]) for row in selection_rows],
        title="Mean selected-minus-candidate base-score lift",
        ylabel="base-score lift",
    )

    global_rows = _read_csv(output_dir / "global_rgb_metrics.csv")
    final_rows = [row for row in global_rows if int(row["relative_step"]) == final_rel]
    render_paths.append(render_dir / "plot_brightness_control_final_psnr.png")
    _bar_plot(
        render_paths[-1],
        [row["branch"] for row in final_rows],
        [float(row["PSNR"]) for row in final_rows],
        title="Final RGB PSNR by branch",
        ylabel="PSNR",
    )

    _update_visual_index(render_dir, render_paths)
    summary = {"final_relative_step": final_rel, "per_view_rgb_rows": len(per_view_rows), "selection_summary_rows": len(selection_rows)}
    _write_json(output_dir / "supplemental_visual_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(json.dumps(summarize(args.repo.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
