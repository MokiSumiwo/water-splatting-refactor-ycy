#!/usr/bin/env python
"""Summarize the Panama BND-UNORM run.

This wrapper reuses the existing staged-Panama summary scaffold, but maps the
comparison slots to the historical K1 baseline and the new absolute-loss UNORM
run. It keeps the earlier setup audit outputs intact and writes a merged
manifest plus a BND-UNORM summary alias for the final report.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def _load_stage_module():
    stage_path = Path(__file__).resolve().with_name("summarize_bnd_stage_panama.py")
    spec = importlib.util.spec_from_file_location("summarize_bnd_stage_panama_alias", stage_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {stage_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _flatten_setup_outputs(setup_manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    train_view_ids = ";".join(setup_manifest.get("train_view_ids", []))
    subset_ids = ";".join(setup_manifest.get("parameter_gradient_subset_view_ids", []))
    for file_path in setup_manifest.get("outputs", []):
        stem = Path(file_path).stem
        rows.append(
            {
                "file_path": file_path,
                "scene": "Panama",
                "run": "BND-UNORM-SETUP",
                "step": "audit",
                "output_type": stem,
                "view_ids": subset_ids if "fixed_state" in stem or "gradient" in stem else train_view_ids,
            }
        )
    return rows


def _flatten_stage_outputs(stage_manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for file_path in stage_manifest.get("metric_outputs", []):
        rows.append(
            {
                "file_path": file_path,
                "scene": "Panama",
                "run": "K1-RST;UNORM",
                "step": "all",
                "output_type": Path(file_path).stem,
                "view_ids": "",
            }
        )
    render_manifest = stage_manifest.get("render_manifest")
    if render_manifest:
        rows.append(
            {
                "file_path": render_manifest,
                "scene": "Panama",
                "run": "K1-RST;UNORM",
                "step": "15000",
                "output_type": "render_manifest",
                "view_ids": "",
            }
        )
    visual_index = stage_manifest.get("visual_index")
    if visual_index:
        rows.append(
            {
                "file_path": visual_index,
                "scene": "Panama",
                "run": "K1-RST;UNORM",
                "step": "15000",
                "output_type": "visual_index",
                "view_ids": "",
            }
        )
    return rows


def _append_alias_rows(rows: List[Dict[str, Any]], file_paths: Sequence[Path]) -> None:
    for path in file_paths:
        if path.exists():
            rows.append(
                {
                    "file_path": str(path),
                    "scene": "Panama",
                    "run": "K1-RST;UNORM",
                    "step": "15000",
                    "output_type": path.stem,
                    "view_ids": "",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bnd_unorm_panama_20260810"))
    parser.add_argument("--render-dir", type=Path, default=Path("renders/bnd_unorm_panama_20260810"))
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/bnd_stage_panama_20260810"),
        help="Historical stage logs reused by the stage summary scaffold.",
    )
    parser.add_argument("--tile-width", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    render_dir = (repo / args.render_dir).resolve() if not args.render_dir.is_absolute() else args.render_dir
    logs_dir = (repo / args.logs_dir).resolve() if not args.logs_dir.is_absolute() else args.logs_dir

    setup_manifest_path = output_dir / "setup_manifest.json"
    if not setup_manifest_path.exists():
        setup_manifest_path = output_dir / "manifest.json"
    setup_manifest = {}
    if setup_manifest_path.exists():
        setup_manifest = json.loads(setup_manifest_path.read_text())
        if setup_manifest_path.name != "setup_manifest.json":
            _copy_if_exists(setup_manifest_path, output_dir / "setup_manifest.json")
            _copy_if_exists(output_dir / "manifest.csv", output_dir / "setup_manifest.csv")

    stage = _load_stage_module()
    stage.RUNS = {
        "M1": stage.RunSpec(
            name="M1",
            config_relpath=(
                "outputs/cross_scene_panama_m1_seed42_15000/water-splatting/"
                "cross_scene_panama_m1_seed42_15000_20260730_cross_scene/config.yml"
            ),
            parameterization="legacy",
            role="reference_m1",
            reused=True,
        ),
        "K1-HIST": stage.RunSpec(
            name="K1-HIST",
            config_relpath=(
                "outputs/dewater_bounded_sh3_cross_scene_20260808/"
                "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
                "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
                "config.yml"
            ),
            parameterization="bounded_sh3",
            role="historical_reference",
            reused=True,
        ),
        "K1-RST": stage.RunSpec(
            name="K1-RST",
            config_relpath=(
                "outputs/dewater_bounded_sh3_cross_scene_20260808/"
                "dewater_bnd_cross_scene_panama_seed42_step0_to_15000/water-splatting/"
                "dewater_bnd_cross_scene_panama_seed42_step0_to_15000_20260808_bounded_sh3_cross_scene_full_panama_bnd_g1p00/"
                "config.yml"
            ),
            parameterization="bounded_sh3",
            role="matched_restart_control",
            reused=True,
        ),
        "STAGE": stage.RunSpec(
            name="STAGE",
            config_relpath=(
                "outputs/bnd_unorm_panama_20260810/"
                "panama_bnd_unorm_seed42_step0_to_15000/water-splatting/"
                "20260810_bnd_unorm/config.yml"
            ),
            parameterization="bounded_sh3",
            role="absolute_photometric_normalization_candidate",
            reused=False,
        ),
    }
    stage.START_SOURCE = stage.RUNS["K1-HIST"]
    stage.MAIN_RUNS = ("M1", "K1-HIST", "K1-RST", "STAGE")
    stage.FINAL_COMPARE_RUNS = ("M1", "K1-RST", "STAGE")
    stage.TRAJECTORY_STEPS = (1000, 3000, 5000, 8000, 10000, 13000, 15000)
    stage.PARAMETER_STEPS = stage.TRAJECTORY_STEPS
    stage._spec_and_config_for_step = lambda repo, run, nominal_step: (stage.RUNS[run], repo / stage.RUNS[run].config_relpath)  # type: ignore[attr-defined]
    stage._phase_for = lambda run, nominal_step: {  # type: ignore[attr-defined]
        "K1-HIST": "BND_K1_HISTORICAL_REFERENCE",
        "K1-RST": "BND_K1_BASELINE",
        "STAGE": "BND_UNORM_FROM_SCRATCH",
    }.get(run, "REFERENCE_FINAL")
    stage._write_phase_visual = lambda repo, render_dir, view_id, tile_width: []  # type: ignore[attr-defined]
    stage.restart_equivalence_audit = lambda repo: []  # type: ignore[assignment]
    stage.medium_parameter_audit = lambda repo: []  # type: ignore[assignment]
    stage.audit_log_rows = lambda repo, logs_dir: ([], [], [])  # type: ignore[assignment]

    stage.parse_args = lambda: argparse.Namespace(  # type: ignore[assignment]
        repo=repo,
        output_dir=output_dir,
        render_dir=render_dir,
        logs_dir=logs_dir,
        tile_width=args.tile_width,
    )
    stage.main()

    stage_manifest_path = output_dir / "manifest.json"
    stage_manifest = json.loads(stage_manifest_path.read_text()) if stage_manifest_path.exists() else {}
    stage_copy = output_dir / "bnd_stage_final_summary.json"
    alias_copy = output_dir / "bnd_unorm_final_summary.json"
    copied_json = _copy_if_exists(stage_copy, alias_copy)
    copied_csv = _copy_if_exists(output_dir / "bnd_stage_final_summary.csv", output_dir / "bnd_unorm_final_summary.csv")

    combined_rows: List[Dict[str, Any]] = []
    if setup_manifest:
        combined_rows.extend(_flatten_setup_outputs(setup_manifest))
    if stage_manifest:
        combined_rows.extend(_flatten_stage_outputs(stage_manifest))
    _append_alias_rows(
        combined_rows,
        [
            alias_copy,
            output_dir / "bnd_unorm_final_summary.csv",
        ],
    )

    combined_manifest = {
        "branch": stage_manifest.get("branch", ""),
        "head": stage_manifest.get("head", ""),
        "repo": str(repo),
        "setup_audit_manifest": setup_manifest,
        "stage_summary_manifest": stage_manifest,
        "aliases": {
            "bnd_stage_final_summary.json": "bnd_unorm_final_summary.json",
            "bnd_stage_final_summary.csv": "bnd_unorm_final_summary.csv",
            "K1-RST": "historical K1 baseline",
            "STAGE": "BND-UNORM absolute-loss run",
        },
        "outputs": combined_rows,
        "final_summary_aliases_created": {
            "json": copied_json,
            "csv": copied_csv,
        },
    }
    _write_json(output_dir / "manifest.json", combined_manifest)
    _write_csv(output_dir / "manifest.csv", combined_rows)

    visual_index_path = render_dir / "VISUAL_COMPARE_INDEX.md"
    existing_visual_index = visual_index_path.read_text() if visual_index_path.exists() else ""
    extra_lines = [
        "",
        "## Supplementary Controls",
        "",
        f"- Responsibility overlay: `{(repo / 'renders/lossresp_panama_20260810/contact_sheet_responsibility_overlay.png').resolve()}`",
        f"- Formal gradients: `{(repo / 'renders/lossresp_panama_20260810/contact_sheet_formal_image_gradients.png').resolve()}`",
        f"- Brightness Q5: `{(repo / 'renders/bnd_hr_panama_20260810/contact_sheet_brightness_q5_k1_hr.png').resolve()}`",
        f"- Boundary pressure: `{(repo / 'renders/bnd_hr_panama_20260810/contact_sheet_boundary_pressure_hr.png').resolve()}`",
    ]
    if "## Supplementary Controls" not in existing_visual_index:
        visual_index_path.write_text(existing_visual_index.rstrip() + "\n" + "\n".join(extra_lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
