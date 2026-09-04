#!/usr/bin/env python3
"""Aggregate the formal four-scene MDRR/CICA matrix and native render audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
AUX_ROOT = REPO_ROOT / "outputs" / "identifiability_module_causal_iui3_20260902"
DEFAULT_ROOT = REPO_ROOT / "outputs" / "direct_mdrr_cica_four_scene_20260903"
A0_RENDERER = REPO_ROOT / "scripts" / "diagnostics" / "render_direct_mdrr_cica_a0.py"
PYTHON = Path("/opt/anaconda3/envs/water_splatting/bin/python")
SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
ARMS = ("A0", "A1", "A2", "A3")
FORMAL_ARMS = ("A1", "A2", "A3")
METRICS = ("PSNR", "SSIM", "LPIPS", "MSE")
COLOR_FIELDS = (
    "clear_raw_mean_r",
    "clear_raw_mean_g",
    "clear_raw_mean_b",
    "clear_raw_p99",
    "clear_raw_blue_minus_red",
    "clear_raw_blue_minus_green",
)
FINAL_STEP = 14999
START_STEP = 3000
SNAPSHOT_STEPS = (5000, 8000, 10000, 13000, 14999)
DELTA_PAIRS = (
    ("A1_minus_A0", "A1", "A0"),
    ("A2_minus_A0", "A2", "A0"),
    ("A3_minus_A0", "A3", "A0"),
    ("A3_minus_A1", "A3", "A1"),
    ("A3_minus_A2", "A3", "A2"),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _num(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"expected a finite numeric value, got {value!r}")
    if not math.isfinite(result):
        raise RuntimeError(f"expected a finite numeric value, got {value!r}")
    return result


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        raise RuntimeError("cannot average an empty sequence")
    return sum(values) / len(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(list(rows), sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def _metric_row(rows: Sequence[Mapping[str, Any]], arm: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("arm") == arm
        and int(row["absolute_step"]) == FINAL_STEP
        and row.get("split") == "eval"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one eval metric row for {arm}@{FINAL_STEP}, got {len(matches)}")
    return matches[0]


def _historical_a0(scene: str) -> Dict[str, Any]:
    root = AUX_ROOT / scene
    checkpoint = root / "checkpoints" / "C1" / f"step-{FINAL_STEP:09d}.ckpt"
    required = (
        root / "evaluation_metrics.csv",
        root / "per_view_metrics.csv",
        root / "topology_metrics.csv",
        root / "decomposition_safety.json",
        root / "start_state_equivalence.json",
        root / "camera_sequence.json",
        checkpoint,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"historical A0 artifacts missing: {missing}")
    return {
        "dir": root,
        "metrics": _read_csv(root / "evaluation_metrics.csv"),
        "per_view": _read_csv(root / "per_view_metrics.csv"),
        "topology": _read_csv(root / "topology_metrics.csv"),
        "decomp": _read_json(root / "decomposition_safety.json"),
        "start": _read_json(root / "start_state_equivalence.json"),
        "sequence": _read_json(root / "camera_sequence.json"),
        "checkpoint": checkpoint,
    }


def _formal_tables(root: Path, scene: str, arm: str) -> Dict[str, Any]:
    scene_dir = root / scene / arm
    names = (
        "training_summary.json",
        f"training_summary_{arm}.csv",
        "evaluation_metrics.csv",
        "per_view_metrics.csv",
        "topology_statistics.csv",
        "decomposition_safety.json",
        "gradient_routing_audit.json",
        "start_state_equivalence.json",
        "camera_sequence_hashes.json",
        "mdrr_partner_mapping.json",
        "cica_camera_bank.json",
        "render_manifest.csv",
        "module_statistics.csv",
    )
    missing = [name for name in names if not (scene_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"formal artifacts missing for {scene}/{arm}: {missing}")
    return {
        "dir": scene_dir,
        "training": _read_json(scene_dir / "training_summary.json"),
        "training_rows": _read_csv(scene_dir / f"training_summary_{arm}.csv"),
        "metrics": _read_csv(scene_dir / "evaluation_metrics.csv"),
        "per_view": _read_csv(scene_dir / "per_view_metrics.csv"),
        "topology": _read_csv(scene_dir / "topology_statistics.csv"),
        "decomp": _read_json(scene_dir / "decomposition_safety.json"),
        "routing": _read_json(scene_dir / "gradient_routing_audit.json"),
        "start": _read_json(scene_dir / "start_state_equivalence.json"),
        "sequence": _read_json(scene_dir / "camera_sequence_hashes.json"),
        "partner": _read_json(scene_dir / "mdrr_partner_mapping.json"),
        "bank": _read_json(scene_dir / "cica_camera_bank.json"),
        "renders": _read_csv(scene_dir / "render_manifest.csv"),
        "modules": _read_csv(scene_dir / "module_statistics.csv"),
    }


def _aggregate_protocol(root: Path, tables: Mapping[Tuple[str, str], Mapping[str, Any]]) -> Dict[str, Any]:
    start_rows: List[Dict[str, Any]] = []
    sequence_rows: List[Dict[str, Any]] = []
    partner_rows: List[Dict[str, Any]] = []
    bank_rows: List[Dict[str, Any]] = []
    all_start_pass = True
    all_sequence_pass = True
    all_partner_pass = True
    all_bank_pass = True
    for scene in SCENES:
        a0 = tables[(scene, "A0")]
        historical_rows = a0["sequence"]["rows"]
        historical_hash = _rows_hash(historical_rows)
        starts = [tables[(scene, arm)]["start"] for arm in FORMAL_ARMS]
        comparable = (
            "checkpoint_model_hash",
            "optimizer_hash",
            "scheduler_hash",
            "scaler_hash",
            "training_rng_manifest",
            "source_checkpoint_sha256",
        )
        hashes_equal = all(len({json.dumps(item[key], sort_keys=True) for item in starts}) == 1 for key in comparable)
        historical_source_match = all(
            item["source_checkpoint_sha256"] == a0["start"]["common_checkpoint_sha256"] for item in starts
        )
        historical_rng_match = all(
            item["training_rng_manifest"] == a0["start"]["continuation_rng_manifest"] for item in starts
        )
        scene_start_pass = bool(
            all(item.get("START_STATE_EQUIVALENCE") is True for item in starts)
            and hashes_equal
            and historical_source_match
            and historical_rng_match
        )
        all_start_pass &= scene_start_pass
        for arm, item in zip(FORMAL_ARMS, starts):
            start_rows.append(
                {
                    "scene": scene,
                    "configuration": arm,
                    "pass": scene_start_pass,
                    **{key: item[key] for key in comparable},
                }
            )
        formal_sequence_hashes = {}
        for arm in FORMAL_ARMS:
            sequence = tables[(scene, arm)]["sequence"]
            sequence_hash = _rows_hash(sequence["rows"])
            formal_sequence_hashes[arm] = sequence_hash
            sequence_rows.append(
                {
                    "scene": scene,
                    "configuration": arm,
                    "row_count": len(sequence["rows"]),
                    "canonical_rows_sha256": sequence_hash,
                    "matches_historical_A0": sequence_hash == historical_hash,
                }
            )
        scene_sequence_pass = bool(
            len(set(formal_sequence_hashes.values())) == 1
            and next(iter(formal_sequence_hashes.values())) == historical_hash
            and len(historical_rows) == FINAL_STEP - START_STEP
        )
        all_sequence_pass &= scene_sequence_pass
        partner_hashes = {arm: tables[(scene, arm)]["partner"]["mapping_sha256"] for arm in ("A1", "A3")}
        bank_hashes = {arm: tables[(scene, arm)]["bank"]["bank_sha256"] for arm in ("A2", "A3")}
        partner_pass = len(set(partner_hashes.values())) == 1
        bank_pass = len(set(bank_hashes.values())) == 1
        all_partner_pass &= partner_pass
        all_bank_pass &= bank_pass
        partner_rows.extend(
            {"scene": scene, "configuration": arm, "mapping_sha256": value, "A1_A3_match": partner_pass}
            for arm, value in partner_hashes.items()
        )
        bank_rows.extend(
            {"scene": scene, "configuration": arm, "bank_sha256": value, "A2_A3_match": bank_pass}
            for arm, value in bank_hashes.items()
        )
    start_payload = {"START_STATE_EQUIVALENCE": all_start_pass, "rows": start_rows}
    sequence_payload = {"CAMERA_SEQUENCE_EXACT_MATCH": all_sequence_pass, "rows": sequence_rows}
    partner_payload = {"A1_A3_PARTNER_MAPPING_MATCH": all_partner_pass, "rows": partner_rows}
    bank_payload = {"A2_A3_CICA_BANK_MATCH": all_bank_pass, "rows": bank_rows}
    _write_json(root / "start_state_equivalence.json", start_payload)
    _write_json(root / "camera_sequence_hashes.json", sequence_payload)
    _write_json(root / "mdrr_partner_mapping.json", partner_payload)
    _write_json(root / "cica_camera_bank.json", bank_payload)
    manifest = {
        "experiment": "DIRECT_TRAINING_MDRR_CICA_AND_COMBINED_FOUR_SCENE",
        "historical_base": "A0 = OCMC + auxiliary appearance regularization",
        "arms": {"A1": "A0 + MDRR", "A2": "A0 + CICA", "A3": "A0 + MDRR + CICA"},
        "scenes": list(SCENES),
        "start_step": START_STEP,
        "final_step": FINAL_STEP,
        "matched_updates": FINAL_STEP - START_STEP,
        "snapshots": list(SNAPSHOT_STEPS),
        "seed": 42,
        "ocmc_on": True,
        "raoc_off": True,
        "no_parameter_or_seed_sweep": True,
        "protocol_pass": all_start_pass and all_sequence_pass and all_partner_pass and all_bank_pass,
    }
    _write_json(root / "experiment_manifest.json", manifest)
    return {"manifest": manifest, "start": start_payload, "sequence": sequence_payload, "partner": partner_payload, "bank": bank_payload}


def _collect(root: Path) -> Dict[str, Any]:
    tables: Dict[Tuple[str, str], Dict[str, Any]] = {}
    values: Dict[Tuple[str, str, str], float] = {}
    per_view_rows: List[Dict[str, Any]] = []
    module_rows: List[Dict[str, Any]] = []
    topology_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []
    color_rows: List[Dict[str, Any]] = []
    training_rows: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in FORMAL_ARMS}
    for scene in SCENES:
        tables[(scene, "A0")] = _historical_a0(scene)
        for arm in FORMAL_ARMS:
            tables[(scene, arm)] = _formal_tables(root, scene, arm)
        for configuration in ARMS:
            table = tables[(scene, configuration)]
            source_arm = "C1" if configuration == "A0" else configuration
            metric = _metric_row(table["metrics"], source_arm)
            for name in METRICS:
                values[(scene, configuration, name)] = _num(metric[name])
            selected_views = [
                row
                for row in table["per_view"]
                if row.get("arm") == source_arm
                and row.get("absolute_step") == str(FINAL_STEP)
                and row.get("split") == "eval"
            ]
            if not selected_views:
                raise RuntimeError(f"no final heldout views for {scene}/{configuration}")
            for row in selected_views:
                normalized = {
                    "scene": scene,
                    "configuration": configuration,
                    "absolute_step": FINAL_STEP,
                    "split": "eval",
                    "view_id": row["view_id"],
                    **{name: _num(row[name]) for name in METRICS},
                }
                per_view_rows.append(normalized)
                color_values = {name: _num(row[name]) for name in COLOR_FIELDS if row.get(name, "") != ""}
                if color_values:
                    color_rows.append({"scene": scene, "configuration": configuration, "view_id": row["view_id"], **color_values})
            selected_topology = [row for row in table["topology"] if row.get("arm") == source_arm and row.get("absolute_step") == str(FINAL_STEP)]
            if len(selected_topology) != 1:
                raise RuntimeError(f"expected one final topology row for {scene}/{configuration}")
            topology_rows.append({"scene": scene, "configuration": configuration, **selected_topology[0]})
            selected_decomp = [
                row
                for row in table["decomp"].get("rows", [])
                if int(row.get("absolute_step", -1)) == FINAL_STEP
                and row.get("split") == "eval"
                and row.get("branch", row.get("arm")) == source_arm
            ]
            if len(selected_decomp) != 1:
                raise RuntimeError(f"expected one final decomposition row for {scene}/{configuration}")
            decomp_rows.append({"scene": scene, "configuration": configuration, **selected_decomp[0]})
            if configuration != "A0":
                module_rows.extend({"scene": scene, "configuration": configuration, **row} for row in table["modules"])
                training_rows[configuration].extend(
                    {"scene": scene, "configuration": configuration, **row}
                    for row in table["training_rows"]
                )
    final_rows: List[Dict[str, Any]] = []
    for scene in (*SCENES, "MEAN"):
        for metric in METRICS:
            row: Dict[str, Any] = {"scene": scene, "metric": metric, "absolute_step": FINAL_STEP}
            for arm in ARMS:
                row[arm] = values[(scene, arm, metric)] if scene != "MEAN" else _mean(values[(item, arm, metric)] for item in SCENES)
            for label, left, right in DELTA_PAIRS:
                row[label] = row[left] - row[right]
            final_rows.append(row)
    per_view_index = {(row["scene"], row["configuration"], row["view_id"]): row for row in per_view_rows}
    per_view_deltas: List[Dict[str, Any]] = []
    for scene in SCENES:
        view_ids = sorted(row["view_id"] for row in per_view_rows if row["scene"] == scene and row["configuration"] == "A0")
        for view_id in view_ids:
            for label, left, right in DELTA_PAIRS:
                left_row = per_view_index[(scene, left, view_id)]
                right_row = per_view_index[(scene, right, view_id)]
                per_view_deltas.append(
                    {
                        "scene": scene,
                        "configuration": label,
                        "absolute_step": FINAL_STEP,
                        "split": "delta_eval",
                        "view_id": view_id,
                        **{metric: left_row[metric] - right_row[metric] for metric in METRICS},
                    }
                )
    protocol = _aggregate_protocol(root, tables)
    decomposition = {
        "pass": all(_num(row["P_J_gt_1"]) == 0.0 for row in decomp_rows),
        "requirement": "P(J > 1) = 0 for every final heldout configuration",
        "rows": decomp_rows,
    }
    _write_csv(root / "final_metrics.csv", final_rows)
    _write_csv(root / "per_view_metrics.csv", per_view_rows + per_view_deltas)
    _write_csv(root / "module_statistics.csv", module_rows)
    _write_csv(root / "topology_statistics.csv", topology_rows)
    _write_csv(root / "clear_color_statistics.csv", color_rows)
    for arm, arm_rows in training_rows.items():
        _write_csv(root / f"training_summary_{arm}.csv", arm_rows)
    _write_json(root / "decomposition_safety.json", decomposition)
    return {
        "tables": tables,
        "values": values,
        "final_rows": final_rows,
        "per_view": per_view_rows,
        "modules": module_rows,
        "topology": topology_rows,
        "decomposition": decomposition,
        "colors": color_rows,
        "training_rows": training_rows,
        "protocol": protocol,
    }


def _interaction(root: Path, collected: Mapping[str, Any]) -> Dict[str, Any]:
    values = collected["values"]
    scene_rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        row: Dict[str, Any] = {"scene": scene}
        for metric in METRICS:
            for label, left, right in DELTA_PAIRS:
                row[f"{metric}_{label}"] = values[(scene, left, metric)] - values[(scene, right, metric)]
        scene_rows.append(row)
    mean_rows = [row for row in collected["final_rows"] if row["scene"] == "MEAN"]
    mean_by_metric = {row["metric"]: row for row in mean_rows}
    positive = {
        arm: sum(values[(scene, arm, "PSNR")] - values[(scene, "A0", "PSNR")] >= 0.0 for scene in SCENES)
        for arm in FORMAL_ARMS
    }
    non_worsening = {
        arm: sum(values[(scene, arm, "PSNR")] - values[(scene, "A0", "PSNR")] >= -0.05 for scene in SCENES)
        for arm in FORMAL_ARMS
    }
    mean_psnr_delta = {arm: mean_by_metric["PSNR"][arm] - mean_by_metric["PSNR"]["A0"] for arm in FORMAL_ARMS}
    a1, a2, a3 = (mean_by_metric["PSNR"][arm] for arm in FORMAL_ARMS)
    if a3 > a1 + 0.05 and a3 > a2 + 0.05:
        interaction = "SYNERGISTIC"
    elif a3 < a1 and a3 < a2:
        interaction = "INTERFERING"
    elif abs(a3 - max(a1, a2)) <= 0.05:
        interaction = "REDUNDANT"
    else:
        interaction = "COMPLEMENTARY"
    protocol_safe = bool(collected["protocol"]["manifest"]["protocol_pass"])
    routing_safe = all(
        tables["routing"].get("GRADIENT_ROUTING_AUDIT") is True
        for (_scene, arm), tables in collected["tables"].items()
        if arm in FORMAL_ARMS
    )
    decomposition_safe = bool(collected["decomposition"]["pass"])
    quantitative = {
        "MDRR": "KEEP" if positive["A1"] >= 3 and mean_psnr_delta["A1"] > 0.0 and protocol_safe and routing_safe and decomposition_safe else "DROP",
        "CICA": "KEEP" if non_worsening["A2"] >= 3 and protocol_safe and routing_safe and decomposition_safe else "DROP",
        "COMBINED": "KEEP" if mean_psnr_delta["A3"] >= 0.0 and protocol_safe and routing_safe and decomposition_safe else "DROP",
        "Interaction": interaction,
        "Final configuration": max(ARMS, key=lambda arm: mean_by_metric["PSNR"][arm]),
    }
    clear_gt_audit = {
        "paired_real_clear_gt_found": False,
        "quantitative_clear_metrics_reported": False,
        "scenes_checked": list(SCENES),
        "dataset_roots": [str(REPO_ROOT / "undistorted_data" / f"undistorted_{scene}") for scene in SCENES],
        "available_image_directories": [
            "images/ColorImage",
            "images/fake_air_Deep_Sea-NN",
            "images/fake_air_Deep_Sea-NN_new",
        ],
        "finding": "ColorImage is the underwater input; fake_air directories are derived outputs and are not paired real clear ground truth.",
    }
    _write_json(root / "clear_gt_audit.json", clear_gt_audit)
    payload = {
        "scene_rows": scene_rows,
        "mean_rows": mean_rows,
        "positive_scene_count_PSNR": positive,
        "non_worsening_scene_count_PSNR_at_minus_0p05dB": non_worsening,
        "mean_PSNR_delta_vs_A0": mean_psnr_delta,
        "best_mean_configuration": {
            "PSNR": max(ARMS, key=lambda arm: mean_by_metric["PSNR"][arm]),
            "SSIM": max(ARMS, key=lambda arm: mean_by_metric["SSIM"][arm]),
            "LPIPS": min(ARMS, key=lambda arm: mean_by_metric["LPIPS"][arm]),
            "MSE": min(ARMS, key=lambda arm: mean_by_metric["MSE"][arm]),
        },
        "protocol_safe": protocol_safe,
        "routing_audit_safe": routing_safe,
        "decomposition_safe": decomposition_safe,
        "interaction_classification": interaction,
        "quantitative_classification_before_required_clear_visual_review": quantitative,
        "classification": quantitative,
        "clear_gt": clear_gt_audit,
        "clear_render_interpretation": "Native clear rendering and color-distribution diagnostics only; less blue is not automatically better.",
        "qualitative_clear_review_required_for_final_CICA_and_combined_decisions": True,
    }
    _write_json(root / "interaction_analysis.json", payload)
    _write_json(root / "final_classification.json", quantitative)
    return payload


def _select_views(per_view: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    selected: Dict[str, str] = {}
    for scene in SCENES:
        rows = sorted(
            (row for row in per_view if row["scene"] == scene and row["configuration"] == "A0" and row["split"] == "eval"),
            key=lambda row: row["view_id"],
        )
        if not rows:
            raise RuntimeError(f"no A0 heldout views for {scene}")
        selected[scene] = rows[len(rows) // 2]["view_id"]
    return selected


def _render_a0(root: Path, scene: str, gpu: str) -> Dict[str, Any]:
    output = root / "a0_renders" / scene
    manifest_path = output / "render_manifest.json"
    if not manifest_path.is_file():
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["CONDA_DEFAULT_ENV"] = "water_splatting"
        result = subprocess.run(
            [str(PYTHON), str(A0_RENDERER), "--scene", scene, "--output-dir", str(output)],
            cwd=str(REPO_ROOT), env=environment, text=True, capture_output=True,
        )
        if result.returncode:
            raise RuntimeError(f"A0 render failed for {scene}: {result.stderr[-4000:]}")
    manifest = _read_json(manifest_path)
    if manifest.get("scene") != scene or int(manifest.get("absolute_step", -1)) != FINAL_STEP:
        raise RuntimeError(f"invalid cached A0 render manifest: {manifest_path}")
    for row in manifest["rows"]:
        path = Path(row["path"])
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"A0 render provenance failure: {path}")
    return manifest


def _formal_render(table: Mapping[str, Any], view_id: str, output_type: str) -> Dict[str, Any]:
    matches = [
        row for row in table["renders"]
        if row.get("view_id") == view_id and row.get("output_type") == output_type and int(row.get("absolute_step", -1)) == FINAL_STEP
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {output_type} render for {view_id}, got {len(matches)}")
    path = Path(matches[0]["path"])
    checkpoint = table["dir"] / "checkpoints" / f"step-{FINAL_STEP:09d}.ckpt"
    if not path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(path if not path.is_file() else checkpoint)
    return {**matches[0], "path": str(path), "sha256": _sha256(path), "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint)}


def _make_figure(path: Path, rows: Sequence[Sequence[Tuple[str, Path]]]) -> None:
    tile_w, tile_h, label_h = 320, 240, 30
    canvas = Image.new("RGB", (tile_w * 5, (tile_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, files in enumerate(rows):
        for column, (label, source_path) in enumerate(files):
            with Image.open(source_path) as source:
                rendered = source.convert("RGB")
                rendered.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (tile_w, tile_h), "black")
                tile.paste(rendered, ((tile_w - rendered.width) // 2, (tile_h - rendered.height) // 2))
            x = column * tile_w
            y = row_index * (tile_h + label_h)
            canvas.paste(tile, (x, y + label_h))
            draw.text((x + 6, y + 7), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _figures(root: Path, collected: Mapping[str, Any], render_gpu: str) -> Dict[str, Any]:
    selected = _select_views(collected["per_view"])
    clear_rows: List[List[Tuple[str, Path]]] = []
    underwater_rows: List[List[Tuple[str, Path]]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        view_id = selected[scene]
        a0_manifest = _render_a0(root, scene, render_gpu)
        a0_index = {(row["view_id"], row["output_type"]): row for row in a0_manifest["rows"]}
        a0_color_keys = {(row["scene"], row["configuration"], row["view_id"]) for row in collected["colors"]}
        for row in a0_manifest["rows"]:
            color_key = (scene, "A0", row["view_id"])
            if row["output_type"] == "clear_raw_display_clamp01" and color_key not in a0_color_keys:
                collected["colors"].append(
                    {
                        "scene": scene,
                        "configuration": "A0",
                        "view_id": row["view_id"],
                        **{name: _num(row[name]) for name in COLOR_FIELDS},
                    }
                )
                a0_color_keys.add(color_key)
        input_row = a0_index[(view_id, "input")]
        a0_clear = a0_index[(view_id, "clear_native")]
        a0_underwater = a0_index[(view_id, "underwater")]
        clear_files = [(f"{scene} | Input", Path(input_row["path"])), (f"{scene} | A0", Path(a0_clear["path"]))]
        underwater_files = [(f"{scene} | Input", Path(input_row["path"])), (f"{scene} | A0", Path(a0_underwater["path"]))]
        for arm, label in (("A1", "+MDRR"), ("A2", "+CICA"), ("A3", "+MDRR+CICA")):
            table = collected["tables"][(scene, arm)]
            clear = _formal_render(table, view_id, "clear_native")
            underwater = _formal_render(table, view_id, "underwater")
            clear_files.append((f"{scene} | {label}", Path(clear["path"])))
            underwater_files.append((f"{scene} | {label}", Path(underwater["path"])))
            manifest_rows.extend(
                [
                    {"scene": scene, "view_id": view_id, "configuration": arm, "output_type": "clear_native", "source_image": input_row.get("source_image"), **clear},
                    {"scene": scene, "view_id": view_id, "configuration": arm, "output_type": "underwater", "source_image": input_row.get("source_image"), **underwater},
                ]
            )
        clear_rows.append(clear_files)
        underwater_rows.append(underwater_files)
        manifest_rows.extend([{**input_row, "configuration": "INPUT"}, {**a0_clear, "configuration": "A0"}, {**a0_underwater, "configuration": "A0"}])
    clear_png = root / "mdrr_cica_rgb_comparison.png"
    clear_pdf = root / "mdrr_cica_rgb_comparison.pdf"
    underwater_png = root / "mdrr_cica_underwater_reconstruction.png"
    _make_figure(clear_png, clear_rows)
    _make_figure(underwater_png, underwater_rows)
    with Image.open(clear_png) as image:
        image.convert("RGB").save(clear_pdf, "PDF", resolution=100.0)
    manifest = {
        "layout": "4 rows x 5 columns",
        "required_figure_semantics": "underwater input followed by native clear/dewatered A0/A1/A2/A3",
        "underwater_safety_figure": str(underwater_png),
        "selected_views": selected,
        "selection_rule": "deterministic median after sorting historical A0 heldout view IDs",
        "paired_real_clear_gt_found": False,
        "postprocessing": "resize, letterbox layout, label, and border only",
        "rows": manifest_rows,
    }
    _write_json(root / "rgb_comparison_manifest.json", manifest)
    _write_csv(root / "clear_color_statistics.csv", collected["colors"])
    return {"clear_png": str(clear_png), "clear_pdf": str(clear_pdf), "underwater_png": str(underwater_png), "manifest": str(root / "rgb_comparison_manifest.json"), "selected_views": selected}


def _format_metric_table(rows: Sequence[Mapping[str, Any]], metric: str) -> List[str]:
    lines = ["| Scene | A0 | A1 | A2 | A3 | A1-A0 | A2-A0 | A3-A0 | A3-A1 | A3-A2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["metric"] == metric:
            lines.append("| {scene} | {A0:.6f} | {A1:.6f} | {A2:.6f} | {A3:.6f} | {A1_minus_A0:+.6f} | {A2_minus_A0:+.6f} | {A3_minus_A0:+.6f} | {A3_minus_A1:+.6f} | {A3_minus_A2:+.6f} |".format(**row))
    return lines


def _research_note(root: Path, collected: Mapping[str, Any], summary: Mapping[str, Any], figures: Mapping[str, Any]) -> None:
    note = REPO_ROOT / "research_notes" / "DIRECT_MDRR_CICA_FOUR_SCENE_EXPERIMENT_2026-09-03.md"
    lines = [
        "# Direct MDRR/CICA Four-Scene Experiment",
        "",
        "## Current Formal Baseline",
        "A0 is the historical OCMC-on, RAOC-off continuation with `A_DETACHED_SH_OPACITY_TANGENT_ORTHOGONALIZATION`. Its prior causal audit favored a generic regularization effect, so it remains auxiliary appearance regularization and is not claimed as an identifiability innovation.",
        "",
        "## Configurations And Matching",
        "A1 is A0 plus MDRR, A2 is A0 plus CICA, and A3 is A0 plus both. Each arm starts from the same scene-specific OCMC C0 checkpoint at step 3000, restores identical model/optimizer/scheduler/scaler state and seed-42 continuation RNG, consumes the same 11,999-camera sequence, and ends at step 14999. OCMC remains locked and RAOC remains disabled.",
        "",
        f"Start-state equivalence: `{summary['protocol_safe']}`. Camera sequence exact match: `{collected['protocol']['sequence']['CAMERA_SEQUENCE_EXACT_MATCH']}`. Partner mapping match: `{collected['protocol']['partner']['A1_A3_PARTNER_MAPPING_MATCH']}`. CICA bank match: `{collected['protocol']['bank']['A2_A3_CICA_BANK_MATCH']}`.",
        "",
        "## MDRR Implementation",
        "MDRR activates at step 5000. Every active update uses its fixed training-camera partner and current model state. Exact classic responsibilities form cross-view residual differences and the full medium response: direct attenuation, finite medium, and tail medium. Detached positive cosine responsibility forms `g_p`; appearance receives `(1-g_p)`, medium receives `g_p`, and geometry/opacity retain the base gradient.",
        "",
        "## CICA Implementation",
        "CICA activates at step 10000 and refreshes at 10000, 12000, and 14000 from at most six deterministic training cameras. A read-only CUDA accumulator follows classic alpha threshold and early termination to compute the DC-logit Jacobian normal equation. Gaussians with at least three views receive an information-weighted median detached log-chroma target. Huber loss acts only on `features_dc`; scale is calibrated once to 10% of first-activation photometric DC gradient norm.",
        "",
        "## Heldout RGB Results",
    ]
    for metric in METRICS:
        lines.extend(["", f"### {metric}", "", *_format_metric_table(collected["final_rows"], metric)])
    lines.extend(
        [
            "",
            "Per-view final values and all five paired deltas are in `per_view_metrics.csv`.",
            "",
            "## Clear Rendering And Underwater Safety",
            "No paired real clear ground truth was found. Clear conclusions are qualitative and distributional only; less blue is not automatically better. Panels are native outputs with no white balance, contrast, saturation, histogram matching, gamma change, or manual dehazing.",
            "",
            f"Native clear comparison: `{figures['clear_png']}` and `{figures['clear_pdf']}`. Underwater safety comparison: `{figures['underwater_png']}`. Raw clear summaries are in `clear_color_statistics.csv`.",
            "",
            "## Decomposition And Topology",
            f"Final heldout decomposition safety pass: `{summary['decomposition_safe']}`. Final populations and split/duplicate/prune/reset counts are in `topology_statistics.csv`; no module-specific topology rule was introduced.",
            "",
            "## Interaction And Recommendation",
            f"Quantitative interaction classification: `{summary['interaction_classification']}`. The preregistered RGB/safety decision is in `final_classification.json`; CICA and combined retention additionally require native clear visual review.",
            "",
            "## Limitations",
            "This is one fixed four-scene, one-seed protocol with no sweep or rescue. Clear rendering has no paired reference, and qualitative preference cannot establish physical correctness. The auxiliary appearance regularizer remains baseline-only and cannot be presented as the second innovation.",
        ]
    )
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines) + "\n", encoding="utf8")


def run(root: Path, render_gpu: str) -> Dict[str, Any]:
    collected = _collect(root)
    summary = _interaction(root, collected)
    figures = _figures(root, collected, render_gpu)
    summary["figures"] = figures
    _write_json(root / "final_summary.json", summary)
    _research_note(root, collected, summary, figures)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--render-gpu", choices=("6", "7", "8", "9"), default="6")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root.resolve(), args.render_gpu), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
