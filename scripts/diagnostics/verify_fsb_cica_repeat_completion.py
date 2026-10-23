#!/usr/bin/env python3
"""Recheck retained repeat artifacts and plot effect against execution variation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")
ARMS = ("C0a", "C1a", "C1b", "C0b")
METRICS = ("PSNR", "SSIM", "LPIPS")


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = root / "summary_20260916"
    status = read_json(root / "status.json")
    assert not status["active"] and not status["failed"] and not any(status["pending"].values())
    assert set(status["completed"]) == set(SCENES)
    expected_jobs = {r + "/" + a for r in ("R0", "R1") for a in ("PREFIX",) + ARMS}
    assert all(set(v) == expected_jobs and all(c == 0 for c in v.values()) for v in status["completed"].values())
    checkpoints = read_csv(summary / "retained_checkpoints.csv")
    assert len(checkpoints) == 40 and len({r["path"] for r in checkpoints}) == 40
    for row in checkpoints:
        path = Path(row["path"])
        assert path.is_file() and sha(path) == row["sha256"], path
    provenance = read_json(summary / "provenance_sha256.json")
    artifact_count = 0
    for group in provenance.values():
        for name, digest in group.items():
            assert sha(Path(name)) == digest, name
            artifact_count += 1
    archive = root / "provenance"
    repo = Path(__file__).resolve().parents[2]
    archives_checked = 0
    for name, digest in read_json(archive / "sha256.json").items():
        source = Path(name)
        path = archive / "binaries" / source.name if source.suffix == ".so" else archive / source.relative_to(repo)
        assert sha(path) == digest, path
        archives_checked += 1
    old_contrasts = {(r["scene"], r["renderer"], r["metric"]): r for r in read_csv(summary / "contrasts.csv")}
    rows, evaluated_views = [], 0
    for scene_index, scene in enumerate(SCENES):
        for renderer in ("R0", "R1"):
            values, restore_states, view_ids = {}, [], []
            for arm in ARMS:
                path = root / scene / renderer / arm
                audit = read_json(path / "resume_state_audit.json")
                assert audit["all_pass"] and audit["absolute_step"] == 9999
                assert all(audit["state_matches"].values()) and all(audit["optimizer_parameter_binding"].values())
                restore_states.append(audit["hashes"])
                final = read_json(path / "final_summary.json")
                assert (final["first_step"], final["final_step"], final["completed_updates"]) == (10000, 14999, 5000)
                assert final["checkpoint_retained"] and final["ocmc_projector_unchanged"]
                manifest = read_json(path / "experiment_manifest.json")
                assert manifest["runtime"]["physical_gpu"] == str(6 + scene_index)
                assert manifest["runtime"]["python"] == "/opt/anaconda3/envs/water_splatting/bin/python"
                assert manifest["module_schedule"]["CICA"]["derivative_mode"] == ("exact_fsb" if arm.startswith("C1") else "legacy")
                views = read_csv(path / "per_view_metrics.csv")
                assert len(views) == (4 if scene == "IUI3-RedSea" else 3)
                view_ids.append(sorted(r["view_id"] for r in views))
                evaluated_views += len(views)
                values[arm] = {m: mean(float(v[m]) for v in views) for m in METRICS}
                global_row, = read_csv(path / "evaluation_metrics.csv")
                assert all(math.isclose(values[arm][m], float(global_row[m]), abs_tol=1e-12) for m in METRICS)
            assert all(s == restore_states[0] for s in restore_states)
            assert all(v == view_ids[0] for v in view_ids)
            for m in METRICS:
                c0 = mean(values[a][m] for a in ("C0a", "C0b"))
                c1 = mean(values[a][m] for a in ("C1a", "C1b"))
                gap0 = abs(values["C0a"][m] - values["C0b"][m])
                gap1 = abs(values["C1a"][m] - values["C1b"][m])
                row = dict(scene=scene, renderer=renderer, metric=m, C0_mean=c0, C1_mean=c1,
                           delta=c1-c0, C0_repeat_abs=gap0, C1_repeat_abs=gap1, noise_reference=(gap0+gap1)/2)
                original = old_contrasts[(scene, renderer, m)]
                assert all(math.isclose(v, float(original[k]), abs_tol=1e-12) for k, v in row.items() if isinstance(v, float))
                rows.append(row)
    for renderer in ("R0", "R1"):
        for m in METRICS:
            selected = [r for r in rows if r["renderer"] == renderer and r["metric"] == m]
            rows.append(dict(scene="MACRO_MEAN", renderer=renderer, metric=m,
                **{k: mean(r[k] for r in selected) for k in selected[0] if k not in ("scene", "renderer", "metric")}))
    assert evaluated_views == 104
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    labels = ["Curasao", "IUI3", "JapaneseGradens", "Panama", "Four-scene mean"]
    for ax, renderer in zip(axes, ("R0", "R1")):
        selected = [next(r for r in rows if r["scene"] == scene and r["renderer"] == renderer and r["metric"] == "PSNR")
                    for scene in SCENES + ("MACRO_MEAN",)]
        y = np.arange(len(selected))
        ax.barh(y-.17, [r["delta"] for r in selected], height=.3, label="C1 - C0 (repeat means)", color="#207da4")
        ax.barh(y+.17, [r["noise_reference"] for r in selected], height=.3, label="Mean absolute repeat gap", color="#a8adb3")
        ax.axvline(0, color="#555", linewidth=.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(-.004, .014)
        ax.set_xlabel("PSNR difference (dB)")
        ax.set_title(renderer + (": historical backward" if renderer == "R0" else ": corrected backward"))
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for yi, r in zip(y, selected):
            ax.text(r["delta"]+.00015, yi-.17, f'{r["delta"]:+.5f}', va="center", fontsize=8)
            ax.text(r["noise_reference"]+.00015, yi+.17, f'{r["noise_reference"]:.5f}', va="center", fontsize=8)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .93),
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Exact FSB CICA derivative: benefit versus repeated-run variation", fontsize=13)
    fig.text(.5, .015, "Two same-state repeats per condition. Gray bars are observed differences, not confidence intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .88))
    fig.savefig(out / "psnr_gain_vs_repeat_gap.png", dpi=180)
    fig.savefig(out / "psnr_gain_vs_repeat_gap.pdf")
    plt.close(fig)
    result = dict(all_pass=True, completed_jobs=40, shared_prefixes=8, continuations=32,
        endpoint_view_evaluations=104, retained_checkpoint_hashes_verified=40,
        source_and_artifact_hashes_verified=artifact_count, archived_source_binary_hashes_verified=archives_checked,
        same_start_state_blocks=8, independently_recomputed_metrics_match=True,
        macro=[r for r in rows if r["scene"] == "MACRO_MEAN"],
        limitation="two execution repeats conditional on each shared prefix; no significance test or multi-seed claim",
        performed_training=False)
    (out / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
