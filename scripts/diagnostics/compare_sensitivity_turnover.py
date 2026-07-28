#!/usr/bin/env python
"""Compare contribution sensitivity turnover across checkpoint diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _load_payload(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if "gaussian_lineage_ids" not in payload:
        n = int(np.asarray(payload["water_score"]).reshape(-1).shape[0])
        payload["gaussian_lineage_ids"] = torch.arange(n, dtype=torch.long)
        payload["lineage_fallback"] = True
    else:
        payload["lineage_fallback"] = False
    payload["_path"] = str(path)
    return payload


def _lineage_scores(payload: Dict[str, Any], score_key: str) -> Tuple[np.ndarray, np.ndarray]:
    ids = _as_numpy(payload["gaussian_lineage_ids"], np.int64).reshape(-1)
    scores = _as_numpy(payload[score_key], np.float64).reshape(-1)
    scores = np.where(np.isfinite(scores) & (scores > 0.0), scores, 0.0)
    unique, inverse = np.unique(ids, return_inverse=True)
    summed = np.zeros(unique.shape[0], dtype=np.float64)
    np.add.at(summed, inverse, scores)
    return unique, summed


def _lineage_selected(payload: Dict[str, Any]) -> np.ndarray:
    ids = _as_numpy(payload["gaussian_lineage_ids"], np.int64).reshape(-1)
    mask = _as_numpy(payload.get("candidate_mask", np.zeros_like(ids, dtype=bool)), bool).reshape(-1)
    if mask.size != ids.size:
        return np.empty(0, dtype=np.int64)
    return np.unique(ids[mask])


def _score_map(lineages: np.ndarray, scores: np.ndarray) -> Dict[int, float]:
    return {int(k): float(v) for k, v in zip(lineages.tolist(), scores.tolist()) if v > 0.0}


def _top_set(lineages: np.ndarray, scores: np.ndarray, top_k: int) -> Tuple[set[int], float]:
    if lineages.size == 0:
        return set(), 0.0
    k = min(int(top_k), lineages.size)
    order = np.argsort(scores)[::-1][:k]
    return set(int(v) for v in lineages[order].tolist()), float(scores[order].sum())


def _mass_for(score_by_lineage: Dict[int, float], lineage_set: Iterable[int]) -> float:
    return float(sum(score_by_lineage.get(int(idx), 0.0) for idx in lineage_set))


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def _point_attr_summary(payload: Dict[str, Any], lineage_set: set[int], score_key: str) -> Dict[str, Any]:
    ids = _as_numpy(payload["gaussian_lineage_ids"], np.int64).reshape(-1)
    if not lineage_set:
        return {"point_count": 0, "lineage_count": 0}
    selected = np.isin(ids, np.fromiter(lineage_set, dtype=np.int64))
    if not selected.any():
        return {"point_count": 0, "lineage_count": 0}

    scores = _as_numpy(payload[score_key], np.float64).reshape(-1)[selected]
    scores = np.where(np.isfinite(scores) & (scores > 0.0), scores, 0.0)
    weights = scores / max(float(scores.sum()), 1e-12)

    def stat_1d(key: str) -> Dict[str, float]:
        if key not in payload:
            return {}
        values = _as_numpy(payload[key], np.float64).reshape(-1)[selected]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {}
        return {
            f"{key}_mean": float(values.mean()),
            f"{key}_p95": float(np.quantile(values, 0.95)),
            f"{key}_weighted_mean": float(np.sum(_as_numpy(payload[key], np.float64).reshape(-1)[selected] * weights)),
        }

    result: Dict[str, Any] = {
        "point_count": int(selected.sum()),
        "lineage_count": int(len(lineage_set)),
        "score_sum": float(scores.sum()),
    }
    result.update(stat_1d("opacity"))
    result.update(stat_1d("max_scale"))
    result.update(stat_1d("sh_rest_norm"))
    if "dc_rgb" in payload:
        rgb = _as_numpy(payload["dc_rgb"], np.float64).reshape(-1, 3)[selected]
        if rgb.size:
            weighted_rgb = np.sum(rgb * weights[:, None], axis=0)
            result["dc_rgb_weighted_mean"] = [float(v) for v in weighted_rgb.tolist()]
            result["dc_bluegreen_minus_red_weighted_mean"] = float(
                max(weighted_rgb[1], weighted_rgb[2]) - weighted_rgb[0]
            )
    return result


def _payload_summary(payload: Dict[str, Any], score_keys: List[str]) -> Dict[str, Any]:
    ids = _as_numpy(payload["gaussian_lineage_ids"], np.int64).reshape(-1)
    summary = {
        "path": payload["_path"],
        "step": int(payload.get("step", -1)),
        "checkpoint": str(payload.get("checkpoint", "")),
        "num_points": int(ids.size),
        "lineage_count": int(np.unique(ids).size),
        "lineage_fallback": bool(payload.get("lineage_fallback", False)),
        "candidate_point_count": int(_as_numpy(payload.get("candidate_mask", np.zeros_like(ids, dtype=bool)), bool).sum()),
        "candidate_lineage_count": int(_lineage_selected(payload).size),
        "scores": {},
    }
    for key in score_keys:
        lineages, scores = _lineage_scores(payload, key)
        positive = scores[scores > 0.0]
        summary["scores"][key] = {
            "total": float(scores.sum()),
            "positive_lineage_count": int(positive.size),
            "max": float(positive.max()) if positive.size else 0.0,
            "p95": float(np.quantile(positive, 0.95)) if positive.size else 0.0,
        }
    return summary


def compare(payloads: List[Dict[str, Any]], score_keys: List[str], top_ks: List[int]) -> Dict[str, Any]:
    payloads = sorted(payloads, key=lambda item: int(item.get("step", -1)))
    summaries = [_payload_summary(payload, score_keys) for payload in payloads]
    pairs: List[Dict[str, Any]] = []
    for prev, cur in zip(payloads, payloads[1:]):
        pair: Dict[str, Any] = {
            "from_step": int(prev.get("step", -1)),
            "to_step": int(cur.get("step", -1)),
            "from_path": prev["_path"],
            "to_path": cur["_path"],
            "score_keys": {},
        }
        prev_candidates = set(int(v) for v in _lineage_selected(prev).tolist())
        cur_candidates = set(int(v) for v in _lineage_selected(cur).tolist())
        pair["candidate_lineage_jaccard"] = _jaccard(prev_candidates, cur_candidates)
        pair["prev_candidate_survival_fraction"] = (
            float(len(prev_candidates & cur_candidates) / len(prev_candidates)) if prev_candidates else 1.0
        )
        for key in score_keys:
            prev_lineages, prev_scores = _lineage_scores(prev, key)
            cur_lineages, cur_scores = _lineage_scores(cur, key)
            cur_score_map = _score_map(cur_lineages, cur_scores)
            cur_total = max(float(cur_scores.sum()), 1e-12)
            key_result: Dict[str, Any] = {
                "cur_total_score": float(cur_scores.sum()),
                "cur_score_share_from_prev_candidates": _mass_for(cur_score_map, prev_candidates) / cur_total,
                "cur_score_share_from_cur_candidates": _mass_for(cur_score_map, cur_candidates) / cur_total,
            }
            for top_k in top_ks:
                prev_top, prev_top_mass = _top_set(prev_lineages, prev_scores, top_k)
                cur_top, cur_top_mass = _top_set(cur_lineages, cur_scores, top_k)
                new_top = cur_top - prev_top
                top_result = {
                    "prev_top_mass": prev_top_mass,
                    "cur_top_mass": cur_top_mass,
                    "top_jaccard": _jaccard(prev_top, cur_top),
                    "prev_top_survival_fraction": float(len(prev_top & cur_top) / len(prev_top)) if prev_top else 1.0,
                    "cur_top_score_share_from_prev_top": _mass_for(cur_score_map, prev_top) / cur_total,
                    "cur_top_new_lineage_score_share": _mass_for(cur_score_map, new_top) / cur_total,
                    "cur_top_attr_summary": _point_attr_summary(cur, cur_top, key),
                    "cur_new_top_attr_summary": _point_attr_summary(cur, new_top, key),
                }
                key_result[f"top_{top_k}"] = top_result
            pair["score_keys"][key] = key_result
        pairs.append(pair)
    return {
        "alignment": "gaussian_lineage_ids",
        "score_keys": score_keys,
        "top_ks": top_ks,
        "checkpoints": summaries,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pt", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--score-key",
        dest="score_keys",
        action="append",
        default=[],
        choices=["candidate_score", "water_score", "water_proxy_bluegreen_score", "features_rest_score"],
    )
    parser.add_argument("--top-k", type=int, action="append", default=[])
    args = parser.parse_args()

    score_keys = args.score_keys or ["candidate_score", "water_score", "water_proxy_bluegreen_score"]
    top_ks = args.top_k or [50, 100, 500]
    payloads = [_load_payload(path) for path in args.candidate_pt]
    result = compare(payloads, score_keys, top_ks)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                "alignment": result["alignment"],
                "steps": [item["step"] for item in result["checkpoints"]],
                "pairs": [
                    {
                        "from_step": item["from_step"],
                        "to_step": item["to_step"],
                        "candidate_lineage_jaccard": item["candidate_lineage_jaccard"],
                    }
                    for item in result["pairs"]
                ],
            },
            indent=2,
        )
    )
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
