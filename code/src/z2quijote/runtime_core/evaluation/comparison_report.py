"""Comparison summary between active-learning and fixed-budget validation results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _load_payload(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def _stable_hash(value: object) -> str | None:
    if value is None:
        return None
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _payload_hashes(payload: dict[str, object]) -> dict[str, str | None]:
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    return {
        "test_thetas": _stable_hash(payload.get("test_thetas")),
        "k_bins": _stable_hash(payload.get("k_bins")),
        "p_pred_batch": _stable_hash(payload.get("p_pred_batch")),
        "train_points": _stable_hash(metadata_dict.get("train_points")),
        "train_thetas_digest": _stable_hash(metadata_dict.get("train_thetas_digest")),
        "train_k_digest": _stable_hash(metadata_dict.get("train_k_digest")),
        "train_pk_digest": _stable_hash(metadata_dict.get("train_pk_digest")),
        "train_linear_digest": _stable_hash(metadata_dict.get("train_linear_digest")),
        "train_cache_path": _stable_hash(metadata_dict.get("train_cache_path")),
        "train_cache_theta_digest": _stable_hash(metadata_dict.get("train_cache_theta_digest")),
        "train_cache_k_digest": _stable_hash(metadata_dict.get("train_cache_k_digest")),
        "validation_thetas_digest": _stable_hash(metadata_dict.get("validation_thetas_digest")),
        "validation_k_digest": _stable_hash(metadata_dict.get("validation_k_digest")),
        "validation_nonlin_digest": _stable_hash(metadata_dict.get("validation_nonlin_digest")),
        "validation_linear_digest": _stable_hash(metadata_dict.get("validation_linear_digest")),
        "validation_cache_path": _stable_hash(metadata_dict.get("validation_cache_path")),
        "validation_cache_theta_digest": _stable_hash(metadata_dict.get("validation_cache_theta_digest")),
        "validation_cache_k_digest": _stable_hash(metadata_dict.get("validation_cache_k_digest")),
    }


def _compare_metric(
    payload_a: dict[str, object],
    payload_b: dict[str, object],
    key: str,
) -> dict[str, object]:
    value_a = float(payload_a.get(key, 0.0))
    value_b = float(payload_b.get(key, 0.0))
    delta = value_a - value_b
    if abs(value_b) > 1.0e-30:
        relative_delta = delta / value_b
    else:
        relative_delta = 0.0
    if value_a < value_b:
        better = "a"
    elif value_a > value_b:
        better = "b"
    else:
        better = "tie"
    return {
        "metric": key,
        "a": value_a,
        "b": value_b,
        "delta_a_minus_b": delta,
        "relative_delta_vs_b": relative_delta,
        "better": better,
    }


def write_comparison_report(
    output_path: str | Path,
    *,
    results_a_path: str | Path,
    results_b_path: str | Path,
    label_a: str = "active_learning",
    label_b: str = "fixed_budget_comparison",
) -> Path:
    payload_a = _load_payload(results_a_path)
    payload_b = _load_payload(results_b_path)
    metrics = [
        "overall_mean_relative_error",
        "overall_p68_relative_error",
        "overall_p95_relative_error",
        "overall_max_relative_error",
        "overall_mean_log_error",
        "overall_p68_log_error",
        "overall_p95_log_error",
        "k_le_1_mean_relative_error",
        "k_le_1_p68_relative_error",
        "k_le_1_max_relative_error",
        "band_relative_error_low_mean",
        "band_relative_error_low_p68",
        "band_relative_error_mid_mean",
        "band_relative_error_mid_p68",
        "band_relative_error_focus_high_mean",
        "band_relative_error_focus_high_p68",
        "band_relative_error_tail_mean",
        "band_relative_error_tail_p68",
        "band_relative_error_high_mean",
        "band_relative_error_high_p68",
        "focus_0p07_3_integrated_relative_error_mean",
        "focus_0p07_3_integrated_relative_error_p68",
        "focus_0p08_3_integrated_relative_error_mean",
        "focus_0p08_3_integrated_relative_error_p68",
        "focus_0p1_3_integrated_relative_error_mean",
        "focus_0p1_3_integrated_relative_error_p68",
        "focus_0p1_5_integrated_relative_error_mean",
        "focus_0p1_5_integrated_relative_error_p68",
        "band_log_error_low_mean",
        "band_log_error_low_p68",
        "band_log_error_mid_mean",
        "band_log_error_mid_p68",
        "band_log_error_focus_high_mean",
        "band_log_error_focus_high_p68",
        "band_log_error_tail_mean",
        "band_log_error_tail_p68",
        "band_log_error_high_mean",
        "band_log_error_high_p68",
        "focus_0p07_3_integrated_log_error_mean",
        "focus_0p07_3_integrated_log_error_p68",
        "focus_0p08_3_integrated_log_error_mean",
        "focus_0p08_3_integrated_log_error_p68",
        "focus_0p1_3_integrated_log_error_mean",
        "focus_0p1_3_integrated_log_error_p68",
        "focus_0p1_5_integrated_log_error_mean",
        "focus_0p1_5_integrated_log_error_p68",
        "sample_mean_relative_error_mean",
        "sample_max_relative_error_mean",
    ]
    comparisons = [_compare_metric(payload_a, payload_b, key) for key in metrics]
    hashes_a = _payload_hashes(payload_a)
    hashes_b = _payload_hashes(payload_b)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "labels": {"a": str(label_a), "b": str(label_b)},
                "results": {
                    "a": str(Path(results_a_path).resolve()),
                    "b": str(Path(results_b_path).resolve()),
                },
                "artifact_hashes": {
                    "a": hashes_a,
                    "b": hashes_b,
                },
                "artifact_hash_consistency": {
                    "test_thetas_match": hashes_a["test_thetas"] == hashes_b["test_thetas"],
                    "k_bins_match": hashes_a["k_bins"] == hashes_b["k_bins"],
                    "validation_thetas_digest_match": (
                        hashes_a["validation_thetas_digest"] == hashes_b["validation_thetas_digest"]
                        if hashes_a["validation_thetas_digest"] is not None
                        and hashes_b["validation_thetas_digest"] is not None
                        else None
                    ),
                    "validation_k_digest_match": (
                        hashes_a["validation_k_digest"] == hashes_b["validation_k_digest"]
                        if hashes_a["validation_k_digest"] is not None
                        and hashes_b["validation_k_digest"] is not None
                        else None
                    ),
                    "validation_nonlin_digest_match": (
                        hashes_a["validation_nonlin_digest"] == hashes_b["validation_nonlin_digest"]
                        if hashes_a["validation_nonlin_digest"] is not None
                        and hashes_b["validation_nonlin_digest"] is not None
                        else None
                    ),
                },
                "comparisons": comparisons,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
