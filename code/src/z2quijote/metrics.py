from __future__ import annotations

from typing import Any

import numpy as np


def metric_block(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p68": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p68": float(np.percentile(arr, 68.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def band_masks(k_bins: np.ndarray, edges: tuple[float, ...], labels: tuple[str, ...]) -> dict[str, np.ndarray]:
    k = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    sorted_edges = tuple(float(item) for item in edges)
    if any(left >= right for left, right in zip(sorted_edges, sorted_edges[1:])):
        raise ValueError("band edges must be strictly increasing.")
    if labels and len(labels) != len(sorted_edges) + 1:
        raise ValueError("band label count must equal edge count + 1.")
    names = labels or tuple(f"band_{index}" for index in range(len(sorted_edges) + 1))
    masks: dict[str, np.ndarray] = {}
    lower = -np.inf
    for index, upper in enumerate(sorted_edges + (np.inf,)):
        masks[names[index]] = (k >= lower) & (k < upper)
        lower = upper
    masks["all"] = np.ones_like(k, dtype=bool)
    return masks


def evaluate_prediction(
    *,
    truth_log_pk: np.ndarray,
    pred_log_pk: np.ndarray,
    k_bins: np.ndarray,
    band_edges: tuple[float, ...],
    band_labels: tuple[str, ...],
    target_kind: str = "direct_cdm_logpk",
) -> dict[str, Any]:
    truth = np.asarray(truth_log_pk, dtype=np.float64)
    pred = np.asarray(pred_log_pk, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"truth and prediction shapes differ: {truth.shape} vs {pred.shape}.")
    log_error = np.abs(pred - truth)
    relative_error = np.abs(np.exp(pred - truth) - 1.0)
    result: dict[str, Any] = {
        "target_kind": str(target_kind),
        "overall_relative_error": metric_block(relative_error),
        "overall_log_error": metric_block(log_error),
        "bands": {},
        "sample_p68_relative_error": np.percentile(relative_error, 68.0, axis=1).astype(float).tolist(),
        "sample_mean_relative_error": np.mean(relative_error, axis=1).astype(float).tolist(),
        "k_count": int(np.asarray(k_bins).shape[0]),
        "sample_count": int(truth.shape[0]),
    }
    for label, mask in band_masks(k_bins, band_edges, band_labels).items():
        if not np.any(mask):
            result["bands"][label] = {"bin_count": 0}
            continue
        result["bands"][label] = {
            "bin_count": int(np.count_nonzero(mask)),
            "relative_error": metric_block(relative_error[:, mask]),
            "log_error": metric_block(log_error[:, mask]),
        }
    return result
