"""Test-set result serialization used by validation and comparison tools."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TEST_SET_RESULTS_FILENAME = "test_set_results.json"
DEFAULT_K_TARGET_MIN = 1.0e-2
DEFAULT_K_TARGET_MAX = 1.0
_MID_K_MIN = 0.07
_MID_K_PEAK = 0.3
_MID_K_MAX = 0.5
_HIGH_K_MIN = 0.5
_TAIL_K_MIN = 1.0
_HIGH_K_MAX = 3.0
_LEGACY_FOCUS_K_MIN = 0.1
_WINDOW_LOG_DEX = 0.05


def _build_logk_trapezoid_weights(k_bins: np.ndarray) -> np.ndarray:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    if k_arr.ndim != 1 or k_arr.size < 2:
        raise ValueError("k_bins must be a 1D array with at least two points.")
    logk = np.log10(np.maximum(k_arr, 1.0e-30))
    weights = np.empty_like(logk, dtype=np.float64)
    weights[0] = 0.5 * (logk[1] - logk[0])
    weights[-1] = 0.5 * (logk[-1] - logk[-2])
    if logk.size > 2:
        weights[1:-1] = 0.5 * (logk[2:] - logk[:-2])
    return np.maximum(weights, 0.0).astype(np.float64)


def _band_masks(k_bins: np.ndarray) -> dict[str, np.ndarray]:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    return {
        "low": (k_arr >= 1.0e-2) & (k_arr < _MID_K_MIN),
        "mid": (k_arr >= _MID_K_MIN) & (k_arr < _HIGH_K_MIN),
        "high": (k_arr >= _HIGH_K_MIN) & (k_arr < _TAIL_K_MIN),
        "tail": (k_arr >= _TAIL_K_MIN) & (k_arr <= _HIGH_K_MAX),
        "mid_high": (k_arr >= _MID_K_MIN) & (k_arr <= _HIGH_K_MAX),
        "focus_0p07_3": (k_arr >= _MID_K_MIN) & (k_arr <= _HIGH_K_MAX),
        "focus_0p08_3": (k_arr >= 0.08) & (k_arr <= _HIGH_K_MAX),
        "focus_0p1_3": (k_arr >= _LEGACY_FOCUS_K_MIN) & (k_arr <= _HIGH_K_MAX),
        # Legacy keys remain for old report consumers; current plotting no longer uses them.
        "focus_high": (k_arr >= _HIGH_K_MIN) & (k_arr <= _HIGH_K_MAX),
        "focus_0p1_5": (k_arr >= _LEGACY_FOCUS_K_MIN) & (k_arr <= _HIGH_K_MAX),
    }


def _midband_weights(k_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    mask = (k_arr >= _MID_K_MIN) & (k_arr <= _MID_K_MAX)
    weights = np.zeros_like(k_arr, dtype=np.float64)
    weights[(k_arr >= _MID_K_MIN) & (k_arr < _MID_K_PEAK)] = 2.0
    weights[(k_arr >= _MID_K_PEAK) & (k_arr <= _MID_K_MAX)] = 1.0
    total = float(np.sum(weights))
    if total > 0.0:
        weights = weights / total
    return mask, weights


def _windowed_quantile_curve(
    k_bins: np.ndarray,
    error_batch: np.ndarray,
    *,
    quantile: float,
    log_window_dex: float = _WINDOW_LOG_DEX,
) -> np.ndarray:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    logk = np.log10(np.maximum(k_arr, 1.0e-12))
    out = np.zeros_like(k_arr, dtype=np.float64)
    for idx, center in enumerate(logk):
        mask = np.abs(logk - center) <= float(log_window_dex) * 0.5
        if not np.any(mask):
            out[idx] = float(np.percentile(error_batch[:, idx], quantile))
            continue
        window_vals = np.mean(error_batch[:, mask], axis=1)
        out[idx] = float(np.percentile(window_vals, quantile))
    return out


def _integrated_midband_quantiles(
    k_bins: np.ndarray,
    relative_error_batch: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    mask_mid, weights_mid = _midband_weights(k_bins)
    if not np.any(mask_mid):
        sample_integrated = np.zeros((relative_error_batch.shape[0],), dtype=np.float64)
    else:
        weight_vec = weights_mid[mask_mid]
        sample_integrated = np.asarray(relative_error_batch[:, mask_mid] @ weight_vec, dtype=np.float64)
    return sample_integrated, {
        "mean": float(np.mean(sample_integrated)),
        "p50": float(np.percentile(sample_integrated, 50)),
        "p68": float(np.percentile(sample_integrated, 68)),
        "p95": float(np.percentile(sample_integrated, 95)),
    }


def _integrated_logk_quantiles(
    k_bins: np.ndarray,
    error_batch: np.ndarray,
    *,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if not np.any(mask):
        sample_integrated = np.zeros((error_batch.shape[0],), dtype=np.float64)
    else:
        logk_weights = _build_logk_trapezoid_weights(k_bins)
        band_weights = logk_weights[mask]
        band_weights = band_weights / max(np.sum(band_weights), 1.0e-30)
        sample_integrated = np.asarray(error_batch[:, mask] @ band_weights, dtype=np.float64)
    return sample_integrated, {
        "mean": float(np.mean(sample_integrated)),
        "p50": float(np.percentile(sample_integrated, 50)),
        "p68": float(np.percentile(sample_integrated, 68)),
        "p95": float(np.percentile(sample_integrated, 95)),
    }


def _band_curve_stats(
    mean_curve: np.ndarray,
    p68_curve: np.ndarray,
    *,
    mask: np.ndarray,
) -> tuple[float, float, float, float]:
    values_mean = np.asarray(mean_curve, dtype=np.float64)[mask]
    values_p68 = np.asarray(p68_curve, dtype=np.float64)[mask]
    if values_mean.size <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(values_mean)),
        float(np.percentile(values_mean, 95)),
        float(np.max(values_mean)),
        float(np.max(values_p68)),
    )


def build_test_set_results_payload(
    test_thetas: np.ndarray,
    k_bins: np.ndarray,
    p_true_batch: np.ndarray,
    p_pred_batch: np.ndarray,
    p_linear_batch: np.ndarray | None = None,
    *,
    spectrum_type: str = "galaxy",
    eps_r: float = 1.0e-12,
    k_target_max: float = DEFAULT_K_TARGET_MAX,
    target_accuracy: float = 0.01,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    test_thetas = np.asarray(test_thetas, dtype=np.float64)
    k_bins = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    p_true_batch = np.asarray(p_true_batch, dtype=np.float64)
    p_pred_batch = np.asarray(p_pred_batch, dtype=np.float64)
    if p_linear_batch is not None:
        p_linear_batch = np.asarray(p_linear_batch, dtype=np.float64)

    eps = float(max(eps_r, 1.0e-30))
    safe_true = np.maximum(p_true_batch, eps)
    safe_pred = np.maximum(p_pred_batch, eps)
    relative_error_batch = np.abs(p_pred_batch - p_true_batch) / safe_true
    log_error_batch = np.abs(np.log(safe_pred) - np.log(safe_true))

    relative_error_mean_arr = np.mean(relative_error_batch, axis=0)
    relative_error_std_arr = np.std(relative_error_batch, axis=0)
    relative_error_p68_arr = np.percentile(relative_error_batch, 68, axis=0)
    relative_error_p95_arr = np.percentile(relative_error_batch, 95, axis=0)
    relative_error_p68_windowed_arr = _windowed_quantile_curve(
        k_bins,
        relative_error_batch,
        quantile=68,
        log_window_dex=_WINDOW_LOG_DEX,
    )

    log_error_mean_arr = np.mean(log_error_batch, axis=0)
    log_error_std_arr = np.std(log_error_batch, axis=0)
    log_error_p68_arr = np.percentile(log_error_batch, 68, axis=0)
    log_error_p95_arr = np.percentile(log_error_batch, 95, axis=0)
    log_error_p68_windowed_arr = _windowed_quantile_curve(
        k_bins,
        log_error_batch,
        quantile=68,
        log_window_dex=_WINDOW_LOG_DEX,
    )

    metric_mask = k_bins >= DEFAULT_K_TARGET_MIN
    if not np.any(metric_mask):
        metric_mask = np.ones_like(k_bins, dtype=bool)
    metric_relative_error_batch = relative_error_batch[:, metric_mask]
    metric_log_error_batch = log_error_batch[:, metric_mask]

    sample_mean_relative_error = np.mean(metric_relative_error_batch, axis=1)
    sample_max_relative_error = np.max(metric_relative_error_batch, axis=1)
    sample_mean_log_error = np.mean(metric_log_error_batch, axis=1)
    sample_max_log_error = np.max(metric_log_error_batch, axis=1)

    mid_integrated_per_sample, mid_integrated_quantiles = _integrated_midband_quantiles(
        k_bins,
        relative_error_batch,
    )
    masks = _band_masks(k_bins)
    focus_integrated_rel_per_sample, focus_integrated_rel_quantiles = _integrated_logk_quantiles(
        k_bins,
        relative_error_batch,
        mask=masks["focus_0p07_3"],
    )
    focus_integrated_log_per_sample, focus_integrated_log_quantiles = _integrated_logk_quantiles(
        k_bins,
        log_error_batch,
        mask=masks["focus_0p07_3"],
    )
    legacy_focus_integrated_rel_per_sample, legacy_focus_integrated_rel_quantiles = _integrated_logk_quantiles(
        k_bins,
        relative_error_batch,
        mask=masks["focus_0p1_5"],
    )
    legacy_focus_integrated_log_per_sample, legacy_focus_integrated_log_quantiles = _integrated_logk_quantiles(
        k_bins,
        log_error_batch,
        mask=masks["focus_0p1_5"],
    )

    mask_le1 = (k_bins >= DEFAULT_K_TARGET_MIN) & (k_bins <= float(k_target_max))
    if np.any(mask_le1):
        relative_error_mean_le1 = np.mean(relative_error_batch[:, mask_le1], axis=0)
        relative_error_p68_le1 = np.percentile(relative_error_batch[:, mask_le1], 68, axis=0)
        k_le_1_max_relative_error = float(np.max(relative_error_mean_le1))
        k_le_1_mean_relative_error = float(np.mean(relative_error_mean_le1))
        k_le_1_p95_relative_error = float(np.percentile(relative_error_mean_le1, 95))
        k_le_1_p68_relative_error = float(np.max(relative_error_p68_le1))
    else:
        k_le_1_max_relative_error = 0.0
        k_le_1_mean_relative_error = 0.0
        k_le_1_p95_relative_error = 0.0
        k_le_1_p68_relative_error = 0.0

    low_mean, low_p95, low_max, low_p68 = _band_curve_stats(
        relative_error_mean_arr,
        relative_error_p68_arr,
        mask=masks["low"],
    )
    mid_mean, mid_p95, mid_max, mid_p68 = _band_curve_stats(
        relative_error_mean_arr,
        relative_error_p68_arr,
        mask=masks["mid"],
    )
    focus_high_mean, focus_high_p95, focus_high_max, focus_high_p68 = _band_curve_stats(
        relative_error_mean_arr,
        relative_error_p68_arr,
        mask=masks["focus_high"],
    )
    tail_mean, tail_p95, tail_max, tail_p68 = _band_curve_stats(
        relative_error_mean_arr,
        relative_error_p68_arr,
        mask=masks["tail"],
    )
    high_mean, high_p95, high_max, high_p68 = _band_curve_stats(
        relative_error_mean_arr,
        relative_error_p68_arr,
        mask=masks["high"],
    )

    log_low_mean, log_low_p95, log_low_max, log_low_p68 = _band_curve_stats(
        log_error_mean_arr,
        log_error_p68_arr,
        mask=masks["low"],
    )
    log_mid_mean, log_mid_p95, log_mid_max, log_mid_p68 = _band_curve_stats(
        log_error_mean_arr,
        log_error_p68_arr,
        mask=masks["mid"],
    )
    log_focus_high_mean, log_focus_high_p95, log_focus_high_max, log_focus_high_p68 = _band_curve_stats(
        log_error_mean_arr,
        log_error_p68_arr,
        mask=masks["focus_high"],
    )
    log_tail_mean, log_tail_p95, log_tail_max, log_tail_p68 = _band_curve_stats(
        log_error_mean_arr,
        log_error_p68_arr,
        mask=masks["tail"],
    )
    log_high_mean, log_high_p95, log_high_max, log_high_p68 = _band_curve_stats(
        log_error_mean_arr,
        log_error_p68_arr,
        mask=masks["high"],
    )

    p_true_mean = np.mean(p_true_batch, axis=0).astype(np.float64)
    p_pred_mean = np.mean(p_pred_batch, axis=0).astype(np.float64)

    payload: dict[str, object] = {
        "test_set_size": int(test_thetas.shape[0]),
        "theta_dim": int(test_thetas.shape[1]) if test_thetas.ndim == 2 else 0,
        "spectrum_type": str(spectrum_type),
        "k_band_scheme": "z2_quijote_four_band_0p01_0p07_0p5_1_3",
        "k_band_edges": {
            "low": [float(DEFAULT_K_TARGET_MIN), float(_MID_K_MIN)],
            "mid": [float(_MID_K_MIN), float(_HIGH_K_MIN)],
            "high": [float(_HIGH_K_MIN), float(_TAIL_K_MIN)],
            "tail": [float(_TAIL_K_MIN), float(_HIGH_K_MAX)],
        },
        "target_accuracy": float(target_accuracy),
        "evaluation_k_min": float(DEFAULT_K_TARGET_MIN),
        "evaluation_k_bin_count": int(np.sum(metric_mask)),
        "k_target_max": float(k_target_max),
        "test_thetas": test_thetas.tolist(),
        "k_bins": k_bins.tolist(),
        "p_true_batch": p_true_batch.tolist(),
        "p_pred_batch": p_pred_batch.tolist(),
        "p_true_mean": p_true_mean.tolist(),
        "p_pred_mean": p_pred_mean.tolist(),
        "power_relative_error_mean": relative_error_mean_arr.tolist(),
        "power_relative_error_std": relative_error_std_arr.tolist(),
        "power_relative_error_p68": relative_error_p68_arr.tolist(),
        "power_relative_error_p95": relative_error_p95_arr.tolist(),
        "power_relative_error_p68_windowed": relative_error_p68_windowed_arr.tolist(),
        "power_log_error_mean": log_error_mean_arr.tolist(),
        "power_log_error_std": log_error_std_arr.tolist(),
        "power_log_error_p68": log_error_p68_arr.tolist(),
        "power_log_error_p95": log_error_p95_arr.tolist(),
        "power_log_error_p68_windowed": log_error_p68_windowed_arr.tolist(),
        "sample_mean_relative_error": sample_mean_relative_error.tolist(),
        "sample_max_relative_error": sample_max_relative_error.tolist(),
        "sample_mean_log_error": sample_mean_log_error.tolist(),
        "sample_max_log_error": sample_max_log_error.tolist(),
        "overall_mean_relative_error": float(np.mean(metric_relative_error_batch)),
        "overall_p68_relative_error": float(np.percentile(metric_relative_error_batch, 68)),
        "overall_p95_relative_error": float(np.percentile(metric_relative_error_batch, 95)),
        "overall_max_relative_error": float(np.max(metric_relative_error_batch)),
        "overall_mean_log_error": float(np.mean(metric_log_error_batch)),
        "overall_p68_log_error": float(np.percentile(metric_log_error_batch, 68)),
        "overall_p95_log_error": float(np.percentile(metric_log_error_batch, 95)),
        "overall_max_log_error": float(np.max(metric_log_error_batch)),
        "sample_mean_relative_error_mean": float(np.mean(sample_mean_relative_error)),
        "sample_mean_relative_error_p95": float(np.percentile(sample_mean_relative_error, 95)),
        "sample_mean_relative_error_max": float(np.max(sample_mean_relative_error)),
        "sample_max_relative_error_mean": float(np.mean(sample_max_relative_error)),
        "sample_max_relative_error_p95": float(np.percentile(sample_max_relative_error, 95)),
        "sample_max_relative_error_max": float(np.max(sample_max_relative_error)),
        "sample_mean_log_error_mean": float(np.mean(sample_mean_log_error)),
        "sample_mean_log_error_p95": float(np.percentile(sample_mean_log_error, 95)),
        "sample_mean_log_error_max": float(np.max(sample_mean_log_error)),
        "sample_max_log_error_mean": float(np.mean(sample_max_log_error)),
        "sample_max_log_error_p95": float(np.percentile(sample_max_log_error, 95)),
        "sample_max_log_error_max": float(np.max(sample_max_log_error)),
        "k_le_1_bin_count": int(np.sum(mask_le1)),
        "k_le_1_max_relative_error": k_le_1_max_relative_error,
        "k_le_1_mean_relative_error": k_le_1_mean_relative_error,
        "k_le_1_p95_relative_error": k_le_1_p95_relative_error,
        "k_le_1_p68_relative_error": k_le_1_p68_relative_error,
        "band_relative_error_low_mean": low_mean,
        "band_relative_error_low_p95": low_p95,
        "band_relative_error_low_max": low_max,
        "band_relative_error_low_p68": low_p68,
        "band_relative_error_mid_mean": mid_mean,
        "band_relative_error_mid_p95": mid_p95,
        "band_relative_error_mid_max": mid_max,
        "band_relative_error_mid_p68": mid_p68,
        "band_relative_error_focus_high_mean": focus_high_mean,
        "band_relative_error_focus_high_p95": focus_high_p95,
        "band_relative_error_focus_high_max": focus_high_max,
        "band_relative_error_focus_high_p68": focus_high_p68,
        "band_relative_error_tail_mean": tail_mean,
        "band_relative_error_tail_p95": tail_p95,
        "band_relative_error_tail_max": tail_max,
        "band_relative_error_tail_p68": tail_p68,
        "band_relative_error_mid_integrated_p50": float(mid_integrated_quantiles["p50"]),
        "band_relative_error_mid_integrated_p68": float(mid_integrated_quantiles["p68"]),
        "band_relative_error_mid_integrated_p95": float(mid_integrated_quantiles["p95"]),
        "band_relative_error_mid_integrated_mean": float(mid_integrated_quantiles["mean"]),
        "band_relative_error_high_mean": high_mean,
        "band_relative_error_high_p95": high_p95,
        "band_relative_error_high_max": high_max,
        "band_relative_error_high_p68": high_p68,
        "focus_0p07_3_integrated_relative_error_mean": float(
            focus_integrated_rel_quantiles["mean"]
        ),
        "focus_0p07_3_integrated_relative_error_p50": float(
            focus_integrated_rel_quantiles["p50"]
        ),
        "focus_0p07_3_integrated_relative_error_p68": float(
            focus_integrated_rel_quantiles["p68"]
        ),
        "focus_0p07_3_integrated_relative_error_p95": float(
            focus_integrated_rel_quantiles["p95"]
        ),
        "focus_0p08_3_integrated_relative_error_mean": float(
            focus_integrated_rel_quantiles["mean"]
        ),
        "focus_0p08_3_integrated_relative_error_p50": float(
            focus_integrated_rel_quantiles["p50"]
        ),
        "focus_0p08_3_integrated_relative_error_p68": float(
            focus_integrated_rel_quantiles["p68"]
        ),
        "focus_0p08_3_integrated_relative_error_p95": float(
            focus_integrated_rel_quantiles["p95"]
        ),
        "focus_0p1_5_integrated_relative_error_mean": float(
            legacy_focus_integrated_rel_quantiles["mean"]
        ),
        "focus_0p1_5_integrated_relative_error_p50": float(
            legacy_focus_integrated_rel_quantiles["p50"]
        ),
        "focus_0p1_5_integrated_relative_error_p68": float(
            legacy_focus_integrated_rel_quantiles["p68"]
        ),
        "focus_0p1_5_integrated_relative_error_p95": float(
            legacy_focus_integrated_rel_quantiles["p95"]
        ),
        "focus_0p1_3_integrated_relative_error_mean": float(
            legacy_focus_integrated_rel_quantiles["mean"]
        ),
        "focus_0p1_3_integrated_relative_error_p50": float(
            legacy_focus_integrated_rel_quantiles["p50"]
        ),
        "focus_0p1_3_integrated_relative_error_p68": float(
            legacy_focus_integrated_rel_quantiles["p68"]
        ),
        "focus_0p1_3_integrated_relative_error_p95": float(
            legacy_focus_integrated_rel_quantiles["p95"]
        ),
        "band_log_error_low_mean": log_low_mean,
        "band_log_error_low_p95": log_low_p95,
        "band_log_error_low_max": log_low_max,
        "band_log_error_low_p68": log_low_p68,
        "band_log_error_mid_mean": log_mid_mean,
        "band_log_error_mid_p95": log_mid_p95,
        "band_log_error_mid_max": log_mid_max,
        "band_log_error_mid_p68": log_mid_p68,
        "band_log_error_focus_high_mean": log_focus_high_mean,
        "band_log_error_focus_high_p95": log_focus_high_p95,
        "band_log_error_focus_high_max": log_focus_high_max,
        "band_log_error_focus_high_p68": log_focus_high_p68,
        "band_log_error_tail_mean": log_tail_mean,
        "band_log_error_tail_p95": log_tail_p95,
        "band_log_error_tail_max": log_tail_max,
        "band_log_error_tail_p68": log_tail_p68,
        "band_log_error_high_mean": log_high_mean,
        "band_log_error_high_p95": log_high_p95,
        "band_log_error_high_max": log_high_max,
        "band_log_error_high_p68": log_high_p68,
        "focus_0p07_3_integrated_log_error_mean": float(
            focus_integrated_log_quantiles["mean"]
        ),
        "focus_0p07_3_integrated_log_error_p50": float(
            focus_integrated_log_quantiles["p50"]
        ),
        "focus_0p07_3_integrated_log_error_p68": float(
            focus_integrated_log_quantiles["p68"]
        ),
        "focus_0p07_3_integrated_log_error_p95": float(
            focus_integrated_log_quantiles["p95"]
        ),
        "focus_0p08_3_integrated_log_error_mean": float(
            focus_integrated_log_quantiles["mean"]
        ),
        "focus_0p08_3_integrated_log_error_p50": float(
            focus_integrated_log_quantiles["p50"]
        ),
        "focus_0p08_3_integrated_log_error_p68": float(
            focus_integrated_log_quantiles["p68"]
        ),
        "focus_0p08_3_integrated_log_error_p95": float(
            focus_integrated_log_quantiles["p95"]
        ),
        "focus_0p1_5_integrated_log_error_mean": float(
            legacy_focus_integrated_log_quantiles["mean"]
        ),
        "focus_0p1_5_integrated_log_error_p50": float(
            legacy_focus_integrated_log_quantiles["p50"]
        ),
        "focus_0p1_5_integrated_log_error_p68": float(
            legacy_focus_integrated_log_quantiles["p68"]
        ),
        "focus_0p1_5_integrated_log_error_p95": float(
            legacy_focus_integrated_log_quantiles["p95"]
        ),
        "focus_0p1_3_integrated_log_error_mean": float(
            legacy_focus_integrated_log_quantiles["mean"]
        ),
        "focus_0p1_3_integrated_log_error_p50": float(
            legacy_focus_integrated_log_quantiles["p50"]
        ),
        "focus_0p1_3_integrated_log_error_p68": float(
            legacy_focus_integrated_log_quantiles["p68"]
        ),
        "focus_0p1_3_integrated_log_error_p95": float(
            legacy_focus_integrated_log_quantiles["p95"]
        ),
        "mid_integrated_relative_error_per_sample": mid_integrated_per_sample.tolist(),
        "focus_0p07_3_integrated_relative_error_per_sample": (
            focus_integrated_rel_per_sample.tolist()
        ),
        "focus_0p08_3_integrated_relative_error_per_sample": (
            focus_integrated_rel_per_sample.tolist()
        ),
        "focus_0p1_5_integrated_relative_error_per_sample": (
            legacy_focus_integrated_rel_per_sample.tolist()
        ),
        "focus_0p1_3_integrated_relative_error_per_sample": (
            legacy_focus_integrated_rel_per_sample.tolist()
        ),
        "focus_0p07_3_integrated_log_error_per_sample": (
            focus_integrated_log_per_sample.tolist()
        ),
        "focus_0p08_3_integrated_log_error_per_sample": (
            focus_integrated_log_per_sample.tolist()
        ),
        "focus_0p1_5_integrated_log_error_per_sample": (
            legacy_focus_integrated_log_per_sample.tolist()
        ),
        "focus_0p1_3_integrated_log_error_per_sample": (
            legacy_focus_integrated_log_per_sample.tolist()
        ),
    }
    if p_linear_batch is not None:
        payload["p_linear_batch"] = p_linear_batch.tolist()
        payload["p_linear_mean"] = np.mean(p_linear_batch, axis=0).astype(np.float64).tolist()
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def write_test_set_results(
    run_dir: Path,
    test_thetas: np.ndarray,
    k_bins: np.ndarray,
    p_true_batch: np.ndarray,
    p_pred_batch: np.ndarray,
    p_linear_batch: np.ndarray | None = None,
    *,
    spectrum_type: str = "galaxy",
    eps_r: float = 1.0e-12,
    k_target_max: float = DEFAULT_K_TARGET_MAX,
    target_accuracy: float = 0.01,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write the shared test-set results format used by plotting and GP tools."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / TEST_SET_RESULTS_FILENAME
    payload = build_test_set_results_payload(
        test_thetas=test_thetas,
        k_bins=k_bins,
        p_true_batch=p_true_batch,
        p_pred_batch=p_pred_batch,
        p_linear_batch=p_linear_batch,
        spectrum_type=spectrum_type,
        eps_r=eps_r,
        k_target_max=k_target_max,
        target_accuracy=target_accuracy,
        metadata=metadata,
    )
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path
