"""Helpers for staged dynamic preprocessing during active learning."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from z2quijote.runtime_core.representation import build_k_band_masks


def compute_band_relative_errors(
    k_bins: np.ndarray,
    p_true_batch: np.ndarray,
    p_pred_batch: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Return mean relative error over the runtime-core low/mid/high k bands."""

    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    true_arr = np.asarray(p_true_batch, dtype=np.float64)
    pred_arr = np.asarray(p_pred_batch, dtype=np.float64)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(f"p_true_batch and p_pred_batch must align, got {true_arr.shape} vs {pred_arr.shape}.")
    if true_arr.ndim != 2 or true_arr.shape[1] != k_arr.shape[0]:
        raise ValueError(
            "p_true_batch and p_pred_batch must be 2D and align with k_bins, "
            f"got {true_arr.shape} vs {k_arr.shape}."
        )
    relative_error = np.abs(pred_arr - true_arr) / np.maximum(np.abs(true_arr), float(max(eps, 1.0e-30)))
    band_errors: list[float] = []
    for mask in build_k_band_masks(k_arr):
        mask_arr = np.asarray(mask, dtype=bool)
        if not np.any(mask_arr):
            band_errors.append(0.0)
            continue
        band_errors.append(float(np.mean(relative_error[:, mask_arr])))
    return np.asarray(band_errors, dtype=np.float64)


def compute_proxy_band_scores_from_sensitivity(
    sensitivity_matrix: np.ndarray,
) -> np.ndarray:
    """Fallback proxy when validation errors are unavailable."""

    sensitivity = np.asarray(sensitivity_matrix, dtype=np.float64)
    if sensitivity.ndim != 2 or sensitivity.shape[1] <= 0:
        raise ValueError(
            "sensitivity_matrix must have shape [num_components, num_bands], "
            f"got {sensitivity.shape}."
        )
    scores = np.mean(sensitivity, axis=0, dtype=np.float64)
    mean_score = float(np.mean(scores))
    if not np.isfinite(mean_score) or mean_score <= 0.0:
        return np.ones((int(sensitivity.shape[1]),), dtype=np.float64)
    return np.maximum(scores, 1.0e-12).astype(np.float64)


def compute_band_posterior_variance_scores(
    pc_variance_batch: np.ndarray,
    pca_band_sensitivity: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Project per-PC posterior variance into the runtime-core PCA k bands."""

    pc_variance = np.asarray(pc_variance_batch, dtype=np.float64)
    sensitivity = np.asarray(pca_band_sensitivity, dtype=np.float64)
    if pc_variance.ndim != 2:
        raise ValueError(f"pc_variance_batch must be 2D, got {pc_variance.shape}.")
    if sensitivity.ndim != 2 or sensitivity.shape[1] <= 0:
        raise ValueError(
            "pca_band_sensitivity must have shape [num_components, num_bands], "
            f"got {sensitivity.shape}."
        )
    if pc_variance.shape[1] != sensitivity.shape[0]:
        raise ValueError(
            "pc_variance_batch columns must align with pca_band_sensitivity rows, "
            f"got {pc_variance.shape} vs {sensitivity.shape}."
        )
    band_variance = np.mean(np.maximum(pc_variance, 0.0) @ sensitivity, axis=0)
    return np.maximum(band_variance, float(max(eps, 1.0e-30))).astype(np.float64)


def _mean_normalized_signal(values: Sequence[float], *, gamma: float, eps: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.ndim != 1 or arr.size <= 0:
        raise ValueError(f"signal values must be a non-empty 1D vector, got {arr.shape}.")
    safe = np.maximum(arr, float(max(eps, 1.0e-30)))
    mean_value = float(np.mean(safe))
    if not np.isfinite(mean_value) or mean_value <= 0.0:
        return np.ones_like(safe, dtype=np.float64)
    signal = np.power(safe / mean_value, float(max(gamma, 0.0)))
    signal = np.maximum(signal, float(max(eps, 1.0e-30)))
    return (signal / float(np.mean(signal))).astype(np.float64)


def _normalize_weight_vector(
    values: Sequence[float],
    *,
    length: int,
    field_name: str,
    eps: float,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (int(length),):
        raise ValueError(f"{field_name} must have shape [{int(length)}], got {arr.shape}.")
    arr = np.maximum(arr, float(max(eps, 1.0e-30)))
    return (arr / float(np.mean(arr))).astype(np.float64)


def _resolve_core_gate(
    errors: np.ndarray,
    core_band_indices: Sequence[int],
    *,
    core_error_good: float,
    core_error_bad: float,
    core_gate_floor: float,
    core_gate_ceiling: float,
) -> tuple[float, float, float, list[int]]:
    band_count = int(errors.size)
    indices: list[int] = []
    for raw_idx in core_band_indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= band_count:
            raise ValueError(
                f"core_band_indices must be within [0,{band_count - 1}], got {idx}."
            )
        if idx not in indices:
            indices.append(idx)
    if not indices:
        indices = list(range(band_count))
    core_error = float(np.mean(errors[np.asarray(indices, dtype=np.int64)]))
    good = float(max(core_error_good, 0.0))
    bad = float(max(core_error_bad, good + 1.0e-12))
    raw_gate = float(np.clip((core_error - good) / max(bad - good, 1.0e-12), 0.0, 1.0))
    smooth_gate = raw_gate * raw_gate * (3.0 - 2.0 * raw_gate)
    floor = float(np.clip(core_gate_floor, 0.0, 1.0))
    ceiling = float(np.clip(core_gate_ceiling, floor, 1.0))
    gate = float(floor + (ceiling - floor) * smooth_gate)
    return gate, raw_gate, core_error, indices


def compute_band_weight_balance_target(
    error_by_band: Sequence[float],
    *,
    gamma: float,
    balance_mode: str = "error_only",
    posterior_variance_by_band: Sequence[float] | None = None,
    posterior_variance_gamma: float = 0.5,
    posterior_variance_eta: float = 0.0,
    error_signal_eta: float = 1.0,
    core_band_indices: Sequence[int] = (1, 2, 3),
    core_error_good: float = 0.0035,
    core_error_bad: float = 0.0060,
    core_gate_floor: float = 0.15,
    core_gate_ceiling: float = 0.85,
    core_priority: Sequence[float] | None = None,
    release_priority: Sequence[float] | None = None,
    band_weight_prior: Sequence[float] | None = None,
    prior_eta: float = 1.0,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    """Build the target band-weight vector before temporal smoothing."""

    errors = np.asarray(error_by_band, dtype=np.float64).reshape(-1)
    if errors.ndim != 1 or errors.size <= 0:
        raise ValueError(f"error_by_band must be a non-empty 1D vector, got {errors.shape}.")
    band_count = int(errors.size)
    mode = str(balance_mode).strip().lower() or "error_only"
    if mode not in {"error_only", "core_posterior_variance"}:
        raise ValueError(
            "balance_mode must be one of {'error_only', 'core_posterior_variance'}, "
            f"got {balance_mode!r}."
        )

    error_signal = _mean_normalized_signal(errors, gamma=gamma, eps=eps)
    if mode == "error_only":
        if band_weight_prior is None:
            target = error_signal.copy()
            priority = np.ones((band_count,), dtype=np.float64)
        else:
            priority = _normalize_weight_vector(
                band_weight_prior,
                length=band_count,
                field_name="band_weight_prior",
                eps=eps,
            )
            eta = min(1.0, max(0.0, float(prior_eta)))
            target = (1.0 - eta) * priority + eta * error_signal
            target = target / float(np.mean(target))
        gate = 1.0
        raw_gate = 1.0
        core_error = float(np.mean(errors))
        core_indices = list(range(band_count))
        variance_signal = np.ones((band_count,), dtype=np.float64)
    else:
        if core_priority is None:
            core_priority = (0.30, 1.35, 1.30, 1.05)
        if release_priority is None:
            release_priority = (0.45, 1.25, 1.20, 1.10)
        core_prior = _normalize_weight_vector(
            core_priority,
            length=band_count,
            field_name="core_priority",
            eps=eps,
        )
        release_prior = _normalize_weight_vector(
            release_priority,
            length=band_count,
            field_name="release_priority",
            eps=eps,
        )
        gate, raw_gate, core_error, core_indices = _resolve_core_gate(
            np.maximum(errors, float(max(eps, 1.0e-30))),
            core_band_indices,
            core_error_good=float(core_error_good),
            core_error_bad=float(core_error_bad),
            core_gate_floor=float(core_gate_floor),
            core_gate_ceiling=float(core_gate_ceiling),
        )
        priority = gate * core_prior + (1.0 - gate) * release_prior
        priority = priority / float(np.mean(priority))
        if posterior_variance_by_band is None:
            variance_signal = np.ones((band_count,), dtype=np.float64)
        else:
            variance_arr = np.asarray(posterior_variance_by_band, dtype=np.float64).reshape(-1)
            if variance_arr.shape != errors.shape:
                raise ValueError(
                    "posterior_variance_by_band must align with error_by_band, "
                    f"got {variance_arr.shape} vs {errors.shape}."
                )
            variance_signal = _mean_normalized_signal(
                variance_arr,
                gamma=float(posterior_variance_gamma),
                eps=eps,
            )
        err_eta = float(np.clip(error_signal_eta, 0.0, 1.0))
        var_eta = float(np.clip(posterior_variance_eta, 0.0, 1.0))
        modulation = 1.0 + err_eta * (error_signal - 1.0) + var_eta * (variance_signal - 1.0)
        target = priority * np.maximum(modulation, float(max(eps, 1.0e-30)))
        target = target / float(np.mean(target))

    return {
        "target": target.astype(np.float64),
        "mode": mode,
        "error_signal": error_signal.astype(np.float64),
        "posterior_variance_signal": variance_signal.astype(np.float64),
        "priority": priority.astype(np.float64),
        "core_gate": float(gate),
        "core_gate_raw": float(raw_gate),
        "core_error": float(core_error),
        "core_error_good": float(core_error_good),
        "core_error_bad": float(core_error_bad),
        "core_gate_floor": float(core_gate_floor),
        "core_gate_ceiling": float(core_gate_ceiling),
        "core_band_indices": [int(value) for value in core_indices],
        "error_signal_eta": float(np.clip(error_signal_eta, 0.0, 1.0)),
        "posterior_variance_eta": float(np.clip(posterior_variance_eta, 0.0, 1.0)),
    }


def update_band_weights_from_errors(
    error_by_band: Sequence[float],
    previous_band_weights: Sequence[float],
    *,
    gamma: float,
    rho: float,
    weight_min: float,
    weight_max: float,
    balance_mode: str = "error_only",
    posterior_variance_by_band: Sequence[float] | None = None,
    posterior_variance_gamma: float = 0.5,
    posterior_variance_eta: float = 0.0,
    error_signal_eta: float = 1.0,
    core_band_indices: Sequence[int] = (1, 2, 3),
    core_error_good: float = 0.0035,
    core_error_bad: float = 0.0060,
    core_gate_floor: float = 0.15,
    core_gate_ceiling: float = 0.85,
    core_priority: Sequence[float] | None = None,
    release_priority: Sequence[float] | None = None,
    band_weight_prior: Sequence[float] | None = None,
    prior_eta: float = 1.0,
    return_metadata: bool = False,
    eps: float = 1.0e-12,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Smoothly update runtime-core band weights from current error and variance signals."""

    errors = np.asarray(error_by_band, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous_band_weights, dtype=np.float64).reshape(-1)
    if errors.ndim != 1 or errors.size <= 0:
        raise ValueError(f"error_by_band must be a non-empty 1D vector, got {errors.shape}.")
    if previous.shape != errors.shape:
        raise ValueError(
            "previous_band_weights must align with error_by_band, "
            f"got {previous.shape} vs {errors.shape}."
        )

    balance = compute_band_weight_balance_target(
        errors,
        gamma=float(gamma),
        balance_mode=balance_mode,
        posterior_variance_by_band=posterior_variance_by_band,
        posterior_variance_gamma=float(posterior_variance_gamma),
        posterior_variance_eta=float(posterior_variance_eta),
        error_signal_eta=float(error_signal_eta),
        core_band_indices=core_band_indices,
        core_error_good=float(core_error_good),
        core_error_bad=float(core_error_bad),
        core_gate_floor=float(core_gate_floor),
        core_gate_ceiling=float(core_gate_ceiling),
        core_priority=core_priority,
        release_priority=release_priority,
        band_weight_prior=band_weight_prior,
        prior_eta=float(prior_eta),
        eps=eps,
    )
    target = np.asarray(balance["target"], dtype=np.float64)
    updated = (1.0 - float(rho)) * previous + float(rho) * target
    updated = np.clip(updated, float(weight_min), float(weight_max))
    updated = updated / float(np.mean(updated))
    updated = np.clip(updated, float(weight_min), float(weight_max))
    updated = updated / float(np.mean(updated))
    updated = updated.astype(np.float64)
    if return_metadata:
        metadata = dict(balance)
        metadata["previous_band_weights"] = previous.astype(np.float64).tolist()
        metadata["updated_band_weights"] = updated.astype(np.float64).tolist()
        metadata["weight_min"] = float(weight_min)
        metadata["weight_max"] = float(weight_max)
        metadata["rho"] = float(rho)
        metadata["gamma"] = float(gamma)
        metadata["posterior_variance_gamma"] = float(posterior_variance_gamma)
        for key in ("target", "error_signal", "posterior_variance_signal", "priority"):
            metadata[key] = np.asarray(metadata[key], dtype=np.float64).tolist()
        return updated, metadata
    return updated


def allocate_band_components(
    residual_component_budget: int,
    band_weights: Sequence[float],
    previous_band_components: Sequence[int],
    *,
    allocation_lambda: float,
    min_band_components: int,
    max_delta_per_update: int,
) -> tuple[int, ...]:
    """Allocate residual PCA components across the current bands with bounded drift."""

    total = int(max(0, residual_component_budget))
    previous = np.asarray(previous_band_components, dtype=np.int64).reshape(-1)
    weights = np.asarray(band_weights, dtype=np.float64).reshape(-1)
    if previous.ndim != 1 or previous.size <= 0:
        raise ValueError(
            f"previous_band_components must be a non-empty 1D vector, got {previous.shape}."
        )
    if weights.shape != previous.shape:
        raise ValueError(f"band_weights must align with previous components, got {weights.shape} vs {previous.shape}.")
    if total <= 0:
        return tuple(0 for _ in previous.tolist())

    band_count = int(previous.size)
    min_count = min(int(max(0, min_band_components)), total // band_count if total >= band_count else 0)
    if min_count * band_count > total:
        min_count = 0

    blend = float(allocation_lambda) / float(band_count) + (1.0 - float(allocation_lambda)) * np.maximum(weights, 1.0e-12)
    blend = blend / float(np.sum(blend))
    target = np.floor(total * blend).astype(np.int64)
    remainder = total - int(np.sum(target))
    if remainder > 0:
        order = np.argsort(-(total * blend - target), kind="mergesort")
        for idx in order[:remainder]:
            target[int(idx)] += 1

    target = np.maximum(target, min_count)
    overflow = int(np.sum(target) - total)
    if overflow > 0:
        order = np.argsort(target - total * blend, kind="mergesort")[::-1]
        for idx in order:
            reducible = int(target[int(idx)] - min_count)
            if reducible <= 0:
                continue
            delta = min(reducible, overflow)
            target[int(idx)] -= delta
            overflow -= delta
            if overflow <= 0:
                break

    bounded = target.copy()
    max_delta = int(max(1, max_delta_per_update))
    for idx in range(band_count):
        low = max(min_count, int(previous[idx]) - max_delta)
        high = int(previous[idx]) + max_delta
        bounded[idx] = int(np.clip(bounded[idx], low, high))

    diff = total - int(np.sum(bounded))
    if diff != 0:
        priority = np.argsort(-(target - bounded), kind="mergesort") if diff > 0 else np.argsort(
            -(bounded - target),
            kind="mergesort",
        )
        for idx in priority:
            index = int(idx)
            if diff > 0:
                bounded[index] += 1
                diff -= 1
            else:
                if bounded[index] <= min_count:
                    continue
                bounded[index] -= 1
                diff += 1
            if diff == 0:
                break

    if int(np.sum(bounded)) != total:
        balanced = np.full((band_count,), total // band_count, dtype=np.int64)
        balanced[: total % band_count] += 1
        return tuple(int(value) for value in balanced.tolist())

    return tuple(int(value) for value in bounded.tolist())


def merge_band_weights_to_grid_fractions(
    band_weights: Sequence[float],
    previous_grid_fractions: Sequence[float],
    *,
    rho: float,
) -> tuple[float, ...]:
    """Smooth band weights into grid fractions for the active k-band scheme."""

    weights = np.asarray(band_weights, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous_grid_fractions, dtype=np.float64).reshape(-1)
    if previous.shape != weights.shape:
        raise ValueError(
            "previous_grid_fractions must align with band_weights, "
            f"got {previous.shape} vs {weights.shape}."
        )
    merged = weights.astype(np.float64)
    merged = merged / float(np.sum(merged))
    updated = (1.0 - float(rho)) * previous + float(rho) * merged
    updated = np.maximum(updated, 1.0e-6)
    updated = updated / float(np.sum(updated))
    return tuple(float(value) for value in updated.tolist())


# Compatibility alias for older call sites and historical notebooks.
compute_four_band_relative_errors = compute_band_relative_errors
