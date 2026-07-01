"""Module 2 facade: per-component GP fitting and spectrum reconstruction."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from z2quijote.runtime_core.config import ValidationRuntimeConfig
from z2quijote.runtime_core.data_source import active_theta_bounds
from z2quijote.runtime_core.representation import (
    PCA_BAND_LABELS,
    build_k_band_masks,
    parse_target_transform,
    reconstruct_power_from_target,
    resolve_target_transform_from_metadata,
)
from z2quijote.runtime_core.sampling import ensure_2d_theta_batch
from z2quijote.runtime_core.types import (
    ContinuousPosteriorState,
    ContinuousVarianceEvaluation,
    EmulatorState,
    Module1Dataset,
    SpectrumPrediction,
)

ProgressCallback = Callable[[str, int, int], None]

_LOGDIFF_DEFAULT_CURVE_LEVELS = np.asarray([0.30, 1.35, 1.30, 1.05], dtype=np.float64)
_LOGDIFF_DEFAULT_CURVE_TRANSITION_DEX = 0.10


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


def _compute_pca_band_sensitivity(
    pca_components: np.ndarray,
    k_bins: np.ndarray,
) -> np.ndarray:
    band_energy, total_energy = _compute_pca_band_variance_integrals(pca_components, k_bins)
    return (band_energy / total_energy[:, None]).astype(np.float64)


def _compute_pca_band_variance_integrals(
    pca_components: np.ndarray,
    k_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    components = np.asarray(pca_components, dtype=np.float64)
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    if components.ndim != 2 or components.shape[1] != k_arr.shape[0]:
        raise ValueError(
            "pca_components must be 2D and align with k_bins, "
            f"got {components.shape} vs {k_arr.shape}."
        )
    integration_weights = _build_logk_trapezoid_weights(k_arr)
    band_masks = np.vstack([mask.astype(np.float64) for mask in build_k_band_masks(k_arr)])
    component_energy = np.square(components)
    weighted_energy = component_energy * integration_weights[None, :]
    total_energy = np.sum(weighted_energy, axis=1, keepdims=True)
    total_energy = np.maximum(total_energy, 1.0e-30)
    band_energy = weighted_energy @ band_masks.T
    return band_energy.astype(np.float64), total_energy.reshape(-1).astype(np.float64)


def compute_pca_band_sensitivity(
    pca_components: np.ndarray,
    k_bins: np.ndarray,
) -> np.ndarray:
    """Return each PCA component's normalized sensitivity across the runtime-core k bands."""

    return _compute_pca_band_sensitivity(pca_components, k_bins)


def build_logdiff_projected_component_weights(
    pca_components: np.ndarray,
    k_bins: np.ndarray,
    *,
    band_multipliers: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    component_groups: Sequence[dict[str, Any]] | None = None,
    global_weight: float = 1.0,
    base_band_levels: Sequence[float] = tuple(_LOGDIFF_DEFAULT_CURVE_LEVELS.tolist()),
    transition_dex: float = _LOGDIFF_DEFAULT_CURVE_TRANSITION_DEX,
) -> dict[str, Any]:
    """Project a smooth logdiff k-weight curve onto PCA components via squared loadings."""

    components = np.asarray(pca_components, dtype=np.float64)
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    if components.ndim != 2 or components.shape[1] != k_arr.shape[0]:
        raise ValueError(
            "pca_components must be 2D and align with k_bins, "
            f"got {components.shape} vs {k_arr.shape}."
        )

    multipliers = np.asarray(band_multipliers, dtype=np.float64).reshape(-1)
    band_count = len(PCA_BAND_LABELS)
    if multipliers.shape != (band_count,):
        raise ValueError(f"band_multipliers must have shape [{band_count}], got {multipliers.shape}.")
    base_levels_arr = np.asarray(base_band_levels, dtype=np.float64).reshape(-1)
    if base_levels_arr.shape != (band_count,):
        raise ValueError(f"base_band_levels must have shape [{band_count}], got {base_levels_arr.shape}.")

    logk = np.log10(np.maximum(k_arr, 1.0e-30))
    integration_weights = _build_logk_trapezoid_weights(k_arr)
    effective_levels = np.maximum(base_levels_arr * multipliers, 1.0e-8)
    width = float(max(1.0e-4, transition_dex))
    curve = np.full_like(k_arr, effective_levels[0], dtype=np.float64)
    transitions = np.asarray([np.log10(0.07), np.log10(0.5), np.log10(1.0)], dtype=np.float64)
    for idx, boundary in enumerate(transitions):
        smooth_step = 0.5 * (1.0 + np.tanh((logk - boundary) / width))
        curve = (1.0 - smooth_step) * curve + smooth_step * effective_levels[idx + 1]
    curve = np.maximum(curve, 1.0e-8)

    curve_mean = float(
        np.sum(curve * integration_weights) / max(np.sum(integration_weights), 1.0e-30)
    )
    normalized_curve = curve / max(curve_mean, 1.0e-30)

    raw_projection = np.sum(
        np.square(components) * integration_weights[None, :] * normalized_curve[None, :],
        axis=1,
    )
    raw_projection = np.maximum(raw_projection, 1.0e-30)

    group_scaling = np.ones((components.shape[0],), dtype=np.float64)
    if component_groups is not None:
        for group in component_groups:
            if str(group.get("group_kind", "")).strip().lower() != "global":
                continue
            indices = np.asarray(group.get("component_indices", []), dtype=np.int64).reshape(-1)
            if indices.size <= 0:
                continue
            valid = indices[(indices >= 0) & (indices < group_scaling.shape[0])]
            if valid.size > 0:
                group_scaling[valid] *= float(max(global_weight, 1.0e-8))

    scaled_projection = raw_projection * group_scaling
    component_weights = scaled_projection / float(np.mean(scaled_projection))

    band_masks = list(build_k_band_masks(k_arr))
    band_curve_means: list[float] = []
    for mask in band_masks:
        if not np.any(mask):
            band_curve_means.append(0.0)
            continue
        band_curve_means.append(
            float(
                np.sum(normalized_curve[mask] * integration_weights[mask])
                / max(np.sum(integration_weights[mask]), 1.0e-30)
            )
        )

    return {
        "component_weights": component_weights.astype(np.float64),
        "details": {
            "mode": "projected_logdiff_k_curve",
            "base_band_levels": base_levels_arr.astype(np.float64).tolist(),
            "band_multipliers": multipliers.astype(np.float64).tolist(),
            "effective_band_levels": effective_levels.astype(np.float64).tolist(),
            "transition_dex": float(width),
            "k_weight_curve": normalized_curve.astype(np.float64).tolist(),
            "k_weight_curve_mean": float(
                np.sum(normalized_curve * integration_weights)
                / max(np.sum(integration_weights), 1.0e-30)
            ),
            "k_weight_curve_band_means": band_curve_means,
            "raw_component_projection": raw_projection.astype(np.float64).tolist(),
            "component_group_scaling": group_scaling.astype(np.float64).tolist(),
            "scaled_component_projection": scaled_projection.astype(np.float64).tolist(),
            "component_weights": component_weights.astype(np.float64).tolist(),
        },
    }


def _coerce_pca_band_labels(
    labels: Sequence[str] | None,
) -> list[str]:
    raw_labels = list(PCA_BAND_LABELS) if labels is None else [str(item).strip() for item in labels]
    if len(raw_labels) != len(PCA_BAND_LABELS) or any(not label for label in raw_labels):
        return list(PCA_BAND_LABELS)
    return raw_labels


def build_pca_component_weights_from_band_sensitivity(
    sensitivity: np.ndarray,
    *,
    alpha: np.ndarray | Sequence[float],
    beta: float,
    weight_min: float,
    weight_max: float,
    weight_function: str = "stable_log_tanh",
    weight_temperature: float = 0.75,
) -> np.ndarray:
    """Map PCA band sensitivities to a stable per-component weight vector."""

    sensitivity_arr = np.asarray(sensitivity, dtype=np.float64)
    band_count = len(PCA_BAND_LABELS)
    if sensitivity_arr.ndim != 2 or sensitivity_arr.shape[1] != band_count:
        raise ValueError(
            f"sensitivity must have shape [num_components, {band_count}], "
            f"got {sensitivity_arr.shape}."
        )
    alpha_vec = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if alpha_vec.shape != (band_count,):
        raise ValueError(f"alpha must have shape [{band_count}], got {alpha_vec.shape}.")
    beta_value = float(beta)
    if beta_value <= 0.0:
        return np.ones((sensitivity_arr.shape[0],), dtype=np.float64)

    weight_fn = str(weight_function).strip().lower() or "stable_log_tanh"
    weighted = sensitivity_arr @ alpha_vec
    weighted = np.maximum(weighted, 1.0e-12)

    if weight_fn == "linear_blend":
        mean_weighted = float(np.mean(weighted))
        if not np.isfinite(mean_weighted) or mean_weighted <= 0.0:
            raise ValueError("Computed weighted PCA band scores are invalid.")
        normalized = weighted / mean_weighted
        clipped = np.clip(normalized, float(weight_min), float(weight_max))
        clipped_mean = float(np.mean(clipped))
        if not np.isfinite(clipped_mean) or clipped_mean <= 0.0:
            raise ValueError("Clipped PCA component weights are invalid.")
        normalized_clipped = clipped / clipped_mean
        component_weights = (1.0 - beta_value) + beta_value * normalized_clipped
        component_weights = component_weights / float(np.mean(component_weights))
        return component_weights.astype(np.float64)

    if weight_fn != "stable_log_tanh":
        raise ValueError(
            "weight_function must be one of {'linear_blend', 'stable_log_tanh'}, "
            f"got {weight_function!r}."
        )

    temperature = float(max(weight_temperature, 1.0e-6))
    centered_log_ratio = np.log(weighted) - float(np.mean(np.log(weighted)))
    compressed = np.tanh(centered_log_ratio / temperature)
    component_weights = 1.0 + beta_value * compressed
    component_weights = np.clip(component_weights, float(weight_min), float(weight_max))
    component_weights = component_weights / float(np.mean(component_weights))
    component_weights = np.clip(component_weights, float(weight_min), float(weight_max))
    component_weights = component_weights / float(np.mean(component_weights))
    return component_weights.astype(np.float64)


def build_iteration_pca_band_diagnostics(
    config: ValidationRuntimeConfig,
    continuous_state: ContinuousPosteriorState,
    *,
    iteration_index: int | None = None,
) -> dict[str, Any]:
    """Build a per-iteration PCA band-sensitivity report using existing training data only."""

    sensitivity = np.asarray(
        continuous_state.metadata.get("pca_band_sensitivity", []),
        dtype=np.float64,
    )
    component_count = int(len(continuous_state.gp_models))
    band_count = len(PCA_BAND_LABELS)
    if sensitivity.shape != (component_count, band_count):
        raise ValueError(
            "continuous_state.metadata['pca_band_sensitivity'] must have shape "
            f"({component_count}, {band_count}), got {sensitivity.shape}."
        )

    band_labels = _coerce_pca_band_labels(
        continuous_state.metadata.get("pca_band_labels", list(PCA_BAND_LABELS))
    )
    focus_alpha = np.asarray(
        [
            float(config.m3.pc_weight_alpha_low),
            float(config.m3.pc_weight_alpha_mid),
            float(config.m3.pc_weight_alpha_focus_high),
            float(config.m3.pc_weight_alpha_tail),
        ],
        dtype=np.float64,
    )
    band_alpha_matrix = np.asarray(
        [
            config.m3.band_alpha_low,
            config.m3.band_alpha_mid,
            config.m3.band_alpha_focus_high,
            config.m3.band_alpha_tail,
        ],
        dtype=np.float64,
    )
    band_betas = np.asarray(
        [
            float(config.m3.band_beta_low),
            float(config.m3.band_beta_mid),
            float(config.m3.band_beta_focus_high),
            float(config.m3.band_beta_tail),
        ],
        dtype=np.float64,
    )

    weight_kwargs = {
        "weight_function": str(config.m3.weight_function),
        "weight_temperature": float(config.m3.weight_temperature),
    }
    focus_weights = build_pca_component_weights_from_band_sensitivity(
        sensitivity,
        alpha=focus_alpha,
        beta=float(config.m3.pc_weight_beta),
        weight_min=float(config.m3.pc_weight_min),
        weight_max=float(config.m3.pc_weight_max),
        **weight_kwargs,
    )
    band_weight_matrix = np.vstack(
        [
            build_pca_component_weights_from_band_sensitivity(
                sensitivity,
                alpha=band_alpha_matrix[band_idx],
                beta=float(band_betas[band_idx]),
                weight_min=float(config.m3.band_weight_min),
                weight_max=float(config.m3.band_weight_max),
                **weight_kwargs,
            )
            for band_idx in range(band_count)
        ]
    ).astype(np.float64)

    dominant_band_idx = np.argmax(sensitivity, axis=1).astype(np.int64)
    dominant_band_labels = [band_labels[int(idx)] for idx in dominant_band_idx]
    focus_sensitivity = np.sum(sensitivity[:, 1:], axis=1).astype(np.float64)
    focus_rank = np.argsort(-focus_sensitivity, kind="mergesort").astype(np.int64)
    top_components_by_band = []
    top_count = min(5, component_count)
    for band_idx, band_label in enumerate(band_labels):
        top_indices = np.argsort(-sensitivity[:, band_idx], kind="mergesort")[:top_count].astype(np.int64)
        top_components_by_band.append(
            {
                "band_index": int(band_idx),
                "band_label": str(band_label),
                "top_component_indices": top_indices.tolist(),
                "top_component_sensitivity": sensitivity[top_indices, band_idx].astype(np.float64).tolist(),
            }
        )

    return {
        "iteration_index": None if iteration_index is None else int(iteration_index),
        "evaluation_mode": "existing_training_data_only",
        "component_count": int(component_count),
        "band_labels": list(band_labels),
        "sensitivity_matrix": sensitivity.astype(np.float64).tolist(),
        "band_mean_sensitivity": np.mean(sensitivity, axis=0).astype(np.float64).tolist(),
        "band_total_sensitivity": np.sum(sensitivity, axis=0).astype(np.float64).tolist(),
        "dominant_band_index": dominant_band_idx.tolist(),
        "dominant_band_label": dominant_band_labels,
        "mid_high_sensitivity": focus_sensitivity.tolist(),
        "mid_high_rank_desc": focus_rank.tolist(),
        "focus_0p08_3_sensitivity": focus_sensitivity.tolist(),
        "focus_0p08_3_rank_desc": focus_rank.tolist(),
        "focus_0p1_3_sensitivity": focus_sensitivity.tolist(),
        "focus_0p1_3_rank_desc": focus_rank.tolist(),
        "focus_top_components": focus_rank[:top_count].tolist(),
        "top_components_by_band": top_components_by_band,
        "recommended_weight_function": str(config.m3.weight_function),
        "recommended_weight_temperature": float(config.m3.weight_temperature),
        "recommended_focus_alpha": focus_alpha.astype(np.float64).tolist(),
        "recommended_focus_beta": float(config.m3.pc_weight_beta),
        "recommended_focus_weights": focus_weights.astype(np.float64).tolist(),
        "recommended_band_alpha_matrix": band_alpha_matrix.astype(np.float64).tolist(),
        "recommended_band_betas": band_betas.astype(np.float64).tolist(),
        "recommended_band_weight_matrix": band_weight_matrix.astype(np.float64).tolist(),
    }


def summarize_lengthscale_upper_hits(
    config: ValidationRuntimeConfig,
    emulator: EmulatorState,
    *,
    rtol: float = 1.0e-3,
    atol: float = 1.0e-8,
) -> dict[str, Any]:
    upper_bound = float(config.gp.length_scale_bounds_high)
    pc_lengthscales = np.vstack(
        [_extract_component_lengthscales(gp) for gp in emulator.gp_models]
    ).astype(np.float64)
    hit_mask = np.isclose(
        pc_lengthscales,
        upper_bound,
        rtol=float(max(rtol, 0.0)),
        atol=float(max(atol, 0.0)),
    ) | (pc_lengthscales >= upper_bound * (1.0 - float(max(rtol, 0.0))))
    return {
        "upper_bound": float(upper_bound),
        "hit_count_total": int(np.sum(hit_mask)),
        "dimension_count_total": int(hit_mask.size),
        "hit_fraction_total": float(np.mean(hit_mask)) if hit_mask.size > 0 else 0.0,
        "hit_count_per_component": np.sum(hit_mask, axis=1).astype(np.int64).tolist(),
        "hit_mask_per_component": hit_mask.astype(bool).tolist(),
    }


def _build_kernel(config: ValidationRuntimeConfig, theta_dim: int):
    gp_cfg = config.gp
    return ConstantKernel(
        constant_value=float(gp_cfg.constant_value),
        constant_value_bounds=(
            float(gp_cfg.constant_value_bounds_low),
            float(gp_cfg.constant_value_bounds_high),
        ),
    ) * RBF(
        length_scale=np.full((theta_dim,), float(gp_cfg.length_scale_initial), dtype=np.float64),
        length_scale_bounds=(
            float(gp_cfg.length_scale_bounds_low),
            float(gp_cfg.length_scale_bounds_high),
        ),
    )


def _extract_component_lengthscales(gp: GaussianProcessRegressor) -> np.ndarray:
    kernel = gp.kernel_
    rbf = getattr(kernel, "k2", None)
    length_scale = getattr(rbf, "length_scale", None)
    if length_scale is None:
        raise ValueError("Unable to extract ARD length scales from fitted GP kernel.")
    return np.asarray(length_scale, dtype=np.float64).reshape(-1)


def _extract_component_signal_variance(gp: GaussianProcessRegressor) -> float:
    kernel = gp.kernel_
    constant = getattr(kernel, "k1", None)
    value = getattr(constant, "constant_value", None)
    if value is None:
        raise ValueError("Unable to extract signal variance from fitted GP kernel.")
    return float(value)


def _interp_logk_batch(batch: np.ndarray, source_k: np.ndarray, target_k: np.ndarray) -> np.ndarray:
    source = np.asarray(source_k, dtype=np.float64).reshape(-1)
    target = np.asarray(target_k, dtype=np.float64).reshape(-1)
    batch_arr = np.asarray(batch, dtype=np.float64)
    if batch_arr.ndim != 2 or batch_arr.shape[1] != source.shape[0]:
        raise ValueError(
            "batch must be 2D and align with source_k, "
            f"got {batch_arr.shape} vs {source.shape}."
        )
    if target.shape == source.shape and np.allclose(target, source):
        return batch_arr.astype(np.float64)
    source_min = float(source[0])
    source_max = float(source[-1])
    target_min = float(target[0])
    target_max = float(target[-1])
    if target_min < source_min * (1.0 - 1.0e-10) or target_max > source_max * (1.0 + 1.0e-10):
        raise ValueError(
            "Interpolation target k grid extends outside the available source coverage, "
            f"got source [{source_min}, {source_max}] and target [{target_min}, {target_max}]."
        )
    log_source = np.log10(np.maximum(source, 1.0e-30))
    log_target = np.log10(np.maximum(target, 1.0e-30))
    rows: list[np.ndarray] = []
    for row in batch_arr:
        interp = PchipInterpolator(log_source, row, extrapolate=False)
        rows.append(np.asarray(interp(log_target), dtype=np.float64))
    return np.vstack(rows).astype(np.float64)


def _resolve_linear_base_batch(
    emulator: EmulatorState,
    raw_thetas: np.ndarray,
    source_k: np.ndarray,
    target_k: np.ndarray,
    p_linear_batch: np.ndarray | None,
) -> np.ndarray:
    provided = None if p_linear_batch is None else np.asarray(p_linear_batch, dtype=np.float64)
    if provided is None:
        dataset_linear = emulator.dataset.p_linear_batch
        if dataset_linear is not None:
            training_raw = np.asarray(emulator.dataset.raw_thetas, dtype=np.float64)
            if training_raw.shape == raw_thetas.shape and np.allclose(training_raw, raw_thetas):
                provided = np.asarray(dataset_linear, dtype=np.float64)
    if provided is None:
        raise ValueError(
            "anchored prediction requires p_linear_batch for the queried thetas "
            "or an exact match against training samples with stored anchor spectra."
        )
    if provided.ndim != 2 or provided.shape[0] != raw_thetas.shape[0]:
        raise ValueError(
            "p_linear_batch must be 2D and align with theta_batch, "
            f"got {provided.shape} vs {raw_thetas.shape}."
        )
    if provided.shape[1] == target_k.shape[0]:
        return provided.astype(np.float64)
    if provided.shape[1] == source_k.shape[0]:
        return _interp_logk_batch(provided, source_k, target_k)
    raise ValueError(
        "p_linear_batch column count must match either source or target k grid, "
        f"got {provided.shape[1]} vs {source_k.shape[0]} or {target_k.shape[0]}."
    )


def _resolve_target_anchor_batch(
    emulator: EmulatorState,
    raw_thetas: np.ndarray,
    source_k: np.ndarray,
    target_k: np.ndarray,
    *,
    target_transform: str,
    p_linear_batch: np.ndarray | None,
) -> np.ndarray | None:
    transform_family, anchor_mode = parse_target_transform(target_transform)
    if transform_family == "direct_logpk":
        return None
    if anchor_mode in {"linear", "halofit", "hmcode2020"}:
        return _resolve_linear_base_batch(
            emulator,
            raw_thetas,
            source_k,
            target_k,
            p_linear_batch,
        )
    raise NotImplementedError(
        f"Target transform {target_transform!r} requires anchor mode {anchor_mode!r}, "
        "which is not yet persisted by the active-learning dataset."
    )


def build_continuous_posterior_state(emulator: EmulatorState) -> ContinuousPosteriorState:
    if not emulator.gp_models:
        raise ValueError("Emulator has no fitted GP models.")
    pc_lengthscales = np.vstack(
        [_extract_component_lengthscales(gp) for gp in emulator.gp_models]
    ).astype(np.float64)
    pc_signal_variances = np.asarray(
        [_extract_component_signal_variance(gp) for gp in emulator.gp_models],
        dtype=np.float64,
    )
    dataset = emulator.dataset
    score_std = np.asarray(dataset.pca_score_std, dtype=np.float64).reshape(-1)
    pca_band_integrals, pca_global_integral = _compute_pca_band_variance_integrals(
        np.asarray(dataset.pca_components, dtype=np.float64),
        np.asarray(dataset.k_bins, dtype=np.float64),
    )
    pca_band_sensitivity = _compute_pca_band_sensitivity(
        np.asarray(dataset.pca_components, dtype=np.float64),
        np.asarray(dataset.k_bins, dtype=np.float64),
    )
    mid_high_sensitivity = np.sum(pca_band_sensitivity[:, 1:], axis=1)
    target_transform = resolve_target_transform_from_metadata(
        dataset.metadata,
        transform_family=str(dataset.metadata.get("representation_transform_family", "logdiff")),
        anchor_mode=str(dataset.metadata.get("representation_anchor_mode", "linear")),
    )
    return ContinuousPosteriorState(
        theta_bounds=np.asarray(emulator.theta_bounds, dtype=np.float64),
        train_raw_thetas=np.asarray(dataset.raw_thetas, dtype=np.float64),
        train_unit_thetas=np.asarray(dataset.unit_thetas, dtype=np.float64),
        gp_models=list(emulator.gp_models),
        kernel_descriptions=list(emulator.kernel_descriptions),
        pc_lengthscales=pc_lengthscales,
        pc_signal_variances=pc_signal_variances,
        metadata={
            "train_size": int(dataset.unit_thetas.shape[0]),
            "theta_dim": int(dataset.unit_thetas.shape[1]),
            "pc_dim": int(len(emulator.gp_models)),
            "kernel_family": str(emulator.metadata.get("kernel_family", "")),
            "pca_score_std": score_std.tolist(),
            "pca_band_labels": list(PCA_BAND_LABELS),
            "pca_band_sensitivity": pca_band_sensitivity.tolist(),
            "pca_band_variance_integrals": pca_band_integrals.tolist(),
            "pca_global_variance_integral": pca_global_integral.tolist(),
            "pca_mid_high_sensitivity": mid_high_sensitivity.astype(np.float64).tolist(),
            "pca_focus_0p08_3_sensitivity": mid_high_sensitivity.astype(np.float64).tolist(),
            "pca_focus_0p1_3_sensitivity": mid_high_sensitivity.astype(np.float64).tolist(),
            "pca_scheme": str(dataset.metadata.get("pca_scheme", "global_pca")),
            "pca_layout": dict(dataset.metadata.get("pca_layout", {})),
            "representation_component_groups": list(
                dataset.metadata.get("pca_layout", {}).get("component_groups", [])
            ),
            "target_transform": target_transform,
        },
    )


def evaluate_continuous_variance(
    continuous_state: ContinuousPosteriorState,
    theta_batch: np.ndarray,
    *,
    input_space: str = "unit",
    pc_indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
) -> ContinuousVarianceEvaluation:
    raw_thetas, unit_thetas = ensure_2d_theta_batch(
        np.asarray(theta_batch, dtype=np.float64),
        continuous_state.theta_bounds,
        input_space=input_space,
    )
    if pc_indices is None:
        selected_indices = np.arange(len(continuous_state.gp_models), dtype=np.int64)
    else:
        selected_indices = np.asarray(pc_indices, dtype=np.int64).reshape(-1)
        if selected_indices.size == 0:
            raise ValueError("pc_indices must be non-empty when provided.")

    scaled_std_cols: list[np.ndarray] = []
    for pc_idx in selected_indices:
        gp = continuous_state.gp_models[int(pc_idx)]
        _, std_col = gp.predict(unit_thetas, return_std=True)
        scaled_std_cols.append(np.asarray(std_col, dtype=np.float64).reshape(-1, 1))

    scaled_pc_std = np.hstack(scaled_std_cols).astype(np.float64)
    all_score_std = np.asarray(
        continuous_state.metadata.get("pca_score_std", []),
        dtype=np.float64,
    ).reshape(-1)
    score_std = all_score_std[selected_indices].reshape(1, -1)
    pc_std = scaled_pc_std * score_std
    pc_var = np.square(pc_std)
    return ContinuousVarianceEvaluation(
        raw_thetas=raw_thetas,
        unit_thetas=unit_thetas,
        pc_var=pc_var,
        pc_std=pc_std,
        metadata={
            "input_space": str(input_space).strip().lower(),
            "query_size": int(unit_thetas.shape[0]),
            "selected_pc_indices": selected_indices.astype(np.int64).tolist(),
        },
    )


def fit_emulator(
    config: ValidationRuntimeConfig,
    dataset: Module1Dataset,
    *,
    progress_callback: ProgressCallback | None = None,
) -> EmulatorState:
    unit_thetas = np.asarray(dataset.unit_thetas, dtype=np.float64)
    pca_scores = np.asarray(dataset.pca_scores, dtype=np.float64)
    if unit_thetas.ndim != 2 or pca_scores.ndim != 2:
        raise ValueError("dataset.unit_thetas and dataset.pca_scores must both be 2D.")
    if unit_thetas.shape[0] != pca_scores.shape[0]:
        raise ValueError("unit_thetas and pca_scores row counts must match.")

    gp_models: list[GaussianProcessRegressor] = []
    kernel_descriptions: list[str] = []
    theta_dim = unit_thetas.shape[1]
    for pc_idx in range(pca_scores.shape[1]):
        if progress_callback is not None:
            progress_callback("module2_gp_fit", pc_idx + 1, pca_scores.shape[1])
        gp = GaussianProcessRegressor(
            kernel=_build_kernel(config, theta_dim),
            alpha=float(config.gp.alpha),
            normalize_y=bool(config.gp.normalize_y),
            random_state=int(config.random_seed + pc_idx),
            n_restarts_optimizer=int(config.gp.n_restarts_optimizer),
        )
        gp.fit(unit_thetas, pca_scores[:, pc_idx])
        gp_models.append(gp)
        kernel_descriptions.append(str(gp.kernel_))

    return EmulatorState(
        dataset=dataset,
        gp_models=gp_models,
        theta_bounds=active_theta_bounds(config),
        kernel_descriptions=kernel_descriptions,
        metadata={
            "train_size": int(unit_thetas.shape[0]),
            "pca_components": int(pca_scores.shape[1]),
            "kernel_family": "ConstantKernel * RBF(ARD)",
        },
    )


def predict_spectra(
    emulator: EmulatorState,
    theta_batch: np.ndarray,
    *,
    input_space: str = "raw",
    k_target: np.ndarray | None = None,
    p_linear_batch: np.ndarray | None = None,
) -> SpectrumPrediction:
    raw_thetas, unit_thetas = ensure_2d_theta_batch(
        np.asarray(theta_batch, dtype=np.float64),
        emulator.theta_bounds,
        input_space=input_space,
    )
    if not emulator.gp_models:
        raise ValueError("Emulator has no fitted GP models.")

    scaled_mean_cols: list[np.ndarray] = []
    scaled_std_cols: list[np.ndarray] = []
    for gp in emulator.gp_models:
        mean_col, std_col = gp.predict(unit_thetas, return_std=True)
        scaled_mean_cols.append(np.asarray(mean_col, dtype=np.float64).reshape(-1, 1))
        scaled_std_cols.append(np.asarray(std_col, dtype=np.float64).reshape(-1, 1))

    scaled_pc_mean = np.hstack(scaled_mean_cols).astype(np.float64)
    scaled_pc_std = np.hstack(scaled_std_cols).astype(np.float64)
    score_mean = np.asarray(emulator.dataset.pca_score_mean, dtype=np.float64).reshape(1, -1)
    score_std = np.asarray(emulator.dataset.pca_score_std, dtype=np.float64).reshape(1, -1)
    pc_mean = scaled_pc_mean * score_std + score_mean
    pc_std = scaled_pc_std * score_std
    source_k = np.asarray(emulator.dataset.k_bins, dtype=np.float64).reshape(-1)
    k_bins = source_k
    if k_target is not None:
        target = np.asarray(k_target, dtype=np.float64).reshape(-1)
        if np.any(target <= 0.0):
            raise ValueError("k_target must be strictly positive.")
        k_bins = target

    target_mean_source = np.asarray(
        emulator.dataset.pca_model.inverse_transform(pc_mean),
        dtype=np.float64,
    )
    target_mean = _interp_logk_batch(target_mean_source, source_k, k_bins)
    target_transform = resolve_target_transform_from_metadata(
        emulator.dataset.metadata,
        transform_family=str(
            emulator.dataset.metadata.get("representation_transform_family", "logdiff")
        ),
        anchor_mode=str(emulator.dataset.metadata.get("representation_anchor_mode", "linear")),
    )
    anchor_batch = _resolve_target_anchor_batch(
        emulator,
        raw_thetas,
        source_k,
        k_bins,
        target_transform=target_transform,
        p_linear_batch=p_linear_batch,
    )
    pk_mean, log_pk_mean = reconstruct_power_from_target(
        target_mean,
        target_transform=target_transform,
        anchor_batch=anchor_batch,
        power_eps=float(emulator.dataset.metadata.get("power_eps", 1.0e-12)),
    )
    return SpectrumPrediction(
        raw_thetas=raw_thetas,
        unit_thetas=unit_thetas,
        k_bins=k_bins,
        pc_mean=pc_mean,
        pc_std=pc_std,
        target_mean=target_mean,
        log_pk_mean=log_pk_mean,
        pk_mean=pk_mean,
        p_linear_batch=anchor_batch,
        metadata={
            "input_space": str(input_space).strip().lower(),
            "query_size": int(raw_thetas.shape[0]),
            "target_transform": target_transform,
            "pca_scheme": str(emulator.dataset.metadata.get("pca_scheme", "global_pca")),
        },
    )
