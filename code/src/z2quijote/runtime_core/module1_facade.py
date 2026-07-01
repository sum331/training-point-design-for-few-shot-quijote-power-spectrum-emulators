"""Module 1 facade: Sobol sampling, CAMB evaluation, and PCA preprocessing."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

from z2quijote.runtime_core.camb_data_provider import CAMBAccuracyConfig, CAMBDataProvider
from z2quijote.runtime_core.config import (
    ValidationRuntimeConfig,
    build_default_k_bins,
    normalize_theta_batch,
)
from z2quijote.runtime_core.data_source import active_theta_bounds, resolve_data_source
from z2quijote.runtime_core.quijote_k_grid import maybe_build_quijote_output_k_bins
from z2quijote.runtime_core.representation import build_target_representation, fit_representation_pca
from z2quijote.runtime_core.sampling import generate_sobol_thetas
from z2quijote.runtime_core.types import Module1Dataset

ProgressCallback = Callable[[str, int, int], None]


def _ensure_2d(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.shape}.")
    return arr


def _standardize_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_arr = np.asarray(scores, dtype=np.float64)
    if score_arr.ndim != 2:
        raise ValueError(f"scores must be 2D, got {score_arr.shape}.")
    mean = np.mean(score_arr, axis=0, dtype=np.float64)
    std = np.std(score_arr, axis=0, dtype=np.float64)
    std = np.maximum(std, 1.0e-12)
    scaled = (score_arr - mean[None, :]) / std[None, :]
    return scaled.astype(np.float64), mean.astype(np.float64), std.astype(np.float64)


def _resample_pk_batch(
    source_k_bins: np.ndarray,
    pk_batch: np.ndarray,
    target_k_bins: np.ndarray,
) -> np.ndarray:
    source_k = np.asarray(source_k_bins, dtype=np.float64).reshape(-1)
    target_k = np.asarray(target_k_bins, dtype=np.float64).reshape(-1)
    batch = np.asarray(pk_batch, dtype=np.float64)
    if source_k.ndim != 1 or target_k.ndim != 1:
        raise ValueError("source_k_bins and target_k_bins must both be 1D.")
    if batch.ndim != 2 or batch.shape[1] != source_k.shape[0]:
        raise ValueError(
            "pk_batch must be 2D and align with source_k_bins, "
            f"got {batch.shape} vs {source_k.shape}."
        )
    if target_k.shape == source_k.shape and np.allclose(target_k, source_k):
        return batch.astype(np.float64)

    log_source = np.log10(np.maximum(source_k, 1.0e-30))
    log_target = np.log10(np.maximum(target_k, 1.0e-30))
    rows: list[np.ndarray] = []
    for row in batch:
        interpolator = PchipInterpolator(log_source, row, extrapolate=False)
        rows.append(np.asarray(interpolator(log_target), dtype=np.float64))
    return np.vstack(rows).astype(np.float64)


def _source_covers_target_range(
    source_k_bins: np.ndarray,
    target_k_bins: np.ndarray,
    *,
    rtol: float = 1.0e-10,
) -> bool:
    source_k = np.asarray(source_k_bins, dtype=np.float64).reshape(-1)
    target_k = np.asarray(target_k_bins, dtype=np.float64).reshape(-1)
    if source_k.ndim != 1 or target_k.ndim != 1:
        raise ValueError("source_k_bins and target_k_bins must both be 1D.")
    if source_k.size <= 0 or target_k.size <= 0:
        raise ValueError("source_k_bins and target_k_bins must both be non-empty.")
    source_min = float(source_k[0])
    source_max = float(source_k[-1])
    target_min = float(target_k[0])
    target_max = float(target_k[-1])
    low_ok = source_min <= target_min * (1.0 + float(rtol))
    high_ok = source_max >= target_max * (1.0 - float(rtol))
    return bool(low_ok and high_ok)


def _resolve_bank_training_k_bins(
    config: ValidationRuntimeConfig,
    source_k_bins: np.ndarray,
) -> tuple[np.ndarray, str]:
    source_k = np.asarray(source_k_bins, dtype=np.float64).reshape(-1)
    runtime_k = np.asarray(_resolve_runtime_k_bins(config), dtype=np.float64)
    if _source_covers_target_range(source_k, runtime_k):
        return runtime_k, "runtime_training_grid"
    return source_k.astype(np.float64), "source_grid_fallback"


def _resolve_runtime_k_bins(config: ValidationRuntimeConfig) -> np.ndarray:
    data_source = resolve_data_source(config)
    if data_source.name == "quijote":
        output_k_bins = maybe_build_quijote_output_k_bins(data_source.metadata)
        if output_k_bins is not None:
            return output_k_bins
    return np.asarray(build_default_k_bins(config.grids), dtype=np.float64)


def _resolve_representation_anchor_batch(
    config: ValidationRuntimeConfig,
    *,
    p_linear_batch: np.ndarray | None,
) -> np.ndarray | None:
    anchor_mode = str(config.representation.anchor_mode).strip().lower() or "linear"
    if anchor_mode in {"linear", "halofit", "hmcode2020"}:
        if p_linear_batch is None and anchor_mode != "linear":
            raise ValueError(
                f"representation.anchor_mode={anchor_mode!r} requires anchor spectra in "
                "the legacy p_linear_batch storage slot."
            )
        return None if p_linear_batch is None else np.asarray(p_linear_batch, dtype=np.float64)
    raise NotImplementedError(
        "Only representation.anchor_mode in {'linear', 'halofit', 'hmcode2020'} is currently "
        "supported by the active-learning pipeline."
    )


def _build_dataset_from_batches(
    config: ValidationRuntimeConfig,
    raw_thetas: np.ndarray,
    k_bins: np.ndarray,
    pk_batch: np.ndarray,
    *,
    p_linear_batch: np.ndarray | None,
    metadata: dict[str, object] | None = None,
) -> Module1Dataset:
    raw_arr = _ensure_2d(raw_thetas, name="raw_thetas")
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    pk_arr = np.asarray(pk_batch, dtype=np.float64)
    if pk_arr.ndim != 2 or pk_arr.shape[0] != raw_arr.shape[0] or pk_arr.shape[1] != k_arr.shape[0]:
        raise ValueError(
            "pk_batch must align with raw_thetas and k_bins, "
            f"got {pk_arr.shape}, {raw_arr.shape}, {k_arr.shape}."
        )
    linear_arr = None if p_linear_batch is None else np.asarray(p_linear_batch, dtype=np.float64)
    if linear_arr is not None and linear_arr.shape != pk_arr.shape:
        raise ValueError(
            "p_linear_batch must align with pk_batch, "
            f"got {linear_arr.shape} vs {pk_arr.shape}."
        )

    bounds = active_theta_bounds(config)
    if raw_arr.shape[1] != bounds.shape[0]:
        raise ValueError(
            f"raw_thetas must have theta dimension {bounds.shape[0]}, got {raw_arr.shape[1]}."
        )
    unit_thetas = normalize_theta_batch(raw_arr, bounds)
    log_pk_batch = np.log(np.maximum(pk_arr, float(config.gp.power_eps)))
    anchor_batch = _resolve_representation_anchor_batch(
        config,
        p_linear_batch=linear_arr,
    )
    target_batch, target_metadata = build_target_representation(
        pk_arr,
        anchor_batch=anchor_batch,
        power_eps=float(config.gp.power_eps),
        transform_family=str(config.representation.transform_family),
        anchor_mode=str(config.representation.anchor_mode),
    )
    pca_model, raw_scores, pca_layout = fit_representation_pca(
        target_batch,
        k_bins=k_arr,
        total_components=int(config.gp.pca_components),
        pca_scheme=str(config.representation.pca_scheme),
        random_seed=int(config.random_seed),
        global_pca_components=int(config.representation.global_pca_components),
        band_pca_components=tuple(config.representation.band_pca_components),
    )
    scaled_scores, score_mean, score_std = _standardize_scores(raw_scores)
    resolved_metadata = dict(metadata or {})
    resolved_metadata.update(target_metadata)
    resolved_metadata.setdefault("train_size", int(raw_arr.shape[0]))
    resolved_metadata["pca_components"] = int(scaled_scores.shape[1])
    resolved_metadata["pca_scores_standardized"] = True
    resolved_metadata["pca_scheme"] = str(config.representation.pca_scheme)
    resolved_metadata["pca_layout"] = dict(pca_layout)
    return Module1Dataset(
        raw_thetas=raw_arr,
        unit_thetas=unit_thetas,
        k_bins=k_arr,
        pk_batch=pk_arr,
        p_linear_batch=linear_arr,
        log_pk_batch=log_pk_batch,
        target_batch=target_batch,
        pca_scores=scaled_scores,
        pca_score_mean=score_mean,
        pca_score_std=score_std,
        pca_components=np.asarray(pca_model.components_, dtype=np.float64),
        pca_mean=np.asarray(pca_model.mean_, dtype=np.float64),
        explained_variance_ratio=np.asarray(pca_model.explained_variance_ratio_, dtype=np.float64),
        pca_model=pca_model,
        metadata=resolved_metadata,
    )


def rebuild_dataset(
    config: ValidationRuntimeConfig,
    camb_data_provider: CAMBDataProvider,
    raw_thetas: np.ndarray,
    *,
    asset_version: str = "active_learning",
    progress_callback: ProgressCallback | None = None,
) -> Module1Dataset:
    raw_thetas = _ensure_2d(raw_thetas, name="raw_thetas")
    bounds = active_theta_bounds(config)
    if raw_thetas.shape[1] != bounds.shape[0]:
        raise ValueError(
            f"raw_thetas must have theta dimension {bounds.shape[0]}, got {raw_thetas.shape[1]}."
        )
    k_bins = _resolve_runtime_k_bins(config)

    pk_rows: list[np.ndarray] = []
    p_linear_rows: list[np.ndarray] = []
    total = int(raw_thetas.shape[0])
    for idx, theta in enumerate(raw_thetas):
        current = idx + 1
        if progress_callback is not None:
            progress_callback("module1_camb", current, total)
        result = camb_data_provider.run_hifi_anchor(
            theta=theta,
            k_bins=k_bins,
            accuracy_config=CAMBAccuracyConfig(mode="hifi"),
            asset_version=asset_version,
        )
        p_linear_rows.append(np.asarray(result["P_linear"], dtype=np.float64))
        pk_rows.append(np.asarray(result["P_nonlin_hifi"], dtype=np.float64))

    return _build_dataset_from_batches(
        config,
        raw_thetas,
        k_bins,
        np.vstack(pk_rows).astype(np.float64),
        p_linear_batch=np.vstack(p_linear_rows).astype(np.float64),
        metadata={
            "asset_version": str(asset_version),
            "train_size": int(raw_thetas.shape[0]),
        },
    )


def build_dataset_from_spectrum_bank(
    config: ValidationRuntimeConfig,
    raw_thetas: np.ndarray,
    source_k_bins: np.ndarray,
    pk_batch: np.ndarray,
    p_linear_batch: np.ndarray | None = None,
) -> Module1Dataset:
    raw_thetas = _ensure_2d(raw_thetas, name="raw_thetas")
    source_k_bins = np.asarray(source_k_bins, dtype=np.float64).reshape(-1)
    pk_batch = np.asarray(pk_batch, dtype=np.float64)
    target_k_bins, k_grid_strategy = _resolve_bank_training_k_bins(config, source_k_bins)
    if pk_batch.ndim != 2 or pk_batch.shape[0] != raw_thetas.shape[0]:
        raise ValueError(
            "pk_batch must be 2D and align with raw_thetas, "
            f"got {pk_batch.shape} vs {raw_thetas.shape}."
        )
    target_pk_batch = _resample_pk_batch(source_k_bins, pk_batch, target_k_bins)
    target_linear_batch = None
    if p_linear_batch is not None:
        target_linear_batch = _resample_pk_batch(source_k_bins, np.asarray(p_linear_batch, dtype=np.float64), target_k_bins)
    return _build_dataset_from_batches(
        config,
        raw_thetas,
        target_k_bins,
        target_pk_batch,
        p_linear_batch=target_linear_batch,
        metadata={
            "source_k_bins": int(source_k_bins.shape[0]),
            "target_k_bins": int(target_k_bins.shape[0]),
            "train_size": int(raw_thetas.shape[0]),
            "k_grid_strategy": str(k_grid_strategy),
            "source_k_min": float(source_k_bins[0]),
            "source_k_max": float(source_k_bins[-1]),
            "target_k_min": float(target_k_bins[0]),
            "target_k_max": float(target_k_bins[-1]),
        },
    )


def build_initial_dataset(
    config: ValidationRuntimeConfig,
    camb_data_provider: CAMBDataProvider,
    *,
    asset_version: str = "active_learning",
    progress_callback: ProgressCallback | None = None,
) -> Module1Dataset:
    raw_thetas = generate_sobol_thetas(
        active_theta_bounds(config),
        config.sampling.initial_sobol_points,
        config.sampling.initial_seed,
    )
    return rebuild_dataset(
        config,
        camb_data_provider,
        raw_thetas,
        asset_version=asset_version,
        progress_callback=progress_callback,
    )


def extend_dataset(
    config: ValidationRuntimeConfig,
    camb_data_provider: CAMBDataProvider,
    dataset: Module1Dataset,
    new_raw_thetas: np.ndarray,
    *,
    asset_version: str = "active_learning",
    progress_callback: ProgressCallback | None = None,
) -> Module1Dataset:
    new_raw_thetas = _ensure_2d(new_raw_thetas, name="new_raw_thetas")
    bounds = active_theta_bounds(config)
    if new_raw_thetas.shape[1] != bounds.shape[0]:
        raise ValueError(
            f"new_raw_thetas must have theta dimension {bounds.shape[0]}, got {new_raw_thetas.shape[1]}."
        )

    k_bins = np.asarray(dataset.k_bins, dtype=np.float64).reshape(-1)
    new_pk_rows: list[np.ndarray] = []
    new_linear_rows: list[np.ndarray] = []
    total = int(new_raw_thetas.shape[0])
    for idx, theta in enumerate(new_raw_thetas):
        current = idx + 1
        if progress_callback is not None:
            progress_callback("module1_camb", current, total)
        result = camb_data_provider.run_hifi_anchor(
            theta=theta,
            k_bins=k_bins,
            accuracy_config=CAMBAccuracyConfig(mode="hifi"),
            asset_version=asset_version,
        )
        new_linear_rows.append(np.asarray(result["P_linear"], dtype=np.float64))
        new_pk_rows.append(np.asarray(result["P_nonlin_hifi"], dtype=np.float64))

    new_pk_batch = np.vstack(new_pk_rows).astype(np.float64)
    new_linear_batch = np.vstack(new_linear_rows).astype(np.float64)
    merged_raw = np.vstack([dataset.raw_thetas, new_raw_thetas]).astype(np.float64)
    merged_pk = np.vstack([dataset.pk_batch, new_pk_batch]).astype(np.float64)
    existing_linear = dataset.p_linear_batch
    merged_linear = None
    if existing_linear is not None:
        merged_linear = np.vstack([existing_linear, new_linear_batch]).astype(np.float64)

    metadata = dict(dataset.metadata)
    metadata.update(
        {
            "asset_version": str(asset_version),
            "train_size": int(merged_raw.shape[0]),
            "added_points": int(new_raw_thetas.shape[0]),
        }
    )
    return _build_dataset_from_batches(
        config,
        merged_raw,
        k_bins,
        merged_pk,
        p_linear_batch=merged_linear,
        metadata=metadata,
    )
