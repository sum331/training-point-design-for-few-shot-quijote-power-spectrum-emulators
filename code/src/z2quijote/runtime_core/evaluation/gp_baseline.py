"""Dedicated fixed-budget Sobol+PCA+GP baselines isolated from active-learning facades."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from z2quijote.runtime_core.cache_manager import SpectrumBank, get_or_create_comparison_training_bank
from z2quijote.runtime_core.camb_data_provider import CAMBDataProvider
from z2quijote.runtime_core.config import (
    ValidationRuntimeConfig,
    build_default_k_bins,
    normalize_theta_batch,
)
from z2quijote.runtime_core.data_source import active_theta_bounds, resolve_data_source
from z2quijote.runtime_core.quijote_k_grid import maybe_build_quijote_output_k_bins
from z2quijote.runtime_core.representation import (
    build_target_representation,
    fit_representation_pca,
    parse_target_transform,
    reconstruct_power_from_target,
    resolve_target_transform_from_metadata,
)
from z2quijote.runtime_core.run_artifacts import run_process_path, run_results_subdir
from z2quijote.runtime_core.sampling import ensure_2d_theta_batch

ProgressCallback = Callable[[str, int, int], None]

STANDARD_SOBOL_GP_TRAIN_POINTS = 128
STANDARD_SOBOL_GP_OUTPUT_SUBDIR = "standard_gp_baseline_128"


@dataclass(slots=True)
class ResolvedGPBaselineHyperparameters:
    pca_components: int
    alpha: float
    normalize_y: bool
    n_restarts_optimizer: int
    constant_value: float
    constant_value_bounds_low: float
    constant_value_bounds_high: float
    length_scale_initial: float
    length_scale_bounds_low: float
    length_scale_bounds_high: float
    power_eps: float
    pca_random_seed: int
    gp_random_seed_base: int
    training_sampling_seed: int
    sources: dict[str, str] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        return {
            "pca_components": int(self.pca_components),
            "alpha": float(self.alpha),
            "normalize_y": bool(self.normalize_y),
            "n_restarts_optimizer": int(self.n_restarts_optimizer),
            "constant_value": float(self.constant_value),
            "constant_value_bounds_low": float(self.constant_value_bounds_low),
            "constant_value_bounds_high": float(self.constant_value_bounds_high),
            "length_scale_initial": float(self.length_scale_initial),
            "length_scale_bounds_low": float(self.length_scale_bounds_low),
            "length_scale_bounds_high": float(self.length_scale_bounds_high),
            "power_eps": float(self.power_eps),
            "pca_random_seed": int(self.pca_random_seed),
            "gp_random_seed_base": int(self.gp_random_seed_base),
            "training_sampling_seed": int(self.training_sampling_seed),
            "sources": dict(self.sources),
        }


@dataclass(slots=True)
class BaselineDataset:
    raw_thetas: np.ndarray
    unit_thetas: np.ndarray
    k_bins: np.ndarray
    pk_batch: np.ndarray
    p_linear_batch: np.ndarray | None
    log_pk_batch: np.ndarray
    target_batch: np.ndarray
    pca_scores: np.ndarray
    pca_score_mean: np.ndarray
    pca_score_std: np.ndarray
    pca_components: np.ndarray
    pca_mean: np.ndarray
    explained_variance_ratio: np.ndarray
    pca_model: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaselineEmulatorState:
    dataset: BaselineDataset
    gp_models: list[GaussianProcessRegressor]
    theta_bounds: np.ndarray
    kernel_descriptions: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GPBaselineHifiCounts:
    cold_start_hifi_count: int
    iteration_hifi_count: int
    total_hifi_unique_points: int


@dataclass(slots=True)
class GPBaselineTrainingArtifacts:
    train_thetas: np.ndarray
    train_k_bins: np.ndarray
    train_nonlin_pk: np.ndarray
    train_linear_pk: np.ndarray | None
    emulator: BaselineEmulatorState
    resolved_hyperparameters: ResolvedGPBaselineHyperparameters

    def predict_on_k(
        self,
        theta_batch: np.ndarray,
        k_target: np.ndarray,
        *,
        p_linear_batch: np.ndarray | None = None,
    ) -> np.ndarray:
        theta_batch = np.asarray(theta_batch, dtype=np.float64)
        k_target = np.asarray(k_target, dtype=np.float64)
        if theta_batch.ndim != 2:
            raise ValueError(f"theta_batch must be 2D, got {theta_batch.shape}.")
        if k_target.ndim != 1:
            raise ValueError(f"k_target must be 1D, got {k_target.shape}.")
        prediction = predict_gp_baseline_spectra(
            self.emulator,
            theta_batch,
            input_space="raw",
            k_target=k_target,
            p_linear_batch=p_linear_batch,
        )
        return np.asarray(prediction.pk_mean, dtype=np.float64)


@dataclass(slots=True)
class BaselineSpectrumPrediction:
    raw_thetas: np.ndarray
    unit_thetas: np.ndarray
    k_bins: np.ndarray
    pc_mean: np.ndarray
    pc_std: np.ndarray
    target_mean: np.ndarray
    log_pk_mean: np.ndarray
    pk_mean: np.ndarray
    p_linear_batch: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Process1GPBaselineResult:
    output_dir: Path
    test_set_results_path: Path
    summary_path: Path
    run_metadata_path: Path
    hifi_bank_path: Path
    data_source: str
    train_points: int


@dataclass(slots=True)
class StandardSobolGPBaselineResult:
    output_dir: Path
    test_set_results_path: Path
    summary_path: Path
    run_metadata_path: Path
    train_points: int


def _ensure_2d(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {arr.shape}.")
    return arr


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


def _build_target_batch(
    pk_batch: np.ndarray,
    anchor_batch: np.ndarray | None,
    *,
    power_eps: float,
    transform_family: str,
    anchor_mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    return build_target_representation(
        pk_batch,
        anchor_batch=anchor_batch,
        power_eps=power_eps,
        transform_family=transform_family,
        anchor_mode=anchor_mode,
    )


def _standardize_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_arr = np.asarray(scores, dtype=np.float64)
    if score_arr.ndim != 2:
        raise ValueError(f"scores must be 2D, got {score_arr.shape}.")
    mean = np.mean(score_arr, axis=0, dtype=np.float64)
    std = np.std(score_arr, axis=0, dtype=np.float64)
    std = np.maximum(std, 1.0e-12)
    scaled = (score_arr - mean[None, :]) / std[None, :]
    return scaled.astype(np.float64), mean.astype(np.float64), std.astype(np.float64)


def resolve_gp_baseline_hyperparameters(
    config: ValidationRuntimeConfig,
) -> ResolvedGPBaselineHyperparameters:
    baseline_cfg = config.gp_baseline
    sources = {
        "pca_components": "gp_baseline.pca_components",
        "alpha": "gp_baseline.gp_alpha",
        "n_restarts_optimizer": "gp_baseline.gp_n_restarts_optimizer",
        "normalize_y": "gp_baseline.normalize_y",
        "constant_value": "gp_baseline.constant_value",
        "constant_value_bounds_low": "gp_baseline.constant_value_bounds_low",
        "constant_value_bounds_high": "gp_baseline.constant_value_bounds_high",
        "length_scale_initial": "gp_baseline.length_scale_initial",
        "length_scale_bounds_low": "gp_baseline.length_scale_bounds_low",
        "length_scale_bounds_high": "gp_baseline.length_scale_bounds_high",
        "power_eps": "gp_baseline.power_eps",
        "training_sampling_seed": "gp_baseline.random_seed",
        "pca_random_seed": "gp_baseline.random_seed",
        "gp_random_seed_base": "gp_baseline.random_seed",
    }
    return ResolvedGPBaselineHyperparameters(
        pca_components=int(baseline_cfg.pca_components),
        alpha=float(baseline_cfg.gp_alpha),
        normalize_y=bool(baseline_cfg.normalize_y),
        n_restarts_optimizer=int(baseline_cfg.gp_n_restarts_optimizer),
        constant_value=float(baseline_cfg.constant_value),
        constant_value_bounds_low=float(baseline_cfg.constant_value_bounds_low),
        constant_value_bounds_high=float(baseline_cfg.constant_value_bounds_high),
        length_scale_initial=float(baseline_cfg.length_scale_initial),
        length_scale_bounds_low=float(baseline_cfg.length_scale_bounds_low),
        length_scale_bounds_high=float(baseline_cfg.length_scale_bounds_high),
        power_eps=float(baseline_cfg.power_eps),
        pca_random_seed=int(baseline_cfg.random_seed),
        gp_random_seed_base=int(baseline_cfg.random_seed),
        training_sampling_seed=int(baseline_cfg.random_seed),
        sources=sources,
    )


def _fit_baseline_pca(
    target_batch: np.ndarray,
    *,
    k_bins: np.ndarray,
    config: ValidationRuntimeConfig,
    resolved_hyperparameters: ResolvedGPBaselineHyperparameters,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    return fit_representation_pca(
        target_batch,
        k_bins=k_bins,
        total_components=int(resolved_hyperparameters.pca_components),
        pca_scheme=str(config.representation.pca_scheme),
        random_seed=int(resolved_hyperparameters.pca_random_seed),
        global_pca_components=int(config.representation.global_pca_components),
        band_pca_components=tuple(config.representation.band_pca_components),
    )


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
        "supported by the fixed-budget baseline."
    )


def _build_baseline_dataset(
    config: ValidationRuntimeConfig,
    raw_thetas: np.ndarray,
    source_k_bins: np.ndarray,
    pk_batch: np.ndarray,
    *,
    resolved_hyperparameters: ResolvedGPBaselineHyperparameters,
    p_linear_batch: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> BaselineDataset:
    raw_arr = _ensure_2d(raw_thetas, name="raw_thetas")
    source_k_arr = np.asarray(source_k_bins, dtype=np.float64).reshape(-1)
    pk_arr = np.asarray(pk_batch, dtype=np.float64)
    if pk_arr.ndim != 2 or pk_arr.shape[0] != raw_arr.shape[0] or pk_arr.shape[1] != source_k_arr.shape[0]:
        raise ValueError(
            "pk_batch must align with raw_thetas and k_bins, "
            f"got {pk_arr.shape}, {raw_arr.shape}, {source_k_arr.shape}."
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

    target_k_bins, k_grid_strategy = _resolve_bank_training_k_bins(config, source_k_arr)
    target_pk_batch = _resample_pk_batch(source_k_arr, pk_arr, target_k_bins)
    target_linear_batch = None
    if linear_arr is not None:
        target_linear_batch = _resample_pk_batch(source_k_arr, linear_arr, target_k_bins)

    unit_thetas = normalize_theta_batch(raw_arr, bounds)
    log_pk_batch = np.log(np.maximum(target_pk_batch, float(resolved_hyperparameters.power_eps)))
    target_anchor_batch = _resolve_representation_anchor_batch(
        config,
        p_linear_batch=target_linear_batch,
    )
    target_batch, target_metadata = _build_target_batch(
        target_pk_batch,
        target_anchor_batch,
        power_eps=float(resolved_hyperparameters.power_eps),
        transform_family=str(config.representation.transform_family),
        anchor_mode=str(config.representation.anchor_mode),
    )
    pca_model, raw_scores, pca_layout = _fit_baseline_pca(
        target_batch,
        k_bins=target_k_bins,
        config=config,
        resolved_hyperparameters=resolved_hyperparameters,
    )
    scaled_scores, score_mean, score_std = _standardize_scores(raw_scores)

    resolved_metadata = dict(metadata or {})
    resolved_metadata.update(target_metadata)
    resolved_metadata.setdefault("train_size", int(raw_arr.shape[0]))
    resolved_metadata.update(
        {
            "source_k_bins": int(source_k_arr.shape[0]),
            "target_k_bins": int(target_k_bins.shape[0]),
            "k_grid_strategy": str(k_grid_strategy),
            "source_k_min": float(source_k_arr[0]),
            "source_k_max": float(source_k_arr[-1]),
            "target_k_min": float(target_k_bins[0]),
            "target_k_max": float(target_k_bins[-1]),
            "pca_components_requested": int(resolved_hyperparameters.pca_components),
            "pca_components": int(scaled_scores.shape[1]),
            "pca_scores_standardized": True,
            "baseline_emulator_kind": "dedicated_fixed_budget_pca_gp",
            "pca_scheme": str(config.representation.pca_scheme),
            "pca_layout": dict(pca_layout),
        }
    )

    return BaselineDataset(
        raw_thetas=raw_arr,
        unit_thetas=unit_thetas,
        k_bins=target_k_bins,
        pk_batch=target_pk_batch,
        p_linear_batch=target_linear_batch,
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


def _build_baseline_kernel(
    resolved_hyperparameters: ResolvedGPBaselineHyperparameters,
    theta_dim: int,
):
    return ConstantKernel(
        constant_value=float(resolved_hyperparameters.constant_value),
        constant_value_bounds=(
            float(resolved_hyperparameters.constant_value_bounds_low),
            float(resolved_hyperparameters.constant_value_bounds_high),
        ),
    ) * RBF(
        length_scale=np.full(
            (theta_dim,),
            float(resolved_hyperparameters.length_scale_initial),
            dtype=np.float64,
        ),
        length_scale_bounds=(
            float(resolved_hyperparameters.length_scale_bounds_low),
            float(resolved_hyperparameters.length_scale_bounds_high),
        ),
    )


def _fit_baseline_emulator(
    dataset: BaselineDataset,
    *,
    resolved_hyperparameters: ResolvedGPBaselineHyperparameters,
    theta_bounds: np.ndarray,
    progress_callback: ProgressCallback | None = None,
) -> BaselineEmulatorState:
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
            progress_callback("gp_baseline_fit", pc_idx + 1, pca_scores.shape[1])
        gp = GaussianProcessRegressor(
            kernel=_build_baseline_kernel(resolved_hyperparameters, theta_dim),
            alpha=float(resolved_hyperparameters.alpha),
            normalize_y=bool(resolved_hyperparameters.normalize_y),
            random_state=int(resolved_hyperparameters.gp_random_seed_base + pc_idx),
            n_restarts_optimizer=int(resolved_hyperparameters.n_restarts_optimizer),
        )
        gp.fit(unit_thetas, pca_scores[:, pc_idx])
        gp_models.append(gp)
        kernel_descriptions.append(str(gp.kernel_))

    return BaselineEmulatorState(
        dataset=dataset,
        gp_models=gp_models,
        theta_bounds=np.asarray(theta_bounds, dtype=np.float64),
        kernel_descriptions=kernel_descriptions,
        metadata={
            "train_size": int(unit_thetas.shape[0]),
            "pca_components": int(pca_scores.shape[1]),
            "kernel_family": "ConstantKernel * RBF(ARD)",
            "resolved_hyperparameters": resolved_hyperparameters.to_metadata(),
        },
    )


def _interp_logk_batch(
    batch: np.ndarray,
    source_k: np.ndarray,
    target_k: np.ndarray,
) -> np.ndarray:
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
    emulator: BaselineEmulatorState,
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
    emulator: BaselineEmulatorState,
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
        "which is not yet persisted by the fixed-budget baseline dataset."
    )


def predict_gp_baseline_spectra(
    emulator: BaselineEmulatorState,
    theta_batch: np.ndarray,
    *,
    input_space: str = "raw",
    k_target: np.ndarray | None = None,
    p_linear_batch: np.ndarray | None = None,
) -> BaselineSpectrumPrediction:
    raw_thetas, unit_thetas = ensure_2d_theta_batch(
        np.asarray(theta_batch, dtype=np.float64),
        emulator.theta_bounds,
        input_space=input_space,
    )
    if not emulator.gp_models:
        raise ValueError("Baseline emulator has no fitted GP models.")

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
        power_eps=float(
            emulator.metadata.get("resolved_hyperparameters", {}).get("power_eps", 1.0e-12)
        ),
    )

    return BaselineSpectrumPrediction(
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
            "baseline_emulator_kind": "dedicated_fixed_budget_pca_gp",
            "pca_scheme": str(emulator.dataset.metadata.get("pca_scheme", "global_pca")),
        },
    )


def count_unique_hifi_points(
    *,
    cold_start_thetas: np.ndarray | None,
    historical_anchors: list[np.ndarray],
) -> GPBaselineHifiCounts:
    cold_arr = (
        np.asarray(cold_start_thetas, dtype=np.float64)
        if cold_start_thetas is not None
        else np.empty((0, 8), dtype=np.float64)
    )
    hist_arr = (
        np.vstack([np.asarray(item, dtype=np.float64) for item in historical_anchors]).astype(np.float64)
        if historical_anchors
        else np.empty((0, cold_arr.shape[1] if cold_arr.size > 0 else 8), dtype=np.float64)
    )

    def _unique_rows(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.reshape(0, arr.shape[1] if arr.ndim == 2 else 0)
        rounded = np.round(arr, decimals=12)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        return arr[np.sort(idx)]

    cold_unique = _unique_rows(cold_arr)
    hist_unique = _unique_rows(hist_arr)
    total_unique = _unique_rows(
        np.vstack([cold_arr, hist_arr]) if cold_arr.size > 0 or hist_arr.size > 0 else hist_arr
    )
    return GPBaselineHifiCounts(
        cold_start_hifi_count=int(cold_unique.shape[0]),
        iteration_hifi_count=int(hist_unique.shape[0]),
        total_hifi_unique_points=int(total_unique.shape[0]),
    )


def _artifacts_from_dataset(
    config: ValidationRuntimeConfig,
    dataset: BaselineDataset,
    *,
    resolved_hyperparameters: ResolvedGPBaselineHyperparameters,
    progress_callback: ProgressCallback | None = None,
) -> GPBaselineTrainingArtifacts:
    emulator = _fit_baseline_emulator(
        dataset,
        resolved_hyperparameters=resolved_hyperparameters,
        theta_bounds=active_theta_bounds(config),
        progress_callback=progress_callback,
    )
    return GPBaselineTrainingArtifacts(
        train_thetas=np.asarray(dataset.raw_thetas, dtype=np.float64),
        train_k_bins=np.asarray(dataset.k_bins, dtype=np.float64),
        train_nonlin_pk=np.asarray(dataset.pk_batch, dtype=np.float64),
        train_linear_pk=None if dataset.p_linear_batch is None else np.asarray(dataset.p_linear_batch, dtype=np.float64),
        emulator=emulator,
        resolved_hyperparameters=resolved_hyperparameters,
    )


def fit_gp_baseline(
    *,
    config: ValidationRuntimeConfig,
    camb_data_provider: CAMBDataProvider,
    asset_version: str,
    total_hifi_unique_points: int,
    progress_callback: ProgressCallback | None = None,
    force_rebuild_training_cache: bool = False,
) -> GPBaselineTrainingArtifacts:
    active_training_bank = get_or_create_comparison_training_bank(
        config,
        camb_data_provider,
        train_points=int(total_hifi_unique_points),
        progress_callback=progress_callback,
        force_rebuild=force_rebuild_training_cache,
    )
    artifacts = fit_gp_baseline_from_spectrum_bank(
        config=config,
        spectrum_bank=active_training_bank,
        progress_callback=progress_callback,
    )
    artifacts.emulator.dataset.metadata.update(
        {
            "source": "comparison_training_cache",
            "cache_name": str(active_training_bank.name),
            "cache_path": str(active_training_bank.npz_path),
            "cache_status": str(active_training_bank.metadata.get("cache_status", "unknown")),
            "requested_asset_version": str(asset_version),
        }
    )
    return artifacts


def fit_gp_baseline_from_spectrum_bank(
    *,
    config: ValidationRuntimeConfig,
    spectrum_bank: SpectrumBank,
    progress_callback: ProgressCallback | None = None,
) -> GPBaselineTrainingArtifacts:
    return fit_gp_baseline_from_hifi_bank(
        config=config,
        train_thetas=spectrum_bank.raw_thetas,
        train_k_bins=spectrum_bank.k_bins,
        train_nonlin_pk=spectrum_bank.p_nonlin_batch,
        train_linear_pk=spectrum_bank.p_linear_batch,
        progress_callback=progress_callback,
        metadata={
            "source": "spectrum_bank",
            "cache_name": str(spectrum_bank.name),
            "cache_path": str(spectrum_bank.npz_path),
            "cache_status": str(spectrum_bank.metadata.get("cache_status", "unknown")),
        },
    )


def fit_gp_baseline_from_hifi_bank(
    *,
    config: ValidationRuntimeConfig,
    train_thetas: np.ndarray,
    train_k_bins: np.ndarray,
    train_nonlin_pk: np.ndarray,
    train_linear_pk: np.ndarray | None = None,
    progress_callback: ProgressCallback | None = None,
    metadata: dict[str, object] | None = None,
) -> GPBaselineTrainingArtifacts:
    train_thetas = np.asarray(train_thetas, dtype=np.float64)
    train_k_bins = np.asarray(train_k_bins, dtype=np.float64)
    train_nonlin_pk = np.asarray(train_nonlin_pk, dtype=np.float64)
    if train_thetas.ndim != 2 or train_nonlin_pk.ndim != 2:
        raise ValueError(
            f"train_thetas must be 2D and train_nonlin_pk 2D, got {train_thetas.shape} and {train_nonlin_pk.shape}."
        )
    if train_nonlin_pk.shape[0] != train_thetas.shape[0]:
        raise ValueError("train_nonlin_pk rows must match train_thetas.")
    if train_nonlin_pk.shape[1] != train_k_bins.shape[0]:
        raise ValueError("train_nonlin_pk columns must match train_k_bins.")
    if train_linear_pk is not None:
        train_linear_pk = np.asarray(train_linear_pk, dtype=np.float64)
        if train_linear_pk.shape != train_nonlin_pk.shape:
            raise ValueError(
                "train_linear_pk must align with train_nonlin_pk, "
                f"got {train_linear_pk.shape} vs {train_nonlin_pk.shape}."
            )

    resolved_hyperparameters = resolve_gp_baseline_hyperparameters(config)
    dataset = _build_baseline_dataset(
        config,
        train_thetas,
        train_k_bins,
        train_nonlin_pk,
        p_linear_batch=train_linear_pk,
        resolved_hyperparameters=resolved_hyperparameters,
        metadata=metadata,
    )
    dataset.metadata.setdefault("source", "hifi_bank")
    return _artifacts_from_dataset(
        config,
        dataset,
        resolved_hyperparameters=resolved_hyperparameters,
        progress_callback=progress_callback,
    )


def _write_gp_baseline_summary(
    summary_path: Path,
    *,
    counts: GPBaselineHifiCounts,
    artifacts: GPBaselineTrainingArtifacts,
    spectrum_type: str,
    mode: str,
    data_source: str,
) -> Path:
    dataset = artifacts.emulator.dataset
    payload = {
        "mode": str(mode),
        "data_source": str(data_source),
        "spectrum_type": str(spectrum_type),
        "counts": {
            "cold_start_hifi_count": int(counts.cold_start_hifi_count),
            "iteration_hifi_count": int(counts.iteration_hifi_count),
            "total_hifi_unique_points": int(counts.total_hifi_unique_points),
        },
        "train_size": int(artifacts.train_thetas.shape[0]),
        "k_bin_size": int(artifacts.train_k_bins.shape[0]),
        "has_linear_pk": bool(artifacts.train_linear_pk is not None),
        "pca_components_requested": int(artifacts.resolved_hyperparameters.pca_components),
        "pca_components": int(len(artifacts.emulator.gp_models)),
        "target_transform": str(dataset.metadata.get("target_transform", "unknown")),
        "k_grid_strategy": str(dataset.metadata.get("k_grid_strategy", "unknown")),
        "source_k_bins": int(dataset.metadata.get("source_k_bins", 0)),
        "target_k_bins": int(dataset.metadata.get("target_k_bins", 0)),
        "resolved_hyperparameters": artifacts.resolved_hyperparameters.to_metadata(),
        "baseline_emulator_kind": "dedicated_fixed_budget_pca_gp",
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def evaluate_gp_baseline_with_reused_truth(
    *,
    artifacts: GPBaselineTrainingArtifacts,
    test_thetas: np.ndarray,
    k_bins: np.ndarray,
    p_true: np.ndarray,
    p_linear: np.ndarray | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del progress_callback
    test_thetas = np.asarray(test_thetas, dtype=np.float64)
    k_bins = np.asarray(k_bins, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    linear_batch = None if p_linear is None else np.asarray(p_linear, dtype=np.float64)
    p_pred = artifacts.predict_on_k(test_thetas, k_bins, p_linear_batch=linear_batch)
    return test_thetas, k_bins, p_true, p_pred


def run_standard_sobol_gp_baseline(
    *,
    run_dir: Path,
    config: ValidationRuntimeConfig,
    test_set_results_path: Path,
    spectrum_type: str,
    camb_data_provider: CAMBDataProvider,
    asset_version: str = "reconstructed",
    output_subdir: str = STANDARD_SOBOL_GP_OUTPUT_SUBDIR,
    n_train_points: int = STANDARD_SOBOL_GP_TRAIN_POINTS,
    progress_callback: ProgressCallback | None = None,
    force_rebuild_training_cache: bool = False,
) -> StandardSobolGPBaselineResult:
    from z2quijote.runtime_core.evaluation.active_learning_validation import evaluate_predictions_against_truth

    with Path(test_set_results_path).open("r", encoding="utf-8") as handle:
        main_results = json.load(handle)
    test_thetas = np.asarray(main_results["test_thetas"], dtype=np.float64)
    k_bins = np.asarray(main_results["k_bins"], dtype=np.float64)
    p_true = np.asarray(main_results["p_true_batch"], dtype=np.float64)
    p_linear = (
        np.asarray(main_results["p_linear_batch"], dtype=np.float64)
        if "p_linear_batch" in main_results
        else None
    )

    artifacts = fit_gp_baseline(
        config=config,
        camb_data_provider=camb_data_provider,
        asset_version=asset_version,
        total_hifi_unique_points=int(n_train_points),
        progress_callback=progress_callback,
        force_rebuild_training_cache=force_rebuild_training_cache,
    )
    _, _, _, p_pred = evaluate_gp_baseline_with_reused_truth(
        artifacts=artifacts,
        test_thetas=test_thetas,
        k_bins=k_bins,
        p_true=p_true,
        p_linear=p_linear,
        progress_callback=progress_callback,
    )

    output_dir = run_results_subdir(run_dir, str(output_subdir).strip(), create=True)
    validation = evaluate_predictions_against_truth(
        config,
        output_dir,
        test_thetas=test_thetas,
        k_bins=k_bins,
        p_true_batch=p_true,
        p_pred_batch=p_pred,
        p_linear_batch=p_linear,
        metadata={
            "mode": "standard_sobol_gp_baseline",
            "train_points": int(n_train_points),
            "train_cache_path": str(artifacts.emulator.dataset.metadata.get("cache_path", "")),
            "train_cache_status": str(artifacts.emulator.dataset.metadata.get("cache_status", "unknown")),
            "target_transform": str(artifacts.emulator.dataset.metadata.get("target_transform", "unknown")),
            "validation_sampling_method": str(
                main_results.get("metadata", {}).get("validation_sampling_method", "unknown")
            ),
            "resolved_hyperparameters": artifacts.resolved_hyperparameters.to_metadata(),
            "baseline_emulator_kind": "dedicated_fixed_budget_pca_gp",
        },
    )
    counts = GPBaselineHifiCounts(
        cold_start_hifi_count=0,
        iteration_hifi_count=int(n_train_points),
        total_hifi_unique_points=int(n_train_points),
    )
    summary_path = _write_gp_baseline_summary(
        output_dir / "gp_baseline_summary.json",
        counts=counts,
        artifacts=artifacts,
        spectrum_type=spectrum_type,
        mode="standard_sobol_gp_baseline",
        data_source="fixed_sobol_hifi",
    )
    run_metadata_path = output_dir / "run_metadata.json"
    run_metadata_path.write_text(
        json.dumps(
            {
                "mode": "standard_sobol_gp_baseline",
                "data_source": "fixed_sobol_hifi",
                "spectrum_type": str(spectrum_type),
                "train_points": int(n_train_points),
                "train_cache_path": str(artifacts.emulator.dataset.metadata.get("cache_path", "")),
                "train_cache_status": str(artifacts.emulator.dataset.metadata.get("cache_status", "unknown")),
                "asset_version": str(asset_version),
                "target_transform": str(artifacts.emulator.dataset.metadata.get("target_transform", "unknown")),
                "resolved_hyperparameters": artifacts.resolved_hyperparameters.to_metadata(),
                "baseline_emulator_kind": "dedicated_fixed_budget_pca_gp",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return StandardSobolGPBaselineResult(
        output_dir=output_dir,
        test_set_results_path=validation.test_set_results_path.resolve(),
        summary_path=summary_path,
        run_metadata_path=run_metadata_path,
        train_points=int(n_train_points),
    )


def run_process1_gp_baseline(
    *,
    run_dir: Path,
    config: ValidationRuntimeConfig,
    test_set_results_path: Path,
    spectrum_type: str,
    output_subdir: str = "process1_gp_baseline",
    camb_data_provider: CAMBDataProvider | None = None,
    asset_version: str = "reconstructed",
    progress_callback: ProgressCallback | None = None,
    existing_bank_source: str = "existing_hifi_bank",
    force_rebuild: bool = False,
) -> Process1GPBaselineResult:
    from z2quijote.runtime_core.evaluation.active_learning_validation import evaluate_predictions_against_truth

    del camb_data_provider, progress_callback, force_rebuild, asset_version
    run_dir = Path(run_dir).resolve()
    test_set_results_path = Path(test_set_results_path).resolve()
    if not test_set_results_path.exists():
        raise FileNotFoundError(f"test_set_results not found: {test_set_results_path}")

    hifi_bank_path = run_process_path(run_dir, "hifi_bank.npz", create=True)
    if not hifi_bank_path.exists():
        raise FileNotFoundError(
            "process1 baseline expects an existing hifi_bank.npz in the run process directory."
        )

    with np.load(hifi_bank_path, allow_pickle=False) as npz:
        train_thetas = np.asarray(npz["train_thetas"], dtype=np.float64)
        train_k_bins = np.asarray(npz["train_k_bins"], dtype=np.float64)
        train_nonlin_pk = np.asarray(npz["train_nonlin_pk"], dtype=np.float64)
        train_linear_pk = (
            np.asarray(npz["train_linear_pk"], dtype=np.float64)
            if "train_linear_pk" in npz.files
            else None
        )

    artifacts = fit_gp_baseline_from_hifi_bank(
        config=config,
        train_thetas=train_thetas,
        train_k_bins=train_k_bins,
        train_nonlin_pk=train_nonlin_pk,
        train_linear_pk=train_linear_pk,
        metadata={
            "source": str(existing_bank_source),
            "hifi_bank_path": str(hifi_bank_path),
        },
    )

    with test_set_results_path.open("r", encoding="utf-8") as handle:
        main_results = json.load(handle)
    test_thetas = np.asarray(main_results["test_thetas"], dtype=np.float64)
    k_bins = np.asarray(main_results["k_bins"], dtype=np.float64)
    p_true = np.asarray(main_results["p_true_batch"], dtype=np.float64)
    p_linear = (
        np.asarray(main_results["p_linear_batch"], dtype=np.float64)
        if "p_linear_batch" in main_results
        else None
    )
    _, _, _, p_pred = evaluate_gp_baseline_with_reused_truth(
        artifacts=artifacts,
        test_thetas=test_thetas,
        k_bins=k_bins,
        p_true=p_true,
        p_linear=p_linear,
    )

    output_dir = run_results_subdir(run_dir, str(output_subdir).strip(), create=True)
    validation = evaluate_predictions_against_truth(
        config,
        output_dir,
        test_thetas=test_thetas,
        k_bins=k_bins,
        p_true_batch=p_true,
        p_pred_batch=p_pred,
        p_linear_batch=p_linear,
        metadata={
            "mode": "process1_gp_baseline",
            "data_source": str(existing_bank_source),
            "target_transform": str(artifacts.emulator.dataset.metadata.get("target_transform", "unknown")),
            "resolved_hyperparameters": artifacts.resolved_hyperparameters.to_metadata(),
            "baseline_emulator_kind": "dedicated_fixed_budget_pca_gp",
        },
    )
    counts = GPBaselineHifiCounts(
        cold_start_hifi_count=0,
        iteration_hifi_count=int(train_thetas.shape[0]),
        total_hifi_unique_points=int(train_thetas.shape[0]),
    )
    summary_path = _write_gp_baseline_summary(
        output_dir / "gp_baseline_summary.json",
        counts=counts,
        artifacts=artifacts,
        spectrum_type=spectrum_type,
        mode="process1_gp_baseline",
        data_source=str(existing_bank_source),
    )
    run_metadata_path = output_dir / "run_metadata.json"
    run_metadata_path.write_text(
        json.dumps(
            {
                "mode": "process1_gp_baseline",
                "data_source": str(existing_bank_source),
                "spectrum_type": str(spectrum_type),
                "train_points": int(train_thetas.shape[0]),
                "target_transform": str(artifacts.emulator.dataset.metadata.get("target_transform", "unknown")),
                "resolved_hyperparameters": artifacts.resolved_hyperparameters.to_metadata(),
                "baseline_emulator_kind": "dedicated_fixed_budget_pca_gp",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return Process1GPBaselineResult(
        output_dir=output_dir,
        test_set_results_path=validation.test_set_results_path.resolve(),
        summary_path=summary_path,
        run_metadata_path=run_metadata_path,
        hifi_bank_path=hifi_bank_path,
        data_source=str(existing_bank_source),
        train_points=int(train_thetas.shape[0]),
    )


__all__ = [
    "GPBaselineHifiCounts",
    "GPBaselineTrainingArtifacts",
    "Process1GPBaselineResult",
    "ResolvedGPBaselineHyperparameters",
    "STANDARD_SOBOL_GP_OUTPUT_SUBDIR",
    "STANDARD_SOBOL_GP_TRAIN_POINTS",
    "StandardSobolGPBaselineResult",
    "count_unique_hifi_points",
    "evaluate_gp_baseline_with_reused_truth",
    "fit_gp_baseline",
    "fit_gp_baseline_from_hifi_bank",
    "fit_gp_baseline_from_spectrum_bank",
    "predict_gp_baseline_spectra",
    "resolve_gp_baseline_hyperparameters",
    "run_process1_gp_baseline",
    "run_standard_sobol_gp_baseline",
]
