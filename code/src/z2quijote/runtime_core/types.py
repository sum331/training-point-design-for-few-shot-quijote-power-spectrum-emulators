"""Shared dataclasses and typing aliases for the active-learning emulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ArrayLike: TypeAlias = Any
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(slots=True)
class Module1Dataset:
    raw_thetas: FloatArray
    unit_thetas: FloatArray
    k_bins: FloatArray
    pk_batch: FloatArray
    p_linear_batch: FloatArray | None
    log_pk_batch: FloatArray
    target_batch: FloatArray
    pca_scores: FloatArray
    pca_score_mean: FloatArray
    pca_score_std: FloatArray
    pca_components: FloatArray
    pca_mean: FloatArray
    explained_variance_ratio: FloatArray
    pca_model: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmulatorState:
    dataset: Module1Dataset
    gp_models: list[Any]
    theta_bounds: FloatArray
    kernel_descriptions: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpectrumPrediction:
    raw_thetas: FloatArray
    unit_thetas: FloatArray
    k_bins: FloatArray
    pc_mean: FloatArray
    pc_std: FloatArray
    target_mean: FloatArray
    log_pk_mean: FloatArray
    pk_mean: FloatArray
    p_linear_batch: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContinuousPosteriorState:
    theta_bounds: FloatArray
    train_raw_thetas: FloatArray
    train_unit_thetas: FloatArray
    gp_models: list[Any]
    kernel_descriptions: list[str]
    pc_lengthscales: FloatArray
    pc_signal_variances: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Module3ContinuousInput:
    continuous_state: ContinuousPosteriorState
    iteration_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContinuousVarianceEvaluation:
    raw_thetas: FloatArray
    unit_thetas: FloatArray
    pc_var: FloatArray
    pc_std: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SelectionResult:
    selected_raw_thetas: FloatArray
    selected_unit_thetas: FloatArray
    selected_source_pc: NDArray[np.int64]
    selected_scores: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IterationRecord:
    iteration_index: int
    train_size_before: int
    train_size_after: int
    selected_raw_thetas: FloatArray
    selected_unit_thetas: FloatArray
    selected_source_pc: NDArray[np.int64]
    selected_scores: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationArtifacts:
    output_dir: Any
    test_set_results_path: Any
    summary_path: Any
    metadata_path: Any
    metadata: dict[str, Any] = field(default_factory=dict)
