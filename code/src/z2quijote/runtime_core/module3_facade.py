"""Module 3 facade: continuous selection contract and selector normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from z2quijote.runtime_core.config import ValidationRuntimeConfig
from z2quijote.runtime_core.module3_shared_delaunay_gpu_selector import SharedHullDelaunayGPUSelector
from z2quijote.runtime_core.types import Module3ContinuousInput, SelectionResult

ProgressCallback = Callable[[str, int, int], None]


class Module3Selector(ABC):
    """Abstract selector interface.

    Implementations consume continuous GP state and return one batch of unique
    next-sample points.
    """

    @abstractmethod
    def select_next_batch(
        self,
        config: ValidationRuntimeConfig,
        module3_input: Module3ContinuousInput,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> SelectionResult:
        """Return the next batch of selected points."""


class InterfaceOnlySelector(Module3Selector):
    """Placeholder selector used until the real module3 algorithm is plugged in."""

    def select_next_batch(
        self,
        config: ValidationRuntimeConfig,
        module3_input: Module3ContinuousInput,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> SelectionResult:
        del module3_input, progress_callback
        raise NotImplementedError(
            "module3 currently only exposes the input/output interface. "
            "Inject a concrete Module3Selector implementation to continue active iterations."
        )


def _coerce_2d_theta(array: np.ndarray, *, name: str, theta_dim: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != theta_dim:
        raise ValueError(f"{name} must have shape [N,{theta_dim}], got {arr.shape}.")
    return arr.astype(np.float64)


def _normalize_selection_result(
    config: ValidationRuntimeConfig,
    module3_input: Module3ContinuousInput,
    selection: SelectionResult,
) -> SelectionResult:
    theta_dim = int(module3_input.continuous_state.theta_bounds.shape[0])
    expected_batch_size = int(config.sampling.batch_size)
    raw = _coerce_2d_theta(
        selection.selected_raw_thetas,
        name="selected_raw_thetas",
        theta_dim=theta_dim,
    )
    unit = _coerce_2d_theta(
        selection.selected_unit_thetas,
        name="selected_unit_thetas",
        theta_dim=theta_dim,
    )
    if raw.shape[0] != unit.shape[0]:
        raise ValueError("selected_raw_thetas and selected_unit_thetas row counts must match.")
    if raw.shape[0] != expected_batch_size:
        raise ValueError(
            "module3 must return exactly config.sampling.batch_size points, "
            f"expected {expected_batch_size}, got {raw.shape[0]}."
        )

    source_pc = np.asarray(selection.selected_source_pc, dtype=np.int64).reshape(-1)
    scores = np.asarray(selection.selected_scores, dtype=np.float64).reshape(-1)
    if source_pc.shape[0] != expected_batch_size:
        raise ValueError("selected_source_pc length must match batch_size.")
    if scores.shape[0] != expected_batch_size:
        raise ValueError("selected_scores length must match batch_size.")

    pc_dim = int(module3_input.continuous_state.pc_lengthscales.shape[0])
    if np.any(source_pc < -1) or np.any(source_pc >= pc_dim):
        raise ValueError(
            f"selected_source_pc entries must be in [-1, {pc_dim - 1}] for the current emulator."
        )

    return SelectionResult(
        selected_raw_thetas=raw,
        selected_unit_thetas=unit,
        selected_source_pc=source_pc,
        selected_scores=scores,
        metadata=dict(selection.metadata),
    )


def select_next_batch(
    config: ValidationRuntimeConfig,
    module3_input: Module3ContinuousInput,
    *,
    selector: Module3Selector | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SelectionResult:
    active_selector = selector or InterfaceOnlySelector()
    selection = active_selector.select_next_batch(
        config=config,
        module3_input=module3_input,
        progress_callback=progress_callback,
    )
    return _normalize_selection_result(config, module3_input, selection)


def build_default_online_selector() -> Module3Selector:
    """Return the default online module3 selector used by CLI entrypoints."""

    return SharedHullDelaunayGPUSelector()


__all__ = [
    "build_default_online_selector",
    "InterfaceOnlySelector",
    "Module3Selector",
    "ProgressCallback",
    "SharedHullDelaunayGPUSelector",
    "select_next_batch",
]
