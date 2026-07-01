"""Shared-hull and domain-support Delaunay selector for continuous module3 search."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product
from typing import Any, Callable, Sequence

import numpy as np
from scipy.spatial import Delaunay, QhullError
from scipy.stats import qmc
import torch

from z2quijote.runtime_core.config import ValidationRuntimeConfig, denormalize_theta_batch
from z2quijote.runtime_core.module2_facade import build_pca_component_weights_from_band_sensitivity
from z2quijote.runtime_core.representation import (
    build_component_weight_vector_from_groups,
    parse_target_transform,
    target_transform_name,
)
from z2quijote.runtime_core.types import Module3ContinuousInput, SelectionResult

ProgressCallback = Callable[[str, int, int], None]

_DEFAULT_BAND_LABELS: tuple[str, ...] = (
    "low_0.01_0.07",
    "mid_0.07_0.5",
    "high_0.5_1",
    "tail_1_3",
)


@dataclass(slots=True)
class _SharedHullGeometry:
    simplices: np.ndarray
    simplex_vertices: np.ndarray
    hull_delaunay: Delaunay


@dataclass(slots=True)
class _SimplexCollection:
    name: str
    simplex_vertices: np.ndarray
    simplex_ids: np.ndarray


@dataclass(slots=True)
class _SimplexScoring:
    collection_name: str
    simplex_vertices: np.ndarray
    simplex_ids: np.ndarray
    repr_points: np.ndarray
    repr_scores: np.ndarray
    barycenter_bary: np.ndarray
    circumcenter_bary: np.ndarray
    circumcenter_valid: np.ndarray
    best_vertex_index: np.ndarray
    facet_center_bary: np.ndarray
    facet_center_scores: np.ndarray
    best_facet_center_index: np.ndarray


@dataclass(slots=True)
class _ComponentCandidateRanking:
    component_index: int
    unit_points: np.ndarray
    scores: np.ndarray
    simplex_ids: np.ndarray


@dataclass(slots=True)
class _ObjectiveSpec:
    objective_index: int
    objective_label: str
    component_weights: np.ndarray
    details: dict[str, Any]


@dataclass(slots=True)
class _RuntimeSelectorParams:
    coverage_mode: str
    domain_support_scheme: str
    domain_support_point_count: int
    domain_neighbor_count: int
    domain_support_fanout: int
    domain_score_scale: float
    objective_mode: str
    target_transform: str
    representation_global_weight: float
    representation_band_weights: tuple[float, ...]
    refinement_architecture: str
    weight_function: str
    weight_temperature: float
    pc_weight_beta: float
    pc_weight_alpha_low: float
    pc_weight_alpha_mid: float
    pc_weight_alpha_focus_high: float
    pc_weight_alpha_tail: float
    pc_weight_min: float
    pc_weight_max: float
    band_beta_low: float
    band_beta_mid: float
    band_beta_focus_high: float
    band_beta_tail: float
    band_alpha_low: tuple[float, ...]
    band_alpha_mid: tuple[float, ...]
    band_alpha_focus_high: tuple[float, ...]
    band_alpha_tail: tuple[float, ...]
    band_weight_min: float
    band_weight_max: float
    acquisition_density_weight_power: float
    acquisition_density_weight_floor: float
    acquisition_spacefill_rerank_top_k: int
    acquisition_spacefill_weight: float
    acquisition_spacefill_guard_top_k: int
    acquisition_spacefill_guard_reject_quantile: float
    acquisition_spacefill_tiebreak_top_k: int
    acquisition_spacefill_tiebreak_score_ratio: float
    acquisition_spacefill_cd_nonworse_top_k: int
    acquisition_spacefill_cd_nonworse_tol: float
    acquisition_p68_set_rerank_top_k: int
    acquisition_p68_set_rerank_score_ratio: float
    acquisition_p68_set_rerank_risk_mode: str
    acquisition_p68_set_rerank_band_weights: tuple[float, ...]
    acquisition_p68_set_rerank_acq_weight: float
    acquisition_p68_set_rerank_p68_weight: float
    acquisition_p68_set_rerank_spacefill_weight: float
    acquisition_p68_set_rerank_boundary_weight: float
    acquisition_p68_set_rerank_boundary_threshold: float
    acquisition_p68_set_rerank_boundary_target_fraction: float
    acquisition_p68_loo_guard_top_k: int
    acquisition_p68_loo_guard_score_ratio: float
    acquisition_p68_loo_guard_reject_quantile: float
    acquisition_p68_loo_guard_bandwidth: float
    acquisition_p68_loo_guard_band_weights: tuple[float, ...]
    acquisition_p68_loo_guard_stage: str
    acquisition_qmc_pool_count: int
    acquisition_qmc_pool_seed_offset: int
    acquisition_qmc_pool_static_seed: bool
    imse_rerank_top_k: int
    imse_probe_count: int
    imse_probe_seed_offset: int
    imse_rerank_mode: str
    imse_quantile: float
    imse_quantile_shell_width: float
    imse_quantile_mean_weight: float
    imse_quantile_max_weight: float
    repr_score_mode: str
    repr_dirichlet_probe_count: int
    stage0_chunk_size: int
    hull_refine_fraction: float
    domain_refine_all: bool
    global_top_k: int
    domain_top_k: int
    hierarchical_stage1_refine_fraction: float
    hierarchical_stage1_top_k: int
    hierarchical_stage1_starts_per_simplex_refine: int
    hierarchical_stage1_max_iter_refine: int
    hierarchical_stage1_history_size_refine: int
    hierarchical_stage1_convergence_tol_refine: float
    hierarchical_stage2_refine_fraction: float
    hierarchical_stage2_top_k: int
    hierarchical_stage2_starts_per_simplex_refine: int
    hierarchical_stage2_max_iter_refine: int
    hierarchical_stage2_history_size_refine: int
    hierarchical_stage2_convergence_tol_refine: float
    starts_per_simplex_refine: int
    max_iter_refine: int
    history_size_refine: int
    polish_top_k: int
    stage3_refine_fraction: float
    polish_starts_per_simplex_refine: int
    polish_max_iter_refine: int
    polish_history_size_refine: int
    polish_convergence_tol_refine: float
    stage3_qmc_top_k: int
    stage3_qmc_sample_count: int
    stage3_qmc_chunk_size: int
    chunk_size: int
    duplicate_tol: float
    armijo_c1: float
    line_search_steps_refine: tuple[float, ...]
    fallback_step_refine: float
    convergence_tol_refine: float
    variance_floor: float
    curvature_tol: float
    perturbation_eps_refine: float


class _TorchPosteriorVarianceEvaluator:
    def __init__(
        self,
        *,
        train_unit_thetas: np.ndarray,
        cholesky_factor: np.ndarray,
        lengthscales: np.ndarray,
        signal_variance: float,
        output_scale_sq: float,
        device: torch.device,
        dtype: torch.dtype,
        variance_floor: float,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.variance_floor = float(variance_floor)
        self.u = torch.as_tensor(train_unit_thetas, device=device, dtype=dtype)
        # Keep the Cholesky factor and solve triangular systems directly.
        # This avoids the large numerical errors that appear near training points
        # when evaluating k K^{-1} k through an explicit inverse in float32.
        self.cholesky_factor = torch.as_tensor(cholesky_factor, device=device, dtype=dtype)
        self.inv_lengthscale_sq = torch.as_tensor(
            1.0 / np.maximum(np.asarray(lengthscales, dtype=np.float64), 1.0e-12) ** 2,
            device=device,
            dtype=dtype,
        )
        self.signal_variance = torch.as_tensor(
            float(max(signal_variance, 1.0e-12)),
            device=device,
            dtype=dtype,
        )
        self.output_scale_sq = torch.as_tensor(
            float(max(output_scale_sq, 1.0e-12)),
            device=device,
            dtype=dtype,
        )

    def _kernel_cross(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diff = query[:, None, :] - self.u[None, :, :]
        scaled_sq = diff.square() * self.inv_lengthscale_sq[None, None, :]
        kernel = self.signal_variance * torch.exp(-0.5 * torch.sum(scaled_sq, dim=2))
        return kernel, diff

    def _kernel_between(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        diff = left[:, None, :] - right[None, :, :]
        scaled_sq = diff.square() * self.inv_lengthscale_sq[None, None, :]
        return self.signal_variance * torch.exp(-0.5 * torch.sum(scaled_sq, dim=2))

    def _solve_kernel_system(self, kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.linalg.solve_triangular(
            self.cholesky_factor,
            kernel.transpose(0, 1),
            upper=False,
        )
        beta_t = torch.linalg.solve_triangular(
            self.cholesky_factor.transpose(0, 1),
            z,
            upper=True,
        )
        beta_rows = beta_t.transpose(0, 1)
        quadratic_form = torch.sum(z.square(), dim=0)
        return beta_rows, quadratic_form

    def value(self, query: torch.Tensor) -> torch.Tensor:
        kernel, _ = self._kernel_cross(query)
        _, quadratic_form = self._solve_kernel_system(kernel)
        posterior = self.signal_variance - quadratic_form
        posterior = torch.clamp(posterior, min=self.variance_floor)
        values = self.output_scale_sq * posterior
        return torch.nan_to_num(
            values,
            nan=self.variance_floor,
            posinf=self.variance_floor,
            neginf=self.variance_floor,
        )

    def normalized_variance(self, query: torch.Tensor) -> torch.Tensor:
        kernel, _ = self._kernel_cross(query)
        _, quadratic_form = self._solve_kernel_system(kernel)
        posterior = self.signal_variance - quadratic_form
        return torch.clamp(
            torch.nan_to_num(
                posterior,
                nan=self.variance_floor,
                posinf=self.variance_floor,
                neginf=self.variance_floor,
            ),
            min=self.variance_floor,
        )

    def posterior_cross_covariance(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_kernel, _ = self._kernel_cross(left)
        right_kernel, _ = self._kernel_cross(right)
        right_beta, _ = self._solve_kernel_system(right_kernel)
        prior = self._kernel_between(left, right)
        correction = left_kernel @ right_beta.transpose(0, 1)
        return torch.nan_to_num(prior - correction, nan=0.0, posinf=0.0, neginf=0.0)

    def imse_reduction(
        self,
        candidates: torch.Tensor,
        probes: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        if candidates.shape[0] == 0 or probes.shape[0] == 0:
            return torch.empty((0,), device=self.device, dtype=self.dtype)
        scores: list[torch.Tensor] = []
        step = max(1, int(chunk_size))
        for start in range(0, candidates.shape[0], step):
            candidate_chunk = candidates[start : start + step]
            covariance = self.posterior_cross_covariance(probes, candidate_chunk)
            candidate_variance = self.normalized_variance(candidate_chunk)
            reduction = torch.mean(
                covariance.square() / torch.clamp(candidate_variance[None, :], min=self.variance_floor),
                dim=0,
            )
            scores.append(self.output_scale_sq * reduction)
        return torch.cat(scores, dim=0)

    def variance_reduction_matrix(
        self,
        candidates: torch.Tensor,
        probes: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        if candidates.shape[0] == 0 or probes.shape[0] == 0:
            return torch.empty((probes.shape[0], candidates.shape[0]), device=self.device, dtype=self.dtype)
        chunks: list[torch.Tensor] = []
        step = max(1, int(chunk_size))
        for start in range(0, candidates.shape[0], step):
            candidate_chunk = candidates[start : start + step]
            covariance = self.posterior_cross_covariance(probes, candidate_chunk)
            candidate_variance = self.normalized_variance(candidate_chunk)
            reduction = covariance.square() / torch.clamp(
                candidate_variance[None, :],
                min=self.variance_floor,
            )
            chunks.append(self.output_scale_sq * reduction)
        return torch.cat(chunks, dim=1)

    def value_and_grad(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kernel, diff = self._kernel_cross(query)
        beta_rows, quadratic_form = self._solve_kernel_system(kernel)
        posterior = self.signal_variance - quadratic_form
        posterior = torch.clamp(posterior, min=self.variance_floor)
        grad = 2.0 * torch.sum(
            beta_rows[:, :, None]
            * diff
            * self.inv_lengthscale_sq[None, None, :]
            * kernel[:, :, None],
            dim=1,
        )
        values = self.output_scale_sq * posterior
        grad = self.output_scale_sq * grad
        return (
            torch.nan_to_num(
                values,
                nan=self.variance_floor,
                posinf=self.variance_floor,
                neginf=self.variance_floor,
            ),
            torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0),
        )


class _TorchAggregatePosteriorVarianceEvaluator:
    def __init__(
        self,
        evaluators: list[_TorchPosteriorVarianceEvaluator],
        component_weights: np.ndarray,
        *,
        device: torch.device,
        dtype: torch.dtype,
        variance_floor: float,
    ) -> None:
        if not evaluators:
            raise ValueError("Aggregate posterior variance evaluator requires at least one component evaluator.")
        self.evaluators = list(evaluators)
        self.device = device
        self.dtype = dtype
        self.variance_floor = float(variance_floor)
        weights = np.asarray(component_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != len(self.evaluators):
            raise ValueError(
                "component_weights must align with evaluators, "
                f"got {weights.shape[0]} vs {len(self.evaluators)}."
            )
        self.component_weights = torch.as_tensor(weights, device=device, dtype=dtype)

    def value(self, query: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((query.shape[0],), device=self.device, dtype=self.dtype)
        for component_idx, evaluator in enumerate(self.evaluators):
            total = total + self.component_weights[component_idx] * evaluator.value(query)
        return torch.clamp(total, min=self.variance_floor)

    def value_and_grad(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_value = torch.zeros((query.shape[0],), device=self.device, dtype=self.dtype)
        total_grad = torch.zeros_like(query)
        for component_idx, evaluator in enumerate(self.evaluators):
            value, grad = evaluator.value_and_grad(query)
            weight = self.component_weights[component_idx]
            total_value = total_value + weight * value
            total_grad = total_grad + weight * grad
        return (
            torch.clamp(total_value, min=self.variance_floor),
            torch.nan_to_num(total_grad, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def imse_reduction(
        self,
        candidates: torch.Tensor,
        probes: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        total = torch.zeros((candidates.shape[0],), device=self.device, dtype=self.dtype)
        for component_idx, evaluator in enumerate(self.evaluators):
            total = total + self.component_weights[component_idx] * evaluator.imse_reduction(
                candidates,
                probes,
                chunk_size=chunk_size,
            )
        return torch.clamp(total, min=self.variance_floor)

    def variance_reduction_matrix(
        self,
        candidates: torch.Tensor,
        probes: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        total = torch.zeros((probes.shape[0], candidates.shape[0]), device=self.device, dtype=self.dtype)
        for component_idx, evaluator in enumerate(self.evaluators):
            total = total + self.component_weights[component_idx] * evaluator.variance_reduction_matrix(
                candidates,
                probes,
                chunk_size=chunk_size,
            )
        return torch.clamp(total, min=0.0)


def _bulk_density_weight_and_grad(
    query: torch.Tensor,
    *,
    power: float,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if float(power) <= 0.0 or float(floor) >= 1.0:
        return (
            torch.ones((query.shape[0],), device=query.device, dtype=query.dtype),
            torch.zeros_like(query),
        )
    floor_value = float(min(1.0, max(0.0, floor)))
    power_value = float(max(0.0, power))
    raw = 4.0 * query * (1.0 - query)
    eps = torch.as_tensor(1.0e-6, device=query.device, dtype=query.dtype)
    interior = torch.clamp(raw, min=eps, max=1.0)
    log_bulk = power_value * torch.mean(torch.log(interior), dim=1)
    bulk = torch.exp(log_bulk)
    active = raw > eps
    raw_grad = 4.0 * (1.0 - 2.0 * query)
    interior_grad = torch.where(active, raw_grad, torch.zeros_like(raw_grad))
    bulk_grad = bulk[:, None] * (power_value / float(query.shape[1])) * (interior_grad / interior)
    weight = floor_value + (1.0 - floor_value) * bulk
    grad = (1.0 - floor_value) * bulk_grad
    return (
        torch.nan_to_num(weight, nan=floor_value, posinf=1.0, neginf=floor_value),
        torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0),
    )


class _DensityWeightedPosteriorVarianceEvaluator:
    def __init__(
        self,
        base: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator,
        *,
        power: float,
        floor: float,
    ) -> None:
        self.base = base
        self.device = base.device
        self.dtype = base.dtype
        self.variance_floor = float(base.variance_floor)
        self.power = float(max(0.0, power))
        self.floor = float(min(1.0, max(0.0, floor)))

    def value(self, query: torch.Tensor) -> torch.Tensor:
        base_value = self.base.value(query)
        weight, _ = _bulk_density_weight_and_grad(
            query,
            power=self.power,
            floor=self.floor,
        )
        return torch.clamp(base_value * weight, min=self.variance_floor)

    def value_and_grad(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_value, base_grad = self.base.value_and_grad(query)
        weight, weight_grad = _bulk_density_weight_and_grad(
            query,
            power=self.power,
            floor=self.floor,
        )
        value = torch.clamp(base_value * weight, min=self.variance_floor)
        grad = base_grad * weight[:, None] + base_value[:, None] * weight_grad
        return (
            torch.nan_to_num(value, nan=self.variance_floor, posinf=self.variance_floor),
            torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def imse_reduction(
        self,
        candidates: torch.Tensor,
        probes: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        return self.base.imse_reduction(candidates, probes, chunk_size=chunk_size)

    def variance_reduction_matrix(
        self,
        candidates: torch.Tensor,
        probes: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        return self.base.variance_reduction_matrix(candidates, probes, chunk_size=chunk_size)


def _positive_score_scale(values: np.ndarray, *, mode: str) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr >= 0.0)]
    if arr.size == 0:
        return 1.0
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "none":
        return 1.0
    if normalized_mode == "median":
        scale = float(np.percentile(arr, 50.0))
    elif normalized_mode == "p68":
        scale = float(np.percentile(arr, 68.0))
    else:
        scale = float(np.percentile(arr, 95.0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(arr))
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return float(scale)


class _BiasAugmentedPosteriorVarianceEvaluator:
    def __init__(
        self,
        base: Any,
        *,
        bias_model: Any,
        uncertainty_scale: float,
        bias_scale: float,
        bias_weight: float,
        normalization: str,
        score_mode: str = "variance_bias",
    ) -> None:
        self.base = base
        self.bias_model = bias_model
        self.device = base.device
        self.dtype = base.dtype
        self.variance_floor = float(base.variance_floor)
        self.uncertainty_scale = float(max(uncertainty_scale, 1.0e-30))
        self.bias_scale = float(max(bias_scale, 1.0e-30))
        self.bias_weight = float(max(bias_weight, 0.0))
        self.normalization = str(normalization)
        normalized_score_mode = str(score_mode).strip().lower()
        if normalized_score_mode not in {"variance_bias", "bias_only"}:
            normalized_score_mode = "variance_bias"
        self.score_mode = normalized_score_mode

    def _bias_tensor(self, query: torch.Tensor) -> torch.Tensor:
        unit_points = query.detach().cpu().numpy().astype(np.float64)
        bias = np.asarray(
            self._bias_numpy(
                unit_points,
                chunk_size=unit_points.shape[0],
                prefer_cuda_truth=True,
            ),
            dtype=np.float64,
        ).reshape(-1)
        bias = np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.as_tensor(bias, device=self.device, dtype=self.dtype)

    def _bias_numpy(
        self,
        unit_points: np.ndarray,
        *,
        chunk_size: int,
        prefer_cuda_truth: bool,
    ) -> np.ndarray:
        unit = np.asarray(unit_points, dtype=np.float64)
        if unit.ndim == 1:
            unit = unit.reshape(1, -1)
        if hasattr(self.bias_model, "bias_for_unit_batch"):
            bias = self.bias_model.bias_for_unit_batch(
                unit,
                chunk_size=int(max(1, chunk_size)),
                prefer_cuda_truth=bool(prefer_cuda_truth),
            )
        else:
            bias = self.bias_model.bias_for_unit(unit)
        return np.nan_to_num(
            np.asarray(bias, dtype=np.float64).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float64)

    def _combine_numpy(self, base_value: np.ndarray, bias: np.ndarray) -> np.ndarray:
        base = np.asarray(base_value, dtype=np.float64).reshape(-1)
        bias_arr = np.asarray(bias, dtype=np.float64).reshape(-1)
        u_norm = base / self.uncertainty_scale
        b_norm = self.bias_weight * bias_arr / self.bias_scale
        if self.score_mode == "bias_only":
            combined = np.abs(b_norm)
        else:
            combined = np.sqrt(np.maximum(u_norm**2 + b_norm**2, 0.0))
        return np.maximum(
            np.nan_to_num(
                combined,
                nan=self.variance_floor,
                posinf=self.variance_floor,
                neginf=self.variance_floor,
            ),
            self.variance_floor,
        ).astype(np.float64)

    def _combine(self, base_value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        u_norm = base_value / self.uncertainty_scale
        b_norm = self.bias_weight * bias / self.bias_scale
        if self.score_mode == "bias_only":
            combined = torch.abs(b_norm)
        else:
            combined = torch.sqrt(torch.clamp(u_norm.square() + b_norm.square(), min=0.0))
        return torch.clamp(
            torch.nan_to_num(combined, nan=self.variance_floor, posinf=self.variance_floor),
            min=self.variance_floor,
        )

    def value_numpy_points(self, points: np.ndarray, *, chunk_size: int) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        batch = int(max(1, min(int(chunk_size), 2048)))
        values: list[np.ndarray] = []
        start = 0
        while start < query.shape[0]:
            unit_chunk = query[start : start + batch]
            try:
                query_chunk = torch.as_tensor(unit_chunk, device=self.device, dtype=self.dtype)
                base_value_np = self.base.value(query_chunk).detach().cpu().numpy().astype(np.float64)
                del query_chunk
                _empty_cuda_cache(self.device)
                bias_np = self._bias_numpy(
                    unit_chunk,
                    chunk_size=batch,
                    prefer_cuda_truth=True,
                )
                _empty_cuda_cache(self.device)
                values.append(self._combine_numpy(base_value_np, bias_np))
                start += unit_chunk.shape[0]
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if self.device.type == "cuda" and batch > 1 and _is_cuda_oom(exc):
                    batch = max(1, batch // 2)
                    _empty_cuda_cache(self.device)
                    continue
                raise
        return np.concatenate(values, axis=0).astype(np.float64)

    def value(self, query: torch.Tensor) -> torch.Tensor:
        base_value = self.base.value(query)
        bias = self._bias_tensor(query)
        return self._combine(base_value, bias)

    def value_and_grad(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_value, base_grad = self.base.value_and_grad(query)
        bias = self._bias_tensor(query)
        u_norm = base_value / self.uncertainty_scale
        b_norm = self.bias_weight * bias / self.bias_scale
        if self.score_mode == "bias_only":
            combined = torch.abs(b_norm)
            grad = torch.zeros_like(base_grad)
        else:
            combined = torch.sqrt(torch.clamp(u_norm.square() + b_norm.square(), min=0.0))
            denom = torch.clamp(combined * self.uncertainty_scale, min=1.0e-30)
            grad = base_grad * (u_norm / denom)[:, None]
        return (
            torch.clamp(
                torch.nan_to_num(combined, nan=self.variance_floor, posinf=self.variance_floor),
                min=self.variance_floor,
            ),
            torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def imse_reduction(
        self,
        candidates: torch.Tensor,
        probes: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        return self.base.imse_reduction(candidates, probes, chunk_size=chunk_size)

    def variance_reduction_matrix(
        self,
        candidates: torch.Tensor,
        probes: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        return self.base.variance_reduction_matrix(candidates, probes, chunk_size=chunk_size)


def _quantile_proxy_reduction_scores(
    current_variance: torch.Tensor,
    reduction_matrix: torch.Tensor,
    *,
    variance_floor: float,
    quantile: float,
    mean_weight: float,
    max_weight: float,
) -> torch.Tensor:
    floor_tensor = torch.as_tensor(float(max(variance_floor, 1.0e-30)), device=current_variance.device, dtype=current_variance.dtype)
    current_error = torch.sqrt(torch.clamp(current_variance.reshape(-1), min=floor_tensor))
    after_variance = torch.clamp(
        current_variance.reshape(-1, 1) - reduction_matrix,
        min=floor_tensor,
    )
    after_error = torch.sqrt(after_variance)
    q = float(min(1.0, max(0.0, quantile)))
    before_q = torch.quantile(current_error, q)
    after_q = torch.quantile(after_error, q, dim=0)
    score = before_q - after_q
    if float(mean_weight) > 0.0:
        score = score + float(mean_weight) * (torch.mean(current_error) - torch.mean(after_error, dim=0))
    if float(max_weight) > 0.0:
        score = score + float(max_weight) * (torch.max(current_error) - torch.max(after_error, dim=0).values)
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _quantile_shell_reduction_scores(
    current_variance: torch.Tensor,
    reduction_matrix: torch.Tensor,
    *,
    variance_floor: float,
    quantile: float,
    shell_width: float,
    mean_weight: float,
    max_weight: float,
) -> torch.Tensor:
    floor_tensor = torch.as_tensor(
        float(max(variance_floor, 1.0e-30)),
        device=current_variance.device,
        dtype=current_variance.dtype,
    )
    current_error = torch.sqrt(torch.clamp(current_variance.reshape(-1), min=floor_tensor))
    after_variance = torch.clamp(
        current_variance.reshape(-1, 1) - reduction_matrix,
        min=floor_tensor,
    )
    after_error = torch.sqrt(after_variance)
    q = float(min(1.0, max(0.0, quantile)))
    half_width = 0.5 * float(min(1.0, max(1.0e-6, shell_width)))
    q_low = float(max(0.0, q - half_width))
    q_high = float(min(1.0, q + half_width))
    if q_high <= q_low:
        q_low = float(max(0.0, q - 0.5e-6))
        q_high = float(min(1.0, q + 0.5e-6))
    bounds = torch.quantile(
        current_error,
        torch.as_tensor([q_low, q_high], device=current_error.device, dtype=current_error.dtype),
    )
    shell_mask = (current_error >= bounds[0]) & (current_error <= bounds[1])
    if not bool(torch.any(shell_mask)):
        nearest_idx = torch.argmin(torch.abs(current_error - torch.quantile(current_error, q)))
        shell_mask = torch.zeros_like(current_error, dtype=torch.bool)
        shell_mask[nearest_idx] = True
    shell_reduction = current_error[:, None] - after_error
    score = torch.mean(shell_reduction[shell_mask, :], dim=0)
    if float(mean_weight) > 0.0:
        score = score + float(mean_weight) * (torch.mean(current_error) - torch.mean(after_error, dim=0))
    if float(max_weight) > 0.0:
        score = score + float(max_weight) * (torch.max(current_error) - torch.max(after_error, dim=0).values)
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _soft_quantile_reduction_scores(
    current_variance: torch.Tensor,
    reduction_matrix: torch.Tensor,
    *,
    variance_floor: float,
    quantile: float,
    bandwidth: float,
    mean_weight: float,
    max_weight: float,
) -> torch.Tensor:
    floor_tensor = torch.as_tensor(
        float(max(variance_floor, 1.0e-30)),
        device=current_variance.device,
        dtype=current_variance.dtype,
    )
    current_error = torch.sqrt(torch.clamp(current_variance.reshape(-1), min=floor_tensor))
    after_variance = torch.clamp(
        current_variance.reshape(-1, 1) - reduction_matrix,
        min=floor_tensor,
    )
    after_error = torch.sqrt(after_variance)
    q = float(min(1.0, max(0.0, quantile)))
    q_value = torch.quantile(current_error, q)
    scale = torch.clamp(
        torch.abs(q_value) * float(max(1.0e-6, bandwidth)),
        min=torch.sqrt(floor_tensor),
    )
    weights = torch.exp(-0.5 * torch.square((current_error - q_value) / scale))
    weights = weights / torch.clamp(torch.sum(weights), min=floor_tensor)
    score = torch.sum(weights[:, None] * (current_error[:, None] - after_error), dim=0)
    if float(mean_weight) > 0.0:
        score = score + float(mean_weight) * (torch.mean(current_error) - torch.mean(after_error, dim=0))
    if float(max_weight) > 0.0:
        score = score + float(max_weight) * (torch.max(current_error) - torch.max(after_error, dim=0).values)
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _rank_body_reduction_scores(
    current_variance: torch.Tensor,
    reduction_matrix: torch.Tensor,
    *,
    variance_floor: float,
    quantile: float,
    shell_width: float,
    mean_weight: float,
    max_weight: float,
) -> torch.Tensor:
    floor_tensor = torch.as_tensor(
        float(max(variance_floor, 1.0e-30)),
        device=current_variance.device,
        dtype=current_variance.dtype,
    )
    current_error = torch.sqrt(torch.clamp(current_variance.reshape(-1), min=floor_tensor))
    after_variance = torch.clamp(
        current_variance.reshape(-1, 1) - reduction_matrix,
        min=floor_tensor,
    )
    after_error = torch.sqrt(after_variance)
    if current_error.numel() == 0:
        return torch.zeros((reduction_matrix.shape[1],), device=reduction_matrix.device, dtype=reduction_matrix.dtype)
    q = float(min(1.0, max(0.0, quantile)))
    half_width = 0.5 * float(min(1.0, max(1.0e-6, shell_width)))
    sorted_indices = torch.argsort(current_error, stable=True)
    rank_positions = (
        torch.arange(
            current_error.numel(),
            device=current_error.device,
            dtype=current_error.dtype,
        )
        + 0.5
    ) / float(current_error.numel())
    rank_quantiles = torch.empty_like(rank_positions)
    rank_quantiles.scatter_(0, sorted_indices, rank_positions)
    distance = torch.abs(rank_quantiles - torch.as_tensor(q, device=current_error.device, dtype=current_error.dtype))
    scaled = distance / max(half_width, 1.0e-6)
    weights = torch.where(
        scaled <= 1.0,
        0.5 * (1.0 + torch.cos(torch.pi * scaled)),
        torch.zeros_like(scaled),
    )
    if not bool(torch.any(weights > 0.0)):
        nearest_idx = torch.argmin(distance)
        weights = torch.zeros_like(current_error)
        weights[nearest_idx] = 1.0
    weights = weights / torch.clamp(torch.sum(weights), min=floor_tensor)
    score = torch.sum(weights[:, None] * (current_error[:, None] - after_error), dim=0)
    if float(mean_weight) > 0.0:
        score = score + float(mean_weight) * (torch.mean(current_error) - torch.mean(after_error, dim=0))
    if float(max_weight) > 0.0:
        score = score + float(max_weight) * (torch.max(current_error) - torch.max(after_error, dim=0).values)
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _exceedance_reduction_scores(
    current_variance: torch.Tensor,
    reduction_matrix: torch.Tensor,
    *,
    variance_floor: float,
    quantile: float,
    mean_weight: float,
    max_weight: float,
) -> torch.Tensor:
    floor_tensor = torch.as_tensor(
        float(max(variance_floor, 1.0e-30)),
        device=current_variance.device,
        dtype=current_variance.dtype,
    )
    current_error = torch.sqrt(torch.clamp(current_variance.reshape(-1), min=floor_tensor))
    after_variance = torch.clamp(
        current_variance.reshape(-1, 1) - reduction_matrix,
        min=floor_tensor,
    )
    after_error = torch.sqrt(after_variance)
    q = float(min(1.0, max(0.0, quantile)))
    threshold = torch.quantile(current_error, q)
    before_excess = torch.clamp(current_error - threshold, min=0.0)
    after_excess = torch.clamp(after_error - threshold, min=0.0)
    score = torch.mean(before_excess[:, None] - after_excess, dim=0)
    if float(mean_weight) > 0.0:
        score = score + float(mean_weight) * (torch.mean(current_error) - torch.mean(after_error, dim=0))
    if float(max_weight) > 0.0:
        score = score + float(max_weight) * (torch.max(current_error) - torch.max(after_error, dim=0).values)
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _project_to_simplex_numpy(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    sorted_arr = np.sort(arr)[::-1]
    cssv = np.cumsum(sorted_arr) - 1.0
    rho_candidates = sorted_arr - cssv / np.arange(1, arr.size + 1) > 0.0
    rho = int(np.nonzero(rho_candidates)[0][-1])
    tau = cssv[rho] / float(rho + 1)
    projected = np.maximum(arr - tau, 0.0)
    total = float(projected.sum())
    if total <= 0.0:
        projected = np.full_like(projected, 1.0 / projected.size)
    else:
        projected /= total
    return projected.astype(np.float64)


def _project_to_simplex_torch(vectors: torch.Tensor) -> torch.Tensor:
    sorted_vectors, _ = torch.sort(vectors, dim=1, descending=True)
    cssv = torch.cumsum(sorted_vectors, dim=1) - 1.0
    k = torch.arange(1, vectors.shape[1] + 1, device=vectors.device, dtype=vectors.dtype)
    cond = sorted_vectors - cssv / k[None, :] > 0.0
    rho = cond.sum(dim=1).clamp_min(1) - 1
    tau = cssv.gather(1, rho.unsqueeze(1)) / (rho.to(vectors.dtype).unsqueeze(1) + 1.0)
    projected = torch.clamp(vectors - tau, min=0.0)
    total = projected.sum(dim=1, keepdim=True)
    return torch.where(
        total > 0.0,
        projected / total,
        torch.full_like(projected, 1.0 / projected.shape[1]),
    )


def _resolve_torch_device(config: ValidationRuntimeConfig) -> torch.device:
    if config.device == "cpu":
        return torch.device("cpu")
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config.device='cuda' was requested but CUDA is not available.")
        device_id = int(config.device_ids[0]) if config.device_ids else 0
        return torch.device(f"cuda:{device_id}")
    if torch.cuda.is_available():
        device_id = int(config.device_ids[0]) if config.device_ids else 0
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    except Exception:
        # Cache release is a best-effort guard against long-run WDDM fragmentation.
        pass


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _value_in_chunks(
    evaluator: Any,
    query: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    if query.shape[0] == 0:
        return torch.empty((0,), device=evaluator.device, dtype=evaluator.dtype)
    step = max(1, int(chunk_size))
    values: list[torch.Tensor] = []
    start = 0
    with torch.no_grad():
        while start < int(query.shape[0]):
            while True:
                end = min(int(query.shape[0]), start + step)
                try:
                    values.append(evaluator.value(query[start:end]))
                    _empty_cuda_cache(evaluator.device)
                    break
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    if evaluator.device.type == "cuda" and step > 1 and _is_cuda_oom(exc):
                        step = max(1, step // 2)
                        _empty_cuda_cache(evaluator.device)
                        continue
                    raise
            start = end
    return torch.cat(values, dim=0)


def _dirichlet_qmc_barycentric(
    *,
    simplex_size: int,
    sample_count: int,
) -> np.ndarray:
    if int(sample_count) <= 0:
        return np.empty((0, simplex_size), dtype=np.float64)
    if int(simplex_size) <= 1:
        return np.ones((int(sample_count), int(simplex_size)), dtype=np.float64)
    sampler = qmc.Sobol(d=int(simplex_size) - 1, scramble=False)
    power = int(np.ceil(np.log2(max(2, int(sample_count) + 1))))
    # Skip the all-zero Sobol point, then use the stick-breaking inverse CDF for
    # Dirichlet(1,...,1). Compared with sorted Sobol cuts, this avoids repeated
    # coordinates collapsing low-count probes onto simplex faces.
    unit = np.asarray(sampler.random_base2(power), dtype=np.float64)[1 : int(sample_count) + 1]
    if unit.shape[0] < int(sample_count):
        extra_power = int(np.ceil(np.log2(max(2, 2 * int(sample_count)))))
        unit = np.asarray(sampler.random_base2(extra_power), dtype=np.float64)[1 : int(sample_count) + 1]
    unit = np.clip(unit, 1.0e-9, 1.0 - 1.0e-9)
    bary = np.empty((unit.shape[0], int(simplex_size)), dtype=np.float64)
    remaining = np.ones((unit.shape[0],), dtype=np.float64)
    for coord_idx in range(int(simplex_size) - 1):
        beta_b = float(int(simplex_size) - coord_idx - 1)
        stick = 1.0 - np.power(1.0 - unit[:, coord_idx], 1.0 / beta_b)
        bary[:, coord_idx] = remaining * stick
        remaining *= 1.0 - stick
    bary[:, -1] = remaining
    # Lightly shrink toward the barycenter so low-count QMC probes stay interior.
    shrink = 0.05
    bary = (1.0 - shrink) * bary + shrink / float(simplex_size)
    bary = np.clip(bary, 1.0e-9, None)
    bary /= np.sum(bary, axis=1, keepdims=True)
    return bary.astype(np.float64)


def _fractional_stage_count(total: int, fraction: float) -> int:
    if int(total) <= 0:
        return 0
    frac = min(1.0, max(0.0, float(fraction)))
    if frac <= 0.0:
        return 0
    return int(min(int(total), max(1, int(np.ceil(frac * float(total))))))


def _append_lbfgs_history(
    *,
    s_history: torch.Tensor,
    y_history: torch.Tensor,
    counts: torch.Tensor,
    state_indices: torch.Tensor,
    s_update: torch.Tensor,
    y_update: torch.Tensor,
    curvature_tol: float,
) -> None:
    curvature = torch.sum(s_update * y_update, dim=1)
    valid = torch.isfinite(curvature) & (curvature > float(curvature_tol))
    if not torch.any(valid):
        return

    idx = state_indices[valid]
    s_valid = s_update[valid]
    y_valid = y_update[valid]
    counts_valid = counts.index_select(0, idx)
    history_size = int(s_history.shape[1])

    not_full = counts_valid < history_size
    if torch.any(not_full):
        idx_nf = idx[not_full]
        pos_nf = counts_valid[not_full].to(torch.int64)
        s_history[idx_nf, pos_nf, :] = s_valid[not_full]
        y_history[idx_nf, pos_nf, :] = y_valid[not_full]
        counts[idx_nf] = counts[idx_nf] + 1

    full = ~not_full
    if torch.any(full):
        idx_full = idx[full]
        shifted_s = s_history.index_select(0, idx_full)
        shifted_y = y_history.index_select(0, idx_full)
        shifted_s[:, :-1, :] = shifted_s[:, 1:, :].clone()
        shifted_y[:, :-1, :] = shifted_y[:, 1:, :].clone()
        shifted_s[:, -1, :] = s_valid[full]
        shifted_y[:, -1, :] = y_valid[full]
        s_history[idx_full] = shifted_s
        y_history[idx_full] = shifted_y


def _lbfgs_two_loop_direction(
    gradients: torch.Tensor,
    s_history: torch.Tensor,
    y_history: torch.Tensor,
    counts: torch.Tensor,
    curvature_tol: float,
) -> torch.Tensor:
    batch_size = gradients.shape[0]
    history_size = int(s_history.shape[1])
    q = gradients.clone()
    alpha = torch.zeros((batch_size, history_size), device=gradients.device, dtype=gradients.dtype)
    rho = torch.zeros_like(alpha)

    for history_idx in range(history_size - 1, -1, -1):
        active = counts > history_idx
        if not torch.any(active):
            continue
        s_col = s_history[:, history_idx, :]
        y_col = y_history[:, history_idx, :]
        sy = torch.sum(s_col * y_col, dim=1)
        valid = active & torch.isfinite(sy) & (sy > float(curvature_tol))
        if not torch.any(valid):
            continue
        rho_valid = 1.0 / sy[valid]
        alpha_valid = rho_valid * torch.sum(s_col[valid] * q[valid], dim=1)
        rho[valid, history_idx] = rho_valid
        alpha[valid, history_idx] = alpha_valid
        q[valid] = q[valid] - alpha_valid[:, None] * y_col[valid]

    gamma = torch.ones((batch_size,), device=gradients.device, dtype=gradients.dtype)
    has_history = counts > 0
    if torch.any(has_history):
        last_idx = (counts[has_history] - 1).to(torch.int64)
        last_s = s_history[has_history, last_idx, :]
        last_y = y_history[has_history, last_idx, :]
        sy = torch.sum(last_s * last_y, dim=1)
        yy = torch.sum(last_y * last_y, dim=1)
        valid = torch.isfinite(sy) & torch.isfinite(yy) & (sy > float(curvature_tol)) & (yy > float(curvature_tol))
        gamma_valid = torch.ones_like(sy)
        gamma_valid[valid] = sy[valid] / yy[valid]
        gamma[has_history] = gamma_valid

    r = gamma[:, None] * q
    for history_idx in range(history_size):
        active = counts > history_idx
        if not torch.any(active):
            continue
        s_col = s_history[:, history_idx, :]
        y_col = y_history[:, history_idx, :]
        valid = active & (rho[:, history_idx] > 0.0)
        if not torch.any(valid):
            continue
        beta = rho[valid, history_idx] * torch.sum(y_col[valid] * r[valid], dim=1)
        r[valid] = r[valid] + (alpha[valid, history_idx] - beta)[:, None] * s_col[valid]
    return r


def _resolve_unique_component_points(
    rankings: list[_ComponentCandidateRanking],
    *,
    duplicate_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_points: list[np.ndarray] = []
    selected_scores: list[float] = []
    source_pc: list[int] = []

    for ranking in rankings:
        found = False
        for point, score in zip(ranking.unit_points, ranking.scores, strict=True):
            if selected_points:
                distances = np.linalg.norm(np.asarray(selected_points) - point[None, :], axis=1)
                if np.any(distances <= float(duplicate_tol)):
                    continue
            selected_points.append(np.asarray(point, dtype=np.float64))
            selected_scores.append(float(score))
            source_pc.append(int(ranking.component_index))
            found = True
            break
        if not found:
            raise RuntimeError(
                f"Component {ranking.component_index} could not provide a unique point "
                f"under duplicate_tol={duplicate_tol}."
            )

    return (
        np.vstack(selected_points).astype(np.float64),
        np.asarray(selected_scores, dtype=np.float64),
        np.asarray(source_pc, dtype=np.int64),
    )


def _resolve_unique_points_from_ranking(
    ranking: _ComponentCandidateRanking,
    *,
    duplicate_tol: float,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(num_points) <= 0:
        raise ValueError("num_points must be positive.")

    selected_points: list[np.ndarray] = []
    selected_scores: list[float] = []
    selected_source_pc: list[int] = []
    for point, score in zip(ranking.unit_points, ranking.scores, strict=True):
        if selected_points:
            distances = np.linalg.norm(np.asarray(selected_points) - point[None, :], axis=1)
            if np.any(distances <= float(duplicate_tol)):
                continue
        selected_points.append(np.asarray(point, dtype=np.float64))
        selected_scores.append(float(score))
        selected_source_pc.append(int(ranking.component_index))
        if len(selected_points) >= int(num_points):
            break

    if len(selected_points) != int(num_points):
        raise RuntimeError(
            f"Unified module3 objective could only supply {len(selected_points)} unique points; "
            f"requested {num_points}."
        )

    return (
        np.vstack(selected_points).astype(np.float64),
        np.asarray(selected_scores, dtype=np.float64),
        np.asarray(selected_source_pc, dtype=np.int64),
    )


class SharedHullDelaunayGPUSelector:
    """Search a unified posterior-variance objective on a hybrid hull/domain simplex set."""

    def __init__(
        self,
        *,
        starts_per_simplex: int | None = None,
        max_iter: int | None = None,
        history_size: int | None = None,
        chunk_size: int | None = None,
        duplicate_tol: float | None = None,
        armijo_c1: float | None = None,
        line_search_steps: tuple[float, ...] | None = None,
        fallback_step: float | None = None,
        circumcenter_tol: float = 1.0e-6,
        convergence_tol: float | None = None,
        variance_floor: float | None = None,
        curvature_tol: float | None = None,
        perturbation_eps: float | None = None,
        global_top_k: int | None = None,
        domain_top_k: int | None = None,
        domain_support_point_count: int | None = None,
        domain_support_scheme: str | None = None,
        coverage_mode: str | None = None,
    ) -> None:
        self.starts_per_simplex_override = starts_per_simplex
        self.max_iter_override = max_iter
        self.history_size_override = history_size
        self.chunk_size_override = chunk_size
        self.duplicate_tol_override = duplicate_tol
        self.armijo_c1_override = armijo_c1
        self.line_search_steps_override = line_search_steps
        self.fallback_step_override = fallback_step
        self.circumcenter_tol = float(circumcenter_tol)
        self.convergence_tol_override = convergence_tol
        self.variance_floor_override = variance_floor
        self.curvature_tol_override = curvature_tol
        self.perturbation_eps_override = perturbation_eps
        self.global_top_k_override = global_top_k
        self.domain_top_k_override = domain_top_k
        self.domain_support_point_count_override = domain_support_point_count
        self.domain_support_scheme_override = domain_support_scheme
        self.coverage_mode_override = coverage_mode

    def _resolve_runtime_params(self, config: ValidationRuntimeConfig) -> _RuntimeSelectorParams:
        m3 = config.m3
        return _RuntimeSelectorParams(
            coverage_mode=str(self.coverage_mode_override or m3.coverage_mode),
            domain_support_scheme=str(self.domain_support_scheme_override or m3.domain_support_scheme),
            domain_support_point_count=int(
                self.domain_support_point_count_override
                if self.domain_support_point_count_override is not None
                else m3.domain_support_point_count
            ),
            domain_neighbor_count=int(m3.domain_neighbor_count),
            domain_support_fanout=int(m3.domain_support_fanout),
            domain_score_scale=float(m3.domain_score_scale),
            objective_mode=str(m3.objective_mode),
            target_transform=target_transform_name(
                transform_family=str(config.representation.transform_family),
                anchor_mode=str(config.representation.anchor_mode),
            ),
            representation_global_weight=float(m3.representation_global_weight),
            representation_band_weights=tuple(float(value) for value in m3.representation_band_weights),
            refinement_architecture=str(m3.refinement_architecture),
            weight_function=str(m3.weight_function),
            weight_temperature=float(m3.weight_temperature),
            pc_weight_beta=float(m3.pc_weight_beta),
            pc_weight_alpha_low=float(m3.pc_weight_alpha_low),
            pc_weight_alpha_mid=float(m3.pc_weight_alpha_mid),
            pc_weight_alpha_focus_high=float(m3.pc_weight_alpha_focus_high),
            pc_weight_alpha_tail=float(m3.pc_weight_alpha_tail),
            pc_weight_min=float(m3.pc_weight_min),
            pc_weight_max=float(m3.pc_weight_max),
            band_beta_low=float(m3.band_beta_low),
            band_beta_mid=float(m3.band_beta_mid),
            band_beta_focus_high=float(m3.band_beta_focus_high),
            band_beta_tail=float(m3.band_beta_tail),
            band_alpha_low=tuple(float(value) for value in m3.band_alpha_low),
            band_alpha_mid=tuple(float(value) for value in m3.band_alpha_mid),
            band_alpha_focus_high=tuple(float(value) for value in m3.band_alpha_focus_high),
            band_alpha_tail=tuple(float(value) for value in m3.band_alpha_tail),
            band_weight_min=float(m3.band_weight_min),
            band_weight_max=float(m3.band_weight_max),
            acquisition_density_weight_power=float(m3.acquisition_density_weight_power),
            acquisition_density_weight_floor=float(m3.acquisition_density_weight_floor),
            acquisition_spacefill_rerank_top_k=int(m3.acquisition_spacefill_rerank_top_k),
            acquisition_spacefill_weight=float(m3.acquisition_spacefill_weight),
            acquisition_spacefill_guard_top_k=int(m3.acquisition_spacefill_guard_top_k),
            acquisition_spacefill_guard_reject_quantile=float(
                m3.acquisition_spacefill_guard_reject_quantile
            ),
            acquisition_spacefill_tiebreak_top_k=int(m3.acquisition_spacefill_tiebreak_top_k),
            acquisition_spacefill_tiebreak_score_ratio=float(
                m3.acquisition_spacefill_tiebreak_score_ratio
            ),
            acquisition_spacefill_cd_nonworse_top_k=int(m3.acquisition_spacefill_cd_nonworse_top_k),
            acquisition_spacefill_cd_nonworse_tol=float(m3.acquisition_spacefill_cd_nonworse_tol),
            acquisition_p68_set_rerank_top_k=int(m3.acquisition_p68_set_rerank_top_k),
            acquisition_p68_set_rerank_score_ratio=float(m3.acquisition_p68_set_rerank_score_ratio),
            acquisition_p68_set_rerank_risk_mode=str(m3.acquisition_p68_set_rerank_risk_mode),
            acquisition_p68_set_rerank_band_weights=tuple(
                float(value) for value in m3.acquisition_p68_set_rerank_band_weights
            ),
            acquisition_p68_set_rerank_acq_weight=float(m3.acquisition_p68_set_rerank_acq_weight),
            acquisition_p68_set_rerank_p68_weight=float(m3.acquisition_p68_set_rerank_p68_weight),
            acquisition_p68_set_rerank_spacefill_weight=float(m3.acquisition_p68_set_rerank_spacefill_weight),
            acquisition_p68_set_rerank_boundary_weight=float(m3.acquisition_p68_set_rerank_boundary_weight),
            acquisition_p68_set_rerank_boundary_threshold=float(m3.acquisition_p68_set_rerank_boundary_threshold),
            acquisition_p68_set_rerank_boundary_target_fraction=float(
                m3.acquisition_p68_set_rerank_boundary_target_fraction
            ),
            acquisition_p68_loo_guard_top_k=int(m3.acquisition_p68_loo_guard_top_k),
            acquisition_p68_loo_guard_score_ratio=float(m3.acquisition_p68_loo_guard_score_ratio),
            acquisition_p68_loo_guard_reject_quantile=float(m3.acquisition_p68_loo_guard_reject_quantile),
            acquisition_p68_loo_guard_bandwidth=float(m3.acquisition_p68_loo_guard_bandwidth),
            acquisition_p68_loo_guard_band_weights=tuple(
                float(value) for value in m3.acquisition_p68_loo_guard_band_weights
            ),
            acquisition_p68_loo_guard_stage=str(m3.acquisition_p68_loo_guard_stage),
            acquisition_qmc_pool_count=int(m3.acquisition_qmc_pool_count),
            acquisition_qmc_pool_seed_offset=int(m3.acquisition_qmc_pool_seed_offset),
            acquisition_qmc_pool_static_seed=bool(m3.acquisition_qmc_pool_static_seed),
            imse_rerank_top_k=int(m3.imse_rerank_top_k),
            imse_probe_count=int(m3.imse_probe_count),
            imse_probe_seed_offset=int(m3.imse_probe_seed_offset),
            imse_rerank_mode=str(m3.imse_rerank_mode),
            imse_quantile=float(m3.imse_quantile),
            imse_quantile_shell_width=float(m3.imse_quantile_shell_width),
            imse_quantile_mean_weight=float(m3.imse_quantile_mean_weight),
            imse_quantile_max_weight=float(m3.imse_quantile_max_weight),
            repr_score_mode=str(m3.repr_score_mode),
            repr_dirichlet_probe_count=int(m3.repr_dirichlet_probe_count),
            stage0_chunk_size=int(getattr(m3, "stage0_chunk_size", 131072)),
            hull_refine_fraction=float(m3.hull_refine_fraction),
            domain_refine_all=bool(m3.domain_refine_all),
            global_top_k=int(self.global_top_k_override if self.global_top_k_override is not None else m3.global_top_k),
            domain_top_k=int(self.domain_top_k_override if self.domain_top_k_override is not None else m3.domain_top_k),
            hierarchical_stage1_refine_fraction=float(getattr(m3, "hierarchical_stage1_refine_fraction", 0.25)),
            hierarchical_stage1_top_k=int(m3.hierarchical_stage1_top_k),
            hierarchical_stage1_starts_per_simplex_refine=int(m3.hierarchical_stage1_starts_per_simplex_refine),
            hierarchical_stage1_max_iter_refine=int(m3.hierarchical_stage1_max_iter_refine),
            hierarchical_stage1_history_size_refine=int(m3.hierarchical_stage1_history_size_refine),
            hierarchical_stage1_convergence_tol_refine=float(m3.hierarchical_stage1_convergence_tol_refine),
            hierarchical_stage2_refine_fraction=float(getattr(m3, "hierarchical_stage2_refine_fraction", 0.25)),
            hierarchical_stage2_top_k=int(m3.hierarchical_stage2_top_k),
            hierarchical_stage2_starts_per_simplex_refine=int(m3.hierarchical_stage2_starts_per_simplex_refine),
            hierarchical_stage2_max_iter_refine=int(m3.hierarchical_stage2_max_iter_refine),
            hierarchical_stage2_history_size_refine=int(m3.hierarchical_stage2_history_size_refine),
            hierarchical_stage2_convergence_tol_refine=float(m3.hierarchical_stage2_convergence_tol_refine),
            starts_per_simplex_refine=int(
                self.starts_per_simplex_override
                if self.starts_per_simplex_override is not None
                else m3.starts_per_simplex_refine
            ),
            max_iter_refine=int(self.max_iter_override if self.max_iter_override is not None else m3.max_iter_refine),
            history_size_refine=int(
                self.history_size_override
                if self.history_size_override is not None
                else m3.history_size_refine
            ),
            polish_top_k=int(m3.polish_top_k),
            stage3_refine_fraction=float(getattr(m3, "stage3_refine_fraction", 0.25)),
            polish_starts_per_simplex_refine=int(m3.polish_starts_per_simplex_refine),
            polish_max_iter_refine=int(m3.polish_max_iter_refine),
            polish_history_size_refine=int(m3.polish_history_size_refine),
            polish_convergence_tol_refine=float(m3.polish_convergence_tol_refine),
            stage3_qmc_top_k=int(getattr(m3, "stage3_qmc_top_k", 8)),
            stage3_qmc_sample_count=int(getattr(m3, "stage3_qmc_sample_count", 4096)),
            stage3_qmc_chunk_size=int(getattr(m3, "stage3_qmc_chunk_size", 32768)),
            chunk_size=int(self.chunk_size_override if self.chunk_size_override is not None else m3.chunk_size),
            duplicate_tol=float(self.duplicate_tol_override if self.duplicate_tol_override is not None else m3.duplicate_tol),
            armijo_c1=float(self.armijo_c1_override if self.armijo_c1_override is not None else m3.armijo_c1),
            line_search_steps_refine=tuple(
                float(step)
                for step in (
                    self.line_search_steps_override
                    if self.line_search_steps_override is not None
                    else m3.line_search_steps_refine
                )
            ),
            fallback_step_refine=float(
                self.fallback_step_override if self.fallback_step_override is not None else m3.fallback_step_refine
            ),
            convergence_tol_refine=float(
                self.convergence_tol_override
                if self.convergence_tol_override is not None
                else m3.convergence_tol_refine
            ),
            variance_floor=float(
                self.variance_floor_override if self.variance_floor_override is not None else m3.variance_floor
            ),
            curvature_tol=float(
                self.curvature_tol_override if self.curvature_tol_override is not None else m3.curvature_tol
            ),
            perturbation_eps_refine=float(
                self.perturbation_eps_override
                if self.perturbation_eps_override is not None
                else m3.perturbation_eps_refine
            ),
        )

    def _resolve_band_sensitivity(
        self,
        module3_input: Module3ContinuousInput,
    ) -> tuple[np.ndarray, list[str]]:
        continuous_state = module3_input.continuous_state
        component_count = int(len(continuous_state.gp_models))
        sensitivity = np.asarray(
            continuous_state.metadata.get("pca_band_sensitivity", []),
            dtype=np.float64,
        )
        band_count = len(_DEFAULT_BAND_LABELS)
        if sensitivity.shape != (component_count, band_count):
            raise ValueError(
                "continuous_state.metadata['pca_band_sensitivity'] must have shape "
                f"({component_count}, {band_count}), got {sensitivity.shape}."
            )
        raw_labels = continuous_state.metadata.get("pca_band_labels", list(_DEFAULT_BAND_LABELS))
        labels = [str(label).strip() for label in raw_labels]
        if len(labels) != band_count or any(not label for label in labels):
            labels = list(_DEFAULT_BAND_LABELS)
        return sensitivity.astype(np.float64), labels

    def _resolve_target_transform(
        self,
        module3_input: Module3ContinuousInput,
        params: _RuntimeSelectorParams,
    ) -> str:
        raw_value = module3_input.continuous_state.metadata.get("target_transform")
        if raw_value is not None and str(raw_value).strip():
            return str(raw_value).strip()
        return str(params.target_transform)

    def _resolve_target_transform_family(
        self,
        module3_input: Module3ContinuousInput,
        params: _RuntimeSelectorParams,
    ) -> str:
        target_transform = self._resolve_target_transform(module3_input, params)
        transform_family, _ = parse_target_transform(target_transform)
        return str(transform_family)

    def _resolve_exact_logdiff_variance_integrals(
        self,
        module3_input: Module3ContinuousInput,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        continuous_state = module3_input.continuous_state
        component_count = int(len(continuous_state.gp_models))
        band_integrals = np.asarray(
            continuous_state.metadata.get("pca_band_variance_integrals", []),
            dtype=np.float64,
        )
        band_count = len(_DEFAULT_BAND_LABELS)
        if band_integrals.shape != (component_count, band_count):
            raise ValueError(
                "continuous_state.metadata['pca_band_variance_integrals'] must have shape "
                f"({component_count}, {band_count}), got {band_integrals.shape}."
            )
        global_integral = np.asarray(
            continuous_state.metadata.get("pca_global_variance_integral", []),
            dtype=np.float64,
        ).reshape(-1)
        if global_integral.shape != (component_count,):
            raise ValueError(
                "continuous_state.metadata['pca_global_variance_integral'] must have shape "
                f"({component_count},), got {global_integral.shape}."
            )
        _, band_labels = self._resolve_band_sensitivity(module3_input)
        return (
            np.maximum(band_integrals.astype(np.float64), 0.0),
            np.maximum(global_integral.astype(np.float64), 0.0),
            list(band_labels),
        )

    def _normalize_exact_component_weights(
        self,
        component_weights: np.ndarray,
    ) -> np.ndarray:
        weights = np.asarray(component_weights, dtype=np.float64).reshape(-1)
        weights = np.maximum(weights, 0.0)
        mean_weight = float(np.mean(weights))
        if not np.isfinite(mean_weight) or mean_weight <= 0.0:
            raise ValueError("Exact logdiff component weights must contain at least one positive value.")
        return (weights / mean_weight).astype(np.float64)

    def _resolve_precomputed_logdiff_component_weights(
        self,
        module3_input: Module3ContinuousInput,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        continuous_state = module3_input.continuous_state
        component_count = int(len(continuous_state.gp_models))
        raw_weights = np.asarray(
            continuous_state.metadata.get("logdiff_projected_component_weights", []),
            dtype=np.float64,
        ).reshape(-1)
        if raw_weights.shape != (component_count,):
            return None, {}
        component_weights = self._normalize_exact_component_weights(raw_weights)
        details = dict(
            continuous_state.metadata.get("logdiff_projected_component_weight_details", {})
        )
        return component_weights.astype(np.float64), details

    def _build_component_weights_from_alpha_beta(
        self,
        *,
        sensitivity: np.ndarray,
        alpha: np.ndarray,
        beta: float,
        weight_min: float,
        weight_max: float,
        weight_function: str,
        weight_temperature: float,
    ) -> np.ndarray:
        return build_pca_component_weights_from_band_sensitivity(
            sensitivity,
            alpha=np.asarray(alpha, dtype=np.float64),
            beta=float(beta),
            weight_min=float(weight_min),
            weight_max=float(weight_max),
            weight_function=str(weight_function),
            weight_temperature=float(weight_temperature),
        ).astype(np.float64)

    def _resolve_component_weights(
        self,
        module3_input: Module3ContinuousInput,
        params: _RuntimeSelectorParams,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        continuous_state = module3_input.continuous_state
        component_count = int(len(continuous_state.gp_models))
        if params.objective_mode == "sum_pc_posterior_variance":
            ones = np.ones((component_count,), dtype=np.float64)
            return ones, {
                "mode": "uniform",
                "beta": 0.0,
                "component_weights": ones.tolist(),
            }

        transform_family = self._resolve_target_transform_family(module3_input, params)
        sensitivity, band_labels = self._resolve_band_sensitivity(module3_input)
        alpha = np.asarray(
            [
                float(params.pc_weight_alpha_low),
                float(params.pc_weight_alpha_mid),
                float(params.pc_weight_alpha_focus_high),
                float(params.pc_weight_alpha_tail),
            ],
            dtype=np.float64,
        )
        if transform_family == "logdiff":
            precomputed_weights, projection_details = self._resolve_precomputed_logdiff_component_weights(
                module3_input
            )
            if precomputed_weights is not None:
                return precomputed_weights.astype(np.float64), {
                    "mode": "projected_logdiff_k_curve",
                    "target_transform_family": "logdiff",
                    "component_weights": precomputed_weights.astype(np.float64).tolist(),
                    "projection_details": dict(projection_details),
                    "legacy_weight_shaping_applied": False,
                }
            band_integrals, global_integral, _ = self._resolve_exact_logdiff_variance_integrals(
                module3_input
            )
            raw_component_weights = band_integrals @ alpha
            component_weights = self._normalize_exact_component_weights(raw_component_weights)
            return component_weights.astype(np.float64), {
                "mode": "mid_high_exact_logdiff_band_variance",
                "target_transform_family": "logdiff",
                "alpha": alpha.astype(np.float64).tolist(),
                "raw_component_weights": raw_component_weights.astype(np.float64).tolist(),
                "component_weights": component_weights.astype(np.float64).tolist(),
                "band_labels": list(band_labels),
                "band_variance_integrals": band_integrals.astype(np.float64).tolist(),
                "global_variance_integral": global_integral.astype(np.float64).tolist(),
                "legacy_weight_shaping_applied": False,
            }
        beta = float(params.pc_weight_beta)
        component_weights = self._build_component_weights_from_alpha_beta(
            sensitivity=sensitivity,
            alpha=alpha,
            beta=beta,
            weight_min=float(params.pc_weight_min),
            weight_max=float(params.pc_weight_max),
            weight_function=str(params.weight_function),
            weight_temperature=float(params.weight_temperature),
        )
        return component_weights.astype(np.float64), {
            "mode": "mid_high_weighted_sum",
            "beta": beta,
            "alpha": alpha.astype(np.float64).tolist(),
            "weight_function": str(params.weight_function),
            "weight_temperature": float(params.weight_temperature),
            "weighted_component_scores": (np.asarray(sensitivity, dtype=np.float64) @ alpha).astype(np.float64).tolist(),
            "component_weights": component_weights.astype(np.float64).tolist(),
            "band_labels": list(band_labels),
            "sensitivity_matrix": sensitivity.astype(np.float64).tolist(),
            "mid_high_sensitivity": np.sum(sensitivity[:, 1:], axis=1).astype(np.float64).tolist(),
            "focus_0p08_3_sensitivity": np.sum(sensitivity[:, 1:], axis=1).astype(np.float64).tolist(),
            "focus_0p1_3_sensitivity": np.sum(sensitivity[:, 1:], axis=1).astype(np.float64).tolist(),
        }

    def _resolve_representation_group_weights(
        self,
        module3_input: Module3ContinuousInput,
        params: _RuntimeSelectorParams,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        continuous_state = module3_input.continuous_state
        component_count = int(len(continuous_state.gp_models))
        component_groups = list(
            continuous_state.metadata.get("representation_component_groups", [])
        )
        effective_global_weight = float(
            continuous_state.metadata.get(
                "m3_effective_global_weight",
                params.representation_global_weight,
            )
        )
        raw_band_weights = continuous_state.metadata.get(
            "m3_effective_band_weights",
            params.representation_band_weights,
        )
        band_weights = np.asarray(raw_band_weights, dtype=np.float64).reshape(-1)
        band_count = len(_DEFAULT_BAND_LABELS)
        if band_weights.shape != (band_count,):
            band_weights = np.asarray(params.representation_band_weights, dtype=np.float64)

        target_transform = self._resolve_target_transform(module3_input, params)
        transform_family = self._resolve_target_transform_family(module3_input, params)
        if transform_family == "logdiff":
            precomputed_weights, projection_details = self._resolve_precomputed_logdiff_component_weights(
                module3_input
            )
            if precomputed_weights is not None:
                return precomputed_weights.astype(np.float64), {
                    "mode": "representation_grouped_projected_logdiff_k_curve",
                    "target_transform_family": "logdiff",
                    "representation_global_weight": float(effective_global_weight),
                    "representation_band_weights": band_weights.astype(np.float64).tolist(),
                    "component_weights": precomputed_weights.astype(np.float64).tolist(),
                    "component_groups": [dict(group) for group in component_groups],
                    "pca_scheme": str(continuous_state.metadata.get("pca_scheme", "global_pca")),
                    "target_transform": target_transform,
                    "projection_details": dict(projection_details),
                    "legacy_group_weighting_applied": False,
                }
            band_integrals, global_integral, band_labels = self._resolve_exact_logdiff_variance_integrals(
                module3_input
            )
            raw_component_weights = (
                float(effective_global_weight) * global_integral
                + band_integrals @ band_weights.astype(np.float64)
            )
            component_weights = self._normalize_exact_component_weights(raw_component_weights)
            return component_weights.astype(np.float64), {
                "mode": "representation_grouped_exact_logdiff_band_variance",
                "target_transform_family": "logdiff",
                "representation_global_weight": float(effective_global_weight),
                "representation_band_weights": band_weights.astype(np.float64).tolist(),
                "raw_component_weights": raw_component_weights.astype(np.float64).tolist(),
                "component_weights": component_weights.astype(np.float64).tolist(),
                "band_labels": list(band_labels),
                "band_variance_integrals": band_integrals.astype(np.float64).tolist(),
                "global_variance_integral": global_integral.astype(np.float64).tolist(),
                "component_groups": [dict(group) for group in component_groups],
                "pca_scheme": str(continuous_state.metadata.get("pca_scheme", "global_pca")),
                "target_transform": target_transform,
                "legacy_group_weighting_applied": False,
            }

        if not component_groups:
            uniform = np.ones((component_count,), dtype=np.float64)
            return uniform, {
                "mode": "representation_grouped_posterior_variance",
                "fallback": "uniform_no_component_groups",
                "component_weights": uniform.tolist(),
                "representation_global_weight": float(effective_global_weight),
                "representation_band_weights": band_weights.astype(np.float64).tolist(),
                "component_groups": [],
            }

        component_weights = build_component_weight_vector_from_groups(
            component_groups,
            total_components=component_count,
            global_weight=float(effective_global_weight),
            band_weights=band_weights.astype(np.float64),
            default_weight=1.0,
        )
        return component_weights.astype(np.float64), {
            "mode": "representation_grouped_posterior_variance",
            "representation_global_weight": float(effective_global_weight),
            "representation_band_weights": band_weights.astype(np.float64).tolist(),
            "component_weights": component_weights.astype(np.float64).tolist(),
            "component_groups": [dict(group) for group in component_groups],
            "pca_scheme": str(continuous_state.metadata.get("pca_scheme", "global_pca")),
            "target_transform": target_transform,
        }

    def _resolve_objective_specs(
        self,
        module3_input: Module3ContinuousInput,
        params: _RuntimeSelectorParams,
    ) -> tuple[list[_ObjectiveSpec], dict[str, Any]]:
        if params.objective_mode == "representation_grouped_posterior_variance":
            component_weights, objective_details = self._resolve_representation_group_weights(
                module3_input,
                params,
            )
            return (
                [
                    _ObjectiveSpec(
                        objective_index=-1,
                        objective_label=str(params.objective_mode),
                        component_weights=component_weights.astype(np.float64),
                        details=dict(objective_details),
                    )
                ],
                dict(objective_details),
            )

        if params.objective_mode != "band_partitioned_posterior_variance":
            component_weights, objective_details = self._resolve_component_weights(module3_input, params)
            return (
                [
                    _ObjectiveSpec(
                        objective_index=-1,
                        objective_label=str(params.objective_mode),
                        component_weights=component_weights.astype(np.float64),
                        details=dict(objective_details),
                    )
                ],
                dict(objective_details),
            )

        transform_family = self._resolve_target_transform_family(module3_input, params)
        sensitivity, band_labels = self._resolve_band_sensitivity(module3_input)
        alpha_matrix = np.asarray(
            [
                params.band_alpha_low,
                params.band_alpha_mid,
                params.band_alpha_focus_high,
                params.band_alpha_tail,
            ],
            dtype=np.float64,
        )
        betas = np.asarray(
            [
                float(params.band_beta_low),
                float(params.band_beta_mid),
                float(params.band_beta_focus_high),
                float(params.band_beta_tail),
            ],
            dtype=np.float64,
        )
        specs: list[_ObjectiveSpec] = []
        band_details: list[dict[str, Any]] = []
        for band_index, band_label in enumerate(band_labels):
            alpha = alpha_matrix[band_index]
            beta = float(betas[band_index])
            if transform_family == "logdiff":
                band_integrals, _, _ = self._resolve_exact_logdiff_variance_integrals(module3_input)
                raw_component_weights = band_integrals[:, band_index]
                component_weights = self._normalize_exact_component_weights(raw_component_weights)
                detail = {
                    "band_index": int(band_index),
                    "band_label": str(band_label),
                    "mode": "band_partitioned_exact_logdiff_band_variance",
                    "target_transform_family": "logdiff",
                    "raw_component_weights": raw_component_weights.astype(np.float64).tolist(),
                    "component_weights": component_weights.astype(np.float64).tolist(),
                    "band_variance_integrals": band_integrals[:, band_index].astype(np.float64).tolist(),
                    "target_band_sensitivity": sensitivity[:, band_index].astype(np.float64).tolist(),
                    "legacy_weight_shaping_applied": False,
                }
            else:
                component_weights = self._build_component_weights_from_alpha_beta(
                    sensitivity=sensitivity,
                    alpha=alpha,
                    beta=beta,
                    weight_min=float(params.band_weight_min),
                    weight_max=float(params.band_weight_max),
                    weight_function=str(params.weight_function),
                    weight_temperature=float(params.weight_temperature),
                )
                detail = {
                    "band_index": int(band_index),
                    "band_label": str(band_label),
                    "beta": beta,
                    "alpha": alpha.astype(np.float64).tolist(),
                    "weight_function": str(params.weight_function),
                    "weight_temperature": float(params.weight_temperature),
                    "weighted_component_scores": (np.asarray(sensitivity, dtype=np.float64) @ alpha).astype(np.float64).tolist(),
                    "component_weights": component_weights.astype(np.float64).tolist(),
                    "target_band_sensitivity": sensitivity[:, band_index].astype(np.float64).tolist(),
                }
            specs.append(
                _ObjectiveSpec(
                    objective_index=int(band_index),
                    objective_label=str(band_label),
                    component_weights=component_weights.astype(np.float64),
                    details=dict(detail),
                )
            )
            band_details.append(detail)
        return specs, {
            "mode": (
                "band_partitioned_exact_logdiff_band_variance"
                if transform_family == "logdiff"
                else "band_partitioned_posterior_variance"
            ),
            "selection_strategy": "per_band_top1",
            "target_transform_family": str(transform_family),
            "band_labels": list(band_labels),
            "sensitivity_matrix": sensitivity.astype(np.float64).tolist(),
            "band_weight_min": float(params.band_weight_min),
            "band_weight_max": float(params.band_weight_max),
            "band_objectives": band_details,
        }

    def _resolve_p68_set_rerank_risk_component_weights(
        self,
        module3_input: Module3ContinuousInput,
        params: _RuntimeSelectorParams,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        band_weights = np.asarray(
            params.acquisition_p68_set_rerank_band_weights,
            dtype=np.float64,
        ).reshape(-1)
        band_count = len(_DEFAULT_BAND_LABELS)
        if band_weights.shape != (band_count,) or float(np.sum(band_weights)) <= 0.0:
            return None, {
                "mode": "base_objective",
                "band_weights": band_weights.astype(np.float64).tolist()
                if band_weights.shape == (band_count,)
                else [],
            }

        transform_family = self._resolve_target_transform_family(module3_input, params)
        if transform_family == "logdiff":
            band_integrals, _, band_labels = self._resolve_exact_logdiff_variance_integrals(
                module3_input
            )
            raw_component_weights = band_integrals @ band_weights.astype(np.float64)
            component_weights = self._normalize_exact_component_weights(raw_component_weights)
            return component_weights.astype(np.float64), {
                "mode": "band_weighted_exact_logdiff_band_variance",
                "target_transform_family": "logdiff",
                "band_weights": band_weights.astype(np.float64).tolist(),
                "band_labels": list(band_labels),
                "raw_component_weights": raw_component_weights.astype(np.float64).tolist(),
                "component_weights": component_weights.astype(np.float64).tolist(),
            }

        sensitivity, band_labels = self._resolve_band_sensitivity(module3_input)
        raw_component_weights = sensitivity @ band_weights.astype(np.float64)
        component_weights = self._normalize_exact_component_weights(raw_component_weights)
        return component_weights.astype(np.float64), {
            "mode": "band_weighted_sensitivity",
            "target_transform_family": str(transform_family),
            "band_weights": band_weights.astype(np.float64).tolist(),
            "band_labels": list(band_labels),
            "raw_component_weights": raw_component_weights.astype(np.float64).tolist(),
            "component_weights": component_weights.astype(np.float64).tolist(),
        }

    def _resolve_p68_set_rerank_band_component_specs(
        self,
        module3_input: Module3ContinuousInput,
        params: _RuntimeSelectorParams,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        band_weights = np.asarray(
            params.acquisition_p68_set_rerank_band_weights,
            dtype=np.float64,
        ).reshape(-1)
        band_count = len(_DEFAULT_BAND_LABELS)
        if band_weights.shape != (band_count,) or not np.any(band_weights > 0.0):
            return [], {
                "mode": "disabled",
                "band_weights": band_weights.astype(np.float64).tolist()
                if band_weights.shape == (band_count,)
                else [],
            }

        active_indices = [int(index) for index, value in enumerate(band_weights) if float(value) > 0.0]
        specs: list[dict[str, Any]] = []
        transform_family = self._resolve_target_transform_family(module3_input, params)
        if transform_family == "logdiff":
            band_integrals, _, band_labels = self._resolve_exact_logdiff_variance_integrals(
                module3_input
            )
            for band_index in active_indices:
                raw_component_weights = band_integrals[:, band_index]
                component_weights = self._normalize_exact_component_weights(raw_component_weights)
                specs.append(
                    {
                        "band_index": int(band_index),
                        "band_label": str(band_labels[band_index]),
                        "band_weight": float(band_weights[band_index]),
                        "raw_component_weights": raw_component_weights.astype(np.float64),
                        "component_weights": component_weights.astype(np.float64),
                    }
                )
            return specs, {
                "mode": "per_band_exact_logdiff_band_variance",
                "target_transform_family": "logdiff",
                "band_weights": band_weights.astype(np.float64).tolist(),
                "band_labels": list(band_labels),
                "active_band_indices": active_indices,
            }

        sensitivity, band_labels = self._resolve_band_sensitivity(module3_input)
        for band_index in active_indices:
            raw_component_weights = sensitivity[:, band_index]
            component_weights = self._normalize_exact_component_weights(raw_component_weights)
            specs.append(
                {
                    "band_index": int(band_index),
                    "band_label": str(band_labels[band_index]),
                    "band_weight": float(band_weights[band_index]),
                    "raw_component_weights": raw_component_weights.astype(np.float64),
                    "component_weights": component_weights.astype(np.float64),
                }
            )
        return specs, {
            "mode": "per_band_sensitivity",
            "target_transform_family": str(transform_family),
            "band_weights": band_weights.astype(np.float64).tolist(),
            "band_labels": list(band_labels),
            "active_band_indices": active_indices,
        }

    def _build_shared_hull_geometry(self, train_unit_thetas: np.ndarray) -> _SharedHullGeometry:
        unit = np.asarray(train_unit_thetas, dtype=np.float64)
        if unit.ndim != 2:
            raise ValueError("train_unit_thetas must be 2D.")
        train_size, theta_dim = unit.shape
        if train_size < theta_dim + 1:
            raise ValueError(
                "Shared hull Delaunay requires at least theta_dim + 1 training points, "
                f"got {train_size} for theta_dim={theta_dim}."
            )
        try:
            delaunay = Delaunay(unit)
        except QhullError as exc:
            raise RuntimeError("Shared hull Delaunay triangulation failed for current training points.") from exc
        simplices = np.asarray(delaunay.simplices, dtype=np.int64)
        return _SharedHullGeometry(
            simplices=simplices,
            simplex_vertices=unit[simplices].astype(np.float64),
            hull_delaunay=delaunay,
        )

    def _generate_domain_support_points(
        self,
        *,
        theta_dim: int,
        point_count: int,
        scheme: str,
    ) -> np.ndarray:
        if int(point_count) <= 0:
            return np.empty((0, theta_dim), dtype=np.float64)

        center = np.full((theta_dim,), 0.5, dtype=np.float64)
        support_points: list[np.ndarray] = []
        if scheme == "structured_boundary_384":
            active_orders = (1, 2, theta_dim)
            for active_dim_count in active_orders:
                for dims in combinations(range(theta_dim), active_dim_count):
                    for bits in product((0.0, 1.0), repeat=active_dim_count):
                        point = center.copy()
                        for dim, bit in zip(dims, bits):
                            point[int(dim)] = float(bit)
                        support_points.append(point)
        else:
            for dim in range(theta_dim):
                low = center.copy()
                low[dim] = 0.0
                high = center.copy()
                high[dim] = 1.0
                support_points.append(low)
                support_points.append(high)
                if len(support_points) >= int(point_count):
                    break

            if scheme == "axis_corner_hybrid" and len(support_points) < int(point_count):
                total_corners = 1 << theta_dim
                gray_codes = np.asarray([idx ^ (idx >> 1) for idx in range(total_corners)], dtype=np.int64)
                corner_pool = ((gray_codes[:, None] >> np.arange(theta_dim)) & 1).astype(np.float64)
                remaining = int(point_count) - len(support_points)
                take = min(remaining, corner_pool.shape[0])
                if take > 0:
                    corner_indices = np.linspace(0, corner_pool.shape[0] - 1, take, dtype=np.int64)
                    for idx in np.unique(corner_indices):
                        support_points.append(corner_pool[int(idx)].copy())

        unique_points: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for point in support_points:
            key = tuple(np.round(point, decimals=12).tolist())
            if key in seen:
                continue
            seen.add(key)
            unique_points.append(np.asarray(point, dtype=np.float64))
            if len(unique_points) >= int(point_count):
                break
        if not unique_points:
            return np.empty((0, theta_dim), dtype=np.float64)
        return np.vstack(unique_points).astype(np.float64)

    def _select_affinely_independent_neighbors(
        self,
        *,
        support_point: np.ndarray,
        train_unit_thetas: np.ndarray,
        candidate_indices: np.ndarray,
        theta_dim: int,
    ) -> np.ndarray | None:
        selected: list[int] = []
        selected_diffs: list[np.ndarray] = []
        for idx in candidate_indices.tolist():
            diff = np.asarray(train_unit_thetas[int(idx)] - support_point, dtype=np.float64)
            if float(np.linalg.norm(diff)) <= 1.0e-12:
                continue
            trial = np.vstack(selected_diffs + [diff]) if selected_diffs else diff.reshape(1, -1)
            rank_before = len(selected_diffs)
            rank_after = int(np.linalg.matrix_rank(trial, tol=1.0e-10))
            if rank_after <= rank_before:
                continue
            selected.append(int(idx))
            selected_diffs.append(diff)
            if len(selected) >= theta_dim:
                return np.asarray(selected, dtype=np.int64)
        return None

    def _select_affinely_independent_neighbor_sets(
        self,
        *,
        support_point: np.ndarray,
        train_unit_thetas: np.ndarray,
        candidate_indices: np.ndarray,
        fallback_indices: np.ndarray,
        theta_dim: int,
        fanout: int,
    ) -> list[np.ndarray]:
        candidate = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
        fallback = np.asarray(fallback_indices, dtype=np.int64).reshape(-1)
        selected_sets: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()

        for source in (candidate, fallback):
            if source.size == 0:
                continue
            max_shifts = max(source.size, int(fanout) * 4)
            for shift in range(max_shifts):
                rolled = np.roll(source, -int(shift % max(1, source.size)))
                chosen = self._select_affinely_independent_neighbors(
                    support_point=support_point,
                    train_unit_thetas=train_unit_thetas,
                    candidate_indices=rolled,
                    theta_dim=theta_dim,
                )
                if chosen is None:
                    continue
                key = tuple(sorted(int(idx) for idx in chosen.tolist()))
                if key in seen:
                    continue
                seen.add(key)
                selected_sets.append(np.asarray(chosen, dtype=np.int64))
                if len(selected_sets) >= int(fanout):
                    return selected_sets
        return selected_sets

    def _build_domain_collection(
        self,
        *,
        train_unit_thetas: np.ndarray,
        hull_geometry: _SharedHullGeometry,
        params: _RuntimeSelectorParams,
    ) -> _SimplexCollection:
        theta_dim = int(train_unit_thetas.shape[1])
        empty_vertices = np.empty((0, theta_dim + 1, theta_dim), dtype=np.float64)
        empty_ids = np.empty((0,), dtype=np.int64)
        if params.coverage_mode != "hull_domain_hybrid" or params.domain_support_point_count <= 0:
            return _SimplexCollection(name="domain", simplex_vertices=empty_vertices, simplex_ids=empty_ids)

        raw_support_points = self._generate_domain_support_points(
            theta_dim=theta_dim,
            point_count=params.domain_support_point_count,
            scheme=params.domain_support_scheme,
        )
        if raw_support_points.shape[0] == 0:
            return _SimplexCollection(name="domain", simplex_vertices=empty_vertices, simplex_ids=empty_ids)

        inside_mask = hull_geometry.hull_delaunay.find_simplex(raw_support_points) >= 0
        support_points = raw_support_points[~inside_mask]
        if support_points.shape[0] == 0:
            return _SimplexCollection(name="domain", simplex_vertices=empty_vertices, simplex_ids=empty_ids)

        domain_vertices: list[np.ndarray] = []
        domain_ids: list[int] = []
        domain_id_base = int(hull_geometry.simplex_vertices.shape[0])
        for support_idx, support_point in enumerate(support_points):
            distances = np.linalg.norm(train_unit_thetas - support_point[None, :], axis=1)
            order = np.argsort(distances, kind="mergesort")
            near_order = order[: max(params.domain_neighbor_count, theta_dim * params.domain_support_fanout)]
            selected_sets = self._select_affinely_independent_neighbor_sets(
                support_point=support_point,
                train_unit_thetas=train_unit_thetas,
                candidate_indices=near_order,
                fallback_indices=order,
                theta_dim=theta_dim,
                fanout=params.domain_support_fanout,
            )
            if not selected_sets:
                continue
            for fan_idx, selected in enumerate(selected_sets):
                simplex_vertices = np.vstack([support_point[None, :], train_unit_thetas[selected]]).astype(np.float64)
                domain_vertices.append(simplex_vertices)
                domain_ids.append(int(domain_id_base + support_idx * max(1, params.domain_support_fanout) + fan_idx))

        if not domain_vertices:
            return _SimplexCollection(name="domain", simplex_vertices=empty_vertices, simplex_ids=empty_ids)
        return _SimplexCollection(
            name="domain",
            simplex_vertices=np.stack(domain_vertices, axis=0).astype(np.float64),
            simplex_ids=np.asarray(domain_ids, dtype=np.int64),
        )

    def _simplex_circumcenter_barycentric(self, simplex_vertices: np.ndarray) -> np.ndarray | None:
        try:
            anchor = simplex_vertices[0]
            system = 2.0 * (simplex_vertices[1:] - anchor[None, :])
            rhs = np.sum(simplex_vertices[1:] ** 2, axis=1) - np.sum(anchor**2)
            circumcenter = np.linalg.solve(system, rhs)
            barycentric_system = np.vstack(
                [simplex_vertices.T, np.ones((1, simplex_vertices.shape[0]), dtype=np.float64)]
            )
            barycentric_rhs = np.concatenate([circumcenter, np.asarray([1.0], dtype=np.float64)])
            weights = np.linalg.solve(barycentric_system, barycentric_rhs).astype(np.float64)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(weights)):
            return None
        if np.any(weights < -self.circumcenter_tol):
            return None
        if not np.isclose(np.sum(weights), 1.0, atol=self.circumcenter_tol):
            return None
        return _project_to_simplex_numpy(weights)

    def _build_component_evaluator(
        self,
        module3_input: Module3ContinuousInput,
        *,
        component_index: int,
        device: torch.device,
        dtype: torch.dtype,
        variance_floor: float,
    ) -> _TorchPosteriorVarianceEvaluator:
        continuous_state = module3_input.continuous_state
        gp = continuous_state.gp_models[int(component_index)]
        score_std_all = np.asarray(
            continuous_state.metadata.get("pca_score_std", []),
            dtype=np.float64,
        ).reshape(-1)
        if score_std_all.shape[0] <= int(component_index):
            raise ValueError("continuous_state.metadata['pca_score_std'] is missing component scales.")
        output_y_scale = float(np.asarray(getattr(gp, "_y_train_std", 1.0), dtype=np.float64).reshape(-1)[0])
        output_scale_sq = float(score_std_all[int(component_index)] ** 2 * output_y_scale**2)
        return _TorchPosteriorVarianceEvaluator(
            train_unit_thetas=np.asarray(continuous_state.train_unit_thetas, dtype=np.float64),
            cholesky_factor=np.asarray(gp.L_, dtype=np.float64),
            lengthscales=np.asarray(continuous_state.pc_lengthscales[int(component_index)], dtype=np.float64),
            signal_variance=float(continuous_state.pc_signal_variances[int(component_index)]),
            output_scale_sq=output_scale_sq,
            device=device,
            dtype=dtype,
            variance_floor=variance_floor,
        )

    def _build_component_evaluators(
        self,
        module3_input: Module3ContinuousInput,
        *,
        device: torch.device,
        dtype: torch.dtype,
        variance_floor: float,
    ) -> list[_TorchPosteriorVarianceEvaluator]:
        return [
            self._build_component_evaluator(
                module3_input,
                component_index=component_index,
                device=device,
                dtype=dtype,
                variance_floor=variance_floor,
            )
            for component_index in range(len(module3_input.continuous_state.gp_models))
        ]

    def _build_aggregate_evaluator_from_components(
        self,
        component_evaluators: list[_TorchPosteriorVarianceEvaluator],
        *,
        component_weights: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
        variance_floor: float,
    ) -> _TorchAggregatePosteriorVarianceEvaluator:
        return _TorchAggregatePosteriorVarianceEvaluator(
            component_evaluators,
            component_weights=component_weights,
            device=device,
            dtype=dtype,
            variance_floor=variance_floor,
        )

    def _build_aggregate_evaluator(
        self,
        module3_input: Module3ContinuousInput,
        *,
        component_weights: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
        variance_floor: float,
    ) -> _TorchAggregatePosteriorVarianceEvaluator:
        component_evaluators = self._build_component_evaluators(
            module3_input,
            device=device,
            dtype=dtype,
            variance_floor=variance_floor,
        )
        return self._build_aggregate_evaluator_from_components(
            component_evaluators,
            component_weights=component_weights,
            device=device,
            dtype=dtype,
            variance_floor=variance_floor,
        )

    def _evaluate_points(
        self,
        *,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator,
        points: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 2 or query.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        if isinstance(evaluator, _BiasAugmentedPosteriorVarianceEvaluator):
            return evaluator.value_numpy_points(query, chunk_size=int(chunk_size))
        values: list[np.ndarray] = []
        for start in range(0, query.shape[0], int(chunk_size)):
            query_chunk = torch.as_tensor(
                query[start : start + int(chunk_size)],
                device=evaluator.device,
                dtype=evaluator.dtype,
            )
            values.append(evaluator.value(query_chunk).detach().cpu().numpy().astype(np.float64))
        return np.concatenate(values, axis=0).astype(np.float64)

    def _score_simplex_collection(
        self,
        *,
        collection: _SimplexCollection,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator,
        params: _RuntimeSelectorParams,
    ) -> _SimplexScoring:
        simplex_vertices = np.asarray(collection.simplex_vertices, dtype=np.float64)
        simplex_count = int(simplex_vertices.shape[0])
        simplex_size = int(simplex_vertices.shape[1]) if simplex_count else 9
        theta_dim = int(simplex_vertices.shape[2]) if simplex_count else 8
        barycenter_bary = np.full((simplex_count, simplex_size), 1.0 / float(simplex_size), dtype=np.float64)
        circumcenter_bary = np.zeros_like(barycenter_bary)
        circumcenter_valid = np.zeros((simplex_count,), dtype=bool)
        best_vertex_index = np.zeros((simplex_count,), dtype=np.int64)
        facet_center_template = np.full(
            (simplex_size, simplex_size),
            1.0 / float(max(1, simplex_size - 1)),
            dtype=np.float64,
        )
        np.fill_diagonal(facet_center_template, 0.0)
        facet_center_bary = np.broadcast_to(
            facet_center_template[None, :, :],
            (simplex_count, simplex_size, simplex_size),
        ).copy()
        facet_center_scores = np.full((simplex_count, simplex_size), -np.inf, dtype=np.float64)
        best_facet_center_index = np.zeros((simplex_count,), dtype=np.int64)
        if simplex_count == 0:
            return _SimplexScoring(
                collection_name=collection.name,
                simplex_vertices=simplex_vertices,
                simplex_ids=np.asarray(collection.simplex_ids, dtype=np.int64),
                repr_points=np.empty((0, theta_dim), dtype=np.float64),
                repr_scores=np.empty((0,), dtype=np.float64),
                barycenter_bary=barycenter_bary,
                circumcenter_bary=circumcenter_bary,
                circumcenter_valid=circumcenter_valid,
                best_vertex_index=best_vertex_index,
                facet_center_bary=facet_center_bary,
                facet_center_scores=facet_center_scores,
                best_facet_center_index=best_facet_center_index,
            )

        barycenter_points = np.einsum("mij,mi->mj", simplex_vertices, barycenter_bary, optimize=True)
        circumcenter_points = barycenter_points.copy()
        for simplex_idx in range(simplex_count):
            weights = self._simplex_circumcenter_barycentric(simplex_vertices[simplex_idx])
            if weights is None:
                continue
            circumcenter_bary[simplex_idx] = weights
            circumcenter_valid[simplex_idx] = True
            circumcenter_points[simplex_idx] = np.einsum(
                "ij,i->j",
                simplex_vertices[simplex_idx],
                weights,
                optimize=True,
            )
        facet_center_points = np.einsum(
            "mij,fi->mfj",
            simplex_vertices,
            facet_center_template,
            optimize=True,
        )
        edge_pairs = [(i, j) for i in range(simplex_size) for j in range(i + 1, simplex_size)]
        edge_bary = np.zeros((len(edge_pairs), simplex_size), dtype=np.float64)
        for edge_idx, (left, right) in enumerate(edge_pairs):
            edge_bary[edge_idx, int(left)] = 0.5
            edge_bary[edge_idx, int(right)] = 0.5
        edge_points = np.einsum(
            "mij,ei->mej",
            simplex_vertices,
            edge_bary,
            optimize=True,
        )
        qmc_bary = _dirichlet_qmc_barycentric(
            simplex_size=simplex_size,
            sample_count=params.repr_dirichlet_probe_count,
        )
        qmc_points = np.empty((simplex_count, 0, theta_dim), dtype=np.float64)
        if qmc_bary.shape[0] > 0:
            qmc_points = np.einsum(
                "mij,qi->mqj",
                simplex_vertices,
                qmc_bary,
                optimize=True,
            )

        stage0_blocks: list[tuple[str, np.ndarray]] = [
            ("barycenter", barycenter_points.reshape(simplex_count, theta_dim)),
            ("circumcenter", circumcenter_points[circumcenter_valid].reshape(-1, theta_dim)),
            ("facet", facet_center_points.reshape(simplex_count * simplex_size, theta_dim)),
            ("edge", edge_points.reshape(simplex_count * edge_bary.shape[0], theta_dim)),
        ]
        if qmc_points.shape[1] > 0:
            stage0_blocks.append(("qmc", qmc_points.reshape(simplex_count * qmc_points.shape[1], theta_dim)))
        stage0_points = np.vstack([points for _, points in stage0_blocks if points.shape[0] > 0])
        stage0_scores_all = self._evaluate_points(
            evaluator=evaluator,
            points=stage0_points,
            chunk_size=params.stage0_chunk_size,
        )
        cursor = 0
        barycenter_scores = stage0_scores_all[cursor : cursor + simplex_count].astype(np.float64)
        cursor += simplex_count
        circumcenter_scores = np.full((simplex_count,), -np.inf, dtype=np.float64)
        circum_count = int(np.count_nonzero(circumcenter_valid))
        if circum_count > 0:
            circumcenter_scores[circumcenter_valid] = stage0_scores_all[cursor : cursor + circum_count]
            cursor += circum_count
        facet_count = int(simplex_count * simplex_size)
        facet_center_scores = stage0_scores_all[cursor : cursor + facet_count].reshape(simplex_count, simplex_size)
        cursor += facet_count
        best_facet_center_index = np.argmax(facet_center_scores, axis=1).astype(np.int64)
        best_facet_center_scores = facet_center_scores[np.arange(simplex_count), best_facet_center_index]
        best_facet_center_points = facet_center_points[np.arange(simplex_count), best_facet_center_index]
        edge_count = int(simplex_count * edge_bary.shape[0])
        edge_scores = stage0_scores_all[cursor : cursor + edge_count].reshape(simplex_count, edge_bary.shape[0])
        cursor += edge_count
        best_edge_index = np.argmax(edge_scores, axis=1).astype(np.int64)
        best_edge_scores = edge_scores[np.arange(simplex_count), best_edge_index]
        best_edge_points = edge_points[np.arange(simplex_count), best_edge_index]
        best_qmc_scores = np.full((simplex_count,), -np.inf, dtype=np.float64)
        best_qmc_points = barycenter_points.copy()
        if qmc_points.shape[1] > 0:
            qmc_count = int(simplex_count * qmc_points.shape[1])
            qmc_scores = stage0_scores_all[cursor : cursor + qmc_count].reshape(simplex_count, qmc_points.shape[1])
            best_qmc_index = np.argmax(qmc_scores, axis=1).astype(np.int64)
            best_qmc_scores = qmc_scores[np.arange(simplex_count), best_qmc_index]
            best_qmc_points = qmc_points[np.arange(simplex_count), best_qmc_index]
            cursor += qmc_count

        best_vertex_scores = np.full((simplex_count,), -np.inf, dtype=np.float64)
        best_vertex_points = barycenter_points.copy()

        repr_scores = barycenter_scores.astype(np.float64).copy()
        repr_points = barycenter_points.astype(np.float64).copy()

        circum_mask = circumcenter_valid & (circumcenter_scores > repr_scores)
        repr_scores[circum_mask] = circumcenter_scores[circum_mask]
        repr_points[circum_mask] = circumcenter_points[circum_mask]

        facet_mask = best_facet_center_scores > repr_scores
        repr_scores[facet_mask] = best_facet_center_scores[facet_mask]
        repr_points[facet_mask] = best_facet_center_points[facet_mask]

        edge_mask = best_edge_scores > repr_scores
        repr_scores[edge_mask] = best_edge_scores[edge_mask]
        repr_points[edge_mask] = best_edge_points[edge_mask]

        qmc_mask = best_qmc_scores > repr_scores
        repr_scores[qmc_mask] = best_qmc_scores[qmc_mask]
        repr_points[qmc_mask] = best_qmc_points[qmc_mask]

        return _SimplexScoring(
            collection_name=collection.name,
            simplex_vertices=simplex_vertices,
            simplex_ids=np.asarray(collection.simplex_ids, dtype=np.int64),
            repr_points=repr_points.astype(np.float64),
            repr_scores=repr_scores.astype(np.float64),
            barycenter_bary=barycenter_bary.astype(np.float64),
            circumcenter_bary=circumcenter_bary.astype(np.float64),
            circumcenter_valid=circumcenter_valid,
            best_vertex_index=best_vertex_index,
            facet_center_bary=facet_center_bary.astype(np.float64),
            facet_center_scores=facet_center_scores.astype(np.float64),
            best_facet_center_index=best_facet_center_index.astype(np.int64),
        )

    def _apply_collection_score_scale(
        self,
        scoring: _SimplexScoring,
        *,
        params: _RuntimeSelectorParams,
    ) -> _SimplexScoring:
        if (
            scoring.collection_name != "domain"
            or scoring.simplex_vertices.shape[0] == 0
            or np.isclose(float(params.domain_score_scale), 1.0)
        ):
            return scoring
        scaled_scores = np.asarray(scoring.repr_scores, dtype=np.float64) * float(params.domain_score_scale)
        return _SimplexScoring(
            collection_name=scoring.collection_name,
            simplex_vertices=scoring.simplex_vertices.astype(np.float64),
            simplex_ids=scoring.simplex_ids.astype(np.int64),
            repr_points=scoring.repr_points.astype(np.float64),
            repr_scores=scaled_scores.astype(np.float64),
            barycenter_bary=scoring.barycenter_bary.astype(np.float64),
            circumcenter_bary=scoring.circumcenter_bary.astype(np.float64),
            circumcenter_valid=scoring.circumcenter_valid.astype(bool),
            best_vertex_index=scoring.best_vertex_index.astype(np.int64),
            facet_center_bary=scoring.facet_center_bary.astype(np.float64),
            facet_center_scores=scoring.facet_center_scores.astype(np.float64),
            best_facet_center_index=scoring.best_facet_center_index.astype(np.int64),
        )

    def _subset_scoring(self, scoring: _SimplexScoring, max_count: int) -> _SimplexScoring:
        if scoring.simplex_vertices.shape[0] == 0 or int(max_count) <= 0:
            return _SimplexScoring(
                collection_name=scoring.collection_name,
                simplex_vertices=scoring.simplex_vertices[:0].copy(),
                simplex_ids=scoring.simplex_ids[:0].copy(),
                repr_points=scoring.repr_points[:0].copy(),
                repr_scores=scoring.repr_scores[:0].copy(),
                barycenter_bary=scoring.barycenter_bary[:0].copy(),
                circumcenter_bary=scoring.circumcenter_bary[:0].copy(),
                circumcenter_valid=scoring.circumcenter_valid[:0].copy(),
                best_vertex_index=scoring.best_vertex_index[:0].copy(),
                facet_center_bary=scoring.facet_center_bary[:0].copy(),
                facet_center_scores=scoring.facet_center_scores[:0].copy(),
                best_facet_center_index=scoring.best_facet_center_index[:0].copy(),
            )
        order = np.argsort(-scoring.repr_scores, kind="mergesort")
        chosen = order[: min(int(max_count), order.shape[0])]
        return _SimplexScoring(
            collection_name=scoring.collection_name,
            simplex_vertices=scoring.simplex_vertices[chosen].astype(np.float64),
            simplex_ids=scoring.simplex_ids[chosen].astype(np.int64),
            repr_points=scoring.repr_points[chosen].astype(np.float64),
            repr_scores=scoring.repr_scores[chosen].astype(np.float64),
            barycenter_bary=scoring.barycenter_bary[chosen].astype(np.float64),
            circumcenter_bary=scoring.circumcenter_bary[chosen].astype(np.float64),
            circumcenter_valid=scoring.circumcenter_valid[chosen].astype(bool),
            best_vertex_index=scoring.best_vertex_index[chosen].astype(np.int64),
            facet_center_bary=scoring.facet_center_bary[chosen].astype(np.float64),
            facet_center_scores=scoring.facet_center_scores[chosen].astype(np.float64),
            best_facet_center_index=scoring.best_facet_center_index[chosen].astype(np.int64),
        )

    def _merge_scoring(self, scorings: Sequence[_SimplexScoring]) -> _SimplexScoring:
        active = [scoring for scoring in scorings if scoring.simplex_vertices.shape[0] > 0]
        if not active:
            return _SimplexScoring(
                collection_name="merged",
                simplex_vertices=np.empty((0, 9, 8), dtype=np.float64),
                simplex_ids=np.empty((0,), dtype=np.int64),
                repr_points=np.empty((0, 8), dtype=np.float64),
                repr_scores=np.empty((0,), dtype=np.float64),
                barycenter_bary=np.empty((0, 9), dtype=np.float64),
                circumcenter_bary=np.empty((0, 9), dtype=np.float64),
                circumcenter_valid=np.empty((0,), dtype=bool),
                best_vertex_index=np.empty((0,), dtype=np.int64),
                facet_center_bary=np.empty((0, 9, 9), dtype=np.float64),
                facet_center_scores=np.empty((0, 9), dtype=np.float64),
                best_facet_center_index=np.empty((0,), dtype=np.int64),
            )
        return _SimplexScoring(
            collection_name="merged",
            simplex_vertices=np.concatenate([item.simplex_vertices for item in active], axis=0).astype(np.float64),
            simplex_ids=np.concatenate([item.simplex_ids for item in active], axis=0).astype(np.int64),
            repr_points=np.concatenate([item.repr_points for item in active], axis=0).astype(np.float64),
            repr_scores=np.concatenate([item.repr_scores for item in active], axis=0).astype(np.float64),
            barycenter_bary=np.concatenate([item.barycenter_bary for item in active], axis=0).astype(np.float64),
            circumcenter_bary=np.concatenate([item.circumcenter_bary for item in active], axis=0).astype(np.float64),
            circumcenter_valid=np.concatenate([item.circumcenter_valid for item in active], axis=0).astype(bool),
            best_vertex_index=np.concatenate([item.best_vertex_index for item in active], axis=0).astype(np.int64),
            facet_center_bary=np.concatenate([item.facet_center_bary for item in active], axis=0).astype(np.float64),
            facet_center_scores=np.concatenate([item.facet_center_scores for item in active], axis=0).astype(np.float64),
            best_facet_center_index=np.concatenate([item.best_facet_center_index for item in active], axis=0).astype(np.int64),
        )

    def _point_to_simplex_barycentric(
        self,
        simplex_vertices: np.ndarray,
        point: np.ndarray,
    ) -> np.ndarray:
        vertices = np.asarray(simplex_vertices, dtype=np.float64)
        target = np.asarray(point, dtype=np.float64).reshape(-1)
        anchor = vertices[0]
        basis = (vertices[1:] - anchor).T
        rhs = target - anchor
        try:
            coeff = np.linalg.solve(basis, rhs)
        except np.linalg.LinAlgError:
            coeff, *_ = np.linalg.lstsq(basis, rhs, rcond=None)
        bary = np.empty((vertices.shape[0],), dtype=np.float64)
        bary[1:] = coeff.reshape(-1)
        bary[0] = 1.0 - float(np.sum(coeff, dtype=np.float64))
        return _project_to_simplex_numpy(bary)

    def _build_legacy_refinement_starts(
        self,
        *,
        scoring: _SimplexScoring,
        params: _RuntimeSelectorParams,
        random_seed: int,
    ) -> np.ndarray:
        simplex_count = int(scoring.simplex_vertices.shape[0])
        simplex_size = int(scoring.simplex_vertices.shape[1]) if simplex_count else 9
        starts_per_simplex = int(params.starts_per_simplex_refine)
        starts = np.empty((simplex_count, starts_per_simplex, simplex_size), dtype=np.float64)
        rng = np.random.default_rng(int(random_seed))

        barycenter = np.full((simplex_size,), 1.0 / float(simplex_size), dtype=np.float64)
        for simplex_idx in range(simplex_count):
            starts[simplex_idx, 0, :] = barycenter
            next_start = 1
            if next_start < starts_per_simplex:
                if scoring.circumcenter_valid[simplex_idx]:
                    starts[simplex_idx, next_start, :] = scoring.circumcenter_bary[simplex_idx]
                else:
                    perturb = rng.standard_normal(simplex_size).astype(np.float64)
                    perturb -= np.mean(perturb, dtype=np.float64)
                    starts[simplex_idx, next_start, :] = _project_to_simplex_numpy(
                        barycenter + params.perturbation_eps_refine * perturb
                    )
                next_start += 1
            if next_start < starts_per_simplex and scoring.facet_center_scores.shape[0] > 0:
                facet_order = np.argsort(
                    -scoring.facet_center_scores[simplex_idx],
                    kind="mergesort",
                )
                for facet_idx in facet_order.tolist():
                    if next_start >= starts_per_simplex:
                        break
                    starts[simplex_idx, next_start, :] = scoring.facet_center_bary[simplex_idx, int(facet_idx)]
                    next_start += 1
            while next_start < starts_per_simplex:
                perturb = rng.standard_normal(simplex_size).astype(np.float64)
                perturb -= np.mean(perturb, dtype=np.float64)
                scale = params.perturbation_eps_refine * (1.0 + 0.15 * float(max(0, next_start - 2)))
                starts[simplex_idx, next_start, :] = _project_to_simplex_numpy(barycenter + scale * perturb)
                next_start += 1
        return starts.astype(np.float64)

    def _build_hierarchical_refinement_starts(
        self,
        *,
        scoring: _SimplexScoring,
        starts_per_simplex: int,
        params: _RuntimeSelectorParams,
        random_seed: int,
        stage_mode: str,
    ) -> np.ndarray:
        simplex_count = int(scoring.simplex_vertices.shape[0])
        simplex_size = int(scoring.simplex_vertices.shape[1]) if simplex_count else 9
        starts = np.empty((simplex_count, starts_per_simplex, simplex_size), dtype=np.float64)
        rng = np.random.default_rng(int(random_seed))
        barycenter = np.full((simplex_size,), 1.0 / float(simplex_size), dtype=np.float64)
        facet_order = np.argsort(
            -np.asarray(scoring.facet_center_scores, dtype=np.float64),
            axis=1,
            kind="mergesort",
        )
        repr_bary = np.asarray(
            [
                self._point_to_simplex_barycentric(
                    scoring.simplex_vertices[simplex_idx],
                    scoring.repr_points[simplex_idx],
                )
                for simplex_idx in range(simplex_count)
            ],
            dtype=np.float64,
        )
        for simplex_idx in range(simplex_count):
            start_list: list[np.ndarray] = [repr_bary[simplex_idx]]
            if stage_mode == "hierarchical_stage2":
                start_list.append(scoring.barycenter_bary[simplex_idx])
            else:
                start_list.append(barycenter)
            if scoring.circumcenter_valid[simplex_idx]:
                start_list.append(scoring.circumcenter_bary[simplex_idx])
            for facet_idx in facet_order[simplex_idx].tolist():
                if len(start_list) >= starts_per_simplex:
                    break
                start_list.append(scoring.facet_center_bary[simplex_idx, int(facet_idx)])
            perturb_scale = (
                float(params.perturbation_eps_refine) * 0.25
                if stage_mode == "hierarchical_stage2"
                else float(params.perturbation_eps_refine) * 0.5
            )
            while len(start_list) < starts_per_simplex:
                perturb = rng.standard_normal(simplex_size).astype(np.float64)
                perturb -= np.mean(perturb, dtype=np.float64)
                center = (
                    repr_bary[simplex_idx]
                    if stage_mode == "hierarchical_stage2"
                    else 0.5 * (repr_bary[simplex_idx] + barycenter)
                )
                start_list.append(_project_to_simplex_numpy(center + perturb_scale * perturb))
            starts[simplex_idx] = np.asarray(start_list[:starts_per_simplex], dtype=np.float64)
        return starts.astype(np.float64)

    def _build_refinement_starts(
        self,
        *,
        scoring: _SimplexScoring,
        params: _RuntimeSelectorParams,
        random_seed: int,
        start_strategy: str = "legacy",
    ) -> np.ndarray:
        if start_strategy == "legacy":
            return self._build_legacy_refinement_starts(
                scoring=scoring,
                params=params,
                random_seed=random_seed,
            )
        if start_strategy in {"hierarchical_stage1", "hierarchical_stage2"}:
            return self._build_hierarchical_refinement_starts(
                scoring=scoring,
                starts_per_simplex=int(params.starts_per_simplex_refine),
                params=params,
                random_seed=random_seed,
                stage_mode=start_strategy,
            )
        raise ValueError(f"Unsupported start_strategy={start_strategy!r}.")

    def _ranking_from_scoring(self, scoring: _SimplexScoring, *, component_index: int = -1) -> _ComponentCandidateRanking:
        if scoring.simplex_vertices.shape[0] == 0:
            theta_dim = 8
            return _ComponentCandidateRanking(
                component_index=int(component_index),
                unit_points=np.empty((0, theta_dim), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float64),
                simplex_ids=np.empty((0,), dtype=np.int64),
            )
        order = np.argsort(-scoring.repr_scores, kind="mergesort")
        return _ComponentCandidateRanking(
            component_index=int(component_index),
            unit_points=scoring.repr_points[order].astype(np.float64),
            scores=scoring.repr_scores[order].astype(np.float64),
            simplex_ids=scoring.simplex_ids[order].astype(np.int64),
        )

    def _merge_rankings(self, rankings: Sequence[_ComponentCandidateRanking]) -> _ComponentCandidateRanking:
        active = [ranking for ranking in rankings if ranking.unit_points.shape[0] > 0]
        if not active:
            return _ComponentCandidateRanking(
                component_index=-1,
                unit_points=np.empty((0, 8), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float64),
                simplex_ids=np.empty((0,), dtype=np.int64),
            )
        points = np.concatenate([ranking.unit_points for ranking in active], axis=0).astype(np.float64)
        scores = np.concatenate([ranking.scores for ranking in active], axis=0).astype(np.float64)
        simplex_ids = np.concatenate([ranking.simplex_ids for ranking in active], axis=0).astype(np.int64)
        order = np.argsort(-scores, kind="mergesort")
        return _ComponentCandidateRanking(
            component_index=-1,
            unit_points=points[order],
            scores=scores[order],
            simplex_ids=simplex_ids[order],
        )

    def _apply_domain_score_scale_to_ranking(
        self,
        ranking: _ComponentCandidateRanking,
        *,
        domain_simplex_id_start: int,
        params: _RuntimeSelectorParams,
    ) -> _ComponentCandidateRanking:
        if ranking.unit_points.shape[0] == 0 or np.isclose(float(params.domain_score_scale), 1.0):
            return ranking
        scaled_scores = np.asarray(ranking.scores, dtype=np.float64).copy()
        simplex_ids = np.asarray(ranking.simplex_ids, dtype=np.int64)
        domain_mask = simplex_ids >= int(domain_simplex_id_start)
        if not np.any(domain_mask):
            return ranking
        scaled_scores[domain_mask] *= float(params.domain_score_scale)
        order = np.argsort(-scaled_scores, kind="mergesort")
        return _ComponentCandidateRanking(
            component_index=int(ranking.component_index),
            unit_points=np.asarray(ranking.unit_points, dtype=np.float64)[order].astype(np.float64),
            scores=scaled_scores[order].astype(np.float64),
            simplex_ids=simplex_ids[order].astype(np.int64),
        )

    def _concat_rankings(self, rankings: Sequence[_ComponentCandidateRanking]) -> _ComponentCandidateRanking:
        active = [ranking for ranking in rankings if ranking.unit_points.shape[0] > 0]
        if not active:
            return _ComponentCandidateRanking(
                component_index=-1,
                unit_points=np.empty((0, 8), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float64),
                simplex_ids=np.empty((0,), dtype=np.int64),
            )
        component_indices = {int(ranking.component_index) for ranking in active}
        points = np.concatenate([ranking.unit_points for ranking in active], axis=0).astype(np.float64)
        scores = np.concatenate([ranking.scores for ranking in active], axis=0).astype(np.float64)
        simplex_ids = np.concatenate([ranking.simplex_ids for ranking in active], axis=0).astype(np.int64)
        order = np.argsort(-scores, kind="mergesort")
        return _ComponentCandidateRanking(
            component_index=component_indices.pop() if len(component_indices) == 1 else -1,
            unit_points=points[order],
            scores=scores[order],
            simplex_ids=simplex_ids[order],
        )

    def _build_stage2_simplex_ids(
        self,
        *,
        stage1_ranking: _ComponentCandidateRanking,
        merged_stage0: _SimplexScoring,
        params: _RuntimeSelectorParams,
    ) -> np.ndarray:
        stage2_budget = int(params.hierarchical_stage2_top_k)
        if stage2_budget <= 0:
            return np.empty((0,), dtype=np.int64)

        ordered_ids: list[int] = []
        seen_ids: set[int] = set()
        candidate_sources = (
            np.asarray(
                stage1_ranking.simplex_ids[: min(stage2_budget, stage1_ranking.simplex_ids.shape[0])],
                dtype=np.int64,
            ),
            np.asarray(
                merged_stage0.simplex_ids[: min(stage2_budget, merged_stage0.simplex_ids.shape[0])],
                dtype=np.int64,
            ),
        )
        for candidate_ids in candidate_sources:
            for simplex_id in candidate_ids.tolist():
                simplex_id_int = int(simplex_id)
                if simplex_id_int in seen_ids:
                    continue
                seen_ids.add(simplex_id_int)
                ordered_ids.append(simplex_id_int)
        return np.asarray(ordered_ids, dtype=np.int64)

    def _rerank_ranking_prefix_exact(
        self,
        *,
        ranking: _ComponentCandidateRanking,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator,
        top_k: int,
    ) -> _ComponentCandidateRanking:
        if ranking.unit_points.shape[0] == 0 or int(top_k) <= 0:
            return ranking
        prefix_count = min(int(top_k), int(ranking.unit_points.shape[0]))
        prefix_points = np.asarray(ranking.unit_points[:prefix_count], dtype=np.float64)
        exact_scores = self._evaluate_points(
            evaluator=evaluator,
            points=prefix_points,
            chunk_size=1,
        )
        order = np.argsort(-exact_scores, kind="mergesort")
        return _ComponentCandidateRanking(
            component_index=int(ranking.component_index),
            unit_points=np.concatenate(
                [
                    prefix_points[order].astype(np.float64),
                    np.asarray(ranking.unit_points[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            scores=np.concatenate(
                [
                    exact_scores[order].astype(np.float64),
                    np.asarray(ranking.scores[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            simplex_ids=np.concatenate(
                [
                    np.asarray(ranking.simplex_ids[:prefix_count], dtype=np.int64)[order].astype(np.int64),
                    np.asarray(ranking.simplex_ids[prefix_count:], dtype=np.int64),
                ],
                axis=0,
            ),
        )

    def _generate_imse_probe_points(
        self,
        *,
        theta_dim: int,
        probe_count: int,
        random_seed: int,
    ) -> np.ndarray:
        if int(probe_count) <= 0:
            return np.empty((0, theta_dim), dtype=np.float64)
        sampler = qmc.Sobol(d=int(theta_dim), scramble=True, seed=int(random_seed))
        power = int(np.ceil(np.log2(max(1, int(probe_count)))))
        return np.asarray(sampler.random_base2(power), dtype=np.float64)[: int(probe_count)]

    def _rerank_ranking_prefix_imse(
        self,
        *,
        ranking: _ComponentCandidateRanking,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator | _DensityWeightedPosteriorVarianceEvaluator,
        params: _RuntimeSelectorParams,
        random_seed: int,
    ) -> _ComponentCandidateRanking:
        if (
            ranking.unit_points.shape[0] == 0
            or int(params.imse_rerank_top_k) <= 0
            or int(params.imse_probe_count) <= 0
        ):
            return ranking
        prefix_count = min(int(params.imse_rerank_top_k), int(ranking.unit_points.shape[0]))
        prefix_points = np.asarray(ranking.unit_points[:prefix_count], dtype=np.float64)
        theta_dim = int(prefix_points.shape[1])
        probe_points = self._generate_imse_probe_points(
            theta_dim=theta_dim,
            probe_count=int(params.imse_probe_count),
            random_seed=int(random_seed),
        )
        candidate_tensor = torch.as_tensor(prefix_points, device=evaluator.device, dtype=evaluator.dtype)
        probe_tensor = torch.as_tensor(probe_points, device=evaluator.device, dtype=evaluator.dtype)
        with torch.no_grad():
            imse_rerank_mode = str(params.imse_rerank_mode)
            if imse_rerank_mode == "p68_proxy":
                current_variance = evaluator.value(probe_tensor)
                reduction_matrix = evaluator.variance_reduction_matrix(
                    candidate_tensor,
                    probe_tensor,
                    chunk_size=max(1, int(params.chunk_size)),
                )
                imse_scores_tensor = _quantile_proxy_reduction_scores(
                    current_variance,
                    reduction_matrix,
                    variance_floor=float(params.variance_floor),
                    quantile=float(params.imse_quantile),
                    mean_weight=float(params.imse_quantile_mean_weight),
                    max_weight=float(params.imse_quantile_max_weight),
                )
            elif imse_rerank_mode == "p68_shell":
                current_variance = evaluator.value(probe_tensor)
                reduction_matrix = evaluator.variance_reduction_matrix(
                    candidate_tensor,
                    probe_tensor,
                    chunk_size=max(1, int(params.chunk_size)),
                )
                imse_scores_tensor = _quantile_shell_reduction_scores(
                    current_variance,
                    reduction_matrix,
                    variance_floor=float(params.variance_floor),
                    quantile=float(params.imse_quantile),
                    shell_width=float(params.imse_quantile_shell_width),
                    mean_weight=float(params.imse_quantile_mean_weight),
                    max_weight=float(params.imse_quantile_max_weight),
                )
            elif imse_rerank_mode == "p68_soft":
                current_variance = evaluator.value(probe_tensor)
                reduction_matrix = evaluator.variance_reduction_matrix(
                    candidate_tensor,
                    probe_tensor,
                    chunk_size=max(1, int(params.chunk_size)),
                )
                imse_scores_tensor = _soft_quantile_reduction_scores(
                    current_variance,
                    reduction_matrix,
                    variance_floor=float(params.variance_floor),
                    quantile=float(params.imse_quantile),
                    bandwidth=float(params.imse_quantile_shell_width),
                    mean_weight=float(params.imse_quantile_mean_weight),
                    max_weight=float(params.imse_quantile_max_weight),
                )
            else:
                imse_scores_tensor = evaluator.imse_reduction(
                    candidate_tensor,
                    probe_tensor,
                    chunk_size=max(1, int(params.chunk_size)),
                )
            imse_scores = imse_scores_tensor.detach().cpu().numpy().astype(np.float64)
        order = np.argsort(-imse_scores, kind="mergesort")
        return _ComponentCandidateRanking(
            component_index=int(ranking.component_index),
            unit_points=np.concatenate(
                [
                    prefix_points[order].astype(np.float64),
                    np.asarray(ranking.unit_points[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            scores=np.concatenate(
                [
                    imse_scores[order].astype(np.float64),
                    np.asarray(ranking.scores[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            simplex_ids=np.concatenate(
                [
                    np.asarray(ranking.simplex_ids[:prefix_count], dtype=np.int64)[order].astype(np.int64),
                    np.asarray(ranking.simplex_ids[prefix_count:], dtype=np.int64),
                ],
                axis=0,
            ),
        )

    @staticmethod
    def _spacefill_discrepancy_gain(
        *,
        prefix_points: np.ndarray,
        train_unit_thetas: np.ndarray,
    ) -> np.ndarray:
        prefix = np.clip(np.asarray(prefix_points, dtype=np.float64), 0.0, 1.0)
        train_points = np.clip(np.asarray(train_unit_thetas, dtype=np.float64), 0.0, 1.0)
        if train_points.ndim != 2 or train_points.shape[0] == 0 or prefix.shape[0] == 0:
            return np.zeros((prefix.shape[0],), dtype=np.float64)
        base_discrepancy = float(qmc.discrepancy(train_points, method="CD"))
        discrepancy_gain = np.empty((prefix.shape[0],), dtype=np.float64)
        for row_idx, point in enumerate(prefix):
            candidate_design = np.vstack([train_points, point.reshape(1, -1)])
            discrepancy_gain[row_idx] = base_discrepancy - float(
                qmc.discrepancy(candidate_design, method="CD")
            )
        return discrepancy_gain

    @staticmethod
    def _unit_scale_values(values: np.ndarray) -> np.ndarray:
        finite = np.asarray(values, dtype=np.float64)
        if finite.shape[0] == 0 or not np.all(np.isfinite(finite)):
            return np.zeros_like(finite, dtype=np.float64)
        min_value = float(np.min(finite))
        span = float(np.max(finite) - min_value)
        if span <= 1.0e-30:
            return np.zeros_like(finite, dtype=np.float64)
        return (finite - min_value) / span

    @staticmethod
    def _boundary_budget_penalty(
        *,
        prefix_points: np.ndarray,
        train_unit_thetas: np.ndarray,
        threshold: float,
        target_fraction: float,
    ) -> np.ndarray:
        prefix = np.clip(np.asarray(prefix_points, dtype=np.float64), 0.0, 1.0)
        train_points = np.clip(np.asarray(train_unit_thetas, dtype=np.float64), 0.0, 1.0)
        if prefix.ndim != 2 or prefix.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        theta_dim = int(prefix.shape[1])
        if theta_dim <= 0:
            return np.zeros((prefix.shape[0],), dtype=np.float64)
        threshold = float(min(0.5, max(0.0, threshold)))
        target_fraction = float(min(1.0, max(0.0, target_fraction)))
        prefix_boundary = np.minimum(prefix, 1.0 - prefix)
        candidate_boundary_counts = np.sum(prefix_boundary < threshold, axis=1).astype(np.float64)
        candidate_boundary_fraction = candidate_boundary_counts / float(theta_dim)
        if train_points.ndim == 2 and train_points.shape[0] > 0:
            train_boundary = np.minimum(train_points, 1.0 - train_points)
            current_boundary_count = float(np.sum(train_boundary < threshold))
            current_coordinate_count = float(train_boundary.size)
        else:
            current_boundary_count = 0.0
            current_coordinate_count = 0.0
        resulting_fraction = (
            current_boundary_count + candidate_boundary_counts
        ) / max(1.0, current_coordinate_count + float(theta_dim))
        budget_excess = np.maximum(0.0, resulting_fraction - target_fraction)
        return (budget_excess + 0.10 * candidate_boundary_fraction).astype(np.float64)

    @staticmethod
    def _residual_guard_prefix_order(
        *,
        acq_scores: np.ndarray,
        local_residual_risk: np.ndarray,
        score_ratio: float,
        reject_quantile: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        scores = np.asarray(acq_scores, dtype=np.float64).reshape(-1)
        risk = np.asarray(local_residual_risk, dtype=np.float64).reshape(-1)
        prefix_count = int(scores.shape[0])
        if prefix_count == 0 or risk.shape[0] != prefix_count:
            return np.arange(prefix_count, dtype=np.int64), {"enabled": False}
        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(risk)):
            return np.arange(prefix_count, dtype=np.int64), {"enabled": False}
        top_score = float(np.max(scores))
        if not np.isfinite(top_score) or top_score <= 0.0:
            return np.arange(prefix_count, dtype=np.int64), {"enabled": False}
        score_ratio = float(min(1.0, max(0.0, score_ratio)))
        reject_quantile = float(min(1.0, max(0.0, reject_quantile)))
        if score_ratio <= 0.0 or reject_quantile <= 0.0:
            return np.arange(prefix_count, dtype=np.int64), {"enabled": False}
        eligible_mask = scores >= top_score * score_ratio
        eligible_indices = np.arange(prefix_count, dtype=np.int64)[eligible_mask]
        if int(eligible_indices.shape[0]) <= 1:
            return np.arange(prefix_count, dtype=np.int64), {
                "enabled": False,
                "eligible_count": int(eligible_indices.shape[0]),
            }
        eligible_risk = risk[eligible_indices]
        threshold = float(np.quantile(eligible_risk, reject_quantile))
        keep_mask = eligible_risk >= threshold
        if not bool(np.any(keep_mask)):
            keep_mask[int(np.argmax(eligible_risk))] = True
        keep_indices = eligible_indices[keep_mask]
        reject_indices = eligible_indices[~keep_mask]
        eligible_set = {int(index) for index in eligible_indices.tolist()}
        remainder = np.asarray(
            [int(index) for index in range(prefix_count) if int(index) not in eligible_set],
            dtype=np.int64,
        )
        order = np.concatenate(
            [
                keep_indices.astype(np.int64),
                reject_indices.astype(np.int64),
                remainder.astype(np.int64),
            ],
            axis=0,
        )
        return order.astype(np.int64), {
            "enabled": True,
            "eligible_count": int(eligible_indices.shape[0]),
            "kept_count": int(keep_indices.shape[0]),
            "rejected_count": int(reject_indices.shape[0]),
            "risk_threshold": threshold,
            "risk_min": float(np.min(eligible_risk)),
            "risk_max": float(np.max(eligible_risk)),
        }

    @staticmethod
    def _component_loo_abs_residual(
        gp: Any,
        *,
        score_scale: float,
    ) -> np.ndarray:
        cholesky = np.asarray(gp.L_, dtype=np.float64)
        inv_l = np.linalg.solve(cholesky, np.eye(cholesky.shape[0], dtype=np.float64))
        diag_inv = np.sum(inv_l * inv_l, axis=0)
        alpha = np.asarray(gp.alpha_, dtype=np.float64).reshape(-1)
        output_y_scale = float(np.asarray(getattr(gp, "_y_train_std", 1.0), dtype=np.float64).reshape(-1)[0])
        residual = alpha / np.maximum(diag_inv, 1.0e-300)
        return np.abs(residual) * float(max(score_scale, 1.0e-300)) * output_y_scale

    def _compute_loo_residual_training_risk(
        self,
        module3_input: Module3ContinuousInput,
        *,
        component_weights: np.ndarray,
    ) -> np.ndarray:
        continuous_state = module3_input.continuous_state
        gp_models = list(continuous_state.gp_models)
        weights = np.asarray(component_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != len(gp_models):
            raise ValueError(
                "LOO residual guard component weights must align with GP components, "
                f"got {weights.shape[0]} vs {len(gp_models)}."
            )
        score_std = np.asarray(
            continuous_state.metadata.get("pca_score_std", []),
            dtype=np.float64,
        ).reshape(-1)
        if score_std.shape[0] != len(gp_models):
            raise ValueError("continuous_state.metadata['pca_score_std'] is missing component scales.")
        total = np.zeros((np.asarray(continuous_state.train_unit_thetas).shape[0],), dtype=np.float64)
        for component_index, gp in enumerate(gp_models):
            total += float(weights[component_index]) * self._component_loo_abs_residual(
                gp,
                score_scale=float(score_std[component_index]),
            )
        return np.maximum(total, 0.0).astype(np.float64)

    @staticmethod
    def _local_rbf_residual_risk(
        *,
        query_points: np.ndarray,
        train_unit_thetas: np.ndarray,
        train_risk: np.ndarray,
        bandwidth: float,
    ) -> np.ndarray:
        query = np.asarray(query_points, dtype=np.float64)
        train = np.asarray(train_unit_thetas, dtype=np.float64)
        risk = np.asarray(train_risk, dtype=np.float64).reshape(-1)
        if query.ndim != 2 or train.ndim != 2 or query.shape[0] == 0 or train.shape[0] == 0:
            return np.zeros((query.shape[0] if query.ndim == 2 else 0,), dtype=np.float64)
        diff = query[:, None, :] - train[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        scale_sq = max(float(bandwidth) ** 2, 1.0e-12)
        weights = np.exp(-0.5 * dist_sq / scale_sq)
        return (weights @ risk) / np.maximum(np.sum(weights, axis=1), 1.0e-300)

    def _guard_ranking_prefix_loo_residual(
        self,
        *,
        ranking: _ComponentCandidateRanking,
        module3_input: Module3ContinuousInput,
        fallback_component_weights: np.ndarray,
        train_unit_thetas: np.ndarray,
        params: _RuntimeSelectorParams,
    ) -> tuple[_ComponentCandidateRanking, dict[str, Any]]:
        if (
            ranking.unit_points.shape[0] == 0
            or int(params.acquisition_p68_loo_guard_top_k) <= 0
            or float(params.acquisition_p68_loo_guard_score_ratio) <= 0.0
            or float(params.acquisition_p68_loo_guard_reject_quantile) <= 0.0
        ):
            return ranking, {"enabled": False}
        prefix_count = min(
            int(params.acquisition_p68_loo_guard_top_k),
            int(ranking.unit_points.shape[0]),
        )
        prefix_points = np.clip(np.asarray(ranking.unit_points[:prefix_count], dtype=np.float64), 0.0, 1.0)
        band_weights = np.asarray(params.acquisition_p68_loo_guard_band_weights, dtype=np.float64).reshape(-1)
        component_weights = np.asarray(fallback_component_weights, dtype=np.float64).reshape(-1)
        weight_details: dict[str, Any] = {"mode": "fallback_objective_component_weights"}
        if band_weights.shape == (len(_DEFAULT_BAND_LABELS),) and float(np.sum(band_weights)) > 0.0:
            p68_weight_params = replace(
                params,
                acquisition_p68_set_rerank_band_weights=tuple(float(value) for value in band_weights),
            )
            resolved_weights, resolved_details = self._resolve_p68_set_rerank_risk_component_weights(
                module3_input,
                p68_weight_params,
            )
            if resolved_weights is not None:
                component_weights = np.asarray(resolved_weights, dtype=np.float64)
                weight_details = dict(resolved_details)
        train_risk = self._compute_loo_residual_training_risk(
            module3_input,
            component_weights=component_weights,
        )
        local_risk = self._local_rbf_residual_risk(
            query_points=prefix_points,
            train_unit_thetas=train_unit_thetas,
            train_risk=train_risk,
            bandwidth=float(params.acquisition_p68_loo_guard_bandwidth),
        )
        order, details = self._residual_guard_prefix_order(
            acq_scores=np.asarray(ranking.scores[:prefix_count], dtype=np.float64),
            local_residual_risk=local_risk,
            score_ratio=float(params.acquisition_p68_loo_guard_score_ratio),
            reject_quantile=float(params.acquisition_p68_loo_guard_reject_quantile),
        )
        if not bool(details.get("enabled")):
            return ranking, details
        guarded = _ComponentCandidateRanking(
            component_index=int(ranking.component_index),
            unit_points=np.concatenate(
                [
                    prefix_points[order].astype(np.float64),
                    np.asarray(ranking.unit_points[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            scores=np.concatenate(
                [
                    np.asarray(ranking.scores[:prefix_count], dtype=np.float64)[order].astype(np.float64),
                    np.asarray(ranking.scores[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            simplex_ids=np.concatenate(
                [
                    np.asarray(ranking.simplex_ids[:prefix_count], dtype=np.int64)[order].astype(np.int64),
                    np.asarray(ranking.simplex_ids[prefix_count:], dtype=np.int64),
                ],
                axis=0,
            ),
        )
        return guarded, {
            **details,
            "top_k": int(prefix_count),
            "score_ratio": float(params.acquisition_p68_loo_guard_score_ratio),
            "reject_quantile": float(params.acquisition_p68_loo_guard_reject_quantile),
            "bandwidth": float(params.acquisition_p68_loo_guard_bandwidth),
            "weight_details": dict(weight_details),
        }

    def _rerank_ranking_prefix_p68_set_aware(
        self,
        *,
        ranking: _ComponentCandidateRanking,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator | _DensityWeightedPosteriorVarianceEvaluator,
        band_risk_evaluators: Sequence[dict[str, Any]] | None = None,
        module3_input: Module3ContinuousInput,
        train_unit_thetas: np.ndarray,
        params: _RuntimeSelectorParams,
        random_seed: int,
    ) -> _ComponentCandidateRanking:
        if (
            ranking.unit_points.shape[0] == 0
            or int(params.acquisition_p68_set_rerank_top_k) <= 0
            or float(params.acquisition_p68_set_rerank_score_ratio) <= 0.0
        ):
            return ranking
        prefix_count = min(
            int(params.acquisition_p68_set_rerank_top_k),
            int(ranking.unit_points.shape[0]),
        )
        prefix_points = np.clip(np.asarray(ranking.unit_points[:prefix_count], dtype=np.float64), 0.0, 1.0)
        acq_scores = np.asarray(ranking.scores[:prefix_count], dtype=np.float64)
        if acq_scores.shape[0] == 0 or not np.all(np.isfinite(acq_scores)):
            return ranking
        top_score = float(np.max(acq_scores))
        if not np.isfinite(top_score) or top_score <= 0.0:
            return ranking
        score_ratio = float(
            min(1.0, max(0.0, params.acquisition_p68_set_rerank_score_ratio))
        )
        eligible_mask = acq_scores >= top_score * score_ratio
        if int(np.count_nonzero(eligible_mask)) <= 1:
            return ranking
        acq_weight = float(max(0.0, params.acquisition_p68_set_rerank_acq_weight))
        p68_weight = float(max(0.0, params.acquisition_p68_set_rerank_p68_weight))
        spacefill_weight = float(max(0.0, params.acquisition_p68_set_rerank_spacefill_weight))
        boundary_weight = float(max(0.0, params.acquisition_p68_set_rerank_boundary_weight))
        if acq_weight <= 0.0 and p68_weight <= 0.0 and spacefill_weight <= 0.0 and boundary_weight <= 0.0:
            return ranking

        eligible_indices = np.arange(prefix_count, dtype=np.int64)[eligible_mask]
        eligible_points = prefix_points[eligible_indices]
        composite = np.zeros((eligible_points.shape[0],), dtype=np.float64)

        if acq_weight > 0.0:
            composite += acq_weight * self._unit_scale_values(acq_scores[eligible_indices])

        if p68_weight > 0.0:
            theta_dim = int(eligible_points.shape[1])
            risk_mode = str(params.acquisition_p68_set_rerank_risk_mode)
            probe_weights: np.ndarray | None = None
            if risk_mode == "validation_probe_shell":
                probe_payload = dict(
                    module3_input.metadata.get(
                        "p68_validation_probe_shell",
                        module3_input.continuous_state.metadata.get("p68_validation_probe_shell", {}),
                    )
                )
                if bool(probe_payload.get("enabled")):
                    probe_points = np.asarray(
                        probe_payload.get("probe_unit_thetas", []),
                        dtype=np.float64,
                    )
                    probe_weights = np.asarray(
                        probe_payload.get("probe_weights", []),
                        dtype=np.float64,
                    ).reshape(-1)
                else:
                    probe_points = np.empty((0, theta_dim), dtype=np.float64)
            elif int(params.imse_probe_count) > 0:
                probe_points = self._generate_imse_probe_points(
                    theta_dim=theta_dim,
                    probe_count=int(params.imse_probe_count),
                    random_seed=int(random_seed),
                )
            else:
                probe_points = np.empty((0, theta_dim), dtype=np.float64)
            probe_points = np.asarray(probe_points, dtype=np.float64)
            if probe_points.ndim != 2 or probe_points.shape[1] != theta_dim:
                probe_points = np.empty((0, theta_dim), dtype=np.float64)
            if probe_points.shape[0] > 0:
                candidate_tensor = torch.as_tensor(eligible_points, device=evaluator.device, dtype=evaluator.dtype)
                probe_tensor = torch.as_tensor(probe_points, device=evaluator.device, dtype=evaluator.dtype)
                if risk_mode == "balanced_exceedance" and band_risk_evaluators:
                    band_score_rows: list[np.ndarray] = []
                    band_score_weights: list[float] = []
                    with torch.no_grad():
                        for band_payload in band_risk_evaluators:
                            band_evaluator = band_payload.get("evaluator")
                            if band_evaluator is None:
                                continue
                            band_reduction_matrix = band_evaluator.variance_reduction_matrix(
                                candidate_tensor,
                                probe_tensor,
                                chunk_size=max(1, int(params.chunk_size)),
                            )
                            band_current_variance = band_evaluator.value(probe_tensor)
                            band_scores_tensor = _exceedance_reduction_scores(
                                band_current_variance,
                                band_reduction_matrix,
                                variance_floor=float(params.variance_floor),
                                quantile=float(params.imse_quantile),
                                mean_weight=float(params.imse_quantile_mean_weight),
                                max_weight=float(params.imse_quantile_max_weight),
                            )
                            band_score_rows.append(
                                self._unit_scale_values(
                                    band_scores_tensor.detach().cpu().numpy().astype(np.float64)
                                )
                            )
                            band_score_weights.append(float(max(0.0, band_payload.get("band_weight", 0.0))))
                    if band_score_rows:
                        score_matrix = np.vstack(band_score_rows).astype(np.float64)
                        weights = np.asarray(band_score_weights, dtype=np.float64).reshape(-1)
                        if weights.shape[0] != score_matrix.shape[0] or float(np.sum(weights)) <= 0.0:
                            weights = np.full((score_matrix.shape[0],), 1.0 / float(score_matrix.shape[0]))
                        else:
                            weights = weights / float(np.sum(weights))
                        balanced_mean = weights @ score_matrix
                        balanced_floor = np.min(score_matrix, axis=0)
                        p68_scores = 0.5 * balanced_mean + 0.5 * balanced_floor
                        composite += p68_weight * self._unit_scale_values(p68_scores)
                else:
                    with torch.no_grad():
                        reduction_matrix = evaluator.variance_reduction_matrix(
                            candidate_tensor,
                            probe_tensor,
                            chunk_size=max(1, int(params.chunk_size)),
                        )
                        if risk_mode == "validation_probe_shell" and probe_weights is not None:
                            weight_arr = np.asarray(probe_weights, dtype=np.float64).reshape(-1)
                            if weight_arr.shape[0] != probe_points.shape[0] or not np.any(weight_arr > 0.0):
                                p68_scores_tensor = torch.zeros(
                                    (eligible_points.shape[0],),
                                    device=evaluator.device,
                                    dtype=evaluator.dtype,
                                )
                            else:
                                weight_tensor = torch.as_tensor(
                                    np.maximum(weight_arr, 0.0),
                                    device=evaluator.device,
                                    dtype=evaluator.dtype,
                                )
                                weight_tensor = weight_tensor / torch.clamp(
                                    torch.sum(weight_tensor),
                                    min=torch.as_tensor(
                                        float(params.variance_floor),
                                        device=evaluator.device,
                                        dtype=evaluator.dtype,
                                    ),
                                )
                                p68_scores_tensor = torch.sum(
                                    reduction_matrix * weight_tensor.reshape(-1, 1),
                                    dim=0,
                                )
                        else:
                            current_variance = evaluator.value(probe_tensor)
                        if risk_mode == "quantile_proxy":
                            p68_scores_tensor = _quantile_proxy_reduction_scores(
                                current_variance,
                                reduction_matrix,
                                variance_floor=float(params.variance_floor),
                                quantile=float(params.imse_quantile),
                                mean_weight=float(params.imse_quantile_mean_weight),
                                max_weight=float(params.imse_quantile_max_weight),
                            )
                        elif risk_mode == "exceedance":
                            p68_scores_tensor = _exceedance_reduction_scores(
                                current_variance,
                                reduction_matrix,
                                variance_floor=float(params.variance_floor),
                                quantile=float(params.imse_quantile),
                                mean_weight=float(params.imse_quantile_mean_weight),
                                max_weight=float(params.imse_quantile_max_weight),
                            )
                        elif risk_mode == "soft_quantile":
                            p68_scores_tensor = _soft_quantile_reduction_scores(
                                current_variance,
                                reduction_matrix,
                                variance_floor=float(params.variance_floor),
                                quantile=float(params.imse_quantile),
                                bandwidth=float(params.imse_quantile_shell_width),
                                mean_weight=float(params.imse_quantile_mean_weight),
                                max_weight=float(params.imse_quantile_max_weight),
                            )
                        elif risk_mode == "rank_body":
                            p68_scores_tensor = _rank_body_reduction_scores(
                                current_variance,
                                reduction_matrix,
                                variance_floor=float(params.variance_floor),
                                quantile=float(params.imse_quantile),
                                shell_width=float(params.imse_quantile_shell_width),
                                mean_weight=float(params.imse_quantile_mean_weight),
                                max_weight=float(params.imse_quantile_max_weight),
                            )
                        elif risk_mode != "validation_probe_shell":
                            p68_scores_tensor = _quantile_shell_reduction_scores(
                                current_variance,
                                reduction_matrix,
                                variance_floor=float(params.variance_floor),
                                quantile=float(params.imse_quantile),
                                shell_width=float(params.imse_quantile_shell_width),
                                mean_weight=float(params.imse_quantile_mean_weight),
                                max_weight=float(params.imse_quantile_max_weight),
                            )
                    p68_scores = p68_scores_tensor.detach().cpu().numpy().astype(np.float64)
                    composite += p68_weight * self._unit_scale_values(p68_scores)

        if spacefill_weight > 0.0:
            discrepancy_gain = self._spacefill_discrepancy_gain(
                prefix_points=eligible_points,
                train_unit_thetas=train_unit_thetas,
            )
            composite += spacefill_weight * self._unit_scale_values(discrepancy_gain)

        if boundary_weight > 0.0:
            boundary_penalty = self._boundary_budget_penalty(
                prefix_points=eligible_points,
                train_unit_thetas=train_unit_thetas,
                threshold=float(params.acquisition_p68_set_rerank_boundary_threshold),
                target_fraction=float(params.acquisition_p68_set_rerank_boundary_target_fraction),
            )
            composite += boundary_weight * self._unit_scale_values(-boundary_penalty)

        if not np.any(np.isfinite(composite)):
            return ranking
        eligible_order = eligible_indices[np.argsort(-composite, kind="mergesort")]
        ordered_set = {int(index) for index in eligible_order.tolist()}
        prefix_indices = np.arange(prefix_count, dtype=np.int64)
        remainder = np.asarray(
            [int(index) for index in prefix_indices.tolist() if int(index) not in ordered_set],
            dtype=np.int64,
        )
        order = np.concatenate([eligible_order.astype(np.int64), remainder], axis=0)
        return _ComponentCandidateRanking(
            component_index=int(ranking.component_index),
            unit_points=np.concatenate(
                [
                    prefix_points[order].astype(np.float64),
                    np.asarray(ranking.unit_points[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            scores=np.concatenate(
                [
                    np.asarray(ranking.scores[:prefix_count], dtype=np.float64)[order].astype(np.float64),
                    np.asarray(ranking.scores[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            simplex_ids=np.concatenate(
                [
                    np.asarray(ranking.simplex_ids[:prefix_count], dtype=np.int64)[order].astype(np.int64),
                    np.asarray(ranking.simplex_ids[prefix_count:], dtype=np.int64),
                ],
                axis=0,
            ),
        )

    def _rerank_ranking_prefix_spacefill(
        self,
        *,
        ranking: _ComponentCandidateRanking,
        train_unit_thetas: np.ndarray,
        params: _RuntimeSelectorParams,
    ) -> _ComponentCandidateRanking:
        if (
            ranking.unit_points.shape[0] == 0
            or int(params.acquisition_spacefill_rerank_top_k) <= 0
            or float(params.acquisition_spacefill_weight) <= 0.0
        ):
            return ranking
        prefix_count = min(
            int(params.acquisition_spacefill_rerank_top_k),
            int(ranking.unit_points.shape[0]),
        )
        prefix_points = np.clip(np.asarray(ranking.unit_points[:prefix_count], dtype=np.float64), 0.0, 1.0)
        discrepancy_gain = self._spacefill_discrepancy_gain(
            prefix_points=prefix_points,
            train_unit_thetas=train_unit_thetas,
        )
        if discrepancy_gain.shape[0] == 0:
            return ranking
        acq_scores = np.asarray(ranking.scores[:prefix_count], dtype=np.float64)

        def _unit_scale(values: np.ndarray) -> np.ndarray:
            finite = np.asarray(values, dtype=np.float64)
            min_value = float(np.min(finite))
            span = float(np.max(finite) - min_value)
            if span <= 1.0e-30:
                return np.zeros_like(finite, dtype=np.float64)
            return (finite - min_value) / span

        weight = float(min(1.0, max(0.0, params.acquisition_spacefill_weight)))
        blended_scores = (1.0 - weight) * _unit_scale(acq_scores) + weight * _unit_scale(discrepancy_gain)
        order = np.argsort(-blended_scores, kind="mergesort")
        return _ComponentCandidateRanking(
            component_index=int(ranking.component_index),
            unit_points=np.concatenate(
                [
                    prefix_points[order].astype(np.float64),
                    np.asarray(ranking.unit_points[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            scores=np.concatenate(
                [
                    blended_scores[order].astype(np.float64),
                    np.asarray(ranking.scores[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            simplex_ids=np.concatenate(
                [
                    np.asarray(ranking.simplex_ids[:prefix_count], dtype=np.int64)[order].astype(np.int64),
                    np.asarray(ranking.simplex_ids[prefix_count:], dtype=np.int64),
                ],
                axis=0,
            ),
        )

    def _guard_ranking_prefix_spacefill(
        self,
        *,
        ranking: _ComponentCandidateRanking,
        train_unit_thetas: np.ndarray,
        params: _RuntimeSelectorParams,
    ) -> _ComponentCandidateRanking:
        if (
            ranking.unit_points.shape[0] == 0
            or int(params.acquisition_spacefill_guard_top_k) <= 0
            or float(params.acquisition_spacefill_guard_reject_quantile) <= 0.0
        ):
            return ranking
        prefix_count = min(
            int(params.acquisition_spacefill_guard_top_k),
            int(ranking.unit_points.shape[0]),
        )
        prefix_points = np.clip(np.asarray(ranking.unit_points[:prefix_count], dtype=np.float64), 0.0, 1.0)
        discrepancy_gain = self._spacefill_discrepancy_gain(
            prefix_points=prefix_points,
            train_unit_thetas=train_unit_thetas,
        )
        if discrepancy_gain.shape[0] == 0 or not np.all(np.isfinite(discrepancy_gain)):
            return ranking
        reject_quantile = float(
            min(1.0, max(0.0, params.acquisition_spacefill_guard_reject_quantile))
        )
        threshold = float(np.quantile(discrepancy_gain, reject_quantile))
        eligible_mask = discrepancy_gain >= threshold
        if bool(np.all(eligible_mask)) or not bool(np.any(eligible_mask)):
            return ranking
        prefix_indices = np.arange(prefix_count, dtype=np.int64)
        guarded_order = np.concatenate(
            [
                prefix_indices[eligible_mask],
                prefix_indices[~eligible_mask],
            ],
            axis=0,
        )
        return _ComponentCandidateRanking(
            component_index=int(ranking.component_index),
            unit_points=np.concatenate(
                [
                    prefix_points[guarded_order].astype(np.float64),
                    np.asarray(ranking.unit_points[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            scores=np.concatenate(
                [
                    np.asarray(ranking.scores[:prefix_count], dtype=np.float64)[guarded_order].astype(np.float64),
                    np.asarray(ranking.scores[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            simplex_ids=np.concatenate(
                [
                    np.asarray(ranking.simplex_ids[:prefix_count], dtype=np.int64)[guarded_order].astype(np.int64),
                    np.asarray(ranking.simplex_ids[prefix_count:], dtype=np.int64),
                ],
                axis=0,
            ),
        )

    def _tiebreak_ranking_prefix_spacefill(
        self,
        *,
        ranking: _ComponentCandidateRanking,
        train_unit_thetas: np.ndarray,
        params: _RuntimeSelectorParams,
    ) -> _ComponentCandidateRanking:
        if (
            ranking.unit_points.shape[0] == 0
            or int(params.acquisition_spacefill_tiebreak_top_k) <= 0
            or float(params.acquisition_spacefill_tiebreak_score_ratio) <= 0.0
        ):
            return ranking
        prefix_count = min(
            int(params.acquisition_spacefill_tiebreak_top_k),
            int(ranking.unit_points.shape[0]),
        )
        prefix_points = np.clip(np.asarray(ranking.unit_points[:prefix_count], dtype=np.float64), 0.0, 1.0)
        acq_scores = np.asarray(ranking.scores[:prefix_count], dtype=np.float64)
        if acq_scores.shape[0] == 0 or not np.all(np.isfinite(acq_scores)):
            return ranking
        top_score = float(np.max(acq_scores))
        if not np.isfinite(top_score) or top_score <= 0.0:
            return ranking
        score_ratio = float(min(1.0, max(0.0, params.acquisition_spacefill_tiebreak_score_ratio)))
        eligible_mask = acq_scores >= top_score * score_ratio
        if int(np.count_nonzero(eligible_mask)) <= 1:
            return ranking
        discrepancy_gain = self._spacefill_discrepancy_gain(
            prefix_points=prefix_points,
            train_unit_thetas=train_unit_thetas,
        )
        if discrepancy_gain.shape[0] == 0 or not np.all(np.isfinite(discrepancy_gain)):
            return ranking
        prefix_indices = np.arange(prefix_count, dtype=np.int64)
        eligible_indices = prefix_indices[eligible_mask]
        eligible_order = eligible_indices[
            np.argsort(-discrepancy_gain[eligible_indices], kind="mergesort")
        ]
        ordered_set = {int(index) for index in eligible_order.tolist()}
        remainder = np.asarray(
            [int(index) for index in prefix_indices.tolist() if int(index) not in ordered_set],
            dtype=np.int64,
        )
        order = np.concatenate([eligible_order.astype(np.int64), remainder], axis=0)
        return _ComponentCandidateRanking(
            component_index=int(ranking.component_index),
            unit_points=np.concatenate(
                [
                    prefix_points[order].astype(np.float64),
                    np.asarray(ranking.unit_points[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            scores=np.concatenate(
                [
                    np.asarray(ranking.scores[:prefix_count], dtype=np.float64)[order].astype(np.float64),
                    np.asarray(ranking.scores[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            simplex_ids=np.concatenate(
                [
                    np.asarray(ranking.simplex_ids[:prefix_count], dtype=np.int64)[order].astype(np.int64),
                    np.asarray(ranking.simplex_ids[prefix_count:], dtype=np.int64),
                ],
                axis=0,
            ),
        )

    def _guard_ranking_prefix_spacefill_cd_nonworse(
        self,
        *,
        ranking: _ComponentCandidateRanking,
        train_unit_thetas: np.ndarray,
        params: _RuntimeSelectorParams,
    ) -> _ComponentCandidateRanking:
        if (
            ranking.unit_points.shape[0] == 0
            or int(params.acquisition_spacefill_cd_nonworse_top_k) <= 0
        ):
            return ranking
        prefix_count = min(
            int(params.acquisition_spacefill_cd_nonworse_top_k),
            int(ranking.unit_points.shape[0]),
        )
        prefix_points = np.clip(np.asarray(ranking.unit_points[:prefix_count], dtype=np.float64), 0.0, 1.0)
        discrepancy_gain = self._spacefill_discrepancy_gain(
            prefix_points=prefix_points,
            train_unit_thetas=train_unit_thetas,
        )
        if discrepancy_gain.shape[0] == 0 or not np.all(np.isfinite(discrepancy_gain)):
            return ranking
        tolerance = float(max(0.0, params.acquisition_spacefill_cd_nonworse_tol))
        eligible_mask = discrepancy_gain >= -tolerance
        if bool(np.all(eligible_mask)) or not bool(np.any(eligible_mask)):
            return ranking
        prefix_indices = np.arange(prefix_count, dtype=np.int64)
        guarded_order = np.concatenate(
            [
                prefix_indices[eligible_mask],
                prefix_indices[~eligible_mask],
            ],
            axis=0,
        )
        return _ComponentCandidateRanking(
            component_index=int(ranking.component_index),
            unit_points=np.concatenate(
                [
                    prefix_points[guarded_order].astype(np.float64),
                    np.asarray(ranking.unit_points[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            scores=np.concatenate(
                [
                    np.asarray(ranking.scores[:prefix_count], dtype=np.float64)[guarded_order].astype(np.float64),
                    np.asarray(ranking.scores[prefix_count:], dtype=np.float64),
                ],
                axis=0,
            ),
            simplex_ids=np.concatenate(
                [
                    np.asarray(ranking.simplex_ids[:prefix_count], dtype=np.int64)[guarded_order].astype(np.int64),
                    np.asarray(ranking.simplex_ids[prefix_count:], dtype=np.int64),
                ],
                axis=0,
            ),
        )

    @staticmethod
    def _generate_qmc_candidate_pool(
        *,
        theta_dim: int,
        candidate_count: int,
        train_unit_thetas: np.ndarray,
        duplicate_tol: float,
        random_seed: int,
    ) -> np.ndarray:
        requested = int(candidate_count)
        if int(theta_dim) <= 0 or requested <= 0:
            return np.empty((0, max(0, int(theta_dim))), dtype=np.float64)
        train_points = np.clip(np.asarray(train_unit_thetas, dtype=np.float64), 0.0, 1.0)
        oversample = max(requested * 2, requested + int(train_points.shape[0]) + 1, 2)
        power = int(np.ceil(np.log2(float(oversample))))
        sampler = qmc.Sobol(d=int(theta_dim), scramble=True, seed=int(random_seed))
        points = np.asarray(sampler.random_base2(power), dtype=np.float64)
        if train_points.ndim == 2 and train_points.shape[0] > 0:
            keep = np.ones((points.shape[0],), dtype=bool)
            chunk_size = 4096
            for start in range(0, points.shape[0], chunk_size):
                chunk = points[start : start + chunk_size]
                distances = np.linalg.norm(chunk[:, None, :] - train_points[None, :, :], axis=2)
                keep[start : start + chunk.shape[0]] = np.min(distances, axis=1) > float(duplicate_tol)
            points = points[keep]
        if points.shape[0] < requested:
            rng = np.random.default_rng(int(random_seed) + 1)
            supplement_chunks: list[np.ndarray] = []
            needed = requested - int(points.shape[0])
            attempts = 0
            while needed > 0 and attempts < 8:
                attempts += 1
                candidate_count = max(needed * 2, needed + 8)
                supplement = rng.random((candidate_count, int(theta_dim)), dtype=np.float64)
                if train_points.ndim == 2 and train_points.shape[0] > 0:
                    distances = np.linalg.norm(
                        supplement[:, None, :] - train_points[None, :, :],
                        axis=2,
                    )
                    supplement = supplement[np.min(distances, axis=1) > float(duplicate_tol)]
                if supplement.shape[0] > 0:
                    take = supplement[:needed]
                    supplement_chunks.append(take.astype(np.float64))
                    needed -= int(take.shape[0])
            if supplement_chunks:
                points = np.concatenate([points, *supplement_chunks], axis=0)
            if points.shape[0] < requested:
                fallback = rng.random((requested - points.shape[0], int(theta_dim)), dtype=np.float64)
                points = np.concatenate([points, fallback], axis=0)
        return np.clip(points[:requested], 0.0, 1.0).astype(np.float64)

    def _build_qmc_candidate_pool_ranking(
        self,
        *,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator,
        train_unit_thetas: np.ndarray,
        params: _RuntimeSelectorParams,
        random_seed: int,
    ) -> _ComponentCandidateRanking:
        train_points = np.asarray(train_unit_thetas, dtype=np.float64)
        theta_dim = int(train_points.shape[1]) if train_points.ndim == 2 else 0
        pool_points = self._generate_qmc_candidate_pool(
            theta_dim=theta_dim,
            candidate_count=int(params.acquisition_qmc_pool_count),
            train_unit_thetas=train_points,
            duplicate_tol=float(params.duplicate_tol),
            random_seed=int(random_seed),
        )
        scores = self._evaluate_points(
            evaluator=evaluator,
            points=pool_points,
            chunk_size=params.chunk_size,
        )
        order = np.argsort(-scores, kind="mergesort")
        ranking = _ComponentCandidateRanking(
            component_index=-1,
            unit_points=pool_points[order].astype(np.float64),
            scores=scores[order].astype(np.float64),
            simplex_ids=-np.arange(1, int(pool_points.shape[0]) + 1, dtype=np.int64)[order],
        )
        return self._rerank_ranking_prefix_imse(
            ranking=ranking,
            evaluator=evaluator,
            params=params,
            random_seed=int(random_seed + params.imse_probe_seed_offset),
        )

    def _subset_scoring_by_simplex_ids(
        self,
        scoring: _SimplexScoring,
        simplex_ids: np.ndarray,
    ) -> _SimplexScoring:
        requested_ids = np.asarray(simplex_ids, dtype=np.int64).reshape(-1)
        if scoring.simplex_vertices.shape[0] == 0 or requested_ids.size == 0:
            return self._subset_scoring(scoring, 0)
        id_to_index = {
            int(simplex_id): row_idx
            for row_idx, simplex_id in enumerate(np.asarray(scoring.simplex_ids, dtype=np.int64).tolist())
        }
        chosen_rows = [
            int(id_to_index[int(simplex_id)])
            for simplex_id in requested_ids.tolist()
            if int(simplex_id) in id_to_index
        ]
        if not chosen_rows:
            return self._subset_scoring(scoring, 0)
        chosen = np.asarray(chosen_rows, dtype=np.int64)
        return _SimplexScoring(
            collection_name=scoring.collection_name,
            simplex_vertices=scoring.simplex_vertices[chosen].astype(np.float64),
            simplex_ids=scoring.simplex_ids[chosen].astype(np.int64),
            repr_points=scoring.repr_points[chosen].astype(np.float64),
            repr_scores=scoring.repr_scores[chosen].astype(np.float64),
            barycenter_bary=scoring.barycenter_bary[chosen].astype(np.float64),
            circumcenter_bary=scoring.circumcenter_bary[chosen].astype(np.float64),
            circumcenter_valid=scoring.circumcenter_valid[chosen].astype(bool),
            best_vertex_index=scoring.best_vertex_index[chosen].astype(np.int64),
            facet_center_bary=scoring.facet_center_bary[chosen].astype(np.float64),
            facet_center_scores=scoring.facet_center_scores[chosen].astype(np.float64),
            best_facet_center_index=scoring.best_facet_center_index[chosen].astype(np.int64),
        )

    def _clone_scoring_with_repr(
        self,
        scoring: _SimplexScoring,
        *,
        repr_points: np.ndarray,
        repr_scores: np.ndarray,
    ) -> _SimplexScoring:
        return _SimplexScoring(
            collection_name=scoring.collection_name,
            simplex_vertices=np.asarray(scoring.simplex_vertices, dtype=np.float64).copy(),
            simplex_ids=np.asarray(scoring.simplex_ids, dtype=np.int64).copy(),
            repr_points=np.asarray(repr_points, dtype=np.float64).copy(),
            repr_scores=np.asarray(repr_scores, dtype=np.float64).copy(),
            barycenter_bary=np.asarray(scoring.barycenter_bary, dtype=np.float64).copy(),
            circumcenter_bary=np.asarray(scoring.circumcenter_bary, dtype=np.float64).copy(),
            circumcenter_valid=np.asarray(scoring.circumcenter_valid, dtype=bool).copy(),
            best_vertex_index=np.asarray(scoring.best_vertex_index, dtype=np.int64).copy(),
            facet_center_bary=np.asarray(scoring.facet_center_bary, dtype=np.float64).copy(),
            facet_center_scores=np.asarray(scoring.facet_center_scores, dtype=np.float64).copy(),
            best_facet_center_index=np.asarray(scoring.best_facet_center_index, dtype=np.int64).copy(),
        )

    def _refine_scoring(
        self,
        *,
        scoring: _SimplexScoring,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator,
        params: _RuntimeSelectorParams,
        progress_callback: ProgressCallback | None,
        random_seed: int,
        stage_name: str = "module3_refinement",
        start_strategy: str = "legacy",
    ) -> _ComponentCandidateRanking:
        simplex_vertices = np.asarray(scoring.simplex_vertices, dtype=np.float64)
        simplex_count = int(simplex_vertices.shape[0])
        if simplex_count == 0:
            theta_dim = int(scoring.repr_points.shape[1]) if scoring.repr_points.ndim == 2 else 8
            return _ComponentCandidateRanking(
                component_index=-1,
                unit_points=np.empty((0, theta_dim), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float64),
                simplex_ids=np.empty((0,), dtype=np.int64),
            )

        starts = self._build_refinement_starts(
            scoring=scoring,
            params=params,
            random_seed=random_seed,
            start_strategy=start_strategy,
        )
        simplex_size = int(simplex_vertices.shape[1])
        theta_dim = int(simplex_vertices.shape[2])
        state_count = simplex_count * int(params.starts_per_simplex_refine)
        # `scoring.simplex_ids` preserves upstream hull/domain IDs for reporting,
        # but GPU refinement must index the local packed `simplex_vertices` tensor.
        local_simplex_indices = np.repeat(
            np.arange(simplex_count, dtype=np.int64),
            int(params.starts_per_simplex_refine),
        )

        device = evaluator.device
        dtype = evaluator.dtype
        z_gpu = torch.as_tensor(simplex_vertices, device=device, dtype=dtype)
        lambda_state = torch.as_tensor(starts.reshape(state_count, simplex_size), device=device, dtype=dtype)
        simplex_id_state = torch.as_tensor(local_simplex_indices, device=device, dtype=torch.int64)
        best_lambda = lambda_state.clone()
        best_value = torch.full((state_count,), -torch.inf, device=device, dtype=dtype)
        prev_lambda = torch.zeros_like(lambda_state)
        prev_grad = torch.zeros_like(lambda_state)
        has_prev = torch.zeros((state_count,), device=device, dtype=torch.bool)
        active = torch.ones((state_count,), device=device, dtype=torch.bool)
        s_history = torch.zeros(
            (state_count, params.history_size_refine, simplex_size),
            device=device,
            dtype=dtype,
        )
        y_history = torch.zeros_like(s_history)
        history_counts = torch.zeros((state_count,), device=device, dtype=torch.int64)

        for iteration_idx in range(int(params.max_iter_refine)):
            active_indices = torch.nonzero(active, as_tuple=False).reshape(-1)
            if active_indices.numel() == 0:
                break
            for start in range(0, int(active_indices.numel()), int(params.chunk_size)):
                chunk_idx = active_indices[start : start + int(params.chunk_size)]
                lambda_chunk = lambda_state.index_select(0, chunk_idx)
                z_chunk = z_gpu.index_select(0, simplex_id_state.index_select(0, chunk_idx))
                q_chunk = torch.sum(z_chunk * lambda_chunk[:, :, None], dim=1)
                value_chunk, grad_q_chunk = evaluator.value_and_grad(q_chunk)
                grad_lambda_chunk = torch.einsum("bik,bk->bi", z_chunk, grad_q_chunk)
                grad_lambda_chunk = torch.nan_to_num(grad_lambda_chunk, nan=0.0, posinf=0.0, neginf=0.0)

                improve_current = value_chunk > best_value.index_select(0, chunk_idx)
                if torch.any(improve_current):
                    idx_current = chunk_idx[improve_current]
                    best_value[idx_current] = value_chunk[improve_current]
                    best_lambda[idx_current] = lambda_chunk[improve_current]

                has_prev_chunk = has_prev.index_select(0, chunk_idx)
                if torch.any(has_prev_chunk):
                    prev_lambda_chunk = prev_lambda.index_select(0, chunk_idx)
                    prev_grad_chunk = prev_grad.index_select(0, chunk_idx)
                    _append_lbfgs_history(
                        s_history=s_history,
                        y_history=y_history,
                        counts=history_counts,
                        state_indices=chunk_idx[has_prev_chunk],
                        s_update=lambda_chunk[has_prev_chunk] - prev_lambda_chunk[has_prev_chunk],
                        y_update=grad_lambda_chunk[has_prev_chunk] - prev_grad_chunk[has_prev_chunk],
                        curvature_tol=params.curvature_tol,
                    )

                direction = _lbfgs_two_loop_direction(
                    grad_lambda_chunk,
                    s_history.index_select(0, chunk_idx),
                    y_history.index_select(0, chunk_idx),
                    history_counts.index_select(0, chunk_idx),
                    params.curvature_tol,
                )
                directional_derivative = torch.sum(grad_lambda_chunk * direction, dim=1)
                non_ascent = ~torch.isfinite(directional_derivative) | (directional_derivative <= 0.0)
                if torch.any(non_ascent):
                    direction = direction.clone()
                    direction[non_ascent] = grad_lambda_chunk[non_ascent]
                    directional_derivative = torch.sum(grad_lambda_chunk * direction, dim=1)

                accepted = torch.zeros((lambda_chunk.shape[0],), device=device, dtype=torch.bool)
                accepted_lambda = lambda_chunk.clone()
                accepted_value = value_chunk.clone()
                armijo_base = value_chunk
                accepted_rows = None
                first_step = None
                fallback_states = None
                fallback_lambda = None
                fallback_q = None
                fallback_value = None

                step_sizes_tensor = torch.as_tensor(
                    params.line_search_steps_refine,
                    device=device,
                    dtype=dtype,
                )
                step_count = int(step_sizes_tensor.numel())
                candidate_lambda_all = _project_to_simplex_torch(
                    (
                        lambda_chunk[None, :, :]
                        + step_sizes_tensor[:, None, None] * direction[None, :, :]
                    ).reshape(step_count * lambda_chunk.shape[0], simplex_size)
                ).reshape(step_count, lambda_chunk.shape[0], simplex_size)
                candidate_q_all = torch.sum(
                    z_chunk[None, :, :, :] * candidate_lambda_all[:, :, :, None],
                    dim=2,
                )
                candidate_value_all = _value_in_chunks(
                    evaluator,
                    candidate_q_all.reshape(step_count * lambda_chunk.shape[0], theta_dim),
                    chunk_size=max(1, min(int(params.chunk_size), 2048)),
                ).reshape(step_count, lambda_chunk.shape[0])
                armijo_threshold = (
                    armijo_base[None, :]
                    + params.armijo_c1 * step_sizes_tensor[:, None] * directional_derivative[None, :]
                )
                acceptable = candidate_value_all >= armijo_threshold
                accepted_any = torch.any(acceptable, dim=0)
                if torch.any(accepted_any):
                    accepted_rows = torch.nonzero(accepted_any, as_tuple=False).reshape(-1)
                    first_step = torch.argmax(acceptable.to(torch.int64), dim=0).index_select(0, accepted_rows)
                    accepted_lambda[accepted_rows] = candidate_lambda_all[first_step, accepted_rows]
                    accepted_value[accepted_rows] = candidate_value_all[first_step, accepted_rows]
                    accepted[accepted_rows] = True

                if torch.any(~accepted):
                    fallback_states = ~accepted
                    fallback_lambda = _project_to_simplex_torch(
                        lambda_chunk[fallback_states] + params.fallback_step_refine * direction[fallback_states]
                    )
                    fallback_q = torch.sum(z_chunk[fallback_states] * fallback_lambda[:, :, None], dim=1)
                    fallback_value = _value_in_chunks(
                        evaluator,
                        fallback_q,
                        chunk_size=max(1, min(int(params.chunk_size), 2048)),
                    )
                    accepted_lambda[fallback_states] = fallback_lambda
                    accepted_value[fallback_states] = fallback_value

                improve_try = accepted_value > best_value.index_select(0, chunk_idx)
                if torch.any(improve_try):
                    idx_try = chunk_idx[improve_try]
                    best_value[idx_try] = accepted_value[improve_try]
                    best_lambda[idx_try] = accepted_lambda[improve_try]

                projected_grad = _project_to_simplex_torch(lambda_chunk + grad_lambda_chunk) - lambda_chunk
                step_norm = torch.linalg.norm(accepted_lambda - lambda_chunk, dim=1)
                projected_grad_norm = torch.linalg.norm(projected_grad, dim=1)
                converged = (
                    step_norm <= float(params.convergence_tol_refine)
                ) | (
                    projected_grad_norm <= float(params.convergence_tol_refine)
                )

                prev_lambda[chunk_idx] = lambda_chunk
                prev_grad[chunk_idx] = grad_lambda_chunk
                has_prev[chunk_idx] = True
                lambda_state[chunk_idx] = accepted_lambda
                if torch.any(converged):
                    active[chunk_idx[converged]] = False
                del (
                    chunk_idx,
                    lambda_chunk,
                    z_chunk,
                    q_chunk,
                    value_chunk,
                    grad_q_chunk,
                    grad_lambda_chunk,
                    direction,
                    directional_derivative,
                    non_ascent,
                    accepted,
                    accepted_lambda,
                    accepted_value,
                    armijo_base,
                    step_sizes_tensor,
                    step_count,
                    candidate_lambda_all,
                    candidate_q_all,
                    candidate_value_all,
                    armijo_threshold,
                    acceptable,
                    accepted_any,
                    accepted_rows,
                    first_step,
                    fallback_states,
                    fallback_lambda,
                    fallback_q,
                    fallback_value,
                    improve_try,
                    projected_grad,
                    step_norm,
                    projected_grad_norm,
                    converged,
                )
                _empty_cuda_cache(device)

            if progress_callback is not None:
                progress_callback(stage_name, iteration_idx + 1, int(params.max_iter_refine))

        if progress_callback is not None:
            progress_callback(stage_name, int(params.max_iter_refine), int(params.max_iter_refine))

        best_points_state = torch.sum(
            z_gpu.index_select(0, simplex_id_state) * best_lambda[:, :, None],
            dim=1,
        )
        best_points_cpu = best_points_state.detach().cpu().numpy().reshape(
            simplex_count,
            int(params.starts_per_simplex_refine),
            theta_dim,
        )
        reevaluated_state_values = self._evaluate_points(
            evaluator=evaluator,
            points=best_points_cpu.reshape(state_count, theta_dim),
            chunk_size=params.chunk_size,
        ).reshape(simplex_count, int(params.starts_per_simplex_refine))
        best_start = np.argmax(reevaluated_state_values, axis=1)
        chosen_points = best_points_cpu[np.arange(simplex_count), best_start]
        chosen_value = reevaluated_state_values[np.arange(simplex_count), best_start]
        order = np.argsort(-chosen_value, kind="mergesort")
        ranking = _ComponentCandidateRanking(
            component_index=-1,
            unit_points=chosen_points[order].astype(np.float64),
            scores=chosen_value[order].astype(np.float64),
            simplex_ids=scoring.simplex_ids[order].astype(np.int64),
        )
        del (
            z_gpu,
            lambda_state,
            simplex_id_state,
            best_lambda,
            best_value,
            prev_lambda,
            prev_grad,
            has_prev,
            active,
            s_history,
            y_history,
            history_counts,
            best_points_state,
        )
        _empty_cuda_cache(device)
        return ranking

    def _qmc_search_simplex_subset(
        self,
        *,
        scoring: _SimplexScoring,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator,
        params: _RuntimeSelectorParams,
        component_index: int,
    ) -> _ComponentCandidateRanking:
        simplex_count = int(scoring.simplex_vertices.shape[0])
        if simplex_count == 0 or int(params.stage3_qmc_sample_count) <= 0:
            theta_dim = int(scoring.repr_points.shape[1]) if scoring.repr_points.ndim == 2 else 8
            return _ComponentCandidateRanking(
                component_index=int(component_index),
                unit_points=np.empty((0, theta_dim), dtype=np.float64),
                scores=np.empty((0,), dtype=np.float64),
                simplex_ids=np.empty((0,), dtype=np.int64),
            )
        simplex_vertices = np.asarray(scoring.simplex_vertices, dtype=np.float64)
        simplex_size = int(simplex_vertices.shape[1])
        theta_dim = int(simplex_vertices.shape[2])
        qmc_bary = _dirichlet_qmc_barycentric(
            simplex_size=simplex_size,
            sample_count=int(params.stage3_qmc_sample_count),
        )
        qmc_points = np.einsum(
            "mij,ni->mnj",
            simplex_vertices,
            qmc_bary,
            optimize=True,
        )
        qmc_scores = self._evaluate_points(
            evaluator=evaluator,
            points=qmc_points.reshape(simplex_count * qmc_bary.shape[0], theta_dim),
            chunk_size=params.stage3_qmc_chunk_size,
        ).reshape(simplex_count, qmc_bary.shape[0])
        best_qmc_index = np.argmax(qmc_scores, axis=1).astype(np.int64)
        best_qmc_scores = qmc_scores[np.arange(simplex_count), best_qmc_index]
        best_qmc_points = qmc_points[np.arange(simplex_count), best_qmc_index]
        order = np.argsort(-best_qmc_scores, kind="mergesort")
        return _ComponentCandidateRanking(
            component_index=int(component_index),
            unit_points=best_qmc_points[order].astype(np.float64),
            scores=best_qmc_scores[order].astype(np.float64),
            simplex_ids=np.asarray(scoring.simplex_ids, dtype=np.int64)[order].astype(np.int64),
        )

    def _merge_rankings_by_simplex(
        self,
        *,
        base_scoring: _SimplexScoring,
        stage1_ranking: _ComponentCandidateRanking | None,
        stage2_ranking: _ComponentCandidateRanking | None,
        component_index: int,
    ) -> _ComponentCandidateRanking:
        merged_points = np.asarray(base_scoring.repr_points, dtype=np.float64).copy()
        merged_scores = np.asarray(base_scoring.repr_scores, dtype=np.float64).copy()
        simplex_ids = np.asarray(base_scoring.simplex_ids, dtype=np.int64).copy()
        id_to_row = {
            int(simplex_id): row_idx
            for row_idx, simplex_id in enumerate(simplex_ids.tolist())
        }
        for ranking in (stage1_ranking, stage2_ranking):
            if ranking is None or ranking.unit_points.shape[0] == 0:
                continue
            for simplex_id, point, score in zip(
                np.asarray(ranking.simplex_ids, dtype=np.int64).tolist(),
                np.asarray(ranking.unit_points, dtype=np.float64),
                np.asarray(ranking.scores, dtype=np.float64).tolist(),
                strict=True,
            ):
                row_idx = id_to_row.get(int(simplex_id))
                if row_idx is None:
                    continue
                merged_points[row_idx] = np.asarray(point, dtype=np.float64)
                merged_scores[row_idx] = float(score)
        order = np.argsort(-merged_scores, kind="mergesort")
        return _ComponentCandidateRanking(
            component_index=int(component_index),
            unit_points=merged_points[order].astype(np.float64),
            scores=merged_scores[order].astype(np.float64),
            simplex_ids=simplex_ids[order].astype(np.int64),
        )

    def _search_objective_candidates_legacy(
        self,
        *,
        objective_spec: _ObjectiveSpec,
        hull_collection: _SimplexCollection,
        domain_collection: _SimplexCollection,
        domain_simplex_id_start: int,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator,
        params: _RuntimeSelectorParams,
        progress_callback: ProgressCallback | None,
        objective_index: int,
        objective_count: int,
        random_seed_base: int,
    ) -> tuple[_ComponentCandidateRanking, dict[str, Any]]:
        scoring_steps = 2 if params.coverage_mode == "hull_domain_hybrid" else 1
        hull_scoring = self._score_simplex_collection(
            collection=hull_collection,
            evaluator=evaluator,
            params=params,
        )
        if progress_callback is not None:
            progress_callback(
                "module3_repr_scoring",
                int(objective_index * scoring_steps + 1),
                int(objective_count * scoring_steps),
            )

        domain_scoring = self._score_simplex_collection(
            collection=domain_collection,
            evaluator=evaluator,
            params=params,
        )
        domain_scoring = self._apply_collection_score_scale(
            domain_scoring,
            params=params,
        )
        if progress_callback is not None and scoring_steps > 1:
            progress_callback(
                "module3_repr_scoring",
                int(objective_index * scoring_steps + scoring_steps),
                int(objective_count * scoring_steps),
            )

        hull_count = int(hull_scoring.simplex_vertices.shape[0])
        domain_count = int(domain_scoring.simplex_vertices.shape[0])
        hull_fraction_count = (
            int(np.ceil(float(params.hull_refine_fraction) * float(hull_count)))
            if hull_count > 0
            else 0
        )
        hull_refine_count = min(
            hull_count,
            max(1 if hull_count > 0 else 0, hull_fraction_count),
        )
        domain_refine_count = (
            domain_count
            if params.domain_refine_all
            else min(domain_count, int(params.domain_top_k))
        )
        hull_refine = self._subset_scoring(hull_scoring, hull_refine_count)
        domain_refine = self._subset_scoring(domain_scoring, domain_refine_count)
        merged_stage1 = self._merge_scoring([hull_refine, domain_refine])

        def _objective_progress(stage: str, current: int, total: int) -> None:
            if progress_callback is None:
                return
            progress_callback(
                stage,
                int(objective_index * int(total) + int(current)),
                int(objective_count * int(total)),
            )

        stage1_ranking = self._refine_scoring(
            scoring=merged_stage1,
            evaluator=evaluator,
            params=params,
            progress_callback=_objective_progress if progress_callback is not None else None,
            random_seed=int(random_seed_base + objective_index),
            stage_name="module3_refinement_stage1",
        )
        stage1_ranking.component_index = int(objective_spec.objective_index)
        stage1_ranking = self._apply_domain_score_scale_to_ranking(
            stage1_ranking,
            domain_simplex_id_start=domain_simplex_id_start,
            params=params,
        )
        stage2_simplex_ids = np.asarray(
            stage1_ranking.simplex_ids[: min(int(params.polish_top_k), stage1_ranking.simplex_ids.shape[0])],
            dtype=np.int64,
        )
        stage2_scoring = self._subset_scoring_by_simplex_ids(merged_stage1, stage2_simplex_ids)
        stage2_params = replace(
            params,
            starts_per_simplex_refine=int(params.polish_starts_per_simplex_refine),
            max_iter_refine=int(params.polish_max_iter_refine),
            history_size_refine=int(params.polish_history_size_refine),
            convergence_tol_refine=float(params.polish_convergence_tol_refine),
        )
        stage2_ranking = self._refine_scoring(
            scoring=stage2_scoring,
            evaluator=evaluator,
            params=stage2_params,
            progress_callback=_objective_progress if progress_callback is not None else None,
            random_seed=int(random_seed_base + 1000 + objective_index),
            stage_name="module3_refinement_stage2",
        )
        stage2_ranking.component_index = int(objective_spec.objective_index)
        stage2_ranking = self._apply_domain_score_scale_to_ranking(
            stage2_ranking,
            domain_simplex_id_start=domain_simplex_id_start,
            params=params,
        )
        repr_ranking = self._merge_rankings(
            [
                self._ranking_from_scoring(
                    hull_scoring,
                    component_index=int(objective_spec.objective_index),
                ),
                self._ranking_from_scoring(
                    domain_scoring,
                    component_index=int(objective_spec.objective_index),
                ),
            ]
        )
        combined_ranking = self._concat_rankings([stage2_ranking, stage1_ranking, repr_ranking])
        combined_ranking = self._rerank_ranking_prefix_exact(
            ranking=combined_ranking,
            evaluator=evaluator,
            top_k=int(params.polish_top_k),
        )
        combined_ranking = self._rerank_ranking_prefix_imse(
            ranking=combined_ranking,
            evaluator=evaluator,
            params=params,
            random_seed=int(random_seed_base + 3000 + objective_index + params.imse_probe_seed_offset),
        )
        combined_ranking.component_index = int(objective_spec.objective_index)
        return combined_ranking, {
            "objective_index": int(objective_spec.objective_index),
            "objective_label": str(objective_spec.objective_label),
            "refinement_architecture": "legacy_full_refine",
            "num_hull_simplices": hull_count,
            "num_domain_simplices": domain_count,
            "num_scored_simplices": int(hull_count + domain_count),
            "num_refined_simplices_stage1": int(merged_stage1.simplex_vertices.shape[0]),
            "num_refined_simplices_stage2": int(stage2_scoring.simplex_vertices.shape[0]),
            "hull_refine_count": int(hull_refine_count),
            "domain_refine_count": int(domain_refine_count),
        }

    def _search_objective_candidates_hierarchical(
        self,
        *,
        objective_spec: _ObjectiveSpec,
        hull_collection: _SimplexCollection,
        domain_collection: _SimplexCollection,
        domain_simplex_id_start: int,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator,
        params: _RuntimeSelectorParams,
        progress_callback: ProgressCallback | None,
        objective_index: int,
        objective_count: int,
        random_seed_base: int,
    ) -> tuple[_ComponentCandidateRanking, dict[str, Any]]:
        scoring_steps = 2 if params.coverage_mode == "hull_domain_hybrid" else 1
        hull_scoring = self._score_simplex_collection(
            collection=hull_collection,
            evaluator=evaluator,
            params=params,
        )
        if progress_callback is not None:
            progress_callback(
                "module3_repr_scoring",
                int(objective_index * scoring_steps + 1),
                int(objective_count * scoring_steps),
            )

        domain_scoring = self._score_simplex_collection(
            collection=domain_collection,
            evaluator=evaluator,
            params=params,
        )
        domain_scoring = self._apply_collection_score_scale(
            domain_scoring,
            params=params,
        )
        if progress_callback is not None and scoring_steps > 1:
            progress_callback(
                "module3_repr_scoring",
                int(objective_index * scoring_steps + scoring_steps),
                int(objective_count * scoring_steps),
            )

        hull_count = int(hull_scoring.simplex_vertices.shape[0])
        domain_count = int(domain_scoring.simplex_vertices.shape[0])
        hull_fraction_count = (
            int(np.ceil(float(params.hull_refine_fraction) * float(hull_count)))
            if hull_count > 0
            else 0
        )
        hull_refine_count = min(
            hull_count,
            max(int(params.global_top_k), hull_fraction_count),
        )
        domain_refine_count = (
            domain_count
            if params.domain_refine_all
            else min(domain_count, int(params.domain_top_k))
        )
        hull_refine = self._subset_scoring(hull_scoring, hull_refine_count)
        domain_refine = self._subset_scoring(domain_scoring, domain_refine_count)
        merged_stage0 = self._merge_scoring([hull_refine, domain_refine])

        def _objective_progress(stage: str, current: int, total: int) -> None:
            if progress_callback is None:
                return
            progress_callback(
                stage,
                int(objective_index * int(total) + int(current)),
                int(objective_count * int(total)),
            )

        stage_qmc_count = _fractional_stage_count(
            int(merged_stage0.simplex_vertices.shape[0]),
            float(params.hierarchical_stage1_refine_fraction),
        )
        stage_qmc_scoring = self._subset_scoring(merged_stage0, stage_qmc_count)
        if progress_callback is not None:
            progress_callback("module3_stage_qmc", int(objective_index * 2 + 1), int(objective_count * 2))
        stage_qmc_ranking = self._qmc_search_simplex_subset(
            scoring=stage_qmc_scoring,
            evaluator=evaluator,
            params=params,
            component_index=int(objective_spec.objective_index),
        )
        if progress_callback is not None:
            progress_callback("module3_stage_qmc", int(objective_index * 2 + 2), int(objective_count * 2))
        stage_qmc_ranking.component_index = int(objective_spec.objective_index)
        stage_qmc_ranking = self._apply_domain_score_scale_to_ranking(
            stage_qmc_ranking,
            domain_simplex_id_start=domain_simplex_id_start,
            params=params,
        )
        stage_qmc_map = {
            int(simplex_id): (
                np.asarray(point, dtype=np.float64),
                float(score),
            )
            for simplex_id, point, score in zip(
                np.asarray(stage_qmc_ranking.simplex_ids, dtype=np.int64).tolist(),
                np.asarray(stage_qmc_ranking.unit_points, dtype=np.float64),
                np.asarray(stage_qmc_ranking.scores, dtype=np.float64).tolist(),
                strict=True,
            )
        }
        polish_count = _fractional_stage_count(
            int(hull_count + domain_count),
            float(params.stage3_refine_fraction),
        )
        polish_simplex_ids = np.asarray(
            stage_qmc_ranking.simplex_ids[: min(polish_count, stage_qmc_ranking.simplex_ids.shape[0])],
            dtype=np.int64,
        )
        polish_scoring_base = self._subset_scoring_by_simplex_ids(merged_stage0, polish_simplex_ids)
        theta_dim = (
            int(polish_scoring_base.simplex_vertices.shape[2])
            if polish_scoring_base.simplex_vertices.ndim == 3 and polish_scoring_base.simplex_vertices.shape[0] > 0
            else int(hull_collection.simplex_vertices.shape[2])
        )
        if polish_scoring_base.simplex_ids.shape[0] > 0:
            polish_repr_points_rows: list[np.ndarray] = []
            polish_repr_scores_rows: list[float] = []
            for row_idx, simplex_id in enumerate(polish_scoring_base.simplex_ids.tolist()):
                if int(simplex_id) in stage_qmc_map:
                    point, score = stage_qmc_map[int(simplex_id)]
                    polish_repr_points_rows.append(np.asarray(point, dtype=np.float64))
                    polish_repr_scores_rows.append(float(score))
                else:
                    polish_repr_points_rows.append(
                        np.asarray(polish_scoring_base.repr_points[row_idx], dtype=np.float64)
                    )
                    polish_repr_scores_rows.append(float(polish_scoring_base.repr_scores[row_idx]))
            polish_repr_points = np.asarray(polish_repr_points_rows, dtype=np.float64)
            polish_repr_scores = np.asarray(polish_repr_scores_rows, dtype=np.float64)
        else:
            polish_repr_points = np.empty((0, theta_dim), dtype=np.float64)
            polish_repr_scores = np.empty((0,), dtype=np.float64)
        polish_scoring = self._clone_scoring_with_repr(
            polish_scoring_base,
            repr_points=polish_repr_points,
            repr_scores=polish_repr_scores,
        )
        repr_ranking = self._merge_rankings(
            [
                self._ranking_from_scoring(
                    hull_scoring,
                    component_index=int(objective_spec.objective_index),
                ),
                self._ranking_from_scoring(
                    domain_scoring,
                    component_index=int(objective_spec.objective_index),
                ),
            ]
        )
        merged_refined = self._merge_rankings_by_simplex(
            base_scoring=merged_stage0,
            stage1_ranking=stage_qmc_ranking,
            stage2_ranking=None,
            component_index=int(objective_spec.objective_index),
        )
        polish_params = replace(
            params,
            starts_per_simplex_refine=int(params.polish_starts_per_simplex_refine),
            max_iter_refine=int(params.polish_max_iter_refine),
            history_size_refine=int(params.polish_history_size_refine),
            convergence_tol_refine=float(params.polish_convergence_tol_refine),
        )
        polish_ranking = self._refine_scoring(
            scoring=polish_scoring,
            evaluator=evaluator,
            params=polish_params,
            progress_callback=_objective_progress if progress_callback is not None else None,
            random_seed=int(random_seed_base + 2000 + objective_index),
            stage_name="module3_refinement_polish",
            start_strategy="legacy",
        )
        polish_ranking.component_index = int(objective_spec.objective_index)
        polish_ranking = self._apply_domain_score_scale_to_ranking(
            polish_ranking,
            domain_simplex_id_start=domain_simplex_id_start,
            params=params,
        )
        combined_ranking = self._concat_rankings(
            [polish_ranking, stage_qmc_ranking, merged_refined, repr_ranking]
        )
        combined_ranking = self._rerank_ranking_prefix_exact(
            ranking=combined_ranking,
            evaluator=evaluator,
            top_k=int(params.polish_top_k),
        )
        combined_ranking = self._rerank_ranking_prefix_imse(
            ranking=combined_ranking,
            evaluator=evaluator,
            params=params,
            random_seed=int(random_seed_base + 3000 + objective_index + params.imse_probe_seed_offset),
        )
        combined_ranking.component_index = int(objective_spec.objective_index)
        return combined_ranking, {
            "objective_index": int(objective_spec.objective_index),
            "objective_label": str(objective_spec.objective_label),
            "refinement_architecture": "stage_qmc_then_polish",
            "num_hull_simplices": hull_count,
            "num_domain_simplices": domain_count,
            "num_scored_simplices": int(hull_count + domain_count),
            "num_refined_simplices_stage1": 0,
            "num_refined_simplices_stage2": 0,
            "num_refined_simplices_stage3": 0,
            "num_refined_simplices_polish": int(polish_scoring.simplex_vertices.shape[0]),
            "num_stage_qmc_simplices": int(stage_qmc_scoring.simplex_vertices.shape[0]),
            "num_stage3_qmc_simplices": 0,
            "hierarchical_stage1_refine_fraction": float(params.hierarchical_stage1_refine_fraction),
            "stage_qmc_keep_fraction": float(params.hierarchical_stage1_refine_fraction),
            "stage_qmc_sample_count": int(params.stage3_qmc_sample_count),
            "stage_qmc_chunk_size": int(params.stage3_qmc_chunk_size),
            "stage3_qmc_top_k": 0,
            "stage3_qmc_sample_count": int(params.stage3_qmc_sample_count),
            "stage3_qmc_chunk_size": int(params.stage3_qmc_chunk_size),
            "num_prefilter_simplices_stage0": int(merged_stage0.simplex_vertices.shape[0]),
            "hull_refine_count": int(hull_refine_count),
            "domain_refine_count": int(domain_refine_count),
            "hierarchical_stage1_top_k": int(params.hierarchical_stage1_top_k),
            "hierarchical_stage2_refine_fraction": float(params.hierarchical_stage2_refine_fraction),
            "hierarchical_stage2_top_k": int(params.hierarchical_stage2_top_k),
            "hierarchical_stage2_actual_pool_size": int(stage_qmc_scoring.simplex_vertices.shape[0]),
            "stage3_refine_fraction": float(params.stage3_refine_fraction),
            "polish_refine_fraction_of_all_simplices": float(params.stage3_refine_fraction),
        }

    def _search_objective_candidates(
        self,
        *,
        objective_spec: _ObjectiveSpec,
        hull_collection: _SimplexCollection,
        domain_collection: _SimplexCollection,
        domain_simplex_id_start: int,
        evaluator: _TorchAggregatePosteriorVarianceEvaluator | _TorchPosteriorVarianceEvaluator,
        params: _RuntimeSelectorParams,
        progress_callback: ProgressCallback | None,
        objective_index: int,
        objective_count: int,
        random_seed_base: int,
    ) -> tuple[_ComponentCandidateRanking, dict[str, Any]]:
        if params.refinement_architecture == "legacy_full_refine":
            return self._search_objective_candidates_legacy(
                objective_spec=objective_spec,
                hull_collection=hull_collection,
                domain_collection=domain_collection,
                domain_simplex_id_start=domain_simplex_id_start,
                evaluator=evaluator,
                params=params,
                progress_callback=progress_callback,
                objective_index=objective_index,
                objective_count=objective_count,
                random_seed_base=random_seed_base,
            )
        return self._search_objective_candidates_hierarchical(
            objective_spec=objective_spec,
            hull_collection=hull_collection,
            domain_collection=domain_collection,
            domain_simplex_id_start=domain_simplex_id_start,
            evaluator=evaluator,
            params=params,
            progress_callback=progress_callback,
            objective_index=objective_index,
            objective_count=objective_count,
            random_seed_base=random_seed_base,
        )

    def select_next_batch(
        self,
        config: ValidationRuntimeConfig,
        module3_input: Module3ContinuousInput,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> SelectionResult:
        continuous_state = module3_input.continuous_state
        expected_batch_size = int(config.sampling.batch_size)
        params = self._resolve_runtime_params(config)
        objective_specs, objective_details = self._resolve_objective_specs(module3_input, params)
        train_unit_thetas = np.asarray(continuous_state.train_unit_thetas, dtype=np.float64)
        qmc_pool_enabled = (
            int(params.acquisition_qmc_pool_count) > 0
            and params.objective_mode != "band_partitioned_posterior_variance"
        )

        if (
            params.objective_mode == "band_partitioned_posterior_variance"
            and expected_batch_size != len(objective_specs)
        ):
            raise ValueError(
                "band_partitioned_posterior_variance requires sampling.batch_size to match "
                f"the number of configured bands ({len(objective_specs)}), got {expected_batch_size}."
            )

        if progress_callback is not None:
            progress_callback("module3_objective_prepare", 1, 2)

        hull_collection: _SimplexCollection | None = None
        domain_collection: _SimplexCollection | None = None
        domain_simplex_id_start = 0
        num_hull_simplices = 0
        num_domain_simplices = 0
        if not qmc_pool_enabled:
            hull_geometry = self._build_shared_hull_geometry(train_unit_thetas)
            if progress_callback is not None:
                progress_callback("module3_hull_geometry", 1, 1)

            hull_collection = _SimplexCollection(
                name="hull",
                simplex_vertices=np.asarray(hull_geometry.simplex_vertices, dtype=np.float64),
                simplex_ids=np.arange(hull_geometry.simplex_vertices.shape[0], dtype=np.int64),
            )
            domain_simplex_id_start = int(hull_collection.simplex_vertices.shape[0])
            domain_collection = self._build_domain_collection(
                train_unit_thetas=train_unit_thetas,
                hull_geometry=hull_geometry,
                params=params,
            )
            num_hull_simplices = int(hull_collection.simplex_vertices.shape[0])
            num_domain_simplices = int(domain_collection.simplex_vertices.shape[0])
            if progress_callback is not None:
                progress_callback("module3_domain_geometry", 1, 1)

        device = _resolve_torch_device(config)
        # Use float64 on CUDA too. The posterior-variance objective repeatedly solves
        # small triangular systems; float32 CUBLAS STRSM has proven unstable for some
        # Quijote active-learning states, while the CPU path already uses float64.
        dtype = torch.float64
        component_evaluators = self._build_component_evaluators(
            module3_input,
            device=device,
            dtype=dtype,
            variance_floor=params.variance_floor,
        )
        if progress_callback is not None:
            progress_callback("module3_objective_prepare", 2, 2)

        objective_evaluators: list[_TorchAggregatePosteriorVarianceEvaluator] = [
            self._build_aggregate_evaluator_from_components(
                component_evaluators,
                component_weights=np.asarray(spec.component_weights, dtype=np.float64),
                device=device,
                dtype=dtype,
                variance_floor=params.variance_floor,
            )
            for spec in objective_specs
        ]
        density_weight_enabled = (
            float(params.acquisition_density_weight_power) > 0.0
            and float(params.acquisition_density_weight_floor) < 1.0
        )
        if density_weight_enabled:
            objective_evaluators = [
                _DensityWeightedPosteriorVarianceEvaluator(
                    evaluator,
                    power=float(params.acquisition_density_weight_power),
                    floor=float(params.acquisition_density_weight_floor),
                )
                for evaluator in objective_evaluators
            ]
        z2_csst_bias_details: dict[str, Any] = {"enabled": False}
        z2_csst_bias_model = continuous_state.metadata.get("z2_csst_bias_model")
        if z2_csst_bias_model is not None and bool(getattr(z2_csst_bias_model, "enabled", True)):
            bias_norm_mode = str(getattr(z2_csst_bias_model, "normalization", "p95")).strip().lower()
            bias_probe_count = int(max(1, getattr(z2_csst_bias_model, "normalization_probe_count", 128)))
            bias_weight = float(max(0.0, getattr(z2_csst_bias_model, "bias_weight", 1.0)))
            bias_score_mode = str(getattr(z2_csst_bias_model, "score_mode", "variance_bias")).strip().lower()
            if bias_score_mode not in {"variance_bias", "bias_only"}:
                bias_score_mode = "variance_bias"
            theta_dim = int(train_unit_thetas.shape[1])
            bias_probe_points = self._generate_imse_probe_points(
                theta_dim=theta_dim,
                probe_count=bias_probe_count,
                random_seed=int(config.random_seed + 91000 + 37 * int(module3_input.iteration_index)),
            )
            uncertainty_probe = self._evaluate_points(
                evaluator=objective_evaluators[0],
                points=bias_probe_points,
                chunk_size=params.chunk_size,
            )
            bias_probe = np.asarray(
                z2_csst_bias_model.bias_for_unit(bias_probe_points),
                dtype=np.float64,
            ).reshape(-1)
            uncertainty_scale = _positive_score_scale(uncertainty_probe, mode=bias_norm_mode)
            bias_scale = _positive_score_scale(bias_probe, mode=bias_norm_mode)
            objective_evaluators = [
                _BiasAugmentedPosteriorVarianceEvaluator(
                    evaluator,
                    bias_model=z2_csst_bias_model,
                    uncertainty_scale=uncertainty_scale,
                    bias_scale=bias_scale,
                    bias_weight=bias_weight,
                    normalization=bias_norm_mode,
                    score_mode=bias_score_mode,
                )
                for evaluator in objective_evaluators
            ]
            score_formula = (
                "bias_weight*B/B_scale"
                if bias_score_mode == "bias_only"
                else "sqrt((U/U_scale)^2 + (bias_weight*B/B_scale)^2)"
            )
            z2_csst_bias_details = {
                "enabled": True,
                "score_mode": str(bias_score_mode),
                "score_formula": score_formula,
                "normalization": bias_norm_mode,
                "normalization_probe_count": int(bias_probe_points.shape[0]),
                "uncertainty_scale": float(uncertainty_scale),
                "bias_scale": float(bias_scale),
                "bias_weight": float(bias_weight),
                "uncertainty_probe_p68": float(np.percentile(uncertainty_probe, 68.0)) if uncertainty_probe.size else 0.0,
                "bias_probe_p50": float(np.percentile(bias_probe, 50.0)) if bias_probe.size else 0.0,
                "bias_probe_p68": float(np.percentile(bias_probe, 68.0)) if bias_probe.size else 0.0,
                "bias_probe_p95": float(np.percentile(bias_probe, 95.0)) if bias_probe.size else 0.0,
                "bias_statistic": str(
                    getattr(
                        z2_csst_bias_model,
                        "bias_weight_details",
                        {},
                    ).get("bias_statistic", "weighted_mean_relative_error_over_k")
                ),
                "bias_weight_details": dict(getattr(z2_csst_bias_model, "bias_weight_details", {})),
            }
        p68_risk_component_weights, p68_risk_details = self._resolve_p68_set_rerank_risk_component_weights(
            module3_input,
            params,
        )
        p68_set_rerank_evaluator: (
            _TorchAggregatePosteriorVarianceEvaluator
            | _TorchPosteriorVarianceEvaluator
            | _DensityWeightedPosteriorVarianceEvaluator
        ) = objective_evaluators[0]
        if p68_risk_component_weights is not None:
            p68_set_rerank_evaluator = self._build_aggregate_evaluator_from_components(
                component_evaluators,
                component_weights=np.asarray(p68_risk_component_weights, dtype=np.float64),
                device=device,
                dtype=dtype,
                variance_floor=params.variance_floor,
            )
        p68_band_risk_details: dict[str, Any] = {"mode": "disabled"}
        p68_band_risk_evaluators: list[dict[str, Any]] = []
        if str(params.acquisition_p68_set_rerank_risk_mode) == "balanced_exceedance":
            p68_band_specs, p68_band_risk_details = self._resolve_p68_set_rerank_band_component_specs(
                module3_input,
                params,
            )
            for band_spec in p68_band_specs:
                p68_band_risk_evaluators.append(
                    {
                        "band_index": int(band_spec["band_index"]),
                        "band_label": str(band_spec["band_label"]),
                        "band_weight": float(band_spec["band_weight"]),
                        "evaluator": self._build_aggregate_evaluator_from_components(
                            component_evaluators,
                            component_weights=np.asarray(
                                band_spec["component_weights"],
                                dtype=np.float64,
                            ),
                            device=device,
                            dtype=dtype,
                            variance_floor=params.variance_floor,
                        ),
                    }
                )
        objective_rankings: list[_ComponentCandidateRanking] = []
        objective_summaries: list[dict[str, Any]] = []
        random_seed_base = int(config.random_seed + 10000 * max(1, module3_input.iteration_index))
        if qmc_pool_enabled:
            if progress_callback is not None:
                progress_callback("module3_qmc_candidate_pool", 1, 1)
            qmc_seed_base = int(config.random_seed) if bool(params.acquisition_qmc_pool_static_seed) else random_seed_base
            ranking = self._build_qmc_candidate_pool_ranking(
                evaluator=objective_evaluators[0],
                train_unit_thetas=train_unit_thetas,
                params=params,
                random_seed=int(qmc_seed_base + params.acquisition_qmc_pool_seed_offset),
            )
            objective_rankings.append(ranking)
            objective_summaries.append(
                {
                    "objective_index": 0,
                    "objective_label": "qmc_candidate_pool",
                    "refinement_architecture": "qmc_candidate_pool",
                    "num_hull_simplices": int(num_hull_simplices),
                    "num_domain_simplices": int(num_domain_simplices),
                    "num_scored_simplices": int(params.acquisition_qmc_pool_count),
                    "num_refined_simplices_stage1": 0,
                    "num_refined_simplices_stage2": 0,
                    "num_refined_simplices_polish": 0,
                    "num_prefilter_simplices_stage0": 0,
                    "hull_refine_count": 0,
                    "domain_refine_count": 0,
                    "qmc_pool_count": int(params.acquisition_qmc_pool_count),
                }
            )
        else:
            assert hull_collection is not None
            assert domain_collection is not None
            for objective_index, (objective_spec, evaluator) in enumerate(
                zip(objective_specs, objective_evaluators, strict=True)
            ):
                ranking, summary = self._search_objective_candidates(
                    objective_spec=objective_spec,
                    hull_collection=hull_collection,
                    domain_collection=domain_collection,
                    domain_simplex_id_start=domain_simplex_id_start,
                    evaluator=evaluator,
                    params=params,
                    progress_callback=progress_callback,
                    objective_index=objective_index,
                    objective_count=len(objective_specs),
                    random_seed_base=random_seed_base,
                )
                objective_rankings.append(ranking)
                objective_summaries.append(summary)

        if progress_callback is not None:
            progress_callback("module3_finalize", 1, 1)

        loo_guard_details: dict[str, Any] = {
            "enabled": False,
            "stage": str(params.acquisition_p68_loo_guard_stage),
        }
        if params.objective_mode == "band_partitioned_posterior_variance":
            selected_unit, _, selected_source_band = _resolve_unique_component_points(
                objective_rankings,
                duplicate_tol=params.duplicate_tol,
            )
            selected_source_pc = np.full((selected_unit.shape[0],), -1, dtype=np.int64)
            selected_scores = np.asarray(
                [
                    self._evaluate_points(
                        evaluator=objective_evaluators[int(source_band)],
                        points=selected_unit[row_idx : row_idx + 1],
                        chunk_size=params.chunk_size,
                    )[0]
                    for row_idx, source_band in enumerate(selected_source_band.tolist())
                ],
                dtype=np.float64,
            )
            selected_band_labels = [
                str(objective_specs[int(source_band)].objective_label)
                for source_band in selected_source_band.tolist()
            ]
        else:
            combined_ranking = self._concat_rankings(objective_rankings)
            combined_ranking = self._rerank_ranking_prefix_spacefill(
                ranking=combined_ranking,
                train_unit_thetas=train_unit_thetas,
                params=params,
            )
            combined_ranking = self._guard_ranking_prefix_spacefill(
                ranking=combined_ranking,
                train_unit_thetas=train_unit_thetas,
                params=params,
            )
            combined_ranking = self._guard_ranking_prefix_spacefill_cd_nonworse(
                ranking=combined_ranking,
                train_unit_thetas=train_unit_thetas,
                params=params,
            )
            combined_ranking = self._tiebreak_ranking_prefix_spacefill(
                ranking=combined_ranking,
                train_unit_thetas=train_unit_thetas,
                params=params,
            )
            loo_guard_stage = str(params.acquisition_p68_loo_guard_stage)
            pre_loo_guard_details: dict[str, Any] = {"enabled": False, "stage": "pre"}
            post_loo_guard_details: dict[str, Any] = {"enabled": False, "stage": "post"}
            if loo_guard_stage in {"pre", "both"}:
                combined_ranking, pre_loo_guard_details = self._guard_ranking_prefix_loo_residual(
                    ranking=combined_ranking,
                    module3_input=module3_input,
                    fallback_component_weights=np.asarray(objective_specs[0].component_weights, dtype=np.float64),
                    train_unit_thetas=train_unit_thetas,
                    params=params,
                )
            combined_ranking = self._rerank_ranking_prefix_p68_set_aware(
                ranking=combined_ranking,
                evaluator=p68_set_rerank_evaluator,
                band_risk_evaluators=p68_band_risk_evaluators,
                module3_input=module3_input,
                train_unit_thetas=train_unit_thetas,
                params=params,
                random_seed=int(random_seed_base + params.imse_probe_seed_offset + 68000),
            )
            if loo_guard_stage in {"post", "both"}:
                combined_ranking, post_loo_guard_details = self._guard_ranking_prefix_loo_residual(
                    ranking=combined_ranking,
                    module3_input=module3_input,
                    fallback_component_weights=np.asarray(objective_specs[0].component_weights, dtype=np.float64),
                    train_unit_thetas=train_unit_thetas,
                    params=params,
                )
            loo_guard_details = {
                "enabled": bool(
                    pre_loo_guard_details.get("enabled") or post_loo_guard_details.get("enabled")
                ),
                "stage": str(loo_guard_stage),
                "pre": dict(pre_loo_guard_details),
                "post": dict(post_loo_guard_details),
            }
            selected_unit, _, selected_source_pc = _resolve_unique_points_from_ranking(
                combined_ranking,
                duplicate_tol=params.duplicate_tol,
                num_points=expected_batch_size,
            )
            selected_scores = self._evaluate_points(
                evaluator=objective_evaluators[0],
                points=selected_unit,
                chunk_size=params.chunk_size,
            )
            selected_source_band = np.full((selected_unit.shape[0],), -1, dtype=np.int64)
            selected_band_labels: list[str] = []

        selected_raw = denormalize_theta_batch(selected_unit, continuous_state.theta_bounds)
        total_refined_stage1 = int(
            sum(int(summary["num_refined_simplices_stage1"]) for summary in objective_summaries)
        )
        total_refined_stage2 = int(
            sum(int(summary["num_refined_simplices_stage2"]) for summary in objective_summaries)
        )
        total_refined_stage3 = int(
            sum(int(summary.get("num_refined_simplices_stage3", 0)) for summary in objective_summaries)
        )
        total_refined_polish = int(
            sum(int(summary.get("num_refined_simplices_polish", 0)) for summary in objective_summaries)
        )
        total_stage3_qmc_simplices = int(
            sum(int(summary.get("num_stage3_qmc_simplices", 0)) for summary in objective_summaries)
        )
        metadata: dict[str, Any] = {
            "mode": "qmc_candidate_pool" if qmc_pool_enabled else f"{str(params.coverage_mode)}_hierarchical_refinement",
            "objective": str(params.objective_mode),
            "refinement_architecture": str(params.refinement_architecture),
            "effective_refinement_architecture": (
                "qmc_candidate_pool" if qmc_pool_enabled else str(params.refinement_architecture)
            ),
            "device": str(device),
            "coverage_mode": str(params.coverage_mode),
            "repr_score_mode": str(params.repr_score_mode),
            "domain_support_scheme": str(params.domain_support_scheme),
            "domain_support_point_count": int(params.domain_support_point_count),
            "domain_support_fanout": int(params.domain_support_fanout),
            "domain_score_scale": float(params.domain_score_scale),
            "acquisition_density_weight_power": float(params.acquisition_density_weight_power),
            "acquisition_density_weight_floor": float(params.acquisition_density_weight_floor),
            "acquisition_density_weight_enabled": bool(density_weight_enabled),
            "z2_csst_bias_details": dict(z2_csst_bias_details),
            "acquisition_spacefill_rerank_top_k": int(params.acquisition_spacefill_rerank_top_k),
            "acquisition_spacefill_weight": float(params.acquisition_spacefill_weight),
            "acquisition_spacefill_enabled": bool(
                int(params.acquisition_spacefill_rerank_top_k) > 0
                and float(params.acquisition_spacefill_weight) > 0.0
            ),
            "acquisition_spacefill_guard_top_k": int(params.acquisition_spacefill_guard_top_k),
            "acquisition_spacefill_guard_reject_quantile": float(
                params.acquisition_spacefill_guard_reject_quantile
            ),
            "acquisition_spacefill_guard_enabled": bool(
                int(params.acquisition_spacefill_guard_top_k) > 0
                and float(params.acquisition_spacefill_guard_reject_quantile) > 0.0
            ),
            "acquisition_spacefill_tiebreak_top_k": int(params.acquisition_spacefill_tiebreak_top_k),
            "acquisition_spacefill_tiebreak_score_ratio": float(
                params.acquisition_spacefill_tiebreak_score_ratio
            ),
            "acquisition_spacefill_tiebreak_enabled": bool(
                int(params.acquisition_spacefill_tiebreak_top_k) > 0
                and float(params.acquisition_spacefill_tiebreak_score_ratio) > 0.0
            ),
            "acquisition_spacefill_cd_nonworse_top_k": int(params.acquisition_spacefill_cd_nonworse_top_k),
            "acquisition_spacefill_cd_nonworse_tol": float(params.acquisition_spacefill_cd_nonworse_tol),
            "acquisition_spacefill_cd_nonworse_enabled": bool(
                int(params.acquisition_spacefill_cd_nonworse_top_k) > 0
            ),
            "acquisition_p68_set_rerank_top_k": int(params.acquisition_p68_set_rerank_top_k),
            "acquisition_p68_set_rerank_score_ratio": float(
                params.acquisition_p68_set_rerank_score_ratio
            ),
            "acquisition_p68_set_rerank_risk_mode": str(
                params.acquisition_p68_set_rerank_risk_mode
            ),
            "acquisition_p68_set_rerank_band_weights": [
                float(value) for value in params.acquisition_p68_set_rerank_band_weights
            ],
            "acquisition_p68_set_rerank_risk_details": dict(p68_risk_details),
            "acquisition_p68_set_rerank_balanced_band_details": dict(p68_band_risk_details),
            "acquisition_p68_set_rerank_balanced_band_count": int(len(p68_band_risk_evaluators)),
            "acquisition_p68_set_rerank_acq_weight": float(
                params.acquisition_p68_set_rerank_acq_weight
            ),
            "acquisition_p68_set_rerank_p68_weight": float(
                params.acquisition_p68_set_rerank_p68_weight
            ),
            "acquisition_p68_set_rerank_spacefill_weight": float(
                params.acquisition_p68_set_rerank_spacefill_weight
            ),
            "acquisition_p68_set_rerank_boundary_weight": float(
                params.acquisition_p68_set_rerank_boundary_weight
            ),
            "acquisition_p68_set_rerank_boundary_threshold": float(
                params.acquisition_p68_set_rerank_boundary_threshold
            ),
            "acquisition_p68_set_rerank_boundary_target_fraction": float(
                params.acquisition_p68_set_rerank_boundary_target_fraction
            ),
            "acquisition_p68_loo_guard_top_k": int(params.acquisition_p68_loo_guard_top_k),
            "acquisition_p68_loo_guard_score_ratio": float(
                params.acquisition_p68_loo_guard_score_ratio
            ),
            "acquisition_p68_loo_guard_reject_quantile": float(
                params.acquisition_p68_loo_guard_reject_quantile
            ),
            "acquisition_p68_loo_guard_bandwidth": float(
                params.acquisition_p68_loo_guard_bandwidth
            ),
            "acquisition_p68_loo_guard_band_weights": [
                float(value) for value in params.acquisition_p68_loo_guard_band_weights
            ],
            "acquisition_p68_loo_guard_stage": str(params.acquisition_p68_loo_guard_stage),
            "acquisition_p68_loo_guard_details": dict(loo_guard_details),
            "acquisition_p68_loo_guard_enabled": bool(
                int(params.acquisition_p68_loo_guard_top_k) > 0
                and float(params.acquisition_p68_loo_guard_score_ratio) > 0.0
                and float(params.acquisition_p68_loo_guard_reject_quantile) > 0.0
            ),
            "acquisition_p68_set_rerank_enabled": bool(
                int(params.acquisition_p68_set_rerank_top_k) > 0
                and float(params.acquisition_p68_set_rerank_score_ratio) > 0.0
                and (
                    float(params.acquisition_p68_set_rerank_acq_weight) > 0.0
                    or float(params.acquisition_p68_set_rerank_p68_weight) > 0.0
                    or float(params.acquisition_p68_set_rerank_spacefill_weight) > 0.0
                    or float(params.acquisition_p68_set_rerank_boundary_weight) > 0.0
                )
            ),
            "acquisition_qmc_pool_count": int(params.acquisition_qmc_pool_count),
            "acquisition_qmc_pool_seed_offset": int(params.acquisition_qmc_pool_seed_offset),
            "acquisition_qmc_pool_static_seed": bool(params.acquisition_qmc_pool_static_seed),
            "acquisition_qmc_pool_enabled": bool(qmc_pool_enabled),
            "imse_rerank_top_k": int(params.imse_rerank_top_k),
            "imse_probe_count": int(params.imse_probe_count),
            "imse_probe_seed_offset": int(params.imse_probe_seed_offset),
            "imse_rerank_mode": str(params.imse_rerank_mode),
            "imse_quantile": float(params.imse_quantile),
            "imse_quantile_shell_width": float(params.imse_quantile_shell_width),
            "imse_quantile_mean_weight": float(params.imse_quantile_mean_weight),
            "imse_quantile_max_weight": float(params.imse_quantile_max_weight),
            "imse_rerank_enabled": bool(
                int(params.imse_rerank_top_k) > 0 and int(params.imse_probe_count) > 0
            ),
            "objective_count": int(len(objective_specs)),
            "num_hull_simplices": int(num_hull_simplices),
            "num_domain_simplices": int(num_domain_simplices),
            "num_scored_simplices": int(
                sum(int(summary["num_scored_simplices"]) for summary in objective_summaries)
            ),
            "num_refined_simplices": total_refined_stage1,
            "num_refined_simplices_stage1": total_refined_stage1,
            "num_refined_simplices_stage2": total_refined_stage2,
            "num_refined_simplices_stage3": total_refined_stage3,
            "num_refined_simplices_polish": total_refined_polish,
            "num_stage3_qmc_simplices": total_stage3_qmc_simplices,
            "global_top_k": int(params.global_top_k),
            "domain_top_k": int(params.domain_top_k),
            "hull_refine_fraction": float(params.hull_refine_fraction),
            "hull_refine_count": int(
                sum(int(summary["hull_refine_count"]) for summary in objective_summaries)
            ),
            "domain_refine_all": bool(params.domain_refine_all),
            "domain_refine_count": int(
                sum(int(summary["domain_refine_count"]) for summary in objective_summaries)
            ),
            "hierarchical_stage1_refine_fraction": float(params.hierarchical_stage1_refine_fraction),
            "hierarchical_stage1_top_k": int(params.hierarchical_stage1_top_k),
            "hierarchical_stage1_starts_per_simplex_refine": int(params.hierarchical_stage1_starts_per_simplex_refine),
            "hierarchical_stage1_max_iter_refine": int(params.hierarchical_stage1_max_iter_refine),
            "hierarchical_stage1_history_size_refine": int(params.hierarchical_stage1_history_size_refine),
            "hierarchical_stage1_convergence_tol_refine": float(params.hierarchical_stage1_convergence_tol_refine),
            "hierarchical_stage2_refine_fraction": float(params.hierarchical_stage2_refine_fraction),
            "hierarchical_stage2_top_k": int(params.hierarchical_stage2_top_k),
            "hierarchical_stage2_starts_per_simplex_refine": int(params.hierarchical_stage2_starts_per_simplex_refine),
            "hierarchical_stage2_max_iter_refine": int(params.hierarchical_stage2_max_iter_refine),
            "hierarchical_stage2_history_size_refine": int(params.hierarchical_stage2_history_size_refine),
            "hierarchical_stage2_convergence_tol_refine": float(params.hierarchical_stage2_convergence_tol_refine),
            "starts_per_simplex_refine": int(params.starts_per_simplex_refine),
            "max_iter_refine": int(params.max_iter_refine),
            "history_size_refine": int(params.history_size_refine),
            "polish_top_k": int(params.polish_top_k),
            "stage3_refine_fraction": float(params.stage3_refine_fraction),
            "polish_starts_per_simplex_refine": int(params.polish_starts_per_simplex_refine),
            "polish_max_iter_refine": int(params.polish_max_iter_refine),
            "polish_history_size_refine": int(params.polish_history_size_refine),
            "polish_convergence_tol_refine": float(params.polish_convergence_tol_refine),
            "stage3_qmc_top_k": int(params.stage3_qmc_top_k),
            "stage3_qmc_sample_count": int(params.stage3_qmc_sample_count),
            "stage3_qmc_chunk_size": int(params.stage3_qmc_chunk_size),
            "chunk_size": int(params.chunk_size),
            "stage0_chunk_size": int(params.stage0_chunk_size),
            "repr_dirichlet_probe_count": int(params.repr_dirichlet_probe_count),
            "line_search_steps_refine": list(params.line_search_steps_refine),
            "fallback_step_refine": float(params.fallback_step_refine),
            "duplicate_tol": float(params.duplicate_tol),
            "aggregated_pc_count": int(len(continuous_state.gp_models)),
            "selected_source_band": selected_source_band.astype(np.int64).tolist(),
            "selected_source_band_label": list(selected_band_labels),
            "objective_run_summaries": [dict(summary) for summary in objective_summaries],
            "objective_details": objective_details,
        }
        return SelectionResult(
            selected_raw_thetas=selected_raw,
            selected_unit_thetas=selected_unit,
            selected_source_pc=selected_source_pc,
            selected_scores=selected_scores,
            metadata=metadata,
        )
