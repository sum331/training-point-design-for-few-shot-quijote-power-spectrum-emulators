from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, QhullError

from z2quijote.parameter_space import boundary_distance


@dataclass(frozen=True, slots=True)
class StandardGeometryConfig:
    tau_lambda: float = 0.16
    h_quantile_low: float = 0.10
    h_quantile_high: float = 0.90
    kappa_quantile_max: float = 0.90
    boundary_min: float = 0.02


@dataclass(frozen=True, slots=True)
class GeometryThresholds:
    tau_lambda: float
    h_min: float
    h_max: float
    kappa_max: float
    boundary_min: float

    def as_dict(self) -> dict[str, float]:
        return {
            "tau_lambda": float(self.tau_lambda),
            "h_min": float(self.h_min),
            "h_max": float(self.h_max),
            "kappa_max": float(self.kappa_max),
            "boundary_min": float(self.boundary_min),
        }


@dataclass(frozen=True, slots=True)
class GeometryBatch:
    inside_hull: np.ndarray
    valid_simplex: np.ndarray
    barycentric: np.ndarray
    lambda_max_deviation: np.ndarray
    lambda_min: np.ndarray
    simplex_scale: np.ndarray
    simplex_condition: np.ndarray
    boundary_distance: np.ndarray

    @property
    def finite_geometry(self) -> np.ndarray:
        return (
            self.inside_hull
            & self.valid_simplex
            & np.isfinite(self.lambda_max_deviation)
            & np.isfinite(self.simplex_scale)
            & np.isfinite(self.simplex_condition)
            & np.isfinite(self.boundary_distance)
        )


def compute_geometry_batch(reference_unit: np.ndarray, design_unit: np.ndarray) -> GeometryBatch:
    """Compute Delaunay/simplex geometry of reference points relative to one design."""

    ref = _coerce_unit(reference_unit, name="reference_unit")
    design = _coerce_unit(design_unit, name="design_unit")
    if ref.shape[1] != design.shape[1]:
        raise ValueError("reference_unit and design_unit must have the same dimension.")
    count, dim = ref.shape
    simplex_vertices = dim + 1

    inside = np.zeros(count, dtype=bool)
    valid = np.zeros(count, dtype=bool)
    bary = np.full((count, simplex_vertices), np.nan, dtype=np.float64)
    maxdev = np.full(count, np.nan, dtype=np.float64)
    minlam = np.full(count, np.nan, dtype=np.float64)
    scale = np.full(count, np.nan, dtype=np.float64)
    condition = np.full(count, np.nan, dtype=np.float64)
    bdist = boundary_distance(ref).astype(np.float64)

    try:
        tri = Delaunay(design, qhull_options="QJ Qbb Qc Qz Q12")
    except QhullError:
        return GeometryBatch(
            inside_hull=inside,
            valid_simplex=valid,
            barycentric=bary,
            lambda_max_deviation=maxdev,
            lambda_min=minlam,
            simplex_scale=scale,
            simplex_condition=condition,
            boundary_distance=bdist,
        )

    simplex_index = tri.find_simplex(ref)
    inside = simplex_index >= 0
    row_index = np.flatnonzero(inside)
    if row_index.size <= 0:
        return GeometryBatch(
            inside_hull=inside,
            valid_simplex=valid,
            barycentric=bary,
            lambda_max_deviation=maxdev,
            lambda_min=minlam,
            simplex_scale=scale,
            simplex_condition=condition,
            boundary_distance=bdist,
        )

    transform = tri.transform[simplex_index[row_index]]
    first = np.einsum("ijk,ik->ij", transform[:, :dim, :], ref[row_index] - transform[:, dim, :])
    bary_rows = np.c_[first, 1.0 - np.sum(first, axis=1)]
    vertices_index = tri.simplices[simplex_index[row_index]]
    valid_rows = np.all(vertices_index < design.shape[0], axis=1)
    if not np.any(valid_rows):
        return GeometryBatch(
            inside_hull=inside,
            valid_simplex=valid,
            barycentric=bary,
            lambda_max_deviation=maxdev,
            lambda_min=minlam,
            simplex_scale=scale,
            simplex_condition=condition,
            boundary_distance=bdist,
        )

    rows = row_index[valid_rows]
    vertices_index = vertices_index[valid_rows]
    bary_rows = bary_rows[valid_rows]
    vertices = design[vertices_index]
    centers = np.mean(vertices, axis=1)
    edge_scale = np.sqrt(np.mean(np.sum((vertices - centers[:, None, :]) ** 2, axis=2), axis=1))

    cond = np.empty(rows.shape[0], dtype=np.float64)
    for i, vertices_row in enumerate(vertices):
        edge_matrix = (vertices_row[1:] - vertices_row[0]).T
        try:
            cond[i] = float(np.linalg.cond(edge_matrix))
        except np.linalg.LinAlgError:
            cond[i] = np.inf

    center = 1.0 / float(simplex_vertices)
    bary[rows] = bary_rows
    maxdev[rows] = np.max(np.abs(bary_rows - center), axis=1)
    minlam[rows] = np.min(bary_rows, axis=1)
    scale[rows] = edge_scale
    condition[rows] = cond
    valid[rows] = True
    return GeometryBatch(
        inside_hull=inside,
        valid_simplex=valid,
        barycentric=bary,
        lambda_max_deviation=maxdev,
        lambda_min=minlam,
        simplex_scale=scale,
        simplex_condition=condition,
        boundary_distance=bdist,
    )


def thresholds_from_geometry(
    batches: list[GeometryBatch],
    config: StandardGeometryConfig,
) -> GeometryThresholds:
    scales: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    for batch in batches:
        mask = batch.finite_geometry
        if np.any(mask):
            scales.append(np.asarray(batch.simplex_scale[mask], dtype=np.float64))
            conditions.append(np.asarray(batch.simplex_condition[mask], dtype=np.float64))
    if not scales:
        raise ValueError("cannot derive standard-geometry thresholds without valid simplex geometry.")
    h_values = np.concatenate(scales)
    k_values = np.concatenate(conditions)
    k_values = k_values[np.isfinite(k_values)]
    if k_values.size <= 0:
        raise ValueError("cannot derive standard-geometry kappa threshold from non-finite values.")
    return GeometryThresholds(
        tau_lambda=float(config.tau_lambda),
        h_min=float(np.quantile(h_values, float(config.h_quantile_low))),
        h_max=float(np.quantile(h_values, float(config.h_quantile_high))),
        kappa_max=float(np.quantile(k_values, float(config.kappa_quantile_max))),
        boundary_min=float(config.boundary_min),
    )


def accepted_mask(batch: GeometryBatch, thresholds: GeometryThresholds) -> np.ndarray:
    return (
        batch.finite_geometry
        & (batch.lambda_max_deviation <= float(thresholds.tau_lambda))
        & (batch.simplex_scale >= float(thresholds.h_min))
        & (batch.simplex_scale <= float(thresholds.h_max))
        & (batch.simplex_condition <= float(thresholds.kappa_max))
        & (batch.boundary_distance >= float(thresholds.boundary_min))
    )


def _coerce_unit(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"{name} must be non-empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return np.clip(arr, 0.0, 1.0).astype(np.float64, copy=False)
