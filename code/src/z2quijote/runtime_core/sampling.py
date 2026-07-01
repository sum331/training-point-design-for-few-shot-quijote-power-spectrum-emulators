"""Sampling helpers for Sobol initialization, LHS validation, and candidate pools."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from z2quijote.runtime_core.config import denormalize_theta_batch, normalize_theta_batch, theta_bounds_as_array


def generate_unit_sobol_samples(
    theta_dim: int,
    sample_size: int,
    random_seed: int,
    *,
    scramble: bool = True,
) -> np.ndarray:
    theta_dim = max(1, int(theta_dim))
    sample_size = max(1, int(sample_size))
    sobol = qmc.Sobol(d=theta_dim, scramble=bool(scramble), seed=int(random_seed))
    return np.asarray(sobol.random(sample_size), dtype=np.float64)


def generate_unit_latin_hypercube_samples(
    theta_dim: int,
    sample_size: int,
    random_seed: int,
) -> np.ndarray:
    theta_dim = max(1, int(theta_dim))
    sample_size = max(1, int(sample_size))
    sampler = qmc.LatinHypercube(d=theta_dim, scramble=True, seed=int(random_seed))
    return np.asarray(sampler.random(n=sample_size), dtype=np.float64)


def generate_sobol_thetas(
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray,
    sample_size: int,
    random_seed: int,
    *,
    scramble: bool = True,
) -> np.ndarray:
    bounds = theta_bounds_as_array(theta_bounds)
    unit = generate_unit_sobol_samples(
        bounds.shape[0],
        sample_size,
        random_seed,
        scramble=scramble,
    )
    return denormalize_theta_batch(unit, bounds)


def generate_test_set_thetas(
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray,
    sample_size: int,
    random_seed: int,
) -> np.ndarray:
    bounds = theta_bounds_as_array(theta_bounds)
    unit = generate_unit_latin_hypercube_samples(bounds.shape[0], sample_size, random_seed)
    return denormalize_theta_batch(unit, bounds)


def generate_candidate_thetas(
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray,
    sample_size: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = theta_bounds_as_array(theta_bounds)
    unit = generate_unit_sobol_samples(bounds.shape[0], sample_size, random_seed, scramble=True)
    raw = denormalize_theta_batch(unit, bounds)
    return raw, unit


def sample_cloud_thetas(
    center_theta: np.ndarray,
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray,
    *,
    cloud_size: int,
    radius_scale: float = 0.05,
    random_seed: int = 0,
) -> np.ndarray:
    center = np.asarray(center_theta, dtype=np.float64).reshape(-1)
    bounds = theta_bounds_as_array(theta_bounds)
    if center.shape[0] != bounds.shape[0]:
        raise ValueError(
            f"center_theta must have length {bounds.shape[0]}, got {center.shape[0]}."
        )
    cloud_size = max(1, int(cloud_size))
    radius_scale = float(max(1.0e-6, radius_scale))
    span = bounds[:, 1] - bounds[:, 0]
    rng = np.random.default_rng(int(random_seed))
    jitter = rng.normal(loc=0.0, scale=radius_scale, size=(cloud_size, center.shape[0]))
    cloud = center[None, :] + jitter * span[None, :]
    return np.clip(cloud, bounds[:, 0], bounds[:, 1]).astype(np.float64)


def unique_rows(array: np.ndarray, *, decimals: int = 12) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"array must be 2D, got {arr.shape}.")
    rounded = np.round(arr, decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return arr[np.sort(idx)]


def filter_novel_points(
    candidate_unit_thetas: np.ndarray,
    existing_unit_thetas: np.ndarray,
    *,
    min_distance: float,
) -> np.ndarray:
    candidates = np.asarray(candidate_unit_thetas, dtype=np.float64)
    existing = np.asarray(existing_unit_thetas, dtype=np.float64)
    if candidates.ndim != 2:
        raise ValueError(f"candidate_unit_thetas must be 2D, got {candidates.shape}.")
    if existing.size == 0:
        return np.ones((candidates.shape[0],), dtype=bool)
    if existing.ndim != 2:
        raise ValueError(f"existing_unit_thetas must be 2D, got {existing.shape}.")
    distances = cdist(candidates, existing)
    return np.min(distances, axis=1) > float(max(0.0, min_distance))


def ensure_2d_theta_batch(
    theta_batch: np.ndarray,
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray,
    *,
    input_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(theta_batch, dtype=np.float64)
    bounds = theta_bounds_as_array(theta_bounds)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != bounds.shape[0]:
        raise ValueError(f"theta_batch must have shape [N,{bounds.shape[0]}], got {arr.shape}.")
    if str(input_space).strip().lower() == "unit":
        unit = np.clip(arr, 0.0, 1.0).astype(np.float64)
        raw = denormalize_theta_batch(unit, bounds)
        return raw, unit
    raw = arr.astype(np.float64)
    unit = normalize_theta_batch(raw, bounds)
    return raw, unit
