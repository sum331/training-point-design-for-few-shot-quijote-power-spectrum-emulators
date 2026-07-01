from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from scipy.stats import qmc


def sobol_unit(count: int, dim: int, *, seed: int, skip: int = 0) -> np.ndarray:
    if count <= 0:
        return np.empty((0, int(dim)), dtype=np.float64)
    total = int(count + max(0, skip))
    power = int(np.ceil(np.log2(max(1, total))))
    sampler = qmc.Sobol(d=int(dim), scramble=True, seed=int(seed))
    sample = sampler.random_base2(power)
    return np.asarray(sample[int(skip) : int(skip) + int(count)], dtype=np.float64)


def latin_hypercube_unit(count: int, dim: int, *, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, int(dim)), dtype=np.float64)
    sampler = qmc.LatinHypercube(d=int(dim), seed=int(seed))
    return np.asarray(sampler.random(int(count)), dtype=np.float64)


def theta_rows_key(theta_unit: np.ndarray, *, decimals: int = 12) -> list[tuple[float, ...]]:
    arr = np.asarray(theta_unit, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("theta_unit must be 2D.")
    rounded = np.round(arr, int(decimals))
    return [tuple(float(x) for x in row) for row in rounded]


def digest_theta(theta_unit: np.ndarray, *, decimals: int = 12) -> str:
    rounded = np.round(np.asarray(theta_unit, dtype=np.float64), int(decimals))
    payload = rounded.astype("<f8", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()


def unique_unit_rows(
    theta_unit: np.ndarray,
    *,
    decimals: int = 12,
    exclude: Iterable[tuple[float, ...]] | None = None,
) -> np.ndarray:
    seen: set[tuple[float, ...]] = set(exclude or [])
    rows: list[np.ndarray] = []
    for row, key in zip(np.asarray(theta_unit, dtype=np.float64), theta_rows_key(theta_unit, decimals=decimals)):
        if key in seen:
            continue
        seen.add(key)
        rows.append(row.astype(np.float64))
    if not rows:
        return np.empty((0, np.asarray(theta_unit).shape[1]), dtype=np.float64)
    return np.vstack(rows).astype(np.float64)


def draw_disjoint_sobol(
    *,
    count: int,
    dim: int,
    seed: int,
    exclude: set[tuple[float, ...]],
    decimals: int,
    skip: int = 0,
) -> tuple[np.ndarray, set[tuple[float, ...]]]:
    if count <= 0:
        return np.empty((0, int(dim)), dtype=np.float64), set(exclude)
    selected: list[np.ndarray] = []
    seen = set(exclude)
    cursor = int(skip)
    attempts = 0
    while len(selected) < int(count):
        needed = int(count) - len(selected)
        draw_count = max(needed * 4, 128)
        batch = sobol_unit(draw_count, dim, seed=seed + attempts, skip=cursor)
        for row, key in zip(batch, theta_rows_key(batch, decimals=decimals)):
            if key in seen:
                continue
            seen.add(key)
            selected.append(row.astype(np.float64))
            if len(selected) >= int(count):
                break
        cursor += draw_count
        attempts += 1
        if attempts > 64 and len(selected) < int(count):
            fallback = latin_hypercube_unit(max(needed * 8, 512), dim, seed=seed + 10000 + attempts)
            for row, key in zip(fallback, theta_rows_key(fallback, decimals=decimals)):
                if key in seen:
                    continue
                seen.add(key)
                selected.append(row.astype(np.float64))
                if len(selected) >= int(count):
                    break
        if attempts > 128 and len(selected) < int(count):
            raise RuntimeError("could not draw enough disjoint Sobol/LHS points.")
    return np.vstack(selected).astype(np.float64), seen
