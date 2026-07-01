from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


THETA_NAMES: tuple[str, ...] = ("Omega_m", "Omega_b", "h", "n_s", "sigma_8")


@dataclass(frozen=True, slots=True)
class ParameterSpace:
    name: str
    theta_names: tuple[str, ...]
    theta_bounds: np.ndarray

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ParameterSpace":
        names = tuple(str(item) for item in payload.get("theta_names", THETA_NAMES))
        raw_bounds = payload.get("theta_bounds")
        if not isinstance(raw_bounds, Mapping):
            raise ValueError("parameter_space.theta_bounds must be a mapping.")
        rows: list[tuple[float, float]] = []
        for name in names:
            if name not in raw_bounds:
                raise ValueError(f"theta_bounds is missing {name!r}.")
            pair = raw_bounds[name]
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError(f"theta bound for {name!r} must be a [low, high] pair.")
            low = float(pair[0])
            high = float(pair[1])
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError(f"invalid theta bound for {name!r}: {(low, high)!r}.")
            rows.append((low, high))
        return cls(
            name=str(payload.get("name", "quijote_bsq5")),
            theta_names=names,
            theta_bounds=np.asarray(rows, dtype=np.float64),
        )

    @property
    def dim(self) -> int:
        return int(self.theta_bounds.shape[0])

    def normalize(self, theta_raw: np.ndarray, *, clip: bool = True) -> np.ndarray:
        raw = _coerce_theta(theta_raw, self.dim)
        span = np.maximum(self.theta_bounds[:, 1] - self.theta_bounds[:, 0], 1.0e-12)
        unit = (raw - self.theta_bounds[:, 0][None, :]) / span[None, :]
        return np.clip(unit, 0.0, 1.0) if clip else unit.astype(np.float64)

    def denormalize(self, theta_unit: np.ndarray) -> np.ndarray:
        unit = _coerce_theta(theta_unit, self.dim)
        span = np.maximum(self.theta_bounds[:, 1] - self.theta_bounds[:, 0], 1.0e-12)
        return self.theta_bounds[:, 0][None, :] + np.clip(unit, 0.0, 1.0) * span[None, :]


def _coerce_theta(theta: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != int(dim):
        raise ValueError(f"theta array must have shape [N,{dim}], got {arr.shape}.")
    if np.any(~np.isfinite(arr)):
        raise ValueError("theta array contains non-finite values.")
    return arr.astype(np.float64, copy=False)


def nearest_distance(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    candidates = np.asarray(x, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if candidates.ndim != 2 or ref.ndim != 2 or candidates.shape[1] != ref.shape[1]:
        raise ValueError("nearest_distance expects aligned 2D arrays.")
    if ref.shape[0] == 0:
        return np.full((candidates.shape[0],), np.inf, dtype=np.float64)
    out = np.empty((candidates.shape[0],), dtype=np.float64)
    chunk = 2048
    for start in range(0, candidates.shape[0], chunk):
        block = candidates[start : start + chunk]
        dist2 = np.sum((block[:, None, :] - ref[None, :, :]) ** 2, axis=2)
        out[start : start + chunk] = np.sqrt(np.min(dist2, axis=1))
    return out


def boundary_distance(theta_unit: np.ndarray) -> np.ndarray:
    unit = np.asarray(theta_unit, dtype=np.float64)
    if unit.ndim != 2:
        raise ValueError("theta_unit must be 2D.")
    return np.min(np.minimum(unit, 1.0 - unit), axis=1)
