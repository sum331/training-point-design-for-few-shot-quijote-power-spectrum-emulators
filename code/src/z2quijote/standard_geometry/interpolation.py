from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class ReliabilityWeightedLocalInterpolator:
    theta_unit: np.ndarray
    bias: np.ndarray
    accepted_count: np.ndarray
    min_count: int = 10
    high_confidence_count: int = 20
    neighbors: int = 96
    fallback_neighbors: int = 160
    _tree: cKDTree = field(init=False, repr=False)
    _global_fallback: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta_unit, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64).reshape(-1)
        count = np.asarray(self.accepted_count, dtype=np.int64).reshape(-1)
        if theta.ndim != 2 or theta.shape[0] != bias.shape[0] or bias.shape != count.shape:
            raise ValueError("theta_unit, bias, and accepted_count must have aligned first dimension.")
        usable = np.isfinite(bias) & (count >= int(self.min_count))
        if int(np.count_nonzero(usable)) <= 0:
            raise ValueError("interpolator requires at least one usable support point.")
        self.theta_unit = theta[usable]
        self.bias = bias[usable]
        self.accepted_count = count[usable]
        self._tree = cKDTree(self.theta_unit)
        self._global_fallback = float(np.nanmedian(self.bias))

    def predict(self, theta_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(theta_unit, dtype=np.float64)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[1] != self.theta_unit.shape[1]:
            raise ValueError("theta_unit query must match fitted dimension.")
        values = np.empty(query.shape[0], dtype=np.float64)
        confidence = np.empty(query.shape[0], dtype=np.float64)
        for index, row in enumerate(query):
            values[index], confidence[index] = self._predict_one(row)
        return values, confidence

    def _predict_one(self, row: np.ndarray) -> tuple[float, float]:
        usable_count = self.theta_unit.shape[0]
        k = min(int(self.neighbors), usable_count)
        distances, indices = self._tree.query(row, k=k)
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        high = self.accepted_count[indices] >= int(self.high_confidence_count)
        if int(np.count_nonzero(high)) < min(32, indices.shape[0]) and usable_count > k:
            k2 = min(int(self.fallback_neighbors), usable_count)
            distances, indices = self._tree.query(row, k=k2)
            distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
            indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        if indices.shape[0] == 0:
            return self._global_fallback, 0.0
        bandwidth = float(np.max(distances))
        if not np.isfinite(bandwidth) or bandwidth <= 1.0e-12:
            exact = distances <= 1.0e-12
            return float(np.mean(self.bias[indices[exact]])), 1.0
        reliability = np.minimum(1.0, self.accepted_count[indices].astype(np.float64) / float(self.high_confidence_count))
        weights = reliability * np.exp(-0.5 * (distances / bandwidth) ** 2)
        total = float(np.sum(weights))
        if total <= 0.0 or not np.isfinite(total):
            return self._global_fallback, 0.0
        value = float(np.sum(weights * self.bias[indices]) / total)
        effective_n = float(total**2 / max(float(np.sum(weights**2)), 1.0e-30))
        confidence = float(min(1.0, effective_n / float(max(1, min(self.neighbors, self.theta_unit.shape[0])))))
        return value, confidence
