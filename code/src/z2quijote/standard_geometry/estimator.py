from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BiasFieldEstimate:
    bias_mean: np.ndarray
    bias_std: np.ndarray
    bias_se: np.ndarray
    accepted_count: np.ndarray
    accepted_fraction: np.ndarray
    high_confidence: np.ndarray
    usable: np.ndarray

    def as_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "bias_mean": self.bias_mean.astype(np.float64),
            "bias_std": self.bias_std.astype(np.float64),
            "bias_se": self.bias_se.astype(np.float64),
            "accepted_count": self.accepted_count.astype(np.int64),
            "accepted_fraction": self.accepted_fraction.astype(np.float64),
            "high_confidence": self.high_confidence.astype(bool),
            "usable": self.usable.astype(bool),
        }


class BiasAccumulator:
    """Streaming accepted-sample mean/variance accumulator for each reference point."""

    def __init__(self, reference_size: int) -> None:
        count = int(reference_size)
        if count <= 0:
            raise ValueError("reference_size must be positive.")
        self._sum = np.zeros(count, dtype=np.float64)
        self._sum_sq = np.zeros(count, dtype=np.float64)
        self._count = np.zeros(count, dtype=np.int64)
        self._design_count = 0

    @property
    def design_count(self) -> int:
        return int(self._design_count)

    def add(self, bias: np.ndarray, accepted: np.ndarray) -> None:
        values = np.asarray(bias, dtype=np.float64).reshape(-1)
        mask = np.asarray(accepted, dtype=bool).reshape(-1)
        if values.shape != self._sum.shape or mask.shape != self._sum.shape:
            raise ValueError("bias and accepted must match reference_size.")
        finite = mask & np.isfinite(values)
        self._sum[finite] += values[finite]
        self._sum_sq[finite] += values[finite] ** 2
        self._count[finite] += 1
        self._design_count += 1

    def add_indices(self, indices: np.ndarray, bias: np.ndarray) -> None:
        """Add accepted-center bias values without materializing a full reference vector."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        values = np.asarray(bias, dtype=np.float64).reshape(-1)
        if idx.shape[0] != values.shape[0]:
            raise ValueError("indices and bias must have the same length.")
        if idx.size > 0 and (np.min(idx) < 0 or np.max(idx) >= self._sum.shape[0]):
            raise ValueError("indices contain values outside the reference range.")
        finite = np.isfinite(values)
        finite_idx = idx[finite]
        finite_values = values[finite]
        np.add.at(self._sum, finite_idx, finite_values)
        np.add.at(self._sum_sq, finite_idx, finite_values**2)
        np.add.at(self._count, finite_idx, 1)
        self._design_count += 1

    def estimate(self, *, usable_min_count: int = 10, high_confidence_min_count: int = 20) -> BiasFieldEstimate:
        count = self._count.copy()
        mean = np.full_like(self._sum, np.nan, dtype=np.float64)
        std = np.full_like(self._sum, np.nan, dtype=np.float64)
        se = np.full_like(self._sum, np.nan, dtype=np.float64)
        positive = count > 0
        mean[positive] = self._sum[positive] / count[positive]
        multi = count > 1
        variance = np.full_like(self._sum, np.nan, dtype=np.float64)
        variance[multi] = (
            self._sum_sq[multi] - (self._sum[multi] ** 2) / count[multi]
        ) / np.maximum(count[multi] - 1, 1)
        variance[multi] = np.maximum(variance[multi], 0.0)
        variance[positive & ~multi] = 0.0
        std[positive] = np.sqrt(variance[positive])
        se[positive] = std[positive] / np.sqrt(np.maximum(count[positive], 1))
        accepted_fraction = np.zeros_like(self._sum, dtype=np.float64)
        if self._design_count > 0:
            accepted_fraction = count.astype(np.float64) / float(self._design_count)
        usable = count >= int(usable_min_count)
        high_confidence = count >= int(high_confidence_min_count)
        return BiasFieldEstimate(
            bias_mean=mean,
            bias_std=std,
            bias_se=se,
            accepted_count=count,
            accepted_fraction=accepted_fraction,
            high_confidence=high_confidence,
            usable=usable,
        )
