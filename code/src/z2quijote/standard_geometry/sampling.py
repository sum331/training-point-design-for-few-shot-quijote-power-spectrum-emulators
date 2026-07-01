from __future__ import annotations

import numpy as np

from z2quijote.sampling import latin_hypercube_unit, sobol_unit


def draw_reference_unit(*, count: int, dim: int, seed: int) -> np.ndarray:
    """Fixed reference grid for the bias-field support."""

    return sobol_unit(int(count), int(dim), seed=int(seed))


def draw_design_unit(
    *,
    design_size: int,
    dim: int,
    seed: int,
    index: int,
    sampler: str = "mixed",
) -> np.ndarray:
    """Draw one N-point design from the configured design distribution."""

    name = str(sampler).strip().lower()
    if name == "sobol" or (name == "mixed" and int(index) % 2 == 0):
        return sobol_unit(int(design_size), int(dim), seed=int(seed) + int(index))
    if name == "lhs" or name == "latin_hypercube" or name == "mixed":
        return latin_hypercube_unit(int(design_size), int(dim), seed=int(seed) + int(index))
    raise ValueError(f"unknown standard-geometry design sampler: {sampler!r}.")
