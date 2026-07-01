from __future__ import annotations

import numpy as np


def density_from_bias(
    bias: np.ndarray,
    *,
    alpha: float = 1.0,
    epsilon: float = 1.0e-12,
    clip_quantile: float | None = 0.95,
) -> np.ndarray:
    values = np.asarray(bias, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("density_from_bias requires at least one finite value.")
    work = values.copy()
    fill = float(np.nanmedian(work[finite]))
    work[~finite] = fill
    if clip_quantile is not None:
        q = float(np.clip(clip_quantile, 0.0, 1.0))
        cap = float(np.quantile(work[np.isfinite(work)], q))
        work = np.minimum(work, cap)
    score = np.maximum(work, 0.0) + float(epsilon)
    score = score ** float(alpha)
    total = float(np.sum(score))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("bias scores cannot be normalized into a finite density.")
    return (score / total).astype(np.float64)
