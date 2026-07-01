"""Isolated PCA+GP surrogate helpers for Quijote BSQ power-spectrum banks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

FloatArray = np.ndarray

QUIJOTE_BSQ_THETA_NAMES: tuple[str, ...] = (
    "Omega_m",
    "Omega_b",
    "h",
    "n_s",
    "sigma_8",
)
QUIJOTE_BSQ5_PARAMETER_SPACE = "quijote_bsq5"
DEFAULT_QUIJOTE_BSQ5_BOUNDS: dict[str, tuple[float, float]] = {
    "Omega_m": (0.10000480, 0.49999857),
    "Omega_b": (0.02000114, 0.07999925),
    "h": (0.50001181, 0.89999719),
    "n_s": (0.80000703, 1.19998813),
    "sigma_8": (0.60000676, 0.99999599),
}


@dataclass(slots=True)
class QuijoteBank:
    raw_thetas: FloatArray
    k_bins: FloatArray
    p_nonlin_batch: FloatArray
    simulation_indices: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QuijoteGPSurrogate:
    theta_bounds: FloatArray
    theta_names: tuple[str, ...]
    k_bins: FloatArray
    pca_model: PCA
    gp_models: list[GaussianProcessRegressor]
    score_mean: FloatArray
    score_std: FloatArray
    power_eps: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(
        self,
        theta_batch: np.ndarray,
        *,
        input_space: str = "raw",
        k_target: np.ndarray | None = None,
        return_std: bool = False,
    ) -> dict[str, np.ndarray]:
        """Predict Quijote real-space P(k) for raw or unit-space 5D theta rows."""

        raw_thetas, unit_thetas = ensure_quijote_theta_batch(
            theta_batch,
            self.theta_bounds,
            input_space=input_space,
        )
        if not self.gp_models:
            raise ValueError("The Quijote surrogate has no fitted GP models.")

        mean_cols: list[np.ndarray] = []
        std_cols: list[np.ndarray] = []
        for gp_model in self.gp_models:
            mean_col, std_col = gp_model.predict(unit_thetas, return_std=True)
            mean_cols.append(np.asarray(mean_col, dtype=np.float64).reshape(-1, 1))
            std_cols.append(np.asarray(std_col, dtype=np.float64).reshape(-1, 1))

        scaled_mean = np.hstack(mean_cols).astype(np.float64)
        scaled_std = np.hstack(std_cols).astype(np.float64)
        pc_mean = scaled_mean * self.score_std.reshape(1, -1) + self.score_mean.reshape(1, -1)
        pc_std = scaled_std * self.score_std.reshape(1, -1)
        log_pk_source = np.asarray(self.pca_model.inverse_transform(pc_mean), dtype=np.float64)
        source_k = np.asarray(self.k_bins, dtype=np.float64).reshape(-1)
        resolved_k = source_k if k_target is None else np.asarray(k_target, dtype=np.float64).reshape(-1)
        if np.any(resolved_k <= 0.0):
            raise ValueError("k_target must be strictly positive.")
        log_pk = _interp_logk_batch(log_pk_source, source_k, resolved_k)
        pk = np.exp(log_pk)
        result = {
            "raw_thetas": raw_thetas,
            "unit_thetas": unit_thetas,
            "k_bins": resolved_k.astype(np.float64),
            "log_pk_mean": log_pk.astype(np.float64),
            "pk_mean": pk.astype(np.float64),
            "pc_mean": pc_mean.astype(np.float64),
        }
        if return_std:
            result["pc_std"] = pc_std.astype(np.float64)
        return result


def quijote_theta_bounds_as_array(
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray | None = None,
) -> FloatArray:
    if theta_bounds is None:
        theta_bounds = DEFAULT_QUIJOTE_BSQ5_BOUNDS
    if isinstance(theta_bounds, np.ndarray):
        bounds = np.asarray(theta_bounds, dtype=np.float64)
        if bounds.shape != (len(QUIJOTE_BSQ_THETA_NAMES), 2):
            raise ValueError(
                "Quijote theta_bounds array must have shape "
                f"({len(QUIJOTE_BSQ_THETA_NAMES)}, 2), got {bounds.shape}."
            )
        return bounds.astype(np.float64)

    ordered: list[tuple[float, float]] = []
    for name in QUIJOTE_BSQ_THETA_NAMES:
        if name not in theta_bounds:
            raise ValueError(f"Quijote theta_bounds is missing required key {name!r}.")
        pair = theta_bounds[name]
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"Quijote theta bound for {name!r} must be a pair.")
        low = float(pair[0])
        high = float(pair[1])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(f"Invalid Quijote theta bound for {name!r}: {(low, high)!r}.")
        ordered.append((low, high))
    return np.asarray(ordered, dtype=np.float64)


def normalize_quijote_theta_batch(theta_batch: np.ndarray, theta_bounds: np.ndarray) -> FloatArray:
    raw = np.asarray(theta_batch, dtype=np.float64)
    bounds = quijote_theta_bounds_as_array(theta_bounds)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2 or raw.shape[1] != bounds.shape[0]:
        raise ValueError(f"theta_batch must have shape [N,{bounds.shape[0]}], got {raw.shape}.")
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    return np.clip((raw - bounds[:, 0][None, :]) / span[None, :], 0.0, 1.0).astype(
        np.float64
    )


def denormalize_quijote_theta_batch(unit_theta_batch: np.ndarray, theta_bounds: np.ndarray) -> FloatArray:
    unit = np.asarray(unit_theta_batch, dtype=np.float64)
    bounds = quijote_theta_bounds_as_array(theta_bounds)
    if unit.ndim == 1:
        unit = unit.reshape(1, -1)
    if unit.ndim != 2 or unit.shape[1] != bounds.shape[0]:
        raise ValueError(f"unit theta_batch must have shape [N,{bounds.shape[0]}], got {unit.shape}.")
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    return (bounds[:, 0][None, :] + np.clip(unit, 0.0, 1.0) * span[None, :]).astype(
        np.float64
    )


def ensure_quijote_theta_batch(
    theta_batch: np.ndarray,
    theta_bounds: np.ndarray,
    *,
    input_space: str,
) -> tuple[FloatArray, FloatArray]:
    arr = np.asarray(theta_batch, dtype=np.float64)
    bounds = quijote_theta_bounds_as_array(theta_bounds)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != bounds.shape[0]:
        raise ValueError(f"theta_batch must have shape [N,{bounds.shape[0]}], got {arr.shape}.")
    if str(input_space).strip().lower() == "unit":
        unit = np.clip(arr, 0.0, 1.0).astype(np.float64)
        raw = denormalize_quijote_theta_batch(unit, bounds)
        return raw, unit
    raw = arr.astype(np.float64)
    unit = normalize_quijote_theta_batch(raw, bounds)
    return raw, unit


def load_quijote_bank(bank_path: str | Path, metadata_path: str | Path | None = None) -> QuijoteBank:
    path = Path(bank_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Quijote bank file not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        raw_thetas = np.asarray(npz["raw_thetas"], dtype=np.float64)
        k_bins = np.asarray(npz["k_bins"], dtype=np.float64)
        p_nonlin_batch = np.asarray(npz["p_nonlin_batch"], dtype=np.float64)
        simulation_indices = (
            np.asarray(npz["simulation_indices"], dtype=np.int64)
            if "simulation_indices" in npz.files
            else None
        )

    resolved_metadata_path = Path(metadata_path).resolve() if metadata_path is not None else path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if resolved_metadata_path.exists():
        metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    metadata.setdefault("bank_path", str(path))
    metadata.setdefault("metadata_path", str(resolved_metadata_path))

    if raw_thetas.ndim != 2 or raw_thetas.shape[1] != len(QUIJOTE_BSQ_THETA_NAMES):
        raise ValueError(
            "Quijote bank raw_thetas must be 2D with 5 columns, "
            f"got {raw_thetas.shape}."
        )
    if k_bins.ndim != 1 or np.any(k_bins <= 0.0) or np.any(np.diff(k_bins) <= 0.0):
        raise ValueError("Quijote bank k_bins must be a strictly increasing positive 1D array.")
    if p_nonlin_batch.ndim != 2 or p_nonlin_batch.shape != (raw_thetas.shape[0], k_bins.shape[0]):
        raise ValueError(
            "Quijote bank p_nonlin_batch must align with raw_thetas and k_bins, "
            f"got {p_nonlin_batch.shape}, {raw_thetas.shape}, {k_bins.shape}."
        )
    if np.any(p_nonlin_batch <= 0.0):
        raise ValueError("Quijote bank contains non-positive P(k) values.")
    return QuijoteBank(
        raw_thetas=raw_thetas,
        k_bins=k_bins,
        p_nonlin_batch=p_nonlin_batch,
        simulation_indices=simulation_indices,
        metadata=metadata,
    )


def derive_theta_bounds_from_bank(bank: QuijoteBank) -> FloatArray:
    raw_thetas = np.asarray(bank.raw_thetas, dtype=np.float64)
    low = np.min(raw_thetas, axis=0)
    high = np.max(raw_thetas, axis=0)
    return np.column_stack((low, high)).astype(np.float64)


def _fit_pca(log_pk: np.ndarray, *, n_components: int, random_seed: int) -> tuple[PCA, FloatArray]:
    log_pk_arr = np.asarray(log_pk, dtype=np.float64)
    resolved = int(min(max(1, int(n_components)), log_pk_arr.shape[0], log_pk_arr.shape[1]))
    pca = PCA(n_components=resolved, svd_solver="auto", random_state=int(random_seed))
    scores = np.asarray(pca.fit_transform(log_pk_arr), dtype=np.float64)
    return pca, scores


def _build_kernel(
    *,
    theta_dim: int,
    constant_value: float,
    constant_value_bounds: tuple[float, float],
    length_scale_initial: float,
    length_scale_bounds: tuple[float, float],
):
    return ConstantKernel(
        constant_value=float(constant_value),
        constant_value_bounds=tuple(float(item) for item in constant_value_bounds),
    ) * RBF(
        length_scale=np.full((int(theta_dim),), float(length_scale_initial), dtype=np.float64),
        length_scale_bounds=tuple(float(item) for item in length_scale_bounds),
    )


def train_quijote_gp_surrogate(
    bank: QuijoteBank,
    *,
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray | None = None,
    train_size: int | None = None,
    train_seed: int = 20260513,
    pca_components: int = 20,
    gp_alpha: float = 1.0e-8,
    normalize_y: bool = True,
    gp_n_restarts_optimizer: int = 0,
    constant_value: float = 1.0,
    constant_value_bounds: tuple[float, float] = (1.0e-4, 1.0e4),
    length_scale_initial: float = 0.2,
    length_scale_bounds: tuple[float, float] = (5.0e-3, 3.0e2),
    power_eps: float = 1.0e-12,
    progress_callback: Any | None = None,
) -> QuijoteGPSurrogate:
    """Fit a standalone direct-logP PCA+GP surrogate from a prepared Quijote bank."""

    raw_thetas = np.asarray(bank.raw_thetas, dtype=np.float64)
    pk_batch = np.asarray(bank.p_nonlin_batch, dtype=np.float64)
    if train_size is None:
        selected_indices = np.arange(raw_thetas.shape[0], dtype=np.int64)
    else:
        resolved_size = int(min(max(1, int(train_size)), raw_thetas.shape[0]))
        rng = np.random.default_rng(int(train_seed))
        selected_indices = np.asarray(
            rng.choice(raw_thetas.shape[0], size=resolved_size, replace=False),
            dtype=np.int64,
        )
        selected_indices.sort()

    train_thetas = raw_thetas[selected_indices]
    train_pk = pk_batch[selected_indices]
    bounds = (
        derive_theta_bounds_from_bank(bank)
        if theta_bounds is None
        else quijote_theta_bounds_as_array(theta_bounds)
    )
    unit_thetas = normalize_quijote_theta_batch(train_thetas, bounds)
    log_pk = np.log(np.maximum(train_pk, float(max(power_eps, 1.0e-30))))
    pca_model, raw_scores = _fit_pca(
        log_pk,
        n_components=int(pca_components),
        random_seed=int(train_seed),
    )
    score_mean = np.mean(raw_scores, axis=0, dtype=np.float64)
    score_std = np.maximum(np.std(raw_scores, axis=0, dtype=np.float64), 1.0e-12)
    scaled_scores = (raw_scores - score_mean.reshape(1, -1)) / score_std.reshape(1, -1)

    gp_models: list[GaussianProcessRegressor] = []
    kernel_descriptions: list[str] = []
    for pc_idx in range(scaled_scores.shape[1]):
        if progress_callback is not None:
            progress_callback("quijote_gp_fit", pc_idx + 1, scaled_scores.shape[1])
        gp = GaussianProcessRegressor(
            kernel=_build_kernel(
                theta_dim=unit_thetas.shape[1],
                constant_value=float(constant_value),
                constant_value_bounds=constant_value_bounds,
                length_scale_initial=float(length_scale_initial),
                length_scale_bounds=length_scale_bounds,
            ),
            alpha=float(gp_alpha),
            normalize_y=bool(normalize_y),
            random_state=int(train_seed + pc_idx),
            n_restarts_optimizer=int(gp_n_restarts_optimizer),
        )
        gp.fit(unit_thetas, scaled_scores[:, pc_idx])
        gp_models.append(gp)
        kernel_descriptions.append(str(gp.kernel_))

    metadata = {
        "surrogate_kind": "quijote_bsq_direct_logpk_pca_gp",
        "parameter_space": QUIJOTE_BSQ5_PARAMETER_SPACE,
        "theta_names": list(QUIJOTE_BSQ_THETA_NAMES),
        "theta_dim": int(bounds.shape[0]),
        "train_size": int(train_thetas.shape[0]),
        "bank_size": int(raw_thetas.shape[0]),
        "k_bin_count": int(bank.k_bins.shape[0]),
        "target_transform": "direct_logpk",
        "has_linear_anchor": False,
        "linear_anchor_note": (
            "The standalone surrogate models Quijote logP only; the runtime provider "
            "adds CAMB linear anchors for the main ratio/logdiff pipeline."
        ),
        "pca_components_requested": int(pca_components),
        "pca_components": int(scaled_scores.shape[1]),
        "gp_alpha": float(gp_alpha),
        "gp_n_restarts_optimizer": int(gp_n_restarts_optimizer),
        "normalize_y": bool(normalize_y),
        "train_seed": int(train_seed),
        "power_eps": float(power_eps),
        "kernel_descriptions": kernel_descriptions,
        "selected_bank_indices_preview": [int(item) for item in selected_indices[:20].tolist()],
        "bank_metadata": dict(bank.metadata),
    }
    return QuijoteGPSurrogate(
        theta_bounds=np.asarray(bounds, dtype=np.float64),
        theta_names=QUIJOTE_BSQ_THETA_NAMES,
        k_bins=np.asarray(bank.k_bins, dtype=np.float64),
        pca_model=pca_model,
        gp_models=gp_models,
        score_mean=score_mean.astype(np.float64),
        score_std=score_std.astype(np.float64),
        power_eps=float(power_eps),
        metadata=metadata,
    )


def save_quijote_gp_surrogate(surrogate: QuijoteGPSurrogate, output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(surrogate, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata_path = path.with_suffix(".json")
    metadata = dict(surrogate.metadata)
    metadata.update(
        {
            "artifact_path": str(path),
            "theta_bounds": np.asarray(surrogate.theta_bounds, dtype=np.float64).tolist(),
            "k_min": float(np.min(surrogate.k_bins)),
            "k_max": float(np.max(surrogate.k_bins)),
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _install_legacy_pickle_aliases(extra_modules: Mapping[str, Any] | None = None) -> None:
    legacy_pkg = sys.modules.get("src")
    if legacy_pkg is None:
        legacy_pkg = types.ModuleType("src")
        legacy_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["src"] = legacy_pkg

    current_module = sys.modules[__name__]
    aliases: dict[str, Any] = {
        "quijote_gp_surrogate": current_module,
    }
    if extra_modules is not None:
        aliases.update(dict(extra_modules))

    for short_name, module in aliases.items():
        sys.modules.setdefault(f"src.{short_name}", module)
        setattr(legacy_pkg, short_name, module)


def load_quijote_gp_surrogate(path: str | Path) -> Any:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Quijote GP surrogate not found: {resolved}")
    extra_aliases: dict[str, Any] = {}
    try:
        from z2quijote.runtime_core import quijote_compact_svgp_surrogate

        extra_aliases["quijote_compact_svgp_surrogate"] = quijote_compact_svgp_surrogate
    except Exception:
        extra_aliases = {}
    _install_legacy_pickle_aliases(extra_aliases)
    with resolved.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, QuijoteGPSurrogate):
        has_provider_surface = (
            hasattr(loaded, "predict")
            and hasattr(loaded, "k_bins")
            and hasattr(loaded, "metadata")
        )
        if not has_provider_surface:
            raise TypeError(
                f"Expected QuijoteGPSurrogate-compatible object in {resolved}, "
                f"got {type(loaded).__name__}."
            )
    return loaded


def _interp_logk_batch(batch: np.ndarray, source_k: np.ndarray, target_k: np.ndarray) -> FloatArray:
    source = np.asarray(source_k, dtype=np.float64).reshape(-1)
    target = np.asarray(target_k, dtype=np.float64).reshape(-1)
    batch_arr = np.asarray(batch, dtype=np.float64)
    if batch_arr.ndim != 2 or batch_arr.shape[1] != source.shape[0]:
        raise ValueError(
            "batch must be 2D and align with source_k, "
            f"got {batch_arr.shape} vs {source.shape}."
        )
    if target.shape == source.shape and np.allclose(target, source):
        return batch_arr.astype(np.float64)
    source_min = float(source[0])
    source_max = float(source[-1])
    target_min = float(target[0])
    target_max = float(target[-1])
    if target_min < source_min * (1.0 - 1.0e-10) or target_max > source_max * (1.0 + 1.0e-10):
        raise ValueError(
            "Interpolation target k grid extends outside Quijote source coverage, "
            f"got source [{source_min}, {source_max}] and target [{target_min}, {target_max}]."
        )
    log_source = np.log10(np.maximum(source, 1.0e-30))
    log_target = np.log10(np.maximum(target, 1.0e-30))
    rows: list[np.ndarray] = []
    for row in batch_arr:
        interp = PchipInterpolator(log_source, row, extrapolate=False)
        rows.append(np.asarray(interp(log_target), dtype=np.float64))
    return np.vstack(rows).astype(np.float64)


__all__ = [
    "DEFAULT_QUIJOTE_BSQ5_BOUNDS",
    "QUIJOTE_BSQ5_PARAMETER_SPACE",
    "QUIJOTE_BSQ_THETA_NAMES",
    "QuijoteBank",
    "QuijoteGPSurrogate",
    "denormalize_quijote_theta_batch",
    "derive_theta_bounds_from_bank",
    "ensure_quijote_theta_batch",
    "load_quijote_bank",
    "load_quijote_gp_surrogate",
    "normalize_quijote_theta_batch",
    "quijote_theta_bounds_as_array",
    "save_quijote_gp_surrogate",
    "train_quijote_gp_surrogate",
]
