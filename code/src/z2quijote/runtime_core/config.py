"""Runtime configuration for the active-learning cosmology emulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import yaml

FloatArray = NDArray[np.float64]

THETA_NAMES: Final[tuple[str, ...]] = (
    "Omegab",
    "Omegacb",
    "H0",
    "ns",
    "A",
    "w",
    "wa",
    "mnu",
)

THETA_UNITS: Final[dict[str, str]] = {
    "Omegab": "dimensionless",
    "Omegacb": "dimensionless",
    "H0": "km/s/Mpc",
    "ns": "dimensionless",
    "A": "1e9 As",
    "w": "dimensionless",
    "wa": "dimensionless",
    "mnu": "eV",
}

DEFAULT_THETA_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "Omegab": (0.04, 0.06),
    "Omegacb": (0.24, 0.40),
    "H0": (60.0, 80.0),
    "ns": (0.92, 1.00),
    "A": (1.7, 2.5),
    "w": (-1.3, -0.7),
    "wa": (-0.5, 0.5),
    "mnu": (0.0, 0.3),
}

DEVICE_CHOICES: Final[tuple[str, ...]] = ("cpu", "cuda", "auto")
SPECTRUM_TYPE_CHOICES: Final[tuple[str, ...]] = ("dark_matter", "galaxy")
REPRESENTATION_ANCHOR_CHOICES: Final[tuple[str, ...]] = ("linear", "halofit", "hmcode2020")
REPRESENTATION_TRANSFORM_CHOICES: Final[tuple[str, ...]] = ("ratio", "logdiff")
PCA_SCHEME_CHOICES: Final[tuple[str, ...]] = (
    "global_pca",
    "bandwise_pca",
    "global_plus_band_residual_pca",
)
DP_ERROR_SOURCE_CHOICES: Final[tuple[str, ...]] = (
    "validation_relative_error",
    "pca_sensitivity_proxy",
)
DP_BAND_WEIGHT_BALANCE_CHOICES: Final[tuple[str, ...]] = (
    "error_only",
    "core_posterior_variance",
)
LOFI_STRATEGY_CHOICES: Final[tuple[str, ...]] = (
    "camb_low_accuracy",
    "noise",
    "linear_ratio_transfer",
)
LOFI_SPEED_TIER_CHOICES: Final[tuple[str, ...]] = ("L1", "L2", "L3")

_LEGACY_KEY_PREFIXES: Final[tuple[str, ...]] = ("module", "branch")
_LEGACY_TOP_LEVEL_KEYS: Final[set[str]] = {
    "run_loops",
    "cycles_total",
    "cold_start_enabled",
    "cold_start_sample_size",
    "anchors_per_iteration",
    "iterations_per_cycle",
    "dante_enabled",
    "dante_enable_after_cycle",
    "midterm",
    "models",
}
_LEGACY_THETA_NAME_ALIASES: Final[dict[str, str]] = {
    "Omegam": "Omegacb",
}
_LEGACY_GRID_KEY_ALIASES: Final[dict[str, str]] = {
    "primary_log_bins": "primary_uniform_bins",
    "high_k_log_bins": "high_k_dense_bins",
}


def _project_root_from_path(path: Path) -> Path:
    path = path.resolve()
    if path.parent.name == "configs":
        return path.parent.parent
    return path.parent


def _is_legacy_key(key: str) -> bool:
    normalized = str(key).strip().lower()
    return normalized in _LEGACY_TOP_LEVEL_KEYS or normalized.startswith(_LEGACY_KEY_PREFIXES)


def _canonicalize_theta_name(name: str) -> str:
    normalized = str(name).strip()
    return _LEGACY_THETA_NAME_ALIASES.get(normalized, normalized)


def _canonicalize_section_keys(
    override_section: Mapping[str, Any],
    aliases: Mapping[str, str],
    *,
    section_name: str,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in override_section.items():
        canonical_key = aliases.get(str(raw_key).strip(), str(raw_key).strip())
        if canonical_key in normalized and normalized[canonical_key] != value:
            raise ValueError(
                f"Conflicting configuration keys provided for {section_name}.{canonical_key}"
            )
        normalized[canonical_key] = value
    return normalized


def _coerce_theta_bounds(
    theta_bounds: Mapping[str, Sequence[float]] | None,
) -> dict[str, tuple[float, float]]:
    resolved: dict[str, tuple[float, float]] = dict(DEFAULT_THETA_BOUNDS)
    if theta_bounds is None:
        return resolved
    seen: dict[str, tuple[float, float]] = {}
    for raw_name, raw_bounds in theta_bounds.items():
        name = _canonicalize_theta_name(str(raw_name))
        if name not in DEFAULT_THETA_BOUNDS:
            raise ValueError(f"Unknown theta bound name: {raw_name!r}")
        if not isinstance(raw_bounds, Sequence) or len(raw_bounds) != 2:
            raise ValueError(f"Theta bounds for {name!r} must be a pair, got {raw_bounds!r}.")
        low = float(raw_bounds[0])
        high = float(raw_bounds[1])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(f"Invalid theta bounds for {name!r}: {(low, high)!r}")
        current = (low, high)
        if name in seen and seen[name] != current:
            raise ValueError(
                f"Conflicting theta bounds provided for aliases of {name!r}: {seen[name]!r} vs {current!r}"
            )
        resolved[name] = (low, high)
        seen[name] = current
    return resolved


def theta_bounds_as_array(
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray | FloatArray,
) -> FloatArray:
    if isinstance(theta_bounds, np.ndarray):
        bounds = np.asarray(theta_bounds, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(
                "theta_bounds array must have shape (D, 2), "
                f"got {bounds.shape}."
            )
        if bounds.shape[0] <= 0:
            raise ValueError("theta_bounds array must contain at least one parameter row.")
        if np.any(~np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError(f"theta_bounds array contains invalid bounds: {bounds!r}.")
        return bounds

    normalized_bounds = _canonicalize_section_keys(
        theta_bounds,
        _LEGACY_THETA_NAME_ALIASES,
        section_name="theta_bounds",
    )
    ordered = []
    for name in THETA_NAMES:
        if name not in normalized_bounds:
            raise ValueError(f"theta_bounds is missing required key {name!r}.")
        low, high = normalized_bounds[name]
        ordered.append((float(low), float(high)))
    return np.asarray(ordered, dtype=np.float64)


def normalize_theta_batch(
    theta_batch: Sequence[Sequence[float]] | np.ndarray,
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray | FloatArray,
) -> FloatArray:
    raw = np.asarray(theta_batch, dtype=np.float64)
    bounds = theta_bounds_as_array(theta_bounds)
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != bounds.shape[0]:
        raise ValueError(f"Expected theta dimension {bounds.shape[0]}, got {raw.shape[1]}.")
    unit = (raw - bounds[:, 0][None, :]) / span[None, :]
    return np.clip(unit, 0.0, 1.0).astype(np.float64)


def denormalize_theta_batch(
    unit_theta_batch: Sequence[Sequence[float]] | np.ndarray,
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray | FloatArray,
) -> FloatArray:
    unit = np.asarray(unit_theta_batch, dtype=np.float64)
    bounds = theta_bounds_as_array(theta_bounds)
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    if unit.ndim == 1:
        unit = unit.reshape(1, -1)
    if unit.shape[1] != bounds.shape[0]:
        raise ValueError(f"Expected theta dimension {bounds.shape[0]}, got {unit.shape[1]}.")
    return (bounds[:, 0][None, :] + np.clip(unit, 0.0, 1.0) * span[None, :]).astype(np.float64)


def build_default_theta_metric(
    theta_bounds: Mapping[str, Sequence[float]] | np.ndarray | FloatArray,
) -> FloatArray:
    bounds = theta_bounds_as_array(theta_bounds)
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    return np.diag(1.0 / span**2).astype(np.float64)


def normalize_device_ids(device_ids: Sequence[int] | None) -> tuple[int, ...]:
    if not isinstance(device_ids, Sequence):
        return tuple()
    normalized: list[int] = []
    for raw in device_ids:
        try:
            normalized.append(int(raw))
        except Exception:
            continue
    return tuple(normalized)


def _coerce_float(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a real number, got {value!r}.") from exc


def _coerce_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}.") from exc


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}.")


def _coerce_float_vector(
    value: Any,
    *,
    field_name: str,
    length: int,
    min_value: float | None = None,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{field_name} must be a sequence of length {int(length)}, got {value!r}."
        )
    if len(value) != int(length):
        raise ValueError(
            f"{field_name} must have length {int(length)}, got {len(value)}."
        )
    coerced: list[float] = []
    lower = None if min_value is None else float(min_value)
    for index, item in enumerate(value):
        current = _coerce_float(item, field_name=f"{field_name}[{index}]")
        if lower is not None:
            current = max(lower, current)
        coerced.append(float(current))
    return tuple(coerced)


@dataclass(slots=True)
class GridConfig:
    k_min: float = 1.0e-2
    k_max: float = 1.0e1
    low_k_max: float = 0.07
    mid_k_max: float = 0.5
    low_k_fraction: float = 0.10
    mid_k_fraction: float = 0.60
    high_k_fraction: float = 0.30
    primary_uniform_bins: int = 3182
    high_k_min: float = 0.5
    high_k_max: float = 10.0
    high_k_dense_bins: int = 1000
    k_eval_size: int = 1000
    log_k_min: float = -2.0
    log_k_max: float = 1.0
    k_work_size: int = 4181

    def __post_init__(self) -> None:
        self.k_min = max(1.0e-6, _coerce_float(self.k_min, field_name="grids.k_min"))
        self.k_max = max(self.k_min + 1.0e-6, _coerce_float(self.k_max, field_name="grids.k_max"))
        self.low_k_max = float(
            min(
                max(self.k_min + 1.0e-6, _coerce_float(self.low_k_max, field_name="grids.low_k_max")),
                self.k_max - 1.0e-6,
            )
        )
        self.mid_k_max = float(
            min(
                max(
                    self.low_k_max + 1.0e-6,
                    _coerce_float(self.mid_k_max, field_name="grids.mid_k_max"),
                ),
                self.k_max,
            )
        )
        low_fraction = _coerce_float(self.low_k_fraction, field_name="grids.low_k_fraction")
        mid_fraction = _coerce_float(self.mid_k_fraction, field_name="grids.mid_k_fraction")
        high_fraction = _coerce_float(self.high_k_fraction, field_name="grids.high_k_fraction")
        fractions = np.asarray([low_fraction, mid_fraction, high_fraction], dtype=np.float64)
        if np.any(~np.isfinite(fractions)) or np.any(fractions <= 0.0):
            raise ValueError("grids low/mid/high k fractions must all be finite positive values.")
        total_fraction = float(np.sum(fractions))
        if total_fraction <= 0.0:
            raise ValueError("grids low/mid/high k fractions must sum to a positive value.")
        # Allow users to provide either normalized fractions (0.1/0.6/0.3)
        # or percentage-style weights (10/60/30).
        normalized = fractions / total_fraction
        self.low_k_fraction = float(normalized[0])
        self.mid_k_fraction = float(normalized[1])
        self.high_k_fraction = float(normalized[2])
        self.primary_uniform_bins = max(
            8,
            _coerce_int(self.primary_uniform_bins, field_name="grids.primary_uniform_bins"),
        )
        high_k_min = _coerce_float(self.high_k_min, field_name="grids.high_k_min")
        high_k_max = _coerce_float(self.high_k_max, field_name="grids.high_k_max")
        self.high_k_min = float(min(max(self.k_min, high_k_min), self.k_max))
        self.high_k_max = float(max(self.high_k_min, min(self.k_max, high_k_max)))
        self.high_k_dense_bins = max(
            1,
            _coerce_int(self.high_k_dense_bins, field_name="grids.high_k_dense_bins"),
        )
        self.k_eval_size = max(8, _coerce_int(self.k_eval_size, field_name="grids.k_eval_size"))
        self.log_k_min = float(np.log10(self.k_min))
        self.log_k_max = float(np.log10(self.k_max))
        self.k_work_size = max(8, _coerce_int(self.k_work_size, field_name="grids.k_work_size"))


def _allocate_piecewise_k_counts(
    total_count: int,
    fractions: Sequence[float],
    *,
    minimum_counts: Sequence[int],
) -> tuple[int, ...]:
    total = int(total_count)
    if total < int(sum(minimum_counts)):
        raise ValueError(
            f"Requested k_work_size={total} is too small for the required minimum per-band allocation "
            f"{tuple(int(v) for v in minimum_counts)}."
        )
    frac = np.asarray(fractions, dtype=np.float64)
    frac = frac / float(np.sum(frac))
    minimum = np.asarray(minimum_counts, dtype=np.int64)
    raw = total * frac
    counts = np.maximum(np.floor(raw).astype(np.int64), minimum)
    leftover = total - int(counts.sum())
    if leftover > 0:
        order = np.argsort(-(raw - counts), kind="mergesort")
        for idx in order[:leftover]:
            counts[int(idx)] += 1
    elif leftover < 0:
        order = np.argsort(counts - raw, kind="mergesort")[::-1]
        deficit = -leftover
        for idx in order:
            band = int(idx)
            reducible = int(counts[band] - minimum[band])
            if reducible <= 0:
                continue
            delta = min(reducible, deficit)
            counts[band] -= delta
            deficit -= delta
            if deficit <= 0:
                break
        if deficit > 0:
            raise ValueError(
                f"Could not satisfy k-grid minimum counts {tuple(int(v) for v in minimum)} "
                f"for total_count={total}."
            )
    return tuple(int(v) for v in counts)


@dataclass(slots=True)
class SamplingConfig:
    initial_sobol_points: int = 64
    iterations: int = 64
    batch_size: int = 1
    candidate_pool_size: int = 4096
    initial_seed: int = 20260330
    candidate_seed: int = 20260331
    default_cloud_size: int = 256
    sobol_num_candidates: int = 4096
    duplicate_distance_threshold: float = 1.0e-6
    gp_l2_train_points: int = 256
    gp_l2_logk_points: int = 200
    gp_l2_pca_components: int = 20

    def __post_init__(self) -> None:
        self.initial_sobol_points = max(
            1,
            _coerce_int(self.initial_sobol_points, field_name="sampling.initial_sobol_points"),
        )
        self.iterations = max(0, _coerce_int(self.iterations, field_name="sampling.iterations"))
        self.batch_size = max(1, _coerce_int(self.batch_size, field_name="sampling.batch_size"))
        self.candidate_pool_size = max(
            self.batch_size,
            _coerce_int(self.candidate_pool_size, field_name="sampling.candidate_pool_size"),
        )
        self.initial_seed = _coerce_int(self.initial_seed, field_name="sampling.initial_seed")
        self.candidate_seed = _coerce_int(self.candidate_seed, field_name="sampling.candidate_seed")
        self.default_cloud_size = max(
            1,
            _coerce_int(self.default_cloud_size, field_name="sampling.default_cloud_size"),
        )
        self.sobol_num_candidates = int(self.candidate_pool_size)
        self.duplicate_distance_threshold = float(
            max(
                0.0,
                _coerce_float(
                    self.duplicate_distance_threshold,
                    field_name="sampling.duplicate_distance_threshold",
                ),
            )
        )
        self.gp_l2_train_points = max(
            8,
            _coerce_int(self.gp_l2_train_points, field_name="sampling.gp_l2_train_points"),
        )
        self.gp_l2_logk_points = max(
            8,
            _coerce_int(self.gp_l2_logk_points, field_name="sampling.gp_l2_logk_points"),
        )
        self.gp_l2_pca_components = max(
            1,
            _coerce_int(self.gp_l2_pca_components, field_name="sampling.gp_l2_pca_components"),
        )

    @property
    def total_budget(self) -> int:
        return int(self.initial_sobol_points + self.iterations * self.batch_size)


@dataclass(slots=True)
class GPConfig:
    pca_components: int = 20
    alpha: float = 1.0e-8
    normalize_y: bool = True
    n_restarts_optimizer: int = 6
    constant_value: float = 1.0
    constant_value_bounds_low: float = 1.0e-4
    constant_value_bounds_high: float = 1.0e4
    length_scale_initial: float = 0.2
    length_scale_bounds_low: float = 5.0e-3
    length_scale_bounds_high: float = 3.0e2
    power_eps: float = 1.0e-12

    def __post_init__(self) -> None:
        self.pca_components = max(1, _coerce_int(self.pca_components, field_name="gp.pca_components"))
        self.alpha = float(max(1.0e-12, _coerce_float(self.alpha, field_name="gp.alpha")))
        self.normalize_y = _coerce_bool(self.normalize_y, field_name="gp.normalize_y")
        self.n_restarts_optimizer = max(
            0,
            _coerce_int(self.n_restarts_optimizer, field_name="gp.n_restarts_optimizer"),
        )
        self.constant_value = float(
            max(1.0e-8, _coerce_float(self.constant_value, field_name="gp.constant_value"))
        )
        self.constant_value_bounds_low = float(
            max(
                1.0e-8,
                _coerce_float(
                    self.constant_value_bounds_low,
                    field_name="gp.constant_value_bounds_low",
                ),
            )
        )
        self.constant_value_bounds_high = float(
            max(
                self.constant_value_bounds_low,
                _coerce_float(
                    self.constant_value_bounds_high,
                    field_name="gp.constant_value_bounds_high",
                ),
            )
        )
        self.length_scale_initial = float(
            max(
                1.0e-6,
                _coerce_float(self.length_scale_initial, field_name="gp.length_scale_initial"),
            )
        )
        self.length_scale_bounds_low = float(
            max(
                1.0e-6,
                _coerce_float(
                    self.length_scale_bounds_low,
                    field_name="gp.length_scale_bounds_low",
                ),
            )
        )
        self.length_scale_bounds_high = float(
            max(
                self.length_scale_bounds_low,
                _coerce_float(
                    self.length_scale_bounds_high,
                    field_name="gp.length_scale_bounds_high",
                ),
            )
        )
        self.power_eps = float(max(1.0e-30, _coerce_float(self.power_eps, field_name="gp.power_eps")))


@dataclass(slots=True)
class GPBaselineConfig:
    enabled: bool = False
    output_subdir: str = "standard_gp_baseline_128"
    random_seed: int = 20260310
    train_points: int = 128
    pca_components: int = 20
    gp_alpha: float = 1.0e-8
    gp_n_restarts_optimizer: int = 6
    normalize_y: bool = True
    constant_value: float = 1.0
    constant_value_bounds_low: float = 1.0e-4
    constant_value_bounds_high: float = 1.0e4
    length_scale_initial: float = 0.2
    length_scale_bounds_low: float = 5.0e-3
    length_scale_bounds_high: float = 3.0e2
    power_eps: float = 1.0e-12

    def __post_init__(self) -> None:
        self.enabled = _coerce_bool(self.enabled, field_name="gp_baseline.enabled")
        self.output_subdir = str(self.output_subdir).strip() or "standard_gp_baseline_128"
        self.random_seed = _coerce_int(self.random_seed, field_name="gp_baseline.random_seed")
        self.train_points = max(
            1,
            _coerce_int(self.train_points, field_name="gp_baseline.train_points"),
        )
        self.pca_components = max(
            1,
            _coerce_int(self.pca_components, field_name="gp_baseline.pca_components"),
        )
        self.gp_alpha = float(
            max(1.0e-12, _coerce_float(self.gp_alpha, field_name="gp_baseline.gp_alpha"))
        )
        self.gp_n_restarts_optimizer = max(
            0,
            _coerce_int(
                self.gp_n_restarts_optimizer,
                field_name="gp_baseline.gp_n_restarts_optimizer",
            ),
        )
        self.normalize_y = _coerce_bool(self.normalize_y, field_name="gp_baseline.normalize_y")
        self.constant_value = float(
            max(
                1.0e-8,
                _coerce_float(self.constant_value, field_name="gp_baseline.constant_value"),
            )
        )
        self.constant_value_bounds_low = float(
            max(
                1.0e-8,
                _coerce_float(
                    self.constant_value_bounds_low,
                    field_name="gp_baseline.constant_value_bounds_low",
                ),
            )
        )
        self.constant_value_bounds_high = float(
            max(
                self.constant_value_bounds_low,
                _coerce_float(
                    self.constant_value_bounds_high,
                    field_name="gp_baseline.constant_value_bounds_high",
                ),
            )
        )
        self.length_scale_initial = float(
            max(
                1.0e-6,
                _coerce_float(
                    self.length_scale_initial,
                    field_name="gp_baseline.length_scale_initial",
                ),
            )
        )
        self.length_scale_bounds_low = float(
            max(
                1.0e-6,
                _coerce_float(
                    self.length_scale_bounds_low,
                    field_name="gp_baseline.length_scale_bounds_low",
                ),
            )
        )
        self.length_scale_bounds_high = float(
            max(
                self.length_scale_bounds_low,
                _coerce_float(
                    self.length_scale_bounds_high,
                    field_name="gp_baseline.length_scale_bounds_high",
                ),
            )
        )
        self.power_eps = float(
            max(1.0e-30, _coerce_float(self.power_eps, field_name="gp_baseline.power_eps"))
        )


@dataclass(slots=True)
class RepresentationConfig:
    anchor_mode: str = "linear"
    transform_family: str = "logdiff"
    pca_scheme: str = "global_plus_band_residual_pca"
    global_pca_components: int = 6
    band_pca_components: tuple[int, ...] = (2, 5, 4, 3)

    def __post_init__(self) -> None:
        anchor_mode = str(self.anchor_mode).strip().lower() or "linear"
        if anchor_mode not in REPRESENTATION_ANCHOR_CHOICES:
            raise ValueError(
                "representation.anchor_mode must be one of "
                f"{REPRESENTATION_ANCHOR_CHOICES}, got {self.anchor_mode!r}."
            )
        self.anchor_mode = anchor_mode

        transform_family = str(self.transform_family).strip().lower() or "ratio"
        if transform_family not in REPRESENTATION_TRANSFORM_CHOICES:
            raise ValueError(
                "representation.transform_family must be one of "
                f"{REPRESENTATION_TRANSFORM_CHOICES}, got {self.transform_family!r}."
            )
        self.transform_family = transform_family

        pca_scheme = str(self.pca_scheme).strip().lower() or "global_pca"
        if pca_scheme not in PCA_SCHEME_CHOICES:
            raise ValueError(
                "representation.pca_scheme must be one of "
                f"{PCA_SCHEME_CHOICES}, got {self.pca_scheme!r}."
            )
        self.pca_scheme = pca_scheme

        self.global_pca_components = max(
            0,
            _coerce_int(
                self.global_pca_components,
                field_name="representation.global_pca_components",
            ),
        )
        self.band_pca_components = tuple(
            max(0, int(value))
            for value in _coerce_float_vector(
                self.band_pca_components,
                field_name="representation.band_pca_components",
                length=4,
                min_value=0.0,
            )
        )


@dataclass(slots=True)
class DynamicPreprocessingConfig:
    enabled: bool = True
    error_source: str = "validation_relative_error"
    band_weight_update_interval: int = 8
    band_component_update_interval: int = 16
    grid_update_interval: int = 32
    update_band_weights: bool = True
    update_pca_allocation: bool = True
    update_k_grid: bool = True
    validation_subset_size: int = 0
    weight_gamma: float = 0.45
    weight_rho: float = 0.12
    weight_min: float = 1.0e-8
    weight_max: float = 1.25
    band_weight_balance_mode: str = "core_posterior_variance"
    core_band_indices: tuple[int, ...] = (1, 2, 3)
    core_error_good: float = 0.0035
    core_error_bad: float = 0.0060
    core_gate_floor: float = 0.15
    core_gate_ceiling: float = 0.85
    core_priority: tuple[float, ...] = (0.30, 1.35, 1.30, 1.05)
    release_priority: tuple[float, ...] = (0.45, 1.25, 1.20, 1.10)
    error_signal_eta: float = 0.25
    posterior_variance_probe_size: int = 512
    posterior_variance_seed_offset: int = 9173
    posterior_variance_gamma: float = 0.50
    posterior_variance_eta: float = 0.35
    band_weight_prior: tuple[float, ...] = (0.30, 1.35, 1.30, 1.05)
    weight_prior_eta: float = 0.35
    allocation_lambda: float = 0.35
    min_band_components: int = 2
    max_component_delta_per_update: int = 1
    proxy_fallback_enabled: bool = True

    def __post_init__(self) -> None:
        self.enabled = _coerce_bool(
            self.enabled,
            field_name="dynamic_preprocessing.enabled",
        )
        error_source = str(self.error_source).strip().lower() or "validation_relative_error"
        if error_source not in DP_ERROR_SOURCE_CHOICES:
            raise ValueError(
                "dynamic_preprocessing.error_source must be one of "
                f"{DP_ERROR_SOURCE_CHOICES}, got {self.error_source!r}."
            )
        self.error_source = error_source
        self.band_weight_update_interval = max(
            1,
            _coerce_int(
                self.band_weight_update_interval,
                field_name="dynamic_preprocessing.band_weight_update_interval",
            ),
        )
        self.band_component_update_interval = max(
            1,
            _coerce_int(
                self.band_component_update_interval,
                field_name="dynamic_preprocessing.band_component_update_interval",
            ),
        )
        self.grid_update_interval = max(
            1,
            _coerce_int(
                self.grid_update_interval,
                field_name="dynamic_preprocessing.grid_update_interval",
            ),
        )
        self.update_band_weights = _coerce_bool(
            self.update_band_weights,
            field_name="dynamic_preprocessing.update_band_weights",
        )
        self.update_pca_allocation = _coerce_bool(
            self.update_pca_allocation,
            field_name="dynamic_preprocessing.update_pca_allocation",
        )
        self.update_k_grid = _coerce_bool(
            self.update_k_grid,
            field_name="dynamic_preprocessing.update_k_grid",
        )
        self.validation_subset_size = max(
            0,
            _coerce_int(
                self.validation_subset_size,
                field_name="dynamic_preprocessing.validation_subset_size",
            ),
        )
        self.weight_gamma = float(
            max(
                0.0,
                _coerce_float(
                    self.weight_gamma,
                    field_name="dynamic_preprocessing.weight_gamma",
                ),
            )
        )
        self.weight_rho = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.weight_rho,
                        field_name="dynamic_preprocessing.weight_rho",
                    ),
                ),
            )
        )
        self.weight_min = float(
            max(
                1.0e-6,
                _coerce_float(
                    self.weight_min,
                    field_name="dynamic_preprocessing.weight_min",
                ),
            )
        )
        self.weight_max = float(
            max(
                self.weight_min,
                _coerce_float(
                    self.weight_max,
                    field_name="dynamic_preprocessing.weight_max",
                ),
            )
        )
        balance_mode = str(self.band_weight_balance_mode).strip().lower() or "error_only"
        if balance_mode not in DP_BAND_WEIGHT_BALANCE_CHOICES:
            raise ValueError(
                "dynamic_preprocessing.band_weight_balance_mode must be one of "
                f"{DP_BAND_WEIGHT_BALANCE_CHOICES}, got {self.band_weight_balance_mode!r}."
            )
        self.band_weight_balance_mode = balance_mode
        if isinstance(self.core_band_indices, (str, bytes)) or not isinstance(
            self.core_band_indices,
            Sequence,
        ):
            raise ValueError(
                "dynamic_preprocessing.core_band_indices must be a sequence of band indices."
            )
        core_indices: list[int] = []
        for index, raw_value in enumerate(self.core_band_indices):
            value = _coerce_int(
                raw_value,
                field_name=f"dynamic_preprocessing.core_band_indices[{index}]",
            )
            if value < 0 or value >= 4:
                raise ValueError(
                    "dynamic_preprocessing.core_band_indices entries must be in [0, 3], "
                    f"got {value}."
                )
            if value not in core_indices:
                core_indices.append(value)
        if not core_indices:
            raise ValueError("dynamic_preprocessing.core_band_indices must not be empty.")
        self.core_band_indices = tuple(core_indices)
        self.core_error_good = float(
            max(
                0.0,
                _coerce_float(
                    self.core_error_good,
                    field_name="dynamic_preprocessing.core_error_good",
                ),
            )
        )
        self.core_error_bad = float(
            max(
                self.core_error_good + 1.0e-12,
                _coerce_float(
                    self.core_error_bad,
                    field_name="dynamic_preprocessing.core_error_bad",
                ),
            )
        )
        self.core_gate_floor = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.core_gate_floor,
                        field_name="dynamic_preprocessing.core_gate_floor",
                    ),
                ),
            )
        )
        self.core_gate_ceiling = float(
            min(
                1.0,
                max(
                    self.core_gate_floor,
                    _coerce_float(
                        self.core_gate_ceiling,
                        field_name="dynamic_preprocessing.core_gate_ceiling",
                    ),
                ),
            )
        )
        self.core_priority = _coerce_float_vector(
            self.core_priority,
            field_name="dynamic_preprocessing.core_priority",
            length=4,
            min_value=1.0e-8,
        )
        self.release_priority = _coerce_float_vector(
            self.release_priority,
            field_name="dynamic_preprocessing.release_priority",
            length=4,
            min_value=1.0e-8,
        )
        self.error_signal_eta = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.error_signal_eta,
                        field_name="dynamic_preprocessing.error_signal_eta",
                    ),
                ),
            )
        )
        self.posterior_variance_probe_size = max(
            0,
            _coerce_int(
                self.posterior_variance_probe_size,
                field_name="dynamic_preprocessing.posterior_variance_probe_size",
            ),
        )
        self.posterior_variance_seed_offset = _coerce_int(
            self.posterior_variance_seed_offset,
            field_name="dynamic_preprocessing.posterior_variance_seed_offset",
        )
        self.posterior_variance_gamma = float(
            max(
                0.0,
                _coerce_float(
                    self.posterior_variance_gamma,
                    field_name="dynamic_preprocessing.posterior_variance_gamma",
                ),
            )
        )
        self.posterior_variance_eta = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.posterior_variance_eta,
                        field_name="dynamic_preprocessing.posterior_variance_eta",
                    ),
                ),
            )
        )
        self.band_weight_prior = _coerce_float_vector(
            self.band_weight_prior,
            field_name="dynamic_preprocessing.band_weight_prior",
            length=4,
            min_value=1.0e-8,
        )
        self.weight_prior_eta = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.weight_prior_eta,
                        field_name="dynamic_preprocessing.weight_prior_eta",
                    ),
                ),
            )
        )
        self.allocation_lambda = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.allocation_lambda,
                        field_name="dynamic_preprocessing.allocation_lambda",
                    ),
                ),
            )
        )
        self.min_band_components = max(
            0,
            _coerce_int(
                self.min_band_components,
                field_name="dynamic_preprocessing.min_band_components",
            ),
        )
        self.max_component_delta_per_update = max(
            1,
            _coerce_int(
                self.max_component_delta_per_update,
                field_name="dynamic_preprocessing.max_component_delta_per_update",
            ),
        )
        self.proxy_fallback_enabled = _coerce_bool(
            self.proxy_fallback_enabled,
            field_name="dynamic_preprocessing.proxy_fallback_enabled",
        )


@dataclass(slots=True)
class M3Config:
    coverage_mode: str = "hull_only"
    domain_support_scheme: str = "structured_boundary_384"
    domain_support_point_count: int = 0
    domain_neighbor_count: int = 64
    domain_support_fanout: int = 2
    domain_score_scale: float = 0.80
    objective_mode: str = "representation_grouped_posterior_variance"
    representation_global_weight: float = 1.0
    representation_band_weights: tuple[float, ...] = (0.30, 1.35, 1.30, 1.05)
    refinement_architecture: str = "hierarchical_warmstart"
    weight_function: str = "stable_log_tanh"
    weight_temperature: float = 0.75
    pc_weight_beta: float = 0.4
    pc_weight_alpha_low: float = 0.80
    pc_weight_alpha_mid: float = 1.20
    pc_weight_alpha_focus_high: float = 1.70
    pc_weight_alpha_tail: float = 0.65
    pc_weight_min: float = 0.60
    pc_weight_max: float = 1.80
    band_beta_low: float = 0.25
    band_beta_mid: float = 0.65
    band_beta_focus_high: float = 0.70
    band_beta_tail: float = 0.25
    band_alpha_low: tuple[float, ...] = (2.40, 0.80, 0.30, 0.10)
    band_alpha_mid: tuple[float, ...] = (0.40, 2.20, 0.70, 0.20)
    band_alpha_focus_high: tuple[float, ...] = (0.20, 0.80, 2.20, 0.70)
    band_alpha_tail: tuple[float, ...] = (0.10, 0.30, 0.80, 2.00)
    band_weight_min: float = 1.0e-8
    band_weight_max: float = 2.40
    acquisition_density_weight_power: float = 0.0
    acquisition_density_weight_floor: float = 1.0
    acquisition_spacefill_rerank_top_k: int = 0
    acquisition_spacefill_weight: float = 0.0
    acquisition_spacefill_guard_top_k: int = 0
    acquisition_spacefill_guard_reject_quantile: float = 0.0
    acquisition_spacefill_tiebreak_top_k: int = 0
    acquisition_spacefill_tiebreak_score_ratio: float = 0.0
    acquisition_spacefill_cd_nonworse_top_k: int = 0
    acquisition_spacefill_cd_nonworse_tol: float = 0.0
    acquisition_p68_set_rerank_top_k: int = 0
    acquisition_p68_set_rerank_score_ratio: float = 0.0
    acquisition_p68_set_rerank_risk_mode: str = "shell"
    acquisition_p68_set_rerank_band_weights: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    acquisition_p68_validation_probe_size: int = 0
    acquisition_p68_validation_probe_focus_k_min: float = 0.1
    acquisition_p68_validation_probe_focus_k_max: float = 3.0
    acquisition_p68_validation_probe_shell_width: float = 0.15
    acquisition_p68_validation_probe_min_weight: float = 0.05
    acquisition_p68_set_rerank_acq_weight: float = 1.0
    acquisition_p68_set_rerank_p68_weight: float = 1.0
    acquisition_p68_set_rerank_spacefill_weight: float = 0.25
    acquisition_p68_set_rerank_boundary_weight: float = 0.25
    acquisition_p68_set_rerank_boundary_threshold: float = 0.05
    acquisition_p68_set_rerank_boundary_target_fraction: float = 0.20
    acquisition_p68_loo_guard_top_k: int = 0
    acquisition_p68_loo_guard_score_ratio: float = 0.0
    acquisition_p68_loo_guard_reject_quantile: float = 0.0
    acquisition_p68_loo_guard_bandwidth: float = 0.25
    acquisition_p68_loo_guard_band_weights: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    acquisition_p68_loo_guard_stage: str = "pre"
    acquisition_qmc_pool_count: int = 0
    acquisition_qmc_pool_seed_offset: int = 314159
    acquisition_qmc_pool_static_seed: bool = False
    imse_rerank_top_k: int = 0
    imse_probe_count: int = 0
    imse_probe_seed_offset: int = 27183
    imse_rerank_mode: str = "mean"
    imse_quantile: float = 0.68
    imse_quantile_shell_width: float = 0.08
    imse_quantile_mean_weight: float = 0.0
    imse_quantile_max_weight: float = 0.0
    repr_score_mode: str = "probe_bundle_max"
    repr_dirichlet_probe_count: int = 8
    stage0_chunk_size: int = 6144
    hull_refine_fraction: float = 0.50
    domain_refine_all: bool = False
    global_top_k: int = 8192
    domain_top_k: int = 0
    hierarchical_stage1_refine_fraction: float = 0.50
    hierarchical_stage1_top_k: int = 2048
    hierarchical_stage1_starts_per_simplex_refine: int = 4
    hierarchical_stage1_max_iter_refine: int = 64
    hierarchical_stage1_history_size_refine: int = 5
    hierarchical_stage1_convergence_tol_refine: float = 1.0e-6
    hierarchical_stage2_refine_fraction: float = 1.0
    hierarchical_stage2_top_k: int = 128
    hierarchical_stage2_starts_per_simplex_refine: int = 8
    hierarchical_stage2_max_iter_refine: int = 160
    hierarchical_stage2_history_size_refine: int = 8
    hierarchical_stage2_convergence_tol_refine: float = 1.0e-7
    starts_per_simplex_refine: int = 24
    max_iter_refine: int = 128
    history_size_refine: int = 10
    polish_top_k: int = 128
    stage3_refine_fraction: float = 0.06
    polish_starts_per_simplex_refine: int = 48
    polish_max_iter_refine: int = 256
    polish_history_size_refine: int = 12
    polish_convergence_tol_refine: float = 1.0e-7
    stage3_qmc_top_k: int = 0
    stage3_qmc_sample_count: int = 4096
    stage3_qmc_chunk_size: int = 6144
    chunk_size: int = 6144
    duplicate_tol: float = 1.0e-4
    armijo_c1: float = 1.0e-4
    line_search_steps_refine: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    fallback_step_refine: float = 0.03125
    convergence_tol_refine: float = 1.0e-6
    variance_floor: float = 1.0e-10
    curvature_tol: float = 1.0e-10
    perturbation_eps_refine: float = 2.0e-2

    def __post_init__(self) -> None:
        coverage_mode = str(self.coverage_mode).strip().lower() or "hull_only"
        if coverage_mode not in {"hull_only", "hull_domain_hybrid"}:
            raise ValueError(
                "m3.coverage_mode must be one of {'hull_only', 'hull_domain_hybrid'}, "
                f"got {self.coverage_mode!r}."
            )
        self.coverage_mode = coverage_mode

        support_scheme = str(self.domain_support_scheme).strip().lower() or "structured_boundary_384"
        if support_scheme not in {"axis_only", "axis_corner_hybrid", "structured_boundary_384"}:
            raise ValueError(
                "m3.domain_support_scheme must be one of "
                "{'axis_only', 'axis_corner_hybrid', 'structured_boundary_384'}, "
                f"got {self.domain_support_scheme!r}."
            )
        self.domain_support_scheme = support_scheme

        refinement_architecture = (
            str(self.refinement_architecture).strip().lower() or "hierarchical_warmstart"
        )
        if refinement_architecture not in {"legacy_full_refine", "hierarchical_warmstart"}:
            raise ValueError(
                "m3.refinement_architecture must be one of "
                "{'legacy_full_refine', 'hierarchical_warmstart'}, "
                f"got {self.refinement_architecture!r}."
            )
        self.refinement_architecture = refinement_architecture

        weight_function = str(self.weight_function).strip().lower() or "stable_log_tanh"
        if weight_function not in {"linear_blend", "stable_log_tanh"}:
            raise ValueError(
                "m3.weight_function must be one of {'linear_blend', 'stable_log_tanh'}, "
                f"got {self.weight_function!r}."
            )
        self.weight_function = weight_function
        self.weight_temperature = float(
            max(1.0e-6, _coerce_float(self.weight_temperature, field_name="m3.weight_temperature"))
        )

        objective_mode = str(self.objective_mode).strip().lower() or "mid_high_weighted_sum"
        if objective_mode == "focus_0p1_5_weighted_sum":
            objective_mode = "mid_high_weighted_sum"
        if objective_mode not in {
            "sum_pc_posterior_variance",
            "mid_high_weighted_sum",
            "band_partitioned_posterior_variance",
            "representation_grouped_posterior_variance",
        }:
            raise ValueError(
                "m3.objective_mode must be one of "
                "{'sum_pc_posterior_variance', 'mid_high_weighted_sum', "
                "'band_partitioned_posterior_variance', "
                "'representation_grouped_posterior_variance'}."
            )
        self.objective_mode = objective_mode
        self.representation_global_weight = float(
            max(
                1.0e-8,
                _coerce_float(
                    self.representation_global_weight,
                    field_name="m3.representation_global_weight",
                ),
            )
        )
        self.representation_band_weights = _coerce_float_vector(
            self.representation_band_weights,
            field_name="m3.representation_band_weights",
            length=4,
            min_value=1.0e-8,
        )
        self.pc_weight_beta = float(
            min(
                1.0,
                max(0.0, _coerce_float(self.pc_weight_beta, field_name="m3.pc_weight_beta")),
            )
        )
        self.pc_weight_alpha_low = float(
            max(1.0e-8, _coerce_float(self.pc_weight_alpha_low, field_name="m3.pc_weight_alpha_low"))
        )
        self.pc_weight_alpha_mid = float(
            max(1.0e-8, _coerce_float(self.pc_weight_alpha_mid, field_name="m3.pc_weight_alpha_mid"))
        )
        self.pc_weight_alpha_focus_high = float(
            max(
                1.0e-8,
                _coerce_float(
                    self.pc_weight_alpha_focus_high,
                    field_name="m3.pc_weight_alpha_focus_high",
                ),
            )
        )
        self.pc_weight_alpha_tail = float(
            max(1.0e-8, _coerce_float(self.pc_weight_alpha_tail, field_name="m3.pc_weight_alpha_tail"))
        )
        self.pc_weight_min = float(
            max(1.0e-8, _coerce_float(self.pc_weight_min, field_name="m3.pc_weight_min"))
        )
        self.pc_weight_max = float(
            max(
                self.pc_weight_min,
                _coerce_float(self.pc_weight_max, field_name="m3.pc_weight_max"),
            )
        )
        self.band_beta_low = float(
            min(1.0, max(0.0, _coerce_float(self.band_beta_low, field_name="m3.band_beta_low")))
        )
        self.band_beta_mid = float(
            min(1.0, max(0.0, _coerce_float(self.band_beta_mid, field_name="m3.band_beta_mid")))
        )
        self.band_beta_focus_high = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(self.band_beta_focus_high, field_name="m3.band_beta_focus_high"),
                ),
            )
        )
        self.band_beta_tail = float(
            min(1.0, max(0.0, _coerce_float(self.band_beta_tail, field_name="m3.band_beta_tail")))
        )
        self.band_alpha_low = _coerce_float_vector(
            self.band_alpha_low,
            field_name="m3.band_alpha_low",
            length=4,
            min_value=1.0e-8,
        )
        self.band_alpha_mid = _coerce_float_vector(
            self.band_alpha_mid,
            field_name="m3.band_alpha_mid",
            length=4,
            min_value=1.0e-8,
        )
        self.band_alpha_focus_high = _coerce_float_vector(
            self.band_alpha_focus_high,
            field_name="m3.band_alpha_focus_high",
            length=4,
            min_value=1.0e-8,
        )
        self.band_alpha_tail = _coerce_float_vector(
            self.band_alpha_tail,
            field_name="m3.band_alpha_tail",
            length=4,
            min_value=1.0e-8,
        )
        self.band_weight_min = float(
            max(1.0e-8, _coerce_float(self.band_weight_min, field_name="m3.band_weight_min"))
        )
        self.band_weight_max = float(
            max(
                self.band_weight_min,
                _coerce_float(self.band_weight_max, field_name="m3.band_weight_max"),
            )
        )
        self.acquisition_density_weight_power = float(
            max(
                0.0,
                _coerce_float(
                    self.acquisition_density_weight_power,
                    field_name="m3.acquisition_density_weight_power",
                ),
            )
        )
        self.acquisition_density_weight_floor = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_density_weight_floor,
                        field_name="m3.acquisition_density_weight_floor",
                    ),
                ),
            )
        )
        self.acquisition_spacefill_rerank_top_k = max(
            0,
            _coerce_int(
                self.acquisition_spacefill_rerank_top_k,
                field_name="m3.acquisition_spacefill_rerank_top_k",
            ),
        )
        self.acquisition_spacefill_weight = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_spacefill_weight,
                        field_name="m3.acquisition_spacefill_weight",
                    ),
                ),
            )
        )
        self.acquisition_spacefill_guard_top_k = max(
            0,
            _coerce_int(
                self.acquisition_spacefill_guard_top_k,
                field_name="m3.acquisition_spacefill_guard_top_k",
            ),
        )
        self.acquisition_spacefill_guard_reject_quantile = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_spacefill_guard_reject_quantile,
                        field_name="m3.acquisition_spacefill_guard_reject_quantile",
                    ),
                ),
            )
        )
        self.acquisition_spacefill_tiebreak_top_k = max(
            0,
            _coerce_int(
                self.acquisition_spacefill_tiebreak_top_k,
                field_name="m3.acquisition_spacefill_tiebreak_top_k",
            ),
        )
        self.acquisition_spacefill_tiebreak_score_ratio = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_spacefill_tiebreak_score_ratio,
                        field_name="m3.acquisition_spacefill_tiebreak_score_ratio",
                    ),
                ),
            )
        )
        self.acquisition_spacefill_cd_nonworse_top_k = max(
            0,
            _coerce_int(
                self.acquisition_spacefill_cd_nonworse_top_k,
                field_name="m3.acquisition_spacefill_cd_nonworse_top_k",
            ),
        )
        self.acquisition_spacefill_cd_nonworse_tol = float(
            max(
                0.0,
                _coerce_float(
                    self.acquisition_spacefill_cd_nonworse_tol,
                    field_name="m3.acquisition_spacefill_cd_nonworse_tol",
                ),
            )
        )
        self.acquisition_p68_set_rerank_top_k = max(
            0,
            _coerce_int(
                self.acquisition_p68_set_rerank_top_k,
                field_name="m3.acquisition_p68_set_rerank_top_k",
            ),
        )
        self.acquisition_p68_set_rerank_score_ratio = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_p68_set_rerank_score_ratio,
                        field_name="m3.acquisition_p68_set_rerank_score_ratio",
                    ),
                ),
            )
        )
        risk_mode = str(self.acquisition_p68_set_rerank_risk_mode).strip().lower() or "shell"
        if risk_mode not in {
            "shell",
            "quantile_proxy",
            "soft_quantile",
            "exceedance",
            "rank_body",
            "balanced_exceedance",
            "validation_probe_shell",
        }:
            raise ValueError(
                "m3.acquisition_p68_set_rerank_risk_mode must be one of "
                "{'shell', 'quantile_proxy', 'soft_quantile', 'exceedance', 'rank_body', "
                "'balanced_exceedance', 'validation_probe_shell'}, "
                f"got {self.acquisition_p68_set_rerank_risk_mode!r}."
            )
        self.acquisition_p68_set_rerank_risk_mode = risk_mode
        self.acquisition_p68_set_rerank_band_weights = _coerce_float_vector(
            self.acquisition_p68_set_rerank_band_weights,
            field_name="m3.acquisition_p68_set_rerank_band_weights",
            length=4,
            min_value=0.0,
        )
        self.acquisition_p68_validation_probe_size = max(
            0,
            _coerce_int(
                self.acquisition_p68_validation_probe_size,
                field_name="m3.acquisition_p68_validation_probe_size",
            ),
        )
        self.acquisition_p68_validation_probe_focus_k_min = float(
            max(
                0.0,
                _coerce_float(
                    self.acquisition_p68_validation_probe_focus_k_min,
                    field_name="m3.acquisition_p68_validation_probe_focus_k_min",
                ),
            )
        )
        self.acquisition_p68_validation_probe_focus_k_max = float(
            max(
                self.acquisition_p68_validation_probe_focus_k_min,
                _coerce_float(
                    self.acquisition_p68_validation_probe_focus_k_max,
                    field_name="m3.acquisition_p68_validation_probe_focus_k_max",
                ),
            )
        )
        self.acquisition_p68_validation_probe_shell_width = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_p68_validation_probe_shell_width,
                        field_name="m3.acquisition_p68_validation_probe_shell_width",
                    ),
                ),
            )
        )
        self.acquisition_p68_validation_probe_min_weight = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_p68_validation_probe_min_weight,
                        field_name="m3.acquisition_p68_validation_probe_min_weight",
                    ),
                ),
            )
        )
        self.acquisition_p68_set_rerank_acq_weight = float(
            max(
                0.0,
                _coerce_float(
                    self.acquisition_p68_set_rerank_acq_weight,
                    field_name="m3.acquisition_p68_set_rerank_acq_weight",
                ),
            )
        )
        self.acquisition_p68_set_rerank_p68_weight = float(
            max(
                0.0,
                _coerce_float(
                    self.acquisition_p68_set_rerank_p68_weight,
                    field_name="m3.acquisition_p68_set_rerank_p68_weight",
                ),
            )
        )
        self.acquisition_p68_set_rerank_spacefill_weight = float(
            max(
                0.0,
                _coerce_float(
                    self.acquisition_p68_set_rerank_spacefill_weight,
                    field_name="m3.acquisition_p68_set_rerank_spacefill_weight",
                ),
            )
        )
        self.acquisition_p68_set_rerank_boundary_weight = float(
            max(
                0.0,
                _coerce_float(
                    self.acquisition_p68_set_rerank_boundary_weight,
                    field_name="m3.acquisition_p68_set_rerank_boundary_weight",
                ),
            )
        )
        self.acquisition_p68_set_rerank_boundary_threshold = float(
            min(
                0.5,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_p68_set_rerank_boundary_threshold,
                        field_name="m3.acquisition_p68_set_rerank_boundary_threshold",
                    ),
                ),
            )
        )
        self.acquisition_p68_set_rerank_boundary_target_fraction = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_p68_set_rerank_boundary_target_fraction,
                        field_name="m3.acquisition_p68_set_rerank_boundary_target_fraction",
                    ),
                ),
            )
        )
        self.acquisition_p68_loo_guard_top_k = max(
            0,
            _coerce_int(
                self.acquisition_p68_loo_guard_top_k,
                field_name="m3.acquisition_p68_loo_guard_top_k",
            ),
        )
        self.acquisition_p68_loo_guard_score_ratio = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_p68_loo_guard_score_ratio,
                        field_name="m3.acquisition_p68_loo_guard_score_ratio",
                    ),
                ),
            )
        )
        self.acquisition_p68_loo_guard_reject_quantile = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.acquisition_p68_loo_guard_reject_quantile,
                        field_name="m3.acquisition_p68_loo_guard_reject_quantile",
                    ),
                ),
            )
        )
        self.acquisition_p68_loo_guard_bandwidth = float(
            max(
                1.0e-6,
                _coerce_float(
                    self.acquisition_p68_loo_guard_bandwidth,
                    field_name="m3.acquisition_p68_loo_guard_bandwidth",
                ),
            )
        )
        self.acquisition_p68_loo_guard_band_weights = _coerce_float_vector(
            self.acquisition_p68_loo_guard_band_weights,
            field_name="m3.acquisition_p68_loo_guard_band_weights",
            length=4,
            min_value=0.0,
        )
        loo_guard_stage = str(self.acquisition_p68_loo_guard_stage).strip().lower() or "pre"
        if loo_guard_stage not in {"pre", "post", "both"}:
            raise ValueError(
                "m3.acquisition_p68_loo_guard_stage must be one of {'pre', 'post', 'both'}, "
                f"got {self.acquisition_p68_loo_guard_stage!r}."
            )
        self.acquisition_p68_loo_guard_stage = loo_guard_stage
        self.acquisition_qmc_pool_count = max(
            0,
            _coerce_int(
                self.acquisition_qmc_pool_count,
                field_name="m3.acquisition_qmc_pool_count",
            ),
        )
        self.acquisition_qmc_pool_seed_offset = _coerce_int(
            self.acquisition_qmc_pool_seed_offset,
            field_name="m3.acquisition_qmc_pool_seed_offset",
        )
        self.acquisition_qmc_pool_static_seed = _coerce_bool(
            self.acquisition_qmc_pool_static_seed,
            field_name="m3.acquisition_qmc_pool_static_seed",
        )
        self.imse_rerank_top_k = max(
            0,
            _coerce_int(self.imse_rerank_top_k, field_name="m3.imse_rerank_top_k"),
        )
        self.imse_probe_count = max(
            0,
            _coerce_int(self.imse_probe_count, field_name="m3.imse_probe_count"),
        )
        self.imse_probe_seed_offset = _coerce_int(
            self.imse_probe_seed_offset,
            field_name="m3.imse_probe_seed_offset",
        )
        imse_rerank_mode = str(self.imse_rerank_mode).strip().lower() or "mean"
        if imse_rerank_mode not in {"mean", "p68_proxy", "p68_shell", "p68_soft"}:
            raise ValueError(
                "m3.imse_rerank_mode must be one of {'mean', 'p68_proxy', 'p68_shell', 'p68_soft'}, "
                f"got {self.imse_rerank_mode!r}."
            )
        self.imse_rerank_mode = imse_rerank_mode
        self.imse_quantile = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(self.imse_quantile, field_name="m3.imse_quantile"),
                ),
            )
        )
        self.imse_quantile_shell_width = float(
            min(
                1.0,
                max(
                    1.0e-6,
                    _coerce_float(
                        self.imse_quantile_shell_width,
                        field_name="m3.imse_quantile_shell_width",
                    ),
                ),
            )
        )
        self.imse_quantile_mean_weight = float(
            max(
                0.0,
                _coerce_float(
                    self.imse_quantile_mean_weight,
                    field_name="m3.imse_quantile_mean_weight",
                ),
            )
        )
        self.imse_quantile_max_weight = float(
            max(
                0.0,
                _coerce_float(
                    self.imse_quantile_max_weight,
                    field_name="m3.imse_quantile_max_weight",
                ),
            )
        )

        repr_score_mode = (
            str(self.repr_score_mode).strip().lower() or "probe_bundle_max"
        )
        if repr_score_mode not in {"probe_bundle_max", "max_barycenter_circumcenter_vertex"}:
            raise ValueError(
                "m3.repr_score_mode must be one of "
                "{'probe_bundle_max', 'max_barycenter_circumcenter_vertex'}."
            )
        self.repr_score_mode = (
            "probe_bundle_max"
            if repr_score_mode == "max_barycenter_circumcenter_vertex"
            else repr_score_mode
        )

        self.domain_support_point_count = max(
            0,
            _coerce_int(
                self.domain_support_point_count,
                field_name="m3.domain_support_point_count",
            ),
        )
        self.domain_neighbor_count = max(
            8,
            _coerce_int(self.domain_neighbor_count, field_name="m3.domain_neighbor_count"),
        )
        self.domain_support_fanout = max(
            1,
            _coerce_int(self.domain_support_fanout, field_name="m3.domain_support_fanout"),
        )
        self.domain_score_scale = float(
            max(1.0e-8, _coerce_float(self.domain_score_scale, field_name="m3.domain_score_scale"))
        )
        self.repr_dirichlet_probe_count = max(
            0,
            _coerce_int(
                self.repr_dirichlet_probe_count,
                field_name="m3.repr_dirichlet_probe_count",
            ),
        )
        self.stage0_chunk_size = max(1, _coerce_int(self.stage0_chunk_size, field_name="m3.stage0_chunk_size"))
        self.hull_refine_fraction = float(
            min(
                1.0,
                max(0.0, _coerce_float(self.hull_refine_fraction, field_name="m3.hull_refine_fraction")),
            )
        )
        self.domain_refine_all = _coerce_bool(
            self.domain_refine_all,
            field_name="m3.domain_refine_all",
        )
        self.global_top_k = max(1, _coerce_int(self.global_top_k, field_name="m3.global_top_k"))
        self.domain_top_k = max(0, _coerce_int(self.domain_top_k, field_name="m3.domain_top_k"))
        self.hierarchical_stage1_refine_fraction = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.hierarchical_stage1_refine_fraction,
                        field_name="m3.hierarchical_stage1_refine_fraction",
                    ),
                ),
            )
        )
        self.hierarchical_stage1_top_k = max(
            1,
            _coerce_int(
                self.hierarchical_stage1_top_k,
                field_name="m3.hierarchical_stage1_top_k",
            ),
        )
        self.hierarchical_stage1_starts_per_simplex_refine = max(
            1,
            _coerce_int(
                self.hierarchical_stage1_starts_per_simplex_refine,
                field_name="m3.hierarchical_stage1_starts_per_simplex_refine",
            ),
        )
        self.hierarchical_stage1_max_iter_refine = max(
            1,
            _coerce_int(
                self.hierarchical_stage1_max_iter_refine,
                field_name="m3.hierarchical_stage1_max_iter_refine",
            ),
        )
        self.hierarchical_stage1_history_size_refine = max(
            1,
            _coerce_int(
                self.hierarchical_stage1_history_size_refine,
                field_name="m3.hierarchical_stage1_history_size_refine",
            ),
        )
        self.hierarchical_stage1_convergence_tol_refine = float(
            max(
                1.0e-10,
                _coerce_float(
                    self.hierarchical_stage1_convergence_tol_refine,
                    field_name="m3.hierarchical_stage1_convergence_tol_refine",
                ),
            )
        )
        self.hierarchical_stage2_refine_fraction = float(
            min(
                1.0,
                max(
                    0.0,
                    _coerce_float(
                        self.hierarchical_stage2_refine_fraction,
                        field_name="m3.hierarchical_stage2_refine_fraction",
                    ),
                ),
            )
        )
        self.hierarchical_stage2_top_k = max(
            1,
            _coerce_int(
                self.hierarchical_stage2_top_k,
                field_name="m3.hierarchical_stage2_top_k",
            ),
        )
        self.hierarchical_stage2_starts_per_simplex_refine = max(
            1,
            _coerce_int(
                self.hierarchical_stage2_starts_per_simplex_refine,
                field_name="m3.hierarchical_stage2_starts_per_simplex_refine",
            ),
        )
        self.hierarchical_stage2_max_iter_refine = max(
            1,
            _coerce_int(
                self.hierarchical_stage2_max_iter_refine,
                field_name="m3.hierarchical_stage2_max_iter_refine",
            ),
        )
        self.hierarchical_stage2_history_size_refine = max(
            1,
            _coerce_int(
                self.hierarchical_stage2_history_size_refine,
                field_name="m3.hierarchical_stage2_history_size_refine",
            ),
        )
        self.hierarchical_stage2_convergence_tol_refine = float(
            max(
                1.0e-10,
                _coerce_float(
                    self.hierarchical_stage2_convergence_tol_refine,
                    field_name="m3.hierarchical_stage2_convergence_tol_refine",
                ),
            )
        )
        self.starts_per_simplex_refine = max(
            1,
            _coerce_int(
                self.starts_per_simplex_refine,
                field_name="m3.starts_per_simplex_refine",
            ),
        )
        self.max_iter_refine = max(
            1,
            _coerce_int(self.max_iter_refine, field_name="m3.max_iter_refine"),
        )
        self.history_size_refine = max(
            1,
            _coerce_int(self.history_size_refine, field_name="m3.history_size_refine"),
        )
        self.polish_top_k = max(1, _coerce_int(self.polish_top_k, field_name="m3.polish_top_k"))
        self.stage3_refine_fraction = float(
            min(
                1.0,
                max(0.0, _coerce_float(self.stage3_refine_fraction, field_name="m3.stage3_refine_fraction")),
            )
        )
        self.polish_starts_per_simplex_refine = max(
            1,
            _coerce_int(
                self.polish_starts_per_simplex_refine,
                field_name="m3.polish_starts_per_simplex_refine",
            ),
        )
        self.polish_max_iter_refine = max(
            1,
            _coerce_int(self.polish_max_iter_refine, field_name="m3.polish_max_iter_refine"),
        )
        self.polish_history_size_refine = max(
            1,
            _coerce_int(
                self.polish_history_size_refine,
                field_name="m3.polish_history_size_refine",
            ),
        )
        self.polish_convergence_tol_refine = float(
            max(
                1.0e-10,
                _coerce_float(
                    self.polish_convergence_tol_refine,
                    field_name="m3.polish_convergence_tol_refine",
                ),
            )
        )
        self.stage3_qmc_top_k = max(0, _coerce_int(self.stage3_qmc_top_k, field_name="m3.stage3_qmc_top_k"))
        self.stage3_qmc_sample_count = max(
            0,
            _coerce_int(self.stage3_qmc_sample_count, field_name="m3.stage3_qmc_sample_count"),
        )
        self.stage3_qmc_chunk_size = max(
            1,
            _coerce_int(self.stage3_qmc_chunk_size, field_name="m3.stage3_qmc_chunk_size"),
        )
        self.chunk_size = max(1, _coerce_int(self.chunk_size, field_name="m3.chunk_size"))
        self.duplicate_tol = float(
            max(0.0, _coerce_float(self.duplicate_tol, field_name="m3.duplicate_tol"))
        )
        self.armijo_c1 = float(
            max(1.0e-8, _coerce_float(self.armijo_c1, field_name="m3.armijo_c1"))
        )
        raw_steps = self.line_search_steps_refine
        if not isinstance(raw_steps, Sequence) or len(raw_steps) == 0:
            raise ValueError("m3.line_search_steps_refine must be a non-empty sequence.")
        self.line_search_steps_refine = tuple(
            float(max(1.0e-8, _coerce_float(step, field_name="m3.line_search_steps_refine")))
            for step in raw_steps
        )
        self.fallback_step_refine = float(
            max(
                1.0e-8,
                _coerce_float(self.fallback_step_refine, field_name="m3.fallback_step_refine"),
            )
        )
        self.convergence_tol_refine = float(
            max(
                1.0e-10,
                _coerce_float(
                    self.convergence_tol_refine,
                    field_name="m3.convergence_tol_refine",
                ),
            )
        )
        self.variance_floor = float(
            max(1.0e-12, _coerce_float(self.variance_floor, field_name="m3.variance_floor"))
        )
        self.curvature_tol = float(
            max(1.0e-12, _coerce_float(self.curvature_tol, field_name="m3.curvature_tol"))
        )
        self.perturbation_eps_refine = float(
            max(
                1.0e-6,
                _coerce_float(
                    self.perturbation_eps_refine,
                    field_name="m3.perturbation_eps_refine",
                ),
            )
        )


@dataclass(slots=True)
class CAMBConfig:
    spectrum_type: str = "dark_matter"
    lofi_strategy: str = "linear_ratio_transfer"
    lofi_speed_tier: str = "L2"
    lofi_l2_backend: str = "legacy"
    lofi_gp_asset_path: str = "artifacts/lofi_gp_l2/default"
    lofi_gp_auto_build: bool = False
    allow_placeholder_backend: bool = False
    backend_name: str = "real_camb"
    placeholder_noise_seed: int = 20260307
    placeholder_lofi_noise_scale: float = 0.01
    placeholder_hifi_boost: float = 0.05
    lofi_accuracy_preset: dict[str, float] = field(default_factory=dict)
    lofi_formula_name: str = "l3_ratio_transfer"
    lofi_formula_pivot_k: float = 0.2
    lofi_formula_eta_w: float = 0.5
    lofi_formula_freeze_ratio: bool = False
    lofi_formula_clip_log_delta: float = 0.25
    lofi_formula_asset_path: str = ""
    camb_hifi_highk_enabled: bool = True
    camb_hifi_highk_kmin: float = 0.1
    camb_hifi_highk_kmax: float = 10.0
    camb_hifi_require_real_camb: bool = True
    camb_hifi_accuracy_boost: float = 2.5
    camb_hifi_l_accuracy_boost: float = 2.5
    camb_hifi_sampling_boost: float = 2.5
    camb_hifi_k_per_logint: int = 80
    camb_hifi_halofit_version: str = "mead2020"
    camb_hifi_use_high_precision_transfer: bool = True

    def __post_init__(self) -> None:
        spectrum_type = str(self.spectrum_type).strip().lower()
        if spectrum_type not in SPECTRUM_TYPE_CHOICES:
            raise ValueError(
                f"camb.spectrum_type must be one of {SPECTRUM_TYPE_CHOICES}, got {self.spectrum_type!r}."
            )
        self.spectrum_type = spectrum_type

        strategy = str(self.lofi_strategy).strip().lower()
        if strategy not in LOFI_STRATEGY_CHOICES:
            raise ValueError(
                f"camb.lofi_strategy must be one of {LOFI_STRATEGY_CHOICES}, got {self.lofi_strategy!r}."
            )
        self.lofi_strategy = strategy

        tier = str(self.lofi_speed_tier).strip().upper()
        if tier not in LOFI_SPEED_TIER_CHOICES:
            raise ValueError(
                f"camb.lofi_speed_tier must be one of {LOFI_SPEED_TIER_CHOICES}, got {self.lofi_speed_tier!r}."
            )
        self.lofi_speed_tier = tier

        backend = str(self.lofi_l2_backend).strip().lower()
        self.lofi_l2_backend = backend if backend in {"legacy", "gp_emulator"} else "legacy"
        self.lofi_gp_asset_path = str(self.lofi_gp_asset_path).strip()
        self.lofi_gp_auto_build = _coerce_bool(
            self.lofi_gp_auto_build,
            field_name="camb.lofi_gp_auto_build",
        )
        self.allow_placeholder_backend = _coerce_bool(
            self.allow_placeholder_backend,
            field_name="camb.allow_placeholder_backend",
        )
        self.backend_name = str(self.backend_name).strip() or "real_camb"
        self.placeholder_noise_seed = _coerce_int(
            self.placeholder_noise_seed,
            field_name="camb.placeholder_noise_seed",
        )
        self.placeholder_lofi_noise_scale = float(
            max(
                0.0,
                _coerce_float(
                    self.placeholder_lofi_noise_scale,
                    field_name="camb.placeholder_lofi_noise_scale",
                ),
            )
        )
        self.placeholder_hifi_boost = float(
            max(
                0.0,
                _coerce_float(
                    self.placeholder_hifi_boost,
                    field_name="camb.placeholder_hifi_boost",
                ),
            )
        )
        self.lofi_accuracy_preset = dict(self.lofi_accuracy_preset)
        self.lofi_formula_name = str(self.lofi_formula_name).strip() or "l3_ratio_transfer"
        self.lofi_formula_pivot_k = float(
            max(
                1.0e-6,
                _coerce_float(
                    self.lofi_formula_pivot_k,
                    field_name="camb.lofi_formula_pivot_k",
                ),
            )
        )
        self.lofi_formula_eta_w = float(
            max(
                0.0,
                _coerce_float(self.lofi_formula_eta_w, field_name="camb.lofi_formula_eta_w"),
            )
        )
        self.lofi_formula_freeze_ratio = _coerce_bool(
            self.lofi_formula_freeze_ratio,
            field_name="camb.lofi_formula_freeze_ratio",
        )
        self.lofi_formula_clip_log_delta = float(
            max(
                1.0e-6,
                _coerce_float(
                    self.lofi_formula_clip_log_delta,
                    field_name="camb.lofi_formula_clip_log_delta",
                ),
            )
        )
        self.lofi_formula_asset_path = str(self.lofi_formula_asset_path).strip()
        self.camb_hifi_highk_enabled = _coerce_bool(
            self.camb_hifi_highk_enabled,
            field_name="camb.camb_hifi_highk_enabled",
        )
        self.camb_hifi_highk_kmin = float(
            max(
                1.0e-4,
                _coerce_float(self.camb_hifi_highk_kmin, field_name="camb.camb_hifi_highk_kmin"),
            )
        )
        self.camb_hifi_highk_kmax = float(
            max(
                self.camb_hifi_highk_kmin,
                _coerce_float(self.camb_hifi_highk_kmax, field_name="camb.camb_hifi_highk_kmax"),
            )
        )
        self.camb_hifi_require_real_camb = _coerce_bool(
            self.camb_hifi_require_real_camb,
            field_name="camb.camb_hifi_require_real_camb",
        )
        self.camb_hifi_accuracy_boost = float(
            max(
                0.1,
                _coerce_float(
                    self.camb_hifi_accuracy_boost,
                    field_name="camb.camb_hifi_accuracy_boost",
                ),
            )
        )
        self.camb_hifi_l_accuracy_boost = float(
            max(
                0.1,
                _coerce_float(
                    self.camb_hifi_l_accuracy_boost,
                    field_name="camb.camb_hifi_l_accuracy_boost",
                ),
            )
        )
        self.camb_hifi_sampling_boost = float(
            max(
                0.1,
                _coerce_float(
                    self.camb_hifi_sampling_boost,
                    field_name="camb.camb_hifi_sampling_boost",
                ),
            )
        )
        self.camb_hifi_k_per_logint = max(
            16,
            _coerce_int(self.camb_hifi_k_per_logint, field_name="camb.camb_hifi_k_per_logint"),
        )
        self.camb_hifi_halofit_version = (
            str(self.camb_hifi_halofit_version).strip().lower() or "mead2020"
        )
        self.camb_hifi_use_high_precision_transfer = _coerce_bool(
            self.camb_hifi_use_high_precision_transfer,
            field_name="camb.camb_hifi_use_high_precision_transfer",
        )


@dataclass(slots=True)
class ValidationRuntimeConfig:
    project_root: str = "."
    device: str = "auto"
    device_ids: list[int] | None = None
    random_seed: int = 20260307
    checkpoint_dir: str = "data/runtime_core/checkpoints"
    reports_dir: str = "data/runtime_core/reports"
    test_set_enabled: bool = True
    test_set_size: int = 256
    test_set_seed: int = 20260332
    eps_r: float = 1.0e-12
    theta_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_THETA_BOUNDS)
    )
    grids: GridConfig = field(default_factory=GridConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    gp_baseline: GPBaselineConfig = field(default_factory=GPBaselineConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    dynamic_preprocessing: DynamicPreprocessingConfig = field(
        default_factory=DynamicPreprocessingConfig
    )
    m3: M3Config = field(default_factory=M3Config)
    camb: CAMBConfig = field(default_factory=CAMBConfig)
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        device = str(self.device).strip().lower()
        self.device = device if device in DEVICE_CHOICES else "auto"
        self.device_ids = list(normalize_device_ids(self.device_ids))
        self.project_root = str(Path(self.project_root).resolve())
        self.random_seed = _coerce_int(self.random_seed, field_name="random_seed")
        self.checkpoint_dir = str(self.checkpoint_dir).strip() or "data/runtime_core/checkpoints"
        self.reports_dir = str(self.reports_dir).strip() or "data/runtime_core/reports"
        self.test_set_enabled = _coerce_bool(self.test_set_enabled, field_name="test_set_enabled")
        self.test_set_size = max(1, _coerce_int(self.test_set_size, field_name="test_set_size"))
        self.test_set_seed = _coerce_int(self.test_set_seed, field_name="test_set_seed")
        self.eps_r = float(max(1.0e-30, _coerce_float(self.eps_r, field_name="eps_r")))
        self.theta_bounds = _coerce_theta_bounds(self.theta_bounds)
        self.extensions = dict(self.extensions)

    @property
    def project_root_path(self) -> Path:
        return Path(self.project_root)

    def resolve_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return (self.project_root_path / candidate).resolve()


def _build_refined_high_k_bins(grids: GridConfig) -> FloatArray:
    return np.linspace(
        float(grids.high_k_min),
        float(grids.high_k_max),
        int(grids.high_k_dense_bins),
        dtype=np.float64,
    )


def build_default_k_bins(grids: GridConfig) -> FloatArray:
    n_low, n_mid, n_high = _allocate_piecewise_k_counts(
        int(grids.k_work_size),
        (
            float(grids.low_k_fraction),
            float(grids.mid_k_fraction),
            float(grids.high_k_fraction),
        ),
        minimum_counts=(2, 1, 1),
    )
    low = np.logspace(
        np.log10(float(grids.k_min)),
        np.log10(float(grids.low_k_max)),
        n_low,
        dtype=np.float64,
    )
    mid = np.logspace(
        np.log10(float(grids.low_k_max)),
        np.log10(float(grids.mid_k_max)),
        n_mid + 1,
        dtype=np.float64,
    )[1:]
    high = np.logspace(
        np.log10(float(grids.mid_k_max)),
        np.log10(float(grids.k_max)),
        n_high + 1,
        dtype=np.float64,
    )[1:]
    merged = np.concatenate([low, mid, high], axis=0)
    return np.round(merged.astype(np.float64), decimals=15)


def build_default_eval_k_bins(grids: GridConfig) -> FloatArray:
    return np.logspace(
        np.log10(float(grids.k_min)),
        np.log10(float(grids.k_max)),
        int(grids.k_eval_size),
    ).astype(np.float64)


def build_default_config(project_root: Path | str | None = None) -> ValidationRuntimeConfig:
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    return ValidationRuntimeConfig(project_root=str(root))


def config_to_dict(config: ValidationRuntimeConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["project_root"] = str(Path(config.project_root).resolve())
    payload["theta_bounds"] = {
        name: [float(bounds[0]), float(bounds[1])]
        for name, bounds in config.theta_bounds.items()
    }
    return payload


def _merge_known_section(
    default_section: dict[str, Any],
    override_section: Mapping[str, Any],
    *,
    section_name: str,
) -> dict[str, Any]:
    merged = dict(default_section)
    for key, value in override_section.items():
        if key not in default_section:
            dotted = f"{section_name}.{key}"
            if _is_legacy_key(key):
                raise ValueError(
                    f"Legacy branch configuration key is no longer supported: {dotted}"
                )
            raise ValueError(f"Unknown configuration key: {dotted}")
        merged[key] = value
    return merged


def load_config(
    config_path: str | Path,
    *,
    project_root: Path | str | None = None,
) -> ValidationRuntimeConfig:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else _project_root_from_path(path)
    )
    config = build_default_config(root)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping, got {type(payload).__name__}.")

    defaults = config_to_dict(config)
    top_level = dict(defaults)
    for key, value in payload.items():
        if key not in defaults:
            if _is_legacy_key(key):
                raise ValueError(
                    f"Legacy branch configuration key is no longer supported: {key}"
                )
            raise ValueError(f"Unknown configuration key: {key}")
        if key == "extensions":
            if value is None:
                top_level[key] = {}
            elif isinstance(value, Mapping):
                top_level[key] = dict(value)
            else:
                raise ValueError("extensions must be a mapping.")
            continue
        if key == "theta_bounds":
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError("theta_bounds must be a mapping.")
            top_level[key] = {
                name: list(bounds) if isinstance(bounds, Sequence) else bounds
                for name, bounds in _coerce_theta_bounds(value).items()
            }
            continue
        if key in {"grids", "sampling", "gp", "gp_baseline", "dynamic_preprocessing", "m3", "camb"}:
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError(f"{key} must be a mapping.")
            if key == "grids":
                value = _canonicalize_section_keys(
                    value,
                    _LEGACY_GRID_KEY_ALIASES,
                    section_name=key,
                )
            top_level[key] = _merge_known_section(
                defaults[key],
                value,
                section_name=key,
            )
            continue
        if key == "representation":
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError(f"{key} must be a mapping.")
            top_level[key] = _merge_known_section(
                defaults[key],
                value,
                section_name=key,
            )
            continue
        top_level[key] = value

    return ValidationRuntimeConfig(
        project_root=str(root),
        device=top_level["device"],
        device_ids=list(top_level["device_ids"]) if top_level["device_ids"] is not None else None,
        random_seed=top_level["random_seed"],
        checkpoint_dir=top_level["checkpoint_dir"],
        reports_dir=top_level["reports_dir"],
        test_set_enabled=top_level["test_set_enabled"],
        test_set_size=top_level["test_set_size"],
        test_set_seed=top_level["test_set_seed"],
        eps_r=top_level["eps_r"],
        theta_bounds=_coerce_theta_bounds(top_level["theta_bounds"]),
        grids=GridConfig(**top_level["grids"]),
        sampling=SamplingConfig(**top_level["sampling"]),
        gp=GPConfig(**top_level["gp"]),
        gp_baseline=GPBaselineConfig(**top_level["gp_baseline"]),
        representation=RepresentationConfig(**top_level["representation"]),
        dynamic_preprocessing=DynamicPreprocessingConfig(**top_level["dynamic_preprocessing"]),
        m3=M3Config(**top_level["m3"]),
        camb=CAMBConfig(**top_level["camb"]),
        extensions=dict(top_level["extensions"]),
    )


def write_config_template(
    output_path: str | Path,
    *,
    project_root: Path | str | None = None,
) -> Path:
    path = Path(output_path).resolve()
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else _project_root_from_path(path)
    )
    config = build_default_config(root)
    payload = config_to_dict(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
