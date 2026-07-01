"""Reusable spectrum-bank caching for training and validation data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from z2quijote.runtime_core.camb_data_provider import CAMBAccuracyConfig, CAMBDataProvider
from z2quijote.runtime_core.config import (
    build_default_k_bins,
    ValidationRuntimeConfig,
)
from z2quijote.runtime_core.data_source import resolve_data_source
from z2quijote.runtime_core.quijote_k_grid import maybe_build_quijote_output_k_bins
from z2quijote.runtime_core.quijote_gp_surrogate import load_quijote_gp_surrogate
from z2quijote.runtime_core.representation import parse_target_transform, target_transform_name
from z2quijote.runtime_core.sampling import (
    ensure_2d_theta_batch,
    generate_sobol_thetas,
    generate_test_set_thetas,
)

VALIDATION_SAMPLING_METHOD = "latin_hypercube"

ProgressCallback = Callable[[str, int, int], None]


def _slugify(text: str) -> str:
    out = []
    for char in str(text).strip().lower():
        if char.isalnum():
            out.append(char)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "cache"


def _array_digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return sha256(arr.tobytes()).hexdigest()


def _load_theta_array_from_path(
    path: Path,
    *,
    requested_key: str,
) -> tuple[np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(f"Custom initial theta file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        preferred_keys = (
            requested_key,
            "raw_thetas",
            "theta_raw",
            "theta_raw_final",
            "theta_raw_initial",
            "thetas",
            "theta",
            "theta_unit",
            "theta_unit_final",
            "theta_unit_initial",
            "unit_thetas",
        )
        with np.load(path, allow_pickle=False) as npz:
            selected = ""
            for key in preferred_keys:
                if key and key in npz.files:
                    selected = key
                    break
            if not selected and len(npz.files) == 1:
                selected = str(npz.files[0])
            if not selected:
                raise KeyError(
                    "Custom initial theta npz must contain one theta array. "
                    f"Available keys: {list(npz.files)}"
                )
            return np.asarray(npz[selected], dtype=np.float64), selected
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64), requested_key
    try:
        return np.asarray(np.loadtxt(path, delimiter=","), dtype=np.float64), requested_key
    except ValueError:
        return np.asarray(np.loadtxt(path), dtype=np.float64), requested_key


def _truthy_metadata_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_initial_manifest(
    config: ValidationRuntimeConfig,
    source_path: Path,
    metadata: dict[str, object],
) -> tuple[Path | None, dict[str, object] | None]:
    explicit = str(
        metadata.get("initial_training_manifest_path")
        or metadata.get("initial_training_theta_manifest_path")
        or metadata.get("initial_design_manifest_path")
        or ""
    ).strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(config.resolve_path(explicit))
    adjacent = source_path.with_suffix(".json")
    if adjacent.exists():
        candidates.append(adjacent)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            raise FileNotFoundError(f"Custom initial theta manifest not found: {resolved}")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Custom initial theta manifest must be a JSON object: {resolved}")
        return resolved, payload
    return None, None


def _manifest_target_kind(manifest: dict[str, object]) -> str:
    explicit = str(
        manifest.get("source_training_target_kind")
        or manifest.get("training_target_kind")
        or manifest.get("training_target")
        or ""
    ).strip().lower()
    if explicit in {"logdiff", "logdiff_hmcode2020_anchor", "log_hi_minus_log_hmcode2020_anchor"}:
        return "logdiff"
    if explicit in {"direct", "direct_log", "direct_logpk", "native", "native_logpk"}:
        return "direct_logpk"

    hints = " ".join(
        str(manifest.get(key, ""))
        for key in (
            "source_target",
            "target",
            "source_validation_cache",
            "validation_cache",
            "source",
            "label",
        )
    ).lower()
    if "direct_logpk" in hints or "directlogpk" in hints:
        return "direct_logpk"
    if "logdiff" in hints or "log_hi_minus_log_" in hints:
        return "logdiff"
    return ""


def _normalize_matter_power_var(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"delta_cdm", "cdm", "cdm_auto", "pk_cdm", "quijote_cdm"}:
        return "delta_cdm"
    if raw in {"delta_tot", "delta_total", "total", "matter", "dark_matter", "quijote_total"}:
        return "delta_tot"
    return raw


def _manifest_anchor_matter_power_var(manifest: dict[str, object]) -> str:
    for key in (
        "source_anchor_matter_power_var",
        "anchor_matter_power_var",
        "source_matter_power_var",
    ):
        resolved = _normalize_matter_power_var(manifest.get(key))
        if resolved:
            return resolved
    hints = " ".join(
        str(manifest.get(key, ""))
        for key in (
            "source",
            "source_target",
            "target",
            "source_validation_cache",
            "validation_cache",
            "label",
        )
    ).lower()
    if "delta_cdm" in hints or "cdm_hmcode" in hints or "_cdm_" in hints:
        return "delta_cdm"
    if "delta_tot" in hints or "total_hmcode" in hints:
        return "delta_tot"
    return ""


def _current_anchor_matter_power_var(config: ValidationRuntimeConfig) -> str:
    data_source = resolve_data_source(config)
    if data_source.name != "quijote":
        return ""
    metadata = data_source.metadata
    explicit = (
        metadata.get("anchor_matter_power_var")
        or metadata.get("matter_power_var")
        or metadata.get("anchor_power_var")
    )
    resolved = _normalize_matter_power_var(explicit)
    if resolved:
        return resolved
    provider_name = str(
        metadata.get("anchor_provider")
        or metadata.get("linear_anchor_provider")
        or metadata.get("anchor_power_provider")
        or ""
    ).strip().lower()
    if "cdm" in provider_name:
        return "delta_cdm"
    if "hmcode2020" in provider_name:
        return "delta_tot"
    return ""


def _current_target_kind(config: ValidationRuntimeConfig) -> str:
    target = target_transform_name(
        transform_family=str(config.representation.transform_family),
        anchor_mode=str(config.representation.anchor_mode),
    )
    family, _ = parse_target_transform(target)
    return str(family)


def _initial_manifest_metadata(
    config: ValidationRuntimeConfig,
    source_path: Path,
    metadata: dict[str, object],
) -> dict[str, object]:
    manifest_path, manifest = _load_initial_manifest(config, source_path, metadata)
    if manifest is None or manifest_path is None:
        return {}

    source_kind = _manifest_target_kind(manifest)
    current_kind = _current_target_kind(config)
    if source_kind and source_kind != current_kind:
        allowed = _truthy_metadata_flag(
            metadata.get("allow_initial_training_target_mismatch")
            or metadata.get("initial_training_allow_target_mismatch")
        )
        if not allowed:
            raise ValueError(
                "Custom initial theta target mismatch: "
                f"manifest {manifest_path} declares source target {source_kind!r}, "
                f"but current representation target is {current_kind!r}. "
                "Use a matching best64 design or set "
                "extensions.<data_source>.allow_initial_training_target_mismatch=true "
                "for an explicit diagnostic-only run."
            )

    source_anchor = _manifest_anchor_matter_power_var(manifest)
    current_anchor = _current_anchor_matter_power_var(config)
    if source_anchor and current_anchor and source_anchor != current_anchor:
        allowed = _truthy_metadata_flag(
            metadata.get("allow_initial_anchor_matter_power_mismatch")
            or metadata.get("initial_training_allow_anchor_matter_power_mismatch")
        )
        if not allowed:
            raise ValueError(
                "Custom initial theta anchor matter-power mismatch: "
                f"manifest {manifest_path} declares anchor matter power {source_anchor!r}, "
                f"but current Quijote anchor matter power is {current_anchor!r}. "
                "Use a matching best64 design or set "
                "extensions.<data_source>.allow_initial_anchor_matter_power_mismatch=true "
                "for an explicit diagnostic-only run."
            )

    result: dict[str, object] = {
        "initial_training_manifest_path": str(manifest_path),
    }
    if source_kind:
        result["initial_training_manifest_target_kind"] = source_kind
    if source_anchor:
        result["initial_training_manifest_anchor_matter_power_var"] = source_anchor
    for source_key, output_key in (
        ("source_target", "initial_training_manifest_target"),
        ("source_validation_cache", "initial_training_manifest_validation_cache"),
        ("source_p68_improvement_fraction", "initial_training_manifest_p68_improvement_fraction"),
        ("source_overall_p68", "initial_training_manifest_overall_p68"),
    ):
        if source_key in manifest:
            result[output_key] = manifest[source_key]
    return result


def _custom_initial_training_thetas(
    config: ValidationRuntimeConfig,
    data_source: Any,
) -> tuple[np.ndarray, dict[str, object]] | None:
    metadata = dict(data_source.metadata)
    raw_path = str(
        metadata.get("initial_training_theta_path")
        or metadata.get("initial_theta_path")
        or ""
    ).strip()
    if not raw_path:
        return None

    source_path = config.resolve_path(raw_path)
    requested_key = str(metadata.get("initial_training_theta_key", "") or "").strip()
    theta_array, selected_key = _load_theta_array_from_path(
        source_path,
        requested_key=requested_key,
    )
    input_space = str(metadata.get("initial_training_theta_space", "") or "").strip().lower()
    if not input_space:
        key_hint = selected_key.lower()
        input_space = "unit" if "unit" in key_hint else "raw"
    raw_thetas, _ = ensure_2d_theta_batch(
        theta_array,
        data_source.theta_bounds,
        input_space=input_space,
    )
    expected_count = int(config.sampling.initial_sobol_points)
    if int(raw_thetas.shape[0]) != expected_count:
        raise ValueError(
            "Custom initial theta file row count must match "
            f"sampling.initial_sobol_points={expected_count}, got {raw_thetas.shape[0]}."
        )
    source_label = str(
        metadata.get("initial_training_label")
        or metadata.get("initial_training_theta_label")
        or source_path.stem
    )
    source_metadata: dict[str, object] = {
        "initial_training_source": "custom_theta_path",
        "initial_training_theta_path": str(source_path),
        "initial_training_theta_key": str(selected_key),
        "initial_training_theta_space": str(input_space),
        "initial_training_label": source_label,
        "initial_training_theta_digest": _array_digest(raw_thetas),
        **_initial_manifest_metadata(config, source_path, metadata),
    }
    return raw_thetas, source_metadata


def _initial_training_request(
    config: ValidationRuntimeConfig,
) -> tuple[str, np.ndarray, str, dict[str, object]]:
    data_source = resolve_data_source(config)
    storage_k_count = int(build_storage_k_bins(config).shape[0])
    custom = _custom_initial_training_thetas(config, data_source)
    if custom is None:
        raw_thetas = generate_sobol_thetas(
            data_source.theta_bounds,
            int(config.sampling.initial_sobol_points),
            int(config.sampling.initial_seed),
        )
        return (
            (
                f"initial_training_n{int(config.sampling.initial_sobol_points)}"
                f"_seed{int(config.sampling.initial_seed)}"
                f"_k{storage_k_count}"
                f"_{data_source.cache_prefix}"
                f"_{data_source.spectrum_type}"
            ),
            raw_thetas,
            "cached_initial_training",
            {
                "seed": int(config.sampling.initial_seed),
                "sample_size": int(config.sampling.initial_sobol_points),
                "data_source": str(data_source.name),
            },
        )

    raw_thetas, source_metadata = custom
    source_digest = str(source_metadata["initial_training_theta_digest"])
    label = _slugify(str(source_metadata["initial_training_label"]))
    return (
        (
            f"initial_training_custom_{label}"
            f"_n{int(config.sampling.initial_sobol_points)}"
            f"_k{storage_k_count}"
            f"_{data_source.cache_prefix}"
            f"_{data_source.spectrum_type}"
            f"_{source_digest[:12]}"
        ),
        raw_thetas,
        "cached_custom_initial_training",
        {
            "seed": int(config.sampling.initial_seed),
            "sample_size": int(config.sampling.initial_sobol_points),
            "data_source": str(data_source.name),
            **source_metadata,
        },
    )


def _comparison_training_request(
    config: ValidationRuntimeConfig,
    *,
    train_points: int,
) -> tuple[str, np.ndarray, str, dict[str, object]]:
    data_source = resolve_data_source(config)
    resolved_points = int(train_points)
    storage_k_count = int(build_storage_k_bins(config).shape[0])
    policy = str(
        data_source.metadata.get("fixed_budget_initial_policy")
        or data_source.metadata.get("comparison_initial_training_policy")
        or "sobol"
    ).strip().lower()
    sobol_thetas = generate_sobol_thetas(
        data_source.theta_bounds,
        resolved_points,
        int(config.gp_baseline.random_seed),
    )
    metadata: dict[str, object] = {
        "seed": int(config.gp_baseline.random_seed),
        "sample_size": resolved_points,
        "data_source": str(data_source.name),
        "fixed_budget_initial_policy": "sobol",
    }
    cache_label = (
        f"fixed_budget_training_n{resolved_points}"
        f"_seed{int(config.gp_baseline.random_seed)}"
        f"_k{storage_k_count}"
        f"_{data_source.cache_prefix}"
        f"_{data_source.spectrum_type}"
    )

    if policy in {"initial_then_sobol_tail", "initial_sobol_then_sobol_tail", "same_initial_then_sobol_tail"}:
        initial_count = int(config.sampling.initial_sobol_points)
        if resolved_points < initial_count:
            raise ValueError(
                "fixed budget train_points must be >= initial Sobol row count, "
                f"got train_points={resolved_points}, initial rows={initial_count}."
            )
        raw_thetas = generate_sobol_thetas(
            data_source.theta_bounds,
            resolved_points,
            int(config.sampling.initial_seed),
        )
        cache_label = (
            f"fixed_budget_training_initial_sobol_plus_sobol_tail"
            f"_n{resolved_points}"
            f"_seed{int(config.sampling.initial_seed)}"
            f"_k{storage_k_count}"
            f"_{data_source.cache_prefix}"
            f"_{data_source.spectrum_type}"
        )
        metadata = {
            **metadata,
            "seed": int(config.sampling.initial_seed),
            "fixed_budget_initial_policy": "initial_then_sobol_tail",
            "initial_training_seed": int(config.sampling.initial_seed),
            "sobol_tail_start_index": initial_count,
            "sobol_tail_count": int(resolved_points - initial_count),
        }
    elif policy in {"custom_then_sobol_tail", "prepend_custom_then_sobol_tail", "custom_initial_then_sobol_tail"}:
        custom = _custom_initial_training_thetas(config, data_source)
        if custom is None:
            raise ValueError(
                "fixed budget policy custom_then_sobol_tail requires "
                "extensions.<data_source>.initial_training_theta_path."
            )
        custom_raw, source_metadata = custom
        initial_count = int(custom_raw.shape[0])
        if resolved_points < initial_count:
            raise ValueError(
                "fixed budget train_points must be >= custom initial row count, "
                f"got train_points={resolved_points}, custom rows={initial_count}."
            )
        raw_thetas = np.vstack([custom_raw, sobol_thetas[initial_count:resolved_points]])
        source_digest = str(source_metadata["initial_training_theta_digest"])
        label = _slugify(str(source_metadata["initial_training_label"]))
        cache_label = (
            f"fixed_budget_training_custom_{label}_plus_sobol_tail"
            f"_n{resolved_points}"
            f"_seed{int(config.gp_baseline.random_seed)}"
            f"_k{storage_k_count}"
            f"_{data_source.cache_prefix}"
            f"_{data_source.spectrum_type}"
            f"_{source_digest[:12]}"
        )
        metadata = {
            **metadata,
            "fixed_budget_initial_policy": "custom_then_sobol_tail",
            "sobol_tail_start_index": initial_count,
            "sobol_tail_count": int(resolved_points - initial_count),
            **source_metadata,
        }
    else:
        raw_thetas = sobol_thetas

    return cache_label, raw_thetas, "cached_fixed_budget_training", metadata


def _camb_signature(config: ValidationRuntimeConfig) -> dict[str, object]:
    camb = config.camb
    return {
        "backend_name": str(camb.backend_name),
        "allow_placeholder_backend": bool(camb.allow_placeholder_backend),
        "spectrum_type": str(camb.spectrum_type),
        "camb_hifi_highk_enabled": bool(camb.camb_hifi_highk_enabled),
        "camb_hifi_highk_kmin": float(camb.camb_hifi_highk_kmin),
        "camb_hifi_highk_kmax": float(camb.camb_hifi_highk_kmax),
        "camb_hifi_accuracy_boost": float(camb.camb_hifi_accuracy_boost),
        "camb_hifi_l_accuracy_boost": float(camb.camb_hifi_l_accuracy_boost),
        "camb_hifi_sampling_boost": float(camb.camb_hifi_sampling_boost),
        "camb_hifi_k_per_logint": int(camb.camb_hifi_k_per_logint),
        "camb_hifi_halofit_version": str(camb.camb_hifi_halofit_version),
        "camb_hifi_use_high_precision_transfer": bool(camb.camb_hifi_use_high_precision_transfer),
    }


def cache_root(config: ValidationRuntimeConfig) -> Path:
    root = config.resolve_path(resolve_data_source(config).cache_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_storage_k_bins(config: ValidationRuntimeConfig) -> np.ndarray:
    data_source = resolve_data_source(config)
    if data_source.name == "quijote":
        output_k_bins = maybe_build_quijote_output_k_bins(data_source.metadata)
        if output_k_bins is not None:
            return output_k_bins
        bank_path = str(data_source.metadata.get("bank_path", "")).strip()
        if bank_path:
            resolved_bank = config.resolve_path(bank_path)
            if resolved_bank.exists():
                with np.load(resolved_bank, allow_pickle=False) as npz:
                    return np.asarray(npz["k_bins"], dtype=np.float64).reshape(-1)
        surrogate_path = str(data_source.metadata.get("surrogate_path", "")).strip()
        if surrogate_path:
            resolved_surrogate = config.resolve_path(surrogate_path)
            if resolved_surrogate.exists():
                surrogate = load_quijote_gp_surrogate(resolved_surrogate)
                return np.asarray(surrogate.k_bins, dtype=np.float64).reshape(-1)
        raise FileNotFoundError(
            "Quijote data source requires a readable bank_path or surrogate_path "
            "so runtime caches can use the native Quijote k grid."
        )
    storage_size = max(10000, int(config.grids.k_eval_size))
    return np.logspace(
        np.log10(float(config.grids.k_min)),
        np.log10(float(config.grids.k_max)),
        storage_size,
    ).astype(np.float64)


@dataclass(slots=True)
class SpectrumBank:
    name: str
    npz_path: Path
    metadata_path: Path
    raw_thetas: np.ndarray
    k_bins: np.ndarray
    p_nonlin_batch: np.ndarray
    p_linear_batch: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def point_count(self) -> int:
        return int(self.raw_thetas.shape[0])


@dataclass(slots=True)
class SpectrumBankInspection:
    name: str
    npz_path: Path
    metadata_path: Path
    status: str
    expected_point_count: int
    stage: str
    asset_version: str
    details: str = ""


def _expected_metadata(
    *,
    config: ValidationRuntimeConfig,
    cache_name: str,
    raw_thetas: np.ndarray,
    k_bins: np.ndarray,
    stage: str,
    asset_version: str,
    extras: dict[str, object] | None,
) -> dict[str, object]:
    data_source = resolve_data_source(config)
    data_source_metadata = dict(data_source.metadata)
    data_source_metadata.pop("target_transform", None)
    return {
        "cache_name": str(cache_name),
        "data_source": str(data_source.name),
        "parameter_space": str(data_source.parameter_space),
        "theta_names": list(data_source.theta_names),
        "stage": str(stage),
        "asset_version": str(asset_version),
        "point_count": int(raw_thetas.shape[0]),
        "theta_dim": int(raw_thetas.shape[1]),
        "k_bin_count": int(k_bins.shape[0]),
        "theta_bounds": np.asarray(data_source.theta_bounds, dtype=np.float64).tolist(),
        "theta_digest": _array_digest(raw_thetas),
        "k_digest": _array_digest(k_bins),
        "camb_signature": _camb_signature(config) if data_source.name == "camb" else {},
        "data_source_signature": {
            "name": str(data_source.name),
            "parameter_space": str(data_source.parameter_space),
            "provider_kind": str(data_source.provider_kind),
            "has_linear_anchor": bool(data_source.has_linear_anchor),
            "metadata": data_source_metadata,
        },
        "extras": dict(extras or {}),
    }


def _load_bank(npz_path: Path, metadata_path: Path) -> SpectrumBank:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as npz:
        raw_thetas = np.asarray(npz["raw_thetas"], dtype=np.float64)
        k_bins = np.asarray(npz["k_bins"], dtype=np.float64)
        p_nonlin_batch = np.asarray(npz["p_nonlin_batch"], dtype=np.float64)
        p_linear_batch = None
        if "p_linear_batch" in npz.files:
            p_linear_batch = np.asarray(npz["p_linear_batch"], dtype=np.float64)
    return SpectrumBank(
        name=str(metadata.get("cache_name", npz_path.stem)),
        npz_path=npz_path.resolve(),
        metadata_path=metadata_path.resolve(),
        raw_thetas=raw_thetas,
        k_bins=k_bins,
        p_nonlin_batch=p_nonlin_batch,
        p_linear_batch=p_linear_batch,
        metadata=metadata,
    )


def _bank_matches(metadata: dict[str, object], expected: dict[str, object]) -> bool:
    keys = (
        "cache_name",
        "stage",
        "asset_version",
        "point_count",
        "theta_dim",
        "k_bin_count",
        "theta_digest",
        "k_digest",
        "theta_bounds",
        "camb_signature",
        "data_source",
        "parameter_space",
        "theta_names",
        "data_source_signature",
        "extras",
    )
    return all(metadata.get(key) == expected.get(key) for key in keys)


def _save_bank(
    *,
    name: str,
    npz_path: Path,
    metadata_path: Path,
    raw_thetas: np.ndarray,
    k_bins: np.ndarray,
    p_nonlin_batch: np.ndarray,
    p_linear_batch: np.ndarray | None,
    metadata: dict[str, object],
) -> SpectrumBank:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if p_linear_batch is None:
        np.savez_compressed(
            npz_path,
            raw_thetas=np.asarray(raw_thetas, dtype=np.float64),
            k_bins=np.asarray(k_bins, dtype=np.float64),
            p_nonlin_batch=np.asarray(p_nonlin_batch, dtype=np.float64),
        )
    else:
        np.savez_compressed(
            npz_path,
            raw_thetas=np.asarray(raw_thetas, dtype=np.float64),
            k_bins=np.asarray(k_bins, dtype=np.float64),
            p_nonlin_batch=np.asarray(p_nonlin_batch, dtype=np.float64),
            p_linear_batch=np.asarray(p_linear_batch, dtype=np.float64),
        )
    metadata_payload = dict(metadata)
    metadata_payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    metadata_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _load_bank(npz_path, metadata_path)


def _prepare_bank_request(
    config: ValidationRuntimeConfig,
    *,
    cache_name: str,
    raw_thetas: np.ndarray,
    asset_version: str,
    stage: str,
    k_bins: np.ndarray | None,
    metadata: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, Path, Path, dict[str, object]]:
    raw_thetas_arr = np.asarray(raw_thetas, dtype=np.float64)
    if raw_thetas_arr.ndim != 2:
        raise ValueError(f"raw_thetas must be 2D, got {raw_thetas_arr.shape}.")
    k_storage = (
        np.asarray(k_bins, dtype=np.float64).reshape(-1)
        if k_bins is not None
        else build_storage_k_bins(config)
    )
    cache_dir = cache_root(config)
    stem = _slugify(cache_name)
    npz_path = (cache_dir / f"{stem}.npz").resolve()
    metadata_path = (cache_dir / f"{stem}.json").resolve()
    expected = _expected_metadata(
        config=config,
        cache_name=cache_name,
        raw_thetas=raw_thetas_arr,
        k_bins=k_storage,
        stage=stage,
        asset_version=asset_version,
        extras=metadata,
    )
    return raw_thetas_arr, k_storage, npz_path, metadata_path, expected


def inspect_spectrum_bank(
    config: ValidationRuntimeConfig,
    *,
    cache_name: str,
    raw_thetas: np.ndarray,
    asset_version: str,
    stage: str,
    k_bins: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> SpectrumBankInspection:
    raw_thetas_arr, _, npz_path, metadata_path, expected = _prepare_bank_request(
        config,
        cache_name=cache_name,
        raw_thetas=raw_thetas,
        asset_version=asset_version,
        stage=stage,
        k_bins=k_bins,
        metadata=metadata,
    )
    if npz_path.exists() and metadata_path.exists():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _bank_matches(loaded, expected):
            status = "hit"
            details = "Exact cache metadata match; cache can be reused."
        else:
            status = "stale"
            details = "Cache files exist but metadata no longer matches current request."
    else:
        status = "missing"
        details = "No compatible cache files found for this request."
    return SpectrumBankInspection(
        name=str(cache_name),
        npz_path=npz_path,
        metadata_path=metadata_path,
        status=status,
        expected_point_count=int(raw_thetas_arr.shape[0]),
        stage=str(stage),
        asset_version=str(asset_version),
        details=details,
    )


def get_or_create_spectrum_bank(
    config: ValidationRuntimeConfig,
    camb_data_provider: CAMBDataProvider,
    *,
    cache_name: str,
    raw_thetas: np.ndarray,
    asset_version: str,
    stage: str,
    progress_callback: ProgressCallback | None = None,
    force_rebuild: bool = False,
    k_bins: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> SpectrumBank:
    raw_thetas, k_storage, npz_path, metadata_path, expected = _prepare_bank_request(
        config,
        cache_name=cache_name,
        raw_thetas=raw_thetas,
        asset_version=asset_version,
        stage=stage,
        k_bins=k_bins,
        metadata=metadata,
    )
    if not force_rebuild and npz_path.exists() and metadata_path.exists():
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _bank_matches(loaded, expected):
            bank = _load_bank(npz_path, metadata_path)
            bank.metadata["cache_status"] = "hit"
            return bank

    p_nonlin_rows: list[np.ndarray] = []
    p_linear_rows: list[np.ndarray] = []
    saw_linear_anchor = False
    total = int(raw_thetas.shape[0])
    for idx, theta in enumerate(raw_thetas):
        if progress_callback is not None:
            progress_callback(stage, idx + 1, total)
        result = camb_data_provider.run_hifi_anchor(
            theta=np.asarray(theta, dtype=np.float64),
            k_bins=k_storage,
            accuracy_config=CAMBAccuracyConfig(mode="hifi"),
            asset_version=asset_version,
        )
        p_linear = result.get("P_linear")
        if p_linear is not None:
            saw_linear_anchor = True
            p_linear_rows.append(np.asarray(p_linear, dtype=np.float64))
        p_nonlin_rows.append(np.asarray(result["P_nonlin_hifi"], dtype=np.float64))

    bank = _save_bank(
        name=cache_name,
        npz_path=npz_path,
        metadata_path=metadata_path,
        raw_thetas=raw_thetas,
        k_bins=k_storage,
        p_nonlin_batch=np.vstack(p_nonlin_rows).astype(np.float64),
        p_linear_batch=(
            np.vstack(p_linear_rows).astype(np.float64)
            if saw_linear_anchor
            else None
        ),
        metadata=expected,
    )
    bank.metadata["cache_status"] = "built"
    return bank


def get_or_create_initial_training_bank(
    config: ValidationRuntimeConfig,
    camb_data_provider: CAMBDataProvider,
    *,
    progress_callback: ProgressCallback | None = None,
    force_rebuild: bool = False,
) -> SpectrumBank:
    cache_name, raw_thetas, asset_version, metadata = _initial_training_request(config)
    return get_or_create_spectrum_bank(
        config,
        camb_data_provider,
        cache_name=cache_name,
        raw_thetas=raw_thetas,
        asset_version=asset_version,
        stage="cache_initial_hifi",
        progress_callback=progress_callback,
        force_rebuild=force_rebuild,
        metadata=metadata,
    )


def inspect_initial_training_bank(config: ValidationRuntimeConfig) -> SpectrumBankInspection:
    cache_name, raw_thetas, asset_version, metadata = _initial_training_request(config)
    return inspect_spectrum_bank(
        config,
        cache_name=cache_name,
        raw_thetas=raw_thetas,
        asset_version=asset_version,
        stage="cache_initial_hifi",
        metadata=metadata,
    )


def get_or_create_comparison_training_bank(
    config: ValidationRuntimeConfig,
    camb_data_provider: CAMBDataProvider,
    *,
    train_points: int | None = None,
    progress_callback: ProgressCallback | None = None,
    force_rebuild: bool = False,
) -> SpectrumBank:
    resolved_points = int(train_points or config.sampling.total_budget)
    cache_name, raw_thetas, asset_version, metadata = _comparison_training_request(
        config,
        train_points=resolved_points,
    )
    return get_or_create_spectrum_bank(
        config,
        camb_data_provider,
        cache_name=cache_name,
        raw_thetas=raw_thetas,
        asset_version=asset_version,
        stage="cache_comparison_hifi",
        progress_callback=progress_callback,
        force_rebuild=force_rebuild,
        metadata=metadata,
    )


def inspect_comparison_training_bank(
    config: ValidationRuntimeConfig,
    *,
    train_points: int | None = None,
) -> SpectrumBankInspection:
    resolved_points = int(train_points or config.sampling.total_budget)
    cache_name, raw_thetas, asset_version, metadata = _comparison_training_request(
        config,
        train_points=resolved_points,
    )
    return inspect_spectrum_bank(
        config,
        cache_name=cache_name,
        raw_thetas=raw_thetas,
        asset_version=asset_version,
        stage="cache_comparison_hifi",
        metadata=metadata,
    )


def get_or_create_validation_bank(
    config: ValidationRuntimeConfig,
    camb_data_provider: CAMBDataProvider,
    *,
    test_set_size: int | None = None,
    test_set_seed: int | None = None,
    progress_callback: ProgressCallback | None = None,
    force_rebuild: bool = False,
) -> SpectrumBank:
    data_source = resolve_data_source(config)
    resolved_size = int(test_set_size or config.test_set_size)
    resolved_seed = int(config.test_set_seed if test_set_seed is None else test_set_seed)
    storage_k_count = int(build_storage_k_bins(config).shape[0])
    raw_thetas = generate_test_set_thetas(
        data_source.theta_bounds,
        resolved_size,
        resolved_seed,
    )
    bank = get_or_create_spectrum_bank(
        config,
        camb_data_provider,
        cache_name=(
            f"validation_truth_n{resolved_size}"
            f"_seed{resolved_seed}"
            f"_{VALIDATION_SAMPLING_METHOD}"
            f"_k{storage_k_count}"
            f"_{data_source.cache_prefix}"
            f"_{data_source.spectrum_type}"
        ),
        raw_thetas=raw_thetas,
        asset_version=f"cached_validation_truth_{VALIDATION_SAMPLING_METHOD}",
        stage="cache_validation_truth",
        progress_callback=progress_callback,
        force_rebuild=force_rebuild,
        metadata={
            "seed": resolved_seed,
            "sample_size": resolved_size,
            "sampling_method": VALIDATION_SAMPLING_METHOD,
            "data_source": str(data_source.name),
        },
    )
    bank.metadata["sampling_method"] = VALIDATION_SAMPLING_METHOD
    return bank


def inspect_validation_bank(
    config: ValidationRuntimeConfig,
    *,
    test_set_size: int | None = None,
    test_set_seed: int | None = None,
) -> SpectrumBankInspection:
    data_source = resolve_data_source(config)
    resolved_size = int(test_set_size or config.test_set_size)
    resolved_seed = int(config.test_set_seed if test_set_seed is None else test_set_seed)
    storage_k_count = int(build_storage_k_bins(config).shape[0])
    raw_thetas = generate_test_set_thetas(
        data_source.theta_bounds,
        resolved_size,
        resolved_seed,
    )
    return inspect_spectrum_bank(
        config,
        cache_name=(
            f"validation_truth_n{resolved_size}"
            f"_seed{resolved_seed}"
            f"_{VALIDATION_SAMPLING_METHOD}"
            f"_k{storage_k_count}"
            f"_{data_source.cache_prefix}"
            f"_{data_source.spectrum_type}"
        ),
        raw_thetas=raw_thetas,
        asset_version=f"cached_validation_truth_{VALIDATION_SAMPLING_METHOD}",
        stage="cache_validation_truth",
        metadata={
            "seed": resolved_seed,
            "sample_size": resolved_size,
            "sampling_method": VALIDATION_SAMPLING_METHOD,
            "data_source": str(data_source.name),
        },
    )
