"""Data-source routing helpers for CAMB and isolated Quijote workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from z2quijote.runtime_core.config import THETA_NAMES, ValidationRuntimeConfig, theta_bounds_as_array
from z2quijote.runtime_core.quijote_gp_data_provider import (
    QuijoteCAMBCDMMHMCODE2020AnchorProvider,
    QuijoteCAMBHMCODE2020AnchorProvider,
    QuijoteCAMBLinearAnchorProvider,
    QuijoteGPDataProvider,
    QuijoteLinearGeneratorProvider,
    QuijoteOfficialLinearAnchorProvider,
)
from z2quijote.runtime_core.quijote_gp_surrogate import (
    DEFAULT_QUIJOTE_BSQ5_BOUNDS,
    QUIJOTE_BSQ5_PARAMETER_SPACE,
    QUIJOTE_BSQ_THETA_NAMES,
    quijote_theta_bounds_as_array,
)
from z2quijote.runtime_core.representation import target_transform_name

DEFAULT_QUIJOTE_RUNTIME_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "parameter_space": QUIJOTE_BSQ5_PARAMETER_SPACE,
    "theta_names": list(QUIJOTE_BSQ_THETA_NAMES),
    "theta_bounds": {
        name: [float(low), float(high)]
        for name, (low, high) in DEFAULT_QUIJOTE_BSQ5_BOUNDS.items()
    },
    "raw_root": "artifacts/quijote/raw/Pk/matter/BSQ",
    "params_file": "artifacts/quijote/raw/BSQ_params.txt",
    "power_file_name": "Pk_CDM_z=0.000.dat",
    "bank_path": "data/source_data/raw_bank/artifacts/quijote/cache/quijote_bsq_z0_bank.npz",
    "bank_metadata_path": "data/source_data/raw_bank/artifacts/quijote/cache/quijote_bsq_z0_bank.json",
    "bank_manifest_path": "data/source_data/raw_bank/artifacts/quijote/cache/quijote_bsq_z0_bank_manifest.csv",
    "surrogate_path": "data/source_data/v2_quijote/artifacts/quijote/gp_surrogates/quijote_bsq_z0_gp.pkl",
    "truth_generator_device": "auto",
    "runtime_cache_root": "artifacts/quijote/runtime_cache",
    "redshift": 0.0,
    "redshift_label": "0.000",
    "space": "real",
    "field": "CDM",
    "spectrum_type": "quijote_cdm",
    "has_linear_anchor": True,
    "anchor_provider": "",
    "linear_anchor_provider": "quijote_official_linear_svgp_generator",
    "linear_power_file_name": "Pk_mm_z=0.000.txt",
    "linear_generator_path": (
        "data/source_data/v2_quijote/artifacts/quijote/gp_surrogates/"
        "quijote_bsq_z0_official_linear_directlogpk_generator.pkl"
    ),
    "linear_generator_device": "auto",
    "linear_anchor_reference_as": 2.1e-9,
}


@dataclass(slots=True)
class DataSourceSpec:
    name: str
    parameter_space: str
    theta_names: tuple[str, ...]
    theta_bounds: np.ndarray
    cache_root: str
    cache_prefix: str
    spectrum_type: str
    target_transform: str
    has_linear_anchor: bool
    provider_kind: str
    metadata: dict[str, Any]

    @property
    def theta_dim(self) -> int:
        return int(self.theta_bounds.shape[0])


def _extension_mapping(config: ValidationRuntimeConfig, name: str) -> dict[str, Any]:
    value = config.extensions.get(name, {})
    if not isinstance(value, Mapping):
        return {}
    return dict(value)


def configured_data_source_name(config: ValidationRuntimeConfig) -> str:
    raw = config.extensions.get("data_source", "")
    if isinstance(raw, Mapping):
        raw = raw.get("name", "")
    if raw in (None, ""):
        quijote = _extension_mapping(config, "quijote")
        if bool(quijote.get("use_in_pipeline", False)):
            raw = "quijote"
        else:
            raw = "camb"
    normalized = str(raw).strip().lower()
    aliases = {
        "default": "camb",
        "camb8": "camb",
        "camb_8d": "camb",
        "quijote_gp": "quijote",
        "quijote_bsq": "quijote",
        "quijote_bsq5": "quijote",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"camb", "quijote"}:
        raise ValueError(f"Unsupported data source {raw!r}. Expected 'camb' or 'quijote'.")
    return normalized


def set_configured_data_source(config: ValidationRuntimeConfig, name: str) -> None:
    normalized = str(name).strip().lower()
    if normalized not in {"camb", "quijote"}:
        raise ValueError(f"Unsupported data source {name!r}.")
    config.extensions["data_source"] = normalized
    if normalized == "quijote":
        quijote = _extension_mapping(config, "quijote")
        defaults = dict(DEFAULT_QUIJOTE_RUNTIME_SETTINGS)
        defaults.update(quijote)
        quijote = defaults
        quijote["enabled"] = True
        quijote["use_in_pipeline"] = True
        config.extensions["quijote"] = quijote


def _coerce_named_bounds(
    raw_bounds: Mapping[str, Sequence[float]] | np.ndarray | None,
    *,
    fallback: Mapping[str, Sequence[float]],
) -> np.ndarray:
    if raw_bounds is None:
        raw_bounds = fallback
    return quijote_theta_bounds_as_array(raw_bounds)


def resolve_data_source(config: ValidationRuntimeConfig) -> DataSourceSpec:
    name = configured_data_source_name(config)
    active_target_transform = target_transform_name(
        transform_family=str(config.representation.transform_family),
        anchor_mode=str(config.representation.anchor_mode),
    )
    if name == "camb":
        bounds = theta_bounds_as_array(config.theta_bounds)
        return DataSourceSpec(
            name="camb",
            parameter_space="camb8",
            theta_names=tuple(THETA_NAMES),
            theta_bounds=bounds,
            cache_root="artifacts/cache",
            cache_prefix="camb",
            spectrum_type=str(config.camb.spectrum_type),
            target_transform=active_target_transform,
            has_linear_anchor=True,
            provider_kind="camb",
            metadata={},
        )

    quijote = _extension_mapping(config, "quijote")
    defaults = dict(DEFAULT_QUIJOTE_RUNTIME_SETTINGS)
    defaults.update(quijote)
    quijote = defaults
    bounds = _coerce_named_bounds(
        quijote.get("theta_bounds"),
        fallback=DEFAULT_QUIJOTE_BSQ5_BOUNDS,
    )
    return DataSourceSpec(
        name="quijote",
        parameter_space=str(quijote.get("parameter_space", QUIJOTE_BSQ5_PARAMETER_SPACE)),
        theta_names=tuple(QUIJOTE_BSQ_THETA_NAMES),
        theta_bounds=bounds,
        cache_root=str(quijote.get("runtime_cache_root", "artifacts/quijote/runtime_cache")),
        cache_prefix=str(quijote.get("cache_prefix", "quijote_bsq5")),
        spectrum_type=str(quijote.get("spectrum_type", "quijote_cdm")),
        target_transform=active_target_transform,
        has_linear_anchor=True,
        provider_kind="quijote_gp",
        metadata=dict(quijote),
    )


def active_theta_bounds(config: ValidationRuntimeConfig) -> np.ndarray:
    return resolve_data_source(config).theta_bounds


def active_theta_names(config: ValidationRuntimeConfig) -> tuple[str, ...]:
    return resolve_data_source(config).theta_names


def active_theta_bounds_mapping(config: ValidationRuntimeConfig) -> dict[str, tuple[float, float]]:
    spec = resolve_data_source(config)
    return {
        name: (float(bounds[0]), float(bounds[1]))
        for name, bounds in zip(spec.theta_names, spec.theta_bounds, strict=True)
    }


def resolve_data_provider(config: ValidationRuntimeConfig, provider: Any | None = None) -> Any:
    if provider is not None:
        return provider
    spec = resolve_data_source(config)
    if spec.name == "quijote":
        surrogate_path = str(spec.metadata.get("surrogate_path", "")).strip()
        if not surrogate_path:
            raise ValueError("extensions.quijote.surrogate_path is required for Quijote runs.")
        path = Path(surrogate_path)
        if not path.is_absolute():
            path = config.resolve_path(path)
        linear_provider_name = str(
            spec.metadata.get("anchor_provider")
            or spec.metadata.get("linear_anchor_provider", "camb_sigma8_calibrated")
        ).strip().lower()
        power_eps = float(spec.metadata.get("power_eps", config.gp.power_eps))
        if linear_provider_name in {
            "camb_hmcode2020_sigma8_calibrated",
            "hmcode2020",
            "camb_hmcode2020",
            "quijote_camb_hmcode2020",
        }:
            linear_power_provider = QuijoteCAMBHMCODE2020AnchorProvider(
                reference_as=float(
                    spec.metadata.get(
                        "hmcode2020_anchor_reference_as",
                        spec.metadata.get("linear_anchor_reference_as", 2.1e-9),
                    )
                ),
                halofit_version=str(spec.metadata.get("hmcode2020_anchor_halofit_version", "mead2020")),
                fixed_w0=float(spec.metadata.get("hmcode2020_anchor_fixed_w0", -1.0)),
                fixed_wa=float(spec.metadata.get("hmcode2020_anchor_fixed_wa", 0.0)),
                fixed_mnu=float(spec.metadata.get("hmcode2020_anchor_fixed_mnu", 0.0)),
                power_eps=power_eps,
            )
        elif linear_provider_name in {
            "camb_cdm_hmcode2020_sigma8_calibrated",
            "cdm_hmcode2020",
            "camb_cdm_hmcode2020",
            "quijote_camb_cdm_hmcode2020",
        }:
            linear_power_provider = QuijoteCAMBCDMMHMCODE2020AnchorProvider(
                reference_as=float(
                    spec.metadata.get(
                        "hmcode2020_anchor_reference_as",
                        spec.metadata.get("linear_anchor_reference_as", 2.1e-9),
                    )
                ),
                halofit_version=str(spec.metadata.get("hmcode2020_anchor_halofit_version", "mead2020")),
                fixed_w0=float(spec.metadata.get("hmcode2020_anchor_fixed_w0", -1.0)),
                fixed_wa=float(spec.metadata.get("hmcode2020_anchor_fixed_wa", 0.0)),
                fixed_mnu=float(spec.metadata.get("hmcode2020_anchor_fixed_mnu", 0.0)),
                power_eps=power_eps,
            )
        elif linear_provider_name in {
            "quijote_official_linear_svgp_generator",
            "quijote_official_linear_generator",
            "quijote_linear_generator",
        }:
            linear_generator_path = str(spec.metadata.get("linear_generator_path", "")).strip()
            if not linear_generator_path:
                raise ValueError(
                    "extensions.quijote.linear_generator_path is required when "
                    "linear_anchor_provider is quijote_official_linear_svgp_generator."
                )
            resolved_linear_generator = Path(linear_generator_path)
            if not resolved_linear_generator.is_absolute():
                resolved_linear_generator = config.resolve_path(resolved_linear_generator)
            linear_power_provider = QuijoteLinearGeneratorProvider(
                generator_path=resolved_linear_generator,
                power_eps=power_eps,
                device=str(spec.metadata.get("linear_generator_device", "cpu")),
            )
        elif linear_provider_name in {
            "quijote_official_discrete_table",
            "quijote_official_linear_table",
        }:
            raw_root = config.resolve_path(str(spec.metadata.get("raw_root", "")))
            params_file = config.resolve_path(str(spec.metadata.get("params_file", "")))
            linear_power_provider = QuijoteOfficialLinearAnchorProvider(
                raw_root=raw_root,
                params_file=params_file,
                file_name=str(spec.metadata.get("linear_power_file_name", "Pk_mm_z=0.000.txt")),
                normfac_file_name=str(spec.metadata.get("linear_normfac_file_name", "Normfac.txt")),
                apply_normfac=bool(spec.metadata.get("linear_apply_normfac", True)),
                power_eps=power_eps,
            )
        else:
            linear_power_provider = QuijoteCAMBLinearAnchorProvider(
                reference_as=float(spec.metadata.get("linear_anchor_reference_as", 2.1e-9)),
                power_eps=power_eps,
            )
        return QuijoteGPDataProvider(
            surrogate_path=path,
            linear_power_provider=linear_power_provider,
            linear_power_reference_as=float(spec.metadata.get("linear_anchor_reference_as", 2.1e-9)),
            power_eps=power_eps,
            surrogate_device=str(
                spec.metadata.get(
                    "truth_generator_device",
                    spec.metadata.get("surrogate_device", "cpu"),
                )
            ),
        )
    from z2quijote.runtime_core.camb_data_provider import CAMBDataProvider

    return CAMBDataProvider(config=config)


__all__ = [
    "DataSourceSpec",
    "active_theta_bounds",
    "active_theta_bounds_mapping",
    "active_theta_names",
    "configured_data_source_name",
    "DEFAULT_QUIJOTE_RUNTIME_SETTINGS",
    "resolve_data_provider",
    "resolve_data_source",
    "set_configured_data_source",
]
