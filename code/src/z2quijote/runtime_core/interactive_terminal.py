"""Interactive runtime configuration helpers for active-learning runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from z2quijote.runtime_core.cache_manager import (
    SpectrumBankInspection,
    inspect_comparison_training_bank,
    inspect_initial_training_bank,
    inspect_validation_bank,
)
from z2quijote.runtime_core.config import (
    ValidationRuntimeConfig,
    config_to_dict,
    load_config,
)
from z2quijote.runtime_core.data_source import resolve_data_source, set_configured_data_source


@dataclass(slots=True)
class SamplingPlanSummary:
    initial_points: int
    batch_size: int
    iterations: int
    total_budget: int


@dataclass(slots=True)
class RepresentationSummary:
    transform_family: str
    anchor_mode: str
    target_transform: str
    pca_scheme: str
    global_pca_components: int
    band_pca_components: tuple[int, ...]


@dataclass(slots=True)
class DynamicPreprocessingSummary:
    enabled: bool
    error_source: str
    band_weight_update_interval: int
    band_component_update_interval: int
    grid_update_interval: int
    weight_gamma: float
    weight_rho: float
    weight_min: float
    weight_max: float
    allocation_lambda: float
    min_band_components: int
    max_component_delta_per_update: int


def _resolve_top_level_field(
    config: ValidationRuntimeConfig,
    dotted_key: str,
) -> tuple[Any, str]:
    sections = {
        "sampling": config.sampling,
        "gp": config.gp,
        "gp_baseline": config.gp_baseline,
        "representation": config.representation,
        "dynamic_preprocessing": config.dynamic_preprocessing,
        "m3": config.m3,
        "camb": config.camb,
    }
    if "." not in dotted_key:
        if not hasattr(config, dotted_key):
            raise ValueError(f"Unknown runtime override key: {dotted_key}")
        return config, dotted_key

    section_name, field_name = dotted_key.split(".", 1)
    if section_name not in sections:
        raise ValueError(f"Unknown runtime override section: {section_name}")
    section = sections[section_name]
    if not hasattr(section, field_name):
        raise ValueError(f"Unknown runtime override key: {dotted_key}")
    return section, field_name


def load_base_runtime_config(
    *,
    config_path: str | Path,
    project_root: str | Path | None,
) -> ValidationRuntimeConfig:
    return load_config(config_path, project_root=project_root)


def summarize_sampling_plan(config: ValidationRuntimeConfig) -> SamplingPlanSummary:
    return SamplingPlanSummary(
        initial_points=int(config.sampling.initial_sobol_points),
        batch_size=int(config.sampling.batch_size),
        iterations=int(config.sampling.iterations),
        total_budget=int(config.sampling.total_budget),
    )


def summarize_gp_hyperparameters(config: ValidationRuntimeConfig) -> dict[str, Any]:
    gp = config.gp
    return {
        "pca_components": int(gp.pca_components),
        "alpha": float(gp.alpha),
        "normalize_y": bool(gp.normalize_y),
        "n_restarts_optimizer": int(gp.n_restarts_optimizer),
        "constant_value": float(gp.constant_value),
        "constant_value_bounds_low": float(gp.constant_value_bounds_low),
        "constant_value_bounds_high": float(gp.constant_value_bounds_high),
        "length_scale_initial": float(gp.length_scale_initial),
        "length_scale_bounds_low": float(gp.length_scale_bounds_low),
        "length_scale_bounds_high": float(gp.length_scale_bounds_high),
        "power_eps": float(gp.power_eps),
    }


def summarize_baseline_gp_hyperparameters(config: ValidationRuntimeConfig) -> dict[str, Any]:
    gp = config.gp_baseline
    return {
        "train_points": int(gp.train_points),
        "pca_components": int(gp.pca_components),
        "gp_alpha": float(gp.gp_alpha),
        "normalize_y": bool(gp.normalize_y),
        "gp_n_restarts_optimizer": int(gp.gp_n_restarts_optimizer),
        "constant_value": float(gp.constant_value),
        "constant_value_bounds_low": float(gp.constant_value_bounds_low),
        "constant_value_bounds_high": float(gp.constant_value_bounds_high),
        "length_scale_initial": float(gp.length_scale_initial),
        "length_scale_bounds_low": float(gp.length_scale_bounds_low),
        "length_scale_bounds_high": float(gp.length_scale_bounds_high),
        "power_eps": float(gp.power_eps),
    }


def summarize_representation_settings(config: ValidationRuntimeConfig) -> RepresentationSummary:
    transform_family = str(config.representation.transform_family)
    anchor_mode = str(config.representation.anchor_mode)
    if transform_family == "ratio":
        target_transform = f"ratio_to_{anchor_mode}"
    else:
        target_transform = f"log_hi_minus_log_{anchor_mode}_anchor"
    return RepresentationSummary(
        transform_family=transform_family,
        anchor_mode=anchor_mode,
        target_transform=target_transform,
        pca_scheme=str(config.representation.pca_scheme),
        global_pca_components=int(config.representation.global_pca_components),
        band_pca_components=tuple(int(value) for value in config.representation.band_pca_components),
    )


def summarize_dynamic_preprocessing_settings(
    config: ValidationRuntimeConfig,
) -> DynamicPreprocessingSummary:
    dynamic = config.dynamic_preprocessing
    return DynamicPreprocessingSummary(
        enabled=bool(dynamic.enabled),
        error_source=str(dynamic.error_source),
        band_weight_update_interval=int(dynamic.band_weight_update_interval),
        band_component_update_interval=int(dynamic.band_component_update_interval),
        grid_update_interval=int(dynamic.grid_update_interval),
        weight_gamma=float(dynamic.weight_gamma),
        weight_rho=float(dynamic.weight_rho),
        weight_min=float(dynamic.weight_min),
        weight_max=float(dynamic.weight_max),
        allocation_lambda=float(dynamic.allocation_lambda),
        min_band_components=int(dynamic.min_band_components),
        max_component_delta_per_update=int(dynamic.max_component_delta_per_update),
    )


def apply_sampling_overrides(
    config: ValidationRuntimeConfig,
    *,
    initial_points: int | None = None,
    batch_size: int | None = None,
    iterations: int | None = None,
) -> ValidationRuntimeConfig:
    if initial_points is not None:
        config.sampling.initial_sobol_points = int(initial_points)
    if batch_size is not None:
        config.sampling.batch_size = int(batch_size)
    if iterations is not None:
        config.sampling.iterations = int(iterations)
    config.sampling.__post_init__()
    return config


def apply_gp_overrides(
    config: ValidationRuntimeConfig,
    *,
    pca_components: int | None = None,
    alpha: float | None = None,
    normalize_y: bool | None = None,
    n_restarts_optimizer: int | None = None,
    constant_value: float | None = None,
    constant_value_bounds_low: float | None = None,
    constant_value_bounds_high: float | None = None,
    length_scale_initial: float | None = None,
    length_scale_bounds_low: float | None = None,
    length_scale_bounds_high: float | None = None,
    power_eps: float | None = None,
) -> ValidationRuntimeConfig:
    gp = config.gp
    if pca_components is not None:
        gp.pca_components = int(pca_components)
    if alpha is not None:
        gp.alpha = float(alpha)
    if normalize_y is not None:
        gp.normalize_y = bool(normalize_y)
    if n_restarts_optimizer is not None:
        gp.n_restarts_optimizer = int(n_restarts_optimizer)
    if constant_value is not None:
        gp.constant_value = float(constant_value)
    if constant_value_bounds_low is not None:
        gp.constant_value_bounds_low = float(constant_value_bounds_low)
    if constant_value_bounds_high is not None:
        gp.constant_value_bounds_high = float(constant_value_bounds_high)
    if length_scale_initial is not None:
        gp.length_scale_initial = float(length_scale_initial)
    if length_scale_bounds_low is not None:
        gp.length_scale_bounds_low = float(length_scale_bounds_low)
    if length_scale_bounds_high is not None:
        gp.length_scale_bounds_high = float(length_scale_bounds_high)
    if power_eps is not None:
        gp.power_eps = float(power_eps)
    gp.__post_init__()
    return config


def apply_representation_overrides(
    config: ValidationRuntimeConfig,
    *,
    transform_family: str | None = None,
    anchor_mode: str | None = None,
    pca_scheme: str | None = None,
    global_pca_components: int | None = None,
    band_pca_components: tuple[int, ...] | list[int] | None = None,
) -> ValidationRuntimeConfig:
    representation = config.representation
    if transform_family is not None:
        representation.transform_family = str(transform_family).strip().lower()
    if anchor_mode is not None:
        representation.anchor_mode = str(anchor_mode).strip().lower()
    if pca_scheme is not None:
        representation.pca_scheme = str(pca_scheme).strip().lower()
    if global_pca_components is not None:
        representation.global_pca_components = int(global_pca_components)
    if band_pca_components is not None:
        representation.band_pca_components = tuple(int(value) for value in band_pca_components)
    representation.__post_init__()
    return config


def apply_dynamic_preprocessing_overrides(
    config: ValidationRuntimeConfig,
    *,
    enabled: bool | None = None,
    error_source: str | None = None,
    band_weight_update_interval: int | None = None,
    band_component_update_interval: int | None = None,
    grid_update_interval: int | None = None,
    weight_gamma: float | None = None,
    weight_rho: float | None = None,
    weight_min: float | None = None,
    weight_max: float | None = None,
    allocation_lambda: float | None = None,
    min_band_components: int | None = None,
    max_component_delta_per_update: int | None = None,
) -> ValidationRuntimeConfig:
    dynamic = config.dynamic_preprocessing
    if enabled is not None:
        dynamic.enabled = bool(enabled)
    if error_source is not None:
        dynamic.error_source = str(error_source).strip().lower()
    if band_weight_update_interval is not None:
        dynamic.band_weight_update_interval = int(band_weight_update_interval)
    if band_component_update_interval is not None:
        dynamic.band_component_update_interval = int(band_component_update_interval)
    if grid_update_interval is not None:
        dynamic.grid_update_interval = int(grid_update_interval)
    if weight_gamma is not None:
        dynamic.weight_gamma = float(weight_gamma)
    if weight_rho is not None:
        dynamic.weight_rho = float(weight_rho)
    if weight_min is not None:
        dynamic.weight_min = float(weight_min)
    if weight_max is not None:
        dynamic.weight_max = float(weight_max)
    if allocation_lambda is not None:
        dynamic.allocation_lambda = float(allocation_lambda)
    if min_band_components is not None:
        dynamic.min_band_components = int(min_band_components)
    if max_component_delta_per_update is not None:
        dynamic.max_component_delta_per_update = int(max_component_delta_per_update)
    dynamic.__post_init__()
    return config


def apply_manual_overrides(
    config: ValidationRuntimeConfig,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ValidationRuntimeConfig:
    if not overrides:
        return config

    touched: dict[int, Any] = {}
    top_level_touched = False
    for dotted_key, value in overrides.items():
        target, field_name = _resolve_top_level_field(config, str(dotted_key).strip())
        setattr(target, field_name, value)
        if target is config:
            top_level_touched = True
        else:
            touched[id(target)] = target

    for section in touched.values():
        post_init = getattr(section, "__post_init__", None)
        if callable(post_init):
            post_init()
    if top_level_touched:
        config.__post_init__()
    return config


def apply_runtime_overrides(
    config: ValidationRuntimeConfig,
    *,
    data_source: str | None = None,
    spectrum_type: str | None = None,
    initial_points: int | None = None,
    batch_size: int | None = None,
    iterations: int | None = None,
    gp_overrides: dict[str, Any] | None = None,
    representation_overrides: dict[str, Any] | None = None,
    dynamic_preprocessing_overrides: dict[str, Any] | None = None,
    manual_overrides: Mapping[str, Any] | None = None,
) -> ValidationRuntimeConfig:
    if data_source is not None:
        set_configured_data_source(config, data_source)
    apply_sampling_overrides(
        config,
        initial_points=initial_points,
        batch_size=batch_size,
        iterations=iterations,
    )
    if gp_overrides:
        apply_gp_overrides(config, **gp_overrides)
    if representation_overrides:
        apply_representation_overrides(config, **representation_overrides)
    if dynamic_preprocessing_overrides:
        apply_dynamic_preprocessing_overrides(config, **dynamic_preprocessing_overrides)
    if spectrum_type is not None and resolve_data_source(config).name == "camb":
        config.camb.spectrum_type = str(spectrum_type).strip().lower()
        config.camb.__post_init__()
    if manual_overrides:
        apply_manual_overrides(config, overrides=manual_overrides)
    return config


def apply_data_source_defaults(config: ValidationRuntimeConfig) -> ValidationRuntimeConfig:
    return config


def build_cache_preview(config: ValidationRuntimeConfig) -> list[SpectrumBankInspection]:
    return [
        inspect_initial_training_bank(config),
        inspect_comparison_training_bank(
            config,
            train_points=int(config.sampling.total_budget),
        ),
        inspect_validation_bank(config),
    ]


def write_runtime_config_snapshot(
    config: ValidationRuntimeConfig,
    *,
    output_dir: str | Path,
    prefix: str = "interactive_runtime",
) -> Path:
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = directory / f"{prefix}_{timestamp}.yaml"
    output_path.write_text(
        yaml.safe_dump(config_to_dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path
