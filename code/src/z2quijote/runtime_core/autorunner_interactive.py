"""Shared interactive Autorunner session helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import yaml

from z2quijote.runtime_core.interactive_terminal import (
    apply_data_source_defaults,
    apply_runtime_overrides,
    build_cache_preview,
    load_base_runtime_config,
    summarize_baseline_gp_hyperparameters,
    summarize_dynamic_preprocessing_settings,
    summarize_gp_hyperparameters,
    summarize_representation_settings,
    summarize_sampling_plan,
    write_runtime_config_snapshot,
)
from z2quijote.runtime_core.data_source import configured_data_source_name, resolve_data_source


@dataclass(slots=True)
class InteractiveAutorunnerOptions:
    base_config_path: Path | None = None
    module3_mode: str = "online"
    data_source: str | None = None
    spectrum_type: str | None = None
    skip_plots: bool = False
    force_rebuild_cache: bool = False
    dry_run: bool = False


ChoiceOption = tuple[str, str, str]


def _prompt_text(message: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{message}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return str(default)


def _prompt_int(message: str, default: int) -> int:
    while True:
        raw = _prompt_text(message, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.", flush=True)


def _prompt_float(message: str, default: float) -> float:
    while True:
        raw = _prompt_text(message, f"{default}")
        try:
            return float(raw)
        except ValueError:
            print("Please enter a real number.", flush=True)


def _prompt_bool(message: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{message} [{hint}]: ").strip().lower()
        if not raw:
            return bool(default)
        if raw in {"y", "yes", "1", "true"}:
            return True
        if raw in {"n", "no", "0", "false"}:
            return False
        print("Please enter y or n.", flush=True)


def _prompt_choice(
    message: str,
    options: list[ChoiceOption],
    *,
    default_key: str,
) -> str:
    lookup = {key: (label, description) for key, label, description in options}
    if default_key not in lookup:
        raise ValueError(f"Unknown default choice {default_key!r} for {message}.")
    _print_section(message)
    for index, (key, label, description) in enumerate(options, start=1):
        default_tag = " (default)" if key == default_key else ""
        print(f"  {index}. {label} -> {key}{default_tag}", flush=True)
        print(f"     {description}", flush=True)
    while True:
        raw = input("Choose option number or key [Enter keeps default]: ").strip()
        if not raw:
            return default_key
        if raw in lookup:
            return raw
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        print("Please enter one of the listed option numbers or keys.", flush=True)


def _prompt_int_vector(
    message: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    default_text = ",".join(str(value) for value in default)
    while True:
        raw = _prompt_text(message, default_text)
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        if len(parts) != len(default):
            print(f"Please enter exactly {len(default)} integers separated by commas.", flush=True)
            continue
        try:
            return tuple(int(item) for item in parts)
        except ValueError:
            print(f"Please enter exactly {len(default)} integers separated by commas.", flush=True)


def _print_section(title: str) -> None:
    print("", flush=True)
    print(f"== {title} ==", flush=True)


def _print_banner(launcher_name: str, base_config_path: Path) -> None:
    line = "=" * 68
    print(line, flush=True)
    print("Cosmology Emulator Control Panel", flush=True)
    print(line, flush=True)
    print(f"Launcher: {launcher_name}", flush=True)
    print(f"Base config: {base_config_path}", flush=True)
    print(
        "This guided setup controls the active-learning run, the fixed-budget comparison, "
        "and the shared target representation.",
        flush=True,
    )


def _print_sampling_summary(initial_points: int, batch_size: int, iterations: int) -> None:
    total_budget = int(initial_points + batch_size * iterations)
    _print_section("Sampling Plan")
    print(f"  m = {initial_points}", flush=True)
    print(f"  n = {batch_size}", flush=True)
    print(f"  t = {iterations}", flush=True)
    print(f"  total = m + n*t = {total_budget}", flush=True)
    print("  meaning: initial points + points per round x number of rounds", flush=True)


def _print_kv_summary(title: str, payload: dict[str, Any]) -> None:
    _print_section(title)
    for key, value in payload.items():
        print(f"  {key} = {value}", flush=True)


def _print_representation_summary(summary: Any) -> None:
    _print_section("Representation")
    print(f"  transform_family = {summary.transform_family}", flush=True)
    print(f"  anchor_mode = {summary.anchor_mode}", flush=True)
    print(f"  target_transform = {summary.target_transform}", flush=True)
    print(f"  pca_scheme = {summary.pca_scheme}", flush=True)
    print(f"  global_pca_components = {summary.global_pca_components}", flush=True)
    print(f"  band_pca_components = {summary.band_pca_components}", flush=True)


def _print_dynamic_summary(summary: Any) -> None:
    _print_section("Dynamic Preprocessing")
    print(f"  enabled = {summary.enabled}", flush=True)
    print(f"  error_source = {summary.error_source}", flush=True)
    print(
        "  cadence = "
        f"weights/{summary.band_weight_update_interval}, "
        f"allocation/{summary.band_component_update_interval}, "
        f"grid/{summary.grid_update_interval}",
        flush=True,
    )
    print(f"  weight_gamma = {summary.weight_gamma}", flush=True)
    print(f"  weight_rho = {summary.weight_rho}", flush=True)
    print(f"  weight_range = [{summary.weight_min}, {summary.weight_max}]", flush=True)
    print(f"  allocation_lambda = {summary.allocation_lambda}", flush=True)
    print(f"  min_band_components = {summary.min_band_components}", flush=True)
    print(f"  max_component_delta_per_update = {summary.max_component_delta_per_update}", flush=True)


def _collect_data_source(config: Any, option_data_source: str | None) -> str:
    if option_data_source is not None:
        return str(option_data_source).strip().lower()
    default = configured_data_source_name(config)
    return _prompt_choice(
        "Choose the data-generation source",
        [
            (
                "camb",
                "CAMB 8D",
                "Use the existing CAMB path with the original 8-parameter theta space.",
            ),
            (
                "quijote",
                "Quijote BSQ 5D",
                "Use the isolated Quijote GP surrogate with Omega_m, Omega_b, h, n_s, sigma_8.",
            ),
        ],
        default_key=default,
    )


def _print_data_source_summary(config: Any) -> None:
    spec = resolve_data_source(config)
    _print_section("Data Source")
    print(f"  data_source = {spec.name}", flush=True)
    print(f"  parameter_space = {spec.parameter_space}", flush=True)
    print(f"  theta_dim = {spec.theta_dim}", flush=True)
    print(f"  theta_names = {', '.join(spec.theta_names)}", flush=True)
    print(f"  cache_root = {spec.cache_root}", flush=True)
    if spec.name == "quijote":
        print(f"  bank_path = {spec.metadata.get('bank_path', '')}", flush=True)
        print(f"  surrogate_path = {spec.metadata.get('surrogate_path', '')}", flush=True)
        print("  runtime k-grid = source Quijote k_bins when the bank is available", flush=True)


def _collect_phase_preset(
    current_representation: Any,
    current_dynamic: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    phase_key = _prompt_choice(
        "Choose the staged workflow preset",
        [
            (
                "keep_current",
                "Keep current",
                "Use the config file as-is and only apply the manual edits below.",
            ),
            (
                "phase1_target_mode",
                "Phase 1",
                "Target-mode study only: ratio/logdiff with the historical global PCA path.",
            ),
            (
                "phase2_grouped_pca",
                "Phase 2",
                "Global + band residual PCA with grouped M3 acquisition.",
            ),
            (
                "phase3_dynamic",
                "Phase 3",
                "Phase 2 plus block-updated dynamic preprocessing every 8/16/32 rounds.",
            ),
        ],
        default_key="keep_current",
    )
    representation = replace(current_representation)
    dynamic = replace(current_dynamic)
    manual_overrides: dict[str, Any] = {}
    if phase_key == "phase1_target_mode":
        representation = replace(
            representation,
            pca_scheme="global_pca",
            global_pca_components=max(0, int(current_representation.global_pca_components)),
        )
        dynamic = replace(dynamic, enabled=False)
        manual_overrides["m3.objective_mode"] = "mid_high_weighted_sum"
    elif phase_key == "phase2_grouped_pca":
        representation = replace(
            representation,
            pca_scheme="global_plus_band_residual_pca",
            global_pca_components=6,
            band_pca_components=(2, 5, 4, 3),
        )
        dynamic = replace(dynamic, enabled=False)
        manual_overrides["m3.objective_mode"] = "representation_grouped_posterior_variance"
    elif phase_key == "phase3_dynamic":
        representation = replace(
            representation,
            pca_scheme="global_plus_band_residual_pca",
            global_pca_components=6,
            band_pca_components=(2, 5, 4, 3),
        )
        dynamic = replace(
            dynamic,
            enabled=True,
            band_weight_update_interval=8,
            band_component_update_interval=16,
            grid_update_interval=32,
            weight_gamma=0.7,
            weight_rho=0.25,
            weight_min=1.0e-6,
            weight_max=1.8,
            band_weight_balance_mode="core_posterior_variance",
            core_band_indices=(1, 2, 3),
            core_error_good=0.0035,
            core_error_bad=0.006,
            core_gate_floor=0.15,
            core_gate_ceiling=0.85,
            core_priority=(0.30, 1.35, 1.30, 1.05),
            release_priority=(0.45, 1.25, 1.20, 1.10),
            error_signal_eta=0.25,
            posterior_variance_probe_size=512,
            posterior_variance_seed_offset=9173,
            posterior_variance_gamma=0.5,
            posterior_variance_eta=0.35,
            allocation_lambda=0.35,
            min_band_components=2,
            max_component_delta_per_update=1,
        )
        manual_overrides["m3.objective_mode"] = "representation_grouped_posterior_variance"
    return representation, dynamic, manual_overrides


def _collect_gp_overrides(current: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not _prompt_bool("Do you want to override primary GP hyperparameters?", False):
        return overrides

    overrides["pca_components"] = _prompt_int("gp.pca_components", int(current["pca_components"]))
    overrides["alpha"] = _prompt_float("gp.alpha", float(current["alpha"]))
    overrides["normalize_y"] = _prompt_bool("gp.normalize_y", bool(current["normalize_y"]))
    overrides["n_restarts_optimizer"] = _prompt_int(
        "gp.n_restarts_optimizer",
        int(current["n_restarts_optimizer"]),
    )
    overrides["constant_value"] = _prompt_float("gp.constant_value", float(current["constant_value"]))
    overrides["constant_value_bounds_low"] = _prompt_float(
        "gp.constant_value_bounds_low",
        float(current["constant_value_bounds_low"]),
    )
    overrides["constant_value_bounds_high"] = _prompt_float(
        "gp.constant_value_bounds_high",
        float(current["constant_value_bounds_high"]),
    )
    overrides["length_scale_initial"] = _prompt_float(
        "gp.length_scale_initial",
        float(current["length_scale_initial"]),
    )
    overrides["length_scale_bounds_low"] = _prompt_float(
        "gp.length_scale_bounds_low",
        float(current["length_scale_bounds_low"]),
    )
    overrides["length_scale_bounds_high"] = _prompt_float(
        "gp.length_scale_bounds_high",
        float(current["length_scale_bounds_high"]),
    )
    overrides["power_eps"] = _prompt_float("gp.power_eps", float(current["power_eps"]))
    return overrides


def _collect_baseline_gp_overrides(current: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not _prompt_bool("Do you want to override baseline-only GP hyperparameters?", False):
        return overrides

    overrides["gp_baseline.train_points"] = _prompt_int(
        "gp_baseline.train_points",
        int(current["train_points"]),
    )
    overrides["gp_baseline.pca_components"] = _prompt_int(
        "gp_baseline.pca_components",
        int(current["pca_components"]),
    )
    overrides["gp_baseline.gp_alpha"] = _prompt_float(
        "gp_baseline.gp_alpha",
        float(current["gp_alpha"]),
    )
    overrides["gp_baseline.normalize_y"] = _prompt_bool(
        "gp_baseline.normalize_y",
        bool(current["normalize_y"]),
    )
    overrides["gp_baseline.gp_n_restarts_optimizer"] = _prompt_int(
        "gp_baseline.gp_n_restarts_optimizer",
        int(current["gp_n_restarts_optimizer"]),
    )
    overrides["gp_baseline.constant_value"] = _prompt_float(
        "gp_baseline.constant_value",
        float(current["constant_value"]),
    )
    overrides["gp_baseline.constant_value_bounds_low"] = _prompt_float(
        "gp_baseline.constant_value_bounds_low",
        float(current["constant_value_bounds_low"]),
    )
    overrides["gp_baseline.constant_value_bounds_high"] = _prompt_float(
        "gp_baseline.constant_value_bounds_high",
        float(current["constant_value_bounds_high"]),
    )
    overrides["gp_baseline.length_scale_initial"] = _prompt_float(
        "gp_baseline.length_scale_initial",
        float(current["length_scale_initial"]),
    )
    overrides["gp_baseline.length_scale_bounds_low"] = _prompt_float(
        "gp_baseline.length_scale_bounds_low",
        float(current["length_scale_bounds_low"]),
    )
    overrides["gp_baseline.length_scale_bounds_high"] = _prompt_float(
        "gp_baseline.length_scale_bounds_high",
        float(current["length_scale_bounds_high"]),
    )
    overrides["gp_baseline.power_eps"] = _prompt_float(
        "gp_baseline.power_eps",
        float(current["power_eps"]),
    )
    return overrides


def _collect_representation_overrides(current: Any) -> dict[str, Any]:
    transform_family = _prompt_choice(
        "Choose the target curve representation",
        [
            (
                "ratio",
                "P1/P2 ratio representation",
                "Target = P_nonlin / P_anchor - 1. This is the most stable default path.",
            ),
            (
                "logdiff",
                "log(P1) - log(P2) difference representation",
                "Target = log(P_nonlin) - log(P_anchor). Useful when multiplicative differences matter more.",
            ),
        ],
        default_key=str(current.transform_family),
    )
    pca_scheme = _prompt_choice(
        "Choose the PCA scheme",
        [
            (
                "global_pca",
                "Single global PCA",
                "One PCA over all k bins. Simplest and closest to the historical pipeline.",
            ),
            (
                "bandwise_pca",
                "Bandwise PCA",
                "Separate PCA blocks by k band. Better when different wave-number regions behave differently.",
            ),
            (
                "global_plus_band_residual_pca",
                "Global PCA plus band residual PCA",
                "Global trend first, then per-band residual correction. Usually the most expressive option.",
            ),
        ],
        default_key=str(current.pca_scheme),
    )
    global_components = int(current.global_pca_components)
    band_components = tuple(int(value) for value in current.band_pca_components)
    if _prompt_bool("Do you want to override PCA component allocation?", False):
        global_components = _prompt_int(
            "representation.global_pca_components",
            global_components,
        )
        band_components = _prompt_int_vector(
            "representation.band_pca_components (low,mid,high)",
            band_components,
        )
    return {
        "transform_family": transform_family,
        "pca_scheme": pca_scheme,
        "global_pca_components": global_components,
        "band_pca_components": band_components,
    }


def _collect_dynamic_preprocessing_overrides(current: Any) -> dict[str, Any]:
    enabled = _prompt_bool(
        "Enable dynamic preprocessing block updates?",
        bool(current.enabled),
    )
    overrides: dict[str, Any] = {"enabled": bool(enabled)}
    if not enabled:
        return overrides
    if not _prompt_bool("Do you want to override dynamic preprocessing hyperparameters?", False):
        return overrides

    error_source = _prompt_choice(
        "Choose the dynamic error signal",
        [
            (
                "validation_relative_error",
                "Validation relative error",
                "Use cached validation-set relative error by band as the primary signal.",
            ),
            (
                "pca_sensitivity_proxy",
                "PCA sensitivity proxy",
                "Use current PCA band sensitivity when validation-driven updates are unavailable.",
            ),
        ],
        default_key=str(current.error_source),
    )
    overrides.update(
        {
            "error_source": error_source,
            "band_weight_update_interval": _prompt_int(
                "dynamic_preprocessing.band_weight_update_interval",
                int(current.band_weight_update_interval),
            ),
            "band_component_update_interval": _prompt_int(
                "dynamic_preprocessing.band_component_update_interval",
                int(current.band_component_update_interval),
            ),
            "grid_update_interval": _prompt_int(
                "dynamic_preprocessing.grid_update_interval",
                int(current.grid_update_interval),
            ),
            "weight_gamma": _prompt_float(
                "dynamic_preprocessing.weight_gamma",
                float(current.weight_gamma),
            ),
            "weight_rho": _prompt_float(
                "dynamic_preprocessing.weight_rho",
                float(current.weight_rho),
            ),
            "weight_min": _prompt_float(
                "dynamic_preprocessing.weight_min",
                float(current.weight_min),
            ),
            "weight_max": _prompt_float(
                "dynamic_preprocessing.weight_max",
                float(current.weight_max),
            ),
            "allocation_lambda": _prompt_float(
                "dynamic_preprocessing.allocation_lambda",
                float(current.allocation_lambda),
            ),
            "min_band_components": _prompt_int(
                "dynamic_preprocessing.min_band_components",
                int(current.min_band_components),
            ),
            "max_component_delta_per_update": _prompt_int(
                "dynamic_preprocessing.max_component_delta_per_update",
                int(current.max_component_delta_per_update),
            ),
        }
    )
    return overrides


def _parse_manual_override_text(raw: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    normalized = str(raw).replace("\r\n", "\n").strip()
    if not normalized:
        return overrides
    parts = [
        item.strip()
        for item in normalized.replace("\n", ";").split(";")
        if item.strip()
    ]
    for item in parts:
        key, separator, value_text = item.partition("=")
        if not separator:
            raise ValueError(
                f"Manual override {item!r} is invalid. Use dotted keys like gp.alpha=1e-8."
            )
        dotted_key = key.strip()
        if not dotted_key:
            raise ValueError(f"Manual override {item!r} is missing a key.")
        overrides[dotted_key] = yaml.safe_load(value_text.strip())
    return overrides


def _collect_manual_overrides() -> dict[str, Any]:
    _print_section("Advanced Overrides")
    print(
        "Optional dotted-key overrides. Example:",
        flush=True,
    )
    print(
        "  gp.alpha=1e-8; gp_baseline.gp_alpha=1e-8; m3.chunk_size=16384",
        flush=True,
    )
    raw = _prompt_text("Manual overrides", "")
    if not raw.strip():
        return {}
    while True:
        try:
            return _parse_manual_override_text(raw)
        except ValueError as exc:
            print(str(exc), flush=True)
            raw = _prompt_text("Manual overrides", "")


def _print_cache_preview(config: Any) -> None:
    _print_section("Cache Preview")
    for inspection in build_cache_preview(config):
        status_map = {
            "hit": "reusable",
            "missing": "missing, will build",
            "stale": "present but mismatched, will rebuild",
        }
        status_text = status_map.get(inspection.status, inspection.status)
        print(
            f"  {inspection.name}: {status_text} | points={inspection.expected_point_count} | {inspection.npz_path}",
            flush=True,
        )


def _print_run_review(
    *,
    plan: Any,
    representation_summary: Any,
    dynamic_summary: Any,
    baseline_summary: dict[str, Any],
    runtime_spectrum_type: str,
    data_source_name: str,
) -> None:
    _print_section("Run Review")
    print(
        f"  active-learning budget = {plan.initial_points} + {plan.batch_size} x {plan.iterations} = {plan.total_budget}",
        flush=True,
    )
    print(
        f"  fixed-budget comparison training size = {plan.total_budget} (auto-synced to active-learning total budget)",
        flush=True,
    )
    print(f"  spectrum_type = {runtime_spectrum_type}", flush=True)
    print(f"  data_source = {data_source_name}", flush=True)
    print(f"  target_transform = {representation_summary.target_transform}", flush=True)
    print(f"  pca_scheme = {representation_summary.pca_scheme}", flush=True)
    print(f"  dynamic_preprocessing = {dynamic_summary.enabled}", flush=True)
    print(
        f"  baseline standalone train_points setting = {baseline_summary['train_points']}",
        flush=True,
    )


def _runtime_snapshot_dir(config: Any) -> Path:
    return Path(config.project_root).resolve() / "artifacts" / "runtime_configs"


def run_interactive_session(
    *,
    project_root: Path,
    run_autorunner_fn: Callable[..., Any],
    options: InteractiveAutorunnerOptions,
    launcher_name: str,
) -> int:
    base_config_path = (
        options.base_config_path.resolve()
        if options.base_config_path is not None
        else (project_root / "configs" / "default.yaml").resolve()
    )
    config = load_base_runtime_config(
        config_path=base_config_path,
        project_root=project_root if options.base_config_path is None else None,
    )
    if options.dry_run and options.data_source is None:
        selected_data_source = configured_data_source_name(config)
    else:
        selected_data_source = _collect_data_source(config, options.data_source)
    apply_runtime_overrides(config, data_source=selected_data_source)
    apply_data_source_defaults(config)
    current_plan = summarize_sampling_plan(config)
    current_gp = summarize_gp_hyperparameters(config)
    current_baseline_gp = summarize_baseline_gp_hyperparameters(config)
    current_representation = summarize_representation_settings(config)
    current_dynamic = summarize_dynamic_preprocessing_settings(config)

    _print_banner(launcher_name, base_config_path)
    _print_data_source_summary(config)
    _print_sampling_summary(
        current_plan.initial_points,
        current_plan.batch_size,
        current_plan.iterations,
    )
    _print_kv_summary("Primary GP Hyperparameters", current_gp)
    _print_kv_summary(
        "Baseline GP Hyperparameters",
        {"baseline." + key: value for key, value in current_baseline_gp.items()},
    )
    _print_representation_summary(current_representation)
    _print_dynamic_summary(current_dynamic)

    if options.dry_run:
        runtime_spectrum_type = options.spectrum_type or resolve_data_source(config).spectrum_type
        apply_runtime_overrides(
            config,
            data_source=selected_data_source,
            spectrum_type=runtime_spectrum_type,
        )
        apply_data_source_defaults(config)
        plan = summarize_sampling_plan(config)
        gp_summary = summarize_gp_hyperparameters(config)
        baseline_summary = summarize_baseline_gp_hyperparameters(config)
        representation_summary = summarize_representation_settings(config)
        dynamic_summary = summarize_dynamic_preprocessing_settings(config)
        _print_sampling_summary(plan.initial_points, plan.batch_size, plan.iterations)
        print("  fixed-budget comparison will automatically use the same total budget.", flush=True)
        _print_kv_summary("Primary GP Hyperparameters", gp_summary)
        _print_kv_summary(
            "Baseline GP Hyperparameters",
            {"baseline." + key: value for key, value in baseline_summary.items()},
        )
        _print_representation_summary(representation_summary)
        _print_dynamic_summary(dynamic_summary)
        _print_run_review(
            plan=plan,
            representation_summary=representation_summary,
            dynamic_summary=dynamic_summary,
            baseline_summary=baseline_summary,
            runtime_spectrum_type=str(runtime_spectrum_type),
            data_source_name=selected_data_source,
        )
        _print_cache_preview(config)
        output_config = write_runtime_config_snapshot(
            config,
            output_dir=_runtime_snapshot_dir(config),
        )
        _print_section("Dry Run")
        print(f"  Runtime config written to {output_config}", flush=True)
        print("  Autorunner was not started.", flush=True)
        return 0

    print("", flush=True)
    print(
        "Press Enter to keep defaults. The default sampling plan is shown above.",
        flush=True,
    )
    preset_representation, preset_dynamic, preset_manual_overrides = _collect_phase_preset(
        current_representation,
        current_dynamic,
    )
    initial_points = _prompt_int("Initial sample count m", current_plan.initial_points)
    batch_size = _prompt_int("Points added per iteration n", current_plan.batch_size)
    iterations = _prompt_int("Iteration count t", current_plan.iterations)
    gp_overrides = _collect_gp_overrides(current_gp)
    baseline_manual_overrides = _collect_baseline_gp_overrides(current_baseline_gp)
    representation_overrides = _collect_representation_overrides(
        SimpleNamespace(
            transform_family=preset_representation.transform_family,
            pca_scheme=preset_representation.pca_scheme,
            global_pca_components=preset_representation.global_pca_components,
            band_pca_components=preset_representation.band_pca_components,
        )
    )
    dynamic_preprocessing_overrides = _collect_dynamic_preprocessing_overrides(preset_dynamic)
    manual_overrides = {**preset_manual_overrides, **_collect_manual_overrides()}
    runtime_spectrum_type = options.spectrum_type
    if runtime_spectrum_type is None and selected_data_source == "camb":
        runtime_spectrum_type = _prompt_text("spectrum_type", config.camb.spectrum_type)
    elif runtime_spectrum_type is None:
        runtime_spectrum_type = resolve_data_source(config).spectrum_type

    apply_runtime_overrides(
        config,
        data_source=selected_data_source,
        spectrum_type=runtime_spectrum_type,
        initial_points=initial_points,
        batch_size=batch_size,
        iterations=iterations,
        gp_overrides=gp_overrides,
        representation_overrides=representation_overrides,
        dynamic_preprocessing_overrides=dynamic_preprocessing_overrides,
        manual_overrides={**baseline_manual_overrides, **manual_overrides},
    )
    apply_data_source_defaults(config)
    plan = summarize_sampling_plan(config)
    gp_summary = summarize_gp_hyperparameters(config)
    baseline_summary = summarize_baseline_gp_hyperparameters(config)
    representation_summary = summarize_representation_settings(config)
    dynamic_summary = summarize_dynamic_preprocessing_settings(config)

    _print_sampling_summary(plan.initial_points, plan.batch_size, plan.iterations)
    print("  fixed-budget comparison will automatically use the same total budget.", flush=True)
    _print_kv_summary("Primary GP Hyperparameters", gp_summary)
    _print_kv_summary(
        "Baseline GP Hyperparameters",
        {"baseline." + key: value for key, value in baseline_summary.items()},
    )
    _print_representation_summary(representation_summary)
    _print_dynamic_summary(dynamic_summary)
    _print_run_review(
        plan=plan,
        representation_summary=representation_summary,
        dynamic_summary=dynamic_summary,
        baseline_summary=baseline_summary,
        runtime_spectrum_type=str(runtime_spectrum_type),
        data_source_name=selected_data_source,
    )
    _print_cache_preview(config)

    output_config = write_runtime_config_snapshot(
        config,
        output_dir=_runtime_snapshot_dir(config),
    )
    _print_section("Snapshot")
    print(f"  Runtime config written to {output_config}", flush=True)

    if not _prompt_bool("Start Autorunner with this configuration?", True):
        print("Run cancelled.", flush=True)
        return 0

    artifacts = run_autorunner_fn(
        config_path=output_config,
        project_root=project_root,
        module3_mode=options.module3_mode,
        force_rebuild_cache=bool(options.force_rebuild_cache),
        skip_plots=bool(options.skip_plots),
        spectrum_type=runtime_spectrum_type,
        data_source=selected_data_source,
    )
    _print_section("Run Started")
    print(f"  Run directory: {artifacts.run_dir}", flush=True)
    if artifacts.comparison_report_path is not None:
        print(f"  Comparison report: {artifacts.comparison_report_path}", flush=True)
    for label, path in artifacts.cache_paths.items():
        print(f"  Cache {label}: {path}", flush=True)
    return 0
