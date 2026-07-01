from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .kgrid import KGrid, build_k_grid
from .parameter_space import ParameterSpace


@dataclass(frozen=True, slots=True)
class TargetConfig:
    kind: str
    anchor_mode: str
    power_eps: float


@dataclass(frozen=True, slots=True)
class FastMockBiasConfig:
    enabled: bool
    provider: str
    score_mode: str
    vendor_path: Path | None
    checkbound: bool
    fixed_w: float
    fixed_wa: float
    fixed_mnu: float
    reference_as: float
    bias_weight: float
    bias_band_weights: tuple[float, ...]
    normalization: str
    normalization_probe_count: int
    cache_decimals: int
    truth_backend: str
    truth_dtype: str
    truth_device: str
    truth_chunk_size: int


@dataclass(frozen=True, slots=True)
class TruthGeneratorConfig:
    kind: str
    path: Path
    device: str
    chunk_size: int


@dataclass(frozen=True, slots=True)
class R2SeedConfig:
    mode: str
    path: Path
    theta_key: str
    manifest_path: Path | None


@dataclass(frozen=True, slots=True)
class CurrentActiveConfig:
    enabled: bool
    path: Path | None
    theta_key: str


@dataclass(frozen=True, slots=True)
class RawBankConfig:
    enabled: bool
    path: Path | None
    metadata_path: Path | None
    sample_size: int
    sample_seed: int
    filter_to_parameter_box: bool


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    use_lofi: bool
    runtime_root: Path
    truth_generator: TruthGeneratorConfig
    r2_seed: R2SeedConfig
    current_active: CurrentActiveConfig
    raw_bank: RawBankConfig


@dataclass(frozen=True, slots=True)
class SplitConfig:
    seed_label: str
    probe_size: int
    pool_size: int
    audit_size: int
    audit_source: str
    audit_path: Path | None
    audit_theta_unit_key: str
    audit_theta_raw_key: str
    sobol64_size: int
    sobol128_size: int
    sobol_tail_size: int
    duplicate_decimals: int
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ModelConfig:
    pca_components: int
    gp_alpha: float
    gp_n_restarts_optimizer: int
    normalize_y: bool
    length_scale_initial: float
    length_scale_bounds: tuple[float, float]
    constant_value: float
    constant_value_bounds: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ActiveLearningConfig:
    candidate_source: str
    active_points: int
    sobol_tail_reserve: int
    batch_size: int
    preselect_factor: int
    probe_error_percentile: float
    probe_hotspot_weight: float
    probe_hotspot_percentile: float
    uncertainty_weight: float
    train_distance_weight: float
    boundary_risk_weight: float
    alc_imse_weight: float
    alc_probe_weight_floor: float
    pca_band_weights: tuple[float, ...]
    pca_weight_function: str
    pca_weight_transition_dex: float
    pca_component_weight_min: float
    pca_component_weight_max: float
    pca_component_weight_normalize: bool
    diversity_weight: float
    reduction_probe_anchors: int
    reduction_length_scale: float
    boundary_guard_threshold: float
    boundary_fraction_cap: float
    continuous_initial_draws: int
    continuous_hotspot_jitter_draws: int
    continuous_restarts: int
    continuous_local_maxiter: int
    continuous_duplicate_distance_threshold: float


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    primary_metric: str
    primary_curve: str
    report_metric_policy: str
    band_edges: tuple[float, ...]
    band_labels: tuple[str, ...]
    output_dir: Path
    save_predictions: bool


@dataclass(frozen=True, slots=True)
class Z2Config:
    config_path: Path
    package_root: Path
    project_root: Path
    data_root: Path
    random_seed: int
    target: TargetConfig
    parameter_space: ParameterSpace
    k_grid: KGrid
    resources: ResourceConfig
    fastmock_bias: FastMockBiasConfig
    splits: SplitConfig
    model: ModelConfig
    active_learning: ActiveLearningConfig
    evaluation: EvaluationConfig
    oracle_kind: str = "z2_truth_generator"

    def summary(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "target": {
                "kind": self.target.kind,
                "anchor_mode": self.target.anchor_mode,
                "power_eps": self.target.power_eps,
            },
            "oracle_kind": self.oracle_kind,
            "truth_generator_path": str(self.resources.truth_generator.path),
            "r2_seed_mode": self.resources.r2_seed.mode,
            "r2_seed_path": str(self.resources.r2_seed.path),
            "use_lofi": self.resources.use_lofi,
            "raw_bank": {
                "enabled": self.resources.raw_bank.enabled,
                "path": str(self.resources.raw_bank.path) if self.resources.raw_bank.path else None,
                "sample_size": self.resources.raw_bank.sample_size,
            },
            "fastmock_bias": {
                "enabled": self.fastmock_bias.enabled,
                "provider": self.fastmock_bias.provider,
                "score_mode": self.fastmock_bias.score_mode,
                "vendor_path": str(self.fastmock_bias.vendor_path) if self.fastmock_bias.vendor_path else None,
                "checkbound": self.fastmock_bias.checkbound,
                "fixed_w": self.fastmock_bias.fixed_w,
                "fixed_wa": self.fastmock_bias.fixed_wa,
                "fixed_mnu": self.fastmock_bias.fixed_mnu,
                "bias_weight": self.fastmock_bias.bias_weight,
                "bias_band_weights": list(self.fastmock_bias.bias_band_weights),
                "normalization": self.fastmock_bias.normalization,
                "normalization_probe_count": self.fastmock_bias.normalization_probe_count,
                "truth_backend": self.fastmock_bias.truth_backend,
                "truth_dtype": self.fastmock_bias.truth_dtype,
                "truth_device": self.fastmock_bias.truth_device,
                "truth_chunk_size": self.fastmock_bias.truth_chunk_size,
            },
            "active_learning": {
                "candidate_source": self.active_learning.candidate_source,
                "active_points": self.active_learning.active_points,
                "sobol_tail_reserve": self.active_learning.sobol_tail_reserve,
                "batch_size": self.active_learning.batch_size,
                "alc_imse_weight": self.active_learning.alc_imse_weight,
                "pca_band_weights": list(self.active_learning.pca_band_weights),
                "pca_weight_function": self.active_learning.pca_weight_function,
                "pca_weight_transition_dex": self.active_learning.pca_weight_transition_dex,
            },
            "evaluation": {
                "primary_metric": self.evaluation.primary_metric,
                "primary_curve": self.evaluation.primary_curve,
                "report_metric_policy": self.evaluation.report_metric_policy,
                "reported_quantile": 68.0,
                "band_edges": list(self.evaluation.band_edges),
                "band_labels": list(self.evaluation.band_labels),
            },
            "k_count": int(self.k_grid.k_bins.shape[0]),
            "split_sizes": {
                "probe": self.splits.probe_size,
                "pool": self.splits.pool_size,
                "audit": self.splits.audit_size,
                "sobol64": self.splits.sobol64_size,
                "sobol128": self.splits.sobol128_size,
                "sobol_tail": self.splits.sobol_tail_size,
            },
            "audit_source": {
                "source": self.splits.audit_source,
                "path": str(self.splits.audit_path) if self.splits.audit_path else None,
                "theta_unit_key": self.splits.audit_theta_unit_key,
                "theta_raw_key": self.splits.audit_theta_raw_key,
            },
        }


def load_config(path: str | Path) -> Z2Config:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"z2 config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("z2 config root must be a mapping.")
    return parse_config(raw, config_path=config_path)


def parse_config(raw: Mapping[str, Any], *, config_path: Path) -> Z2Config:
    package_root = config_path.parent.resolve()
    project_root = _resolve_path(raw.get("project_root", "../.."), package_root)
    data_root_text = os.environ.get("Z2QUIJOTE_DATA_ROOT", raw.get("data_root", "data"))
    data_root = _resolve_path(data_root_text, package_root)

    target_payload = _section(raw, "target")
    target = TargetConfig(
        kind=str(target_payload.get("kind", "direct_cdm_logpk")),
        anchor_mode=str(target_payload.get("anchor_mode", "none")),
        power_eps=float(target_payload.get("power_eps", 1.0e-12)),
    )

    parameter_space = ParameterSpace.from_mapping(_section(raw, "parameter_space"))
    k_grid = build_k_grid(_section(raw, "k_grid"), base_path=package_root)
    resource_payload = _section(raw, "resources")
    # `v2_root` is accepted only as a legacy config alias; new z2 configs use
    # `runtime_root`.
    runtime_root_payload = resource_payload.get("runtime_root", resource_payload.get("v2_root", "."))
    runtime_root = _resolve_path(runtime_root_payload, package_root)
    truth_payload = _section(resource_payload, "truth_generator")
    r2_payload = _section(resource_payload, "r2_seed")
    current_payload = resource_payload.get("current_active", {}) or {}
    if not isinstance(current_payload, Mapping):
        raise ValueError("resources.current_active must be a mapping.")
    raw_bank_payload = resource_payload.get("raw_bank", {}) or {}
    if not isinstance(raw_bank_payload, Mapping):
        raise ValueError("resources.raw_bank must be a mapping.")

    resources = ResourceConfig(
        use_lofi=bool(resource_payload.get("use_lofi", False)),
        runtime_root=runtime_root,
        truth_generator=TruthGeneratorConfig(
            kind=str(truth_payload.get("kind", "z2_direct_logpk_truth_generator")),
            path=_resolve_path(truth_payload["path"], runtime_root),
            device=str(truth_payload.get("device", "auto")),
            chunk_size=int(truth_payload.get("chunk_size", 256)),
        ),
        r2_seed=R2SeedConfig(
            mode=str(r2_payload.get("mode", "geometry_only")),
            path=_resolve_path(r2_payload["path"], runtime_root),
            theta_key=str(r2_payload.get("theta_key", "theta_raw")),
            manifest_path=(
                _resolve_path(r2_payload["manifest_path"], runtime_root)
                if r2_payload.get("manifest_path")
                else None
            ),
        ),
        current_active=CurrentActiveConfig(
            enabled=bool(current_payload.get("enabled", False)),
            path=(
                _resolve_path(current_payload["path"], package_root)
                if current_payload.get("path")
                else None
            ),
            theta_key=str(current_payload.get("theta_key", "theta_raw")),
        ),
        raw_bank=RawBankConfig(
            enabled=bool(raw_bank_payload.get("enabled", False)),
            path=(
                _resolve_path(raw_bank_payload["path"], runtime_root)
                if raw_bank_payload.get("path")
                else None
            ),
            metadata_path=(
                _resolve_path(raw_bank_payload["metadata_path"], runtime_root)
                if raw_bank_payload.get("metadata_path")
                else None
            ),
            sample_size=int(raw_bank_payload.get("sample_size", 256)),
            sample_seed=int(raw_bank_payload.get("sample_seed", 20260604)),
            filter_to_parameter_box=bool(raw_bank_payload.get("filter_to_parameter_box", False)),
        ),
    )

    fastmock_payload = raw.get("fastmock_bias", {}) or {}
    if not isinstance(fastmock_payload, Mapping):
        raise ValueError("fastmock_bias must be a mapping.")
    fastmock_bias = FastMockBiasConfig(
        enabled=bool(fastmock_payload.get("enabled", False)),
        provider=str(fastmock_payload.get("provider", "csst")),
        score_mode=str(fastmock_payload.get("score_mode", "variance_bias")),
        vendor_path=(
            _resolve_path(fastmock_payload["vendor_path"], package_root)
            if fastmock_payload.get("vendor_path")
            else None
        ),
        checkbound=bool(fastmock_payload.get("checkbound", False)),
        fixed_w=float(fastmock_payload.get("fixed_w", -1.0)),
        fixed_wa=float(fastmock_payload.get("fixed_wa", 0.0)),
        fixed_mnu=float(fastmock_payload.get("fixed_mnu", 0.0)),
        reference_as=float(fastmock_payload.get("reference_as", 2.1e-9)),
        bias_weight=float(fastmock_payload.get("bias_weight", 1.0)),
        bias_band_weights=_float_tuple(fastmock_payload.get("bias_band_weights", ())),
        normalization=str(fastmock_payload.get("normalization", "p95")),
        normalization_probe_count=int(fastmock_payload.get("normalization_probe_count", 128)),
        cache_decimals=int(fastmock_payload.get("cache_decimals", 10)),
        truth_backend=str(fastmock_payload.get("truth_backend", "cpu_batch")),
        truth_dtype=str(fastmock_payload.get("truth_dtype", "float32")),
        truth_device=str(fastmock_payload.get("truth_device", "auto")),
        truth_chunk_size=int(fastmock_payload.get("truth_chunk_size", 6144)),
    )

    splits_payload = _section(raw, "splits")
    audit_path = splits_payload.get("audit_path")
    splits = SplitConfig(
        seed_label=str(splits_payload.get("seed_label", "r2_seed_geometry_64")),
        probe_size=int(splits_payload.get("probe_size", 256)),
        pool_size=int(splits_payload.get("pool_size", 8192)),
        audit_size=int(splits_payload.get("audit_size", 512)),
        audit_source=str(splits_payload.get("audit_source", "sobol")),
        audit_path=(
            _resolve_path(audit_path, package_root)
            if audit_path not in (None, "")
            else None
        ),
        audit_theta_unit_key=str(splits_payload.get("audit_theta_unit_key", "theta_unit")),
        audit_theta_raw_key=str(splits_payload.get("audit_theta_raw_key", "theta_raw")),
        sobol64_size=int(splits_payload.get("sobol64_size", 64)),
        sobol128_size=int(splits_payload.get("sobol128_size", 128)),
        sobol_tail_size=int(splits_payload.get("sobol_tail_size", 64)),
        duplicate_decimals=int(splits_payload.get("duplicate_decimals", 12)),
        output_dir=_resolve_path(splits_payload.get("output_dir", "manifests"), data_root),
    )

    model_payload = _section(raw, "model")
    model = ModelConfig(
        pca_components=int(model_payload.get("pca_components", 16)),
        gp_alpha=float(model_payload.get("gp_alpha", 1.0e-8)),
        gp_n_restarts_optimizer=int(model_payload.get("gp_n_restarts_optimizer", 0)),
        normalize_y=bool(model_payload.get("normalize_y", True)),
        length_scale_initial=float(model_payload.get("length_scale_initial", 0.25)),
        length_scale_bounds=_float_pair(model_payload.get("length_scale_bounds", (0.01, 20.0))),
        constant_value=float(model_payload.get("constant_value", 1.0)),
        constant_value_bounds=_float_pair(model_payload.get("constant_value_bounds", (0.01, 100.0))),
    )

    active_payload = _section(raw, "active_learning")
    active = ActiveLearningConfig(
        candidate_source=str(active_payload.get("candidate_source", "pool")),
        active_points=int(active_payload.get("active_points", 64)),
        sobol_tail_reserve=int(active_payload.get("sobol_tail_reserve", 0)),
        batch_size=int(active_payload.get("batch_size", 16)),
        preselect_factor=int(active_payload.get("preselect_factor", 12)),
        probe_error_percentile=float(active_payload.get("probe_error_percentile", 68.0)),
        probe_hotspot_weight=float(active_payload.get("probe_hotspot_weight", 0.0)),
        probe_hotspot_percentile=float(active_payload.get("probe_hotspot_percentile", 80.0)),
        uncertainty_weight=float(active_payload.get("uncertainty_weight", 0.35)),
        train_distance_weight=float(active_payload.get("train_distance_weight", 0.15)),
        boundary_risk_weight=float(active_payload.get("boundary_risk_weight", 0.10)),
        alc_imse_weight=float(active_payload.get("alc_imse_weight", 0.0)),
        alc_probe_weight_floor=float(active_payload.get("alc_probe_weight_floor", 0.05)),
        pca_band_weights=_float_tuple(active_payload.get("pca_band_weights", ())),
        pca_weight_function=str(active_payload.get("pca_weight_function", "smooth_logk_curve")),
        pca_weight_transition_dex=float(active_payload.get("pca_weight_transition_dex", 0.10)),
        pca_component_weight_min=float(active_payload.get("pca_component_weight_min", 0.25)),
        pca_component_weight_max=float(active_payload.get("pca_component_weight_max", 4.0)),
        pca_component_weight_normalize=bool(active_payload.get("pca_component_weight_normalize", True)),
        diversity_weight=float(active_payload.get("diversity_weight", 0.45)),
        reduction_probe_anchors=int(active_payload.get("reduction_probe_anchors", 32)),
        reduction_length_scale=float(active_payload.get("reduction_length_scale", 0.20)),
        boundary_guard_threshold=float(active_payload.get("boundary_guard_threshold", 0.055)),
        boundary_fraction_cap=float(active_payload.get("boundary_fraction_cap", 0.35)),
        continuous_initial_draws=int(active_payload.get("continuous_initial_draws", 4096)),
        continuous_hotspot_jitter_draws=int(active_payload.get("continuous_hotspot_jitter_draws", 1024)),
        continuous_restarts=int(active_payload.get("continuous_restarts", 24)),
        continuous_local_maxiter=int(active_payload.get("continuous_local_maxiter", 35)),
        continuous_duplicate_distance_threshold=float(
            active_payload.get("continuous_duplicate_distance_threshold", 1.0e-4)
        ),
    )

    evaluation_payload = _section(raw, "evaluation")
    evaluation = EvaluationConfig(
        primary_metric=str(evaluation_payload.get("primary_metric", "overall_relative_error.p68")),
        primary_curve=str(evaluation_payload.get("primary_curve", "kwise_p68_relative_error")),
        report_metric_policy=str(evaluation_payload.get("report_metric_policy", "p68_only")),
        band_edges=tuple(float(item) for item in evaluation_payload.get("band_edges", [])),
        band_labels=tuple(str(item) for item in evaluation_payload.get("band_labels", [])),
        output_dir=_resolve_path(evaluation_payload.get("output_dir", "runs"), data_root),
        save_predictions=bool(evaluation_payload.get("save_predictions", False)),
    )

    config = Z2Config(
        config_path=config_path,
        package_root=package_root,
        project_root=project_root,
        data_root=data_root,
        random_seed=int(raw.get("random_seed", 20260604)),
        target=target,
        parameter_space=parameter_space,
        k_grid=k_grid,
        resources=resources,
        fastmock_bias=fastmock_bias,
        splits=splits,
        model=model,
        active_learning=active,
        evaluation=evaluation,
        oracle_kind=str(raw.get("oracle_kind", "z2_truth_generator")),
    )
    validate_config(config)
    return config


def make_smoke_config(config: Z2Config, *, data_root: Path | None = None) -> Z2Config:
    root = (data_root.resolve() if data_root else config.data_root / "smoke").resolve()
    smoke_k_grid = build_k_grid(
        {
            "bands": [
                {"name": "low", "k_min": 0.01, "k_max": 0.07, "count": 12},
                {"name": "mid", "k_min": 0.07, "k_max": 0.5, "count": 24},
                {"name": "focus_high", "k_min": 0.5, "k_max": 1.0, "count": 14},
                {"name": "tail", "k_min": 1.0, "k_max": 3.0, "count": 14},
            ]
        }
    )
    smoke_splits = replace(
        config.splits,
        probe_size=12,
        pool_size=48,
        audit_size=16,
        audit_source="sobol",
        audit_path=None,
        sobol64_size=16,
        sobol128_size=16,
        sobol_tail_size=8,
        output_dir=root / "manifests",
    )
    smoke_model = replace(config.model, pca_components=4, gp_n_restarts_optimizer=0)
    smoke_active = replace(
        config.active_learning,
        active_points=8,
        sobol_tail_reserve=min(config.active_learning.sobol_tail_reserve, 2),
        batch_size=4,
        reduction_probe_anchors=8,
    )
    smoke_eval = replace(config.evaluation, output_dir=root / "runs", save_predictions=False)
    return replace(
        config,
        data_root=root,
        target=TargetConfig(kind="direct_cdm_logpk", anchor_mode="none", power_eps=config.target.power_eps),
        k_grid=smoke_k_grid,
        fastmock_bias=replace(config.fastmock_bias, enabled=False),
        splits=smoke_splits,
        model=smoke_model,
        active_learning=smoke_active,
        evaluation=smoke_eval,
        oracle_kind="synthetic_direct_cdm",
    )


def validate_config(config: Z2Config) -> None:
    target_kind = str(config.target.kind).strip().lower()
    anchor_mode = str(config.target.anchor_mode).strip().lower()
    if target_kind not in {"direct_cdm_logpk", "cdm_logdiff"}:
        raise ValueError("z2 target.kind must be direct_cdm_logpk or cdm_logdiff.")
    if target_kind == "direct_cdm_logpk" and anchor_mode not in {"none", "direct", "cdm"}:
        raise ValueError("direct_cdm_logpk requires target.anchor_mode to be none/direct/cdm.")
    if target_kind == "cdm_logdiff" and anchor_mode not in {
        "camb_cdm_hmcode2020",
        "camb_cdm_nonlinear",
        "cdm_hmcode2020",
    }:
        raise ValueError(
            "cdm_logdiff requires target.anchor_mode to be one of "
            "{'camb_cdm_hmcode2020', 'camb_cdm_nonlinear', 'cdm_hmcode2020'}."
        )
    if config.resources.use_lofi:
        raise ValueError("z2 forbids LoFi usage; resources.use_lofi must be false.")
    if config.resources.r2_seed.mode != "geometry_only":
        raise ValueError("R2 seed resource must be used in geometry_only mode.")
    if config.resources.truth_generator.kind not in {
        "z2_direct_logpk_truth_generator",
        "v2_direct_logpk_truth_generator",
    }:
        raise ValueError("truth generator kind must be z2_direct_logpk_truth_generator.")
    for value, name in (
        (config.splits.probe_size, "splits.probe_size"),
        (config.splits.pool_size, "splits.pool_size"),
        (config.splits.audit_size, "splits.audit_size"),
        (config.splits.sobol64_size, "splits.sobol64_size"),
        (config.splits.sobol128_size, "splits.sobol128_size"),
        (config.splits.sobol_tail_size, "splits.sobol_tail_size"),
        (config.active_learning.active_points, "active_learning.active_points"),
        (config.active_learning.batch_size, "active_learning.batch_size"),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")
    audit_source = str(config.splits.audit_source).strip().lower()
    if audit_source not in {"sobol", "npz"}:
        raise ValueError("splits.audit_source must be sobol or npz.")
    if audit_source == "npz" and config.splits.audit_path is None:
        raise ValueError("splits.audit_path is required when splits.audit_source=npz.")
    if config.active_learning.candidate_source not in {"pool", "continuous", "m3"}:
        raise ValueError("active_learning.candidate_source must be pool, continuous, or m3.")
    if config.active_learning.sobol_tail_reserve < 0:
        raise ValueError("active_learning.sobol_tail_reserve must be non-negative.")
    if config.active_learning.sobol_tail_reserve >= config.active_learning.active_points:
        raise ValueError("active_learning.sobol_tail_reserve must be smaller than active_points.")
    if config.active_learning.continuous_initial_draws <= 0:
        raise ValueError("active_learning.continuous_initial_draws must be positive.")
    if config.active_learning.continuous_restarts <= 0:
        raise ValueError("active_learning.continuous_restarts must be positive.")
    if abs(float(config.active_learning.probe_error_percentile) - 68.0) > 1.0e-9:
        raise ValueError("z2 uses only the p68 line; active_learning.probe_error_percentile must be 68.0.")
    if str(config.evaluation.primary_metric).strip() != "overall_relative_error.p68":
        raise ValueError("z2 evaluation.primary_metric must be overall_relative_error.p68.")
    if str(config.evaluation.primary_curve).strip() != "kwise_p68_relative_error":
        raise ValueError("z2 evaluation.primary_curve must be kwise_p68_relative_error.")
    if str(config.evaluation.report_metric_policy).strip() != "p68_only":
        raise ValueError("z2 evaluation.report_metric_policy must be p68_only.")
    if config.active_learning.alc_imse_weight < 0.0:
        raise ValueError("active_learning.alc_imse_weight must be non-negative.")
    if config.active_learning.alc_probe_weight_floor < 0.0:
        raise ValueError("active_learning.alc_probe_weight_floor must be non-negative.")
    if config.active_learning.pca_component_weight_min <= 0.0:
        raise ValueError("active_learning.pca_component_weight_min must be positive.")
    if config.active_learning.pca_component_weight_max < config.active_learning.pca_component_weight_min:
        raise ValueError("active_learning.pca_component_weight_max must be >= pca_component_weight_min.")
    weight_function = str(config.active_learning.pca_weight_function).strip().lower()
    if weight_function not in {"band_integral", "smooth_logk_curve"}:
        raise ValueError(
            "active_learning.pca_weight_function must be one of "
            "{'band_integral', 'smooth_logk_curve'}."
        )
    if config.active_learning.pca_weight_transition_dex <= 0.0:
        raise ValueError("active_learning.pca_weight_transition_dex must be positive.")
    if config.active_learning.pca_band_weights:
        if len(config.active_learning.pca_band_weights) != len(config.k_grid.bands):
            raise ValueError("active_learning.pca_band_weights must match the number of k_grid.bands.")
        if any((not np.isfinite(value) or value < 0.0) for value in config.active_learning.pca_band_weights):
            raise ValueError("active_learning.pca_band_weights must contain finite non-negative values.")
        if sum(config.active_learning.pca_band_weights) <= 0.0:
            raise ValueError("active_learning.pca_band_weights must not be all zero.")
    if config.fastmock_bias.enabled:
        if str(config.fastmock_bias.provider).strip().lower() != "csst":
            raise ValueError("fastmock_bias.provider must be csst.")
        if str(config.fastmock_bias.score_mode).strip().lower() not in {"variance_bias", "bias_only"}:
            raise ValueError("fastmock_bias.score_mode must be variance_bias or bias_only.")
        if config.fastmock_bias.vendor_path is None:
            raise ValueError("fastmock_bias.vendor_path is required when fastmock_bias.enabled is true.")
        if target_kind != "cdm_logdiff":
            raise ValueError("fastmock_bias currently requires target.kind=cdm_logdiff.")
        if config.active_learning.candidate_source != "m3":
            raise ValueError("fastmock_bias currently only applies to active_learning.candidate_source=m3.")
        if config.fastmock_bias.bias_weight < 0.0:
            raise ValueError("fastmock_bias.bias_weight must be non-negative.")
        if config.fastmock_bias.bias_band_weights:
            if len(config.fastmock_bias.bias_band_weights) != len(config.k_grid.bands):
                raise ValueError("fastmock_bias.bias_band_weights must match the number of k_grid.bands.")
            if any((not np.isfinite(value) or value < 0.0) for value in config.fastmock_bias.bias_band_weights):
                raise ValueError("fastmock_bias.bias_band_weights must contain finite non-negative values.")
            if sum(config.fastmock_bias.bias_band_weights) <= 0.0:
                raise ValueError("fastmock_bias.bias_band_weights must not be all zero.")
        if config.fastmock_bias.normalization.strip().lower() not in {"none", "p68", "p95", "median"}:
            raise ValueError("fastmock_bias.normalization must be one of {'none', 'p68', 'p95', 'median'}.")
        if config.fastmock_bias.normalization_probe_count <= 0:
            raise ValueError("fastmock_bias.normalization_probe_count must be positive.")
        if config.fastmock_bias.cache_decimals < 0:
            raise ValueError("fastmock_bias.cache_decimals must be non-negative.")
        if config.fastmock_bias.truth_backend.strip().lower() not in {
            "auto",
            "cuda",
            "cuda_torch",
            "torch_cuda",
            "cuda_if_available",
            "cpu",
            "cpu_batch",
            "numpy",
        }:
            raise ValueError("fastmock_bias.truth_backend must be cpu_batch, auto, or cuda_torch.")
        if config.fastmock_bias.truth_dtype.strip().lower() not in {"float32", "single", "float64", "double"}:
            raise ValueError("fastmock_bias.truth_dtype must be float32 or float64.")
        if config.fastmock_bias.truth_chunk_size <= 0:
            raise ValueError("fastmock_bias.truth_chunk_size must be positive.")


def _section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = payload.get(key, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return section


def _resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _float_pair(value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expected a pair, got {value!r}.")
    return (float(value[0]), float(value[1]))


def _float_tuple(value: object) -> tuple[float, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"expected a list of floats, got {value!r}.")
    return tuple(float(item) for item in value)
