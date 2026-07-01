from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import Z2Config
from .emulator import PCAGPDirectCDMEmulator
from .runtime_core.config import ValidationRuntimeConfig, load_config as load_runtime_config
from .runtime_core.module3_facade import build_default_online_selector, select_next_batch
from .runtime_core.types import ContinuousPosteriorState, Module3ContinuousInput


@dataclass(frozen=True, slots=True)
class M3SelectionBatch:
    selected_theta_raw: np.ndarray
    selected_theta_unit: np.ndarray
    selected_source_pc: np.ndarray
    selected_scores: np.ndarray
    metadata: dict[str, Any]


def select_m3_batch(
    *,
    config: Z2Config,
    emulator: PCAGPDirectCDMEmulator,
    train_theta_raw: np.ndarray,
    iteration_index: int,
    batch_size: int,
    bias_model: Any | None = None,
    probe_theta_raw: np.ndarray | None = None,
    probe_truth_log: np.ndarray | None = None,
    k_bins: np.ndarray | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> M3SelectionBatch:
    """Select one batch with the z2-owned Module-3 selector.

    z2 supplies a direct-CDM PCA+GP posterior. The runtime-core selector consumes
    only the posterior state and runtime M3 settings, so the target remains z2's
    direct CDM target.
    """

    runtime = _load_runtime_config()
    _adapt_runtime_for_z2_target(runtime, config=config)
    runtime.random_seed = int(config.random_seed + 1000 * int(iteration_index))
    runtime.sampling.batch_size = int(batch_size)
    runtime.device = "cuda"
    runtime.m3.chunk_size = _env_int("Z2_M3_CHUNK_SIZE", 6144)
    runtime.m3.stage0_chunk_size = _env_int("Z2_M3_STAGE0_CHUNK_SIZE", 6144)
    runtime.m3.stage3_qmc_chunk_size = _env_int("Z2_M3_STAGE_QMC_CHUNK_SIZE", 7168)
    p68_probe_shell = _build_z2_p68_probe_shell(
        config=config,
        runtime=runtime,
        emulator=emulator,
        probe_theta_raw=probe_theta_raw,
        probe_truth_log=probe_truth_log,
        k_bins=k_bins,
    )

    continuous_state = ContinuousPosteriorState(
        theta_bounds=np.asarray(config.parameter_space.theta_bounds, dtype=np.float64),
        train_raw_thetas=np.asarray(train_theta_raw, dtype=np.float64),
        train_unit_thetas=config.parameter_space.normalize(train_theta_raw),
        gp_models=list(emulator.gp_models or []),
        kernel_descriptions=[str(getattr(gp, "kernel_", gp)) for gp in (emulator.gp_models or [])],
        pc_lengthscales=_extract_lengthscales(emulator),
        pc_signal_variances=_extract_signal_variances(emulator),
        metadata=_continuous_metadata(
            config,
            emulator,
            train_theta_raw,
            bias_model=bias_model,
            p68_probe_shell=p68_probe_shell,
        ),
    )
    target_transform = _target_transform(config)
    module3_input = Module3ContinuousInput(
        continuous_state=continuous_state,
        iteration_index=int(iteration_index),
        metadata={
            "iteration_index": int(iteration_index),
            "batch_size": int(batch_size),
            "train_size": int(np.asarray(train_theta_raw).shape[0]),
            "target_transform": target_transform,
            "z2_adapter": "z2_target_to_runtime_core_m3",
        "pca_weight_function": str(config.active_learning.pca_weight_function),
        "pca_band_weights": [float(value) for value in config.active_learning.pca_band_weights],
        "fastmock_bias_enabled": bool(bias_model is not None),
        "fastmock_bias_band_weights": (
            []
            if bias_model is None
            else [float(value) for value in np.asarray(bias_model.bias_band_weights, dtype=np.float64).reshape(-1)]
        ),
        "p68_validation_probe_shell": dict(p68_probe_shell),
        },
    )
    selection = select_next_batch(
        runtime,
        module3_input,
        selector=build_default_online_selector(),
        progress_callback=progress_callback,
    )
    metadata = dict(selection.metadata)
    metadata["z2_m3_adapter"] = {
        "target_transform": target_transform,
        "target_kind": str(config.target.kind),
        "anchor_mode": str(config.target.anchor_mode),
        "pca_weight_function": str(config.active_learning.pca_weight_function),
        "pca_band_weights": [float(value) for value in config.active_learning.pca_band_weights],
        "runtime_objective_mode": str(getattr(runtime.m3, "objective_mode", "unknown")),
        "runtime_device": str(getattr(runtime, "device", "unknown")),
        "runtime_m3_chunk_size": int(getattr(runtime.m3, "chunk_size", 0)),
        "runtime_representation_band_weights": [
            float(value) for value in getattr(runtime.m3, "representation_band_weights", ())
        ],
        "fastmock_bias_enabled": bool(bias_model is not None),
        "fastmock_bias_band_weights": (
            []
            if bias_model is None
            else [float(value) for value in np.asarray(bias_model.bias_band_weights, dtype=np.float64).reshape(-1)]
        ),
        "fastmock_bias_weight_details": (
            {} if bias_model is None else dict(getattr(bias_model, "bias_weight_details", {}))
        ),
        "p68_probe_shell_enabled": bool(p68_probe_shell.get("enabled", False)),
        "p68_probe_shell_source": str(p68_probe_shell.get("source", "disabled")),
    }
    return M3SelectionBatch(
        selected_theta_raw=np.asarray(selection.selected_raw_thetas, dtype=np.float64),
        selected_theta_unit=np.asarray(selection.selected_unit_thetas, dtype=np.float64),
        selected_source_pc=np.asarray(selection.selected_source_pc, dtype=np.int64),
        selected_scores=np.asarray(selection.selected_scores, dtype=np.float64),
        metadata=metadata,
    )


def _load_runtime_config() -> ValidationRuntimeConfig:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "module3_runtime.yaml"
    if config_path.exists():
        return load_runtime_config(config_path)
    return ValidationRuntimeConfig()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    return max(1, value)


def _adapt_runtime_for_z2_target(runtime: Any, *, config: Z2Config) -> None:
    # Direct logP has no representation groups in z2, so keep using the
    # runtime-core band-sensitivity path. Logdiff can use exact k-band integrals.
    if (
        str(config.target.kind).strip().lower() == "direct_cdm_logpk"
        and str(getattr(runtime.m3, "objective_mode", "")).strip() == "representation_grouped_posterior_variance"
    ):
        runtime.m3.objective_mode = "mid_high_weighted_sum"
    if config.active_learning.pca_band_weights:
        band_weights = tuple(float(value) for value in config.active_learning.pca_band_weights)
        runtime.m3.representation_band_weights = band_weights
        if len(band_weights) == 4:
            runtime.m3.acquisition_p68_set_rerank_band_weights = band_weights
    runtime.m3.acquisition_p68_set_rerank_boundary_threshold = float(
        config.active_learning.boundary_guard_threshold
    )
    runtime.m3.acquisition_p68_set_rerank_boundary_target_fraction = float(
        config.active_learning.boundary_fraction_cap
    )


def _extract_lengthscales(emulator: PCAGPDirectCDMEmulator) -> np.ndarray:
    rows: list[np.ndarray] = []
    for gp in emulator.gp_models or []:
        kernel = getattr(gp, "kernel_", None)
        rbf = getattr(kernel, "k2", kernel)
        length_scale = np.asarray(getattr(rbf, "length_scale", 1.0), dtype=np.float64).reshape(-1)
        if length_scale.size == 1:
            length_scale = np.repeat(length_scale, emulator.parameter_space.dim)
        rows.append(length_scale.astype(np.float64))
    if not rows:
        return np.empty((0, emulator.parameter_space.dim), dtype=np.float64)
    return np.vstack(rows).astype(np.float64)


def _extract_signal_variances(emulator: PCAGPDirectCDMEmulator) -> np.ndarray:
    values: list[float] = []
    for gp in emulator.gp_models or []:
        kernel = getattr(gp, "kernel_", None)
        constant = getattr(getattr(kernel, "k1", kernel), "constant_value", 1.0)
        values.append(float(max(float(constant), 1.0e-12)))
    return np.asarray(values, dtype=np.float64)


def _continuous_metadata(
    config: Z2Config,
    emulator: PCAGPDirectCDMEmulator,
    train_theta_raw: np.ndarray,
    *,
    bias_model: Any | None = None,
    p68_probe_shell: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pca_components = np.asarray(emulator.pca.components_, dtype=np.float64)
    k_bins = np.asarray(emulator.k_bins, dtype=np.float64)
    band_integrals, global_integral = _pca_band_integrals(pca_components, k_bins)
    sensitivity = _normalize_rows(band_integrals)
    target_transform = _target_transform(config)
    transform_family = "logdiff" if str(config.target.kind).strip().lower() == "cdm_logdiff" else "direct_logpk"
    anchor_mode = str(config.target.anchor_mode) if transform_family == "logdiff" else "none"
    metadata = {
        "train_size": int(np.asarray(train_theta_raw).shape[0]),
        "theta_dim": int(config.parameter_space.dim),
        "pc_dim": int(pca_components.shape[0]),
        "kernel_family": "sklearn_constant_rbf",
        "pca_score_std": np.asarray(emulator.score_std, dtype=np.float64).reshape(-1).tolist(),
        "pca_band_labels": ["low_0.01_0.07", "mid_0.07_0.5", "high_0.5_1", "tail_1_3"],
        "pca_band_sensitivity": sensitivity.tolist(),
        "pca_band_variance_integrals": band_integrals.tolist(),
        "pca_global_variance_integral": global_integral.tolist(),
        "pca_mid_high_sensitivity": np.sum(sensitivity[:, 1:], axis=1).astype(np.float64).tolist(),
        "pca_focus_0p08_3_sensitivity": np.sum(sensitivity[:, 1:], axis=1).astype(np.float64).tolist(),
        "pca_focus_0p1_3_sensitivity": np.sum(sensitivity[:, 1:], axis=1).astype(np.float64).tolist(),
        "pca_scheme": "global_pca",
        "pca_layout": {},
        "representation_component_groups": [],
        "target_transform": target_transform,
        "representation_transform_family": transform_family,
        "representation_anchor_mode": anchor_mode,
        "z2_target_kind": config.target.kind,
        "z2_adapter": "z2_target_to_runtime_core_m3",
    }
    if bias_model is not None:
        metadata["z2_csst_bias_model"] = bias_model
        metadata["z2_csst_bias_enabled"] = True
    if p68_probe_shell is not None:
        metadata["p68_validation_probe_shell"] = dict(p68_probe_shell)
    return metadata


def _build_z2_p68_probe_shell(
    *,
    config: Z2Config,
    runtime: Any,
    emulator: PCAGPDirectCDMEmulator,
    probe_theta_raw: np.ndarray | None,
    probe_truth_log: np.ndarray | None,
    k_bins: np.ndarray | None,
) -> dict[str, Any]:
    m3 = runtime.m3
    if str(m3.acquisition_p68_set_rerank_risk_mode).strip().lower() != "validation_probe_shell":
        return {"enabled": False, "reason": "risk_mode_disabled"}
    if int(m3.acquisition_p68_set_rerank_top_k) <= 0:
        return {"enabled": False, "reason": "rerank_top_k_zero"}
    if float(m3.acquisition_p68_set_rerank_p68_weight) <= 0.0:
        return {"enabled": False, "reason": "p68_weight_zero"}
    if int(m3.acquisition_p68_validation_probe_size) <= 0:
        return {"enabled": False, "reason": "probe_size_zero"}
    if probe_theta_raw is None or probe_truth_log is None or k_bins is None:
        return {"enabled": False, "reason": "z2_probe_not_supplied"}

    probe_raw_all = np.asarray(probe_theta_raw, dtype=np.float64)
    truth_all = np.asarray(probe_truth_log, dtype=np.float64)
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    if probe_raw_all.ndim != 2 or truth_all.ndim != 2 or truth_all.shape[0] != probe_raw_all.shape[0]:
        return {"enabled": False, "reason": "probe_shape_mismatch"}
    probe_size = min(int(m3.acquisition_p68_validation_probe_size), int(probe_raw_all.shape[0]))
    if probe_size <= 1:
        return {"enabled": False, "reason": "not_enough_probe_points"}

    probe_raw = probe_raw_all[:probe_size]
    truth_log = truth_all[:probe_size]
    prediction = emulator.predict(probe_raw)
    pred_log = np.asarray(prediction.log_pk_mean, dtype=np.float64)
    if pred_log.shape != truth_log.shape or pred_log.shape[1] != k_arr.shape[0]:
        return {"enabled": False, "reason": "prediction_shape_mismatch"}

    errors = _integrated_log_residual_relative_error_per_sample(
        k_bins=k_arr,
        truth_log=truth_log,
        pred_log=pred_log,
        k_min=float(m3.acquisition_p68_validation_probe_focus_k_min),
        k_max=float(m3.acquisition_p68_validation_probe_focus_k_max),
        eps=float(config.target.power_eps),
    )
    q68 = float(np.percentile(errors, 68.0))
    shell_width = float(m3.acquisition_p68_validation_probe_shell_width)
    scale = max(shell_width * max(q68, float(config.target.power_eps)), 1.0e-12)
    shell = np.exp(-0.5 * np.square((errors - q68) / scale))
    floor = float(m3.acquisition_p68_validation_probe_min_weight)
    weights = floor + (1.0 - floor) * shell
    weights = weights / max(float(np.mean(weights)), 1.0e-30)
    return {
        "enabled": True,
        "mode": "validation_probe_shell",
        "source": "z2_split_probe",
        "probe_size": int(probe_size),
        "probe_unit_thetas": config.parameter_space.normalize(probe_raw).astype(np.float64).tolist(),
        "probe_weights": weights.astype(np.float64).tolist(),
        "target_kind": str(config.target.kind),
        "focus_k_min": float(m3.acquisition_p68_validation_probe_focus_k_min),
        "focus_k_max": float(m3.acquisition_p68_validation_probe_focus_k_max),
        "shell_width_fraction": shell_width,
        "min_weight": floor,
        "error_q68": q68,
        "error_min": float(np.min(errors)),
        "error_max": float(np.max(errors)),
        "error_mean": float(np.mean(errors)),
        "error_statistic": "sample_mean_relative_error_then_probe_q68_shell",
    }


def _integrated_log_residual_relative_error_per_sample(
    *,
    k_bins: np.ndarray,
    truth_log: np.ndarray,
    pred_log: np.ndarray,
    k_min: float,
    k_max: float,
    eps: float,
) -> np.ndarray:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth_log, dtype=np.float64)
    pred = np.asarray(pred_log, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 2 or truth.shape[1] != k_arr.shape[0]:
        raise ValueError("truth_log, pred_log, and k_bins must align for z2 p68 probe shell.")
    mask = (k_arr >= float(k_min)) & (k_arr <= float(k_max))
    if not np.any(mask):
        mask = np.ones_like(k_arr, dtype=bool)
    relative = np.abs(np.exp(pred[:, mask] - truth[:, mask]) - 1.0)
    relative = np.nan_to_num(relative, nan=0.0, posinf=0.0, neginf=0.0)
    return np.mean(np.maximum(relative, float(max(eps, 0.0))), axis=1).astype(np.float64)


def _target_transform(config: Z2Config) -> str:
    if str(config.target.kind).strip().lower() == "direct_cdm_logpk":
        return "direct_logpk"
    anchor = str(config.target.anchor_mode).strip().lower() or "camb_cdm_hmcode2020"
    return f"log_hi_minus_log_{anchor}_anchor"


def _pca_band_integrals(pca_components: np.ndarray, k_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masks = [
        (k_bins >= 0.01) & (k_bins < 0.07),
        (k_bins >= 0.07) & (k_bins < 0.5),
        (k_bins >= 0.5) & (k_bins < 1.0),
        (k_bins >= 1.0) & (k_bins <= 3.0),
    ]
    rows: list[np.ndarray] = []
    squared = np.asarray(pca_components, dtype=np.float64) ** 2
    for mask in masks:
        if np.any(mask):
            rows.append(np.sum(squared[:, mask], axis=1))
        else:
            rows.append(np.zeros((squared.shape[0],), dtype=np.float64))
    band_integrals = np.vstack(rows).T.astype(np.float64)
    global_integral = np.sum(band_integrals, axis=1).astype(np.float64)
    return band_integrals, global_integral


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    denom = np.maximum(np.sum(arr, axis=1, keepdims=True), 1.0e-12)
    return arr / denom
