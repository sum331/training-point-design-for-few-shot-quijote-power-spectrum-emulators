from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import Z2Config
from .csst_fastmock import fit_csst_bias_model
from .direct_cdm import DirectCDMOracle
from .emulator import PCAGPDirectCDMEmulator
from .m3_adapter import select_m3_batch
from .parameter_space import boundary_distance, nearest_distance
from .sampling import sobol_unit, unique_unit_rows


@dataclass(slots=True)
class ProbeErrorCalibrator:
    percentile: float
    scaler: StandardScaler | None = None
    model: Ridge | None = None
    fallback_scale: float = 1.0

    def fit(self, features: np.ndarray, errors: np.ndarray) -> "ProbeErrorCalibrator":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(errors, dtype=np.float64).reshape(-1)
        mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y) & (y >= 0.0)
        x = x[mask]
        y = y[mask]
        if x.shape[0] < max(4, x.shape[1] + 1):
            self.fallback_scale = float(np.nanmedian(y)) if y.size else 1.0
            return self
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        model = Ridge(alpha=1.0)
        model.fit(x_scaled, np.log(np.maximum(y, 1.0e-12)))
        self.scaler = scaler
        self.model = model
        self.fallback_scale = float(np.nanmedian(y))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if self.scaler is None or self.model is None:
            return np.full((x.shape[0],), max(self.fallback_scale, 1.0e-12), dtype=np.float64)
        return np.exp(self.model.predict(self.scaler.transform(x))).astype(np.float64)


@dataclass(frozen=True, slots=True)
class ActiveSelectionResult:
    selected_theta_raw: np.ndarray
    selected_pool_indices: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ALCIMSEState:
    enabled: bool
    probe_unit: np.ndarray
    probe_weights: np.ndarray
    band_weights: np.ndarray
    component_weights: np.ndarray
    weight_details: dict[str, Any]
    component_gain: np.ndarray
    gp_models: tuple[Any, ...]
    v_probe: tuple[np.ndarray, ...]
    target_variance_scale: np.ndarray
    observation_noise: np.ndarray


@dataclass(frozen=True, slots=True)
class ScoreContext:
    uncertainty_bounds: tuple[float, float]
    train_distance_bounds: tuple[float, float]
    boundary_penalty_bounds: tuple[float, float]
    calibrated_bounds: tuple[float, float]
    hotspot_bounds: tuple[float, float]
    alc_imse_bounds: tuple[float, float]
    alc_state: ALCIMSEState | None


@dataclass(slots=True)
class DynamicWeightState:
    variance_band_weights_default: np.ndarray
    variance_band_weights_current: np.ndarray
    bias_band_weights_default: np.ndarray
    bias_band_weights_current: np.ndarray
    lambda_bias_weight: float

    @classmethod
    def from_config(cls, config: Z2Config) -> "DynamicWeightState":
        variance = _pca_band_weights(config)
        bias = _default_bias_band_weights(config)
        return cls(
            variance_band_weights_default=variance.astype(np.float64).copy(),
            variance_band_weights_current=variance.astype(np.float64).copy(),
            bias_band_weights_default=bias.astype(np.float64).copy(),
            bias_band_weights_current=bias.astype(np.float64).copy(),
            lambda_bias_weight=float(config.fastmock_bias.bias_weight),
        )

    def round_config(self, config: Z2Config) -> Z2Config:
        variance_weights = tuple(float(value) for value in self.variance_band_weights_current.reshape(-1))
        bias_weights = tuple(float(value) for value in self.bias_band_weights_current.reshape(-1))
        active_learning = config.active_learning
        fastmock_bias = config.fastmock_bias
        if variance_weights != tuple(float(value) for value in config.active_learning.pca_band_weights):
            active_learning = replace(
                active_learning,
                pca_band_weights=variance_weights,
            )
        if bias_weights != tuple(float(value) for value in config.fastmock_bias.bias_band_weights):
            fastmock_bias = replace(
                fastmock_bias,
                bias_band_weights=bias_weights,
            )
        if active_learning is config.active_learning and fastmock_bias is config.fastmock_bias:
            return config
        return replace(
            config,
            active_learning=active_learning,
            fastmock_bias=fastmock_bias,
        )

    def update_variance_band_weights(self, weights: np.ndarray) -> None:
        self.variance_band_weights_current = _coerce_dynamic_weights(
            weights,
            expected_size=int(self.variance_band_weights_default.size),
            name="variance_band_weights",
        )

    def update_bias_band_weights(self, weights: np.ndarray) -> None:
        self.bias_band_weights_current = _coerce_dynamic_weights(
            weights,
            expected_size=int(self.bias_band_weights_default.size),
            name="bias_band_weights",
        )

    def restore_defaults(self) -> None:
        self.variance_band_weights_current = self.variance_band_weights_default.astype(np.float64).copy()
        self.bias_band_weights_current = self.bias_band_weights_default.astype(np.float64).copy()

    def snapshot(self) -> dict[str, Any]:
        return {
            "variance_band_weights": self.variance_band_weights_current.astype(np.float64).tolist(),
            "bias_band_weights": self.bias_band_weights_current.astype(np.float64).tolist(),
            "lambda_bias_weight": float(self.lambda_bias_weight),
        }


def select_z2_active_points(
    *,
    config: Z2Config,
    oracle: DirectCDMOracle,
    seed_theta_raw: np.ndarray,
    probe_theta_raw: np.ndarray,
    pool_theta_raw: np.ndarray,
    k_bins: np.ndarray,
    resume_selected_theta_raw: np.ndarray | None = None,
    resume_rounds: list[dict[str, Any]] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ActiveSelectionResult:
    target_total = int(config.active_learning.active_points)
    batch_size = int(config.active_learning.batch_size)
    candidate_source = str(config.active_learning.candidate_source)
    selected_source_indices: list[int] = []
    selected_theta_rows: list[np.ndarray] = []
    remaining_indices = np.arange(pool_theta_raw.shape[0], dtype=np.int64)
    dynamic_weights = DynamicWeightState.from_config(config)
    resumed = np.asarray(resume_selected_theta_raw, dtype=np.float64) if resume_selected_theta_raw is not None else None
    if resumed is not None and resumed.size:
        if resumed.ndim != 2 or resumed.shape[1] != seed_theta_raw.shape[1]:
            raise ValueError(
                "resume_selected_theta_raw must have shape "
                f"(n, {seed_theta_raw.shape[1]}), got {resumed.shape}."
            )
        selected_theta_rows.extend(row.copy() for row in resumed)
        selected_source_indices.extend([-3] * int(resumed.shape[0]))
        train_theta = np.vstack([np.asarray(seed_theta_raw, dtype=np.float64), resumed])
    else:
        train_theta = np.asarray(seed_theta_raw, dtype=np.float64)
    train_log = oracle.evaluate(train_theta, k_bins).log_pk
    probe_truth = oracle.evaluate(probe_theta_raw, k_bins).log_pk
    rounds: list[dict[str, Any]] = [dict(item) for item in (resume_rounds or [])]
    if rounds and len(rounds) != len(selected_theta_rows):
        raise ValueError(
            "resume_rounds and resume_selected_theta_raw must describe the same completed count, "
            f"got {len(rounds)} rounds vs {len(selected_theta_rows)} selected rows."
        )

    while len(selected_theta_rows) < target_total and (
        candidate_source in {"continuous", "m3"} or remaining_indices.size > 0
    ):
        round_config = dynamic_weights.round_config(config)
        round_index = len(rounds)
        remaining_needed = target_total - len(selected_theta_rows)
        this_batch = min(batch_size, remaining_needed)
        if candidate_source == "pool":
            this_batch = min(this_batch, remaining_indices.size)
        _emit_selection_progress(
            progress_callback,
            {
                "event": "active_round_started",
                "round_index": int(round_index),
                "selected_so_far": int(len(selected_theta_rows)),
                "target_total": int(target_total),
                "train_size": int(train_theta.shape[0]),
                "batch_size": int(this_batch),
                "candidate_source": candidate_source,
            },
        )
        emulator = PCAGPDirectCDMEmulator(
            round_config.parameter_space,
            round_config.model,
            target_kind=str(round_config.target.kind),
        ).fit(
            train_theta,
            train_log,
            k_bins,
        )
        if candidate_source == "m3":
            bias_model = (
                fit_csst_bias_model(
                    config=round_config,
                    train_theta_raw=train_theta,
                    k_bins=k_bins,
                )
                if round_config.fastmock_bias.enabled
                else None
            )
            def _m3_progress(stage: str, current: int, total: int) -> None:
                _emit_selection_progress(
                    progress_callback,
                    {
                        "event": "m3_progress",
                        "round_index": int(round_index),
                        "selected_so_far": int(len(selected_theta_rows)),
                        "target_total": int(target_total),
                        "stage": str(stage),
                        "current": int(current),
                        "total": int(total),
                        "train_size": int(train_theta.shape[0]),
                    },
                )

            m3_batch = select_m3_batch(
                config=round_config,
                emulator=emulator,
                train_theta_raw=train_theta,
                iteration_index=round_index,
                batch_size=this_batch,
                bias_model=bias_model,
                probe_theta_raw=probe_theta_raw,
                probe_truth_log=probe_truth,
                k_bins=k_bins,
                progress_callback=_m3_progress,
            )
            chosen_theta = np.asarray(m3_batch.selected_theta_raw, dtype=np.float64)
            chosen_global = np.full((chosen_theta.shape[0],), -2, dtype=np.int64)
            acquisition = np.asarray(m3_batch.selected_scores, dtype=np.float64)
            round_report = {
                    "round_index": int(round_index),
                    "candidate_source": candidate_source,
                    "selected_count": int(chosen_theta.shape[0]),
                    "training_points_after_round": int(train_theta.shape[0] + chosen_theta.shape[0]),
                    "candidate_pool_remaining": None,
                    "continuous_candidate_count": None,
                    "m3_selected_source_pc": np.asarray(m3_batch.selected_source_pc, dtype=np.int64).tolist(),
                    "m3_score_p50": float(np.percentile(acquisition, 50.0)),
                    "m3_score_p95": float(np.percentile(acquisition, 95.0)),
                    "m3_metadata": dict(m3_batch.metadata),
                    "selected_theta_raw": chosen_theta.astype(np.float64).tolist(),
                    "selected_theta_unit": round_config.parameter_space.normalize(chosen_theta).astype(np.float64).tolist(),
                    "selected_scores": acquisition.astype(np.float64).tolist(),
                }
            _emit_selection_progress(
                progress_callback,
                {
                    "event": "active_round_candidate_selected",
                    "round_index": int(round_index),
                    "selected_this_round": int(chosen_theta.shape[0]),
                    "selected_so_far": int(len(selected_theta_rows) + chosen_theta.shape[0]),
                    "target_total": int(target_total),
                    "training_points_after_round": int(train_theta.shape[0] + chosen_theta.shape[0]),
                    "candidate_source": candidate_source,
                    "round": round_report,
                },
            )
            chosen_log = oracle.evaluate(chosen_theta, k_bins).log_pk
            selected_source_indices.extend(int(item) for item in chosen_global.tolist())
            selected_theta_rows.extend(row.copy() for row in chosen_theta)
            train_theta = np.vstack([train_theta, chosen_theta])
            train_log = np.vstack([train_log, chosen_log])
            rounds.append(round_report)
            _emit_selection_progress(
                progress_callback,
                {
                    "event": "active_round_completed",
                    "round_index": int(round_index),
                    "selected_this_round": int(chosen_theta.shape[0]),
                    "selected_so_far": int(len(selected_theta_rows)),
                    "target_total": int(target_total),
                    "training_points_after_round": int(train_theta.shape[0]),
                    "candidate_source": candidate_source,
                    "round": round_report,
                },
            )
            continue

        probe_pred = emulator.predict(probe_theta_raw)
        probe_error = _sample_relative_error_percentile(
            truth_log=probe_truth,
            pred_log=probe_pred.log_pk_mean,
            percentile=round_config.active_learning.probe_error_percentile,
        )
        probe_features = _candidate_features(round_config, emulator, probe_theta_raw, train_theta, seed_theta_raw)
        calibrator = ProbeErrorCalibrator(round_config.active_learning.probe_error_percentile).fit(
            probe_features,
            probe_error,
        )

        if candidate_source == "continuous":
            candidate_theta = _continuous_candidate_source(
                config=round_config,
                emulator=emulator,
                calibrator=calibrator,
                probe_theta_raw=probe_theta_raw,
                probe_error=probe_error,
                train_theta_raw=train_theta,
                seed_theta_raw=seed_theta_raw,
                round_index=len(rounds),
            )
            candidate_indices = np.full((candidate_theta.shape[0],), -1, dtype=np.int64)
        else:
            candidate_indices = remaining_indices
            candidate_theta = pool_theta_raw[candidate_indices]

        features, calibrated, hotspot, alc_imse, acquisition, alc_state = _score_candidate_batch(
            config=round_config,
            emulator=emulator,
            calibrator=calibrator,
            candidate_theta_raw=candidate_theta,
            probe_theta_raw=probe_theta_raw,
            probe_error=probe_error,
            train_theta_raw=train_theta,
            seed_theta_raw=seed_theta_raw,
            context=None,
        )
        preselect_count = min(
            candidate_theta.shape[0],
            max(this_batch, int(round_config.active_learning.preselect_factor) * this_batch),
        )
        top_local = _boundary_aware_preselect(
            config=round_config,
            theta_raw=candidate_theta,
            scores=acquisition,
            count=preselect_count,
        )
        local_selected = _select_diverse_subset(
            config=round_config,
            candidate_theta_raw=candidate_theta[top_local],
            candidate_scores=acquisition[top_local],
            train_theta_raw=train_theta,
            probe_theta_raw=probe_theta_raw,
            batch_size=this_batch,
        )
        chosen_local = top_local[local_selected]
        chosen_global = candidate_indices[chosen_local]
        chosen_theta = pool_theta_raw[chosen_global]
        if candidate_source == "continuous":
            chosen_theta = candidate_theta[chosen_local]
        chosen_log = oracle.evaluate(chosen_theta, k_bins).log_pk
        selected_source_indices.extend(int(item) for item in chosen_global.tolist())
        selected_theta_rows.extend(row.copy() for row in chosen_theta)
        train_theta = np.vstack([train_theta, chosen_theta])
        train_log = np.vstack([train_log, chosen_log])
        if candidate_source == "pool":
            remaining_indices = np.setdiff1d(remaining_indices, chosen_global, assume_unique=False)
        rounds.append(
            {
                "round_index": int(round_index),
                "candidate_source": candidate_source,
                "selected_count": int(chosen_theta.shape[0]),
                "training_points_after_round": int(train_theta.shape[0]),
                "candidate_pool_remaining": int(remaining_indices.size) if candidate_source == "pool" else None,
                "continuous_candidate_count": int(candidate_theta.shape[0]) if candidate_source == "continuous" else None,
                "probe_error_p50": float(np.percentile(probe_error, 50.0)),
                "probe_error_p68": float(np.percentile(probe_error, 68.0)),
                "candidate_score_p95": float(np.percentile(acquisition, 95.0)),
                "alc_imse_weight": float(round_config.active_learning.alc_imse_weight),
                "alc_imse_enabled": bool(alc_state.enabled),
                "alc_imse_p50": float(np.percentile(alc_imse, 50.0)),
                "alc_imse_p95": float(np.percentile(alc_imse, 95.0)),
                "pca_band_weights": alc_state.band_weights.astype(np.float64).tolist(),
                "pca_weight_function": str(alc_state.weight_details.get("function", "unknown")),
                "pca_weight_curve_band_means": list(
                    alc_state.weight_details.get("k_weight_curve_band_means", [])
                ),
                "pca_component_weight_p50": float(np.percentile(alc_state.component_weights, 50.0)),
                "pca_component_weight_p95": float(np.percentile(alc_state.component_weights, 95.0)),
            }
        )
        _emit_selection_progress(
            progress_callback,
            {
                "event": "active_round_completed",
                "round_index": int(round_index),
                "selected_this_round": int(chosen_theta.shape[0]),
                "selected_so_far": int(len(selected_theta_rows)),
                "target_total": int(target_total),
                "training_points_after_round": int(train_theta.shape[0]),
                "candidate_source": candidate_source,
                "round": rounds[-1],
            },
        )

    selected = np.asarray(selected_source_indices, dtype=np.int64)
    selected_theta = (
        np.vstack(selected_theta_rows).astype(np.float64)
        if selected_theta_rows
        else np.empty((0, seed_theta_raw.shape[1]), dtype=np.float64)
    )
    dynamic_final_before_restore = dynamic_weights.snapshot()
    dynamic_weights.restore_defaults()
    dynamic_after_restore = dynamic_weights.snapshot()
    return ActiveSelectionResult(
        selected_theta_raw=selected_theta,
        selected_pool_indices=selected,
        report={
            "target_kind": str(config.target.kind),
            "selection_method": (
                "m3_continuous_posterior_selector"
                if candidate_source == "m3"
                else (
                    "weighted_alc_imse_probe_calibrated_coreset"
                    if float(config.active_learning.alc_imse_weight) > 0.0
                    else "probe_calibrated_reduction_embedding_coreset"
                )
            ),
            "candidate_source": candidate_source,
            "active_points_requested": target_total,
            "active_points_selected": int(selected_theta.shape[0]),
            "batch_size": batch_size,
            "continuous_initial_draws": int(config.active_learning.continuous_initial_draws),
            "continuous_restarts": int(config.active_learning.continuous_restarts),
            "rounds": rounds,
            "uses_lofi": False,
            "uses_anchor": str(config.target.kind).strip().lower() == "cdm_logdiff",
            "fastmock_bias": {
                "enabled": bool(config.fastmock_bias.enabled),
                "provider": str(config.fastmock_bias.provider),
                "normalization": str(config.fastmock_bias.normalization),
                "bias_weight": float(config.fastmock_bias.bias_weight),
                "bias_band_weights": _default_bias_band_weights(config).astype(np.float64).tolist(),
            },
            "dynamic_weight_state": {
                "enabled": True,
                "lambda_bias_weight_dynamic": False,
                "lambda_bias_weight": float(config.fastmock_bias.bias_weight),
                "variance_band_weights_default": dynamic_weights.variance_band_weights_default.astype(np.float64).tolist(),
                "bias_band_weights_default": dynamic_weights.bias_band_weights_default.astype(np.float64).tolist(),
                "final_before_restore": dict(dynamic_final_before_restore),
                "after_restore": dict(dynamic_after_restore),
                "restored_after_completion": bool(
                    np.allclose(
                        np.asarray(dynamic_after_restore["variance_band_weights"], dtype=np.float64),
                        dynamic_weights.variance_band_weights_default,
                    )
                    and np.allclose(
                        np.asarray(dynamic_after_restore["bias_band_weights"], dtype=np.float64),
                        dynamic_weights.bias_band_weights_default,
                    )
                ),
                "bias_band_weights_applied_to_score": True,
            },
        },
    )


def _emit_selection_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _sample_relative_error_percentile(*, truth_log: np.ndarray, pred_log: np.ndarray, percentile: float) -> np.ndarray:
    relative = np.abs(np.exp(np.asarray(pred_log) - np.asarray(truth_log)) - 1.0)
    return np.percentile(relative, float(percentile), axis=1).astype(np.float64)


def _candidate_features(
    config: Z2Config,
    emulator: PCAGPDirectCDMEmulator,
    theta_raw: np.ndarray,
    train_theta_raw: np.ndarray,
    seed_theta_raw: np.ndarray,
) -> np.ndarray:
    theta_unit = config.parameter_space.normalize(theta_raw)
    train_unit = config.parameter_space.normalize(train_theta_raw)
    seed_unit = config.parameter_space.normalize(seed_theta_raw)
    uncertainty = emulator.uncertainty_scalar(theta_raw)
    d_train = nearest_distance(theta_unit, train_unit)
    d_seed = nearest_distance(theta_unit, seed_unit)
    d_boundary = boundary_distance(theta_unit)
    return np.column_stack(
        [
            np.log1p(np.maximum(uncertainty, 0.0)),
            d_train,
            d_seed,
            d_boundary,
            1.0 / np.maximum(d_boundary, 1.0e-6),
        ]
    ).astype(np.float64)


def _combine_acquisition_score(
    config: Z2Config,
    features: np.ndarray,
    calibrated_error: np.ndarray,
    hotspot_score: np.ndarray,
    alc_imse_score: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    calibrated = np.asarray(calibrated_error, dtype=np.float64)
    uncertainty = _robust_unit(x[:, 0])
    d_train = _robust_unit(x[:, 1])
    boundary_penalty = _robust_unit(x[:, 4])
    cal = _robust_unit(calibrated)
    hotspot = _robust_unit(hotspot_score)
    alc_imse = _robust_unit(alc_imse_score)
    return (
        cal
        + config.active_learning.probe_hotspot_weight * hotspot
        + config.active_learning.alc_imse_weight * alc_imse
        + config.active_learning.uncertainty_weight * uncertainty
        + config.active_learning.train_distance_weight * d_train
        - config.active_learning.boundary_risk_weight * boundary_penalty
    ).astype(np.float64)


def _score_candidate_batch(
    *,
    config: Z2Config,
    emulator: PCAGPDirectCDMEmulator,
    calibrator: ProbeErrorCalibrator,
    candidate_theta_raw: np.ndarray,
    probe_theta_raw: np.ndarray,
    probe_error: np.ndarray,
    train_theta_raw: np.ndarray,
    seed_theta_raw: np.ndarray,
    context: ScoreContext | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ALCIMSEState]:
    features = _candidate_features(config, emulator, candidate_theta_raw, train_theta_raw, seed_theta_raw)
    calibrated = calibrator.predict(features)
    hotspot = _probe_hotspot_score(
        config=config,
        candidate_theta_raw=candidate_theta_raw,
        probe_theta_raw=probe_theta_raw,
        probe_error=probe_error,
    )
    alc_state = context.alc_state if context is not None else None
    if alc_state is None:
        alc_state = _build_alc_imse_state(
            config=config,
            emulator=emulator,
            probe_theta_raw=probe_theta_raw,
            probe_error=probe_error,
        )
    alc_imse = _weighted_alc_imse_score(
        config=config,
        candidate_theta_raw=candidate_theta_raw,
        state=alc_state,
    )
    if context is None:
        acquisition = _combine_acquisition_score(config, features, calibrated, hotspot, alc_imse)
    else:
        acquisition = _combine_acquisition_score_with_context(config, features, calibrated, hotspot, alc_imse, context)
    return features, calibrated, hotspot, alc_imse, acquisition, alc_state


def _fit_score_context(
    features: np.ndarray,
    calibrated: np.ndarray,
    hotspot: np.ndarray,
    alc_imse: np.ndarray,
    alc_state: ALCIMSEState | None,
) -> ScoreContext:
    x = np.asarray(features, dtype=np.float64)
    return ScoreContext(
        uncertainty_bounds=_percentile_bounds(x[:, 0]),
        train_distance_bounds=_percentile_bounds(x[:, 1]),
        boundary_penalty_bounds=_percentile_bounds(x[:, 4]),
        calibrated_bounds=_percentile_bounds(calibrated),
        hotspot_bounds=_percentile_bounds(hotspot),
        alc_imse_bounds=_percentile_bounds(alc_imse),
        alc_state=alc_state,
    )


def _combine_acquisition_score_with_context(
    config: Z2Config,
    features: np.ndarray,
    calibrated_error: np.ndarray,
    hotspot_score: np.ndarray,
    alc_imse_score: np.ndarray,
    context: ScoreContext,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return (
        _unit_from_bounds(calibrated_error, context.calibrated_bounds)
        + config.active_learning.probe_hotspot_weight
        * _unit_from_bounds(hotspot_score, context.hotspot_bounds)
        + config.active_learning.alc_imse_weight
        * _unit_from_bounds(alc_imse_score, context.alc_imse_bounds)
        + config.active_learning.uncertainty_weight
        * _unit_from_bounds(x[:, 0], context.uncertainty_bounds)
        + config.active_learning.train_distance_weight
        * _unit_from_bounds(x[:, 1], context.train_distance_bounds)
        - config.active_learning.boundary_risk_weight
        * _unit_from_bounds(x[:, 4], context.boundary_penalty_bounds)
    ).astype(np.float64)


def _continuous_candidate_source(
    *,
    config: Z2Config,
    emulator: PCAGPDirectCDMEmulator,
    calibrator: ProbeErrorCalibrator,
    probe_theta_raw: np.ndarray,
    probe_error: np.ndarray,
    train_theta_raw: np.ndarray,
    seed_theta_raw: np.ndarray,
    round_index: int,
) -> np.ndarray:
    dim = config.parameter_space.dim
    draw_count = int(config.active_learning.continuous_initial_draws)
    seed = int(config.random_seed + 7919 + 101 * int(round_index))
    base_unit = sobol_unit(draw_count, dim, seed=seed)
    jitter_unit = _draw_hotspot_jitter_units(
        config=config,
        probe_theta_raw=probe_theta_raw,
        probe_error=probe_error,
        seed=seed + 17,
    )
    candidate_unit = np.vstack([base_unit, jitter_unit]) if jitter_unit.size else base_unit
    candidate_unit = unique_unit_rows(candidate_unit, decimals=config.splits.duplicate_decimals)
    candidate_unit = _filter_against_training_units(config, candidate_unit, train_theta_raw)
    if candidate_unit.shape[0] == 0:
        candidate_unit = base_unit[: max(1, min(draw_count, int(config.active_learning.continuous_restarts)))]

    candidate_theta = config.parameter_space.denormalize(candidate_unit)
    features, calibrated, hotspot, alc_imse, acquisition, alc_state = _score_candidate_batch(
        config=config,
        emulator=emulator,
        calibrator=calibrator,
        candidate_theta_raw=candidate_theta,
        probe_theta_raw=probe_theta_raw,
        probe_error=probe_error,
        train_theta_raw=train_theta_raw,
        seed_theta_raw=seed_theta_raw,
        context=None,
    )
    context = _fit_score_context(features, calibrated, hotspot, alc_imse, alc_state)
    restart_count = min(int(config.active_learning.continuous_restarts), candidate_theta.shape[0])
    restart_indices = _boundary_aware_preselect(
        config=config,
        theta_raw=candidate_theta,
        scores=acquisition,
        count=restart_count,
    )

    optimized_units: list[np.ndarray] = []
    bounds = [(0.0, 1.0)] * dim
    for start_unit in candidate_unit[restart_indices]:
        result = minimize(
            lambda unit: -float(
                _continuous_score_unit(
                    config=config,
                    emulator=emulator,
                    calibrator=calibrator,
                    unit_theta=np.asarray(unit, dtype=np.float64),
                    probe_theta_raw=probe_theta_raw,
                    probe_error=probe_error,
                    train_theta_raw=train_theta_raw,
                    seed_theta_raw=seed_theta_raw,
                    context=context,
                )
            ),
            np.asarray(start_unit, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(config.active_learning.continuous_local_maxiter), "ftol": 1.0e-6},
        )
        optimized_units.append(np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0))

    keep_count = min(candidate_unit.shape[0], max(restart_count * 4, int(config.active_learning.batch_size) * 8))
    top_indices = _boundary_aware_preselect(
        config=config,
        theta_raw=candidate_theta,
        scores=acquisition,
        count=keep_count,
    )
    combined = np.vstack([np.vstack(optimized_units), candidate_unit[top_indices]])
    combined = unique_unit_rows(combined, decimals=config.splits.duplicate_decimals)
    combined = _filter_against_training_units(config, combined, train_theta_raw)
    if combined.shape[0] == 0:
        combined = candidate_unit[top_indices]
    return config.parameter_space.denormalize(combined)


def _continuous_score_unit(
    *,
    config: Z2Config,
    emulator: PCAGPDirectCDMEmulator,
    calibrator: ProbeErrorCalibrator,
    unit_theta: np.ndarray,
    probe_theta_raw: np.ndarray,
    probe_error: np.ndarray,
    train_theta_raw: np.ndarray,
    seed_theta_raw: np.ndarray,
    context: ScoreContext,
) -> float:
    theta_raw = config.parameter_space.denormalize(np.asarray(unit_theta, dtype=np.float64).reshape(1, -1))
    _, _, _, _, score, _ = _score_candidate_batch(
        config=config,
        emulator=emulator,
        calibrator=calibrator,
        candidate_theta_raw=theta_raw,
        probe_theta_raw=probe_theta_raw,
        probe_error=probe_error,
        train_theta_raw=train_theta_raw,
        seed_theta_raw=seed_theta_raw,
        context=context,
    )
    return float(score[0])


def _draw_hotspot_jitter_units(
    *,
    config: Z2Config,
    probe_theta_raw: np.ndarray,
    probe_error: np.ndarray,
    seed: int,
) -> np.ndarray:
    count = int(config.active_learning.continuous_hotspot_jitter_draws)
    if count <= 0:
        return np.empty((0, config.parameter_space.dim), dtype=np.float64)
    probe_unit = config.parameter_space.normalize(probe_theta_raw)
    errors = np.asarray(probe_error, dtype=np.float64).reshape(-1)
    threshold = float(np.percentile(errors, float(config.active_learning.probe_hotspot_percentile)))
    hot = probe_unit[errors >= threshold]
    if hot.shape[0] == 0:
        hot = probe_unit[[int(np.argmax(errors))]]
    rng = np.random.default_rng(int(seed))
    centers = hot[rng.integers(0, hot.shape[0], size=count)]
    scale = max(float(config.active_learning.reduction_length_scale), 1.0e-6)
    jitter = rng.normal(loc=0.0, scale=scale, size=(count, config.parameter_space.dim))
    return np.clip(centers + jitter, 0.0, 1.0).astype(np.float64)


def _filter_against_training_units(
    config: Z2Config,
    candidate_unit: np.ndarray,
    train_theta_raw: np.ndarray,
) -> np.ndarray:
    unit = np.asarray(candidate_unit, dtype=np.float64)
    if unit.ndim != 2 or unit.shape[0] == 0:
        return np.empty((0, config.parameter_space.dim), dtype=np.float64)
    train_unit = config.parameter_space.normalize(train_theta_raw)
    dist = nearest_distance(unit, train_unit)
    threshold = float(config.active_learning.continuous_duplicate_distance_threshold)
    return unit[dist > threshold].astype(np.float64)


def _probe_hotspot_score(
    *,
    config: Z2Config,
    candidate_theta_raw: np.ndarray,
    probe_theta_raw: np.ndarray,
    probe_error: np.ndarray,
) -> np.ndarray:
    candidate_unit = config.parameter_space.normalize(candidate_theta_raw)
    probe_unit = config.parameter_space.normalize(probe_theta_raw)
    errors = np.asarray(probe_error, dtype=np.float64).reshape(-1)
    if probe_unit.shape[0] == 0 or errors.size != probe_unit.shape[0]:
        return np.zeros((candidate_unit.shape[0],), dtype=np.float64)
    threshold = float(np.percentile(errors, float(config.active_learning.probe_hotspot_percentile)))
    hot_mask = errors >= threshold
    if not np.any(hot_mask):
        hot_mask[np.argmax(errors)] = True
    hot_probe = probe_unit[hot_mask]
    hot_weights = _robust_unit(errors[hot_mask]) + 1.0e-3
    length = max(float(config.active_learning.reduction_length_scale), 1.0e-6)
    scores = np.zeros((candidate_unit.shape[0],), dtype=np.float64)
    chunk = 2048
    for start in range(0, candidate_unit.shape[0], chunk):
        block = candidate_unit[start : start + chunk]
        dist2 = np.sum((block[:, None, :] - hot_probe[None, :, :]) ** 2, axis=2)
        sim = np.exp(-0.5 * dist2 / (length * length))
        scores[start : start + chunk] = sim @ hot_weights / max(float(np.sum(hot_weights)), 1.0e-12)
    return scores


def _build_alc_imse_state(
    *,
    config: Z2Config,
    emulator: PCAGPDirectCDMEmulator,
    probe_theta_raw: np.ndarray,
    probe_error: np.ndarray,
) -> ALCIMSEState:
    band_weights = _pca_band_weights(config)
    component_weights, weight_details = _pca_component_weight_details(config, emulator)
    disabled = ALCIMSEState(
        enabled=False,
        probe_unit=np.empty((0, config.parameter_space.dim), dtype=np.float64),
        probe_weights=np.empty((0,), dtype=np.float64),
        band_weights=band_weights,
        component_weights=component_weights,
        weight_details=weight_details,
        component_gain=np.empty((0,), dtype=np.float64),
        gp_models=(),
        v_probe=(),
        target_variance_scale=np.empty((0,), dtype=np.float64),
        observation_noise=np.empty((0,), dtype=np.float64),
    )
    if float(config.active_learning.alc_imse_weight) <= 0.0:
        return disabled
    if emulator.gp_models is None or emulator.score_std is None or emulator.pca is None:
        return disabled
    probe_unit = config.parameter_space.normalize(probe_theta_raw)
    if probe_unit.shape[0] == 0:
        return disabled
    gp_models = tuple(emulator.gp_models)
    score_std = np.asarray(emulator.score_std, dtype=np.float64).reshape(-1)
    n_components = min(len(gp_models), score_std.size, component_weights.size)
    if n_components <= 0:
        return disabled
    probe_weights = _probe_imse_weights(config, probe_error, probe_unit.shape[0])
    kept_gps: list[Any] = []
    kept_v_probe: list[np.ndarray] = []
    kept_target_scale: list[float] = []
    kept_noise: list[float] = []
    kept_gain: list[float] = []
    for index, gp in enumerate(gp_models[:n_components]):
        try:
            x_train = np.asarray(gp.X_train_, dtype=np.float64)
            k_train_probe = np.asarray(gp.kernel_(x_train, probe_unit), dtype=np.float64)
            v_probe = np.linalg.solve(np.asarray(gp.L_, dtype=np.float64), k_train_probe)
        except Exception:
            continue
        target_scale = _gp_target_variance_scale(gp)
        kept_gps.append(gp)
        kept_v_probe.append(v_probe.astype(np.float64))
        kept_target_scale.append(target_scale)
        kept_noise.append(_gp_observation_noise(gp, target_scale))
        kept_gain.append(
            float(max(component_weights[index], 0.0))
            * float(max(score_std[index], 0.0) ** 2)
        )
    if not kept_gps:
        return disabled
    return ALCIMSEState(
        enabled=True,
        probe_unit=probe_unit.astype(np.float64),
        probe_weights=probe_weights.astype(np.float64),
        band_weights=band_weights.astype(np.float64),
        component_weights=component_weights.astype(np.float64),
        weight_details=weight_details,
        component_gain=np.asarray(kept_gain, dtype=np.float64),
        gp_models=tuple(kept_gps),
        v_probe=tuple(kept_v_probe),
        target_variance_scale=np.asarray(kept_target_scale, dtype=np.float64),
        observation_noise=np.asarray(kept_noise, dtype=np.float64),
    )


def _weighted_alc_imse_score(
    *,
    config: Z2Config,
    candidate_theta_raw: np.ndarray,
    state: ALCIMSEState,
) -> np.ndarray:
    candidate = np.asarray(candidate_theta_raw, dtype=np.float64)
    if candidate.ndim == 1:
        candidate = candidate.reshape(1, -1)
    if candidate.shape[0] == 0 or not state.enabled:
        return np.zeros((candidate.shape[0],), dtype=np.float64)
    candidate_unit = config.parameter_space.normalize(candidate)
    scores = np.zeros((candidate_unit.shape[0],), dtype=np.float64)
    for index, gp in enumerate(state.gp_models):
        reduction = _posterior_variance_reduction(
            gp=gp,
            candidate_unit=candidate_unit,
            probe_unit=state.probe_unit,
            probe_weights=state.probe_weights,
            v_probe=state.v_probe[index],
            target_variance_scale=float(state.target_variance_scale[index]),
            observation_noise=float(state.observation_noise[index]),
        )
        scores += float(state.component_gain[index]) * reduction
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def _posterior_variance_reduction(
    *,
    gp: Any,
    candidate_unit: np.ndarray,
    probe_unit: np.ndarray,
    probe_weights: np.ndarray,
    v_probe: np.ndarray,
    target_variance_scale: float,
    observation_noise: float,
) -> np.ndarray:
    x_train = np.asarray(gp.X_train_, dtype=np.float64)
    out = np.zeros((candidate_unit.shape[0],), dtype=np.float64)
    chunk = 512
    for start in range(0, candidate_unit.shape[0], chunk):
        block = candidate_unit[start : start + chunk]
        k_train_candidate = np.asarray(gp.kernel_(x_train, block), dtype=np.float64)
        v_candidate = np.linalg.solve(np.asarray(gp.L_, dtype=np.float64), k_train_candidate)
        k_probe_candidate = np.asarray(gp.kernel_(probe_unit, block), dtype=np.float64)
        posterior_cov = float(target_variance_scale) * (
            k_probe_candidate - np.asarray(v_probe, dtype=np.float64).T @ v_candidate
        )
        _, std = gp.predict(block, return_std=True)
        variance = np.maximum(np.asarray(std, dtype=np.float64).reshape(-1) ** 2, 0.0)
        denom = np.maximum(variance + float(observation_noise), 1.0e-18)
        out[start : start + chunk] = np.sum(
            probe_weights.reshape(-1, 1) * posterior_cov**2 / denom.reshape(1, -1),
            axis=0,
        )
    return np.maximum(out, 0.0).astype(np.float64)


def _probe_imse_weights(config: Z2Config, probe_error: np.ndarray, probe_count: int) -> np.ndarray:
    errors = np.asarray(probe_error, dtype=np.float64).reshape(-1)
    if probe_count <= 0:
        return np.empty((0,), dtype=np.float64)
    if errors.size != int(probe_count) or not np.any(np.isfinite(errors)):
        return np.full((int(probe_count),), 1.0 / float(probe_count), dtype=np.float64)
    errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
    floor = max(float(config.active_learning.alc_probe_weight_floor), 0.0)
    weights = floor + _robust_unit(errors)
    threshold = float(np.percentile(errors, float(config.active_learning.probe_hotspot_percentile)))
    hot_mask = errors >= threshold
    if np.any(hot_mask):
        weights[hot_mask] *= 1.0 + max(float(config.active_learning.probe_hotspot_weight), 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full((int(probe_count),), 1.0 / float(probe_count), dtype=np.float64)
    return (weights / total).astype(np.float64)


def _pca_component_weights(config: Z2Config, emulator: PCAGPDirectCDMEmulator) -> np.ndarray:
    weights, _ = _pca_component_weight_details(config, emulator)
    return weights


def _pca_component_weight_details(
    config: Z2Config,
    emulator: PCAGPDirectCDMEmulator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if emulator.pca is None:
        return np.ones((0,), dtype=np.float64), {"function": "unavailable", "reason": "missing_pca"}
    pca_components = np.asarray(emulator.pca.components_, dtype=np.float64)
    if pca_components.ndim != 2 or pca_components.shape[0] == 0:
        return np.ones((0,), dtype=np.float64), {"function": "unavailable", "reason": "empty_pca_components"}
    sensitivity, band_integrals, total_integral = _pca_band_sensitivity_details(config, emulator)
    band_weights = _pca_band_weights(config)
    function = str(config.active_learning.pca_weight_function).strip().lower()
    details: dict[str, Any] = {
        "function": function,
        "band_labels": [str(band.name) for band in config.k_grid.bands],
        "band_weights": band_weights.astype(np.float64).tolist(),
        "band_sensitivity": sensitivity.astype(np.float64).tolist(),
        "band_variance_integrals": band_integrals.astype(np.float64).tolist(),
        "component_unweighted_integral": total_integral.astype(np.float64).tolist(),
    }
    if sensitivity.shape[1] != band_weights.size:
        raw_weights = np.ones((pca_components.shape[0],), dtype=np.float64)
        details["fallback_reason"] = "band_weight_shape_mismatch"
    elif function == "band_integral":
        weights = sensitivity @ band_weights.reshape(-1, 1)
        raw_weights = weights.reshape(-1)
    elif function == "smooth_logk_curve":
        k_bins = np.asarray(emulator.k_bins, dtype=np.float64).reshape(-1)
        curve, curve_band_means = _smooth_logk_weight_curve(config, k_bins, band_weights)
        integration_weights = _logk_trapezoid_weights(k_bins)
        numerator = np.sum(
            np.square(pca_components) * integration_weights.reshape(1, -1) * curve.reshape(1, -1),
            axis=1,
        )
        raw_weights = numerator / np.maximum(total_integral, 1.0e-30)
        details["pca_weight_transition_dex"] = float(config.active_learning.pca_weight_transition_dex)
        details["k_weight_curve_band_means"] = curve_band_means.astype(np.float64).tolist()
    else:
        raw_weights = np.ones((pca_components.shape[0],), dtype=np.float64)
        details["fallback_reason"] = f"unsupported_weight_function:{function}"
    weights = np.nan_to_num(raw_weights, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float64)
    if bool(config.active_learning.pca_component_weight_normalize):
        mean = float(np.mean(weights))
        if mean > 0.0 and np.isfinite(mean):
            weights = weights / mean
            details["normalized_by_mean"] = mean
    details["raw_component_weights"] = raw_weights.astype(np.float64).tolist()
    weights = np.clip(
        weights,
        float(config.active_learning.pca_component_weight_min),
        float(config.active_learning.pca_component_weight_max),
    )
    details["component_weight_min"] = float(np.min(weights)) if weights.size else 0.0
    details["component_weight_p50"] = float(np.percentile(weights, 50.0)) if weights.size else 0.0
    details["component_weight_max"] = float(np.max(weights)) if weights.size else 0.0
    return weights.astype(np.float64), details


def _pca_band_sensitivity(config: Z2Config, emulator: PCAGPDirectCDMEmulator) -> np.ndarray:
    sensitivity, _, _ = _pca_band_sensitivity_details(config, emulator)
    return sensitivity


def _pca_band_sensitivity_details(
    config: Z2Config,
    emulator: PCAGPDirectCDMEmulator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if emulator.pca is None or emulator.k_bins is None:
        empty = np.empty((0, 0), dtype=np.float64)
        return empty, empty, np.empty((0,), dtype=np.float64)
    components = np.asarray(emulator.pca.components_, dtype=np.float64)
    k_bins = np.asarray(emulator.k_bins, dtype=np.float64).reshape(-1)
    squared = components**2
    integration_weights = _logk_trapezoid_weights(k_bins)
    rows: list[np.ndarray] = []
    for index, band in enumerate(config.k_grid.bands):
        if index == len(config.k_grid.bands) - 1:
            mask = (k_bins >= float(band.k_min)) & (k_bins <= float(band.k_max))
        else:
            mask = (k_bins >= float(band.k_min)) & (k_bins < float(band.k_max))
        if np.any(mask):
            rows.append(np.sum(squared[:, mask] * integration_weights[mask].reshape(1, -1), axis=1))
        else:
            rows.append(np.zeros((squared.shape[0],), dtype=np.float64))
    if not rows:
        integrals = np.ones((components.shape[0], 1), dtype=np.float64)
        return integrals, integrals, np.ones((components.shape[0],), dtype=np.float64)
    integrals = np.vstack(rows).T.astype(np.float64)
    total = np.maximum(np.sum(integrals, axis=1), 1.0e-30)
    return integrals / total.reshape(-1, 1), integrals, total


def _logk_trapezoid_weights(k_bins: np.ndarray) -> np.ndarray:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    if k_arr.ndim != 1 or k_arr.size < 2:
        raise ValueError("k_bins must be a 1D array with at least two points.")
    logk = np.log10(np.maximum(k_arr, 1.0e-30))
    weights = np.empty_like(logk, dtype=np.float64)
    weights[0] = 0.5 * (logk[1] - logk[0])
    weights[-1] = 0.5 * (logk[-1] - logk[-2])
    if logk.size > 2:
        weights[1:-1] = 0.5 * (logk[2:] - logk[:-2])
    return np.maximum(weights, 0.0).astype(np.float64)


def _smooth_logk_weight_curve(
    config: Z2Config,
    k_bins: np.ndarray,
    band_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    weights = np.asarray(band_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != len(config.k_grid.bands):
        return np.ones_like(k_arr, dtype=np.float64), np.ones((len(config.k_grid.bands),), dtype=np.float64)
    logk = np.log10(np.maximum(k_arr, 1.0e-30))
    width = float(max(config.active_learning.pca_weight_transition_dex, 1.0e-4))
    curve = np.full_like(k_arr, max(float(weights[0]), 1.0e-12), dtype=np.float64)
    for band_index, band in enumerate(config.k_grid.bands[:-1]):
        boundary = np.log10(float(band.k_max))
        next_level = max(float(weights[band_index + 1]), 1.0e-12)
        smooth_step = 0.5 * (1.0 + np.tanh((logk - boundary) / width))
        curve = (1.0 - smooth_step) * curve + smooth_step * next_level
    integration_weights = _logk_trapezoid_weights(k_arr)
    curve_mean = float(
        np.sum(curve * integration_weights) / max(np.sum(integration_weights), 1.0e-30)
    )
    curve = curve / max(curve_mean, 1.0e-30)
    band_means: list[float] = []
    for index, band in enumerate(config.k_grid.bands):
        if index == len(config.k_grid.bands) - 1:
            mask = (k_arr >= float(band.k_min)) & (k_arr <= float(band.k_max))
        else:
            mask = (k_arr >= float(band.k_min)) & (k_arr < float(band.k_max))
        if not np.any(mask):
            band_means.append(0.0)
            continue
        local_weights = integration_weights[mask]
        band_means.append(
            float(np.sum(curve[mask] * local_weights) / max(np.sum(local_weights), 1.0e-30))
        )
    return curve.astype(np.float64), np.asarray(band_means, dtype=np.float64)


def _pca_band_weights(config: Z2Config) -> np.ndarray:
    configured = tuple(float(value) for value in config.active_learning.pca_band_weights)
    if configured:
        return np.asarray(configured, dtype=np.float64)
    return np.ones((len(config.k_grid.bands),), dtype=np.float64)


def _default_bias_band_weights(config: Z2Config) -> np.ndarray:
    configured = tuple(float(value) for value in config.fastmock_bias.bias_band_weights)
    if configured:
        return np.asarray(configured, dtype=np.float64)
    return np.ones((len(config.k_grid.bands),), dtype=np.float64)


def _coerce_dynamic_weights(values: np.ndarray, *, expected_size: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (int(expected_size),):
        raise ValueError(f"{name} must have shape ({int(expected_size)},), got {arr.shape}.")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain finite positive values.")
    return arr.astype(np.float64)


def _gp_target_variance_scale(gp: Any) -> float:
    y_std = np.asarray(getattr(gp, "_y_train_std", 1.0), dtype=np.float64).reshape(-1)
    scale = float(y_std[0]) if y_std.size else 1.0
    return max(scale * scale, 1.0e-24)


def _gp_observation_noise(gp: Any, target_variance_scale: float) -> float:
    alpha = np.asarray(getattr(gp, "alpha", 1.0e-10), dtype=np.float64).reshape(-1)
    if alpha.size == 0:
        return 1.0e-10 * float(target_variance_scale)
    return max(float(np.mean(alpha)), 0.0) * float(target_variance_scale)


def _boundary_aware_preselect(
    *,
    config: Z2Config,
    theta_raw: np.ndarray,
    scores: np.ndarray,
    count: int,
) -> np.ndarray:
    theta_unit = config.parameter_space.normalize(theta_raw)
    is_boundary = boundary_distance(theta_unit) < float(config.active_learning.boundary_guard_threshold)
    boundary_cap = int(np.floor(float(config.active_learning.boundary_fraction_cap) * int(count)))
    boundary_cap = max(0, min(boundary_cap, int(count)))
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    selected: list[int] = []
    boundary_count = 0
    deferred_boundary: list[int] = []
    for idx in order:
        idx_int = int(idx)
        if is_boundary[idx_int]:
            if boundary_count < boundary_cap:
                selected.append(idx_int)
                boundary_count += 1
            else:
                deferred_boundary.append(idx_int)
        else:
            selected.append(idx_int)
        if len(selected) >= int(count):
            break
    if len(selected) < int(count):
        for idx in deferred_boundary:
            selected.append(idx)
            if len(selected) >= int(count):
                break
    return np.asarray(selected[: int(count)], dtype=np.int64)


def _select_diverse_subset(
    *,
    config: Z2Config,
    candidate_theta_raw: np.ndarray,
    candidate_scores: np.ndarray,
    train_theta_raw: np.ndarray,
    probe_theta_raw: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if candidate_theta_raw.shape[0] <= int(batch_size):
        return np.arange(candidate_theta_raw.shape[0], dtype=np.int64)
    z = _reduction_embedding(config, candidate_theta_raw, candidate_scores, train_theta_raw, probe_theta_raw)
    scores = _robust_unit(candidate_scores)
    theta_unit = config.parameter_space.normalize(candidate_theta_raw)
    is_boundary = boundary_distance(theta_unit) < float(config.active_learning.boundary_guard_threshold)
    boundary_cap = int(np.floor(float(config.active_learning.boundary_fraction_cap) * int(batch_size)))
    boundary_count = 0
    selected: list[int] = []
    available = set(range(candidate_theta_raw.shape[0]))

    first_order = np.argsort(scores)[::-1]
    for idx in first_order:
        if is_boundary[idx] and boundary_count >= boundary_cap:
            continue
        selected.append(int(idx))
        available.remove(int(idx))
        boundary_count += int(is_boundary[idx])
        break
    if not selected:
        selected.append(int(first_order[0]))
        available.remove(int(first_order[0]))
        boundary_count += int(is_boundary[int(first_order[0])])

    while len(selected) < int(batch_size) and available:
        chosen_z = z[np.asarray(selected, dtype=np.int64)]
        candidate_list = np.asarray(sorted(available), dtype=np.int64)
        dist = _min_distance(z[candidate_list], chosen_z)
        combined = (
            (1.0 - float(config.active_learning.diversity_weight)) * scores[candidate_list]
            + float(config.active_learning.diversity_weight) * _robust_unit(dist)
        )
        order = candidate_list[np.argsort(combined)[::-1]]
        chosen = None
        for idx in order:
            if is_boundary[idx] and boundary_count >= boundary_cap:
                continue
            chosen = int(idx)
            break
        if chosen is None:
            chosen = int(order[0])
        selected.append(chosen)
        available.remove(chosen)
        boundary_count += int(is_boundary[chosen])
    return np.asarray(selected, dtype=np.int64)


def _reduction_embedding(
    config: Z2Config,
    candidate_theta_raw: np.ndarray,
    candidate_scores: np.ndarray,
    train_theta_raw: np.ndarray,
    probe_theta_raw: np.ndarray,
) -> np.ndarray:
    candidate_unit = config.parameter_space.normalize(candidate_theta_raw)
    train_unit = config.parameter_space.normalize(train_theta_raw)
    probe_unit = config.parameter_space.normalize(probe_theta_raw)
    anchor_count = min(int(config.active_learning.reduction_probe_anchors), probe_unit.shape[0])
    anchors = _farthest_subset(probe_unit, anchor_count) if anchor_count > 0 else np.empty((0, candidate_unit.shape[1]))
    if anchors.shape[0] > 0:
        length = max(float(config.active_learning.reduction_length_scale), 1.0e-6)
        dist2 = np.sum((candidate_unit[:, None, :] - anchors[None, :, :]) ** 2, axis=2)
        reduction = np.exp(-0.5 * dist2 / (length * length)) * _robust_unit(candidate_scores).reshape(-1, 1)
    else:
        reduction = np.empty((candidate_unit.shape[0], 0), dtype=np.float64)
    d_train = nearest_distance(candidate_unit, train_unit).reshape(-1, 1)
    d_boundary = boundary_distance(candidate_unit).reshape(-1, 1)
    score_col = _robust_unit(candidate_scores).reshape(-1, 1)
    z = np.hstack([reduction, candidate_unit, score_col, d_train, d_boundary])
    mean = np.mean(z, axis=0, keepdims=True)
    std = np.maximum(np.std(z, axis=0, keepdims=True), 1.0e-12)
    return (z - mean) / std


def _farthest_subset(points: np.ndarray, count: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if count <= 0 or arr.shape[0] == 0:
        return np.empty((0, arr.shape[1]), dtype=np.float64)
    selected = [0]
    while len(selected) < min(int(count), arr.shape[0]):
        dist = _min_distance(arr, arr[np.asarray(selected)])
        selected.append(int(np.argmax(dist)))
    return arr[np.asarray(selected, dtype=np.int64)]


def _min_distance(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.shape[0] == 0:
        return np.full((x.shape[0],), np.inf, dtype=np.float64)
    dist2 = np.sum((x[:, None, :] - reference[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(dist2, axis=1))


def _robust_unit(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    lo = float(np.percentile(arr, 5.0))
    hi = float(np.percentile(arr, 95.0))
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _percentile_bounds(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    lo = float(np.percentile(arr, 5.0))
    hi = float(np.percentile(arr, 95.0))
    return lo, hi


def _unit_from_bounds(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
