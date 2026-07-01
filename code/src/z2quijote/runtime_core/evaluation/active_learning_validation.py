"""Validation helpers for active-learning and fixed-budget emulators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from z2quijote.runtime_core.cache_manager import SpectrumBank, get_or_create_validation_bank
from z2quijote.runtime_core.camb_data_provider import CAMBDataProvider
from z2quijote.runtime_core.config import ValidationRuntimeConfig
from z2quijote.runtime_core.data_source import resolve_data_source
from z2quijote.runtime_core.evaluation.test_set import (
    write_test_set_results,
)
from z2quijote.runtime_core.module2_facade import predict_spectra
from z2quijote.runtime_core.types import EmulatorState, ValidationArtifacts

ProgressCallback = Callable[[str, int, int], None]


def _array_digest(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    arr = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()[:16]


def _resolved_pca_components(run_metadata: dict[str, object]) -> int:
    raw_value = run_metadata.get("pca_components", None)
    try:
        resolved = int(raw_value) if raw_value is not None else 0
    except (TypeError, ValueError):
        resolved = 0
    if resolved > 0:
        return resolved

    hyperparameters = run_metadata.get("resolved_hyperparameters", {})
    if isinstance(hyperparameters, dict):
        for key in ("pca_components", "pca_components_requested"):
            try:
                fallback = int(hyperparameters.get(key, 0))
            except (TypeError, ValueError):
                fallback = 0
            if fallback > 0:
                return fallback
    return 0


def _build_validation_summary(
    results_payload: dict[str, object],
    *,
    run_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "mode": str(run_metadata.get("mode", "validation")),
        "data_source": str(run_metadata.get("data_source", "unknown")),
        "parameter_space": str(run_metadata.get("parameter_space", "unknown")),
        "theta_dim": int(run_metadata.get("theta_dim", 0)),
        "train_size": int(run_metadata.get("train_size", 0)),
        "validation_points": int(results_payload.get("test_set_size", 0)),
        "k_eval_size": len(results_payload.get("k_bins", [])),
        "pca_components": _resolved_pca_components(run_metadata),
        "overall_mean_relative_error": float(results_payload.get("overall_mean_relative_error", 0.0)),
        "overall_p95_relative_error": float(results_payload.get("overall_p95_relative_error", 0.0)),
        "overall_max_relative_error": float(results_payload.get("overall_max_relative_error", 0.0)),
        "overall_mean_log_error": float(results_payload.get("overall_mean_log_error", 0.0)),
        "overall_p95_log_error": float(results_payload.get("overall_p95_log_error", 0.0)),
        "overall_max_log_error": float(results_payload.get("overall_max_log_error", 0.0)),
        "sample_mean_relative_error_mean": float(
            results_payload.get("sample_mean_relative_error_mean", 0.0)
        ),
        "sample_mean_relative_error_p95": float(
            results_payload.get("sample_mean_relative_error_p95", 0.0)
        ),
        "sample_max_relative_error_mean": float(
            results_payload.get("sample_max_relative_error_mean", 0.0)
        ),
        "sample_max_relative_error_p95": float(
            results_payload.get("sample_max_relative_error_p95", 0.0)
        ),
        "k_le_1_mean_relative_error": float(
            results_payload.get("k_le_1_mean_relative_error", 0.0)
        ),
        "k_le_1_p68_relative_error": float(
            results_payload.get("k_le_1_p68_relative_error", 0.0)
        ),
        "k_le_1_max_relative_error": float(
            results_payload.get("k_le_1_max_relative_error", 0.0)
        ),
        "band_relative_error_low_mean": float(
            results_payload.get("band_relative_error_low_mean", 0.0)
        ),
        "band_relative_error_mid_mean": float(
            results_payload.get("band_relative_error_mid_mean", 0.0)
        ),
        "band_relative_error_focus_high_mean": float(
            results_payload.get("band_relative_error_focus_high_mean", 0.0)
        ),
        "band_relative_error_tail_mean": float(
            results_payload.get("band_relative_error_tail_mean", 0.0)
        ),
        "band_relative_error_high_mean": float(
            results_payload.get("band_relative_error_high_mean", 0.0)
        ),
        "focus_0p08_3_integrated_relative_error_mean": float(
            results_payload.get(
                "focus_0p08_3_integrated_relative_error_mean",
                results_payload.get("focus_0p1_3_integrated_relative_error_mean", 0.0),
            )
        ),
        "focus_0p08_3_integrated_relative_error_p68": float(
            results_payload.get(
                "focus_0p08_3_integrated_relative_error_p68",
                results_payload.get("focus_0p1_3_integrated_relative_error_p68", 0.0),
            )
        ),
        "focus_0p1_3_integrated_relative_error_mean": float(
            results_payload.get(
                "focus_0p1_3_integrated_relative_error_mean",
                results_payload.get("focus_0p1_5_integrated_relative_error_mean", 0.0),
            )
        ),
        "focus_0p1_3_integrated_relative_error_p68": float(
            results_payload.get(
                "focus_0p1_3_integrated_relative_error_p68",
                results_payload.get("focus_0p1_5_integrated_relative_error_p68", 0.0),
            )
        ),
        "focus_0p1_5_integrated_relative_error_mean": float(
            results_payload.get(
                "focus_0p1_5_integrated_relative_error_mean",
                results_payload.get("focus_0p1_3_integrated_relative_error_mean", 0.0),
            )
        ),
        "focus_0p1_5_integrated_relative_error_p68": float(
            results_payload.get(
                "focus_0p1_5_integrated_relative_error_p68",
                results_payload.get("focus_0p1_3_integrated_relative_error_p68", 0.0),
            )
        ),
        "band_log_error_low_mean": float(
            results_payload.get("band_log_error_low_mean", 0.0)
        ),
        "band_log_error_mid_mean": float(
            results_payload.get("band_log_error_mid_mean", 0.0)
        ),
        "band_log_error_focus_high_mean": float(
            results_payload.get("band_log_error_focus_high_mean", 0.0)
        ),
        "band_log_error_tail_mean": float(
            results_payload.get("band_log_error_tail_mean", 0.0)
        ),
        "band_log_error_high_mean": float(
            results_payload.get("band_log_error_high_mean", 0.0)
        ),
        "focus_0p08_3_integrated_log_error_mean": float(
            results_payload.get(
                "focus_0p08_3_integrated_log_error_mean",
                results_payload.get("focus_0p1_3_integrated_log_error_mean", 0.0),
            )
        ),
        "focus_0p08_3_integrated_log_error_p68": float(
            results_payload.get(
                "focus_0p08_3_integrated_log_error_p68",
                results_payload.get("focus_0p1_3_integrated_log_error_p68", 0.0),
            )
        ),
        "focus_0p1_3_integrated_log_error_mean": float(
            results_payload.get(
                "focus_0p1_3_integrated_log_error_mean",
                results_payload.get("focus_0p1_5_integrated_log_error_mean", 0.0),
            )
        ),
        "focus_0p1_3_integrated_log_error_p68": float(
            results_payload.get(
                "focus_0p1_3_integrated_log_error_p68",
                results_payload.get("focus_0p1_5_integrated_log_error_p68", 0.0),
            )
        ),
        "focus_0p1_5_integrated_log_error_mean": float(
            results_payload.get(
                "focus_0p1_5_integrated_log_error_mean",
                results_payload.get("focus_0p1_3_integrated_log_error_mean", 0.0),
            )
        ),
        "focus_0p1_5_integrated_log_error_p68": float(
            results_payload.get(
                "focus_0p1_5_integrated_log_error_p68",
                results_payload.get("focus_0p1_3_integrated_log_error_p68", 0.0),
            )
        ),
    }


def evaluate_predictions_against_truth(
    config: ValidationRuntimeConfig,
    output_dir: Path,
    *,
    test_thetas: np.ndarray,
    k_bins: np.ndarray,
    p_true_batch: np.ndarray,
    p_pred_batch: np.ndarray,
    p_linear_batch: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> ValidationArtifacts:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_source = resolve_data_source(config)
    run_metadata = dict(metadata or {})
    run_metadata.setdefault("data_source", str(data_source.name))
    run_metadata.setdefault("parameter_space", str(data_source.parameter_space))
    run_metadata.setdefault("theta_dim", int(data_source.theta_dim))
    run_metadata.setdefault("theta_names", list(data_source.theta_names))
    run_metadata.setdefault("spectrum_type", str(data_source.spectrum_type))
    run_metadata.setdefault("data_provider_kind", str(data_source.provider_kind))
    run_metadata.setdefault("has_linear_anchor", bool(data_source.has_linear_anchor))
    run_metadata.setdefault("target_transform", str(data_source.target_transform))
    run_metadata.setdefault("data_source_metadata", dict(data_source.metadata))
    results_path = write_test_set_results(
        output_dir,
        test_thetas=test_thetas,
        k_bins=k_bins,
        p_true_batch=p_true_batch,
        p_pred_batch=p_pred_batch,
        p_linear_batch=p_linear_batch,
        spectrum_type=str(data_source.spectrum_type),
        eps_r=float(config.eps_r),
        metadata=run_metadata,
    )
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))
    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(
            _build_validation_summary(results_payload, run_metadata=run_metadata),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ValidationArtifacts(
        output_dir=output_dir,
        test_set_results_path=results_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
        metadata=run_metadata,
    )


def evaluate_emulator_with_validation_bank(
    config: ValidationRuntimeConfig,
    emulator: EmulatorState,
    validation_bank: SpectrumBank,
    output_dir: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> ValidationArtifacts:
    prediction = predict_spectra(
        emulator,
        validation_bank.raw_thetas,
        input_space="raw",
        k_target=validation_bank.k_bins,
        p_linear_batch=validation_bank.p_linear_batch,
    )
    run_metadata = dict(metadata or {})
    run_metadata.update(
        {
            "validation_cache_path": str(validation_bank.npz_path),
            "validation_cache_status": str(validation_bank.metadata.get("cache_status", "unknown")),
            "validation_thetas_digest": _array_digest(validation_bank.raw_thetas),
            "validation_k_digest": _array_digest(validation_bank.k_bins),
            "validation_nonlin_digest": _array_digest(validation_bank.p_nonlin_batch),
            "validation_linear_digest": _array_digest(validation_bank.p_linear_batch),
            "validation_k_size": int(validation_bank.k_bins.shape[0]),
            "validation_points": int(validation_bank.raw_thetas.shape[0]),
            "validation_sampling_method": str(
                validation_bank.metadata.get("sampling_method", "unknown")
            ),
            "train_size": int(emulator.dataset.raw_thetas.shape[0]),
            "train_thetas_digest": _array_digest(emulator.dataset.raw_thetas),
            "train_k_digest": _array_digest(emulator.dataset.k_bins),
            "train_pk_digest": _array_digest(emulator.dataset.pk_batch),
            "train_linear_digest": _array_digest(emulator.dataset.p_linear_batch),
            "pca_components": int(emulator.dataset.pca_scores.shape[1]),
            "target_transform": str(emulator.dataset.metadata.get("target_transform", "unknown")),
            "comparison_space": "power_spectrum",
        }
    )
    return evaluate_predictions_against_truth(
        config,
        output_dir,
        test_thetas=validation_bank.raw_thetas,
        k_bins=validation_bank.k_bins,
        p_true_batch=validation_bank.p_nonlin_batch,
        p_pred_batch=prediction.pk_mean,
        p_linear_batch=validation_bank.p_linear_batch,
        metadata=run_metadata,
    )


def evaluate_emulator_on_validation_set(
    config: ValidationRuntimeConfig,
    camb_data_provider: CAMBDataProvider,
    emulator: EmulatorState,
    output_dir: Path,
    *,
    asset_version: str = "validation",
    progress_callback: ProgressCallback | None = None,
    metadata: dict[str, object] | None = None,
    validation_bank: SpectrumBank | None = None,
    force_rebuild_validation_cache: bool = False,
) -> ValidationArtifacts:
    del asset_version
    active_bank = validation_bank or get_or_create_validation_bank(
        config,
        camb_data_provider,
        progress_callback=progress_callback,
        force_rebuild=force_rebuild_validation_cache,
    )
    run_metadata = dict(metadata or {})
    run_metadata.setdefault("mode", "validation")
    return evaluate_emulator_with_validation_bank(
        config,
        emulator,
        active_bank,
        output_dir,
        metadata=run_metadata,
    )


__all__ = [
    "TEST_SET_RESULTS_FILENAME",
    "build_test_set_results_payload",
    "evaluate_emulator_on_validation_set",
    "evaluate_emulator_with_validation_bank",
    "evaluate_predictions_against_truth",
]
