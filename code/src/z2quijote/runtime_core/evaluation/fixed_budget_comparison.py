"""Fixed-budget Sobol+PCA+GP comparison pipeline without active iterations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from z2quijote.runtime_core.cache_manager import (
    SpectrumBank,
    get_or_create_comparison_training_bank,
    get_or_create_validation_bank,
)
from z2quijote.runtime_core.camb_data_provider import CAMBDataProvider
from z2quijote.runtime_core.config import ValidationRuntimeConfig
from z2quijote.runtime_core.data_source import resolve_data_source
from z2quijote.runtime_core.evaluation.active_learning_validation import evaluate_predictions_against_truth
from z2quijote.runtime_core.evaluation.active_learning_validation import _array_digest
from z2quijote.runtime_core.evaluation.gp_baseline import fit_gp_baseline_from_spectrum_bank
from z2quijote.runtime_core.run_artifacts import run_results_subdir

ProgressCallback = Callable[[str, int, int], None]


@dataclass(slots=True)
class FixedBudgetComparisonArtifacts:
    output_dir: Path
    test_set_results_path: Path
    summary_path: Path
    metadata_path: Path
    train_cache_path: Path
    validation_cache_path: Path
    train_points: int


def run_fixed_budget_comparison(
    *,
    run_dir: Path,
    config: ValidationRuntimeConfig,
    camb_data_provider: CAMBDataProvider,
    validation_bank: SpectrumBank | None = None,
    training_bank: SpectrumBank | None = None,
    output_subdir: str = "fixed_budget_comparison",
    train_points: int | None = None,
    progress_callback: ProgressCallback | None = None,
    force_rebuild_training_cache: bool = False,
    force_rebuild_validation_cache: bool = False,
) -> FixedBudgetComparisonArtifacts:
    resolved_train_points = int(train_points or config.sampling.total_budget)
    active_training_bank = training_bank or get_or_create_comparison_training_bank(
        config,
        camb_data_provider,
        train_points=resolved_train_points,
        progress_callback=progress_callback,
        force_rebuild=force_rebuild_training_cache,
    )
    artifacts = fit_gp_baseline_from_spectrum_bank(
        config=config,
        spectrum_bank=active_training_bank,
        progress_callback=progress_callback,
    )
    output_dir = run_results_subdir(run_dir, str(output_subdir).strip(), create=True)
    active_validation_bank = validation_bank or get_or_create_validation_bank(
        config,
        camb_data_provider,
        progress_callback=progress_callback,
        force_rebuild=force_rebuild_validation_cache,
    )
    data_source = resolve_data_source(config)
    p_pred = artifacts.predict_on_k(
        active_validation_bank.raw_thetas,
        active_validation_bank.k_bins,
        p_linear_batch=active_validation_bank.p_linear_batch,
    )
    validation = evaluate_predictions_against_truth(
        config,
        output_dir,
        test_thetas=active_validation_bank.raw_thetas,
        k_bins=active_validation_bank.k_bins,
        p_true_batch=active_validation_bank.p_nonlin_batch,
        p_pred_batch=p_pred,
        p_linear_batch=active_validation_bank.p_linear_batch,
        metadata={
            "mode": "fixed_budget_comparison",
            "data_source": str(data_source.name),
            "parameter_space": str(data_source.parameter_space),
            "theta_dim": int(data_source.theta_dim),
            "theta_names": list(data_source.theta_names),
            "spectrum_type": str(data_source.spectrum_type),
            "data_provider_kind": str(data_source.provider_kind),
            "has_linear_anchor": bool(data_source.has_linear_anchor),
            "train_size": int(artifacts.train_thetas.shape[0]),
            "train_points": int(artifacts.train_thetas.shape[0]),
            "pca_components": int(artifacts.emulator.dataset.pca_scores.shape[1]),
            "train_cache_path": str(active_training_bank.npz_path),
            "train_cache_status": str(active_training_bank.metadata.get("cache_status", "unknown")),
            "train_thetas_digest": _array_digest(active_training_bank.raw_thetas),
            "train_k_digest": _array_digest(active_training_bank.k_bins),
            "train_pk_digest": _array_digest(active_training_bank.p_nonlin_batch),
            "train_linear_digest": _array_digest(active_training_bank.p_linear_batch),
            "train_cache_theta_digest": str(active_training_bank.metadata.get("theta_digest", "")),
            "train_cache_k_digest": str(active_training_bank.metadata.get("k_digest", "")),
            "validation_cache_path": str(active_validation_bank.npz_path),
            "validation_cache_status": str(active_validation_bank.metadata.get("cache_status", "unknown")),
            "validation_thetas_digest": _array_digest(active_validation_bank.raw_thetas),
            "validation_k_digest": _array_digest(active_validation_bank.k_bins),
            "validation_nonlin_digest": _array_digest(active_validation_bank.p_nonlin_batch),
            "validation_linear_digest": _array_digest(active_validation_bank.p_linear_batch),
            "validation_cache_theta_digest": str(active_validation_bank.metadata.get("theta_digest", "")),
            "validation_cache_k_digest": str(active_validation_bank.metadata.get("k_digest", "")),
            "validation_k_size": int(active_validation_bank.k_bins.shape[0]),
            "validation_points": int(active_validation_bank.raw_thetas.shape[0]),
            "validation_sampling_method": str(
                active_validation_bank.metadata.get("sampling_method", "unknown")
            ),
            "target_transform": str(
                artifacts.emulator.dataset.metadata.get("target_transform", "unknown")
            ),
            "comparison_space": "power_spectrum",
            "resolved_hyperparameters": artifacts.resolved_hyperparameters.to_metadata(),
            "baseline_emulator_kind": "dedicated_fixed_budget_pca_gp",
        },
    )
    validation_cache_path = Path(
        validation.metadata.get("validation_cache_path", output_dir / "validation_cache_missing.npz")
    ).resolve()
    return FixedBudgetComparisonArtifacts(
        output_dir=output_dir,
        test_set_results_path=Path(validation.test_set_results_path).resolve(),
        summary_path=Path(validation.summary_path).resolve(),
        metadata_path=Path(validation.metadata_path).resolve(),
        train_cache_path=active_training_bank.npz_path.resolve(),
        validation_cache_path=validation_cache_path,
        train_points=int(artifacts.train_thetas.shape[0]),
    )
