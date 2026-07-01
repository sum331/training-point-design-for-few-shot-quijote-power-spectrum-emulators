from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.plotting.plot_active_learning_summary import plot_active_learning_summary
from scripts.plotting.plot_fair_comparison_summary import generate_summary_suite
from z2quijote.config import Z2Config
from z2quijote.direct_cdm import make_oracle
from z2quijote.emulator import PCAGPDirectCDMEmulator
from z2quijote.runtime_core.evaluation.comparison_report import write_comparison_report
from z2quijote.runtime_core.evaluation.test_set import build_test_set_results_payload
from z2quijote.runtime_core.run_artifacts import run_process_path, run_results_dir, run_results_path, run_results_subdir
from z2quijote.splits import SplitBundle, load_split_bundle


@dataclass(slots=True)
class ReportValidationArtifacts:
    output_dir: Path
    test_set_results_path: Path
    summary_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _save_json(path: Path, payload: dict[str, Any]) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _summary_path_from_run_dir(run_dir: Path, summary_path: Path | None = None) -> Path:
    if summary_path is not None:
        return Path(summary_path).resolve()
    run_dir = Path(run_dir).resolve()
    candidates = sorted(run_dir.glob("*_fair_comparison_summary.json"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No fair-comparison summary found under {run_dir}")


def _load_design_points(design_path: Path) -> np.ndarray:
    with np.load(Path(design_path).resolve(), allow_pickle=False) as data:
        if "theta_raw" in data.files:
            return np.asarray(data["theta_raw"], dtype=np.float64)
        if "selected_theta_raw" in data.files:
            return np.asarray(data["selected_theta_raw"], dtype=np.float64)
    raise KeyError(f"design file does not contain theta_raw or selected_theta_raw: {design_path}")


def _primary_design_name(design_results: dict[str, Any]) -> str:
    hybrid = sorted(name for name in design_results if "_plus_z2_hybrid" in name)
    if hybrid:
        return hybrid[0]
    active = sorted(name for name in design_results if "_plus_z2_active" in name)
    if active:
        return active[0]
    raise KeyError("No active design found in summary design_results.")


def _same_budget_sobol_name(design_results: dict[str, Any], budget: int) -> str | None:
    exact = f"sobol{budget}"
    if exact in design_results:
        return exact
    for name, result in design_results.items():
        if not name.startswith("sobol"):
            continue
        try:
            if int(result.get("training_points", -1)) == int(budget):
                return name
        except (TypeError, ValueError):
            continue
    return None


def _baseline_seed_name(design_results: dict[str, Any]) -> str | None:
    candidates = sorted(
        name for name in design_results if name.startswith("ppr") and "_plus_" not in name
    )
    return candidates[0] if candidates else None


def _build_validation_artifacts(
    *,
    config: Z2Config,
    oracle: Any,
    design_name: str,
    design_path: Path,
    audit_theta_raw: np.ndarray,
    audit_truth_log: np.ndarray,
    k_bins: np.ndarray,
    output_dir: Path,
    mode: str,
    design_label: str,
) -> ReportValidationArtifacts:
    theta_raw = _load_design_points(design_path)
    labels = oracle.evaluate(theta_raw, k_bins)
    emulator = PCAGPDirectCDMEmulator(
        config.parameter_space,
        config.model,
        target_kind=str(config.target.kind),
    ).fit(theta_raw, labels.log_pk, k_bins)
    prediction = emulator.predict(audit_theta_raw)
    truth_power = np.exp(np.asarray(audit_truth_log, dtype=np.float64))
    pred_power = np.exp(np.asarray(prediction.log_pk_mean, dtype=np.float64))
    signed_relative_bias = pred_power / np.maximum(truth_power, 1.0e-30) - 1.0
    metadata = {
        "mode": str(mode),
        "design_name": str(design_name),
        "design_label": str(design_label),
        "train_points": int(theta_raw.shape[0]),
        "train_size": int(theta_raw.shape[0]),
        "train_design_path": str(Path(design_path).resolve()),
        "target_kind": str(config.target.kind),
        "data_source": str(config.target.kind),
        "parameter_space": str(config.parameter_space.name),
        "theta_dim": int(config.parameter_space.dim),
        "theta_names": list(config.parameter_space.theta_names),
        "spectrum_type": str(config.target.kind),
        "data_provider_kind": "z2quijote",
        "has_linear_anchor": bool(config.target.anchor_mode != "none"),
        "target_transform": str(config.target.anchor_mode),
        "comparison_space": "power_spectrum",
    }
    compact_payload = build_test_set_results_payload(
        test_thetas=audit_theta_raw,
        k_bins=k_bins,
        p_true_batch=truth_power,
        p_pred_batch=pred_power,
        spectrum_type=str(config.target.kind),
        metadata=metadata,
    )
    spectra_npz_path = Path(output_dir).resolve() / "spectra_arrays.npz"
    np.savez_compressed(
        spectra_npz_path,
        test_thetas=np.asarray(audit_theta_raw, dtype=np.float64),
        k_bins=np.asarray(k_bins, dtype=np.float64),
        p_true_batch=np.asarray(truth_power, dtype=np.float64),
        p_pred_batch=np.asarray(pred_power, dtype=np.float64),
        signed_relative_bias=np.asarray(signed_relative_bias, dtype=np.float64),
    )
    for key in (
        "test_thetas",
        "k_bins",
        "p_true_batch",
        "p_pred_batch",
        "p_true_mean",
        "p_pred_mean",
        "p_linear_batch",
        "p_linear_mean",
    ):
        compact_payload.pop(key, None)
    compact_payload["spectra_npz_path"] = str(spectra_npz_path)
    results_path = Path(output_dir).resolve() / "test_set_results.json"
    results_path.write_text(json.dumps(_json_safe(compact_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    results_payload = dict(compact_payload)
    summary_payload = {
        "mode": str(mode),
        "data_source": str(metadata["data_source"]),
        "parameter_space": str(metadata["parameter_space"]),
        "theta_dim": int(metadata["theta_dim"]),
        "train_size": int(metadata["train_points"]),
        "validation_points": int(results_payload.get("test_set_size", 0)),
        "k_eval_size": int(np.asarray(k_bins, dtype=np.float64).reshape(-1).shape[0]),
        "overall_mean_relative_error": float(results_payload.get("overall_mean_relative_error", 0.0)),
        "overall_p68_relative_error": float(results_payload.get("overall_p68_relative_error", 0.0)),
        "overall_p95_relative_error": float(results_payload.get("overall_p95_relative_error", 0.0)),
        "overall_max_relative_error": float(results_payload.get("overall_max_relative_error", 0.0)),
        "overall_mean_log_error": float(results_payload.get("overall_mean_log_error", 0.0)),
        "overall_p68_log_error": float(results_payload.get("overall_p68_log_error", 0.0)),
        "overall_p95_log_error": float(results_payload.get("overall_p95_log_error", 0.0)),
        "overall_max_log_error": float(results_payload.get("overall_max_log_error", 0.0)),
        "sample_mean_relative_error_mean": float(results_payload.get("sample_mean_relative_error_mean", 0.0)),
        "sample_max_relative_error_mean": float(results_payload.get("sample_max_relative_error_mean", 0.0)),
        "k_le_1_mean_relative_error": float(results_payload.get("k_le_1_mean_relative_error", 0.0)),
        "k_le_1_p68_relative_error": float(results_payload.get("k_le_1_p68_relative_error", 0.0)),
        "band_relative_error_low_mean": float(results_payload.get("band_relative_error_low_mean", 0.0)),
        "band_relative_error_mid_mean": float(results_payload.get("band_relative_error_mid_mean", 0.0)),
        "band_relative_error_focus_high_mean": float(results_payload.get("band_relative_error_focus_high_mean", 0.0)),
        "band_relative_error_tail_mean": float(results_payload.get("band_relative_error_tail_mean", 0.0)),
        "band_relative_error_high_mean": float(results_payload.get("band_relative_error_high_mean", 0.0)),
    }
    summary_path = Path(output_dir).resolve() / "validation_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path = Path(output_dir).resolve() / "run_metadata.json"
    metadata_path.write_text(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    return ReportValidationArtifacts(
        output_dir=Path(output_dir).resolve(),
        test_set_results_path=Path(results_path).resolve(),
        summary_path=summary_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def _write_iteration_history(
    *,
    run_dir: Path,
    active_report: dict[str, Any],
    seed_theta_raw: np.ndarray,
) -> Path:
    rounds = list(active_report.get("rounds", []))
    history: list[dict[str, Any]] = []
    train_size_before = int(np.asarray(seed_theta_raw, dtype=np.float64).shape[0])
    for round_item in rounds:
        round_index = int(round_item.get("round_index", len(history)))
        selected_count = int(round_item.get("selected_count", 0))
        train_size_after = int(round_item.get("training_points_after_round", train_size_before + selected_count))
        m3_metadata = dict(round_item.get("m3_metadata", {}))
        num_simplices = int(
            m3_metadata.get(
                "num_scored_simplices",
                m3_metadata.get("num_hull_simplices", m3_metadata.get("num_simplices", 0)),
            )
            or 0
        )
        history.append(
            {
                "iteration_index": round_index + 1,
                "round_index": round_index,
                "train_size_before": train_size_before,
                "train_size_after": train_size_after,
                "selected_raw_thetas": round_item.get("selected_theta_raw", []),
                "selected_unit_thetas": round_item.get("selected_theta_unit", []),
                "selected_source_pc": round_item.get("m3_selected_source_pc", []),
                "selected_scores": round_item.get("selected_scores", []),
                "metadata": {
                    "candidate_source": round_item.get("candidate_source", "m3"),
                    "selected_count": selected_count,
                    "training_points_after_round": train_size_after,
                    "num_simplices": num_simplices,
                    "m3_metadata": m3_metadata,
                },
            }
        )
        train_size_before = train_size_after

    path = run_process_path(run_dir, "iteration_history.json", create=True)
    path.write_text(json.dumps(_json_safe(history), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_training_point_summary(
    *,
    run_dir: Path,
    config: Z2Config,
    seed_theta_raw: np.ndarray,
    active_report: dict[str, Any],
) -> Path:
    selected_batches = [np.asarray(item.get("selected_theta_raw", []), dtype=np.float64) for item in active_report.get("rounds", [])]
    selected_unit_batches = [np.asarray(item.get("selected_theta_unit", []), dtype=np.float64) for item in active_report.get("rounds", [])]
    selected_raw_batches = [batch for batch in selected_batches if batch.size > 0]
    selected_unit_batches = [batch for batch in selected_unit_batches if batch.size > 0]
    if selected_raw_batches:
        final_raw = np.vstack([np.asarray(seed_theta_raw, dtype=np.float64)] + selected_raw_batches)
    else:
        final_raw = np.asarray(seed_theta_raw, dtype=np.float64)

    payload = {
        "parameter_names": list(config.parameter_space.theta_names),
        "initial_raw_thetas": np.asarray(seed_theta_raw, dtype=np.float64).tolist(),
        "final_raw_thetas": final_raw.tolist(),
        "initial_train_size": int(np.asarray(seed_theta_raw, dtype=np.float64).shape[0]),
        "final_train_size": int(final_raw.shape[0]),
        "selected_raw_thetas_by_iteration": [batch.tolist() for batch in selected_batches],
        "selected_unit_thetas_by_iteration": [batch.tolist() for batch in selected_unit_batches],
    }
    path = run_process_path(run_dir, "training_point_summary.json", create=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def generate_run_report(
    *,
    config: Z2Config,
    run_dir: Path,
    summary: dict[str, Any] | None = None,
    summary_path: Path | None = None,
    split_bundle: SplitBundle | None = None,
    oracle: Any | None = None,
    active_result: Any | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    results_dir = run_results_dir(run_dir, create=True)
    process_dir = run_process_path(run_dir, create=True)
    resolved_summary_path = _summary_path_from_run_dir(run_dir, summary_path)
    resolved_summary = dict(summary or _load_json(resolved_summary_path))
    design_results = dict(resolved_summary.get("design_results", {}))
    if not design_results:
        raise ValueError("Summary does not contain design_results.")

    bundle = split_bundle
    if bundle is None:
        split_manifest = resolved_summary.get("split_manifest_path")
        if not split_manifest:
            raise ValueError("Summary does not contain split_manifest_path.")
        bundle = load_split_bundle(Path(split_manifest))

    resolved_oracle = oracle or make_oracle(config)
    active_report = dict(active_result.report if active_result is not None else resolved_summary.get("active_selection", {}))
    primary_name = _primary_design_name(design_results)
    primary_entry = dict(design_results[primary_name])
    primary_budget = int(primary_entry.get("training_points", bundle.seed_theta_raw.shape[0] + 32))
    sobol_name = _same_budget_sobol_name(design_results, primary_budget)
    if sobol_name is None:
        sobol_name = next((name for name in design_results if name.startswith("sobol")), None)
    if sobol_name is None:
        raise KeyError("No same-budget Sobol design found in summary.")
    seed_name = _baseline_seed_name(design_results)

    active_validation_dir = run_results_subdir(run_dir, "active_learning_validation", create=True)
    fixed_validation_dir = run_results_subdir(run_dir, "fixed_budget_comparison", create=True)
    plot_dir = run_results_subdir(run_dir, "plots", create=True)

    audit_theta_raw = np.asarray(bundle.audit_theta_raw, dtype=np.float64)
    k_bins = np.asarray(bundle.audit_k_bins if hasattr(bundle, "audit_k_bins") else config.k_grid.k_bins, dtype=np.float64)
    audit_truth = resolved_oracle.evaluate(audit_theta_raw, k_bins)

    active_artifacts = _build_validation_artifacts(
        config=config,
        oracle=resolved_oracle,
        design_name=primary_name,
        design_path=Path(primary_entry["design_path"]),
        audit_theta_raw=audit_theta_raw,
        audit_truth_log=audit_truth.log_pk,
        k_bins=k_bins,
        output_dir=active_validation_dir,
        mode="active_learning_validation",
        design_label=primary_name,
    )
    fixed_artifacts = _build_validation_artifacts(
        config=config,
        oracle=resolved_oracle,
        design_name=sobol_name,
        design_path=Path(design_results[sobol_name]["design_path"]),
        audit_theta_raw=audit_theta_raw,
        audit_truth_log=audit_truth.log_pk,
        k_bins=k_bins,
        output_dir=fixed_validation_dir,
        mode="fixed_budget_comparison",
        design_label=sobol_name,
    )

    iteration_history_path = _write_iteration_history(
        run_dir=run_dir,
        active_report=active_report,
        seed_theta_raw=np.asarray(bundle.seed_theta_raw, dtype=np.float64),
    )
    training_point_summary_path = _write_training_point_summary(
        run_dir=run_dir,
        config=config,
        seed_theta_raw=np.asarray(bundle.seed_theta_raw, dtype=np.float64),
        active_report=active_report,
    )

    comparison_report_path = run_results_path(run_dir, "comparison_report.json", create=True)
    write_comparison_report(
        comparison_report_path,
        results_a_path=active_artifacts.test_set_results_path,
        results_b_path=fixed_artifacts.test_set_results_path,
        label_a=primary_name,
        label_b=sobol_name,
    )

    summary_copy_path = run_results_path(run_dir, "fair_comparison_summary.json", create=True)
    _save_json(summary_copy_path, resolved_summary)

    plot_outputs = []
    plot_outputs.extend(
        plot_active_learning_summary(
            run_dir,
            plot_dir,
        )
    )
    plot_outputs.extend(
        generate_summary_suite(
            resolved_summary_path,
            plot_dir,
        )
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "summary_path": str(resolved_summary_path),
        "summary_copy_path": str(summary_copy_path),
        "primary_design_name": primary_name,
        "fixed_budget_design_name": sobol_name,
        "seed_design_name": seed_name,
        "artifacts": {
            "active_learning_validation": {
                "output_dir": str(active_validation_dir),
                "test_set_results_path": str(active_artifacts.test_set_results_path),
                "summary_path": str(active_artifacts.summary_path),
                "metadata_path": str(active_artifacts.metadata_path),
            },
            "fixed_budget_comparison": {
                "output_dir": str(fixed_validation_dir),
                "test_set_results_path": str(fixed_artifacts.test_set_results_path),
                "summary_path": str(fixed_artifacts.summary_path),
                "metadata_path": str(fixed_artifacts.metadata_path),
            },
            "process": {
                "iteration_history_path": str(iteration_history_path),
                "training_point_summary_path": str(training_point_summary_path),
                "process_dir": str(process_dir),
            },
            "comparison_report_path": str(comparison_report_path),
            "plot_dir": str(plot_dir),
        },
        "plot_paths": [str(Path(path).resolve()) for path in plot_outputs],
    }
    manifest_path = run_results_path(run_dir, "report_manifest.json", create=True)
    manifest["report_manifest_path"] = str(manifest_path)
    _save_json(manifest_path, manifest)
    return manifest
