from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .acquisition import ActiveSelectionResult, select_z2_active_points
from .config import Z2Config
from .direct_cdm import make_oracle
from .emulator import PCAGPDirectCDMEmulator
from .metrics import evaluate_prediction
from .raw_bank import RawBankSample, load_raw_bank_sample
from .resources import load_optional_current_active
from .sampling import digest_theta
from .splits import SplitBundle, build_split_bundle
from .reporting import generate_run_report


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    run_dir: Path
    summary_path: Path
    summary: dict[str, Any]


def run_fair_comparison(
    config: Z2Config,
    *,
    split_bundle: SplitBundle | None = None,
    resume_run_dir: Path | None = None,
) -> ExperimentResult:
    bundle = split_bundle or build_split_bundle(config)
    oracle = make_oracle(config)
    k_bins = config.k_grid.k_bins
    run_dir = Path(resume_run_dir).resolve() if resume_run_dir is not None else _make_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    process_dir = run_dir / "process"
    process_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    if resume_run_dir is not None:
        _write_progress(progress_path, "run_resumed", split_manifest_path=str(bundle.manifest_path))
    else:
        _write_progress(progress_path, "run_started", split_manifest_path=str(bundle.manifest_path))

    def _active_progress(payload: dict[str, Any]) -> None:
        event = str(payload.get("event", "active_selection_progress"))
        event_payload = {key: value for key, value in payload.items() if key != "event"}
        round_payload = event_payload.pop("round", None)
        if round_payload is not None:
            round_index = int(round_payload.get("round_index", event_payload.get("round_index", -1)))
            round_path = process_dir / f"active_round_{round_index:02d}.json"
            _write_text_retry(
                round_path,
                json.dumps(_json_safe(round_payload), ensure_ascii=False, indent=2),
            )
            event_payload["round_record_path"] = str(round_path)
        state_payload = {
            "latest_event": event,
            "selected_so_far": int(event_payload.get("selected_so_far", 0)),
            "target_total": int(event_payload.get("target_total", config.active_learning.active_points)),
            "latest_round_index": event_payload.get("round_index"),
            "latest_stage": event_payload.get("stage"),
            "latest_stage_current": event_payload.get("current"),
            "latest_stage_total": event_payload.get("total"),
            "run_dir": str(run_dir),
            "process_dir": str(process_dir),
        }
        _write_text_retry(
            process_dir / "active_selection_state.json",
            json.dumps(_json_safe(state_payload), ensure_ascii=False, indent=2),
        )
        _write_progress(progress_path, event, **event_payload)

    resume_selected_theta_raw, resume_rounds = _load_active_resume_state(process_dir, dim=bundle.seed_theta_raw.shape[1])
    _write_progress(
        progress_path,
        "active_selection_started",
        resumed_rounds=len(resume_rounds),
        resumed_active_points=int(resume_selected_theta_raw.shape[0]),
    )
    active_result = select_z2_active_points(
        config=config,
        oracle=oracle,
        seed_theta_raw=bundle.seed_theta_raw,
        probe_theta_raw=bundle.probe_theta_raw,
        pool_theta_raw=bundle.pool_theta_raw,
        k_bins=k_bins,
        resume_selected_theta_raw=resume_selected_theta_raw,
        resume_rounds=resume_rounds,
        progress_callback=_active_progress,
    )
    _write_progress(
        progress_path,
        "active_selection_completed",
        active_points_selected=int(active_result.selected_theta_raw.shape[0]),
    )
    seed_budget = int(bundle.seed_theta_raw.shape[0])
    active_budget = int(active_result.selected_theta_raw.shape[0])
    primary_budget = seed_budget + active_budget
    seed_name = f"ppr{seed_budget}"
    active_name = f"{seed_name}_plus_z2_active{active_budget}"
    sobol_active_name = f"{seed_name}_plus_sobol{active_budget}"
    sobol_budget = int(bundle.sobol64_theta_raw.shape[0])
    designs: dict[str, np.ndarray] = {
        seed_name: bundle.seed_theta_raw,
        active_name: np.vstack([bundle.seed_theta_raw, active_result.selected_theta_raw]),
        f"sobol{sobol_budget}": bundle.sobol64_theta_raw,
    }
    if bundle.sobol_tail_theta_raw.shape[0] >= active_budget:
        designs[sobol_active_name] = np.vstack(
            [bundle.seed_theta_raw, bundle.sobol_tail_theta_raw[:active_budget]]
        )
    reserve = int(config.active_learning.sobol_tail_reserve)
    if reserve > 0:
        active_keep = int(config.active_learning.active_points) - reserve
        designs[f"{seed_name}_plus_z2_hybrid{active_keep}_sobol{reserve}"] = np.vstack(
            [
                bundle.seed_theta_raw,
                active_result.selected_theta_raw[:active_keep],
                bundle.sobol_tail_theta_raw[:reserve],
            ]
        )
    current_active_theta, current_active_meta = load_optional_current_active(config)
    if current_active_theta is not None:
        designs[f"{seed_name}_plus_current_active{current_active_theta.shape[0]}"] = np.vstack(
            [bundle.seed_theta_raw, current_active_theta]
        )

    _write_progress(progress_path, "audit_truth_started", audit_size=int(bundle.audit_theta_raw.shape[0]))
    audit_truth = oracle.evaluate(bundle.audit_theta_raw, k_bins)
    _write_progress(progress_path, "audit_truth_completed")
    raw_bank_sample, raw_bank_metadata = load_raw_bank_sample(config, k_bins)
    _write_progress(progress_path, "raw_bank_checked", **raw_bank_metadata)
    design_results: dict[str, Any] = {}
    for name, theta in designs.items():
        _write_progress(progress_path, "design_evaluation_started", design_name=name, training_points=int(theta.shape[0]))
        design_results[name] = _evaluate_design(
            config=config,
            oracle=oracle,
            design_name=name,
            theta_raw=theta,
            audit_theta_raw=bundle.audit_theta_raw,
            audit_truth_log=audit_truth.log_pk,
            k_bins=k_bins,
            run_dir=run_dir,
            raw_bank_sample=raw_bank_sample,
        )
        metrics = design_results[name]["metrics"]["overall_relative_error"]
        _write_progress(
            progress_path,
            "design_evaluation_completed",
            design_name=name,
            p68=float(metrics["p68"]),
        )
    comparison = _comparison_block(design_results)
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package": "z2quijote",
        "target_kind": str(config.target.kind),
        "anchor_mode": str(config.target.anchor_mode),
        "uses_lofi": False,
        "oracle_metadata": audit_truth.metadata,
        "config_summary": config.summary(),
        "split_manifest_path": str(bundle.manifest_path),
        "split_arrays_path": str(bundle.arrays_path),
        "active_selection": active_result.report,
        "active_design_path": str(_save_active_design(run_dir, active_result)),
        "raw_bank": raw_bank_metadata,
        "current_active_resource": current_active_meta,
        "design_results": design_results,
        "comparison": comparison,
        "metric_policy": _metric_policy(config),
        "success_criteria": {
            "primary": "primary z2 design overall_relative_error.p68 < same-budget Sobol overall_relative_error.p68",
            "reported_only": "p68 is the only decision/reporting metric; p50, p95, mean, and max are auxiliary diagnostics only.",
            "evaluated_on": "untouched audit split",
        },
    }
    summary_path = run_dir / f"z2_{str(config.target.kind)}_fair_comparison_summary.json"
    _write_text_retry(summary_path, json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    _write_progress(progress_path, "summary_written", summary_path=str(summary_path))
    _write_progress(progress_path, "report_generation_started", summary_path=str(summary_path))
    report_manifest = generate_run_report(
        config=config,
        run_dir=run_dir,
        summary=summary,
        summary_path=summary_path,
        split_bundle=bundle,
        oracle=oracle,
        active_result=active_result,
    )
    summary["report_manifest_path"] = str(report_manifest["report_manifest_path"])
    summary["report_artifacts"] = dict(report_manifest.get("artifacts", {}))
    summary["plot_paths"] = list(report_manifest.get("plot_paths", []))
    _write_text_retry(summary_path, json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    _write_progress(
        progress_path,
        "report_generated",
        report_manifest_path=str(report_manifest["report_manifest_path"]),
        plot_dir=str(report_manifest.get("artifacts", {}).get("plot_dir", "")),
        plot_count=len(report_manifest.get("plot_paths", [])),
    )
    return ExperimentResult(run_dir=run_dir, summary_path=summary_path, summary=summary)


def _evaluate_design(
    *,
    config: Z2Config,
    oracle: Any,
    design_name: str,
    theta_raw: np.ndarray,
    audit_theta_raw: np.ndarray,
    audit_truth_log: np.ndarray,
    k_bins: np.ndarray,
    run_dir: Path,
    raw_bank_sample: RawBankSample | None,
) -> dict[str, Any]:
    labels = oracle.evaluate(theta_raw, k_bins)
    emulator = PCAGPDirectCDMEmulator(
        config.parameter_space,
        config.model,
        target_kind=str(config.target.kind),
    ).fit(
        theta_raw,
        labels.log_pk,
        k_bins,
    )
    pred = emulator.predict(audit_theta_raw)
    metrics = evaluate_prediction(
        truth_log_pk=audit_truth_log,
        pred_log_pk=pred.log_pk_mean,
        k_bins=k_bins,
        band_edges=config.evaluation.band_edges,
        band_labels=config.evaluation.band_labels,
        target_kind=str(config.target.kind),
    )
    design_path = run_dir / f"{design_name}_design.npz"
    np.savez_compressed(
        design_path,
        theta_raw=np.asarray(theta_raw, dtype=np.float64),
        k_bins=np.asarray(k_bins, dtype=np.float64),
    )
    prediction_path = None
    if config.evaluation.save_predictions:
        prediction_path = run_dir / f"{design_name}_audit_predictions.npz"
        np.savez_compressed(
            prediction_path,
            theta_raw=np.asarray(audit_theta_raw, dtype=np.float64),
            k_bins=np.asarray(k_bins, dtype=np.float64),
            truth_log_pk=np.asarray(audit_truth_log, dtype=np.float64),
            pred_log_pk=np.asarray(pred.log_pk_mean, dtype=np.float64),
        )
    raw_bank_metrics = None
    if raw_bank_sample is not None:
        raw_pred = emulator.predict(raw_bank_sample.theta_raw)
        raw_bank_metrics = evaluate_prediction(
            truth_log_pk=raw_bank_sample.log_pk,
            pred_log_pk=raw_pred.log_pk_mean,
            k_bins=raw_bank_sample.k_bins,
            band_edges=config.evaluation.band_edges,
            band_labels=config.evaluation.band_labels,
            target_kind=str(config.target.kind),
        )
    return {
        "design_name": design_name,
        "target_kind": str(config.target.kind),
        "training_points": int(theta_raw.shape[0]),
        "theta_digest": digest_theta(config.parameter_space.normalize(theta_raw), decimals=config.splits.duplicate_decimals),
        "design_path": str(design_path),
        "prediction_path": str(prediction_path) if prediction_path else None,
        "emulator": emulator.metadata,
        "metrics": metrics,
        "raw_bank_metrics": raw_bank_metrics,
    }


def _comparison_block(results: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    primary_name = _primary_design_name(results)
    primary_budget = int(results.get(primary_name, {}).get("training_points", -1))
    baseline_names = sorted(
        name
        for name, result in results.items()
        if (
            name.startswith("sobol")
            or ("_plus_sobol" in name)
            or (name.startswith("ppr") and "_plus_" not in name)
        )
        and (
            int(result.get("training_points", -2)) in {primary_budget, primary_budget // 2}
            or (name.startswith("ppr") and "_plus_" not in name)
        )
    )
    for baseline_name in baseline_names:
        _add_comparison(
            comparison,
            results=results,
            primary_name=primary_name,
            baseline_name=baseline_name,
            success_baseline=(baseline_name == f"sobol{primary_budget}"),
        )
    _add_same_budget_sobol_comparisons(comparison, results)
    return comparison


def _primary_design_name(results: dict[str, Any]) -> str:
    hybrid = sorted(name for name in results if "_plus_z2_hybrid" in name)
    if hybrid:
        return hybrid[0]
    active = sorted(name for name in results if "_plus_z2_active" in name)
    if active:
        return active[0]
    raise KeyError("no z2 active design found in results.")


def _add_same_budget_sobol_comparisons(comparison: dict[str, Any], results: dict[str, Any]) -> None:
    sobol_by_budget = {
        int(result["training_points"]): name
        for name, result in results.items()
        if name.startswith("sobol") and "training_points" in result
    }
    for name, result in results.items():
        if name.startswith("sobol") or "training_points" not in result:
            continue
        budget = int(result["training_points"])
        baseline_name = sobol_by_budget.get(budget)
        if baseline_name is None:
            continue
        _add_comparison(
            comparison,
            results=results,
            primary_name=name,
            baseline_name=baseline_name,
            success_baseline=False,
        )


def _add_comparison(
    comparison: dict[str, Any],
    *,
    results: dict[str, Any],
    primary_name: str,
    baseline_name: str,
    success_baseline: bool = False,
) -> None:
    z2 = results.get(primary_name)
    baseline = results.get(baseline_name)
    if not z2 or not baseline:
        return
    base = baseline["metrics"]["overall_relative_error"]
    new = z2["metrics"]["overall_relative_error"]
    block = {
        "primary_metric": "overall_relative_error.p68",
        "p68_improvement_fraction": _improvement(base["p68"], new["p68"]),
    }
    if success_baseline:
        block["primary_success"] = bool(new["p68"] < base["p68"])
    comparison[f"{primary_name}_vs_{baseline_name}"] = block
    if z2.get("raw_bank_metrics") and baseline.get("raw_bank_metrics"):
        raw_base = baseline["raw_bank_metrics"]["overall_relative_error"]
        raw_new = z2["raw_bank_metrics"]["overall_relative_error"]
        comparison[f"{primary_name}_raw_bank_vs_{baseline_name}"] = {
            "primary_metric": "overall_relative_error.p68",
            "p68_improvement_fraction": _improvement(raw_base["p68"], raw_new["p68"]),
        }


def _alias_design_result(result: dict[str, Any], *, alias_name: str, source_name: str) -> dict[str, Any]:
    alias = copy.deepcopy(result)
    alias["design_name"] = str(alias_name)
    alias["alias_of"] = str(source_name)
    return alias


def _improvement(base: float, value: float) -> float:
    return float(1.0 - float(value) / max(float(base), 1.0e-300))


def _metric_policy(config: Z2Config) -> dict[str, Any]:
    return {
        "primary_metric": str(config.evaluation.primary_metric),
        "primary_curve": str(config.evaluation.primary_curve),
        "report_metric_policy": str(config.evaluation.report_metric_policy),
        "reported_quantiles": [68.0],
        "auxiliary_only": ["p50", "p95", "mean", "max", "signed_bias"],
    }


def _make_run_dir(config: Z2Config) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (config.evaluation.output_dir / f"z2_{str(config.target.kind)}_{timestamp}").resolve()


def _load_active_resume_state(process_dir: Path, *, dim: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rounds: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []
    for path in sorted(Path(process_dir).glob("active_round_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            break
        expected_index = len(rounds)
        if int(payload.get("round_index", expected_index)) != expected_index:
            break
        selected = np.asarray(payload.get("selected_theta_raw", []), dtype=np.float64)
        if selected.ndim == 1 and selected.size:
            selected = selected.reshape(1, -1)
        if selected.ndim != 2 or selected.shape[1] != int(dim) or selected.shape[0] == 0:
            break
        rounds.append(payload)
        rows.append(selected)
    selected_theta = (
        np.vstack(rows).astype(np.float64)
        if rows
        else np.empty((0, int(dim)), dtype=np.float64)
    )
    return selected_theta, rounds


def _save_active_design(run_dir: Path, active: ActiveSelectionResult) -> Path:
    path = run_dir / "r2_seed_plus_z2_active_new_points.npz"
    np.savez_compressed(
        path,
        selected_theta_raw=active.selected_theta_raw,
        selected_pool_indices=active.selected_pool_indices,
    )
    return path


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


def _write_progress(progress_file: Path, event: str, **payload: Any) -> None:
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    line = json.dumps(_json_safe(record), ensure_ascii=False) + "\n"
    for attempt in range(3):
        try:
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            with progress_file.open("a", encoding="utf-8") as handle:
                handle.write(line)
            return
        except OSError:
            if attempt >= 2:
                raise
            time.sleep(0.05 * float(attempt + 1))


def _write_text_retry(path: Path, text: str) -> None:
    target = Path(path)
    for attempt in range(3):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            return
        except OSError:
            if attempt >= 2:
                raise
            time.sleep(0.05 * float(attempt + 1))
