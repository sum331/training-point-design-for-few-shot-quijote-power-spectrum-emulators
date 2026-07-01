from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from z2quijote.acquisition import select_z2_active_points
from z2quijote.config import Z2Config, load_config
from z2quijote.direct_cdm import make_oracle
from z2quijote.experiment import _evaluate_design, _json_safe
from z2quijote.raw_bank import RawBankSample
from z2quijote.splits import SplitBundle, load_split_bundle


CONDITION_ORDER = [
    "sobol32",
    "ppr32",
    "sobol64",
    "ppr32_plus_sobol32",
    "ppr32_plus_variance_only_al32",
    "ppr32_plus_bias_only_al32",
    "ppr32_plus_variance_bias_al32",
    "sobol32_plus_variance_bias_al32",
]

ACTIVE_CONDITIONS = {
    "ppr32_plus_variance_only_al32",
    "ppr32_plus_bias_only_al32",
    "sobol32_plus_variance_bias_al32",
}

BASE_SUMMARY_NAME = "z2_cdm_logdiff_fair_comparison_summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or register the mechanism-ablation suite for the z2 Quijote paper."
    )
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config_ppr3_64.yaml"))
    parser.add_argument(
        "--base-run-dir",
        default="data/runs/z2_cdm_logdiff_20260628T084451Z",
        help="Existing full-method run used for PPR32/Sobol64/full-method reference records.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Ablation output root. Defaults to data/ablation/mechanism_<timestamp>.",
    )
    parser.add_argument(
        "--suite-dir",
        default=None,
        help="Resume or extend an existing ablation suite directory.",
    )
    parser.add_argument(
        "--condition",
        choices=["next", "all", *CONDITION_ORDER],
        default="next",
        help="Condition to run/register. 'next' follows the fixed checklist order.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute a completed condition.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    base_run_dir = Path(args.base_run_dir).resolve()
    suite_dir = _resolve_suite_dir(config, args)
    suite_dir.mkdir(parents=True, exist_ok=True)

    base_summary = _load_base_summary(base_run_dir)
    split_manifest = Path(base_summary["split_manifest_path"]).resolve()
    bundle = load_split_bundle(split_manifest)
    oracle = make_oracle(config)
    k_bins = np.asarray(config.k_grid.k_bins, dtype=np.float64)
    audit_truth_log = _load_or_build_audit_truth(
        suite_dir=suite_dir,
        oracle=oracle,
        audit_theta_raw=bundle.audit_theta_raw,
        k_bins=k_bins,
    )

    _write_suite_manifest(
        suite_dir=suite_dir,
        config=config,
        base_run_dir=base_run_dir,
        split_manifest=split_manifest,
        base_summary_path=base_run_dir / BASE_SUMMARY_NAME,
    )

    requested = _resolve_requested_conditions(suite_dir, str(args.condition))
    completed: list[dict[str, Any]] = []
    for condition in requested:
        if _condition_completed(suite_dir, condition) and not bool(args.force):
            completed.append(_load_condition_summary(suite_dir, condition))
            continue
        if condition in ACTIVE_CONDITIONS:
            completed.append(
                _run_active_condition(
                    condition=condition,
                    config=config,
                    suite_dir=suite_dir,
                    bundle=bundle,
                    oracle=oracle,
                    k_bins=k_bins,
                    audit_truth_log=audit_truth_log,
                )
            )
        else:
            completed.append(
                _register_or_evaluate_fixed_condition(
                    condition=condition,
                    config=config,
                    suite_dir=suite_dir,
                    base_run_dir=base_run_dir,
                    base_summary=base_summary,
                    bundle=bundle,
                    oracle=oracle,
                    k_bins=k_bins,
                    audit_truth_log=audit_truth_log,
                )
            )
        _write_registry(suite_dir)

    _write_registry(suite_dir)
    print(
        json.dumps(
            {
                "suite_dir": str(suite_dir),
                "requested": requested,
                "registry_json": str(suite_dir / "ablation_registry.json"),
                "registry_csv": str(suite_dir / "ablation_registry.csv"),
                "condition_summaries": [item.get("summary_path") for item in completed],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


def _resolve_suite_dir(config: Z2Config, args: argparse.Namespace) -> Path:
    if args.suite_dir:
        return Path(args.suite_dir).resolve()
    if args.output_root:
        return Path(args.output_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (config.data_root / "ablation" / f"mechanism_{stamp}").resolve()


def _load_base_summary(base_run_dir: Path) -> dict[str, Any]:
    path = base_run_dir / BASE_SUMMARY_NAME
    if not path.exists():
        candidates = sorted(base_run_dir.glob("*_fair_comparison_summary.json"))
        if not candidates:
            raise FileNotFoundError(f"No fair-comparison summary found under {base_run_dir}")
        path = candidates[0]
    return json.loads(path.read_text(encoding="utf-8"))


def _write_suite_manifest(
    *,
    suite_dir: Path,
    config: Z2Config,
    base_run_dir: Path,
    split_manifest: Path,
    base_summary_path: Path,
) -> None:
    path = suite_dir / "ablation_suite_manifest.json"
    if path.exists():
        return
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "MNRAS mechanism-attribution ablation suite",
        "condition_order": CONDITION_ORDER,
        "config_path": str(config.config_path),
        "config_summary": config.summary(),
        "base_run_dir": str(base_run_dir),
        "base_summary_path": str(base_summary_path),
        "split_manifest": str(split_manifest),
        "validation_protocol": {
            "audit_source": str(config.splits.audit_source),
            "audit_path": str(config.splits.audit_path) if config.splits.audit_path else None,
            "audit_size": int(config.splits.audit_size),
            "k_count": int(config.k_grid.k_bins.shape[0]),
            "primary_metric": str(config.evaluation.primary_metric),
        },
    }
    _write_json(path, payload)


def _load_or_build_audit_truth(
    *,
    suite_dir: Path,
    oracle: Any,
    audit_theta_raw: np.ndarray,
    k_bins: np.ndarray,
) -> np.ndarray:
    path = suite_dir / "audit_truth_cache.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            return np.asarray(payload["audit_truth_log"], dtype=np.float64)
    truth = oracle.evaluate(audit_theta_raw, k_bins).log_pk
    np.savez_compressed(
        path,
        audit_theta_raw=np.asarray(audit_theta_raw, dtype=np.float64),
        k_bins=np.asarray(k_bins, dtype=np.float64),
        audit_truth_log=np.asarray(truth, dtype=np.float64),
    )
    return np.asarray(truth, dtype=np.float64)


def _resolve_requested_conditions(suite_dir: Path, request: str) -> list[str]:
    if request == "all":
        return list(CONDITION_ORDER)
    if request == "next":
        for condition in CONDITION_ORDER:
            if not _condition_completed(suite_dir, condition):
                return [condition]
        return []
    return [request]


def _condition_dir(suite_dir: Path, condition: str) -> Path:
    order = CONDITION_ORDER.index(condition) + 1
    return suite_dir / f"{order:02d}_{condition}"


def _condition_summary_path(suite_dir: Path, condition: str) -> Path:
    return _condition_dir(suite_dir, condition) / "condition_summary.json"


def _condition_completed(suite_dir: Path, condition: str) -> bool:
    path = _condition_summary_path(suite_dir, condition)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(payload.get("status", "")).lower() == "completed"


def _load_condition_summary(suite_dir: Path, condition: str) -> dict[str, Any]:
    return json.loads(_condition_summary_path(suite_dir, condition).read_text(encoding="utf-8"))


def _register_or_evaluate_fixed_condition(
    *,
    condition: str,
    config: Z2Config,
    suite_dir: Path,
    base_run_dir: Path,
    base_summary: dict[str, Any],
    bundle: SplitBundle,
    oracle: Any,
    k_bins: np.ndarray,
    audit_truth_log: np.ndarray,
) -> dict[str, Any]:
    condition_dir = _condition_dir(suite_dir, condition)
    condition_dir.mkdir(parents=True, exist_ok=True)
    progress_path = condition_dir / "progress.jsonl"
    _write_progress(progress_path, "condition_started", condition=condition)

    existing_name = {
        "ppr32": "ppr32",
        "sobol64": "sobol64",
        "ppr32_plus_sobol32": "ppr32_plus_sobol32",
        "ppr32_plus_variance_bias_al32": "ppr32_plus_z2_active32",
    }.get(condition)
    if existing_name is not None:
        existing = base_summary.get("design_results", {}).get(existing_name)
        if not existing:
            raise KeyError(f"Base summary does not contain design {existing_name!r}.")
        source_design_path = Path(existing["design_path"]).resolve()
        design_path = condition_dir / f"{condition}_design.npz"
        theta_raw = _load_theta_raw(source_design_path)
        np.savez_compressed(
            design_path,
            theta_raw=np.asarray(theta_raw, dtype=np.float64),
            k_bins=np.asarray(k_bins, dtype=np.float64),
            source_design_path=str(source_design_path),
        )
        summary = _summary_from_design_result(
            condition=condition,
            condition_dir=condition_dir,
            acquisition="registered_existing",
            initial_design=_initial_design_label(condition),
            design_path=design_path,
            result=existing,
            source_summary=str(base_run_dir / BASE_SUMMARY_NAME),
            status="completed",
        )
        _write_json(condition_dir / "condition_summary.json", summary)
        _write_progress(progress_path, "condition_completed", condition=condition, p68=summary["overall_p68"])
        return summary

    if condition != "sobol32":
        raise ValueError(f"Unsupported fixed condition: {condition}")

    theta_raw = np.asarray(bundle.sobol64_theta_raw[:32], dtype=np.float64)
    result = _evaluate_design(
        config=config,
        oracle=oracle,
        design_name=condition,
        theta_raw=theta_raw,
        audit_theta_raw=bundle.audit_theta_raw,
        audit_truth_log=audit_truth_log,
        k_bins=k_bins,
        run_dir=condition_dir,
        raw_bank_sample=None,
    )
    summary = _summary_from_design_result(
        condition=condition,
        condition_dir=condition_dir,
        acquisition="fixed_sobol_first32",
        initial_design="sobol32",
        design_path=Path(result["design_path"]),
        result=result,
        source_summary=None,
        status="completed",
    )
    _write_json(condition_dir / "condition_summary.json", summary)
    _write_progress(progress_path, "condition_completed", condition=condition, p68=summary["overall_p68"])
    return summary


def _run_active_condition(
    *,
    condition: str,
    config: Z2Config,
    suite_dir: Path,
    bundle: SplitBundle,
    oracle: Any,
    k_bins: np.ndarray,
    audit_truth_log: np.ndarray,
) -> dict[str, Any]:
    condition_dir = _condition_dir(suite_dir, condition)
    condition_dir.mkdir(parents=True, exist_ok=True)
    process_dir = condition_dir / "process"
    process_dir.mkdir(parents=True, exist_ok=True)
    progress_path = condition_dir / "progress.jsonl"
    _write_progress(progress_path, "condition_started", condition=condition)

    condition_config = _condition_config(config, condition)
    seed_raw, seed_unit, initial_label = _condition_seed(config, condition, bundle)
    completed_theta, completed_rounds = _load_completed_rounds(process_dir, dim=seed_raw.shape[1])

    def _active_progress(payload: dict[str, Any]) -> None:
        event = str(payload.get("event", "active_selection_progress"))
        event_payload = {key: value for key, value in payload.items() if key != "event"}
        round_payload = event_payload.pop("round", None)
        if round_payload is not None:
            round_index = int(round_payload.get("round_index", event_payload.get("round_index", -1)))
            round_path = process_dir / f"active_round_{round_index:02d}.json"
            _write_json(round_path, round_payload)
            event_payload["round_record_path"] = str(round_path)
        state_payload = {
            "latest_event": event,
            "condition": condition,
            "selected_so_far": int(event_payload.get("selected_so_far", 0)),
            "target_total": int(event_payload.get("target_total", condition_config.active_learning.active_points)),
            "latest_round_index": event_payload.get("round_index"),
            "latest_stage": event_payload.get("stage"),
            "latest_stage_current": event_payload.get("current"),
            "latest_stage_total": event_payload.get("total"),
            "run_dir": str(condition_dir),
            "process_dir": str(process_dir),
        }
        _write_json(process_dir / "active_selection_state.json", state_payload)
        _write_progress(progress_path, event, **event_payload)

    active_result = select_z2_active_points(
        config=condition_config,
        oracle=oracle,
        seed_theta_raw=seed_raw,
        probe_theta_raw=bundle.probe_theta_raw,
        pool_theta_raw=bundle.pool_theta_raw,
        k_bins=k_bins,
        resume_selected_theta_raw=completed_theta,
        resume_rounds=completed_rounds,
        progress_callback=_active_progress,
    )

    active_path = condition_dir / f"{condition}_active_points.npz"
    np.savez_compressed(
        active_path,
        selected_theta_raw=np.asarray(active_result.selected_theta_raw, dtype=np.float64),
        selected_theta_unit=condition_config.parameter_space.normalize(active_result.selected_theta_raw),
        selected_scores=np.asarray(active_result.report.get("selected_scores", []), dtype=np.float64),
    )
    final_theta = np.vstack([seed_raw, active_result.selected_theta_raw]).astype(np.float64)
    final_design_path = condition_dir / f"{condition}_design.npz"
    np.savez_compressed(
        final_design_path,
        theta_raw=final_theta,
        theta_unit=condition_config.parameter_space.normalize(final_theta),
        seed_theta_raw=seed_raw,
        seed_theta_unit=seed_unit,
        active_theta_raw=np.asarray(active_result.selected_theta_raw, dtype=np.float64),
        k_bins=np.asarray(k_bins, dtype=np.float64),
        active_points_path=str(active_path),
    )

    result = _evaluate_design(
        config=condition_config,
        oracle=oracle,
        design_name=condition,
        theta_raw=final_theta,
        audit_theta_raw=bundle.audit_theta_raw,
        audit_truth_log=audit_truth_log,
        k_bins=k_bins,
        run_dir=condition_dir,
        raw_bank_sample=None,
    )
    summary = _summary_from_design_result(
        condition=condition,
        condition_dir=condition_dir,
        acquisition=_condition_acquisition_label(condition),
        initial_design=initial_label,
        design_path=final_design_path,
        result=result,
        source_summary=None,
        status="completed",
    )
    summary["active_selection"] = {
        "selected_count": int(active_result.selected_theta_raw.shape[0]),
        "active_points_path": str(active_path),
        "report": active_result.report,
    }
    summary["condition_config_summary"] = condition_config.summary()
    _write_json(condition_dir / "condition_summary.json", summary)
    _write_progress(progress_path, "condition_completed", condition=condition, p68=summary["overall_p68"])
    return summary


def _condition_config(config: Z2Config, condition: str) -> Z2Config:
    if condition == "ppr32_plus_variance_only_al32":
        return replace(config, fastmock_bias=replace(config.fastmock_bias, enabled=False))
    if condition == "ppr32_plus_bias_only_al32":
        return replace(config, fastmock_bias=replace(config.fastmock_bias, enabled=True, score_mode="bias_only"))
    if condition == "sobol32_plus_variance_bias_al32":
        return replace(config, fastmock_bias=replace(config.fastmock_bias, enabled=True, score_mode="variance_bias"))
    raise ValueError(f"Condition is not active: {condition}")


def _condition_seed(
    config: Z2Config,
    condition: str,
    bundle: SplitBundle,
) -> tuple[np.ndarray, np.ndarray, str]:
    if condition in {"ppr32_plus_variance_only_al32", "ppr32_plus_bias_only_al32"}:
        return (
            np.asarray(bundle.seed_theta_raw, dtype=np.float64),
            np.asarray(bundle.seed_theta_unit, dtype=np.float64),
            "ppr32",
        )
    if condition == "sobol32_plus_variance_bias_al32":
        raw = np.asarray(bundle.sobol64_theta_raw[:32], dtype=np.float64)
        return raw, config.parameter_space.normalize(raw), "sobol32"
    raise ValueError(f"Unsupported active condition: {condition}")


def _condition_acquisition_label(condition: str) -> str:
    return {
        "ppr32_plus_variance_only_al32": "variance_only_al32",
        "ppr32_plus_bias_only_al32": "bias_only_al32",
        "sobol32_plus_variance_bias_al32": "variance_bias_al32",
    }[condition]


def _initial_design_label(condition: str) -> str:
    if condition.startswith("ppr32"):
        return "ppr32"
    if condition.startswith("sobol32"):
        return "sobol32"
    if condition.startswith("sobol64"):
        return "sobol64"
    return "unknown"


def _summary_from_design_result(
    *,
    condition: str,
    condition_dir: Path,
    acquisition: str,
    initial_design: str,
    design_path: Path,
    result: dict[str, Any],
    source_summary: str | None,
    status: str,
) -> dict[str, Any]:
    overall = result["metrics"]["overall_relative_error"]
    bands = _extract_band_p68(result["metrics"])
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "condition": condition,
        "condition_dir": str(condition_dir),
        "summary_path": str(condition_dir / "condition_summary.json"),
        "initial_design": initial_design,
        "acquisition": acquisition,
        "training_points": int(result.get("training_points", 0)),
        "design_path": str(Path(design_path).resolve()),
        "source_summary": source_summary,
        "target_kind": str(result.get("target_kind", "")),
        "theta_digest": result.get("theta_digest"),
        "overall_p50": float(overall.get("p50", np.nan)),
        "overall_p68": float(overall.get("p68", np.nan)),
        "overall_p90": float(overall.get("p90", np.nan)),
        "overall_p95": float(overall.get("p95", np.nan)),
        "overall_p98": float(overall.get("p98", np.nan)),
        "overall_mean": float(overall.get("mean", np.nan)),
        "overall_max": float(overall.get("max", np.nan)),
        "band_p68": bands,
        "emulator": result.get("emulator", {}),
        "metrics": result.get("metrics", {}),
    }


def _extract_band_p68(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.get("bands", {}).items():
        if isinstance(value, dict) and "relative_error" in value:
            rel = value["relative_error"]
            if isinstance(rel, dict) and "p68" in rel:
                out[str(key)] = float(rel["p68"])
    for key, value in metrics.items():
        if key.startswith("band_relative_error_") and key.endswith("_p68"):
            label = key.removeprefix("band_relative_error_").removesuffix("_p68")
            out[label] = float(value)
    return out


def _load_theta_raw(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "theta_raw" in payload.files:
            return np.asarray(payload["theta_raw"], dtype=np.float64)
        if "selected_theta_raw" in payload.files:
            return np.asarray(payload["selected_theta_raw"], dtype=np.float64)
    raise KeyError(f"{path} does not contain theta_raw or selected_theta_raw.")


def _load_completed_rounds(process_dir: Path, *, dim: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    round_paths = sorted(process_dir.glob("active_round_*.json"))
    if not round_paths:
        return np.empty((0, dim), dtype=np.float64), []
    rows: list[np.ndarray] = []
    rounds: list[dict[str, Any]] = []
    for expected, path in enumerate(round_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        round_index = int(payload.get("round_index", expected))
        if round_index != expected:
            raise ValueError(f"Non-consecutive round file {path}: {round_index} != {expected}")
        theta = np.asarray(payload.get("selected_theta_raw", []), dtype=np.float64)
        if theta.ndim != 2 or theta.shape[1] != int(dim):
            raise ValueError(f"Invalid selected_theta_raw shape in {path}: {theta.shape}")
        rows.extend(row.copy() for row in theta)
        rounds.append(payload)
    selected = np.vstack(rows).astype(np.float64) if rows else np.empty((0, dim), dtype=np.float64)
    return selected, rounds


def _write_registry(suite_dir: Path) -> None:
    records: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        path = _condition_summary_path(suite_dir, condition)
        if not path.exists():
            records.append({"order": CONDITION_ORDER.index(condition) + 1, "condition": condition, "status": "pending"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "order": CONDITION_ORDER.index(condition) + 1,
                "condition": condition,
                "status": payload.get("status", "unknown"),
                "initial_design": payload.get("initial_design", ""),
                "acquisition": payload.get("acquisition", ""),
                "training_points": payload.get("training_points", ""),
                "overall_p68": payload.get("overall_p68", ""),
                "overall_mean": payload.get("overall_mean", ""),
                "overall_p95": payload.get("overall_p95", ""),
                "overall_max": payload.get("overall_max", ""),
                "design_path": payload.get("design_path", ""),
                "summary_path": payload.get("summary_path", ""),
            }
        )
    _write_json(suite_dir / "ablation_registry.json", {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "records": records})
    csv_path = suite_dir / "ablation_registry.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_progress(path: Path, event: str, **payload: Any) -> None:
    row = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

