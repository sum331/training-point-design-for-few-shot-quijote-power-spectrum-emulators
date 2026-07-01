from __future__ import annotations

import argparse
import copy
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
from z2quijote.config import load_config
from z2quijote.direct_cdm import make_oracle
from z2quijote.experiment import (
    _comparison_block,
    _evaluate_design,
    _json_safe,
    _metric_policy,
    _save_active_design,
    _write_text_retry,
    _write_progress,
)
from z2quijote.raw_bank import load_raw_bank_sample
from z2quijote.reporting import generate_run_report
from z2quijote.resources import load_optional_current_active
from z2quijote.splits import load_split_bundle


def _load_completed_rounds(run_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    process_dir = run_dir / "process"
    if not process_dir.exists():
        raise FileNotFoundError(f"process directory not found: {process_dir}")
    round_records: list[dict[str, Any]] = []
    selected_rows: list[np.ndarray] = []
    for expected_index, path in enumerate(sorted(process_dir.glob("active_round_*.json"))):
        record = json.loads(path.read_text(encoding="utf-8"))
        round_index = int(record.get("round_index", expected_index))
        if round_index != expected_index:
            raise ValueError(f"non-consecutive round record {path}: {round_index} != {expected_index}")
        theta = np.asarray(record.get("selected_theta_raw", []), dtype=np.float64)
        if theta.ndim != 2 or theta.shape[0] <= 0:
            raise ValueError(f"round record has no selected theta rows: {path}")
        selected_rows.extend(row.copy() for row in theta)
        round_records.append(record)
    if not selected_rows:
        return np.empty((0, 0), dtype=np.float64), []
    return np.vstack(selected_rows).astype(np.float64), round_records


def resume_fair_comparison(*, config_path: Path, split_manifest: Path, run_dir: Path) -> dict[str, str]:
    config = load_config(config_path)
    bundle = load_split_bundle(split_manifest)
    oracle = make_oracle(config)
    k_bins = config.k_grid.k_bins
    run_dir = run_dir.resolve()
    process_dir = run_dir / "process"
    process_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    completed_theta, completed_rounds = _load_completed_rounds(run_dir)
    if completed_theta.size == 0:
        raise ValueError(f"no completed active rounds found in {process_dir}")
    target_total = int(config.active_learning.active_points)
    if completed_theta.shape[0] >= target_total:
        raise ValueError(
            f"run already has {completed_theta.shape[0]} completed active points; target is {target_total}."
        )
    _write_progress(
        progress_path,
        "resume_started",
        completed_active_points=int(completed_theta.shape[0]),
        target_total=target_total,
        split_manifest_path=str(bundle.manifest_path),
    )

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
            "target_total": int(event_payload.get("target_total", target_total)),
            "latest_round_index": event_payload.get("round_index"),
            "latest_stage": event_payload.get("stage"),
            "latest_stage_current": event_payload.get("current"),
            "latest_stage_total": event_payload.get("total"),
            "run_dir": str(run_dir),
            "process_dir": str(process_dir),
            "resumed_from_completed_active_points": int(completed_theta.shape[0]),
        }
        _write_text_retry(
            process_dir / "active_selection_state.json",
            json.dumps(_json_safe(state_payload), ensure_ascii=False, indent=2),
        )
        _write_progress(progress_path, event, **event_payload)

    active_result = select_z2_active_points(
        config=config,
        oracle=oracle,
        seed_theta_raw=bundle.seed_theta_raw,
        probe_theta_raw=bundle.probe_theta_raw,
        pool_theta_raw=bundle.pool_theta_raw,
        k_bins=k_bins,
        resume_selected_theta_raw=completed_theta,
        resume_rounds=completed_rounds,
        progress_callback=_active_progress,
    )
    _write_progress(
        progress_path,
        "active_selection_completed",
        active_points_selected=int(active_result.selected_theta_raw.shape[0]),
        resumed_active_points=int(completed_theta.shape[0]),
        newly_selected_active_points=int(active_result.selected_theta_raw.shape[0] - completed_theta.shape[0]),
    )

    seed_budget = int(bundle.seed_theta_raw.shape[0])
    active_budget = int(active_result.selected_theta_raw.shape[0])
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
        _write_progress(progress_path, "design_evaluation_completed", design_name=name, p68=float(metrics["p68"]))

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
        "comparison": _comparison_block(design_results),
        "metric_policy": _metric_policy(config),
        "resume": {
            "resumed": True,
            "completed_active_points_before_resume": int(completed_theta.shape[0]),
            "newly_selected_active_points": int(active_result.selected_theta_raw.shape[0] - completed_theta.shape[0]),
            "source_run_dir": str(run_dir),
        },
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
    summary["report_artifacts"] = copy.deepcopy(dict(report_manifest.get("artifacts", {})))
    summary["plot_paths"] = list(report_manifest.get("plot_paths", []))
    _write_text_retry(summary_path, json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    _write_progress(
        progress_path,
        "report_generated",
        report_manifest_path=str(report_manifest["report_manifest_path"]),
        plot_dir=str(report_manifest.get("artifacts", {}).get("plot_dir", "")),
        plot_count=len(report_manifest.get("plot_paths", [])),
    )
    return {"run_dir": str(run_dir), "summary_path": str(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume a z2 Quijote fair comparison from completed active_round JSON files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = resume_fair_comparison(
        config_path=Path(args.config),
        split_manifest=Path(args.split_manifest),
        run_dir=Path(args.run_dir),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
