from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from z2quijote.acquisition import select_z2_active_points
from z2quijote.config import Z2Config, load_config
from z2quijote.direct_cdm import make_oracle
from z2quijote.m3_adapter import _load_runtime_config
from z2quijote.splits import build_split_bundle


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _stage_durations(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stages: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") != "m3_progress":
            continue
        stage = str(event.get("stage", "unknown"))
        elapsed = float(event.get("elapsed_seconds", 0.0))
        item = stages.setdefault(
            stage,
            {
                "first_elapsed_seconds": elapsed,
                "last_elapsed_seconds": elapsed,
                "updates": 0,
                "last_current": None,
                "last_total": None,
            },
        )
        item["last_elapsed_seconds"] = elapsed
        item["updates"] = int(item["updates"]) + 1
        item["last_current"] = int(event.get("current", 0))
        item["last_total"] = int(event.get("total", 0))
    for item in stages.values():
        item["duration_seconds"] = float(item["last_elapsed_seconds"] - item["first_elapsed_seconds"])
    return stages


def _build_train64(config: Z2Config) -> tuple[np.ndarray, dict[str, Any]]:
    bundle = build_split_bundle(config, force=True)
    seed = np.asarray(bundle.seed_theta_raw, dtype=np.float64)
    need = 64 - int(seed.shape[0])
    if need < 0:
        train = seed[:64].copy()
        source = "seed_prefix64"
    elif need == 0:
        train = seed.copy()
        source = "seed_exact64"
    else:
        tail = np.asarray(bundle.sobol_tail_theta_raw, dtype=np.float64)
        if tail.shape[0] < need:
            raise ValueError(f"Need {need} Sobol tail rows to build train64, got {tail.shape[0]}.")
        train = np.vstack([seed, tail[:need]]).astype(np.float64)
        source = "seed_plus_sobol_tail"
    return train, {
        "source": source,
        "seed_size": int(seed.shape[0]),
        "sobol_tail_used": int(max(0, need)),
        "train_size": int(train.shape[0]),
        "split_manifest_path": str(bundle.manifest_path),
        "split_arrays_path": str(bundle.arrays_path),
        "probe_size": int(bundle.probe_theta_raw.shape[0]),
        "pool_size": int(bundle.pool_theta_raw.shape[0]),
    }


def _runtime_m3_summary() -> dict[str, Any]:
    runtime = _load_runtime_config()
    return {
        "runtime_config_path": str(PACKAGE_ROOT / "configs" / "module3_runtime.yaml"),
        "device": str(getattr(runtime, "device", "unknown")),
        "chunk_size": int(getattr(runtime.m3, "chunk_size", 0)),
        "stage0_chunk_size": int(getattr(runtime.m3, "stage0_chunk_size", 0)),
        "stage_qmc_keep_fraction": float(getattr(runtime.m3, "hierarchical_stage1_refine_fraction", 0.0)),
        "stage3_refine_fraction": float(getattr(runtime.m3, "stage3_refine_fraction", 0.0)),
        "stage_qmc_sample_count": int(getattr(runtime.m3, "stage3_qmc_sample_count", 0)),
        "stage_qmc_chunk_size": int(getattr(runtime.m3, "stage3_qmc_chunk_size", 0)),
        "polish_starts_per_simplex": int(getattr(runtime.m3, "polish_starts_per_simplex_refine", 0)),
        "polish_max_iter": int(getattr(runtime.m3, "polish_max_iter_refine", 0)),
        "stage3_qmc_top_k": int(getattr(runtime.m3, "stage3_qmc_top_k", 0)),
    }


def run_pressure_test(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    config = replace(
        config,
        active_learning=replace(
            config.active_learning,
            active_points=1,
            batch_size=1,
            candidate_source="m3",
            sobol_tail_reserve=0,
        ),
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else config.data_root / "pressure_tests"
    run_dir = out_dir / f"m3_single64_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    summary_path = run_dir / "summary.json"

    train64, train_meta = _build_train64(config)
    oracle = make_oracle(config)
    start_time = time.perf_counter()
    events: list[dict[str, Any]] = []

    def progress_callback(payload: dict[str, Any]) -> None:
        event = dict(payload)
        event["created_at_utc"] = _utc_now()
        event["elapsed_seconds"] = float(time.perf_counter() - start_time)
        events.append(event)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")

    initial_summary = {
        "status": "running",
        "created_at_utc": _utc_now(),
        "run_dir": str(run_dir),
        "config_path": str(Path(args.config).resolve()),
        "k_count": int(config.k_grid.k_bins.shape[0]),
        "k_min": float(config.k_grid.k_bins[0]),
        "k_max": float(config.k_grid.k_bins[-1]),
        "train64": train_meta,
        "fastmock_bias": {
            "enabled": bool(config.fastmock_bias.enabled),
            "truth_backend": str(config.fastmock_bias.truth_backend),
            "truth_dtype": str(config.fastmock_bias.truth_dtype),
            "truth_chunk_size": int(config.fastmock_bias.truth_chunk_size),
        },
        "m3_runtime": _runtime_m3_summary(),
    }
    _write_json(summary_path, initial_summary)

    try:
        result = select_z2_active_points(
            config=config,
            oracle=oracle,
            seed_theta_raw=train64,
            probe_theta_raw=np.empty((0, config.parameter_space.dim), dtype=np.float64),
            pool_theta_raw=np.empty((0, config.parameter_space.dim), dtype=np.float64),
            k_bins=config.k_grid.k_bins,
            progress_callback=progress_callback,
        )
        total_seconds = float(time.perf_counter() - start_time)
        summary = dict(initial_summary)
        summary.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "total_seconds": total_seconds,
                "total_minutes": total_seconds / 60.0,
                "selected_theta_raw": result.selected_theta_raw.astype(np.float64).tolist(),
                "selected_pool_indices": result.selected_pool_indices.astype(np.int64).tolist(),
                "active_report": result.report,
                "stage_durations": _stage_durations(events),
                "progress_path": str(progress_path),
                "summary_path": str(summary_path),
            }
        )
        _write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        return summary
    except BaseException as exc:
        total_seconds = float(time.perf_counter() - start_time)
        summary = dict(initial_summary)
        summary.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "total_seconds": total_seconds,
                "total_minutes": total_seconds / 60.0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "stage_durations": _stage_durations(events),
                "progress_path": str(progress_path),
                "summary_path": str(summary_path),
            }
        )
        _write_json(summary_path, summary)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one full z2 M3 pressure test from a 64-point train state.")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config.yaml"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    run_pressure_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
