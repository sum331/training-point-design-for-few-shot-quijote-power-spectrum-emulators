from __future__ import annotations

import argparse
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

from z2quijote.config import load_config
from z2quijote.direct_cdm import make_oracle
from z2quijote.emulator import PCAGPDirectCDMEmulator
from z2quijote.metrics import evaluate_prediction, metric_block, band_masks
from z2quijote.sampling import digest_theta, latin_hypercube_unit, sobol_unit, theta_rows_key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a Sobol-64 Quijote GP on a fresh LHS-256 validation set."
    )
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config.yaml"))
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    train_seed = int(args.train_seed if args.train_seed is not None else config.random_seed + 6401)
    validation_seed = int(
        args.validation_seed if args.validation_seed is not None else config.random_seed + 25691
    )
    run_dir = _make_run_dir(config, args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    progress_path = run_dir / "progress.jsonl"
    _write_progress(
        progress_path,
        "started",
        config_path=str(config.config_path),
        train_seed=train_seed,
        validation_seed=validation_seed,
    )

    k_bins = np.asarray(config.k_grid.k_bins, dtype=np.float64)
    train_unit = sobol_unit(
        int(args.train_size),
        config.parameter_space.dim,
        seed=train_seed,
    )
    train_theta = config.parameter_space.denormalize(train_unit)
    train_path = run_dir / f"sobol{int(args.train_size)}_training_points.npz"
    np.savez_compressed(
        train_path,
        theta_unit=train_unit.astype(np.float64),
        theta_raw=train_theta.astype(np.float64),
        theta_names=np.asarray(config.parameter_space.theta_names),
    )
    _write_progress(
        progress_path,
        "train_sobol_created",
        train_size=int(train_theta.shape[0]),
        train_path=str(train_path),
    )

    validation_unit = _fresh_lhs_validation_unit(
        validation_size=int(args.validation_size),
        dim=config.parameter_space.dim,
        seed=validation_seed,
        exclude=train_unit,
        decimals=config.splits.duplicate_decimals,
    )
    validation_theta = config.parameter_space.denormalize(validation_unit)
    validation_path = run_dir / "lhs256_validation_points.npz"
    np.savez_compressed(
        validation_path,
        theta_unit=validation_unit.astype(np.float64),
        theta_raw=validation_theta.astype(np.float64),
        theta_names=np.asarray(config.parameter_space.theta_names),
    )
    _write_progress(
        progress_path,
        "validation_lhs_created",
        validation_size=int(validation_theta.shape[0]),
        validation_path=str(validation_path),
    )

    oracle = make_oracle(config)
    _write_progress(progress_path, "train_truth_started", train_size=int(train_theta.shape[0]))
    train_truth = oracle.evaluate(train_theta, k_bins)
    _write_progress(progress_path, "train_truth_completed")

    emulator = PCAGPDirectCDMEmulator(
        config.parameter_space,
        config.model,
        target_kind=str(config.target.kind),
    ).fit(train_theta, train_truth.log_pk, k_bins)
    _write_progress(progress_path, "fixed_gp_fit_completed", emulator=emulator.metadata)

    _write_progress(progress_path, "validation_truth_started", validation_size=int(validation_theta.shape[0]))
    validation_truth = oracle.evaluate(validation_theta, k_bins)
    _write_progress(progress_path, "validation_truth_completed")

    prediction = emulator.predict(validation_theta)
    relative_error = np.abs(np.exp(prediction.log_pk_mean - validation_truth.log_pk) - 1.0)
    kwise_p68_relative_error = np.percentile(relative_error, 68.0, axis=0)
    metrics = evaluate_prediction(
        truth_log_pk=validation_truth.log_pk,
        pred_log_pk=prediction.log_pk_mean,
        k_bins=k_bins,
        band_edges=config.evaluation.band_edges,
        band_labels=config.evaluation.band_labels,
        target_kind=str(config.target.kind),
    )
    signed_bias = _signed_bias_metrics(
        truth_log=validation_truth.log_pk,
        pred_log=prediction.log_pk_mean,
        k_bins=k_bins,
        band_edges=config.evaluation.band_edges,
        band_labels=config.evaluation.band_labels,
    )
    prediction_path = run_dir / "fixed64_lhs256_predictions.npz"
    np.savez_compressed(
        prediction_path,
        train_theta_raw=train_theta.astype(np.float64),
        train_theta_unit=train_unit.astype(np.float64),
        validation_theta_raw=validation_theta.astype(np.float64),
        validation_theta_unit=validation_unit.astype(np.float64),
        k_bins=k_bins.astype(np.float64),
        truth_target=validation_truth.log_pk.astype(np.float64),
        pred_target=prediction.log_pk_mean.astype(np.float64),
        kwise_p68_relative_error=kwise_p68_relative_error.astype(np.float64),
        signed_relative_bias=(np.exp(prediction.log_pk_mean - validation_truth.log_pk) - 1.0).astype(np.float64),
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "sobol64_lhs256_quijote_bias",
        "config_path": str(config.config_path),
        "target_kind": str(config.target.kind),
        "anchor_mode": str(config.target.anchor_mode),
        "parameter_space": {
            "name": str(config.parameter_space.name),
            "theta_names": list(config.parameter_space.theta_names),
            "theta_bounds": config.parameter_space.theta_bounds.astype(float).tolist(),
        },
        "k_grid": {
            "k_min": float(k_bins[0]),
            "k_max": float(k_bins[-1]),
            "k_count": int(k_bins.shape[0]),
            "bands": [
                {
                    "name": str(band.name),
                    "k_min": float(band.k_min),
                    "k_max": float(band.k_max),
                    "count": int(band.count),
                }
                for band in config.k_grid.bands
            ],
        },
        "train": {
            "size": int(train_theta.shape[0]),
            "sampler": "sobol",
            "seed": int(train_seed),
            "theta_digest": digest_theta(train_unit, decimals=config.splits.duplicate_decimals),
            "path": str(train_path),
        },
        "validation": {
            "size": int(validation_theta.shape[0]),
            "sampler": "latin_hypercube",
            "seed": int(validation_seed),
            "theta_digest": digest_theta(validation_unit, decimals=config.splits.duplicate_decimals),
            "path": str(validation_path),
        },
        "emulator": emulator.metadata,
        "metric_policy": _metric_policy(config),
        "primary_result": _primary_result(
            config=config,
            metrics=metrics,
            k_bins=k_bins,
            kwise_p68_relative_error=kwise_p68_relative_error,
        ),
        "auxiliary_metrics": metrics,
        "auxiliary_signed_bias": signed_bias,
        "prediction_path": str(prediction_path),
    }
    summary_path = run_dir / "sobol64_lhs256_quijote_bias_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_progress(progress_path, "summary_written", summary_path=str(summary_path))
    print(json.dumps({"run_dir": str(run_dir), "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


def _fresh_lhs_validation_unit(
    *,
    validation_size: int,
    dim: int,
    seed: int,
    exclude: np.ndarray,
    decimals: int,
) -> np.ndarray:
    seen = set(theta_rows_key(exclude, decimals=decimals))
    rows: list[np.ndarray] = []
    attempts = 0
    while len(rows) < int(validation_size):
        draw = latin_hypercube_unit(
            max(int(validation_size) - len(rows), 32),
            dim,
            seed=int(seed + 1009 * attempts),
        )
        for row, key in zip(draw, theta_rows_key(draw, decimals=decimals), strict=True):
            if key in seen:
                continue
            seen.add(key)
            rows.append(row.astype(np.float64))
            if len(rows) >= int(validation_size):
                break
        attempts += 1
        if attempts > 128:
            raise RuntimeError("Could not draw enough fresh LHS validation points.")
    return np.vstack(rows).astype(np.float64)


def _signed_bias_metrics(
    *,
    truth_log: np.ndarray,
    pred_log: np.ndarray,
    k_bins: np.ndarray,
    band_edges: tuple[float, ...],
    band_labels: tuple[str, ...],
) -> dict[str, Any]:
    signed = np.exp(np.asarray(pred_log, dtype=np.float64) - np.asarray(truth_log, dtype=np.float64)) - 1.0
    log_signed = np.asarray(pred_log, dtype=np.float64) - np.asarray(truth_log, dtype=np.float64)
    result: dict[str, Any] = {
        "overall_signed_relative_bias": _signed_metric_block(signed),
        "overall_signed_log_bias": _signed_metric_block(log_signed),
        "sample_p68_abs_relative_bias": np.percentile(np.abs(signed), 68.0, axis=1).astype(float).tolist(),
        "sample_mean_signed_relative_bias": np.mean(signed, axis=1).astype(float).tolist(),
        "bands": {},
    }
    for label, mask in band_masks(k_bins, band_edges, band_labels).items():
        if not np.any(mask):
            result["bands"][label] = {"bin_count": 0}
            continue
        result["bands"][label] = {
            "bin_count": int(np.count_nonzero(mask)),
            "signed_relative_bias": _signed_metric_block(signed[:, mask]),
            "signed_log_bias": _signed_metric_block(log_signed[:, mask]),
            "abs_relative_bias": metric_block(np.abs(signed[:, mask])),
        }
    return result


def _metric_policy(config: Any) -> dict[str, Any]:
    return {
        "primary_metric": str(config.evaluation.primary_metric),
        "primary_curve": str(config.evaluation.primary_curve),
        "report_metric_policy": str(config.evaluation.report_metric_policy),
        "reported_quantiles": [68.0],
        "auxiliary_only": ["p50", "p95", "mean", "max", "signed_bias"],
    }


def _primary_result(
    *,
    config: Any,
    metrics: dict[str, Any],
    k_bins: np.ndarray,
    kwise_p68_relative_error: np.ndarray,
) -> dict[str, Any]:
    band_p68: dict[str, float] = {}
    for label in config.evaluation.band_labels:
        band = metrics.get("bands", {}).get(str(label), {})
        if isinstance(band, dict) and "relative_error" in band:
            band_p68[str(label)] = float(band["relative_error"]["p68"])
    return {
        "overall_p68_relative_error": float(metrics["overall_relative_error"]["p68"]),
        "band_p68_relative_error": band_p68,
        "curve": {
            "name": str(config.evaluation.primary_curve),
            "quantile": 68.0,
            "k_min": float(np.asarray(k_bins)[0]),
            "k_max": float(np.asarray(k_bins)[-1]),
            "k_count": int(np.asarray(k_bins).shape[0]),
            "min": float(np.min(kwise_p68_relative_error)),
            "max": float(np.max(kwise_p68_relative_error)),
        },
    }


def _signed_metric_block(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "p16": float("nan"),
            "p50": float("nan"),
            "p84": float("nan"),
            "p95": float("nan"),
            "max_abs": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "p16": float(np.percentile(arr, 16.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p84": float(np.percentile(arr, 84.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max_abs": float(np.max(np.abs(arr))),
    }


def _make_run_dir(config: Any, output_dir: str | None) -> Path:
    root = Path(output_dir).resolve() if output_dir else Path(config.evaluation.output_dir).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "diagnostics" / f"sobol64_lhs256_quijote_bias_{timestamp}"


def _write_progress(progress_file: Path, event: str, **payload: Any) -> None:
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with progress_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


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


if __name__ == "__main__":
    raise SystemExit(main())
