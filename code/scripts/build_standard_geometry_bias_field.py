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
from z2quijote.sampling import digest_theta
from z2quijote.standard_geometry import (
    BiasAccumulator,
    StandardGeometryConfig,
    density_from_bias,
    draw_design_unit,
    draw_reference_unit,
)
from z2quijote.standard_geometry.geometry import accepted_mask, compute_geometry_batch, thresholds_from_geometry
from z2quijote.standard_geometry.interpolation import ReliabilityWeightedLocalInterpolator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standard-geometry conditional bias field for z2/PPR. "
            "Defaults implement M=4096, N=64, S=600, tau_lambda=0.16."
        )
    )
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config.yaml"))
    parser.add_argument("--reference-size", type=int, default=4096)
    parser.add_argument("--design-size", type=int, default=64)
    parser.add_argument("--design-count", type=int, default=600)
    parser.add_argument("--tau-lambda", type=float, default=0.16)
    parser.add_argument("--sampler", default="mixed", choices=["mixed", "sobol", "lhs", "latin_hypercube"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--observe-bias", action="store_true", help="Train emulators and compute accepted bias means.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--usable-min-count", type=int, default=10)
    parser.add_argument("--high-confidence-min-count", type=int, default=20)
    parser.add_argument("--density-alpha", type=float, default=1.0)
    parser.add_argument("--density-clip-quantile", type=float, default=0.95)
    parser.add_argument("--reference-chunk-size", type=int, default=512)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.random_seed + 330016)
    run_dir = _make_run_dir(config, args.output_dir, observe_bias=bool(args.observe_bias))
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    _write_progress(progress_path, "started", args=vars(args), config_path=str(config.config_path))

    sg_config = StandardGeometryConfig(tau_lambda=float(args.tau_lambda))
    reference_unit = draw_reference_unit(
        count=int(args.reference_size),
        dim=config.parameter_space.dim,
        seed=seed,
    )
    reference_theta = config.parameter_space.denormalize(reference_unit)
    np.savez_compressed(
        run_dir / "reference_theta.npz",
        theta_unit=reference_unit.astype(np.float64),
        theta_raw=reference_theta.astype(np.float64),
        theta_names=np.asarray(config.parameter_space.theta_names),
    )
    _write_progress(progress_path, "reference_created", reference_size=int(reference_unit.shape[0]))

    designs = [
        draw_design_unit(
            design_size=int(args.design_size),
            dim=config.parameter_space.dim,
            seed=seed + 10000,
            index=index,
            sampler=str(args.sampler),
        )
        for index in range(int(args.design_count))
    ]
    np.savez_compressed(
        run_dir / "training_designs_unit.npz",
        theta_unit=np.asarray(designs, dtype=np.float64),
        sampler=str(args.sampler),
        design_size=int(args.design_size),
        design_count=int(args.design_count),
    )
    _write_progress(progress_path, "designs_created", design_count=len(designs), design_size=int(args.design_size))

    geometry_batches = []
    for index, design in enumerate(designs):
        geometry_batches.append(compute_geometry_batch(reference_unit, design))
        if (index + 1) % 25 == 0 or index + 1 == len(designs):
            _write_progress(progress_path, "geometry_pass_progress", completed=index + 1, total=len(designs))
    thresholds = thresholds_from_geometry(geometry_batches, sg_config)
    _write_progress(progress_path, "geometry_thresholds_ready", thresholds=thresholds.as_dict())

    accepted_counts = np.zeros(reference_unit.shape[0], dtype=np.int64)
    inside_counts = np.zeros(reference_unit.shape[0], dtype=np.int64)
    for batch in geometry_batches:
        inside_counts += batch.finite_geometry.astype(np.int64)
        accepted_counts += accepted_mask(batch, thresholds).astype(np.int64)
    geometry_summary = _geometry_summary(
        accepted_counts=accepted_counts,
        inside_counts=inside_counts,
        design_count=len(designs),
        thresholds=thresholds.as_dict(),
        args=vars(args),
    )
    (run_dir / "geometry_summary.json").write_text(
        json.dumps(_json_safe(geometry_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "geometry_counts.npz",
        accepted_count=accepted_counts.astype(np.int64),
        inside_count=inside_counts.astype(np.int64),
        accepted_fraction=(accepted_counts / max(len(designs), 1)).astype(np.float64),
        inside_fraction=(inside_counts / max(len(designs), 1)).astype(np.float64),
    )
    _write_progress(progress_path, "geometry_summary_written")

    if not args.observe_bias:
        print(json.dumps({"run_dir": str(run_dir), "geometry_summary": geometry_summary}, ensure_ascii=False, indent=2))
        return 0

    oracle = make_oracle(config)
    k_bins = np.asarray(config.k_grid.k_bins, dtype=np.float64)
    support_indices = np.flatnonzero(accepted_counts > 0)
    support_lookup = np.full(reference_unit.shape[0], -1, dtype=np.int64)
    support_lookup[support_indices] = np.arange(support_indices.shape[0], dtype=np.int64)
    np.savez_compressed(
        run_dir / "accepted_reference_support.npz",
        support_indices=support_indices.astype(np.int64),
        theta_unit=reference_unit[support_indices].astype(np.float64),
        theta_raw=reference_theta[support_indices].astype(np.float64),
        accepted_count=accepted_counts[support_indices].astype(np.int64),
    )
    _write_progress(
        progress_path,
        "reference_truth_started",
        reference_size=int(reference_theta.shape[0]),
        support_size=int(support_indices.shape[0]),
        skipped_zero_acceptance=int(reference_theta.shape[0] - support_indices.shape[0]),
        bias_evaluation="accepted_center_points_only",
    )
    reference_truth = oracle.evaluate(reference_theta[support_indices], k_bins)
    np.savez_compressed(
        run_dir / "accepted_reference_truth_log.npz",
        support_indices=support_indices.astype(np.int64),
        theta_unit=reference_unit[support_indices].astype(np.float64),
        theta_raw=reference_theta[support_indices].astype(np.float64),
        k_bins=k_bins.astype(np.float64),
        log_pk=reference_truth.log_pk.astype(np.float64),
    )
    _write_progress(progress_path, "reference_truth_completed", support_size=int(support_indices.shape[0]))

    accumulator = BiasAccumulator(reference_size=reference_unit.shape[0])
    observed_pair_count = 0
    for index, (design_unit, batch) in enumerate(zip(designs, geometry_batches)):
        accepted = accepted_mask(batch, thresholds)
        accepted_indices = np.flatnonzero(accepted)
        if accepted_indices.size <= 0:
            accumulator.add_indices(accepted_indices, np.empty((0,), dtype=np.float64))
            if (index + 1) % 10 == 0 or index + 1 == len(designs):
                _write_progress(
                    progress_path,
                    "bias_pass_progress",
                    completed=index + 1,
                    total=len(designs),
                    accepted_pairs_done=int(observed_pair_count),
                    last_accepted_count=0,
                )
            continue
        design_theta = config.parameter_space.denormalize(design_unit)
        train_truth = oracle.evaluate(design_theta, k_bins)
        emulator = PCAGPDirectCDMEmulator(
            config.parameter_space,
            config.model,
            target_kind=str(config.target.kind),
        ).fit(design_theta, train_truth.log_pk, k_bins)
        support_rows = support_lookup[accepted_indices]
        if np.any(support_rows < 0):
            raise RuntimeError("Accepted reference index was not present in the reference support table.")
        bias = _reference_bias_in_chunks(
            emulator=emulator,
            reference_theta=reference_theta[accepted_indices],
            reference_truth_log=reference_truth.log_pk[support_rows],
            chunk_size=int(args.reference_chunk_size),
        )
        accumulator.add_indices(accepted_indices, bias)
        observed_pair_count += int(accepted_indices.shape[0])
        if (index + 1) % 10 == 0 or index + 1 == len(designs):
            _write_progress(
                progress_path,
                "bias_pass_progress",
                completed=index + 1,
                total=len(designs),
                accepted_pairs_done=int(observed_pair_count),
                last_accepted_count=int(accepted_indices.shape[0]),
            )

    estimate = accumulator.estimate(
        usable_min_count=int(args.usable_min_count),
        high_confidence_min_count=int(args.high_confidence_min_count),
    )
    bias_path = run_dir / "standard_geometry_bias_field.npz"
    np.savez_compressed(
        bias_path,
        theta_unit=reference_unit.astype(np.float64),
        theta_raw=reference_theta.astype(np.float64),
        theta_names=np.asarray(config.parameter_space.theta_names),
        **estimate.as_npz_payload(),
    )

    density = density_from_bias(
        estimate.bias_mean,
        alpha=float(args.density_alpha),
        clip_quantile=float(args.density_clip_quantile),
    )
    np.savez_compressed(
        run_dir / "density_field.npz",
        theta_unit=reference_unit.astype(np.float64),
        theta_raw=reference_theta.astype(np.float64),
        density=density.astype(np.float64),
        bias_mean=estimate.bias_mean.astype(np.float64),
        accepted_count=estimate.accepted_count.astype(np.int64),
    )

    interpolator_payload_path = run_dir / "interpolator_support.npz"
    interpolator = ReliabilityWeightedLocalInterpolator(
        reference_unit,
        estimate.bias_mean,
        estimate.accepted_count,
        min_count=int(args.usable_min_count),
        high_confidence_count=int(args.high_confidence_min_count),
    )
    interp_self, interp_conf = interpolator.predict(reference_unit)
    np.savez_compressed(
        interpolator_payload_path,
        theta_unit=reference_unit.astype(np.float64),
        bias_mean=estimate.bias_mean.astype(np.float64),
        accepted_count=estimate.accepted_count.astype(np.int64),
        self_interpolated_bias=interp_self.astype(np.float64),
        self_interpolation_confidence=interp_conf.astype(np.float64),
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "standard_geometry_bias_field",
        "config_path": str(config.config_path),
        "target_kind": str(config.target.kind),
        "parameter_space": {
            "name": str(config.parameter_space.name),
            "theta_names": list(config.parameter_space.theta_names),
            "theta_bounds": config.parameter_space.theta_bounds.astype(float).tolist(),
        },
        "reference_size": int(reference_unit.shape[0]),
        "design_size": int(args.design_size),
        "design_count": int(args.design_count),
        "tau_lambda": float(args.tau_lambda),
        "thresholds": thresholds.as_dict(),
        "geometry": geometry_summary,
        "bias_evaluation": {
            "mode": "accepted_center_points_only",
            "reference_support_size": int(support_indices.shape[0]),
            "observed_pair_count": int(observed_pair_count),
            "zero_acceptance_reference_count": int(reference_unit.shape[0] - support_indices.shape[0]),
        },
        "bias_estimate": {
            "usable_min_count": int(args.usable_min_count),
            "high_confidence_min_count": int(args.high_confidence_min_count),
            "usable_fraction": float(np.mean(estimate.usable)),
            "high_confidence_fraction": float(np.mean(estimate.high_confidence)),
            "finite_bias_fraction": float(np.mean(np.isfinite(estimate.bias_mean))),
            "accepted_count_quantiles": _quantiles(estimate.accepted_count),
            "bias_mean_quantiles": _quantiles(estimate.bias_mean[np.isfinite(estimate.bias_mean)]),
        },
        "artifacts": {
            "bias_field": str(bias_path),
            "density_field": str(run_dir / "density_field.npz"),
            "interpolator_support": str(interpolator_payload_path),
        },
        "reference_digest": digest_theta(reference_unit, decimals=config.splits.duplicate_decimals),
    }
    summary_path = run_dir / "standard_geometry_bias_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_progress(progress_path, "completed", summary_path=str(summary_path))
    print(json.dumps({"run_dir": str(run_dir), "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


def _sample_p68_relative_bias(truth_log: np.ndarray, pred_log: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth_log, dtype=np.float64)
    pred = np.asarray(pred_log, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 2:
        raise ValueError("truth_log and pred_log must have matching shape [N,K].")
    relative = np.abs(np.exp(pred - truth) - 1.0)
    return np.percentile(relative, 68.0, axis=1).astype(np.float64)


def _reference_bias_in_chunks(
    *,
    emulator: PCAGPDirectCDMEmulator,
    reference_theta: np.ndarray,
    reference_truth_log: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    theta = np.asarray(reference_theta, dtype=np.float64)
    truth = np.asarray(reference_truth_log, dtype=np.float64)
    if theta.ndim != 2 or truth.ndim != 2 or theta.shape[0] != truth.shape[0]:
        raise ValueError("reference_theta and reference_truth_log must have aligned first dimension.")
    chunk = int(max(1, chunk_size))
    rows: list[np.ndarray] = []
    for start in range(0, theta.shape[0], chunk):
        stop = min(start + chunk, theta.shape[0])
        prediction = emulator.predict(theta[start:stop])
        rows.append(_sample_p68_relative_bias(truth[start:stop], prediction.log_pk_mean))
    return np.concatenate(rows).astype(np.float64)


def _make_run_dir(config: Any, output_dir: str | None, *, observe_bias: bool) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "bias" if observe_bias else "geometry"
    return config.data_root / "standard_geometry_bias" / f"{mode}_{stamp}"


def _geometry_summary(
    *,
    accepted_counts: np.ndarray,
    inside_counts: np.ndarray,
    design_count: int,
    thresholds: dict[str, float],
    args: dict[str, Any],
) -> dict[str, Any]:
    accepted = np.asarray(accepted_counts, dtype=np.int64)
    inside = np.asarray(inside_counts, dtype=np.int64)
    return {
        "reference_size": int(accepted.shape[0]),
        "design_count": int(design_count),
        "accepted_total": int(np.sum(accepted)),
        "inside_total": int(np.sum(inside)),
        "accepted_pair_fraction": float(np.sum(accepted) / max(int(design_count) * accepted.shape[0], 1)),
        "inside_pair_fraction": float(np.sum(inside) / max(int(design_count) * inside.shape[0], 1)),
        "accepted_count_quantiles": _quantiles(accepted),
        "inside_count_quantiles": _quantiles(inside),
        "nonzero_accepted_fraction": float(np.mean(accepted > 0)),
        "ge10_accepted_fraction": float(np.mean(accepted >= 10)),
        "ge20_accepted_fraction": float(np.mean(accepted >= 20)),
        "thresholds": thresholds,
        "parameters": {
            "reference_size": int(args["reference_size"]),
            "design_size": int(args["design_size"]),
            "design_count": int(args["design_count"]),
            "tau_lambda": float(args["tau_lambda"]),
            "sampler": str(args["sampler"]),
        },
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return {}
    probs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    qs = np.quantile(arr, probs)
    return {f"q{int(prob * 100):02d}": float(value) for prob, value in zip(probs, qs)}


def _write_progress(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": str(event), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
