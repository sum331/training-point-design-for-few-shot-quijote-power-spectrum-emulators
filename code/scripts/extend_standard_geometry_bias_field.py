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

from build_standard_geometry_bias_field import (  # noqa: E402
    _geometry_summary,
    _json_safe,
    _quantiles,
    _reference_bias_in_chunks,
    _write_progress,
)
from z2quijote.config import load_config  # noqa: E402
from z2quijote.csst_fastmock import CSSTQuijote5DOracle  # noqa: E402
from z2quijote.direct_cdm import make_oracle  # noqa: E402
from z2quijote.emulator import PCAGPDirectCDMEmulator  # noqa: E402
from z2quijote.sampling import digest_theta  # noqa: E402
from z2quijote.standard_geometry import (  # noqa: E402
    StandardGeometryConfig,
    density_from_bias,
    draw_design_unit,
    draw_reference_unit,
)
from z2quijote.standard_geometry.geometry import (  # noqa: E402
    GeometryThresholds,
    accepted_mask,
    compute_geometry_batch,
)
from z2quijote.standard_geometry.interpolation import ReliabilityWeightedLocalInterpolator  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build or resume a denser standard-geometry bias field. "
            "This script is checkpointed and intended for long M/S runs."
        )
    )
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "config.yaml"))
    parser.add_argument("--source-run-dir", default=None)
    parser.add_argument("--reference-size", type=int, default=32768)
    parser.add_argument("--design-size", type=int, default=64)
    parser.add_argument("--design-count", type=int, default=900)
    parser.add_argument("--tau-lambda", type=float, default=0.16)
    parser.add_argument("--sampler", default="mixed", choices=["mixed", "sobol", "lhs", "latin_hypercube"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--usable-min-count", type=int, default=10)
    parser.add_argument("--high-confidence-min-count", type=int, default=20)
    parser.add_argument("--density-alpha", type=float, default=1.0)
    parser.add_argument("--density-clip-quantile", type=float, default=0.95)
    parser.add_argument("--reference-chunk-size", type=int, default=512)
    parser.add_argument("--truth-chunk-size", type=int, default=0)
    parser.add_argument("--bias-truth-source", choices=["csst_residual", "z2_logdiff"], default="csst_residual")
    parser.add_argument("--prefer-cuda-reference-truth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-cuda-train-truth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--state-save-interval", type=int, default=10)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    bias_truth_source = str(args.bias_truth_source).strip().lower()
    truth_chunk_size = int(args.truth_chunk_size)
    if truth_chunk_size <= 0:
        truth_chunk_size = (
            int(config.fastmock_bias.truth_chunk_size)
            if bias_truth_source == "csst_residual"
            else 64
        )
    args_payload = vars(args).copy()
    args_payload["resolved_truth_chunk_size"] = int(truth_chunk_size)
    seed = int(args.seed if args.seed is not None else config.random_seed + 330016)
    run_dir = _make_run_dir(
        config=config,
        output_dir=args.output_dir,
        reference_size=int(args.reference_size),
        design_size=int(args.design_size),
        design_count=int(args.design_count),
        tau_lambda=float(args.tau_lambda),
        k_count=len(config.k_grid.k_bins),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.jsonl"
    _write_progress(progress_path, "started", args=args_payload, config_path=str(config.config_path), run_dir=str(run_dir))

    reference_unit, reference_theta, reference_info = _prepare_reference(
        config=config,
        seed=seed,
        reference_size=int(args.reference_size),
        run_dir=run_dir,
        source_run_dir=Path(args.source_run_dir).resolve() if args.source_run_dir else None,
        progress_path=progress_path,
    )
    designs, design_info = _prepare_designs(
        config=config,
        seed=seed,
        design_size=int(args.design_size),
        design_count=int(args.design_count),
        sampler=str(args.sampler),
        run_dir=run_dir,
        source_run_dir=Path(args.source_run_dir).resolve() if args.source_run_dir else None,
        progress_path=progress_path,
    )

    thresholds = _prepare_thresholds(
        reference_unit=reference_unit,
        designs=designs,
        sg_config=StandardGeometryConfig(tau_lambda=float(args.tau_lambda)),
        run_dir=run_dir,
        progress_path=progress_path,
    )
    accepted_counts, inside_counts, offsets, accepted_indices_all = _prepare_accepted_indices(
        reference_unit=reference_unit,
        designs=designs,
        thresholds=thresholds,
        run_dir=run_dir,
        progress_path=progress_path,
        args_payload=args_payload,
    )

    bias_oracle, training_target_kind = _make_bias_oracle(config, truth_source=bias_truth_source)
    k_bins = np.asarray(config.k_grid.k_bins, dtype=np.float64)
    _write_progress(
        progress_path,
        "bias_truth_source_resolved",
        truth_source=bias_truth_source,
        training_target_kind=training_target_kind,
        reference_truth_chunk_size=int(truth_chunk_size),
        prefer_cuda_reference_truth=bool(args.prefer_cuda_reference_truth),
        prefer_cuda_train_truth=bool(args.prefer_cuda_train_truth),
    )
    support_indices, reference_truth_log = _prepare_reference_truth(
        oracle=bias_oracle,
        truth_source=bias_truth_source,
        reference_unit=reference_unit,
        reference_theta=reference_theta,
        accepted_counts=accepted_counts,
        k_bins=k_bins,
        run_dir=run_dir,
        progress_path=progress_path,
        chunk_size=int(truth_chunk_size),
        prefer_cuda_truth=bool(args.prefer_cuda_reference_truth),
    )
    support_lookup = np.full(reference_unit.shape[0], -1, dtype=np.int64)
    support_lookup[support_indices] = np.arange(support_indices.shape[0], dtype=np.int64)

    state = _run_bias_pass(
        config=config,
        oracle=bias_oracle,
        truth_source=bias_truth_source,
        training_target_kind=training_target_kind,
        prefer_cuda_train_truth=bool(args.prefer_cuda_train_truth),
        k_bins=k_bins,
        reference_theta=reference_theta,
        reference_truth_log=reference_truth_log,
        support_lookup=support_lookup,
        designs=designs,
        offsets=offsets,
        accepted_indices_all=accepted_indices_all,
        chunk_size=int(args.reference_chunk_size),
        save_interval=int(args.state_save_interval),
        run_dir=run_dir,
        progress_path=progress_path,
    )

    summary_path = _write_final_artifacts(
        config=config,
        run_dir=run_dir,
        reference_unit=reference_unit,
        reference_theta=reference_theta,
        theta_names=np.asarray(config.parameter_space.theta_names),
        state=state,
        design_count=int(args.design_count),
        usable_min_count=int(args.usable_min_count),
        high_confidence_min_count=int(args.high_confidence_min_count),
        density_alpha=float(args.density_alpha),
        density_clip_quantile=float(args.density_clip_quantile),
        geometry_summary=_geometry_summary(
            accepted_counts=accepted_counts,
            inside_counts=inside_counts,
            design_count=int(args.design_count),
            thresholds=thresholds.as_dict(),
            args=vars(args),
        ),
        observed_pair_count=int(state["observed_pair_count"]),
        source_info={
            "source_run_dir": str(Path(args.source_run_dir).resolve()) if args.source_run_dir else None,
            "source_bias_statistics_reused": False,
            "source_bias_statistics_reuse_note": (
                "The source run can define the reference/design prefix only. "
                "Bias statistics are recomputed under the current config/k-grid."
            ),
            "bias_truth_source": bias_truth_source,
            "training_target_kind": training_target_kind,
            "reference_prefix": reference_info,
            "design_prefix": design_info,
        },
        args_payload=args_payload,
    )
    _write_progress(progress_path, "completed", summary_path=str(summary_path))
    print(json.dumps({"run_dir": str(run_dir), "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


def _make_run_dir(
    *,
    config: Any,
    output_dir: str | None,
    reference_size: int,
    design_size: int,
    design_count: int,
    tau_lambda: float,
    k_count: int,
) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tau_slug = f"{tau_lambda:.4g}".replace(".", "p")
    name = f"bias_m{reference_size}_n{design_size}_s{design_count}_tau{tau_slug}_k{k_count}_{stamp}"
    return config.data_root / "standard_geometry_bias" / name


def _prepare_reference(
    *,
    config: Any,
    seed: int,
    reference_size: int,
    run_dir: Path,
    source_run_dir: Path | None,
    progress_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = run_dir / "reference_theta.npz"
    if path.exists():
        payload = np.load(path)
        reference_unit = np.asarray(payload["theta_unit"], dtype=np.float64)
        reference_theta = np.asarray(payload["theta_raw"], dtype=np.float64)
        _write_progress(progress_path, "reference_loaded", reference_size=int(reference_unit.shape[0]), artifact_path=str(path))
    else:
        reference_unit = draw_reference_unit(count=reference_size, dim=config.parameter_space.dim, seed=seed)
        reference_theta = config.parameter_space.denormalize(reference_unit)
        np.savez_compressed(
            path,
            theta_unit=reference_unit.astype(np.float64),
            theta_raw=reference_theta.astype(np.float64),
            theta_names=np.asarray(config.parameter_space.theta_names),
        )
        _write_progress(progress_path, "reference_created", reference_size=int(reference_unit.shape[0]), artifact_path=str(path))
    if reference_unit.shape != (reference_size, config.parameter_space.dim):
        raise ValueError(f"reference shape mismatch: got {reference_unit.shape}, expected {(reference_size, config.parameter_space.dim)}")
    info = _check_source_reference_prefix(source_run_dir, reference_unit) if source_run_dir else {"checked": False}
    if info.get("checked"):
        _write_progress(progress_path, "source_reference_prefix_checked", **info)
    return reference_unit, reference_theta, info


def _prepare_designs(
    *,
    config: Any,
    seed: int,
    design_size: int,
    design_count: int,
    sampler: str,
    run_dir: Path,
    source_run_dir: Path | None,
    progress_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = run_dir / "training_designs_unit.npz"
    if path.exists():
        payload = np.load(path)
        designs = np.asarray(payload["theta_unit"], dtype=np.float64)
        _write_progress(progress_path, "designs_loaded", design_count=int(designs.shape[0]), design_size=int(designs.shape[1]), artifact_path=str(path))
    else:
        designs = np.asarray(
            [
                draw_design_unit(
                    design_size=design_size,
                    dim=config.parameter_space.dim,
                    seed=seed + 10000,
                    index=index,
                    sampler=sampler,
                )
                for index in range(design_count)
            ],
            dtype=np.float64,
        )
        np.savez_compressed(
            path,
            theta_unit=designs.astype(np.float64),
            sampler=str(sampler),
            design_size=int(design_size),
            design_count=int(design_count),
        )
        _write_progress(progress_path, "designs_created", design_count=int(designs.shape[0]), design_size=int(designs.shape[1]), artifact_path=str(path))
    expected = (design_count, design_size, config.parameter_space.dim)
    if designs.shape != expected:
        raise ValueError(f"design shape mismatch: got {designs.shape}, expected {expected}")
    info = _check_source_design_prefix(source_run_dir, designs) if source_run_dir else {"checked": False}
    if info.get("checked"):
        _write_progress(progress_path, "source_design_prefix_checked", **info)
    return designs, info


def _check_source_reference_prefix(source_run_dir: Path | None, reference_unit: np.ndarray) -> dict[str, Any]:
    if source_run_dir is None:
        return {"checked": False}
    source_path = source_run_dir / "reference_theta.npz"
    if not source_path.exists():
        return {"checked": False, "reason": f"missing {source_path}"}
    old = np.asarray(np.load(source_path)["theta_unit"], dtype=np.float64)
    if old.shape[0] > reference_unit.shape[0]:
        raise ValueError("source reference is larger than target reference.")
    equal = bool(np.allclose(old, reference_unit[: old.shape[0]], atol=0.0, rtol=0.0))
    if not equal:
        raise ValueError("source reference is not a strict prefix of target reference.")
    return {"checked": True, "source_count": int(old.shape[0]), "target_count": int(reference_unit.shape[0]), "prefix_equal": True}


def _check_source_design_prefix(source_run_dir: Path | None, designs: np.ndarray) -> dict[str, Any]:
    if source_run_dir is None:
        return {"checked": False}
    source_path = source_run_dir / "training_designs_unit.npz"
    if not source_path.exists():
        return {"checked": False, "reason": f"missing {source_path}"}
    old = np.asarray(np.load(source_path)["theta_unit"], dtype=np.float64)
    if old.shape[0] > designs.shape[0]:
        raise ValueError("source designs are larger than target designs.")
    equal = bool(np.allclose(old, designs[: old.shape[0]], atol=0.0, rtol=0.0))
    if not equal:
        raise ValueError("source designs are not a strict prefix of target designs.")
    return {"checked": True, "source_count": int(old.shape[0]), "target_count": int(designs.shape[0]), "prefix_equal": True}


def _prepare_thresholds(
    *,
    reference_unit: np.ndarray,
    designs: np.ndarray,
    sg_config: StandardGeometryConfig,
    run_dir: Path,
    progress_path: Path,
) -> GeometryThresholds:
    path = run_dir / "geometry_thresholds.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _write_progress(progress_path, "geometry_thresholds_loaded", artifact_path=str(path), thresholds=payload)
        return GeometryThresholds(**{key: float(payload[key]) for key in ("tau_lambda", "h_min", "h_max", "kappa_max", "boundary_min")})

    scales: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    for index, design in enumerate(designs):
        batch = compute_geometry_batch(reference_unit, design)
        finite = batch.finite_geometry
        if np.any(finite):
            scales.append(np.asarray(batch.simplex_scale[finite], dtype=np.float64))
            cond = np.asarray(batch.simplex_condition[finite], dtype=np.float64)
            conditions.append(cond[np.isfinite(cond)])
        if (index + 1) % 25 == 0 or index + 1 == designs.shape[0]:
            _write_progress(progress_path, "geometry_threshold_pass_progress", completed=index + 1, total=int(designs.shape[0]))

    if not scales or not conditions:
        raise ValueError("cannot derive standard-geometry thresholds without valid geometry.")
    h_values = np.concatenate(scales)
    k_values = np.concatenate(conditions)
    thresholds = GeometryThresholds(
        tau_lambda=float(sg_config.tau_lambda),
        h_min=float(np.quantile(h_values, float(sg_config.h_quantile_low))),
        h_max=float(np.quantile(h_values, float(sg_config.h_quantile_high))),
        kappa_max=float(np.quantile(k_values[np.isfinite(k_values)], float(sg_config.kappa_quantile_max))),
        boundary_min=float(sg_config.boundary_min),
    )
    path.write_text(json.dumps(_json_safe(thresholds.as_dict()), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_progress(progress_path, "geometry_thresholds_ready", thresholds=thresholds.as_dict(), artifact_path=str(path))
    return thresholds


def _prepare_accepted_indices(
    *,
    reference_unit: np.ndarray,
    designs: np.ndarray,
    thresholds: GeometryThresholds,
    run_dir: Path,
    progress_path: Path,
    args_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices_path = run_dir / "accepted_indices_by_design.npz"
    counts_path = run_dir / "geometry_counts.npz"
    summary_path = run_dir / "geometry_summary.json"
    if indices_path.exists() and counts_path.exists():
        idx_payload = np.load(indices_path)
        count_payload = np.load(counts_path)
        _write_progress(progress_path, "accepted_indices_loaded", artifact_path=str(indices_path))
        return (
            np.asarray(count_payload["accepted_count"], dtype=np.int64),
            np.asarray(count_payload["inside_count"], dtype=np.int64),
            np.asarray(idx_payload["offsets"], dtype=np.int64),
            np.asarray(idx_payload["indices"], dtype=np.int64),
        )

    accepted_counts = np.zeros(reference_unit.shape[0], dtype=np.int64)
    inside_counts = np.zeros(reference_unit.shape[0], dtype=np.int64)
    offsets = [0]
    chunks: list[np.ndarray] = []
    for index, design in enumerate(designs):
        batch = compute_geometry_batch(reference_unit, design)
        inside_counts += batch.finite_geometry.astype(np.int64)
        accepted = accepted_mask(batch, thresholds)
        accepted_indices = np.flatnonzero(accepted).astype(np.int64)
        accepted_counts += accepted.astype(np.int64)
        chunks.append(accepted_indices)
        offsets.append(offsets[-1] + int(accepted_indices.shape[0]))
        if (index + 1) % 25 == 0 or index + 1 == designs.shape[0]:
            _write_progress(
                progress_path,
                "accepted_index_pass_progress",
                completed=index + 1,
                total=int(designs.shape[0]),
                accepted_pairs_done=int(offsets[-1]),
                last_accepted_count=int(accepted_indices.shape[0]),
            )

    accepted_indices_all = np.concatenate(chunks).astype(np.int64) if chunks else np.empty(0, dtype=np.int64)
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    np.savez_compressed(indices_path, offsets=offsets_arr, indices=accepted_indices_all)
    np.savez_compressed(
        counts_path,
        accepted_count=accepted_counts.astype(np.int64),
        inside_count=inside_counts.astype(np.int64),
        accepted_fraction=(accepted_counts / max(designs.shape[0], 1)).astype(np.float64),
        inside_fraction=(inside_counts / max(designs.shape[0], 1)).astype(np.float64),
    )
    geometry_summary = _geometry_summary(
        accepted_counts=accepted_counts,
        inside_counts=inside_counts,
        design_count=int(designs.shape[0]),
        thresholds=thresholds.as_dict(),
        args=args_payload,
    )
    summary_path.write_text(json.dumps(_json_safe(geometry_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_progress(progress_path, "accepted_indices_ready", artifact_path=str(indices_path), accepted_pairs=int(accepted_indices_all.shape[0]))
    return accepted_counts, inside_counts, offsets_arr, accepted_indices_all


def _prepare_reference_truth(
    *,
    oracle: Any,
    truth_source: str,
    reference_unit: np.ndarray,
    reference_theta: np.ndarray,
    accepted_counts: np.ndarray,
    k_bins: np.ndarray,
    run_dir: Path,
    progress_path: Path,
    chunk_size: int,
    prefer_cuda_truth: bool,
) -> tuple[np.ndarray, np.ndarray]:
    support_indices = np.flatnonzero(np.asarray(accepted_counts, dtype=np.int64) > 0).astype(np.int64)
    source_slug = _source_slug(truth_source)
    support_path = run_dir / "accepted_reference_support.npz"
    truth_path = run_dir / f"accepted_reference_truth_{source_slug}_log.npz"
    if truth_path.exists():
        payload = np.load(truth_path)
        saved_support = np.asarray(payload["support_indices"], dtype=np.int64)
        saved_k = np.asarray(payload["k_bins"], dtype=np.float64)
        saved_source = _npz_string(payload, "truth_source")
        if (
            saved_source == str(truth_source)
            and np.array_equal(saved_support, support_indices)
            and saved_k.shape == k_bins.shape
            and np.allclose(saved_k, k_bins, atol=0.0, rtol=0.0)
        ):
            _write_progress(
                progress_path,
                "reference_truth_loaded",
                truth_source=truth_source,
                support_size=int(saved_support.shape[0]),
                artifact_path=str(truth_path),
            )
            return support_indices, np.asarray(payload["log_pk"], dtype=np.float64)
        raise ValueError("existing reference truth cache does not match current support/k-grid.")

    np.savez_compressed(
        support_path,
        support_indices=support_indices.astype(np.int64),
        theta_unit=reference_unit[support_indices].astype(np.float64),
        theta_raw=reference_theta[support_indices].astype(np.float64),
        accepted_count=np.asarray(accepted_counts[support_indices], dtype=np.int64),
    )
    truth_chunk_dir = run_dir / f"reference_truth_chunks_{source_slug}"
    truth_chunk_dir.mkdir(parents=True, exist_ok=True)
    truth_rows: list[np.ndarray] = []
    total = int(support_indices.shape[0])
    chunk_size = max(1, int(chunk_size))
    _write_progress(
        progress_path,
        "reference_truth_started",
        truth_source=truth_source,
        support_size=total,
        k_count=int(k_bins.shape[0]),
        chunk_size=chunk_size,
        prefer_cuda_truth=bool(prefer_cuda_truth),
    )
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk_path = truth_chunk_dir / f"truth_{start:06d}_{stop:06d}.npz"
        if chunk_path.exists():
            try:
                payload = np.load(chunk_path)
                chunk_log = np.asarray(payload["log_pk"], dtype=np.float64)
                chunk_support = np.asarray(payload["support_indices"], dtype=np.int64)
                chunk_source = _npz_string(payload, "truth_source")
                if (
                    chunk_source == str(truth_source)
                    and chunk_log.shape == (stop - start, k_bins.shape[0])
                    and np.array_equal(chunk_support, support_indices[start:stop])
                ):
                    truth_rows.append(chunk_log)
                    _write_progress(
                        progress_path,
                        "reference_truth_chunk_loaded",
                        truth_source=truth_source,
                        start=start,
                        stop=stop,
                        total=total,
                    )
                    continue
            except Exception:
                pass
            chunk_path.unlink(missing_ok=True)

        _write_progress(
            progress_path,
            "reference_truth_chunk_started",
            truth_source=truth_source,
            start=start,
            stop=stop,
            total=total,
        )
        truth = _evaluate_truth(
            oracle,
            reference_theta[support_indices[start:stop]],
            k_bins,
            prefer_cuda_truth=bool(prefer_cuda_truth),
        )
        chunk_log = truth.log_pk.astype(np.float64)
        truth_metadata = dict(getattr(truth, "metadata", {}) or {})
        base_metadata = dict(truth_metadata.get("base_oracle_metadata", {}) or {})
        np.savez_compressed(
            chunk_path,
            support_indices=support_indices[start:stop].astype(np.int64),
            k_bins=k_bins.astype(np.float64),
            log_pk=chunk_log,
            truth_source=np.asarray(str(truth_source)),
        )
        truth_rows.append(chunk_log)
        _write_progress(
            progress_path,
            "reference_truth_chunk_completed",
            truth_source=truth_source,
            start=start,
            stop=stop,
            total=total,
            generator_requested_device=base_metadata.get("truth_generator_requested_device"),
            generator_devices_used=base_metadata.get("truth_generator_devices_used"),
            generator_cuda_fallback_count=base_metadata.get("truth_generator_cuda_fallback_count"),
            csst_truth_backend_used=truth_metadata.get("csst_truth_backend_used"),
            csst_truth_batch_vectorized=truth_metadata.get("csst_truth_batch_vectorized"),
        )

    truth_log = np.vstack(truth_rows) if truth_rows else np.empty((0, int(k_bins.shape[0])), dtype=np.float64)
    tmp_truth_path = truth_path.with_name(f"{truth_path.stem}.tmp{truth_path.suffix}")
    np.savez_compressed(
        tmp_truth_path,
        support_indices=support_indices.astype(np.int64),
        theta_unit=reference_unit[support_indices].astype(np.float64),
        theta_raw=reference_theta[support_indices].astype(np.float64),
        k_bins=k_bins.astype(np.float64),
        log_pk=truth_log,
        truth_source=np.asarray(str(truth_source)),
    )
    tmp_truth_path.replace(truth_path)
    _write_progress(
        progress_path,
        "reference_truth_completed",
        truth_source=truth_source,
        support_size=int(support_indices.shape[0]),
        artifact_path=str(truth_path),
    )
    return support_indices, truth_log


def _make_bias_oracle(config: Any, *, truth_source: str) -> tuple[Any, str]:
    normalized = str(truth_source).strip().lower()
    if normalized == "csst_residual":
        return CSSTQuijote5DOracle(config), "csst_fastmock_official_log_residual"
    if normalized == "z2_logdiff":
        return make_oracle(config), str(config.target.kind)
    raise ValueError(f"unsupported bias truth source: {truth_source!r}")


def _evaluate_truth(
    oracle: Any,
    theta_raw: np.ndarray,
    k_bins: np.ndarray,
    *,
    prefer_cuda_truth: bool,
) -> Any:
    try:
        return oracle.evaluate(theta_raw, k_bins, prefer_cuda_truth=bool(prefer_cuda_truth))
    except TypeError as exc:
        if "prefer_cuda_truth" not in str(exc):
            raise
        return oracle.evaluate(theta_raw, k_bins)


def _source_slug(value: str) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    slug = "".join(chars).strip("_")
    return slug or "unknown"


def _npz_string(payload: Any, key: str) -> str:
    if key not in payload.files:
        return ""
    value = payload[key]
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def _run_bias_pass(
    *,
    config: Any,
    oracle: Any,
    truth_source: str,
    training_target_kind: str,
    prefer_cuda_train_truth: bool,
    k_bins: np.ndarray,
    reference_theta: np.ndarray,
    reference_truth_log: np.ndarray,
    support_lookup: np.ndarray,
    designs: np.ndarray,
    offsets: np.ndarray,
    accepted_indices_all: np.ndarray,
    chunk_size: int,
    save_interval: int,
    run_dir: Path,
    progress_path: Path,
) -> dict[str, np.ndarray | int]:
    state_path = run_dir / "bias_accumulator_state.npz"
    if state_path.exists():
        payload = np.load(state_path)
        saved_truth_source = _npz_string(payload, "truth_source")
        saved_training_target_kind = _npz_string(payload, "training_target_kind")
        if saved_truth_source != str(truth_source) or saved_training_target_kind != str(training_target_kind):
            raise ValueError(
                "existing bias accumulator state was built for a different truth source "
                f"or training target: state=({saved_truth_source!r}, {saved_training_target_kind!r}), "
                f"current=({truth_source!r}, {training_target_kind!r})."
            )
        sum_values = np.asarray(payload["sum"], dtype=np.float64)
        sum_sq_values = np.asarray(payload["sum_sq"], dtype=np.float64)
        count_values = np.asarray(payload["count"], dtype=np.int64)
        start_index = int(payload["next_design_index"])
        observed_pair_count = int(payload["observed_pair_count"])
        _write_progress(progress_path, "bias_state_loaded", next_design_index=start_index, observed_pair_count=observed_pair_count)
    else:
        sum_values = np.zeros(reference_theta.shape[0], dtype=np.float64)
        sum_sq_values = np.zeros(reference_theta.shape[0], dtype=np.float64)
        count_values = np.zeros(reference_theta.shape[0], dtype=np.int64)
        start_index = 0
        observed_pair_count = 0
        _write_progress(progress_path, "bias_state_initialized")

    for index in range(start_index, int(designs.shape[0])):
        accepted_indices = accepted_indices_all[int(offsets[index]) : int(offsets[index + 1])]
        if accepted_indices.size > 0:
            design_theta = config.parameter_space.denormalize(designs[index])
            train_truth = _evaluate_truth(
                oracle,
                design_theta,
                k_bins,
                prefer_cuda_truth=bool(prefer_cuda_train_truth),
            )
            emulator = PCAGPDirectCDMEmulator(
                config.parameter_space,
                config.model,
                target_kind=str(training_target_kind),
            ).fit(design_theta, train_truth.log_pk, k_bins)
            support_rows = support_lookup[accepted_indices]
            if np.any(support_rows < 0):
                raise RuntimeError("Accepted reference index was not present in the reference support table.")
            bias = _reference_bias_in_chunks(
                emulator=emulator,
                reference_theta=reference_theta[accepted_indices],
                reference_truth_log=reference_truth_log[support_rows],
                chunk_size=chunk_size,
            )
            finite = np.isfinite(bias)
            finite_indices = accepted_indices[finite]
            finite_bias = bias[finite]
            np.add.at(sum_values, finite_indices, finite_bias)
            np.add.at(sum_sq_values, finite_indices, finite_bias**2)
            np.add.at(count_values, finite_indices, 1)
            observed_pair_count += int(finite_bias.shape[0])

        completed = index + 1
        if completed % max(1, save_interval) == 0 or completed == int(designs.shape[0]):
            _save_bias_state(
                state_path,
                sum_values=sum_values,
                sum_sq_values=sum_sq_values,
                count_values=count_values,
                next_design_index=completed,
                observed_pair_count=observed_pair_count,
                truth_source=truth_source,
                training_target_kind=training_target_kind,
            )
            _write_progress(
                progress_path,
                "bias_pass_progress",
                completed=completed,
                total=int(designs.shape[0]),
                accepted_pairs_done=int(observed_pair_count),
                last_accepted_count=int(accepted_indices.shape[0]),
                state_path=str(state_path),
            )

    return {
        "sum": sum_values,
        "sum_sq": sum_sq_values,
        "count": count_values,
        "observed_pair_count": int(observed_pair_count),
        "design_count": int(designs.shape[0]),
    }


def _save_bias_state(
    path: Path,
    *,
    sum_values: np.ndarray,
    sum_sq_values: np.ndarray,
    count_values: np.ndarray,
    next_design_index: int,
    observed_pair_count: int,
    truth_source: str,
    training_target_kind: str,
) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        sum=sum_values.astype(np.float64),
        sum_sq=sum_sq_values.astype(np.float64),
        count=count_values.astype(np.int64),
        next_design_index=np.asarray(int(next_design_index), dtype=np.int64),
        observed_pair_count=np.asarray(int(observed_pair_count), dtype=np.int64),
        truth_source=np.asarray(str(truth_source)),
        training_target_kind=np.asarray(str(training_target_kind)),
    )
    if tmp.exists():
        tmp.replace(path)


def _write_final_artifacts(
    *,
    config: Any,
    run_dir: Path,
    reference_unit: np.ndarray,
    reference_theta: np.ndarray,
    theta_names: np.ndarray,
    state: dict[str, np.ndarray | int],
    design_count: int,
    usable_min_count: int,
    high_confidence_min_count: int,
    density_alpha: float,
    density_clip_quantile: float,
    geometry_summary: dict[str, Any],
    observed_pair_count: int,
    source_info: dict[str, Any],
    args_payload: dict[str, Any],
) -> Path:
    estimate = _estimate_from_state(
        sum_values=np.asarray(state["sum"], dtype=np.float64),
        sum_sq_values=np.asarray(state["sum_sq"], dtype=np.float64),
        count_values=np.asarray(state["count"], dtype=np.int64),
        design_count=design_count,
        usable_min_count=usable_min_count,
        high_confidence_min_count=high_confidence_min_count,
    )
    bias_path = run_dir / "standard_geometry_bias_field.npz"
    np.savez_compressed(
        bias_path,
        theta_unit=reference_unit.astype(np.float64),
        theta_raw=reference_theta.astype(np.float64),
        theta_names=theta_names,
        bias_truth_source=np.asarray(str(args_payload.get("bias_truth_source", ""))),
        training_target_kind=np.asarray(str(source_info.get("training_target_kind", ""))),
        bias_mean=estimate["bias_mean"].astype(np.float64),
        bias_std=estimate["bias_std"].astype(np.float64),
        bias_se=estimate["bias_se"].astype(np.float64),
        accepted_count=estimate["accepted_count"].astype(np.int64),
        accepted_fraction=estimate["accepted_fraction"].astype(np.float64),
        high_confidence=estimate["high_confidence"].astype(bool),
        usable=estimate["usable"].astype(bool),
    )

    density = density_from_bias(
        np.asarray(estimate["bias_mean"], dtype=np.float64),
        alpha=density_alpha,
        clip_quantile=density_clip_quantile,
    )
    np.savez_compressed(
        run_dir / "density_field.npz",
        theta_unit=reference_unit.astype(np.float64),
        theta_raw=reference_theta.astype(np.float64),
        bias_truth_source=np.asarray(str(args_payload.get("bias_truth_source", ""))),
        training_target_kind=np.asarray(str(source_info.get("training_target_kind", ""))),
        density=density.astype(np.float64),
        bias_mean=np.asarray(estimate["bias_mean"], dtype=np.float64),
        accepted_count=np.asarray(estimate["accepted_count"], dtype=np.int64),
    )

    interpolator = ReliabilityWeightedLocalInterpolator(
        reference_unit,
        np.asarray(estimate["bias_mean"], dtype=np.float64),
        np.asarray(estimate["accepted_count"], dtype=np.int64),
        min_count=usable_min_count,
        high_confidence_count=high_confidence_min_count,
    )
    interp_self, interp_conf = interpolator.predict(reference_unit)
    interpolator_payload_path = run_dir / "interpolator_support.npz"
    np.savez_compressed(
        interpolator_payload_path,
        theta_unit=reference_unit.astype(np.float64),
        bias_truth_source=np.asarray(str(args_payload.get("bias_truth_source", ""))),
        training_target_kind=np.asarray(str(source_info.get("training_target_kind", ""))),
        bias_mean=np.asarray(estimate["bias_mean"], dtype=np.float64),
        accepted_count=np.asarray(estimate["accepted_count"], dtype=np.int64),
        self_interpolated_bias=interp_self.astype(np.float64),
        self_interpolation_confidence=interp_conf.astype(np.float64),
    )

    finite_bias = np.asarray(estimate["bias_mean"], dtype=np.float64)
    finite_bias = finite_bias[np.isfinite(finite_bias)]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "standard_geometry_bias_field_extended",
        "config_path": str(config.config_path),
        "target_kind": str(config.target.kind),
        "anchor_mode": str(config.target.anchor_mode),
        "bias_truth_source": str(args_payload.get("bias_truth_source", "")),
        "training_target_kind": str(source_info.get("training_target_kind", "")),
        "k_count": int(len(config.k_grid.k_bins)),
        "k_min": float(config.k_grid.k_bins[0]),
        "k_max": float(config.k_grid.k_bins[-1]),
        "parameter_space": {
            "name": str(config.parameter_space.name),
            "theta_names": list(config.parameter_space.theta_names),
            "theta_bounds": config.parameter_space.theta_bounds.astype(float).tolist(),
        },
        "reference_size": int(reference_unit.shape[0]),
        "design_size": int(args_payload["design_size"]),
        "design_count": int(design_count),
        "tau_lambda": float(args_payload["tau_lambda"]),
        "source_info": source_info,
        "geometry": geometry_summary,
        "bias_evaluation": {
            "mode": "accepted_center_points_only",
            "truth_source": str(args_payload.get("bias_truth_source", "")),
            "training_target_kind": str(source_info.get("training_target_kind", "")),
            "reference_support_size": int(np.count_nonzero(np.asarray(estimate["accepted_count"], dtype=np.int64) > 0)),
            "observed_pair_count": int(observed_pair_count),
            "zero_acceptance_reference_count": int(np.count_nonzero(np.asarray(estimate["accepted_count"], dtype=np.int64) <= 0)),
        },
        "bias_estimate": {
            "usable_min_count": int(usable_min_count),
            "high_confidence_min_count": int(high_confidence_min_count),
            "usable_fraction": float(np.mean(np.asarray(estimate["usable"], dtype=bool))),
            "high_confidence_fraction": float(np.mean(np.asarray(estimate["high_confidence"], dtype=bool))),
            "finite_bias_fraction": float(np.mean(np.isfinite(np.asarray(estimate["bias_mean"], dtype=np.float64)))),
            "accepted_count_quantiles": _quantiles(np.asarray(estimate["accepted_count"], dtype=np.int64)),
            "bias_mean_quantiles": _quantiles(finite_bias),
        },
        "artifacts": {
            "bias_field": str(bias_path),
            "density_field": str(run_dir / "density_field.npz"),
            "interpolator_support": str(interpolator_payload_path),
            "bias_accumulator_state": str(run_dir / "bias_accumulator_state.npz"),
            "accepted_indices_by_design": str(run_dir / "accepted_indices_by_design.npz"),
            "reference_truth": str(run_dir / f"accepted_reference_truth_{_source_slug(str(args_payload.get('bias_truth_source', '')))}_log.npz"),
        },
        "reference_digest": digest_theta(reference_unit, decimals=config.splits.duplicate_decimals),
    }
    summary_path = run_dir / "standard_geometry_bias_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def _estimate_from_state(
    *,
    sum_values: np.ndarray,
    sum_sq_values: np.ndarray,
    count_values: np.ndarray,
    design_count: int,
    usable_min_count: int,
    high_confidence_min_count: int,
) -> dict[str, np.ndarray]:
    count = np.asarray(count_values, dtype=np.int64)
    total = np.asarray(sum_values, dtype=np.float64)
    total_sq = np.asarray(sum_sq_values, dtype=np.float64)
    mean = np.full(total.shape, np.nan, dtype=np.float64)
    std = np.full(total.shape, np.nan, dtype=np.float64)
    se = np.full(total.shape, np.nan, dtype=np.float64)
    positive = count > 0
    mean[positive] = total[positive] / count[positive]
    variance = np.full(total.shape, np.nan, dtype=np.float64)
    multi = count > 1
    variance[multi] = (total_sq[multi] - (total[multi] ** 2) / count[multi]) / np.maximum(count[multi] - 1, 1)
    variance[multi] = np.maximum(variance[multi], 0.0)
    variance[positive & ~multi] = 0.0
    std[positive] = np.sqrt(variance[positive])
    se[positive] = std[positive] / np.sqrt(np.maximum(count[positive], 1))
    accepted_fraction = np.zeros_like(total, dtype=np.float64)
    if design_count > 0:
        accepted_fraction = count.astype(np.float64) / float(design_count)
    return {
        "bias_mean": mean,
        "bias_std": std,
        "bias_se": se,
        "accepted_count": count,
        "accepted_fraction": accepted_fraction,
        "high_confidence": count >= int(high_confidence_min_count),
        "usable": count >= int(usable_min_count),
    }


if __name__ == "__main__":
    raise SystemExit(main())
