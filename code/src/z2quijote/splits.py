from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Z2Config
from .resources import load_r2_seed_geometry
from .sampling import digest_theta, draw_disjoint_sobol, theta_rows_key


@dataclass(frozen=True, slots=True)
class SplitBundle:
    manifest_path: Path
    arrays_path: Path
    seed_theta_raw: np.ndarray
    seed_theta_unit: np.ndarray
    probe_theta_raw: np.ndarray
    probe_theta_unit: np.ndarray
    pool_theta_raw: np.ndarray
    pool_theta_unit: np.ndarray
    audit_theta_raw: np.ndarray
    audit_theta_unit: np.ndarray
    sobol64_theta_raw: np.ndarray
    sobol64_theta_unit: np.ndarray
    sobol128_theta_raw: np.ndarray
    sobol128_theta_unit: np.ndarray
    sobol_tail_theta_raw: np.ndarray
    sobol_tail_theta_unit: np.ndarray
    metadata: dict[str, Any]


def build_split_bundle(config: Z2Config, *, force: bool = False) -> SplitBundle:
    seed = load_r2_seed_geometry(config)
    decimals = config.splits.duplicate_decimals
    seen = set(theta_rows_key(seed.theta_unit, decimals=decimals))
    dim = config.parameter_space.dim
    base_seed = int(config.random_seed)

    probe_unit, seen = draw_disjoint_sobol(
        count=config.splits.probe_size,
        dim=dim,
        seed=base_seed + 11,
        exclude=seen,
        decimals=decimals,
    )
    pool_unit, seen = draw_disjoint_sobol(
        count=config.splits.pool_size,
        dim=dim,
        seed=base_seed + 23,
        exclude=seen,
        decimals=decimals,
    )
    audit_unit, audit_raw, audit_metadata, seen = _build_audit_split(
        config,
        seen=seen,
        dim=dim,
        seed=base_seed + 37,
        decimals=decimals,
    )
    sobol64_unit, seen = draw_disjoint_sobol(
        count=config.splits.sobol64_size,
        dim=dim,
        seed=base_seed + 39,
        exclude=seen,
        decimals=decimals,
    )
    sobol128_unit, seen = draw_disjoint_sobol(
        count=config.splits.sobol128_size,
        dim=dim,
        seed=base_seed + 41,
        exclude=seen,
        decimals=decimals,
    )
    sobol_tail_unit, seen = draw_disjoint_sobol(
        count=config.splits.sobol_tail_size,
        dim=dim,
        seed=base_seed + 53,
        exclude=seen,
        decimals=decimals,
    )

    probe_raw = config.parameter_space.denormalize(probe_unit)
    pool_raw = config.parameter_space.denormalize(pool_unit)
    sobol64_raw = config.parameter_space.denormalize(sobol64_unit)
    sobol128_raw = config.parameter_space.denormalize(sobol128_unit)
    sobol_tail_raw = config.parameter_space.denormalize(sobol_tail_unit)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"z2_direct_cdm_{config.splits.seed_label}_{timestamp}"
    out_dir = config.splits.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / f"{label}.npz"
    manifest_path = out_dir / f"{label}.json"
    if not force and (arrays_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"split output already exists: {arrays_path} / {manifest_path}")

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package": "z2quijote",
        "target_kind": config.target.kind,
        "anchor_mode": config.target.anchor_mode,
        "use_lofi": False,
        "r2_seed": seed.metadata,
        "parameter_space": {
            "name": config.parameter_space.name,
            "theta_names": list(config.parameter_space.theta_names),
            "theta_bounds": config.parameter_space.theta_bounds.tolist(),
        },
        "k_count": int(config.k_grid.k_bins.shape[0]),
        "arrays_path": str(arrays_path),
        "role_shapes": {
            "seed": list(seed.theta_raw.shape),
            "probe": list(probe_raw.shape),
            "pool": list(pool_raw.shape),
            "audit": list(audit_raw.shape),
            "sobol64": list(sobol64_raw.shape),
            "sobol128": list(sobol128_raw.shape),
            "sobol_tail": list(sobol_tail_raw.shape),
        },
        "digests": {
            "seed_theta_unit": digest_theta(seed.theta_unit, decimals=decimals),
            "probe_theta_unit": digest_theta(probe_unit, decimals=decimals),
            "pool_theta_unit": digest_theta(pool_unit, decimals=decimals),
            "audit_theta_unit": digest_theta(audit_unit, decimals=decimals),
            "sobol64_theta_unit": digest_theta(sobol64_unit, decimals=decimals),
            "sobol128_theta_unit": digest_theta(sobol128_unit, decimals=decimals),
            "sobol_tail_theta_unit": digest_theta(sobol_tail_unit, decimals=decimals),
        },
        "audit_source": audit_metadata,
        "duplicate_decimals": int(decimals),
        "split_policy": "all roles are duplicate-filtered against seed and previous roles",
    }

    np.savez_compressed(
        arrays_path,
        seed_theta_raw=seed.theta_raw,
        seed_theta_unit=seed.theta_unit,
        probe_theta_raw=probe_raw,
        probe_theta_unit=probe_unit,
        pool_theta_raw=pool_raw,
        pool_theta_unit=pool_unit,
        audit_theta_raw=audit_raw,
        audit_theta_unit=audit_unit,
        sobol64_theta_raw=sobol64_raw,
        sobol64_theta_unit=sobol64_unit,
        sobol128_theta_raw=sobol128_raw,
        sobol128_theta_unit=sobol128_unit,
        sobol_tail_theta_raw=sobol_tail_raw,
        sobol_tail_theta_unit=sobol_tail_unit,
    )
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return _bundle_from_arrays(manifest_path, arrays_path, metadata)


def _build_audit_split(
    config: Z2Config,
    *,
    seen: set[tuple[float, ...]],
    dim: int,
    seed: int,
    decimals: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], set[tuple[float, ...]]]:
    source = str(config.splits.audit_source).strip().lower()
    if source == "sobol":
        audit_unit, updated_seen = draw_disjoint_sobol(
            count=config.splits.audit_size,
            dim=dim,
            seed=seed,
            exclude=seen,
            decimals=decimals,
        )
        audit_raw = config.parameter_space.denormalize(audit_unit)
        return (
            audit_unit,
            audit_raw,
            {
                "source": "sobol",
                "seed": int(seed),
                "size": int(audit_unit.shape[0]),
            },
            updated_seen,
        )
    if source != "npz":
        raise ValueError(f"unsupported audit source: {config.splits.audit_source!r}")
    if config.splits.audit_path is None:
        raise ValueError("splits.audit_path is required when splits.audit_source=npz.")
    audit_path = Path(config.splits.audit_path).resolve()
    if not audit_path.exists():
        raise FileNotFoundError(f"fixed audit npz not found: {audit_path}")
    with np.load(audit_path, allow_pickle=False) as payload:
        files = set(payload.files)
        unit_key = str(config.splits.audit_theta_unit_key)
        raw_key = str(config.splits.audit_theta_raw_key)
        if unit_key in files:
            audit_unit = np.asarray(payload[unit_key], dtype=np.float64)
            if audit_unit.ndim != 2 or audit_unit.shape[1] != int(dim):
                raise ValueError(
                    f"fixed audit {unit_key!r} must have shape [N,{dim}], got {audit_unit.shape}."
                )
            audit_raw = config.parameter_space.denormalize(audit_unit)
        elif raw_key in files:
            audit_raw = np.asarray(payload[raw_key], dtype=np.float64)
            if audit_raw.ndim != 2 or audit_raw.shape[1] != int(dim):
                raise ValueError(
                    f"fixed audit {raw_key!r} must have shape [N,{dim}], got {audit_raw.shape}."
                )
            audit_unit = config.parameter_space.normalize(audit_raw)
        else:
            raise KeyError(
                f"fixed audit npz must contain {unit_key!r} or {raw_key!r}; available keys: {sorted(files)}"
            )
    if audit_unit.shape[0] != int(config.splits.audit_size):
        raise ValueError(
            "fixed audit size does not match splits.audit_size: "
            f"{audit_unit.shape[0]} != {config.splits.audit_size}."
        )
    if np.any(~np.isfinite(audit_unit)) or np.any(audit_unit < -1.0e-12) or np.any(audit_unit > 1.0 + 1.0e-12):
        raise ValueError("fixed audit theta_unit contains non-finite or out-of-box values.")
    audit_unit = np.clip(audit_unit, 0.0, 1.0).astype(np.float64)
    audit_raw = config.parameter_space.denormalize(audit_unit)
    audit_keys = set(theta_rows_key(audit_unit, decimals=decimals))
    overlap_count = len(set(seen).intersection(audit_keys))
    if overlap_count > 0:
        raise ValueError(f"fixed audit split overlaps with earlier roles at {overlap_count} rounded rows.")
    updated_seen = set(seen)
    updated_seen.update(audit_keys)
    return (
        audit_unit,
        audit_raw,
        {
            "source": "npz",
            "path": str(audit_path),
            "theta_unit_key": str(config.splits.audit_theta_unit_key),
            "theta_raw_key": str(config.splits.audit_theta_raw_key),
            "size": int(audit_unit.shape[0]),
            "theta_unit_digest": digest_theta(audit_unit, decimals=decimals),
        },
        updated_seen,
    )


def load_split_bundle(manifest_path: Path) -> SplitBundle:
    path = manifest_path.resolve()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    arrays_path = Path(metadata["arrays_path"]).resolve()
    return _bundle_from_arrays(path, arrays_path, metadata)


def _bundle_from_arrays(manifest_path: Path, arrays_path: Path, metadata: dict[str, Any]) -> SplitBundle:
    if not arrays_path.exists():
        raise FileNotFoundError(f"split arrays not found: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key], dtype=np.float64) for key in payload.files}
    return SplitBundle(
        manifest_path=manifest_path,
        arrays_path=arrays_path,
        seed_theta_raw=arrays["seed_theta_raw"],
        seed_theta_unit=arrays["seed_theta_unit"],
        probe_theta_raw=arrays["probe_theta_raw"],
        probe_theta_unit=arrays["probe_theta_unit"],
        pool_theta_raw=arrays["pool_theta_raw"],
        pool_theta_unit=arrays["pool_theta_unit"],
        audit_theta_raw=arrays["audit_theta_raw"],
        audit_theta_unit=arrays["audit_theta_unit"],
        sobol64_theta_raw=arrays["sobol64_theta_raw"],
        sobol64_theta_unit=arrays["sobol64_theta_unit"],
        sobol128_theta_raw=arrays["sobol128_theta_raw"],
        sobol128_theta_unit=arrays["sobol128_theta_unit"],
        sobol_tail_theta_raw=arrays["sobol_tail_theta_raw"],
        sobol_tail_theta_unit=arrays["sobol_tail_theta_unit"],
        metadata=metadata,
    )
