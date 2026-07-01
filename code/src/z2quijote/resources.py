from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Z2Config


@dataclass(frozen=True, slots=True)
class SeedGeometry:
    theta_raw: np.ndarray
    theta_unit: np.ndarray
    metadata: dict[str, Any]


def load_r2_seed_geometry(config: Z2Config) -> SeedGeometry:
    path = config.resources.r2_seed.path
    if not path.exists():
        raise FileNotFoundError(f"R2 seed geometry file not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        key = config.resources.r2_seed.theta_key
        if key not in payload.files:
            raise ValueError(f"R2 seed file {path} does not contain key {key!r}; keys={payload.files}.")
        theta_payload = np.asarray(payload[key], dtype=np.float64)
    if theta_payload.ndim != 2 or theta_payload.shape[1] != config.parameter_space.dim:
        raise ValueError(
            f"R2 seed theta must have shape [N,{config.parameter_space.dim}], got {theta_payload.shape}."
        )
    if str(key).startswith("theta_unit"):
        theta_unit = np.clip(theta_payload, 0.0, 1.0).astype(np.float64)
        theta_raw = config.parameter_space.denormalize(theta_unit)
        theta_semantics = "unit_geometry_renormalized_to_current_parameter_box"
    else:
        theta_raw = theta_payload.astype(np.float64)
        theta_unit = config.parameter_space.normalize(theta_raw)
        theta_semantics = "raw_theta_in_current_parameter_box"

    source_manifest = _read_json(config.resources.r2_seed.manifest_path)
    legacy_target = {
        "source_training_target_kind": source_manifest.get("source_training_target_kind"),
        "source_target": source_manifest.get("source_target"),
        "source_runtime_target": source_manifest.get("source_runtime_target"),
    }
    metadata: dict[str, Any] = {
        "resource_kind": "r2_seed_geometry",
        "resource_use": "geometry_only",
        "source_target_ignored": True,
        "theta_key": config.resources.r2_seed.theta_key,
        "theta_semantics": theta_semantics,
        "theta_shape": list(theta_raw.shape),
        "source_npz_path": str(path),
        "source_manifest_path": str(config.resources.r2_seed.manifest_path)
        if config.resources.r2_seed.manifest_path
        else None,
        "legacy_target_metadata": legacy_target,
    }
    return SeedGeometry(theta_raw=theta_raw, theta_unit=theta_unit, metadata=metadata)


def load_optional_current_active(config: Z2Config) -> tuple[np.ndarray | None, dict[str, Any]]:
    active = config.resources.current_active
    if not active.enabled:
        return None, {"enabled": False, "status": "disabled"}
    if active.path is None:
        return None, {"enabled": True, "status": "missing_path"}
    if not active.path.exists():
        return None, {"enabled": True, "status": "missing_file", "path": str(active.path)}
    with np.load(active.path, allow_pickle=False) as payload:
        if active.theta_key not in payload.files:
            return None, {
                "enabled": True,
                "status": "missing_key",
                "path": str(active.path),
                "theta_key": active.theta_key,
                "available_keys": list(payload.files),
            }
        theta_raw = np.asarray(payload[active.theta_key], dtype=np.float64)
    return theta_raw, {
        "enabled": True,
        "status": "loaded",
        "path": str(active.path),
        "theta_key": active.theta_key,
        "theta_shape": list(theta_raw.shape),
        "resource_use": "external_current_active_geometry_only",
    }


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
