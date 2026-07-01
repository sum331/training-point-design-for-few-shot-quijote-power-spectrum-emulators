from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from .config import Z2Config
from .direct_cdm import compute_anchor_batch, make_anchor_provider


@dataclass(frozen=True, slots=True)
class RawBankSample:
    theta_raw: np.ndarray
    k_bins: np.ndarray
    log_pk: np.ndarray
    metadata: dict[str, Any]


def load_raw_bank_sample(config: Z2Config, target_k_bins: np.ndarray) -> tuple[RawBankSample | None, dict[str, Any]]:
    raw_cfg = config.resources.raw_bank
    if not raw_cfg.enabled:
        return None, {"enabled": False, "status": "disabled"}
    if raw_cfg.path is None:
        return None, {"enabled": True, "status": "missing_path"}
    if not raw_cfg.path.exists():
        return None, {"enabled": True, "status": "skipped_missing_file", "path": str(raw_cfg.path)}

    with np.load(raw_cfg.path, allow_pickle=False) as payload:
        theta = np.asarray(payload["raw_thetas"], dtype=np.float64)
        source_k = np.asarray(payload["k_bins"], dtype=np.float64)
        pk = np.asarray(payload["p_nonlin_batch"], dtype=np.float64)
        simulation_indices = (
            np.asarray(payload["simulation_indices"], dtype=np.int64)
            if "simulation_indices" in payload.files
            else np.arange(theta.shape[0], dtype=np.int64)
        )
    mask = np.ones((theta.shape[0],), dtype=bool)
    if raw_cfg.filter_to_parameter_box:
        bounds = config.parameter_space.theta_bounds
        mask = np.all((theta >= bounds[:, 0][None, :]) & (theta <= bounds[:, 1][None, :]), axis=1)
    available = np.flatnonzero(mask)
    if available.size == 0:
        return None, {"enabled": True, "status": "empty_after_filter", "path": str(raw_cfg.path)}
    rng = np.random.default_rng(int(raw_cfg.sample_seed))
    count = int(min(max(1, raw_cfg.sample_size), available.size))
    selected = np.asarray(rng.choice(available, size=count, replace=False), dtype=np.int64)
    selected.sort()
    target_k = np.asarray(target_k_bins, dtype=np.float64).reshape(-1)
    log_pk = _interp_log_pk(np.log(np.maximum(pk[selected], config.target.power_eps)), source_k, target_k)
    target_transform = "direct_logpk"
    anchor_metadata: dict[str, Any] = {"enabled": False}
    if str(config.target.kind).strip().lower() == "cdm_logdiff":
        anchor_provider = make_anchor_provider(config)
        anchor = compute_anchor_batch(
            anchor_provider,
            theta[selected],
            target_k,
            power_eps=float(config.target.power_eps),
        )
        log_pk = log_pk - np.log(np.maximum(anchor, config.target.power_eps))
        target_transform = f"log_hi_minus_log_{str(config.target.anchor_mode).strip().lower()}_anchor"
        anchor_metadata = {
            "enabled": True,
            "provider": str(anchor_provider.provider_name),
            "power_label": str(anchor_provider.power_label),
        }
    file_meta = _read_json(raw_cfg.metadata_path)
    metadata: dict[str, Any] = {
        "enabled": True,
        "status": "loaded",
        "path": str(raw_cfg.path),
        "metadata_path": str(raw_cfg.metadata_path) if raw_cfg.metadata_path else None,
        "source_bank_size": int(theta.shape[0]),
        "available_after_filter": int(available.size),
        "sample_size": int(count),
        "sample_seed": int(raw_cfg.sample_seed),
        "filter_to_parameter_box": bool(raw_cfg.filter_to_parameter_box),
        "target_kind": str(config.target.kind),
        "target_transform": target_transform,
        "anchor": anchor_metadata,
        "source_simulation_indices_preview": [int(item) for item in simulation_indices[selected[:10]].tolist()],
        "source_metadata": file_meta,
    }
    return RawBankSample(
        theta_raw=theta[selected],
        k_bins=target_k,
        log_pk=log_pk,
        metadata=metadata,
    ), metadata


def _interp_log_pk(log_pk: np.ndarray, source_k: np.ndarray, target_k: np.ndarray) -> np.ndarray:
    source = np.asarray(source_k, dtype=np.float64).reshape(-1)
    target = np.asarray(target_k, dtype=np.float64).reshape(-1)
    if target[0] < source[0] * (1.0 - 1.0e-10) or target[-1] > source[-1] * (1.0 + 1.0e-10):
        raise ValueError("target k grid extends outside raw-bank source k coverage.")
    if source.shape == target.shape and np.allclose(source, target, rtol=0.0, atol=0.0):
        return np.asarray(log_pk, dtype=np.float64)
    log_source = np.log10(source)
    log_target = np.log10(target)
    rows = [PchipInterpolator(log_source, row, extrapolate=False)(log_target) for row in log_pk]
    return np.vstack(rows).astype(np.float64)


def _read_json(path) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
