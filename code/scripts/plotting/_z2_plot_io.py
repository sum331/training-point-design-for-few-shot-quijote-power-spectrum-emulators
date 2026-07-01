from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_payload(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        npz_ref = payload.get("spectra_npz_path") or payload.get("spectra_arrays_path") or payload.get("data_npz_path")
        if npz_ref:
            npz_path = Path(npz_ref)
            if not npz_path.is_absolute():
                npz_path = (resolved.parent / npz_path).resolve()
            if npz_path.exists():
                with np.load(npz_path, allow_pickle=False) as data:
                    for key in data.files:
                        payload.setdefault(key, np.asarray(data[key]))
                payload["spectra_npz_path"] = str(npz_path)
        return payload
    if suffix == ".npz":
        with np.load(resolved, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    raise ValueError(f"Unsupported payload format: {resolved}")


def as_array(value: Any, *, dtype: np.dtype[np.float64] = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def extract_ratio_arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    k_bins = as_array(payload["k_bins"])
    if "p_true_batch" in payload and "p_pred_batch" in payload:
        return k_bins, as_array(payload["p_true_batch"]), as_array(payload["p_pred_batch"])
    if "truth_target" in payload and "pred_target" in payload:
        return k_bins, as_array(payload["truth_target"]), as_array(payload["pred_target"])
    return k_bins, None, None


def extract_ratio_percent_batch(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    k_bins, p_true_batch, p_pred_batch = extract_ratio_arrays(payload)
    if "signed_relative_bias" in payload:
        signed = as_array(payload["signed_relative_bias"])
        if signed.ndim == 2 and signed.shape[1] == k_bins.shape[0]:
            return k_bins, signed * 100.0
    if p_true_batch is None or p_pred_batch is None:
        raise KeyError("payload does not contain p_true/p_pred arrays or signed_relative_bias.")
    safe_true = np.maximum(p_true_batch, 1.0e-30)
    ratio_percent = (p_pred_batch / safe_true - 1.0) * 100.0
    return k_bins, ratio_percent


def extract_relative_error_batch(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    k_bins, p_true_batch, p_pred_batch = extract_ratio_arrays(payload)
    if "signed_relative_bias" in payload:
        signed = as_array(payload["signed_relative_bias"])
        if signed.ndim == 2 and signed.shape[1] == k_bins.shape[0]:
            return k_bins, np.abs(signed)
    if p_true_batch is None or p_pred_batch is None:
        raise KeyError("payload does not contain p_true/p_pred arrays or signed_relative_bias.")
    safe_true = np.maximum(p_true_batch, 1.0e-30)
    relative_error = np.abs(p_pred_batch - p_true_batch) / safe_true
    return k_bins, relative_error


def extract_mean_spectra(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_bins, p_true_batch, p_pred_batch = extract_ratio_arrays(payload)
    if p_true_batch is None or p_pred_batch is None:
        raise KeyError("payload does not contain p_true/p_pred arrays.")
    return k_bins, np.mean(p_true_batch, axis=0), np.mean(p_pred_batch, axis=0)
