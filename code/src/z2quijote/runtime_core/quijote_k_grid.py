"""Quijote-specific k-grid helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def build_quijote_output_k_bins(metadata: Mapping[str, Any]) -> np.ndarray:
    """Build the configured dense Quijote output grid.

    The current official dense grid keeps the configured total and redistributes
    bins across Quijote k bands.
    """

    count = int(metadata["truth_generator_output_k_count"])
    k_min = float(metadata["truth_generator_output_k_min"])
    k_max = float(metadata["truth_generator_output_k_max"])
    spacing = str(metadata.get("truth_generator_output_k_spacing", "logspace")).strip().lower()
    if count <= 1:
        raise ValueError("truth_generator_output_k_count must be greater than 1.")
    if k_min <= 0.0 or k_max <= k_min:
        raise ValueError("truth_generator_output_k_min/max must define a positive increasing range.")

    if spacing == "logspace":
        return np.logspace(np.log10(k_min), np.log10(k_max), count).astype(np.float64)

    if spacing != "piecewise_logspace":
        raise ValueError(f"Unsupported Quijote output k spacing {spacing!r}.")

    bands = metadata.get("truth_generator_output_k_bands")
    if not isinstance(bands, Mapping):
        raise ValueError("piecewise_logspace requires truth_generator_output_k_bands.")
    band_specs = []
    for name, raw_spec in bands.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Quijote output k band {name!r} must be a mapping.")
        band_count = int(raw_spec["count"])
        band_min = float(raw_spec["k_min"])
        band_max = float(raw_spec["k_max"])
        band_specs.append((str(name), band_count, band_min, band_max))
    band_specs.sort(key=lambda item: item[2])
    if sum(item[1] for item in band_specs) != count:
        raise ValueError(
            "truth_generator_output_k_bands counts must sum to "
            f"truth_generator_output_k_count={count}."
        )
    if min(item[1] for item in band_specs) <= 0:
        raise ValueError("Each Quijote output k band must contain at least one bin.")
    if abs(band_specs[0][2] - k_min) > max(1.0e-12, 1.0e-10 * abs(k_min)):
        raise ValueError("First Quijote output k band must start at truth_generator_output_k_min.")
    if abs(band_specs[-1][3] - k_max) > max(1.0e-12, 1.0e-10 * abs(k_max)):
        raise ValueError("Last Quijote output k band must end at truth_generator_output_k_max.")
    for previous, current in zip(band_specs[:-1], band_specs[1:], strict=True):
        if not (previous[2] < previous[3] <= current[2] < current[3]):
            raise ValueError("Quijote piecewise k bands must be ordered and non-overlapping.")
        if abs(previous[3] - current[2]) > max(1.0e-12, 1.0e-10 * abs(previous[3])):
            raise ValueError("Adjacent Quijote output k bands must share a boundary.")

    pieces = []
    for idx, (_, band_count, band_min, band_max) in enumerate(band_specs):
        start = band_min if idx == 0 else np.nextafter(band_min, np.inf)
        pieces.append(
            np.logspace(
                np.log10(start),
                np.log10(band_max),
                band_count,
                endpoint=(idx == len(band_specs) - 1),
            )
        )
    out = np.concatenate(pieces).astype(np.float64)
    if out.shape != (count,) or not np.all(np.diff(out) > 0.0):
        raise RuntimeError("Constructed Quijote output k grid is not strictly increasing.")
    return out


def maybe_build_quijote_output_k_bins(metadata: Mapping[str, Any]) -> np.ndarray | None:
    if "truth_generator_output_k_count" not in metadata:
        return None
    return build_quijote_output_k_bins(metadata)


__all__ = ["build_quijote_output_k_bins", "maybe_build_quijote_output_k_bins"]
