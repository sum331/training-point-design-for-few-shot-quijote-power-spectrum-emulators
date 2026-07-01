from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class KBand:
    name: str
    k_min: float
    k_max: float
    count: int


@dataclass(frozen=True, slots=True)
class KGrid:
    k_bins: np.ndarray
    bands: tuple[KBand, ...]

    @property
    def band_edges(self) -> tuple[float, ...]:
        return tuple(float(band.k_max) for band in self.bands[:-1])

    @property
    def band_labels(self) -> tuple[str, ...]:
        return tuple(band.name for band in self.bands)


def _resolve_path(value: object, base_path: Path | None) -> Path:
    path = Path(str(value))
    if path.is_absolute() or base_path is None:
        return path.resolve()
    return (base_path / path).resolve()


def _build_bands_from_edges(k_bins: np.ndarray, raw_bands: list[object]) -> tuple[KBand, ...]:
    bands: list[KBand] = []
    for index, item in enumerate(raw_bands):
        if not isinstance(item, Mapping):
            raise ValueError("k_grid.bands entries must be mappings.")
        k_min = float(item["k_min"])
        k_max = float(item["k_max"])
        if k_min <= 0.0 or k_max <= k_min:
            raise ValueError(f"invalid k band bounds: {item!r}.")
        if index == len(raw_bands) - 1:
            mask = (k_bins >= k_min) & (k_bins <= k_max)
        else:
            mask = (k_bins >= k_min) & (k_bins < k_max)
        bands.append(
            KBand(
                name=str(item.get("name", f"band_{index}")),
                k_min=k_min,
                k_max=k_max,
                count=int(np.count_nonzero(mask)),
            )
        )
    if any(band.count <= 0 for band in bands):
        raise ValueError(f"k_grid source produced an empty band: {bands!r}.")
    return tuple(bands)


def build_k_grid(payload: Mapping[str, object], *, base_path: str | Path | None = None) -> KGrid:
    source = str(payload.get("source", "")).strip().lower()
    if source in {"npz", "numpy", "file"}:
        raw_path = payload.get("path")
        if raw_path in (None, ""):
            raise ValueError("k_grid.path is required when k_grid.source='npz'.")
        path = _resolve_path(raw_path, Path(base_path).resolve() if base_path is not None else None)
        if not path.exists():
            raise FileNotFoundError(f"k_grid source npz not found: {path}")
        key = str(payload.get("key", "k_bins"))
        with np.load(path, allow_pickle=False) as npz:
            if key not in npz:
                raise KeyError(f"k_grid source npz {path} does not contain key {key!r}.")
            source_k = np.asarray(npz[key], dtype=np.float64).reshape(-1)
        if source_k.ndim != 1 or source_k.size <= 0 or np.any(source_k <= 0.0):
            raise ValueError("k_grid source bins must be a non-empty positive 1D array.")
        source_k = np.unique(source_k.astype(np.float64))
        source_k.sort()
        k_min = float(payload.get("k_min", float(source_k[0])))
        k_max = float(payload.get("k_max", float(source_k[-1])))
        mask = (source_k >= k_min) & (source_k <= k_max)
        k_bins = source_k[mask].astype(np.float64)
        if k_bins.size <= 0:
            raise ValueError(
                f"k_grid source {path} has no bins inside [{k_min:.6g}, {k_max:.6g}]."
            )
        expected_count = payload.get("expected_count")
        if expected_count not in (None, "") and int(expected_count) != int(k_bins.size):
            raise ValueError(
                f"k_grid source produced {int(k_bins.size)} bins, expected {int(expected_count)}."
            )
        if np.any(np.diff(k_bins) <= 0.0):
            raise ValueError("k_grid source bins must be strictly increasing after filtering.")
        raw_bands = payload.get("bands")
        if isinstance(raw_bands, list) and raw_bands:
            bands = _build_bands_from_edges(k_bins, raw_bands)
        else:
            bands = (KBand(name="all", k_min=float(k_bins[0]), k_max=float(k_bins[-1]), count=int(k_bins.size)),)
        return KGrid(k_bins=k_bins, bands=bands)

    raw_bands = payload.get("bands")
    if not isinstance(raw_bands, list) or not raw_bands:
        count = int(payload.get("k_count", 256))
        k_min = float(payload.get("k_min", 0.01))
        k_max = float(payload.get("k_max", 3.0))
        raw_bands = [{"name": "all", "k_min": k_min, "k_max": k_max, "count": count}]

    bands: list[KBand] = []
    pieces: list[np.ndarray] = []
    for index, item in enumerate(raw_bands):
        if not isinstance(item, Mapping):
            raise ValueError("k_grid.bands entries must be mappings.")
        band = KBand(
            name=str(item.get("name", f"band_{index}")),
            k_min=float(item["k_min"]),
            k_max=float(item["k_max"]),
            count=int(item["count"]),
        )
        if band.k_min <= 0.0 or band.k_max <= band.k_min or band.count <= 0:
            raise ValueError(f"invalid k band: {band!r}.")
        endpoint = index == len(raw_bands) - 1
        pieces.append(np.geomspace(band.k_min, band.k_max, band.count, endpoint=endpoint))
        bands.append(band)

    k_bins = np.concatenate(pieces).astype(np.float64)
    if np.any(k_bins <= 0.0) or np.any(np.diff(k_bins) <= 0.0):
        raise ValueError("constructed k grid must be strictly increasing and positive.")
    return KGrid(k_bins=k_bins, bands=tuple(bands))
