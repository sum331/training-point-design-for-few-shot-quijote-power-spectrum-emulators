"""Shared target-representation and PCA-scheme helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA

PCA_BAND_LABELS: tuple[str, ...] = (
    "low_0.01_0.07",
    "mid_0.07_0.5",
    "high_0.5_1",
    "tail_1_3",
)


def build_k_band_masks(k_bins: np.ndarray) -> tuple[np.ndarray, ...]:
    k_arr = np.asarray(k_bins, dtype=np.float64).reshape(-1)
    return (
        (k_arr >= 0.01) & (k_arr < 0.07),
        (k_arr >= 0.07) & (k_arr < 0.5),
        (k_arr >= 0.5) & (k_arr < 1.0),
        (k_arr >= 1.0) & (k_arr <= 3.0),
    )


def parse_target_transform(target_transform: str) -> tuple[str, str | None]:
    normalized = str(target_transform).strip().lower() or "direct_logpk"
    if normalized == "direct_logpk":
        return "direct_logpk", None
    if normalized.startswith("ratio_to_"):
        return "ratio", normalized[len("ratio_to_") :]
    prefix = "log_hi_minus_log_"
    suffix = "_anchor"
    if normalized.startswith(prefix) and normalized.endswith(suffix):
        return "logdiff", normalized[len(prefix) : -len(suffix)]
    raise ValueError(f"Unsupported target_transform {target_transform!r}.")


def target_transform_name(
    *,
    transform_family: str,
    anchor_mode: str,
) -> str:
    transform = str(transform_family).strip().lower() or "ratio"
    anchor = str(anchor_mode).strip().lower() or "linear"
    if transform == "ratio":
        return f"ratio_to_{anchor}"
    if transform == "logdiff":
        return f"log_hi_minus_log_{anchor}_anchor"
    raise ValueError(
        "transform_family must be one of {'ratio', 'logdiff'}, "
        f"got {transform_family!r}."
    )


def resolve_target_transform_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    transform_family: str,
    anchor_mode: str,
) -> str:
    """Resolve a target-transform name without silently falling back to direct logP."""

    if metadata is not None:
        raw_value = metadata.get("target_transform")
        if raw_value is not None and str(raw_value).strip():
            return str(raw_value).strip()
    return target_transform_name(
        transform_family=transform_family,
        anchor_mode=anchor_mode,
    )


def build_target_representation(
    pk_batch: np.ndarray,
    *,
    anchor_batch: np.ndarray | None,
    power_eps: float,
    transform_family: str,
    anchor_mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    pk_arr = np.asarray(pk_batch, dtype=np.float64)
    eps = float(max(power_eps, 1.0e-30))
    if anchor_batch is None:
        log_pk = np.log(np.maximum(pk_arr, eps))
        return log_pk.astype(np.float64), {
            "target_transform": "direct_logpk",
            "reconstruction_space": "power_spectrum",
            "representation_transform_family": "direct_logpk",
            "representation_anchor_mode": None,
            "representation_has_anchor": False,
        }

    anchor_arr = np.asarray(anchor_batch, dtype=np.float64)
    if anchor_arr.shape != pk_arr.shape:
        raise ValueError(
            "anchor_batch must align with pk_batch, "
            f"got {anchor_arr.shape} vs {pk_arr.shape}."
        )

    normalized_transform = str(transform_family).strip().lower() or "ratio"
    normalized_anchor = str(anchor_mode).strip().lower() or "linear"
    if normalized_transform == "ratio":
        ratio = pk_arr / np.maximum(anchor_arr, eps)
        target = ratio - 1.0
    elif normalized_transform == "logdiff":
        target = np.log(np.maximum(pk_arr, eps)) - np.log(np.maximum(anchor_arr, eps))
    else:
        raise ValueError(
            "transform_family must be one of {'ratio', 'logdiff'}, "
            f"got {transform_family!r}."
        )

    return target.astype(np.float64), {
        "target_transform": target_transform_name(
            transform_family=normalized_transform,
            anchor_mode=normalized_anchor,
        ),
        "reconstruction_space": "power_spectrum",
        "representation_transform_family": normalized_transform,
        "representation_anchor_mode": normalized_anchor,
        "representation_has_anchor": True,
    }


def reconstruct_power_from_target(
    target_mean: np.ndarray,
    *,
    target_transform: str,
    anchor_batch: np.ndarray | None,
    power_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_arr = np.asarray(target_mean, dtype=np.float64)
    eps = float(max(power_eps, 1.0e-30))
    transform_family, _ = parse_target_transform(target_transform)
    if transform_family == "direct_logpk":
        log_pk_mean = target_arr
        return np.exp(log_pk_mean).astype(np.float64), log_pk_mean.astype(np.float64)

    if anchor_batch is None:
        raise ValueError(
            f"{target_transform} reconstruction requires anchor_batch for the queried thetas."
        )
    anchor_arr = np.asarray(anchor_batch, dtype=np.float64)
    if anchor_arr.shape != target_arr.shape:
        raise ValueError(
            "anchor_batch must align with target_mean, "
            f"got {anchor_arr.shape} vs {target_arr.shape}."
        )

    if transform_family == "ratio":
        ratio = np.maximum(1.0 + target_arr, 1.0e-8)
        pk_mean = np.maximum(anchor_arr * ratio, eps)
        log_pk_mean = np.log(pk_mean)
        return pk_mean.astype(np.float64), log_pk_mean.astype(np.float64)
    if transform_family == "logdiff":
        log_pk_mean = np.log(np.maximum(anchor_arr, eps)) + target_arr
        pk_mean = np.exp(log_pk_mean)
        return pk_mean.astype(np.float64), log_pk_mean.astype(np.float64)
    raise ValueError(f"Unsupported target_transform {target_transform!r}.")


def _allocate_component_counts(
    total_components: int,
    weights: Sequence[int | float],
    *,
    active_mask: Sequence[bool] | None = None,
) -> tuple[int, ...]:
    total = max(0, int(total_components))
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if active_mask is None:
        active = np.ones_like(weight_arr, dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool).reshape(-1)
        if active.shape != weight_arr.shape:
            raise ValueError("active_mask must align with weights.")
    if total <= 0 or not np.any(active):
        return tuple(0 for _ in range(weight_arr.shape[0]))

    positive = np.maximum(weight_arr, 0.0)
    if float(np.sum(positive[active])) <= 0.0:
        positive[active] = 1.0

    raw = np.zeros_like(positive)
    raw[active] = total * positive[active] / float(np.sum(positive[active]))
    counts = np.floor(raw).astype(np.int64)
    leftover = total - int(np.sum(counts))
    if leftover > 0:
        order = np.argsort(-(raw - counts), kind="mergesort")
        for idx in order:
            if not active[int(idx)]:
                continue
            counts[int(idx)] += 1
            leftover -= 1
            if leftover <= 0:
                break
    if leftover < 0:
        order = np.argsort(raw - counts, kind="mergesort")
        for idx in order:
            if not active[int(idx)] or counts[int(idx)] <= 0:
                continue
            counts[int(idx)] -= 1
            leftover += 1
            if leftover >= 0:
                break
    return tuple(int(max(0, value)) for value in counts.tolist())


def resolve_pca_component_layout(
    *,
    requested_total_components: int,
    pca_scheme: str,
    global_pca_components: int,
    band_pca_components: Sequence[int | float],
    k_bins: np.ndarray,
) -> dict[str, Any]:
    requested_total = max(0, int(requested_total_components))
    scheme = str(pca_scheme).strip().lower() or "global_pca"
    masks = build_k_band_masks(k_bins)
    active_bands = tuple(bool(np.any(mask)) for mask in masks)

    if scheme == "global_pca":
        empty_band_counts = tuple(0 for _ in PCA_BAND_LABELS)
        return {
            "scheme": scheme,
            "requested_total_components": requested_total,
            "resolved_total_components": requested_total,
            "requested_global_components": requested_total,
            "resolved_global_components": requested_total,
            "requested_band_components": empty_band_counts,
            "resolved_band_components": empty_band_counts,
            "band_labels": list(PCA_BAND_LABELS),
        }

    if scheme == "bandwise_pca":
        resolved_band = _allocate_component_counts(
            requested_total,
            band_pca_components,
            active_mask=active_bands,
        )
        return {
            "scheme": scheme,
            "requested_total_components": requested_total,
            "resolved_total_components": int(sum(resolved_band)),
            "requested_global_components": 0,
            "resolved_global_components": 0,
            "requested_band_components": tuple(int(max(0, value)) for value in band_pca_components),
            "resolved_band_components": resolved_band,
            "band_labels": list(PCA_BAND_LABELS),
        }

    if scheme != "global_plus_band_residual_pca":
        raise ValueError(
            "pca_scheme must be one of "
            "{'global_pca', 'bandwise_pca', 'global_plus_band_residual_pca'}, "
            f"got {pca_scheme!r}."
        )

    resolved_global = min(max(0, int(global_pca_components)), requested_total)
    residual_total = max(0, requested_total - resolved_global)
    resolved_band = _allocate_component_counts(
        residual_total,
        band_pca_components,
        active_mask=active_bands,
    )
    return {
        "scheme": scheme,
        "requested_total_components": requested_total,
        "resolved_total_components": int(resolved_global + sum(resolved_band)),
        "requested_global_components": int(max(0, global_pca_components)),
        "resolved_global_components": int(resolved_global),
        "requested_band_components": tuple(int(max(0, value)) for value in band_pca_components),
        "resolved_band_components": resolved_band,
        "band_labels": list(PCA_BAND_LABELS),
    }


def build_representation_component_groups(
    pca_layout: Mapping[str, Any],
) -> list[dict[str, Any]]:
    scheme = str(pca_layout.get("scheme", "global_pca")).strip().lower() or "global_pca"
    band_labels = [
        str(label).strip() or default_label
        for label, default_label in zip(
            pca_layout.get("band_labels", list(PCA_BAND_LABELS)),
            PCA_BAND_LABELS,
            strict=False,
        )
    ]
    if len(band_labels) != len(PCA_BAND_LABELS):
        band_labels = list(PCA_BAND_LABELS)

    groups: list[dict[str, Any]] = []
    cursor = 0

    def _append_group(
        *,
        group_key: str,
        group_label: str,
        group_kind: str,
        component_count: int,
        band_index: int | None,
    ) -> None:
        nonlocal cursor
        count = int(max(0, component_count))
        if count <= 0:
            return
        start = int(cursor)
        stop = int(cursor + count)
        groups.append(
            {
                "group_index": int(len(groups)),
                "group_key": str(group_key),
                "group_label": str(group_label),
                "group_kind": str(group_kind),
                "band_index": None if band_index is None else int(band_index),
                "component_start": start,
                "component_stop": stop,
                "component_count": count,
                "component_indices": list(range(start, stop)),
            }
        )
        cursor = stop

    if scheme == "global_pca":
        _append_group(
            group_key="global",
            group_label="global",
            group_kind="global",
            component_count=int(pca_layout.get("resolved_total_components", 0)),
            band_index=None,
        )
        return groups

    if scheme == "bandwise_pca":
        for band_index, (band_label, component_count) in enumerate(
            zip(
                band_labels,
                pca_layout.get("resolved_band_components", tuple(0 for _ in PCA_BAND_LABELS)),
                strict=True,
            )
        ):
            _append_group(
                group_key=f"band:{band_label}",
                group_label=str(band_label),
                group_kind="band",
                component_count=int(component_count),
                band_index=int(band_index),
            )
        return groups

    if scheme == "global_plus_band_residual_pca":
        _append_group(
            group_key="global",
            group_label="global",
            group_kind="global",
            component_count=int(pca_layout.get("resolved_global_components", 0)),
            band_index=None,
        )
        for band_index, (band_label, component_count) in enumerate(
            zip(
                band_labels,
                pca_layout.get("resolved_band_components", tuple(0 for _ in PCA_BAND_LABELS)),
                strict=True,
            )
        ):
            _append_group(
                group_key=f"band:{band_label}",
                group_label=str(band_label),
                group_kind="band",
                component_count=int(component_count),
                band_index=int(band_index),
            )
        return groups

    raise ValueError(
        "Unsupported PCA layout scheme for component groups, "
        f"got {pca_layout.get('scheme')!r}."
    )


def build_component_weight_vector_from_groups(
    component_groups: Sequence[Mapping[str, Any]],
    *,
    total_components: int,
    global_weight: float = 1.0,
    band_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    default_weight: float = 1.0,
) -> np.ndarray:
    weights = np.full((int(max(0, total_components)),), float(default_weight), dtype=np.float64)
    band_weight_arr = np.asarray(band_weights, dtype=np.float64).reshape(-1)
    if band_weight_arr.shape != (len(PCA_BAND_LABELS),):
        raise ValueError(
            f"band_weights must have shape [{len(PCA_BAND_LABELS)}], got {band_weight_arr.shape}."
        )
    for group in component_groups:
        indices = np.asarray(group.get("component_indices", []), dtype=np.int64).reshape(-1)
        if indices.size <= 0:
            continue
        kind = str(group.get("group_kind", "")).strip().lower()
        if kind == "global":
            weight = float(global_weight)
        elif kind == "band":
            band_index = int(group.get("band_index", -1))
            if 0 <= band_index < len(PCA_BAND_LABELS):
                weight = float(band_weight_arr[band_index])
            else:
                weight = float(default_weight)
        else:
            weight = float(default_weight)
        weights[indices] = weight
    return weights.astype(np.float64)


@dataclass(slots=True)
class BandwisePCAAdapter:
    k_size: int
    masks: tuple[np.ndarray, ...]
    band_labels: tuple[str, ...]
    band_pcas: tuple[PCA | None, ...]
    components_: np.ndarray = field(init=False, repr=False)
    mean_: np.ndarray = field(init=False, repr=False)
    explained_variance_ratio_: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        components: list[np.ndarray] = []
        means = np.zeros((self.k_size,), dtype=np.float64)
        explained: list[np.ndarray] = []
        for mask, pca in zip(self.masks, self.band_pcas, strict=True):
            if pca is None:
                continue
            means[np.asarray(mask, dtype=bool)] = np.asarray(pca.mean_, dtype=np.float64)
            local_components = np.asarray(pca.components_, dtype=np.float64)
            embedded = np.zeros((local_components.shape[0], self.k_size), dtype=np.float64)
            embedded[:, np.asarray(mask, dtype=bool)] = local_components
            components.append(embedded)
            explained.append(np.asarray(pca.explained_variance_ratio_, dtype=np.float64))
        self.components_ = (
            np.vstack(components).astype(np.float64)
            if components
            else np.empty((0, self.k_size), dtype=np.float64)
        )
        self.mean_ = means.astype(np.float64)
        self.explained_variance_ratio_ = (
            np.concatenate(explained).astype(np.float64)
            if explained
            else np.empty((0,), dtype=np.float64)
        )

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        score_arr = np.asarray(scores, dtype=np.float64)
        if score_arr.ndim == 1:
            score_arr = score_arr.reshape(1, -1)
        reconstructed = np.zeros((score_arr.shape[0], self.k_size), dtype=np.float64)
        cursor = 0
        for mask, pca in zip(self.masks, self.band_pcas, strict=True):
            if pca is None:
                continue
            width = int(pca.components_.shape[0])
            local_scores = score_arr[:, cursor : cursor + width]
            reconstructed[:, np.asarray(mask, dtype=bool)] = np.asarray(
                pca.inverse_transform(local_scores),
                dtype=np.float64,
            )
            cursor += width
        if cursor != score_arr.shape[1]:
            raise ValueError(
                f"Bandwise PCA score width mismatch: consumed {cursor}, got {score_arr.shape[1]}."
            )
        return reconstructed.astype(np.float64)


@dataclass(slots=True)
class GlobalPlusBandResidualPCAAdapter:
    global_pca: PCA | None
    residual_adapter: BandwisePCAAdapter
    k_size: int
    components_: np.ndarray = field(init=False, repr=False)
    mean_: np.ndarray = field(init=False, repr=False)
    explained_variance_ratio_: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        global_components = (
            np.asarray(self.global_pca.components_, dtype=np.float64)
            if self.global_pca is not None
            else np.empty((0, self.k_size), dtype=np.float64)
        )
        global_mean = (
            np.asarray(self.global_pca.mean_, dtype=np.float64)
            if self.global_pca is not None
            else np.zeros((self.k_size,), dtype=np.float64)
        )
        global_explained = (
            np.asarray(self.global_pca.explained_variance_ratio_, dtype=np.float64)
            if self.global_pca is not None
            else np.empty((0,), dtype=np.float64)
        )
        self.components_ = np.vstack([global_components, self.residual_adapter.components_]).astype(
            np.float64
        )
        self.mean_ = (global_mean + self.residual_adapter.mean_).astype(np.float64)
        self.explained_variance_ratio_ = np.concatenate(
            [global_explained, self.residual_adapter.explained_variance_ratio_]
        ).astype(np.float64)

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        score_arr = np.asarray(scores, dtype=np.float64)
        if score_arr.ndim == 1:
            score_arr = score_arr.reshape(1, -1)
        global_width = 0 if self.global_pca is None else int(self.global_pca.components_.shape[0])
        if self.global_pca is None:
            global_recon = np.zeros((score_arr.shape[0], self.k_size), dtype=np.float64)
        else:
            global_recon = np.asarray(
                self.global_pca.inverse_transform(score_arr[:, :global_width]),
                dtype=np.float64,
            )
        residual_scores = score_arr[:, global_width:]
        residual_recon = self.residual_adapter.inverse_transform(residual_scores)
        return (global_recon + residual_recon).astype(np.float64)


def _fit_vanilla_pca(
    target_batch: np.ndarray,
    *,
    n_components: int,
    random_seed: int,
) -> tuple[PCA, np.ndarray]:
    target_arr = np.asarray(target_batch, dtype=np.float64)
    resolved = int(min(max(0, n_components), target_arr.shape[0], target_arr.shape[1]))
    if resolved <= 0:
        raise ValueError("Resolved PCA component count must be positive.")
    pca = PCA(n_components=resolved, svd_solver="auto", random_state=int(random_seed))
    scores = np.asarray(pca.fit_transform(target_arr), dtype=np.float64)
    return pca, scores


def _fit_bandwise_pca(
    target_batch: np.ndarray,
    *,
    k_bins: np.ndarray,
    band_component_counts: Sequence[int],
    random_seed: int,
) -> tuple[BandwisePCAAdapter, np.ndarray, dict[str, Any]]:
    target_arr = np.asarray(target_batch, dtype=np.float64)
    masks = build_k_band_masks(k_bins)
    band_pcas: list[PCA | None] = []
    score_cols: list[np.ndarray] = []
    resolved_counts: list[int] = []
    per_band_metadata: list[dict[str, Any]] = []
    for band_idx, (label, mask, requested_count) in enumerate(
        zip(PCA_BAND_LABELS, masks, band_component_counts, strict=True)
    ):
        band_target = target_arr[:, np.asarray(mask, dtype=bool)]
        if band_target.shape[1] <= 0 or int(requested_count) <= 0:
            band_pcas.append(None)
            resolved_counts.append(0)
            per_band_metadata.append(
                {
                    "band_index": int(band_idx),
                    "band_label": str(label),
                    "requested_components": int(max(0, requested_count)),
                    "resolved_components": 0,
                    "k_count": int(band_target.shape[1]),
                }
            )
            continue
        pca, scores = _fit_vanilla_pca(
            band_target,
            n_components=int(requested_count),
            random_seed=int(random_seed + band_idx),
        )
        band_pcas.append(pca)
        score_cols.append(scores)
        resolved_counts.append(int(scores.shape[1]))
        per_band_metadata.append(
            {
                "band_index": int(band_idx),
                "band_label": str(label),
                "requested_components": int(max(0, requested_count)),
                "resolved_components": int(scores.shape[1]),
                "k_count": int(band_target.shape[1]),
            }
        )
    adapter = BandwisePCAAdapter(
        k_size=int(target_arr.shape[1]),
        masks=tuple(np.asarray(mask, dtype=bool) for mask in masks),
        band_labels=PCA_BAND_LABELS,
        band_pcas=tuple(band_pcas),
    )
    if score_cols:
        scores_arr = np.hstack(score_cols).astype(np.float64)
    else:
        scores_arr = np.empty((target_arr.shape[0], 0), dtype=np.float64)
    metadata = {
        "scheme": "bandwise_pca",
        "resolved_band_components": resolved_counts,
        "per_band": per_band_metadata,
    }
    return adapter, scores_arr, metadata


def fit_representation_pca(
    target_batch: np.ndarray,
    *,
    k_bins: np.ndarray,
    total_components: int,
    pca_scheme: str,
    random_seed: int,
    global_pca_components: int,
    band_pca_components: Sequence[int | float],
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    target_arr = np.asarray(target_batch, dtype=np.float64)
    layout = resolve_pca_component_layout(
        requested_total_components=int(total_components),
        pca_scheme=pca_scheme,
        global_pca_components=int(global_pca_components),
        band_pca_components=band_pca_components,
        k_bins=k_bins,
    )
    scheme = str(layout["scheme"])
    if scheme == "global_pca":
        pca, scores = _fit_vanilla_pca(
            target_arr,
            n_components=int(layout["resolved_global_components"]),
            random_seed=int(random_seed),
        )
        metadata = dict(layout)
        metadata["resolved_total_components"] = int(scores.shape[1])
        metadata["resolved_global_components"] = int(scores.shape[1])
        metadata["component_groups"] = build_representation_component_groups(metadata)
        return pca, scores.astype(np.float64), metadata

    if scheme == "bandwise_pca":
        adapter, scores, metadata = _fit_bandwise_pca(
            target_arr,
            k_bins=k_bins,
            band_component_counts=layout["resolved_band_components"],
            random_seed=int(random_seed),
        )
        merged = dict(layout)
        merged.update(metadata)
        merged["resolved_total_components"] = int(scores.shape[1])
        merged["component_groups"] = build_representation_component_groups(merged)
        return adapter, scores.astype(np.float64), merged

    global_pca: PCA | None
    global_scores: np.ndarray
    global_count = int(layout["resolved_global_components"])
    if global_count > 0:
        global_pca, global_scores = _fit_vanilla_pca(
            target_arr,
            n_components=global_count,
            random_seed=int(random_seed),
        )
        global_recon = np.asarray(global_pca.inverse_transform(global_scores), dtype=np.float64)
    else:
        global_pca = None
        global_scores = np.empty((target_arr.shape[0], 0), dtype=np.float64)
        global_recon = np.zeros_like(target_arr, dtype=np.float64)

    residual = target_arr - global_recon
    residual_adapter, residual_scores, residual_metadata = _fit_bandwise_pca(
        residual,
        k_bins=k_bins,
        band_component_counts=layout["resolved_band_components"],
        random_seed=int(random_seed + 97),
    )
    adapter = GlobalPlusBandResidualPCAAdapter(
        global_pca=global_pca,
        residual_adapter=residual_adapter,
        k_size=int(target_arr.shape[1]),
    )
    merged = dict(layout)
    merged.update(
        {
            "scheme": "global_plus_band_residual_pca",
            "global_components_resolved_from_fit": int(global_scores.shape[1]),
            "residual_bandwise": residual_metadata,
            "resolved_band_components": tuple(
                int(value) for value in residual_metadata["resolved_band_components"]
            ),
            "resolved_total_components": int(global_scores.shape[1] + residual_scores.shape[1]),
        }
    )
    merged["component_groups"] = build_representation_component_groups(merged)
    scores = np.hstack([global_scores, residual_scores]).astype(np.float64)
    return adapter, scores, merged
