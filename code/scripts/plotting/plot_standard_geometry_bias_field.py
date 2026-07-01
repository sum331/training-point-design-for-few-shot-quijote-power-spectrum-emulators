"""Plot observed standard-geometry absolute bias fields.

The plots in this script use only observed ``bias_mean`` values from
``standard_geometry_bias_field.npz``. Empty bins are left empty; no emulator,
image generation, or interpolation is used to fabricate a surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


PARAMETER_LABELS = {
    "Omega_m": r"$\Omega_m$",
    "Omega_b": r"$\Omega_b$",
    "h": r"$h$",
    "n_s": r"$n_s$",
    "A": r"$A$",
}


def _load_field(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _labels(names: np.ndarray) -> list[str]:
    out = []
    for name in names.astype(str).tolist():
        out.append(PARAMETER_LABELS.get(name, name))
    return out


def _median_hexbin_pairwise(
    theta_raw: np.ndarray,
    bias_percent: np.ndarray,
    labels: list[str],
    output_path: Path,
    *,
    gridsize: int,
    title: str,
) -> Path:
    dim = theta_raw.shape[1]
    finite = np.isfinite(bias_percent)
    if not np.any(finite):
        raise ValueError("No finite bias values are available for plotting.")
    vmin, vmax = np.nanquantile(bias_percent[finite], [0.02, 0.98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(bias_percent[finite]))
        vmax = float(np.nanmax(bias_percent[finite]))

    fig, axes = plt.subplots(dim, dim, figsize=(2.45 * dim, 2.35 * dim), squeeze=False)
    mappable = None
    for row in range(dim):
        for col in range(dim):
            ax = axes[row, col]
            if row < col:
                ax.axis("off")
                continue
            if row == col:
                ax.hist(theta_raw[:, col], bins=26, color="#D0D7DE", edgecolor="#6E7781", linewidth=0.6)
                ax.set_yticks([])
                ax.set_title(labels[col], fontsize=10)
            else:
                hb = ax.hexbin(
                    theta_raw[:, col],
                    theta_raw[:, row],
                    C=bias_percent,
                    gridsize=gridsize,
                    reduce_C_function=np.nanmedian,
                    mincnt=1,
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0.0,
                )
                mappable = hb
            if row == dim - 1:
                ax.set_xlabel(labels[col], fontsize=9)
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row], fontsize=9)
            elif col != 0:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=8)
    fig.suptitle(title, fontsize=14, y=0.995)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015)
        cbar.set_label("observed p68 absolute relative bias (%)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _pca2(theta_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = theta_unit - np.mean(theta_unit, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T
    return projected, vt[:2]


def _plot_pca_surface(
    theta_unit: np.ndarray,
    bias_percent: np.ndarray,
    accepted_count: np.ndarray,
    output_path: Path,
    *,
    title: str,
    gridsize: int,
) -> Path:
    xy, _ = _pca2(theta_unit)
    finite = np.isfinite(bias_percent)
    vmin, vmax = np.nanquantile(bias_percent[finite], [0.02, 0.98])
    fig, (ax_bias, ax_count) = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    hb_bias = ax_bias.hexbin(
        xy[:, 0],
        xy[:, 1],
        C=bias_percent,
        gridsize=gridsize,
        reduce_C_function=np.nanmedian,
        mincnt=1,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.0,
    )
    ax_bias.set_title("Median observed bias in PCA projection")
    ax_bias.set_xlabel("PC1 of unit parameter box")
    ax_bias.set_ylabel("PC2 of unit parameter box")
    cbar_bias = fig.colorbar(hb_bias, ax=ax_bias)
    cbar_bias.set_label("observed p68 absolute relative bias (%)")

    hb_count = ax_count.hexbin(
        xy[:, 0],
        xy[:, 1],
        C=accepted_count,
        gridsize=gridsize,
        reduce_C_function=np.nanmedian,
        mincnt=1,
        cmap="viridis",
        linewidths=0.0,
    )
    ax_count.set_title("Median accepted geometry count")
    ax_count.set_xlabel("PC1 of unit parameter box")
    ax_count.set_ylabel("PC2 of unit parameter box")
    cbar_count = fig.colorbar(hb_count, ax=ax_count)
    cbar_count.set_label("accepted count across 600 designs")
    fig.suptitle(title, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_distribution_audit(
    bias_percent: np.ndarray,
    accepted_count: np.ndarray,
    usable: np.ndarray,
    high_confidence: np.ndarray,
    output_path: Path,
) -> Path:
    fig, (ax_bias, ax_count) = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    bins = np.linspace(0.0, min(float(np.nanquantile(bias_percent, 0.995)), 10.0), 45)
    ax_bias.hist(bias_percent, bins=bins, color="#9D4EDD", alpha=0.78, edgecolor="white")
    q = np.nanquantile(bias_percent, [0.5, 0.68, 0.9, 0.95, 0.99])
    for value, label in zip(q, ["q50", "q68", "q90", "q95", "q99"], strict=True):
        ax_bias.axvline(value, linestyle="--", linewidth=1.1, alpha=0.8)
        ax_bias.text(value, ax_bias.get_ylim()[1] * 0.92, label, rotation=90, va="top", ha="right", fontsize=8)
    ax_bias.set_xlabel("observed p68 absolute relative bias (%)")
    ax_bias.set_ylabel("reference point count")
    ax_bias.set_title("Bias distribution over usable observed points")
    ax_bias.grid(True, axis="y", alpha=0.2)

    max_count = int(np.nanmax(accepted_count))
    ax_count.hist(accepted_count, bins=np.arange(0, max_count + 2) - 0.5, color="#2A9D8F", alpha=0.78)
    ax_count.axvline(10, color="#E76F51", linestyle="--", linewidth=1.2, label="usable >= 10")
    ax_count.axvline(20, color="#264653", linestyle="--", linewidth=1.2, label="high confidence >= 20")
    ax_count.set_xlim(-1, min(max_count + 1, 140))
    ax_count.set_xlabel("accepted count across 600 designs")
    ax_count.set_ylabel("reference point count")
    ax_count.set_title(
        f"Geometry support: usable={np.mean(usable):.1%}, high-confidence={np.mean(high_confidence):.1%}"
    )
    ax_count.legend(frameon=False)
    ax_count.grid(True, axis="y", alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_bias_field(
    field_path: Path,
    output_dir: Path,
    *,
    min_count: int = 10,
    high_confidence_min_count: int = 20,
    gridsize: int = 28,
) -> dict[str, object]:
    field = _load_field(Path(field_path).resolve())
    theta_raw = np.asarray(field["theta_raw"], dtype=np.float64)
    theta_unit = np.asarray(field["theta_unit"], dtype=np.float64)
    names = np.asarray(field["theta_names"]).astype(str)
    bias = np.asarray(field["bias_mean"], dtype=np.float64)
    accepted_count = np.asarray(field["accepted_count"], dtype=np.int64)
    usable = np.isfinite(bias) & (accepted_count >= int(min_count))
    high_confidence = np.isfinite(bias) & (accepted_count >= int(high_confidence_min_count))
    if "usable" in field:
        usable = usable & np.asarray(field["usable"], dtype=bool)
    if "high_confidence" in field:
        high_confidence = high_confidence & np.asarray(field["high_confidence"], dtype=bool)

    theta_raw_usable = theta_raw[usable]
    theta_unit_usable = theta_unit[usable]
    bias_percent = bias[usable] * 100.0
    accepted_usable = accepted_count[usable]
    labels = _labels(names)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        _median_hexbin_pairwise(
            theta_raw_usable,
            bias_percent,
            labels,
            output_dir / "image2_true_absolute_bias_pairwise_surface.png",
            gridsize=gridsize,
            title=(
                "PPR standard-geometry absolute bias field "
                f"(observed points only, accepted_count >= {min_count})"
            ),
        ),
        _plot_pca_surface(
            theta_unit_usable,
            bias_percent,
            accepted_usable,
            output_dir / "image2_true_absolute_bias_pca_surface.png",
            gridsize=gridsize,
            title="PPR standard-geometry absolute bias field, true observed support",
        ),
        _plot_distribution_audit(
            bias_percent,
            accepted_usable,
            usable,
            high_confidence,
            output_dir / "image2_true_absolute_bias_distribution_audit.png",
        ),
    ]
    summary = {
        "field_path": str(Path(field_path).resolve()),
        "output_dir": str(output_dir),
        "reference_size": int(theta_raw.shape[0]),
        "finite_count": int(np.sum(np.isfinite(bias))),
        "usable_min_count": int(min_count),
        "usable_count": int(np.sum(usable)),
        "high_confidence_min_count": int(high_confidence_min_count),
        "high_confidence_count": int(np.sum(high_confidence)),
        "bias_percent_quantiles_usable": {
            f"q{int(q * 100):02d}": float(v)
            for q, v in zip(
                [0.0, 0.1, 0.25, 0.5, 0.68, 0.75, 0.9, 0.95, 0.99, 1.0],
                np.nanquantile(bias_percent, [0.0, 0.1, 0.25, 0.5, 0.68, 0.75, 0.9, 0.95, 0.99, 1.0]),
                strict=True,
            )
        },
        "outputs": [str(path) for path in outputs],
    }
    summary_path = output_dir / "image2_true_absolute_bias_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot true observed standard-geometry PPR absolute bias field.")
    parser.add_argument("--field", type=Path, required=True, help="standard_geometry_bias_field.npz path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plot outputs.")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--high-confidence-min-count", type=int, default=20)
    parser.add_argument("--gridsize", type=int, default=28)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = plot_bias_field(
        field_path=args.field,
        output_dir=args.output_dir,
        min_count=int(args.min_count),
        high_confidence_min_count=int(args.high_confidence_min_count),
        gridsize=int(args.gridsize),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
