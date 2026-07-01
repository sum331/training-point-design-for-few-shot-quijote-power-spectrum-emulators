"""Plot KUN bias-proxy evidence for the cold-start design prior.

The KUN standard-geometry field is not treated as a calibrated point-wise
Quijote error estimator. This diagnostic asks a narrower question: whether
regions assigned a higher KUN bias-prior quantile are relevant to the Quijote
cold-start difficulty and whether PPR reduces the corresponding Sobol32 errors.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = PACKAGE_ROOT / "code"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from z2quijote.standard_geometry.interpolation import (  # noqa: E402
    ReliabilityWeightedLocalInterpolator,
)


BIAS_RUN_DIR = PACKAGE_ROOT / "data" / "standard_geometry_bias"
VALIDATION_POINTS_PATH = PACKAGE_ROOT / "data" / "validation" / "lhs256_validation_points.npz"
ABLATION_DIR = PACKAGE_ROOT / "data" / "ablation" / "condition_summaries"
GALLERY_DIR = PACKAGE_ROOT
FIGURE_DIR = GALLERY_DIR / "figures"
SOURCE_DIR = GALLERY_DIR / "data" / "figures" / "source_data"
FIGURE_STEM = "Fig12_kun_proxy_quijote_enrichment"


def _rank_quantile(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    support = np.asarray(support, dtype=float)
    support = support[np.isfinite(support)]
    support.sort()
    if support.size == 0:
        raise ValueError("Empty finite support for quantile ranking.")
    ranks = np.searchsorted(support, values, side="right")
    return ranks / float(support.size)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    try:
        from scipy.stats import spearmanr

        stat = spearmanr(x, y)
        return float(stat.statistic), float(stat.pvalue)
    except Exception:
        xr = np.argsort(np.argsort(x)).astype(float)
        yr = np.argsort(np.argsort(y)).astype(float)
        xr -= xr.mean()
        yr -= yr.mean()
        denom = np.sqrt(np.sum(xr * xr) * np.sum(yr * yr))
        return float(np.sum(xr * yr) / denom), None


def _condition_sample_p68(condition: str) -> np.ndarray:
    path = ABLATION_DIR / f"{condition}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data["metrics"]["sample_p68_relative_error"]
    return np.asarray(values, dtype=float)


def _binned_summary(
    quantile: np.ndarray,
    kun_bias: np.ndarray,
    sobol32_error: np.ndarray,
    ppr32_error: np.ndarray,
    final_error: np.ndarray,
    bins: np.ndarray,
) -> list[dict[str, float]]:
    top_threshold = float(np.quantile(sobol32_error, 0.75))
    improvement = sobol32_error - ppr32_error
    rows: list[dict[str, float]] = []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if i == len(bins) - 2:
            mask = (quantile >= lo) & (quantile <= hi)
        else:
            mask = (quantile >= lo) & (quantile < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        top_fraction = float(np.mean(sobol32_error[mask] >= top_threshold))
        rows.append(
            {
                "bin_id": i + 1,
                "q_lo": float(lo),
                "q_hi": float(hi),
                "n": n,
                "kun_quantile_mean": float(np.mean(quantile[mask])),
                "kun_bias_mean": float(np.mean(kun_bias[mask])),
                "sobol32_error_median": float(np.median(sobol32_error[mask])),
                "sobol32_error_q25": float(np.quantile(sobol32_error[mask], 0.25)),
                "sobol32_error_q75": float(np.quantile(sobol32_error[mask], 0.75)),
                "ppr32_error_median": float(np.median(ppr32_error[mask])),
                "ppr32_error_q25": float(np.quantile(ppr32_error[mask], 0.25)),
                "ppr32_error_q75": float(np.quantile(ppr32_error[mask], 0.75)),
                "ppr32_minus_sobol32_improvement_median": float(np.median(improvement[mask])),
                "ppr32_minus_sobol32_improvement_q25": float(np.quantile(improvement[mask], 0.25)),
                "ppr32_minus_sobol32_improvement_q75": float(np.quantile(improvement[mask], 0.75)),
                "final_method_error_median": float(np.median(final_error[mask])),
                "final_method_error_q25": float(np.quantile(final_error[mask], 0.25)),
                "final_method_error_q75": float(np.quantile(final_error[mask], 0.75)),
                "top_quartile_fraction": top_fraction,
                "enrichment_lift": top_fraction / 0.25,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    bias_field = np.load(BIAS_RUN_DIR / "standard_geometry_bias_field.npz")
    support = np.load(BIAS_RUN_DIR / "interpolator_support.npz")
    validation = np.load(VALIDATION_POINTS_PATH)

    interpolator = ReliabilityWeightedLocalInterpolator(
        support["theta_unit"],
        support["bias_mean"],
        support["accepted_count"],
        neighbors=96,
        fallback_neighbors=160,
        min_count=10,
        high_confidence_count=20,
    )
    kun_bias, kun_confidence = interpolator.predict(validation["theta_unit"])
    usable_support = (
        np.isfinite(bias_field["bias_mean"])
        & np.asarray(bias_field["usable"], dtype=bool)
        & (bias_field["accepted_count"] >= 10)
    )
    kun_quantile = _rank_quantile(kun_bias, bias_field["bias_mean"][usable_support])

    sobol32_error = _condition_sample_p68("sobol32")
    ppr32_error = _condition_sample_p68("ppr32")
    final_error = _condition_sample_p68("ppr32_plus_variance_bias_al32")
    improvement = sobol32_error - ppr32_error

    rho_sobol32, p_sobol32 = _spearman(kun_quantile, sobol32_error)
    rho_improvement, p_improvement = _spearman(kun_quantile, improvement)

    bins = np.linspace(0.0, 1.0, 6)
    rows = _binned_summary(
        kun_quantile,
        kun_bias,
        sobol32_error,
        ppr32_error,
        final_error,
        bins,
    )
    csv_path = SOURCE_DIR / f"{FIGURE_STEM}_source.csv"
    _write_csv(csv_path, rows)

    top_threshold = float(np.quantile(sobol32_error, 0.75))
    high_mask = kun_quantile >= 0.8
    low_mask = kun_quantile < 0.2
    high_top_fraction = float(np.mean(sobol32_error[high_mask] >= top_threshold))
    low_top_fraction = float(np.mean(sobol32_error[low_mask] >= top_threshold))
    high_improvement_median = float(np.median(improvement[high_mask]))
    low_improvement_median = float(np.median(improvement[low_mask]))

    summary = {
        "figure": FIGURE_STEM,
        "bias_field_run": "data/standard_geometry_bias/",
        "validation_points": "data/validation/lhs256_validation_points.npz",
        "n_validation": int(validation["theta_unit"].shape[0]),
        "difficulty_definition": "per-validation-point p68 over all k of abs relative error for the Sobol32 Quijote emulator",
        "kun_proxy_definition": "reliability-weighted interpolation of accepted-only standard-geometry KUN bias field, ranked by support quantile",
        "interpretation": "cold-start design-prior diagnostic, not calibrated point-wise Quijote error estimation",
        "kun_interpolation_confidence_median": float(np.median(kun_confidence)),
        "kun_interpolation_confidence_min": float(np.min(kun_confidence)),
        "spearman_sobol32_cold_start": {"rho": rho_sobol32, "p_value": p_sobol32},
        "spearman_ppr32_improvement_vs_sobol32": {
            "rho": rho_improvement,
            "p_value": p_improvement,
        },
        "top_quartile_error_threshold": top_threshold,
        "top_quantile_0p8_1_top_error_fraction": high_top_fraction,
        "top_quantile_0p8_1_enrichment_lift": high_top_fraction / 0.25,
        "low_quantile_0_0p2_top_error_fraction": low_top_fraction,
        "high_quantile_ppr32_improvement_median": high_improvement_median,
        "low_quantile_ppr32_improvement_median": low_improvement_median,
        "source_csv": f"data/figures/source_data/{FIGURE_STEM}_source.csv",
    }
    summary_path = SOURCE_DIR / f"{FIGURE_STEM}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    centers = np.array([0.5 * (r["q_lo"] + r["q_hi"]) for r in rows])
    sobol_med = np.array([r["sobol32_error_median"] for r in rows])
    sobol_q25 = np.array([r["sobol32_error_q25"] for r in rows])
    sobol_q75 = np.array([r["sobol32_error_q75"] for r in rows])
    top_frac = np.array([r["top_quartile_fraction"] for r in rows])
    imp_med = np.array([r["ppr32_minus_sobol32_improvement_median"] for r in rows])
    imp_q25 = np.array([r["ppr32_minus_sobol32_improvement_q25"] for r in rows])
    imp_q75 = np.array([r["ppr32_minus_sobol32_improvement_q75"] for r in rows])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.50), constrained_layout=True)

    blue = "#2a6fbb"
    orange = "#d96c06"
    teal = "#0b7f7c"
    gray = "#555555"

    axes[0].fill_between(centers, sobol_q25, sobol_q75, color=blue, alpha=0.18, linewidth=0)
    axes[0].plot(centers, sobol_med, color=blue, marker="o", markersize=3.4, lw=1.6)
    axes[0].set_xlabel("KUN bias-field quantile")
    axes[0].set_ylabel("Sobol32 Quijote\nper-sample p68")
    axes[0].set_title("a  Cold-start difficulty")
    axes[0].text(
        0.04,
        0.95,
        rf"$\rho={rho_sobol32:.2f}$",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        color=gray,
    )

    axes[1].bar(
        centers,
        top_frac,
        width=0.105,
        color=orange,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.75,
    )
    axes[1].axhline(0.25, color=gray, ls="--", lw=1.1)
    axes[1].text(
        0.98,
        0.265,
        "random rate",
        ha="right",
        va="bottom",
        color=gray,
        fontsize=7.0,
    )
    axes[1].set_xlabel("KUN bias-field quantile")
    axes[1].set_ylabel("Top-quartile Sobol32\nerror fraction")
    axes[1].set_title("b  Difficulty enrichment")

    axes[2].axhline(0.0, color=gray, ls="--", lw=1.1)
    axes[2].fill_between(centers, imp_q25, imp_q75, color=teal, alpha=0.18, linewidth=0)
    axes[2].plot(centers, imp_med, color=teal, marker="o", markersize=3.4, lw=1.6)
    axes[2].set_xlabel("KUN bias-field quantile")
    axes[2].set_ylabel("PPR32 improvement\nvs Sobol32")
    axes[2].set_title("c  Reduction after PPR")

    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="y", color="#d9d9d9", lw=0.55, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.5,
        -0.06,
        "KUN proxy is evaluated on fixed LHS256 validation coordinates. "
        "The plot tests the cold-start design prior, not point-wise error calibration.",
        ha="center",
        va="top",
        fontsize=7.0,
        color="#555555",
    )

    for ext in ("png", "pdf", "svg"):
        out = FIGURE_DIR / f"{FIGURE_STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
