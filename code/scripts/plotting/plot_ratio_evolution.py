"""Plot the validation relative-error envelope from test-set results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plotting._z2_plot_io import extract_relative_error_batch, load_payload


_K_DISPLAY_MIN = 1.0e-2
_K_DISPLAY_MAX = 10.0
_SMOOTH_POINTS_DEFAULT = 500
_MIN_POINTS_FOR_SPLINE = 4


def _setup_chinese_font() -> None:
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
        if any(font.name == name for font in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _smooth_curve(k: np.ndarray, y: np.ndarray, n_fine: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if k.shape[0] != y.shape[0] or k.shape[0] < _MIN_POINTS_FOR_SPLINE:
        return k, y
    idx = np.argsort(k)
    log_k = np.log10(np.maximum(k[idx], 1.0e-10))
    y_sorted = y[idx]
    fine_count = min(int(n_fine), max(_MIN_POINTS_FOR_SPLINE * 2, k.shape[0] * 2))
    log_k_fine = np.linspace(log_k.min(), log_k.max(), fine_count, dtype=np.float64)
    interp = PchipInterpolator(log_k, y_sorted)
    return np.power(10.0, log_k_fine), np.asarray(interp(log_k_fine), dtype=np.float64)


def _output_k_le_1(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_k_le_1{output_path.suffix}"


def plot_ratio_evolution_from_test_set(
    test_set_results_path: Path,
    output_path: Path,
    *,
    target_accuracy: float = 0.01,
    k_target_max: float = 1.0,
    k_min: float = _K_DISPLAY_MIN,
    k_max: float = _K_DISPLAY_MAX,
    smooth_points: int = _SMOOTH_POINTS_DEFAULT,
) -> list[Path]:
    payload = load_payload(Path(test_set_results_path).resolve())
    k_bins, rel_batch = extract_relative_error_batch(payload)
    if "p_true_batch" in payload:
        test_set_size = int(np.asarray(payload["p_true_batch"]).shape[0])
    elif "truth_target" in payload:
        test_set_size = int(np.asarray(payload["truth_target"]).shape[0])
    else:
        test_set_size = int(payload.get("test_set_size", payload.get("validation_points", 0)))
    rel_p68 = np.percentile(rel_batch, 68.0, axis=0)
    rel_windowed = np.asarray(payload.get("power_relative_error_p68_windowed", rel_p68), dtype=np.float64)
    rel_p50 = np.percentile(rel_batch, 50, axis=0)
    rel_p95 = np.percentile(rel_batch, 95, axis=0)

    sort_idx = np.argsort(k_bins)
    k_sorted = k_bins[sort_idx]
    rel_p50 = rel_p50[sort_idx]
    rel_p68 = rel_p68[sort_idx]
    rel_p95 = rel_p95[sort_idx]
    rel_windowed = rel_windowed[sort_idx]

    k_plot, rel_p50_plot = _smooth_curve(k_sorted, rel_p50, smooth_points)
    _, rel_p68_plot = _smooth_curve(k_sorted, rel_p68, smooth_points)
    _, rel_p95_plot = _smooth_curve(k_sorted, rel_p95, smooth_points)
    _, rel_windowed_plot = _smooth_curve(k_sorted, rel_windowed, smooth_points)

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    spectrum_type = str(payload.get("spectrum_type", "unknown"))
    spectrum_label = (
        "Dark Matter" if spectrum_type == "dark_matter" else
        "Galaxy" if spectrum_type == "galaxy" else
        "Quijote CDM" if spectrum_type == "quijote_cdm" else
        spectrum_type
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(max(k_min, 1.0e-2), min(k_max, 0.1), alpha=0.08, color="tab:gray")
    ax.axvspan(max(k_min, 0.1), min(k_max, 1.0), alpha=0.12, color="green")
    ax.axvspan(max(k_min, 1.0), min(k_max, 10.0), alpha=0.08, color="tab:orange")
    ax.fill_between(
        k_plot,
        rel_p50_plot,
        rel_p95_plot,
        alpha=0.20,
        color="steelblue",
        label="Envelope (P50 - P95)",
    )
    ax.plot(k_plot, rel_p68_plot, color="tab:blue", linewidth=1.6, label="P68")
    ax.plot(
        k_plot,
        rel_windowed_plot,
        color="tab:green",
        linewidth=1.2,
        linestyle="-.",
        label="0.05 dex sliding P68",
    )
    ax.axhline(
        float(target_accuracy),
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"Target accuracy {target_accuracy:.2e}",
    )
    ax.set_xscale("log")
    ax.set_xlim(k_min, min(k_max, float(np.max(k_sorted))))
    ax.set_ylim(0.0, max(float(np.max(rel_p95)) * 1.05, 1.0e-12))
    ax.set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    ax.set_ylabel("Relative error")
    ax.set_title(
        f"Validation relative-error envelope - {spectrum_label} ({test_set_size} pts)"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    summary_lines: list[str] = []
    for key, label in (
        ("k_le_1_p68_relative_error", "k<=1 p68"),
        ("k_le_1_max_relative_error", "k<=1 max"),
        ("band_relative_error_mid_integrated_p68", "mid-band integrated p68"),
    ):
        value = payload.get(key)
        if value is not None:
            summary_lines.append(f"{label}={float(value):.3e}")
    if summary_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )
    ax.legend(loc="upper right", fontsize=10)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    outputs = [output]
    mask_le_1 = k_sorted <= float(k_target_max)
    if np.any(mask_le_1):
        k_le_1 = k_sorted[mask_le_1]
        rel_p50_le_1 = rel_p50[mask_le_1]
        rel_p68_le_1 = rel_p68[mask_le_1]
        rel_p95_le_1 = rel_p95[mask_le_1]
        k_le_1_plot, rel_p50_le_1_plot = _smooth_curve(k_le_1, rel_p50_le_1, smooth_points)
        _, rel_p68_le_1_plot = _smooth_curve(k_le_1, rel_p68_le_1, smooth_points)
        _, rel_p95_le_1_plot = _smooth_curve(k_le_1, rel_p95_le_1, smooth_points)

        output_le_1 = _output_k_le_1(output)
        fig_le_1, ax_le_1 = plt.subplots(figsize=(8.5, 5))
        ax_le_1.fill_between(
            k_le_1_plot,
            rel_p50_le_1_plot,
            rel_p95_le_1_plot,
            alpha=0.20,
            color="steelblue",
            label="Envelope (P50 - P95)",
        )
        ax_le_1.plot(k_le_1_plot, rel_p68_le_1_plot, color="tab:blue", linewidth=1.6, label="P68")
        ax_le_1.axhline(
            float(target_accuracy),
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"Target accuracy {target_accuracy:.2e}",
        )
        ax_le_1.set_xscale("log")
        ax_le_1.set_xlim(k_min, min(k_max, float(np.max(k_le_1)) * 1.01))
        ax_le_1.set_ylim(0.0, max(float(np.max(rel_p95_le_1)) * 1.05, 1.0e-12))
        ax_le_1.set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
        ax_le_1.set_ylabel("Relative error")
        ax_le_1.set_title(f"Validation relative-error envelope - {spectrum_label} (k<=1)")
        ax_le_1.grid(True, which="both", linestyle="--", alpha=0.3)
        ax_le_1.legend(loc="upper right", fontsize=10)
        fig_le_1.savefig(output_le_1, dpi=300, bbox_inches="tight")
        plt.close(fig_le_1)
        outputs.append(output_le_1)

    for path in outputs:
        logging.info("Ratio evolution plot saved to: %s", path)
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation relative-error curves from test_set_results.json.",
    )
    parser.add_argument("--test-set-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-accuracy", type=float, default=0.01)
    parser.add_argument("--k-target-max", type=float, default=1.0)
    parser.add_argument("--k-min", type=float, default=_K_DISPLAY_MIN)
    parser.add_argument("--k-max", type=float, default=_K_DISPLAY_MAX)
    parser.add_argument("--smooth-points", type=int, default=_SMOOTH_POINTS_DEFAULT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = plot_ratio_evolution_from_test_set(
        args.test_set_results,
        args.output,
        target_accuracy=args.target_accuracy,
        k_target_max=args.k_target_max,
        k_min=args.k_min,
        k_max=args.k_max,
        smooth_points=args.smooth_points,
    )
    for path in outputs:
        print(f"[Plot] {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
