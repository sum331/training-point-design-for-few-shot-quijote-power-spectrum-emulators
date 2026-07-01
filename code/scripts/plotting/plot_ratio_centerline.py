"""Plot a single signed prediction/truth ratio centerline from test-set results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plotting._z2_plot_io import extract_ratio_percent_batch, load_payload

_SMOOTH_WINDOW_DEFAULT = 31
_Y_STEP_PERCENT = 0.025
_MIN_Y_BOUND_PERCENT = 0.05
_DEFAULT_K_MARKERS = (0.07, 0.5, 1.0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _setup_chinese_font() -> None:
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
        if any(font.name == name for font in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


def _moving_average(y: np.ndarray, width: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y
    width = max(1, int(width))
    if width <= 1 or y.size < 3:
        return y.copy()
    if width % 2 == 0:
        width += 1
    width = min(width, y.size if y.size % 2 == 1 else y.size - 1)
    if width < 3:
        return y.copy()
    pad = width // 2
    padded = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _adaptive_symmetric_ylim(curve_percent: np.ndarray) -> float:
    finite = np.asarray(curve_percent, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return _MIN_Y_BOUND_PERCENT
    max_abs = float(np.max(np.abs(finite)))
    if max_abs <= 0.0:
        return _MIN_Y_BOUND_PERCENT
    bound = np.ceil(max_abs * 1.2 / _Y_STEP_PERCENT) * _Y_STEP_PERCENT
    return float(max(_MIN_Y_BOUND_PERCENT, bound))


def _title_from_payload(payload: dict[str, object], fallback: str) -> str:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    mode = str(metadata.get("mode") or payload.get("mode") or "").strip()
    source = str(metadata.get("data_source") or payload.get("data_source") or "").strip()
    spectrum = str(metadata.get("spectrum_type") or payload.get("spectrum_type") or "").strip()
    train_points = (
        metadata.get("train_points")
        or metadata.get("train_size")
        or payload.get("train_points")
        or payload.get("train_size")
    )
    parts = [part for part in (fallback, mode, source, spectrum) if part]
    title = " - ".join(dict.fromkeys(parts))
    if train_points is not None:
        title = f"{title} ({train_points} pts)"
    return title


def _signed_center_curve(ratio_percent: np.ndarray) -> np.ndarray:
    return np.median(ratio_percent, axis=0)


def _signed_p68_curve(ratio_percent: np.ndarray) -> np.ndarray:
    abs_ratio = np.abs(ratio_percent)
    abs_p68 = np.percentile(abs_ratio, 68.0, axis=0)
    nearest = np.argmin(np.abs(abs_ratio - abs_p68[None, :]), axis=0)
    return ratio_percent[nearest, np.arange(ratio_percent.shape[1])]


def _compute_curve(ratio_percent: np.ndarray, statistic: str) -> np.ndarray:
    key = statistic.strip().lower().replace("-", "_")
    if key in {"center", "median", "signed_center", "signed_median"}:
        return _signed_center_curve(ratio_percent)
    if key in {"signed_p68", "p68"}:
        return _signed_p68_curve(ratio_percent)
    raise ValueError(f"Unknown ratio centerline statistic: {statistic!r}")


def plot_ratio_centerline_from_test_set(
    test_set_results_path: Path,
    output_path: Path,
    *,
    title_label: str = "Validation",
    statistic: str = "center",
    smooth_window: int = _SMOOTH_WINDOW_DEFAULT,
    k_markers: Iterable[float] = _DEFAULT_K_MARKERS,
    write_csv: bool = True,
) -> list[Path]:
    """Plot one signed ratio line with y-limits driven by that line only.

    The default statistic is the signed centerline
    ``median(P_pred/P_true - 1)`` in percent. A signed p68 line can be requested
    with ``statistic="signed_p68"``; it is intentionally not the default because
    p68 is an error-amplitude statistic and often requires a much wider y-axis.
    """

    input_path = Path(test_set_results_path).resolve()
    payload = load_payload(input_path)
    try:
        k_bins, ratio_percent = extract_ratio_percent_batch(payload)
    except (KeyError, ValueError) as exc:
        logging.warning("Ratio centerline skipped for %s; %s", input_path, exc)
        return []

    if ratio_percent.ndim != 2:
        raise ValueError(f"ratio batch must be 2D, got {ratio_percent.shape}.")
    if k_bins.ndim != 1 or k_bins.shape[0] != ratio_percent.shape[1]:
        raise ValueError(f"k_bins must align with spectra, got {k_bins.shape} vs {ratio_percent.shape}.")

    order = np.argsort(k_bins)
    k_sorted = k_bins[order]
    ratio_percent = ratio_percent[:, order]
    raw_curve = _compute_curve(ratio_percent, statistic)
    curve = _moving_average(raw_curve, smooth_window)
    y_bound = _adaptive_symmetric_ylim(curve)

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    title = _title_from_payload(payload, title_label)

    fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    ax.plot(k_sorted, curve, color="#1d4ed8", linewidth=2.4, label=str(statistic).replace("_", " "))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.70)
    ax.set_xscale("log")
    ax.set_ylim(-y_bound, y_bound)
    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P_{\mathrm{pred}}/P_{\mathrm{true}}-1$ (%)")
    ax.set_title(f"{title} ratio residual")
    ax.grid(True, which="both", alpha=0.22)
    for marker in k_markers:
        marker = float(marker)
        if float(k_sorted.min()) < marker < float(k_sorted.max()):
            ax.axvline(marker, color="black", linestyle="--", linewidth=0.9, alpha=0.45)
            ax.text(marker, y_bound, f" k={marker:g}", ha="left", va="top", fontsize=9, alpha=0.75)
    ax.legend(loc="upper left")
    fig.savefig(output, dpi=220)
    plt.close(fig)

    outputs = [output]
    if write_csv:
        csv_path = output.with_suffix(".csv")
        center_curve = _signed_center_curve(ratio_percent)
        signed_p68_curve = _signed_p68_curve(ratio_percent)
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write(
                "k,selected_ratio_percent,selected_ratio_smoothed_percent,"
                "center_ratio_percent,signed_p68_ratio_percent\n"
            )
            for values in zip(k_sorted, raw_curve, curve, center_curve, signed_p68_curve):
                handle.write(",".join(f"{float(value):.12g}" for value in values) + "\n")
        outputs.append(csv_path)

    logging.info("Ratio centerline plot saved to: %s", output)
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a single signed ratio centerline from test_set_results.json.")
    parser.add_argument("--test-set-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title-label", type=str, default="Validation")
    parser.add_argument("--statistic", type=str, default="center", choices=("center", "median", "signed_p68", "p68"))
    parser.add_argument("--smooth-window", type=int, default=_SMOOTH_WINDOW_DEFAULT)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = plot_ratio_centerline_from_test_set(
        args.test_set_results,
        args.output,
        title_label=args.title_label,
        statistic=args.statistic,
        smooth_window=args.smooth_window,
        write_csv=not bool(args.no_csv),
    )
    for path in outputs:
        print(f"[Plot] {path}", flush=True)
    return 0 if outputs else 1


if __name__ == "__main__":
    raise SystemExit(main())
