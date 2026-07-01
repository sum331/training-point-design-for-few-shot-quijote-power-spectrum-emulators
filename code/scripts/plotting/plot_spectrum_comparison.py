"""Plot P(k) comparison (P_true vs P_pred) from test set results (TS6).

Data source: test_set_results.json. When file is missing, exits with message.
See??????????????TS6.

?????????????????**?? P(k)**??????????? P/P_pivot?? ??(k)??
???? CAMB ?? P ????? (Mpc/h)????? y ???????????? P(k) [(Mpc/h)?]?

??????? --test-set-results ????? artifacts/reports/runs/ ???
run ???? test_set_results.json?
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plotting._z2_plot_io import extract_mean_spectra, load_payload


def _setup_chinese_font() -> None:
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


_K_DISPLAY_MIN = 1e-2
_K_DISPLAY_MAX = 10.0
_RUNS_DIR = PROJECT_ROOT / "artifacts" / "reports" / "runs"


def _find_latest_test_set_results() -> Path | None:
    """Scan artifacts/reports/runs/ and return test_set_results.json from
    the most recent run directory (sorted by name descending)."""
    if not _RUNS_DIR.is_dir():
        return None
    run_dirs = sorted(
        (d for d in _RUNS_DIR.iterdir() if d.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for d in run_dirs:
        candidate = d / "test_set_results.json"
        if candidate.is_file():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot P(k) comparison from test set results.")
    parser.add_argument(
        "--test-set-results",
        type=Path,
        default=None,
        help="Path to test_set_results.json. ?????????? run ???",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. ????????????? run ???",
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=_K_DISPLAY_MIN,
        help=f"Plot x-axis lower bound (default {_K_DISPLAY_MIN}).",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=_K_DISPLAY_MAX,
        help=f"Plot x-axis upper bound (default {_K_DISPLAY_MAX}).",
    )
    return parser.parse_args()


def plot_spectrum_comparison(
    test_set_results_path: Path,
    output_path: Path,
    k_min: float = _K_DISPLAY_MIN,
    k_max: float = _K_DISPLAY_MAX,
) -> int:
    """Core plotting logic, callable both from CLI and programmatically."""
    if not test_set_results_path.exists():
        logging.warning("No test-set data file found: %s", test_set_results_path)
        return 0

    data = load_payload(test_set_results_path)
    try:
        k_bins, p_true_mean, p_pred_mean = extract_mean_spectra(data)
    except KeyError as exc:
        logging.warning("No spectrum payload found in %s: %s", test_set_results_path, exc)
        return 0
    if "p_true_batch" in data:
        n_pts = int(np.asarray(data["p_true_batch"]).shape[0])
    elif "truth_target" in data:
        n_pts = int(np.asarray(data["truth_target"]).shape[0])
    else:
        n_pts = int(data.get("test_set_size", data.get("validation_points", 0)))
    spectrum_type = data.get("spectrum_type", "unknown")

    st_name = (
        "Galaxy"
        if spectrum_type == "galaxy"
        else "Dark Matter"
        if spectrum_type == "dark_matter"
        else "Quijote CDM"
        if spectrum_type == "quijote_cdm"
        else str(spectrum_type)
    )

    if len(p_true_mean) != len(k_bins) or len(p_pred_mean) != len(k_bins):
        logging.error("test_set_results: k_bins and p_true_mean/p_pred_mean length mismatch")
        return 1

    if len(k_bins) >= 2 and not np.all(np.diff(k_bins) > 0):
        logging.error("test_set_results: k_bins must be strictly ascending")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(k_bins, p_true_mean, "k-", linewidth=1.5, label="CAMB Truth (P_true)")
    ax.loglog(k_bins, p_pred_mean, "--", color="tab:blue", linewidth=1.2, label="Emulator (P_pred)")
    ax.set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    ax.set_ylabel(r"$P(k)$")
    k_min_actual, k_max_actual = float(np.min(k_bins)), float(np.max(k_bins))
    ax.set_title(
        f"P(k) Comparison - spectrum: {st_name} - test set ({n_pts} pts) - "
        f"k in [{k_min_actual:.4g}, {k_max_actual:.4g}] h/Mpc"
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.set_xlim(k_min, min(k_max, float(np.max(k_bins))))
    ax.legend(loc="best", fontsize=10)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("P(k) comparison saved to: %s", output_path)
    return 0


def main() -> int:
    args = parse_args()

    test_set_path = args.test_set_results
    if test_set_path is None or not test_set_path.exists():
        found = _find_latest_test_set_results() if test_set_path is None else None
        if found is not None:
            test_set_path = found
            logging.info("Auto-detected latest test-set data: %s", test_set_path)
        else:
            logging.warning("No test-set data provided or file not found: %s", args.test_set_results)
            return 0

    output = args.output
    if output is None:
        output = test_set_path.parent / "spectrum_comparison.png"

    return plot_spectrum_comparison(
        test_set_results_path=test_set_path,
        output_path=output,
        k_min=args.k_min,
        k_max=args.k_max,
    )


if __name__ == "__main__":
    sys.exit(main())

