"""Compare GP baseline results on the same figure.

??????????? run ???? gp_baseline / process1_gp_baseline?:
    python scripts/plotting/plot_gp_baseline_comparison.py

??????:
    python scripts/plotting/plot_gp_baseline_comparison.py \
        --results-a  artifacts/.../gp_baseline/test_set_results.json \
        --results-b  artifacts/.../process1_gp_baseline/test_set_results.json \
        --results-c  artifacts/.../standard_gp_baseline_128/test_set_results.json \
        --output     artifacts/.../gp_baseline_comparison.png

??????????:
    from scripts.plotting.plot_gp_baseline_comparison import plot_gp_baseline_comparison
    plot_gp_baseline_comparison(results_a=..., results_b=..., output=...)
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
from scipy.interpolate import PchipInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.plotting.plot_ratio_centerline import plot_ratio_centerline_from_test_set
from z2quijote.runtime_core.run_artifacts import derive_run_dir_from_artifact, run_results_path, run_results_subdir

_K_DISPLAY_MIN = 1e-2
_K_DISPLAY_MAX = 10.0
_SMOOTH_POINTS = 500
_MIN_POINTS_FOR_SPLINE = 4
_RUNS_DIR = PROJECT_ROOT / "artifacts" / "reports" / "runs"
_STANDARD_SUBDIR = "standard_gp_baseline_128"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _setup_chinese_font():
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
        if any(f.name == name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


def _smooth(k: np.ndarray, y: np.ndarray, n: int = _SMOOTH_POINTS):
    k = np.asarray(k, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if k.shape[0] < _MIN_POINTS_FOR_SPLINE:
        return k, y
    idx = np.argsort(k)
    logk = np.log10(np.maximum(k[idx], 1e-10))
    ys = y[idx]
    logk_fine = np.linspace(logk.min(), logk.max(), n)
    y_fine = PchipInterpolator(logk, ys)(logk_fine)
    return np.power(10.0, logk_fine), y_fine


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    k = np.asarray(data["k_bins"], dtype=np.float64)
    p_true = np.asarray(data["p_true_batch"], dtype=np.float64)
    p_pred = np.asarray(data["p_pred_batch"], dtype=np.float64)
    denom = np.maximum(p_true, 1e-12)
    rel = np.abs(p_pred - p_true) / denom
    idx = np.argsort(k)
    return {
        "k": k[idx],
        "p50": np.percentile(rel, 50, axis=0)[idx],
        "p68": np.percentile(rel, 68, axis=0)[idx],
        "p95": np.percentile(rel, 95, axis=0)[idx],
        "n_pts": int(data.get("test_set_size", p_true.shape[0])),
        "metadata": dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), dict) else {},
        "spectrum_type": data.get("spectrum_type", "unknown"),
        "k_le_1_p68": data.get("k_le_1_p68_relative_error"),
        "k_le_1_max": data.get("k_le_1_max_relative_error"),
    }


def _load_meta(results_path: Path) -> dict:
    """Try to load run_metadata.json next to test_set_results.json."""
    meta_path = results_path.parent / "run_metadata.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _label_from_meta(meta: dict, fallback: str) -> str:
    mode = meta.get("mode", "")
    source = meta.get("data_source", "")
    parameter_space = meta.get("parameter_space", "")
    train_n = meta.get("train_points") or meta.get("sobol_train_points") or meta.get("train_size")
    if mode == "gp_baseline":
        tag = f"Sobol resample ({train_n} pts)" if train_n else "Sobol resample"
    elif mode == "active_learning_validation":
        tag = f"Active learning ({train_n} pts)" if train_n else "Active learning"
    elif mode == "active_learning_partial_validation":
        tag = f"Active learning partial ({train_n} pts)" if train_n else "Active learning partial"
    elif mode == "fixed_budget_comparison":
        tag = f"Fixed-budget Sobol-GP ({train_n} pts)" if train_n else "Fixed-budget Sobol-GP"
    elif mode == "standard_sobol_gp_baseline" or source == "fixed_sobol_hifi":
        tag = f"Standard Sobol-GP ({train_n} pts)" if train_n else "Standard Sobol-GP"
    elif mode == "process1_gp_baseline" or source == "in_memory_historical_hifi":
        tag = f"Historical HiFi data ({train_n} pts)" if train_n else "Historical HiFi data"
    else:
        tag = fallback
    if parameter_space:
        tag = f"{tag} [{parameter_space}]"
    return tag


def _spectrum_label(series: dict, meta: dict) -> str:
    raw = str(meta.get("spectrum_type") or series.get("spectrum_type") or "unknown")
    if raw == "dark_matter":
        return "Dark Matter"
    if raw == "galaxy":
        return "Galaxy"
    if raw == "quijote_cdm":
        return "Quijote CDM"
    return raw


def _find_latest_run_dir() -> Path | None:
    """Return the most recent run directory under artifacts/reports/runs/."""
    if not _RUNS_DIR.is_dir():
        return None
    run_dirs = sorted(
        (d for d in _RUNS_DIR.iterdir() if d.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def _auto_detect_pair(run_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """In a run directory, look for gp_baseline / process1 / standard results."""
    a = run_results_subdir(run_dir, "gp_baseline") / "test_set_results.json"
    b = run_results_subdir(run_dir, "process1_gp_baseline") / "test_set_results.json"
    c = run_results_subdir(run_dir, _STANDARD_SUBDIR) / "test_set_results.json"
    return (
        a if a.is_file() else None,
        b if b.is_file() else None,
        c if c.is_file() else None,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare two GP baselines. ??????????? run ???",
    )
    p.add_argument("--results-a", type=Path, default=None,
                   help="gp_baseline ? test_set_results.json?????????")
    p.add_argument("--results-b", type=Path, default=None,
                   help="process1_gp_baseline ? test_set_results.json?????????")
    p.add_argument("--results-c", type=Path, default=None,
                   help="?? 128 ? Sobol-GP ? test_set_results.json?????????")
    p.add_argument("--label-a", type=str, default=None, help="Label override for A")
    p.add_argument("--label-b", type=str, default=None, help="Label override for B")
    p.add_argument("--label-c", type=str, default=None, help="Label override for C")
    p.add_argument("--output", type=Path, default=None,
                   help="????????????? run ????")
    p.add_argument("--target-accuracy", type=float, default=0.01)
    p.add_argument("--k-target-max", type=float, default=1.0)
    return p.parse_args()


COLOR_A = "tab:blue"
COLOR_B = "tab:orange"
COLOR_C = "tab:purple"


def _draw_series(ax, k, curves, label, color):
    ks, p50 = _smooth(k, curves["p50"])
    _, p68 = _smooth(k, curves["p68"])
    _, p95 = _smooth(k, curves["p95"])
    ax.fill_between(ks, p50, p95, alpha=0.12, color=color)
    ax.plot(ks, p68, color=color, linewidth=1.8, label=f"{label} - P68")
    ax.plot(ks, p95, color=color, linewidth=0.8, linestyle=":", alpha=0.6, label=f"{label} - P95")


def _draw_comparison(ax, series, target_acc, k_min, k_max, show_bands=True):
    if show_bands:
        ax.axvspan(max(k_min, 1e-2), min(k_max, 0.1), alpha=0.06, color="tab:gray")
        ax.axvspan(max(k_min, 0.1), min(k_max, 1.0), alpha=0.08, color="green")
        ax.axvspan(max(k_min, 1.0), min(k_max, 10.0), alpha=0.06, color="tab:orange")

    for item in series:
        _draw_series(
            ax,
            item["k"],
            {"p50": item["p50"], "p68": item["p68"], "p95": item["p95"]},
            item["label"],
            item["color"],
        )

    ax.axhline(target_acc, color="tab:red", ls="--", lw=1.0, label=f"???? {target_acc:.0e}")
    ax.axhline(0.0, color="black", ls="--", lw=0.8)

    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h/\mathrm{Mpc}$]")
    ax.set_ylabel("Relative Error")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xlim(k_min, k_max)


def plot_gp_baseline_comparison(
    results_a: Path,
    results_b: Path,
    output: Path,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    results_c: Path | None = None,
    label_c: str | None = None,
    target_accuracy: float = 0.01,
    k_target_max: float = 1.0,
) -> int:
    """Core comparison logic, callable both from CLI and programmatically.

    Returns 0 on success, 1 on error.
    """
    if not results_a.exists() or not results_b.exists():
        logging.warning("GP ????: ??????? (a=%s, b=%s)", results_a, results_b)
        return 1

    da = _load(results_a)
    db = _load(results_b)
    meta_a = _load_meta(results_a)
    meta_b = _load_meta(results_b)
    if not meta_a:
        meta_a = dict(da.get("metadata", {}))
    if not meta_b:
        meta_b = dict(db.get("metadata", {}))
    dc = _load(results_c) if results_c is not None and results_c.exists() else None
    meta_c = _load_meta(results_c) if results_c is not None and results_c.exists() else None
    if dc is not None and not meta_c:
        meta_c = dict(dc.get("metadata", {}))

    label_a = label_a or _label_from_meta(meta_a, "GP-A")
    label_b = label_b or _label_from_meta(meta_b, "GP-B")
    if dc is not None:
        label_c = label_c or _label_from_meta(meta_c or {}, "GP-C")

    st_zh = _spectrum_label(da, meta_a)
    n_pts = da["n_pts"]

    series_full = [
        {"k": da["k"], "p50": da["p50"], "p68": da["p68"], "p95": da["p95"], "label": label_a, "color": COLOR_A},
        {"k": db["k"], "p50": db["p50"], "p68": db["p68"], "p95": db["p95"], "label": label_b, "color": COLOR_B},
    ]
    if dc is not None and label_c is not None:
        series_full.append(
            {"k": dc["k"], "p50": dc["p50"], "p68": dc["p68"], "p95": dc["p95"], "label": label_c, "color": COLOR_C}
        )

    ymax_full = max(float(np.max(item["p95"])) for item in series_full) * 1.05

    output.parent.mkdir(parents=True, exist_ok=True)

    # ?? Full k range ??
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_comparison(ax, series_full, target_accuracy, _K_DISPLAY_MIN, _K_DISPLAY_MAX)
    ax.set_ylim(0.0, ymax_full)
    ax.set_title(f"GP baseline comparison - P-space relative error - spectrum: {st_zh} ({n_pts} pts)")

    info_lines = []
    if da["k_le_1_p68"] is not None:
        info_lines.append(f"{label_a}  k<=1 p68={da['k_le_1_p68']:.4e}")
    if db["k_le_1_p68"] is not None:
        info_lines.append(f"{label_b}  k<=1 p68={db['k_le_1_p68']:.4e}")
    if dc is not None and dc["k_le_1_p68"] is not None and label_c is not None:
        info_lines.append(f"{label_c}  k<=1 p68={dc['k_le_1_p68']:.4e}")
    if info_lines:
        ax.text(0.02, 0.98, "\n".join(info_lines),
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("[Comparison] Saved full-k comparison plot: %s", output)

    # ?? k ? 1 subplot ??
    stem, suffix = output.stem, output.suffix
    out_k_le_1 = output.parent / f"{stem}_k_le_1{suffix}"

    mask_a = da["k"] <= k_target_max
    mask_b = db["k"] <= k_target_max
    mask_c = dc["k"] <= k_target_max if dc is not None else None
    if np.any(mask_a) and np.any(mask_b):
        series_le1 = [
            {
                "k": da["k"][mask_a],
                "p50": da["p50"][mask_a],
                "p68": da["p68"][mask_a],
                "p95": da["p95"][mask_a],
                "label": label_a,
                "color": COLOR_A,
            },
            {
                "k": db["k"][mask_b],
                "p50": db["p50"][mask_b],
                "p68": db["p68"][mask_b],
                "p95": db["p95"][mask_b],
                "label": label_b,
                "color": COLOR_B,
            },
        ]
        if dc is not None and mask_c is not None and np.any(mask_c) and label_c is not None:
            series_le1.append(
                {
                    "k": dc["k"][mask_c],
                    "p50": dc["p50"][mask_c],
                    "p68": dc["p68"][mask_c],
                    "p95": dc["p95"][mask_c],
                    "label": label_c,
                    "color": COLOR_C,
                }
            )
        ymax_le1 = max(float(np.max(item["p95"])) for item in series_le1) * 1.05

        fig2, ax2 = plt.subplots(figsize=(9, 5.5))
        _draw_comparison(
            ax2,
            series_le1,
            target_accuracy,
            _K_DISPLAY_MIN,
            float(max(item["k"].max() for item in series_le1)) * 1.01,
            show_bands=False,
        )
        ax2.set_ylim(0.0, ymax_le1)
        ax2.set_title(
            f"GP baseline comparison - target region k<= {k_target_max:g} - "
            f"spectrum: {st_zh} ({n_pts} pts)"
        )
        info2 = []
        if da["k_le_1_p68"] is not None:
            info2.append(f"{label_a}  p68={da['k_le_1_p68']:.4e}")
        if db["k_le_1_p68"] is not None:
            info2.append(f"{label_b}  p68={db['k_le_1_p68']:.4e}")
        if dc is not None and dc["k_le_1_p68"] is not None and label_c is not None:
            info2.append(f"{label_c}  p68={dc['k_le_1_p68']:.4e}")
        if info2:
            ax2.text(0.02, 0.98, "\n".join(info2),
                     transform=ax2.transAxes, fontsize=9, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax2.legend(loc="upper right", fontsize=9)
        fig2.savefig(out_k_le_1, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        logging.info("[Comparison] Saved k<=1 comparison plot: %s", out_k_le_1)

    ratio_inputs: list[tuple[str, Path | None, str]] = [
        ("a", results_a, label_a),
        ("b", results_b, label_b),
    ]
    if results_c is not None and dc is not None and label_c is not None:
        ratio_inputs.append(("c", results_c, label_c))
    for suffix_name, result_path, label in ratio_inputs:
        if result_path is None:
            continue
        try:
            plot_ratio_centerline_from_test_set(
                result_path,
                output.parent / f"{output.stem}_{suffix_name}_ratio_centerline{output.suffix}",
                title_label=label,
            )
        except Exception as exc:  # pragma: no cover - plot generation should not fail the comparison.
            logging.warning("Ratio centerline plot skipped for %s: %s", result_path, exc)

    return 0


def main():
    args = parse_args()

    results_a = args.results_a
    results_b = args.results_b
    results_c = args.results_c
    output = args.output

    if results_a is None or results_b is None:
        run_dir = _find_latest_run_dir()
        if run_dir is None:
            logging.warning("????? run ???")
            return 1
        auto_a, auto_b, auto_c = _auto_detect_pair(run_dir)
        if results_a is None:
            results_a = auto_a
        if results_b is None:
            results_b = auto_b
        if results_c is None:
            results_c = auto_c
        if results_a is None or results_b is None:
            logging.warning(
                "?? run ?? %s ?????? gp_baseline ? process1_gp_baseline ? test_set_results.json",
                run_dir.name,
            )
            return 1
        logging.info("??????? run: %s", run_dir.name)
        logging.info("  results-a: %s", results_a)
        logging.info("  results-b: %s", results_b)
        if results_c is not None:
            logging.info("  results-c: %s", results_c)

    if output is None:
        output = run_results_path(derive_run_dir_from_artifact(results_a), "gp_baseline_comparison.png", create=True)

    return plot_gp_baseline_comparison(
        results_a=results_a,
        results_b=results_b,
        output=output,
        label_a=args.label_a,
        label_b=args.label_b,
        results_c=results_c,
        label_c=args.label_c,
        target_accuracy=args.target_accuracy,
        k_target_max=args.k_target_max,
    )


if __name__ == "__main__":
    sys.exit(main())

