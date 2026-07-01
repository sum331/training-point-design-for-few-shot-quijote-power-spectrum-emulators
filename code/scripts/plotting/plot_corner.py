"""Plot the active-learning training-point distribution for one run."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import corner
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from z2quijote.runtime_core.run_artifacts import run_process_path, run_results_path


_PARAMETER_LABELS: dict[str, str] = {
    "Omegab": r"$\Omega_\mathrm{b}$",
    "Omegacb": r"$\Omega_\mathrm{cb}$",
    "Omegam": r"$\Omega_\mathrm{m}$",
    "Omega_m": r"$\Omega_\mathrm{m}$",
    "Omega_b": r"$\Omega_\mathrm{b}$",
    "H0": r"$H_0$",
    "h": r"$h$",
    "ns": r"$n_\mathrm{s}$",
    "n_s": r"$n_\mathrm{s}$",
    "A": r"$10^9 A_\mathrm{s}$",
    "sigma_8": r"$\sigma_8$",
    "w": r"$w_0$",
    "wa": r"$w_a$",
    "mnu": r"$\Sigma m_\nu$",
}

_INITIAL_FACE = "#F2C14E"
_INITIAL_EDGE = "#7A4F01"
_ITER_FACE = "#D81B60"
_ITER_EDGE = "#4A148C"


def _setup_chinese_font() -> None:
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
        if any(font.name == name for font in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stack_batches(batches: list[object], theta_dim: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for batch in batches:
        arr = np.asarray(batch, dtype=np.float64)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[1] == theta_dim:
            rows.append(arr)
    if not rows:
        return np.empty((0, theta_dim), dtype=np.float64)
    return np.vstack(rows).astype(np.float64)


def _resolve_labels(parameter_names: list[str], theta_dim: int) -> list[str]:
    names = list(parameter_names[:theta_dim])
    if len(names) < theta_dim:
        names.extend(f"theta_{idx + 1}" for idx in range(len(names), theta_dim))
    return [_PARAMETER_LABELS.get(name, name) for name in names]


def plot_corner_distribution(run_dir: Path, output_path: Path | None = None) -> Path:
    run_dir = Path(run_dir).resolve()
    summary_path = run_process_path(run_dir, "training_point_summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"Training point summary not found: {summary_path}")

    summary = _load_json(summary_path)
    initial = np.asarray(summary.get("initial_raw_thetas", []), dtype=np.float64)
    if initial.ndim == 1 and initial.size > 0:
        initial = initial.reshape(1, -1)
    if initial.ndim != 2:
        raise ValueError(f"initial_raw_thetas must be 2D, got {initial.shape}.")

    theta_dim = int(initial.shape[1]) if initial.size > 0 else 8
    iterative = _stack_batches(
        list(summary.get("selected_raw_thetas_by_iteration", [])),
        theta_dim,
    )
    point_groups = [arr for arr in (initial, iterative) if arr.size > 0]
    if not point_groups:
        raise ValueError("No training points were found in training_point_summary.json.")

    all_points = np.vstack(point_groups).astype(np.float64)
    labels = _resolve_labels(list(summary.get("parameter_names", [])), all_points.shape[1])
    resolved_output = (
        Path(output_path).resolve()
        if output_path is not None
        else run_results_path(run_dir, "plots", "corner_distribution.png", create=True)
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    if all_points.shape[0] <= all_points.shape[1]:
        fig, ax = plt.subplots(figsize=(8.5, 2.8))
        ax.axis("off")
        ax.text(
            0.01,
            0.6,
            (
                "Corner plot downgraded: not enough samples.\n"
                f"samples={all_points.shape[0]}, dims={all_points.shape[1]}"
            ),
            fontsize=11,
            va="center",
            ha="left",
        )
        fig.savefig(resolved_output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return resolved_output

    figure = corner.corner(
        all_points,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        color="#444444",
        plot_datapoints=False,
        plot_density=False,
        plot_contours=False,
        range=[0.995] * all_points.shape[1],
    )

    first_lower_idx: int | None = None
    ndim = all_points.shape[1]
    if len(figure.axes) >= ndim * ndim:
        for row in range(ndim):
            for col in range(ndim):
                if col >= row:
                    continue
                idx = row * ndim + col
                if first_lower_idx is None:
                    first_lower_idx = idx
                ax = figure.axes[idx]
                if initial.size > 0:
                    ax.scatter(
                        initial[:, col],
                        initial[:, row],
                        c=_INITIAL_FACE,
                        s=64,
                        alpha=0.96,
                        marker="P",
                        edgecolors=_INITIAL_EDGE,
                        linewidths=0.8,
                        label="Initial Sobol" if idx == first_lower_idx else None,
                        zorder=5,
                    )
                if iterative.size > 0:
                    ax.scatter(
                        iterative[:, col],
                        iterative[:, row],
                        c=_ITER_FACE,
                        s=56,
                        alpha=0.92,
                        marker="X",
                        edgecolors=_ITER_EDGE,
                        linewidths=0.8,
                        label="Selected by iteration" if idx == first_lower_idx else None,
                        zorder=6,
                    )
        if first_lower_idx is not None:
            figure.axes[first_lower_idx].legend(
                loc="upper right",
                fontsize=9,
                frameon=True,
                framealpha=0.9,
                facecolor="white",
            )

    figure.savefig(resolved_output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logging.info("Corner plot saved to: %s", resolved_output)
    return resolved_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot initial and iteratively selected training points for one run.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = plot_corner_distribution(args.run_dir, args.output)
    print(f"[Plot] {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
